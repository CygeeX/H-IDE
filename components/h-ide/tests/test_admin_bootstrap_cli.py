from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from services.control_api import admin_cli
from services.control_api.admin_cli import AdminBootstrapError, bootstrap_first_admin
from services.control_api.auth import ScryptPasswordHasher
from services.control_api.store import SQLiteStore


PASSWORD = "offline bootstrap test password"


def _initialized_database(path: Path) -> Path:
    store = SQLiteStore(str(path))
    store.close()
    return path


def _database_counts(path: Path) -> tuple[int, int]:
    connection = sqlite3.connect(path)
    try:
        users = int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        audits = int(connection.execute("SELECT COUNT(*) FROM login_audits").fetchone()[0])
        return users, audits
    finally:
        connection.close()


def test_importing_offline_cli_does_not_construct_the_web_application(tmp_path) -> None:
    sentinel_database = tmp_path / "must-not-be-created.sqlite3"
    script = (
        "import os, sys; "
        f"os.environ['CONTROL_DB_PATH'] = {str(sentinel_database)!r}; "
        "import services.control_api.admin_cli; "
        "assert 'services.control_api.main' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not sentinel_database.exists()


def test_control_api_package_keeps_lazy_app_and_factory_exports_compatible(tmp_path) -> None:
    database = tmp_path / "explicit-import.sqlite3"
    script = (
        "import fastapi, services.control_api as package; "
        "assert 'services.control_api.main' not in __import__('sys').modules; "
        "from services.control_api import app, create_app; "
        "assert isinstance(app, fastapi.FastAPI); "
        "assert callable(create_app); "
        "assert package.app is app; "
        "assert package.create_app is create_app"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "CONTROL_DB_PATH": str(database),
            "SIMULATION_DB_PATH": str(database),
            "ROBOT_AGENT_GATEWAY_ENABLED": "false",
        }
    )
    for name in (
        "AUTH_BOOTSTRAP_ADMIN_USERNAME",
        "AUTH_BOOTSTRAP_ADMIN_PASSWORD",
        "AUTH_BOOTSTRAP_ADMIN_DISPLAY_NAME",
    ):
        environment.pop(name, None)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_first_admin_is_created_with_shared_scrypt_and_password_free_audit(tmp_path) -> None:
    database = _initialized_database(tmp_path / "control.sqlite3")

    result = bootstrap_first_admin(
        database_path=database,
        username="Root.Admin",
        display_name="Root Administrator",
        site_id="site-a",
        password=PASSWORD,
        credential_source="interactive_getpass",
    )

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        user = connection.execute("SELECT * FROM users").fetchone()
        audit = connection.execute("SELECT * FROM login_audits").fetchone()
    finally:
        connection.close()
    assert result.username == "root.admin"
    assert user["user_id"] == result.user_id
    assert user["role"] == "ADMIN"
    assert user["status"] == "active"
    assert user["site_id"] == "site-a"
    assert ScryptPasswordHasher().verify(PASSWORD, user["password_hash"])
    assert PASSWORD not in user["password_hash"]
    assert audit["outcome"] == "bootstrap_succeeded"
    detail = json.loads(audit["detail_json"])
    assert detail == {
        "action": "offline_first_admin_bootstrap",
        "credentialSource": "interactive_getpass",
        "passwordPersisted": False,
        "role": "ADMIN",
        "siteId": "site-a",
    }
    assert PASSWORD not in audit["detail_json"]


@pytest.mark.parametrize("existing_role", ["USER", "ADMIN"])
def test_existing_user_or_admin_fails_closed_and_rerun_is_idempotent(
    tmp_path,
    existing_role,
) -> None:
    database = _initialized_database(tmp_path / f"existing-{existing_role}.sqlite3")
    store = SQLiteStore(str(database))
    store.create_user(
        user_id="existing-user",
        username="existing",
        display_name="Existing",
        password_hash=ScryptPasswordHasher().hash(PASSWORD),
        role=existing_role,
        site_id="default",
    )
    store.close()

    with pytest.raises(AdminBootstrapError, match="already contains users"):
        bootstrap_first_admin(
            database_path=database,
            username="new-admin",
            password=PASSWORD,
        )

    assert _database_counts(database) == (1, 0)


def test_second_run_adds_neither_user_nor_audit(tmp_path) -> None:
    database = _initialized_database(tmp_path / "rerun.sqlite3")
    bootstrap_first_admin(
        database_path=database,
        username="first-admin",
        password=PASSWORD,
    )

    with pytest.raises(AdminBootstrapError, match="already contains users"):
        bootstrap_first_admin(
            database_path=database,
            username="first-admin",
            password=PASSWORD,
        )

    assert _database_counts(database) == (1, 1)


def test_concurrent_bootstrap_allows_exactly_one_commit(tmp_path) -> None:
    database = _initialized_database(tmp_path / "concurrent.sqlite3")
    barrier = threading.Barrier(2)

    def attempt(suffix: int):
        barrier.wait(timeout=5)
        try:
            return bootstrap_first_admin(
                database_path=database,
                username=f"admin-{suffix}",
                password=f"{PASSWORD}-{suffix}",
            )
        except AdminBootstrapError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, (1, 2)))

    assert sum(not isinstance(item, Exception) for item in outcomes) == 1
    assert sum(isinstance(item, AdminBootstrapError) for item in outcomes) == 1
    assert _database_counts(database) == (1, 1)


def test_secret_file_requires_exact_0600_and_cli_output_never_leaks_password(
    tmp_path,
    capsys,
) -> None:
    database = _initialized_database(tmp_path / "secret.sqlite3")
    secret = tmp_path / "admin.secret"
    secret.write_text(PASSWORD + "\n", encoding="utf-8")
    secret.chmod(0o600)

    assert admin_cli.main(
        [
            "--database",
            str(database),
            "--username",
            "file-admin",
            "--secret-file",
            str(secret),
        ]
    ) == 0
    captured = capsys.readouterr()
    assert PASSWORD not in captured.out
    assert PASSWORD not in captured.err
    assert str(secret) not in captured.out
    assert str(secret) not in captured.err

    denied_database = _initialized_database(tmp_path / "denied.sqlite3")
    secret.chmod(0o640)
    assert admin_cli.main(
        [
            "--database",
            str(denied_database),
            "--username",
            "denied-admin",
            "--secret-file",
            str(secret),
        ]
    ) == 1
    captured = capsys.readouterr()
    assert "exactly 0600" in captured.err
    assert PASSWORD not in captured.out + captured.err
    assert _database_counts(denied_database) == (0, 0)


def test_secret_file_symlink_is_refused(tmp_path) -> None:
    target = tmp_path / "secret-target"
    target.write_text(PASSWORD, encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "secret-link"
    link.symlink_to(target)

    with pytest.raises(AdminBootstrapError, match="cannot be opened safely"):
        admin_cli._read_secret_file(link)


def test_interactive_getpass_is_confirmed_without_secret_in_result_or_repr(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    database = _initialized_database(tmp_path / "interactive.sqlite3")
    replies = iter((PASSWORD, PASSWORD))
    monkeypatch.setattr(admin_cli.getpass, "getpass", lambda _prompt: next(replies))

    assert admin_cli.main(
        ["--database", str(database), "--username", "interactive-admin"]
    ) == 0
    captured = capsys.readouterr()
    assert PASSWORD not in captured.out + captured.err

    connection = sqlite3.connect(database)
    try:
        row = connection.execute("SELECT detail_json FROM login_audits").fetchone()
    finally:
        connection.close()
    assert PASSWORD not in row[0]


def test_audit_failure_rolls_back_user_and_reports_no_secret(tmp_path) -> None:
    database = _initialized_database(tmp_path / "rollback.sqlite3")
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TRIGGER reject_bootstrap_audit BEFORE INSERT ON login_audits "
        "BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END"
    )
    connection.commit()
    connection.close()

    with pytest.raises(AdminBootstrapError) as captured:
        bootstrap_first_admin(
            database_path=database,
            username="rollback-admin",
            password=PASSWORD,
        )

    assert PASSWORD not in str(captured.value)
    assert PASSWORD not in repr(captured.value)
    assert _database_counts(database) == (0, 0)


def test_missing_database_and_unsupported_credential_source_do_not_create_files(
    tmp_path,
) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(AdminBootstrapError, match="does not exist"):
        bootstrap_first_admin(
            database_path=missing,
            username="missing-admin",
            password=PASSWORD,
        )
    assert not missing.exists()

    database = _initialized_database(tmp_path / "source.sqlite3")
    with pytest.raises(AdminBootstrapError, match="credential source is unsupported"):
        bootstrap_first_admin(
            database_path=database,
            username="source-admin",
            password=PASSWORD,
            credential_source=PASSWORD,
        )
    assert _database_counts(database) == (0, 0)
