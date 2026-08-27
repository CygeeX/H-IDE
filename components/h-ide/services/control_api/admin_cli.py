"""Fail-closed, offline bootstrap for the first H-IDE administrator.

This module deliberately does not construct the FastAPI application or a
``SQLiteStore``.  The target database must already exist and contain the auth
schema.  A single ``BEGIN IMMEDIATE`` transaction checks that the users table
is empty, inserts the administrator, and writes its password-free audit row.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import sqlite3
import stat
import sys
from typing import Sequence
from urllib.parse import quote
import uuid
import warnings

from .auth import (
    Role,
    ScryptPasswordHasher,
    _clean_display_name,
    _clean_site_id,
    _clean_username,
)


_MAX_SECRET_FILE_BYTES = 1_026
_CREDENTIAL_SOURCES = frozenset({"interactive_getpass", "0600_secret_file"})
_REQUIRED_USERS_COLUMNS = frozenset(
    {
        "user_id",
        "username",
        "display_name",
        "password_hash",
        "role",
        "site_id",
        "status",
        "created_at",
        "updated_at",
    }
)
_REQUIRED_AUDIT_COLUMNS = frozenset(
    {
        "user_id",
        "username",
        "outcome",
        "remote_address",
        "user_agent_hash",
        "detail_json",
        "occurred_at",
    }
)


class AdminBootstrapError(RuntimeError):
    """A safe, user-displayable bootstrap refusal without secret material."""


@dataclass(frozen=True, slots=True)
class AdminBootstrapResult:
    """Non-secret result returned after the transaction is committed."""

    user_id: str
    username: str
    display_name: str
    site_id: str
    created_at: str


def _existing_database(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value).expanduser()
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise AdminBootstrapError("the database file does not exist") from exc
    except OSError as exc:
        raise AdminBootstrapError("the database file cannot be inspected") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise AdminBootstrapError("the database path is not a regular file")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise AdminBootstrapError("the database path cannot be resolved") from exc


def _read_secret_file(path_value: str | os.PathLike[str]) -> str:
    """Read one UTF-8 password without following a symlink or relaxing mode."""

    path = Path(path_value).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AdminBootstrapError("the secret file cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise AdminBootstrapError("the secret path is not a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise AdminBootstrapError("the secret file permissions must be exactly 0600")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise AdminBootstrapError("the secret file must be owned by the current user")
        payload = os.read(descriptor, _MAX_SECRET_FILE_BYTES)
        if len(payload) == _MAX_SECRET_FILE_BYTES:
            raise AdminBootstrapError("the secret file is too large")
    except OSError as exc:
        raise AdminBootstrapError("the secret file cannot be read safely") from exc
    finally:
        os.close(descriptor)

    try:
        password = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AdminBootstrapError("the secret file must contain UTF-8 text") from exc
    if password.endswith("\n"):
        password = password[:-1]
    if "\n" in password or "\r" in password:
        raise AdminBootstrapError("the secret file must contain exactly one line")
    return password


def _read_password_interactively() -> str:
    """Read and confirm a password while refusing getpass' echoed fallback."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            first = getpass.getpass("Administrator password: ")
            confirmation = getpass.getpass("Confirm administrator password: ")
    except (EOFError, KeyboardInterrupt, getpass.GetPassWarning) as exc:
        raise AdminBootstrapError("a private interactive terminal is required") from exc
    if first != confirmation:
        raise AdminBootstrapError("the password confirmation does not match")
    return first


def _require_schema(connection: sqlite3.Connection) -> None:
    users_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(users)").fetchall()
    }
    audit_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(login_audits)").fetchall()
    }
    if not _REQUIRED_USERS_COLUMNS <= users_columns:
        raise AdminBootstrapError("the database does not contain the H-IDE users schema")
    if not _REQUIRED_AUDIT_COLUMNS <= audit_columns:
        raise AdminBootstrapError("the database does not contain the H-IDE audit schema")


def _open_existing_database(path: Path) -> sqlite3.Connection:
    # ``mode=rw`` prevents a path race from silently creating a different DB.
    uri = f"file:{quote(str(path), safe='/')}?mode=rw"
    try:
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=30.0,
            isolation_level=None,
        )
    except sqlite3.Error as exc:
        raise AdminBootstrapError("the database cannot be opened for bootstrap") from exc
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
    except sqlite3.Error as exc:
        connection.close()
        raise AdminBootstrapError("the database cannot be prepared for bootstrap") from exc
    return connection


def bootstrap_first_admin(
    *,
    database_path: str | os.PathLike[str],
    username: str,
    password: str,
    display_name: str = "Administrator",
    site_id: str = "default",
    credential_source: str = "interactive_getpass",
) -> AdminBootstrapResult:
    """Atomically create the sole first user as ADMIN and audit the operation.

    The password is hashed before the write transaction.  It is never included
    in the result, SQL audit data, exception text, or command output.
    """

    try:
        clean_username = _clean_username(username)
        clean_display_name = _clean_display_name(
            display_name,
            fallback=clean_username,
        )
        clean_site_id = _clean_site_id(site_id)
        if credential_source not in _CREDENTIAL_SOURCES:
            raise ValueError("credential source is unsupported")
        password_hash = ScryptPasswordHasher().hash(password)
    except (TypeError, ValueError) as exc:
        # The shared validators only produce policy descriptions, not input
        # values.  Keep that property when translating to the CLI exception.
        raise AdminBootstrapError(str(exc)) from exc

    path = _existing_database(database_path)
    connection = _open_existing_database(path)
    transaction_started = False
    try:
        _require_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        transaction_started = True
        user_count = int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        if user_count != 0:
            raise AdminBootstrapError(
                "bootstrap refused because the database already contains users"
            )

        created_at = datetime.now(timezone.utc).isoformat()
        user_id = f"user-{uuid.uuid4().hex}"
        connection.execute(
            "INSERT INTO users(user_id, username, display_name, password_hash, role, "
            "site_id, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
            (
                user_id,
                clean_username,
                clean_display_name,
                password_hash,
                Role.ADMIN.value,
                clean_site_id,
                created_at,
                created_at,
            ),
        )
        audit_detail = json.dumps(
            {
                "action": "offline_first_admin_bootstrap",
                "credentialSource": credential_source,
                "passwordPersisted": False,
                "role": Role.ADMIN.value,
                "siteId": clean_site_id,
            },
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        connection.execute(
            "INSERT INTO login_audits(user_id, username, outcome, remote_address, "
            "user_agent_hash, detail_json, occurred_at) "
            "VALUES (?, ?, 'bootstrap_succeeded', NULL, NULL, ?, ?)",
            (user_id, clean_username, audit_detail, created_at),
        )
        connection.commit()
        transaction_started = False
        return AdminBootstrapResult(
            user_id=user_id,
            username=clean_username,
            display_name=clean_display_name,
            site_id=clean_site_id,
            created_at=created_at,
        )
    except AdminBootstrapError:
        if transaction_started:
            connection.rollback()
        raise
    except sqlite3.Error as exc:
        if transaction_started:
            connection.rollback()
        raise AdminBootstrapError(
            "the atomic administrator bootstrap transaction failed"
        ) from exc
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="h-ide-admin-bootstrap",
        description=(
            "Offline, one-time creation of the first H-IDE ADMIN. "
            "The existing database must have an empty users table."
        ),
    )
    parser.add_argument(
        "--database",
        required=True,
        help="explicit path to an existing H-IDE SQLite database",
    )
    parser.add_argument("--username", required=True, help="administrator login name")
    parser.add_argument(
        "--display-name",
        default="Administrator",
        help="administrator display name (default: Administrator)",
    )
    parser.add_argument(
        "--site-id",
        default="default",
        help="administrator site identifier (default: default)",
    )
    parser.add_argument(
        "--secret-file",
        help=(
            "read the password from an owned regular UTF-8 file with exact mode 0600; "
            "otherwise prompt twice with getpass"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    password: str | None = None
    try:
        if args.secret_file:
            password = _read_secret_file(args.secret_file)
            credential_source = "0600_secret_file"
        else:
            password = _read_password_interactively()
            credential_source = "interactive_getpass"
        result = bootstrap_first_admin(
            database_path=args.database,
            username=args.username,
            display_name=args.display_name,
            site_id=args.site_id,
            password=password,
            credential_source=credential_source,
        )
    except AdminBootstrapError as exc:
        print(f"Administrator bootstrap refused: {exc}", file=sys.stderr)
        return 1
    finally:
        password = None

    print(
        "Created first H-IDE administrator "
        f"'{result.username}' for site '{result.site_id}'."
    )
    return 0


def cli() -> int:
    return main()


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` tests.
    raise SystemExit(main())


__all__ = [
    "AdminBootstrapError",
    "AdminBootstrapResult",
    "bootstrap_first_admin",
    "cli",
    "main",
]
