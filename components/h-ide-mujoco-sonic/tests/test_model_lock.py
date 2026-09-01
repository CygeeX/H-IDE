from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from tools import verify_model_lock


ROOT = Path(__file__).resolve().parents[1]


def _fixture_repo_root(lock_path: Path) -> Path:
    return lock_path.parent / "audit-repository"


def _write_lock(tmp_path: Path, *, runtime_executable: bool = True) -> tuple[Path, Path]:
    audit_root = tmp_path / "audit-repository"
    audit_root.mkdir()
    subprocess.run(["git", "init", "-q", str(audit_root)], check=True)
    source = audit_root / "vendor" / "source"
    source.mkdir(parents=True)
    (source / "runtime.py").write_text("RUNTIME = 'locked'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "runtime.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=H-IDE Test",
            "-c",
            "user.email=h-ide-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    source_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    lock_path = tmp_path / "lock.json"
    audited_config = audit_root / "audited.yaml"
    audited_config.write_text(
        yaml.safe_dump(
            {
                "sonic_model_profile": "fixture",
                "sonic_model_lock": str(lock_path),
                "sonic_model_repo": "fixture",
                "sonic_model_revision": "0" * 40,
                "lerobot_commit": source_revision,
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(audit_root), "add", "audited.yaml", "vendor/source"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(audit_root),
            "-c",
            "user.name=H-IDE Test",
            "-c",
            "user.email=h-ide-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "audited fixture",
        ],
        check=True,
    )
    audit_revision = subprocess.check_output(
        ["git", "-C", str(audit_root), "rev-parse", "HEAD"], text=True
    ).strip()
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"locked-sonic-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lock = {
        "schema_version": "h-ide.sonic-model-lock/1",
        "repository_revision_at_audit": audit_revision,
        "model_repository": {"repo_id": "fixture", "revision": "0" * 40},
        "source_repositories": {
            "fixture-source": {
                "path": "vendor/source",
                "revision": source_revision,
            }
        },
        "audited_runtime": {
            "config": {
                "path": "audited.yaml",
                "size": audited_config.stat().st_size,
                "sha256": hashlib.sha256(audited_config.read_bytes()).hexdigest(),
            },
            "declared_model_revision": "0" * 40,
            "declared_lerobot_revision": source_revision,
            "observed_lerobot_revision": source_revision,
            "lock_safe": True,
            "reason": "fixture audit is reproducible",
        },
        "profiles": {
            "fixture": {
                "availability": "available" if runtime_executable else "unavailable",
                "availability_reason": "fixture is intentionally unavailable",
                "bundle_integrity": "complete",
                "runtime_executable": runtime_executable,
                "runtime_blockers": ["fixture controller is not built"],
                "model_revision": "0" * 40,
                "repositories": ["fixture-source"],
                "controller": {
                    "source_repository": "fixture-source",
                    "files": ["controller"],
                },
                "observation_config": {"files": ["observation"]},
                "files": [
                    {
                        "id": role if role not in {"controller", "observation_config"} else (
                            "controller" if role == "controller" else "observation"
                        ),
                        "role": role,
                        "path": str(artifact),
                        "present": True,
                        "size": artifact.stat().st_size,
                        "sha256": digest,
                    }
                    for role in ("planner", "encoder", "decoder", "controller", "observation_config")
                ],
            }
        },
    }
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path, artifact


def _write_runtime_config(lock_path: Path) -> Path:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    revision = lock["source_repositories"]["fixture-source"]["revision"]
    config_path = lock_path.parent / "runtime.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sonic_model_profile": "fixture",
                "sonic_model_lock": str(lock_path),
                "sonic_model_repo": "fixture",
                "sonic_model_revision": "0" * 40,
                "lerobot_commit": revision,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _verify_fixture_runtime(
    lock_path: Path, config_path: Path
) -> verify_model_lock.VerificationResult:
    return verify_model_lock.verify_runtime_config(
        lock_path, config_path, repo_root=_fixture_repo_root(lock_path)
    )


def _commit_fixture(audit_root: Path, *paths: str, message: str) -> str:
    subprocess.run(["git", "-C", str(audit_root), "add", *paths], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(audit_root),
            "-c",
            "user.name=H-IDE Test",
            "-c",
            "user.email=h-ide-test@example.invalid",
            "commit",
            "-q",
            "-m",
            message,
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(audit_root), "rev-parse", "HEAD"], text=True
    ).strip()


def test_complete_profile_verifies_and_detects_hash_drift(tmp_path: Path) -> None:
    lock_path, artifact = _write_lock(tmp_path)
    result = verify_model_lock.verify_profile(
        lock_path, "fixture", repo_root=_fixture_repo_root(lock_path)
    )
    assert result.ok, result.errors
    assert result.verified_files == 5
    assert result.verified_repositories == 1

    artifact.write_bytes(b"changed")
    drifted = verify_model_lock.verify_profile(
        lock_path, "fixture", repo_root=_fixture_repo_root(lock_path)
    )
    assert not drifted.ok
    assert any("mismatch" in error for error in drifted.errors)


def test_runtime_config_is_bound_to_the_exact_lock_profile(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    config_path = _write_runtime_config(lock_path)

    result = _verify_fixture_runtime(lock_path, config_path)
    assert result.ok, result.errors
    assert result.scope == "runtime-config"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["sonic_model_revision"] = "f" * 40
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    drifted = _verify_fixture_runtime(lock_path, config_path)
    assert not drifted.ok
    assert any("sonic_model_revision mismatch" in error for error in drifted.errors)


def test_runtime_audit_must_be_closed_and_reproducible(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    config_path = _write_runtime_config(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    lock["audited_runtime"]["lock_safe"] = False
    lock["audited_runtime"]["reason"] = "pending repository anchor"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    pending = _verify_fixture_runtime(lock_path, config_path)
    assert not pending.ok
    assert any("not lock-safe" in error for error in pending.errors)
    pending_profile = verify_model_lock.verify_profile(
        lock_path, "fixture", repo_root=_fixture_repo_root(lock_path)
    )
    assert not pending_profile.ok
    assert any("not lock-safe" in error for error in pending_profile.errors)

    lock["audited_runtime"]["lock_safe"] = True
    lock["repository_revision_at_audit"] = "f" * 40
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    unknown = _verify_fixture_runtime(lock_path, config_path)
    assert not unknown.ok
    assert any("repository_revision_at_audit" in error for error in unknown.errors)


def test_runtime_audit_rejects_selector_and_gitlink_drift(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    config_path = _write_runtime_config(lock_path)
    audit_root = _fixture_repo_root(lock_path)
    audited_config = audit_root / "audited.yaml"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    selectors = yaml.safe_load(audited_config.read_text(encoding="utf-8"))
    selectors["sonic_model_profile"] = "wrong-profile"
    audited_config.write_text(yaml.safe_dump(selectors), encoding="utf-8")
    lock["repository_revision_at_audit"] = _commit_fixture(
        audit_root, "audited.yaml", message="selector drift"
    )
    lock["audited_runtime"]["config"]["size"] = audited_config.stat().st_size
    lock["audited_runtime"]["config"]["sha256"] = hashlib.sha256(
        audited_config.read_bytes()
    ).hexdigest()
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    selector_drift = _verify_fixture_runtime(lock_path, config_path)
    assert not selector_drift.ok
    assert any("selector sonic_model_profile" in error for error in selector_drift.errors)

    source = audit_root / "vendor" / "source"
    (source / "runtime.py").write_text("RUNTIME = 'new revision'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "runtime.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=H-IDE Test",
            "-c",
            "user.email=h-ide-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "source drift",
        ],
        check=True,
    )
    new_source_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    lock["source_repositories"]["fixture-source"]["revision"] = new_source_revision
    lock["audited_runtime"]["declared_lerobot_revision"] = new_source_revision
    lock["audited_runtime"]["observed_lerobot_revision"] = new_source_revision
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["lerobot_commit"] = new_source_revision
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    gitlink_drift = _verify_fixture_runtime(lock_path, config_path)
    assert not gitlink_drift.ok
    assert any("source gitlink" in error for error in gitlink_drift.errors)


def test_runtime_audit_rejects_current_and_anchored_blob_drift(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    config_path = _write_runtime_config(lock_path)
    audit_root = _fixture_repo_root(lock_path)
    audited_config = audit_root / "audited.yaml"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    audited_config.write_bytes(audited_config.read_bytes() + b"# drift\n")
    lock["repository_revision_at_audit"] = _commit_fixture(
        audit_root, "audited.yaml", message="blob drift"
    )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _verify_fixture_runtime(lock_path, config_path)
    assert not result.ok
    assert any("current audited runtime config" in error for error in result.errors)
    assert any("Git blob" in error for error in result.errors)


def test_runtime_audit_rejects_anchored_model_lock_path_drift(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    config_path = _write_runtime_config(lock_path)
    audit_root = _fixture_repo_root(lock_path)
    audited_config = audit_root / "audited.yaml"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    selectors = yaml.safe_load(audited_config.read_text(encoding="utf-8"))
    selectors["sonic_model_lock"] = str(tmp_path / "different-lock.json")
    audited_config.write_text(yaml.safe_dump(selectors), encoding="utf-8")
    lock["repository_revision_at_audit"] = _commit_fixture(
        audit_root, "audited.yaml", message="model lock drift"
    )
    lock["audited_runtime"]["config"]["size"] = audited_config.stat().st_size
    lock["audited_runtime"]["config"]["sha256"] = hashlib.sha256(
        audited_config.read_bytes()
    ).hexdigest()
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _verify_fixture_runtime(lock_path, config_path)
    assert not result.ok
    assert any("selects a different model lock" in error for error in result.errors)


def test_runtime_audit_revision_must_be_an_ancestor(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    config_path = _write_runtime_config(lock_path)
    audit_root = _fixture_repo_root(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    tree = subprocess.check_output(
        ["git", "-C", str(audit_root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    side_commit = subprocess.run(
        [
            "git",
            "-C",
            str(audit_root),
            "-c",
            "user.name=H-IDE Test",
            "-c",
            "user.email=h-ide-test@example.invalid",
            "commit-tree",
            tree,
        ],
        input="unrelated audit commit\n",
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    lock["repository_revision_at_audit"] = side_commit
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = _verify_fixture_runtime(lock_path, config_path)
    assert not result.ok
    assert any("not an ancestor" in error for error in result.errors)


def test_source_worktree_must_be_clean(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source = Path(lock["source_repositories"]["fixture-source"]["path"])
    if not source.is_absolute():
        source = _fixture_repo_root(lock_path) / source
    (source / "runtime.py").write_text("RUNTIME = 'dirty'\n", encoding="utf-8")

    result = verify_model_lock.verify_profile(
        lock_path, "fixture", repo_root=_fixture_repo_root(lock_path)
    )

    assert not result.ok
    assert result.verified_repositories == 0
    assert any("source worktree is dirty" in error for error in result.errors)


def test_unavailable_profile_fails_closed(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path, runtime_executable=False)
    result = verify_model_lock.verify_profile(
        lock_path, "fixture", repo_root=_fixture_repo_root(lock_path)
    )
    assert not result.ok
    assert "runtime is not executable" in result.errors[0]
    assert result.verified_files == 0

    bundle = verify_model_lock.verify_profile(
        lock_path,
        "fixture",
        repo_root=_fixture_repo_root(lock_path),
        bundle_only=True,
    )
    assert bundle.ok, bundle.errors
    assert bundle.scope == "bundle"
    assert bundle.runtime_executable is False
    assert bundle.verified_files == 5


def test_repository_lock_separates_complete_bundles_from_executable_runtime() -> None:
    lock = json.loads((ROOT / "config" / "sonic-models.lock.json").read_text(encoding="utf-8"))
    profiles = lock["profiles"]
    assert profiles["sonic-python-experimental"]["availability"] == "available"
    assert profiles["sonic-official-default"]["availability"] == "unavailable"
    assert profiles["sonic-official-v1_1"]["availability"] == "unavailable"

    assert profiles["sonic-python-experimental"]["runtime_executable"] is True

    for profile_id in ("sonic-official-default", "sonic-official-v1_1"):
        profile = profiles[profile_id]
        assert profile["bundle_integrity"] == "complete"
        assert profile["runtime_executable"] is False
        assert profile["controller"]["built_binary_present"] is False
        assert all(entry["present"] for entry in profile["files"])
        assert all(entry["size"] and entry["sha256"] for entry in profile["files"])


def test_vendored_tree_hash_ignores_generated_python_metadata(tmp_path: Path) -> None:
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    (vendor / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    baseline = verify_model_lock._directory_sha256(vendor)

    cache = vendor / "__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-312.pyc").write_bytes(b"generated")
    egg_info = vendor / "runtime.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated\n", encoding="utf-8")
    assert verify_model_lock._directory_sha256(vendor) == baseline

    (vendor / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert verify_model_lock._directory_sha256(vendor) != baseline


def test_repository_runtime_source_lock_matches_every_runtime_selector() -> None:
    lock = json.loads(
        (ROOT / "config" / "sonic-models.lock.json").read_text(encoding="utf-8")
    )
    profile_id = "sonic-python-experimental"
    profile = lock["profiles"][profile_id]
    source_revision = lock["source_repositories"]["lerobot-python"]["revision"]
    vendor_root = ROOT / "vendor" / "lerobot"

    source_entry = lock["source_repositories"]["lerobot-python"]
    if source_entry.get("kind", "git") == "vendored":
        # The integration release stores this source as ordinary monorepo
        # files. A deterministic tree hash replaces the nested Git gitlink
        # check, while the manifest still binds the upstream revision.
        manifest = json.loads(
            (ROOT / ".." / ".." / "manifests" / "vendor" / "lerobot.json")
            .read_text(encoding="utf-8")
        )
        assert manifest["upstream_revision"] == source_revision
        assert manifest["content_sha256"] == source_entry["content_sha256"]
        assert verify_model_lock._directory_sha256(vendor_root) == source_entry["content_sha256"]
    else:
        assert subprocess.check_output(
            ["git", "-C", str(vendor_root), "rev-parse", "HEAD"],
            text=True,
        ).strip() == source_revision
        assert subprocess.check_output(
            [
                "git",
                "-C",
                str(vendor_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            text=True,
        ) == ""
        gitlink = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-tree", "HEAD", "vendor/lerobot"],
            text=True,
        ).split()
        assert gitlink[:3] == ["160000", "commit", source_revision]

    manifest = yaml.safe_load(
        (ROOT / "profiles" / "sonic-python-experimental.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["source"]["revision"] == source_revision
    for relative in (
        "config/sim.yaml",
        "config/headless.yaml",
        "config/ros-headless.yaml",
    ):
        runtime = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        assert runtime["sonic_model_profile"] == profile_id
        assert runtime["lerobot_commit"] == source_revision

    locked_files = {entry["id"]: entry for entry in profile["files"]}
    for file_id in ("python-controller", "python-pipeline", "python-profile"):
        entry = locked_files[file_id]
        path = (ROOT / entry["path"]).resolve()
        assert path.stat().st_size == entry["size"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]

    audited = lock["audited_runtime"]
    audited_config = ROOT / audited["config"]["path"]
    if audited["lock_safe"] is True:
        audit_revision = lock["repository_revision_at_audit"]
        if lock.get("audit_mode") == "monorepo_tree" and audit_revision is None:
            # The top-level Skillstore commit is the release anchor for a
            # component tree; current size/hash checks remain mandatory.
            assert audited_config.read_bytes()
        else:
            assert isinstance(audit_revision, str) and len(audit_revision) == 40
            anchored = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "show",
                    f"{audit_revision}:{audited['config']['path']}",
                ]
            )
            assert anchored == audited_config.read_bytes()
        for runtime_path in (
            ROOT / "config" / "sim.yaml",
            ROOT / "config" / "headless.yaml",
            ROOT / "config" / "ros-headless.yaml",
        ):
            verified = verify_model_lock.verify_runtime_config(
                ROOT / "config" / "sonic-models.lock.json",
                runtime_path,
                repo_root=ROOT,
            )
            assert verified.ok, verified.errors
        profile_cli_expected_rc = 0
    else:
        assert lock["repository_revision_at_audit"] is None
        assert "pending" in audited["reason"]
        pending = verify_model_lock.verify_runtime_config(
            ROOT / "config" / "sonic-models.lock.json",
            ROOT / "config" / "ros-headless.yaml",
            repo_root=ROOT,
        )
        assert not pending.ok
        assert any("not lock-safe" in error for error in pending.errors)
        profile_cli_expected_rc = 1
    python_bin = ROOT / ".venv" / "bin" / "python"
    if not python_bin.is_file():
        # A clean clone can run the pure verifier with the active test
        # interpreter before the optional runtime venv is provisioned.
        python_bin = Path(sys.executable)
    profile_cli = subprocess.run(
        [
            str(python_bin),
            str(ROOT / "tools" / "verify_model_lock.py"),
            "--lock",
            str(ROOT / "config" / "sonic-models.lock.json"),
            "--profile",
            profile_id,
            "--json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert profile_cli.returncode == profile_cli_expected_rc, profile_cli.stdout
    assert audited["declared_model_revision"] == profile["model_revision"]
    assert audited["declared_lerobot_revision"] == source_revision
    assert audited["observed_lerobot_revision"] == source_revision
    assert audited_config.stat().st_size == audited["config"]["size"]
    assert hashlib.sha256(audited_config.read_bytes()).hexdigest() == (
        audited["config"]["sha256"]
    )
