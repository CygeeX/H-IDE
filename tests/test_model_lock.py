from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from tools import verify_model_lock


ROOT = Path(__file__).resolve().parents[1]


def _write_lock(tmp_path: Path, *, runtime_executable: bool = True) -> tuple[Path, Path]:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"locked-sonic-artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lock = {
        "schema_version": "h-ide.sonic-model-lock/1",
        "model_repository": {"repo_id": "fixture", "revision": "0" * 40},
        "source_repositories": {
            "fixture-source": {
                "path": str(ROOT),
                "revision": __import__("subprocess").check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
            }
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
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    return lock_path, artifact


def test_complete_profile_verifies_and_detects_hash_drift(tmp_path: Path) -> None:
    lock_path, artifact = _write_lock(tmp_path)
    result = verify_model_lock.verify_profile(lock_path, "fixture", repo_root=ROOT)
    assert result.ok, result.errors
    assert result.verified_files == 5
    assert result.verified_repositories == 1

    artifact.write_bytes(b"changed")
    drifted = verify_model_lock.verify_profile(lock_path, "fixture", repo_root=ROOT)
    assert not drifted.ok
    assert any("mismatch" in error for error in drifted.errors)


def test_runtime_config_is_bound_to_the_exact_lock_profile(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path)
    revision = __import__("subprocess").check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    config_path = tmp_path / "runtime.yaml"
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

    result = verify_model_lock.verify_runtime_config(lock_path, config_path, repo_root=ROOT)
    assert result.ok, result.errors
    assert result.scope == "runtime-config"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["sonic_model_revision"] = "f" * 40
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    drifted = verify_model_lock.verify_runtime_config(lock_path, config_path, repo_root=ROOT)
    assert not drifted.ok
    assert any("sonic_model_revision mismatch" in error for error in drifted.errors)


def test_unavailable_profile_fails_closed(tmp_path: Path) -> None:
    lock_path, _ = _write_lock(tmp_path, runtime_executable=False)
    result = verify_model_lock.verify_profile(lock_path, "fixture", repo_root=ROOT)
    assert not result.ok
    assert "runtime is not executable" in result.errors[0]
    assert result.verified_files == 0

    bundle = verify_model_lock.verify_profile(
        lock_path,
        "fixture",
        repo_root=ROOT,
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
