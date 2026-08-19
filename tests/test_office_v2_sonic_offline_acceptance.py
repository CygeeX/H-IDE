from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.office_v2_sonic_offline_acceptance import (
    ExclusiveJsonOutput,
    LOCK_PATH,
    MODEL_PATH,
    PROFILE_ID,
    REPO_ROOT,
)


RUN_ENV = "HIDE_RUN_OFFICE_V2_SONIC_OFFLINE"
TOOL = REPO_ROOT / "tools" / "office_v2_sonic_offline_acceptance.py"


def test_evidence_output_is_exclusive_and_rejects_symlinks(tmp_path: Path) -> None:
    output = tmp_path / "evidence.json"
    writer = ExclusiveJsonOutput(output)
    writer.write({"status": "fixture"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "fixture"}

    with pytest.raises(FileExistsError, match="overwrite"):
        ExclusiveJsonOutput(output)

    target = tmp_path / "target.json"
    target.write_text("do not replace", encoding="utf-8")
    final_link = tmp_path / "final-link.json"
    final_link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        ExclusiveJsonOutput(final_link)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        ExclusiveJsonOutput(linked_parent / "through-link.json")


def _require_real_runtime() -> None:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 to run the real 5-second OfficeV2/SONIC acceptance")
    missing_modules = [
        name
        for name in ("huggingface_hub", "mujoco", "numpy", "onnxruntime")
        if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        pytest.skip(f"offline acceptance dependencies are missing: {', '.join(missing_modules)}")
    if not MODEL_PATH.is_file():
        pytest.skip(f"OfficeV2 runtime model is missing: {MODEL_PATH}")
    if not LOCK_PATH.is_file():
        pytest.skip(f"SONIC model lock is missing: {LOCK_PATH}")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    profile = lock.get("profiles", {}).get(PROFILE_ID)
    if not isinstance(profile, dict):
        pytest.skip(f"SONIC profile is missing: {PROFILE_ID}")
    missing_models = [
        str(Path(entry["path"]).expanduser())
        for entry in profile.get("files", [])
        if entry.get("role") in {"planner", "encoder", "decoder"}
        and not Path(entry["path"]).expanduser().is_file()
    ]
    if missing_models:
        pytest.skip(f"locked offline SONIC models are missing: {', '.join(missing_models)}")


def test_real_office_v2_binary_authority_closed_loop(tmp_path: Path) -> None:
    _require_real_runtime()
    output = tmp_path / "office-v2-sonic-evidence.json"
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(TOOL), "--output", str(output)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n"
        f"evidence:\n{output.read_text(encoding='utf-8') if output.exists() else '<missing>'}"
    )
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    assert evidence["contract"]["hf_offline"] is True
    assert evidence["contract"]["dds_started"] is False
    assert evidence["contract"]["service_started"] is False
    assert evidence["contract"]["physics_hz"] == 250
    assert evidence["contract"]["sonic_hz"] == 50
    assert evidence["contract"]["duration_s"] == 5.0

    assert evidence["provenance"]["profile_id"] == PROFILE_ID
    assert len(evidence["provenance"]["outer_git_sha"]) == 40
    assert len(evidence["provenance"]["vendor_git_sha"]) == 40
    assert len(evidence["provenance"]["acceptance_runner_sha256"]) == 64
    assert len(evidence["provenance"]["outer_worktree"]["worktree_fingerprint_sha256"]) == 64
    assert len(evidence["provenance"]["vendor_worktree"]["worktree_fingerprint_sha256"]) == 64
    assert all(item["lock_match"] for item in evidence["provenance"]["model_files"])

    transaction = evidence["reference_transaction"]
    assert transaction["commit_count"] == 250
    assert transaction["failures"] == []
    assert evidence["motion"]["root_dx_m"] > 0.02
    assert evidence["motion"]["root_reverse_m"] < 0.03
    assert evidence["motion"]["whole_body_com_dx_m"] > 0.02
    assert evidence["motion"]["whole_body_com_reverse_m"] < 0.03
    assert evidence["motion"]["complete_contact_cycle"] is True
    assert evidence["dynamics"]["finite"] is True
    assert not any(evidence["dynamics"]["warning_counts"].values())
    assert evidence["dynamics"]["max_saturated_joint_count"] == 0
    assert evidence["actions"]["arm_violations"] == []
    assert all(gate["pass"] for gate in evidence["gates"].values())
