#!/usr/bin/env python3
"""Run a service-free SONIC + OfficeV2 MuJoCo acceptance experiment.

This is an evaluation-only, high-fidelity reconstruction of the Python SONIC
profile.  It deliberately starts no DDS participant, HTTP service, ROS node, or
real-time motion process.  MuJoCo advances at 250 Hz and the controller advances
at 50 Hz from the same virtual clock.  Hugging Face is forced offline before the
runtime is imported.

Evidence is written once to the caller's explicit ``--output`` path.  Existing
files, symbolic-link outputs, and symbolic-link ancestors are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


SCHEMA_VERSION = "h-ide.office-v2-sonic-offline-acceptance/1"
PROFILE_ID = "sonic-python-experimental"
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor" / "lerobot"
LOCK_PATH = REPO_ROOT / "config" / "sonic-models.lock.json"
PROFILE_PATH = REPO_ROOT / "profiles" / "sonic-python-experimental.yaml"
MODEL_PATH = REPO_ROOT / "scenes" / "office_v2" / "generated" / "mujoco" / "office_v2.runtime.xml"

PHYSICS_DT_S = 0.004
CONTROL_EVERY = 5
CONTROL_DT_S = PHYSICS_DT_S * CONTROL_EVERY
DURATION_S = 5.0
PHYSICS_STEPS = int(DURATION_S / PHYSICS_DT_S)
CONTACT_FORCE_THRESHOLD_N = 5.0
CONTACT_DEBOUNCE_STEPS = 5
CONTACT_EVALUATION_START_S = 0.5
SPAWN_XY = (-6.45, 0.0)

# Existing navigation arm verifier: shoulder/elbow 6 rad/s, wrists 4 rad/s.
ARM_STEP_LIMITS_RAD = (0.12, 0.12, 0.12, 0.12, 0.08, 0.08, 0.08) * 2


class ExclusiveJsonOutput:
    """Reserve and write one JSON object without following symbolic links."""

    def __init__(self, raw_path: str | Path) -> None:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.absolute()
        if candidate.name in {"", ".", ".."}:
            raise ValueError("--output must name a JSON file")

        cursor = candidate
        while True:
            if cursor.is_symlink():
                raise ValueError(f"refusing output path with symbolic link: {cursor}")
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        if candidate.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {candidate}")
        if not candidate.parent.is_dir():
            raise ValueError(f"output parent must already be a directory: {candidate.parent}")

        resolved_parent = candidate.parent.resolve(strict=True)
        self.path = resolved_parent / candidate.name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self._fd = os.open(self.path, flags, 0o600)
        self._written = False

    def write(self, evidence: dict[str, Any]) -> None:
        if self._written:
            raise RuntimeError("acceptance evidence has already been written")
        serialized = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        file_descriptor = self._fd
        self._fd = -1
        with os.fdopen(file_descriptor, "wb", closefd=True) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        self._written = True

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_worktree_evidence(path: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
    ).stdout
    unstaged = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    staged = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "--cached", "HEAD"],
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    for label, value in (
        (b"status\0", status),
        (b"unstaged\0", unstaged),
        (b"staged\0", staged),
    ):
        digest.update(label)
        digest.update(value)
    return {
        "dirty": bool(status),
        "porcelain": status.decode("utf-8", errors="replace").splitlines(),
        "tracked_diff_bytes": len(unstaged) + len(staged),
        "worktree_fingerprint_sha256": digest.hexdigest(),
        "fingerprint_scope": "porcelain plus staged/unstaged tracked binary diffs",
    }


def _load_model_provenance() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    profile = lock["profiles"][PROFILE_ID]
    model_revision = profile["model_revision"]
    model_entries = []
    for entry in profile["files"]:
        if entry.get("role") not in {"planner", "encoder", "decoder"}:
            continue
        path = Path(entry["path"]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"locked SONIC model is missing: {path}")
        size = path.stat().st_size
        digest = _sha256(path)
        if size != entry["size"] or digest != entry["sha256"]:
            raise RuntimeError(f"locked SONIC model hash/size mismatch: {path}")
        model_entries.append(
            {
                "id": entry["id"],
                "role": entry["role"],
                "path": str(path),
                "size": size,
                "sha256": digest,
                "lock_match": True,
            }
        )

    cache_root = Path(lock["model_repository"]["cache_root"]).expanduser()
    main_ref = cache_root / "refs" / "main"
    if not main_ref.is_file() or main_ref.read_text(encoding="utf-8").strip() != model_revision:
        raise RuntimeError("offline Hugging Face main ref does not select the locked SONIC revision")

    vendor_sha = _git_sha(VENDOR_ROOT)
    source_lock = lock["source_repositories"]["lerobot-python"]["revision"]
    return {
        "profile_id": PROFILE_ID,
        "profile_activation": profile["activation"],
        "profile_motion_accepted": profile["motion_accepted"],
        "profile_manifest": str(PROFILE_PATH),
        "profile_manifest_sha256": _sha256(PROFILE_PATH),
        "model_repository": lock["model_repository"]["repo_id"],
        "model_revision": model_revision,
        "model_variant": profile["model_variant"],
        "model_files": model_entries,
        "offline_main_ref": str(main_ref),
        "outer_git_sha": _git_sha(REPO_ROOT),
        "vendor_git_sha": vendor_sha,
        "outer_worktree": _git_worktree_evidence(REPO_ROOT),
        "vendor_worktree": _git_worktree_evidence(VENDOR_ROOT),
        "acceptance_runner_sha256": _sha256(Path(__file__).resolve()),
        "profile_locked_vendor_sha": source_lock,
        "profile_source_lock_matches_runtime": vendor_sha == source_lock,
        "runtime_source_note": (
            "The model profile remains evaluation_only; the exact live vendor SHA and source hashes "
            "are recorded independently because parity fixes postdate its source lock."
        ),
        "controller_sha256": _sha256(
            VENDOR_ROOT
            / "src"
            / "lerobot"
            / "robots"
            / "unitree_g1"
            / "controllers"
            / "sonic_whole_body.py"
        ),
        "pipeline_sha256": _sha256(
            VENDOR_ROOT
            / "src"
            / "lerobot"
            / "robots"
            / "unitree_g1"
            / "controllers"
            / "sonic_pipeline.py"
        ),
        "office_v2_model": str(MODEL_PATH),
        "office_v2_model_sha256": _sha256(MODEL_PATH),
    }


def _configure_offline_imports() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for source_root in (REPO_ROOT / "src", VENDOR_ROOT / "src"):
        value = str(source_root)
        if value not in sys.path:
            sys.path.insert(0, value)


def _observation(data: Any, joint_enum: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, motor in enumerate(joint_enum):
        result[f"{motor.name}.q"] = float(data.qpos[7 + index])
        result[f"{motor.name}.dq"] = float(data.qvel[6 + index])
    for axis, value in zip("wxyz", data.qpos[3:7], strict=True):
        result[f"imu.quat.{axis}"] = float(value)
    for axis, value in zip("xyz", data.qvel[3:6], strict=True):
        result[f"imu.gyro.{axis}"] = float(value)
    return result


def _target_vector(action: dict[str, float], joint_enum: Any, np: Any) -> Any:
    return np.array([action[f"{motor.name}.q"] for motor in joint_enum], dtype=np.float64)


def _descendant_bodies(model: Any, root_body: int) -> set[int]:
    result = {root_body}
    for body_id in range(model.nbody):
        current = body_id
        while current:
            if current == root_body:
                result.add(body_id)
                break
            current = int(model.body_parentid[current])
    return result


def _foot_loads(
    mujoco: Any,
    np: Any,
    model: Any,
    data: Any,
    robot_bodies: set[int],
    left_bodies: set[int],
    right_bodies: set[int],
) -> tuple[list[float], list[int]]:
    loads = [0.0, 0.0]
    counts = [0, 0]
    force = np.zeros(6)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body1 = int(model.geom_bodyid[int(contact.geom1)])
        body2 = int(model.geom_bodyid[int(contact.geom2)])
        side = None
        if (body1 in left_bodies and body2 not in robot_bodies) or (
            body2 in left_bodies and body1 not in robot_bodies
        ):
            side = 0
        elif (body1 in right_bodies and body2 not in robot_bodies) or (
            body2 in right_bodies and body1 not in robot_bodies
        ):
            side = 1
        if side is None:
            continue
        mujoco.mj_contactForce(model, data, contact_id, force)
        loads[side] += abs(float(force[0]))
        counts[side] += 1
    return loads, counts


def _body_pose(mujoco: Any, np: Any, model: Any, data: Any, body_id: int) -> Any:
    pose = np.concatenate((data.xpos[body_id], data.xquat[body_id], np.zeros(6)))
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        pose[7:13],
        0,
    )
    angular = pose[7:10].copy()
    pose[7:10] = pose[10:13]
    pose[10:13] = angular
    return pose


def _tilt_rad(np: Any, quaternion: Any) -> float:
    values = np.asarray(quaternion, dtype=np.float64)
    values /= np.linalg.norm(values)
    up_z = 1.0 - 2.0 * (values[1] ** 2 + values[2] ** 2)
    return math.acos(float(np.clip(up_z, -1.0, 1.0)))


def _contact_events(np: Any, contacts: list[list[float | int]]) -> dict[str, list[dict[str, Any]]]:
    loaded = np.asarray(contacts, dtype=np.float64)[:, :2] >= CONTACT_FORCE_THRESHOLD_N
    start = int(round(CONTACT_EVALUATION_START_S / PHYSICS_DT_S))
    events: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for side, name in enumerate(("left", "right")):
        state = bool(loaded[start, side])
        candidate = state
        run = 1
        for tick in range(start + 1, len(loaded)):
            raw = bool(loaded[tick, side])
            if raw == candidate:
                run += 1
            else:
                candidate = raw
                run = 1
            if candidate != state and run >= CONTACT_DEBOUNCE_STEPS:
                state = candidate
                events[name].append(
                    {
                        "kind": "touchdown" if state else "liftoff",
                        "physics_tick": tick,
                        "time_s": tick * PHYSICS_DT_S,
                    }
                )
    return events


def _has_complete_cycle(events: dict[str, list[dict[str, Any]]]) -> bool:
    for side in ("left", "right"):
        lifted = False
        completed = False
        for event in events[side]:
            if event["kind"] == "liftoff":
                lifted = True
            elif event["kind"] == "touchdown" and lifted:
                completed = True
                break
        if not completed:
            return False
    return True


def _gate(actual: Any, passed: bool, requirement: str) -> dict[str, Any]:
    return {"actual": actual, "pass": bool(passed), "requirement": requirement}


def run_acceptance(*, force_cpu: bool, reference_timeout_s: float) -> dict[str, Any]:
    _configure_offline_imports()
    provenance_started = time.perf_counter()
    provenance = _load_model_provenance()
    provenance_s = time.perf_counter() - provenance_started

    import mujoco
    import numpy as np
    import onnxruntime as ort

    from hide_mujoco_sonic.plant import stabilized_support_wrench
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
        DEFAULT_ANGLES_MUJOCO,
        LM,
        compute_kp_kd,
        normalized_action_to_hardware_target,
    )
    from lerobot.robots.unitree_g1.controllers.sonic_whole_body import SonicRuntime
    from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

    experiment_started = time.perf_counter()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise RuntimeError(f"unexpected OfficeV2 model dimensions: {(model.nq, model.nv, model.nu)}")
    if not math.isclose(float(model.opt.timestep), PHYSICS_DT_S, abs_tol=1e-12):
        raise RuntimeError(f"unexpected OfficeV2 timestep: {model.opt.timestep}")

    data.qpos[:3] = (SPAWN_XY[0], SPAWN_XY[1], model.qpos0[2])
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:] = DEFAULT_ANGLES_MUJOCO
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    robot_bodies = _descendant_bodies(model, pelvis_id)
    left_bodies = _descendant_bodies(model, left_id)
    right_bodies = _descendant_bodies(model, right_id)
    support_anchor = np.array((SPAWN_XY[0], SPAWN_XY[1], data.xpos[torso_id, 2]))
    hold_target = np.asarray(data.qpos[7:], dtype=np.float64).copy()

    session_started = time.perf_counter()
    runtime = SonicRuntime(force_cpu=force_cpu, auto_reference_authority=False)
    session_load_s = time.perf_counter() - session_started
    try:
        measured = _observation(data, G1_29_JointIndex)
        initialization_started = time.perf_counter()
        runtime.initialize_from_observation(measured)
        initialization_s = time.perf_counter() - initialization_started

        runtime.ms.mode = LM.SLOW_WALK
        runtime.ms.speed = 0.2
        runtime.ms.height = -1.0
        runtime.ms.facing_angle = 0.0
        runtime.ms.movement_angle = 0.0
        runtime.ms.has_movement = True
        runtime.ms.needs_replan = True
        runtime.request_reference(measured)
        unmatched_before = {
            "step": runtime.step,
            "cursor": int(runtime.controller.ref_cursor),
            "feedback_pending": runtime.execution_feedback_pending,
            "authority": runtime.reference_authority,
        }
        reference_started = time.perf_counter()
        deadline = reference_started + reference_timeout_s
        while not runtime.poll_reference():
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"matching SONIC reference did not arrive within {reference_timeout_s:.1f}s"
                )
            time.sleep(0.001)
        reference_wait_s = time.perf_counter() - reference_started
        unmatched_after = {
            "step": runtime.step,
            "cursor": int(runtime.controller.ref_cursor),
            "feedback_pending": runtime.execution_feedback_pending,
            "authority": runtime.reference_authority,
        }
        runtime.acquire_reference(
            measured,
            executed_target=hold_target,
            expected_install_epoch=runtime.reference_install_epoch,
        )

        initial_reference_frames = int(runtime.controller.motion_timesteps)
        planner_positions = runtime.controller.motion_body_pos[:initial_reference_frames].copy()
        planner_quaternions = runtime.controller.motion_body_quats[:initial_reference_frames].copy()
        planner_prefix_dx = float(planner_positions[-1, 0] - planner_positions[0, 0])
        planner_prefix_reverse = max(
            0.0,
            float(planner_positions[0, 0] - np.min(planner_positions[:, 0])),
        )

        kp, kd = (np.asarray(values, dtype=np.float64) for values in compute_kp_kd())
        published = hold_target.copy()
        root_initial = np.asarray(data.qpos[:7], dtype=np.float64).copy()
        com_initial = np.asarray(data.subtree_com[pelvis_id], dtype=np.float64).copy()
        root_x: list[float] = []
        com_x: list[float] = []
        contact_samples: list[list[float | int]] = []
        physics_trace: list[dict[str, Any]] = []
        control_trace: list[dict[str, Any]] = []
        reference_installs: list[dict[str, Any]] = []
        transaction_failures: list[dict[str, Any]] = []
        arm_step_limits = np.asarray(ARM_STEP_LIMITS_RAD, dtype=np.float64)
        max_arm_steps = np.zeros(14, dtype=np.float64)
        first_arm_steps: Any | None = None
        arm_violations: list[dict[str, Any]] = []
        control_ms: list[float] = []
        max_target_step = 0.0
        max_action_roundtrip_error = 0.0
        max_tilt = 0.0
        max_joint_dq = 0.0
        max_qacc = 0.0
        max_abs_ctrl = 0.0
        max_ctrl_utilization = 0.0
        max_saturated_joint_count = 0
        finite = True
        root_z_min = float(data.qpos[2])
        root_z_max = float(data.qpos[2])
        warning_names = (
            "mjWARN_BADQPOS",
            "mjWARN_BADQVEL",
            "mjWARN_BADQACC",
            "mjWARN_BADCTRL",
        )
        warning_before = {
            name: int(data.warning[int(getattr(mujoco.mjtWarning, name))].number)
            for name in warning_names
        }

        for physics_tick in range(PHYSICS_STEPS):
            if physics_tick % CONTROL_EVERY == 0:
                control_tick = physics_tick // CONTROL_EVERY
                epoch_before = runtime.reference_install_epoch
                cursor_before = int(runtime.controller.ref_cursor)
                control_started = time.perf_counter_ns()
                action = runtime.tick(
                    _observation(data, G1_29_JointIndex),
                    use_joystick=False,
                    debug=False,
                )
                epoch_after_decode = runtime.reference_install_epoch
                cursor_after_decode = int(runtime.controller.ref_cursor)
                # Public boundary snapshot: tick() has returned its candidate,
                # and record_executed_target() will reject a stale identity.
                # The runner intentionally does not read runtime._pending_*.
                candidate_epoch = epoch_after_decode
                candidate_cursor = cursor_after_decode

                install_record = None
                if epoch_after_decode != epoch_before:
                    positions = runtime.controller.motion_body_pos[
                        : runtime.controller.motion_timesteps
                    ].copy()
                    install_record = {
                        "control_tick": control_tick,
                        "time_s": float(data.time),
                        "epoch_before": epoch_before,
                        "epoch_after": epoch_after_decode,
                        "cursor_before": cursor_before,
                        "candidate_cursor": candidate_cursor,
                        "frames": int(runtime.controller.motion_timesteps),
                        "prefix_dx_m": float(positions[-1, 0] - positions[0, 0]),
                        "prefix_reverse_m": max(
                            0.0,
                            float(positions[0, 0] - np.min(positions[:, 0])),
                        ),
                    }
                    reference_installs.append(install_record)

                raw_action = runtime.controller.last_raw_action_il.copy()
                policy_target = runtime.controller.last_policy_target_mujoco.copy()
                candidate = _target_vector(action, G1_29_JointIndex, np)
                target_delta = np.abs(candidate - published)
                max_target_step = max(max_target_step, float(np.max(target_delta)))
                arm_delta = target_delta[15:29]
                max_arm_steps = np.maximum(max_arm_steps, arm_delta)
                if control_tick == 0:
                    first_arm_steps = arm_delta.copy()
                for arm_index in np.flatnonzero(arm_delta > arm_step_limits + 1e-9):
                    arm_violations.append(
                        {
                            "control_tick": control_tick,
                            "joint": G1_29_JointIndex(15 + int(arm_index)).name,
                            "step_rad": float(arm_delta[arm_index]),
                            "limit_rad": float(arm_step_limits[arm_index]),
                        }
                    )
                published = candidate.copy()
                executed_action = runtime.record_executed_target(published)
                executed_target = normalized_action_to_hardware_target(executed_action)
                roundtrip_error = float(np.max(np.abs(executed_target - published)))
                max_action_roundtrip_error = max(max_action_roundtrip_error, roundtrip_error)
                epoch_after_commit = runtime.reference_install_epoch
                cursor_after_commit = int(runtime.controller.ref_cursor)
                expected_cursor = min(
                    int(candidate_cursor) + 1,
                    int(runtime.controller.motion_timesteps) - 1,
                )
                if epoch_after_commit != candidate_epoch:
                    transaction_failures.append(
                        {
                            "control_tick": control_tick,
                            "kind": "commit_epoch_changed",
                            "candidate": candidate_epoch,
                            "commit": epoch_after_commit,
                        }
                    )
                if cursor_after_commit != expected_cursor:
                    transaction_failures.append(
                        {
                            "control_tick": control_tick,
                            "kind": "commit_cursor_mismatch",
                            "actual": cursor_after_commit,
                            "expected": expected_cursor,
                        }
                    )
                if install_record is not None:
                    install_record["cursor_after_commit"] = cursor_after_commit
                elapsed_ms = (time.perf_counter_ns() - control_started) / 1e6
                control_ms.append(elapsed_ms)
                control_trace.append(
                    {
                        "control_tick": control_tick,
                        "time_s": float(data.time),
                        "epoch_before_tick": epoch_before,
                        "candidate_epoch": candidate_epoch,
                        "candidate_cursor": candidate_cursor,
                        "commit_cursor": cursor_after_commit,
                        "raw_action_isaaclab": raw_action.tolist(),
                        "raw_policy_target_mujoco": policy_target.tolist(),
                        "published_target_mujoco": published.tolist(),
                        "executed_action_isaaclab": executed_action.tolist(),
                        "roundtrip_error": roundtrip_error,
                        "target_step_max_rad": float(np.max(target_delta)),
                        "control_wall_ms": elapsed_ms,
                    }
                )

            pose = _body_pose(mujoco, np, model, data, torso_id)
            raw_wrench = np.zeros(6)
            raw_wrench[:3] = (
                10000.0 * (support_anchor - pose[:3]) - 1000.0 * pose[7:10]
            )
            data.xfrc_applied[torso_id] = stabilized_support_wrench(
                raw_wrench,
                pose,
                "locomotion",
            )
            q = np.asarray(data.qpos[7:])
            dq = np.asarray(data.qvel[6:])
            raw_ctrl = kp * (published - q) - kd * dq
            data.ctrl[:] = np.clip(
                raw_ctrl,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            saturated = np.abs(raw_ctrl - data.ctrl) > 1e-9
            saturated_count = int(np.count_nonzero(saturated))
            ctrl_denominator = np.maximum(
                np.abs(model.actuator_ctrlrange[:, 0]),
                np.abs(model.actuator_ctrlrange[:, 1]),
            )
            max_saturated_joint_count = max(max_saturated_joint_count, saturated_count)
            max_abs_ctrl = max(max_abs_ctrl, float(np.max(np.abs(data.ctrl))))
            max_ctrl_utilization = max(
                max_ctrl_utilization,
                float(np.max(np.abs(data.ctrl) / ctrl_denominator)),
            )
            mujoco.mj_step(model, data)

            loads, counts = _foot_loads(
                mujoco,
                np,
                model,
                data,
                robot_bodies,
                left_bodies,
                right_bodies,
            )
            contact_samples.append([loads[0], loads[1], counts[0], counts[1]])
            root = np.asarray(data.qpos[:7], dtype=np.float64).copy()
            com = np.asarray(data.subtree_com[pelvis_id], dtype=np.float64).copy()
            root_x.append(float(root[0]))
            com_x.append(float(com[0]))
            tick_tilt = _tilt_rad(np, root[3:7])
            tick_dq = float(np.max(np.abs(data.qvel[6:])))
            tick_qacc = float(np.max(np.abs(data.qacc)))
            max_tilt = max(max_tilt, tick_tilt)
            max_joint_dq = max(max_joint_dq, tick_dq)
            max_qacc = max(max_qacc, tick_qacc)
            root_z_min = min(root_z_min, float(root[2]))
            root_z_max = max(root_z_max, float(root[2]))
            tick_finite = all(
                np.all(np.isfinite(values))
                for values in (data.qpos, data.qvel, data.qacc, data.ctrl)
            )
            finite = finite and tick_finite
            physics_trace.append(
                {
                    "physics_tick": physics_tick,
                    "time_s": float(data.time),
                    "root_qpos": root.tolist(),
                    "whole_body_com": com.tolist(),
                    "left_load_N": loads[0],
                    "right_load_N": loads[1],
                    "left_contacts": counts[0],
                    "right_contacts": counts[1],
                    "tilt_rad": tick_tilt,
                    "max_joint_dq_rps": tick_dq,
                    "max_qacc": tick_qacc,
                    "saturated_joint_count": saturated_count,
                }
            )
            if not tick_finite:
                break

        warning_counts = {
            name: int(data.warning[int(getattr(mujoco.mjtWarning, name))].number)
            - warning_before[name]
            for name in warning_names
        }
        events = _contact_events(np, contact_samples)
        complete_contact_cycle = _has_complete_cycle(events)
        root_final = np.asarray(data.qpos[:7], dtype=np.float64).copy()
        com_final = np.asarray(data.subtree_com[pelvis_id], dtype=np.float64).copy()
        root_dx = float(root_final[0] - root_initial[0])
        root_reverse = max(0.0, float(root_initial[0] - min(root_x)))
        com_dx = float(com_final[0] - com_initial[0])
        com_reverse = max(0.0, float(com_initial[0] - min(com_x)))
        control_percentiles = np.percentile(
            np.asarray(control_ms, dtype=np.float64),
            [50, 95, 99, 100],
        )

        unmatched_stable = unmatched_before == unmatched_after == {
            "step": 0,
            "cursor": 0,
            "feedback_pending": False,
            "authority": False,
        }
        arm_verifier = [
            {
                "joint": G1_29_JointIndex(15 + index).name,
                "first_step_rad": float(first_arm_steps[index]),
                "max_step_rad": float(max_arm_steps[index]),
                "limit_rad": float(arm_step_limits[index]),
                "pass": bool(max_arm_steps[index] <= arm_step_limits[index] + 1e-9),
            }
            for index in range(14)
        ]
        gates = {
            "unmatched_does_not_advance": _gate(
                [unmatched_before, unmatched_after],
                unmatched_stable,
                "step/cursor/pending/authority remain 0/0/false/false before MATCHED",
            ),
            "planner_direction": _gate(
                {"dx_m": planner_prefix_dx, "reverse_m": planner_prefix_reverse},
                planner_prefix_dx > 0.10 and planner_prefix_reverse < 0.03,
                "planner prefix dx > 0.10 m and reverse < 0.03 m",
            ),
            "reference_transaction": _gate(
                {"commits": len(control_trace), "failures": transaction_failures},
                len(control_trace) == int(DURATION_S / CONTROL_DT_S)
                and not transaction_failures,
                "every candidate commits exactly its own (epoch,cursor)",
            ),
            "reference_refresh_exercised": _gate(
                reference_installs,
                bool(reference_installs),
                "at least one periodic refresh installs before its candidate is decoded",
            ),
            "root_direction": _gate(
                {"dx_m": root_dx, "reverse_m": root_reverse},
                root_dx > 0.02 and root_reverse < 0.03,
                "final dx > 0.02 m and reverse < 0.03 m",
            ),
            "com_direction": _gate(
                {"dx_m": com_dx, "reverse_m": com_reverse},
                com_dx > 0.02 and com_reverse < 0.03,
                "final dx > 0.02 m and reverse < 0.03 m",
            ),
            "complete_contact_cycle": _gate(
                events,
                complete_contact_cycle,
                "each foot has a debounced liftoff followed by touchdown",
            ),
            "finite": _gate(finite, finite, "qpos/qvel/qacc/ctrl remain finite"),
            "mujoco_warnings": _gate(
                warning_counts,
                not any(warning_counts.values()),
                "BADQPOS/BADQVEL/BADQACC/BADCTRL counts remain zero",
            ),
            "tilt": _gate(max_tilt, max_tilt < 0.35, "max root tilt < 0.35 rad"),
            "joint_velocity": _gate(
                max_joint_dq,
                max_joint_dq <= 7.0,
                "max measured joint |dq| <= existing field gate of 7 rad/s",
            ),
            "offline_qacc_diagnostic": _gate(
                max_qacc,
                max_qacc < 1000.0,
                (
                    "offline numerical-stability diagnostic: max generalized |qacc| < 1000; "
                    "not an official SONIC or product contract"
                ),
            ),
            "root_height": _gate(
                {"min_m": root_z_min, "max_m": root_z_max},
                root_z_min >= 0.55 and root_z_max <= 1.15,
                "root z remains in [0.55, 1.15] m",
            ),
            "torque_saturation": _gate(
                {
                    "max_saturated_joint_count": max_saturated_joint_count,
                    "max_abs_ctrl_Nm": max_abs_ctrl,
                    "max_ctrl_utilization": max_ctrl_utilization,
                },
                max_saturated_joint_count == 0,
                "PD ctrl never reaches an actuator ctrlrange limit",
            ),
            "action_roundtrip": _gate(
                max_action_roundtrip_error,
                max_action_roundtrip_error < 1e-6,
                "published target -> executed action -> target error < 1e-6",
            ),
            "arm_target_verifier": _gate(
                arm_verifier,
                not arm_violations,
                "14 arm target steps obey existing 6/4 rad/s x 20 ms verifier",
            ),
            "control_timing": _gate(
                float(control_percentiles[2]),
                float(control_percentiles[2]) < CONTROL_DT_S * 1000.0,
                "50 Hz controller wall-time p99 < 20 ms (offline diagnostic)",
            ),
        }
        passed = all(gate["pass"] for gate in gates.values())
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if passed else "FAIL",
            "generated_at_unix_s": time.time(),
            "contract": {
                "high_fidelity_reconstructed": True,
                "bit_exact": False,
                "profile_scope": "experimental Python ONNX evaluation; not official C++ acceptance",
                "hf_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                "dds_started": False,
                "service_started": False,
                "control_output_external_to_mujoco": False,
                "physics_hz": int(round(1.0 / PHYSICS_DT_S)),
                "sonic_hz": int(round(1.0 / CONTROL_DT_S)),
                "duration_s": DURATION_S,
                "binary_reference_authority": True,
                "outer_target_blend": False,
                "qacc_threshold_scope": (
                    "offline numerical-stability diagnostic only; not an official/product contract"
                ),
            },
            "provenance": provenance,
            "runtime_versions": {
                "python": platform.python_version(),
                "mujoco": getattr(mujoco, "__version__", "unknown"),
                "onnxruntime": ort.__version__,
                "available_ort_providers": ort.get_available_providers(),
                "selected_gpu": bool(runtime.use_gpu),
            },
            "timing": {
                "provenance_hash_s": provenance_s,
                "session_load_s": session_load_s,
                "measured_initialization_s": initialization_s,
                "matching_reference_wait_s": reference_wait_s,
                "control_ms_p50": float(control_percentiles[0]),
                "control_ms_p95": float(control_percentiles[1]),
                "control_ms_p99": float(control_percentiles[2]),
                "control_ms_max": float(control_percentiles[3]),
                "experiment_wall_s": time.perf_counter() - experiment_started,
            },
            "planner": {
                "initial_frames": initial_reference_frames,
                "initial_root_prefix_positions": planner_positions.tolist(),
                "initial_root_prefix_quaternions": planner_quaternions.tolist(),
                "initial_prefix_dx_m": planner_prefix_dx,
                "initial_prefix_reverse_m": planner_prefix_reverse,
                "refresh_installs": reference_installs,
            },
            "reference_transaction": {
                "identity_evidence": (
                    "public post-tick reference_install_epoch/controller.ref_cursor plus "
                    "successful record_executed_target enforcement"
                ),
                "unmatched_before": unmatched_before,
                "unmatched_after": unmatched_after,
                "commit_count": len(control_trace),
                "failures": transaction_failures,
            },
            "motion": {
                "root_initial_qpos": root_initial.tolist(),
                "root_final_qpos": root_final.tolist(),
                "root_dx_m": root_dx,
                "root_reverse_m": root_reverse,
                "whole_body_com_initial": com_initial.tolist(),
                "whole_body_com_final": com_final.tolist(),
                "whole_body_com_dx_m": com_dx,
                "whole_body_com_reverse_m": com_reverse,
                "contact_force_threshold_N": CONTACT_FORCE_THRESHOLD_N,
                "contact_debounce_steps": CONTACT_DEBOUNCE_STEPS,
                "contact_events": events,
                "complete_contact_cycle": complete_contact_cycle,
            },
            "dynamics": {
                "finite": finite,
                "warning_counts": warning_counts,
                "max_tilt_rad": max_tilt,
                "max_joint_dq_rps": max_joint_dq,
                "max_qacc": max_qacc,
                "root_z_min_m": root_z_min,
                "root_z_max_m": root_z_max,
                "max_abs_ctrl_Nm": max_abs_ctrl,
                "max_ctrl_utilization": max_ctrl_utilization,
                "max_saturated_joint_count": max_saturated_joint_count,
            },
            "actions": {
                "max_target_step_rad_record_only": max_target_step,
                "leg_target_step_has_no_unverified_rate_gate": True,
                "max_action_roundtrip_error": max_action_roundtrip_error,
                "arm_verifier": arm_verifier,
                "arm_violations": arm_violations,
            },
            "gates": gates,
            "control_trace": control_trace,
            "physics_trace": physics_trace,
        }
    finally:
        runtime.shutdown()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="new JSON evidence path; existing paths and symbolic links are rejected",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="use CPUExecutionProvider even when CUDA is available",
    )
    parser.add_argument(
        "--reference-timeout-s",
        type=float,
        default=30.0,
        help="wall-time limit for the initial async matching reference",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not math.isfinite(args.reference_timeout_s) or args.reference_timeout_s <= 0:
        print("--reference-timeout-s must be finite and positive", file=sys.stderr)
        return 2
    try:
        writer = ExclusiveJsonOutput(args.output)
    except (OSError, ValueError) as exc:
        print(f"cannot reserve --output: {exc}", file=sys.stderr)
        return 2
    try:
        try:
            evidence = run_acceptance(
                force_cpu=bool(args.force_cpu),
                reference_timeout_s=float(args.reference_timeout_s),
            )
        except Exception as exc:
            evidence = {
                "schema_version": SCHEMA_VERSION,
                "status": "ERROR",
                "generated_at_unix_s": time.time(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "contract": {
                    "hf_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                    "dds_started": False,
                    "service_started": False,
                },
            }
        writer.write(evidence)
    finally:
        writer.close()
    if evidence["status"] == "PASS":
        print(f"PASS: wrote OfficeV2 SONIC offline evidence to {writer.path}")
        return 0
    print(
        f"{evidence['status']}: wrote OfficeV2 SONIC offline evidence to {writer.path}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
