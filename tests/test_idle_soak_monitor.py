from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
idle_soak = importlib.import_module("idle_soak_monitor")


PROCESSES = {
    "mujoco-plant": {"pid": 101, "startTimeTicks": 1001},
    "sonic-driver": {"pid": 102, "startTimeTicks": 1002},
    "ros-cmd-vel-bridge": {"pid": 103, "startTimeTicks": 1003},
    "robot-agent-g1-sim": {"pid": 104, "startTimeTicks": 1004},
}
FIXTURE_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "idle_soak_cases.json").read_text(
        encoding="utf-8"
    )
)
OBSERVATION_WORKER_BOOT_ID = "22222222-2222-4222-8222-222222222222"
VIEWER_WORKER_BOOT_ID = "33333333-3333-4333-8333-333333333333"


def _worker_render_timing(
    *,
    stage: str | None = None,
    count: int = 0,
    last_ms: float = 5.0,
    max_ms: float | None = None,
    last_over_ms: float | None = None,
    last_over_at_unix_ns: int = 1_800_000_001_000_000_000,
) -> dict:
    stages = {}
    for name in idle_soak._OBSERVATION_RENDER_TIMING_STAGES:
        changed = name == stage
        value = last_ms if changed else 5.0
        stages[name] = {
            "lastMs": value,
            "maxMs": (
                max(8.0, value)
                if not changed or max_ms is None
                else max_ms
            ),
            "overThresholdCount": count if changed else 0,
            "lastOverThresholdMs": (
                (value if last_over_ms is None else last_over_ms)
                if changed and count
                else None
            ),
            "lastOverThresholdAtUnixNs": (
                last_over_at_unix_ns if changed and count else None
            ),
        }
    return {
        "thresholdMs": idle_soak.EXPECTED_WORKER_STALL_THRESHOLD_MS,
        "stages": stages,
    }


def _healthy_worker(*, pid: int, boot_id: str, viewer: bool = False) -> dict:
    value = {
        "status": "running",
        "ready": True,
        "alive": True,
        "exitCode": None,
        "workerPid": pid,
        "workerBootId": boot_id,
        "failureCount": 0,
        "lastFailureAtUnixNs": None,
        "lastError": None,
    }
    if viewer:
        value.update(
            {
                "authorityFeedbackEnabled": False,
                "keyCallbackRegistered": False,
                "lastSuccessSequence": 100,
                "lastSuccessAtUnixNs": 1_800_000_000_000_000_000,
                "mailboxPublishedCount": 100,
                "mailboxDroppedCount": 0,
                "lastSuccessMailboxDroppedCount": 0,
                "progressAgeMs": 10.0,
                "staleAfterMs": 2_000.0,
            }
        )
    else:
        value["renderTiming"] = _worker_render_timing()
    return value


def _disabled_viewer() -> dict:
    return {
        "status": "disabled",
        "workerPid": None,
        "workerBootId": None,
        "failureCount": 0,
        "lastFailureAtUnixNs": None,
        "lastError": None,
        "authorityFeedbackEnabled": False,
        "keyCallbackRegistered": False,
    }


def _plant(
    index: int,
    *,
    baseline_count: int = 7,
    stage: str | None = None,
    count_delta: int = 0,
    linear_speed: float = 0.0,
) -> dict:
    timestamp = 1_800_000_000.0 + index
    stages = {
        name: {
            "lastS": 0.005,
            "maxS": 0.08,
            "overThresholdCount": (
                baseline_count + count_delta if name == stage else baseline_count
            ),
            "lastOverThresholdAtUnixNs": None,
        }
        for name in idle_soak._PLANT_STAGES
    }
    return {
        "status": "running",
        "physicsStatus": "running",
        "observationStatus": "running",
        "schemaVersion": "h-ide.mujoco-capture/2",
        "robotId": "g1-sim",
        "sourceId": "mujoco:unitree-g1:g1-sim",
        "localizationEpochId": "g1-sim-map-epoch-1",
        "sequence": 1000 + index,
        "capturedAtUnixNs": int(timestamp * 1_000_000_000),
        "producerBootId": "11111111-1111-4111-8111-111111111111",
        "observationWorker": _healthy_worker(
            pid=201,
            boot_id=OBSERVATION_WORKER_BOOT_ID,
        ),
        "viewerWorker": _disabled_viewer(),
        "pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.763,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        },
        "rootVelocity": {
            "frameId": "base_link",
            "source": "mujoco.mj_objectVelocity(local)",
            "vx": linear_speed,
            "vy": 0.0,
            "vz": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "linearSpeedMps": linear_speed,
            "angularSpeedRps": 0.0,
        },
        "support": {
            "mode": "safety_stop",
            "updatedAt": timestamp,
            "fault": None,
            "safetyStopLatched": True,
            "safetyStopEpoch": 1,
            "lastRearmGeneration": None,
        },
        "runtimeTiming": {
            "stallThresholdS": 0.1,
            "simLoopGap": {
                "lastS": 0.006,
                "maxS": 0.08,
                "overThresholdCount": baseline_count,
                "lastOverThresholdAtUnixNs": None,
            },
            "stages": stages,
            "lastSlowStage": stage if count_delta else None,
            "lastSlowDurationS": 0.12 if count_delta else None,
            "lastSlowAtUnixNs": int(timestamp * 1_000_000_000) if count_delta else None,
        },
    }


def _runtime_diagnostics(
    *,
    baseline_count: int = 3,
    prefix: str | None = None,
    count_delta: int = 0,
) -> dict:
    value = {
        "runtime_stall_threshold_s": 0.1,
        "python_gc_enabled": True,
        "python_gc_pause_active": False,
        "python_gc_collection_count": 10,
        "python_gc_last_generation": 1,
        "python_gc_last_pause_s": 0.001,
        "python_gc_max_pause_s": 0.01,
        "python_gc_total_pause_s": 0.02,
        "python_gc_last_completed_at_unix_s": 1_800_000_000.0,
        "python_gc_warmup_collected_objects": 17,
        "python_gc_freeze_applied": True,
        "python_gc_frozen_objects": 123,
        "python_gc_freeze_error": None,
    }
    for name in idle_soak._SONIC_GAP_PREFIXES:
        changed = count_delta if name == prefix else 0
        value.update(
            {
                f"{name}_last_gap_s": 0.02,
                f"{name}_max_gap_s": 0.15,
                f"{name}_over_threshold_count": baseline_count + changed,
                f"{name}_last_over_threshold_at_unix_s": (
                    1_800_000_001.0 if changed else 1_799_999_999.0
                ),
                f"{name}_last_gap_gc_pause_s": 0.0,
                f"{name}_last_over_threshold_gap_s": 0.15,
                f"{name}_last_over_threshold_gc_pause_s": 0.0,
                f"{name}_over_threshold_gc_overlap_count": 1,
                f"{name}_over_threshold_non_gc_residual_count": (
                    baseline_count + changed
                ),
            }
        )
    return value


def _sonic(
    index: int,
    *,
    baseline_count: int = 3,
    gap_prefix: str | None = None,
    count_delta: int = 0,
) -> dict:
    timestamp = 1_800_000_000.0 + index
    return {
        "backend": "mujoco_sonic",
        "robot_id": "g1-sim",
        "status": "running",
        "motion_ready": True,
        "updated_at": timestamp,
        "plant_tick": 5000 + index * 1000,
        "control_tick_ms": 5000 + index * 1000,
        "control_clock_status": "simulation_time_50hz_motion_ready",
        "nav2_state": "safety_stop",
        "sonic_mode": "IDLE",
        "sonic_speed_mps": 0.0,
        "sonic_policy_speed_mps": 0.0,
        "last_cmd_vx": 0.0,
        "last_cmd_vy": 0.0,
        "last_cmd_wz": 0.0,
        "navigation_task_active": False,
        "static_hold_active": True,
        "motion_handoff_active": False,
        "motion_handoff_ratio": 0.0,
        "sonic_planner_reference_ready": False,
        "sonic_planner_required_pending": False,
        "sonic_reference_authority_state": "REVOKED",
        "sonic_reference_actuator_authority": False,
        "sonic_reference_authority_dds_target_sequence": None,
        "sonic_execution_feedback_pending": False,
        "plant_support_mode": "safety_stop",
        "plant_support_requested_mode": "safety_stop",
        "plant_support_confirmed_mode": "safety_stop",
        "plant_support_stop_pending": False,
        "plant_support_safety_latched": True,
        "plant_support_stop_generation": 1,
        "fault_trace_path": None,
        "fault_trace_sample_count": 0,
        "last_error": "Robot Agent requested simulation safe-stop",
        "navigation_arm_constrained_joints": 0,
        "navigation_arm_max_step_rad": 0.0,
        "navigation_arm_raw_max_deviation_rad": 0.68,
        "navigation_arm_raw_max_error_from_measured_rad": 0.02,
        "navigation_arm_measured_max_deviation_rad": 0.70,
        "navigation_arm_max_velocity_rps": 0.01,
        "navigation_arm_max_estimated_torque_nm": 0.3,
        "navigation_task_peak_arm_constrained_joints": 0,
        "navigation_task_peak_arm_raw_deviation_rad": 0.0,
        "navigation_task_peak_arm_tracking_error_rad": 0.0,
        "navigation_task_peak_arm_measured_deviation_rad": 0.0,
        "navigation_task_peak_arm_velocity_rps": 0.0,
        "navigation_task_peak_arm_estimated_torque_nm": 0.0,
        "lower_body_target_max_error_rad": 0.08,
        "lower_body_target_rms_error_rad": 0.035,
        "lower_body_max_velocity_rps": 0.01,
        "lower_body_max_estimated_torque_nm": 2.3,
        "runtime_diagnostics": _runtime_diagnostics(
            baseline_count=baseline_count,
            prefix=gap_prefix,
            count_delta=count_delta,
        ),
    }


def _robot_agent(*, ready: bool = True, nonfinite: bool = False) -> dict:
    control = {
        "state": "SAFE_HOLD",
        "active": False,
        "standby": ready,
        "canAcquire": ready,
        "authorityRevoked": True,
        "safeStopState": "confirmed",
        "safeStopEpoch": 12,
        "leaseValid": False,
        "leaseRemainingMs": float("nan") if nonfinite else None,
        "activeCommandCount": 0,
    }
    fault = {
        "required": True,
        "ready": ready,
        "reason": None if ready else "motion_safety_fault_latched",
        "latched": not ready,
        "eventId": None if ready else "event-1",
        "source": None if ready else "sonic_driver",
        "code": None if ready else "PLANT_LOWSTATE_STALE",
        "counters": {
            "accepted": 0 if ready else 1,
            "duplicate": 0,
            "invalid": 0,
            "sequenceConflict": 0,
            "staleSequence": 0,
            "wrongEpoch": 0,
            "wrongRobot": 0,
        },
    }
    return {
        "schemaVersion": "robot-agent.diagnostics/1",
        "robotId": "g1-sim",
        "robotBootId": "11111111-1111-4111-8111-111111111111",
        "ready": ready,
        "commandReadiness": {
            "ready": ready,
            "mode": "STANDBY" if ready else "BLOCKED",
            "reason": None,
        },
        "controlState": control,
        "motionSafetyFault": fault,
        "checks": {
            "runtime": {"required": True, "ready": True},
            "transportSession": {"required": True, "ready": True},
            "control": {"required": True, "ready": ready},
            "safeStop": {"required": True, "ready": True},
            "safetyEndpoints": {"required": True, "ready": True},
            "rosExecutor": {"required": True, "ready": True},
            "nav2Actions": {"required": True, "ready": True},
            "localization": {"required": True, "ready": True},
            "observation": {"required": True, "ready": True},
            "motionSafetyFault": fault,
        },
    }


def _batch(
    index: int,
    *,
    plant_count: int = 7,
    plant_stage: str | None = None,
    plant_delta: int = 0,
    sonic_count: int = 3,
    sonic_prefix: str | None = None,
    sonic_delta: int = 0,
    robot_ready: bool = True,
    robot_nonfinite: bool = False,
) -> idle_soak.EndpointBatch:
    return idle_soak.EndpointBatch(
        payloads={
            "plant": _plant(
                index,
                baseline_count=plant_count,
                stage=plant_stage,
                count_delta=plant_delta,
            ),
            "sonic": _sonic(
                index,
                baseline_count=sonic_count,
                gap_prefix=sonic_prefix,
                count_delta=sonic_delta,
            ),
            "robotAgent": _robot_agent(
                ready=robot_ready,
                nonfinite=robot_nonfinite,
            ),
        },
        latency_ms={"plant": 1.0, "sonic": 1.0, "robotAgent": 2.0},
        errors={},
    )


def _empty_log() -> idle_soak.LogScan:
    return idle_soak.LogScan((), (), (), {})


def _observe(
    evaluator: idle_soak.IdleSoakEvaluator,
    batch: idle_soak.EndpointBatch,
    index: int,
    *,
    processes: dict | None = None,
    log_scan: idle_soak.LogScan | None = None,
) -> dict:
    return evaluator.observe(
        batch,
        process_snapshot=PROCESSES if processes is None else processes,
        process_errors=(),
        log_scan=_empty_log() if log_scan is None else log_scan,
        observed_s=float(index),
    )


def test_baseline_nonzero_counters_do_not_block_a_clean_soak() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    second = _batch(1)
    for batch in (first, second):
        batch.payloads["plant"]["observationWorker"]["renderTiming"] = (
            _worker_render_timing(
                stage="totalMs",
                count=4,
                last_ms=5.0,
                max_ms=150.0,
                last_over_ms=150.0,
            )
        )
    _observe(evaluator, first, 0)
    _observe(evaluator, second, 1)

    result = evaluator.finalize()

    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is True
    assert result["quarantineExercised"] is False
    assert set(result["runtimeCounterDeltas"].values()) == {0}
    assert set(result["workerRenderTimingCounterDeltas"].values()) == {0}
    assert result["plantWorkerBaseline"]["observationWorker"]["identity"] == {
        "workerPid": 201,
        "workerBootId": OBSERVATION_WORKER_BOOT_ID,
    }
    assert result["plantWorkerFinal"]["viewerWorker"]["mode"] == "disabled"


def test_worker_identities_and_failure_evidence_remain_stable() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    baseline = _observe(evaluator, _batch(0), 0)
    sample = _observe(evaluator, _batch(1), 1)

    result = evaluator.finalize()

    assert baseline["plantWorkers"]["baseline"] is True
    assert sample["plantWorkers"]["baseline"] is False
    assert sample["plantWorkers"]["identityChanges"] == {}
    assert sample["plantWorkers"]["failureChanges"] == {}
    assert result["safeHoldSafetyPass"] is True


def test_observation_worker_identity_change_is_a_hard_failure_with_evidence() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    changed = _batch(1)
    changed.payloads["plant"]["observationWorker"].update(
        {
            "workerPid": 301,
            "workerBootId": "44444444-4444-4444-8444-444444444444",
        }
    )

    derived = _observe(evaluator, changed, 1)
    result = evaluator.finalize()

    evidence = derived["plantWorkers"]["identityChanges"]["observationWorker"]
    assert evidence["baseline"]["workerPid"] == 201
    assert evidence["current"]["workerPid"] == 301
    failure = next(
        failure
        for failure in result["failures"]
        if failure["code"] == "PLANT_WORKER_IDENTITY_CHANGED"
    )
    assert '"baseline"' in failure["detail"]
    assert '"current"' in failure["detail"]
    assert result["safeHoldSafetyPass"] is False


def test_observation_worker_failure_change_is_a_hard_failure_with_evidence() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    changed = _batch(1)
    changed.payloads["plant"]["observationWorker"].update(
        {
            "failureCount": 1,
            "lastFailureAtUnixNs": 1_800_000_001_000_000_000,
            "lastError": "jpeg encoder failed once",
        }
    )

    derived = _observe(evaluator, changed, 1)
    result = evaluator.finalize()

    evidence = derived["plantWorkers"]["failureChanges"]["observationWorker"]
    assert evidence["baseline"]["failureCount"] == 0
    assert evidence["current"]["failureCount"] == 1
    failure = next(
        failure
        for failure in result["failures"]
        if failure["code"] == "PLANT_WORKER_FAILURE_CHANGED"
    )
    assert "jpeg encoder failed once" in failure["detail"]
    assert result["safeHoldSafetyPass"] is False


def test_enabled_viewer_identity_and_failure_changes_are_preserved_as_evidence() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    second = _batch(1)
    first.payloads["plant"]["viewerWorker"] = _healthy_worker(
        pid=202,
        boot_id=VIEWER_WORKER_BOOT_ID,
        viewer=True,
    )
    second.payloads["plant"]["viewerWorker"] = _healthy_worker(
        pid=302,
        boot_id="55555555-5555-4555-8555-555555555555",
        viewer=True,
    )
    second.payloads["plant"]["viewerWorker"].update(
        {
            "failureCount": 1,
            "lastFailureAtUnixNs": 1_800_000_001_000_000_000,
            "lastError": "viewer context was recreated",
        }
    )
    _observe(evaluator, first, 0)

    derived = _observe(evaluator, second, 1)
    result = evaluator.finalize()

    assert "viewerWorker" in derived["plantWorkers"]["identityChanges"]
    assert "viewerWorker" in derived["plantWorkers"]["failureChanges"]
    codes = {failure["code"] for failure in result["failures"]}
    assert codes >= {
        "PLANT_WORKER_IDENTITY_CHANGED",
        "PLANT_WORKER_FAILURE_CHANGED",
    }
    details = " ".join(failure["detail"] for failure in result["failures"])
    assert "viewer context was recreated" in details


def test_enabled_viewer_is_stable_and_cannot_share_the_plant_pid() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    second = _batch(1)
    for batch in (first, second):
        batch.payloads["plant"]["viewerWorker"] = _healthy_worker(
            pid=202,
            boot_id=VIEWER_WORKER_BOOT_ID,
            viewer=True,
        )
    _observe(evaluator, first, 0)
    derived = _observe(evaluator, second, 1)

    result = evaluator.finalize()

    viewer = derived["plantWorkers"]["workers"]["viewerWorker"]
    assert viewer["mode"] == "enabled"
    assert viewer["identity"]["workerPid"] == 202
    assert result["safeHoldSafetyPass"] is True

    invalid = idle_soak.IdleSoakEvaluator()
    wrong = _batch(0)
    wrong.payloads["plant"]["viewerWorker"] = _healthy_worker(
        pid=PROCESSES["mujoco-plant"]["pid"],
        boot_id=VIEWER_WORKER_BOOT_ID,
        viewer=True,
    )
    _observe(invalid, wrong, 0)
    _observe(invalid, wrong, 1)
    invalid_result = invalid.finalize()
    assert {failure["code"] for failure in invalid_result["failures"]} >= {
        "PLANT_WORKER_IDENTITY_INVALID"
    }


def test_disabled_viewer_rejects_phantom_identity_or_failure() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    first.payloads["plant"]["viewerWorker"].update(
        {
            "workerPid": 202,
            "failureCount": 1,
            "lastError": "worker should not exist",
        }
    )
    _observe(evaluator, first, 0)
    _observe(evaluator, _batch(1), 1)

    result = evaluator.finalize()

    codes = {failure["code"] for failure in result["failures"]}
    assert codes >= {
        "PLANT_WORKER_HEALTH_INVALID",
        "PLANT_WORKER_IDENTITY_INVALID",
    }
    details = " ".join(failure["detail"] for failure in result["failures"])
    assert "viewerWorker.workerPid" in details
    assert "viewerWorker.failureCount" in details


@pytest.mark.parametrize("stage", ["snapshotPublish", "viewerSnapshotPublish"])
def test_new_snapshot_publish_stages_are_tracked(stage: str) -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    _observe(
        evaluator,
        _batch(1, plant_stage=stage, plant_delta=1),
        1,
    )

    result = evaluator.finalize()

    assert result["runtimeCounterDeltas"][
        f"plant.stages.{stage}.overThresholdCount"
    ] == 1
    assert result["movementReleaseAllowed"] is False


def test_continuous_250ms_observation_worker_total_samples_force_hold() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    last_derived = None
    for index in (1, 2):
        batch = _batch(index)
        batch.payloads["plant"]["observationWorker"]["renderTiming"] = (
            _worker_render_timing(
                stage="totalMs",
                count=index,
                last_ms=250.0,
            )
        )
        last_derived = _observe(evaluator, batch, index)

    assert last_derived is not None
    result = evaluator.finalize()
    timing = last_derived["plantWorkers"]["renderTiming"]
    path = (
        "plant.observationWorker.renderTiming.stages.totalMs."
        "overThresholdCount"
    )

    assert timing["stallEvent"] is True
    assert timing["counterChanges"][path] == {"previous": 1, "current": 2}
    assert timing["eventEvidence"]["stages"]["totalMs"]["lastMs"] == 250.0
    assert result["workerRenderTimingCounterDeltas"][path] == 2
    assert len(result["workerRenderTimingCounterDeltas"]) == len(
        idle_soak._OBSERVATION_RENDER_TIMING_STAGES
    )
    assert set(result["runtimeCounterDeltas"].values()) == {0}
    assert result["runtimeStallSeen"] is True
    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is False
    assert result["quarantineExercised"] is True


def test_worker_last_slow_duration_survives_a_normal_render_sample() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    samples = (
        _worker_render_timing(
            stage="totalMs",
            count=1,
            last_ms=150.0,
            max_ms=150.0,
            last_over_at_unix_ns=1_800_000_001_000_000_000,
        ),
        _worker_render_timing(
            stage="totalMs",
            count=2,
            last_ms=120.0,
            max_ms=150.0,
            last_over_at_unix_ns=1_800_000_002_000_000_000,
        ),
        _worker_render_timing(
            stage="totalMs",
            count=2,
            last_ms=5.0,
            max_ms=150.0,
            last_over_ms=120.0,
            last_over_at_unix_ns=1_800_000_002_000_000_000,
        ),
    )
    for index, timing in enumerate(samples, start=1):
        batch = _batch(index)
        batch.payloads["plant"]["observationWorker"]["renderTiming"] = timing
        derived = _observe(evaluator, batch, index)

    stage = derived["plantWorkers"]["renderTiming"]["stages"]["totalMs"]
    result = evaluator.finalize()

    assert derived["plantWorkers"]["renderTiming"]["stallEvent"] is False
    assert stage == {
        "lastMs": 5.0,
        "maxMs": 150.0,
        "overThresholdCount": 2,
        "lastOverThresholdMs": 120.0,
        "lastOverThresholdAtUnixNs": 1_800_000_002_000_000_000,
    }
    assert result["runtimeStallSeen"] is True
    assert result["movementReleaseAllowed"] is False


def test_every_published_observation_worker_stage_counter_is_tracked() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    second = _batch(1)
    extra_stage = {
        "lastMs": 1.0,
        "maxMs": 1.0,
        "overThresholdCount": 0,
        "lastOverThresholdMs": None,
        "lastOverThresholdAtUnixNs": None,
    }
    first.payloads["plant"]["observationWorker"]["renderTiming"]["stages"][
        "gpuFenceMs"
    ] = dict(extra_stage)
    second.payloads["plant"]["observationWorker"]["renderTiming"]["stages"][
        "gpuFenceMs"
    ] = {
        **extra_stage,
        "lastMs": 150.0,
        "maxMs": 150.0,
        "overThresholdCount": 1,
        "lastOverThresholdMs": 150.0,
        "lastOverThresholdAtUnixNs": 1_800_000_001_000_000_000,
    }

    _observe(evaluator, first, 0)
    derived = _observe(evaluator, second, 1)
    result = evaluator.finalize()
    path = (
        "plant.observationWorker.renderTiming.stages.gpuFenceMs."
        "overThresholdCount"
    )

    assert derived["plantWorkers"]["renderTiming"]["counterChanges"][path]
    assert result["workerRenderTimingCounterDeltas"][path] == 1
    assert result["movementReleaseAllowed"] is False


def test_enabled_viewer_progress_ignores_wall_clock_rollback() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    second = _batch(1)
    first.payloads["plant"]["viewerWorker"] = _healthy_worker(
        pid=202,
        boot_id=VIEWER_WORKER_BOOT_ID,
        viewer=True,
    )
    first.payloads["plant"]["viewerWorker"]["lastSuccessAtUnixNs"] = (
        1_800_000_010_000_000_000
    )
    second.payloads["plant"]["viewerWorker"] = _healthy_worker(
        pid=202,
        boot_id=VIEWER_WORKER_BOOT_ID,
        viewer=True,
    )
    second.payloads["plant"]["viewerWorker"].update(
        {
            "lastSuccessSequence": 101,
            "lastSuccessAtUnixNs": 1_800_000_009_000_000_000,
            "mailboxPublishedCount": 101,
        }
    )

    _observe(evaluator, first, 0)
    derived = _observe(evaluator, second, 1)
    result = evaluator.finalize()

    assert derived["plantWorkers"]["viewerProgressErrors"] == []
    assert derived["plantWorkers"]["identityChanges"] == {}
    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is True


def test_enabled_viewer_fresh_concurrent_progress_commit_is_healthy() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    second = _batch(1)
    for batch in (first, second):
        batch.payloads["plant"]["viewerWorker"] = _healthy_worker(
            pid=202,
            boot_id=VIEWER_WORKER_BOOT_ID,
            viewer=True,
        )
    second.payloads["plant"]["viewerWorker"].update(
        {
            "lastSuccessSequence": 101,
            "lastSuccessAtUnixNs": 1_800_000_011_000_000_000,
            "mailboxPublishedCount": 101,
            "progressAgeMs": 0.0,
        }
    )

    _observe(evaluator, first, 0)
    derived = _observe(evaluator, second, 1)
    result = evaluator.finalize()

    assert derived["plantWorkers"]["viewerProgressErrors"] == []
    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is True


def test_enabled_viewer_mailbox_growth_within_stale_window_is_not_stale() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0)
    second = _batch(1)
    for batch in (first, second):
        batch.payloads["plant"]["viewerWorker"] = _healthy_worker(
            pid=202,
            boot_id=VIEWER_WORKER_BOOT_ID,
            viewer=True,
        )
    second.payloads["plant"]["viewerWorker"]["mailboxPublishedCount"] = 101

    _observe(evaluator, first, 0)
    derived = _observe(evaluator, second, 1)
    result = evaluator.finalize()

    assert derived["plantWorkers"]["viewerProgressErrors"] == []
    assert derived["plantWorkers"]["viewerProgressEvidence"] == {}
    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is True


def test_enabled_viewer_exact_stale_boundary_is_not_stale() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    for index in (0, 1):
        batch = _batch(index)
        batch.payloads["plant"]["viewerWorker"] = _healthy_worker(
            pid=202,
            boot_id=VIEWER_WORKER_BOOT_ID,
            viewer=True,
        )
        batch.payloads["plant"]["viewerWorker"].update(
            {
                "mailboxPublishedCount": 101,
                "progressAgeMs": 2_000.0,
            }
        )
        derived = _observe(evaluator, batch, index)

    result = evaluator.finalize()

    assert derived["plantWorkers"]["viewerProgressErrors"] == []
    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is True


def test_enabled_viewer_with_published_mailbox_and_two_second_stale_success_holds() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    for index in (0, 1):
        batch = _batch(index)
        batch.payloads["plant"]["viewerWorker"] = _healthy_worker(
            pid=202,
            boot_id=VIEWER_WORKER_BOOT_ID,
            viewer=True,
        )
        batch.payloads["plant"]["viewerWorker"].update(
            {
                "status": "stale",
                "ready": False,
                "mailboxPublishedCount": 101,
                "progressAgeMs": 2_000.001,
            }
        )
        derived = _observe(evaluator, batch, index)

    result = evaluator.finalize()

    assert any(
        "last success" in error
        for error in derived["plantWorkers"]["viewerProgressErrors"]
    )
    assert result["safeHoldSafetyPass"] is False
    assert result["movementReleaseAllowed"] is False


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=lambda case: case["name"])
def test_offline_dual_verdict_fixtures(case: dict) -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    plant_delta = int(case["plant_delta"])
    sonic_delta = int(case["sonic_delta"])
    _observe(
        evaluator,
        _batch(
            1,
            plant_stage="rgbRender" if plant_delta else None,
            plant_delta=plant_delta,
            sonic_prefix="lowstate_callback" if sonic_delta else None,
            sonic_delta=sonic_delta,
            robot_ready=bool(case["robot_ready"]),
        ),
        1,
    )

    result = evaluator.finalize()

    assert result["safeHoldSafetyPass"] is case["safe_hold_pass"]
    assert result["movementReleaseAllowed"] is case[
        "movement_release_allowed"
    ]
    assert result["quarantineExercised"] is case["quarantine_exercised"]


def test_runtime_counter_increment_keeps_safe_hold_but_blocks_movement() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    derived = _observe(
        evaluator,
        _batch(
            1,
            plant_stage="rgbRender",
            plant_delta=1,
            sonic_prefix="lowstate_callback",
            sonic_delta=1,
        ),
        1,
    )

    result = evaluator.finalize()

    assert derived["runtime"]["stallEvent"] is True
    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is False
    assert result["quarantineExercised"] is True
    assert result["runtimeCounterDeltas"][
        "plant.stages.rgbRender.overThresholdCount"
    ] == 1
    assert result["runtimeCounterDeltas"][
        "sonic.lowstate_callback_over_threshold_count"
    ] == 1


def test_stall_that_latches_robot_agent_fails_both_verdicts() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    _observe(
        evaluator,
        _batch(1, sonic_prefix="lowstate_callback", sonic_delta=1, robot_ready=False),
        1,
    )

    result = evaluator.finalize()

    assert result["safeHoldSafetyPass"] is False
    assert result["movementReleaseAllowed"] is False
    assert result["quarantineExercised"] is False
    assert {failure["code"] for failure in result["failures"]} >= {
        "ROBOT_AGENT_NOT_READY"
    }


def test_runtime_counter_rollback_is_a_hard_failure() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0, plant_count=7), 0)
    _observe(evaluator, _batch(1, plant_count=6), 1)

    result = evaluator.finalize()

    assert result["safeHoldSafetyPass"] is False
    assert {failure["code"] for failure in result["failures"]} >= {
        "RUNTIME_COUNTER_ROLLBACK"
    }


def test_process_restart_and_qacc_log_are_hard_failures() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    restarted = dict(PROCESSES)
    restarted["sonic-driver"] = {"pid": 202, "startTimeTicks": 2002}
    qacc = idle_soak.LogScan(
        hard_matches=(
            {"log": "mujoco-plant.log", "offset": 4, "line": "QACC is invalid"},
        ),
        stale_matches=(),
        errors=(),
        offsets={"mujoco-plant.log": 20},
    )
    _observe(evaluator, _batch(1), 1, processes=restarted, log_scan=qacc)

    result = evaluator.finalize()

    assert result["safeHoldSafetyPass"] is False
    assert {failure["code"] for failure in result["failures"]} >= {
        "PROCESS_RESTARTED",
        "NUMERICAL_RESET_OR_CRASH_LOG",
    }


def test_nonfinite_robot_agent_and_excess_root_speed_fail() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    first = _batch(0, robot_nonfinite=True)
    first.payloads["plant"] = _plant(0, linear_speed=0.03)
    _observe(evaluator, first, 0)
    _observe(evaluator, _batch(1), 1)

    result = evaluator.finalize()

    assert result["safeHoldSafetyPass"] is False
    assert {failure["code"] for failure in result["failures"]} >= {
        "ROBOT_AGENT_NOT_READY",
        "IDLE_ROOT_SPEED_EXCEEDED",
    }


class _Response:
    status = 200

    def __init__(self, url: str, body: bytes) -> None:
        self.url = url
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, _limit: int) -> bytes:
        return self.body


def test_sampler_uses_only_bodyless_gets_to_the_exact_allowlist() -> None:
    requests = []

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return _Response(request.full_url, b'{"ok":true}')

    sampler = idle_soak.ReadonlyIdleSoakSampler(opener=opener)
    batch = sampler.sample(include_robot_agent=True)

    assert batch.errors == {}
    assert set(batch.payloads) == {"plant", "sonic", "robotAgent"}
    assert [request.full_url for request, _timeout in requests] == [
        idle_soak.PLANT_URL,
        idle_soak.SONIC_URL,
        idle_soak.ROBOT_AGENT_URL,
    ]
    assert all(request.method == "GET" for request, _timeout in requests)
    assert all(request.data is None for request, _timeout in requests)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plant_url": "http://127.0.0.1:8794/support"},
        {"sonic_url": "http://localhost:8792/telemetry"},
        {"robot_agent_url": "http://127.0.0.1:8796/readyz"},
        {
            "plant_url": idle_soak.SONIC_URL,
            "sonic_url": idle_soak.PLANT_URL,
        },
    ],
)
def test_sampler_rejects_every_url_outside_its_exact_role(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="fixed loopback GET allowlist"):
        idle_soak.ReadonlyIdleSoakSampler(**kwargs)


def test_safe_writer_refuses_existing_files_and_symlinks(tmp_path: Path) -> None:
    existing = tmp_path / "existing.jsonl"
    existing.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        idle_soak.SafeJsonlEvidenceWriter(existing)

    real = tmp_path / "real.jsonl"
    real.write_text("preserve\n", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        idle_soak.SafeJsonlEvidenceWriter(link)


def test_process_identity_probe_reads_pid_and_start_ticks(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    proc = tmp_path / "proc"
    runtime.mkdir()
    (runtime / "worker.pid").write_text("123\n", encoding="ascii")
    process = proc / "123"
    process.mkdir(parents=True)
    fields = ["S", *(str(field) for field in range(4, 53))]
    (process / "stat").write_text(
        "123 (worker with space) " + " ".join(fields),
        encoding="utf-8",
    )

    values, errors = idle_soak.ProcessIdentityProbe(
        runtime,
        proc_root=proc,
        names=("worker",),
    ).snapshot()

    assert errors == ()
    assert values == {"worker": {"pid": 123, "startTimeTicks": 22}}


def test_runtime_log_scanner_reads_only_appended_failure_clues(tmp_path: Path) -> None:
    log = tmp_path / "plant.log"
    log.write_text("healthy startup\n", encoding="utf-8")
    scanner = idle_soak.RuntimeLogScanner(tmp_path, log_names=("plant.log",))

    baseline = scanner.scan()
    with log.open("a", encoding="utf-8") as stream:
        stream.write("motion health PLANT_LOWSTATE_STALE quarantined\n")
        stream.write(
            "SONIC_HEALTH_PROBE_TIMEOUT state=PENDING count=1/2 "
            "totalTimeoutCount=1 recoveredTimeoutCount=0 "
            "confirmedTimeoutCount=0\n"
        )
        stream.write(
            "SONIC_HEALTH_PROBE_TIMEOUT state=RECOVERED count=0/2 "
            "totalTimeoutCount=1 recoveredTimeoutCount=1 "
            "confirmedTimeoutCount=0\n"
        )
        stream.write("MuJoCo QACC became non-finite\n")
    changed = scanner.scan()
    recovered = scanner.scan()

    assert baseline.hard_matches == baseline.stale_matches == ()
    assert len(changed.stale_matches) == 1
    assert len(changed.transport_timeout_matches) == 2
    assert changed.transport_timeout_counter_rollbacks == ()
    assert len(changed.hard_matches) == 1
    assert changed.hard_matches[0]["line"] == "MuJoCo QACC became non-finite"
    assert recovered.hard_matches == recovered.stale_matches == ()
    assert recovered.transport_timeout_matches == ()


def test_recovered_health_transport_timeout_keeps_safe_hold_but_fails_release() -> None:
    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    timeout_log = idle_soak.LogScan(
        hard_matches=(),
        stale_matches=(),
        errors=(),
        offsets={"ros-cmd-vel-bridge.log": 100},
        transport_timeout_matches=(
            {
                "log": "ros-cmd-vel-bridge.log",
                "offset": 20,
                "line": (
                    "SONIC_HEALTH_PROBE_TIMEOUT state=RECOVERED "
                    "totalTimeoutCount=1 recoveredTimeoutCount=1 "
                    "confirmedTimeoutCount=0"
                ),
            },
        ),
    )
    _observe(evaluator, _batch(1), 1, log_scan=timeout_log)

    result = evaluator.finalize()

    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is False
    assert result["temporalStaleLogSeen"] is False
    assert result["healthProbeTransportTimeoutLogSeen"] is True
    assert result["healthProbeTransportTimeoutExercised"] is True


def test_health_transport_timeout_counter_rollback_is_a_hard_failure(
    tmp_path: Path,
) -> None:
    log = tmp_path / "ros-cmd-vel-bridge.log"
    log.write_text(
        "SONIC_HEALTH_PROBE_TIMEOUT state=PENDING count=1/2 "
        "totalTimeoutCount=2 recoveredTimeoutCount=1 "
        "confirmedTimeoutCount=0\n",
        encoding="utf-8",
    )
    scanner = idle_soak.RuntimeLogScanner(
        tmp_path,
        log_names=("ros-cmd-vel-bridge.log",),
    )
    scanner.scan()
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            "SONIC_HEALTH_PROBE_TIMEOUT state=RECOVERED count=0/2 "
            "totalTimeoutCount=1 recoveredTimeoutCount=0 "
            "confirmedTimeoutCount=0\n"
        )

    rolled_back = scanner.scan()
    assert len(rolled_back.transport_timeout_counter_rollbacks) == 1

    evaluator = idle_soak.IdleSoakEvaluator()
    _observe(evaluator, _batch(0), 0)
    _observe(evaluator, _batch(1), 1, log_scan=rolled_back)
    result = evaluator.finalize()
    assert result["safeHoldSafetyPass"] is False
    assert result["movementReleaseAllowed"] is False
    assert {failure["code"] for failure in result["failures"]} >= {
        "HEALTH_PROBE_TIMEOUT_COUNTER_ROLLBACK"
    }


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class _FakeSampler:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def sample(self, *, include_robot_agent: bool) -> idle_soak.EndpointBatch:
        index = len(self.calls)
        self.calls.append(include_robot_agent)
        plant_delta = 1 if index >= 1 else 0
        batch = _batch(
            index,
            plant_stage="depthRender" if plant_delta else None,
            plant_delta=plant_delta,
        )
        if not include_robot_agent:
            batch.payloads.pop("robotAgent")
            batch.latency_ms.pop("robotAgent")
        return batch


class _WorkerDriftSampler:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def sample(self, *, include_robot_agent: bool) -> idle_soak.EndpointBatch:
        index = len(self.calls)
        self.calls.append(include_robot_agent)
        batch = _batch(index)
        if index == 1:
            batch.payloads["plant"]["observationWorker"].update(
                {
                    "workerPid": 301,
                    "workerBootId": "66666666-6666-4666-8666-666666666666",
                }
            )
        if not include_robot_agent:
            batch.payloads.pop("robotAgent")
            batch.latency_ms.pop("robotAgent")
        return batch


class _WorkerRenderStallSampler:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def sample(self, *, include_robot_agent: bool) -> idle_soak.EndpointBatch:
        index = len(self.calls)
        self.calls.append(include_robot_agent)
        batch = _batch(index)
        if index >= 1:
            batch.payloads["plant"]["observationWorker"]["renderTiming"] = (
                _worker_render_timing(
                    stage="totalMs",
                    count=1,
                    last_ms=250.0,
                )
            )
        if not include_robot_agent:
            batch.payloads.pop("robotAgent")
            batch.latency_ms.pop("robotAgent")
        return batch


class _FakeProcessProbe:
    def snapshot(self):
        return PROCESSES, ()


class _FakeLogScanner:
    def scan(self):
        return _empty_log()


class _MemoryWriter:
    def __init__(self) -> None:
        self.values: list[dict] = []

    def write(self, value):
        self.values.append(dict(value))


def test_short_run_preserves_rates_and_writes_raw_runtime_event() -> None:
    clock = _FakeClock()
    sampler = _FakeSampler()
    writer = _MemoryWriter()

    result = idle_soak.run_soak(
        sampler=sampler,
        evaluator=idle_soak.IdleSoakEvaluator(),
        writer=writer,
        process_probe=_FakeProcessProbe(),
        log_scanner=_FakeLogScanner(),
        duration_s=0.6,
        session_id="session-test",
        clock=clock,
        sleeper=clock.sleep,
    )

    assert sampler.calls == [True, False, True]
    assert result["safeHoldSafetyPass"] is True
    assert result["movementReleaseAllowed"] is False
    assert result["quarantineExercised"] is True
    kinds = [value["kind"] for value in writer.values]
    assert kinds[0] == "session_start"
    assert "baseline" in kinds
    assert "runtime_event" in kinds
    assert "final_checkpoint" in kinds
    assert kinds[-1] == "result"
    event = next(value for value in writer.values if value["kind"] == "runtime_event")
    assert set(event["payloads"]) == {"plant", "sonic"}
    assert event["derived"]["runtime"]["stallEvent"] is True
    for kind in ("baseline", "sample", "final_checkpoint"):
        record = next(value for value in writer.values if value["kind"] == kind)
        plant = (
            record["payloads"]["plant"]
            if kind != "sample"
            else record["plant"]
        )
        assert plant["observationWorker"]["workerPid"] == 201
        assert plant["viewerWorker"]["status"] == "disabled"
        assert record["derived"]["plantWorkers"]["failureChanges"] == {}


def test_short_run_writes_full_worker_drift_evidence() -> None:
    clock = _FakeClock()
    writer = _MemoryWriter()

    result = idle_soak.run_soak(
        sampler=_WorkerDriftSampler(),
        evaluator=idle_soak.IdleSoakEvaluator(),
        writer=writer,
        process_probe=_FakeProcessProbe(),
        log_scanner=_FakeLogScanner(),
        duration_s=0.6,
        session_id="session-worker-drift",
        clock=clock,
        sleeper=clock.sleep,
    )

    event = next(value for value in writer.values if value["kind"] == "worker_event")
    evidence = event["derived"]["plantWorkers"]["identityChanges"]
    assert evidence["observationWorker"]["baseline"]["workerPid"] == 201
    assert evidence["observationWorker"]["current"]["workerPid"] == 301
    assert event["payloads"]["plant"]["observationWorker"]["workerPid"] == 301
    assert result["safeHoldSafetyPass"] is False


def test_short_run_writes_full_worker_render_stall_evidence() -> None:
    clock = _FakeClock()
    writer = _MemoryWriter()

    result = idle_soak.run_soak(
        sampler=_WorkerRenderStallSampler(),
        evaluator=idle_soak.IdleSoakEvaluator(),
        writer=writer,
        process_probe=_FakeProcessProbe(),
        log_scanner=_FakeLogScanner(),
        duration_s=0.6,
        session_id="session-worker-render-stall",
        clock=clock,
        sleeper=clock.sleep,
    )

    event = next(value for value in writer.values if value["kind"] == "worker_event")
    timing = event["derived"]["plantWorkers"]["renderTiming"]
    path = (
        "plant.observationWorker.renderTiming.stages.totalMs."
        "overThresholdCount"
    )
    assert timing["counterChanges"][path] == {"previous": 0, "current": 1}
    assert timing["eventEvidence"]["stages"]["totalMs"]["lastMs"] == 250.0
    assert event["payloads"]["plant"]["observationWorker"]["renderTiming"][
        "stages"
    ]["totalMs"]["overThresholdCount"] == 1
    assert set(result["runtimeCounterDeltas"].values()) == {0}
    assert result["runtimeStallSeen"] is True
    assert result["movementReleaseAllowed"] is False


def test_documented_defaults_are_2hz_5s_10s_and_605s() -> None:
    assert idle_soak.SAMPLE_INTERVAL_S == 0.5
    assert idle_soak.ROBOT_AGENT_INTERVAL_S == 5.0
    assert idle_soak.RAW_CHECKPOINT_INTERVAL_S == 10.0
    assert idle_soak.DEFAULT_DURATION_S == 605.0
    assert "snapshotPublish" in idle_soak._PLANT_STAGES
    assert "viewerSnapshotPublish" in idle_soak._PLANT_STAGES
    assert idle_soak.EXPECTED_WORKER_STALL_THRESHOLD_MS == 100.0
    assert idle_soak.VIEWER_PROGRESS_TIMEOUT_NS == 2_000_000_000
    assert set(idle_soak._OBSERVATION_RENDER_TIMING_STAGES) == {
        "totalMs",
        "rgbRenderMs",
        "jpegEncodeMs",
        "depthRenderMs",
        "stateApplyMs",
        "mjForwardMs",
        "queueAgeAtStartMs",
    }
