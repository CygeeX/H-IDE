from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

_MONITOR_PATH = Path(__file__).parents[1] / "tools" / "motion_acceptance_monitor.py"
_SPEC = importlib.util.spec_from_file_location("motion_acceptance_monitor", _MONITOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MONITOR = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MONITOR
_SPEC.loader.exec_module(_MONITOR)
AcceptanceLimits = _MONITOR.AcceptanceLimits
MotionAcceptanceEvaluator = _MONITOR.MotionAcceptanceEvaluator
ReadonlyHttpSampler = _MONITOR.ReadonlyHttpSampler


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "motion_acceptance_cases.json"


def _payloads(sample: dict[str, object], index: int) -> tuple[dict, dict]:
    timestamp = 1_800_000_000.0 + float(sample["t"])
    yaw = float(sample["yaw"])
    state = str(sample["state"])
    safe_hold = state == "safe_hold"
    active = state in {"active", "fault"}
    faulted = state == "fault"
    sequence = int(sample.get("sequence", 100 + index))
    linear_speed = 0.15 if active else 0.0
    angular_speed = 0.20 if active else 0.0
    support_mode = "safety_stop" if safe_hold else ("locomotion" if active else "idle")
    arm_velocity = float(sample.get("arm_velocity", 1.2 if active else 0.05))
    fault_trace_count = int(sample.get("fault_trace_count", 0))
    last_error = sample.get("last_error")
    if safe_hold:
        last_error = "plant safety_stop requires explicit control.acquire rearm"
    plant = {
        "status": "running",
        "schemaVersion": "h-ide.mujoco-capture/1",
        "sequence": sequence,
        "capturedAtUnixNs": int(timestamp * 1_000_000_000),
        "pose": {
            "x": float(sample["x"]),
            "y": float(sample["y"]),
            "z": float(sample["z"]),
            "qw": math.cos(yaw / 2.0),
            "qx": 0.0,
            "qy": 0.0,
            "qz": math.sin(yaw / 2.0),
        },
        "rootVelocity": {
            "frameId": "base_link",
            "vx": linear_speed,
            "vy": 0.0,
            "vz": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": angular_speed,
            "linearSpeedMps": linear_speed,
            "angularSpeedRps": angular_speed,
        },
        "support": {
            "mode": support_mode,
            "updatedAt": timestamp,
            "fault": None,
            "safetyStopLatched": safe_hold,
            "safetyStopEpoch": 1,
            "lastRearmGeneration": None if safe_hold else 1,
        },
    }
    sonic = {
        "status": "running",
        "motion_ready": True,
        "updated_at": timestamp,
        "plant_tick": int(float(sample["t"]) * 1000) + 5000,
        "control_tick_ms": int(float(sample["t"]) * 1000) + 5000,
        "control_clock_status": "simulation_time_50hz_motion_ready",
        "nav2_state": "safety_stop" if safe_hold else ("commanding" if active else "idle"),
        "sonic_mode": "IDLE" if not active else "WALK",
        "sonic_speed_mps": linear_speed,
        "sonic_policy_speed_mps": linear_speed,
        "last_cmd_vx": float(sample.get("last_cmd_vx", linear_speed if active else 0.0)),
        "last_cmd_vy": 0.0,
        "last_cmd_wz": angular_speed if active else 0.0,
        "navigation_task_active": active,
        "static_hold_active": bool(sample.get("static_hold_active", not active)),
        "plant_support_safety_latched": safe_hold,
        "fault_trace_path": "/tmp/sonic-fault-trace.json" if fault_trace_count else None,
        "fault_trace_sample_count": fault_trace_count,
        "last_error": last_error,
        "navigation_arm_constrained_joints": 0,
        "navigation_arm_max_step_rad": 0.04,
        "navigation_arm_raw_max_deviation_rad": 1.0,
        "navigation_arm_raw_max_error_from_measured_rad": 0.20,
        "navigation_arm_measured_max_deviation_rad": 1.0,
        "navigation_arm_max_velocity_rps": arm_velocity,
        "navigation_arm_max_estimated_torque_nm": 8.0,
        "navigation_task_peak_arm_constrained_joints": 0,
        "navigation_task_peak_arm_raw_deviation_rad": 1.0,
        "navigation_task_peak_arm_tracking_error_rad": 0.20,
        "navigation_task_peak_arm_measured_deviation_rad": 1.0,
        "navigation_task_peak_arm_velocity_rps": arm_velocity,
        "navigation_task_peak_arm_estimated_torque_nm": 8.0,
    }
    if faulted:
        sonic["nav2_state"] = "fault_latched"
    return plant, sonic


@pytest.mark.parametrize("case", json.loads(FIXTURE_PATH.read_text(encoding="utf-8")), ids=lambda case: case["name"])
def test_offline_acceptance_fixtures(case: dict[str, object]) -> None:
    limits = AcceptanceLimits(
        phase=str(case["phase"]),
        expected_turn_rad=math.radians(float(case.get("expected_turn_deg", 90.0))),
    )
    evaluator = MotionAcceptanceEvaluator(limits)
    for index, sample in enumerate(case["samples"]):
        plant, sonic = _payloads(sample, index)
        evaluator.observe(plant, sonic, observed_s=float(sample["t"]))

    result = evaluator.finalize()

    assert result["pass"] is case["expected_pass"]
    actual_codes = {failure["code"] for failure in result["failures"]}
    assert actual_codes.issuperset(case.get("expected_failure_codes", []))


def test_nonfinite_source_value_fails_and_can_still_be_recorded_safely() -> None:
    plant, sonic = _payloads(
        {"t": 0.0, "state": "safe_hold", "x": 0.0, "y": 0.0, "z": 0.78, "yaw": 0.0},
        0,
    )
    sonic["navigation_arm_max_velocity_rps"] = float("nan")
    evaluator = MotionAcceptanceEvaluator(AcceptanceLimits(phase="idle"))

    evaluator.observe(plant, sonic, observed_s=0.0)
    result = evaluator.finalize()

    assert result["pass"] is False
    assert {failure["code"] for failure in result["failures"]}.issuperset(
        {"NONFINITE_TELEMETRY", "SCHEMA_INVALID"}
    )


class _Response:
    status = 200

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit: int) -> bytes:
        return self._body


def test_http_sampler_is_structurally_get_only_and_bodyless() -> None:
    requests = []
    bodies = iter((b'{"status":"running"}', b'{"status":"running"}'))

    def opener(request, *, timeout):
        requests.append((request, timeout))
        return _Response(next(bodies))

    sampler = ReadonlyHttpSampler(
        "http://127.0.0.1:8794/state",
        "http://127.0.0.1:8792/telemetry",
        opener=opener,
    )

    plant, sonic = sampler.sample()

    assert plant == sonic == {"status": "running"}
    assert len(requests) == 2
    assert all(request.method == "GET" for request, _timeout in requests)
    assert all(request.data is None for request, _timeout in requests)
    assert [request.full_url for request, _timeout in requests] == [
        "http://127.0.0.1:8794/state",
        "http://127.0.0.1:8792/telemetry",
    ]


@pytest.mark.parametrize(
    ("plant", "sonic"),
    [
        ("http://127.0.0.1:8794/support", "http://127.0.0.1:8792/telemetry"),
        ("http://127.0.0.1:8794/state", "http://127.0.0.1:8793/cmd_vel"),
        ("http://127.0.0.1:8794/state?write=true", "http://127.0.0.1:8792/telemetry"),
    ],
)
def test_http_sampler_rejects_every_non_readonly_path(plant: str, sonic: str) -> None:
    with pytest.raises(ValueError):
        ReadonlyHttpSampler(plant, sonic)
