from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
gate = importlib.import_module("startup_safe_hold_gate")

ROBOT_BOOT_ID = "11111111-1111-4111-8111-111111111111"
PLANT_BOOT_ID = "22222222-2222-4222-8222-222222222222"


def _triplet(
    index: int,
    *,
    plant_epoch: int = 7,
    sonic_generation: int = 11,
    robot_boot_id: str = ROBOT_BOOT_ID,
    robot_epoch: int = 12,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    timestamp = 1_800_000_000.0 + index * 0.5
    plant = {
        "status": "running",
        "schemaVersion": "h-ide.mujoco-capture/2",
        "robotId": "g1-sim",
        "sourceId": "mujoco:unitree-g1:g1-sim",
        "localizationEpochId": "g1-sim-map-epoch-1",
        "producerBootId": PLANT_BOOT_ID,
        "sequence": 100 + index,
        "capturedAtUnixNs": int(timestamp * 1_000_000_000),
        "pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.78,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        },
        "rootVelocity": {
            "frameId": "base_link",
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "wx": 0.0,
            "wy": 0.0,
            "wz": 0.0,
            "linearSpeedMps": 0.0,
            "angularSpeedRps": 0.0,
        },
        "support": {
            "mode": "safety_stop",
            "updatedAt": timestamp,
            "fault": None,
            "safetyStopLatched": True,
            "safetyStopEpoch": plant_epoch,
            "lastRearmGeneration": 10,
        },
    }
    sonic = {
        "backend": "mujoco_sonic",
        "status": "running",
        "robot_id": "g1-sim",
        "motion_ready": True,
        "updated_at": timestamp,
        "plant_tick": 5_000 + index * 500,
        "control_tick_ms": 5_000 + index * 500,
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
        "plant_support_stop_generation": sonic_generation,
        "fault_trace_path": None,
        "fault_trace_sample_count": 0,
        "last_error": None,
        "navigation_arm_constrained_joints": 0,
        "navigation_arm_max_step_rad": 0.0,
        "navigation_arm_raw_max_deviation_rad": 0.68,
        "navigation_arm_raw_max_error_from_measured_rad": 0.02,
        "navigation_arm_measured_max_deviation_rad": 0.70,
        "navigation_arm_max_velocity_rps": 0.0,
        "navigation_arm_max_estimated_torque_nm": 0.0,
        "navigation_task_peak_arm_constrained_joints": 0,
        "navigation_task_peak_arm_raw_deviation_rad": 0.0,
        "navigation_task_peak_arm_tracking_error_rad": 0.0,
        "navigation_task_peak_arm_measured_deviation_rad": 0.0,
        "navigation_task_peak_arm_velocity_rps": 0.0,
        "navigation_task_peak_arm_estimated_torque_nm": 0.0,
    }
    fault = {
        "required": True,
        "ready": True,
        "reason": None,
        "latched": False,
        "eventId": None,
        "source": None,
        "code": None,
        "counters": {
            "accepted": 0,
            "duplicate": 0,
            "invalid": 0,
            "sequenceConflict": 0,
            "staleSequence": 0,
            "wrongEpoch": 0,
            "wrongRobot": 0,
        },
    }
    checks = {
        name: {"required": True, "ready": True, "reason": None}
        for name in gate._REQUIRED_ROBOT_AGENT_CHECKS
    }
    checks["motionSafetyFault"] = fault
    robot_agent = {
        "schemaVersion": "robot-agent.diagnostics/1",
        "robotId": "g1-sim",
        "robotBootId": robot_boot_id,
        "observedAt": datetime.fromtimestamp(timestamp, UTC).isoformat(),
        "ready": True,
        "commandReadiness": {
            "ready": True,
            "mode": "STANDBY",
            "reason": None,
        },
        "controlState": {
            "state": "SAFE_HOLD",
            "active": False,
            "standby": True,
            "canAcquire": True,
            "authorityRevoked": True,
            "safeStopState": "confirmed",
            "safeStopEpoch": robot_epoch,
            "leaseValid": False,
            "leaseRemainingMs": None,
            "activeCommandCount": 0,
        },
        "motionSafetyFault": fault,
        "checks": checks,
    }
    return plant, sonic, robot_agent


def _sample(
    triplet: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    observed_s: float,
) -> gate.StartupSample:
    observed_wall_s = float(triplet[1]["updated_at"]) + 0.1
    return (*triplet, observed_s, observed_wall_s)


def test_two_coherent_groups_pass_the_complete_safe_hold_contract() -> None:
    first = _triplet(0)
    second = _triplet(1)

    result = gate.evaluate_startup_sequence(
        [_sample(first, 0.0), _sample(second, 0.5)]
    )

    assert result.ready is True
    assert result.healthy_settled_safe_hold is True
    assert result.motion_failure_codes == ()
    assert result.errors == ()
    assert result.stability_key == (7, 11, ROBOT_BOOT_ID, 12)


def test_one_group_is_never_enough() -> None:
    result = gate.evaluate_startup_sequence([_sample(_triplet(0), 0.0)])

    assert result.ready is False
    assert any("at least 2 consecutive" in error for error in result.errors)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda _plant, sonic, _agent: sonic.__setitem__("last_cmd_vx", 0.001),
        lambda _plant, _sonic, agent: agent.__setitem__("ready", False),
        lambda _plant, _sonic, agent: agent["commandReadiness"].__setitem__(
            "mode", "ACTIVE"
        ),
        lambda _plant, _sonic, agent: agent["controlState"].__setitem__(
            "authorityRevoked", False
        ),
        lambda _plant, _sonic, agent: agent["controlState"].__setitem__(
            "leaseRemainingMs", 1.0
        ),
        lambda _plant, _sonic, agent: agent["motionSafetyFault"].__setitem__(
            "latched", True
        ),
        lambda _plant, _sonic, agent: agent["checks"]["control"].update(
            required=False,
            ready=False,
        ),
        lambda _plant, _sonic, agent: agent["checks"].__setitem__(
            "futureRequiredCheck", {"required": True, "ready": False}
        ),
        lambda _plant, _sonic, agent: agent.__setitem__("robotId", "other-robot"),
    ],
    ids=(
        "sonic-nonzero",
        "agent-not-ready",
        "agent-not-standby",
        "authority-not-revoked",
        "lease-present",
        "fault-latched",
        "known-check-downgraded",
        "future-required-check-not-ready",
        "crossed-robot-id",
    ),
)
def test_single_snapshot_fails_closed_on_contract_breaks(mutate: Any) -> None:
    plant, sonic, robot_agent = _triplet(0)
    mutate(plant, sonic, robot_agent)

    result = gate.evaluate_startup_snapshot(
        plant,
        sonic,
        robot_agent,
        observed_wall_s=float(sonic["updated_at"]) + 0.1,
    )

    assert result.ready is False
    assert result.errors


def test_sequence_rejects_stability_change_without_cross_domain_epoch_equality() -> None:
    first = _triplet(0, plant_epoch=7, sonic_generation=11, robot_epoch=12)
    second = _triplet(1, plant_epoch=8, sonic_generation=11, robot_epoch=12)

    result = gate.evaluate_startup_sequence(
        [_sample(first, 0.0), _sample(second, 0.5)]
    )

    assert result.ready is False
    assert any("stability identity changed" in error for error in result.errors)


def test_sequence_rejects_two_identical_cached_groups() -> None:
    cached = _triplet(0)

    result = gate.evaluate_startup_sequence(
        [_sample(cached, 0.0), _sample(deepcopy(cached), 0.5)]
    )

    assert result.ready is False
    assert any("did not strictly advance" in error for error in result.errors)


@pytest.mark.parametrize(
    "marker",
    [
        "plant.sequence",
        "plant.capturedAtUnixNs",
        "sonic.updated_at",
        "sonic.plant_tick",
        "sonic.control_tick_ms",
        "robotAgent.observedAt",
    ],
)
def test_each_source_progress_marker_must_strictly_advance(marker: str) -> None:
    first = _triplet(0)
    second = _triplet(1)
    owner, field = marker.split(".", 1)
    mappings = {
        "plant": (first[0], second[0]),
        "sonic": (first[1], second[1]),
        "robotAgent": (first[2], second[2]),
    }
    first_mapping, second_mapping = mappings[owner]
    second_mapping[field] = first_mapping[field]

    result = gate.evaluate_startup_sequence(
        [_sample(first, 0.0), _sample(second, 0.5)]
    )

    assert result.ready is False
    assert any(
        f"{marker} did not strictly advance" in error
        for error in result.errors
    )


def test_sequence_requires_a_stable_plant_producer_boot() -> None:
    first = _triplet(0)
    second = _triplet(1)
    second[0]["producerBootId"] = "33333333-3333-4333-8333-333333333333"

    result = gate.evaluate_startup_sequence(
        [_sample(first, 0.0), _sample(second, 0.5)]
    )

    assert result.ready is False
    assert "Plant producerBootId changed between startup samples" in result.errors


def test_snapshot_requires_each_source_to_be_fresh() -> None:
    plant, sonic, robot_agent = _triplet(0)

    result = gate.evaluate_startup_snapshot(
        plant,
        sonic,
        robot_agent,
        observed_wall_s=float(sonic["updated_at"]) + gate.MAX_SOURCE_AGE_S + 0.1,
    )

    assert result.ready is False
    assert sum("is stale by" in error for error in result.errors) == 3


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0
        self.wall_base = 1_800_000_000.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration_s: float) -> None:
        self.value += duration_s

    def wall(self) -> float:
        return self.wall_base + self.value


class _SequenceSampler:
    def __init__(self, samples: list[Any]) -> None:
        self.samples = list(samples)
        self.index = 0

    def sample(self, **_kwargs: Any) -> tuple[dict, dict, dict]:
        value = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


def test_gate_requires_two_consecutive_groups_with_the_same_tuple() -> None:
    first = _triplet(0, plant_epoch=7)
    changed = _triplet(1, plant_epoch=8)
    stable = _triplet(2, plant_epoch=8)
    clock = _Clock()

    result = gate.wait_for_startup_safe_hold(
        sampler=_SequenceSampler([first, changed, stable]),
        deadline_s=2.0,
        interval_s=0.1,
        clock=clock,
        wall_clock=clock.wall,
        sleeper=clock.sleep,
    )

    assert result.passed is True
    assert result.attempts == 3
    assert result.consecutive_samples == 2
    assert result.stability_key == (8, 11, ROBOT_BOOT_ID, 12)


def test_endpoint_failure_clears_the_candidate_streak() -> None:
    first = _triplet(0)
    second = _triplet(1)
    third = _triplet(2)
    clock = _Clock()

    result = gate.wait_for_startup_safe_hold(
        sampler=_SequenceSampler([first, OSError("offline"), second, third]),
        deadline_s=2.0,
        interval_s=0.1,
        clock=clock,
        wall_clock=clock.wall,
        sleeper=clock.sleep,
    )

    assert result.passed is True
    assert result.attempts == 4


class _FailingSampler:
    def sample(self, **_kwargs: Any) -> tuple[dict, dict, dict]:
        raise TimeoutError("still unavailable")


def test_gate_uses_one_bounded_deadline() -> None:
    clock = _Clock()

    result = gate.wait_for_startup_safe_hold(
        sampler=_FailingSampler(),
        deadline_s=1.0,
        interval_s=0.3,
        clock=clock,
        wall_clock=clock.wall,
        sleeper=clock.sleep,
    )

    assert result.passed is False
    assert result.elapsed_s == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must not exceed 30s"):
        gate.wait_for_startup_safe_hold(
            sampler=_FailingSampler(),
            deadline_s=30.01,
            clock=clock,
            wall_clock=clock.wall,
            sleeper=clock.sleep,
        )


class _Response:
    def __init__(self, payload: Any, *, final_url: str) -> None:
        self.status = 200
        self.payload = payload
        self.final_url = final_url

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.final_url

    def read(self, _limit: int) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_sampler_uses_only_the_three_exact_get_resources() -> None:
    requests: list[tuple[str, str, Any, float]] = []

    def open_response(request: Any, *, timeout: float) -> _Response:
        requests.append(
            (request.full_url, request.get_method(), request.data, timeout)
        )
        return _Response({}, final_url=request.full_url)

    sampler = gate.ReadonlyStartupSampler(timeout_s=1.0, opener=open_response)
    sampler.sample(deadline_s=5.0, clock=lambda: 0.0)

    assert [item[0] for item in requests] == list(gate.ENDPOINTS)
    assert {item[1] for item in requests} == {"GET"}
    assert {item[2] for item in requests} == {None}
    assert {item[3] for item in requests} == {1.0}


def test_sampler_rejects_a_redirected_final_url() -> None:
    sampler = gate.ReadonlyStartupSampler(
        opener=lambda request, **_kwargs: _Response(
            {},
            final_url="http://example.invalid/redirected",
        )
    )

    with pytest.raises(RuntimeError, match="redirected to forbidden target"):
        sampler.sample(deadline_s=5.0, clock=lambda: 0.0)


def test_default_sampler_disables_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://example.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:8080")

    sampler = gate.ReadonlyStartupSampler()
    opener = sampler._opener.__self__
    proxy_handlers = [
        handler
        for handler in opener.handlers
        if isinstance(handler, gate.ProxyHandler)
    ]

    assert proxy_handlers == []


def test_cli_reports_a_successful_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = gate.StartupGateResult(
        passed=True,
        attempts=2,
        consecutive_samples=2,
        elapsed_s=0.5,
        stability_key=(7, 11, ROBOT_BOOT_ID, 12),
        errors=(),
    )
    monkeypatch.setattr(gate, "wait_for_startup_safe_hold", lambda **_kwargs: expected)

    assert gate.main(["--deadline-seconds", "30"]) == 0


def test_launcher_orders_the_semantic_gate_after_agent_health_before_status() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    agent_health = launcher.index("http://127.0.0.1:8796/healthz")
    semantic_gate = launcher.index("tools/startup_safe_hold_gate.py")
    final_status = launcher.index("scripts/workstation-demo-status.sh")
    assert agent_health < semantic_gate < final_status
    assert "--deadline-seconds 30" in launcher
    assert "support_safety_clue = (" in launcher
    assert "if support_safety_clue" in launcher


def test_launcher_rejects_partial_motion_topology_before_startup_work() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    topology_gate = launcher.index(
        "Refusing to reuse a partial MuJoCo/SONIC/Nav2 motion topology"
    )
    model_lock = launcher.index("tools/verify_model_lock.py")
    plant_start = launcher.index("start_process \\\n  mujoco-plant")
    assert topology_gate < model_lock < plant_start
    for unit in (
        "mujoco-plant",
        "sonic-driver",
        "ros-cmd-vel-bridge",
        "ros-observation-bridge",
        "ros-safe-stop",
        "strict-navigation-adapter",
        "sim-robot-agent-gateway",
        "observation-gateway",
        "robot-agent-g1-sim",
        "robot-agent-diagnostics-proxy",
    ):
        assert f"  {unit}\n" in launcher
    assert "HIDE_DEMO_NAV2_ACTIVE" in launcher
    assert "Refusing to reuse a first-person viewer without its owning MuJoCo Plant" in launcher


def test_launcher_has_a_cold_isolated_motion_core_mode() -> None:
    launcher_path = ROOT / "scripts" / "start-workstation-demo.sh"
    launcher = launcher_path.read_text(encoding="utf-8")

    parser = launcher.index("--motion-core)")
    viewer_requirement = launcher.index("--motion-core requires --viewer")
    isolation_gate = launcher.index("require_motion_core_isolation()")
    model_lock = launcher.index("tools/verify_model_lock.py")
    nav2_ready = launcher.index("if ! wait_for_nav2_active 60")
    isolated_exit = launcher.index("Motion-core is stationary and isolated")
    robot_agent_start = launcher.index("start_process \\\n  robot-agent-g1-sim")
    assert parser < viewer_requirement < isolation_gate < model_lock
    assert nav2_ready < isolated_exit < robot_agent_start
    assert 'for port in (8796, 8094):' in launcher
    assert "HIDE_DEMO_MOTION_CORE_ISOLATED_UNITS=(" in launcher
    assert "No motion command has been sent" in launcher

    completed = subprocess.run(
        ["bash", str(launcher_path), "--motion-core"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "--motion-core requires --viewer" in completed.stderr


def test_full_chain_starts_strict_navigation_before_any_remote_motion_owner() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    nav2_ready = launcher.index("if ! wait_for_nav2_active 60")
    motion_core_exit = launcher.index("Motion-core is stationary and isolated")
    adapter_start = launcher.index(
        "start_process \\\n  strict-navigation-adapter"
    )
    adapter_gate = launcher.index("if ! wait_for_strict_navigation_adapter 30")
    gateway_start = launcher.index("start_process \\\n  sim-robot-agent-gateway")
    robot_agent_start = launcher.index("start_process \\\n  robot-agent-g1-sim")

    assert nav2_ready < motion_core_exit < adapter_start < adapter_gate
    assert adapter_gate < gateway_start < robot_agent_start
    assert "/h_ide/strict/navigate_to_pose" in launcher
    assert "/h_ide/strict/spin" in launcher
    assert "tools/strict_navigation_adapter.py" in launcher

def test_launcher_secures_runtime_evidence_directory_before_model_lock() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    symlink_gate = launcher.index(
        "runtime directory must not be a symbolic link"
    )
    owner_gate = launcher.index("runtime directory is not an owned directory")
    chmod = launcher.index('chmod 0700 -- "${HIDE_DEMO_RUN_ROOT}"')
    mode_gate = launcher.index("runtime directory permissions are not 0700")
    model_lock = launcher.index("tools/verify_model_lock.py")
    assert symlink_gate < owner_gate < chmod < mode_gate < model_lock


def test_launcher_requires_mapped_fresh_first_person_frames_before_ros_ingress() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    first_person_start = launcher.index("start_process \\\n    first-person-viewer")
    first_person_gate = launcher.index(
        "wait_for_first_person_viewer", first_person_start
    )
    ros_ingress = launcher.index("start_process \\\n  ros-cmd-vel-bridge")
    assert first_person_start < first_person_gate < ros_ingress
    gate_source = launcher[
        launcher.index("wait_for_first_person_viewer()") : first_person_start
    ]
    for evidence in (
        'payload.get("status") == "running"',
        'payload.get("healthy") is True',
        'payload.get("mapped") is True',
        'payload.get("plantSessionId")',
        'payload["sequence"] >= 2',
        'payload["plantSequence"] > 0',
        '0.0 <= success_age_s <= 2.0',
    ):
        assert evidence in gate_source


def test_launcher_requires_healthy_motion_services_before_tf_and_nav2() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    safe_stop_start = launcher.index("start_process \\\n  ros-safe-stop")
    ingress_gate = launcher.index("wait_for_ros_motion_ingress 30")
    tf_gate = launcher.index("wait_for_simulation_tf 60")
    nav2_start = launcher.index("docker start h-ide-nav2-sim")
    assert safe_stop_start < ingress_gate < tf_gate < nav2_start

    function_start = launcher.index("wait_for_ros_motion_ingress()")
    function_end = launcher.index("wait_for_simulation_tf()", function_start)
    gate_source = launcher[function_start:function_end]
    for unit in (
        "ros-cmd-vel-bridge",
        "ros-observation-bridge",
        "ros-safe-stop",
    ):
        assert unit in gate_source
    for service in (
        "/h_ide/mujoco/bridge_health",
        "/h_ide/mujoco/normal_stop",
        "/robot/control_arm",
        "/robot/safe_stop",
    ):
        assert service in gate_source
    assert 'payload.get("healthy") is True' in gate_source
    assert 'payload.get("fault") is None' in gate_source


def test_pre_agent_gate_cannot_use_empty_last_error_to_bypass_a_support_latch() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )
    predicate_start = launcher.index("base_ready = (")
    predicate = launcher[predicate_start : launcher.index(
        "raise SystemExit(0 if ready else 1)",
        predicate_start,
    )]

    assert 'value.get("plant_support_safety_latched") is True' in predicate
    assert 'value.get("plant_support_stop_generation") is not None' in predicate
    assert "settled_safe_hold\n    if support_safety_clue" in predicate
    assert "last_error in {None, \"\"} or settled_safe_hold" not in predicate


def _run_pre_agent_predicate(telemetry: dict[str, Any]) -> int:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )
    function_start = launcher.index("wait_for_sonic_motion_ready()")
    source_start = launcher.index("import json", function_start)
    source_end = launcher.index(
        "raise SystemExit(0 if ready else 1)",
        source_start,
    ) + len("raise SystemExit(0 if ready else 1)")
    completed = subprocess.run(
        [sys.executable, "-c", launcher[source_start:source_end]],
        input=json.dumps(telemetry),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode


def test_pre_agent_gate_rejects_active_locomotion_with_no_last_error() -> None:
    _plant, telemetry, _agent = _triplet(0)
    telemetry.update(
        nav2_state="commanding",
        sonic_mode="WALK",
        sonic_speed_mps=0.2,
        sonic_policy_speed_mps=0.2,
        static_hold_active=False,
        motion_handoff_active=True,
        motion_handoff_ratio=1.0,
        navigation_task_active=True,
        last_cmd_vx=0.2,
        plant_support_mode="locomotion",
        plant_support_requested_mode="locomotion",
        plant_support_confirmed_mode="locomotion",
        plant_support_safety_latched=False,
        plant_support_stop_generation=None,
        last_error=None,
    )

    assert _run_pre_agent_predicate(telemetry) == 1


def test_pre_agent_gate_accepts_only_an_exact_settled_idle_without_clues() -> None:
    _plant, telemetry, _agent = _triplet(0)
    telemetry.update(
        nav2_state="idle",
        plant_support_mode="idle",
        plant_support_requested_mode="idle",
        plant_support_confirmed_mode="idle",
        plant_support_safety_latched=False,
        plant_support_stop_generation=None,
        last_error=None,
    )

    assert _run_pre_agent_predicate(telemetry) == 0


def test_pre_agent_gate_requires_complete_structure_for_a_support_latch() -> None:
    _plant, telemetry, _agent = _triplet(0)
    assert telemetry["last_error"] is None

    assert _run_pre_agent_predicate(telemetry) == 0

    telemetry["nav2_state"] = "idle"
    assert _run_pre_agent_predicate(telemetry) == 1


def test_launcher_requires_opt_in_for_uninterrupted_sonic_experiment() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT="${HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT:-0}"'
        in launcher
    )
    assert (
        "HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT must be exactly 0 or 1"
        in launcher
    )
    assert (
        '--setenv="HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT=${HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT}"'
        in launcher
    )
