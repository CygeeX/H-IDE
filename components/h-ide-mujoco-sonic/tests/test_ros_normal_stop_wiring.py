from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from uuid import UUID

from hide_mujoco_sonic.gait_adapter import GaitState, SonicIntent
from hide_mujoco_sonic.motion_fault import SonicHealth
from hide_mujoco_sonic.ros_bridge import (
    _ActionStatusObservation,
    _actions_confirmed_inactive_after,
    _create_safety_timers,
    _global_goal_progress_active,
    _resident_policy_reentry_generation,
    _suppress_for_correlated_stop,
    BridgeRuntimeHealth,
)


ROOT = Path(__file__).resolve().parents[1]


def ready_sonic_health() -> SonicHealth:
    return SonicHealth(
        request_id=UUID("11000000-0000-4000-8000-000000000001"),
        robot_id="g1-sim",
        driver_boot_id=UUID("21000000-0000-4000-8000-000000000002"),
        driver_session_id=UUID("31000000-0000-4000-8000-000000000003"),
        status="ready",
        terminal_fault=False,
        reason=None,
        motion_ready=True,
        last_lowstate_age_s=0.01,
        last_control_step_age_s=0.01,
        plant_support_mode="safety_stop",
        plant_support_safety_latched=True,
        plant_support_stop_pending=False,
        static_hold_active=True,
        navigation_task_active=False,
        motion_authorized=False,
        nav2_command_zero=True,
        matched_reference_actuator_authority=False,
        reference_authority_state="REVOKED",
        reference_authority_generation=None,
        reference_authority_source_sequence=None,
        reference_authority_intent_generation=None,
        reference_authority_epoch=0,
        reference_authority_granted_at_monotonic_s=None,
        reference_authority_granted_tick_ms=None,
        reference_authority_dds_target_sequence=None,
    )


def balanced_policy_idle_health() -> SonicHealth:
    return replace(
        ready_sonic_health(),
        plant_support_mode="idle",
        plant_support_safety_latched=False,
        static_hold_active=False,
        motion_authorized=True,
        controlled_policy_stop_active=True,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=100.25,
        reference_authority_granted_tick_ms=2500,
        reference_authority_dds_target_sequence=11,
    )


def test_resident_balanced_policy_reentry_reuses_only_exact_generation() -> None:
    health = balanced_policy_idle_health()

    assert (
        _resident_policy_reentry_generation(
            health=health,
            gait_state=GaitState.IDLE,
            bridge_generation=42,
            gait_generation=42,
            observation_generation=42,
        )
        == 42
    )

    for field in (
        "bridge_generation",
        "gait_generation",
        "observation_generation",
    ):
        values = {
            "bridge_generation": 42,
            "gait_generation": 42,
            "observation_generation": 42,
        }
        values[field] = 43
        try:
            _resident_policy_reentry_generation(
                health=health,
                gait_state=GaitState.IDLE,
                **values,
            )
        except ValueError as exc:
            assert "generation identity mismatch" in str(exc)
        else:
            raise AssertionError(f"{field} mismatch must fail closed")


def test_physical_safe_hold_still_uses_new_generation_rearm_path() -> None:
    assert (
        _resident_policy_reentry_generation(
            health=ready_sonic_health(),
            gait_state=GaitState.IDLE,
            bridge_generation=42,
            gait_generation=42,
            observation_generation=42,
        )
        is None
    )


def bridge_health_snapshot(
    evidence: BridgeRuntimeHealth,
    *,
    now_s: float,
    pending: bool = False,
    confirmed: bool = False,
    health_pending: bool = False,
    quarantined: bool = False,
    fault: str | None = None,
) -> dict[str, object]:
    return evidence.snapshot(
        gait_state=GaitState.IDLE,
        control_generation=19,
        bridge_fault=fault,
        health_transport_pending=pending,
        health_transport_confirmed=confirmed,
        health_fault_pending=health_pending,
        health_quarantined=quarantined,
        now_s=now_s,
    )


def test_bridge_health_query_never_manufactures_progress() -> None:
    evidence = BridgeRuntimeHealth(
        boot_id=UUID("41000000-0000-4000-8000-000000000004"),
        session_id=UUID("51000000-0000-4000-8000-000000000005"),
    )

    empty = bridge_health_snapshot(evidence, now_s=10.0)
    assert empty["healthy"] is False
    assert empty["sequence"] == 0
    assert empty["lastProgressAtMonotonicS"] is None
    assert bridge_health_snapshot(evidence, now_s=10.0)["sequence"] == 0

    evidence.observe_sonic_probe(ready_sonic_health(), observed_at_s=10.0)
    first = bridge_health_snapshot(evidence, now_s=10.05)
    replay = bridge_health_snapshot(evidence, now_s=10.06)
    assert first["healthy"] is True
    assert first["sequence"] == 1
    assert replay["sequence"] == 1
    assert first["sonicDriverBootId"] == "21000000-0000-4000-8000-000000000002"
    assert first["sonicDriverSessionId"] == "31000000-0000-4000-8000-000000000003"


def test_bridge_health_fails_closed_for_every_fence_and_stale_progress() -> None:
    evidence = BridgeRuntimeHealth(
        boot_id=UUID("41000000-0000-4000-8000-000000000004"),
        session_id=UUID("51000000-0000-4000-8000-000000000005"),
    )
    evidence.observe_sonic_probe(ready_sonic_health(), observed_at_s=10.0)

    assert bridge_health_snapshot(evidence, now_s=10.100001)["healthy"] is False
    assert bridge_health_snapshot(evidence, now_s=10.01, pending=True)["healthy"] is False
    assert bridge_health_snapshot(evidence, now_s=10.01, confirmed=True)["healthy"] is False
    assert bridge_health_snapshot(evidence, now_s=10.01, health_pending=True)["healthy"] is False
    assert bridge_health_snapshot(evidence, now_s=10.01, quarantined=True)["healthy"] is False
    assert bridge_health_snapshot(evidence, now_s=10.01, fault="latched")["healthy"] is False

    evidence.observe_sonic_probe(ready_sonic_health(), observed_at_s=10.2)
    recovered = bridge_health_snapshot(evidence, now_s=10.2)
    assert recovered["healthy"] is True
    assert recovered["sequence"] == 2


def gait_intent(*, state: GaitState) -> SonicIntent:
    return SonicIntent(
        sequence=2,
        generation=10,
        generated_at_s=1.0,
        state=state.value,
        mode="IDLE",
        velocity_x_mps=0.0,
        velocity_y_mps=0.0,
        linear_speed_mps=0.0,
        movement_angle_rad=0.0,
        yaw_rate_rps=0.0,
        facing_delta_rad=0.0,
        motion_enabled=False,
        stop_requested=True,
        stop_acknowledged=False,
        head_yaw_rad=0.0,
        head_pitch_rad=0.0,
        source="nav2",
        reason="test",
    )


class BlockingCoordinator:
    def blocks_generic_intent(self, generation: int | None) -> bool:
        return generation == 10


class RecordingTimerNode:
    def __init__(self) -> None:
        self.calls: list[tuple[float, object, object, object]] = []

    def create_timer(
        self,
        timer_period_sec: float,
        callback,
        *,
        callback_group: object,
        clock: object,
    ) -> object:
        self.calls.append(
            (timer_period_sec, callback, callback_group, clock)
        )
        return object()


def test_active_normal_stop_suppresses_generic_envelopes_but_never_fault() -> None:
    coordinator = BlockingCoordinator()

    assert _suppress_for_correlated_stop(
        coordinator,  # type: ignore[arg-type]
        gait_intent(state=GaitState.STOPPING),
    ) is True
    assert _suppress_for_correlated_stop(
        coordinator,  # type: ignore[arg-type]
        gait_intent(state=GaitState.DISARMED),
    ) is True
    assert _suppress_for_correlated_stop(
        coordinator,  # type: ignore[arg-type]
        gait_intent(state=GaitState.FAULT),
    ) is False


def test_safety_timers_share_the_supplied_clock_and_state_group() -> None:
    node = RecordingTimerNode()
    state_group = object()
    steady_clock = object()

    def watchdog() -> None:
        pass

    def health_probe() -> None:
        pass

    timers = _create_safety_timers(
        node,
        command_timeout_s=0.25,
        health_probe_period_s=0.025,
        watchdog=watchdog,
        health_probe=health_probe,
        callback_group=state_group,
        clock=steady_clock,
    )

    assert len(timers) == 2
    assert node.calls == [
        (0.1, watchdog, state_group, steady_clock),
        (0.025, health_probe, state_group, steady_clock),
    ]


def test_motion_fault_reset_requires_post_fault_inactive_action_evidence() -> None:
    statuses = {
        "navigate_to_pose": None,
        "spin": None,
        "drive_on_heading": None,
    }
    assert not _actions_confirmed_inactive_after(
        statuses,
        fault_latched_at_s=10.0,
    )

    statuses["navigate_to_pose"] = _ActionStatusObservation(False, 10.1)
    statuses["spin"] = _ActionStatusObservation(False, 9.9)
    assert not _actions_confirmed_inactive_after(
        statuses,
        fault_latched_at_s=10.0,
    )

    statuses["spin"] = _ActionStatusObservation(True, 10.2)
    assert not _actions_confirmed_inactive_after(
        statuses,
        fault_latched_at_s=10.0,
    )

    statuses["spin"] = _ActionStatusObservation(False, 10.2)
    assert not _actions_confirmed_inactive_after(
        statuses,
        fault_latched_at_s=10.0,
    )

    statuses["drive_on_heading"] = _ActionStatusObservation(False, 10.2)
    assert _actions_confirmed_inactive_after(
        statuses,
        fault_latched_at_s=10.0,
    )


def test_global_goal_progress_belongs_only_to_navigate_to_pose() -> None:
    statuses = {
        "navigate_to_pose": None,
        "spin": None,
        "drive_on_heading": None,
    }
    assert _global_goal_progress_active(statuses) is False

    statuses["navigate_to_pose"] = _ActionStatusObservation(True, 10.0)
    assert _global_goal_progress_active(statuses) is True

    statuses["drive_on_heading"] = _ActionStatusObservation(True, 10.1)
    assert _global_goal_progress_active(statuses) is False

    statuses["drive_on_heading"] = _ActionStatusObservation(False, 10.2)
    statuses["spin"] = _ActionStatusObservation(True, 10.3)
    assert _global_goal_progress_active(statuses) is False

    statuses["spin"] = _ActionStatusObservation(False, 10.4)
    statuses["navigate_to_pose"] = _ActionStatusObservation(False, 10.5)
    assert _global_goal_progress_active(statuses) is False


def test_ros_command_wires_goal_progress_to_action_ownership() -> None:
    path = ROOT / "src/hide_mujoco_sonic/ros_bridge.py"
    command = ast.unparse(_nested_function(path, "on_cmd"))
    status = ast.unparse(_nested_function(path, "on_action_status"))

    assert (
        "tracking_enabled=_global_goal_progress_active(action_statuses)" in command
    )
    assert "goal_guard.suspend_tracking()" in status


def _keyword_name(call: ast.Call, keyword: str) -> str | None:
    for item in call.keywords:
        if item.arg == keyword and isinstance(item.value, ast.Name):
            return item.value.id
    return None


def _calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


def _nested_function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def test_ros_bridge_assigns_every_callback_entity_and_uses_two_threads() -> None:
    calls = _calls(ROOT / "src/hide_mujoco_sonic/ros_bridge.py")
    entity_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr
        in {
            "create_subscription",
            "create_service",
            "create_timer",
            "create_guard_condition",
        }
    ]
    assert entity_calls

    subscriptions = [
        call
        for call in entity_calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_subscription"
    ]
    timers = [
        call
        for call in entity_calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_timer"
    ]
    guards = [
        call
        for call in entity_calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_guard_condition"
    ]
    assert len(subscriptions) == 7
    assert len(timers) == 2
    assert len(guards) == 1
    assert all(
        _keyword_name(call, "callback_group") == "state_group"
        for call in subscriptions + guards
    )
    assert all(
        _keyword_name(call, "callback_group") == "callback_group"
        and _keyword_name(call, "clock") == "clock"
        for call in timers
    )

    timer_installers = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "_create_safety_timers"
    ]
    assert len(timer_installers) == 1
    assert _keyword_name(timer_installers[0], "callback_group") == "state_group"
    assert _keyword_name(timer_installers[0], "clock") == "safety_timer_clock"

    clock_constructors = [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "Clock"
    ]
    assert len(clock_constructors) == 1
    assert any(
        keyword.arg == "clock_type"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "ClockType"
        and keyword.value.attr == "STEADY_TIME"
        for keyword in clock_constructors[0].keywords
    )

    services = {
        call.args[1].value: _keyword_name(call, "callback_group")
        for call in entity_calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_service"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
    }
    assert services == {
        "/h_ide/mujoco/bridge_health": "state_group",
        "/robot/control_arm": "state_group",
        "/h_ide/mujoco/reset_motion_fault": "state_group",
        "/h_ide/mujoco/normal_stop": "normal_stop_group",
    }

    executors = [
        call
        for call in calls
        if isinstance(call.func, ast.Name)
        and call.func.id == "MultiThreadedExecutor"
    ]
    assert len(executors) == 1
    assert any(
        keyword.arg == "num_threads"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == 2
        for keyword in executors[0].keywords
    )
    assert not any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "rclpy"
        and call.func.attr == "spin"
        for call in calls
    )


def test_health_transport_fence_precedes_every_control_arm_side_effect() -> None:
    path = ROOT / "src/hide_mujoco_sonic/ros_bridge.py"
    source = ast.unparse(_nested_function(path, "perform_control_arm"))

    fence = source.index("check_health_transport_fence")
    probe = source.index("health_request_id = uuid4()")
    generation = source.index("generation['value'] = next_generation")
    arm = source.index("gait.arm(next_generation")
    driver_io = source.index("acknowledgement = request_json")

    assert fence < probe < generation < arm < driver_io


def test_motion_fault_stop_binds_sonic_request_to_fault_event_identity() -> None:
    path = ROOT / "src/hide_mujoco_sonic/ros_bridge.py"
    source = ast.unparse(_nested_function(path, "latch_stop"))

    assert "event = fault_latch.event" in source
    assert "'request_id': str(event.event_id)" in source
    assert source.index("event = fault_latch.event") < source.index(
        "'type': 'motion_fault_latch'"
    )


def test_health_transport_fence_wires_command_auto_arm_and_delivery_paths() -> None:
    path = ROOT / "src/hide_mujoco_sonic/ros_bridge.py"
    command = ast.unparse(_nested_function(path, "on_cmd"))
    odometry = ast.unparse(_nested_function(path, "on_odom"))
    delivery = ast.unparse(_nested_function(path, "deliver_intent"))
    watchdog = ast.unparse(_nested_function(path, "watchdog"))

    assert command.index("check_health_transport_fence") < command.index(
        "gait.ingest_twist"
    )
    assert "_health_transport_command_action" in command
    assert "transport_action == 'cancel_start'" in command
    assert command.index("gait.cancel") < command.index("gait.ingest_twist")
    assert "not transport_fenced" in odometry
    assert "gait.state is GaitState.DISARMED" in odometry
    assert odometry.index("not transport_fenced") < odometry.index("gait.arm")
    assert "_suppress_for_unknown_health_transport" in delivery
    assert delivery.index("_suppress_for_unknown_health_transport") < delivery.index(
        "send_intent(intent)"
    )
    assert "deliver_intent(intent, context='gait watchdog')" in watchdog


def test_odometry_intent_uses_the_same_sample_yaw_not_the_pose_callback_cache() -> None:
    """A queued map-pose callback must not freeze SONIC's Spin heading feedback."""

    path = ROOT / "src/hide_mujoco_sonic/ros_bridge.py"
    odometry = ast.unparse(_nested_function(path, "on_odom"))

    assert "latest_heading['yaw'] = yaw" in odometry
    assert "measured_heading_yaw=yaw" in odometry
    assert "measured_heading_yaw=latest_heading['yaw']" not in odometry
    assert odometry.index("latest_heading['yaw'] = yaw") < odometry.index(
        "gait.ingest_observation"
    )


def test_normal_stop_timeout_logs_and_reraises_without_mutating_transport_gate() -> None:
    path = ROOT / "src/hide_mujoco_sonic/ros_bridge.py"
    source = ast.unparse(_nested_function(path, "probe_normal_stop_health"))

    assert "except JsonResponseTimeoutError" in source
    assert "SONIC_HEALTH_PROBE_TIMEOUT state=NORMAL_STOP" in source
    assert source.rstrip().endswith("raise")
    assert "health_transport_gate" not in source
    assert "observe_timeout" not in source
    assert "observe_strict_success" not in source


def test_safe_stop_proxy_uses_separate_groups_and_two_thread_executor() -> None:
    calls = _calls(ROOT / "src/hide_mujoco_sonic/safe_stop_service.py")
    clients = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_client"
    ]
    services = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "create_service"
    ]
    assert len(clients) == 1
    assert len(services) == 1
    assert _keyword_name(clients[0], "callback_group") == "client_group"
    assert _keyword_name(services[0], "callback_group") == "inbound_group"
    assert any(
        isinstance(call.func, ast.Name)
        and call.func.id == "MultiThreadedExecutor"
        and any(
            keyword.arg == "num_threads"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 2
            for keyword in call.keywords
        )
        for call in calls
    )
