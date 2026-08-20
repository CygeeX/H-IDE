from __future__ import annotations

import ast
from pathlib import Path

from hide_mujoco_sonic.gait_adapter import GaitState, SonicIntent
from hide_mujoco_sonic.ros_bridge import _suppress_for_correlated_stop


ROOT = Path(__file__).resolve().parents[1]


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


def _keyword_name(call: ast.Call, keyword: str) -> str | None:
    for item in call.keywords:
        if item.arg == keyword and isinstance(item.value, ast.Name):
            return item.value.id
    return None


def _calls(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


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
    assert len(subscriptions) == 4
    assert len(timers) == 2
    assert len(guards) == 1
    assert all(
        _keyword_name(call, "callback_group") == "state_group"
        for call in subscriptions + timers + guards
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
        "/robot/control_arm": "state_group",
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
