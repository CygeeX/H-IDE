from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from action_msgs.msg import GoalStatus
from robot_agent.ros_adapter import (
    RosAdapter,
    RosAdapterError,
    RosTypeError,
    _await_ros_future,
    _extract_field,
    _goal_status_name,
    _set_exception_if_pending,
    _set_result_if_pending,
    _wait_for_ready,
    build_message,
)

from robot_agent.config import CommandConfig, StateSourceConfig


def service_command() -> CommandConfig:
    return CommandConfig.model_validate(
        {
            "adapter": "service",
            "ros_name": "/unused",
            "ros_type": "std_srvs/srv/Trigger",
            "request_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "max_ttl_ms": 1000,
            "timeout_ms": 1000,
            "resources": ["test"],
        }
    )


@pytest.mark.asyncio
async def test_ros_adapter_lifecycle_and_dynamic_type_guards() -> None:
    adapter = RosAdapter()
    with pytest.raises(RosAdapterError, match="started"):
        await adapter.execute(uuid4(), service_command(), {})
    with pytest.raises(RosTypeError, match="unable to load state source"):
        adapter.configure_state_sources(
            {
                "robot.mode": StateSourceConfig.model_validate(
                    {
                        "topic": "/unused",
                        "ros_type": "missing_interfaces/msg/Nope",
                        "field": "data",
                        "qos": {
                            "reliability": "reliable",
                            "durability": "volatile",
                            "depth": 1,
                        },
                    }
                )
            }
        )
    with pytest.raises(RosTypeError, match="unable to load ROS service"):
        adapter._service_client("/unused", "missing_interfaces/srv/Nope")
    with pytest.raises(RosTypeError, match="unable to load ROS action"):
        adapter._action_client("/unused", "missing_interfaces/action/Nope")

    adapter.configure_heartbeat(
        topic="/robot_agent/test_heartbeat",
        period_ms=250,
        state_supplier=lambda: "SAFE_HOLD",
    )
    with pytest.raises(RosAdapterError, match="already configured"):
        adapter.configure_heartbeat(
            topic="/robot_agent/test_heartbeat",
            period_ms=250,
            state_supplier=lambda: "SAFE_HOLD",
        )
    adapter.start()
    adapter.start()
    assert adapter.node.get_name().startswith("robot_agent_")
    with pytest.raises(RosAdapterError, match=r"before.*starts"):
        adapter.set_callbacks(
            state_callback=lambda path, value, at: None,
            feedback_callback=lambda command_id, feedback: None,
        )
    await adapter.close()
    await adapter.close()


@pytest.mark.asyncio
async def test_ros_future_bridge_ready_timeout_and_settlement_helpers() -> None:
    await _wait_for_ready(lambda: True, "ready", timeout=0)
    with pytest.raises(RosAdapterError, match="timed out"):
        await _wait_for_ready(lambda: False, "missing", timeout=0)

    loop = asyncio.get_running_loop()
    source: asyncio.Future[str] = loop.create_future()
    waiting = asyncio.create_task(_await_ros_future(source))
    await asyncio.sleep(0)
    source.set_result("done")
    assert await waiting == "done"

    failed: asyncio.Future[str] = loop.create_future()
    failure_wait = asyncio.create_task(_await_ros_future(failed))
    await asyncio.sleep(0)
    failed.set_exception(ValueError("ROS failed"))
    with pytest.raises(ValueError, match="ROS failed"):
        await failure_wait

    settled: asyncio.Future[str] = loop.create_future()
    _set_result_if_pending(settled, "first")
    _set_result_if_pending(settled, "second")
    _set_exception_if_pending(settled, RuntimeError("ignored"))
    assert settled.result() == "first"


def test_ros_value_conversion_action_goal_and_status_names() -> None:
    goal = build_message(
        "example_interfaces/action/Fibonacci",
        section="goal",
        values={"order": 5},
    )
    assert goal.order == 5
    assert _extract_field(SimpleNamespace(data=b"abc"), "data") == [97, 98, 99]
    assert _extract_field(SimpleNamespace(data=SimpleNamespace(tolist=lambda: [1, 2])), "data") == [
        1,
        2,
    ]
    with pytest.raises(RosTypeError, match="no field"):
        _extract_field(SimpleNamespace(), "missing")

    assert _goal_status_name(GoalStatus.STATUS_SUCCEEDED) == "SUCCEEDED"
    assert _goal_status_name(999) == "UNKNOWN_999"
