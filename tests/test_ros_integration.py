from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any
from uuid import UUID, uuid4

import pytest
import rclpy
from example_interfaces.action import Fibonacci
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from robot_agent.ros_adapter import RosAdapter, heartbeat_qos
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from robot_agent.config import CommandConfig, SafeStopConfig, StateSourceConfig


def command_config(
    *,
    adapter: str,
    ros_name: str,
    ros_type: str,
    schema: dict[str, Any],
) -> CommandConfig:
    return CommandConfig.model_validate(
        {
            "adapter": adapter,
            "ros_name": ros_name,
            "ros_type": ros_type,
            "request_schema": schema,
            "max_ttl_ms": 5000,
            "timeout_ms": 3000,
            "resources": ["test"],
        }
    )


class MockRosGraph:
    def __init__(self) -> None:
        suffix = uuid4().hex[:8]
        self.prefix = f"/robot_agent_test_{suffix}"
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node(f"mock_robot_{suffix}", context=self.context)
        self.executor = MultiThreadedExecutor(num_threads=4, context=self.context)
        self.executor.add_node(self.node)
        self._running = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.safe_stop_calls = 0

        def set_bool(request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
            response.success = request.data
            response.message = "enabled" if request.data else "disabled"
            return response

        def safe_stop(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            del request
            self.safe_stop_calls += 1
            response.success = True
            response.message = "stopped"
            return response

        group = ReentrantCallbackGroup()
        self.node.create_service(
            SetBool,
            f"{self.prefix}/power_enable",
            set_bool,
            callback_group=group,
        )
        self.node.create_service(
            Trigger,
            f"{self.prefix}/safe_stop",
            safe_stop,
            callback_group=group,
        )

        def execute(goal_handle: Any) -> Fibonacci.Result:
            sequence = [0, 1]
            for index in range(2, goal_handle.request.order):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result = Fibonacci.Result()
                    result.sequence = sequence
                    return result
                sequence.append(sequence[index - 1] + sequence[index - 2])
                feedback = Fibonacci.Feedback()
                feedback.sequence = sequence
                goal_handle.publish_feedback(feedback)
                time.sleep(0.01)
            goal_handle.succeed()
            result = Fibonacci.Result()
            result.sequence = sequence
            return result

        self.action_server = ActionServer(
            self.node,
            Fibonacci,
            f"{self.prefix}/fibonacci",
            execute_callback=execute,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=group,
        )

        self.mode_publisher = self.node.create_publisher(
            String,
            f"{self.prefix}/mode",
            1,
        )
        self.mode_timer = self.node.create_timer(0.05, self._publish_mode)
        self.heartbeats: list[dict[str, Any]] = []
        self.heartbeat_event = threading.Event()

        def heartbeat(message: String) -> None:
            self.heartbeats.append(json.loads(message.data))
            self.heartbeat_event.set()

        self.node.create_subscription(
            String,
            f"{self.prefix}/heartbeat",
            heartbeat,
            heartbeat_qos(),
            callback_group=group,
        )

    def _publish_mode(self) -> None:
        message = String()
        message.data = "IDLE"
        self.mode_publisher.publish(message)

    def start(self) -> None:
        self._running.set()
        self.thread.start()

    def _spin(self) -> None:
        while self._running.is_set():
            self.executor.spin_once(timeout_sec=0.1)

    def close(self) -> None:
        self._running.clear()
        self.executor.wake()
        self.thread.join(timeout=2.0)
        self.executor.shutdown(timeout_sec=2.0)
        self.action_server.destroy()
        self.node.destroy_node()
        self.context.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ros_adapter_service_action_topic_heartbeat_and_cancel() -> None:
    graph = MockRosGraph()
    state_event = threading.Event()
    feedback_event = threading.Event()
    states: list[tuple[str, Any]] = []
    feedback: list[tuple[UUID, dict[str, Any]]] = []

    def on_state(path: str, value: Any, observed_at: object) -> None:
        del observed_at
        states.append((path, value))
        state_event.set()

    def on_feedback(command_id: UUID, value: dict[str, Any]) -> None:
        feedback.append((command_id, value))
        feedback_event.set()

    adapter = RosAdapter(state_callback=on_state, feedback_callback=on_feedback)
    adapter.configure_state_sources(
        {
            "robot.mode": StateSourceConfig.model_validate(
                {
                    "topic": f"{graph.prefix}/mode",
                    "ros_type": "std_msgs/msg/String",
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
    adapter.configure_heartbeat(
        topic=f"{graph.prefix}/heartbeat",
        period_ms=100,
        state_supplier=lambda: "SAFE_HOLD",
    )
    graph.start()
    adapter.start()
    try:
        service = command_config(
            adapter="service",
            ros_name=f"{graph.prefix}/power_enable",
            ros_type="std_srvs/srv/SetBool",
            schema={
                "type": "object",
                "properties": {"data": {"type": "boolean"}},
                "required": ["data"],
                "additionalProperties": False,
            },
        )
        service_result = await asyncio.wait_for(
            adapter.execute(uuid4(), service, {"data": True}), timeout=4
        )
        assert service_result.succeeded is True
        assert service_result.result == {"success": True, "message": "enabled"}

        action = command_config(
            adapter="action",
            ros_name=f"{graph.prefix}/fibonacci",
            ros_type="example_interfaces/action/Fibonacci",
            schema={
                "type": "object",
                "properties": {"order": {"type": "integer"}},
                "required": ["order"],
                "additionalProperties": False,
            },
        )
        action_id = uuid4()
        action_result = await asyncio.wait_for(
            adapter.execute(action_id, action, {"order": 7}), timeout=4
        )
        assert action_result.succeeded is True
        assert action_result.result == {"sequence": [0, 1, 1, 2, 3, 5, 8]}
        assert any(item[0] == action_id for item in feedback)

        feedback_event.clear()
        cancel_id = uuid4()
        action_task = asyncio.create_task(adapter.execute(cancel_id, action, {"order": 100}))
        assert await asyncio.to_thread(feedback_event.wait, 3)
        await adapter.cancel(cancel_id)
        cancelled = await asyncio.wait_for(action_task, timeout=4)
        assert cancelled.succeeded is False
        assert "CANCELED" in cancelled.detail

        assert await asyncio.to_thread(state_event.wait, 3)
        assert ("robot.mode", "IDLE") in states
        assert await asyncio.to_thread(graph.heartbeat_event.wait, 3)
        assert graph.heartbeats[-1]["state"] == "SAFE_HOLD"

        safe_stop = SafeStopConfig.model_validate(
            {
                "adapter": "service",
                "ros_name": f"{graph.prefix}/safe_stop",
                "ros_type": "std_srvs/srv/Trigger",
                "request": {},
                "timeout_ms": 1000,
            }
        )
        stop_result = await asyncio.wait_for(adapter.safe_stop(safe_stop), timeout=3)
        assert stop_result.succeeded is True
        assert graph.safe_stop_calls == 1
    finally:
        await adapter.close()
        graph.close()
