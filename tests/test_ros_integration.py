from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
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
from robot_agent.arbitration import ResourceArbiter
from robot_agent.coordinator import CommandCancellationError, CommandCoordinator
from robot_agent.ledger import CommandLedger, CommandState
from robot_agent.protocol import CommandRequest
from robot_agent.ros_adapter import RosAdapter, heartbeat_qos
from robot_agent.session import ControlState, SessionController, SessionStateError
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from robot_agent.config import CommandConfig, SafeStopConfig, StateSourceConfig


def command_config(
    *,
    adapter: str,
    ros_name: str,
    ros_type: str,
    schema: dict[str, Any],
    timeout_ms: int = 3000,
) -> CommandConfig:
    return CommandConfig.model_validate(
        {
            "adapter": adapter,
            "ros_name": ros_name,
            "ros_type": ros_type,
            "request_schema": schema,
            "max_ttl_ms": 5000,
            "timeout_ms": timeout_ms,
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
        self.slow_service_started = threading.Event()
        self.slow_service_completed = threading.Event()
        self.started_goal_event = threading.Event()
        self.cancelled_goal_event = threading.Event()
        self.completed_goal_event = threading.Event()

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

        def slow_service(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            del request
            self.slow_service_started.set()
            time.sleep(1.0)
            self.slow_service_completed.set()
            response.success = True
            response.message = "slow side effect completed"
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
        self.node.create_service(
            Trigger,
            f"{self.prefix}/slow_service",
            slow_service,
            callback_group=group,
        )

        def execute(goal_handle: Any) -> Fibonacci.Result:
            self.started_goal_event.set()
            sequence = [0, 1]
            for index in range(2, goal_handle.request.order):
                if goal_handle.is_cancel_requested:
                    time.sleep(0.1)
                    goal_handle.canceled()
                    self.cancelled_goal_event.set()
                    result = Fibonacci.Result()
                    result.sequence = sequence
                    return result
                sequence.append(sequence[index - 1] + sequence[index - 2])
                feedback = Fibonacci.Feedback()
                feedback.sequence = sequence
                goal_handle.publish_feedback(feedback)
                time.sleep(0.01)
            goal_handle.succeed()
            self.completed_goal_event.set()
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
        self.executor._executor.shutdown(wait=True, cancel_futures=False)
        self.executor.shutdown(timeout_sec=2.0)
        self.action_server.destroy()
        self.node.destroy_node()
        self.context.shutdown()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ros_adapter_service_action_topic_heartbeat_and_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    asyncio_thread = threading.get_ident()
    client_creation_threads: list[int] = []
    original_service_client = adapter._service_client
    original_action_client = adapter._action_client

    def observed_service_client(ros_name: str, ros_type: str) -> Any:
        client_creation_threads.append(threading.get_ident())
        return original_service_client(ros_name, ros_type)

    def observed_action_client(ros_name: str, ros_type: str) -> Any:
        client_creation_threads.append(threading.get_ident())
        return original_action_client(ros_name, ros_type)

    monkeypatch.setattr(adapter, "_service_client", observed_service_client)
    monkeypatch.setattr(adapter, "_action_client", observed_action_client)
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
        assert client_creation_threads
        assert set(client_creation_threads) != {asyncio_thread}
        assert asyncio_thread not in client_creation_threads
    finally:
        await adapter.close()
        graph.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_coordinator_timeout_cancels_real_ros_goal_before_releasing_resource(
    tmp_path: Path,
) -> None:
    graph = MockRosGraph()
    adapter = RosAdapter()
    graph.start()
    adapter.start()
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
        timeout_ms=300,
    )
    try:
        warmup = action.model_copy(update={"timeout_ms": 3000})
        assert (await adapter.execute(uuid4(), warmup, {"order": 3})).succeeded is True
        graph.started_goal_event.clear()
        graph.cancelled_goal_event.clear()
        graph.completed_goal_event.clear()

        async def no_op_safe_stop(reason: str) -> None:
            del reason

        session = SessionController(
            lease_timeout=timedelta(seconds=2),
            safe_stop=no_op_safe_stop,
        )
        session_id = uuid4()
        issued_at = datetime.now(timezone.utc)
        await session.connected(session_id)
        await session.acquire(session_id, operator_id="integration", at=issued_at)
        request = CommandRequest.model_validate(
            {
                "version": "1.0",
                "type": "command.request",
                "message_id": uuid4(),
                "robot_id": "g1-dev",
                "session_id": session_id,
                "sent_at": issued_at,
                "payload": {
                    "command_id": uuid4(),
                    "kind": "task.execute",
                    "issued_at": issued_at,
                    "ttl_ms": 5000,
                    "params": {"order": 100},
                },
            }
        )
        with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
            coordinator = CommandCoordinator(
                commands={"task.execute": action},
                ledger=ledger,
                arbiter=ResourceArbiter(),
                session=session,
                executor=adapter,
            )

            record = await coordinator.submit(request, at=issued_at)

            assert record.state is CommandState.TIMED_OUT
            assert graph.started_goal_event.is_set()
            assert graph.cancelled_goal_event.is_set()
            assert graph.completed_goal_event.is_set() is False
            assert coordinator.active_commands == frozenset()
    finally:
        await adapter.close()
        graph.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_service_timeout_latches_safe_hold_while_ros_side_effect_continues(
    tmp_path: Path,
) -> None:
    graph = MockRosGraph()
    adapter = RosAdapter()
    graph.start()
    adapter.start()
    service = command_config(
        adapter="service",
        ros_name=f"{graph.prefix}/slow_service",
        ros_type="std_srvs/srv/Trigger",
        schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        timeout_ms=300,
    )
    try:
        assert (await adapter.execute(uuid4(), service, {})).succeeded is True
        graph.slow_service_started.clear()
        graph.slow_service_completed.clear()

        async def no_op_safe_stop(reason: str) -> None:
            del reason

        session = SessionController(
            lease_timeout=timedelta(seconds=2),
            safe_stop=no_op_safe_stop,
        )
        session_id = uuid4()
        issued_at = datetime.now(timezone.utc)
        await session.connected(session_id)
        await session.acquire(session_id, operator_id="integration", at=issued_at)
        request = CommandRequest.model_validate(
            {
                "version": "1.0",
                "type": "command.request",
                "message_id": uuid4(),
                "robot_id": "g1-dev",
                "session_id": session_id,
                "sent_at": issued_at,
                "payload": {
                    "command_id": uuid4(),
                    "kind": "power.slow",
                    "issued_at": issued_at,
                    "ttl_ms": 5000,
                    "params": {},
                },
            }
        )
        with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
            coordinator = CommandCoordinator(
                commands={"power.slow": service},
                ledger=ledger,
                arbiter=ResourceArbiter(),
                session=session,
                executor=adapter,
            )

            with pytest.raises(CommandCancellationError) as failed:
                await coordinator.submit(request, at=issued_at)

            assert failed.value.record.state is CommandState.TIMED_OUT
            assert graph.slow_service_started.is_set()
            assert graph.slow_service_completed.is_set() is False
            assert session.state is ControlState.SAFE_HOLD
            with pytest.raises(SessionStateError, match="safety fault is latched"):
                await session.acquire(
                    session_id,
                    operator_id="integration",
                    at=datetime.now(timezone.utc),
                )
            assert await asyncio.to_thread(graph.slow_service_completed.wait, 2)
    finally:
        await adapter.close()
        graph.close()
