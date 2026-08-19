from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from action_msgs.msg import GoalStatus

from robot_agent.config import CommandConfig, StateSourceConfig
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


class ImmediateFuture:
    def __init__(self, value: Any = None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.cancelled = False

    def add_done_callback(self, callback: Any) -> None:
        callback(self)

    def result(self) -> Any:
        if self.error is not None:
            raise self.error
        return self.value

    def done(self) -> bool:
        return True

    def cancel(self) -> None:
        self.cancelled = True


class LiveThread:
    def __init__(self, *, alive: bool = True) -> None:
        self.alive = alive
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        del timeout


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
    for adapter_kind, ros_type, error in (
        ("service", "missing_interfaces/srv/Nope", "unable to load ROS service"),
        ("action", "missing_interfaces/action/Nope", "unable to load ROS action"),
    ):
        missing_command = CommandConfig.model_validate(
            {
                "adapter": adapter_kind,
                "ros_name": "/unused",
                "ros_type": ros_type,
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
        with pytest.raises(RosTypeError, match=error):
            await adapter.execute(uuid4(), missing_command, {})
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


def test_ros_thread_bridge_settles_success_failure_and_shutdown_requests() -> None:
    adapter = RosAdapter()
    success: concurrent.futures.Future[Any] = concurrent.futures.Future()
    failure: concurrent.futures.Future[Any] = concurrent.futures.Future()
    cancelled: concurrent.futures.Future[Any] = concurrent.futures.Future()
    cancelled.cancel()
    adapter._requests.put((lambda: "done", success))
    adapter._requests.put((lambda: (_ for _ in ()).throw(ValueError("boom")), failure))
    adapter._requests.put((lambda: "ignored", cancelled))

    adapter._drain_requests()

    assert success.result() == "done"
    with pytest.raises(ValueError, match="boom"):
        failure.result()
    assert cancelled.cancelled()

    stopped: concurrent.futures.Future[Any] = concurrent.futures.Future()
    already_cancelled: concurrent.futures.Future[Any] = concurrent.futures.Future()
    already_cancelled.cancel()
    adapter._requests.put((lambda: None, stopped))
    adapter._requests.put((lambda: None, already_cancelled))
    adapter._fail_requests(RosAdapterError("stopped"))
    with pytest.raises(RosAdapterError, match="stopped"):
        stopped.result()

    with pytest.raises(RosAdapterError, match="owner thread"):
        adapter._require_owner_thread()
    with pytest.raises(RosAdapterError, match="node is not initialized"):
        adapter._require_node()
    with pytest.raises(RosAdapterError, match="executor is not initialized"):
        adapter._require_executor()

    adapter._thread_error = RuntimeError("spin failed")
    with pytest.raises(RosAdapterError, match="spin failed"):
        adapter.check_health()
    adapter._thread_error = None
    adapter._started = True
    with pytest.raises(RosAdapterError, match="not running"):
        adapter.check_health()

    adapter._thread = LiveThread()  # type: ignore[assignment]
    adapter._owner_thread_id = threading.get_ident()
    adapter._accepting_requests = True
    assert adapter._submit_ros_call(lambda: 42).result() == 42
    failed_call = adapter._submit_ros_call(
        lambda: (_ for _ in ()).throw(RuntimeError("call failed"))
    )
    with pytest.raises(RuntimeError, match="call failed"):
        failed_call.result()


def test_ros_thread_bridge_rejects_work_when_bounded_queue_is_full() -> None:
    adapter = RosAdapter()
    adapter._started = True
    adapter._thread = LiveThread()  # type: ignore[assignment]
    adapter._owner_thread_id = -1
    adapter._accepting_requests = True
    pending = [
        adapter._submit_ros_call(lambda: None)
        for _ in range(adapter._requests.maxsize)
    ]

    with pytest.raises(RosAdapterError, match="queue is full"):
        adapter._submit_ros_call(lambda: None)

    adapter._fail_requests(RosAdapterError("test cleanup"))
    assert all(future.done() for future in pending)


def test_action_send_is_refused_if_cancel_won_before_owner_thread_dispatch() -> None:
    adapter = RosAdapter()
    adapter._owner_thread_id = threading.get_ident()
    command_id = uuid4()
    adapter._cancel_requested.add(command_id)

    class Client:
        calls = 0

        def send_goal_async(self, goal: object, *, feedback_callback: Any) -> ImmediateFuture:
            del goal, feedback_callback
            self.calls += 1
            return ImmediateFuture(SimpleNamespace(accepted=True))

    client = Client()
    with pytest.raises(RosAdapterError, match="cancelled before the ROS goal"):
        adapter._begin_action_send(command_id, client, object(), lambda message: None)

    assert client.calls == 0
    assert command_id not in adapter._pending


def test_ros_start_reports_closed_timeout_startup_and_stopped_thread() -> None:
    closed = RosAdapter()
    closed._closed = True
    with pytest.raises(RosAdapterError, match="closed"):
        closed.start()

    timed_out = RosAdapter()
    timed_out._thread = LiveThread()  # type: ignore[assignment]
    timed_out._startup_ready = SimpleNamespace(wait=lambda timeout: False)
    with pytest.raises(RosAdapterError, match="did not become ready"):
        timed_out.start()

    startup_failed = RosAdapter()
    startup_failed._thread = LiveThread()  # type: ignore[assignment]
    startup_failed._startup_ready = SimpleNamespace(wait=lambda timeout: True)
    startup_failed._startup_error = ValueError("bad ROS setup")
    with pytest.raises(RosAdapterError, match="bad ROS setup"):
        startup_failed.start()

    stopped = RosAdapter()
    stopped._thread = LiveThread(alive=False)  # type: ignore[assignment]
    stopped._startup_ready = SimpleNamespace(wait=lambda timeout: True)
    with pytest.raises(RosAdapterError, match="stopped during startup"):
        stopped.start()


@pytest.mark.asyncio
async def test_ros_action_tracking_handles_rejection_failures_and_cancel_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RosAdapter()
    adapter._started = True
    adapter._thread = LiveThread()  # type: ignore[assignment]
    adapter._owner_thread_id = threading.get_ident()
    adapter._accepting_requests = True
    command_id = uuid4()

    adapter._action_goal_response(command_id, ImmediateFuture(error=ValueError("rejected")))
    adapter._action_goal_response(
        command_id,
        ImmediateFuture(SimpleNamespace(accepted=False)),
    )
    requested: list[object] = []
    monkeypatch.setattr(adapter, "_request_cancel_without_waiting", requested.append)
    adapter._cancel_requested.add(command_id)
    accepted = SimpleNamespace(accepted=True)
    adapter._action_goal_response(command_id, ImmediateFuture(accepted))
    assert requested == [command_id]

    class GoalHandle:
        accepted = True

        def __init__(self) -> None:
            self.result_future = ImmediateFuture("result")
            self.cancel_future = ImmediateFuture(SimpleNamespace(goals_canceling=[]))

        def get_result_async(self) -> ImmediateFuture:
            return self.result_future

        def cancel_goal_async(self) -> ImmediateFuture:
            return self.cancel_future

    goal_handle = GoalHandle()
    adapter._begin_action_result(command_id, goal_handle)
    first_cancel = adapter._begin_goal_cancel(command_id, goal_handle)
    assert adapter._begin_goal_cancel(command_id, goal_handle) is first_cancel
    with pytest.raises(RosAdapterError, match="rejected goal cancellation"):
        await adapter.cancel(command_id)

    pending_service = uuid4()
    pending_future = ImmediateFuture()
    adapter._pending[pending_service] = pending_future
    with pytest.raises(RosAdapterError, match="service execution cannot be cancellation"):
        await adapter.cancel(pending_service)
    assert pending_future.cancelled is False

    failed_acceptance = uuid4()
    adapter._pending[failed_acceptance] = ImmediateFuture(error=ValueError("accept failed"))
    adapter._pending_actions.add(failed_acceptance)
    with pytest.raises(ValueError, match="accept failed"):
        await adapter.cancel(failed_acceptance)

    tracked = uuid4()
    adapter._goals[tracked] = object()
    adapter._late_cancel_completed(
        tracked,
        ImmediateFuture(error=RuntimeError("late")),
        cleanup=False,
    )
    assert tracked in adapter._goals
    adapter._late_action_completed(tracked, ImmediateFuture(error=RuntimeError("late result")))
    assert tracked not in adapter._goals
