"""Restricted ROS 2 Service, Action, Topic, and safety-heartbeat adapter."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action.client import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_action, get_message, get_service
from std_msgs.msg import String

from robot_agent.config import (
    CommandConfig,
    QoSConfig,
    SafeStopConfig,
    StateSourceConfig,
)
from robot_agent.coordinator import ExecutionResult


class RosAdapterError(RuntimeError):
    """Base error for restricted ROS adapter operations."""


class RosTypeError(RosAdapterError):
    """Raised when a configured ROS interface or value cannot be loaded."""


def build_message(
    ros_type: str,
    *,
    section: Literal["request", "goal"],
    values: dict[str, Any],
) -> Any:
    """Dynamically build only the locally configured request or goal type."""
    try:
        interface = get_service(ros_type) if section == "request" else get_action(ros_type)
        message_type = interface.Request if section == "request" else interface.Goal
        message = message_type()
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError) as exc:
        raise RosTypeError(f"unable to load ROS type {ros_type!r}: {exc}") from exc
    try:
        set_message_fields(message, values)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RosTypeError(f"invalid fields for {ros_type!r}: {exc}") from exc
    return message


def heartbeat_qos() -> QoSProfile:
    """QoS contract consumed by a future independent safety monitor."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        deadline=Duration(seconds=0.5),
        lifespan=Duration(seconds=1.0),
    )


def state_source_qos(config: QoSConfig) -> QoSProfile:
    reliability = (
        ReliabilityPolicy.RELIABLE
        if config.reliability == "reliable"
        else ReliabilityPolicy.BEST_EFFORT
    )
    durability = (
        DurabilityPolicy.VOLATILE
        if config.durability == "volatile"
        else DurabilityPolicy.TRANSIENT_LOCAL
    )
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=config.depth,
        reliability=reliability,
        durability=durability,
    )


def service_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def action_feedback_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def action_status_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


StateCallback = Callable[[str, Any, datetime], None]
FeedbackCallback = Callable[[UUID, dict[str, Any]], None]
StateSupplier = Callable[[], str]


class RosAdapter:
    """Own all ROS entities on a dedicated MultiThreadedExecutor thread."""

    def __init__(
        self,
        *,
        state_callback: StateCallback | None = None,
        feedback_callback: FeedbackCallback | None = None,
        node_name: str | None = None,
    ) -> None:
        self._context = Context()
        rclpy.init(args=None, context=self._context)
        self._node = Node(
            node_name or f"robot_agent_{uuid4().hex[:8]}",
            context=self._context,
        )
        self._executor = MultiThreadedExecutor(num_threads=4, context=self._context)
        self._executor.add_node(self._node)
        self._running = threading.Event()
        self._thread = threading.Thread(
            target=self._spin,
            name="robot-agent-rclpy-executor",
            daemon=False,
        )
        self._started = False
        self._closed = False
        self._state_callback = state_callback
        self._feedback_callback = feedback_callback
        self._service_clients: dict[tuple[str, str], Any] = {}
        self._action_clients: dict[tuple[str, str], Any] = {}
        self._subscriptions: list[Any] = []
        self._callback_groups: list[Any] = []
        self._goals: dict[UUID, Any] = {}
        self._pending: dict[UUID, Any] = {}
        self._goal_lock = threading.RLock()
        self._heartbeat_publisher: Any | None = None
        self._heartbeat_timer: Any | None = None
        self._heartbeat_sequence = 0
        self._boot_id = uuid4()

    def start(self) -> None:
        if self._closed:
            raise RosAdapterError("ROS adapter is closed")
        if self._started:
            return
        self._running.set()
        self._thread.start()
        self._started = True

    def set_callbacks(
        self,
        *,
        state_callback: StateCallback,
        feedback_callback: FeedbackCallback,
    ) -> None:
        if self._started:
            raise RosAdapterError("callbacks must be configured before the ROS adapter starts")
        self._state_callback = state_callback
        self._feedback_callback = feedback_callback

    def _spin(self) -> None:
        while self._running.is_set():
            self._executor.spin_once(timeout_sec=0.1)

    def configure_state_sources(self, sources: Mapping[str, StateSourceConfig]) -> None:
        if self._closed:
            raise RosAdapterError("ROS adapter is closed")
        for logical_path, source in sources.items():
            try:
                message_type = get_message(source.ros_type)
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                raise RosTypeError(
                    f"unable to load state source type {source.ros_type!r}: {exc}"
                ) from exc
            callback_group = ReentrantCallbackGroup()
            self._callback_groups.append(callback_group)

            def on_message(
                message: Any,
                *,
                path: str = logical_path,
                field: str = source.field,
            ) -> None:
                if self._state_callback is None:
                    return
                try:
                    value = _extract_field(message, field)
                    self._state_callback(path, value, datetime.now(timezone.utc))
                except Exception as exc:
                    self._node.get_logger().error(f"state source {path!r} conversion failed: {exc}")

            subscription = self._node.create_subscription(
                message_type,
                source.topic,
                on_message,
                state_source_qos(source.qos),
                callback_group=callback_group,
            )
            self._subscriptions.append(subscription)

    def configure_heartbeat(
        self,
        *,
        topic: str,
        period_ms: int,
        state_supplier: StateSupplier,
    ) -> None:
        if self._heartbeat_publisher is not None:
            raise RosAdapterError("safety heartbeat is already configured")
        callback_group = MutuallyExclusiveCallbackGroup()
        self._callback_groups.append(callback_group)
        self._heartbeat_publisher = self._node.create_publisher(
            String,
            topic,
            heartbeat_qos(),
            callback_group=callback_group,
        )

        def publish_heartbeat() -> None:
            assert self._heartbeat_publisher is not None
            self._heartbeat_sequence += 1
            message = String()
            message.data = json.dumps(
                {
                    "boot_id": str(self._boot_id),
                    "sequence": self._heartbeat_sequence,
                    "state": state_supplier(),
                    "monotonic_ns": time.monotonic_ns(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            self._heartbeat_publisher.publish(message)

        self._heartbeat_timer = self._node.create_timer(
            period_ms / 1000.0,
            publish_heartbeat,
            callback_group=callback_group,
        )

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult:
        self._require_started()
        if command.adapter == "service":
            return await self._execute_service(
                command_id,
                ros_name=command.ros_name,
                ros_type=command.ros_type,
                values=params,
            )
        return await self._execute_action(
            command_id,
            ros_name=command.ros_name,
            ros_type=command.ros_type,
            values=params,
        )

    async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
        self._require_started()
        command_id = uuid4()
        if config.adapter == "service":
            return await self._execute_service(
                command_id,
                ros_name=config.ros_name,
                ros_type=config.ros_type,
                values=config.request,
            )
        return await self._execute_action(
            command_id,
            ros_name=config.ros_name,
            ros_type=config.ros_type,
            values=config.request,
        )

    async def _execute_service(
        self,
        command_id: UUID,
        *,
        ros_name: str,
        ros_type: str,
        values: dict[str, Any],
    ) -> ExecutionResult:
        client = self._service_client(ros_name, ros_type)
        await _wait_for_ready(client.service_is_ready, f"service {ros_name}")
        request = build_message(ros_type, section="request", values=values)
        ros_future = client.call_async(request)
        with self._goal_lock:
            self._pending[command_id] = ros_future
        try:
            response = await _await_ros_future(ros_future)
        finally:
            with self._goal_lock:
                self._pending.pop(command_id, None)
        result = _message_dict(response)
        succeeded = bool(getattr(response, "success", True))
        detail = str(getattr(response, "message", "service completed"))
        return ExecutionResult(succeeded=succeeded, detail=detail, result=result)

    async def _execute_action(
        self,
        command_id: UUID,
        *,
        ros_name: str,
        ros_type: str,
        values: dict[str, Any],
    ) -> ExecutionResult:
        client = self._action_client(ros_name, ros_type)
        await _wait_for_ready(client.server_is_ready, f"action {ros_name}")
        goal = build_message(ros_type, section="goal", values=values)

        def on_feedback(feedback_message: Any) -> None:
            if self._feedback_callback is not None:
                self._feedback_callback(
                    command_id,
                    _message_dict(feedback_message.feedback),
                )

        send_future = client.send_goal_async(goal, feedback_callback=on_feedback)
        with self._goal_lock:
            self._pending[command_id] = send_future
        goal_handle = await _await_ros_future(send_future)
        if not goal_handle.accepted:
            with self._goal_lock:
                self._pending.pop(command_id, None)
            return ExecutionResult(
                succeeded=False,
                detail="action goal REJECTED",
                result=None,
            )
        with self._goal_lock:
            self._goals[command_id] = goal_handle
        result_future = goal_handle.get_result_async()
        with self._goal_lock:
            self._pending[command_id] = result_future
        try:
            response = await _await_ros_future(result_future)
        finally:
            with self._goal_lock:
                self._goals.pop(command_id, None)
                self._pending.pop(command_id, None)
        status_name = _goal_status_name(response.status)
        return ExecutionResult(
            succeeded=response.status == GoalStatus.STATUS_SUCCEEDED,
            detail=f"action finished with status {status_name}",
            result=_message_dict(response.result),
        )

    async def cancel(self, command_id: UUID) -> None:
        with self._goal_lock:
            goal_handle = self._goals.get(command_id)
            pending = self._pending.get(command_id)
        if goal_handle is not None:
            await _await_ros_future(goal_handle.cancel_goal_async())
        elif pending is not None:
            pending.cancel()

    def _service_client(self, ros_name: str, ros_type: str) -> Any:
        key = (ros_name, ros_type)
        client = self._service_clients.get(key)
        if client is not None:
            return client
        try:
            interface = get_service(ros_type)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            raise RosTypeError(f"unable to load ROS service {ros_type!r}: {exc}") from exc
        group = ReentrantCallbackGroup()
        self._callback_groups.append(group)
        client = self._node.create_client(
            interface,
            ros_name,
            qos_profile=service_qos(),
            callback_group=group,
        )
        self._service_clients[key] = client
        return client

    def _action_client(self, ros_name: str, ros_type: str) -> Any:
        key = (ros_name, ros_type)
        client = self._action_clients.get(key)
        if client is not None:
            return client
        try:
            interface = get_action(ros_type)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            raise RosTypeError(f"unable to load ROS action {ros_type!r}: {exc}") from exc
        group = ReentrantCallbackGroup()
        self._callback_groups.append(group)
        client = ActionClient(
            self._node,
            interface,
            ros_name,
            callback_group=group,
            goal_service_qos_profile=service_qos(),
            result_service_qos_profile=service_qos(),
            cancel_service_qos_profile=service_qos(),
            feedback_sub_qos_profile=action_feedback_qos(),
            status_sub_qos_profile=action_status_qos(),
        )
        self._action_clients[key] = client
        return client

    def _require_started(self) -> None:
        if not self._started or self._closed:
            raise RosAdapterError("ROS adapter must be started before execution")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._shutdown)

    def _shutdown(self) -> None:
        self._running.clear()
        self._executor.wake()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._executor.shutdown(timeout_sec=2.0)
        for client in self._action_clients.values():
            client.destroy()
        self._node.destroy_node()
        if self._context.ok():
            self._context.shutdown()

    @property
    def node(self) -> Node:
        return self._node


async def _wait_for_ready(
    predicate: Callable[[], bool],
    description: str,
    *,
    timeout: float = 5.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise RosAdapterError(f"timed out waiting for {description}")
        await asyncio.sleep(min(0.05, remaining))


async def _await_ros_future(future: Any) -> Any:
    loop = asyncio.get_running_loop()
    proxy: asyncio.Future[Any] = loop.create_future()

    def completed(source: Any) -> None:
        try:
            result = source.result()
        except Exception as exc:
            loop.call_soon_threadsafe(_set_exception_if_pending, proxy, exc)
        else:
            loop.call_soon_threadsafe(_set_result_if_pending, proxy, result)

    future.add_done_callback(completed)
    try:
        return await proxy
    except asyncio.CancelledError:
        future.cancel()
        raise


def _set_result_if_pending(future: asyncio.Future[Any], value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception_if_pending(future: asyncio.Future[Any], exception: Exception) -> None:
    if not future.done():
        future.set_exception(exception)


def _message_dict(message: Any) -> dict[str, Any]:
    converted = message_to_ordereddict(message)
    return cast(dict[str, Any], json.loads(json.dumps(converted, allow_nan=False)))


def _extract_field(message: Any, field: str) -> Any:
    value = message
    for segment in field.split("."):
        if not hasattr(value, segment):
            raise RosTypeError(f"message has no field {field!r}")
        value = getattr(value, segment)
    if hasattr(value, "get_fields_and_field_types"):
        return _message_dict(value)
    if isinstance(value, (bytes, bytearray)):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _goal_status_name(status: int) -> str:
    names = {
        GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
        GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
        GoalStatus.STATUS_EXECUTING: "EXECUTING",
        GoalStatus.STATUS_CANCELING: "CANCELING",
        GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
        GoalStatus.STATUS_CANCELED: "CANCELED",
        GoalStatus.STATUS_ABORTED: "ABORTED",
    }
    return names.get(status, f"UNKNOWN_{status}")
