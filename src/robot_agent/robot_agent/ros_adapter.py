"""Restricted ROS 2 Service, Action, Topic, and safety-heartbeat adapter."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import math
import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action.client import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
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
    MotionSafetyFaultConfig,
    QoSConfig,
    SafeStopConfig,
    StateSourceConfig,
)
from robot_agent.coordinator import ExecutionResult
from robot_agent.observation import ObservationHealthSample


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


def navigate_to_pose_goal_values(
    goal_pose: Mapping[str, Any],
    *,
    stamp_sec: int,
    stamp_nanosec: int,
) -> dict[str, Any]:
    """Create the only ROS fields allowed by the H-IDE navigation boundary."""
    if goal_pose.get("frameId") != "map":
        raise RosTypeError("navigation goalPose.frameId must be map")
    yaw = goal_pose.get("yaw")
    if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
        raise RosTypeError("navigation goalPose.yaw must be numeric")
    try:
        x = float(goal_pose["x"])
        y = float(goal_pose["y"])
        z = float(goal_pose["z"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RosTypeError("navigation goalPose x/y/z must be numeric") from exc
    yaw_value = float(yaw)
    if not all(math.isfinite(value) for value in (x, y, z, yaw_value)):
        raise RosTypeError("navigation goalPose values must be finite")
    if not -math.pi <= yaw_value <= math.pi:
        raise RosTypeError("navigation goalPose.yaw must be within [-pi, pi]")
    if stamp_sec < 0 or not 0 <= stamp_nanosec < 1_000_000_000:
        raise RosTypeError("navigation PoseStamped time is invalid")
    return {
        "pose": {
            "header": {
                "stamp": {"sec": stamp_sec, "nanosec": stamp_nanosec},
                "frame_id": "map",
            },
            "pose": {
                "position": {"x": x, "y": y, "z": z},
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": math.sin(yaw_value / 2.0),
                    "w": math.cos(yaw_value / 2.0),
                },
            },
        },
        "behavior_tree": "",
    }


def nav2_spin_goal_values(params: Mapping[str, Any]) -> dict[str, Any]:
    """Translate an authorized relative yaw into the Nav2 Spin action only.

    Angular velocity remains a Robot Agent/Nav2 configuration concern.  H-IDE
    may request a bounded relative orientation change, but it cannot publish a
    velocity or inject a behavior tree.
    """

    target_yaw = params.get("targetYawRad")
    if isinstance(target_yaw, bool) or not isinstance(target_yaw, (int, float)):
        raise RosTypeError("navigation targetYawRad must be numeric")
    target_yaw_value = float(target_yaw)
    if not math.isfinite(target_yaw_value):
        raise RosTypeError("navigation targetYawRad must be finite")
    if target_yaw_value == 0.0 or abs(target_yaw_value) > math.tau:
        raise RosTypeError("navigation targetYawRad must be within [-2pi, 2pi] and non-zero")

    allowance_ms = params.get("timeAllowanceMs")
    if isinstance(allowance_ms, bool) or not isinstance(allowance_ms, int):
        raise RosTypeError("navigation timeAllowanceMs must be an integer")
    if not 100 <= allowance_ms <= 300_000:
        raise RosTypeError("navigation timeAllowanceMs must be within [100, 300000]")
    return {
        "target_yaw": target_yaw_value,
        "time_allowance": {
            "sec": allowance_ms // 1_000,
            "nanosec": (allowance_ms % 1_000) * 1_000_000,
        },
    }


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


def motion_safety_fault_qos() -> QoSProfile:
    """Durable one-sample contract for a latched motion-safety fault."""

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
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
ObservationHealthCallback = Callable[[ObservationHealthSample], None]
MotionSafetyFaultCallback = Callable[[str, int], None]
StateSupplier = Callable[[], str]

_ROS_REQUEST_QUEUE_SIZE = 256
_TERMINAL_GOAL_STATUSES = frozenset(
    {
        GoalStatus.STATUS_SUCCEEDED,
        GoalStatus.STATUS_CANCELED,
        GoalStatus.STATUS_ABORTED,
    }
)


@dataclass(frozen=True)
class RosActionReadiness:
    ready: bool
    reason: str | None = None


@dataclass(frozen=True)
class MotionFaultCancelEvidence:
    """Monotonic evidence for one exact Nav2 cancellation boundary."""

    command_id: UUID
    received_monotonic_ns: int
    cancel_begin_monotonic_ns: int
    ros_cancel_call_monotonic_ns: int | None
    nav2_terminal_monotonic_ns: int | None
    outcome: str

    @property
    def cancel_begin_latency_ms(self) -> float:
        return (self.cancel_begin_monotonic_ns - self.received_monotonic_ns) / 1_000_000

    @property
    def nav2_terminal_latency_ms(self) -> float | None:
        if self.nav2_terminal_monotonic_ns is None:
            return None
        return (
            self.nav2_terminal_monotonic_ns - self.received_monotonic_ns
        ) / 1_000_000

    @property
    def ros_cancel_call_latency_ms(self) -> float | None:
        if self.ros_cancel_call_monotonic_ns is None:
            return None
        return (
            self.ros_cancel_call_monotonic_ns - self.received_monotonic_ns
        ) / 1_000_000


class MotionFaultCancellationError(RosAdapterError):
    """Exact fault cancellation missed or could not prove its deadline."""

    def __init__(self, message: str, evidence: MotionFaultCancelEvidence) -> None:
        self.evidence = evidence
        super().__init__(message)


class RosAdapter:
    """Own all ROS entities on a dedicated MultiThreadedExecutor thread."""

    def __init__(
        self,
        *,
        state_callback: StateCallback | None = None,
        feedback_callback: FeedbackCallback | None = None,
        node_name: str | None = None,
        robot_boot_id: UUID | None = None,
    ) -> None:
        self._node_name = node_name or f"robot_agent_{uuid4().hex[:8]}"
        self._context: Context | None = None
        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._running = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._thread = threading.Thread(
            target=self._spin,
            name="robot-agent-rclpy-executor",
            daemon=False,
        )
        self._startup_ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread_error: BaseException | None = None
        self._owner_thread_id: int | None = None
        self._requests: queue.Queue[tuple[Callable[[], Any], concurrent.futures.Future[Any]]] = (
            queue.Queue(maxsize=_ROS_REQUEST_QUEUE_SIZE)
        )
        self._started = False
        self._accepting_requests = False
        self._closed = False
        self._state_callback = state_callback
        self._feedback_callback = feedback_callback
        self._observation_health_callback: ObservationHealthCallback | None = None
        self._motion_fault_configuration: tuple[
            MotionSafetyFaultConfig, MotionSafetyFaultCallback
        ] | None = None
        self._service_clients: dict[tuple[str, str], Any] = {}
        self._action_clients: dict[tuple[str, str], Any] = {}
        self._subscriptions: list[Any] = []
        self._callback_groups: list[Any] = []
        self._configured_state_sources: dict[str, StateSourceConfig] = {}
        self._heartbeat_configuration: tuple[str, int, StateSupplier] | None = None
        self._goals: dict[UUID, Any] = {}
        self._pending: dict[UUID, Any] = {}
        self._pending_actions: set[UUID] = set()
        self._service_waiters: set[UUID] = set()
        self._cancel_requested: set[UUID] = set()
        self._cancel_futures: dict[UUID, Any] = {}
        self._cancel_started_monotonic_ns: dict[UUID, int] = {}
        self._action_waiters: set[UUID] = set()
        self._goal_lock = threading.RLock()
        self._heartbeat_publisher: Any | None = None
        self._heartbeat_timer: Any | None = None
        self._heartbeat_sequence = 0
        self._observation_configuration: tuple[str, str, Any, Any] | None = None
        self._observation_demand_publisher: Any | None = None
        self._observation_demand_sequence = 0
        self._boot_id = robot_boot_id or uuid4()

    def start(self) -> None:
        if self._closed:
            raise RosAdapterError("ROS adapter is closed")
        if self._started:
            return
        self._running.set()
        self._thread.start()
        if not self._startup_ready.wait(timeout=5.0):
            self._running.clear()
            self._wake_executor()
            raise RosAdapterError("ROS executor thread did not become ready")
        if self._startup_error is not None:
            raise RosAdapterError(
                f"ROS executor startup failed: {self._startup_error}"
            ) from self._startup_error
        if not self._thread.is_alive():
            raise RosAdapterError("ROS executor thread stopped during startup")
        with self._lifecycle_lock:
            self._started = True
            self._accepting_requests = True

    def set_callbacks(
        self,
        *,
        state_callback: StateCallback,
        feedback_callback: FeedbackCallback,
        observation_health_callback: ObservationHealthCallback | None = None,
    ) -> None:
        if self._started:
            raise RosAdapterError("callbacks must be configured before the ROS adapter starts")
        self._state_callback = state_callback
        self._feedback_callback = feedback_callback
        self._observation_health_callback = observation_health_callback

    def _spin(self) -> None:
        self._owner_thread_id = threading.get_ident()
        try:
            context = Context()
            rclpy.init(args=None, context=context)
            node = Node(self._node_name, context=context)
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            self._context = context
            self._node = node
            self._executor = executor
            self._create_state_sources()
            self._create_heartbeat()
            self._create_observation_bridge()
            self._create_motion_safety_fault_subscription()
        except BaseException as exc:
            self._startup_error = exc
            self._running.clear()
        finally:
            self._startup_ready.set()
        try:
            while self._running.is_set():
                self._drain_requests()
                executor = self._require_executor()
                executor.spin_once(timeout_sec=0.05)
        except BaseException as exc:
            self._thread_error = exc
            self._running.clear()
        finally:
            with self._lifecycle_lock:
                self._accepting_requests = False
                self._running.clear()
                self._fail_requests(RosAdapterError("ROS executor thread stopped"))
            self._cleanup_ros()

    def _drain_requests(self) -> None:
        while True:
            try:
                callback, future = self._requests.get_nowait()
            except queue.Empty:
                return
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = callback()
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def _fail_requests(self, error: BaseException) -> None:
        while True:
            try:
                _, future = self._requests.get_nowait()
            except queue.Empty:
                return
            if future.set_running_or_notify_cancel():
                future.set_exception(error)

    def _cleanup_ros(self) -> None:
        executor = self._executor
        node = self._node
        context = self._context
        try:
            if executor is not None:
                executor.shutdown(timeout_sec=2.0)
            for client in self._action_clients.values():
                client.destroy()
            if node is not None:
                node.destroy_node()
            if context is not None and context.ok():
                context.shutdown()
        except BaseException as exc:
            if self._thread_error is None:
                self._thread_error = exc

    def configure_state_sources(self, sources: Mapping[str, StateSourceConfig]) -> None:
        if self._closed:
            raise RosAdapterError("ROS adapter is closed")
        if self._started:
            raise RosAdapterError("state sources must be configured before the ROS adapter starts")
        for source in sources.values():
            try:
                get_message(source.ros_type)
            except (ImportError, ModuleNotFoundError, ValueError) as exc:
                raise RosTypeError(
                    f"unable to load state source type {source.ros_type!r}: {exc}"
                ) from exc
        self._configured_state_sources = dict(sources)

    def configure_motion_safety_fault(
        self,
        *,
        config: MotionSafetyFaultConfig,
        callback: MotionSafetyFaultCallback,
    ) -> None:
        if self._closed:
            raise RosAdapterError("ROS adapter is closed")
        if self._started:
            raise RosAdapterError(
                "motion-safety-fault subscription must be configured before startup"
            )
        if self._motion_fault_configuration is not None:
            raise RosAdapterError("motion-safety-fault subscription is already configured")
        if config.enabled:
            self._motion_fault_configuration = (config, callback)

    def _create_motion_safety_fault_subscription(self) -> None:
        configuration = self._motion_fault_configuration
        if configuration is None:
            return
        config, callback = configuration
        group = ReentrantCallbackGroup()
        self._callback_groups.append(group)

        def on_message(message: String) -> None:
            # Capture T0 as the first operation in the callback.  Parsing,
            # SQLite and the asyncio bridge are all charged against this time.
            received_monotonic_ns = time.monotonic_ns()
            callback(message.data, received_monotonic_ns)

        subscription = self._require_node().create_subscription(
            String,
            config.topic,
            on_message,
            motion_safety_fault_qos(),
            callback_group=group,
        )
        self._subscriptions.append(subscription)

    def _create_state_sources(self) -> None:
        for logical_path, source in self._configured_state_sources.items():
            message_type = get_message(source.ros_type)
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
                    self._require_node().get_logger().error(
                        f"state source {path!r} conversion failed: {exc}"
                    )

            subscription = self._require_node().create_subscription(
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
        if self._started:
            raise RosAdapterError(
                "safety heartbeat must be configured before the ROS adapter starts"
            )
        if self._heartbeat_configuration is not None:
            raise RosAdapterError("safety heartbeat is already configured")
        self._heartbeat_configuration = (topic, period_ms, state_supplier)

    def configure_observation(self, *, demand_topic: str, health_topic: str) -> None:
        if self._started:
            raise RosAdapterError(
                "observation bridge must be configured before the ROS adapter starts"
            )
        if self._observation_configuration is not None:
            raise RosAdapterError("observation bridge is already configured")
        try:
            from robot_observation_interfaces.msg import ObservationDemand, ObservationHealth
        except ImportError as exc:
            raise RosTypeError(
                "robot_observation_interfaces is unavailable; build and source the ROS workspace"
            ) from exc
        self._observation_configuration = (
            demand_topic,
            health_topic,
            ObservationDemand,
            ObservationHealth,
        )

    def _create_observation_bridge(self) -> None:
        if self._observation_configuration is None:
            return
        demand_topic, health_topic, demand_type, health_type = self._observation_configuration
        demand_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        health_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        callback_group = ReentrantCallbackGroup()
        self._callback_groups.append(callback_group)
        self._observation_demand_publisher = self._require_node().create_publisher(
            demand_type,
            demand_topic,
            demand_qos,
            callback_group=callback_group,
        )

        def on_health(message: Any) -> None:
            callback = self._observation_health_callback
            if callback is None:
                return
            callback(
                ObservationHealthSample(
                    robot_id=str(message.robot_id),
                    robot_boot_id=str(message.robot_boot_id),
                    gateway_instance_id=str(message.gateway_instance_id),
                    observation_session_id=str(message.observation_session_id),
                    sensor_id=str(message.sensor_id),
                    scene_version=str(message.scene_version),
                    localization_epoch_id=str(message.localization_epoch_id),
                    status=int(message.status),
                    reason=int(message.reason),
                    has_last_accepted_sequence=bool(message.has_last_accepted_sequence),
                    last_accepted_sequence=int(message.last_accepted_sequence),
                    last_accepted_monotonic_ns=int(message.last_accepted_monotonic_ns),
                    capture_age_ms=float(message.capture_age_ms),
                    ack_age_ms=float(message.ack_age_ms),
                    drop_count=int(message.drop_count),
                    heartbeat_sequence=int(message.heartbeat_sequence),
                )
            )

        subscription = self._require_node().create_subscription(
            health_type,
            health_topic,
            on_health,
            health_qos,
            callback_group=callback_group,
        )
        self._subscriptions.append(subscription)

    def _create_heartbeat(self) -> None:
        if self._heartbeat_configuration is None:
            return
        topic, period_ms, state_supplier = self._heartbeat_configuration
        callback_group = MutuallyExclusiveCallbackGroup()
        self._callback_groups.append(callback_group)
        self._heartbeat_publisher = self._require_node().create_publisher(
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

        self._heartbeat_timer = self._require_node().create_timer(
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
        values = params
        if command.goal_transform == "navigate_to_pose_v1":
            values = await self._run_on_ros_thread(lambda: self._navigate_to_pose_values(params))
        elif command.goal_transform == "nav2_spin_v2":
            values = nav2_spin_goal_values(params)
        return await self._execute_action(
            command_id,
            ros_name=command.ros_name,
            ros_type=command.ros_type,
            values=values,
        )

    def _navigate_to_pose_values(self, params: dict[str, Any]) -> dict[str, Any]:
        """Translate the reviewed H-IDE map target into only a Nav2 PoseStamped."""
        self._require_owner_thread()
        goal_pose = params.get("goalPose")
        if not isinstance(goal_pose, dict):
            raise RosTypeError("navigation goalPose must be an object")
        stamp = self._require_node().get_clock().now().to_msg()
        return navigate_to_pose_goal_values(
            goal_pose,
            stamp_sec=int(stamp.sec),
            stamp_nanosec=int(stamp.nanosec),
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

    async def control_arm(self, config: SafeStopConfig) -> ExecutionResult:
        """Request one local, measured rearm after remote control.acquire."""

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

    async def publish_observation_demand(
        self,
        *,
        command_id: UUID,
        active: bool,
        observation_session_id: str,
        sensor_id: str,
        scene_version: str,
        localization_epoch_id: str,
    ) -> None:
        self._require_started()

        def publish() -> None:
            self._require_owner_thread()
            if self._observation_configuration is None:
                raise RosAdapterError("observation bridge is not configured")
            publisher = self._observation_demand_publisher
            if publisher is None:
                raise RosAdapterError("observation demand publisher is unavailable")
            demand_type = self._observation_configuration[2]
            message = demand_type()
            message.emitted_at = self._require_node().get_clock().now().to_msg()
            self._observation_demand_sequence += 1
            message.demand_sequence = self._observation_demand_sequence
            message.command_id = str(command_id)
            message.active = active
            message.observation_session_id = observation_session_id
            message.sensor_id = sensor_id
            message.scene_version = scene_version
            message.localization_epoch_id = localization_epoch_id
            publisher.publish(message)

        await self._run_on_ros_thread(publish)

    async def _execute_service(
        self,
        command_id: UUID,
        *,
        ros_name: str,
        ros_type: str,
        values: dict[str, Any],
    ) -> ExecutionResult:
        ros_future: Any | None = None
        begin_request: concurrent.futures.Future[Any] | None = None
        with self._goal_lock:
            self._service_waiters.add(command_id)
        try:
            client = await self._run_on_ros_thread(lambda: self._service_client(ros_name, ros_type))
            await self._wait_for_ros_ready(client.service_is_ready, f"service {ros_name}")
            request = build_message(ros_type, section="request", values=values)
            begin_request = self._submit_ros_call(
                lambda: self._begin_service_call(command_id, client, request)
            )
            ros_future = await asyncio.wrap_future(begin_request)
            response = await _await_ros_future(ros_future, cancel_source=False)
        finally:
            with self._goal_lock:
                self._service_waiters.discard(command_id)
                if ros_future is not None and self._pending.get(command_id) is ros_future:
                    self._pending.pop(command_id, None)
                service_started = self._pending.get(command_id) is not None
                if (
                    ros_future is None
                    and not service_started
                    and (begin_request is None or begin_request.cancelled() or begin_request.done())
                ):
                    self._cancel_requested.discard(command_id)
        result = _message_dict(response)
        succeeded = bool(getattr(response, "success", True))
        detail = str(getattr(response, "message", "service completed"))
        return ExecutionResult(succeeded=succeeded, detail=detail, result=result)

    def _begin_service_call(self, command_id: UUID, client: Any, request: Any) -> Any:
        self._require_owner_thread()
        with self._goal_lock:
            if command_id in self._cancel_requested:
                raise RosAdapterError("service dispatch was cancelled before the ROS call")
            ros_future = client.call_async(request)
            self._pending[command_id] = ros_future
        ros_future.add_done_callback(
            lambda completed: self._service_call_completed(command_id, completed)
        )
        return ros_future

    def _service_call_completed(self, command_id: UUID, completed: Any) -> None:
        with contextlib.suppress(Exception):
            completed.result()
        with self._goal_lock:
            cancelled = command_id in self._cancel_requested
        if cancelled:
            self._cleanup_command_tracking(command_id)

    async def _execute_action(
        self,
        command_id: UUID,
        *,
        ros_name: str,
        ros_type: str,
        values: dict[str, Any],
    ) -> ExecutionResult:
        send_future: Any | None = None
        result_future: Any | None = None
        send_request: concurrent.futures.Future[Any] | None = None
        with self._goal_lock:
            self._action_waiters.add(command_id)
        try:
            client = await self._run_on_ros_thread(lambda: self._action_client(ros_name, ros_type))
            await self._wait_for_ros_ready(client.server_is_ready, f"action {ros_name}")
            goal = build_message(ros_type, section="goal", values=values)

            def on_feedback(feedback_message: Any) -> None:
                if self._feedback_callback is not None:
                    self._feedback_callback(
                        command_id,
                        _message_dict(feedback_message.feedback),
                    )

            send_request = self._submit_ros_call(
                lambda: self._begin_action_send(command_id, client, goal, on_feedback)
            )
            send_future = await asyncio.wrap_future(send_request)
            goal_handle = await _await_ros_future(send_future, cancel_source=False)
            if not goal_handle.accepted:
                return ExecutionResult(
                    succeeded=False,
                    detail="action goal REJECTED",
                    result=None,
                )
            with self._goal_lock:
                self._goals[command_id] = goal_handle
            ros_goal_id = _goal_id_string(goal_handle)
            result_future = await self._run_on_ros_thread(
                lambda: self._begin_action_result(command_id, goal_handle)
            )
            response = await _await_ros_future(result_future, cancel_source=False)
            status_name = _goal_status_name(response.status)
            return ExecutionResult(
                succeeded=response.status == GoalStatus.STATUS_SUCCEEDED,
                detail=f"action finished with status {status_name}",
                result={
                    "rosGoalId": ros_goal_id,
                    "actionResult": _message_dict(response.result),
                },
            )
        except asyncio.CancelledError:
            self._request_cancel_without_waiting(command_id)
            raise
        finally:
            with self._goal_lock:
                self._action_waiters.discard(command_id)
                send_finished = send_future is not None and send_future.done()
                cancellation_pending = command_id in self._cancel_requested
                pending = self._pending.get(command_id)
                dispatch_can_still_start = (
                    send_request is not None
                    and not send_request.cancelled()
                    and not send_request.done()
                )
            if result_future is not None and cancellation_pending and not result_future.done():
                result_future.add_done_callback(
                    lambda completed: self._late_action_completed(command_id, completed)
                )
            elif send_finished or (pending is None and not dispatch_can_still_start):
                self._cleanup_command_tracking(command_id)

    async def cancel(self, command_id: UUID) -> None:
        self._require_started()
        with self._goal_lock:
            self._cancel_requested.add(command_id)
            goal_handle = self._goals.get(command_id)
            pending = self._pending.get(command_id)
            pending_action = command_id in self._pending_actions
            service_waiting = command_id in self._service_waiters
            action_waiting = command_id in self._action_waiters
        if goal_handle is None and pending is not None and not pending_action:
            raise RosAdapterError(
                "ROS service execution cannot be cancellation-acknowledged after dispatch"
            )
        if goal_handle is None and pending is not None and pending_action:
            try:
                candidate = await _await_ros_future(pending, cancel_source=False)
            except Exception:
                self._cleanup_command_tracking(command_id)
                raise
            if getattr(candidate, "accepted", False):
                goal_handle = candidate
            else:
                self._cleanup_command_tracking(command_id)
                return
        if goal_handle is not None:
            cancel_future = await self._run_on_ros_thread(
                lambda: self._begin_goal_cancel(command_id, goal_handle)
            )
            if cancel_future is not None:
                response = await _await_ros_future(cancel_future, cancel_source=False)
                if not response.goals_canceling:
                    raise RosAdapterError("ROS action server rejected goal cancellation")
                result_future = await self._run_on_ros_thread(
                    lambda: self._begin_action_result(command_id, goal_handle)
                )
                terminal = await _await_ros_future(result_future, cancel_source=False)
                if terminal.status not in _TERMINAL_GOAL_STATUSES:
                    raise RosAdapterError(
                        f"ROS action cancellation ended in non-terminal status {terminal.status}"
                    )
        else:
            with self._goal_lock:
                if not service_waiting and not action_waiting:
                    self._cancel_requested.discard(command_id)

    async def cancel_nav2_for_motion_fault(
        self,
        command_id: UUID,
        *,
        received_monotonic_ns: int,
        cancel_begin_deadline_ms: int = 100,
        nav2_terminal_deadline_ms: int = 300,
    ) -> MotionFaultCancelEvidence:
        """Cancel exactly one action goal under the fault-specific deadline.

        This deliberately does not use the ROS ``CancelGoal`` service and has
        no "cancel all" fallback.  A command without an owned goal is treated
        as prevented-before-dispatch only while its local action coroutine is
        still registered.
        """

        if cancel_begin_deadline_ms != 100 or nav2_terminal_deadline_ms != 300:
            raise ValueError("motion-fault deadlines are fixed at 100 ms / 300 ms")
        self._require_started()
        begin_ns = time.monotonic_ns()
        begin_limit_ns = received_monotonic_ns + cancel_begin_deadline_ms * 1_000_000
        terminal_limit_ns = (
            received_monotonic_ns + nav2_terminal_deadline_ms * 1_000_000
        )
        ros_call_ns: int | None = None

        def evidence(
            outcome: str,
            *,
            terminal_ns: int | None = None,
        ) -> MotionFaultCancelEvidence:
            return MotionFaultCancelEvidence(
                command_id=command_id,
                received_monotonic_ns=received_monotonic_ns,
                cancel_begin_monotonic_ns=begin_ns,
                ros_cancel_call_monotonic_ns=(
                    ros_call_ns
                    if ros_call_ns is not None
                    else self._cancel_started_monotonic_ns.get(command_id)
                ),
                nav2_terminal_monotonic_ns=terminal_ns,
                outcome=outcome,
            )

        if begin_ns > begin_limit_ns:
            proof = evidence("cancel_begin_deadline_missed")
            raise MotionFaultCancellationError(
                "motion-fault cancellation did not begin within 100 ms",
                proof,
            )
        with self._goal_lock:
            self._cancel_requested.add(command_id)
            goal_handle = self._goals.get(command_id)
            pending = self._pending.get(command_id)
            pending_action = command_id in self._pending_actions
            action_waiting = command_id in self._action_waiters

        if goal_handle is None and pending is not None and not pending_action:
            # A result future can already be installed before _goals becomes
            # visible only during teardown.  It still belongs to this command;
            # wait for its exact terminal rather than touching another goal.
            if not action_waiting:
                terminal_ns = time.monotonic_ns()
                return evidence("already_terminal", terminal_ns=terminal_ns)
            proof = evidence("goal_ownership_ambiguous")
            raise MotionFaultCancellationError(
                "active command has no provable Nav2 goal ownership",
                proof,
            )

        if goal_handle is None and pending is not None and pending_action:
            try:
                candidate = await _await_ros_future_before(
                    pending,
                    absolute_deadline_ns=terminal_limit_ns,
                )
            except Exception as exc:
                proof = evidence("pending_goal_acceptance_failed")
                raise MotionFaultCancellationError(
                    f"pending Nav2 goal acceptance failed: {type(exc).__name__}: {exc}",
                    proof,
                ) from exc
            if not getattr(candidate, "accepted", False):
                terminal_ns = time.monotonic_ns()
                self._cleanup_command_tracking(command_id)
                return evidence("goal_rejected_before_execution", terminal_ns=terminal_ns)
            goal_handle = candidate

        if goal_handle is None:
            terminal_ns = time.monotonic_ns()
            if action_waiting:
                # _begin_action_send observes _cancel_requested under the same
                # lock, proving this command can no longer dispatch a goal.
                return evidence("prevented_before_dispatch", terminal_ns=terminal_ns)
            with self._goal_lock:
                self._cancel_requested.discard(command_id)
            return evidence("already_terminal", terminal_ns=terminal_ns)

        status = getattr(goal_handle, "status", None)
        if status in _TERMINAL_GOAL_STATUSES:
            return evidence("already_terminal", terminal_ns=time.monotonic_ns())

        cancel_future = await self._run_on_ros_thread(
            lambda: self._begin_goal_cancel(command_id, goal_handle)
        )
        with self._goal_lock:
            ros_call_ns = self._cancel_started_monotonic_ns.get(command_id)
        if ros_call_ns is None or ros_call_ns > begin_limit_ns:
            proof = evidence("ros_cancel_begin_deadline_missed")
            raise MotionFaultCancellationError(
                "exact Nav2 cancellation did not begin within 100 ms",
                proof,
            )
        cancel_rejected = False
        if cancel_future is not None:
            try:
                response = await _await_ros_future_before(
                    cancel_future,
                    absolute_deadline_ns=terminal_limit_ns,
                )
            except Exception as exc:
                proof = evidence("cancel_ack_deadline_or_failure")
                raise MotionFaultCancellationError(
                    f"exact Nav2 cancel acknowledgement failed: {type(exc).__name__}: {exc}",
                    proof,
                ) from exc
            if not response.goals_canceling:
                # A goal can become terminal between our ownership snapshot
                # and Nav2 processing the exact cancel request.  The result
                # future below distinguishes that benign race from a live
                # cancellation rejection without broadening the target.
                cancel_rejected = True

        result_future = await self._run_on_ros_thread(
            lambda: self._begin_action_result(command_id, goal_handle)
        )
        try:
            terminal = await _await_ros_future_before(
                result_future,
                absolute_deadline_ns=terminal_limit_ns,
            )
        except Exception as exc:
            proof = evidence("nav2_terminal_deadline_or_failure")
            raise MotionFaultCancellationError(
                f"Nav2 did not reach terminal within 300 ms: {type(exc).__name__}: {exc}",
                proof,
            ) from exc
        terminal_ns = time.monotonic_ns()
        if terminal.status not in _TERMINAL_GOAL_STATUSES or terminal_ns > terminal_limit_ns:
            proof = evidence("nav2_terminal_deadline_missed", terminal_ns=terminal_ns)
            raise MotionFaultCancellationError(
                "Nav2 terminal proof exceeded the 300 ms fault deadline",
                proof,
            )
        return evidence(
            "terminal_raced_cancel" if cancel_rejected else "nav2_terminal",
            terminal_ns=terminal_ns,
        )

    def _begin_action_send(
        self,
        command_id: UUID,
        client: Any,
        goal: Any,
        feedback_callback: Callable[[Any], None],
    ) -> Any:
        self._require_owner_thread()
        with self._goal_lock:
            if command_id in self._cancel_requested:
                raise RosAdapterError("action dispatch was cancelled before the ROS goal")
            send_future = client.send_goal_async(goal, feedback_callback=feedback_callback)
            self._pending[command_id] = send_future
            self._pending_actions.add(command_id)
        send_future.add_done_callback(
            lambda completed: self._action_goal_response(command_id, completed)
        )
        return send_future

    def _action_goal_response(self, command_id: UUID, completed: Any) -> None:
        try:
            goal_handle = completed.result()
        except Exception:
            self._cleanup_command_tracking(command_id)
            return
        if not goal_handle.accepted:
            self._cleanup_command_tracking(command_id)
            return
        with self._goal_lock:
            self._goals[command_id] = goal_handle
            cancel_requested = command_id in self._cancel_requested
        if cancel_requested:
            self._request_cancel_without_waiting(command_id)

    def _begin_action_result(self, command_id: UUID, goal_handle: Any) -> Any:
        self._require_owner_thread()
        with self._goal_lock:
            existing = self._pending.get(command_id)
            if existing is not None and command_id not in self._pending_actions:
                return existing
        result_future = goal_handle.get_result_async()
        with self._goal_lock:
            self._pending[command_id] = result_future
            self._pending_actions.discard(command_id)
            self._service_waiters.discard(command_id)
        return result_future

    def _begin_goal_cancel(self, command_id: UUID, goal_handle: Any) -> Any:
        self._require_owner_thread()
        with self._goal_lock:
            existing = self._cancel_futures.get(command_id)
            if existing is not None:
                return existing
            self._goals[command_id] = goal_handle
        started_ns = time.monotonic_ns()
        cancel_future = goal_handle.cancel_goal_async()
        with self._goal_lock:
            self._cancel_futures[command_id] = cancel_future
            self._cancel_started_monotonic_ns[command_id] = started_ns
        return cancel_future

    def _request_cancel_without_waiting(self, command_id: UUID) -> None:
        with self._goal_lock:
            self._cancel_requested.add(command_id)
            goal_handle = self._goals.get(command_id)
            existing = self._cancel_futures.get(command_id)
        if goal_handle is None:
            return
        if existing is not None:
            with self._goal_lock:
                orphaned = command_id not in self._action_waiters
            existing.add_done_callback(
                lambda completed: self._late_cancel_completed(
                    command_id,
                    completed,
                    cleanup=orphaned,
                )
            )
            return
        try:
            submitted = self._submit_ros_call(
                lambda: self._begin_goal_cancel(command_id, goal_handle)
            )
        except RosAdapterError:
            return

        def cancel_submitted(completed: concurrent.futures.Future[Any]) -> None:
            try:
                cancel_future = completed.result()
            except Exception:
                return
            with self._goal_lock:
                orphaned = command_id not in self._action_waiters
            if cancel_future is not None:
                cancel_future.add_done_callback(
                    lambda source: self._late_cancel_completed(
                        command_id,
                        source,
                        cleanup=orphaned,
                    )
                )

        submitted.add_done_callback(cancel_submitted)

    def _late_cancel_completed(
        self,
        command_id: UUID,
        completed: Any,
        *,
        cleanup: bool = True,
    ) -> None:
        try:
            response = completed.result()
        except Exception:
            if cleanup:
                self._cleanup_command_tracking(command_id)
            return
        if not cleanup:
            return
        with self._goal_lock:
            goal_handle = self._goals.get(command_id)
        if not response.goals_canceling or goal_handle is None:
            self._cleanup_command_tracking(command_id)
            return
        try:
            result_future = self._begin_action_result(command_id, goal_handle)
        except Exception:
            self._cleanup_command_tracking(command_id)
            return
        result_future.add_done_callback(
            lambda source: self._late_action_completed(command_id, source)
        )

    def _late_action_completed(self, command_id: UUID, completed: Any) -> None:
        with contextlib.suppress(Exception):
            completed.result()
        self._cleanup_command_tracking(command_id)

    def _cleanup_command_tracking(self, command_id: UUID) -> None:
        with self._goal_lock:
            self._goals.pop(command_id, None)
            self._pending.pop(command_id, None)
            self._pending_actions.discard(command_id)
            self._cancel_requested.discard(command_id)
            self._cancel_futures.pop(command_id, None)
            self._cancel_started_monotonic_ns.pop(command_id, None)

    def _service_client(self, ros_name: str, ros_type: str) -> Any:
        self._require_owner_thread()
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
        client = self._require_node().create_client(
            interface,
            ros_name,
            qos_profile=service_qos(),
            callback_group=group,
        )
        self._service_clients[key] = client
        return client

    def _action_client(self, ros_name: str, ros_type: str) -> Any:
        self._require_owner_thread()
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
            self._require_node(),
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
        self.check_health()

    def _require_owner_thread(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RosAdapterError("ROS entity API called outside its owner thread")

    def _require_node(self) -> Node:
        if self._node is None:
            raise RosAdapterError("ROS node is not initialized")
        return self._node

    def _require_executor(self) -> SingleThreadedExecutor:
        if self._executor is None:
            raise RosAdapterError("ROS executor is not initialized")
        return self._executor

    def check_health(self) -> None:
        if self._thread_error is not None:
            raise RosAdapterError(
                f"ROS executor failed: {self._thread_error}"
            ) from self._thread_error
        if self._started and not self._thread.is_alive():
            raise RosAdapterError("ROS executor thread is not running")
        if self._started and not self._accepting_requests:
            raise RosAdapterError("ROS executor is not accepting requests")

    async def nav2_action_readiness(
        self,
        commands: Mapping[str, CommandConfig],
    ) -> dict[str, RosActionReadiness]:
        """Probe configured Nav2 action servers without dispatching a goal."""
        self._require_started()
        results: dict[str, RosActionReadiness] = {}
        for skill_id, command in sorted(commands.items()):
            if command.goal_transform not in {"navigate_to_pose_v1", "nav2_spin_v2"}:
                continue
            try:
                ready = bool(
                    await self._run_on_ros_thread(
                        partial(self._nav2_action_server_is_ready, command)
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                results[skill_id] = RosActionReadiness(
                    ready=False,
                    reason=f"probe_failed:{type(exc).__name__}",
                )
            else:
                results[skill_id] = RosActionReadiness(
                    ready=ready,
                    reason=None if ready else "action_server_unavailable",
                )
        return results

    async def safety_endpoint_readiness(
        self,
        endpoints: Mapping[str, SafeStopConfig],
    ) -> dict[str, RosActionReadiness]:
        """Probe safety services/actions without dispatching a request or goal."""

        self._require_started()
        results: dict[str, RosActionReadiness] = {}
        for name, config in sorted(endpoints.items()):
            try:
                ready = bool(
                    await self._run_on_ros_thread(
                        partial(self._safety_endpoint_is_ready, config)
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                results[name] = RosActionReadiness(
                    ready=False,
                    reason=f"probe_failed:{type(exc).__name__}",
                )
            else:
                results[name] = RosActionReadiness(
                    ready=ready,
                    reason=None if ready else "endpoint_unavailable",
                )
        return results

    def _safety_endpoint_is_ready(self, config: SafeStopConfig) -> bool:
        self._require_owner_thread()
        if config.adapter == "service":
            return bool(
                self._service_client(config.ros_name, config.ros_type).service_is_ready()
            )
        return bool(
            self._action_client(config.ros_name, config.ros_type).server_is_ready()
        )

    def _nav2_action_server_is_ready(self, command: CommandConfig) -> bool:
        self._require_owner_thread()
        return bool(self._action_client(command.ros_name, command.ros_type).server_is_ready())

    def _submit_ros_call(self, callback: Callable[[], Any]) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        with self._lifecycle_lock:
            self._require_started()
            if not self._accepting_requests:
                raise RosAdapterError("ROS executor is not accepting requests")
            if threading.get_ident() == self._owner_thread_id:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(callback())
                    except BaseException as exc:
                        future.set_exception(exc)
                return future
            try:
                self._requests.put_nowait((callback, future))
            except queue.Full as exc:
                raise RosAdapterError("ROS request queue is full") from exc
            self._wake_executor()
        return future

    async def _run_on_ros_thread(self, callback: Callable[[], Any]) -> Any:
        return await asyncio.wrap_future(self._submit_ros_call(callback))

    async def _wait_for_ros_ready(
        self,
        predicate: Callable[[], bool],
        description: str,
        *,
        timeout: float = 5.0,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not await self._run_on_ros_thread(predicate):
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RosAdapterError(f"timed out waiting for {description}")
            await asyncio.sleep(min(0.05, remaining))

    def _wake_executor(self) -> None:
        executor = self._executor
        if executor is not None:
            executor.wake()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.to_thread(self._shutdown)

    def _shutdown(self) -> None:
        with self._lifecycle_lock:
            self._accepting_requests = False
            self._running.clear()
            self._fail_requests(RosAdapterError("ROS adapter is shutting down"))
            self._wake_executor()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RosAdapterError("ROS executor thread did not stop within 2 seconds")

    @property
    def node(self) -> Node:
        return self._require_node()


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


async def _await_ros_future(future: Any, *, cancel_source: bool = True) -> Any:
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
        if cancel_source:
            future.cancel()
        raise


async def _await_ros_future_before(
    future: Any,
    *,
    absolute_deadline_ns: int,
) -> Any:
    """Await a ROS future without extending or cancelling its absolute deadline."""

    remaining = (absolute_deadline_ns - time.monotonic_ns()) / 1_000_000_000
    if remaining <= 0:
        raise asyncio.TimeoutError("absolute monotonic deadline already expired")
    return await asyncio.wait_for(
        _await_ros_future(future, cancel_source=False),
        timeout=remaining,
    )


def _set_result_if_pending(future: asyncio.Future[Any], value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception_if_pending(future: asyncio.Future[Any], exception: Exception) -> None:
    if not future.done():
        future.set_exception(exception)


def _message_dict(message: Any) -> dict[str, Any]:
    converted = message_to_ordereddict(message)
    return cast(dict[str, Any], json.loads(json.dumps(converted, allow_nan=False)))


def _goal_id_string(goal_handle: Any) -> str | None:
    goal_id = getattr(goal_handle, "goal_id", None)
    raw = getattr(goal_id, "uuid", None)
    if raw is None:
        return None
    try:
        return str(UUID(bytes=bytes(raw)))
    except (TypeError, ValueError):
        return None


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
