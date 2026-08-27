"""ROS 2 ingress for synchronized RealSense RGB-D, metadata, TF, and localization."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from robot_observation_gateway.calibration import calibration_sha256
from robot_observation_gateway.config import GatewayConfig
from robot_observation_gateway.errors import DropReason, FrameValidationError, GatewayError
from robot_observation_gateway.health import (
    GatewayHealth,
    ObservationDemandSnapshot,
)
from robot_observation_gateway.latest_slot import LatestSlot
from robot_observation_gateway.metrics import DropTracker
from robot_observation_gateway.model import (
    Calibration,
    CameraIntrinsics,
    LocalizationEstimate,
    Quaternion,
    RawObservation,
    RegionOfInterest,
    RigidTransform,
    SensorMetadata,
    TimePoint,
    Vector3,
)

T = TypeVar("T")

@dataclass(frozen=True)
class _Cached(Generic[T]):
    stamp_ns: int
    arrived_monotonic_ns: int
    value: T


class _TimedCache(Generic[T]):
    def __init__(self, *, max_size: int, ttl_ms: int) -> None:
        self._items: deque[_Cached[T]] = deque(maxlen=max_size)
        self._ttl_ns = ttl_ms * 1_000_000
        self._condition = threading.Condition()

    def add(self, *, stamp_ns: int, value: T) -> None:
        now_ns = time.monotonic_ns()
        with self._condition:
            self._items.append(_Cached(stamp_ns, now_ns, value))
            self._prune(now_ns)
            self._condition.notify_all()

    def nearest(self, *, stamp_ns: int, max_delta_ms: int, wait_ms: int) -> T | None:
        deadline_ns = time.monotonic_ns() + wait_ms * 1_000_000
        max_delta_ns = max_delta_ms * 1_000_000
        with self._condition:
            while True:
                self._prune(time.monotonic_ns())
                if self._items:
                    candidate = min(self._items, key=lambda item: abs(item.stamp_ns - stamp_ns))
                    if abs(candidate.stamp_ns - stamp_ns) <= max_delta_ns:
                        return candidate.value
                remaining_ns = deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    return None
                self._condition.wait(remaining_ns / 1_000_000_000.0)

    def latest_at_or_before(self, *, stamp_ns: int, max_age_ms: int) -> T | None:
        max_age_ns = max_age_ms * 1_000_000
        with self._condition:
            self._prune(time.monotonic_ns())
            candidates = [item for item in self._items if item.stamp_ns <= stamp_ns]
            if not candidates:
                return None
            candidate = max(candidates, key=lambda item: item.stamp_ns)
            if stamp_ns - candidate.stamp_ns > max_age_ns:
                return None
            return candidate.value

    def _prune(self, now_ns: int) -> None:
        while self._items and now_ns - self._items[0].arrived_monotonic_ns > self._ttl_ns:
            self._items.popleft()


class _DropFrame(Exception):
    def __init__(self, reason: DropReason, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def time_point_from_ros(stamp: Any) -> TimePoint:
    return TimePoint(sec=int(stamp.sec), nanosec=int(stamp.nanosec))


def parse_realsense_metadata(message: Any) -> SensorMetadata:
    try:
        raw = json.loads(str(message.json_data))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrameValidationError("RealSense metadata is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise FrameValidationError("RealSense metadata JSON must be an object")
    frame_number = raw.get("frame_number")
    frame_timestamp = raw.get("frame_timestamp")
    clock_domain = raw.get("clock_domain")
    if type(frame_number) is not int or not isinstance(frame_timestamp, (int, float)):
        raise FrameValidationError("RealSense metadata lacks frame_number/frame_timestamp")
    if not isinstance(clock_domain, str):
        raise FrameValidationError("RealSense metadata lacks clock_domain")
    return SensorMetadata(
        frame_number=frame_number,
        sensor_timestamp_ms=float(frame_timestamp),
        clock_domain=clock_domain.strip().lower(),
        header_stamp=time_point_from_ros(message.header.stamp),
    )


def camera_intrinsics_from_ros(message: Any) -> CameraIntrinsics:
    matrix = tuple(float(value) for value in message.k)
    if len(matrix) != 9:
        raise FrameValidationError("CameraInfo.k must contain 9 values")
    roi = message.roi
    return CameraIntrinsics(
        width=int(message.width),
        height=int(message.height),
        fx=matrix[0],
        fy=matrix[4],
        cx=matrix[2],
        cy=matrix[5],
        distortion_model=str(message.distortion_model),
        distortion_coefficients=tuple(float(value) for value in message.d),
        binning_x=int(message.binning_x),
        binning_y=int(message.binning_y),
        roi=RegionOfInterest(
            x_offset=int(roi.x_offset),
            y_offset=int(roi.y_offset),
            width=int(roi.width),
            height=int(roi.height),
            do_rectify=bool(roi.do_rectify),
        ),
    )


def _image_bytes(message: Any, *, bytes_per_pixel: int) -> bytes:
    width = int(message.width)
    height = int(message.height)
    row_bytes = width * bytes_per_pixel
    step = int(message.step)
    data = bytes(message.data)
    if width <= 0 or height <= 0 or step < row_bytes or len(data) != step * height:
        raise FrameValidationError("ROS image dimensions, step, and data length disagree")
    if step == row_bytes:
        return data
    return b"".join(data[row * step : row * step + row_bytes] for row in range(height))


def _transform_from_ros(message: Any, *, parent: str, child: str) -> RigidTransform:
    transform = message.transform
    return RigidTransform(
        parent_frame_id=parent,
        child_frame_id=child,
        translation=Vector3(
            x=float(transform.translation.x),
            y=float(transform.translation.y),
            z=float(transform.translation.z),
        ),
        rotation=Quaternion(
            x=float(transform.rotation.x),
            y=float(transform.rotation.y),
            z=float(transform.rotation.z),
            w=float(transform.rotation.w),
        ),
    )


class RosObservationSource:
    """Owns ROS callbacks and emits only fully validated, newest-only observations."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        drops: DropTracker,
        health_provider: Callable[[], GatewayHealth],
    ) -> None:
        self._config = config
        self._drops = drops
        self._health_provider = health_provider
        self.raw_slot: LatestSlot[RawObservation] = LatestSlot()
        self._candidates: LatestSlot[Any] = LatestSlot()
        self._color_metadata: _TimedCache[SensorMetadata] = _TimedCache(
            max_size=config.stream.metadata_cache_size,
            ttl_ms=config.stream.metadata_ttl_ms,
        )
        self._depth_metadata: _TimedCache[SensorMetadata] = _TimedCache(
            max_size=config.stream.metadata_cache_size,
            ttl_ms=config.stream.metadata_ttl_ms,
        )
        self._localization: _TimedCache[Any] = _TimedCache(
            max_size=config.stream.localization_cache_size,
            ttl_ms=config.stream.localization_ttl_ms,
        )
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._calibration: Calibration | None = None
        self._demand: ObservationDemandSnapshot | None = None
        self._last_camera_monotonic_ns = 0
        self._last_drop_reason: DropReason | None = None
        self._last_drop_detail = ""
        self._localization_epoch_id = config.sensor.localization_epoch_id
        self._heartbeat_sequence = 0
        self._node: Any = None
        self._executor: Any = None
        self._tf_buffer: Any = None
        self._ros_time_type: Any = None
        self._duration_type: Any = None
        self._executor_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._owns_rclpy_context = False

    def start(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseWithCovarianceStamped
            from rclpy.duration import Duration
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from rclpy.time import Time
            from realsense2_camera_msgs.msg import RGBD, Metadata
            from std_msgs.msg import String
            from tf2_ros import Buffer, TransformListener

            from robot_observation_interfaces.msg import ObservationDemand, ObservationHealth
        except ImportError as exc:
            raise GatewayError(
                "ROS observation dependencies are unavailable; build/source the ROS workspace "
                "and install realsense2_camera_msgs"
            ) from exc

        if not rclpy.ok():  # type: ignore[attr-defined]
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        node = Node("robot_observation_gateway")
        self._node = node
        self._duration_type = Duration
        self._ros_time_type = Time
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, node, spin_thread=False)

        sensor_qos = QoSProfile(  # type: ignore[no-untyped-call]
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        metadata_qos = QoSProfile(  # type: ignore[no-untyped-call]
            history=HistoryPolicy.KEEP_LAST,
            depth=self._config.stream.metadata_cache_size,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        localization_qos = QoSProfile(  # type: ignore[no-untyped-call]
            history=HistoryPolicy.KEEP_LAST,
            depth=self._config.stream.localization_cache_size,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        demand_qos = QoSProfile(  # type: ignore[no-untyped-call]
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        health_qos = QoSProfile(  # type: ignore[no-untyped-call]
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._subscriptions = [
            node.create_subscription(
                RGBD,
                self._config.sensor.rgbd_topic,
                self._on_rgbd,
                sensor_qos,
            ),
            node.create_subscription(
                Metadata,
                self._config.sensor.color_metadata_topic,
                self._on_color_metadata,
                metadata_qos,
            ),
            node.create_subscription(
                Metadata,
                self._config.sensor.depth_metadata_topic,
                self._on_depth_metadata,
                metadata_qos,
            ),
            node.create_subscription(
                PoseWithCovarianceStamped,
                self._config.sensor.localization_topic,
                self._on_localization,
                localization_qos,
            ),
            node.create_subscription(
                String,
                self._config.sensor.localization_epoch_topic,
                self._on_localization_epoch,
                demand_qos,
            ),
            node.create_subscription(
                ObservationDemand,
                self._config.health_bridge.demand_topic,
                self._on_demand,
                demand_qos,
            ),
        ]
        self._health_message_type = ObservationHealth
        self._health_publisher = node.create_publisher(
            ObservationHealth,
            self._config.health_bridge.health_topic,
            health_qos,
        )
        self._health_timer = node.create_timer(
            self._config.health_bridge.heartbeat_period_ms / 1_000.0,
            self._publish_health,
        )

        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(node)
        self._executor_thread = threading.Thread(
            target=self._executor.spin,
            name="observation-ros-executor",
            daemon=True,
        )
        self._worker_thread = threading.Thread(
            target=self._enrichment_loop,
            name="observation-enrichment",
            daemon=True,
        )
        self._executor_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._candidates.close()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
        self.raw_slot.close()
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2.0)
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=2.0)
        if hasattr(self, "_tf_listener"):
            self._tf_listener.unregister()
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy_context:
            import rclpy

            if rclpy.ok():  # type: ignore[attr-defined]
                rclpy.shutdown()

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)

    @property
    def calibration(self) -> Calibration | None:
        with self._lock:
            return self._calibration

    @property
    def source_mode(self) -> str:
        # MuJoCo uses this same strict ROS ingress (capture-time metadata,
        # localization and TF), but its frames must be labelled synthetic in
        # the workstation protocol.  The ordinary RealSense path remains live.
        return "synthetic" if self._config.sensor.source_mode == "mujoco" else "live"

    @property
    def source_id(self) -> str:
        return self._config.sensor.source_id

    @property
    def localization_epoch_id(self) -> str:
        with self._lock:
            return self._localization_epoch_id

    @property
    def demand(self) -> ObservationDemandSnapshot | None:
        with self._lock:
            return self._demand

    @property
    def last_camera_age_ms(self) -> float:
        with self._lock:
            last = self._last_camera_monotonic_ns
        if last == 0:
            return float("inf")
        return (time.monotonic_ns() - last) / 1_000_000.0

    @property
    def last_drop(self) -> tuple[DropReason | None, str]:
        with self._lock:
            return self._last_drop_reason, self._last_drop_detail

    def _on_rgbd(self, message: Any) -> None:
        with self._lock:
            self._last_camera_monotonic_ns = time.monotonic_ns()
        if self._candidates.put(message):
            self._drops.increment(DropReason.RAW_OVERWRITTEN)

    def _on_color_metadata(self, message: Any) -> None:
        self._cache_metadata(message, self._color_metadata)

    def _on_depth_metadata(self, message: Any) -> None:
        self._cache_metadata(message, self._depth_metadata)

    def _cache_metadata(self, message: Any, cache: _TimedCache[SensorMetadata]) -> None:
        try:
            metadata = parse_realsense_metadata(message)
        except FrameValidationError as exc:
            self._drops.increment(DropReason.METADATA_MISSING)
            self._record_drop(DropReason.METADATA_MISSING, str(exc))
            return
        cache.add(stamp_ns=metadata.header_stamp.unix_ns, value=metadata)

    def _on_localization(self, message: Any) -> None:
        stamp = time_point_from_ros(message.header.stamp)
        self._localization.add(stamp_ns=stamp.unix_ns, value=message)

    def _on_localization_epoch(self, message: Any) -> None:
        epoch_id = str(message.data).strip()
        if not epoch_id:
            return
        with self._lock:
            self._localization_epoch_id = epoch_id

    def _on_demand(self, message: Any) -> None:
        snapshot = ObservationDemandSnapshot(
            demand_sequence=int(message.demand_sequence),
            command_id=str(message.command_id),
            active=bool(message.active),
            observation_session_id=str(message.observation_session_id),
            sensor_id=str(message.sensor_id),
            scene_version=str(message.scene_version),
            localization_epoch_id=str(message.localization_epoch_id),
            received_monotonic_ns=time.monotonic_ns(),
        )
        with self._lock:
            if self._demand is None or snapshot.demand_sequence >= self._demand.demand_sequence:
                self._demand = snapshot

    def _enrichment_loop(self) -> None:
        while not self._stop.is_set():
            candidate = self._candidates.wait_pop(timeout=0.2)
            if candidate is None:
                continue
            try:
                raw = self._enrich(candidate)
            except _DropFrame as exc:
                self._drops.increment(exc.reason)
                self._record_drop(exc.reason, str(exc))
                continue
            except FrameValidationError as exc:
                self._drops.increment(DropReason.INVALID_FRAME)
                self._record_drop(DropReason.INVALID_FRAME, str(exc))
                continue
            if self.raw_slot.put(raw):
                self._drops.increment(DropReason.RAW_OVERWRITTEN)
            with self._lock:
                self._calibration = raw.calibration
            self._ready.set()

    def _record_drop(self, reason: DropReason, detail: str) -> None:
        with self._lock:
            self._last_drop_reason = reason
            self._last_drop_detail = detail

    def _enrich(self, message: Any) -> RawObservation:
        capture = time_point_from_ros(message.header.stamp)
        if str(message.header.frame_id) != self._config.sensor.rgbd_frame:
            raise FrameValidationError(
                "RGBD frame_id does not match the configured synchronized RGB-D frame"
            )
        if (
            str(message.rgb_camera_info.header.frame_id) != self._config.sensor.camera_frame
            or str(message.depth_camera_info.header.frame_id) != self._config.sensor.camera_frame
        ):
            raise FrameValidationError(
                "aligned RGB/depth CameraInfo does not use the configured camera frame"
            )
        wait_ms = self._config.stream.metadata_max_delta_ms
        color_metadata = self._color_metadata.nearest(
            stamp_ns=capture.unix_ns,
            max_delta_ms=self._config.stream.metadata_max_delta_ms,
            wait_ms=wait_ms,
        )
        depth_metadata = self._depth_metadata.nearest(
            stamp_ns=capture.unix_ns,
            max_delta_ms=self._config.stream.metadata_max_delta_ms,
            wait_ms=wait_ms,
        )
        if color_metadata is None or depth_metadata is None:
            raise _DropFrame(DropReason.METADATA_MISSING, "matching sensor metadata unavailable")
        if color_metadata.clock_domain != depth_metadata.clock_domain:
            raise _DropFrame(DropReason.METADATA_MISSING, "sensor clock domains disagree")

        # The MuJoCo bridge emits one synchronized RGB-D bundle, while ROS
        # delivers the two auxiliary Metadata topics independently.  Under
        # CPU load a cache lookup can therefore pair the color metadata from
        # frame N with depth metadata from frame N-1 even though the RGB and
        # depth bytes in this bundle share the same capture stamp.  Keep the
        # strict timestamp check for live RealSense data; for the explicitly
        # synthetic MuJoCo source, derive both sensor timestamps from the
        # bundle's capture timestamp and preserve the source's clock domain.
        if self._config.sensor.source_mode == "mujoco":
            canonical_timestamp_ms = capture.unix_ns / 1_000_000.0
            canonical_metadata = SensorMetadata(
                frame_number=max(color_metadata.frame_number, depth_metadata.frame_number),
                sensor_timestamp_ms=canonical_timestamp_ms,
                clock_domain=color_metadata.clock_domain,
                header_stamp=capture,
            )
            color_metadata = canonical_metadata
            depth_metadata = canonical_metadata

        localization_message = self._localization.latest_at_or_before(
            stamp_ns=capture.unix_ns,
            max_age_ms=self._config.stream.localization_max_age_ms,
        )
        if localization_message is None:
            raise _DropFrame(
                DropReason.LOCALIZATION_MISSING,
                "localization at or before capture time is unavailable",
            )
        localization_stamp = time_point_from_ros(localization_message.header.stamp)
        localization_frame = str(localization_message.header.frame_id)
        if localization_frame != self._config.sensor.map_frame:
            raise _DropFrame(
                DropReason.LOCALIZATION_MISSING,
                "localization frame does not match configured map frame",
            )
        localization = LocalizationEstimate(
            observed_at=localization_stamp,
            frame_id=localization_frame,
            covariance=tuple(float(value) for value in localization_message.pose.covariance),
            age_ms=(capture.unix_ns - localization_stamp.unix_ns) / 1_000_000.0,
        )

        t_map_camera = self._lookup_transform(self._config.sensor.map_frame, capture)
        t_base_camera = self._lookup_transform(self._config.sensor.base_frame, capture)
        rgb_intrinsics = camera_intrinsics_from_ros(message.rgb_camera_info)
        depth_intrinsics = camera_intrinsics_from_ros(message.depth_camera_info)
        if (
            rgb_intrinsics.width != depth_intrinsics.width
            or rgb_intrinsics.height != depth_intrinsics.height
        ):
            raise FrameValidationError("aligned depth dimensions do not match RGB")
        calibration_digest = calibration_sha256(
            sensor_id=self._config.sensor.sensor_id,
            device_serial=self._config.sensor.device_serial,
            device_model=self._config.sensor.device_model,
            firmware_version=self._config.sensor.firmware_version,
            color_profile=self._config.sensor.color_profile,
            depth_profile=self._config.sensor.depth_profile,
            rgb_intrinsics=rgb_intrinsics,
            aligned_depth_intrinsics=depth_intrinsics,
            depth_scale_meters_per_unit=self._config.sensor.depth_scale_meters_per_unit,
        )
        calibration = Calibration(
            sensor_id=self._config.sensor.sensor_id,
            device_serial=self._config.sensor.device_serial,
            device_model=self._config.sensor.device_model,
            firmware_version=self._config.sensor.firmware_version,
            color_profile=self._config.sensor.color_profile,
            depth_profile=self._config.sensor.depth_profile,
            rgb_intrinsics=rgb_intrinsics,
            aligned_depth_intrinsics=depth_intrinsics,
            depth_scale_meters_per_unit=self._config.sensor.depth_scale_meters_per_unit,
            sha256=calibration_digest,
        )
        rgb_encoding = str(message.rgb.encoding).lower()
        depth_encoding = str(message.depth.encoding).lower()
        if depth_encoding not in {"16uc1", "mono16"}:
            raise FrameValidationError(f"unsupported aligned-depth encoding: {depth_encoding}")
        if (
            int(message.rgb.width) != rgb_intrinsics.width
            or int(message.rgb.height) != rgb_intrinsics.height
        ):
            raise FrameValidationError("RGB image dimensions do not match CameraInfo")
        if (
            int(message.depth.width) != rgb_intrinsics.width
            or int(message.depth.height) != rgb_intrinsics.height
        ):
            raise FrameValidationError("aligned depth image dimensions do not match RGB CameraInfo")
        return RawObservation(
            capture_time_ros=capture,
            received_monotonic_ns=time.monotonic_ns(),
            rgb_metadata=color_metadata,
            depth_metadata=depth_metadata,
            rgb=_image_bytes(message.rgb, bytes_per_pixel=3),
            rgb_encoding=rgb_encoding,
            aligned_depth=_image_bytes(message.depth, bytes_per_pixel=2),
            depth_big_endian=bool(message.depth.is_bigendian),
            intrinsics=rgb_intrinsics,
            calibration=calibration,
            t_map_camera=t_map_camera,
            t_base_camera=t_base_camera,
            localization=localization,
        )

    def _lookup_transform(self, target_frame: str, capture: TimePoint) -> RigidTransform:
        assert self._tf_buffer is not None
        assert self._ros_time_type is not None
        assert self._duration_type is not None
        ros_time = self._ros_time_type(nanoseconds=capture.unix_ns)
        timeout = self._duration_type(
            nanoseconds=self._config.stream.tf_lookup_timeout_ms * 1_000_000
        )
        try:
            message = self._tf_buffer.lookup_transform(
                target_frame,
                self._config.sensor.camera_frame,
                ros_time,
                timeout=timeout,
            )
        except Exception as exc:
            raise _DropFrame(
                DropReason.TF_MISSING,
                f"capture-time TF {target_frame}->{self._config.sensor.camera_frame} unavailable",
            ) from exc
        return _transform_from_ros(
            message,
            parent=target_frame,
            child=self._config.sensor.camera_frame,
        )

    def _publish_health(self) -> None:
        try:
            health = self._health_provider()
            message = self._health_message_type()
            message.emitted_at = self._node.get_clock().now().to_msg()
            message.heartbeat_sequence = self._heartbeat_sequence
            self._heartbeat_sequence += 1
            message.robot_id = health.robot_id
            message.robot_boot_id = health.robot_boot_id
            message.gateway_instance_id = health.gateway_instance_id
            message.observation_session_id = health.observation_session_id
            message.sensor_id = health.sensor_id
            message.scene_version = health.scene_version
            message.localization_epoch_id = health.localization_epoch_id
            message.status = int(health.status)
            message.reason = int(health.reason)
            message.has_last_accepted_sequence = health.last_accepted_sequence is not None
            message.last_accepted_sequence = health.last_accepted_sequence or 0
            message.last_accepted_monotonic_ns = health.last_accepted_monotonic_ns
            message.capture_age_ms = float(health.capture_age_ms)
            message.ack_age_ms = float(health.ack_age_ms)
            message.drop_count = health.drop_count
            self._health_publisher.publish(message)
        except Exception as exc:
            if self._node is not None:
                self._node.get_logger().error(f"failed to publish observation health: {exc}")
