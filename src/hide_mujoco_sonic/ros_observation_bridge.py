"""Republish loopback MuJoCo captures into the Robot Agent ROS RGB-D contract."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import time
from typing import Iterable, Mapping
from urllib.parse import urlencode
import uuid

from .config import SimConfig
from .frames import BodyTwist, PoseDifferenceDiagnostic, PoseDifferenceResult
from .render_workers import MAX_CAPTURE_AGE_NS


_CAPTURE_BOOT_HEADER = "X-Hide-Capture-Boot-Id"
_CAPTURE_SEQUENCE_HEADER = "X-Hide-Capture-Sequence"
_CAPTURE_TIME_HEADER = "X-Hide-Captured-At-Unix-Ns"


class CaptureContractError(ValueError):
    """A state or image response cannot belong to one atomic RGB-D capture."""


@dataclass(frozen=True, slots=True)
class CaptureIdentity:
    producer_boot_id: str
    sequence: int
    captured_at_unix_ns: int

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "CaptureIdentity":
        producer_boot_id = state.get("producerBootId")
        if not isinstance(producer_boot_id, str) or not producer_boot_id.strip():
            raise CaptureContractError("MuJoCo state has no producerBootId")
        try:
            uuid.UUID(producer_boot_id)
        except ValueError as exc:
            raise CaptureContractError("MuJoCo producerBootId is not a UUID") from exc
        try:
            raw_sequence = state["sequence"]
            raw_captured_at = state["capturedAtUnixNs"]
        except KeyError as exc:
            raise CaptureContractError(
                "MuJoCo state has an invalid capture sequence or timestamp"
            ) from exc
        if (
            isinstance(raw_sequence, bool)
            or not isinstance(raw_sequence, int)
            or isinstance(raw_captured_at, bool)
            or not isinstance(raw_captured_at, int)
        ):
            raise CaptureContractError(
                "MuJoCo state has an invalid capture sequence or timestamp"
            )
        sequence = raw_sequence
        captured_at_unix_ns = raw_captured_at
        if sequence <= 0:
            raise CaptureContractError("MuJoCo capture sequence is not positive")
        if captured_at_unix_ns <= 0:
            raise CaptureContractError("MuJoCo capture timestamp is invalid")
        return cls(producer_boot_id, sequence, captured_at_unix_ns)

    def exact_resource_url(self, base_url: str, resource: str) -> str:
        if not resource.startswith("/"):
            raise ValueError("capture resource must be an absolute path")
        query = urlencode(
            {"boot": self.producer_boot_id, "sequence": str(self.sequence)}
        )
        return f"{base_url}{resource}?{query}"

    def validate_age(self, now_unix_ns: int) -> int:
        age_ns = int(now_unix_ns) - self.captured_at_unix_ns
        if age_ns < 0:
            raise CaptureContractError("MuJoCo capture timestamp is in the future")
        if age_ns > MAX_CAPTURE_AGE_NS:
            raise CaptureContractError(
                f"MuJoCo capture is stale ({age_ns / 1_000_000.0:.1f} ms)"
            )
        return age_ns

    def validate_response_headers(
        self,
        headers: Mapping[str, object],
        resource: str,
    ) -> None:
        # HTTP header names are case-insensitive.  Normalize explicitly so the
        # contract remains testable without depending on requests' CaseInsensitiveDict.
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}

        def required(name: str) -> str:
            try:
                return normalized[name.lower()]
            except KeyError as exc:
                raise CaptureContractError(
                    f"{resource} response is missing {name}"
                ) from exc

        response_boot_id = required(_CAPTURE_BOOT_HEADER)
        try:
            response_sequence = int(required(_CAPTURE_SEQUENCE_HEADER))
            response_captured_at_ns = int(required(_CAPTURE_TIME_HEADER))
        except ValueError as exc:
            raise CaptureContractError(
                f"{resource} response has malformed capture headers"
            ) from exc
        if response_boot_id != self.producer_boot_id:
            raise CaptureContractError(
                f"{resource} response belongs to a different producer boot"
            )
        if response_sequence != self.sequence:
            raise CaptureContractError(
                f"{resource} response belongs to sequence {response_sequence}, "
                f"expected {self.sequence}"
            )
        if response_captured_at_ns != self.captured_at_unix_ns:
            raise CaptureContractError(
                f"{resource} response has a mismatched capture timestamp"
            )


def is_new_capture(
    candidate: CaptureIdentity,
    last_producer_boot_id: str | None,
    last_sequence: int,
) -> bool:
    """Apply a strict sequence order within a boot, with reset on a new boot."""

    return (
        last_producer_boot_id is None
        or candidate.producer_boot_id != last_producer_boot_id
        or candidate.sequence > last_sequence
    )


class MujocoObservationBridge:
    def __init__(self, config: SimConfig):
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from realsense2_camera_msgs.msg import RGBD, Metadata
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        from tf2_ros import TransformBroadcaster

        self.config = config
        self.rclpy = rclpy
        self.node = Node("h_ide_mujoco_observation_bridge")
        self.RGBD, self.Metadata = RGBD, Metadata
        self.CameraInfo, self.Image = CameraInfo, Image
        self.PoseWithCovarianceStamped, self.TransformStamped = PoseWithCovarianceStamped, TransformStamped
        self.Odometry = Odometry
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        epoch_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.rgbd_pub = self.node.create_publisher(RGBD, "/camera/camera/rgbd", qos)
        self.color_meta_pub = self.node.create_publisher(Metadata, "/camera/camera/color/metadata", qos)
        self.depth_meta_pub = self.node.create_publisher(Metadata, "/camera/camera/depth/metadata", qos)
        self.pose_pub = self.node.create_publisher(PoseWithCovarianceStamped, config.localization_topic, qos)
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", qos)
        self.epoch_pub = self.node.create_publisher(String, "/localization/epoch_id", epoch_qos)
        self.tf = TransformBroadcaster(self.node)
        self._last_producer_boot_id: str | None = None
        self._last_sequence = 0
        self._last_captured_at_ns = 0
        self._pose_difference_diagnostic = PoseDifferenceDiagnostic()
        self._last_pose_difference = PoseDifferenceResult("initialized")
        self._poll_count = 0
        self._base_url = f"http://{config.telemetry_host}:{config.plant_port}"
        epoch = String(); epoch.data = config.localization_epoch_id; self.epoch_pub.publish(epoch)
        self.timer = self.node.create_timer(1.0 / config.observation_hz, self.poll)

    @staticmethod
    def _quaternion_multiply(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )

    @classmethod
    def _base_to_camera(
        cls,
        base_pose: tuple[float, float, float, float, float, float, float],
        camera_pose: tuple[float, float, float, float, float, float, float],
    ) -> tuple[float, float, float, float, float, float, float]:
        import numpy as np

        x, y, z, qw, qx, qy, qz = base_pose
        camera_x, camera_y, camera_z, camera_qw, camera_qx, camera_qy, camera_qz = (
            camera_pose
        )
        base_rotation = np.asarray(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )
        relative_position = base_rotation.T @ np.asarray(
            [camera_x - x, camera_y - y, camera_z - z],
            dtype=float,
        )
        relative_quaternion = cls._quaternion_multiply(
            (qw, -qx, -qy, -qz),
            (camera_qw, camera_qx, camera_qy, camera_qz),
        )
        norm = math.sqrt(sum(value * value for value in relative_quaternion))
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError("MuJoCo camera relative quaternion is invalid")
        relative_quaternion = tuple(value / norm for value in relative_quaternion)
        if relative_quaternion[0] < 0:
            relative_quaternion = tuple(-value for value in relative_quaternion)
        return (
            float(relative_position[0]),
            float(relative_position[1]),
            float(relative_position[2]),
            *relative_quaternion,
        )

    @staticmethod
    def _stamp(unix_ns: int):
        from builtin_interfaces.msg import Time
        value = Time()
        value.sec = int(unix_ns // 1_000_000_000)
        value.nanosec = int(unix_ns % 1_000_000_000)
        return value

    @staticmethod
    def _assign_twist(message, twist: BodyTwist) -> None:
        """Populate all six ROS Twist axes from one base_link contract."""

        message.linear.x = twist.vx
        message.linear.y = twist.vy
        message.linear.z = twist.vz
        message.angular.x = twist.wx
        message.angular.y = twist.wy
        message.angular.z = twist.wz

    def _observe_pose_difference(
        self,
        capture: CaptureIdentity,
        position: tuple[float, float, float],
        quaternion: tuple[float, float, float, float],
    ) -> None:
        """Keep diagnostic finite differences inside one producer boot."""

        if (
            self._last_producer_boot_id is not None
            and capture.producer_boot_id != self._last_producer_boot_id
        ):
            self._pose_difference_diagnostic = PoseDifferenceDiagnostic()
            self._last_pose_difference = PoseDifferenceResult("initialized")
        self._last_pose_difference = self._pose_difference_diagnostic.observe(
            capture.captured_at_unix_ns,
            position,
            quaternion,
        )

    def poll(self):
        import requests

        poll_started = time.perf_counter()
        try:
            state_response = requests.get(f"{self._base_url}/state", timeout=0.15)
            state_response.raise_for_status()
            state = state_response.json()
            if state.get("status") != "running":
                raise RuntimeError(
                    "MuJoCo capture is not healthy: "
                    f"{state.get('support', {}).get('fault') or state.get('observationStatus') or state.get('status')}"
                )
            capture = CaptureIdentity.from_state(state)
            capture.validate_response_headers(state_response.headers, "/state")
            if not is_new_capture(
                capture,
                self._last_producer_boot_id,
                self._last_sequence,
            ):
                return
            # Check the source timestamp before spending either network timeout.
            capture.validate_age(time.time_ns())
            rgb_response = requests.get(
                capture.exact_resource_url(self._base_url, "/rgb.jpg"),
                timeout=0.2,
            )
            rgb_response.raise_for_status()
            capture.validate_response_headers(rgb_response.headers, "/rgb.jpg")
            depth_response = requests.get(
                capture.exact_resource_url(self._base_url, "/depth.raw"),
                timeout=0.2,
            )
            depth_response.raise_for_status()
            capture.validate_response_headers(depth_response.headers, "/depth.raw")
            # Network latency is part of true capture age.  Recheck only after
            # both exact resources have arrived; a 409 or mismatch never reaches
            # any ROS publisher.
            capture.validate_age(time.time_ns())

            import cv2
            import numpy as np

            rgb_bgr = cv2.imdecode(np.frombuffer(rgb_response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
            if rgb_bgr is None or rgb_bgr.shape[:2] != (480, 640):
                raise ValueError("MuJoCo RGB dimensions are invalid")
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            depth = np.frombuffer(depth_response.content, dtype="<u2")
            if depth.size != 640 * 480:
                raise ValueError("MuJoCo depth dimensions are invalid")
            depth = depth.reshape((480, 640))
        except Exception as exc:
            self.node.get_logger().warning(f"MuJoCo capture unavailable: {exc}")
            return

        sequence = capture.sequence
        captured_at_ns = capture.captured_at_unix_ns

        stamp = self._stamp(captured_at_ns)
        raw_pose = state["pose"]
        values = [float(raw_pose[key]) for key in ("x", "y", "z", "qw", "qx", "qy", "qz")]
        if not all(math.isfinite(value) for value in values):
            self.node.get_logger().error("MuJoCo pose contains non-finite values")
            return
        x, y, z, qw, qx, qy, qz = values
        raw_root_velocity = state.get("rootVelocity")
        if not isinstance(raw_root_velocity, dict):
            self.node.get_logger().error(
                "MuJoCo state does not contain rootVelocity"
            )
            return
        try:
            root_velocity = BodyTwist.from_capture_state(raw_root_velocity)
        except ValueError as exc:
            self.node.get_logger().error(str(exc))
            return
        raw_camera_pose = state.get("cameraPose")
        if not isinstance(raw_camera_pose, dict):
            self.node.get_logger().error("MuJoCo state does not contain cameraPose")
            return
        camera_values = [
            float(raw_camera_pose[key])
            for key in ("x", "y", "z", "qw", "qx", "qy", "qz")
        ]
        if not all(math.isfinite(value) for value in camera_values):
            self.node.get_logger().error("MuJoCo camera pose contains non-finite values")
            return
        base_to_camera = self._base_to_camera(
            (x, y, z, qw, qx, qy, qz),
            tuple(camera_values),
        )
        raw_intrinsics = state.get("intrinsics")
        if not isinstance(raw_intrinsics, dict):
            self.node.get_logger().error("MuJoCo state does not contain intrinsics")
            return
        fx, fy, cx, cy = [
            float(raw_intrinsics[key])
            for key in ("fx", "fy", "cx", "cy")
        ]
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            self.node.get_logger().error("MuJoCo camera intrinsics are invalid")
            return
        try:
            # Decode and state validation also consume the 500 ms budget.  This
            # is the final admission gate immediately before the first publish.
            capture.validate_age(time.time_ns())
        except CaptureContractError as exc:
            self.node.get_logger().warning(f"MuJoCo capture unavailable: {exc}")
            return
        # Pose differencing is diagnostic-only and starts a new anchor when
        # the capture producer restarts.  Run it after all admission checks so
        # a rejected capture cannot contaminate a later accepted generation.
        self._observe_pose_difference(
            capture,
            (x, y, z),
            (qw, qx, qy, qz),
        )
        pose = self.PoseWithCovarianceStamped()
        pose.header.stamp, pose.header.frame_id = stamp, "map"
        pose.pose.pose.position.x, pose.pose.pose.position.y, pose.pose.pose.position.z = x, y, z
        pose.pose.pose.orientation.w, pose.pose.pose.orientation.x = qw, qx
        pose.pose.pose.orientation.y, pose.pose.pose.orientation.z = qy, qz
        pose.pose.covariance = [0.0] * 36
        pose.pose.covariance[0] = pose.pose.covariance[7] = pose.pose.covariance[14] = 1e-5
        pose.pose.covariance[35] = 1e-5
        self.pose_pub.publish(pose)
        # Nav2 consumes the conventional map -> odom -> base_link chain.  The
        # simulator gives us a perfect map pose, so odom is an identity frame;
        # publishing the split chain avoids competing TF authorities while still
        # satisfying the observation gateway's map-to-camera lookup.
        map_to_odom = self.TransformStamped()
        map_to_odom.header.stamp, map_to_odom.header.frame_id, map_to_odom.child_frame_id = stamp, "map", "odom"
        map_to_odom.transform.rotation.w = 1.0
        odom_to_base = self.TransformStamped()
        odom_to_base.header.stamp, odom_to_base.header.frame_id, odom_to_base.child_frame_id = stamp, "odom", "base_link"
        odom_to_base.transform.translation.x, odom_to_base.transform.translation.y, odom_to_base.transform.translation.z = x, y, z
        odom_to_base.transform.rotation.w, odom_to_base.transform.rotation.x = qw, qx
        odom_to_base.transform.rotation.y, odom_to_base.transform.rotation.z = qy, qz
        camera_transforms = []
        for child in ("camera_color_optical_frame", "camera_rgbd_optical_frame"):
            camera_transform = self.TransformStamped()
            camera_transform.header.stamp = stamp
            camera_transform.header.frame_id = "base_link"
            camera_transform.child_frame_id = child
            (
                camera_transform.transform.translation.x,
                camera_transform.transform.translation.y,
                camera_transform.transform.translation.z,
                camera_transform.transform.rotation.w,
                camera_transform.transform.rotation.x,
                camera_transform.transform.rotation.y,
                camera_transform.transform.rotation.z,
            ) = base_to_camera
            camera_transforms.append(camera_transform)
        self.tf.sendTransform([map_to_odom, odom_to_base, *camera_transforms])

        odom = self.Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = x, y, z
        odom.pose.pose.orientation.w, odom.pose.pose.orientation.x = qw, qx
        odom.pose.pose.orientation.y, odom.pose.pose.orientation.z = qy, qz
        odom.pose.covariance = [0.0] * 36
        odom.pose.covariance[0] = odom.pose.covariance[7] = odom.pose.covariance[14] = 1e-5
        odom.pose.covariance[35] = 1e-5
        self._assign_twist(odom.twist.twist, root_velocity)
        self.odom_pub.publish(odom)

        def camera_info():
            info = self.CameraInfo()
            info.header.stamp, info.header.frame_id = stamp, "camera_color_optical_frame"
            info.width, info.height, info.distortion_model = 640, 480, "plumb_bob"
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.d = [0.0] * 5
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            return info

        rgb_message = self.Image()
        rgb_message.header.stamp, rgb_message.header.frame_id = stamp, "camera_color_optical_frame"
        rgb_message.height, rgb_message.width, rgb_message.encoding = 480, 640, "rgb8"
        rgb_message.step, rgb_message.data = 640 * 3, rgb.tobytes()
        depth_message = self.Image()
        depth_message.header.stamp, depth_message.header.frame_id = stamp, "camera_color_optical_frame"
        depth_message.height, depth_message.width, depth_message.encoding = 480, 640, "16UC1"
        depth_message.step, depth_message.data = 640 * 2, depth.tobytes()
        bundle = self.RGBD()
        bundle.header.stamp, bundle.header.frame_id = stamp, "camera_rgbd_optical_frame"
        bundle.rgb_camera_info, bundle.depth_camera_info = camera_info(), camera_info()
        bundle.rgb, bundle.depth = rgb_message, depth_message
        # The capture timestamp is sourced from Unix wall time, not MuJoCo's
        # relative simulation clock.  Declare the protocol's matching domain
        # so the observation gateway can validate capture age end to end.
        metadata_json = json.dumps({"frame_number": sequence, "frame_timestamp": int(state["capturedAtUnixNs"]) / 1_000_000.0, "clock_domain": "global_time"})
        color_metadata, depth_metadata = self.Metadata(), self.Metadata()
        color_metadata.header.stamp = stamp; depth_metadata.header.stamp = stamp
        color_metadata.json_data = metadata_json; depth_metadata.json_data = metadata_json
        # Metadata and localization must be present before RGBD reaches the
        # strict ingress cache.
        self.color_meta_pub.publish(color_metadata)
        self.depth_meta_pub.publish(depth_metadata)
        self.rgbd_pub.publish(bundle)
        self._last_producer_boot_id = capture.producer_boot_id
        self._last_sequence = sequence
        self._last_captured_at_ns = captured_at_ns
        self._poll_count += 1
        if self._poll_count % 10 == 0:
            velocity_diagnostic = "fd_status=" + self._last_pose_difference.status
            if self._last_pose_difference.body_linear_velocity is not None:
                fd_vx, fd_vy, fd_vz = (
                    self._last_pose_difference.body_linear_velocity
                )
                velocity_error = math.sqrt(
                    (fd_vx - root_velocity.vx) ** 2
                    + (fd_vy - root_velocity.vy) ** 2
                    + (fd_vz - root_velocity.vz) ** 2
                )
                velocity_diagnostic += f" fd_error_mps={velocity_error:.3f}"
            self.node.get_logger().info(
                f"published synthetic RGB-D sequence={sequence} "
                f"poll_ms={(time.perf_counter() - poll_started) * 1000.0:.1f} "
                f"capture_age_ms={(time.time_ns() - captured_at_ns) / 1_000_000.0:.1f} "
                f"{velocity_diagnostic}"
            )


def run(config: SimConfig) -> None:
    os.environ["ROS_DOMAIN_ID"] = str(config.ros_domain_id)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    if config.interface == "lo":
        os.environ["CYCLONEDDS_URI"] = (
            "<CycloneDDS><Domain><General>"
            "<Interfaces><NetworkInterface name=\"lo\"/></Interfaces>"
            "<AllowMulticast>true</AllowMulticast>"
            "</General></Domain></CycloneDDS>"
        )
    import rclpy
    rclpy.init(args=None)
    bridge = MujocoObservationBridge(config)
    bridge.node.get_logger().info(
        f"MuJoCo synthetic RGB-D bridge started for {config.robot_id} on domain {config.ros_domain_id}"
    )
    try:
        rclpy.spin(bridge.node)
    finally:
        bridge.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    run(SimConfig.from_file(args.config))


if __name__ == "__main__":
    main()
