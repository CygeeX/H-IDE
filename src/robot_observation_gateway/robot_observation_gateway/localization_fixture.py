"""Local-only moving TF/localization fixture for strict real-camera bench verification."""

from __future__ import annotations

import math
from collections.abc import Sequence

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


class LocalizationFixture(Node):
    def __init__(self) -> None:
        super().__init__("observation_localization_fixture")
        self._pose = self.create_publisher(
            PoseWithCovarianceStamped,
            "/localization/pose",
            10,
        )
        self._epoch = self.create_publisher(
            String,
            "/localization/epoch_id",
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._started_ns = self.get_clock().now().nanoseconds
        self._publish_count = 0
        self._publish_base_camera_mount()
        epoch = String()
        epoch.data = "q40-map-epoch-dev-1"
        self._epoch.publish(epoch)
        self._timer = self.create_timer(0.01, self._publish_localization)
        self.get_logger().info("strict localization/TF fixture started at 100 Hz")

    def _publish_base_camera_mount(self) -> None:
        value = TransformStamped()
        value.header.stamp = self.get_clock().now().to_msg()
        value.header.frame_id = "base_link"
        value.child_frame_id = "camera_link"
        value.transform.translation.x = 0.20
        value.transform.translation.y = 0.0
        value.transform.translation.z = 1.0
        value.transform.rotation.w = 1.0
        self._static_tf.sendTransform(value)

    def _publish_localization(self) -> None:
        now = self.get_clock().now()
        elapsed = (now.nanoseconds - self._started_ns) / 1_000_000_000.0
        x = 0.05 * math.sin(elapsed * 0.2)
        yaw = 0.02 * math.sin(elapsed * 0.1)
        half_yaw = yaw / 2.0

        transform = TransformStamped()
        transform.header.stamp = now.to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = x
        transform.transform.rotation.z = math.sin(half_yaw)
        transform.transform.rotation.w = math.cos(half_yaw)
        self._tf.sendTransform(transform)

        pose = PoseWithCovarianceStamped()
        pose.header.stamp = now.to_msg()
        pose.header.frame_id = "map"
        pose.pose.pose.position.x = x
        pose.pose.pose.orientation.z = math.sin(half_yaw)
        pose.pose.pose.orientation.w = math.cos(half_yaw)
        pose.pose.covariance = [0.0] * 36
        pose.pose.covariance[0] = 0.01
        pose.pose.covariance[7] = 0.01
        pose.pose.covariance[14] = 0.02
        pose.pose.covariance[21] = 0.01
        pose.pose.covariance[28] = 0.01
        pose.pose.covariance[35] = 0.02
        self._pose.publish(pose)
        self._publish_count += 1
        if self._publish_count == 1:
            self.get_logger().info("published first localization sample")


def main(argv: Sequence[str] | None = None) -> int:
    rclpy.init(args=list(argv) if argv is not None else None)
    node = LocalizationFixture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  # type: ignore[no-untyped-call]
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
