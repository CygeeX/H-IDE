"""ROS 2/Nav2 bridge for the isolated simulation domain.

The bridge subscribes to Nav2's velocity output and forwards only bounded JSON
commands to the SONIC driver.  Ground-truth pose and camera publication are
kept in the plant/observation adapter; this node never invents localization.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Iterable

from .config import SimConfig
from .safety import (
    BodyStabilityGuard,
    GoalProgressGuard,
    MotionDirectionGuard,
    RotationDirectionGuard,
    TranslationHeadingGuard,
)
from .transport import send_json, set_plant_support


class RosBridgeUnavailable(RuntimeError):
    pass


def run_ros_bridge(config: SimConfig) -> None:
    os.environ["ROS_DOMAIN_ID"] = str(config.ros_domain_id)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    if config.interface == "lo":
        # The host's default CycloneDDS profile advertises the Wi-Fi interface;
        # that makes an otherwise healthy isolated domain invisible to local
        # ROS CLI/tools.  Keep simulation discovery loopback-only by default.
        os.environ["CYCLONEDDS_URI"] = (
            "<CycloneDDS><Domain><General>"
            "<Interfaces><NetworkInterface name=\"lo\"/></Interfaces>"
            "<AllowMulticast>true</AllowMulticast>"
            "</General></Domain></CycloneDDS>"
        )
    try:
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
        from nav_msgs.msg import Path
    except ImportError as exc:  # pragma: no cover - depends on ROS installation
        raise RosBridgeUnavailable("rclpy and geometry_msgs are required for ros-bridge") from exc

    rclpy.init()
    node = rclpy.create_node("h_ide_mujoco_sonic_bridge")
    last_message = {"at": time.monotonic()}
    latest_heading = {"yaw": None}
    direction_guard = MotionDirectionGuard()
    goal_guard = GoalProgressGuard()
    heading_guard = TranslationHeadingGuard(
        min_linear_mps=config.min_walk_command_mps,
        max_heading_error_rad=config.max_translation_heading_error_rad,
        max_expected_delta_rad=config.max_translation_facing_offset_rad,
        violation_s=config.translation_heading_violation_s,
    )
    rotation_guard = RotationDirectionGuard(
        max_reverse_rad=config.max_spin_reverse_rad,
        max_translation_m=config.max_spin_translation_m,
        translation_violation_s=config.spin_translation_violation_s,
    )
    stability_guard = BodyStabilityGuard()

    def interlock_reason() -> str | None:
        reasons = [
            guard.reason
            for guard in (
                direction_guard,
                goal_guard,
                heading_guard,
                rotation_guard,
                stability_guard,
            )
            if guard.latched and guard.reason
        ]
        return "; ".join(reasons) if reasons else None

    def send_stop(*, source: str, reason: str | None = None):
        payload = {"type": "stop", "source": source}
        if reason:
            payload["reason"] = reason
        send_json(config.command_host, config.command_port, payload)
        if reason:
            set_plant_support(
                "127.0.0.1",
                config.plant_port,
                "safety_stop",
                timeout=0.3,
            )

    def on_pose(message: PoseWithCovarianceStamped):
        pose = message.pose.pose
        orientation = pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        roll = math.atan2(
            2.0 * (orientation.w * orientation.x + orientation.y * orientation.z),
            1.0 - 2.0 * (orientation.x * orientation.x + orientation.y * orientation.y),
        )
        pitch_term = 2.0 * (
            orientation.w * orientation.y - orientation.z * orientation.x
        )
        pitch = math.asin(max(-1.0, min(1.0, pitch_term)))
        latest_heading["yaw"] = yaw
        try:
            direction_tripped = direction_guard.observe_pose(float(pose.position.x), float(pose.position.y), yaw)
            goal_tripped = goal_guard.observe_pose(float(pose.position.x), float(pose.position.y))
            heading_tripped = heading_guard.observe_pose(yaw)
            rotation_tripped = rotation_guard.observe_pose(float(pose.position.x), float(pose.position.y), yaw)
            stability_tripped = stability_guard.observe(roll, pitch)
        except ValueError as exc:
            node.get_logger().warning(f"discarding invalid localization pose: {exc}")
            return
        if (
            direction_tripped
            or goal_tripped
            or heading_tripped
            or rotation_tripped
            or stability_tripped
        ):
            reason = interlock_reason() or "measured navigation motion mismatch"
            node.get_logger().error(f"SONIC navigation interlock latched: {reason}")
            try:
                send_stop(source="navigation_guard", reason=reason)
            except Exception as exc:
                node.get_logger().error(f"failed to deliver navigation interlock stop: {exc}")

    def on_path(message: Path):
        if not message.poses:
            return
        endpoint = message.poses[-1].pose.position
        try:
            changed = goal_guard.observe_goal(float(endpoint.x), float(endpoint.y))
        except ValueError as exc:
            node.get_logger().warning(f"discarding invalid Nav2 global path: {exc}")
            return
        if changed:
            node.get_logger().info(
                f"tracking Nav2 path endpoint ({float(endpoint.x):.3f}, {float(endpoint.y):.3f})"
            )

    def on_cmd(message: Twist):
        vx = float(message.linear.x)
        vy = float(message.linear.y)
        wz = float(message.angular.z)
        if not all(math.isfinite(value) for value in (vx, vy, wz)):
            node.get_logger().warning("discarding non-finite /cmd_vel")
            return
        direction_admitted = direction_guard.accept_command(vx, vy, wz)
        goal_admitted = goal_guard.accept_command(vx, vy, wz)
        heading_admitted = heading_guard.accept_command(vx, vy, wz)
        rotation_admitted = rotation_guard.accept_command(vx, vy, wz)
        stability_admitted = stability_guard.admit_command(vx, vy, wz)
        if (
            not direction_admitted
            or not goal_admitted
            or not heading_admitted
            or not rotation_admitted
            or not stability_admitted
        ):
            try:
                send_stop(
                    source="navigation_guard",
                    reason=interlock_reason() or "navigation interlock awaiting sustained Nav2 idle",
                )
            except Exception as exc:
                node.get_logger().error(f"failed to maintain navigation interlock stop: {exc}")
            last_message["at"] = time.monotonic()
            return
        payload = {"type": "cmd_vel", "vx": vx, "vy": vy, "wz": wz, "source": "nav2"}
        if latest_heading["yaw"] is not None:
            payload["heading_yaw"] = latest_heading["yaw"]
        send_json(config.command_host, config.command_port, payload)
        last_message["at"] = time.monotonic()

    node.create_subscription(Twist, config.cmd_vel_topic, on_cmd, 10)
    node.create_subscription(PoseWithCovarianceStamped, config.localization_topic, on_pose, 10)
    node.create_subscription(Path, config.nav_path_topic, on_path, 10)

    def watchdog():
        if time.monotonic() - last_message["at"] > config.command_timeout_s:
            # Absence of Nav2 commands while the downstream driver is already
            # held at zero is an idle observation.  Feed it to both latches so
            # they can clear only after their configured sustained-zero window
            # instead of permanently rejecting the next reviewed goal.
            direction_guard.accept_command(0.0, 0.0, 0.0)
            goal_guard.accept_command(0.0, 0.0, 0.0)
            heading_guard.accept_command(0.0, 0.0, 0.0)
            rotation_guard.accept_command(0.0, 0.0, 0.0)
            stability_guard.admit_command(0.0, 0.0, 0.0)
            try:
                send_stop(
                    source=(
                        "navigation_guard"
                        if direction_guard.latched
                        or goal_guard.latched
                        or heading_guard.latched
                        or rotation_guard.latched
                        or stability_guard.latched
                        else "nav2_watchdog"
                    ),
                    reason=interlock_reason(),
                )
            except Exception:
                pass

    node.create_timer(min(config.command_timeout_s / 2, 0.1), watchdog)
    node.get_logger().info(
        f"H-IDE MuJoCo SONIC bridge listening to {config.cmd_vel_topic} "
        f"and {config.nav_path_topic} on ROS domain {config.ros_domain_id}"
    )
    try:
        rclpy.spin(node)
    finally:
        try:
            send_json(config.command_host, config.command_port, {"type": "stop", "source": "bridge_shutdown"})
        except OSError:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    run_ros_bridge(SimConfig.from_file(args.config))
