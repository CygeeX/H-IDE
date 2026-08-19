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
from .gait_adapter import (
    BaseObservation,
    GaitAdapter,
    GaitAdapterConfig,
    GaitState,
    Nav2Twist,
    SonicIntent,
)
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


def _new_bridge_generation() -> int:
    """Return a process-start generation without claiming plant continuity.

    The capture protocol currently has no explicit plant-generation field.
    Linux's monotonic clock is shared by processes and advances for the whole
    boot, so a restarted bridge gets a strictly newer value while the existing
    SONIC driver is alive.  This generation is *only* a bridge lifecycle: a
    MuJoCo restart is independently detected from ``LowState.tick`` by the
    driver and is a terminal fault.  It is therefore never used to pretend that
    SONIC state survived a plant restart.
    """

    generation = int(time.monotonic_ns())
    if generation <= 0:  # pragma: no cover - supported Linux clocks are positive
        raise RuntimeError("monotonic bridge generation is unavailable")
    return generation


def _intent_payload(
    intent: SonicIntent,
    *,
    measured_heading_yaw: float | None,
) -> dict[str, object]:
    """Build the only motion-bearing bridge-to-driver wire message."""

    payload: dict[str, object] = {"type": "sonic_intent", **intent.as_dict()}
    if measured_heading_yaw is not None:
        payload["measured_heading_yaw"] = float(measured_heading_yaw)
    return payload


def _ros_stamp_in_monotonic_domain(
    *,
    seconds: int,
    nanoseconds: int,
    wall_now_s: float,
    monotonic_now_s: float,
) -> float:
    """Preserve capture age while crossing ROS wall time into monotonic time.

    Treating callback arrival as the observation timestamp makes a delayed
    odometry sample appear fresh.  The adapter uses monotonic time for timeout
    safety, so carry over the ROS header's wall-clock age without ever using
    wall time for elapsed-time comparisons.
    """

    if (
        isinstance(seconds, bool)
        or isinstance(nanoseconds, bool)
        or not isinstance(seconds, int)
        or not isinstance(nanoseconds, int)
        or seconds <= 0
        or not 0 <= nanoseconds < 1_000_000_000
    ):
        raise ValueError("odometry header timestamp is invalid")
    values = (float(wall_now_s), float(monotonic_now_s))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bridge clocks are non-finite")
    captured_wall_s = float(seconds) + float(nanoseconds) / 1_000_000_000.0
    age_s = float(wall_now_s) - captured_wall_s
    if age_s < -0.05:
        raise ValueError("odometry timestamp is in the future")
    return float(monotonic_now_s) - max(0.0, age_s)


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
        from nav_msgs.msg import Odometry, Path
    except ImportError as exc:  # pragma: no cover - depends on ROS installation
        raise RosBridgeUnavailable("rclpy and geometry_msgs are required for ros-bridge") from exc

    rclpy.init()
    node = rclpy.create_node("h_ide_mujoco_sonic_bridge")
    generation = _new_bridge_generation()
    gait = GaitAdapter(
        GaitAdapterConfig(
            command_timeout_s=config.command_timeout_s,
            observation_timeout_s=max(0.30, 3.0 / config.observation_hz),
            max_linear_mps=config.max_linear_mps,
            max_angular_rps=config.max_angular_rps,
            linear_deadband_mps=config.min_walk_command_mps,
            angular_deadband_rps=0.05,
            start_timeout_s=max(1.5, config.motion_handoff_warmup_s + config.motion_handoff_blend_s + 0.75),
            stop_timeout_s=2.5,
            max_facing_delta_rad=config.max_facing_lead_rad,
        )
    )
    # ``arm`` is attempted only from a fresh measured odometry callback.  It
    # cannot arm while the base is moving, and the adapter refuses same-
    # generation recovery after stale/cancel/fault transitions.
    gait.advance_generation(generation)
    latest_heading = {"yaw": None}
    bridge_fault = {"reason": None}
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

    def send_intent(intent: SonicIntent) -> None:
        send_json(
            config.command_host,
            config.command_port,
            _intent_payload(
                intent,
                measured_heading_yaw=latest_heading["yaw"],
            ),
        )

    def latch_stop(*, source: str, reason: str) -> None:
        """Latch a bridge safety event and revoke this lifecycle's motion."""

        if bridge_fault["reason"] is None:
            bridge_fault["reason"] = f"{source}: {reason}"
        now = time.monotonic()
        intent = gait.cancel(generation, now, now=now)
        send_intent(intent)
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
                latch_stop(source="navigation_guard", reason=reason)
            except Exception as exc:
                node.get_logger().error(f"failed to deliver navigation interlock stop: {exc}")

    def on_odom(message: Odometry):
        """Feed authoritative base_link velocity/pose into the gait adapter."""

        now = time.monotonic()
        if message.header.frame_id != "odom" or message.child_frame_id != "base_link":
            reason = (
                "odometry frame contract mismatch "
                f"({message.header.frame_id!r}->{message.child_frame_id!r})"
            )
            node.get_logger().error(reason)
            try:
                latch_stop(source="odometry_contract", reason=reason)
            except Exception as exc:
                node.get_logger().error(f"failed to deliver odometry-frame stop: {exc}")
            return
        try:
            observation_timestamp = _ros_stamp_in_monotonic_domain(
                seconds=int(message.header.stamp.sec),
                nanoseconds=int(message.header.stamp.nanosec),
                wall_now_s=time.time(),
                monotonic_now_s=now,
            )
        except (TypeError, ValueError) as exc:
            reason = f"odometry time contract mismatch: {exc}"
            node.get_logger().error(reason)
            try:
                latch_stop(source="odometry_contract", reason=reason)
            except Exception as delivery_exc:
                node.get_logger().error(
                    f"failed to deliver odometry-time stop: {delivery_exc}"
                )
            return
        pose = message.pose.pose
        orientation = pose.orientation
        quaternion = tuple(
            float(value)
            for value in (
                orientation.w,
                orientation.x,
                orientation.y,
                orientation.z,
            )
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        if (
            not all(math.isfinite(value) for value in quaternion)
            or not 0.999 <= quaternion_norm <= 1.001
        ):
            reason = "base_link odometry quaternion is non-finite or not normalized"
            node.get_logger().error(reason)
            try:
                latch_stop(source="odometry_contract", reason=reason)
            except Exception as exc:
                node.get_logger().error(f"failed to deliver invalid-odometry stop: {exc}")
            return
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
        twist = message.twist.twist
        observation = BaseObservation(
            generation=generation,
            timestamp_s=observation_timestamp,
            x=float(pose.position.x),
            y=float(pose.position.y),
            yaw=yaw,
            roll=roll,
            pitch=pitch,
            vx_body=float(twist.linear.x),
            vy_body=float(twist.linear.y),
            wz=float(twist.angular.z),
        )
        intent = gait.ingest_observation(observation, now=now)
        if bridge_fault["reason"] is None and gait.state is GaitState.DISARMED:
            # A successful result proves this exact observation was fresh and
            # stationary; rejected attempts remain fail-closed.
            armed = gait.arm(generation, now=now)
            if armed.state == GaitState.IDLE.value:
                intent = armed
        try:
            send_intent(intent)
        except Exception as exc:
            node.get_logger().error(f"failed to deliver measured gait state: {exc}")

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
        now = time.monotonic()
        vx = float(message.linear.x)
        vy = float(message.linear.y)
        wz = float(message.angular.z)
        if not all(math.isfinite(value) for value in (vx, vy, wz)):
            node.get_logger().error("non-finite /cmd_vel latched as a navigation fault")
            try:
                latch_stop(
                    source="nav2_cmd_vel",
                    reason="Nav2 emitted a non-finite velocity command",
                )
            except Exception as exc:
                node.get_logger().error(f"failed to deliver invalid-command stop: {exc}")
            return
        if bridge_fault["reason"] is not None:
            try:
                send_intent(gait.tick(now=now))
            except Exception as exc:
                node.get_logger().error(f"failed to maintain bridge fault stop: {exc}")
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
                latch_stop(
                    source="navigation_guard",
                    reason=interlock_reason() or "navigation interlock awaiting sustained Nav2 idle",
                )
            except Exception as exc:
                node.get_logger().error(f"failed to maintain navigation interlock stop: {exc}")
            return
        intent = gait.ingest_twist(
            Nav2Twist(
                generation=generation,
                timestamp_s=now,
                vx=vx,
                vy=vy,
                wz=wz,
                source="nav2",
            ),
            now=now,
        )
        try:
            send_intent(intent)
        except Exception as exc:
            node.get_logger().error(f"failed to deliver Nav2 gait intent: {exc}")

    node.create_subscription(Twist, config.cmd_vel_topic, on_cmd, 10)
    node.create_subscription(Odometry, config.odom_topic, on_odom, 10)
    node.create_subscription(PoseWithCovarianceStamped, config.localization_topic, on_pose, 10)
    node.create_subscription(Path, config.nav_path_topic, on_path, 10)

    def watchdog():
        now = time.monotonic()
        # The adapter owns command/observation expiry.  Its first timeout tick
        # emits ``stop_requested`` before the driver performs any model mapping.
        intent = gait.tick(now=now)
        try:
            send_intent(intent)
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
            send_json(
                config.command_host,
                config.command_port,
                {
                    "type": "stop",
                    "source": "bridge_shutdown",
                    "reason": "ROS bridge process is shutting down",
                },
            )
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
