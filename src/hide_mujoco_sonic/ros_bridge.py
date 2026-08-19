"""ROS 2/Nav2 bridge for the isolated simulation domain.

The bridge subscribes to Nav2's velocity output and forwards only bounded JSON
commands to the SONIC driver.  Ground-truth pose and camera publication are
kept in the plant/observation adapter; this node never invents localization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import os
import time
from typing import Iterable
from uuid import UUID, uuid4

from .config import SimConfig
from .gait_adapter import (
    BaseObservation,
    GaitAdapter,
    GaitAdapterConfig,
    GaitState,
    Nav2Twist,
    SonicIntent,
)
from .motion_fault import (
    MotionFaultLatch,
    MotionSafetyFault,
    SonicHealthFaultGate,
    SonicHealth,
    motion_fault_qos,
    require_fresh_motion_health,
    sonic_health_probe,
)
from .safety import (
    BodyStabilityGuard,
    GoalProgressGuard,
    MotionDirectionGuard,
    RotationDirectionGuard,
    TranslationHeadingGuard,
)
from .transport import (
    latch_plant_physics_fault,
    request_json,
    send_json,
)


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
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
    except ImportError as exc:  # pragma: no cover - depends on ROS installation
        raise RosBridgeUnavailable("rclpy and geometry_msgs are required for ros-bridge") from exc

    rclpy.init()
    node = rclpy.create_node("h_ide_mujoco_sonic_bridge")
    generation = {"value": _new_bridge_generation()}
    fault_latch = MotionFaultLatch(
        robot_id=config.robot_id,
        localization_epoch_id=config.localization_epoch_id,
    )
    fault_publisher = node.create_publisher(
        String,
        config.motion_fault_topic,
        motion_fault_qos(),
    )
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
    # GaitAdapter may recover a never-moved IDLE observation reconnect within
    # one generation, but that local synchronization never releases a Plant
    # safety_stop.  Only /robot/control_arm, reached from control.acquire,
    # advances generation and carries the explicit Plant rearm bit.
    gait.advance_generation(generation["value"])
    latest_heading = {"yaw": None}
    latest_observation: dict[str, BaseObservation | None] = {"value": None}
    bridge_fault = {"reason": None}
    health_fault_gate = SonicHealthFaultGate(
        plant_lowstate_stale_s=config.plant_lowstate_stale_s,
        sonic_control_step_stale_s=config.sonic_control_step_stale_s,
    )
    health_quarantine: dict[str, str | bool | None] = {
        "active": False,
        "code": None,
    }
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

    def request_sonic_health(request_id: UUID) -> SonicHealth:
        """Read one correlated, strict v2 health snapshot from SONIC."""

        response = request_json(
            config.command_host,
            config.command_port,
            sonic_health_probe(
                robot_id=config.robot_id,
                producer_boot_id=fault_latch.producer_boot_id,
                request_id=request_id,
            ),
            timeout=config.sonic_health_probe_timeout_s,
        )
        return SonicHealth.from_response(
            response,
            expected_request_id=request_id,
            expected_robot_id=config.robot_id,
        )

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

    def publish_fault(event: MotionSafetyFault) -> None:
        message = String()
        message.data = event.to_json()
        fault_publisher.publish(message)

    def latch_stop(
        *,
        source: str,
        code: str,
        reason: str,
        stop_intent: SonicIntent | None = None,
        downstream_boot_id: UUID | None = None,
        probe_request_id: UUID | None = None,
    ) -> None:
        """Publish one immutable fault, then enforce the stronger local stop."""

        bounded_reason = str(reason).strip()[:512] or "motion safety fault"
        if bridge_fault["reason"] is None:
            bridge_fault["reason"] = f"{source}/{code}: {bounded_reason}"

        def local_stop() -> None:
            errors: list[str] = []
            intent = stop_intent
            # A motion-safety event is terminal for this Plant generation.
            # Latch the physics fault before delivering any joint-level stop so
            # the simulator cannot integrate a dynamically unsupported frozen
            # pose.  Recovery is an explicit process restart; no reset or
            # teleport path exists here.
            try:
                latch_plant_physics_fault(
                    "127.0.0.1",
                    config.plant_port,
                    f"{source}/{code}: {bounded_reason}"[:512],
                    timeout=0.3,
                )
            except Exception as exc:
                errors.append(f"Plant terminal halt: {exc}")
            try:
                if intent is None:
                    now = time.monotonic()
                    intent = gait.cancel(generation["value"], now, now=now)
                send_intent(intent)
            except Exception as exc:
                errors.append(f"SONIC stop delivery: {exc}")
            if errors:
                raise RuntimeError("; ".join(errors))

        event = fault_latch.trip(
            control_generation=generation["value"],
            source=source,
            code=code,
            reason=bounded_reason,
            publish=publish_fault,
            local_stop=local_stop,
            downstream_boot_id=downstream_boot_id,
            probe_request_id=probe_request_id,
        )
        node.get_logger().error(
            "motion safety fault latched "
            f"event={event.event_id} source={event.source} code={event.code}: "
            f"{event.reason}"
        )

    def deliver_intent(intent: SonicIntent, *, context: str) -> bool:
        existing_fault = fault_latch.event
        if existing_fault is not None:
            if fault_latch.complete:
                return False
            try:
                latch_stop(
                    source=existing_fault.source,
                    code=existing_fault.code,
                    reason=existing_fault.reason,
                    stop_intent=(
                        intent
                        if intent.state == GaitState.FAULT.value
                        else None
                    ),
                    downstream_boot_id=existing_fault.downstream_boot_id,
                    probe_request_id=existing_fault.probe_request_id,
                )
            except Exception as exc:
                node.get_logger().error(
                    f"failed to retry incomplete motion-fault handling: {exc}"
                )
            return False
        if intent.state == GaitState.FAULT.value:
            try:
                latch_stop(
                    source="gait_adapter",
                    code="GAIT_ADAPTER_FAULT",
                    reason=intent.reason or f"GaitAdapter fault during {context}",
                    stop_intent=intent,
                )
            except Exception as exc:
                node.get_logger().error(
                    f"failed to enforce GaitAdapter fault stop: {exc}"
                )
            return False
        try:
            send_intent(intent)
        except Exception as exc:
            reason = f"SONIC command channel unavailable during {context}: {exc}"
            node.get_logger().error(reason)
            try:
                latch_stop(
                    source="sonic_driver",
                    code="SONIC_UNAVAILABLE",
                    reason=reason,
                )
            except Exception as delivery_exc:
                node.get_logger().error(
                    f"failed to enforce unavailable-SONIC stop: {delivery_exc}"
                )
            return False
        return True

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
                latch_stop(
                    source="navigation_guard",
                    code="NAVIGATION_GUARD_FAULT",
                    reason=reason,
                )
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
                latch_stop(
                    source="ros_bridge",
                    code="ODOMETRY_CONTRACT_FAULT",
                    reason=reason,
                )
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
                latch_stop(
                    source="ros_bridge",
                    code="ODOMETRY_CONTRACT_FAULT",
                    reason=reason,
                )
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
                latch_stop(
                    source="ros_bridge",
                    code="ODOMETRY_CONTRACT_FAULT",
                    reason=reason,
                )
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
            generation=generation["value"],
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
        latest_observation["value"] = observation
        intent = gait.ingest_observation(observation, now=now)
        if intent.state == GaitState.FAULT.value:
            deliver_intent(intent, context="odometry observation")
            return
        if bridge_fault["reason"] is None and gait.state is GaitState.DISARMED:
            # A successful result proves this exact observation was fresh and
            # stationary; rejected attempts remain fail-closed.
            armed = gait.arm(generation["value"], now=now)
            if armed.state == GaitState.IDLE.value:
                intent = armed
        deliver_intent(intent, context="measured gait state")

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
                    source="ros_bridge",
                    code="NAV2_COMMAND_FAULT",
                    reason="Nav2 emitted a non-finite velocity command",
                )
            except Exception as exc:
                node.get_logger().error(f"failed to deliver invalid-command stop: {exc}")
            return
        if bridge_fault["reason"] is not None:
            deliver_intent(gait.tick(now=now), context="latched bridge fault")
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
                    code="NAVIGATION_GUARD_FAULT",
                    reason=interlock_reason() or "navigation interlock awaiting sustained Nav2 idle",
                )
            except Exception as exc:
                node.get_logger().error(f"failed to maintain navigation interlock stop: {exc}")
            return
        intent = gait.ingest_twist(
            Nav2Twist(
                generation=generation["value"],
                timestamp_s=now,
                vx=vx,
                vy=vy,
                wz=wz,
                source="nav2",
            ),
            now=now,
        )
        deliver_intent(intent, context="Nav2 gait intent")

    def control_arm(_request, response):
        """Use one explicit Robot Agent control.acquire to rearm the plant."""

        if bridge_fault["reason"] is not None:
            response.success = False
            response.message = (
                "navigation safety interlock is latched; process restart is required"
            )
            return response
        if gait.state in {
            GaitState.STARTING,
            GaitState.WALKING,
            GaitState.TURNING,
            GaitState.STOPPING,
        }:
            response.success = False
            response.message = "control arm rejected while a gait is active or stopping"
            return response

        health_request_id = uuid4()
        try:
            health = request_sonic_health(health_request_id)
        except Exception as exc:
            reason = f"SONIC health probe failed during control arm: {exc}"
            try:
                latch_stop(
                    source="sonic_driver",
                    code="SONIC_UNAVAILABLE",
                    reason=reason,
                )
            except Exception as delivery_exc:
                node.get_logger().error(
                    f"failed to enforce control-arm health stop: {delivery_exc}"
                )
            response.success = False
            response.message = reason
            return response

        health_decision = health_fault_gate.observe(
            health,
            gait_state=gait.state,
        )
        if health_decision.confirmed_cause is not None:
            cause = health_decision.confirmed_cause
            try:
                latch_stop(
                    source=cause.source,
                    code=cause.code,
                    reason=cause.reason,
                    downstream_boot_id=health.driver_boot_id,
                    probe_request_id=health_request_id,
                )
            except Exception as delivery_exc:
                node.get_logger().error(
                    f"failed to enforce control-arm health fault: {delivery_exc}"
                )
            response.success = False
            response.message = f"control arm rejected: {cause.code}"
            return response
        try:
            require_fresh_motion_health(
                health,
                plant_lowstate_stale_s=config.plant_lowstate_stale_s,
                sonic_control_step_stale_s=config.sonic_control_step_stale_s,
            )
        except ValueError as exc:
            # Age-only degradation is recoverable while the exact safety hold
            # remains confirmed, but it can never authorize a new generation.
            response.success = False
            response.message = f"control arm requires fresh SONIC health: {exc}"
            return response

        now = time.monotonic()
        measured = latest_observation["value"]
        if measured is None:
            response.success = False
            response.message = "control arm requires a measured base observation"
            return response
        if now - measured.timestamp_s > gait.config.observation_timeout_s:
            response.success = False
            response.message = "control arm requires a fresh base observation"
            return response
        if not gait._is_stationary(measured):
            response.success = False
            response.message = "control arm rejected while the measured base is moving"
            return response

        next_generation = max(
            generation["value"] + 1,
            _new_bridge_generation(),
        )
        generation["value"] = next_generation
        gait.advance_generation(next_generation, now=now)
        current_observation = replace(measured, generation=next_generation)
        latest_observation["value"] = current_observation
        gait.ingest_observation(current_observation, now=now)
        armed = gait.arm(next_generation, now=now)
        if armed.state != GaitState.IDLE.value:
            response.success = False
            response.message = armed.reason or "fresh stationary gait arm was rejected"
            return response
        payload = _intent_payload(
            armed,
            measured_heading_yaw=latest_heading["yaw"],
        )
        payload["control_arm"] = True
        try:
            acknowledgement = request_json(
                config.command_host,
                config.command_port,
                payload,
                timeout=0.75,
            )
            if (
                acknowledgement.get("mode") != "idle"
                or acknowledgement.get("generation") != next_generation
            ):
                raise RuntimeError("SONIC did not confirm the armed IDLE generation")
        except Exception as exc:
            failed_at = time.monotonic()
            stopped = gait.cancel(
                next_generation,
                failed_at,
                now=failed_at,
            )
            try:
                latch_stop(
                    source="sonic_driver",
                    code="SONIC_UNAVAILABLE",
                    reason=f"control arm failed closed: {exc}",
                    stop_intent=stopped,
                )
            except Exception as delivery_exc:
                node.get_logger().error(
                    f"failed to enforce control-arm stop: {delivery_exc}"
                )
            response.success = False
            response.message = f"control arm failed closed: {exc}"
            return response

        response.success = True
        response.message = (
            f"control generation {next_generation} armed at stationary IDLE"
        )
        return response

    node.create_subscription(Twist, config.cmd_vel_topic, on_cmd, 10)
    node.create_subscription(Odometry, config.odom_topic, on_odom, 10)
    node.create_subscription(PoseWithCovarianceStamped, config.localization_topic, on_pose, 10)
    node.create_subscription(Path, config.nav_path_topic, on_path, 10)
    node.create_service(Trigger, "/robot/control_arm", control_arm)

    def watchdog():
        now = time.monotonic()
        # The adapter owns command/observation expiry.  Its first timeout tick
        # emits ``stop_requested`` before the driver performs any model mapping.
        intent = gait.tick(now=now)
        deliver_intent(intent, context="gait watchdog")

    def health_probe():
        if bridge_fault["reason"] is not None:
            return
        request_id = uuid4()
        try:
            health = request_sonic_health(request_id)
        except Exception as exc:
            try:
                latch_stop(
                    source="sonic_driver",
                    code="SONIC_UNAVAILABLE",
                    reason=f"SONIC health probe failed: {exc}",
                    probe_request_id=request_id,
                )
            except Exception as delivery_exc:
                node.get_logger().error(
                    f"failed to enforce SONIC health-probe stop: {delivery_exc}"
                )
            return

        previous_pending = health_fault_gate.pending
        decision = health_fault_gate.observe(
            health,
            gait_state=gait.state,
        )
        pending = decision.pending
        quarantined = decision.quarantined_cause
        if quarantined is not None:
            if (
                health_quarantine["active"] is not True
                or health_quarantine["code"] != quarantined.code
            ):
                node.get_logger().warning(
                    "motion health age fault quarantined in confirmed safe hold "
                    f"driverBootId={decision.driver_boot_id} "
                    f"source={quarantined.source} code={quarantined.code} "
                    f"quarantinedCount={decision.quarantined_stale_count}; "
                    "new control remains blocked until health is fresh"
                )
            health_quarantine["active"] = True
            health_quarantine["code"] = quarantined.code
        elif health_quarantine["active"] is True:
            node.get_logger().info(
                "motion health safe-hold quarantine cleared "
                f"driverBootId={decision.driver_boot_id}"
            )
            health_quarantine["active"] = False
            health_quarantine["code"] = None
        if pending is not None and pending != previous_pending:
            node.get_logger().warning(
                "motion health stale confirmation pending "
                f"driverBootId={pending.driver_boot_id} "
                f"firstSource={pending.first_source} "
                f"firstCode={pending.first_code} "
                f"latestSource={pending.latest_source} "
                f"latestCode={pending.latest_code} "
                f"count={pending.count}/{decision.required_count} "
                f"firstAgeS={pending.first_age_s:.6f} "
                f"latestAgeS={pending.latest_age_s:.6f} "
                "filteredTransientCount="
                f"{decision.filtered_transient_count}"
            )
        elif (
            previous_pending is not None
            and pending is None
            and decision.confirmed_cause is None
        ):
            node.get_logger().info(
                "motion health stale confirmation cleared by healthy probe "
                f"driverBootId={previous_pending.driver_boot_id} "
                f"firstSource={previous_pending.first_source} "
                f"firstCode={previous_pending.first_code} "
                f"latestSource={previous_pending.latest_source} "
                f"latestCode={previous_pending.latest_code} "
                f"priorCount={previous_pending.count}/"
                f"{decision.required_count} "
                f"firstAgeS={previous_pending.first_age_s:.6f} "
                "filteredTransientCount="
                f"{decision.filtered_transient_count}"
            )

        cause = decision.confirmed_cause
        if cause is None:
            return
        try:
            latch_stop(
                source=cause.source,
                code=cause.code,
                reason=cause.reason,
                downstream_boot_id=health.driver_boot_id,
                probe_request_id=request_id,
            )
        except Exception as delivery_exc:
            node.get_logger().error(
                f"failed to enforce SONIC/Plant health stop: {delivery_exc}"
            )

    node.create_timer(min(config.command_timeout_s / 2, 0.1), watchdog)
    node.create_timer(config.sonic_health_probe_period_s, health_probe)
    node.get_logger().info(
        f"H-IDE MuJoCo SONIC bridge listening to {config.cmd_vel_topic} "
        f"and {config.nav_path_topic} on ROS domain {config.ros_domain_id}; "
        f"faults publish on {config.motion_fault_topic}"
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
