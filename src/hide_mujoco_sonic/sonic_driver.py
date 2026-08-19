"""Unitree DDS SONIC driver.

This process is intentionally separate from the MuJoCo plant.  The plant and
driver communicate only through the official `rt/lowstate`/`rt/lowcmd` DDS
topics, which keeps the simulation boundary close to the real Unitree stack.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib import request

from .config import SimConfig
from .safety import (
    CmdVelGuard,
    NavigationArmGuard,
    PlantControlClock,
    PlantControlClockFault,
    PlantControlStep,
    PlantTickGuard,
    ReplanIntentGate,
    StaticJointHandoff,
)
from .telemetry import (
    JointFaultTraceBuffer,
    TelemetryState,
    TelemetryStore,
    serve_telemetry,
)
from .transport import parse_lines

logger = logging.getLogger(__name__)


class SonicDriver:
    def __init__(self, config: SimConfig, telemetry: TelemetryStore):
        self.config = config
        self.telemetry = telemetry
        self.guard = CmdVelGuard(
            max_linear_mps=config.max_linear_mps,
            min_walk_command_mps=config.min_walk_command_mps,
            max_angular_rps=config.max_angular_rps,
            max_facing_lead_rad=config.max_facing_lead_rad,
            max_translation_facing_offset_rad=(
                config.max_translation_facing_offset_rad
            ),
            timeout_s=config.command_timeout_s,
        )
        self._lowstate = None
        self._lowstate_lock = threading.RLock()
        self._control_condition = threading.Condition(self._lowstate_lock)
        self._control_clock = PlantControlClock(control_hz=config.control_hz)
        self._control_clock_armed = False
        self._pending_control_sample: tuple[Any, PlantControlStep] | None = None
        self._wall_clock = time.monotonic
        self._control_clock_armed_at_wall_s: float | None = None
        self._last_lowstate_wall_s: float | None = None
        self._lowstate_stall_timeout_s = max(1.0, config.command_timeout_s * 4.0)
        self._stop = threading.Event()
        self._fault_latched = threading.Event()
        self._terminal_fault_reason: str | None = None
        self._tick_guard = PlantTickGuard()
        self._motion_armed = True
        self._navigation_task_active = False
        self._input_buffer = b""
        self._last_intent_signature: tuple[bool, float, float, float] | None = None
        self._decoder_facing_rad: float | None = None
        self._pending_decoder_heading_delta_rad = 0.0
        self._intent_gate = ReplanIntentGate()
        self._joint_handoff = StaticJointHandoff(
            warmup_s=config.motion_handoff_warmup_s,
            blend_s=config.motion_handoff_blend_s,
        )
        self._static_hold_active = True
        self._plant_support_mode: str | None = None

        # These imports are intentionally lazy: `check` and safety tests must
        # work without ROS/Unitree SDK/ONNX installed.
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from lerobot.robots.unitree_g1.controllers.sonic_whole_body import SonicWholeBodyController

        self._sdk = (ChannelPublisher, ChannelSubscriber, LowCmd_, LowState_, CRC, unitree_hg_msg_dds__LowCmd_)
        ChannelFactoryInitialize(config.ros_domain_id, config.interface)
        self._publisher = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        self._subscriber.Init(self._on_lowstate, 10)
        self._crc = CRC()
        self._msg = unitree_hg_msg_dds__LowCmd_()
        self._controller_type = SonicWholeBodyController
        self._controller = self._controller_type(force_cpu=config.force_cpu)
        self._kp = self._controller.kp
        self._kd = self._controller.kd
        self._arm_guard = self._new_arm_guard()
        self._joint_names = tuple(self._joint_name(index) for index in range(29))
        self._fault_trace = JointFaultTraceBuffer(
            self._joint_names,
            max_samples=max(1, config.control_hz * 5),
        )
        self._fault_trace_path = self._default_fault_trace_path()

    @staticmethod
    def _default_fault_trace_path() -> Path:
        override = os.environ.get("HIDE_SONIC_FAULT_TRACE_PATH")
        if override:
            return Path(override)
        runtime_root = Path(f"/run/user/{os.getuid()}")
        if not runtime_root.is_dir():
            runtime_root = Path("/tmp")
        return (
            runtime_root
            / f"h-ide-mujoco-demo-{os.getuid()}"
            / "sonic-fault-trace.json"
        )

    def _new_arm_guard(self) -> NavigationArmGuard:
        from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

        # H-IDE currently exposes navigation only, but SONIC is a whole-body
        # locomotion policy and relies on shoulder/elbow counter-swing for
        # balance.  The earlier narrow envelope constrained most arm joints on
        # every tick and turned a safety guard into an untrained policy rewrite.
        # Keep a physical-range envelope and a high slew ceiling that catches
        # discontinuities while leaving nominal gait dynamics untouched.
        per_arm_deviation = (2.50, 1.40, 2.20, 1.45, 1.80, 1.50, 1.50)
        per_arm_velocity = (6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0)
        return NavigationArmGuard(
            DEFAULT_ANGLES[15:29],
            max_deviation_rad=per_arm_deviation * 2,
            max_velocity_rps=per_arm_velocity * 2,
        )

    def _on_lowstate(self, message):
        try:
            tick = int(message.tick)
            restarted = self._tick_guard.observe(tick)
        except (AttributeError, TypeError, ValueError) as exc:
            self._latch_terminal_fault(f"invalid lowstate tick: {exc}")
            return
        clock_fault: str | None = None
        received_at_wall_s = self._wall_clock()
        with self._control_condition:
            self._lowstate = message
            self._last_lowstate_wall_s = received_at_wall_s
            if restarted:
                clock_fault = (
                    "MuJoCo plant tick rolled back; explicit process restart required"
                )
            elif self._control_clock_armed:
                try:
                    control_step = self._control_clock.observe(tick)
                except PlantControlClockFault as exc:
                    clock_fault = str(exc)
                else:
                    if control_step is not None:
                        if self._pending_control_sample is not None:
                            pending_tick = self._pending_control_sample[1].tick_ms
                            clock_fault = (
                                "MuJoCo control sample overrun: "
                                f"{pending_tick}ms was not consumed before "
                                f"{control_step.tick_ms}ms; catch-up is forbidden"
                            )
                        else:
                            self._pending_control_sample = (message, control_step)
                            self._control_condition.notify_all()
        self.telemetry.update(plant_tick=tick)
        if clock_fault is not None:
            # Any clock discontinuity invalidates policy history and actuator
            # continuity.  Never rebuild history or replay missed policy ticks.
            self._latch_terminal_fault(clock_fault)

    def _arm_simulation_clock(self) -> None:
        """Start policy advancement from the freshest completed plant sample."""

        with self._control_condition:
            self._control_clock.reset()
            self._pending_control_sample = None
            self._control_clock_armed = True
            self._control_clock_armed_at_wall_s = self._wall_clock()
            lowstate = self._lowstate
            if lowstate is not None:
                step = self._control_clock.observe(int(lowstate.tick))
                assert step is not None
                self._pending_control_sample = (lowstate, step)
            self._control_condition.notify_all()
        self.telemetry.update(control_clock_status="simulation_time_50hz")

    def _next_control_sample(
        self,
        *,
        wait_timeout_s: float = 0.10,
    ) -> tuple[Any, PlantControlStep] | None:
        """Consume at most one admitted sample; wall time is only a wake timeout."""

        clock_fault: str | None = None
        with self._control_condition:
            if (
                self._pending_control_sample is None
                and not self._fault_latched.is_set()
                and not self._stop.is_set()
            ):
                self._control_condition.wait(timeout=wait_timeout_s)
            sample = self._pending_control_sample
            self._pending_control_sample = None
            if (
                sample is None
                and self._control_clock_armed
                and not self._fault_latched.is_set()
                and not self._stop.is_set()
            ):
                source_at = (
                    self._last_lowstate_wall_s
                    if self._last_lowstate_wall_s is not None
                    else self._control_clock_armed_at_wall_s
                )
                if (
                    source_at is not None
                    and self._wall_clock() - source_at
                    > self._lowstate_stall_timeout_s
                ):
                    clock_fault = (
                        "MuJoCo LowState stream stalled for more than "
                        f"{self._lowstate_stall_timeout_s:.3f}s wall time; "
                        "wall time is not used to catch up policy steps"
                    )
        if clock_fault is not None:
            # Wall time is used only as a liveness interlock.  It never emits a
            # control step or advances any controller/handoff state.
            self._latch_terminal_fault(clock_fault)
        return sample

    def _publish_measured_hold(self, lowstate) -> None:
        for index in range(29):
            motor = self._msg.motor_cmd[index]
            motor.mode = 1
            motor.kp = float(self._kp[index])
            motor.kd = float(self._kd[index])
            motor.q = float(lowstate.motor_state[index].q)
            motor.dq = 0.0
            motor.tau = 0.0
        self._msg.crc = self._crc.Crc(self._msg)
        self._publisher.Write(self._msg)

    def _latch_terminal_fault(self, reason: str) -> None:
        """Latch a non-recoverable process fault without rebuilding state."""

        if self._fault_latched.is_set():
            return
        self._terminal_fault_reason = str(reason)
        self.guard.stop(source="terminal_fault")
        self._navigation_task_active = False
        self._fault_latched.set()
        trace_path: str | None = None
        trace_samples = 0
        trace = getattr(self, "_fault_trace", None)
        if trace is not None:
            try:
                payload = trace.snapshot(
                    reason=self._terminal_fault_reason,
                    kp=tuple(float(value) for value in self._kp),
                    kd=tuple(float(value) for value in self._kd),
                )
                trace_samples = int(payload["sample_count"])
                path = trace.persist(self._fault_trace_path, payload)
                trace_path = str(path)
                logger.error(
                    "terminal SONIC fault; wrote %d pre-fault samples to %s: %s",
                    trace_samples,
                    trace_path,
                    self._terminal_fault_reason,
                )
            except Exception:
                logger.exception("failed to persist terminal SONIC fault trace")
        changes: dict[str, Any] = dict(
            status="failed",
            nav2_state="fault_latched",
            sonic_mode="IDLE",
            sonic_speed_mps=0.0,
            static_hold_active=True,
            motion_handoff_active=False,
            motion_handoff_ratio=0.0,
            navigation_task_active=False,
            control_clock_status="fault",
            last_error=self._terminal_fault_reason,
        )
        if trace is not None:
            changes.update(
                fault_trace_path=trace_path,
                fault_trace_sample_count=trace_samples,
                fault_joint_names=list(trace.joint_names),
            )
        self.telemetry.update(**changes)
        condition = getattr(self, "_control_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()

    def _set_plant_support_mode(self, mode: str) -> bool:
        """Switch only MuJoCo's fall-arrest profile over loopback.

        The endpoint has no qpos/qvel or navigation controls.  A failed release
        therefore prevents policy locomotion instead of silently walking while
        the simulator tether is in an unknown state.
        """

        if mode == self._plant_support_mode:
            return True
        body = (f'{{"mode":"{mode}"}}').encode("utf-8")
        http_request = request.Request(
            f"http://127.0.0.1:{self.config.plant_port}/support",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=0.20) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"plant support endpoint returned {response.status}"
                    )
                response.read(1024)
        except Exception as exc:
            self.telemetry.update(
                plant_support_mode="unavailable",
                last_error=f"MuJoCo support transition failed: {exc}",
            )
            return False
        self._plant_support_mode = mode
        self.telemetry.update(plant_support_mode=mode)
        return True

    def _apply_intent(self, dt: float, *, simulation_now_s: float):
        intent = self.guard.intent(dt=dt)
        controller = self._controller
        decoder = controller.controller
        # This is the sole ROS/Nav2 -> SONIC mapping.  It changes SONIC's
        # high-level MovementState; it never writes qpos or a motor command.
        from lerobot.robots.unitree_g1.controllers.sonic_pipeline import LM
        # Use SONIC's static IDLE mode while Nav2 is not commanding motion.  A
        # SLOW_WALK plan with ``has_movement=False`` still contains a locomotion
        # gait in the upstream planner, so leaving that mode selected would make
        # the simulated G1 drift even after the cmd_vel watchdog had stopped.
        # Switch to SLOW_WALK only for an actual Nav2 command and request a new
        # reference whenever the hold/move boundary changes.
        previous = self._intent_gate.last
        # Replanning is part of the learned controller's state evolution.  Its
        # debounce window must therefore use the same authoritative MuJoCo
        # clock as the policy cursor, handoff blend, and arm slew guard.  Wall
        # time remains valid only for ingress freshness/watchdog decisions.
        admitted = self._intent_gate.admit(intent, now=simulation_now_s)
        starting_motion = intent.mode != "IDLE" and (previous is None or previous.mode == "IDLE")
        if intent.mode == "IDLE":
            if previous is None or previous.mode != "IDLE":
                # Re-anchor SONIC's decoder to the measured IMU heading while
                # the actuator side is held static.  The next Nav2 action then
                # starts from the robot's real orientation rather than a stale
                # decoder reference left by a previous task.
                decoder.reinit_heading = True
                decoder.delta_heading = 0.0
        elif starting_motion:
            decoder.reinit_heading = True
            decoder.delta_heading = 0.0
            # ``step()`` consumes reinit_heading later in this control tick and
            # resets delta_heading.  The planner keeps one cumulative canonical
            # facing across actions, while this freshly re-anchored decoder must
            # receive only the new action's relative increment.
            previous_facing = (
                0.0 if self._decoder_facing_rad is None else self._decoder_facing_rad
            )
            self._pending_decoder_heading_delta_rad = self._angular_delta(
                intent.facing_angle_rad,
                previous_facing,
            )
            self._decoder_facing_rad = intent.facing_angle_rad
        elif self._decoder_facing_rad is not None:
            delta = self._angular_delta(intent.facing_angle_rad, self._decoder_facing_rad)
            decoder.delta_heading += delta
            self._decoder_facing_rad = intent.facing_angle_rad
        if admitted:
            controller.ms.mode = LM.SLOW_WALK if intent.mode == "SLOW_WALK" else LM.IDLE
            controller.ms.speed = intent.speed_mps if intent.has_movement else -1.0
            controller.ms.movement_angle = intent.movement_angle_rad
            controller.ms.facing_angle = intent.facing_angle_rad
            controller.ms.has_movement = intent.has_movement
            controller.ms.needs_replan = True
            self._last_intent_signature = (
                intent.has_movement,
                round(intent.movement_angle_rad, 3),
                round(intent.facing_angle_rad, 3),
                round(intent.speed_mps if intent.has_movement else -1.0, 3),
            )
        else:
            controller.ms.needs_replan = False
        received_at = self.guard.last_received_at
        self.telemetry.update(
            sonic_mode=intent.mode,
            sonic_speed_mps=intent.speed_mps,
            sonic_movement_angle_rad=intent.movement_angle_rad,
            sonic_facing_angle_rad=intent.facing_angle_rad,
            sonic_decoder_heading_delta_rad=float(decoder.delta_heading),
            cmd_vel_age_s=(None if received_at is None else max(0.0, time.monotonic() - received_at)),
        )
        return intent

    @staticmethod
    def _angular_delta(value: float, previous: float) -> float:
        return (float(value) - float(previous) + math.pi) % (2.0 * math.pi) - math.pi

    def _action_targets(self, action: dict[str, Any], lowstate) -> tuple[float, ...]:
        return tuple(
            float(action.get(f"{self._joint_name(index)}.q", lowstate.motor_state[index].q))
            for index in range(29)
        )

    def _publish(
        self,
        action: dict[str, Any],
        lowstate,
        *,
        control_dt_s: float,
        constrain_arms: bool = True,
    ) -> tuple[float, ...]:
        # Populate the official Unitree lowcmd message.  The simulator consumes
        # this exact DDS message, just as a G1 hardware bridge would.
        self._msg.mode_machine = getattr(lowstate, "mode_machine", 0)
        arm_names = [self._joint_name(index) for index in range(15, 29)]
        measured_arm = [float(lowstate.motor_state[index].q) for index in range(15, 29)]
        raw_arm = [
            float(action.get(f"{name}.q", measured_arm[offset]))
            for offset, name in enumerate(arm_names)
        ]
        from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

        nominal_arm = DEFAULT_ANGLES[15:29]
        raw_max_deviation = max(
            abs(raw - float(nominal))
            for raw, nominal in zip(raw_arm, nominal_arm, strict=True)
        )
        raw_max_error_from_measured = max(
            abs(raw - measured_value)
            for raw, measured_value in zip(raw_arm, measured_arm, strict=True)
        )
        measured_max_deviation = max(
            abs(measured_value - float(nominal))
            for measured_value, nominal in zip(measured_arm, nominal_arm, strict=True)
        )
        arm_velocities = [
            abs(float(lowstate.motor_state[index].dq))
            for index in range(15, 29)
        ]
        worst_arm_velocity_offset = max(
            range(len(arm_velocities)),
            key=arm_velocities.__getitem__,
        )
        arm_torques = [
            abs(float(lowstate.motor_state[index].tau_est))
            for index in range(15, 29)
        ]
        guarded_arm = (
            self._arm_guard.apply(
                raw_arm,
                measured=measured_arm,
                dt=control_dt_s,
            )
            if constrain_arms
            else None
        )
        guarded_by_index = (
            dict(zip(range(15, 29), guarded_arm.targets, strict=True))
            if guarded_arm is not None
            else {}
        )
        raw_targets = [
            float(
                action.get(
                    f"{self._joint_name(index)}.q",
                    lowstate.motor_state[index].q,
                )
            )
            for index in range(29)
        ]
        published_targets = tuple(
            float(
                guarded_by_index[index]
                if index in guarded_by_index
                else raw_targets[index]
            )
            for index in range(29)
        )
        for index in range(29):
            motor = self._msg.motor_cmd[index]
            motor.mode = 1
            motor.kp = float(self._kp[index])
            motor.kd = float(self._kd[index])
            motor.q = published_targets[index]
            motor.dq = 0.0
            motor.tau = 0.0
        self._msg.crc = self._crc.Crc(self._msg)
        self._publisher.Write(self._msg)
        nonzero = any(abs(float(self._msg.motor_cmd[i].q)) > 1e-4 for i in range(29))
        # Legs and waist are the locomotion plant.  Tracking their commanded
        # joint references separately from the arms tells an operator whether
        # SONIC is asking for too little motion or the simulated actuators are
        # failing to realize a perfectly valid gait reference.
        lower_body_errors = [
            abs(raw_targets[index] - float(lowstate.motor_state[index].q))
            for index in range(15)
        ]
        worst_lower_body_index = max(
            range(15),
            key=lower_body_errors.__getitem__,
        )
        lower_body_rms = math.sqrt(
            sum(error * error for error in lower_body_errors)
            / len(lower_body_errors)
        )
        lower_body_velocities = [
            abs(float(lowstate.motor_state[index].dq))
            for index in range(15)
        ]
        lower_body_torques = [
            abs(float(lowstate.motor_state[index].tau_est))
            for index in range(15)
        ]
        self.telemetry.update(
            joint_target_nonzero=nonzero,
            navigation_arm_constrained_joints=(0 if guarded_arm is None else guarded_arm.constrained_joints),
            navigation_arm_max_step_rad=(0.0 if guarded_arm is None else guarded_arm.max_step_rad),
            navigation_arm_raw_max_deviation_rad=raw_max_deviation,
            navigation_arm_raw_max_error_from_measured_rad=(
                raw_max_error_from_measured
            ),
            navigation_arm_measured_max_deviation_rad=measured_max_deviation,
            navigation_arm_max_velocity_rps=arm_velocities[
                worst_arm_velocity_offset
            ],
            navigation_arm_max_estimated_torque_nm=max(arm_torques),
            navigation_arm_worst_velocity_joint=arm_names[
                worst_arm_velocity_offset
            ],
            lower_body_target_max_error_rad=lower_body_errors[
                worst_lower_body_index
            ],
            lower_body_target_rms_error_rad=lower_body_rms,
            lower_body_worst_tracking_joint=self._joint_name(
                worst_lower_body_index
            ),
            lower_body_max_velocity_rps=max(lower_body_velocities),
            lower_body_max_estimated_torque_nm=max(lower_body_torques),
        )
        return published_targets

    @staticmethod
    def _joint_name(index: int) -> str:
        from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
        return G1_29_JointIndex(index).name

    def _record_control_trace(
        self,
        *,
        lowstate,
        control_step: PlantControlStep,
        intent,
        policy_targets: tuple[float, ...],
        handoff_targets: tuple[float, ...],
        published_targets: tuple[float, ...],
        handoff_ratio: float,
    ) -> None:
        states = lowstate.motor_state
        self._fault_trace.record(
            plant_tick_ms=control_step.tick_ms,
            control_dt_s=control_step.dt_s,
            intent={
                "mode": intent.mode,
                "speed_mps": intent.speed_mps,
                "movement_angle_rad": intent.movement_angle_rad,
                "facing_angle_rad": intent.facing_angle_rad,
                "has_movement": intent.has_movement,
                "stale": intent.stale,
            },
            handoff_ratio=handoff_ratio,
            plant_support_mode=self._plant_support_mode or "unknown",
            measured_q_rad=tuple(float(states[index].q) for index in range(29)),
            measured_dq_rps=tuple(float(states[index].dq) for index in range(29)),
            measured_ddq_rps2=tuple(
                float(getattr(states[index], "ddq", 0.0)) for index in range(29)
            ),
            measured_tau_nm=tuple(
                float(states[index].tau_est) for index in range(29)
            ),
            policy_target_q_rad=policy_targets,
            handoff_target_q_rad=handoff_targets,
            published_target_q_rad=published_targets,
        )

    def run(self):
        self.telemetry.update(status="running")
        self._arm_simulation_clock()
        while not self._stop.is_set():
            sample = self._next_control_sample()
            if self._fault_latched.is_set():
                with self._lowstate_lock:
                    lowstate = self._lowstate
                try:
                    if lowstate is not None:
                        self._publish_measured_hold(lowstate)
                finally:
                    self._stop.set()
                break
            if sample is None:
                continue
            lowstate, control_step = sample
            try:
                intent = self._apply_intent(
                    control_step.dt_s,
                    simulation_now_s=control_step.tick_ms / 1_000.0,
                )
                # Feed empty external action: the controller reads the mapped
                # MovementState above and executes the official SONIC pipeline.
                # This call, its history, and its reference cursor advance only
                # after 20 ms of authoritative MuJoCo time.
                policy_action = self._controller.run_step({}, lowstate)
                self.telemetry.update(
                    sonic_policy_speed_mps=(
                        max(0.0, float(self._controller.ms.speed))
                        if intent.mode != "IDLE"
                        else 0.0
                    ),
                )
                if self._pending_decoder_heading_delta_rad:
                    # ``run_step`` has now consumed ``reinit_heading`` and
                    # reset the decoder delta.  Apply the bounded Nav2 Spin
                    # look-ahead after that reset, before the next policy tick.
                    self._controller.controller.delta_heading += self._pending_decoder_heading_delta_rad
                    self._pending_decoder_heading_delta_rad = 0.0
                measured = tuple(
                    float(lowstate.motor_state[index].q) for index in range(29)
                )
                policy_targets = self._action_targets(policy_action, lowstate)
                decision = self._joint_handoff.decide(
                    motion_active=intent.mode != "IDLE",
                    measured=measured,
                    policy=policy_targets,
                    now=control_step.tick_ms / 1_000.0,
                )
                if decision.static_hold_active:
                    desired_support_mode = "idle"
                elif intent.mode != "IDLE" and decision.handoff_ratio > 0.0:
                    desired_support_mode = "locomotion"
                else:
                    desired_support_mode = "idle"
                support_ready = self._set_plant_support_mode(
                    desired_support_mode
                )
                if not support_ready and desired_support_mode == "locomotion":
                    raise RuntimeError(
                        "MuJoCo support transition failed during locomotion"
                    )
                if decision.static_hold_active and not self._static_hold_active:
                    self._arm_guard.reset()
                self._static_hold_active = decision.static_hold_active
                output_action = {
                    f"{self._joint_name(index)}.q": target
                    for index, target in enumerate(decision.targets)
                }
                published_targets = self._publish(
                    output_action,
                    lowstate,
                    control_dt_s=control_step.dt_s,
                    constrain_arms=not decision.static_hold_active,
                )
                self._record_control_trace(
                    lowstate=lowstate,
                    control_step=control_step,
                    intent=intent,
                    policy_targets=policy_targets,
                    handoff_targets=decision.targets,
                    published_targets=published_targets,
                    handoff_ratio=decision.handoff_ratio,
                )
                self.telemetry.update(
                    control_tick_ms=control_step.tick_ms,
                    control_dt_s=control_step.dt_s,
                    static_hold_active=decision.static_hold_active,
                    motion_handoff_active=decision.motion_handoff_active,
                    motion_handoff_ratio=decision.handoff_ratio,
                )
            except Exception as exc:
                self._latch_terminal_fault(f"SONIC runtime fault: {exc}")
                try:
                    self._publish_measured_hold(lowstate)
                finally:
                    self._stop.set()
                break
        try:
            self._controller.shutdown()
        finally:
            if not self._fault_latched.is_set():
                self.telemetry.update(status="stopped")

    def stop(self):
        self._stop.set()
        self.guard.stop()
        condition = getattr(self, "_control_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()

    def accept_json(self, payload: dict[str, Any]) -> None:
        if self._fault_latched.is_set():
            self.guard.stop(source="terminal_fault")
            return
        if payload.get("type") == "stop":
            self.guard.stop(source="bridge_stop")
            self._motion_armed = True
            self._navigation_task_active = False
            reason = payload.get("reason")
            self.telemetry.update(
                nav2_state="safety_stop" if reason else "idle",
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                last_error=str(reason) if reason else None,
            )
            return
        if payload.get("type") != "cmd_vel":
            raise ValueError("unsupported command type")
        vx = float(payload.get("vx", 0.0))
        vy = float(payload.get("vy", 0.0))
        wz = float(payload.get("wz", 0.0))
        if payload.get("heading_yaw") is not None:
            self.guard.observe_heading(float(payload["heading_yaw"]))
        nonzero = any(abs(value) > 1e-6 for value in (vx, vy, wz))
        if nonzero and not self._motion_armed:
            self.guard.stop(source="plant_restart_interlock")
            self.telemetry.update(
                nav2_state="restart_hold",
                last_cmd_vx=vx,
                last_cmd_vy=vy,
                last_cmd_wz=wz,
                last_error="non-zero Nav2 command rejected until an idle/stop command is observed",
            )
            return
        if not nonzero:
            self._motion_armed = True
        elif not self._navigation_task_active:
            self.telemetry.reset_navigation_task_peaks()
            self._navigation_task_active = True
            self.telemetry.update(navigation_task_active=True)
        self.guard.accept(vx, vy, wz, source=str(payload.get("source", "nav2")))
        self.telemetry.update(
            nav2_state="commanding" if nonzero else "idle",
            last_cmd_vx=vx,
            last_cmd_vy=vy,
            last_cmd_wz=wz,
            last_error=None,
        )


def run_driver(config: SimConfig) -> None:
    telemetry = TelemetryStore(TelemetryState(robot_id=config.robot_id, ros_domain_id=config.ros_domain_id))
    serve_telemetry(telemetry, config.telemetry_bind_host, config.telemetry_port)
    driver = SonicDriver(config, telemetry)

    # A tiny localhost TCP command server.  ROS bridge is the only intended
    # client; it sends one JSON object per line and never receives actuator data.
    import socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((config.command_bind_host, config.command_port))
    server.listen(4)
    server.settimeout(0.2)

    def accept_loop():
        while not driver._stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            with conn:
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                try:
                    messages, _ = parse_lines(data + (b"" if data.endswith(b"\n") else b"\n"))
                    for message in messages:
                        driver.accept_json(message)
                except Exception as exc:
                    logger.warning("invalid bridge command: %s", exc)

    threading.Thread(target=accept_loop, daemon=True, name="sonic-command-server").start()
    try:
        driver.run()
    finally:
        driver.stop()
        server.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_driver(SimConfig.from_file(args.config))
