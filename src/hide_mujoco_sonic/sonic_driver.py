"""Unitree DDS SONIC driver.

This process is intentionally separate from the MuJoCo plant.  The plant and
driver communicate only through the official `rt/lowstate`/`rt/lowcmd` DDS
topics, which keeps the simulation boundary close to the real Unitree stack.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from .config import SimConfig
from .motion_fault import (
    SONIC_HEALTH_SCHEMA_VERSION,
    validate_sonic_health_request,
)
from .network import require_loopback_bind_host
from .planner_evidence import (
    PlannerReferenceDeadlineExceeded,
    PlannerReferenceEvidenceGate,
)
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
from .transport import parse_lines, rearm_plant_support, set_plant_support

logger = logging.getLogger(__name__)


def _bind_command_server(host: str, port: int):
    """Create the ROS-bridge command socket with loopback defense in depth."""

    import socket

    bind_host = require_loopback_bind_host(host, field="command_bind_host")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_host, port))
        server.listen(4)
        server.settimeout(0.2)
    except Exception:
        server.close()
        raise
    return server


class _WarmupPlannerIsolation:
    """Planner boundary used by the sacrificial full-controller warm-up.

    A startup warm-up must exercise the exact ``run_step`` Python/ORT path, but
    it must never consume a real async planner result or enqueue a request that
    could later affect the live controller.  The first SONIC step is not
    expected to request a replan; raising here makes that invariant explicit.
    """

    @staticmethod
    def request_replan(_cursor, _movement_state) -> None:
        raise RuntimeError("isolated SONIC warm-up attempted to request a replan")

    @staticmethod
    def try_get_new_motion():
        return None


@dataclass(frozen=True, slots=True)
class IngressDecision:
    """Validated bridge command before any SONIC/model-specific mapping."""

    action: str
    generation: int | None
    sequence: int | None
    state: str
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    heading_yaw: float | None = None
    reason: str | None = None
    arm_acknowledged: bool = False


class SonicIntentIngress:
    """Fail-closed generation/sequence gate for gait-adapter wire intents.

    A newly observed bridge lifecycle is held at stop until it sends a measured
    ``IDLE`` acknowledgement.  The ROS bridge only emits that acknowledgement
    after its GaitAdapter has received a fresh, stationary ``base_link`` odom
    sample.  A driver never infers or auto-rearms this state itself.
    """

    _MOTION_STATES = frozenset({"STARTING", "WALKING", "TURNING"})
    _STOP_STATES = frozenset({"DISARMED", "STOPPING", "FAULT"})

    def __init__(self, *, clock=time.monotonic, max_age_s: float = 0.50) -> None:
        if not math.isfinite(float(max_age_s)) or max_age_s <= 0:
            raise ValueError("intent max age must be finite and positive")
        self._clock = clock
        self.max_age_s = float(max_age_s)
        self.generation: int | None = None
        self.sequence = 0
        self.armed = False
        self.support_stop_generation: int | None = None

    def revoke(self) -> None:
        """Drop motion authority without manufacturing a new generation."""

        self.armed = False

    def latch_support_stop(self) -> int:
        """Record the generation stopped by an externally confirmed Plant latch."""

        stopped = 0 if self.generation is None else self.generation
        if self.support_stop_generation is None:
            self.support_stop_generation = stopped
        else:
            self.support_stop_generation = max(
                self.support_stop_generation, stopped
            )
        self.armed = False
        return self.support_stop_generation

    def confirm_support_rearm(self, generation: int) -> None:
        if (
            self.support_stop_generation is None
            or generation <= self.support_stop_generation
            or generation != self.generation
            or not self.armed
        ):
            raise ValueError("support rearm confirmation does not match the armed generation")
        self.support_stop_generation = None

    @staticmethod
    def _integer(value: object, *, minimum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError("intent generation/sequence must be a positive integer")
        return value

    @staticmethod
    def _finite(payload: dict[str, Any], name: str) -> float:
        try:
            value = float(payload[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"intent {name} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"intent {name} must be finite")
        return value

    def accept(self, payload: dict[str, Any]) -> IngressDecision:
        if payload.get("type") != "sonic_intent":
            raise ValueError("motion input must use the sonic_intent contract")
        if payload.get("schema_version") != "h-ide.sonic-intent/v1":
            raise ValueError("unsupported sonic intent schema")
        if payload.get("source") != "nav2":
            raise ValueError("only Nav2 gait intents are accepted")

        generation = self._integer(payload.get("generation"), minimum=0)
        sequence = self._integer(payload.get("sequence"), minimum=1)
        generated_at = self._finite(payload, "generated_at_s")
        now = float(self._clock())
        if generated_at > now + 0.10 or now - generated_at > self.max_age_s:
            raise ValueError("sonic intent is stale or from the future")

        state = str(payload.get("state", ""))
        if state not in self._MOTION_STATES | self._STOP_STATES | {"IDLE"}:
            raise ValueError("unsupported gait adapter state")
        motion_enabled = payload.get("motion_enabled")
        stop_requested = payload.get("stop_requested")
        stop_acknowledged = payload.get("stop_acknowledged")
        if not all(
            isinstance(value, bool)
            for value in (motion_enabled, stop_requested, stop_acknowledged)
        ):
            raise ValueError("intent safety flags must be booleans")
        if payload.get("mode") not in {"IDLE", "SLOW_WALK"}:
            raise ValueError("unsupported SONIC mode")
        control_arm = payload.get("control_arm", False)
        if not isinstance(control_arm, bool):
            raise ValueError("control_arm must be a boolean")

        head_yaw = self._finite(payload, "head_yaw_rad")
        head_pitch = self._finite(payload, "head_pitch_rad")
        if abs(head_yaw) > 1e-9 or abs(head_pitch) > 1e-9:
            raise ValueError("navigation profile requires fixed zero head targets")

        heading: float | None = None
        if payload.get("measured_heading_yaw") is not None:
            heading = self._finite(payload, "measured_heading_yaw")

        if self.generation is not None and generation < self.generation:
            return IngressDecision(
                "ignore", generation, sequence, state, reason="old bridge generation ignored"
            )
        if generation != self.generation:
            self.generation = generation
            self.sequence = 0
            self.armed = False
        if sequence <= self.sequence:
            return IngressDecision(
                "ignore", generation, sequence, state, reason="out-of-order intent ignored"
            )
        self.sequence = sequence

        # Stop authority is evaluated before reading any motion values.  Even a
        # malformed non-zero tail on a stop envelope can therefore never reach
        # CmdVelGuard or the learned model.
        if stop_requested or not motion_enabled:
            arm_candidate = (
                state == "IDLE"
                and not stop_requested
                and bool(stop_acknowledged)
            )
            reason = str(payload["reason"]) if payload.get("reason") else None
            if control_arm and not arm_candidate:
                raise ValueError(
                    "control_arm requires a stopped, acknowledged IDLE intent"
                )
            if self.support_stop_generation is not None:
                if not control_arm:
                    arm_candidate = False
                    reason = "plant safety_stop requires explicit control.acquire rearm"
                elif generation <= self.support_stop_generation:
                    arm_candidate = False
                    reason = "control rearm requires a strictly newer generation"
            was_armed = self.armed
            self.armed = arm_candidate
            return IngressDecision(
                "stop",
                generation,
                sequence,
                state,
                heading_yaw=heading,
                reason=reason,
                arm_acknowledged=(arm_candidate and not was_armed),
            )

        if control_arm:
            raise ValueError("control_arm cannot carry motion")

        if not self.armed:
            return IngressDecision(
                "stop",
                generation,
                sequence,
                state,
                heading_yaw=heading,
                reason="motion rejected until fresh stationary IDLE acknowledgement",
            )
        if state not in self._MOTION_STATES or payload.get("mode") != "SLOW_WALK":
            raise ValueError("motion-enabled intent has an invalid gait state")

        vx = self._finite(payload, "velocity_x_mps")
        vy = self._finite(payload, "velocity_y_mps")
        wz = self._finite(payload, "yaw_rate_rps")
        linear = self._finite(payload, "linear_speed_mps")
        if abs(linear - math.hypot(vx, vy)) > 1e-6:
            raise ValueError("intent linear speed does not match its velocity vector")
        if not any(abs(value) > 1e-6 for value in (vx, vy, wz)):
            raise ValueError("motion-enabled intent contains no Nav2 motion")
        return IngressDecision(
            "motion", generation, sequence, state, vx, vy, wz, heading
        )


class SonicDriver:
    _STARTUP_QUALIFICATION_STEPS = 5

    def __init__(self, config: SimConfig, telemetry: TelemetryStore):
        self.config = config
        self.telemetry = telemetry
        self._driver_boot_id = uuid4()
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
        # ``_control_consumer_ready`` is only the callback/consumer handshake.
        # Motion remains blocked until ``_motion_ready`` is set after startup
        # qualification has observed a sustained real 50 Hz simulation stream.
        self._control_consumer_ready = threading.Event()
        self._motion_ready = threading.Event()
        self._startup_qualifying = False
        self._startup_qualification_required_steps = (
            self._STARTUP_QUALIFICATION_STEPS
        )
        self._startup_qualification_consecutive_steps = 0
        self._startup_qualification_dropped_samples = 0
        self._startup_qualification_drop_generation = 0
        self._pending_control_sample: (
            tuple[Any, PlantControlStep, int] | None
        ) = None
        self._wall_clock = time.monotonic
        self._control_clock_armed_at_wall_s: float | None = None
        self._last_lowstate_wall_s: float | None = None
        self._last_control_step_wall_s: float | None = None
        self._lowstate_stall_timeout_s = max(1.0, config.command_timeout_s * 4.0)
        self._stop = threading.Event()
        self._fault_latched = threading.Event()
        self._terminal_fault_reason: str | None = None
        self._tick_guard = PlantTickGuard()
        self._intent_ingress = SonicIntentIngress(
            max_age_s=max(0.50, config.command_timeout_s * 2.0)
        )
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
        self._plant_support_stop_pending = False
        self._plant_support_safety_latched = False
        self._plant_support_stop_generation: int | None = None
        self._last_published_targets: tuple[float, ...] | None = None
        self._safe_hold_targets: tuple[float, ...] | None = None
        self._support_lock = threading.RLock()

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
        self._planner_evidence = PlannerReferenceEvidenceGate(
            required_timeout_s=config.planner_reference_timeout_s
        )
        self._planner_evidence.bind(
            self._controller.controller.planner,
            reference_owner=self._controller.controller,
        )
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
            if not self._control_clock_armed:
                self._control_condition.notify_all()
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
                            if self._startup_qualifying:
                                # Before motion is armed, the plant is held at
                                # its measured pose and policy state is frozen.
                                # A cold scheduler delay may therefore discard
                                # an obsolete *stationary measurement* in favor
                                # of the latest one.  This is synchronization,
                                # not policy catch-up: no dropped sample is ever
                                # replayed or presented to the controller.
                                self._startup_qualification_dropped_samples += 1
                                self._startup_qualification_drop_generation += 1
                                self._startup_qualification_consecutive_steps = 0
                                self._pending_control_sample = (
                                    message,
                                    control_step,
                                    self._startup_qualification_drop_generation,
                                )
                                self._control_condition.notify_all()
                            else:
                                pending_tick = self._pending_control_sample[1].tick_ms
                                clock_fault = (
                                    "MuJoCo control sample overrun: "
                                    f"{pending_tick}ms was not consumed before "
                                    f"{control_step.tick_ms}ms; catch-up is forbidden"
                                )
                        else:
                            self._pending_control_sample = (
                                message,
                                control_step,
                                self._startup_qualification_drop_generation,
                            )
                            self._control_condition.notify_all()
        self.telemetry.update(
            plant_tick=tick,
            startup_qualification_dropped_samples=(
                self._startup_qualification_dropped_samples
            ),
            startup_qualification_consecutive_steps=(
                self._startup_qualification_consecutive_steps
            ),
        )
        if clock_fault is not None:
            # Any clock discontinuity invalidates policy history and actuator
            # continuity.  Never rebuild history or replay missed policy ticks.
            self._latch_terminal_fault(clock_fault)

    def _wait_for_initial_lowstate(self, *, timeout_s: float = 5.0):
        """Wait for plant state while the strict control clock remains disarmed."""

        deadline = self._wall_clock() + timeout_s
        with self._control_condition:
            while (
                self._lowstate is None
                and not self._fault_latched.is_set()
                and not self._stop.is_set()
            ):
                remaining = deadline - self._wall_clock()
                if remaining <= 0:
                    break
                self._control_condition.wait(timeout=min(0.10, remaining))
            if self._lowstate is None:
                raise RuntimeError("timed out waiting for initial MuJoCo LowState")
            return self._lowstate

    @staticmethod
    def _clone_state_owner(owner, *, excluded: frozenset[str]):
        """Shallow-clone an owner, then detach all explicitly mutable state."""

        clone = copy.copy(owner)
        for name, value in vars(owner).items():
            if name not in excluded:
                setattr(clone, name, copy.deepcopy(value))
        return clone

    @classmethod
    def _isolated_warmup_controller(cls, controller):
        """Clone policy state while sharing only immutable/read-only ORT sessions.

        Constructing another official controller would load three new sessions
        and leave *that* new instance cold.  Instead this clone shares the live
        encoder/decoder session objects, but owns every history/reference array,
        movement state and cursor.  Its planner is a no-I/O boundary, so the
        async live planner queues cannot be observed or changed by warm-up.
        """

        if getattr(controller, "_smpl_stream", None) is not None:
            raise RuntimeError(
                "isolated SONIC warm-up requires the navigation profile without SMPL stream"
            )
        runtime = controller._runtime
        pipeline = runtime.controller
        if controller.controller is not pipeline or controller.ms is not runtime.ms:
            raise RuntimeError("unexpected SONIC controller/runtime alias layout")
        if int(runtime.step) != 0:
            raise RuntimeError("SONIC full-path warm-up must precede every policy step")

        warm_pipeline = cls._clone_state_owner(
            pipeline,
            excluded=frozenset({"encoder", "decoder", "planner", "motion_lock"}),
        )
        # ORT InferenceSession objects are intentionally shared: exercising the
        # same sessions is the purpose of the cold-path warm-up.
        warm_pipeline.encoder = pipeline.encoder
        warm_pipeline.decoder = pipeline.decoder
        warm_pipeline.motion_lock = threading.Lock()

        warm_ms = copy.deepcopy(runtime.ms)
        warm_planner = _WarmupPlannerIsolation()
        warm_pipeline.planner = warm_planner

        warm_runtime = cls._clone_state_owner(
            runtime,
            excluded=frozenset({"controller", "planner", "ms"}),
        )
        warm_runtime.controller = warm_pipeline
        warm_runtime.planner = warm_planner
        warm_runtime.ms = warm_ms

        warm_controller = cls._clone_state_owner(
            controller,
            excluded=frozenset(
                {"_runtime", "controller", "ms", "_smpl_stream"}
            ),
        )
        warm_controller._runtime = warm_runtime
        warm_controller.controller = warm_pipeline
        warm_controller.ms = warm_ms
        warm_controller._smpl_stream = None
        return warm_controller

    @classmethod
    def _policy_state_digest(cls, controller) -> str:
        """Digest all live policy/history state, excluding sessions and locks."""

        import numpy as np

        digest = hashlib.sha256()

        def update(value) -> None:
            digest.update(type(value).__qualname__.encode("utf-8"))
            digest.update(b"\0")
            if value is None or isinstance(value, (bool, int, float, str, bytes)):
                digest.update(repr(value).encode("utf-8"))
            elif isinstance(value, np.ndarray):
                digest.update(value.dtype.str.encode("ascii"))
                digest.update(repr(value.shape).encode("ascii"))
                digest.update(np.ascontiguousarray(value).tobytes())
            elif isinstance(value, dict):
                for key in sorted(value, key=repr):
                    update(key)
                    update(value[key])
            elif isinstance(value, (list, tuple, deque)):
                for item in value:
                    update(item)
            elif isinstance(value, (set, frozenset)):
                for item in sorted(value, key=repr):
                    update(item)
            elif hasattr(value, "__dict__"):
                update(vars(value))
            else:
                digest.update(repr(value).encode("utf-8"))
            digest.update(b"\xff")

        runtime = controller._runtime
        pipeline = runtime.controller
        sections = {
            "wrapper": {
                name: value
                for name, value in vars(controller).items()
                if name
                not in {"_runtime", "controller", "ms", "_smpl_stream", "kp", "kd"}
            },
            "runtime": {
                name: value
                for name, value in vars(runtime).items()
                if name not in {"controller", "planner", "ms", "kp", "kd"}
            },
            "pipeline": {
                name: value
                for name, value in vars(pipeline).items()
                if name not in {"encoder", "decoder", "planner", "motion_lock"}
            },
            "movement_state": vars(runtime.ms),
        }
        update(sections)
        return digest.hexdigest()

    def _warm_controller_inference(self, lowstate=None) -> None:
        """Warm one complete controller step without advancing live policy state."""

        import numpy as np

        if lowstate is None:
            lowstate = self._lowstate
        if lowstate is None:
            raise RuntimeError("SONIC full-path warm-up requires a LowState")

        self.telemetry.update(control_clock_status="warming_full_controller_path")
        started_at = self._wall_clock()
        state_before = self._policy_state_digest(self._controller)
        warm_controller = self._isolated_warmup_controller(self._controller)
        output = warm_controller.run_step({}, lowstate)
        values = np.asarray(list(output.values()), dtype=np.float64)
        if len(output) != 29 or values.shape != (29,) or not np.all(np.isfinite(values)):
            raise RuntimeError(
                "SONIC full-controller warm-up did not produce 29 finite targets"
            )
        state_after = self._policy_state_digest(self._controller)
        if state_after != state_before:
            raise RuntimeError("SONIC warm-up contaminated live controller state")

        elapsed = max(0.0, self._wall_clock() - started_at)
        logger.info(
            "SONIC full-controller warm-up complete in %.3fs; live policy state unchanged",
            elapsed,
        )
        self.telemetry.update(
            controller_warmup_complete=True,
            controller_warmup_wall_s=elapsed,
            controller_warmup_state_unchanged=True,
            control_clock_status="warm_ready_to_qualify",
        )

    def _prepare_idle_startup(self) -> None:
        """Hold the plant and warm inference before strict clock handoff."""

        lowstate = self._wait_for_initial_lowstate()
        if self._fault_latched.is_set():
            raise RuntimeError(self._terminal_fault_reason or "startup fault latched")
        if not self._set_plant_support_mode("idle"):
            raise RuntimeError("MuJoCo idle support is unavailable during SONIC warm-up")
        self._publish_measured_hold(lowstate)
        self._warm_controller_inference(lowstate)
        if self._fault_latched.is_set():
            raise RuntimeError(self._terminal_fault_reason or "startup fault latched")

    def _next_control_sample(
        self,
        *,
        wait_timeout_s: float = 0.10,
    ) -> tuple[Any, PlantControlStep, int] | None:
        """Consume at most one admitted sample; wall time is only a wake timeout."""

        clock_fault: str | None = None
        with self._control_condition:
            if (
                not self._control_clock_armed
                and not self._fault_latched.is_set()
                and not self._stop.is_set()
            ):
                # This is the startup handshake.  The current LowState is only
                # a time baseline; it is deliberately discarded rather than
                # queued as a policy step.  Because this happens while holding
                # the condition lock, the callback cannot observe the armed
                # flag until this consumer has entered ``wait`` below.
                if self._lowstate is None:
                    raise RuntimeError("cannot arm SONIC clock without LowState")
                self._control_clock.reset()
                baseline = self._control_clock.observe(int(self._lowstate.tick))
                assert baseline is not None
                self._pending_control_sample = None
                self._control_clock_armed_at_wall_s = self._wall_clock()
                self._startup_qualifying = True
                self._startup_qualification_consecutive_steps = 0
                self._startup_qualification_dropped_samples = 0
                self._startup_qualification_drop_generation = 0
                self._motion_ready.clear()
                self._control_clock_armed = True
                self._control_consumer_ready.set()
                self.telemetry.update(
                    control_clock_status="startup_qualification_waiting_first_step",
                    startup_qualification_required_steps=(
                        self._startup_qualification_required_steps
                    ),
                    startup_qualification_consecutive_steps=0,
                    startup_qualification_dropped_samples=0,
                    motion_ready=False,
                )
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

    def _complete_startup_qualification_step(
        self,
        *,
        sample_drop_generation: int,
    ) -> bool:
        """Credit one stationary sample only if no newer sample overran it.

        The caller publishes a measured-position hold before invoking this
        method.  If two new control samples arrived during that work, the
        callback has already replaced the obsolete pending sample and advanced
        the drop generation; this step then earns no credit.  No policy,
        handoff, arm guard, replan timer or reference cursor advances here.
        """

        with self._control_condition:
            if not self._startup_qualifying:
                return False
            if (
                sample_drop_generation
                != self._startup_qualification_drop_generation
            ):
                self._startup_qualification_consecutive_steps = 0
            else:
                self._startup_qualification_consecutive_steps += 1
            complete = (
                self._startup_qualification_consecutive_steps
                >= self._startup_qualification_required_steps
            )
            if complete:
                self._startup_qualifying = False
                self._motion_ready.set()
            consecutive = self._startup_qualification_consecutive_steps
            dropped = self._startup_qualification_dropped_samples

        self.telemetry.update(
            status="running" if complete else "qualifying",
            control_clock_status=(
                "simulation_time_50hz_motion_ready"
                if complete
                else "startup_qualification"
            ),
            startup_qualification_consecutive_steps=consecutive,
            startup_qualification_dropped_samples=dropped,
            motion_ready=complete,
        )
        if complete:
            logger.info(
                "SONIC startup qualified after %d consecutive simulation-time steps; motion may now be accepted",
                consecutive,
            )
        return complete

    def _publish_position_hold(
        self,
        targets: tuple[float, ...],
    ) -> None:
        if len(targets) != 29 or not all(
            math.isfinite(float(value)) for value in targets
        ):
            raise ValueError("position hold requires 29 finite joint targets")
        with self._support_lock:
            selected = self._safe_hold_targets or targets
            for index in range(29):
                motor = self._msg.motor_cmd[index]
                motor.mode = 1
                motor.kp = float(self._kp[index])
                motor.kd = float(self._kd[index])
                motor.q = float(selected[index])
                motor.dq = 0.0
                motor.tau = 0.0
            self._msg.crc = self._crc.Crc(self._msg)
            self._publisher.Write(self._msg)
            self._last_published_targets = tuple(
                float(value) for value in selected
            )
            self._arm_guard.anchor(self._last_published_targets[15:29])
        self._mark_control_step_published()

    def _mark_control_step_published(self) -> None:
        """Record only a successful actuator write on the monotonic clock."""

        published_at = self._wall_clock()
        with self._control_condition:
            self._last_control_step_wall_s = published_at

    def _publish_measured_hold(self, lowstate) -> None:
        self._publish_position_hold(
            tuple(float(lowstate.motor_state[index].q) for index in range(29))
        )

    def _freeze_last_published_hold(self) -> bool:
        """Keep the actuator boundary continuous across a stop transition."""

        self._revoke_planner_reference("actuator boundary entered static hold")
        with self._support_lock:
            targets = self._last_published_targets
            if targets is None:
                return False
            self._safe_hold_targets = targets
            self._joint_handoff.freeze(targets)
            self._arm_guard.anchor(targets[15:29])
            return True

    def _freeze_planner_handoff_boundary(self) -> bool:
        """Start a new planner generation from the last actuator target.

        A matching SONIC reference can arrive asynchronously after Nav2 has
        changed its intent.  Until then, and on the first admitted reference
        frame, the handoff must retain the exact target that most recently
        crossed DDS rather than re-anchor to the measured joint angles.
        Unlike a physical stop this does not latch ``_safe_hold_targets`` or
        revoke motion authority; it only establishes the blend boundary.
        """

        with self._support_lock:
            targets = self._last_published_targets
            if targets is None:
                return False
            self._joint_handoff.freeze(targets)
            self._arm_guard.anchor(targets[15:29])
            return True

    def _publish_frozen_hold(self, lowstate) -> None:
        with self._support_lock:
            targets = self._safe_hold_targets or self._last_published_targets
        if targets is None:
            self._publish_measured_hold(lowstate)
            return
        self._publish_position_hold(targets)

    def _latch_terminal_fault(self, reason: str) -> None:
        """Latch a non-recoverable process fault without rebuilding state."""

        if self._fault_latched.is_set():
            return
        self._terminal_fault_reason = str(reason)
        self._freeze_last_published_hold()
        self.guard.stop(source="terminal_fault")
        self._navigation_task_active = False
        self._startup_qualifying = False
        motion_ready = getattr(self, "_motion_ready", None)
        if motion_ready is not None:
            motion_ready.clear()
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
            motion_ready=False,
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

    def _latch_plant_support_stop(self, *, reason: str) -> int:
        """Synchronize an externally confirmed Plant stop into SONIC.

        This method never releases support.  It invalidates the current ingress
        authority and records the stopped generation so periodic IDLE traffic,
        an old ``/cmd_vel``, or a bridge reconnect cannot clear the latch.
        """

        self._freeze_last_published_hold()
        with self._support_lock:
            stopped_generation = self._intent_ingress.latch_support_stop()
            self._plant_support_mode = "safety_stop"
            self._plant_support_stop_pending = False
            self._plant_support_safety_latched = True
            self._plant_support_stop_generation = stopped_generation
        self.guard.stop(source="plant_safety_stop")
        self._navigation_task_active = False
        self._arm_guard.reset()
        self.telemetry.update(
            nav2_state="safety_stop",
            navigation_task_active=False,
            last_cmd_vx=0.0,
            last_cmd_vy=0.0,
            last_cmd_wz=0.0,
            static_hold_active=True,
            plant_support_mode="safety_stop",
            plant_support_requested_mode="safety_stop",
            plant_support_confirmed_mode="safety_stop",
            plant_support_stop_pending=False,
            plant_support_safety_latched=True,
            plant_support_stop_generation=stopped_generation,
            last_error=reason,
        )
        return stopped_generation

    def _ensure_plant_support_stopped(self, *, reason: str) -> bool:
        """Strengthen Plant support after an uncertain/racing transition.

        A lost HTTP acknowledgement can mean that a rearm reached the Plant
        even though SONIC saw a timeout.  Always issue the stronger idempotent
        safety transition before reporting failure.  If that confirmation is
        also unavailable, software authority remains revoked and telemetry
        reports the Plant as unavailable rather than guessing its state.
        """

        self._freeze_last_published_hold()
        try:
            state = set_plant_support(
                "127.0.0.1",
                self.config.plant_port,
                "safety_stop",
                timeout=0.30,
            )
            if state.get("mode") != "safety_stop":
                raise RuntimeError("Plant did not confirm safety_stop")
        except Exception as exc:
            with self._support_lock:
                stopped_generation = self._intent_ingress.latch_support_stop()
                self._plant_support_mode = None
                self._plant_support_stop_pending = True
                self._plant_support_safety_latched = True
                self._plant_support_stop_generation = stopped_generation
            self.guard.stop(source="plant_safety_stop_unconfirmed")
            self._navigation_task_active = False
            self._arm_guard.reset()
            self.telemetry.update(
                nav2_state="safety_stop",
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                static_hold_active=True,
                plant_support_mode="unavailable",
                plant_support_requested_mode="safety_stop",
                plant_support_confirmed_mode="unavailable",
                plant_support_stop_pending=True,
                plant_support_safety_latched=True,
                plant_support_stop_generation=stopped_generation,
                last_error=f"{reason}; Plant safety-stop confirmation failed: {exc}",
            )
            return False
        self._latch_plant_support_stop(reason=reason)
        return True

    def _rearm_plant_support(self, generation: int) -> bool:
        """Confirm one explicit control.acquire rearm, only back to IDLE."""

        with self._support_lock:
            stopped_generation = self._plant_support_stop_generation
            if (
                not self._plant_support_safety_latched
                or self._plant_support_stop_pending
                or stopped_generation is None
                or generation <= stopped_generation
                or generation != self._intent_ingress.generation
                or not self._intent_ingress.armed
            ):
                self.telemetry.update(
                    plant_support_mode="safety_stop",
                    plant_support_requested_mode="idle",
                    plant_support_confirmed_mode="safety_stop",
                    plant_support_safety_latched=True,
                    last_error=(
                        "plant safety_stop rearm rejected without a strictly "
                        "newer armed generation"
                    ),
                )
                return False
        try:
            state = rearm_plant_support(
                "127.0.0.1",
                self.config.plant_port,
                generation,
                timeout=0.30,
            )
        except Exception as exc:
            self._ensure_plant_support_stopped(
                reason=f"MuJoCo support rearm failed: {exc}"
            )
            return False
        with self._support_lock:
            try:
                self._intent_ingress.confirm_support_rearm(generation)
            except ValueError as exc:
                # The Plant has moved only to IDLE, never locomotion.  Relatch
                # immediately if the ingress generation changed during the
                # request; a failed relatch is surfaced as unavailable/faulted
                # rather than treating the stale release as motion authority.
                self._ensure_plant_support_stopped(
                    reason=f"Plant rearm raced ingress: {exc}"
                )
                return False
            self._plant_support_mode = str(state["mode"])
            self._plant_support_stop_pending = False
            self._plant_support_safety_latched = False
            self._plant_support_stop_generation = None
        self._arm_guard.reset()
        self.telemetry.update(
            nav2_state="idle",
            navigation_task_active=False,
            plant_support_mode="idle",
            plant_support_requested_mode="idle",
            plant_support_confirmed_mode="idle",
            plant_support_stop_pending=False,
            plant_support_safety_latched=False,
            plant_support_stop_generation=None,
            plant_support_rearm_generation=generation,
            last_error=None,
        )
        return True

    def _set_plant_support_mode(self, mode: str) -> bool:
        """Switch only MuJoCo's confirmed fall-arrest profile over loopback."""

        self.telemetry.update(plant_support_requested_mode=mode)
        with self._support_lock:
            if (
                self._plant_support_stop_pending
                or self._plant_support_safety_latched
            ):
                # safety_stop is a stronger static hold than IDLE.  It remains
                # latched until _rearm_plant_support confirms control.acquire;
                # the 50 Hz IDLE loop must never write it away.
                if mode in {"idle", "safety_stop"}:
                    confirmed_mode = (
                        "safety_stop"
                        if self._plant_support_safety_latched
                        else self._plant_support_mode or "unknown"
                    )
                    self.telemetry.update(
                        plant_support_mode=confirmed_mode,
                        plant_support_requested_mode="safety_stop",
                        plant_support_confirmed_mode=confirmed_mode,
                        plant_support_stop_pending=self._plant_support_stop_pending,
                        plant_support_safety_latched=self._plant_support_safety_latched,
                    )
                    return True
                confirmed_mode = (
                    "safety_stop"
                    if self._plant_support_safety_latched
                    else self._plant_support_mode or "unknown"
                )
                self.telemetry.update(
                    plant_support_mode=confirmed_mode,
                    plant_support_requested_mode="safety_stop",
                    plant_support_confirmed_mode=confirmed_mode,
                    plant_support_stop_pending=self._plant_support_stop_pending,
                    plant_support_safety_latched=self._plant_support_safety_latched,
                    last_error=(
                        "MuJoCo locomotion rejected while plant safety_stop is latched"
                    ),
                )
                return False
            if mode == self._plant_support_mode:
                return True
        try:
            state = set_plant_support(
                "127.0.0.1",
                self.config.plant_port,
                mode,
                timeout=0.20,
            )
        except Exception as exc:
            self.telemetry.update(
                plant_support_mode="unavailable",
                plant_support_confirmed_mode="unavailable",
                last_error=f"MuJoCo support transition failed: {exc}",
            )
            return False
        confirmed_mode = str(state.get("mode", ""))
        if confirmed_mode != mode:
            self.telemetry.update(
                plant_support_mode="unavailable",
                plant_support_confirmed_mode="unavailable",
                last_error=(
                    "MuJoCo support transition returned an unexpected confirmed mode"
                ),
            )
            return False
        with self._support_lock:
            stop_raced_transition = (
                self._plant_support_stop_pending
                or self._plant_support_safety_latched
            )
            stop_already_confirmed = self._plant_support_safety_latched
        if stop_raced_transition:
            # Never let a slower idle/locomotion response erase a stronger
            # stop accepted on the command thread while this request was in
            # flight.  A pending stop is synchronously strengthened here; an
            # already confirmed latch needs no second Plant write.
            if not stop_already_confirmed:
                self._ensure_plant_support_stopped(
                    reason="Plant safety-stop raced a support transition"
                )
            return mode != "locomotion"
        with self._support_lock:
            self._plant_support_mode = confirmed_mode
        self.telemetry.update(
            plant_support_mode=confirmed_mode,
            plant_support_confirmed_mode=confirmed_mode,
            plant_support_stop_pending=False,
            plant_support_safety_latched=False,
        )
        return True

    def _planner_reference_cursor(self) -> int | None:
        controller = getattr(self, "_controller", None)
        pipeline = getattr(controller, "controller", None)
        raw_cursor = getattr(pipeline, "ref_cursor", None)
        if isinstance(raw_cursor, bool):
            return None
        try:
            cursor = int(raw_cursor)
        except (TypeError, ValueError):
            return None
        return cursor if cursor >= 0 else None

    def _planner_telemetry_fields(self) -> dict[str, Any]:
        evidence = getattr(self, "_planner_evidence", None)
        if evidence is None:
            return {}
        return evidence.telemetry_fields(
            ref_cursor=self._planner_reference_cursor()
        )

    def _revoke_planner_reference(self, reason: str) -> None:
        evidence = getattr(self, "_planner_evidence", None)
        if evidence is None:
            return
        evidence.revoke(reason)
        self.telemetry.update(**self._planner_telemetry_fields())

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
        # Every newly required motion generation, and every transition back to
        # IDLE, starts at the exact 29-DoF actuator target that crossed DDS.
        # The IDLE branch matters when the local cmd_vel watchdog fires before
        # an explicit Robot Agent stop arrives: StaticJointHandoff must not
        # replace the gravity-balancing target with lagging measured angles.
        requires_actuator_boundary = admitted and (
            intent.mode != "IDLE"
            or (previous is not None and previous.mode != "IDLE")
        )
        if requires_actuator_boundary:
            if not self._freeze_planner_handoff_boundary():
                raise RuntimeError(
                    "intent transition has no published actuator boundary"
                )
        self._planner_evidence.observe_intent(
            intent,
            admitted=admitted,
            simulation_now_s=simulation_now_s,
        )
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
            **self._planner_telemetry_fields(),
        )
        return intent

    def _enforce_planner_reference_deadline(
        self,
        *,
        simulation_now_s: float,
    ) -> None:
        """Turn a required-reference timeout into a terminal actuator hold."""

        try:
            self._planner_evidence.enforce_required_deadline(
                simulation_now_s=simulation_now_s
            )
        except PlannerReferenceDeadlineExceeded as exc:
            self._latch_terminal_fault(str(exc))
            raise

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
        raw_targets = tuple(
            float(
                action.get(
                    f"{self._joint_name(index)}.q",
                    lowstate.motor_state[index].q,
                )
            )
            for index in range(29)
        )
        guarded_arm = None
        with self._support_lock:
            # Serialize the final DDS write with stop target freezing.  If a
            # stop wins the lock, this in-flight policy result is replaced by
            # the exact prior target; if this write wins, the stop freezes the
            # just-published target.  Either ordering yields a zero target
            # discontinuity at the actuator boundary.
            if self._safe_hold_targets is not None:
                published_targets = self._safe_hold_targets
            elif constrain_arms:
                # The arm guard is actuator state too.  Run it under the same
                # boundary lock as stop freezing, and seed it from each frozen
                # target so a zero-ratio planner handoff cannot re-anchor to
                # lagging measured joints.
                guarded_arm = self._arm_guard.apply(
                    raw_arm,
                    measured=measured_arm,
                    dt=control_dt_s,
                )
                guarded_by_index = dict(
                    zip(range(15, 29), guarded_arm.targets, strict=True)
                )
                published_targets = tuple(
                    float(
                        guarded_by_index[index]
                        if index in guarded_by_index
                        else raw_targets[index]
                    )
                    for index in range(29)
                )
            else:
                published_targets = raw_targets
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
            self._last_published_targets = published_targets
            if guarded_arm is None:
                self._arm_guard.anchor(published_targets[15:29])
        self._mark_control_step_published()
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
        planner_evidence = getattr(self, "_planner_evidence", None)
        planner_trace = (
            {}
            if planner_evidence is None
            else planner_evidence.trace_fields(
                ref_cursor=self._planner_reference_cursor()
            )
        )
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
                **planner_trace,
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
        self.telemetry.update(
            status="warming",
            control_clock_status="disarmed_static_warmup",
        )
        try:
            self._prepare_idle_startup()
        except Exception as exc:
            self._latch_terminal_fault(f"SONIC startup fault: {exc}")
            self._stop.set()
        else:
            self.telemetry.update(
                status="qualifying",
                control_clock_status="warm_ready_to_qualify",
                motion_ready=False,
            )
        while not self._stop.is_set():
            sample = self._next_control_sample()
            if self._fault_latched.is_set():
                with self._lowstate_lock:
                    lowstate = self._lowstate
                try:
                    if lowstate is not None:
                        self._publish_frozen_hold(lowstate)
                finally:
                    self._stop.set()
                break
            if sample is None:
                continue
            lowstate, control_step, sample_drop_generation = sample
            try:
                if self._startup_qualifying:
                    # Qualification is intentionally policy-free.  The plant is
                    # stationary, obsolete samples may still be latest-only
                    # synchronized, and only a measured hold is published.
                    self._publish_measured_hold(lowstate)
                    self._complete_startup_qualification_step(
                        sample_drop_generation=sample_drop_generation
                    )
                    continue
                intent = self._apply_intent(
                    control_step.dt_s,
                    simulation_now_s=control_step.tick_ms / 1_000.0,
                )
                # Feed empty external action: the controller reads the mapped
                # MovementState above and executes the official SONIC pipeline.
                # This call, its history, and its reference cursor advance only
                # after 20 ms of authoritative MuJoCo time.
                policy_action = self._controller.run_step({}, lowstate)
                self._enforce_planner_reference_deadline(
                    simulation_now_s=control_step.tick_ms / 1_000.0
                )
                self.telemetry.update(
                    sonic_policy_speed_mps=(
                        max(0.0, float(self._controller.ms.speed))
                        if intent.mode != "IDLE"
                        else 0.0
                    ),
                    **self._planner_telemetry_fields(),
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
                # The command thread may freeze the exact last-published
                # targets while this 50 Hz thread is preparing a stop.  Use
                # the support lock as the single serialization boundary so a
                # measured re-anchor cannot race that stronger hold.
                with self._support_lock:
                    decision = self._joint_handoff.decide(
                        motion_active=intent.mode != "IDLE",
                        reference_ready=self._planner_evidence.reference_ready,
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
                    with self._support_lock:
                        expected_safety_stop = (
                            self._plant_support_stop_pending
                            or self._plant_support_safety_latched
                        )
                    if expected_safety_stop:
                        # A lease release/cancel can race a policy tick that
                        # was already computing.  Publish only measured hold
                        # for that tick and keep the recoverable safety latch;
                        # this is not a controller/physics fault that should
                        # force a process restart.
                        self.guard.stop(source="plant_safety_stop")
                        self._navigation_task_active = False
                        self._arm_guard.reset()
                        self._static_hold_active = True
                        self._publish_frozen_hold(lowstate)
                        self.telemetry.update(
                            nav2_state="safety_stop",
                            navigation_task_active=False,
                            static_hold_active=True,
                            motion_handoff_active=False,
                            motion_handoff_ratio=0.0,
                        )
                        continue
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
                    control_clock_status="simulation_time_50hz_motion_ready",
                    static_hold_active=decision.static_hold_active,
                    motion_handoff_active=decision.motion_handoff_active,
                    motion_handoff_ratio=decision.handoff_ratio,
                    **self._planner_telemetry_fields(),
                )
            except Exception as exc:
                self._latch_terminal_fault(f"SONIC runtime fault: {exc}")
                try:
                    self._publish_frozen_hold(lowstate)
                finally:
                    self._stop.set()
                break
        try:
            self._controller.shutdown()
        finally:
            if not self._fault_latched.is_set():
                self.telemetry.update(status="stopped", motion_ready=False)

    def stop(self):
        self._stop.set()
        self.guard.stop()
        motion_ready = getattr(self, "_motion_ready", None)
        if motion_ready is not None:
            motion_ready.clear()
        condition = getattr(self, "_control_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()

    def accept_json(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if payload.get("type") == "sonic_health_probe":
            request_id, _producer_boot_id = validate_sonic_health_request(
                payload,
                expected_robot_id=self.config.robot_id,
            )
            with self._control_condition:
                checked_at = self._wall_clock()
                last_lowstate_at = self._last_lowstate_wall_s
                last_control_step_at = self._last_control_step_wall_s
                lowstate_age = (
                    None
                    if last_lowstate_at is None
                    else max(0.0, checked_at - last_lowstate_at)
                )
                control_step_age = (
                    None
                    if last_control_step_at is None
                    else max(0.0, checked_at - last_control_step_at)
                )
            # The command server serializes intents, stops, control.arm and
            # probes. This support lock therefore produces one authoritative
            # driver snapshot without reading the eventually-consistent HTTP
            # telemetry surface or guessing from display strings.
            with self._support_lock:
                plant_support_mode = self._plant_support_mode or "unavailable"
                plant_support_safety_latched = (
                    self._plant_support_safety_latched
                )
                plant_support_stop_pending = self._plant_support_stop_pending
                motion_authorized = self._intent_ingress.armed
                static_hold_active = self._static_hold_active
                navigation_task_active = self._navigation_task_active
                nav2_command_zero = self.guard.command_zero
            terminal_fault = self._fault_latched.is_set()
            motion_ready = (
                not terminal_fault
                and not self._stop.is_set()
                and self._motion_ready.is_set()
            )
            if terminal_fault:
                status = "fault"
            elif self._stop.is_set():
                status = "stopped"
            elif motion_ready:
                status = "ready"
            else:
                status = "warming"
            terminal_reason = (
                None
                if not terminal_fault
                else (
                    (
                        str(self._terminal_fault_reason or "").strip()
                        or "SONIC terminal fault"
                    )[:512]
                )
            )
            # This path is deliberately observational: it does not touch the
            # ingress gate, planner, controller, telemetry or actuator state.
            return {
                "schema_version": SONIC_HEALTH_SCHEMA_VERSION,
                "robot_id": self.config.robot_id,
                "driver_boot_id": str(self._driver_boot_id),
                "status": status,
                "terminal_fault": terminal_fault,
                "reason": terminal_reason,
                "motion_ready": motion_ready,
                "last_lowstate_age_s": lowstate_age,
                "last_control_step_age_s": control_step_age,
                "plant_support_mode": plant_support_mode,
                "plant_support_safety_latched": plant_support_safety_latched,
                "plant_support_stop_pending": plant_support_stop_pending,
                "static_hold_active": static_hold_active,
                "navigation_task_active": navigation_task_active,
                "motion_authorized": motion_authorized,
                "nav2_command_zero": nav2_command_zero,
            }
        if payload.get("type") == "plant_support_state":
            if payload.get("schema_version") != "h-ide.plant-support-state/v1":
                raise ValueError("unsupported plant support state schema")
            if payload.get("mode") != "safety_stop":
                raise ValueError("external support notifications may only strengthen safety_stop")
            if payload.get("source") not in {
                "robot_agent_safe_stop",
                "navigation_guard",
            }:
                raise ValueError("unsupported plant support state source")
            reason = str(
                payload.get("reason") or "MuJoCo plant safety_stop was engaged"
            )
            stopped_generation = self._latch_plant_support_stop(reason=reason)
            if self._fault_latched.is_set():
                self.telemetry.update(
                    status="failed",
                    control_clock_status="fault",
                    motion_ready=False,
                    last_error=self._terminal_fault_reason,
                )
            return {
                "ok": True,
                "mode": "safety_stop",
                "stopped_generation": stopped_generation,
            }
        if self._fault_latched.is_set():
            self._freeze_last_published_hold()
            self.guard.stop(source="terminal_fault")
            if payload.get("control_arm") or payload.get("request_id"):
                raise RuntimeError(
                    "SONIC terminal fault requires an explicit process restart"
                )
            return None
        if payload.get("type") == "stop":
            self._freeze_last_published_hold()
            self.guard.stop(source="bridge_stop")
            self._navigation_task_active = False
            reason = payload.get("reason")
            with self._support_lock:
                self._intent_ingress.revoke()
                if payload.get("source") == "robot_agent_safe_stop":
                    self._plant_support_stop_pending = True
                support_latched = self._plant_support_safety_latched
                support_pending = self._plant_support_stop_pending
                confirmed_mode = self._plant_support_mode or "unknown"
            if support_pending:
                self._arm_guard.reset()
            self.telemetry.update(
                nav2_state=(
                    "safety_stop"
                    if reason or support_pending or support_latched
                    else "idle"
                ),
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                plant_support_mode=confirmed_mode,
                plant_support_requested_mode=(
                    "safety_stop" if support_pending else confirmed_mode
                ),
                plant_support_confirmed_mode=confirmed_mode,
                plant_support_stop_pending=support_pending,
                plant_support_safety_latched=support_latched,
                last_error=str(reason) if reason else None,
            )
            return None
        control_arm = payload.get("control_arm", False)
        try:
            with self._support_lock:
                decision = self._intent_ingress.accept(payload)
        except (KeyError, TypeError, ValueError) as exc:
            self._freeze_last_published_hold()
            self.guard.stop(source="invalid_gait_intent")
            self._navigation_task_active = False
            self._ensure_plant_support_stopped(
                reason=f"invalid gait intent: {exc}"
            )
            self.telemetry.update(
                nav2_state="safety_stop",
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                last_error=f"invalid gait intent: {exc}",
            )
            raise
        if decision.action == "ignore":
            return None
        if decision.heading_yaw is not None:
            self.guard.observe_heading(decision.heading_yaw)
        if control_arm:
            if not decision.arm_acknowledged or decision.generation is None:
                raise ValueError(
                    "control_arm did not produce a fresh stationary IDLE acknowledgement"
                )
            if not self._rearm_plant_support(decision.generation):
                with self._support_lock:
                    self._intent_ingress.revoke()
                raise RuntimeError("explicit Plant support rearm failed")
            return {
                "ok": True,
                "mode": "idle",
                "generation": decision.generation,
            }
        if decision.action == "stop":
            # Stop/motion_enabled authority is obeyed before any values enter
            # CmdVelGuard or the SONIC model mapping.
            self._freeze_last_published_hold()
            self.guard.stop(source="gait_adapter")
            self._navigation_task_active = False
            if decision.state == "FAULT":
                self._ensure_plant_support_stopped(
                    reason=decision.reason
                    or f"gait adapter entered {decision.state}"
                )
            with self._support_lock:
                support_latched = (
                    self._plant_support_stop_pending
                    or self._plant_support_safety_latched
                )
            self.telemetry.update(
                nav2_state=(
                    "idle"
                    if decision.state == "IDLE" and not support_latched
                    else "safety_stop"
                ),
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                last_error=decision.reason,
            )
            return None
        with self._support_lock:
            support_latched = (
                self._plant_support_stop_pending
                or self._plant_support_safety_latched
            )
        if support_latched:
            self.guard.stop(source="plant_safety_stop")
            self._navigation_task_active = False
            self.telemetry.update(
                nav2_state="safety_stop",
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                plant_support_mode="safety_stop",
                plant_support_safety_latched=True,
                last_error="motion rejected while Plant safety_stop is latched",
            )
            return None
        if not self._motion_ready.is_set():
            # A command received while ONNX is warming cannot be replayed after
            # startup/qualification.  Hold it at the actuator boundary and
            # require a fresh gait-adapter intent after motion readiness.
            self.guard.stop(source="sonic_startup_hold")
            self._navigation_task_active = False
            self.telemetry.update(
                nav2_state="startup_hold",
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                last_error="SONIC control loop is still warming",
            )
            return None
        if not self._navigation_task_active:
            self.telemetry.reset_navigation_task_peaks()
            self._navigation_task_active = True
            self.telemetry.update(navigation_task_active=True)
        with self._support_lock:
            self._safe_hold_targets = None
        self.guard.accept(decision.vx, decision.vy, decision.wz, source="nav2")
        self.telemetry.update(
            nav2_state="commanding",
            last_cmd_vx=decision.vx,
            last_cmd_vy=decision.vy,
            last_cmd_wz=decision.wz,
            last_error=None,
        )
        return None


def _handle_command_connection(driver: SonicDriver, conn) -> None:
    """Serve one bounded-lifetime 8793 client without changing its semantics."""

    data = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    messages, _ = parse_lines(data + (b"" if data.endswith(b"\n") else b"\n"))
    for message in messages:
        request_id = message.get("request_id")
        try:
            response = driver.accept_json(message)
            if request_id is not None:
                acknowledgement = {
                    "request_id": request_id,
                    "ok": True,
                    **(response or {}),
                }
                conn.sendall(
                    (
                        json.dumps(acknowledgement, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                )
        except Exception as exc:
            if request_id is None:
                raise
            conn.sendall(
                (
                    json.dumps(
                        {
                            "request_id": request_id,
                            "ok": False,
                            "error": str(exc)[:512],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )


def run_driver(config: SimConfig) -> None:
    telemetry = TelemetryStore(TelemetryState(robot_id=config.robot_id, ros_domain_id=config.ros_domain_id))
    serve_telemetry(
        telemetry,
        config.telemetry_bind_host,
        config.telemetry_port,
        docker_observation_host=config.docker_observation_bind_host,
    )
    driver = SonicDriver(config, telemetry)

    # A tiny localhost TCP command server.  ROS bridge is the only intended
    # client; it sends one JSON object per line and never receives actuator data.
    server = _bind_command_server(config.command_bind_host, config.command_port)
    command_server_stop = threading.Event()

    def accept_loop():
        while not command_server_stop.is_set():
            try:
                conn, _ = server.accept()
            except TimeoutError:
                continue
            except OSError:
                if command_server_stop.is_set():
                    return
                raise
            with conn:
                try:
                    _handle_command_connection(driver, conn)
                except Exception as exc:
                    logger.warning("invalid bridge command: %s", exc)

    threading.Thread(target=accept_loop, daemon=True, name="sonic-command-server").start()
    try:
        driver.run()
    finally:
        if driver._fault_latched.is_set():
            # Keep the existing read-only 8793 probe surface alive briefly so
            # the 40 Hz bridge can report the terminal reason rather than only
            # a generic disconnect.  Motion is already latched off.
            command_server_stop.wait(
                timeout=min(
                    0.10,
                    2.0 * config.sonic_health_probe_period_s
                    + config.sonic_health_probe_timeout_s,
                )
            )
        driver.stop()
        command_server_stop.set()
        server.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_driver(SimConfig.from_file(args.config))
