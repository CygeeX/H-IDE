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
from typing import Any, Mapping
from uuid import UUID, uuid4

from .config import SimConfig
from .motion_fault import (
    SONIC_HEALTH_SCHEMA_VERSION,
    validate_sonic_health_request,
)
from .network import require_loopback_bind_host
from .normal_stop import NormalStopIdentity, NormalStopSnapshot, NormalStopState
from .normal_stop_wire import (
    NormalStopBeginRequest,
    NormalStopPlantRequest,
    NormalStopStationaryRequest,
    NormalStopWireError,
    canonical_normal_stop_json,
    normal_stop_response,
    parse_normal_stop_request,
    parse_normal_stop_response,
)
from .planner_evidence import (
    PlannerReferenceDeadlineExceeded,
    PlannerReferenceEvidenceGate,
)
from .reference_authority import ReferenceAuthorityState
from .runtime_diagnostics import SonicRuntimeDiagnostics
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
from .transport import (
    latch_plant_physics_fault,
    parse_lines,
    rearm_plant_support,
    set_plant_support,
)

logger = logging.getLogger(__name__)


_NORMAL_STOP_INTENT_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "sequence",
        "generation",
        "generated_at_s",
        "state",
        "mode",
        "velocity_x_mps",
        "velocity_y_mps",
        "linear_speed_mps",
        "movement_angle_rad",
        "yaw_rate_rps",
        "facing_delta_rad",
        "motion_enabled",
        "stop_requested",
        "stop_acknowledged",
        "head_yaw_rad",
        "head_pitch_rad",
        "source",
        "reason",
    }
)
_NORMAL_STOP_ZERO_FIELDS = (
    "velocity_x_mps",
    "velocity_y_mps",
    "linear_speed_mps",
    "movement_angle_rad",
    "yaw_rate_rps",
    "facing_delta_rad",
    "head_yaw_rad",
    "head_pitch_rad",
)


class _NormalStopBoundaryError(RuntimeError):
    """A local actuator/reference stop boundary could not be established."""


class _NormalStopLocalHoldError(RuntimeError):
    """Plant stopped, but the driver cannot prove its exact local hold."""


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
        self.staged_control_arm_generation: int | None = None

    def revoke(self) -> None:
        """Drop motion authority without manufacturing a new generation."""

        self.armed = False
        self.staged_control_arm_generation = None

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
        self.staged_control_arm_generation = None
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

    def commit_staged_support_rearm(self, generation: int) -> None:
        """Atomically publish authority only after Plant and health commit.

        A control-arm envelope is first validated with ``defer_control_arm`` so
        the Plant may be asked to release only to IDLE without making motion
        authority observable.  The driver calls this method under its support
        lock after the post-rearm freshness check has also succeeded.
        """

        if (
            self.staged_control_arm_generation != generation
            or self.support_stop_generation is None
            or generation <= self.support_stop_generation
            or generation != self.generation
            or self.armed
        ):
            raise ValueError(
                "staged support rearm does not match the validated generation"
            )
        self.armed = True
        self.support_stop_generation = None
        self.staged_control_arm_generation = None

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

    def accept(
        self,
        payload: dict[str, Any],
        *,
        defer_control_arm: bool = False,
    ) -> IngressDecision:
        if not isinstance(defer_control_arm, bool):
            raise TypeError("defer_control_arm must be a boolean")
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
            self.staged_control_arm_generation = None
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
            if defer_control_arm and control_arm and arm_candidate:
                self.armed = False
                self.staged_control_arm_generation = generation
            else:
                self.armed = arm_candidate
                self.staged_control_arm_generation = None
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

        self.staged_control_arm_generation = None

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

    def __init__(
        self,
        config: SimConfig,
        telemetry: TelemetryStore,
        runtime_diagnostics: SonicRuntimeDiagnostics | None = None,
    ):
        self.config = config
        self.telemetry = telemetry
        self._runtime_diagnostics = runtime_diagnostics
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
        self._terminal_fault_lock = threading.Lock()
        self._terminal_plant_halt_attempted = False
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
        self._normal_stop = NormalStopState()
        self._normal_stop_bridge_boot_id: UUID | None = None
        self._normal_stop_seen_bridge_boot_ids: set[UUID] = set()
        self._normal_stop_plant_safety_stop_epoch: int | None = None
        self._normal_stop_last_confirmed_plant_epoch = 0
        self._normal_stop_begin_intent: bytes | None = None
        self._normal_stop_seen_request_ids: set[UUID] = set()
        self._normal_stop_replays: dict[
            UUID,
            tuple[bytes, dict[str, object]],
        ] = {}
        self._reference_authority = ReferenceAuthorityState()
        self._reference_source_identity: tuple[int, int] | None = None
        self._reference_request_pending = False
        self._vendor_revoke_pending = False
        self._dds_target_sequence = 0

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
        self._controller = self._controller_type(
            force_cpu=config.force_cpu,
            auto_reference_authority=False,
            enable_startup_blend=False,
            allow_operator_intent=False,
        )
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
        runtime_diagnostics = getattr(self, "_runtime_diagnostics", None)
        if runtime_diagnostics is not None:
            runtime_diagnostics.observe_lowstate(received_at_wall_s)
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
        self.telemetry.update_hot(
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
        # The live controller remains manual-authority.  This isolated clone is
        # deliberately auto-authorized so one full encoder/decoder pass can be
        # warmed without manufacturing an outer reference transaction.
        warm_runtime.auto_reference_authority = True

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
        if not self._controller.initialize_from_lowstate(lowstate):
            raise RuntimeError("SONIC live runtime was initialized before startup preparation")
        self._warm_controller_inference(lowstate)
        runtime_diagnostics = getattr(self, "_runtime_diagnostics", None)
        if runtime_diagnostics is not None:
            runtime_diagnostics.freeze_warm_runtime()
            gc_state = runtime_diagnostics.snapshot()
            if gc_state["python_gc_freeze_applied"]:
                logger.info(
                    "SONIC warm runtime frozen after collecting %d objects; "
                    "automatic GC remains enabled",
                    gc_state["python_gc_warmup_collected_objects"],
                )
            else:
                logger.warning(
                    "SONIC warm-runtime GC optimization was not applied: %s",
                    gc_state["python_gc_freeze_error"],
                )
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
            terminal_boundary = self._enforce_terminal_boundary_locked()
            if not terminal_boundary:
                self._consume_vendor_revoke_locked()
            if terminal_boundary and self._last_published_targets is None:
                raise RuntimeError(
                    "terminal SONIC hold has no executed actuator boundary"
                )
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
            # Final pre-Write linearization.  A fault may be claimed while the
            # 29-axis command and CRC are being prepared after the earlier
            # support snapshot.  If it already won, replace every q with the
            # exact executed hold and recompute CRC.  A claim after this short
            # lock is treated by the post-Write transaction below; never hold
            # the terminal lock across Unitree's unbounded Write call.
            with self._terminal_fault_lock:
                if self._fault_latched.is_set():
                    self._enforce_terminal_boundary_locked()
                    selected = (
                        self._safe_hold_targets
                        or self._last_published_targets
                    )
                    if selected is None:
                        raise RuntimeError(
                            "terminal SONIC hold has no executed actuator boundary"
                        )
                    for index in range(29):
                        self._msg.motor_cmd[index].q = float(selected[index])
                    self._msg.crc = self._crc.Crc(self._msg)
            written = self._publisher.Write(self._msg)
            if written is not True:
                raise RuntimeError("Unitree DDS position-hold write failed")
            # A successful Write is physical fact.  Serialize its terminal
            # observation, vendor feedback, and local executed boundary with
            # the once-only fault claim: if fault wins, revoke first and record
            # the actual target without cursor authority; if feedback wins,
            # terminal teardown subsequently freezes this newly executed
            # target.  Never hold this lock across the potentially blocking
            # DDS Write or Plant halt.
            with self._terminal_fault_lock:
                terminal_error: Exception | None = None
                terminal_after_write = self._fault_latched.is_set()
                if terminal_after_write:
                    final_targets = self._record_successful_dds_target(selected)
                    self._safe_hold_targets = final_targets
                    try:
                        self._enforce_terminal_boundary_locked()
                    except Exception as exc:
                        terminal_error = exc
                    try:
                        self._record_vendor_execution_feedback_if_pending(
                            final_targets
                        )
                    except Exception as exc:
                        if terminal_error is None:
                            terminal_error = exc
                else:
                    self._commit_executed_target_if_pending(selected)
                if terminal_error is not None:
                    raise terminal_error
            assert self._last_published_targets is not None
            self._arm_guard.anchor(self._last_published_targets[15:29])
        self._mark_control_step_published()

    def _commit_executed_target_if_pending(
        self,
        targets: tuple[float, ...],
    ) -> bool:
        """Commit the exact target immediately after one successful DDS write.

        Startup/ordinary holds have no policy candidate and are an intentional
        feedback no-op.  A stop racing inference still commits the frozen target
        that actually crossed DDS, while revoked vendor authority prevents its
        reference cursor from advancing.
        """

        final_targets = self._record_successful_dds_target(targets)
        return self._record_vendor_execution_feedback_if_pending(final_targets)

    def _record_successful_dds_target(
        self,
        targets: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Register the target that physically crossed a successful DDS Write."""

        final_targets = tuple(float(value) for value in targets)
        if len(final_targets) != 29 or not all(
            math.isfinite(value) for value in final_targets
        ):
            raise ValueError("executed DDS target requires 29 finite values")
        self._last_published_targets = final_targets
        self._dds_target_sequence += 1
        return final_targets

    def _record_vendor_execution_feedback_if_pending(
        self,
        final_targets: tuple[float, ...],
    ) -> bool:
        """Write exact executed history after the local DDS fact is registered."""

        controller = getattr(self, "_controller", None)
        if controller is None or not bool(
            getattr(controller, "execution_feedback_pending", False)
        ):
            return False
        controller.record_executed_target(final_targets)
        return True

    def _mark_control_step_published(self) -> None:
        """Record only a successful actuator write on the monotonic clock."""

        published_at = self._wall_clock()
        with self._control_condition:
            self._last_control_step_wall_s = published_at
        runtime_diagnostics = getattr(self, "_runtime_diagnostics", None)
        if runtime_diagnostics is not None:
            runtime_diagnostics.observe_control_publish(published_at)

    def _publish_measured_hold(self, lowstate) -> None:
        self._publish_position_hold(
            tuple(float(lowstate.motor_state[index].q) for index in range(29))
        )

    def _freeze_last_published_hold(
        self,
        *,
        publish_telemetry: bool = True,
    ) -> bool:
        """Keep the actuator boundary continuous across a stop transition."""

        failures: list[Exception] = []
        with self._support_lock:
            targets = self._last_published_targets
            if targets is not None:
                self._safe_hold_targets = targets
            # Revoke the externally visible authority first.  Every remaining
            # local/vendor action is still attempted so one broken verifier or
            # controller cannot leave a GRANTED snapshot behind.
            try:
                self._revoke_planner_reference_locked(
                    "actuator boundary entered static hold"
                )
            except Exception as exc:
                failures.append(exc)
            if targets is not None:
                try:
                    self._joint_handoff.freeze(targets)
                except Exception as exc:
                    failures.append(exc)
                try:
                    self._arm_guard.anchor(targets[15:29])
                except Exception as exc:
                    failures.append(exc)
        if publish_telemetry:
            self.telemetry.update(**self._planner_telemetry_fields())
        if failures:
            raise failures[0]
        return targets is not None

    def _stop_navigation_at_boundary(
        self,
        *,
        source: str,
        publish_telemetry: bool = True,
    ) -> bool:
        """Atomically freeze/revoke and invalidate every sampled Nav2 intent."""

        failures: list[Exception] = []
        frozen = False
        with self._support_lock:
            try:
                frozen = self._freeze_last_published_hold(
                    publish_telemetry=publish_telemetry
                )
            except Exception as exc:
                failures.append(exc)
            try:
                self.guard.stop(source=source)
            except Exception as exc:
                failures.append(exc)
            self._navigation_task_active = False
            intent_gate = getattr(self, "_intent_gate", None)
            reset_intent_gate = getattr(intent_gate, "reset", None)
            if callable(reset_intent_gate):
                try:
                    reset_intent_gate()
                except Exception as exc:
                    failures.append(exc)
        if failures:
            raise failures[0]
        return frozen

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

    def _latch_terminal_fault(
        self,
        reason: str,
        *,
        boundary_already_stopped: bool = False,
    ) -> None:
        """Latch a non-recoverable process fault without rebuilding state."""

        terminal_lock = getattr(self, "_terminal_fault_lock", None)
        if terminal_lock is None:
            terminal_lock = threading.Lock()
            self._terminal_fault_lock = terminal_lock
        with terminal_lock:
            if self._fault_latched.is_set() or getattr(
                self, "_terminal_plant_halt_attempted", False
            ):
                return
            # This is the terminal linearization point shared with successful
            # DDS feedback/grant transactions.  Publish may not observe the
            # once-only claim without also observing the canonical reason and
            # fault event.
            self._terminal_plant_halt_attempted = True
            self._terminal_fault_reason = (
                str(reason).strip() or "SONIC terminal fault"
            )
            # Latch before best-effort controller teardown.  A corrupt vendor
            # runtime must never strand this path between the once-only claim
            # and the Plant halt, nor make a later caller mistake it for
            # recoverable.
            self._fault_latched.set()
        terminal_boundary_exception: Exception | None = None
        terminal_boundary_complete = bool(boundary_already_stopped)
        self._startup_qualifying = False
        motion_ready = getattr(self, "_motion_ready", None)
        if motion_ready is not None:
            motion_ready.clear()
        if not terminal_boundary_complete and self._support_lock.acquire(
            blocking=False
        ):
            try:
                self._stop_navigation_at_boundary(
                    source="terminal_fault",
                    publish_telemetry=False,
                )
            except Exception as exc:
                terminal_boundary_exception = exc
            else:
                terminal_boundary_complete = True
            finally:
                self._support_lock.release()
        # The Plant can cross a physical safety boundary in less time than
        # an inference or fault-trace persistence. Halt it immediately after
        # the lock-free terminal event; local boundary teardown follows and
        # cannot delay the physics stop if another thread is inside ONNX.
        plant_terminal_halt_confirmed = False
        plant_terminal_halt_error: str | None = None
        try:
            latch_plant_physics_fault(
                "127.0.0.1",
                self.config.plant_port,
                self._terminal_fault_reason[:512],
                timeout=0.30,
            )
        except Exception as exc:
            plant_terminal_halt_error = str(exc)
        else:
            plant_terminal_halt_confirmed = True
        if not terminal_boundary_complete:
            try:
                self._stop_navigation_at_boundary(
                    source="terminal_fault",
                    publish_telemetry=False,
                )
            except Exception as exc:
                if terminal_boundary_exception is None:
                    terminal_boundary_exception = exc
        if terminal_boundary_exception is not None:
            logger.error(
                "failed to freeze/revoke SONIC while latching terminal fault: %s",
                terminal_boundary_exception,
            )
        if plant_terminal_halt_error is not None:
            logger.error(
                "failed to confirm terminal MuJoCo Plant halt: %s",
                plant_terminal_halt_error,
            )
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
            terminal_boundary_error=(
                None
                if terminal_boundary_exception is None
                else str(terminal_boundary_exception)
            ),
            plant_terminal_halt_confirmed=plant_terminal_halt_confirmed,
            plant_terminal_halt_error=plant_terminal_halt_error,
        )
        if trace is not None:
            changes.update(
                fault_trace_path=trace_path,
                fault_trace_sample_count=trace_samples,
                fault_joint_names=list(trace.joint_names),
            )
        try:
            changes.update(self._planner_telemetry_fields())
        except Exception:
            # The controller itself may be the corrupt component. Plant halt
            # has already been attempted, so stale display fields are safer
            # than allowing diagnostics to unwind the terminal latch.
            logger.exception("failed to snapshot terminal reference telemetry")
        try:
            self.telemetry.update(**changes)
        except Exception:
            logger.exception("failed to publish terminal SONIC telemetry")
        condition = getattr(self, "_control_condition", None)
        if condition is not None:
            try:
                with condition:
                    condition.notify_all()
            except Exception:
                logger.exception("failed to notify SONIC control loop of terminal fault")

    def _latch_plant_support_stop(self, *, reason: str) -> int:
        """Synchronize an externally confirmed Plant stop into SONIC.

        This method never releases support.  It invalidates the current ingress
        authority and records the stopped generation so periodic IDLE traffic,
        an old ``/cmd_vel``, or a bridge reconnect cannot clear the latch.
        """

        boundary_error: Exception | None = None
        with self._support_lock:
            try:
                self._stop_navigation_at_boundary(
                    source="plant_safety_stop",
                    publish_telemetry=False,
                )
            except Exception as exc:
                boundary_error = exc
            stopped_generation = self._intent_ingress.latch_support_stop()
            self._plant_support_mode = "safety_stop"
            self._plant_support_stop_pending = False
            self._plant_support_safety_latched = True
            self._plant_support_stop_generation = stopped_generation
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
            **self._planner_telemetry_fields(),
        )
        if boundary_error is not None:
            self._latch_terminal_fault(
                f"Plant support stop boundary fault: {boundary_error}",
                boundary_already_stopped=True,
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

        boundary_error: Exception | None = None
        with self._support_lock:
            try:
                self._stop_navigation_at_boundary(
                    source="plant_safety_stop",
                    publish_telemetry=False,
                )
            except Exception as exc:
                boundary_error = exc
            stopped_generation = self._intent_ingress.latch_support_stop()
            # Publish the stronger software interlock before loopback I/O. A
            # racing control tick or intent can observe only static hold while
            # the Plant acknowledgement is pending.
            self._plant_support_stop_pending = True
            self._plant_support_safety_latched = True
            self._plant_support_stop_generation = stopped_generation
        if boundary_error is not None:
            self._latch_terminal_fault(
                f"Plant support stop boundary fault: {boundary_error}",
                boundary_already_stopped=True,
            )
            return False
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
                self._plant_support_mode = None
                self._plant_support_stop_pending = True
                self._plant_support_safety_latched = True
                self._plant_support_stop_generation = stopped_generation
                self.guard.stop(source="plant_safety_stop_unconfirmed")
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
                **self._planner_telemetry_fields(),
            )
            return False
        self._latch_plant_support_stop(reason=reason)
        return True

    def _require_fresh_control_arm_health_locked(self, *, phase: str) -> None:
        """Fail closed on one lock-consistent control-arm health snapshot.

        The caller owns both ``_control_condition`` and ``_support_lock``.  The
        first check is immediately before ingress validation can change its
        generation/sequence state.  The second check happens after Plant has
        replied ``idle`` but before SONIC publishes either that mode or motion
        authority, so a failed check can only return to the stronger latch.
        """

        label = str(phase).strip() or "control arm"
        if self._fault_latched.is_set():
            raise RuntimeError(f"{label}: SONIC terminal fault is latched")
        if self._stop.is_set():
            raise RuntimeError(f"{label}: SONIC is stopping")
        if not self._motion_ready.is_set():
            raise RuntimeError(f"{label}: SONIC is not motion-ready")

        checked_at = self._wall_clock()
        ages = (
            (
                "LowState",
                self._last_lowstate_wall_s,
                float(self.config.plant_lowstate_stale_s),
            ),
            (
                "control step",
                self._last_control_step_wall_s,
                float(self.config.sonic_control_step_stale_s),
            ),
        )
        for source, observed_at, threshold_s in ages:
            if observed_at is None:
                raise RuntimeError(f"{label}: {source} freshness is unavailable")
            age_s = max(0.0, checked_at - observed_at)
            if age_s > threshold_s:
                raise RuntimeError(
                    f"{label}: {source} age {age_s:.6f}s exceeds "
                    f"the {threshold_s:.6f}s safety bound"
                )

        safe_hold = (
            self._plant_support_mode == "safety_stop"
            and self._plant_support_safety_latched
            and not self._plant_support_stop_pending
            and self._static_hold_active
            and not self._navigation_task_active
            and not self._intent_ingress.armed
            and self.guard.command_zero
        )
        if not safe_hold:
            raise RuntimeError(
                f"{label}: control acquisition requires a confirmed settled "
                "safety hold"
            )

    def _request_staged_plant_support_rearm(self, generation: int) -> bool:
        """Ask Plant for IDLE while SONIC authority remains uncommitted."""

        with self._support_lock:
            stopped_generation = self._plant_support_stop_generation
            if (
                not self._plant_support_safety_latched
                or self._plant_support_stop_pending
                or stopped_generation is None
                or generation <= stopped_generation
                or generation != self._intent_ingress.generation
                or self._intent_ingress.armed
                or self._intent_ingress.staged_control_arm_generation
                != generation
            ):
                self.telemetry.update(
                    plant_support_mode="safety_stop",
                    plant_support_requested_mode="idle",
                    plant_support_confirmed_mode="safety_stop",
                    plant_support_safety_latched=True,
                    last_error=(
                        "plant safety_stop rearm rejected without a strictly "
                        "newer staged generation"
                    ),
                )
                return False
        try:
            rearm_plant_support(
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

        # Do not publish Plant's IDLE acknowledgement yet.  The command server
        # is serialized, the ingress remains unauthorized, and the local latch
        # keeps the 50 Hz thread in static hold until the post-rearm health gate
        # commits both facts together.
        with self._support_lock:
            transition_unchanged = (
                self._plant_support_safety_latched
                and not self._plant_support_stop_pending
                and self._plant_support_stop_generation == stopped_generation
                and self._intent_ingress.generation == generation
                and not self._intent_ingress.armed
                and self._intent_ingress.staged_control_arm_generation
                == generation
            )
        if transition_unchanged:
            return True
        self._ensure_plant_support_stopped(
            reason="Plant rearm raced a stronger stop or ingress transition"
        )
        return False

    def _commit_staged_plant_support_rearm_locked(self, generation: int) -> None:
        """Commit the already checked IDLE release and authority together."""

        if (
            not self._plant_support_safety_latched
            or self._plant_support_stop_pending
            or self._plant_support_mode != "safety_stop"
            or self._plant_support_stop_generation is None
        ):
            raise RuntimeError("Plant safety hold changed before rearm commit")
        self._intent_ingress.commit_staged_support_rearm(generation)
        self._plant_support_mode = "idle"
        self._plant_support_stop_pending = False
        self._plant_support_safety_latched = False
        self._plant_support_stop_generation = None

    def _publish_committed_plant_rearm(self, generation: int) -> None:
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

    def _support_transition_token_locked(self) -> tuple[int, int | None]:
        """Correlate support I/O with the stop epoch and ingress generation."""

        normal_stop = getattr(self, "_normal_stop", None)
        stop_epoch = (
            normal_stop.snapshot.stop_epoch
            if isinstance(normal_stop, NormalStopState)
            else 0
        )
        return stop_epoch, self._intent_ingress.generation

    def _set_plant_support_mode(self, mode: str) -> bool:
        """Switch only MuJoCo's confirmed fall-arrest profile over loopback."""

        self.telemetry.update(plant_support_requested_mode=mode)
        with self._support_lock:
            if (
                mode == "locomotion"
                and self._normal_stop_static_boundary_locked()
            ):
                # A moving tick prepared before braking/rearm may arrive here
                # after the stronger exact hold was established.  In
                # particular, DRIVER_CONFIRMED is not a permanent phase
                # fence: the safe target itself blocks Plant locomotion until
                # a matching reference has been primed and releases it.
                return False
            transition_token = self._support_transition_token_locked()
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
            normal_stop = getattr(self, "_normal_stop", None)
            normal_stop_phase = (
                normal_stop.snapshot.phase
                if isinstance(normal_stop, NormalStopState)
                else "NONE"
            )
            if normal_stop_phase in {
                "BRAKING",
                "STATIONARY_ACK",
                "PLANT_CONFIRMED",
                "FAILED",
            }:
                # Braking needs ordinary support until measured stationarity.
                # Never let a new control-loop request reach Plant while the
                # correlated stop owns this boundary.
                return mode == "idle"
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
            with self._support_lock:
                if transition_token != self._support_transition_token_locked():
                    return False
                self.telemetry.update(
                    plant_support_mode="unavailable",
                    plant_support_confirmed_mode="unavailable",
                    last_error=f"MuJoCo support transition failed: {exc}",
                )
            return False
        with self._support_lock:
            if transition_token != self._support_transition_token_locked():
                # A normal-stop begin or generation change won while this HTTP
                # call was in flight.  Drop the obsolete response locally.  In
                # particular, do not use the legacy pending-stop strengthening
                # path, which would touch Plant before stationary evidence.
                return False
        confirmed_mode = str(state.get("mode", ""))
        if confirmed_mode != mode:
            with self._support_lock:
                if transition_token != self._support_transition_token_locked():
                    return False
                self.telemetry.update(
                    plant_support_mode="unavailable",
                    plant_support_confirmed_mode="unavailable",
                    last_error=(
                        "MuJoCo support transition returned an unexpected "
                        "confirmed mode"
                    ),
                )
            return False
        with self._support_lock:
            if transition_token != self._support_transition_token_locked():
                return False
            stop_raced_transition = (
                self._plant_support_stop_pending
                or self._plant_support_safety_latched
            )
            stop_already_confirmed = self._plant_support_safety_latched
            if not stop_raced_transition:
                # Token validation, safety flags, local mode, and health-facing
                # telemetry are one support-lock commit.  A normal stop cannot
                # enter between the last validation and this write.
                self._plant_support_mode = confirmed_mode
                self.telemetry.update(
                    plant_support_mode=confirmed_mode,
                    plant_support_confirmed_mode=confirmed_mode,
                    plant_support_stop_pending=False,
                    plant_support_safety_latched=False,
                )
                return True
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
        raise AssertionError("unreachable Plant support transition state")

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
        fields = (
            {}
            if evidence is None
            else evidence.telemetry_fields(
                ref_cursor=self._planner_reference_cursor()
            )
        )
        authority_state = getattr(self, "_reference_authority", None)
        if authority_state is not None:
            authority = authority_state.snapshot
            fields.update(
                sonic_reference_authority_state=authority.phase,
                sonic_reference_authority_generation=authority.generation,
                sonic_reference_authority_source_sequence=(
                    authority.source_sequence
                ),
                sonic_reference_authority_intent_generation=(
                    authority.planner_intent_generation
                ),
                sonic_reference_authority_epoch=authority.reference_epoch,
                sonic_reference_actuator_authority=(
                    authority.actuator_authority
                ),
                sonic_reference_authority_dds_target_sequence=(
                    authority.dds_target_sequence
                ),
            )
        controller = getattr(self, "_controller", None)
        fields["sonic_execution_feedback_pending"] = bool(
            getattr(controller, "execution_feedback_pending", False)
        )
        return fields

    def _revoke_planner_reference(self, reason: str) -> None:
        with self._support_lock:
            self._revoke_planner_reference_locked(reason)
        self.telemetry.update(**self._planner_telemetry_fields())

    def _revoke_planner_reference_locked(self, reason: str) -> None:
        failures: list[Exception] = []
        authority = getattr(self, "_reference_authority", None)
        if authority is not None:
            try:
                authority.revoke(reason)
            except Exception as exc:
                failures.append(exc)
        evidence = getattr(self, "_planner_evidence", None)
        if evidence is not None:
            try:
                evidence.revoke(reason)
            except Exception as exc:
                failures.append(exc)
        controller = getattr(self, "_controller", None)
        revoke = getattr(controller, "revoke_reference_authority", None)
        if callable(revoke):
            # A command/terminal thread may win while ONNX is decoding. Never
            # enter vendor state from that thread: the 50 Hz owner consumes
            # this request after inference and before any feedback or publish.
            self._vendor_revoke_pending = True
        self._reference_request_pending = False
        if failures:
            raise failures[0]

    def _consume_vendor_revoke_locked(self) -> bool:
        """Apply one deferred vendor revoke at a policy-owner safe point."""

        if not self._vendor_revoke_pending:
            return False
        controller = getattr(self, "_controller", None)
        pipeline = getattr(controller, "controller", None)
        set_pipeline_authority = getattr(
            pipeline,
            "set_reference_authority",
            None,
        )
        if hasattr(pipeline, "playing"):
            # Cursor advancement checks ``playing`` first.  Drop it before any
            # callable cleanup so even a broken authority setter cannot move
            # the reference after a terminal DDS fact.
            pipeline.playing = False
        if callable(set_pipeline_authority):
            # Establish the cursor interlock before the higher-level wrapper,
            # whose additional cleanup may fail.  Successful DDS feedback must
            # still write the actual executed history after such a failure,
            # and PlannerController.advance_cursor() is then a guaranteed
            # no-op.  This is the same public pipeline authority primitive the
            # vendor wrapper itself uses.
            set_pipeline_authority(
                actuator_authority=False,
                matching_reference_ready=False,
            )
        revoke = getattr(controller, "revoke_reference_authority", None)
        if callable(revoke):
            revoke(
                discard_reference=True,
                discard_pending_execution=False,
            )
        self._vendor_revoke_pending = False
        return True

    def _enforce_terminal_boundary_locked(self) -> bool:
        """Revoke/consume a latched terminal stop at an actuator safe point."""

        if not self._fault_latched.is_set():
            return False
        failures: list[Exception] = []
        try:
            self._stop_navigation_at_boundary(
                source="terminal_fault",
                publish_telemetry=False,
            )
        except Exception as exc:
            failures.append(exc)
        try:
            self._consume_vendor_revoke_locked()
        except Exception as exc:
            failures.append(exc)
        if failures:
            raise failures[0]
        return True

    def _apply_intent(self, dt: float, *, simulation_now_s: float):
        # Command admission, stop, and reference ownership share one lock.
        # A stop that wins this boundary resets the intent gate before a
        # control tick can resurrect its already-sampled motion as REQUIRED.
        with self._support_lock:
            return self._apply_intent_locked(
                dt,
                simulation_now_s=simulation_now_s,
            )

    def _apply_intent_locked(self, dt: float, *, simulation_now_s: float):
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
        if admitted and intent.mode != "IDLE":
            with self._support_lock:
                identity = self._reference_source_identity
                if identity is None:
                    raise RuntimeError(
                        "admitted SONIC motion has no first source identity"
                    )
                self._revoke_planner_reference_locked(
                    "new Nav2 intent requires a matching planner reference"
                )
                self._reference_authority.require(
                    generation=identity[0],
                    source_sequence=identity[1],
                )
                self._reference_request_pending = True
        elif admitted and intent.mode == "IDLE":
            self._revoke_planner_reference("Nav2 intent is IDLE")
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
            # Manual authority requests the reference explicitly on the
            # planning-only path; the policy tick must not issue a duplicate.
            controller.ms.needs_replan = False
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

    def _progress_reference_transaction(self, lowstate) -> bool:
        """Plan, install, and prime one required reference without policy work."""

        with self._support_lock:
            self._consume_vendor_revoke_locked()
            authority = self._reference_authority.snapshot
            if authority.phase != "REQUIRED":
                return self._reference_authority.executable
            controller = self._controller
            if not controller.initialized:
                raise RuntimeError("manual SONIC reference requested before initialization")
            if self._reference_request_pending:
                if controller.execution_feedback_pending:
                    return False
                controller.request_reference(lowstate)
                self._reference_request_pending = False
            if controller.execution_feedback_pending:
                raise RuntimeError(
                    "cannot poll a SONIC reference with execution feedback pending"
                )
            if not controller.poll_reference():
                return False

            proof = self._planner_evidence.authority_evidence
            if (
                not proof.reference_ready
                or proof.accepted_intent_generation is None
                or proof.accepted_intent_generation != proof.intent_generation
                or proof.accepted_intent_signature != proof.intent_signature
                or not controller.matching_reference_installed
            ):
                raise RuntimeError(
                    "installed SONIC reference lacks matching planner authority evidence"
                )
            epoch = int(controller.reference_install_epoch)
            if authority.generation is None or authority.source_sequence is None:
                raise RuntimeError("required SONIC reference identity is incomplete")
            self._reference_authority.mark_installed(
                generation=authority.generation,
                source_sequence=authority.source_sequence,
                planner_intent_generation=proof.accepted_intent_generation,
                reference_epoch=epoch,
            )
            executed_target = self._last_published_targets
            if executed_target is None:
                raise RuntimeError(
                    "matching SONIC reference has no executed actuator boundary"
                )
            controller.acquire_reference(
                lowstate,
                executed_target=executed_target,
                expected_install_epoch=epoch,
            )
            self._reference_authority.prime(
                generation=authority.generation,
                source_sequence=authority.source_sequence,
                reference_epoch=epoch,
            )
            # An earlier stop hold may be released only after the measured
            # history and exact executed boundary have been atomically rebased.
            self._safe_hold_targets = None
            return True

    def _prime_periodic_reference_refresh(self) -> None:
        """Bind an in-flight candidate to a newly installed same-intent epoch."""

        with self._support_lock:
            authority = self._reference_authority.snapshot
            controller = self._controller
            epoch = int(controller.reference_install_epoch)
            if epoch == authority.reference_epoch:
                return
            if epoch < authority.reference_epoch:
                raise RuntimeError("SONIC reference install epoch moved backwards")
            if authority.phase == "REVOKED":
                # A stop may win after the vendor installed/refreshed and
                # decoded this candidate. The frozen DDS write still commits
                # that exact pending epoch/cursor internally, but never grants
                # it outer actuator authority.
                return
            if authority.phase != "GRANTED":
                raise RuntimeError(
                    "SONIC candidate changed reference epoch without granted authority"
                )
            if not controller.execution_feedback_pending:
                raise RuntimeError(
                    "SONIC reference epoch changed without a pending policy candidate"
                )
            proof = self._planner_evidence.authority_evidence
            if (
                not proof.reference_ready
                or proof.accepted_intent_generation
                != authority.planner_intent_generation
                or proof.accepted_intent_signature != proof.intent_signature
                or not controller.matching_reference_installed
            ):
                raise RuntimeError(
                    "periodic SONIC refresh lacks same-intent authority evidence"
                )
            assert authority.generation is not None
            assert authority.source_sequence is not None
            assert authority.planner_intent_generation is not None
            self._reference_authority.refresh(
                generation=authority.generation,
                source_sequence=authority.source_sequence,
                planner_intent_generation=authority.planner_intent_generation,
                reference_epoch=epoch,
            )

    def _run_policy_candidate(self, lowstate) -> dict[str, Any]:
        """Decode and project any installed refresh as one authority snapshot."""

        with self._support_lock:
            if self._enforce_terminal_boundary_locked():
                return {}
            self._consume_vendor_revoke_locked()
            if not self._reference_authority.executable:
                return {}
        action = self._controller.run_step({}, lowstate)
        with self._support_lock:
            if self._enforce_terminal_boundary_locked():
                return {}
            self._consume_vendor_revoke_locked()
            self._prime_periodic_reference_refresh()
        return action

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
        control_tick_ms: int | None = None,
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
            terminal_boundary = self._enforce_terminal_boundary_locked()
            if not terminal_boundary:
                self._consume_vendor_revoke_locked()
            if terminal_boundary and self._last_published_targets is None:
                raise RuntimeError(
                    "terminal SONIC publish has no executed actuator boundary"
                )
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
            with self._terminal_fault_lock:
                if self._fault_latched.is_set():
                    self._enforce_terminal_boundary_locked()
                    published_targets = (
                        self._safe_hold_targets
                        or self._last_published_targets
                    )
                    if published_targets is None:
                        raise RuntimeError(
                            "terminal SONIC publish has no executed actuator boundary"
                        )
                    for index in range(29):
                        self._msg.motor_cmd[index].q = float(
                            published_targets[index]
                        )
                    self._msg.crc = self._crc.Crc(self._msg)
            written = self._publisher.Write(self._msg)
            if written is not True:
                raise RuntimeError("Unitree DDS policy-target write failed")
            # The post-Write terminal check, exact executed feedback, and
            # possible first-DDS grant form one transaction against the
            # terminal claim.  DDS itself stays outside this lock because its
            # implementation has no timeout and must never delay Plant halt.
            with self._terminal_fault_lock:
                terminal_error: Exception | None = None
                terminal_after_write = self._fault_latched.is_set()
                if terminal_after_write:
                    final_targets = self._record_successful_dds_target(
                        published_targets
                    )
                    self._safe_hold_targets = final_targets
                    try:
                        self._enforce_terminal_boundary_locked()
                    except Exception as exc:
                        terminal_error = exc
                    try:
                        feedback_committed = (
                            self._record_vendor_execution_feedback_if_pending(
                                final_targets
                            )
                        )
                    except Exception as exc:
                        feedback_committed = False
                        if terminal_error is None:
                            terminal_error = exc
                else:
                    feedback_committed = self._commit_executed_target_if_pending(
                        published_targets
                    )
                authority = self._reference_authority.snapshot
                if not terminal_after_write and authority.phase == "PRIMED":
                    if not feedback_committed:
                        raise RuntimeError(
                            "primed SONIC reference wrote no policy execution feedback"
                        )
                    if (
                        control_tick_ms is None
                        or isinstance(control_tick_ms, bool)
                        or int(control_tick_ms) < 0
                        or authority.generation is None
                        or authority.source_sequence is None
                    ):
                        raise RuntimeError(
                            "first SONIC DDS grant lacks authoritative tick/identity"
                        )
                    self._reference_authority.grant(
                        generation=authority.generation,
                        source_sequence=authority.source_sequence,
                        reference_epoch=authority.reference_epoch,
                        granted_at_monotonic_s=self._wall_clock(),
                        granted_at_tick_ms=int(control_tick_ms),
                        dds_target_sequence=self._dds_target_sequence,
                    )
                if terminal_error is not None:
                    raise terminal_error
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
            **self._planner_telemetry_fields(),
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

    def _normal_stop_fences_motion_locked(self) -> bool:
        normal_stop = getattr(self, "_normal_stop", None)
        phase = (
            normal_stop.snapshot.phase
            if isinstance(normal_stop, NormalStopState)
            else "NONE"
        )
        return phase in {
            "BRAKING",
            "STATIONARY_ACK",
            "PLANT_CONFIRMED",
            "FAILED",
        }

    def _normal_stop_static_boundary_locked(self) -> bool:
        """Return the current recoverable stop's physical/local hold fence."""

        return (
            self._plant_support_stop_pending
            or self._plant_support_safety_latched
            or self._safe_hold_targets is not None
            or self._normal_stop_fences_motion_locked()
        )

    def _rejected_locomotion_is_static_boundary_locked(self) -> bool:
        """Classify a failed old support request without escalating a stop."""

        return self._normal_stop_static_boundary_locked()

    def _commit_rejected_locomotion_tick_locked(
        self,
        transition_token: tuple[int, int | None],
    ) -> bool:
        """Drop an old tick if rearm changed its established stop boundary."""

        if (
            transition_token != self._support_transition_token_locked()
            or not self._rejected_locomotion_is_static_boundary_locked()
        ):
            return False
        self._commit_static_hold_decision_locked(True)
        return True

    def _publish_rejected_locomotion_tick_if_current(
        self,
        lowstate: object,
        transition_token: tuple[int, int | None],
    ) -> bool:
        """Publish one frozen tick without inverting control/support locks."""

        with self._support_lock:
            if not self._commit_rejected_locomotion_tick_locked(
                transition_token
            ):
                return False
            frozen_targets = (
                self._safe_hold_targets or self._last_published_targets
            )
        # The publish path eventually marks the control step under
        # _control_condition.  Release support first because control.arm owns
        # that condition before acquiring support.
        if frozen_targets is None:
            self._publish_frozen_hold(lowstate)
        else:
            self._publish_position_hold(frozen_targets)
        return True

    def _commit_static_hold_decision_locked(
        self,
        decision_static_hold_active: bool,
    ) -> bool:
        """Fence a prepared 50 Hz decision against a correlated stop.

        A policy tick may calculate a moving handoff before the command thread
        freezes the actuator boundary, then resume after the complete normal
        stop.  Its final health-facing write must share the support lock and
        may never overwrite a stronger active, failed, Plant-latched, or exact
        actuator hold.  A completed transaction stops fencing only after an
        explicit rearm and matching reference have released those physical
        facts.

        The caller owns ``_support_lock``.
        """

        if not isinstance(decision_static_hold_active, bool):
            raise TypeError("static-hold decision must be boolean")
        effective_static_hold = (
            decision_static_hold_active
            or self._normal_stop_static_boundary_locked()
        )
        if decision_static_hold_active and not self._static_hold_active:
            self._arm_guard.reset()
        self._static_hold_active = effective_static_hold
        return effective_static_hold

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
                measured = tuple(
                    float(lowstate.motor_state[index].q) for index in range(29)
                )
                if intent.mode == "IDLE":
                    with self._support_lock:
                        decision = self._joint_handoff.decide(
                            motion_active=False,
                            reference_ready=False,
                            measured=measured,
                            policy=measured,
                            now=control_step.tick_ms / 1_000.0,
                        )
                    self._set_plant_support_mode("idle")
                    self._static_hold_active = True
                    self._publish_position_hold(decision.targets)
                    assert self._last_published_targets is not None
                    self._record_control_trace(
                        lowstate=lowstate,
                        control_step=control_step,
                        intent=intent,
                        policy_targets=decision.targets,
                        handoff_targets=decision.targets,
                        published_targets=self._last_published_targets,
                        handoff_ratio=0.0,
                    )
                    self.telemetry.update(
                        sonic_policy_speed_mps=0.0,
                        control_tick_ms=control_step.tick_ms,
                        control_dt_s=control_step.dt_s,
                        control_clock_status="simulation_time_50hz_motion_ready",
                        static_hold_active=True,
                        motion_handoff_active=False,
                        motion_handoff_ratio=0.0,
                        **self._planner_telemetry_fields(),
                    )
                    continue

                if not self._progress_reference_transaction(lowstate):
                    self._enforce_planner_reference_deadline(
                        simulation_now_s=control_step.tick_ms / 1_000.0
                    )
                    with self._support_lock:
                        waiting = self._joint_handoff.decide(
                            motion_active=True,
                            reference_ready=False,
                            measured=measured,
                            policy=measured,
                            now=control_step.tick_ms / 1_000.0,
                        )
                    self._set_plant_support_mode("idle")
                    self._static_hold_active = True
                    self._publish_position_hold(waiting.targets)
                    assert self._last_published_targets is not None
                    self._record_control_trace(
                        lowstate=lowstate,
                        control_step=control_step,
                        intent=intent,
                        policy_targets=waiting.targets,
                        handoff_targets=waiting.targets,
                        published_targets=self._last_published_targets,
                        handoff_ratio=0.0,
                    )
                    self.telemetry.update(
                        sonic_policy_speed_mps=0.0,
                        control_tick_ms=control_step.tick_ms,
                        control_dt_s=control_step.dt_s,
                        control_clock_status="waiting_for_matched_reference",
                        static_hold_active=True,
                        motion_handoff_active=False,
                        motion_handoff_ratio=0.0,
                        **self._planner_telemetry_fields(),
                    )
                    continue

                # Feed empty external action: the controller reads the mapped
                # MovementState above and executes the official SONIC pipeline.
                # Decode creates a pending candidate. Its action history and
                # reference cursor commit only after the complete DDS write.
                policy_action = self._run_policy_candidate(lowstate)
                self.telemetry.update(
                    sonic_policy_speed_mps=(
                        max(0.0, float(self._controller.ms.speed))
                    ),
                    **self._planner_telemetry_fields(),
                )
                if self._pending_decoder_heading_delta_rad:
                    # ``run_step`` has now consumed ``reinit_heading`` and
                    # reset the decoder delta.  Apply the bounded Nav2 Spin
                    # look-ahead after that reset, before the next policy tick.
                    self._controller.controller.delta_heading += self._pending_decoder_heading_delta_rad
                    self._pending_decoder_heading_delta_rad = 0.0
                policy_targets = self._action_targets(policy_action, lowstate)
                # The command thread may freeze the exact last-published
                # targets while this 50 Hz thread is preparing a stop.  Use
                # the support lock as the single serialization boundary so a
                # measured re-anchor cannot race that stronger hold.
                with self._support_lock:
                    decision = self._joint_handoff.decide(
                        motion_active=True,
                        reference_ready=self._reference_authority.executable,
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
                            self._rejected_locomotion_is_static_boundary_locked()
                        )
                        rejected_transition_token = (
                            self._support_transition_token_locked()
                        )
                    if expected_safety_stop:
                        # A lease release/cancel can race a policy tick that
                        # was already computing.  Publish only measured hold
                        # for that tick and keep the recoverable safety latch;
                        # this is not a controller/physics fault that should
                        # force a process restart.
                        if not self._publish_rejected_locomotion_tick_if_current(
                            lowstate,
                            rejected_transition_token,
                        ):
                            # A successful newer rearm/reference release won
                            # after the old support response.  It owns the
                            # boundary; this tick has no remaining authority
                            # to revoke or publish.
                            continue
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
                with self._support_lock:
                    effective_static_hold = (
                        self._commit_static_hold_decision_locked(
                            decision.static_hold_active
                        )
                    )
                output_action = {
                    f"{self._joint_name(index)}.q": target
                    for index, target in enumerate(decision.targets)
                }
                published_targets = self._publish(
                    output_action,
                    lowstate,
                    control_dt_s=control_step.dt_s,
                    control_tick_ms=control_step.tick_ms,
                    constrain_arms=not effective_static_hold,
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
                    static_hold_active=effective_static_hold,
                    motion_handoff_active=(
                        decision.motion_handoff_active
                        if not effective_static_hold
                        else False
                    ),
                    motion_handoff_ratio=(
                        decision.handoff_ratio
                        if not effective_static_hold
                        else 0.0
                    ),
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
        # Shutdown and normal-stop commits share this short linearization
        # lock.  Never retain it while acquiring/supporting the actuator lock.
        with self._terminal_fault_lock:
            self._stop.set()
        try:
            self._stop_navigation_at_boundary(source="process_stop")
        except Exception:
            logger.exception("failed to freeze SONIC during process stop")
        motion_ready = getattr(self, "_motion_ready", None)
        if motion_ready is not None:
            motion_ready.clear()
        condition = getattr(self, "_control_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()

    def _ensure_normal_stop_runtime_locked(self) -> NormalStopState:
        """Initialize normal-stop fields for lightweight/offline driver shells."""

        state = getattr(self, "_normal_stop", None)
        if state is None:
            state = NormalStopState()
            self._normal_stop = state
        if not isinstance(state, NormalStopState):
            raise RuntimeError("normal-stop state is invalid")
        if not hasattr(self, "_normal_stop_bridge_boot_id"):
            self._normal_stop_bridge_boot_id = None
        if not hasattr(self, "_normal_stop_seen_bridge_boot_ids"):
            self._normal_stop_seen_bridge_boot_ids = set()
        if not hasattr(self, "_normal_stop_plant_safety_stop_epoch"):
            self._normal_stop_plant_safety_stop_epoch = None
        if not hasattr(self, "_normal_stop_last_confirmed_plant_epoch"):
            self._normal_stop_last_confirmed_plant_epoch = 0
        if not hasattr(self, "_normal_stop_begin_intent"):
            self._normal_stop_begin_intent = None
        if not hasattr(self, "_normal_stop_seen_request_ids"):
            self._normal_stop_seen_request_ids = set()
        if not hasattr(self, "_normal_stop_replays"):
            self._normal_stop_replays = {}
        return state

    @staticmethod
    def _validate_normal_stop_intent(
        value: Mapping[str, object],
        *,
        expected_state: str,
        expected_stop_acknowledged: bool,
    ) -> tuple[dict[str, object], int, int]:
        """Validate an exact, zero gait envelope before ingress can mutate."""

        if not isinstance(value, Mapping):
            raise NormalStopWireError("normal-stop intent must be an object")
        fields = set(value)
        allowed = (
            _NORMAL_STOP_INTENT_FIELDS,
            _NORMAL_STOP_INTENT_FIELDS | {"measured_heading_yaw"},
        )
        if fields not in allowed:
            raise NormalStopWireError("normal-stop intent fields do not match v1")
        intent = dict(value)
        if (
            intent.get("type") != "sonic_intent"
            or intent.get("schema_version") != "h-ide.sonic-intent/v1"
            or intent.get("source") != "nav2"
            or intent.get("state") != expected_state
            or intent.get("mode") != "IDLE"
            or intent.get("motion_enabled") is not False
            or intent.get("stop_requested") is not True
            or intent.get("stop_acknowledged") is not expected_stop_acknowledged
        ):
            raise NormalStopWireError(
                f"normal-stop intent is not an exact {expected_state} stop"
            )
        generation = intent.get("generation")
        sequence = intent.get("sequence")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise NormalStopWireError("normal-stop intent generation is invalid")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
        ):
            raise NormalStopWireError("normal-stop intent sequence is invalid")
        generated_at = intent.get("generated_at_s")
        if (
            isinstance(generated_at, bool)
            or not isinstance(generated_at, (int, float))
            or not math.isfinite(float(generated_at))
            or float(generated_at) < 0.0
        ):
            raise NormalStopWireError("normal-stop intent timestamp is invalid")
        reason = intent.get("reason")
        if reason is not None and (
            not isinstance(reason, str)
            or len(reason) > 512
            or "\x00" in reason
        ):
            raise NormalStopWireError("normal-stop intent reason is invalid")
        for name in _NORMAL_STOP_ZERO_FIELDS:
            raw = intent.get(name)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) != 0.0
            ):
                raise NormalStopWireError(
                    f"normal-stop intent {name} must be exactly zero"
                )
        if "measured_heading_yaw" in intent:
            measured = intent["measured_heading_yaw"]
            if (
                isinstance(measured, bool)
                or not isinstance(measured, (int, float))
                or not math.isfinite(float(measured))
            ):
                raise NormalStopWireError(
                    "normal-stop measured heading must be finite"
                )
        return intent, generation, sequence

    def _normal_stop_replay_locked(
        self,
        *,
        request_id: UUID,
        canonical_request: bytes,
    ) -> dict[str, object] | None:
        replay = self._normal_stop_replays.get(request_id)
        if replay is not None:
            previous_request, previous_response = replay
            if previous_request != canonical_request:
                raise NormalStopWireError(
                    "normal-stop request_id was reused with another payload"
                )
            return copy.deepcopy(previous_response)
        if request_id in self._normal_stop_seen_request_ids:
            raise NormalStopWireError("normal-stop request_id belongs to an older transaction")
        return None

    def _record_normal_stop_replay_locked(
        self,
        *,
        request_id: UUID,
        canonical_request: bytes,
        response: dict[str, object],
    ) -> dict[str, object]:
        self._normal_stop_seen_request_ids.add(request_id)
        self._normal_stop_replays[request_id] = (
            canonical_request,
            copy.deepcopy(response),
        )
        return copy.deepcopy(response)

    def _normal_stop_response_locked(
        self,
        *,
        request_id: UUID,
        snapshot: NormalStopSnapshot,
        plant_epoch: int | None,
        expected_phases: frozenset[str],
    ) -> dict[str, object]:
        response = normal_stop_response(
            request_id=request_id,
            ok=snapshot.phase != "FAILED",
            robot_id=self.config.robot_id,
            driver_boot_id=self._driver_boot_id,
            snapshot=snapshot,
            plant_safety_stop_epoch=plant_epoch,
        )
        identity = snapshot.identity
        assert identity is not None
        parse_normal_stop_response(
            response,
            expected_request_id=request_id,
            expected_robot_id=self.config.robot_id,
            expected_driver_boot_id=self._driver_boot_id,
            expected_identity=identity,
            expected_control_generation=identity.control_generation,
            expected_gait_source_sequence=identity.gait_source_sequence,
            expected_phases=expected_phases,
        )
        return response

    def _handle_normal_stop_begin(
        self,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        canonical_request = canonical_normal_stop_json(payload)
        request = parse_normal_stop_request(
            payload,
            expected_robot_id=self.config.robot_id,
        )
        if not isinstance(request, NormalStopBeginRequest):
            raise NormalStopWireError("normal-stop begin request type is invalid")
        intent, generation, sequence = self._validate_normal_stop_intent(
            request.intent,
            expected_state="STOPPING",
            expected_stop_acknowledged=False,
        )
        intent_signature = canonical_normal_stop_json(intent)
        try:
            with self._support_lock:
                state = self._ensure_normal_stop_runtime_locked()
                current = state.snapshot
                active_bridge_boot = self._normal_stop_bridge_boot_id
                if (
                    current.phase != "NONE"
                    and request.bridge_boot_id != active_bridge_boot
                    and request.bridge_boot_id
                    in self._normal_stop_seen_bridge_boot_ids
                ):
                    raise NormalStopWireError(
                        "normal-stop begin belongs to an older bridge boot"
                    )
                with self._terminal_fault_lock:
                    if self._fault_latched.is_set():
                        raise RuntimeError(
                            "SONIC terminal fault cannot enter normal stop"
                        )
                    if self._stop.is_set():
                        raise RuntimeError(
                            "SONIC shutdown cannot enter normal stop"
                        )
                    replay = self._normal_stop_replay_locked(
                        request_id=request.request_id,
                        canonical_request=canonical_request,
                    )
                if replay is not None:
                    return replay

                same_association = (
                    current.control_generation == generation
                    and current.gait_source_sequence == sequence
                )
                if active_bridge_boot is None:
                    if current.phase != "NONE":
                        raise RuntimeError(
                            "normal-stop bridge binding is missing"
                        )
                    allow_sequence_reset = False
                elif request.bridge_boot_id == active_bridge_boot:
                    allow_sequence_reset = False
                else:
                    if current.phase != "DRIVER_CONFIRMED":
                        raise NormalStopWireError(
                            "bridge boot changed during an active normal stop"
                        )
                    assert current.control_generation is not None
                    if generation <= current.control_generation:
                        raise NormalStopWireError(
                            "new bridge boot requires a newer control generation"
                        )
                    allow_sequence_reset = True

                if same_association:
                    if (
                        request.bridge_boot_id != active_bridge_boot
                        or current.phase != "BRAKING"
                        or intent_signature != self._normal_stop_begin_intent
                    ):
                        raise NormalStopWireError(
                            "normal-stop begin retry does not match the braking request"
                        )
                    with self._terminal_fault_lock:
                        if self._fault_latched.is_set() or self._stop.is_set():
                            raise RuntimeError(
                                "SONIC shutdown/fault won normal-stop begin retry"
                            )
                        snapshot = state.begin(
                            control_generation=generation,
                            gait_source_sequence=sequence,
                        )
                else:
                    if current.phase == "FAILED":
                        raise NormalStopWireError(
                            "normal-stop failure requires a driver restart"
                        )
                    if current.phase in {
                        "BRAKING",
                        "STATIONARY_ACK",
                        "PLANT_CONFIRMED",
                    }:
                        raise NormalStopWireError(
                            "another normal-stop transaction is active"
                        )
                    if current.phase == "DRIVER_CONFIRMED":
                        assert current.control_generation is not None
                        assert current.gait_source_sequence is not None
                        if generation <= current.control_generation:
                            raise NormalStopWireError(
                                "normal stop requires a newer control generation"
                            )
                        if (
                            sequence <= current.gait_source_sequence
                            and not allow_sequence_reset
                        ):
                            raise NormalStopWireError(
                                "same bridge boot requires a newer gait source sequence"
                            )
                    # Revoke and freeze before the normal-stop transaction is
                    # made visible.  These are one-way safety strengthenings:
                    # a concurrent shutdown/fault may win after this point and
                    # leave the ingress sequence and normal-stop identity
                    # untouched.  Never retain the terminal lock while the
                    # boundary reaches planner/guard/handoff locks.
                    self._intent_ingress.revoke()
                    try:
                        self._stop_navigation_at_boundary(
                            source="normal_stop_begin",
                            publish_telemetry=False,
                        )
                    except Exception as exc:
                        raise _NormalStopBoundaryError(str(exc)) from exc
                    self._static_hold_active = True
                    with self._terminal_fault_lock:
                        if self._fault_latched.is_set():
                            raise RuntimeError(
                                "SONIC terminal fault won normal-stop begin"
                            )
                        if self._stop.is_set():
                            raise RuntimeError(
                                "SONIC shutdown won normal-stop begin"
                            )
                        decision = self._intent_ingress.accept(intent)
                        if (
                            decision.action != "stop"
                            or decision.state != "STOPPING"
                            or decision.generation != generation
                            or decision.sequence != sequence
                        ):
                            raise NormalStopWireError(
                                "normal-stop begin intent was not admitted exactly"
                            )
                        self._intent_ingress.revoke()
                        snapshot = state.begin(
                            control_generation=generation,
                            gait_source_sequence=sequence,
                            allow_source_sequence_reset=allow_sequence_reset,
                        )
                        self._normal_stop_replays.clear()
                        self._normal_stop_bridge_boot_id = request.bridge_boot_id
                        self._normal_stop_seen_bridge_boot_ids.add(
                            request.bridge_boot_id
                        )
                        self._normal_stop_plant_safety_stop_epoch = None
                        self._normal_stop_begin_intent = intent_signature

                response = self._normal_stop_response_locked(
                    request_id=request.request_id,
                    snapshot=snapshot,
                    plant_epoch=None,
                    expected_phases=frozenset({"BRAKING"}),
                )
                return self._record_normal_stop_replay_locked(
                    request_id=request.request_id,
                    canonical_request=canonical_request,
                    response=response,
                )
        except _NormalStopBoundaryError as exc:
            self._latch_terminal_fault(
                f"normal-stop actuator boundary fault: {exc}",
                boundary_already_stopped=True,
            )
            raise RuntimeError(
                f"normal-stop actuator boundary fault: {exc}"
            ) from exc

    def _handle_normal_stop_stationary(
        self,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        canonical_request = canonical_normal_stop_json(payload)
        with self._support_lock:
            state = self._ensure_normal_stop_runtime_locked()
            current = state.snapshot
            identity = current.identity
            bridge_boot_id = self._normal_stop_bridge_boot_id
            if identity is None or bridge_boot_id is None:
                raise NormalStopWireError(
                    "normal-stop stationary acknowledgement has no active identity"
                )
            request = parse_normal_stop_request(
                payload,
                expected_robot_id=self.config.robot_id,
                expected_bridge_boot_id=bridge_boot_id,
                expected_driver_boot_id=self._driver_boot_id,
                expected_identity=identity,
            )
            if not isinstance(request, NormalStopStationaryRequest):
                raise NormalStopWireError(
                    "normal-stop stationary request type is invalid"
                )
            with self._terminal_fault_lock:
                if self._fault_latched.is_set():
                    raise RuntimeError(
                        "SONIC terminal fault cannot acknowledge normal stop"
                    )
                if self._stop.is_set():
                    raise RuntimeError(
                        "SONIC shutdown cannot acknowledge normal stop"
                    )
                replay = self._normal_stop_replay_locked(
                    request_id=request.request_id,
                    canonical_request=canonical_request,
                )
            if replay is not None:
                return replay
            if current.phase != "BRAKING":
                raise NormalStopWireError(
                    "stationary acknowledgement requires the BRAKING phase"
                )
            intent, generation, sequence = self._validate_normal_stop_intent(
                request.intent,
                expected_state="DISARMED",
                expected_stop_acknowledged=True,
            )
            if generation != identity.control_generation:
                raise NormalStopWireError(
                    "stationary intent control generation mismatch"
                )
            if sequence <= identity.gait_source_sequence:
                raise NormalStopWireError(
                    "stationary intent must follow the first braking sequence"
                )
            # Begin already established the physical/local static boundary.
            # Keep this commit short so terminal Plant halt never waits on
            # planner, guard, or handoff work.
            with self._terminal_fault_lock:
                if self._fault_latched.is_set():
                    raise RuntimeError(
                        "SONIC terminal fault won stationary acknowledgement"
                    )
                if self._stop.is_set():
                    raise RuntimeError(
                        "SONIC shutdown won stationary acknowledgement"
                    )
                decision = self._intent_ingress.accept(intent)
                if (
                    decision.action != "stop"
                    or decision.state != "DISARMED"
                    or decision.generation != generation
                    or decision.sequence != sequence
                ):
                    raise NormalStopWireError(
                        "stationary intent was not admitted exactly"
                    )
                self._intent_ingress.revoke()
                snapshot = state.acknowledge_stationary(identity)
                # This flag begins only after the exact post-request
                # stationary evidence was admitted.  It now blocks any
                # legacy support release while Bridge obtains Plant proof.
                self._plant_support_stop_pending = True
            response = self._normal_stop_response_locked(
                request_id=request.request_id,
                snapshot=snapshot,
                plant_epoch=None,
                expected_phases=frozenset({"STATIONARY_ACK"}),
            )
            return self._record_normal_stop_replay_locked(
                request_id=request.request_id,
                canonical_request=canonical_request,
                response=response,
            )

    def _require_normal_stop_local_hold_locked(
        self,
        identity: NormalStopIdentity,
        *,
        plant_epoch: int,
    ) -> None:
        """Prove every local fact required for DRIVER_CONFIRMED.

        The caller owns the support and terminal locks.  Plant evidence may
        establish the external support facts, but it cannot manufacture an
        actuator boundary, revoke a reference, or make a non-zero command
        static.  Those facts must already be exact from the braking phase.
        """

        generation = identity.control_generation
        last_targets = self._last_published_targets
        safe_targets = self._safe_hold_targets
        try:
            exact_targets = (
                isinstance(last_targets, tuple)
                and isinstance(safe_targets, tuple)
                and len(last_targets) == 29
                and len(safe_targets) == 29
                and all(math.isfinite(float(value)) for value in last_targets)
                and all(math.isfinite(float(value)) for value in safe_targets)
                and safe_targets == last_targets
            )
            authority = self._reference_authority.snapshot
            facts = {
                "exact executed static-hold targets": exact_targets,
                "static hold": self._static_hold_active is True,
                "zero command": self.guard.command_zero is True,
                "inactive navigation task": self._navigation_task_active is False,
                "revoked ingress": (
                    self._intent_ingress.generation == generation
                    and self._intent_ingress.armed is False
                    and self._intent_ingress.staged_control_arm_generation is None
                    and self._intent_ingress.support_stop_generation == generation
                ),
                "revoked reference authority": (
                    authority.phase == "REVOKED"
                    and authority.actuator_authority is False
                    and self._reference_request_pending is False
                ),
                "confirmed Plant support latch": (
                    self._plant_support_mode == "safety_stop"
                    and self._plant_support_safety_latched is True
                    and self._plant_support_stop_pending is False
                    and self._plant_support_stop_generation == generation
                    and self._normal_stop_plant_safety_stop_epoch == plant_epoch
                ),
            }
        except Exception as exc:
            raise _NormalStopLocalHoldError(
                f"normal-stop local safe hold could not be inspected: {exc}"
            ) from exc
        missing = [name for name, present in facts.items() if not present]
        if missing:
            raise _NormalStopLocalHoldError(
                "normal-stop local safe hold is incomplete: " + ", ".join(missing)
            )

    def _handle_normal_stop_plant(
        self,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        canonical_request = canonical_normal_stop_json(payload)
        try:
            with self._support_lock:
                state = self._ensure_normal_stop_runtime_locked()
                current = state.snapshot
                identity = current.identity
                bridge_boot_id = self._normal_stop_bridge_boot_id
                if identity is None or bridge_boot_id is None:
                    raise NormalStopWireError(
                        "normal-stop Plant acknowledgement has no active identity"
                    )
                # Correlation is complete before Plant evidence is returned to
                # this handler.  An old identity/boot therefore cannot mutate
                # ingress, support, epoch, or the normal-stop state machine.
                request = parse_normal_stop_request(
                    payload,
                    expected_robot_id=self.config.robot_id,
                    expected_bridge_boot_id=bridge_boot_id,
                    expected_driver_boot_id=self._driver_boot_id,
                    expected_identity=identity,
                )
                if not isinstance(request, NormalStopPlantRequest):
                    raise NormalStopWireError(
                        "normal-stop Plant request type is invalid"
                    )
                with self._terminal_fault_lock:
                    if self._fault_latched.is_set():
                        raise RuntimeError(
                            "SONIC terminal fault cannot confirm normal stop"
                        )
                    if self._stop.is_set():
                        raise RuntimeError(
                            "SONIC shutdown cannot confirm normal stop"
                        )
                    replay = self._normal_stop_replay_locked(
                        request_id=request.request_id,
                        canonical_request=canonical_request,
                    )
                if replay is not None:
                    return replay

                evidence = request.plant
                plant_epoch = evidence.safety_stop_epoch
                failure_reason: str | None = None
                if not evidence.confirmed:
                    failure_reason = (
                        "Plant normal-stop confirmation failed: "
                        f"{evidence.error}"
                    )[:512]
                else:
                    assert plant_epoch is not None
                    if (
                        evidence.last_rearm_generation is not None
                        and evidence.last_rearm_generation
                        > identity.control_generation
                    ):
                        failure_reason = (
                            "Plant normal-stop evidence has a rearm generation "
                            "newer than the stopped control generation"
                        )

                if current.phase == "DRIVER_CONFIRMED":
                    if failure_reason is not None:
                        raise NormalStopWireError(
                            "a confirmed normal stop cannot accept failed Plant evidence"
                        )
                    if (
                        plant_epoch
                        != self._normal_stop_plant_safety_stop_epoch
                    ):
                        raise NormalStopWireError(
                            "normal-stop Plant retry epoch does not match the transaction"
                        )
                    with self._terminal_fault_lock:
                        if self._fault_latched.is_set() or self._stop.is_set():
                            raise RuntimeError(
                                "SONIC shutdown/fault won normal-stop Plant retry"
                            )
                        if (
                            plant_epoch
                            != self._normal_stop_last_confirmed_plant_epoch
                        ):
                            raise NormalStopWireError(
                                "normal-stop Plant retry epoch is not the last "
                                "driver-confirmed epoch"
                            )
                        try:
                            self._require_normal_stop_local_hold_locked(
                                identity,
                                plant_epoch=plant_epoch,
                            )
                        except _NormalStopLocalHoldError as exc:
                            # A new request ID is a fresh observation, not a
                            # byte-equivalent transport replay.  A legitimate
                            # later rearm or fact drift invalidates the old
                            # safe-hold result without manufacturing a terminal
                            # fault.  The original request ID remains cached as
                            # the historical transaction response.
                            raise NormalStopWireError(
                                "normal-stop Plant retry no longer has the "
                                f"confirmed local hold: {exc}"
                            ) from exc
                        snapshot = state.confirm_driver(identity)
                    response = self._normal_stop_response_locked(
                        request_id=request.request_id,
                        snapshot=snapshot,
                        plant_epoch=plant_epoch,
                        expected_phases=frozenset({"DRIVER_CONFIRMED"}),
                    )
                    return self._record_normal_stop_replay_locked(
                        request_id=request.request_id,
                        canonical_request=canonical_request,
                        response=response,
                    )

                if current.phase != "STATIONARY_ACK":
                    raise NormalStopWireError(
                        "Plant acknowledgement requires the STATIONARY_ACK phase"
                    )
                if (
                    failure_reason is None
                    and plant_epoch is not None
                    and plant_epoch
                    <= self._normal_stop_last_confirmed_plant_epoch
                ):
                    failure_reason = (
                        "Plant safety-stop epoch did not advance beyond the "
                        "last confirmed transaction"
                    )
                if (
                    failure_reason is None
                    and self._normal_stop_plant_safety_stop_epoch is not None
                ):
                    raise _NormalStopLocalHoldError(
                        "stationary normal stop already carries a Plant epoch"
                    )

                if failure_reason is not None:
                    with self._terminal_fault_lock:
                        if self._fault_latched.is_set():
                            raise RuntimeError(
                                "SONIC terminal fault won failed Plant evidence"
                            )
                        if self._stop.is_set():
                            raise RuntimeError(
                                "SONIC shutdown won failed Plant evidence"
                            )
                        snapshot = state.fail(
                            identity,
                            reason=failure_reason,
                        )
                    response = self._normal_stop_response_locked(
                        request_id=request.request_id,
                        snapshot=snapshot,
                        plant_epoch=None,
                        expected_phases=frozenset({"FAILED"}),
                    )
                    return self._record_normal_stop_replay_locked(
                        request_id=request.request_id,
                        canonical_request=canonical_request,
                        response=response,
                    )

                assert plant_epoch is not None
                # The external Plant proof and every local authority fact form
                # one short commit.  No boundary, network, planner, guard, or
                # handoff call occurs while the terminal lock is retained.
                with self._terminal_fault_lock:
                    if self._fault_latched.is_set():
                        raise RuntimeError(
                            "SONIC terminal fault won Plant confirmation"
                        )
                    if self._stop.is_set():
                        raise RuntimeError(
                            "SONIC shutdown won Plant confirmation"
                        )
                    state.confirm_plant(identity)
                    stopped_generation = self._intent_ingress.latch_support_stop()
                    self._plant_support_mode = "safety_stop"
                    self._plant_support_stop_pending = False
                    self._plant_support_safety_latched = True
                    self._plant_support_stop_generation = identity.control_generation
                    self._normal_stop_plant_safety_stop_epoch = plant_epoch
                    if stopped_generation != identity.control_generation:
                        raise _NormalStopLocalHoldError(
                            "Plant stop generation does not match the transaction"
                        )
                    self._require_normal_stop_local_hold_locked(
                        identity,
                        plant_epoch=plant_epoch,
                    )
                    snapshot = state.confirm_driver(identity)
                    self._normal_stop_last_confirmed_plant_epoch = plant_epoch

                response = self._normal_stop_response_locked(
                    request_id=request.request_id,
                    snapshot=snapshot,
                    plant_epoch=plant_epoch,
                    expected_phases=frozenset({"DRIVER_CONFIRMED"}),
                )
                return self._record_normal_stop_replay_locked(
                    request_id=request.request_id,
                    canonical_request=canonical_request,
                    response=response,
                )
        except _NormalStopLocalHoldError as exc:
            self._latch_terminal_fault(
                f"normal-stop driver hold fault: {exc}",
            )
            raise RuntimeError(
                f"normal-stop driver hold fault: {exc}"
            ) from exc

    def accept_json(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if payload.get("type") == "normal_stop_begin":
            return self._handle_normal_stop_begin(payload)
        if payload.get("type") == "normal_stop_stationary_ack":
            return self._handle_normal_stop_stationary(payload)
        if payload.get("type") == "normal_stop_plant_ack":
            return self._handle_normal_stop_plant(payload)
        if (
            isinstance(payload.get("type"), str)
            and str(payload["type"]).startswith("normal_stop_")
        ):
            raise NormalStopWireError("unsupported normal-stop request type")
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
            runtime_diagnostics = getattr(self, "_runtime_diagnostics", None)
            if runtime_diagnostics is not None:
                runtime_diagnostics.observe_health_probe(checked_at)
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
                authority = self._reference_authority.snapshot
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
            # The terminal claim is lock-free with respect to support so Plant
            # halt cannot wait for DDS or inference.  During the short local
            # teardown catch-up window, health must nevertheless expose the
            # strengthened terminal authority boundary rather than an
            # impossible terminal+GRANTED combination.
            visible_authority_phase = (
                "REVOKED" if terminal_fault else authority.phase
            )
            visible_actuator_authority = (
                False if terminal_fault else authority.actuator_authority
            )
            visible_granted_at_monotonic_s = (
                None
                if terminal_fault
                else authority.granted_at_monotonic_s
            )
            visible_granted_at_tick_ms = (
                None if terminal_fault else authority.granted_at_tick_ms
            )
            visible_dds_target_sequence = (
                None if terminal_fault else authority.dds_target_sequence
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
                "matched_reference_actuator_authority": (
                    visible_actuator_authority
                ),
                "reference_authority_state": visible_authority_phase,
                "reference_authority_generation": authority.generation,
                "reference_authority_source_sequence": authority.source_sequence,
                "reference_authority_intent_generation": (
                    authority.planner_intent_generation
                ),
                "reference_authority_epoch": authority.reference_epoch,
                "reference_authority_granted_at_monotonic_s": (
                    visible_granted_at_monotonic_s
                ),
                "reference_authority_granted_tick_ms": (
                    visible_granted_at_tick_ms
                ),
                "reference_authority_dds_target_sequence": (
                    visible_dds_target_sequence
                ),
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
            if payload.get("control_arm") or payload.get("request_id"):
                raise RuntimeError(
                    "SONIC terminal fault requires an explicit process restart"
                )
            return None
        if payload.get("type") == "stop":
            reason = payload.get("reason")
            boundary_error: Exception | None = None
            with self._support_lock:
                try:
                    self._stop_navigation_at_boundary(
                        source="bridge_stop",
                        publish_telemetry=False,
                    )
                except Exception as exc:
                    boundary_error = exc
                self._intent_ingress.revoke()
                if payload.get("source") == "robot_agent_safe_stop":
                    self._plant_support_stop_pending = True
                support_latched = self._plant_support_safety_latched
                support_pending = self._plant_support_stop_pending
                confirmed_mode = self._plant_support_mode or "unknown"
            if boundary_error is not None:
                self._latch_terminal_fault(
                    f"bridge stop boundary fault: {boundary_error}",
                    boundary_already_stopped=True,
                )
                return None
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
                **self._planner_telemetry_fields(),
            )
            return None
        control_arm = payload.get("control_arm", False)
        stop_boundary_applied = False
        stop_boundary_error: Exception | None = None
        try:
            if control_arm:
                # Timestamp and support predicates are sampled under their two
                # owner locks, and ingress is staged in that same critical
                # section.  No generation/sequence mutation can precede this
                # fresh, complete safe-hold gate.
                with self._control_condition:
                    with self._support_lock:
                        self._require_fresh_control_arm_health_locked(
                            phase="control arm precheck"
                        )
                        decision = self._intent_ingress.accept(
                            payload,
                            defer_control_arm=True,
                        )
            else:
                with self._support_lock:
                    decision = self._intent_ingress.accept(payload)
                    if decision.action == "stop":
                        try:
                            self._stop_navigation_at_boundary(
                                source=(
                                    "terminal_fault"
                                    if decision.state == "FAULT"
                                    else "gait_adapter"
                                ),
                                publish_telemetry=False,
                            )
                        except Exception as exc:
                            stop_boundary_error = exc
                        stop_boundary_applied = True
        except (KeyError, TypeError, ValueError) as exc:
            self._stop_navigation_at_boundary(source="invalid_gait_intent")
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
        if not control_arm and decision.action == "stop":
            # Ingress acceptance and the in-memory actuator/reference stop are
            # one support-lock transaction. Plant/telemetry I/O follows only
            # after control can no longer consume the accepted stale intent.
            assert stop_boundary_applied
            if decision.state == "FAULT" or stop_boundary_error is not None:
                terminal_reason = decision.reason or "gait adapter entered FAULT"
                if stop_boundary_error is not None:
                    terminal_reason = (
                        f"{terminal_reason}; gait stop boundary fault: "
                        f"{stop_boundary_error}"
                    )
                self._latch_terminal_fault(
                    terminal_reason,
                    boundary_already_stopped=True,
                )
                return None
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
                **self._planner_telemetry_fields(),
            )
            return None
        if control_arm and decision.heading_yaw is not None:
            self.guard.observe_heading(decision.heading_yaw)
        if control_arm:
            if not decision.arm_acknowledged or decision.generation is None:
                with self._support_lock:
                    self._intent_ingress.revoke()
                raise ValueError(
                    "control_arm did not produce a fresh stationary IDLE acknowledgement"
                )
            if not self._request_staged_plant_support_rearm(decision.generation):
                with self._support_lock:
                    self._intent_ingress.revoke()
                raise RuntimeError("explicit Plant support rearm failed")
            try:
                with self._control_condition:
                    with self._support_lock:
                        self._require_fresh_control_arm_health_locked(
                            phase="control arm post-rearm check"
                        )
                        self._commit_staged_plant_support_rearm_locked(
                            decision.generation
                        )
            except Exception as exc:
                with self._support_lock:
                    self._intent_ingress.revoke()
                self._ensure_plant_support_stopped(
                    reason=f"control arm post-rearm health failed: {exc}"
                )
                raise RuntimeError(
                    f"control arm post-rearm health failed: {exc}"
                ) from exc
            self._publish_committed_plant_rearm(decision.generation)
            return {
                "ok": True,
                "mode": "idle",
                "generation": decision.generation,
            }
        with self._support_lock:
            support_latched = (
                self._plant_support_stop_pending
                or self._plant_support_safety_latched
            )
        if support_latched:
            self._stop_navigation_at_boundary(source="plant_safety_stop")
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
            self._stop_navigation_at_boundary(source="sonic_startup_hold")
            self.telemetry.update(
                nav2_state="startup_hold",
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                last_error="SONIC control loop is still warming",
            )
            return None
        new_navigation_task = False
        with self._support_lock:
            if (
                not self._intent_ingress.armed
                or decision.generation != self._intent_ingress.generation
                or decision.sequence != self._intent_ingress.sequence
            ):
                # A stop or a newer envelope won after this motion decision was
                # parsed. It already owns the boundary; never replay the stale
                # decision into CmdVelGuard after reacquiring the lock.
                return None
            if (
                self._plant_support_stop_pending
                or self._plant_support_safety_latched
            ):
                self._stop_navigation_at_boundary(
                    source="plant_safety_stop"
                )
                return None
            if not self._navigation_task_active:
                if decision.generation is None or decision.sequence is None:
                    raise RuntimeError("motion intent lacks generation/sequence identity")
                self._reference_source_identity = (
                    decision.generation,
                    decision.sequence,
                )
                self._navigation_task_active = True
                new_navigation_task = True
            elif (
                self._reference_source_identity is None
                or self._reference_source_identity[0] != decision.generation
            ):
                raise RuntimeError(
                    "active navigation task changed generation without a stop boundary"
                )
            if decision.heading_yaw is not None:
                self.guard.observe_heading(decision.heading_yaw)
            # Keep any frozen actuator target until the matched reference is
            # installed and measured history is rebased. Repeated STARTING
            # envelopes therefore cannot overwrite the first admitted identity.
            self.guard.accept(decision.vx, decision.vy, decision.wz, source="nav2")
        if new_navigation_task:
            self.telemetry.reset_navigation_task_peaks()
            self.telemetry.update(navigation_task_active=True)
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
    runtime_diagnostics = SonicRuntimeDiagnostics(
        stall_threshold_s=min(
            config.plant_lowstate_stale_s,
            config.sonic_control_step_stale_s,
        )
    )
    telemetry.set_runtime_diagnostics_provider(runtime_diagnostics.snapshot)
    try:
        driver = SonicDriver(
            config,
            telemetry,
            runtime_diagnostics=runtime_diagnostics,
        )
    except Exception:
        runtime_diagnostics.close()
        raise

    # A tiny localhost TCP command server.  ROS bridge is the only intended
    # client; it sends one JSON object per line and never receives actuator data.
    try:
        server = _bind_command_server(
            config.command_bind_host,
            config.command_port,
        )
    except Exception:
        driver.stop()
        runtime_diagnostics.close()
        raise
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
        runtime_diagnostics.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_driver(SimConfig.from_file(args.config))
