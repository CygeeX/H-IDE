"""Unitree DDS SONIC driver.

This process is intentionally separate from the MuJoCo plant.  The plant and
driver communicate only through the official `rt/lowstate`/`rt/lowcmd` DDS
topics, which keeps the simulation boundary close to the real Unitree stack.
"""

from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import dataclass, replace
import faulthandler
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

from .arm_limiter_evidence import (
    ARM_JOINT_INDICES,
    ArmLimiterEvidenceError,
    ArmLimiterEvidenceRecorder,
    source_bundle_sha256,
)
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
    JointGuardResult,
    NavigationArmGuard,
    NominalPolicyActivation,
    NominalPolicyActivationDecision,
    NominalPolicyActivationFault,
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
    clear_plant_motion_fault,
    latch_plant_physics_fault,
    latch_plant_motion_fault,
    parse_lines,
    read_plant_identity,
    rearm_plant_support,
    release_plant_fall_arrest,
    set_plant_support,
)
from .unitree_dds import initialize_unitree_control_channel

logger = logging.getLogger(__name__)

# Correlate each simulator-only LowCmd with the LowState control tick that
# produced it.  The matching Plant bridge reads these application-reserved
# fields before advancing untethered MuJoCo physics.  An out-of-band stop/fault
# write uses the second marker and is valid at any waiting tick.
_SIM_CONTROL_TICK_MAGIC = 0x48494445
_SIM_ASYNC_BOUNDARY_MAGIC = 0x48494446


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
        self.braking = False
        self.support_stop_generation: int | None = None
        self.staged_control_arm_generation: int | None = None

    def revoke(self) -> None:
        """Drop motion authority without manufacturing a new generation."""

        self.armed = False
        self.braking = False
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
        self.braking = False
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
        self.braking = False
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
            self.braking = False
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
            controlled_brake = bool(
                state == "STOPPING"
                and stop_requested
                and not motion_enabled
                and not stop_acknowledged
                and not control_arm
                and self.support_stop_generation is None
                and (self.armed or self.braking)
            )
            if defer_control_arm and control_arm and arm_candidate:
                self.armed = False
                self.staged_control_arm_generation = generation
            else:
                self.armed = arm_candidate
                self.staged_control_arm_generation = None
            self.braking = controlled_brake
            return IngressDecision(
                "brake" if controlled_brake else "stop",
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
        self.braking = False
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
        self._driver_session_id = uuid4()
        self.telemetry.update(
            driver_boot_id=str(self._driver_boot_id),
            driver_session_id=str(self._driver_session_id),
        )
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
        # Controller construction and the isolated first inference can starve
        # the Python DDS callback while MuJoCo continues publishing a held
        # robot.  Track callback progress independently from the strict clock
        # so startup can drain that cold-path backlog before choosing its
        # baseline.  Once the first control-arm transaction commits,
        # ``_runtime_clock_committed`` never becomes false again: identical
        # gaps during or after motion remain terminal.
        self._lowstate_sequence = 0
        self._runtime_clock_committed = False
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
        self._control_loop_stage = "startup"
        self._control_stall_diagnostic_armed = threading.Event()
        self._control_stall_diagnostic_dumped = threading.Event()
        self._lowstate_stall_timeout_s = max(1.0, config.command_timeout_s * 4.0)
        self._stop = threading.Event()
        self._fault_latched = threading.Event()
        self._terminal_fault_lock = threading.Lock()
        self._terminal_plant_halt_attempted = False
        self._terminal_fault_reason: str | None = None
        self._recoverable_motion_fault = False
        self._motion_fault_safe_hold_frames = 0
        self._motion_fault_last_hold_tick_ms: int | None = None
        self._motion_fault_reset_replays: dict[
            UUID, tuple[str, dict[str, object]]
        ] = {}
        self._motion_fault_latch_replays: dict[
            UUID, tuple[str, dict[str, object]]
        ] = {}
        self._tick_guard = PlantTickGuard()
        self._intent_ingress = SonicIntentIngress(
            max_age_s=max(0.50, config.command_timeout_s * 2.0)
        )
        self._navigation_task_active = False
        self._controlled_stop_active = False
        self._balanced_policy_idle_active = False
        self._controlled_stop_source_identity: tuple[int, int] | None = None
        self._controlled_stop_reference_installed = False
        self._input_buffer = b""
        self._last_intent_signature: tuple[bool, float, float, float] | None = None
        self._decoder_facing_rad: float | None = None
        self._pending_decoder_heading_delta_rad = 0.0
        self._intent_gate = ReplanIntentGate()
        self._joint_handoff = StaticJointHandoff(
            warmup_s=config.motion_handoff_warmup_s,
            blend_s=config.motion_handoff_blend_s,
        )
        self._policy_activation = NominalPolicyActivation(
            duration_s=config.policy_activation_s,
            # The product profile releases the fall-arrest only after the
            # first learned-policy DDS target has succeeded.  This explicit
            # flag is intentionally independent of the legacy standalone
            # Plant timer: a duration must never substitute for a correlated
            # actuator/activation epoch.
            release_fall_arrest=config.policy_release_fall_arrest,
        )
        self._policy_activation_requested = False
        self._policy_activation_release_epoch: int | None = None
        self._policy_activation_release_result: dict[str, Any] | None = None
        self._policy_activation_release_error: Exception | None = None
        self._policy_activation_release_thread: threading.Thread | None = None
        self._policy_activation_release_confirmed_epoch: int | None = None
        self._policy_activation_prepared_tick_ms: int | None = None
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
        self._normal_stop_policy_braking = False
        self._normal_stop_seen_request_ids: set[UUID] = set()
        self._normal_stop_replays: dict[
            UUID,
            tuple[bytes, dict[str, object]],
        ] = {}
        self._reference_authority = ReferenceAuthorityState()
        self._reference_source_identity: tuple[int, int] | None = None
        self._reference_request_pending = False
        self._reference_request_continuous = False
        self._vendor_revoke_pending = False
        self._dds_target_sequence = 0
        self._control_evidence_context = threading.local()
        # Evidence control requests share this outer boundary with every DDS
        # write.  It never participates in actuator/stop lock ordering; it
        # merely prevents begin/finalize from slicing through a physical write.
        self._arm_evidence_boundary_lock = threading.RLock()
        self._arm_evidence_write_condition = threading.Condition()
        self._arm_evidence_writes_inflight: set[object] = set()

        # These imports are intentionally lazy: `check` and safety tests must
        # work without ROS/Unitree SDK/ONNX installed.
        from unitree_sdk2py.core.channel import Channel, ChannelFactory, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from lerobot.robots.unitree_g1.controllers.sonic_whole_body import SonicWholeBodyController

        self._sdk = (ChannelPublisher, ChannelSubscriber, LowCmd_, LowState_, CRC, unitree_hg_msg_dds__LowCmd_)
        initialize_unitree_control_channel(
            ChannelFactory(),
            Channel,
            domain_id=config.unitree_domain_id,
            interface=config.interface,
        )
        self._publisher = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        # Do not use Unitree SDK's Listener -> BQueue callback adapter here.
        # Its listener takes exactly one DDS sample per data-available edge.  If
        # the 250 Hz Plant gets ahead of that callback, unread samples can keep
        # the edge asserted without another callback, leaving the Python queue
        # empty even though a direct DataReader still sees a healthy stream.
        # A dedicated blocking reader drains the authoritative DDS reader
        # directly and then enters the existing serialized LowState callback.
        self._subscriber.Init()
        self._lowstate_reader_stop = threading.Event()
        self._lowstate_reader_close_lock = threading.Lock()
        self._lowstate_reader_closed = False
        self._lowstate_reader_thread: threading.Thread | None = None
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
        self._arm_evidence = ArmLimiterEvidenceRecorder(
            sonic_driver_boot_id=str(self._driver_boot_id),
            sonic_driver_session_id=str(self._driver_session_id),
            sonic_driver_source_sha256=source_bundle_sha256(
                (
                    Path(__file__).resolve(),
                    Path(__file__).with_name("safety.py").resolve(),
                    Path(__file__).with_name("arm_limiter_evidence.py").resolve(),
                )
            ),
        )
        self._arm_evidence_request_replays: dict[
            UUID, tuple[bytes, dict[str, object]]
        ] = {}
        self._joint_names = tuple(self._joint_name(index) for index in range(29))
        self._fault_trace = JointFaultTraceBuffer(
            self._joint_names,
            max_samples=max(1, config.control_hz * 5),
        )
        self._fault_trace_path = self._default_fault_trace_path()
        self._lowstate_reader_thread = threading.Thread(
            target=self._run_lowstate_reader,
            daemon=True,
            name="sonic-lowstate-reader",
        )
        self._lowstate_reader_thread.start()

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
        # balance.  Enforce the checked-in Unitree G1 XML's authoritative,
        # asymmetric arm joint ranges instead of inventing a symmetric range
        # around DEFAULT_ANGLES.  The latter rejected policy-valid shoulder
        # counter-swing before it reached MuJoCo.  The per-tick velocity bound
        # remains unchanged.  Saturation duration is recorded as evidence but
        # is not assigned an invented time-to-fault absent a vendor contract.
        per_arm_lower = (
            -3.0892,
            -1.5882,
            -2.618,
            -1.0472,
            -1.97222,
            -1.61443,
            -1.61443,
        )
        per_arm_upper = (
            2.6704,
            2.2515,
            2.618,
            2.0944,
            1.97222,
            1.61443,
            1.61443,
        )
        per_arm_velocity = (6.0, 6.0, 6.0, 6.0, 4.0, 4.0, 4.0)
        return NavigationArmGuard(
            DEFAULT_ANGLES[15:29],
            lower_limits_rad=per_arm_lower + (
                per_arm_lower[0],
                -2.2515,
                *per_arm_lower[2:],
            ),
            upper_limits_rad=per_arm_upper + (
                per_arm_upper[0],
                1.5882,
                *per_arm_upper[2:],
            ),
            max_velocity_rps=per_arm_velocity * 2,
            max_continuous_saturation_s=None,
        )

    def _on_lowstate(self, message):
        try:
            tick = int(message.tick)
            restarted = self._tick_guard.observe(tick)
        except (AttributeError, TypeError, ValueError) as exc:
            self._latch_terminal_fault(f"invalid lowstate tick: {exc}")
            return
        clock_fault: str | None = None
        startup_requalified = False
        received_at_wall_s = self._wall_clock()
        runtime_diagnostics = getattr(self, "_runtime_diagnostics", None)
        if runtime_diagnostics is not None:
            runtime_diagnostics.observe_lowstate(received_at_wall_s)
        with self._control_condition:
            self._lowstate = message
            self._lowstate_sequence += 1
            self._last_lowstate_wall_s = received_at_wall_s
            if not self._control_clock_armed:
                self._control_condition.notify_all()
            if restarted:
                clock_fault = (
                    "MuJoCo plant tick rolled back; explicit process restart required"
                )
            elif self._control_clock_armed:
                previous_tick = self._control_clock.last_observed_tick_ms
                pre_authority_forward_gap = bool(
                    not self._runtime_clock_committed
                    and previous_tick is not None
                    and tick - previous_tick
                    > self._control_clock.max_source_gap_ms
                )
                if pre_authority_forward_gap:
                    # ONNX/GPU warmup may hold the Python DDS callback while
                    # Plant continues to publish a stationary measured hold.
                    # Before the first successful control-arm transaction no
                    # controller state, policy cursor, or actuator motion is
                    # authorized, so invalidate all pending credit and require
                    # a fresh consecutive 50 Hz window.  The identical gap
                    # after that authority boundary remains terminal.
                    self._control_clock.rebase_stationary_qualification(tick)
                    self._pending_control_sample = None
                    self._startup_qualification_dropped_samples += 1
                    self._startup_qualification_drop_generation += 1
                    self._startup_qualification_consecutive_steps = 0
                    self._startup_qualifying = True
                    self._motion_ready.clear()
                    startup_requalified = True
                    self._control_condition.notify_all()
                    control_step = None
                else:
                    try:
                        control_step = self._control_clock.observe(tick)
                    except PlantControlClockFault as exc:
                        clock_fault = str(exc)
                        control_step = None
                if clock_fault is None:
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
        if startup_requalified:
            self.telemetry.update(
                status="qualifying",
                control_clock_status="startup_requalification",
                motion_ready=False,
            )
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
            if clock_fault.startswith("MuJoCo control sample overrun:"):
                logger.error(
                    "capturing all SONIC thread stacks at control-sample overrun"
                )
                faulthandler.dump_traceback(all_threads=True)
            self._latch_terminal_fault(clock_fault)

    def _run_lowstate_reader(self) -> None:
        """Drain LowState with a blocking DataReader instead of SDK callbacks."""

        while (
            not self._lowstate_reader_stop.is_set()
            and not self._stop.is_set()
        ):
            try:
                message = self._subscriber.Read(timeout=0.05)
            except Exception as exc:
                if not self._lowstate_reader_stop.is_set() and not self._stop.is_set():
                    self._latch_terminal_fault(
                        "MuJoCo LowState direct reader failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                return
            if message is not None:
                self._on_lowstate(message)
                # The DDS reader owns a reliable queue, while the control loop
                # owns the single admitted slot below.  Do not drain a burst
                # into that slot: wait until the exact sample has been taken
                # before reading the next queued sample.  The existing wall
                # liveness interlock still faults a genuinely stalled control
                # loop; this handshake neither drops nor fabricates ticks.
                with self._control_condition:
                    while (
                        self._pending_control_sample is not None
                        and not self._lowstate_reader_stop.is_set()
                        and not self._stop.is_set()
                        and not (
                            self._fault_latched.is_set()
                            and not getattr(
                                self, "_recoverable_motion_fault", False
                            )
                        )
                    ):
                        self._control_condition.wait(timeout=0.05)

    def _close_lowstate_reader(self) -> None:
        """Bound and idempotently release the dedicated DDS reader."""

        stop_event = getattr(self, "_lowstate_reader_stop", None)
        if stop_event is None:
            return
        stop_event.set()
        thread = getattr(self, "_lowstate_reader_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.25)
            if thread.is_alive():
                logger.warning("SONIC LowState reader did not stop within 250ms")
                return
        close_lock = getattr(self, "_lowstate_reader_close_lock", None)
        if close_lock is None:
            return
        with close_lock:
            if self._lowstate_reader_closed:
                return
            self._subscriber.Close()
            self._lowstate_reader_closed = True

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

    def _wait_for_post_warm_lowstate_stream(self, *, timeout_s: float = 5.0):
        """Choose a startup baseline only after cold inference backlog drains.

        The subscriber is created before the three ONNX sessions so Plant can
        remain held while those sessions load.  A long GIL/ORT scheduling
        pause can nevertheless leave ``_lowstate`` older than the first
        callback delivered after warm-up.  Arming against that stale sample
        manufactures a multi-second source jump even though no motion was
        possible.  Require a fresh, bounded-gap callback window while the
        strict clock is still disarmed; no sample earns policy or startup
        qualification credit here.
        """

        if not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
            raise ValueError("post-warm LowState timeout must be positive and finite")
        deadline = self._wall_clock() + float(timeout_s)
        required_samples = self._STARTUP_QUALIFICATION_STEPS + 1
        previous_tick: int | None = None
        consecutive = 0
        with self._control_condition:
            if self._control_clock_armed:
                raise RuntimeError("post-warm LowState drain requires a disarmed clock")
            observed_sequence = self._lowstate_sequence
            self.telemetry.update(
                control_clock_status="waiting_for_fresh_post_warm_lowstate",
                motion_ready=False,
            )
            while not self._fault_latched.is_set() and not self._stop.is_set():
                if self._lowstate_sequence != observed_sequence:
                    observed_sequence = self._lowstate_sequence
                    if self._lowstate is None:
                        raise RuntimeError("post-warm LowState callback lost its sample")
                    tick = int(self._lowstate.tick)
                    gap = None if previous_tick is None else tick - previous_tick
                    if gap is not None and 0 < gap <= self._control_clock.max_source_gap_ms:
                        consecutive += 1
                    else:
                        consecutive = 1
                    previous_tick = tick
                    if consecutive >= required_samples:
                        return self._lowstate
                remaining = deadline - self._wall_clock()
                if remaining <= 0.0:
                    break
                self._control_condition.wait(timeout=min(0.10, remaining))
        if self._fault_latched.is_set():
            raise RuntimeError(self._terminal_fault_reason or "startup fault latched")
        if self._stop.is_set():
            raise RuntimeError("SONIC stopped while draining post-warm LowState")
        raise RuntimeError(
            "timed out waiting for a fresh post-warm MuJoCo LowState window"
        )

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
            if (
                gc_state["python_gc_freeze_applied"]
                and gc_state["python_gc_automatic_disabled"]
                and not gc_state["python_gc_enabled"]
            ):
                logger.info(
                    "SONIC warm runtime frozen after collecting %d objects; "
                    "automatic cyclic GC is disabled for deterministic control",
                    gc_state["python_gc_warmup_collected_objects"],
                )
            else:
                raise RuntimeError(
                    "SONIC deterministic cyclic-GC setup failed: "
                    + str(gc_state["python_gc_freeze_error"] or "incomplete state")
                )
        if self._fault_latched.is_set():
            raise RuntimeError(self._terminal_fault_reason or "startup fault latched")
        self._wait_for_post_warm_lowstate_stream()
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
                and not (
                    self._fault_latched.is_set()
                    and not getattr(self, "_recoverable_motion_fault", False)
                )
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
                self._control_condition.notify_all()
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
                and not (
                    self._fault_latched.is_set()
                    and not getattr(self, "_recoverable_motion_fault", False)
                )
                and not self._stop.is_set()
            ):
                self._control_condition.wait(timeout=wait_timeout_s)
            sample = self._pending_control_sample
            self._pending_control_sample = None
            if sample is not None:
                self._control_condition.notify_all()
            if (
                sample is None
                and self._control_clock_armed
                and not (
                    self._fault_latched.is_set()
                    and not getattr(self, "_recoverable_motion_fault", False)
                )
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
                or self._pending_control_sample is not None
            ):
                # Qualification may transition to runtime only at a clean
                # producer/consumer boundary.  A newer stationary sample can
                # legitimately arrive while the measured hold for this sample
                # is being published.  If it is already pending, completing
                # qualification here would reclassify that queued startup
                # sample as a runtime policy step; one further callback could
                # then report a false runtime overrun before the consumer gets
                # scheduled.  Keep the latest sample, discard all accumulated
                # qualification credit, and require a fresh clean window.
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

        self._update_telemetry_hot(
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

    def _stamp_sim_control_identity(
        self,
        control_tick_ms: int | None = None,
    ) -> None:
        """Stamp one LowCmd with its exact Plant tick or async stop identity."""

        tick = control_tick_ms
        if tick is None:
            context = getattr(self, "_control_evidence_context", None)
            step = getattr(context, "control_step", None)
            if isinstance(step, PlantControlStep):
                tick = step.tick_ms
        reserve = getattr(self._msg, "reserve", None)
        if not isinstance(reserve, list) or len(reserve) < 2:
            raise RuntimeError("Unitree LowCmd lacks the simulator correlation reserve")
        if tick is None:
            reserve[0] = _SIM_ASYNC_BOUNDARY_MAGIC
            reserve[1] = 0
            return
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("Plant control correlation tick must be non-negative")
        reserve[0] = _SIM_CONTROL_TICK_MAGIC
        reserve[1] = tick

    def _publish_position_hold(
        self,
        targets: tuple[float, ...],
    ) -> None:
        with self._arm_evidence_boundary_lock:
            self._publish_position_hold_unfenced(targets)

    def _publish_position_hold_unfenced(
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
            self._stamp_sim_control_identity()
            self._msg.crc = self._crc.Crc(self._msg)
            # Final pre-Write linearization.  A fault may be claimed while the
            # 29-axis command and CRC are being prepared after the earlier
            # support snapshot.  If it already won, replace every q with the
            # exact executed hold and recompute CRC.  A claim after this short
            # lock is treated by the post-Write transaction below; never hold
            # the terminal lock across Unitree's unbounded Write call.
            with self._terminal_fault_lock:
                terminal_before_write = self._fault_latched.is_set()
                process_stop_before_write = (
                    self._process_stop_fences_motion_locked()
                )
                if terminal_before_write or process_stop_before_write:
                    if terminal_before_write:
                        self._enforce_terminal_boundary_locked()
                    else:
                        self._freeze_last_published_hold(
                            publish_telemetry=False
                        )
                        self._consume_vendor_revoke_locked()
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
            evidence_write_token = self._begin_arm_evidence_write()
            try:
                written = self._publisher.Write(self._msg)
            except BaseException:
                self._finish_arm_evidence_write(evidence_write_token)
                raise
            if written is not True:
                self._finish_arm_evidence_write(evidence_write_token)
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
                process_stop_after_write = (
                    self._process_stop_fences_motion_locked()
                )
                stopping_after_write = (
                    terminal_after_write or process_stop_after_write
                )
                if stopping_after_write:
                    final_targets = self._record_successful_dds_target(
                        selected,
                        evidence_write_token=evidence_write_token,
                    )
                    self._safe_hold_targets = final_targets
                    try:
                        if terminal_after_write:
                            self._enforce_terminal_boundary_locked()
                        else:
                            self._freeze_last_published_hold(
                                publish_telemetry=False
                            )
                            self._consume_vendor_revoke_locked()
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
                    self._commit_executed_target_if_pending(
                        selected,
                        evidence_write_token=evidence_write_token,
                    )
                if terminal_error is not None:
                    raise terminal_error
            assert self._last_published_targets is not None
            self._arm_guard.anchor(self._last_published_targets[15:29])
        self._mark_control_step_published()

    def _commit_executed_target_if_pending(
        self,
        targets: tuple[float, ...],
        *,
        evidence_write_token: object | None = None,
        arm_guard_result: JointGuardResult | None = None,
    ) -> bool:
        """Commit the exact target immediately after one successful DDS write.

        Startup/ordinary holds have no policy candidate and are an intentional
        feedback no-op.  A stop racing inference still commits the frozen target
        that actually crossed DDS, while revoked vendor authority prevents its
        reference cursor from advancing.
        """

        final_targets = self._record_successful_dds_target(
            targets,
            evidence_write_token=evidence_write_token,
            arm_guard_result=arm_guard_result,
        )
        return self._record_vendor_execution_feedback_if_pending(final_targets)

    def _begin_arm_evidence_write(self) -> object:
        """Fence an in-flight DDS write against evidence begin/finalize."""

        token = object()
        with self._arm_evidence_write_condition:
            self._arm_evidence_writes_inflight.add(token)
        return token

    def _finish_arm_evidence_write(self, token: object) -> None:
        with self._arm_evidence_write_condition:
            if token not in self._arm_evidence_writes_inflight:
                raise RuntimeError("arm evidence DDS write token is not active")
            self._arm_evidence_writes_inflight.remove(token)
            self._arm_evidence_write_condition.notify_all()

    def _wait_for_arm_evidence_boundary(self, *, timeout_s: float = 1.0) -> None:
        deadline = self._wall_clock() + float(timeout_s)
        with self._arm_evidence_write_condition:
            while self._arm_evidence_writes_inflight:
                remaining = deadline - self._wall_clock()
                if remaining <= 0.0:
                    raise RuntimeError(
                        "timed out waiting for the authoritative DDS evidence boundary"
                    )
                self._arm_evidence_write_condition.wait(timeout=remaining)

    def _record_successful_dds_target(
        self,
        targets: tuple[float, ...],
        *,
        evidence_write_token: object | None = None,
        arm_guard_result: JointGuardResult | None = None,
    ) -> tuple[float, ...]:
        """Register the target that physically crossed a successful DDS Write."""

        final_targets = tuple(float(value) for value in targets)
        if len(final_targets) != 29 or not all(
            math.isfinite(value) for value in final_targets
        ):
            raise ValueError("executed DDS target requires 29 finite values")
        self._last_published_targets = final_targets
        self._dds_target_sequence += 1
        try:
            if self._arm_evidence.active:
                if evidence_write_token is None:
                    self._arm_evidence.fail_active(
                        "successful DDS write was not fenced for arm evidence"
                    )
                control_step = getattr(
                    self._control_evidence_context,
                    "control_step",
                    None,
                )
                if not isinstance(control_step, PlantControlStep):
                    self._arm_evidence.fail_active(
                        "successful DDS write has no authoritative Plant control step"
                    )
                if arm_guard_result is None:
                    constrained_global: tuple[int, ...] = ()
                    saturation_by_joint = {
                        joint: 0.0 for joint in ARM_JOINT_INDICES
                    }
                else:
                    if len(arm_guard_result.continuous_saturation_s) != len(
                        ARM_JOINT_INDICES
                    ):
                        self._arm_evidence.fail_active(
                            "arm limiter result has an invalid joint dimension"
                        )
                    constrained_global = tuple(
                        ARM_JOINT_INDICES[index]
                        for index in arm_guard_result.constrained_joint_indices
                    )
                    saturation_by_joint = dict(
                        zip(
                            ARM_JOINT_INDICES,
                            arm_guard_result.continuous_saturation_s,
                            strict=True,
                        )
                    )
                self._arm_evidence.record(
                    control_tick=self._dds_target_sequence,
                    t_s=control_step.tick_ms / 1_000.0,
                    dt_s=control_step.dt_s,
                    constrained_joint_indices=constrained_global,
                    continuous_saturation_by_joint_s=saturation_by_joint,
                )
            return final_targets
        finally:
            if evidence_write_token is not None:
                self._finish_arm_evidence_write(evidence_write_token)

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

    def _arm_control_stall_diagnostic(self) -> None:
        """Enable one pre-fault stack capture for the current motion task."""

        diagnostic = getattr(self, "_control_stall_diagnostic_dumped", None)
        if diagnostic is not None:
            diagnostic.clear()
        armed = getattr(self, "_control_stall_diagnostic_armed", None)
        if armed is not None:
            armed.set()

    def _disarm_control_stall_diagnostic(self) -> None:
        armed = getattr(self, "_control_stall_diagnostic_armed", None)
        if armed is not None:
            armed.clear()

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
            self._controlled_stop_active = False
            self._balanced_policy_idle_active = False
            self._controlled_stop_source_identity = None
            self._controlled_stop_reference_installed = False
            self._normal_stop_policy_braking = False
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
            activation = getattr(self, "_policy_activation", None)
            abort_activation = getattr(activation, "abort", None)
            if callable(abort_activation):
                try:
                    abort_activation()
                except Exception as exc:
                    failures.append(exc)
            try:
                self._reset_policy_activation_release_locked()
            except Exception as exc:
                failures.append(exc)
            self._policy_activation_requested = False
            self._disarm_control_stall_diagnostic()
            # Even terminal/fault paths suppressing broader planner telemetry
            # must publish this local transaction fact; otherwise diagnostics
            # can claim PREPARING after the activation epoch was invalidated.
            try:
                self.telemetry.update(
                    balanced_policy_idle_active=False,
                    **self._policy_activation_telemetry_fields_locked()
                )
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

    def _begin_controlled_policy_stop_locked(
        self,
        decision: IngressDecision,
        *,
        correlated: bool = False,
    ) -> bool:
        """Rebase IDLE at the executed boundary while rejecting new motion.

        STOPPING is recoverable deceleration, not a terminal safety event.
        Freeze the exact last successful DDS target, withdraw only the outer
        actuator authority, and request a heading-neutral IDLE reference while
        preserving SONIC's acknowledged recurrent history and reference
        cursor.  Rebuilding history from one measured sample worked for a
        forward stop but destabilized the live reverse stop.  The regular gait
        channel owns this exact source identity; a later explicit normal-stop
        transaction may join it, but action completion does not depend on a
        second RPC.

        Fault, process-stop, support-latch, startup, and pre-grant paths still
        fall back to the stronger static boundary.  The caller owns
        ``_support_lock``.
        """

        if (
            not isinstance(correlated, bool)
            or decision.action != "brake"
            or decision.state != "STOPPING"
            or decision.generation is None
            or decision.sequence is None
        ):
            return False
        if not self._controlled_policy_stop_eligible_locked(decision.generation):
            return False
        identity = (decision.generation, decision.sequence)
        already_active = bool(getattr(self, "_controlled_stop_active", False))
        authority = self._reference_authority.snapshot
        if (
            already_active
            and correlated
            and authority.phase in {"MATCHED_INSTALLED", "PRIMED", "GRANTED"}
            and (
                authority.generation != identity[0]
                or authority.source_sequence != identity[1]
            )
        ):
            # The regular gait channel can finish installing and granting its
            # continuous IDLE reference just before the correlated normal-stop
            # request arrives with the next source sequence. Do not rewrite an
            # executable reference, but do not revoke learned balance either.
            # The normal-stop wire identity remains the newer sequence while
            # this exact controller identity keeps DDS until Plant confirms its
            # safety-stop latch.
            active_identity = getattr(
                self, "_controlled_stop_source_identity", None
            )
            if (
                not isinstance(active_identity, tuple)
                or len(active_identity) != 2
                or active_identity[0] != identity[0]
                or active_identity[1] >= identity[1]
                or authority.generation != active_identity[0]
                or authority.source_sequence != active_identity[1]
                or not self._controlled_stop_reference_installed
            ):
                return False
            self._balanced_policy_idle_active = False
            self._navigation_task_active = False
            return True
        if not already_active:
            self.guard.stop_at_measured_heading(
                source="gait_adapter_controlled_stop"
            )
            if not self._freeze_planner_handoff_boundary():
                return False
            failures = self._withdraw_external_planner_authority_locked(
                "controlled stop requires a continuous IDLE reference"
            )
            if failures:
                raise failures[0]
            self._static_hold_active = True
            self._controlled_stop_active = True
            self._balanced_policy_idle_active = False
            self._controlled_stop_source_identity = identity
            self._controlled_stop_reference_installed = False
            self._reference_authority.require(
                generation=identity[0],
                source_sequence=identity[1],
            )
            self._reference_source_identity = identity
            self._reference_request_pending = True
            self._reference_request_continuous = True
        elif correlated:
            # A regular STOPPING envelope may already have enqueued the
            # continuous IDLE planner request. The later correlated
            # normal-stop envelope takes ownership of the source identity,
            # but it must not turn an in-flight request back into a pending
            # enqueue.  Doing so creates two requests with the same upstream
            # ``gen_frame`` while the reference cursor is frozen; strict
            # planner evidence then (correctly) rejects both as ambiguous and
            # the otherwise healthy stop eventually times out.
            request_still_needs_enqueue = self._reference_request_pending
            authority = self._reference_authority.snapshot
            if authority.phase in {"MATCHED_INSTALLED", "PRIMED", "GRANTED"}:
                self._controlled_stop_reference_installed = True
            else:
                self._reference_authority.require(
                    generation=identity[0],
                    source_sequence=identity[1],
                )
                self._controlled_stop_reference_installed = False
            self._controlled_stop_source_identity = identity
            self._reference_source_identity = identity
            self._reference_request_pending = request_still_needs_enqueue
        self._navigation_task_active = False
        return True

    def _complete_balanced_policy_idle_locked(
        self,
        decision: IngressDecision,
    ) -> bool:
        """Commit a measured stationary IDLE without revoking balance.

        GaitAdapter's acknowledged ``IDLE`` proves that the base stopped.  It
        does not prove that an untethered humanoid can remain upright after
        dynamic control is removed.  Keep the freshly rebased IDLE reference
        on DDS and retain Plant ``idle`` support; only the explicit correlated
        normal-stop/fault/shutdown paths may freeze and revoke it.

        The caller owns ``_support_lock`` and has already admitted the terminal
        gait envelope through :class:`SonicIntentIngress`.
        """

        identity = getattr(self, "_controlled_stop_source_identity", None)
        authority = self._reference_authority.snapshot
        controller = getattr(self, "_controller", None)
        activation = getattr(self, "_policy_activation", None)
        valid = bool(
            decision.action == "stop"
            and decision.state == "IDLE"
            and decision.generation is not None
            and decision.sequence is not None
            and identity is not None
            and decision.generation == identity[0]
            and decision.sequence > identity[1]
            and getattr(self, "_controlled_stop_active", False)
            and getattr(self, "_controlled_stop_reference_installed", False)
            # ``armed`` is the durable result of an acknowledged stationary
            # IDLE envelope.  ``arm_acknowledged`` is only its one-envelope
            # rising edge.  The edge may arrive while the async IDLE reference
            # is still PRIMED; requiring it here makes the next envelope
            # impossible to commit and used to collapse an untethered robot
            # into a static joint hold after one policy frame.
            and self._intent_ingress.armed
            and not self._intent_ingress.braking
            and not self._navigation_task_active
            and self.guard.target_zero
            and self.guard.command_zero
            and authority.phase == "GRANTED"
            and authority.actuator_authority
            and authority.generation == identity[0]
            and authority.source_sequence == identity[1]
            and self._reference_source_identity == identity
            and self._reference_request_pending is False
            and controller is not None
            and int(getattr(controller.ms, "mode", -1)) == 0
            and not bool(getattr(controller.ms, "has_movement", True))
            and activation is not None
            and bool(getattr(activation, "prepared", False))
            and self._safe_hold_targets is None
            and self._static_hold_active is False
            and self._plant_support_mode == "idle"
            and not self._plant_support_stop_pending
            and not self._plant_support_safety_latched
            and not self._normal_stop_fences_motion_locked()
            and not self._process_stop_fences_motion_locked()
            and not self._fault_latched.is_set()
        )
        if not valid:
            return False
        if decision.heading_yaw is not None:
            self.guard.observe_heading(decision.heading_yaw)
        self._balanced_policy_idle_active = True
        self._normal_stop_policy_braking = False
        return True

    def _balanced_policy_idle_completion_pending_locked(
        self,
        decision: IngressDecision,
    ) -> bool:
        """Recognize a valid stationary acknowledgement awaiting DDS grant.

        GaitAdapter stationarity and SONIC planner/reference readiness are
        independent asynchronous facts.  A valid IDLE acknowledgement can
        therefore overtake the first controlled-policy DDS write.  This state
        is recoverable and remains bounded by the existing planner deadline,
        control-step watchdog, process-stop, normal-stop, and terminal-fault
        fences.  It must not be converted to a static joint hold while the
        Plant fall-arrest support is released.

        The caller owns ``_support_lock``.
        """

        identity = getattr(self, "_controlled_stop_source_identity", None)
        authority = self._reference_authority.snapshot
        controller = getattr(self, "_controller", None)
        activation = getattr(self, "_policy_activation", None)
        return bool(
            decision.action == "stop"
            and decision.state == "IDLE"
            and decision.generation is not None
            and decision.sequence is not None
            and identity is not None
            and decision.generation == identity[0]
            and decision.sequence > identity[1]
            and getattr(self, "_controlled_stop_active", False)
            and self._intent_ingress.armed
            and not self._intent_ingress.braking
            and not self._navigation_task_active
            and self.guard.target_zero
            and self.guard.command_zero
            # GRANTED is not transitional: if the full completion predicate
            # still fails there, retain the stronger terminal fallback rather
            # than masking an inconsistent executable authority state.
            and authority.phase in {"REQUIRED", "MATCHED_INSTALLED", "PRIMED"}
            and authority.generation == identity[0]
            and authority.source_sequence == identity[1]
            and self._reference_source_identity == identity
            and controller is not None
            and bool(getattr(controller, "initialized", False))
            and int(getattr(controller.ms, "mode", -1)) == 0
            and not bool(getattr(controller.ms, "has_movement", True))
            and activation is not None
            and bool(getattr(activation, "prepared", False))
            and self._safe_hold_targets is None
            and self._plant_support_mode == "idle"
            and not self._plant_support_stop_pending
            and not self._plant_support_safety_latched
            and not self._normal_stop_fences_motion_locked()
            and not self._process_stop_fences_motion_locked()
            and not self._fault_latched.is_set()
        )

    def _controlled_policy_stop_eligible_locked(self, generation: int) -> bool:
        """Check whether STOPPING may rebase the current learned controller."""

        authority = self._reference_authority.snapshot
        activation = getattr(self, "_policy_activation", None)
        controller = getattr(self, "_controller", None)
        common_blocked = (
            authority.generation != generation
            or self._last_published_targets is None
            or activation is None
            or not bool(getattr(activation, "prepared", False))
            or controller is None
            or not bool(getattr(controller, "initialized", False))
            or not bool(
                self._intent_ingress.armed or self._intent_ingress.braking
            )
            or self._fault_latched.is_set()
            or self._process_stop_fences_motion_locked()
            or self._plant_support_stop_pending
            or self._plant_support_safety_latched
        )
        if common_blocked:
            return False
        if getattr(self, "_controlled_stop_active", False):
            identity = getattr(self, "_controlled_stop_source_identity", None)
            return bool(
                identity is not None
                and identity[0] == generation
                and authority.phase in {
                    "REVOKED",
                    "REQUIRED",
                    "MATCHED_INSTALLED",
                    "PRIMED",
                    "GRANTED",
                }
            )
        return authority.phase == "GRANTED"

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
            recoverable = bool(
                getattr(self, "_recoverable_motion_fault", False)
            )
            if (
                (self._fault_latched.is_set() and not recoverable)
                or getattr(self, "_terminal_plant_halt_attempted", False)
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
            self._recoverable_motion_fault = False
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

    def _latch_motion_fault(
        self,
        reason: str,
        *,
        boundary_already_stopped: bool = False,
    ) -> None:
        """Latch a recoverable navigation fault while keeping SONIC alive."""

        detail = str(reason).strip() or "SONIC motion safety fault"
        with self._terminal_fault_lock:
            if self._fault_latched.is_set():
                if (
                    self._recoverable_motion_fault
                    and self._terminal_fault_reason == detail
                ):
                    return
                raise RuntimeError("SONIC already carries another motion fault")
            self._terminal_fault_reason = detail
            self._recoverable_motion_fault = True
            self._motion_fault_safe_hold_frames = 0
            self._motion_fault_last_hold_tick_ms = None
            self._fault_latched.set()
        self._startup_qualifying = False
        motion_ready = getattr(self, "_motion_ready", None)
        if motion_ready is not None:
            motion_ready.clear()
        # Freeze the acceptance recorder at the original safety fault before
        # any context-free recovery hold can cross DDS.  The evidence boundary
        # waits for an already-started control write to finish, while the fault
        # latch above prevents a newer policy write from acquiring authority.
        evidence_boundary = getattr(self, "_arm_evidence_boundary_lock", None)
        arm_evidence = getattr(self, "_arm_evidence", None)
        close_failed_evidence = getattr(
            arm_evidence, "close_failed_active", None
        )
        if evidence_boundary is not None and callable(close_failed_evidence):
            with evidence_boundary:
                close_failed_evidence(detail)
        boundary_error: Exception | None = None
        if not boundary_already_stopped:
            try:
                self._stop_navigation_at_boundary(
                    source="motion_fault",
                    publish_telemetry=False,
                )
            except Exception as exc:
                boundary_error = exc
        try:
            plant_state = latch_plant_motion_fault(
                "127.0.0.1",
                self.config.plant_port,
                detail[:512],
                timeout=0.30,
            )
        except Exception as exc:
            # Losing the physical safety latch upgrades the incident to the
            # existing unrecoverable process/physics fault path.
            self._latch_terminal_fault(
                f"{detail}; recoverable Plant motion-fault latch failed: {exc}",
                boundary_already_stopped=(boundary_error is None),
            )
            return
        with self._support_lock:
            self._plant_support_mode = "safety_stop"
            self._plant_support_safety_latched = True
            self._plant_support_stop_pending = False
        trace_path: str | None = None
        trace_samples = 0
        trace = getattr(self, "_fault_trace", None)
        if trace is not None:
            try:
                payload = trace.snapshot(
                    reason=detail,
                    kp=tuple(float(value) for value in self._kp),
                    kd=tuple(float(value) for value in self._kd),
                )
                trace_samples = int(payload["sample_count"])
                trace_path = str(trace.persist(self._fault_trace_path, payload))
            except Exception:
                logger.exception("failed to persist recoverable motion-fault trace")
        self.telemetry.update(
            status="failed",
            nav2_state="fault_latched",
            sonic_mode="IDLE",
            sonic_speed_mps=0.0,
            static_hold_active=True,
            motion_handoff_active=False,
            motion_handoff_ratio=0.0,
            navigation_task_active=False,
            motion_ready=False,
            control_clock_status="motion_fault_hold",
            last_error=detail,
            terminal_boundary_error=(
                None if boundary_error is None else str(boundary_error)
            ),
            plant_terminal_halt_confirmed=False,
            plant_terminal_halt_error=None,
            plant_support_mode=str(plant_state["mode"]),
            plant_support_safety_latched=True,
            fault_trace_path=trace_path,
            fault_trace_sample_count=trace_samples,
            fault_joint_names=(
                [] if trace is None else list(trace.joint_names)
            ),
            recoverable_motion_fault=True,
            motion_fault_safe_hold_frames=0,
        )
        condition = getattr(self, "_control_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()

    def _record_motion_fault_hold_frame(self, tick_ms: int) -> None:
        """Count coherent post-fault DDS holds for explicit-reset admission."""

        with self._terminal_fault_lock:
            if not (
                self._fault_latched.is_set()
                and self._recoverable_motion_fault
            ):
                return
            previous = self._motion_fault_last_hold_tick_ms
            control_hz = int(getattr(self.config, "control_hz", 50))
            expected_tick_delta_ms = round(1_000.0 / control_hz)
            if (
                previous is not None
                and tick_ms - previous != expected_tick_delta_ms
            ):
                self._motion_fault_safe_hold_frames = 0
            self._motion_fault_last_hold_tick_ms = int(tick_ms)
            self._motion_fault_safe_hold_frames += 1
            frames = self._motion_fault_safe_hold_frames
        self.telemetry.update(motion_fault_safe_hold_frames=frames)

    def _handle_motion_fault_reset(
        self,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Clear one recoverable fault without releasing actuator authority."""

        if set(payload) != {"type", "request_id", "reason"}:
            raise ValueError("motion fault reset fields do not match v1")
        request_id = payload.get("request_id")
        try:
            parsed_request_id = UUID(str(request_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("motion fault reset request_id must be a UUID") from exc
        if str(parsed_request_id) != str(request_id):
            raise ValueError("motion fault reset request_id must be canonical")
        reason = str(payload.get("reason", "")).strip()
        replays = getattr(self, "_motion_fault_reset_replays", None)
        if replays is None:
            replays = {}
            self._motion_fault_reset_replays = replays
        replay = replays.get(parsed_request_id)
        if replay is not None:
            previous_reason, previous_response = replay
            if previous_reason != reason:
                raise RuntimeError(
                    "motion fault reset request_id was reused with another reason"
                )
            return copy.deepcopy(previous_response)
        with self._terminal_fault_lock:
            if not self._fault_latched.is_set():
                raise RuntimeError("SONIC has no latched motion fault")
            if not self._recoverable_motion_fault:
                raise RuntimeError(
                    "unrecoverable SONIC fault requires an explicit process restart"
                )
            if reason != self._terminal_fault_reason:
                raise RuntimeError("SONIC motion fault identity mismatch")
            safe_hold_frames = self._motion_fault_safe_hold_frames
        with self._control_condition:
            lowstate = self._lowstate
        if lowstate is None:
            raise RuntimeError("motion fault reset requires current LowState")
        joint_q = tuple(float(lowstate.motor_state[index].q) for index in range(29))
        joint_dq = tuple(float(lowstate.motor_state[index].dq) for index in range(29))
        gyroscope = tuple(float(value) for value in lowstate.imu_state.gyroscope)
        rpy_raw = getattr(lowstate.imu_state, "rpy", None)
        if rpy_raw is None or len(rpy_raw) != 3:
            raise RuntimeError("motion fault reset requires Plant roll/pitch evidence")
        rpy = tuple(float(value) for value in rpy_raw)
        if not all(
            math.isfinite(value)
            for value in joint_q + joint_dq + gyroscope + rpy
        ):
            raise RuntimeError("motion fault reset requires finite Plant state")
        with self._support_lock:
            authority = self._reference_authority.snapshot
            reset_allowed = bool(
                safe_hold_frames >= 2
                and not self._navigation_task_active
                and not self._intent_ingress.armed
                and self.guard.command_zero
                and authority.phase == "REVOKED"
                and self._safe_hold_targets is not None
                and self._static_hold_active
                and self._plant_support_safety_latched
                and not self._plant_support_stop_pending
                and max(abs(value) for value in joint_dq) <= 0.20
                and max(abs(value) for value in gyroscope) <= 0.15
                and max(abs(value) for value in rpy[:2]) <= 0.15
            )
        if not reset_allowed:
            raise RuntimeError(
                "motion fault reset requires two coherent safe-hold frames, "
                "zero command, revoked authority, and finite stationary Plant"
            )
        clear_plant_motion_fault(
            "127.0.0.1",
            self.config.plant_port,
            reason,
            timeout=0.30,
        )
        with self._terminal_fault_lock:
            if (
                not self._fault_latched.is_set()
                or not self._recoverable_motion_fault
                or self._terminal_fault_reason != reason
            ):
                raise RuntimeError("SONIC motion fault changed during reset")
            self._fault_latched.clear()
            self._recoverable_motion_fault = False
            self._terminal_fault_reason = None
            self._motion_fault_safe_hold_frames = 0
            self._motion_fault_last_hold_tick_ms = None
        self._motion_ready.set()
        self.telemetry.update(
            status="running",
            nav2_state="safety_stop",
            control_clock_status="simulation_time_50hz_motion_ready",
            motion_ready=True,
            last_error=None,
            recoverable_motion_fault=False,
            motion_fault_safe_hold_frames=0,
        )
        response = {
            "status": "reset",
            "mode": "safety_stop",
            "authority": "REVOKED",
            "safe_hold_frames": safe_hold_frames,
        }
        replays[parsed_request_id] = (
            reason,
            copy.deepcopy(response),
        )
        return response

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
        normal_stop = getattr(self, "_normal_stop", None)
        if not isinstance(normal_stop, NormalStopState):
            raise RuntimeError(
                f"{label}: correlated normal-stop confirmation is unavailable"
            )
        normal_snapshot = normal_stop.snapshot
        normal_identity = normal_snapshot.identity
        plant_epoch = self._normal_stop_plant_safety_stop_epoch
        if (
            normal_snapshot.phase != "DRIVER_CONFIRMED"
            or normal_identity is None
            or plant_epoch is None
            or plant_epoch != self._normal_stop_last_confirmed_plant_epoch
            or self._plant_support_stop_generation
            != normal_identity.control_generation
            or self._intent_ingress.support_stop_generation
            != normal_identity.control_generation
        ):
            raise RuntimeError(
                f"{label}: control acquisition requires the current exact "
                "DRIVER_CONFIRMED normal stop"
            )
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

    def _publish_committed_plant_rearm(self, generation: int) -> bool:
        """Project a rearm only while its exact committed state is current.

        The Plant acknowledgement and authority commit precede this display
        update.  A stronger normal/terminal stop may therefore win before this
        method runs.  Serialize and revalidate the complete rearm identity so
        that an older publisher cannot erase the newer stop's arm anchor or
        overwrite its SAFE_HOLD telemetry with ``idle``.
        """

        with self._support_lock:
            # Match the support -> terminal order used by normal-stop commits.
            # Terminal fault claims release their own lock before reaching for
            # support, so this nests safely while making the predicate, arm
            # reset, and IDLE projection one terminal-linearized operation.
            with self._terminal_fault_lock:
                if (
                    self._fault_latched.is_set()
                    or self._stop.is_set()
                    or self._intent_ingress.generation != generation
                    or not self._intent_ingress.armed
                    or (
                        self._intent_ingress.staged_control_arm_generation
                        is not None
                    )
                    or self._intent_ingress.support_stop_generation is not None
                    or self._plant_support_mode != "idle"
                    or self._plant_support_stop_pending
                    or self._plant_support_safety_latched
                    or self._plant_support_stop_generation is not None
                ):
                    return False
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

        self._update_telemetry_hot(plant_support_requested_mode=mode)
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

    def _update_telemetry_hot(self, **changes: Any) -> None:
        """Publish control-rate telemetry without cloning its full state.

        Production always supplies :class:`TelemetryStore`.  The fallback
        preserves the deliberately tiny telemetry doubles used by boundary
        tests without putting their compatibility branch on the production
        hot path.
        """

        update_hot = getattr(self.telemetry, "update_hot", None)
        if callable(update_hot):
            update_hot(**changes)
            return
        self.telemetry.update(**changes)

    def _reset_navigation_task_peaks_hot(self) -> None:
        reset_hot = getattr(
            self.telemetry,
            "reset_navigation_task_peaks_hot",
            None,
        )
        if callable(reset_hot):
            reset_hot()
            return
        self.telemetry.reset_navigation_task_peaks()

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
        pipeline = getattr(controller, "controller", None)
        readapt_state = getattr(pipeline, "idle_readapt_state", None)
        readapt_state_name = getattr(readapt_state, "name", None)
        fields.update(
            sonic_idle_readapt_state=(
                readapt_state_name
                if isinstance(readapt_state_name, str)
                else "IDLE"
            ),
            sonic_idle_readapt_stored=bool(
                getattr(pipeline, "idle_readapt_stored", False)
            ),
            sonic_idle_readapt_average_error_rad=float(
                getattr(pipeline, "idle_readapt_average_error_rad", 0.0)
            ),
        )
        fields["sonic_execution_feedback_pending"] = bool(
            getattr(controller, "execution_feedback_pending", False)
        )
        return fields

    def _revoke_planner_reference(self, reason: str) -> None:
        with self._support_lock:
            self._revoke_planner_reference_locked(reason)
        self.telemetry.update(**self._planner_telemetry_fields())

    def _withdraw_external_planner_authority_locked(
        self,
        reason: str,
    ) -> list[Exception]:
        """Withdraw driver/evidence authority without resetting vendor history.

        Controlled braking needs the vendor controller's acknowledged cursor
        and recurrent histories until its dedicated continuous request has
        captured them.  Terminal stops and ordinary measured rebases use the
        stronger ``_revoke_planner_reference_locked`` operation below.
        """

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
        return failures

    def _revoke_planner_reference_locked(self, reason: str) -> None:
        failures = self._withdraw_external_planner_authority_locked(reason)
        controller = getattr(self, "_controller", None)
        revoke = getattr(controller, "revoke_reference_authority", None)
        if callable(revoke):
            # A command/terminal thread may win while ONNX is decoding. Never
            # enter vendor state from that thread: the 50 Hz owner consumes
            # this request after inference and before any feedback or publish.
            self._vendor_revoke_pending = True
        self._reference_request_pending = False
        self._reference_request_continuous = False
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
        started_at = self._wall_clock()
        with self._support_lock:
            intent = self._apply_intent_locked(
                dt,
                simulation_now_s=simulation_now_s,
            )
        elapsed_s = max(0.0, self._wall_clock() - started_at)
        if elapsed_s > 0.05:
            logger.warning(
                "slow SONIC control stage stage=intent_apply elapsed_s=%.6f mode=%s",
                elapsed_s,
                getattr(intent, "mode", "unknown"),
            )
        return intent

    def _apply_intent_locked(self, dt: float, *, simulation_now_s: float):
        intent = self.guard.intent(dt=dt)
        controller = self._controller
        decoder = controller.controller
        controlled_stop = bool(
            getattr(self, "_controlled_stop_active", False)
            and intent.mode == "IDLE"
        )
        if controlled_stop:
            # Once the correlated STOPPING transaction owns deceleration, the
            # zero command is driver-owned rather than a live Nav2 command.
            # CmdVelGuard will eventually mark that same zero stale and change
            # its source to ``watchdog``.  Those freshness fields do not alter
            # the IDLE planner request, and treating them as a new intent makes
            # a second planner generation reuse the same gait source sequence.
            # Keep one canonical planner identity for the entire controlled
            # stop; cmd_vel age remains independently visible in telemetry.
            intent = replace(
                intent,
                stale=False,
                source="gait_adapter_controlled_stop",
            )
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
        starting_motion = intent.mode != "IDLE" and (
            previous is None or previous.mode == "IDLE"
        )
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
        requires_actuator_boundary = admitted and not controlled_stop and (
            intent.mode != "IDLE"
            or (previous is not None and previous.mode != "IDLE")
        )
        if requires_actuator_boundary:
            if not self._freeze_planner_handoff_boundary():
                raise RuntimeError(
                    "intent transition has no published actuator boundary"
                )
        activation = getattr(self, "_policy_activation", None)
        activation_prepared = bool(
            activation is None or getattr(activation, "prepared", False)
        )
        # A pure-rotation ramp is deliberately held below the planner's
        # verified 0.05-rad request floor.  That raw, not-yet-admitted intent
        # does not own a reference transaction and therefore must not divert
        # the 50 Hz loop into nominal activation.  Doing so used to make the
        # activation precheck return (authority was still REVOKED) while the
        # caller skipped the normal static DDS hold, creating a deterministic
        # 250+ ms control-publish gap before every Spin.
        if starting_motion and admitted and not activation_prepared:
            self._policy_activation_requested = True
        defer_planner_for_activation = bool(
            intent.mode != "IDLE"
            and getattr(self, "_policy_activation_requested", False)
            and not activation_prepared
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
                self._reference_request_pending = not defer_planner_for_activation
        elif admitted and intent.mode == "IDLE":
            if controlled_stop:
                authority = self._reference_authority.snapshot
                controlled_identity = self._controlled_stop_source_identity
                if controlled_identity is None:
                    raise RuntimeError(
                        "controlled stop has no correlated SONIC identity"
                    )
                if authority.phase != "REVOKED" and (
                    authority.generation != controlled_identity[0]
                    or authority.source_sequence != controlled_identity[1]
                ):
                    raise RuntimeError(
                        "controlled-stop IDLE reference identity diverged"
                    )
            else:
                self._revoke_planner_reference("Nav2 intent is IDLE")
                abort_activation = getattr(activation, "abort", None)
                if callable(abort_activation):
                    abort_activation()
                self._policy_activation_requested = False
        if not defer_planner_for_activation:
            self._planner_evidence.observe_intent(
                intent,
                admitted=admitted,
                simulation_now_s=simulation_now_s,
                allow_static_reference=controlled_stop,
            )
        if intent.mode == "IDLE":
            if not controlled_stop and (previous is None or previous.mode != "IDLE"):
                # Re-anchor SONIC's decoder to the measured IMU heading while
                # the actuator side is held static.  The next Nav2 action then
                # starts from the robot's real orientation rather than a stale
                # decoder reference left by a previous task.
                decoder.reinit_heading = True
                decoder.delta_heading = 0.0
        elif starting_motion and admitted:
            decoder.reinit_heading = True
            decoder.delta_heading = 0.0
            # The pinned NVIDIA ROS2 input path expresses navigation heading
            # solely through MovementState.facing_direction.  ``delta_heading``
            # belongs to the keyboard/ZMQ operator path.  Applying the same
            # Nav2 angle to both planner facing and decoder heading doubled the
            # first short Spin request and made later reference transitions
            # inconsistent with the planner's root-yaw evidence.
            self._pending_decoder_heading_delta_rad = 0.0
            self._decoder_facing_rad = intent.facing_angle_rad
        elif admitted and self._decoder_facing_rad is not None:
            # Planner-facing updates are reference transactions; do not also
            # inject them through the operator-only decoder offset.
            self._pending_decoder_heading_delta_rad = 0.0
            self._decoder_facing_rad = intent.facing_angle_rad
        if admitted:
            # Official SONIC ROS2 maps angular-only navigation to an IDLE
            # planner motion whose facing direction changes. Keep the outer
            # intent active for authority/watchdog semantics, but do not ask
            # the planner for a zero-translation SLOW_WALK clip: that clip
            # reaches only a partial turn and then plateaus in the deployment
            # MuJoCo model.
            pure_rotation = intent.mode == "SLOW_WALK" and not intent.has_movement
            controller.ms.mode = (
                LM.IDLE
                if pure_rotation or intent.mode == "IDLE"
                else LM.SLOW_WALK
            )
            controller.ms.speed = intent.speed_mps if intent.has_movement else -1.0
            controller.ms.movement_angle = intent.movement_angle_rad
            controller.ms.facing_angle = intent.facing_angle_rad
            controller.ms.has_movement = intent.has_movement
            # Manual authority requests every fresh reference explicitly on
            # the planning-only path.  In particular, controlled stop now
            # rebases from current LowState instead of letting ``run_step``
            # blend a moving clip into IDLE.
            controller.ms.needs_replan = False
            self._last_intent_signature = (
                intent.has_movement,
                round(intent.movement_angle_rad, 3),
                round(intent.facing_angle_rad, 3),
                round(intent.speed_mps if intent.has_movement else -1.0, 3),
            )
        elif not controlled_stop:
            controller.ms.needs_replan = False
        received_at = self.guard.last_received_at
        self._update_telemetry_hot(
            sonic_mode=intent.mode,
            sonic_speed_mps=intent.speed_mps,
            sonic_movement_angle_rad=intent.movement_angle_rad,
            sonic_facing_angle_rad=intent.facing_angle_rad,
            sonic_command_facing_angle_rad=float(
                getattr(self.guard, "command_facing", intent.facing_angle_rad)
            ),
            sonic_measured_heading_yaw_rad=getattr(
                self.guard, "measured_heading", None
            ),
            sonic_decoder_heading_delta_rad=float(decoder.delta_heading),
            cmd_vel_age_s=(None if received_at is None else max(0.0, time.monotonic() - received_at)),
            **self._planner_telemetry_fields(),
        )
        return intent

    def _release_safe_hold_for_policy_activation_locked(
        self,
        decision: NominalPolicyActivationDecision,
    ) -> bool:
        """Release only the explicitly rearmed boundary for nominal INIT.

        The caller owns ``_support_lock``.  A later stop may re-establish
        ``_safe_hold_targets`` before the DDS write; the publish path then
        substitutes that stronger target and the activation commit fails.
        """

        if not self._policy_activation_boundary_valid_locked(decision=decision):
            return False
        self._safe_hold_targets = None
        return True

    def _policy_activation_boundary_valid_locked(
        self,
        *,
        decision: NominalPolicyActivationDecision | None = None,
    ) -> bool:
        activation = self._policy_activation
        authority = self._reference_authority.snapshot
        identity = self._reference_source_identity
        ingress = self._intent_ingress
        controller = self._controller
        activation_current = (
            activation.is_current(decision)
            if decision is not None
            else activation.phase
            in {
                "SETTLING_TETHERED",
                "WAITING_FOR_SUPPORT_RELEASE",
                "PREPARED",
            }
        )
        return bool(
            activation_current
            and getattr(self, "_policy_activation_requested", False)
            and authority.phase == "REQUIRED"
            and identity is not None
            and authority.generation == identity[0]
            and authority.source_sequence == identity[1]
            and ingress.armed
            and ingress.generation == identity[0]
            and self._navigation_task_active
            and not self._process_stop_fences_motion_locked()
            and not self._fault_latched.is_set()
            and not self._plant_support_stop_pending
            and not self._plant_support_safety_latched
            and not self._normal_stop_fences_motion_locked()
            and not bool(getattr(controller, "execution_feedback_pending", False))
        )

    def _policy_activation_telemetry_fields_locked(self) -> dict[str, object]:
        """Return activation facts while the caller owns ``_support_lock``."""

        activation = getattr(self, "_policy_activation", None)
        if activation is None:
            return {
                "policy_activation_epoch": 0,
                "policy_activation_path": None,
                "policy_activation_phase": "UNPREPARED",
                "policy_activation_ratio": 0.0,
                "policy_activation_provenance": None,
            }
        observation = activation.last_settle_observation
        release_error = getattr(
            self, "_policy_activation_release_error", None
        )
        release_thread = getattr(
            self, "_policy_activation_release_thread", None
        )
        return {
            "policy_activation_epoch": activation.epoch,
            "policy_activation_path": activation.provenance["activation_path"],
            "policy_activation_phase": activation.phase,
            "policy_activation_ratio": activation.ratio,
            "policy_activation_provenance": activation.provenance,
            "policy_activation_settle_stage": (
                None if observation is None else observation.stage
            ),
            "policy_activation_settle_stable": (
                None if observation is None else observation.stable
            ),
            "policy_activation_settle_consecutive_samples": (
                0 if observation is None else observation.consecutive_samples
            ),
            "policy_activation_settle_required_consecutive_samples": (
                activation.settle_consecutive_samples
            ),
            "policy_activation_settle_elapsed_s": (
                None if observation is None else observation.elapsed_s
            ),
            "policy_activation_settle_observed_max_joint_error_rad": (
                None if observation is None else observation.max_joint_error_rad
            ),
            "policy_activation_settle_observed_max_joint_velocity_rps": (
                None
                if observation is None
                else observation.max_joint_velocity_rps
            ),
            "policy_activation_settle_observed_max_gyro_rps": (
                None if observation is None else observation.max_gyro_rps
            ),
            "policy_activation_settle_observed_roll_rad": (
                None if observation is None else observation.roll_rad
            ),
            "policy_activation_settle_observed_pitch_rad": (
                None if observation is None else observation.pitch_rad
            ),
            "policy_activation_settle_max_joint_error_rad": (
                activation.max_settle_joint_error_rad
            ),
            "policy_activation_settle_max_joint_velocity_rps": (
                activation.max_settle_joint_velocity_rps
            ),
            "policy_activation_settle_max_gyro_rps": (
                activation.max_settle_gyro_rps
            ),
            "policy_activation_settle_max_abs_roll_pitch_rad": (
                activation.max_settle_roll_pitch_rad
            ),
            "policy_activation_settle_timeout_s": activation.settle_timeout_s,
            "policy_activation_release_epoch": getattr(
                self, "_policy_activation_release_epoch", None
            ),
            "policy_activation_release_confirmed_epoch": getattr(
                self, "_policy_activation_release_confirmed_epoch", None
            ),
            "policy_activation_release_pending": bool(
                release_thread is not None and release_thread.is_alive()
            ),
            "policy_activation_fall_arrest_retained": bool(
                not activation.release_fall_arrest
            ),
            "policy_activation_release_error": (
                None if release_error is None else str(release_error)
            ),
            "policy_activation_prepared_tick_ms": getattr(
                self, "_policy_activation_prepared_tick_ms", None
            ),
        }

    def _reset_policy_activation_release_locked(self) -> None:
        """Forget a local async release result without reviving its epoch."""

        self._policy_activation_release_epoch = None
        self._policy_activation_release_result = None
        self._policy_activation_release_error = None
        self._policy_activation_release_thread = None

    def _start_policy_activation_release_locked(self) -> bool:
        """Start the exact-epoch Plant proof after policy owns DDS."""

        activation = self._policy_activation
        if not activation.waiting_for_support_release:
            return False
        if activation.activation_path != "tethered_release":
            return False
        if self._reference_authority.snapshot.phase != "GRANTED":
            return False
        epoch = activation.epoch
        existing_epoch = getattr(
            self, "_policy_activation_release_epoch", None
        )
        existing_thread = getattr(
            self, "_policy_activation_release_thread", None
        )
        if existing_epoch == epoch and existing_thread is not None:
            return False
        if existing_thread is not None and existing_thread.is_alive():
            raise NominalPolicyActivationFault(
                "another Plant fall-arrest release is still pending"
            )
        self._policy_activation_release_epoch = epoch
        self._policy_activation_release_result = None
        self._policy_activation_release_error = None

        def release_worker() -> None:
            result: dict[str, Any] | None = None
            error: Exception | None = None
            try:
                result = release_plant_fall_arrest(
                    "127.0.0.1",
                    self.config.plant_port,
                    epoch,
                )
            except Exception as exc:
                error = exc
            with self._support_lock:
                if error is None and result is not None:
                    try:
                        self._policy_activation.record_support_release(epoch)
                    except Exception as exc:
                        error = exc
                        result = None
                    else:
                        self._policy_activation_release_confirmed_epoch = epoch
                if self._policy_activation_release_epoch != epoch:
                    return
                self._policy_activation_release_result = result
                self._policy_activation_release_error = error

        thread = threading.Thread(
            target=release_worker,
            name=f"sonic-fall-arrest-release-{epoch}",
            daemon=True,
        )
        self._policy_activation_release_thread = thread
        thread.start()
        return True

    def _consume_policy_activation_release_locked(self, *, tick_ms: int) -> bool:
        """Advance only from the async helper's authoritative Plant proof."""

        activation = self._policy_activation
        epoch = getattr(self, "_policy_activation_release_epoch", None)
        if epoch != activation.epoch:
            return False
        error = getattr(self, "_policy_activation_release_error", None)
        if error is not None:
            self._policy_activation_release_thread = None
            raise NominalPolicyActivationFault(
                f"Plant fall-arrest release failed for activation epoch {epoch}: {error}"
            )
        result = getattr(self, "_policy_activation_release_result", None)
        if result is None:
            return False
        confirmed_epoch = result.get("fallArrestReleaseConfirmedEpoch")
        if confirmed_epoch != epoch or result.get("fallArrestEnabled") is not False:
            raise NominalPolicyActivationFault(
                "Plant fall-arrest release proof did not match activation epoch"
            )
        if not activation.confirm_support_release(tick_ms=tick_ms):
            raise NominalPolicyActivationFault(
                "Plant release proof arrived outside the activation wait phase"
            )
        self._policy_activation_release_confirmed_epoch = epoch
        self._policy_activation_release_result = None
        self._policy_activation_release_error = None
        self._policy_activation_release_thread = None
        return True

    def _commit_policy_activation_locked(
        self,
        decision: NominalPolicyActivationDecision,
        published_targets: tuple[float, ...],
        *,
        simulation_now_s: float,
    ) -> tuple[bool, bool]:
        """Commit a DDS INIT fact as ``(accepted, ramp_completed)``."""

        activation = self._policy_activation
        authority = self._reference_authority.snapshot
        identity = self._reference_source_identity
        admitted_intent = getattr(self._intent_gate, "last", None)
        if (
            not activation.is_current(decision)
            or not getattr(self, "_policy_activation_requested", False)
            or authority.phase != "REQUIRED"
            or identity is None
            or authority.generation != identity[0]
            or authority.source_sequence != identity[1]
            or not self._intent_ingress.armed
            or self._intent_ingress.generation != identity[0]
            or self._plant_support_stop_pending
            or self._plant_support_safety_latched
            or self._normal_stop_fences_motion_locked()
            or self._process_stop_fences_motion_locked()
            or self._fault_latched.is_set()
            or admitted_intent is None
            or admitted_intent.mode == "IDLE"
            or tuple(published_targets) != decision.targets
        ):
            return False, False
        return True, activation.commit(decision, published_targets)

    def _complete_policy_activation_locked(
        self,
        published_targets: tuple[float, ...],
        expected_boundary: tuple[float, ...],
        *,
        tick_ms: int,
        simulation_now_s: float,
    ) -> bool:
        """Open planning after tethered INIT, before fall-arrest release.

        The learned controller must own a successful DDS target before the
        simulation gantry is released.  Otherwise a fixed nominal PD target is
        asked to balance an unsupported floating humanoid, which is not a
        valid state in the official deployment either.
        """

        activation = self._policy_activation
        admitted_intent = getattr(self._intent_gate, "last", None)
        if (
            not activation.waiting_for_support_release
            or not self._policy_activation_boundary_valid_locked()
            or admitted_intent is None
            or admitted_intent.mode == "IDLE"
            or activation.ready_targets != tuple(expected_boundary)
            or tuple(published_targets) != expected_boundary
        ):
            return False
        # Every policy-free waiting tick and the later binary policy handoff
        # now originates at the exact nominal DDS boundary. Retaining the
        # pre-activation hold here would command a one-tick jump backwards.
        self._joint_handoff.freeze(published_targets)
        self._planner_evidence.observe_intent(
            admitted_intent,
            admitted=True,
            simulation_now_s=simulation_now_s,
        )
        self._reference_request_continuous = False
        self._reference_request_pending = True
        self._policy_activation_requested = False
        return True

    def _advance_policy_activation_release_after_grant(
        self,
        *,
        tick_ms: int,
    ) -> bool:
        """Complete the configured support transition after first policy DDS.

        This is the software equivalent of NVIDIA's documented gantry
        sequence: the policy starts balancing while the robot is supported,
        then the support is released.  The initial release is asynchronous and
        exact-epoch bound; a cancellation that lost that one-way race resumes
        from its frozen policy boundary without attempting another release.
        """

        with self._support_lock:
            activation = getattr(self, "_policy_activation", None)
            if activation is None:
                return True
            if activation.prepared:
                return True
            if not activation.waiting_for_support_release:
                return False
            if self._reference_authority.snapshot.phase != "GRANTED":
                return False
            if activation.activation_path == "support_off_reinit":
                if not activation.confirm_support_release(tick_ms=tick_ms):
                    raise NominalPolicyActivationFault(
                        "support-off recovery left the activation wait phase"
                    )
            elif activation.activation_path == "tethered_release":
                self._consume_policy_activation_release_locked(tick_ms=tick_ms)
                if activation.waiting_for_support_release:
                    self._start_policy_activation_release_locked()
            elif activation.activation_path == "tethered_retained":
                if not activation.confirm_supported_takeover(tick_ms=tick_ms):
                    raise NominalPolicyActivationFault(
                        "retained-support takeover left the activation wait phase"
                    )
            else:
                raise NominalPolicyActivationFault(
                    "policy activation path is unavailable after DDS grant"
                )
            if activation.prepared:
                self._policy_activation_prepared_tick_ms = int(tick_ms)
                return True
            return False

    @staticmethod
    def _policy_activation_stability_inputs(
        lowstate: Any,
    ) -> tuple[tuple[float, ...], tuple[float, float, float], float, float]:
        states = getattr(lowstate, "motor_state", None)
        if states is None or len(states) < 29:
            raise NominalPolicyActivationFault(
                "activation settling requires 29 measured joint states"
            )
        velocities = tuple(float(states[index].dq) for index in range(29))
        imu = getattr(lowstate, "imu_state", None)
        gyroscope_raw = None if imu is None else getattr(imu, "gyroscope", None)
        quaternion_raw = (
            None if imu is None else getattr(imu, "quaternion", None)
        )
        if gyroscope_raw is None or len(gyroscope_raw) != 3:
            raise NominalPolicyActivationFault(
                "activation settling requires measured IMU gyroscope"
            )
        if quaternion_raw is None or len(quaternion_raw) != 4:
            raise NominalPolicyActivationFault(
                "activation settling requires measured IMU quaternion"
            )
        gyroscope = tuple(float(value) for value in gyroscope_raw)
        quaternion = tuple(float(value) for value in quaternion_raw)
        if not all(
            math.isfinite(value)
            for value in velocities + gyroscope + quaternion
        ):
            raise NominalPolicyActivationFault(
                "activation settling requires finite joint/IMU evidence"
            )
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not 0.999 <= norm <= 1.001:
            raise NominalPolicyActivationFault(
                "activation settling requires a normalized IMU quaternion"
            )
        qw, qx, qy, qz = (value / norm for value in quaternion)
        roll = math.atan2(
            2.0 * (qw * qx + qy * qz),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
        pitch = math.asin(
            max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
        )
        return velocities, gyroscope, roll, pitch

    def _run_nominal_policy_activation(
        self,
        *,
        lowstate: Any,
        control_step: PlantControlStep,
        intent: Any,
        measured: tuple[float, ...],
    ) -> bool:
        """Run one policy-free 50 Hz INIT tick at the actuator boundary."""

        from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
            DEFAULT_ANGLES,
        )

        started_at = self._wall_clock()
        nominal_targets = tuple(float(value) for value in DEFAULT_ANGLES)
        activation = self._policy_activation
        with self._support_lock:
            authority = self._reference_authority.snapshot
            identity = self._reference_source_identity
            if (
                not self._policy_activation_requested
                or activation.prepared
                or intent.mode == "IDLE"
                or authority.phase != "REQUIRED"
                or identity is None
                or authority.generation != identity[0]
                or authority.source_sequence != identity[1]
                or not self._intent_ingress.armed
                or self._intent_ingress.generation != identity[0]
                or not self._navigation_task_active
                or self._process_stop_fences_motion_locked()
                or self._fault_latched.is_set()
                or self._plant_support_stop_pending
                or self._plant_support_safety_latched
                or self._normal_stop_fences_motion_locked()
            ):
                return False
            boundary = self._last_published_targets
            if boundary is None:
                raise RuntimeError(
                    "nominal policy activation has no executed actuator boundary"
                )
            activation.begin(boundary, tick_ms=control_step.tick_ms)
            phase_at_tick = activation.phase
            decision: NominalPolicyActivationDecision | None = None
            settle_phase_to_observe: str | None = None
            if activation.preparing:
                decision = activation.propose(
                    nominal_targets,
                    measured=measured,
                    tick_ms=control_step.tick_ms,
                )
                released = self._release_safe_hold_for_policy_activation_locked(
                    decision
                )
                requested_targets = decision.targets
            else:
                released = self._policy_activation_boundary_valid_locked()
                if released:
                    self._safe_hold_targets = None
                requested_targets = activation.ready_targets or nominal_targets
                if phase_at_tick == "SETTLING_TETHERED":
                    settle_phase_to_observe = phase_at_tick

        proposed_at = self._wall_clock()
        support_ready = self._set_plant_support_mode("idle")
        support_ready_at = self._wall_clock()
        with self._support_lock:
            effective_static_hold = self._commit_static_hold_decision_locked(
                not released or not support_ready
            )
        with self._support_lock:
            output_targets = (
                requested_targets
                if released and support_ready and not effective_static_hold
                else (
                    self._safe_hold_targets
                    or self._last_published_targets
                    or boundary
                )
            )
        output_action = {
            f"{self._joint_name(index)}.q": target
            for index, target in enumerate(output_targets)
        }
        published_targets = self._publish(
            output_action,
            lowstate,
            control_dt_s=control_step.dt_s,
            control_tick_ms=control_step.tick_ms,
            constrain_arms=not effective_static_hold,
        )
        published_at = self._wall_clock()
        with self._support_lock:
            # This flag proves the actuator/support boundary for this tick; it
            # does not mean the multi-tick activation has completed. A stable
            # sample below the consecutive threshold and an async release with
            # no result yet are both successful, legal transaction ticks.
            activation_tick_committed = False
            if released and support_ready and not effective_static_hold:
                if decision is not None:
                    activation_tick_committed, _ = (
                        self._commit_policy_activation_locked(
                            decision,
                            published_targets,
                            simulation_now_s=control_step.tick_ms / 1_000.0,
                        )
                    )
                else:
                    activation_tick_committed = bool(
                        self._policy_activation_boundary_valid_locked()
                        and tuple(published_targets) == nominal_targets
                    )
                    if (
                        activation_tick_committed
                        and settle_phase_to_observe is not None
                        and activation.phase == settle_phase_to_observe
                    ):
                        velocities, gyroscope, roll, pitch = (
                            self._policy_activation_stability_inputs(lowstate)
                        )
                        observation = activation.observe_settling(
                            nominal=nominal_targets,
                            measured=measured,
                            velocities=velocities,
                            gyroscope=gyroscope,
                            roll_rad=roll,
                            pitch_rad=pitch,
                            tick_ms=control_step.tick_ms,
                        )
                    if (
                        activation_tick_committed
                        and activation.waiting_for_support_release
                    ):
                        activation_tick_committed = (
                            self._complete_policy_activation_locked(
                                published_targets,
                                requested_targets,
                                tick_ms=control_step.tick_ms,
                                simulation_now_s=(
                                    control_step.tick_ms / 1_000.0
                                ),
                            )
                        )
            if (
                not activation_tick_committed
                and self._policy_activation_requested
                and not activation.prepared
            ):
                # Only an actual support/DDS/boundary failure reaches here.
                # Incomplete tethered settling samples retain their phase and
                # epoch across valid ticks while publishing exact nominal hold.
                activation.abort()
                self._reset_policy_activation_release_locked()

        self._record_control_trace(
            lowstate=lowstate,
            control_step=control_step,
            intent=intent,
            policy_targets=nominal_targets,
            handoff_targets=requested_targets,
            published_targets=published_targets,
            handoff_ratio=(activation.ratio if decision is None else decision.ratio),
        )
        traced_at = self._wall_clock()
        with self._support_lock:
            phase = activation.phase
            current_static_hold = self._static_hold_active
            activation_active = bool(
                self._policy_activation_requested
                and phase != "PREPARED"
            )
            activation_status = {
                "PREPARING": "simulation_time_50hz_preparing_nominal",
                "SETTLING_TETHERED": "simulation_time_50hz_settling_tethered",
                "WAITING_FOR_SUPPORT_RELEASE": "waiting_for_matched_reference_before_support_release",
                "PREPARED": "simulation_time_50hz_nominal_prepared",
            }.get(phase, "simulation_time_50hz_motion_ready")
            self._update_telemetry_hot(
                sonic_policy_speed_mps=0.0,
                control_tick_ms=control_step.tick_ms,
                control_dt_s=control_step.dt_s,
                control_clock_status=activation_status,
                static_hold_active=current_static_hold,
                motion_handoff_active=(
                    activation_active and not current_static_hold
                ),
                motion_handoff_ratio=(
                    activation.ratio
                    if activation_active and not current_static_hold
                    else 0.0
                ),
                **self._policy_activation_telemetry_fields_locked(),
                **self._planner_telemetry_fields(),
            )
        completed_at = self._wall_clock()
        elapsed_s = max(0.0, completed_at - started_at)
        if elapsed_s > 0.05:
            logger.warning(
                "slow SONIC activation tick total_s=%.6f propose_s=%.6f "
                "support_s=%.6f publish_s=%.6f trace_s=%.6f telemetry_s=%.6f "
                "phase=%s",
                elapsed_s,
                max(0.0, proposed_at - started_at),
                max(0.0, support_ready_at - proposed_at),
                max(0.0, published_at - support_ready_at),
                max(0.0, traced_at - published_at),
                max(0.0, completed_at - traced_at),
                phase,
            )
        return True

    def _progress_reference_transaction(self, lowstate) -> bool:
        """Plan, install, and prime one required reference without policy work."""

        with self._support_lock:
            continuous_stop = bool(
                getattr(self, "_reference_request_continuous", False)
            )
            if continuous_stop and self._vendor_revoke_pending:
                raise RuntimeError(
                    "continuous IDLE reference cannot consume a vendor revoke"
                )
            self._consume_vendor_revoke_locked()
            authority = self._reference_authority.snapshot
            if authority.phase != "REQUIRED":
                return self._reference_authority.executable
            controlled_stop = bool(
                getattr(self, "_controlled_stop_active", False)
            )
            if controlled_stop != continuous_stop:
                raise RuntimeError(
                    "controlled-stop reference lost its continuous-history identity"
                )
            controller = self._controller
            if not controller.initialized:
                raise RuntimeError("manual SONIC reference requested before initialization")
            if self._reference_request_pending:
                if controller.execution_feedback_pending:
                    return False
                if continuous_stop:
                    request_continuous_idle = getattr(
                        controller,
                        "request_continuous_idle_reference",
                        None,
                    )
                    if not callable(request_continuous_idle):
                        raise RuntimeError(
                            "SONIC controller lacks continuous IDLE reference support"
                        )
                    request_continuous_idle()
                else:
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
            if continuous_stop:
                acquire_continuous = getattr(
                    controller,
                    "acquire_continuous_reference",
                    None,
                )
                if not callable(acquire_continuous):
                    raise RuntimeError(
                        "SONIC controller lacks continuous reference acquisition"
                    )
                acquire_continuous(
                    executed_target=executed_target,
                    expected_install_epoch=epoch,
                )
            else:
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
            if getattr(self, "_controlled_stop_active", False):
                controlled_identity = getattr(
                    self, "_controlled_stop_source_identity", None
                )
                if (
                    controlled_identity is None
                    or controlled_identity[0] != authority.generation
                    or controlled_identity[1] != authority.source_sequence
                    or int(getattr(controller.ms, "mode", -1)) != 0
                    or bool(getattr(controller.ms, "has_movement", True))
                ):
                    raise RuntimeError(
                        "fresh controlled-stop reference lacks exact IDLE identity"
                    )
                # The continuous acquisition has installed a heading-neutral
                # planner context and armed the decoder's own heading
                # reinitialization without replacing recurrent histories. Keep
                # the driver's bookkeeping in that same local frame. Retaining
                # the previous action's absolute controller angle here makes
                # the next motion compute a compensating turn toward the old
                # frame.
                self._decoder_facing_rad = 0.0
                self._pending_decoder_heading_delta_rad = 0.0
                self._reference_source_identity = controlled_identity
                self._controlled_stop_reference_installed = True
                self._reference_request_continuous = False
            # An earlier stop hold may be released only after the preserved
            # history and exact executed boundary have been atomically bound to
            # the newly accepted reference.
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
                or proof.accepted_intent_signature != proof.intent_signature
                or not controller.matching_reference_installed
            ):
                raise RuntimeError(
                    "installed SONIC refresh lacks matching authority evidence"
                )
            assert authority.generation is not None
            assert authority.source_sequence is not None
            assert authority.planner_intent_generation is not None
            controlled_stop_identity = getattr(
                self, "_controlled_stop_source_identity", None
            )
            if getattr(self, "_controlled_stop_active", False):
                controlled_stop_reference_installed = bool(
                    getattr(self, "_controlled_stop_reference_installed", False)
                )
                if (
                    controlled_stop_identity is None
                    or controlled_stop_identity[0] != authority.generation
                    or int(getattr(controller.ms, "mode", -1)) != 0
                    or bool(getattr(controller.ms, "has_movement", True))
                ):
                    raise RuntimeError(
                        "controlled-stop IDLE reference lacks exact transition identity"
                    )
                if controlled_stop_reference_installed:
                    if (
                        controlled_stop_identity[1] != authority.source_sequence
                        or proof.accepted_intent_generation
                        != authority.planner_intent_generation
                    ):
                        raise RuntimeError(
                            "installed controlled-stop reference changed identity"
                        )
                    self._reference_authority.refresh(
                        generation=authority.generation,
                        source_sequence=authority.source_sequence,
                        planner_intent_generation=(
                            authority.planner_intent_generation
                        ),
                        reference_epoch=epoch,
                    )
                    return
                if (
                    proof.accepted_intent_generation is None
                    or proof.accepted_intent_generation
                    == authority.planner_intent_generation
                ):
                    raise RuntimeError(
                        "controlled-stop IDLE reference lacks a new planner identity"
                    )
                self._reference_authority.transition(
                    generation=controlled_stop_identity[0],
                    source_sequence=controlled_stop_identity[1],
                    planner_intent_generation=proof.accepted_intent_generation,
                    reference_epoch=epoch,
                )
                self._reference_source_identity = controlled_stop_identity
                self._controlled_stop_reference_installed = True
                return
            if (
                proof.accepted_intent_generation
                != authority.planner_intent_generation
            ):
                raise RuntimeError(
                    "periodic SONIC refresh changed planner intent generation"
                )
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
        with self._arm_evidence_boundary_lock:
            return self._publish_unfenced(
                action,
                lowstate,
                control_dt_s=control_dt_s,
                control_tick_ms=control_tick_ms,
                constrain_arms=constrain_arms,
            )

    def _publish_unfenced(
        self,
        action: dict[str, Any],
        lowstate,
        *,
        control_dt_s: float,
        control_tick_ms: int | None = None,
        constrain_arms: bool = True,
    ) -> tuple[float, ...]:
        publish_started_at = self._wall_clock()
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
        publish_prepared_at = self._wall_clock()
        guarded_arm = None
        with self._support_lock:
            terminal_boundary = self._enforce_terminal_boundary_locked()
            process_stop = self._process_stop_fences_motion_locked()
            if process_stop and not terminal_boundary:
                self._freeze_last_published_hold(publish_telemetry=False)
            if not terminal_boundary:
                self._consume_vendor_revoke_locked()
            if (
                (terminal_boundary or process_stop)
                and self._last_published_targets is None
            ):
                raise RuntimeError(
                    "stopping SONIC publish has no executed actuator boundary"
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
            self._stamp_sim_control_identity(control_tick_ms)
            self._msg.crc = self._crc.Crc(self._msg)
            with self._terminal_fault_lock:
                terminal_before_write = self._fault_latched.is_set()
                process_stop_before_write = (
                    self._process_stop_fences_motion_locked()
                )
                if terminal_before_write or process_stop_before_write:
                    if terminal_before_write:
                        self._enforce_terminal_boundary_locked()
                    else:
                        self._freeze_last_published_hold(
                            publish_telemetry=False
                        )
                        self._consume_vendor_revoke_locked()
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
            evidence_write_token = self._begin_arm_evidence_write()
            try:
                written = self._publisher.Write(self._msg)
            except BaseException:
                self._finish_arm_evidence_write(evidence_write_token)
                raise
            if written is not True:
                self._finish_arm_evidence_write(evidence_write_token)
                raise RuntimeError("Unitree DDS policy-target write failed")
            # The post-Write terminal check, exact executed feedback, and
            # possible first-DDS grant form one transaction against the
            # terminal claim.  DDS itself stays outside this lock because its
            # implementation has no timeout and must never delay Plant halt.
            with self._terminal_fault_lock:
                terminal_error: Exception | None = None
                terminal_after_write = self._fault_latched.is_set()
                process_stop_after_write = (
                    self._process_stop_fences_motion_locked()
                )
                stopping_after_write = (
                    terminal_after_write or process_stop_after_write
                )
                if stopping_after_write:
                    final_targets = self._record_successful_dds_target(
                        published_targets,
                        evidence_write_token=evidence_write_token,
                        arm_guard_result=guarded_arm,
                    )
                    self._safe_hold_targets = final_targets
                    try:
                        if terminal_after_write:
                            self._enforce_terminal_boundary_locked()
                        else:
                            self._freeze_last_published_hold(
                                publish_telemetry=False
                            )
                            self._consume_vendor_revoke_locked()
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
                        published_targets,
                        evidence_write_token=evidence_write_token,
                        arm_guard_result=guarded_arm,
                    )
                authority = self._reference_authority.snapshot
                if not stopping_after_write and authority.phase == "PRIMED":
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
        publish_committed_at = self._wall_clock()
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
        self._update_telemetry_hot(
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
        publish_completed_at = self._wall_clock()
        publish_elapsed_s = max(0.0, publish_completed_at - publish_started_at)
        if publish_elapsed_s > 0.05:
            logger.warning(
                "slow SONIC publish total_s=%.6f prepare_s=%.6f "
                "boundary_dds_s=%.6f metrics_s=%.6f",
                publish_elapsed_s,
                max(0.0, publish_prepared_at - publish_started_at),
                max(0.0, publish_committed_at - publish_prepared_at),
                max(0.0, publish_completed_at - publish_committed_at),
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
                "command_facing_angle_rad": float(
                    getattr(self.guard, "command_facing", intent.facing_angle_rad)
                ),
                "measured_heading_yaw_rad": getattr(
                    self.guard, "measured_heading", None
                ),
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

    @staticmethod
    def _desired_policy_support_mode(
        *,
        intent_mode: str,
        controlled_stop: bool,
        decision_static_hold: bool,
        activation_waiting_for_release: bool,
        handoff_ratio: float,
    ) -> str:
        """Select Plant support without treating learned braking as locomotion.

        A correlated normal stop keeps SONIC's already-granted controller only
        long enough to run the official IDLE policy to measured stationarity.
        The Plant transaction deliberately owns the ``idle`` support state
        throughout that interval.  Asking for ``locomotion`` here contradicts
        :meth:`_set_plant_support_mode`, which rejects new locomotion as soon
        as BRAKING owns the boundary.
        """

        if (
            controlled_stop
            or decision_static_hold
            or activation_waiting_for_release
        ):
            return "idle"
        if intent_mode != "IDLE" and float(handoff_ratio) > 0.0:
            return "locomotion"
        return "idle"

    def _old_locomotion_tick_lost_to_controlled_stop_locked(
        self,
        *,
        controlled_stop_at_tick_start: bool,
    ) -> bool:
        """Recognize only a pre-BRAKING locomotion decision that lost its race."""

        normal_stop = getattr(self, "_normal_stop", None)
        return bool(
            not controlled_stop_at_tick_start
            and getattr(self, "_normal_stop_policy_braking", False)
            and isinstance(normal_stop, NormalStopState)
            and normal_stop.snapshot.phase == "BRAKING"
        )

    def _process_stop_fences_motion_locked(self) -> bool:
        stop_event = getattr(self, "_stop", None)
        return bool(stop_event is not None and stop_event.is_set())

    def _normal_stop_static_boundary_locked(self) -> bool:
        """Return the current recoverable stop's physical/local hold fence."""

        normal_stop = getattr(self, "_normal_stop", None)
        phase = (
            normal_stop.snapshot.phase
            if isinstance(normal_stop, NormalStopState)
            else "NONE"
        )
        policy_transition = bool(
            getattr(self, "_normal_stop_policy_braking", False)
            and phase == "BRAKING"
        )
        policy_balance_until_plant = bool(
            getattr(self, "_normal_stop_policy_braking", False)
            and phase in {"STATIONARY_ACK", "FAILED"}
            and getattr(self, "_controlled_stop_active", False)
            and getattr(self, "_controlled_stop_reference_installed", False)
            and self._reference_authority.snapshot.phase == "GRANTED"
            and self._reference_authority.snapshot.actuator_authority
            and self._safe_hold_targets is None
            and self._plant_support_mode == "idle"
            and not self._plant_support_safety_latched
            and not self._process_stop_fences_motion_locked()
            and not self._fault_latched.is_set()
        )
        policy_owned_stop = policy_transition or policy_balance_until_plant
        return (
            self._process_stop_fences_motion_locked()
            or (
                self._plant_support_stop_pending
                and not policy_balance_until_plant
            )
            or self._plant_support_safety_latched
            or self._safe_hold_targets is not None
            or (
                self._normal_stop_fences_motion_locked()
                and not policy_owned_stop
            )
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
            loop_started_at = self._wall_clock()
            self._control_loop_stage = "waiting_for_control_sample"
            sample = self._next_control_sample()
            if self._fault_latched.is_set():
                if getattr(self, "_recoverable_motion_fault", False):
                    if sample is None:
                        continue
                    lowstate, control_step, _sample_drop_generation = sample
                    try:
                        self._publish_frozen_hold(lowstate)
                        self._record_motion_fault_hold_frame(
                            control_step.tick_ms
                        )
                    except Exception as exc:
                        self._latch_terminal_fault(
                            "recoverable motion-fault hold failed: " + str(exc)
                        )
                    continue
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
            self._control_evidence_context.control_step = control_step
            try:
                if self._startup_qualifying:
                    self._control_loop_stage = "startup_qualification_hold"
                    # Qualification is intentionally policy-free.  The plant is
                    # stationary, obsolete samples may still be latest-only
                    # synchronized, and only a measured hold is published.
                    self._publish_measured_hold(lowstate)
                    self._complete_startup_qualification_step(
                        sample_drop_generation=sample_drop_generation
                    )
                    continue
                self._control_loop_stage = "intent_apply"
                intent = self._apply_intent(
                    control_step.dt_s,
                    simulation_now_s=control_step.tick_ms / 1_000.0,
                )
                measured = tuple(
                    float(lowstate.motor_state[index].q) for index in range(29)
                )
                controlled_stop = bool(
                    getattr(self, "_controlled_stop_active", False)
                    and intent.mode == "IDLE"
                )
                if intent.mode == "IDLE" and not controlled_stop:
                    self._control_loop_stage = "idle_position_hold"
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
                    with self._support_lock:
                        activation_fields = (
                            self._policy_activation_telemetry_fields_locked()
                        )
                    self._update_telemetry_hot(
                        sonic_policy_speed_mps=0.0,
                        control_tick_ms=control_step.tick_ms,
                        control_dt_s=control_step.dt_s,
                        control_clock_status="simulation_time_50hz_motion_ready",
                        static_hold_active=True,
                        motion_handoff_active=False,
                        motion_handoff_ratio=0.0,
                        **activation_fields,
                        **self._planner_telemetry_fields(),
                    )
                    continue

                activation = getattr(self, "_policy_activation", None)
                if (
                    activation is not None
                    and getattr(self, "_policy_activation_requested", False)
                    and not activation.prepared
                ):
                    self._control_loop_stage = "nominal_policy_activation"
                    try:
                        activation_tick_published = (
                            self._run_nominal_policy_activation(
                                lowstate=lowstate,
                                control_step=control_step,
                                intent=intent,
                                measured=measured,
                            )
                        )
                    except NominalPolicyActivationFault as exc:
                        self._latch_motion_fault(
                            "policy_activation/POLICY_ACTIVATION_FAULT: "
                            + str(exc)
                        )
                        continue
                    # INIT and tethered settling are policy-free.  Their final
                    # exact DDS boundary opens reference planning; the learned
                    # controller then receives DDS authority before the
                    # simulation fall arrest is released.
                    if activation_tick_published:
                        continue

                self._control_loop_stage = "reference_transaction"
                if not self._progress_reference_transaction(lowstate):
                    self._control_loop_stage = "reference_wait_hold"
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
                    self._update_telemetry_hot(
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

                if controlled_stop:
                    self._enforce_planner_reference_deadline(
                        simulation_now_s=control_step.tick_ms / 1_000.0
                    )

                try:
                    self._advance_policy_activation_release_after_grant(
                        tick_ms=control_step.tick_ms
                    )
                except NominalPolicyActivationFault as exc:
                    self._latch_motion_fault(
                        "policy_activation/POLICY_ACTIVATION_FAULT: "
                        + str(exc)
                    )
                    continue

                # Feed empty external action: the controller reads the mapped
                # MovementState above and executes the official SONIC pipeline.
                # Decode creates a pending candidate. Its action history and
                # reference cursor commit only after the complete DDS write.
                self._control_loop_stage = "policy_candidate"
                policy_action = self._run_policy_candidate(lowstate)
                self._update_telemetry_hot(
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
                activation_waiting_for_release = bool(
                    activation is not None
                    and not activation.prepared
                    and activation.waiting_for_support_release
                )
                desired_support_mode = self._desired_policy_support_mode(
                    intent_mode=intent.mode,
                    controlled_stop=controlled_stop,
                    decision_static_hold=decision.static_hold_active,
                    activation_waiting_for_release=(
                        activation_waiting_for_release
                    ),
                    handoff_ratio=decision.handoff_ratio,
                )
                support_ready = self._set_plant_support_mode(
                    desired_support_mode
                )
                if not support_ready and desired_support_mode == "locomotion":
                    with self._support_lock:
                        old_tick_lost_to_controlled_stop = (
                            self._old_locomotion_tick_lost_to_controlled_stop_locked(
                                controlled_stop_at_tick_start=controlled_stop,
                            )
                        )
                        expected_safety_stop = (
                            self._rejected_locomotion_is_static_boundary_locked()
                        )
                        rejected_transition_token = (
                            self._support_transition_token_locked()
                        )
                    if old_tick_lost_to_controlled_stop:
                        # This tick sampled motion before the correlated stop
                        # acquired the boundary.  The next 50 Hz tick will run
                        # the already-granted IDLE policy under Plant idle
                        # support.  A policy candidate already exists, so
                        # publish the exact prior actuator target once and
                        # feed that physical fact back to SONIC.  Skipping DDS
                        # here would strand unpublished execution feedback;
                        # publishing the obsolete motion proposal would cross
                        # the newly-owned stop boundary.
                        with self._support_lock:
                            controlled_hold_targets = (
                                self._safe_hold_targets
                                or self._last_published_targets
                            )
                        if controlled_hold_targets is None:
                            raise RuntimeError(
                                "controlled stop has no executed actuator boundary"
                            )
                        self._publish_position_hold(controlled_hold_targets)
                        assert self._last_published_targets is not None
                        self._record_control_trace(
                            lowstate=lowstate,
                            control_step=control_step,
                            intent=intent,
                            policy_targets=policy_targets,
                            handoff_targets=decision.targets,
                            published_targets=self._last_published_targets,
                            handoff_ratio=decision.handoff_ratio,
                        )
                        continue
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
                self._control_loop_stage = "policy_publish"
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
                try:
                    self._advance_policy_activation_release_after_grant(
                        tick_ms=control_step.tick_ms
                    )
                except NominalPolicyActivationFault as exc:
                    self._latch_motion_fault(
                        "policy_activation/POLICY_ACTIVATION_FAULT: "
                        + str(exc)
                    )
                    continue
                with self._support_lock:
                    activation_fields = (
                        self._policy_activation_telemetry_fields_locked()
                    )
                    activation_release_pending = bool(
                        not self._policy_activation.prepared
                        and self._policy_activation.waiting_for_support_release
                    )
                self._update_telemetry_hot(
                    control_tick_ms=control_step.tick_ms,
                    control_dt_s=control_step.dt_s,
                    control_clock_status=(
                        "policy_balancing_before_support_release"
                        if activation_release_pending
                        else "simulation_time_50hz_motion_ready"
                    ),
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
                    **activation_fields,
                    **self._planner_telemetry_fields(),
                )
            except Exception as exc:
                self._latch_terminal_fault(f"SONIC runtime fault: {exc}")
                try:
                    self._publish_frozen_hold(lowstate)
                finally:
                    self._stop.set()
                break
            finally:
                loop_elapsed_s = max(0.0, self._wall_clock() - loop_started_at)
                if loop_elapsed_s > 0.035:
                    logger.warning(
                        "slow SONIC control loop elapsed_s=%.6f tick_ms=%s",
                        loop_elapsed_s,
                        getattr(control_step, "tick_ms", "unavailable"),
                    )
                self._control_evidence_context.control_step = None
                self._control_loop_stage = "waiting_for_control_sample"
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
        self._close_lowstate_reader()

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
        if not hasattr(self, "_normal_stop_policy_braking"):
            self._normal_stop_policy_braking = False
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
                    policy_candidate = self._controlled_policy_stop_eligible_locked(
                        generation
                    )
                    if not policy_candidate:
                        # Startup/pre-grant and already-static stops establish
                        # the exact boundary before the transaction becomes
                        # visible, preserving the original shutdown ordering.
                        try:
                            self._stop_navigation_at_boundary(
                                source="normal_stop_begin",
                                publish_telemetry=False,
                            )
                        except Exception as exc:
                            raise _NormalStopBoundaryError(str(exc)) from exc
                        self._static_hold_active = True
                    try:
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
                            policy_braking = bool(
                                policy_candidate
                                and self._begin_controlled_policy_stop_locked(
                                    decision,
                                    correlated=True,
                                )
                            )
                            if policy_candidate and not policy_braking:
                                # A regular STOPPING envelope may already own
                                # a newer installed/GRANTED IDLE identity.  An
                                # explicit correlated normal-stop is the
                                # stronger operator boundary: if it cannot
                                # join that exact identity, freeze/revoke now
                                # instead of leaving an untracked policy active.
                                try:
                                    self._stop_navigation_at_boundary(
                                        source="normal_stop_begin",
                                        publish_telemetry=False,
                                    )
                                except Exception as exc:
                                    raise _NormalStopBoundaryError(
                                        str(exc)
                                    ) from exc
                                self._static_hold_active = True
                            admitted_action = (
                                decision.action == "brake"
                                if policy_braking
                                else decision.action in {"brake", "stop"}
                            )
                            if (
                                not admitted_action
                                or decision.state != "STOPPING"
                                or decision.generation != generation
                                or decision.sequence != sequence
                            ):
                                raise NormalStopWireError(
                                    "normal-stop begin intent was not admitted exactly"
                                )
                            if policy_braking:
                                controlled_identity = (
                                    self._controlled_stop_source_identity
                                )
                                if (
                                    controlled_identity is None
                                    or controlled_identity[0] != generation
                                    or controlled_identity[1] > sequence
                                ):
                                    raise NormalStopWireError(
                                        "normal-stop policy identity is not bound "
                                        "to the braking generation"
                                    )
                            else:
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
                            self._normal_stop_policy_braking = policy_braking
                    except ValueError:
                        if policy_candidate:
                            # A structurally valid but stale live stop still
                            # strengthens the actuator boundary before its
                            # identity is rejected.
                            try:
                                self._stop_navigation_at_boundary(
                                    source="normal_stop_begin",
                                    publish_telemetry=False,
                                )
                            except Exception as exc:
                                raise _NormalStopBoundaryError(str(exc)) from exc
                            self._static_hold_active = True
                        raise

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
            if getattr(self, "_normal_stop_policy_braking", False):
                authority = self._reference_authority.snapshot
                controlled_identity = getattr(
                    self, "_controlled_stop_source_identity", None
                )
                if (
                    not isinstance(controlled_identity, tuple)
                    or len(controlled_identity) != 2
                    or controlled_identity[0] != identity.control_generation
                    or controlled_identity[1] > identity.gait_source_sequence
                    or not getattr(
                        self, "_controlled_stop_reference_installed", False
                    )
                    or authority.phase != "GRANTED"
                    or authority.generation != controlled_identity[0]
                    or authority.source_sequence != controlled_identity[1]
                ):
                    raise NormalStopWireError(
                        "stationary acknowledgement requires the fresh IDLE "
                        "reference to cross DDS"
                    )
                # GaitAdapter stationarity proves that the base stopped; it does
                # not prove that an untethered humanoid can stand after learned
                # control is removed. Keep the continuous IDLE policy on DDS
                # until Plant independently confirms its safety-stop latch.
                # Freezing here creates an unsupported interval between this RPC
                # and the later Plant RPC, which made the live model fall even
                # though the navigation action had stopped successfully.
            # The startup/pre-grant branch already owns an exact static hold.
            # The learned-policy branch deliberately remains dynamic until the
            # correlated Plant acknowledgement reaches _handle_normal_stop_plant.
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
        static.  For a learned-policy stop those facts are established only
        after the Plant latch is externally confirmed; startup/pre-grant stops
        may already carry them from the begin phase.
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
                if getattr(self, "_normal_stop_policy_braking", False):
                    # Plant is already physically latched at this point. It is
                    # now safe to replace the learned IDLE balance controller
                    # with the exact last successful DDS target and revoke its
                    # reference authority. Never perform this transition on the
                    # earlier stationary acknowledgement: without the Plant
                    # latch, a static joint target cannot balance the humanoid.
                    try:
                        self._stop_navigation_at_boundary(
                            source="normal_stop_plant_confirmed",
                            publish_telemetry=False,
                        )
                    except Exception as exc:
                        raise _NormalStopLocalHoldError(str(exc)) from exc
                    self._static_hold_active = True
                # Snapshot the now-revoked reference telemetry while support is
                # serialized, before entering the terminal linearization lock;
                # the short commit below performs no planner/guard/handoff calls.
                planner_telemetry = self._planner_telemetry_fields()
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
                    # Publish the complete externally observed SAFE_HOLD in the
                    # same support/terminal critical section as DRIVER_CONFIRMED.
                    # An older rearm publisher revalidates these owner fields
                    # under _support_lock and therefore cannot write IDLE later.
                    self.telemetry.update(
                        nav2_state="safety_stop",
                        sonic_mode="IDLE",
                        sonic_speed_mps=0.0,
                        sonic_policy_speed_mps=0.0,
                        navigation_task_active=False,
                        last_cmd_vx=0.0,
                        last_cmd_vy=0.0,
                        last_cmd_wz=0.0,
                        static_hold_active=True,
                        motion_handoff_active=False,
                        motion_handoff_ratio=0.0,
                        plant_support_mode="safety_stop",
                        plant_support_requested_mode="safety_stop",
                        plant_support_confirmed_mode="safety_stop",
                        plant_support_stop_pending=False,
                        plant_support_safety_latched=True,
                        plant_support_stop_generation=(
                            identity.control_generation
                        ),
                        **planner_telemetry,
                    )

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

    @staticmethod
    def _arm_evidence_request_identity(
        payload: Mapping[str, object],
    ) -> tuple[UUID, bytes]:
        request_id = payload.get("request_id")
        try:
            parsed = UUID(str(request_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("arm evidence request_id must be a UUID") from exc
        if str(parsed) != request_id:
            raise ValueError("arm evidence request_id must be canonical")
        try:
            canonical = json.dumps(
                dict(payload),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("arm evidence request is not canonical JSON") from exc
        return parsed, canonical

    def _arm_evidence_activation_binding_locked(self) -> tuple[int, int]:
        activation = self._policy_activation
        if activation.phase not in {
            "UNPREPARED",
            "UNPREPARED_SUPPORT_OFF",
            "PREPARED",
        }:
            raise RuntimeError(
                "arm evidence cannot begin during a policy activation transition"
            )
        policy_epoch = (
            activation.epoch
            if activation.prepared
            else activation.epoch + 1
        )
        support = activation.provenance["support_release"]
        if not isinstance(support, Mapping):
            raise RuntimeError("policy activation support provenance is unavailable")
        confirmed_epoch = support.get("confirmed_epoch")
        if confirmed_epoch is None:
            support_epoch = policy_epoch
        elif (
            isinstance(confirmed_epoch, bool)
            or not isinstance(confirmed_epoch, int)
            or confirmed_epoch <= 0
            or confirmed_epoch > policy_epoch
        ):
            raise RuntimeError("policy activation support epoch is invalid")
        else:
            support_epoch = confirmed_epoch
        if activation.prepared and confirmed_epoch is None:
            raise RuntimeError(
                "prepared SONIC policy lacks a confirmed fall-arrest release"
            )
        return policy_epoch, support_epoch

    def _require_arm_evidence_begin_locked(self) -> tuple[int, int]:
        if self._fault_latched.is_set() or self._stop.is_set():
            raise RuntimeError("arm evidence cannot begin while SONIC is stopping")
        if not self._motion_ready.is_set():
            raise RuntimeError("arm evidence requires a motion-ready SONIC driver")
        checked_at = self._wall_clock()
        for label, observed_at, threshold in (
            ("LowState", self._last_lowstate_wall_s, self.config.plant_lowstate_stale_s),
            (
                "control step",
                self._last_control_step_wall_s,
                self.config.sonic_control_step_stale_s,
            ),
        ):
            if observed_at is None or checked_at - observed_at > float(threshold):
                raise RuntimeError(f"arm evidence requires a fresh {label}")
        authority = self._reference_authority.snapshot
        static_idle_boundary = bool(
            self._intent_ingress.armed
            and self._intent_ingress.staged_control_arm_generation is None
            and self._plant_support_mode == "idle"
            and not self._plant_support_stop_pending
            and not self._plant_support_safety_latched
            and self._static_hold_active
            and not self._navigation_task_active
            and self.guard.command_zero
            and not self._normal_stop_fences_motion_locked()
            and authority.phase == "REVOKED"
            and not authority.actuator_authority
            and self._last_published_targets is not None
        )
        if not (
            static_idle_boundary
            or self._arm_evidence_balanced_idle_boundary_locked()
        ):
            raise RuntimeError(
                "arm evidence requires a freshly armed stationary IDLE boundary"
            )
        return self._arm_evidence_activation_binding_locked()

    def _arm_evidence_balanced_idle_boundary_locked(self) -> bool:
        """Prove that learned IDLE still owns the stationary actuator boundary.

        Successful humanoid actions remain on the learned IDLE controller so
        the untethered model keeps balancing.  Evidence archival must not
        require a static joint freeze merely to declare that action complete.
        This predicate mirrors the driver-owned facts committed by
        :meth:`_complete_balanced_policy_idle_locked`; it is intentionally
        stronger than the telemetry flag sampled by the acceptance runner.
        """

        authority = self._reference_authority.snapshot
        activation = self._policy_activation
        controller = getattr(self, "_controller", None)
        identity = getattr(self, "_controlled_stop_source_identity", None)
        return bool(
            getattr(self, "_balanced_policy_idle_active", False)
            and self._intent_ingress.armed
            and not self._intent_ingress.braking
            and self._intent_ingress.staged_control_arm_generation is None
            and not self._normal_stop_fences_motion_locked()
            and not self._process_stop_fences_motion_locked()
            and self._plant_support_mode == "idle"
            and not self._plant_support_stop_pending
            and not self._plant_support_safety_latched
            and not self._static_hold_active
            and getattr(self, "_safe_hold_targets", None) is None
            and not self._navigation_task_active
            and self.guard.target_zero
            and self.guard.command_zero
            and authority.phase == "GRANTED"
            and authority.actuator_authority
            and isinstance(identity, tuple)
            and len(identity) == 2
            and authority.generation == identity[0]
            and authority.source_sequence == identity[1]
            and getattr(self, "_reference_source_identity", None) == identity
            and getattr(self, "_reference_request_pending", True) is False
            and getattr(self, "_controlled_stop_active", False)
            and getattr(self, "_controlled_stop_reference_installed", False)
            and controller is not None
            and int(getattr(controller.ms, "mode", -1)) == 0
            and not bool(getattr(controller.ms, "has_movement", True))
            and activation.prepared
            and self._last_published_targets is not None
        )

    def _require_arm_evidence_finalize_locked(
        self,
        identity: Mapping[str, object],
    ) -> None:
        if self._fault_latched.is_set() or self._stop.is_set():
            raise RuntimeError("faulted or stopping SONIC evidence cannot succeed")
        normal_stop = self._normal_stop.snapshot
        authority = self._reference_authority.snapshot
        activation = self._policy_activation
        support = activation.provenance["support_release"]
        confirmed_epoch = (
            support.get("confirmed_epoch")
            if isinstance(support, Mapping)
            else None
        )
        static_normal_stop_boundary = bool(
            normal_stop.phase == "DRIVER_CONFIRMED"
            and not self._intent_ingress.armed
            and self._intent_ingress.staged_control_arm_generation is None
            and self._plant_support_safety_latched
            and not self._plant_support_stop_pending
            and self._plant_support_mode == "safety_stop"
            and self._static_hold_active
            and not self._navigation_task_active
            and self.guard.command_zero
            and authority.phase == "REVOKED"
            and not authority.actuator_authority
        )
        if (
            not activation.prepared
            or identity.get("policyActivationEpoch") != activation.epoch
            or identity.get("supportReleaseConfirmedEpoch") != confirmed_epoch
            or not (
                static_normal_stop_boundary
                or self._arm_evidence_balanced_idle_boundary_locked()
            )
        ):
            raise RuntimeError(
                "arm evidence requires an exact successful stationary boundary "
                "(normal-stop boundary or balanced-policy IDLE boundary)"
            )

    def _handle_arm_evidence_request(
        self,
        payload: dict[str, Any],
    ) -> dict[str, object]:
        operation = payload.get("type")
        fields = {
            "arm_limiter_evidence_begin": {
                "type",
                "request_id",
                "case_id",
                "plant_boot_id",
                "plant_session_id",
            },
            "arm_limiter_evidence_finalize": {
                "type",
                "request_id",
                "case_id",
            },
            "arm_limiter_evidence_abort": {
                "type",
                "request_id",
                "case_id",
            },
            "arm_limiter_evidence_get": {
                "type",
                "request_id",
                "case_id",
            },
        }
        expected_fields = fields.get(operation)
        if expected_fields is None or set(payload) != expected_fields:
            raise ValueError("arm evidence request fields do not match v1")
        request_id, canonical = self._arm_evidence_request_identity(payload)
        replays = self._arm_evidence_request_replays
        replay = replays.get(request_id)
        if replay is not None:
            previous, response = replay
            if previous != canonical:
                raise RuntimeError(
                    "arm evidence request_id was reused for another operation"
                )
            return copy.deepcopy(response)
        if len(replays) >= 256:
            raise RuntimeError("arm evidence replay ledger is full")

        case_id = payload.get("case_id")
        if operation == "arm_limiter_evidence_begin":
            plant_identity = read_plant_identity(
                "127.0.0.1",
                self.config.plant_port,
            )
            if (
                payload.get("plant_boot_id")
                != plant_identity["producerBootId"]
                or payload.get("plant_session_id")
                != plant_identity["sessionId"]
            ):
                raise RuntimeError(
                    "arm evidence Plant identity does not match the live process"
                )
            with self._arm_evidence_boundary_lock:
                with self._control_condition:
                    with self._support_lock:
                        with self._terminal_fault_lock:
                            policy_epoch, support_epoch = (
                                self._require_arm_evidence_begin_locked()
                            )
                            self._arm_evidence.begin(
                                case_id=case_id,
                                policy_activation_epoch=policy_epoch,
                                support_release_confirmed_epoch=support_epoch,
                                plant_boot_id=plant_identity["producerBootId"],
                                plant_session_id=plant_identity["sessionId"],
                            )
            response: dict[str, object] = {
                "status": "recording",
                "case_id": self._arm_evidence.active_case_id,
                "policy_activation_epoch": policy_epoch,
                "support_release_confirmed_epoch": support_epoch,
                "plant_boot_id": plant_identity["producerBootId"],
                "plant_session_id": plant_identity["sessionId"],
                "sonic_driver_boot_id": str(self._driver_boot_id),
                "sonic_driver_session_id": str(self._driver_session_id),
            }
        elif operation == "arm_limiter_evidence_finalize":
            plant_identity = read_plant_identity(
                "127.0.0.1",
                self.config.plant_port,
            )
            with self._arm_evidence_boundary_lock:
                identity = self._arm_evidence.active_identity
                if identity is None or identity.get("caseId") != case_id:
                    completed = self._arm_evidence.completed(case_id=case_id)
                    if completed is None:
                        raise ArmLimiterEvidenceError(
                            "no matching arm evidence session is active"
                        )
                    response = {"status": "complete", "evidence": completed}
                else:
                    if (
                        identity.get("plantBootId")
                        != plant_identity["producerBootId"]
                        or identity.get("plantSessionId")
                        != plant_identity["sessionId"]
                    ):
                        self._arm_evidence.fail_active(
                            "Plant identity changed during arm evidence"
                        )
                    with self._control_condition:
                        with self._support_lock:
                            with self._terminal_fault_lock:
                                self._require_arm_evidence_finalize_locked(
                                    identity
                                )
                                completed = self._arm_evidence.finalize(
                                    case_id=case_id
                                )
                    response = {"status": "complete", "evidence": completed}
        elif operation == "arm_limiter_evidence_abort":
            with self._arm_evidence_boundary_lock:
                aborted = self._arm_evidence.abort(case_id=case_id)
            response = {"status": "aborted", "aborted": aborted}
        else:
            completed = self._arm_evidence.completed(case_id=case_id)
            if completed is not None:
                response = {"status": "complete", "evidence": completed}
            elif self._arm_evidence.active_case_id == case_id:
                response = {"status": "recording", "evidence": None}
            else:
                raise ArmLimiterEvidenceError(
                    "no matching arm evidence session exists"
                )

        replays[request_id] = (canonical, copy.deepcopy(response))
        return response

    def accept_json(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if (
            isinstance(payload.get("type"), str)
            and str(payload["type"]).startswith("arm_limiter_evidence_")
        ):
            return self._handle_arm_evidence_request(payload)
        if payload.get("type") == "motion_fault_latch":
            if set(payload) != {"type", "request_id", "reason"}:
                raise ValueError("motion fault latch fields do not match v1")
            request_id = payload.get("request_id")
            try:
                parsed_request_id = UUID(str(request_id))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(
                    "motion fault latch request_id must be a UUID"
                ) from exc
            if str(parsed_request_id) != str(request_id):
                raise ValueError(
                    "motion fault latch request_id must be canonical"
                )
            reason = str(payload.get("reason", "")).strip()
            if not reason or len(reason) > 512:
                raise ValueError(
                    "motion fault latch reason must contain 1..512 characters"
                )
            replays = getattr(self, "_motion_fault_latch_replays", None)
            if replays is None:
                replays = {}
                self._motion_fault_latch_replays = replays
            replay = replays.get(parsed_request_id)
            if replay is not None:
                previous_reason, previous_response = replay
                if previous_reason != reason:
                    raise RuntimeError(
                        "motion fault latch request_id was reused with another reason"
                    )
                return copy.deepcopy(previous_response)
            self._latch_motion_fault(reason)
            with self._terminal_fault_lock:
                if not (
                    self._fault_latched.is_set()
                    and self._recoverable_motion_fault
                    and self._terminal_fault_reason == reason
                ):
                    raise RuntimeError(
                        "SONIC did not confirm a recoverable motion fault"
                    )
            response: dict[str, object] = {
                "status": "fault",
                "mode": "safety_stop",
                "recoverable": True,
                "reason": reason,
            }
            replays[parsed_request_id] = (reason, copy.deepcopy(response))
            return response
        if payload.get("type") == "motion_fault_reset":
            return self._handle_motion_fault_reset(payload)
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
                # Health reports the latest admitted Nav2 target, not the
                # previous control tick's filtered value.  On STOPPING the
                # target becomes zero atomically with driver-owned braking,
                # while ``command_zero`` becomes true on the next 20 ms tick.
                # Treating that bounded tail as a non-zero external command
                # caused a false GRANTED-authority contract fault in live
                # motion.  Final static-stop proof below/elsewhere still uses
                # the stronger ``guard.command_zero`` predicate.
                nav2_command_zero = self.guard.target_zero
                authority = self._reference_authority.snapshot
                # STOPPING becomes actuator ownership as soon as the driver
                # admits the zero-command braking intent.  The correlated
                # normal-stop service may arrive a few scheduler slices later;
                # requiring that second transaction here creates an impossible
                # GRANTED + zero-command health snapshot during the legitimate
                # policy-braking interval.
                balanced_policy_idle_active = bool(
                    getattr(self, "_balanced_policy_idle_active", False)
                    and self._intent_ingress.armed
                    and not self._intent_ingress.braking
                    and authority.phase == "GRANTED"
                    and authority.actuator_authority
                    and not static_hold_active
                    and plant_support_mode == "idle"
                )
                controlled_policy_stop_active = bool(
                    getattr(self, "_controlled_stop_active", False)
                    and not navigation_task_active
                    and nav2_command_zero
                    and (
                        (
                            self._intent_ingress.braking
                            and not self._intent_ingress.armed
                        )
                        or balanced_policy_idle_active
                    )
                )
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
                "driver_session_id": str(self._driver_session_id),
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
                "controlled_policy_stop_active": (
                    controlled_policy_stop_active
                ),
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
                    (
                        "SONIC motion fault requires explicit reset"
                        if getattr(self, "_recoverable_motion_fault", False)
                        else "SONIC terminal fault requires an explicit process restart"
                    )
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
                if (
                    payload.get("source") == "robot_agent_safe_stop"
                    and not self._normal_stop_fences_motion_locked()
                ):
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
        gait_state = payload.get("state")
        terminal_gait_fault = (
            payload.get("type") == "sonic_intent"
            and gait_state == "FAULT"
        )
        stop_boundary_applied = False
        controlled_brake_applied = False
        balanced_idle_applied = False
        balanced_idle_pending = False
        stop_boundary_error: Exception | None = None
        try:
            if control_arm:
                # Timestamp and support predicates are sampled under their two
                # owner locks, and ingress is staged in that same critical
                # section.  No generation/sequence mutation can precede this
                # fresh, complete safe-hold gate.
                with self._control_condition:
                    with self._support_lock:
                        with self._terminal_fault_lock:
                            self._require_fresh_control_arm_health_locked(
                                phase="control arm precheck"
                            )
                            decision = self._intent_ingress.accept(
                                payload,
                                defer_control_arm=True,
                            )
            else:
                with self._support_lock:
                    if (
                        self._normal_stop_fences_motion_locked()
                        and not terminal_gait_fault
                    ):
                        if gait_state in SonicIntentIngress._MOTION_STATES:
                            raise RuntimeError(
                                "correlated normal stop rejects new gait/control "
                                "authority"
                            )
                        # Periodic STOPPING/DISARMED/IDLE envelopes are obsolete
                        # once the strict transaction owns the stop sequence.
                        # Check and ignore under the ingress owner lock so a
                        # normal-stop begin cannot linearize between the phase
                        # decision and mutation of generation/sequence.
                        return None
                    decision = self._intent_ingress.accept(payload)
                    if decision.action == "brake":
                        controlled_brake_applied = (
                            self._begin_controlled_policy_stop_locked(decision)
                        )
                        if not controlled_brake_applied:
                            try:
                                self._stop_navigation_at_boundary(
                                    source="gait_adapter_pregrant_stop",
                                    publish_telemetry=False,
                                )
                            except Exception as exc:
                                stop_boundary_error = exc
                            stop_boundary_applied = True
                    elif decision.action == "stop":
                        controlled_idle_completion = bool(
                            decision.state == "IDLE"
                            and getattr(self, "_controlled_stop_active", False)
                        )
                        if controlled_idle_completion:
                            balanced_idle_applied = (
                                self._complete_balanced_policy_idle_locked(
                                    decision
                                )
                            )
                            if not balanced_idle_applied:
                                balanced_idle_pending = (
                                    self._balanced_policy_idle_completion_pending_locked(
                                        decision
                                    )
                                )
                        if not balanced_idle_applied and not balanced_idle_pending:
                            try:
                                self._stop_navigation_at_boundary(
                                    source=(
                                        "motion_fault"
                                        if decision.state == "FAULT"
                                        else "gait_adapter"
                                    ),
                                    publish_telemetry=False,
                                )
                            except Exception as exc:
                                stop_boundary_error = exc
                            if (
                                stop_boundary_error is None
                                and controlled_idle_completion
                            ):
                                stop_boundary_error = RuntimeError(
                                    "acknowledged IDLE lacked an executable "
                                    "balanced-policy boundary"
                                )
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
        if not control_arm and decision.action == "brake":
            if stop_boundary_error is not None:
                self._latch_terminal_fault(
                    "controlled-stop fallback boundary fault: "
                    + str(stop_boundary_error),
                    boundary_already_stopped=True,
                )
                return None
            self.telemetry.update(
                nav2_state=(
                    "braking" if controlled_brake_applied else "safety_stop"
                ),
                navigation_task_active=False,
                last_cmd_vx=0.0,
                last_cmd_vy=0.0,
                last_cmd_wz=0.0,
                static_hold_active=self._static_hold_active,
                balanced_policy_idle_active=False,
                last_error=decision.reason,
                **self._planner_telemetry_fields(),
            )
            return None
        if not control_arm and decision.action == "stop":
            # Ingress acceptance and the in-memory actuator/reference stop are
            # one support-lock transaction. Plant/telemetry I/O follows only
            # after control can no longer consume the accepted stale intent.
            if balanced_idle_applied:
                self.telemetry.update(
                    nav2_state="idle",
                    sonic_mode="IDLE",
                    navigation_task_active=False,
                    last_cmd_vx=0.0,
                    last_cmd_vy=0.0,
                    last_cmd_wz=0.0,
                    static_hold_active=False,
                    balanced_policy_idle_active=True,
                    last_error=None,
                    **self._planner_telemetry_fields(),
                )
                return None
            if balanced_idle_pending:
                # Stationarity won the race with the first executable IDLE
                # policy DDS fact.  Keep the continuous learned controller
                # transaction alive; the next validated IDLE envelope will
                # commit once GRANTED.  Existing planner/gait/terminal
                # deadlines retain the bounded fail-closed behavior.
                self.telemetry.update(
                    nav2_state="braking",
                    sonic_mode="IDLE",
                    navigation_task_active=False,
                    last_cmd_vx=0.0,
                    last_cmd_vy=0.0,
                    last_cmd_wz=0.0,
                    static_hold_active=self._static_hold_active,
                    balanced_policy_idle_active=False,
                    last_error=None,
                    **self._planner_telemetry_fields(),
                )
                return None
            assert stop_boundary_applied
            if decision.state == "FAULT" or stop_boundary_error is not None:
                terminal_reason = decision.reason or "gait adapter entered FAULT"
                if stop_boundary_error is not None:
                    terminal_reason = (
                        f"{terminal_reason}; gait stop boundary fault: "
                        f"{stop_boundary_error}"
                    )
                if decision.state == "FAULT" and stop_boundary_error is None:
                    self._latch_motion_fault(
                        terminal_reason,
                        boundary_already_stopped=True,
                    )
                else:
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
                balanced_policy_idle_active=False,
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
                        with self._terminal_fault_lock:
                            self._require_fresh_control_arm_health_locked(
                                phase="control arm post-rearm check"
                            )
                            self._commit_staged_plant_support_rearm_locked(
                                decision.generation
                            )
                            # This is the first boundary at which a strict
                            # runtime clock discontinuity can invalidate
                            # authorized policy/history state.  It is
                            # monotonic for the lifetime of this process.
                            self._runtime_clock_committed = True
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
                if getattr(self, "_balanced_policy_idle_active", False):
                    authority = self._reference_authority.snapshot
                    idle_identity = self._controlled_stop_source_identity
                    if (
                        idle_identity is None
                        or decision.generation != idle_identity[0]
                        or authority.phase != "GRANTED"
                        or authority.generation != idle_identity[0]
                        or authority.source_sequence != idle_identity[1]
                        or not self._controlled_stop_reference_installed
                    ):
                        raise RuntimeError(
                            "balanced IDLE authority changed before motion admission"
                        )
                    self._balanced_policy_idle_active = False
                    self._controlled_stop_active = False
                    self._controlled_stop_source_identity = None
                    self._controlled_stop_reference_installed = False
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
            self._arm_control_stall_diagnostic()
            self._reset_navigation_task_peaks_hot()
            self._update_telemetry_hot(
                navigation_task_active=True,
                balanced_policy_idle_active=False,
            )
        self._update_telemetry_hot(
            nav2_state="commanding",
            last_cmd_vx=decision.vx,
            last_cmd_vy=decision.vy,
            last_cmd_wz=decision.wz,
            last_error=None,
        )
        return None


def _run_control_stall_diagnostic_watchdog(
    driver: SonicDriver,
    stop_event: threading.Event,
) -> None:
    """Capture one pre-fault stack snapshot without changing fault policy."""

    threshold_s = max(
        0.020,
        0.60 * float(driver.config.sonic_control_step_stale_s),
    )
    armed = driver._control_stall_diagnostic_armed
    dumped = driver._control_stall_diagnostic_dumped
    while not stop_event.is_set():
        if not armed.wait(timeout=0.25):
            continue
        if stop_event.wait(timeout=0.005):
            return
        if not armed.is_set() or dumped.is_set():
            continue
        last_published_at = driver._last_control_step_wall_s
        if last_published_at is None:
            continue
        checked_at = driver._wall_clock()
        age_s = max(0.0, checked_at - last_published_at)
        if age_s < threshold_s:
            continue
        dumped.set()
        logger.error(
            "capturing all SONIC thread stacks before control stale fault "
            "age_s=%.6f diagnostic_threshold_s=%.6f stage=%s",
            age_s,
            threshold_s,
            str(getattr(driver, "_control_loop_stage", "unknown"))[:96],
        )
        faulthandler.dump_traceback(all_threads=True)


def _handle_command_connection(driver: SonicDriver, conn) -> None:
    """Serve one bounded-lifetime 8793 client without changing its semantics."""

    connection_started_at = time.monotonic()
    receive_completed_at = connection_started_at
    parse_completed_at = connection_started_at
    accept_elapsed_s = 0.0
    send_elapsed_s = 0.0
    message_types: list[str] = []
    data = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        receive_completed_at = time.monotonic()
        messages, _ = parse_lines(
            data + (b"" if data.endswith(b"\n") else b"\n")
        )
        parse_completed_at = time.monotonic()
        for message in messages:
            message_type = str(message.get("type", "missing"))
            message_types.append(
                " ".join(message_type.replace("\x00", "").split())[:64]
                or "empty"
            )
            request_id = message.get("request_id")
            accept_started_at = time.monotonic()
            try:
                response = driver.accept_json(message)
                accept_completed_at = time.monotonic()
                accept_elapsed_s += max(
                    0.0,
                    accept_completed_at - accept_started_at,
                )
                if request_id is not None:
                    acknowledgement = {
                        "request_id": request_id,
                        "ok": True,
                        **(response or {}),
                    }
                    send_started_at = time.monotonic()
                    conn.sendall(
                        (
                            json.dumps(acknowledgement, separators=(",", ":"))
                            + "\n"
                        ).encode("utf-8")
                    )
                    send_elapsed_s += max(
                        0.0,
                        time.monotonic() - send_started_at,
                    )
            except Exception as exc:
                accept_completed_at = time.monotonic()
                accept_elapsed_s += max(
                    0.0,
                    accept_completed_at - accept_started_at,
                )
                if request_id is None:
                    raise
                send_started_at = time.monotonic()
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
                send_elapsed_s += max(
                    0.0,
                    time.monotonic() - send_started_at,
                )
    finally:
        connection_completed_at = time.monotonic()
        total_s = max(0.0, connection_completed_at - connection_started_at)
        if total_s > 0.020:
            logger.warning(
                "slow SONIC command stages types=%s total_s=%.6f recv_s=%.6f "
                "parse_s=%.6f accept_s=%.6f send_s=%.6f",
                ",".join(message_types) or "unparsed",
                total_s,
                max(0.0, receive_completed_at - connection_started_at),
                max(0.0, parse_completed_at - receive_completed_at),
                accept_elapsed_s,
                send_elapsed_s,
            )


def run_driver(config: SimConfig) -> None:
    # The package CLI calls ``run_driver`` directly rather than this module's
    # standalone ``main`` function.  Configure the process logger here so
    # safety faults and bounded-latency diagnostics are never silently lost
    # when SONIC is launched through ``python -m hide_mujoco_sonic sonic``.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    telemetry = TelemetryStore(
        TelemetryState(
            robot_id=config.robot_id,
            ros_domain_id=config.ros_domain_id,
            unitree_domain_id=config.unitree_domain_id,
        )
    )
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
    threading.Thread(
        target=_run_control_stall_diagnostic_watchdog,
        args=(driver, command_server_stop),
        daemon=True,
        name="sonic-control-stall-diagnostic",
    ).start()
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
