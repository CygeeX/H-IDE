"""ROS 2/Nav2 bridge for the isolated simulation domain.

The bridge subscribes to Nav2's velocity output and forwards only bounded JSON
commands to the SONIC driver.  Ground-truth pose and camera publication are
kept in the plant/observation adapter; this node never invents localization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import os
import threading
import time
from typing import Callable, Iterable, Mapping, Protocol
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
    HealthFaultDecision,
    HealthTransportDecision,
    MotionFaultLatch,
    MotionSafetyFault,
    SonicHealthFaultGate,
    SonicHealthTransportGate,
    SonicHealth,
    motion_fault_qos,
    require_fresh_motion_health,
    sonic_health_probe,
)
from .normal_stop_coordinator import BridgeNormalStopCoordinator
from .render_workers import MAX_CAPTURE_AGE_NS
from .safety import (
    BodyStabilityGuard,
    GoalProgressGuard,
    MotionDirectionGuard,
    RotationDirectionGuard,
    TranslationHeadingGuard,
)
from .transport import (
    JsonResponseContractError,
    JsonResponseTimeoutError,
    JsonResponseUnavailableError,
    latch_plant_motion_fault,
    request_json,
    send_json,
    set_plant_support,
)


class RosBridgeUnavailable(RuntimeError):
    pass


BRIDGE_HEALTH_SCHEMA_VERSION = "h-ide.ros-bridge-health/v1"
BRIDGE_HEALTH_MAX_PROGRESS_AGE_S = 0.10
RETAIN_SONIC_ON_AUTOMATIC_FAULT_ENV = (
    "HIDE_SIM_RETAIN_SONIC_ON_AUTOMATIC_FAULT"
)


def _retain_sonic_on_automatic_fault_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return the explicit simulation-only automatic-fault experiment.

    The default remains the reviewed fail-closed behavior.  The workstation
    launcher opts into the experiment with ``1`` and can roll back without a
    source change by launching with ``0``.  Reject ambiguous spellings so a
    typo cannot silently select either control policy.
    """

    source = os.environ if environ is None else environ
    value = source.get(RETAIN_SONIC_ON_AUTOMATIC_FAULT_ENV, "0")
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(
        f"{RETAIN_SONIC_ON_AUTOMATIC_FAULT_ENV} must be exactly 0 or 1"
    )


def _gait_observation_timeout_s(observation_hz: object) -> float:
    """Match gait freshness to the authoritative capture validity window.

    Odometry is derived from the same atomic MuJoCo capture as RGB-D.  A
    single bounded HTTP miss must not terminally remove the learned balance
    controller while that source generation is still valid.  Conversely, no
    pose older than the capture contract's 500 ms ceiling may authorize
    continued motion.
    """

    if isinstance(observation_hz, bool):
        raise ValueError("observation_hz must be finite and positive")
    try:
        hz = float(observation_hz)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("observation_hz must be finite and positive") from exc
    if not math.isfinite(hz) or hz <= 0.0:
        raise ValueError("observation_hz must be finite and positive")
    capture_validity_s = MAX_CAPTURE_AGE_NS / 1_000_000_000.0
    return max(capture_validity_s, 3.0 / hz)


def _resident_policy_reentry_generation(
    *,
    health: SonicHealth,
    gait_state: GaitState,
    bridge_generation: int | None,
    gait_generation: int | None,
    observation_generation: int | None,
) -> int | None:
    """Return the exact already-armed generation, or reject split identity.

    A physical safe hold still follows the ordinary rearm path.  Only the
    driver's strict balanced-policy-IDLE health shape can reuse a generation,
    and the ROS bridge, gait adapter, and latest measured observation must all
    agree with SONIC's matched reference generation.
    """

    if not health.confirmed_balanced_policy_idle:
        return None
    if gait_state is not GaitState.IDLE:
        raise ValueError("resident balanced policy requires bridge gait IDLE")
    authority_generation = health.reference_authority_generation
    identities = (
        authority_generation,
        bridge_generation,
        gait_generation,
        observation_generation,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in identities
    ):
        raise ValueError("resident balanced policy generation is incomplete")
    assert authority_generation is not None
    if any(value != authority_generation for value in identities[1:]):
        raise ValueError("resident balanced policy generation identity mismatch")
    return authority_generation


class BridgeRuntimeHealth:
    """Read-only evidence backed only by successful SONIC probe work.

    Reading a health snapshot never advances ``sequence``.  That counter is
    committed only after the bridge has received and parsed a strict,
    correlated SONIC health response, so a diagnostics poll cannot manufacture
    liveness while the bridge callback loop is stalled.
    """

    def __init__(
        self,
        *,
        boot_id: UUID,
        session_id: UUID,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(boot_id, UUID) or not isinstance(session_id, UUID):
            raise TypeError("bridge health identities must be UUIDs")
        self._boot_id = boot_id
        self._session_id = session_id
        self._clock = clock
        self._lock = threading.RLock()
        self._sequence = 0
        self._last_progress_at_s: float | None = None
        self._sonic_health: SonicHealth | None = None

    @staticmethod
    def _monotonic(value: float, field: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
        return result

    def observe_sonic_probe(
        self,
        health: SonicHealth,
        *,
        observed_at_s: float | None = None,
    ) -> None:
        if not isinstance(health, SonicHealth):
            raise TypeError("health must be a SonicHealth")
        observed = self._monotonic(
            self._clock() if observed_at_s is None else observed_at_s,
            "observed_at_s",
        )
        with self._lock:
            if (
                self._last_progress_at_s is not None
                and observed < self._last_progress_at_s
            ):
                raise ValueError("bridge health progress clock moved backwards")
            self._sequence += 1
            self._last_progress_at_s = observed
            self._sonic_health = health

    def snapshot(
        self,
        *,
        gait_state: GaitState,
        control_generation: int,
        bridge_fault: str | None,
        health_transport_pending: bool,
        health_transport_confirmed: bool,
        health_fault_pending: bool,
        health_quarantined: bool,
        now_s: float | None = None,
    ) -> dict[str, object]:
        if not isinstance(gait_state, GaitState):
            raise TypeError("gait_state must be a GaitState")
        if (
            isinstance(control_generation, bool)
            or not isinstance(control_generation, int)
            or control_generation < 0
        ):
            raise ValueError("control_generation must be a non-negative integer")
        if not all(
            isinstance(value, bool)
            for value in (
                health_transport_pending,
                health_transport_confirmed,
                health_fault_pending,
                health_quarantined,
            )
        ):
            raise TypeError("bridge health state flags must be booleans")
        if bridge_fault is not None:
            bridge_fault = str(bridge_fault).strip()
            if not bridge_fault:
                raise ValueError("bridge_fault must be non-empty or null")
        sampled = self._monotonic(
            self._clock() if now_s is None else now_s,
            "now_s",
        )
        with self._lock:
            sequence = self._sequence
            progressed = self._last_progress_at_s
            health = self._sonic_health
        progress_age_s = None if progressed is None else sampled - progressed
        if progress_age_s is not None and progress_age_s < 0.0:
            raise ValueError("bridge health sample preceded its latest progress")
        stale = (
            progress_age_s is None
            or progress_age_s > BRIDGE_HEALTH_MAX_PROGRESS_AGE_S
        )
        sonic_fault = None
        if health is not None and health.terminal_fault:
            sonic_fault = health.reason or "SONIC terminal fault"
        fault = bridge_fault or sonic_fault
        healthy = bool(
            sequence > 0
            and health is not None
            and health.motion_ready
            and not health.terminal_fault
            and not stale
            and not health_transport_pending
            and not health_transport_confirmed
            and not health_fault_pending
            and not health_quarantined
            and fault is None
        )
        return {
            "schemaVersion": BRIDGE_HEALTH_SCHEMA_VERSION,
            "healthy": healthy,
            "bootId": str(self._boot_id),
            "sessionId": str(self._session_id),
            "sequence": sequence,
            "lastProgressAtMonotonicS": progressed,
            "progressAgeS": progress_age_s,
            "sonicDriverBootId": (
                None if health is None else str(health.driver_boot_id)
            ),
            "sonicDriverSessionId": (
                None if health is None else str(health.driver_session_id)
            ),
            "sonicStatus": None if health is None else health.status,
            "gaitState": gait_state.value,
            "controlGeneration": control_generation,
            "healthTransportPending": health_transport_pending,
            "healthTransportConfirmed": health_transport_confirmed,
            "healthFaultPending": health_fault_pending,
            "healthQuarantined": health_quarantined,
            "fault": fault,
        }


class _SafetyTimerNode(Protocol):
    def create_timer(
        self,
        timer_period_sec: float,
        callback: Callable[[], None],
        *,
        callback_group: object,
        clock: object,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class _HealthProbeAttempt:
    """One health I/O attempt plus its transport-fence transition."""

    health: SonicHealth | None
    transport: HealthTransportDecision
    timeout_error: JsonResponseTimeoutError | None
    request_attempted: bool


@dataclass(frozen=True, slots=True)
class _ActionStatusObservation:
    active: bool
    observed_at_s: float


def _global_goal_progress_active(
    statuses: Mapping[str, _ActionStatusObservation | None],
) -> bool:
    """Bind global-goal regression checks to their owning Nav2 action.

    ``/plan`` is transient evidence and commonly outlives NavigateToPose.  It
    must never become authority over standalone behavior actions or the strict
    lateral runner.  FollowPath is intentionally absent because this product
    profile does not accept it for motion; NavigateToPose is the sole owner of
    global endpoint progress here.
    """

    navigation = statuses.get("navigate_to_pose")
    if (
        not isinstance(navigation, _ActionStatusObservation)
        or not navigation.active
    ):
        return False
    for name in ("spin", "drive_on_heading"):
        standalone = statuses.get(name)
        if isinstance(standalone, _ActionStatusObservation) and standalone.active:
            return False
    return True


def _actions_confirmed_inactive_after(
    statuses: Mapping[str, _ActionStatusObservation | None],
    *,
    fault_latched_at_s: float,
) -> bool:
    """Require post-fault negative evidence from every motion action server."""

    boundary = float(fault_latched_at_s)
    if not math.isfinite(boundary):
        raise ValueError("fault latch timestamp must be finite")
    if not statuses:
        return False
    return all(
        isinstance(observation, _ActionStatusObservation)
        and math.isfinite(observation.observed_at_s)
        and observation.observed_at_s >= boundary
        and not observation.active
        for observation in statuses.values()
    )


def _health_transport_command_action(
    *,
    fenced: bool,
    gait_state: GaitState,
    vx: float,
    vy: float,
    wz: float,
) -> str:
    """Return ``allow``, ``block``, or ``cancel_start`` for one Nav2 input."""

    if not isinstance(fenced, bool):
        raise TypeError("fenced must be a boolean")
    if not isinstance(gait_state, GaitState):
        raise TypeError("gait_state must be a GaitState")
    values = tuple(float(value) for value in (vx, vy, wz))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nav2 command must be finite")
    if not fenced or not any(value != 0.0 for value in values):
        return "allow"
    if gait_state is GaitState.STARTING:
        return "cancel_start"
    # An already-authorized gait may retain only its last accepted target for
    # the bounded confirmation window.  No new non-zero callback (including a
    # changed target while WALKING/TURNING) is consumed while health is unknown.
    return "block"


def _suppress_for_unknown_health_transport(
    fenced: bool,
    intent: SonicIntent,
) -> bool:
    """Fence every ordinary motion-bearing envelope, never a stop or FAULT."""

    if not isinstance(fenced, bool):
        raise TypeError("fenced must be a boolean")
    if not isinstance(intent, SonicIntent):
        raise TypeError("intent must be a SonicIntent")
    return bool(
        fenced
        and intent.state != GaitState.FAULT.value
        and intent.motion_enabled
    )


def _attempt_sonic_health_probe(
    request_id: UUID,
    *,
    request_health: Callable[[UUID], SonicHealth],
    transport_gate: SonicHealthTransportGate,
    clock: Callable[[], float] = time.monotonic,
) -> _HealthProbeAttempt:
    """Run one probe; only a typed socket timeout enters confirmation.

    :func:`request_json` wraps only socket connect/send/receive timeouts as
    :class:`JsonResponseTimeoutError`.  Bare application ``TimeoutError`` and
    connection/EOF/protocol/schema failures deliberately propagate to their
    immediate terminal classification.  A caller must fully parse the v2
    response inside ``request_health`` before this function can report strict
    success and clear an earlier transport fence.
    """

    if not isinstance(request_id, UUID):
        raise TypeError("request_id must be a UUID")
    started_at = float(clock())
    before = transport_gate.check_deadline(now_s=started_at)
    if before.confirmed_cause is not None:
        return _HealthProbeAttempt(
            health=None,
            transport=before,
            timeout_error=None,
            request_attempted=False,
        )
    try:
        health = request_health(request_id)
    except JsonResponseTimeoutError as exc:
        completed_at = float(clock())
        decision = transport_gate.observe_timeout(
            request_id=request_id,
            probe_started_at_s=started_at,
            probe_completed_at_s=completed_at,
        )
        return _HealthProbeAttempt(
            health=None,
            transport=decision,
            timeout_error=exc,
            request_attempted=True,
        )
    completed_at = float(clock())
    decision = transport_gate.observe_strict_success(
        request_id=request_id,
        probe_started_at_s=started_at,
        probe_completed_at_s=completed_at,
    )
    return _HealthProbeAttempt(
        health=health,
        transport=decision,
        timeout_error=None,
        request_attempted=True,
    )


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


def _observe_health_reference_authority(
    gait: GaitAdapter,
    health: SonicHealth,
    decision: HealthFaultDecision,
    *,
    now: float,
    transport_decision: HealthTransportDecision | None = None,
) -> bool:
    """Forward a correlated first-DDS grant only from a healthy snapshot."""

    if (
        (
            transport_decision is not None
            and (
                transport_decision.confirmed_cause is not None
                or transport_decision.pending is not None
            )
        )
        or
        decision.confirmed_cause is not None
        or decision.pending is not None
        or decision.quarantined_cause is not None
        or health.terminal_fault
        or not health.motion_ready
        or not health.matched_reference_actuator_authority
    ):
        return False
    generation = health.reference_authority_generation
    source_sequence = health.reference_authority_source_sequence
    reference_epoch = health.reference_authority_epoch
    dds_sequence = health.reference_authority_dds_target_sequence
    granted_at = health.reference_authority_granted_at_monotonic_s
    if (
        generation is None
        or source_sequence is None
        or reference_epoch <= 0
        or dds_sequence is None
        or granted_at is None
    ):
        raise ValueError("granted SONIC health lacks reference authority evidence")
    return gait.observe_reference_authority(
        generation=generation,
        source_sequence=source_sequence,
        reference_epoch=reference_epoch,
        dds_target_sequence=dds_sequence,
        granted_at_monotonic_s=granted_at,
        now=now,
    )


def _suppress_for_correlated_stop(
    coordinator: BridgeNormalStopCoordinator | None,
    intent: SonicIntent,
) -> bool:
    """Keep ordinary envelopes behind strict stop correlation, never FAULT."""

    return bool(
        coordinator is not None
        and intent.state != GaitState.FAULT.value
        and coordinator.blocks_generic_intent(intent.generation)
    )


def _create_safety_timers(
    node: _SafetyTimerNode,
    *,
    command_timeout_s: float,
    health_probe_period_s: float,
    watchdog: Callable[[], None],
    health_probe: Callable[[], None],
    callback_group: object,
    clock: object,
) -> tuple[object, object]:
    """Create elapsed-time safety timers on one explicitly supplied clock."""

    return (
        node.create_timer(
            min(command_timeout_s / 2, 0.1),
            watchdog,
            callback_group=callback_group,
            clock=clock,
        ),
        node.create_timer(
            health_probe_period_s,
            health_probe,
            callback_group=callback_group,
            clock=clock,
        ),
    )


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
    retain_sonic_on_automatic_fault = (
        _retain_sonic_on_automatic_fault_enabled()
    )
    try:
        import rclpy
        from action_msgs.msg import GoalStatusArray
        from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
        from nav_msgs.msg import Odometry, Path
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.clock import Clock
        from rclpy.clock_type import ClockType
        from rclpy.executors import MultiThreadedExecutor
        from std_msgs.msg import String
        from std_srvs.srv import Trigger
    except ImportError as exc:  # pragma: no cover - depends on ROS installation
        raise RosBridgeUnavailable("rclpy and geometry_msgs are required for ros-bridge") from exc

    rclpy.init()
    node = rclpy.create_node("h_ide_mujoco_sonic_bridge")
    safety_timer_clock = Clock(clock_type=ClockType.STEADY_TIME)
    state_group = MutuallyExclusiveCallbackGroup()
    normal_stop_group = MutuallyExclusiveCallbackGroup()
    generation = {"value": _new_bridge_generation()}
    bridge_boot_id = uuid4()
    bridge_session_id = uuid4()
    bridge_runtime_health = BridgeRuntimeHealth(
        boot_id=bridge_boot_id,
        session_id=bridge_session_id,
    )
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
            observation_timeout_s=_gait_observation_timeout_s(
                config.observation_hz
            ),
            max_linear_mps=config.max_linear_mps,
            max_angular_rps=config.max_angular_rps,
            linear_deadband_mps=config.min_walk_command_mps,
            angular_deadband_rps=0.05,
            # A learned whole-body step can initially counter-shift its pelvis
            # and may need several seconds to produce an odometry-scale base
            # velocity after a fresh Spin/Drive reference. Let the bounded
            # Nav2 action own an ordinary no-progress case instead of latching
            # the whole simulator after 1.5 seconds of benign foot placement.
            start_linear_ack_mps=0.005,
            start_angular_ack_rps=0.005,
            start_timeout_s=12.0,
            reverse_translation_m=0.30,
            reverse_min_travel_m=0.30,
            startup_reverse_grace_s=6.0,
            startup_progress_deadline_s=10.0,
            reverse_rotation_rad=0.35,
            stop_timeout_s=2.5,
            # learned-IDLE is an active balancing policy. Ordinary action
            # completion may need several seconds to absorb a whole-body
            # turn/translation correction; emergency cancel/safe-stop keeps
            # the independent 2.5 s limit above.
            balanced_idle_stop_timeout_s=15.0,
            balanced_idle_stop_linear_excursion_m=0.08,
            balanced_idle_stop_yaw_excursion_rad=0.25,
            balanced_idle_stop_max_linear_mps=0.20,
            balanced_idle_stop_max_angular_rps=0.24,
            max_facing_delta_rad=config.max_facing_lead_rad,
            trip_on_excessive_tilt=not retain_sonic_on_automatic_fault,
            trip_on_automatic_fault=not retain_sonic_on_automatic_fault,
        )
    )
    # GaitAdapter may recover a never-moved IDLE observation reconnect within
    # one generation, but that local synchronization never releases a Plant
    # safety_stop.  Only /robot/control_arm, reached from control.acquire,
    # advances generation and carries the explicit Plant rearm bit.
    gait.advance_generation(generation["value"])
    latest_heading = {"yaw": None}
    latest_observation: dict[str, BaseObservation | None] = {"value": None}
    action_statuses: dict[str, _ActionStatusObservation | None] = {
        "navigate_to_pose": None,
        "spin": None,
        "drive_on_heading": None,
    }
    motion_fault_reset_request_id: dict[str, UUID | None] = {"value": None}
    bridge_fault: dict[str, str | float | None] = {
        "reason": None,
        "latched_at_s": None,
    }
    retained_automatic_fault: dict[str, str | int | None] = {
        "count": 0,
        "last": None,
    }
    normal_stop_coordinator: dict[
        str, BridgeNormalStopCoordinator | None
    ] = {"value": None}
    health_fault_gate = SonicHealthFaultGate(
        plant_lowstate_stale_s=config.plant_lowstate_stale_s,
        sonic_control_step_stale_s=config.sonic_control_step_stale_s,
    )
    health_transport_gate = SonicHealthTransportGate(
        probe_period_s=config.sonic_health_probe_period_s,
        probe_timeout_s=config.sonic_health_probe_timeout_s,
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
    stability_guard = BodyStabilityGuard(
        latch_on_excessive_tilt=not retain_sonic_on_automatic_fault,
    )
    if retain_sonic_on_automatic_fault:
        node.get_logger().warning(
            "simulation uninterrupted-SONIC experiment enabled: automatic "
            "gait, navigation and health observations will be logged without "
            "revoking actuator authority; explicit cancel and safe-stop remain active"
        )
    navigation_guard_authority_identity: dict[
        str,
        tuple[int, int, int, int] | None,
    ] = {"value": None}

    def request_sonic_health(
        request_id: UUID,
        *,
        timeout_s: float | None = None,
    ) -> SonicHealth:
        """Read one correlated, strict v3 health snapshot from SONIC."""

        response = request_json(
            config.command_host,
            config.command_port,
            sonic_health_probe(
                robot_id=config.robot_id,
                producer_boot_id=fault_latch.producer_boot_id,
                request_id=request_id,
            ),
            timeout=(
                config.sonic_health_probe_timeout_s
                if timeout_s is None
                else float(timeout_s)
            ),
        )
        return SonicHealth.from_response(
            response,
            expected_request_id=request_id,
            expected_robot_id=config.robot_id,
        )

    def attempt_sonic_health(
        request_id: UUID,
        *,
        timeout_s: float | None = None,
    ) -> _HealthProbeAttempt:
        return _attempt_sonic_health_probe(
            request_id,
            request_health=lambda identity: request_sonic_health(
                identity,
                timeout_s=timeout_s,
            ),
            transport_gate=health_transport_gate,
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

    def record_retained_automatic_fault(
        *,
        source: str,
        code: str,
        reason: str,
    ) -> None:
        """Record an experiment-only warning without changing authority."""

        bounded = str(reason).strip()[:512] or "automatic motion observation"
        signature = f"{source}/{code}: {bounded}"
        count = int(retained_automatic_fault["count"] or 0) + 1
        previous = retained_automatic_fault["last"]
        retained_automatic_fault["count"] = count
        retained_automatic_fault["last"] = signature
        # Repeated health samples can report the same observation at high rate.
        # Keep it visible without flooding the journal.
        if signature != previous or count & (count - 1) == 0:
            node.get_logger().warning(
                "automatic SONIC cutoff suppressed by simulation experiment "
                f"count={count} {signature}"
            )

    def clear_navigation_guard_latches() -> None:
        for guard in (
            direction_guard,
            goal_guard,
            heading_guard,
            rotation_guard,
            stability_guard,
        ):
            if guard.latched:
                guard.clear()

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
        if retain_sonic_on_automatic_fault:
            record_retained_automatic_fault(
                source=source,
                code=code,
                reason=bounded_reason,
            )
            return
        if bridge_fault["reason"] is None:
            bridge_fault["reason"] = f"{source}/{code}: {bounded_reason}"
            bridge_fault["latched_at_s"] = time.monotonic()
        coordinator = normal_stop_coordinator["value"]
        if coordinator is not None:
            coordinator.notify_terminal(
                f"{source}/{code}: {bounded_reason}"
            )

        def local_stop() -> None:
            errors: list[str] = []
            intent = stop_intent
            fault_reason = f"{source}/{code}: {bounded_reason}"[:512]
            # A navigation/gait safety event revokes actuator authority and
            # latches Plant safety_stop, but it is not evidence that MuJoCo
            # physics is corrupt. Keep the Plant and both viewers alive so an
            # operator can inspect the stopped incident and explicitly reset.
            try:
                latch_plant_motion_fault(
                    "127.0.0.1",
                    config.plant_port,
                    fault_reason,
                    timeout=0.3,
                )
            except Exception as exc:
                errors.append(f"Plant motion-fault stop: {exc}")
            try:
                event = fault_latch.event
                if event is None:
                    raise RuntimeError("motion fault identity was not allocated")
                request_json(
                    config.command_host,
                    config.command_port,
                    {
                        "type": "motion_fault_latch",
                        "request_id": str(event.event_id),
                        "reason": fault_reason,
                    },
                    timeout=0.5,
                )
            except Exception as exc:
                errors.append(f"SONIC motion-fault latch: {exc}")
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

    def log_health_transport_transition(
        decision: HealthTransportDecision,
        *,
        context: str,
    ) -> None:
        if decision.transition not in {"pending", "recovered", "confirmed"}:
            return
        pending = decision.pending
        details = (
            "SONIC_HEALTH_PROBE_TIMEOUT "
            f"state={decision.transition.upper()} context={context} "
            f"count={0 if pending is None else pending.count}/"
            f"{decision.required_count} "
            f"totalTimeoutCount={decision.total_timeout_count} "
            f"recoveredTimeoutCount={decision.recovered_timeout_count} "
            f"confirmedTimeoutCount={decision.confirmed_timeout_count} "
            f"confirmationBudgetS={decision.confirmation_budget_s:.6f}"
        )
        if pending is not None:
            details += (
                f" firstRequestId={pending.first_request_id} "
                f"latestRequestId={pending.latest_request_id} "
                f"deadlineMonotonicS={pending.confirmation_deadline_s:.9f}"
            )
        if decision.transition == "confirmed":
            node.get_logger().error(details)
        elif decision.transition == "pending":
            node.get_logger().warning(details)
        else:
            node.get_logger().info(details)

    def enforce_transport_decision(
        decision: HealthTransportDecision,
        *,
        context: str,
        probe_request_id: UUID | None = None,
    ) -> bool:
        """Log the transition and terminalize an expired/confirmed fence."""

        log_health_transport_transition(decision, context=context)
        if retain_sonic_on_automatic_fault:
            cause = decision.confirmed_cause
            if cause is not None:
                record_retained_automatic_fault(
                    source=cause.source,
                    code=cause.code,
                    reason=f"{cause.reason} during {context}",
                )
            return False
        cause = decision.confirmed_cause
        if cause is None:
            return decision.pending is not None
        if bridge_fault["reason"] is None:
            try:
                latch_stop(
                    source=cause.source,
                    code=cause.code,
                    reason=cause.reason,
                    probe_request_id=probe_request_id,
                )
            except Exception as delivery_exc:
                node.get_logger().error(
                    "failed to enforce SONIC health transport stop: "
                    f"{delivery_exc}"
                )
        return True

    def check_health_transport_fence(*, context: str) -> bool:
        decision = health_transport_gate.check_deadline(
            now_s=time.monotonic()
        )
        return enforce_transport_decision(decision, context=context)

    def classify_health_probe_exception(
        exc: Exception,
    ) -> tuple[str, str]:
        if isinstance(exc, (JsonResponseContractError, ValueError)):
            return (
                "SONIC_HEALTH_CONTRACT_FAULT",
                f"SONIC health response contract failed: {exc}",
            )
        if isinstance(
            exc,
            (
                JsonResponseUnavailableError,
                ConnectionError,
                OSError,
                EOFError,
            ),
        ):
            return (
                "SONIC_UNAVAILABLE",
                f"SONIC health transport is unavailable: {exc}",
            )
        return (
            "SONIC_UNAVAILABLE",
            f"SONIC health probe failed: {exc}",
        )

    def latch_hard_health_probe_failure(
        exc: Exception,
        *,
        context: str,
        probe_request_id: UUID,
    ) -> str:
        code, reason = classify_health_probe_exception(exc)
        bounded = f"{reason} during {context}"[:512]
        try:
            latch_stop(
                source="sonic_driver",
                code=code,
                reason=bounded,
                probe_request_id=probe_request_id,
            )
        except Exception as delivery_exc:
            node.get_logger().error(
                f"failed to enforce hard SONIC health-probe stop: {delivery_exc}"
            )
        return bounded

    def deliver_intent(intent: SonicIntent, *, context: str) -> bool:
        transport_fenced = check_health_transport_fence(
            context=f"intent_delivery:{context}"
        )
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
        if _suppress_for_unknown_health_transport(
            transport_fenced,
            intent,
        ):
            node.get_logger().warning(
                "motion-bearing gait intent suppressed while SONIC health "
                f"transport is unknown context={context} "
                f"state={intent.state} sequence={intent.sequence}"
            )
            return False
        if _suppress_for_correlated_stop(
            normal_stop_coordinator["value"],
            intent,
        ):
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

    normal_stop_guard: dict[str, object | None] = {"value": None}

    def normal_stop_terminal_reason() -> str | None:
        reason = bridge_fault["reason"]
        if reason is not None:
            return str(reason)
        event = fault_latch.event
        if event is None:
            return None
        return f"{event.source}/{event.code}: {event.reason}"

    def request_normal_gait_stop(
        control_generation: int,
        now: float,
    ) -> dict[str, object]:
        intent = gait.request_safe_stop(control_generation, now=now)
        if intent.state == GaitState.FAULT.value:
            # FAULT is never repackaged as a recoverable normal failure.
            deliver_intent(intent, context="correlated safe-stop request")
        return _intent_payload(
            intent,
            measured_heading_yaw=latest_heading["yaw"],
        )

    def request_normal_stop_driver(
        payload: Mapping[str, object],
        timeout_s: float,
    ) -> dict[str, object]:
        return request_json(
            config.command_host,
            config.command_port,
            dict(payload),
            timeout=timeout_s,
            require_ok=False,
        )

    def stop_normal_stop_plant(timeout_s: float) -> dict[str, object]:
        return set_plant_support(
            "127.0.0.1",
            config.plant_port,
            "safety_stop",
            timeout=timeout_s,
        )

    def probe_normal_stop_health(
        request_id: UUID,
        timeout_s: float,
    ) -> SonicHealth:
        # The normal-stop worker has its own reservation/deadline and must not
        # mutate the state-group transport gate with concurrent probe I/O.
        # Still leave a persistent acceptance marker for every lost reply.
        try:
            return request_sonic_health(request_id, timeout_s=timeout_s)
        except JsonResponseTimeoutError:
            node.get_logger().warning(
                "SONIC_HEALTH_PROBE_TIMEOUT state=NORMAL_STOP "
                f"context=correlated_normal_stop requestId={request_id}"
            )
            raise

    def wake_normal_stop_pump() -> None:
        guard = normal_stop_guard["value"]
        if guard is None:
            raise RuntimeError("normal-stop GuardCondition is unavailable")
        guard.trigger()  # type: ignore[attr-defined]

    coordinator = BridgeNormalStopCoordinator(
        robot_id=config.robot_id,
        bridge_boot_id=bridge_boot_id,
        generation=lambda: generation["value"],
        request_gait_stop=request_normal_gait_stop,
        request_driver=request_normal_stop_driver,
        stop_plant=stop_normal_stop_plant,
        probe_health=probe_normal_stop_health,
        wake_state_pump=wake_normal_stop_pump,
        terminal_reason=normal_stop_terminal_reason,
        plant_lowstate_stale_s=config.plant_lowstate_stale_s,
        sonic_control_step_stale_s=config.sonic_control_step_stale_s,
    )
    normal_stop_coordinator["value"] = coordinator

    def pump_normal_stop() -> None:
        coordinator.pump()

    normal_stop_guard["value"] = node.create_guard_condition(
        pump_normal_stop,
        callback_group=state_group,
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
            node.get_logger().error(f"SONIC navigation interlock observed: {reason}")
            try:
                latch_stop(
                    source="navigation_guard",
                    code="NAVIGATION_GUARD_FAULT",
                    reason=reason,
                )
            except Exception as exc:
                node.get_logger().error(f"failed to deliver navigation interlock stop: {exc}")
            if retain_sonic_on_automatic_fault:
                clear_navigation_guard_latches()

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
        # Bind the SONIC heading observation to this exact odometry sample.
        # The observation bridge publishes odom immediately before the matching
        # map pose.  Both subscriptions share the bridge's mutually-exclusive
        # state group, and synchronous SONIC delivery from ``on_odom`` can keep
        # the later pose callback queued behind newer odometry.  Using the
        # separately cached pose yaw here therefore let a live Spin execute a
        # stale ~0.016 rad heading while MuJoCo had already reached ~0.20 rad;
        # the measured-lead limiter then held the planner request near 0.066 rad
        # forever.  Odom carries the same capture quaternion, so make its yaw the
        # authoritative value for the intent emitted from this callback.
        latest_heading["yaw"] = yaw
        transport_fenced = check_health_transport_fence(
            context="odometry_auto_arm"
        )
        intent = gait.ingest_observation(observation, now=now)
        if intent.state == GaitState.FAULT.value:
            deliver_intent(intent, context="odometry observation")
            return
        if (
            bridge_fault["reason"] is None
            and not transport_fenced
            and gait.state is GaitState.DISARMED
        ):
            # A successful result proves this exact observation was fresh and
            # stationary; rejected attempts remain fail-closed.
            armed = gait.arm(generation["value"], now=now)
            if armed.state == GaitState.IDLE.value:
                intent = armed
        coordinator.observe_stationary_intent(
            _intent_payload(
                intent,
                measured_heading_yaw=yaw,
            )
        )
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

    def on_action_status(name: str, message: GoalStatusArray) -> None:
        active = any(int(item.status) in {1, 2, 3} for item in message.status_list)
        action_statuses[name] = _ActionStatusObservation(
            active=active,
            observed_at_s=time.monotonic(),
        )
        if (name == "navigate_to_pose" and not active) or (
            name in {"spin", "drive_on_heading"} and active
        ):
            goal_guard.suspend_tracking()

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
        transport_fenced = check_health_transport_fence(
            context="nav2_command"
        )
        if bridge_fault["reason"] is not None:
            deliver_intent(gait.tick(now=now), context="latched bridge fault")
            return
        transport_action = _health_transport_command_action(
            fenced=transport_fenced,
            gait_state=gait.state,
            vx=vx,
            vy=vy,
            wz=wz,
        )
        if transport_action != "allow":
            if transport_action == "cancel_start":
                stopped = gait.cancel(
                    generation["value"],
                    now,
                    now=now,
                )
                deliver_intent(
                    stopped,
                    context="health transport command-start revocation",
                )
            node.get_logger().warning(
                "non-zero Nav2 command blocked while SONIC health transport "
                "is unknown"
            )
            return
        direction_admitted = direction_guard.accept_command(vx, vy, wz)
        goal_admitted = goal_guard.accept_command(
            vx,
            vy,
            wz,
            tracking_enabled=_global_goal_progress_active(action_statuses),
        )
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
            if retain_sonic_on_automatic_fault:
                clear_navigation_guard_latches()
            else:
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

    def perform_control_arm(_request, response):
        """Use one explicit Robot Agent control.acquire to rearm the plant."""

        if bridge_fault["reason"] is not None:
            response.success = False
            response.message = (
                "navigation safety interlock is latched; explicit motion-fault "
                "reset is required before control can be acquired again"
            )
            return response
        if check_health_transport_fence(context="control_arm_admission"):
            response.success = False
            response.message = (
                "control arm rejected while SONIC health transport is unknown"
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
            attempt = attempt_sonic_health(health_request_id)
        except Exception as exc:
            reason = latch_hard_health_probe_failure(
                exc,
                context="control_arm",
                probe_request_id=health_request_id,
            )
            response.success = False
            response.message = reason
            return response
        transport_blocked = enforce_transport_decision(
            attempt.transport,
            context="control_arm",
            probe_request_id=health_request_id,
        )
        if transport_blocked or attempt.health is None:
            response.success = False
            response.message = (
                "control arm rejected while SONIC health transport is unknown"
            )
            return response
        health = attempt.health

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

        try:
            resident_generation = _resident_policy_reentry_generation(
                health=health,
                gait_state=gait.state,
                bridge_generation=generation["value"],
                gait_generation=gait.generation,
                observation_generation=measured.generation,
            )
        except ValueError as exc:
            response.success = False
            response.message = f"control arm rejected: {exc}"
            return response
        if resident_generation is not None:
            # RobotAgent released only its remote lease after the previous
            # command.  SONIC, the gait adapter, and DDS still own one exact
            # stationary learned-IDLE generation, so creating a new physical
            # rearm transaction would contradict that deliberate residency.
            response.success = True
            response.message = (
                "resident balanced policy generation "
                f"{resident_generation} re-authorized at stationary IDLE"
            )
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

    def control_arm(request, response):
        blocked = coordinator.reserve_control_arm()
        if blocked is not None:
            response.success = False
            response.message = f"control arm rejected: {blocked}"
            return response
        try:
            return perform_control_arm(request, response)
        finally:
            coordinator.release_control_arm()

    node.create_subscription(
        Twist,
        config.cmd_vel_topic,
        on_cmd,
        10,
        callback_group=state_group,
    )
    node.create_subscription(
        Odometry,
        config.odom_topic,
        on_odom,
        10,
        callback_group=state_group,
    )
    node.create_subscription(
        PoseWithCovarianceStamped,
        config.localization_topic,
        on_pose,
        10,
        callback_group=state_group,
    )
    node.create_subscription(
        Path,
        config.nav_path_topic,
        on_path,
        10,
        callback_group=state_group,
    )
    node.create_subscription(
        GoalStatusArray,
        "/navigate_to_pose/_action/status",
        lambda message: on_action_status("navigate_to_pose", message),
        10,
        callback_group=state_group,
    )
    node.create_subscription(
        GoalStatusArray,
        "/spin/_action/status",
        lambda message: on_action_status("spin", message),
        10,
        callback_group=state_group,
    )
    node.create_subscription(
        GoalStatusArray,
        "/drive_on_heading/_action/status",
        lambda message: on_action_status("drive_on_heading", message),
        10,
        callback_group=state_group,
    )

    def bridge_health(_request, response):
        snapshot = bridge_runtime_health.snapshot(
            gait_state=gait.state,
            control_generation=generation["value"],
            bridge_fault=(
                None
                if bridge_fault["reason"] is None
                else str(bridge_fault["reason"])
            ),
            health_transport_pending=health_transport_gate.pending is not None,
            health_transport_confirmed=(
                health_transport_gate.confirmed_cause is not None
            ),
            health_fault_pending=health_fault_gate.pending is not None,
            health_quarantined=health_quarantine["active"] is True,
        )
        response.success = bool(snapshot["healthy"])
        response.message = json.dumps(
            snapshot,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return response

    node.create_service(
        Trigger,
        "/h_ide/mujoco/bridge_health",
        bridge_health,
        callback_group=state_group,
    )
    node.create_service(
        Trigger,
        "/robot/control_arm",
        control_arm,
        callback_group=state_group,
    )

    def reset_motion_fault(_request, response):
        """Clear one stopped fault; a later control_arm must reauthorize."""

        nonlocal health_fault_gate, health_transport_gate
        event = fault_latch.event
        reason_value = bridge_fault["reason"]
        reason = None if reason_value is None else str(reason_value)
        latched_at_value = bridge_fault["latched_at_s"]
        now = time.monotonic()
        measured = latest_observation["value"]
        if event is None or reason is None:
            response.success = False
            response.message = "no motion safety fault is latched"
            return response
        if not fault_latch.complete:
            response.success = False
            response.message = "motion fault stop transaction is incomplete"
            return response
        if (
            latched_at_value is None
            or not _actions_confirmed_inactive_after(
                action_statuses,
                fault_latched_at_s=float(latched_at_value),
            )
        ):
            response.success = False
            response.message = (
                "motion fault reset requires post-fault confirmation that all "
                "Nav2 actions are inactive"
            )
            return response
        if (
            measured is None
            or now - measured.timestamp_s > gait.config.observation_timeout_s
            or not gait._is_stationary(measured)
            or max(abs(measured.roll), abs(measured.pitch)) > 0.15
        ):
            response.success = False
            response.message = (
                "motion fault reset requires fresh finite stationary upright odometry"
            )
            return response
        request_id = motion_fault_reset_request_id["value"]
        if request_id is None:
            request_id = uuid4()
            motion_fault_reset_request_id["value"] = request_id
        try:
            reset_result = request_json(
                config.command_host,
                config.command_port,
                {
                    "type": "motion_fault_reset",
                    "request_id": str(request_id),
                    "reason": reason,
                },
                timeout=0.75,
            )
            if (
                reset_result.get("status") != "reset"
                or reset_result.get("mode") != "safety_stop"
                or reset_result.get("authority") != "REVOKED"
                or int(reset_result.get("safe_hold_frames", 0)) < 2
            ):
                raise RuntimeError("SONIC did not confirm the reset boundary")

            next_generation = max(
                generation["value"] + 1,
                _new_bridge_generation(),
            )
            generation["value"] = next_generation
            gait.advance_generation(next_generation, now=now)
            current_observation = replace(measured, generation=next_generation)
            latest_observation["value"] = current_observation
            gait.ingest_observation(current_observation, now=now)
            if gait.state is not GaitState.DISARMED:
                raise RuntimeError("gait reset did not remain DISARMED")
            for guard in (
                direction_guard,
                goal_guard,
                heading_guard,
                rotation_guard,
                stability_guard,
            ):
                guard.clear()
            fault_latch.reset(event.event_id)
            bridge_fault["reason"] = None
            bridge_fault["latched_at_s"] = None
            health_fault_gate = SonicHealthFaultGate(
                plant_lowstate_stale_s=config.plant_lowstate_stale_s,
                sonic_control_step_stale_s=config.sonic_control_step_stale_s,
            )
            health_transport_gate = SonicHealthTransportGate(
                probe_period_s=config.sonic_health_probe_period_s,
                probe_timeout_s=config.sonic_health_probe_timeout_s,
            )
            health_quarantine.update(active=False, code=None)
            motion_fault_reset_request_id["value"] = None
        except Exception as exc:
            response.success = False
            response.message = f"motion fault reset failed closed: {exc}"
            return response
        response.success = True
        response.message = (
            f"motion fault reset at generation {next_generation}; "
            "control authority remains revoked"
        )
        return response

    node.create_service(
        Trigger,
        "/h_ide/mujoco/reset_motion_fault",
        reset_motion_fault,
        callback_group=state_group,
    )

    def normal_stop(_request, response):
        result = coordinator.request_stop(
            "Robot Agent requested a correlated simulation safe-stop"
        )
        response.success = result.success
        response.message = result.message
        return response

    node.create_service(
        Trigger,
        "/h_ide/mujoco/normal_stop",
        normal_stop,
        callback_group=normal_stop_group,
    )

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
            attempt = attempt_sonic_health(request_id)
        except Exception as exc:
            latch_hard_health_probe_failure(
                exc,
                context="periodic_health",
                probe_request_id=request_id,
            )
            return

        transport_blocked = enforce_transport_decision(
            attempt.transport,
            context="periodic_health",
            probe_request_id=request_id,
        )
        if attempt.timeout_error is not None:
            # STARTING has not yet been acknowledged by this bridge as having
            # actuator authority.  Revoke that incomplete start immediately;
            # an old/late DDS grant must not land inside the timeout window.
            if (
                not retain_sonic_on_automatic_fault
                and
                attempt.transport.confirmed_cause is None
                and gait.state is GaitState.STARTING
            ):
                now = time.monotonic()
                stopped = gait.cancel(
                    generation["value"],
                    now,
                    now=now,
                )
                node.get_logger().warning(
                    "SONIC health transport became unknown during STARTING; "
                    "revoking the unconfirmed motion start"
                )
                deliver_intent(
                    stopped,
                    context="health transport STARTING revocation",
                )
            return
        if transport_blocked or attempt.health is None:
            return
        health = attempt.health
        bridge_runtime_health.observe_sonic_probe(health)

        previous_pending = health_fault_gate.pending
        decision = health_fault_gate.observe(
            health,
            gait_state=gait.state,
        )
        try:
            granted = _observe_health_reference_authority(
                gait,
                health,
                decision,
                now=time.monotonic(),
                transport_decision=attempt.transport,
            )
        except ValueError as exc:
            try:
                latch_stop(
                    source="sonic_driver",
                    code="SONIC_HEALTH_CONTRACT_FAULT",
                    reason=f"invalid reference authority health: {exc}",
                    downstream_boot_id=health.driver_boot_id,
                    probe_request_id=request_id,
                )
            except Exception as delivery_exc:
                node.get_logger().error(
                    "failed to enforce reference-authority health fault: "
                    f"{delivery_exc}"
                )
            return
        if granted:
            authority_identity = (
                int(health.reference_authority_generation),
                int(health.reference_authority_source_sequence),
                int(health.reference_authority_epoch),
                int(health.reference_authority_dds_target_sequence),
            )
            if navigation_guard_authority_identity["value"] != authority_identity:
                direction_guard.reanchor_at_actuator_grant()
                goal_guard.reanchor_at_actuator_grant()
                heading_guard.reanchor_at_actuator_grant()
                rotation_guard.reanchor_at_actuator_grant()
                navigation_guard_authority_identity["value"] = authority_identity
            node.get_logger().info(
                "matched SONIC reference obtained actuator authority "
                f"generation={health.reference_authority_generation} "
                f"sourceSequence={health.reference_authority_source_sequence} "
                f"referenceEpoch={health.reference_authority_epoch} "
                "ddsTargetSequence="
                f"{health.reference_authority_dds_target_sequence}"
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

    _safety_timers = _create_safety_timers(
        node,
        command_timeout_s=config.command_timeout_s,
        health_probe_period_s=config.sonic_health_probe_period_s,
        watchdog=watchdog,
        health_probe=health_probe,
        callback_group=state_group,
        clock=safety_timer_clock,
    )
    node.get_logger().info(
        f"H-IDE MuJoCo SONIC bridge listening to {config.cmd_vel_topic} "
        f"and {config.nav_path_topic} on ROS domain {config.ros_domain_id}; "
        f"faults publish on {config.motion_fault_topic}"
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        coordinator.shutdown()
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
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    run_ros_bridge(SimConfig.from_file(args.config))
