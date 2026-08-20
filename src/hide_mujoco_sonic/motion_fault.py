"""Versioned local contracts for latched simulation motion faults.

The simulator is not an action owner and therefore never names or cancels a
Nav2 goal.  It publishes one robot-scoped, immutable fault event; Robot Agent
correlates that event with the exact goal handles it owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re
import threading
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

from .gait_adapter import GaitState
from .reference_authority import REFERENCE_AUTHORITY_PHASES


MOTION_FAULT_SCHEMA_VERSION = "h-ide.motion-safety-fault/v1"
SONIC_HEALTH_SCHEMA_VERSION = "h-ide.sonic-health/v2"
MOTION_FAULT_MAX_BYTES = 4096
STALE_HEALTH_CONFIRMATION_PROBES = 2
SONIC_HEALTH_TRANSPORT_TIMEOUT_CONFIRMATION_PROBES = 2

_FAULT_FIELDS = frozenset(
    {
        "schemaVersion",
        "eventId",
        "robotId",
        "localizationEpochId",
        "producerBootId",
        "sourceSequence",
        "controlGeneration",
        "source",
        "code",
        "reason",
        "latched",
        "detectedAt",
        "downstreamBootId",
        "probeRequestId",
    }
)
_FAULT_SOURCES = frozenset(
    {
        "ros_bridge",
        "gait_adapter",
        "navigation_guard",
        "sonic_driver",
        "mujoco_plant",
    }
)
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_ROBOT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TEMPORAL_STALE_CODES = frozenset(
    {
        "PLANT_LOWSTATE_STALE",
        "SONIC_CONTROL_STALE",
    }
)

_HEALTH_REQUEST_FIELDS = frozenset(
    {
        "type",
        "schema_version",
        "request_id",
        "robot_id",
        "producer_boot_id",
    }
)
_HEALTH_RESPONSE_FIELDS = frozenset(
    {
        "request_id",
        "ok",
        "schema_version",
        "robot_id",
        "driver_boot_id",
        "status",
        "terminal_fault",
        "reason",
        "motion_ready",
        "last_lowstate_age_s",
        "last_control_step_age_s",
        "plant_support_mode",
        "plant_support_safety_latched",
        "plant_support_stop_pending",
        "static_hold_active",
        "navigation_task_active",
        "motion_authorized",
        "nav2_command_zero",
        "matched_reference_actuator_authority",
        "reference_authority_state",
        "reference_authority_generation",
        "reference_authority_source_sequence",
        "reference_authority_intent_generation",
        "reference_authority_epoch",
        "reference_authority_granted_at_monotonic_s",
        "reference_authority_granted_tick_ms",
        "reference_authority_dds_target_sequence",
    }
)
_PLANT_SUPPORT_MODES = frozenset(
    {
        "unknown",
        "unavailable",
        "startup",
        "idle",
        "locomotion",
        "safety_stop",
    }
)


def _uuid(value: object, field: str) -> UUID:
    try:
        parsed = UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a UUID") from exc
    if str(parsed) != str(value):
        raise ValueError(f"{field} must use canonical UUID text")
    return parsed


def _bounded_text(value: object, field: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be a non-empty string of at most {maximum} characters")
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _detected_at(value: object) -> str:
    text = _bounded_text(value, "detectedAt", maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("detectedAt must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("detectedAt must include a timezone")
    return text


@dataclass(frozen=True, slots=True)
class MotionSafetyFault:
    event_id: UUID
    robot_id: str
    localization_epoch_id: str
    producer_boot_id: UUID
    source_sequence: int
    control_generation: int
    source: str
    code: str
    reason: str
    detected_at: str
    downstream_boot_id: UUID | None = None
    probe_request_id: UUID | None = None

    def __post_init__(self) -> None:
        if not _ROBOT_ID.fullmatch(self.robot_id):
            raise ValueError("robotId is invalid")
        _bounded_text(
            self.localization_epoch_id,
            "localizationEpochId",
            maximum=256,
        )
        _integer(self.source_sequence, "sourceSequence", minimum=1)
        _integer(self.control_generation, "controlGeneration", minimum=0)
        if self.source not in _FAULT_SOURCES:
            raise ValueError("source is not a supported motion-fault producer")
        if not _CODE.fullmatch(self.code):
            raise ValueError("code must be an uppercase machine-readable identifier")
        _bounded_text(self.reason, "reason", maximum=512)
        _detected_at(self.detected_at)

    def as_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": MOTION_FAULT_SCHEMA_VERSION,
            "eventId": str(self.event_id),
            "robotId": self.robot_id,
            "localizationEpochId": self.localization_epoch_id,
            "producerBootId": str(self.producer_boot_id),
            "sourceSequence": self.source_sequence,
            "controlGeneration": self.control_generation,
            "source": self.source,
            "code": self.code,
            "reason": self.reason,
            "latched": True,
            "detectedAt": self.detected_at,
            "downstreamBootId": (
                None
                if self.downstream_boot_id is None
                else str(self.downstream_boot_id)
            ),
            "probeRequestId": (
                None if self.probe_request_id is None else str(self.probe_request_id)
            ),
        }

    def to_json(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(encoded.encode("utf-8")) > MOTION_FAULT_MAX_BYTES:
            raise ValueError("motion-fault event exceeds the bounded wire size")
        return encoded

    @classmethod
    def from_json(cls, raw: str) -> "MotionSafetyFault":
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MOTION_FAULT_MAX_BYTES:
            raise ValueError("motion-fault event must be bounded JSON text")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("motion-fault event is not valid JSON") from exc
        if not isinstance(value, dict) or set(value) != _FAULT_FIELDS:
            raise ValueError("motion-fault event fields do not match v1")
        if value.get("schemaVersion") != MOTION_FAULT_SCHEMA_VERSION:
            raise ValueError("unsupported motion-fault schema")
        if value.get("latched") is not True:
            raise ValueError("motion-fault event must be latched")
        downstream = value.get("downstreamBootId")
        probe = value.get("probeRequestId")
        return cls(
            event_id=_uuid(value.get("eventId"), "eventId"),
            robot_id=str(value.get("robotId", "")),
            localization_epoch_id=str(value.get("localizationEpochId", "")),
            producer_boot_id=_uuid(value.get("producerBootId"), "producerBootId"),
            source_sequence=_integer(
                value.get("sourceSequence"), "sourceSequence", minimum=1
            ),
            control_generation=_integer(
                value.get("controlGeneration"), "controlGeneration", minimum=0
            ),
            source=str(value.get("source", "")),
            code=str(value.get("code", "")),
            reason=str(value.get("reason", "")),
            detected_at=_detected_at(value.get("detectedAt")),
            downstream_boot_id=(
                None if downstream is None else _uuid(downstream, "downstreamBootId")
            ),
            probe_request_id=(
                None if probe is None else _uuid(probe, "probeRequestId")
            ),
        )


class MotionFaultLatch:
    """Create one immutable event per bridge boot and publish it before stop I/O."""

    def __init__(
        self,
        *,
        robot_id: str,
        localization_epoch_id: str,
        producer_boot_id: UUID | None = None,
        clock: Callable[[], datetime] | None = None,
        event_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if not _ROBOT_ID.fullmatch(robot_id):
            raise ValueError("robot_id is invalid")
        _bounded_text(localization_epoch_id, "localization_epoch_id", maximum=256)
        self.robot_id = robot_id
        self.localization_epoch_id = localization_epoch_id
        self.producer_boot_id = producer_boot_id or uuid4()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._event_id_factory = event_id_factory
        self._event: MotionSafetyFault | None = None
        self._published = False
        self._stopped = False
        self._sequence = 0
        self._lock = threading.RLock()

    @property
    def event(self) -> MotionSafetyFault | None:
        with self._lock:
            return self._event

    @property
    def complete(self) -> bool:
        with self._lock:
            return self._published and self._stopped

    def trip(
        self,
        *,
        control_generation: int,
        source: str,
        code: str,
        reason: str,
        publish: Callable[[MotionSafetyFault], None],
        local_stop: Callable[[], None],
        downstream_boot_id: UUID | None = None,
        probe_request_id: UUID | None = None,
    ) -> MotionSafetyFault:
        """Attempt publication first, then execute the stronger local stop once.

        Either failed operation may be retried by a repeated trip, but completed
        operations are not duplicated and the first-cause identity never changes.
        """

        with self._lock:
            if self._event is None:
                self._sequence += 1
                detected = self._clock()
                if detected.tzinfo is None or detected.utcoffset() is None:
                    raise ValueError("motion-fault clock must be timezone-aware")
                self._event = MotionSafetyFault(
                    event_id=self._event_id_factory(),
                    robot_id=self.robot_id,
                    localization_epoch_id=self.localization_epoch_id,
                    producer_boot_id=self.producer_boot_id,
                    source_sequence=self._sequence,
                    control_generation=control_generation,
                    source=source,
                    code=code,
                    reason=reason,
                    detected_at=detected.astimezone(timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    ),
                    downstream_boot_id=downstream_boot_id,
                    probe_request_id=probe_request_id,
                )
            event = self._event
            should_publish = not self._published
            should_stop = not self._stopped

        publish_error: Exception | None = None
        if should_publish:
            try:
                publish(event)
            except Exception as exc:  # local stop must still execute
                publish_error = exc
            else:
                with self._lock:
                    self._published = True

        stop_error: Exception | None = None
        if should_stop:
            try:
                local_stop()
            except Exception as exc:
                stop_error = exc
            else:
                with self._lock:
                    self._stopped = True

        if publish_error is not None or stop_error is not None:
            details = "; ".join(
                item
                for item in (
                    (
                        f"fault publication failed: {publish_error}"
                        if publish_error is not None
                        else ""
                    ),
                    f"local stop failed: {stop_error}" if stop_error is not None else "",
                )
                if item
            )
            raise RuntimeError(details)
        return event


def motion_fault_qos():
    """Return the latched reliable QoS used by both producer and consumer."""

    from rclpy.qos import (  # lazy: unit tests do not require a ROS install
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def sonic_health_probe(
    *,
    robot_id: str,
    producer_boot_id: UUID,
    request_id: UUID,
) -> dict[str, object]:
    if not _ROBOT_ID.fullmatch(robot_id):
        raise ValueError("robot_id is invalid")
    return {
        "type": "sonic_health_probe",
        "schema_version": SONIC_HEALTH_SCHEMA_VERSION,
        "request_id": str(request_id),
        "robot_id": robot_id,
        "producer_boot_id": str(producer_boot_id),
    }


def validate_sonic_health_request(
    payload: Mapping[str, object],
    *,
    expected_robot_id: str,
) -> tuple[UUID, UUID]:
    if set(payload) != _HEALTH_REQUEST_FIELDS:
        raise ValueError("SONIC health-probe fields do not match v1")
    if payload.get("type") != "sonic_health_probe":
        raise ValueError("invalid SONIC health-probe type")
    if payload.get("schema_version") != SONIC_HEALTH_SCHEMA_VERSION:
        raise ValueError("unsupported SONIC health-probe schema")
    if payload.get("robot_id") != expected_robot_id:
        raise ValueError("SONIC health-probe robot identity mismatch")
    return (
        _uuid(payload.get("request_id"), "request_id"),
        _uuid(payload.get("producer_boot_id"), "producer_boot_id"),
    )


@dataclass(frozen=True, slots=True)
class SonicHealth:
    request_id: UUID
    robot_id: str
    driver_boot_id: UUID
    status: str
    terminal_fault: bool
    reason: str | None
    motion_ready: bool
    last_lowstate_age_s: float | None
    last_control_step_age_s: float | None
    plant_support_mode: str
    plant_support_safety_latched: bool
    plant_support_stop_pending: bool
    static_hold_active: bool
    navigation_task_active: bool
    motion_authorized: bool
    nav2_command_zero: bool
    matched_reference_actuator_authority: bool
    reference_authority_state: str
    reference_authority_generation: int | None
    reference_authority_source_sequence: int | None
    reference_authority_intent_generation: int | None
    reference_authority_epoch: int
    reference_authority_granted_at_monotonic_s: float | None
    reference_authority_granted_tick_ms: int | None
    reference_authority_dds_target_sequence: int | None

    @property
    def confirmed_safe_hold(self) -> bool:
        """Whether the driver proves a settled, authority-free physical hold.

        This deliberately excludes the bridge-side gait state.  The caller must
        additionally prove :class:`GaitState.IDLE` before an age-only fault may
        be quarantined; a wire response cannot attest to the ROS state machine.
        """

        return (
            self.plant_support_mode == "safety_stop"
            and self.plant_support_safety_latched
            and not self.plant_support_stop_pending
            and self.static_hold_active
            and not self.navigation_task_active
            and not self.motion_authorized
            and self.nav2_command_zero
        )

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, object],
        *,
        expected_request_id: UUID,
        expected_robot_id: str,
    ) -> "SonicHealth":
        if set(value) != _HEALTH_RESPONSE_FIELDS:
            raise ValueError("SONIC health response fields do not match v1")
        if value.get("ok") is not True:
            raise ValueError("SONIC health response is not acknowledged")
        if value.get("schema_version") != SONIC_HEALTH_SCHEMA_VERSION:
            raise ValueError("unsupported SONIC health response schema")
        request_id = _uuid(value.get("request_id"), "request_id")
        if request_id != expected_request_id:
            raise ValueError("SONIC health response request identity mismatch")
        robot_id = str(value.get("robot_id", ""))
        if robot_id != expected_robot_id:
            raise ValueError("SONIC health response robot identity mismatch")
        status = str(value.get("status", ""))
        if status not in {"warming", "ready", "stopped", "fault"}:
            raise ValueError("SONIC health response status is invalid")
        boolean_fields = (
            "terminal_fault",
            "motion_ready",
            "plant_support_safety_latched",
            "plant_support_stop_pending",
            "static_hold_active",
            "navigation_task_active",
            "motion_authorized",
            "nav2_command_zero",
            "matched_reference_actuator_authority",
        )
        if not all(isinstance(value.get(field), bool) for field in boolean_fields):
            raise ValueError("SONIC health response flags must be booleans")
        terminal_fault = value["terminal_fault"]
        motion_ready = value["motion_ready"]
        support_mode = value.get("plant_support_mode")
        if not isinstance(support_mode, str) or support_mode not in _PLANT_SUPPORT_MODES:
            raise ValueError("SONIC health response Plant support mode is invalid")
        reason_value = value.get("reason")
        if reason_value is not None:
            reason_value = _bounded_text(reason_value, "reason", maximum=512)
        raw_age = value.get("last_lowstate_age_s")
        age: float | None
        if raw_age is None:
            age = None
        elif isinstance(raw_age, bool):
            raise ValueError("last_lowstate_age_s must be finite and non-negative")
        else:
            try:
                age = float(raw_age)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "last_lowstate_age_s must be finite and non-negative"
                ) from exc
            if not math.isfinite(age) or age < 0:
                raise ValueError("last_lowstate_age_s must be finite and non-negative")
        raw_control_age = value.get("last_control_step_age_s")
        control_age: float | None
        if raw_control_age is None:
            control_age = None
        elif isinstance(raw_control_age, bool):
            raise ValueError(
                "last_control_step_age_s must be finite and non-negative"
            )
        else:
            try:
                control_age = float(raw_control_age)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "last_control_step_age_s must be finite and non-negative"
                ) from exc
            if not math.isfinite(control_age) or control_age < 0:
                raise ValueError(
                    "last_control_step_age_s must be finite and non-negative"
                )
        if terminal_fault != (status == "fault"):
            raise ValueError("SONIC terminal-fault flag and status disagree")
        if motion_ready != (status == "ready"):
            raise ValueError("SONIC motion-ready flag and status disagree")
        if terminal_fault and reason_value is None:
            raise ValueError("SONIC terminal fault must include a reason")
        if status == "ready" and (age is None or control_age is None):
            raise ValueError(
                "ready SONIC health must include LowState and control-step ages"
            )
        support_latched = value["plant_support_safety_latched"]
        support_pending = value["plant_support_stop_pending"]
        motion_authorized = value["motion_authorized"]
        navigation_task_active = value["navigation_task_active"]
        nav2_command_zero = value["nav2_command_zero"]
        actuator_authority = value["matched_reference_actuator_authority"]
        authority_state = value.get("reference_authority_state")
        if (
            not isinstance(authority_state, str)
            or authority_state not in REFERENCE_AUTHORITY_PHASES
        ):
            raise ValueError("SONIC reference authority state is invalid")

        def optional_integer(field: str, *, minimum: int) -> int | None:
            raw = value.get(field)
            if raw is None:
                return None
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
                raise ValueError(f"{field} must be an integer >= {minimum} or null")
            return raw

        authority_generation = optional_integer(
            "reference_authority_generation", minimum=0
        )
        authority_source_sequence = optional_integer(
            "reference_authority_source_sequence", minimum=1
        )
        authority_intent_generation = optional_integer(
            "reference_authority_intent_generation", minimum=1
        )
        authority_epoch = optional_integer("reference_authority_epoch", minimum=0)
        if authority_epoch is None:
            raise ValueError("reference_authority_epoch must not be null")
        authority_granted_tick = optional_integer(
            "reference_authority_granted_tick_ms", minimum=0
        )
        authority_dds_sequence = optional_integer(
            "reference_authority_dds_target_sequence", minimum=1
        )
        raw_granted_at = value.get("reference_authority_granted_at_monotonic_s")
        authority_granted_at: float | None
        if raw_granted_at is None:
            authority_granted_at = None
        elif isinstance(raw_granted_at, bool):
            raise ValueError(
                "reference_authority_granted_at_monotonic_s must be finite and non-negative"
            )
        else:
            try:
                authority_granted_at = float(raw_granted_at)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "reference_authority_granted_at_monotonic_s must be finite and non-negative"
                ) from exc
            if not math.isfinite(authority_granted_at) or authority_granted_at < 0:
                raise ValueError(
                    "reference_authority_granted_at_monotonic_s must be finite and non-negative"
                )
        grant_evidence = (
            authority_granted_at,
            authority_granted_tick,
            authority_dds_sequence,
        )
        if actuator_authority != (authority_state == "GRANTED"):
            raise ValueError(
                "SONIC actuator authority flag and reference authority state disagree"
            )
        if actuator_authority:
            if (
                authority_generation is None
                or authority_source_sequence is None
                or authority_intent_generation is None
                or authority_epoch <= 0
                or any(item is None for item in grant_evidence)
            ):
                raise ValueError(
                    "granted SONIC reference authority lacks correlation evidence"
                )
        elif any(item is not None for item in grant_evidence):
            raise ValueError(
                "non-granted SONIC reference authority carries stale DDS grant evidence"
            )
        if support_latched and motion_authorized:
            raise ValueError(
                "SONIC health response cannot authorize motion while support is latched"
            )
        if navigation_task_active and not motion_authorized:
            raise ValueError(
                "SONIC health response cannot report a task without motion authority"
            )
        if navigation_task_active and nav2_command_zero:
            raise ValueError(
                "SONIC health response active task must carry a non-zero Nav2 command"
            )
        if actuator_authority and (
            support_latched
            or support_pending
            or not motion_authorized
            or not navigation_task_active
            or nav2_command_zero
        ):
            raise ValueError(
                "matched reference actuator authority conflicts with the safety snapshot"
            )
        return cls(
            request_id=request_id,
            robot_id=robot_id,
            driver_boot_id=_uuid(value.get("driver_boot_id"), "driver_boot_id"),
            status=status,
            terminal_fault=terminal_fault,
            reason=reason_value,
            motion_ready=motion_ready,
            last_lowstate_age_s=age,
            last_control_step_age_s=control_age,
            plant_support_mode=support_mode,
            plant_support_safety_latched=support_latched,
            plant_support_stop_pending=support_pending,
            static_hold_active=value["static_hold_active"],
            navigation_task_active=navigation_task_active,
            motion_authorized=motion_authorized,
            nav2_command_zero=nav2_command_zero,
            matched_reference_actuator_authority=actuator_authority,
            reference_authority_state=authority_state,
            reference_authority_generation=authority_generation,
            reference_authority_source_sequence=authority_source_sequence,
            reference_authority_intent_generation=authority_intent_generation,
            reference_authority_epoch=authority_epoch,
            reference_authority_granted_at_monotonic_s=authority_granted_at,
            reference_authority_granted_tick_ms=authority_granted_tick,
            reference_authority_dds_target_sequence=authority_dds_sequence,
        )


@dataclass(frozen=True, slots=True)
class MotionFaultCause:
    source: str
    code: str
    reason: str


@dataclass(frozen=True, slots=True)
class PendingStaleHealth:
    driver_boot_id: UUID
    source: str
    code: str
    latest_source: str
    latest_code: str
    count: int
    first_age_s: float
    latest_age_s: float

    @property
    def first_source(self) -> str:
        return self.source

    @property
    def first_code(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class HealthFaultDecision:
    confirmed_cause: MotionFaultCause | None
    pending: PendingStaleHealth | None
    required_count: int
    driver_boot_id: UUID
    filtered_transient_count: int
    quarantined_cause: MotionFaultCause | None
    settled_safe_hold: bool
    quarantined_stale_count: int


@dataclass(frozen=True, slots=True)
class PendingHealthTransportTimeout:
    """One bounded streak of health replies whose transport timed out."""

    first_request_id: UUID
    latest_request_id: UUID
    count: int
    first_probe_started_at_s: float
    latest_probe_completed_at_s: float
    confirmation_deadline_s: float


@dataclass(frozen=True, slots=True)
class HealthTransportDecision:
    """Immutable evidence emitted by :class:`SonicHealthTransportGate`."""

    confirmed_cause: MotionFaultCause | None
    pending: PendingHealthTransportTimeout | None
    required_count: int
    confirmation_budget_s: float
    total_timeout_count: int
    recovered_timeout_count: int
    confirmed_timeout_count: int
    transition: str


class SonicHealthTransportGate:
    """Confirm only SONIC health *transport* timeouts within one hard budget.

    This gate is deliberately separate from :class:`SonicHealthFaultGate`.
    A missing reply is unknown transport state, not age-only health evidence,
    and it can never be quarantined as a healthy SAFE_HOLD sample.  The first
    timeout fences all new motion authority.  A complete, correlated v2 reply
    may clear that fence before the deadline; a second consecutive timeout or
    any observation at/after the deadline is terminal.

    Callers serialize admitted probe attempts on one state-machine boundary.
    The internal lock only makes read-only diagnostics safe; it does not turn
    concurrently issued probes into valid consecutive confirmation samples.
    """

    def __init__(
        self,
        *,
        probe_period_s: float,
        probe_timeout_s: float,
        required_timeout_samples: int = (
            SONIC_HEALTH_TRANSPORT_TIMEOUT_CONFIRMATION_PROBES
        ),
        cancellation_start_budget_s: float = 0.10,
    ) -> None:
        for value, field in (
            (probe_period_s, "probe_period_s"),
            (probe_timeout_s, "probe_timeout_s"),
            (cancellation_start_budget_s, "cancellation_start_budget_s"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if (
            isinstance(required_timeout_samples, bool)
            or required_timeout_samples
            != SONIC_HEALTH_TRANSPORT_TIMEOUT_CONFIRMATION_PROBES
        ):
            raise ValueError(
                "required_timeout_samples is fixed at "
                f"{SONIC_HEALTH_TRANSPORT_TIMEOUT_CONFIRMATION_PROBES}"
            )
        if probe_timeout_s > probe_period_s:
            raise ValueError("probe_timeout_s must not exceed probe_period_s")
        confirmation_budget_s = (
            required_timeout_samples * float(probe_period_s)
            + float(probe_timeout_s)
        )
        if confirmation_budget_s >= float(cancellation_start_budget_s):
            raise ValueError(
                "health transport timeout confirmation must remain below the "
                "cancellation-start budget"
            )

        self._required_count = int(required_timeout_samples)
        self._confirmation_budget_s = confirmation_budget_s
        self._lock = threading.RLock()
        self._pending: PendingHealthTransportTimeout | None = None
        self._pending_request_ids: set[UUID] = set()
        self._confirmed_cause: MotionFaultCause | None = None
        self._total_timeout_count = 0
        self._recovered_timeout_count = 0
        self._confirmed_timeout_count = 0

    @property
    def confirmation_budget_s(self) -> float:
        return self._confirmation_budget_s

    @property
    def pending(self) -> PendingHealthTransportTimeout | None:
        with self._lock:
            return self._pending

    @property
    def confirmed_cause(self) -> MotionFaultCause | None:
        with self._lock:
            return self._confirmed_cause

    @staticmethod
    def _monotonic(value: float, field: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
        return result

    @staticmethod
    def _request_id(value: UUID) -> UUID:
        if not isinstance(value, UUID):
            raise TypeError("request_id must be a UUID")
        return value

    def _decision(self, transition: str = "none") -> HealthTransportDecision:
        return HealthTransportDecision(
            confirmed_cause=self._confirmed_cause,
            pending=self._pending,
            required_count=self._required_count,
            confirmation_budget_s=self._confirmation_budget_s,
            total_timeout_count=self._total_timeout_count,
            recovered_timeout_count=self._recovered_timeout_count,
            confirmed_timeout_count=self._confirmed_timeout_count,
            transition=transition,
        )

    def _confirm_locked(self, reason: str) -> HealthTransportDecision:
        if self._confirmed_cause is None:
            self._confirmed_cause = MotionFaultCause(
                source="sonic_driver",
                code="SONIC_UNAVAILABLE",
                reason=str(reason).strip()[:512]
                or "SONIC health transport confirmation failed",
            )
            self._confirmed_timeout_count += 1
            return self._decision("confirmed")
        return self._decision("latched")

    def check_deadline(self, *, now_s: float) -> HealthTransportDecision:
        """Return the current fence and terminalize an expired pending one."""

        now = self._monotonic(now_s, "now_s")
        with self._lock:
            pending = self._pending
            if (
                self._confirmed_cause is None
                and pending is not None
                and now >= pending.confirmation_deadline_s
            ):
                return self._confirm_locked(
                    "SONIC health transport confirmation deadline expired "
                    f"after {self._confirmation_budget_s:.6f}s "
                    f"(firstRequestId={pending.first_request_id}, "
                    f"latestRequestId={pending.latest_request_id}, "
                    f"count={pending.count}/{self._required_count})"
                )
            return self._decision()

    def observe_timeout(
        self,
        *,
        request_id: UUID,
        probe_started_at_s: float,
        probe_completed_at_s: float,
    ) -> HealthTransportDecision:
        """Record only one genuine socket/``TimeoutError`` probe outcome."""

        identity = self._request_id(request_id)
        started = self._monotonic(probe_started_at_s, "probe_started_at_s")
        completed = self._monotonic(
            probe_completed_at_s, "probe_completed_at_s"
        )
        if completed < started:
            raise ValueError("probe completion preceded its monotonic start")
        with self._lock:
            if self._confirmed_cause is not None:
                return self._decision("latched")
            self._total_timeout_count += 1
            previous = self._pending
            if previous is None:
                self._pending = PendingHealthTransportTimeout(
                    first_request_id=identity,
                    latest_request_id=identity,
                    count=1,
                    first_probe_started_at_s=started,
                    latest_probe_completed_at_s=completed,
                    confirmation_deadline_s=(
                        started + self._confirmation_budget_s
                    ),
                )
                self._pending_request_ids = {identity}
            else:
                if completed >= previous.confirmation_deadline_s:
                    return self._confirm_locked(
                        "SONIC health transport timeout crossed the confirmation "
                        f"deadline (firstRequestId={previous.first_request_id}, "
                        f"latestRequestId={identity}, "
                        f"count={previous.count}/{self._required_count})"
                    )
                if (
                    identity in self._pending_request_ids
                    or started < previous.latest_probe_completed_at_s
                ):
                    # Concurrent callbacks and idempotent request replays are
                    # diagnostic timeout events, but they are not a *new
                    # consecutive* sample after the first timeout completed.
                    return self._decision("ignored_overlap")
                self._pending = PendingHealthTransportTimeout(
                    first_request_id=previous.first_request_id,
                    latest_request_id=identity,
                    count=previous.count + 1,
                    first_probe_started_at_s=previous.first_probe_started_at_s,
                    latest_probe_completed_at_s=completed,
                    confirmation_deadline_s=previous.confirmation_deadline_s,
                )
                self._pending_request_ids.add(identity)

            pending = self._pending
            if completed >= pending.confirmation_deadline_s:
                return self._confirm_locked(
                    "SONIC health transport timeout crossed the confirmation "
                    f"deadline (firstRequestId={pending.first_request_id}, "
                    f"latestRequestId={pending.latest_request_id}, "
                    f"count={pending.count}/{self._required_count})"
                )
            if pending.count >= self._required_count:
                return self._confirm_locked(
                    "SONIC health transport timed out on consecutive probes "
                    f"(firstRequestId={pending.first_request_id}, "
                    f"latestRequestId={pending.latest_request_id}, "
                    f"count={pending.count}/{self._required_count})"
                )
            return self._decision("pending")

    def observe_strict_success(
        self,
        *,
        request_id: UUID,
        probe_started_at_s: float,
        probe_completed_at_s: float,
    ) -> HealthTransportDecision:
        """Clear pending only after the caller parsed a strict correlated v2."""

        self._request_id(request_id)
        started = self._monotonic(probe_started_at_s, "probe_started_at_s")
        completed = self._monotonic(
            probe_completed_at_s, "probe_completed_at_s"
        )
        if completed < started:
            raise ValueError("probe completion preceded its monotonic start")
        with self._lock:
            if self._confirmed_cause is not None:
                return self._decision("latched")
            pending = self._pending
            if pending is None:
                return self._decision()
            if completed >= pending.confirmation_deadline_s:
                return self._confirm_locked(
                    "SONIC health reply arrived after the transport "
                    f"confirmation deadline (firstRequestId="
                    f"{pending.first_request_id}, replyRequestId={request_id}, "
                    f"count={pending.count}/{self._required_count})"
                )
            if (
                request_id in self._pending_request_ids
                or started < pending.latest_probe_completed_at_s
            ):
                # Only a new probe that started after the last timed-out probe
                # completed may prove recovery.  A late reply from an older
                # overlapping request cannot reopen motion authority.
                return self._decision("ignored_overlap")
            self._pending = None
            self._pending_request_ids.clear()
            self._recovered_timeout_count += 1
            return self._decision("recovered")


class SonicHealthFaultGate:
    """Confirm age-only health faults without delaying hard fault classes.

    A single scheduler spike must not permanently poison a simulator boot.  The
    two temporal age faults therefore share one consecutive-unhealthy streak
    for the same driver boot.  Switching between LowState and control-step
    staleness cannot reset the cancellation deadline.  Terminal/unavailable
    health and a driver boot change remain immediate.  The immutable
    :class:`MotionFaultLatch` remains responsible for exactly-once publication
    and local stop I/O.
    """

    def __init__(
        self,
        *,
        plant_lowstate_stale_s: float,
        sonic_control_step_stale_s: float,
        required_stale_samples: int = STALE_HEALTH_CONFIRMATION_PROBES,
    ) -> None:
        for value, field in (
            (plant_lowstate_stale_s, "plant_lowstate_stale_s"),
            (sonic_control_step_stale_s, "sonic_control_step_stale_s"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if (
            isinstance(required_stale_samples, bool)
            or required_stale_samples != STALE_HEALTH_CONFIRMATION_PROBES
        ):
            raise ValueError(
                f"required_stale_samples is fixed at {STALE_HEALTH_CONFIRMATION_PROBES}"
            )
        self._plant_lowstate_stale_s = float(plant_lowstate_stale_s)
        self._sonic_control_step_stale_s = float(sonic_control_step_stale_s)
        self._required_count = required_stale_samples
        self._driver_boot_id: UUID | None = None
        self._pending: PendingStaleHealth | None = None
        self._filtered_transient_count = 0
        self._quarantined_stale_count = 0

    @property
    def pending(self) -> PendingStaleHealth | None:
        return self._pending

    @property
    def filtered_transient_count(self) -> int:
        return self._filtered_transient_count

    def _decision(
        self,
        *,
        health: SonicHealth,
        confirmed_cause: MotionFaultCause | None = None,
        quarantined_cause: MotionFaultCause | None = None,
        settled_safe_hold: bool = False,
    ) -> HealthFaultDecision:
        return HealthFaultDecision(
            confirmed_cause=confirmed_cause,
            pending=self._pending,
            required_count=self._required_count,
            driver_boot_id=health.driver_boot_id,
            filtered_transient_count=self._filtered_transient_count,
            quarantined_cause=quarantined_cause,
            settled_safe_hold=settled_safe_hold,
            quarantined_stale_count=self._quarantined_stale_count,
        )

    @staticmethod
    def _age_for_cause(health: SonicHealth, cause: MotionFaultCause) -> float:
        if cause.code == "PLANT_LOWSTATE_STALE":
            age = health.last_lowstate_age_s
        elif cause.code == "SONIC_CONTROL_STALE":
            age = health.last_control_step_age_s
        else:  # pragma: no cover - caller restricts this helper to stale codes
            raise ValueError("cause is not a temporal stale fault")
        if age is None or not math.isfinite(age) or age < 0:
            raise ValueError("temporal stale fault did not include a valid age")
        return float(age)

    def _clear_pending(self, *, filtered: bool) -> None:
        if filtered and self._pending is not None:
            self._filtered_transient_count += 1
        self._pending = None

    def observe(
        self,
        health: SonicHealth,
        *,
        gait_state: GaitState,
    ) -> HealthFaultDecision:
        if not isinstance(gait_state, GaitState):
            raise TypeError("gait_state must be a GaitState")
        previous_boot = self._driver_boot_id
        if previous_boot is None:
            self._driver_boot_id = health.driver_boot_id
        elif previous_boot != health.driver_boot_id:
            self._clear_pending(filtered=True)
            self._driver_boot_id = health.driver_boot_id
            return self._decision(
                health=health,
                confirmed_cause=MotionFaultCause(
                    source="sonic_driver",
                    code="SONIC_BOOT_CHANGED",
                    reason=(
                        "SONIC driver boot identity changed while the ROS "
                        f"bridge remained active ({previous_boot} -> "
                        f"{health.driver_boot_id})"
                    ),
                ),
            )

        cause = classify_sonic_health(
            health,
            plant_lowstate_stale_s=self._plant_lowstate_stale_s,
            sonic_control_step_stale_s=self._sonic_control_step_stale_s,
        )
        if cause is None:
            self._clear_pending(filtered=True)
            return self._decision(health=health)
        if cause.code not in _TEMPORAL_STALE_CODES:
            self._clear_pending(filtered=True)
            return self._decision(health=health, confirmed_cause=cause)

        settled_safe_hold = (
            gait_state is GaitState.IDLE and health.confirmed_safe_hold
        )
        if settled_safe_hold:
            self._clear_pending(filtered=True)
            self._quarantined_stale_count += 1
            return self._decision(
                health=health,
                quarantined_cause=cause,
                settled_safe_hold=True,
            )

        age = self._age_for_cause(health, cause)
        previous = self._pending
        if previous is None or previous.driver_boot_id != health.driver_boot_id:
            self._clear_pending(filtered=previous is not None)
            self._pending = PendingStaleHealth(
                driver_boot_id=health.driver_boot_id,
                source=cause.source,
                code=cause.code,
                latest_source=cause.source,
                latest_code=cause.code,
                count=1,
                first_age_s=age,
                latest_age_s=age,
            )
            return self._decision(health=health)

        self._pending = PendingStaleHealth(
            driver_boot_id=previous.driver_boot_id,
            source=previous.source,
            code=previous.code,
            latest_source=cause.source,
            latest_code=cause.code,
            count=previous.count + 1,
            first_age_s=previous.first_age_s,
            latest_age_s=age,
        )
        return self._decision(
            health=health,
            confirmed_cause=(
                cause if self._pending.count >= self._required_count else None
            ),
        )


def require_fresh_motion_health(
    health: SonicHealth,
    *,
    plant_lowstate_stale_s: float,
    sonic_control_step_stale_s: float,
) -> None:
    """Reject control acquisition unless the current motion path is fresh.

    A settled safe hold may quarantine age-only faults for observation, but it
    never converts stale evidence into new motion authority.
    """

    cause = classify_sonic_health(
        health,
        plant_lowstate_stale_s=plant_lowstate_stale_s,
        sonic_control_step_stale_s=sonic_control_step_stale_s,
    )
    if cause is not None:
        raise ValueError(f"{cause.code}: {cause.reason}")
    if not health.confirmed_safe_hold:
        raise ValueError(
            "SONIC control acquisition requires a confirmed settled safety hold"
        )


def classify_sonic_health(
    health: SonicHealth,
    *,
    plant_lowstate_stale_s: float,
    sonic_control_step_stale_s: float,
) -> MotionFaultCause | None:
    """Map an authenticated health response to one fail-closed first cause."""

    threshold = float(plant_lowstate_stale_s)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("plant LowState stale threshold must be finite and positive")
    control_threshold = float(sonic_control_step_stale_s)
    if not math.isfinite(control_threshold) or control_threshold <= 0:
        raise ValueError("SONIC control stale threshold must be finite and positive")
    if health.terminal_fault:
        return MotionFaultCause(
            source="sonic_driver",
            code="SONIC_TERMINAL_FAULT",
            reason=health.reason or "SONIC driver latched a terminal fault",
        )
    if health.status != "ready" or not health.motion_ready:
        return MotionFaultCause(
            source="sonic_driver",
            code="SONIC_UNAVAILABLE",
            reason=f"SONIC driver is not motion-ready (status={health.status})",
        )
    if (
        health.last_lowstate_age_s is None
        or health.last_lowstate_age_s > threshold
    ):
        age = (
            "unavailable"
            if health.last_lowstate_age_s is None
            else f"{health.last_lowstate_age_s:.6f}s"
        )
        return MotionFaultCause(
            source="sonic_driver",
            code="PLANT_LOWSTATE_STALE",
            reason=(
                f"LowState reception age {age} exceeds the "
                f"{threshold:.6f}s safety bound"
            ),
        )
    if (
        health.last_control_step_age_s is None
        or health.last_control_step_age_s > control_threshold
    ):
        age = (
            "unavailable"
            if health.last_control_step_age_s is None
            else f"{health.last_control_step_age_s:.6f}s"
        )
        return MotionFaultCause(
            source="sonic_driver",
            code="SONIC_CONTROL_STALE",
            reason=(
                f"SONIC control-step age {age} exceeds the "
                f"{control_threshold:.6f}s safety bound"
            ),
        )
    return None
