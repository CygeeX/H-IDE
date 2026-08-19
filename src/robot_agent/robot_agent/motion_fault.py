"""Strict robot-scoped motion-safety-fault ingress contract."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID

MOTION_FAULT_SCHEMA_VERSION = "h-ide.motion-safety-fault/v1"
MOTION_FAULT_MAX_BYTES = 4096

_FIELDS = frozenset(
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
_SOURCES = frozenset(
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


def _uuid(value: object, field: str) -> UUID:
    try:
        parsed = UUID(str(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a UUID") from exc
    if str(parsed) != str(value):
        raise ValueError(f"{field} must use canonical UUID text")
    return parsed


def _text(value: object, field: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{field} must be a non-empty string of at most {maximum} characters"
        )
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _detected_at(value: object) -> str:
    text = _text(value, "detectedAt", maximum=64)
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
    downstream_boot_id: UUID | None
    probe_request_id: UUID | None

    @classmethod
    def from_json(cls, raw: str) -> MotionSafetyFault:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MOTION_FAULT_MAX_BYTES:
            raise ValueError("motion-fault event must be bounded JSON text")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("motion-fault event is not valid JSON") from exc
        if not isinstance(value, dict) or set(value) != _FIELDS:
            raise ValueError("motion-fault event fields do not match v1")
        if value.get("schemaVersion") != MOTION_FAULT_SCHEMA_VERSION:
            raise ValueError("unsupported motion-fault schema")
        if value.get("latched") is not True:
            raise ValueError("motion-fault event must be latched")
        robot_id = value.get("robotId")
        if not isinstance(robot_id, str) or _ROBOT_ID.fullmatch(robot_id) is None:
            raise ValueError("robotId is invalid")
        localization_epoch_id = _text(
            value.get("localizationEpochId"),
            "localizationEpochId",
            maximum=256,
        )
        source = value.get("source")
        if source not in _SOURCES:
            raise ValueError("source is not a supported motion-fault producer")
        code = value.get("code")
        if not isinstance(code, str) or _CODE.fullmatch(code) is None:
            raise ValueError("code must be an uppercase machine-readable identifier")
        downstream = value.get("downstreamBootId")
        probe = value.get("probeRequestId")
        return cls(
            event_id=_uuid(value.get("eventId"), "eventId"),
            robot_id=robot_id,
            localization_epoch_id=localization_epoch_id,
            producer_boot_id=_uuid(value.get("producerBootId"), "producerBootId"),
            source_sequence=_integer(
                value.get("sourceSequence"), "sourceSequence", minimum=1
            ),
            control_generation=_integer(
                value.get("controlGeneration"), "controlGeneration", minimum=0
            ),
            source=source,
            code=code,
            reason=_text(value.get("reason"), "reason", maximum=512),
            detected_at=_detected_at(value.get("detectedAt")),
            downstream_boot_id=(
                None if downstream is None else _uuid(downstream, "downstreamBootId")
            ),
            probe_request_id=(
                None if probe is None else _uuid(probe, "probeRequestId")
            ),
        )


class DeduplicationDecision(str, Enum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE_SEQUENCE = "stale_sequence"
    SEQUENCE_CONFLICT = "sequence_conflict"


class MotionFaultDeduplicator:
    """Bounded boot/sequence and event-ID replay protection."""

    def __init__(self, *, maximum_events: int = 1024) -> None:
        if maximum_events < 1:
            raise ValueError("maximum_events must be positive")
        self._maximum_events = maximum_events
        self._events: OrderedDict[UUID, None] = OrderedDict()
        self._boot_sequences: OrderedDict[UUID, tuple[int, UUID]] = OrderedDict()

    def observe(self, event: MotionSafetyFault) -> DeduplicationDecision:
        if event.event_id in self._events:
            self._events.move_to_end(event.event_id)
            return DeduplicationDecision.DUPLICATE
        prior = self._boot_sequences.get(event.producer_boot_id)
        if prior is not None:
            self._boot_sequences.move_to_end(event.producer_boot_id)
            sequence, event_id = prior
            if event.source_sequence == sequence:
                return (
                    DeduplicationDecision.DUPLICATE
                    if event.event_id == event_id
                    else DeduplicationDecision.SEQUENCE_CONFLICT
                )
            if event.source_sequence < sequence:
                return DeduplicationDecision.STALE_SEQUENCE
        self._boot_sequences[event.producer_boot_id] = (
            event.source_sequence,
            event.event_id,
        )
        self._events[event.event_id] = None
        while len(self._events) > self._maximum_events:
            self._events.popitem(last=False)
        while len(self._boot_sequences) > self._maximum_events:
            self._boot_sequences.popitem(last=False)
        return DeduplicationDecision.ACCEPTED
