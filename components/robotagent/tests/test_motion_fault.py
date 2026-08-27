from __future__ import annotations

import json
from copy import deepcopy
from uuid import uuid4

import pytest

from robot_agent.motion_fault import (
    DeduplicationDecision,
    MotionFaultDeduplicator,
    MotionSafetyFault,
)


def fault_payload(
    *,
    event_id: str | None = None,
    producer_boot_id: str | None = None,
    source_sequence: int = 1,
    robot_id: str = "g1-sim",
    localization_epoch_id: str = "g1-sim-map-epoch-1",
) -> dict[str, object]:
    return {
        "schemaVersion": "h-ide.motion-safety-fault/v1",
        "eventId": event_id or str(uuid4()),
        "robotId": robot_id,
        "localizationEpochId": localization_epoch_id,
        "producerBootId": producer_boot_id or str(uuid4()),
        "sourceSequence": source_sequence,
        "controlGeneration": 7,
        "source": "navigation_guard",
        "code": "POSE_DIVERGED",
        "reason": "measured base pose diverged from the controller",
        "latched": True,
        "detectedAt": "2026-08-19T07:00:00.123Z",
        "downstreamBootId": None,
        "probeRequestId": None,
    }


def decode(payload: dict[str, object]) -> MotionSafetyFault:
    return MotionSafetyFault.from_json(json.dumps(payload))


def test_motion_fault_v1_requires_exact_strict_contract() -> None:
    payload = fault_payload()
    fault = decode(payload)
    assert fault.robot_id == "g1-sim"
    assert fault.source_sequence == 1
    assert fault.control_generation == 7

    invalid_payloads: list[dict[str, object]] = []
    for field in tuple(payload):
        candidate = deepcopy(payload)
        candidate.pop(field)
        invalid_payloads.append(candidate)
    candidate = deepcopy(payload)
    candidate["unexpected"] = True
    invalid_payloads.append(candidate)
    for field, value in (
        ("eventId", str(uuid4()).upper()),
        ("sourceSequence", True),
        ("controlGeneration", -1),
        ("source", "unknown"),
        ("code", "not_uppercase"),
        ("latched", False),
        ("detectedAt", "2026-08-19T07:00:00"),
    ):
        candidate = deepcopy(payload)
        candidate[field] = value
        invalid_payloads.append(candidate)

    for candidate in invalid_payloads:
        with pytest.raises(ValueError):
            decode(candidate)


def test_motion_fault_deduplicates_event_and_boot_sequence_without_cross_boot_poisoning() -> None:
    deduplicator = MotionFaultDeduplicator(maximum_events=2)
    boot = str(uuid4())
    first = decode(fault_payload(producer_boot_id=boot, source_sequence=1))
    second = decode(fault_payload(producer_boot_id=boot, source_sequence=2))
    conflict = decode(
        fault_payload(producer_boot_id=boot, source_sequence=2)
    )
    stale = decode(fault_payload(producer_boot_id=boot, source_sequence=1))
    restarted = decode(fault_payload(producer_boot_id=str(uuid4()), source_sequence=1))

    assert deduplicator.observe(first) is DeduplicationDecision.ACCEPTED
    assert deduplicator.observe(first) is DeduplicationDecision.DUPLICATE
    assert deduplicator.observe(second) is DeduplicationDecision.ACCEPTED
    assert deduplicator.observe(conflict) is DeduplicationDecision.SEQUENCE_CONFLICT
    assert deduplicator.observe(stale) is DeduplicationDecision.STALE_SEQUENCE
    assert deduplicator.observe(restarted) is DeduplicationDecision.ACCEPTED
