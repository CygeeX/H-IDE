from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from robot_agent.protocol import (
    FrameValidationError,
    parse_wire_message,
    protocol_json_schema,
    serialize_wire_message,
)

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)


def envelope(message_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "version": "1.0",
        "type": message_type,
        "message_id": str(uuid4()),
        "robot_id": "g1-dev",
        "session_id": str(uuid4()),
        "sent_at": NOW.isoformat().replace("+00:00", "Z"),
        "payload": payload,
    }


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        (
            "session.welcome",
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        ),
        ("control.acquire", {"operator_id": "operator-7"}),
        (
            "control.status",
            {"state": "ACTIVE", "reason": "control acquired", "lease_sequence": 0},
        ),
        ("lease.status", {"sequence": 9, "expires_at": "2026-08-07T12:00:02Z"}),
        ("command.cancel", {"command_id": str(uuid4()), "reason": "operator request"}),
        (
            "command.status",
            {
                "command_id": str(uuid4()),
                "state": "RUNNING",
                "detail": "goal accepted",
                "progress": 0.25,
                "result": None,
            },
        ),
        (
            "state.snapshot",
            {
                "sequence": 10,
                "observed_at": "2026-08-07T12:00:00Z",
                "state": {"mode": "IDLE", "battery": {"percentage": 0.87}},
            },
        ),
        (
            "state.update",
            {
                "sequence": 11,
                "observed_at": "2026-08-07T12:00:00Z",
                "changes": {"mode": "MOVING"},
            },
        ),
        (
            "safety.event",
            {
                "code": "LEASE_EXPIRED",
                "state": "SAFE_HOLD",
                "reason": "lease renewal timed out",
                "latched": True,
            },
        ),
        (
            "error",
            {
                "code": "INVALID_MESSAGE",
                "message": "unknown field",
                "related_message_id": str(uuid4()),
                "retryable": False,
            },
        ),
    ],
)
def test_parse_supports_every_server_and_control_message_family(
    message_type: str, payload: dict[str, object]
) -> None:
    parsed = parse_wire_message(
        json.dumps(envelope(message_type, payload)),
        expected_robot_id="g1-dev",
        received_at=NOW,
    )

    assert parsed.type == message_type


def test_parse_rejects_expired_command_and_future_envelope() -> None:
    expired = envelope(
        "command.request",
        {
            "command_id": str(uuid4()),
            "kind": "task.execute",
            "issued_at": (NOW - timedelta(seconds=3)).isoformat(),
            "ttl_ms": 2000,
            "params": {},
        },
    )
    with pytest.raises(FrameValidationError, match="expired"):
        parse_wire_message(json.dumps(expired), expected_robot_id="g1-dev", received_at=NOW)

    future = envelope("control.acquire", {"operator_id": "operator-7"})
    future["sent_at"] = (NOW + timedelta(seconds=6)).isoformat()
    with pytest.raises(FrameValidationError, match="future"):
        parse_wire_message(json.dumps(future), expected_robot_id="g1-dev", received_at=NOW)


def test_protocol_schema_is_versioned_and_closed() -> None:
    schema = protocol_json_schema()

    assert schema["title"] == "robot-agent.v1"
    assert schema["$id"].endswith("/robot-agent.v1.schema.json")
    assert "discriminator" in schema


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        (
            "control.status",
            {"state": "ACTIVE", "reason": "control acquired", "lease_sequence": 0},
        ),
        (
            "safety.event",
            {
                "code": "LEASE_EXPIRED",
                "state": "SAFE_HOLD",
                "reason": "lease renewal timed out",
                "latched": True,
            },
        ),
    ],
)
def test_legacy_safe_stop_payload_does_not_gain_implied_truth_on_round_trip(
    message_type: str,
    payload: dict[str, object],
) -> None:
    parsed = parse_wire_message(
        json.dumps(envelope(message_type, payload)),
        expected_robot_id="g1-dev",
        received_at=NOW,
    )

    serialized = json.loads(serialize_wire_message(parsed))

    assert serialized["payload"] == payload


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        (
            "control.status",
            {
                "state": "ACTIVE",
                "reason": "control acquired",
                "lease_sequence": 0,
                "authority_revoked": False,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 7,
            },
        ),
        (
            "safety.event",
            {
                "code": "SAFE_STOP_FAILED",
                "state": "SAFE_HOLD",
                "reason": "physical stop failed",
                "latched": True,
                "authority_revoked": True,
                "safe_stop_state": "failed",
                "safe_stop_epoch": 8,
            },
        ),
    ],
)
def test_explicit_safe_stop_truth_is_preserved_on_round_trip(
    message_type: str,
    payload: dict[str, object],
) -> None:
    parsed = parse_wire_message(
        json.dumps(envelope(message_type, payload)),
        expected_robot_id="g1-dev",
        received_at=NOW,
    )

    serialized = json.loads(serialize_wire_message(parsed))

    assert serialized["payload"] == payload


@pytest.mark.parametrize(
    ("message_type", "payload"),
    [
        (
            "control.status",
            {
                "state": "ACTIVE",
                "reason": "partial truth",
                "lease_sequence": 0,
                "authority_revoked": False,
            },
        ),
        (
            "control.status",
            {
                "state": "ACTIVE",
                "reason": "contradictory authority",
                "lease_sequence": 0,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 7,
            },
        ),
        (
            "control.status",
            {
                "state": "SAFE_HOLD",
                "reason": "contradictory authority",
                "lease_sequence": None,
                "authority_revoked": False,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 7,
            },
        ),
        (
            "safety.event",
            {
                "code": "LEASE_EXPIRED",
                "state": "ACTIVE",
                "reason": "contradictory stop state",
                "latched": True,
                "authority_revoked": False,
                "safe_stop_state": "requested",
                "safe_stop_epoch": 8,
            },
        ),
        (
            "safety.event",
            {
                "code": "LEASE_EXPIRED",
                "state": "SAFE_HOLD",
                "reason": "explicit null truth",
                "latched": True,
                "authority_revoked": None,
                "safe_stop_state": None,
                "safe_stop_epoch": None,
            },
        ),
    ],
)
def test_safe_stop_truth_rejects_partial_or_contradictory_values(
    message_type: str,
    payload: dict[str, object],
) -> None:
    with pytest.raises(FrameValidationError):
        parse_wire_message(
            json.dumps(envelope(message_type, payload)),
            expected_robot_id="g1-dev",
            received_at=NOW,
        )
