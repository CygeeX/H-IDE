from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from robot_agent.protocol import FrameValidationError, parse_wire_message, serialize_wire_message


def valid_hello() -> dict[str, object]:
    return {
        "version": "1.0",
        "type": "session.hello",
        "message_id": str(uuid4()),
        "robot_id": "g1-dev",
        "session_id": None,
        "sent_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": {
            "agent_version": "0.1.0",
            "boot_id": str(uuid4()),
            "supported_versions": ["1.0"],
            "transport_mode": "client",
            "capabilities_digest": "sha256:" + "a" * 64,
        },
    }


def test_parse_wire_message_accepts_strict_hello_envelope() -> None:
    parsed = parse_wire_message(json.dumps(valid_hello()), expected_robot_id="g1-dev")

    assert parsed.type == "session.hello"
    assert parsed.robot_id == "g1-dev"
    assert parsed.payload.agent_version == "0.1.0"


def test_parse_wire_message_rejects_unknown_envelope_field() -> None:
    raw = valid_hello()
    raw["unexpected"] = True

    with pytest.raises(FrameValidationError, match="unexpected"):
        parse_wire_message(json.dumps(raw), expected_robot_id="g1-dev")


def test_parse_wire_message_rejects_binary_and_oversized_frames() -> None:
    encoded = json.dumps(valid_hello())

    with pytest.raises(FrameValidationError, match="binary"):
        parse_wire_message(encoded.encode(), expected_robot_id="g1-dev")
    with pytest.raises(FrameValidationError, match="65536"):
        parse_wire_message(encoded + " " * 65536, expected_robot_id="g1-dev")


def test_parse_wire_message_rejects_wrong_robot() -> None:
    with pytest.raises(FrameValidationError, match="robot_id mismatch"):
        parse_wire_message(json.dumps(valid_hello()), expected_robot_id="g1-other")


def test_parse_wire_message_accepts_command_request() -> None:
    session_id = uuid4()
    command_id = uuid4()
    raw = {
        "version": "1.0",
        "type": "command.request",
        "message_id": str(uuid4()),
        "robot_id": "g1-dev",
        "session_id": str(session_id),
        "sent_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": {
            "command_id": str(command_id),
            "kind": "task.execute",
            "issued_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ttl_ms": 5000,
            "params": {"order": 5},
        },
    }

    parsed = parse_wire_message(json.dumps(raw), expected_robot_id="g1-dev")

    assert parsed.type == "command.request"
    assert parsed.payload.command_id == command_id
    assert parsed.payload.params == {"order": 5}


def test_protocol_round_trip_preserves_lease_message() -> None:
    raw = {
        "version": "1.0",
        "type": "lease.renew",
        "message_id": str(uuid4()),
        "robot_id": "g1-dev",
        "session_id": str(uuid4()),
        "sent_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": {"sequence": 7},
    }
    parsed = parse_wire_message(json.dumps(raw), expected_robot_id="g1-dev")

    reparsed = parse_wire_message(serialize_wire_message(parsed), expected_robot_id="g1-dev")

    assert reparsed == parsed


def test_control_release_is_strict_and_round_trips() -> None:
    raw = {
        "version": "1.0",
        "type": "control.release",
        "message_id": str(uuid4()),
        "robot_id": "g1-dev",
        "session_id": str(uuid4()),
        "sent_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": {"reason": "approved mission reached a terminal state"},
    }

    parsed = parse_wire_message(json.dumps(raw), expected_robot_id="g1-dev")
    reparsed = parse_wire_message(
        serialize_wire_message(parsed), expected_robot_id="g1-dev"
    )

    assert parsed.type == "control.release"
    assert reparsed == parsed
    raw["payload"] = {"reason": "done", "force": True}
    with pytest.raises(FrameValidationError, match="force"):
        parse_wire_message(json.dumps(raw), expected_robot_id="g1-dev")


def test_command_request_rejects_invalid_kind_and_ttl() -> None:
    raw = {
        "version": "1.0",
        "type": "command.request",
        "message_id": str(uuid4()),
        "robot_id": "g1-dev",
        "session_id": str(uuid4()),
        "sent_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload": {
            "command_id": str(uuid4()),
            "kind": "/cmd_vel",
            "issued_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "ttl_ms": 0,
            "params": {},
        },
    }

    with pytest.raises(FrameValidationError):
        parse_wire_message(json.dumps(raw), expected_robot_id="g1-dev")
