"""Strict helpers for the frozen robot-agent.v1 JSON WebSocket envelope."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Mapping
from uuid import UUID, uuid4


PROTOCOL_VERSION = "1.0"
SUBPROTOCOL = "robot-agent.v1"
MAX_TEXT_FRAME_BYTES = 65_536
_ENVELOPE_FIELDS = {"version", "type", "message_id", "robot_id", "session_id", "sent_at", "payload"}


class WireProtocolError(ValueError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise WireProtocolError("wire timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WireProtocolError("wire timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WireProtocolError("wire timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def build_message(
    message_type: str,
    *,
    robot_id: str,
    session_id: str | None,
    payload: Mapping[str, Any],
) -> str:
    value = {
        "version": PROTOCOL_VERSION,
        "type": message_type,
        "message_id": str(uuid4()),
        "robot_id": robot_id,
        "session_id": session_id,
        "sent_at": utc_now().isoformat().replace("+00:00", "Z"),
        "payload": dict(payload),
    }
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
        raise WireProtocolError("outbound WebSocket frame exceeds 65536 bytes")
    return encoded


def parse_message(
    raw: str | bytes,
    *,
    expected_robot_id: str,
    expected_session_id: str | None | object = ...,
) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raise WireProtocolError("binary WebSocket frames are forbidden")
    if not isinstance(raw, str):
        raise WireProtocolError("WebSocket frame must be UTF-8 text")
    if len(raw.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
        raise WireProtocolError("WebSocket frame exceeds 65536 bytes")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WireProtocolError("WebSocket frame is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != _ENVELOPE_FIELDS:
        raise WireProtocolError("robot-agent.v1 envelope fields are invalid")
    if value["version"] != PROTOCOL_VERSION:
        raise WireProtocolError("robot-agent.v1 version mismatch")
    if not isinstance(value["type"], str) or not value["type"]:
        raise WireProtocolError("wire message type is invalid")
    try:
        UUID(value["message_id"])
    except (TypeError, ValueError) as exc:
        raise WireProtocolError("wire message_id must be a UUID") from exc
    if value["robot_id"] != expected_robot_id:
        raise WireProtocolError("wire robot_id mismatch")
    raw_session = value["session_id"]
    if raw_session is not None:
        try:
            UUID(raw_session)
        except (TypeError, ValueError) as exc:
            raise WireProtocolError("wire session_id must be a UUID or null") from exc
    if expected_session_id is not ... and raw_session != expected_session_id:
        raise WireProtocolError("wire session_id mismatch")
    parse_timestamp(value["sent_at"])
    if not isinstance(value["payload"], dict):
        raise WireProtocolError("wire payload must be an object")
    return value


def require_payload_fields(
    payload: Mapping[str, Any],
    fields: set[str],
    *,
    optional_fields: set[str] | None = None,
) -> None:
    optional = optional_fields or set()
    if not fields.issubset(payload) or not set(payload).issubset(fields | optional):
        raise WireProtocolError("wire payload fields are invalid")
