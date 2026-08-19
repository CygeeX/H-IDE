"""Strict, versioned Robot Agent wire protocol models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)

MAX_TEXT_FRAME_BYTES = 65_536
LOGICAL_COMMAND_PATTERN = r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$"
_SAFE_STOP_TRUTH_FIELDS = frozenset(
    {"authority_revoked", "safe_stop_state", "safe_stop_epoch"}
)


class FrameValidationError(ValueError):
    """Raised when an incoming WebSocket frame violates the wire contract."""


class StrictModel(BaseModel):
    """Protocol model that never accepts undeclared fields or type coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


def _validate_safe_stop_truth(
    *,
    state: str,
    authority_revoked: bool | None,
    safe_stop_state: str | None,
    safe_stop_epoch: int | None,
    supplied_fields: set[str],
) -> None:
    """Validate one optional, atomic physical safe-stop truth tuple."""

    supplied_truth_fields = supplied_fields.intersection(_SAFE_STOP_TRUTH_FIELDS)
    if not supplied_truth_fields:
        return
    if supplied_truth_fields != _SAFE_STOP_TRUTH_FIELDS or any(
        value is None
        for value in (authority_revoked, safe_stop_state, safe_stop_epoch)
    ):
        raise ValueError(
            "authority_revoked, safe_stop_state, and safe_stop_epoch "
            "must be provided together"
        )
    if state == "ACTIVE":
        if authority_revoked or safe_stop_state != "confirmed":
            raise ValueError(
                "ACTIVE requires authority_revoked=false and "
                "safe_stop_state=confirmed"
            )
        return
    if not authority_revoked:
        raise ValueError("inactive control state requires authority_revoked=true")


class Envelope(StrictModel):
    """Fields shared by every v1 protocol message."""

    version: Literal["1.0"]
    message_id: UUID
    robot_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    sent_at: AwareDatetime


class SessionHelloPayload(StrictModel):
    agent_version: str = Field(min_length=1, max_length=64)
    boot_id: UUID
    supported_versions: list[Literal["1.0"]] = Field(min_length=1)
    transport_mode: Literal["client", "server"]
    capabilities_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    robot_profile: dict[str, JsonValue] | None = None
    capability_manifest: dict[str, JsonValue] | None = None


class SessionHello(Envelope):
    type: Literal["session.hello"]
    session_id: None
    payload: SessionHelloPayload


class CommandRequestPayload(StrictModel):
    command_id: UUID
    kind: str = Field(min_length=3, max_length=128, pattern=LOGICAL_COMMAND_PATTERN)
    issued_at: AwareDatetime
    ttl_ms: int = Field(gt=0, le=60_000)
    params: dict[str, JsonValue]


class CommandRequest(Envelope):
    type: Literal["command.request"]
    session_id: UUID
    payload: CommandRequestPayload


class LeaseRenewPayload(StrictModel):
    sequence: int = Field(ge=0)


class LeaseRenew(Envelope):
    type: Literal["lease.renew"]
    session_id: UUID
    payload: LeaseRenewPayload


class SessionWelcomePayload(StrictModel):
    accepted_version: Literal["1.0"]
    lease_interval_ms: int = Field(gt=0, le=5_000)
    lease_timeout_ms: int = Field(gt=0, le=30_000)
    control_state: Literal["UNARMED", "ACTIVE", "SAFE_HOLD"]


class SessionWelcome(Envelope):
    type: Literal["session.welcome"]
    session_id: UUID
    payload: SessionWelcomePayload


class SessionGoodbyePayload(StrictModel):
    reason: str = Field(min_length=1, max_length=256)


class SessionGoodbye(Envelope):
    type: Literal["session.goodbye"]
    session_id: UUID
    payload: SessionGoodbyePayload


class ControlAcquirePayload(StrictModel):
    operator_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._@-]*$",
    )


class ControlAcquire(Envelope):
    type: Literal["control.acquire"]
    session_id: UUID
    payload: ControlAcquirePayload


class ControlStatusPayload(StrictModel):
    state: Literal["UNARMED", "ACTIVE", "SAFE_HOLD"]
    reason: str = Field(min_length=1, max_length=256)
    lease_sequence: int | None = Field(default=None, ge=0)
    # Additive V1 safety truth is atomic. Legacy three-field payloads retain no
    # implied physical-stop claim when parsed and serialized again; the runtime
    # explicitly supplies all three measured fields.
    authority_revoked: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    safe_stop_state: Literal["requested", "confirmed", "failed"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    safe_stop_epoch: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_safe_stop_truth(self) -> ControlStatusPayload:
        _validate_safe_stop_truth(
            state=self.state,
            authority_revoked=self.authority_revoked,
            safe_stop_state=self.safe_stop_state,
            safe_stop_epoch=self.safe_stop_epoch,
            supplied_fields=self.model_fields_set,
        )
        return self


class ControlStatus(Envelope):
    type: Literal["control.status"]
    session_id: UUID
    payload: ControlStatusPayload


class LeaseStatusPayload(StrictModel):
    sequence: int = Field(ge=0)
    expires_at: AwareDatetime


class LeaseStatus(Envelope):
    type: Literal["lease.status"]
    session_id: UUID
    payload: LeaseStatusPayload


class CommandCancelPayload(StrictModel):
    command_id: UUID
    reason: str = Field(min_length=1, max_length=256)


class CommandCancel(Envelope):
    type: Literal["command.cancel"]
    session_id: UUID
    payload: CommandCancelPayload


class CommandStatusPayload(StrictModel):
    command_id: UUID
    state: Literal[
        "RECEIVED",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "REJECTED",
        "TIMED_OUT",
        "ABORTED_AGENT_RESTART",
    ]
    detail: str = Field(min_length=1, max_length=512)
    progress: float | None = Field(default=None, ge=0.0, le=1.0)
    result: JsonValue | None = None


class CommandStatus(Envelope):
    type: Literal["command.status"]
    session_id: UUID
    payload: CommandStatusPayload


class StateSnapshotPayload(StrictModel):
    sequence: int = Field(ge=0)
    observed_at: AwareDatetime
    state: dict[str, JsonValue]


class StateSnapshot(Envelope):
    type: Literal["state.snapshot"]
    session_id: UUID
    payload: StateSnapshotPayload


class StateUpdatePayload(StrictModel):
    sequence: int = Field(ge=0)
    observed_at: AwareDatetime
    changes: dict[str, JsonValue]


class StateUpdate(Envelope):
    type: Literal["state.update"]
    session_id: UUID
    payload: StateUpdatePayload


class SafetyEventPayload(StrictModel):
    code: str = Field(min_length=1, max_length=128, pattern=r"^[A-Z][A-Z0-9_]*$")
    state: Literal["UNARMED", "ACTIVE", "SAFE_HOLD"]
    reason: str = Field(min_length=1, max_length=512)
    latched: bool
    authority_revoked: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    safe_stop_state: Literal["requested", "confirmed", "failed"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    safe_stop_epoch: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )

    @model_validator(mode="after")
    def validate_safe_stop_truth(self) -> SafetyEventPayload:
        _validate_safe_stop_truth(
            state=self.state,
            authority_revoked=self.authority_revoked,
            safe_stop_state=self.safe_stop_state,
            safe_stop_epoch=self.safe_stop_epoch,
            supplied_fields=self.model_fields_set,
        )
        return self


class SafetyEvent(Envelope):
    type: Literal["safety.event"]
    session_id: UUID
    payload: SafetyEventPayload


class ErrorPayload(StrictModel):
    code: str = Field(min_length=1, max_length=128, pattern=r"^[A-Z][A-Z0-9_]*$")
    message: str = Field(min_length=1, max_length=512)
    related_message_id: UUID | None = None
    retryable: bool


class ErrorMessage(Envelope):
    type: Literal["error"]
    session_id: UUID | None
    payload: ErrorPayload


WireMessage = Annotated[
    SessionHello
    | SessionWelcome
    | SessionGoodbye
    | ControlAcquire
    | ControlStatus
    | LeaseRenew
    | LeaseStatus
    | CommandRequest
    | CommandCancel
    | CommandStatus
    | StateSnapshot
    | StateUpdate
    | SafetyEvent
    | ErrorMessage,
    Field(discriminator="type"),
]
_WIRE_MESSAGE_ADAPTER: TypeAdapter[WireMessage] = TypeAdapter(WireMessage)


def parse_wire_message(
    raw: str | bytes,
    *,
    expected_robot_id: str,
    max_frame_bytes: int = MAX_TEXT_FRAME_BYTES,
    received_at: datetime | None = None,
    max_message_age: timedelta = timedelta(seconds=30),
    max_future_skew: timedelta = timedelta(seconds=5),
) -> WireMessage:
    """Parse and validate one UTF-8 JSON text frame."""
    if isinstance(raw, bytes):
        raise FrameValidationError("binary WebSocket frames are not supported")
    if not isinstance(raw, str):
        raise FrameValidationError("WebSocket frame must be text")
    frame_size = len(raw.encode("utf-8"))
    if frame_size > max_frame_bytes:
        raise FrameValidationError(
            f"WebSocket frame exceeds {max_frame_bytes} bytes: received {frame_size}"
        )

    try:
        message = _WIRE_MESSAGE_ADAPTER.validate_json(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        raise FrameValidationError(str(exc)) from exc
    if message.robot_id != expected_robot_id:
        raise FrameValidationError(
            f"robot_id mismatch: expected {expected_robot_id!r}, received {message.robot_id!r}"
        )
    now = received_at or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise FrameValidationError("received_at must include a timezone")
    if message.sent_at < now - max_message_age:
        raise FrameValidationError("message sent_at is stale")
    if message.sent_at > now + max_future_skew:
        raise FrameValidationError("message sent_at is too far in the future")
    if isinstance(message, CommandRequest):
        expires_at = message.payload.issued_at + timedelta(milliseconds=message.payload.ttl_ms)
        if expires_at <= now:
            raise FrameValidationError("command has expired")
        if message.payload.issued_at > now + max_future_skew:
            raise FrameValidationError("command issued_at is too far in the future")
    return message


def serialize_wire_message(message: WireMessage) -> str:
    """Serialize one validated wire message as compact JSON text."""
    return message.model_dump_json()


def protocol_json_schema() -> dict[str, JsonValue]:
    """Return the canonical JSON Schema for ``robot-agent.v1``."""
    schema = _WIRE_MESSAGE_ADAPTER.json_schema()
    schema["$id"] = "https://phibotics.local/schemas/robot-agent.v1.schema.json"
    schema["title"] = "robot-agent.v1"
    return schema
