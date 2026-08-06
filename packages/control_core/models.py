"""Transport-independent, validated domain contracts for robot actions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import copy
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .validation import (
    CLOCK_SKEW,
    as_utc,
    ensure_acyclic_steps,
    freeze_json_object,
    freeze_json_value,
    non_empty_id,
    require_json_schema,
    validate_expiry,
    utc_now,
)


class _ValueEnum(str, Enum):
    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            lowered = value.lower()
            for member in cls:
                if member.value.lower() == lowered or member.name.lower() == lowered:
                    return member
        return None


class CommandSource(_ValueEnum):
    MANUAL = "manual"
    AGENT = "agent"
    VOICE = "voice"
    SYSTEM = "system"


class Transport(_ValueEnum):
    HTTP = "http"
    MQTT = "mqtt"
    ROS2 = "ros2"
    ROBOT_AGENT = "robot_agent"
    FAKE = "fake"


class CommandKind(_ValueEnum):
    MOTION = "motion"
    SAFETY = "safety"
    NON_MOTION = "non_motion"


class CommandStatus(_ValueEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class RobotStatus(_ValueEnum):
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    ESTOP = "estop"


class SafetyMode(_ValueEnum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    STOPPED = "stopped"
    ESTOP = "estop"
    UNKNOWN = "unknown"


class ContractModel(BaseModel):
    # Domain messages are snapshots.  Use ``model_copy(update=...)`` to create
    # a new version rather than mutating an accepted command or plan in place.
    # This also makes failed cross-field validation unable to leave partial
    # assignments behind.
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False):
        """Return a revalidated snapshot instead of Pydantic's unchecked copy."""
        data = self.model_dump(mode="python", by_alias=False)
        if deep:
            data = copy.deepcopy(data)
        if update:
            normalized: dict[str, Any] = {}
            fields = type(self).model_fields
            for key, value in update.items():
                canonical = key
                for name, field in fields.items():
                    aliases = {name}
                    if field.alias:
                        aliases.add(field.alias)
                    validation_alias = field.validation_alias
                    if isinstance(validation_alias, AliasChoices):
                        aliases.update(validation_alias.choices)
                    elif isinstance(validation_alias, str):
                        aliases.add(validation_alias)
                    if key in aliases:
                        canonical = name
                        break
                normalized[canonical] = value
            data.update(normalized)
        return type(self).model_validate(data)


class CommandDefinition(ContractModel):
    commandKey: str = Field(..., alias="command_key")
    commandVersion: str = Field(
        ..., alias="command_version", validation_alias=AliasChoices("command_version", "version")
    )
    payloadSchema: dict[str, Any] = Field(
        default_factory=dict,
        alias="payload_schema",
        validation_alias=AliasChoices("payload_schema", "schema"),
        validate_default=True,
    )
    transports: list[Transport] = Field(default_factory=lambda: list(Transport), validate_default=True)
    kind: CommandKind = CommandKind.MOTION
    description: str | None = None

    @field_validator("commandKey", "commandVersion")
    @classmethod
    def _non_empty(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("payloadSchema")
    @classmethod
    def _schema(cls, value):
        return freeze_json_object(require_json_schema(value))

    @field_validator("transports")
    @classmethod
    def _transports(cls, value):
        if not value:
            raise ValueError("at least one transport is required")
        from .validation import FrozenList
        return FrozenList(dict.fromkeys(value))


class CommandRequest(ContractModel):
    commandId: str = Field(..., alias="command_id")
    robotId: str = Field(..., alias="robot_id")
    source: CommandSource
    transport: Transport
    commandKey: str = Field(..., alias="command_key")
    commandVersion: str = Field(..., alias="command_version")
    payload: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    issuedAt: datetime = Field(default_factory=utc_now, alias="issued_at")
    expiresAt: datetime = Field(..., alias="expires_at")
    idempotencyKey: str = Field(..., alias="idempotency_key")
    sessionId: str = Field(..., alias="session_id")
    traceId: str = Field(..., alias="trace_id")

    @field_validator("commandId", "robotId", "commandKey", "commandVersion", "idempotencyKey", "sessionId", "traceId")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("issuedAt", "expiresAt")
    @classmethod
    def _timestamps(cls, value):
        return as_utc(value)

    @field_validator("payload")
    @classmethod
    def _payload(cls, value):
        return freeze_json_object(value)

    @model_validator(mode="after")
    def _lifetime(self):
        issued, expires = validate_expiry(self.issuedAt, self.expiresAt)
        object.__setattr__(self, "issuedAt", issued)
        object.__setattr__(self, "expiresAt", expires)
        return self


class PlanStep(ContractModel):
    stepId: str = Field(..., alias="step_id")
    commandKey: str = Field(..., alias="command_key")
    payload: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    dependsOn: list[str] = Field(default_factory=list, alias="depends_on", validate_default=True)

    @field_validator("stepId", "commandKey")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("payload")
    @classmethod
    def _payload(cls, value):
        return freeze_json_object(value)

    @field_validator("dependsOn")
    @classmethod
    def _dependencies(cls, value):
        from .validation import FrozenList
        return FrozenList(non_empty_id(item, "dependsOn item") for item in value)


class PlanDraft(ContractModel):
    planId: str = Field(..., alias="plan_id")
    robotId: str = Field(..., alias="robot_id")
    steps: list[PlanStep] = Field(..., min_length=1, max_length=100)
    createdAt: datetime = Field(default_factory=utc_now, alias="created_at")
    expiresAt: datetime | None = Field(default=None, alias="expires_at")

    @field_validator("planId", "robotId")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("createdAt", "expiresAt")
    @classmethod
    def _timestamps(cls, value):
        return None if value is None else as_utc(value)

    @field_validator("steps")
    @classmethod
    def _freeze_steps(cls, value):
        from .validation import FrozenList
        return FrozenList(value)

    @model_validator(mode="after")
    def _dag_and_expiry(self):
        ensure_acyclic_steps(self.steps)
        if self.createdAt > utc_now() + CLOCK_SKEW:
            raise ValueError("createdAt cannot be in the future")
        if self.expiresAt is not None and self.expiresAt <= self.createdAt:
            raise ValueError("expiresAt must be later than createdAt")
        if self.expiresAt is not None and self.expiresAt <= utc_now():
            raise ValueError("expiresAt is in the past")
        return self


class CommandReceipt(ContractModel):
    commandId: str = Field(..., alias="command_id")
    status: CommandStatus
    robotId: str | None = Field(default=None, alias="robot_id")
    acceptedAt: datetime = Field(default_factory=utc_now, alias="accepted_at")
    startedAt: datetime | None = Field(default=None, alias="started_at")
    completedAt: datetime | None = Field(default=None, alias="completed_at")
    result: Any = Field(default=None, validate_default=True)
    error: str | None = None

    @field_validator("commandId")
    @classmethod
    def _command_id(cls, value):
        return non_empty_id(value, "commandId")

    @field_validator("robotId")
    @classmethod
    def _robot_id(cls, value):
        return None if value is None else non_empty_id(value, "robotId")

    @field_validator("acceptedAt", "startedAt", "completedAt")
    @classmethod
    def _timestamps(cls, value):
        return None if value is None else as_utc(value)

    @field_validator("result")
    @classmethod
    def _result(cls, value):
        return freeze_json_value(value)


class RobotState(ContractModel):
    robotId: str = Field(..., alias="robot_id")
    status: RobotStatus = RobotStatus.UNKNOWN
    connected: bool = True
    lastSeenAt: datetime | None = Field(default=None, alias="last_seen_at")
    safetyMode: SafetyMode = Field(default=SafetyMode.NORMAL, alias="safety_mode")
    details: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("robotId")
    @classmethod
    def _robot_id(cls, value):
        return non_empty_id(value, "robotId")

    @field_validator("lastSeenAt")
    @classmethod
    def _timestamp(cls, value):
        return None if value is None else as_utc(value)

    @field_validator("details")
    @classmethod
    def _details(cls, value):
        return freeze_json_object(value)


class CapabilitySnapshot(ContractModel):
    agent: str
    protocol: str
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema", serialization_alias="schema", validate_default=True)
    capabilities: list[str] = Field(default_factory=list, validate_default=True)
    safetyMode: SafetyMode = Field(default=SafetyMode.NORMAL, alias="safety_mode")
    robotId: str | None = Field(default=None, alias="robot_id")

    @field_validator("agent", "protocol")
    @classmethod
    def _names(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @property
    def schema(self) -> dict[str, Any]:
        """Public schema API without shadowing ``BaseModel.schema`` internally."""

        return self.schema_

    @field_validator("schema_")
    @classmethod
    def _schema(cls, value):
        return freeze_json_object(require_json_schema(value))

    @field_validator("capabilities")
    @classmethod
    def _caps(cls, value):
        from .validation import FrozenList
        return FrozenList(non_empty_id(item, "capability") for item in value)

    @field_validator("robotId")
    @classmethod
    def _robot_id(cls, value):
        return None if value is None else non_empty_id(value, "robotId")


class TargetProfile(ContractModel):
    targetId: str = Field(..., alias="target_id")
    transport: Transport
    endpoint: str | None = None
    timeoutMs: int = Field(
        default=5000,
        ge=0,
        alias="timeout_ms",
        validation_alias=AliasChoices("timeout_ms", "timeout"),
    )
    leaseSeconds: float = Field(
        default=30.0,
        ge=0,
        alias="lease_seconds",
        validation_alias=AliasChoices("lease_seconds", "lease"),
    )
    metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("targetId")
    @classmethod
    def _target_id(cls, value):
        return non_empty_id(value, "targetId")

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, value):
        return freeze_json_object(value)


class MappingVersion(ContractModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)

    mappingId: str = Field(..., alias="mapping_id")
    version: str
    commandKey: str = Field(..., alias="command_key")
    transport: Transport
    mapping: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    payloadSchema: dict[str, Any] = Field(
        default_factory=dict,
        alias="payload_schema",
        validate_default=True,
    )
    createdAt: datetime = Field(default_factory=utc_now, alias="created_at")

    @field_validator("mappingId", "version", "commandKey")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("mapping", "payloadSchema")
    @classmethod
    def _dicts(cls, value):
        return freeze_json_object(value)

    @field_validator("createdAt")
    @classmethod
    def _timestamp(cls, value):
        return as_utc(value)


# Friendly aliases used by adapters and callers that prefer explicit names.
CommandTransport = Transport
Source = CommandSource

__all__ = [
    "CapabilitySnapshot",
    "CommandDefinition",
    "CommandReceipt",
    "CommandRequest",
    "CommandKind",
    "CommandSource",
    "Source",
    "CommandStatus",
    "CommandTransport",
    "MappingVersion",
    "PlanDraft",
    "PlanStep",
    "RobotState",
    "RobotStatus",
    "SafetyMode",
    "TargetProfile",
    "Transport",
]
