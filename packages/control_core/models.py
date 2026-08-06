"""Transport-independent, validated domain contracts for robot actions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .validation import as_utc, ensure_acyclic_steps, non_empty_id, require_json_schema, validate_expiry, utc_now


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
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CommandDefinition(ContractModel):
    commandKey: str = Field(..., alias="command_key")
    commandVersion: str = Field(
        ..., alias="command_version", validation_alias=AliasChoices("command_version", "version")
    )
    payloadSchema: dict[str, Any] = Field(
        default_factory=dict,
        alias="payload_schema",
        validation_alias=AliasChoices("payload_schema", "schema"),
    )
    transports: list[Transport] = Field(default_factory=lambda: list(Transport))
    description: str | None = None

    @field_validator("commandKey", "commandVersion")
    @classmethod
    def _non_empty(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("payloadSchema")
    @classmethod
    def _schema(cls, value):
        return require_json_schema(value)

    @field_validator("transports")
    @classmethod
    def _transports(cls, value):
        if not value:
            raise ValueError("at least one transport is required")
        return list(dict.fromkeys(value))


class CommandRequest(ContractModel):
    commandId: str = Field(..., alias="command_id")
    robotId: str = Field(..., alias="robot_id")
    source: CommandSource
    transport: Transport
    commandKey: str = Field(..., alias="command_key")
    commandVersion: str = Field(..., alias="command_version")
    payload: dict[str, Any] = Field(default_factory=dict)
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

    @model_validator(mode="after")
    def _lifetime(self):
        issued, expires = validate_expiry(self.issuedAt, self.expiresAt)
        object.__setattr__(self, "issuedAt", issued)
        object.__setattr__(self, "expiresAt", expires)
        return self


class PlanStep(ContractModel):
    stepId: str = Field(..., alias="step_id")
    commandKey: str = Field(..., alias="command_key")
    payload: dict[str, Any] = Field(default_factory=dict)
    dependsOn: list[str] = Field(default_factory=list, alias="depends_on")

    @field_validator("stepId", "commandKey")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("dependsOn")
    @classmethod
    def _dependencies(cls, value):
        return [non_empty_id(item, "dependsOn item") for item in value]


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

    @model_validator(mode="after")
    def _dag_and_expiry(self):
        ensure_acyclic_steps(self.steps)
        if self.expiresAt is not None and self.expiresAt <= self.createdAt:
            raise ValueError("expiresAt must be later than createdAt")
        return self


class CommandReceipt(ContractModel):
    commandId: str = Field(..., alias="command_id")
    status: CommandStatus
    robotId: str | None = Field(default=None, alias="robot_id")
    acceptedAt: datetime = Field(default_factory=utc_now, alias="accepted_at")
    startedAt: datetime | None = Field(default=None, alias="started_at")
    completedAt: datetime | None = Field(default=None, alias="completed_at")
    result: Any = None
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


class RobotState(ContractModel):
    robotId: str = Field(..., alias="robot_id")
    status: RobotStatus = RobotStatus.UNKNOWN
    connected: bool = True
    lastSeenAt: datetime | None = Field(default=None, alias="last_seen_at")
    safetyMode: SafetyMode = Field(default=SafetyMode.NORMAL, alias="safety_mode")
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("robotId")
    @classmethod
    def _robot_id(cls, value):
        return non_empty_id(value, "robotId")

    @field_validator("lastSeenAt")
    @classmethod
    def _timestamp(cls, value):
        return None if value is None else as_utc(value)


class CapabilitySnapshot(ContractModel):
    agent: str
    protocol: str
    schema: dict[str, Any] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    safetyMode: SafetyMode = Field(default=SafetyMode.NORMAL, alias="safety_mode")
    robotId: str | None = Field(default=None, alias="robot_id")

    @field_validator("agent", "protocol")
    @classmethod
    def _names(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("schema")
    @classmethod
    def _schema(cls, value):
        return require_json_schema(value)

    @field_validator("capabilities")
    @classmethod
    def _caps(cls, value):
        return [non_empty_id(item, "capability") for item in value]

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
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("targetId")
    @classmethod
    def _target_id(cls, value):
        return non_empty_id(value, "targetId")


class MappingVersion(ContractModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)

    mappingId: str = Field(..., alias="mapping_id")
    version: str
    commandKey: str = Field(..., alias="command_key")
    transport: Transport
    mapping: dict[str, Any] = Field(default_factory=dict)
    payloadSchema: dict[str, Any] = Field(default_factory=dict, alias="payload_schema")
    createdAt: datetime = Field(default_factory=utc_now, alias="created_at")

    @field_validator("mappingId", "version", "commandKey")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("mapping", "payloadSchema")
    @classmethod
    def _dicts(cls, value):
        return require_json_schema(value)

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
