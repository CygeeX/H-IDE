"""Versioned robot embodiment and skill capability contracts.

The manifest is deliberately split from live capability status.  A mission
pins the immutable manifest/profile digests, while availability and health may
change as fresh Robot Agent observations arrive.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
import json
import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from .models import ContractModel
from .validation import (
    FrozenList,
    as_utc,
    freeze_json_object,
    non_empty_id,
    require_json_schema,
    utc_now,
)


SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class CapabilityKind(str, Enum):
    MOTION = "motion"
    OBSERVATION = "observation"
    MANIPULATION = "manipulation"
    SAFETY = "safety"
    NON_MOTION = "non_motion"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CapabilityHealth(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class SensorKind(str, Enum):
    RGB = "rgb"
    DEPTH = "depth"
    RGBD = "rgbd"
    LIDAR = "lidar"
    IMU = "imu"
    OTHER = "other"


class ConditionSpec(ContractModel):
    conditionId: str
    description: str
    evaluator: str
    parameters: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    requiredEvidenceKinds: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("conditionId", "description", "evaluator")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("parameters")
    @classmethod
    def _parameters(cls, value):
        return freeze_json_object(value)

    @field_validator("requiredEvidenceKinds")
    @classmethod
    def _evidence(cls, value):
        return FrozenList(non_empty_id(item, "requiredEvidenceKind") for item in value)


class CapabilityDescriptor(ContractModel):
    skillId: str
    skillVersion: str
    description: str
    kind: CapabilityKind
    riskLevel: RiskLevel
    inputSchema: dict[str, Any]
    resources: list[str] = Field(min_length=1, max_length=32)
    cancellable: bool
    interruptible: bool
    timeoutMs: int = Field(gt=0, le=600_000)
    requiredSensorIds: list[str] = Field(default_factory=list, max_length=16)
    preconditions: list[ConditionSpec] = Field(default_factory=list, max_length=32)
    postconditions: list[ConditionSpec] = Field(default_factory=list, max_length=32)
    executor: Literal["nav2", "behavior_tree", "ros_action", "ros_service", "local"]
    executorProfile: str
    metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("skillId", "skillVersion", "description", "executorProfile")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("inputSchema")
    @classmethod
    def _schema(cls, value):
        schema = require_json_schema(value)
        if schema.get("type") != "object":
            raise ValueError("capability inputSchema root type must be object")
        if schema.get("additionalProperties") is not False:
            raise ValueError("capability inputSchema must reject additional properties")
        return freeze_json_object(schema)

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, value):
        return freeze_json_object(value)

    @field_validator("resources", "requiredSensorIds")
    @classmethod
    def _unique_ids(cls, value, info):
        normalized = [non_empty_id(item, info.field_name) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must be unique")
        return FrozenList(normalized)

    @field_validator("preconditions", "postconditions")
    @classmethod
    def _conditions(cls, value, info):
        identifiers = [item.conditionId for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{info.field_name} condition IDs must be unique")
        return FrozenList(value)


class SensorProfile(ContractModel):
    sensorId: str
    kind: SensorKind
    frameId: str
    fixedMount: bool
    horizontalFovRad: float | None = Field(default=None, gt=0, le=math.tau)
    depthCapable: bool = False
    calibrationSha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("sensorId", "frameId")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)


def _canonical_value(value: Any) -> Any:
    """Normalize nested contract values before hashing constructor input."""

    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        return as_utc(value).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class RobotProfile(ContractModel):
    schemaVersion: Literal["h-ide.robot-profile/2"] = "h-ide.robot-profile/2"
    robotId: str
    baseType: str
    headMotion: Literal["fixed", "pan_tilt"]
    mapFrameId: str = "map"
    baseFrameId: str = "base_link"
    sensors: list[SensorProfile] = Field(default_factory=list, max_length=32)
    metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    profileDigest: str = Field(pattern=SHA256_PATTERN)

    @field_validator("robotId", "baseType", "mapFrameId", "baseFrameId")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("sensors")
    @classmethod
    def _sensors(cls, value):
        identifiers = [item.sensorId for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("sensor IDs must be unique")
        return FrozenList(value)

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, value):
        return freeze_json_object(value)

    def digest_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("profileDigest", None)
        return payload

    @model_validator(mode="after")
    def _valid_digest(self, info: ValidationInfo):
        if info.context and info.context.get("skip_digest_validation"):
            return self
        if self.profileDigest != _canonical_digest(self.digest_payload()):
            raise ValueError("robot profile digest does not match its canonical payload")
        return self

    @classmethod
    def build(cls, **values: Any) -> RobotProfile:
        payload = dict(values)
        payload.setdefault("schemaVersion", "h-ide.robot-profile/2")
        payload["profileDigest"] = "sha256:" + "0" * 64
        normalized = cls.model_validate(
            payload,
            context={"skip_digest_validation": True},
        ).digest_payload()
        payload = normalized
        payload["profileDigest"] = _canonical_digest(normalized)
        return cls.model_validate(payload)


class CapabilityManifest(ContractModel):
    schemaVersion: Literal["h-ide.capability-manifest/2"] = "h-ide.capability-manifest/2"
    robotId: str
    protocolVersion: Literal["2.0"] = "2.0"
    agentVersion: str
    robotProfileDigest: str = Field(pattern=SHA256_PATTERN)
    generatedAt: datetime = Field(default_factory=utc_now)
    capabilities: list[CapabilityDescriptor] = Field(min_length=1, max_length=256)
    manifestDigest: str = Field(pattern=SHA256_PATTERN)

    @field_validator("robotId", "agentVersion")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("generatedAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)

    @field_validator("capabilities")
    @classmethod
    def _capabilities(cls, value):
        identifiers = [(item.skillId, item.skillVersion) for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("capability skill ID/version pairs must be unique")
        return FrozenList(value)

    def digest_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("manifestDigest", None)
        return payload

    @model_validator(mode="after")
    def _valid_digest(self, info: ValidationInfo):
        if info.context and info.context.get("skip_digest_validation"):
            return self
        if self.manifestDigest != _canonical_digest(self.digest_payload()):
            raise ValueError("capability manifest digest does not match its canonical payload")
        return self

    @classmethod
    def build(cls, **values: Any) -> CapabilityManifest:
        payload = dict(values)
        payload.setdefault("schemaVersion", "h-ide.capability-manifest/2")
        payload.setdefault("protocolVersion", "2.0")
        payload.setdefault("generatedAt", utc_now())
        payload["manifestDigest"] = "sha256:" + "0" * 64
        normalized = cls.model_validate(
            payload,
            context={"skip_digest_validation": True},
        ).digest_payload()
        payload = normalized
        payload["manifestDigest"] = _canonical_digest(normalized)
        return cls.model_validate(payload)


class CapabilityStatusEntry(ContractModel):
    skillId: str
    skillVersion: str
    health: CapabilityHealth
    available: bool
    reason: str | None = None

    @field_validator("skillId", "skillVersion")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @model_validator(mode="after")
    def _availability_matches_health(self):
        if self.available and self.health in {CapabilityHealth.UNAVAILABLE, CapabilityHealth.UNKNOWN}:
            raise ValueError("available capability cannot have unavailable or unknown health")
        return self


class CapabilityStatus(ContractModel):
    schemaVersion: Literal["h-ide.capability-status/2"] = "h-ide.capability-status/2"
    robotId: str
    robotBootId: str
    manifestDigest: str = Field(pattern=SHA256_PATTERN)
    sequence: int = Field(ge=0)
    observedAt: datetime
    entries: list[CapabilityStatusEntry] = Field(min_length=1, max_length=256)

    @field_validator("robotId", "robotBootId")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("observedAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)

    @field_validator("entries")
    @classmethod
    def _entries(cls, value):
        identifiers = [(item.skillId, item.skillVersion) for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("capability status entries must be unique")
        return FrozenList(value)


__all__ = [
    "CapabilityDescriptor",
    "CapabilityHealth",
    "CapabilityKind",
    "CapabilityManifest",
    "CapabilityStatus",
    "CapabilityStatusEntry",
    "ConditionSpec",
    "RiskLevel",
    "RobotProfile",
    "SensorKind",
    "SensorProfile",
]
