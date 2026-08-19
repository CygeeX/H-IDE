"""Canonical Robot Agent V2 profile, capability manifest, and live status.

The manifest describes locally configured executable capabilities.  It is not
the semantic catalog sent to Qwen: H-IDE projects these physical capabilities
into its own coordinate-free planner skills.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from robot_agent.config import AgentConfig, CommandConfig

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"


class CapabilityContractError(ValueError):
    """Raised when a V2 profile or capability projection cannot be built."""


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class SensorProfileWire(WireModel):
    sensorId: str
    kind: Literal["rgb", "depth", "rgbd", "lidar", "imu", "other"]
    frameId: str
    fixedMount: bool
    horizontalFovRad: float | None = None
    depthCapable: bool = False
    calibrationSha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class RobotProfileWire(WireModel):
    schemaVersion: Literal["h-ide.robot-profile/2"] = "h-ide.robot-profile/2"
    robotId: str
    baseType: str
    headMotion: Literal["fixed", "pan_tilt"]
    mapFrameId: str = "map"
    baseFrameId: str = "base_link"
    sensors: list[SensorProfileWire]
    metadata: dict[str, Any] = Field(default_factory=dict)
    profileDigest: str = Field(pattern=SHA256_PATTERN)

    def digest_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("profileDigest", None)
        return payload

    @model_validator(mode="after")
    def validate_digest(self, info: ValidationInfo) -> RobotProfileWire:
        if info.context and info.context.get("skip_digest_validation"):
            return self
        if self.profileDigest != canonical_digest(self.digest_payload()):
            raise ValueError("robot profile digest does not match canonical payload")
        return self

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> RobotProfileWire:
        candidate = dict(payload)
        candidate.setdefault("schemaVersion", "h-ide.robot-profile/2")
        candidate["profileDigest"] = "sha256:" + "0" * 64
        normalized = cls.model_validate(
            candidate,
            context={"skip_digest_validation": True},
        ).digest_payload()
        return cls.model_validate(
            {**normalized, "profileDigest": canonical_digest(normalized)}
        )


class CapabilityDescriptorWire(WireModel):
    skillId: str
    skillVersion: str
    description: str
    kind: Literal["motion", "observation", "manipulation", "safety", "non_motion"]
    riskLevel: Literal["low", "medium", "high", "critical"]
    inputSchema: dict[str, Any]
    resources: list[str]
    cancellable: bool
    interruptible: bool
    timeoutMs: int
    requiredSensorIds: list[str] = Field(default_factory=list)
    preconditions: list[dict[str, Any]] = Field(default_factory=list)
    postconditions: list[dict[str, Any]] = Field(default_factory=list)
    executor: Literal["nav2", "behavior_tree", "ros_action", "ros_service", "local"]
    executorProfile: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CapabilityManifestWire(WireModel):
    schemaVersion: Literal["h-ide.capability-manifest/2"] = (
        "h-ide.capability-manifest/2"
    )
    robotId: str
    protocolVersion: Literal["2.0"] = "2.0"
    agentVersion: str
    robotProfileDigest: str = Field(pattern=SHA256_PATTERN)
    generatedAt: datetime
    capabilities: list[CapabilityDescriptorWire]
    manifestDigest: str = Field(pattern=SHA256_PATTERN)

    def digest_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("manifestDigest", None)
        return payload

    @model_validator(mode="after")
    def validate_digest(self, info: ValidationInfo) -> CapabilityManifestWire:
        if info.context and info.context.get("skip_digest_validation"):
            return self
        if self.manifestDigest != canonical_digest(self.digest_payload()):
            raise ValueError("capability manifest digest does not match canonical payload")
        return self

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> CapabilityManifestWire:
        candidate = dict(payload)
        candidate.setdefault("schemaVersion", "h-ide.capability-manifest/2")
        candidate.setdefault("protocolVersion", "2.0")
        candidate.setdefault("generatedAt", _utc_now())
        candidate["manifestDigest"] = "sha256:" + "0" * 64
        normalized = cls.model_validate(
            candidate,
            context={"skip_digest_validation": True},
        ).digest_payload()
        return cls.model_validate(
            {**normalized, "manifestDigest": canonical_digest(normalized)}
        )


class CapabilityStatusEntryWire(WireModel):
    skillId: str
    skillVersion: str
    health: Literal["ready", "degraded", "unavailable", "unknown"]
    available: bool
    reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> CapabilityStatusEntryWire:
        if self.available and self.health in {"unavailable", "unknown"}:
            raise ValueError("available capability cannot have unavailable or unknown health")
        return self


class CapabilityStatusWire(WireModel):
    schemaVersion: Literal["h-ide.capability-status/2"] = "h-ide.capability-status/2"
    robotId: str
    robotBootId: str
    manifestDigest: str = Field(pattern=SHA256_PATTERN)
    sequence: int = Field(ge=0)
    observedAt: datetime
    entries: list[CapabilityStatusEntryWire]


def _executor(command: CommandConfig) -> Literal[
    "nav2", "behavior_tree", "ros_action", "ros_service", "local"
]:
    if command.goal_transform in {"navigate_to_pose_v1", "nav2_spin_v2"}:
        return "nav2"
    return "ros_action" if command.adapter == "action" else "ros_service"


def _executor_profile(command: CommandConfig) -> str:
    if command.executor_profile is not None:
        return command.executor_profile
    return f"{command.adapter}:{command.ros_type}:{command.ros_name}"


def build_robot_profile(config: AgentConfig) -> RobotProfileWire:
    profile = config.robot_profile
    if profile is None:
        raise CapabilityContractError("robot_profile is required for Mission V2")
    return RobotProfileWire.build(
        {
            "robotId": config.robot_id,
            "baseType": profile.base_type,
            "headMotion": profile.head_motion,
            "mapFrameId": profile.map_frame_id,
            "baseFrameId": profile.base_frame_id,
            "sensors": [
                {
                    "sensorId": sensor.sensor_id,
                    "kind": sensor.kind,
                    "frameId": sensor.frame_id,
                    "fixedMount": sensor.fixed_mount,
                    "horizontalFovRad": sensor.horizontal_fov_rad,
                    "depthCapable": sensor.depth_capable,
                    "calibrationSha256": sensor.calibration_sha256,
                }
                for sensor in profile.sensors
            ],
            "metadata": profile.metadata,
        }
    )


def _descriptor(skill_id: str, command: CommandConfig) -> CapabilityDescriptorWire:
    action = command.adapter == "action"
    return CapabilityDescriptorWire(
        skillId=skill_id,
        skillVersion=command.skill_version,
        description=command.description or f"Locally configured {skill_id} capability",
        kind=command.capability_kind,
        riskLevel=command.risk_level,
        inputSchema=command.request_schema,
        resources=list(command.resources),
        cancellable=action if command.cancellable is None else command.cancellable,
        interruptible=action if command.interruptible is None else command.interruptible,
        timeoutMs=command.timeout_ms,
        requiredSensorIds=list(command.observation_sensor_ids),
        executor=_executor(command),
        executorProfile=_executor_profile(command),
        metadata={"requiresObservation": command.requires_observation},
    )


def build_capability_manifest(
    config: AgentConfig,
    *,
    generated_at: datetime | None = None,
) -> tuple[RobotProfileWire, CapabilityManifestWire]:
    profile = build_robot_profile(config)
    manifest = CapabilityManifestWire.build(
        {
            "robotId": config.robot_id,
            "agentVersion": config.agent_version,
            "robotProfileDigest": profile.profileDigest,
            "generatedAt": generated_at or _utc_now(),
            "capabilities": [
                _descriptor(skill_id, command)
                for skill_id, command in sorted(config.commands.items())
            ],
        }
    )
    return profile, manifest


def build_capability_status(
    *,
    config: AgentConfig,
    manifest: CapabilityManifestWire,
    robot_boot_id: str,
    sequence: int,
    unavailable: Mapping[str, str] | None = None,
    observed_at: datetime | None = None,
) -> CapabilityStatusWire:
    unavailable = unavailable or {}
    return CapabilityStatusWire(
        robotId=config.robot_id,
        robotBootId=robot_boot_id,
        manifestDigest=manifest.manifestDigest,
        sequence=sequence,
        observedAt=observed_at or _utc_now(),
        entries=[
            CapabilityStatusEntryWire(
                skillId=descriptor.skillId,
                skillVersion=descriptor.skillVersion,
                health=("unavailable" if descriptor.skillId in unavailable else "ready"),
                available=descriptor.skillId not in unavailable,
                reason=unavailable.get(descriptor.skillId),
            )
            for descriptor in manifest.capabilities
        ],
    )


__all__ = [
    "CapabilityContractError",
    "CapabilityDescriptorWire",
    "CapabilityManifestWire",
    "CapabilityStatusEntryWire",
    "CapabilityStatusWire",
    "RobotProfileWire",
    "SensorProfileWire",
    "build_capability_manifest",
    "build_capability_status",
    "build_robot_profile",
    "canonical_digest",
]
