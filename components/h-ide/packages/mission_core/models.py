"""Immutable Mission V2 contracts shared by the control plane and PWA.

These models intentionally contain semantic skills and authorization bounds,
not ROS endpoints, paths, velocities, or model-authored map coordinates.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from packages.control_core.capabilities import RiskLevel, SHA256_PATTERN
from packages.control_core.models import ContractModel
from packages.control_core.validation import (
    CLOCK_SKEW,
    FrozenList,
    as_utc,
    freeze_json_object,
    freeze_json_value,
    non_empty_id,
    utc_now,
)
from packages.scene_core import Point2D, SceneRef


class MissionStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    READY = "ready"
    PLANNING = "planning"
    OBSERVING = "observing"
    DISPATCHING = "dispatching"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    NEEDS_INPUT = "needs_input"
    AWAITING_REVIEW = "awaiting_review"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ApprovalDisposition(str, Enum):
    HUMAN_APPROVED = "human_approved"
    POLICY_INHERITED = "policy_inherited"
    HUMAN_REQUIRED = "human_required"


class MissionInterventionKind(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    MODIFY_GOAL = "modify_goal"
    APPEND_REQUIREMENT = "append_requirement"
    ANSWER = "answer"


class SkillResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


class WorldRecordKind(str, Enum):
    HYPOTHESIS = "hypothesis"
    FACT = "fact"


class ModelPin(ContractModel):
    role: Literal["text_planner", "visual_grounder", "segmentation", "geometry"]
    provider: str
    model: str
    promptVersion: str | None = None

    @field_validator("provider", "model")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("promptVersion")
    @classmethod
    def _optional_string(cls, value: str | None):
        return None if value is None else non_empty_id(value, "promptVersion")


_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}

_EXECUTABLE_INTENT_KEYS = frozenset(
    {
        "angularvelocity",
        "cmdvel",
        "executor",
        "frameid",
        "goalpose",
        "linearvelocity",
        "path",
        "pose",
        "position",
        "quaternion",
        "rosaction",
        "rosservice",
        "waypoints",
        "x",
        "y",
        "yaw",
        "z",
    }
)


def _reject_executable_intent(value: Any, *, path: str = "intent") -> None:
    """Keep planner output semantic and outside the physical command boundary."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = "".join(character for character in key.casefold() if character.isalnum())
            if normalized in _EXECUTABLE_INTENT_KEYS:
                raise ValueError(f"{path}.{key} is executable geometry, not semantic intent")
            _reject_executable_intent(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_executable_intent(item, path=f"{path}[{index}]")


class Geofence(ContractModel):
    geofenceId: str
    frameId: Literal["map"] = "map"
    polygonXY: list[Point2D] = Field(min_length=3, max_length=1_000)

    @field_validator("geofenceId")
    @classmethod
    def _id(cls, value: str):
        return non_empty_id(value, "geofenceId")

    @field_validator("polygonXY")
    @classmethod
    def _polygon(cls, value):
        if value[0] == value[-1]:
            value = value[:-1]
        if len(value) < 3:
            raise ValueError("geofence polygon requires three distinct vertices")
        return FrozenList(value)


class MissionEnvelope(ContractModel):
    schemaVersion: Literal["h-ide.mission-envelope/2"] = "h-ide.mission-envelope/2"
    robotId: str
    robotBootId: str
    localizationEpochId: str
    capabilityManifestDigest: str = Field(pattern=SHA256_PATTERN)
    robotProfileDigest: str = Field(pattern=SHA256_PATTERN)
    workstationCapabilityDigest: str = Field(pattern=SHA256_PATTERN)
    allowedSkillIds: list[str] = Field(min_length=1, max_length=100)
    maximumRiskLevel: RiskLevel = RiskLevel.MEDIUM
    allowedTargetCategories: list[str] = Field(default_factory=list, max_length=100)
    allowedRegionIds: list[str] = Field(default_factory=list, max_length=500)
    sceneRefs: list[SceneRef] = Field(default_factory=list, max_length=8)
    geofences: list[Geofence] = Field(default_factory=list, max_length=16)
    maximumGoalDistanceM: float = Field(default=5.0, gt=0, le=25)
    maximumCumulativeDistanceM: float = Field(default=20.0, gt=0, le=200)
    maximumSpeedMps: float = Field(default=0.2, gt=0, le=1.0)
    maximumDurationSeconds: int = Field(default=600, gt=0, le=7_200)
    maximumAutoRevisions: int = Field(default=3, ge=0, le=20)
    maximumStageAttempts: int = Field(default=3, ge=1, le=10)
    # This is a cumulative Mission budget. A multi-room inspection may perform
    # many individually bounded scans while every single physical action
    # remains constrained by the role/grant limits.
    maximumScanYawRad: float = Field(default=math.tau, gt=0, le=40 * math.pi)
    modelPins: list[ModelPin] = Field(min_length=2, max_length=16)
    expiresAt: datetime

    @field_validator("robotId", "robotBootId", "localizationEpochId")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("allowedSkillIds", "allowedTargetCategories", "allowedRegionIds")
    @classmethod
    def _unique_strings(cls, value, info):
        normalized = [non_empty_id(item, info.field_name) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must be unique")
        return FrozenList(normalized)

    @field_validator("sceneRefs")
    @classmethod
    def _scenes(cls, value):
        identities = [(item.sceneId, str(item.revision)) for item in value]
        if len(identities) != len(set(identities)):
            raise ValueError("sceneRefs must be unique")
        return FrozenList(value)

    @field_validator("geofences", "modelPins")
    @classmethod
    def _unique_objects(cls, value, info):
        if info.field_name == "geofences":
            identifiers = [item.geofenceId for item in value]
        else:
            identifiers = [item.role for item in value]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{info.field_name} must be unique")
        if info.field_name == "modelPins":
            required = {"text_planner", "visual_grounder"}
            missing = required - set(identifiers)
            if missing:
                raise ValueError(f"modelPins missing required role: {sorted(missing)[0]}")
        return FrozenList(value)

    @field_validator("expiresAt")
    @classmethod
    def _expiry(cls, value: datetime):
        # Expiry is an execution policy, not a serialization invariant.
        # Historical Mission records must remain parseable for audit,
        # reconciliation and cancellation after their authorization window.
        return as_utc(value)


class MissionNode(ContractModel):
    nodeId: str
    skillId: str
    skillVersion: str
    purpose: str
    intent: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    dependsOn: list[str] = Field(default_factory=list, max_length=32)
    preconditionIds: list[str] = Field(default_factory=list, max_length=32)
    postconditionIds: list[str] = Field(default_factory=list, max_length=32)
    riskLevel: RiskLevel

    @field_validator("nodeId", "skillId", "skillVersion", "purpose")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("intent")
    @classmethod
    def _intent(cls, value):
        _reject_executable_intent(value)
        return freeze_json_object(value)

    @field_validator("dependsOn", "preconditionIds", "postconditionIds")
    @classmethod
    def _lists(cls, value, info):
        normalized = [non_empty_id(item, info.field_name) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must be unique")
        return FrozenList(normalized)


def _validate_node_dag(nodes: list[MissionNode]) -> None:
    identifiers = [node.nodeId for node in nodes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("mission node IDs must be unique")
    known = set(identifiers)
    graph = {node.nodeId: set(node.dependsOn) for node in nodes}
    for node_id, dependencies in graph.items():
        if node_id in dependencies:
            raise ValueError("mission dependency cycle detected")
        unknown = dependencies - known
        if unknown:
            raise ValueError(f"unknown mission dependency: {sorted(unknown)[0]}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("mission dependency cycle detected")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in graph[node_id]:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for identifier in identifiers:
        visit(identifier)


class MissionPlanDraft(ContractModel):
    planType: Literal["mission"] = "mission"
    schemaVersion: Literal["h-ide.mission/2"] = "h-ide.mission/2"
    planId: str
    robotId: str
    objective: str
    nodes: list[MissionNode] = Field(min_length=1, max_length=100)
    envelope: MissionEnvelope
    createdAt: datetime = Field(default_factory=utc_now)
    expiresAt: datetime

    @field_validator("planId", "robotId", "objective")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("nodes")
    @classmethod
    def _nodes(cls, value):
        _validate_node_dag(value)
        return FrozenList(value)

    @field_validator("createdAt", "expiresAt")
    @classmethod
    def _times(cls, value: datetime):
        return as_utc(value)

    @model_validator(mode="after")
    def _invariants(self):
        if self.robotId != self.envelope.robotId:
            raise ValueError("mission envelope robotId mismatch")
        if self.createdAt > utc_now() + CLOCK_SKEW:
            raise ValueError("createdAt cannot be in the future")
        if self.expiresAt <= self.createdAt:
            raise ValueError("mission draft has an invalid lifetime")
        if self.expiresAt > self.envelope.expiresAt:
            raise ValueError("mission draft cannot outlive its envelope")
        allowed = set(self.envelope.allowedSkillIds)
        unknown = [node.skillId for node in self.nodes if node.skillId not in allowed]
        if unknown:
            raise ValueError(f"mission uses skill outside envelope: {unknown[0]}")
        excessive_risk = [
            node.nodeId
            for node in self.nodes
            if _RISK_ORDER[node.riskLevel] > _RISK_ORDER[self.envelope.maximumRiskLevel]
        ]
        if excessive_risk:
            raise ValueError(f"mission node exceeds envelope risk: {excessive_risk[0]}")
        return self


class MissionAuthorization(ContractModel):
    authorizationId: str
    missionId: str
    planId: str
    envelopeSha256: str = Field(pattern=SHA256_PATTERN)
    disposition: Literal["human_approved"] = "human_approved"
    approvedBy: str
    approvedAt: datetime = Field(default_factory=utc_now)
    expiresAt: datetime

    @field_validator("authorizationId", "missionId", "planId", "approvedBy")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("approvedAt", "expiresAt")
    @classmethod
    def _times(cls, value: datetime):
        return as_utc(value)

    @model_validator(mode="after")
    def _lifetime(self):
        if self.expiresAt <= self.approvedAt:
            raise ValueError("mission authorization has an invalid lifetime")
        return self


class PlanRevision(ContractModel):
    schemaVersion: Literal["h-ide.plan-revision/2"] = "h-ide.plan-revision/2"
    revisionId: str
    missionId: str
    sequence: int = Field(ge=1)
    parentRevisionId: str | None = None
    objective: str
    nodes: list[MissionNode] = Field(min_length=1, max_length=100)
    reasonCode: str
    reason: str
    changeSummary: list[str] = Field(min_length=1, max_length=50)
    disposition: ApprovalDisposition
    authorizationId: str | None = None
    createdBy: str
    createdAt: datetime = Field(default_factory=utc_now)

    @field_validator(
        "revisionId",
        "missionId",
        "objective",
        "reasonCode",
        "reason",
        "createdBy",
    )
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("parentRevisionId", "authorizationId")
    @classmethod
    def _optional_strings(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("nodes")
    @classmethod
    def _nodes(cls, value):
        _validate_node_dag(value)
        return FrozenList(value)

    @field_validator("changeSummary")
    @classmethod
    def _changes(cls, value):
        return FrozenList(non_empty_id(item, "changeSummary") for item in value)

    @field_validator("createdAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)

    @model_validator(mode="after")
    def _authorization(self):
        if self.disposition is ApprovalDisposition.HUMAN_REQUIRED and self.authorizationId:
            raise ValueError("human-required revision cannot already have an authorization")
        if self.disposition is not ApprovalDisposition.HUMAN_REQUIRED and not self.authorizationId:
            raise ValueError("approved revision must reference an authorization")
        return self


class MissionIntervention(ContractModel):
    interventionId: str
    missionId: str
    kind: MissionInterventionKind
    text: str | None = None
    requestedBy: str
    requestedAt: datetime = Field(default_factory=utc_now)

    @field_validator("interventionId", "missionId", "requestedBy")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("text")
    @classmethod
    def _text(cls, value: str | None):
        return None if value is None else non_empty_id(value, "text")

    @field_validator("requestedAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)


class MissionEvent(ContractModel):
    schemaVersion: Literal["h-ide.mission-event/2"] = "h-ide.mission-event/2"
    eventId: str
    missionId: str
    sequence: int = Field(ge=1)
    eventType: str
    occurredAt: datetime = Field(default_factory=utc_now)
    causationId: str | None = None
    correlationId: str
    revisionId: str | None = None
    nodeId: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("eventId", "missionId", "eventType", "correlationId")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("causationId", "revisionId", "nodeId")
    @classmethod
    def _optional_strings(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("occurredAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)

    @field_validator("payload")
    @classmethod
    def _payload(cls, value):
        return freeze_json_object(value)


class WorldRecord(ContractModel):
    schemaVersion: Literal["h-ide.world-record/2"] = "h-ide.world-record/2"
    recordId: str
    missionId: str
    robotId: str
    kind: WorldRecordKind
    subject: str
    predicate: str
    value: Any = Field(validate_default=True)
    source: Literal[
        "qwen",
        "geometry",
        "robot_agent",
        "scene",
        "operator",
        "verifier",
        "daaam",
    ]
    sourceRef: str
    confidence: float = Field(ge=0, le=1)
    observedAt: datetime
    validUntil: datetime | None = None
    frameId: str | None = None
    sceneRef: SceneRef | None = None
    evidenceIds: list[str] = Field(default_factory=list, max_length=50)
    supersedesRecordId: str | None = None

    @field_validator(
        "recordId", "missionId", "robotId", "subject", "predicate", "sourceRef"
    )
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("frameId", "supersedesRecordId")
    @classmethod
    def _optional_strings(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("value")
    @classmethod
    def _value(cls, value):
        return freeze_json_value(value)

    @field_validator("observedAt", "validUntil")
    @classmethod
    def _times(cls, value):
        return None if value is None else as_utc(value)

    @field_validator("evidenceIds")
    @classmethod
    def _evidence(cls, value):
        normalized = [non_empty_id(item, "evidenceId") for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("evidenceIds must be unique")
        return FrozenList(normalized)

    @model_validator(mode="after")
    def _validity(self):
        if self.validUntil is not None and self.validUntil <= self.observedAt:
            raise ValueError("validUntil must be later than observedAt")
        if self.kind is WorldRecordKind.FACT and self.source in {"qwen", "daaam"}:
            raise ValueError(
                "semantic observation output cannot be promoted directly to a fact"
            )
        return self


class SkillRequest(ContractModel):
    schemaVersion: Literal["h-ide.skill-request/2"] = "h-ide.skill-request/2"
    requestId: str
    missionId: str
    revisionId: str
    nodeId: str
    robotId: str
    skillId: str
    skillVersion: str
    input: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    capabilityManifestDigest: str = Field(pattern=SHA256_PATTERN)
    authorizationId: str
    idempotencyKey: str
    issuedAt: datetime = Field(default_factory=utc_now)
    expiresAt: datetime

    @field_validator(
        "requestId", "missionId", "revisionId", "nodeId", "robotId", "skillId",
        "skillVersion", "authorizationId", "idempotencyKey"
    )
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("input")
    @classmethod
    def _input(cls, value):
        return freeze_json_object(value)

    @field_validator("issuedAt", "expiresAt")
    @classmethod
    def _times(cls, value: datetime):
        return as_utc(value)

    @model_validator(mode="after")
    def _lifetime(self):
        if self.expiresAt <= self.issuedAt:
            raise ValueError("skill request has an invalid lifetime")
        return self


class SkillProgress(ContractModel):
    requestId: str
    sequence: int = Field(ge=0)
    progress: float | None = Field(default=None, ge=0, le=1)
    phase: str
    observedAt: datetime
    detail: str | None = None

    @field_validator("requestId", "phase")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("observedAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)


class SkillResult(ContractModel):
    requestId: str
    status: SkillResultStatus
    completedAt: datetime
    result: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    evidenceIds: list[str] = Field(default_factory=list, max_length=50)
    errorCode: str | None = None
    detail: str | None = None

    @field_validator("requestId")
    @classmethod
    def _id(cls, value: str):
        return non_empty_id(value, "requestId")

    @field_validator("completedAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)

    @field_validator("result")
    @classmethod
    def _result(cls, value):
        return freeze_json_object(value)

    @field_validator("evidenceIds")
    @classmethod
    def _evidence(cls, value):
        return FrozenList(non_empty_id(item, "evidenceId") for item in value)


class MissionSnapshot(ContractModel):
    missionId: str
    robotId: str
    status: MissionStatus
    lastSequence: int = Field(ge=0)
    activeRevisionId: str | None = None
    activeNodeId: str | None = None
    activeSkillRequestId: str | None = None
    autoRevisionCount: int = Field(default=0, ge=0)
    cumulativeDistanceM: float = Field(default=0, ge=0)
    cumulativeScanYawRad: float = Field(default=0, ge=0)
    updatedAt: datetime
    errorCode: str | None = None
    detail: str | None = None

    @field_validator("missionId", "robotId")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("activeRevisionId", "activeNodeId", "activeSkillRequestId")
    @classmethod
    def _optional_strings(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("updatedAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)
