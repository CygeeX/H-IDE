from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, TypeAlias

from pydantic import Field, TypeAdapter, field_validator, model_validator

from packages.control_core.models import ContractModel, PlanDraft
from packages.control_core.validation import FrozenList, as_utc, non_empty_id, utc_now
from packages.scene_core import SceneRef


class VisualMissionMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    LAB_EXECUTE = "lab_execute"
    ROBOT_EXECUTE = "robot_execute"


class VisualMissionStageKind(str, Enum):
    LOCATE = "locate"
    APPROACH = "approach"
    VERIFY = "verify"
    COMPLETE = "complete"


class VisualMissionStage(ContractModel):
    stageId: str
    kind: VisualMissionStageKind
    targetDescription: str
    goalPurpose: Literal["observe", "approach", "final"] = "approach"

    @field_validator("stageId", "targetDescription")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)


class VisualMissionSpec(ContractModel):
    missionId: str
    robotId: str
    instruction: str
    locale: str = "zh-CN"
    targetDescription: str
    requestedSourceMode: Literal["live", "replay", "synthetic"]
    fusionMode: Literal["vision_only", "scene_only", "fused"] = "vision_only"
    stages: list[VisualMissionStage] = Field(min_length=1, max_length=6)
    robotBootId: str
    localizationEpochId: str
    sensorIds: list[str] = Field(min_length=1, max_length=4)
    allowedTargetCategories: list[str] = Field(min_length=1, max_length=20)
    unsupportedCapabilities: list[str] = Field(default_factory=list, max_length=20)
    sceneRef: SceneRef | None = None
    maximumGoalDistanceM: float = Field(default=5, gt=0, le=5)
    maximumCumulativeDistanceM: float = Field(default=20, gt=0, le=20)
    maximumDurationSeconds: int = Field(default=600, gt=0, le=600)
    vlmProvider: str
    vlmModel: str
    promptVersion: str
    visualAlgorithmVersion: str
    approachPolicyVersion: str
    approachPolicySha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximumSpeedMps: float = Field(default=0.2, gt=0, le=0.5)
    reviewExpiresAt: datetime

    @field_validator(
        "missionId",
        "robotId",
        "instruction",
        "locale",
        "targetDescription",
        "robotBootId",
        "localizationEpochId",
        "vlmProvider",
        "vlmModel",
        "promptVersion",
        "visualAlgorithmVersion",
        "approachPolicyVersion",
    )
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("stages", "sensorIds", "allowedTargetCategories", "unsupportedCapabilities")
    @classmethod
    def _lists(cls, value):
        return FrozenList(value)

    @field_validator("reviewExpiresAt")
    @classmethod
    def _time(cls, value: datetime):
        return as_utc(value)

    @model_validator(mode="after")
    def _mission_invariants(self):
        if self.reviewExpiresAt <= utc_now():
            raise ValueError("reviewExpiresAt is in the past")
        ids = [stage.stageId for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("visual mission stage IDs must be unique")
        if self.fusionMode in {"scene_only", "fused"} and self.sceneRef is None:
            raise ValueError("scene fusion requires an immutable sceneRef")
        return self


class VisualMissionPlanDraft(ContractModel):
    planType: Literal["visual_mission"] = "visual_mission"
    planId: str
    robotId: str
    mission: VisualMissionSpec
    createdAt: datetime = Field(default_factory=utc_now)
    expiresAt: datetime | None = None

    @field_validator("planId", "robotId")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("createdAt", "expiresAt")
    @classmethod
    def _times(cls, value):
        return None if value is None else as_utc(value)

    @model_validator(mode="after")
    def _same_robot_and_expiry(self):
        if self.robotId != self.mission.robotId:
            raise ValueError("mission.robotId does not match draft.robotId")
        if self.planId != self.mission.missionId:
            raise ValueError("missionId must equal planId")
        if self.expiresAt is not None and self.expiresAt <= self.createdAt:
            raise ValueError("expiresAt must be later than createdAt")
        if self.expiresAt is not None and self.expiresAt <= utc_now():
            raise ValueError("expiresAt is in the past")
        return self


PlanDraftUnion: TypeAlias = Annotated[
    PlanDraft | VisualMissionPlanDraft,
    Field(discriminator="planType"),
]
_PLAN_ADAPTER = TypeAdapter(PlanDraftUnion)


def parse_plan_draft(value: Mapping[str, Any] | PlanDraft | VisualMissionPlanDraft) -> PlanDraftUnion:
    if isinstance(value, (PlanDraft, VisualMissionPlanDraft)):
        return value
    payload = dict(value)
    if "planType" not in payload:
        payload["planType"] = "visual_mission" if "mission" in payload else "action"
    return _PLAN_ADAPTER.validate_python(payload)
