"""Strict contracts shared by H-IDE and the planning-only Agent service.

These models intentionally contain logical commands and robot context only.
Transport mappings, ROS names, adapter endpoints, and credentials never cross
the planning boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.scene_core import SceneContext
from packages.visual_mission_core.models import VisualMissionSpec

from .models import CapabilitySnapshot, CommandKind, RobotState
from .validation import (
    CLOCK_SKEW,
    FrozenList,
    as_utc,
    ensure_acyclic_steps,
    freeze_json_object,
    non_empty_id,
    require_json_schema,
    utc_now,
)


class AssistantContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TaskModality(str, Enum):
    TEXT = "text"
    VOICE = "voice"


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TaskRequest(AssistantContract):
    schemaVersion: Literal["1.0"] = "1.0"
    taskId: str
    conversationId: str
    robotId: str
    requestedBy: str
    modality: TaskModality
    text: str = Field(min_length=1, max_length=4000)
    locale: str = Field(default="zh-CN", min_length=2, max_length=35)
    transcriptionId: str | None = None
    revisionOfDraftId: str | None = None
    sceneContext: SceneContext | None = None
    createdAt: datetime = Field(default_factory=utc_now)

    @field_validator(
        "taskId",
        "conversationId",
        "robotId",
        "requestedBy",
        "text",
        "locale",
        "transcriptionId",
        "revisionOfDraftId",
    )
    @classmethod
    def _non_empty(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("createdAt")
    @classmethod
    def _created_at(cls, value: datetime) -> datetime:
        value = as_utc(value)
        if value > utc_now() + CLOCK_SKEW:
            raise ValueError("createdAt cannot be in the future")
        return value


class ConversationMessage(AssistantContract):
    role: ConversationRole
    content: str = Field(min_length=1, max_length=4000)
    createdAt: datetime

    @field_validator("content")
    @classmethod
    def _content(cls, value: str) -> str:
        return non_empty_id(value, "content")

    @field_validator("createdAt")
    @classmethod
    def _created_at(cls, value: datetime) -> datetime:
        return as_utc(value)


class LogicalCommand(AssistantContract):
    commandKey: str
    version: str
    kind: CommandKind
    description: str | None = None
    payloadSchema: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("commandKey", "version")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("description")
    @classmethod
    def _description(cls, value: str | None):
        return None if value is None else non_empty_id(value, "description")

    @field_validator("payloadSchema")
    @classmethod
    def _schema(cls, value):
        return freeze_json_object(require_json_schema(value))


class PlanningEnvelope(AssistantContract):
    taskRequest: TaskRequest
    contextVersion: str
    history: list[ConversationMessage] = Field(default_factory=list, max_length=20)
    robotState: RobotState
    capabilitySnapshot: CapabilitySnapshot
    commandCatalog: list[LogicalCommand] = Field(min_length=1, max_length=100)
    sceneContext: SceneContext | None = None

    @field_validator("contextVersion")
    @classmethod
    def _context_version(cls, value):
        return non_empty_id(value, "contextVersion")

    @field_validator("history", "commandCatalog")
    @classmethod
    def _frozen_lists(cls, value):
        return FrozenList(value)

    @model_validator(mode="after")
    def _same_robot(self):
        robot_id = self.taskRequest.robotId
        if self.robotState.robotId != robot_id:
            raise ValueError("robotState does not match taskRequest.robotId")
        if self.capabilitySnapshot.robotId not in {None, robot_id}:
            raise ValueError("capabilitySnapshot does not match taskRequest.robotId")
        if self.sceneContext != self.taskRequest.sceneContext:
            raise ValueError("sceneContext does not match taskRequest.sceneContext")
        if self.sceneContext is not None and self.sceneContext.currentPose != self.robotState.pose:
            raise ValueError("sceneContext.currentPose does not match robotState.pose")
        return self


class FragmentStep(AssistantContract):
    stepId: str
    commandKey: str
    payload: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    dependsOn: list[str] = Field(default_factory=list, validate_default=True)
    rationale: str = Field(min_length=1, max_length=1000)

    @field_validator("stepId", "commandKey", "rationale")
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
        return FrozenList(non_empty_id(item, "dependsOn item") for item in value)


class PlanFragment(AssistantContract):
    agentId: str
    contribution: str
    summary: str = Field(min_length=1, max_length=2000)
    steps: list[FragmentStep] = Field(min_length=1, max_length=10)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    warnings: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("agentId", "contribution", "summary")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("steps", "assumptions", "warnings")
    @classmethod
    def _lists(cls, value):
        return FrozenList(value)

    @model_validator(mode="after")
    def _unique_step_ids(self):
        ids = [step.stepId for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("fragment step IDs must be unique")
        return self


class ProposalStep(FragmentStep):
    contributedBy: str

    @field_validator("contributedBy")
    @classmethod
    def _contributor(cls, value):
        return non_empty_id(value, "contributedBy")


class PlanProposal(AssistantContract):
    robotId: str
    summary: str = Field(min_length=1, max_length=2000)
    steps: list[ProposalStep] = Field(default_factory=list, max_length=10)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    warnings: list[str] = Field(default_factory=list, max_length=20)
    contributions: list[PlanFragment] = Field(default_factory=list, max_length=8)
    contextVersion: str

    @field_validator("robotId", "summary", "contextVersion")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("steps", "assumptions", "warnings", "contributions")
    @classmethod
    def _lists(cls, value):
        return FrozenList(value)

    @model_validator(mode="after")
    def _validate_merged_plan(self):
        if not self.steps and not self.contributions:
            return self
        if not self.steps or not self.contributions:
            raise ValueError("action plan steps and contributions must be supplied together")
        ensure_acyclic_steps(self.steps)
        agent_ids = [fragment.agentId for fragment in self.contributions]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("worker contribution agent IDs must be unique")
        known_agents = set(agent_ids)
        for step in self.steps:
            if step.contributedBy not in known_agents:
                raise ValueError(f"plan step contributor is missing: {step.contributedBy}")
        source_steps: dict[str, tuple[str, FragmentStep]] = {}
        for fragment in self.contributions:
            for source in fragment.steps:
                if source.stepId in source_steps:
                    raise ValueError(f"worker fragments reuse step ID: {source.stepId}")
                source_steps[source.stepId] = (fragment.agentId, source)
        if len(self.steps) != len(source_steps):
            raise ValueError("merged plan omitted or invented a worker fragment step")
        for step in self.steps:
            source_entry = source_steps.get(step.stepId)
            if source_entry is None:
                raise ValueError(f"merged plan invented worker step: {step.stepId}")
            agent_id, source = source_entry
            if (
                step.contributedBy != agent_id
                or step.commandKey != source.commandKey
                or step.payload != source.payload
                or step.rationale != source.rationale
                or any(dependency not in step.dependsOn for dependency in source.dependsOn)
            ):
                raise ValueError(f"merged plan rewrote worker fragment step: {step.stepId}")
        return self


class NeedsInputResult(AssistantContract):
    status: Literal["needs_input"]
    assistantMessage: str = Field(min_length=1, max_length=4000)


class CannotPlanResult(AssistantContract):
    status: Literal["cannot_plan"]
    assistantMessage: str = Field(min_length=1, max_length=4000)
    reason: str = Field(min_length=1, max_length=1000)
    missingCapabilities: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("missingCapabilities")
    @classmethod
    def _missing(cls, value):
        return FrozenList(non_empty_id(item, "missing capability") for item in value)


class DraftReadyResult(AssistantContract):
    status: Literal["draft_ready"]
    assistantMessage: str = Field(min_length=1, max_length=4000)
    proposal: PlanProposal
    visualMission: VisualMissionSpec | None = None

    @model_validator(mode="after")
    def _exactly_one_plan_kind(self):
        has_action = bool(self.proposal.steps)
        if has_action == (self.visualMission is not None):
            raise ValueError("draft_ready must contain exactly one action or visual mission plan")
        if self.visualMission is not None and self.visualMission.robotId != self.proposal.robotId:
            raise ValueError("visual mission robot does not match proposal")
        return self


PlanningTurnResult = Annotated[
    NeedsInputResult | CannotPlanResult | DraftReadyResult,
    Field(discriminator="status"),
]


__all__ = [
    "AssistantContract",
    "CannotPlanResult",
    "ConversationMessage",
    "ConversationRole",
    "DraftReadyResult",
    "FragmentStep",
    "LogicalCommand",
    "NeedsInputResult",
    "PlanFragment",
    "PlanProposal",
    "PlanningEnvelope",
    "PlanningTurnResult",
    "ProposalStep",
    "TaskModality",
    "TaskRequest",
]
