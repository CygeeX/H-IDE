"""Strict contracts shared by H-IDE and the planning-only Agent service.

These models intentionally contain logical commands and robot context only.
Transport mappings, ROS names, adapter endpoints, and credentials never cross
the planning boundary.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import hashlib
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


class ConversationIntent(str, Enum):
    """Authoritative Assistant V2 intent classes returned by h-ide-agent."""

    SOCIAL_CHAT = "SOCIAL_CHAT"
    GENERAL_QA = "GENERAL_QA"
    ROBOT_OBSERVE = "ROBOT_OBSERVE"
    ROBOT_ACTION = "ROBOT_ACTION"
    SCENE_TEACH = "SCENE_TEACH"
    ADMIN_DEV = "ADMIN_DEV"


class ConversationRoute(str, Enum):
    REPLY = "reply"
    VISUAL_OBSERVATION = "visual_observation"
    SCENE_TEACHING = "scene_teaching"
    MISSION_REVIEW = "mission_review"


class ConversationHistoryMessage(AssistantContract):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def _content(cls, value: str) -> str:
        return non_empty_id(value, "content")


class ConversationDurableSummary(AssistantContract):
    """Auditable H-IDE-owned conversational summary.

    The digest makes accidental/cross-row summary substitution fail before the
    internal Agent boundary.  The summary is conversational context only; it
    is deliberately not a scene, localization, or safety contract.
    """

    summaryId: str = Field(min_length=1, max_length=160)
    revision: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=12000)
    coveredThroughTurn: int = Field(ge=0)
    generatedAt: datetime
    method: Literal["SERVER_COMPACTION", "OPERATOR_APPROVED"]
    contentSha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("summaryId", "text")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("generatedAt")
    @classmethod
    def _generated_at(cls, value: datetime) -> datetime:
        return as_utc(value)

    @model_validator(mode="after")
    def _digest_matches_text(self):
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if digest != self.contentSha256:
            raise ValueError("durable summary digest does not match its text")
        return self


class ConversationMemorySnippet(AssistantContract):
    memoryId: str = Field(min_length=1, max_length=160)
    kind: Literal[
        "USER_PREFERENCE",
        "CONVERSATION_FACT",
        "PRIOR_DECISION",
        "ONGOING_TOPIC",
        "CONVERSATION_EXCERPT",
    ]
    text: str = Field(min_length=1, max_length=2000)
    recordedAt: datetime
    sourceTurn: int | None = Field(default=None, ge=0)
    relevanceScore: float | None = Field(default=None, ge=0, le=1)

    @field_validator("memoryId", "text")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("recordedAt")
    @classmethod
    def _recorded_at(cls, value: datetime) -> datetime:
        return as_utc(value)


class ConversationMemoryContext(AssistantContract):
    """Server-selected, non-physical context for conversation continuity."""

    schemaVersion: Literal["h-ide.conversation-memory-context/1"] = (
        "h-ide.conversation-memory-context/1"
    )
    providedBy: Literal["H_IDE_SERVER"] = "H_IDE_SERVER"
    authority: Literal["CONVERSATIONAL_ONLY"] = "CONVERSATIONAL_ONLY"
    resourceId: str = Field(min_length=1, max_length=160)
    revision: int = Field(ge=0)
    durableSummary: ConversationDurableSummary | None = None
    relevantMemories: list[ConversationMemorySnippet] = Field(
        default_factory=list,
        max_length=8,
    )

    @field_validator("resourceId")
    @classmethod
    def _resource_id(cls, value: str) -> str:
        return non_empty_id(value, "resourceId")

    @field_validator("relevantMemories")
    @classmethod
    def _memory_budget(cls, value):
        if sum(len(item.text) for item in value) > 12000:
            raise ValueError("relevant memory snippets exceed the 12000 character budget")
        return FrozenList(value)


class ConversationTurnRequest(AssistantContract):
    """Bounded request sent from the H-IDE gateway to the routing Agent."""

    schemaVersion: Literal["h-ide.conversation-turn/2"] = "h-ide.conversation-turn/2"
    conversationId: str
    robotId: str | None = None
    locale: str = Field(default="zh-CN", min_length=2, max_length=35)
    text: str = Field(min_length=1, max_length=8000)
    history: list[ConversationHistoryMessage] = Field(default_factory=list, max_length=20)
    # Compatibility only. New H-IDE servers build memoryContext from their
    # persisted transcript; browser-provided summaries are never forwarded.
    summary: str | None = Field(default=None, min_length=1, max_length=12000)
    memoryContext: ConversationMemoryContext | None = None

    @field_validator("conversationId", "robotId", "locale", "text", "summary")
    @classmethod
    def _strings(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("history")
    @classmethod
    def _history(cls, value):
        if sum(len(item.content) for item in value) > 40000:
            raise ValueError("recent conversation history exceeds the 40000 character budget")
        return FrozenList(value)


class ConversationTurnResponse(AssistantContract):
    """Routing answer. This contract never contains executable robot data."""

    schemaVersion: Literal["h-ide.conversation-result/2"]
    conversationId: str
    robotId: str | None = None
    intent: ConversationIntent
    assistantMessage: str = Field(min_length=1, max_length=4000)
    needsRobot: bool
    shouldCreateMission: bool
    clarification: str | None = Field(default=None, min_length=1, max_length=1000)

    @field_validator("conversationId", "robotId", "assistantMessage", "clarification")
    @classmethod
    def _strings(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @model_validator(mode="after")
    def _derived_authority_flags(self):
        robot_intents = {
            ConversationIntent.ROBOT_OBSERVE,
            ConversationIntent.ROBOT_ACTION,
            ConversationIntent.SCENE_TEACH,
        }
        if self.needsRobot != (self.intent in robot_intents):
            raise ValueError("needsRobot must be derived from intent")
        expected_mission = (
            self.intent is ConversationIntent.ROBOT_ACTION
            and self.clarification is None
        )
        if self.shouldCreateMission != expected_mission:
            raise ValueError("only an unambiguous ROBOT_ACTION may create a Mission")
        return self


class AssistantTurnEvent(AssistantContract):
    """Durable event-poll contract that can later be exposed as SSE unchanged."""

    schemaVersion: Literal["h-ide.assistant-event/2"] = "h-ide.assistant-event/2"
    eventId: str
    taskId: str
    conversationId: str
    sequence: int = Field(ge=1)
    type: Literal[
        "turn.accepted",
        "turn.started",
        "route.selected",
        "turn.completed",
        "turn.failed",
        "turn.cancelled",
    ]
    data: dict[str, Any] = Field(default_factory=dict)
    createdAt: datetime = Field(default_factory=utc_now)

    @field_validator("eventId", "taskId", "conversationId")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("data")
    @classmethod
    def _data(cls, value):
        return freeze_json_object(value)

    @field_validator("createdAt")
    @classmethod
    def _created_at(cls, value: datetime) -> datetime:
        return as_utc(value)


class ConversationGatewayResult(AssistantContract):
    """Terminal H-IDE routing result persisted with an assistant turn."""

    schemaVersion: Literal["h-ide.assistant-result/2"] = "h-ide.assistant-result/2"
    taskId: str
    conversationId: str
    robotId: str | None = None
    intent: ConversationIntent
    route: ConversationRoute
    routeStatus: Literal[
        "completed",
        "needs_input",
        "needs_robot",
        "unavailable",
        "awaiting_confirmation",
        "review",
    ]
    assistantMessage: str = Field(min_length=1, max_length=4000)
    clarification: str | None = Field(default=None, min_length=1, max_length=1000)
    missionDraftId: str | None = None
    visualResult: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=100)

    @field_validator(
        "taskId",
        "conversationId",
        "robotId",
        "assistantMessage",
        "clarification",
        "missionDraftId",
    )
    @classmethod
    def _strings(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("visualResult")
    @classmethod
    def _visual_result(cls, value):
        return None if value is None else freeze_json_object(value)

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, value):
        return FrozenList(freeze_json_object(item) for item in value)

    @model_validator(mode="after")
    def _route_is_non_escalating(self):
        expected_route = {
            ConversationIntent.SOCIAL_CHAT: ConversationRoute.REPLY,
            ConversationIntent.GENERAL_QA: ConversationRoute.REPLY,
            ConversationIntent.ADMIN_DEV: ConversationRoute.REPLY,
            ConversationIntent.ROBOT_OBSERVE: ConversationRoute.VISUAL_OBSERVATION,
            ConversationIntent.SCENE_TEACH: ConversationRoute.SCENE_TEACHING,
            ConversationIntent.ROBOT_ACTION: ConversationRoute.MISSION_REVIEW,
        }[self.intent]
        if self.route is not expected_route:
            raise ValueError("conversation intent was routed to the wrong authority boundary")
        has_mission = self.missionDraftId is not None
        if has_mission != (
            self.intent is ConversationIntent.ROBOT_ACTION
            and self.routeStatus == "review"
        ):
            raise ValueError("a Mission draft belongs only to a reviewable ROBOT_ACTION")
        if self.route is not ConversationRoute.VISUAL_OBSERVATION and (
            self.visualResult is not None or self.evidence
        ):
            raise ValueError("visual evidence belongs only to a read-only observation route")
        return self


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
    "AssistantTurnEvent",
    "AssistantContract",
    "CannotPlanResult",
    "ConversationDurableSummary",
    "ConversationGatewayResult",
    "ConversationHistoryMessage",
    "ConversationIntent",
    "ConversationMemoryContext",
    "ConversationMemorySnippet",
    "ConversationMessage",
    "ConversationRole",
    "ConversationRoute",
    "ConversationTurnRequest",
    "ConversationTurnResponse",
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
