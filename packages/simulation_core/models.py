"""Versioned contracts for the H-IDE simulation boundary.

These contracts describe a simulated execution environment, not a second
robot-control protocol.  A simulator can provide observations and simulated
receipts, but physical execution remains impossible through these models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import math
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from packages.control_core.models import ContractModel
from packages.control_core.validation import (
    FrozenList,
    as_utc,
    freeze_json_object,
    non_empty_id,
    utc_now,
)
from packages.scene_core import Point2D, PoseMap, SceneRef


class SimulationBackend(str, Enum):
    DETERMINISTIC = "deterministic"
    MUJOCO_SONIC = "mujoco_sonic"
    GAZEBO = "gazebo"
    ISAAC = "isaac"


class SimulationMode(str, Enum):
    ORACLE = "oracle"
    QWEN = "qwen"


class SimulationStatus(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"


class SimulationEventType(str, Enum):
    STARTED = "simulation.started"
    PAUSED = "simulation.paused"
    RESET = "simulation.reset"
    STOPPED = "simulation.stopped"
    NAVIGATION_GOAL = "navigation.goal_received"
    NAVIGATION_FEEDBACK = "navigation.feedback"
    NAVIGATION_SUCCEEDED = "navigation.succeeded"
    NAVIGATION_BLOCKED = "navigation.blocked"
    NAVIGATION_FAILED = "navigation.failed"
    TARGET_LOST = "vision.target_lost"
    TARGET_FOUND = "vision.target_found"
    POSE_UPDATED = "localization.pose_updated"
    MISSION_BOUND = "mission.bound"


class SimulationPolicy(ContractModel):
    """Authorization policy attached to one simulation session.

    The initial policy intentionally requires a human decision for every
    mission-level revision and never permits a physical execution profile.
    ``allowAutoRevision`` is reserved for the later supervised release mode.
    """

    schemaVersion: Literal["h-ide.simulation-policy/1"] = "h-ide.simulation-policy/1"
    manualRevisionApproval: bool = True
    allowAutoRevision: bool = False
    allowPhysicalExecution: bool = False
    maximumAutoRevisions: int = Field(default=0, ge=0, le=20)
    maximumGoalDistanceM: float = Field(default=5.0, gt=0, le=25)
    maximumCumulativeDistanceM: float = Field(default=20.0, gt=0, le=200)
    maximumDurationSeconds: int = Field(default=600, gt=0, le=7_200)

    @model_validator(mode="after")
    def _safety_invariants(self):
        if self.allowPhysicalExecution:
            raise ValueError("simulation policy cannot allow physical execution")
        if not self.allowAutoRevision and self.maximumAutoRevisions != 0:
            raise ValueError(
                "maximumAutoRevisions must be zero while automatic revisions are disabled"
            )
        if self.manualRevisionApproval and self.allowAutoRevision:
            # This is allowed for a future supervised mode: revisions inside
            # the envelope may inherit policy, while out-of-envelope changes
            # still require a human.  Keep the explicit flag rather than
            # silently changing the authority boundary.
            return self
        return self


class SimulationObstacle(ContractModel):
    obstacleId: str
    label: str = "static obstacle"
    center: Point2D
    radiusM: float = Field(default=0.35, gt=0, le=10)
    static: bool = True

    @field_validator("obstacleId", "label")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)


class SimulationLandmark(ContractModel):
    landmarkId: str
    label: str
    pose: PoseMap
    aliases: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("landmarkId", "label")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, value):
        normalized = [non_empty_id(item, "aliases") for item in value]
        return FrozenList(dict.fromkeys(normalized))


class SimulationSessionCreate(ContractModel):
    schemaVersion: Literal["h-ide.simulation-session-create/1"] = "h-ide.simulation-session-create/1"
    backend: SimulationBackend = SimulationBackend.DETERMINISTIC
    executionProfile: Literal["simulation"] = "simulation"
    sourceMode: Literal["synthetic", "replay"] = "synthetic"
    # Reuse the seeded workstation robot by default so Mission V2 and the
    # simulation observatory share one identity/boot/epoch binding.
    robotId: str = "robot-1"
    robotBootId: str | None = None
    localizationEpochId: str | None = None
    missionId: str | None = None
    sceneRef: SceneRef | None = None
    mode: SimulationMode = SimulationMode.ORACLE
    startPose: PoseMap = Field(
        default_factory=lambda: PoseMap(x=0.0, y=0.0, z=0.0, yaw=0.0, observedAt=utc_now())
    )
    policy: SimulationPolicy = Field(default_factory=SimulationPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("robotId")
    @classmethod
    def _robot_id(cls, value: str):
        return non_empty_id(value, "robotId")

    @field_validator("robotBootId", "localizationEpochId")
    @classmethod
    def _optional_ids(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("missionId")
    @classmethod
    def _mission_id(cls, value: str | None):
        return None if value is None else non_empty_id(value, "missionId")

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, value):
        return freeze_json_object(value)


class SimulationSession(ContractModel):
    schemaVersion: Literal["h-ide.simulation-session/1"] = "h-ide.simulation-session/1"
    sessionId: str
    backend: SimulationBackend
    executionProfile: Literal["simulation"] = "simulation"
    sourceMode: Literal["synthetic", "replay"]
    robotId: str
    robotBootId: str
    localizationEpochId: str
    missionId: str | None = None
    sceneRef: SceneRef | None = None
    mode: SimulationMode
    status: SimulationStatus
    releaseState: Literal["simulation_only", "validated", "supervised_auto"] = "simulation_only"
    policy: SimulationPolicy
    metadata: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    createdAt: datetime
    updatedAt: datetime

    @field_validator(
        "sessionId",
        "robotId",
        "robotBootId",
        "localizationEpochId",
    )
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("missionId")
    @classmethod
    def _mission_id(cls, value: str | None):
        return None if value is None else non_empty_id(value, "missionId")

    @field_validator("metadata")
    @classmethod
    def _metadata(cls, value):
        return freeze_json_object(value)

    @field_validator("createdAt", "updatedAt")
    @classmethod
    def _timestamps(cls, value):
        return as_utc(value)

    @model_validator(mode="after")
    def _invariants(self):
        if self.executionProfile != "simulation":
            raise ValueError("simulation session must use executionProfile=simulation")
        if self.releaseState == "supervised_auto" and not self.policy.allowAutoRevision:
            raise ValueError("supervised_auto requires allowAutoRevision")
        return self


class SimulationSnapshot(ContractModel):
    schemaVersion: Literal["h-ide.simulation-snapshot/1"] = "h-ide.simulation-snapshot/1"
    sessionId: str
    status: SimulationStatus
    navigationState: Literal["idle", "moving", "blocked", "arrived", "failed"] = "idle"
    pose: PoseMap
    goalPose: PoseMap | None = None
    trajectory: list[PoseMap] = Field(default_factory=list, max_length=10_000)
    obstacles: list[SimulationObstacle] = Field(default_factory=list, max_length=500)
    landmarks: list[SimulationLandmark] = Field(default_factory=list, max_length=500)
    plannedWaypoints: list[PoseMap] = Field(default_factory=list, max_length=100)
    activeMissionId: str | None = None
    activeRevisionId: str | None = None
    pendingReview: "SimulationReview | None" = None
    lastEventSequence: int = Field(default=0, ge=0)
    updatedAt: datetime

    @field_validator("sessionId", "activeMissionId", "activeRevisionId")
    @classmethod
    def _optional_ids(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("trajectory", "obstacles", "landmarks", "plannedWaypoints")
    @classmethod
    def _freeze_lists(cls, value, info):
        return FrozenList(value)

    @field_validator("updatedAt")
    @classmethod
    def _timestamp(cls, value):
        return as_utc(value)


class SimulationReview(ContractModel):
    """A durable, non-executable review marker for simulation anomalies.

    This is deliberately separate from a Mission ``PlanRevision``.  The
    marker tells the UI that a blocked/failed/visual-loss event needs review;
    only Mission V2 may create/approve a revision, and no field here can
    authorize an actuator call.
    """

    schemaVersion: Literal["h-ide.simulation-review/1"] = "h-ide.simulation-review/1"
    status: Literal["pending", "resolved", "error"] = "pending"
    triggerEventId: str
    triggerEventType: str
    missionId: str | None = None
    decisionKind: Literal["replan", "needs_review", "pause", "continue", "error"] = "needs_review"
    revisionId: str | None = None
    detail: str
    createdAt: datetime
    updatedAt: datetime

    @field_validator("triggerEventId", "triggerEventType", "detail")
    @classmethod
    def _strings(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("missionId", "revisionId")
    @classmethod
    def _optional_ids(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)

    @field_validator("createdAt", "updatedAt")
    @classmethod
    def _times(cls, value):
        return as_utc(value)


class SimulationEventInput(ContractModel):
    eventType: SimulationEventType
    payload: dict[str, Any] = Field(default_factory=dict, validate_default=True)

    @field_validator("payload")
    @classmethod
    def _payload(cls, value):
        return freeze_json_object(value)


class SimulationEvent(ContractModel):
    schemaVersion: Literal["h-ide.simulation-event/1"] = "h-ide.simulation-event/1"
    eventId: str
    sessionId: str
    sequence: int = Field(ge=1)
    eventType: SimulationEventType
    source: Literal["operator", "simulator", "robot_agent", "nav2", "oracle", "qwen"]
    payload: dict[str, Any] = Field(default_factory=dict, validate_default=True)
    occurredAt: datetime

    @field_validator("eventId", "sessionId")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("payload")
    @classmethod
    def _payload(cls, value):
        return freeze_json_object(value)

    @field_validator("occurredAt")
    @classmethod
    def _timestamp(cls, value):
        return as_utc(value)


class NavigationGoalInput(ContractModel):
    goalPose: PoseMap
    missionId: str | None = None
    revisionId: str | None = None
    nodeId: str | None = None

    @field_validator("missionId", "revisionId", "nodeId")
    @classmethod
    def _optional_ids(cls, value: str | None, info):
        return None if value is None else non_empty_id(value, info.field_name)


class SimulationStepInput(ContractModel):
    distanceM: float = Field(default=0.25, gt=0, le=2.0)


class SimulationReceipt(ContractModel):
    schemaVersion: Literal["h-ide.simulation-receipt/1"] = "h-ide.simulation-receipt/1"
    receiptId: str
    sessionId: str
    commandId: str
    status: Literal["accepted", "running", "succeeded", "blocked", "failed", "cancelled"]
    executionBackend: str = "deterministic_simulator"
    motion: Literal["simulated", "mujoco"] = "simulated"
    goalPose: PoseMap | None = None
    detail: str | None = None
    createdAt: datetime

    @field_validator("receiptId", "sessionId", "commandId")
    @classmethod
    def _ids(cls, value: str, info):
        return non_empty_id(value, info.field_name)

    @field_validator("createdAt")
    @classmethod
    def _timestamp(cls, value):
        return as_utc(value)


def distance_xy(first: PoseMap, second: PoseMap) -> float:
    return math.hypot(second.x - first.x, second.y - first.y)
