from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


NonEmpty = Annotated[str, Field(min_length=1)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class ResolutionStatus(StrEnum):
    QUEUED = "queued"
    RESOLVING = "resolving"
    SUCCEEDED = "succeeded"
    NEEDS_INPUT = "needs_input"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"


class GoalPurpose(StrEnum):
    OBSERVE = "observe"
    APPROACH = "approach"
    FINAL = "final"


class VisualTaskMode(StrEnum):
    OBJECT_APPROACH = "object_approach"
    LANDMARK_OBSERVATION = "landmark_observation"
    DIRECTIONAL_FREE_SPACE = "directional_free_space"
    ARRIVAL_VERIFICATION = "arrival_verification"
    SCENE_DESCRIPTION = "scene_description"


class VerificationOutcome(StrEnum):
    CONFIRMED = "confirmed"
    NOT_CONFIRMED = "not_confirmed"
    UNKNOWN = "unknown"


class CameraIntrinsics(FrozenModel):
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float
    cy: float

    @field_validator("fx", "fy", "cx", "cy")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("intrinsic must be finite")
        return value


class Pose3D(FrozenModel):
    frameId: NonEmpty = "map"
    x: float
    y: float
    z: float
    yaw: float = 0.0

    @field_validator("x", "y", "z", "yaw")
    @classmethod
    def finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("pose value must be finite")
        return value


class ResolutionSource(FrozenModel):
    robotId: NonEmpty
    robotBootId: NonEmpty
    gatewayInstanceId: NonEmpty
    sensorId: NonEmpty
    observationSessionId: NonEmpty
    localizationEpochId: NonEmpty
    sourceMode: Literal["live", "replay", "synthetic"]
    sourceId: NonEmpty
    bundleId: NonEmpty
    capturedAt: datetime
    calibrationSha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    sceneId: str | None = None
    sceneRevision: int | None = Field(default=None, ge=1)


class ResolutionVersions(FrozenModel):
    vlmProvider: Literal["qwen"] = "qwen"
    vlmModel: NonEmpty
    promptVersion: NonEmpty
    visualAlgorithmVersion: NonEmpty
    approachPolicyVersion: NonEmpty
    approachPolicySha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    segmentationProvider: Literal["sam2"] | None = None
    segmentationModel: str | None = None
    poseModel: str | None = None


class SafetyGate(FrozenModel):
    name: NonEmpty
    passed: bool
    measured: float | int | str | bool | None = None
    threshold: float | int | str | bool | None = None
    detail: str | None = None


class EvidenceRef(FrozenModel):
    kind: NonEmpty
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    mediaType: NonEmpty
    producer: Literal[
        "observation_gateway",
        "qwen",
        "sam2",
        "deterministic_rgbd",
        "legacy_unverified",
    ] = "legacy_unverified"
    model: str | None = None
    expiresAt: datetime | None = None


class AmbiguityCandidate(FrozenModel):
    candidateId: NonEmpty
    label: NonEmpty
    box: tuple[int, int, int, int]
    score: float | None = None


class VisualResolutionRequest(FrozenModel):
    requestId: NonEmpty
    missionId: NonEmpty
    stageId: NonEmpty
    attempt: int = Field(ge=1)
    traceId: NonEmpty
    instruction: NonEmpty
    targetDescription: NonEmpty
    robotId: NonEmpty
    robotBootId: NonEmpty
    localizationEpochId: NonEmpty
    sensorId: NonEmpty
    requestedSourceMode: Literal["live", "replay", "synthetic"]
    sceneId: str | None = None
    sceneRevision: int | None = Field(default=None, ge=1)
    goalPurpose: GoalPurpose = GoalPurpose.APPROACH
    candidateId: str | None = None
    selectionSourceVisualResolutionId: str | None = None

    @model_validator(mode="after")
    def candidate_selection_is_bound(self) -> VisualResolutionRequest:
        if (self.candidateId is None) != (self.selectionSourceVisualResolutionId is None):
            raise ValueError(
                "candidateId and selectionSourceVisualResolutionId must be supplied together"
            )
        if self.candidateId is not None and self.attempt < 2:
            raise ValueError("candidate selection requires a new resolution attempt")
        return self


class VisualResolution(FrozenModel):
    visualResolutionId: NonEmpty
    missionId: NonEmpty
    stageId: NonEmpty
    attempt: int = Field(ge=1)
    traceId: NonEmpty
    status: ResolutionStatus
    executable: bool
    targetPhrase: NonEmpty
    source: ResolutionSource
    versions: ResolutionVersions
    landmarkPose: Pose3D | None = None
    goalPose: Pose3D | None = None
    goalPurpose: GoalPurpose = GoalPurpose.APPROACH
    safetyGates: tuple[SafetyGate, ...] = ()
    ambiguityCandidates: tuple[AmbiguityCandidate, ...] = ()
    selectedCandidateId: str | None = None
    selectionSourceVisualResolutionId: str | None = None
    warnings: tuple[str, ...] = ()
    rejectionReason: str | None = None
    evidence: tuple[EvidenceRef, ...] = ()
    createdAt: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def executable_is_deterministic(self) -> VisualResolution:
        if (self.selectedCandidateId is None) != (
            self.selectionSourceVisualResolutionId is None
        ):
            raise ValueError("selected candidate binding is incomplete")
        if self.selectedCandidateId is not None and self.attempt < 2:
            raise ValueError("selected candidate resolution requires a new attempt")
        if self.status is ResolutionStatus.NEEDS_INPUT and not self.ambiguityCandidates:
            raise ValueError("needs_input resolution requires ambiguity candidates")
        gates_pass = bool(self.safetyGates) and all(gate.passed for gate in self.safetyGates)
        valid_success = (
            self.status is ResolutionStatus.SUCCEEDED
            and self.landmarkPose is not None
            and self.goalPose is not None
            and gates_pass
            and not self.ambiguityCandidates
            and self.rejectionReason is None
        )
        if self.executable != valid_success:
            raise ValueError("executable must equal deterministic resolution gates")
        return self


class ApproachPolicy(FrozenModel):
    version: NonEmpty = "v1"
    robotRadiusM: float = Field(default=0.35, gt=0)
    targetStandOffM: float = Field(default=0.75, ge=0.25, le=3)
    minimumClearanceM: float = Field(default=0.15, ge=0.05, le=1)
    maximumGoalDistanceM: float = Field(default=5.0, gt=0, le=5)
    minimumValidDepthRatio: float = Field(default=0.60, ge=0, le=1)
    maximumFrameAgeMs: float = Field(default=1000, gt=0)
    maximumRgbDepthDeltaMs: float = Field(default=30, gt=0)
    maximumTfDeltaMs: float = Field(default=50, gt=0)
    maximumClockUncertaintyMs: float = Field(default=100, gt=0)
    requiredStableFrames: int = Field(default=3, ge=3, le=10)
    maximumClusterRadiusM: float = Field(default=0.25, gt=0)

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class VisualHypothesis(FrozenModel):
    """A semantic interpretation that is never itself a map-coordinate fact."""

    hypothesisId: NonEmpty
    source: Literal["qwen"] = "qwen"
    label: NonEmpty
    pixelBox: tuple[int, int, int, int]
    score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("pixelBox")
    @classmethod
    def ordered_box(
        cls, value: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        if any(item < 0 for item in value):
            raise ValueError("pixelBox coordinates must be non-negative")
        if value[0] >= value[2] or value[1] >= value[3]:
            raise ValueError("pixelBox must have positive area")
        return value


class VisualGeometryCandidate(FrozenModel):
    """Deterministic RGB-D/TF output eligible for H-IDE selection."""

    geometryCandidateId: NonEmpty
    hypothesisId: str | None = None
    source: Literal["deterministic_rgbd"] = "deterministic_rgbd"
    role: Literal["landmark", "approach", "free_space", "verification"]
    landmarkPose: Pose3D | None = None
    targetPose: Pose3D | None = None
    verified: bool
    safetyGates: tuple[SafetyGate, ...]
    evidenceIds: tuple[NonEmpty, ...] = ()

    @model_validator(mode="after")
    def verified_only_by_geometry_gates(self) -> VisualGeometryCandidate:
        gates_pass = bool(self.safetyGates) and all(gate.passed for gate in self.safetyGates)
        if self.verified != gates_pass:
            raise ValueError("verified must equal deterministic geometry gates")
        if self.landmarkPose is None and self.targetPose is None:
            raise ValueError("geometry candidate requires a landmark or target pose")
        if self.role in {"approach", "free_space"} and self.targetPose is None:
            raise ValueError("motion target candidate requires targetPose")
        return self


class VisualTaskRequest(FrozenModel):
    schemaVersion: Literal["h-ide.visual-task/2"] = "h-ide.visual-task/2"
    requestId: NonEmpty
    missionId: NonEmpty
    revisionId: NonEmpty
    nodeId: NonEmpty
    authorizationId: NonEmpty
    attempt: int = Field(ge=1, le=20)
    traceId: NonEmpty
    mode: VisualTaskMode
    instruction: NonEmpty
    targetDescription: str | None = None
    direction: Literal["forward", "left", "right"] | None = None
    requestedDistanceM: float | None = Field(default=None, gt=0, le=5)
    expectedTargetPose: Pose3D | None = None
    verificationMaximumDistanceM: float = Field(default=1.25, gt=0, le=3)
    verificationMaximumYawRad: float = Field(default=0.35, gt=0, le=math.pi)
    robotId: NonEmpty
    robotBootId: NonEmpty
    localizationEpochId: NonEmpty
    robotProfileDigest: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    capabilityManifestDigest: Annotated[
        str, Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ]
    sensorId: NonEmpty
    requestedSourceMode: Literal["live", "replay", "synthetic"]
    sceneId: str | None = None
    sceneRevision: int | None = Field(default=None, ge=1)
    candidateId: str | None = None
    selectionSourceAnalysisId: str | None = None

    @model_validator(mode="after")
    def mode_inputs_are_explicit(self) -> VisualTaskRequest:
        target_modes = {
            VisualTaskMode.OBJECT_APPROACH,
            VisualTaskMode.LANDMARK_OBSERVATION,
            VisualTaskMode.ARRIVAL_VERIFICATION,
        }
        if self.mode in target_modes and not (self.targetDescription or "").strip():
            raise ValueError("visual target mode requires targetDescription")
        if self.mode is VisualTaskMode.DIRECTIONAL_FREE_SPACE:
            if self.direction is None or self.requestedDistanceM is None:
                raise ValueError(
                    "directional_free_space requires direction and requestedDistanceM"
                )
        elif self.direction is not None or self.requestedDistanceM is not None:
            raise ValueError("direction fields are only valid for directional_free_space")
        if (self.candidateId is None) != (self.selectionSourceAnalysisId is None):
            raise ValueError(
                "candidateId and selectionSourceAnalysisId must be supplied together"
            )
        if self.candidateId is not None and self.attempt < 2:
            raise ValueError("candidate selection requires a new analysis attempt")
        return self


class VisualTaskResult(FrozenModel):
    schemaVersion: Literal["h-ide.visual-result/2"] = "h-ide.visual-result/2"
    analysisId: NonEmpty
    requestId: NonEmpty
    requestSha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    missionId: NonEmpty
    revisionId: NonEmpty
    nodeId: NonEmpty
    attempt: int = Field(ge=1, le=20)
    mode: VisualTaskMode
    status: ResolutionStatus
    source: ResolutionSource
    versions: ResolutionVersions
    hypotheses: tuple[VisualHypothesis, ...] = ()
    geometryCandidates: tuple[VisualGeometryCandidate, ...] = ()
    verification: VerificationOutcome | None = None
    observationSummary: str | None = Field(default=None, max_length=4000)
    directionalClear: bool | None = None
    ambiguityCandidates: tuple[AmbiguityCandidate, ...] = ()
    selectedCandidateId: str | None = None
    selectionSourceAnalysisId: str | None = None
    rejectionReason: str | None = None
    warnings: tuple[str, ...] = ()
    evidence: tuple[EvidenceRef, ...] = ()
    createdAt: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def terminal_invariants(self) -> VisualTaskResult:
        if (self.selectedCandidateId is None) != (
            self.selectionSourceAnalysisId is None
        ):
            raise ValueError("selected candidate binding is incomplete")
        if self.status is ResolutionStatus.NEEDS_INPUT and not self.ambiguityCandidates:
            raise ValueError("needs_input result requires ambiguity candidates")
        if self.status is ResolutionStatus.SUCCEEDED:
            if self.rejectionReason is not None:
                raise ValueError("successful visual result cannot have rejectionReason")
            if self.mode is VisualTaskMode.SCENE_DESCRIPTION:
                if not (self.observationSummary or "").strip():
                    raise ValueError(
                        "successful scene description requires observationSummary"
                    )
                if self.geometryCandidates:
                    raise ValueError(
                        "scene description must not fabricate geometry candidates"
                    )
            elif self.mode is VisualTaskMode.DIRECTIONAL_FREE_SPACE:
                if len(self.geometryCandidates) != 1:
                    raise ValueError(
                        "successful directional observation requires one geometry candidate"
                    )
                candidate = self.geometryCandidates[0]
                if candidate.role != "free_space":
                    raise ValueError(
                        "directional observation requires a free_space candidate"
                    )
                if self.directionalClear is None:
                    raise ValueError(
                        "successful directional observation requires directionalClear"
                    )
                if self.directionalClear != candidate.verified:
                    raise ValueError(
                        "directionalClear must match deterministic geometry gates"
                    )
            elif not self.geometryCandidates or not all(
                candidate.verified for candidate in self.geometryCandidates
            ):
                raise ValueError("successful visual result requires verified geometry")
        if self.status in {ResolutionStatus.REJECTED, ResolutionStatus.FAILED}:
            if not self.rejectionReason:
                raise ValueError("rejected visual result requires rejectionReason")
        if self.mode is VisualTaskMode.ARRIVAL_VERIFICATION:
            if self.status is ResolutionStatus.SUCCEEDED and self.verification is None:
                raise ValueError("arrival verification requires an outcome")
        elif self.verification is not None:
            raise ValueError("verification outcome is only valid for arrival verification")
        if self.mode is not VisualTaskMode.SCENE_DESCRIPTION and self.observationSummary is not None:
            raise ValueError("observationSummary is only valid for scene description")
        if self.mode is not VisualTaskMode.DIRECTIONAL_FREE_SPACE and self.directionalClear is not None:
            raise ValueError("directionalClear is only valid for directional free space")
        return self
