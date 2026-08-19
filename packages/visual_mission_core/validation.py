from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Mapping

from pydantic import ValidationError

from packages.scene_core import PoseMap, SceneRef, VisualNavigationPayload

from .models import VisualMissionPlanDraft, VisualMissionStage


class VisualResolutionRejected(ValueError):
    pass


def validate_visual_resolution_binding(
    draft: VisualMissionPlanDraft,
    stage: VisualMissionStage,
    value: Mapping[str, Any],
    *,
    now: datetime | None = None,
    expected_attempt: int | None = None,
    selected_candidate_id: str | None = None,
    selection_source_visual_resolution_id: str | None = None,
) -> None:
    """Validate immutable request bindings for every terminal resolution.

    A deterministic safety rejection is a valid, auditable result, but it must
    satisfy the same mission, observation and version bindings as a successful
    resolution before the runtime may persist it as such.
    """
    now = now or datetime.now(timezone.utc)

    def required(name: str) -> Any:
        result = value.get(name)
        if result is None:
            raise VisualResolutionRejected(f"visual resolution is missing {name}")
        return result

    if required("missionId") != draft.mission.missionId or required("stageId") != stage.stageId:
        raise VisualResolutionRejected("visual resolution mission/stage mismatch")
    if expected_attempt is not None and value.get("attempt") != expected_attempt:
        raise VisualResolutionRejected("visual resolution attempt mismatch")
    if value.get("selectedCandidateId") != selected_candidate_id:
        raise VisualResolutionRejected("visual resolution selected candidate mismatch")
    if (
        value.get("selectionSourceVisualResolutionId")
        != selection_source_visual_resolution_id
    ):
        raise VisualResolutionRejected("visual resolution selection source mismatch")
    source = required("source")
    versions = required("versions")
    if not isinstance(source, Mapping) or not isinstance(versions, Mapping):
        raise VisualResolutionRejected("visual resolution source/versions are invalid")
    mission = draft.mission
    bindings = {
        "robotId": mission.robotId,
        "robotBootId": mission.robotBootId,
        "localizationEpochId": mission.localizationEpochId,
    }
    for key, expected in bindings.items():
        if source.get(key) != expected:
            raise VisualResolutionRejected(f"visual resolution {key} mismatch")
    if source.get("sensorId") not in mission.sensorIds:
        raise VisualResolutionRejected("visual resolution sensor is not allowed")
    if source.get("sourceMode") != mission.requestedSourceMode:
        raise VisualResolutionRejected("visual resolution source mode mismatch")
    if mission.requestedSourceMode != "live" and value.get("releaseState") == "physical_robot_validated":
        raise VisualResolutionRejected("synthetic/replay resolution cannot claim physical validation")
    version_bindings = {
        "vlmProvider": mission.vlmProvider,
        "vlmModel": mission.vlmModel,
        "promptVersion": mission.promptVersion,
        "visualAlgorithmVersion": mission.visualAlgorithmVersion,
        "approachPolicyVersion": mission.approachPolicyVersion,
        "approachPolicySha256": mission.approachPolicySha256,
    }
    for key, expected in version_bindings.items():
        if versions.get(key) != expected:
            raise VisualResolutionRejected(f"visual resolution {key} changed after review")
    if mission.sceneRef is not None:
        if source.get("sceneId") != mission.sceneRef.sceneId or source.get("sceneRevision") != mission.sceneRef.revision:
            raise VisualResolutionRejected("visual resolution scene binding mismatch")
    created_at = value.get("createdAt")
    try:
        created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise VisualResolutionRejected("visual resolution timestamp is invalid") from exc
    if created > now or (now - created).total_seconds() > 30:
        raise VisualResolutionRejected("visual resolution is stale")


def validate_visual_resolution(
    draft: VisualMissionPlanDraft,
    stage: VisualMissionStage,
    value: Mapping[str, Any],
    *,
    now: datetime | None = None,
    expected_attempt: int | None = None,
    selected_candidate_id: str | None = None,
    selection_source_visual_resolution_id: str | None = None,
) -> VisualNavigationPayload:
    now = now or datetime.now(timezone.utc)

    validate_visual_resolution_binding(
        draft,
        stage,
        value,
        now=now,
        expected_attempt=expected_attempt,
        selected_candidate_id=selected_candidate_id,
        selection_source_visual_resolution_id=selection_source_visual_resolution_id,
    )

    def required(name: str) -> Any:
        result = value.get(name)
        if result is None:
            raise VisualResolutionRejected(f"visual resolution is missing {name}")
        return result

    if value.get("status") != "succeeded" or value.get("executable") is not True:
        raise VisualResolutionRejected(str(value.get("rejectionReason") or "resolution is not executable"))
    source = required("source")
    mission = draft.mission
    gates = value.get("safetyGates")
    if not isinstance(gates, list) or not gates or not all(
        isinstance(gate, Mapping) and gate.get("passed") is True for gate in gates
    ):
        raise VisualResolutionRejected("visual resolution safety gates did not pass")
    try:
        goal = PoseMap.model_validate(required("goalPose"))
    except ValidationError as exc:
        raise VisualResolutionRejected("visual resolution goalPose is invalid") from exc
    if goal.frameId != "map":
        raise VisualResolutionRejected("visual resolution goal must use map frame")
    distance = math.hypot(goal.x, goal.y)
    if distance > mission.maximumGoalDistanceM:
        raise VisualResolutionRejected("visual resolution exceeds mission distance limit")
    return VisualNavigationPayload(
        goalPose=goal,
        goalPurpose=stage.goalPurpose,
        missionId=mission.missionId,
        stageId=stage.stageId,
        visualResolutionId=str(required("visualResolutionId")),
        observationSessionId=str(required("source").get("observationSessionId")),
        robotBootId=mission.robotBootId,
        localizationEpochId=mission.localizationEpochId,
        maximumSpeedMps=mission.maximumSpeedMps,
        sceneRef=None if mission.sceneRef is None else mission.sceneRef.model_dump(mode="json"),
    )
