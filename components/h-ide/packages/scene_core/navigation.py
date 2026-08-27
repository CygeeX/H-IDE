"""Validation for deterministic spatial plans crossing the Agent boundary."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Literal, Mapping

from pydantic import Field, model_validator

from .calibration import poses_within_drift
from .models import PoseMap, SceneContext, SceneModel, SpatialResolution
from .spatial import RESOLVER_VERSION, resolve_spatial_target


NAVIGATION_COMMAND_KEY = "navigation.navigate_to_pose"


def _same_pose(left: PoseMap, right: PoseMap, tolerance: float = 1e-7) -> bool:
    return (
        left.frameId == right.frameId
        and all(
            math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)
            for a, b in (
                (left.x, right.x),
                (left.y, right.y),
                (left.z, right.z),
                (left.yaw, right.yaw),
            )
        )
        and left.observedAt == right.observedAt
    )


class NavigationPayload(SceneModel):
    goalPose: PoseMap
    spatialResolution: SpatialResolution

    @model_validator(mode="after")
    def _same_goal(self):
        if not _same_pose(self.goalPose, self.spatialResolution.goalPose):
            raise ValueError("goalPose does not match spatialResolution.goalPose")
        return self


class VisualNavigationPayload(SceneModel):
    """One audited visual target pose; never a path or velocity sequence."""

    payloadType: Literal["visual_resolution"] = "visual_resolution"
    goalPose: PoseMap
    goalPurpose: Literal["observe", "approach", "final"]
    missionId: str = Field(min_length=1, max_length=256)
    stageId: str = Field(min_length=1, max_length=256)
    visualResolutionId: str = Field(min_length=1, max_length=256)
    observationSessionId: str = Field(min_length=1, max_length=256)
    robotBootId: str = Field(min_length=1, max_length=256)
    localizationEpochId: str = Field(min_length=1, max_length=256)
    maximumSpeedMps: float = Field(gt=0, le=0.5)
    sceneRef: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _robot_agent_bounds(self):
        if not -math.pi <= self.goalPose.yaw <= math.pi:
            raise ValueError("goalPose.yaw must be between -pi and pi")
        if self.sceneRef is not None:
            if set(self.sceneRef) != {"sceneId", "revision"}:
                raise ValueError("sceneRef must contain only sceneId and revision")
            scene_id = self.sceneRef.get("sceneId")
            revision = self.sceneRef.get("revision")
            if not isinstance(scene_id, str) or not scene_id.strip():
                raise ValueError("sceneRef.sceneId must be non-empty")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise ValueError("sceneRef.revision must be an integer greater than zero")
        return self


class SpatialNavigationExecutionPayload(SceneModel):
    """One reviewed static-map candidate plus fresh Robot Agent bindings."""

    payloadType: Literal["spatial_resolution"] = "spatial_resolution"
    goalPose: PoseMap
    goalPurpose: Literal["final"] = "final"
    missionId: str = Field(min_length=1, max_length=256)
    stageId: str = Field(min_length=1, max_length=256)
    spatialResolutionId: str = Field(min_length=1, max_length=256)
    observationSessionId: str = Field(min_length=1, max_length=256)
    robotBootId: str = Field(min_length=1, max_length=256)
    localizationEpochId: str = Field(min_length=1, max_length=256)
    maximumSpeedMps: float = Field(gt=0, le=0.5)
    sceneRef: dict[str, Any]

    @model_validator(mode="after")
    def _robot_agent_bounds(self):
        if not -math.pi <= self.goalPose.yaw <= math.pi:
            raise ValueError("goalPose.yaw must be between -pi and pi")
        if set(self.sceneRef) != {"sceneId", "revision"}:
            raise ValueError("sceneRef must contain only sceneId and revision")
        scene_id = self.sceneRef.get("sceneId")
        revision = self.sceneRef.get("revision")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise ValueError("sceneRef.sceneId must be non-empty")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("sceneRef.revision must be an integer greater than zero")
        return self


class RelativeNavigationExecutionPayload(SceneModel):
    """A reviewed relative translation resolved from a fresh map pose."""

    payloadType: Literal["relative_resolution"] = "relative_resolution"
    goalPose: PoseMap
    goalPurpose: Literal["final"] = "final"
    missionId: str = Field(min_length=1, max_length=256)
    stageId: str = Field(min_length=1, max_length=256)
    relativeResolutionId: str = Field(min_length=1, max_length=256)
    observationSessionId: str = Field(min_length=1, max_length=256)
    robotBootId: str = Field(min_length=1, max_length=256)
    localizationEpochId: str = Field(min_length=1, max_length=256)
    maximumSpeedMps: float = Field(gt=0, le=0.5)

    @model_validator(mode="after")
    def _robot_agent_bounds(self):
        if not -math.pi <= self.goalPose.yaw <= math.pi:
            raise ValueError("goalPose.yaw must be between -pi and pi")
        return self


_VISUAL_NAVIGATION_COMMAND_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:h-ide:robot-navigation:v1:navigation.navigate_to_pose",
    "title": "H-IDE visual navigation target",
    "description": (
        "One reviewed map-frame target pose. This contract contains no path or "
        "velocity sequence."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": [
        "payloadType",
        "goalPose",
        "goalPurpose",
        "missionId",
        "stageId",
        "visualResolutionId",
        "observationSessionId",
        "robotBootId",
        "localizationEpochId",
        "maximumSpeedMps",
    ],
    "properties": {
        "payloadType": {"const": "visual_resolution"},
        "goalPose": {
            "type": "object",
            "additionalProperties": False,
            "required": ["frameId", "x", "y", "z", "yaw"],
            "properties": {
                "frameId": {"const": "map"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "yaw": {"type": "number", "minimum": -math.pi, "maximum": math.pi},
                "observedAt": {"type": ["string", "null"], "format": "date-time"},
            },
        },
        "goalPurpose": {"enum": ["observe", "approach", "final"]},
        "missionId": {"type": "string", "minLength": 1, "maxLength": 256},
        "stageId": {"type": "string", "minLength": 1, "maxLength": 256},
        "visualResolutionId": {"type": "string", "minLength": 1, "maxLength": 256},
        "observationSessionId": {"type": "string", "minLength": 1, "maxLength": 256},
        "robotBootId": {"type": "string", "minLength": 1, "maxLength": 256},
        "localizationEpochId": {"type": "string", "minLength": 1, "maxLength": 256},
        "maximumSpeedMps": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 0.5,
        },
        "sceneRef": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["sceneId", "revision"],
            "properties": {
                "sceneId": {"type": "string", "minLength": 1},
                "revision": {"type": "integer", "minimum": 1},
            },
        },
    },
}


def _robot_navigation_command_schema() -> dict[str, Any]:
    schema = visual_navigation_command_schema()
    schema["title"] = "H-IDE reviewed navigation target"
    schema["description"] = (
        "One reviewed visual, static-map, or relative target pose with fresh Robot Agent "
        "observation bindings. This contract contains no path or velocity sequence."
    )
    schema["properties"]["payloadType"] = {
        "enum": ["visual_resolution", "spatial_resolution", "relative_resolution"]
    }
    schema["properties"]["spatialResolutionId"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
    }
    schema["properties"]["relativeResolutionId"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
    }
    schema["required"].remove("visualResolutionId")
    schema["allOf"] = [
        {
            "if": {"properties": {"payloadType": {"const": "visual_resolution"}}},
            "then": {
                "required": ["visualResolutionId"],
                "not": {
                    "anyOf": [
                        {"required": ["spatialResolutionId"]},
                        {"required": ["relativeResolutionId"]},
                    ]
                },
            },
        },
        {
            "if": {"properties": {"payloadType": {"const": "spatial_resolution"}}},
            "then": {
                "required": ["spatialResolutionId", "sceneRef"],
                "properties": {
                    "goalPurpose": {"const": "final"},
                    "sceneRef": {"type": "object"},
                },
                "not": {
                    "anyOf": [
                        {"required": ["visualResolutionId"]},
                        {"required": ["relativeResolutionId"]},
                    ]
                },
            },
        },
        {
            "if": {"properties": {"payloadType": {"const": "relative_resolution"}}},
            "then": {
                "required": ["relativeResolutionId"],
                "properties": {"goalPurpose": {"const": "final"}},
                "not": {
                    "anyOf": [
                        {"required": ["visualResolutionId"]},
                        {"required": ["spatialResolutionId"]},
                        {"required": ["sceneRef"]},
                    ]
                },
            },
        },
    ]
    return schema


def visual_navigation_command_schema() -> dict[str, Any]:
    """Return the frozen Robot Agent V1 request schema as a defensive copy."""

    return deepcopy(_VISUAL_NAVIGATION_COMMAND_SCHEMA)


def robot_navigation_command_schema() -> dict[str, Any]:
    """Return the additive visual-or-spatial Robot Agent handoff schema."""

    return deepcopy(_robot_navigation_command_schema())


def build_spatial_navigation_execution_payload(
    value: NavigationPayload | Mapping[str, Any],
    robot_details: Mapping[str, Any],
    *,
    mission_id: str,
    stage_id: str,
    maximum_speed_mps: float = 0.2,
) -> SpatialNavigationExecutionPayload:
    """Bind a reviewed PCD candidate to the current live observation session."""

    navigation = (
        value if isinstance(value, NavigationPayload) else NavigationPayload.model_validate(value)
    )
    resolution = navigation.spatialResolution
    return SpatialNavigationExecutionPayload(
        goalPose=navigation.goalPose,
        missionId=mission_id,
        stageId=stage_id,
        spatialResolutionId=resolution.resolutionId,
        observationSessionId=robot_details.get("observationSessionId"),
        robotBootId=robot_details.get("robotBootId"),
        localizationEpochId=robot_details.get("localizationEpochId"),
        maximumSpeedMps=maximum_speed_mps,
        sceneRef=resolution.sceneRef.model_dump(mode="json"),
    )


def navigation_command_schema() -> dict[str, Any]:
    """Compatibility schema for PCD/Fake and visual demonstration commands."""

    legacy = NavigationPayload.model_json_schema(by_alias=True)
    legacy_defs = dict(legacy.pop("$defs", {}))
    return {
        "type": "object",
        "$defs": legacy_defs,
        "oneOf": [legacy, visual_navigation_command_schema()],
    }


def validate_navigation_payload(
    context: SceneContext,
    value: NavigationPayload | Mapping[str, Any],
    *,
    translation_threshold_m: float = 0.0,
    yaw_threshold_rad: float = 0.0,
) -> NavigationPayload:
    """Re-resolve a navigation payload against canonical immutable scene facts."""

    payload = value if isinstance(value, NavigationPayload) else NavigationPayload.model_validate(value)
    resolution = payload.spatialResolution
    if resolution.sceneRef != context.sceneRef:
        raise ValueError("spatial resolution uses a stale or different scene revision")
    if resolution.resolverVersion != RESOLVER_VERSION:
        raise ValueError("spatial resolution uses an unsupported resolver version")

    if resolution.sourcePose is None:
        if context.currentPose is not None:
            raise ValueError("spatial resolution omitted the planning pose")
    else:
        if context.currentPose is None:
            raise ValueError("current robot pose is unavailable")
        if not poses_within_drift(
            resolution.sourcePose,
            context.currentPose,
            translation_threshold_m,
            yaw_threshold_rad,
        ):
            raise ValueError("robot pose drift exceeds the allowed confirmation threshold")
        if (
            translation_threshold_m == 0
            and yaw_threshold_rad == 0
            and resolution.sourcePose.observedAt != context.currentPose.observedAt
        ):
            raise ValueError("spatial resolution uses a different planning pose snapshot")

    snapshot_context = context.model_copy(update={"currentPose": resolution.sourcePose})
    expected = resolve_spatial_target(
        snapshot_context,
        resolution.intent,
        resolution.resolutionId,
        resolution.computedAt,
    )
    if not _same_pose(expected.goalPose, resolution.goalPose):
        raise ValueError("spatial goal does not match deterministic resolution")
    if list(resolution.warnings) != list(expected.warnings):
        raise ValueError("spatial resolution warnings were altered")
    return payload


__all__ = [
    "NAVIGATION_COMMAND_KEY",
    "NavigationPayload",
    "SpatialNavigationExecutionPayload",
    "VisualNavigationPayload",
    "build_spatial_navigation_execution_payload",
    "navigation_command_schema",
    "robot_navigation_command_schema",
    "visual_navigation_command_schema",
    "validate_navigation_payload",
]
