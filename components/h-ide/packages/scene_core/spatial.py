"""Pure deterministic spatial resolution helpers."""

from __future__ import annotations

from datetime import datetime
import math

from .models import (
    PoseMap,
    RelativeSpatialIntent,
    SceneContext,
    SelectedPointSpatialIntent,
    SemanticSpatialIntent,
    SpatialIntent,
    SpatialResolution,
)


RESOLVER_VERSION = "scene-core/1"


def normalize_yaw(angle: float) -> float:
    """Normalize an angle to the half-open interval [-pi, pi)."""

    if not math.isfinite(angle):
        raise ValueError("yaw angle must be finite")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def resolve_spatial_target(
    context: SceneContext,
    intent: SpatialIntent,
    resolution_id: str,
    computed_at: datetime,
) -> SpatialResolution:
    """Resolve an intent into one immutable map-frame goal pose."""

    source = context.currentPose
    if isinstance(intent, RelativeSpatialIntent):
        if source is None:
            raise ValueError("relative intent requires currentPose")
        cosine, sine = math.cos(source.yaw), math.sin(source.yaw)
        goal = PoseMap(
            x=source.x + intent.forwardMeters * cosine - intent.lateralMeters * sine,
            y=source.y + intent.forwardMeters * sine + intent.lateralMeters * cosine,
            z=source.z,
            yaw=normalize_yaw(source.yaw + intent.yawDeltaRad),
        )
    elif isinstance(intent, SemanticSpatialIntent):
        match = next(
            (region for region in context.annotations if region.locationId == intent.locationId),
            None,
        )
        if match is None:
            raise ValueError(f"unknown semantic locationId: {intent.locationId}")
        goal = match.entryPose
    elif isinstance(intent, SelectedPointSpatialIntent):
        selected = context.selectedPoint
        if selected is None:
            raise ValueError("selected_point intent requires SceneContext.selectedPoint")
        if any(
            not math.isclose(selected_value, intent_value, rel_tol=0.0, abs_tol=1e-9)
            for selected_value, intent_value in (
                (selected.x, intent.pointMap.x),
                (selected.y, intent.pointMap.y),
                (selected.z, intent.pointMap.z),
            )
        ):
            raise ValueError("selected_point pointMap must match SceneContext.selectedPoint")
        if intent.yaw is not None:
            yaw = normalize_yaw(intent.yaw)
        elif source is not None:
            yaw = normalize_yaw(math.atan2(selected.y - source.y, selected.x - source.x))
        else:
            raise ValueError("selected_point intent requires yaw when currentPose is unavailable")
        goal = PoseMap(
            x=selected.x,
            y=selected.y,
            z=selected.z,
            yaw=yaw,
        )
    else:  # pragma: no cover - closed validated union, defensive for direct Python callers
        raise TypeError(f"unsupported spatial intent: {type(intent).__name__}")

    if not context.bounds.contains(goal):
        raise ValueError("resolved goal is outside scene bounds")
    return SpatialResolution(
        resolutionId=resolution_id,
        sceneRef=context.sceneRef,
        intent=intent,
        sourcePose=source,
        goalPose=goal,
        resolverVersion=RESOLVER_VERSION,
        warnings=[],
        computedAt=computed_at,
    )


__all__ = ["RESOLVER_VERSION", "normalize_yaw", "resolve_spatial_target"]
