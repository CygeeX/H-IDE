"""Validated scene contracts and deterministic spatial math."""

from .models import (
    Bounds3D,
    Calibration,
    Matrix4,
    Point2D,
    Point3D,
    PoseMap,
    RelativeSpatialIntent,
    SceneContext,
    SceneMapSummary,
    SceneRef,
    SelectedPointSpatialIntent,
    SemanticSpatialIntent,
    SemanticRegion,
    SpatialIntent,
    SpatialResolution,
    TargetMapCheck,
    normalize_semantic_label,
    normalized_semantic_labels,
    semantic_label_collisions,
)
from .spatial import RESOLVER_VERSION, normalize_yaw, resolve_spatial_target
from .calibration import (
    compute_interactive_similarity,
    identity_calibration,
    pose_within_thresholds,
    poses_within_drift,
)
from .navigation import (
    NAVIGATION_COMMAND_KEY,
    NavigationPayload,
    RelativeNavigationExecutionPayload,
    SpatialNavigationExecutionPayload,
    VisualNavigationPayload,
    build_spatial_navigation_execution_payload,
    navigation_command_schema,
    robot_navigation_command_schema,
    visual_navigation_command_schema,
    validate_navigation_payload,
)

# Short aliases are convenient at command/agent boundaries while preserving the
# explicit class names in schemas and diagnostics.
RelativeIntent = RelativeSpatialIntent
SemanticIntent = SemanticSpatialIntent
SelectedPointIntent = SelectedPointSpatialIntent

__all__ = [
    "Bounds3D",
    "Calibration",
    "compute_interactive_similarity",
    "Matrix4",
    "NAVIGATION_COMMAND_KEY",
    "NavigationPayload",
    "RelativeNavigationExecutionPayload",
    "SpatialNavigationExecutionPayload",
    "VisualNavigationPayload",
    "build_spatial_navigation_execution_payload",
    "navigation_command_schema",
    "robot_navigation_command_schema",
    "visual_navigation_command_schema",
    "Point2D",
    "Point3D",
    "identity_calibration",
    "PoseMap",
    "RESOLVER_VERSION",
    "RelativeIntent",
    "RelativeSpatialIntent",
    "SceneContext",
    "SceneMapSummary",
    "SceneRef",
    "SelectedPointIntent",
    "SelectedPointSpatialIntent",
    "SemanticIntent",
    "SemanticSpatialIntent",
    "SemanticRegion",
    "SpatialIntent",
    "SpatialResolution",
    "TargetMapCheck",
    "normalize_semantic_label",
    "normalize_yaw",
    "pose_within_thresholds",
    "poses_within_drift",
    "normalized_semantic_labels",
    "semantic_label_collisions",
    "resolve_spatial_target",
    "validate_navigation_payload",
]
