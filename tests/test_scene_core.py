from datetime import datetime, timezone
import importlib
import importlib.util
import math

import pytest
from pydantic import ValidationError


def test_scene_core_package_is_available():
    assert importlib.util.find_spec("packages.scene_core") is not None


def _scene():
    return importlib.import_module("packages.scene_core")


def _pose(**overrides):
    values = {
        "frameId": "map",
        "x": 1.0,
        "y": 2.0,
        "z": 0.5,
        "yaw": 0.25,
        "observedAt": datetime(2026, 8, 10, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return _scene().PoseMap(**values)


def test_pose_map_requires_map_frame_finite_values_and_aware_timestamp():
    pose = _pose()
    assert pose.frameId == "map"
    assert pose.observedAt == datetime(2026, 8, 10, tzinfo=timezone.utc)

    with pytest.raises(ValidationError, match="frameId"):
        _pose(frameId="odom")
    with pytest.raises(ValidationError, match="finite"):
        _pose(x=math.inf)
    with pytest.raises(ValidationError, match="timezone-aware"):
        _pose(observedAt=datetime(2026, 8, 10))


def test_scene_reference_and_bounds_validate_identifiers_and_ordering():
    scene = _scene()
    ref = scene.SceneRef(sceneId="warehouse", revision="rev-4")
    bounds = scene.Bounds3D(min=(-2, -3, 0), max=(4, 5, 2))
    assert ref.sceneId == "warehouse"
    assert bounds.min.x == -2
    assert bounds.contains(scene.Point3D(x=4, y=5, z=2))

    with pytest.raises(ValidationError):
        scene.SceneRef(sceneId=" ", revision="rev-4")
    with pytest.raises(ValidationError, match="ordered"):
        scene.Bounds3D(min=(1, 0, 0), max=(0, 1, 1))
    with pytest.raises(ValidationError, match="finite"):
        scene.Bounds3D(min=(0, 0, 0), max=(1, math.nan, 1))


def test_semantic_region_validates_polygon_height_and_aliases():
    scene = _scene()
    region = scene.SemanticRegion(
        locationId="loading-bay",
        name="  Loading   Bay ",
        aliases=["Dock", "Receiving"],
        polygonXY=[(0, 0), (3, 0), (3, 2), (0, 2)],
        zMin=0,
        zMax=2.5,
        entryPose=_pose(x=1, y=1, z=0),
    )
    assert region.name == "Loading Bay"
    assert region.aliases == ["Dock", "Receiving"]
    assert scene.normalized_semantic_labels(region) == {
        "loading bay",
        "dock",
        "receiving",
    }

    invalid_values = [
        {"polygonXY": [(0, 0), (1, 1), (0, 1), (1, 0)]},
        {"polygonXY": [(0, 0), (1, 0)]},
        {"zMin": 3, "zMax": 2},
        {"aliases": ["dock", " DOCK "]},
        {"aliases": [" Loading Bay "]},
        {"aliases": [" "]},
    ]
    base = {
        "locationId": "loading-bay",
        "name": "Loading Bay",
        "aliases": ["Dock"],
        "polygonXY": [(0, 0), (3, 0), (3, 2), (0, 2)],
        "zMin": 0,
        "zMax": 2.5,
        "entryPose": _pose(),
    }
    for overrides in invalid_values:
        with pytest.raises(ValidationError):
            scene.SemanticRegion(**(base | overrides))


def test_omitted_collection_defaults_are_immutable():
    scene = _scene()
    region = scene.SemanticRegion(
        locationId="default-aliases",
        name="Default Aliases",
        polygonXY=[(0, 0), (1, 0), (0, 1)],
        zMin=0,
        zMax=1,
        entryPose=_pose(),
    )
    context = scene.SceneContext(
        sceneRef=scene.SceneRef(sceneId="warehouse", revision="r1"),
        bounds=scene.Bounds3D(min=(-1, -1, -1), max=(1, 1, 1)),
    )
    resolution = scene.SpatialResolution(
        resolutionId="default-warnings",
        sceneRef=context.sceneRef,
        intent=scene.RelativeSpatialIntent(forwardMeters=0, lateralMeters=0, yawDeltaRad=0),
        goalPose=_pose(x=0, y=0, z=0),
        resolverVersion="scene-core/1",
        computedAt=datetime.now(timezone.utc),
    )

    for values in (region.aliases, context.annotations, resolution.warnings):
        with pytest.raises(TypeError):
            values.append("mutation")


def test_semantic_collision_helper_uses_normalized_names_and_aliases():
    scene = _scene()

    def region(location_id, name, aliases):
        return scene.SemanticRegion(
            locationId=location_id,
            name=name,
            aliases=aliases,
            polygonXY=[(0, 0), (1, 0), (0, 1)],
            zMin=0,
            zMax=1,
            entryPose=_pose(),
        )

    collisions = scene.semantic_label_collisions(
        [
            region("dock-east", "East Dock", ["Receiving"]),
            region("receiving", " receiving ", []),
            region("office", "Office", []),
        ]
    )
    assert collisions == {"receiving": ("dock-east", "receiving")}


def test_calibration_contract_requires_valid_inverse_and_metrics():
    scene = _scene()
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    calibration = scene.Calibration(
        mode="ros_map_identity",
        TMapFromScene=identity,
        TSceneFromMap=identity,
        scaleMetersPerUnit=1,
        residualMeters=0,
    )
    assert calibration.TMapFromScene[0][0] == 1

    with pytest.raises(ValidationError):
        calibration.model_copy(update={"scaleMetersPerUnit": 0})
    with pytest.raises(ValidationError):
        calibration.model_copy(update={"residualMeters": -0.1})
    with pytest.raises(ValidationError, match="inverse"):
        calibration.model_copy(
            update={"TSceneFromMap": [[2, 0, 0, 0], *identity[1:]]}
        )


def test_calibration_rejects_ill_conditioned_one_way_inverse_pair():
    scene = _scene()
    map_from_scene = [
        [1e-20, 1.0, 0.0, 0.0],
        [0.0, 1e20, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    # map_from_scene * scene_from_map is identity within 1e-7, but the
    # reverse product has an off-diagonal value of 1e20.
    scene_from_map = [
        [1e20, 0.0, 0.0, 0.0],
        [0.0, 1e-20, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    with pytest.raises(ValidationError, match="inverse"):
        scene.Calibration(
            mode="interactive_similarity",
            TMapFromScene=map_from_scene,
            TSceneFromMap=scene_from_map,
            scaleMetersPerUnit=1,
            residualMeters=0,
        )


def test_robot_state_accepts_an_optional_typed_map_pose():
    from packages.control_core.models import RobotState

    state = RobotState(robotId="robot-1", pose=_pose())
    assert state.pose.x == 1
    without_pose = RobotState(robotId="robot-2")
    assert without_pose.pose is None
    assert getattr(RobotState.model_fields["pose"], "exclude_if", None) is None
    assert "pose" not in without_pose.model_dump(mode="json")
    assert state.model_dump(mode="json")["pose"]["frameId"] == "map"
    with pytest.raises(ValidationError):
        RobotState(robotId="robot-3", pose={"frameId": "odom", "x": 0, "y": 0, "z": 0, "yaw": 0})


def _region(location_id="dock", entry_pose=None):
    scene = _scene()
    return scene.SemanticRegion(
        locationId=location_id,
        name=location_id.title(),
        polygonXY=[(-1, -1), (1, -1), (1, 1), (-1, 1)],
        zMin=0,
        zMax=2,
        entryPose=entry_pose or _pose(x=0, y=0, z=0, yaw=0),
    )


def _context(**overrides):
    scene = _scene()
    values = {
        "sceneRef": scene.SceneRef(sceneId="warehouse", revision="r1"),
        "bounds": scene.Bounds3D(min=(-10, -10, -1), max=(10, 10, 3)),
        "annotations": [_region()],
        "currentPose": _pose(x=1, y=2, z=0.5, yaw=math.pi / 2),
        "selectedPoint": scene.Point3D(x=4, y=6, z=0.25),
    }
    values.update(overrides)
    return scene.SceneContext(**values)


def test_relative_target_uses_current_yaw_and_wraps_goal_yaw():
    scene = _scene()
    context = _context(currentPose=_pose(x=1, y=2, z=0.5, yaw=3.0))
    intent = scene.RelativeSpatialIntent(
        forwardMeters=2,
        lateralMeters=1,
        yawDeltaRad=0.5,
    )
    computed_at = datetime(2026, 8, 10, 8, tzinfo=timezone.utc)
    resolution = scene.resolve_spatial_target(context, intent, "resolution-1", computed_at)

    assert resolution.resolutionId == "resolution-1"
    assert resolution.sceneRef == context.sceneRef
    assert resolution.intent.kind == "relative"
    assert resolution.sourcePose == context.currentPose
    assert resolution.goalPose.x == pytest.approx(1 + 2 * math.cos(3) - math.sin(3))
    assert resolution.goalPose.y == pytest.approx(2 + 2 * math.sin(3) + math.cos(3))
    assert resolution.goalPose.z == 0.5
    assert resolution.goalPose.yaw == pytest.approx(3.5 - 2 * math.pi)
    assert resolution.computedAt == computed_at


def test_relative_target_requires_current_pose():
    scene = _scene()
    with pytest.raises(ValueError, match="currentPose"):
        scene.resolve_spatial_target(
            _context(currentPose=None),
            scene.RelativeSpatialIntent(forwardMeters=1, lateralMeters=0, yawDeltaRad=0),
            "resolution-2",
            datetime.now(timezone.utc),
        )


def test_semantic_target_resolves_exact_location_id_entry_pose():
    scene = _scene()
    entry = _pose(x=-3, y=4, z=0, yaw=-1.25)
    context = _context(annotations=[_region("dock", entry), _region("office")])
    resolution = scene.resolve_spatial_target(
        context,
        scene.SemanticSpatialIntent(locationId="dock"),
        "resolution-semantic",
        datetime.now(timezone.utc),
    )
    assert resolution.goalPose == entry
    with pytest.raises(ValueError, match="unknown semantic locationId"):
        scene.resolve_spatial_target(
            context,
            scene.SemanticSpatialIntent(locationId="Dock"),
            "resolution-unknown",
            datetime.now(timezone.utc),
        )


def test_selected_point_uses_explicit_yaw_or_faces_from_current_pose():
    scene = _scene()
    context = _context(currentPose=_pose(x=1, y=2, z=0.5, yaw=-2))
    point = scene.Point3D(x=4, y=6, z=0.25)
    facing = scene.resolve_spatial_target(
        context,
        scene.SelectedPointSpatialIntent(pointMap=point),
        "resolution-facing",
        datetime.now(timezone.utc),
    )
    assert facing.goalPose.yaw == pytest.approx(math.atan2(4, 3))

    explicit = scene.resolve_spatial_target(
        context,
        scene.SelectedPointSpatialIntent(pointMap=point, yaw=4),
        "resolution-explicit",
        datetime.now(timezone.utc),
    )
    assert explicit.goalPose.yaw == pytest.approx(4 - 2 * math.pi)

    with pytest.raises(ValueError, match="yaw"):
        scene.resolve_spatial_target(
            _context(currentPose=None),
            scene.SelectedPointSpatialIntent(pointMap=point),
            "resolution-no-heading",
            datetime.now(timezone.utc),
        )


def test_selected_point_rejects_intent_coordinate_unrelated_to_context_selection():
    scene = _scene()
    with pytest.raises(ValueError, match="SceneContext.selectedPoint"):
        scene.resolve_spatial_target(
            _context(selectedPoint=scene.Point3D(x=4, y=6, z=0.25)),
            scene.SelectedPointSpatialIntent(pointMap=(4.5, 6, 0.25), yaw=0),
            "resolution-substituted-point",
            datetime.now(timezone.utc),
        )


def test_selected_point_requires_authoritative_context_selection():
    scene = _scene()
    with pytest.raises(ValueError, match="SceneContext.selectedPoint"):
        scene.resolve_spatial_target(
            _context(selectedPoint=None),
            scene.SelectedPointSpatialIntent(pointMap=(4, 6, 0.25), yaw=0),
            "resolution-missing-selection",
            datetime.now(timezone.utc),
        )


def test_resolver_rejects_any_goal_outside_xyz_bounds():
    scene = _scene()
    context = _context(bounds=scene.Bounds3D(min=(0, 0, 0), max=(2, 3, 1)))
    with pytest.raises(ValueError, match="outside scene bounds"):
        scene.resolve_spatial_target(
            context,
            scene.RelativeSpatialIntent(forwardMeters=4, lateralMeters=0, yawDeltaRad=0),
            "resolution-oob-xy",
            datetime.now(timezone.utc),
        )
    with pytest.raises(ValueError, match="outside scene bounds"):
        scene.resolve_spatial_target(
            context.model_copy(update={"selectedPoint": (1, 1, 2)}),
            scene.SelectedPointSpatialIntent(pointMap=(1, 1, 2), yaw=0),
            "resolution-oob-z",
            datetime.now(timezone.utc),
        )


def _transform(matrix, point):
    homogeneous = [point[0], point[1], point[2], 1.0]
    result = [sum(matrix[row][column] * homogeneous[column] for column in range(4)) for row in range(4)]
    return tuple(result[:3])


def test_identity_calibration_has_identity_transforms():
    scene = _scene()
    calibration = scene.identity_calibration()
    assert calibration.mode == "ros_map_identity"
    assert calibration.scaleMetersPerUnit == 1
    assert calibration.residualMeters == 0
    assert _transform(calibration.TMapFromScene, (2, -3, 4)) == pytest.approx((2, -3, 4))
    assert calibration.TMapFromScene == calibration.TSceneFromMap


def test_interactive_similarity_recovers_known_rotated_scaled_transform():
    scene = _scene()
    # Ground is the scene plane x=2. Scene +Y is calibrated as map +X,
    # scene +Z becomes map +Y, and the fitted +X normal becomes map +Z.
    calibration = scene.compute_interactive_similarity(
        ground_points=[(2, 0, 0), (2, 1, 0), (2, 0, 1), (2, 2, 3)],
        origin_scene=(2, 3, 4),
        x_axis_scene=(2, 5, 4),
        known_distance_m=6,
        map_origin=(10, -1, 0),
    )

    assert calibration.mode == "interactive_similarity"
    assert calibration.scaleMetersPerUnit == pytest.approx(3)
    assert calibration.residualMeters == pytest.approx(0, abs=1e-12)
    assert _transform(calibration.TMapFromScene, (2, 3, 4)) == pytest.approx((10, -1, 0))
    assert _transform(calibration.TMapFromScene, (2, 5, 4)) == pytest.approx((16, -1, 0))
    map_point = _transform(calibration.TMapFromScene, (2, 4, 6))
    assert _transform(calibration.TSceneFromMap, map_point) == pytest.approx((2, 4, 6))


def test_interactive_similarity_rejects_origin_far_from_fitted_ground():
    scene = _scene()
    with pytest.raises(ValueError, match="origin_scene.*ground plane"):
        scene.compute_interactive_similarity(
            ground_points=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            origin_scene=(0, 0, 1),
            x_axis_scene=(1, 0, 1),
            known_distance_m=1,
            map_origin=(0, 0, 0),
        )


def test_interactive_similarity_projects_realistic_origin_pick_noise_to_ground():
    scene = _scene()
    calibration = scene.compute_interactive_similarity(
        # Exactly three samples have a mathematically zero plane residual.  A
        # separately picked LiDAR return still needs a practical tolerance.
        ground_points=[(0, 0, 0), (2, 0, 0), (0, 2, 0)],
        origin_scene=(0.25, 0.5, 0.03),
        x_axis_scene=(1.25, 0.5, 0.04),
        known_distance_m=1,
        map_origin=(10, 20, 0),
    )

    assert _transform(calibration.TMapFromScene, (0.25, 0.5, 0)) == pytest.approx((10, 20, 0))
    assert calibration.scaleMetersPerUnit == pytest.approx(1)
    assert calibration.residualMeters == pytest.approx(0.015)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ground_points": [(0, 0, 0), (1, 0, 0)]}, "at least 3"),
        ({"ground_points": [(0, 0, 0), (1, 0, 0), (2, 0, 0)]}, "non-collinear"),
        ({"known_distance_m": 0}, "known_distance_m"),
        ({"x_axis_scene": (0, 0, 2)}, "project"),
    ],
)
def test_interactive_similarity_rejects_invalid_plane_and_scale(overrides, message):
    scene = _scene()
    values = {
        "ground_points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        "origin_scene": (0, 0, 0),
        "x_axis_scene": (1, 0, 0),
        "known_distance_m": 1,
        "map_origin": (0, 0, 0),
    }
    with pytest.raises(ValueError, match=message):
        scene.compute_interactive_similarity(**(values | overrides))


def test_pose_drift_comparison_uses_translation_and_shortest_yaw_delta():
    scene = _scene()
    reference = _pose(x=0, y=0, z=0, yaw=math.pi - 0.05)
    observed = _pose(x=0.3, y=0.4, z=0, yaw=-math.pi + 0.05)
    assert scene.poses_within_drift(
        reference,
        observed,
        translation_threshold_m=0.5,
        yaw_threshold_rad=0.1,
    )
    assert not scene.poses_within_drift(reference, observed, 0.49, 0.1)
    assert not scene.poses_within_drift(reference, observed, 0.5, 0.09)
    with pytest.raises(ValueError, match="threshold"):
        scene.poses_within_drift(reference, observed, -1, 0.1)


@pytest.mark.parametrize("angle", [math.inf, -math.inf, math.nan])
def test_normalize_yaw_rejects_non_finite_angles(angle):
    with pytest.raises(ValueError, match="finite"):
        _scene().normalize_yaw(angle)
