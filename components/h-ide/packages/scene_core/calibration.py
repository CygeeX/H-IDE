"""Pure scene-to-map calibration and pose comparison math."""

from __future__ import annotations

import math
from typing import Sequence

from .models import Calibration, Point3D, PoseMap
from .spatial import normalize_yaw


_IDENTITY_4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

# A point picked from a real point cloud is a surface sample, not an exact
# mathematical point on the fitted plane.  Keep the tolerance in map metres so
# it behaves consistently for scans whose source units are not metres.
_MIN_ORIGIN_PLANE_TOLERANCE_METERS = 0.05
_MAX_ORIGIN_PLANE_TOLERANCE_METERS = 0.20
_REFERENCE_LENGTH_TOLERANCE_RATIO = 0.02


def identity_calibration() -> Calibration:
    """Return the ROS map identity calibration."""

    return Calibration(
        mode="ros_map_identity",
        TMapFromScene=_IDENTITY_4,
        TSceneFromMap=_IDENTITY_4,
        scaleMetersPerUnit=1.0,
        residualMeters=0.0,
    )


def _point(value) -> Point3D:
    if isinstance(value, PoseMap):
        return Point3D(x=value.x, y=value.y, z=value.z)
    return Point3D.model_validate(value)


def _coordinates(point: Point3D) -> tuple[float, float, float]:
    return point.x, point.y, point.z


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: Sequence[float]) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1e-15:
        raise ValueError("cannot normalize a zero-length vector")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


def _plane_normal_and_eigenvalues(
    points: Sequence[Point3D],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    count = len(points)
    centroid = tuple(
        sum(_coordinates(point)[axis] for point in points) / count for axis in range(3)
    )
    covariance = [[0.0] * 3 for _ in range(3)]
    for point in points:
        centered = tuple(value - centroid[axis] for axis, value in enumerate(_coordinates(point)))
        for row in range(3):
            for column in range(3):
                covariance[row][column] += centered[row] * centered[column]

    # Jacobi diagonalization of a real symmetric 3x3 covariance matrix.
    eigenvectors = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
    for _ in range(64):
        row, column = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(covariance[pair[0]][pair[1]]))
        off_diagonal = covariance[row][column]
        if abs(off_diagonal) <= 1e-15:
            break
        angle = 0.5 * math.atan2(
            2.0 * off_diagonal,
            covariance[column][column] - covariance[row][row],
        )
        cosine, sine = math.cos(angle), math.sin(angle)
        rr, cc = covariance[row][row], covariance[column][column]
        covariance[row][row] = cosine * cosine * rr - 2 * sine * cosine * off_diagonal + sine * sine * cc
        covariance[column][column] = sine * sine * rr + 2 * sine * cosine * off_diagonal + cosine * cosine * cc
        covariance[row][column] = covariance[column][row] = 0.0
        for other in range(3):
            if other in (row, column):
                continue
            old_row, old_column = covariance[other][row], covariance[other][column]
            covariance[other][row] = covariance[row][other] = cosine * old_row - sine * old_column
            covariance[other][column] = covariance[column][other] = sine * old_row + cosine * old_column
        for other in range(3):
            old_row, old_column = eigenvectors[other][row], eigenvectors[other][column]
            eigenvectors[other][row] = cosine * old_row - sine * old_column
            eigenvectors[other][column] = sine * old_row + cosine * old_column

    eigenvalues = tuple(max(covariance[index][index], 0.0) for index in range(3))
    ordered = tuple(sorted(range(3), key=lambda index: eigenvalues[index]))
    smallest, middle, largest = (eigenvalues[index] for index in ordered)
    if largest <= 0 or middle <= largest * 1e-12:
        raise ValueError("ground_points must contain at least 3 non-collinear points")
    normal = tuple(eigenvectors[row][ordered[0]] for row in range(3))
    dominant_axis = max(range(3), key=lambda index: abs(normal[index]))
    if normal[dominant_axis] < 0:
        normal = tuple(-component for component in normal)
    return _unit(normal), centroid, (smallest, middle, largest)


def compute_interactive_similarity(
    ground_points,
    origin_scene,
    x_axis_scene,
    known_distance_m: float,
    map_origin,
) -> Calibration:
    """Fit a Z-up, right-handed uniform similarity transform from scene picks."""

    points = [_point(value) for value in ground_points]
    if len(points) < 3:
        raise ValueError("ground_points requires at least 3 points")
    if not math.isfinite(known_distance_m) or known_distance_m <= 0:
        raise ValueError("known_distance_m must be finite and greater than zero")
    origin, x_axis, map_point = _point(origin_scene), _point(x_axis_scene), _point(map_origin)
    normal, centroid, _ = _plane_normal_and_eigenvalues(points)
    plane_residual_scene = math.sqrt(
        sum(
            _dot(
                normal,
                tuple(value - centroid[index] for index, value in enumerate(_coordinates(point))),
            )
            ** 2
            for point in points
        )
        / len(points)
    )
    origin_offset_scene = _dot(
        normal,
        tuple(value - centroid[index] for index, value in enumerate(_coordinates(origin))),
    )
    raw_x = tuple(
        axis - origin_value
        for axis, origin_value in zip(_coordinates(x_axis), _coordinates(origin))
    )
    normal_component = _dot(raw_x, normal)
    projected_x = tuple(value - normal_component * normal[index] for index, value in enumerate(raw_x))
    projected_length = _norm(projected_x)
    if projected_length <= 1e-12:
        raise ValueError("x_axis_scene direction must project onto the fitted ground plane")
    x_basis = _unit(projected_x)
    z_basis = normal
    y_basis = _unit(_cross(z_basis, x_basis))
    scale = known_distance_m / projected_length
    plane_residual_meters = scale * plane_residual_scene
    origin_plane_distance_meters = scale * abs(origin_offset_scene)
    origin_tolerance_meters = max(
        _MIN_ORIGIN_PLANE_TOLERANCE_METERS,
        min(_MAX_ORIGIN_PLANE_TOLERANCE_METERS, 3.0 * plane_residual_meters),
        min(
            _MAX_ORIGIN_PLANE_TOLERANCE_METERS,
            _REFERENCE_LENGTH_TOLERANCE_RATIO * known_distance_m,
        ),
    )
    if origin_plane_distance_meters > origin_tolerance_meters:
        raise ValueError(
            "origin_scene is too far from the fitted ground plane "
            f"({origin_plane_distance_meters:.3f} m > {origin_tolerance_meters:.3f} m)"
        )

    # Use the operator's XY selection while projecting its point-cloud noise
    # onto the fitted floor.  This makes map Z=0 (or map_origin.z) represent the
    # floor rather than the height of one noisy return.
    origin_values = tuple(
        value - origin_offset_scene * normal[index]
        for index, value in enumerate(_coordinates(origin))
    )
    map_values = _coordinates(map_point)
    basis = (x_basis, y_basis, z_basis)
    linear = [[scale * component for component in row] for row in basis]
    translation = [map_values[row] - _dot(linear[row], origin_values) for row in range(3)]
    map_from_scene = [
        [*linear[row], translation[row]] for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]

    inverse_linear = [
        [basis[column][row] / scale for column in range(3)]
        for row in range(3)
    ]
    inverse_translation = [
        origin_values[row] - _dot(inverse_linear[row], map_values) for row in range(3)
    ]
    scene_from_map = [
        [*inverse_linear[row], inverse_translation[row]] for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    residual = math.sqrt(
        (
            len(points) * plane_residual_meters * plane_residual_meters
            + origin_plane_distance_meters * origin_plane_distance_meters
        )
        / (len(points) + 1)
    )
    return Calibration(
        mode="interactive_similarity",
        TMapFromScene=map_from_scene,
        TSceneFromMap=scene_from_map,
        scaleMetersPerUnit=scale,
        residualMeters=residual,
    )


def poses_within_drift(
    reference: PoseMap,
    observed: PoseMap,
    translation_threshold_m: float,
    yaw_threshold_rad: float,
) -> bool:
    """Compare 3D translation and shortest-angle yaw drift inclusively."""

    thresholds = (translation_threshold_m, yaw_threshold_rad)
    if any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("pose drift thresholds must be finite and non-negative")
    translation = math.sqrt(
        (observed.x - reference.x) ** 2
        + (observed.y - reference.y) ** 2
        + (observed.z - reference.z) ** 2
    )
    yaw = abs(normalize_yaw(observed.yaw - reference.yaw))
    epsilon = 1e-12
    return translation <= translation_threshold_m + epsilon and yaw <= yaw_threshold_rad + epsilon


pose_within_thresholds = poses_within_drift


__all__ = [
    "compute_interactive_similarity",
    "identity_calibration",
    "pose_within_thresholds",
    "poses_within_drift",
]
