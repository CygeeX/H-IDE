"""Immutable scene-domain contracts."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _FrozenList(list):
    def _immutable(self, *args, **kwargs):
        raise TypeError("scene contract values are immutable")

    append = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _immutable
    __setitem__ = __delitem__ = _immutable


class SceneModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False):
        data = self.model_dump(mode="python")
        if update:
            data.update(update)
        return type(self).model_validate(data)


class Point2D(SceneModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _from_sequence(cls, value):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
            if len(value) != 2:
                raise ValueError("2D point requires exactly two coordinates")
            return {"x": value[0], "y": value[1]}
        return value


class Point3D(SceneModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _from_sequence(cls, value):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
            if len(value) != 3:
                raise ValueError("3D point requires exactly three coordinates")
            return {"x": value[0], "y": value[1], "z": value[2]}
        return value


class PoseMap(SceneModel):
    frameId: Literal["map"] = "map"
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    z: float = Field(allow_inf_nan=False)
    yaw: float = Field(allow_inf_nan=False)
    observedAt: datetime | None = None

    @field_validator("observedAt")
    @classmethod
    def _aware_timestamp(cls, value: datetime | None):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observedAt must be timezone-aware")
        return value


class SceneRef(SceneModel):
    sceneId: str
    revision: str | int

    @field_validator("sceneId")
    @classmethod
    def _scene_id(cls, value: str):
        if not value.strip():
            raise ValueError("sceneId must be non-empty")
        return value.strip()

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str | int):
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("revision must be non-empty")
            return value.strip()
        if value < 0:
            raise ValueError("revision must be non-negative")
        return value


class Bounds3D(SceneModel):
    min: Point3D
    max: Point3D

    @model_validator(mode="after")
    def _ordered(self):
        if self.min.x > self.max.x or self.min.y > self.max.y or self.min.z > self.max.z:
            raise ValueError("bounds min/max must be ordered")
        return self

    def contains(self, point: Point3D | PoseMap) -> bool:
        return (
            self.min.x <= point.x <= self.max.x
            and self.min.y <= point.y <= self.max.y
            and self.min.z <= point.z <= self.max.z
        )


def normalize_semantic_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _display_label(value: str, field_name: str) -> str:
    normalized_spacing = " ".join(value.split())
    if not normalized_spacing:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized_spacing


def _orientation(a: Point2D, b: Point2D, c: Point2D) -> float:
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _on_segment(a: Point2D, b: Point2D, p: Point2D) -> bool:
    return (
        min(a.x, b.x) <= p.x <= max(a.x, b.x)
        and min(a.y, b.y) <= p.y <= max(a.y, b.y)
    )


def _segments_intersect(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> bool:
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    if (ab_c > 0 > ab_d or ab_d > 0 > ab_c) and (cd_a > 0 > cd_b or cd_b > 0 > cd_a):
        return True
    epsilon = 1e-12
    return (
        (abs(ab_c) <= epsilon and _on_segment(a, b, c))
        or (abs(ab_d) <= epsilon and _on_segment(a, b, d))
        or (abs(cd_a) <= epsilon and _on_segment(c, d, a))
        or (abs(cd_b) <= epsilon and _on_segment(c, d, b))
    )


def _validate_simple_polygon(points: list[Point2D]) -> None:
    count = len(points)
    if len({(point.x, point.y) for point in points}) != count:
        raise ValueError("polygon vertices must be distinct")
    area_twice = sum(
        points[index].x * points[(index + 1) % count].y
        - points[(index + 1) % count].x * points[index].y
        for index in range(count)
    )
    if abs(area_twice) <= 1e-12:
        raise ValueError("polygon must have non-zero area")
    for first in range(count):
        a, b = points[first], points[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or second == (first + 1) % count:
                continue
            if first == 0 and second == count - 1:
                continue
            c, d = points[second], points[(second + 1) % count]
            if _segments_intersect(a, b, c, d):
                raise ValueError("polygon must not self-intersect")


class SemanticRegion(SceneModel):
    locationId: str
    name: str
    aliases: list[str] = Field(default_factory=list, validate_default=True)
    parentLocationId: str | None = None
    anchorRoles: list[
        Literal["region", "entry", "interior", "observation_vantage"]
    ] = Field(default_factory=lambda: ["region"], validate_default=True)
    purpose: str | None = None
    scanHeadingsRad: list[float] = Field(
        default_factory=list,
        max_length=16,
        validate_default=True,
    )
    polygonXY: list[Point2D] = Field(min_length=3)
    zMin: float = Field(allow_inf_nan=False)
    zMax: float = Field(allow_inf_nan=False)
    entryPose: PoseMap

    @field_validator("locationId")
    @classmethod
    def _location_id(cls, value: str):
        return _display_label(value, "locationId")

    @field_validator("name")
    @classmethod
    def _name(cls, value: str):
        return _display_label(value, "name")

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, values: list[str]):
        return _FrozenList(_display_label(value, "alias") for value in values)

    @field_validator("parentLocationId", "purpose")
    @classmethod
    def _optional_labels(cls, value: str | None, info):
        return None if value is None else _display_label(value, info.field_name)

    @field_validator("anchorRoles")
    @classmethod
    def _anchor_roles(cls, values: list[str]):
        if not values or len(values) != len(set(values)):
            raise ValueError("anchorRoles must be non-empty and unique")
        return _FrozenList(values)

    @field_validator("scanHeadingsRad")
    @classmethod
    def _scan_headings(cls, values: list[float]):
        normalized = [float(value) for value in values]
        if any(not math.isfinite(value) for value in normalized):
            raise ValueError("scanHeadingsRad must contain finite angles")
        return _FrozenList(normalized)

    @field_validator("polygonXY")
    @classmethod
    def _polygon(cls, points: list[Point2D]):
        _validate_simple_polygon(points)
        return _FrozenList(points)

    @model_validator(mode="after")
    def _height_and_labels(self):
        if self.zMin > self.zMax:
            raise ValueError("zMin/zMax must be ordered")
        labels = [self.name, *self.aliases]
        normalized = [normalize_semantic_label(label) for label in labels]
        if len(normalized) != len(set(normalized)):
            raise ValueError("name and aliases must be unique after normalization")
        return self


def normalized_semantic_labels(region: SemanticRegion) -> set[str]:
    return {normalize_semantic_label(label) for label in (region.name, *region.aliases)}


def semantic_label_collisions(regions: Sequence[SemanticRegion]) -> dict[str, tuple[str, ...]]:
    locations_by_label: dict[str, list[str]] = {}
    for region in regions:
        for label in normalized_semantic_labels(region):
            locations_by_label.setdefault(label, []).append(region.locationId)
    return {
        label: tuple(location_ids)
        for label, location_ids in locations_by_label.items()
        if len(set(location_ids)) > 1
    }


Matrix4 = list[list[float]]


def _validated_matrix(value: list[list[float]]) -> _FrozenList:
    if len(value) != 4 or any(len(row) != 4 for row in value):
        raise ValueError("transform must be a 4x4 matrix")
    if any(not math.isfinite(item) for row in value for item in row):
        raise ValueError("transform values must be finite")
    return _FrozenList(_FrozenList(float(item) for item in row) for row in value)


def _matrix_product(left: Matrix4, right: Matrix4) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][column] for k in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _is_identity_matrix(matrix: Matrix4, *, tolerance: float = 1e-7) -> bool:
    return all(
        math.isclose(
            matrix[row][column],
            1.0 if row == column else 0.0,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for row in range(4)
        for column in range(4)
    )


class Calibration(SceneModel):
    mode: Literal["ros_map_identity", "interactive_similarity"]
    TMapFromScene: Matrix4
    TSceneFromMap: Matrix4
    scaleMetersPerUnit: float = Field(gt=0, allow_inf_nan=False)
    residualMeters: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("TMapFromScene", "TSceneFromMap")
    @classmethod
    def _matrix(cls, value):
        return _validated_matrix(value)

    @model_validator(mode="after")
    def _inverse(self):
        forward_product = _matrix_product(self.TMapFromScene, self.TSceneFromMap)
        reverse_product = _matrix_product(self.TSceneFromMap, self.TMapFromScene)
        if not _is_identity_matrix(forward_product) or not _is_identity_matrix(reverse_product):
            raise ValueError("TSceneFromMap must be the inverse of TMapFromScene")
        return self


class SceneMapSummary(SceneModel):
    """Small immutable map facts safe to send across the Agent boundary."""

    schemaVersion: Literal["h-ide.scene-map/1"] = "h-ide.scene-map/1"
    mapVersion: str
    frameId: Literal["scene", "map"]
    verified: bool
    status: Literal["ready", "degraded"]
    resolution: float = Field(gt=0, allow_inf_nan=False)
    groundZ: float = Field(allow_inf_nan=False)
    observedCoverageRatio: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence: Literal["low", "medium", "high"]
    warnings: list[str] = Field(default_factory=list, max_length=10, validate_default=True)

    @field_validator("mapVersion")
    @classmethod
    def _map_version(cls, value: str):
        return _display_label(value, "mapVersion")

    @field_validator("warnings")
    @classmethod
    def _map_warnings(cls, values: list[str]):
        return _FrozenList(_display_label(value, "warning") for value in values)


class TargetMapCheck(SceneModel):
    """Non-authoritative occupancy evidence shown during human review."""

    mapVersion: str
    classification: Literal["observed_free", "occupied", "unknown", "outside"]
    clearanceMeters: float = Field(gt=0, allow_inf_nan=False)
    distanceToObservedObstacleMeters: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    warnings: list[str] = Field(default_factory=list, max_length=10, validate_default=True)

    @field_validator("mapVersion")
    @classmethod
    def _check_map_version(cls, value: str):
        return _display_label(value, "mapVersion")

    @field_validator("warnings")
    @classmethod
    def _check_warnings(cls, values: list[str]):
        return _FrozenList(_display_label(value, "warning") for value in values)


class SceneContext(SceneModel):
    sceneRef: SceneRef
    bounds: Bounds3D
    annotations: list[SemanticRegion] = Field(default_factory=list, validate_default=True)
    mapSummary: SceneMapSummary | None = None
    currentPose: PoseMap | None = None
    selectedPoint: Point3D | None = None

    @field_validator("annotations")
    @classmethod
    def _annotations(cls, values: list[SemanticRegion]):
        location_ids = [value.locationId for value in values]
        if len(location_ids) != len(set(location_ids)):
            raise ValueError("annotation locationId values must be unique")
        return _FrozenList(values)

class RelativeSpatialIntent(SceneModel):
    kind: Literal["relative"] = "relative"
    forwardMeters: float = Field(allow_inf_nan=False)
    lateralMeters: float = Field(allow_inf_nan=False)
    yawDeltaRad: float = Field(allow_inf_nan=False)


class SemanticSpatialIntent(SceneModel):
    kind: Literal["semantic"] = "semantic"
    locationId: str

    @field_validator("locationId")
    @classmethod
    def _location_id(cls, value: str):
        return _display_label(value, "locationId")


class SelectedPointSpatialIntent(SceneModel):
    kind: Literal["selected_point"] = "selected_point"
    pointMap: Point3D
    yaw: float | None = Field(default=None, allow_inf_nan=False)


SpatialIntent = Annotated[
    RelativeSpatialIntent | SemanticSpatialIntent | SelectedPointSpatialIntent,
    Field(discriminator="kind"),
]


class SpatialResolution(SceneModel):
    resolutionId: str
    sceneRef: SceneRef
    intent: SpatialIntent
    sourcePose: PoseMap | None = None
    goalPose: PoseMap
    resolverVersion: str
    warnings: list[str] = Field(default_factory=list, validate_default=True)
    computedAt: datetime

    @field_validator("resolutionId", "resolverVersion")
    @classmethod
    def _identifiers(cls, value: str, info):
        return _display_label(value, info.field_name)

    @field_validator("warnings")
    @classmethod
    def _warnings(cls, values: list[str]):
        return _FrozenList(_display_label(value, "warning") for value in values)

    @field_validator("computedAt")
    @classmethod
    def _computed_at(cls, value: datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("computedAt must be timezone-aware")
        return value


__all__ = [
    "Bounds3D",
    "Calibration",
    "Matrix4",
    "Point2D",
    "Point3D",
    "PoseMap",
    "SceneRef",
    "SceneContext",
    "SceneMapSummary",
    "SelectedPointSpatialIntent",
    "SemanticSpatialIntent",
    "SemanticRegion",
    "SpatialIntent",
    "SpatialResolution",
    "TargetMapCheck",
    "RelativeSpatialIntent",
    "normalize_semantic_label",
    "normalized_semantic_labels",
    "semantic_label_collisions",
]
