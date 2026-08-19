from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.scene_core import Bounds3D, Calibration, Point3D, SceneMapSummary, SemanticRegion


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class UploadCreate(ApiModel):
    file_name: str = Field(alias="fileName", min_length=1, max_length=255)
    size_bytes: int = Field(alias="sizeBytes", gt=0)
    last_modified: int | float | str | None = Field(default=None, alias="lastModified")


class ClaimRequest(ApiModel):
    worker_id: str = Field(alias="workerId", min_length=1, max_length=128)
    lease_seconds: int = Field(alias="leaseSeconds", ge=1, le=3600)


class WorkerRequest(ClaimRequest):
    pass


class PotreeArtifactManifest(ApiModel):
    provider: Literal["potree"]
    files: dict[str, str]
    source_bounds: Bounds3D = Field(alias="sourceBounds")

    @field_validator("files")
    @classmethod
    def _files(cls, value: dict[str, str]) -> dict[str, str]:
        required = {"metadata.json", "hierarchy.bin", "octree.bin"}
        if set(value) != required:
            raise ValueError("Potree manifest must contain metadata.json, hierarchy.bin and octree.bin")
        for storage_key in value.values():
            path = PurePosixPath(storage_key)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "artifacts":
                raise ValueError("artifact storage keys must stay under artifacts")
        return value


IMPORT_PUBLIC_FILES = {
    "metadata.json",
    "hierarchy.bin",
    "octree.bin",
    "scene-summary.json",
    "preview/map.json",
    "preview/occupancy.png",
    "preview/height.png",
}
IMPORT_INTERNAL_FILES = {"analysis.ply", "preview/occupancy.bin"}
MAP_PUBLIC_FILES = {"map/map.json", "map/occupancy.png", "map/height.png"}
MAP_INTERNAL_FILES = {"map/occupancy.bin"}


def _validate_storage_keys(value: dict[str, str]) -> dict[str, str]:
    for logical_name, storage_key in value.items():
        logical = PurePosixPath(logical_name)
        path = PurePosixPath(storage_key)
        if logical.is_absolute() or ".." in logical.parts or not logical.parts:
            raise ValueError("artifact logical names must be safe relative paths")
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "artifacts":
            raise ValueError("artifact storage keys must stay under artifacts")
    return value


class SceneArtifactManifest(ApiModel):
    schema_version: Literal["h-ide.scene-artifacts/2"] = Field(alias="schemaVersion")
    provider: Literal["h-ide-scene", "h-ide-map"]
    job_kind: Literal["import", "map_derivation"] = Field(alias="jobKind")
    public_files: dict[str, str] = Field(alias="publicFiles")
    internal_files: dict[str, str] = Field(alias="internalFiles")
    source_bounds: Bounds3D = Field(alias="sourceBounds")
    processing_summary: dict[str, Any] | None = Field(default=None, alias="processingSummary")
    map_summary: SceneMapSummary = Field(alias="mapSummary")
    input_hash: str | None = Field(default=None, alias="inputHash", min_length=1, max_length=128)

    @field_validator("public_files", "internal_files")
    @classmethod
    def _storage_keys(cls, value: dict[str, str]) -> dict[str, str]:
        return _validate_storage_keys(value)

    @model_validator(mode="after")
    def _files_for_kind(self):
        if self.job_kind == "import":
            if self.provider != "h-ide-scene":
                raise ValueError("import manifest must use h-ide-scene provider")
            if set(self.public_files) != IMPORT_PUBLIC_FILES or set(self.internal_files) != IMPORT_INTERNAL_FILES:
                raise ValueError("import manifest contains an unexpected artifact set")
            if self.processing_summary is None or self.input_hash is not None:
                raise ValueError("import manifest processing summary is required")
            if self.map_summary.frameId != "scene" or self.map_summary.verified:
                raise ValueError("import manifest must contain an unverified scene-frame preview")
        else:
            if self.provider != "h-ide-map":
                raise ValueError("map manifest must use h-ide-map provider")
            if set(self.public_files) != MAP_PUBLIC_FILES or set(self.internal_files) != MAP_INTERNAL_FILES:
                raise ValueError("map manifest contains an unexpected artifact set")
            if self.input_hash is None or self.processing_summary is not None:
                raise ValueError("map manifest input hash is required")
            if self.map_summary.frameId != "map" or not self.map_summary.verified:
                raise ValueError("map manifest must contain a verified map-frame summary")
        return self


class CompleteJobRequest(ApiModel):
    worker_id: str = Field(alias="workerId", min_length=1, max_length=128)
    artifact_manifest: PotreeArtifactManifest | SceneArtifactManifest = Field(alias="artifactManifest")


class FailJobRequest(ApiModel):
    worker_id: str = Field(alias="workerId", min_length=1, max_length=128)
    deterministic: bool
    error_code: str = Field(alias="errorCode", pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    message: str = Field(max_length=1000)


class RevisionCreate(ApiModel):
    base_revision: int | None = Field(default=None, alias="baseRevision", ge=1)


class CalibrationUpdate(ApiModel):
    calibration: Calibration


class InteractiveCalibrationUpdate(ApiModel):
    ground_points: list[Point3D] = Field(alias="groundPoints", min_length=3)
    origin_scene: Point3D = Field(alias="originScene")
    x_axis_scene: Point3D = Field(alias="xAxisScene")
    known_distance_m: float = Field(alias="knownDistanceM", gt=0, allow_inf_nan=False)
    map_origin: Point3D = Field(alias="mapOrigin")


class AnnotationUpdate(ApiModel):
    annotations: list[SemanticRegion]


class SceneBindingUpdate(ApiModel):
    scene_id: str = Field(alias="sceneId", min_length=1)
    revision: int = Field(ge=1)


class TargetCheckRequest(ApiModel):
    point_map: Point3D = Field(alias="pointMap")
