"""Deterministic CPU scene analysis and single-floor raster generation.

Open3D is intentionally isolated in the worker image.  The Scene API and
Control API only exchange validated JSON summaries and never import this
module in production.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, Protocol

try:  # Kept optional outside the isolated Scene Worker image.
    import numpy as np
    from PIL import Image
except ImportError:  # pragma: no cover - unit tests inject a lightweight processor
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

from packages.scene_core import Bounds3D, Point3D, SceneMapSummary


MAP_SCHEMA_VERSION = "h-ide.scene-map/1"
MAX_ANALYSIS_POINTS = 2_000_000
MAX_VISUAL_POINTS = 750_000
MAX_GRID_SIDE = 2048
FORMAL_MAP_MIN_RESOLUTION_METERS = 0.05
OBSTACLE_MIN_HEIGHT_METERS = 0.10
OBSTACLE_MAX_HEIGHT_METERS = 1.80
GROUND_TOLERANCE_METERS = 0.08


@dataclass(frozen=True)
class ProcessingProfile:
    source_points: int
    analysis_input_points: int
    finite_points: int
    decimation_step: float
    median_spacing: float
    voxel_size: float
    bounds: Bounds3D
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "sourcePointCount": self.source_points,
            "analysisInputPointCount": self.analysis_input_points,
            "finitePointCount": self.finite_points,
            "decimationStep": self.decimation_step,
            "medianSpacing": self.median_spacing,
            "voxelSize": self.voxel_size,
            "sourceBounds": self.bounds.model_dump(mode="json"),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MapBuildResult:
    summary: SceneMapSummary
    metadata: dict[str, Any]


class SceneProcessor(Protocol):
    def analyze(self, sample_ply: Path, *, source_points: int, decimation_step: float) -> ProcessingProfile: ...

    def build_map(
        self,
        analysis_ply: Path,
        output_directory: Path,
        *,
        map_version: str,
        frame_id: Literal["scene", "map"],
        verified: bool,
        voxel_size: float,
        transform: list[list[float]] | None = None,
    ) -> MapBuildResult: ...


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:  # pragma: no cover - exercised in the deployable worker image
        raise RuntimeError("Open3D CPU runtime is unavailable") from error
    return o3d


def _require_scientific_runtime() -> None:
    if np is None or Image is None:
        raise RuntimeError("NumPy and Pillow scene runtimes are unavailable")


def _bounds(points: np.ndarray) -> Bounds3D:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return Bounds3D(
        min=Point3D(x=float(minimum[0]), y=float(minimum[1]), z=float(minimum[2])),
        max=Point3D(x=float(maximum[0]), y=float(maximum[1]), z=float(maximum[2])),
    )


def _finite_points(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("point cloud must contain XYZ coordinates")
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("point cloud has no finite XYZ points")
    return finite


class Open3DSceneProcessor:
    """Open3D-backed geometry analysis with a compact browser raster output."""

    def analyze(self, sample_ply: Path, *, source_points: int, decimation_step: float) -> ProcessingProfile:
        _require_scientific_runtime()
        o3d = _open3d()
        cloud = o3d.io.read_point_cloud(str(sample_ply), remove_nan_points=True, remove_infinite_points=True)
        points = _finite_points(np.asarray(cloud.points, dtype=np.float64))
        finite_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        bounds = _bounds(points)
        spans = points.max(axis=0) - points.min(axis=0)
        longest_span = max(float(spans.max()), 1e-6)

        # A deterministic stride keeps nearest-neighbour estimation bounded.
        spacing_points = points[:: max(1, math.ceil(len(points) / 100_000))]
        spacing_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(spacing_points))
        distances = np.asarray(spacing_cloud.compute_nearest_neighbor_distance(), dtype=np.float64)
        positive = distances[np.isfinite(distances) & (distances > 0)]
        median_spacing = float(np.median(positive)) if len(positive) else longest_span / 4096
        base_voxel = max(1.5 * median_spacing, longest_span / 4096, 1e-6)
        voxel_size = base_voxel

        if len(points) > MAX_VISUAL_POINTS:
            low, high = base_voxel, max(base_voxel * 2, longest_span)
            for _ in range(18):
                candidate = (low + high) / 2
                count = len(finite_cloud.voxel_down_sample(candidate).points)
                if count > MAX_VISUAL_POINTS:
                    low = candidate
                else:
                    high = candidate
            voxel_size = high

        warnings: list[str] = []
        if decimation_step > 1:
            warnings.append("输入点数较大，场景分析使用确定性抽样；原始 PCD 已完整保留")
        return ProcessingProfile(
            source_points=source_points,
            analysis_input_points=len(points),
            finite_points=len(points),
            decimation_step=decimation_step,
            median_spacing=median_spacing,
            voxel_size=voxel_size,
            bounds=bounds,
            warnings=tuple(warnings),
        )

    def build_map(
        self,
        analysis_ply: Path,
        output_directory: Path,
        *,
        map_version: str,
        frame_id: Literal["scene", "map"],
        verified: bool,
        voxel_size: float,
        transform: list[list[float]] | None = None,
    ) -> MapBuildResult:
        _require_scientific_runtime()
        o3d = _open3d()
        cloud = o3d.io.read_point_cloud(str(analysis_ply), remove_nan_points=True, remove_infinite_points=True)
        points = _finite_points(np.asarray(cloud.points, dtype=np.float64))
        if transform is not None:
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("map transform must be a finite 4x4 matrix")
            homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
            transformed = homogeneous @ matrix.T
            if np.any(np.abs(transformed[:, 3] - 1.0) > 1e-6):
                raise ValueError("map transform must be affine")
            points = transformed[:, :3]

        spans = points.max(axis=0) - points.min(axis=0)
        longest_xy = max(float(spans[0]), float(spans[1]), 1e-6)
        grid_span = max(1, MAX_GRID_SIDE - 2)
        resolution = (
            max(FORMAL_MAP_MIN_RESOLUTION_METERS, longest_xy / grid_span)
            if verified
            else max(2 * voxel_size, longest_xy / grid_span, 1e-6)
        )
        ground_z, confidence, plane_warning = self._ground_plane(points, resolution, verified)
        warnings = [] if plane_warning is None else [plane_warning]
        if not verified:
            warnings.insert(0, "预览假设源坐标 XY 为水平面，完成标定后才可用于 map 坐标")

        result = self._rasterize(
            points,
            output_directory,
            map_version=map_version,
            frame_id=frame_id,
            verified=verified,
            voxel_size=voxel_size,
            resolution=resolution,
            ground_z=ground_z,
            confidence=confidence,
            warnings=warnings,
        )
        return result

    @staticmethod
    def _ground_plane(
        points: np.ndarray,
        resolution: float,
        verified: bool,
    ) -> tuple[float, Literal["low", "medium", "high"], str | None]:
        o3d = _open3d()
        z_values = points[:, 2]
        lower = float(np.quantile(z_values, 0.05))
        candidate_ceiling = float(np.quantile(z_values, 0.30))
        candidates = points[z_values <= candidate_ceiling]
        if len(candidates) < 50:
            return lower, "low", "地面样本不足，使用低分位高度生成降级地图"
        candidate_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(candidates))
        try:
            o3d.utility.random.seed(0)
        except AttributeError:  # pragma: no cover - older compatible Open3D builds
            pass
        threshold = max(resolution / 2, 0.02 if verified else resolution / 3)
        try:
            plane, inliers = candidate_cloud.segment_plane(
                distance_threshold=threshold,
                ransac_n=3,
                num_iterations=1000,
            )
        except RuntimeError:
            return lower, "low", "主地面拟合失败，使用低分位高度生成降级地图"
        normal = np.asarray(plane[:3], dtype=np.float64)
        normal_length = float(np.linalg.norm(normal))
        inlier_ratio = len(inliers) / max(1, len(candidates))
        vertical_alignment = abs(float(normal[2])) / normal_length if normal_length > 0 else 0
        if vertical_alignment < math.cos(math.radians(15)):
            return lower, "low", "检测到的主平面不是水平地面，使用低分位高度生成降级地图"
        ground_z = float(np.median(candidates[np.asarray(inliers, dtype=np.int64), 2]))
        if inlier_ratio >= 0.25:
            return ground_z, "high", None
        if inlier_ratio >= 0.08:
            return ground_z, "medium", None
        return ground_z, "low", "地面拟合置信度较低，目标校验仅作为人工审核提示"

    @staticmethod
    def _rasterize(
        points: np.ndarray,
        output_directory: Path,
        *,
        map_version: str,
        frame_id: Literal["scene", "map"],
        verified: bool,
        voxel_size: float,
        resolution: float,
        ground_z: float,
        confidence: Literal["low", "medium", "high"],
        warnings: list[str],
    ) -> MapBuildResult:
        output_directory.mkdir(parents=True, exist_ok=True)
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        origin_x = math.floor(float(minimum[0]) / resolution) * resolution
        origin_y = math.floor(float(minimum[1]) / resolution) * resolution
        width = max(1, min(MAX_GRID_SIDE, math.ceil((float(maximum[0]) - origin_x) / resolution) + 1))
        height = max(1, min(MAX_GRID_SIDE, math.ceil((float(maximum[1]) - origin_y) / resolution) + 1))

        columns = np.floor((points[:, 0] - origin_x) / resolution).astype(np.int64)
        rows = np.floor((points[:, 1] - origin_y) / resolution).astype(np.int64)
        inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        columns, rows, z_values = columns[inside], rows[inside], points[inside, 2]

        obstacle_min = OBSTACLE_MIN_HEIGHT_METERS if verified else max(2 * voxel_size, resolution)
        obstacle_max = OBSTACLE_MAX_HEIGHT_METERS if verified else max(obstacle_min + resolution, 36 * resolution)
        ground_tolerance = GROUND_TOLERANCE_METERS if verified else max(voxel_size * 1.5, resolution)
        occupied = np.zeros((height, width), dtype=np.bool_)
        observed_ground = np.zeros((height, width), dtype=np.bool_)
        height_max = np.full((height, width), -np.inf, dtype=np.float32)

        relative_z = z_values - ground_z
        obstacle_mask = (relative_z >= obstacle_min) & (relative_z <= obstacle_max)
        ground_mask = np.abs(relative_z) <= ground_tolerance
        occupied[rows[obstacle_mask], columns[obstacle_mask]] = True
        observed_ground[rows[ground_mask], columns[ground_mask]] = True
        np.maximum.at(height_max, (rows, columns), z_values.astype(np.float32))

        # A ground return represents a small surface patch, not a mathematical
        # point. Fill only its immediate sampling footprint so the browser map
        # reads as floor regions while unknown space remains explicit.
        ground_inflation_cells = min(2, max(1, math.ceil(voxel_size / resolution / 2)))
        padded = np.pad(observed_ground, ground_inflation_cells, mode="constant")
        expanded_ground = np.zeros_like(observed_ground)
        for row_offset in range(2 * ground_inflation_cells + 1):
            for column_offset in range(2 * ground_inflation_cells + 1):
                expanded_ground |= padded[
                    row_offset:row_offset + height,
                    column_offset:column_offset + width,
                ]
        observed_ground = expanded_ground

        classes = np.zeros((height, width), dtype=np.uint8)
        classes[observed_ground & ~occupied] = 1
        classes[occupied] = 2
        coverage = float(np.count_nonzero(classes)) / float(classes.size)

        # Binary rows increase with map Y. PNG rows point down, so image outputs are flipped.
        (output_directory / "occupancy.bin").write_bytes(classes.tobytes(order="C"))
        palette = np.asarray(
            [[14, 23, 35, 255], [214, 226, 229, 255], [47, 67, 77, 255]],
            dtype=np.uint8,
        )
        Image.fromarray(np.flipud(palette[classes]), mode="RGBA").save(output_directory / "occupancy.png")

        height_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        valid_height = np.isfinite(height_max)
        normalized = np.clip((height_max - ground_z) / max(obstacle_max, resolution), 0, 1)
        height_rgba[..., 0] = np.where(valid_height, 30 + normalized * 170, 14).astype(np.uint8)
        height_rgba[..., 1] = np.where(valid_height, 190 - normalized * 100, 23).astype(np.uint8)
        height_rgba[..., 2] = np.where(valid_height, 220 - normalized * 80, 35).astype(np.uint8)
        height_rgba[..., 3] = 255
        Image.fromarray(np.flipud(height_rgba), mode="RGBA").save(output_directory / "height.png")

        status: Literal["ready", "degraded"] = "ready" if verified and confidence != "low" else "degraded"
        summary = SceneMapSummary(
            mapVersion=map_version,
            frameId=frame_id,
            verified=verified,
            status=status,
            resolution=resolution,
            groundZ=ground_z,
            observedCoverageRatio=coverage,
            confidence=confidence,
            warnings=warnings,
        )
        metadata: dict[str, Any] = {
            "schemaVersion": MAP_SCHEMA_VERSION,
            "mapVersion": map_version,
            "frameId": frame_id,
            "verified": verified,
            "status": status,
            "resolution": resolution,
            "origin": {"x": origin_x, "y": origin_y},
            "width": width,
            "height": height,
            "binaryRowOrder": "positive_y",
            "imageRowOrder": "negative_y",
            "classes": {"unknown": 0, "observedFree": 1, "occupied": 2},
            "groundZ": ground_z,
            "obstacleHeightRange": {"min": obstacle_min, "max": obstacle_max},
            "voxelSize": voxel_size,
            "groundEvidenceInflationCells": ground_inflation_cells,
            "bounds": _bounds(points).model_dump(mode="json"),
            "summary": summary.model_dump(mode="json"),
        }
        (output_directory / "map.json").write_text(
            json.dumps(metadata, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        return MapBuildResult(summary=summary, metadata=metadata)


__all__ = [
    "FORMAL_MAP_MIN_RESOLUTION_METERS",
    "MAP_SCHEMA_VERSION",
    "MAX_ANALYSIS_POINTS",
    "MAX_VISUAL_POINTS",
    "MapBuildResult",
    "Open3DSceneProcessor",
    "ProcessingProfile",
    "SceneProcessor",
]
