from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from h_ide_vision.contracts import CameraIntrinsics

from .deprojection import deproject_pixel


@dataclass(frozen=True)
class LandmarkEstimate:
    point_camera: np.ndarray
    valid_depth_ratio: float
    depth_median_m: float
    depth_mad_m: float
    sample_count: int


def estimate_landmark(
    depth_z16: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float,
) -> LandmarkEstimate:
    depth = np.asarray(depth_z16)
    selected = np.asarray(mask, dtype=bool)
    if depth.shape != (intrinsics.height, intrinsics.width) or selected.shape != depth.shape:
        raise ValueError("depth/mask dimensions do not match intrinsics")
    mask_count = int(selected.sum())
    if mask_count == 0:
        raise ValueError("mask is empty")
    values = depth[selected].astype(np.float64) * depth_scale
    valid = np.isfinite(values) & (values > 0.10) & (values < 20.0)
    valid_ratio = float(valid.sum() / mask_count)
    valid_values = values[valid]
    if valid_values.size == 0:
        raise ValueError("mask has no valid depth")
    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - median)))
    tolerance = max(0.05, 3.5 * 1.4826 * mad)
    inliers = valid & (np.abs(values - median) <= tolerance)
    selected_pixels = np.argwhere(selected)
    inlier_pixels = selected_pixels[inliers]
    inlier_depths = values[inliers]
    if inlier_depths.size < 3:
        raise ValueError("insufficient robust depth samples")
    points = np.stack(
        [
            deproject_pixel(float(col), float(row), float(value), intrinsics)
            for (row, col), value in zip(inlier_pixels, inlier_depths, strict=True)
        ]
    )
    point = np.median(points, axis=0)
    return LandmarkEstimate(
        point_camera=point,
        valid_depth_ratio=valid_ratio,
        depth_median_m=median,
        depth_mad_m=mad,
        sample_count=int(inlier_depths.size),
    )

