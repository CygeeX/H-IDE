from __future__ import annotations

import math

import numpy as np

from h_ide_vision.contracts import CameraIntrinsics


def deproject_pixel(u: float, v: float, depth_m: float, intrinsics: CameraIntrinsics) -> np.ndarray:
    if not math.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth must be finite and positive")
    if not (0 <= u < intrinsics.width and 0 <= v < intrinsics.height):
        raise ValueError("pixel is outside image")
    return np.asarray(
        [
            (u - intrinsics.cx) * depth_m / intrinsics.fx,
            (v - intrinsics.cy) * depth_m / intrinsics.fy,
            depth_m,
        ],
        dtype=np.float64,
    )


def transform_point(point_xyz: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    point = np.asarray(point_xyz, dtype=np.float64)
    transform = np.asarray(matrix_4x4, dtype=np.float64)
    if point.shape != (3,) or transform.shape != (4, 4):
        raise ValueError("expected point (3,) and transform (4,4)")
    if not np.isfinite(point).all() or not np.isfinite(transform).all():
        raise ValueError("point and transform must be finite")
    homogeneous = np.concatenate((point, np.asarray([1.0])))
    mapped = transform @ homogeneous
    if abs(mapped[3]) < 1e-12:
        raise ValueError("invalid homogeneous transform")
    return np.asarray(mapped[:3] / mapped[3], dtype=np.float64)
