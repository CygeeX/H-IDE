from __future__ import annotations

import numpy as np
import pytest

from h_ide_vision.contracts import ApproachPolicy, CameraIntrinsics
from h_ide_vision.geometry import deproject_pixel, derive_approach_pose, estimate_landmark, transform_point


def test_deprojection_and_transform() -> None:
    intrinsics = CameraIntrinsics(width=640, height=480, fx=600, fy=600, cx=320, cy=240)
    point = deproject_pixel(320, 240, 2.0, intrinsics)
    assert point == pytest.approx((0, 0, 2))
    transform = np.eye(4)
    transform[:3, 3] = (1, 2, 0)
    assert transform_point(point, transform) == pytest.approx((1, 2, 2))


def test_mask_depth_rejects_background_outlier() -> None:
    intrinsics = CameraIntrinsics(width=8, height=6, fx=8, fy=8, cx=3.5, cy=2.5)
    depth = np.zeros((6, 8), dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=bool)
    mask[1:5, 2:6] = True
    depth[mask] = 2000
    depth[2, 3] = 9000
    estimate = estimate_landmark(depth, mask, intrinsics, 0.001)
    assert estimate.depth_median_m == pytest.approx(2.0)
    assert estimate.point_camera[2] == pytest.approx(2.0)
    assert estimate.sample_count == 15


def test_approach_pose_stops_short_and_faces_target() -> None:
    pose = derive_approach_pose(np.asarray([2.0, 0, 0]), np.asarray([0, 0, 0]), ApproachPolicy())
    assert pose.x == pytest.approx(1.25)
    assert pose.y == pytest.approx(0)
    assert pose.yaw == pytest.approx(0)

