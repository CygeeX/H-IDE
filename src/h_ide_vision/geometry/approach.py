from __future__ import annotations

import math

import numpy as np

from h_ide_vision.contracts import ApproachPolicy, Pose3D


def derive_approach_pose(
    landmark_map: np.ndarray,
    robot_map: np.ndarray,
    policy: ApproachPolicy,
) -> Pose3D:
    landmark = np.asarray(landmark_map, dtype=np.float64)
    robot = np.asarray(robot_map, dtype=np.float64)
    if landmark.shape != (3,) or robot.shape != (3,):
        raise ValueError("landmark and robot positions must be 3-vectors")
    delta = landmark[:2] - robot[:2]
    distance = float(np.linalg.norm(delta))
    if not math.isfinite(distance) or distance <= policy.targetStandOffM:
        raise ValueError("target is too close to derive a safe approach pose")
    if distance > policy.maximumGoalDistanceM + policy.targetStandOffM:
        raise ValueError("target exceeds maximum goal distance")
    direction = delta / distance
    goal_xy = landmark[:2] - direction * policy.targetStandOffM
    yaw = math.atan2(float(delta[1]), float(delta[0]))
    return Pose3D(frameId="map", x=float(goal_xy[0]), y=float(goal_xy[1]), z=0.0, yaw=yaw)

