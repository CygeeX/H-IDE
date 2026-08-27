from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from h_ide_vision.contracts import CameraIntrinsics


@dataclass(frozen=True)
class ObservationIdentity:
    robot_id: str
    robot_boot_id: str
    gateway_instance_id: str
    sensor_id: str
    observation_session_id: str
    localization_epoch_id: str
    source_mode: str
    source_id: str
    scene_version: str | None


@dataclass(frozen=True)
class ObservationBundle:
    identity: ObservationIdentity
    sequence: int
    captured_at: datetime
    rgb_frame_number: int
    depth_frame_number: int
    rgb_sensor_timestamp_ms: float
    depth_sensor_timestamp_ms: float
    clock_uncertainty_ms: float
    rgb_jpeg: bytes
    rgb_crc32c: int
    depth_z16: np.ndarray
    depth_crc32c: int
    depth_scale: float
    intrinsics: CameraIntrinsics
    calibration_sha256: str
    t_map_camera: np.ndarray
    t_base_camera: np.ndarray
    localization_age_ms: float
    localization_covariance: tuple[float, ...]


@dataclass(frozen=True)
class AcceptedObservation:
    bundle: ObservationBundle
    accepted_at: datetime
    bundle_id: str

