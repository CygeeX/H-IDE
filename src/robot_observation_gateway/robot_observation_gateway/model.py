"""Protocol-independent immutable observation models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot_observation_gateway.errors import FrameValidationError


@dataclass(frozen=True)
class TimePoint:
    sec: int
    nanosec: int

    def __post_init__(self) -> None:
        if self.sec < 0 or not 0 <= self.nanosec < 1_000_000_000:
            raise FrameValidationError("invalid non-negative timestamp")

    @property
    def unix_ns(self) -> int:
        return self.sec * 1_000_000_000 + self.nanosec


@dataclass(frozen=True)
class RegionOfInterest:
    x_offset: int = 0
    y_offset: int = 0
    width: int = 0
    height: int = 0
    do_rectify: bool = False


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: tuple[float, ...]
    binning_x: int = 0
    binning_y: int = 0
    roi: RegionOfInterest = RegionOfInterest()

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise FrameValidationError("camera dimensions must be positive")
        if self.fx <= 0 or self.fy <= 0:
            raise FrameValidationError("camera focal lengths must be positive")
        values = (self.fx, self.fy, self.cx, self.cy, *self.distortion_coefficients)
        if not all(math.isfinite(value) for value in values):
            raise FrameValidationError("camera intrinsics must be finite")
        if not self.distortion_model:
            raise FrameValidationError("distortion_model must not be empty")


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.z, self.w)
        if not all(math.isfinite(value) for value in values):
            raise FrameValidationError("quaternion must be finite")
        norm = math.sqrt(sum(value * value for value in values))
        if not 0.99 <= norm <= 1.01:
            raise FrameValidationError("quaternion must be normalized")


@dataclass(frozen=True)
class RigidTransform:
    parent_frame_id: str
    child_frame_id: str
    translation: Vector3
    rotation: Quaternion

    def __post_init__(self) -> None:
        if not self.parent_frame_id or not self.child_frame_id:
            raise FrameValidationError("transform frame IDs must not be empty")
        if not all(
            math.isfinite(value)
            for value in (self.translation.x, self.translation.y, self.translation.z)
        ):
            raise FrameValidationError("transform translation must be finite")


@dataclass(frozen=True)
class LocalizationEstimate:
    observed_at: TimePoint
    frame_id: str
    covariance: tuple[float, ...]
    age_ms: float

    def __post_init__(self) -> None:
        if len(self.covariance) != 36:
            raise FrameValidationError("localization covariance must contain 36 values")
        if not all(math.isfinite(value) for value in self.covariance):
            raise FrameValidationError("localization covariance must be finite")
        if not self.frame_id or not math.isfinite(self.age_ms) or self.age_ms < 0:
            raise FrameValidationError("invalid localization metadata")


@dataclass(frozen=True)
class SensorMetadata:
    frame_number: int
    sensor_timestamp_ms: float
    clock_domain: str
    header_stamp: TimePoint

    def __post_init__(self) -> None:
        if self.frame_number < 0 or not math.isfinite(self.sensor_timestamp_ms):
            raise FrameValidationError("invalid sensor metadata")
        if not self.clock_domain:
            raise FrameValidationError("sensor clock domain must not be empty")


@dataclass(frozen=True)
class Calibration:
    sensor_id: str
    device_serial: str
    device_model: str
    firmware_version: str
    color_profile: str
    depth_profile: str
    rgb_intrinsics: CameraIntrinsics
    aligned_depth_intrinsics: CameraIntrinsics
    depth_scale_meters_per_unit: float
    sha256: str

    def __post_init__(self) -> None:
        if self.depth_scale_meters_per_unit <= 0:
            raise FrameValidationError("depth scale must be positive")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise FrameValidationError("calibration sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True)
class RawObservation:
    capture_time_ros: TimePoint
    received_monotonic_ns: int
    rgb_metadata: SensorMetadata
    depth_metadata: SensorMetadata
    rgb: bytes
    rgb_encoding: str
    aligned_depth: bytes
    depth_big_endian: bool
    intrinsics: CameraIntrinsics
    calibration: Calibration
    t_map_camera: RigidTransform
    t_base_camera: RigidTransform
    localization: LocalizationEstimate

    def __post_init__(self) -> None:
        pixels = self.intrinsics.width * self.intrinsics.height
        channels = 3 if self.rgb_encoding.lower() in {"rgb8", "bgr8"} else 0
        if channels == 0:
            raise FrameValidationError(f"unsupported raw RGB encoding: {self.rgb_encoding}")
        if len(self.rgb) != pixels * channels:
            raise FrameValidationError("raw RGB byte length does not match dimensions")
        if len(self.aligned_depth) != pixels * 2:
            raise FrameValidationError("raw Z16 byte length does not match dimensions")
        if self.received_monotonic_ns < 0:
            raise FrameValidationError("received monotonic timestamp must be non-negative")


@dataclass(frozen=True)
class EncodedObservation:
    raw: RawObservation
    rgb_jpeg: bytes
    depth_zstd: bytes
    rgb_crc32c: int
    depth_crc32c: int
    encode_ms: float
    encoded_monotonic_ns: int

    @property
    def payload_bytes(self) -> int:
        return len(self.rgb_jpeg) + len(self.depth_zstd)
