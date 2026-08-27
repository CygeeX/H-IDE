from __future__ import annotations

from datetime import UTC

import numpy as np
import zstandard

from h_ide_vision.contracts import CameraIntrinsics
from h_ide_vision.generated.robot_observation.v1 import observation_pb2 as pb

from .models import ObservationBundle, ObservationIdentity


def _quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = (
        (1 - yy - zz, xy - wz, xz + wy),
        (xy + wz, 1 - xx - zz, yz - wx),
        (xz - wy, yz + wx, 1 - xx - yy),
    )
    return result


def _transform(message: pb.PoseTransform) -> np.ndarray:
    value = _quaternion_matrix(
        message.rotation.x, message.rotation.y, message.rotation.z, message.rotation.w
    )
    value[:3, 3] = (message.translation.x, message.translation.y, message.translation.z)
    return value


def decode_frame(message: pb.ObservationFrame) -> ObservationBundle:
    if message.rgb_encoding != pb.RGB_ENCODING_JPEG:
        raise ValueError("unsupported RGB encoding")
    if message.depth_encoding != pb.DEPTH_ENCODING_ZSTD_Z16_LE:
        raise ValueError("unsupported depth encoding")
    raw_depth = zstandard.ZstdDecompressor().decompress(
        message.aligned_depth, max_output_size=message.uncompressed_depth_bytes
    )
    intrinsics = CameraIntrinsics(
        width=message.intrinsics.width,
        height=message.intrinsics.height,
        fx=message.intrinsics.fx,
        fy=message.intrinsics.fy,
        cx=message.intrinsics.cx,
        cy=message.intrinsics.cy,
    )
    expected = intrinsics.width * intrinsics.height * 2
    if len(raw_depth) != expected:
        raise ValueError("decoded depth byte count is invalid")
    source_mode = pb.SourceMode.Name(message.source_mode).removeprefix("SOURCE_MODE_").lower()
    scene = message.scene_version if message.HasField("scene_version") else None
    return ObservationBundle(
        identity=ObservationIdentity(
            robot_id=message.robot_id,
            robot_boot_id=message.robot_boot_id,
            gateway_instance_id=message.gateway_instance_id,
            sensor_id=message.sensor_id,
            observation_session_id=message.observation_session_id,
            localization_epoch_id=message.localization_epoch_id,
            source_mode=source_mode,
            source_id=message.source_id,
            scene_version=scene,
        ),
        sequence=message.sequence,
        captured_at=message.capture_time.ToDatetime(tzinfo=UTC),
        rgb_frame_number=message.rgb_frame_number,
        depth_frame_number=message.depth_frame_number,
        rgb_sensor_timestamp_ms=message.rgb_sensor_timestamp_ms,
        depth_sensor_timestamp_ms=message.depth_sensor_timestamp_ms,
        clock_uncertainty_ms=message.clock_uncertainty_ms,
        rgb_jpeg=bytes(message.rgb),
        rgb_crc32c=message.rgb_crc32c,
        depth_z16=np.frombuffer(raw_depth, dtype="<u2").reshape((intrinsics.height, intrinsics.width)),
        depth_crc32c=message.depth_crc32c,
        depth_scale=message.depth_scale_meters_per_unit,
        intrinsics=intrinsics,
        calibration_sha256=message.calibration_sha256,
        t_map_camera=_transform(message.t_map_camera),
        t_base_camera=_transform(message.t_base_camera),
        localization_age_ms=message.localization.age_ms,
        localization_covariance=tuple(message.localization.covariance),
    )

