from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import google_crc32c
import numpy as np

from h_ide_vision.contracts import ApproachPolicy

from .models import ObservationBundle


class ObservationRejected(ValueError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


def _crc(data: bytes) -> int:
    return int(google_crc32c.value(data))


def bundle_id(bundle: ObservationBundle) -> str:
    digest = hashlib.sha256()
    digest.update(bundle.identity.observation_session_id.encode())
    digest.update(bundle.sequence.to_bytes(8, "big"))
    digest.update(bundle.rgb_jpeg)
    digest.update(bundle.depth_z16.astype("<u2", copy=False).tobytes())
    return digest.hexdigest()


def validate_bundle(bundle: ObservationBundle, policy: ApproachPolicy, *, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    identity = bundle.identity
    required = (
        identity.robot_id,
        identity.robot_boot_id,
        identity.gateway_instance_id,
        identity.sensor_id,
        identity.observation_session_id,
        identity.localization_epoch_id,
        identity.source_id,
    )
    if any(not value.strip() for value in required):
        raise ObservationRejected("identity_mismatch", "observation identity is incomplete")
    if identity.source_mode not in {"live", "replay", "synthetic"}:
        raise ObservationRejected("invalid_metadata", "source mode is invalid")
    age_ms = (now - bundle.captured_at.astimezone(UTC)).total_seconds() * 1000
    if age_ms < -policy.maximumClockUncertaintyMs or age_ms > policy.maximumFrameAgeMs:
        raise ObservationRejected("stale", f"capture age {age_ms:.1f} ms")
    if abs(bundle.rgb_sensor_timestamp_ms - bundle.depth_sensor_timestamp_ms) > policy.maximumRgbDepthDeltaMs:
        raise ObservationRejected("invalid_metadata", "RGB and depth are not synchronized")
    if bundle.clock_uncertainty_ms > policy.maximumClockUncertaintyMs:
        raise ObservationRejected("invalid_metadata", "clock uncertainty exceeds policy")
    if _crc(bundle.rgb_jpeg) != bundle.rgb_crc32c:
        raise ObservationRejected("bad_checksum", "RGB CRC32C mismatch")
    depth_bytes = bundle.depth_z16.astype("<u2", copy=False).tobytes()
    if _crc(depth_bytes) != bundle.depth_crc32c:
        raise ObservationRejected("bad_checksum", "depth CRC32C mismatch")
    if bundle.depth_z16.shape != (bundle.intrinsics.height, bundle.intrinsics.width):
        raise ObservationRejected("invalid_metadata", "depth dimensions do not match intrinsics")
    if bundle.depth_scale <= 0:
        raise ObservationRejected("invalid_metadata", "depth scale must be positive")
    if len(bundle.localization_covariance) != 36:
        raise ObservationRejected("invalid_metadata", "localization covariance must have 36 values")
    if bundle.localization_age_ms > policy.maximumFrameAgeMs:
        raise ObservationRejected("localization_mismatch", "localization is stale")
    for transform in (bundle.t_map_camera, bundle.t_base_camera):
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ObservationRejected("invalid_metadata", "transform is invalid")
    if len(bundle.calibration_sha256) != 64:
        raise ObservationRejected("calibration_mismatch", "calibration hash is invalid")
    return bundle_id(bundle)

