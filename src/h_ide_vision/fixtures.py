from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO

import google_crc32c
import numpy as np
from PIL import Image, ImageDraw

from h_ide_vision.contracts import CameraIntrinsics
from h_ide_vision.ingress import ObservationBundle, ObservationIdentity


def fixture_rgb(width: int = 64, height: int = 48) -> bytes:
    image = Image.new("RGB", (width, height), (22, 28, 36))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 16, 39, 31), fill=(235, 183, 48))
    output = BytesIO()
    image.save(output, format="JPEG", quality=90)
    return output.getvalue()


def fixture_bundle(
    *,
    sequence: int,
    captured_at: datetime | None = None,
    session_id: str = "fixture-session-1",
    localization_epoch_id: str = "lab-map-epoch-1",
    source_mode: str = "synthetic",
) -> ObservationBundle:
    width, height = 64, 48
    rgb = fixture_rgb(width, height)
    depth = np.zeros((height, width), dtype="<u2")
    depth[16:32, 24:40] = 2000
    depth_bytes = depth.tobytes()
    intrinsics = CameraIntrinsics(width=width, height=height, fx=60, fy=60, cx=31.5, cy=23.5)
    # Fixture camera optical +Z points along map +X. Keeping this explicit
    # catches accidental treatment of optical depth as map height.
    transform = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ObservationBundle(
        identity=ObservationIdentity(
            robot_id="fixture-robot",
            robot_boot_id="fixture-boot-1",
            gateway_instance_id="fixture-gateway-1",
            sensor_id="head_rgbd",
            observation_session_id=session_id,
            localization_epoch_id=localization_epoch_id,
            source_mode=source_mode,
            source_id="fixture:known-target-v1",
            scene_version=None,
        ),
        sequence=sequence,
        captured_at=captured_at or datetime.now(UTC),
        rgb_frame_number=sequence,
        depth_frame_number=sequence,
        rgb_sensor_timestamp_ms=float(sequence * 100),
        depth_sensor_timestamp_ms=float(sequence * 100 + 2),
        clock_uncertainty_ms=1,
        rgb_jpeg=rgb,
        rgb_crc32c=int(google_crc32c.value(rgb)),
        depth_z16=depth,
        depth_crc32c=int(google_crc32c.value(depth_bytes)),
        depth_scale=0.001,
        intrinsics=intrinsics,
        calibration_sha256=sha256(b"fixture-calibration-v1").hexdigest(),
        t_map_camera=transform,
        t_base_camera=transform,
        localization_age_ms=5,
        localization_covariance=tuple(0.001 if index % 7 == 0 else 0.0 for index in range(36)),
    )
