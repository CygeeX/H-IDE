"""Canonical calibration hashing shared by session and frame messages."""

from __future__ import annotations

import hashlib
import json

from robot_observation_gateway.model import CameraIntrinsics


def calibration_sha256(
    *,
    sensor_id: str,
    device_serial: str,
    device_model: str,
    firmware_version: str,
    color_profile: str,
    depth_profile: str,
    rgb_intrinsics: CameraIntrinsics,
    aligned_depth_intrinsics: CameraIntrinsics,
    depth_scale_meters_per_unit: float,
) -> str:
    payload = {
        "aligned_depth_intrinsics": _intrinsics_dict(aligned_depth_intrinsics),
        "color_profile": color_profile,
        "depth_profile": depth_profile,
        "depth_scale_meters_per_unit": depth_scale_meters_per_unit,
        "device_model": device_model,
        "device_serial": device_serial,
        "firmware_version": firmware_version,
        "rgb_intrinsics": _intrinsics_dict(rgb_intrinsics),
        "sensor_id": sensor_id,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _intrinsics_dict(value: CameraIntrinsics) -> dict[str, object]:
    return {
        "binning_x": value.binning_x,
        "binning_y": value.binning_y,
        "cx": value.cx,
        "cy": value.cy,
        "distortion_coefficients": list(value.distortion_coefficients),
        "distortion_model": value.distortion_model,
        "fx": value.fx,
        "fy": value.fy,
        "height": value.height,
        "roi": {
            "do_rectify": value.roi.do_rectify,
            "height": value.roi.height,
            "width": value.roi.width,
            "x_offset": value.roi.x_offset,
            "y_offset": value.roi.y_offset,
        },
        "width": value.width,
    }
