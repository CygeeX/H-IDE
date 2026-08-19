from __future__ import annotations

import time

import google_crc32c
import pytest
import zstandard

from robot_observation_gateway.calibration import calibration_sha256
from robot_observation_gateway.encoder import ObservationEncoder
from robot_observation_gateway.errors import FrameTooLargeError, SessionProtocolError
from robot_observation_gateway.latest_slot import AsyncLatestSlot, LatestSlot
from robot_observation_gateway.model import (
    Calibration,
    CameraIntrinsics,
    LocalizationEstimate,
    Quaternion,
    RawObservation,
    RigidTransform,
    SensorMetadata,
    TimePoint,
    Vector3,
)
from robot_observation_gateway.session import GatewaySessionController, GatewaySessionState


def intrinsics(width: int = 4, height: int = 3) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=300.0,
        fy=301.0,
        cx=width / 2,
        cy=height / 2,
        distortion_model="plumb_bob",
        distortion_coefficients=(0.0,) * 5,
    )


def raw_observation() -> RawObservation:
    camera = intrinsics()
    digest = calibration_sha256(
        sensor_id="head_rgbd",
        device_serial="serial",
        device_model="D435i",
        firmware_version="5.15",
        color_profile="4x3x30",
        depth_profile="4x3x30",
        rgb_intrinsics=camera,
        aligned_depth_intrinsics=camera,
        depth_scale_meters_per_unit=0.001,
    )
    calibration = Calibration(
        sensor_id="head_rgbd",
        device_serial="serial",
        device_model="D435i",
        firmware_version="5.15",
        color_profile="4x3x30",
        depth_profile="4x3x30",
        rgb_intrinsics=camera,
        aligned_depth_intrinsics=camera,
        depth_scale_meters_per_unit=0.001,
        sha256=digest,
    )
    timestamp = TimePoint(sec=100, nanosec=20)
    metadata = SensorMetadata(
        frame_number=7,
        sensor_timestamp_ms=123.5,
        clock_domain="global_time",
        header_stamp=timestamp,
    )
    transform = RigidTransform(
        parent_frame_id="map",
        child_frame_id="camera_color_optical_frame",
        translation=Vector3(1.0, 2.0, 3.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )
    localization = LocalizationEstimate(
        observed_at=timestamp,
        frame_id="map",
        covariance=tuple(float(index) for index in range(36)),
        age_ms=0.0,
    )
    return RawObservation(
        capture_time_ros=timestamp,
        received_monotonic_ns=time.monotonic_ns(),
        rgb_metadata=metadata,
        depth_metadata=metadata,
        rgb=bytes(range(36)),
        rgb_encoding="rgb8",
        aligned_depth=b"\x01\x00" * 12,
        depth_big_endian=False,
        intrinsics=camera,
        calibration=calibration,
        t_map_camera=transform,
        t_base_camera=RigidTransform(
            parent_frame_id="base_link",
            child_frame_id="camera_color_optical_frame",
            translation=Vector3(0.1, 0.2, 0.3),
            rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        ),
        localization=localization,
    )


def test_latest_slot_overwrites_without_growing() -> None:
    slot: LatestSlot[int] = LatestSlot()
    assert slot.put(1) is False
    assert slot.put(2) is True
    assert slot.qsize == 1
    assert slot.overwritten_count == 1
    assert slot.pop() == 2
    assert slot.qsize == 0


async def test_async_latest_slot_overwrites_and_closes() -> None:
    slot: AsyncLatestSlot[str] = AsyncLatestSlot()
    assert await slot.put("old") is False
    assert await slot.put("new") is True
    assert slot.qsize == 1
    assert await slot.get() == "new"
    await slot.close()
    with pytest.raises(EOFError):
        await slot.get()


def test_encoder_produces_independent_jpeg_and_lossless_z16() -> None:
    raw = raw_observation()
    encoder = ObservationEncoder(jpeg_quality=85, zstd_level=3, max_frame_bytes=4096)

    encoded = encoder.encode(raw)

    assert encoded.rgb_jpeg.startswith(b"\xff\xd8")
    assert zstandard.ZstdDecompressor().decompress(encoded.depth_zstd) == raw.aligned_depth
    assert encoded.rgb_crc32c == google_crc32c.value(encoded.rgb_jpeg)
    assert encoded.depth_crc32c == google_crc32c.value(raw.aligned_depth)
    assert encoded.encode_ms >= 0.0


def test_encoder_rejects_oversize_frame() -> None:
    encoder = ObservationEncoder(jpeg_quality=85, zstd_level=3, max_frame_bytes=8)

    with pytest.raises(FrameTooLargeError, match="limit is 8"):
        encoder.encode(raw_observation())


def test_session_enforces_single_stream_monotonic_sequence_and_accepted_ack() -> None:
    session = GatewaySessionController(minimum_rate_hz=1.0, maximum_rate_hz=5.0)
    session.connected()
    session.welcome(server_instance_id="server-1")
    active = session.start(
        request_id="request-1",
        requester_id="operator",
        purpose_code=1,
        trace_id="trace-1",
        sensor_id="head_rgbd",
        scene_version="scene-1",
        localization_epoch_id="epoch-1",
        requested_rate_hz=20.0,
    )

    assert active.rate_hz == 5.0
    assert session.allocate_sequence() == 0
    assert session.allocate_sequence() == 1
    session.mark_sent(0)
    assert session.acknowledge(
        observation_session_id=active.observation_session_id,
        sequence=0,
        accepted=False,
        monotonic_now=10.0,
    ) is False
    assert session.last_accepted_monotonic is None
    assert session.acknowledge(
        observation_session_id=active.observation_session_id,
        sequence=0,
        accepted=True,
        monotonic_now=11.0,
    ) is True
    assert session.last_accepted_sequence == 0

    with pytest.raises(SessionProtocolError, match="only one"):
        session.start(
            request_id="request-2",
            requester_id="operator",
            purpose_code=1,
            trace_id="trace-2",
            sensor_id="head_rgbd",
            scene_version="scene-1",
            localization_epoch_id="epoch-1",
            requested_rate_hz=5.0,
        )

    stopped = session.stop(active.observation_session_id)
    assert stopped == active
    assert session.state is GatewaySessionState.IDLE
    assert session.last_sent_sequence is None
    assert session.last_accepted_sequence is None
    assert session.last_accepted_monotonic is None
    session.disconnect()
    assert session.state is GatewaySessionState.DISCONNECTED
