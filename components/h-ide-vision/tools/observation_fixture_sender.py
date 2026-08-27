from __future__ import annotations

import argparse
import asyncio
import math
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import grpc
import zstandard
from google.protobuf.timestamp_pb2 import Timestamp

from h_ide_vision.generated.robot_observation.v1 import observation_pb2 as pb
from h_ide_vision.generated.robot_observation.v1 import observation_pb2_grpc as pb_grpc
from tools.observation_fixture import fixture_bundle


def _timestamp(value: datetime) -> Timestamp:
    result = Timestamp()
    result.FromDatetime(value.astimezone(UTC))
    return result


def frame_message(sequence: int, *, session_id: str) -> pb.ObservationFrame:
    bundle = fixture_bundle(sequence=sequence, session_id=session_id)
    depth = bundle.depth_z16.astype("<u2", copy=False).tobytes()
    compressed = zstandard.ZstdCompressor(level=3).compress(depth)
    intrinsics = pb.CameraIntrinsics(
        width=bundle.intrinsics.width,
        height=bundle.intrinsics.height,
        fx=bundle.intrinsics.fx,
        fy=bundle.intrinsics.fy,
        cx=bundle.intrinsics.cx,
        cy=bundle.intrinsics.cy,
    )
    identity = bundle.identity
    optical_to_map = pb.PoseTransform(
        parent_frame_id="map",
        child_frame_id="camera_color_optical_frame",
        translation=pb.Vector3(),
        rotation=pb.Quaternion(y=math.sin(math.pi / 4), w=math.cos(math.pi / 4)),
    )
    return pb.ObservationFrame(
        robot_id=identity.robot_id,
        robot_boot_id=identity.robot_boot_id,
        gateway_instance_id=identity.gateway_instance_id,
        observation_session_id=identity.observation_session_id,
        sequence=sequence,
        sensor_id=identity.sensor_id,
        capture_time=_timestamp(bundle.captured_at),
        rgb_frame_number=sequence,
        depth_frame_number=sequence,
        rgb_sensor_timestamp_ms=bundle.rgb_sensor_timestamp_ms,
        depth_sensor_timestamp_ms=bundle.depth_sensor_timestamp_ms,
        clock_domain=pb.CLOCK_DOMAIN_SYSTEM_TIME,
        rgb_encoding=pb.RGB_ENCODING_JPEG,
        rgb=bundle.rgb_jpeg,
        rgb_crc32c=bundle.rgb_crc32c,
        depth_encoding=pb.DEPTH_ENCODING_ZSTD_Z16_LE,
        aligned_depth=compressed,
        depth_crc32c=bundle.depth_crc32c,
        depth_scale_meters_per_unit=bundle.depth_scale,
        intrinsics=intrinsics,
        calibration_sha256=bundle.calibration_sha256,
        t_map_camera=optical_to_map,
        t_base_camera=optical_to_map,
        localization=pb.LocalizationEstimate(
            observed_at=_timestamp(bundle.captured_at),
            frame_id="map",
            covariance=bundle.localization_covariance,
            age_ms=bundle.localization_age_ms,
        ),
        jpeg_quality=90,
        zstd_level=3,
        uncompressed_rgb_bytes=len(bundle.rgb_jpeg),
        uncompressed_depth_bytes=len(depth),
        localization_epoch_id=identity.localization_epoch_id,
        source_mode=pb.SOURCE_MODE_SYNTHETIC,
        source_id=identity.source_id,
        clock_uncertainty_ms=bundle.clock_uncertainty_ms,
    )


async def messages(
    *,
    frames: int,
    schema_sha256: str,
    session_id: str = "fixture-session-1",
    continuous: bool = False,
    interval_seconds: float = 0.4,
):
    yield pb.GatewayToWorkstation(
        hello=pb.GatewayHello(
            protocol=pb.ProtocolVersion(major=1, minor=0, schema_sha256=schema_sha256),
            robot_id="fixture-robot",
            robot_boot_id="fixture-boot-1",
            gateway_instance_id="fixture-gateway-1",
            gateway_version="fixture-sender-v1",
            features=("aligned_depth", "capture_tf", "localization_epoch"),
            sensors=(
                pb.SensorDescriptor(
                    sensor_id="head_rgbd",
                    source_id="fixture:known-target-v1",
                    device_serial="fixture-only",
                    device_model="synthetic-rgbd",
                    rgb_encodings=(pb.RGB_ENCODING_JPEG,),
                    depth_encodings=(pb.DEPTH_ENCODING_ZSTD_Z16_LE,),
                    max_rate_hz=5,
                    max_frame_bytes=8 * 1024 * 1024,
                ),
            ),
            source_mode=pb.SOURCE_MODE_SYNTHETIC,
            source_id="fixture:known-target-v1",
        )
    )
    yield pb.GatewayToWorkstation(
        session_started=pb.SessionStarted(
            request_id="fixture-start-1",
            robot_id="fixture-robot",
            robot_boot_id="fixture-boot-1",
            gateway_instance_id="fixture-gateway-1",
            observation_session_id=session_id,
            sensor_id="head_rgbd",
            effective_rate_hz=1.0 / interval_seconds,
            started_at=_timestamp(datetime.now(UTC)),
            localization_epoch_id="lab-map-epoch-1",
            source_mode=pb.SOURCE_MODE_SYNTHETIC,
            source_id="fixture:known-target-v1",
        )
    )
    sequence = 1
    while continuous or sequence <= frames:
        yield pb.GatewayToWorkstation(frame=frame_message(sequence, session_id=session_id))
        sequence += 1
        if continuous or sequence <= frames:
            await asyncio.sleep(interval_seconds)
    if not continuous:
        yield pb.GatewayToWorkstation(
            session_ended=pb.SessionEnded(
                observation_session_id=session_id,
                reason=pb.SESSION_END_REASON_REQUESTED,
                detail="fixture complete",
                ended_at=_timestamp(datetime.now(UTC)),
            )
        )


async def run(endpoint: str, frames: int, *, continuous: bool, interval_seconds: float) -> None:
    lock_path = Path(__file__).resolve().parents[1] / "contracts/robot_observation/PROTOCOL.lock"
    lock = dict(line.split("=", 1) for line in lock_path.read_text().splitlines() if line)
    session_id = f"fixture-session-{uuid4().hex}"
    async with grpc.aio.insecure_channel(endpoint) as channel:
        client = pb_grpc.RobotObservationServiceStub(channel)
        responses = client.Stream(
            messages(
                frames=frames,
                schema_sha256=lock["schema_sha256"],
                session_id=session_id,
                continuous=continuous,
                interval_seconds=interval_seconds,
            )
        )
        async for response in responses:
            body = response.WhichOneof("body")
            if body == "ack":
                result = pb.AckResult.Name(response.ack.result)
                print(
                    f"ack sequence={response.ack.sequence} "
                    f"result={result} detail={response.ack.detail}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="127.0.0.1:50061")
    parser.add_argument("--frames", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=0.4)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be greater than zero")
    asyncio.run(
        run(
            args.endpoint,
            args.frames,
            continuous=args.continuous,
            interval_seconds=args.interval_seconds,
        )
    )


if __name__ == "__main__":
    main()
