"""Strict conversion between internal models and the canonical v1 Protobuf wire format."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore[import-untyped]

from robot_observation.v1 import observation_pb2 as pb
from robot_observation_gateway.config import GatewayConfig
from robot_observation_gateway.encoder import ObservationEncoder
from robot_observation_gateway.errors import DropReason, FrameTooLargeError, FrameValidationError
from robot_observation_gateway.metrics import DropTracker, LatencySnapshot
from robot_observation_gateway.model import (
    Calibration,
    CameraIntrinsics,
    EncodedObservation,
    RigidTransform,
    TimePoint,
)
from robot_observation_gateway.session import ActiveObservationSession

_PROTOCOL_LOCK_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "proto/robot_observation/PROTOCOL.lock",
    Path(__file__).resolve().parent / "PROTOCOL.lock",
)


def _protocol_lock() -> dict[str, str]:
    """Load the H-IDE-owned protocol identity copied into this repository."""
    lock_path = next((path for path in _PROTOCOL_LOCK_CANDIDATES if path.is_file()), None)
    if lock_path is None:
        raise RuntimeError("observation PROTOCOL.lock is not installed")
    try:
        lines = lock_path.read_text(encoding="ascii").splitlines()
        values = dict(line.split("=", 1) for line in lines if line)
        if set(values) != {"major", "minor", "schema_sha256"}:
            raise ValueError("unexpected keys")
        if len(values["schema_sha256"]) != 64:
            raise ValueError("invalid schema_sha256")
        return values
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid observation protocol lock {lock_path}: {exc}") from exc


PROTOCOL_LOCK = _protocol_lock()


@dataclass(frozen=True)
class GatewayIdentity:
    robot_id: str
    robot_boot_id: str
    gateway_instance_id: str


def timestamp_from_time_point(value: TimePoint) -> Timestamp:
    return Timestamp(seconds=value.sec, nanos=value.nanosec)


def timestamp_now() -> Timestamp:
    value = Timestamp()
    value.GetCurrentTime()
    return value


def _intrinsics(value: CameraIntrinsics) -> pb.CameraIntrinsics:
    return pb.CameraIntrinsics(
        width=value.width,
        height=value.height,
        fx=value.fx,
        fy=value.fy,
        cx=value.cx,
        cy=value.cy,
        distortion_model=value.distortion_model,
        distortion_coefficients=value.distortion_coefficients,
        binning_x=value.binning_x,
        binning_y=value.binning_y,
        roi=pb.RegionOfInterest(
            x_offset=value.roi.x_offset,
            y_offset=value.roi.y_offset,
            width=value.roi.width,
            height=value.roi.height,
            do_rectify=value.roi.do_rectify,
        ),
    )


def _transform(value: RigidTransform) -> pb.PoseTransform:
    return pb.PoseTransform(
        parent_frame_id=value.parent_frame_id,
        child_frame_id=value.child_frame_id,
        translation=pb.Vector3(
            x=value.translation.x,
            y=value.translation.y,
            z=value.translation.z,
        ),
        rotation=pb.Quaternion(
            x=value.rotation.x,
            y=value.rotation.y,
            z=value.rotation.z,
            w=value.rotation.w,
        ),
    )


def calibration_message(value: Calibration) -> pb.CameraCalibration:
    return pb.CameraCalibration(
        sensor_id=value.sensor_id,
        device_serial=value.device_serial,
        device_model=value.device_model,
        firmware_version=value.firmware_version,
        color_profile=value.color_profile,
        depth_profile=value.depth_profile,
        rgb_intrinsics=_intrinsics(value.rgb_intrinsics),
        aligned_depth_intrinsics=_intrinsics(value.aligned_depth_intrinsics),
        depth_scale_meters_per_unit=value.depth_scale_meters_per_unit,
        calibration_sha256=value.sha256,
    )


class ObservationProtocol:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        identity: GatewayIdentity,
        drops: DropTracker,
        encoder: ObservationEncoder,
        gateway_version: str,
        source_mode: str | None = None,
        source_id: str | None = None,
    ) -> None:
        self._config = config
        self._identity = identity
        self._drops = drops
        self._encoder = encoder
        self._gateway_version = gateway_version
        self._source_mode = source_mode or config.sensor.source_mode
        self._source_id = source_id or config.sensor.source_id

    @property
    def source_mode(self) -> int:
        return {
            "live": pb.SOURCE_MODE_LIVE,
            "replay": pb.SOURCE_MODE_REPLAY,
            "synthetic": pb.SOURCE_MODE_SYNTHETIC,
        }[self._source_mode]

    def hello(self) -> pb.GatewayToWorkstation:
        sensor = self._config.sensor
        stream = self._config.stream
        return pb.GatewayToWorkstation(
            hello=pb.GatewayHello(
                protocol=pb.ProtocolVersion(
                    major=int(PROTOCOL_LOCK["major"]),
                    minor=int(PROTOCOL_LOCK["minor"]),
                    schema_sha256=PROTOCOL_LOCK["schema_sha256"],
                ),
                robot_id=self._identity.robot_id,
                robot_boot_id=self._identity.robot_boot_id,
                gateway_instance_id=self._identity.gateway_instance_id,
                gateway_version=self._gateway_version,
                features=(
                    "newest-only",
                    "capture-time-tf",
                    "jpeg-rgb",
                    "zstd-z16-le",
                    "crc32c",
                ),
                sensors=(
                    pb.SensorDescriptor(
                        sensor_id=sensor.sensor_id,
                        source_id=self._source_id,
                        device_serial=sensor.device_serial,
                        device_model=sensor.device_model,
                        rgb_encodings=(pb.RGB_ENCODING_JPEG,),
                        depth_encodings=(pb.DEPTH_ENCODING_ZSTD_Z16_LE,),
                        max_rate_hz=stream.target_rate_hz,
                        max_frame_bytes=stream.max_frame_bytes,
                    ),
                ),
                source_mode=cast(Any, self.source_mode),
                source_id=self._source_id,
            )
        )

    def session_started(
        self,
        active: ActiveObservationSession,
        calibration: Calibration,
    ) -> pb.GatewayToWorkstation:
        started = pb.SessionStarted(
            request_id=active.request_id,
            robot_id=self._identity.robot_id,
            robot_boot_id=self._identity.robot_boot_id,
            gateway_instance_id=self._identity.gateway_instance_id,
            observation_session_id=active.observation_session_id,
            sensor_id=active.sensor_id,
            effective_rate_hz=active.rate_hz,
            started_at=timestamp_now(),
            calibration=calibration_message(calibration),
            localization_epoch_id=active.localization_epoch_id,
            source_mode=cast(Any, self.source_mode),
            source_id=self._source_id,
        )
        if active.scene_version is not None:
            started.scene_version = active.scene_version
        return pb.GatewayToWorkstation(session_started=started)

    def frame(
        self,
        active: ActiveObservationSession,
        *,
        sequence: int,
        encoded: EncodedObservation,
        queue_ms: float,
        wall_now_ns: int | None = None,
    ) -> pb.GatewayToWorkstation:
        raw = encoded.raw
        if active.sensor_id != raw.calibration.sensor_id:
            raise FrameValidationError("frame sensor does not match the active session")
        clock_domain = _clock_domain(raw.rgb_metadata.clock_domain)
        if _clock_domain(raw.depth_metadata.clock_domain) != clock_domain:
            raise FrameValidationError("RGB and depth clock domains disagree")
        now_ns = time.time_ns() if wall_now_ns is None else wall_now_ns
        capture_age_ms = (now_ns - raw.capture_time_ros.unix_ns) / 1_000_000.0
        future_tolerance = self._config.clock.estimated_uncertainty_ms
        if capture_age_ms < -future_tolerance:
            raise FrameValidationError("capture timestamp is beyond clock uncertainty")
        capture_age_ms = max(0.0, capture_age_ms)
        frame = pb.ObservationFrame(
            robot_id=self._identity.robot_id,
                robot_boot_id=self._identity.robot_boot_id,
                gateway_instance_id=self._identity.gateway_instance_id,
                observation_session_id=active.observation_session_id,
                sequence=sequence,
                sensor_id=active.sensor_id,
                capture_time=timestamp_from_time_point(raw.capture_time_ros),
                rgb_frame_number=raw.rgb_metadata.frame_number,
                depth_frame_number=raw.depth_metadata.frame_number,
                rgb_sensor_timestamp_ms=raw.rgb_metadata.sensor_timestamp_ms,
                depth_sensor_timestamp_ms=raw.depth_metadata.sensor_timestamp_ms,
                clock_domain=cast(Any, clock_domain),
                rgb_encoding=pb.RGB_ENCODING_JPEG,
                rgb=encoded.rgb_jpeg,
                rgb_crc32c=encoded.rgb_crc32c,
                depth_encoding=pb.DEPTH_ENCODING_ZSTD_Z16_LE,
                aligned_depth=encoded.depth_zstd,
                depth_crc32c=encoded.depth_crc32c,
                depth_scale_meters_per_unit=raw.calibration.depth_scale_meters_per_unit,
                intrinsics=_intrinsics(raw.intrinsics),
                calibration_sha256=raw.calibration.sha256,
                t_map_camera=_transform(raw.t_map_camera),
                t_base_camera=_transform(raw.t_base_camera),
                localization=pb.LocalizationEstimate(
                    observed_at=timestamp_from_time_point(raw.localization.observed_at),
                    frame_id=raw.localization.frame_id,
                    covariance=raw.localization.covariance,
                    age_ms=raw.localization.age_ms,
                ),
                metrics=pb.FrameMetrics(
                    capture_age_ms=capture_age_ms,
                    encode_ms=encoded.encode_ms,
                    queue_ms=queue_ms,
                    drop_count=self._drops.total,
                ),
                jpeg_quality=self._encoder.jpeg_quality,
                zstd_level=self._encoder.zstd_level,
                uncompressed_rgb_bytes=len(raw.rgb),
                uncompressed_depth_bytes=len(raw.aligned_depth),
                localization_epoch_id=active.localization_epoch_id,
                source_mode=cast(Any, self.source_mode),
                source_id=self._source_id,
            clock_uncertainty_ms=self._config.clock.estimated_uncertainty_ms,
        )
        if active.scene_version is not None:
            frame.scene_version = active.scene_version
        message = pb.GatewayToWorkstation(frame=frame)
        size = message.ByteSize()
        if size > self._config.stream.max_frame_bytes:
            raise FrameTooLargeError(
                f"serialized observation is {size} bytes; limit is "
                f"{self._config.stream.max_frame_bytes}"
            )
        return message

    def telemetry(
        self,
        *,
        active: ActiveObservationSession | None,
        actual_send_fps: float,
        bandwidth_mbps: float,
        ack_latency: LatencySnapshot,
        network_latency: LatencySnapshot,
        write_blocked_ms: float,
        last_sent_sequence: int | None,
        last_accepted_sequence: int | None,
    ) -> pb.GatewayToWorkstation:
        counts = self._drops.snapshot()
        return pb.GatewayToWorkstation(
            telemetry=pb.GatewayTelemetry(
                observation_session_id=(
                    active.observation_session_id if active is not None else ""
                ),
                emitted_at=timestamp_now(),
                actual_send_fps=actual_send_fps,
                bandwidth_mbps=bandwidth_mbps,
                ack_rtt_ms=ack_latency.current_ms,
                ack_rtt_p50_ms=ack_latency.p50_ms,
                ack_rtt_p95_ms=ack_latency.p95_ms,
                ack_rtt_max_ms=ack_latency.max_ms,
                network_ms=network_latency.current_ms,
                write_blocked_ms=write_blocked_ms,
                clock_uncertainty_ms=self._config.clock.estimated_uncertainty_ms,
                cross_machine_age_valid=self._config.clock.cross_machine_age_valid,
                drops=pb.DropCounters(
                    raw_overwritten=counts[DropReason.RAW_OVERWRITTEN],
                    metadata_missing=counts[DropReason.METADATA_MISSING],
                    tf_missing=counts[DropReason.TF_MISSING],
                    localization_missing=counts[DropReason.LOCALIZATION_MISSING],
                    stale_before_encode=counts[DropReason.STALE_BEFORE_ENCODE],
                    encode_failed=counts[DropReason.ENCODE_FAILED],
                    encoded_overwritten=counts[DropReason.ENCODED_OVERWRITTEN],
                    frame_too_large=counts[DropReason.FRAME_TOO_LARGE],
                    write_timeout=counts[DropReason.WRITE_TIMEOUT],
                    workstation_rejected=counts[DropReason.WORKSTATION_REJECTED],
                    invalid_frame=counts[DropReason.INVALID_FRAME],
                ),
                last_sent_sequence=last_sent_sequence or 0,
                last_accepted_sequence=last_accepted_sequence or 0,
            )
        )

    @staticmethod
    def session_ended(
        active: ActiveObservationSession,
        *,
        reason: int,
        detail: str,
    ) -> pb.GatewayToWorkstation:
        return pb.GatewayToWorkstation(
            session_ended=pb.SessionEnded(
                observation_session_id=active.observation_session_id,
                reason=cast(Any, reason),
                detail=detail,
                ended_at=timestamp_now(),
            )
        )

    @staticmethod
    def error(
        *,
        code: int,
        detail: str,
        related_session_id: str = "",
        related_sequence: int | None = None,
    ) -> pb.GatewayToWorkstation:
        value = pb.GatewayError(
            code=cast(Any, code),
            detail=detail,
            related_session_id=related_session_id,
        )
        if related_sequence is not None:
            value.related_sequence = related_sequence
        return pb.GatewayToWorkstation(error=value)


def _clock_domain(value: str) -> int:
    normalized = value.strip().lower().replace("-", "_")
    mapping = {
        "global_time": pb.CLOCK_DOMAIN_GLOBAL_TIME,
        "hardware_clock": pb.CLOCK_DOMAIN_HARDWARE_CLOCK,
        "system_time": pb.CLOCK_DOMAIN_SYSTEM_TIME,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise FrameValidationError(f"unsupported sensor clock domain: {value}") from exc
