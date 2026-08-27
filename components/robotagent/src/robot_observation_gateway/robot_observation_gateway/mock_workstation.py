"""In-memory validating workstation used by integration tests and the local visual demo."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import cv2
import google_crc32c
import grpc  # type: ignore[import-untyped]
import numpy as np
import zstandard
from google.protobuf.timestamp_pb2 import Timestamp  # type: ignore[import-untyped]

from robot_observation.v1 import observation_pb2 as pb
from robot_observation.v1 import observation_pb2_grpc as pb_grpc
from robot_observation_gateway.protocol import PROTOCOL_LOCK

LOGGER = logging.getLogger(__name__)


@dataclass
class WorkstationBehavior:
    auto_start: bool = True
    requester_id: str = "local-test-workstation"
    purpose_code: int = pb.PURPOSE_CODE_LOCAL_TEST
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    scene_version: str | None = None
    required_localization_epoch_id: str | None = None
    requested_rate_hz: float = 5.0
    keepalive_interval_ms: int = 250
    keepalive_timeout_ms: int = 1_000
    max_frame_bytes: int = 3 * 1024 * 1024
    stale_frame_ms: int = 500
    ack_delay_ms: int = 0
    pause_acks: bool = False
    reject_next_reason: int | None = None


@dataclass(frozen=True)
class WorkstationSnapshot:
    connected: bool
    robot_id: str
    gateway_instance_id: str
    observation_session_id: str
    scene_version: str
    robot_boot_id: str
    localization_epoch_id: str
    accepted_frames: int
    rejected_frames: int
    last_sequence: int | None
    last_sequence_gap: int
    last_capture_age_ms: float
    last_server_processing_ms: float
    last_rgb_jpeg: bytes | None
    last_depth_png: bytes | None
    last_frame: dict[str, float | int | str]
    latest_telemetry: dict[str, float | int | bool]
    last_error: str


class MockWorkstationState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connected = False
        self._robot_id = ""
        self._gateway_instance_id = ""
        self._observation_session_id = ""
        self._scene_version = ""
        self._robot_boot_id = ""
        self._localization_epoch_id = ""
        self._accepted_frames = 0
        self._rejected_frames = 0
        self._last_sequence: int | None = None
        self._last_sequence_gap = 0
        self._last_capture_age_ms = 0.0
        self._last_server_processing_ms = 0.0
        self._last_rgb_jpeg: bytes | None = None
        self._last_depth_png: bytes | None = None
        self._last_frame: dict[str, float | int | str] = {}
        self._latest_telemetry: dict[str, float | int | bool] = {}
        self._last_error = ""

    def connection(self, hello: pb.GatewayHello) -> None:
        with self._lock:
            self._connected = True
            self._robot_id = hello.robot_id
            self._gateway_instance_id = hello.gateway_instance_id
            self._robot_boot_id = hello.robot_boot_id
            self._last_error = ""

    def disconnected(self) -> None:
        with self._lock:
            self._connected = False
            self._observation_session_id = ""

    def session_started(self, message: pb.SessionStarted) -> None:
        with self._lock:
            self._observation_session_id = message.observation_session_id
            self._scene_version = message.scene_version
            self._localization_epoch_id = message.localization_epoch_id
            self._last_sequence = None
            self._last_sequence_gap = 0

    def session_ended(self, message: pb.SessionEnded) -> None:
        with self._lock:
            if self._observation_session_id == message.observation_session_id:
                self._observation_session_id = ""

    def accepted_frame(
        self,
        *,
        frame: pb.ObservationFrame,
        sequence_gap: int,
        capture_age_ms: float,
        processing_ms: float,
        depth_png: bytes,
    ) -> None:
        with self._lock:
            self._accepted_frames += 1
            self._last_sequence = frame.sequence
            self._last_sequence_gap = sequence_gap
            self._last_capture_age_ms = capture_age_ms
            self._last_server_processing_ms = processing_ms
            self._last_rgb_jpeg = bytes(frame.rgb)
            self._last_depth_png = depth_png
            self._last_frame = {
                "captureAgeMs": frame.metrics.capture_age_ms,
                "encodeMs": frame.metrics.encode_ms,
                "queueMs": frame.metrics.queue_ms,
                "dropCount": frame.metrics.drop_count,
                "rgbFrameNumber": frame.rgb_frame_number,
                "depthFrameNumber": frame.depth_frame_number,
                "width": frame.intrinsics.width,
                "height": frame.intrinsics.height,
                "serializedBytes": frame.ByteSize(),
                "localizationAgeMs": frame.localization.age_ms,
                "calibrationSha256": frame.calibration_sha256,
                "mapFrame": frame.t_map_camera.parent_frame_id,
                "cameraFrame": frame.t_map_camera.child_frame_id,
            }

    def rejected_frame(self, detail: str) -> None:
        with self._lock:
            self._rejected_frames += 1
            self._last_error = detail

    def telemetry(self, value: pb.GatewayTelemetry) -> None:
        with self._lock:
            self._latest_telemetry = {
                "actualSendFps": value.actual_send_fps,
                "bandwidthMbps": value.bandwidth_mbps,
                "ackRttMs": value.ack_rtt_ms,
                "ackRttP95Ms": value.ack_rtt_p95_ms,
                "networkMs": value.network_ms,
                "writeBlockedMs": value.write_blocked_ms,
                "dropCount": sum(
                    (
                        value.drops.raw_overwritten,
                        value.drops.metadata_missing,
                        value.drops.tf_missing,
                        value.drops.localization_missing,
                        value.drops.stale_before_encode,
                        value.drops.encode_failed,
                        value.drops.encoded_overwritten,
                        value.drops.frame_too_large,
                        value.drops.write_timeout,
                        value.drops.workstation_rejected,
                        value.drops.invalid_frame,
                    )
                ),
                "crossMachineAgeValid": value.cross_machine_age_valid,
            }

    def error(self, detail: str) -> None:
        with self._lock:
            self._last_error = detail

    def snapshot(self) -> WorkstationSnapshot:
        with self._lock:
            return WorkstationSnapshot(
                connected=self._connected,
                robot_id=self._robot_id,
                gateway_instance_id=self._gateway_instance_id,
                observation_session_id=self._observation_session_id,
                scene_version=self._scene_version,
                robot_boot_id=self._robot_boot_id,
                localization_epoch_id=self._localization_epoch_id,
                accepted_frames=self._accepted_frames,
                rejected_frames=self._rejected_frames,
                last_sequence=self._last_sequence,
                last_sequence_gap=self._last_sequence_gap,
                last_capture_age_ms=self._last_capture_age_ms,
                last_server_processing_ms=self._last_server_processing_ms,
                last_rgb_jpeg=self._last_rgb_jpeg,
                last_depth_png=self._last_depth_png,
                last_frame=dict(self._last_frame),
                latest_telemetry=dict(self._latest_telemetry),
                last_error=self._last_error,
            )


@dataclass
class _Connection:
    outbound: asyncio.Queue[pb.WorkstationToGateway | None]
    hello: pb.GatewayHello | None = None
    calibration_sha256: str = ""
    session_id: str = ""
    sensor_id: str = ""
    scene_version: str = ""
    localization_epoch_id: str = ""
    last_sequence: int | None = None
    start_request_id: str = ""


class MockObservationServicer(pb_grpc.RobotObservationServiceServicer):
    def __init__(
        self,
        *,
        behavior: WorkstationBehavior | None = None,
        state: MockWorkstationState | None = None,
        require_robot_san: bool = False,
    ) -> None:
        self.behavior = behavior or WorkstationBehavior()
        self.state = state or MockWorkstationState()
        self._require_robot_san = require_robot_san
        self._server_instance_id = str(uuid4())
        self._connection: _Connection | None = None
        self._connection_lock = asyncio.Lock()

    async def Stream(self, request_iterator: Any, context: Any) -> Any:
        connection = _Connection(outbound=asyncio.Queue(maxsize=32))
        async with self._connection_lock:
            if self._connection is not None:
                await context.abort(
                    grpc.StatusCode.ALREADY_EXISTS,
                    "only one active observation gateway is allowed",
                )
            self._connection = connection
        consumer = asyncio.create_task(
            self._consume(request_iterator, context, connection),
            name="mock-workstation-consumer",
        )
        keepalive = asyncio.create_task(
            self._keepalives(connection),
            name="mock-workstation-keepalive",
        )
        try:
            while True:
                outbound = await connection.outbound.get()
                if outbound is None:
                    await consumer
                    return
                yield outbound
        finally:
            consumer.cancel()
            keepalive.cancel()
            await asyncio.gather(consumer, keepalive, return_exceptions=True)
            async with self._connection_lock:
                if self._connection is connection:
                    self._connection = None
            self.state.disconnected()

    async def start_observation(self) -> bool:
        connection = self._connection
        if connection is None or connection.hello is None or connection.session_id:
            return False
        await self._send_start(connection)
        return True

    async def stop_observation(self, *, reason: str = "local test stop") -> bool:
        connection = self._connection
        if connection is None or not connection.session_id:
            return False
        await connection.outbound.put(
            pb.WorkstationToGateway(
                stop=pb.StopObservation(
                    observation_session_id=connection.session_id,
                    reason=reason,
                )
            )
        )
        return True

    async def _consume(self, request_iterator: Any, context: Any, connection: _Connection) -> None:
        try:
            async for message in request_iterator:
                body = message.WhichOneof("body")
                if connection.hello is None:
                    if body != "hello":
                        await context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT,
                            "gateway hello must be the first message",
                        )
                    await self._accept_hello(message.hello, context, connection)
                    continue
                if body == "session_started":
                    self._accept_session_started(message.session_started, connection)
                elif body == "frame":
                    await self._accept_frame(message.frame, connection)
                elif body == "session_ended":
                    self.state.session_ended(message.session_ended)
                    connection.session_id = ""
                elif body == "telemetry":
                    self.state.telemetry(message.telemetry)
                elif body == "error":
                    self.state.error(message.error.detail)
                    if (
                        self.behavior.auto_start
                        and message.error.code == pb.GATEWAY_ERROR_CODE_CAMERA_UNAVAILABLE
                        and not connection.session_id
                    ):
                        await asyncio.sleep(0.25)
                        await self._send_start(connection)
                elif body == "hello":
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "duplicate hello")
                else:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        f"unknown gateway message: {body}",
                    )
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                connection.outbound.put_nowait(None)

    async def _accept_hello(
        self,
        hello: pb.GatewayHello,
        context: Any,
        connection: _Connection,
    ) -> None:
        expected_protocol = pb.ProtocolVersion(
            major=int(PROTOCOL_LOCK["major"]),
            minor=int(PROTOCOL_LOCK["minor"]),
            schema_sha256=PROTOCOL_LOCK["schema_sha256"],
        )
        if (
            hello.protocol != expected_protocol
            or not hello.robot_id
            or not hello.source_id
            or hello.source_mode == pb.SOURCE_MODE_UNSPECIFIED
            or len(hello.sensors) != 1
            or hello.sensors[0].source_id != hello.source_id
        ):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "invalid gateway identity/protocol",
            )
        if self._require_robot_san:
            sans = context.auth_context().get("x509_subject_alternative_name", ())
            decoded = {value.decode("utf-8", errors="strict") for value in sans}
            if f"robot:{hello.robot_id}" not in decoded:
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "client certificate SAN does not bind the declared robotId",
                )
        connection.hello = hello
        self.state.connection(hello)
        await connection.outbound.put(
            pb.WorkstationToGateway(
                welcome=pb.WorkstationWelcome(
                    accepted_protocol=expected_protocol,
                    server_instance_id=self._server_instance_id,
                    observation_keepalive_interval_ms=self.behavior.keepalive_interval_ms,
                    observation_keepalive_timeout_ms=self.behavior.keepalive_timeout_ms,
                    max_frame_bytes=self.behavior.max_frame_bytes,
                )
            )
        )
        if self.behavior.auto_start:
            await self._send_start(connection)

    async def _send_start(self, connection: _Connection) -> None:
        assert connection.hello is not None
        connection.start_request_id = str(uuid4())
        start = pb.StartObservation(
                    request_id=connection.start_request_id,
                    requester_id=self.behavior.requester_id,
                    purpose_code=cast(Any, self.behavior.purpose_code),
                    trace_id=self.behavior.trace_id,
                    sensor_id=connection.hello.sensors[0].sensor_id,
                    requested_rate_hz=self.behavior.requested_rate_hz,
                )
        if self.behavior.scene_version is not None:
            start.scene_version = self.behavior.scene_version
        if self.behavior.required_localization_epoch_id is not None:
            start.required_localization_epoch_id = (
                self.behavior.required_localization_epoch_id
            )
        await connection.outbound.put(pb.WorkstationToGateway(start=start))

    def _accept_session_started(
        self,
        message: pb.SessionStarted,
        connection: _Connection,
    ) -> None:
        hello = connection.hello
        assert hello is not None
        if (
            message.request_id != connection.start_request_id
            or message.robot_id != hello.robot_id
            or message.robot_boot_id != hello.robot_boot_id
            or message.gateway_instance_id != hello.gateway_instance_id
            or message.sensor_id != hello.sensors[0].sensor_id
            or (
                message.scene_version if message.HasField("scene_version") else None
            )
            != self.behavior.scene_version
            or message.source_mode != hello.source_mode
            or message.source_id != hello.source_id
            or not message.localization_epoch_id
            or not message.observation_session_id
            or len(message.calibration.calibration_sha256) != 64
        ):
            raise ValueError("gateway returned an invalid SessionStarted")
        connection.session_id = message.observation_session_id
        connection.sensor_id = message.sensor_id
        connection.scene_version = (
            message.scene_version if message.HasField("scene_version") else ""
        )
        connection.localization_epoch_id = message.localization_epoch_id
        connection.calibration_sha256 = message.calibration.calibration_sha256
        connection.last_sequence = None
        self.state.session_started(message)

    async def _accept_frame(self, frame: pb.ObservationFrame, connection: _Connection) -> None:
        started_ns = time.monotonic_ns()
        reject_reason, detail, capture_age_ms, depth_png = _validate_frame(
            frame,
            connection=connection,
            stale_frame_ms=self.behavior.stale_frame_ms,
            max_frame_bytes=self.behavior.max_frame_bytes,
        )
        if self.behavior.reject_next_reason is not None:
            reject_reason = self.behavior.reject_next_reason
            detail = "fault injection: rejected next frame"
            self.behavior.reject_next_reason = None
        sequence_gap = (
            0
            if connection.last_sequence is None
            else max(0, frame.sequence - connection.last_sequence - 1)
        )
        processing_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
        accepted = reject_reason == pb.ACK_REJECT_REASON_NONE
        if accepted:
            assert depth_png is not None
            connection.last_sequence = frame.sequence
            self.state.accepted_frame(
                frame=frame,
                sequence_gap=sequence_gap,
                capture_age_ms=capture_age_ms,
                processing_ms=processing_ms,
                depth_png=depth_png,
            )
        else:
            self.state.rejected_frame(detail)
        if self.behavior.pause_acks:
            return
        if self.behavior.ack_delay_ms:
            await asyncio.sleep(self.behavior.ack_delay_ms / 1_000.0)
        received_at = Timestamp()
        received_at.GetCurrentTime()
        await connection.outbound.put(
            pb.WorkstationToGateway(
                ack=pb.ObservationAck(
                    observation_session_id=frame.observation_session_id,
                    sequence=frame.sequence,
                    result=(pb.ACK_RESULT_ACCEPTED if accepted else pb.ACK_RESULT_REJECTED),
                    reject_reason=cast(Any, reject_reason),
                    received_at=received_at,
                    sequence_gap=sequence_gap,
                    ingest_age_ms=capture_age_ms,
                    clock_uncertainty_ms=10.0,
                    server_processing_ms=processing_ms,
                    detail=detail,
                )
            )
        )

    async def _keepalives(self, connection: _Connection) -> None:
        sequence = 1
        while True:
            await asyncio.sleep(self.behavior.keepalive_interval_ms / 1_000.0)
            if connection.hello is None:
                continue
            sent_at = Timestamp()
            sent_at.GetCurrentTime()
            await connection.outbound.put(
                pb.WorkstationToGateway(
                    keepalive=pb.ObservationKeepalive(sequence=sequence, sent_at=sent_at)
                )
            )
            sequence += 1


def _validate_frame(
    frame: pb.ObservationFrame,
    *,
    connection: _Connection,
    stale_frame_ms: int,
    max_frame_bytes: int,
) -> tuple[int, str, float, bytes | None]:
    hello = connection.hello
    assert hello is not None
    capture_ns = frame.capture_time.seconds * 1_000_000_000 + frame.capture_time.nanos
    capture_age_ms = (time.time_ns() - capture_ns) / 1_000_000.0
    if (
        frame.robot_id != hello.robot_id
        or frame.robot_boot_id != hello.robot_boot_id
        or frame.gateway_instance_id != hello.gateway_instance_id
        or frame.observation_session_id != connection.session_id
        or frame.sensor_id != connection.sensor_id
        or frame.localization_epoch_id != connection.localization_epoch_id
        or frame.source_mode != hello.source_mode
        or frame.source_id != hello.source_id
    ):
        return (
            pb.ACK_REJECT_REASON_IDENTITY_MISMATCH,
            "frame identity mismatch",
            capture_age_ms,
            None,
        )
    frame_scene = frame.scene_version if frame.HasField("scene_version") else ""
    if frame_scene != connection.scene_version:
        return pb.ACK_REJECT_REASON_SCENE_MISMATCH, "scene version mismatch", capture_age_ms, None
    if connection.last_sequence is not None and frame.sequence <= connection.last_sequence:
        return pb.ACK_REJECT_REASON_OUT_OF_ORDER, "sequence did not increase", capture_age_ms, None
    if capture_age_ms > stale_frame_ms or capture_age_ms < -50.0:
        return (
            pb.ACK_REJECT_REASON_STALE,
            "capture timestamp is stale/invalid",
            capture_age_ms,
            None,
        )
    if frame.ByteSize() > max_frame_bytes:
        return (
            pb.ACK_REJECT_REASON_TOO_LARGE,
            "serialized frame exceeds limit",
            capture_age_ms,
            None,
        )
    if (
        frame.rgb_encoding != pb.RGB_ENCODING_JPEG
        or frame.depth_encoding != pb.DEPTH_ENCODING_ZSTD_Z16_LE
    ):
        return (
            pb.ACK_REJECT_REASON_UNSUPPORTED_ENCODING,
            "unsupported frame encoding",
            capture_age_ms,
            None,
        )
    if (
        google_crc32c.value(frame.rgb) != frame.rgb_crc32c
    ):
        return pb.ACK_REJECT_REASON_BAD_CHECKSUM, "CRC32C mismatch", capture_age_ms, None
    if (
        frame.calibration_sha256 != connection.calibration_sha256
        or len(frame.localization.covariance) != 36
        or frame.t_map_camera.parent_frame_id != "map"
        or frame.t_base_camera.parent_frame_id != "base_link"
        or frame.intrinsics.width == 0
        or frame.intrinsics.height == 0
        or frame.depth_scale_meters_per_unit <= 0
    ):
        return (
            pb.ACK_REJECT_REASON_INVALID_METADATA,
            "required calibration/pose/localization metadata is invalid",
            capture_age_ms,
            None,
        )
    try:
        rgb = cv2.imdecode(np.frombuffer(frame.rgb, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape[:2] != (frame.intrinsics.height, frame.intrinsics.width):
            raise ValueError("JPEG dimensions are invalid")
        raw_depth = zstandard.ZstdDecompressor().decompress(
            frame.aligned_depth,
            max_output_size=frame.uncompressed_depth_bytes,
        )
        expected_depth_bytes = frame.intrinsics.width * frame.intrinsics.height * 2
        if len(raw_depth) != expected_depth_bytes:
            raise ValueError("depth dimensions are invalid")
        if google_crc32c.value(raw_depth) != frame.depth_crc32c:
            raise ValueError("depth CRC32C mismatch")
        depth = np.frombuffer(raw_depth, dtype="<u2").reshape(
            (frame.intrinsics.height, frame.intrinsics.width)
        )
        visible = np.clip(depth.astype(np.float32) * frame.depth_scale_meters_per_unit, 0, 5)
        visible = ((1.0 - visible / 5.0) * 255.0).astype(np.uint8)
        colorized = cv2.applyColorMap(visible, cv2.COLORMAP_TURBO)
        ok, encoded = cv2.imencode(".png", colorized)
        if not ok:
            raise ValueError("depth PNG encoding failed")
    except (ValueError, zstandard.ZstdError):
        return pb.ACK_REJECT_REASON_DECODE_ERROR, "RGB/depth decode failed", capture_age_ms, None
    return pb.ACK_REJECT_REASON_NONE, "accepted", capture_age_ms, encoded.tobytes()


class MockWorkstationServer:
    def __init__(
        self,
        *,
        bind: str = "127.0.0.1:0",
        behavior: WorkstationBehavior | None = None,
        server_cert: Path | None = None,
        server_key: Path | None = None,
        client_ca: Path | None = None,
        require_robot_san: bool = False,
    ) -> None:
        self._bind = bind
        self.servicer = MockObservationServicer(
            behavior=behavior,
            require_robot_san=require_robot_san,
        )
        self._server_cert = server_cert
        self._server_key = server_key
        self._client_ca = client_ca
        self._server: grpc.aio.Server | None = None
        self.port = 0

    async def start(self) -> None:
        server = grpc.aio.server(
            options=(
                ("grpc.max_send_message_length", 4 * 1024 * 1024),
                ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            )
        )
        pb_grpc.add_RobotObservationServiceServicer_to_server(  # type: ignore[no-untyped-call]
            self.servicer,
            server,
        )
        if self._server_cert is None:
            self.port = server.add_insecure_port(self._bind)
        else:
            assert self._server_key is not None
            assert self._client_ca is not None
            credentials = grpc.ssl_server_credentials(
                ((self._server_key.read_bytes(), self._server_cert.read_bytes()),),
                root_certificates=self._client_ca.read_bytes(),
                require_client_auth=True,
            )
            self.port = server.add_secure_port(self._bind, credentials)
        if self.port == 0:
            raise RuntimeError(f"failed to bind mock workstation at {self._bind}")
        await server.start()
        self._server = server
        LOGGER.info("mock observation workstation listening on port %s", self.port)

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop(grace=0.2)
            await self._server.wait_for_termination(timeout=2.0)
            self._server = None

    async def wait_for_frames(self, count: int, *, timeout: float = 5.0) -> WorkstationSnapshot:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            snapshot = self.servicer.state.snapshot()
            if snapshot.accepted_frames >= count:
                return snapshot
            await asyncio.sleep(0.01)
        raise TimeoutError(f"did not receive {count} accepted frames")
