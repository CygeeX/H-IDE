from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

from h_ide_vision.generated.robot_observation.v1 import observation_pb2 as pb
from h_ide_vision.generated.robot_observation.v1 import observation_pb2_grpc as pb_grpc

from .proto_codec import decode_frame
from .store import LatestFrameStore
from .validation import ObservationRejected

LOGGER = logging.getLogger(__name__)

# Three stable observations must fit comfortably inside the default one-second
# freshness window.  Two FPS puts the oldest of three samples exactly on the
# boundary before transport and decode latency are included.
OBSERVATION_REQUEST_RATE_HZ = 5.0


def _timestamp() -> Timestamp:
    value = Timestamp()
    value.FromDatetime(datetime.now(UTC))
    return value


def _peer_certificate_identities(context) -> set[str]:  # type: ignore[no-untyped-def]
    identities: set[str] = set()
    for key in ("x509_common_name", "x509_subject_alternative_name"):
        for raw in context.auth_context().get(key, ()):
            identities.add(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
    return identities


class ObservationIngress(pb_grpc.RobotObservationServiceServicer):
    def __init__(
        self,
        *,
        frames: LatestFrameStore,
        schema_sha256: str,
        require_peer_identity: bool = False,
    ) -> None:
        self.frames = frames
        self.schema_sha256 = schema_sha256
        self.require_peer_identity = require_peer_identity
        self.server_instance_id = f"vision-{uuid4().hex}"

    async def Stream(self, request_iterator, context):  # type: ignore[no-untyped-def]
        hello = None
        session_id = None
        session = None
        requested_sensor_id = None
        keepalive_sequence = 0
        next_keepalive_at: float | None = None
        iterator = request_iterator.__aiter__()
        pending_read = asyncio.create_task(anext(iterator))
        try:
            while True:
                if (
                    next_keepalive_at is not None
                    and time.monotonic() >= next_keepalive_at
                ):
                    keepalive_sequence += 1
                    yield pb.WorkstationToGateway(
                        keepalive=pb.ObservationKeepalive(
                            sequence=keepalive_sequence,
                            sent_at=_timestamp(),
                        )
                    )
                    next_keepalive_at = time.monotonic() + 0.25
                    continue
                timeout = (
                    None
                    if next_keepalive_at is None
                    else max(0.0, next_keepalive_at - time.monotonic())
                )
                done, _ = await asyncio.wait((pending_read,), timeout=timeout)
                if not done:
                    keepalive_sequence += 1
                    yield pb.WorkstationToGateway(
                        keepalive=pb.ObservationKeepalive(
                            sequence=keepalive_sequence,
                            sent_at=_timestamp(),
                        )
                    )
                    next_keepalive_at = time.monotonic() + 0.25
                    continue
                try:
                    incoming = pending_read.result()
                except StopAsyncIteration:
                    return
                pending_read = asyncio.create_task(anext(iterator))
                body = incoming.WhichOneof("body")
                async for response in self._handle_message(
                    incoming,
                    body,
                    context,
                    hello=hello,
                    session=session,
                    session_id=session_id,
                    requested_sensor_id=requested_sensor_id,
                ):
                    marker = response[0]
                    if marker == "state":
                        hello, session, session_id, requested_sensor_id = response[1:]
                        if next_keepalive_at is None and hello is not None:
                            next_keepalive_at = time.monotonic() + 0.25
                    else:
                        yield response[1]
                if body == "session_ended":
                    return
        finally:
            if not pending_read.done():
                pending_read.cancel()
            await asyncio.gather(pending_read, return_exceptions=True)

    async def _handle_message(  # type: ignore[no-untyped-def]
        self,
        incoming,
        body,
        context,
        *,
        hello,
        session,
        session_id,
        requested_sensor_id,
    ):
        """Validate one inbound message and yield state/output tuples.

        Keeping the read loop separate lets ``Stream`` emit the protocol
        keepalive on schedule even while the camera-side iterator is idle.
        """

        if hello is None:
            if body != "hello":
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "first message must be hello",
                )
            hello = incoming.hello
            version = hello.protocol
            if (
                version.major != 1
                or version.minor != 0
                or version.schema_sha256 != self.schema_sha256
            ):
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details("protocol version/hash mismatch")
                return
            if not all(
                value.strip()
                for value in (
                    hello.robot_id,
                    hello.robot_boot_id,
                    hello.gateway_instance_id,
                    hello.source_id,
                )
            ) or not hello.sensors:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "gateway hello identity and sensor list are required",
                )
            if (
                self.require_peer_identity
                and hello.gateway_instance_id not in _peer_certificate_identities(context)
            ):
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details(
                    "client certificate identity does not match gateway instance"
                )
                return
            yield "output", pb.WorkstationToGateway(
                welcome=pb.WorkstationWelcome(
                    accepted_protocol=pb.ProtocolVersion(
                        major=1,
                        minor=0,
                        schema_sha256=self.schema_sha256,
                    ),
                    server_instance_id=self.server_instance_id,
                    observation_keepalive_interval_ms=250,
                    observation_keepalive_timeout_ms=1000,
                    max_frame_bytes=8 * 1024 * 1024,
                )
            )
            requested_sensor_id = hello.sensors[0].sensor_id
            yield "output", pb.WorkstationToGateway(
                start=pb.StartObservation(
                    request_id=f"start-{uuid4().hex}",
                    requester_id="h-ide-vision",
                    purpose_code=pb.PURPOSE_CODE_VISUAL_TARGET_RESOLUTION,
                    trace_id=f"trace-{uuid4().hex}",
                    sensor_id=requested_sensor_id,
                    requested_rate_hz=OBSERVATION_REQUEST_RATE_HZ,
                )
            )
            yield "state", hello, session, session_id, requested_sensor_id
            return
        if body == "session_started":
            if session is not None:
                await context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "an observation stream may start only one session",
                )
            started = incoming.session_started
            if (
                started.robot_id != hello.robot_id
                or started.robot_boot_id != hello.robot_boot_id
                or started.gateway_instance_id != hello.gateway_instance_id
                or started.sensor_id != requested_sensor_id
                or started.source_mode != hello.source_mode
                or started.source_id != hello.source_id
                or not started.observation_session_id.strip()
                or not started.localization_epoch_id.strip()
            ):
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    "session identity does not match gateway hello",
                )
            self.frames.activate_session(
                started.robot_id,
                started.sensor_id,
                started.observation_session_id,
            )
            session_id = started.observation_session_id
            session = started
            LOGGER.info(
                "observation session started robot=%s sensor=%s session=%s source=%s",
                started.robot_id,
                started.sensor_id,
                started.observation_session_id,
                started.source_id,
            )
            yield "state", hello, session, session_id, requested_sensor_id
            return
        if body == "frame":
            started = time.perf_counter()
            frame = incoming.frame
            result = pb.ACK_RESULT_ACCEPTED
            reason = pb.ACK_REJECT_REASON_NONE
            detail = "accepted"
            try:
                if session is None or session_id is None:
                    raise ObservationRejected("identity_mismatch", "frame session is not active")
                frame_scene = frame.scene_version if frame.HasField("scene_version") else None
                session_scene = (
                    session.scene_version if session.HasField("scene_version") else None
                )
                if (
                    frame.robot_id != session.robot_id
                    or frame.robot_boot_id != session.robot_boot_id
                    or frame.gateway_instance_id != session.gateway_instance_id
                    or frame.observation_session_id != session_id
                    or frame.sensor_id != session.sensor_id
                    or frame.localization_epoch_id != session.localization_epoch_id
                    or frame.source_mode != session.source_mode
                    or frame.source_id != session.source_id
                    or frame_scene != session_scene
                ):
                    raise ObservationRejected(
                        "identity_mismatch", "frame identity does not match active session"
                    )
                self.frames.accept(decode_frame(frame))
            except (ObservationRejected, ValueError) as exc:
                result = pb.ACK_RESULT_REJECTED
                key = getattr(exc, "reason", "decode_error")
                reason = {
                    "stale": pb.ACK_REJECT_REASON_STALE,
                    "bad_checksum": pb.ACK_REJECT_REASON_BAD_CHECKSUM,
                    "out_of_order": pb.ACK_REJECT_REASON_OUT_OF_ORDER,
                    "identity_mismatch": pb.ACK_REJECT_REASON_IDENTITY_MISMATCH,
                    "calibration_mismatch": pb.ACK_REJECT_REASON_CALIBRATION_MISMATCH,
                    "localization_mismatch": pb.ACK_REJECT_REASON_LOCALIZATION_MISMATCH,
                }.get(key, pb.ACK_REJECT_REASON_INVALID_METADATA)
                detail = str(exc)
                LOGGER.warning(
                    "observation frame rejected robot=%s session=%s sequence=%s reason=%s detail=%s",
                    frame.robot_id,
                    frame.observation_session_id,
                    frame.sequence,
                    key,
                    detail,
                )
            yield "output", pb.WorkstationToGateway(
                ack=pb.ObservationAck(
                    observation_session_id=frame.observation_session_id,
                    sequence=frame.sequence,
                    result=result,
                    reject_reason=reason,
                    received_at=_timestamp(),
                    server_processing_ms=(time.perf_counter() - started) * 1000,
                    detail=detail,
                )
            )


async def create_grpc_server(
    *,
    frames: LatestFrameStore,
    schema_sha256: str,
    host: str,
    port: int,
    allow_insecure_loopback: bool = True,
    tls_cert: Path | None = None,
    tls_key: Path | None = None,
    tls_ca: Path | None = None,
):
    server = grpc.aio.server(options=(("grpc.max_receive_message_length", 8 * 1024 * 1024),))
    secure = all(value is not None for value in (tls_cert, tls_key, tls_ca))
    pb_grpc.add_RobotObservationServiceServicer_to_server(
        ObservationIngress(
            frames=frames,
            schema_sha256=schema_sha256,
            require_peer_identity=secure,
        ),
        server,
    )
    tls_values = (tls_cert, tls_key, tls_ca)
    if all(value is not None for value in tls_values):
        assert tls_cert is not None and tls_key is not None and tls_ca is not None
        credentials = grpc.ssl_server_credentials(
            ((tls_key.read_bytes(), tls_cert.read_bytes()),),
            root_certificates=tls_ca.read_bytes(),
            require_client_auth=True,
        )
        if server.add_secure_port(f"{host}:{port}", credentials) == 0:
            raise RuntimeError("failed to bind secure observation ingress")
    elif any(value is not None for value in tls_values):
        raise ValueError("H-IDE Vision mTLS requires cert, key, and CA together")
    else:
        normalized = host.rstrip(".").lower()
        # A host-side Robot Agent reaches this service through a Docker-published
        # loopback port.  Inside the container, binding to 127.0.0.1 would make
        # that published port accept and then close connections because the
        # Docker bridge targets the container interface.  0.0.0.0 is therefore
        # allowed only under the same explicit insecure-development switch; the
        # compose port remains bound to 127.0.0.1 on the workstation.
        allowed_plaintext_hosts = {"127.0.0.1", "::1", "localhost"}
        if not allow_insecure_loopback or normalized not in allowed_plaintext_hosts:
            raise ValueError("plaintext observation ingress is restricted to explicit local development")
        if server.add_insecure_port(f"{host}:{port}") == 0:
            raise RuntimeError("failed to bind plaintext observation ingress")
    return server


class ObservationGrpcServerThread:
    """Run observation ingress on an event loop isolated from HTTP/inference.

    Resolution jobs can execute heavyweight native ML code.  Keeping gRPC on
    its own loop prevents that work from starving observation ACKs and lease
    keepalives.  ``LatestFrameStore`` is explicitly thread-safe and remains the
    only shared mutable object between ingress and inference.
    """

    def __init__(
        self,
        *,
        frames: LatestFrameStore,
        schema_sha256: str,
        host: str,
        port: int,
        allow_insecure_loopback: bool = True,
        tls_cert: Path | None = None,
        tls_key: Path | None = None,
        tls_ca: Path | None = None,
    ) -> None:
        self._server_kwargs = {
            "frames": frames,
            "schema_sha256": schema_sha256,
            "host": host,
            "port": port,
            "allow_insecure_loopback": allow_insecure_loopback,
            "tls_cert": tls_cert,
            "tls_key": tls_key,
            "tls_ca": tls_ca,
        }
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="observation-grpc",
            daemon=True,
        )
        self._state_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._failure: BaseException | None = None
        self._grace_seconds = 2.0
        self._started = False

    def start(self, *, timeout_seconds: float = 10.0) -> None:
        with self._state_lock:
            if self._started:
                raise RuntimeError("observation gRPC server thread is already started")
            self._started = True
        self._thread.start()
        if not self._ready.wait(timeout_seconds):
            self.stop(grace_seconds=0, timeout_seconds=timeout_seconds)
            raise RuntimeError("observation gRPC server did not become ready")
        if self._failure is not None:
            self._thread.join(timeout_seconds)
            raise RuntimeError("observation gRPC server failed to start") from self._failure

    def stop(
        self,
        *,
        grace_seconds: float = 2.0,
        timeout_seconds: float = 10.0,
    ) -> None:
        with self._state_lock:
            if not self._started or self._stopped.is_set():
                return
            self._grace_seconds = grace_seconds
            loop = self._loop
            stop_event = self._stop_event
        if loop is not None and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
        self._thread.join(timeout_seconds)
        if self._thread.is_alive():
            raise RuntimeError("observation gRPC server did not stop")
        if self._failure is not None:
            raise RuntimeError("observation gRPC server failed") from self._failure

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._failure = exc
            self._ready.set()
        finally:
            self._stopped.set()

    async def _serve(self) -> None:
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        with self._state_lock:
            self._loop = loop
            self._stop_event = stop_event
        server = await create_grpc_server(**self._server_kwargs)
        await server.start()
        self._ready.set()
        try:
            await stop_event.wait()
        finally:
            await server.stop(grace=self._grace_seconds)
