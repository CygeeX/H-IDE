"""Async gateway runtime: one reverse gRPC stream and a capacity-one image pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import grpc  # type: ignore[import-untyped]

from robot_observation.v1 import observation_pb2 as pb
from robot_observation.v1 import observation_pb2_grpc as pb_grpc
from robot_observation_gateway.config import GatewayConfig, validate_tls_files
from robot_observation_gateway.encoder import ObservationEncoder
from robot_observation_gateway.errors import (
    DropReason,
    FrameTooLargeError,
    FrameValidationError,
    SessionProtocolError,
)
from robot_observation_gateway.health import GatewayHealth, HealthReason, HealthStatus
from robot_observation_gateway.identity import new_gateway_instance_id, read_robot_boot_id
from robot_observation_gateway.latest_slot import AsyncLatestSlot, LatestSlot
from robot_observation_gateway.metrics import (
    DropTracker,
    RollingLatency,
    TrafficWindow,
)
from robot_observation_gateway.model import Calibration, EncodedObservation, RawObservation
from robot_observation_gateway.protocol import GatewayIdentity, ObservationProtocol
from robot_observation_gateway.ros_ingress import RosObservationSource
from robot_observation_gateway.session import GatewaySessionController, GatewaySessionState
from robot_observation_gateway.synthetic_source import SyntheticObservationSource

LOGGER = logging.getLogger(__name__)
GATEWAY_VERSION = "0.1.0"


class _Reconnect(RuntimeError):
    pass


class ObservationSource(Protocol):
    raw_slot: LatestSlot[RawObservation]

    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def calibration(self) -> Calibration | None: ...

    @property
    def last_camera_age_ms(self) -> float: ...

    @property
    def source_mode(self) -> str: ...

    @property
    def source_id(self) -> str: ...

    @property
    def localization_epoch_id(self) -> str: ...


@dataclass(frozen=True)
class _Inflight:
    observation_session_id: str
    sequence: int
    sent_monotonic_ns: int
    capture_unix_ns: int
    serialized_bytes: int


class GatewayRuntime:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        source_factory: (
            Callable[[DropTracker, Callable[[], GatewayHealth]], ObservationSource] | None
        ) = None,
    ) -> None:
        self._config = config
        self._identity = GatewayIdentity(
            robot_id=config.identity.robot_id,
            robot_boot_id=read_robot_boot_id(),
            gateway_instance_id=new_gateway_instance_id(),
        )
        self._drops = DropTracker()
        self._ack_latency = RollingLatency()
        self._network_latency = RollingLatency()
        self._write_latency = RollingLatency()
        self._encode_latency = RollingLatency()
        self._traffic = TrafficWindow(window_seconds=2.0)
        self._encoder = ObservationEncoder(
            jpeg_quality=config.stream.jpeg_quality,
            zstd_level=config.stream.zstd_level,
            max_frame_bytes=config.stream.max_frame_bytes,
        )
        self._session = GatewaySessionController(
            minimum_rate_hz=config.stream.minimum_rate_hz,
            maximum_rate_hz=config.stream.target_rate_hz,
        )
        self._state_lock = threading.RLock()
        self._encoded_slot: AsyncLatestSlot[EncodedObservation] = AsyncLatestSlot()
        self._control_queue: asyncio.Queue[pb.GatewayToWorkstation] | None = None
        self._stop = asyncio.Event()
        self._ack_event = asyncio.Event()
        self._inflight: _Inflight | None = None
        self._active_calibration_sha256: str | None = None
        self._last_accepted_capture_unix_ns = 0
        self._last_ended_session_id = ""
        self._transport_detail = "not connected"
        self._call: Any = None
        self._encoder_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="observation-encoder",
        )
        self._source: ObservationSource
        if source_factory is None:
            if config.sensor.source_mode == "synthetic":
                # The MuJoCo profile intentionally has no RealSense dependency.  Keep
                # this selection in the sidecar (instead of in the launcher) so the
                # protocol still reports SOURCE_MODE_SYNTHETIC and all normal session,
                # encoding, ACK, and health checks remain identical to the live path.
                self._source = SyntheticObservationSource(config)
            elif config.sensor.source_mode in {"live", "mujoco"}:
                self._source = RosObservationSource(
                    config,
                    drops=self._drops,
                    health_provider=self.health,
                )
            else:
                raise ValueError(
                    "source_mode='replay' requires an explicit source_factory; "
                    "replay is not a default runtime path"
                )
        else:
            self._source = source_factory(self._drops, self.health)
        self._protocol = ObservationProtocol(
            config,
            identity=self._identity,
            drops=self._drops,
            encoder=self._encoder,
            gateway_version=GATEWAY_VERSION,
            source_mode=self._source.source_mode,
            source_id=self._source.source_id,
        )

    async def run(self) -> None:
        validate_tls_files(self._config.tls)
        self._source.start()
        backoff_ms = self._config.transport.reconnect_initial_ms
        try:
            while not self._stop.is_set():
                try:
                    await self._run_connection()
                    backoff_ms = self._config.transport.reconnect_initial_ms
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._mark_disconnected(str(exc))
                    if self._stop.is_set():
                        break
                    LOGGER.warning("observation stream disconnected: %s", exc)
                    delay = random.uniform(0.0, backoff_ms / 1_000.0)
                    await self._wait_for_stop(delay)
                    backoff_ms = min(
                        self._config.transport.reconnect_max_ms,
                        backoff_ms * 2,
                    )
        finally:
            self._mark_disconnected("gateway shutdown")
            await self._encoded_slot.close()
            self._source.stop()
            self._encoder_pool.shutdown(wait=True, cancel_futures=True)

    async def stop(self) -> None:
        self._stop.set()
        call = self._call
        if call is not None:
            call.cancel()

    async def _wait_for_stop(self, delay_seconds: float) -> None:
        """Sleep interruptibly on Python 3.10 without wait_for's Event race."""
        if self._stop.is_set() or delay_seconds <= 0:
            return
        sleep_task = asyncio.create_task(asyncio.sleep(delay_seconds))
        stop_task = asyncio.create_task(self._stop.wait())
        done, pending = await asyncio.wait(
            (sleep_task, stop_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()

    def health(self) -> GatewayHealth:
        now_monotonic = time.monotonic()
        now_wall_ns = time.time_ns()
        with self._state_lock:
            state = self._session.state
            active = self._session.active
            last_keepalive = self._session.last_keepalive_monotonic
            last_ack = self._session.last_accepted_monotonic
            last_sequence = self._session.last_accepted_sequence
            last_capture_ns = self._last_accepted_capture_unix_ns
            transport_detail = self._transport_detail

        status = HealthStatus.DISCONNECTED
        reason = HealthReason.TRANSPORT_ERROR
        if state is GatewaySessionState.CONNECTED:
            status = HealthStatus.CONNECTED
            reason = HealthReason.NO_SESSION
        elif state is GatewaySessionState.IDLE:
            status = HealthStatus.IDLE
            reason = HealthReason.NO_SESSION
        elif state is GatewaySessionState.STREAMING:
            status = HealthStatus.STREAMING_WAITING_ACK
            reason = HealthReason.NO_ACCEPTED_ACK
            if self._source.last_camera_age_ms > self._config.stream.stale_frame_ms:
                status = HealthStatus.DEGRADED
                reason = HealthReason.CAMERA_UNAVAILABLE
            elif (
                last_keepalive is None
                or (now_monotonic - last_keepalive) * 1_000
                > self._config.observation_lease.keepalive_timeout_ms
            ):
                status = HealthStatus.DEGRADED
                reason = HealthReason.KEEPALIVE_STALE
            elif last_ack is not None:
                ack_age_ms = (now_monotonic - last_ack) * 1_000
                if ack_age_ms <= self._config.transport.ack_timeout_ms:
                    status = HealthStatus.USABLE
                    reason = HealthReason.NONE
                else:
                    status = HealthStatus.DEGRADED
                    reason = HealthReason.ACK_STALE
        ack_age_ms = (
            (now_monotonic - last_ack) * 1_000 if last_ack is not None else float("inf")
        )
        capture_age_ms = (
            max(0.0, (now_wall_ns - last_capture_ns) / 1_000_000.0)
            if last_capture_ns
            else float("inf")
        )
        _ = transport_detail  # Retained for logs without exposing arbitrary text in ROS health.
        return GatewayHealth(
            robot_id=self._identity.robot_id,
            robot_boot_id=self._identity.robot_boot_id,
            gateway_instance_id=self._identity.gateway_instance_id,
            observation_session_id=(active.observation_session_id if active else ""),
            sensor_id=(active.sensor_id if active else self._config.sensor.sensor_id),
            scene_version=(active.scene_version or "" if active else ""),
            localization_epoch_id=self._source.localization_epoch_id,
            status=status,
            reason=reason,
            last_accepted_sequence=last_sequence,
            last_accepted_monotonic_ns=(
                int(last_ack * 1_000_000_000) if last_ack is not None else 0
            ),
            capture_age_ms=capture_age_ms,
            ack_age_ms=ack_age_ms,
            drop_count=self._drops.total,
        )

    async def _run_connection(self) -> None:
        channel = self._make_channel()
        self._control_queue = asyncio.Queue(maxsize=16)
        self._inflight = None
        self._ack_event.clear()
        try:
            await asyncio.wait_for(
                channel.channel_ready(),
                timeout=self._config.transport.connect_timeout_ms / 1_000.0,
            )
            stub = pb_grpc.RobotObservationServiceStub(channel)  # type: ignore[no-untyped-call]
            call = stub.Stream()
            self._call = call
            with self._state_lock:
                self._session.connected()
                self._transport_detail = "connected; awaiting welcome"
            await self._write(call, self._protocol.hello())
            inbound = await asyncio.wait_for(
                call.read(),
                timeout=self._config.transport.connect_timeout_ms / 1_000.0,
            )
            if inbound is grpc.aio.EOF:
                raise _Reconnect("workstation closed before welcome")
            self._accept_welcome(inbound)
            tasks = {
                asyncio.create_task(self._read_loop(call), name="observation-reader"),
                asyncio.create_task(self._write_loop(call), name="observation-writer"),
                asyncio.create_task(self._encode_loop(), name="observation-encoder-loop"),
                asyncio.create_task(self._telemetry_loop(), name="observation-telemetry"),
                asyncio.create_task(self._lease_loop(), name="observation-lease"),
            }
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                if task.cancelled():
                    continue
                error = task.exception()
                if error is not None:
                    raise error
            if not self._stop.is_set():
                raise _Reconnect("observation stream task stopped unexpectedly")
        finally:
            self._call = None
            await self._encoded_slot.clear()
            self._source.raw_slot.clear()
            call = locals().get("call")
            if call is not None:
                call.cancel()
            await channel.close(grace=0.2)
            self._mark_disconnected("transport closed")

    def _make_channel(self) -> grpc.aio.Channel:
        transport = self._config.transport
        options: tuple[tuple[str, int | str], ...] = (
            ("grpc.max_send_message_length", transport.grpc_max_message_bytes),
            ("grpc.max_receive_message_length", transport.grpc_max_message_bytes),
            ("grpc.keepalive_time_ms", transport.http2_keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", transport.http2_keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", 1),
            ("grpc.enable_retries", 0),
        )
        if not self._config.tls.enabled:
            return grpc.aio.insecure_channel(transport.target, options=options)
        tls = self._config.tls
        assert tls.ca_file is not None
        assert tls.cert_file is not None
        assert tls.key_file is not None
        assert tls.server_name is not None
        credentials = grpc.ssl_channel_credentials(
            root_certificates=Path(tls.ca_file).read_bytes(),
            private_key=Path(tls.key_file).read_bytes(),
            certificate_chain=Path(tls.cert_file).read_bytes(),
        )
        secure_options = (*options, ("grpc.ssl_target_name_override", tls.server_name))
        return grpc.aio.secure_channel(
            transport.target,
            credentials,
            options=secure_options,
        )

    def _accept_welcome(self, inbound: pb.WorkstationToGateway) -> None:
        if inbound.WhichOneof("body") != "welcome":
            raise SessionProtocolError("first workstation message must be welcome")
        welcome = inbound.welcome
        expected = self._protocol.hello().hello.protocol
        if welcome.accepted_protocol != expected:
            raise SessionProtocolError("workstation accepted a different protocol version/hash")
        if welcome.max_frame_bytes < self._config.stream.max_frame_bytes:
            raise SessionProtocolError("workstation frame limit is below configured hard limit")
        lease = self._config.observation_lease
        if (
            welcome.observation_keepalive_interval_ms != lease.keepalive_interval_ms
            or welcome.observation_keepalive_timeout_ms != lease.keepalive_timeout_ms
        ):
            raise SessionProtocolError("workstation observation lease does not match configuration")
        now = time.monotonic()
        with self._state_lock:
            self._session.welcome(server_instance_id=welcome.server_instance_id)
            self._session.keepalive(sequence=0, monotonic_now=now)
            self._transport_detail = "welcome accepted"

    async def _read_loop(self, call: Any) -> None:
        while not self._stop.is_set():
            inbound = await call.read()
            if inbound is grpc.aio.EOF:
                raise _Reconnect("workstation closed the observation stream")
            body = inbound.WhichOneof("body")
            if body == "start":
                await self._handle_start(inbound.start)
            elif body == "stop":
                await self._handle_stop(inbound.stop)
            elif body == "ack":
                self._handle_ack(inbound.ack)
            elif body == "keepalive":
                self._handle_keepalive(inbound.keepalive)
            elif body == "rate_hint":
                self._handle_rate_hint(inbound.rate_hint)
            else:
                raise SessionProtocolError(f"unexpected workstation message: {body}")

    async def _handle_start(self, message: pb.StartObservation) -> None:
        await self._await_source_ready()
        calibration = self._source.calibration
        if message.sensor_id != self._config.sensor.sensor_id:
            self._enqueue_control(
                self._protocol.error(
                    code=pb.GATEWAY_ERROR_CODE_CAMERA_UNAVAILABLE,
                    detail="requested sensor is unavailable",
                )
            )
            return
        if (
            calibration is None
            or self._source.last_camera_age_ms > self._config.stream.stale_frame_ms
        ):
            self._enqueue_control(
                self._protocol.error(
                    code=pb.GATEWAY_ERROR_CODE_CAMERA_UNAVAILABLE,
                    detail="camera has no recent fully localized RGB-D frame",
                )
            )
            return
        if (
            message.HasField("required_localization_epoch_id")
            and message.required_localization_epoch_id
            != self._source.localization_epoch_id
        ):
            self._enqueue_control(
                self._protocol.error(
                    code=pb.GATEWAY_ERROR_CODE_PROTOCOL_VIOLATION,
                    detail="required localization epoch is not active on this robot",
                )
            )
            return
        try:
            with self._state_lock:
                active = self._session.start(
                    request_id=message.request_id,
                    requester_id=message.requester_id,
                    purpose_code=message.purpose_code,
                    trace_id=message.trace_id,
                    sensor_id=message.sensor_id,
                    scene_version=(
                        message.scene_version if message.HasField("scene_version") else None
                    ),
                    localization_epoch_id=self._source.localization_epoch_id,
                    requested_rate_hz=message.requested_rate_hz,
                )
                self._active_calibration_sha256 = calibration.sha256
                self._last_accepted_capture_unix_ns = 0
        except SessionProtocolError as exc:
            LOGGER.warning("observation session start rejected: %s", exc)
            self._enqueue_control(
                self._protocol.error(
                    code=pb.GATEWAY_ERROR_CODE_SESSION_CONFLICT,
                    detail=str(exc),
                )
            )
            return
        self._source.raw_slot.clear()
        await self._encoded_slot.clear()
        LOGGER.info(
            "observation session active session=%s rate_hz=%.2f calibration=%s",
            active.observation_session_id,
            active.rate_hz,
            calibration.sha256,
        )
        self._enqueue_control(self._protocol.session_started(active, calibration))

    async def _handle_stop(self, message: pb.StopObservation) -> None:
        try:
            with self._state_lock:
                active = self._session.stop(message.observation_session_id)
                self._last_ended_session_id = active.observation_session_id
                self._active_calibration_sha256 = None
                self._last_accepted_capture_unix_ns = 0
        except SessionProtocolError as exc:
            self._enqueue_control(
                self._protocol.error(
                    code=pb.GATEWAY_ERROR_CODE_PROTOCOL_VIOLATION,
                    detail=str(exc),
                    related_session_id=message.observation_session_id,
                )
            )
            return
        self._inflight = None
        self._ack_event.set()
        self._source.raw_slot.clear()
        await self._encoded_slot.clear()
        self._enqueue_control(
            self._protocol.session_ended(
                active,
                reason=pb.SESSION_END_REASON_REQUESTED,
                detail=message.reason or "workstation requested stop",
            )
        )

    def _handle_ack(self, message: pb.ObservationAck) -> None:
        inflight = self._inflight
        if inflight is None:
            with self._state_lock:
                if (
                    message.observation_session_id == self._last_ended_session_id
                    or message.sequence == self._session.last_accepted_sequence
                ):
                    return
            raise SessionProtocolError("ACK received with no frame in flight")
        if (
            message.observation_session_id != inflight.observation_session_id
            or message.sequence != inflight.sequence
        ):
            raise SessionProtocolError("ACK does not match the single frame in flight")
        accepted = message.result == pb.ACK_RESULT_ACCEPTED
        if accepted and message.reject_reason != pb.ACK_REJECT_REASON_NONE:
            raise SessionProtocolError("accepted ACK must not include a rejection reason")
        if not accepted and (
            message.result != pb.ACK_RESULT_REJECTED
            or message.reject_reason == pb.ACK_REJECT_REASON_NONE
        ):
            raise SessionProtocolError("rejected ACK must include a rejection reason")
        if not math.isfinite(message.server_processing_ms) or message.server_processing_ms < 0:
            raise SessionProtocolError("ACK server_processing_ms must be finite and non-negative")
        now = time.monotonic()
        with self._state_lock:
            refreshed = self._session.acknowledge(
                observation_session_id=message.observation_session_id,
                sequence=message.sequence,
                accepted=accepted,
                monotonic_now=now,
            )
            self._session.note_peer_activity(monotonic_now=now)
            if refreshed:
                self._last_accepted_capture_unix_ns = inflight.capture_unix_ns
        rtt_ms = (time.monotonic_ns() - inflight.sent_monotonic_ns) / 1_000_000.0
        self._ack_latency.observe(rtt_ms)
        self._network_latency.observe(max(0.0, rtt_ms - message.server_processing_ms))
        if not accepted:
            self._drops.increment(DropReason.WORKSTATION_REJECTED)
        self._inflight = None
        self._ack_event.set()

    def _handle_keepalive(self, message: pb.ObservationKeepalive) -> None:
        with self._state_lock:
            self._session.keepalive(
                sequence=message.sequence,
                monotonic_now=time.monotonic(),
            )

    def _handle_rate_hint(self, message: pb.RateHint) -> None:
        with self._state_lock:
            self._session.apply_rate_hint(
                observation_session_id=message.observation_session_id,
                target_rate_hz=message.target_rate_hz,
            )

    async def _write_loop(self, call: Any) -> None:
        while not self._stop.is_set():
            queue = self._control_queue
            assert queue is not None
            try:
                control = queue.get_nowait()
            except asyncio.QueueEmpty:
                control = None
            if control is not None:
                await self._write(call, control)
                continue
            if self._inflight is not None:
                try:
                    await asyncio.wait_for(self._ack_event.wait(), timeout=0.02)
                    self._ack_event.clear()
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                encoded = await asyncio.wait_for(self._encoded_slot.get(), timeout=0.02)
            except asyncio.TimeoutError:
                continue
            except EOFError:
                return
            await self._write_encoded(call, encoded)

    async def _write_encoded(self, call: Any, encoded: EncodedObservation) -> None:
        with self._state_lock:
            active = self._session.active
            expected_calibration = self._active_calibration_sha256
        if active is None:
            return
        if encoded.raw.calibration.sha256 != expected_calibration:
            LOGGER.warning(
                "observation calibration changed expected=%s actual=%s",
                expected_calibration,
                encoded.raw.calibration.sha256,
            )
            await self._write(
                call,
                self._protocol.session_ended(
                    active,
                    reason=pb.SESSION_END_REASON_CALIBRATION_CHANGED,
                    detail="camera calibration digest changed; a new session is required",
                ),
            )
            with self._state_lock:
                self._session.stop(active.observation_session_id)
                self._last_ended_session_id = active.observation_session_id
                self._active_calibration_sha256 = None
                self._last_accepted_capture_unix_ns = 0
            await self._encoded_slot.clear()
            return
        if self._source.localization_epoch_id != active.localization_epoch_id:
            LOGGER.warning(
                "observation localization epoch changed expected=%s actual=%s",
                active.localization_epoch_id,
                self._source.localization_epoch_id,
            )
            await self._write(
                call,
                self._protocol.session_ended(
                    active,
                    reason=pb.SESSION_END_REASON_LOCALIZATION_CHANGED,
                    detail="localization epoch changed; a new observation session is required",
                ),
            )
            with self._state_lock:
                self._session.stop(active.observation_session_id)
                self._last_ended_session_id = active.observation_session_id
                self._active_calibration_sha256 = None
                self._last_accepted_capture_unix_ns = 0
            await self._encoded_slot.clear()
            return
        capture_age_ms = (time.time_ns() - encoded.raw.capture_time_ros.unix_ns) / 1_000_000.0
        if capture_age_ms > self._config.stream.stale_frame_ms:
            self._drops.increment(DropReason.STALE_BEFORE_ENCODE)
            return
        with self._state_lock:
            sequence = self._session.allocate_sequence()
        queue_ms = (time.monotonic_ns() - encoded.encoded_monotonic_ns) / 1_000_000.0
        try:
            message = self._protocol.frame(
                active,
                sequence=sequence,
                encoded=encoded,
                queue_ms=queue_ms,
            )
        except FrameTooLargeError:
            self._drops.increment(DropReason.FRAME_TOO_LARGE)
            LOGGER.warning("dropping encoded observation because frame is too large")
            return
        except FrameValidationError as exc:
            self._drops.increment(DropReason.INVALID_FRAME)
            LOGGER.warning("dropping encoded observation because frame validation failed: %s", exc)
            return
        sent_ns = time.monotonic_ns()
        with self._state_lock:
            self._session.mark_sent(sequence)
        serialized_bytes = message.ByteSize()
        self._inflight = _Inflight(
            observation_session_id=active.observation_session_id,
            sequence=sequence,
            sent_monotonic_ns=sent_ns,
            capture_unix_ns=encoded.raw.capture_time_ros.unix_ns,
            serialized_bytes=serialized_bytes,
        )
        await self._write(call, message)
        self._traffic.observe(serialized_bytes, monotonic_ns=sent_ns)

    async def _write(self, call: Any, message: pb.GatewayToWorkstation) -> None:
        started_ns = time.monotonic_ns()
        try:
            await asyncio.wait_for(
                call.write(message),
                timeout=self._config.transport.write_timeout_ms / 1_000.0,
            )
        except asyncio.TimeoutError as exc:
            self._drops.increment(DropReason.WRITE_TIMEOUT)
            raise _Reconnect("gRPC write blocked beyond the configured timeout") from exc
        self._write_latency.observe((time.monotonic_ns() - started_ns) / 1_000_000.0)

    async def _encode_loop(self) -> None:
        loop = asyncio.get_running_loop()
        next_due = time.monotonic()
        while not self._stop.is_set():
            with self._state_lock:
                active = self._session.active
            if active is None:
                self._source.raw_slot.clear()
                await asyncio.sleep(0.02)
                next_due = time.monotonic()
                continue
            now = time.monotonic()
            if now < next_due:
                await asyncio.sleep(min(next_due - now, 0.02))
                continue
            raw = self._source.raw_slot.pop()
            next_due = now + 1.0 / active.rate_hz
            if raw is None:
                await asyncio.sleep(0.005)
                continue
            capture_age_ms = (time.time_ns() - raw.capture_time_ros.unix_ns) / 1_000_000.0
            if capture_age_ms > self._config.stream.stale_frame_ms:
                self._drops.increment(DropReason.STALE_BEFORE_ENCODE)
                continue
            try:
                encoded = await loop.run_in_executor(self._encoder_pool, self._encoder.encode, raw)
            except FrameTooLargeError:
                self._drops.increment(DropReason.FRAME_TOO_LARGE)
                continue
            except Exception:
                self._drops.increment(DropReason.ENCODE_FAILED)
                LOGGER.exception("RGB-D frame encoding failed")
                continue
            self._encode_latency.observe(encoded.encode_ms)
            if await self._encoded_slot.put(encoded):
                self._drops.increment(DropReason.ENCODED_OVERWRITTEN)
            self._adapt_rate()

    def _adapt_rate(self) -> None:
        traffic = self._traffic.snapshot()
        limit = self._config.stream.max_bandwidth_mbps
        with self._state_lock:
            active = self._session.active
            if active is None:
                return
            if (
                traffic.megabits_per_second > limit
                and active.rate_hz > self._config.stream.minimum_rate_hz
            ):
                target = max(self._config.stream.minimum_rate_hz, active.rate_hz * 0.8)
                self._session.apply_rate_hint(
                    observation_session_id=active.observation_session_id,
                    target_rate_hz=target,
                )
            elif (
                0 < traffic.megabits_per_second < limit * 0.7
                and active.rate_hz < self._config.stream.target_rate_hz
            ):
                target = min(self._config.stream.target_rate_hz, active.rate_hz + 0.25)
                self._session.apply_rate_hint(
                    observation_session_id=active.observation_session_id,
                    target_rate_hz=target,
                )

    async def _telemetry_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1.0)
            with self._state_lock:
                active = self._session.active
                last_sent = self._session.last_sent_sequence
                last_accepted = self._session.last_accepted_sequence
            traffic = self._traffic.snapshot()
            telemetry = self._protocol.telemetry(
                active=active,
                actual_send_fps=traffic.frames_per_second,
                bandwidth_mbps=traffic.megabits_per_second,
                ack_latency=self._ack_latency.snapshot(),
                network_latency=self._network_latency.snapshot(),
                write_blocked_ms=self._write_latency.current,
                last_sent_sequence=last_sent,
                last_accepted_sequence=last_accepted,
            )
            self._enqueue_control(telemetry, required=False)

    async def _lease_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.05)
            now = time.monotonic()
            with self._state_lock:
                last_keepalive = self._session.last_keepalive_monotonic
                active = self._session.active
            if (
                last_keepalive is None
                or (now - last_keepalive) * 1_000
                > self._config.observation_lease.keepalive_timeout_ms
            ):
                if active is not None:
                    self._enqueue_control(
                        self._protocol.session_ended(
                            active,
                            reason=pb.SESSION_END_REASON_KEEPALIVE_EXPIRED,
                            detail="observation keepalive expired",
                        ),
                        required=False,
                    )
                raise _Reconnect("observation keepalive expired")
            inflight = self._inflight
            if (
                inflight is not None
                and (time.monotonic_ns() - inflight.sent_monotonic_ns) / 1_000_000.0
                > self._config.transport.ack_timeout_ms
            ):
                raise _Reconnect("observation ACK timeout")

    def _enqueue_control(
        self,
        message: pb.GatewayToWorkstation,
        *,
        required: bool = True,
    ) -> None:
        queue = self._control_queue
        if queue is None:
            if required:
                raise _Reconnect("control queue is unavailable")
            return
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            if required:
                raise _Reconnect("control message queue is full") from None

    async def _await_source_ready(self) -> None:
        deadline = time.monotonic() + self._config.transport.connect_timeout_ms / 1_000.0
        while time.monotonic() < deadline:
            if (
                self._source.calibration is not None
                and self._source.last_camera_age_ms <= self._config.stream.stale_frame_ms
            ):
                return
            await asyncio.sleep(0.05)

    def _mark_disconnected(self, detail: str) -> None:
        with self._state_lock:
            if self._session.state is not GatewaySessionState.DISCONNECTED:
                previous = self._session.disconnect()
                if previous is not None:
                    self._last_ended_session_id = previous.observation_session_id
            self._active_calibration_sha256 = None
            self._last_accepted_capture_unix_ns = 0
            self._transport_detail = detail
        self._inflight = None
        self._ack_event.set()
