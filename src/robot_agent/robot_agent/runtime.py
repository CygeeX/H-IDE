"""Robot Agent application handshake, routing, safety, and runtime lifecycle."""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, TypeVar, cast
from uuid import UUID, uuid4

from websockets.exceptions import ConnectionClosed

from robot_agent.arbitration import ResourceArbiter
from robot_agent.capabilities import (
    CapabilityManifestWire,
    RobotProfileWire,
    build_capability_manifest,
)
from robot_agent.config import (
    AgentConfig,
    ClientTransportConfig,
    CommandConfig,
    SafeStopConfig,
    ServerTransportConfig,
    StateSourceConfig,
)
from robot_agent.coordinator import (
    CommandCancellationError,
    CommandCoordinator,
    ExecutionResult,
)
from robot_agent.identity import read_robot_boot_id
from robot_agent.ledger import TERMINAL_STATES, CommandIdConflict, CommandLedger, CommandRecord
from robot_agent.observation import (
    ObservationGuard,
    ObservationHealthSample,
    ObservationHealthStatus,
)
from robot_agent.outbound import (
    OutboundItem,
    OutboundPriority,
    PriorityOutboundQueue,
)
from robot_agent.protocol import (
    CommandCancel,
    CommandRequest,
    CommandStatus,
    CommandStatusPayload,
    ControlAcquire,
    ControlStatus,
    ControlStatusPayload,
    ErrorMessage,
    ErrorPayload,
    FrameValidationError,
    LeaseRenew,
    LeaseStatus,
    LeaseStatusPayload,
    SafetyEvent,
    SafetyEventPayload,
    SessionGoodbye,
    SessionHello,
    SessionHelloPayload,
    SessionWelcome,
    StateSnapshot,
    StateSnapshotPayload,
    StateUpdate,
    StateUpdatePayload,
    WireMessage,
    parse_wire_message,
    serialize_wire_message,
)
from robot_agent.ros_adapter import FeedbackCallback, RosAdapter, StateCallback
from robot_agent.session import LeaseError, SessionController, SessionStateError
from robot_agent.state import StateCoalescer, StateEmission
from robot_agent.transport import ClientTransport, ServerTransport

_RETENTION_CHECK_INTERVAL_SECONDS = 3600.0
_PREEMPTION_TIMEOUT_SECONDS = 0.5
_MAX_IN_FLIGHT_COMMANDS = 64
_TaskResult = TypeVar("_TaskResult")
LOGGER = logging.getLogger(__name__)


class RuntimeErrorBase(RuntimeError):
    """Base error for Agent runtime failures."""


class RuntimeSafetyError(RuntimeErrorBase):
    """Raised when the mandatory local safe-stop adapter fails."""


class ConnectionLike(Protocol):
    mode: str

    async def recv_text(self) -> str: ...

    async def send_text(self, text: str) -> None: ...

    async def close(self, *, code: int = 1000, reason: str = "") -> None: ...


class RosRuntimeAdapter(Protocol):
    def set_callbacks(
        self,
        *,
        state_callback: StateCallback,
        feedback_callback: FeedbackCallback,
        observation_health_callback: Any | None = None,
    ) -> None: ...

    def configure_state_sources(self, sources: Mapping[str, StateSourceConfig]) -> None: ...

    def configure_observation(self, *, demand_topic: str, health_topic: str) -> None: ...

    def configure_heartbeat(
        self,
        *,
        topic: str,
        period_ms: int,
        state_supplier: Any,
    ) -> None: ...

    def start(self) -> None: ...

    def check_health(self) -> None: ...

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult: ...

    async def cancel(self, command_id: UUID) -> None: ...

    async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult: ...

    async def publish_observation_demand(
        self,
        *,
        command_id: UUID,
        active: bool,
        observation_session_id: str,
        sensor_id: str,
        scene_version: str,
        localization_epoch_id: str,
    ) -> None: ...

    async def close(self) -> None: ...


class RobotAgentRuntime:
    """One fail-closed runtime shared by WebSocket client and server modes."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        ros_adapter: RosRuntimeAdapter | None = None,
    ) -> None:
        self.config = config
        self._boot_id = read_robot_boot_id()
        self._robot_profile: RobotProfileWire | None = None
        self._capability_manifest: CapabilityManifestWire | None = None
        if config.schema_version == "2":
            self._robot_profile, self._capability_manifest = (
                build_capability_manifest(config)
            )
        self._ros: RosRuntimeAdapter = ros_adapter or RosAdapter(
            robot_boot_id=self._boot_id
        )
        self._ledger = CommandLedger(config.storage.ledger_path)
        self._state = StateCoalescer(
            change_hz=config.state.change_hz,
            snapshot_hz=config.state.snapshot_hz,
        )
        self._session = SessionController(
            lease_timeout=timedelta(milliseconds=config.session.lease_timeout_ms),
            safe_stop=self._perform_safe_stop,
        )
        self._observation_guard = (
            ObservationGuard(
                config.observation,
                robot_id=config.robot_id,
                robot_boot_id=str(self._boot_id),
                publisher=self._ros,
            )
            if config.observation.enabled
            else None
        )
        self._coordinator = CommandCoordinator(
            commands=config.commands,
            ledger=self._ledger,
            arbiter=ResourceArbiter(),
            session=self._session,
            executor=self._ros,
            observation_guard=self._observation_guard,
            status_callback=self._enqueue_command_status,
        )
        self._outbound: PriorityOutboundQueue | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._connection_session_id: UUID | None = None
        self._command_tasks: set[asyncio.Task[None]] = set()
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._detached_safety_tasks: set[asyncio.Task[Any]] = set()
        self._background_failure: asyncio.Future[None] | None = None
        self._initialized = False
        self._closed = False
        self.ready = asyncio.Event()

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._closed:
            raise RuntimeErrorBase("runtime is closed")
        self._event_loop = asyncio.get_running_loop()
        initialized_at = _now()
        await asyncio.to_thread(
            self._ledger.abort_incomplete_after_restart,
            at=initialized_at,
        )
        await self._prune_ledger(at=initialized_at)
        if self._observation_guard is None:
            self._ros.set_callbacks(
                state_callback=self._on_ros_state,
                feedback_callback=self._on_ros_feedback,
            )
        else:
            self._ros.set_callbacks(
                state_callback=self._on_ros_state,
                feedback_callback=self._on_ros_feedback,
                observation_health_callback=self._on_observation_health,
            )
            self._ros.configure_observation(
                demand_topic=self.config.observation.demand_topic,
                health_topic=self.config.observation.health_topic,
            )
        self._ros.configure_state_sources(self.config.state_sources)
        self._ros.configure_heartbeat(
            topic=self.config.safety.heartbeat_topic,
            period_ms=self.config.safety.heartbeat_period_ms,
            state_supplier=lambda: self._session.state.value,
        )
        self._ros.start()
        await self._perform_safe_stop("agent startup")
        self._initialized = True
        self.ready.set()

    async def run(self, stop: asyncio.Event) -> None:
        transport_task: asyncio.Task[None] | None = None
        retention_task: asyncio.Task[None] | None = None
        try:
            await self.initialize()
            transport_config = self.config.transport
            if isinstance(transport_config, ClientTransportConfig):
                transport = ClientTransport(transport_config, self.config.tls)
                transport_task = asyncio.create_task(
                    transport.run(self.handle_connection, stop)
                )
            elif isinstance(transport_config, ServerTransportConfig):
                server = ServerTransport(transport_config, self.config.tls)
                transport_task = asyncio.create_task(
                    server.run(self.handle_connection, stop)
                )
            else:  # pragma: no cover - Pydantic's discriminated union prevents this.
                raise RuntimeErrorBase("unsupported transport configuration")
            retention_task = asyncio.create_task(self._retention_loop())
            done, _ = await asyncio.wait(
                {transport_task, retention_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            completed = next(iter(done))
            await completed
        finally:
            for task in (transport_task, retention_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (transport_task, retention_task) if task is not None),
                return_exceptions=True,
            )
            await self.shutdown()

    async def _retention_loop(self) -> None:
        while True:
            await asyncio.sleep(_RETENTION_CHECK_INTERVAL_SECONDS)
            await self._prune_ledger(at=_now())

    async def _prune_ledger(self, *, at: datetime) -> None:
        await asyncio.to_thread(
            self._ledger.prune_terminal_before,
            before=at - timedelta(days=self.config.storage.retention_days),
        )

    async def handle_connection(self, connection: ConnectionLike) -> None:
        if not self._initialized:
            raise RuntimeErrorBase("runtime must be initialized before accepting a connection")
        hello = SessionHello(
            version="1.0",
            type="session.hello",
            message_id=uuid4(),
            robot_id=self.config.robot_id,
            session_id=None,
            sent_at=_now(),
            payload=SessionHelloPayload(
                agent_version=self.config.agent_version,
                boot_id=self._boot_id,
                supported_versions=["1.0"],
                transport_mode=cast(Any, connection.mode),
                capabilities_digest=(
                    self._capability_manifest.manifestDigest
                    if self._capability_manifest is not None
                    else self.config.capabilities_digest
                ),
                robot_profile=(
                    self._robot_profile.model_dump(mode="json")
                    if self._robot_profile is not None
                    else None
                ),
                capability_manifest=(
                    self._capability_manifest.model_dump(mode="json")
                    if self._capability_manifest is not None
                    else None
                ),
            ),
        )
        await connection.send_text(serialize_wire_message(hello))
        try:
            raw_welcome = await asyncio.wait_for(connection.recv_text(), timeout=5.0)
            welcome = self._parse(raw_welcome)
        except (asyncio.TimeoutError, FrameValidationError) as exc:
            await connection.close(code=1002, reason="invalid application handshake")
            raise RuntimeErrorBase(f"application handshake failed: {exc}") from exc
        if not isinstance(welcome, SessionWelcome):
            await connection.close(code=1002, reason="session.welcome required")
            raise RuntimeErrorBase("first peer message must be session.welcome")
        if (
            welcome.payload.lease_interval_ms != self.config.session.lease_interval_ms
            or welcome.payload.lease_timeout_ms != self.config.session.lease_timeout_ms
            or welcome.payload.control_state != "SAFE_HOLD"
        ):
            await connection.close(code=1002, reason="peer lease policy mismatch")
            raise RuntimeErrorBase(
                "peer lease policy must match the local fail-closed configuration"
            )

        session_id = welcome.session_id
        await self._session.connected(session_id)
        self._connection_session_id = session_id
        queue = PriorityOutboundQueue(maxsize=self.config.state.outbound_queue_size)
        self._outbound = queue
        background_failure = asyncio.get_running_loop().create_future()
        self._background_failure = background_failure
        writer = asyncio.create_task(self._writer(connection, queue))
        lease_monitor = asyncio.create_task(self._lease_monitor())
        state_emitter = asyncio.create_task(self._state_emitter())
        reader = asyncio.create_task(self._reader_loop(connection, session_id))
        disconnect_reason = "websocket disconnected"
        try:
            await self._enqueue_control_status()
            supervised: set[asyncio.Future[Any]] = {
                reader,
                writer,
                lease_monitor,
                state_emitter,
                background_failure,
            }
            done, _ = await asyncio.wait(
                supervised,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if background_failure in done:
                await background_failure
            elif reader in done:
                disconnect_reason = await reader
            else:
                completed = next(
                    task for task in (writer, lease_monitor, state_emitter) if task in done
                )
                await completed
                raise RuntimeErrorBase("connection background task stopped unexpectedly")
        except ConnectionClosed as exc:
            disconnect_reason = f"websocket closed ({exc})"
        finally:
            try:
                await self._session.disconnected(disconnect_reason)
            finally:
                for command_task in tuple(self._command_tasks):
                    if not command_task.done():
                        command_task.cancel()
                if self._command_tasks:
                    await asyncio.gather(*self._command_tasks, return_exceptions=True)
                for event_task in tuple(self._event_tasks):
                    event_task.cancel()
                if self._event_tasks:
                    await asyncio.gather(*self._event_tasks, return_exceptions=True)
                for background_task in (reader, lease_monitor, state_emitter, writer):
                    background_task.cancel()
                await asyncio.gather(
                    reader,
                    lease_monitor,
                    state_emitter,
                    writer,
                    return_exceptions=True,
                )
                if not background_failure.done():
                    background_failure.cancel()
                await asyncio.gather(background_failure, return_exceptions=True)
                self._outbound = None
                self._connection_session_id = None
                if self._background_failure is background_failure:
                    self._background_failure = None

    def _parse(self, raw: str | bytes) -> WireMessage:
        return parse_wire_message(
            raw,
            expected_robot_id=self.config.robot_id,
            received_at=_now(),
            max_message_age=timedelta(milliseconds=self.config.session.max_message_age_ms),
            max_future_skew=timedelta(milliseconds=self.config.session.max_future_skew_ms),
        )

    async def _reader_loop(
        self,
        connection: ConnectionLike,
        session_id: UUID,
    ) -> str:
        while True:
            raw = await connection.recv_text()
            try:
                message = self._parse(raw)
            except FrameValidationError as exc:
                LOGGER.warning("rejecting invalid Robot Agent protocol message: %s", exc)
                await self._enqueue_error(
                    "INVALID_MESSAGE",
                    str(exc),
                    related_message_id=None,
                )
                await connection.close(code=1002, reason="invalid protocol message")
                return "invalid protocol message"
            if message.session_id != session_id:
                await self._enqueue_error(
                    "SESSION_MISMATCH",
                    "message session_id does not own this connection",
                    related_message_id=message.message_id,
                )
                continue
            reason: str | None = None
            try:
                reason = await self._route_message(message, session_id)
            except LeaseError as exc:
                await self._enqueue_error(
                    "INVALID_LEASE",
                    str(exc),
                    related_message_id=message.message_id,
                )
            except SessionStateError as exc:
                await self._enqueue_error(
                    "CONTROL_NOT_ACTIVE",
                    str(exc),
                    related_message_id=message.message_id,
                )
            if reason is not None:
                return reason

    async def _route_message(
        self,
        message: WireMessage,
        session_id: UUID,
    ) -> str | None:
        if isinstance(message, ControlAcquire):
            await self._session.acquire(
                session_id,
                operator_id=message.payload.operator_id,
                at=_now(),
            )
            await self._enqueue_control_status()
            return None
        if isinstance(message, LeaseRenew):
            snapshot = await self._session.renew(
                session_id,
                sequence=message.payload.sequence,
                at=_now(),
            )
            assert snapshot.lease_deadline is not None
            await self._enqueue(
                OutboundPriority.CONTROL,
                "lease.status",
                LeaseStatus(
                    version="1.0",
                    type="lease.status",
                    message_id=uuid4(),
                    robot_id=self.config.robot_id,
                    session_id=session_id,
                    sent_at=_now(),
                    payload=LeaseStatusPayload(
                        sequence=message.payload.sequence,
                        expires_at=snapshot.lease_deadline,
                    ),
                ),
            )
            return None
        if isinstance(message, CommandRequest):
            self._session.assert_can_command(session_id)
            existing = await self._coordinator.record(message.payload.command_id)
            if existing is not None:
                # Route through register so a changed payload still raises the
                # required command-id conflict. Exact duplicates return quickly.
                task = asyncio.create_task(self._execute_and_report(message))
                self._command_tasks.add(task)
                task.add_done_callback(
                    lambda completed: self._tracked_task_completed(
                        completed, self._command_tasks
                    )
                )
                await asyncio.sleep(0)
                return None
            if len(self._command_tasks) >= _MAX_IN_FLIGHT_COMMANDS:
                await self._enqueue_error(
                    "COMMAND_QUEUE_FULL",
                    f"at most {_MAX_IN_FLIGHT_COMMANDS} commands may be in flight",
                    related_message_id=message.message_id,
                )
                return None
            task = asyncio.create_task(self._execute_and_report(message))
            self._command_tasks.add(task)
            task.add_done_callback(
                lambda completed: self._tracked_task_completed(completed, self._command_tasks)
            )
            # Give the coordinator one turn to register the submission before a
            # back-to-back command.cancel can be routed from the same connection.
            await asyncio.sleep(0)
            return None
        if isinstance(message, CommandCancel):
            self._session.assert_can_command(session_id)
            try:
                cancelled = await self._coordinator.cancel(
                    message.payload.command_id,
                    reason=f"remote_operator: {message.payload.reason}",
                    at=_now(),
                )
            except CommandCancellationError as exc:
                await self._perform_safe_stop(str(exc))
                return None
            if not cancelled:
                record = await self._coordinator.record(message.payload.command_id)
                if record is not None and record.state in TERMINAL_STATES:
                    await self._enqueue_command_status(record)
                else:
                    await self._enqueue_error(
                        "COMMAND_NOT_ACTIVE",
                        f"command {message.payload.command_id} is not active",
                        related_message_id=message.message_id,
                    )
            return None
        if isinstance(message, SessionGoodbye):
            return message.payload.reason
        await self._enqueue_error(
            "UNEXPECTED_MESSAGE",
            f"message type {message.type!r} isn't accepted from the peer",
            related_message_id=message.message_id,
        )
        return None

    async def _execute_and_report(self, request: CommandRequest) -> None:
        try:
            record = await self._coordinator.submit(request, at=_now())
        except CommandIdConflict as exc:
            await self._enqueue_error(
                "COMMAND_ID_CONFLICT",
                str(exc),
                related_message_id=request.message_id,
            )
            return
        except CommandCancellationError as exc:
            await self._enqueue_command_status(exc.record)
            await self._perform_safe_stop(str(exc))
            return
        await self._enqueue_command_status(record)

    async def _enqueue_command_status(self, record: CommandRecord) -> None:
        session_id = self._connection_session_id
        if session_id is None:
            return
        await self._enqueue(
            OutboundPriority.COMMAND,
            "command.status",
            CommandStatus(
                version="1.0",
                type="command.status",
                message_id=uuid4(),
                robot_id=self.config.robot_id,
                session_id=session_id,
                sent_at=_now(),
                payload=CommandStatusPayload(
                    command_id=record.command_id,
                    state=cast(Any, record.state.value),
                    detail=record.detail,
                    progress=None,
                    result=record.result,
                ),
            ),
        )

    async def _enqueue_control_status(self) -> None:
        snapshot = self._session.snapshot
        if snapshot.session_id is None:
            return
        await self._enqueue(
            OutboundPriority.CONTROL,
            "control.status",
            ControlStatus(
                version="1.0",
                type="control.status",
                message_id=uuid4(),
                robot_id=self.config.robot_id,
                session_id=snapshot.session_id,
                sent_at=_now(),
                payload=ControlStatusPayload(
                    state=cast(Any, snapshot.state.value),
                    reason=snapshot.reason,
                    lease_sequence=snapshot.lease_sequence,
                ),
            ),
        )

    async def _enqueue_error(
        self,
        code: str,
        message: str,
        *,
        related_message_id: UUID | None,
    ) -> None:
        await self._enqueue(
            OutboundPriority.CONTROL,
            "error",
            ErrorMessage(
                version="1.0",
                type="error",
                message_id=uuid4(),
                robot_id=self.config.robot_id,
                session_id=self._connection_session_id,
                sent_at=_now(),
                payload=ErrorPayload(
                    code=code,
                    message=message[:512],
                    related_message_id=related_message_id,
                    retryable=False,
                ),
            ),
        )

    async def _enqueue_safety(self, code: str, reason: str) -> None:
        session_id = self._connection_session_id
        if session_id is None:
            return
        await self._enqueue(
            OutboundPriority.SAFETY,
            "safety.event",
            SafetyEvent(
                version="1.0",
                type="safety.event",
                message_id=uuid4(),
                robot_id=self.config.robot_id,
                session_id=session_id,
                sent_at=_now(),
                payload=SafetyEventPayload(
                    code=code,
                    state="SAFE_HOLD",
                    reason=reason[:512],
                    latched=True,
                ),
            ),
        )

    async def _enqueue(
        self,
        priority: OutboundPriority,
        kind: str,
        message: WireMessage,
        *,
        coalesce_key: str | None = None,
    ) -> None:
        queue = self._outbound
        if queue is None:
            return
        await queue.put(
            OutboundItem(
                priority=priority,
                kind=kind,
                payload=message,
                coalesce_key=coalesce_key,
            )
        )

    async def _writer(
        self,
        connection: ConnectionLike,
        queue: PriorityOutboundQueue,
    ) -> None:
        while True:
            item = await queue.get()
            await connection.send_text(serialize_wire_message(item.payload))

    async def _lease_monitor(self) -> None:
        while True:
            await asyncio.sleep(0.05)
            self._ros.check_health()
            await self._check_observation_watchdog()
            if await self._session.check_lease(_now()):
                await self._enqueue_control_status()

    async def _state_emitter(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            for emission in self._state.poll(monotonic_now=loop.time()):
                await self._enqueue_state(emission)
            await asyncio.sleep(0.05)

    async def _enqueue_state(self, emission: StateEmission) -> None:
        session_id = self._connection_session_id
        if session_id is None:
            return
        if emission.kind == "snapshot":
            message: WireMessage = StateSnapshot(
                version="1.0",
                type="state.snapshot",
                message_id=uuid4(),
                robot_id=self.config.robot_id,
                session_id=session_id,
                sent_at=_now(),
                payload=StateSnapshotPayload(
                    sequence=emission.sequence,
                    observed_at=emission.observed_at,
                    state=emission.data,
                ),
            )
            priority = OutboundPriority.SNAPSHOT
            key = "state.snapshot"
        else:
            message = StateUpdate(
                version="1.0",
                type="state.update",
                message_id=uuid4(),
                robot_id=self.config.robot_id,
                session_id=session_id,
                sent_at=_now(),
                payload=StateUpdatePayload(
                    sequence=emission.sequence,
                    observed_at=emission.observed_at,
                    changes=emission.data,
                ),
            )
            priority = OutboundPriority.STATE
            key = "state.update"
        await self._enqueue(priority, message.type, message, coalesce_key=key)

    async def _perform_safe_stop(self, reason: str) -> None:
        if self._detached_safety_tasks:
            detail = "a prior safety operation ignored cancellation and is still running"
            await self._enqueue_safety("SAFE_STOP_FAILED", f"{reason}: {detail}")
            raise RuntimeSafetyError(detail)
        timeout_seconds = self.config.safety.safe_stop.timeout_ms / 1000.0
        physical_stop = asyncio.create_task(self._ros.safe_stop(self.config.safety.safe_stop))
        preemption = asyncio.create_task(self._coordinator.preempt_all(reason=reason, at=_now()))
        (result, physical_error), (_, preemption_error) = await asyncio.gather(
            _capture_bounded_task(
                physical_stop,
                timeout=timeout_seconds,
                description="physical safe stop",
                detached_tasks=self._detached_safety_tasks,
            ),
            _capture_bounded_task(
                preemption,
                timeout=_PREEMPTION_TIMEOUT_SECONDS,
                description="command preemption",
                detached_tasks=self._detached_safety_tasks,
            ),
        )
        if physical_error is not None:
            await self._enqueue_safety("SAFE_STOP_FAILED", f"{reason}: {physical_error}")
            raise RuntimeSafetyError(
                f"mandatory safe stop failed: {physical_error}"
            ) from physical_error
        assert result is not None
        if not result.succeeded:
            await self._enqueue_safety(
                "SAFE_STOP_FAILED",
                f"{reason}: {result.detail}",
            )
            raise RuntimeSafetyError(f"mandatory safe stop was rejected: {result.detail}")
        if preemption_error is not None:
            await self._enqueue_safety(
                "SAFE_STOP_FAILED",
                f"{reason}: command preemption failed: {preemption_error}",
            )
            raise RuntimeSafetyError(
                f"command preemption failed during mandatory safe stop: {preemption_error}"
            ) from preemption_error
        await self._enqueue_safety(_safety_code(reason), reason)

    def _on_ros_state(self, path: str, value: Any, observed_at: datetime) -> None:
        self._state.update(path, value, observed_at=observed_at)

    def _on_ros_feedback(self, command_id: UUID, feedback: dict[str, Any]) -> None:
        loop = self._event_loop
        if loop is None or loop.is_closed() or self._connection_session_id is None:
            return
        loop.call_soon_threadsafe(self._schedule_feedback, command_id, feedback)

    def _on_observation_health(self, sample: ObservationHealthSample) -> None:
        loop = self._event_loop
        if loop is None or loop.is_closed() or self._observation_guard is None:
            return
        loop.call_soon_threadsafe(self._accept_observation_health, sample)

    def _accept_observation_health(self, sample: ObservationHealthSample) -> None:
        guard = self._observation_guard
        loop = self._event_loop
        if guard is None or loop is None:
            return
        previous = guard.current_health()
        guard.update_health(sample, monotonic_now=loop.time())
        current = guard.current_health()
        if current is not sample or current is previous:
            return
        status_name = (
            "USABLE"
            if sample.status == ObservationHealthStatus.USABLE
            else "DEGRADED"
            if sample.status == ObservationHealthStatus.DEGRADED
            else "NOT_USABLE"
        )
        observed_at = _now()
        for path, value in (
            ("observation.status", status_name),
            ("observation.sensor_id", sample.sensor_id),
            ("observation.session_id", sample.observation_session_id),
            ("observation.scene_version", sample.scene_version),
            ("observation.localization_epoch_id", sample.localization_epoch_id),
            (
                "observation.ack_age_ms",
                sample.ack_age_ms if math.isfinite(sample.ack_age_ms) else None,
            ),
            (
                "observation.capture_age_ms",
                sample.capture_age_ms if math.isfinite(sample.capture_age_ms) else None,
            ),
            ("observation.drop_count", sample.drop_count),
        ):
            self._state.update(path, value, observed_at=observed_at)

    async def _check_observation_watchdog(self) -> None:
        guard = self._observation_guard
        loop = self._event_loop
        if guard is None or loop is None:
            return
        for binding, detail in guard.stale_bindings(monotonic_now=loop.time()):
            reason = f"observation_watchdog: {detail}"
            try:
                await self._coordinator.cancel(
                    binding.command_id,
                    reason=reason,
                    at=_now(),
                )
            except CommandCancellationError as exc:
                await self._perform_safe_stop(str(exc))

    def _schedule_feedback(self, command_id: UUID, feedback: dict[str, Any]) -> None:
        session_id = self._connection_session_id
        if session_id is None:
            return
        message = CommandStatus(
            version="1.0",
            type="command.status",
            message_id=uuid4(),
            robot_id=self.config.robot_id,
            session_id=session_id,
            sent_at=_now(),
            payload=CommandStatusPayload(
                command_id=command_id,
                state="RUNNING",
                detail="action feedback",
                progress=None,
                result=feedback,
            ),
        )
        task = asyncio.create_task(
            self._enqueue(
                OutboundPriority.COMMAND,
                "command.status",
                message,
                coalesce_key=f"command.feedback.{command_id}",
            )
        )
        self._event_tasks.add(task)
        task.add_done_callback(
            lambda completed: self._tracked_task_completed(completed, self._event_tasks)
        )

    def _tracked_task_completed(
        self,
        task: asyncio.Task[None],
        collection: set[asyncio.Task[None]],
    ) -> None:
        collection.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        failure = self._background_failure
        if exception is not None and failure is not None and not failure.done():
            failure.set_exception(exception)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._ros.close()
        await asyncio.to_thread(self._ledger.close)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safety_code(reason: str) -> str:
    code = re.sub(r"[^A-Z0-9]+", "_", reason.upper()).strip("_")
    return (code or "SAFE_STOP")[:128]


async def _bounded_task_result(
    task: asyncio.Task[_TaskResult],
    *,
    timeout: float,
    description: str,
    detached_tasks: set[asyncio.Task[Any]],
) -> _TaskResult:
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
    except asyncio.CancelledError:
        _cancel_and_track(task, detached_tasks)
        raise
    if task not in done:
        _cancel_and_track(task, detached_tasks)
        raise asyncio.TimeoutError(f"{description} exceeded {timeout:.3f}s deadline")
    if task.cancelled():
        raise RuntimeError(f"{description} was cancelled")
    return task.result()


async def _capture_bounded_task(
    task: asyncio.Task[_TaskResult],
    *,
    timeout: float,
    description: str,
    detached_tasks: set[asyncio.Task[Any]],
) -> tuple[_TaskResult | None, Exception | None]:
    try:
        return (
            await _bounded_task_result(
                task,
                timeout=timeout,
                description=description,
                detached_tasks=detached_tasks,
            ),
            None,
        )
    except Exception as exc:
        return None, exc


def _cancel_and_track(
    task: asyncio.Task[Any],
    detached_tasks: set[asyncio.Task[Any]],
) -> None:
    if not task.done():
        task.cancel()

    detached_tasks.add(task)

    def completed(completed_task: asyncio.Task[Any]) -> None:
        detached_tasks.discard(completed_task)
        _consume_task_outcome(completed_task)

    task.add_done_callback(completed)


def _consume_task_outcome(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()
