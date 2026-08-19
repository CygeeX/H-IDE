"""Robot Agent application handshake, routing, safety, and runtime lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import re
import threading
import time
from collections.abc import Iterator, Mapping
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
    MotionSafetyFaultConfig,
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
from robot_agent.ledger import (
    TERMINAL_STATES,
    CommandIdConflict,
    CommandLedger,
    CommandRecord,
    SafeStopAuditState,
)
from robot_agent.motion_fault import (
    DeduplicationDecision,
    MotionFaultDeduplicator,
    MotionSafetyFault,
)
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
from robot_agent.ros_adapter import (
    FeedbackCallback,
    MotionFaultCancelEvidence,
    MotionFaultCancellationError,
    MotionSafetyFaultCallback,
    RosActionReadiness,
    RosAdapter,
    StateCallback,
)
from robot_agent.session import (
    ControlState,
    LeaseError,
    SafeStopState,
    SessionController,
    SessionStateError,
)
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

    def configure_motion_safety_fault(
        self,
        *,
        config: MotionSafetyFaultConfig,
        callback: MotionSafetyFaultCallback,
    ) -> None: ...

    def configure_heartbeat(
        self,
        *,
        topic: str,
        period_ms: int,
        state_supplier: Any,
    ) -> None: ...

    def start(self) -> None: ...

    def check_health(self) -> None: ...

    async def nav2_action_readiness(
        self,
        commands: Mapping[str, CommandConfig],
    ) -> dict[str, RosActionReadiness]: ...

    async def safety_endpoint_readiness(
        self,
        endpoints: Mapping[str, SafeStopConfig],
    ) -> dict[str, RosActionReadiness]: ...

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult: ...

    async def cancel(self, command_id: UUID) -> None: ...

    async def cancel_nav2_for_motion_fault(
        self,
        command_id: UUID,
        *,
        received_monotonic_ns: int,
        cancel_begin_deadline_ms: int = 100,
        nav2_terminal_deadline_ms: int = 300,
    ) -> MotionFaultCancelEvidence: ...

    async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult: ...

    async def control_arm(self, config: SafeStopConfig) -> ExecutionResult: ...

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
            self._robot_profile, self._capability_manifest = build_capability_manifest(config)
        self._ros: RosRuntimeAdapter = ros_adapter or RosAdapter(robot_boot_id=self._boot_id)
        self._ledger = CommandLedger(config.storage.ledger_path)
        self._state = StateCoalescer(
            change_hz=config.state.change_hz,
            snapshot_hz=config.state.snapshot_hz,
        )
        self._session = SessionController(
            lease_timeout=timedelta(milliseconds=config.session.lease_timeout_ms),
            safe_stop=self._perform_safe_stop,
            initial_safe_stop_epoch=self._ledger.latest_safe_stop_epoch,
        )
        # Fault ingress runs on the ROS owner thread while command dispatch runs
        # on asyncio.  This one lock makes latch-vs-dispatch ordering explicit:
        # an accepted fault either blocks the final dispatch gate, or observes
        # the command already registered as the exact locally owned execution.
        self._motion_fault_lock = threading.RLock()
        self._motion_fault_ingress_latched = threading.Event()
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
            dispatch_boundary=self._motion_fault_dispatch_boundary,
            dispatch_interlock=self._assert_motion_fault_not_latched,
        )
        self._outbound: PriorityOutboundQueue | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._connection_session_id: UUID | None = None
        self._command_tasks: set[asyncio.Task[None]] = set()
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._motion_fault_tasks: set[asyncio.Task[None]] = set()
        self._detached_safety_tasks: set[asyncio.Task[Any]] = set()
        self._detached_control_arm_tasks: set[asyncio.Task[Any]] = set()
        self._background_failure: asyncio.Future[None] | None = None
        self._diagnostics_lock = asyncio.Lock()
        self._state_freshness_lock = threading.RLock()
        self._state_received_monotonic: dict[str, float] = {}
        self._motion_fault_deduplicator = MotionFaultDeduplicator()
        self._motion_fault_latest: MotionSafetyFault | None = None
        self._motion_fault_latest_evidence: MotionFaultCancelEvidence | None = None
        self._motion_fault_latest_outcome: str | None = None
        self._motion_fault_latest_detail: str | None = None
        self._motion_fault_latest_safe_stop_epoch: int | None = None
        self._motion_fault_counters: dict[str, int] = {
            "accepted": 0,
            "invalid": 0,
            "wrongRobot": 0,
            "wrongEpoch": 0,
            "duplicate": 0,
            "staleSequence": 0,
            "sequenceConflict": 0,
        }
        self._motion_fault_handling_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        self.ready = asyncio.Event()

    async def initialize(self) -> None:
        if self._initialized:
            return
        if self._closed:
            raise RuntimeErrorBase("runtime is closed")
        self._event_loop = asyncio.get_running_loop()
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
        if self.config.safety.motion_safety_fault.enabled:
            self._ros.configure_motion_safety_fault(
                config=self.config.safety.motion_safety_fault,
                callback=self._on_motion_safety_fault,
            )
        self._ros.configure_heartbeat(
            topic=self.config.safety.heartbeat_topic,
            period_ms=self.config.safety.heartbeat_period_ms,
            state_supplier=lambda: self._session.state.value,
        )
        self._ros.start()
        # A new process must establish a fresh physical boundary before old
        # non-terminal commands are made terminal by restart recovery.
        await self._session.ensure_safe_hold("agent startup", refresh=True)
        initialized_at = _now()
        await asyncio.to_thread(
            self._ledger.abort_incomplete_after_restart,
            at=initialized_at,
        )
        await self._prune_ledger(at=initialized_at)
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
                transport_task = asyncio.create_task(transport.run(self.handle_connection, stop))
            elif isinstance(transport_config, ServerTransportConfig):
                server = ServerTransport(transport_config, self.config.tls)
                transport_task = asyncio.create_task(server.run(self.handle_connection, stop))
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
            self._assert_motion_fault_not_latched()
            try:
                if self._session.state is not ControlState.ACTIVE:
                    await self._assert_downstream_ready_for_acquire()
                await self._session.acquire(
                    session_id,
                    operator_id=message.payload.operator_id,
                    at=_now(),
                    before_activate=(
                        self._perform_control_arm
                        if self.config.safety.control_arm is not None
                        else None
                    ),
                )
            except Exception:
                await self._enqueue_control_status()
                raise
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
            self._assert_motion_fault_not_latched()
            self._session.assert_can_command(session_id)
            existing = await self._coordinator.record(message.payload.command_id)
            if existing is not None:
                # Route through register so a changed payload still raises the
                # required command-id conflict. Exact duplicates return quickly.
                task = asyncio.create_task(self._execute_and_report(message))
                self._command_tasks.add(task)
                task.add_done_callback(
                    lambda completed: self._tracked_task_completed(completed, self._command_tasks)
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
            self._assert_motion_fault_not_latched()
            self._session.assert_can_command(session_id)
            if not self._coordinator.is_command_active(message.payload.command_id):
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
            cancel_reason = (
                f"remote operator cancelled command {message.payload.command_id}"
            )
            await self._revoke_authority_for_safe_stop(cancel_reason)
            try:
                cancelled = await self._coordinator.cancel(
                    message.payload.command_id,
                    reason=f"remote_operator: {message.payload.reason}",
                    at=_now(),
                    defer_terminal_until_safe_stop=True,
                )
            except CommandCancellationError as exc:
                LOGGER.error("command cancellation was not acknowledged: %s", exc)
                await self._enter_safe_hold_after_cancel(cancel_reason)
                return None
            if cancelled:
                await self._enter_safe_hold_after_cancel(cancel_reason)
            else:
                # The command may have completed between the active preflight
                # and cancellation.  Authority was already revoked, so finish
                # that requested physical boundary before reporting anything.
                await self._enter_safe_hold_after_cancel(cancel_reason)
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
            await self._enter_safe_hold_after_cancel(str(exc))
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
                    authority_revoked=snapshot.authority_revoked,
                    safe_stop_state=snapshot.safe_stop_state.value,
                    safe_stop_epoch=snapshot.safe_stop_epoch,
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

    async def _enqueue_safety(
        self,
        code: str,
        reason: str,
        *,
        safe_stop_state: SafeStopState | None = None,
        safe_stop_epoch: int | None = None,
    ) -> None:
        session_id = self._connection_session_id
        if session_id is None:
            return
        snapshot = self._session.snapshot
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
                    authority_revoked=True,
                    safe_stop_state=(
                        safe_stop_state or snapshot.safe_stop_state
                    ).value,
                    safe_stop_epoch=(
                        snapshot.safe_stop_epoch
                        if safe_stop_epoch is None
                        else safe_stop_epoch
                    ),
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
        snapshot = self._session.snapshot
        epoch = snapshot.safe_stop_epoch
        if snapshot.safe_stop_state is not SafeStopState.REQUESTED:
            # Kept only for direct diagnostic/test invocation. Runtime safety
            # paths first revoke authority through SessionController.
            epoch = max(epoch, self._ledger.latest_safe_stop_epoch) + 1
        audit_error: Exception | None = None
        try:
            await asyncio.to_thread(
                self._ledger.begin_safe_stop,
                epoch=epoch,
                reason=reason,
                at=_now(),
            )
        except Exception as exc:
            # Storage truth is mandatory for reacquire, but stopping the robot
            # is more urgent than recording why. Continue to the physical path
            # and fail closed after it has been exercised.
            audit_error = exc
        if self._detached_safety_tasks:
            detail = "a prior safety operation ignored cancellation and is still running"
            await self._record_safe_stop_failure(epoch, reason, detail)
            raise RuntimeSafetyError(detail)
        timeout_seconds = self.config.safety.safe_stop.timeout_ms / 1000.0
        physical_stop = asyncio.create_task(self._ros.safe_stop(self.config.safety.safe_stop))
        preemption = asyncio.create_task(self._coordinator.preempt_all(reason=reason, at=_now()))
        try:
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
        except asyncio.CancelledError:
            await self._record_safe_stop_failure(
                epoch,
                reason,
                "mandatory safe-stop operation was cancelled",
            )
            raise
        if physical_error is not None:
            await self._record_safe_stop_failure(
                epoch,
                reason,
                str(physical_error),
            )
            raise RuntimeSafetyError(
                f"mandatory safe stop failed: {physical_error}"
            ) from physical_error
        assert result is not None
        if not result.succeeded:
            await self._record_safe_stop_failure(epoch, reason, result.detail)
            raise RuntimeSafetyError(f"mandatory safe stop was rejected: {result.detail}")
        if preemption_error is not None:
            detail = f"command preemption failed: {preemption_error}"
            await self._record_safe_stop_failure(
                epoch,
                reason,
                detail,
            )
            raise RuntimeSafetyError(
                f"command preemption failed during mandatory safe stop: {preemption_error}"
            ) from preemption_error
        try:
            await self._coordinator.confirm_safe_stop(epoch=epoch, at=_now())
        except Exception as exc:
            detail = f"cancel terminalization failed: {type(exc).__name__}: {exc}"
            await self._record_safe_stop_failure(epoch, reason, detail)
            raise RuntimeSafetyError(detail) from exc
        if audit_error is not None:
            detail = (
                "physical safe-stop confirmed but audit request was not durable: "
                f"{type(audit_error).__name__}: {audit_error}"
            )
            await self._record_safe_stop_failure(epoch, reason, detail)
            raise RuntimeSafetyError(detail) from audit_error
        try:
            await asyncio.to_thread(
                self._ledger.finish_safe_stop,
                epoch=epoch,
                state=SafeStopAuditState.CONFIRMED,
                detail=result.detail,
                at=_now(),
            )
        except Exception as exc:
            detail = (
                "physical safe-stop confirmed but audit completion failed: "
                f"{type(exc).__name__}: {exc}"
            )
            await self._record_safe_stop_failure(epoch, reason, detail)
            raise RuntimeSafetyError(detail) from exc
        await self._enqueue_safety(
            _safety_code(reason),
            reason,
            safe_stop_state=SafeStopState.CONFIRMED,
            safe_stop_epoch=epoch,
        )

    async def _record_safe_stop_failure(
        self,
        epoch: int,
        reason: str,
        detail: str,
    ) -> None:
        try:
            await asyncio.to_thread(
                self._ledger.finish_safe_stop,
                epoch=epoch,
                state=SafeStopAuditState.FAILED,
                detail=detail,
                at=_now(),
            )
        except Exception as exc:
            LOGGER.error(
                "failed to persist safe-stop failure epoch %s: %s",
                epoch,
                exc,
            )
            detail = f"{detail}; audit failure: {type(exc).__name__}: {exc}"
        await self._enqueue_safety(
            "SAFE_STOP_FAILED",
            f"{reason}: {detail}",
            safe_stop_state=SafeStopState.FAILED,
            safe_stop_epoch=epoch,
        )

    async def _enter_safe_hold_after_cancel(self, reason: str) -> None:
        """A cancellation revokes the current control arm until reacquired."""

        try:
            await self._session.ensure_safe_hold(reason, refresh=True)
        finally:
            # A failed physical stop must be visible as failed, never as a
            # confirmed SAFE_HOLD.  The session has already revoked authority.
            await self._enqueue_control_status()

    async def _revoke_authority_for_safe_stop(
        self,
        reason: str,
        *,
        latch_fault: bool = False,
    ) -> None:
        revoked = (
            await self._session.latch_safety_fault(reason)
            if latch_fault
            else await self._session.revoke_authority(reason)
        )
        try:
            await asyncio.to_thread(
                self._ledger.begin_safe_stop,
                epoch=revoked.safe_stop_epoch,
                reason=reason,
                at=_now(),
            )
        except Exception as exc:
            # Never delay authority revocation or the physical stop for an
            # audit write. _perform_safe_stop retries and then fails closed
            # after exercising the physical adapter if storage stays down.
            LOGGER.error(
                "failed to persist safe-stop request epoch %s: %s",
                revoked.safe_stop_epoch,
                exc,
            )
        # Advertise authority revocation immediately, while truthfully
        # retaining safe_stop_state=requested until the physical boundary.
        await self._enqueue_control_status()

    async def _perform_control_arm(self) -> None:
        """Pair remote control.acquire with one local measured arm boundary."""

        config = self.config.safety.control_arm
        if config is None:
            return
        if self._coordinator.active_commands:
            raise SessionStateError(
                "control arm requires zero active local commands"
            )
        timeout_seconds = config.timeout_ms / 1000.0
        operation = asyncio.create_task(self._ros.control_arm(config))
        result, error = await _capture_bounded_task(
            operation,
            timeout=timeout_seconds,
            description="local control arm",
            # A timed-out arm may still be unwinding inside the ROS executor,
            # but it must never suppress the stronger safe-stop used to
            # re-latch the Plant after a lost acknowledgement.
            detached_tasks=self._detached_control_arm_tasks,
        )
        if error is None and result is not None and result.succeeded:
            return
        detail = (
            str(error)
            if error is not None
            else result.detail
            if result is not None
            else "local control arm returned no result"
        )
        raise SessionStateError(f"control arm failed closed: {detail}")

    def _on_ros_state(self, path: str, value: Any, observed_at: datetime) -> None:
        self._state.update(path, value, observed_at=observed_at)
        with self._state_freshness_lock:
            self._state_received_monotonic[path] = time.monotonic()

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

    def _on_motion_safety_fault(
        self,
        raw: str,
        received_monotonic_ns: int,
    ) -> None:
        """Strictly isolate and deduplicate a ROS-thread fault ingress."""

        try:
            fault = MotionSafetyFault.from_json(raw)
        except ValueError as exc:
            with self._motion_fault_lock:
                self._motion_fault_counters["invalid"] += 1
            LOGGER.warning("rejecting invalid motion-safety-fault event: %s", exc)
            return
        config = self.config.safety.motion_safety_fault
        with self._motion_fault_lock:
            if fault.robot_id != self.config.robot_id:
                self._motion_fault_counters["wrongRobot"] += 1
                return
            if fault.localization_epoch_id != config.localization_epoch_id:
                self._motion_fault_counters["wrongEpoch"] += 1
                return
            decision = self._motion_fault_deduplicator.observe(fault)
            counter = {
                DeduplicationDecision.DUPLICATE: "duplicate",
                DeduplicationDecision.STALE_SEQUENCE: "staleSequence",
                DeduplicationDecision.SEQUENCE_CONFLICT: "sequenceConflict",
            }.get(decision)
            if counter is not None:
                self._motion_fault_counters[counter] += 1
                return
            self._motion_fault_counters["accepted"] += 1
            self._motion_fault_latest = fault
            # This thread-safe latch closes the small interval before the
            # asyncio SessionController can durably revoke authority.
            self._motion_fault_ingress_latched.set()
        loop = self._event_loop
        if loop is None or loop.is_closed():
            LOGGER.critical(
                "accepted motion-safety-fault without a live runtime event loop"
            )
            return
        loop.call_soon_threadsafe(
            self._schedule_motion_safety_fault,
            fault,
            received_monotonic_ns,
        )

    def _schedule_motion_safety_fault(
        self,
        fault: MotionSafetyFault,
        received_monotonic_ns: int,
    ) -> None:
        task = asyncio.create_task(
            self._handle_motion_safety_fault(fault, received_monotonic_ns)
        )
        self._motion_fault_tasks.add(task)
        task.add_done_callback(self._motion_fault_task_completed)

    def _motion_fault_task_completed(self, task: asyncio.Task[None]) -> None:
        self._motion_fault_tasks.discard(task)
        if task.cancelled():
            LOGGER.critical("motion-safety-fault handling was cancelled")
            return
        exception = task.exception()
        if exception is not None:
            LOGGER.critical(
                "motion-safety-fault handling failed closed: %s",
                exception,
                exc_info=exception,
            )

    async def _handle_motion_safety_fault(
        self,
        fault: MotionSafetyFault,
        received_monotonic_ns: int,
    ) -> None:
        async with self._motion_fault_handling_lock:
            reason = (
                f"motion_safety_fault[{fault.event_id}] "
                f"{fault.source}/{fault.code}: {fault.reason}"
            )
            received_at = _now()
            audit_task = asyncio.create_task(
                asyncio.to_thread(
                    self._ledger.begin_motion_fault,
                    fault=fault,
                    received_monotonic_ns=received_monotonic_ns,
                    at=received_at,
                )
            )
            revoke_task = asyncio.create_task(
                self._revoke_authority_for_safe_stop(reason, latch_fault=True)
            )

            command_ids = self._coordinator.active_nav2_commands
            command_id = next(iter(command_ids)) if len(command_ids) == 1 else None
            cancel_task: asyncio.Task[MotionFaultCancelEvidence] | None = None
            coordinator_task: asyncio.Task[bool] | None = None
            outcome = "no_active_nav2_goal"
            detail = "fault latched with no active Nav2 command"
            evidence: MotionFaultCancelEvidence | None = None
            if command_id is not None:
                fault_config = self.config.safety.motion_safety_fault
                cancel_task = asyncio.create_task(
                    self._ros.cancel_nav2_for_motion_fault(
                        command_id,
                        received_monotonic_ns=received_monotonic_ns,
                        cancel_begin_deadline_ms=fault_config.cancel_begin_deadline_ms,
                        nav2_terminal_deadline_ms=fault_config.nav2_terminal_deadline_ms,
                    )
                )
                prepared = self._coordinator.prepare_safe_stop_cancel(
                    command_id,
                    reason=reason,
                    at=received_at,
                    defer_terminal_until_safe_stop=True,
                    external_cancel_task=cancel_task,
                )
                if prepared:
                    coordinator_task = asyncio.create_task(
                        self._coordinator.cancel(
                            command_id,
                            reason=reason,
                            at=received_at,
                            defer_terminal_until_safe_stop=True,
                            external_cancel_task=cancel_task,
                        )
                    )
            elif len(command_ids) > 1:
                outcome = "ambiguous_nav2_ownership"
                detail = (
                    "multiple locally active Nav2 commands; refused any broad ROS cancel"
                )

            try:
                if coordinator_task is not None:
                    await coordinator_task
                if cancel_task is not None:
                    evidence = await cancel_task
                    outcome = evidence.outcome
                    detail = "exact command-owned Nav2 goal reached terminal"
            except MotionFaultCancellationError as exc:
                evidence = exc.evidence
                outcome = evidence.outcome
                detail = str(exc)
            except CommandCancellationError as exc:
                if cancel_task is not None and cancel_task.done():
                    try:
                        evidence = cancel_task.result()
                    except MotionFaultCancellationError as cancel_exc:
                        evidence = cancel_exc.evidence
                    except Exception:
                        pass
                outcome = (
                    evidence.outcome
                    if evidence is not None
                    else "command_cancellation_unacknowledged"
                )
                detail = str(exc)
            except Exception as exc:
                if isinstance(exc, MotionFaultCancellationError):
                    evidence = exc.evidence
                outcome = "fault_cancel_failed"
                detail = f"{type(exc).__name__}: {exc}"

            # Authority revocation and its epoch audit run concurrently with
            # the 300 ms cancellation boundary, never in front of it.
            with contextlib.suppress(Exception):
                await revoke_task
            safe_stop_epoch = self._session.snapshot.safe_stop_epoch
            with self._motion_fault_lock:
                self._motion_fault_latest_evidence = evidence
                self._motion_fault_latest_outcome = outcome
                self._motion_fault_latest_detail = detail[:1024]
                self._motion_fault_latest_safe_stop_epoch = safe_stop_epoch
            try:
                await self._enter_safe_hold_after_cancel(reason)
            except Exception as exc:
                outcome = f"{outcome}:safe_stop_failed"
                detail = f"{detail}; physical safe-stop failed: {type(exc).__name__}: {exc}"

            with self._motion_fault_lock:
                self._motion_fault_latest_evidence = evidence
                self._motion_fault_latest_outcome = outcome
                self._motion_fault_latest_detail = detail[:1024]
                self._motion_fault_latest_safe_stop_epoch = safe_stop_epoch

            try:
                audit = await audit_task
                await asyncio.to_thread(
                    self._ledger.finish_motion_fault,
                    audit.handling_id,
                    safe_stop_epoch=safe_stop_epoch,
                    command_id=command_id,
                    cancel_begin_monotonic_ns=(
                        None if evidence is None else evidence.cancel_begin_monotonic_ns
                    ),
                    ros_cancel_call_monotonic_ns=(
                        None if evidence is None else evidence.ros_cancel_call_monotonic_ns
                    ),
                    nav2_terminal_monotonic_ns=(
                        None if evidence is None else evidence.nav2_terminal_monotonic_ns
                    ),
                    outcome=outcome,
                    detail=detail,
                    at=_now(),
                )
            except Exception as exc:
                LOGGER.error("failed to persist motion-fault correlation: %s", exc)

    def _assert_motion_fault_not_latched(self) -> None:
        if self._motion_fault_ingress_latched.is_set():
            raise SessionStateError(
                "control blocked: a motion-safety fault is latched"
            )

    @contextlib.contextmanager
    def _motion_fault_dispatch_boundary(self) -> Iterator[None]:
        """Serialize final local dispatch against ROS-thread fault ingress."""

        with self._motion_fault_lock:
            yield

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
            if not self._coordinator.is_command_active(binding.command_id):
                continue
            await self._revoke_authority_for_safe_stop(reason)
            try:
                cancelled = await self._coordinator.cancel(
                    binding.command_id,
                    reason=reason,
                    at=_now(),
                    defer_terminal_until_safe_stop=True,
                )
            except CommandCancellationError as exc:
                LOGGER.error("watchdog cancellation was not acknowledged: %s", exc)
                await self._enter_safe_hold_after_cancel(reason)
            else:
                # A race to command completion does not restore authority once
                # this safety boundary has been requested.
                if not cancelled:
                    LOGGER.info(
                        "watchdog command %s completed while cancellation raced",
                        binding.command_id,
                    )
                await self._enter_safe_hold_after_cancel(reason)

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
        self.ready.clear()
        if self._motion_fault_tasks:
            await asyncio.gather(
                *tuple(self._motion_fault_tasks),
                return_exceptions=True,
            )
        await self._ros.close()
        await asyncio.to_thread(self._ledger.close)

    async def diagnostics_snapshot(self) -> dict[str, Any]:
        """Build one fail-closed, redacted view for the local diagnostics API."""
        async with self._diagnostics_lock:
            return await self._diagnostics_snapshot_locked()

    async def _assert_downstream_ready_for_acquire(self) -> None:
        diagnostics = await self.diagnostics_snapshot()
        blockers = [
            name
            for name, check in diagnostics["checks"].items()
            if name not in {"transportSession", "control", "safeStop"}
            and bool(check["required"])
            and not bool(check["ready"])
        ]
        if blockers:
            raise SessionStateError(
                "control blocked: downstream readiness unavailable: "
                + ", ".join(sorted(blockers))
            )

    async def _diagnostics_snapshot_locked(self) -> dict[str, Any]:
        probe_lifecycle_ready = self._initialized and not self._closed
        ros_ready = False
        ros_reason: str | None = "runtime_not_initialized"
        if probe_lifecycle_ready:
            try:
                self._ros.check_health()
            except Exception as exc:
                ros_reason = f"ros_executor_unavailable:{type(exc).__name__}"
            else:
                ros_ready = True
                ros_reason = None

        nav2_commands = {
            skill_id: command
            for skill_id, command in self.config.commands.items()
            if command.goal_transform in {"navigate_to_pose_v1", "nav2_spin_v2"}
        }
        nav2_results: dict[str, RosActionReadiness] = {}
        nav2_probe_reason: str | None = None
        if nav2_commands and ros_ready:
            try:
                nav2_results = await asyncio.wait_for(
                    self._ros.nav2_action_readiness(nav2_commands),
                    timeout=self.config.diagnostics.probe_timeout_ms / 1_000.0,
                )
            except asyncio.TimeoutError:
                nav2_probe_reason = "nav2_probe_timed_out"
            except Exception as exc:
                nav2_probe_reason = f"nav2_probe_failed:{type(exc).__name__}"
        elif nav2_commands:
            nav2_probe_reason = "ros_executor_unavailable"
        configured_safety_endpoints = {
            "safeStop": self.config.safety.safe_stop,
            **(
                {}
                if self.config.safety.control_arm is None
                else {"controlArm": self.config.safety.control_arm}
            ),
        }
        safety_endpoint_results: dict[str, RosActionReadiness] = {}
        safety_endpoint_probe_reason: str | None = None
        if ros_ready:
            try:
                safety_endpoint_results = await asyncio.wait_for(
                    self._ros.safety_endpoint_readiness(configured_safety_endpoints),
                    timeout=self.config.diagnostics.probe_timeout_ms / 1_000.0,
                )
            except asyncio.TimeoutError:
                safety_endpoint_probe_reason = "safety_endpoint_probe_timed_out"
            except Exception as exc:
                safety_endpoint_probe_reason = (
                    f"safety_endpoint_probe_failed:{type(exc).__name__}"
                )
        else:
            safety_endpoint_probe_reason = "ros_executor_unavailable"
        observed_at = _now()
        monotonic_now = time.monotonic()
        lifecycle_ready = self._initialized and not self._closed
        session_snapshot = self._session.snapshot
        with self._motion_fault_lock:
            motion_fault_latched = self._motion_fault_ingress_latched.is_set()
            motion_fault = self._motion_fault_latest
            motion_fault_evidence = self._motion_fault_latest_evidence
            motion_fault_outcome = self._motion_fault_latest_outcome
            motion_fault_detail = self._motion_fault_latest_detail
            motion_fault_safe_stop_epoch = self._motion_fault_latest_safe_stop_epoch
            motion_fault_counters = dict(self._motion_fault_counters)
        connection_session_id = self._connection_session_id
        transport_ready = (
            lifecycle_ready
            and connection_session_id is not None
            and session_snapshot.session_id == connection_session_id
        )
        lease_remaining_ms = (
            None
            if session_snapshot.lease_deadline is None
            else (session_snapshot.lease_deadline - observed_at).total_seconds() * 1_000.0
        )
        control_active = session_snapshot.state.value == "ACTIVE"
        lease_valid = (
            control_active
            and lease_remaining_ms is not None
            and lease_remaining_ms > 0
        )
        active_command_count = len(self._coordinator.active_commands)
        if ros_ready:
            try:
                self._ros.check_health()
            except Exception as exc:
                ros_ready = False
                ros_reason = f"ros_executor_unavailable:{type(exc).__name__}"

        nav2_actions: dict[str, dict[str, Any]] = {}
        for skill_id in sorted(nav2_commands):
            result = nav2_results.get(skill_id)
            nav2_actions[skill_id] = {
                "ready": False if result is None else result.ready,
                "reason": (
                    result.reason
                    if result is not None
                    else nav2_probe_reason or "nav2_probe_incomplete"
                ),
            }
        nav2_required = bool(nav2_commands)
        nav2_ready = bool(nav2_actions) and all(
            bool(action["ready"]) for action in nav2_actions.values()
        )
        safety_endpoints: dict[str, dict[str, Any]] = {}
        for name in sorted(configured_safety_endpoints):
            result = safety_endpoint_results.get(name)
            safety_endpoints[name] = {
                "ready": False if result is None else result.ready,
                "reason": (
                    result.reason
                    if result is not None
                    else safety_endpoint_probe_reason or "safety_endpoint_probe_incomplete"
                ),
            }
        safety_endpoints_ready = bool(safety_endpoints) and all(
            bool(endpoint["ready"]) for endpoint in safety_endpoints.values()
        )

        localization_paths = tuple(
            sorted(
                path
                for path in self.config.state_sources
                if path == "localization" or path.startswith("localization.")
            )
        )
        with self._state_freshness_lock:
            received = {
                path: self._state_received_monotonic[path]
                for path in localization_paths
                if path in self._state_received_monotonic
            }
        localization_complete = bool(localization_paths) and len(received) == len(
            localization_paths
        )
        localization_age_ms = (
            (monotonic_now - min(received.values())) * 1_000.0 if localization_complete else None
        )
        localization_fresh = (
            localization_age_ms is not None
            and 0 <= localization_age_ms <= self.config.diagnostics.localization_max_age_ms
        )
        localization_required = nav2_required
        if not localization_paths:
            localization_reason = "localization_not_configured"
        elif not localization_complete:
            localization_reason = "localization_incomplete"
        elif localization_age_ms is not None and localization_age_ms < 0:
            localization_reason = "monotonic_clock_regressed"
        elif not localization_fresh:
            localization_reason = "localization_stale"
        else:
            localization_reason = None

        observation_required = any(
            command.requires_observation for command in self.config.commands.values()
        )
        if self._observation_guard is None:
            observation = {
                "ready": False,
                "status": "DISABLED",
                "reason": "observation_disabled",
                "sensorId": None,
                "localAckAgeMs": None,
                "healthAgeMs": None,
                "captureAgeMs": None,
                "gatewayAckAgeMs": None,
                "maxAgeMs": self.config.observation.admission_max_age_ms,
            }
        else:
            freshness = self._observation_guard.freshness(monotonic_now=monotonic_now)
            observation = {
                "ready": freshness.ready,
                "status": freshness.status,
                "reason": freshness.reason,
                "sensorId": freshness.sensor_id,
                "localAckAgeMs": _rounded_age(freshness.local_ack_age_ms),
                "healthAgeMs": _rounded_age(freshness.health_age_ms),
                "captureAgeMs": _rounded_age(freshness.capture_age_ms),
                "gatewayAckAgeMs": _rounded_age(freshness.gateway_ack_age_ms),
                "maxAgeMs": freshness.max_age_ms,
            }

        downstream_ready_for_acquire = (
            lifecycle_ready
            and ros_ready
            and safety_endpoints_ready
            and not motion_fault_latched
            and (not nav2_required or nav2_ready)
            and (not localization_required or localization_fresh)
            and (not observation_required or bool(observation["ready"]))
        )
        effective_can_acquire = (
            transport_ready
            and session_snapshot.can_acquire
            and downstream_ready_for_acquire
            and not motion_fault_latched
        )
        standby_ready = (
            not control_active
            and effective_can_acquire
            and active_command_count == 0
        )
        active_control_ready = (
            transport_ready and lease_valid and not motion_fault_latched
        )
        control_ready = standby_ready or active_control_ready
        if motion_fault_latched:
            control_reason = "motion_safety_fault_latched"
        elif not transport_ready:
            control_reason = "transport_session_unavailable"
        elif active_command_count and not lease_valid:
            control_reason = "active_commands_without_valid_lease"
        elif control_active and not lease_valid:
            control_reason = "control_lease_expired"
        elif standby_ready:
            control_reason = "standby_waiting_for_control_acquire"
        elif not session_snapshot.can_acquire:
            control_reason = "control_acquire_blocked"
        elif not downstream_ready_for_acquire:
            control_reason = "downstream_readiness_unavailable"
        else:
            control_reason = None
        control_mode = (
            "ACTIVE" if active_control_ready else "STANDBY" if standby_ready else "BLOCKED"
        )
        safe_stop_ready = (
            session_snapshot.safe_stop_state is SafeStopState.CONFIRMED
        )

        checks: dict[str, Any] = {
            "runtime": {
                "required": True,
                "ready": lifecycle_ready,
                "reason": None if lifecycle_ready else "runtime_not_initialized",
            },
            "transportSession": {
                "required": True,
                "ready": transport_ready,
                "reason": None if transport_ready else "transport_session_unavailable",
            },
            "control": {
                "required": True,
                "ready": control_ready,
                "state": session_snapshot.state.value,
                "mode": control_mode,
                "active": control_active,
                "canAcquire": effective_can_acquire,
                "leaseValid": lease_valid,
                "activeCommandCount": active_command_count,
                "reason": control_reason,
                "leaseRemainingMs": _rounded_age(lease_remaining_ms),
            },
            "safeStop": {
                "required": True,
                "ready": safe_stop_ready,
                "reason": (
                    None
                    if safe_stop_ready
                    else f"safe_stop_{session_snapshot.safe_stop_state.value}"
                ),
                "authorityRevoked": session_snapshot.authority_revoked,
                "state": session_snapshot.safe_stop_state.value,
                "epoch": session_snapshot.safe_stop_epoch,
            },
            "safetyEndpoints": {
                "required": True,
                "ready": safety_endpoints_ready,
                "reason": (
                    None
                    if safety_endpoints_ready
                    else safety_endpoint_probe_reason
                    or "one_or_more_safety_endpoints_unavailable"
                ),
                "endpoints": safety_endpoints,
            },
            "rosExecutor": {
                "required": True,
                "ready": ros_ready,
                "reason": ros_reason,
            },
            "nav2Actions": {
                "required": nav2_required,
                "ready": nav2_ready,
                "reason": (
                    None
                    if nav2_ready
                    else nav2_probe_reason or "one_or_more_nav2_actions_unavailable"
                ),
                "actions": nav2_actions,
            },
            "localization": {
                "required": localization_required,
                "ready": localization_fresh,
                "reason": localization_reason,
                "ageMs": _rounded_age(localization_age_ms),
                "maxAgeMs": self.config.diagnostics.localization_max_age_ms,
                "configuredPaths": list(localization_paths),
                "observedPaths": sorted(received),
            },
            "observation": {
                "required": observation_required,
                **observation,
            },
            "motionSafetyFault": {
                "required": self.config.safety.motion_safety_fault.enabled,
                "ready": not motion_fault_latched,
                "reason": (
                    "motion_safety_fault_latched" if motion_fault_latched else None
                ),
                "latched": motion_fault_latched,
                "eventId": None if motion_fault is None else str(motion_fault.event_id),
                "source": None if motion_fault is None else motion_fault.source,
                "code": None if motion_fault is None else motion_fault.code,
                "controlGeneration": (
                    None if motion_fault is None else motion_fault.control_generation
                ),
                "localizationEpochId": (
                    None if motion_fault is None else motion_fault.localization_epoch_id
                ),
                "safeStopEpoch": motion_fault_safe_stop_epoch,
                "commandId": (
                    None
                    if motion_fault_evidence is None
                    else str(motion_fault_evidence.command_id)
                ),
                "outcome": motion_fault_outcome,
                "detail": motion_fault_detail,
                "cancelBeginLatencyMs": (
                    None
                    if motion_fault_evidence is None
                    else _rounded_age(motion_fault_evidence.cancel_begin_latency_ms)
                ),
                "nav2TerminalLatencyMs": (
                    None
                    if motion_fault_evidence is None
                    else _rounded_age(motion_fault_evidence.nav2_terminal_latency_ms)
                ),
                "rosCancelCallLatencyMs": (
                    None
                    if motion_fault_evidence is None
                    else _rounded_age(motion_fault_evidence.ros_cancel_call_latency_ms)
                ),
                "counters": motion_fault_counters,
            },
        }
        ready = all(bool(check["ready"]) for check in checks.values() if bool(check["required"]))
        command_readiness_reason = (
            None
            if ready
            else control_reason
            if not control_ready
            else "required_dependency_unavailable"
        )
        return {
            "schemaVersion": "robot-agent.diagnostics/1",
            "robotId": self.config.robot_id,
            "robotBootId": str(self._boot_id),
            "observedAt": observed_at.isoformat(),
            "ready": ready,
            "commandReadiness": {
                "ready": ready,
                "mode": control_mode if ready else "BLOCKED",
                "reason": command_readiness_reason,
            },
            "controlState": {
                "state": session_snapshot.state.value,
                "active": control_active,
                "standby": standby_ready,
                "canAcquire": effective_can_acquire,
                "authorityRevoked": session_snapshot.authority_revoked,
                "safeStopState": session_snapshot.safe_stop_state.value,
                "safeStopEpoch": session_snapshot.safe_stop_epoch,
                "leaseValid": lease_valid,
                "leaseRemainingMs": _rounded_age(lease_remaining_ms),
                "activeCommandCount": active_command_count,
            },
            "motionSafetyFault": checks["motionSafetyFault"],
            "checks": checks,
        }

    async def capabilities_snapshot(self) -> dict[str, Any]:
        """Return configured capability metadata without schemas or credentials."""
        diagnostics = await self.diagnostics_snapshot()
        nav2_actions = diagnostics["checks"]["nav2Actions"]["actions"]
        descriptors = (
            {
                descriptor.skillId: descriptor
                for descriptor in self._capability_manifest.capabilities
            }
            if self._capability_manifest is not None
            else {}
        )
        capabilities: list[dict[str, Any]] = []
        for skill_id, command in sorted(self.config.commands.items()):
            descriptor = descriptors.get(skill_id)
            endpoint = nav2_actions.get(skill_id)
            executor = (
                descriptor.executor
                if descriptor is not None
                else "nav2"
                if command.goal_transform in {"navigate_to_pose_v1", "nav2_spin_v2"}
                else "ros_action"
                if command.adapter == "action"
                else "ros_service"
            )
            capabilities.append(
                {
                    "skillId": skill_id,
                    "skillVersion": command.skill_version,
                    "description": command.description,
                    "kind": command.capability_kind,
                    "riskLevel": command.risk_level,
                    "executor": executor,
                    "executorProfile": (
                        descriptor.executorProfile
                        if descriptor is not None
                        else command.executor_profile
                    ),
                    "cancellable": (
                        command.adapter == "action"
                        if command.cancellable is None
                        else command.cancellable
                    ),
                    "interruptible": (
                        command.adapter == "action"
                        if command.interruptible is None
                        else command.interruptible
                    ),
                    "timeoutMs": command.timeout_ms,
                    "resources": list(command.resources),
                    "requiredSensorIds": list(command.observation_sensor_ids),
                    "endpointReady": None if endpoint is None else endpoint["ready"],
                    "endpointReason": None if endpoint is None else endpoint["reason"],
                }
            )
        profile = self.config.robot_profile
        return {
            "schemaVersion": "robot-agent.capabilities/1",
            "robotId": self.config.robot_id,
            "robotBootId": str(self._boot_id),
            "observedAt": diagnostics["observedAt"],
            "capabilitiesDigest": (
                self._capability_manifest.manifestDigest
                if self._capability_manifest is not None
                else self.config.capabilities_digest
            ),
            "robotProfile": (
                None
                if profile is None
                else {
                    "baseType": profile.base_type,
                    "headMotion": profile.head_motion,
                    "mapFrameId": profile.map_frame_id,
                    "baseFrameId": profile.base_frame_id,
                    "sensors": [
                        {
                            "sensorId": sensor.sensor_id,
                            "kind": sensor.kind,
                            "frameId": sensor.frame_id,
                            "fixedMount": sensor.fixed_mount,
                            "depthCapable": sensor.depth_capable,
                        }
                        for sensor in profile.sensors
                    ],
                }
            ),
            "capabilities": capabilities,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rounded_age(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, 3)


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
