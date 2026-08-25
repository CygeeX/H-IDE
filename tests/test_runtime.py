from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

import robot_agent.runtime as runtime_module
from robot_agent.config import (
    AgentConfig,
    CommandConfig,
    MotionSafetyFaultConfig,
    SafeStopConfig,
)
from robot_agent.coordinator import CommandCancellationError, ExecutionResult
from robot_agent.ledger import CommandRecord, CommandState
from robot_agent.observation import (
    ObservationBinding,
    ObservationHealthSample,
    ObservationHealthStatus,
)
from robot_agent.outbound import PriorityOutboundQueue
from robot_agent.protocol import CommandCancel, CommandRequest, WireMessage, parse_wire_message
from robot_agent.ros_adapter import MotionFaultCancelEvidence, RosActionReadiness
from robot_agent.runtime import RobotAgentRuntime, RuntimeSafetyError
from robot_agent.session import ControlState, SessionStateError


def runtime_config(
    tmp_path: Path,
    *,
    lease_interval_ms: int = 500,
    lease_timeout_ms: int = 2000,
    observation_enabled: bool = False,
    control_arm_enabled: bool = False,
    control_arm_timeout_ms: int = 1000,
) -> AgentConfig:
    safety: dict[str, Any] = {
        "safe_stop": {
            "adapter": "service",
            "ros_name": "/robot/safe_stop",
            "ros_type": "std_srvs/srv/Trigger",
            "request": {},
            "timeout_ms": 1000,
        },
        "heartbeat_topic": "/robot_agent/heartbeat",
        "heartbeat_period_ms": 250,
    }
    if control_arm_enabled:
        safety["control_arm"] = {
            "adapter": "service",
            "ros_name": "/robot/control_arm",
            "ros_type": "std_srvs/srv/Trigger",
            "request": {},
            "timeout_ms": control_arm_timeout_ms,
        }
    return AgentConfig.model_validate(
        {
            "schema_version": "1",
            "robot_id": "g1-dev",
            "transport": {
                "mode": "client",
                "uri": "ws://127.0.0.1:8765/agent",
                "reconnect_initial_ms": 50,
                "reconnect_max_ms": 100,
            },
            "tls": {"enabled": False, "allow_plaintext_loopback": True},
            "session": {
                "lease_interval_ms": lease_interval_ms,
                "lease_timeout_ms": lease_timeout_ms,
            },
            "observation": {
                "enabled": observation_enabled,
                "admission_max_age_ms": 500,
                "watchdog_timeout_ms": 750,
            },
            "storage": {"ledger_path": str(tmp_path / "ledger.sqlite3")},
            "state": {
                "change_hz": 5.0,
                "snapshot_hz": 1.0,
                "outbound_queue_size": 32,
            },
            "safety": safety,
            "commands": {
                "power.enable": {
                    "adapter": "service",
                    "ros_name": "/robot/power_enable",
                    "ros_type": "std_srvs/srv/SetBool",
                    "request_schema": {
                        "type": "object",
                        "properties": {"data": {"type": "boolean"}},
                        "required": ["data"],
                        "additionalProperties": False,
                    },
                    "max_ttl_ms": 5000,
                    "timeout_ms": 2000,
                    "resources": ["power"],
                }
            },
            "state_sources": {},
        }
    )


class FakeRosAdapter:
    def __init__(self) -> None:
        self.state_callback: Any = None
        self.feedback_callback: Any = None
        self.observation_health_callback: Any = None
        self.observation_topics: tuple[str, str] | None = None
        self.observation_demands: list[tuple[UUID, bool, str, str, str, str]] = []
        self.started = False
        self.closed = False
        self.executions: list[tuple[UUID, str, dict[str, Any]]] = []
        self.cancelled: list[UUID] = []
        self.safe_stop_reasons = 0
        self.safe_stop_succeeded = True
        self.safe_stop_exception: Exception | None = None
        self.control_arm_calls = 0
        self.control_arm_succeeded = True
        self.control_arm_exception: Exception | None = None
        self.control_arm_delay_s = 0.0
        self.health_exception: Exception | None = None
        self.safety_endpoints_ready = True

    def set_callbacks(
        self,
        *,
        state_callback: Any,
        feedback_callback: Any,
        observation_health_callback: Any = None,
    ) -> None:
        self.state_callback = state_callback
        self.feedback_callback = feedback_callback
        self.observation_health_callback = observation_health_callback

    def configure_observation(self, *, demand_topic: str, health_topic: str) -> None:
        self.observation_topics = (demand_topic, health_topic)

    def configure_state_sources(self, sources: object) -> None:
        del sources

    def configure_heartbeat(self, *, topic: str, period_ms: int, state_supplier: Any) -> None:
        del topic, period_ms, state_supplier

    def start(self) -> None:
        self.started = True

    def check_health(self) -> None:
        if self.health_exception is not None:
            raise self.health_exception

    async def safety_endpoint_readiness(
        self,
        endpoints: dict[str, SafeStopConfig],
    ) -> dict[str, RosActionReadiness]:
        return {
            name: RosActionReadiness(
                ready=self.safety_endpoints_ready,
                reason=None if self.safety_endpoints_ready else "endpoint_unavailable",
            )
            for name in endpoints
        }

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult:
        self.executions.append((command_id, command.ros_name, params))
        return ExecutionResult(
            succeeded=True,
            detail="enabled",
            result={"success": True, "message": "enabled"},
        )

    async def cancel(self, command_id: UUID) -> None:
        self.cancelled.append(command_id)

    async def publish_observation_demand(
        self,
        *,
        command_id: UUID,
        active: bool,
        observation_session_id: str,
        sensor_id: str,
        scene_version: str,
        localization_epoch_id: str,
    ) -> None:
        self.observation_demands.append(
            (
                command_id,
                active,
                observation_session_id,
                sensor_id,
                scene_version,
                localization_epoch_id,
            )
        )

    async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
        assert config.ros_name == "/robot/safe_stop"
        self.safe_stop_reasons += 1
        if self.safe_stop_exception is not None:
            raise self.safe_stop_exception
        return ExecutionResult(
            succeeded=self.safe_stop_succeeded,
            detail="stopped" if self.safe_stop_succeeded else "rejected",
            result={"success": self.safe_stop_succeeded},
        )

    async def control_arm(self, config: SafeStopConfig) -> ExecutionResult:
        assert config.ros_name == "/robot/control_arm"
        self.control_arm_calls += 1
        if self.control_arm_delay_s:
            await asyncio.sleep(self.control_arm_delay_s)
        if self.control_arm_exception is not None:
            raise self.control_arm_exception
        return ExecutionResult(
            succeeded=self.control_arm_succeeded,
            detail="armed" if self.control_arm_succeeded else "rejected",
            result={"success": self.control_arm_succeeded},
        )

    async def close(self) -> None:
        self.closed = True


class QueueConnection:
    mode = "client"

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.outgoing: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False
        self.close_code: int | None = None

    async def recv_text(self) -> str:
        return await self.incoming.get()

    async def send_text(self, text: str) -> None:
        await self.outgoing.put(text)

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        del reason
        self.closed = True
        self.close_code = code


def inbound(
    message_type: str,
    session_id: UUID,
    payload: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "version": "1.0",
            "type": message_type,
            "message_id": str(uuid4()),
            "robot_id": "g1-dev",
            "session_id": str(session_id),
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
    )


async def next_message(connection: QueueConnection) -> WireMessage:
    raw = await asyncio.wait_for(connection.outgoing.get(), timeout=2)
    return parse_wire_message(raw, expected_robot_id="g1-dev")


@pytest.mark.asyncio
async def test_control_acquire_is_the_only_latched_rearm_and_duplicates_are_idempotent(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, control_arm_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    message = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )

    await runtime._route_message(message, session_id)
    assert runtime._session.state is ControlState.ACTIVE
    assert ros.control_arm_calls == 1

    # A duplicated acquire cannot create another generation while ACTIVE.
    await runtime._route_message(message, session_id)
    assert ros.control_arm_calls == 1

    await runtime._session.force_safe_hold("lease expired")
    assert runtime._session.state is ControlState.SAFE_HOLD
    await runtime._route_message(message, session_id)
    assert runtime._session.state is ControlState.ACTIVE
    assert ros.control_arm_calls == 2
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_control_release_relinquishes_remote_authority_without_stopping_sonic(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, control_arm_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    acquire = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )
    release = parse_wire_message(
        inbound(
            "control.release",
            session_id,
            {"reason": "approved mission reached a terminal state"},
        ),
        expected_robot_id="g1-dev",
    )

    await runtime._route_message(acquire, session_id)
    runtime._connection_session_id = session_id
    runtime._outbound = PriorityOutboundQueue(maxsize=32)
    before = runtime._session.snapshot
    initial_stops = ros.safe_stop_reasons
    await runtime._route_message(release, session_id)
    released = runtime._session.snapshot

    assert released.state is ControlState.SAFE_HOLD
    assert released.authority_revoked is True
    assert released.safe_stop_state.value == "confirmed"
    assert released.safe_stop_epoch == before.safe_stop_epoch
    assert released.can_acquire is True
    assert ros.safe_stop_reasons == initial_stops
    outbound = []
    while runtime._outbound.qsize:
        outbound.append((await runtime._outbound.get()).payload)
    release_statuses = [
        item.payload
        for item in outbound
        if getattr(item, "type", None) == "control.status"
    ]
    assert [item.safe_stop_state for item in release_statuses] == ["confirmed"]
    assert all(item.authority_revoked is True for item in release_statuses)
    assert all(item.safe_stop_epoch == released.safe_stop_epoch for item in release_statuses)

    # A retried wire message is still a logical release.  It cannot execute a
    # physical stop or advance the durable physical-stop epoch.
    await runtime._route_message(release, session_id)
    assert runtime._session.snapshot.safe_stop_epoch == released.safe_stop_epoch
    assert ros.safe_stop_reasons == initial_stops
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_control_release_rejects_active_command_without_revoking_authority(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    acquire = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )
    release = parse_wire_message(
        inbound("control.release", session_id, {"reason": "terminal"}),
        expected_robot_id="g1-dev",
    )
    await runtime._route_message(acquire, session_id)
    command_id = uuid4()
    assert runtime._coordinator._arbiter.claim(command_id, ("motion",)).acquired
    initial_stops = ros.safe_stop_reasons

    with pytest.raises(SessionStateError, match="zero active or pending"):
        await runtime._route_message(release, session_id)

    assert runtime._session.state is ControlState.ACTIVE
    assert runtime._session.snapshot.authority_revoked is False
    assert ros.safe_stop_reasons == initial_stops
    runtime._coordinator._arbiter.release(command_id)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_control_release_never_calls_a_failing_physical_stop_adapter(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    acquire = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )
    release = parse_wire_message(
        inbound("control.release", session_id, {"reason": "terminal"}),
        expected_robot_id="g1-dev",
    )
    await runtime._route_message(acquire, session_id)
    ros.safe_stop_succeeded = False

    initial_stops = ros.safe_stop_reasons
    await runtime._route_message(release, session_id)

    snapshot = runtime._session.snapshot
    assert snapshot.state is ControlState.SAFE_HOLD
    assert snapshot.authority_revoked is True
    assert snapshot.safe_stop_state.value == "confirmed"
    assert snapshot.can_acquire is True
    assert ros.safe_stop_reasons == initial_stops
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_control_release_maximum_reason_keeps_status_wire_valid(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    acquire = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )
    release = parse_wire_message(
        inbound("control.release", session_id, {"reason": "r" * 256}),
        expected_robot_id="g1-dev",
    )
    await runtime._route_message(acquire, session_id)
    runtime._connection_session_id = session_id
    runtime._outbound = PriorityOutboundQueue(maxsize=32)

    await runtime._route_message(release, session_id)

    assert len(runtime._session.snapshot.reason) == 256
    while runtime._outbound.qsize:
        item = (await runtime._outbound.get()).payload
        if getattr(item, "type", None) == "control.status":
            assert len(item.payload.reason) <= 256
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_failed_control_arm_latches_remote_fault_without_stopping_sonic(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    ros.control_arm_succeeded = False
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, control_arm_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    message = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )

    with pytest.raises(SessionStateError, match="failed closed"):
        await runtime._route_message(message, session_id)
    assert runtime._session.state is ControlState.SAFE_HOLD
    assert ros.control_arm_calls == 1
    # Only the explicit process-start boundary used the physical stop.
    assert ros.safe_stop_reasons == 1
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_timed_out_control_arm_latches_fault_without_stopping_sonic(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    ros.control_arm_delay_s = 60.0
    runtime = RobotAgentRuntime(
        runtime_config(
            tmp_path,
            control_arm_enabled=True,
            control_arm_timeout_ms=10,
        ),
        ros_adapter=ros,
    )
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    message = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )

    with pytest.raises(SessionStateError, match=r"exceeded 0\.010s deadline"):
        await runtime._route_message(message, session_id)
    assert runtime._session.state is ControlState.SAFE_HOLD
    assert ros.control_arm_calls == 1
    assert ros.safe_stop_reasons == 1
    for _ in range(4):
        if not runtime._detached_control_arm_tasks:
            break
        await asyncio.sleep(0)
    assert runtime._detached_control_arm_tasks == set()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_control_arm_rejects_any_preexisting_active_command(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, control_arm_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    command_id = uuid4()
    assert runtime._coordinator._arbiter.claim(
        command_id, ("motion",)
    ).acquired is True
    message = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )

    with pytest.raises(SessionStateError, match="zero active"):
        await runtime._route_message(message, session_id)
    assert runtime._session.state is ControlState.SAFE_HOLD
    assert ros.control_arm_calls == 0
    assert ros.safe_stop_reasons == 1
    runtime._coordinator._arbiter.release(command_id)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_latched_session_fault_blocks_control_arm_before_local_release(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, control_arm_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    await runtime._session.latch_safety_fault("physical stop acknowledgement lost")
    message = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )

    with pytest.raises(SessionStateError, match="safety fault is latched"):
        await runtime._route_message(message, session_id)
    assert runtime._session.state is ControlState.SAFE_HOLD
    assert ros.control_arm_calls == 0
    assert ros.safe_stop_reasons == 1
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_control_acquire_is_rejected_when_safety_endpoint_is_unavailable(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    runtime._connection_session_id = session_id
    ros.safety_endpoints_ready = False
    message = parse_wire_message(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"}),
        expected_robot_id="g1-dev",
    )

    with pytest.raises(SessionStateError, match="safetyEndpoints"):
        await runtime._route_message(message, session_id)
    assert runtime._session.state is ControlState.SAFE_HOLD
    assert (await runtime.diagnostics_snapshot())["controlState"]["canAcquire"] is False
    await runtime.shutdown()


def observation_health(
    *,
    robot_boot_id: str = "boot-1",
    status: int = ObservationHealthStatus.USABLE,
    heartbeat_sequence: int = 1,
    capture_age_ms: float = 10.0,
    ack_age_ms: float = 10.0,
) -> ObservationHealthSample:
    return ObservationHealthSample(
        robot_id="g1-dev",
        robot_boot_id=robot_boot_id,
        gateway_instance_id="gateway-1",
        observation_session_id="obs-1",
        sensor_id="head_rgbd",
        scene_version="scene-1",
        localization_epoch_id="epoch-1",
        status=status,
        reason=0,
        has_last_accepted_sequence=True,
        last_accepted_sequence=heartbeat_sequence,
        last_accepted_monotonic_ns=1,
        capture_age_ms=capture_age_ms,
        ack_age_ms=ack_age_ms,
        drop_count=3,
        heartbeat_sequence=heartbeat_sequence,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (ObservationHealthStatus.DEGRADED, "DEGRADED"),
        (ObservationHealthStatus.STREAMING_WAITING_ACK, "NOT_USABLE"),
    ],
)
@pytest.mark.asyncio
async def test_runtime_normalizes_non_usable_observation_health(
    tmp_path: Path,
    status: int,
    expected: str,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, observation_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    try:
        sample = observation_health(robot_boot_id=str(runtime._boot_id), status=status)
        if status != ObservationHealthStatus.USABLE:
            # Non-usable messages are still valid health, but do not refresh an ACK.
            sample = ObservationHealthSample(
                **{
                    **sample.__dict__,
                    "has_last_accepted_sequence": False,
                    "last_accepted_sequence": 0,
                    "last_accepted_monotonic_ns": 0,
                    "capture_age_ms": float("inf"),
                    "ack_age_ms": float("inf"),
                }
            )
        runtime._accept_observation_health(sample)
        snapshot = runtime._state.snapshot()
        assert snapshot["observation"]["status"] == expected
        assert snapshot["observation"]["capture_age_ms"] is None
        assert snapshot["observation"]["ack_age_ms"] is None
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_configures_and_publishes_observation_health_state(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, observation_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    try:
        assert ros.observation_topics == (
            "/robot_agent/observation_demand",
            "/robot_observation_gateway/health",
        )
        assert ros.observation_health_callback is not None
        sample = observation_health(robot_boot_id=str(runtime._boot_id))
        ros.observation_health_callback(sample)
        await asyncio.sleep(0)
        snapshot = runtime._state.snapshot()
        assert snapshot["observation"]["status"] == "USABLE"
        assert snapshot["observation"]["session_id"] == "obs-1"
        assert snapshot["observation"]["drop_count"] == 3

        # A repeated heartbeat and malformed ages cannot replace the accepted sample.
        ros.observation_health_callback(sample)
        ros.observation_health_callback(
            observation_health(
                robot_boot_id=str(runtime._boot_id),
                heartbeat_sequence=2,
                capture_age_ms=float("nan"),
            )
        )
        await asyncio.sleep(0)
        assert runtime._observation_guard is not None
        assert runtime._observation_guard.current_health() is sample
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_observation_watchdog_failure_latches_remote_fault_without_stopping_sonic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(
        runtime_config(tmp_path, observation_enabled=True),
        ros_adapter=ros,
    )
    await runtime.initialize()
    command_id = uuid4()
    now = datetime.now(timezone.utc)
    record = CommandRecord(
        command_id=command_id,
        kind="navigation.vln.execute",
        payload_hash="a" * 64,
        payload={},
        state=CommandState.FAILED,
        detail="CANCEL_NOT_ACKNOWLEDGED",
        result=None,
        session_id=uuid4(),
        created_at=now,
        updated_at=now,
    )
    assert runtime._observation_guard is not None
    monkeypatch.setattr(
        runtime._observation_guard,
        "stale_bindings",
        lambda **_kwargs: (
            (
                ObservationBinding(
                    command_id,
                    "obs-1",
                    "head_rgbd",
                    "scene-1",
                    "boot-1",
                    "epoch-1",
                ),
                "ACK timeout",
            ),
        ),
    )

    async def fail_cancel(*_args: Any, **_kwargs: Any) -> bool:
        raise CommandCancellationError(record)

    monkeypatch.setattr(runtime._coordinator, "is_command_active", lambda _command_id: True)
    monkeypatch.setattr(runtime._coordinator, "cancel", fail_cancel)
    initial_safe_stops = ros.safe_stop_reasons
    try:
        await runtime._check_observation_watchdog()
        assert ros.safe_stop_reasons == initial_safe_stops
        assert runtime._session.snapshot.authority_revoked is True
        assert runtime._session.snapshot.safe_stop_state.value == "confirmed"
        assert runtime._session.snapshot.can_acquire is False
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_handshake_control_lease_command_and_disconnect(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    assert ros.started is True
    assert ros.safe_stop_reasons == 1

    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    hello = await next_message(connection)
    assert hello.type == "session.hello"
    assert hello.payload.capabilities_digest == config.capabilities_digest

    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    status = await next_message(connection)
    assert status.type == "control.status"
    assert status.payload.state == "SAFE_HOLD"

    command_id = uuid4()
    command_payload = {
        "command_id": str(command_id),
        "kind": "power.enable",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "ttl_ms": 5000,
        "params": {"data": True},
    }
    await connection.incoming.put(inbound("command.request", session_id, command_payload))
    rejected = await next_message(connection)
    assert rejected.type == "error"
    assert rejected.payload.code == "CONTROL_NOT_ACTIVE"
    assert ros.executions == []

    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    acquired = await next_message(connection)
    assert acquired.type == "control.status"
    assert acquired.payload.state == "ACTIVE"

    await connection.incoming.put(inbound("lease.renew", session_id, {"sequence": 0}))
    lease = await next_message(connection)
    assert lease.type == "lease.status"
    assert lease.payload.sequence == 0

    await connection.incoming.put(inbound("command.request", session_id, command_payload))
    running = await next_message(connection)
    assert running.type == "command.status"
    assert running.payload.state == "RUNNING"
    completed = await next_message(connection)
    assert completed.type == "command.status"
    assert completed.payload.state == "SUCCEEDED"
    assert ros.executions == [(command_id, "/robot/power_enable", {"data": True})]

    changed_payload = {**command_payload, "params": {"data": False}}
    await connection.incoming.put(inbound("command.request", session_id, changed_payload))
    conflict = await next_message(connection)
    assert conflict.type == "error"
    assert conflict.payload.code == "COMMAND_ID_CONFLICT"

    await connection.incoming.put(
        inbound("session.goodbye", session_id, {"reason": "test complete"})
    )
    await asyncio.wait_for(handler, timeout=2)
    assert ros.safe_stop_reasons == 1

    await runtime.shutdown()
    assert ros.closed is True


@pytest.mark.asyncio
async def test_runtime_lease_expiry_quiesces_without_physical_sonic_stop(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, lease_interval_ms=100, lease_timeout_ms=500)
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 100,
                "lease_timeout_ms": 500,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)

    status = await next_message(connection)
    assert status.type == "control.status"
    assert status.payload.state == "SAFE_HOLD"
    assert status.payload.safe_stop_state == "confirmed"
    assert ros.safe_stop_reasons == 1

    await connection.incoming.put(inbound("session.goodbye", session_id, {"reason": "done"}))
    await asyncio.wait_for(handler, timeout=2)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_rejects_peer_lease_policy_override(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    await connection.incoming.put(
        inbound(
            "session.welcome",
            uuid4(),
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 1000,
                "lease_timeout_ms": 5000,
                "control_state": "SAFE_HOLD",
            },
        )
    )

    with pytest.raises(RuntimeError, match="lease policy"):
        await asyncio.wait_for(handler, timeout=2)
    assert connection.close_code == 1002
    await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_message", ["{}", "control.acquire"])
async def test_runtime_rejects_invalid_application_handshake(
    tmp_path: Path, first_message: str
) -> None:
    config = runtime_config(tmp_path)
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    if first_message == "{}":
        await connection.incoming.put(first_message)
    else:
        await connection.incoming.put(
            inbound(first_message, uuid4(), {"operator_id": "operator-7"})
        )
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(handler, timeout=2)
    assert connection.close_code == 1002
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_routes_state_feedback_session_errors_and_cancel(tmp_path: Path) -> None:
    config = runtime_config(tmp_path)
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)

    assert ros.state_callback is not None
    ros.state_callback("robot.mode", "IDLE", datetime.now(timezone.utc))
    snapshot = await next_message(connection)
    assert snapshot.type == "state.snapshot"
    ros.state_callback("robot.mode", "MOVING", datetime.now(timezone.utc))
    update = await next_message(connection)
    assert update.type == "state.update"

    feedback_id = uuid4()
    assert ros.feedback_callback is not None
    ros.feedback_callback(feedback_id, {"sequence": [0, 1]})
    feedback = await next_message(connection)
    assert feedback.type == "command.status"
    assert feedback.payload.command_id == feedback_id

    await connection.incoming.put(
        inbound(
            "lease.status",
            session_id,
            {"sequence": 0, "expires_at": datetime.now(timezone.utc).isoformat()},
        )
    )
    unexpected = await next_message(connection)
    assert unexpected.type == "error"
    assert unexpected.payload.code == "UNEXPECTED_MESSAGE"

    await connection.incoming.put(
        inbound("control.acquire", uuid4(), {"operator_id": "operator-7"})
    )
    mismatch = await next_message(connection)
    assert mismatch.type == "error"
    assert mismatch.payload.code == "SESSION_MISMATCH"

    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)
    await connection.incoming.put(inbound("lease.renew", session_id, {"sequence": 4}))
    await next_message(connection)
    await connection.incoming.put(inbound("lease.renew", session_id, {"sequence": 3}))
    bad_lease = await next_message(connection)
    assert bad_lease.type == "error"
    assert bad_lease.payload.code == "INVALID_LEASE"

    await connection.incoming.put(
        inbound(
            "command.cancel",
            session_id,
            {"command_id": str(uuid4()), "reason": "not active"},
        )
    )
    not_active = await next_message(connection)
    assert not_active.type == "error"
    assert not_active.payload.code == "COMMAND_NOT_ACTIVE"

    await connection.incoming.put(inbound("session.goodbye", session_id, {"reason": "done"}))
    await asyncio.wait_for(handler, timeout=2)
    await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_runtime_fails_closed_when_mandatory_safe_stop_fails(
    tmp_path: Path, raises: bool
) -> None:
    ros = FakeRosAdapter()
    if raises:
        ros.safe_stop_exception = RuntimeError("safe stop unavailable")
    else:
        ros.safe_stop_succeeded = False
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)

    with pytest.raises(RuntimeSafetyError, match="mandatory safe stop"):
        await runtime.initialize()
    await runtime.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.initialize()


@pytest.mark.asyncio
async def test_safe_stop_deadline_is_bounded_when_adapter_suppresses_cancellation(
    tmp_path: Path,
) -> None:
    class UncooperativeSafeStopAdapter(FakeRosAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.release = asyncio.Event()
            self.stopped = asyncio.Event()

        async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
            del config
            try:
                while not self.release.is_set():
                    try:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        continue
                return ExecutionResult(succeeded=True, detail="released")
            finally:
                self.stopped.set()

    config = runtime_config(tmp_path)
    config = config.model_copy(
        update={
            "safety": config.safety.model_copy(
                update={
                    "safe_stop": config.safety.safe_stop.model_copy(
                        update={"timeout_ms": 50}
                    )
                }
            )
        }
    )
    ros = UncooperativeSafeStopAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    initialization = asyncio.create_task(runtime.initialize())

    try:
        done, _ = await asyncio.wait({initialization}, timeout=0.3)
        assert initialization in done
        with pytest.raises(RuntimeSafetyError, match="deadline"):
            await initialization
        assert len(runtime._detached_safety_tasks) == 1
    finally:
        ros.release.set()
        await asyncio.wait_for(ros.stopped.wait(), timeout=1)
        await asyncio.sleep(0)
        if not initialization.done():
            await asyncio.wait_for(initialization, timeout=1)
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_cancelling_safe_stop_cleans_both_parallel_safety_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingSafetyAdapter(FakeRosAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.physical_started = asyncio.Event()
            self.physical_stopped = asyncio.Event()

        async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
            del config
            self.physical_started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                self.physical_stopped.set()

    ros = BlockingSafetyAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    preemption_started = asyncio.Event()
    preemption_stopped = asyncio.Event()

    async def blocking_preemption(*, reason: str, at: datetime) -> tuple[UUID, ...]:
        del reason, at
        preemption_started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        finally:
            preemption_stopped.set()

    monkeypatch.setattr(runtime._coordinator, "preempt_all", blocking_preemption)
    safe_stop = asyncio.create_task(runtime._perform_safe_stop("outer cancellation"))
    await asyncio.wait_for(ros.physical_started.wait(), timeout=1)
    await asyncio.wait_for(preemption_started.wait(), timeout=1)
    safe_stop.cancel()

    with pytest.raises(asyncio.CancelledError):
        await safe_stop
    await asyncio.wait_for(ros.physical_stopped.wait(), timeout=1)
    await asyncio.wait_for(preemption_stopped.wait(), timeout=1)
    await asyncio.sleep(0)
    assert runtime._detached_safety_tasks == set()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_self_cancelled_physical_safe_stop_is_a_safety_failure(
    tmp_path: Path,
) -> None:
    class SelfCancellingSafeStopAdapter(FakeRosAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
            self.calls += 1
            if self.calls == 1:
                return await super().safe_stop(config)
            raise asyncio.CancelledError("adapter cancelled its own safe-stop task")

    runtime = RobotAgentRuntime(
        runtime_config(tmp_path),
        ros_adapter=SelfCancellingSafeStopAdapter(),
    )
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    await runtime._session.acquire(
        session_id,
        operator_id="operator-7",
        at=datetime.now(timezone.utc),
    )

    with pytest.raises(RuntimeSafetyError, match="physical safe stop was cancelled"):
        await runtime._session.force_safe_hold("lease expired")
    with pytest.raises(SessionStateError, match="safety fault is latched"):
        await runtime._session.acquire(
            session_id,
            operator_id="operator-7",
            at=datetime.now(timezone.utc),
        )
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_run_cleans_ros_after_initialization_failure(tmp_path: Path) -> None:
    ros = FakeRosAdapter()
    ros.safe_stop_exception = RuntimeError("safe stop unavailable")
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)

    with pytest.raises(RuntimeSafetyError):
        await runtime.run(asyncio.Event())
    assert ros.closed is True


@pytest.mark.asyncio
async def test_runtime_prunes_terminal_ledger_periodically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    pruned = threading.Event()
    original_prune = runtime._ledger.prune_terminal_before

    def observed_prune(*, before: datetime) -> int:
        result = original_prune(before=before)
        pruned.set()
        return result

    monkeypatch.setattr(runtime._ledger, "prune_terminal_before", observed_prune)
    monkeypatch.setattr(runtime_module, "_RETENTION_CHECK_INTERVAL_SECONDS", 0.01)
    retention = asyncio.create_task(runtime._retention_loop())
    try:
        assert await asyncio.to_thread(pruned.wait, 0.5)
    finally:
        retention.cancel()
        await asyncio.gather(retention, return_exceptions=True)
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_lease_monitor_never_consults_physical_stop_adapter(
    tmp_path: Path,
) -> None:
    config = runtime_config(tmp_path, lease_interval_ms=100, lease_timeout_ms=500)
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 100,
                "lease_timeout_ms": 500,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)
    ros.safe_stop_exception = RuntimeError("safe stop unavailable")

    try:
        await asyncio.sleep(0.7)
        assert handler.done() is False
        assert runtime._session.state is ControlState.SAFE_HOLD
        assert runtime._session.snapshot.safe_stop_state.value == "confirmed"
        assert ros.safe_stop_reasons == 1
        await connection.incoming.put(
            inbound("session.goodbye", session_id, {"reason": "done"})
        )
        await asyncio.wait_for(handler, timeout=1)
    finally:
        if not handler.done():
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
        await runtime.shutdown()


class FaultAwareFakeRosAdapter(FakeRosAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.motion_fault_callback: Any = None
        self.motion_fault_config: MotionSafetyFaultConfig | None = None
        self.execution_started = asyncio.Event()
        self.fault_cancelled = asyncio.Event()
        self.exact_fault_cancels: list[UUID] = []

    def configure_motion_safety_fault(
        self,
        *,
        config: MotionSafetyFaultConfig,
        callback: Any,
    ) -> None:
        self.motion_fault_config = config
        self.motion_fault_callback = callback

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult:
        self.executions.append((command_id, command.ros_name, params))
        self.execution_started.set()
        await asyncio.Event().wait()
        raise AssertionError("blocked fault test execution unexpectedly resumed")

    async def cancel_nav2_for_motion_fault(
        self,
        command_id: UUID,
        *,
        received_monotonic_ns: int,
        cancel_begin_deadline_ms: int = 100,
        nav2_terminal_deadline_ms: int = 300,
    ) -> MotionFaultCancelEvidence:
        assert cancel_begin_deadline_ms == 100
        assert nav2_terminal_deadline_ms == 300
        self.exact_fault_cancels.append(command_id)
        self.fault_cancelled.set()
        begin = time.monotonic_ns()
        terminal = time.monotonic_ns()
        return MotionFaultCancelEvidence(
            command_id=command_id,
            received_monotonic_ns=received_monotonic_ns,
            cancel_begin_monotonic_ns=begin,
            ros_cancel_call_monotonic_ns=begin,
            nav2_terminal_monotonic_ns=terminal,
            outcome="nav2_terminal",
        )

def fault_runtime_config(tmp_path: Path) -> AgentConfig:
    base = runtime_config(tmp_path)
    fault_config = MotionSafetyFaultConfig.model_validate(
        {
            "enabled": True,
            "topic": "/h_ide/g1_sim/motion_safety_fault",
            "localization_epoch_id": "g1-sim-map-epoch-1",
            "cancel_begin_deadline_ms": 100,
            "nav2_terminal_deadline_ms": 300,
        }
    )
    safety = base.safety.model_copy(
        update={"motion_safety_fault": fault_config}
    )
    nav2_command = CommandConfig.model_validate(
        {
            "adapter": "action",
            "ros_name": "/navigate_to_pose",
            "ros_type": "nav2_msgs/action/NavigateToPose",
            "request_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "max_ttl_ms": 5000,
            "timeout_ms": 4000,
            "resources": ["base_motion"],
            "capability_kind": "motion",
            "goal_transform": "navigate_to_pose_v1",
        }
    )
    return base.model_copy(
        update={
            "safety": safety,
            "commands": {"navigation.navigate_to_pose": nav2_command},
        }
    )


def motion_fault_json(
    *,
    event_id: UUID,
    producer_boot_id: UUID,
    robot_id: str = "g1-dev",
    localization_epoch_id: str = "g1-sim-map-epoch-1",
) -> str:
    return json.dumps(
        {
            "schemaVersion": "h-ide.motion-safety-fault/v1",
            "eventId": str(event_id),
            "robotId": robot_id,
            "localizationEpochId": localization_epoch_id,
            "producerBootId": str(producer_boot_id),
            "sourceSequence": 1,
            "controlGeneration": 12,
            "source": "mujoco_plant",
            "code": "BASE_FALL_DETECTED",
            "reason": "base tilt crossed the latched safety threshold",
            "latched": True,
            "detectedAt": datetime.now(timezone.utc).isoformat(),
            "downstreamBootId": None,
            "probeRequestId": None,
        }
    )


@pytest.mark.asyncio
async def test_motion_fault_cancels_exact_action_without_stopping_sonic(
    tmp_path: Path,
) -> None:
    config = fault_runtime_config(tmp_path)
    ros = FaultAwareFakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    assert ros.motion_fault_config is not None
    assert ros.motion_fault_config.topic == "/h_ide/g1_sim/motion_safety_fault"
    assert ros.motion_fault_callback is not None

    # Wrong robot/epoch events are isolated before they can poison dedupe.
    event_id = uuid4()
    producer_boot_id = uuid4()
    ros.motion_fault_callback(
        motion_fault_json(
            event_id=uuid4(),
            producer_boot_id=producer_boot_id,
            robot_id="other-robot",
        ),
        time.monotonic_ns(),
    )
    ros.motion_fault_callback(
        motion_fault_json(
            event_id=uuid4(),
            producer_boot_id=producer_boot_id,
            localization_epoch_id="old-map-epoch",
        ),
        time.monotonic_ns(),
    )
    assert runtime._motion_fault_ingress_latched.is_set() is False

    session_id = uuid4()
    await runtime._session.connected(session_id)
    await runtime._session.acquire(
        session_id,
        operator_id="operator-7",
        at=datetime.now(timezone.utc),
    )
    command_id = uuid4()
    request = parse_wire_message(
        inbound(
            "command.request",
            session_id,
            {
                "command_id": str(command_id),
                "kind": "navigation.navigate_to_pose",
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "ttl_ms": 5000,
                "params": {},
            },
        ),
        expected_robot_id="g1-dev",
    )
    assert isinstance(request, CommandRequest)
    submission = asyncio.create_task(
        runtime._coordinator.submit(request, at=datetime.now(timezone.utc))
    )
    await asyncio.wait_for(ros.execution_started.wait(), timeout=1)

    retained_event = motion_fault_json(
        event_id=event_id,
        producer_boot_id=producer_boot_id,
    )
    ros.motion_fault_callback(retained_event, time.monotonic_ns())
    ros.motion_fault_callback(retained_event, time.monotonic_ns())
    await asyncio.wait_for(ros.fault_cancelled.wait(), timeout=1)
    await asyncio.wait_for(submission, timeout=1)
    await asyncio.gather(*tuple(runtime._motion_fault_tasks))

    assert ros.exact_fault_cancels == [command_id]
    assert runtime._coordinator.pending_cancel_confirmations == frozenset()

    terminal = runtime._ledger.get(command_id)
    assert terminal.state is CommandState.CANCELLED
    assert "physical safe-stop" not in terminal.detail
    assert runtime._session.snapshot.safe_stop_state.value == "confirmed"
    assert runtime._session.snapshot.can_acquire is False
    assert ros.safe_stop_reasons == 1
    assert runtime._motion_fault_counters["wrongRobot"] == 1
    assert runtime._motion_fault_counters["wrongEpoch"] == 1
    assert runtime._motion_fault_counters["accepted"] == 1
    assert runtime._motion_fault_counters["duplicate"] == 1
    audit = runtime._ledger.motion_fault_events()[-1]
    assert audit.event_id == event_id
    assert audit.command_id == command_id
    assert audit.safe_stop_epoch is None
    assert audit.source == "mujoco_plant"
    assert audit.code == "BASE_FALL_DETECTED"
    assert audit.control_generation == 12
    diagnostics = await runtime.diagnostics_snapshot()
    assert diagnostics["controlState"]["canAcquire"] is False
    assert diagnostics["motionSafetyFault"]["latched"] is True
    assert diagnostics["motionSafetyFault"]["eventId"] == str(event_id)
    assert diagnostics["motionSafetyFault"]["commandId"] == str(command_id)
    await runtime.shutdown()

    # A fresh process/deduplicator accepts the transient-local replay, but the
    # automatic handler still cannot create a physical safe-stop epoch.
    replay_ros = FaultAwareFakeRosAdapter()
    replay_runtime = RobotAgentRuntime(config, ros_adapter=replay_ros)
    await replay_runtime.initialize()
    before_replay_epoch = replay_runtime._session.snapshot.safe_stop_epoch
    assert replay_ros.motion_fault_callback is not None
    replay_ros.motion_fault_callback(retained_event, time.monotonic_ns())
    await asyncio.sleep(0)
    await asyncio.gather(*tuple(replay_runtime._motion_fault_tasks))
    assert replay_runtime._session.snapshot.safe_stop_epoch == before_replay_epoch
    assert replay_runtime._session.snapshot.can_acquire is False
    assert len(replay_runtime._ledger.motion_fault_events()) == 2
    assert replay_ros.safe_stop_reasons == 1
    await replay_runtime.shutdown()


@pytest.mark.asyncio
async def test_motion_fault_at_observation_await_tail_interlocks_final_dispatch(
    tmp_path: Path,
) -> None:
    """A ROS-thread latch must win even before its asyncio handler can run."""

    config = fault_runtime_config(tmp_path)
    ros = FaultAwareFakeRosAdapter()
    runtime = RobotAgentRuntime(config, ros_adapter=ros)
    await runtime.initialize()
    assert ros.motion_fault_callback is not None

    session_id = uuid4()
    await runtime._session.connected(session_id)
    await runtime._session.acquire(
        session_id,
        operator_id="operator-7",
        at=datetime.now(timezone.utc),
    )
    command_id = uuid4()
    event_id = uuid4()
    producer_boot_id = uuid4()
    fault_raw = motion_fault_json(
        event_id=event_id,
        producer_boot_id=producer_boot_id,
    )

    class FaultAtObservationTail:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.fault_emitted = asyncio.Event()
            self.finished = asyncio.Event()

        def admit(
            self,
            requested_command_id: UUID,
            command: CommandConfig,
            params: dict[str, Any],
            *,
            monotonic_now: float,
        ) -> ObservationBinding:
            del command, params, monotonic_now
            return ObservationBinding(
                command_id=requested_command_id,
                observation_session_id="observation-race",
                sensor_id="head_rgbd",
                scene_version="",
                robot_boot_id=str(runtime._boot_id),
                localization_epoch_id="g1-sim-map-epoch-1",
            )

        async def command_started(self, binding: ObservationBinding | None) -> None:
            assert binding is not None
            self.entered.set()
            await self.release.wait()
            # This is deliberately the final operation before the awaited
            # method returns.  _handle_motion_safety_fault cannot run until
            # this coroutine yields again, so only the cross-thread ingress
            # latch and final dispatch boundary can prevent execution.
            assert ros.motion_fault_callback is not None
            ros.motion_fault_callback(fault_raw, time.monotonic_ns())
            self.fault_emitted.set()

        async def command_finished(self, binding: ObservationBinding | None) -> None:
            assert binding is not None
            self.finished.set()

    observation = FaultAtObservationTail()
    runtime._coordinator._observation_guard = observation  # type: ignore[assignment]
    request = parse_wire_message(
        inbound(
            "command.request",
            session_id,
            {
                "command_id": str(command_id),
                "kind": "navigation.navigate_to_pose",
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "ttl_ms": 5000,
                "params": {},
            },
        ),
        expected_robot_id="g1-dev",
    )
    assert isinstance(request, CommandRequest)

    submission = asyncio.create_task(
        runtime._coordinator.submit(request, at=datetime.now(timezone.utc))
    )
    await asyncio.wait_for(observation.entered.wait(), timeout=1)
    observation.release.set()
    await asyncio.wait_for(observation.fault_emitted.wait(), timeout=1)
    terminal = await asyncio.wait_for(submission, timeout=1)

    assert terminal.state is CommandState.CANCELLED
    assert "motion-safety fault is latched" in terminal.detail
    assert ros.executions == []
    assert ros.exact_fault_cancels == []
    assert observation.finished.is_set() is True
    assert runtime._motion_fault_ingress_latched.is_set() is True

    await asyncio.sleep(0)
    await asyncio.gather(*tuple(runtime._motion_fault_tasks))

    assert runtime._session.snapshot.safe_stop_state.value == "confirmed"
    assert runtime._session.snapshot.can_acquire is False
    assert ros.safe_stop_reasons == 1
    audit = runtime._ledger.motion_fault_events()[-1]
    assert audit.event_id == event_id
    assert audit.command_id is None
    assert audit.outcome == "no_active_nav2_goal"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_ros_executor_failure_terminates_active_connection(tmp_path: Path) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)
    ros.health_exception = RuntimeError("ROS executor failed")

    try:
        with pytest.raises(RuntimeError, match="ROS executor failed"):
            await asyncio.wait_for(handler, timeout=0.5)
        assert ros.safe_stop_reasons == 1
    finally:
        if not handler.done():
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_disconnect_cancel_hang_latches_remote_fault_without_stopping_sonic(
    tmp_path: Path,
) -> None:
    class HangingCancelAdapter(FakeRosAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.execute_started = asyncio.Event()
            self.cancel_started = asyncio.Event()
            self.safe_stop_called = asyncio.Event()

        async def execute(
            self,
            command_id: UUID,
            command: CommandConfig,
            params: dict[str, Any],
        ) -> ExecutionResult:
            del command_id, command, params
            self.execute_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def cancel(self, command_id: UUID) -> None:
            self.cancelled.append(command_id)
            self.cancel_started.set()
            await asyncio.Event().wait()

        async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
            result = await super().safe_stop(config)
            self.safe_stop_called.set()
            return result

    ros = HangingCancelAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    ros.safe_stop_called.clear()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound(
            "command.request",
            session_id,
            {
                "command_id": str(uuid4()),
                "kind": "power.enable",
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "ttl_ms": 5000,
                "params": {"data": True},
            },
        )
    )
    await asyncio.wait_for(ros.execute_started.wait(), timeout=1)
    await connection.incoming.put(
        inbound("session.goodbye", session_id, {"reason": "test safe stop"})
    )

    try:
        await asyncio.wait_for(ros.cancel_started.wait(), timeout=1)
        await asyncio.wait_for(handler, timeout=2)
        assert ros.safe_stop_called.is_set() is False
        assert ros.safe_stop_reasons == 1
        replacement = uuid4()
        await runtime._session.connected(replacement)
        with pytest.raises(SessionStateError, match="safety fault is latched"):
            await runtime._session.acquire(
                replacement,
                operator_id="operator-8",
                at=datetime.now(timezone.utc),
            )
    finally:
        if not handler.done():
            handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_bounds_in_flight_command_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)
    release = asyncio.Event()

    async def hold_command(request: object) -> None:
        del request
        await release.wait()

    monkeypatch.setattr(runtime, "_execute_and_report", hold_command)
    for _ in range(runtime_module._MAX_IN_FLIGHT_COMMANDS + 1):
        await connection.incoming.put(
            inbound(
                "command.request",
                session_id,
                {
                    "command_id": str(uuid4()),
                    "kind": "power.enable",
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                    "ttl_ms": 5000,
                    "params": {"data": True},
                },
            )
        )

    full = await next_message(connection)
    assert full.type == "error"
    assert full.payload.code == "COMMAND_QUEUE_FULL"
    assert len(runtime._command_tasks) == runtime_module._MAX_IN_FLIGHT_COMMANDS

    release.set()
    await connection.incoming.put(
        inbound("session.goodbye", session_id, {"reason": "test complete"})
    )
    await asyncio.wait_for(handler, timeout=2)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_back_to_back_request_and_cancel_never_dispatches_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)
    original_register = runtime._ledger.register

    def slow_register(**kwargs: Any) -> Any:
        time.sleep(0.2)
        return original_register(**kwargs)

    monkeypatch.setattr(runtime._ledger, "register", slow_register)
    command_id = uuid4()
    await connection.incoming.put(
        inbound(
            "command.request",
            session_id,
            {
                "command_id": str(command_id),
                "kind": "power.enable",
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "ttl_ms": 5000,
                "params": {"data": True},
            },
        )
    )
    await connection.incoming.put(
        inbound(
            "command.cancel",
            session_id,
            {"command_id": str(command_id), "reason": "operator changed intent"},
        )
    )

    revoked = await next_message(connection)
    assert revoked.type == "control.status"
    assert revoked.payload.authority_revoked is True
    assert revoked.payload.safe_stop_state == "confirmed"
    cancelled = await next_message(connection)
    assert cancelled.type == "command.status"
    assert cancelled.payload.command_id == command_id
    assert cancelled.payload.state == "CANCELLED"
    assert ros.executions == []
    assert ros.safe_stop_reasons == 1

    await connection.incoming.put(
        inbound("session.goodbye", session_id, {"reason": "test complete"})
    )
    await asyncio.wait_for(handler, timeout=2)
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_cancel_terminalizes_after_adapter_quiescence_without_stopping_sonic(
    tmp_path: Path,
) -> None:
    class BlockingCancelBoundaryAdapter(FakeRosAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.execute_started = asyncio.Event()

        async def execute(
            self,
            command_id: UUID,
            command: CommandConfig,
            params: dict[str, Any],
        ) -> ExecutionResult:
            del command_id, command, params
            self.execute_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    ros = BlockingCancelBoundaryAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    await runtime._session.acquire(
        session_id,
        operator_id="operator-7",
        at=datetime.now(timezone.utc),
    )
    command_id = uuid4()
    command = parse_wire_message(
        inbound(
            "command.request",
            session_id,
            {
                "command_id": str(command_id),
                "kind": "power.enable",
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "ttl_ms": 5000,
                "params": {"data": True},
            },
        ),
        expected_robot_id="g1-dev",
    )
    cancellation = parse_wire_message(
        inbound(
            "command.cancel",
            session_id,
            {"command_id": str(command_id), "reason": "operator changed intent"},
        ),
        expected_robot_id="g1-dev",
    )
    assert isinstance(command, CommandRequest)
    assert isinstance(cancellation, CommandCancel)
    submission = asyncio.create_task(
        runtime._coordinator.submit(command, at=datetime.now(timezone.utc))
    )
    await asyncio.wait_for(ros.execute_started.wait(), timeout=1)
    await runtime._route_message(cancellation, session_id)
    await asyncio.wait_for(submission, timeout=1)
    terminal = runtime._ledger.get(command_id)
    assert terminal.state is CommandState.CANCELLED
    assert "physical safe-stop" not in terminal.detail
    assert runtime._session.snapshot.safe_stop_state.value == "confirmed"
    assert runtime._ledger.safe_stop_events()[-1].state.value == "confirmed"
    assert len(runtime._ledger.safe_stop_events()) == 1
    assert ros.safe_stop_reasons == 1
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_failed_physical_safe_stop_blocks_acquire_and_is_diagnostic_truth(
    tmp_path: Path,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    runtime._connection_session_id = session_id
    await runtime._session.acquire(
        session_id,
        operator_id="operator-7",
        at=datetime.now(timezone.utc),
    )
    ros.safe_stop_succeeded = False

    with pytest.raises(RuntimeSafetyError, match="rejected"):
        await runtime._session.force_safe_hold("downstream fault")

    diagnostics = await runtime.diagnostics_snapshot()
    assert diagnostics["controlState"]["canAcquire"] is False
    assert diagnostics["controlState"]["authorityRevoked"] is True
    assert diagnostics["controlState"]["safeStopState"] == "failed"
    assert diagnostics["checks"]["safeStop"]["ready"] is False
    assert diagnostics["checks"]["safeStop"]["reason"] == "safe_stop_failed"
    assert runtime._ledger.safe_stop_events()[-1].state.value == "failed"
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_safe_stop_audit_failure_never_skips_the_physical_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    session_id = uuid4()
    await runtime._session.connected(session_id)
    await runtime._session.acquire(
        session_id,
        operator_id="operator-7",
        at=datetime.now(timezone.utc),
    )

    def unavailable_audit(**_kwargs: Any) -> None:
        raise RuntimeError("disk unavailable")

    monkeypatch.setattr(runtime._ledger, "begin_safe_stop", unavailable_audit)
    initial_physical_stops = ros.safe_stop_reasons
    with pytest.raises(RuntimeSafetyError, match="audit request was not durable"):
        await runtime._session.force_safe_hold("lease expired")

    assert ros.safe_stop_reasons == initial_physical_stops + 1
    assert runtime._session.snapshot.safe_stop_state.value == "failed"
    assert runtime._session.snapshot.can_acquire is False
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_command_background_failure_terminates_active_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ros = FakeRosAdapter()
    runtime = RobotAgentRuntime(runtime_config(tmp_path), ros_adapter=ros)
    await runtime.initialize()
    connection = QueueConnection()
    handler = asyncio.create_task(runtime.handle_connection(connection))
    await next_message(connection)
    session_id = uuid4()
    await connection.incoming.put(
        inbound(
            "session.welcome",
            session_id,
            {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        )
    )
    await next_message(connection)
    await connection.incoming.put(
        inbound("control.acquire", session_id, {"operator_id": "operator-7"})
    )
    await next_message(connection)

    async def fail_status(record: object) -> None:
        del record
        raise RuntimeError("outbound command status failed")

    monkeypatch.setattr(runtime, "_enqueue_command_status", fail_status)
    await connection.incoming.put(
        inbound(
            "command.request",
            session_id,
            {
                "command_id": str(uuid4()),
                "kind": "power.enable",
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "ttl_ms": 5000,
                "params": {"data": True},
            },
        )
    )

    try:
        await asyncio.sleep(0.1)
        assert handler.done() is True
        with pytest.raises(RuntimeError, match="outbound command status failed"):
            await handler
        assert ros.safe_stop_reasons == 1
    finally:
        if not handler.done():
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
        await runtime.shutdown()
