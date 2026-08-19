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
from robot_agent.config import AgentConfig, CommandConfig, SafeStopConfig
from robot_agent.coordinator import CommandCancellationError, ExecutionResult
from robot_agent.ledger import CommandRecord, CommandState
from robot_agent.observation import (
    ObservationBinding,
    ObservationHealthSample,
    ObservationHealthStatus,
)
from robot_agent.protocol import WireMessage, parse_wire_message
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
async def test_failed_control_arm_relatches_stop_and_keeps_safe_hold(
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
    # Startup stop plus the failure relatch.
    assert ros.safe_stop_reasons == 2
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_timed_out_control_arm_does_not_block_the_failure_relatch(
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
    # The timed-out arm task is isolated from the mandatory safety-operation
    # tracker, so startup stop plus the acquire-failure relatch both execute.
    assert ros.safe_stop_reasons == 2
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
    assert ros.safe_stop_reasons == 2
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
async def test_observation_watchdog_cancellation_failure_forces_safe_stop(
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

    monkeypatch.setattr(runtime._coordinator, "cancel", fail_cancel)
    initial_safe_stops = ros.safe_stop_reasons
    try:
        await runtime._check_observation_watchdog()
        assert ros.safe_stop_reasons == initial_safe_stops + 1
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
    assert ros.safe_stop_reasons == 2

    await runtime.shutdown()
    assert ros.closed is True


@pytest.mark.asyncio
async def test_runtime_lease_expiry_emits_safety_and_returns_to_safe_hold(
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

    safety = await next_message(connection)
    assert safety.type == "safety.event"
    assert safety.payload.code == "LEASE_EXPIRED"
    status = await next_message(connection)
    assert status.type == "control.status"
    assert status.payload.state == "SAFE_HOLD"
    assert ros.safe_stop_reasons == 2

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
async def test_lease_monitor_safe_stop_failure_terminates_connection_promptly(
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
        assert handler.done() is True
        with pytest.raises(RuntimeSafetyError, match="mandatory safe stop"):
            await handler
    finally:
        if not handler.done():
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
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
        assert ros.safe_stop_reasons == 2
    finally:
        if not handler.done():
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_safe_stop_invokes_physical_stop_even_when_action_cancel_hangs(
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
        assert ros.safe_stop_called.is_set() is True
        with pytest.raises(RuntimeSafetyError, match="preempt"):
            await asyncio.wait_for(handler, timeout=2)
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

    cancelled = await next_message(connection)
    assert cancelled.type == "command.status"
    assert cancelled.payload.command_id == command_id
    assert cancelled.payload.state == "CANCELLED"
    assert ros.executions == []
    safety = await next_message(connection)
    assert safety.type == "safety.event"
    status = await next_message(connection)
    assert status.type == "control.status"
    assert status.payload.state == "SAFE_HOLD"

    await connection.incoming.put(
        inbound("session.goodbye", session_id, {"reason": "test complete"})
    )
    await asyncio.wait_for(handler, timeout=2)
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
        assert ros.safe_stop_reasons == 2
    finally:
        if not handler.done():
            handler.cancel()
            await asyncio.gather(handler, return_exceptions=True)
        await runtime.shutdown()
