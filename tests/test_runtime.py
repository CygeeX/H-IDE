from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from robot_agent.coordinator import ExecutionResult
from robot_agent.protocol import WireMessage, parse_wire_message
from robot_agent.runtime import RobotAgentRuntime

from robot_agent.config import AgentConfig, CommandConfig, SafeStopConfig


def runtime_config(
    tmp_path: Path,
    *,
    lease_interval_ms: int = 500,
    lease_timeout_ms: int = 2000,
) -> AgentConfig:
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
            "storage": {"ledger_path": str(tmp_path / "ledger.sqlite3")},
            "state": {
                "change_hz": 5.0,
                "snapshot_hz": 1.0,
                "outbound_queue_size": 32,
            },
            "safety": {
                "safe_stop": {
                    "adapter": "service",
                    "ros_name": "/robot/safe_stop",
                    "ros_type": "std_srvs/srv/Trigger",
                    "request": {},
                    "timeout_ms": 1000,
                },
                "heartbeat_topic": "/robot_agent/heartbeat",
                "heartbeat_period_ms": 250,
            },
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
        self.started = False
        self.closed = False
        self.executions: list[tuple[UUID, str, dict[str, Any]]] = []
        self.cancelled: list[UUID] = []
        self.safe_stop_reasons = 0

    def set_callbacks(self, *, state_callback: Any, feedback_callback: Any) -> None:
        self.state_callback = state_callback
        self.feedback_callback = feedback_callback

    def configure_state_sources(self, sources: object) -> None:
        del sources

    def configure_heartbeat(self, *, topic: str, period_ms: int, state_supplier: Any) -> None:
        del topic, period_ms, state_supplier

    def start(self) -> None:
        self.started = True

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

    async def safe_stop(self, config: SafeStopConfig) -> ExecutionResult:
        assert config.ros_name == "/robot/safe_stop"
        self.safe_stop_reasons += 1
        return ExecutionResult(succeeded=True, detail="stopped", result={"success": True})

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
    completed = await next_message(connection)
    assert completed.type == "command.status"
    assert completed.payload.state == "SUCCEEDED"
    assert ros.executions == [(command_id, "/robot/power_enable", {"data": True})]

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
