from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from websockets.exceptions import ConnectionClosed

import robot_agent.runtime as runtime_module
from robot_agent.config import AgentConfig, ClientTransportConfig, TLSConfig
from robot_agent.ledger import CommandLedger, CommandNotFound, CommandState
from robot_agent.outbound import PriorityOutboundQueue
from robot_agent.protocol import (
    CommandStatus,
    ControlStatus,
    ErrorMessage,
    SessionHello,
    WireMessage,
    parse_wire_message,
)
from robot_agent.ros_adapter import RosAdapter, heartbeat_qos
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.session import ControlState
from robot_agent.transport import AgentConnection, ClientTransport


class SoakRosGraph:
    def __init__(self) -> None:
        suffix = uuid4().hex[:8]
        self.prefix = f"/robot_agent_soak_{suffix}"
        self.command_service_name = f"{self.prefix}/command"
        self.safe_stop_service_name = f"{self.prefix}/safe_stop"
        self.state_topic = f"{self.prefix}/state"
        self.heartbeat_topic = f"{self.prefix}/heartbeat"
        self.context = Context()
        rclpy.init(context=self.context)
        self.node = Node(f"robot_agent_soak_{suffix}", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.running = threading.Event()
        self.thread = threading.Thread(
            target=self._spin,
            name=f"robot-agent-soak-graph-{suffix}",
            daemon=False,
        )
        self.command_calls = 0
        self.safe_stop_calls = 0
        self.heartbeat_calls = 0
        self.state_sequence = 0
        callback_group = ReentrantCallbackGroup()

        def command(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            del request
            self.command_calls += 1
            response.success = True
            response.message = "soak command completed"
            return response

        def safe_stop(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            del request
            self.safe_stop_calls += 1
            response.success = True
            response.message = "soak safe stop completed"
            return response

        self.command_service = self.node.create_service(
            Trigger,
            self.command_service_name,
            command,
            callback_group=callback_group,
        )
        self.safe_stop_service = self.node.create_service(
            Trigger,
            self.safe_stop_service_name,
            safe_stop,
            callback_group=callback_group,
        )
        self.state_publisher = self.node.create_publisher(String, self.state_topic, 1)
        self.state_timer = self.node.create_timer(0.05, self._publish_state)

        def heartbeat(message: String) -> None:
            payload = json.loads(message.data)
            assert payload["sequence"] > 0
            self.heartbeat_calls += 1

        self.heartbeat_subscription = self.node.create_subscription(
            String,
            self.heartbeat_topic,
            heartbeat,
            heartbeat_qos(),
            callback_group=callback_group,
        )

    def _publish_state(self) -> None:
        self.state_sequence += 1
        message = String()
        message.data = str(self.state_sequence)
        self.state_publisher.publish(message)

    def start(self) -> None:
        self.running.set()
        self.thread.start()

    def _spin(self) -> None:
        while self.running.is_set():
            self.executor.spin_once(timeout_sec=0.1)

    def close(self) -> None:
        self.running.clear()
        self.executor.wake()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            raise RuntimeError("soak ROS graph executor did not stop")
        self.executor.shutdown(timeout_sec=2.0)
        self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()


@dataclass
class ActiveSession:
    connection: AgentConnection
    session_id: UUID
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    command_ids: set[UUID] = field(default_factory=set)
    renew_task: asyncio.Task[None] | None = None
    reader_task: asyncio.Task[None] | None = None
    queue: PriorityOutboundQueue | None = None


def resident_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def outbound(
    message_type: str,
    session_id: UUID,
    payload: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "version": "1.0",
            "type": message_type,
            "message_id": str(uuid4()),
            "robot_id": "g1-soak",
            "session_id": str(session_id),
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        },
        separators=(",", ":"),
    )


def soak_config(tmp_path: Path, graph: SoakRosGraph, *, port: int) -> AgentConfig:
    return AgentConfig.model_validate(
        {
            "schema_version": "1",
            "robot_id": "g1-soak",
            "transport": {
                "mode": "server",
                "host": "127.0.0.1",
                "port": port,
            },
            "tls": {"enabled": False, "allow_plaintext_loopback": True},
            "session": {
                "lease_interval_ms": 200,
                "lease_timeout_ms": 1000,
                "max_message_age_ms": 5000,
                "max_future_skew_ms": 1000,
            },
            "storage": {
                "ledger_path": str(tmp_path / "soak-ledger.sqlite3"),
                "retention_days": 1,
            },
            "state": {
                "change_hz": 5.0,
                "snapshot_hz": 1.0,
                "outbound_queue_size": 64,
            },
            "safety": {
                "safe_stop": {
                    "adapter": "service",
                    "ros_name": graph.safe_stop_service_name,
                    "ros_type": "std_srvs/srv/Trigger",
                    "request": {},
                    "timeout_ms": 3000,
                },
                "heartbeat_topic": graph.heartbeat_topic,
                "heartbeat_period_ms": 250,
            },
            "commands": {
                "soak.trigger": {
                    "adapter": "service",
                    "ros_name": graph.command_service_name,
                    "ros_type": "std_srvs/srv/Trigger",
                    "request_schema": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                    "max_ttl_ms": 5000,
                    "timeout_ms": 3000,
                    "resources": ["soak"],
                }
            },
            "state_sources": {
                "robot.sequence": {
                    "topic": graph.state_topic,
                    "ros_type": "std_msgs/msg/String",
                    "field": "data",
                    "qos": {
                        "reliability": "reliable",
                        "durability": "volatile",
                        "depth": 1,
                    },
                }
            },
        }
    )


@pytest.mark.soak
@pytest.mark.asyncio
async def test_runtime_interfaces_remain_bounded_for_30_minutes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.environ.get("ROBOT_AGENT_RUN_SOAK") != "1":
        pytest.skip("set ROBOT_AGENT_RUN_SOAK=1 for the 30-minute acceptance soak")
    duration = float(os.environ.get("ROBOT_AGENT_SOAK_SECONDS", "1800"))
    assert duration > 0
    graph = SoakRosGraph()
    graph.start()
    config = soak_config(tmp_path, graph, port=unused_port())
    runtime = RobotAgentRuntime(config)
    stop = asyncio.Event()
    runtime_task: asyncio.Task[None] | None = None
    active: ActiveSession | None = None
    wire_counts: Counter[str] = Counter()
    errors: list[str] = []
    terminal_states: dict[UUID, str] = {}
    command_ids: set[UUID] = set()
    queues: dict[int, PriorityOutboundQueue] = {}
    lease_renewals = 0
    sessions_opened = 0
    max_queue_depth = 0
    max_command_tasks = 0
    max_ros_request_queue = 0
    max_event_loop_lag = 0.0
    max_work_cycle_lag = 0.0
    retention_pruned = threading.Event()
    original_prune = runtime._ledger.prune_terminal_before

    def observed_prune(*, before: datetime) -> int:
        pruned = original_prune(before=before)
        if pruned > 0:
            retention_pruned.set()
        return pruned

    monkeypatch.setattr(runtime._ledger, "prune_terminal_before", observed_prune)
    retention_interval = min(5.0, max(0.25, duration / 4.0))
    monkeypatch.setattr(
        runtime_module,
        "_RETENTION_CHECK_INTERVAL_SECONDS",
        retention_interval,
    )

    def observe(message: WireMessage) -> None:
        wire_counts[message.type] += 1
        if isinstance(message, ErrorMessage):
            errors.append(f"{message.payload.code}: {message.payload.message}")
        elif isinstance(message, CommandStatus) and message.payload.state in {
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
            "REJECTED",
            "TIMED_OUT",
            "ABORTED_AGENT_RESTART",
        }:
            terminal_states[message.payload.command_id] = message.payload.state

    async def receive_until(
        connection: AgentConnection,
        predicate: Any,
        *,
        timeout: float = 5.0,
    ) -> WireMessage:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise AssertionError("timed out waiting for Robot Agent protocol message")
            raw = await asyncio.wait_for(connection.recv_text(), timeout=remaining)
            message = parse_wire_message(raw, expected_robot_id="g1-soak")
            observe(message)
            if predicate(message):
                return message

    async def open_session() -> ActiveSession:
        nonlocal lease_renewals, sessions_opened
        transport = ClientTransport(
            ClientTransportConfig(
                mode="client",
                uri=f"ws://127.0.0.1:{config.transport.port}",
                reconnect_initial_ms=50,
                reconnect_max_ms=100,
            ),
            TLSConfig(enabled=False, allow_plaintext_loopback=True),
        )
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            connection: AgentConnection | None = None
            try:
                connection = await transport.connect_once()
                hello = await receive_until(
                    connection,
                    lambda message: isinstance(message, SessionHello),
                    timeout=1.0,
                )
                assert isinstance(hello, SessionHello)
                assert hello.payload.transport_mode == "server"
                session_id = uuid4()
                await connection.send_text(
                    outbound(
                        "session.welcome",
                        session_id,
                        {
                            "accepted_version": "1.0",
                            "lease_interval_ms": 200,
                            "lease_timeout_ms": 1000,
                            "control_state": "SAFE_HOLD",
                        },
                    )
                )
                safe_hold = await receive_until(
                    connection,
                    lambda message: (
                        isinstance(message, ControlStatus)
                        and message.payload.state == "SAFE_HOLD"
                    ),
                )
                assert isinstance(safe_hold, ControlStatus)
                await connection.send_text(
                    outbound(
                        "control.acquire",
                        session_id,
                        {"operator_id": f"soak-{sessions_opened + 1}"},
                    )
                )
                active_status = await receive_until(
                    connection,
                    lambda message: (
                        isinstance(message, ControlStatus)
                        and message.payload.state == "ACTIVE"
                    ),
                )
                assert isinstance(active_status, ControlStatus)
                link = ActiveSession(
                    connection=connection,
                    session_id=session_id,
                    queue=runtime._outbound,
                )
                sessions_opened += 1

                async def renew(session_link: ActiveSession = link) -> None:
                    nonlocal lease_renewals
                    sequence = 0
                    while True:
                        await asyncio.sleep(0.2)
                        async with session_link.send_lock:
                            await session_link.connection.send_text(
                                outbound(
                                    "lease.renew",
                                    session_link.session_id,
                                    {"sequence": sequence},
                                )
                            )
                        sequence += 1
                        lease_renewals += 1

                async def read(session_link: ActiveSession = link) -> None:
                    try:
                        while True:
                            raw = await session_link.connection.recv_text()
                            observe(parse_wire_message(raw, expected_robot_id="g1-soak"))
                    except ConnectionClosed:
                        return

                link.renew_task = asyncio.create_task(renew())
                link.reader_task = asyncio.create_task(read())
                return link
            except (ConnectionClosed, ConnectionRefusedError, OSError, asyncio.TimeoutError):
                if connection is not None:
                    await connection.close()
                if asyncio.get_running_loop().time() >= deadline:
                    raise
                await asyncio.sleep(0.05)

    async def wait_for_terminals(ids: set[UUID], *, timeout: float = 5.0) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while not ids.issubset(terminal_states):
            if active is not None and active.reader_task is not None and active.reader_task.done():
                await active.reader_task
                raise AssertionError("Robot Agent connection closed before command completion")
            if asyncio.get_running_loop().time() >= deadline:
                missing = sorted(str(item) for item in ids.difference(terminal_states))
                raise AssertionError(f"commands did not reach terminal state: {missing[:3]}")
            await asyncio.sleep(0.01)

    async def close_session(link: ActiveSession) -> None:
        await wait_for_terminals(link.command_ids)
        assert link.renew_task is not None
        assert link.reader_task is not None
        link.renew_task.cancel()
        await asyncio.gather(link.renew_task, return_exceptions=True)
        async with link.send_lock:
            await link.connection.send_text(
                outbound(
                    "session.goodbye",
                    link.session_id,
                    {"reason": "scheduled soak reconnect"},
                )
            )
        await asyncio.wait_for(link.reader_task, timeout=5.0)
        if not link.connection.closed:
            await link.connection.close()
        deadline = asyncio.get_running_loop().time() + 2.0
        while runtime._connection_session_id is not None:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("Robot Agent did not release the prior session")
            await asyncio.sleep(0.01)

    async def lag_monitor() -> None:
        nonlocal max_event_loop_lag
        loop = asyncio.get_running_loop()
        next_probe = loop.time()
        while True:
            next_probe += 0.01
            await asyncio.sleep(max(0.0, next_probe - loop.time()))
            max_event_loop_lag = max(max_event_loop_lag, loop.time() - next_probe)

    lag_task: asyncio.Task[None] | None = None
    started = 0.0
    iterations = 0
    start_rss = 0
    max_rss = 0
    try:
        runtime_task = asyncio.create_task(runtime.run(stop))
        await asyncio.wait_for(runtime.ready.wait(), timeout=10.0)

        expired_id = uuid4()
        expired_at = datetime.now(timezone.utc) - timedelta(days=2)
        await asyncio.to_thread(
            runtime._ledger.register,
            command_id=expired_id,
            kind="soak.expired",
            payload={"expired": True},
            session_id=uuid4(),
            at=expired_at,
        )
        await asyncio.to_thread(
            runtime._ledger.transition,
            expired_id,
            CommandState.RUNNING,
            detail="expired soak fixture running",
            at=expired_at,
        )
        await asyncio.to_thread(
            runtime._ledger.transition,
            expired_id,
            CommandState.SUCCEEDED,
            detail="expired soak fixture",
            at=expired_at,
        )
        assert await asyncio.to_thread(
            retention_pruned.wait,
            retention_interval + 3.0,
        )
        with pytest.raises(CommandNotFound):
            await asyncio.to_thread(runtime._ledger.get, expired_id)

        active = await open_session()
        loop = asyncio.get_running_loop()
        started = loop.time()
        next_tick = started
        next_command = started
        reconnect_interval = max(2.0, min(300.0, duration / 3.0))
        next_reconnect = started + reconnect_interval
        start_rss = resident_bytes()
        max_rss = start_rss
        lag_task = asyncio.create_task(lag_monitor())

        while loop.time() - started < duration:
            iterations += 1
            now = loop.time()
            if runtime_task.done():
                await runtime_task
                raise AssertionError("Robot Agent runtime stopped during soak")
            assert active.reader_task is not None
            assert active.renew_task is not None
            if active.reader_task.done():
                await active.reader_task
                raise AssertionError("Robot Agent reader stopped during soak")
            if active.renew_task.done():
                await active.renew_task
                raise AssertionError("lease renewer stopped during soak")

            if now >= next_reconnect and now + 0.5 < started + duration:
                await close_session(active)
                active = await open_session()
                next_reconnect = now + reconnect_interval
                next_command = loop.time()

            if now >= next_command:
                command_id = uuid4()
                issued_at = datetime.now(timezone.utc)
                async with active.send_lock:
                    await active.connection.send_text(
                        outbound(
                            "command.request",
                            active.session_id,
                            {
                                "command_id": str(command_id),
                                "kind": "soak.trigger",
                                "issued_at": issued_at.isoformat(),
                                "ttl_ms": 5000,
                                "params": {},
                            },
                        )
                    )
                active.command_ids.add(command_id)
                command_ids.add(command_id)
                next_command = now + 0.2

            queue = runtime._outbound
            if queue is not None:
                queues[id(queue)] = queue
                max_queue_depth = max(max_queue_depth, queue.qsize)
            max_command_tasks = max(max_command_tasks, len(runtime._command_tasks))
            ros_adapter = runtime._ros
            assert isinstance(ros_adapter, RosAdapter)
            max_ros_request_queue = max(
                max_ros_request_queue,
                ros_adapter._requests.qsize(),
            )
            if iterations % 10 == 0:
                max_rss = max(max_rss, resident_bytes())

            next_tick += 0.01
            await asyncio.sleep(max(0.0, next_tick - loop.time()))
            max_work_cycle_lag = max(max_work_cycle_lag, loop.time() - next_tick)

        await wait_for_terminals(command_ids)
        await close_session(active)
        active = None
        stop.set()
        await asyncio.wait_for(runtime_task, timeout=10.0)
    finally:
        if lag_task is not None:
            lag_task.cancel()
            await asyncio.gather(lag_task, return_exceptions=True)
        if active is not None:
            if active.renew_task is not None:
                active.renew_task.cancel()
                await asyncio.gather(active.renew_task, return_exceptions=True)
            with contextlib.suppress(Exception):
                await active.connection.close()
            if active.reader_task is not None:
                active.reader_task.cancel()
                await asyncio.gather(active.reader_task, return_exceptions=True)
        stop.set()
        if runtime_task is not None and not runtime_task.done():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(runtime_task, timeout=10.0)
        if runtime_task is not None and not runtime_task.done():
            runtime_task.cancel()
            await asyncio.gather(runtime_task, return_exceptions=True)
        await asyncio.to_thread(graph.close)

    elapsed = asyncio.get_running_loop().time() - started
    ledger_states: dict[UUID, CommandState] = {}
    ledger_details: dict[UUID, str] = {}
    with CommandLedger(config.storage.ledger_path) as ledger:
        for command_id in command_ids:
            record = ledger.get(command_id)
            ledger_states[command_id] = record.state
            ledger_details[command_id] = record.detail
    dropped_count = sum(queue.dropped_count for queue in queues.values())
    coalesced_count = sum(queue.coalesced_count for queue in queues.values())
    succeeded_ids = {
        command_id
        for command_id, state in terminal_states.items()
        if state == "SUCCEEDED"
    }
    rejected_ids = {
        command_id
        for command_id, state in terminal_states.items()
        if state == "REJECTED"
    }
    report = {
        "duration_seconds": round(elapsed, 3),
        "iterations": iterations,
        "sessions_opened": sessions_opened,
        "commands": len(command_ids),
        "terminal_statuses": len(terminal_states),
        "succeeded_commands": len(succeeded_ids),
        "resource_conflict_rejections": len(rejected_ids),
        "lease_renewals": lease_renewals,
        "state_snapshots": wire_counts["state.snapshot"],
        "state_updates": wire_counts["state.update"],
        "start_rss_bytes": start_rss,
        "max_rss_bytes": max_rss,
        "rss_growth_bytes": max_rss - start_rss,
        "max_outbound_queue_depth": max_queue_depth,
        "outbound_dropped_count": dropped_count,
        "outbound_coalesced_count": coalesced_count,
        "max_command_tasks": max_command_tasks,
        "max_ros_request_queue_depth": max_ros_request_queue,
        "max_event_loop_lag_seconds": round(max_event_loop_lag, 6),
        "max_work_cycle_lag_seconds": round(max_work_cycle_lag, 6),
        "ros_command_calls": graph.command_calls,
        "ros_safe_stop_calls": graph.safe_stop_calls,
        "ros_heartbeats": graph.heartbeat_calls,
    }
    print(f"SOAK_RESULT={json.dumps(report, sort_keys=True)}", flush=True)

    assert iterations >= int(duration * 50)
    assert sessions_opened >= 1
    if duration >= 5.0:
        assert sessions_opened >= 2
    assert len(command_ids) >= max(1, int(duration * 3))
    assert set(terminal_states) == command_ids
    assert set(terminal_states.values()) <= {"SUCCEEDED", "REJECTED"}
    assert {
        command_id: state.value for command_id, state in ledger_states.items()
    } == terminal_states
    assert all(
        ledger_details[command_id].startswith("resource conflict: soak=")
        for command_id in rejected_ids
    )
    conflict_owner_ids = {
        UUID(ledger_details[command_id].removeprefix("resource conflict: soak="))
        for command_id in rejected_ids
    }
    assert conflict_owner_ids <= succeeded_ids
    assert graph.command_calls == len(succeeded_ids)
    assert graph.command_calls >= max(1, int(duration * 3))
    assert graph.safe_stop_calls == sessions_opened + 1
    assert graph.heartbeat_calls > 0
    assert wire_counts["lease.status"] > 0
    assert wire_counts["state.snapshot"] > 0
    assert wire_counts["state.update"] > 0
    assert errors == []
    assert retention_pruned.is_set()
    assert max_queue_depth <= config.state.outbound_queue_size
    assert dropped_count == 0
    assert max_command_tasks <= runtime_module._MAX_IN_FLIGHT_COMMANDS
    assert max_ros_request_queue <= runtime._ros._requests.maxsize
    assert max_rss - start_rss < 64 * 1024 * 1024
    assert max_event_loop_lag < 0.5
    assert runtime._session.state is ControlState.SAFE_HOLD
    assert not any(
        thread.name == "robot-agent-rclpy-executor" and thread.is_alive()
        for thread in threading.enumerate()
    )
