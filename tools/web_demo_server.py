#!/usr/bin/env python3
"""Local visual BFF for exercising Robot Agent from a browser."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import webbrowser
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pydantic import JsonValue
from robot_agent.protocol import (
    CommandCancel,
    CommandCancelPayload,
    CommandRequest,
    CommandRequestPayload,
    CommandStatus,
    ControlAcquire,
    ControlAcquirePayload,
    ControlStatus,
    ErrorMessage,
    LeaseRenew,
    LeaseRenewPayload,
    LeaseStatus,
    SafetyEvent,
    SessionGoodbye,
    SessionHello,
    SessionWelcome,
    SessionWelcomePayload,
    StateSnapshot,
    StateUpdate,
    WireMessage,
    parse_wire_message,
    serialize_wire_message,
)
from robot_agent.transport import SUBPROTOCOL
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response
from websockets.typing import Origin, Subprotocol

from robot_agent.config import AgentConfig, ClientTransportConfig, load_config

UI_SUBPROTOCOL: Subprotocol = Subprotocol("robot-agent-demo.v1")
MAX_UI_CLIENTS = 8
MAX_EVENTS = 120
MAX_COMMANDS = 50
UI_SEND_TIMEOUT_SECONDS = 0.5
ASSET_DIRECTORY = Path(__file__).with_name("web_demo")
WORKSPACE_DIRECTORY = Path(__file__).resolve().parents[1]
TERMINAL_STATES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "REJECTED",
        "TIMED_OUT",
        "ABORTED_AGENT_RESTART",
    }
)

AssetMap = Mapping[str, tuple[str, bytes]]


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now().isoformat()


@dataclass(frozen=True)
class DemoServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    robot_id: str = "g1-dev"
    operator_id: str = "web-demo-operator"
    lease_interval_ms: int = 500
    lease_timeout_ms: int = 2000
    max_ui_clients: int = MAX_UI_CLIENTS

    def __post_init__(self) -> None:
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("the visual demo server is restricted to a loopback host")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if not 1 <= self.lease_interval_ms <= 5_000:
            raise ValueError("lease_interval_ms must be between 1 and 5000")
        if not self.lease_interval_ms < self.lease_timeout_ms <= 30_000:
            raise ValueError("lease_timeout_ms must be greater than the interval and at most 30000")
        if not 1 <= self.max_ui_clients <= MAX_UI_CLIENTS:
            raise ValueError(f"max_ui_clients must be between 1 and {MAX_UI_CLIENTS}")


class DemoActionError(RuntimeError):
    """Raised when a dashboard action is invalid for the current state."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DashboardBridge:
    """Translate bounded dashboard actions into the strict Robot Agent protocol."""

    def __init__(self, config: DemoServerConfig) -> None:
        self.config = config
        self._agent: ServerConnection | None = None
        self._agent_send_lock = asyncio.Lock()
        self._ui_clients: dict[ServerConnection, asyncio.Lock] = {}
        self._session_id: UUID | None = None
        self._agent_info: dict[str, Any] = {}
        self._control_state = "SAFE_HOLD"
        self._control_reason = "waiting for Robot Agent"
        self._lease_enabled = False
        self._lease_sequence = -1
        self._lease_expires_at: str | None = None
        self._lease_task: asyncio.Task[None] | None = None
        self._robot_state: dict[str, Any] = {}
        self._commands: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)
        self._messages_from_agent = 0
        self._messages_to_agent = 0
        self._last_agent_message_at: str | None = None
        self._started_at = iso_now()
        self._record_event(
            direction="system",
            kind="demo.ready",
            title="可视化 BFF 已启动",
            detail="等待 Robot Agent 连接 /agent",
            level="info",
        )

    @property
    def agent_connected(self) -> bool:
        return self._agent is not None

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "snapshot",
            "generated_at": iso_now(),
            "robot_id": self.config.robot_id,
            "agent": {
                "connected": self.agent_connected,
                **self._agent_info,
            },
            "control": {
                "state": self._control_state,
                "reason": self._control_reason,
            },
            "lease": {
                "enabled": self._lease_enabled,
                "sequence": self._lease_sequence if self._lease_sequence >= 0 else None,
                "expires_at": self._lease_expires_at,
                "interval_ms": self.config.lease_interval_ms,
                "timeout_ms": self.config.lease_timeout_ms,
            },
            "robot_state": dict(self._robot_state),
            "commands": list(reversed(self._commands.values())),
            "events": list(reversed(self._events)),
            "metrics": {
                "messages_from_agent": self._messages_from_agent,
                "messages_to_agent": self._messages_to_agent,
                "last_agent_message_at": self._last_agent_message_at,
                "ui_clients": len(self._ui_clients),
                "started_at": self._started_at,
            },
        }

    def _record_event(
        self,
        *,
        direction: str,
        kind: str,
        title: str,
        detail: str,
        level: str,
    ) -> None:
        self._events.append(
            {
                "event_id": str(uuid4()),
                "at": iso_now(),
                "direction": direction,
                "kind": kind,
                "title": title[:120],
                "detail": detail[:512],
                "level": level,
            }
        )

    async def _send_ui_text(self, peer: ServerConnection, text: str) -> bool:
        lock = self._ui_clients.get(peer)
        if lock is None:
            return False
        try:
            async with lock:
                await asyncio.wait_for(
                    peer.send(text),
                    timeout=UI_SEND_TIMEOUT_SECONDS,
                )
        except (asyncio.TimeoutError, ConnectionClosed, RuntimeError):
            return False
        return True

    async def broadcast_snapshot(self) -> None:
        if not self._ui_clients:
            return
        text = json.dumps(self.snapshot(), ensure_ascii=False, separators=(",", ":"))
        peers = tuple(self._ui_clients)
        results = await asyncio.gather(
            *(self._send_ui_text(peer, text) for peer in peers),
            return_exceptions=False,
        )
        for peer, sent in zip(peers, results, strict=True):
            if not sent:
                self._ui_clients.pop(peer, None)

    async def _send_ui_error(
        self,
        peer: ServerConnection,
        *,
        code: str,
        message: str,
    ) -> None:
        await self._send_ui_text(
            peer,
            json.dumps(
                {"type": "error", "code": code, "message": message, "at": iso_now()},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    async def handle_ui(self, peer: ServerConnection) -> None:
        if len(self._ui_clients) >= self.config.max_ui_clients:
            await peer.close(code=1013, reason="dashboard client limit reached")
            return
        self._ui_clients[peer] = asyncio.Lock()
        self._record_event(
            direction="system",
            kind="ui.connected",
            title="测试网页已连接",
            detail=f"当前 {len(self._ui_clients)} 个网页客户端",
            level="success",
        )
        await self._send_ui_text(
            peer,
            json.dumps(self.snapshot(), ensure_ascii=False, separators=(",", ":")),
        )
        await self.broadcast_snapshot()
        try:
            async for raw in peer:
                if not isinstance(raw, str):
                    await self._send_ui_error(
                        peer,
                        code="INVALID_UI_MESSAGE",
                        message="dashboard accepts JSON text messages only",
                    )
                    continue
                try:
                    request = json.loads(raw)
                    if not isinstance(request, dict):
                        raise DemoActionError(
                            "INVALID_UI_MESSAGE",
                            "dashboard message must be a JSON object",
                        )
                    await self._handle_ui_action(request)
                except json.JSONDecodeError:
                    await self._send_ui_error(
                        peer,
                        code="INVALID_UI_MESSAGE",
                        message="dashboard message is not valid JSON",
                    )
                except DemoActionError as exc:
                    await self._send_ui_error(peer, code=exc.code, message=str(exc))
        finally:
            self._ui_clients.pop(peer, None)
            await self.broadcast_snapshot()

    async def _handle_ui_action(self, request: dict[str, Any]) -> None:
        action = request.get("action")
        if not isinstance(action, str):
            raise DemoActionError("INVALID_UI_ACTION", "action must be a string")
        if action == "control.acquire":
            await self._acquire_control()
        elif action == "lease.set":
            enabled = request.get("enabled")
            if not isinstance(enabled, bool):
                raise DemoActionError("INVALID_UI_ACTION", "lease enabled must be boolean")
            if enabled and self._control_state != "ACTIVE":
                raise DemoActionError(
                    "CONTROL_NOT_ACTIVE",
                    "acquire control before resuming lease renewals",
                )
            await self._set_lease_enabled(enabled)
            self._record_event(
                direction="bff",
                kind="lease.control",
                title="续租已恢复" if enabled else "续租已暂停",
                detail=(
                    f"每 {self.config.lease_interval_ms} ms 续租"
                    if enabled
                    else f"约 {self.config.lease_timeout_ms} ms 后应进入 SAFE_HOLD"
                ),
                level="success" if enabled else "warning",
            )
        elif action == "command.send":
            await self._send_command(request)
        elif action == "command.cancel":
            await self._cancel_command(request)
        elif action == "events.clear":
            self._events.clear()
            self._record_event(
                direction="system",
                kind="events.cleared",
                title="事件记录已清空",
                detail="新的协议事件将继续显示",
                level="info",
            )
        else:
            raise DemoActionError("INVALID_UI_ACTION", f"unsupported dashboard action {action!r}")
        await self.broadcast_snapshot()

    def _require_agent(self) -> tuple[ServerConnection, UUID]:
        if self._agent is None or self._session_id is None:
            raise DemoActionError("AGENT_OFFLINE", "Robot Agent is not connected")
        return self._agent, self._session_id

    async def _send_agent_message(
        self,
        message: WireMessage,
        *,
        record_event: bool = True,
    ) -> None:
        agent, _ = self._require_agent()
        try:
            async with self._agent_send_lock:
                if self._agent is not agent:
                    raise DemoActionError("AGENT_OFFLINE", "Robot Agent disconnected")
                await agent.send(serialize_wire_message(message))
        except ConnectionClosed as exc:
            raise DemoActionError("AGENT_OFFLINE", "Robot Agent disconnected") from exc
        self._messages_to_agent += 1
        if record_event:
            self._record_event(
                direction="bff",
                kind=message.type,
                title=f"BFF → Agent · {message.type}",
                detail=_outgoing_detail(message),
                level="command",
            )

    async def _acquire_control(self) -> None:
        _, session_id = self._require_agent()
        message = ControlAcquire(
            version="1.0",
            type="control.acquire",
            message_id=uuid4(),
            robot_id=self.config.robot_id,
            session_id=session_id,
            sent_at=now(),
            payload=ControlAcquirePayload(operator_id=self.config.operator_id),
        )
        await self._send_agent_message(message)
        await self._set_lease_enabled(True)

    async def _set_lease_enabled(self, enabled: bool) -> None:
        if enabled:
            self._require_agent()
            self._lease_enabled = True
            if self._lease_task is None or self._lease_task.done():
                self._lease_task = asyncio.create_task(self._renew_lease())
            return
        self._lease_enabled = False
        self._lease_expires_at = None
        task = self._lease_task
        self._lease_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _renew_lease(self) -> None:
        try:
            while self._lease_enabled:
                await asyncio.sleep(self.config.lease_interval_ms / 1000.0)
                _, session_id = self._require_agent()
                self._lease_sequence += 1
                renewal = LeaseRenew(
                    version="1.0",
                    type="lease.renew",
                    message_id=uuid4(),
                    robot_id=self.config.robot_id,
                    session_id=session_id,
                    sent_at=now(),
                    payload=LeaseRenewPayload(sequence=self._lease_sequence),
                )
                await self._send_agent_message(renewal, record_event=False)
        except (DemoActionError, ConnectionClosed):
            self._lease_enabled = False
        finally:
            if self._lease_task is asyncio.current_task():
                self._lease_task = None

    async def _send_command(self, request: dict[str, Any]) -> None:
        _, session_id = self._require_agent()
        if self._control_state != "ACTIVE":
            raise DemoActionError(
                "CONTROL_NOT_ACTIVE",
                "commands are disabled until control is ACTIVE",
            )
        kind = request.get("kind")
        params = request.get("params")
        command_params: dict[str, JsonValue]
        if kind == "power.enable":
            if (
                not isinstance(params, dict)
                or set(params) != {"data"}
                or not isinstance(params["data"], bool)
            ):
                raise DemoActionError(
                    "INVALID_COMMAND",
                    "power.enable requires boolean params.data",
                )
            command_params = {"data": params["data"]}
            ttl_ms = 5_000
        elif kind == "task.execute":
            if not isinstance(params, dict):
                raise DemoActionError(
                    "INVALID_COMMAND",
                    "task.execute requires an object in params",
                )
            order = params.get("order")
            if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 100:
                raise DemoActionError(
                    "INVALID_COMMAND",
                    "task.execute requires integer params.order from 1 to 100",
                )
            if set(params) != {"order"}:
                raise DemoActionError("INVALID_COMMAND", "task.execute accepts params.order only")
            command_params = {"order": order}
            ttl_ms = 10_000
        else:
            raise DemoActionError(
                "INVALID_COMMAND",
                "the visual demo permits power.enable and task.execute only",
            )

        command_id = uuid4()
        issued_at = now()
        message = CommandRequest(
            version="1.0",
            type="command.request",
            message_id=uuid4(),
            robot_id=self.config.robot_id,
            session_id=session_id,
            sent_at=issued_at,
            payload=CommandRequestPayload(
                command_id=command_id,
                kind=kind,
                issued_at=issued_at,
                ttl_ms=ttl_ms,
                params=command_params,
            ),
        )
        command_key = str(command_id)
        self._commands[command_key] = {
            "command_id": command_key,
            "kind": kind,
            "params": command_params,
            "state": "REQUESTED",
            "detail": "sent to Robot Agent",
            "progress": None,
            "result": None,
            "issued_at": issued_at.isoformat(),
            "updated_at": issued_at.isoformat(),
        }
        while len(self._commands) > MAX_COMMANDS:
            self._commands.popitem(last=False)
        try:
            await self._send_agent_message(message)
        except DemoActionError:
            self._commands[command_key]["state"] = "SEND_FAILED"
            self._commands[command_key]["detail"] = "Robot Agent disconnected before send"
            self._commands[command_key]["updated_at"] = iso_now()
            raise

    async def _cancel_command(self, request: dict[str, Any]) -> None:
        _, session_id = self._require_agent()
        raw_command_id = request.get("command_id")
        try:
            command_id = UUID(raw_command_id) if isinstance(raw_command_id, str) else None
        except ValueError as exc:
            raise DemoActionError("INVALID_COMMAND", "command_id must be a UUID") from exc
        if command_id is None or str(command_id) not in self._commands:
            raise DemoActionError("INVALID_COMMAND", "select a known command to cancel")
        record = self._commands[str(command_id)]
        if record["state"] in TERMINAL_STATES:
            raise DemoActionError("COMMAND_NOT_ACTIVE", "the selected command is already terminal")
        message = CommandCancel(
            version="1.0",
            type="command.cancel",
            message_id=uuid4(),
            robot_id=self.config.robot_id,
            session_id=session_id,
            sent_at=now(),
            payload=CommandCancelPayload(
                command_id=command_id,
                reason="cancelled from visual test dashboard",
            ),
        )
        await self._send_agent_message(message)

    async def handle_agent(self, peer: ServerConnection) -> None:
        if self._agent is not None:
            await peer.close(code=1013, reason="another Robot Agent is already connected")
            return
        self._agent = peer
        self._control_state = "SAFE_HOLD"
        self._control_reason = "application handshake in progress"
        try:
            raw = await asyncio.wait_for(peer.recv(), timeout=5.0)
            message = parse_wire_message(raw, expected_robot_id=self.config.robot_id)
            if not isinstance(message, SessionHello):
                raise DemoActionError(
                    "INVALID_AGENT_HANDSHAKE",
                    "Robot Agent must initiate with session.hello",
                )
            self._messages_from_agent += 1
            self._last_agent_message_at = iso_now()
            self._session_id = uuid4()
            self._agent_info = {
                "agent_version": message.payload.agent_version,
                "boot_id": str(message.payload.boot_id),
                "transport_mode": message.payload.transport_mode,
                "capabilities_digest": message.payload.capabilities_digest,
            }
            welcome = SessionWelcome(
                version="1.0",
                type="session.welcome",
                message_id=uuid4(),
                robot_id=self.config.robot_id,
                session_id=self._session_id,
                sent_at=now(),
                payload=SessionWelcomePayload(
                    accepted_version="1.0",
                    lease_interval_ms=self.config.lease_interval_ms,
                    lease_timeout_ms=self.config.lease_timeout_ms,
                    control_state="SAFE_HOLD",
                ),
            )
            self._record_event(
                direction="agent",
                kind="session.hello",
                title="Agent → BFF · session.hello",
                detail=(
                    f"boot {message.payload.boot_id} · capabilities "
                    f"{message.payload.capabilities_digest[:19]}…"
                ),
                level="success",
            )
            await self._send_agent_message(welcome)
            self._control_reason = "new session requires explicit control.acquire"
            await self.broadcast_snapshot()

            async for raw_message in peer:
                parsed = parse_wire_message(raw_message, expected_robot_id=self.config.robot_id)
                if parsed.session_id != self._session_id:
                    raise DemoActionError(
                        "SESSION_MISMATCH",
                        "Robot Agent message used an unexpected session_id",
                    )
                self._messages_from_agent += 1
                self._last_agent_message_at = iso_now()
                await self._consume_agent_message(parsed)
                await self.broadcast_snapshot()
        except asyncio.TimeoutError:
            self._record_event(
                direction="system",
                kind="agent.handshake_timeout",
                title="Agent 握手超时",
                detail="5 秒内没有收到 session.hello",
                level="danger",
            )
            await peer.close(code=1002, reason="session.hello timeout")
        except DemoActionError as exc:
            self._record_event(
                direction="system",
                kind=exc.code.lower(),
                title="Agent 协议错误",
                detail=str(exc),
                level="danger",
            )
            await peer.close(code=1002, reason=str(exc)[:120])
        except (ConnectionClosed, ValueError) as exc:
            self._record_event(
                direction="system",
                kind="agent.disconnected",
                title="Robot Agent 连接已断开",
                detail=str(exc) or "WebSocket connection closed",
                level="warning",
            )
        finally:
            if self._agent is peer:
                await self._set_lease_enabled(False)
                self._agent = None
                self._session_id = None
                self._agent_info = {}
                self._control_state = "SAFE_HOLD"
                self._control_reason = "Robot Agent disconnected; local safe-stop expected"
                self._lease_sequence = -1
                self._robot_state = {}
                await self.broadcast_snapshot()

    async def _consume_agent_message(self, message: WireMessage) -> None:
        level = "info"
        detail = _incoming_detail(message)
        if isinstance(message, ControlStatus):
            self._control_state = message.payload.state
            self._control_reason = message.payload.reason
            level = "success" if message.payload.state == "ACTIVE" else "warning"
            if message.payload.state != "ACTIVE" and self._lease_enabled:
                await self._set_lease_enabled(False)
        elif isinstance(message, LeaseStatus):
            self._lease_sequence = message.payload.sequence
            self._lease_expires_at = message.payload.expires_at.isoformat()
        elif isinstance(message, CommandStatus):
            command_key = str(message.payload.command_id)
            existing = self._commands.get(command_key, {})
            existing.update(
                {
                    "command_id": command_key,
                    "kind": existing.get("kind", "unknown"),
                    "params": existing.get("params", {}),
                    "state": message.payload.state,
                    "detail": message.payload.detail,
                    "progress": message.payload.progress,
                    "result": message.payload.result,
                    "issued_at": existing.get("issued_at", message.sent_at.isoformat()),
                    "updated_at": message.sent_at.isoformat(),
                }
            )
            self._commands[command_key] = existing
            level = (
                "success"
                if message.payload.state == "SUCCEEDED"
                else "danger"
                if message.payload.state in TERMINAL_STATES
                else "command"
            )
        elif isinstance(message, StateSnapshot):
            self._robot_state = dict(message.payload.state)
        elif isinstance(message, StateUpdate):
            self._robot_state.update(message.payload.changes)
        elif isinstance(message, SafetyEvent):
            self._control_state = message.payload.state
            self._control_reason = message.payload.reason
            level = "danger" if message.payload.latched else "warning"
        elif isinstance(message, ErrorMessage):
            level = "danger"
        elif isinstance(message, SessionGoodbye):
            level = "warning"

        if not isinstance(message, LeaseStatus):
            self._record_event(
                direction="agent",
                kind=message.type,
                title=f"Agent → BFF · {message.type}",
                detail=detail,
                level=level,
            )

    async def close(self) -> None:
        await self._set_lease_enabled(False)
        agent = self._agent
        if agent is not None:
            await agent.close(code=1001, reason="visual demo server stopping")
        peers = tuple(self._ui_clients)
        await asyncio.gather(
            *(peer.close(code=1001, reason="visual demo server stopping") for peer in peers),
            return_exceptions=True,
        )
        self._ui_clients.clear()


def _incoming_detail(message: WireMessage) -> str:
    if isinstance(message, ControlStatus):
        return f"{message.payload.state} · {message.payload.reason}"
    if isinstance(message, CommandStatus):
        return f"{message.payload.state} · {message.payload.detail}"
    if isinstance(message, StateSnapshot):
        return f"snapshot #{message.payload.sequence} · {len(message.payload.state)} fields"
    if isinstance(message, StateUpdate):
        return f"update #{message.payload.sequence} · {len(message.payload.changes)} changes"
    if isinstance(message, SafetyEvent):
        return f"{message.payload.code} · {message.payload.reason}"
    if isinstance(message, ErrorMessage):
        return f"{message.payload.code} · {message.payload.message}"
    if isinstance(message, SessionGoodbye):
        return message.payload.reason
    return message.type


def _outgoing_detail(message: WireMessage) -> str:
    if isinstance(message, SessionWelcome):
        return f"session {message.session_id} · initial SAFE_HOLD"
    if isinstance(message, ControlAcquire):
        return f"operator {message.payload.operator_id}"
    if isinstance(message, CommandRequest):
        return f"{message.payload.kind} · command {message.payload.command_id}"
    if isinstance(message, CommandCancel):
        return f"cancel {message.payload.command_id}"
    return message.type


def load_assets(directory: Path = ASSET_DIRECTORY) -> dict[str, tuple[str, bytes]]:
    files = {
        "/": ("text/html; charset=utf-8", "index.html"),
        "/index.html": ("text/html; charset=utf-8", "index.html"),
        "/styles.css": ("text/css; charset=utf-8", "styles.css"),
        "/app.js": ("text/javascript; charset=utf-8", "app.js"),
    }
    assets: dict[str, tuple[str, bytes]] = {}
    for route, (content_type, filename) in files.items():
        assets[route] = (content_type, (directory / filename).read_bytes())
    return assets


def _secure_response(
    connection: ServerConnection,
    status: HTTPStatus,
    body: bytes,
    *,
    content_type: str,
) -> Response:
    response = connection.respond(status, body.decode("utf-8"))
    del response.headers["Content-Type"]
    response.headers["Content-Type"] = content_type
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self' ws: wss:; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    )
    return response


def create_server(
    bridge: DashboardBridge,
    config: DemoServerConfig,
    assets: AssetMap,
) -> Any:
    async def process_request(
        connection: ServerConnection,
        request: Request,
    ) -> Response | None:
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        path = urlsplit(request.path).path
        if path == "/healthz":
            body = json.dumps(
                {"agent_connected": bridge.agent_connected, "status": "ok"},
                separators=(",", ":"),
            ).encode("utf-8")
            return _secure_response(
                connection,
                HTTPStatus.OK,
                body,
                content_type="application/json; charset=utf-8",
            )
        asset = assets.get(path)
        if asset is None:
            return _secure_response(
                connection,
                HTTPStatus.NOT_FOUND,
                b"not found\n",
                content_type="text/plain; charset=utf-8",
            )
        content_type, body = asset
        return _secure_response(
            connection,
            HTTPStatus.OK,
            body,
            content_type=content_type,
        )

    async def handler(connection: ServerConnection) -> None:
        request_path = connection.request.path if connection.request is not None else ""
        path = urlsplit(request_path).path
        if path == "/agent" and connection.subprotocol == SUBPROTOCOL:
            await bridge.handle_agent(connection)
            return
        if path == "/ui" and connection.subprotocol == UI_SUBPROTOCOL:
            await bridge.handle_ui(connection)
            return
        await connection.close(code=1008, reason="invalid demo endpoint or subprotocol")

    return serve(
        handler,
        config.host,
        config.port,
        origins=_allowed_origins(config),
        subprotocols=[SUBPROTOCOL, UI_SUBPROTOCOL],
        compression=None,
        process_request=process_request,
        max_size=65_536,
        max_queue=16,
        write_limit=32_768,
        server_header="Robot-Agent-Visual-Demo/1.0",
    )


def _allowed_origins(config: DemoServerConfig) -> list[Origin | None]:
    ipv6_origin = f"http://[{config.host}]:{config.port}"
    configured_origin = (
        ipv6_origin if ":" in config.host else f"http://{config.host}:{config.port}"
    )
    origins = {
        Origin(configured_origin),
        Origin(f"http://127.0.0.1:{config.port}"),
        Origin(f"http://localhost:{config.port}"),
    }
    return [None, *sorted(origins)]


def _load_agent_config(path: Path) -> tuple[str, AgentConfig]:
    resolved = str(path.resolve())
    return resolved, load_config(resolved)


async def _launch_local_stack(args: argparse.Namespace) -> list[asyncio.subprocess.Process]:
    workspace = WORKSPACE_DIRECTORY
    agent_config_path, agent_config = await asyncio.to_thread(
        _load_agent_config,
        args.agent_config,
    )
    if not isinstance(agent_config.transport, ClientTransportConfig):
        raise DemoActionError(
            "INVALID_DEMO_CONFIG",
            "--launch-stack requires Robot Agent client transport",
        )
    if agent_config.robot_id != args.robot_id:
        raise DemoActionError(
            "INVALID_DEMO_CONFIG",
            "--robot-id must match the Robot Agent configuration",
        )
    if (
        agent_config.session.lease_interval_ms != args.lease_interval_ms
        or agent_config.session.lease_timeout_ms != args.lease_timeout_ms
    ):
        raise DemoActionError(
            "INVALID_DEMO_CONFIG",
            "demo lease arguments must match the Robot Agent configuration",
        )
    endpoint = urlsplit(agent_config.transport.uri)
    endpoint_port = endpoint.port or (443 if endpoint.scheme == "wss" else 80)
    if endpoint.hostname not in {"127.0.0.1", "localhost", "::1"} or (
        endpoint_port != args.port or endpoint.path != "/agent"
    ):
        raise DemoActionError(
            "INVALID_DEMO_CONFIG",
            "Agent config must target the demo server loopback port and /agent path",
        )

    environment = os.environ.copy()
    environment.pop("CYCLONEDDS_URI", None)
    environment["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    environment["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    source_path = str(workspace / "src" / "robot_agent")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_path}:{existing_pythonpath}" if existing_pythonpath else source_path
    )
    mock_robot = await asyncio.create_subprocess_exec(
        sys.executable,
        str(workspace / "tools" / "mock_robot.py"),
        cwd=workspace,
        env=environment,
        start_new_session=True,
    )
    try:
        await asyncio.sleep(1.0)
        agent = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "robot_agent.cli",
            "run",
            "--config",
            agent_config_path,
            cwd=workspace,
            env=environment,
            start_new_session=True,
        )
    except BaseException:
        await _stop_local_stack([mock_robot])
        raise
    return [agent, mock_robot]


async def _stop_local_stack(processes: list[asyncio.subprocess.Process]) -> None:
    for process in processes:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.send_signal(signal.SIGINT)
    for process in processes:
        if process.returncode is not None:
            continue
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()


async def run(args: argparse.Namespace) -> None:
    if not 0 <= args.ros_domain_id <= 232:
        raise DemoActionError(
            "INVALID_DEMO_CONFIG",
            "--ros-domain-id must be between 0 and 232",
        )
    config = DemoServerConfig(
        host=args.host,
        port=args.port,
        robot_id=args.robot_id,
        operator_id=args.operator_id,
        lease_interval_ms=args.lease_interval_ms,
        lease_timeout_ms=args.lease_timeout_ms,
    )
    bridge = DashboardBridge(config)
    processes: list[asyncio.subprocess.Process] = []
    async with create_server(bridge, config, load_assets()):
        display_host = f"[{args.host}]" if ":" in args.host else args.host
        url = f"http://{display_host}:{args.port}/"
        print(f"Robot Agent visual demo: {url}", flush=True)
        try:
            if args.launch_stack:
                processes = await _launch_local_stack(args)
                print("Mock ROS graph and Robot Agent started", flush=True)
            if args.open_browser:
                await asyncio.to_thread(webbrowser.open, url)
            await asyncio.Future()
        finally:
            await bridge.close()
            await _stop_local_stack(processes)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--robot-id", default="g1-dev")
    result.add_argument("--operator-id", default="web-demo-operator")
    result.add_argument("--lease-interval-ms", type=int, default=500)
    result.add_argument("--lease-timeout-ms", type=int, default=2000)
    result.add_argument(
        "--ros-domain-id",
        type=int,
        default=232,
        help="isolated ROS domain used by --launch-stack (default: 232)",
    )
    result.add_argument(
        "--launch-stack",
        action="store_true",
        help="also launch the local Mock ROS graph and Robot Agent",
    )
    result.add_argument(
        "--agent-config",
        type=Path,
        default=Path("src/robot_agent/config/robot_agent.local.yaml"),
    )
    result.add_argument("--open-browser", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
