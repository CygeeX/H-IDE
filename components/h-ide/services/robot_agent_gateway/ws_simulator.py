"""Protocol-conformant WebSocket peer used only for H-IDE boundary acceptance."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import threading
from typing import Any
from uuid import UUID, uuid4

from websockets.sync.server import Server, ServerConnection, serve

from .protocol import ProtocolCommand, ProtocolSimulator
from .wire import SUBPROTOCOL, build_message, parse_message, parse_timestamp, require_payload_fields


class WebSocketProtocolSimulator:
    """Loopback-only robot-agent.v1 server; it never runs Nav2 or physical motion."""

    def __init__(self, backend: ProtocolSimulator | None = None, *, mode: str = "lab_execute") -> None:
        if mode == "robot_execute":
            raise ValueError("protocol simulator cannot start in robot_execute mode")
        self.backend = backend or ProtocolSimulator()
        self.mode = mode
        self.capabilities = ["navigation.navigate_to_pose", "command.cancel", "command.query"]
        canonical = json.dumps(self.capabilities, sort_keys=True, separators=(",", ":")).encode()
        self.capabilities_digest = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
        self._server: Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def uri(self) -> str:
        if self._server is None:
            raise RuntimeError("protocol simulator is not started")
        host, port = self._server.socket.getsockname()[:2]
        return f"ws://{host}:{port}"

    def start(self) -> WebSocketProtocolSimulator:
        if self._server is not None:
            return self
        self._server = serve(
            self._handle,
            "127.0.0.1",
            0,
            subprotocols=[SUBPROTOCOL],
            compression=None,
            max_size=65_536,
            max_queue=8,
            ping_interval=10,
            ping_timeout=10,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="robot-agent-protocol-simulator",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
        if thread is not None:
            thread.join(timeout=2)

    def __enter__(self) -> WebSocketProtocolSimulator:
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _send(self, websocket: ServerConnection, message_type: str, session_id: str, payload: dict[str, Any]) -> None:
        websocket.send(build_message(
            message_type,
            robot_id=self.backend.robot_id,
            session_id=session_id,
            payload=payload,
        ))

    def _handle(self, websocket: ServerConnection) -> None:
        if websocket.subprotocol != SUBPROTOCOL:
            websocket.close(1002, "robot-agent.v1 subprotocol required")
            return
        websocket.send(build_message(
            "session.hello",
            robot_id=self.backend.robot_id,
            session_id=None,
            payload={
                "agent_version": "protocol-simulator-1",
                "boot_id": self.backend.robot_boot_id,
                "supported_versions": ["1.0"],
                "transport_mode": "server",
                "capabilities_digest": self.capabilities_digest,
            },
        ))
        session_id: str | None = None
        try:
            welcome = parse_message(websocket.recv(), expected_robot_id=self.backend.robot_id)
            if welcome["type"] != "session.welcome":
                raise ValueError("session.welcome required")
            session_id = welcome["session_id"]
            if session_id is None:
                raise ValueError("session.welcome requires session_id")
            require_payload_fields(
                welcome["payload"],
                {"accepted_version", "lease_interval_ms", "lease_timeout_ms", "control_state"},
            )
            if welcome["payload"] != {
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            }:
                raise ValueError("peer lease policy mismatch")
            self._send(websocket, "control.status", session_id, {
                "state": "SAFE_HOLD",
                "reason": "new session requires explicit control.acquire",
                "lease_sequence": 0,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 0,
            })
            for raw in websocket:
                message = parse_message(
                    raw,
                    expected_robot_id=self.backend.robot_id,
                    expected_session_id=session_id,
                )
                payload = message["payload"]
                if message["type"] == "control.acquire":
                    require_payload_fields(payload, {"operator_id"})
                    state = self.backend.acquire(self.backend.robot_id)
                    self._send(websocket, "control.status", session_id, {
                        "state": state["state"],
                        "reason": "simulator control acquired",
                        "lease_sequence": state["leaseSequence"],
                        "authority_revoked": False,
                        "safe_stop_state": "confirmed",
                        "safe_stop_epoch": 0,
                    })
                elif message["type"] == "lease.renew":
                    require_payload_fields(payload, {"sequence"})
                    renewed = self.backend.renew(self.backend.robot_id, int(payload["sequence"]))
                    self._send(websocket, "lease.status", session_id, {
                        "sequence": renewed["sequence"],
                        "expires_at": renewed["expiresAt"],
                    })
                elif message["type"] == "control.release":
                    require_payload_fields(payload, {"reason"})
                    reason = payload["reason"]
                    if (
                        not isinstance(reason, str)
                        or not reason.strip()
                        or len(reason.strip()) > 256
                    ):
                        raise ValueError("control release reason is invalid")
                    released = self.backend.release(
                        self.backend.robot_id,
                        reason,
                    )
                    self._send(websocket, "control.status", session_id, {
                        "state": released["state"],
                        "reason": "simulator control released",
                        "lease_sequence": None,
                        "authority_revoked": released["authorityRevoked"],
                        "safe_stop_state": released["safeStopState"],
                        "safe_stop_epoch": released["safeStopEpoch"],
                    })
                elif message["type"] == "command.request":
                    require_payload_fields(payload, {"command_id", "kind", "issued_at", "ttl_ms", "params"})
                    params = payload["params"]
                    if not isinstance(params, dict):
                        raise ValueError("command params must be an object")
                    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
                    status = self.backend.submit(ProtocolCommand(
                        command_id=str(UUID(payload["command_id"])),
                        robot_id=self.backend.robot_id,
                        kind=str(payload["kind"]),
                        issued_at=parse_timestamp(payload["issued_at"]),
                        ttl_ms=int(payload["ttl_ms"]),
                        params=params,
                        payload_hash=hashlib.sha256(canonical).hexdigest(),
                    ))
                    self._send_status(websocket, session_id, status)
                elif message["type"] == "command.cancel":
                    require_payload_fields(payload, {"command_id", "reason"})
                    self._send_status(
                        websocket,
                        session_id,
                        self.backend.cancel(str(UUID(payload["command_id"]))),
                    )
                elif message["type"] == "session.goodbye":
                    return
                else:
                    raise ValueError(f"unexpected peer message {message['type']}")
        except Exception as exc:
            if session_id is not None:
                try:
                    self._send(websocket, "error", session_id, {
                        "code": "SIMULATOR_PROTOCOL_ERROR",
                        "message": str(exc)[:512] or type(exc).__name__,
                        "related_message_id": None,
                        "retryable": False,
                    })
                except Exception:
                    pass
            websocket.close(1002, "protocol error")
        finally:
            self.backend.disconnect()

    def _send_status(self, websocket: ServerConnection, session_id: str, status) -> None:
        self._send(websocket, "command.status", session_id, {
            "command_id": status.command_id,
            "state": status.state,
            "detail": status.detail,
            "progress": status.progress,
            "result": status.result,
        })
