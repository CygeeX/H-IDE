"""Production-shaped Robot Agent Gateway WebSocket session client."""

from __future__ import annotations

import ipaddress
import ssl
import threading
import time
from collections.abc import Mapping
from datetime import UTC
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

from websockets.sync.client import ClientConnection, connect

from .capability_handshake import validate_optional_capability_handshake
from .protocol import ProtocolCommand, ProtocolCommandStatus
from .wire import (
    SUBPROTOCOL,
    ControlStatusPayload,
    build_message,
    parse_control_status_payload,
    parse_message,
    require_payload_fields,
)


class RobotAgentWebSocketPeer:
    """Synchronous gateway peer implementing handshake, lease, command, query and cancel."""

    backend_name = "robot_agent_websocket"

    def __init__(
        self,
        uri: str,
        *,
        robot_id: str,
        ssl_context: ssl.SSLContext | None = None,
        expected_capabilities_digest: str | None = None,
        capabilities: tuple[str, ...] = (
            "navigation.navigate_to_pose",
            "command.cancel",
            "command.query",
        ),
        execution_backend: str = "real_robot_agent_pending",
        motion: str = "not_validated",
        allow_plaintext_loopback: bool = False,
        heartbeat: bool = True,
    ) -> None:
        parsed = urlparse(uri)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise ValueError("Robot Agent URI must be absolute ws:// or wss://")
        if parsed.scheme == "ws":
            try:
                loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                loopback = parsed.hostname == "localhost"
            if not allow_plaintext_loopback or not loopback:
                raise ValueError("plaintext Robot Agent WebSocket is restricted to explicit loopback development")
        elif ssl_context is None:
            raise ValueError("wss Robot Agent connection requires an explicit mTLS SSL context")
        self.uri = uri
        self.robot_id = robot_id
        self.ssl_context = ssl_context
        self.expected_capabilities_digest = expected_capabilities_digest
        self._configured_capabilities = tuple(capabilities)
        self.capabilities = tuple(capabilities)
        self.execution_backend = execution_backend
        self.motion = motion
        self.heartbeat = heartbeat
        self.robot_boot_id: str | None = None
        self.capabilities_digest: str | None = None
        self.robot_profile: dict[str, Any] | None = None
        self.capability_manifest: dict[str, Any] | None = None
        self.session_id: str | None = None
        self._socket: ClientConnection | None = None
        self._io_lock = threading.RLock()
        self._lease_sequence = 0
        self._control_safe_stop_truth_supported = False
        self._control_safe_stop_epoch: int | None = None
        self._release_baseline_epoch: int | None = None
        self._release_baseline_boot_id: str | None = None
        self._commands: dict[str, ProtocolCommand] = {}
        self._statuses: dict[str, ProtocolCommandStatus] = {}
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def discover(self, robot_id: str | None = None) -> Mapping[str, Any]:
        if robot_id not in {None, self.robot_id}:
            return {"available": False, "reason": "robot identity mismatch"}
        try:
            self._ensure_connected()
        except Exception as exc:
            return {"available": False, "reason": str(exc)}
        return {
            "available": True,
            "protocol": SUBPROTOCOL,
            "robotId": self.robot_id,
            "robotBootId": self.robot_boot_id,
            "capabilitiesDigest": self.capabilities_digest,
            "capabilities": list(self.capabilities),
            "robotProfile": self.robot_profile,
            "capabilityManifest": self.capability_manifest,
            "executionBackend": self.execution_backend,
            "motion": self.motion,
        }

    def acquire(self, robot_id: str) -> Mapping[str, Any]:
        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        with self._io_lock:
            self._ensure_connected_locked()
            self._send("control.acquire", {"operator_id": "h-ide-gateway"})
            deadline = time.monotonic() + 5
            while True:
                message = self._receive("control.status", deadline=deadline)
                status = self._record_control_status(message["payload"])
                if status.state == "ACTIVE":
                    self._lease_sequence = int(status.lease_sequence or 0)
                    break
                if status.state != "SAFE_HOLD":
                    raise RuntimeError(
                        f"Robot Agent control is not active: {status.reason}"
                    )
        self._start_heartbeat()
        return {"state": "ACTIVE", "leaseSequence": self._lease_sequence}

    def renew(self, robot_id: str, sequence: int) -> Mapping[str, Any]:
        del sequence
        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        with self._io_lock:
            self._ensure_connected_locked()
            return self._renew_locked()

    def release(self, robot_id: str, reason: str) -> Mapping[str, Any]:
        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("control release reason is required")
        detail = reason.strip()
        if len(detail) > 256:
            raise ValueError("control release reason is too long")
        self._stop_heartbeat()
        with self._io_lock:
            self._ensure_connected_locked()
            if self._release_baseline_epoch is None:
                if (
                    not self._control_safe_stop_truth_supported
                    or self._control_safe_stop_epoch is None
                ):
                    raise RuntimeError(
                        "Robot Agent control release is unconfirmed: peer does not "
                        "publish atomic safe-stop truth"
                    )
                self._release_baseline_epoch = self._control_safe_stop_epoch
                self._release_baseline_boot_id = self.robot_boot_id
            baseline_epoch = self._release_baseline_epoch
            baseline_boot_id = self._release_baseline_boot_id
            if self.robot_boot_id != baseline_boot_id:
                raise RuntimeError(
                    "Robot Agent restarted before control release was confirmed"
                )
            self._send("control.release", {"reason": detail})
            deadline = time.monotonic() + 5.0
            while True:
                message = self._receive("control.status", deadline=deadline)
                status = self._record_control_status(message["payload"])
                if self.robot_boot_id != baseline_boot_id:
                    raise RuntimeError(
                        "Robot Agent restarted before control release was confirmed"
                    )
                epoch = status.safe_stop_epoch
                fresh = epoch is not None and epoch > baseline_epoch
                if (
                    fresh
                    and status.state == "SAFE_HOLD"
                    and status.authority_revoked is True
                    and status.safe_stop_state == "confirmed"
                ):
                    result = {
                        "state": "SAFE_HOLD",
                        "authorityRevoked": True,
                        "safeStopState": "confirmed",
                        "safeStopEpoch": epoch,
                    }
                    self._release_baseline_epoch = None
                    self._release_baseline_boot_id = None
                    return result
                if (
                    fresh
                    and status.state == "SAFE_HOLD"
                    and status.authority_revoked is True
                    and status.safe_stop_state == "failed"
                ):
                    raise RuntimeError(
                        "Robot Agent physical safe-stop failed during control release"
                    )

    def submit(self, command: ProtocolCommand) -> ProtocolCommandStatus:
        if command.robot_id != self.robot_id:
            return ProtocolCommandStatus(command.command_id, "REJECTED", "robot identity mismatch")
        with self._io_lock:
            self._ensure_connected_locked()
            self._commands[command.command_id] = command
            self._send("command.request", {
                "command_id": str(UUID(command.command_id)),
                "kind": command.kind,
                "issued_at": command.issued_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "ttl_ms": command.ttl_ms,
                "params": dict(command.params),
            })
            try:
                status = self._status(
                    self._receive(
                        "command.status",
                        command_id=command.command_id,
                        deadline=time.monotonic() + 2.0,
                    )
                )
            except TimeoutError:
                status = ProtocolCommandStatus(
                    command.command_id,
                    "RECEIVED",
                    "command sent; awaiting Robot Agent status",
                )
            self._statuses[command.command_id] = status
            return status

    def query(self, command_id: str) -> ProtocolCommandStatus:
        command = self._commands.get(command_id)
        if command is None:
            return ProtocolCommandStatus(command_id, "REJECTED", "command not found in gateway ledger")
        current = self._statuses.get(command_id)
        if current is not None and current.state in {
            "SUCCEEDED",
            "FAILED",
            "CANCELLED",
            "REJECTED",
            "TIMED_OUT",
            "ABORTED_AGENT_RESTART",
        }:
            return current
        with self._io_lock:
            self._ensure_connected_locked()
            self._send("command.request", {
                "command_id": str(UUID(command.command_id)),
                "kind": command.kind,
                "issued_at": command.issued_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "ttl_ms": command.ttl_ms,
                "params": dict(command.params),
            })
            try:
                status = self._status(
                    self._receive(
                        "command.status",
                        command_id=command_id,
                        deadline=time.monotonic() + 2.0,
                    )
                )
            except TimeoutError:
                return current or ProtocolCommandStatus(
                    command_id,
                    "RECEIVED",
                    "command status is still pending",
                )
            self._statuses[command_id] = status
            return status

    def cancel(self, command_id: str) -> ProtocolCommandStatus:
        with self._io_lock:
            self._ensure_connected_locked()
            self._send("command.cancel", {
                "command_id": str(UUID(command_id)),
                "reason": "H-IDE mission or operator cancellation",
            })
            status = self._status(self._receive("command.status", command_id=command_id))
            self._statuses[command_id] = status
            return status

    def close(self) -> None:
        self._stop_heartbeat(timeout=1.0, strict=False)
        with self._io_lock:
            socket = self._socket
            self._socket = None
            if socket is not None:
                try:
                    if self.session_id is not None:
                        socket.send(build_message(
                            "session.goodbye",
                            robot_id=self.robot_id,
                            session_id=self.session_id,
                            payload={"reason": "H-IDE Gateway shutdown"},
                        ))
                finally:
                    socket.close()
            self.session_id = None
            self._control_safe_stop_truth_supported = False
            self._control_safe_stop_epoch = None

    def _ensure_connected(self) -> None:
        with self._io_lock:
            self._ensure_connected_locked()

    def _ensure_connected_locked(self) -> None:
        if self._socket is not None:
            return
        socket = connect(
            self.uri,
            ssl=self.ssl_context,
            subprotocols=[SUBPROTOCOL],
            compression=None,
            proxy=None,
            max_size=65_536,
            max_queue=8,
            open_timeout=5,
        )
        if socket.subprotocol != SUBPROTOCOL:
            socket.close()
            raise RuntimeError("Robot Agent did not negotiate robot-agent.v1")
        hello = parse_message(socket.recv(timeout=5), expected_robot_id=self.robot_id, expected_session_id=None)
        if hello["type"] != "session.hello":
            socket.close()
            raise RuntimeError("Robot Agent must initiate with session.hello")
        payload = hello["payload"]
        require_payload_fields(
            payload,
            {
                "agent_version",
                "boot_id",
                "supported_versions",
                "transport_mode",
                "capabilities_digest",
            },
            optional_fields={"robot_profile", "capability_manifest"},
        )
        boot_id = str(UUID(payload["boot_id"]))
        digest = payload["capabilities_digest"]
        capability_handshake = validate_optional_capability_handshake(
            payload,
            expected_robot_id=self.robot_id,
        )
        if self.expected_capabilities_digest is not None and digest != self.expected_capabilities_digest:
            socket.close()
            raise RuntimeError("Robot Agent capabilities digest mismatch")
        session_id = str(uuid4())
        socket.send(build_message(
            "session.welcome",
            robot_id=self.robot_id,
            session_id=session_id,
            payload={
                "accepted_version": "1.0",
                "lease_interval_ms": 500,
                "lease_timeout_ms": 2000,
                "control_state": "SAFE_HOLD",
            },
        ))
        self._socket = socket
        self.session_id = session_id
        self.robot_boot_id = boot_id
        self.capabilities_digest = digest
        if capability_handshake is None:
            self.capabilities = self._configured_capabilities
            self.robot_profile = None
            self.capability_manifest = None
        else:
            self.capabilities = capability_handshake.capability_ids
            self.robot_profile = capability_handshake.robot_profile.model_dump(mode="json")
            self.capability_manifest = (
                capability_handshake.capability_manifest.model_dump(mode="json")
            )
        self._control_safe_stop_truth_supported = False
        self._control_safe_stop_epoch = None
        self._heartbeat_stop.clear()

    def _record_control_status(
        self,
        payload: Mapping[str, Any],
    ) -> ControlStatusPayload:
        status = parse_control_status_payload(payload)
        if (
            self._control_safe_stop_truth_supported
            and not status.has_safe_stop_truth
        ):
            raise ValueError(
                "control.status omitted safety truth after it was provided"
            )
        if status.has_safe_stop_truth:
            assert status.safe_stop_epoch is not None
            if (
                self._control_safe_stop_epoch is not None
                and status.safe_stop_epoch < self._control_safe_stop_epoch
            ):
                raise ValueError("control.status safe_stop_epoch decreased")
            self._control_safe_stop_truth_supported = True
            self._control_safe_stop_epoch = status.safe_stop_epoch
        return status

    def _send(self, message_type: str, payload: Mapping[str, Any]) -> None:
        if self._socket is None or self.session_id is None:
            raise RuntimeError("Robot Agent session is unavailable")
        self._socket.send(build_message(
            message_type,
            robot_id=self.robot_id,
            session_id=self.session_id,
            payload=payload,
        ))

    def _receive(
        self,
        expected_type: str,
        *,
        command_id: str | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        if self._socket is None or self.session_id is None:
            raise RuntimeError("Robot Agent session is unavailable")
        while True:
            timeout = 5.0 if deadline is None else deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError(f"timed out waiting for Robot Agent {expected_type}")
            message = parse_message(
                self._socket.recv(timeout=timeout),
                expected_robot_id=self.robot_id,
                expected_session_id=self.session_id,
            )
            if message["type"] == "error":
                raise RuntimeError(str(message["payload"].get("message") or "Robot Agent protocol error"))
            if message["type"] != expected_type:
                continue
            if command_id is not None and message["payload"].get("command_id") != command_id:
                continue
            return message

    def _renew_locked(self) -> Mapping[str, Any]:
        self._lease_sequence += 1
        self._send("lease.renew", {"sequence": self._lease_sequence})
        message = self._receive("lease.status")
        payload = message["payload"]
        require_payload_fields(payload, {"sequence", "expires_at"})
        if payload["sequence"] != self._lease_sequence:
            raise RuntimeError("Robot Agent lease sequence mismatch")
        return {"sequence": self._lease_sequence, "expiresAt": payload["expires_at"]}

    def _start_heartbeat(self) -> None:
        if not self.heartbeat:
            return
        previous = self._heartbeat_thread
        if previous is not None and previous.is_alive():
            if not self._heartbeat_stop.is_set():
                return
            if previous is not threading.current_thread():
                previous.join(timeout=2.5)
            if previous.is_alive():
                raise RuntimeError("previous Robot Agent lease heartbeat did not stop")
        if self._heartbeat_thread is previous:
            self._heartbeat_thread = None
        self._heartbeat_stop.clear()

        def loop() -> None:
            try:
                while not self._heartbeat_stop.wait(0.5):
                    try:
                        with self._io_lock:
                            if self._socket is None:
                                return
                            self._renew_locked()
                    except Exception:  # transport boundary; next dispatch reconnects
                        self._heartbeat_stop.set()
                        return
            finally:
                if self._heartbeat_thread is threading.current_thread():
                    self._heartbeat_thread = None

        thread = threading.Thread(
            target=loop,
            name="robot-agent-gateway-lease",
            daemon=True,
        )
        self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(
        self,
        *,
        timeout: float = 2.5,
        strict: bool = True,
    ) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if strict and thread.is_alive():
                raise RuntimeError("Robot Agent lease heartbeat did not stop")
        if self._heartbeat_thread is thread and (
            thread is None or not thread.is_alive()
        ):
            self._heartbeat_thread = None

    @staticmethod
    def _status(message: Mapping[str, Any]) -> ProtocolCommandStatus:
        payload = message["payload"]
        require_payload_fields(payload, {"command_id", "state", "detail", "progress", "result"})
        return ProtocolCommandStatus(
            command_id=str(payload["command_id"]),
            state=str(payload["state"]),
            detail=str(payload["detail"]),
            progress=None if payload["progress"] is None else float(payload["progress"]),
            result=payload["result"] if isinstance(payload["result"], Mapping) else None,
        )
