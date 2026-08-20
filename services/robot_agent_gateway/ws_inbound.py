"""Thread-safe inbound robot-agent.v1 session owned by the workstation Gateway."""

from __future__ import annotations

import asyncio
from concurrent.futures import (
    CancelledError as FutureCancelledError,
    TimeoutError as FutureTimeoutError,
)
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from starlette.websockets import WebSocketDisconnect

from .capability_handshake import validate_optional_capability_handshake
from .protocol import TERMINAL, ProtocolCommand, ProtocolCommandStatus
from .wire import (
    SUBPROTOCOL,
    build_message,
    parse_control_status_payload,
    parse_message,
    parse_timestamp,
    require_payload_fields,
)


_OBSERVATION_STATE_KEYS = {
    "observation.status",
    "observation.sensor_id",
    "observation.session_id",
    "observation.scene_version",
    "observation.localization_epoch_id",
    "observation.ack_age_ms",
    "observation.capture_age_ms",
    "observation.drop_count",
}

_LOCALIZATION_POSE_STATE_KEYS = {
    "localization.frame_id",
    "localization.stamp.sec",
    "localization.stamp.nanosec",
    "localization.position.x",
    "localization.position.y",
    "localization.position.z",
    "localization.orientation.x",
    "localization.orientation.y",
    "localization.orientation.z",
    "localization.orientation.w",
}

_PROJECTED_STATE_KEYS = _OBSERVATION_STATE_KEYS | _LOCALIZATION_POSE_STATE_KEYS

# A cancellation is not acknowledged by an arbitrary status revision.  The
# Robot Agent must publish a command terminal state and, for CANCELLED, a new
# SAFE_HOLD control status after the request.  Keep the bound private so tests
# can shorten it without widening the service configuration surface.
_CANCEL_CONFIRMATION_TIMEOUT_SECONDS = 5.0

# Normal terminal completion is not permission to retain motion authority.
# The peer sends an explicit ``control.release`` and waits for a *new* measured
# SAFE_HOLD boundary before the terminal command is allowed to escape to the
# Control API.  Keep this private for the same reason as cancellation: tests
# may shorten it, while deployments cannot silently widen the safety budget.
_CONTROL_RELEASE_CONFIRMATION_TIMEOUT_SECONDS = 5.0


@dataclass
class _PendingTerminalPose:
    terminal_state_revision: int
    terminal_state: str
    started_monotonic: float
    last_pose: dict[str, Any] | None
    last_sample_monotonic: float | None
    stable_since_monotonic: float | None = None


def _project_observation_values(
    values: Mapping[str, Any],
    *,
    snapshot: bool,
) -> dict[str, Any]:
    """Normalize Robot Agent state into the Gateway's dotted-path projection.

    State snapshots contain the normalized state tree (for example
    ``{"observation": {"status": "USABLE"}}``), while state updates contain
    dotted leaf paths (for example ``{"observation.status": "USABLE"}``).
    """

    projected = {key: values[key] for key in _PROJECTED_STATE_KEYS if key in values}
    if not snapshot:
        return projected
    for key in _PROJECTED_STATE_KEYS:
        current: Any = values
        for segment in key.split("."):
            if not isinstance(current, Mapping) or segment not in current:
                break
            current = current[segment]
        else:
            projected[key] = current
    return projected


class RobotAgentInboundPeer:
    """Accept one Robot Agent outbound connection and expose the sync peer API.

    FastAPI owns the asynchronous WebSocket receive loop. Control API calls run
    in worker threads, so outbound messages are scheduled onto that loop and
    status updates are published through a condition variable.
    """

    backend_name = "robot_agent_websocket_inbound"

    def __init__(
        self,
        *,
        robot_id: str,
        expected_capabilities_digest: str | None,
        capabilities: tuple[str, ...],
        allow_plaintext_loopback: bool = False,
        execution_backend: str = "real_robot_agent",
        motion: str = "not_validated",
        heartbeat: bool = True,
        allowed_peer_hosts: tuple[str, ...] = (),
        observation_preflight_timeout_seconds: float = 2.0,
        observation_preflight_max_age_ms: float = 400.0,
        terminal_pose_stable_seconds: float = 0.5,
        terminal_pose_max_wait_seconds: float = 4.0,
        terminal_pose_linear_speed_mps: float = 0.02,
        terminal_pose_angular_speed_rps: float = 0.05,
    ) -> None:
        if not robot_id.strip():
            raise ValueError("Robot Agent robot_id is required")
        expected_capabilities_digest = (expected_capabilities_digest or "").strip() or None
        if (
            expected_capabilities_digest is not None
            and not expected_capabilities_digest.startswith("sha256:")
        ):
            raise ValueError("expected Robot Agent capabilities digest is invalid")
        if not allow_plaintext_loopback:
            raise ValueError(
                "inbound plaintext Robot Agent is restricted to explicit loopback development"
            )
        if not math.isfinite(observation_preflight_timeout_seconds) or not (
            0.1 <= observation_preflight_timeout_seconds <= 10.0
        ):
            raise ValueError(
                "observation preflight timeout must be between 0.1 and 10 seconds"
            )
        if not math.isfinite(observation_preflight_max_age_ms) or not (
            50.0 <= observation_preflight_max_age_ms <= 5_000.0
        ):
            raise ValueError(
                "observation preflight maximum age must be between 50 and 5000 ms"
            )
        if not math.isfinite(terminal_pose_stable_seconds) or not (
            0.01 <= terminal_pose_stable_seconds <= 2.0
        ):
            raise ValueError(
                "terminal pose stable time must be between 0.01 and 2 seconds"
            )
        if not math.isfinite(terminal_pose_max_wait_seconds) or not (
            terminal_pose_stable_seconds < terminal_pose_max_wait_seconds <= 10.0
        ):
            raise ValueError(
                "terminal pose maximum wait must exceed stable time and be at most 10 seconds"
            )
        if not math.isfinite(terminal_pose_linear_speed_mps) or not (
            0.001 <= terminal_pose_linear_speed_mps <= 1.0
        ):
            raise ValueError(
                "terminal pose linear-speed threshold must be between 0.001 and 1 m/s"
            )
        if not math.isfinite(terminal_pose_angular_speed_rps) or not (
            0.001 <= terminal_pose_angular_speed_rps <= 5.0
        ):
            raise ValueError(
                "terminal pose angular-speed threshold must be between 0.001 and 5 rad/s"
            )
        self.robot_id = robot_id
        self.expected_capabilities_digest = expected_capabilities_digest
        self._configured_capabilities = tuple(capabilities)
        self.capabilities = tuple(capabilities)
        self.execution_backend = execution_backend
        self.motion = motion
        self.heartbeat = heartbeat
        # Plaintext remains opt-in.  The extra hosts are only for an explicit
        # local container bridge (the Docker userland proxy presents the host
        # client as its bridge gateway instead of 127.0.0.1).
        self._allowed_peer_hosts = frozenset(
            value.strip() for value in allowed_peer_hosts if value.strip()
        )
        self.observation_preflight_timeout_seconds = (
            observation_preflight_timeout_seconds
        )
        self.observation_preflight_max_age_ms = observation_preflight_max_age_ms
        self.terminal_pose_stable_seconds = terminal_pose_stable_seconds
        self.terminal_pose_max_wait_seconds = terminal_pose_max_wait_seconds
        self.terminal_pose_linear_speed_mps = terminal_pose_linear_speed_mps
        self.terminal_pose_angular_speed_rps = terminal_pose_angular_speed_rps

        self.robot_boot_id: str | None = None
        self.capabilities_digest: str | None = None
        self.robot_profile: dict[str, Any] | None = None
        self.capability_manifest: dict[str, Any] | None = None
        self.session_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._websocket: Any | None = None
        self._async_send_lock: asyncio.Lock | None = None
        self._accepting = False
        self._connected = False
        self._control_state = "SAFE_HOLD"
        self._control_reason = "Robot Agent is not connected"
        self._control_revision = 0
        self._control_authority_revoked: bool | None = None
        self._control_safe_stop_state: str | None = None
        self._control_safe_stop_epoch: int | None = None
        self._control_safe_stop_truth_supported = False
        # A release retry must compare against the epoch that preceded the
        # *first* release request. Robot Agent deliberately keeps the same
        # epoch for an idempotent retry after SAFE_HOLD is already confirmed.
        self._release_baseline_revision: int | None = None
        self._release_baseline_epoch: int | None = None
        self._release_baseline_boot_id: str | None = None
        self._lease_sequence = -1
        self._lease_acks: dict[int, str] = {}
        self._commands: dict[str, ProtocolCommand] = {}
        self._statuses: dict[str, ProtocolCommandStatus] = {}
        self._status_revisions: dict[str, int] = {}
        self._terminal_pose_pending: dict[str, _PendingTerminalPose] = {}
        self._last_error: str | None = None
        self._projected_state: dict[str, Any] = {}
        self._state_sequence = -1
        self._state_revision = 0
        self._state_received_monotonic: float | None = None
        # These are set only when a localization field actually changes in a
        # received state envelope.  The wire timestamp is Robot Agent's UTC
        # clock; ROS header stamps may belong to a simulated clock and are
        # provenance only.
        self._localization_source_observed_at: datetime | None = None
        self._state_lock = threading.RLock()
        self._condition = threading.Condition(self._state_lock)
        self._lease_lock = threading.Lock()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    @staticmethod
    async def _close(websocket: Any, *, code: int, reason: str) -> None:
        """Close either a pre-accept ASGI socket or an accepted WebSocket."""

        try:
            await websocket.close(code=code, reason=reason)
        except RuntimeError:
            # Starlette rejects ``WebSocket.close`` before accept on some
            # versions. Deny the handshake explicitly in that case.
            await websocket.send({"type": "websocket.close", "code": code, "reason": reason})

    async def handle_connection(self, websocket: Any) -> None:
        """Run one inbound Robot Agent connection until it disconnects."""

        offered = {
            item.strip()
            for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
            if item.strip()
        }
        if SUBPROTOCOL not in offered:
            await self._close(
                websocket,
                code=1002,
                reason="robot-agent.v1 subprotocol required",
            )
            return
        client = getattr(websocket, "client", None)
        host = None if client is None else client.host
        allowed_hosts = {"127.0.0.1", "::1", "localhost", "testclient"}
        allowed_hosts.update(self._allowed_peer_hosts)
        if host not in allowed_hosts:
            await self._close(
                websocket,
                code=1008,
                reason="plaintext Robot Agent is restricted to loopback development",
            )
            return
        with self._condition:
            if self._connected or self._accepting:
                busy = True
            else:
                busy = False
                self._accepting = True
        if busy:
            await self._close(
                websocket,
                code=1013,
                reason="a Robot Agent session is already active",
            )
            return

        accepted = False
        connection_session: str | None = None
        try:
            await websocket.accept(subprotocol=SUBPROTOCOL)
            accepted = True
            raw_hello = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            hello = parse_message(
                raw_hello,
                expected_robot_id=self.robot_id,
                expected_session_id=None,
            )
            if hello["type"] != "session.hello":
                raise ValueError("Robot Agent must initiate with session.hello")
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
            supported = payload["supported_versions"]
            if not isinstance(supported, list) or "1.0" not in supported:
                raise ValueError("Robot Agent does not support protocol version 1.0")
            if payload["transport_mode"] != "client":
                raise ValueError("inbound endpoint requires Robot Agent client transport mode")
            digest = str(payload["capabilities_digest"])
            capability_handshake = validate_optional_capability_handshake(
                payload,
                expected_robot_id=self.robot_id,
            )
            if (
                self.expected_capabilities_digest is not None
                and digest != self.expected_capabilities_digest
            ):
                raise ValueError(
                    "Robot Agent capabilities digest mismatch "
                    f"(received={digest}, expected={self.expected_capabilities_digest})"
                )
            if capability_handshake is None:
                if self.expected_capabilities_digest is None:
                    raise ValueError(
                        "legacy Robot Agent requires an expected capabilities digest"
                    )
                if not self._configured_capabilities:
                    raise ValueError(
                        "legacy Robot Agent capabilities must be configured explicitly"
                    )
                negotiated_capabilities = self._configured_capabilities
                robot_profile = None
                capability_manifest = None
            else:
                negotiated_capabilities = capability_handshake.capability_ids
                robot_profile = capability_handshake.robot_profile.model_dump(mode="json")
                capability_manifest = capability_handshake.capability_manifest.model_dump(
                    mode="json"
                )
            boot_id = str(UUID(str(payload["boot_id"])))
            connection_session = str(uuid4())
            await websocket.send_text(
                build_message(
                    "session.welcome",
                    robot_id=self.robot_id,
                    session_id=connection_session,
                    payload={
                        "accepted_version": "1.0",
                        "lease_interval_ms": 500,
                        "lease_timeout_ms": 2000,
                        "control_state": "SAFE_HOLD",
                    },
                )
            )
            with self._condition:
                self._loop = asyncio.get_running_loop()
                self._websocket = websocket
                self._async_send_lock = asyncio.Lock()
                self._connected = True
                self.session_id = connection_session
                self.robot_boot_id = boot_id
                self.capabilities_digest = digest
                self.capabilities = negotiated_capabilities
                self.robot_profile = robot_profile
                self.capability_manifest = capability_manifest
                self._control_state = "SAFE_HOLD"
                self._control_reason = "new session requires explicit control.acquire"
                self._control_authority_revoked = None
                self._control_safe_stop_state = None
                self._control_safe_stop_epoch = None
                self._control_safe_stop_truth_supported = False
                self._last_error = None
                self._projected_state = {}
                self._state_sequence = -1
                self._state_revision += 1
                self._state_received_monotonic = None
                self._localization_source_observed_at = None
                self._condition.notify_all()

            while True:
                raw = await websocket.receive_text()
                message = parse_message(
                    raw,
                    expected_robot_id=self.robot_id,
                    expected_session_id=connection_session,
                )
                self._record_message(message)
        except WebSocketDisconnect:
            pass
        except asyncio.CancelledError:
            raise
        except (ValueError, RuntimeError, TimeoutError) as exc:
            with self._condition:
                self._last_error = str(exc) or type(exc).__name__
                self._condition.notify_all()
            if accepted:
                try:
                    await websocket.close(code=1002, reason="Robot Agent protocol error")
                except (RuntimeError, WebSocketDisconnect):
                    self._last_error = self._last_error or "failed to close invalid Robot Agent session"
        finally:
            with self._condition:
                self._accepting = False
                if connection_session is not None and self.session_id == connection_session:
                    self._mark_disconnected_locked("Robot Agent WebSocket disconnected")
                self._condition.notify_all()

    def discover(self, robot_id: str | None = None) -> Mapping[str, Any]:
        if robot_id not in {None, self.robot_id}:
            return {"available": False, "reason": "robot identity mismatch"}
        with self._condition:
            if not self._connected:
                return {
                    "available": False,
                    "reason": self._last_error or "waiting for Robot Agent inbound connection",
                }
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
                "transportMode": "agent_client",
                "observation": self._observation_snapshot_locked(),
                **(
                    {"currentPose": current_pose}
                    if (current_pose := self._current_pose_snapshot_locked()) is not None
                    else {}
                ),
            }

    def acquire(self, robot_id: str) -> Mapping[str, Any]:
        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        with self._condition:
            if self._control_state == "ACTIVE" and self._connected:
                return {"state": "ACTIVE", "leaseSequence": max(0, self._lease_sequence)}
            revision = self._control_revision
        self._send("control.acquire", {"operator_id": "h-ide-gateway"})
        deadline = time.monotonic() + 5.0
        with self._condition:
            while self._connected and (
                self._control_revision <= revision or self._control_state != "ACTIVE"
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for Robot Agent control.status")
                self._condition.wait(remaining)
            if not self._connected:
                raise RuntimeError("Robot Agent disconnected while acquiring control")
            if self._control_state != "ACTIVE":
                raise RuntimeError(f"Robot Agent control is not active: {self._control_reason}")
        self._start_heartbeat()
        return {"state": "ACTIVE", "leaseSequence": max(0, self._lease_sequence)}

    def renew(self, robot_id: str, sequence: int) -> Mapping[str, Any]:
        del sequence
        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        with self._lease_lock:
            with self._condition:
                if not self._connected or self._control_state != "ACTIVE":
                    raise RuntimeError("Robot Agent control lease is not active")
                self._lease_sequence += 1
                current = self._lease_sequence
            self._send("lease.renew", {"sequence": current})
            deadline = time.monotonic() + 2.0
            with self._condition:
                while self._connected and current not in self._lease_acks:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for Robot Agent lease.status")
                    self._condition.wait(remaining)
                if not self._connected:
                    raise RuntimeError("Robot Agent disconnected while renewing lease")
                expires_at = self._lease_acks.pop(current)
            return {"sequence": current, "expiresAt": expires_at}

    def release(self, robot_id: str, reason: str) -> Mapping[str, Any]:
        """Relinquish control and prove a fresh physical SAFE_HOLD boundary.

        Stopping the heartbeat first prevents a concurrent renewal from
        restoring an ACTIVE lease after ``control.release``.  Legacy peers do
        not expose enough physical truth to acknowledge a normal release, so
        they fail closed while the stopped heartbeat still lets their lease
        watchdog remove authority.
        """

        if robot_id != self.robot_id:
            raise ValueError("robot identity mismatch")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("control release reason is required")
        detail = reason.strip()
        if len(detail) > 256:
            raise ValueError("control release reason is too long")
        self._stop_heartbeat()
        with self._condition:
            if not self._connected:
                raise RuntimeError("Robot Agent disconnected before control release")
            if self._release_baseline_epoch is None:
                if (
                    not self._control_safe_stop_truth_supported
                    or self._control_safe_stop_epoch is None
                ):
                    raise RuntimeError(
                        "Robot Agent control release is unconfirmed: peer does not "
                        "publish atomic safe-stop truth"
                    )
                self._release_baseline_revision = self._control_revision
                self._release_baseline_epoch = self._control_safe_stop_epoch
                self._release_baseline_boot_id = self.robot_boot_id
            baseline_revision = self._release_baseline_revision
            baseline_epoch = self._release_baseline_epoch
            baseline_boot_id = self._release_baseline_boot_id
            if self.robot_boot_id != baseline_boot_id:
                raise RuntimeError(
                    "Robot Agent restarted before control release was confirmed"
                )
        assert baseline_revision is not None
        assert baseline_epoch is not None
        self._send("control.release", {"reason": detail})
        deadline = time.monotonic() + _CONTROL_RELEASE_CONFIRMATION_TIMEOUT_SECONDS
        with self._condition:
            while True:
                if not self._connected:
                    raise RuntimeError(
                        "Robot Agent disconnected before confirmed control release"
                    )
                if self.robot_boot_id != baseline_boot_id:
                    raise RuntimeError(
                        "Robot Agent restarted before control release was confirmed"
                    )
                epoch = self._control_safe_stop_epoch
                fresh = (
                    self._control_revision > baseline_revision
                    and epoch is not None
                    and epoch > baseline_epoch
                )
                if (
                    fresh
                    and self._control_state == "SAFE_HOLD"
                    and self._control_authority_revoked is True
                    and self._control_safe_stop_state == "confirmed"
                ):
                    result = {
                        "state": "SAFE_HOLD",
                        "authorityRevoked": True,
                        "safeStopState": "confirmed",
                        "safeStopEpoch": epoch,
                    }
                    self._release_baseline_revision = None
                    self._release_baseline_epoch = None
                    self._release_baseline_boot_id = None
                    return result
                if (
                    fresh
                    and self._control_state == "SAFE_HOLD"
                    and self._control_authority_revoked is True
                    and self._control_safe_stop_state == "failed"
                ):
                    raise RuntimeError(
                        "Robot Agent physical safe-stop failed during control release"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for a new confirmed SAFE_HOLD control release"
                    )
                self._condition.wait(remaining)

    def submit(self, command: ProtocolCommand) -> ProtocolCommandStatus:
        if command.robot_id != self.robot_id:
            return ProtocolCommandStatus(command.command_id, "REJECTED", "robot identity mismatch")
        try:
            command_id = str(UUID(command.command_id))
        except (TypeError, ValueError):
            return ProtocolCommandStatus(command.command_id, "REJECTED", "command ID must be a UUID")
        with self._condition:
            existing = self._commands.get(command_id)
            if existing is not None and existing.payload_hash != command.payload_hash:
                return ProtocolCommandStatus(command_id, "REJECTED", "command ID payload conflict")
            current = self._statuses.get(command_id)
            # The command ID is the idempotency key. Once accepted, repeated
            # dispatch attempts return the current ledger entry without sending a
            # second command.request to the robot.
            if existing is not None and current is not None:
                return current
        preflight_error = self._wait_for_observation_preflight(command)
        if preflight_error is not None:
            rejected = ProtocolCommandStatus(
                command_id,
                "REJECTED",
                f"workstation observation preflight failed: {preflight_error}",
            )
            with self._condition:
                existing = self._commands.get(command_id)
                current = self._statuses.get(command_id)
                if existing is not None and current is not None:
                    return current
                self._commands[command_id] = command
                self._statuses[command_id] = rejected
                self._status_revisions[command_id] = (
                    self._status_revisions.get(command_id, 0) + 1
                )
                self._condition.notify_all()
            return rejected
        with self._condition:
            existing = self._commands.get(command_id)
            current = self._statuses.get(command_id)
            if existing is not None and current is not None:
                return current
            self._commands[command_id] = command
            sent = ProtocolCommandStatus(
                command_id,
                "RECEIVED",
                "command sent; awaiting Robot Agent status",
            )
            self._statuses[command_id] = sent
            self._status_revisions[command_id] = self._status_revisions.get(command_id, 0) + 1
        try:
            self._send(
                "command.request",
                {
                    "command_id": command_id,
                    "kind": command.kind,
                    "issued_at": command.issued_at.astimezone(UTC)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "ttl_ms": command.ttl_ms,
                    "params": dict(command.params),
                },
            )
        except (ValueError, RuntimeError, TimeoutError, FutureTimeoutError) as exc:
            failed = ProtocolCommandStatus(command_id, "ABORTED_AGENT_RESTART", str(exc))
            with self._condition:
                self._statuses[command_id] = failed
                self._status_revisions[command_id] += 1
                self._condition.notify_all()
            return failed
        return sent

    def query(self, command_id: str) -> ProtocolCommandStatus:
        with self._condition:
            status = self._statuses.get(command_id)
            if status is not None:
                pending = self._terminal_pose_pending.get(command_id)
                if pending is not None and status.state == "SUCCEEDED":
                    elapsed = time.monotonic() - pending.started_monotonic
                    if elapsed < self.terminal_pose_max_wait_seconds:
                        return ProtocolCommandStatus(
                            command_id=status.command_id,
                            state="RUNNING",
                            detail="Nav2 terminal received; final pose is settling",
                            progress=status.progress,
                            result=status.result,
                        )
                    result = dict(status.result or {})
                    result["finalPoseRelation"] = "settlement_timeout"
                    result["terminalSettleDurationMs"] = round(
                        elapsed * 1000.0, 3
                    )
                    status = ProtocolCommandStatus(
                        command_id=status.command_id,
                        state=status.state,
                        detail=status.detail,
                        progress=status.progress,
                        result=result,
                    )
                    self._statuses[command_id] = status
                    self._terminal_pose_pending.pop(command_id, None)
                    self._status_revisions[command_id] = (
                        self._status_revisions.get(command_id, 0) + 1
                    )
                return status
        return ProtocolCommandStatus(command_id, "REJECTED", "command not found in gateway ledger")

    def cancel(self, command_id: str) -> ProtocolCommandStatus:
        with self._condition:
            current = self._statuses.get(command_id)
            if current is None:
                return ProtocolCommandStatus(command_id, "REJECTED", "command not found")
            if current.state in TERMINAL and current.state != "CANCELLED":
                return current
            try:
                canonical_command_id = str(UUID(command_id))
            except (TypeError, ValueError):
                return ProtocolCommandStatus(
                    command_id,
                    "REJECTED",
                    "command ID must be a UUID",
                )
            control_revision = self._control_revision
            safe_stop_epoch = (
                self._control_safe_stop_epoch
                if self._control_safe_stop_truth_supported
                else None
            )
            cancel_session_id = self.session_id
            cancel_boot_id = self.robot_boot_id
            # Fence future renewals without delaying the safety-critical
            # cancellation send on an in-flight heartbeat join.
            self._heartbeat_stop.set()
        deadline = time.monotonic() + _CANCEL_CONFIRMATION_TIMEOUT_SECONDS
        try:
            self._send(
                "command.cancel",
                {
                    "command_id": canonical_command_id,
                    "reason": "H-IDE mission or operator cancellation",
                },
            )
        except (
            ValueError,
            RuntimeError,
            TimeoutError,
            FutureCancelledError,
            FutureTimeoutError,
            WebSocketDisconnect,
        ) as exc:
            try:
                self._stop_heartbeat(
                    timeout=max(0.0, deadline - time.monotonic()),
                    strict=False,
                )
            except (
                ValueError,
                RuntimeError,
                TimeoutError,
                FutureCancelledError,
                FutureTimeoutError,
                WebSocketDisconnect,
            ):
                pass
            return ProtocolCommandStatus(
                command_id,
                "ABORTED_AGENT_RESTART",
                f"cancellation outcome is unknown: command.cancel was not confirmed: {exc}",
            )
        try:
            self._stop_heartbeat(timeout=max(0.0, deadline - time.monotonic()))
        except (ValueError, RuntimeError, TimeoutError, FutureTimeoutError) as exc:
            return ProtocolCommandStatus(
                command_id,
                "ABORTED_AGENT_RESTART",
                "cancellation outcome is unknown: command.cancel was dispatched "
                f"but the lease heartbeat did not stop: {exc}",
            )
        with self._condition:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = self._statuses.get(command_id, current)
                    if status.state == "CANCELLED":
                        detail = (
                            "cancellation outcome is unknown: CANCELLED was received "
                            "without a new confirmed SAFE_HOLD safety truth"
                        )
                    else:
                        detail = (
                            "cancellation outcome is unknown: timed out waiting for "
                            "a terminal command status"
                        )
                    return ProtocolCommandStatus(
                        command_id,
                        "ABORTED_AGENT_RESTART",
                        detail,
                    )
                status = self._statuses.get(command_id, current)
                if not self._connected:
                    if status.state == "ABORTED_AGENT_RESTART":
                        return status
                    return ProtocolCommandStatus(
                        command_id,
                        "ABORTED_AGENT_RESTART",
                        "cancellation outcome is unknown: Robot Agent disconnected "
                        "before SAFE_HOLD confirmation",
                    )
                if (
                    self.session_id != cancel_session_id
                    or self.robot_boot_id != cancel_boot_id
                ):
                    return ProtocolCommandStatus(
                        command_id,
                        "ABORTED_AGENT_RESTART",
                        "cancellation outcome is unknown: Robot Agent session identity "
                        "changed before SAFE_HOLD confirmation",
                    )
                if status.state in TERMINAL:
                    if status.state != "CANCELLED":
                        return status
                    if (
                        self._control_revision > control_revision
                        and self._control_state == "SAFE_HOLD"
                        and safe_stop_epoch is not None
                        and self._control_safe_stop_truth_supported
                        and self._control_authority_revoked is True
                        and self._control_safe_stop_state == "confirmed"
                        and self._control_safe_stop_epoch is not None
                        and self._control_safe_stop_epoch > safe_stop_epoch
                    ):
                        return status
                    if (
                        self._control_revision > control_revision
                        and self._control_state == "SAFE_HOLD"
                        and safe_stop_epoch is not None
                        and self._control_safe_stop_truth_supported
                        and self._control_authority_revoked is True
                        and self._control_safe_stop_state == "failed"
                        and self._control_safe_stop_epoch is not None
                        and self._control_safe_stop_epoch > safe_stop_epoch
                    ):
                        return ProtocolCommandStatus(
                            command_id,
                            "ABORTED_AGENT_RESTART",
                            "cancellation outcome is unknown: Robot Agent physical "
                            "safe-stop confirmation failed",
                        )
                self._condition.wait(remaining)

    def connection_snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "connected": self._connected,
                "robotId": self.robot_id,
                "robotBootId": self.robot_boot_id,
                "controlState": self._control_state,
                "authorityRevoked": self._control_authority_revoked,
                "safeStopState": self._control_safe_stop_state,
                "safeStopEpoch": self._control_safe_stop_epoch,
                "lastError": self._last_error,
                "observation": self._observation_snapshot_locked(),
            }

    def close(self) -> None:
        self._stop_heartbeat(timeout=1.0, strict=False)
        with self._condition:
            connected = self._connected
        if connected:
            try:
                self._send("session.goodbye", {"reason": "H-IDE Gateway shutdown"})
            except (ValueError, RuntimeError, TimeoutError, FutureTimeoutError) as exc:
                with self._condition:
                    self._last_error = f"Robot Agent goodbye failed: {exc}"
            self._close_socket()

    def _record_message(self, message: Mapping[str, Any]) -> None:
        message_type = message["type"]
        payload = message["payload"]
        with self._condition:
            if message_type == "control.status":
                self._record_control_status_locked(payload)
            elif message_type == "lease.status":
                require_payload_fields(payload, {"sequence", "expires_at"})
                self._lease_acks[int(payload["sequence"])] = str(payload["expires_at"])
            elif message_type == "command.status":
                status = self._status(message)
                command = self._commands.get(status.command_id)
                if (
                    status.state in TERMINAL
                    and command is not None
                    and command.kind == "navigation.navigate_to_pose"
                ):
                    # Begin a bounded terminal-evidence handoff. A later
                    # H-IDE/browser reconciliation must not use an arbitrary
                    # post-stop or post-restart pose and call it the Nav2
                    # arrival point.
                    result = dict(status.result or {})
                    result.setdefault(
                        "terminalReceivedAt",
                        datetime.now(UTC).isoformat(),
                    )
                    # A Robot Agent supplied finalPose is authoritative. When
                    # it is absent, retain the latest pre-terminal pose only as
                    # a bounded fallback, then replace it with the first state
                    # update received after this command.status envelope.
                    if "finalPose" not in result:
                        terminal_pose = self._current_pose_snapshot_locked()
                        if terminal_pose is not None:
                            result["finalPose"] = terminal_pose
                            result["finalPoseRelation"] = "latest_before_terminal"
                        else:
                            result["finalPoseRelation"] = "awaiting_post_terminal"
                        now = time.monotonic()
                        self._terminal_pose_pending[status.command_id] = (
                            _PendingTerminalPose(
                                terminal_state_revision=self._state_revision,
                                terminal_state=status.state,
                                started_monotonic=now,
                                last_pose=terminal_pose,
                                last_sample_monotonic=(
                                    now if terminal_pose is not None else None
                                ),
                            )
                        )
                    status = ProtocolCommandStatus(
                        command_id=status.command_id,
                        state=status.state,
                        detail=status.detail,
                        progress=status.progress,
                        result=result,
                    )
                self._statuses[status.command_id] = status
                self._status_revisions[status.command_id] = (
                    self._status_revisions.get(status.command_id, 0) + 1
                )
            elif message_type in {"state.snapshot", "state.update"}:
                self._record_state_locked(message_type, payload)
            elif message_type == "error":
                self._last_error = str(payload.get("message") or "Robot Agent protocol error")
            # Safety and telemetry envelopes remain accepted without being
            # projected into the command path.
            self._condition.notify_all()

    def _record_control_status_locked(self, payload: Mapping[str, Any]) -> None:
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
        self._control_state = status.state
        self._control_reason = status.reason
        self._control_authority_revoked = status.authority_revoked
        self._control_safe_stop_state = status.safe_stop_state
        self._control_safe_stop_epoch = status.safe_stop_epoch
        self._control_revision += 1

    def _record_state_locked(
        self,
        message_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        value_key = "state" if message_type == "state.snapshot" else "changes"
        require_payload_fields(payload, {"sequence", "observed_at", value_key})
        sequence = payload["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Robot Agent state sequence is invalid")
        if sequence <= self._state_sequence:
            raise ValueError("Robot Agent state sequence did not increase")
        parse_timestamp(payload["observed_at"])
        values = payload[value_key]
        if not isinstance(values, Mapping) or not all(
            isinstance(key, str) for key in values
        ):
            raise ValueError("Robot Agent state values are invalid")
        previous_localization = {
            key: self._projected_state.get(key)
            for key in _LOCALIZATION_POSE_STATE_KEYS
        }
        projected = _project_observation_values(
            values,
            snapshot=message_type == "state.snapshot",
        )
        if message_type == "state.snapshot":
            self._projected_state = projected
        else:
            self._projected_state.update(projected)
        self._state_sequence = sequence
        self._state_revision += 1
        self._state_received_monotonic = time.monotonic()
        localization_changed = any(
            self._projected_state.get(key) != previous_localization.get(key)
            for key in _LOCALIZATION_POSE_STATE_KEYS
        )
        if localization_changed:
            self._localization_source_observed_at = parse_timestamp(
                payload["observed_at"]
            )
            terminal_pose = self._current_pose_snapshot_locked()
            if terminal_pose is not None:
                now = time.monotonic()
                for command_id, pending in list(
                    self._terminal_pose_pending.items()
                ):
                    if self._state_revision <= pending.terminal_state_revision:
                        continue
                    status = self._statuses.get(command_id)
                    if status is None or status.state not in TERMINAL:
                        self._terminal_pose_pending.pop(command_id, None)
                        continue
                    result = dict(status.result or {})
                    result["finalPose"] = terminal_pose

                    # Failures and cancellations must reach the operator
                    # promptly. Successful navigation is different: a walking
                    # body can still coast for a short time after Nav2 emits
                    # its terminal status, so wait for a bounded run of stable
                    # localization samples before calling the pose final.
                    finished = pending.terminal_state != "SUCCEEDED"
                    linear_speed = 0.0
                    angular_speed = 0.0
                    if (
                        pending.last_pose is not None
                        and pending.last_sample_monotonic is not None
                    ):
                        dt = max(1e-6, now - pending.last_sample_monotonic)
                        linear_speed = math.hypot(
                            float(terminal_pose["x"]) - float(pending.last_pose["x"]),
                            float(terminal_pose["y"]) - float(pending.last_pose["y"]),
                        ) / dt
                        yaw_delta = float(terminal_pose["yaw"]) - float(
                            pending.last_pose["yaw"]
                        )
                        angular_speed = abs(
                            math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))
                        ) / dt
                    stable = (
                        linear_speed <= self.terminal_pose_linear_speed_mps
                        and angular_speed <= self.terminal_pose_angular_speed_rps
                    )
                    if stable:
                        if pending.stable_since_monotonic is None:
                            pending.stable_since_monotonic = now
                    else:
                        pending.stable_since_monotonic = None
                    stable_for = (
                        0.0
                        if pending.stable_since_monotonic is None
                        else now - pending.stable_since_monotonic
                    )
                    elapsed = now - pending.started_monotonic
                    result["terminalLinearSpeedMps"] = linear_speed
                    result["terminalAngularSpeedRps"] = angular_speed
                    result["terminalSettleDurationMs"] = round(elapsed * 1000.0, 3)
                    if finished:
                        result["finalPoseRelation"] = "first_at_or_after_terminal"
                    elif stable_for >= self.terminal_pose_stable_seconds:
                        result["finalPoseRelation"] = "stable_after_terminal"
                        finished = True
                    elif elapsed >= self.terminal_pose_max_wait_seconds:
                        result["finalPoseRelation"] = "settlement_timeout"
                        finished = True
                    else:
                        result["finalPoseRelation"] = "settling_after_terminal"
                    self._statuses[command_id] = ProtocolCommandStatus(
                        command_id=status.command_id,
                        state=status.state,
                        detail=status.detail,
                        progress=status.progress,
                        result=result,
                    )
                    self._status_revisions[command_id] = (
                        self._status_revisions.get(command_id, 0) + 1
                    )
                    if finished:
                        self._terminal_pose_pending.pop(command_id, None)
                    else:
                        pending.last_pose = terminal_pose
                        pending.last_sample_monotonic = now

    def _wait_for_observation_preflight(
        self,
        command: ProtocolCommand,
    ) -> str | None:
        if command.kind != "navigation.navigate_to_pose":
            return None
        session_id = command.params.get("observationSessionId")
        localization_epoch_id = command.params.get("localizationEpochId")
        if not isinstance(session_id, str) or not session_id:
            return "command observationSessionId is missing"
        if not isinstance(localization_epoch_id, str) or not localization_epoch_id:
            return "command localizationEpochId is missing"
        deadline = time.monotonic() + self.observation_preflight_timeout_seconds
        with self._condition:
            baseline_revision = self._state_revision
            reason = self._observation_preflight_reason_locked(
                session_id=session_id,
                localization_epoch_id=localization_epoch_id,
            )
            if reason is None:
                return None
            while self._connected:
                if self._state_revision > baseline_revision:
                    reason = self._observation_preflight_reason_locked(
                        session_id=session_id,
                        localization_epoch_id=localization_epoch_id,
                    )
                    if reason is None:
                        return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return reason
                self._condition.wait(remaining)
            return "Robot Agent disconnected before observation became ready"

    def _observation_preflight_reason_locked(
        self,
        *,
        session_id: str,
        localization_epoch_id: str,
    ) -> str | None:
        state = self._projected_state
        if self._state_received_monotonic is None:
            return "Robot Agent observation state has not been received"
        received_age_ms = (
            time.monotonic() - self._state_received_monotonic
        ) * 1000.0
        if received_age_ms < 0 or received_age_ms > self.observation_preflight_max_age_ms:
            return "Robot Agent observation state is too old"
        if state.get("observation.status") != "USABLE":
            return "Robot Agent observation status is not USABLE"
        if state.get("observation.session_id") != session_id:
            return "Robot Agent observation session does not match the visual resolution"
        if state.get("observation.localization_epoch_id") != localization_epoch_id:
            return "Robot Agent localization epoch does not match the visual resolution"
        for key, label in (
            ("observation.ack_age_ms", "ACK"),
            ("observation.capture_age_ms", "capture"),
        ):
            raw_age = state.get(key)
            if isinstance(raw_age, bool) or not isinstance(raw_age, (int, float)):
                return f"Robot Agent observation {label} age is unavailable"
            age_ms = float(raw_age)
            if not math.isfinite(age_ms) or age_ms < 0:
                return f"Robot Agent observation {label} age is invalid"
            # The Robot Agent reports the age at the instant it builds the
            # state envelope. Account for the time spent receiving/queueing
            # that envelope before allowing a command to leave the workstation.
            effective_age_ms = age_ms + received_age_ms
            if effective_age_ms > self.observation_preflight_max_age_ms:
                return f"Robot Agent observation {label} is too old"
        return None

    def _observation_snapshot_locked(self) -> dict[str, Any]:
        received_age_ms = (
            None
            if self._state_received_monotonic is None
            else max(0.0, (time.monotonic() - self._state_received_monotonic) * 1000)
        )
        return {
            "sequence": None if self._state_sequence < 0 else self._state_sequence,
            "receivedAgeMs": received_age_ms,
            "status": self._projected_state.get("observation.status"),
            "sensorId": self._projected_state.get("observation.sensor_id"),
            "sessionId": self._projected_state.get("observation.session_id"),
            "localizationEpochId": self._projected_state.get(
                "observation.localization_epoch_id"
            ),
            "ackAgeMs": self._projected_state.get("observation.ack_age_ms"),
            "captureAgeMs": self._projected_state.get(
                "observation.capture_age_ms"
            ),
        }

    def _current_pose_snapshot_locked(self) -> dict[str, Any] | None:
        state = self._projected_state
        if state.get("localization.frame_id") != "map":
            return None
        keys = (
            "localization.position.x",
            "localization.position.y",
            "localization.position.z",
            "localization.orientation.x",
            "localization.orientation.y",
            "localization.orientation.z",
            "localization.orientation.w",
        )
        raw = [state.get(key) for key in keys]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw
        ):
            return None
        x, y, z, qx, qy, qz, qw = (float(value) for value in raw)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-6:
            return None
        qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
        stamp_sec = state.get("localization.stamp.sec")
        stamp_nanosec = state.get("localization.stamp.nanosec")
        if (
            isinstance(stamp_sec, bool)
            or not isinstance(stamp_sec, int)
            or stamp_sec < 0
            or isinstance(stamp_nanosec, bool)
            or not isinstance(stamp_nanosec, int)
            or not 0 <= stamp_nanosec < 1_000_000_000
        ):
            return None
        # The ROS stamp is intentionally parsed only as a structural sanity
        # check.  Its clock domain may be ROS simulation time and therefore is
        # not suitable for comparing against H-IDE's UTC wall clock.
        try:
            datetime.fromtimestamp(
                stamp_sec + stamp_nanosec / 1_000_000_000,
                tz=UTC,
            )
        except (OverflowError, OSError, ValueError):
            return None
        observed_at = self._localization_source_observed_at
        if observed_at is None:
            return None
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        return {
            "frameId": "map",
            "x": x,
            "y": y,
            "z": z,
            "yaw": yaw,
            "observedAt": observed_at.isoformat(),
        }

    def _send(self, message_type: str, payload: Mapping[str, Any]) -> None:
        with self._condition:
            loop = self._loop
            websocket = self._websocket
            send_lock = self._async_send_lock
            session_id = self.session_id
            if not self._connected or loop is None or websocket is None or send_lock is None:
                raise RuntimeError("Robot Agent inbound session is unavailable")
        encoded = build_message(
            message_type,
            robot_id=self.robot_id,
            session_id=session_id,
            payload=payload,
        )

        async def send_serialized() -> None:
            async with send_lock:
                await websocket.send_text(encoded)

        future = asyncio.run_coroutine_threadsafe(send_serialized(), loop)
        future.result(timeout=5.0)

    def _close_socket(self) -> None:
        with self._condition:
            loop = self._loop
            websocket = self._websocket
        if loop is None or websocket is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            websocket.close(code=1000, reason="H-IDE Gateway shutdown"), loop
        )
        try:
            future.result(timeout=2.0)
        except (RuntimeError, TimeoutError, FutureTimeoutError) as exc:
            with self._condition:
                self._last_error = f"Robot Agent socket close failed: {exc}"

    def _mark_disconnected_locked(self, reason: str) -> None:
        self._connected = False
        self._control_state = "SAFE_HOLD"
        self._control_reason = reason
        self._control_authority_revoked = None
        self._control_safe_stop_state = None
        self._control_safe_stop_epoch = None
        self._control_safe_stop_truth_supported = False
        self._loop = None
        self._websocket = None
        self._async_send_lock = None
        self.session_id = None
        self._projected_state = {}
        self._state_sequence = -1
        self._state_revision += 1
        self._state_received_monotonic = None
        self._localization_source_observed_at = None
        self._terminal_pose_pending.clear()
        self._heartbeat_stop.set()
        for command_id, status in list(self._statuses.items()):
            if status.state not in TERMINAL:
                self._statuses[command_id] = ProtocolCommandStatus(
                    command_id,
                    "ABORTED_AGENT_RESTART",
                    reason,
                )
                self._status_revisions[command_id] = self._status_revisions.get(command_id, 0) + 1

    def _start_heartbeat(self) -> None:
        if not self.heartbeat:
            return
        with self._condition:
            previous = self._heartbeat_thread
        if previous is not None and previous.is_alive():
            if not self._heartbeat_stop.is_set():
                return
            if previous is not threading.current_thread():
                previous.join(timeout=2.5)
            if previous.is_alive():
                raise RuntimeError("previous Robot Agent lease heartbeat did not stop")
        with self._condition:
            if self._heartbeat_thread is previous:
                self._heartbeat_thread = None
        self._heartbeat_stop.clear()

        def loop() -> None:
            try:
                while not self._heartbeat_stop.wait(0.5):
                    try:
                        self.renew(self.robot_id, 0)
                    except (
                        ValueError,
                        RuntimeError,
                        TimeoutError,
                        FutureTimeoutError,
                    ) as exc:
                        with self._condition:
                            if self._heartbeat_stop.is_set():
                                return
                            self._last_error = f"Robot Agent lease renewal failed: {exc}"
                            self._condition.notify_all()
                        self._heartbeat_stop.set()
                        return
            finally:
                with self._condition:
                    if self._heartbeat_thread is threading.current_thread():
                        self._heartbeat_thread = None

        thread = threading.Thread(
            target=loop,
            name="robot-agent-inbound-lease",
            daemon=True,
        )
        with self._condition:
            self._heartbeat_thread = thread
        thread.start()

    def _stop_heartbeat(
        self,
        *,
        timeout: float = 2.5,
        strict: bool = True,
    ) -> None:
        """Stop and join the lease-renewal owner before releasing authority."""

        with self._condition:
            self._heartbeat_stop.set()
            thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if strict and thread.is_alive():
                raise RuntimeError("Robot Agent lease heartbeat did not stop")
        with self._condition:
            if self._heartbeat_thread is thread and (
                thread is None or not thread.is_alive()
            ):
                self._heartbeat_thread = None

    @staticmethod
    def _status(message: Mapping[str, Any]) -> ProtocolCommandStatus:
        payload = message["payload"]
        require_payload_fields(payload, {"command_id", "state", "detail", "progress", "result"})
        result = payload["result"]
        return ProtocolCommandStatus(
            command_id=str(payload["command_id"]),
            state=str(payload["state"]),
            detail=str(payload["detail"]),
            progress=None if payload["progress"] is None else float(payload["progress"]),
            result=result if isinstance(result, Mapping) else None,
        )
