from __future__ import annotations

from collections.abc import Mapping
from datetime import timezone
import hashlib
import json
import threading
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from packages.control_core.adapters import AdapterResult
from packages.control_core.models import CommandRequest, MappingVersion

from .protocol import ProtocolCommand, ProtocolCommandStatus, RobotAgentProtocol


STATUS_MAP = {
    "RECEIVED": "accepted",
    "RUNNING": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "REJECTED": "rejected",
    "TIMED_OUT": "expired",
    "ABORTED_AGENT_RESTART": "unknown",
}


class RobotAgentGatewayDriver:
    """Control API driver for a robot-agent.v1 peer; no ROS credentials live here."""

    def __init__(self, peer: RobotAgentProtocol) -> None:
        self.peer = peer
        self._lease_sequence = 0
        self._active_robot: str | None = None
        self._lease_lock = threading.RLock()

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        value = self.peer.discover(robot_id)
        if not value.get("available"):
            with self._lease_lock:
                self._active_robot = None
        return AdapterResult(
            status="available" if value.get("available") else "unavailable",
            result=dict(value),
            error=None if value.get("available") else str(value.get("reason")),
        )

    def probe(self, mapping: MappingVersion) -> AdapterResult:
        value = self.peer.discover(mapping.mapping.get("targetId"))
        capabilities = value.get("capabilities")
        if not value.get("available") or not isinstance(capabilities, list) or mapping.commandKey not in capabilities:
            return AdapterResult(status="unavailable", error="Robot Agent capability is unavailable")
        return AdapterResult(
            status="probed",
            details={
                "protocol": value.get("protocol"),
                "capability": mapping.commandKey,
                "executionBackend": value.get("executionBackend"),
            },
        )

    def dispatch(self, request: CommandRequest, mapping: MappingVersion) -> AdapterResult:
        self._ensure_lease(request.robotId)
        canonical = json.dumps(request.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ttl_ms = max(1, int((request.expiresAt - request.issuedAt).total_seconds() * 1000))
        status = self.peer.submit(
            ProtocolCommand(
                command_id=self._wire_command_id(request.commandId),
                robot_id=request.robotId,
                kind=request.commandKey,
                issued_at=request.issuedAt.astimezone(timezone.utc),
                ttl_ms=ttl_ms,
                params=request.payload,
                payload_hash=hashlib.sha256(canonical).hexdigest(),
            )
        )
        return self._result(status)

    def cancel(self, command_id: str) -> AdapterResult:
        return self._result(self.peer.cancel(self._wire_command_id(command_id)))

    def query(self, command_id: str) -> AdapterResult:
        return self._result(self.peer.query(self._wire_command_id(command_id)))

    @staticmethod
    def _wire_command_id(command_id: str) -> str:
        """Normalize logical Mission IDs to Robot Agent's UUID wire contract.

        Mission V2 keeps a human-readable ``skill-...`` request ID for audit,
        durable state and UI correlation. Robot Agent v1 requires UUID command
        IDs, so Mission V2's ``skill-`` values use a stable UUID5 derived from
        that logical ID. Retries, query and cancellation therefore reuse the
        same physical command identity when the underlying ledger can
        reconcile it after a restart; existing UUID IDs pass through
        unchanged. Other arbitrary non-UUID values are left for the protocol
        peer to reject.
        """

        try:
            return str(UUID(command_id))
        except (TypeError, ValueError):
            if isinstance(command_id, str) and command_id.startswith("skill-"):
                return str(uuid5(NAMESPACE_URL, f"h-ide:robot-agent:command:{command_id}"))
            return command_id

    def _ensure_lease(self, robot_id: str) -> None:
        with self._lease_lock:
            if self._active_robot == robot_id:
                self._lease_sequence += 1
                try:
                    self.peer.renew(robot_id, self._lease_sequence)
                    return
                except (RuntimeError, TimeoutError, ValueError):
                    # A reconnected Robot Agent starts in SAFE_HOLD. Reacquire
                    # explicitly instead of leaving the driver stuck with a
                    # stale local lease marker.
                    self._active_robot = None
            self.peer.acquire(robot_id)
            self._active_robot = robot_id
            self._lease_sequence = 1
            try:
                self.peer.renew(robot_id, self._lease_sequence)
            except (RuntimeError, TimeoutError, ValueError):
                self._active_robot = None
                raise

    @staticmethod
    def _result(status: ProtocolCommandStatus) -> AdapterResult:
        normalized = STATUS_MAP.get(status.state, "unknown")
        if status.result is None:
            result: dict[str, Any] = {}
        elif isinstance(status.result, Mapping):
            result = dict(status.result)
        else:
            # Robot Agent's wire contract permits any JSON value. Keep the
            # adapter result object-shaped for Control API receipts without
            # dropping a scalar/list diagnostic.
            result = {"value": status.result}
        result.setdefault("robotAgentCommandId", status.command_id)
        return AdapterResult(
            status=normalized,
            result=result,
            error=None if normalized in {"accepted", "running", "succeeded", "cancelled"} else status.detail,
            details={"protocolState": status.state, "detail": status.detail},
        )
