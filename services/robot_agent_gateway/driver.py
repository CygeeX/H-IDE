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


_PRE_DISPATCH_CANCEL_CAPACITY = 4096


class RobotAgentGatewayDriver:
    """Control API driver for a robot-agent.v1 peer; no ROS credentials live here."""

    def __init__(self, peer: RobotAgentProtocol) -> None:
        self.peer = peer
        self._lease_sequence = 0
        self._active_robot: str | None = None
        self._lease_lock = threading.RLock()
        self._command_robots: dict[str, str] = {}
        self._command_identities: dict[str, tuple[str, str, str]] = {}
        self._active_commands: dict[str, set[str]] = {}
        self._terminal_statuses: dict[str, ProtocolCommandStatus] = {}
        self._terminal_unreleased: dict[str, set[str]] = {}
        self._released_commands: dict[str, dict[str, Any]] = {}
        self._cancel_confirmed_commands: set[str] = set()
        self._release_pending: set[str] = set()
        # A cancel can race a dispatch on separate HTTP connections.  If the
        # cancel enters this driver first, remember the normalized command ID
        # so the later dispatch cannot acquire control or reach the Robot
        # Agent.  The ledger is deliberately process-lifetime state: eviction
        # could turn a proved cancel-first ordering into a false negative.
        self._pre_dispatch_cancels: dict[str, AdapterResult] = {}
        self._pre_dispatch_cancel_saturated = False

    @staticmethod
    def _canonical_wire_uuid(command_id: str) -> bool:
        try:
            return str(UUID(command_id)) == command_id
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _suppressed_dispatch(command_id: str) -> AdapterResult:
        truth = {
            "robotAgentCommandId": command_id,
            "dispatchSuppressed": True,
        }
        return AdapterResult(
            status="cancelled",
            result=truth,
            details={
                "protocolState": "CANCELLED",
                "dispatchSuppressed": True,
            },
        )

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
        wire_command_id = self._wire_command_id(request.commandId)
        canonical = json.dumps(
            request.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        payload_hash = hashlib.sha256(canonical).hexdigest()
        identity = (request.robotId, request.commandKey, payload_hash)
        with self._lease_lock:
            existing_identity = self._command_identities.get(wire_command_id)
            if wire_command_id in self._pre_dispatch_cancels:
                return self._suppressed_dispatch(wire_command_id)
            if (
                existing_identity is None
                and self._pre_dispatch_cancel_saturated
            ):
                return AdapterResult(
                    status="rejected",
                    error=(
                        "Robot Agent dispatch is fail-closed because the "
                        "pre-dispatch cancel ledger is saturated"
                    ),
                    details={
                        "protocolState": "REJECTED",
                        "dispatchSuppressed": True,
                    },
                )
            if existing_identity is not None and existing_identity != identity:
                return AdapterResult(
                    status="rejected",
                    error="Robot Agent command ID payload conflict",
                    details={"protocolState": "REJECTED"},
                )
            terminal = self._terminal_statuses.get(wire_command_id)
            if terminal is not None:
                return self._result_after_terminal_release(
                    terminal,
                    robot_id=request.robotId,
                    wire_command_id=wire_command_id,
                )
            self._ensure_lease(request.robotId)
            ttl_ms = max(
                1,
                int(
                    (request.expiresAt - request.issuedAt).total_seconds()
                    * 1000
                ),
            )
            self._command_robots[wire_command_id] = request.robotId
            self._command_identities[wire_command_id] = identity
            self._active_commands.setdefault(request.robotId, set()).add(
                wire_command_id
            )
            status = self.peer.submit(
                ProtocolCommand(
                    command_id=wire_command_id,
                    robot_id=request.robotId,
                    kind=request.commandKey,
                    issued_at=request.issuedAt.astimezone(timezone.utc),
                    ttl_ms=ttl_ms,
                    params=request.payload,
                    payload_hash=payload_hash,
                )
            )
            return self._result_after_terminal_release(
                status,
                robot_id=request.robotId,
                wire_command_id=wire_command_id,
            )

    def cancel(self, command_id: str) -> AdapterResult:
        wire_command_id = self._wire_command_id(command_id)
        with self._lease_lock:
            pre_dispatch = self._pre_dispatch_cancels.get(wire_command_id)
            if pre_dispatch is not None:
                return pre_dispatch
            terminal = self._terminal_statuses.get(wire_command_id)
            if terminal is not None:
                robot_id = self._command_robots.get(wire_command_id)
                if robot_id is None:
                    return self._result(terminal)
                return self._result_after_terminal_release(
                    terminal,
                    robot_id=robot_id,
                    wire_command_id=wire_command_id,
                )
            if (
                wire_command_id not in self._command_identities
                and self._canonical_wire_uuid(wire_command_id)
            ):
                if len(self._pre_dispatch_cancels) >= _PRE_DISPATCH_CANCEL_CAPACITY:
                    # Never evict a live tombstone.  Saturation instead closes
                    # admission for every previously unseen command ID.
                    self._pre_dispatch_cancel_saturated = True
                else:
                    # Install before touching the peer.  A dispatch arriving
                    # on another HTTP connection must serialize after this
                    # exact cancellation boundary.
                    self._pre_dispatch_cancels[wire_command_id] = (
                        self._suppressed_dispatch(wire_command_id)
                    )
            try:
                status = self.peer.cancel(wire_command_id)
            except Exception as exc:
                uncertain = AdapterResult(
                    status="unavailable",
                    error=(
                        "Robot Agent cancellation is unconfirmed: "
                        f"{type(exc).__name__}"
                    ),
                    details={
                        "protocolState": "UNKNOWN",
                        "dispatchSuppressed": (
                            wire_command_id in self._pre_dispatch_cancels
                            or self._pre_dispatch_cancel_saturated
                        ),
                    },
                )
                if wire_command_id in self._pre_dispatch_cancels:
                    self._pre_dispatch_cancels[wire_command_id] = uncertain
                return uncertain
            robot_id = self._command_robots.get(wire_command_id)
            if robot_id is None:
                if wire_command_id in self._pre_dispatch_cancels:
                    # REJECTED merely means the Robot Agent had not seen this
                    # ID yet.  The local tombstone is the authoritative proof
                    # that any later dispatch will be suppressed.  If the
                    # Robot Agent did know the ID, preserve its terminal truth.
                    if status.state == "REJECTED":
                        return self._pre_dispatch_cancels[wire_command_id]
                    result = self._result(status)
                    self._pre_dispatch_cancels[wire_command_id] = result
                    return result
                return self._result(status)
            if status.state == "CANCELLED":
                # The frozen cancel contract already waits for a new confirmed
                # SAFE_HOLD truth before returning CANCELLED.
                self._mark_terminal(robot_id, wire_command_id)
                self._terminal_statuses[wire_command_id] = status
                self._cancel_confirmed_commands.add(wire_command_id)
                pending = self._terminal_unreleased.get(robot_id)
                if pending is not None:
                    pending.discard(wire_command_id)
                    if not pending:
                        self._terminal_unreleased.pop(robot_id, None)
                if not self._active_commands.get(robot_id):
                    self._active_robot = None
                    self._release_pending.discard(robot_id)
                return self._result(status)
            return self._result_after_terminal_release(
                status,
                robot_id=robot_id,
                wire_command_id=wire_command_id,
            )

    def query(self, command_id: str) -> AdapterResult:
        wire_command_id = self._wire_command_id(command_id)
        with self._lease_lock:
            pre_dispatch = self._pre_dispatch_cancels.get(wire_command_id)
            if pre_dispatch is not None:
                return pre_dispatch
            status = self._terminal_statuses.get(wire_command_id)
            if status is None:
                status = self.peer.query(wire_command_id)
            robot_id = self._command_robots.get(wire_command_id)
            if robot_id is None:
                return self._result(status)
            return self._result_after_terminal_release(
                status,
                robot_id=robot_id,
                wire_command_id=wire_command_id,
            )

    @staticmethod
    def _terminal(status: ProtocolCommandStatus) -> bool:
        return status.state in STATUS_MAP and status.state not in {
            "RECEIVED",
            "RUNNING",
        }

    def _mark_terminal(self, robot_id: str, wire_command_id: str) -> None:
        active = self._active_commands.setdefault(robot_id, set())
        active.discard(wire_command_id)
        if not active:
            self._active_commands.pop(robot_id, None)
        self._terminal_unreleased.setdefault(robot_id, set()).add(
            wire_command_id
        )

    @staticmethod
    def _with_control_release(
        status: ProtocolCommandStatus,
        truth: Mapping[str, Any],
    ) -> AdapterResult:
        result = RobotAgentGatewayDriver._result(status)
        release = dict(truth)
        payload = dict(result.result) if isinstance(result.result, Mapping) else {}
        payload["controlRelease"] = release
        return AdapterResult(
            status=result.status,
            result=payload,
            error=result.error,
            details={**result.details, "controlRelease": release},
        )

    def _result_after_terminal_release(
        self,
        status: ProtocolCommandStatus,
        *,
        robot_id: str,
        wire_command_id: str,
    ) -> AdapterResult:
        """Expose a terminal command only after idle control is released.

        A second command may share the same Robot Agent control session.  In
        that case the first terminal must not revoke authority underneath the
        still-active command.  The last terminal command owns the release and
        keeps returning ``unavailable`` on any uncertain handshake so
        ActionRuntime retains its local motion lease.
        """

        if not self._terminal(status):
            self._active_commands.setdefault(robot_id, set()).add(
                wire_command_id
            )
            return self._result(status)
        self._terminal_statuses[wire_command_id] = status
        if wire_command_id in self._cancel_confirmed_commands:
            return self._result(status)
        released = self._released_commands.get(wire_command_id)
        if released is not None:
            return self._with_control_release(status, released)
        self._mark_terminal(robot_id, wire_command_id)
        if self._active_commands.get(robot_id):
            return self._result(status)
        self._release_pending.add(robot_id)
        try:
            released = self.peer.release(
                robot_id,
                f"terminal command {wire_command_id} released control",
            )
        except Exception as exc:
            return AdapterResult(
                status="unavailable",
                error=f"Robot Agent terminal control release is unconfirmed: {exc}",
                details={"protocolState": status.state},
            )
        expected: dict[str, Any] = {
            "state": "SAFE_HOLD",
            "authorityRevoked": True,
            "safeStopState": "confirmed",
        }
        if not isinstance(released, Mapping) or any(
            released.get(key) != value for key, value in expected.items()
        ):
            return AdapterResult(
                status="unavailable",
                error="Robot Agent terminal control release returned invalid safety truth",
                details={"protocolState": status.state},
            )
        epoch = released.get("safeStopEpoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
            return AdapterResult(
                status="unavailable",
                error="Robot Agent terminal control release returned invalid safe-stop epoch",
                details={"protocolState": status.state},
            )
        release_truth = {**expected, "safeStopEpoch": epoch}
        for terminal_command_id in self._terminal_unreleased.pop(
            robot_id, set()
        ):
            self._released_commands[terminal_command_id] = release_truth
        self._release_pending.discard(robot_id)
        if self._active_robot == robot_id:
            self._active_robot = None
        return self._with_control_release(status, release_truth)

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
            if robot_id in self._release_pending:
                raise RuntimeError(
                    "a prior terminal control release is not confirmed"
                )
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
