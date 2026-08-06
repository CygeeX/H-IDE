"""Action command lifecycle and per-robot safety leases.

The runtime owns only validated immutable domain objects.  Adapter calls are
deliberately made outside the runtime lock: a lost robot connection must not
prevent a stop or cancellation from acquiring the lease state lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from typing import Any, Callable, Mapping

from .adapters import AdapterResult, CommandAdapter
from .models import CommandDefinition, CommandReceipt, CommandRequest, CommandStatus, MappingVersion
from .validation import utc_now, validate_command_request


@dataclass
class _Lease:
    command_id: str
    expires_at: datetime


@dataclass
class _InFlight:
    request: CommandRequest
    mapping: MappingVersion
    adapter: CommandAdapter
    motion: bool


class ActionRuntime:
    def __init__(
        self,
        adapters: Mapping[str, CommandAdapter] | None = None,
        *,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], datetime] = utc_now,
        lease_seconds: float = 30.0,
    ):
        self.adapters = dict(adapters or {})
        self.event_sink = event_sink
        self.clock = clock
        self.lease_seconds = max(0.0, float(lease_seconds))
        self._receipts: dict[str, CommandReceipt] = {}
        self._idempotency: dict[str, str] = {}
        self._requests: dict[str, CommandRequest] = {}
        self._mappings: dict[str, MappingVersion] = {}
        self._leases: dict[str, _Lease] = {}
        self._inflight: dict[str, _InFlight] = {}
        self._stop_barriers: set[str] = set()
        self._stop_epochs: dict[str, int] = {}
        self._stop_owners: dict[str, tuple[int, str]] = {}
        self._lock = threading.RLock()
        # A local diagnostic trail is intentionally small and dependency-free.
        self.diagnostics: list[dict[str, Any]] = []

    @staticmethod
    def _adapter_key(value: Any) -> str:
        return getattr(value, "value", value)

    @staticmethod
    def _priority(key: str) -> bool:
        normalized = key.lower().replace("_", ".").replace("-", ".")
        return any(token in normalized for token in ("stop", "cancel", "estop", "emergency"))

    def _warn(self, message: str, *, exc: Exception | None = None) -> None:
        item: dict[str, Any] = {"message": message}
        if exc is not None:
            item["error"] = str(exc)
        self.diagnostics.append(item)

    def _event(self, request: CommandRequest, mapping: MappingVersion, status: CommandStatus, **extra: Any) -> None:
        if not self.event_sink:
            return
        event = {
            "commandId": request.commandId,
            "robotId": request.robotId,
            "mappingVersion": mapping.version,
            "traceId": request.traceId,
            "status": status.value,
            "at": self.clock(),
        }
        event.update(extra)
        try:
            self.event_sink(event)
        except Exception as exc:
            self._warn("event sink failed", exc=exc)

    @staticmethod
    def _normalize_result(value: Any, operation: str) -> AdapterResult:
        if not isinstance(value, AdapterResult):
            return AdapterResult(status="failed", error=f"adapter {operation} returned invalid result")
        status = getattr(value, "status", None)
        if not isinstance(status, str) or not status.strip():
            return AdapterResult(status="failed", error=f"adapter {operation} returned result without status")
        return value

    def _receipt(self, request: CommandRequest, status: CommandStatus, **values: Any) -> CommandReceipt:
        current = self._receipts.get(request.commandId)
        accepted = current.acceptedAt if current is not None else self.clock()
        event_extra: dict[str, Any] = {}
        for event_key in ("safetyStop", "safetyWarning"):
            if event_key in values:
                event_extra[event_key] = values.pop(event_key)
        fields: dict[str, Any] = {
            "commandId": request.commandId,
            "robotId": request.robotId,
            "status": status,
            "acceptedAt": accepted,
            "startedAt": values.pop("startedAt", current.startedAt if current else None),
            "completedAt": values.pop("completedAt", current.completedAt if current else None),
            "result": values.pop("result", current.result if current else None),
            "error": values.pop("error", current.error if current else None),
        }
        fields.update(values)
        receipt = CommandReceipt(**fields)
        self._receipts[request.commandId] = receipt
        self._event(request, self._mappings[request.commandId], status, error=receipt.error, **event_extra)
        return receipt

    def _rejected(self, request: CommandRequest, mapping: MappingVersion, reason: str) -> CommandReceipt:
        self._mappings[request.commandId] = mapping
        return self._receipt(request, CommandStatus.REJECTED, error=reason, completedAt=self.clock())

    def _release_lease(self, robot_id: str, command_id: str) -> None:
        lease = self._leases.get(robot_id)
        if lease is not None and lease.command_id == command_id:
            self._leases.pop(robot_id, None)

    def _clear_stop_barrier(self, robot_id: str, command_id: str, epoch: int | None) -> None:
        if epoch is not None and self._stop_owners.get(robot_id) == (epoch, command_id):
            self._stop_owners.pop(robot_id, None)
            self._stop_barriers.discard(robot_id)

    def _cancel_external(self, action: _InFlight | None, command_id: str) -> str | None:
        """Call adapter cancellation outside the lock and return a warning."""
        if action is None:
            return "adapter unavailable during safety stop"
        try:
            outcome = self._normalize_result(action.adapter.cancel(command_id), "cancel")
        except Exception as exc:
            return f"safety stop failed: {exc}"
        status = str(outcome.status).lower()
        if status not in {"cancelled", "canceled"}:
            return outcome.error or f"adapter returned {outcome.status} while cancelling"
        return None

    def _expire_leases(self) -> None:
        actions: list[tuple[CommandRequest, _InFlight | None]] = []
        now = self.clock()
        with self._lock:
            expired = [robot for robot, lease in self._leases.items() if lease.expires_at <= now]
            for robot in expired:
                lease = self._leases.pop(robot)
                req = self._requests.get(lease.command_id)
                action = self._inflight.get(lease.command_id)
                if req:
                    mapping = self._mappings[req.commandId]
                    self._receipt(req, CommandStatus.EXPIRED, error="motion lease expired", completedAt=now, safetyStop=True)
                    actions.append((req, action))
        for req, action in actions:
            warning = self._cancel_external(action, req.commandId)
            if warning:
                with self._lock:
                    if self._receipts.get(req.commandId, None) is not None:
                        self._receipt(req, CommandStatus.EXPIRED, error=f"motion lease expired; {warning}", completedAt=now, safetyStop=True, safetyWarning=warning)

    def tick(self) -> None:
        """Expire stale leases and issue adapter safety cancellation."""
        self._expire_leases()

    expire_leases = tick

    def dispatch(
        self,
        request: CommandRequest,
        definition: CommandDefinition,
        mapping: MappingVersion,
        *,
        lease_seconds: float | None = None,
    ) -> CommandReceipt:
        self._expire_leases()
        cancel_actions: list[tuple[CommandRequest, _InFlight | None]] = []
        with self._lock:
            context = self._begin_dispatch(request, definition, mapping, lease_seconds)
            if isinstance(context, CommandReceipt):
                return context
            adapter, motion, cancel_actions, is_stop, stop_epoch = context
            # Stop commands may have displaced an old lease.  Their receipt is
            # terminal before any external cancellation is attempted.
        for old_req, old_action in cancel_actions:
            warning = self._cancel_external(old_action, old_req.commandId)
            if warning:
                with self._lock:
                    self._receipt(old_req, CommandStatus.CANCELLED, error=warning, completedAt=self.clock(), safetyStop=True, safetyWarning=warning)
        try:
            outcome = self._normalize_result(adapter.dispatch(request, mapping), "dispatch")
        except Exception as exc:
            outcome = AdapterResult(status="failed", error=f"adapter dispatch failed: {exc}")
        with self._lock:
            return self._finish_dispatch(request, mapping, outcome, motion, is_stop, stop_epoch)

    def _begin_dispatch(self, request: CommandRequest, definition: CommandDefinition, mapping: MappingVersion, lease_seconds: float | None):
        if request.commandId in self._receipts:
            return self._receipts[request.commandId]
        if request.idempotencyKey in self._idempotency:
            return self._receipts[self._idempotency[request.idempotencyKey]]
        self._idempotency[request.idempotencyKey] = request.commandId
        self._requests[request.commandId] = request
        self._mappings[request.commandId] = mapping
        try:
            if request.expiresAt <= self.clock():
                return self._rejected(request, mapping, "expiresAt is in the past")
            validate_command_request(request)
        except ValueError as exc:
            return self._rejected(request, mapping, str(exc))
        if request.commandKey != definition.commandKey or request.commandVersion != definition.commandVersion:
            return self._rejected(request, mapping, "command definition mismatch")
        if mapping.commandKey != request.commandKey or mapping.transport != request.transport:
            return self._rejected(request, mapping, "mapping mismatch")
        if self._adapter_key(request.transport) not in {self._adapter_key(t) for t in definition.transports}:
            return self._rejected(request, mapping, "transport is not supported by command definition")
        robot = request.robotId
        is_stop = self._priority(request.commandKey)
        stop_epoch: int | None = None
        cancel_actions: list[tuple[CommandRequest, _InFlight | None]] = []
        if is_stop:
            stop_epoch = self._stop_epochs.get(robot, 0) + 1
            self._stop_epochs[robot] = stop_epoch
            self._stop_barriers.add(robot)
            self._stop_owners[robot] = (stop_epoch, request.commandId)
            active = self._leases.pop(robot, None)
            if active:
                prior_req = self._requests.get(active.command_id)
                prior_action = self._inflight.get(active.command_id)
                if prior_req:
                    self._receipt(prior_req, CommandStatus.CANCELLED, completedAt=self.clock(), safetyStop=True)
                    cancel_actions.append((prior_req, prior_action))
        elif robot in self._stop_barriers:
            return self._rejected(request, mapping, "robot stop barrier is active")
        elif robot in self._leases:
            return self._rejected(request, mapping, "robot motion lease is held")
        self._receipt(request, CommandStatus.ACCEPTED)
        adapter = self.adapters.get(self._adapter_key(request.transport))
        if adapter is None:
            if is_stop:
                self._clear_stop_barrier(robot, request.commandId, stop_epoch)
            return self._receipt(request, CommandStatus.FAILED, error="adapter unavailable", completedAt=self.clock())
        self._receipt(request, CommandStatus.RUNNING, startedAt=self.clock())
        motion = not self._priority(request.commandKey)
        if motion:
            seconds = self.lease_seconds if lease_seconds is None else max(0.0, float(lease_seconds))
            self._leases[robot] = _Lease(request.commandId, min(request.expiresAt, self.clock() + timedelta(seconds=seconds)))
        self._inflight[request.commandId] = _InFlight(request, mapping, adapter, motion)
        return (adapter, motion, cancel_actions, is_stop, stop_epoch)

    def _finish_dispatch(self, request: CommandRequest, mapping: MappingVersion, outcome: AdapterResult, motion: bool, is_stop: bool = False, stop_epoch: int | None = None) -> CommandReceipt:
        current = self._receipts[request.commandId]
        if current.status not in {CommandStatus.RUNNING, CommandStatus.ACCEPTED}:
            self._inflight.pop(request.commandId, None)
            if is_stop:
                self._clear_stop_barrier(request.robotId, request.commandId, stop_epoch)
            return current
        status = str(outcome.status).lower()
        if status in {"succeeded", "success", "ok", "completed"}:
            receipt = self._receipt(request, CommandStatus.SUCCEEDED, result=outcome.result, completedAt=self.clock())
            self._release_lease(request.robotId, request.commandId)
            self._inflight.pop(request.commandId, None)
            if is_stop:
                self._clear_stop_barrier(request.robotId, request.commandId, stop_epoch)
            return receipt
        if status in {"cancelled", "canceled"}:
            receipt = self._receipt(request, CommandStatus.CANCELLED, error=outcome.error, completedAt=self.clock())
            self._release_lease(request.robotId, request.commandId)
            self._inflight.pop(request.commandId, None)
            if is_stop:
                self._clear_stop_barrier(request.robotId, request.commandId, stop_epoch)
            return receipt
        if status in {"running", "accepted", "pending"}:
            return current
        self._release_lease(request.robotId, request.commandId)
        self._inflight.pop(request.commandId, None)
        if is_stop:
            self._clear_stop_barrier(request.robotId, request.commandId, stop_epoch)
        return self._receipt(request, CommandStatus.FAILED, error=outcome.error or f"adapter returned {outcome.status}", completedAt=self.clock())

    submit = dispatch

    def cancel(self, command_id: str) -> CommandReceipt:
        with self._lock:
            if command_id not in self._receipts:
                raise ValueError("command not found")
            request = self._requests[command_id]
            current = self._receipts[command_id]
            if current.status in {CommandStatus.SUCCEEDED, CommandStatus.FAILED, CommandStatus.CANCELLED, CommandStatus.EXPIRED, CommandStatus.REJECTED}:
                return current
            action = self._inflight.get(command_id)
            stop_epoch = self._stop_owners.get(request.robotId, (None, None))[0] if self._priority(request.commandKey) else None
            self._release_lease(request.robotId, command_id)
            self._receipt(request, CommandStatus.CANCELLED, completedAt=self.clock(), safetyStop=True)
        warning = self._cancel_external(action, command_id)
        if warning:
            with self._lock:
                if self._priority(request.commandKey):
                    self._clear_stop_barrier(request.robotId, request.commandId, stop_epoch)
                return self._receipt(request, CommandStatus.FAILED, error=warning, completedAt=self.clock(), safetyStop=True, safetyWarning=warning)
        with self._lock:
            if self._priority(request.commandKey):
                self._clear_stop_barrier(request.robotId, request.commandId, stop_epoch)
        return self.receipt(command_id)

    def receipt(self, command_id: str) -> CommandReceipt:
        with self._lock:
            return self._receipts[command_id]


__all__ = ["ActionRuntime"]
