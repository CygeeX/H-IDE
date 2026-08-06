"""Action command lifecycle and per-robot safety leases.

The runtime owns only validated immutable domain objects.  Adapter calls are
deliberately made outside the runtime lock: a lost robot connection must not
prevent a stop or cancellation from acquiring the lease state lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import threading
from typing import Any, Callable, Mapping

from .adapters import AdapterResult, CommandAdapter
from .models import CommandDefinition, CommandKind, CommandReceipt, CommandRequest, CommandStatus, MappingVersion
from .registry import validate_mapping_safety
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
    kind: CommandKind


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
        self.lease_seconds = float(lease_seconds)
        if not math.isfinite(self.lease_seconds) or self.lease_seconds < 0:
            raise ValueError("lease_seconds must be finite and non-negative")
        self._receipts: dict[str, CommandReceipt] = {}
        self._idempotency: dict[str, str] = {}
        self._requests: dict[str, CommandRequest] = {}
        self._definitions: dict[str, CommandDefinition] = {}
        self._mappings: dict[str, MappingVersion] = {}
        self._leases: dict[str, _Lease] = {}
        self._inflight: dict[str, _InFlight] = {}
        self._safety_barriers: set[str] = set()
        self._safety_epochs: dict[str, int] = {}
        self._safety_owners: dict[str, tuple[int, str]] = {}
        self._lock = threading.RLock()
        # A local diagnostic trail is intentionally small and dependency-free.
        self.diagnostics: list[dict[str, Any]] = []
        self._event_queue: list[dict[str, Any]] = []
        self._event_flush_lock = threading.Lock()

    @staticmethod
    def _adapter_key(value: Any) -> str:
        return getattr(value, "value", value)

    def _warn(self, message: str, *, exc: Exception | None = None) -> None:
        item: dict[str, Any] = {"message": message}
        if exc is not None:
            item["error"] = str(exc)
        with self._lock:
            self.diagnostics.append(item)

    def _event(self, request: CommandRequest, mapping: MappingVersion, status: CommandStatus, **extra: Any) -> None:
        event = {
            "commandId": request.commandId,
            "robotId": request.robotId,
            "mappingVersion": mapping.version,
            "traceId": request.traceId,
            "status": status.value,
            "at": self.clock(),
        }
        event.update(extra)
        self._event_queue.append(event)

    def _flush_events(self) -> None:
        # A callback already being flushed by another caller must not make a
        # safety operation wait; the current flusher drains events queued while
        # its callback was running.
        if not self._event_flush_lock.acquire(blocking=False):
            return
        try:
            while True:
                with self._lock:
                    events = self._event_queue
                    self._event_queue = []
                if not self.event_sink:
                    break
                if not events:
                    break
                for event in events:
                    try:
                        self.event_sink(event)
                    except Exception as exc:
                        self._warn("event sink failed", exc=exc)
        finally:
            self._event_flush_lock.release()

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

    def _claim_safety_barrier(self, robot_id: str, owner: str, *, supersede: bool = False) -> int | None:
        if robot_id in self._safety_barriers and not supersede:
            return None
        epoch = self._safety_epochs.get(robot_id, 0) + 1
        self._safety_epochs[robot_id] = epoch
        self._safety_owners[robot_id] = (epoch, owner)
        self._safety_barriers.add(robot_id)
        return epoch

    def _clear_safety_barrier(self, robot_id: str, owner: str, epoch: int | None) -> None:
        if epoch is not None and self._safety_owners.get(robot_id) == (epoch, owner):
            self._safety_owners.pop(robot_id, None)
            self._safety_barriers.discard(robot_id)

    def _transfer_safety_barrier(self, robot_id: str, from_owner: str, to_owner: str) -> int | None:
        current = self._safety_owners.get(robot_id)
        if current is None or current[1] != from_owner:
            return None
        self._safety_owners[robot_id] = (current[0], to_owner)
        return current[0]

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
        actions: list[tuple[CommandRequest, _InFlight | None, str, int | None]] = []
        now = self.clock()
        with self._lock:
            expired = [robot for robot, lease in self._leases.items() if lease.expires_at <= now]
            for robot in expired:
                lease = self._leases.pop(robot)
                req = self._requests.get(lease.command_id)
                action = self._inflight.get(lease.command_id)
                if req:
                    owner = f"expiry:{req.commandId}"
                    epoch = self._claim_safety_barrier(req.robotId, owner)
                    self._receipt(req, CommandStatus.EXPIRED, error="motion lease expired", completedAt=now, safetyStop=True)
                    actions.append((req, action, owner, epoch))
        self._flush_events()
        for req, action, owner, epoch in actions:
            warning = self._cancel_external(action, req.commandId)
            with self._lock:
                if warning:
                    if self._receipts.get(req.commandId, None) is not None:
                        self._receipt(req, CommandStatus.EXPIRED, error=f"motion lease expired; {warning}", completedAt=now, safetyStop=True, safetyWarning=warning)
                self._clear_safety_barrier(req.robotId, owner, epoch)
            self._flush_events()

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
        self._flush_events()
        if isinstance(context, CommandReceipt):
            return context
        adapter, motion, cancel_actions, is_safety, safety_epoch, safety_owner, pre_failure = context
        # Safety commands may displace an old lease. Its receipt is terminal
        # before any external cancellation is attempted.
        cancel_warnings: list[str] = []
        for old_req, old_action in cancel_actions:
            warning = self._cancel_external(old_action, old_req.commandId)
            if warning:
                cancel_warnings.append(warning)
                with self._lock:
                    self._receipt(old_req, CommandStatus.CANCELLED, error=warning, completedAt=self.clock(), safetyStop=True, safetyWarning=warning)
        self._flush_events()
        if adapter is None:
            with self._lock:
                details = pre_failure or "adapter unavailable"
                if cancel_warnings:
                    details = f"{details}; previous cancellation failed: {'; '.join(cancel_warnings)}"
                receipt = self._receipt(request, CommandStatus.FAILED, error=details, completedAt=self.clock(), safetyStop=is_safety, safetyWarning=(cancel_warnings[0] if cancel_warnings else None))
                if is_safety:
                    self._clear_safety_barrier(request.robotId, safety_owner, safety_epoch)
            self._flush_events()
            return receipt
        try:
            outcome = self._normalize_result(adapter.dispatch(request, mapping), "dispatch")
        except Exception as exc:
            outcome = AdapterResult(status="failed", error=f"adapter dispatch failed: {exc}")
        with self._lock:
            receipt = self._finish_dispatch(request, mapping, outcome, motion, is_safety, safety_epoch, safety_owner)
            if cancel_warnings and is_safety:
                receipt = self._receipt(request, CommandStatus.FAILED, error=f"previous cancellation failed: {'; '.join(cancel_warnings)}", completedAt=self.clock(), safetyStop=True, safetyWarning=cancel_warnings[0])
        self._flush_events()
        return receipt

    def _begin_dispatch(self, request: CommandRequest, definition: CommandDefinition, mapping: MappingVersion, lease_seconds: float | None):
        if request.commandId in self._receipts:
            return self._receipts[request.commandId]
        if request.idempotencyKey in self._idempotency:
            return self._receipts[self._idempotency[request.idempotencyKey]]
        self._idempotency[request.idempotencyKey] = request.commandId
        self._requests[request.commandId] = request
        self._definitions[request.commandId] = definition
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
        try:
            validate_mapping_safety(mapping)
        except ValueError as exc:
            return self._rejected(request, mapping, str(exc))
        if self._adapter_key(request.transport) not in {self._adapter_key(t) for t in definition.transports}:
            return self._rejected(request, mapping, "transport is not supported by command definition")
        try:
            seconds = self.lease_seconds if lease_seconds is None else float(lease_seconds)
        except (TypeError, ValueError, OverflowError):
            return self._rejected(request, mapping, "leaseSeconds must be finite and non-negative")
        if not math.isfinite(seconds) or seconds < 0:
            return self._rejected(request, mapping, "leaseSeconds must be finite and non-negative")
        try:
            lease_duration = timedelta(seconds=seconds)
        except OverflowError:
            return self._rejected(request, mapping, "leaseSeconds is outside the supported range")
        robot = request.robotId
        is_safety = definition.kind is CommandKind.SAFETY
        motion = definition.kind is CommandKind.MOTION
        safety_owner = f"dispatch:{request.commandId}"
        safety_epoch: int | None = None
        cancel_actions: list[tuple[CommandRequest, _InFlight | None]] = []
        if is_safety:
            safety_epoch = self._claim_safety_barrier(robot, safety_owner, supersede=True)
            active = self._leases.pop(robot, None)
            if active:
                prior_req = self._requests.get(active.command_id)
                prior_action = self._inflight.get(active.command_id)
                if prior_req:
                    self._receipt(prior_req, CommandStatus.CANCELLED, completedAt=self.clock(), safetyStop=True)
                    cancel_actions.append((prior_req, prior_action))
        elif motion and robot in self._safety_barriers:
            return self._rejected(request, mapping, "robot safety barrier is active")
        elif motion and robot in self._leases:
            return self._rejected(request, mapping, "robot motion lease is held")
        self._receipt(request, CommandStatus.ACCEPTED)
        adapter = self.adapters.get(self._adapter_key(request.transport))
        if adapter is None:
            if is_safety and cancel_actions:
                return (None, motion, cancel_actions, is_safety, safety_epoch, safety_owner, "adapter unavailable")
            if is_safety:
                self._clear_safety_barrier(robot, safety_owner, safety_epoch)
            return self._receipt(request, CommandStatus.FAILED, error="adapter unavailable", completedAt=self.clock())
        self._receipt(request, CommandStatus.RUNNING, startedAt=self.clock())
        if motion:
            self._leases[robot] = _Lease(request.commandId, min(request.expiresAt, self.clock() + lease_duration))
        self._inflight[request.commandId] = _InFlight(request, mapping, adapter, motion, definition.kind)
        return (adapter, motion, cancel_actions, is_safety, safety_epoch, safety_owner, None)

    def _finish_dispatch(self, request: CommandRequest, mapping: MappingVersion, outcome: AdapterResult, motion: bool, is_safety: bool = False, safety_epoch: int | None = None, safety_owner: str = "") -> CommandReceipt:
        current = self._receipts[request.commandId]
        if current.status not in {CommandStatus.RUNNING, CommandStatus.ACCEPTED}:
            self._inflight.pop(request.commandId, None)
            if is_safety:
                self._clear_safety_barrier(request.robotId, safety_owner, safety_epoch)
            return current
        status = str(outcome.status).lower()
        if status in {"succeeded", "success", "ok", "completed"}:
            receipt = self._receipt(request, CommandStatus.SUCCEEDED, result=outcome.result, completedAt=self.clock())
            self._release_lease(request.robotId, request.commandId)
            self._inflight.pop(request.commandId, None)
            if is_safety:
                self._clear_safety_barrier(request.robotId, safety_owner, safety_epoch)
            return receipt
        if status in {"cancelled", "canceled"}:
            receipt = self._receipt(request, CommandStatus.CANCELLED, error=outcome.error, completedAt=self.clock())
            self._release_lease(request.robotId, request.commandId)
            self._inflight.pop(request.commandId, None)
            if is_safety:
                self._clear_safety_barrier(request.robotId, safety_owner, safety_epoch)
            return receipt
        if status in {"running", "accepted", "pending"}:
            return current
        self._release_lease(request.robotId, request.commandId)
        self._inflight.pop(request.commandId, None)
        if is_safety:
            self._clear_safety_barrier(request.robotId, safety_owner, safety_epoch)
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
            owner = f"cancel:{command_id}"
            if action is not None and action.kind is CommandKind.SAFETY:
                safety_epoch = self._transfer_safety_barrier(request.robotId, f"dispatch:{command_id}", owner)
            else:
                safety_epoch = self._claim_safety_barrier(request.robotId, owner)
            self._release_lease(request.robotId, command_id)
            self._receipt(request, CommandStatus.CANCELLED, completedAt=self.clock(), safetyStop=True)
        self._flush_events()
        warning = self._cancel_external(action, command_id)
        with self._lock:
            if warning:
                receipt = self._receipt(request, CommandStatus.FAILED, error=warning, completedAt=self.clock(), safetyStop=True, safetyWarning=warning)
            else:
                receipt = self._receipts[command_id]
            self._clear_safety_barrier(request.robotId, owner, safety_epoch)
        self._flush_events()
        return receipt

    def receipt(self, command_id: str) -> CommandReceipt:
        with self._lock:
            return self._receipts[command_id]


__all__ = ["ActionRuntime"]
