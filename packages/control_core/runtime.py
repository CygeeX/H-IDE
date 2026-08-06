"""Action command lifecycle and per-robot safety leases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import threading
from typing import Any, Callable, Mapping

from .adapters import AdapterResult, CommandAdapter
from .models import CommandDefinition, CommandReceipt, CommandRequest, CommandStatus, MappingVersion
from .validation import utc_now, validate_command_request


@dataclass
class _Lease:
    command_id: str
    expires_at: datetime


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
        self._lock = threading.RLock()

    @staticmethod
    def _adapter_key(value: Any) -> str:
        return getattr(value, "value", value)

    @staticmethod
    def _priority(key: str) -> bool:
        normalized = key.lower().replace("_", ".").replace("-", ".")
        return any(token in normalized for token in ("stop", "cancel", "estop", "emergency"))

    def _event(self, request: CommandRequest, mapping: MappingVersion, status: CommandStatus, **extra: Any) -> None:
        if self.event_sink:
            event = {
                "commandId": request.commandId,
                "robotId": request.robotId,
                "mappingVersion": mapping.version,
                "traceId": request.traceId,
                "status": status.value,
                "at": self.clock(),
            }
            event.update(extra)
            self.event_sink(event)

    def _receipt(self, request: CommandRequest, status: CommandStatus, **values: Any) -> CommandReceipt:
        current = self._receipts.get(request.commandId)
        accepted = current.acceptedAt if current is not None else self.clock()
        event_extra = {}
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

    def _expire_leases(self) -> None:
        now = self.clock()
        expired = [robot for robot, lease in self._leases.items() if lease.expires_at <= now]
        for robot in expired:
            lease = self._leases.pop(robot)
            req = self._requests.get(lease.command_id)
            mapping = self._mappings.get(lease.command_id)
            if req and mapping:
                adapter = self.adapters.get(self._adapter_key(req.transport))
                if adapter:
                    adapter.cancel(req.commandId)
                self._receipt(req, CommandStatus.EXPIRED, error="motion lease expired", completedAt=now, safetyStop=True)

    def tick(self) -> None:
        """Expire stale leases and issue adapter safety cancellation."""
        with self._lock:
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
        with self._lock:
            return self._dispatch_unlocked(request, definition, mapping, lease_seconds=lease_seconds)

    def _dispatch_unlocked(
        self,
        request: CommandRequest,
        definition: CommandDefinition,
        mapping: MappingVersion,
        *,
        lease_seconds: float | None = None,
    ) -> CommandReceipt:
        self._expire_leases()
        if request.commandId in self._receipts:
            return self._receipts[request.commandId]
        if request.idempotencyKey in self._idempotency:
            return self._receipts[self._idempotency[request.idempotencyKey]]
        self._idempotency[request.idempotencyKey] = request.commandId
        self._requests[request.commandId] = request
        self._mappings[request.commandId] = mapping

        try:
            validate_command_request(request)
        except ValueError as exc:
            return self._rejected(request, mapping, str(exc))
        if request.commandKey != definition.commandKey or request.commandVersion != definition.commandVersion:
            return self._rejected(request, mapping, "command definition mismatch")
        if mapping.commandKey != request.commandKey or mapping.transport != request.transport:
            return self._rejected(request, mapping, "mapping mismatch")
        transports = {self._adapter_key(t) for t in definition.transports}
        if self._adapter_key(request.transport) not in transports:
            return self._rejected(request, mapping, "transport is not supported by command definition")

        robot = request.robotId
        if self._priority(request.commandKey):
            active = self._leases.pop(robot, None)
            if active:
                prior_req = self._requests.get(active.command_id)
                if prior_req:
                    prior_adapter = self.adapters.get(self._adapter_key(prior_req.transport))
                    cancel_error = None
                    if prior_adapter:
                        try:
                            cancel_result = prior_adapter.cancel(active.command_id)
                            if str(cancel_result.status).lower() in {"failed", "unavailable", "unknown"}:
                                cancel_error = cancel_result.error or f"adapter returned {cancel_result.status}"
                        except Exception as exc:
                            cancel_error = str(exc)
                    else:
                        cancel_error = "adapter unavailable during safety stop"
                    self._receipt(
                        prior_req,
                        CommandStatus.CANCELLED,
                        error=cancel_error,
                        completedAt=self.clock(),
                        safetyStop=True,
                        safetyWarning=cancel_error,
                    )
        elif robot in self._leases:
            return self._rejected(request, mapping, "robot motion lease is held")

        self._receipt(request, CommandStatus.ACCEPTED)
        adapter = self.adapters.get(self._adapter_key(request.transport))
        if adapter is None:
            return self._receipt(request, CommandStatus.FAILED, error="adapter unavailable", completedAt=self.clock())
        self._receipt(request, CommandStatus.RUNNING, startedAt=self.clock())
        try:
            outcome = adapter.dispatch(request, mapping)
        except Exception as exc:
            outcome = AdapterResult(status="failed", error=str(exc))
        status = str(outcome.status).lower()
        if status in {"succeeded", "success", "ok", "completed"}:
            result = self._receipt(request, CommandStatus.SUCCEEDED, result=outcome.result, completedAt=self.clock())
            self._leases.pop(robot, None)
            return result
        if status in {"cancelled", "canceled"}:
            result = self._receipt(request, CommandStatus.CANCELLED, error=outcome.error, completedAt=self.clock())
            self._leases.pop(robot, None)
            return result
        if status in {"running", "accepted", "pending"}:
            seconds = self.lease_seconds if lease_seconds is None else max(0.0, float(lease_seconds))
            self._leases[robot] = _Lease(request.commandId, min(request.expiresAt, self.clock() + timedelta(seconds=seconds)))
            return self._receipts[request.commandId]
        # Unknown/unavailable must be explicit failure; never claim success.
        self._leases.pop(robot, None)
        return self._receipt(request, CommandStatus.FAILED, error=outcome.error or f"adapter returned {outcome.status}", completedAt=self.clock())

    submit = dispatch

    def cancel(self, command_id: str) -> CommandReceipt:
        with self._lock:
            return self._cancel_unlocked(command_id)

    def _cancel_unlocked(self, command_id: str) -> CommandReceipt:
        if command_id not in self._receipts:
            raise ValueError("command not found")
        request = self._requests[command_id]
        mapping = self._mappings[command_id]
        current = self._receipts[command_id]
        if current.status in {CommandStatus.SUCCEEDED, CommandStatus.FAILED, CommandStatus.CANCELLED, CommandStatus.EXPIRED, CommandStatus.REJECTED}:
            return current
        adapter = self.adapters.get(self._adapter_key(request.transport))
        if adapter:
            outcome = adapter.cancel(command_id)
            if str(outcome.status).lower() in {"failed", "unavailable"}:
                self._leases.pop(request.robotId, None)
                message = outcome.error or "cancel failed"
                return self._receipt(request, CommandStatus.FAILED, error=message, completedAt=self.clock(), safetyStop=True, safetyWarning=message)
        self._leases.pop(request.robotId, None)
        return self._receipt(request, CommandStatus.CANCELLED, completedAt=self.clock())

    def receipt(self, command_id: str) -> CommandReceipt:
        return self._receipts[command_id]


__all__ = ["ActionRuntime"]
