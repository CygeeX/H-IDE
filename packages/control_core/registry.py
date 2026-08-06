"""Versioned command mapping registry with explicit validation and activation."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import re
import threading
import unicodedata
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from .adapters import AdapterResult, CommandAdapter
from .models import MappingVersion, Transport
from .validation import freeze_json_object, freeze_json_value


def validate_mapping_safety(mapping: MappingVersion) -> None:
    """Pure validation for transport/mapping values safe to dispatch."""
    if mapping.transport not in set(Transport):
        raise ValueError("transport is not allowlisted")
    banned_tokens = ("script", "endpoint", "exec", "shell", "url")

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                raw_key = unicodedata.normalize("NFKC", str(key)).casefold()
                if not raw_key.isascii():
                    raise ValueError(f"mapping field {key} contains unsupported characters")
                normalized = re.sub(r"[^a-z0-9]", "", raw_key)
                if any(token in normalized for token in banned_tokens) or normalized in {"command", "commandline", "run"}:
                    raise ValueError(f"mapping field {key} is not allowed")
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        elif isinstance(value, str) and (urlparse(value).scheme in {"http", "https", "file", "ftp", "javascript", "data", "ws", "wss"} or value.startswith(("//", "\\\\")) or "://" in value):
            raise ValueError("arbitrary URL access is not allowed")

    walk(mapping.mapping)


def mapping_target_values(value: Any) -> tuple[str, ...]:
    """Collect target aliases recursively from an untrusted mapping payload."""
    targets: list[str] = []

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = re.sub(
                    r"[^a-z0-9]", "", unicodedata.normalize("NFKC", str(key)).casefold()
                )
                if normalized in {"targetid", "target"}:
                    if not isinstance(child, str) or not child.strip():
                        raise ValueError("target must be a non-empty string")
                    targets.append(child.strip())
                else:
                    walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    walk(value)
    if len(set(targets)) > 1:
        raise ValueError("mapping target aliases conflict")
    return tuple(targets)


def _snapshot_probe_result(value: Any) -> AdapterResult:
    if not isinstance(value, AdapterResult):
        return AdapterResult(status="unavailable", error="adapter probe returned invalid result")
    status = getattr(value, "status", None)
    if not isinstance(status, str) or not status.strip():
        return AdapterResult(status="failed", error="adapter probe returned result without status")
    if value.error is not None and not isinstance(value.error, str):
        return AdapterResult(status="failed", error="adapter probe returned invalid error")
    try:
        result = freeze_json_value(value.result)
        details = freeze_json_object(value.details)
    except ValueError as exc:
        return AdapterResult(status="failed", error=f"adapter probe returned non-JSON data: {exc}")
    return AdapterResult(status=status, result=result, error=value.error, details=details)


@dataclass(frozen=True)
class MappingRecord:
    mapping: MappingVersion
    status: str = "draft"
    target_id: str | None = None
    probe: AdapterResult | None = None

    @property
    def version(self) -> str:
        return self.mapping.version

    @property
    def state(self) -> str:
        return self.status


class MappingRegistry:
    """Owns immutable mapping versions and active pointers per command key."""

    def __init__(
        self,
        adapters: Mapping[str | Transport, CommandAdapter] | None = None,
        *,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        allowed_targets: set[str] | None = None,
    ):
        self.adapters = {getattr(k, "value", k): value for k, value in (adapters or {}).items()}
        self.audit_sink = audit_sink
        self.allowed_targets = frozenset(allowed_targets) if allowed_targets is not None else None
        self._records: dict[tuple[str, str], MappingRecord] = {}
        self._active: dict[str, tuple[str, str]] = {}
        self._history: dict[str, list[tuple[str, str]]] = {}
        self._lock = threading.RLock()
        self._probing: set[tuple[str, str]] = set()
        self.diagnostics: list[dict[str, Any]] = []
        self._audit_queue: list[dict[str, Any]] = []
        self._audit_flush_lock = threading.Lock()
        self._audit_worker_pending = False
        # API lifecycle routes can stage audit records until their SQLite
        # mapping transaction commits.  While deferred, no background worker
        # may observe the temporary sink and publish a false transition.
        self._audit_deferred = False
        self._deferred_events: list[dict[str, Any]] | None = None

    def _audit(self, action: str, record: MappingRecord) -> None:
        self._audit_queue.append({"action": action, "mappingId": record.mapping.mappingId, "version": record.version, "commandKey": record.mapping.commandKey, "status": record.status})
        if self.audit_sink and not self._audit_deferred and not self._audit_worker_pending and self._audit_flush_lock.locked():
            self._audit_worker_pending = True
            threading.Thread(target=self._audit_worker, daemon=True).start()

    def _schedule_audit_worker(self) -> None:
        with self._lock:
            if not self.audit_sink or self._audit_worker_pending or not self._audit_queue:
                return
            self._audit_worker_pending = True
        threading.Thread(target=self._audit_worker, daemon=True).start()

    def _deliver_audits(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            try:
                self.audit_sink(event)
            except Exception as exc:
                with self._lock:
                    self.diagnostics.append({"message": "audit sink failed", "error": str(exc)})

    def _audit_worker(self) -> None:
        while True:
            with self._audit_flush_lock:
                while True:
                    with self._lock:
                        events = self._audit_queue
                        self._audit_queue = []
                    if not events:
                        break
                    self._deliver_audits(events)
            with self._lock:
                if self._audit_queue:
                    continue
                self._audit_worker_pending = False
                if self._audit_queue:
                    self._audit_worker_pending = True
                    continue
                return

    def _flush_audits(self) -> None:
        if self._audit_deferred:
            self.drain_deferred_audits()
            return
        if not self._audit_flush_lock.acquire(blocking=False):
            self._schedule_audit_worker()
            return
        try:
            while True:
                with self._lock:
                    events = self._audit_queue
                    self._audit_queue = []
                if self.audit_sink:
                    if not events:
                        break
                    self._deliver_audits(events)
                else:
                    break
        finally:
            self._audit_flush_lock.release()

    def drain_deferred_audits(self) -> None:
        """Move queued events into the active deferred transaction list."""
        with self._lock:
            if not self._audit_deferred or self._deferred_events is None:
                return
            if self._audit_queue:
                self._deferred_events.extend(self._audit_queue)
                self._audit_queue = []

    @contextmanager
    def defer_audits(self):
        """Stage registry audit events until the caller's durable commit.

        The flush lock is held while the sink is replaced, which first waits
        for any already-running worker.  ``_audit_deferred`` prevents a
        registry operation from starting another worker while the temporary
        sink is installed.  All queued events are returned to the caller and
        are therefore either committed with the mapping transaction or
        discarded together with the in-memory snapshot.
        """
        events: list[dict[str, Any]] = []
        self._audit_flush_lock.acquire()
        previous_sink: Callable[[dict[str, Any]], None] | None = None
        previous_deferred = False
        previous_events: list[dict[str, Any]] | None = None
        try:
            with self._lock:
                previous_sink = self.audit_sink
                previous_deferred = self._audit_deferred
                previous_events = self._deferred_events
                pending = self._audit_queue
                self._audit_queue = []
                self._audit_deferred = True
                self._deferred_events = events
                self.audit_sink = events.append
            # Events queued before this transaction belong to the previous
            # operation; deliver them before changing the sink's ownership.
            if pending and previous_sink is not None:
                self._deliver_audits(pending)
            elif pending:
                events.extend(pending)
            yield events
        finally:
            with self._lock:
                pending = self._audit_queue
                self._audit_queue = []
            events.extend(pending)
            with self._lock:
                self.audit_sink = previous_sink
                self._audit_deferred = previous_deferred
                self._deferred_events = previous_events
            self._audit_flush_lock.release()
            if not previous_deferred:
                self._schedule_audit_worker()

    def _key(self, mapping_id: str, version: str) -> tuple[str, str]:
        return mapping_id, version

    def create_draft(self, mapping: MappingVersion, target_id: str | None = None) -> MappingRecord:
        with self._lock:
            record = self._create_draft(mapping, target_id)
        self._flush_audits()
        return record

    def _create_draft(self, mapping: MappingVersion, target_id: str | None = None) -> MappingRecord:
        if target_id is not None:
            if not isinstance(target_id, str) or not target_id.strip():
                raise ValueError("target_id must be non-empty")
            target_id = target_id.strip()
        if target_id is not None and self.allowed_targets is not None and target_id not in self.allowed_targets:
            raise ValueError("target is not allowlisted")
        self._static_validate(mapping)
        key = self._key(mapping.mappingId, mapping.version)
        if key in self._records:
            raise ValueError("mapping version already exists")
        embedded_target = self._mapping_target(mapping)
        if target_id is not None and embedded_target is not None and target_id != embedded_target:
            raise ValueError("target_id does not match mapping target")
        if self.allowed_targets is not None:
            if embedded_target is None:
                raise ValueError("target is required when targets are allowlisted")
            if embedded_target not in self.allowed_targets:
                raise ValueError("target is not allowlisted")
        # Keep one canonical target binding on the record.  An embedded
        # target is authoritative when callers omit the explicit argument;
        # later lifecycle snapshots then retain the same binding.
        if target_id is None:
            target_id = embedded_target
        rec = MappingRecord(mapping=mapping, status="draft", target_id=target_id)
        self._records[self._key(mapping.mappingId, mapping.version)] = rec
        self._audit("draft", rec)
        return rec

    draft = create_draft

    def _require(self, mapping_id: str, version: str) -> MappingRecord:
        try:
            return self._records[self._key(mapping_id, version)]
        except KeyError as exc:
            raise ValueError("mapping version not found") from exc

    _static_validate = staticmethod(validate_mapping_safety)

    @staticmethod
    def _mapping_target(mapping: MappingVersion) -> str | None:
        targets = mapping_target_values(mapping.mapping)
        return targets[0] if targets else None

    def validate(self, mapping_id: str, version: str) -> MappingRecord:
        with self._lock:
            if (mapping_id, version) in self._probing:
                raise ValueError("mapping is currently being probed")
            record = self._validate(mapping_id, version)
        self._flush_audits()
        return record

    def _validate(self, mapping_id: str, version: str) -> MappingRecord:
        rec = self._require(mapping_id, version)
        if rec.status not in {"draft", "validated"}:
            raise ValueError("invalid mapping state for validation")
        self._static_validate(rec.mapping)
        rec = MappingRecord(rec.mapping, "validated", rec.target_id, rec.probe)
        self._records[(mapping_id, version)] = rec
        self._audit("validated", rec)
        return rec

    def probe(self, mapping_id: str, version: str) -> MappingRecord:
        return self._probe(mapping_id, version)

    def _probe(self, mapping_id: str, version: str) -> MappingRecord:
        key = (mapping_id, version)
        with self._lock:
            rec = self._require(mapping_id, version)
            if rec.status not in {"validated", "probed"}:
                raise ValueError("mapping must be validated before probing")
            if key in self._probing:
                raise ValueError("mapping is already being probed")
            self._probing.add(key)
            mapping = rec.mapping
            target_id = rec.target_id
            adapter = self.adapters.get(mapping.transport.value)
        if adapter is None:
            result = AdapterResult(status="unavailable", error=f"no adapter for {mapping.transport.value}")
        else:
            try:
                result = adapter.probe(mapping)
            except Exception as exc:
                result = AdapterResult(status="failed", error=f"adapter probe failed: {exc}")
        try:
            result = _snapshot_probe_result(result)
        except Exception as exc:
            result = AdapterResult(status="failed", error=f"adapter probe result normalization failed: {exc}")
        probe_status = result.status
        good = probe_status.lower() in {"probed", "available", "succeeded", "ok"}
        failure: ValueError | None = None
        success_record: MappingRecord | None = None
        with self._lock:
            try:
                current = self._records.get(key)
                if current is None or current.mapping != mapping or current.status not in {"validated", "probed"}:
                    failure = ValueError("mapping changed while probing")
                elif not good:
                    failed_record = MappingRecord(mapping, current.status, target_id, result)
                    self._audit("probe_failed", failed_record)
                    failure = ValueError(result.error or "capability probe failed")
                else:
                    success_record = MappingRecord(mapping, "probed", target_id, result)
                    self._records[key] = success_record
                    self._audit("probed", success_record)
            finally:
                self._probing.discard(key)
        self._flush_audits()
        if failure is not None:
            raise failure
        assert success_record is not None
        return success_record

    def activate(self, mapping_id: str, version: str) -> MappingRecord:
        with self._lock:
            if (mapping_id, version) in self._probing:
                raise ValueError("mapping is currently being probed")
            record = self._activate(mapping_id, version)
        self._flush_audits()
        return record

    def _activate(self, mapping_id: str, version: str) -> MappingRecord:
        rec = self._require(mapping_id, version)
        if rec.status != "probed":
            raise ValueError("mapping must pass capability probe before activation")
        # Pointer update is one assignment, so readers see either old or new version.
        old = self._active.get(rec.mapping.commandKey)
        self._active[rec.mapping.commandKey] = (rec.mapping.mappingId, version)
        self._history.setdefault(rec.mapping.commandKey, [])
        if old and old != (rec.mapping.mappingId, version) and old not in self._history[rec.mapping.commandKey]:
            self._history[rec.mapping.commandKey].append(old)
            old_record = self._records.get(old)
            if old_record is not None:
                self._records[old] = MappingRecord(old_record.mapping, "superseded", old_record.target_id, old_record.probe)
        rec = MappingRecord(rec.mapping, "active", rec.target_id, rec.probe)
        self._records[(mapping_id, version)] = rec
        self._audit("activated", rec)
        return rec

    def active(self, command_key: str) -> MappingVersion:
        with self._lock:
            pointer = self._active.get(command_key)
            if pointer is None:
                raise ValueError("no active mapping for command")
            record = self._records.get(pointer)
        if record is None:
            raise ValueError("active mapping not found")
        return record.mapping

    get_active = active

    def rollback(self, command_key: str, version: str | None = None) -> MappingRecord:
        with self._lock:
            history = self._history.get(command_key, [])
            target = next((item for item in reversed(history) if item[1] == version), None) if version else (history[-1] if history else None)
            if target is None:
                raise ValueError("no previous mapping to rollback")
            rec = self._records.get(target)
            if not rec:
                raise ValueError("rollback mapping not found")
            current = self._active.get(command_key)
            self._active[command_key] = target
            self._history[command_key] = [item for item in history if item != target] + ([current] if current else [])
            if current and current != target:
                current_record = self._records.get(current)
                if current_record is not None:
                    self._records[current] = MappingRecord(current_record.mapping, "superseded", current_record.target_id, current_record.probe)
            rec = MappingRecord(rec.mapping, "active", rec.target_id, rec.probe)
            self._records[target] = rec
            self._audit("rollback", rec)
        self._flush_audits()
        return rec

    def get(self, mapping_id: str, version: str) -> MappingRecord:
        with self._lock:
            return self._require(mapping_id, version)

    def restore_record(self, record: MappingRecord) -> None:
        """Hydrate a persisted snapshot without probing or emitting audits."""
        with self._lock:
            key = self._key(record.mapping.mappingId, record.mapping.version)
            if key in self._records:
                raise ValueError("mapping version already exists")
            self._records[key] = record

    def restore_active_pointer(self, command_key: str, mapping_id: str, version: str) -> None:
        """Hydrate the durable active pointer after all snapshots are loaded."""
        with self._lock:
            record = self._require(mapping_id, version)
            self._active[command_key] = (mapping_id, version)
            self._history.setdefault(command_key, [])
            if record.status != "active":
                self._records[(mapping_id, version)] = MappingRecord(
                    record.mapping, "active", record.target_id, record.probe
                )

    def restore_history(self, command_key: str, history: list[tuple[str, str]]) -> None:
        with self._lock:
            self._history[command_key] = list(history)

    def snapshot_state(self) -> tuple[dict[tuple[str, str], MappingRecord], dict[str, tuple[str, str]], dict[str, list[tuple[str, str]]], list[dict[str, Any]]]:
        """Capture mutable registry indexes for an API transaction rollback."""
        with self._lock:
            return (
                dict(self._records),
                dict(self._active),
                {key: list(value) for key, value in self._history.items()},
                list(self._audit_queue),
            )

    def restore_state(self, snapshot) -> None:
        with self._lock:
            records, active, history, audit_queue = snapshot
            self._records = dict(records)
            self._active = dict(active)
            self._history = {key: list(value) for key, value in history.items()}
            self._audit_queue = list(audit_queue)


__all__ = ["MappingRecord", "MappingRegistry", "mapping_target_values", "validate_mapping_safety"]
