"""Versioned command mapping registry with explicit validation and activation."""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
from typing import Any, Callable, Mapping
from urllib.parse import urlparse

from .adapters import AdapterResult, CommandAdapter
from .models import MappingVersion, Transport


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
        self.allowed_targets = allowed_targets
        self._records: dict[tuple[str, str], MappingRecord] = {}
        self._active: dict[str, tuple[str, str]] = {}
        self._history: dict[str, list[tuple[str, str]]] = {}
        self._lock = threading.RLock()

    def _audit(self, action: str, record: MappingRecord) -> None:
        if self.audit_sink:
            self.audit_sink({"action": action, "mappingId": record.mapping.mappingId, "version": record.version, "commandKey": record.mapping.commandKey, "status": record.status})

    def _key(self, mapping_id: str, version: str) -> tuple[str, str]:
        return mapping_id, version

    def create_draft(self, mapping: MappingVersion, target_id: str | None = None) -> MappingRecord:
        with self._lock:
            return self._create_draft(mapping, target_id)

    def _create_draft(self, mapping: MappingVersion, target_id: str | None = None) -> MappingRecord:
        if target_id is not None and not str(target_id).strip():
            raise ValueError("target_id must be non-empty")
        if target_id is not None and self.allowed_targets is not None and target_id not in self.allowed_targets:
            raise ValueError("target is not allowlisted")
        self._static_validate(mapping)
        embedded_target = self._mapping_target(mapping)
        if self.allowed_targets is not None:
            if embedded_target is None:
                raise ValueError("target is required when targets are allowlisted")
            if embedded_target not in self.allowed_targets:
                raise ValueError("target is not allowlisted")
            if target_id is not None and target_id != embedded_target:
                raise ValueError("target_id does not match mapping target")
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

    @staticmethod
    def _static_validate(mapping: MappingVersion) -> None:
        if mapping.transport not in set(Transport):
            raise ValueError("transport is not allowlisted")
        banned_tokens = ("script", "endpoint", "exec", "shell", "url")
        def walk(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                    if any(token in normalized for token in banned_tokens) or normalized in {"command", "commandline", "run"}:
                        raise ValueError(f"mapping field {key} is not allowed")
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            elif isinstance(value, str) and (urlparse(value).scheme in {"http", "https", "file", "ftp", "javascript", "data", "ws", "wss"} or value.startswith(("//", "\\\\")) or "://" in value):
                raise ValueError("arbitrary URL access is not allowed")
        walk(mapping.mapping)

    @staticmethod
    def _mapping_target(mapping: MappingVersion) -> str | None:
        for key, value in mapping.mapping.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in {"targetid", "target"}:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("target must be a non-empty string")
                return value.strip()
        return None

    def validate(self, mapping_id: str, version: str) -> MappingRecord:
        with self._lock:
            return self._validate(mapping_id, version)

    def _validate(self, mapping_id: str, version: str) -> MappingRecord:
        rec = self._require(mapping_id, version)
        self._static_validate(rec.mapping)
        rec = MappingRecord(rec.mapping, "validated", rec.target_id, rec.probe)
        self._records[(mapping_id, version)] = rec
        self._audit("validated", rec)
        return rec

    def probe(self, mapping_id: str, version: str) -> MappingRecord:
        with self._lock:
            return self._probe(mapping_id, version)

    def _probe(self, mapping_id: str, version: str) -> MappingRecord:
        rec = self._require(mapping_id, version)
        if rec.status not in {"validated", "probed"}:
            raise ValueError("mapping must be validated before probing")
        adapter = self.adapters.get(rec.mapping.transport.value)
        if adapter is None:
            result = AdapterResult(status="unavailable", error=f"no adapter for {rec.mapping.transport.value}")
        else:
            result = adapter.probe(rec.mapping)
        if result.status not in {"probed", "available", "succeeded", "ok"}:
            raise ValueError(result.error or "capability probe failed")
        rec = MappingRecord(rec.mapping, "probed", rec.target_id, result)
        self._records[(mapping_id, version)] = rec
        self._audit("probed", rec)
        return rec

    def activate(self, mapping_id: str, version: str) -> MappingRecord:
        with self._lock:
            return self._activate(mapping_id, version)

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
            target = next((item for item in history if item[1] == version), None) if version else (history[-1] if history else None)
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
            return rec

    def get(self, mapping_id: str, version: str) -> MappingRecord:
        with self._lock:
            return self._require(mapping_id, version)


__all__ = ["MappingRecord", "MappingRegistry"]
