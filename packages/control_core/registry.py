"""Versioned command mapping registry with explicit validation and activation."""

from __future__ import annotations

from dataclasses import dataclass
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
        self._active: dict[str, str] = {}
        self._history: dict[str, list[str]] = {}

    def _audit(self, action: str, record: MappingRecord) -> None:
        if self.audit_sink:
            self.audit_sink({"action": action, "mappingId": record.mapping.mappingId, "version": record.version, "commandKey": record.mapping.commandKey, "status": record.status})

    def _key(self, mapping_id: str, version: str) -> tuple[str, str]:
        return mapping_id, version

    def create_draft(self, mapping: MappingVersion, target_id: str | None = None) -> MappingRecord:
        if target_id is not None and self.allowed_targets is not None and target_id not in self.allowed_targets:
            raise ValueError("target is not allowlisted")
        self._static_validate(mapping)
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
        banned = {"url", "endpoint", "script", "command", "exec", "shell"}
        def walk(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if str(key).lower() in banned:
                        raise ValueError(f"mapping field {key} is not allowed")
                    walk(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    walk(item)
            elif isinstance(value, str) and urlparse(value).scheme in {"http", "https", "file", "ftp", "javascript"}:
                raise ValueError("arbitrary URL access is not allowed")
        walk(mapping.mapping)

    def validate(self, mapping_id: str, version: str) -> MappingRecord:
        rec = self._require(mapping_id, version)
        self._static_validate(rec.mapping)
        rec = MappingRecord(rec.mapping, "validated", rec.target_id, rec.probe)
        self._records[(mapping_id, version)] = rec
        self._audit("validated", rec)
        return rec

    def probe(self, mapping_id: str, version: str) -> MappingRecord:
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
        rec = self._require(mapping_id, version)
        if rec.status != "probed":
            raise ValueError("mapping must pass capability probe before activation")
        # Pointer update is one assignment, so readers see either old or new version.
        old = self._active.get(rec.mapping.commandKey)
        self._active[rec.mapping.commandKey] = version
        self._history.setdefault(rec.mapping.commandKey, [])
        if old and old != version and old not in self._history[rec.mapping.commandKey]:
            self._history[rec.mapping.commandKey].append(old)
        rec = MappingRecord(rec.mapping, "active", rec.target_id, rec.probe)
        self._records[(mapping_id, version)] = rec
        self._audit("activated", rec)
        return rec

    def active(self, command_key: str) -> MappingVersion:
        version = self._active.get(command_key)
        if version is None:
            raise ValueError("no active mapping for command")
        candidates = [r.mapping for r in self._records.values() if r.mapping.commandKey == command_key and r.mapping.version == version]
        if not candidates:
            raise ValueError("active mapping not found")
        return candidates[0]

    get_active = active

    def rollback(self, command_key: str, version: str | None = None) -> MappingRecord:
        history = self._history.get(command_key, [])
        target = version or (history[-1] if history else None)
        if target is None:
            raise ValueError("no previous mapping to rollback")
        records = [r for r in self._records.values() if r.mapping.commandKey == command_key and r.mapping.version == target]
        if not records:
            raise ValueError("rollback mapping not found")
        rec = records[0]
        self._active[command_key] = target
        self._audit("rollback", rec)
        return rec

    def get(self, mapping_id: str, version: str) -> MappingRecord:
        return self._require(mapping_id, version)


__all__ = ["MappingRecord", "MappingRegistry"]
