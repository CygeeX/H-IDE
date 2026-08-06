"""FastAPI boundary for the robot action control plane.

The HTTP layer is intentionally thin: it validates/constructs immutable
control-core contracts, resolves a pinned mapping, and sends every execution
through :class:`ActionRuntime`.  SQLite is only a repository and event sink;
it is never used as a second command execution path.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from hmac import compare_digest
import math
import os
from pathlib import Path
import threading
import uuid
from typing import Any, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from packages.control_core.adapters import AdapterResult, CommandAdapter, FakeAdapter
from packages.control_core.models import (
    CapabilitySnapshot,
    CommandDefinition,
    CommandKind,
    CommandReceipt,
    CommandRequest,
    CommandSource,
    MappingVersion,
    PlanDraft,
    RobotState,
    RobotStatus,
    SafetyMode,
    Transport,
    utc_now,
)
from packages.control_core.registry import MappingRecord, MappingRegistry, mapping_target_values
from packages.control_core.runtime import ActionRuntime

from .store import SQLiteStore, _jsonable


def _plain(value: Any) -> Any:
    """Return JSON-safe camelCase data for API responses."""
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", by_alias=False)
        if "schema_" in payload and "schema" not in payload:
            payload["schema"] = payload.pop("schema_")
        return payload
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return _jsonable(value)


def _receipt_json(receipt: CommandReceipt) -> dict[str, Any]:
    return dict(_plain(receipt))


def _mapping_json(record: MappingRecord) -> dict[str, Any]:
    mapping = _plain(record.mapping)
    probe = None if record.probe is None else _plain(record.probe)
    return {
        "mapping": mapping,
        "mappingId": record.mapping.mappingId,
        "version": record.version,
        "commandKey": record.mapping.commandKey,
        "status": record.status,
        "targetId": record.target_id,
        "probe": probe,
    }


def _parse_bearer(value: str | None, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    scheme, separator, token = value.partition(" ")
    return bool(separator and scheme.lower() == "bearer" and token and compare_digest(token, expected))


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _CommandIdentityConflict(ValueError):
    """A command id is already owned by another idempotency key."""


def _embedded_mapping_target(mapping: MappingVersion) -> str | None:
    targets = mapping_target_values(mapping.mapping)
    return targets[0] if targets else None


def _mapping_target_binding(record: MappingRecord) -> str | None:
    """Return the canonical target bound to a mapping record.

    A mapping may carry its target inside the transport payload or in the
    registry's explicit ``target_id`` column.  Treat either representation as
    a binding at the API boundary, and reject a corrupt persisted record where
    the two disagree.
    """

    embedded = _embedded_mapping_target(record.mapping)
    explicit = record.target_id
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("mapping target must be a non-empty string")
        explicit = explicit.strip()
        if embedded is not None and embedded != explicit:
            raise ValueError("mapping target aliases conflict")
        return explicit
    return embedded


def _payload_expired(payload: Mapping[str, Any] | Any) -> bool:
    """Check a persisted plan expiry without relying on creation-time validation."""
    if not isinstance(payload, Mapping):
        return False
    raw = payload.get("expiresAt", payload.get("expires_at"))
    if raw is None:
        return False
    if isinstance(raw, datetime):
        value = raw
    elif isinstance(raw, str):
        try:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return False
    else:
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc) <= utc_now()


def _motion_safety_error(robot: Mapping[str, Any], definition: CommandDefinition) -> str | None:
    """Reject motion at the API boundary when the persisted robot is unsafe."""
    if definition.kind is not CommandKind.MOTION:
        return None
    status = str(robot.get("status", "unknown")).lower()
    safety_mode = str(robot.get("safetyMode", robot.get("safety_mode", "unknown"))).lower()
    if robot.get("connected") is False:
        return "robot is disconnected"
    if status in {"offline", "estop"}:
        return f"robot status {status} does not allow motion"
    if safety_mode in {"stopped", "estop", "unknown"}:
        return f"robot safety mode {safety_mode} does not allow motion"
    return None


class _ControlContext:
    def __init__(self, store: SQLiteStore, token: str, adapters: Mapping[str | Transport, CommandAdapter]):
        self.store = store
        self.token = token
        normalized = {getattr(key, "value", key): value for key, value in adapters.items()}
        self.adapters = normalized
        self.definitions: dict[tuple[str, str], CommandDefinition] = {}
        self.registry = MappingRegistry(adapters=normalized, audit_sink=store.record_audit)
        self.runtime = ActionRuntime(adapters=normalized, event_sink=self._runtime_event)

    def _runtime_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Persist events and terminal runtime receipts (including watchdog expiry)."""
        persisted = self.store.record_event(event)
        if str(event.get("status", "")).lower() in {"succeeded", "failed", "cancelled", "expired", "rejected"}:
            command_id = event.get("commandId")
            request = None
            receipt = None
            with self.runtime._lock:
                if isinstance(command_id, str):
                    request = self.runtime._requests.get(command_id)
                    receipt = self.runtime._receipts.get(command_id)
            if request is not None and receipt is not None:
                try:
                    self.store.save_command(request, receipt)
                except Exception as exc:
                    # Event durability remains useful even if a receipt write
                    # is temporarily unavailable; the next reconciliation can
                    # inspect the runtime event and retry persistence.
                    self.runtime.diagnostics.append({"message": "terminal receipt persistence failed", "error": str(exc)})
        return persisted

    def _seed_default_mapping(self, definition: CommandDefinition, robot_id: str) -> None:
        mapping = MappingVersion(
            mappingId=f"seed-{definition.commandKey.replace('.', '-')}",
            version="v1",
            commandKey=definition.commandKey,
            transport=Transport.FAKE,
            mapping={"targetId": robot_id, "topic": f"/{definition.commandKey}"},
        )
        try:
            record = self.registry.create_draft(mapping)
            self.store.save_mapping(record)
            record = self.registry.validate(mapping.mappingId, mapping.version)
            self.store.save_mapping(record)
            record = self.registry.probe(mapping.mappingId, mapping.version)
            self.store.save_mapping(record)
            record = self.registry.activate(mapping.mappingId, mapping.version)
            self.store.save_mapping(record)
        except ValueError:
            # A custom adapter may intentionally be unavailable.  The mapping
            # remains absent and the API reports a controlled error.
            return

    def seed(self) -> None:
        # A process restart cannot know whether a robot accepted each step of
        # an in-flight confirmation.  Mark it unknown; never silently replay.
        self.store.recover_confirming_drafts()
        self.store.mark_unowned_commands_for_reconciliation()
        robot = RobotState(
            robotId="robot-1",
            status=RobotStatus.ONLINE,
            connected=True,
            safetyMode=SafetyMode.NORMAL,
            details={"adapter": "fake"},
        )
        capability = CapabilitySnapshot(
            agent="fake-agent",
            protocol="fake",
            robotId=robot.robotId,
            capabilities=["move.forward", "stop", "gripper.open"],
            safetyMode=SafetyMode.NORMAL,
            schema={"type": "object"},
        )
        # Seed only missing records.  A restart must not overwrite a robot's
        # persisted ESTOP/offline/degraded state or its last capability view.
        if self.store.get_robot(robot.robotId) is None:
            self.store.upsert_robot(robot)
        if self.store.get_capabilities(robot.robotId) is None:
            self.store.upsert_capabilities(capability, robot.robotId)

        definitions = (
            CommandDefinition(
                commandKey="move.forward",
                commandVersion="1",
                transports=[Transport.FAKE],
                kind=CommandKind.MOTION,
                payloadSchema={"type": "object"},
            ),
            CommandDefinition(
                commandKey="stop",
                commandVersion="1",
                transports=[Transport.FAKE],
                kind=CommandKind.SAFETY,
                payloadSchema={"type": "object"},
            ),
            CommandDefinition(
                commandKey="gripper.open",
                commandVersion="1",
                transports=[Transport.FAKE],
                kind=CommandKind.NON_MOTION,
                payloadSchema={"type": "object"},
            ),
        )
        for definition in definitions:
            self.definitions[(definition.commandKey, definition.commandVersion)] = definition

        # Restore persisted mappings when possible.  A fresh in-memory app has
        # no rows and receives the same deterministic seed mappings.
        persisted = self.store.list_mappings()
        restored_rows: list[Mapping[str, Any]] = []
        if persisted:
            for row in persisted:
                if self._restore_mapping(row):
                    restored_rows.append(row)
            # The active pointer is the atomic source of truth.  Historical
            # superseded rows are retained solely for explicit rollback.
            grouped: dict[str, list[Mapping[str, Any]]] = {}
            for row in restored_rows:
                if not isinstance(row, Mapping):
                    continue
                mapping_payload = row.get("mapping")
                if not isinstance(mapping_payload, Mapping):
                    continue
                command_key = mapping_payload.get("commandKey") or mapping_payload.get("command_key")
                if isinstance(command_key, str):
                    grouped.setdefault(command_key, []).append(row)
            pointers = self.store.list_active_pointers()
            for command_key, rows in grouped.items():
                pointer = pointers.get(command_key)
                if pointer is None:
                    active_rows = [row for row in rows if row.get("status") == "active"]
                    if active_rows:
                        active_rows.sort(key=lambda row: row.get("updatedAt", ""))
                        pointer = (
                            active_rows[-1]["mapping"].get("mappingId"),
                            active_rows[-1]["mapping"].get("version"),
                        )
                if pointer and all(isinstance(item, str) and item for item in pointer):
                    try:
                        self.registry.restore_active_pointer(command_key, pointer[0], pointer[1])
                    except ValueError:
                        continue
                history = []
                for row in sorted(rows, key=lambda row: row.get("updatedAt", "")):
                    if row.get("status") == "superseded":
                        mapping_payload = row["mapping"]
                        if not isinstance(mapping_payload, Mapping):
                            continue
                        history.append((mapping_payload.get("mappingId"), mapping_payload.get("version")))
                history = [item for item in history if all(isinstance(part, str) and part for part in item)]
                if history:
                    self.registry.restore_history(command_key, history)
        # Fill missing command definitions even when a database contains only
        # corrupt/draft snapshots.  Existing valid active mappings are never
        # replaced.
        for definition in definitions:
            try:
                self.registry.active(definition.commandKey)
            except ValueError:
                self._seed_default_mapping(definition, robot.robotId)

    def _restore_mapping(self, row: Mapping[str, Any]) -> bool:
        # Rebuilding the in-memory registry must not write the replayed
        # snapshots back to the repository.  Doing so would turn historical
        # ``superseded`` rows into ``active`` rows on every restart and would
        # also duplicate lifecycle audit records.  The persisted transition
        # order is already the source of truth; registry operations below are
        # only an in-memory replay.
        if not isinstance(row, Mapping):
            return False
        mapping_payload = row.get("mapping")
        if not isinstance(mapping_payload, Mapping):
            return False
        try:
            mapping = MappingVersion.model_validate(mapping_payload)
            target_id = row.get("targetId")
            status = str(row.get("status", "draft"))
            probe_payload = row.get("probe")
            probe = None
            if isinstance(probe_payload, Mapping):
                try:
                    probe = AdapterResult(**dict(probe_payload))
                except (TypeError, ValueError):
                    probe = AdapterResult(status="failed", error="persisted probe is invalid")
            record = MappingRecord(mapping=mapping, status=status, target_id=target_id, probe=probe)
            self.registry.restore_record(record)
        except (ValueError, ValidationError):
            # Ignore stale/corrupt rows during startup; the repository remains
            # readable and callers can submit a new validated mapping.
            return False
        return True

    def definition(self, command_key: str, version: str | None = None) -> CommandDefinition:
        requested = version or "1"
        definition = self.definitions.get((command_key, requested))
        if definition is None:
            raise HTTPException(status_code=404, detail="command definition not found")
        return definition

    def mapping_record(self, command_key: str, mapping_id: str | None = None, version: str | None = None) -> MappingRecord:
        try:
            if mapping_id is not None:
                record = self.registry.get(mapping_id, version or "v1")
                if record.status != "active":
                    raise HTTPException(status_code=409, detail="mapping version is not active")
                return record
            mapping = self.registry.active(command_key)
            record = self.registry.get(mapping.mappingId, mapping.version)
            if record.status != "active":
                raise HTTPException(status_code=409, detail="mapping version is not active")
            return record
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def mapping(self, command_key: str, mapping_id: str | None = None, version: str | None = None) -> MappingVersion:
        return self.mapping_record(command_key, mapping_id, version).mapping


def create_app(
    db_path: str = ":memory:",
    auth_token: str | None = None,
    adapters: Mapping[str | Transport, CommandAdapter] | None = None,
    watchdog_interval: float = 0.1,
) -> FastAPI:
    """Create an isolated API instance suitable for tests or deployment."""

    token = auth_token or os.getenv("CONTROL_API_TOKEN", "dev-token")
    adapter_map: dict[str | Transport, CommandAdapter] = dict(adapters or {})
    adapter_map.setdefault("fake", FakeAdapter())
    store = SQLiteStore(db_path)
    context = _ControlContext(store, token, adapter_map)
    context.seed()

    # The control surface is authenticated; exposing an unauthenticated schema
    # would disclose robot capabilities and command contracts.  Operators can
    # publish a separately authenticated schema if needed.
    app = FastAPI(
        title="Robot Action Control API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.control = context
    app.state.store = store
    app.state.runtime = context.runtime
    app.state.registry = context.registry
    app.state.definitions = context.definitions

    try:
        watchdog_seconds = float(watchdog_interval)
    except (TypeError, ValueError, OverflowError):
        watchdog_seconds = 0.1
    if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
        watchdog_seconds = 0.1
    watchdog_stop = threading.Event()

    def watchdog() -> None:
        while not watchdog_stop.wait(watchdog_seconds):
            try:
                context.runtime.tick()
            except Exception as exc:
                context.runtime.diagnostics.append({"message": "runtime watchdog failed", "error": str(exc)})

    watchdog_thread = threading.Thread(target=watchdog, name="control-runtime-watchdog", daemon=True)
    watchdog_thread.start()

    def stop_watchdog() -> None:
        watchdog_stop.set()
        if threading.current_thread() is not watchdog_thread:
            watchdog_thread.join(timeout=max(0.5, watchdog_seconds * 4))

    app.state.stop_watchdog = stop_watchdog
    app.router.add_event_handler("shutdown", stop_watchdog)

    def require_auth(request: Request) -> None:
        if not _parse_bearer(request.headers.get("authorization"), token):
            raise HTTPException(status_code=401, detail="authentication required", headers={"WWW-Authenticate": "Bearer"})

    def get_context(request: Request) -> _ControlContext:
        return request.app.state.control

    # Registry state and the SQLite active pointer are committed as one
    # logical transition.  Serialize lifecycle mutations within this process
    # so a concurrent activate/rollback cannot persist a stale ``previous``
    # snapshot after a newer transition.
    mapping_mutation_lock = threading.RLock()

    def fail_validation(exc: ValidationError) -> None:
        # Pydantic's rich error entries may contain raw datetimes/exception
        # objects in ``input``/``ctx``.  Keep the HTTP contract JSON-safe and
        # retain the actionable validation text.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _build_request(
        body: Mapping[str, Any],
        definition: CommandDefinition,
        mapping: MappingVersion,
        *,
        source: CommandSource = CommandSource.MANUAL,
        command_id: str | None = None,
        idempotency_prefix: str = "api",
    ) -> CommandRequest:
        def pick(*names: str, default: Any = None) -> Any:
            for name in names:
                if name in body:
                    return body[name]
            return default

        now = utc_now()
        expires = pick("expiresAt", "expires_at", default=now + timedelta(seconds=30))
        data = {
            "commandId": command_id or pick("commandId", "command_id", default=_new_id("cmd")),
            "robotId": pick("robotId", "robot_id"),
            "source": pick("source", default=source.value),
            "transport": pick("transport", default=mapping.transport.value),
            "commandKey": pick("commandKey", "command_key", default=definition.commandKey),
            "commandVersion": pick("commandVersion", "command_version", "version", default=definition.commandVersion),
            "payload": pick("payload", default={}),
            "issuedAt": pick("issuedAt", "issued_at", default=now),
            "expiresAt": expires,
            "idempotencyKey": pick("idempotencyKey", "idempotency_key", default=_new_id(idempotency_prefix)),
            "sessionId": pick("sessionId", "session_id", default="api-session"),
            "traceId": pick("traceId", "trace_id", default=_new_id("trace")),
        }
        try:
            return CommandRequest.model_validate(data)
        except ValidationError as exc:
            fail_validation(exc)
            raise AssertionError("unreachable")

    def _dispatch_and_save(context: _ControlContext, request: CommandRequest, definition: CommandDefinition, mapping: MappingVersion) -> CommandReceipt:
        claim = context.store.claim_command(request)
        if not claim.owned:
            if claim.conflict:
                raise _CommandIdentityConflict("commandId is already bound to another idempotencyKey")
            if not claim.record:
                raise RuntimeError("idempotency claim lost without a durable record")
            try:
                return CommandReceipt.model_validate(claim.record["receipt"])
            except ValidationError as exc:
                raise RuntimeError("persisted command receipt is invalid") from exc
        try:
            receipt = context.runtime.dispatch(request, definition, mapping)
        except Exception as exc:
            # A claimed row must become terminal even if an unexpected runtime
            # exception crosses the API boundary; otherwise retries could
            # dispatch an unknown command after a restart.
            receipt = CommandReceipt(
                commandId=request.commandId,
                robotId=request.robotId,
                status="failed",
                error=f"runtime dispatch failed: {exc}",
                completedAt=utc_now(),
            )
        persisted_request = context.runtime._requests.get(request.commandId, request)
        context.store.save_command(persisted_request, receipt)
        return receipt

    def _save_mapping_change(context: _ControlContext, record: MappingRecord, previous: MappingVersion | None = None) -> None:
        records = [record]
        if previous is not None and (previous.mappingId, previous.version) != (record.mapping.mappingId, record.version):
            try:
                records.insert(0, context.registry.get(previous.mappingId, previous.version))
            except ValueError:
                pass
        active = record.mapping if record.status == "active" else None
        context.store.save_mapping_batch(records, active=active)

    @app.get("/api/v1/robots", dependencies=[Depends(require_auth)])
    def list_robots(context: _ControlContext = Depends(get_context)):
        return context.store.list_robots()

    @app.get("/api/v1/robots/{robot_id}/capabilities", dependencies=[Depends(require_auth)])
    def get_capabilities(robot_id: str, context: _ControlContext = Depends(get_context)):
        value = context.store.get_capabilities(robot_id)
        if value is None:
            raise HTTPException(status_code=404, detail="robot capabilities not found")
        return value

    @app.post("/api/v1/commands", dependencies=[Depends(require_auth)])
    def submit_command(body: dict[str, Any], context: _ControlContext = Depends(get_context)):
        requested_source = body.get("source")
        if requested_source is not None and str(requested_source).lower() != CommandSource.MANUAL.value:
            raise HTTPException(status_code=422, detail="manual endpoint requires source=manual")
        # Resolve durable identity before looking up a robot, definition, or
        # mapping.  A retry must return its original receipt even after a
        # deployment changed or removed the mapping used by that command.
        supplied_key = body.get("idempotencyKey", body.get("idempotency_key"))
        if isinstance(supplied_key, str) and supplied_key:
            existing = context.store.get_command_by_idempotency(supplied_key)
            if existing is not None:
                return existing["receipt"]
        supplied_command_id = body.get("commandId", body.get("command_id"))
        if isinstance(supplied_command_id, str) and supplied_command_id:
            existing = context.store.get_command_record(supplied_command_id)
            if existing is not None:
                existing_key = existing.get("idempotencyKey")
                if existing_key == supplied_key:
                    return existing["receipt"]
                raise HTTPException(status_code=409, detail="commandId is already bound to another idempotencyKey")
        robot_id = body.get("robotId", body.get("robot_id"))
        robot = context.store.get_robot(robot_id) if isinstance(robot_id, str) else None
        if not isinstance(robot_id, str) or robot is None:
            raise HTTPException(status_code=404, detail="robot not found")
        command_key = body.get("commandKey", body.get("command_key"))
        if not isinstance(command_key, str) or not command_key.strip():
            raise HTTPException(status_code=422, detail="commandKey is required")
        version = body.get("commandVersion", body.get("command_version", body.get("version", "1")))
        definition = context.definition(command_key, version)
        safety_error = _motion_safety_error(robot, definition)
        if safety_error:
            raise HTTPException(status_code=409, detail=safety_error)
        mapping_record = context.mapping_record(
            command_key,
            body.get("mappingId", body.get("mapping_id")),
            body.get("mappingVersion", body.get("mapping_version")),
        )
        mapping = mapping_record.mapping
        request = _build_request(body, definition, mapping)
        if request.commandKey != mapping.commandKey or request.commandVersion != definition.commandVersion or request.transport != mapping.transport:
            raise HTTPException(status_code=422, detail="command, version, transport, and mapping do not match")
        try:
            target = _mapping_target_binding(mapping_record)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if target is not None and target != request.robotId:
            raise HTTPException(status_code=422, detail="mapping target does not match robotId")
        try:
            receipt = _dispatch_and_save(context, request, definition, mapping)
        except _CommandIdentityConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _receipt_json(receipt)

    @app.get("/api/v1/commands", dependencies=[Depends(require_auth)])
    def list_commands(context: _ControlContext = Depends(get_context)):
        return context.store.list_commands()

    @app.get("/api/v1/commands/{command_id}", dependencies=[Depends(require_auth)])
    def get_command(command_id: str, context: _ControlContext = Depends(get_context)):
        receipt = context.store.get_command(command_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail="command not found")
        return receipt

    @app.post("/api/v1/commands/{command_id}/cancel", dependencies=[Depends(require_auth)])
    def cancel_command(command_id: str, context: _ControlContext = Depends(get_context)):
        persisted_record = context.store.get_command_record(command_id)
        if persisted_record is None:
            raise HTTPException(status_code=404, detail="command not found")
        persisted_receipt = persisted_record["receipt"]
        persisted_status = str(persisted_receipt.get("status", "")).lower()
        if persisted_status in {"succeeded", "failed", "cancelled", "canceled", "expired", "rejected"}:
            return persisted_receipt
        # Runtime state is intentionally in-memory.  A restarted process must
        # not pretend it can cancel an adapter operation it does not own.
        if command_id not in context.runtime._requests or command_id not in context.runtime._receipts:
            raise HTTPException(
                status_code=409,
                detail="command is persisted but runtime ownership is unavailable after restart; reconciliation required",
            )
        try:
            receipt = context.runtime.cancel(command_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        record = context.runtime._requests.get(command_id)
        if record is not None:
            context.store.save_command(record, receipt)
        return _receipt_json(receipt)

    @app.post("/api/v1/mappings/drafts", dependencies=[Depends(require_auth)])
    def create_mapping_draft(body: dict[str, Any], context: _ControlContext = Depends(get_context)):
        try:
            mapping_data = dict(body)
            explicit_targets = [mapping_data.pop(name) for name in ("targetId", "target_id") if name in mapping_data]
            if len({str(value).strip() for value in explicit_targets}) > 1:
                raise ValueError("target aliases conflict")
            target_id = explicit_targets[0] if explicit_targets else None
            mapping = MappingVersion.model_validate(mapping_data)
            record = context.registry.create_draft(mapping, target_id=target_id)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        _save_mapping_change(context, record)
        return _mapping_json(record)

    @app.get("/api/v1/mappings", dependencies=[Depends(require_auth)])
    def list_mappings(context: _ControlContext = Depends(get_context)):
        return context.store.list_mappings()

    @app.get("/api/v1/mappings/active/{command_key}", dependencies=[Depends(require_auth)])
    def get_active_mapping(command_key: str, context: _ControlContext = Depends(get_context)):
        return _plain(context.mapping(command_key))

    def lifecycle_action(action: str):
        def route(mapping_id: str, version: str, context: _ControlContext = Depends(get_context)):
            with mapping_mutation_lock:
                snapshot = context.registry.snapshot_state()
                try:
                    previous = None
                    if action == "activate":
                        current = context.registry.get(mapping_id, version).mapping
                        try:
                            previous = context.registry.active(current.commandKey)
                        except ValueError:
                            previous = None
                    method = getattr(context.registry, action)
                    record = method(mapping_id, version)
                    _save_mapping_change(context, record, previous)
                    return _mapping_json(record)
                except ValueError as exc:
                    context.registry.restore_state(snapshot)
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                except Exception as exc:
                    # Do not report a successful lifecycle transition when the
                    # durable batch failed; restore in-memory indexes to the
                    # last committed snapshot and require retry/reconciliation.
                    context.registry.restore_state(snapshot)
                    raise HTTPException(status_code=503, detail="mapping persistence failed; transition was not committed") from exc

        return route

    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/validate", lifecycle_action("validate"), methods=["POST"], dependencies=[Depends(require_auth)])
    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/probe", lifecycle_action("probe"), methods=["POST"], dependencies=[Depends(require_auth)])
    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/activate", lifecycle_action("activate"), methods=["POST"], dependencies=[Depends(require_auth)])

    @app.post("/api/v1/mappings/{command_key}/rollback", dependencies=[Depends(require_auth)])
    def rollback_mapping(command_key: str, body: dict[str, Any] | None = None, context: _ControlContext = Depends(get_context)):
        version = None if not body else body.get("version")
        with mapping_mutation_lock:
            snapshot = context.registry.snapshot_state()
            try:
                try:
                    previous = context.registry.active(command_key)
                except ValueError:
                    previous = None
                record = context.registry.rollback(command_key, version=version)
                _save_mapping_change(context, record, previous)
                return _mapping_json(record)
            except ValueError as exc:
                context.registry.restore_state(snapshot)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                context.registry.restore_state(snapshot)
                raise HTTPException(status_code=503, detail="mapping persistence failed; transition was not committed") from exc

    @app.post("/api/v1/assistant/drafts", dependencies=[Depends(require_auth)])
    def create_assistant_draft(body: dict[str, Any], context: _ControlContext = Depends(get_context)):
        draft_id = body.get("draftId", body.get("draft_id")) or body.get("planId", body.get("plan_id")) or _new_id("draft")
        plan_data = dict(body)
        plan_data.pop("draftId", None)
        plan_data.pop("draft_id", None)
        plan_data.setdefault("planId", draft_id)
        try:
            plan = PlanDraft.model_validate(plan_data)
        except ValidationError as exc:
            fail_validation(exc)
            raise AssertionError("unreachable")
        if not context.store.save_assistant_draft(draft_id, _plain(plan), status="draft"):
            raise HTTPException(status_code=409, detail="assistant draft already exists")
        return {"draftId": draft_id, "status": "draft", "plan": _plain(plan)}

    @app.get("/api/v1/assistant/drafts/{draft_id}", dependencies=[Depends(require_auth)])
    def get_assistant_draft(draft_id: str, context: _ControlContext = Depends(get_context)):
        draft = context.store.get_assistant_draft(draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="assistant draft not found")
        return draft

    @app.post("/api/v1/assistant/drafts/{draft_id}/confirm", dependencies=[Depends(require_auth)])
    def confirm_assistant_draft(draft_id: str, body: dict[str, Any] | None = None, context: _ControlContext = Depends(get_context)):
        if body and body.get("confirmed") is False:
            raise HTTPException(status_code=400, detail="confirmation is required")
        stored = context.store.get_assistant_draft(draft_id)
        if stored is None:
            raise HTTPException(status_code=404, detail="assistant draft not found")
        if stored["status"] != "draft":
            raise HTTPException(status_code=409, detail=f"assistant draft is not confirmable (status={stored['status']})")
        if _payload_expired(stored.get("plan", {})):
            context.store.set_assistant_draft_status(
                draft_id, "expired", expected_status="draft", error="assistant plan has expired"
            )
            raise HTTPException(status_code=409, detail="assistant plan has expired")
        if not context.store.claim_assistant_draft(draft_id):
            raise HTTPException(status_code=409, detail="assistant draft is already being confirmed")

        raw_execution = stored.get("execution")
        execution_invalid = not isinstance(raw_execution, Mapping)
        execution: dict[str, Any] = dict(raw_execution) if isinstance(raw_execution, Mapping) else {}
        execution.setdefault("steps", [])
        execution.setdefault("receipts", [])
        if not isinstance(execution["steps"], list) or not isinstance(execution["receipts"], list):
            execution_invalid = True
            execution = {"steps": [], "receipts": []}
        receipts: list[CommandReceipt] = []

        def terminal_failure(status: str, message: str) -> HTTPException:
            context.store.set_assistant_draft_status(
                draft_id,
                status,
                expected_status="confirming",
                execution=execution,
                error=message,
            )
            return HTTPException(status_code=409, detail=message)

        try:
            if execution_invalid:
                raise terminal_failure("failed", "stored assistant execution state is invalid")
            if _payload_expired(stored.get("plan", {})):
                raise terminal_failure("expired", "assistant plan has expired")
            plan = PlanDraft.model_validate(stored["plan"])
        except ValidationError as exc:
            message = str(exc)
            if "expiresAt is in the past" in message or "expires_at" in message and "past" in message:
                raise terminal_failure("expired", "assistant plan has expired") from exc
            raise terminal_failure("failed", "stored assistant draft is invalid") from exc
        except HTTPException:
            raise
        try:
            # Resolve every step before dispatching any of them, preserving the
            # confirmation gate and preventing a partially unknown plan.
            ordered = _topological_steps(plan)
            resolved: list[tuple[Any, CommandDefinition, MappingVersion]] = []
            for step in ordered:
                definition = context.definition(step.commandKey, "1")
                mapping = context.mapping(step.commandKey)
                robot = context.store.get_robot(plan.robotId)
                if robot is None:
                    raise HTTPException(status_code=404, detail="robot not found")
                safety_error = _motion_safety_error(robot, definition)
                if safety_error:
                    raise HTTPException(status_code=409, detail=safety_error)
                try:
                    target = _mapping_target_binding(context.mapping_record(step.commandKey))
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if target is not None and target != plan.robotId:
                    raise HTTPException(status_code=409, detail="mapping target does not match robotId")
                resolved.append((step, definition, mapping))
            for step, definition, mapping in resolved:
                request = _build_request(
                    {
                        "robotId": plan.robotId,
                        "commandKey": step.commandKey,
                        "commandVersion": definition.commandVersion,
                        "transport": mapping.transport.value,
                        "payload": step.payload,
                        "expiresAt": plan.expiresAt or (utc_now() + timedelta(seconds=30)),
                        "idempotencyKey": f"{draft_id}:{step.stepId}",
                        "sessionId": draft_id,
                    },
                    definition,
                    mapping,
                    source=CommandSource.AGENT,
                    idempotency_prefix=f"{draft_id}-{step.stepId}",
                )
                receipt = _dispatch_and_save(context, request, definition, mapping)
                receipts.append(receipt)
                execution["steps"].append(
                    {
                        "stepId": step.stepId,
                        "commandId": receipt.commandId,
                        "status": receipt.status.value,
                        "receipt": _receipt_json(receipt),
                    }
                )
                execution["receipts"].append(_receipt_json(receipt))
                if not context.store.update_assistant_execution(draft_id, execution):
                    raise terminal_failure("unknown", "assistant confirmation ownership was lost")
                if receipt.status.value != "succeeded":
                    if receipt.status.value in {"failed", "rejected", "cancelled", "expired"}:
                        status = "partial" if len(receipts) > 1 or any(
                            item.get("status") == "succeeded" for item in execution["steps"][:-1]
                        ) else "failed"
                    else:
                        # ``accepted``/``running``/``pending`` has no durable
                        # completion proof.  Do not continue dependent steps
                        # or mark the whole plan confirmed; this attempt is
                        # unknown and must be reconciled out of band.
                        status = "unknown"
                    raise terminal_failure(status, receipt.error or f"assistant step {step.stepId} did not complete")
            if not context.store.set_assistant_draft_status(
                draft_id, "confirmed", expected_status="confirming", execution=execution
            ):
                raise terminal_failure("unknown", "assistant confirmation ownership was lost")
            return {"draftId": draft_id, "status": "confirmed", "receipts": [_receipt_json(receipt) for receipt in receipts]}
        except HTTPException as exc:
            # Resolution can fail before the first adapter call (for example
            # a removed command definition or active mapping).  It is still a
            # terminal confirmation attempt; leaving ``confirming`` here
            # would make a later restart look like an ambiguous replay.
            current = context.store.get_assistant_draft(draft_id)
            if current is not None and current.get("status") == "confirming":
                status = "partial" if execution["steps"] else "failed"
                detail = str(exc.detail)
                raise terminal_failure(status, detail) from exc
            raise
        except _CommandIdentityConflict as exc:
            status = "partial" if execution["steps"] else "failed"
            raise terminal_failure(status, str(exc)) from exc
        except Exception as exc:
            status = "partial" if execution["steps"] else "failed"
            raise terminal_failure(status, f"assistant confirmation failed: {exc}") from exc

    # Singular aliases keep clients that model a single assistant draft or
    # mapping draft compatible without duplicating execution logic.
    app.add_api_route("/api/v1/mappings", create_mapping_draft, methods=["POST"], dependencies=[Depends(require_auth)], include_in_schema=False)
    app.add_api_route("/api/v1/assistant/draft", create_assistant_draft, methods=["POST"], dependencies=[Depends(require_auth)], include_in_schema=False)
    app.add_api_route("/api/v1/assistant/draft/{draft_id}", get_assistant_draft, methods=["GET"], dependencies=[Depends(require_auth)], include_in_schema=False)
    app.add_api_route("/api/v1/assistant/draft/{draft_id}/confirm", confirm_assistant_draft, methods=["POST"], dependencies=[Depends(require_auth)], include_in_schema=False)

    @app.get("/api/v1/events", dependencies=[Depends(require_auth)])
    def list_events(after: int = 0, context: _ControlContext = Depends(get_context)):
        return context.store.list_events(after_id=after)

    @app.websocket("/api/v1/events")
    async def events_websocket(websocket: WebSocket):
        query_token = websocket.query_params.get("token")
        if not (_parse_bearer(websocket.headers.get("authorization"), token) or (query_token is not None and compare_digest(query_token, token))):
            await websocket.close(code=4401)
            return
        await websocket.accept()
        try:
            try:
                after = int(websocket.query_params.get("after", "0"))
            except ValueError:
                after = 0
            last_id = after
            while True:
                events = websocket.app.state.control.store.list_events(after_id=last_id)
                for event in events:
                    await websocket.send_json(event)
                    last_id = max(last_id, int(event.get("eventId", last_id)))
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return

    dist = Path(__file__).resolve().parents[2] / "apps" / "control-pwa" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="pwa")
    return app


def _topological_steps(plan: PlanDraft) -> list[Any]:
    steps = {step.stepId: step for step in plan.steps}
    ordered: list[Any] = []
    visited: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in visited:
            return
        step = steps[step_id]
        for dependency in step.dependsOn:
            visit(dependency)
        visited.add(step_id)
        ordered.append(step)

    for step in plan.steps:
        visit(step.stepId)
    return ordered


app = create_app()


__all__ = ["app", "create_app"]
