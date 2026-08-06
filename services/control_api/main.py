"""FastAPI boundary for the robot action control plane.

The HTTP layer is intentionally thin: it validates/constructs immutable
control-core contracts, resolves a pinned mapping, and sends every execution
through :class:`ActionRuntime`.  SQLite is only a repository and event sink;
it is never used as a second command execution path.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from hmac import compare_digest
import os
from pathlib import Path
import re
import unicodedata
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
from packages.control_core.registry import MappingRecord, MappingRegistry
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


def _embedded_mapping_target(mapping: MappingVersion) -> str | None:
    targets: list[str] = []
    for key, value in mapping.mapping.items():
        normalized = re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", str(key)).casefold())
        if normalized in {"targetid", "target"}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("mapping target must be a non-empty string")
            targets.append(value.strip())
    if not targets:
        return None
    if len(set(targets)) != 1:
        raise ValueError("mapping target aliases conflict")
    return targets[0]


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


class _ControlContext:
    def __init__(self, store: SQLiteStore, token: str, adapters: Mapping[str | Transport, CommandAdapter]):
        self.store = store
        self.token = token
        normalized = {getattr(key, "value", key): value for key, value in adapters.items()}
        self.adapters = normalized
        self.definitions: dict[tuple[str, str], CommandDefinition] = {}
        self.registry = MappingRegistry(adapters=normalized, audit_sink=store.record_audit)
        self.runtime = ActionRuntime(adapters=normalized, event_sink=store.record_event)

    def seed(self) -> None:
        self.store.reset_confirming_drafts()
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
        self.store.upsert_robot(robot)
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
        if persisted:
            for row in persisted:
                self._restore_mapping(row)
        else:
            for definition in definitions:
                mapping = MappingVersion(
                    mappingId=f"seed-{definition.commandKey.replace('.', '-')}",
                    version="v1",
                    commandKey=definition.commandKey,
                    transport=Transport.FAKE,
                    mapping={"targetId": robot.robotId, "topic": f"/{definition.commandKey}"},
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
                    # A custom adapter may intentionally be unavailable.  The
                    # mapping remains absent and the API reports a controlled
                    # error rather than failing app construction.
                    continue

    def _restore_mapping(self, row: Mapping[str, Any]) -> None:
        # Rebuilding the in-memory registry must not write the replayed
        # snapshots back to the repository.  Doing so would turn historical
        # ``superseded`` rows into ``active`` rows on every restart and would
        # also duplicate lifecycle audit records.  The persisted transition
        # order is already the source of truth; registry operations below are
        # only an in-memory replay.
        audit_sink = self.registry.audit_sink
        self.registry.audit_sink = None
        try:
            mapping = MappingVersion.model_validate(row["mapping"])
            target_id = row.get("targetId")
            status = str(row.get("status", "draft"))
            record = self.registry.create_draft(mapping, target_id=target_id)
            if status in {"validated", "probed", "active", "superseded"}:
                record = self.registry.validate(mapping.mappingId, mapping.version)
            if status in {"probed", "active", "superseded"}:
                record = self.registry.probe(mapping.mappingId, mapping.version)
            if status in {"active", "superseded"}:
                # Replaying superseded activations rebuilds the registry's
                # rollback history while the chronological store order keeps
                # the persisted active pointer as the final one.
                record = self.registry.activate(mapping.mappingId, mapping.version)
        except (ValueError, ValidationError):
            # Ignore stale/corrupt rows during startup; the repository remains
            # readable and callers can submit a new validated mapping.
            return
        finally:
            self.registry.audit_sink = audit_sink

    def definition(self, command_key: str, version: str | None = None) -> CommandDefinition:
        requested = version or "1"
        definition = self.definitions.get((command_key, requested))
        if definition is None:
            raise HTTPException(status_code=404, detail="command definition not found")
        return definition

    def mapping_record(self, command_key: str, mapping_id: str | None = None, version: str | None = None) -> MappingRecord:
        try:
            if mapping_id is not None:
                return self.registry.get(mapping_id, version or "v1")
            mapping = self.registry.active(command_key)
            return self.registry.get(mapping.mappingId, mapping.version)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def mapping(self, command_key: str, mapping_id: str | None = None, version: str | None = None) -> MappingVersion:
        return self.mapping_record(command_key, mapping_id, version).mapping


def create_app(
    db_path: str = ":memory:",
    auth_token: str | None = None,
    adapters: Mapping[str | Transport, CommandAdapter] | None = None,
) -> FastAPI:
    """Create an isolated API instance suitable for tests or deployment."""

    token = auth_token or os.getenv("CONTROL_API_TOKEN", "dev-token")
    adapter_map: dict[str | Transport, CommandAdapter] = dict(adapters or {})
    adapter_map.setdefault("fake", FakeAdapter())
    store = SQLiteStore(db_path)
    context = _ControlContext(store, token, adapter_map)
    context.seed()

    app = FastAPI(title="Robot Action Control API", version="0.1.0")
    app.state.control = context
    app.state.store = store
    app.state.runtime = context.runtime
    app.state.registry = context.registry
    app.state.definitions = context.definitions

    def require_auth(request: Request) -> None:
        if not _parse_bearer(request.headers.get("authorization"), token):
            raise HTTPException(status_code=401, detail="authentication required", headers={"WWW-Authenticate": "Bearer"})

    def get_context(request: Request) -> _ControlContext:
        return request.app.state.control

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
        existing = context.store.get_command_record(request.commandId)
        receipt = context.runtime.dispatch(request, definition, mapping)
        if existing is None:
            # An idempotency retry can return a receipt owned by an earlier
            # command ID.  Persist that original immutable request instead of
            # overwriting it with the retry's payload.
            persisted_request = context.runtime._requests.get(receipt.commandId, request)
            context.store.save_command(persisted_request, receipt)
        return receipt

    def _save_mapping_change(context: _ControlContext, record: MappingRecord, previous: MappingVersion | None = None) -> None:
        if previous is not None and (previous.mappingId, previous.version) != (record.mapping.mappingId, record.version):
            try:
                # Persist the superseded pointer before the new active one so
                # updated_at remains a replayable transition sequence.
                context.store.save_mapping(context.registry.get(previous.mappingId, previous.version))
            except ValueError:
                pass
        context.store.save_mapping(record)

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
        robot_id = body.get("robotId", body.get("robot_id"))
        if not isinstance(robot_id, str) or context.store.get_robot(robot_id) is None:
            raise HTTPException(status_code=404, detail="robot not found")
        command_key = body.get("commandKey", body.get("command_key"))
        if not isinstance(command_key, str) or not command_key.strip():
            raise HTTPException(status_code=422, detail="commandKey is required")
        version = body.get("commandVersion", body.get("command_version", body.get("version", "1")))
        definition = context.definition(command_key, version)
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
        receipt = _dispatch_and_save(context, request, definition, mapping)
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
        if context.store.get_command(command_id) is None:
            raise HTTPException(status_code=404, detail="command not found")
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
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            _save_mapping_change(context, record, previous)
            return _mapping_json(record)

        return route

    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/validate", lifecycle_action("validate"), methods=["POST"], dependencies=[Depends(require_auth)])
    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/probe", lifecycle_action("probe"), methods=["POST"], dependencies=[Depends(require_auth)])
    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/activate", lifecycle_action("activate"), methods=["POST"], dependencies=[Depends(require_auth)])

    @app.post("/api/v1/mappings/{command_key}/rollback", dependencies=[Depends(require_auth)])
    def rollback_mapping(command_key: str, body: dict[str, Any] | None = None, context: _ControlContext = Depends(get_context)):
        version = None if not body else body.get("version")
        try:
            previous = context.registry.active(command_key)
        except ValueError:
            previous = None
        try:
            record = context.registry.rollback(command_key, version=version)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _save_mapping_change(context, record, previous)
        return _mapping_json(record)

    @app.post("/api/v1/assistant/drafts", dependencies=[Depends(require_auth)])
    def create_assistant_draft(body: dict[str, Any], context: _ControlContext = Depends(get_context)):
        draft_id = body.get("draftId", body.get("draft_id")) or body.get("planId", body.get("plan_id")) or _new_id("draft")
        if context.store.get_assistant_draft(draft_id) is not None:
            raise HTTPException(status_code=409, detail="assistant draft already exists")
        plan_data = dict(body)
        plan_data.pop("draftId", None)
        plan_data.pop("draft_id", None)
        plan_data.setdefault("planId", draft_id)
        try:
            plan = PlanDraft.model_validate(plan_data)
        except ValidationError as exc:
            fail_validation(exc)
            raise AssertionError("unreachable")
        context.store.save_assistant_draft(draft_id, _plain(plan), status="draft")
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
            raise HTTPException(status_code=409, detail="assistant draft is already confirmed")
        if not context.store.claim_assistant_draft(draft_id):
            raise HTTPException(status_code=409, detail="assistant draft is already being confirmed")
        try:
            plan = PlanDraft.model_validate(stored["plan"])
        except ValidationError as exc:
            context.store.set_assistant_draft_status(draft_id, "draft")
            raise HTTPException(status_code=409, detail="stored assistant draft is invalid") from exc
        try:
            # Resolve every step before dispatching any of them, preserving the
            # confirmation gate and preventing a partially unknown plan.
            ordered = _topological_steps(plan)
            resolved: list[tuple[Any, CommandDefinition, MappingVersion]] = []
            for step in ordered:
                definition = context.definition(step.commandKey, "1")
                mapping = context.mapping(step.commandKey)
                if context.store.get_robot(plan.robotId) is None:
                    raise HTTPException(status_code=404, detail="robot not found")
                try:
                    target = _mapping_target_binding(context.mapping_record(step.commandKey))
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if target is not None and target != plan.robotId:
                    raise HTTPException(status_code=409, detail="mapping target does not match robotId")
                resolved.append((step, definition, mapping))
            receipts: list[CommandReceipt] = []
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
            context.store.set_assistant_draft_status(draft_id, "confirmed")
            return {"draftId": draft_id, "status": "confirmed", "receipts": [_receipt_json(receipt) for receipt in receipts]}
        except Exception:
            context.store.set_assistant_draft_status(draft_id, "draft")
            raise

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
