"""FastAPI boundary for the robot action control plane.

The HTTP layer is intentionally thin: it validates/constructs immutable
control-core contracts, resolves a pinned mapping, and sends every execution
through :class:`ActionRuntime`.  SQLite is only a repository and event sink;
it is never used as a second command execution path.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hmac
from itertools import permutations
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Mapping

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from packages.scene_core import (
    NAVIGATION_COMMAND_KEY,
    NavigationPayload,
    PoseMap,
    SceneContext,
    VisualNavigationPayload,
    build_spatial_navigation_execution_payload,
    navigation_command_schema,
    robot_navigation_command_schema,
    visual_navigation_command_schema,
    validate_navigation_payload,
)
from packages.control_core.adapters import AdapterResult, CommandAdapter, FakeAdapter, RobotAgentAdapter
from packages.control_core.capabilities import CapabilityKind, CapabilityManifest, RobotProfile
from packages.control_core.models import (
    CapabilitySnapshot,
    CommandDefinition,
    CommandKind,
    CommandReceipt,
    CommandRequest,
    CommandSource,
    CommandStatus,
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
from packages.visual_mission_core import VisualMissionPlanDraft, parse_plan_draft
from packages.visual_mission_core.clients import (
    VisionClient,
    VisualStageClient,
    vision_client_from_env,
    visual_stage_client_from_env,
)
from packages.visual_mission_core.runtime import VisualMissionRuntime
from packages.mission_core import (
    MissionPlanDraft,
    MissionRuntime,
    MissionSnapshot,
    SkillProgress,
    SkillRequest,
    SkillResult,
)
from packages.simulation_core import (
    SimulationEvent,
    SimulationEventType,
    SimulationConflict,
    SimulationReview,
    SimulationRuntime,
    SimulationBackend,
    SimulationSession,
    SimulationSnapshot,
)
from services.robot_agent_gateway import RobotAgentGatewayHttpDriver
from services.speech_gateway import (
    AudioStreamConfig,
    StreamingSpeechGateway,
    asr_provider_from_env,
    create_speech_router,
    provider_capability,
    tts_provider_from_env,
)
from services.speech_gateway.errors import (
    InvalidClientMessage,
    SpeechAuthenticationRequired,
    SpeechAuthorizationDenied,
    SpeechGatewayError,
)

from .assistant_service import AssistantNotFound, AssistantService
from .conversation_client import (
    ConversationClient,
    UnavailableConversationClient,
    conversation_client_from_env,
)
from .office_inspection import (
    MISSION_REPORT_SKILL,
    OFFICE_INSPECTION_SKILL,
    build_report as build_office_inspection_report,
    is_full_office_inspection,
)
from .conversation_gateway import ConversationGateway
from .developer_diagnostics import (
    DeveloperDiagnosticsService,
    create_developer_diagnostics_router,
)
from .auth import (
    AuthService,
    AuthorizationDenied,
    CSRFValidationFailed,
    CSRF_COOKIE,
    Principal,
    Scope,
    SESSION_COOKIE,
    create_auth_router,
    require_resource_access,
    require_scope,
)
from .planning_client import PlanningClient, planning_client_from_env
from .mission_client import (
    MissionPlanningClient,
    mission_planning_client_from_env,
)
from .mission_service import MissionService, MissionServiceError
from .mission_routes import create_mission_router
from .scene_memory import create_scene_memory_router
from .simulation_routes import create_simulation_router
from .scene_client import (
    SceneContextClient,
    SceneContextConflict,
    SceneContextUnavailable,
    scene_context_client_from_env,
)
from .speech import (
    MAX_WAV_BYTES,
    SpeechClient,
    SpeechServiceUnavailable,
    speech_client_from_env,
    validate_wav,
)
from .visual_task_client import (
    VisualTaskClient,
    VisualTaskUnavailable,
    visual_task_client_from_env,
)

from .store import (
    AssistantConversationConflict,
    AssistantTurnConflict,
    MappingPointerConflict,
    MappingStateConflict,
    SQLiteStore,
    _jsonable,
)


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


def _environment_flag(name: str, *, default: bool = False) -> bool:
    """Read one explicit boolean deployment switch.

    Adapter selection changes the authority boundary of the process, so a
    misspelled value must stop startup instead of silently selecting either a
    simulated or physical execution profile.
    """

    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean")


def _robot_agent_localization_epoch_id() -> str | None:
    """Return the operator-bound localization epoch for the real Agent.

    The Robot Agent control protocol deliberately does not carry observation
    identity. H-IDE must therefore bind the current localization epoch from
    deployment configuration instead of inventing one or accepting a value
    proposed by a planner.
    """

    value = os.getenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "").strip()
    if not value:
        return None
    if len(value) > 256:
        raise ValueError(
            "ROBOT_AGENT_LOCALIZATION_EPOCH_ID must be at most 256 characters"
        )
    return value


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _CommandIdentityConflict(ValueError):
    """A command id is already owned by another idempotency key."""


_CONTROL_PWA_DEEP_LINKS = (
    "/app",
    "/app/tasks/{conversation_id}",
    "/app/scene-memory",
    "/developer",
    "/developer/tasks/{conversation_id}",
    "/developer/simulation",
    "/developer/interfaces",
    "/developer/scene-memory",
    # Compatibility-only entry points.  The Assistant V2 product router no
    # longer exposes a scenes workspace, but old bookmarks still land in the
    # shell and no scene data is removed by this routing change.
    "/tasks/{conversation_id}",
    "/scenes",
    "/simulation",
    "/interfaces",
)


def _mount_control_pwa(app: FastAPI, dist: Path) -> bool:
    """Mount built PWA assets and explicit production history fallbacks."""

    index_file = dist / "index.html"
    if not dist.is_dir() or not index_file.is_file():
        return False

    async def serve_control_pwa_route(
        conversation_id: str | None = None,
    ) -> FileResponse:
        del conversation_id
        return FileResponse(
            index_file,
            media_type="text/html",
            headers={"cache-control": "no-cache"},
        )

    for position, path in enumerate(_CONTROL_PWA_DEEP_LINKS):
        app.add_api_route(
            path,
            serve_control_pwa_route,
            methods=["GET"],
            include_in_schema=False,
            name=f"control_pwa_deep_link_{position}",
        )
    app.mount("/", StaticFiles(directory=dist, html=True), name="pwa")
    return True


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


def _draft_scene_context(store: SQLiteStore, draft_id: str) -> SceneContext | None:
    """Recover the exact canonical scene snapshot used to create a planner draft."""

    metadata = store.get_assistant_draft_review(draft_id)
    if metadata is None:
        return None
    turn = store.get_assistant_turn(metadata["taskId"])
    request = None if turn is None else turn.get("request")
    payload = request.get("sceneContext") if isinstance(request, Mapping) else None
    return None if payload is None else SceneContext.model_validate(payload)


def _motion_safety_error(robot: Mapping[str, Any], definition: CommandDefinition) -> str | None:
    """Reject every non-safety action when the persisted robot is unsafe."""
    if definition.kind is CommandKind.SAFETY:
        return None

    # The repository accepts a JSON boundary for forward compatibility, but
    # an unsafe or malformed snapshot must fail closed.  Check the raw
    # connected flag before Pydantic's permissive coercions, then validate the
    # complete state contract and allow only explicitly safe states.
    if not isinstance(robot, Mapping) or robot.get("connected") is not True:
        return "robot connection state is not confirmed safe"
    try:
        state = RobotState.model_validate(robot)
    except (TypeError, ValueError, ValidationError):
        return "robot state is invalid or unknown"
    if state.status not in {RobotStatus.ONLINE, RobotStatus.DEGRADED}:
        return f"robot status {state.status.value} does not allow motion"
    if state.safetyMode not in {SafetyMode.NORMAL, SafetyMode.DEGRADED}:
        return f"robot safety mode {state.safetyMode.value} does not allow motion"
    return None


def _minimum_travel_heading_order(
    headings: list[float],
    *,
    current_yaw: float,
) -> list[tuple[int, float]]:
    """Order one reviewed heading set without changing its report identity.

    Station headings are absolute map yaws, while RobotAgent Spin accepts a
    relative yaw.  Visiting a fixed four-heading set in catalog order can cost
    a full 2*pi plus small physical correction, even though the same reviewed
    set needs only 3*pi/2 when entered at the nearest heading.  Exhaustive
    ordering is bounded by the skill schema's eight-heading maximum and gives
    a deterministic, minimum-yaw route.  The original one-based indexes are
    retained so evidence can be sorted back into immutable catalog order.
    """

    if not math.isfinite(current_yaw) or any(
        not math.isfinite(value) for value in headings
    ):
        raise ValueError("Office inspection headings must be finite")
    indexed = tuple(enumerate(headings, start=1))
    if not indexed:
        return []

    def route_key(
        route: tuple[tuple[int, float], ...],
    ) -> tuple[float, tuple[int, ...]]:
        yaw = current_yaw
        travel = 0.0
        for _, target in route:
            travel += abs(
                math.atan2(math.sin(target - yaw), math.cos(target - yaw))
            )
            yaw = target
        return travel, tuple(index for index, _ in route)

    return list(min(permutations(indexed), key=route_key))


class _RobotAgentGatewayRouter:
    """Route one Robot Agent transport across live and isolated sim peers.

    Robot identity is the routing authority.  H-IDE still exposes one logical
    ``robot_agent`` transport to ActionRuntime, so Mission execution cannot
    bypass the normal authorization, auditing or Nav2-only motion boundary.
    """

    def __init__(
        self,
        default_driver: RobotAgentGatewayHttpDriver,
        robot_drivers: Mapping[str, RobotAgentGatewayHttpDriver],
    ) -> None:
        self.default_driver = default_driver
        self.robot_drivers = dict(robot_drivers)
        self._command_drivers: dict[str, RobotAgentGatewayHttpDriver] = {}
        self._lock = threading.RLock()

    def _for_robot(self, robot_id: str | None) -> RobotAgentGatewayHttpDriver:
        return self.robot_drivers.get(robot_id or "", self.default_driver)

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        return self._for_robot(robot_id).discover(robot_id)

    def probe(self, mapping: MappingVersion) -> AdapterResult:
        target = mapping.mapping.get("targetId")
        return self._for_robot(target if isinstance(target, str) else None).probe(mapping)

    def dispatch(self, request: CommandRequest, mapping: MappingVersion) -> AdapterResult:
        driver = self._for_robot(request.robotId)
        result = driver.dispatch(request, mapping)
        if result.status not in {"unavailable", "failed", "rejected"}:
            with self._lock:
                self._command_drivers[request.commandId] = driver
        return result

    def _command_driver(self, command_id: str) -> RobotAgentGatewayHttpDriver | None:
        with self._lock:
            return self._command_drivers.get(command_id)

    def restore_command_route(self, command_id: str, robot_id: str) -> None:
        """Restore one durable command route after a Control API restart.

        The immutable CommandRequest remains the routing authority.  Without
        this restoration a restarted process cannot query a simulation command
        that is still owned by the independently running gateway, even though
        the persisted request contains the exact robot identity.
        """

        if not isinstance(command_id, str) or not command_id.strip():
            raise ValueError("command route identity is invalid")
        if not isinstance(robot_id, str) or not robot_id.strip():
            raise ValueError("command robot route identity is invalid")
        driver = self._for_robot(robot_id)
        with self._lock:
            existing = self._command_drivers.get(command_id)
            if existing is not None and existing is not driver:
                raise ValueError("command route is already bound to another robot")
            self._command_drivers[command_id] = driver

    def cancel(self, command_id: str) -> AdapterResult:
        driver = self._command_driver(command_id)
        if driver is not None:
            return driver.cancel(command_id)
        # An unowned command is never broadcast to multiple robots.  Runtime
        # recovery marks it unknown and requires operator reconciliation.
        return AdapterResult(status="unknown", error="Robot Agent command route is unavailable")

    def query(self, command_id: str) -> AdapterResult:
        driver = self._command_driver(command_id)
        if driver is not None:
            return driver.query(command_id)
        return AdapterResult(status="unknown", error="Robot Agent command route is unavailable")


class _ControlContext:
    def __init__(
        self,
        store: SQLiteStore,
        token: str,
        adapters: Mapping[str | Transport, CommandAdapter],
        scene_context_client: SceneContextClient,
    ):
        self.store = store
        self.token = token
        normalized = {getattr(key, "value", key): value for key, value in adapters.items()}
        self.adapters = normalized
        self.scene_context_client = scene_context_client
        self.definitions: dict[tuple[str, str], CommandDefinition] = {}
        self.registry = MappingRegistry(adapters=normalized, audit_sink=store.record_audit)
        self.runtime = ActionRuntime(adapters=normalized, event_sink=self._runtime_event)

    def _runtime_event(self, event: dict[str, Any]) -> dict[str, Any]:
        """Persist events and terminal runtime receipts (including watchdog expiry)."""
        persisted = self.store.record_event(event)
        if str(event.get("status", "")).lower() in {
            "succeeded",
            "failed",
            "cancelled",
            "expired",
            "rejected",
            "unknown",
        }:
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
        use_robot_agent = (
            definition.commandKey == NAVIGATION_COMMAND_KEY
            and "robot_agent" in self.adapters
        )
        transport = Transport.ROBOT_AGENT if use_robot_agent else Transport.FAKE
        mapping_id = f"seed-{definition.commandKey.replace('.', '-')}"
        if definition.commandKey == NAVIGATION_COMMAND_KEY:
            # Seed mappings are immutable. Distinct transport IDs let one
            # durable database switch between the demo and Robot Agent
            # profiles without mutating an existing mapping version.
            mapping_id = f"{mapping_id}-{transport.value.replace('_', '-')}"
        mapping = MappingVersion(
            mappingId=mapping_id,
            version="v1",
            commandKey=definition.commandKey,
            transport=transport,
            mapping={
                "targetId": robot_id,
                **(
                    {"adapterProfile": "robot-agent-gateway", "capabilityKey": definition.commandKey}
                    if use_robot_agent
                    else {"topic": f"/{definition.commandKey}"}
                ),
            },
        )
        for _attempt in range(3):
            try:
                try:
                    record = self.registry.get(mapping.mappingId, mapping.version)
                except ValueError:
                    record = self.registry.create_draft(mapping)
                    self.store.save_mapping(record)
                if record.status == "draft":
                    record = self.registry.validate(mapping.mappingId, mapping.version)
                    self.store.save_mapping(record)
                if record.status == "validated":
                    record = self.registry.probe(mapping.mappingId, mapping.version)
                    self.store.save_mapping(record)
                if record.status == "probed":
                    record = self.registry.activate(mapping.mappingId, mapping.version)
                    self.store.save_mapping(record)
                return
            except ValueError:
                # Two app processes can seed the same deterministic mapping at
                # the same time.  Reconcile the local registry with the row
                # that won the durable transition, then retry the remaining
                # lifecycle step instead of leaving this process with a draft-
                # only in-memory view.  An unavailable adapter still exits
                # after the bounded attempts and remains an explicit error.
                try:
                    self.reload_mappings()
                except Exception:
                    pass
        # A custom adapter may intentionally be unavailable.  The mapping
        # remains in its durable intermediate state and the API reports a
        # controlled error until a later restart/reconciliation.
        return

    def seed(self) -> None:
        # A process restart cannot know whether a robot accepted each step of
        # an in-flight confirmation.  Mark it unknown; never silently replay.
        self.store.recover_confirming_drafts()
        self.store.recover_visual_missions()
        self.store.mark_unowned_commands_for_reconciliation()
        using_robot_agent = "robot_agent" in self.adapters
        configured_robot_id = os.getenv("ROBOT_AGENT_ID", "robot-1").strip() if using_robot_agent else "robot-1"
        if not configured_robot_id:
            raise ValueError("ROBOT_AGENT_ID must not be empty")
        discovered_robot: Mapping[str, Any] | None = None
        if using_robot_agent:
            discovered = self.adapters["robot_agent"].discover(configured_robot_id)
            if discovered.status == "available" and isinstance(discovered.result, Mapping):
                discovered_robot = discovered.result
        fake_runtime = not using_robot_agent
        localization_epoch_id = (
            _robot_agent_localization_epoch_id()
            if using_robot_agent
            else "lab-map-epoch-1"
        )
        details: dict[str, Any] = {
            "adapter": "robot_agent" if using_robot_agent else "fake",
            "robotBootId": (
                discovered_robot.get("robotBootId")
                if discovered_robot is not None
                else (None if using_robot_agent else "fixture-boot-1")
            ),
        }
        if localization_epoch_id is not None:
            details["localizationEpochId"] = localization_epoch_id
        initial_pose: PoseMap | None = None
        if fake_runtime:
            initial_pose = PoseMap(x=0, y=0, z=0, yaw=0, observedAt=utc_now())
            details["poseSource"] = "fake_fixture"
        elif discovered_robot is not None and isinstance(
            discovered_robot.get("currentPose"), Mapping
        ):
            try:
                candidate_pose = PoseMap.model_validate(
                    discovered_robot["currentPose"]
                )
            except ValidationError:
                candidate_pose = None
            if candidate_pose is not None and candidate_pose.observedAt is not None:
                initial_pose = candidate_pose
                details["poseSource"] = "robot_agent_state"
        robot = RobotState(
            robotId=configured_robot_id,
            status=(
                RobotStatus.ONLINE
                if discovered_robot is not None or fake_runtime
                else RobotStatus.UNKNOWN
            ),
            connected=discovered_robot is not None or fake_runtime,
            safetyMode=SafetyMode.NORMAL,
            pose=initial_pose,
            details=details,
        )
        discovered_capabilities = (
            discovered_robot.get("capabilities") if discovered_robot is not None else None
        )
        capability = CapabilitySnapshot(
            agent=(
                str(discovered_robot.get("executionBackend") or "robot-agent")
                if discovered_robot is not None
                else ("robot-agent-unavailable" if using_robot_agent else "fake-agent")
            ),
            protocol="robot-agent.v1" if using_robot_agent else "fake",
            robotId=robot.robotId,
            capabilities=(
                [item for item in discovered_capabilities if isinstance(item, str)]
                if isinstance(discovered_capabilities, list)
                else ([] if using_robot_agent else ["move.forward", "stop", "gripper.open", NAVIGATION_COMMAND_KEY])
            ),
            safetyMode=SafetyMode.NORMAL,
            schema={"type": "object"},
        )
        # Seed only missing records.  A restart must not overwrite a robot's
        # persisted ESTOP/offline/degraded state or its last capability view.
        existing_robot = self.store.get_robot(robot.robotId)
        if existing_robot is None:
            self.store.upsert_robot(robot)
        elif existing_robot.get("pose") is None and existing_robot.get("details", {}).get("adapter") == "fake":
            self.store.upsert_robot(
                RobotState.model_validate(existing_robot).model_copy(update={"pose": robot.pose})
            )
        existing_capability = self.store.get_capabilities(robot.robotId)
        if existing_capability is None:
            self.store.upsert_capabilities(capability, robot.robotId)
        elif existing_capability.get("protocol") == "fake" and not using_robot_agent:
            available = existing_capability.get("capabilities", [])
            if isinstance(available, list) and NAVIGATION_COMMAND_KEY not in available:
                self.store.upsert_capabilities(
                    CapabilitySnapshot.model_validate(existing_capability).model_copy(
                        update={"capabilities": [*available, NAVIGATION_COMMAND_KEY]}
                    ),
                    robot.robotId,
                )

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
            CommandDefinition(
                commandKey=NAVIGATION_COMMAND_KEY,
                commandVersion="1",
                transports=[Transport.FAKE, Transport.ROBOT_AGENT],
                kind=CommandKind.MOTION,
                description="Navigate to one deterministic ROS map-frame pose",
                payloadSchema=(
                    robot_navigation_command_schema()
                    if using_robot_agent
                    else navigation_command_schema()
                ),
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
                self._restore_safe_pointer(self.registry, command_key, rows, pointer)
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
        # mutated. The navigation pointer does follow the explicit execution
        # profile by activating a distinct immutable seed mapping.
        for definition in definitions:
            if using_robot_agent and Transport.ROBOT_AGENT not in definition.transports:
                continue
            try:
                active = self.registry.active(definition.commandKey)
            except ValueError:
                self._seed_default_mapping(definition, robot.robotId)
                continue
            desired_transport = (
                Transport.ROBOT_AGENT
                if definition.commandKey == NAVIGATION_COMMAND_KEY
                and using_robot_agent
                else Transport.FAKE
            )
            if (
                definition.commandKey == NAVIGATION_COMMAND_KEY
                and active.transport is not desired_transport
            ):
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

    @staticmethod
    def _restore_safe_pointer(
        registry: MappingRegistry,
        command_key: str,
        rows: list[Mapping[str, Any]],
        pointer: tuple[str, str] | None,
    ) -> bool:
        """Restore only an active, successfully probed durable mapping.

        A corrupt/stale ``mapping_active`` pointer must never promote a draft
        row during process startup.  If the pointer is unusable, choose the
        newest safe active snapshot; the caller can seed a default when none
        exists.
        """
        candidates: list[tuple[str, str]] = []
        if pointer and all(isinstance(item, str) and item for item in pointer):
            candidates.append(pointer)
        active_rows = [
            row for row in rows
            if row.get("status") == "active" and isinstance(row.get("probe"), Mapping)
        ]
        active_rows.sort(key=lambda row: row.get("updatedAt", ""), reverse=True)
        for row in active_rows:
            payload = row.get("mapping")
            if not isinstance(payload, Mapping):
                continue
            mapping_id, version = payload.get("mappingId"), payload.get("version")
            if isinstance(mapping_id, str) and mapping_id and isinstance(version, str) and version:
                candidate = (mapping_id, version)
                if candidate not in candidates:
                    candidates.append(candidate)
        for mapping_id, version in candidates:
            try:
                registry.restore_active_pointer(command_key, mapping_id, version)
                return True
            except ValueError:
                continue
        return False

    def reload_mappings(self) -> None:
        """Reconcile this process's registry with the durable SQLite view.

        A pointer CAS can lose after another app instance commits a mapping
        transition.  Restoring a fresh registry snapshot prevents the losing
        process from serving or mutating the stale in-memory active pointer.
        No registry lifecycle methods are used here, so reconciliation emits
        no audits and performs no adapter probes.
        """
        fresh = MappingRegistry(adapters=self.adapters, audit_sink=None)
        persisted = self.store.list_mappings()
        restored_rows: list[Mapping[str, Any]] = []
        for row in persisted:
            if not isinstance(row, Mapping):
                continue
            mapping_payload = row.get("mapping")
            if not isinstance(mapping_payload, Mapping):
                continue
            try:
                mapping = MappingVersion.model_validate(mapping_payload)
                status = str(row.get("status", "draft"))
                if status not in {"draft", "validated", "probed", "active", "superseded"}:
                    continue
                target_id = row.get("targetId")
                probe_payload = row.get("probe")
                probe = None
                if isinstance(probe_payload, Mapping):
                    try:
                        probe = AdapterResult(**dict(probe_payload))
                    except (TypeError, ValueError, ValidationError):
                        probe = AdapterResult(status="failed", error="persisted probe is invalid")
                fresh.restore_record(MappingRecord(mapping=mapping, status=status, target_id=target_id, probe=probe))
                restored_rows.append(row)
            except (ValueError, ValidationError):
                continue

        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in restored_rows:
            mapping_payload = row.get("mapping")
            if not isinstance(mapping_payload, Mapping):
                continue
            command_key = mapping_payload.get("commandKey") or mapping_payload.get("command_key")
            if isinstance(command_key, str) and command_key:
                grouped.setdefault(command_key, []).append(row)
        pointers = self.store.list_active_pointers()
        for command_key, rows in grouped.items():
            pointer = pointers.get(command_key)
            self._restore_safe_pointer(fresh, command_key, rows, pointer)
            history: list[tuple[str, str]] = []
            for row in sorted(rows, key=lambda item: item.get("updatedAt", "")):
                if row.get("status") != "superseded":
                    continue
                payload = row.get("mapping")
                if not isinstance(payload, Mapping):
                    continue
                mapping_id, version = payload.get("mappingId"), payload.get("version")
                if isinstance(mapping_id, str) and mapping_id and isinstance(version, str) and version:
                    history.append((mapping_id, version))
            if history:
                fresh.restore_history(command_key, history)
        self.registry.restore_state(fresh.snapshot_state())

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
    db_path: str | None = None,
    auth_token: str | None = None,
    adapters: Mapping[str | Transport, CommandAdapter] | None = None,
    watchdog_interval: float = 0.1,
    mission_supervisor_interval: float | None = None,
    planning_client: PlanningClient | None = None,
    conversation_client: ConversationClient | None = None,
    mission_planning_client: MissionPlanningClient | None = None,
    speech_client: SpeechClient | None = None,
    streaming_speech_gateway: StreamingSpeechGateway | None = None,
    scene_context_client: SceneContextClient | None = None,
    assistant_draft_ttl_seconds: float | None = None,
    vision_client: VisionClient | None = None,
    visual_stage_client: VisualStageClient | None = None,
    visual_mission_mode: str | None = None,
    visual_task_client: VisualTaskClient | None = None,
    simulation_runtime: SimulationRuntime | None = None,
    simulation_db_path: str | None = None,
    developer_diagnostics_service: DeveloperDiagnosticsService | None = None,
) -> FastAPI:
    """Create an isolated API instance suitable for tests or deployment."""

    token = auth_token or os.getenv("CONTROL_API_TOKEN", "dev-token")
    adapter_map: dict[str | Transport, CommandAdapter] = dict(adapters or {})
    gateway_enabled = _environment_flag("ROBOT_AGENT_GATEWAY_ENABLED")
    gateway_url = os.getenv("ROBOT_AGENT_GATEWAY_URL", "").strip()
    gateway_token = os.getenv("ROBOT_AGENT_GATEWAY_TOKEN", "").strip()
    simulation_gateway_url = os.getenv("SIMULATION_ROBOT_AGENT_GATEWAY_URL", "").strip()
    simulation_gateway_token = os.getenv("SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN", "").strip()
    simulation_robot_id = os.getenv("SIMULATION_ROBOT_AGENT_ID", "g1-sim").strip()
    simulation_gateway_driver = (
        RobotAgentGatewayHttpDriver(
            simulation_gateway_url,
            simulation_gateway_token,
            timeout_seconds=float(os.getenv("SIMULATION_ROBOT_AGENT_TIMEOUT_SECONDS", "120")),
        )
        if simulation_gateway_url and simulation_gateway_token and simulation_robot_id
        else None
    )
    configured_transports = {getattr(key, "value", key) for key in adapter_map}
    if gateway_enabled and "robot_agent" not in configured_transports:
        if not gateway_url or not gateway_token:
            raise ValueError(
                "ROBOT_AGENT_GATEWAY_ENABLED requires ROBOT_AGENT_GATEWAY_URL "
                "and ROBOT_AGENT_GATEWAY_TOKEN"
            )
        live_gateway_driver = RobotAgentGatewayHttpDriver(gateway_url, gateway_token)
        robot_agent_driver = (
            _RobotAgentGatewayRouter(
                live_gateway_driver,
                {simulation_robot_id: simulation_gateway_driver},
            )
            if simulation_gateway_driver is not None
            else live_gateway_driver
        )
        adapter_map["robot_agent"] = RobotAgentAdapter(driver=robot_agent_driver)
    # No adapter arguments means the safe, self-contained demonstration
    # profile. A real Robot Agent profile never gains a silent Fake fallback;
    # hybrid tests/deployments must pass both adapters explicitly.
    if not adapter_map:
        adapter_map["fake"] = FakeAdapter()
    # An explicit path remains useful for tests and embedded callers.  The
    # deployment factory uses CONTROL_DB_PATH so a uvicorn restart does not
    # silently fall back to an ephemeral in-memory repository.
    resolved_db_path = db_path if db_path is not None else (os.getenv("CONTROL_DB_PATH") or ":memory:")
    store = SQLiteStore(resolved_db_path)
    auth_service = AuthService(store, legacy_bearer_token=token)
    auth_service.bootstrap_admin_from_env()
    simulation = simulation_runtime or SimulationRuntime(
        db_path=(
            simulation_db_path
            if simulation_db_path is not None
            else (os.getenv("SIMULATION_DB_PATH") or resolved_db_path)
        )
    )
    scene_adapter = scene_context_client or scene_context_client_from_env()
    try:
        spatial_translation_drift = float(
            os.getenv("SPATIAL_CONFIRM_TRANSLATION_DRIFT_METERS", "0.5")
        )
        spatial_yaw_drift = float(os.getenv("SPATIAL_CONFIRM_YAW_DRIFT_RADIANS", "0.35"))
    except (TypeError, ValueError) as exc:
        raise ValueError("spatial confirmation drift thresholds must be numbers") from exc
    if (
        not math.isfinite(spatial_translation_drift)
        or not math.isfinite(spatial_yaw_drift)
        or spatial_translation_drift < 0
        or spatial_yaw_drift < 0
    ):
        raise ValueError("spatial confirmation drift thresholds must be finite and non-negative")
    context = _ControlContext(store, token, adapter_map, scene_adapter)
    context.seed()
    primary_robot_id = (
        os.getenv("ROBOT_AGENT_ID", "robot-1").strip()
        if "robot_agent" in context.adapters
        else "robot-1"
    )
    primary_target = (
        "sim"
        if "robot_agent" not in context.adapters
        or (
            simulation_gateway_driver is not None
            and primary_robot_id == simulation_robot_id
        )
        else "real"
    )
    store.bind_robot_site(
        primary_robot_id,
        site_id=auth_service.default_site_id,
        execution_target=primary_target,
    )
    if simulation_gateway_driver is not None:
        store.bind_robot_site(
            simulation_robot_id,
            site_id=auth_service.default_site_id,
            execution_target="sim",
        )
    raw_robot_bindings = os.getenv("ROBOT_SITE_BINDINGS_JSON", "").strip()
    if raw_robot_bindings:
        try:
            configured_bindings = json.loads(raw_robot_bindings)
        except json.JSONDecodeError as exc:
            raise ValueError("ROBOT_SITE_BINDINGS_JSON must be valid JSON") from exc
        if not isinstance(configured_bindings, Mapping):
            raise ValueError("ROBOT_SITE_BINDINGS_JSON must be a robot-keyed object")
        for configured_robot, configured_binding in configured_bindings.items():
            if (
                not isinstance(configured_robot, str)
                or not configured_robot.strip()
                or not isinstance(configured_binding, Mapping)
            ):
                raise ValueError("ROBOT_SITE_BINDINGS_JSON contains an invalid binding")
            configured_site = configured_binding.get("siteId")
            configured_target = configured_binding.get("executionTarget")
            if (
                not isinstance(configured_site, str)
                or not configured_site.strip()
                or configured_site.strip() == "*"
                or any(
                    character.isspace() or ord(character) < 32
                    for character in configured_site.strip()
                )
                or configured_target not in {"sim", "real"}
            ):
                raise ValueError(
                    "ROBOT_SITE_BINDINGS_JSON bindings require siteId and sim/real executionTarget"
                )
            store.bind_robot_site(
                configured_robot.strip(),
                site_id=configured_site.strip(),
                execution_target=str(configured_target),
            )

    # Optional live simulation path. The normal Robot Agent adapter above
    # continues to serve the physical robot. A second, explicitly configured
    # Gateway endpoint owns ``g1-sim`` so the browser can exercise the same
    # Robot Agent -> Nav2 contract without sharing a lease or ROS domain.
    if simulation_gateway_driver is not None:
        simulation_gateway = simulation_gateway_driver

        def simulation_identity(robot_id: str) -> Mapping[str, Any]:
            discovered = simulation_gateway.discover(robot_id)
            if discovered.status != "available" or not isinstance(discovered.result, Mapping):
                raise RuntimeError(discovered.error or "simulation Robot Agent is unavailable")
            value = dict(discovered.result)
            observation = value.get("observation")
            if isinstance(observation, Mapping):
                value.setdefault("observationSessionId", observation.get("sessionId"))
                value.setdefault("localizationEpochId", observation.get("localizationEpochId"))
            return value

        def dispatch_simulation_navigation(
            session: SimulationSession,
            request,
        ) -> Mapping[str, Any]:
            identity = simulation_identity(session.robotId)
            observation_session_id = str(
                session.metadata.get("observationSessionId")
                or identity.get("observationSessionId")
                or ""
            )
            if not observation_session_id:
                raise RuntimeError("simulation observation session is unavailable")
            mission_id = request.missionId or session.missionId or session.sessionId
            payload = {
                "payloadType": "spatial_resolution",
                "goalPose": request.goalPose.model_dump(mode="json"),
                "goalPurpose": "final",
                "missionId": mission_id,
                "stageId": request.nodeId or "simulation-navigation",
                "spatialResolutionId": request.revisionId or f"sim-resolution-{session.sessionId}",
                "observationSessionId": observation_session_id,
                "robotBootId": session.robotBootId,
                "localizationEpochId": session.localizationEpochId,
                "maximumSpeedMps": min(0.25, session.policy.maximumGoalDistanceM / 10.0),
                "sceneRef": {"sceneId": "mujoco-office-floor", "revision": 1},
            }
            issued = utc_now()
            command = CommandRequest(
                commandId=str(uuid.uuid4()),
                robotId=session.robotId,
                source=CommandSource.MANUAL,
                transport=Transport.ROBOT_AGENT,
                commandKey=NAVIGATION_COMMAND_KEY,
                commandVersion="2.0.0",
                payload=payload,
                issuedAt=issued,
                # The MuJoCo Robot Agent advertises a 30 s Nav2 action
                # timeout. Keep the wire TTL within both that capability and
                # robot-agent.v1's 60 s protocol ceiling.
                expiresAt=issued + timedelta(seconds=30),
                idempotencyKey=f"simulation:{session.sessionId}:{request.revisionId or request.nodeId or uuid.uuid4().hex}",
                sessionId=session.sessionId,
                traceId=f"sim-trace-{uuid.uuid4().hex}",
            )
            mapping = MappingVersion(
                mappingId=f"simulation-nav2-{session.robotId}",
                version="v1",
                commandKey=NAVIGATION_COMMAND_KEY,
                transport=Transport.ROBOT_AGENT,
                mapping={
                    "targetId": session.robotId,
                    "adapterProfile": "robot-agent-gateway-simulation",
                    "capabilityKey": NAVIGATION_COMMAND_KEY,
                },
                payloadSchema=robot_navigation_command_schema(),
            )
            result = simulation_gateway.dispatch(command, mapping)
            if result.status not in {"accepted", "running", "succeeded"}:
                raise RuntimeError(result.error or f"simulation Robot Agent rejected navigation ({result.status})")
            return {
                "status": result.status,
                "commandId": result.result.get("robotAgentCommandId"),
                "executionBackend": "robot_agent_nav2_sonic",
                "motion": "mujoco",
                "detail": result.details.get("detail") or "NavigateToPose dispatched through Robot Agent and Nav2",
                "result": result.result,
            }

        def simulation_navigation_status(command_id: str) -> Mapping[str, Any]:
            result = simulation_gateway.query(command_id)
            return {
                "status": result.status,
                "commandId": command_id,
                "detail": result.details.get("detail") or result.error,
                "result": result.result,
            }

        simulation.set_identity_provider(simulation_identity)
        simulation.set_navigation_dispatcher(dispatch_simulation_navigation)
        simulation.set_navigation_status_provider(simulation_navigation_status)

    def project_simulation_snapshot(session: SimulationSession, snapshot: SimulationSnapshot) -> None:
        """Expose simulation pose as a clearly-labelled workstation observation.

        Never overwrite a live Robot Agent state with simulated data. The
        deterministic profile may update the seeded fake robot (or a robot
        already labelled as simulation), which lets Mission V2 use the same
        boot/epoch admission checks without creating a second robot identity.
        """

        existing = store.get_robot(session.robotId)
        existing_details = existing.get("details") if isinstance(existing, Mapping) else {}
        if not isinstance(existing_details, Mapping):
            existing_details = {}
        existing_adapter = existing_details.get("adapter")
        if existing is not None and existing_adapter not in {None, "fake", "simulation"}:
            return
        details = dict(existing_details)
        details.update(
            {
                "adapter": "simulation",
                "executionBackend": f"{session.backend.value}_simulator",
                "simulationSessionId": session.sessionId,
                "sourceMode": session.sourceMode,
                "robotBootId": session.robotBootId,
                "localizationEpochId": session.localizationEpochId,
                "poseSource": "simulation_ground_truth",
            }
        )
        current_safety = SafetyMode.NORMAL
        if isinstance(existing, Mapping):
            try:
                current_safety = RobotState.model_validate(existing).safetyMode
            except (TypeError, ValueError, ValidationError):
                current_safety = SafetyMode.NORMAL
        store.upsert_robot(
            RobotState(
                robotId=session.robotId,
                status=RobotStatus.DEGRADED if session.status.value == "failed" else RobotStatus.ONLINE,
                connected=True,
                safetyMode=current_safety,
                pose=snapshot.pose,
                details=details,
            )
        )

    simulation.set_snapshot_sink(project_simulation_snapshot)
    raw_draft_ttl = (
        assistant_draft_ttl_seconds
        if assistant_draft_ttl_seconds is not None
        else os.getenv("ASSISTANT_DRAFT_TTL_SECONDS", "900")
    )
    try:
        draft_ttl_seconds = float(raw_draft_ttl)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ASSISTANT_DRAFT_TTL_SECONDS must be a number") from exc
    if not math.isfinite(draft_ttl_seconds) or draft_ttl_seconds <= 0:
        raise ValueError("ASSISTANT_DRAFT_TTL_SECONDS must be finite and greater than zero")
    mission_runtime = MissionRuntime(store)
    mission_service = MissionService(
        store,
        mission_runtime,
        mission_planning_client or mission_planning_client_from_env(),
        allow_synthetic_profile=(
            "robot_agent" not in context.adapters
            and _environment_flag("MISSION_V2_ALLOW_SYNTHETIC_PROFILE", default=True)
        ),
    )
    mission_service.scene_context_client = scene_adapter
    conversation_adapter = conversation_client or conversation_client_from_env()
    conversation_gateway = (
        None
        if isinstance(conversation_adapter, UnavailableConversationClient)
        else ConversationGateway(
            conversation_adapter,
            mission_draft_creator=mission_service.create_draft,
        )
    )
    assistant_service = AssistantService(
        store,
        context.definitions,
        planning_client or planning_client_from_env(),
        scene_adapter,
        draft_ttl_seconds=draft_ttl_seconds,
        conversation_gateway=conversation_gateway,
    )

    def bridge_simulation_event(
        session: SimulationSession,
        snapshot: SimulationSnapshot,
        event: SimulationEvent,
    ) -> None:
        """Turn simulated failures into a Mission review proposal only.

        The bridge is intentionally one-way.  It may ask Mastra to evaluate a
        mission event and store a human-required PlanRevision, but it never
        calls ActionRuntime or a Robot Agent adapter.  A missing mission or a
        planner outage leaves the generic durable review marker in place.
        """

        if event.eventType not in {
            SimulationEventType.NAVIGATION_BLOCKED,
            SimulationEventType.NAVIGATION_FAILED,
            SimulationEventType.TARGET_LOST,
        }:
            return
        # Only the explicit session binding opts into planner callbacks.
        # A navigation goal may carry a correlation ID for a demo, but that
        # must not accidentally be treated as an existing Mission identity.
        mission_id = session.missionId
        if not mission_id:
            return
        event_type = (
            "observation.updated"
            if event.eventType is SimulationEventType.TARGET_LOST
            else "verification.failed"
        )
        payload = {
            **dict(event.payload),
            "replan": True,
            "targetLost": event.eventType is SimulationEventType.TARGET_LOST,
            "detail": event.payload.get("reason") or event.eventType.value,
            "simulationSessionId": session.sessionId,
            "simulationEventId": event.eventId,
        }
        try:
            result = mission_service.evaluate_event(
                mission_id,
                {
                    "eventType": event_type,
                    "payload": payload,
                    "correlationId": event.eventId,
                    "occurredAt": event.occurredAt.isoformat(),
                },
                force_human_review=True,
            )
            decision = result.get("decision") if isinstance(result, Mapping) else {}
            if not isinstance(decision, Mapping):
                decision = {}
            raw_revision = decision.get("revision")
            revision_id = (
                raw_revision.get("revisionId")
                if isinstance(raw_revision, Mapping)
                else None
            )
            kind = str(decision.get("kind") or "needs_review")
            detail = str(decision.get("reason") or "仿真事件已提交 Mission 审核")
            review = SimulationReview(
                status="pending",
                triggerEventId=event.eventId,
                triggerEventType=event.eventType.value,
                missionId=mission_id,
                decisionKind=kind if kind in {"replan", "needs_review", "pause", "continue"} else "needs_review",
                revisionId=revision_id,
                detail=detail[:2_000],
                createdAt=event.occurredAt,
                updatedAt=utc_now(),
            )
        except Exception as exc:
            review = SimulationReview(
                status="error",
                triggerEventId=event.eventId,
                triggerEventType=event.eventType.value,
                missionId=mission_id,
                decisionKind="error",
                detail=f"Mission review proposal failed: {str(exc)[:1_800]}",
                createdAt=event.occurredAt,
                updatedAt=utc_now(),
            )
        simulation.set_review(session.sessionId, review)

    simulation.set_event_sink(bridge_simulation_event)
    speech_adapter = speech_client or speech_client_from_env()
    realtime_speech = streaming_speech_gateway or StreamingSpeechGateway(
        asr_provider_from_env(),
        tts_provider=tts_provider_from_env(),
    )
    visual_task_adapter = visual_task_client or visual_task_client_from_env()
    evidence_reader = getattr(visual_task_adapter, "get_evidence", None)
    if callable(evidence_reader):
        mission_service.visual_evidence_reader = evidence_reader
    try:
        assistant_command_wait_seconds = float(
            os.getenv("ASSISTANT_COMMAND_WAIT_SECONDS", "10")
        )
        assistant_command_poll_seconds = float(
            os.getenv("ASSISTANT_COMMAND_POLL_SECONDS", "0.1")
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("assistant command wait settings must be numbers") from exc
    if (
        not math.isfinite(assistant_command_wait_seconds)
        or not 0 <= assistant_command_wait_seconds <= 60
        or not math.isfinite(assistant_command_poll_seconds)
        or not 0.01 <= assistant_command_poll_seconds <= 5
    ):
        raise ValueError("assistant command wait settings are outside their safe ranges")

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
    app.state.auth = auth_service
    app.state.simulation = simulation
    app.state.runtime = context.runtime
    app.state.registry = context.registry
    app.state.definitions = context.definitions
    app.state.assistant = assistant_service
    app.state.mission_runtime = mission_runtime
    app.state.mission = mission_service
    app.state.mission_policy = mission_service.policy
    app.state.speech = speech_adapter
    app.state.realtime_speech = realtime_speech
    app.state.visual_mission = VisualMissionRuntime(
        store=store,
        action_runtime=context.runtime,
        vision_client=vision_client or vision_client_from_env(),
        stage_client=visual_stage_client or visual_stage_client_from_env(),
        mode=visual_mission_mode or os.getenv("VISUAL_MISSION_MODE", "observe"),
        mapping_resolver=context.mapping,
        command_poll_interval_seconds=float(
            os.getenv("VISUAL_MISSION_COMMAND_POLL_SECONDS", "0.25")
        ),
    )
    app.include_router(create_auth_router(auth_service))
    app.include_router(create_scene_memory_router(store))
    app.include_router(
        create_developer_diagnostics_router(
            developer_diagnostics_service or DeveloperDiagnosticsService(store)
        )
    )

    robot_agent_refresh_lock = threading.RLock()

    def refresh_robot_agent_snapshot(requested_robot_id: str | None = None) -> None:
        """Project the current Gateway connection into fail-closed API state."""

        adapter = context.adapters.get("robot_agent")
        if adapter is None:
            return
        with robot_agent_refresh_lock:
            primary_robot_id = os.getenv("ROBOT_AGENT_ID", "robot-1").strip()
            robot_id = (requested_robot_id or primary_robot_id).strip()
            if not robot_id:
                return
            discovered = adapter.discover(robot_id)
            if discovered.status != "available" or not isinstance(
                discovered.result, Mapping
            ):
                existing = context.store.get_robot(robot_id)
                if existing is not None:
                    current = RobotState.model_validate(existing)
                    # Transport loss never clears an operator/safety ESTOP.
                    disconnected_status = (
                        RobotStatus.ESTOP
                        if current.status is RobotStatus.ESTOP
                        else RobotStatus.UNKNOWN
                    )
                    context.store.upsert_robot(
                        current.model_copy(
                            update={
                                "status": disconnected_status,
                                "connected": False,
                            }
                        )
                    )
                existing_capability = context.store.get_capabilities(robot_id)
                if existing_capability is not None:
                    context.store.upsert_capabilities(
                        CapabilitySnapshot.model_validate(
                            existing_capability
                        ).model_copy(update={"capabilities": []}),
                        robot_id,
                    )
                return
            value = discovered.result
            existing = context.store.get_robot(robot_id)
            if existing is None:
                context.store.upsert_robot(
                    RobotState(
                        robotId=robot_id,
                        status=RobotStatus.UNKNOWN,
                        connected=False,
                        safetyMode=SafetyMode.NORMAL,
                        details={"adapter": "robot_agent"},
                    )
                )
                existing = context.store.get_robot(robot_id)
            if existing is not None:
                current = RobotState.model_validate(existing)
                details = dict(current.details)
                previous_boot_id = details.get("robotBootId")
                previous_epoch_id = details.get("localizationEpochId")
                details.update(
                    {
                        "adapter": "robot_agent",
                        "robotAgentObservedAt": utc_now().isoformat(),
                        "robotBootId": value.get("robotBootId"),
                        "capabilitiesDigest": value.get("capabilitiesDigest"),
                        "executionBackend": value.get("executionBackend"),
                        "simulation": robot_id == simulation_robot_id,
                    }
                )
                raw_profile = value.get("robotProfile")
                raw_manifest = value.get("capabilityManifest")
                details.pop("capabilityManifestError", None)
                if isinstance(raw_profile, Mapping) and isinstance(raw_manifest, Mapping):
                    try:
                        profile = RobotProfile.model_validate(raw_profile)
                        manifest = CapabilityManifest.model_validate(raw_manifest)
                        if profile.robotId != robot_id or manifest.robotId != robot_id:
                            raise ValueError("Robot Agent V2 identity mismatch")
                        if manifest.robotProfileDigest != profile.profileDigest:
                            raise ValueError("Robot Agent V2 profile digest mismatch")
                        if value.get("capabilitiesDigest") != manifest.manifestDigest:
                            raise ValueError("Robot Agent V2 manifest digest mismatch")
                        advertised_skills = {
                            item for item in value.get("capabilities", [])
                            if isinstance(item, str)
                        }
                        manifest_skills = {
                            descriptor.skillId for descriptor in manifest.capabilities
                        }
                        if advertised_skills != manifest_skills:
                            raise ValueError("Robot Agent V2 capability list mismatch")
                    except (TypeError, ValueError, ValidationError) as exc:
                        details.pop("robotProfile", None)
                        details.pop("capabilityManifest", None)
                        details.pop("robotProfileDigest", None)
                        details.pop("capabilityManifestDigest", None)
                        details.pop("sourceMode", None)
                        details["capabilityManifestError"] = str(exc)
                    else:
                        details["robotProfile"] = profile.model_dump(mode="json")
                        details["capabilityManifest"] = manifest.model_dump(mode="json")
                        details["robotProfileDigest"] = profile.profileDigest
                        details["capabilityManifestDigest"] = manifest.manifestDigest
                        profile_source_mode = profile.metadata.get("sourceMode")
                        if profile_source_mode in {"live", "synthetic", "replay"}:
                            details["sourceMode"] = profile_source_mode
                        else:
                            details.pop("sourceMode", None)
                else:
                    # A legacy handshake cannot inherit a V2 manifest from an
                    # older connection. Mission V2 remains unavailable until
                    # both immutable documents are published together.
                    details.pop("robotProfile", None)
                    details.pop("capabilityManifest", None)
                    details.pop("robotProfileDigest", None)
                    details.pop("capabilityManifestDigest", None)
                    details.pop("sourceMode", None)
                observation = value.get("observation")
                if isinstance(observation, Mapping):
                    details["observationStatus"] = str(
                        observation.get("status") or "UNKNOWN"
                    ).upper()
                    details["observationObservedAt"] = utc_now().isoformat()
                    observation_session_id = observation.get("sessionId")
                    if isinstance(observation_session_id, str) and observation_session_id:
                        details["observationSessionId"] = observation_session_id
                    else:
                        details.pop("observationSessionId", None)
                    observation_epoch_id = observation.get("localizationEpochId")
                    if isinstance(observation_epoch_id, str) and observation_epoch_id:
                        details["observationLocalizationEpochId"] = observation_epoch_id
                    else:
                        details.pop("observationLocalizationEpochId", None)
                else:
                    details.pop("observationSessionId", None)
                    details.pop("observationLocalizationEpochId", None)
                    details.pop("observationStatus", None)
                    details.pop("observationObservedAt", None)
                localization_epoch_id = (
                    _robot_agent_localization_epoch_id()
                    if robot_id == primary_robot_id
                    else (
                        observation.get("localizationEpochId")
                        if isinstance(observation, Mapping)
                        else value.get("localizationEpochId")
                    )
                )
                if not isinstance(localization_epoch_id, str) or not localization_epoch_id:
                    localization_epoch_id = None
                if localization_epoch_id is None:
                    # Remove values written by older H-IDE releases. A missing
                    # binding is safer than silently retaining a guessed epoch.
                    details.pop("localizationEpochId", None)
                else:
                    details["localizationEpochId"] = localization_epoch_id
                projected_pose = current.pose
                raw_current_pose = value.get("currentPose")
                observation_epoch = (
                    observation.get("localizationEpochId")
                    if isinstance(observation, Mapping)
                    else None
                )
                details.pop("currentPoseError", None)
                if isinstance(raw_current_pose, Mapping):
                    try:
                        live_pose = PoseMap.model_validate(raw_current_pose)
                        if live_pose.observedAt is None:
                            raise ValueError("Robot Agent currentPose has no observedAt")
                        if (
                            localization_epoch_id is not None
                            and observation_epoch != localization_epoch_id
                        ):
                            raise ValueError(
                                "Robot Agent currentPose localization epoch mismatch"
                            )
                    except (TypeError, ValueError, ValidationError) as exc:
                        details["currentPoseError"] = str(exc)
                        projected_pose = None
                        details.pop("poseSource", None)
                        details.pop("poseSourceCommandId", None)
                    else:
                        projected_pose = live_pose
                        details["poseSource"] = "robot_agent_state"
                elif (
                    previous_boot_id != value.get("robotBootId")
                    or previous_epoch_id != localization_epoch_id
                    or details.get("poseSource") != "nav2_terminal_goal"
                ):
                    # Never carry a missing live state pose (or a fabricated or
                    # cross-boot pose) into a new decision. A later state
                    # snapshot can repopulate it without restarting Control
                    # API. A just-completed Nav2 goal is the one bounded
                    # conservative fallback that may survive a transient
                    # discovery snapshot; _current_robot_pose still applies
                    # its short freshness TTL.
                    projected_pose = None
                    details.pop("poseSource", None)
                    details.pop("poseSourceCommandId", None)
                safe_to_mark_online = (
                    current.status is not RobotStatus.ESTOP
                    and current.safetyMode
                    not in {SafetyMode.STOPPED, SafetyMode.ESTOP}
                )
                context.store.upsert_robot(
                    current.model_copy(
                        update={
                            "status": (
                                RobotStatus.ONLINE
                                if safe_to_mark_online
                                else current.status
                            ),
                            "connected": True,
                            "lastSeenAt": utc_now(),
                            "pose": projected_pose,
                            "details": details,
                        }
                    )
                )
            capabilities = value.get("capabilities")
            if isinstance(capabilities, list):
                context.store.upsert_capabilities(
                    CapabilitySnapshot(
                        agent=str(value.get("executionBackend") or "robot-agent"),
                        protocol=str(value.get("protocol") or "robot-agent.v1"),
                        robotId=robot_id,
                        capabilities=[
                            item for item in capabilities if isinstance(item, str)
                        ],
                        safetyMode=SafetyMode.NORMAL,
                        schema={"type": "object"},
                    ),
                    robot_id,
                )
            try:
                navigation_mapping = context.mapping(NAVIGATION_COMMAND_KEY)
            except (HTTPException, ValueError):
                definition = context.definition(NAVIGATION_COMMAND_KEY, "1")
                context._seed_default_mapping(definition, robot_id)
            else:
                if navigation_mapping.transport is not Transport.ROBOT_AGENT:
                    definition = context.definition(NAVIGATION_COMMAND_KEY, "1")
                    context._seed_default_mapping(definition, robot_id)

    # The simulation Robot Agent is a first-class planning target, not merely
    # a side-channel used by the observatory.  Seed its immutable profile,
    # capability manifest, current pose and observation identity at startup.
    if simulation_gateway_driver is not None:
        refresh_robot_agent_snapshot(simulation_robot_id)

    def require_current_capability(
        robot_id: str,
        command_key: str,
        *,
        refresh: bool = True,
    ) -> None:
        """Require a command in the latest authoritative capability snapshot."""

        if refresh:
            refresh_robot_agent_snapshot(robot_id)
        capability = context.store.get_capabilities(robot_id)
        available = capability.get("capabilities") if isinstance(capability, Mapping) else None
        if not isinstance(available, list) or command_key not in available:
            raise HTTPException(
                status_code=409,
                detail=f"current robot capability is missing: {command_key}",
            )

    def require_robot_agent_capability(robot_id: str, command_key: str) -> None:
        # A Robot Agent may come online after Control API startup. Refresh first so
        # discovery can activate the validated mapping from the live capability set.
        refresh_robot_agent_snapshot(robot_id)
        mapping = context.mapping(command_key)
        if mapping.transport is Transport.ROBOT_AGENT:
            require_current_capability(robot_id, command_key, refresh=False)

    def project_succeeded_navigation_pose(
        skill: SkillRequest,
        receipt: CommandReceipt,
    ) -> None:
        """Keep a bounded current-pose fallback from a terminal Nav2 result."""

        if (
            skill.skillId != NAVIGATION_COMMAND_KEY
            or receipt.status.value != "succeeded"
        ):
            return
        goal = PoseMap.model_validate(skill.input.get("goalPose"))
        current = context.store.get_robot(skill.robotId)
        if current is None:
            return
        details = dict(current.get("details") or {})
        details["poseSource"] = "nav2_terminal_goal"
        details["poseSourceCommandId"] = skill.requestId
        context.store.upsert_robot(
            RobotState.model_validate(current).model_copy(
                update={
                    "pose": goal.model_copy(update={"observedAt": utc_now()}),
                    "details": details,
                }
            )
        )

    def dispatch_mission_skill(
        skill: SkillRequest,
        *,
        inspection_parent: SkillRequest | None = None,
    ) -> dict[str, Any]:
        """Bridge one authorized skill to Robot Agent through ActionRuntime."""

        if skill.expiresAt <= utc_now():
            raise ValueError("Mission skill request has expired")
        refresh_robot_agent_snapshot(skill.robotId)
        robot = context.store.get_robot(skill.robotId)
        if robot is None:
            raise ValueError("robot not found")
        details = robot.get("details")
        if not isinstance(details, Mapping):
            raise ValueError("current Robot Agent details are unavailable")
        manifest = CapabilityManifest.model_validate(details.get("capabilityManifest"))
        if manifest.manifestDigest != skill.capabilityManifestDigest:
            raise ValueError("skill request capability manifest is stale")
        descriptor = next(
            (
                item
                for item in manifest.capabilities
                if item.skillId == skill.skillId and item.skillVersion == skill.skillVersion
            ),
            None,
        )
        if descriptor is None:
            raise ValueError("skill is absent from the current Robot Agent manifest")
        require_current_capability(skill.robotId, skill.skillId, refresh=False)
        if descriptor.kind is CapabilityKind.MOTION and descriptor.executor != "nav2":
            raise ValueError("Mission V2 motion must be executed by Nav2")

        def reject_direct_motion(value: Any, *, path: str = "input") -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    normalized = "".join(
                        character for character in str(key).casefold() if character.isalnum()
                    )
                    if normalized in {
                        "angularvelocity",
                        "cmdvel",
                        "linearvelocity",
                        "path",
                        "waypoints",
                    }:
                        raise ValueError(f"{path}.{key} bypasses the Nav2 execution boundary")
                    reject_direct_motion(item, path=f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    reject_direct_motion(item, path=f"{path}[{index}]")

        if descriptor.kind is CapabilityKind.MOTION:
            reject_direct_motion(skill.input)
        command_kind = {
            CapabilityKind.MOTION: CommandKind.MOTION,
            CapabilityKind.SAFETY: CommandKind.SAFETY,
        }.get(descriptor.kind, CommandKind.NON_MOTION)
        definition = CommandDefinition(
            commandKey=descriptor.skillId,
            commandVersion=descriptor.skillVersion,
            transports=[Transport.ROBOT_AGENT],
            kind=command_kind,
            description=descriptor.description,
            payloadSchema=descriptor.inputSchema,
        )
        safety_error = _motion_safety_error(robot, definition)
        if safety_error:
            raise ValueError(safety_error)
        # The issue-time check is intentionally repeated against the freshly
        # refreshed Robot Agent snapshot immediately before ActionRuntime.
        # A stale/revoked/cross-revision grant or changed health can therefore
        # never ride on an earlier successful decision.
        if inspection_parent is None:
            mission_service.policy.authorize_dispatch(skill)
        else:
            mission_service.policy.authorize_inspection_child(
                inspection_parent,
                skill,
            )
        mapping = MappingVersion(
            mappingId=(
                f"mission-{descriptor.skillId.replace('.', '-')}-"
                f"{manifest.manifestDigest.removeprefix('sha256:')[:16]}"
            ),
            version="v1",
            commandKey=descriptor.skillId,
            transport=Transport.ROBOT_AGENT,
            mapping={
                "targetId": skill.robotId,
                "adapterProfile": "robot-agent-gateway",
                "capabilityKey": descriptor.skillId,
                "manifestDigest": manifest.manifestDigest,
            },
        )
        command = CommandRequest(
            commandId=skill.requestId,
            robotId=skill.robotId,
            source=CommandSource.AGENT,
            transport=Transport.ROBOT_AGENT,
            commandKey=descriptor.skillId,
            commandVersion=descriptor.skillVersion,
            payload=skill.input,
            issuedAt=skill.issuedAt,
            expiresAt=skill.expiresAt,
            idempotencyKey=skill.idempotencyKey,
            sessionId=skill.missionId,
            traceId=skill.requestId,
        )
        receipt = context.runtime.dispatch(
            command,
            definition,
            mapping,
            lease_seconds=min(descriptor.timeoutMs / 1000, 2_400),
        )
        context.store.save_command(command, receipt)
        project_succeeded_navigation_pose(skill, receipt)
        return {
            "mode": "robot_agent",
            "receipt": _receipt_json(receipt),
            "mapping": _plain(mapping),
        }

    def dispatch_assistant_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run one fixed-head scene-description task without creating motion authority."""

        task_id = payload.get("taskId")
        robot_id = payload.get("robotId")
        instruction = payload.get("instruction")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (task_id, robot_id, instruction)
        ):
            raise ValueError("assistant observation requires taskId, robotId, and instruction")
        robot, profile, manifest = mission_service._robot_manifest(robot_id)
        details = robot.get("details") if isinstance(robot.get("details"), Mapping) else {}
        configured_sensors = [
            item.strip()
            for item in os.getenv("HIDE_VISION_SENSOR_IDS", "head_rgbd").split(",")
            if item.strip()
        ]
        sensor_id = next(
            (
                item.sensorId
                for item in profile.sensors
                if item.sensorId in configured_sensors
                and item.depthCapable
                and item.fixedMount
            ),
            None,
        )
        if sensor_id is None:
            raise ValueError("no configured fixed RGB-D sensor is present in RobotProfile")
        source_mode = details.get("sourceMode") or profile.metadata.get("sourceMode")
        if source_mode not in {"live", "synthetic", "replay"}:
            source_mode = "synthetic" if details.get("adapter") == "fake" else "live"
        robot_boot_id = details.get("robotBootId")
        localization_epoch_id = details.get("localizationEpochId")
        if details.get("adapter") == "fake":
            robot_boot_id = robot_boot_id or "synthetic-boot"
            localization_epoch_id = localization_epoch_id or "synthetic-epoch"
        if not isinstance(robot_boot_id, str) or not robot_boot_id:
            raise ValueError("assistant observation robotBootId is unavailable")
        if not isinstance(localization_epoch_id, str) or not localization_epoch_id:
            raise ValueError("assistant observation localizationEpochId is unavailable")
        visual_request = {
            "schemaVersion": "h-ide.visual-task/2",
            "requestId": f"assistant-observe-{task_id}",
            "missionId": f"assistant-observe-{task_id}",
            "revisionId": f"assistant-observe-revision-{task_id}",
            "nodeId": "observe-current-scene",
            "authorizationId": f"assistant-readonly-{task_id}",
            "attempt": 1,
            "traceId": task_id,
            "mode": "scene_description",
            "instruction": instruction,
            "robotId": robot_id,
            "robotBootId": robot_boot_id,
            "localizationEpochId": localization_epoch_id,
            "robotProfileDigest": profile.profileDigest,
            "capabilityManifestDigest": manifest.manifestDigest,
            "sensorId": sensor_id,
            "requestedSourceMode": source_mode,
        }
        result = visual_task_adapter.analyze(visual_request)
        if result.get("schemaVersion") != "h-ide.visual-result/2":
            raise ValueError("H-IDE Vision returned an unsupported result schema")
        for field in ("requestId", "missionId", "revisionId", "nodeId"):
            if result.get(field) != visual_request[field]:
                raise ValueError(f"H-IDE Vision observation changed {field}")
        if result.get("mode") != "scene_description":
            raise ValueError("H-IDE Vision observation changed the read-only mode")
        if result.get("geometryCandidates"):
            raise ValueError("scene-description observation returned unexpected motion geometry")
        return dict(result)

    if conversation_gateway is not None:
        conversation_gateway.visual_observer = dispatch_assistant_observation

    def dispatch_visual_workstation_skill(skill: SkillRequest) -> dict[str, Any]:
        if skill.expiresAt <= utc_now():
            raise ValueError("Mission workstation skill request has expired")
        if skill.skillId != "vision.observe":
            raise ValueError(f"unsupported workstation skill: {skill.skillId}")
        refresh_robot_agent_snapshot(skill.robotId)
        robot = context.store.get_robot(skill.robotId)
        mission = context.store.get_mission(skill.missionId)
        if robot is None or mission is None:
            raise ValueError("visual skill robot or Mission context is unavailable")
        details = robot.get("details")
        if not isinstance(details, Mapping):
            raise ValueError("visual skill Robot Agent identity is unavailable")
        plan = mission.get("plan")
        envelope = plan.get("envelope") if isinstance(plan, Mapping) else None
        if not isinstance(envelope, Mapping):
            raise ValueError("visual skill Mission envelope is unavailable")
        configured_sensors = [
            item.strip()
            for item in os.getenv("HIDE_VISION_SENSOR_IDS", "head_rgbd").split(",")
            if item.strip()
        ]
        raw_profile = details.get("robotProfile")
        if isinstance(raw_profile, Mapping):
            profile = RobotProfile.model_validate(raw_profile)
            sensor_id = next(
                (
                    item.sensorId
                    for item in profile.sensors
                    if item.sensorId in configured_sensors
                    and item.depthCapable
                    and item.fixedMount
                ),
                None,
            )
        elif details.get("adapter") == "fake":
            # FakeAdapter deliberately has no Robot Agent profile endpoint.
            # MissionService already bound the plan to its immutable
            # synthetic fixed-head profile, so use the configured fixture
            # sensor only in this explicit development mode.
            sensor_id = configured_sensors[0] if configured_sensors else "head_rgbd"
        else:
            sensor_id = None
        if sensor_id is None:
            raise ValueError("no configured fixed RGB-D sensor is present in RobotProfile")
        try:
            attempt = int(skill.requestId.rsplit("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("visual skill request attempt is invalid") from exc
        scene_refs = envelope.get("sceneRefs")
        scene_ref = scene_refs[0] if isinstance(scene_refs, list) and scene_refs else None
        visual_input = dict(skill.input)
        robot_source_mode = details.get("sourceMode")
        if robot_source_mode not in {"live", "synthetic", "replay"}:
            robot_source_mode = os.getenv("HIDE_VISION_SOURCE_MODE", "live")
        request_payload: dict[str, Any] = {
            "schemaVersion": "h-ide.visual-task/2",
            "requestId": skill.requestId,
            "missionId": skill.missionId,
            "revisionId": skill.revisionId,
            "nodeId": skill.nodeId,
            "authorizationId": skill.authorizationId,
            "attempt": attempt,
            "traceId": skill.requestId,
            "mode": visual_input.get("mode"),
            "instruction": visual_input.get("instruction"),
            "targetDescription": visual_input.get("targetDescription"),
            "direction": visual_input.get("direction"),
            "requestedDistanceM": visual_input.get("requestedDistanceM"),
            "expectedTargetPose": visual_input.get("expectedTargetPose"),
            "verificationMaximumDistanceM": visual_input.get(
                "verificationMaximumDistanceM", 1.25
            ),
            "verificationMaximumYawRad": visual_input.get(
                "verificationMaximumYawRad", 0.35
            ),
            "robotId": skill.robotId,
            "robotBootId": envelope.get("robotBootId"),
            "localizationEpochId": envelope.get("localizationEpochId"),
            "robotProfileDigest": envelope.get("robotProfileDigest"),
            "capabilityManifestDigest": envelope.get("capabilityManifestDigest"),
            "sensorId": sensor_id,
            # Bind the visual task to the selected robot's observation source.
            # A workstation may host a live Robot Agent and an isolated
            # MuJoCo Robot Agent at the same time; one process-wide source
            # mode would make one of those streams fail identity admission.
            "requestedSourceMode": robot_source_mode,
            "sceneId": scene_ref.get("sceneId") if isinstance(scene_ref, Mapping) else None,
            "sceneRevision": scene_ref.get("revision") if isinstance(scene_ref, Mapping) else None,
            "candidateId": visual_input.get("candidateId"),
            "selectionSourceAnalysisId": visual_input.get("selectionSourceAnalysisId"),
        }
        # Pydantic accepts omitted optional fields, while explicit JSON null is
        # kept only for fields that the VisualTask contract declares nullable.
        result = visual_task_adapter.analyze(request_payload)
        return {"mode": "h_ide_vision", "visualResult": dict(result)}

    office_inspection_lock = threading.RLock()
    office_inspection_workers: dict[str, dict[str, Any]] = {}

    class _OfficeInspectionCancelled(RuntimeError):
        pass

    def _wrapped_yaw_delta(target: float, current: float) -> float:
        return math.atan2(math.sin(target - current), math.cos(target - current))

    def _inspection_parent_is_active(skill: SkillRequest) -> bool:
        mission = context.store.get_mission(skill.missionId)
        snapshot = mission.get("snapshot") if isinstance(mission, Mapping) else None
        if not isinstance(snapshot, Mapping):
            return False
        if snapshot.get("activeSkillRequestId") != skill.requestId:
            return False
        execution = next(
            (
                item
                for item in mission.get("skillExecutions", [])
                if isinstance(item, Mapping)
                and item.get("requestId") == skill.requestId
            ),
            None,
        )
        return isinstance(execution, Mapping) and execution.get("result") is None

    def _inspection_progress(
        skill: SkillRequest,
        *,
        sequence: int,
        progress: float | None,
        phase: str,
        detail: str,
    ) -> None:
        mission_service.runtime.record_progress(
            skill.missionId,
            SkillProgress(
                requestId=skill.requestId,
                sequence=sequence,
                progress=progress,
                phase=phase,
                observedAt=utc_now(),
                detail=detail,
            ),
        )

    def _inspection_child_rotation(
        parent: SkillRequest,
        *,
        heading_index: int,
        target_heading: float,
    ) -> SkillRequest | None:
        pose = mission_service._current_robot_pose(parent.robotId)
        delta = _wrapped_yaw_delta(float(target_heading), float(pose.yaw))
        if abs(delta) <= 0.05:
            return None
        issued_at = utc_now()
        expires_at = min(parent.expiresAt, issued_at + timedelta(seconds=180))
        if expires_at <= issued_at:
            raise ValueError("Office inspection rotation lifetime has expired")
        request_id = f"{parent.requestId}-rotate-{heading_index:02d}"
        mission = mission_service.runtime.get(parent.missionId)
        plan = mission.get("plan")
        envelope = plan.get("envelope") if isinstance(plan, Mapping) else None
        if not isinstance(envelope, Mapping):
            raise ValueError("Office inspection Mission envelope is unavailable")
        robot_boot_id = envelope.get("robotBootId")
        localization_epoch_id = envelope.get("localizationEpochId")
        if not isinstance(robot_boot_id, str) or not isinstance(
            localization_epoch_id, str
        ):
            raise ValueError("Office inspection motion identity is unavailable")
        return SkillRequest(
            requestId=request_id,
            missionId=parent.missionId,
            revisionId=parent.revisionId,
            nodeId=parent.nodeId,
            robotId=parent.robotId,
            skillId="navigation.rotate",
            skillVersion="2.0.0",
            input={
                "targetYawRad": delta,
                "timeAllowanceMs": 180_000,
                "missionId": parent.missionId,
                "revisionId": parent.revisionId,
                "nodeId": parent.nodeId,
                "authorizationId": parent.authorizationId,
                "skillRequestId": request_id,
                "idempotencyKey": request_id,
                "robotBootId": robot_boot_id,
                "localizationEpochId": localization_epoch_id,
            },
            capabilityManifestDigest=parent.capabilityManifestDigest,
            authorizationId=parent.authorizationId,
            idempotencyKey=request_id,
            issuedAt=issued_at,
            expiresAt=expires_at,
        )

    def _inspection_wait_for_rotation(
        parent: SkillRequest,
        child: SkillRequest,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        with state["commandLock"]:
            if state["cancelEvent"].is_set():
                raise _OfficeInspectionCancelled()
            state["activeChild"] = child
            dispatch = dispatch_mission_skill(
                child,
                inspection_parent=parent,
            )
            if state["cancelEvent"].is_set():
                cancel_robot_agent_skill(child)
                state["activeChild"] = None
                raise _OfficeInspectionCancelled()
        receipt = dispatch.get("receipt") if isinstance(dispatch, Mapping) else None
        if not isinstance(receipt, Mapping):
            raise ValueError("Office inspection rotation returned no receipt")
        deadline = min(
            time.monotonic() + 185.0,
            time.monotonic() + max(0.0, (child.expiresAt - utc_now()).total_seconds()),
        )
        try:
            while str(receipt.get("status") or "").lower() in {
                "accepted",
                "running",
                "pending",
            }:
                if state["cancelEvent"].wait(0.5):
                    raise _OfficeInspectionCancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Office inspection rotation did not finish in time"
                    )
                refreshed = reconcile_mission_skill(child)
                receipt = (
                    refreshed.get("receipt")
                    if isinstance(refreshed, Mapping)
                    else None
                )
                if not isinstance(receipt, Mapping):
                    raise ValueError(
                        "Office inspection rotation reconciliation failed"
                    )
        except Exception:
            # The child is a real RobotAgent motion command.  Never let the
            # background worker clear its in-memory pointer before the owning
            # command has reached a cancelled/terminal state: the public
            # Mission cancel path and this wake-up path deliberately race.
            with state["commandLock"]:
                if state.get("activeChild") is child:
                    cancel_robot_agent_skill(child)
                    state["activeChild"] = None
            raise
        with state["commandLock"]:
            if state.get("activeChild") is child:
                state["activeChild"] = None
        if str(receipt.get("status") or "").lower() != "succeeded":
            receipt_error = receipt.get("error")
            suffix = (
                f": {receipt_error}"
                if isinstance(receipt_error, str) and receipt_error.strip()
                else ""
            )
            raise RuntimeError(
                f"Office inspection rotation did not succeed{suffix}"
            )
        return dict(receipt)

    def _inspection_visual_request(
        parent: SkillRequest,
        *,
        heading_index: int,
        target_heading: float,
        attempt: int,
    ) -> SkillRequest:
        issued_at = utc_now()
        expires_at = min(parent.expiresAt, issued_at + timedelta(seconds=150))
        if expires_at <= issued_at:
            raise ValueError("Office inspection observation lifetime has expired")
        request_id = (
            f"{parent.requestId}-vision-h{heading_index:02d}-{attempt}"
        )
        station_id = str(parent.input["stationId"])
        region_id = str(parent.input["regionId"])
        return SkillRequest(
            requestId=request_id,
            missionId=parent.missionId,
            revisionId=parent.revisionId,
            # VisualTaskResult is durably unique on
            # (missionId, revisionId, nodeId, attempt). A station scan is a
            # composite node with several independently auditable camera
            # views, so every heading owns a distinct child-node identity.
            nodeId=f"{parent.nodeId}-view-{heading_index:02d}",
            robotId=parent.robotId,
            skillId="vision.observe",
            skillVersion="2.0.0",
            input={
                "mode": "scene_description",
                "instruction": (
                    "请只根据当前固定头 RGB-D 画面，客观描述这个巡检方向可见的"
                    "房间标牌、家具、设备、通道、障碍物和其他沿途信息。"
                    "不要猜测画面外内容；无法确认的内容必须明确标为不确定。"
                    f"当前产品目录区域为 {region_id}，巡检站为 {station_id}，"
                    f"目标地图朝向为 {target_heading:.6f} 弧度。"
                ),
                "targetDescription": f"{region_id}/{station_id}",
            },
            capabilityManifestDigest=parent.capabilityManifestDigest,
            authorizationId=parent.authorizationId,
            idempotencyKey=request_id,
            issuedAt=issued_at,
            expiresAt=expires_at,
        )

    def _validate_inspection_visual(
        request: SkillRequest,
        dispatch: Mapping[str, Any],
    ) -> dict[str, Any]:
        visual = dispatch.get("visualResult")
        if not isinstance(visual, Mapping):
            raise ValueError("Office inspection observation returned no visual result")
        if visual.get("schemaVersion") != "h-ide.visual-result/2":
            raise ValueError("Office inspection observation schema is invalid")
        for field, expected in {
            "requestId": request.requestId,
            "missionId": request.missionId,
            "revisionId": request.revisionId,
            "nodeId": request.nodeId,
            "mode": "scene_description",
        }.items():
            if visual.get(field) != expected:
                raise ValueError(
                    f"Office inspection visual result changed {field}"
                )
        try:
            expected_attempt = int(request.requestId.rsplit("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("Office inspection visual attempt is invalid") from exc
        if visual.get("attempt") != expected_attempt:
            raise ValueError("Office inspection visual attempt changed")
        mission = mission_service.runtime.get(request.missionId)
        plan = mission.get("plan")
        envelope = plan.get("envelope") if isinstance(plan, Mapping) else None
        source = visual.get("source")
        robot = context.store.get_robot(request.robotId)
        robot_details = (
            robot.get("details")
            if isinstance(robot, Mapping) and isinstance(robot.get("details"), Mapping)
            else None
        )
        if not isinstance(envelope, Mapping) or not isinstance(source, Mapping):
            raise ValueError("Office inspection visual source identity is unavailable")
        if (
            source.get("robotId") != request.robotId
            or source.get("robotBootId") != envelope.get("robotBootId")
            or source.get("localizationEpochId")
            != envelope.get("localizationEpochId")
            or not isinstance(source.get("observationSessionId"), str)
            or not source.get("observationSessionId")
            or not isinstance(source.get("sensorId"), str)
            or not source.get("sensorId")
        ):
            raise ValueError("Office inspection visual source identity changed")
        expected_observation_session = (
            robot_details.get("observationSessionId")
            if isinstance(robot_details, Mapping)
            else None
        )
        if (
            isinstance(expected_observation_session, str)
            and expected_observation_session
            and source.get("observationSessionId") != expected_observation_session
        ):
            raise ValueError("Office inspection visual observation session changed")
        analysis_id = visual.get("analysisId")
        request_sha256 = visual.get("requestSha256")
        if (
            not isinstance(analysis_id, str)
            or not analysis_id.strip()
            or not isinstance(request_sha256, str)
            or len(request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in request_sha256.lower())
        ):
            raise ValueError("Office inspection visual analysis identity is invalid")
        versions = visual.get("versions")
        if (
            not isinstance(versions, Mapping)
            or versions.get("vlmProvider") != "qwen"
            or versions.get("vlmModel") != "qwen3-vl-plus"
            or versions.get("segmentationProvider") != "sam2"
            or versions.get("poseModel") not in {None, ""}
        ):
            raise ValueError("Office inspection visual model profile changed")
        if str(visual.get("status") or "").lower() != "succeeded":
            raise ValueError("Office inspection visual task did not succeed")
        summary = visual.get("observationSummary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Office inspection visual result has no summary")
        if visual.get("geometryCandidates"):
            raise ValueError("Office scene description returned motion geometry")
        evidence = visual.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Office inspection visual result has no evidence")
        for item in evidence:
            sha256 = item.get("sha256") if isinstance(item, Mapping) else None
            if (
                not isinstance(item, Mapping)
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or any(character not in "0123456789abcdef" for character in sha256.lower())
                or not isinstance(item.get("mediaType"), str)
            ):
                raise ValueError("Office inspection visual evidence is invalid")
        return dict(visual)

    def _run_office_inspection(
        parent: SkillRequest,
        state: dict[str, Any],
    ) -> None:
        headings = [float(value) for value in parent.input["scanHeadingsRad"]]
        initial_pose = mission_service._current_robot_pose(parent.robotId)
        ordered_headings = _minimum_travel_heading_order(
            headings,
            current_yaw=float(initial_pose.yaw),
        )
        maximum_yaw = float(parent.input["maximumCumulativeYawRad"])
        capture_policy = parent.input.get("capturePolicy")
        if not isinstance(capture_policy, Mapping):
            raise ValueError("Office inspection capture policy is unavailable")
        settle_seconds = float(capture_policy.get("stopBeforeCaptureS", 1.0))
        if not math.isfinite(settle_seconds) or not 0.0 <= settle_seconds <= 5.0:
            raise ValueError("Office inspection settle time is invalid")
        heading_results: list[dict[str, Any]] = []
        cumulative_yaw = 0.0
        evidence_ids: list[str] = []
        try:
            for scan_sequence, (heading_index, heading) in enumerate(
                ordered_headings,
                start=1,
            ):
                if state["cancelEvent"].is_set() or not _inspection_parent_is_active(parent):
                    raise _OfficeInspectionCancelled()
                state["failureStage"] = "rotation"
                state["headingIndex"] = heading_index
                state["scanSequence"] = scan_sequence
                child = _inspection_child_rotation(
                    parent,
                    heading_index=heading_index,
                    target_heading=heading,
                )
                receipt: dict[str, Any] | None = None
                if child is not None:
                    delta = abs(float(child.input["targetYawRad"]))
                    cumulative_yaw += delta
                    if cumulative_yaw > maximum_yaw + 1e-6:
                        raise ValueError("Office inspection exceeded its yaw reservation")
                    receipt = _inspection_wait_for_rotation(parent, child, state)
                if state["cancelEvent"].wait(settle_seconds):
                    raise _OfficeInspectionCancelled()
                state["failureStage"] = "visual_observation"
                visual: dict[str, Any] | None = None
                visual_request: SkillRequest | None = None
                attempts = 0
                for attempt in (1, 2):
                    attempts = attempt
                    visual_request = _inspection_visual_request(
                        parent,
                        heading_index=heading_index,
                        target_heading=heading,
                        attempt=attempt,
                    )
                    try:
                        visual = _validate_inspection_visual(
                            visual_request,
                            dispatch_visual_workstation_skill(visual_request),
                        )
                    except Exception:
                        if attempt == 2:
                            raise
                        if state["cancelEvent"].wait(1.0):
                            raise _OfficeInspectionCancelled()
                        continue
                    break
                assert visual is not None and visual_request is not None
                for evidence in visual.get("evidence", []):
                    sha = evidence.get("sha256") if isinstance(evidence, Mapping) else None
                    if isinstance(sha, str) and sha not in evidence_ids:
                        evidence_ids.append(sha)
                heading_results.append(
                    {
                        "headingIndex": heading_index,
                        "headingRad": heading,
                        "rotationRequestId": child.requestId if child is not None else None,
                        "rotationReceipt": receipt,
                        "visualRequestId": visual_request.requestId,
                        "visualAttempts": attempts,
                        "visualResult": visual,
                    }
                )
                _inspection_progress(
                    parent,
                    sequence=scan_sequence,
                    progress=scan_sequence / len(headings),
                    phase="captured_heading",
                    detail=(
                        f"Captured heading {scan_sequence} of {len(headings)}"
                    ),
                )
                state["failureStage"] = "heading_complete"
            if state["cancelEvent"].is_set() or not _inspection_parent_is_active(parent):
                raise _OfficeInspectionCancelled()
            mission_service.runtime.record_result(
                parent.missionId,
                SkillResult(
                    requestId=parent.requestId,
                    status="succeeded",
                    completedAt=utc_now(),
                    result={
                        "inspectionResult": {
                            "schemaVersion": "h-ide.office-inspection/1",
                            "stationId": parent.input["stationId"],
                            "regionId": parent.input["regionId"],
                            "cumulativeYawRad": cumulative_yaw,
                            "headings": sorted(
                                heading_results,
                                key=lambda item: int(item["headingIndex"]),
                            ),
                        }
                    },
                    evidenceIds=evidence_ids[:50],
                ),
            )
            state["status"] = "completed"
        except _OfficeInspectionCancelled:
            state["status"] = "cancelled"
        except Exception as exc:
            state["status"] = "failed"
            context.runtime.diagnostics.append(
                {
                    "message": "office inspection worker failed",
                    "requestId": parent.requestId,
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if _inspection_parent_is_active(parent) and not state["cancelEvent"].is_set():
                try:
                    mission_service.runtime.record_result(
                        parent.missionId,
                        SkillResult(
                            requestId=parent.requestId,
                            status="failed",
                            completedAt=utc_now(),
                            result={
                                "dispatchAuthority": "workstation",
                                "stationId": parent.input.get("stationId"),
                                "failure": {
                                    "stage": str(
                                        state.get("failureStage") or "unknown"
                                    ),
                                    "headingIndex": state.get("headingIndex"),
                                    "scanSequence": state.get("scanSequence"),
                                    "errorType": type(exc).__name__,
                                },
                            },
                            errorCode="OFFICE_INSPECTION_FAILED",
                            detail=(
                                "Office station inspection did not complete "
                                f"during {state.get('failureStage') or 'unknown'}"
                            ),
                        ),
                    )
                except Exception:
                    mission_service.runtime.record_recovery_required(
                        parent.missionId,
                        request_id=parent.requestId,
                        error_code="WORKSTATION_INSPECTION_RECOVERY_REQUIRED",
                        detail=(
                            "Office inspection result persistence was interrupted; "
                            "manual recovery is required"
                        ),
                    )
        finally:
            # An unexpected exception outside the normal rotation wait must
            # not orphan a physical child.  Clear ownership only after the
            # RobotAgent cancellation boundary confirms a terminal receipt.
            try:
                with state["commandLock"]:
                    child = state.get("activeChild")
                    if isinstance(child, SkillRequest):
                        cancel_robot_agent_skill(child)
                        state["activeChild"] = None
            except Exception:
                context.runtime.diagnostics.append(
                    {
                        "message": (
                            "office inspection active child could not be "
                            "cancelled during worker cleanup"
                        ),
                        "requestId": parent.requestId,
                    }
                )
                if _inspection_parent_is_active(parent):
                    try:
                        mission_service.runtime.record_recovery_required(
                            parent.missionId,
                            request_id=parent.requestId,
                            error_code=(
                                "WORKSTATION_INSPECTION_RECOVERY_REQUIRED"
                            ),
                            detail=(
                                "Office inspection child motion ownership could "
                                "not be released; manual recovery is required"
                            ),
                        )
                    except Exception:
                        pass
            state["doneEvent"].set()

    def _start_office_inspection(skill: SkillRequest) -> dict[str, Any]:
        with office_inspection_lock:
            existing = office_inspection_workers.get(skill.requestId)
            if existing is not None:
                return {
                    "mode": "office_inspection_pending",
                    "requestId": skill.requestId,
                    "status": str(existing.get("status") or "running"),
                }
            _inspection_progress(
                skill,
                sequence=0,
                progress=0.0,
                phase="inspection_started",
                detail="Office station inspection started",
            )
            state: dict[str, Any] = {
                "parent": skill,
                "cancelEvent": threading.Event(),
                "doneEvent": threading.Event(),
                "commandLock": threading.Lock(),
                "activeChild": None,
                "status": "running",
                "failureStage": "inspection_start",
                "headingIndex": None,
                "scanSequence": None,
            }
            worker = threading.Thread(
                target=_run_office_inspection,
                args=(skill, state),
                name=f"office-inspection-{skill.requestId[-24:]}",
                daemon=True,
            )
            state["thread"] = worker
            office_inspection_workers[skill.requestId] = state
            try:
                worker.start()
            except Exception:
                office_inspection_workers.pop(skill.requestId, None)
                raise
        return {
            "mode": "office_inspection_pending",
            "requestId": skill.requestId,
            "status": "running",
        }

    def reconcile_workstation_skill(skill: SkillRequest) -> dict[str, Any]:
        if skill.skillId != OFFICE_INSPECTION_SKILL:
            return {"status": "lost"}
        with office_inspection_lock:
            state = office_inspection_workers.get(skill.requestId)
            if state is None:
                return {"status": "lost"}
            status = str(state.get("status") or "running")
        return {
            "mode": "office_inspection_pending",
            "requestId": skill.requestId,
            "status": "pending" if status == "running" else status,
        }

    def dispatch_workstation_skill(skill: SkillRequest) -> dict[str, Any]:
        if skill.expiresAt <= utc_now():
            raise ValueError("Mission workstation skill request has expired")
        if skill.skillId == "vision.observe":
            return dispatch_visual_workstation_skill(skill)
        if skill.skillId == OFFICE_INSPECTION_SKILL:
            return _start_office_inspection(skill)
        if skill.skillId == MISSION_REPORT_SKILL:
            mission = mission_service.runtime.get(skill.missionId)
            if mission.get("snapshot", {}).get("activeSkillRequestId") != skill.requestId:
                raise ValueError("Office report no longer owns the active Mission request")
            source_missions: list[Mapping[str, Any]] = []
            raw_source_ids = skill.input.get("sourceMissionIds")
            if raw_source_ids is not None:
                if (
                    not isinstance(raw_source_ids, list)
                    or not raw_source_ids
                    or len(raw_source_ids) > 4
                    or len(set(raw_source_ids)) != len(raw_source_ids)
                    or any(
                        not isinstance(item, str)
                        or not item
                        or item == skill.missionId
                        for item in raw_source_ids
                    )
                ):
                    raise ValueError("Office report source Mission binding is invalid")
                current_plan = MissionPlanDraft.model_validate(mission["plan"])
                for source_mission_id in raw_source_ids:
                    source = mission_service.runtime.get(source_mission_id)
                    source_plan = MissionPlanDraft.model_validate(source["plan"])
                    source_snapshot = MissionSnapshot.model_validate(source["snapshot"])
                    if (
                        source_plan.robotId != current_plan.robotId
                        or source_plan.objective != current_plan.objective
                        or not is_full_office_inspection(source_plan.objective)
                        or source_snapshot.activeSkillRequestId is not None
                    ):
                        raise ValueError(
                            "Office report source Mission identity is incompatible"
                        )
                    source_missions.append(source)
            return {
                "mode": "office_mission_report",
                "missionReport": build_office_inspection_report(
                    mission,
                    source_missions=source_missions,
                ),
            }
        raise ValueError(f"unsupported workstation skill: {skill.skillId}")

    def stop_office_inspection_workers() -> None:
        with office_inspection_lock:
            states = list(office_inspection_workers.values())
            for state in states:
                state["cancelEvent"].set()
        for state in states:
            parent = state.get("parent")
            command_lock = state.get("commandLock")
            acquired = False
            if callable(getattr(command_lock, "acquire", None)):
                acquired = bool(command_lock.acquire(timeout=5.0))
            try:
                if acquired:
                    child = state.get("activeChild")
                    if isinstance(child, SkillRequest):
                        try:
                            cancel_robot_agent_skill(child)
                        except Exception:
                            context.runtime.diagnostics.append(
                                {
                                    "message": (
                                        "office inspection child cancellation failed "
                                        "during Control API shutdown"
                                    ),
                                    "requestId": child.requestId,
                                }
                            )
                        state["activeChild"] = None
            finally:
                if acquired:
                    command_lock.release()
            worker = state.get("thread")
            if isinstance(worker, threading.Thread) and worker.is_alive():
                worker.join(timeout=2.0)
            if isinstance(parent, SkillRequest) and _inspection_parent_is_active(parent):
                try:
                    mission_service.runtime.record_recovery_required(
                        parent.missionId,
                        request_id=parent.requestId,
                        error_code="WORKSTATION_INSPECTION_RECOVERY_REQUIRED",
                        detail=(
                            "Office inspection was interrupted by Control API shutdown; "
                            "manual recovery is required"
                        ),
                    )
                except Exception:
                    context.runtime.diagnostics.append(
                        {
                            "message": (
                                "office inspection shutdown recovery marker failed"
                            ),
                            "requestId": parent.requestId,
                        }
                    )

    def reconcile_mission_skill(skill: SkillRequest) -> dict[str, Any]:
        record = context.store.get_command_record(skill.requestId)
        if record is None:
            raise ValueError("Mission command receipt is unavailable for reconciliation")
        try:
            receipt = context.runtime.reconcile(skill.requestId)
        except ValueError:
            # A restarted Control API has no in-memory ActionRuntime owner,
            # but the independently deployed Gateway retains the immutable
            # command ledger. Query that exact ID; never redispatch it and
            # never convert an unproved/not-found response into a terminal.
            persisted = record.get("receipt")
            persisted_request = record.get("request")
            if not isinstance(persisted, Mapping):
                raise ValueError("persisted Mission command receipt is invalid")
            if not isinstance(persisted_request, Mapping):
                raise ValueError("persisted Mission command request is invalid")
            command = CommandRequest.model_validate(persisted_request)
            if (
                command.commandId != skill.requestId
                or command.robotId != skill.robotId
                or command.commandKey != skill.skillId
                or command.transport is not Transport.ROBOT_AGENT
            ):
                raise ValueError("persisted Mission command identity is invalid")
            adapter = context.adapters.get(Transport.ROBOT_AGENT.value)
            if adapter is None:
                return {"mode": "robot_agent", "receipt": dict(persisted)}
            restore_route = getattr(
                getattr(adapter, "driver", None),
                "restore_command_route",
                None,
            )
            if callable(restore_route):
                restore_route(command.commandId, command.robotId)
            outcome = adapter.query(command.commandId)
            outcome_status = str(outcome.status).strip().lower()
            if outcome_status in {"unavailable", "running", "accepted", "pending"}:
                return {"mode": "robot_agent", "receipt": dict(persisted)}
            release = (
                outcome.result.get("controlRelease")
                if isinstance(outcome.result, Mapping)
                else None
            )
            release_epoch = (
                release.get("safeStopEpoch")
                if isinstance(release, Mapping)
                else None
            )
            release_confirmed = (
                isinstance(release, Mapping)
                and release.get("state") == "SAFE_HOLD"
                and release.get("authorityRevoked") is True
                and release.get("safeStopState") == "confirmed"
                and isinstance(release_epoch, int)
                and not isinstance(release_epoch, bool)
                and release_epoch >= 1
            )
            # A Gateway restart may answer "command not found" even though
            # the old Robot Agent outcome is unknown. Only its fresh release
            # proof permits a recovered physical terminal to escape.
            if not release_confirmed:
                return {"mode": "robot_agent", "receipt": dict(persisted)}
            terminal_status = {
                "succeeded": CommandStatus.SUCCEEDED,
                "success": CommandStatus.SUCCEEDED,
                "ok": CommandStatus.SUCCEEDED,
                "completed": CommandStatus.SUCCEEDED,
                "failed": CommandStatus.FAILED,
                "cancelled": CommandStatus.CANCELLED,
                "canceled": CommandStatus.CANCELLED,
                "expired": CommandStatus.EXPIRED,
                "timed_out": CommandStatus.EXPIRED,
                "timeout": CommandStatus.EXPIRED,
                "rejected": CommandStatus.REJECTED,
                "unknown": CommandStatus.UNKNOWN,
            }.get(outcome_status)
            if terminal_status is None:
                return {"mode": "robot_agent", "receipt": dict(persisted)}
            previous = CommandReceipt.model_validate(persisted)
            receipt = CommandReceipt.model_validate(
                {
                    **previous.model_dump(mode="json", by_alias=False),
                    "status": terminal_status,
                    "completedAt": utc_now(),
                    "result": outcome.result,
                    "error": (
                        None
                        if terminal_status is CommandStatus.SUCCEEDED
                        else outcome.error
                        or f"Robot Agent command {outcome_status}"
                    ),
                }
            )
        # The immutable request may legitimately be expired by the time an
        # asynchronous Robot Agent command reaches a terminal state.  It was
        # fully validated at dispatch; reconciliation must update only its
        # receipt instead of re-admitting the historical request as new work.
        persisted_request = record.get("request")
        if not isinstance(persisted_request, Mapping):
            raise ValueError("persisted Mission command request is invalid")
        context.store.save_command(persisted_request, receipt)
        project_succeeded_navigation_pose(skill, receipt)
        return {"mode": "robot_agent", "receipt": _receipt_json(receipt)}

    def cancel_robot_agent_skill(skill: SkillRequest) -> dict[str, Any]:
        record = context.store.get_command_record(skill.requestId)
        if record is None:
            raise MissionServiceError(
                "active physical skill command is unavailable for cancellation"
            )
        persisted = record.get("receipt")
        if not isinstance(persisted, Mapping):
            raise MissionServiceError("persisted physical command receipt is invalid")
        persisted_status = str(persisted.get("status") or "").lower()
        if persisted_status in {
            "succeeded",
            "failed",
            "cancelled",
            "canceled",
            "expired",
            "rejected",
            "unknown",
        }:
            return {"mode": "robot_agent", "receipt": dict(persisted)}
        if (
            skill.requestId not in context.runtime._requests
            or skill.requestId not in context.runtime._receipts
        ):
            raise MissionServiceError(
                "physical command ownership is unavailable after restart; "
                "explicit Robot Agent reconciliation is required"
            )
        try:
            receipt = context.runtime.cancel(skill.requestId)
        except ValueError as exc:
            raise MissionServiceError(str(exc)) from exc
        command = context.runtime._requests.get(skill.requestId)
        if command is not None:
            context.store.save_command(command, receipt)
        return {"mode": "robot_agent", "receipt": _receipt_json(receipt)}

    def cancel_mission_skill(skill: SkillRequest) -> dict[str, Any]:
        if skill.skillId != OFFICE_INSPECTION_SKILL:
            return cancel_robot_agent_skill(skill)
        with office_inspection_lock:
            state = office_inspection_workers.get(skill.requestId)
            if state is None:
                raise MissionServiceError(
                    "office inspection worker ownership is unavailable after restart"
                )
            state["cancelEvent"].set()
        child_receipt: Mapping[str, Any] | None = None
        with state["commandLock"]:
            child = state.get("activeChild")
            if isinstance(child, SkillRequest):
                child_receipt = cancel_robot_agent_skill(child).get("receipt")
                state["activeChild"] = None
        state["status"] = "cancelled"
        now = utc_now().isoformat()
        return {
            "mode": "office_inspection",
            "receipt": {
                "commandId": skill.requestId,
                "robotId": skill.robotId,
                "status": "cancelled",
                "acceptedAt": skill.issuedAt.isoformat(),
                "completedAt": now,
                "result": {
                    "dispatchAuthority": "workstation",
                    "activeChildReceipt": (
                        dict(child_receipt)
                        if isinstance(child_receipt, Mapping)
                        else None
                    ),
                },
                "error": None,
            },
        }

    app.state.refresh_robot_agent_snapshot = refresh_robot_agent_snapshot
    app.state.require_current_capability = require_current_capability
    app.state.require_robot_agent_capability = require_robot_agent_capability
    mission_service.robot_refresher = refresh_robot_agent_snapshot
    if "robot_agent" in context.adapters:
        mission_service.skill_dispatcher = dispatch_mission_skill
        mission_service.skill_reconciler = reconcile_mission_skill
        mission_service.skill_canceller = cancel_mission_skill
    mission_service.workstation_skill_dispatcher = dispatch_workstation_skill
    mission_service.workstation_skill_reconciler = reconcile_workstation_skill

    def resolve_simulation_review(session_id: str, revision_id: str | None) -> dict[str, Any]:
        """Resolve a simulation gate only after Mission V2 is no longer pending."""

        simulation_record = simulation.get(session_id)
        session_payload = simulation_record["session"]
        snapshot_payload = simulation_record["snapshot"]
        mission_id = session_payload.get("missionId")
        review = snapshot_payload.get("pendingReview")
        if not mission_id:
            raise SimulationConflict("simulation is not bound to a Mission")
        if not isinstance(review, Mapping) or review.get("status") == "resolved":
            return simulation_record
        pending_revision = review.get("revisionId")
        if not isinstance(pending_revision, str) or not pending_revision:
            raise SimulationConflict(
                "simulation review has no approved Mission revision; reset or provide operator input"
            )
        if revision_id is not None and revision_id != pending_revision:
            raise SimulationConflict("review revision does not match the pending proposal")
        try:
            mission = mission_service.get_mission(str(mission_id))
        except Exception as exc:
            raise SimulationConflict(f"Mission review status is unavailable: {exc}") from exc
        mission_snapshot = mission.get("snapshot") if isinstance(mission, Mapping) else None
        if not isinstance(mission_snapshot, Mapping):
            raise SimulationConflict("Mission review snapshot is unavailable")
        mission_status = str(mission_snapshot.get("status") or "unknown")
        if mission_status == "awaiting_review":
            raise SimulationConflict("Mission revision still requires human approval")
        if pending_revision:
            active_revision_id = mission_snapshot.get("activeRevisionId")
            activated = active_revision_id == pending_revision
            if not activated and isinstance(mission, Mapping):
                active_revision = next(
                    (
                        item
                        for item in mission.get("revisions", [])
                        if isinstance(item, Mapping) and item.get("revisionId") == active_revision_id
                    ),
                    None,
                )
                activated = bool(
                    isinstance(active_revision, Mapping)
                    and active_revision.get("parentRevisionId") == pending_revision
                    and active_revision.get("disposition") == "human_approved"
                )
            if not activated:
                raise SimulationConflict("Mission has not activated the reviewed revision")
        return simulation.resolve_review(
            session_id,
            revision_id=pending_revision if isinstance(pending_revision, str) else revision_id,
        )

    mission_internal_token = os.getenv("MISSION_V2_INTERNAL_TOKEN", "").strip()
    if mission_internal_token and hmac.compare_digest(mission_internal_token, token):
        raise ValueError(
            "MISSION_V2_INTERNAL_TOKEN must be independent from the browser legacy token"
        )
    app.include_router(
        create_mission_router(mission_service, mission_internal_token or None)
    )
    app.include_router(
        create_simulation_router(
            simulation,
            token,
            review_resolver=resolve_simulation_review,
        )
    )

    try:
        watchdog_seconds = float(watchdog_interval)
    except (TypeError, ValueError, OverflowError):
        watchdog_seconds = 0.1
    if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0:
        watchdog_seconds = 0.1
    try:
        supervisor_seconds = float(
            os.getenv("MISSION_SUPERVISOR_INTERVAL_SECONDS", "1.0")
            if mission_supervisor_interval is None
            else mission_supervisor_interval
        )
    except (TypeError, ValueError, OverflowError):
        supervisor_seconds = 1.0
    if not math.isfinite(supervisor_seconds) or supervisor_seconds <= 0:
        supervisor_seconds = 1.0
    watchdog_stop = threading.Event()
    diagnostic_lock = threading.Lock()

    def background_diagnostic(message: str, exc: Exception) -> None:
        # A disconnected robot must not grow this in-memory trail without
        # bound while the supervisor keeps retrying fail closed.
        with diagnostic_lock:
            context.runtime.diagnostics.append(
                {"message": message, "error": str(exc)}
            )
            if len(context.runtime.diagnostics) > 200:
                del context.runtime.diagnostics[:-200]

    def watchdog() -> None:
        while not watchdog_stop.wait(watchdog_seconds):
            try:
                context.runtime.tick()
            except Exception as exc:
                background_diagnostic("runtime watchdog failed", exc)

    def mission_supervisor() -> None:
        # Network reconciliation is intentionally isolated from the local
        # motion-lease watchdog. A slow Gateway cannot delay lease expiry.
        while not watchdog_stop.wait(supervisor_seconds):
            try:
                mission_service.converge_supervised_missions()
            except Exception as exc:
                background_diagnostic("Mission supervisor failed", exc)

    watchdog_thread = threading.Thread(target=watchdog, name="control-runtime-watchdog", daemon=True)
    mission_supervisor_thread = threading.Thread(
        target=mission_supervisor,
        name="control-mission-supervisor",
        daemon=True,
    )
    watchdog_thread.start()
    mission_supervisor_thread.start()

    def stop_watchdog() -> None:
        watchdog_stop.set()
        if threading.current_thread() is not watchdog_thread:
            watchdog_thread.join(timeout=max(0.5, watchdog_seconds * 4))
        if threading.current_thread() is not mission_supervisor_thread:
            mission_supervisor_thread.join(
                timeout=min(5.0, max(0.5, supervisor_seconds * 4))
            )

    app.state.stop_watchdog = stop_watchdog
    app.state.watchdog_thread = watchdog_thread
    app.state.mission_supervisor_thread = mission_supervisor_thread
    app.router.add_event_handler("shutdown", stop_office_inspection_workers)
    app.router.add_event_handler("shutdown", stop_watchdog)
    app.router.add_event_handler("shutdown", assistant_service.close)
    app.router.add_event_handler("shutdown", simulation.close)

    # Scope dependencies are defined once and reused by route decorators. The
    # static deployment token maps only to a legacy DEVELOPER principal; a
    # browser-provided role flag never participates in these decisions.
    require_robots_read = require_scope(Scope.ROBOTS_READ)
    require_assistant = require_scope(Scope.ASSISTANT_USE)
    require_speech = require_scope(Scope.SPEECH_USE)
    require_commands = require_scope(Scope.COMMANDS_MANAGE)
    require_mappings = require_scope(Scope.MAPPINGS_MANAGE)
    require_events = require_scope(Scope.EVENTS_READ)

    async def authorize_realtime_speech(
        websocket: WebSocket,
        start: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Credentials in query strings leak through access logs, browser
        # history, and copied URLs.  Cookie sessions and explicit Bearer
        # headers are the only accepted authentication carriers here.
        if any(
            key.casefold() in {"token", "access_token", "authorization"}
            for key in websocket.query_params.keys()
        ):
            raise InvalidClientMessage("Speech credentials must not be sent in the URL")
        try:
            principal = auth_service.authenticate_websocket(
                authorization=websocket.headers.get("authorization"),
                session_token=websocket.cookies.get(SESSION_COOKIE),
                legacy_query_token=None,
                csrf_cookie=websocket.cookies.get(CSRF_COOKIE),
                csrf_token=(
                    start.get("csrfToken")
                    if isinstance(start.get("csrfToken"), str)
                    else None
                ),
                require_csrf=True,
            )
        except CSRFValidationFailed as exc:
            raise SpeechAuthorizationDenied() from exc
        except AuthorizationDenied as exc:
            raise SpeechAuthenticationRequired() from exc
        try:
            auth_service.policy.require(principal, Scope.SPEECH_USE)
        except AuthorizationDenied as exc:
            raise SpeechAuthorizationDenied() from exc
        config = AudioStreamConfig.from_start_message(start)
        return {"principal": principal, "locale": config.locale}

    async def issue_realtime_transcription(
        event: Mapping[str, Any],
        authorization_context: Any,
    ) -> Mapping[str, Any]:
        if not isinstance(authorization_context, Mapping):
            raise RuntimeError("speech authorization context is unavailable")
        principal = authorization_context.get("principal")
        locale = authorization_context.get("locale")
        if not isinstance(principal, Principal) or not isinstance(locale, str):
            raise RuntimeError("speech authorization context is invalid")
        text = event.get("text")
        speech_session_id = event.get("sessionId")
        utterance_id = event.get("utteranceId")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (text, speech_session_id, utterance_id)
        ):
            raise RuntimeError("final transcript identity is incomplete")
        record = await run_in_threadpool(
            context.store.save_streaming_transcription,
            text=text.strip(),
            language=(
                event.get("language")
                if isinstance(event.get("language"), str)
                else None
            ),
            locale=locale,
            owner_user_id=principal.user_id,
            site_id=principal.site_id,
            auth_session_id=principal.session_id,
            speech_session_id=speech_session_id.strip(),
            utterance_id=utterance_id.strip(),
        )
        return {"transcriptionId": record["transcriptionId"]}

    app.include_router(
        create_speech_router(
            realtime_speech,
            authorize=authorize_realtime_speech,
            issue_final_transcript=issue_realtime_transcription,
            path="/api/v2/speech/stream",
        )
    )

    def realtime_speech_capabilities() -> dict[str, Any]:
        asr = provider_capability(realtime_speech.asr_provider)
        tts = provider_capability(realtime_speech.tts_provider)
        available = bool(asr.get("available")) and bool(tts.get("available"))
        return {
            "status": "ready" if available else "unavailable",
            "protocol": "speech-stream.v2",
            "asr": asr,
            "tts": tts,
        }

    @app.get(
        "/api/v2/speech/healthz",
        dependencies=[Depends(require_speech)],
    )
    def realtime_speech_health() -> dict[str, Any]:
        capabilities = realtime_speech_capabilities()
        return {**capabilities, "status": "ok"}

    @app.get(
        "/api/v2/speech/readyz",
        dependencies=[Depends(require_speech)],
    )
    def realtime_speech_readiness() -> Response:
        capabilities = realtime_speech_capabilities()
        return JSONResponse(
            capabilities,
            status_code=200 if capabilities["status"] == "ready" else 503,
            headers={"Cache-Control": "no-store"},
        )

    @app.post(
        "/api/v2/speech/tts",
        dependencies=[Depends(require_speech)],
        response_class=StreamingResponse,
    )
    async def stream_voice_reply(body: dict[str, Any]) -> StreamingResponse:
        text = body.get("text")
        if not isinstance(text, str) or not text.strip() or len(text.strip()) > 4_000:
            raise HTTPException(status_code=422, detail="text must contain 1-4000 characters")
        locale = body.get("locale", "zh-CN")
        try:
            speech_config = AudioStreamConfig.from_start_message(
                {"type": "session.start", "locale": locale}
            )
            stream = await realtime_speech.tts_provider.synthesize(
                text.strip(), locale=speech_config.locale
            )
        except (InvalidClientMessage, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SpeechGatewayError as exc:
            status_code = 503 if exc.code == "TTS_UNAVAILABLE" else 502
            if exc.code == "TTS_TIMEOUT":
                status_code = 504
            raise HTTPException(status_code=status_code, detail=exc.message) from exc

        async def audio_chunks():
            try:
                async for chunk in stream.chunks():
                    yield chunk.audio
            finally:
                await stream.close()

        return StreamingResponse(
            audio_chunks(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "X-Content-Type-Options": "nosniff",
                "X-Speech-Stream-Id": stream.stream_id,
                "X-Speech-Audio-Encoding": "pcm_s16le",
                "X-Speech-Sample-Rate-Hz": "24000",
                "X-Speech-Channels": "1",
            },
        )

    def bind_principal_resource(
        principal: Principal,
        resource_type: str,
        resource_id: str,
    ) -> None:
        if principal.is_legacy:
            return
        context.store.bind_resource_owner(
            resource_type,
            resource_id,
            owner_user_id=principal.user_id,
            site_id=principal.site_id,
        )

    def assistant_resource_visible(
        principal: Principal,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        binding = context.store.get_resource_owner(resource_type, resource_id)
        try:
            auth_service.policy.require(
                principal,
                Scope.ASSISTANT_USE,
                owner_user_id=None if binding is None else str(binding["ownerUserId"]),
                site_id=None if binding is None else str(binding["siteId"]),
                resource_is_bound=binding is not None,
            )
            return True
        except AuthorizationDenied:
            return False

    def require_conversation_resource(
        request: Request,
        principal: Principal,
        conversation_id: str,
    ) -> None:
        require_resource_access(
            request,
            principal,
            Scope.ASSISTANT_USE,
            resource_type="assistant_conversation",
            resource_id=conversation_id,
        )

    def require_turn_resource(
        request: Request,
        principal: Principal,
        task_id: str,
    ) -> None:
        turn = context.store.get_assistant_turn(task_id)
        if turn is None:
            raise HTTPException(status_code=404, detail="assistant turn not found")
        require_conversation_resource(request, principal, str(turn["conversationId"]))

    def require_draft_resource(
        request: Request,
        principal: Principal,
        draft_id: str,
    ) -> None:
        metadata = context.store.get_assistant_draft_review(draft_id)
        if metadata is not None:
            require_conversation_resource(
                request,
                principal,
                str(metadata["conversationId"]),
            )
            return
        require_resource_access(
            request,
            principal,
            Scope.ASSISTANT_USE,
            resource_type="assistant_draft",
            resource_id=draft_id,
        )

    def require_visual_mission_resource(
        request: Request,
        principal: Principal,
        mission_id: str,
    ) -> None:
        mission = context.store.get_visual_mission(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="visual mission not found")
        require_draft_resource(request, principal, str(mission["draftId"]))

    def get_context(request: Request) -> _ControlContext:
        return request.app.state.control

    def get_assistant_service(request: Request) -> AssistantService:
        return request.app.state.assistant

    def get_mission_service(request: Request) -> MissionService:
        return request.app.state.mission

    def get_visual_runtime(request: Request) -> VisualMissionRuntime:
        return request.app.state.visual_mission

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
        idempotency_key = pick(
            "idempotencyKey",
            "idempotency_key",
            default=_new_id(idempotency_prefix),
        )
        # Robot Agent's frozen wire protocol requires a UUID command ID.  Base
        # generated IDs on the durable idempotency key so rebuilding the same
        # request cannot silently acquire a different transport identity.
        generated_command_id = (
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"h-ide:robot-agent:{idempotency_key}"))
            if mapping.transport is Transport.ROBOT_AGENT
            else _new_id("cmd")
        )
        data = {
            "commandId": command_id or pick(
                "commandId", "command_id", default=generated_command_id
            ),
            "robotId": pick("robotId", "robot_id"),
            "source": pick("source", default=source.value),
            "transport": pick("transport", default=mapping.transport.value),
            "commandKey": pick("commandKey", "command_key", default=definition.commandKey),
            "commandVersion": pick("commandVersion", "command_version", "version", default=definition.commandVersion),
            "payload": pick("payload", default={}),
            "issuedAt": pick("issuedAt", "issued_at", default=now),
            "expiresAt": expires,
            "idempotencyKey": idempotency_key,
            "sessionId": pick("sessionId", "session_id", default="api-session"),
            "traceId": pick("traceId", "trace_id", default=_new_id("trace")),
        }
        try:
            return CommandRequest.model_validate(data)
        except ValidationError as exc:
            fail_validation(exc)
            raise AssertionError("unreachable")

    def _dispatch_and_save(context: _ControlContext, request: CommandRequest, definition: CommandDefinition, mapping: MappingVersion) -> CommandReceipt:
        def persist_fake_pose(receipt: CommandReceipt) -> None:
            if (
                receipt.status.value != "succeeded"
                or request.commandKey != NAVIGATION_COMMAND_KEY
                or mapping.transport is not Transport.FAKE
            ):
                return
            goal = PoseMap.model_validate(request.payload.get("goalPose"))
            observed = goal.model_copy(update={"observedAt": utc_now()})
            current = context.store.get_robot(request.robotId)
            if current is None:
                raise RuntimeError("robot disappeared after fake navigation")
            context.store.upsert_robot(
                RobotState.model_validate(current).model_copy(update={"pose": observed})
            )

        claim = context.store.claim_command(request)
        if not claim.owned:
            if claim.conflict:
                raise _CommandIdentityConflict("commandId is already bound to another idempotencyKey")
            if not claim.record:
                raise RuntimeError("idempotency claim lost without a durable record")
            try:
                receipt = CommandReceipt.model_validate(claim.record["receipt"])
                persist_fake_pose(receipt)
                return receipt
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
        persist_fake_pose(receipt)
        return receipt

    def _save_mapping_change(
        context: _ControlContext,
        record: MappingRecord,
        previous: MappingVersion | None = None,
        *,
        audit_events: list[Mapping[str, Any]] | None = None,
        expected_probes: Mapping[tuple[str, str], Any] | None = None,
    ) -> None:
        records = [record]
        if previous is not None and (previous.mappingId, previous.version) != (record.mapping.mappingId, record.version):
            try:
                records.insert(0, context.registry.get(previous.mappingId, previous.version))
            except ValueError as exc:
                raise MappingStateConflict(
                    previous.mappingId,
                    previous.version,
                    "missing",
                    "superseded",
                ) from exc
        active = record.mapping if record.status == "active" else None
        kwargs: dict[str, Any] = {}
        if active is not None:
            kwargs["expected_active"] = None if previous is None else (previous.mappingId, previous.version)
        if audit_events is not None:
            kwargs["audit_events"] = audit_events
        if expected_probes is not None:
            kwargs["expected_probes"] = expected_probes
        context.store.save_mapping_batch(records, active=active, **kwargs)

    @app.get("/api/v1/robots", dependencies=[Depends(require_robots_read)])
    def list_robots(context: _ControlContext = Depends(get_context)):
        refresh_robot_agent_snapshot()
        if simulation_gateway_driver is not None:
            refresh_robot_agent_snapshot(simulation_robot_id)
        return context.store.list_robots()

    @app.get("/api/v1/robots/{robot_id}/capabilities", dependencies=[Depends(require_robots_read)])
    def get_capabilities(robot_id: str, context: _ControlContext = Depends(get_context)):
        refresh_robot_agent_snapshot(robot_id)
        value = context.store.get_capabilities(robot_id)
        if value is None:
            raise HTTPException(status_code=404, detail="robot capabilities not found")
        return value

    @app.get("/api/v1/assistant/conversations", dependencies=[Depends(require_assistant)])
    def list_assistant_conversations(
        includeArchived: bool = False,
        includeDeleted: bool = False,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        return [
            conversation
            for conversation in service.list_conversations(
                include_archived=includeArchived,
                include_deleted=includeDeleted,
            )
            if assistant_resource_visible(
                principal,
                "assistant_conversation",
                str(conversation["conversationId"]),
            )
        ]

    @app.post(
        "/api/v1/assistant/conversations",
        dependencies=[Depends(require_assistant)],
        status_code=201,
    )
    def create_assistant_conversation(
        body: dict[str, Any],
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            payload = dict(body)
            if not principal.is_legacy:
                payload["requestedBy"] = principal.user_id
            robot_id = payload.get("robotId")
            if isinstance(robot_id, str):
                refresh_robot_agent_snapshot(robot_id)
            conversation = service.create_conversation(payload)
            bind_principal_resource(
                principal,
                "assistant_conversation",
                str(conversation["conversationId"]),
            )
            return conversation
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/api/v1/assistant/conversations/{conversation_id}",
        dependencies=[Depends(require_assistant)],
    )
    def get_assistant_conversation(
        conversation_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_conversation_resource(request, principal, conversation_id)
            return service.get_conversation(conversation_id)
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.patch(
        "/api/v1/assistant/conversations/{conversation_id}",
        dependencies=[Depends(require_assistant)],
    )
    def update_assistant_conversation(
        conversation_id: str,
        body: dict[str, Any],
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_conversation_resource(request, principal, conversation_id)
            return service.update_conversation(conversation_id, body)
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AssistantConversationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete(
        "/api/v1/assistant/conversations/{conversation_id}",
        dependencies=[Depends(require_assistant)],
    )
    def delete_assistant_conversation(
        conversation_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_conversation_resource(request, principal, conversation_id)
            return service.delete_conversation(conversation_id)
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AssistantConversationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/v1/assistant/conversations/{conversation_id}/messages",
        dependencies=[Depends(require_assistant)],
        status_code=202,
    )
    def submit_assistant_message(
        conversation_id: str,
        body: dict[str, Any],
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_conversation_resource(request, principal, conversation_id)
            conversation = service.store.get_assistant_conversation(conversation_id)
            if (
                service.conversation_gateway is None
                and isinstance(conversation, Mapping)
                and isinstance(conversation.get("robotId"), str)
            ):
                refresh_robot_agent_snapshot(conversation["robotId"])
            return service.submit_message(
                conversation_id,
                body,
                caller_user_id=principal.user_id,
                caller_site_id=principal.site_id,
                caller_role=principal.role.value,
                caller_is_legacy=principal.is_legacy,
            )
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AssistantTurnConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AssistantConversationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SceneContextConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SceneContextUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/v1/assistant/turns/{task_id}", dependencies=[Depends(require_assistant)])
    def get_assistant_turn(
        task_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_turn_resource(request, principal, task_id)
            return service.get_turn(task_id)
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get(
        "/api/v1/assistant/turns/{task_id}/events",
        dependencies=[Depends(require_assistant)],
    )
    def get_assistant_turn_events(
        task_id: str,
        request: Request,
        afterSequence: int = 0,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_turn_resource(request, principal, task_id)
            if afterSequence < 0:
                raise ValueError("afterSequence cannot be negative")
            events = service.get_turn_events(task_id, after_sequence=afterSequence)
            return {
                "schemaVersion": "h-ide.assistant-events/2",
                "taskId": task_id,
                "events": events,
                "lastSequence": events[-1]["sequence"] if events else afterSequence,
            }
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/api/v1/assistant/turns/{task_id}/visual-evidence/{analysis_id}/{sha256}",
        dependencies=[Depends(require_assistant)],
    )
    def get_assistant_turn_visual_evidence(
        task_id: str,
        analysis_id: str,
        sha256: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ) -> Response:
        require_turn_resource(request, principal, task_id)
        turn = service.get_turn(task_id)
        result = turn.get("result")
        visual = result.get("visualResult") if isinstance(result, Mapping) else None
        if (
            not isinstance(visual, Mapping)
            or visual.get("analysisId") != analysis_id
        ):
            raise HTTPException(status_code=404, detail="assistant visual evidence not found")
        evidence = visual.get("evidence")
        if not isinstance(evidence, list) or not any(
            isinstance(item, Mapping) and item.get("sha256") == sha256
            for item in evidence
        ):
            raise HTTPException(status_code=404, detail="assistant visual evidence not found")
        try:
            content, media_type = visual_task_adapter.get_evidence(analysis_id, sha256)
        except VisualTaskUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "Pragma": "no-cache",
                "Content-Security-Policy": "default-src 'none'; sandbox",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post(
        "/api/v1/assistant/turns/{task_id}/cancel",
        dependencies=[Depends(require_assistant)],
    )
    def cancel_assistant_turn(
        task_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_turn_resource(request, principal, task_id)
            return service.cancel_turn(task_id)
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post(
        "/api/v1/assistant/drafts/{draft_id}/reject",
        dependencies=[Depends(require_assistant)],
    )
    def reject_assistant_draft(
        draft_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_draft_resource(request, principal, draft_id)
            return service.reject_draft(draft_id, body or {})
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AssistantTurnConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/v1/assistant/drafts/{draft_id}/revise",
        dependencies=[Depends(require_assistant)],
        status_code=202,
    )
    def revise_assistant_draft(
        draft_id: str,
        body: dict[str, Any],
        request: Request,
        principal: Principal = Depends(require_assistant),
        service: AssistantService = Depends(get_assistant_service),
    ):
        try:
            require_draft_resource(request, principal, draft_id)
            return service.revise_draft(draft_id, body)
        except AssistantNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AssistantTurnConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SceneContextConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SceneContextUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/v1/speech/transcriptions", dependencies=[Depends(require_speech)])
    async def create_speech_transcription(
        request: Request,
        file: UploadFile = File(...),
        locale: str = Form("zh-CN"),
        principal: Principal = Depends(require_speech),
        context: _ControlContext = Depends(get_context),
    ):
        filename = file.filename or "command.wav"
        media_type = (file.content_type or "").lower()
        if media_type not in {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="only PCM WAV uploads are accepted")
        wav_bytes = await file.read(MAX_WAV_BYTES + 1)
        if len(wav_bytes) > MAX_WAV_BYTES:
            raise HTTPException(status_code=413, detail="WAV upload exceeds 10 MiB")
        try:
            validate_wav(wav_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result = await run_in_threadpool(
                request.app.state.speech.transcribe,
                wav_bytes,
                filename=filename,
                locale=locale,
            )
        except SpeechServiceUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"speech service failed: {exc}") from exc
        text = result.get("text") if isinstance(result, Mapping) else None
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(status_code=502, detail="speech service returned an empty transcript")
        return context.store.save_transcription(
            _new_id("transcription"),
            text.strip(),
            language=result.get("language") if isinstance(result, Mapping) else None,
            locale=locale,
            filename=filename,
            owner_user_id=principal.user_id,
            site_id=principal.site_id,
            auth_session_id=principal.session_id,
            source="batch_wav",
        )

    @app.post("/api/v1/commands", dependencies=[Depends(require_commands)])
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
        if not isinstance(robot_id, str):
            raise HTTPException(status_code=404, detail="robot not found")
        # The Robot Agent is an outbound client and may connect after Control
        # API startup. Refresh before reading safety state or active mappings.
        refresh_robot_agent_snapshot(robot_id)
        robot = context.store.get_robot(robot_id)
        if robot is None:
            raise HTTPException(status_code=404, detail="robot not found")
        command_key = body.get("commandKey", body.get("command_key"))
        if not isinstance(command_key, str) or not command_key.strip():
            raise HTTPException(status_code=422, detail="commandKey is required")
        version = body.get("commandVersion", body.get("command_version", body.get("version", "1")))
        definition = context.definition(command_key, version)
        # Capability discovery is an execution-time fact. Never let a stale
        # mapping (or a missing real adapter) fall through to a Fake command.
        require_current_capability(robot_id, command_key, refresh=False)
        mission_owner = VisualMissionRuntime.motion_owner(robot_id)
        if mission_owner is not None and definition.kind is CommandKind.MOTION:
            raise HTTPException(
                status_code=409,
                detail=f"robot motion is owned by visual mission {mission_owner}; stop/cancel remains available",
            )
        safety_error = _motion_safety_error(robot, definition)
        if safety_error:
            raise HTTPException(status_code=409, detail=safety_error)
        mapping_record = context.mapping_record(
            command_key,
            body.get("mappingId", body.get("mapping_id")),
            body.get("mappingVersion", body.get("mapping_version")),
        )
        mapping = mapping_record.mapping
        if mapping.transport is Transport.ROBOT_AGENT:
            require_robot_agent_capability(robot_id, command_key)
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

    @app.get("/api/v1/commands", dependencies=[Depends(require_commands)])
    def list_commands(context: _ControlContext = Depends(get_context)):
        return context.store.list_commands()

    @app.get("/api/v1/commands/{command_id}", dependencies=[Depends(require_commands)])
    def get_command(command_id: str, context: _ControlContext = Depends(get_context)):
        if command_id in context.runtime._inflight:
            try:
                refreshed = context.runtime.reconcile(command_id)
            except ValueError:
                refreshed = None
            if refreshed is not None:
                request = context.runtime._requests.get(command_id)
                if request is not None:
                    context.store.save_command(request, refreshed)
        receipt = context.store.get_command(command_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail="command not found")
        return receipt

    @app.post("/api/v1/commands/{command_id}/cancel", dependencies=[Depends(require_commands)])
    def cancel_command(command_id: str, context: _ControlContext = Depends(get_context)):
        persisted_record = context.store.get_command_record(command_id)
        if persisted_record is None:
            raise HTTPException(status_code=404, detail="command not found")
        persisted_receipt = persisted_record["receipt"]
        persisted_status = str(persisted_receipt.get("status", "")).lower()
        if persisted_status in {
            "succeeded",
            "failed",
            "cancelled",
            "canceled",
            "expired",
            "rejected",
            "unknown",
        }:
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

    @app.post("/api/v1/mappings/drafts", dependencies=[Depends(require_mappings)])
    def create_mapping_draft(body: dict[str, Any], context: _ControlContext = Depends(get_context)):
        # Parsing is side-effect free; registry insertion and durable save are
        # deliberately enclosed by the same mutation lock as lifecycle
        # operations so a draft cannot race a validate/probe/activate snapshot.
        try:
            mapping_data = dict(body)
            explicit_targets = [mapping_data.pop(name) for name in ("targetId", "target_id") if name in mapping_data]
            if len({str(value).strip() for value in explicit_targets}) > 1:
                raise ValueError("target aliases conflict")
            target_id = explicit_targets[0] if explicit_targets else None
            mapping = MappingVersion.model_validate(mapping_data)
        except (ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with mapping_mutation_lock:
            snapshot = context.registry.snapshot_state()
            audit_events: list[Mapping[str, Any]] = []
            try:
                with context.registry.defer_audits() as audit_events:
                    record = context.registry.create_draft(mapping, target_id=target_id)
                    context.registry.drain_deferred_audits()
                    _save_mapping_change(context, record, audit_events=audit_events)
                return _mapping_json(record)
            except (MappingPointerConflict, MappingStateConflict) as exc:
                context.registry.restore_state(snapshot)
                try:
                    context.reload_mappings()
                except Exception:
                    # Keep the pre-transition snapshot if reconciliation itself
                    # is unavailable; the durable write was already rejected.
                    context.registry.restore_state(snapshot)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ValueError as exc:
                context.registry.restore_state(snapshot)
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except Exception as exc:
                context.registry.restore_state(snapshot)
                raise HTTPException(status_code=503, detail="mapping persistence failed; draft was not committed") from exc

    @app.get("/api/v1/mappings", dependencies=[Depends(require_mappings)])
    def list_mappings(context: _ControlContext = Depends(get_context)):
        return [
            row
            for row in context.store.list_mappings()
            if isinstance(row.get("mapping"), Mapping)
            and row["mapping"].get("transport") in context.adapters
        ]

    @app.get("/api/v1/mappings/active/{command_key}", dependencies=[Depends(require_mappings)])
    def get_active_mapping(command_key: str, context: _ControlContext = Depends(get_context)):
        return _plain(context.mapping(command_key))

    def lifecycle_action(action: str):
        def route(mapping_id: str, version: str, context: _ControlContext = Depends(get_context)):
            with mapping_mutation_lock:
                snapshot = context.registry.snapshot_state()
                audit_events: list[Mapping[str, Any]] = []
                try:
                    previous = None
                    expected_probes = None
                    if action == "probe":
                        # Capture the durable probe snapshot before invoking
                        # an external adapter.  The store checks this token in
                        # the same transaction as the new result.
                        durable = context.store.get_mapping(mapping_id, version)
                        expected_probes = {
                            (mapping_id, version): None if durable is None else durable.get("probe")
                        }
                    if action == "activate":
                        current = context.registry.get(mapping_id, version).mapping
                        try:
                            previous = context.registry.active(current.commandKey)
                        except ValueError:
                            previous = None
                    method = getattr(context.registry, action)
                    with context.registry.defer_audits() as audit_events:
                        record = method(mapping_id, version)
                        context.registry.drain_deferred_audits()
                        _save_mapping_change(
                            context,
                            record,
                            previous,
                            audit_events=audit_events,
                            expected_probes=expected_probes,
                        )
                    return _mapping_json(record)
                except (MappingPointerConflict, MappingStateConflict) as exc:
                    context.registry.restore_state(snapshot)
                    try:
                        context.reload_mappings()
                    except Exception:
                        context.registry.restore_state(snapshot)
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                except ValueError as exc:
                    context.registry.restore_state(snapshot)
                    # A failed probe does not mutate its durable mapping row,
                    # but its diagnostic audit remains useful.  Successful
                    # lifecycle audits stay staged and are discarded on any
                    # persistence failure.
                    for event in audit_events:
                        if event.get("action") == "probe_failed":
                            context.store.record_audit(event)
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                except Exception as exc:
                    # Do not report a successful lifecycle transition when the
                    # durable batch failed; restore in-memory indexes to the
                    # last committed snapshot and require retry/reconciliation.
                    context.registry.restore_state(snapshot)
                    raise HTTPException(status_code=503, detail="mapping persistence failed; transition was not committed") from exc

        return route

    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/validate", lifecycle_action("validate"), methods=["POST"], dependencies=[Depends(require_mappings)])
    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/probe", lifecycle_action("probe"), methods=["POST"], dependencies=[Depends(require_mappings)])
    app.add_api_route("/api/v1/mappings/{mapping_id}/{version}/activate", lifecycle_action("activate"), methods=["POST"], dependencies=[Depends(require_mappings)])

    @app.post("/api/v1/mappings/{command_key}/rollback", dependencies=[Depends(require_mappings)])
    def rollback_mapping(command_key: str, body: dict[str, Any] | None = None, context: _ControlContext = Depends(get_context)):
        version = None if not body else body.get("version")
        with mapping_mutation_lock:
            snapshot = context.registry.snapshot_state()
            audit_events: list[Mapping[str, Any]] = []
            try:
                try:
                    previous = context.registry.active(command_key)
                except ValueError:
                    previous = None
                with context.registry.defer_audits() as audit_events:
                    record = context.registry.rollback(command_key, version=version)
                    context.registry.drain_deferred_audits()
                    _save_mapping_change(context, record, previous, audit_events=audit_events)
                return _mapping_json(record)
            except (MappingPointerConflict, MappingStateConflict) as exc:
                context.registry.restore_state(snapshot)
                try:
                    context.reload_mappings()
                except Exception:
                    context.registry.restore_state(snapshot)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ValueError as exc:
                context.registry.restore_state(snapshot)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                context.registry.restore_state(snapshot)
                raise HTTPException(status_code=503, detail="mapping persistence failed; transition was not committed") from exc

    @app.post("/api/v1/assistant/drafts", dependencies=[Depends(require_assistant)])
    def create_assistant_draft(
        body: dict[str, Any],
        principal: Principal = Depends(require_assistant),
        context: _ControlContext = Depends(get_context),
    ):
        draft_id = body.get("draftId", body.get("draft_id")) or body.get("planId", body.get("plan_id")) or _new_id("draft")
        plan_data = dict(body)
        plan_data.pop("draftId", None)
        plan_data.pop("draft_id", None)
        plan_data.setdefault("planId", draft_id)
        try:
            plan = parse_plan_draft(plan_data)
        except ValidationError as exc:
            fail_validation(exc)
            raise AssertionError("unreachable")
        if not context.store.save_assistant_draft(draft_id, _plain(plan), status="draft"):
            raise HTTPException(status_code=409, detail="assistant draft already exists")
        bind_principal_resource(principal, "assistant_draft", str(draft_id))
        return {"draftId": draft_id, "status": "draft", "plan": _plain(plan)}

    @app.get("/api/v1/assistant/drafts/{draft_id}", dependencies=[Depends(require_assistant)])
    def get_assistant_draft(
        draft_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        context: _ControlContext = Depends(get_context),
    ):
        require_draft_resource(request, principal, draft_id)
        draft = context.store.get_assistant_draft(draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="assistant draft not found")
        metadata = context.store.get_assistant_draft_review(draft_id)
        if metadata is not None:
            draft["review"] = metadata["review"]
            draft["revisionOfDraftId"] = metadata["revisionOfDraftId"]
        return draft

    @app.post("/api/v1/assistant/drafts/{draft_id}/confirm", dependencies=[Depends(require_assistant)])
    def confirm_assistant_draft(
        draft_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(require_assistant),
        context: _ControlContext = Depends(get_context),
    ):
        require_draft_resource(request, principal, draft_id)
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
            context.store.set_assistant_conversation_status_for_draft(draft_id, "failed")
            raise HTTPException(status_code=409, detail="assistant plan has expired")
        try:
            parsed_plan = parse_plan_draft(stored["plan"])
        except ValidationError as exc:
            fail_validation(exc)
            raise AssertionError("unreachable")
        if isinstance(parsed_plan, VisualMissionPlanDraft):
            if not context.store.claim_assistant_draft(draft_id):
                raise HTTPException(status_code=409, detail="assistant draft is already being confirmed")
            context.store.set_assistant_conversation_status_for_draft(draft_id, "executing")
            runtime = app.state.visual_mission
            try:
                mission = runtime.start(parsed_plan, draft_id=draft_id)
            except ValueError as exc:
                context.store.set_assistant_draft_status(
                    draft_id,
                    "failed",
                    expected_status="confirming",
                    error=str(exc),
                )
                context.store.set_assistant_conversation_status_for_draft(draft_id, "failed")
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            if not context.store.set_assistant_draft_status(
                draft_id,
                "confirmed",
                expected_status="confirming",
                execution={"missionId": mission["missionId"], "status": mission["status"]},
            ):
                raise HTTPException(status_code=409, detail="assistant draft state changed during confirmation")
            context.store.set_assistant_conversation_status_for_draft(draft_id, "executing")
            return {
                "draftId": draft_id,
                "status": "confirmed",
                "mission": mission,
                "receipts": [],
            }
        if not context.store.claim_assistant_draft(draft_id):
            raise HTTPException(status_code=409, detail="assistant draft is already being confirmed")
        context.store.set_assistant_conversation_status_for_draft(draft_id, "executing")

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
            context.store.set_assistant_conversation_status_for_draft(draft_id, "failed")
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
            refresh_robot_agent_snapshot(plan.robotId)
            robot = context.store.get_robot(plan.robotId)
            if robot is None:
                raise HTTPException(status_code=404, detail="robot not found")
            robot_state = RobotState.model_validate(robot)
            planned_scene_context = _draft_scene_context(context.store, draft_id)
            navigation_steps = [
                step for step in ordered if step.commandKey == NAVIGATION_COMMAND_KEY
            ]
            latest_scene_context = None
            if navigation_steps:
                if planned_scene_context is None:
                    raise HTTPException(
                        status_code=409,
                        detail="navigation draft has no authoritative planning scene context",
                    )
                try:
                    latest_scene_context = context.scene_context_client.get_context(
                        plan.robotId,
                        current_pose=robot_state.pose,
                        expected_ref=planned_scene_context.sceneRef,
                        selected_point=planned_scene_context.selectedPoint,
                    )
                except (SceneContextConflict, SceneContextUnavailable) as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if latest_scene_context is None:
                    raise HTTPException(
                        status_code=409,
                        detail="published robot scene binding is no longer available",
                    )
            capability = context.store.get_capabilities(plan.robotId)
            current_capabilities = (
                capability.get("capabilities") if isinstance(capability, Mapping) else None
            )
            if not isinstance(current_capabilities, list):
                raise HTTPException(
                    status_code=409,
                    detail="current robot capability snapshot is unavailable",
                )
            available_commands = {
                item for item in current_capabilities if isinstance(item, str)
            }
            resolved: list[tuple[Any, CommandDefinition, MappingVersion, Mapping[str, Any]]] = []
            for step in ordered:
                definition = context.definition(step.commandKey, "1")
                mapping = context.mapping(step.commandKey)
                if step.commandKey not in available_commands:
                    raise HTTPException(
                        status_code=409,
                        detail=f"current robot capability is missing: {step.commandKey}",
                    )
                safety_error = _motion_safety_error(robot, definition)
                if safety_error:
                    raise HTTPException(status_code=409, detail=safety_error)
                try:
                    target = _mapping_target_binding(context.mapping_record(step.commandKey))
                except ValueError as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                if target is not None and target != plan.robotId:
                    raise HTTPException(status_code=409, detail="mapping target does not match robotId")
                if step.commandKey == NAVIGATION_COMMAND_KEY:
                    try:
                        validate_navigation_payload(planned_scene_context, step.payload)
                        validate_navigation_payload(
                            latest_scene_context,
                            step.payload,
                            translation_threshold_m=spatial_translation_drift,
                            yaw_threshold_rad=spatial_yaw_drift,
                        )
                    except (TypeError, ValueError, ValidationError) as exc:
                        raise HTTPException(
                            status_code=409,
                            detail=f"navigation plan is stale or invalid: {exc}",
                        ) from exc
                execution_payload: Mapping[str, Any] = step.payload
                if (
                    step.commandKey == NAVIGATION_COMMAND_KEY
                    and mapping.transport is Transport.ROBOT_AGENT
                ):
                    try:
                        execution_payload = build_spatial_navigation_execution_payload(
                            step.payload,
                            robot_state.details,
                            mission_id=draft_id,
                            stage_id=step.stepId,
                        ).model_dump(mode="json")
                    except (TypeError, ValueError, ValidationError) as exc:
                        raise HTTPException(
                            status_code=409,
                            detail=f"current Robot Agent observation binding is unavailable: {exc}",
                        ) from exc
                resolved.append((step, definition, mapping, execution_payload))
            for step, definition, mapping, execution_payload in resolved:
                request = _build_request(
                    {
                        "robotId": plan.robotId,
                        "commandKey": step.commandKey,
                        "commandVersion": definition.commandVersion,
                        "transport": mapping.transport.value,
                        "payload": execution_payload,
                        # A PlanDraft may remain reviewable for many minutes,
                        # while robot-agent.v1 caps one command TTL at 60 s.
                        # Confirmation starts a fresh bounded execution window.
                        "expiresAt": min(
                            plan.expiresAt or (utc_now() + timedelta(seconds=30)),
                            utc_now() + timedelta(seconds=30),
                        ),
                        "idempotencyKey": f"{draft_id}:{step.stepId}",
                        "sessionId": draft_id,
                    },
                    definition,
                    mapping,
                    source=CommandSource.AGENT,
                    idempotency_prefix=f"{draft_id}-{step.stepId}",
                )
                receipt = _dispatch_and_save(context, request, definition, mapping)
                if (
                    mapping.transport is Transport.ROBOT_AGENT
                    and receipt.status.value in {"accepted", "running"}
                ):
                    deadline = time.monotonic() + assistant_command_wait_seconds
                    while (
                        receipt.status.value in {"accepted", "running"}
                        and time.monotonic() < deadline
                    ):
                        time.sleep(assistant_command_poll_seconds)
                        receipt = context.runtime.reconcile(request.commandId)
                        context.store.save_command(request, receipt)
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
            context.store.set_assistant_conversation_status_for_draft(draft_id, "completed")
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

    @app.get("/api/v1/visual-missions/{mission_id}", dependencies=[Depends(require_assistant)])
    def get_visual_mission(
        mission_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        runtime: VisualMissionRuntime = Depends(get_visual_runtime),
    ):
        require_visual_mission_resource(request, principal, mission_id)
        mission = runtime.get(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="visual mission not found")
        return mission

    @app.post("/api/v1/visual-missions/{mission_id}/run", dependencies=[Depends(require_assistant)])
    def run_visual_mission(
        mission_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        runtime: VisualMissionRuntime = Depends(get_visual_runtime),
    ):
        try:
            require_visual_mission_resource(request, principal, mission_id)
            mission = runtime.get(mission_id)
            if mission is None:
                raise ValueError("visual mission not found")
            require_current_capability(mission["robotId"], NAVIGATION_COMMAND_KEY)
            robot = context.store.get_robot(mission["robotId"])
            if robot is None:
                raise HTTPException(status_code=404, detail="robot not found")
            plan = VisualMissionPlanDraft.model_validate(mission["plan"])
            current_details = robot.get("details")
            if not isinstance(current_details, Mapping):
                raise HTTPException(
                    status_code=409,
                    detail="current robot observation identity is unavailable",
                )
            current_boot_id = current_details.get("robotBootId")
            current_epoch_id = current_details.get("localizationEpochId")
            if not isinstance(current_boot_id, str) or not current_boot_id:
                raise HTTPException(
                    status_code=409,
                    detail="current robotBootId is unavailable",
                )
            if not isinstance(current_epoch_id, str) or not current_epoch_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "current localizationEpochId is unavailable; configure "
                        "ROBOT_AGENT_LOCALIZATION_EPOCH_ID"
                    ),
                )
            if plan.mission.robotBootId != current_boot_id:
                raise HTTPException(
                    status_code=409,
                    detail="visual mission robotBootId does not match current robot state",
                )
            if plan.mission.localizationEpochId != current_epoch_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "visual mission localizationEpochId does not match current robot state"
                    ),
                )
            safety_error = _motion_safety_error(
                robot,
                context.definition(NAVIGATION_COMMAND_KEY, "1"),
            )
            if safety_error:
                raise HTTPException(status_code=409, detail=safety_error)
            return runtime.run(mission_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/visual-missions/{mission_id}/cancel", dependencies=[Depends(require_assistant)])
    def cancel_visual_mission(
        mission_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        runtime: VisualMissionRuntime = Depends(get_visual_runtime),
    ):
        try:
            require_visual_mission_resource(request, principal, mission_id)
            return runtime.cancel(mission_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post(
        "/api/v1/visual-missions/{mission_id}/candidates/{candidate_id}",
        dependencies=[Depends(require_assistant)],
    )
    def select_visual_mission_candidate(
        mission_id: str,
        candidate_id: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        runtime: VisualMissionRuntime = Depends(get_visual_runtime),
    ):
        try:
            require_visual_mission_resource(request, principal, mission_id)
            return runtime.select_candidate(mission_id, candidate_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get(
        "/api/v1/visual-missions/{mission_id}/evidence/{sha256}",
        dependencies=[Depends(require_assistant)],
        response_class=Response,
    )
    def get_visual_mission_evidence(
        mission_id: str,
        sha256: str,
        request: Request,
        principal: Principal = Depends(require_assistant),
        runtime: VisualMissionRuntime = Depends(get_visual_runtime),
    ) -> Response:
        require_visual_mission_resource(request, principal, mission_id)
        mission = runtime.get(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="visual mission not found")
        bound_resolution_id = None
        for stage in mission["stages"]:
            refs = stage.get("state", {}).get("evidence", [])
            if any(isinstance(ref, Mapping) and ref.get("sha256") == sha256 for ref in refs):
                bound_resolution_id = stage.get("visualResolutionId")
                break
        if not bound_resolution_id:
            raise HTTPException(status_code=404, detail="evidence is not bound to visual mission")
        try:
            body, media_type = runtime.vision_client.get_evidence(bound_resolution_id, sha256)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return Response(
            content=body,
            media_type=media_type,
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "Content-Security-Policy": "default-src 'none'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    # Singular aliases keep clients that model a single assistant draft or
    # mapping draft compatible without duplicating execution logic.
    app.add_api_route("/api/v1/mappings", create_mapping_draft, methods=["POST"], dependencies=[Depends(require_mappings)], include_in_schema=False)
    app.add_api_route("/api/v1/assistant/draft", create_assistant_draft, methods=["POST"], dependencies=[Depends(require_assistant)], include_in_schema=False)
    app.add_api_route("/api/v1/assistant/draft/{draft_id}", get_assistant_draft, methods=["GET"], dependencies=[Depends(require_assistant)], include_in_schema=False)
    app.add_api_route("/api/v1/assistant/draft/{draft_id}/confirm", confirm_assistant_draft, methods=["POST"], dependencies=[Depends(require_assistant)], include_in_schema=False)

    @app.get("/api/v1/events", dependencies=[Depends(require_events)])
    def list_events(after: int = 0, context: _ControlContext = Depends(get_context)):
        return context.store.list_events(after_id=after)

    @app.websocket("/api/v1/events")
    async def events_websocket(websocket: WebSocket):
        query_token = websocket.query_params.get("token")
        try:
            principal = auth_service.authenticate_websocket(
                authorization=websocket.headers.get("authorization"),
                session_token=websocket.cookies.get(SESSION_COOKIE),
                # Query credentials remain supported only for the legacy
                # developer token. Browser session secrets are never accepted
                # in URLs where logs and referrers can capture them.
                legacy_query_token=query_token,
            )
            auth_service.policy.require(principal, Scope.EVENTS_READ)
        except AuthorizationDenied:
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
    _mount_control_pwa(app, dist)
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
