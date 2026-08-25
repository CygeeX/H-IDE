"""Durable, observation-only DAAAM semantic-memory integration.

The API admits a post-mission DAAAM export only when it is bound to the exact
Mission, Robot Agent boot, localization epoch, and published scene revision.
Map centroids remain provenance and are never projected into Mission input.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from packages.mission_core import MissionPlanDraft
from packages.semantic_memory import DaaamMemoryImport
from packages.scene_core import SceneRef

from .auth import Principal, Role, Scope, require_resource_access, require_scope
from .store import SQLiteStore, _decode, _encode


DAAAM_MEMORY_SCOPE = Scope.SCENE_MEMORY_USE
DAAAM_IMPORT_RESOURCE = "daaam_memory_import"


class DaaamMemoryNotFound(LookupError):
    """The requested semantic-memory import does not exist."""


class DaaamMemoryConflict(ValueError):
    """The import conflicts with durable provenance or identity."""


def _normalized_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _revision_key(value: str | int) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class DaaamMemoryRepository:
    """Atomic repository for immutable DAAAM imports and their entities."""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self._connection = store._connection
        self._lock = store._lock

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _public(payload: Mapping[str, Any], *, replayed: bool = False) -> dict[str, Any]:
        return {
            **dict(payload),
            "authority": "observation_only",
            "executionEligible": False,
            "navigationDispatched": False,
            "replayed": replayed,
        }

    def _validate_mission_binding(
        self,
        body: DaaamMemoryImport,
        actor: Principal,
    ) -> tuple[str, str]:
        mission = self.store.get_mission(body.missionId)
        if mission is None:
            raise DaaamMemoryNotFound("source Mission not found")
        owner = self.store.get_resource_owner("mission", body.missionId)
        if owner is None:
            raise DaaamMemoryConflict(
                "source Mission has no authenticated owner; memory import is forbidden"
            )
        owner_user_id = str(owner["ownerUserId"])
        site_id = str(owner["siteId"])
        if not actor.is_legacy and actor.role is not Role.ADMIN:
            if actor.site_id != site_id:
                raise DaaamMemoryNotFound("source Mission not found")
        if mission.get("robotId") != body.robotId:
            raise DaaamMemoryConflict("DAAAM import is bound to another robot")
        try:
            plan = MissionPlanDraft.model_validate(mission["plan"])
        except Exception as exc:
            raise DaaamMemoryConflict("source Mission plan is invalid") from exc
        if plan.envelope.robotBootId != body.robotBootId:
            raise DaaamMemoryConflict("DAAAM import Robot Agent boot does not match")
        if plan.envelope.localizationEpochId != body.localizationEpochId:
            raise DaaamMemoryConflict("DAAAM import localization epoch does not match")
        observation_sessions: set[str] = set()
        for record in mission.get("worldRecords", []):
            if (
                isinstance(record, Mapping)
                and record.get("source") == "robot_agent"
                and record.get("predicate") == "mission_origin"
                and isinstance(record.get("sourceRef"), str)
                and record.get("sourceRef") != plan.envelope.robotBootId
            ):
                observation_sessions.add(str(record["sourceRef"]))
        for execution in mission.get("skillExecutions", []):
            if not isinstance(execution, Mapping):
                continue
            request = execution.get("request")
            request_input = request.get("input") if isinstance(request, Mapping) else None
            if isinstance(request_input, Mapping) and isinstance(
                request_input.get("observationSessionId"), str
            ):
                observation_sessions.add(str(request_input["observationSessionId"]))
            terminal = execution.get("result")
            result = terminal.get("result") if isinstance(terminal, Mapping) else None
            visual = result.get("visualResult") if isinstance(result, Mapping) else None
            source = visual.get("source") if isinstance(visual, Mapping) else None
            if isinstance(source, Mapping) and isinstance(
                source.get("observationSessionId"), str
            ):
                observation_sessions.add(str(source["observationSessionId"]))
        if body.observationSessionId not in observation_sessions:
            raise DaaamMemoryConflict(
                "DAAAM import observation session is not bound to the source Mission"
            )
        scene_identities = {
            (item.sceneId, _revision_key(item.revision))
            for item in plan.envelope.sceneRefs
        }
        if (
            body.sceneRef.sceneId,
            _revision_key(body.sceneRef.revision),
        ) not in scene_identities:
            raise DaaamMemoryConflict("DAAAM import scene revision is outside the Mission")
        allowed_regions = set(plan.envelope.allowedRegionIds)
        for entity in body.entities:
            if entity.regionId is not None and entity.regionId not in allowed_regions:
                raise DaaamMemoryConflict(
                    f"DAAAM entity region is outside the Mission: {entity.regionId}"
                )
        return owner_user_id, site_id

    def ingest(
        self,
        body: DaaamMemoryImport,
        actor: Principal,
    ) -> dict[str, Any]:
        owner_user_id, site_id = self._validate_mission_binding(body, actor)
        payload = body.model_dump(mode="json")
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    "SELECT payload_json, payload_sha256 FROM daaam_memory_imports "
                    "WHERE import_id = ?",
                    (body.importId,),
                ).fetchone()
                if existing is not None:
                    if str(existing["payload_sha256"]) != body.payloadSha256:
                        raise DaaamMemoryConflict(
                            "DAAAM import ID is already bound to different evidence"
                        )
                    self._connection.rollback()
                    return self._public(
                        _decode(existing["payload_json"]),
                        replayed=True,
                    )
                self._connection.execute(
                    "INSERT INTO daaam_memory_imports("
                    "import_id, mission_id, robot_id, owner_user_id, site_id, "
                    "robot_boot_id, localization_epoch_id, observation_session_id, "
                    "scene_id, scene_revision, captured_from, captured_through, "
                    "payload_sha256, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        body.importId,
                        body.missionId,
                        body.robotId,
                        owner_user_id,
                        site_id,
                        body.robotBootId,
                        body.localizationEpochId,
                        body.observationSessionId,
                        body.sceneRef.sceneId,
                        _revision_key(body.sceneRef.revision),
                        body.capturedFrom.isoformat(),
                        body.capturedThrough.isoformat(),
                        body.payloadSha256,
                        _encode(payload),
                        now,
                    ),
                )
                for entity in body.entities:
                    entity_payload = entity.model_dump(mode="json")
                    self._connection.execute(
                        "INSERT INTO daaam_memory_entities("
                        "import_id, entity_id, semantic_id, normalized_label, region_id, "
                        "last_observed_at, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            body.importId,
                            entity.entityId,
                            entity.semanticId,
                            _normalized_label(entity.label),
                            entity.regionId,
                            entity.lastObservedAt.isoformat(),
                            _encode(entity_payload),
                        ),
                    )
                self.store._bind_resource_owner_locked(
                    DAAAM_IMPORT_RESOURCE,
                    body.importId,
                    owner_user_id=owner_user_id,
                    site_id=site_id,
                    now=now,
                )
                self._connection.execute(
                    "INSERT INTO audits(payload_json, created_at) VALUES (?, ?)",
                    (
                        _encode(
                            {
                                "eventType": "daaam_memory.imported",
                                "actorUserId": actor.user_id,
                                "actorRole": actor.role.value,
                                "siteId": site_id,
                                "resourceType": DAAAM_IMPORT_RESOURCE,
                                "resourceId": body.importId,
                                "missionId": body.missionId,
                                "robotId": body.robotId,
                                "payloadSha256": body.payloadSha256,
                                "entityCount": len(body.entities),
                                "authority": "observation_only",
                                "executionEligible": False,
                                "navigationDispatched": False,
                            }
                        ),
                        now,
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self._public(payload)

    def get_import(self, import_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM daaam_memory_imports WHERE import_id = ?",
                (import_id,),
            ).fetchone()
        if row is None:
            raise DaaamMemoryNotFound("DAAAM import not found")
        return self._public(_decode(row["payload_json"]))

    def list_imports(
        self,
        *,
        site_id: str | None,
        owner_user_id: str | None,
        robot_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        if site_id is not None:
            clauses.append("site_id = ?")
            parameters.append(site_id)
        if owner_user_id is not None:
            clauses.append("owner_user_id = ?")
            parameters.append(owner_user_id)
        if robot_id is not None:
            clauses.append("robot_id = ?")
            parameters.append(robot_id)
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        parameters.append(max(1, min(int(limit), 500)))
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM daaam_memory_imports"
                f"{where} ORDER BY captured_through DESC, import_id DESC LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return [self._public(_decode(row["payload_json"])) for row in rows]

    def list_entities(self, import_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM daaam_memory_entities "
                "WHERE import_id = ? ORDER BY semantic_id, entity_id",
                (import_id,),
            ).fetchall()
            exists = self._connection.execute(
                "SELECT 1 FROM daaam_memory_imports WHERE import_id = ?",
                (import_id,),
            ).fetchone()
        if not rows and exists is None:
            raise DaaamMemoryNotFound("DAAAM import not found")
        return [
            {
                **_decode(row["payload_json"]),
                "authority": "observation_only",
                "executionEligible": False,
            }
            for row in rows
        ]

    def mission_context_entities(
        self,
        *,
        owner_user_id: str,
        site_id: str,
        robot_id: str,
        scene_ref: SceneRef,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return latest region-bound hypotheses, never executable centroids."""

        with self._lock:
            rows = self._connection.execute(
                "WITH ranked AS ("
                "SELECT i.import_id, i.payload_sha256, i.captured_through, "
                "e.normalized_label, e.region_id, e.payload_json, "
                "ROW_NUMBER() OVER ("
                "PARTITION BY e.normalized_label, e.region_id "
                "ORDER BY i.captured_through DESC, i.import_id DESC, e.entity_id"
                ") AS memory_rank "
                "FROM daaam_memory_entities e "
                "JOIN daaam_memory_imports i ON i.import_id = e.import_id "
                "WHERE i.owner_user_id = ? AND i.site_id = ? AND i.robot_id = ? "
                "AND i.scene_id = ? AND i.scene_revision = ? "
                "AND e.region_id IS NOT NULL"
                ") "
                "SELECT import_id, payload_sha256, captured_through, "
                "normalized_label, region_id, payload_json FROM ranked "
                "WHERE memory_rank = 1 "
                "ORDER BY captured_through DESC, import_id DESC, normalized_label "
                "LIMIT ?",
                (
                    owner_user_id,
                    site_id,
                    robot_id,
                    scene_ref.sceneId,
                    _revision_key(scene_ref.revision),
                    max(1, min(int(limit), 100)),
                ),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            entity = _decode(row["payload_json"])
            # The centroid never crosses this projection boundary.
            entity.pop("centroidMap", None)
            result.append(
                {
                    **entity,
                    "importId": str(row["import_id"]),
                    "payloadSha256": str(row["payload_sha256"]),
                    "authority": "observation_only",
                    "executionEligible": False,
                }
            )
        return result


def _visibility(principal: Principal) -> tuple[str | None, str | None]:
    if principal.role is Role.ADMIN or principal.is_legacy:
        return None, None
    if principal.role is Role.DEVELOPER:
        return principal.site_id, None
    return principal.site_id, principal.user_id


def _translate(exc: Exception) -> HTTPException:
    if isinstance(exc, DaaamMemoryNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, DaaamMemoryConflict):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (ValueError, TypeError, sqlite3.IntegrityError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="DAAAM memory operation failed")


def create_daaam_memory_router(
    repository: DaaamMemoryRepository,
) -> APIRouter:
    router = APIRouter(
        prefix="/api/v2/semantic-memory/daaam",
        tags=["semantic-memory"],
    )
    use_memory = require_scope(DAAAM_MEMORY_SCOPE)

    @router.get("/capabilities")
    def capabilities(principal: Principal = Depends(use_memory)):
        del principal
        return {
            "schemaVersion": "h-ide.daaam-semantic-memory/1",
            "mode": "post_mission_async",
            "authority": "observation_only",
            "executionEligible": False,
            "navigationDispatch": "forbidden",
            "missionProjection": "region_semantics_without_coordinates",
            "upstreamRuntimeRequired": False,
        }

    @router.post("/imports", status_code=status.HTTP_201_CREATED)
    def create_import(
        body: DaaamMemoryImport,
        response: Response,
        principal: Principal = Depends(use_memory),
    ):
        if principal.role not in {Role.DEVELOPER, Role.ADMIN}:
            raise HTTPException(
                status_code=403,
                detail="DAAAM memory imports require a developer service identity",
            )
        try:
            result = repository.ingest(body, principal)
        except Exception as exc:
            raise _translate(exc) from exc
        if result.get("replayed"):
            # Exact import retries are successful replays, not duplicate creates.
            response.status_code = status.HTTP_200_OK
            return result
        return result

    @router.get("/imports")
    def list_imports(
        robotId: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        principal: Principal = Depends(use_memory),
    ):
        site_id, owner_user_id = _visibility(principal)
        return repository.list_imports(
            site_id=site_id,
            owner_user_id=owner_user_id,
            robot_id=robotId,
            limit=limit,
        )

    @router.get("/imports/{import_id}")
    def get_import(
        import_id: str,
        request: Request,
        principal: Principal = Depends(use_memory),
    ):
        require_resource_access(
            request,
            principal,
            DAAAM_MEMORY_SCOPE,
            resource_type=DAAAM_IMPORT_RESOURCE,
            resource_id=import_id,
        )
        try:
            return repository.get_import(import_id)
        except Exception as exc:
            raise _translate(exc) from exc

    @router.get("/imports/{import_id}/entities")
    def list_entities(
        import_id: str,
        request: Request,
        principal: Principal = Depends(use_memory),
    ):
        require_resource_access(
            request,
            principal,
            DAAAM_MEMORY_SCOPE,
            resource_type=DAAAM_IMPORT_RESOURCE,
            resource_id=import_id,
        )
        try:
            return repository.list_entities(import_id)
        except Exception as exc:
            raise _translate(exc) from exc

    router.daaam_memory_repository = repository  # type: ignore[attr-defined]
    return router


__all__ = [
    "DAAAM_IMPORT_RESOURCE",
    "DAAAM_MEMORY_SCOPE",
    "DaaamMemoryConflict",
    "DaaamMemoryNotFound",
    "DaaamMemoryRepository",
    "create_daaam_memory_router",
]
