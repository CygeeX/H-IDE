"""Authenticated, tenant-bound HTTP routes for Mission V2."""

from __future__ import annotations

import hmac
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import ValidationError

from packages.mission_core import MissionConflict, MissionNotFound

from .auth import (
    Principal,
    Scope,
    require_resource_access,
    require_scope,
)
from .autonomy_policy import AutonomyMode, ExecutionTarget
from .mission_client import MissionPlanningUnavailable
from .mission_policy import MissionPolicyDenied, PolicyActor
from .mission_service import MissionService, MissionServiceError
from .visual_task_client import VisualTaskUnavailable


_SERVER_IDENTITY_FIELDS = frozenset(
    {"approvedBy", "requestedBy", "authorizationId"}
)


def create_mission_router(
    service: MissionService,
    internal_token: str | None = None,
) -> APIRouter:
    """Create human and trusted-callback Mission routes.

    ``internal_token`` is deliberately unrelated to browser authentication.
    If it is absent, Robot Agent callback endpoints fail closed with 503.
    """

    router = APIRouter(prefix="/api/v2", tags=["mission-v2"])
    mission_user = require_scope(Scope.MISSIONS_USE)

    def invoke(operation):
        try:
            return operation()
        except MissionNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (MissionPlanningUnavailable, VisualTaskUnavailable) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except MissionPolicyDenied as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except MissionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except MissionServiceError as exc:
            detail = str(exc)
            if "not found" in detail:
                code = 404
            elif any(
                marker in detail
                for marker in (
                    "already",
                    "active physical",
                    "not confirmable",
                    "unavailable",
                    "not published",
                    "mismatch",
                    "stale",
                    "identity changed",
                    "epoch changed",
                    "policy",
                    "grant",
                )
            ):
                code = 409
            else:
                code = 422
            raise HTTPException(status_code=code, detail=detail) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def actor(principal: Principal) -> PolicyActor:
        return PolicyActor.from_principal(principal)

    def reject_server_identity(body: Mapping[str, Any] | None) -> dict[str, Any]:
        """Drop legacy client identity hints; authenticated identity wins.

        Keeping this tolerant avoids breaking older PWA builds while making
        spoofing inert.  No dropped value is forwarded to a planner/runtime.
        """
        payload = dict(body or {})
        for field in _SERVER_IDENTITY_FIELDS:
            payload.pop(field, None)
        return payload

    def access(
        request: Request,
        principal: Principal,
        resource_type: str,
        resource_id: str,
    ) -> Mapping[str, Any] | None:
        require_resource_access(
            request,
            principal,
            Scope.MISSIONS_USE,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        return service.store.get_resource_owner(resource_type, resource_id)

    def visible(
        request: Request,
        principal: Principal,
        resource_type: str,
        resource_id: str,
    ) -> bool:
        try:
            access(request, principal, resource_type, resource_id)
            return True
        except HTTPException as exc:
            if exc.status_code == 404:
                return False
            raise

    def require_internal(request: Request) -> None:
        configured = (internal_token or "").strip()
        if not configured:
            raise HTTPException(
                status_code=503,
                detail="Mission internal callback credential is not configured",
            )
        supplied = request.headers.get("x-h-ide-mission-internal-token", "")
        if not supplied or not hmac.compare_digest(supplied, configured):
            raise HTTPException(status_code=404, detail="resource not found")

    @router.get("/autonomy/preferences")
    def get_preferences(
        principal: Principal = Depends(mission_user),
    ) -> dict[str, Any]:
        current_actor = actor(principal)
        values: list[dict[str, Any]] = []
        for target in ExecutionTarget:
            stored = service.store.get_autonomy_preference(
                current_actor.user_id, current_actor.site_id, target.value
            )
            values.append(
                stored
                or {
                    "userId": current_actor.user_id,
                    "siteId": current_actor.site_id,
                    "executionTarget": target.value,
                    "mode": AutonomyMode.ASK_EVERY_ACTION.value,
                    "version": 0,
                }
            )
        return {"preferences": values}

    @router.put("/autonomy/preferences/{execution_target}")
    def put_preference(
        execution_target: str,
        body: dict[str, Any],
        principal: Principal = Depends(mission_user),
    ):
        payload = reject_server_identity(body)
        try:
            target = ExecutionTarget(execution_target)
            mode = AutonomyMode(str(payload.get("mode")))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        expected = payload.get("expectedVersion")
        if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int)):
            raise HTTPException(status_code=422, detail="expectedVersion must be an integer")
        return invoke(
            lambda: service.policy.set_preference(
                actor(principal),
                target=target,
                mode=mode,
                expected_version=expected,
            )
        )

    @router.get("/missions/drafts")
    def list_drafts(
        request: Request,
        limit: int = 100,
        principal: Principal = Depends(mission_user),
    ):
        return [
            item
            for item in service.store.list_mission_drafts(limit=limit)
            if visible(request, principal, "mission_draft", str(item["draftId"]))
        ]

    @router.post("/missions/drafts", status_code=status.HTTP_201_CREATED)
    def create_draft(
        request: Request,
        body: dict[str, Any],
        principal: Principal = Depends(mission_user),
    ):
        payload = reject_server_identity(body)
        resume_from = payload.get("resumeFromMissionId")
        if resume_from is not None:
            if not isinstance(resume_from, str) or not resume_from.strip():
                raise HTTPException(
                    status_code=422,
                    detail="resumeFromMissionId must be a non-empty string",
                )
            access(request, principal, "mission", resume_from)
        payload["requestedBy"] = principal.user_id
        return invoke(
            lambda: service.create_draft(payload, actor=actor(principal))
        )

    @router.get("/missions/drafts/{draft_id}")
    def get_draft(
        draft_id: str,
        request: Request,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission_draft", draft_id)
        return invoke(lambda: service.get_draft(draft_id))

    @router.post("/missions/drafts/{draft_id}/reject")
    def reject_draft(
        draft_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission_draft", draft_id)
        payload = reject_server_identity(body)
        return invoke(
            lambda: service.reject_draft(
                draft_id, str(payload.get("reason") or "人工审核拒绝")
            )
        )

    @router.post("/missions/drafts/{draft_id}/approve")
    def approve_draft(
        draft_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(mission_user),
    ):
        binding = access(request, principal, "mission_draft", draft_id)
        reject_server_identity(body)
        return invoke(
            lambda: service.approve_draft(
                draft_id,
                principal.user_id,
                actor=actor(principal),
                resource_owner_user_id=(
                    None if binding is None else str(binding["ownerUserId"])
                ),
                resource_site_id=(
                    None if binding is None else str(binding["siteId"])
                ),
            )
        )

    @router.get("/missions")
    def list_missions(
        request: Request,
        limit: int = 100,
        principal: Principal = Depends(mission_user),
    ):
        return [
            item
            for item in service.runtime.list(limit=limit)
            if visible(request, principal, "mission", str(item["missionId"]))
        ]

    @router.get("/missions/{mission_id}")
    def get_mission(
        mission_id: str,
        request: Request,
        afterSequence: int = 0,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        return invoke(
            lambda: service.get_mission(
                mission_id, after_sequence=max(0, afterSequence)
            )
        )

    @router.get("/missions/{mission_id}/visual-evidence/{analysis_id}/{sha256}")
    def get_visual_evidence(
        mission_id: str,
        analysis_id: str,
        sha256: str,
        request: Request,
        principal: Principal = Depends(mission_user),
    ) -> Response:
        access(request, principal, "mission", mission_id)
        content, media_type = invoke(
            lambda: service.get_visual_evidence(mission_id, analysis_id, sha256)
        )
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

    @router.post("/missions/{mission_id}/start")
    def start_mission(
        mission_id: str,
        request: Request,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        return invoke(
            lambda: service.start(mission_id, actor=actor(principal))
        )

    @router.post("/missions/{mission_id}/advance")
    def advance_mission(
        mission_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        payload = reject_server_identity(body)
        maximum_steps = payload.get("maximumSteps", 20)
        if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
            raise HTTPException(status_code=422, detail="maximumSteps must be an integer")
        return invoke(
            lambda: service.advance(
                mission_id,
                maximum_steps=maximum_steps,
                actor=actor(principal),
            )
        )

    @router.post("/missions/{mission_id}/messages")
    def send_mission_message(
        mission_id: str,
        body: dict[str, Any],
        request: Request,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        payload = reject_server_identity(body)
        payload["requestedBy"] = principal.user_id
        return invoke(lambda: service.respond(mission_id, payload))

    @router.post("/missions/{mission_id}/cancel")
    def cancel_mission(
        mission_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        payload = reject_server_identity(body)
        payload["requestedBy"] = principal.user_id
        return invoke(lambda: service.cancel(mission_id, payload))

    @router.post(
        "/missions/{mission_id}/nodes/{node_id}/skill-requests",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def issue_skill(
        mission_id: str,
        node_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        if body:
            raise HTTPException(
                status_code=422,
                detail=(
                    "skill input is server-owned; update the PlanDraft/WorldRecord "
                    "and issue the reviewed node without an input body"
                ),
            )
        return invoke(
            lambda: service.issue_resolved_skill(
                mission_id, node_id, actor=actor(principal)
            )
        )

    # Robot Agent / Vision callbacks use an independent service credential.
    # Browser principals, including ADMIN, cannot forge progress or results.
    @router.post("/missions/{mission_id}/skill-progress")
    def report_progress(
        mission_id: str, body: dict[str, Any], request: Request
    ):
        require_internal(request)
        return invoke(lambda: service.progress(mission_id, body))

    @router.post("/missions/{mission_id}/skill-results")
    def report_result(
        mission_id: str, body: dict[str, Any], request: Request
    ):
        require_internal(request)
        return invoke(lambda: service.result(mission_id, body))

    @router.post("/missions/{mission_id}/world-records")
    def record_world(
        mission_id: str,
        body: dict[str, Any],
        request: Request,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        payload = reject_server_identity(body)
        if payload.get("source") not in {"operator", "scene"}:
            # Backward-compatible callback path: non-operator evidence still
            # requires the independent internal service credential.
            require_internal(request)
        payload["missionId"] = mission_id
        return invoke(lambda: service.world_record(payload))

    @router.post("/internal/missions/{mission_id}/world-records")
    def record_world_internal(
        mission_id: str, body: dict[str, Any], request: Request
    ):
        require_internal(request)
        payload = dict(body)
        payload["missionId"] = mission_id
        return invoke(lambda: service.world_record(payload))

    @router.post("/missions/{mission_id}/interventions")
    def intervene(
        mission_id: str,
        body: dict[str, Any],
        request: Request,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        payload = reject_server_identity(body)
        payload["requestedBy"] = principal.user_id
        return invoke(lambda: service.intervene(mission_id, payload))

    @router.post("/missions/{mission_id}/revisions")
    def request_revision(
        mission_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        payload = reject_server_identity(body)
        payload["requestedBy"] = principal.user_id
        return invoke(lambda: service.request_revision(mission_id, payload))

    @router.post("/missions/{mission_id}/revisions/{revision_id}/approve")
    def approve_revision(
        mission_id: str,
        revision_id: str,
        request: Request,
        body: dict[str, Any] | None = None,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        reject_server_identity(body)
        return invoke(
            lambda: service.approve_revision(
                mission_id,
                revision_id,
                approved_by=principal.user_id,
                actor=actor(principal),
            )
        )

    @router.post("/missions/{mission_id}/events/evaluate")
    def evaluate_event(
        mission_id: str, body: dict[str, Any], request: Request
    ):
        require_internal(request)
        return invoke(
            lambda: service.evaluate_event(
                mission_id, body, force_human_review=True
            )
        )

    @router.post("/missions/{mission_id}/candidate-scores")
    def score_candidates(
        mission_id: str,
        body: dict[str, Any],
        request: Request,
        principal: Principal = Depends(mission_user),
    ):
        access(request, principal, "mission", mission_id)
        return invoke(lambda: service.score_candidates(mission_id, body))

    return router


__all__ = ["create_mission_router"]
