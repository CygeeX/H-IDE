"""Authenticated HTTP routes for the durable H-IDE Mission V2 runtime."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import ValidationError

from packages.mission_core import MissionConflict, MissionNotFound

from .mission_client import MissionPlanningUnavailable
from .mission_service import MissionService, MissionServiceError
from .visual_task_client import VisualTaskUnavailable
from .auth import Scope, require_scope


def create_mission_router(service: MissionService, token: str | None = None) -> APIRouter:
    # ``token`` is retained in the factory signature for embedded callers;
    # authentication now comes from the app-scoped AuthService so browser
    # sessions and the explicit legacy developer Bearer identity share policy.
    del token
    router = APIRouter(prefix="/api/v2", tags=["mission-v2"])
    dependencies = [Depends(require_scope(Scope.MISSIONS_USE))]

    def invoke(operation):
        try:
            return operation()
        except MissionNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (MissionPlanningUnavailable, VisualTaskUnavailable) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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

    @router.get("/missions/drafts", dependencies=dependencies)
    def list_drafts(limit: int = 100):
        return service.store.list_mission_drafts(limit=limit)

    @router.post(
        "/missions/drafts",
        dependencies=dependencies,
        status_code=status.HTTP_201_CREATED,
    )
    def create_draft(body: dict[str, Any]):
        return invoke(lambda: service.create_draft(body))

    @router.get("/missions/drafts/{draft_id}", dependencies=dependencies)
    def get_draft(draft_id: str):
        return invoke(lambda: service.get_draft(draft_id))

    @router.post("/missions/drafts/{draft_id}/reject", dependencies=dependencies)
    def reject_draft(draft_id: str, body: dict[str, Any] | None = None):
        payload = body or {}
        return invoke(
            lambda: service.reject_draft(
                draft_id,
                str(payload.get("reason") or "人工审核拒绝"),
            )
        )

    @router.post("/missions/drafts/{draft_id}/approve", dependencies=dependencies)
    def approve_draft(draft_id: str, body: dict[str, Any] | None = None):
        payload = body or {}
        approved_by = payload.get("approvedBy", "operator")
        return invoke(lambda: service.approve_draft(draft_id, approved_by))

    @router.get("/missions", dependencies=dependencies)
    def list_missions(limit: int = 100):
        return service.runtime.list(limit=limit)

    @router.get("/missions/{mission_id}", dependencies=dependencies)
    def get_mission(mission_id: str, afterSequence: int = 0):
        return invoke(
            lambda: service.get_mission(
                mission_id,
                after_sequence=max(0, afterSequence),
            )
        )

    @router.get(
        "/missions/{mission_id}/visual-evidence/{analysis_id}/{sha256}",
        dependencies=dependencies,
    )
    def get_visual_evidence(
        mission_id: str,
        analysis_id: str,
        sha256: str,
    ) -> Response:
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

    @router.post("/missions/{mission_id}/start", dependencies=dependencies)
    def start_mission(mission_id: str):
        return invoke(lambda: service.start(mission_id))

    @router.post("/missions/{mission_id}/advance", dependencies=dependencies)
    def advance_mission(mission_id: str, body: dict[str, Any] | None = None):
        payload = body or {}
        maximum_steps = payload.get("maximumSteps", 20)
        if isinstance(maximum_steps, bool) or not isinstance(maximum_steps, int):
            raise HTTPException(status_code=422, detail="maximumSteps must be an integer")
        return invoke(
            lambda: service.advance(mission_id, maximum_steps=maximum_steps)
        )

    @router.post("/missions/{mission_id}/messages", dependencies=dependencies)
    def send_mission_message(mission_id: str, body: dict[str, Any]):
        return invoke(lambda: service.respond(mission_id, body))

    @router.post("/missions/{mission_id}/cancel", dependencies=dependencies)
    def cancel_mission(mission_id: str, body: dict[str, Any] | None = None):
        return invoke(lambda: service.cancel(mission_id, body or {}))

    @router.post(
        "/missions/{mission_id}/nodes/{node_id}/skill-requests",
        dependencies=dependencies,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def issue_skill(mission_id: str, node_id: str, body: dict[str, Any] | None = None):
        payload = body or {}
        skill_input = payload.get("input", payload)
        if not isinstance(skill_input, dict):
            raise HTTPException(status_code=422, detail="input must be an object")
        return invoke(lambda: service.issue_skill(mission_id, node_id, skill_input))

    @router.post("/missions/{mission_id}/skill-progress", dependencies=dependencies)
    def report_progress(mission_id: str, body: dict[str, Any]):
        return invoke(lambda: service.progress(mission_id, body))

    @router.post("/missions/{mission_id}/skill-results", dependencies=dependencies)
    def report_result(mission_id: str, body: dict[str, Any]):
        return invoke(lambda: service.result(mission_id, body))

    @router.post("/missions/{mission_id}/world-records", dependencies=dependencies)
    def record_world(mission_id: str, body: dict[str, Any]):
        payload = dict(body)
        payload.setdefault("missionId", mission_id)
        return invoke(lambda: service.world_record(payload))

    @router.post("/missions/{mission_id}/interventions", dependencies=dependencies)
    def intervene(mission_id: str, body: dict[str, Any]):
        return invoke(lambda: service.intervene(mission_id, body))

    @router.post("/missions/{mission_id}/revisions", dependencies=dependencies)
    def request_revision(mission_id: str, body: dict[str, Any] | None = None):
        return invoke(lambda: service.request_revision(mission_id, body or {}))

    @router.post(
        "/missions/{mission_id}/revisions/{revision_id}/approve",
        dependencies=dependencies,
    )
    def approve_revision(
        mission_id: str,
        revision_id: str,
        body: dict[str, Any] | None = None,
    ):
        payload = body or {}
        return invoke(
            lambda: service.approve_revision(
                mission_id,
                revision_id,
                approved_by=payload.get("approvedBy", "operator"),
            )
        )

    @router.post("/missions/{mission_id}/events/evaluate", dependencies=dependencies)
    def evaluate_event(mission_id: str, body: dict[str, Any]):
        return invoke(lambda: service.evaluate_event(mission_id, body))

    @router.post("/missions/{mission_id}/candidate-scores", dependencies=dependencies)
    def score_candidates(mission_id: str, body: dict[str, Any]):
        return invoke(lambda: service.score_candidates(mission_id, body))

    return router


__all__ = ["create_mission_router"]
