"""Authenticated API for H-IDE simulation sessions."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Response, status

from packages.simulation_core import (
    NavigationGoalInput,
    SimulationConflict,
    SimulationEventInput,
    SimulationNotFound,
    SimulationRuntime,
    SimulationStepInput,
)
from .auth import Scope, require_scope


def create_simulation_router(
    runtime: SimulationRuntime,
    token: str | None = None,
    *,
    review_resolver: Callable[[str, str | None], Any] | None = None,
) -> APIRouter:
    del token
    router = APIRouter(prefix="/api/v2", tags=["simulation"])
    dependencies = [Depends(require_scope(Scope.SIMULATIONS_MANAGE))]

    def invoke(operation):
        try:
            return operation()
        except SimulationNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except SimulationConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/simulations", dependencies=dependencies)
    def list_simulations(limit: int = 100):
        return runtime.list(limit=limit)

    @router.get("/simulations/backends", dependencies=dependencies)
    def list_simulation_backends():
        return runtime.backends()

    @router.post("/simulations", dependencies=dependencies, status_code=status.HTTP_201_CREATED)
    def create_simulation(body: dict[str, Any] | None = None):
        return invoke(lambda: runtime.create(body or {}))

    @router.get("/simulations/{session_id}", dependencies=dependencies)
    def get_simulation(session_id: str):
        return invoke(lambda: runtime.get(session_id))

    @router.get("/simulations/{session_id}/live", dependencies=dependencies)
    def get_live_simulation(session_id: str):
        """Read-only live telemetry/camera metadata from MuJoCo+SONIC."""
        return invoke(lambda: runtime.live(session_id))

    @router.get("/simulations/{session_id}/rgb.jpg", dependencies=dependencies)
    def get_live_rgb_frame(session_id: str):
        frame = invoke(lambda: runtime.rgb_frame(session_id))
        return Response(
            content=frame,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @router.get("/simulations/{session_id}/events", dependencies=dependencies)
    def list_simulation_events(session_id: str, after: int = 0, limit: int = 500):
        return invoke(lambda: runtime.events(session_id, after=after, limit=limit))

    @router.post("/simulations/{session_id}/start", dependencies=dependencies)
    def start_simulation(session_id: str):
        return invoke(lambda: runtime.start(session_id))

    @router.post("/simulations/{session_id}/pause", dependencies=dependencies)
    def pause_simulation(session_id: str):
        return invoke(lambda: runtime.pause(session_id))

    @router.post("/simulations/{session_id}/stop", dependencies=dependencies)
    def stop_simulation(session_id: str):
        return invoke(lambda: runtime.stop(session_id))

    @router.post("/simulations/{session_id}/mission-binding", dependencies=dependencies)
    def bind_simulation_mission(session_id: str, body: dict[str, Any]):
        return invoke(lambda: runtime.bind_mission(session_id, body.get("missionId")))

    @router.post("/simulations/{session_id}/mission-review/resolve", dependencies=dependencies)
    def resolve_simulation_review(session_id: str, body: dict[str, Any] | None = None):
        if review_resolver is None:
            raise HTTPException(status_code=409, detail="Mission review resolver is not configured")
        payload = body or {}
        revision_id = payload.get("revisionId")
        if revision_id is not None and not isinstance(revision_id, str):
            raise HTTPException(status_code=422, detail="revisionId must be a string")
        return invoke(lambda: review_resolver(session_id, revision_id))

    @router.post("/simulations/{session_id}/reset", dependencies=dependencies)
    def reset_simulation(session_id: str):
        return invoke(lambda: runtime.reset(session_id))

    @router.post("/simulations/{session_id}/navigation-goals", dependencies=dependencies)
    def set_navigation_goal(session_id: str, body: dict[str, Any]):
        return invoke(lambda: runtime.set_goal(session_id, NavigationGoalInput.model_validate(body)))

    @router.post("/simulations/{session_id}/step", dependencies=dependencies)
    def step_simulation(session_id: str, body: dict[str, Any] | None = None):
        return invoke(lambda: runtime.step(session_id, SimulationStepInput.model_validate(body or {})))

    @router.post("/simulations/{session_id}/events", dependencies=dependencies)
    def inject_simulation_event(session_id: str, body: dict[str, Any]):
        return invoke(lambda: runtime.inject_event(session_id, SimulationEventInput.model_validate(body)))

    return router


__all__ = ["create_simulation_router"]
