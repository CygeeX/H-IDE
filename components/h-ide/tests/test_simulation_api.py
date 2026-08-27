from __future__ import annotations

from fastapi.testclient import TestClient
import pytest
from typing import Any, Mapping

from packages.simulation_core import SimulationRuntime
from packages.control_core.validation import utc_now
from services.control_api.main import create_app
from tests.test_mission_v2_api import FakeMissionPlanningClient


TOKEN = "simulation-test-token"
INTERNAL_TOKEN = "simulation-mission-internal-test-token"


@pytest.fixture(autouse=True)
def mission_internal_callback_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISSION_V2_INTERNAL_TOKEN", INTERNAL_TOKEN)


def auth() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "X-H-IDE-Mission-Internal-Token": INTERNAL_TOKEN,
    }


class SimulationReplanClient(FakeMissionPlanningClient):
    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        active = event["activeRevision"]
        nodes = [dict(item) for item in active["nodes"]]
        nodes[0] = {
            **nodes[0],
            "purpose": "仿真异常后重新确认目标",
        }
        return {
            "kind": "replan",
            "revision": {
                "schemaVersion": "h-ide.plan-revision/2",
                "revisionId": f"revision-{event['correlationId']}",
                "missionId": event["missionId"],
                "sequence": active["sequence"] + 1,
                "parentRevisionId": active["revisionId"],
                "objective": active["objective"],
                "nodes": nodes,
                "reasonCode": "simulation_event",
                "reason": "simulator reported a changed observation",
                "changeSummary": ["仿真异常触发重新观察"],
                # Deliberately return an inherited proposal. H-IDE must
                # coerce it back to human_required for simulation sessions.
                "disposition": "policy_inherited",
                "authorizationId": event["payload"].get("authorizationId"),
                "createdBy": "h-ide-agent",
                "createdAt": utc_now().isoformat(),
            },
        }


def test_simulation_api_requires_auth_and_runs_a_deterministic_session() -> None:
    app = create_app(auth_token=TOKEN, simulation_runtime=SimulationRuntime())
    with TestClient(app) as client:
        assert client.get("/api/v2/simulations").status_code == 401
        created = client.post("/api/v2/simulations", headers=auth(), json={})
        assert created.status_code == 201
        session_id = created.json()["session"]["sessionId"]
        assert created.json()["session"]["executionProfile"] == "simulation"

        started = client.post(f"/api/v2/simulations/{session_id}/start", headers=auth())
        assert started.status_code == 200
        goal = client.post(
            f"/api/v2/simulations/{session_id}/navigation-goals",
            headers=auth(),
            json={
                "goalPose": {
                    "frameId": "map",
                    "x": 0.25,
                    "y": 0.0,
                    "z": 0.0,
                    "yaw": 0.0,
                }
            },
        )
        assert goal.status_code == 200
        stepped = client.post(
            f"/api/v2/simulations/{session_id}/step",
            headers=auth(),
            json={"distanceM": 0.25},
        )
        assert stepped.status_code == 200
        assert stepped.json()["snapshot"]["navigationState"] == "arrived"
        robot = client.get("/api/v1/robots/robot-1", headers=auth())
        # The simulation projection is visible to the workstation state, but
        # remains labelled as simulation rather than a physical Robot Agent.
        assert robot.status_code == 404  # the API exposes the collection, not a detail route
        robots = client.get("/api/v1/robots", headers=auth()).json()
        simulated = next(item for item in robots if item["robotId"] == "robot-1")
        assert simulated["details"]["adapter"] == "simulation"
        assert simulated["pose"]["x"] == 0.25
        events = client.get(f"/api/v2/simulations/{session_id}/events", headers=auth())
        assert events.status_code == 200
        assert events.json()[-1]["eventType"] == "navigation.succeeded"


def test_simulation_api_rejects_physical_profile_and_unsupported_isaac_backend() -> None:
    app = create_app(auth_token=TOKEN, simulation_runtime=SimulationRuntime())
    with TestClient(app) as client:
        physical = client.post(
            "/api/v2/simulations",
            headers=auth(),
            json={"executionProfile": "physical"},
        )
        assert physical.status_code == 422
        isaac = client.post(
            "/api/v2/simulations",
            headers=auth(),
            json={"backend": "isaac"},
        )
        assert isaac.status_code == 409


def test_simulation_binding_and_anomaly_review_are_visible_without_physical_dispatch() -> None:
    app = create_app(auth_token=TOKEN, simulation_runtime=SimulationRuntime())
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/simulations",
            headers=auth(),
            json={"robotId": "robot-1"},
        )
        assert created.status_code == 201
        session_id = created.json()["session"]["sessionId"]
        bound = client.post(
            f"/api/v2/simulations/{session_id}/mission-binding",
            headers=auth(),
            json={"missionId": "mission-sim-1"},
        )
        assert bound.status_code == 200
        assert bound.json()["session"]["missionId"] == "mission-sim-1"

        injected = client.post(
            f"/api/v2/simulations/{session_id}/events",
            headers=auth(),
            json={"eventType": "vision.target_lost", "payload": {"target": "桌子"}},
        )
        assert injected.status_code == 200
        review = injected.json()["snapshot"]["pendingReview"]
        assert review["status"] == "error"  # no Mission record/planner was configured
        assert review["missionId"] == "mission-sim-1"
        assert injected.json()["session"]["executionProfile"] == "simulation"
        assert injected.json()["snapshot"]["navigationState"] == "idle"


def test_bound_simulation_forces_a_planner_revision_through_human_review() -> None:
    app = create_app(
        auth_token=TOKEN,
        simulation_runtime=SimulationRuntime(),
        mission_planning_client=SimulationReplanClient(),
    )
    with TestClient(app) as client:
        mission = client.post(
            "/api/v2/missions/drafts",
            headers=auth(),
            json={
                "requestId": "simulation-mission-request",
                "missionId": "simulation-mission-1",
                "robotId": "robot-1",
                "objective": "仿真中持续导航",
                "requestedBy": "operator-test",
            },
        )
        assert mission.status_code == 201, mission.text
        approved = client.post(
            "/api/v2/missions/drafts/simulation-mission-1/approve",
            headers=auth(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text

        session = client.post(
            "/api/v2/simulations",
            headers=auth(),
            json={"missionId": "simulation-mission-1"},
        )
        assert session.status_code == 201, session.text
        session_id = session.json()["session"]["sessionId"]
        injected = client.post(
            f"/api/v2/simulations/{session_id}/events",
            headers=auth(),
            json={"eventType": "vision.target_lost", "payload": {"target": "厨房"}},
        )
        assert injected.status_code == 200, injected.text
        review = injected.json()["snapshot"]["pendingReview"]
        assert review["status"] == "pending"
        assert review["decisionKind"] == "replan"
        assert review["revisionId"]

        mission_after = client.get(
            "/api/v2/missions/simulation-mission-1",
            headers=auth(),
        )
        assert mission_after.status_code == 200, mission_after.text
        assert mission_after.json()["snapshot"]["status"] == "awaiting_review"
        assert mission_after.json()["revisions"][-1]["disposition"] == "human_required"

        revision_id = review["revisionId"]
        approved = client.post(
            f"/api/v2/missions/simulation-mission-1/revisions/{revision_id}/approve",
            headers=auth(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        resolved = client.post(
            f"/api/v2/simulations/{session_id}/mission-review/resolve",
            headers=auth(),
            json={"revisionId": revision_id},
        )
        assert resolved.status_code == 200, resolved.text
        assert resolved.json()["snapshot"]["pendingReview"]["status"] == "resolved"

        started = client.post(
            f"/api/v2/simulations/{session_id}/start",
            headers=auth(),
        )
        assert started.status_code == 200, started.text
        next_goal = client.post(
            f"/api/v2/simulations/{session_id}/navigation-goals",
            headers=auth(),
            json={"goalPose": {"frameId": "map", "x": 0.25, "y": 0, "z": 0, "yaw": 0}},
        )
        assert next_goal.status_code == 200, next_goal.text
        assert next_goal.json()["receipt"]["motion"] == "simulated"
