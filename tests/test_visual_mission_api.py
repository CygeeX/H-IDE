from __future__ import annotations

from fastapi.testclient import TestClient

from packages.control_core.adapters import FakeAdapter
from packages.visual_mission_core import parse_plan_draft
from packages.visual_mission_core.runtime import VisualMissionRuntime
from services.control_api.main import create_app

from .test_visual_mission_contracts import visual_plan
from .test_visual_mission_runtime import (
    AmbiguousThenSelectedStageClient,
    AmbiguousThenSelectedVisionClient,
    StageClient,
    VisionClient,
    resolution,
)


TOKEN = "visual-api-token"


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def plan_json():
    return parse_plan_draft(visual_plan()).model_dump(mode="json")


def test_visual_draft_requires_review_then_returns_parent_mission() -> None:
    adapter = FakeAdapter()
    app = create_app(
        auth_token=TOKEN,
        adapters={"fake": adapter},
        vision_client=VisionClient(resolution()),
        visual_stage_client=StageClient(),
        visual_mission_mode="lab_execute",
    )
    client = TestClient(app)
    created = client.post("/api/v1/assistant/drafts", headers=auth(), json=plan_json())
    assert created.status_code == 200, created.text
    assert created.json()["plan"]["planType"] == "visual_mission"
    assert adapter.dispatches == []
    confirmed = client.post(
        "/api/v1/assistant/drafts/visual-plan-1/confirm",
        headers=auth(),
        json={"confirmed": True},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["mission"]["status"] == "approved"
    assert confirmed.json()["receipts"] == []
    assert adapter.dispatches == []
    ran = client.post("/api/v1/visual-missions/visual-plan-1/run", headers=auth())
    assert ran.status_code == 200, ran.text
    assert ran.json()["status"] == "completed"
    assert ran.json()["motion"] == "simulated"
    assert len(adapter.dispatches) == 1
    evidence = client.get(
        f"/api/v1/visual-missions/visual-plan-1/evidence/{'b' * 64}",
        headers=auth(),
    )
    assert evidence.status_code == 200
    assert evidence.content == b"fixture-evidence"
    assert evidence.headers["cache-control"] == "private, no-store, max-age=0"
    assert client.get(
        f"/api/v1/visual-missions/visual-plan-1/evidence/{'c' * 64}",
        headers=auth(),
    ).status_code == 404


def test_observe_mode_confirmation_never_dispatches() -> None:
    adapter = FakeAdapter()
    app = create_app(
        auth_token=TOKEN,
        adapters={"fake": adapter},
        vision_client=VisionClient(resolution()),
        visual_stage_client=StageClient(),
        visual_mission_mode="observe",
    )
    client = TestClient(app)
    assert client.post("/api/v1/assistant/drafts", headers=auth(), json=plan_json()).status_code == 200
    assert client.post(
        "/api/v1/assistant/drafts/visual-plan-1/confirm", headers=auth()
    ).status_code == 200
    blocked = client.post("/api/v1/visual-missions/visual-plan-1/run", headers=auth())
    assert blocked.status_code == 409
    assert "observe mode" in blocked.text
    assert adapter.dispatches == []


def test_visual_candidate_selection_requires_a_known_ambiguous_candidate() -> None:
    adapter = FakeAdapter()
    stage = AmbiguousThenSelectedStageClient()
    app = create_app(
        auth_token=TOKEN,
        adapters={"fake": adapter},
        vision_client=AmbiguousThenSelectedVisionClient(resolution()),
        visual_stage_client=stage,
        visual_mission_mode="lab_execute",
    )
    client = TestClient(app)
    assert client.post("/api/v1/assistant/drafts", headers=auth(), json=plan_json()).status_code == 200
    assert client.post(
        "/api/v1/assistant/drafts/visual-plan-1/confirm", headers=auth()
    ).status_code == 200
    first = client.post("/api/v1/visual-missions/visual-plan-1/run", headers=auth())
    assert first.status_code == 200
    assert first.json()["status"] == "needs_input"
    assert adapter.dispatches == []

    unknown = client.post(
        "/api/v1/visual-missions/visual-plan-1/candidates/not-a-candidate",
        headers=auth(),
    )
    assert unknown.status_code == 409
    selected = client.post(
        "/api/v1/visual-missions/visual-plan-1/candidates/candidate-a",
        headers=auth(),
    )
    assert selected.status_code == 200
    assert selected.json()["status"] == "approved"
    completed = client.post("/api/v1/visual-missions/visual-plan-1/run", headers=auth())
    assert completed.status_code == 200, completed.text
    assert completed.json()["status"] == "completed"
    assert len(adapter.dispatches) == 1


def test_visual_resolution_safety_rejection_returns_auditable_terminal_result() -> None:
    adapter = FakeAdapter()
    rejected = resolution(
        status="rejected",
        executable=False,
        goalPose=None,
        rejectionReason="approach_rejected:target is too close",
        safetyGates=[{"name": "approach_pose", "passed": False}],
    )
    app = create_app(
        auth_token=TOKEN,
        adapters={"fake": adapter},
        vision_client=VisionClient(rejected),
        visual_stage_client=StageClient(),
        visual_mission_mode="lab_execute",
    )
    client = TestClient(app)
    assert client.post("/api/v1/assistant/drafts", headers=auth(), json=plan_json()).status_code == 200
    assert client.post(
        "/api/v1/assistant/drafts/visual-plan-1/confirm", headers=auth()
    ).status_code == 200

    response = client.post("/api/v1/visual-missions/visual-plan-1/run", headers=auth())

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert response.json()["stages"][0]["visualResolutionId"] == "vr-1"
    assert adapter.dispatches == []


def test_visual_confirmation_is_atomic_under_repeated_requests() -> None:
    app = create_app(
        auth_token=TOKEN,
        adapters={"fake": FakeAdapter()},
        vision_client=VisionClient(resolution()),
        visual_stage_client=StageClient(),
        visual_mission_mode="lab_execute",
    )
    client = TestClient(app)
    assert client.post("/api/v1/assistant/drafts", headers=auth(), json=plan_json()).status_code == 200
    first = client.post("/api/v1/assistant/drafts/visual-plan-1/confirm", headers=auth())
    second = client.post("/api/v1/assistant/drafts/visual-plan-1/confirm", headers=auth())
    assert first.status_code == 200
    assert second.status_code == 409
    assert app.state.store.get_visual_mission("visual-plan-1")["status"] == "approved"


def test_visual_motion_owner_blocks_manual_motion_but_not_stop_or_non_motion() -> None:
    client = TestClient(create_app(auth_token=TOKEN))
    VisualMissionRuntime._claim_motion("robot-1", "mission-owner")
    try:
        move = client.post("/api/v1/commands", headers=auth(), json={
            "robotId": "robot-1",
            "commandKey": "move.forward",
            "payload": {},
            "idempotencyKey": "manual-motion-while-visual",
        })
        assert move.status_code == 409
        assert "visual mission mission-owner" in move.text
        non_motion = client.post("/api/v1/commands", headers=auth(), json={
            "robotId": "robot-1",
            "commandKey": "gripper.open",
            "payload": {},
            "idempotencyKey": "non-motion-while-visual",
        })
        assert non_motion.status_code == 200
        stop = client.post("/api/v1/commands", headers=auth(), json={
            "robotId": "robot-1",
            "commandKey": "stop",
            "payload": {},
            "idempotencyKey": "stop-while-visual",
        })
        assert stop.status_code == 200
    finally:
        VisualMissionRuntime._release_motion("robot-1", "mission-owner")
