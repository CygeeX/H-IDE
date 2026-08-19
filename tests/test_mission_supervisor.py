from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.control_core.adapters import AdapterResult, RobotAgentAdapter
from packages.control_core.validation import utc_now
from services.control_api.main import create_app

from .test_mission_v2_api import (
    FakeVisualTaskClient,
    MissionRobotDriver,
    VisualMissionPlanningClient,
)


TOKEN = "mission-supervisor-token"
INTERNAL_TOKEN = "mission-supervisor-internal-token"


@pytest.fixture(autouse=True)
def supervisor_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISSION_V2_INTERNAL_TOKEN", INTERNAL_TOKEN)
    monkeypatch.setenv("MISSION_POLICY_ALLOW_LEGACY_REAL", "true")
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "X-H-IDE-Mission-Internal-Token": INTERNAL_TOKEN,
    }


def _release_result(*, goal_reached: bool = True) -> dict[str, Any]:
    return {
        "goalReached": goal_reached,
        "controlRelease": {
            "state": "SAFE_HOLD",
            "authorityRevoked": True,
            "safeStopState": "confirmed",
            "safeStopEpoch": 9,
        },
    }


class _ConvergingDriver(MissionRobotDriver):
    def __init__(self, *, terminal_status: str = "succeeded") -> None:
        super().__init__()
        self.terminal_status = terminal_status
        self.query_calls: list[str] = []

    def dispatch(self, request, mapping) -> AdapterResult:
        self.dispatches.append((request, mapping))
        return AdapterResult(status="running", result={"accepted": True})

    def query(self, command_id: str) -> AdapterResult:
        self.query_calls.append(command_id)
        return AdapterResult(
            status=self.terminal_status,
            result=_release_result(
                goal_reached=self.terminal_status == "succeeded"
            ),
            error=(
                None
                if self.terminal_status == "succeeded"
                else "Nav2 could not reach the reviewed target"
            ),
        )


class _FailureReplanClient(VisualMissionPlanningClient):
    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        active = event["activeRevision"]
        return {
            "kind": "replan",
            "revision": {
                "schemaVersion": "h-ide.plan-revision/2",
                "revisionId": f"revision-replan-{event['correlationId']}",
                "missionId": event["missionId"],
                "sequence": active["sequence"] + 1,
                "parentRevisionId": active["revisionId"],
                "objective": active["objective"],
                "nodes": active["nodes"],
                "reasonCode": "navigation_failed",
                "reason": "Nav2 failure requires a reviewed target revision",
                "changeSummary": ["Retry remains behind human review"],
                "disposition": "policy_inherited",
                "authorizationId": active["authorizationId"],
                "createdBy": "mission-supervisor-test",
                "createdAt": utc_now().isoformat(),
            },
        }


def _approve_and_advance(client: TestClient, request_id: str) -> str:
    draft = client.post(
        "/api/v2/missions/drafts",
        headers=_headers(),
        json={
            "requestId": request_id,
            "objective": "找到银白色机器人并移动到它旁边",
        },
    )
    assert draft.status_code == 201, draft.text
    mission_id = draft.json()["draftId"]
    approved = client.post(
        f"/api/v2/missions/drafts/{mission_id}/approve",
        headers=_headers(),
        json={"approvedBy": "operator-test"},
    )
    assert approved.status_code == 200, approved.text
    advanced = client.post(
        f"/api/v2/missions/{mission_id}/advance",
        headers=_headers(),
        json={"maximumSteps": 10},
    )
    assert advanced.status_code == 200, advanced.text
    assert advanced.json()["stoppedBecause"] in {"dispatching", "executing"}
    return mission_id


def _wait_for_status(
    client: TestClient,
    mission_id: str,
    expected: str,
    *,
    timeout: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(
            f"/api/v2/missions/{mission_id}", headers=_headers()
        )
        assert response.status_code == 200, response.text
        record = response.json()
        if record["snapshot"]["status"] == expected:
            return record
        time.sleep(0.01)
    raise AssertionError(f"Mission did not reach {expected}")


def test_supervisor_completes_without_browser_advance_and_stops_cleanly() -> None:
    driver = _ConvergingDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=VisualMissionPlanningClient(),
        visual_task_client=FakeVisualTaskClient(),
        mission_supervisor_interval=0.01,
    )
    with TestClient(app) as client:
        mission_id = _approve_and_advance(client, "supervisor-complete")
        completed = _wait_for_status(client, mission_id, "completed")
        assert len(driver.dispatches) == 1
        assert driver.query_calls
        receipt = completed["skillExecutions"][-1]["result"]["result"][
            "commandReceipt"
        ]
        assert receipt["result"]["controlRelease"]["safeStopState"] == (
            "confirmed"
        )

    assert not app.state.watchdog_thread.is_alive()
    assert not app.state.mission_supervisor_thread.is_alive()


def test_supervisor_stops_every_physical_replan_at_human_review() -> None:
    driver = _ConvergingDriver(terminal_status="failed")
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=_FailureReplanClient(),
        visual_task_client=FakeVisualTaskClient(),
        mission_supervisor_interval=0.01,
    )
    with TestClient(app) as client:
        mission_id = _approve_and_advance(client, "supervisor-replan")
        pending = _wait_for_status(client, mission_id, "awaiting_review")
        pending_revision = pending["revisions"][-1]
        assert pending_revision["disposition"] == "human_required"
        assert pending_revision["authorizationId"] is None
        assert len(driver.dispatches) == 1
        queries = len(driver.query_calls)
        time.sleep(0.05)
        still_pending = client.get(
            f"/api/v2/missions/{mission_id}", headers=_headers()
        ).json()
        assert still_pending["snapshot"]["status"] == "awaiting_review"
        assert len(driver.dispatches) == 1
        assert len(driver.query_calls) == queries


def test_supervisor_after_control_api_restart_queries_but_never_redispatches(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "mission-supervisor-restart.db")
    driver = _ConvergingDriver()
    first_app = create_app(
        db_path=db_path,
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=VisualMissionPlanningClient(),
        visual_task_client=FakeVisualTaskClient(),
        mission_supervisor_interval=60.0,
    )
    with TestClient(first_app) as first_client:
        mission_id = _approve_and_advance(
            first_client, "supervisor-restart"
        )
        assert len(driver.dispatches) == 1

    restarted_app = create_app(
        db_path=db_path,
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=VisualMissionPlanningClient(),
        visual_task_client=FakeVisualTaskClient(),
        mission_supervisor_interval=0.01,
    )
    with TestClient(restarted_app) as restarted_client:
        completed = _wait_for_status(
            restarted_client, mission_id, "completed"
        )
        assert completed["snapshot"]["activeSkillRequestId"] is None
        assert len(driver.dispatches) == 1
        assert driver.query_calls
