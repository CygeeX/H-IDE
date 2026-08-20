from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient

from packages.control_core.adapters import AdapterResult, RobotAgentAdapter
from packages.control_core.validation import utc_now
from packages.mission_core import MissionConflict
from services.control_api.main import create_app
from services.control_api.mission_client import MissionPlanningUnavailable

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


class _ParallelFailureReplanClient(_FailureReplanClient):
    def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]:
        response = super().create_draft(brief)
        plan = dict(response["plan"])
        template = dict(plan["nodes"][1])
        plan["nodes"] = [
            {
                **template,
                "nodeId": "navigate-a",
                "purpose": "到达并行目标 A",
                "intent": {"targetDescription": "并行目标 A"},
                "dependsOn": [],
            },
            {
                **template,
                "nodeId": "navigate-b",
                "purpose": "到达并行目标 B",
                "intent": {"targetDescription": "并行目标 B"},
                "dependsOn": [],
            },
        ]
        return {"plan": plan}


class _FailOnceFailureReplanClient(_FailureReplanClient):
    def __init__(self) -> None:
        self.event_calls = 0

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.event_calls += 1
        if self.event_calls == 1:
            raise MissionPlanningUnavailable("fixture planner timeout")
        return super().evaluate_event(event, context)


class _FailOnceContinueClient(VisualMissionPlanningClient):
    def __init__(self) -> None:
        self.event_calls = 0

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del event, context
        self.event_calls += 1
        if self.event_calls == 1:
            raise MissionPlanningUnavailable("fixture planner timeout")
        return {"kind": "continue", "reason": "planner found no safe revision"}


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


def test_supervisor_retries_the_exact_failed_skill_event_after_planner_timeout() -> None:
    driver = _ConvergingDriver(terminal_status="failed")
    planner = _FailOnceFailureReplanClient()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=planner,
        visual_task_client=FakeVisualTaskClient(),
        mission_supervisor_interval=0.01,
    )
    with TestClient(app) as client:
        mission_id = _approve_and_advance(client, "supervisor-planner-retry")
        pending = _wait_for_status(client, mission_id, "awaiting_review")
        assert planner.event_calls >= 2
        assert pending["snapshot"]["errorCode"] is None
        assert pending["revisions"][-1]["disposition"] == "human_required"
        assert len(driver.dispatches) == 1


def test_failed_skill_continue_converges_to_human_input_after_planner_retry() -> None:
    driver = _ConvergingDriver(terminal_status="failed")
    planner = _FailOnceContinueClient()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=planner,
        visual_task_client=FakeVisualTaskClient(),
        mission_supervisor_interval=0.01,
    )
    with TestClient(app) as client:
        mission_id = _approve_and_advance(client, "supervisor-planner-no-change")
        needs_input = _wait_for_status(client, mission_id, "needs_input")
        assert planner.event_calls == 2
        assert needs_input["snapshot"]["errorCode"] == "REPLAN_NO_CHANGE"
        assert len(driver.dispatches) == 1


def test_failed_parallel_navigation_cannot_issue_sibling_until_replan_is_ready() -> None:
    driver = _ConvergingDriver(terminal_status="failed")
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=_ParallelFailureReplanClient(),
        visual_task_client=FakeVisualTaskClient(),
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=_headers(),
            json={
                "requestId": "parallel-nav-failure-fence",
                "objective": "尝试两个并行导航目标",
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
        for suffix, x in (("a", 0.5), ("b", 0.75)):
            recorded = client.post(
                f"/api/v2/missions/{mission_id}/world-records",
                headers=_headers(),
                json={
                    "recordId": f"parallel-target-{suffix}",
                    "robotId": "robot-1",
                    "kind": "fact",
                    "subject": f"并行目标 {suffix.upper()}",
                    "predicate": "target_pose",
                    "value": {
                        "goalPose": {
                            "frameId": "map",
                            "x": x,
                            "y": 0.0,
                            "z": 0.0,
                            "yaw": 0.0,
                        }
                    },
                    "source": "operator",
                    "sourceRef": f"parallel-target-{suffix}",
                    "confidence": 1.0,
                    "observedAt": utc_now().isoformat(),
                    "frameId": "map",
                },
            )
            assert recorded.status_code == 200, recorded.text
        started = client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=_headers(),
        )
        assert started.status_code == 200, started.text

        issued_a = client.post(
            f"/api/v2/missions/{mission_id}/nodes/navigate-a/skill-requests",
            headers=_headers(),
        )
        assert issued_a.status_code == 202, issued_a.text
        assert len(driver.dispatches) == 1
        failed = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=_headers(),
            json={"maximumSteps": 1},
        )
        assert failed.status_code == 200, failed.text
        assert failed.json()["mission"]["snapshot"]["status"] == "planning"

        for kind in ("pause", "resume"):
            bypass = client.post(
                f"/api/v2/missions/{mission_id}/interventions",
                headers=_headers(),
                json={
                    "interventionId": f"parallel-planning-{kind}",
                    "kind": kind,
                    "text": f"must not {kind} unresolved planning",
                },
            )
            assert bypass.status_code == 409, bypass.text
        assert client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=_headers(),
        ).status_code == 409
        with pytest.raises(MissionConflict, match="cannot skip nodes from planning"):
            app.state.mission_runtime.skip_nodes(
                mission_id,
                ["navigate-b"],
                reason="must not bypass failed-skill planning",
                causation_id="parallel-planning-skip",
            )

        issued_b_while_planning = client.post(
            f"/api/v2/missions/{mission_id}/nodes/navigate-b/skill-requests",
            headers=_headers(),
        )
        assert issued_b_while_planning.status_code == 409, (
            issued_b_while_planning.text
        )
        assert len(driver.dispatches) == 1

        outcomes = app.state.mission.converge_supervised_missions()
        assert len(outcomes) == 1
        assert not any(
            action.get("kind") == "skill_issued"
            for action in outcomes[0]["actions"]
        )
        pending = app.state.mission_runtime.get(mission_id)
        assert pending["snapshot"]["status"] == "awaiting_review"
        assert len(driver.dispatches) == 1

        revision_id = pending["revisions"][-1]["revisionId"]
        reapproved = client.post(
            f"/api/v2/missions/{mission_id}/revisions/{revision_id}/approve",
            headers=_headers(),
            json={"approvedBy": "operator-test"},
        )
        assert reapproved.status_code == 200, reapproved.text
        assert reapproved.json()["snapshot"]["status"] == "ready"
        paused = client.post(
            f"/api/v2/missions/{mission_id}/interventions",
            headers=_headers(),
            json={
                "interventionId": "parallel-ready-pause",
                "kind": "pause",
                "text": "operator pause before retry",
            },
        )
        assert paused.status_code == 200, paused.text
        assert paused.json()["snapshot"]["status"] == "suspended"
        resumed = client.post(
            f"/api/v2/missions/{mission_id}/interventions",
            headers=_headers(),
            json={
                "interventionId": "parallel-ready-resume",
                "kind": "resume",
                "text": "operator resume after pause",
            },
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["snapshot"]["status"] == "ready"
        assert resumed.json()["snapshot"]["errorCode"] is None
        issued_b_after_review = client.post(
            f"/api/v2/missions/{mission_id}/nodes/navigate-b/skill-requests",
            headers=_headers(),
        )
        assert issued_b_after_review.status_code == 202, issued_b_after_review.text
        assert len(driver.dispatches) == 2


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
