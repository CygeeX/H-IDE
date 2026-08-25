from __future__ import annotations

from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from fastapi.testclient import TestClient
import pytest

from packages.control_core.adapters import AdapterResult, RobotAgentAdapter
from packages.control_core.validation import utc_now
from services.control_api.main import create_app
from services.control_api.auth import Role
from services.control_api.autonomy_policy import ExecutionTarget
from services.control_api.mission_policy import MissionPolicy, PolicyActor
from services.control_api.mission_service import MissionServiceError
from services.control_api.store import SQLiteStore
from tests.test_mission_v2_api import (
    FakeMissionPlanningClient,
    MissionRobotDriver,
    RespondingMissionPlanningClient,
)


LEGACY_TOKEN = "policy-acl-legacy-token"
INTERNAL_TOKEN = "policy-acl-internal-token"
PASSWORD = "correct horse battery staple"


@pytest.fixture(autouse=True)
def policy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISSION_V2_INTERNAL_TOKEN", INTERNAL_TOKEN)
    monkeypatch.delenv("MISSION_POLICY_ALLOW_LEGACY_REAL", raising=False)


def test_simulation_grant_ttl_is_long_without_weakening_real_target_ttl(
    tmp_path,
) -> None:
    store = SQLiteStore(tmp_path / "policy-ttl.sqlite3")
    policy = MissionPolicy(
        store,
        grant_ttl_seconds=600,
        simulation_grant_ttl_seconds=3600,
    )

    assert policy.grant_ttl_seconds == 600
    assert policy.simulation_grant_ttl_seconds == 3600
    assert policy.grant_ttl_for_target(ExecutionTarget.SIM) == 3600
    assert policy.grant_ttl_for_target(ExecutionTarget.REAL) == 600


def _register_login(
    client: TestClient, username: str
) -> tuple[dict[str, Any], dict[str, str]]:
    registered = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": PASSWORD},
    )
    assert registered.status_code == 201, registered.text
    user = registered.json()["user"]
    logged_in = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": PASSWORD},
    )
    assert logged_in.status_code == 200, logged_in.text
    return user, {"X-CSRF-Token": logged_in.json()["csrfToken"]}


def _create_draft(
    client: TestClient,
    headers: Mapping[str, str],
    *,
    mission_id: str,
) -> dict[str, Any]:
    response = client.post(
        "/api/v2/missions/drafts",
        headers=dict(headers),
        json={
            "missionId": mission_id,
            "requestId": f"request-{mission_id}",
            "robotId": "robot-1",
            "objective": "去厨房",
            "requestedBy": "forged-admin",
            "authorizationId": "forged-authorization",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_mission_owner_acl_and_server_identity_are_not_client_controlled() -> None:
    app = create_app(
        auth_token=LEGACY_TOKEN,
        mission_planning_client=FakeMissionPlanningClient(),
    )
    alice_client = TestClient(app)
    bob_client = TestClient(app)
    try:
        alice, alice_headers = _register_login(alice_client, "policy-alice")
        _, bob_headers = _register_login(bob_client, "policy-bob")
        draft = _create_draft(
            alice_client, alice_headers, mission_id="mission-owned-by-alice"
        )
        binding = app.state.store.get_resource_owner(
            "mission_draft", draft["draftId"]
        )
        assert binding == {
            **binding,
            "ownerUserId": alice["userId"],
            "siteId": alice["siteId"],
        }

        # Cross-owner existence is hidden from reads, mutations and list APIs.
        assert bob_client.get(
            f"/api/v2/missions/drafts/{draft['draftId']}"
        ).status_code == 404
        assert bob_client.post(
            f"/api/v2/missions/drafts/{draft['draftId']}/approve",
            headers=bob_headers,
            json={"approvedBy": alice["userId"]},
        ).status_code == 404
        assert draft["draftId"] not in {
            item["draftId"]
            for item in bob_client.get("/api/v2/missions/drafts").json()
        }

        approved = alice_client.post(
            f"/api/v2/missions/drafts/{draft['draftId']}/approve",
            headers=alice_headers,
            json={
                "approvedBy": "forged-admin",
                "authorizationId": "forged-authorization",
            },
        )
        assert approved.status_code == 200, approved.text
        mission = approved.json()
        assert mission["authorization"]["approvedBy"] == alice["userId"]
        assert mission["authorization"]["authorizationId"] != "forged-authorization"
        mission_binding = app.state.store.get_resource_owner(
            "mission", draft["draftId"]
        )
        assert mission_binding is not None
        assert mission_binding["ownerUserId"] == alice["userId"]
        revision_binding = app.state.store.get_resource_owner(
            "mission_revision", mission["activeRevision"]["revisionId"]
        )
        assert revision_binding is not None
        assert revision_binding["ownerUserId"] == alice["userId"]
        assert revision_binding["siteId"] == alice["siteId"]
        assert bob_client.get(
            f"/api/v2/missions/{draft['draftId']}"
        ).status_code == 404
    finally:
        alice_client.close()
        bob_client.close()


def test_preferences_are_separate_and_resource_bindings_are_immutable() -> None:
    app = create_app(auth_token=LEGACY_TOKEN)
    with TestClient(app) as client:
        user, headers = _register_login(client, "policy-preference")
        defaults = client.get("/api/v2/autonomy/preferences").json()["preferences"]
        assert {(item["executionTarget"], item["mode"], item["version"]) for item in defaults} == {
            ("sim", "ASK_EVERY_ACTION", 0),
            ("real", "ASK_EVERY_ACTION", 0),
        }
        changed = client.put(
            "/api/v2/autonomy/preferences/sim",
            headers=headers,
            json={"mode": "AUTO_WITHIN_LIMITS", "expectedVersion": 0},
        )
        assert changed.status_code == 200, changed.text
        values = client.get("/api/v2/autonomy/preferences").json()["preferences"]
        assert {(item["executionTarget"], item["mode"]) for item in values} == {
            ("sim", "AUTO_WITHIN_LIMITS"),
            ("real", "ASK_EVERY_ACTION"),
        }
        assert changed.json()["userId"] == user["userId"]

    store = SQLiteStore(":memory:")
    store.bind_robot_site("immutable-robot", site_id="site-a", execution_target="sim")
    with pytest.raises(ValueError, match="immutable"):
        store.bind_robot_site(
            "immutable-robot", site_id="site-b", execution_target="real"
        )


def test_legacy_bearer_cannot_authorize_real_motion_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    driver = MissionRobotDriver()
    app = create_app(
        auth_token=LEGACY_TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    legacy = {"Authorization": f"Bearer {LEGACY_TOKEN}"}
    with TestClient(app) as client:
        draft = _create_draft(
            client, legacy, mission_id="mission-legacy-real-denied"
        )
        denied = client.post(
            f"/api/v2/missions/drafts/{draft['draftId']}/approve",
            headers=legacy,
            json={"approvedBy": "pretend-admin"},
        )
        assert denied.status_code == 403
        assert "legacy bearer" in denied.json()["detail"]
        assert app.state.store.get_mission(draft["draftId"]) is None
        assert app.state.store.list_autonomy_policy_decisions(
            plan_id=draft["draftId"]
        ) == []
        assert driver.dispatches == []


class _StaleAtDispatchDriver(MissionRobotDriver):
    stale = False

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        discovered = super().discover(robot_id)
        result = dict(discovered.result)
        if self.stale:
            result["currentPose"] = {
                **dict(result["currentPose"]),
                "observedAt": (utc_now() - timedelta(seconds=60)).isoformat(),
            }
            result["observation"] = {
                **dict(result["observation"]),
                "status": "STALE",
            }
        return AdapterResult(status=discovered.status, result=result)


def test_fresh_dispatch_gate_blocks_state_that_became_stale_after_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("MISSION_POLICY_ALLOW_LEGACY_REAL", "true")
    driver = _StaleAtDispatchDriver()
    app = create_app(
        auth_token=LEGACY_TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    headers = {
        "Authorization": f"Bearer {LEGACY_TOKEN}",
        "X-H-IDE-Mission-Internal-Token": INTERNAL_TOKEN,
    }
    with TestClient(app) as client:
        draft = _create_draft(
            client, headers, mission_id="mission-stale-at-dispatch"
        )
        assert client.post(
            f"/api/v2/missions/drafts/{draft['draftId']}/approve", headers=headers
        ).status_code == 200
        assert client.post(
            f"/api/v2/missions/{draft['draftId']}/start", headers=headers
        ).status_code == 200
        target = client.post(
            f"/api/v2/missions/{draft['draftId']}/world-records",
            headers=headers,
            json={
                "recordId": "stale-dispatch-target",
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "厨房",
                "predicate": "target_pose",
                "value": {
                    "targetPose": {
                        "frameId": "map",
                        "x": 1.0,
                        "y": 0.0,
                        "z": 0.0,
                        "yaw": 0.0,
                    }
                },
                "source": "geometry",
                "sourceRef": "trusted-geometry",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
                "frameId": "map",
            },
        )
        assert target.status_code == 200, target.text
        original_dispatcher = app.state.mission.skill_dispatcher
        assert original_dispatcher is not None

        def become_stale(skill):
            driver.stale = True
            return original_dispatcher(skill)

        app.state.mission.skill_dispatcher = become_stale
        issued = client.post(
            f"/api/v2/missions/{draft['draftId']}/nodes/navigate-kitchen/skill-requests",
            headers=headers,
        )
        assert issued.status_code in {403, 409}, issued.text
        assert "policy" in issued.json()["detail"]
        assert driver.dispatches == []
        decisions = app.state.store.list_autonomy_policy_decisions(
            plan_id=draft["draftId"]
        )
        assert any(
            item["operation"] == "skill.dispatch"
            and item["effect"] == "DENY"
            and item["code"] == "deny.health"
            for item in decisions
        )


def test_internal_callback_cannot_be_forged_by_browser_principal() -> None:
    app = create_app(
        auth_token=LEGACY_TOKEN,
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        _, headers = _register_login(client, "policy-callback-user")
        missing = client.post(
            "/api/v2/missions/not-visible/skill-results",
            headers=headers,
            json={},
        )
        assert missing.status_code == 404
        forged = client.post(
            "/api/v2/missions/not-visible/skill-results",
            headers={**headers, "X-H-IDE-Mission-Internal-Token": "wrong"},
            json={},
        )
        assert forged.status_code == 404


def test_every_physical_replan_requires_human_approval_and_rotates_exact_grant() -> None:
    app = create_app(
        auth_token=LEGACY_TOKEN,
        mission_planning_client=RespondingMissionPlanningClient(),
    )
    with TestClient(app) as client:
        user, headers = _register_login(client, "policy-replanner")
        draft = _create_draft(
            client, headers, mission_id="mission-physical-replan"
        )
        approved = client.post(
            f"/api/v2/missions/drafts/{draft['draftId']}/approve",
            headers=headers,
            json={"approvedBy": "forged-first-approver"},
        )
        assert approved.status_code == 200, approved.text
        first_authorization = approved.json()["authorization"]["authorizationId"]
        first_grant = app.state.store.get_autonomy_grant_by_authorization(
            first_authorization
        )
        assert first_grant is not None
        assert first_grant["subjectUserId"] == user["userId"]
        assert first_grant["planRevision"] == 1

        replanned = client.post(
            f"/api/v2/missions/{draft['draftId']}/messages",
            headers=headers,
            json={
                "text": "改到厨房门外",
                "kind": "modify_goal",
                "requestedBy": "forged-requester",
                "authorizationId": first_authorization,
            },
        )
        assert replanned.status_code == 200, replanned.text
        pending = replanned.json()["decision"]["revision"]
        assert pending["disposition"] == "human_required"
        assert pending["authorizationId"] is None
        assert replanned.json()["mission"]["snapshot"]["status"] == "awaiting_review"

        second = client.post(
            f"/api/v2/missions/{draft['draftId']}/revisions/{pending['revisionId']}/approve",
            headers=headers,
            json={"approvedBy": "forged-second-approver"},
        )
        assert second.status_code == 200, second.text
        second_mission = second.json()
        second_authorization = second_mission["authorization"]["authorizationId"]
        assert second_authorization != first_authorization
        assert second_mission["authorization"]["approvedBy"] == user["userId"]
        revoked_first = app.state.store.get_autonomy_grant_by_authorization(
            first_authorization
        )
        second_grant = app.state.store.get_autonomy_grant_by_authorization(
            second_authorization
        )
        assert revoked_first is not None and revoked_first["revokedAt"] is not None
        assert second_grant is not None
        assert second_grant["revokedAt"] is None
        assert second_grant["planRevision"] == second_mission["activeRevision"]["sequence"]
        assert second_grant["subjectUserId"] == user["userId"]
        assert set(second_grant["actionScope"]) == {"navigation.navigate_to_pose"}
        for revision in second_mission["revisions"]:
            revision_binding = app.state.store.get_resource_owner(
                "mission_revision", revision["revisionId"]
            )
            assert revision_binding is not None
            assert revision_binding["ownerUserId"] == user["userId"]
            assert revision_binding["siteId"] == user["siteId"]


def test_concurrent_approval_has_one_winner_and_one_atomic_grant() -> None:
    app = create_app(
        auth_token=LEGACY_TOKEN,
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        user, headers = _register_login(client, "policy-concurrent-approver")
        draft = _create_draft(
            client, headers, mission_id="mission-concurrent-approval"
        )
        actor = PolicyActor(
            user_id=user["userId"],
            role=Role.USER,
            site_id=user["siteId"],
        )

        def approve() -> str:
            try:
                app.state.mission.approve_draft(
                    draft["draftId"],
                    user["userId"],
                    actor=actor,
                    resource_owner_user_id=user["userId"],
                    resource_site_id=user["siteId"],
                )
            except MissionServiceError:
                return "conflict"
            return "approved"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(lambda _: approve(), range(2)))
        assert sorted(outcomes) == ["approved", "conflict"]
        mission = app.state.store.get_mission(draft["draftId"])
        assert mission is not None
        grant = app.state.store.get_autonomy_grant_by_authorization(
            mission["authorization"]["authorizationId"]
        )
        assert grant is not None and grant["revokedAt"] is None
        decisions = app.state.store.list_autonomy_policy_decisions(
            plan_id=draft["draftId"]
        )
        assert len([item for item in decisions if item["operation"] == "plan.confirm"]) == 1
