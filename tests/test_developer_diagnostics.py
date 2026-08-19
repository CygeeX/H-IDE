from __future__ import annotations

import json
from typing import Any, Mapping

from fastapi.testclient import TestClient

from services.control_api.developer_diagnostics import (
    DeveloperDiagnosticsService,
    ProbeResult,
)
from services.control_api.auth import Principal, Role
from services.control_api.main import create_app
from services.control_api.store import SQLiteStore


TOKEN = "developer-diagnostics-test-token"
PASSWORD = "correct horse battery staple"


class StubProbe:
    def __init__(self, responses: Mapping[str, ProbeResult]) -> None:
        self.responses = dict(responses)
        self.urls: list[str] = []

    def get(self, url: str) -> ProbeResult:
        self.urls.append(url)
        return self.responses.get(url, ProbeResult(None, {}, None, "unexpected_probe"))


def result(payload: Mapping[str, Any], status: int = 200, latency: float = 4.5) -> ProbeResult:
    return ProbeResult(status, payload, latency)


def healthy_responses() -> dict[str, ProbeResult]:
    return {
        "http://agent/healthz": result({"status": "ok"}),
        "http://agent/readyz": result({"status": "ready", "reasons": []}),
        "http://vision/healthz": result(
            {
                "status": "ok",
                "releaseState": "pure_vlm_product_profile",
                "capabilities": {
                    "vlm": {"provider": "qwen", "model": "qwen3-vl-plus", "available": True},
                    "segmentation": {"provider": "sam2", "available": True},
                },
                "apiKey": "secret-must-never-cross-the-bff",
            },
            latency=18.25,
        ),
        "http://sim-agent/healthz": result({"status": "alive"}),
        "http://sim-agent/v1/diagnostics": result(
            {
                "schemaVersion": "robot-agent.diagnostics/1",
                "robotId": "g1-sim",
                "ready": True,
                "secretToken": "secret-must-never-cross-the-bff",
                "rawAction": {"goalPose": {"x": 9, "y": 8, "yaw": 1.2}},
                "commandReadiness": {"ready": True, "mode": "STANDBY", "reason": None},
                "controlState": {"standby": True},
                "checks": {
                    "transportSession": {"required": True, "ready": True, "reason": None},
                    "control": {
                        "required": True,
                        "ready": True,
                        "state": "SAFE_HOLD",
                        "mode": "STANDBY",
                        "leaseValid": False,
                        "activeCommandCount": 0,
                        "reason": "standby_waiting_for_control_acquire",
                    },
                    "rosExecutor": {"required": True, "ready": True, "reason": None},
                    "nav2Actions": {
                        "required": True,
                        "ready": True,
                        "reason": None,
                        "actions": {
                            "navigation.navigate_to_pose": {"ready": True, "reason": None},
                            "navigation.rotate": {"ready": True, "reason": None},
                        },
                    },
                    "localization": {
                        "required": True,
                        "ready": True,
                        "ageMs": 31.4,
                        "maxAgeMs": 500,
                    },
                    "observation": {
                        "required": True,
                        "ready": True,
                        "status": "USABLE",
                        "captureAgeMs": 44.2,
                        "healthAgeMs": 22.1,
                        "gatewayAckAgeMs": 17.5,
                        "maxAgeMs": 750,
                    },
                },
            },
            latency=2.25,
        ),
        "http://sim-gateway/healthz": result(
            {
                "status": "ok",
                "protocol": "robot-agent.v1",
                "executionBackend": "websocket_inbound",
                "robotAgent": {
                    "connected": True,
                    "robotId": "g1-sim",
                    "robotBootId": "private-boot-id",
                    "controlState": "SAFE_HOLD",
                    "lastError": None,
                    "observation": {"status": "USABLE", "rawFrame": "must-not-cross"},
                },
            }
        ),
        "http://sonic/healthz": result(
            {
                "status": "running",
                "motion_ready": True,
                "sonic_mode": "IDLE",
                "nav2_state": "idle",
                "control_clock_status": "simulation_time_50hz_motion_ready",
                "last_error": None,
                "rgb_frames": 202,
                "depth_frames": 202,
                "policy_tensor": [1, 2, 3],
            }
        ),
        "http://plant/healthz": result(
            {
                "status": "running",
                "sequence": 202,
                "pose": {"x": -6.45, "y": 0, "z": 0.8},
                "rgb": {"width": 640, "height": 480, "encoding": "jpeg"},
                "depth": {"width": 640, "height": 480, "encoding": "z16_le"},
                "support": {
                    "mode": "static_hold",
                    "safetyStopLatched": False,
                    "fault": None,
                    "privilegedControl": "must-not-cross",
                },
            }
        ),
    }


def service(store: SQLiteStore, probe: StubProbe) -> DeveloperDiagnosticsService:
    return DeveloperDiagnosticsService(
        store,
        probe=probe,
        planning_url="http://agent",
        vision_url="http://vision",
        sonic_url="http://sonic",
        plant_url="http://plant",
        robot_agent_url="http://real-agent",
        simulation_robot_agent_url="http://sim-agent",
        robot_agent_gateway_url="http://real-gateway",
        simulation_robot_agent_gateway_url="http://sim-gateway",
        primary_robot_id="g1-dev",
        simulation_robot_id="g1-sim",
        ros_distro="jazzy",
    )


def test_developer_bff_projects_health_without_secrets_images_poses_or_raw_actions() -> None:
    store = SQLiteStore(":memory:")
    probe = StubProbe(healthy_responses())
    app = create_app(
        auth_token=TOKEN,
        developer_diagnostics_service=service(store, probe),
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/v2/developer/diagnostics?robotId=g1-sim",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "private, no-store, max-age=0"
    payload = response.json()
    assert payload["summary"] == {
        "status": "ready",
        "ready": True,
        "processAliveIsNotReadiness": True,
        "detail": "All required execution dependencies report ready",
    }
    assert payload["release"]["rosDistro"] == "jazzy"
    assert payload["components"]["vision"]["latestAnalysis"] is None
    assert payload["components"]["vision"]["provider"] == "qwen"
    assert payload["components"]["vision"]["model"] == "qwen3-vl-plus"
    assert payload["components"]["robotAgent"]["control"]["mode"] == "STANDBY"
    assert payload["components"]["gateway"]["connected"] is True
    assert payload["components"]["nav2"]["actions"]["navigation.rotate"]["ready"] is True
    assert payload["components"]["localization"]["poseAgeMs"] == 31.4
    assert payload["components"]["localization"]["tfReady"] is None
    assert payload["components"]["observation"]["captureAgeMs"] == 44.2
    assert payload["components"]["sonic"]["motionReady"] is True
    assert payload["components"]["plant"]["rgbReady"] is True
    assert payload["components"]["plant"]["depthReady"] is True
    assert payload["components"]["plant"]["safetyStopLatched"] is False
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "secret-must-never-cross-the-bff" not in serialized
    assert "goalPose" not in serialized
    assert '"pose"' not in serialized
    assert "policy_tensor" not in serialized
    assert "private-boot-id" not in serialized
    assert "rawFrame" not in serialized
    assert "privilegedControl" not in serialized
    assert "http://sim-gateway/healthz" in probe.urls
    assert all(url.startswith("http://") for url in probe.urls)
    store.close()


def test_alive_robot_agent_is_not_reported_ready_when_control_or_nav2_is_blocked() -> None:
    responses = healthy_responses()
    diagnostics = dict(responses["http://sim-agent/v1/diagnostics"].payload)
    diagnostics["ready"] = False
    diagnostics["commandReadiness"] = {
        "ready": False,
        "mode": "BLOCKED",
        "reason": "required_dependency_unavailable",
    }
    checks = dict(diagnostics["checks"])
    checks["control"] = {
        "required": True,
        "ready": False,
        "state": "SAFE_HOLD",
        "mode": "BLOCKED",
        "leaseValid": False,
        "activeCommandCount": 1,
        "reason": "active_commands_without_valid_lease",
    }
    checks["nav2Actions"] = {
        "required": True,
        "ready": False,
        "reason": "one_or_more_nav2_actions_unavailable",
        "actions": {"navigation.navigate_to_pose": {"ready": False, "reason": "action_server_missing"}},
    }
    diagnostics["checks"] = checks
    responses["http://sim-agent/v1/diagnostics"] = result(diagnostics)
    store = SQLiteStore(":memory:")
    app = create_app(
        auth_token=TOKEN,
        developer_diagnostics_service=service(store, StubProbe(responses)),
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/v2/developer/diagnostics?robotId=g1-sim",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["status"] == "degraded"
    assert payload["components"]["robotAgent"]["alive"] is True
    assert payload["components"]["robotAgent"]["ready"] is False
    assert payload["components"]["robotAgent"]["control"]["reason"] == "active_commands_without_valid_lease"
    assert payload["components"]["nav2"]["ready"] is False
    assert payload["components"]["nav2"]["actions"]["navigation.navigate_to_pose"]["reason"] == "action_server_missing"
    store.close()


def test_latched_plant_is_alive_but_not_motion_ready() -> None:
    responses = healthy_responses()
    plant = dict(responses["http://plant/healthz"].payload)
    plant["status"] = "running"
    plant["support"] = {
        "mode": "safety_stop",
        "safetyStopLatched": True,
        "fault": None,
    }
    responses["http://plant/healthz"] = result(plant)
    store = SQLiteStore(":memory:")
    app = create_app(
        auth_token=TOKEN,
        developer_diagnostics_service=service(store, StubProbe(responses)),
    )
    with TestClient(app) as client:
        response = client.get(
            "/api/v2/developer/diagnostics?robotId=g1-sim",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["plant"]["alive"] is True
    assert payload["components"]["plant"]["ready"] is False
    assert payload["components"]["plant"]["safetyStopLatched"] is True
    assert payload["summary"]["ready"] is False
    store.close()


def test_simulation_gateway_default_is_current_8094_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("SIMULATION_ROBOT_AGENT_GATEWAY_URL", raising=False)
    store = SQLiteStore(":memory:")
    diagnostics = DeveloperDiagnosticsService(store, probe=StubProbe({}))
    assert diagnostics.simulation_robot_agent_gateway_url == "http://127.0.0.1:8094"
    store.close()


def test_diagnostics_requires_developer_scope_and_exposes_no_mutating_route() -> None:
    store = SQLiteStore(":memory:")
    app = create_app(
        auth_token=TOKEN,
        developer_diagnostics_service=service(store, StubProbe(healthy_responses())),
    )
    with TestClient(app) as client:
        registered = client.post(
            "/api/v1/auth/register",
            json={"username": "normal-user", "password": PASSWORD},
        )
        assert registered.status_code == 201
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "normal-user", "password": PASSWORD},
        )
        assert login.status_code == 200
        denied = client.get("/api/v2/developer/diagnostics?robotId=g1-sim")
        assert denied.status_code == 403
        assert "robotId" not in denied.text
        assert "coordinates" not in denied.text
        assert "ROS" not in denied.text

        method_denied = client.post(
            "/api/v2/developer/diagnostics?robotId=g1-sim",
            headers={"X-CSRF-Token": login.json()["csrfToken"]},
        )
        assert method_denied.status_code == 405

        unknown = client.get(
            "/api/v2/developer/diagnostics?robotId=unconfigured-robot",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert unknown.status_code == 404
    store.close()


def test_diagnostics_service_is_site_scoped_for_non_admin_developers() -> None:
    store = SQLiteStore(":memory:")
    diagnostics = service(store, StubProbe(healthy_responses()))
    other_site_developer = Principal(
        user_id="developer-other-site",
        username="developer-other-site",
        display_name="Other Site Developer",
        role=Role.DEVELOPER,
        site_id="other-site",
        authentication_method="session",
        session_id="session-other-site",
    )
    try:
        diagnostics.snapshot(principal=other_site_developer, robot_id="g1-sim")
    except PermissionError as exc:
        assert "another site" in str(exc)
    else:
        raise AssertionError("cross-site developer diagnostics must fail closed")
    assert diagnostics.probe.urls == []
    store.close()


def test_latest_visual_diagnostic_projects_only_site_scoped_metadata() -> None:
    store = SQLiteStore(":memory:")
    for site, suffix in (("site-a", "a"), ("site-b", "b")):
        store.create_user(
            user_id=f"developer-{suffix}",
            username=f"developer-{suffix}",
            display_name=f"Developer {suffix.upper()}",
            password_hash="test-only-hash",
            role="DEVELOPER",
            site_id=site,
        )
        conversation_id = f"conversation-{suffix}"
        task_id = f"task-{suffix}"
        store.create_assistant_conversation(
            conversation_id,
            "g1-sim",
            f"developer-{suffix}",
            "zh-CN",
        )
        store.bind_resource_owner(
            "assistant_conversation",
            conversation_id,
            owner_user_id=f"developer-{suffix}",
            site_id=site,
        )
        store.create_assistant_turn(
            {
                "taskId": task_id,
                "conversationId": conversation_id,
                "text": "只观察",
                "modality": "text",
            },
            message_id=f"message-{suffix}",
        )
        assert store.mark_assistant_turn_running(task_id)
        assert store.complete_assistant_turn(
            task_id,
            status="completed",
            result={
                "routeStatus": "completed",
                "visualResult": {"status": "succeeded", "privatePrompt": "must-not-cross"},
                "evidence": [
                    {"kind": f"rgb-{suffix}", "sha256": "a" * 64, "content": "must-not-cross"},
                    {"kind": "semantic_json", "sha256": "b" * 64},
                ],
            },
            assistant_message="观察完成",
            message_id=f"assistant-message-{suffix}",
            conversation_status="idle",
        )

    projected = store.latest_visual_diagnostic(site_id="site-a")
    assert projected is not None
    assert projected["status"] == "succeeded"
    assert projected["evidenceCount"] == 2
    assert projected["evidenceKinds"] == ["rgb-a", "semantic_json"]
    assert projected["latencyMs"] is not None
    serialized = json.dumps(projected)
    assert "rgb-b" not in serialized
    assert "must-not-cross" not in serialized
    assert "sha256" not in serialized
    store.close()
