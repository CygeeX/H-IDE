from __future__ import annotations

from fastapi.testclient import TestClient

from packages.control_core.adapters import RobotAgentAdapter
from packages.visual_mission_core import parse_plan_draft
from services.control_api.main import create_app
from services.robot_agent_gateway import ProtocolSimulator, RobotAgentGatewayHttpDriver
from services.robot_agent_gateway import RobotAgentWebSocketPeer, WebSocketProtocolSimulator
from services.robot_agent_gateway.main import create_app as create_gateway_app

from .test_visual_mission_contracts import visual_plan
from .test_visual_mission_runtime import StageClient, VisionClient, resolution


def test_reviewed_fixture_uses_production_robot_agent_adapter_not_fake(monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "lab-map-epoch-1")
    peer = ProtocolSimulator(
        localization_epoch_id="lab-map-epoch-1",
        observation_session_id="fixture-session-1",
    )
    with WebSocketProtocolSimulator(peer) as websocket_server:
        websocket_peer = RobotAgentWebSocketPeer(
            websocket_server.uri,
            robot_id="robot-1",
            expected_capabilities_digest=websocket_server.capabilities_digest,
            execution_backend="protocol_simulator",
            motion="simulated",
            allow_plaintext_loopback=True,
            heartbeat=False,
        )
        gateway_client = TestClient(
            create_gateway_app(websocket_peer, internal_token="fixture-gateway-token")
        )
        adapter = RobotAgentAdapter(driver=RobotAgentGatewayHttpDriver(
            "http://robot-agent-gateway",
            "fixture-gateway-token",
            client=gateway_client,
        ))
        app = create_app(
            auth_token="fixture-e2e-token",
            adapters={"robot_agent": adapter},
            vision_client=VisionClient(resolution(source={
                **resolution()["source"],
                "robotBootId": "11111111-1111-4111-8111-111111111111",
            })),
            visual_stage_client=StageClient(),
            visual_mission_mode="lab_execute",
        )
        client = TestClient(app)
        headers = {"Authorization": "Bearer fixture-e2e-token"}
        plan = parse_plan_draft(
            visual_plan(robotBootId="11111111-1111-4111-8111-111111111111")
        ).model_dump(mode="json")

        assert client.post("/api/v1/assistant/drafts", headers=headers, json=plan).status_code == 200
        confirmed = client.post(
            "/api/v1/assistant/drafts/visual-plan-1/confirm",
            headers=headers,
            json={"confirmed": True},
        )
        assert confirmed.status_code == 200, confirmed.text
        completed = client.post("/api/v1/visual-missions/visual-plan-1/run", headers=headers)
        assert completed.status_code == 200, completed.text
        body = completed.json()
        assert body["status"] == "completed"
        assert body["motion"] == "simulated"
        assert body["releaseState"] == "code_complete_robot_agent_pending"
        assert peer.submit_count == 1
        command, status = next(iter(peer.commands.values()))
        assert command.kind == "navigation.navigate_to_pose"
        assert command.params["visualResolutionId"] == "vr-1"
        assert status.result["executionBackend"] == "protocol_simulator"
        assert status.result["motion"] == "simulated"
        assert status.result["rosGoalId"] is None
        assert app.state.control.mapping("navigation.navigate_to_pose").transport.value == "robot_agent"
        assert "fake" not in app.state.runtime.adapters

        repeated = client.post("/api/v1/visual-missions/visual-plan-1/run", headers=headers)
        assert repeated.status_code == 200
        assert peer.submit_count == 1
        websocket_peer.close()
