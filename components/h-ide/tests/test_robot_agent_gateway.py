from fastapi.testclient import TestClient

from services.robot_agent_gateway.main import create_app
from services.robot_agent_gateway import ProtocolSimulator

from .test_robot_agent_protocol_simulator import mapping, request


def test_gateway_health_never_claims_real_robot_or_nav2() -> None:
    client = TestClient(create_app(ProtocolSimulator()))
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "protocol": "robot-agent.v1",
        "executionBackend": "protocol_simulator",
        "motion": "simulated",
        "releaseState": "code_complete_robot_agent_pending",
    }


def test_gateway_internal_routes_require_auth_and_drive_the_driver() -> None:
    peer = ProtocolSimulator()
    client = TestClient(create_app(peer, internal_token="gateway-test-token"))
    assert client.get("/internal/v1/discover").status_code == 401
    response = client.get(
        "/internal/v1/discover?robotId=robot-1",
        headers={"Authorization": "Bearer gateway-test-token"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "available"
    assert response.json()["result"]["executionBackend"] == "protocol_simulator"


def test_http_cancel_before_dispatch_suppresses_the_late_command() -> None:
    peer = ProtocolSimulator(terminal="RUNNING")
    client = TestClient(create_app(peer, internal_token="gateway-test-token"))
    headers = {"Authorization": "Bearer gateway-test-token"}
    command = request("ffffffff-ffff-4fff-8fff-ffffffffffff")
    body = {
        "request": command.model_dump(mode="json"),
        "mapping": mapping().model_dump(mode="json"),
    }

    cancelled = client.post(
        f"/internal/v1/commands/{command.commandId}/cancel",
        headers=headers,
    )
    late = client.post("/internal/v1/commands", headers=headers, json=body)

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["result"]["dispatchSuppressed"] is True
    assert late.status_code == 200
    assert late.json()["status"] == "cancelled"
    assert late.json()["result"] == cancelled.json()["result"]
    assert peer.submit_count == 0
    assert peer.control_state == "UNARMED"
