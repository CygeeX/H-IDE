from fastapi.testclient import TestClient

from services.robot_agent_gateway.main import create_app
from services.robot_agent_gateway import ProtocolSimulator


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
