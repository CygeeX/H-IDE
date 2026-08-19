from __future__ import annotations

from packages.control_core.adapters import RobotAgentAdapter
from services.robot_agent_gateway import (
    ProtocolSimulator,
    RobotAgentGatewayDriver,
    RobotAgentWebSocketPeer,
    WebSocketProtocolSimulator,
)

from .test_robot_agent_protocol_simulator import mapping, request


def test_gateway_uses_real_robot_agent_v1_websocket_envelopes() -> None:
    backend = ProtocolSimulator()
    with WebSocketProtocolSimulator(backend) as server:
        peer = RobotAgentWebSocketPeer(
            server.uri,
            robot_id="robot-1",
            expected_capabilities_digest=server.capabilities_digest,
            execution_backend="protocol_simulator",
            motion="simulated",
            allow_plaintext_loopback=True,
            heartbeat=False,
        )
        adapter = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
        discovered = adapter.discover("robot-1")
        assert discovered.status == "available"
        assert discovered.result["protocol"] == "robot-agent.v1"
        assert discovered.result["robotBootId"] == backend.robot_boot_id
        assert adapter.probe(mapping()).status == "probed"
        command = request()
        first = adapter.dispatch(command, mapping())
        second = adapter.query(command.commandId)
        assert first.status == second.status == "succeeded"
        assert first.result["executionBackend"] == "protocol_simulator"
        assert first.result["motion"] == "simulated"
        assert backend.submit_count == 1
        peer.close()


def test_websocket_simulator_cannot_claim_robot_execute() -> None:
    try:
        WebSocketProtocolSimulator(mode="robot_execute")
    except ValueError as exc:
        assert "robot_execute" in str(exc)
    else:
        raise AssertionError("protocol simulator must reject robot_execute")
