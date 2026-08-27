from __future__ import annotations

from datetime import timedelta

from packages.control_core.adapters import AdapterResult
from packages.control_core.models import (
    CommandRequest,
    CommandSource,
    MappingVersion,
    Transport,
    utc_now,
)
from services.control_api.main import _RobotAgentGatewayRouter


class _Driver:
    def __init__(self, name: str) -> None:
        self.name = name
        self.discovered: list[str | None] = []
        self.dispatched: list[str] = []

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        self.discovered.append(robot_id)
        return AdapterResult(status="available", result={"robotId": robot_id, "driver": self.name})

    def probe(self, mapping: MappingVersion) -> AdapterResult:
        return AdapterResult(status="probed", result={"driver": self.name})

    def dispatch(self, request: CommandRequest, mapping: MappingVersion) -> AdapterResult:
        del mapping
        self.dispatched.append(request.commandId)
        return AdapterResult(status="accepted", result={"driver": self.name})

    def cancel(self, command_id: str) -> AdapterResult:
        return AdapterResult(status="cancelled", result={"driver": self.name, "commandId": command_id})

    def query(self, command_id: str) -> AdapterResult:
        return AdapterResult(status="succeeded", result={"driver": self.name, "commandId": command_id})


def test_robot_agent_gateway_router_keeps_simulation_commands_on_the_sim_peer() -> None:
    live = _Driver("live")
    simulation = _Driver("simulation")
    router = _RobotAgentGatewayRouter(live, {"g1-sim": simulation})

    assert router.discover("g1-dev").result["driver"] == "live"
    assert router.discover("g1-sim").result["driver"] == "simulation"

    now = utc_now()
    request = CommandRequest(
        commandId="command-sim-1",
        robotId="g1-sim",
        source=CommandSource.AGENT,
        transport=Transport.ROBOT_AGENT,
        commandKey="navigation.navigate_to_pose",
        commandVersion="2.0.0",
        payload={"goalPose": {"frameId": "map", "x": 1, "y": 0, "z": 0, "yaw": 0}},
        issuedAt=now,
        expiresAt=now + timedelta(seconds=30),
        idempotencyKey="routing-sim-1",
        sessionId="mission-sim-1",
        traceId="trace-sim-1",
    )
    mapping = MappingVersion(
        mappingId="route-sim",
        version="v1",
        commandKey=request.commandKey,
        transport=Transport.ROBOT_AGENT,
        mapping={"targetId": "g1-sim"},
    )

    assert router.dispatch(request, mapping).result["driver"] == "simulation"
    assert simulation.dispatched == ["command-sim-1"]
    assert live.dispatched == []
    assert router.query("command-sim-1").result["driver"] == "simulation"


def test_robot_agent_gateway_router_restores_persisted_simulation_command_route() -> None:
    live = _Driver("live")
    simulation = _Driver("simulation")
    router = _RobotAgentGatewayRouter(live, {"g1-sim": simulation})

    assert router.query("command-sim-recovered").status == "unknown"

    router.restore_command_route("command-sim-recovered", "g1-sim")

    assert router.query("command-sim-recovered").result == {
        "driver": "simulation",
        "commandId": "command-sim-recovered",
    }
