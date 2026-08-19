from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from packages.control_core.adapters import RobotAgentAdapter
from packages.control_core.models import (
    CommandRequest,
    CommandSource,
    MappingVersion,
    Transport,
    utc_now,
)
from services.robot_agent_gateway import ProtocolSimulator, RobotAgentGatewayDriver


def mapping():
    return MappingVersion(
        mappingId="robot-agent-navigation",
        version="v1",
        commandKey="navigation.navigate_to_pose",
        transport="robot_agent",
        mapping={"targetId": "robot-1", "endpoint": "protocol-simulator"},
    )


def request(command_id="11111111-1111-4111-8111-111111111111", **payload_overrides):
    now = utc_now()
    payload = {
        "payloadType": "visual_resolution",
        "goalPose": {"frameId": "map", "x": 1.25, "y": 0, "z": 0, "yaw": 0},
        "goalPurpose": "approach",
        "missionId": "mission-1",
        "stageId": "stage-1",
        "visualResolutionId": "vr-1",
        "observationSessionId": "obs-1",
        "robotBootId": "11111111-1111-4111-8111-111111111111",
        "localizationEpochId": "epoch-1",
        "maximumSpeedMps": 0.2,
    }
    payload.update(payload_overrides)
    return CommandRequest(
        commandId=command_id,
        robotId="robot-1",
        source=CommandSource.AGENT,
        transport=Transport.ROBOT_AGENT,
        commandKey="navigation.navigate_to_pose",
        commandVersion="1",
        payload=payload,
        issuedAt=now,
        expiresAt=now + timedelta(seconds=10),
        idempotencyKey=f"idem-{command_id}",
        sessionId="mission-1",
        traceId="trace-1",
    )


def test_protocol_driver_exercises_lease_command_query_and_idempotency() -> None:
    peer = ProtocolSimulator()
    adapter = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
    assert adapter.discover("robot-1").result["motion"] == "simulated"
    assert adapter.probe(mapping()).status == "probed"
    first = adapter.dispatch(request(), mapping())
    second = adapter.dispatch(request(), mapping())
    assert first.status == second.status == "succeeded"
    assert first.result["executionBackend"] == "protocol_simulator"
    assert first.result["rosGoalId"] is None
    assert peer.submit_count == 1
    assert adapter.query(request().commandId).status == "succeeded"


def test_lease_loss_bad_goal_and_disconnect_fail_closed() -> None:
    peer = ProtocolSimulator(terminal="RUNNING")
    driver = RobotAgentGatewayDriver(peer)
    adapter = RobotAgentAdapter(driver=driver)
    bad = adapter.dispatch(request(goalPose={"frameId": "odom", "x": 1, "y": 0, "z": 0, "yaw": 0}), mapping())
    assert bad.status == "rejected"
    running = adapter.dispatch(request("22222222-2222-4222-8222-222222222222"), mapping())
    assert running.status == "running"
    peer.disconnect()
    assert adapter.query("22222222-2222-4222-8222-222222222222").status == "unknown"


def test_protocol_rejects_stale_boot_epoch_observation_and_non_uuid_id() -> None:
    peer = ProtocolSimulator()
    adapter = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
    assert adapter.dispatch(request(robotBootId="old-boot"), mapping()).status == "rejected"
    assert adapter.dispatch(
        request("22222222-2222-4222-8222-222222222221", localizationEpochId="old-epoch"),
        mapping(),
    ).status == "rejected"
    assert adapter.dispatch(
        request("22222222-2222-4222-8222-222222222223", observationSessionId="old-session"),
        mapping(),
    ).status == "rejected"
    assert adapter.dispatch(request("not-a-uuid"), mapping()).status == "rejected"
    assert peer.submit_count == 0


def test_cancel_acknowledges_only_active_protocol_command() -> None:
    peer = ProtocolSimulator(terminal="RUNNING")
    adapter = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
    command = request("33333333-3333-4333-8333-333333333333")
    assert adapter.dispatch(command, mapping()).status == "running"
    assert adapter.cancel(command.commandId).status == "cancelled"
    assert adapter.query(command.commandId).status == "cancelled"


def test_driver_stably_maps_human_mission_skill_ids_to_uuid_wire_ids() -> None:
    peer = ProtocolSimulator(terminal="RUNNING")
    driver = RobotAgentGatewayDriver(peer)
    adapter = RobotAgentAdapter(driver=driver)
    logical_id = "skill-mission-1-1-navigate-target-1"

    dispatched = adapter.dispatch(request(logical_id), mapping())

    assert dispatched.status == "running"
    wire_ids = list(peer.commands)
    assert len(wire_ids) == 1
    UUID(wire_ids[0])
    assert wire_ids[0] == driver._wire_command_id(logical_id)
    assert driver._wire_command_id(logical_id) == driver._wire_command_id(logical_id)
    assert adapter.query(logical_id).status == "running"
    assert adapter.cancel(logical_id).status == "cancelled"
