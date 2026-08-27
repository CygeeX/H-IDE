from __future__ import annotations

from datetime import timedelta
import threading
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
from services.robot_agent_gateway import driver as gateway_driver


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


class _CountingProtocolSimulator(ProtocolSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cancel_count = 0

    def cancel(self, command_id: str):
        self.cancel_count += 1
        return super().cancel(command_id)


class _BlockingUnknownCancelPeer(_CountingProtocolSimulator):
    def __init__(self) -> None:
        super().__init__(terminal="RUNNING")
        self.cancel_entered = threading.Event()
        self.cancel_release = threading.Event()

    def cancel(self, command_id: str):
        self.cancel_entered.set()
        assert self.cancel_release.wait(timeout=1.0)
        return super().cancel(command_id)


class _UncertainUnknownCancelPeer(_CountingProtocolSimulator):
    def cancel(self, command_id: str):
        self.cancel_count += 1
        raise TimeoutError("lost cancellation response")


class _BlockingSubmitPeer(_CountingProtocolSimulator):
    def __init__(self) -> None:
        super().__init__(terminal="RUNNING")
        self.submit_entered = threading.Event()
        self.submit_release = threading.Event()

    def submit(self, command):
        self.submit_entered.set()
        assert self.submit_release.wait(timeout=1.0)
        return super().submit(command)


def test_cancel_before_dispatch_tombstones_uuid_and_suppresses_late_command() -> None:
    peer = _CountingProtocolSimulator(terminal="RUNNING")
    driver = RobotAgentGatewayDriver(peer)
    adapter = RobotAgentAdapter(driver=driver)
    command = request("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    first_cancel = adapter.cancel(command.commandId)

    assert first_cancel.status == "cancelled"
    assert first_cancel.result == {
        "robotAgentCommandId": command.commandId,
        "dispatchSuppressed": True,
    }
    assert peer.cancel_count == 1
    assert peer.submit_count == 0
    assert peer.control_state == "UNARMED"

    late = adapter.dispatch(command, mapping())
    assert late.status == "cancelled"
    assert late.result == first_cancel.result
    assert peer.submit_count == 0
    assert peer.control_state == "UNARMED"

    # The process-lifetime ledger is deterministic and never asks the peer to
    # manufacture a second cancellation truth.
    assert adapter.cancel(command.commandId) == first_cancel
    assert adapter.query(command.commandId) == first_cancel
    assert peer.cancel_count == 1


def test_cancel_first_linearizes_before_concurrent_late_dispatch() -> None:
    peer = _BlockingUnknownCancelPeer()
    driver = RobotAgentGatewayDriver(peer)
    adapter = RobotAgentAdapter(driver=driver)
    command = request("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
    outcomes: list[tuple[str, str]] = []

    cancelling = threading.Thread(
        target=lambda: outcomes.append(
            ("cancel", adapter.cancel(command.commandId).status)
        )
    )
    dispatching = threading.Thread(
        target=lambda: outcomes.append(
            ("dispatch", adapter.dispatch(command, mapping()).status)
        )
    )
    cancelling.start()
    assert peer.cancel_entered.wait(timeout=0.5)
    dispatching.start()
    dispatching.join(timeout=0.03)
    assert dispatching.is_alive()
    assert peer.submit_count == 0
    assert peer.control_state == "UNARMED"

    peer.cancel_release.set()
    cancelling.join(timeout=1.0)
    dispatching.join(timeout=1.0)

    assert not cancelling.is_alive()
    assert not dispatching.is_alive()
    assert sorted(outcomes) == [("cancel", "cancelled"), ("dispatch", "cancelled")]
    assert peer.submit_count == 0
    assert peer.control_state == "UNARMED"


def test_uncertain_unknown_cancel_keeps_tombstone_and_truthful_result() -> None:
    peer = _UncertainUnknownCancelPeer(terminal="RUNNING")
    adapter = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
    command = request("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")

    uncertain = adapter.cancel(command.commandId)

    assert uncertain.status == "unavailable"
    assert uncertain.details["dispatchSuppressed"] is True
    assert adapter.cancel(command.commandId) == uncertain
    assert peer.cancel_count == 1
    assert adapter.dispatch(command, mapping()).status == "cancelled"
    assert peer.submit_count == 0
    assert peer.control_state == "UNARMED"


def test_dispatch_first_linearizes_before_concurrent_cancel() -> None:
    peer = _BlockingSubmitPeer()
    driver = RobotAgentGatewayDriver(peer)
    adapter = RobotAgentAdapter(driver=driver)
    command = request("99999999-9999-4999-8999-999999999999")
    outcomes: list[tuple[str, str]] = []

    dispatching = threading.Thread(
        target=lambda: outcomes.append(
            ("dispatch", adapter.dispatch(command, mapping()).status)
        )
    )
    cancelling = threading.Thread(
        target=lambda: outcomes.append(
            ("cancel", adapter.cancel(command.commandId).status)
        )
    )
    dispatching.start()
    assert peer.submit_entered.wait(timeout=0.5)
    cancelling.start()
    cancelling.join(timeout=0.03)
    assert cancelling.is_alive()

    peer.submit_release.set()
    dispatching.join(timeout=1.0)
    cancelling.join(timeout=1.0)

    assert not dispatching.is_alive()
    assert not cancelling.is_alive()
    assert sorted(outcomes) == [("cancel", "cancelled"), ("dispatch", "running")]
    assert peer.submit_count == 1
    assert command.commandId not in driver._pre_dispatch_cancels


def test_new_gateway_still_forwards_unknown_cancel_to_peer_before_tombstoning() -> None:
    peer = _CountingProtocolSimulator(terminal="RUNNING")
    old_gateway = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
    command = request("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    assert old_gateway.dispatch(command, mapping()).status == "running"

    # A replacement Gateway has no local command identity, while the Robot
    # Agent still does.  The pre-dispatch tombstone must not replace the real
    # downstream cancellation handshake.
    replacement = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
    cancelled = replacement.cancel(command.commandId)

    assert cancelled.status == "cancelled"
    assert peer.cancel_count == 1
    assert replacement.dispatch(command, mapping()).status == "cancelled"
    assert peer.submit_count == 1


def test_pre_dispatch_cancel_capacity_saturates_fail_closed_without_eviction(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gateway_driver, "_PRE_DISPATCH_CANCEL_CAPACITY", 2)
    peer = _CountingProtocolSimulator(terminal="RUNNING")
    adapter = RobotAgentAdapter(driver=RobotAgentGatewayDriver(peer))
    first = "c1111111-1111-4111-8111-111111111111"
    second = "c2222222-2222-4222-8222-222222222222"
    overflow = "c3333333-3333-4333-8333-333333333333"
    unseen = "c4444444-4444-4444-8444-444444444444"

    assert adapter.cancel(first).status == "cancelled"
    assert adapter.cancel(second).status == "cancelled"
    # The third cancel cannot evict either earlier proof.  It closes admission
    # for every unseen ID for the remainder of this Gateway process.
    assert adapter.cancel(overflow).status == "rejected"
    assert adapter.dispatch(request(first), mapping()).status == "cancelled"
    assert adapter.dispatch(request(second), mapping()).status == "cancelled"
    rejected = adapter.dispatch(request(unseen), mapping())
    assert rejected.status == "rejected"
    assert rejected.details["dispatchSuppressed"] is True
    assert peer.submit_count == 0


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
