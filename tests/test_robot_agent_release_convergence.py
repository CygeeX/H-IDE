from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import pytest

from services.robot_agent_gateway import ws_inbound
from services.robot_agent_gateway.driver import RobotAgentGatewayDriver
from services.robot_agent_gateway.protocol import (
    ProtocolCommand,
    ProtocolCommandStatus,
)
from services.robot_agent_gateway.ws_inbound import RobotAgentInboundPeer

from .test_robot_agent_inbound_gateway import BOOT_ID, CAPABILITIES, CAPABILITY_DIGEST
from .test_robot_agent_protocol_simulator import mapping, request


class _ScriptedPeer:
    backend_name = "scripted"

    def __init__(self) -> None:
        self.statuses: dict[str, ProtocolCommandStatus] = {}
        self.acquire_calls = 0
        self.release_calls = 0
        self.epoch = 0
        self.release_started = threading.Event()
        self.release_continue = threading.Event()
        self.block_release = False
        self.lose_first_release_response = False

    def discover(self, robot_id: str | None = None) -> Mapping[str, Any]:
        return {
            "available": robot_id in {None, "robot-1"},
            "protocol": "robot-agent.v1",
            "capabilities": ["navigation.navigate_to_pose"],
        }

    def acquire(self, robot_id: str) -> Mapping[str, Any]:
        assert robot_id == "robot-1"
        self.acquire_calls += 1
        return {"state": "ACTIVE", "leaseSequence": 0}

    def renew(self, robot_id: str, sequence: int) -> Mapping[str, Any]:
        assert robot_id == "robot-1"
        return {"sequence": sequence, "expiresAt": "2099-01-01T00:00:00Z"}

    def submit(self, command: ProtocolCommand) -> ProtocolCommandStatus:
        status = self.statuses.setdefault(
            command.command_id,
            ProtocolCommandStatus(command.command_id, "RUNNING", "moving"),
        )
        return status

    def query(self, command_id: str) -> ProtocolCommandStatus:
        return self.statuses[command_id]

    def cancel(self, command_id: str) -> ProtocolCommandStatus:
        return self.statuses[command_id]

    def release(self, robot_id: str, reason: str) -> Mapping[str, Any]:
        assert robot_id == "robot-1"
        assert reason
        self.release_calls += 1
        if self.release_calls == 1:
            self.epoch += 1
        self.release_started.set()
        if self.block_release:
            assert self.release_continue.wait(timeout=1.0)
        truth = {
            "state": "SAFE_HOLD",
            "authorityRevoked": True,
            "safeStopState": "confirmed",
            "safeStopEpoch": self.epoch,
        }
        if self.lose_first_release_response and self.release_calls == 1:
            raise TimeoutError("confirmed response was lost")
        return truth

    def terminal(self, command_id: str, state: str = "SUCCEEDED") -> None:
        self.statuses[command_id] = ProtocolCommandStatus(
            command_id,
            state,
            "terminal",
            progress=1.0,
            result={"goalReached": state == "SUCCEEDED"},
        )


def test_two_commands_release_only_after_last_terminal_and_cache_truth() -> None:
    peer = _ScriptedPeer()
    driver = RobotAgentGatewayDriver(peer)
    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"

    assert driver.dispatch(request(first_id), mapping()).status == "running"
    assert driver.dispatch(request(second_id), mapping()).status == "running"
    peer.terminal(first_id)
    assert driver.query(first_id).status == "succeeded"
    assert peer.release_calls == 0

    peer.terminal(second_id)
    last = driver.query(second_id)
    assert last.status == "succeeded"
    assert last.result["controlRelease"]["safeStopEpoch"] == 1
    assert peer.release_calls == 1

    # Both terminal commands belong to the released control session. Their
    # repeat reads reuse the same proof and never ask an idempotent Robot Agent
    # to manufacture another epoch.
    assert driver.query(first_id).result["controlRelease"] == (
        last.result["controlRelease"]
    )
    assert driver.query(second_id).result["controlRelease"] == (
        last.result["controlRelease"]
    )
    assert peer.release_calls == 1
    assert driver.dispatch(request(first_id), mapping()).status == "succeeded"
    assert peer.acquire_calls == 1
    assert peer.release_calls == 1
    conflict = driver.dispatch(
        request(
            first_id,
            goalPose={
                "frameId": "map",
                "x": 9.0,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
            },
        ),
        mapping(),
    )
    assert conflict.status == "rejected"
    assert peer.release_calls == 1


def test_lost_confirmed_release_response_retries_same_attempt_then_caches() -> None:
    peer = _ScriptedPeer()
    peer.lose_first_release_response = True
    driver = RobotAgentGatewayDriver(peer)
    command_id = "33333333-3333-4333-8333-333333333333"
    assert driver.dispatch(request(command_id), mapping()).status == "running"
    peer.terminal(command_id)

    uncertain = driver.query(command_id)
    assert uncertain.status == "unavailable"
    assert peer.epoch == 1
    recovered = driver.query(command_id)
    assert recovered.status == "succeeded"
    assert recovered.result["controlRelease"]["safeStopEpoch"] == 1
    assert peer.release_calls == 2

    assert driver.query(command_id).status == "succeeded"
    assert peer.release_calls == 2


def test_release_serializes_a_new_dispatch_until_safe_hold_is_confirmed() -> None:
    peer = _ScriptedPeer()
    peer.block_release = True
    driver = RobotAgentGatewayDriver(peer)
    first_id = "44444444-4444-4444-8444-444444444444"
    second_id = "55555555-5555-4555-8555-555555555555"
    assert driver.dispatch(request(first_id), mapping()).status == "running"
    peer.terminal(first_id)
    results: list[str] = []

    releasing = threading.Thread(
        target=lambda: results.append(driver.query(first_id).status)
    )
    dispatching = threading.Thread(
        target=lambda: results.append(
            driver.dispatch(request(second_id), mapping()).status
        )
    )
    releasing.start()
    assert peer.release_started.wait(timeout=0.5)
    dispatching.start()
    dispatching.join(timeout=0.03)
    assert dispatching.is_alive()
    assert peer.acquire_calls == 1

    peer.release_continue.set()
    releasing.join(timeout=1.0)
    dispatching.join(timeout=1.0)
    assert not releasing.is_alive()
    assert not dispatching.is_alive()
    assert sorted(results) == ["running", "succeeded"]
    assert peer.acquire_calls == 2


def _release_peer(monkeypatch: pytest.MonkeyPatch) -> RobotAgentInboundPeer:
    monkeypatch.setattr(
        ws_inbound,
        "_CONTROL_RELEASE_CONFIRMATION_TIMEOUT_SECONDS",
        0.02,
    )
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=CAPABILITY_DIGEST,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
    )
    with peer._condition:
        peer._connected = True
        peer.robot_boot_id = BOOT_ID
        peer._control_state = "ACTIVE"
        peer._control_revision = 3
        peer._control_safe_stop_truth_supported = True
        peer._control_authority_revoked = False
        peer._control_safe_stop_state = "confirmed"
        peer._control_safe_stop_epoch = 7
    monkeypatch.setattr(peer, "_send", lambda *_args, **_kwargs: None)
    return peer


def test_inbound_release_retry_keeps_the_first_epoch_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _release_peer(monkeypatch)
    with pytest.raises(TimeoutError):
        peer.release("g1-dev", "terminal command complete")

    # The original call timed out, but the same Robot Agent then publishes
    # the confirmation. A retry must accept epoch 8 relative to the original
    # epoch 7, rather than treating 8 as a new baseline and waiting forever.
    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "release confirmed",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 8,
            },
        }
    )
    assert peer.release("g1-dev", "terminal command complete") == {
        "state": "SAFE_HOLD",
        "authorityRevoked": True,
        "safeStopState": "confirmed",
        "safeStopEpoch": 8,
    }


def test_inbound_release_does_not_accept_a_new_boot_epoch_as_its_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    peer = _release_peer(monkeypatch)
    with pytest.raises(TimeoutError):
        peer.release("g1-dev", "terminal command complete")
    with peer._condition:
        peer.robot_boot_id = "22222222-2222-4222-8222-222222222222"
        peer._control_state = "SAFE_HOLD"
        peer._control_revision += 1
        peer._control_authority_revoked = True
        peer._control_safe_stop_state = "confirmed"
        peer._control_safe_stop_epoch = 8

    with pytest.raises(RuntimeError, match="restarted"):
        peer.release("g1-dev", "terminal command complete")
