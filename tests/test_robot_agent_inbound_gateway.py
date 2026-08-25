from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import CancelledError as FutureCancelledError
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from packages.control_core.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityManifest,
    RiskLevel,
    RobotProfile,
    SensorKind,
    SensorProfile,
)
from services.robot_agent_gateway import ws_inbound
from services.robot_agent_gateway.main import create_app
from services.robot_agent_gateway.protocol import ProtocolCommand, ProtocolCommandStatus
from services.robot_agent_gateway.wire import (
    SUBPROTOCOL,
    WireProtocolError,
    build_message,
    parse_control_status_payload,
    parse_message,
)
from services.robot_agent_gateway.ws_inbound import RobotAgentInboundPeer

CAPABILITIES = ("navigation.navigate_to_pose", "command.cancel", "command.query")
CAPABILITY_DIGEST = "sha256:" + hashlib.sha256(
    json.dumps(list(CAPABILITIES), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
BOOT_ID = "11111111-1111-4111-8111-111111111111"
OBSERVATION_SESSION_ID = "observation-session-test"
LOCALIZATION_EPOCH_ID = "localization-epoch-test"


def command(command_id: str) -> ProtocolCommand:
    issued = datetime.now(UTC)
    return ProtocolCommand(
        command_id=command_id,
        robot_id="g1-dev",
        kind="navigation.navigate_to_pose",
        issued_at=issued,
        ttl_ms=10_000,
        params={
            "goalPose": {"frameId": "map", "x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.0},
            "observationSessionId": OBSERVATION_SESSION_ID,
            "localizationEpochId": LOCALIZATION_EPOCH_ID,
        },
        payload_hash="a" * 64,
    )


def run_fake_agent(
    websocket,
    *,
    terminal_delay: float = 0.05,
    observation_status: str = "USABLE",
    received_types: list[str] | None = None,
    hello_payload_overrides: dict[str, object] | None = None,
    transient_control_arm_rejections: int = 0,
) -> None:
    hello_payload = {
        "agent_version": "test-agent",
        "boot_id": BOOT_ID,
        "supported_versions": ["1.0"],
        "transport_mode": "client",
        "capabilities_digest": CAPABILITY_DIGEST,
    }
    hello_payload.update(hello_payload_overrides or {})
    websocket.send_text(build_message(
        "session.hello",
        robot_id="g1-dev",
        session_id=None,
        payload=hello_payload,
    ))
    welcome = parse_message(websocket.receive_text(), expected_robot_id="g1-dev")
    session_id = welcome["session_id"]
    assert welcome["type"] == "session.welcome"
    websocket.send_text(build_message(
        "control.status",
        robot_id="g1-dev",
        session_id=session_id,
        payload={
            "state": "SAFE_HOLD",
            "reason": "new session requires control",
            "lease_sequence": None,
        },
    ))
    websocket.send_text(build_message(
        "state.snapshot",
        robot_id="g1-dev",
        session_id=session_id,
        payload={
            "sequence": 0,
            "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "state": {
                "observation": {
                    "status": observation_status,
                    "sensor_id": "camera-test",
                    "session_id": OBSERVATION_SESSION_ID,
                    "scene_version": "scene-test",
                    "localization_epoch_id": LOCALIZATION_EPOCH_ID,
                    "ack_age_ms": 5.0,
                    "capture_age_ms": 5.0,
                    "drop_count": 0,
                },
                "localization": {
                    "frame_id": "map",
                    # Deliberately use a ROS/simulation-clock stamp.  Gateway
                    # freshness must come from the UTC state-envelope time,
                    # not from this unrelated clock domain.
                    "stamp": {"sec": 0, "nanosec": 0},
                    "position": {"x": 0.4, "y": -0.2, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
        },
    ))
    control_acquire_attempt = 0
    while True:
        incoming = parse_message(
            websocket.receive_text(),
            expected_robot_id="g1-dev",
            expected_session_id=session_id,
        )
        if received_types is not None:
            received_types.append(incoming["type"])
        payload = incoming["payload"]
        if incoming["type"] == "control.acquire":
            control_acquire_attempt += 1
            if control_acquire_attempt <= transient_control_arm_rejections:
                websocket.send_text(build_message(
                    "control.status",
                    robot_id="g1-dev",
                    session_id=session_id,
                    payload={
                        "state": "SAFE_HOLD",
                        "reason": (
                            "control arm failed closed: control arm rejected "
                            "while the measured base is moving"
                        ),
                        "lease_sequence": None,
                    },
                ))
                continue
            websocket.send_text(build_message(
                "control.status",
                robot_id="g1-dev",
                session_id=session_id,
                payload={"state": "ACTIVE", "reason": "control acquired", "lease_sequence": None},
            ))
        elif incoming["type"] == "lease.renew":
            websocket.send_text(build_message(
                "lease.status",
                robot_id="g1-dev",
                session_id=session_id,
                payload={
                    "sequence": payload["sequence"],
                    "expires_at": (datetime.now(UTC) + timedelta(seconds=2))
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
            ))
        elif incoming["type"] == "command.request":
            command_id = payload["command_id"]
            websocket.send_text(build_message(
                "command.status",
                robot_id="g1-dev",
                session_id=session_id,
                payload={
                    "command_id": command_id,
                    "state": "RUNNING",
                    "detail": "mock Nav2 is moving",
                    "progress": 0.25,
                    "result": None,
                },
            ))
            time.sleep(terminal_delay)
            websocket.send_text(build_message(
                "command.status",
                robot_id="g1-dev",
                session_id=session_id,
                payload={
                    "command_id": command_id,
                    "state": "SUCCEEDED",
                    "detail": "mock Nav2 reached the goal",
                    "progress": 1.0,
                    "result": {"goalReached": True},
                },
            ))
            websocket.send_text(build_message(
                "state.update",
                robot_id="g1-dev",
                session_id=session_id,
                payload={
                    "sequence": 1,
                    "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "changes": {"localization.position.x": 0.9},
                },
            ))
            time.sleep(0.03)
            websocket.send_text(build_message(
                "state.update",
                robot_id="g1-dev",
                session_id=session_id,
                payload={
                    "sequence": 2,
                    "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "changes": {"localization.position.x": 0.9001},
                },
            ))
            time.sleep(0.03)
            websocket.send_text(build_message(
                "state.update",
                robot_id="g1-dev",
                session_id=session_id,
                payload={
                    "sequence": 3,
                    "observed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "changes": {"localization.position.x": 0.9002},
                },
            ))
        elif incoming["type"] == "session.goodbye":
            return


def _cancel_test_peer(
    monkeypatch,
    *,
    timeout_seconds: float,
    safety_truth: bool = True,
) -> tuple[RobotAgentInboundPeer, str, threading.Event]:
    monkeypatch.setattr(
        ws_inbound,
        "_CANCEL_CONFIRMATION_TIMEOUT_SECONDS",
        timeout_seconds,
    )
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=CAPABILITY_DIGEST,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
    )
    command_id = str(uuid4())
    sent = threading.Event()
    with peer._condition:
        peer._connected = True
        peer._control_state = "ACTIVE"
        peer._control_revision = 3
        peer._control_safe_stop_truth_supported = safety_truth
        peer._control_authority_revoked = False if safety_truth else None
        peer._control_safe_stop_state = "confirmed" if safety_truth else None
        peer._control_safe_stop_epoch = 7 if safety_truth else None
        peer._statuses[command_id] = ProtocolCommandStatus(
            command_id,
            "RUNNING",
            "mock Nav2 is moving",
        )
        peer._status_revisions[command_id] = 1

    def capture_send(message_type: str, payload: dict[str, object]) -> None:
        assert message_type == "command.cancel"
        assert payload["command_id"] == command_id
        sent.set()

    monkeypatch.setattr(peer, "_send", capture_send)
    return peer, command_id, sent


def _record_command_status(
    peer: RobotAgentInboundPeer,
    command_id: str,
    state: str,
) -> None:
    peer._record_message(
        {
            "type": "command.status",
            "payload": {
                "command_id": command_id,
                "state": state,
                "detail": f"test {state.lower()}",
                "progress": None,
                "result": None,
            },
        }
    )


def test_cancel_ignores_running_revisions_and_waits_for_new_safe_hold(
    monkeypatch,
) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)

    _record_command_status(peer, command_id, "RUNNING")
    _record_command_status(peer, command_id, "RUNNING")
    worker.join(timeout=0.02)
    assert worker.is_alive()

    _record_command_status(peer, command_id, "CANCELLED")
    worker.join(timeout=0.02)
    assert worker.is_alive()

    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "safe stop requested",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "requested",
                "safe_stop_epoch": 8,
            },
        }
    )
    worker.join(timeout=0.02)
    assert worker.is_alive()

    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "safe stop completed",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 8,
            },
        }
    )
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results[0].state == "CANCELLED"


def test_cancelled_status_before_safe_hold_still_fences_and_waits(
    monkeypatch,
) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    _record_command_status(peer, command_id, "CANCELLED")

    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)
    assert peer._heartbeat_stop.is_set()
    worker.join(timeout=0.02)
    assert worker.is_alive()

    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "safe stop completed after terminal status",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 8,
            },
        }
    )
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results[0].state == "CANCELLED"


def test_cancel_rejects_fresh_safe_hold_received_after_deadline(monkeypatch) -> None:
    peer, command_id, _sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.05,
    )

    def confirm_after_deadline(
        message_type: str,
        payload: dict[str, object],
    ) -> None:
        assert message_type == "command.cancel"
        assert payload["command_id"] == command_id
        time.sleep(0.07)
        _record_command_status(peer, command_id, "CANCELLED")
        peer._record_message(
            {
                "type": "control.status",
                "payload": {
                    "state": "SAFE_HOLD",
                    "reason": "late safe stop completed",
                    "lease_sequence": None,
                    "authority_revoked": True,
                    "safe_stop_state": "confirmed",
                    "safe_stop_epoch": 8,
                },
            }
        )

    monkeypatch.setattr(peer, "_send", confirm_after_deadline)
    result = peer.cancel(command_id)

    assert result.state == "ABORTED_AGENT_RESTART"


@pytest.mark.parametrize(
    "send_error",
    [
        pytest.param(WebSocketDisconnect(code=1006), id="websocket-disconnect"),
        pytest.param(FutureCancelledError(), id="cancelled-send-future"),
    ],
)
def test_cancel_transport_failure_joins_blocked_heartbeat(
    monkeypatch,
    send_error: Exception,
) -> None:
    peer, command_id, _sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    peer.heartbeat = True
    renew_started = threading.Event()
    release_renew = threading.Event()

    def blocked_renew(robot_id: str, sequence: int) -> dict[str, object]:
        del robot_id, sequence
        renew_started.set()
        if not release_renew.wait(timeout=1.0):
            raise TimeoutError("test did not release the blocked renewal")
        raise RuntimeError("Robot Agent control lease is not active")

    def disconnect_send(message_type: str, payload: dict[str, object]) -> None:
        assert message_type == "command.cancel"
        assert payload["command_id"] == command_id
        raise send_error

    monkeypatch.setattr(peer, "renew", blocked_renew)
    monkeypatch.setattr(peer, "_send", disconnect_send)
    peer._start_heartbeat()
    assert renew_started.wait(timeout=1.0)

    results: list[ProtocolCommandStatus] = []
    errors: list[BaseException] = []

    def cancel_worker() -> None:
        try:
            results.append(peer.cancel(command_id))
        except BaseException as exc:  # pragma: no cover - asserted empty below
            errors.append(exc)

    worker = threading.Thread(target=cancel_worker)
    worker.start()
    try:
        assert peer._heartbeat_stop.wait(timeout=0.5)
        worker.join(timeout=0.02)
        assert worker.is_alive()
        release_renew.set()
        worker.join(timeout=0.5)
        assert not worker.is_alive()
        assert errors == []
        assert results[0].state == "ABORTED_AGENT_RESTART"
        assert "command.cancel was not confirmed" in results[0].detail
        with peer._condition:
            assert peer._heartbeat_thread is None
        assert peer.connection_snapshot()["lastError"] is None
    finally:
        release_renew.set()
        peer._stop_heartbeat(timeout=0.5, strict=False)
        worker.join(timeout=0.6)


def test_cancel_cannot_join_old_status_to_new_session_safe_hold(monkeypatch) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    with peer._condition:
        peer.session_id = str(uuid4())
        peer.robot_boot_id = BOOT_ID
    _record_command_status(peer, command_id, "CANCELLED")

    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)
    worker.join(timeout=0.02)
    assert worker.is_alive()

    with peer._condition:
        peer._mark_disconnected_locked("synthetic session replacement")
        peer._connected = True
        peer.session_id = str(uuid4())
        peer.robot_boot_id = "22222222-2222-4222-8222-222222222222"
        peer._condition.notify_all()
    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "new session safe stop completed",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 8,
            },
        }
    )

    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results[0].state == "ABORTED_AGENT_RESTART"
    assert "session identity changed" in results[0].detail


def test_cancel_stops_heartbeat_without_recording_expected_renew_failure(
    monkeypatch,
) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    peer.heartbeat = True
    renew_started = threading.Event()
    release_renew = threading.Event()

    def renew_while_cancel_stops_heartbeat(
        robot_id: str,
        sequence: int,
    ) -> dict[str, object]:
        del robot_id, sequence
        renew_started.set()
        if not release_renew.wait(timeout=1.0):
            raise TimeoutError("test did not release the in-flight renewal")
        raise RuntimeError("Robot Agent control lease is not active")

    monkeypatch.setattr(peer, "renew", renew_while_cancel_stops_heartbeat)
    peer._start_heartbeat()
    assert renew_started.wait(timeout=1.0)

    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    try:
        assert peer._heartbeat_stop.wait(timeout=0.5)
        assert sent.wait(timeout=0.5)
        with peer._condition:
            heartbeat_thread = peer._heartbeat_thread
        assert heartbeat_thread is not None
        assert heartbeat_thread.is_alive()
        assert worker.is_alive()
        release_renew.set()
        _record_command_status(peer, command_id, "CANCELLED")
        peer._record_message(
            {
                "type": "control.status",
                "payload": {
                    "state": "SAFE_HOLD",
                    "reason": "safe stop completed",
                    "lease_sequence": None,
                    "authority_revoked": True,
                    "safe_stop_state": "confirmed",
                    "safe_stop_epoch": 8,
                },
            }
        )
        worker.join(timeout=0.5)
        assert not worker.is_alive()
        assert results[0].state == "CANCELLED"
        with peer._condition:
            assert peer._heartbeat_thread is None
        assert peer.connection_snapshot()["lastError"] is None
    finally:
        release_renew.set()
        peer._stop_heartbeat(timeout=0.5, strict=False)
        worker.join(timeout=0.6)


def test_cancel_unknown_command_does_not_stop_heartbeat(monkeypatch) -> None:
    peer, _command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )

    result = peer.cancel(str(uuid4()))

    assert result.state == "REJECTED"
    assert not peer._heartbeat_stop.is_set()
    assert not sent.is_set()


def test_cancel_send_failure_joins_heartbeat_and_fails_closed(monkeypatch) -> None:
    peer, command_id, _sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    peer.heartbeat = True
    peer._start_heartbeat()

    def fail_send(message_type: str, payload: dict[str, object]) -> None:
        del message_type, payload
        raise RuntimeError("synthetic cancel send failure")

    monkeypatch.setattr(peer, "_send", fail_send)
    result = peer.cancel(command_id)

    assert result.state == "ABORTED_AGENT_RESTART"
    assert "command.cancel was not confirmed" in result.detail
    assert peer._heartbeat_stop.is_set()
    with peer._condition:
        assert peer._heartbeat_thread is None


def test_cancel_join_failure_cannot_return_cancelled(monkeypatch) -> None:
    peer, command_id, _sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )

    def confirm_cancel(message_type: str, payload: dict[str, object]) -> None:
        assert message_type == "command.cancel"
        assert payload["command_id"] == command_id
        _record_command_status(peer, command_id, "CANCELLED")
        peer._record_message(
            {
                "type": "control.status",
                "payload": {
                    "state": "SAFE_HOLD",
                    "reason": "safe stop completed",
                    "lease_sequence": None,
                    "authority_revoked": True,
                    "safe_stop_state": "confirmed",
                    "safe_stop_epoch": 8,
                },
            }
        )

    def fail_join(*, timeout: float = 2.5, strict: bool = True) -> None:
        del timeout, strict
        raise RuntimeError("synthetic heartbeat join timeout")

    monkeypatch.setattr(peer, "_send", confirm_cancel)
    monkeypatch.setattr(peer, "_stop_heartbeat", fail_join)
    result = peer.cancel(command_id)

    assert result.state == "ABORTED_AGENT_RESTART"
    assert "lease heartbeat did not stop" in result.detail


def test_unexpected_heartbeat_renew_failure_records_last_error(monkeypatch) -> None:
    peer, _command_id, _sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    peer.heartbeat = True

    def fail_renew(robot_id: str, sequence: int) -> dict[str, object]:
        del robot_id, sequence
        raise RuntimeError("synthetic unexpected renewal failure")

    monkeypatch.setattr(peer, "renew", fail_renew)
    peer._start_heartbeat()
    deadline = time.monotonic() + 1.0
    while peer.connection_snapshot()["lastError"] is None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert peer.connection_snapshot()["lastError"] == (
        "Robot Agent lease renewal failed: synthetic unexpected renewal failure"
    )
    with peer._condition:
        thread = peer._heartbeat_thread
    if thread is not None:
        thread.join(timeout=0.5)
    with peer._condition:
        assert peer._heartbeat_thread is None


def test_cancel_without_new_safe_hold_times_out_as_unknown(monkeypatch) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.05,
    )
    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)
    _record_command_status(peer, command_id, "CANCELLED")

    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results[0].state == "ABORTED_AGENT_RESTART"
    assert "without a new confirmed SAFE_HOLD" in results[0].detail


def test_cancel_legacy_safe_hold_without_truth_times_out_as_unknown(monkeypatch) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.05,
        safety_truth=False,
    )
    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)
    _record_command_status(peer, command_id, "CANCELLED")
    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "legacy safe hold",
                "lease_sequence": None,
            },
        }
    )

    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results[0].state == "ABORTED_AGENT_RESTART"
    assert "safety truth" in results[0].detail


def test_cancel_failed_safe_stop_returns_unknown_immediately(monkeypatch) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.5,
    )
    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)
    _record_command_status(peer, command_id, "CANCELLED")
    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "safe stop failed",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "failed",
                "safe_stop_epoch": 8,
            },
        }
    )

    worker.join(timeout=0.2)
    assert not worker.is_alive()
    assert results[0].state == "ABORTED_AGENT_RESTART"
    assert "safe-stop confirmation failed" in results[0].detail


def test_cancel_accepts_fresh_logical_release_with_unchanged_stop_epoch(
    monkeypatch,
) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.05,
    )
    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)
    _record_command_status(peer, command_id, "CANCELLED")
    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "remote authority released; SONIC remains resident",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 7,
            },
        }
    )

    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results[0].state == "CANCELLED"


def test_cancel_rejects_regressed_safe_stop_epoch(monkeypatch) -> None:
    peer, command_id, sent = _cancel_test_peer(
        monkeypatch,
        timeout_seconds=0.05,
    )
    results: list[ProtocolCommandStatus] = []
    worker = threading.Thread(target=lambda: results.append(peer.cancel(command_id)))
    worker.start()
    assert sent.wait(timeout=0.5)
    _record_command_status(peer, command_id, "CANCELLED")
    with pytest.raises(ValueError, match="safe_stop_epoch decreased"):
        peer._record_message(
            {
                "type": "control.status",
                "payload": {
                    "state": "SAFE_HOLD",
                    "reason": "regressed safe stop epoch",
                    "lease_sequence": None,
                    "authority_revoked": True,
                    "safe_stop_state": "confirmed",
                    "safe_stop_epoch": 6,
                },
            }
        )

    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert results[0].state == "ABORTED_AGENT_RESTART"


@pytest.mark.parametrize(
    "overrides",
    [
        {"authority_revoked": True},
        {
            "authority_revoked": "true",
            "safe_stop_state": "confirmed",
            "safe_stop_epoch": 1,
        },
        {
            "authority_revoked": True,
            "safe_stop_state": "complete",
            "safe_stop_epoch": 1,
        },
        {
            "authority_revoked": True,
            "safe_stop_state": "confirmed",
            "safe_stop_epoch": -1,
        },
    ],
)
def test_control_status_rejects_partial_or_invalid_safety_truth(
    overrides: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "state": "SAFE_HOLD",
        "reason": "test safety truth",
        "lease_sequence": None,
        **overrides,
    }
    with pytest.raises(WireProtocolError):
        parse_control_status_payload(payload)


def test_control_status_cannot_downgrade_after_safety_truth() -> None:
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=CAPABILITY_DIGEST,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
    )
    peer._record_message(
        {
            "type": "control.status",
            "payload": {
                "state": "SAFE_HOLD",
                "reason": "safe stop confirmed",
                "lease_sequence": None,
                "authority_revoked": True,
                "safe_stop_state": "confirmed",
                "safe_stop_epoch": 4,
            },
        }
    )
    with pytest.raises(ValueError, match="omitted safety truth"):
        peer._record_message(
            {
                "type": "control.status",
                "payload": {
                    "state": "SAFE_HOLD",
                    "reason": "unsafe protocol downgrade",
                    "lease_sequence": None,
                },
            }
        )


def test_inbound_handshake_and_reconnect_accept_additive_control_truth() -> None:
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=CAPABILITY_DIGEST,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
    )

    def handshake(websocket, *, epoch: int) -> str:
        websocket.send_text(
            build_message(
                "session.hello",
                robot_id="g1-dev",
                session_id=None,
                payload={
                    "agent_version": "test-agent",
                    "boot_id": BOOT_ID,
                    "supported_versions": ["1.0"],
                    "transport_mode": "client",
                    "capabilities_digest": CAPABILITY_DIGEST,
                },
            )
        )
        welcome = parse_message(
            websocket.receive_text(),
            expected_robot_id="g1-dev",
        )
        assert welcome["type"] == "session.welcome"
        session_id = welcome["session_id"]
        assert isinstance(session_id, str)
        websocket.send_text(
            build_message(
                "control.status",
                robot_id="g1-dev",
                session_id=session_id,
                payload={
                    "state": "SAFE_HOLD",
                    "reason": "safe stop confirmed",
                    "lease_sequence": None,
                    "authority_revoked": True,
                    "safe_stop_state": "confirmed",
                    "safe_stop_epoch": epoch,
                },
            )
        )
        deadline = time.monotonic() + 1.0
        snapshot = peer.connection_snapshot()
        while snapshot["safeStopEpoch"] != epoch and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = peer.connection_snapshot()
        assert snapshot["connected"] is True
        assert snapshot["authorityRevoked"] is True
        assert snapshot["safeStopState"] == "confirmed"
        assert snapshot["safeStopEpoch"] == epoch
        assert snapshot["lastError"] is None
        return session_id

    with TestClient(create_app(peer, internal_token="test-token")) as client:
        with client.websocket_connect(
            "/agent",
            subprotocols=[SUBPROTOCOL],
        ) as websocket:
            first_session = handshake(websocket, epoch=11)

        deadline = time.monotonic() + 1.0
        while peer.connection_snapshot()["connected"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert peer.connection_snapshot()["connected"] is False

        with client.websocket_connect(
            "/agent",
            subprotocols=[SUBPROTOCOL],
        ) as websocket:
            second_session = handshake(websocket, epoch=12)
            assert second_session != first_session


def test_inbound_gateway_accepts_agent_client_and_tracks_terminal_status() -> None:
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=CAPABILITY_DIGEST,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
        terminal_pose_stable_seconds=0.02,
        terminal_pose_max_wait_seconds=0.2,
    )
    with (
        TestClient(create_app(peer, internal_token="test-token")) as client,
        client.websocket_connect("/agent", subprotocols=[SUBPROTOCOL]) as websocket,
    ):
        agent = threading.Thread(target=run_fake_agent, args=(websocket,), daemon=True)
        agent.start()
        deadline = time.monotonic() + 1.0
        while not peer.discover("g1-dev").get("available") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert peer.discover("g1-dev")["robotBootId"] == BOOT_ID
        deadline = time.monotonic() + 1.0
        while (
            peer.discover("g1-dev")["observation"]["status"] != "USABLE"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert peer.discover("g1-dev")["observation"]["sessionId"] == (
            OBSERVATION_SESSION_ID
        )
        current_pose = peer.discover("g1-dev")["currentPose"]
        assert current_pose["frameId"] == "map"
        assert current_pose["x"] == 0.4
        assert current_pose["y"] == -0.2
        assert current_pose["yaw"] == 0.0
        assert peer.acquire("g1-dev")["state"] == "ACTIVE"
        assert peer.renew("g1-dev", 1)["sequence"] == 0
        command_id = str(uuid4())
        first = peer.submit(command(command_id))
        assert first.state == "RECEIVED"
        deadline = time.monotonic() + 1.0
        status = peer.query(command_id)
        while (
            status.state != "SUCCEEDED"
            or status.result is None
            or status.result.get("finalPoseRelation") != "stable_after_terminal"
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
            status = peer.query(command_id)
        assert status.state == "SUCCEEDED"
        assert status.result is not None
        assert status.result["goalReached"] is True
        assert status.result["finalPose"]["frameId"] == "map"
        assert status.result["finalPose"]["x"] == 0.9002
        assert status.result["finalPose"]["y"] == -0.2
        assert status.result["finalPose"]["observedAt"]
        assert status.result["finalPoseRelation"] == "stable_after_terminal"
        # Repeated dispatch is a ledger lookup, never a second robot request.
        assert peer.submit(command(command_id)) == status
        peer.close()
        agent.join(timeout=1)


def test_inbound_gateway_retries_only_transient_measured_motion_control_arm() -> None:
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=CAPABILITY_DIGEST,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
    )
    received_types: list[str] = []
    with (
        TestClient(create_app(peer, internal_token="test-token")) as client,
        client.websocket_connect("/agent", subprotocols=[SUBPROTOCOL]) as websocket,
    ):
        agent = threading.Thread(
            target=run_fake_agent,
            args=(websocket,),
            kwargs={
                "received_types": received_types,
                "transient_control_arm_rejections": 2,
            },
            daemon=True,
        )
        agent.start()
        deadline = time.monotonic() + 1.0
        while not peer.discover("g1-dev").get("available") and time.monotonic() < deadline:
            time.sleep(0.01)

        assert peer.acquire("g1-dev")["state"] == "ACTIVE"
        assert received_types.count("control.acquire") == 3

        peer.close()
        agent.join(timeout=1)


def test_inbound_gateway_validates_v2_manifest_and_derives_capabilities() -> None:
    profile = RobotProfile.build(
        robotId="g1-dev",
        baseType="differential_drive",
        headMotion="fixed",
        sensors=[
            SensorProfile(
                sensorId="head_rgbd",
                kind=SensorKind.RGBD,
                frameId="camera_color_optical_frame",
                fixedMount=True,
                horizontalFovRad=1.211,
                depthCapable=True,
            )
        ],
        metadata={"motionOwner": "robot_agent"},
    )
    manifest = CapabilityManifest.build(
        robotId="g1-dev",
        agentVersion="0.2.0",
        robotProfileDigest=profile.profileDigest,
        capabilities=[
            CapabilityDescriptor(
                skillId="navigation.rotate",
                skillVersion="2.0.0",
                description="Rotate through Nav2 Spin",
                kind=CapabilityKind.MOTION,
                riskLevel=RiskLevel.MEDIUM,
                inputSchema={
                    "type": "object",
                    "properties": {"targetYawRad": {"type": "number"}},
                    "required": ["targetYawRad"],
                    "additionalProperties": False,
                },
                resources=["motion", "navigation"],
                cancellable=True,
                interruptible=True,
                timeoutMs=60_000,
                executor="nav2",
                executorProfile="nav2.behaviors.spin",
            )
        ],
    )
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=None,
        capabilities=(),
        allow_plaintext_loopback=True,
        heartbeat=False,
    )
    with (
        TestClient(create_app(peer, internal_token="test-token")) as client,
        client.websocket_connect("/agent", subprotocols=[SUBPROTOCOL]) as websocket,
    ):
        agent = threading.Thread(
            target=run_fake_agent,
            args=(websocket,),
            kwargs={
                "hello_payload_overrides": {
                    "agent_version": "0.2.0",
                    "capabilities_digest": manifest.manifestDigest,
                    "robot_profile": profile.model_dump(mode="json"),
                    "capability_manifest": manifest.model_dump(mode="json"),
                }
            },
            daemon=True,
        )
        agent.start()
        deadline = time.monotonic() + 1.0
        snapshot = peer.discover("g1-dev")
        while not snapshot.get("available") and time.monotonic() < deadline:
            time.sleep(0.01)
            snapshot = peer.discover("g1-dev")
        assert snapshot["capabilities"] == ["navigation.rotate"]
        assert snapshot["robotProfile"]["headMotion"] == "fixed"
        assert snapshot["capabilityManifest"]["manifestDigest"] == (
            manifest.manifestDigest
        )
        peer.close()
        agent.join(timeout=1)


def test_inbound_gateway_rejects_degraded_observation_without_dispatch() -> None:
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest=CAPABILITY_DIGEST,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
        observation_preflight_timeout_seconds=0.1,
    )
    received_types: list[str] = []
    with (
        TestClient(create_app(peer, internal_token="test-token")) as client,
        client.websocket_connect("/agent", subprotocols=[SUBPROTOCOL]) as websocket,
    ):
        agent = threading.Thread(
            target=run_fake_agent,
            args=(websocket,),
            kwargs={
                "observation_status": "DEGRADED",
                "received_types": received_types,
            },
            daemon=True,
        )
        agent.start()
        deadline = time.monotonic() + 1.0
        while (
            peer.connection_snapshot()["observation"]["status"] != "DEGRADED"
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert peer.acquire("g1-dev")["state"] == "ACTIVE"

        status = peer.submit(command(str(uuid4())))

        assert status.state == "REJECTED"
        assert "status is not USABLE" in status.detail
        assert "command.request" not in received_types
        peer.close()
        agent.join(timeout=1)


def test_inbound_gateway_rejects_unexpected_capability_digest() -> None:
    peer = RobotAgentInboundPeer(
        robot_id="g1-dev",
        expected_capabilities_digest="sha256:" + "0" * 64,
        capabilities=CAPABILITIES,
        allow_plaintext_loopback=True,
        heartbeat=False,
    )
    with (
        TestClient(create_app(peer, internal_token="test-token")) as client,
        client.websocket_connect("/agent", subprotocols=[SUBPROTOCOL]) as websocket,
    ):
        websocket.send_text(build_message(
                "session.hello",
                robot_id="g1-dev",
                session_id=None,
                payload={
                    "agent_version": "test-agent",
                    "boot_id": BOOT_ID,
                    "supported_versions": ["1.0"],
                    "transport_mode": "client",
                    "capabilities_digest": CAPABILITY_DIGEST,
                },
        ))
        try:
            websocket.receive_text()
        except (RuntimeError, WebSocketDisconnect):
            pass
        snapshot = peer.connection_snapshot()
        assert snapshot["connected"] is False
        assert "digest mismatch" in (snapshot["lastError"] or "")
