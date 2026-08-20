from __future__ import annotations

from collections.abc import Callable
import json
import socket
import threading
import time
from uuid import UUID

import pytest

from hide_mujoco_sonic.gait_adapter import GaitState, SonicIntent
from hide_mujoco_sonic.motion_fault import (
    SONIC_HEALTH_SCHEMA_VERSION,
    SonicHealth,
    SonicHealthTransportGate,
    sonic_health_probe,
)
from hide_mujoco_sonic.ros_bridge import (
    _attempt_sonic_health_probe,
    _health_transport_command_action,
    _suppress_for_unknown_health_transport,
)
from hide_mujoco_sonic.transport import (
    JsonResponseContractError,
    JsonResponseTimeoutError,
    JsonResponseUnavailableError,
    request_json,
)


BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000002")
DRIVER_BOOT_ID = UUID("30000000-0000-4000-8000-000000000003")
REQUEST_ID = UUID("40000000-0000-4000-8000-000000000004")
NEXT_REQUEST_ID = UUID("50000000-0000-4000-8000-000000000005")


class SequenceClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def gate() -> SonicHealthTransportGate:
    return SonicHealthTransportGate(
        probe_period_s=0.025,
        probe_timeout_s=0.020,
    )


def health_response(request_id: UUID) -> dict[str, object]:
    return {
        "request_id": str(request_id),
        "ok": True,
        "schema_version": SONIC_HEALTH_SCHEMA_VERSION,
        "robot_id": "g1-sim",
        "driver_boot_id": str(DRIVER_BOOT_ID),
        "status": "ready",
        "terminal_fault": False,
        "reason": None,
        "motion_ready": True,
        "last_lowstate_age_s": 0.01,
        "last_control_step_age_s": 0.01,
        "plant_support_mode": "safety_stop",
        "plant_support_safety_latched": True,
        "plant_support_stop_pending": False,
        "static_hold_active": True,
        "navigation_task_active": False,
        "motion_authorized": False,
        "nav2_command_zero": True,
        "matched_reference_actuator_authority": False,
        "reference_authority_state": "REVOKED",
        "reference_authority_generation": None,
        "reference_authority_source_sequence": None,
        "reference_authority_intent_generation": None,
        "reference_authority_epoch": 0,
        "reference_authority_granted_at_monotonic_s": None,
        "reference_authority_granted_tick_ms": None,
        "reference_authority_dds_target_sequence": None,
    }


def strict_health(request_id: UUID) -> SonicHealth:
    return SonicHealth.from_response(
        health_response(request_id),
        expected_request_id=request_id,
        expected_robot_id="g1-sim",
    )


def timeout(_request_id: UUID) -> SonicHealth:
    raise JsonResponseTimeoutError("bounded receive timed out")


def test_attempt_first_timeout_then_second_timeout_confirms_within_70ms() -> None:
    transport = gate()

    first = _attempt_sonic_health_probe(
        REQUEST_ID,
        request_health=timeout,
        transport_gate=transport,
        clock=SequenceClock(10.000, 10.020),
    )
    second = _attempt_sonic_health_probe(
        NEXT_REQUEST_ID,
        request_health=timeout,
        transport_gate=transport,
        clock=SequenceClock(10.025, 10.045),
    )

    assert first.timeout_error is not None
    assert isinstance(first.timeout_error, JsonResponseTimeoutError)
    assert first.transport.transition == "pending"
    assert first.transport.confirmed_cause is None
    assert second.timeout_error is not None
    assert second.transport.transition == "confirmed"
    assert second.transport.confirmed_cause is not None
    assert second.transport.confirmation_budget_s == pytest.approx(0.070)
    assert second.transport.confirmed_timeout_count == 1


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError("refused"),
        ConnectionResetError("reset"),
        JsonResponseUnavailableError("EOF"),
        JsonResponseContractError("invalid JSON"),
        ValueError("invalid v2 schema"),
        TimeoutError("application timeout"),
    ],
)
def test_non_timeout_probe_failures_propagate_without_entering_gate(
    error: Exception,
) -> None:
    transport = gate()

    def fail(_request_id: UUID) -> SonicHealth:
        raise error

    with pytest.raises(type(error), match=str(error)):
        _attempt_sonic_health_probe(
            REQUEST_ID,
            request_health=fail,
            transport_gate=transport,
            clock=SequenceClock(20.0),
        )

    assert transport.pending is None
    assert transport.confirmed_cause is None


def test_only_fully_parsed_correlated_v2_can_recover_timeout() -> None:
    transport = gate()
    _attempt_sonic_health_probe(
        REQUEST_ID,
        request_health=timeout,
        transport_gate=transport,
        clock=SequenceClock(30.000, 30.020),
    )

    def invalid(identity: UUID) -> SonicHealth:
        value = health_response(identity)
        value["motion_ready"] = "yes"
        return SonicHealth.from_response(
            value,
            expected_request_id=identity,
            expected_robot_id="g1-sim",
        )

    with pytest.raises(ValueError, match="flags must be booleans"):
        _attempt_sonic_health_probe(
            NEXT_REQUEST_ID,
            request_health=invalid,
            transport_gate=transport,
            clock=SequenceClock(30.025),
        )
    assert transport.pending is not None

    recovered = _attempt_sonic_health_probe(
        NEXT_REQUEST_ID,
        request_health=strict_health,
        transport_gate=transport,
        clock=SequenceClock(30.030, 30.035),
    )

    assert recovered.health is not None
    assert recovered.health.request_id == NEXT_REQUEST_ID
    assert recovered.transport.transition == "recovered"
    assert recovered.transport.pending is None
    assert recovered.transport.recovered_timeout_count == 1


def test_expired_pending_gate_does_not_issue_another_request() -> None:
    transport = gate()
    _attempt_sonic_health_probe(
        REQUEST_ID,
        request_health=timeout,
        transport_gate=transport,
        clock=SequenceClock(40.000, 40.020),
    )
    calls: list[UUID] = []

    def record(identity: UUID) -> SonicHealth:
        calls.append(identity)
        return strict_health(identity)

    expired = _attempt_sonic_health_probe(
        NEXT_REQUEST_ID,
        request_health=record,
        transport_gate=transport,
        clock=SequenceClock(40.070),
    )

    assert expired.request_attempted is False
    assert expired.health is None
    assert expired.transport.transition == "confirmed"
    assert expired.transport.confirmed_cause is not None
    assert calls == []


@pytest.mark.parametrize(
    ("fenced", "state", "command", "expected"),
    [
        (True, GaitState.IDLE, (0.2, 0.0, 0.0), "block"),
        (True, GaitState.DISARMED, (0.2, 0.0, 0.0), "block"),
        (True, GaitState.STARTING, (0.2, 0.0, 0.0), "cancel_start"),
        (True, GaitState.STOPPING, (0.2, 0.0, 0.0), "block"),
        (True, GaitState.WALKING, (0.2, 0.0, 0.0), "block"),
        (True, GaitState.TURNING, (0.0, 0.0, 0.2), "block"),
        (True, GaitState.IDLE, (0.0, 0.0, 0.0), "allow"),
        (False, GaitState.IDLE, (0.2, 0.0, 0.0), "allow"),
    ],
)
def test_health_transport_command_fence_has_no_new_start_window(
    fenced: bool,
    state: GaitState,
    command: tuple[float, float, float],
    expected: str,
) -> None:
    assert _health_transport_command_action(
        fenced=fenced,
        gait_state=state,
        vx=command[0],
        vy=command[1],
        wz=command[2],
    ) == expected


def intent(*, state: GaitState, motion_enabled: bool) -> SonicIntent:
    return SonicIntent(
        sequence=2,
        generation=10,
        generated_at_s=1.0,
        state=state.value,
        mode="WALK" if motion_enabled else "IDLE",
        velocity_x_mps=0.2 if motion_enabled else 0.0,
        velocity_y_mps=0.0,
        linear_speed_mps=0.2 if motion_enabled else 0.0,
        movement_angle_rad=0.0,
        yaw_rate_rps=0.0,
        facing_delta_rad=0.0,
        motion_enabled=motion_enabled,
        stop_requested=not motion_enabled,
        stop_acknowledged=False,
        head_yaw_rad=0.0,
        head_pitch_rad=0.0,
        source="test",
        reason="test",
    )


@pytest.mark.parametrize(
    ("fenced", "state", "motion_enabled", "suppressed"),
    [
        (True, GaitState.WALKING, True, True),
        (True, GaitState.TURNING, True, True),
        (True, GaitState.IDLE, False, False),
        (True, GaitState.STOPPING, False, False),
        (True, GaitState.DISARMED, False, False),
        (True, GaitState.FAULT, True, False),
        (False, GaitState.WALKING, True, False),
    ],
)
def test_unknown_transport_suppresses_motion_envelopes_but_never_stop_or_fault(
    fenced: bool,
    state: GaitState,
    motion_enabled: bool,
    suppressed: bool,
) -> None:
    assert _suppress_for_unknown_health_transport(
        fenced,
        intent(state=state, motion_enabled=motion_enabled),
    ) is suppressed


def _serve_timeout_then_success(
    server: socket.socket,
    first_closed: threading.Event,
) -> None:
    first, _address = server.accept()
    with first:
        first.recv(4096)
        time.sleep(0.030)
    first_closed.set()
    second, _address = server.accept()
    with second:
        raw = b""
        while b"\n" not in raw:
            raw += second.recv(4096)
        request = json.loads(raw.split(b"\n", 1)[0])
        response = health_response(UUID(str(request["request_id"])))
        second.sendall(
            (json.dumps(response, separators=(",", ":")) + "\n").encode()
        )


def test_fake_server_first_timeout_then_strict_v2_recovery() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(2)
    first_closed = threading.Event()
    worker = threading.Thread(
        target=_serve_timeout_then_success,
        args=(server, first_closed),
        daemon=True,
    )
    worker.start()
    port = int(server.getsockname()[1])
    transport = gate()

    def request_health(identity: UUID) -> SonicHealth:
        response = request_json(
            "127.0.0.1",
            port,
            sonic_health_probe(
                robot_id="g1-sim",
                producer_boot_id=BRIDGE_BOOT_ID,
                request_id=identity,
            ),
            timeout=0.005,
        )
        return SonicHealth.from_response(
            response,
            expected_request_id=identity,
            expected_robot_id="g1-sim",
        )

    try:
        first = _attempt_sonic_health_probe(
            REQUEST_ID,
            request_health=request_health,
            transport_gate=transport,
        )
        assert first.transport.transition == "pending"
        assert first_closed.wait(timeout=0.060)
        recovered = _attempt_sonic_health_probe(
            NEXT_REQUEST_ID,
            request_health=request_health,
            transport_gate=transport,
        )
    finally:
        server.close()
        worker.join(timeout=1.0)

    assert recovered.health is not None
    assert recovered.transport.transition == "recovered"
    assert recovered.transport.pending is None


def test_request_json_classifies_clean_eof_as_unavailable() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])

    def close_without_reply() -> None:
        connection, _address = server.accept()
        with connection:
            connection.recv(4096)

    worker = threading.Thread(target=close_without_reply, daemon=True)
    worker.start()
    try:
        with pytest.raises(JsonResponseUnavailableError, match="complete response"):
            request_json(
                "127.0.0.1",
                port,
                {"type": "probe", "request_id": str(REQUEST_ID)},
                timeout=0.1,
            )
    finally:
        server.close()
        worker.join(timeout=1.0)
