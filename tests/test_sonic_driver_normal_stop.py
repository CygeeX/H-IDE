from __future__ import annotations

import threading
from types import SimpleNamespace
from uuid import UUID

import pytest

from hide_mujoco_sonic.normal_stop import NormalStopIdentity, NormalStopState
from hide_mujoco_sonic.normal_stop_wire import (
    NormalStopWireError,
    normal_stop_begin_request,
    normal_stop_stationary_request,
    parse_normal_stop_response,
)
from hide_mujoco_sonic.reference_authority import ReferenceAuthorityState
from hide_mujoco_sonic.sonic_driver import SonicDriver, SonicIntentIngress


DRIVER_BOOT_ID = UUID("30000000-0000-4000-8000-000000000003")
BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000002")
NEW_BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000004")
REQUEST_ID = UUID("10000000-0000-4000-8000-000000000001")
STATIONARY_REQUEST_ID = UUID("10000000-0000-4000-8000-000000000010")
ROBOT_ID = "g1-sim"


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class Guard:
    def __init__(self) -> None:
        self.stops: list[str] = []
        self._command_zero = True

    @property
    def command_zero(self) -> bool:
        return self._command_zero

    def stop(self, *, source: str) -> None:
        self.stops.append(source)
        self._command_zero = True


class Telemetry:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, **values: object) -> None:
        self.values.update(values)


def bare_driver(clock: Clock) -> SonicDriver:
    driver = SonicDriver.__new__(SonicDriver)
    driver.config = SimpleNamespace(robot_id=ROBOT_ID, plant_port=8794)
    driver.telemetry = Telemetry()
    driver._driver_boot_id = DRIVER_BOOT_ID
    driver._support_lock = threading.RLock()
    driver._terminal_fault_lock = threading.Lock()
    driver._fault_latched = threading.Event()
    driver._terminal_plant_halt_attempted = False
    driver._terminal_fault_reason = None
    driver._stop = threading.Event()
    driver._intent_ingress = SonicIntentIngress(clock=clock, max_age_s=0.5)
    driver.guard = Guard()
    driver._navigation_task_active = False
    driver._intent_gate = None
    driver._joint_handoff = SimpleNamespace(freeze=lambda _targets: None)
    driver._arm_guard = SimpleNamespace(anchor=lambda _targets: None)
    driver._last_published_targets = tuple(0.0 for _ in range(29))
    driver._safe_hold_targets = None
    driver._static_hold_active = False
    driver._plant_support_mode = "idle"
    driver._plant_support_stop_pending = False
    driver._plant_support_safety_latched = False
    driver._plant_support_stop_generation = None
    driver._reference_authority = ReferenceAuthorityState()
    driver._reference_request_pending = False
    driver._vendor_revoke_pending = False
    driver._normal_stop = NormalStopState()
    driver._normal_stop_bridge_boot_id = None
    driver._normal_stop_seen_bridge_boot_ids = set()
    driver._normal_stop_plant_safety_stop_epoch = None
    driver._normal_stop_last_confirmed_plant_epoch = 0
    driver._normal_stop_begin_intent = None
    driver._normal_stop_seen_request_ids = set()
    driver._normal_stop_replays = {}
    return driver


def stopping_intent(
    clock: Clock,
    *,
    generation: int,
    sequence: int,
) -> dict[str, object]:
    return {
        "type": "sonic_intent",
        "schema_version": "h-ide.sonic-intent/v1",
        "sequence": sequence,
        "generation": generation,
        "generated_at_s": clock.value,
        "state": "STOPPING",
        "mode": "IDLE",
        "velocity_x_mps": 0.0,
        "velocity_y_mps": 0.0,
        "linear_speed_mps": 0.0,
        "movement_angle_rad": 0.0,
        "yaw_rate_rps": 0.0,
        "facing_delta_rad": 0.0,
        "motion_enabled": False,
        "stop_requested": True,
        "stop_acknowledged": False,
        "head_yaw_rad": 0.0,
        "head_pitch_rad": 0.0,
        "source": "nav2",
        "reason": "correlated normal stop",
        "measured_heading_yaw": 0.25,
    }


def begin_request(
    clock: Clock,
    *,
    request_id: UUID = REQUEST_ID,
    bridge_boot_id: UUID = BRIDGE_BOOT_ID,
    generation: int = 10,
    sequence: int = 20,
) -> dict[str, object]:
    return normal_stop_begin_request(
        request_id=request_id,
        robot_id=ROBOT_ID,
        bridge_boot_id=bridge_boot_id,
        reason="Robot Agent requested a controlled stop",
        intent=stopping_intent(
            clock,
            generation=generation,
            sequence=sequence,
        ),
    )


def stationary_intent(
    clock: Clock,
    *,
    generation: int,
    sequence: int,
) -> dict[str, object]:
    intent = stopping_intent(
        clock,
        generation=generation,
        sequence=sequence,
    )
    intent["state"] = "DISARMED"
    intent["stop_acknowledged"] = True
    intent["reason"] = "fresh stationary odometry acknowledged"
    return intent


def stationary_request(
    clock: Clock,
    *,
    identity: NormalStopIdentity,
    request_id: UUID = STATIONARY_REQUEST_ID,
    bridge_boot_id: UUID = BRIDGE_BOOT_ID,
    driver_boot_id: UUID = DRIVER_BOOT_ID,
    generation: int | None = None,
    sequence: int | None = None,
) -> dict[str, object]:
    return normal_stop_stationary_request(
        request_id=request_id,
        robot_id=ROBOT_ID,
        bridge_boot_id=bridge_boot_id,
        driver_boot_id=driver_boot_id,
        identity=identity,
        intent=stationary_intent(
            clock,
            generation=(
                identity.control_generation
                if generation is None
                else generation
            ),
            sequence=(
                identity.gait_source_sequence + 1
                if sequence is None
                else sequence
            ),
        ),
    )


def complete_state(driver: SonicDriver) -> None:
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    driver._normal_stop.acknowledge_stationary(identity)
    driver._normal_stop.confirm_plant(identity)
    driver._normal_stop.confirm_driver(identity)


def test_begin_atomically_freezes_and_binds_one_braking_transaction() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    payload = begin_request(clock)

    response = driver.accept_json(payload)

    assert response is not None
    parsed = parse_normal_stop_response(
        response,
        expected_request_id=REQUEST_ID,
        expected_robot_id=ROBOT_ID,
        expected_driver_boot_id=DRIVER_BOOT_ID,
        expected_control_generation=10,
        expected_gait_source_sequence=20,
        expected_phases=frozenset({"BRAKING"}),
    )
    assert parsed.normal_stop.stop_epoch == 1
    assert parsed.normal_stop.safe_hold_confirmed is False
    assert driver._normal_stop_bridge_boot_id == BRIDGE_BOOT_ID
    assert driver._normal_stop_seen_bridge_boot_ids == {BRIDGE_BOOT_ID}
    assert driver._intent_ingress.generation == 10
    assert driver._intent_ingress.sequence == 20
    assert driver._intent_ingress.armed is False
    assert driver._safe_hold_targets == tuple(0.0 for _ in range(29))
    assert driver._static_hold_active is True
    assert driver._plant_support_stop_pending is False
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert driver.guard.stops == ["normal_stop_begin"]
    assert driver.accept_json(payload) == response
    assert driver.guard.stops == ["normal_stop_begin"]


def test_same_request_id_replays_first_response_through_all_later_phases() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    payload = begin_request(clock)
    first_response = driver.accept_json(payload)
    assert first_response is not None
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    )

    driver._normal_stop.acknowledge_stationary(identity)
    assert driver.accept_json(payload) == first_response
    driver._normal_stop.confirm_plant(identity)
    driver._normal_stop_plant_safety_stop_epoch = 9
    assert driver.accept_json(payload) == first_response
    driver._normal_stop.confirm_driver(identity)
    assert driver.accept_json(payload) == first_response

    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    ) == ingress_before
    assert driver.guard.stops == ["normal_stop_begin"]

    new_request_id = UUID("10000000-0000-4000-8000-000000000099")
    with pytest.raises(NormalStopWireError, match="braking request"):
        driver.accept_json(
            begin_request(clock, request_id=new_request_id)
        )
    assert driver.guard.stops == ["normal_stop_begin"]


def test_begin_rejects_nonzero_extra_and_stale_intent_before_state_mutation() -> None:
    clock = Clock()

    for mutate, expected_stops in (
        (lambda intent: intent.__setitem__("velocity_x_mps", 0.01), []),
        (lambda intent: intent.__setitem__("extra", True), []),
        (
            lambda intent: intent.__setitem__(
                "generated_at_s", clock.value - 1.0
            ),
            ["normal_stop_begin"],
        ),
    ):
        driver = bare_driver(clock)
        payload = begin_request(clock)
        intent = dict(payload["intent"])  # type: ignore[arg-type]
        mutate(intent)
        payload = {**payload, "intent": intent}

        with pytest.raises((NormalStopWireError, ValueError)):
            driver.accept_json(payload)

        assert driver._normal_stop.snapshot.phase == "NONE"
        assert driver._normal_stop_bridge_boot_id is None
        assert driver._intent_ingress.generation is None
        # Freshness is enforced by ingress after the one-way local freeze.
        # Invalid shape/nonzero values are rejected by the strict wire layer
        # before any boundary work.
        assert driver.guard.stops == expected_stops
        assert driver._plant_support_stop_pending is False


def test_late_support_response_is_dropped_without_early_plant_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    entered = threading.Event()
    release = threading.Event()
    plant_calls: list[str] = []

    def delayed_support(_host: str, _port: int, mode: str, **_kwargs: object):
        plant_calls.append(mode)
        entered.set()
        assert release.wait(timeout=2.0)
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        delayed_support,
    )
    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(driver._set_plant_support_mode("locomotion"))
    )
    worker.start()
    assert entered.wait(timeout=2.0)

    driver.accept_json(begin_request(clock))
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._plant_support_stop_pending is False
    release.set()
    worker.join(timeout=2.0)

    assert worker.is_alive() is False
    assert result == [False]
    assert plant_calls == ["locomotion"]
    assert driver._plant_support_mode == "idle"
    assert driver._plant_support_safety_latched is False
    assert driver._normal_stop.snapshot.phase == "BRAKING"


def test_support_response_final_check_and_commit_share_the_stop_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        lambda _host, _port, mode, **_kwargs: {"mode": mode},
    )
    original_token = driver._support_transition_token_locked
    at_final_check = threading.Event()
    release_final_check = threading.Event()
    token_calls = 0

    def gated_token() -> tuple[int, int | None]:
        nonlocal token_calls
        token_calls += 1
        token = original_token()
        if token_calls == 3:
            at_final_check.set()
            assert release_final_check.wait(timeout=2.0)
        return token

    driver._support_transition_token_locked = gated_token  # type: ignore[method-assign]
    support_result: list[bool] = []
    support_worker = threading.Thread(
        target=lambda: support_result.append(
            driver._set_plant_support_mode("locomotion")
        )
    )
    support_worker.start()
    assert at_final_check.wait(timeout=2.0)

    begin_result: list[dict[str, object] | None] = []
    begin_worker = threading.Thread(
        target=lambda: begin_result.append(driver.accept_json(begin_request(clock)))
    )
    begin_worker.start()
    assert begin_worker.is_alive() is True
    release_final_check.set()
    support_worker.join(timeout=2.0)
    begin_worker.join(timeout=2.0)

    assert support_result == [True]
    assert len(begin_result) == 1
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    # Model the later, correlated Plant confirmation after stationary ACK.
    # The already-returned old HTTP response cannot overwrite these facts.
    with driver._support_lock:
        driver._plant_support_mode = "safety_stop"
        driver._plant_support_safety_latched = True
        driver._plant_support_stop_pending = False
        driver.telemetry.update(
            plant_support_mode="safety_stop",
            plant_support_confirmed_mode="safety_stop",
            plant_support_safety_latched=True,
        )

    assert support_worker.is_alive() is False
    assert begin_worker.is_alive() is False
    assert driver._plant_support_mode == "safety_stop"
    assert driver._plant_support_safety_latched is True
    assert driver.telemetry.values["plant_support_confirmed_mode"] == "safety_stop"


def test_request_id_replay_requires_the_same_canonical_payload() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    payload = begin_request(clock)
    response = driver.accept_json(payload)
    assert response is not None
    before = driver._normal_stop.snapshot

    changed = {**payload, "reason": "different controlled stop reason"}
    with pytest.raises(NormalStopWireError, match="reused"):
        driver.accept_json(changed)

    assert driver._normal_stop.snapshot is before
    assert driver._intent_ingress.sequence == 20
    assert driver.guard.stops == ["normal_stop_begin"]


def test_stationary_ack_is_exact_and_only_then_marks_plant_pending() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.accept_json(begin_request(clock))
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    payload = stationary_request(clock, identity=identity)

    response = driver.accept_json(payload)

    assert response is not None
    parsed = parse_normal_stop_response(
        response,
        expected_request_id=STATIONARY_REQUEST_ID,
        expected_robot_id=ROBOT_ID,
        expected_driver_boot_id=DRIVER_BOOT_ID,
        expected_identity=identity,
        expected_control_generation=10,
        expected_gait_source_sequence=20,
        expected_phases=frozenset({"STATIONARY_ACK"}),
    )
    assert parsed.plant_safety_stop_epoch is None
    assert parsed.normal_stop.safe_hold_confirmed is False
    assert driver._intent_ingress.sequence == 21
    assert driver._intent_ingress.armed is False
    assert driver._plant_support_stop_pending is True
    assert driver.guard.stops == ["normal_stop_begin"]

    driver._normal_stop.confirm_plant(identity)
    driver._normal_stop_plant_safety_stop_epoch = 9
    assert driver.accept_json(payload) == response
    driver._normal_stop.confirm_driver(identity)
    assert driver.accept_json(payload) == response
    assert driver._intent_ingress.sequence == 21
    assert driver.guard.stops == ["normal_stop_begin"]


def test_old_stationary_identity_and_boot_are_rejected_before_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.accept_json(begin_request(clock))
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    old_identity = NormalStopIdentity(
        UUID("40000000-0000-4000-8000-000000000040"),
        identity.control_generation,
        identity.gait_source_sequence,
        identity.stop_epoch,
    )
    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    )

    def unexpected_ingress(_payload: dict[str, object]):
        raise AssertionError("old stationary evidence reached intent ingress")

    monkeypatch.setattr(driver._intent_ingress, "accept", unexpected_ingress)
    with pytest.raises(NormalStopWireError, match="transaction identity"):
        driver.accept_json(stationary_request(clock, identity=old_identity))
    with pytest.raises(NormalStopWireError, match="bridge boot"):
        driver.accept_json(
            stationary_request(
                clock,
                identity=identity,
                bridge_boot_id=NEW_BRIDGE_BOOT_ID,
            )
        )

    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    ) == ingress_before
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._plant_support_stop_pending is False


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("generation", 11, "generation mismatch"),
        ("sequence", 20, "follow the first"),
        ("velocity_x_mps", 0.01, "exactly zero"),
        ("stop_acknowledged", False, "exact DISARMED"),
    ),
)
def test_stationary_contract_failure_has_no_ingress_or_pending_side_effect(
    field: str,
    value: object,
    match: str,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.accept_json(begin_request(clock))
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    payload = stationary_request(clock, identity=identity)
    intent = dict(payload["intent"])  # type: ignore[arg-type]
    intent[field] = value
    payload = {**payload, "intent": intent}
    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    )

    with pytest.raises(NormalStopWireError, match=match):
        driver.accept_json(payload)

    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    ) == ingress_before
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._plant_support_stop_pending is False


@pytest.mark.parametrize("terminal", [False, True])
def test_late_stationary_ack_after_shutdown_or_terminal_has_no_side_effect(
    terminal: bool,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.accept_json(begin_request(clock))
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    payload = stationary_request(clock, identity=identity)
    before = driver._normal_stop.snapshot
    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    )
    if terminal:
        driver._fault_latched.set()
        expected = "terminal fault"
    else:
        driver._stop.set()
        expected = "shutdown"

    with pytest.raises(RuntimeError, match=expected):
        driver.accept_json(payload)

    assert driver._normal_stop.snapshot is before
    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
    ) == ingress_before
    assert driver._plant_support_stop_pending is False


@pytest.mark.parametrize("stage", ["begin", "stationary"])
def test_shutdown_winning_linearization_prevents_normal_stop_commit(
    stage: str,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    if stage == "stationary":
        driver.accept_json(begin_request(clock))
        identity = driver._normal_stop.snapshot.identity
        assert identity is not None
        payload = stationary_request(clock, identity=identity)
        expected_phase = "BRAKING"
        expected_sequence = 20
        expected_bridge_boot = BRIDGE_BOOT_ID
    else:
        payload = begin_request(clock)
        expected_phase = "NONE"
        expected_sequence = 0
        expected_bridge_boot = None
    before = driver._normal_stop.snapshot
    entered_validation = threading.Event()
    release_validation = threading.Event()
    original_validate = driver._validate_normal_stop_intent

    def gated_validate(*args: object, **kwargs: object):
        result = original_validate(*args, **kwargs)  # type: ignore[arg-type]
        entered_validation.set()
        assert release_validation.wait(timeout=2.0)
        return result

    driver._validate_normal_stop_intent = gated_validate  # type: ignore[method-assign]
    errors: list[Exception] = []

    def invoke_handler() -> None:
        try:
            driver.accept_json(payload)
        except Exception as exc:
            errors.append(exc)

    handler = threading.Thread(target=invoke_handler)
    handler.start()
    assert entered_validation.wait(timeout=2.0)
    shutdown = threading.Thread(target=driver.stop)
    shutdown.start()
    assert driver._stop.wait(timeout=2.0)
    release_validation.set()
    handler.join(timeout=2.0)
    shutdown.join(timeout=2.0)

    assert len(errors) == 1
    assert "shutdown" in str(errors[0])
    assert driver._normal_stop.snapshot is before
    assert driver._normal_stop.snapshot.phase == expected_phase
    assert driver._normal_stop_bridge_boot_id == expected_bridge_boot
    assert driver._intent_ingress.sequence == expected_sequence
    assert driver._plant_support_stop_pending is False


@pytest.mark.parametrize("stage", ["begin", "stationary"])
def test_normal_stop_commit_winning_linearization_precedes_shutdown(
    stage: str,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    if stage == "stationary":
        driver.accept_json(begin_request(clock))
        identity = driver._normal_stop.snapshot.identity
        assert identity is not None
        payload = stationary_request(clock, identity=identity)
        expected_phase = "STATIONARY_ACK"
        expected_pending = True
    else:
        payload = begin_request(clock)
        expected_phase = "BRAKING"
        expected_pending = False
    entered_commit = threading.Event()
    release_commit = threading.Event()
    original_accept = driver._intent_ingress.accept

    def gated_accept(*args: object, **kwargs: object):
        result = original_accept(*args, **kwargs)  # type: ignore[arg-type]
        entered_commit.set()
        assert release_commit.wait(timeout=2.0)
        return result

    driver._intent_ingress.accept = gated_accept  # type: ignore[method-assign]
    responses: list[dict[str, object] | None] = []
    errors: list[Exception] = []

    def invoke_handler() -> None:
        try:
            responses.append(driver.accept_json(payload))
        except Exception as exc:
            errors.append(exc)

    handler = threading.Thread(target=invoke_handler)
    handler.start()
    assert entered_commit.wait(timeout=2.0)
    shutdown_started = threading.Event()

    def invoke_shutdown() -> None:
        shutdown_started.set()
        driver.stop()

    shutdown = threading.Thread(target=invoke_shutdown)
    shutdown.start()
    assert shutdown_started.wait(timeout=2.0)
    release_commit.set()
    handler.join(timeout=2.0)
    shutdown.join(timeout=2.0)

    assert errors == []
    assert len(responses) == 1
    assert driver._normal_stop.snapshot.phase == expected_phase
    assert driver._plant_support_stop_pending is expected_pending
    assert driver._stop.is_set() is True


def test_terminal_fault_claim_and_plant_halt_do_not_wait_for_begin_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    entered_boundary = threading.Event()
    release_boundary = threading.Event()
    plant_halt_called = threading.Event()
    original_boundary = driver._stop_navigation_at_boundary

    def gated_boundary(*, source: str, publish_telemetry: bool = True) -> bool:
        if source == "normal_stop_begin":
            entered_boundary.set()
            assert release_boundary.wait(timeout=2.0)
        return original_boundary(
            source=source,
            publish_telemetry=publish_telemetry,
        )

    def plant_halt(
        _host: str,
        _port: int,
        _reason: str,
        *,
        timeout: float,
    ) -> None:
        assert timeout == pytest.approx(0.30)
        plant_halt_called.set()

    driver._stop_navigation_at_boundary = gated_boundary  # type: ignore[method-assign]
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        plant_halt,
    )
    begin_errors: list[Exception] = []

    def invoke_begin() -> None:
        try:
            driver.accept_json(begin_request(clock))
        except Exception as exc:
            begin_errors.append(exc)

    begin_worker = threading.Thread(target=invoke_begin)
    begin_worker.start()
    assert entered_boundary.wait(timeout=2.0)
    terminal_worker = threading.Thread(
        target=lambda: driver._latch_terminal_fault("concurrent terminal fault")
    )
    terminal_worker.start()

    assert plant_halt_called.wait(timeout=2.0)
    assert driver._fault_latched.is_set() is True
    assert terminal_worker.is_alive() is True
    assert driver._normal_stop.snapshot.phase == "NONE"
    assert driver._intent_ingress.sequence == 0

    release_boundary.set()
    begin_worker.join(timeout=2.0)
    terminal_worker.join(timeout=2.0)

    assert begin_worker.is_alive() is False
    assert terminal_worker.is_alive() is False
    assert len(begin_errors) == 1
    assert "terminal fault" in str(begin_errors[0])
    assert driver._normal_stop.snapshot.phase == "NONE"
    assert driver._normal_stop_bridge_boot_id is None
    assert driver._intent_ingress.sequence == 0
    assert driver._plant_support_stop_pending is False


def test_new_bridge_boot_resets_only_sequence_after_confirmed_generation() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    first_request = begin_request(clock, generation=10, sequence=100)
    driver.accept_json(first_request)
    first_identity = driver._normal_stop.snapshot.identity
    assert first_identity is not None
    complete_state(driver)

    clock.value += 0.01
    second_request_id = UUID("10000000-0000-4000-8000-000000000002")
    second = begin_request(
        clock,
        request_id=second_request_id,
        bridge_boot_id=NEW_BRIDGE_BOOT_ID,
        generation=11,
        sequence=2,
    )
    response = driver.accept_json(second)
    assert response is not None
    parsed = parse_normal_stop_response(
        response,
        expected_request_id=second_request_id,
        expected_control_generation=11,
        expected_gait_source_sequence=2,
        expected_phases=frozenset({"BRAKING"}),
    )
    assert parsed.normal_stop.stop_epoch == 2
    assert driver._normal_stop_bridge_boot_id == NEW_BRIDGE_BOOT_ID
    assert driver._normal_stop_seen_bridge_boot_ids == {
        BRIDGE_BOOT_ID,
        NEW_BRIDGE_BOOT_ID,
    }

    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
    )
    with pytest.raises(NormalStopWireError, match="older bridge boot"):
        driver.accept_json(first_request)
    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
    ) == ingress_before
    assert driver._normal_stop.snapshot.identity == parsed.normal_stop.identity

    complete_state(driver)
    clock.value += 0.01
    same_boot_reset = begin_request(
        clock,
        request_id=UUID("10000000-0000-4000-8000-000000000003"),
        bridge_boot_id=NEW_BRIDGE_BOOT_ID,
        generation=12,
        sequence=1,
    )
    with pytest.raises(NormalStopWireError, match="newer gait source sequence"):
        driver.accept_json(same_boot_reset)
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
    assert driver._normal_stop.snapshot.stop_epoch == 2


def test_failed_new_boot_begin_does_not_change_the_atomic_boot_binding() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.accept_json(begin_request(clock, generation=10, sequence=100))
    complete_state(driver)
    confirmed = driver._normal_stop.snapshot
    clock.value += 0.01
    stale = begin_request(
        clock,
        request_id=UUID("10000000-0000-4000-8000-000000000004"),
        bridge_boot_id=NEW_BRIDGE_BOOT_ID,
        generation=11,
        sequence=2,
    )
    stale_intent = dict(stale["intent"])  # type: ignore[arg-type]
    stale_intent["generated_at_s"] = clock.value - 1.0
    stale = {**stale, "intent": stale_intent}

    with pytest.raises(ValueError, match="stale"):
        driver.accept_json(stale)

    assert driver._normal_stop.snapshot is confirmed
    assert driver._normal_stop_bridge_boot_id == BRIDGE_BOOT_ID
    assert NEW_BRIDGE_BOOT_ID not in driver._normal_stop_seen_bridge_boot_ids


def test_terminal_fault_never_enters_normal_stop() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver._fault_latched.set()

    with pytest.raises(RuntimeError, match="terminal fault"):
        driver.accept_json(begin_request(clock))

    assert driver._normal_stop.snapshot.phase == "NONE"
    assert driver._normal_stop_bridge_boot_id is None
    assert driver.guard.stops == []
