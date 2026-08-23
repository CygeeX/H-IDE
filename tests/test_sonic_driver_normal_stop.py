from __future__ import annotations

import threading
from types import SimpleNamespace
from uuid import UUID

import pytest

from hide_mujoco_sonic.motion_fault import SonicHealth, sonic_health_probe
from hide_mujoco_sonic.normal_stop import NormalStopIdentity, NormalStopState
from hide_mujoco_sonic.normal_stop_wire import (
    NormalStopWireError,
    normal_stop_begin_request,
    normal_stop_plant_request,
    normal_stop_stationary_request,
    parse_normal_stop_response,
)
from hide_mujoco_sonic.reference_authority import ReferenceAuthorityState
from hide_mujoco_sonic.safety import NominalPolicyActivation
from hide_mujoco_sonic.sonic_driver import (
    SonicDriver,
    SonicIntentIngress,
    _bind_command_server,
    _handle_command_connection,
)
from hide_mujoco_sonic.transport import request_json


DRIVER_BOOT_ID = UUID("30000000-0000-4000-8000-000000000003")
BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000002")
NEW_BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000004")
REQUEST_ID = UUID("10000000-0000-4000-8000-000000000001")
STATIONARY_REQUEST_ID = UUID("10000000-0000-4000-8000-000000000010")
PLANT_REQUEST_ID = UUID("10000000-0000-4000-8000-000000000020")
ROBOT_ID = "g1-sim"


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class Guard:
    def __init__(self) -> None:
        self.stops: list[str] = []
        self.measured_heading_stops: list[str] = []
        self.accepted: list[tuple[float, float, float, str]] = []
        self._target_zero = True
        self._command_zero = True

    @property
    def target_zero(self) -> bool:
        return self._target_zero

    @property
    def command_zero(self) -> bool:
        return self._command_zero

    def stop(self, *, source: str) -> None:
        self.stops.append(source)
        self._target_zero = True
        self._command_zero = True

    def stop_at_measured_heading(self, *, source: str) -> None:
        self.measured_heading_stops.append(source)
        self.stop(source=source)

    def observe_heading(self, _heading_yaw: float) -> None:
        return None

    def accept(self, vx: float, vy: float, wz: float, *, source: str) -> None:
        self.accepted.append((vx, vy, wz, source))
        self._target_zero = not any(
            abs(value) > 1e-9 for value in (vx, vy, wz)
        )
        self._command_zero = self._target_zero


class Telemetry:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, **values: object) -> None:
        self.values.update(values)

    def reset_navigation_task_peaks(self) -> None:
        return None


class ObservedLock:
    """Plain lock that exposes a deterministic blocked-acquire test hook."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquire_attempted = threading.Event()

    def acquire(self, blocking: bool = True, timeout: float = -1.0) -> bool:
        self.acquire_attempted.set()
        if timeout < 0:
            return self._lock.acquire(blocking)
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> ObservedLock:
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


def bare_driver(clock: Clock) -> SonicDriver:
    driver = SonicDriver.__new__(SonicDriver)
    driver.config = SimpleNamespace(robot_id=ROBOT_ID, plant_port=0)
    driver.telemetry = Telemetry()
    driver._driver_boot_id = DRIVER_BOOT_ID
    driver._driver_session_id = UUID("31000000-0000-4000-8000-000000000003")
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
    driver._arm_guard = SimpleNamespace(
        anchor=lambda _targets: None,
        reset=lambda: None,
    )
    driver._last_published_targets = tuple(0.0 for _ in range(29))
    driver._safe_hold_targets = None
    driver._static_hold_active = False
    driver._plant_support_mode = "idle"
    driver._plant_support_stop_pending = False
    driver._plant_support_safety_latched = False
    driver._plant_support_stop_generation = None
    driver._reference_authority = ReferenceAuthorityState()
    driver._reference_request_pending = False
    driver._reference_request_continuous = False
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


def balanced_idle_intent(
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
    intent["state"] = "IDLE"
    intent["stop_requested"] = False
    intent["stop_acknowledged"] = True
    intent["reason"] = None
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


def plant_success(
    *,
    epoch: int = 9,
    last_rearm_generation: int | None = 10,
) -> dict[str, object]:
    return {
        "confirmed": True,
        "mode": "safety_stop",
        "safety_stop_latched": True,
        "safety_stop_epoch": epoch,
        "last_rearm_generation": last_rearm_generation,
        "fault": None,
        "error": None,
    }


def plant_failure(error: str = "Plant confirmation timed out") -> dict[str, object]:
    return {
        "confirmed": False,
        "mode": None,
        "safety_stop_latched": None,
        "safety_stop_epoch": None,
        "last_rearm_generation": None,
        "fault": None,
        "error": error,
    }


def plant_request(
    *,
    identity: NormalStopIdentity,
    request_id: UUID = PLANT_REQUEST_ID,
    bridge_boot_id: UUID = BRIDGE_BOOT_ID,
    driver_boot_id: UUID = DRIVER_BOOT_ID,
    plant: dict[str, object] | None = None,
) -> dict[str, object]:
    return normal_stop_plant_request(
        request_id=request_id,
        robot_id=ROBOT_ID,
        bridge_boot_id=bridge_boot_id,
        driver_boot_id=driver_boot_id,
        identity=identity,
        plant=plant_success() if plant is None else plant,
    )


def enter_stationary(driver: SonicDriver, clock: Clock) -> NormalStopIdentity:
    driver.accept_json(begin_request(clock))
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    driver.accept_json(stationary_request(clock, identity=identity))
    assert driver._normal_stop.snapshot.phase == "STATIONARY_ACK"
    return identity


def generic_gait_intent(
    clock: Clock,
    *,
    generation: int,
    sequence: int,
    kind: str,
) -> dict[str, object]:
    payload = stopping_intent(
        clock,
        generation=generation,
        sequence=sequence,
    )
    if kind == "motion":
        payload.update(
            state="WALKING",
            mode="SLOW_WALK",
            velocity_x_mps=0.2,
            linear_speed_mps=0.2,
            motion_enabled=True,
            stop_requested=False,
            stop_acknowledged=False,
        )
    elif kind in {"idle", "control_arm"}:
        payload.update(
            state="IDLE",
            stop_requested=False,
            stop_acknowledged=True,
        )
        if kind == "control_arm":
            payload["control_arm"] = True
            payload["request_id"] = "control-arm-test"
    elif kind == "fault":
        payload.update(
            state="FAULT",
            stop_requested=True,
            stop_acknowledged=False,
            reason="deterministic terminal gait fault",
        )
    elif kind != "stop":
        raise AssertionError(f"unsupported test gait intent kind: {kind}")
    return payload


def enter_normal_phase(
    driver: SonicDriver,
    clock: Clock,
    phase: str,
) -> NormalStopIdentity:
    if phase == "BRAKING":
        driver.accept_json(begin_request(clock))
        identity = driver._normal_stop.snapshot.identity
        assert identity is not None
        return identity
    identity = enter_stationary(driver, clock)
    if phase == "PLANT_CONFIRMED":
        driver._normal_stop.confirm_plant(identity)
    elif phase == "FAILED":
        driver._normal_stop.fail(identity, reason="deterministic normal failure")
    elif phase != "STATIONARY_ACK":
        raise AssertionError(f"unsupported normal-stop phase: {phase}")
    return identity


def configure_control_arm_health(driver: SonicDriver, clock: Clock) -> None:
    driver._control_condition = threading.Condition(threading.RLock())
    driver._motion_ready = threading.Event()
    driver._motion_ready.set()
    driver._wall_clock = clock
    driver._last_lowstate_wall_s = clock.value
    driver._last_control_step_wall_s = clock.value
    driver.config.plant_lowstate_stale_s = 0.10
    driver.config.sonic_control_step_stale_s = 0.10


def commit_rearm_without_publication(
    driver: SonicDriver,
    clock: Clock,
    *,
    generation: int = 11,
) -> None:
    arm_intent = stationary_intent(clock, generation=generation, sequence=1)
    arm_intent.update(
        state="IDLE",
        stop_requested=False,
        control_arm=True,
        reason="explicit control rearm",
    )
    with driver._support_lock:
        decision = driver._intent_ingress.accept(
            arm_intent,
            defer_control_arm=True,
        )
        assert decision.arm_acknowledged is True
        driver._commit_staged_plant_support_rearm_locked(generation)


def _capture_exception(
    errors: list[Exception],
    callback: object,
    *args: object,
) -> None:
    try:
        callback(*args)  # type: ignore[operator]
    except Exception as exc:
        errors.append(exc)


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


def test_granted_motion_rebases_idle_from_exact_boundary_before_braking() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    activation = NominalPolicyActivation(
        duration_s=3.0,
        settle_consecutive_samples=1,
    )
    targets = tuple(0.0 for _ in range(29))
    assert activation.begin(targets, tick_ms=0)
    final = activation.propose(targets, measured=targets, tick_ms=3_000)
    assert activation.commit(final, final.targets)
    activation.observe_settling(
        nominal=targets,
        measured=targets,
        velocities=targets,
        gyroscope=(0.0, 0.0, 0.0),
        roll_rad=0.0,
        pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert activation.record_support_release(activation.epoch)
    assert activation.confirm_support_release(tick_ms=3_020)
    assert activation.prepared
    driver._policy_activation = activation
    driver._controller = SimpleNamespace(initialized=True)
    driver._intent_ingress.generation = 10
    driver._intent_ingress.sequence = 19
    driver._intent_ingress.armed = True
    driver._reference_authority.require(generation=10, source_sequence=19)
    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=19,
        planner_intent_generation=1,
        reference_epoch=1,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
    )
    driver._reference_authority.grant(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_000,
        dds_target_sequence=7,
    )

    response = driver.accept_json(begin_request(clock))

    assert response is not None
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._normal_stop_policy_braking is True
    assert driver._controlled_stop_active is True
    assert driver._controlled_stop_source_identity == (10, 20)
    assert driver._reference_source_identity == (10, 20)
    assert driver._intent_ingress.braking is True
    assert driver._safe_hold_targets is None
    assert driver._static_hold_active is True
    # Correlated STOPPING withdraws outer actuator authority but preserves the
    # acknowledged vendor history/cursor until its continuous IDLE request has
    # captured them.  No deferred vendor revoke may erase that history.
    assert driver._reference_authority.snapshot.phase == "REQUIRED"
    assert driver._reference_authority.snapshot.source_sequence == 20
    assert driver._reference_request_pending is True
    assert driver._reference_request_continuous is True
    assert driver._vendor_revoke_pending is False
    assert driver.guard.stops == ["gait_adapter_controlled_stop"]
    assert driver.guard.measured_heading_stops == [
        "gait_adapter_controlled_stop"
    ]

    configure_control_arm_health(driver, clock)
    request_id = UUID("40000000-0000-4000-8000-000000000004")
    health_payload = driver.accept_json(
        sonic_health_probe(
            robot_id=ROBOT_ID,
            producer_boot_id=BRIDGE_BOOT_ID,
            request_id=request_id,
        )
    )
    assert health_payload is not None
    health = SonicHealth.from_response(
        {"request_id": str(request_id), "ok": True, **health_payload},
        expected_request_id=request_id,
        expected_robot_id=ROBOT_ID,
    )
    assert health.controlled_policy_stop_active is True
    assert health.matched_reference_actuator_authority is False
    assert health.motion_authorized is False
    assert health.navigation_task_active is False
    assert health.nav2_command_zero is True

    identity = driver._normal_stop.snapshot.identity
    assert identity is not None
    clock.value += 0.02
    stationary_payload = stationary_request(clock, identity=identity)
    with pytest.raises(
        NormalStopWireError,
        match="fresh IDLE reference to cross DDS",
    ):
        driver.accept_json(stationary_payload)
    assert driver._normal_stop.snapshot.phase == "BRAKING"

    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=20,
        planner_intent_generation=2,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
    )
    driver._reference_authority.grant(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_020,
        dds_target_sequence=8,
    )
    driver._controlled_stop_reference_installed = True
    stationary = driver.accept_json(stationary_payload)

    assert stationary is not None
    assert driver._normal_stop.snapshot.phase == "STATIONARY_ACK"
    assert driver._controlled_stop_active is False
    assert getattr(driver, "_normal_stop_policy_braking", False) is False
    assert driver._safe_hold_targets == tuple(0.0 for _ in range(29))
    assert driver._static_hold_active is True
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert driver.guard.stops == [
        "gait_adapter_controlled_stop",
        "normal_stop_stationary",
    ]


def test_generic_stopping_intent_is_health_visible_before_correlated_begin() -> None:
    """Zero cmd may precede the normal-stop RPC without inventing a fault."""

    clock = Clock()
    driver = bare_driver(clock)
    activation = NominalPolicyActivation(
        duration_s=3.0,
        settle_consecutive_samples=1,
    )
    targets = tuple(0.0 for _ in range(29))
    assert activation.begin(targets, tick_ms=0)
    final = activation.propose(targets, measured=targets, tick_ms=3_000)
    assert activation.commit(final, final.targets)
    activation.observe_settling(
        nominal=targets,
        measured=targets,
        velocities=targets,
        gyroscope=(0.0, 0.0, 0.0),
        roll_rad=0.0,
        pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert activation.record_support_release(activation.epoch)
    assert activation.confirm_support_release(tick_ms=3_020)
    assert activation.prepared
    driver._policy_activation = activation
    driver._controller = SimpleNamespace(initialized=True)
    driver._intent_ingress.generation = 10
    driver._intent_ingress.sequence = 19
    driver._intent_ingress.armed = True
    driver._navigation_task_active = True
    driver.guard._target_zero = False
    driver.guard._command_zero = False
    driver._reference_authority.require(generation=10, source_sequence=19)
    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=19,
        planner_intent_generation=1,
        reference_epoch=1,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
    )
    driver._reference_authority.grant(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_000,
        dds_target_sequence=7,
    )

    # This is the real scheduling order: the first zero /cmd_vel reaches the
    # regular gait-intent channel before the correlated normal-stop RPC.
    assert driver.accept_json(stopping_intent(clock, generation=10, sequence=20)) is None
    assert driver._normal_stop.snapshot.phase == "NONE"
    assert driver._controlled_stop_active is True
    assert getattr(driver, "_normal_stop_policy_braking", False) is False
    assert driver._reference_authority.snapshot.phase == "REQUIRED"
    assert driver._reference_authority.snapshot.source_sequence == 20
    assert driver._reference_request_pending is True
    assert driver._reference_request_continuous is True
    assert driver._vendor_revoke_pending is False

    # The health probe may run after the zero target is admitted but before
    # the next 50 Hz control tick consumes the previous filtered command.
    # This is a legitimate, driver-owned braking transition, not active Nav2
    # motion and not a final stationary acknowledgement.
    driver.guard._command_zero = False
    assert driver.guard.target_zero is True
    assert driver.guard.command_zero is False

    configure_control_arm_health(driver, clock)
    request_id = UUID("40000000-0000-4000-8000-000000000005")
    health_payload = driver.accept_json(
        sonic_health_probe(
            robot_id=ROBOT_ID,
            producer_boot_id=BRIDGE_BOOT_ID,
            request_id=request_id,
        )
    )
    assert health_payload is not None
    health = SonicHealth.from_response(
        {"request_id": str(request_id), "ok": True, **health_payload},
        expected_request_id=request_id,
        expected_robot_id=ROBOT_ID,
    )
    assert health.controlled_policy_stop_active is True
    assert health.matched_reference_actuator_authority is False
    assert health.motion_authorized is False
    assert health.navigation_task_active is False
    assert health.nav2_command_zero is True
    assert driver.guard.command_zero is False

    # The strict transaction joins the already-owned braking generation with
    # the next GaitAdapter source sequence.  The bridge's mutually-exclusive
    # state callback prevents another zero-command intent from overtaking it.
    response = driver.accept_json(begin_request(clock, sequence=21))
    assert response is not None
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._normal_stop_policy_braking is True
    assert driver._controlled_stop_source_identity == (10, 21)
    assert driver._reference_source_identity == (10, 21)
    assert driver._reference_authority.snapshot.phase == "REQUIRED"
    assert driver._reference_authority.snapshot.source_sequence == 21
    assert driver._reference_request_pending is True


def test_generic_stationary_idle_keeps_fresh_policy_balance_authority() -> None:
    """A successful action stops locomotion without freezing the humanoid."""

    clock = Clock()
    driver = bare_driver(clock)
    activation = NominalPolicyActivation(
        duration_s=3.0,
        settle_consecutive_samples=1,
    )
    targets = tuple(0.0 for _ in range(29))
    assert activation.begin(targets, tick_ms=0)
    final = activation.propose(targets, measured=targets, tick_ms=3_000)
    assert activation.commit(final, final.targets)
    activation.observe_settling(
        nominal=targets,
        measured=targets,
        velocities=targets,
        gyroscope=(0.0, 0.0, 0.0),
        roll_rad=0.0,
        pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert activation.record_support_release(activation.epoch)
    assert activation.confirm_support_release(tick_ms=3_020)
    driver._policy_activation = activation
    driver._controller = SimpleNamespace(
        initialized=True,
        ms=SimpleNamespace(mode=0, has_movement=False),
    )
    driver._intent_ingress.generation = 10
    driver._intent_ingress.sequence = 19
    driver._intent_ingress.armed = True
    driver._navigation_task_active = True
    driver.guard._target_zero = False
    driver.guard._command_zero = False
    driver._reference_authority.require(generation=10, source_sequence=19)
    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=19,
        planner_intent_generation=1,
        reference_epoch=1,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
    )
    driver._reference_authority.grant(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_000,
        dds_target_sequence=7,
    )

    driver.accept_json(stopping_intent(clock, generation=10, sequence=20))
    assert driver._reference_authority.snapshot.phase == "REQUIRED"
    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=20,
        planner_intent_generation=2,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
    )
    driver._reference_authority.grant(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_020,
        dds_target_sequence=8,
    )
    driver._controlled_stop_reference_installed = True
    driver._reference_request_pending = False
    driver._static_hold_active = False
    driver.guard._target_zero = True
    driver.guard._command_zero = True

    clock.value += 0.40
    assert (
        driver.accept_json(
            balanced_idle_intent(clock, generation=10, sequence=21)
        )
        is None
    )

    assert driver._balanced_policy_idle_active is True
    assert driver._controlled_stop_active is True
    assert driver._intent_ingress.armed is True
    assert driver._intent_ingress.braking is False
    assert driver._navigation_task_active is False
    assert driver._safe_hold_targets is None
    assert driver._static_hold_active is False
    assert driver._plant_support_mode == "idle"
    assert driver._reference_authority.snapshot.phase == "GRANTED"
    assert driver.guard.stops == ["gait_adapter_controlled_stop"]
    assert driver.telemetry.values["balanced_policy_idle_active"] is True

    configure_control_arm_health(driver, clock)
    request_id = UUID("40000000-0000-4000-8000-000000000006")
    health_payload = driver.accept_json(
        sonic_health_probe(
            robot_id=ROBOT_ID,
            producer_boot_id=BRIDGE_BOOT_ID,
            request_id=request_id,
        )
    )
    assert health_payload is not None
    health = SonicHealth.from_response(
        {"request_id": str(request_id), "ok": True, **health_payload},
        expected_request_id=request_id,
        expected_robot_id=ROBOT_ID,
    )
    assert health.controlled_policy_stop_active is True
    assert health.motion_authorized is True
    assert health.navigation_task_active is False
    assert health.nav2_command_zero is True
    assert health.matched_reference_actuator_authority is True
    assert health.confirmed_safe_hold is False

    clock.value += 0.02
    motion = stopping_intent(clock, generation=10, sequence=22)
    motion.update(
        state="STARTING",
        mode="SLOW_WALK",
        velocity_x_mps=0.30,
        linear_speed_mps=0.30,
        motion_enabled=True,
        stop_requested=False,
        stop_acknowledged=False,
        reason=None,
    )
    assert driver.accept_json(motion) is None
    assert driver._balanced_policy_idle_active is False
    assert driver._controlled_stop_active is False
    assert driver._controlled_stop_source_identity is None
    assert driver._controlled_stop_reference_installed is False
    assert driver._navigation_task_active is True
    assert driver._reference_source_identity == (10, 22)
    assert driver.guard.accepted == [(0.30, 0.0, 0.0, "nav2")]
    assert driver.telemetry.values["balanced_policy_idle_active"] is False


def test_stationary_idle_ack_waits_for_first_controlled_policy_dds_grant() -> None:
    """A ROS acknowledgement may race the first executable IDLE policy tick.

    The GaitAdapter proves base stationarity independently from SONIC's async
    planner/reference transaction.  If that acknowledgement arrives while the
    controlled IDLE reference is only PRIMED, freezing the last joint target
    removes learned balance from an untethered humanoid.  Keep the controlled
    stop alive; a later IDLE envelope must commit after the first DDS grant
    even though the ingress arm edge was already consumed by the earlier
    acknowledgement.
    """

    clock = Clock()
    driver = bare_driver(clock)
    activation = NominalPolicyActivation(
        duration_s=3.0,
        settle_consecutive_samples=1,
    )
    targets = tuple(0.0 for _ in range(29))
    assert activation.begin(targets, tick_ms=0)
    final = activation.propose(targets, measured=targets, tick_ms=3_000)
    assert activation.commit(final, final.targets)
    activation.observe_settling(
        nominal=targets,
        measured=targets,
        velocities=targets,
        gyroscope=(0.0, 0.0, 0.0),
        roll_rad=0.0,
        pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert activation.record_support_release(activation.epoch)
    assert activation.confirm_support_release(tick_ms=3_020)
    driver._policy_activation = activation
    driver._controller = SimpleNamespace(
        initialized=True,
        ms=SimpleNamespace(mode=0, has_movement=False),
    )
    driver._intent_ingress.generation = 10
    driver._intent_ingress.sequence = 19
    driver._intent_ingress.armed = True
    driver._navigation_task_active = True
    driver.guard._target_zero = False
    driver.guard._command_zero = False
    driver._reference_authority.require(generation=10, source_sequence=19)
    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=19,
        planner_intent_generation=1,
        reference_epoch=1,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
    )
    driver._reference_authority.grant(
        generation=10,
        source_sequence=19,
        reference_epoch=1,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_000,
        dds_target_sequence=7,
    )

    driver.accept_json(stopping_intent(clock, generation=10, sequence=20))
    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=20,
        planner_intent_generation=2,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
    )
    driver._controlled_stop_reference_installed = True
    driver._reference_request_pending = False
    driver.guard._target_zero = True
    driver.guard._command_zero = True

    clock.value += 0.40
    driver.accept_json(balanced_idle_intent(clock, generation=10, sequence=21))

    assert driver._controlled_stop_active is True
    assert driver._balanced_policy_idle_active is False
    assert driver._safe_hold_targets is None
    assert driver._fault_latched.is_set() is False

    driver._reference_authority.grant(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_020,
        dds_target_sequence=8,
    )
    driver._static_hold_active = False
    clock.value += 0.02
    driver.accept_json(balanced_idle_intent(clock, generation=10, sequence=22))

    assert driver._controlled_stop_active is True
    assert driver._balanced_policy_idle_active is True
    assert driver._safe_hold_targets is None
    assert driver._reference_authority.snapshot.phase == "GRANTED"


def test_explicit_normal_stop_revokes_an_existing_balanced_idle_policy() -> None:
    """An operator/emergency stop remains stronger than balanced IDLE."""

    clock = Clock()
    driver = bare_driver(clock)
    activation = NominalPolicyActivation(
        duration_s=3.0,
        settle_consecutive_samples=1,
    )
    targets = tuple(0.0 for _ in range(29))
    assert activation.begin(targets, tick_ms=0)
    final = activation.propose(targets, measured=targets, tick_ms=3_000)
    assert activation.commit(final, final.targets)
    activation.observe_settling(
        nominal=targets,
        measured=targets,
        velocities=targets,
        gyroscope=(0.0, 0.0, 0.0),
        roll_rad=0.0,
        pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert activation.record_support_release(activation.epoch)
    assert activation.confirm_support_release(tick_ms=3_020)
    driver._policy_activation = activation
    driver._controller = SimpleNamespace(
        initialized=True,
        ms=SimpleNamespace(mode=0, has_movement=False),
    )
    driver._intent_ingress.generation = 10
    driver._intent_ingress.sequence = 21
    driver._intent_ingress.armed = True
    driver._controlled_stop_active = True
    driver._balanced_policy_idle_active = True
    driver._controlled_stop_source_identity = (10, 20)
    driver._controlled_stop_reference_installed = True
    driver._reference_source_identity = (10, 20)
    driver._reference_authority.require(generation=10, source_sequence=20)
    driver._reference_authority.mark_installed(
        generation=10,
        source_sequence=20,
        planner_intent_generation=2,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
    )
    driver._reference_authority.grant(
        generation=10,
        source_sequence=20,
        reference_epoch=2,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=10_020,
        dds_target_sequence=8,
    )

    response = driver.accept_json(begin_request(clock, sequence=22))

    assert response is not None
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._normal_stop_policy_braking is False
    assert driver._balanced_policy_idle_active is False
    assert driver._controlled_stop_active is False
    assert driver._intent_ingress.armed is False
    assert driver._safe_hold_targets == targets
    assert driver._static_hold_active is True
    assert driver._reference_authority.snapshot.phase == "REVOKED"


@pytest.mark.parametrize(
    (
        "intent_mode",
        "controlled_stop",
        "decision_static_hold",
        "activation_waiting_for_release",
        "handoff_ratio",
        "expected",
    ),
    [
        ("SLOW_WALK", False, False, False, 1.0, "locomotion"),
        ("IDLE", True, False, False, 1.0, "idle"),
        ("SLOW_WALK", False, True, False, 1.0, "idle"),
        ("SLOW_WALK", False, False, True, 1.0, "idle"),
        ("SLOW_WALK", False, False, False, 0.0, "idle"),
    ],
)
def test_policy_support_mode_keeps_controlled_braking_in_plant_idle(
    intent_mode: str,
    controlled_stop: bool,
    decision_static_hold: bool,
    activation_waiting_for_release: bool,
    handoff_ratio: float,
    expected: str,
) -> None:
    assert SonicDriver._desired_policy_support_mode(
        intent_mode=intent_mode,
        controlled_stop=controlled_stop,
        decision_static_hold=decision_static_hold,
        activation_waiting_for_release=activation_waiting_for_release,
        handoff_ratio=handoff_ratio,
    ) == expected


def test_only_pre_braking_locomotion_tick_is_dropped_after_policy_stop_wins() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver._normal_stop.begin(
        control_generation=10,
        gait_source_sequence=20,
    )
    driver._normal_stop_policy_braking = True

    with driver._support_lock:
        assert driver._old_locomotion_tick_lost_to_controlled_stop_locked(
            controlled_stop_at_tick_start=False,
        ) is True
        assert driver._old_locomotion_tick_lost_to_controlled_stop_locked(
            controlled_stop_at_tick_start=True,
        ) is False

    driver._normal_stop_policy_braking = False
    with driver._support_lock:
        assert driver._old_locomotion_tick_lost_to_controlled_stop_locked(
            controlled_stop_at_tick_start=False,
        ) is False


def test_braking_ingress_rejects_new_motion_until_stationary_rearm() -> None:
    clock = Clock()
    ingress = SonicIntentIngress(clock=clock, max_age_s=0.5)
    ingress.generation = 10
    ingress.sequence = 19
    ingress.armed = True

    stopping = ingress.accept(stopping_intent(clock, generation=10, sequence=20))
    assert stopping.action == "brake"
    assert ingress.braking is True

    clock.value += 0.02
    motion = stopping_intent(clock, generation=10, sequence=21)
    motion.update(
        state="STARTING",
        mode="SLOW_WALK",
        velocity_x_mps=0.2,
        linear_speed_mps=0.2,
        motion_enabled=True,
        stop_requested=False,
    )
    rejected = ingress.accept(motion)
    assert rejected.action == "stop"
    assert "stationary IDLE" in str(rejected.reason)
    assert ingress.armed is False


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


@pytest.mark.parametrize("last_rearm_generation", [None, 9, 10])
def test_plant_ack_atomically_confirms_exact_driver_hold(
    last_rearm_generation: int | None,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.telemetry.update(
        nav2_state="idle",
        sonic_mode="SLOW_WALK",
        sonic_speed_mps=0.2,
        sonic_policy_speed_mps=0.2,
        navigation_task_active=True,
        last_cmd_vx=0.2,
        last_cmd_vy=0.1,
        last_cmd_wz=0.3,
        static_hold_active=False,
        motion_handoff_active=True,
        motion_handoff_ratio=1.0,
        plant_support_mode="idle",
        plant_support_requested_mode="idle",
        plant_support_confirmed_mode="idle",
        plant_support_stop_pending=True,
        plant_support_safety_latched=False,
        plant_support_stop_generation=None,
    )
    identity = enter_stationary(driver, clock)
    driver._normal_stop_last_confirmed_plant_epoch = 8
    payload = plant_request(
        identity=identity,
        plant=plant_success(
            epoch=9,
            last_rearm_generation=last_rearm_generation,
        ),
    )

    response = driver.accept_json(payload)

    assert response is not None
    parsed = parse_normal_stop_response(
        response,
        expected_request_id=PLANT_REQUEST_ID,
        expected_robot_id=ROBOT_ID,
        expected_driver_boot_id=DRIVER_BOOT_ID,
        expected_identity=identity,
        expected_control_generation=10,
        expected_gait_source_sequence=20,
        expected_phases=frozenset({"DRIVER_CONFIRMED"}),
    )
    assert parsed.ok is True
    assert parsed.normal_stop.safe_hold_confirmed is True
    assert parsed.plant_safety_stop_epoch == 9
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
    assert driver._normal_stop_plant_safety_stop_epoch == 9
    assert driver._normal_stop_last_confirmed_plant_epoch == 9
    assert driver._plant_support_mode == "safety_stop"
    assert driver._plant_support_safety_latched is True
    assert driver._plant_support_stop_pending is False
    assert driver._plant_support_stop_generation == 10
    assert driver._intent_ingress.support_stop_generation == 10
    assert driver._intent_ingress.armed is False
    assert driver._safe_hold_targets == driver._last_published_targets
    assert driver._static_hold_active is True
    assert driver.guard.command_zero is True
    assert driver._navigation_task_active is False
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    expected_telemetry = {
        "nav2_state": "safety_stop",
        "sonic_mode": "IDLE",
        "sonic_speed_mps": 0.0,
        "sonic_policy_speed_mps": 0.0,
        "navigation_task_active": False,
        "last_cmd_vx": 0.0,
        "last_cmd_vy": 0.0,
        "last_cmd_wz": 0.0,
        "static_hold_active": True,
        "motion_handoff_active": False,
        "motion_handoff_ratio": 0.0,
        "plant_support_mode": "safety_stop",
        "plant_support_requested_mode": "safety_stop",
        "plant_support_confirmed_mode": "safety_stop",
        "plant_support_stop_pending": False,
        "plant_support_safety_latched": True,
        "plant_support_stop_generation": 10,
        "sonic_reference_authority_state": "REVOKED",
        "sonic_reference_actuator_authority": False,
        "sonic_reference_authority_dds_target_sequence": None,
        "sonic_execution_feedback_pending": False,
    }
    assert {
        name: driver.telemetry.values[name] for name in expected_telemetry
    } == expected_telemetry
    assert driver.accept_json(payload) == response


def test_plant_retry_with_new_request_id_requires_the_same_epoch() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    first = driver.accept_json(plant_request(identity=identity))
    assert first is not None
    retry_id = UUID("10000000-0000-4000-8000-000000000021")

    retry = driver.accept_json(
        plant_request(identity=identity, request_id=retry_id)
    )

    assert retry is not None
    parsed = parse_normal_stop_response(
        retry,
        expected_request_id=retry_id,
        expected_identity=identity,
        expected_phases=frozenset({"DRIVER_CONFIRMED"}),
    )
    assert parsed.plant_safety_stop_epoch == 9
    before = driver._normal_stop.snapshot
    with pytest.raises(NormalStopWireError, match="retry epoch"):
        driver.accept_json(
            plant_request(
                identity=identity,
                request_id=UUID("10000000-0000-4000-8000-000000000022"),
                plant=plant_success(epoch=10),
            )
        )
    assert driver._normal_stop.snapshot is before
    assert driver._normal_stop_last_confirmed_plant_epoch == 9


@pytest.mark.parametrize("drift", ["rearmed", "target_changed"])
def test_new_plant_request_id_rechecks_the_current_local_hold(drift: str) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    original_payload = plant_request(identity=identity)
    original_response = driver.accept_json(original_payload)
    assert original_response is not None
    if drift == "rearmed":
        driver._plant_support_mode = "idle"
        driver._plant_support_safety_latched = False
        driver._plant_support_stop_generation = None
        driver._intent_ingress.support_stop_generation = None
    else:
        driver._safe_hold_targets = tuple(0.2 for _ in range(29))

    with pytest.raises(NormalStopWireError, match="no longer has"):
        driver.accept_json(
            plant_request(
                identity=identity,
                request_id=UUID("10000000-0000-4000-8000-000000000023"),
            )
        )

    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
    assert driver._fault_latched.is_set() is False
    assert driver.accept_json(original_payload) == original_response


@pytest.mark.parametrize(
    "evidence_case",
    ["failed", "stale_epoch", "future_rearm"],
)
def test_failed_or_dynamically_invalid_plant_evidence_latches_normal_failure(
    evidence_case: str,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    if evidence_case == "failed":
        plant = plant_failure()
    elif evidence_case == "stale_epoch":
        driver._normal_stop_last_confirmed_plant_epoch = 9
        plant = plant_success(epoch=9)
    else:
        plant = plant_success(epoch=10, last_rearm_generation=11)
    payload = plant_request(identity=identity, plant=plant)

    response = driver.accept_json(payload)

    assert response is not None
    parsed = parse_normal_stop_response(
        response,
        expected_request_id=PLANT_REQUEST_ID,
        expected_identity=identity,
        expected_phases=frozenset({"FAILED"}),
    )
    assert parsed.ok is False
    assert parsed.normal_stop.safe_hold_confirmed is False
    assert parsed.plant_safety_stop_epoch is None
    assert driver._normal_stop.snapshot.phase == "FAILED"
    assert driver._normal_stop_plant_safety_stop_epoch is None
    assert driver._plant_support_stop_pending is True
    assert driver._plant_support_safety_latched is False
    assert driver._fault_latched.is_set() is False
    assert driver._terminal_plant_halt_attempted is False
    assert driver.accept_json(payload) == response


def test_old_plant_identity_and_boot_are_rejected_before_state_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    old_identity = NormalStopIdentity(
        UUID("40000000-0000-4000-8000-000000000041"),
        identity.control_generation,
        identity.gait_source_sequence,
        identity.stop_epoch,
    )
    before = driver._normal_stop.snapshot

    def unexpected_plant(_identity: NormalStopIdentity):
        raise AssertionError("old Plant evidence reached the state machine")

    monkeypatch.setattr(driver._normal_stop, "confirm_plant", unexpected_plant)
    with pytest.raises(NormalStopWireError, match="transaction identity"):
        driver.accept_json(plant_request(identity=old_identity))
    with pytest.raises(NormalStopWireError, match="bridge boot"):
        driver.accept_json(
            plant_request(identity=identity, bridge_boot_id=NEW_BRIDGE_BOOT_ID)
        )
    with pytest.raises(NormalStopWireError, match="driver boot"):
        driver.accept_json(
            plant_request(
                identity=identity,
                driver_boot_id=UUID("30000000-0000-4000-8000-000000000030"),
            )
        )

    assert driver._normal_stop.snapshot is before
    assert driver._normal_stop_plant_safety_stop_epoch is None
    assert driver._normal_stop_last_confirmed_plant_epoch == 0
    assert driver._plant_support_stop_pending is True


@pytest.mark.parametrize(
    "missing_fact",
    [
        "safe_targets",
        "target_mismatch",
        "target_nonfinite",
        "static_hold",
        "zero_command",
        "navigation_task",
        "ingress_latch",
        "reference_authority",
        "reference_request",
    ],
)
def test_every_missing_local_hold_fact_prevents_driver_confirmation(
    missing_fact: str,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    if missing_fact == "safe_targets":
        driver._safe_hold_targets = None
    elif missing_fact == "target_mismatch":
        driver._safe_hold_targets = tuple(0.1 for _ in range(29))
    elif missing_fact == "target_nonfinite":
        driver._safe_hold_targets = (float("nan"),) + tuple(
            0.0 for _ in range(28)
        )
    elif missing_fact == "static_hold":
        driver._static_hold_active = False
    elif missing_fact == "zero_command":
        driver.guard._command_zero = False
    elif missing_fact == "navigation_task":
        driver._navigation_task_active = True
    elif missing_fact == "ingress_latch":
        driver._intent_ingress.latch_support_stop = lambda: 10  # type: ignore[method-assign]
    elif missing_fact == "reference_authority":
        driver._reference_authority.require(generation=10, source_sequence=20)
    else:
        driver._reference_request_pending = True
    terminal_reasons: list[str] = []

    def fake_terminal(reason: str, **_kwargs: object) -> None:
        terminal_reasons.append(reason)
        driver._fault_latched.set()

    driver._latch_terminal_fault = fake_terminal  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="driver hold fault"):
        driver.accept_json(plant_request(identity=identity))

    assert len(terminal_reasons) == 1
    assert driver._fault_latched.is_set() is True
    assert driver._normal_stop.snapshot.phase == "PLANT_CONFIRMED"
    assert driver._normal_stop.snapshot.safe_hold_confirmed is False
    assert driver._normal_stop_last_confirmed_plant_epoch == 0


@pytest.mark.parametrize("terminal", [False, True])
def test_late_plant_ack_after_shutdown_or_terminal_has_no_side_effect(
    terminal: bool,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    before = driver._normal_stop.snapshot
    support_before = (
        driver._plant_support_mode,
        driver._plant_support_stop_pending,
        driver._plant_support_safety_latched,
        driver._plant_support_stop_generation,
        driver._intent_ingress.support_stop_generation,
    )
    if terminal:
        driver._fault_latched.set()
        expected = "terminal fault"
    else:
        driver._stop.set()
        expected = "shutdown"

    with pytest.raises(RuntimeError, match=expected):
        driver.accept_json(plant_request(identity=identity))

    assert driver._normal_stop.snapshot is before
    assert (
        driver._plant_support_mode,
        driver._plant_support_stop_pending,
        driver._plant_support_safety_latched,
        driver._plant_support_stop_generation,
        driver._intent_ingress.support_stop_generation,
    ) == support_before
    assert driver._normal_stop_plant_safety_stop_epoch is None
    assert driver._normal_stop_last_confirmed_plant_epoch == 0


def test_shutdown_winning_plant_parse_prevents_final_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    payload = plant_request(identity=identity)
    before = driver._normal_stop.snapshot
    parsed = threading.Event()
    release_parse = threading.Event()
    from hide_mujoco_sonic import sonic_driver as sonic_driver_module

    original_parse = sonic_driver_module.parse_normal_stop_request

    def gated_parse(*args: object, **kwargs: object):
        result = original_parse(*args, **kwargs)  # type: ignore[arg-type]
        parsed.set()
        assert release_parse.wait(timeout=2.0)
        return result

    monkeypatch.setattr(sonic_driver_module, "parse_normal_stop_request", gated_parse)
    errors: list[Exception] = []

    def invoke_handler() -> None:
        try:
            driver.accept_json(payload)
        except Exception as exc:
            errors.append(exc)

    handler = threading.Thread(target=invoke_handler)
    handler.start()
    assert parsed.wait(timeout=2.0)
    shutdown = threading.Thread(target=driver.stop)
    shutdown.start()
    assert driver._stop.wait(timeout=2.0)
    release_parse.set()
    handler.join(timeout=2.0)
    shutdown.join(timeout=2.0)

    assert len(errors) == 1
    assert "shutdown" in str(errors[0])
    assert driver._normal_stop.snapshot is before
    assert driver._normal_stop_plant_safety_stop_epoch is None
    assert driver._normal_stop_last_confirmed_plant_epoch == 0
    assert driver._plant_support_stop_pending is True


def test_plant_commit_winning_linearization_precedes_shutdown() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    payload = plant_request(identity=identity)
    entered_commit = threading.Event()
    release_commit = threading.Event()
    original_confirm = driver._normal_stop.confirm_plant

    def gated_confirm(value: NormalStopIdentity):
        snapshot = original_confirm(value)
        entered_commit.set()
        assert release_commit.wait(timeout=2.0)
        return snapshot

    driver._normal_stop.confirm_plant = gated_confirm  # type: ignore[method-assign]
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
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
    assert driver._normal_stop_last_confirmed_plant_epoch == 9
    assert driver._stop.is_set() is True


def test_late_moving_tick_cannot_overwrite_confirmed_static_hold() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    decision_ready = threading.Event()
    release_old_tick = threading.Event()
    outcomes: list[tuple[bool, bool, bool]] = []

    def finish_old_tick() -> None:
        # The tick prepared a moving decision before normal-stop begin.
        prepared_static_decision = False
        decision_ready.set()
        assert release_old_tick.wait(timeout=2.0)
        support_ready = driver._set_plant_support_mode("locomotion")
        with driver._support_lock:
            expected_static_boundary = (
                driver._rejected_locomotion_is_static_boundary_locked()
            )
            effective_static_hold = driver._commit_static_hold_decision_locked(
                prepared_static_decision
            )
        outcomes.append(
            (
                support_ready,
                expected_static_boundary,
                effective_static_hold,
            )
        )

    old_tick = threading.Thread(target=finish_old_tick)
    old_tick.start()
    assert decision_ready.wait(timeout=2.0)
    identity = enter_stationary(driver, clock)
    original = driver.accept_json(plant_request(identity=identity))
    assert original is not None
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"

    release_old_tick.set()
    old_tick.join(timeout=2.0)

    assert old_tick.is_alive() is False
    assert outcomes == [(False, True, True)]
    assert driver._static_hold_active is True
    retry_id = UUID("10000000-0000-4000-8000-000000000024")
    retry = driver.accept_json(
        plant_request(identity=identity, request_id=retry_id)
    )
    assert retry is not None
    assert retry["normal_stop"]["safe_hold_confirmed"] is True  # type: ignore[index]


def test_confirmed_stop_releases_static_fence_only_after_rearm_and_reference() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"

    clock.value += 0.01
    arm_intent = stationary_intent(clock, generation=11, sequence=1)
    arm_intent.update(
        state="IDLE",
        stop_requested=False,
        control_arm=True,
        reason="explicit control rearm",
    )
    with driver._support_lock:
        decision = driver._intent_ingress.accept(
            arm_intent,
            defer_control_arm=True,
        )
        assert decision.arm_acknowledged is True
        driver._commit_staged_plant_support_rearm_locked(11)
        # Rearm alone retains the exact actuator hold until the matching
        # reference is installed and primed.
        assert driver._normal_stop_static_boundary_locked() is True
        assert driver._commit_static_hold_decision_locked(False) is True

        driver._reference_authority.require(generation=11, source_sequence=1)
        driver._reference_authority.mark_installed(
            generation=11,
            source_sequence=1,
            planner_intent_generation=1,
            reference_epoch=1,
        )
        driver._reference_authority.prime(
            generation=11,
            source_sequence=1,
            reference_epoch=1,
        )
        # This is the exact release performed by
        # _progress_reference_transaction after the measured rebase.
        driver._safe_hold_targets = None
        assert driver._normal_stop_static_boundary_locked() is False
        assert driver._commit_static_hold_decision_locked(False) is False
        assert driver._rejected_locomotion_is_static_boundary_locked() is False

    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
    assert driver._static_hold_active is False


def test_late_rearm_publisher_cannot_overwrite_a_newer_confirmed_stop() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    first_identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=first_identity))

    clock.value += 0.01
    commit_rearm_without_publication(driver, clock)

    resets: list[str] = []
    driver._arm_guard = SimpleNamespace(
        reset=lambda: resets.append("obsolete rearm reset"),
        anchor=lambda _targets: None,
    )
    clock.value += 0.01
    driver.accept_json(
        begin_request(
            clock,
            request_id=UUID("10000000-0000-4000-8000-000000000030"),
            generation=12,
            sequence=30,
        )
    )
    second_identity = driver._normal_stop.snapshot.identity
    assert second_identity is not None
    driver.accept_json(
        stationary_request(
            clock,
            identity=second_identity,
            request_id=UUID("10000000-0000-4000-8000-000000000031"),
        )
    )
    driver.accept_json(
        plant_request(
            identity=second_identity,
            request_id=UUID("10000000-0000-4000-8000-000000000032"),
            plant=plant_success(epoch=10, last_rearm_generation=11),
        )
    )
    telemetry_before = dict(driver.telemetry.values)

    assert driver._publish_committed_plant_rearm(11) is False
    assert resets == []
    assert driver.telemetry.values == telemetry_before
    assert driver.telemetry.values["nav2_state"] == "safety_stop"
    assert driver.telemetry.values["plant_support_mode"] == "safety_stop"
    assert driver.telemetry.values["plant_support_safety_latched"] is True


def test_terminal_claim_before_rearm_projection_never_writes_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    clock.value += 0.01
    commit_rearm_without_publication(driver, clock)
    resets: list[str] = []
    driver._arm_guard = SimpleNamespace(
        reset=lambda: resets.append("obsolete rearm reset"),
        anchor=lambda _targets: None,
    )

    terminal_claimed = threading.Barrier(2)

    def claimed_terminal_plant_halt(*_args: object, **_kwargs: object) -> None:
        terminal_claimed.wait(timeout=2.0)

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        claimed_terminal_plant_halt,
    )
    projected_states: list[str] = []
    original_update = driver.telemetry.update

    def record_update(**values: object) -> None:
        nav2_state = values.get("nav2_state")
        if isinstance(nav2_state, str):
            projected_states.append(nav2_state)
        original_update(**values)

    driver.telemetry.update = record_update  # type: ignore[method-assign]
    terminal_errors: list[Exception] = []
    terminal_thread = threading.Thread(
        target=lambda: _capture_exception(
            terminal_errors,
            driver._latch_terminal_fault,
            "terminal claim won before rearm projection",
        )
    )
    terminal_thread.start()
    terminal_claimed.wait(timeout=2.0)
    assert driver._fault_latched.is_set() is True

    assert driver._publish_committed_plant_rearm(11) is False
    terminal_thread.join(timeout=2.0)

    assert terminal_thread.is_alive() is False
    assert terminal_errors == []
    assert resets == []
    assert "idle" not in projected_states
    assert projected_states[-1] == "fault_latched"
    assert driver.telemetry.values["nav2_state"] == "fault_latched"


def test_rearm_wins_before_late_rejected_tick_without_revoking_new_authority() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    classified = threading.Event()
    release_old_tick = threading.Event()
    outcomes: list[bool] = []
    publishes: list[object] = []

    def finish_old_tick() -> None:
        assert driver._set_plant_support_mode("locomotion") is False
        with driver._support_lock:
            assert driver._rejected_locomotion_is_static_boundary_locked()
            token = driver._support_transition_token_locked()
        classified.set()
        assert release_old_tick.wait(timeout=2.0)
        with driver._support_lock:
            committed = driver._commit_rejected_locomotion_tick_locked(token)
            if committed:
                publishes.append("obsolete frozen publish")
        outcomes.append(committed)

    old_tick = threading.Thread(target=finish_old_tick)
    old_tick.start()
    assert classified.wait(timeout=2.0)

    clock.value += 0.01
    arm_intent = stationary_intent(clock, generation=11, sequence=1)
    arm_intent.update(
        state="IDLE",
        stop_requested=False,
        control_arm=True,
        reason="explicit control rearm",
    )
    with driver._support_lock:
        decision = driver._intent_ingress.accept(
            arm_intent,
            defer_control_arm=True,
        )
        assert decision.arm_acknowledged is True
        driver._commit_staged_plant_support_rearm_locked(11)
        driver._reference_authority.require(generation=11, source_sequence=1)
        driver._reference_authority.mark_installed(
            generation=11,
            source_sequence=1,
            planner_intent_generation=1,
            reference_epoch=1,
        )
        driver._reference_authority.prime(
            generation=11,
            source_sequence=1,
            reference_epoch=1,
        )
        driver._safe_hold_targets = None

    release_old_tick.set()
    old_tick.join(timeout=2.0)

    assert old_tick.is_alive() is False
    assert outcomes == [False]
    assert publishes == []
    assert driver._intent_ingress.generation == 11
    assert driver._intent_ingress.armed is True
    assert driver._reference_authority.snapshot.phase == "PRIMED"
    assert driver.guard.stops == ["normal_stop_begin"]
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"


def test_late_tick_cannot_release_plant_before_matching_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    decision_ready = threading.Event()
    release_old_tick = threading.Event()
    plant_calls: list[str] = []
    outcomes: list[bool] = []

    def plant_support(
        _host: str,
        _port: int,
        mode: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        plant_calls.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        plant_support,
    )

    def finish_old_tick() -> None:
        # This locomotion decision predates the entire stop transaction.
        decision_ready.set()
        assert release_old_tick.wait(timeout=2.0)
        outcomes.append(driver._set_plant_support_mode("locomotion"))

    old_tick = threading.Thread(target=finish_old_tick)
    old_tick.start()
    assert decision_ready.wait(timeout=2.0)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))

    clock.value += 0.01
    arm_intent = stationary_intent(clock, generation=11, sequence=1)
    arm_intent.update(
        state="IDLE",
        stop_requested=False,
        control_arm=True,
        reason="explicit control rearm",
    )
    with driver._support_lock:
        decision = driver._intent_ingress.accept(
            arm_intent,
            defer_control_arm=True,
        )
        assert decision.arm_acknowledged is True
        driver._commit_staged_plant_support_rearm_locked(11)
        assert driver._safe_hold_targets is not None

    release_old_tick.set()
    old_tick.join(timeout=2.0)

    assert old_tick.is_alive() is False
    assert outcomes == [False]
    assert plant_calls == []
    assert driver._plant_support_mode == "idle"

    with driver._support_lock:
        driver._reference_authority.require(generation=11, source_sequence=1)
        driver._reference_authority.mark_installed(
            generation=11,
            source_sequence=1,
            planner_intent_generation=1,
            reference_epoch=1,
        )
        driver._reference_authority.prime(
            generation=11,
            source_sequence=1,
            reference_epoch=1,
        )
        driver._safe_hold_targets = None

    assert driver._set_plant_support_mode("locomotion") is True
    assert plant_calls == ["locomotion"]
    assert driver._plant_support_mode == "locomotion"


def test_rejected_tick_publish_does_not_invert_control_and_support_locks() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver._control_condition = threading.Condition(threading.RLock())
    driver.accept_json(begin_request(clock))
    with driver._support_lock:
        token = driver._support_transition_token_locked()
    control_holds_condition = threading.Event()
    publish_entered = threading.Event()
    support_acquired = threading.Event()
    publish_finished = threading.Event()
    results: list[bool] = []

    def fake_publish(_targets: tuple[float, ...]) -> None:
        publish_entered.set()
        with driver._control_condition:
            publish_finished.set()

    driver._publish_position_hold = fake_publish  # type: ignore[method-assign]

    def control_arm_lock_order() -> None:
        with driver._control_condition:
            control_holds_condition.set()
            assert publish_entered.wait(timeout=2.0)
            acquired = driver._support_lock.acquire(timeout=1.0)
            if acquired:
                try:
                    support_acquired.set()
                finally:
                    driver._support_lock.release()

    control_thread = threading.Thread(target=control_arm_lock_order)
    control_thread.start()
    assert control_holds_condition.wait(timeout=2.0)
    publish_thread = threading.Thread(
        target=lambda: results.append(
            driver._publish_rejected_locomotion_tick_if_current(
                object(),
                token,
            )
        )
    )
    publish_thread.start()
    control_thread.join(timeout=2.0)
    publish_thread.join(timeout=2.0)

    assert control_thread.is_alive() is False
    assert publish_thread.is_alive() is False
    assert support_acquired.is_set() is True
    assert publish_finished.is_set() is True
    assert results == [True]


def test_late_locomotion_support_rejection_is_expected_during_braking() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    decision_ready = threading.Event()
    release_old_tick = threading.Event()
    outcomes: list[tuple[bool, bool]] = []

    def finish_old_tick() -> None:
        decision_ready.set()
        assert release_old_tick.wait(timeout=2.0)
        support_ready = driver._set_plant_support_mode("locomotion")
        with driver._support_lock:
            expected_static_boundary = (
                driver._rejected_locomotion_is_static_boundary_locked()
            )
        outcomes.append((support_ready, expected_static_boundary))

    old_tick = threading.Thread(target=finish_old_tick)
    old_tick.start()
    assert decision_ready.wait(timeout=2.0)
    driver.accept_json(begin_request(clock))
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._plant_support_stop_pending is False
    assert driver._plant_support_safety_latched is False

    release_old_tick.set()
    old_tick.join(timeout=2.0)

    assert old_tick.is_alive() is False
    assert outcomes == [(False, True)]
    assert driver._fault_latched.is_set() is False
    assert driver._normal_stop.snapshot.phase == "BRAKING"


@pytest.mark.parametrize(
    "phase",
    ["BRAKING", "STATIONARY_ACK", "PLANT_CONFIRMED", "FAILED"],
)
@pytest.mark.parametrize("kind", ["stop", "idle", "motion", "control_arm"])
def test_normal_stop_phase_barrier_precedes_generic_ingress_mutation(
    phase: str,
    kind: str,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    enter_normal_phase(driver, clock, phase)
    snapshot = driver._normal_stop.snapshot
    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
        driver._intent_ingress.support_stop_generation,
    )
    stops_before = list(driver.guard.stops)
    pending_before = driver._plant_support_stop_pending
    payload = generic_gait_intent(
        clock,
        generation=10,
        sequence=driver._intent_ingress.sequence + 1,
        kind=kind,
    )

    if kind == "control_arm":
        configure_control_arm_health(driver, clock)
        with pytest.raises(RuntimeError, match="current exact DRIVER_CONFIRMED"):
            driver.accept_json(payload)
    elif kind == "motion":
        with pytest.raises(RuntimeError, match="rejects new gait/control"):
            driver.accept_json(payload)
    else:
        assert driver.accept_json(payload) is None

    assert driver._normal_stop.snapshot is snapshot
    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
        driver._intent_ingress.support_stop_generation,
    ) == ingress_before
    assert driver.guard.stops == stops_before
    assert driver._plant_support_stop_pending is pending_before


@pytest.mark.parametrize(
    "phase",
    ["BRAKING", "STATIONARY_ACK", "PLANT_CONFIRMED", "FAILED"],
)
def test_real_gait_fault_bypasses_normal_barrier_into_recoverable_hold(
    phase: str,
    monkeypatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    enter_normal_phase(driver, clock, phase)
    snapshot = driver._normal_stop.snapshot
    recoverable_reasons: list[str] = []
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_motion_fault",
        lambda _host, _port, reason, **_kwargs: (
            recoverable_reasons.append(reason)
            or {
                "mode": "safety_stop",
                "safetyStopLatched": True,
                "fault": None,
                "motionFault": reason,
            }
        ),
    )
    sequence = driver._intent_ingress.sequence + 1
    payload = generic_gait_intent(
        clock,
        generation=10,
        sequence=sequence,
        kind="fault",
    )

    assert driver.accept_json(payload) is None

    assert recoverable_reasons == ["deterministic terminal gait fault"]
    assert driver._fault_latched.is_set() is True
    assert driver._recoverable_motion_fault is True
    assert driver._stop.is_set() is False
    assert driver._normal_stop.snapshot is snapshot
    assert driver._intent_ingress.sequence == sequence
    assert driver.guard.stops[-1] == "motion_fault"


def test_control_arm_requires_current_driver_confirmed_identity_before_ingress() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    configure_control_arm_health(driver, clock)
    payload = generic_gait_intent(
        clock,
        generation=1,
        sequence=1,
        kind="control_arm",
    )

    with pytest.raises(RuntimeError, match="current exact DRIVER_CONFIRMED"):
        driver.accept_json(payload)

    assert driver._normal_stop.snapshot.phase == "NONE"
    assert driver._intent_ingress.generation is None
    assert driver._intent_ingress.sequence == 0


def test_begin_winning_after_generic_candidate_read_blocks_ingress_mutation() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    candidate_read = threading.Barrier(2)
    release_candidate = threading.Barrier(2)
    errors: list[Exception] = []
    payload = generic_gait_intent(
        clock,
        generation=11,
        sequence=23,
        kind="motion",
    )

    def deliver_candidate() -> None:
        assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
        candidate_read.wait(timeout=2.0)
        release_candidate.wait(timeout=2.0)
        try:
            driver.accept_json(payload)
        except Exception as exc:
            errors.append(exc)

    generic_thread = threading.Thread(target=deliver_candidate)
    generic_thread.start()
    candidate_read.wait(timeout=2.0)
    clock.value += 0.01
    driver.accept_json(
        begin_request(
            clock,
            request_id=UUID("10000000-0000-4000-8000-000000000031"),
            generation=11,
            sequence=22,
        )
    )
    ingress_after_begin = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
        driver._intent_ingress.support_stop_generation,
    )
    release_candidate.wait(timeout=2.0)
    generic_thread.join(timeout=2.0)

    assert generic_thread.is_alive() is False
    assert len(errors) == 1
    assert "rejects new gait/control authority" in str(errors[0])
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
        driver._intent_ingress.support_stop_generation,
    ) == ingress_after_begin


def test_terminal_winning_control_arm_precheck_prevents_staging_and_plant_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    configure_control_arm_health(driver, clock)
    terminal_lock = ObservedLock()
    driver._terminal_fault_lock = terminal_lock  # type: ignore[assignment]
    rearm_calls: list[int] = []
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        lambda _host, _port, generation, **_kwargs: rearm_calls.append(generation),
    )
    payload = generic_gait_intent(
        clock,
        generation=11,
        sequence=1,
        kind="control_arm",
    )
    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
        driver._intent_ingress.support_stop_generation,
    )
    errors: list[Exception] = []

    terminal_lock.acquire()
    terminal_lock.acquire_attempted.clear()
    arm_thread = threading.Thread(
        target=lambda: _capture_exception(errors, driver.accept_json, payload)
    )
    arm_thread.start()
    assert terminal_lock.acquire_attempted.wait(timeout=2.0)
    driver._terminal_fault_reason = "terminal precheck won"
    driver._fault_latched.set()
    terminal_lock.release()
    arm_thread.join(timeout=2.0)

    assert arm_thread.is_alive() is False
    assert len(errors) == 1
    assert "terminal fault is latched" in str(errors[0])
    assert rearm_calls == []
    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
        driver._intent_ingress.support_stop_generation,
    ) == ingress_before


def test_terminal_winning_control_arm_postcheck_relatches_without_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    configure_control_arm_health(driver, clock)
    rearm_entered = threading.Event()
    release_rearm = threading.Event()
    support_calls: list[str] = []

    def blocked_rearm(*_args: object, **_kwargs: object) -> None:
        rearm_entered.set()
        assert release_rearm.wait(timeout=2.0)

    def confirm_support(
        _host: str,
        _port: int,
        mode: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        support_calls.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        blocked_rearm,
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        confirm_support,
    )
    payload = generic_gait_intent(
        clock,
        generation=11,
        sequence=1,
        kind="control_arm",
    )
    responses: list[dict[str, object] | None] = []
    errors: list[Exception] = []

    def arm() -> None:
        try:
            responses.append(driver.accept_json(payload))
        except Exception as exc:
            errors.append(exc)

    arm_thread = threading.Thread(target=arm)
    arm_thread.start()
    assert rearm_entered.wait(timeout=2.0)
    with driver._terminal_fault_lock:
        driver._terminal_plant_halt_attempted = True
        driver._terminal_fault_reason = "terminal postcheck won"
        driver._fault_latched.set()
    release_rearm.set()
    arm_thread.join(timeout=2.0)

    assert arm_thread.is_alive() is False
    assert responses == []
    assert len(errors) == 1
    assert "post-rearm health failed" in str(errors[0])
    assert "terminal fault is latched" in str(errors[0])
    assert support_calls == ["safety_stop"]
    assert driver._intent_ingress.armed is False
    assert driver._intent_ingress.staged_control_arm_generation is None
    assert driver._plant_support_mode == "safety_stop"
    assert driver._plant_support_safety_latched is True


def test_control_arm_commit_winning_precedes_terminal_revoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    configure_control_arm_health(driver, clock)
    terminal_lock = ObservedLock()
    driver._terminal_fault_lock = terminal_lock  # type: ignore[assignment]
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        lambda *_args, **_kwargs: None,
    )
    plant_faults: list[str] = []
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        lambda _host, _port, reason, **_kwargs: plant_faults.append(reason),
    )
    commit_entered = threading.Event()
    release_commit = threading.Event()
    original_commit = driver._commit_staged_plant_support_rearm_locked

    def blocked_commit(generation: int) -> None:
        commit_entered.set()
        assert release_commit.wait(timeout=2.0)
        original_commit(generation)

    driver._commit_staged_plant_support_rearm_locked = blocked_commit  # type: ignore[method-assign]
    payload = generic_gait_intent(
        clock,
        generation=11,
        sequence=1,
        kind="control_arm",
    )
    responses: list[dict[str, object] | None] = []
    arm_errors: list[Exception] = []
    terminal_errors: list[Exception] = []

    def arm_with_errors() -> None:
        try:
            responses.append(driver.accept_json(payload))
        except Exception as exc:
            arm_errors.append(exc)

    arm_thread = threading.Thread(target=arm_with_errors)
    arm_thread.start()
    assert commit_entered.wait(timeout=2.0)
    terminal_lock.acquire_attempted.clear()
    terminal_thread = threading.Thread(
        target=lambda: _capture_exception(
            terminal_errors,
            driver._latch_terminal_fault,
            "terminal after committed rearm",
        )
    )
    terminal_thread.start()
    assert terminal_lock.acquire_attempted.wait(timeout=2.0)
    release_commit.set()
    arm_thread.join(timeout=2.0)
    terminal_thread.join(timeout=2.0)

    assert arm_thread.is_alive() is False
    assert terminal_thread.is_alive() is False
    assert arm_errors == []
    assert terminal_errors == []
    assert responses == [{"ok": True, "mode": "idle", "generation": 11}]
    assert driver._fault_latched.is_set() is True
    assert driver._terminal_plant_halt_attempted is True
    assert plant_faults == ["terminal after committed rearm"]
    assert driver._safe_hold_targets == driver._last_published_targets
    assert driver.guard.command_zero is True
    assert driver._reference_authority.snapshot.phase == "REVOKED"


def test_control_arm_succeeds_only_from_exact_driver_confirmed_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    configure_control_arm_health(driver, clock)
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        lambda *_args, **_kwargs: None,
    )
    payload = generic_gait_intent(
        clock,
        generation=11,
        sequence=1,
        kind="control_arm",
    )

    response = driver.accept_json(payload)

    assert response == {"ok": True, "mode": "idle", "generation": 11}
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
    assert driver._normal_stop_plant_safety_stop_epoch == 9
    assert driver._intent_ingress.generation == 11
    assert driver._intent_ingress.armed is True
    assert driver._intent_ingress.support_stop_generation is None
    assert driver._plant_support_mode == "idle"
    assert driver._plant_support_safety_latched is False


def test_normal_begin_racing_control_arm_postcheck_relatches_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    driver.accept_json(plant_request(identity=identity))
    configure_control_arm_health(driver, clock)
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        lambda _host, _port, mode, **_kwargs: {"mode": mode},
    )
    original_rearm = driver._request_staged_plant_support_rearm

    def rearm_then_begin(generation: int) -> bool:
        assert original_rearm(generation) is True
        clock.value += 0.01
        driver.accept_json(
            begin_request(
                clock,
                request_id=UUID("10000000-0000-4000-8000-000000000030"),
                generation=12,
                sequence=22,
            )
        )
        return True

    driver._request_staged_plant_support_rearm = rearm_then_begin  # type: ignore[method-assign]
    payload = generic_gait_intent(
        clock,
        generation=11,
        sequence=1,
        kind="control_arm",
    )

    with pytest.raises(RuntimeError, match="post-rearm health failed"):
        driver.accept_json(payload)

    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._normal_stop.snapshot.control_generation == 12
    assert driver._normal_stop_plant_safety_stop_epoch is None
    assert driver._intent_ingress.generation == 12
    assert driver._intent_ingress.armed is False
    assert driver._intent_ingress.support_stop_generation == 12
    assert driver._plant_support_mode == "safety_stop"
    assert driver._plant_support_safety_latched is True
    assert driver._plant_support_stop_pending is False
    assert driver._fault_latched.is_set() is False


def test_legacy_safe_stop_does_not_start_plant_pending_during_braking() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.accept_json(begin_request(clock))
    before = driver._normal_stop.snapshot
    assert driver._plant_support_stop_pending is False

    assert driver.accept_json(
        {
            "type": "stop",
            "source": "robot_agent_safe_stop",
            "reason": "obsolete outer safe-stop path",
        }
    ) is None

    assert driver._normal_stop.snapshot is before
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._plant_support_stop_pending is False


def test_legacy_plant_notification_only_strengthens_without_phase_progress() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver.accept_json(begin_request(clock))
    before = driver._normal_stop.snapshot

    response = driver.accept_json(
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "mode": "safety_stop",
            "source": "navigation_guard",
            "reason": "independent safety strengthening",
        }
    )

    assert response == {"ok": True, "mode": "safety_stop", "stopped_generation": 10}
    assert driver._normal_stop.snapshot is before
    assert driver._normal_stop.snapshot.phase == "BRAKING"
    assert driver._normal_stop_plant_safety_stop_epoch is None
    assert driver._normal_stop_last_confirmed_plant_epoch == 0
    assert driver._plant_support_mode == "safety_stop"
    assert driver._plant_support_safety_latched is True


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


def test_real_driver_failed_response_crosses_negative_truth_transport() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    identity = enter_stationary(driver, clock)
    payload = plant_request(
        identity=identity,
        plant=plant_failure("Plant confirmation was unavailable"),
    )
    server = _bind_command_server("127.0.0.1", 0)
    port = int(server.getsockname()[1])
    failures: list[Exception] = []

    def serve() -> None:
        try:
            connection, _address = server.accept()
            with connection:
                _handle_command_connection(driver, connection)
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        response = request_json(
            "127.0.0.1",
            port,
            payload,
            timeout=1.0,
            require_ok=False,
        )
    finally:
        worker.join(timeout=2.0)
        server.close()

    assert worker.is_alive() is False
    assert failures == []
    parsed = parse_normal_stop_response(
        response,
        expected_request_id=PLANT_REQUEST_ID,
        expected_driver_boot_id=DRIVER_BOOT_ID,
        expected_identity=identity,
        expected_phases=frozenset({"FAILED"}),
    )
    assert parsed.ok is False
    assert driver._normal_stop.snapshot.phase == "FAILED"
