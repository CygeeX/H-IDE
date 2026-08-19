from __future__ import annotations

import math
import threading
from types import SimpleNamespace
from uuid import UUID

import pytest

from hide_mujoco_sonic.gait_adapter import (
    BaseObservation,
    GaitAdapter,
    GaitAdapterConfig,
    Nav2Twist,
)
from hide_mujoco_sonic.ros_bridge import (
    _intent_payload,
    _ros_stamp_in_monotonic_domain,
)
from hide_mujoco_sonic.motion_fault import SonicHealth, sonic_health_probe
from hide_mujoco_sonic.planner_evidence import (
    PlannerReferenceDeadlineExceeded,
    PlannerReferenceEvidenceGate,
)
from hide_mujoco_sonic.reference_authority import ReferenceAuthorityState
from hide_mujoco_sonic.safety import (
    MovementIntent,
    PlantControlClock,
    PlantTickGuard,
    ReplanIntentGate,
    StaticJointHandoff,
)
from hide_mujoco_sonic.sonic_driver import (
    SonicDriver,
    SonicIntentIngress,
    _bind_command_server,
    _handle_command_connection,
)
from hide_mujoco_sonic.transport import request_json, send_json


class Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def observation(
    clock: Clock,
    *,
    generation: int = 1,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
) -> BaseObservation:
    return BaseObservation(
        generation=generation,
        timestamp_s=clock.value,
        x=x,
        y=y,
        yaw=yaw,
        roll=roll,
        pitch=0.0,
        vx_body=vx,
        vy_body=vy,
        wz=wz,
    )


def twist(
    clock: Clock,
    *,
    generation: int = 1,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
) -> Nav2Twist:
    return Nav2Twist(generation, clock.value, vx, vy, wz, source="nav2")


def runtime(clock: Clock, *, generation: int = 1):
    config = GaitAdapterConfig(
        command_timeout_s=0.25,
        observation_timeout_s=0.30,
        max_linear_mps=0.25,
        max_angular_rps=0.25,
        linear_deadband_mps=0.10,
        angular_deadband_rps=0.05,
    )
    adapter = GaitAdapter(config, clock=clock)
    ingress = SonicIntentIngress(clock=clock, max_age_s=0.50)
    adapter.advance_generation(generation, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=generation), now=clock.value
    )
    idle = adapter.arm(generation, now=clock.value)
    decision = ingress.accept(_intent_payload(idle, measured_heading_yaw=0.0))
    assert decision.action == "stop"
    assert ingress.armed is True
    return adapter, ingress


def decide(ingress: SonicIntentIngress, intent):
    return ingress.accept(_intent_payload(intent, measured_heading_yaw=0.0))


def test_ros_stamp_preserves_capture_age_in_monotonic_domain() -> None:
    assert _ros_stamp_in_monotonic_domain(
        seconds=1_000,
        nanoseconds=250_000_000,
        wall_now_s=1_000.50,
        monotonic_now_s=42.0,
    ) == pytest.approx(41.75)

    with pytest.raises(ValueError, match="future"):
        _ros_stamp_in_monotonic_domain(
            seconds=1_001,
            nanoseconds=0,
            wall_now_s=1_000.0,
            monotonic_now_s=42.0,
        )
    with pytest.raises(ValueError, match="invalid"):
        _ros_stamp_in_monotonic_domain(
            seconds=1_000,
            nanoseconds=1_000_000_000,
            wall_now_s=1_000.0,
            monotonic_now_s=42.0,
        )


def test_replay_straight_turn_and_zero_stops_before_mapping() -> None:
    clock = Clock()
    adapter, ingress = runtime(clock)

    clock.value += 0.01
    starting = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    motion = decide(ingress, starting)
    assert (motion.action, motion.vx, motion.vy, motion.wz) == (
        "motion",
        0.20,
        0.0,
        0.0,
    )

    clock.value += 0.01
    walking = adapter.ingest_observation(
        observation(clock, vx=0.06), now=clock.value
    )
    assert decide(ingress, walking).action == "motion"

    clock.value += 0.01
    stopping = adapter.ingest_twist(twist(clock), now=clock.value)
    payload = _intent_payload(stopping, measured_heading_yaw=0.0)
    # A stop envelope is authoritative even if unused motion fields are noisy;
    # they must never reach the model-side command guard.
    payload["velocity_x_mps"] = 999.0
    payload["yaw_rate_rps"] = -999.0
    stopped = ingress.accept(payload)
    assert stopped.action == "stop"
    assert (stopped.vx, stopped.vy, stopped.wz) == (0.0, 0.0, 0.0)
    assert ingress.armed is False

    clock.value += 0.01
    adapter.ingest_observation(observation(clock), now=clock.value)
    clock.value += 0.16
    idle = adapter.ingest_observation(observation(clock), now=clock.value)
    assert idle.state == "IDLE"
    assert decide(ingress, idle).action == "stop"
    assert ingress.armed is True

    clock.value += 0.01
    turning = adapter.ingest_twist(twist(clock, wz=-0.20), now=clock.value)
    decision = decide(ingress, turning)
    assert decision.action == "motion"
    assert decision.vx == decision.vy == 0.0
    assert decision.wz == pytest.approx(-0.20)
    assert turning.head_yaw_rad == turning.head_pitch_rad == 0.0


@pytest.mark.parametrize("terminal", ["cancel", "stale", "reverse", "fall"])
def test_replay_terminal_events_never_reach_model_motion(terminal: str) -> None:
    clock = Clock()
    adapter, ingress = runtime(clock)
    clock.value += 0.01
    decide(
        ingress,
        adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value),
    )
    clock.value += 0.01
    decide(
        ingress,
        adapter.ingest_observation(
            observation(clock, vx=0.06), now=clock.value
        ),
    )

    if terminal == "cancel":
        clock.value += 0.01
        intent = adapter.cancel(1, clock.value, now=clock.value)
    elif terminal == "stale":
        # Keep odometry fresh after the Nav2 sample has expired.
        clock.value += 0.26
        adapter.ingest_observation(
            observation(clock, vx=0.02), now=clock.value
        )
        intent = adapter.tick(now=clock.value)
    elif terminal == "reverse":
        clock.value += 0.01
        adapter.ingest_observation(
            observation(clock, vx=-0.08, x=-0.02), now=clock.value
        )
        clock.value += 0.06
        intent = adapter.ingest_observation(
            observation(clock, vx=-0.08, x=-0.07), now=clock.value
        )
    else:
        clock.value += 0.01
        intent = adapter.ingest_observation(
            observation(clock, roll=0.70), now=clock.value
        )

    decision = decide(ingress, intent)
    assert decision.action == "stop"
    assert decision.vx == decision.vy == decision.wz == 0.0
    assert ingress.armed is False


def test_bridge_restart_generation_requires_new_stationary_ack() -> None:
    clock = Clock()
    adapter, ingress = runtime(clock, generation=100)
    clock.value += 0.01
    assert decide(
        ingress,
        adapter.ingest_twist(twist(clock, generation=100, vx=0.20), now=clock.value),
    ).action == "motion"

    # A new bridge process first selects a new generation.  Its initial
    # DISARMED envelope stops the old generation and cannot itself arm motion.
    new_adapter = GaitAdapter(clock=clock)
    clock.value += 0.01
    selected = new_adapter.advance_generation(200, now=clock.value)
    assert decide(ingress, selected).action == "stop"
    assert ingress.armed is False

    # Even a structurally valid motion envelope is rejected until IDLE ack.
    malicious = _intent_payload(
        adapter.ingest_twist(
            twist(clock, generation=200, vx=0.20), now=clock.value
        ),
        measured_heading_yaw=0.0,
    )
    malicious.update(
        generation=200,
        sequence=int(selected.sequence) + 1,
        state="STARTING",
        mode="SLOW_WALK",
        motion_enabled=True,
        stop_requested=False,
        stop_acknowledged=False,
        velocity_x_mps=0.20,
        velocity_y_mps=0.0,
        linear_speed_mps=0.20,
        yaw_rate_rps=0.0,
    )
    assert ingress.accept(malicious).action == "stop"

    clock.value += 0.01
    new_adapter.ingest_observation(
        observation(clock, generation=200), now=clock.value
    )
    idle = new_adapter.arm(200, now=clock.value)
    assert decide(ingress, idle).action == "stop"
    assert ingress.armed is True

    old = _intent_payload(selected, measured_heading_yaw=0.0)
    old.update(generation=100, sequence=999)
    assert ingress.accept(old).action == "ignore"


class FakeGuard:
    def __init__(self) -> None:
        self.accepted: list[tuple[float, float, float, str]] = []
        self.stops: list[str] = []
        self._command_zero = True

    def accept(self, vx, vy, wz, *, source):
        self.accepted.append((vx, vy, wz, source))
        self._command_zero = not any(abs(value) > 1e-9 for value in (vx, vy, wz))

    def stop(self, *, source="watchdog"):
        self.stops.append(source)
        self._command_zero = True

    @property
    def command_zero(self):
        return self._command_zero

    def observe_heading(self, yaw):
        assert math.isfinite(yaw)


class FakeTelemetry:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, **values):
        self.values.update(values)

    def update_hot(self, **values):
        self.values.update(values)

    def reset_navigation_task_peaks(self):
        pass


def bare_driver(clock: Clock) -> SonicDriver:
    driver = SonicDriver.__new__(SonicDriver)
    driver.config = SimpleNamespace(
        plant_port=8794,
        robot_id="g1-sim",
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )
    driver._driver_boot_id = UUID("30000000-0000-4000-8000-000000000003")
    driver.guard = FakeGuard()
    driver.telemetry = FakeTelemetry()
    driver._intent_ingress = SonicIntentIngress(clock=clock, max_age_s=0.5)
    driver._fault_latched = threading.Event()
    driver._terminal_fault_reason = None
    driver._navigation_task_active = False
    driver._lowstate = None
    driver._lowstate_lock = threading.RLock()
    driver._control_condition = threading.Condition(driver._lowstate_lock)
    driver._control_clock = PlantControlClock(control_hz=50)
    driver._control_clock_armed = False
    driver._control_consumer_ready = threading.Event()
    driver._control_consumer_ready.set()
    driver._motion_ready = threading.Event()
    driver._motion_ready.set()
    driver._startup_qualifying = False
    driver._startup_qualification_required_steps = 5
    driver._startup_qualification_consecutive_steps = 0
    driver._startup_qualification_dropped_samples = 0
    driver._startup_qualification_drop_generation = 0
    driver._pending_control_sample = None
    driver._wall_clock = clock
    driver._control_clock_armed_at_wall_s = None
    driver._last_lowstate_wall_s = clock.value
    driver._last_control_step_wall_s = clock.value
    driver._lowstate_stall_timeout_s = 1.0
    driver._stop = threading.Event()
    driver._support_lock = threading.RLock()
    driver._reference_authority = ReferenceAuthorityState()
    driver._dds_target_sequence = 0
    driver._plant_support_mode = "idle"
    driver._plant_support_stop_pending = False
    driver._plant_support_safety_latched = False
    driver._plant_support_stop_generation = None
    driver._last_published_targets = tuple(0.0 for _ in range(29))
    driver._safe_hold_targets = None
    driver._static_hold_active = True
    driver._joint_handoff = SimpleNamespace(
        reset=lambda: None,
        freeze=lambda _targets: None,
    )
    driver._arm_guard = SimpleNamespace(
        reset=lambda: None,
        anchor=lambda _targets: None,
    )
    return driver


def test_sonic_health_probe_is_strict_read_only_and_reports_terminal_fault() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver._last_lowstate_wall_s = clock.value - 0.02
    driver._last_control_step_wall_s = clock.value - 0.01
    producer_boot = UUID("20000000-0000-4000-8000-000000000002")
    request_id = UUID("40000000-0000-4000-8000-000000000004")
    probe = sonic_health_probe(
        robot_id="g1-sim",
        producer_boot_id=producer_boot,
        request_id=request_id,
    )
    telemetry_before = dict(driver.telemetry.values)
    guard_before = (list(driver.guard.accepted), list(driver.guard.stops))
    observed_health_probes = []
    driver._runtime_diagnostics = SimpleNamespace(
        observe_health_probe=observed_health_probes.append,
    )

    response = driver.accept_json(probe)
    assert response is not None
    ready = SonicHealth.from_response(
        {"request_id": str(request_id), "ok": True, **response},
        expected_request_id=request_id,
        expected_robot_id="g1-sim",
    )
    assert ready.status == "ready"
    assert ready.motion_ready is True
    assert ready.last_lowstate_age_s == pytest.approx(0.02)
    assert ready.last_control_step_age_s == pytest.approx(0.01)
    assert ready.plant_support_mode == "idle"
    assert ready.plant_support_safety_latched is False
    assert ready.plant_support_stop_pending is False
    assert ready.static_hold_active is True
    assert ready.navigation_task_active is False
    assert ready.motion_authorized is False
    assert ready.nav2_command_zero is True
    assert ready.confirmed_safe_hold is False
    assert observed_health_probes == [clock.value]
    assert driver.telemetry.values == telemetry_before
    assert (driver.guard.accepted, driver.guard.stops) == guard_before

    driver._terminal_fault_reason = "MuJoCo plant tick rolled back"
    driver._fault_latched.set()
    terminal_response = driver.accept_json(probe)
    assert terminal_response is not None
    terminal = SonicHealth.from_response(
        {"request_id": str(request_id), "ok": True, **terminal_response},
        expected_request_id=request_id,
        expected_robot_id="g1-sim",
    )
    assert terminal.status == "fault"
    assert terminal.terminal_fault is True
    assert terminal.motion_ready is False
    assert terminal.reason == "MuJoCo plant tick rolled back"
    assert observed_health_probes == [clock.value, clock.value]
    assert driver.telemetry.values == telemetry_before
    assert (driver.guard.accepted, driver.guard.stops) == guard_before


def test_adjacent_intent_and_health_probe_fit_the_20ms_loopback_budget() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver._last_lowstate_wall_s = clock.value - 0.01
    driver._last_control_step_wall_s = clock.value - 0.01
    server = _bind_command_server("127.0.0.1", 0)
    port = int(server.getsockname()[1])
    failures: list[Exception] = []

    def serve_two_connections() -> None:
        try:
            for _ in range(2):
                conn, _address = server.accept()
                with conn:
                    _handle_command_connection(driver, conn)
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=serve_two_connections, daemon=True)
    worker.start()
    request_id = UUID("50000000-0000-4000-8000-000000000005")
    try:
        send_json(
            "127.0.0.1",
            port,
            _intent_payload(
                adapter.tick(now=clock.value),
                measured_heading_yaw=0.0,
            ),
            timeout=0.020,
        )
        response = request_json(
            "127.0.0.1",
            port,
            sonic_health_probe(
                robot_id="g1-sim",
                producer_boot_id=UUID(
                    "20000000-0000-4000-8000-000000000002"
                ),
                request_id=request_id,
            ),
            timeout=0.020,
        )
    finally:
        worker.join(timeout=1.0)
        server.close()

    assert worker.is_alive() is False
    assert failures == []
    parsed = SonicHealth.from_response(
        response,
        expected_request_id=request_id,
        expected_robot_id="g1-sim",
    )
    assert parsed.status == "ready"
    assert driver._intent_ingress.armed is True


def test_driver_obeys_stop_before_model_guard_mapping() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    idle = adapter.tick(now=clock.value)
    driver.accept_json(_intent_payload(idle, measured_heading_yaw=0.0))
    clock.value += 0.01
    motion = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    driver.accept_json(_intent_payload(motion, measured_heading_yaw=0.0))
    assert driver.guard.accepted == [(0.20, 0.0, 0.0, "nav2")]

    clock.value += 0.01
    stop = adapter.ingest_twist(twist(clock), now=clock.value)
    payload = _intent_payload(stop, measured_heading_yaw=0.0)
    payload["velocity_x_mps"] = 10_000.0
    driver.accept_json(payload)
    assert driver.guard.accepted == [(0.20, 0.0, 0.0, "nav2")]
    assert driver.guard.stops[-1] == "gait_adapter"


def test_gait_fault_freezes_last_published_targets_and_latches_plant(
    monkeypatch,
) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    published = tuple(0.01 * index for index in range(29))
    frozen: list[tuple[float, ...]] = []
    driver._last_published_targets = published
    driver._joint_handoff = SimpleNamespace(
        freeze=lambda targets: frozen.append(tuple(targets)),
    )
    support_calls: list[str] = []

    def confirm_support(_host, _port, mode, **_kwargs):
        support_calls.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        confirm_support,
    )
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    clock.value += 0.01
    driver.accept_json(
        _intent_payload(
            adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value),
            measured_heading_yaw=0.0,
        )
    )
    clock.value += 0.01
    adapter.ingest_observation(
        observation(clock, vx=0.06, x=0.01),
        now=clock.value,
    )
    clock.value += 0.01
    fault = adapter.ingest_observation(
        observation(clock, vx=-0.10, x=-0.07),
        now=clock.value,
    )

    driver.accept_json(
        _intent_payload(fault, measured_heading_yaw=0.0)
    )

    assert fault.state == "FAULT"
    assert support_calls == ["safety_stop"]
    assert driver._plant_support_safety_latched is True
    assert driver.telemetry.values["plant_support_mode"] == "safety_stop"
    assert driver.telemetry.values["nav2_state"] == "safety_stop"
    assert driver._safe_hold_targets == published
    assert frozen and all(targets == published for targets in frozen)


def test_new_planner_generation_freezes_the_exact_actuator_boundary() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    published = tuple(0.01 * index for index in range(29))
    frozen: list[tuple[float, ...]] = []
    driver._last_published_targets = published
    driver._joint_handoff = SimpleNamespace(
        freeze=lambda targets: frozen.append(tuple(targets)),
    )

    assert driver._freeze_planner_handoff_boundary() is True
    assert frozen == [published]
    assert driver._safe_hold_targets is None


def _publishing_driver(clock: Clock):
    driver = bare_driver(clock)
    writes: list[tuple[float, ...]] = []

    class Publisher:
        @staticmethod
        def Write(message) -> None:
            writes.append(
                tuple(float(message.motor_cmd[index].q) for index in range(29))
            )

    driver._msg = SimpleNamespace(
        mode_machine=0,
        crc=0,
        motor_cmd=[
            SimpleNamespace(mode=0, kp=0.0, kd=0.0, q=0.0, dq=0.0, tau=0.0)
            for _ in range(29)
        ],
    )
    driver._publisher = Publisher()
    driver._crc = SimpleNamespace(Crc=lambda _message: 0)
    driver._kp = tuple(20.0 for _ in range(29))
    driver._kd = tuple(1.0 for _ in range(29))
    driver._last_control_step_wall_s = None
    driver._joint_handoff = StaticJointHandoff(
        warmup_s=0.35,
        blend_s=0.40,
    )
    driver._arm_guard = driver._new_arm_guard()
    return driver, writes


def _lowstate(targets: tuple[float, ...], *, arm_lag_rad: float = 0.0):
    states = []
    for index, target in enumerate(targets):
        measured = target - arm_lag_rad if index >= 15 else target
        states.append(SimpleNamespace(q=measured, dq=0.0, tau_est=0.0))
    return SimpleNamespace(mode_machine=0, motor_state=states)


def _action(driver: SonicDriver, targets: tuple[float, ...]) -> dict[str, float]:
    return {
        f"{driver._joint_name(index)}.q": float(target)
        for index, target in enumerate(targets)
    }


class _PlannerState:
    def __init__(self, movement_angle_rad: float) -> None:
        self.mode = 1
        self.speed = 0.2
        self.height = -1.0
        self.has_movement = True
        self.movement_angle_rad = float(movement_angle_rad)

    @property
    def movement_direction(self) -> tuple[float, float, float]:
        return (
            math.cos(self.movement_angle_rad),
            math.sin(self.movement_angle_rad),
            0.0,
        )

    @property
    def facing_direction(self) -> tuple[float, float, float]:
        return (1.0, 0.0, 0.0)


class _AsyncPlanner:
    def __init__(self) -> None:
        self.gen_frame = 0
        self.result = None

    def request_replan(self, cursor, _movement_state) -> None:
        self.gen_frame = int(cursor) + 10

    def try_get_new_motion(self):
        result = self.result
        self.result = None
        return result


def _planner_motion(dx: float, dy: float = 0.0):
    return SimpleNamespace(
        timesteps=3,
        body_positions=(
            (0.0, 0.0, 0.8),
            (dx / 2.0, dy / 2.0, 0.8),
            (dx, dy, 0.8),
        ),
    )


def _movement_intent(angle_rad: float) -> MovementIntent:
    return MovementIntent(
        mode="SLOW_WALK",
        speed_mps=0.2,
        movement_angle_rad=float(angle_rad),
        facing_angle_rad=0.0,
        has_movement=True,
        stale=False,
    )


def test_actual_29dof_dds_targets_are_continuous_at_start_replan_and_stop() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    initial = tuple(float(value) for value in DEFAULT_ANGLES)
    measured = _lowstate(initial, arm_lag_rad=0.50)
    policy = tuple(value + 0.40 for value in initial)
    driver._last_published_targets = initial

    # First required reference: ratio zero must cross the real publish path as
    # the complete frozen 29-DoF target, even though measured arms lag by 0.5rad.
    assert driver._freeze_planner_handoff_boundary() is True
    first = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=tuple(state.q for state in measured.motor_state),
        policy=policy,
        now=1.0,
    )
    assert first.handoff_ratio == 0.0
    published = driver._publish(
        _action(driver, first.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert published == pytest.approx(initial)
    assert writes[-1] == pytest.approx(initial)

    # A mid-gait required generation freezes the last actual DDS command. Both
    # the waiting frame and the newly accepted reference's ratio-zero frame are
    # identical at all 29 actuators; the following blend advances smoothly.
    assert driver._freeze_planner_handoff_boundary() is True
    waiting = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=False,
        measured=tuple(state.q for state in measured.motor_state),
        policy=policy,
        now=2.0,
    )
    waiting_published = driver._publish(
        _action(driver, waiting.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=False,
    )
    assert waiting_published == pytest.approx(published)
    assert writes[-1] == pytest.approx(published)

    admitted = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=tuple(state.q for state in measured.motor_state),
        policy=policy,
        now=2.1,
    )
    admitted_published = driver._publish(
        _action(driver, admitted.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert admitted.handoff_ratio == 0.0
    assert admitted_published == pytest.approx(waiting_published)
    assert writes[-1] == pytest.approx(waiting_published)

    blended = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=tuple(state.q for state in measured.motor_state),
        policy=policy,
        now=2.47,
    )
    blended_published = driver._publish(
        _action(driver, blended.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert 0.0 < blended.handoff_ratio < 1.0
    assert blended_published == pytest.approx(blended.targets)
    assert writes[-1] == pytest.approx(blended.targets)

    # A stop racing the policy path freezes and republishes the exact last DDS
    # vector rather than any measured position.
    assert driver._freeze_last_published_hold() is True
    stopped_measurement = _lowstate(
        tuple(value - 0.30 for value in blended_published)
    )
    driver._publish_frozen_hold(stopped_measurement)
    assert writes[-1] == pytest.approx(blended_published)
    assert driver._last_published_targets == pytest.approx(blended_published)


def test_local_idle_watchdog_transition_freezes_actual_29dof_dds_target() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    measured = _lowstate(boundary, arm_lag_rad=0.40)
    driver._publish(
        _action(driver, boundary),
        measured,
        control_dt_s=0.02,
        constrain_arms=False,
    )

    driver._intent_gate = ReplanIntentGate()
    assert driver._intent_gate.admit(_movement_intent(0.0), now=1.0) is True
    idle = MovementIntent(
        mode="IDLE",
        speed_mps=0.0,
        movement_angle_rad=0.0,
        facing_angle_rad=0.0,
        has_movement=False,
        stale=True,
    )
    driver.guard = SimpleNamespace(intent=lambda **_kwargs: idle, last_received_at=None)
    driver._controller = SimpleNamespace(
        controller=SimpleNamespace(reinit_heading=False, delta_heading=0.0),
        ms=SimpleNamespace(
            mode=None,
            speed=None,
            movement_angle=None,
            facing_angle=None,
            has_movement=None,
            needs_replan=None,
        ),
    )
    driver._planner_evidence = PlannerReferenceEvidenceGate()
    driver._decoder_facing_rad = 0.0
    driver._pending_decoder_heading_delta_rad = 0.0
    driver._last_intent_signature = None

    stopped_intent = driver._apply_intent(0.02, simulation_now_s=2.0)
    assert stopped_intent.mode == "IDLE"
    stopped = driver._joint_handoff.decide(
        motion_active=False,
        reference_ready=False,
        measured=tuple(state.q for state in measured.motor_state),
        policy=tuple(value + 0.10 for value in boundary),
        now=2.0,
    )
    stopped_published = driver._publish(
        _action(driver, stopped.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=False,
    )

    assert stopped.static_hold_active is True
    assert stopped.handoff_ratio == 0.0
    assert stopped_published == pytest.approx(boundary)
    assert writes[-1] == pytest.approx(boundary)


def test_mid_gait_replan_reject_wait_accept_and_bad_refresh_use_actual_dds_boundary() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    measured = _lowstate(
        tuple(float(value) for value in DEFAULT_ANGLES),
        arm_lag_rad=0.35,
    )
    initial_target = tuple(float(value) for value in DEFAULT_ANGLES)
    driver._publish(
        _action(driver, initial_target),
        measured,
        control_dt_s=0.02,
        constrain_arms=False,
    )

    planner = _AsyncPlanner()
    reference_owner = SimpleNamespace(ref_cursor=0)
    driver._planner_evidence = PlannerReferenceEvidenceGate(
        required_timeout_s=0.75
    )
    driver._planner_evidence.bind(planner, reference_owner=reference_owner)

    # Establish a fully active +X reference so the next intent is genuinely a
    # mid-gait replan rather than another startup case.
    driver._planner_evidence.observe_intent(
        _movement_intent(0.0), admitted=True, simulation_now_s=1.0
    )
    planner.request_replan(5, _PlannerState(0.0))
    planner.result = (_planner_motion(0.50), 15)
    assert planner.try_get_new_motion() is not None
    assert driver._freeze_planner_handoff_boundary() is True
    driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=tuple(state.q for state in measured.motor_state),
        policy=initial_target,
        now=1.0,
    )
    active_target = tuple(value + 0.02 for value in initial_target)
    active = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=tuple(state.q for state in measured.motor_state),
        policy=active_target,
        now=1.80,
    )
    assert active.handoff_ratio == 1.0
    active_published = driver._publish(
        _action(driver, active.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert active_published == pytest.approx(active_target)

    # A new -X intent freezes all 29 last-published targets. Its first result
    # predicts +X and is rejected; an empty planner cycle also keeps ratio zero.
    assert driver._freeze_planner_handoff_boundary() is True
    driver._planner_evidence.observe_intent(
        _movement_intent(math.pi), admitted=True, simulation_now_s=2.0
    )
    planner.request_replan(6, _PlannerState(math.pi))
    planner.result = (_planner_motion(0.40), 16)
    assert planner.try_get_new_motion() is None
    rejected = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=driver._planner_evidence.reference_ready,
        measured=tuple(state.q for state in measured.motor_state),
        policy=tuple(value - 0.05 for value in active_target),
        now=2.10,
    )
    assert rejected.handoff_ratio == 0.0
    rejected_published = driver._publish(
        _action(driver, rejected.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=False,
    )
    assert rejected_published == pytest.approx(active_published)

    assert planner.try_get_new_motion() is None
    waiting = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=driver._planner_evidence.reference_ready,
        measured=tuple(state.q for state in measured.motor_state),
        policy=tuple(value - 0.05 for value in active_target),
        now=2.60,
    )
    waiting_published = driver._publish(
        _action(driver, waiting.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=False,
    )
    assert waiting.handoff_ratio == 0.0
    assert waiting_published == pytest.approx(active_published)

    # The matching -X generation unlocks only a new ratio-zero frame, then
    # blends from the same frozen actuator boundary.
    planner.request_replan(7, _PlannerState(math.pi))
    planner.result = (_planner_motion(-0.50), 17)
    assert planner.try_get_new_motion() is not None
    admitted = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=driver._planner_evidence.reference_ready,
        measured=tuple(state.q for state in measured.motor_state),
        policy=tuple(value - 0.02 for value in active_target),
        now=2.70,
    )
    admitted_published = driver._publish(
        _action(driver, admitted.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert admitted.handoff_ratio == 0.0
    assert admitted_published == pytest.approx(active_published)

    new_policy_target = tuple(value - 0.01 for value in active_target)
    blended = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=driver._planner_evidence.reference_ready,
        measured=tuple(state.q for state in measured.motor_state),
        policy=new_policy_target,
        now=3.50,
    )
    blended_published = driver._publish(
        _action(driver, blended.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert blended.handoff_ratio == 1.0
    assert blended_published == pytest.approx(new_policy_target)

    # A later periodic result for the same intent is directionally bad. It is
    # discarded while the accepted reference remains active; the next policy
    # frame therefore continues, rather than snapping to measured or ratio zero.
    accepted_generation = driver._planner_evidence.telemetry_fields(
        ref_cursor=reference_owner.ref_cursor
    )["sonic_planner_accepted_generation"]
    planner.request_replan(8, _PlannerState(math.pi))
    planner.result = (_planner_motion(0.30), 18)
    assert planner.try_get_new_motion() is None
    assert driver._planner_evidence.reference_ready is True
    continued_target = tuple(value - 0.01 for value in new_policy_target)
    continued = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=driver._planner_evidence.reference_ready,
        measured=tuple(state.q for state in measured.motor_state),
        policy=continued_target,
        now=3.52,
    )
    continued_published = driver._publish(
        _action(driver, continued.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert continued.handoff_ratio == 1.0
    assert continued_published == pytest.approx(continued_target)
    evidence = driver._planner_evidence.telemetry_fields(
        ref_cursor=reference_owner.ref_cursor
    )
    assert evidence["sonic_planner_accepted_generation"] == accepted_generation
    assert evidence["sonic_planner_required_pending"] is False
    assert writes[-1] == pytest.approx(continued_target)


def test_required_planner_deadline_latches_terminal_exact_target_hold() -> None:
    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(0.01 * index for index in range(29))
    driver._last_published_targets = boundary
    driver._joint_handoff = StaticJointHandoff()
    driver._planner_evidence = PlannerReferenceEvidenceGate(
        required_timeout_s=0.75
    )
    driver._planner_evidence.observe_intent(
        _movement_intent(0.0),
        admitted=True,
        simulation_now_s=4.0,
    )
    assert driver._freeze_planner_handoff_boundary() is True
    measured = _lowstate(boundary, arm_lag_rad=0.40)
    waiting = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=False,
        measured=tuple(state.q for state in measured.motor_state),
        policy=tuple(value + 0.10 for value in boundary),
        now=4.74,
    )
    waiting_published = driver._publish(
        _action(driver, waiting.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=False,
    )
    assert waiting.handoff_ratio == 0.0
    assert waiting_published == pytest.approx(boundary)

    with pytest.raises(PlannerReferenceDeadlineExceeded, match="timed out"):
        driver._enforce_planner_reference_deadline(simulation_now_s=4.75)

    driver._publish_frozen_hold(measured)

    assert driver._fault_latched.is_set() is True
    assert driver._safe_hold_targets == boundary
    assert writes[-1] == pytest.approx(boundary)
    assert driver._last_published_targets == pytest.approx(boundary)
    assert driver.guard.stops[-1] == "terminal_fault"
    assert "required SONIC planner reference timed out" in str(
        driver.telemetry.values["last_error"]
    )


def test_driver_requires_startup_qualification_before_accepting_motion() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver._motion_ready.clear()

    idle = adapter.tick(now=clock.value)
    driver.accept_json(_intent_payload(idle, measured_heading_yaw=0.0))
    clock.value += 0.01
    motion = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    driver.accept_json(_intent_payload(motion, measured_heading_yaw=0.0))

    assert driver.guard.accepted == []
    assert driver.guard.stops[-1] == "sonic_startup_hold"
    assert driver.telemetry.values["nav2_state"] == "startup_hold"

    # The rejected command is never replayed. Only a fresh, later gait intent
    # may cross the boundary after qualification becomes motion-ready.
    driver._motion_ready.set()
    clock.value += 0.01
    fresh = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    driver.accept_json(_intent_payload(fresh, measured_heading_yaw=0.0))
    assert driver.guard.accepted == [(0.20, 0.0, 0.0, "nav2")]


def test_idle_observation_reconnect_emits_clean_driver_idle_without_replay() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)

    initial_idle = adapter.tick(now=clock.value)
    driver.accept_json(_intent_payload(initial_idle, measured_heading_yaw=0.0))
    assert driver.telemetry.values["nav2_state"] == "idle"

    clock.value += 0.31
    stale = adapter.tick(now=clock.value)
    driver.accept_json(_intent_payload(stale, measured_heading_yaw=0.0))
    assert stale.state == "DISARMED"
    assert driver.telemetry.values["nav2_state"] == "safety_stop"
    assert "stale while idle" in str(driver.telemetry.values["last_error"])
    assert driver.guard.accepted == []

    clock.value += 0.01
    fresh_observation = adapter.ingest_observation(
        observation(clock), now=clock.value
    )
    driver.accept_json(
        _intent_payload(fresh_observation, measured_heading_yaw=0.0)
    )
    clean_idle = adapter.arm(1, now=clock.value)
    driver.accept_json(_intent_payload(clean_idle, measured_heading_yaw=0.0))

    assert clean_idle.state == "IDLE"
    assert clean_idle.reason is None
    assert driver.telemetry.values["nav2_state"] == "idle"
    assert driver.telemetry.values["last_error"] is None
    assert driver.guard.accepted == []


def test_support_stop_requires_new_generation_and_explicit_control_arm() -> None:
    clock = Clock()
    adapter, ingress = runtime(clock)
    assert ingress.latch_support_stop() == 1

    clock.value += 0.01
    same_generation = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    same_generation["control_arm"] = True
    rejected = ingress.accept(same_generation)
    assert rejected.action == "stop"
    assert rejected.arm_acknowledged is False
    assert ingress.armed is False
    assert "strictly newer" in str(rejected.reason)

    clock.value += 0.01
    adapter.advance_generation(2, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    idle = adapter.arm(2, now=clock.value)

    # A bridge restart or periodic IDLE alone does not release safety_stop.
    periodic = ingress.accept(_intent_payload(idle, measured_heading_yaw=0.0))
    assert periodic.arm_acknowledged is False
    assert ingress.armed is False
    assert "control.acquire" in str(periodic.reason)

    # A fresh non-zero Nav2 command in the new generation still has no
    # authority before the explicit arm acknowledgement and is not buffered.
    clock.value += 0.01
    pre_arm_motion = adapter.ingest_twist(
        twist(clock, generation=2, vx=0.20), now=clock.value
    )
    rejected_motion = decide(ingress, pre_arm_motion)
    assert rejected_motion.action == "stop"
    assert ingress.armed is False

    clock.value += 0.01
    adapter.ingest_twist(
        twist(clock, generation=2), now=clock.value
    )
    clock.value += 0.01
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    clock.value += 0.16
    idle_again = adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    explicit_payload = _intent_payload(
        idle_again, measured_heading_yaw=0.0
    )
    explicit_payload["control_arm"] = True
    armed = ingress.accept(explicit_payload)
    assert armed.arm_acknowledged is True
    assert ingress.armed is True
    ingress.confirm_support_rearm(2)

    clock.value += 0.01
    motion = adapter.ingest_twist(
        twist(clock, generation=2, vx=0.20), now=clock.value
    )
    accepted = decide(ingress, motion)
    assert accepted.action == "motion"
    assert accepted.vx == pytest.approx(0.20)


def test_driver_support_notification_stays_latched_until_control_arm(
    monkeypatch,
) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )

    acknowledgement = driver.accept_json(
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "source": "robot_agent_safe_stop",
            "mode": "safety_stop",
            "reason": "agent startup",
        }
    )
    assert acknowledgement == {
        "ok": True,
        "mode": "safety_stop",
        "stopped_generation": 1,
    }
    assert driver.telemetry.values["plant_support_mode"] == "safety_stop"
    assert driver.telemetry.values["plant_support_safety_latched"] is True
    assert driver._set_plant_support_mode("idle") is True
    assert driver._set_plant_support_mode("locomotion") is False
    assert driver.guard.accepted == []

    clock.value += 0.01
    same_generation = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    same_generation["control_arm"] = True
    with pytest.raises(ValueError, match="fresh stationary IDLE"):
        driver.accept_json(same_generation)
    assert driver._plant_support_safety_latched is True

    calls: list[int] = []

    def rearm(_host, _port, generation, **_kwargs):
        calls.append(generation)
        return {
            "mode": "idle",
            "lastRearmGeneration": generation,
            "safetyStopLatched": False,
        }

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support", rearm
    )
    clock.value += 0.01
    adapter.advance_generation(2, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    adapter.arm(2, now=clock.value)
    clock.value += 0.01
    control_arm = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    control_arm["control_arm"] = True
    assert driver.accept_json(control_arm) == {
        "ok": True,
        "mode": "idle",
        "generation": 2,
    }
    assert calls == [2]
    assert driver.telemetry.values["plant_support_mode"] == "idle"
    assert driver.telemetry.values["plant_support_safety_latched"] is False
    assert driver.guard.accepted == []

    clock.value += 0.01
    motion = adapter.ingest_twist(
        twist(clock, generation=2, vx=0.20), now=clock.value
    )
    driver.accept_json(_intent_payload(motion, measured_heading_yaw=0.0))
    assert driver.guard.accepted == [(0.20, 0.0, 0.0, "nav2")]


def test_control_arm_rechecks_freshness_after_an_earlier_probe(
    monkeypatch,
) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    driver.accept_json(
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "source": "robot_agent_safe_stop",
            "mode": "safety_stop",
            "reason": "agent startup",
        }
    )
    probe = sonic_health_probe(
        robot_id="g1-sim",
        producer_boot_id=UUID("20000000-0000-4000-8000-000000000002"),
        request_id=UUID("60000000-0000-4000-8000-000000000006"),
    )
    health = driver.accept_json(probe)
    assert health is not None
    assert health["last_lowstate_age_s"] == pytest.approx(0.0)
    assert health["last_control_step_age_s"] == pytest.approx(0.0)

    called = False

    def unexpected_rearm(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("stale control arm must not reach Plant rearm")

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        unexpected_rearm,
    )
    ingress_before = (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
    )
    clock.value += 0.11
    adapter.advance_generation(2, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    adapter.arm(2, now=clock.value)
    payload = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    payload["control_arm"] = True

    with pytest.raises(RuntimeError, match="LowState age"):
        driver.accept_json(payload)

    assert called is False
    assert (
        driver._intent_ingress.generation,
        driver._intent_ingress.sequence,
        driver._intent_ingress.armed,
        driver._intent_ingress.staged_control_arm_generation,
    ) == ingress_before
    assert driver._plant_support_mode == "safety_stop"
    assert driver._plant_support_safety_latched is True


def test_control_arm_rechecks_freshness_after_plant_rearm(
    monkeypatch,
) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    driver.accept_json(
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "source": "robot_agent_safe_stop",
            "mode": "safety_stop",
            "reason": "agent startup",
        }
    )
    clock.value += 0.01
    adapter.advance_generation(2, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    adapter.arm(2, now=clock.value)
    payload = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    payload["control_arm"] = True
    rearm_calls: list[int] = []
    support_calls: list[str] = []

    def delayed_rearm(_host, _port, generation, **_kwargs):
        rearm_calls.append(generation)
        assert driver._intent_ingress.armed is False
        assert driver._intent_ingress.staged_control_arm_generation == generation
        clock.value += 0.11
        return {
            "mode": "idle",
            "lastRearmGeneration": generation,
            "safetyStopLatched": False,
        }

    def confirm_stop(_host, _port, mode, **_kwargs):
        support_calls.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        delayed_rearm,
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        confirm_stop,
    )

    with pytest.raises(RuntimeError, match="post-rearm health failed.*LowState age"):
        driver.accept_json(payload)

    assert rearm_calls == [2]
    assert support_calls == ["safety_stop"]
    assert driver._intent_ingress.armed is False
    assert driver._intent_ingress.staged_control_arm_generation is None
    assert driver._intent_ingress.support_stop_generation == 2
    assert driver._plant_support_mode == "safety_stop"
    assert driver._plant_support_safety_latched is True
    assert driver._plant_support_stop_pending is False
    assert driver.guard.accepted == []


def test_safe_stop_pending_blocks_a_racing_control_arm(monkeypatch) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    driver.accept_json(
        {
            "type": "stop",
            "source": "robot_agent_safe_stop",
            "reason": "lease expired",
        }
    )
    assert driver.telemetry.values["plant_support_requested_mode"] == "safety_stop"
    assert driver.telemetry.values["plant_support_confirmed_mode"] == "idle"
    assert driver.telemetry.values["plant_support_stop_pending"] is True

    called = False

    def unexpected_rearm(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("pending safe stop must block Plant rearm")

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        unexpected_rearm,
    )
    clock.value += 0.01
    adapter.advance_generation(2, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    adapter.arm(2, now=clock.value)
    clock.value += 0.01
    payload = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    payload["control_arm"] = True
    with pytest.raises(RuntimeError, match="confirmed settled safety hold"):
        driver.accept_json(payload)
    assert called is False
    assert driver.guard.accepted == []

    driver.accept_json(
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "source": "robot_agent_safe_stop",
            "mode": "safety_stop",
            "reason": "lease expired",
        }
    )
    assert driver.telemetry.values["plant_support_confirmed_mode"] == "safety_stop"
    assert driver.telemetry.values["plant_support_stop_pending"] is False
    assert driver.telemetry.values["plant_support_safety_latched"] is True


def test_failed_explicit_rearm_keeps_stop_latched_and_drops_authority(
    monkeypatch,
) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    driver.accept_json(
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "source": "robot_agent_safe_stop",
            "mode": "safety_stop",
            "reason": "operator stop",
        }
    )

    support_calls: list[str] = []

    def fail_rearm(*_args, **_kwargs):
        raise RuntimeError("Plant unavailable")

    def confirm_stop(_host, _port, mode, **_kwargs):
        support_calls.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support", fail_rearm
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support", confirm_stop
    )
    clock.value += 0.01
    adapter.advance_generation(2, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    adapter.arm(2, now=clock.value)
    clock.value += 0.01
    payload = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    payload["control_arm"] = True
    with pytest.raises(RuntimeError, match="rearm failed"):
        driver.accept_json(payload)

    assert driver._plant_support_safety_latched is True
    assert driver._intent_ingress.armed is False
    assert driver.telemetry.values["plant_support_confirmed_mode"] == "safety_stop"
    assert driver.guard.accepted == []
    assert support_calls == ["safety_stop"]


def test_support_transition_race_preserves_the_stronger_stop(monkeypatch) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    calls: list[str] = []

    def racing_support(_host, _port, mode, **_kwargs):
        calls.append(mode)
        if mode == "locomotion":
            # Model the command thread accepting Robot Agent's stop while the
            # control thread is waiting for the older locomotion response.
            driver._plant_support_stop_pending = True
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support", racing_support
    )

    assert driver._set_plant_support_mode("locomotion") is False
    assert calls == ["locomotion", "safety_stop"]
    assert driver._plant_support_safety_latched is True
    assert driver._plant_support_stop_pending is False
    assert driver._intent_ingress.armed is False
    assert driver.telemetry.values["plant_support_requested_mode"] == "safety_stop"
    assert driver.telemetry.values["plant_support_confirmed_mode"] == "safety_stop"
    assert driver.guard.accepted == []


def test_terminal_fault_cannot_be_rearmed_online(monkeypatch) -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver._fault_latched.set()
    driver._terminal_fault_reason = "physics fault"
    driver.accept_json(
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "source": "robot_agent_safe_stop",
            "mode": "safety_stop",
            "reason": "physics fault",
        }
    )

    called = False

    def unexpected_rearm(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("terminal fault must not call Plant rearm")

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.rearm_plant_support",
        unexpected_rearm,
    )
    clock.value += 0.01
    adapter.advance_generation(2, now=clock.value)
    adapter.ingest_observation(
        observation(clock, generation=2), now=clock.value
    )
    adapter.arm(2, now=clock.value)
    payload = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    payload["control_arm"] = True
    payload["request_id"] = "arm-after-fault"

    with pytest.raises(RuntimeError, match="process restart"):
        driver.accept_json(payload)
    assert called is False
    assert driver._plant_support_safety_latched is True


def test_driver_rejects_raw_cmd_vel_and_fails_closed(monkeypatch) -> None:
    clock = Clock()
    driver = bare_driver(clock)

    def unavailable_support(*_args, **_kwargs):
        raise OSError("isolated test Plant is unavailable")

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        unavailable_support,
    )
    with pytest.raises(ValueError, match="sonic_intent"):
        driver.accept_json(
            {
                "type": "cmd_vel",
                "source": "nav2",
                "vx": 0.20,
                "vy": 0.0,
                "wz": 0.0,
            }
        )
    assert driver.guard.accepted == []
    assert "invalid_gait_intent" in driver.guard.stops
    assert driver.guard.stops[-1] == "plant_safety_stop_unconfirmed"


def test_driver_tick_rollback_is_terminal_and_never_resyncs() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver._tick_guard = PlantTickGuard()
    driver._on_lowstate(SimpleNamespace(tick=5_000))
    assert driver._fault_latched.is_set() is False
    driver._on_lowstate(SimpleNamespace(tick=1))
    assert driver._fault_latched.is_set() is True
    assert driver.guard.stops[-1] == "terminal_fault"
    assert driver.telemetry.values["status"] == "failed"
    assert "explicit process restart" in str(driver.telemetry.values["last_error"])
