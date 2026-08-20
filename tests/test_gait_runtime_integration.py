from __future__ import annotations

from dataclasses import replace
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
    _observe_health_reference_authority,
    _ros_stamp_in_monotonic_domain,
)
from hide_mujoco_sonic.motion_fault import (
    HealthFaultDecision,
    MotionFaultCause,
    SonicHealth,
    sonic_health_probe,
)
from hide_mujoco_sonic.normal_stop import NormalStopIdentity, NormalStopState
from hide_mujoco_sonic.normal_stop_wire import (
    normal_stop_begin_request,
    normal_stop_plant_request,
    normal_stop_stationary_request,
)
from hide_mujoco_sonic.planner_evidence import (
    PlannerAuthorityEvidence,
    PlannerReferenceDeadlineExceeded,
    PlannerReferenceEvidenceGate,
)
from hide_mujoco_sonic.reference_authority import ReferenceAuthorityState
from hide_mujoco_sonic.safety import (
    MovementIntent,
    PlantControlClock,
    PlantControlStep,
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


def grant_reference(
    adapter: GaitAdapter,
    starting,
    *,
    granted_at: float,
    reference_epoch: int = 1,
    dds_sequence: int = 1,
) -> None:
    assert adapter.observe_reference_authority(
        generation=int(starting.generation),
        source_sequence=starting.sequence,
        reference_epoch=reference_epoch,
        dds_target_sequence=dds_sequence,
        granted_at_monotonic_s=granted_at,
        now=granted_at,
    ) is True


def reference_health(starting, *, granted_at: float) -> SonicHealth:
    return SonicHealth(
        request_id=UUID("40000000-0000-4000-8000-000000000004"),
        robot_id="g1-sim",
        driver_boot_id=UUID("30000000-0000-4000-8000-000000000003"),
        status="ready",
        terminal_fault=False,
        reason=None,
        motion_ready=True,
        last_lowstate_age_s=0.01,
        last_control_step_age_s=0.01,
        plant_support_mode="locomotion",
        plant_support_safety_latched=False,
        plant_support_stop_pending=False,
        static_hold_active=False,
        navigation_task_active=True,
        motion_authorized=True,
        nav2_command_zero=False,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=int(starting.generation),
        reference_authority_source_sequence=starting.sequence,
        reference_authority_intent_generation=1,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=granted_at,
        reference_authority_granted_tick_ms=500,
        reference_authority_dds_target_sequence=7,
    )


def healthy_decision(health: SonicHealth) -> HealthFaultDecision:
    return HealthFaultDecision(
        confirmed_cause=None,
        pending=None,
        required_count=2,
        driver_boot_id=health.driver_boot_id,
        filtered_transient_count=0,
        quarantined_cause=None,
        settled_safe_hold=False,
        quarantined_stale_count=0,
    )


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


def test_bridge_starts_measurement_clock_from_healthy_first_dds_grant() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    clock.value = 10.01
    starting = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    later = adapter.tick(now=10.02)
    health = reference_health(starting, granted_at=10.015)

    assert later.sequence > starting.sequence
    assert _observe_health_reference_authority(
        adapter,
        health,
        healthy_decision(health),
        now=10.025,
    ) is True
    assert adapter._start_at == pytest.approx(10.015)


def test_bridge_does_not_grant_authority_before_hard_fault_decision() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    clock.value = 10.01
    starting = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    terminal = replace(
        reference_health(starting, granted_at=10.015),
        status="fault",
        terminal_fault=True,
        reason="terminal driver fault",
        motion_ready=False,
    )
    decision = replace(
        healthy_decision(terminal),
        confirmed_cause=MotionFaultCause(
            source="sonic_driver",
            code="SONIC_TERMINAL_FAULT",
            reason="terminal driver fault",
        ),
    )

    assert _observe_health_reference_authority(
        adapter,
        terminal,
        decision,
        now=10.025,
    ) is False
    assert adapter._start_at is None


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
    grant_reference(adapter, starting, granted_at=clock.value)

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
    starting = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    decide(
        ingress,
        starting,
    )
    grant_reference(adapter, starting, granted_at=clock.value)
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
        # Unit-test drivers must never inherit the production Plant endpoint.
        # Tests exercising a Plant transition inject the transport explicitly.
        plant_port=0,
        robot_id="g1-sim",
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )
    driver._driver_boot_id = UUID("30000000-0000-4000-8000-000000000003")
    driver.guard = FakeGuard()
    driver.telemetry = FakeTelemetry()
    driver._intent_ingress = SonicIntentIngress(clock=clock, max_age_s=0.5)
    driver._fault_latched = threading.Event()
    driver._terminal_fault_lock = threading.Lock()
    driver._terminal_plant_halt_attempted = False
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
    driver._reference_source_identity = None
    driver._reference_request_pending = False
    driver._vendor_revoke_pending = False
    driver._intent_gate = None
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
    driver._normal_stop = NormalStopState()
    driver._normal_stop_bridge_boot_id = None
    driver._normal_stop_seen_bridge_boot_ids = set()
    driver._normal_stop_plant_safety_stop_epoch = None
    driver._normal_stop_last_confirmed_plant_epoch = 0
    driver._normal_stop_begin_intent = None
    driver._normal_stop_seen_request_ids = set()
    driver._normal_stop_replays = {}
    return driver


def confirm_correlated_driver_stop(
    adapter: GaitAdapter,
    driver: SonicDriver,
    clock: Clock,
) -> NormalStopIdentity:
    """Drive the real gait/wire/driver stop path to DRIVER_CONFIRMED."""

    generation = adapter.generation
    assert generation is not None
    stopping = adapter.request_safe_stop(generation, now=clock.value)
    assert stopping.state == "STOPPING"
    bridge_boot_id = UUID("20000000-0000-4000-8000-000000000002")
    driver.accept_json(
        normal_stop_begin_request(
            request_id=UUID("10000000-0000-4000-8000-000000000001"),
            robot_id="g1-sim",
            bridge_boot_id=bridge_boot_id,
            reason="integration test correlated stop",
            intent=_intent_payload(stopping, measured_heading_yaw=0.0),
        )
    )
    identity = driver._normal_stop.snapshot.identity
    assert identity is not None

    clock.value += 0.01
    first = adapter.ingest_observation(
        observation(clock, generation=generation), now=clock.value
    )
    assert first.state == "STOPPING"
    clock.value += 0.16
    stationary = adapter.ingest_observation(
        observation(clock, generation=generation), now=clock.value
    )
    assert stationary.state == "DISARMED"
    assert stationary.stop_acknowledged is True
    driver.accept_json(
        normal_stop_stationary_request(
            request_id=UUID("10000000-0000-4000-8000-000000000002"),
            robot_id="g1-sim",
            bridge_boot_id=bridge_boot_id,
            driver_boot_id=driver._driver_boot_id,
            identity=identity,
            intent=_intent_payload(stationary, measured_heading_yaw=0.0),
        )
    )
    driver.accept_json(
        normal_stop_plant_request(
            request_id=UUID("10000000-0000-4000-8000-000000000003"),
            robot_id="g1-sim",
            bridge_boot_id=bridge_boot_id,
            driver_boot_id=driver._driver_boot_id,
            identity=identity,
            plant={
                "confirmed": True,
                "mode": "safety_stop",
                "safety_stop_latched": True,
                "safety_stop_epoch": 1,
                "last_rearm_generation": generation,
                "fault": None,
                "error": None,
            },
        )
    )
    assert driver._normal_stop.snapshot.phase == "DRIVER_CONFIRMED"
    driver._last_lowstate_wall_s = clock.value
    driver._last_control_step_wall_s = clock.value
    return identity


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

    driver._reference_authority.require(generation=7, source_sequence=3)
    driver._reference_authority.mark_installed(
        generation=7,
        source_sequence=3,
        planner_intent_generation=2,
        reference_epoch=4,
    )
    driver._reference_authority.prime(
        generation=7,
        source_sequence=3,
        reference_epoch=4,
    )
    driver._reference_authority.grant(
        generation=7,
        source_sequence=3,
        reference_epoch=4,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=1_020,
        dds_target_sequence=8,
    )
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
    assert terminal.matched_reference_actuator_authority is False
    assert terminal.reference_authority_state == "REVOKED"
    assert terminal.reference_authority_granted_at_monotonic_s is None
    assert terminal.reference_authority_granted_tick_ms is None
    assert terminal.reference_authority_dds_target_sequence is None
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


def test_legacy_stop_atomically_revokes_ingress_before_racing_motion_accept() -> None:
    clock = Clock()
    driver = bare_driver(clock)
    revoke_entered = threading.Event()
    release_revoke = threading.Event()
    motion_complete = threading.Event()
    events: list[str] = []

    class BlockingIngress:
        revoked = False

        def revoke(self) -> None:
            events.append("revoke_entered")
            revoke_entered.set()
            assert release_revoke.wait(timeout=2.0)
            self.revoked = True
            events.append("revoke_complete")

        def accept(self, _payload):
            events.append("motion_accept")
            if self.revoked:
                return SimpleNamespace(
                    action="stop",
                    state="IDLE",
                    reason="ingress revoked",
                    heading_yaw=None,
                )
            return SimpleNamespace(
                action="motion",
                state="STARTING",
                reason=None,
                heading_yaw=None,
                generation=1,
                sequence=1,
                vx=0.2,
                vy=0.0,
                wz=0.0,
            )

    driver._intent_ingress = BlockingIngress()
    stop_errors: list[Exception] = []
    motion_errors: list[Exception] = []

    def legacy_stop() -> None:
        try:
            driver.accept_json({"type": "stop"})
        except Exception as exc:
            stop_errors.append(exc)

    def racing_motion() -> None:
        try:
            driver.accept_json({"type": "sonic_intent"})
        except Exception as exc:
            motion_errors.append(exc)
        finally:
            motion_complete.set()

    stop_thread = threading.Thread(target=legacy_stop)
    motion_thread = threading.Thread(target=racing_motion)
    stop_thread.start()
    assert revoke_entered.wait(timeout=2.0)
    motion_thread.start()
    assert motion_complete.wait(timeout=0.05) is False
    release_revoke.set()
    stop_thread.join(timeout=2.0)
    motion_thread.join(timeout=2.0)

    assert stop_errors == []
    assert motion_errors == []
    assert events == ["revoke_entered", "revoke_complete", "motion_accept"]
    assert driver.guard.accepted == []
    assert driver._navigation_task_active is False
    assert driver._reference_authority.snapshot.phase == "REVOKED"


def test_parsed_motion_cannot_replay_after_legacy_stop_wins_final_boundary() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    ready_entered = threading.Event()
    release_ready = threading.Event()

    class BlockingMotionReady:
        @staticmethod
        def is_set() -> bool:
            ready_entered.set()
            assert release_ready.wait(timeout=2.0)
            return True

        @staticmethod
        def clear() -> None:
            return None

    driver._motion_ready = BlockingMotionReady()
    clock.value += 0.01
    motion = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    motion_payload = _intent_payload(motion, measured_heading_yaw=0.0)
    motion_errors: list[Exception] = []

    def accept_motion() -> None:
        try:
            driver.accept_json(motion_payload)
        except Exception as exc:
            motion_errors.append(exc)

    motion_thread = threading.Thread(target=accept_motion)
    motion_thread.start()
    assert ready_entered.wait(timeout=2.0)
    driver.accept_json({"type": "stop"})
    release_ready.set()
    motion_thread.join(timeout=2.0)

    assert motion_errors == []
    assert driver.guard.accepted == []
    assert driver._intent_ingress.armed is False
    assert driver._navigation_task_active is False
    assert driver._reference_authority.snapshot.phase == "REVOKED"


def test_driver_keeps_first_starting_sequence_across_repeated_bridge_envelopes() -> None:
    clock = Clock()
    adapter, _ = runtime(clock)
    driver = bare_driver(clock)
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )

    clock.value += 0.01
    admitted = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    driver.accept_json(_intent_payload(admitted, measured_heading_yaw=0.0))
    first_identity = (int(admitted.generation), admitted.sequence)
    assert driver._reference_source_identity == first_identity

    clock.value += 0.01
    repeated = adapter.tick(now=clock.value)
    assert repeated.sequence > admitted.sequence
    driver.accept_json(_intent_payload(repeated, measured_heading_yaw=0.0))
    assert driver._reference_source_identity == first_identity


def test_gait_fault_freezes_target_and_latches_terminal_without_recoverable_support(
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
    terminal_halts: list[str] = []

    def confirm_support(_host, _port, mode, **_kwargs):
        support_calls.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        confirm_support,
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        lambda _host, _port, reason, **_kwargs: (
            terminal_halts.append(reason)
            or {"mode": "safety_stop", "safetyStopLatched": True, "fault": reason}
        ),
    )
    driver.accept_json(
        _intent_payload(adapter.tick(now=clock.value), measured_heading_yaw=0.0)
    )
    clock.value += 0.01
    starting = adapter.ingest_twist(twist(clock, vx=0.20), now=clock.value)
    driver.accept_json(_intent_payload(starting, measured_heading_yaw=0.0))
    grant_reference(adapter, starting, granted_at=clock.value)
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
    assert support_calls == []
    assert terminal_halts == [fault.reason]
    assert driver._fault_latched.is_set()
    assert driver._terminal_fault_reason == fault.reason
    assert driver._plant_support_safety_latched is False
    assert driver.telemetry.values["status"] == "failed"
    assert driver.telemetry.values["nav2_state"] == "fault_latched"
    assert driver._safe_hold_targets == published
    assert frozen and all(targets == published for targets in frozen)


def test_terminal_fault_halts_plant_before_trace_even_if_vendor_revoke_fails(
    monkeypatch,
) -> None:
    clock = Clock()
    driver = bare_driver(clock)
    driver._intent_gate = ReplanIntentGate()
    driver._kp = tuple(20.0 for _ in range(29))
    driver._kd = tuple(1.0 for _ in range(29))
    events: list[str] = []
    recoverable_support_calls: list[str] = []
    vendor_revoke_calls: list[dict[str, object]] = []

    class BrokenController:
        execution_feedback_pending = False

        @staticmethod
        def revoke_reference_authority(**kwargs) -> None:
            vendor_revoke_calls.append(dict(kwargs))

    class SlowTrace:
        joint_names = ("joint",)

        @staticmethod
        def snapshot(**_kwargs):
            events.append("trace_snapshot")
            return {"sample_count": 0}

        @staticmethod
        def persist(_path, _payload):
            events.append("trace_persist")
            return "/tmp/terminal-trace.json"

    driver._controller = BrokenController()
    driver._joint_handoff = SimpleNamespace(
        freeze=lambda _targets: (_ for _ in ()).throw(
            RuntimeError("injected boundary freeze failure")
        )
    )
    driver._fault_trace = SlowTrace()
    driver._fault_trace_path = "/tmp/terminal-trace.json"
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        lambda _host, _port, mode, **_kwargs: (
            recoverable_support_calls.append(mode) or {"mode": mode}
        ),
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        lambda _host, _port, _reason, **_kwargs: (
            events.append("plant_halt")
            or {"mode": "safety_stop", "safetyStopLatched": True}
        ),
    )

    driver._latch_terminal_fault("SONIC runtime fault: injected verifier failure")
    driver._latch_terminal_fault("duplicate terminal callback")

    assert driver._fault_latched.is_set()
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert events == ["plant_halt", "trace_snapshot", "trace_persist"]
    assert recoverable_support_calls == []
    assert vendor_revoke_calls == []
    assert driver._vendor_revoke_pending is True
    assert driver.telemetry.values["plant_terminal_halt_confirmed"] is True
    assert driver.telemetry.values["terminal_boundary_error"] == (
        "injected boundary freeze failure"
    )
    assert driver.telemetry.values["sonic_reference_authority_state"] == "REVOKED"
    assert driver.telemetry.values["sonic_reference_actuator_authority"] is False
    assert driver.telemetry.values["sonic_execution_feedback_pending"] is False


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
        def Write(message) -> bool:
            writes.append(
                tuple(float(message.motor_cmd[index].q) for index in range(29))
            )
            return True

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


class _TransactionalEvidence:
    def __init__(self) -> None:
        self.intent_generation = 0
        self.intent_signature: str | None = None
        self.accepted_request_generation = 0
        self.ready = False

    def observe_intent(self, intent, *, admitted, simulation_now_s=None) -> None:
        if not admitted:
            return
        if intent.mode == "IDLE":
            self.revoke("IDLE")
            return
        self.intent_generation += 1
        self.intent_signature = f"intent-{self.intent_generation}"
        self.ready = False

    def accept_current(self) -> None:
        self.accepted_request_generation += 1
        self.ready = True

    def revoke(self, _reason) -> None:
        self.ready = False

    @property
    def reference_ready(self) -> bool:
        return self.ready

    @property
    def authority_evidence(self) -> PlannerAuthorityEvidence:
        return PlannerAuthorityEvidence(
            reference_ready=self.ready,
            intent_generation=(
                self.intent_generation if self.intent_generation else None
            ),
            intent_signature=self.intent_signature,
            accepted_request_generation=(
                self.accepted_request_generation if self.ready else None
            ),
            accepted_intent_generation=(
                self.intent_generation if self.ready else None
            ),
            accepted_intent_signature=(
                self.intent_signature if self.ready else None
            ),
        )

    def enforce_required_deadline(self, *, simulation_now_s) -> None:
        assert math.isfinite(simulation_now_s)

    def telemetry_fields(self, *, ref_cursor):
        return {
            "sonic_planner_reference_ready": self.ready,
            "sonic_reference_cursor": ref_cursor,
        }

    def trace_fields(self, *, ref_cursor):
        return self.telemetry_fields(ref_cursor=ref_cursor)


class _TransactionalController:
    def __init__(self, evidence: _TransactionalEvidence, target) -> None:
        self.initialized = True
        self.reference_install_epoch = 1
        self.matching_reference_installed = True
        self.execution_feedback_pending = False
        self.reference_authority = False
        self.install_ready = False
        self.refresh_ready = False
        self.request_calls = 0
        self.poll_calls = 0
        self.acquire_calls = []
        self.run_calls = 0
        self.commits = []
        self.revoke_calls: list[tuple[bool, bool]] = []
        self._candidate_identity = None
        self._target = tuple(float(value) for value in target)
        self._evidence = evidence

        def set_reference_authority(
            *,
            actuator_authority,
            matching_reference_ready,
        ) -> None:
            self.reference_authority = bool(
                actuator_authority and matching_reference_ready
            )

        self.controller = SimpleNamespace(
            ref_cursor=0,
            reinit_heading=False,
            delta_heading=0.0,
            playing=True,
            set_reference_authority=set_reference_authority,
        )
        self.ms = SimpleNamespace(
            mode=0,
            speed=-1.0,
            movement_angle=0.0,
            facing_angle=0.0,
            has_movement=False,
            needs_replan=False,
        )

    def request_reference(self, _lowstate) -> None:
        assert self.execution_feedback_pending is False
        self.request_calls += 1
        self.matching_reference_installed = False
        self.reference_authority = False

    def poll_reference(self) -> bool:
        assert self.execution_feedback_pending is False
        self.poll_calls += 1
        if not self.install_ready:
            return False
        self.install_ready = False
        self.reference_install_epoch += 1
        self.matching_reference_installed = True
        self._evidence.accept_current()
        return True

    def acquire_reference(
        self,
        _lowstate,
        *,
        executed_target,
        expected_install_epoch,
    ) -> None:
        assert expected_install_epoch == self.reference_install_epoch
        self.acquire_calls.append(tuple(float(value) for value in executed_target))
        self.reference_authority = True
        self.controller.playing = True
        self.controller.ref_cursor = 0

    def run_step(self, _action, _lowstate):
        assert self.reference_authority is True
        assert self.execution_feedback_pending is False
        if self.refresh_ready:
            self.refresh_ready = False
            self.reference_install_epoch += 1
            self._evidence.accept_current()
        self.run_calls += 1
        self.execution_feedback_pending = True
        self._candidate_identity = (
            self.reference_install_epoch,
            self.controller.ref_cursor,
        )
        return {
            f"joint_{index}.q": target
            for index, target in enumerate(self._target)
        }

    def record_executed_target(self, target) -> None:
        if not self.execution_feedback_pending:
            raise RuntimeError("no pending candidate")
        if self._candidate_identity != (
            self.reference_install_epoch,
            self.controller.ref_cursor,
        ):
            raise RuntimeError("candidate epoch/cursor mismatch")
        self.commits.append(tuple(float(value) for value in target))
        self.execution_feedback_pending = False
        if self.controller.playing and self.reference_authority:
            self.controller.ref_cursor += 1

    def revoke_reference_authority(
        self,
        *,
        discard_reference=True,
        discard_pending_execution=False,
    ) -> None:
        self.revoke_calls.append(
            (bool(discard_reference), bool(discard_pending_execution))
        )
        self.reference_authority = False
        if discard_reference:
            self.matching_reference_installed = False
        if discard_pending_execution:
            self.execution_feedback_pending = False
            self._candidate_identity = None


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


def test_reference_execution_is_transactional_across_grant_refresh_and_stop_race() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    target = tuple(value + 0.01 for value in boundary)
    lowstate = _lowstate(boundary)
    driver._last_published_targets = boundary
    driver._joint_handoff.freeze(boundary)
    driver._arm_guard.anchor(boundary[15:29])
    driver._reference_source_identity = (42, 7)
    driver._navigation_task_active = True
    driver._intent_gate = ReplanIntentGate()
    driver._decoder_facing_rad = None
    driver._pending_decoder_heading_delta_rad = 0.0
    driver._last_intent_signature = None

    class IntentGuard:
        last_received_at = clock.value

        @staticmethod
        def intent(**_kwargs):
            return _movement_intent(0.0)

        @staticmethod
        def stop(*, source="watchdog"):
            del source

    driver.guard = IntentGuard()
    evidence = _TransactionalEvidence()
    controller = _TransactionalController(evidence, target)
    driver._planner_evidence = evidence
    driver._controller = controller

    intent = driver._apply_intent(0.02, simulation_now_s=1.0)
    assert intent.mode == "SLOW_WALK"
    assert driver._reference_authority.snapshot.phase == "REQUIRED"
    assert driver._reference_authority.snapshot.source_sequence == 7
    assert controller.request_calls == controller.run_calls == 0

    # Planning-only wait requests/polls but never decodes or advances history.
    assert driver._progress_reference_transaction(lowstate) is False
    assert controller.request_calls == 1
    assert controller.poll_calls == 1
    assert controller.run_calls == 0
    assert controller.controller.ref_cursor == 0
    driver._publish_position_hold(boundary)
    assert controller.commits == []

    # A matched installation is rebased from the exact last DDS target and is
    # PRIMED, not health-visible actuator authority, until its own first write.
    controller.install_ready = True
    assert driver._progress_reference_transaction(lowstate) is True
    primed = driver._reference_authority.snapshot
    assert primed.phase == "PRIMED"
    assert primed.reference_epoch == 2
    assert primed.actuator_authority is False
    assert controller.acquire_calls == [boundary]
    driver._run_policy_candidate(lowstate)
    first = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=driver._reference_authority.executable,
        measured=boundary,
        policy=target,
        now=1.02,
    )
    published = driver._publish(
        _action(driver, first.targets),
        lowstate,
        control_dt_s=0.02,
        control_tick_ms=1_020,
        constrain_arms=False,
    )
    granted = driver._reference_authority.snapshot
    assert published == pytest.approx(target)
    assert controller.commits[-1] == pytest.approx(target)
    assert controller.controller.ref_cursor == 1
    assert granted.phase == "GRANTED"
    assert granted.reference_epoch == 2
    assert granted.dds_target_sequence == 2

    # A periodic result is installed before decode. The old epoch loses health
    # authority atomically, and only the new epoch's successful DDS write grants.
    controller.refresh_ready = True
    driver._run_policy_candidate(lowstate)
    assert controller.reference_install_epoch == 3
    refreshed = driver._reference_authority.snapshot
    assert refreshed.phase == "PRIMED"
    assert refreshed.reference_epoch == 3
    assert refreshed.actuator_authority is False
    driver._publish(
        _action(driver, target),
        lowstate,
        control_dt_s=0.02,
        control_tick_ms=1_040,
        constrain_arms=False,
    )
    assert driver._reference_authority.snapshot.phase == "GRANTED"
    assert driver._reference_authority.snapshot.reference_epoch == 3
    assert controller.controller.ref_cursor == 2

    # Stop wins after decode. The frozen target, not the discarded candidate,
    # is committed to executed history and revoked authority blocks cursor motion.
    driver._run_policy_candidate(lowstate)
    candidate_cursor = controller.controller.ref_cursor
    assert driver._freeze_last_published_hold() is True
    frozen = driver._safe_hold_targets
    assert frozen is not None
    driver._publish_position_hold(tuple(value - 0.5 for value in boundary))
    assert controller.commits[-1] == pytest.approx(frozen)
    assert controller.controller.ref_cursor == candidate_cursor
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert writes[-1] == pytest.approx(frozen)


def test_health_keeps_last_dds_grant_while_refresh_decode_is_inflight() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, _writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    lowstate = _lowstate(boundary)
    evidence = _TransactionalEvidence()
    evidence.intent_generation = 1
    evidence.intent_signature = "intent-1"
    evidence.accept_current()
    refresh_installed = threading.Event()
    release_decode = threading.Event()

    class BlockingRefreshController(_TransactionalController):
        def run_step(self, action, observed_lowstate):
            result = super().run_step(action, observed_lowstate)
            refresh_installed.set()
            assert release_decode.wait(timeout=2.0)
            return result

    controller = BlockingRefreshController(evidence, boundary)
    controller.reference_install_epoch = 2
    controller.reference_authority = True
    controller.refresh_ready = True
    driver._controller = controller
    driver._planner_evidence = evidence
    driver._reference_authority.require(generation=4, source_sequence=2)
    driver._reference_authority.mark_installed(
        generation=4,
        source_sequence=2,
        planner_intent_generation=1,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=4,
        source_sequence=2,
        reference_epoch=2,
    )
    driver._reference_authority.grant(
        generation=4,
        source_sequence=2,
        reference_epoch=2,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=1_000,
        dds_target_sequence=1,
    )
    decode_errors: list[Exception] = []
    probe_result: list[dict[str, object]] = []
    probe_complete = threading.Event()

    def decode() -> None:
        try:
            driver._run_policy_candidate(lowstate)
        except Exception as exc:
            decode_errors.append(exc)

    probe = sonic_health_probe(
        robot_id="g1-sim",
        producer_boot_id=UUID("20000000-0000-4000-8000-000000000002"),
        request_id=UUID("40000000-0000-4000-8000-000000000004"),
    )

    def read_health() -> None:
        response = driver.accept_json(probe)
        assert response is not None
        probe_result.append(response)
        probe_complete.set()

    decode_thread = threading.Thread(target=decode)
    probe_thread = threading.Thread(target=read_health)
    decode_thread.start()
    assert refresh_installed.wait(timeout=2.0)
    assert controller.reference_install_epoch == 3
    probe_thread.start()
    assert probe_complete.wait(timeout=0.50) is True
    assert probe_result[0]["reference_authority_state"] == "GRANTED"
    assert probe_result[0]["reference_authority_epoch"] == 2
    assert probe_result[0]["matched_reference_actuator_authority"] is True
    release_decode.set()
    decode_thread.join(timeout=2.0)
    probe_thread.join(timeout=2.0)

    assert decode_errors == []
    assert len(probe_result) == 1
    assert driver._reference_authority.snapshot.phase == "PRIMED"
    assert driver._reference_authority.snapshot.reference_epoch == 3


def test_stop_before_policy_run_consumes_deferred_revoke_and_skips_decode() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, _writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    evidence = _TransactionalEvidence()
    evidence.intent_generation = 1
    evidence.intent_signature = "intent-1"
    evidence.accept_current()
    controller = _TransactionalController(evidence, boundary)
    controller.reference_install_epoch = 2
    controller.reference_authority = True
    driver._controller = controller
    driver._planner_evidence = evidence
    driver._intent_gate = ReplanIntentGate()
    driver._navigation_task_active = True
    driver._last_published_targets = boundary
    driver._reference_authority.require(generation=4, source_sequence=2)
    driver._reference_authority.mark_installed(
        generation=4,
        source_sequence=2,
        planner_intent_generation=1,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=4,
        source_sequence=2,
        reference_epoch=2,
    )
    driver._reference_authority.grant(
        generation=4,
        source_sequence=2,
        reference_epoch=2,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=1_000,
        dds_target_sequence=1,
    )

    driver._stop_navigation_at_boundary(source="bridge_stop")
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert driver._vendor_revoke_pending is True
    assert controller.revoke_calls == []

    assert driver._run_policy_candidate(_lowstate(boundary)) == {}
    assert controller.run_calls == 0
    assert controller.revoke_calls == [(True, False)]
    assert driver._vendor_revoke_pending is False


def test_terminal_halt_does_not_wait_for_support_or_inflight_decode(
    monkeypatch,
) -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    target = tuple(value + 0.02 for value in boundary)
    lowstate = _lowstate(boundary)
    evidence = _TransactionalEvidence()
    evidence.intent_generation = 1
    evidence.intent_signature = "intent-1"
    evidence.accept_current()
    decode_entered = threading.Event()
    release_decode = threading.Event()

    class BlockingController(_TransactionalController):
        def run_step(self, action, observed_lowstate):
            result = super().run_step(action, observed_lowstate)
            decode_entered.set()
            assert release_decode.wait(timeout=2.0)
            return result

    controller = BlockingController(evidence, target)
    controller.reference_install_epoch = 2
    controller.reference_authority = True
    driver._controller = controller
    driver._planner_evidence = evidence
    driver._intent_gate = ReplanIntentGate()
    driver._navigation_task_active = True
    driver._last_published_targets = boundary
    driver._fault_trace = None
    driver._reference_authority.require(generation=4, source_sequence=2)
    driver._reference_authority.mark_installed(
        generation=4,
        source_sequence=2,
        planner_intent_generation=1,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=4,
        source_sequence=2,
        reference_epoch=2,
    )
    driver._reference_authority.grant(
        generation=4,
        source_sequence=2,
        reference_epoch=2,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=1_000,
        dds_target_sequence=1,
    )
    plant_entered = threading.Event()
    release_plant = threading.Event()
    plant_calls: list[str] = []

    def blocking_plant_halt(_host, _port, reason, **_kwargs):
        plant_calls.append(reason)
        plant_entered.set()
        assert release_plant.wait(timeout=2.0)
        return {"mode": "safety_stop", "safetyStopLatched": True}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        blocking_plant_halt,
    )
    decode_results: list[dict[str, object]] = []
    decode_errors: list[Exception] = []
    terminal_errors: list[Exception] = []

    def decode() -> None:
        try:
            decode_results.append(driver._run_policy_candidate(lowstate))
        except Exception as exc:
            decode_errors.append(exc)

    def terminal() -> None:
        try:
            driver._latch_terminal_fault("injected inflight runtime fault")
        except Exception as exc:
            terminal_errors.append(exc)

    decode_thread = threading.Thread(target=decode)
    terminal_thread = threading.Thread(target=terminal)
    decode_thread.start()
    assert decode_entered.wait(timeout=2.0)

    # Force the opportunistic local boundary to miss.  The terminal claim and
    # Plant request must still cross while this lock and ONNX are both blocked.
    driver._support_lock.acquire()
    try:
        terminal_thread.start()
        assert plant_entered.wait(timeout=0.50)
        assert driver._fault_latched.is_set()
        assert terminal_thread.is_alive()
    finally:
        driver._support_lock.release()

    # Plant HTTP remains blocked.  Returning inference must observe the fault,
    # revoke at its safe point, and discard the policy result before any DDS.
    release_decode.set()
    decode_thread.join(timeout=2.0)
    assert not decode_thread.is_alive()
    assert decode_errors == []
    assert decode_results == [{}]
    assert writes == []
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert controller.revoke_calls == [(True, False)]
    assert controller.controller.ref_cursor == 0
    assert controller.execution_feedback_pending is True

    # The only allowed DDS after the late candidate is the exact frozen
    # actuator boundary.  Revoked vendor authority records history but not a
    # reference cursor step.
    driver._publish_position_hold(target)
    assert writes == [pytest.approx(boundary)]
    assert controller.commits == [pytest.approx(boundary)]
    assert controller.controller.ref_cursor == 0
    assert controller.execution_feedback_pending is False
    assert driver._last_published_targets == pytest.approx(boundary)

    release_plant.set()
    terminal_thread.join(timeout=2.0)
    assert not terminal_thread.is_alive()
    assert terminal_errors == []
    assert plant_calls == ["injected inflight runtime fault"]


@pytest.mark.parametrize("publish_path", ["policy", "position_hold"])
def test_terminal_claim_before_final_write_replaces_target_with_boundary(
    monkeypatch,
    publish_path: str,
) -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    proposed = tuple(value + 0.025 for value in boundary)
    lowstate = _lowstate(boundary)
    evidence = _TransactionalEvidence()
    evidence.intent_generation = 1
    evidence.intent_signature = "intent-1"
    evidence.accept_current()
    controller = _TransactionalController(evidence, proposed)
    controller.reference_install_epoch = 2
    controller.reference_authority = True
    controller.run_step({}, lowstate)
    driver._controller = controller
    driver._planner_evidence = evidence
    driver._intent_gate = ReplanIntentGate()
    driver._navigation_task_active = True
    driver._last_published_targets = boundary
    driver._fault_trace = None
    driver._reference_authority.require(generation=9, source_sequence=3)
    driver._reference_authority.mark_installed(
        generation=9,
        source_sequence=3,
        planner_intent_generation=1,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=9,
        source_sequence=3,
        reference_epoch=2,
    )
    crc_entered = threading.Event()
    release_crc = threading.Event()
    crc_calls = 0

    def blocking_crc(_message) -> int:
        nonlocal crc_calls
        crc_calls += 1
        if crc_calls == 1:
            crc_entered.set()
            assert release_crc.wait(timeout=2.0)
        return 0

    driver._crc = SimpleNamespace(Crc=blocking_crc)
    plant_entered = threading.Event()
    release_plant = threading.Event()

    def blocking_plant_halt(*_args, **_kwargs):
        plant_entered.set()
        assert release_plant.wait(timeout=2.0)
        return {"mode": "safety_stop", "safetyStopLatched": True}

    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        blocking_plant_halt,
    )
    publish_results: list[object] = []
    publish_errors: list[Exception] = []
    terminal_errors: list[Exception] = []

    def publish() -> None:
        try:
            if publish_path == "policy":
                result = driver._publish(
                    _action(driver, proposed),
                    lowstate,
                    control_dt_s=0.02,
                    control_tick_ms=1_020,
                    constrain_arms=False,
                )
            else:
                result = driver._publish_position_hold(proposed)
            publish_results.append(result)
        except Exception as exc:
            publish_errors.append(exc)

    def terminal() -> None:
        try:
            driver._latch_terminal_fault("fault before final DDS check")
        except Exception as exc:
            terminal_errors.append(exc)

    publish_thread = threading.Thread(target=publish)
    terminal_thread = threading.Thread(target=terminal)
    publish_thread.start()
    assert crc_entered.wait(timeout=2.0)
    terminal_thread.start()
    assert plant_entered.wait(timeout=0.50)
    assert driver._fault_latched.is_set()
    release_crc.set()
    publish_thread.join(timeout=2.0)
    assert not publish_thread.is_alive()

    assert publish_errors == []
    assert writes == [pytest.approx(boundary)]
    assert controller.commits == [pytest.approx(boundary)]
    assert controller.controller.ref_cursor == 0
    assert controller.execution_feedback_pending is False
    assert driver._last_published_targets == pytest.approx(boundary)
    assert driver._safe_hold_targets == pytest.approx(boundary)
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    if publish_path == "policy":
        assert publish_results == [pytest.approx(boundary)]
    else:
        assert publish_results == [None]

    release_plant.set()
    terminal_thread.join(timeout=2.0)
    assert not terminal_thread.is_alive()
    assert terminal_errors == []


def test_terminal_claim_after_successful_write_records_actual_without_grant(
    monkeypatch,
) -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    actual = tuple(value + 0.03 for value in boundary)
    lowstate = _lowstate(boundary)
    evidence = _TransactionalEvidence()
    evidence.intent_generation = 1
    evidence.intent_signature = "intent-1"
    evidence.accept_current()
    controller = _TransactionalController(evidence, actual)
    controller.reference_install_epoch = 2
    controller.reference_authority = True
    controller.run_step({}, lowstate)
    driver._controller = controller
    driver._planner_evidence = evidence
    driver._intent_gate = ReplanIntentGate()
    driver._navigation_task_active = True
    driver._last_published_targets = boundary
    driver._fault_trace = None
    driver._reference_authority.require(generation=9, source_sequence=3)
    driver._reference_authority.mark_installed(
        generation=9,
        source_sequence=3,
        planner_intent_generation=1,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=9,
        source_sequence=3,
        reference_epoch=2,
    )
    write_crossed = threading.Event()
    plant_called = threading.Event()

    class BarrierPublisher:
        @staticmethod
        def Write(message) -> bool:
            writes.append(
                tuple(
                    float(message.motor_cmd[index].q)
                    for index in range(29)
                )
            )
            write_crossed.set()
            assert driver._fault_latched.wait(timeout=2.0)
            return True

    driver._publisher = BarrierPublisher()
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        lambda _host, _port, _reason, **_kwargs: (
            plant_called.set()
            or {"mode": "safety_stop", "safetyStopLatched": True}
        ),
    )
    publish_results: list[tuple[float, ...]] = []
    publish_errors: list[Exception] = []
    terminal_errors: list[Exception] = []

    def publish() -> None:
        try:
            publish_results.append(
                driver._publish(
                    _action(driver, actual),
                    lowstate,
                    control_dt_s=0.02,
                    control_tick_ms=1_020,
                    constrain_arms=False,
                )
            )
        except Exception as exc:
            publish_errors.append(exc)

    def terminal() -> None:
        try:
            driver._latch_terminal_fault("fault after DDS crossed")
        except Exception as exc:
            terminal_errors.append(exc)

    publish_thread = threading.Thread(target=publish)
    terminal_thread = threading.Thread(target=terminal)
    publish_thread.start()
    assert write_crossed.wait(timeout=2.0)
    terminal_thread.start()
    assert plant_called.wait(timeout=0.50)
    publish_thread.join(timeout=2.0)
    terminal_thread.join(timeout=2.0)

    assert not publish_thread.is_alive()
    assert not terminal_thread.is_alive()
    assert publish_errors == []
    assert terminal_errors == []
    assert publish_results == [pytest.approx(actual)]
    assert writes == [pytest.approx(actual)]
    assert driver._last_published_targets == pytest.approx(actual)
    assert driver._safe_hold_targets == pytest.approx(actual)
    assert driver._dds_target_sequence == 1
    assert controller.commits == [pytest.approx(actual)]
    assert controller.controller.ref_cursor == 0
    assert controller.execution_feedback_pending is False
    assert controller.revoke_calls == [(True, False)]
    authority = driver._reference_authority.snapshot
    assert authority.phase == "REVOKED"
    assert authority.actuator_authority is False
    assert authority.dds_target_sequence is None


def test_post_write_revoke_failure_still_records_actual_with_cursor_interlock(
    monkeypatch,
) -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    actual = tuple(value + 0.04 for value in boundary)
    lowstate = _lowstate(boundary)
    evidence = _TransactionalEvidence()
    evidence.intent_generation = 1
    evidence.intent_signature = "intent-1"
    evidence.accept_current()

    class BrokenRevokeController(_TransactionalController):
        def __init__(self, planner_evidence, target) -> None:
            super().__init__(planner_evidence, target)
            self.revoke_attempts = 0

        def revoke_reference_authority(
            self,
            *,
            discard_reference=True,
            discard_pending_execution=False,
        ) -> None:
            self.revoke_calls.append(
                (bool(discard_reference), bool(discard_pending_execution))
            )
            self.revoke_attempts += 1
            if self.revoke_attempts == 1:
                raise RuntimeError("injected vendor wrapper revoke failure")
            super().revoke_reference_authority(
                discard_reference=discard_reference,
                discard_pending_execution=discard_pending_execution,
            )

    controller = BrokenRevokeController(evidence, actual)
    controller.reference_install_epoch = 2
    controller.reference_authority = True
    controller.run_step({}, lowstate)
    driver._controller = controller
    driver._planner_evidence = evidence
    driver._intent_gate = ReplanIntentGate()
    driver._navigation_task_active = True
    driver._last_published_targets = boundary
    driver._fault_trace = None
    driver._reference_authority.require(generation=9, source_sequence=3)
    driver._reference_authority.mark_installed(
        generation=9,
        source_sequence=3,
        planner_intent_generation=1,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=9,
        source_sequence=3,
        reference_epoch=2,
    )
    write_crossed = threading.Event()

    class BarrierPublisher:
        @staticmethod
        def Write(message) -> bool:
            writes.append(
                tuple(
                    float(message.motor_cmd[index].q)
                    for index in range(29)
                )
            )
            write_crossed.set()
            assert driver._fault_latched.wait(timeout=2.0)
            return True

    driver._publisher = BarrierPublisher()
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        lambda *_args, **_kwargs: {
            "mode": "safety_stop",
            "safetyStopLatched": True,
        },
    )
    publish_errors: list[Exception] = []

    def publish() -> None:
        try:
            driver._publish(
                _action(driver, actual),
                lowstate,
                control_dt_s=0.02,
                control_tick_ms=1_020,
                constrain_arms=False,
            )
        except Exception as exc:
            publish_errors.append(exc)

    publish_thread = threading.Thread(target=publish)
    terminal_thread = threading.Thread(
        target=lambda: driver._latch_terminal_fault(
            "fault before broken revoke"
        )
    )
    publish_thread.start()
    assert write_crossed.wait(timeout=2.0)
    terminal_thread.start()
    publish_thread.join(timeout=2.0)
    terminal_thread.join(timeout=2.0)

    assert not publish_thread.is_alive()
    assert not terminal_thread.is_alive()
    assert len(publish_errors) == 1
    assert str(publish_errors[0]) == "injected vendor wrapper revoke failure"
    assert controller.controller.playing is False
    assert controller.reference_authority is False
    assert controller.commits == [pytest.approx(actual)]
    assert controller.controller.ref_cursor == 0
    assert controller.execution_feedback_pending is False
    assert driver._last_published_targets == pytest.approx(actual)
    assert driver._safe_hold_targets == pytest.approx(actual)
    assert driver._reference_authority.snapshot.phase == "REVOKED"

    # Once the wrapper recovers, the next terminal hold repeats the actual DDS
    # boundary rather than jumping back to the pre-fault target.
    driver._publisher = SimpleNamespace(
        Write=lambda message: (
            writes.append(
                tuple(
                    float(message.motor_cmd[index].q)
                    for index in range(29)
                )
            )
            or True
        )
    )
    driver._publish_position_hold(boundary)
    assert writes[-1] == pytest.approx(actual)
    assert driver._last_published_targets == pytest.approx(actual)
    assert driver._safe_hold_targets == pytest.approx(actual)
    assert controller.controller.ref_cursor == 0


def test_failed_dds_write_never_commits_or_grants_policy_or_hold_feedback() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, _writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    target = tuple(value + 0.01 for value in boundary)
    lowstate = _lowstate(boundary)
    evidence = _TransactionalEvidence()
    evidence.intent_generation = 1
    evidence.intent_signature = "intent-1"
    evidence.accept_current()
    controller = _TransactionalController(evidence, target)
    controller.reference_install_epoch = 2
    controller.reference_authority = True
    controller.run_step({}, lowstate)
    driver._controller = controller
    driver._planner_evidence = evidence
    driver._last_published_targets = boundary
    driver._reference_authority.require(generation=9, source_sequence=3)
    driver._reference_authority.mark_installed(
        generation=9,
        source_sequence=3,
        planner_intent_generation=1,
        reference_epoch=2,
    )
    driver._reference_authority.prime(
        generation=9,
        source_sequence=3,
        reference_epoch=2,
    )
    driver._publisher = SimpleNamespace(Write=lambda _message: False)

    with pytest.raises(RuntimeError, match="policy-target write failed"):
        driver._publish(
            _action(driver, target),
            lowstate,
            control_dt_s=0.02,
            control_tick_ms=1_020,
            constrain_arms=False,
        )

    assert driver._last_published_targets == boundary
    assert driver._dds_target_sequence == 0
    assert controller.execution_feedback_pending is True
    assert controller.commits == []
    assert controller.controller.ref_cursor == 0
    assert driver._reference_authority.snapshot.phase == "PRIMED"

    # The stop-race hold path is governed by the same successful-Write rule.
    driver._reference_authority.revoke("stop won")
    driver._safe_hold_targets = boundary
    with pytest.raises(RuntimeError, match="position-hold write failed"):
        driver._publish_position_hold(target)
    assert driver._last_published_targets == boundary
    assert driver._dds_target_sequence == 0
    assert controller.execution_feedback_pending is True
    assert controller.commits == []
    assert controller.controller.ref_cursor == 0


def test_outer_publish_preserves_real_vendor_candidate_epoch_cursor_proof() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES
    from lerobot.robots.unitree_g1.controllers.sonic_whole_body import (
        SonicRuntime,
        SonicWholeBodyController,
    )

    class Pipeline:
        execution_feedback_pending = False
        reference_authority = True

        def __init__(self, cursor: int) -> None:
            self.ref_cursor = cursor
            self.record_calls: list[tuple[float, ...]] = []
            self.advance_calls = 0

        def record_executed_target(self, target):
            committed = tuple(float(value) for value in target)
            self.record_calls.append(committed)
            return committed

        def advance_cursor(self) -> None:
            self.advance_calls += 1
            self.ref_cursor += 1

    def controller_for(*, pending_epoch: int, pending_cursor: int, epoch: int, cursor: int):
        pipeline = Pipeline(cursor)
        runtime = SonicRuntime.__new__(SonicRuntime)
        runtime.controller = pipeline
        runtime._policy_step_pending = True
        runtime._pending_reference_install_epoch = pending_epoch
        runtime._pending_ref_cursor = pending_cursor
        runtime._reference_install_epoch = epoch
        controller = SonicWholeBodyController.__new__(SonicWholeBodyController)
        controller._runtime = runtime
        controller.controller = pipeline
        return controller, runtime, pipeline

    clock = Clock()
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)

    stale_driver, stale_writes = _publishing_driver(clock)
    stale, stale_runtime, stale_pipeline = controller_for(
        pending_epoch=2,
        pending_cursor=61,
        epoch=3,
        cursor=0,
    )
    stale_driver._controller = stale
    with pytest.raises(RuntimeError, match="epoch/cursor"):
        stale_driver._publish_position_hold(boundary)
    assert len(stale_writes) == 1
    assert stale_pipeline.record_calls == []
    assert stale_pipeline.advance_calls == 0
    assert stale_pipeline.ref_cursor == 0
    assert stale_runtime.execution_feedback_pending is True

    matched_driver, matched_writes = _publishing_driver(clock)
    matched, matched_runtime, matched_pipeline = controller_for(
        pending_epoch=3,
        pending_cursor=0,
        epoch=3,
        cursor=0,
    )
    matched_driver._controller = matched
    matched_driver._publish_position_hold(boundary)
    assert len(matched_writes) == 1
    assert matched_pipeline.record_calls == [pytest.approx(boundary)]
    assert matched_pipeline.advance_calls == 1
    assert matched_pipeline.ref_cursor == 1
    assert matched_runtime.execution_feedback_pending is False


def test_stop_serializes_against_preplan_intent_and_cannot_be_resurrected() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, _writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    driver._last_published_targets = boundary
    driver._reference_source_identity = (42, 7)
    driver._navigation_task_active = True
    driver._decoder_facing_rad = None
    driver._pending_decoder_heading_delta_rad = 0.0
    driver._last_intent_signature = None
    evidence = _TransactionalEvidence()
    controller = _TransactionalController(evidence, boundary)
    driver._planner_evidence = evidence
    driver._controller = controller
    gate_entered = threading.Event()
    release_gate = threading.Event()
    stop_complete = threading.Event()
    inner_gate = ReplanIntentGate()

    class BlockingGate:
        @property
        def last(self):
            return inner_gate.last

        @staticmethod
        def admit(intent, *, now):
            gate_entered.set()
            assert release_gate.wait(timeout=2.0)
            return inner_gate.admit(intent, now=now)

        @staticmethod
        def reset() -> None:
            inner_gate.reset()

    class IntentGuard:
        last_received_at = clock.value

        @staticmethod
        def intent(**_kwargs):
            return _movement_intent(0.0)

        @staticmethod
        def stop(*, source="watchdog"):
            del source

    driver._intent_gate = BlockingGate()
    driver.guard = IntentGuard()
    control_errors: list[Exception] = []
    stop_errors: list[Exception] = []

    def apply_control() -> None:
        try:
            driver._apply_intent(0.02, simulation_now_s=1.0)
        except Exception as exc:
            control_errors.append(exc)

    def apply_stop() -> None:
        try:
            driver._stop_navigation_at_boundary(source="bridge_stop")
        except Exception as exc:
            stop_errors.append(exc)
        finally:
            stop_complete.set()

    control_thread = threading.Thread(target=apply_control)
    stop_thread = threading.Thread(target=apply_stop)
    control_thread.start()
    assert gate_entered.wait(timeout=2.0)
    stop_thread.start()
    assert stop_complete.wait(timeout=0.05) is False
    release_gate.set()
    control_thread.join(timeout=2.0)
    stop_thread.join(timeout=2.0)

    assert not control_thread.is_alive()
    assert not stop_thread.is_alive()
    assert control_errors == []
    assert stop_errors == []
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert driver._reference_request_pending is False
    assert driver._navigation_task_active is False
    assert driver._safe_hold_targets == boundary
    assert inner_gate.last is None
    assert controller.request_calls == controller.run_calls == 0


def test_ingress_accepted_stop_revokes_before_control_can_cross_boundary() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, _writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    driver._last_published_targets = boundary
    driver._reference_source_identity = (42, 7)
    driver._navigation_task_active = True
    driver._intent_gate = ReplanIntentGate()
    driver._intent_gate.admit(_movement_intent(0.0), now=0.5)
    driver._decoder_facing_rad = None
    driver._pending_decoder_heading_delta_rad = 0.0
    driver._last_intent_signature = None
    evidence = _TransactionalEvidence()
    controller = _TransactionalController(evidence, boundary)
    driver._planner_evidence = evidence
    driver._controller = controller
    ingress_entered = threading.Event()
    release_ingress = threading.Event()
    control_complete = threading.Event()

    class BlockingIngress:
        @staticmethod
        def accept(_payload):
            ingress_entered.set()
            assert release_ingress.wait(timeout=2.0)
            return SimpleNamespace(
                action="stop",
                state="IDLE",
                reason="cancelled",
                heading_yaw=None,
            )

    class IntentGuard:
        last_received_at = clock.value
        moving = True

        def intent(self, **_kwargs):
            return _movement_intent(0.0) if self.moving else MovementIntent(
                mode="IDLE",
                speed_mps=0.0,
                movement_angle_rad=0.0,
                facing_angle_rad=0.0,
                has_movement=False,
                stale=False,
            )

        def stop(self, *, source="watchdog") -> None:
            del source
            self.moving = False

    driver._intent_ingress = BlockingIngress()
    driver.guard = IntentGuard()
    accept_errors: list[Exception] = []
    control_errors: list[Exception] = []

    def accept_stop() -> None:
        try:
            driver.accept_json({"type": "sonic_intent"})
        except Exception as exc:
            accept_errors.append(exc)

    def apply_control() -> None:
        try:
            driver._apply_intent(0.02, simulation_now_s=1.0)
        except Exception as exc:
            control_errors.append(exc)
        finally:
            control_complete.set()

    accept_thread = threading.Thread(target=accept_stop)
    control_thread = threading.Thread(target=apply_control)
    accept_thread.start()
    assert ingress_entered.wait(timeout=2.0)
    control_thread.start()
    assert control_complete.wait(timeout=0.05) is False
    release_ingress.set()
    accept_thread.join(timeout=2.0)
    control_thread.join(timeout=2.0)

    assert accept_errors == []
    assert control_errors == []
    assert driver._reference_authority.snapshot.phase == "REVOKED"
    assert driver._reference_request_pending is False
    assert driver._navigation_task_active is False
    assert driver._safe_hold_targets == boundary
    assert controller.request_calls == controller.run_calls == 0


def test_new_generation_same_motion_is_replanned_without_an_idle_control_tick() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, _writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    driver._last_published_targets = boundary
    driver._intent_gate = ReplanIntentGate()
    driver._decoder_facing_rad = None
    driver._pending_decoder_heading_delta_rad = 0.0
    driver._last_intent_signature = None
    evidence = _TransactionalEvidence()
    controller = _TransactionalController(evidence, boundary)
    driver._planner_evidence = evidence
    driver._controller = controller

    class TaskGuard:
        last_received_at = clock.value
        moving = True

        def intent(self, **_kwargs):
            return _movement_intent(0.0) if self.moving else MovementIntent(
                mode="IDLE",
                speed_mps=0.0,
                movement_angle_rad=0.0,
                facing_angle_rad=0.0,
                has_movement=False,
                stale=False,
            )

        def stop(self, *, source="watchdog") -> None:
            del source
            self.moving = False

    guard = TaskGuard()
    driver.guard = guard
    driver._reference_source_identity = (10, 4)
    driver._navigation_task_active = True
    driver._apply_intent(0.02, simulation_now_s=1.0)
    assert driver._reference_authority.snapshot.phase == "REQUIRED"
    assert driver._reference_authority.snapshot.generation == 10

    driver._stop_navigation_at_boundary(source="bridge_stop")
    assert driver._intent_gate.last is None
    guard.moving = True
    driver._reference_source_identity = (11, 1)
    driver._navigation_task_active = True
    driver._apply_intent(0.02, simulation_now_s=1.02)

    required = driver._reference_authority.snapshot
    assert required.phase == "REQUIRED"
    assert (required.generation, required.source_sequence) == (11, 1)
    assert driver._reference_request_pending is True
    assert controller.request_calls == controller.run_calls == 0


def test_arm_verifier_runtime_fault_halts_plant_without_normal_safety_stop(
    monkeypatch,
) -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    boundary = tuple(float(value) for value in DEFAULT_ANGLES)
    invalid = list(boundary)
    invalid[15] += 3.0
    invalid = tuple(invalid)
    lowstate = _lowstate(boundary)
    driver._last_published_targets = boundary
    driver._arm_guard.anchor(boundary[15:29])
    driver._fault_trace = None
    driver._startup_qualifying = False
    driver._pending_decoder_heading_delta_rad = 0.0
    support_modes: list[str] = []
    static_decisions: list[bool] = []
    terminal_halts: list[str] = []
    original_static_commit = driver._commit_static_hold_decision_locked

    def observed_static_commit(decision: bool) -> bool:
        static_decisions.append(decision)
        return original_static_commit(decision)

    driver._commit_static_hold_decision_locked = (  # type: ignore[method-assign]
        observed_static_commit
    )

    class RuntimeController:
        execution_feedback_pending = False
        reference_install_epoch = 1
        matching_reference_installed = True

        def __init__(self) -> None:
            self.ms = SimpleNamespace(speed=0.2)
            self.controller = SimpleNamespace(
                ref_cursor=0,
                delta_heading=0.0,
            )
            self.shutdown_called = False

        @staticmethod
        def run_step(_action, _lowstate):
            return _action_for_invalid

        @staticmethod
        def revoke_reference_authority(**_kwargs) -> None:
            return None

        def shutdown(self) -> None:
            self.shutdown_called = True

    _action_for_invalid = _action(driver, invalid)
    controller = RuntimeController()
    driver._controller = controller
    driver._reference_authority.require(generation=1, source_sequence=1)
    driver._reference_authority.mark_installed(
        generation=1,
        source_sequence=1,
        planner_intent_generation=1,
        reference_epoch=1,
    )
    driver._reference_authority.prime(
        generation=1,
        source_sequence=1,
        reference_epoch=1,
    )
    driver._reference_authority.grant(
        generation=1,
        source_sequence=1,
        reference_epoch=1,
        granted_at_monotonic_s=clock.value,
        granted_at_tick_ms=1_000,
        dds_target_sequence=1,
    )
    driver._prepare_idle_startup = lambda: None
    samples = [(lowstate, PlantControlStep(tick_ms=1_020, dt_s=0.02), 0)]
    driver._next_control_sample = lambda: samples.pop(0)
    driver._apply_intent = lambda *_args, **_kwargs: _movement_intent(0.0)
    driver._progress_reference_transaction = lambda _lowstate: True
    driver._prime_periodic_reference_refresh = lambda: None
    driver._joint_handoff = SimpleNamespace(
        decide=lambda **kwargs: SimpleNamespace(
            targets=tuple(kwargs["policy"]),
            static_hold_active=False,
            handoff_ratio=1.0,
            motion_handoff_active=False,
        ),
        freeze=lambda _targets: None,
    )
    driver._set_plant_support_mode = lambda mode: (
        support_modes.append(mode) or True
    )
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.latch_plant_physics_fault",
        lambda _host, _port, reason, **_kwargs: (
            terminal_halts.append(reason)
            or {"mode": "safety_stop", "safetyStopLatched": True}
        ),
    )

    driver.run()

    assert driver._fault_latched.is_set()
    assert "arm policy target" in str(driver._terminal_fault_reason)
    assert terminal_halts == [driver._terminal_fault_reason]
    assert "safety_stop" not in support_modes
    assert support_modes == ["locomotion"]
    assert static_decisions == [False]
    assert len(writes) == 1
    assert writes[0] == pytest.approx(boundary)
    assert controller.shutdown_called is True


def test_actual_29dof_dds_targets_are_continuous_at_start_replan_and_stop() -> None:
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES

    clock = Clock()
    driver, writes = _publishing_driver(clock)
    initial = tuple(float(value) for value in DEFAULT_ANGLES)
    measured = _lowstate(initial, arm_lag_rad=0.50)
    policy = tuple(value + 0.02 for value in initial)
    driver._last_published_targets = initial

    # A granted reference crosses the real publish path unchanged.  The exact
    # frozen boundary seeds the verifier even though measured arms lag by 0.5rad.
    assert driver._freeze_planner_handoff_boundary() is True
    first = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=tuple(state.q for state in measured.motor_state),
        policy=policy,
        now=1.0,
    )
    assert first.handoff_ratio == 1.0
    published = driver._publish(
        _action(driver, first.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert published == pytest.approx(policy)
    assert writes[-1] == pytest.approx(policy)

    # A mid-gait required generation freezes the last actual DDS command while
    # waiting. Once authority is granted, the policy owns the complete target;
    # no post-policy blend creates an unrecorded synthetic action.
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
    assert admitted.handoff_ratio == 1.0
    assert admitted_published == pytest.approx(policy)
    assert writes[-1] == pytest.approx(policy)

    continued = driver._joint_handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=tuple(state.q for state in measured.motor_state),
        policy=policy,
        now=2.47,
    )
    continued_published = driver._publish(
        _action(driver, continued.targets),
        measured,
        control_dt_s=0.02,
        constrain_arms=True,
    )
    assert continued.handoff_ratio == 1.0
    assert continued_published == pytest.approx(continued.targets)
    assert writes[-1] == pytest.approx(continued.targets)

    # A stop racing the policy path freezes and republishes the exact last DDS
    # vector rather than any measured position.
    assert driver._freeze_last_published_hold() is True
    stopped_measurement = _lowstate(
        tuple(value - 0.30 for value in continued_published)
    )
    driver._publish_frozen_hold(stopped_measurement)
    assert writes[-1] == pytest.approx(continued_published)
    assert driver._last_published_targets == pytest.approx(continued_published)


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

    # The matching -X generation transfers binary authority to the exact policy
    # target; the actuator verifier observes it but does not blend or rewrite it.
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
    assert admitted.handoff_ratio == 1.0
    assert admitted_published == pytest.approx(admitted.targets)

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


def test_required_planner_deadline_latches_terminal_exact_target_hold(
    block_live_driver_plant_writes,
) -> None:
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
    assert [call[0] for call in block_live_driver_plant_writes] == [
        "latch_plant_physics_fault"
    ]


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
    monkeypatch.setattr(
        "hide_mujoco_sonic.sonic_driver.set_plant_support",
        lambda _host, _port, mode, **_kwargs: {"mode": mode},
    )
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
    confirm_correlated_driver_stop(adapter, driver, clock)

    clock.value += 0.01
    same_generation = _intent_payload(
        adapter.tick(now=clock.value), measured_heading_yaw=0.0
    )
    same_generation["control_arm"] = True
    with pytest.raises(ValueError, match="stopped, acknowledged IDLE"):
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
    confirm_correlated_driver_stop(adapter, driver, clock)
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
    confirm_correlated_driver_stop(adapter, driver, clock)
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
    with pytest.raises(RuntimeError, match="current exact DRIVER_CONFIRMED"):
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
    confirm_correlated_driver_stop(adapter, driver, clock)

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
    # This test isolates an already-authorized support transition.  Model the
    # matching-reference rebase that releases the earlier IDLE actuator hold.
    # The still-frozen case is covered by the normal-stop late-tick tests.
    driver._safe_hold_targets = None
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
