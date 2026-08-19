from __future__ import annotations

import math
import threading
from types import SimpleNamespace

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
from hide_mujoco_sonic.safety import PlantControlClock, PlantTickGuard
from hide_mujoco_sonic.sonic_driver import SonicDriver, SonicIntentIngress


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
    roll: float = 0.0,
) -> BaseObservation:
    return BaseObservation(
        generation=generation,
        timestamp_s=clock.value,
        x=0.0,
        y=0.0,
        yaw=0.0,
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
        reverse_hold_s=0.05,
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
            observation(clock, vx=-0.08), now=clock.value
        )
        clock.value += 0.06
        intent = adapter.ingest_observation(
            observation(clock, vx=-0.08), now=clock.value
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

    def accept(self, vx, vy, wz, *, source):
        self.accepted.append((vx, vy, wz, source))

    def stop(self, *, source="watchdog"):
        self.stops.append(source)

    def observe_heading(self, yaw):
        assert math.isfinite(yaw)


class FakeTelemetry:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, **values):
        self.values.update(values)

    def reset_navigation_task_peaks(self):
        pass


def bare_driver(clock: Clock) -> SonicDriver:
    driver = SonicDriver.__new__(SonicDriver)
    driver.config = SimpleNamespace(plant_port=8794)
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
    driver._last_lowstate_wall_s = None
    driver._lowstate_stall_timeout_s = 1.0
    driver._stop = threading.Event()
    driver._support_lock = threading.RLock()
    driver._plant_support_mode = "idle"
    driver._plant_support_stop_pending = False
    driver._plant_support_safety_latched = False
    driver._plant_support_stop_generation = None
    driver._joint_handoff = SimpleNamespace(reset=lambda: None)
    driver._arm_guard = SimpleNamespace(reset=lambda: None)
    return driver


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
    with pytest.raises(RuntimeError, match="rearm failed"):
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


def test_driver_rejects_raw_cmd_vel_and_fails_closed() -> None:
    clock = Clock()
    driver = bare_driver(clock)
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
    assert driver.guard.stops[-1] == "invalid_gait_intent"


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
