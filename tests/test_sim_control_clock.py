from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import threading

import numpy as np

from hide_mujoco_sonic.safety import PlantControlClock, PlantTickGuard
from hide_mujoco_sonic.sonic_driver import SonicDriver


class _Guard:
    def __init__(self) -> None:
        self.stops: list[str] = []

    def stop(self, *, source: str = "watchdog") -> None:
        self.stops.append(source)


class _Telemetry:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, **values: object) -> None:
        self.values.update(values)


def _bare_driver(
    *,
    wall_now: list[float] | None = None,
    control_clock_armed: bool = True,
) -> SonicDriver:
    driver = SonicDriver.__new__(SonicDriver)
    driver.guard = _Guard()
    driver.telemetry = _Telemetry()
    driver._lowstate = None
    driver._lowstate_lock = threading.RLock()
    driver._control_condition = threading.Condition(driver._lowstate_lock)
    driver._control_clock = PlantControlClock(control_hz=50)
    driver._control_clock_armed = control_clock_armed
    driver._control_consumer_ready = threading.Event()
    driver._motion_ready = threading.Event()
    driver._startup_qualifying = False
    driver._startup_qualification_required_steps = 5
    driver._startup_qualification_consecutive_steps = 0
    driver._startup_qualification_dropped_samples = 0
    driver._startup_qualification_drop_generation = 0
    if control_clock_armed:
        driver._control_consumer_ready.set()
        driver._motion_ready.set()
    driver._pending_control_sample = None
    clock_value = [10.0] if wall_now is None else wall_now
    driver._wall_clock = lambda: clock_value[0]
    driver._control_clock_armed_at_wall_s = clock_value[0]
    driver._last_lowstate_wall_s = None
    driver._lowstate_stall_timeout_s = 1.0
    driver._tick_guard = PlantTickGuard()
    driver._fault_latched = threading.Event()
    driver._terminal_fault_reason = None
    driver._navigation_task_active = False
    driver._stop = threading.Event()
    return driver


def test_driver_consumes_exactly_one_admitted_simulation_step() -> None:
    driver = _bare_driver()
    first_message = SimpleNamespace(tick=1_000)
    driver._on_lowstate(first_message)

    first = driver._next_control_sample(wait_timeout_s=0.0)
    assert first is not None
    assert first[0] is first_message
    assert first[1].tick_ms == 1_000
    assert first[1].dt_s == 0.02

    for tick in (1_004, 1_008, 1_012, 1_016):
        driver._on_lowstate(SimpleNamespace(tick=tick))
        assert driver._next_control_sample(wait_timeout_s=0.0) is None

    driver._on_lowstate(SimpleNamespace(tick=1_020))
    second = driver._next_control_sample(wait_timeout_s=0.0)
    assert second is not None
    assert second[1].tick_ms == 1_020
    assert second[1].dt_s == 0.02


def test_driver_fails_closed_instead_of_catching_up_pending_policy_steps() -> None:
    driver = _bare_driver()
    driver._on_lowstate(SimpleNamespace(tick=1_000))
    for tick in (1_004, 1_008, 1_012, 1_016, 1_020):
        driver._on_lowstate(SimpleNamespace(tick=tick))

    assert driver._fault_latched.is_set()
    assert driver.guard.stops == ["terminal_fault"]
    assert "control sample overrun" in str(driver.telemetry.values["last_error"])


def test_driver_fails_closed_on_duplicate_simulation_tick() -> None:
    driver = _bare_driver()
    driver._on_lowstate(SimpleNamespace(tick=1_000))
    driver._on_lowstate(SimpleNamespace(tick=1_000))

    assert driver._fault_latched.is_set()
    assert "control clock duplicate" in str(driver.telemetry.values["last_error"])


def test_driver_wall_watchdog_only_faults_and_never_creates_catchup_step() -> None:
    wall_now = [10.0]
    driver = _bare_driver(wall_now=wall_now)
    driver._on_lowstate(SimpleNamespace(tick=1_000))
    assert driver._next_control_sample(wait_timeout_s=0.0) is not None

    wall_now[0] = 11.01
    assert driver._next_control_sample(wait_timeout_s=0.0) is None
    assert driver._fault_latched.is_set()
    assert "LowState stream stalled" in str(driver.telemetry.values["last_error"])


def test_cold_start_discards_baseline_then_waits_before_arming_clock() -> None:
    driver = _bare_driver(control_clock_armed=False)
    driver._on_lowstate(SimpleNamespace(tick=3_008))
    consumed = []

    consumer = threading.Thread(
        target=lambda: consumed.append(
            driver._next_control_sample(wait_timeout_s=1.0)
        )
    )
    consumer.start()
    assert driver._control_consumer_ready.wait(timeout=0.5)

    # 3008 ms is a baseline only.  The first qualification step is 3028 ms;
    # there is no queued startup sample that can race with it.
    for tick in (3_012, 3_016, 3_020, 3_024, 3_028):
        driver._on_lowstate(SimpleNamespace(tick=tick))
    consumer.join(timeout=0.5)

    assert consumer.is_alive() is False
    assert driver._fault_latched.is_set() is False
    assert len(consumed) == 1
    assert consumed[0] is not None
    assert consumed[0][1].tick_ms == 3_028


def test_controller_warmup_runs_full_path_on_isolated_state() -> None:
    calls = []

    @dataclass
    class _MovementState:
        mode: int = 0
        speed: float = -1.0

    class _Pipeline:
        def __init__(self) -> None:
            self.encoder = object()
            self.decoder = object()
            self.planner = object()
            self.motion_lock = threading.Lock()
            self.history = [np.zeros(2, dtype=np.float32)]
            self.ref_cursor = 0

    class _Runtime:
        def __init__(self, pipeline, movement_state) -> None:
            self.controller = pipeline
            self.planner = object()
            self.ms = movement_state
            self.step = 0
            self.replan_timer = 0.0

    class _Controller:
        def __init__(self) -> None:
            pipeline = _Pipeline()
            movement_state = _MovementState()
            self._runtime = _Runtime(pipeline, movement_state)
            self.controller = pipeline
            self.ms = movement_state
            self._smpl_stream = None
            self._init_step = 0

        def run_step(self, action, lowstate):
            assert action == {}
            calls.append(self)
            self._runtime.step += 1
            self._init_step += 1
            self.controller.ref_cursor += 1
            self.controller.history[0][0] = float(lowstate.tick)
            self.ms.speed = 0.2
            return {f"joint_{index}.q": float(index) for index in range(29)}

    wall_now = [10.0]
    driver = _bare_driver(wall_now=wall_now, control_clock_armed=False)
    driver._controller = _Controller()
    lowstate = SimpleNamespace(tick=123)
    before = driver._policy_state_digest(driver._controller)

    driver._warm_controller_inference(lowstate)

    assert len(calls) == 1
    assert calls[0] is not driver._controller
    assert driver._controller._runtime.step == 0
    assert driver._controller._init_step == 0
    assert driver._controller.controller.ref_cursor == 0
    assert driver._controller.controller.history[0].tolist() == [0.0, 0.0]
    assert driver._controller.ms.speed == -1.0
    assert driver._policy_state_digest(driver._controller) == before
    assert driver._control_clock_armed is False
    assert driver._control_consumer_ready.is_set() is False
    assert driver._motion_ready.is_set() is False
    assert driver.telemetry.values["controller_warmup_complete"] is True
    assert driver.telemetry.values["controller_warmup_state_unchanged"] is True
    assert driver.telemetry.values["control_clock_status"] == "warm_ready_to_qualify"


def test_startup_qualification_latest_only_syncs_without_policy_catchup() -> None:
    driver = _bare_driver(control_clock_armed=False)
    driver._startup_qualification_required_steps = 3
    driver._lowstate = SimpleNamespace(tick=1_000)

    # Arm against a discarded baseline. No policy sample exists yet.
    assert driver._next_control_sample(wait_timeout_s=0.0) is None
    assert driver._startup_qualifying is True
    assert driver._motion_ready.is_set() is False

    # The consumer is deliberately late. During qualification only, the stale
    # stationary measurement is replaced by the latest one and never replayed.
    driver._on_lowstate(SimpleNamespace(tick=1_020))
    driver._on_lowstate(SimpleNamespace(tick=1_040))
    assert driver._fault_latched.is_set() is False
    assert driver._startup_qualification_dropped_samples == 1
    first = driver._next_control_sample(wait_timeout_s=0.0)
    assert first is not None
    assert first[1].tick_ms == 1_040
    driver._complete_startup_qualification_step(
        sample_drop_generation=first[2]
    )
    assert driver._startup_qualification_consecutive_steps == 1

    for tick in (1_060, 1_080):
        driver._on_lowstate(SimpleNamespace(tick=tick))
        sample = driver._next_control_sample(wait_timeout_s=0.0)
        assert sample is not None
        driver._complete_startup_qualification_step(
            sample_drop_generation=sample[2]
        )

    assert driver._motion_ready.is_set() is True
    assert driver._startup_qualifying is False
    assert driver.telemetry.values["motion_ready"] is True

    # Once ready, precisely the same pending-sample condition is terminal.
    driver._on_lowstate(SimpleNamespace(tick=1_100))
    driver._on_lowstate(SimpleNamespace(tick=1_120))
    assert driver._fault_latched.is_set() is True
    assert "control sample overrun" in str(driver.telemetry.values["last_error"])


def test_qualification_step_gets_no_credit_if_processing_was_overrun() -> None:
    driver = _bare_driver(control_clock_armed=False)
    driver._startup_qualification_required_steps = 2
    driver._lowstate = SimpleNamespace(tick=2_000)
    assert driver._next_control_sample(wait_timeout_s=0.0) is None

    driver._on_lowstate(SimpleNamespace(tick=2_020))
    sample = driver._next_control_sample(wait_timeout_s=0.0)
    assert sample is not None

    # While this sample is being handled, one next sample may wait; a second
    # causes latest-only replacement and invalidates the current credit.
    driver._on_lowstate(SimpleNamespace(tick=2_040))
    driver._on_lowstate(SimpleNamespace(tick=2_060))
    driver._complete_startup_qualification_step(
        sample_drop_generation=sample[2]
    )

    assert driver._fault_latched.is_set() is False
    assert driver._startup_qualification_consecutive_steps == 0
    assert driver._motion_ready.is_set() is False


def test_startup_holds_plant_before_warmup_and_leaves_clock_disarmed() -> None:
    driver = _bare_driver(control_clock_armed=False)
    lowstate = SimpleNamespace(tick=3_008)
    driver._lowstate = lowstate
    events = []

    def support(mode: str) -> bool:
        events.append(("support", mode))
        return True

    driver._set_plant_support_mode = support
    driver._publish_measured_hold = lambda state: events.append(("hold", state))
    driver._warm_controller_inference = lambda state: events.append(("warm", state))

    driver._prepare_idle_startup()

    assert events == [
        ("support", "idle"),
        ("hold", lowstate),
        ("warm", lowstate),
    ]
    assert driver._control_clock_armed is False
    assert driver._control_consumer_ready.is_set() is False
    assert driver._motion_ready.is_set() is False
