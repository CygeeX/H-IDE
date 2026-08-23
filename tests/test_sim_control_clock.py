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

    def update_hot(self, **values: object) -> None:
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
    driver._lowstate_sequence = 0
    driver._runtime_clock_committed = control_clock_armed
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
    driver._support_lock = threading.RLock()
    driver._last_published_targets = None
    driver._safe_hold_targets = None
    driver._joint_handoff = SimpleNamespace(freeze=lambda _targets: None)
    driver._arm_guard = SimpleNamespace(
        reset=lambda: None,
        anchor=lambda _targets: None,
    )
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


def test_driver_observes_lowstate_and_successful_publish_gaps() -> None:
    driver = _bare_driver()
    lowstates = []
    publishes = []
    driver._runtime_diagnostics = SimpleNamespace(
        observe_lowstate=lowstates.append,
        observe_control_publish=publishes.append,
    )

    driver._on_lowstate(SimpleNamespace(tick=1_000))
    driver._mark_control_step_published()

    assert lowstates == [10.0]
    assert publishes == [10.0]
    assert driver._last_control_step_wall_s == 10.0


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
    assert calls[0]._runtime.auto_reference_authority is True
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


def test_startup_qualification_rebases_forward_gap_then_requires_fresh_window() -> None:
    driver = _bare_driver(control_clock_armed=False)
    driver._startup_qualification_required_steps = 2
    driver._lowstate = SimpleNamespace(tick=1_000)
    assert driver._next_control_sample(wait_timeout_s=0.0) is None

    # A long model warmup starved the Python DDS callback while Plant remained
    # held and kept advancing simulation time.  This observation is a new
    # baseline, never a catch-up policy step or qualification credit.
    driver._on_lowstate(SimpleNamespace(tick=2_836))
    assert driver._fault_latched.is_set() is False
    assert driver._pending_control_sample is None
    assert driver._startup_qualification_dropped_samples == 1
    assert driver._startup_qualification_consecutive_steps == 0
    assert driver._control_clock.last_observed_tick_ms == 2_836
    assert driver._control_clock.last_control_tick_ms == 2_836

    for tick in (2_840, 2_844, 2_848, 2_852, 2_856):
        driver._on_lowstate(SimpleNamespace(tick=tick))
    first = driver._next_control_sample(wait_timeout_s=0.0)
    assert first is not None
    assert first[1].tick_ms == 2_856
    driver._complete_startup_qualification_step(
        sample_drop_generation=first[2]
    )
    assert driver._motion_ready.is_set() is False

    for tick in (2_860, 2_864, 2_868, 2_872, 2_876):
        driver._on_lowstate(SimpleNamespace(tick=tick))
    second = driver._next_control_sample(wait_timeout_s=0.0)
    assert second is not None
    driver._complete_startup_qualification_step(
        sample_drop_generation=second[2]
    )
    assert driver._motion_ready.is_set() is True
    assert driver._startup_qualifying is False

    # Once the first control-arm transaction commits, the exact same
    # discontinuity is terminal for the lifetime of the process.
    driver._runtime_clock_committed = True
    driver._on_lowstate(SimpleNamespace(tick=4_712))
    assert driver._fault_latched.is_set() is True
    assert "control clock source jumped 1836ms" in str(
        driver.telemetry.values["last_error"]
    )


def test_pre_authority_idle_gap_revokes_readiness_and_requalifies() -> None:
    driver = _bare_driver(control_clock_armed=False)
    driver._startup_qualification_required_steps = 1
    driver._lowstate = SimpleNamespace(tick=1_000)
    assert driver._next_control_sample(wait_timeout_s=0.0) is None

    driver._on_lowstate(SimpleNamespace(tick=1_020))
    sample = driver._next_control_sample(wait_timeout_s=0.0)
    assert sample is not None
    assert driver._complete_startup_qualification_step(
        sample_drop_generation=sample[2]
    ) is True
    assert driver._motion_ready.is_set() is True

    driver._on_lowstate(SimpleNamespace(tick=2_780))

    assert driver._fault_latched.is_set() is False
    assert driver._runtime_clock_committed is False
    assert driver._motion_ready.is_set() is False
    assert driver._startup_qualifying is True
    assert driver.telemetry.values["motion_ready"] is False
    assert driver.telemetry.values["control_clock_status"] == "startup_requalification"
    assert driver._control_clock.last_observed_tick_ms == 2_780
    assert driver._pending_control_sample is None


def test_post_warm_drain_requires_a_fresh_bounded_lowstate_window() -> None:
    driver = _bare_driver(control_clock_armed=False)
    driver._lowstate = SimpleNamespace(tick=1_000)
    result = []

    waiter = threading.Thread(
        target=lambda: result.append(
            driver._wait_for_post_warm_lowstate_stream(timeout_s=1.0)
        )
    )
    waiter.start()
    for _ in range(100):
        if (
            driver.telemetry.values.get("control_clock_status")
            == "waiting_for_fresh_post_warm_lowstate"
        ):
            break
        threading.Event().wait(0.001)

    # The first callback is a stale-backlog jump and earns no continuity.
    # Six subsequent 4 ms samples prove five fresh bounded intervals.
    for tick in (2_760, 2_764, 2_768, 2_772, 2_776, 2_780, 2_784):
        driver._on_lowstate(SimpleNamespace(tick=tick))
        threading.Event().wait(0.005)

    waiter.join(timeout=0.5)
    assert waiter.is_alive() is False
    assert result and result[0].tick == 2_780
    assert driver._control_clock_armed is False
    assert driver._motion_ready.is_set() is False


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


def test_qualification_commits_only_with_no_newer_sample_pending() -> None:
    driver = _bare_driver(control_clock_armed=False)
    driver._startup_qualification_required_steps = 1
    driver._lowstate = SimpleNamespace(tick=3_000)
    assert driver._next_control_sample(wait_timeout_s=0.0) is None

    driver._on_lowstate(SimpleNamespace(tick=3_020))
    sample = driver._next_control_sample(wait_timeout_s=0.0)
    assert sample is not None

    # Reproduce the live startup race: the next stationary sample arrives
    # while the final qualification hold is being published.  It has not been
    # dropped or overwritten, so the generation alone cannot detect it.
    driver._on_lowstate(SimpleNamespace(tick=3_040))
    assert driver._complete_startup_qualification_step(
        sample_drop_generation=sample[2]
    ) is False
    assert driver._fault_latched.is_set() is False
    assert driver._startup_qualifying is True
    assert driver._motion_ready.is_set() is False
    assert driver._startup_qualification_consecutive_steps == 0

    # Consuming that queued stationary sample at a clean boundary may qualify.
    latest = driver._next_control_sample(wait_timeout_s=0.0)
    assert latest is not None
    assert latest[1].tick_ms == 3_040
    assert driver._complete_startup_qualification_step(
        sample_drop_generation=latest[2]
    ) is True
    assert driver._startup_qualifying is False
    assert driver._motion_ready.is_set() is True


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
    driver._controller = SimpleNamespace(
        initialize_from_lowstate=lambda state: (
            events.append(("initialize", state)) or True
        )
    )
    driver._warm_controller_inference = lambda state: events.append(("warm", state))
    driver._runtime_diagnostics = SimpleNamespace(
        freeze_warm_runtime=lambda: events.append(("gc_freeze", None)),
        snapshot=lambda: {
            "python_gc_freeze_applied": True,
            "python_gc_warmup_collected_objects": 11,
            "python_gc_freeze_error": None,
        },
    )
    driver._wait_for_post_warm_lowstate_stream = lambda: events.append(
        ("fresh", None)
    )

    driver._prepare_idle_startup()

    assert events == [
        ("support", "idle"),
        ("hold", lowstate),
        ("initialize", lowstate),
        ("warm", lowstate),
        ("gc_freeze", None),
        ("fresh", None),
    ]
    assert driver._control_clock_armed is False
    assert driver._control_consumer_ready.is_set() is False
    assert driver._motion_ready.is_set() is False
