from __future__ import annotations

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
    if control_clock_armed:
        driver._control_consumer_ready.set()
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

    # 3008 ms is a baseline only.  The first earned policy step is 3028 ms;
    # there is no queued startup sample that can race with it.
    for tick in (3_012, 3_016, 3_020, 3_024, 3_028):
        driver._on_lowstate(SimpleNamespace(tick=tick))
    consumer.join(timeout=0.5)

    assert consumer.is_alive() is False
    assert driver._fault_latched.is_set() is False
    assert len(consumed) == 1
    assert consumed[0] is not None
    assert consumed[0][1].tick_ms == 3_028


def test_controller_warmup_runs_sessions_without_policy_step() -> None:
    class _Decoder:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, _outputs, inputs):
            self.calls += 1
            assert set(inputs) == {"decoder_input"}
            return [np.asarray([[0.1, 0.2]], dtype=np.float32)]

    class _Pipeline:
        decoder_input = "decoder_input"

        def __init__(self) -> None:
            self.encoder_calls = 0
            self.decoder = _Decoder()

        def run_encoder(self):
            self.encoder_calls += 1
            return np.asarray([0.3, 0.4], dtype=np.float32)

        @staticmethod
        def build_decoder_obs():
            return np.asarray([0.0, 1.0], dtype=np.float32)

    wall_now = [10.0]
    driver = _bare_driver(wall_now=wall_now, control_clock_armed=False)
    pipeline = _Pipeline()
    driver._controller = SimpleNamespace(controller=pipeline)

    driver._warm_controller_inference()

    assert pipeline.encoder_calls == 1
    assert pipeline.decoder.calls == 1
    assert driver._control_clock_armed is False
    assert driver._control_consumer_ready.is_set() is False
    assert driver.telemetry.values["controller_warmup_complete"] is True
    assert driver.telemetry.values["control_clock_status"] == "warm_ready_to_arm"


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
    driver._warm_controller_inference = lambda: events.append(("warm", None))

    driver._prepare_idle_startup()

    assert events == [
        ("support", "idle"),
        ("hold", lowstate),
        ("warm", None),
    ]
    assert driver._control_clock_armed is False
    assert driver._control_consumer_ready.is_set() is False
