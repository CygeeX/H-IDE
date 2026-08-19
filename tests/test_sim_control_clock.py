from __future__ import annotations

from types import SimpleNamespace
import threading

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


def _bare_driver(*, wall_now: list[float] | None = None) -> SonicDriver:
    driver = SonicDriver.__new__(SonicDriver)
    driver.guard = _Guard()
    driver.telemetry = _Telemetry()
    driver._lowstate = None
    driver._lowstate_lock = threading.RLock()
    driver._control_condition = threading.Condition(driver._lowstate_lock)
    driver._control_clock = PlantControlClock(control_hz=50)
    driver._control_clock_armed = True
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
