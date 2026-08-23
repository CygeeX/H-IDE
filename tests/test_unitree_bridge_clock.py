from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hide_mujoco_sonic.safety import PlantControlClock


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATH = (
    ROOT / "models" / "unitree-g1-mujoco" / "sim" / "unitree_sdk2py_bridge.py"
)


def _load_bridge_module():
    spec = importlib.util.spec_from_file_location(
        "_hide_test_unitree_sdk2py_bridge",
        BRIDGE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rounded_lowstate_tick_preserves_long_running_50hz_control_period() -> None:
    bridge = _load_bridge_module()
    clock = PlantControlClock(control_hz=50)
    simulation_time_s = 0.0
    admitted = []

    # Repeated addition reproduces MuJoCo's accumulated binary floating-point
    # time, including the old failure around 155.696 seconds.
    for _ in range(40_000):
        tick_ms = bridge.simulation_time_to_tick_ms(simulation_time_s)
        step = clock.observe(tick_ms)
        if step is not None:
            admitted.append(step)
        simulation_time_s += 0.004

    assert len(admitted) == 8_000
    assert {step.dt_s for step in admitted} == {0.02}
    assert admitted[-1].tick_ms == 159_980


@pytest.mark.parametrize("value", [True, -0.001, float("nan"), float("inf")])
def test_lowstate_tick_rejects_invalid_simulation_time(value: object) -> None:
    bridge = _load_bridge_module()
    with pytest.raises(ValueError, match="finite and non-negative"):
        bridge.simulation_time_to_tick_ms(value)


def _bare_bridge(module):
    value = module.UnitreeSdk2Bridge.__new__(module.UnitreeSdk2Bridge)
    value.low_cmd_lock = threading.Lock()
    value.low_cmd_condition = threading.Condition(value.low_cmd_lock)
    value.low_cmd_sequence = 0
    value.low_cmd_control_tick_ms = None
    value.low_cmd_async_boundary = False
    value.low_cmd = None
    value.low_cmd_received = False
    value.new_low_cmd = False
    return value


def test_lowcmd_sequence_wait_observes_one_fresh_dds_command() -> None:
    module = _load_bridge_module()
    bridge = _bare_bridge(module)
    result: list[bool] = []
    waiter = threading.Thread(
        target=lambda: result.append(bridge.WaitForLowCmdAfter(0, 0.5))
    )
    waiter.start()
    time.sleep(0.01)
    message = SimpleNamespace()
    bridge.LowCmdHandler(message)
    waiter.join(timeout=0.5)

    assert waiter.is_alive() is False
    assert result == [True]
    assert bridge.LowCmdSequence() == 1
    assert bridge.low_cmd is message


def test_lowcmd_sequence_wait_times_out_without_fabricating_a_command() -> None:
    module = _load_bridge_module()
    bridge = _bare_bridge(module)

    assert bridge.WaitForLowCmdAfter(0, 0.001) is False
    assert bridge.LowCmdSequence() == 0


def test_lowcmd_tick_correlation_rejects_delayed_prior_command() -> None:
    module = _load_bridge_module()
    bridge = _bare_bridge(module)
    stale = SimpleNamespace(
        reserve=[module.HIDE_CONTROL_TICK_MAGIC, 1_000, 0, 0]
    )
    bridge.LowCmdHandler(stale)

    assert bridge.WaitForLowCmdForTick(0, 1_020, 0.001) is False

    current = SimpleNamespace(
        reserve=[module.HIDE_CONTROL_TICK_MAGIC, 1_020, 0, 0]
    )
    bridge.LowCmdHandler(current)
    assert bridge.WaitForLowCmdForTick(1, 1_020, 0.001) is True


def test_lowcmd_async_stop_satisfies_any_waiting_control_tick() -> None:
    module = _load_bridge_module()
    bridge = _bare_bridge(module)
    bridge.LowCmdHandler(
        SimpleNamespace(reserve=[module.HIDE_ASYNC_BOUNDARY_MAGIC, 0, 0, 0])
    )

    assert bridge.WaitForLowCmdForTick(0, 9_999, 0.001) is True
