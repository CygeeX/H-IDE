from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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
