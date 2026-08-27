from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest


MODEL_ROOT = Path(__file__).resolve().parents[1] / "models" / "unitree-g1-mujoco"
if not MODEL_ROOT.is_dir():
    pytest.skip(
        "G1 model checkout is an operator-fetched L2 asset",
        allow_module_level=True,
    )
sys.path.insert(0, str(MODEL_ROOT))

from sim.base_sim import (  # noqa: E402
    DefaultEnv,
    RESET_POLICY_AUTOMATIC,
    RESET_POLICY_EXPLICIT_PROCESS_RESTART,
    SimulationRestartRequired,
    reset_policy,
)


def test_reset_policy_defaults_to_upstream_compatible_automatic() -> None:
    assert reset_policy({}) == RESET_POLICY_AUTOMATIC


def test_reset_policy_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="RESET_POLICY must be one of"):
        reset_policy({"RESET_POLICY": "teleport_on_fault"})


def test_strict_policy_never_mutates_mujoco_state() -> None:
    env = DefaultEnv.__new__(DefaultEnv)
    env.reset_policy = RESET_POLICY_EXPLICIT_PROCESS_RESTART

    with patch("sim.base_sim.mujoco.mj_resetData") as reset_data:
        with pytest.raises(SimulationRestartRequired, match="restart it explicitly"):
            env.reset(reason="physics became unstable")

    reset_data.assert_not_called()


def test_automatic_policy_keeps_upstream_reset_for_non_acceptance_users() -> None:
    env = DefaultEnv.__new__(DefaultEnv)
    env.reset_policy = RESET_POLICY_AUTOMATIC
    env.mj_model = object()
    env.mj_data = type(
        "Data",
        (),
        {"qpos": np.zeros(7, dtype=float)},
    )()

    with (
        patch("sim.base_sim.mujoco.mj_resetData") as reset_data,
        patch("sim.base_sim.mujoco.mj_forward") as forward,
    ):
        env.reset()

    reset_data.assert_called_once_with(env.mj_model, env.mj_data)
    forward.assert_called_once_with(env.mj_model, env.mj_data)
    assert env.mj_data.qpos[3:7].tolist() == [1.0, 0.0, 0.0, 0.0]
