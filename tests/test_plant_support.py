import math

import pytest
import numpy as np

from hide_mujoco_sonic.plant import (
    PlantPhysicsFault,
    PlantSupportController,
    stabilized_support_wrench,
)


def test_support_controller_finishes_startup_without_overriding_locomotion():
    support = PlantSupportController()
    assert support.state()["mode"] == "startup"
    assert support.finish_startup() == "idle"
    assert support.set_mode("locomotion") == "locomotion"
    assert support.finish_startup() == "locomotion"


def test_support_controller_rejects_unknown_mode():
    support = PlantSupportController()
    with pytest.raises(ValueError, match="support mode"):
        support.set_mode("teleport")
    assert support.state()["mode"] == "startup"


def test_support_controller_latches_physics_time_rollback_until_restart():
    support = PlantSupportController()
    assert support.observe_simulation_time(12.0) is False
    assert support.observe_simulation_time(0.004) is True
    assert support.state()["mode"] == "safety_stop"
    assert "rolled back" in str(support.state()["fault"])
    assert support.set_mode("safety_stop") == "safety_stop"
    with pytest.raises(ValueError, match="physics fault is latched"):
        support.set_mode("locomotion")


def test_support_controller_fault_terminates_instead_of_resetting() -> None:
    support = PlantSupportController()

    with pytest.raises(PlantPhysicsFault, match="numerical instability"):
        support.fail("numerical instability")

    assert support.state()["mode"] == "safety_stop"
    assert support.state()["fault"] == "numerical instability"
    with pytest.raises(PlantPhysicsFault, match="numerical instability"):
        support.require_healthy()


def test_safety_stop_is_sticky_until_strictly_newer_explicit_rearm() -> None:
    support = PlantSupportController()
    support.finish_startup()
    assert support.set_mode("safety_stop") == "safety_stop"
    state = support.state()
    assert state["safetyStopLatched"] is True
    assert state["safetyStopEpoch"] == 1

    for mode in ("idle", "locomotion"):
        with pytest.raises(ValueError, match="explicit control rearm"):
            support.set_mode(mode)
    assert support.state()["mode"] == "safety_stop"

    assert support.rearm(101) == "idle"
    assert support.state()["lastRearmGeneration"] == 101
    assert support.set_mode("locomotion") == "locomotion"
    assert support.set_mode("safety_stop") == "safety_stop"
    with pytest.raises(ValueError, match="strictly newer"):
        support.rearm(101)
    assert support.rearm(102) == "idle"


def test_physics_fault_cannot_be_rearmed_online() -> None:
    support = PlantSupportController()
    support.latch_fault("physics invalid")

    with pytest.raises(ValueError, match="physics fault is latched"):
        support.rearm(1)
    assert support.state()["mode"] == "safety_stop"


def test_locomotion_keeps_vertical_roll_and_pitch_fall_arrest() -> None:
    raw = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    pose = np.zeros(13)
    pose[3] = 1.0
    pose[10:12] = [0.2, -0.3]

    result = stabilized_support_wrench(raw, pose, "locomotion")

    assert result.tolist() == [0.0, 0.0, 30.0, -6.0, 9.0, 0.0]


def test_tilt_fall_arrest_is_continuous_across_pi_yaw() -> None:
    from scipy.spatial.transform import Rotation

    torques = []
    for yaw in (math.pi - 1e-4, -math.pi + 1e-4):
        xyzw = Rotation.from_euler("xyz", [0.08, -0.06, yaw]).as_quat()
        pose = np.zeros(13)
        pose[3:7] = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
        torques.append(
            stabilized_support_wrench(
                np.zeros(6),
                pose,
                "locomotion",
            )[3:5]
        )

    assert torques[0] == pytest.approx(torques[1], abs=0.01)


def test_safety_stop_damps_velocity_without_pulling_to_origin() -> None:
    raw = np.asarray([999.0, 999.0, 30.0, 40.0, 50.0, 999.0])
    pose = np.zeros(13)
    pose[3] = 1.0
    pose[0:2] = [8.0, -6.0]
    pose[7:9] = [1.5, -2.0]
    pose[10:12] = [-0.2, 0.3]
    pose[12] = 0.5

    result = stabilized_support_wrench(raw, pose, "safety_stop")

    assert result.tolist() == [-270.0, 360.0, 30.0, 6.0, -9.0, -40.0]
