from concurrent.futures import ThreadPoolExecutor
import math
import sys
import threading
import time

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


def test_support_controller_rejects_free_threaded_python(monkeypatch) -> None:
    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)

    with pytest.raises(RuntimeError, match="require the GIL"):
        PlantSupportController()


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


def test_motion_fault_keeps_physics_healthy_and_requires_explicit_reset() -> None:
    support = PlantSupportController()
    support.finish_startup()
    reason = "gait_adapter/GAIT_ADAPTER_FAULT: reverse startup"

    assert support.latch_motion_fault(reason) is True
    state = support.state()
    assert state["mode"] == "safety_stop"
    assert state["fault"] is None
    assert state["motionFault"] == reason
    support.require_healthy()

    with pytest.raises(ValueError, match="explicitly reset"):
        support.rearm(1)
    with pytest.raises(ValueError, match="identity mismatch"):
        support.clear_motion_fault("another fault")

    assert support.clear_motion_fault(reason) is True
    assert support.clear_motion_fault(reason) is False
    cleared = support.state()
    assert cleared["mode"] == "safety_stop"
    assert cleared["motionFault"] is None
    assert support.rearm(1) == "idle"


def test_fall_arrest_release_is_one_way_and_bound_to_activation_epoch() -> None:
    support = PlantSupportController(fall_arrest_enabled=True)
    support.finish_startup()

    assert support.request_fall_arrest_release(17) is True
    pending = support.state()
    assert pending["fallArrestEnabled"] is True
    assert pending["fallArrestReleaseRequestedEpoch"] == 17
    assert pending["fallArrestReleaseConfirmedEpoch"] is None
    assert pending["fallArrestReleaseRejectedEpoch"] is None

    released: list[bool] = []
    assert support.consume_fall_arrest_release(
        lambda: released.append(True)
    ) == 17
    assert released == [True]
    released = support.state()
    assert released["fallArrestEnabled"] is False
    assert released["fallArrestReleaseConfirmedEpoch"] == 17
    assert support.request_fall_arrest_release(17) is False
    assert support.consume_fall_arrest_release(lambda: None) is None

    with pytest.raises(ValueError, match="another activation"):
        support.request_fall_arrest_release(18)


def test_fall_arrest_release_fails_closed_at_unsafe_boundaries() -> None:
    support = PlantSupportController(fall_arrest_enabled=True)

    with pytest.raises(ValueError, match="confirmed idle"):
        support.request_fall_arrest_release(1)
    support.finish_startup()
    support.latch_motion_fault("injected motion fault")
    with pytest.raises(ValueError, match="motion safety boundary"):
        support.request_fall_arrest_release(1)


def test_simulation_thread_rejects_release_that_races_safety_stop() -> None:
    support = PlantSupportController(fall_arrest_enabled=True)
    support.finish_startup()
    assert support.request_fall_arrest_release(31) is True

    support.set_mode("safety_stop")
    assert support.consume_fall_arrest_release(lambda: None) is None
    state = support.state()
    assert state["fallArrestEnabled"] is True
    assert state["fallArrestReleaseConfirmedEpoch"] is None
    assert state["fallArrestReleaseRejectedEpoch"] == 31

    assert support.rearm(1) == "idle"
    assert support.consume_fall_arrest_release(lambda: None) is None
    assert support.state()["fallArrestEnabled"] is True
    with pytest.raises(ValueError, match="physics boundary"):
        support.request_fall_arrest_release(31)


def test_failed_physical_band_disable_rejects_epoch_without_false_confirmation() -> None:
    support = PlantSupportController(fall_arrest_enabled=True)
    support.finish_startup()
    assert support.request_fall_arrest_release(41) is True

    with pytest.raises(RuntimeError, match="injected band failure"):
        support.consume_fall_arrest_release(
            lambda: (_ for _ in ()).throw(RuntimeError("injected band failure"))
        )

    state = support.state()
    assert state["fallArrestEnabled"] is True
    assert state["fallArrestReleaseConfirmedEpoch"] is None
    assert state["fallArrestReleaseRejectedEpoch"] == 41


def test_plant_without_fall_arrest_reports_support_already_off() -> None:
    support = PlantSupportController(fall_arrest_enabled=False)
    support.finish_startup()

    state = support.state()
    assert state["fallArrestEnabled"] is False
    assert state["fallArrestReleaseConfirmedEpoch"] == 0
    assert support.request_fall_arrest_release(9) is False


def test_physics_hot_path_never_waits_for_http_command_writer() -> None:
    support = PlantSupportController()
    support.finish_startup()
    support.observe_simulation_time(1.0)
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_http_writer_lock() -> None:
        with support._write_lock:
            lock_held.set()
            assert release_lock.wait(2.0)

    holder = threading.Thread(target=hold_http_writer_lock, daemon=True)
    holder.start()
    assert lock_held.wait(1.0)

    def exercise_physics_reads() -> int:
        started_ns = time.perf_counter_ns()
        for index in range(100):
            assert support.mode == "idle"
            assert support.state()["mode"] == "idle"
            assert support.observe_simulation_time(1.001 + index * 0.001) is False
            support.require_healthy()
        return time.perf_counter_ns() - started_ns

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        elapsed_ns = executor.submit(exercise_physics_reads).result(timeout=1.0)
        # This is a millisecond-scale regression ceiling for 400 hot-path
        # operations, measured inside the already-scheduled worker thread.
        assert elapsed_ns < 20_000_000
    finally:
        release_lock.set()
        executor.shutdown(wait=True)
        holder.join(timeout=1.0)
    assert not holder.is_alive()


def test_physics_fault_latch_never_waits_for_http_and_advances_stop_epoch() -> None:
    support = PlantSupportController()
    support.finish_startup()
    support.set_mode("safety_stop")
    assert support.rearm(1) == "idle"
    epoch_before = int(support.state()["safetyStopEpoch"])
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_http_writer_lock() -> None:
        with support._write_lock:
            lock_held.set()
            assert release_lock.wait(2.0)

    holder = threading.Thread(target=hold_http_writer_lock, daemon=True)
    holder.start()
    assert lock_held.wait(1.0)
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        assert executor.submit(support.latch_fault, "injected fault").result(
            timeout=1.0
        ) is True
    finally:
        release_lock.set()
        executor.shutdown(wait=True)
        holder.join(timeout=1.0)

    state = support.state()
    assert state["fault"] == "injected fault"
    assert state["mode"] == "safety_stop"
    assert state["safetyStopEpoch"] == epoch_before + 1


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


def test_safety_stop_damps_translation_without_pulling_or_yaw_feedback() -> None:
    raw = np.asarray([999.0, 999.0, 30.0, 40.0, 50.0, 999.0])
    pose = np.zeros(13)
    pose[3] = 1.0
    pose[0:2] = [8.0, -6.0]
    pose[7:9] = [1.5, -2.0]
    pose[10:12] = [-0.2, 0.3]
    pose[12] = 0.5

    result = stabilized_support_wrench(raw, pose, "safety_stop")

    assert result.tolist() == [-270.0, 360.0, 30.0, 6.0, -9.0, 0.0]
