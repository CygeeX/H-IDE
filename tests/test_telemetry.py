from hide_mujoco_sonic.telemetry import TelemetryState, TelemetryStore


def test_navigation_task_peaks_keep_the_maximum_and_reset_for_the_next_goal():
    store = TelemetryStore(TelemetryState(robot_id="g1-sim"))

    # Idle diagnostics remain visible as current values but cannot pollute a
    # future navigation task's peak window.
    idle = store.update(navigation_arm_raw_max_error_from_measured_rad=0.4)
    assert idle.navigation_task_peak_arm_tracking_error_rad == 0.0

    store.update(navigation_task_active=True)
    store.update(
        navigation_arm_constrained_joints=2,
        navigation_arm_raw_max_error_from_measured_rad=0.23,
        navigation_arm_max_velocity_rps=1.7,
        lower_body_target_max_error_rad=0.8,
    )
    first = store.update(
        navigation_arm_constrained_joints=0,
        navigation_arm_raw_max_error_from_measured_rad=0.1,
        navigation_arm_max_velocity_rps=0.4,
        lower_body_target_max_error_rad=0.2,
    )

    assert first.navigation_task_peak_arm_constrained_joints == 2
    assert first.navigation_task_peak_arm_tracking_error_rad == 0.23
    assert first.navigation_task_peak_arm_velocity_rps == 1.7
    assert first.navigation_task_peak_lower_body_target_max_error_rad == 0.8

    reset = store.reset_navigation_task_peaks()
    assert reset.navigation_task_peak_arm_constrained_joints == 0
    assert reset.navigation_task_peak_arm_tracking_error_rad == 0.0
    assert reset.navigation_task_peak_arm_velocity_rps == 0.0
    assert reset.navigation_task_peak_lower_body_target_max_error_rad == 0.0

    store.update(navigation_task_active=False)
    stopped = store.update(navigation_arm_raw_max_error_from_measured_rad=0.9)
    assert stopped.navigation_task_peak_arm_tracking_error_rad == 0.0
