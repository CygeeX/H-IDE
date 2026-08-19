import json

from hide_mujoco_sonic.telemetry import (
    JointFaultTraceBuffer,
    TelemetryState,
    TelemetryStore,
)


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


def test_hot_path_update_does_not_clone_full_telemetry(monkeypatch):
    store = TelemetryStore(TelemetryState(robot_id="g1-sim"))

    def fail_if_copied(_self):
        raise AssertionError("hot update cloned the complete telemetry state")

    monkeypatch.setattr(TelemetryState, "json", fail_if_copied)
    assert store.update_hot(plant_tick=17, sonic_speed_mps=0.1) is None


def test_regular_update_keeps_its_isolated_snapshot_return_value():
    store = TelemetryStore(TelemetryState(robot_id="g1-sim"))

    updated = store.update(plant_tick=17)
    store.update(plant_tick=18)

    assert updated.plant_tick == 17
    assert store.get().plant_tick == 18


def test_reference_transaction_and_terminal_fields_are_strict_schema_members():
    store = TelemetryStore(TelemetryState(robot_id="g1-sim"))

    updated = store.update(
        sonic_reference_authority_state="GRANTED",
        sonic_reference_authority_generation=8,
        sonic_reference_authority_source_sequence=13,
        sonic_reference_authority_intent_generation=5,
        sonic_reference_authority_epoch=3,
        sonic_reference_actuator_authority=True,
        sonic_reference_authority_dds_target_sequence=21,
        sonic_execution_feedback_pending=True,
        terminal_boundary_error="vendor revoke failed",
        plant_terminal_halt_confirmed=False,
        plant_terminal_halt_error="timeout",
    )

    assert updated.sonic_reference_authority_state == "GRANTED"
    assert updated.sonic_reference_authority_epoch == 3
    assert updated.sonic_reference_actuator_authority is True
    assert updated.sonic_execution_feedback_pending is True
    assert updated.plant_terminal_halt_confirmed is False
    assert updated.plant_terminal_halt_error == "timeout"


def test_snapshot_json_adds_runtime_diagnostics_without_mutating_state():
    store = TelemetryStore(TelemetryState(robot_id="g1-sim"))
    store.set_runtime_diagnostics_provider(
        lambda: {"lowstate_callback_max_gap_s": 0.149}
    )

    payload = store.snapshot_json()

    assert payload["runtime_diagnostics"] == {
        "lowstate_callback_max_gap_s": 0.149
    }
    assert not hasattr(store.get(), "runtime_diagnostics")


def test_runtime_diagnostics_failure_does_not_break_telemetry_snapshot():
    store = TelemetryStore(TelemetryState(robot_id="g1-sim"))

    def fail():
        raise RuntimeError("probe failed")

    store.set_runtime_diagnostics_provider(fail)

    payload = store.snapshot_json()
    assert payload["robot_id"] == "g1-sim"
    assert payload["runtime_diagnostics"] == {
        "snapshot_error": "RuntimeError: probe failed"
    }


def test_joint_fault_trace_is_bounded_named_and_strict_json(tmp_path):
    trace = JointFaultTraceBuffer(("hip", "left_shoulder_roll"), max_samples=2)

    for tick in (0, 20, 40):
        values = (float(tick), float("nan") if tick == 40 else 0.2)
        trace.record(
            plant_tick_ms=tick,
            control_dt_s=0.02,
            intent={
                "mode": "SLOW_WALK",
                "speed_mps": 0.1,
                "intent_generation": 7,
                "intent_signature": "intent-sha256",
                "planner_request_generation": 11,
                "planner_accepted_generation": 11,
                "planner_ref_cursor": 3,
                "planner_predicted_root_direction_x": 1.0,
                "planner_predicted_root_direction_y": 0.0,
                "planner_predicted_root_projection_m": 0.4,
                "planner_requested_yaw_error_rad": 0.10,
                "planner_predicted_yaw_delta_rad": 0.08,
                "planner_yaw_consistent": True,
            },
            handoff_ratio=0.5,
            plant_support_mode="locomotion",
            measured_q_rad=values,
            measured_dq_rps=(0.0, 0.0),
            measured_ddq_rps2=(0.0, 0.0),
            measured_tau_nm=(1.0, 2.0),
            policy_target_q_rad=(0.1, 0.2),
            handoff_target_q_rad=(0.1, 0.2),
            published_target_q_rad=(0.1, 0.2),
        )

    payload = trace.snapshot(reason="QACC at DOF 22", kp=(10.0, 20.0), kd=(1.0, 2.0))
    assert payload["joint_names"] == ["hip", "left_shoulder_roll"]
    assert payload["sample_count"] == 2
    assert [sample["plant_tick_ms"] for sample in payload["samples"]] == [20, 40]
    assert payload["samples"][-1]["measured_q_rad"][-1] is None
    assert payload["samples"][-1]["intent"]["intent_generation"] == 7.0
    assert payload["samples"][-1]["intent"]["intent_signature"] == "intent-sha256"
    assert payload["samples"][-1]["intent"]["planner_request_generation"] == 11.0
    assert payload["samples"][-1]["intent"]["planner_accepted_generation"] == 11.0
    assert payload["samples"][-1]["intent"]["planner_ref_cursor"] == 3.0
    assert payload["samples"][-1]["intent"][
        "planner_predicted_root_projection_m"
    ] == 0.4
    assert payload["samples"][-1]["intent"][
        "planner_requested_yaw_error_rad"
    ] == 0.1
    assert payload["samples"][-1]["intent"][
        "planner_predicted_yaw_delta_rad"
    ] == 0.08
    assert payload["samples"][-1]["intent"]["planner_yaw_consistent"] is True
    assert payload["gains"] == {"kp": [10.0, 20.0], "kd": [1.0, 2.0]}

    path = trace.persist(tmp_path / "sonic-fault-trace.json", payload)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["reason"] == "QACC at DOF 22"
    assert restored["samples"][-1]["measured_q_rad"][-1] is None
