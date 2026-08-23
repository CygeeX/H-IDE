from __future__ import annotations

import math

import pytest
import numpy as np

from hide_mujoco_sonic.ros_observation_bridge import MujocoObservationBridge
from hide_mujoco_sonic.safety import (
    BodyStabilityGuard,
    CmdVelGuard,
    GoalProgressGuard,
    MovementIntent,
    MotionDirectionGuard,
    NavigationArmGuard,
    NominalPolicyActivation,
    PlantControlClock,
    PlantControlClockFault,
    PlantTickGuard,
    ReplanIntentGate,
    RotationDirectionGuard,
    StaticJointHandoff,
    TranslationHeadingGuard,
)


def test_body_stability_guard_requires_stable_pose_and_nav2_zero_before_recovery() -> None:
    now = [10.0]
    guard = BodyStabilityGuard(
        max_tilt_rad=0.35,
        violation_s=0.15,
        recovery_s=0.75,
        clock=lambda: now[0],
    )
    assert guard.observe(0.40, 0.0) is False
    now[0] = 10.16
    assert guard.observe(0.41, 0.0) is True
    assert guard.latched is True
    assert guard.admit_command(0.2, 0.0, 0.0) is False

    now[0] = 10.30
    assert guard.observe(0.05, 0.04) is False
    assert guard.admit_command(0.0, 0.0, 0.0) is True
    now[0] = 11.10
    # A still-active Nav2 goal must not resume automatically after recovery.
    assert guard.admit_command(0.2, 0.0, 0.0) is False
    assert guard.latched is True
    assert guard.admit_command(0.0, 0.0, 0.0) is True
    now[0] = 11.86
    assert guard.admit_command(0.0, 0.0, 0.0) is True
    assert guard.latched is False


def test_plant_tick_guard_distinguishes_delayed_sample_from_restart():
    guard = PlantTickGuard(rollback_threshold_ms=1_000)
    assert guard.observe(50_000) is False
    assert guard.observe(49_900) is False
    assert guard.last_tick == 50_000
    assert guard.observe(100) is True
    assert guard.last_tick == 100
    assert guard.restart_count == 1
    assert guard.observe(110) is False


def test_plant_control_clock_admits_one_step_per_20ms_of_simulation_time():
    clock = PlantControlClock(control_hz=50)

    first = clock.observe(1_000)
    assert first is not None
    assert first.tick_ms == 1_000
    assert first.dt_s == pytest.approx(0.02)

    for tick in (1_004, 1_008, 1_012, 1_016):
        assert clock.observe(tick) is None
    second = clock.observe(1_020)
    assert second is not None
    assert second.tick_ms == 1_020
    assert second.dt_s == pytest.approx(0.02)


def test_plant_control_clock_accepts_long_running_rounded_4ms_mujoco_tick_stream():
    clock = PlantControlClock(control_hz=50)
    simulation_time_s = 0.0
    admitted = []

    # The bridge rounds accumulated MuJoCo time to the nearest integer
    # millisecond.  That keeps the integral 4 ms physics / 20 ms control
    # cadence exact even after long-running floating-point accumulation.
    for _ in range(25_000):
        step = clock.observe(int(round(simulation_time_s * 1_000)))
        if step is not None:
            admitted.append(step)
        simulation_time_s += 0.004

    assert len(admitted) == 5_000
    assert {step.dt_s for step in admitted} == {0.02}


@pytest.mark.parametrize(
    ("ticks", "message"),
    [
        ((1_000, 1_000), "duplicate"),
        ((1_000, 996), "backwards"),
        ((1_000, 1_024), "source jumped"),
        ((1_000, 1_004, 1_008, 1_012, 1_016, 1_025), "interval jumped"),
    ],
)
def test_plant_control_clock_fails_closed_on_discontinuous_ticks(ticks, message):
    clock = PlantControlClock(control_hz=50)
    with pytest.raises(PlantControlClockFault, match=message):
        for tick in ticks:
            clock.observe(tick)


def test_motion_direction_guard_latches_opposite_map_displacement_and_requires_idle():
    now = [10.0]
    guard = MotionDirectionGuard(clock=lambda: now[0])
    assert guard.observe_pose(0.0, 0.0, 0.0) is False
    assert guard.accept_command(0.2, 0.0, 0.0) is True
    assert guard.observe_pose(-0.03, 0.0, 0.0) is False
    assert guard.observe_pose(-0.07, 0.0, 0.0) is True
    assert guard.latched is True
    assert guard.accept_command(0.2, 0.0, 0.0) is False

    assert guard.accept_command(0.0, 0.0, 0.0) is False
    now[0] += 0.51
    assert guard.accept_command(0.0, 0.0, 0.0) is True
    assert guard.latched is False


def test_translation_heading_guard_latches_sustained_uncommanded_yaw() -> None:
    now = [10.0]
    guard = TranslationHeadingGuard(
        max_heading_error_rad=0.25,
        violation_s=0.25,
        clock=lambda: now[0],
    )
    assert guard.observe_pose(0.0) is False
    assert guard.accept_command(0.2, 0.0, 0.0) is True
    now[0] = 10.1
    assert guard.observe_pose(0.30) is False
    now[0] = 10.36
    assert guard.observe_pose(0.31) is True
    assert guard.latched is True
    assert "heading diverges" in str(guard.reason)
    assert guard.accept_command(0.2, 0.0, 0.0) is False


def test_translation_heading_guard_accounts_for_nav2_yaw_command() -> None:
    now = [20.0]
    guard = TranslationHeadingGuard(
        max_heading_error_rad=0.25,
        violation_s=0.25,
        clock=lambda: now[0],
    )
    guard.observe_pose(0.0)
    guard.accept_command(0.2, 0.0, 0.2)
    now[0] = 21.0
    # Command integration is deliberately bounded to a 0.5 s sample gap.
    guard.accept_command(0.2, 0.0, 0.2)
    assert guard.observe_pose(0.30) is False


def test_translation_heading_guard_caps_expected_yaw_to_downstream_authority() -> None:
    now = [30.0]
    guard = TranslationHeadingGuard(
        max_heading_error_rad=0.25,
        max_expected_delta_rad=0.05,
        violation_s=0.25,
        clock=lambda: now[0],
    )
    guard.observe_pose(0.0)
    guard.accept_command(0.2, 0.0, 0.3)
    for step in range(1, 7):
        now[0] = 30.0 + step * 0.5
        guard.accept_command(0.2, 0.0, 0.3)

    # Raw Nav2 wz would integrate to almost one radian, but SONIC only receives
    # a 0.05-rad mixed-translation correction.  The matching measured response
    # must not be rejected by the safety observer.
    assert guard.observe_pose(0.05) is False
    now[0] += 0.3
    assert guard.observe_pose(0.06) is False
    assert guard.latched is False


def test_motion_direction_guard_rotates_base_command_into_map_frame():
    guard = MotionDirectionGuard()
    guard.observe_pose(2.0, 3.0, math.pi / 2.0)
    assert guard.accept_command(0.2, 0.0, 0.0) is True
    assert guard.observe_pose(2.0, 3.1, math.pi / 2.0) is False
    assert guard.latched is False


def test_motion_direction_guard_ignores_in_place_rotation():
    guard = MotionDirectionGuard()
    guard.observe_pose(0.0, 0.0, 0.0)
    assert guard.accept_command(0.0, 0.0, 0.3) is True
    assert guard.observe_pose(-0.2, 0.0, 0.5) is False
    assert guard.latched is False


def test_motion_direction_guard_keeps_anchor_through_nav2_turn_transient():
    guard = MotionDirectionGuard()
    guard.observe_pose(0.0, 0.0, 0.0)
    assert guard.accept_command(0.2, 0.0, 0.0) is True
    assert guard.accept_command(0.05, 0.0, 0.4) is True
    assert guard.observe_pose(-0.07, 0.0, 0.4) is True
    assert guard.latched is True


def test_rotation_direction_guard_rejects_reverse_spin_and_clears_after_idle():
    now = [10.0]
    guard = RotationDirectionGuard(clock=lambda: now[0])
    guard.observe_pose(0.0, 0.0, 0.2)
    assert guard.accept_command(0.0, 0.0, 0.2) is True
    assert guard.observe_pose(0.01, 0.0, 0.17) is False
    assert guard.observe_pose(0.02, 0.0, 0.14) is True
    assert guard.latched is True
    assert guard.accept_command(0.0, 0.0, 0.2) is False
    assert guard.accept_command(0.0, 0.0, 0.0) is False
    now[0] += 0.51
    assert guard.accept_command(0.0, 0.0, 0.0) is True


def test_rotation_direction_guard_allows_bounded_left_spin_but_rejects_translation():
    now = [10.0]
    guard = RotationDirectionGuard(
        max_translation_m=0.10,
        translation_violation_s=0.50,
        clock=lambda: now[0],
    )
    guard.observe_pose(1.0, 2.0, -0.2)
    assert guard.accept_command(0.0, 0.0, 0.2) is True
    assert guard.observe_pose(1.02, 2.01, 0.1) is False
    assert guard.observe_pose(1.12, 2.0, 0.2) is False
    now[0] += 0.51
    assert guard.observe_pose(1.12, 2.0, 0.2) is True
    assert "translation envelope" in (guard.reason or "")


def test_rotation_direction_guard_ignores_a_transient_pelvis_sway():
    now = [10.0]
    guard = RotationDirectionGuard(
        max_translation_m=0.10,
        translation_violation_s=0.50,
        clock=lambda: now[0],
    )
    guard.observe_pose(1.0, 2.0, 0.0)
    guard.accept_command(0.0, 0.0, 0.2)
    assert guard.observe_pose(1.12, 2.0, 0.1) is False
    now[0] += 0.30
    assert guard.observe_pose(1.05, 2.0, 0.2) is False
    assert guard.latched is False


def test_rotation_direction_guard_unwraps_forward_spin_across_pi_boundary():
    guard = RotationDirectionGuard()
    guard.observe_pose(0.0, 0.0, math.pi - 0.20)
    assert guard.accept_command(0.0, 0.0, 0.2) is True
    assert guard.observe_pose(0.0, 0.0, math.pi - 0.05) is False
    assert guard.observe_pose(0.0, 0.0, -math.pi + 0.10) is False
    assert guard.observe_pose(0.0, 0.0, -math.pi + 0.30) is False
    assert guard.latched is False


def test_goal_progress_guard_latches_after_progress_is_reversed():
    now = [10.0]
    guard = GoalProgressGuard(clock=lambda: now[0])
    guard.observe_pose(0.0, 0.0)
    assert guard.observe_goal(1.0, 0.0) is True
    assert guard.observe_pose(0.2, 0.0) is False
    assert guard.observe_pose(0.03, 0.0) is True
    assert guard.latched is True
    assert "regresses from Nav2 goal" in (guard.reason or "")
    assert guard.accept_command(0.2, 0.0, 0.0) is False
    assert guard.accept_command(0.0, 0.0, 0.0) is False
    now[0] += 0.51
    assert guard.accept_command(0.0, 0.0, 0.0) is True


def test_goal_progress_guard_ignores_small_detour_and_equivalent_replan():
    guard = GoalProgressGuard()
    guard.observe_pose(0.0, 0.0)
    assert guard.observe_goal(1.0, 0.0) is True
    assert guard.observe_pose(0.2, 0.0) is False
    assert guard.observe_goal(1.04, 0.02) is False
    assert guard.observe_pose(0.08, 0.0) is False
    assert guard.latched is False


def test_navigation_guards_reanchor_only_at_policy_actuator_grant() -> None:
    now = [10.0]

    direction = MotionDirectionGuard(clock=lambda: now[0])
    direction.observe_pose(0.0, 0.0, 0.0)
    direction.accept_command(0.30, 0.0, 0.0)
    direction.observe_pose(-0.05, 0.0, 0.0)
    assert direction.reanchor_at_actuator_grant() is True
    assert direction.observe_pose(-0.09, 0.0, 0.0) is False

    progress = GoalProgressGuard(clock=lambda: now[0])
    progress.observe_pose(0.0, 0.0)
    progress.observe_goal(1.0, 0.0)
    progress.observe_pose(-0.10, 0.0)
    assert progress.reanchor_at_actuator_grant() is True
    assert progress.observe_pose(-0.14, 0.0) is False

    heading = TranslationHeadingGuard(clock=lambda: now[0])
    heading.observe_pose(0.0)
    heading.accept_command(0.30, 0.0, 0.0)
    heading.observe_pose(0.20)
    assert heading.reanchor_at_actuator_grant() is True
    now[0] += 0.30
    assert heading.observe_pose(0.21) is False

    rotation = RotationDirectionGuard(clock=lambda: now[0])
    rotation.observe_pose(0.0, 0.0, 0.0)
    rotation.accept_command(0.0, 0.0, 0.20)
    rotation.observe_pose(0.02, 0.0, -0.04)
    assert rotation.reanchor_at_actuator_grant() is True
    assert rotation.observe_pose(0.03, 0.0, -0.06) is False


def test_static_joint_handoff_is_binary_hold_or_exact_policy_authority():
    now = [1.0]
    handoff = StaticJointHandoff(joint_count=2, warmup_s=0.3, blend_s=0.4, clock=lambda: now[0])
    idle = handoff.decide(motion_active=False, measured=[1.0, 2.0], policy=[9.0, 9.0])
    assert idle.targets == pytest.approx((1.0, 2.0))
    assert idle.static_hold_active is True

    now[0] = 2.0
    start = handoff.decide(motion_active=True, measured=[1.1, 2.1], policy=[3.0, 4.0])
    assert start.targets == pytest.approx((3.0, 4.0))
    assert start.handoff_ratio == 1.0
    assert start.motion_handoff_active is False
    now[0] = 2.5
    active = handoff.decide(motion_active=True, measured=[1.2, 2.2], policy=[3.1, 3.9])
    assert active.handoff_ratio == 1.0
    assert active.targets == pytest.approx((3.1, 3.9))
    assert active.motion_handoff_active is False

    stopped = handoff.decide(motion_active=False, measured=[2.8, 3.8], policy=[0.0, 0.0])
    assert stopped.targets == pytest.approx((2.8, 3.8))
    assert stopped.static_hold_active is True
    assert stopped.motion_handoff_active is False

    held = handoff.decide(
        motion_active=False,
        measured=[9.0, 9.0],
        policy=[0.0, 0.0],
    )
    assert held.targets == pytest.approx((2.8, 3.8))


def test_static_joint_handoff_freezes_published_preload_without_reanchoring():
    handoff = StaticJointHandoff(joint_count=2)
    # These are actuator targets, not measured angles.  Their small error from
    # measured position represents the PD preload balancing gravity.
    handoff.freeze([1.10, 1.90])

    stopped = handoff.decide(
        motion_active=False,
        measured=[1.00, 2.00],
        policy=[9.0, 9.0],
    )
    assert stopped.targets == pytest.approx((1.10, 1.90))
    held = handoff.decide(
        motion_active=False,
        measured=[0.95, 2.05],
        policy=[9.0, 9.0],
    )
    assert held.targets == pytest.approx((1.10, 1.90))

    started = handoff.decide(
        motion_active=True,
        reference_ready=True,
        measured=[0.90, 2.10],
        policy=[1.50, 1.50],
        now=10.0,
    )
    assert started.handoff_ratio == 1.0
    assert started.targets == pytest.approx((1.50, 1.50))


def test_nominal_policy_activation_uses_simulation_ticks_and_exact_dds_commits():
    activation = NominalPolicyActivation(
        joint_count=2,
        duration_s=3.0,
        settle_consecutive_samples=2,
    )
    assert activation.begin([0.10, -0.20], tick_ms=1_000) is True

    first = activation.propose(
        [0.40, 0.10], measured=[0.10, -0.20], tick_ms=1_000,
    )
    assert first.ratio == 0.0
    assert first.targets == (0.10, -0.20)
    assert activation.commit(first, first.targets) is False

    middle = activation.propose(
        [0.40, 0.10], measured=[0.16, -0.14], tick_ms=2_500,
    )
    assert middle.ratio == pytest.approx(0.5)
    assert middle.targets == pytest.approx((0.28, -0.02))
    # A target that did not cross DDS cannot advance the transaction.
    assert activation.commit(middle, first.targets) is False
    assert activation.ratio == 0.0
    assert activation.commit(middle, middle.targets) is False

    final = activation.propose(
        [0.40, 0.10], measured=[0.35, 0.05], tick_ms=4_000,
    )
    assert final.complete is True
    assert activation.commit(final, final.targets) is True
    assert activation.phase == "SETTLING_TETHERED"
    assert activation.ratio == 1.0
    for tick_ms in (4_020, 4_040):
        tethered = activation.observe_settling(
            nominal=[0.40, 0.10],
            measured=[0.40, 0.10],
            velocities=[0.0, 0.0],
            gyroscope=[0.0, 0.0, 0.0],
            roll_rad=0.0,
            pitch_rad=0.0,
            tick_ms=tick_ms,
        )
    assert tethered.stage_complete is True
    assert activation.phase == "WAITING_FOR_SUPPORT_RELEASE"
    assert activation.ready_targets == pytest.approx((0.40, 0.10))
    assert activation.record_support_release(activation.epoch) is True
    assert activation.confirm_support_release(tick_ms=4_040) is True
    assert activation.phase == "PREPARED"
    # Official INIT is once per process, not once per Mission.
    assert activation.begin([9.0, 9.0], tick_ms=5_000) is False


def test_activation_stop_after_policy_grant_and_release_preserves_init():
    activation = NominalPolicyActivation(
        joint_count=1,
        duration_s=3.0,
        settle_consecutive_samples=1,
    )
    assert activation.begin([0.1], tick_ms=0)
    final = activation.propose([0.0], measured=[0.0], tick_ms=3_000)
    assert activation.commit(final, final.targets)
    tethered = activation.observe_settling(
        nominal=[0.0], measured=[0.0], velocities=[0.0],
        gyroscope=[0.0, 0.0, 0.0], roll_rad=0.0, pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert tethered.stage_complete
    with pytest.raises(ValueError, match="predates tethered evidence"):
        activation.confirm_support_release(tick_ms=3_019)
    released_epoch = activation.epoch
    assert activation.record_support_release(released_epoch)

    # A release proof can only be requested after the first policy DDS grant.
    # If stop wins before the owner tick consumes the response, INIT is still
    # complete and must never be repeated unsupported.
    assert activation.abort() is True
    assert activation.phase == "PREPARED"
    assert activation.support_release_confirmed is True
    assert activation.provenance["support_release"] == {
        "mode": "one_way_release",
        "one_way": True,
        "confirmed_epoch": released_epoch,
        "cancel_recovery": "resume_policy_from_exact_frozen_boundary",
    }
    assert activation.begin([0.05], tick_ms=4_000) is False


def test_nominal_policy_activation_can_retain_nonpropulsive_fall_arrest():
    activation = NominalPolicyActivation(
        joint_count=1,
        duration_s=3.0,
        settle_consecutive_samples=1,
        release_fall_arrest=False,
    )
    assert activation.begin([0.1], tick_ms=0)
    final = activation.propose([0.0], measured=[0.0], tick_ms=3_000)
    assert activation.commit(final, final.targets)
    settled = activation.observe_settling(
        nominal=[0.0], measured=[0.0], velocities=[0.0],
        gyroscope=[0.0, 0.0, 0.0], roll_rad=0.0, pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert settled.stage_complete
    assert activation.activation_path == "tethered_retained"
    assert activation.support_release_confirmed is False
    assert activation.confirm_supported_takeover(tick_ms=3_040)
    assert activation.prepared
    assert activation.provenance["support_release"] == {
        "mode": "retained_vertical_attitude_fall_arrest",
        "one_way": False,
        "confirmed_epoch": None,
        "cancel_recovery": "resume_policy_from_exact_frozen_boundary",
    }
    assert activation.provenance["policy_takeover"] == (
        "first_successful_policy_dds_with_fall_arrest_retained"
    )


def test_legacy_support_off_recovery_skips_static_init():
    activation = NominalPolicyActivation(
        joint_count=1,
        duration_s=3.0,
        settle_consecutive_samples=1,
    )
    assert activation.begin([0.1], tick_ms=0)
    assert activation.abort()
    released_epoch = 1
    assert activation.record_support_release(released_epoch)
    assert activation.phase == "UNPREPARED_SUPPORT_OFF"

    assert activation.begin([0.05], tick_ms=4_000)
    assert activation.phase == "WAITING_FOR_SUPPORT_RELEASE"
    assert activation.epoch != released_epoch
    assert activation.ready_targets == (0.05,)
    assert activation.provenance["activation_epoch"] == activation.epoch
    assert activation.provenance["activation_path"] == "support_off_reinit"

    # A newly granted reference may immediately reuse the old release proof
    # from the exact frozen actuator boundary.
    assert activation.confirm_support_release(tick_ms=4_020)
    assert activation.prepared
    assert activation.provenance["activation_epoch"] != released_epoch
    assert activation.provenance["support_release"]["confirmed_epoch"] == released_epoch


def test_nominal_policy_activation_stability_gate_resets_and_times_out():
    activation = NominalPolicyActivation(
        joint_count=1,
        duration_s=3.0,
        settle_consecutive_samples=2,
        settle_timeout_s=3.0,
    )
    assert activation.begin([0.0], tick_ms=0)
    final = activation.propose([0.0], measured=[0.0], tick_ms=3_000)
    assert activation.commit(final, final.targets) is True

    stable = activation.observe_settling(
        nominal=[0.0],
        measured=[0.09],
        velocities=[0.19],
        gyroscope=[0.14, 0.0, 0.0],
        roll_rad=0.14,
        pitch_rad=0.0,
        tick_ms=3_020,
    )
    assert stable.stable is True
    assert stable.consecutive_samples == 1
    with pytest.raises(RuntimeError, match="fresh settling sample"):
        activation.observe_settling(
            nominal=[0.0],
            measured=[0.0],
            velocities=[0.0],
            gyroscope=[0.0, 0.0, 0.0],
            roll_rad=0.0,
            pitch_rad=0.0,
            tick_ms=3_020,
        )
    unstable = activation.observe_settling(
        nominal=[0.0],
        measured=[0.100001],
        velocities=[0.0],
        gyroscope=[0.0, 0.0, 0.0],
        roll_rad=0.0,
        pitch_rad=0.0,
        tick_ms=3_040,
    )
    assert unstable.stable is False
    assert unstable.consecutive_samples == 0

    activation.observe_settling(
        nominal=[0.0], measured=[0.0], velocities=[0.0],
        gyroscope=[0.0, 0.0, 0.0], roll_rad=0.0, pitch_rad=0.0,
        tick_ms=3_060,
    )
    gap = activation.observe_settling(
        nominal=[0.0], measured=[0.0], velocities=[0.0],
        gyroscope=[0.0, 0.0, 0.0], roll_rad=0.0, pitch_rad=0.0,
        tick_ms=3_100,
    )
    assert gap.consecutive_samples == 1

    with pytest.raises(RuntimeError, match="settling timeout"):
        activation.observe_settling(
            nominal=[0.0],
            measured=[0.2],
            velocities=[0.0],
            gyroscope=[0.0, 0.0, 0.0],
            roll_rad=0.0,
            pitch_rad=0.0,
            tick_ms=6_001,
        )


def test_nominal_policy_activation_provenance_exposes_fixed_gate():
    activation = NominalPolicyActivation(joint_count=1, duration_s=3.0)
    provenance = activation.provenance
    assert provenance["source"] == (
        "NVlabs/GR00T-WholeBodyControl::G1Deploy::InitControl"
    )
    assert provenance["init_interpolation"] == (
        "measured_current_to_nominal_each_tick"
    )
    assert provenance["settle_gate"] == {
        "max_joint_error_rad": 0.10,
        "max_joint_velocity_rps": 0.20,
        "max_gyro_rps": 0.15,
        "max_abs_roll_pitch_rad": 0.15,
        "required_consecutive_samples": 10,
        "timeout_s": 3.0,
        "sample_period_ms": 20,
        "stages": ["tethered"],
    }
    assert provenance["policy_takeover"] == (
        "first_successful_policy_dds_then_exact_epoch_support_release"
    )


@pytest.mark.parametrize("invalid_duration", [True, 2.999, 3.001, float("nan")])
def test_nominal_policy_activation_requires_exact_official_duration(
    invalid_duration,
) -> None:
    with pytest.raises(ValueError, match="exactly 3s"):
        NominalPolicyActivation(duration_s=invalid_duration)


@pytest.mark.parametrize("invalid_joint_count", [True, 1.5, 0])
def test_nominal_policy_activation_requires_exact_integer_joint_count(
    invalid_joint_count,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        NominalPolicyActivation(joint_count=invalid_joint_count)


def test_nominal_policy_activation_abort_invalidates_inflight_decision():
    activation = NominalPolicyActivation(joint_count=1, duration_s=3.0)
    assert activation.begin([0.1], tick_ms=10_000) is True
    stale = activation.propose([0.0], measured=[0.1], tick_ms=11_000)
    previous_epoch = stale.epoch

    assert activation.abort() is True
    assert activation.phase == "UNPREPARED"
    assert activation.epoch == previous_epoch + 1
    assert activation.commit(stale, stale.targets) is False
    with pytest.raises(RuntimeError, match="not preparing"):
        activation.propose([0.0], measured=[0.0], tick_ms=12_000)

    assert activation.begin([0.08], tick_ms=20_000) is True
    with pytest.raises(ValueError, match="moved backwards"):
        activation.propose([0.0], measured=[0.0], tick_ms=19_999)


def test_official_nominal_ramp_preserves_captured_right_wrist_slew_limit():
    # Incident boundary for arm-local index 11 / global G1 joint 26.
    boundary = 0.010501991026103497
    activation = NominalPolicyActivation(joint_count=1, duration_s=3.0)
    guard = NavigationArmGuard(
        [0.0],
        max_deviation_rad=[1.50],
        max_velocity_rps=[4.0],
    )
    guard.anchor([boundary])

    # The captured first-policy jump is projected to—not through—the original
    # 4 rad/s limit.  No threshold is widened, and the exact projected target
    # becomes the next actuator/history anchor.
    projected = guard.apply(
        [boundary + 0.0800001],
        measured=[boundary],
        dt=0.02,
    )
    assert projected.targets == pytest.approx((boundary + 0.08,))
    assert projected.constrained_joints == 1
    assert projected.constrained_joint_indices == (0,)
    assert projected.continuous_saturation_s[0] == pytest.approx(0.02)
    assert projected.max_step_rad == pytest.approx(0.08)

    assert activation.begin([boundary], tick_ms=1_000) is True
    last = boundary
    for frame in range(151):
        decision = activation.propose(
            [0.0], measured=[boundary], tick_ms=1_000 + frame * 20,
        )
        guarded = guard.apply(
            decision.targets,
            measured=[boundary],
            dt=0.02,
        )
        assert abs(guarded.targets[0] - last) <= 0.08
        completed = activation.commit(decision, guarded.targets)
        last = guarded.targets[0]
    assert completed is True
    assert activation.phase == "SETTLING_TETHERED"
    assert last == pytest.approx(0.0)


def test_cmd_vel_is_bounded_and_converted_to_slow_walk():
    now = [100.0]
    guard = CmdVelGuard(max_linear_mps=0.25, max_angular_rps=0.5, timeout_s=0.25, clock=lambda: now[0], smoothing_alpha=1.0)
    guard.accept(2.0, 0.0, 2.0)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.speed_mps == pytest.approx(0.25)
    assert intent.has_movement is True
    assert intent.mode == "SLOW_WALK"


def test_stale_command_fails_to_idle():
    now = [100.0]
    guard = CmdVelGuard(timeout_s=0.25, clock=lambda: now[0], smoothing_alpha=1.0)
    guard.accept(0.2, 0.0, 0.0)
    now[0] = 100.3
    intent = guard.intent(dt=0.02)
    assert intent.stale is True
    assert intent.has_movement is False
    assert intent.mode == "IDLE"


def test_explicit_nav2_zero_stops_without_a_smoothing_tail():
    now = [100.0]
    guard = CmdVelGuard(timeout_s=0.25, clock=lambda: now[0], smoothing_alpha=0.5)
    guard.accept(0.0, 0.0, 0.2)
    assert guard.intent(now=100.01, dt=0.02).mode == "SLOW_WALK"
    guard.accept(0.0, 0.0, 0.0)
    stopped = guard.intent(now=100.02, dt=0.02)
    assert stopped.mode == "IDLE"
    assert stopped.has_movement is False


def test_command_zero_is_read_only_and_requires_the_filtered_tail_to_stop():
    now = [100.0]
    guard = CmdVelGuard(
        timeout_s=0.25,
        clock=lambda: now[0],
        smoothing_alpha=0.5,
    )
    assert guard.target_zero is True
    assert guard.command_zero is True

    guard.accept(0.2, 0.0, 0.0)
    assert guard.target_zero is False
    assert guard.command_zero is False
    guard.intent(now=100.01, dt=0.02)
    guard.stop(source="test_stop")
    assert guard.target_zero is True
    assert guard.command_zero is False

    stopped = guard.intent(now=100.02, dt=0.02)
    assert stopped.mode == "IDLE"
    assert guard.target_zero is True
    assert guard.command_zero is True


@pytest.mark.parametrize("raw_speed", [0.011, 0.05, 0.199])
def test_active_nav2_ramp_uses_official_forward_slow_walk_speed(raw_speed):
    now = [100.0]
    guard = CmdVelGuard(
        timeout_s=0.25,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    accepted = guard.accept(raw_speed, 0.0, 0.0)
    assert accepted.vx == pytest.approx(0.30)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.speed_mps == pytest.approx(0.30)
    assert intent.movement_angle_rad == pytest.approx(0.0)


def test_official_forward_slow_walk_speed_is_reached_on_first_filtered_tick():
    guard = CmdVelGuard(smoothing_alpha=0.35, clock=lambda: 100.0)
    guard.accept(0.20, 0.0, 0.0)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.speed_mps == pytest.approx(0.30)
    assert intent.has_movement is True


@pytest.mark.parametrize(
    ("angle", "expected_speed"),
    (
        (0.0, 0.30),
        (math.pi / 4.0, 0.30),
        (-math.pi / 4.0, 0.30),
        (math.pi / 2.0, 0.35),
        (-math.pi / 2.0, 0.35),
        (3.0 * math.pi / 4.0, 0.25),
        (-3.0 * math.pi / 4.0, 0.25),
        (math.pi, 0.20),
    ),
)
def test_official_ros2_eight_way_slow_walk_contract(angle, expected_speed):
    guard = CmdVelGuard(smoothing_alpha=1.0, clock=lambda: 100.0)
    accepted = guard.accept(math.cos(angle), math.sin(angle), 0.0)
    assert math.hypot(accepted.vx, accepted.vy) == pytest.approx(expected_speed)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.speed_mps == pytest.approx(expected_speed)
    assert CmdVelGuard._wrap_angle(intent.movement_angle_rad - angle) == pytest.approx(
        0.0
    )


def test_angular_only_nav2_command_selects_sonic_turning_plan():
    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    guard.accept(0.0, 0.0, 0.3)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.mode == "SLOW_WALK"
    assert intent.has_movement is False
    assert intent.speed_mps == pytest.approx(0.0)
    assert intent.facing_angle_rad == pytest.approx(0.006)


def test_pure_rotation_follows_official_ros2_command_integral():
    now = [100.0]
    guard = CmdVelGuard(
        max_angular_rps=0.5,
        max_facing_lead_rad=0.05,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.6)
    for frame in range(20):
        now[0] += 0.02
        guard.accept(0.0, 0.0, 0.5)
        saturated = guard.intent(now=now[0], dt=0.02)
    assert guard.command_facing == pytest.approx(0.20)
    assert saturated.facing_angle_rad == pytest.approx(0.20)

    # Measured progress is observational for a pure Spin.  It neither replaces
    # nor limits the official command-space planner facing.
    guard.observe_heading(0.7)
    now[0] += 0.02
    guard.accept(0.0, 0.0, 0.5)
    progressed = guard.intent(now=now[0], dt=0.02)
    assert guard.command_facing == pytest.approx(0.21)
    assert progressed.facing_angle_rad == pytest.approx(0.21)


def test_translating_curve_retains_measured_facing_lead_limit():
    now = [100.0]
    guard = CmdVelGuard(
        max_angular_rps=0.5,
        max_facing_lead_rad=0.05,
        max_translation_facing_offset_rad=0.05,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.6)
    for _frame in range(20):
        now[0] += 0.02
        guard.accept(0.2, 0.0, 0.5)
        intent = guard.intent(now=now[0], dt=0.02)
    assert guard.command_facing == pytest.approx(0.20)
    assert intent.facing_angle_rad == pytest.approx(0.05)


def test_small_reverse_measured_spin_does_not_reverse_command_integral():
    now = [100.0]
    guard = CmdVelGuard(
        max_angular_rps=0.25,
        max_facing_lead_rad=0.05,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.4)
    guard.accept(0.0, 0.0, 0.2)
    assert guard.intent(now=100.01, dt=0.02).facing_angle_rad == pytest.approx(0.004)

    guard.observe_heading(0.39)
    second = guard.intent(now=100.02, dt=0.02)
    assert guard.command_facing == pytest.approx(0.008)
    assert second.facing_angle_rad == pytest.approx(0.008)


def test_new_nav2_action_preserves_the_planner_canonical_facing_after_idle():
    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    guard.observe_heading(0.2)
    guard.accept(0.0, 0.0, 0.5)
    guard.intent(now=100.01, dt=0.02)
    guard.accept(0.0, 0.0, 0.0)
    guard.observe_heading(-0.4)
    guard.accept(0.2, 0.0, 0.0)
    forward = guard.intent(now=100.02, dt=0.02)
    assert forward.movement_angle_rad == pytest.approx(0.01)
    assert forward.facing_angle_rad == pytest.approx(0.01)


def test_sequential_spin_actions_accumulate_in_the_sonic_planner_frame():
    now = [100.0]
    guard = CmdVelGuard(
        max_facing_lead_rad=0.15,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.2)
    guard.accept(0.0, 0.0, 0.2)
    assert guard.intent(now=100.01, dt=0.02).facing_angle_rad == pytest.approx(0.004)
    guard.accept(0.0, 0.0, 0.0)
    guard.intent(now=100.02, dt=0.02)

    guard.observe_heading(0.35)
    guard.accept(0.0, 0.0, 0.2)
    assert guard.intent(now=100.03, dt=0.02).facing_angle_rad == pytest.approx(0.008)


def test_base_forward_command_tracks_accumulated_sonic_facing():
    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    guard.accept(0.0, 0.0, 0.5)
    turn = guard.intent(now=100.01, dt=0.02)
    assert turn.facing_angle_rad == pytest.approx(0.01)
    guard.accept(0.2, 0.0, 0.0)
    forward = guard.intent(now=100.02, dt=0.02)
    assert forward.movement_angle_rad == pytest.approx(0.01)
    assert forward.facing_angle_rad == pytest.approx(0.01)


def test_translation_drift_does_not_feed_back_into_command_integral():
    now = [100.0]
    guard = CmdVelGuard(
        max_facing_lead_rad=0.15,
        max_translation_facing_offset_rad=0.05,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.0)
    guard.accept(0.2, 0.0, 0.25)
    initial = guard.intent(now=100.01, dt=0.02)
    assert initial.facing_angle_rad == pytest.approx(0.005)

    # The emitted target remains within the measured-yaw lead bound, while the
    # hidden command goal still advances only by the second admitted wz tick.
    guard.observe_heading(0.7)
    drifted = guard.intent(now=100.02, dt=0.02)
    assert guard.command_facing == pytest.approx(0.01)
    assert drifted.facing_angle_rad == pytest.approx(0.65)
    assert drifted.movement_angle_rad == pytest.approx(0.65)

    guard.accept(0.0, 0.0, 0.0)
    idle = guard.intent(now=100.03, dt=0.02)
    assert idle.facing_angle_rad == pytest.approx(0.65)


@pytest.mark.parametrize(
    ("vy", "expected_angle"),
    [(0.80, math.pi / 2.0), (-0.80, -math.pi / 2.0)],
)
def test_lateral_commands_are_mirrored_at_official_point_three_five(
    vy,
    expected_angle,
):
    guard = CmdVelGuard(clock=lambda: 100.0, smoothing_alpha=0.35)
    guard.accept(0.0, vy, 0.0)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.speed_mps == pytest.approx(0.35)
    assert intent.movement_angle_rad == pytest.approx(expected_angle)
    assert intent.facing_angle_rad == pytest.approx(0.0)


def test_straight_translation_never_integrates_facing():
    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    guard.observe_heading(0.3)
    guard.accept(0.20, 0.0, 0.0)
    for frame in range(20):
        now[0] += 0.02
        # Reproduce the second-action incident: the torso yaws progressively
        # even though DriveOnHeading continues to command exactly zero wz.
        # Measured drift must not drag SONIC's facing goal with it.
        guard.observe_heading(0.3 + (frame + 1) * 0.015)
        guard.accept(0.20, 0.0, 0.0)
        intent = guard.intent(now=now[0], dt=0.02)
        assert intent.facing_angle_rad == pytest.approx(0.0)
        assert intent.movement_angle_rad == pytest.approx(0.0)
        assert guard.command_facing == pytest.approx(0.0)


def test_mixed_curve_facing_integral_accumulates_and_mirrors_past_old_cap():
    def run_curve(sign: float) -> tuple[float, float]:
        now = [100.0]
        guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
        heading_origin = 0.3
        guard.observe_heading(heading_origin)
        guard.accept(0.20, 0.0, sign * 0.25)
        intent = None
        for frame in range(1, 31):
            measured_progress = sign * (frame - 1) * 0.005
            guard.observe_heading(heading_origin + measured_progress)
            now[0] += 0.02
            guard.accept(0.20, 0.0, sign * 0.25)
            intent = guard.intent(now=now[0], dt=0.02)
            emitted_lead = CmdVelGuard._wrap_angle(
                intent.facing_angle_rad - measured_progress
            )
            assert abs(emitted_lead) <= 0.05 + 1e-12
        assert intent is not None
        return guard.command_facing, intent.facing_angle_rad

    positive_goal, positive_output = run_curve(1.0)
    negative_goal, negative_output = run_curve(-1.0)
    assert positive_goal > 0.05
    assert positive_goal == pytest.approx(0.15)
    assert negative_goal == pytest.approx(-positive_goal)
    assert negative_output == pytest.approx(-positive_output)


def test_spin_command_integral_opens_successive_planner_facing_segments():
    """A long official Spin must not depend on measured motion to replan."""

    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    gate = ReplanIntentGate()
    idle = MovementIntent("IDLE", 0.0, 0.0, 0.0, False, False)
    assert gate.admit(idle, now=0.0)

    guard.observe_heading(0.0)
    guard.accept(0.0, 0.0, 0.20)
    admitted_segments = []
    # Reproduce cold INIT: the command integral keeps advancing for 3 s while
    # measured yaw remains near the starting pose.  The replan gate emits each
    # new absolute planner target after its 0.05-rad/0.15-rad thresholds.
    for frame in range(150):
        now[0] += 0.02
        guard.observe_heading(0.0)
        guard.accept(0.0, 0.0, 0.20)
        candidate = guard.intent(now=now[0], dt=0.02)
        if gate.admit(candidate, now=(frame + 1) * 0.02):
            admitted_segments.append(candidate.facing_angle_rad)
    assert admitted_segments == pytest.approx([0.052, 0.204, 0.356, 0.508])
    assert guard.command_facing == pytest.approx(0.60)
    assert gate.last is not None
    assert gate.last.facing_angle_rad == pytest.approx(0.508)


@pytest.mark.parametrize("invalid_dt", [float("nan"), 0.0, 0.10])
def test_invalid_integration_dt_fails_without_advancing_facing(invalid_dt):
    guard = CmdVelGuard(clock=lambda: 100.0, smoothing_alpha=1.0)
    guard.observe_heading(0.0)
    guard.accept(0.20, 0.0, 0.25)
    first = guard.intent(now=100.01, dt=0.02)
    with pytest.raises(ValueError, match="integration dt"):
        guard.intent(now=100.02, dt=invalid_dt)
    assert guard.command_facing == pytest.approx(0.005)
    assert first.facing_angle_rad == pytest.approx(0.005)


def test_stop_freezes_and_next_action_reanchors_command_facing():
    guard = CmdVelGuard(clock=lambda: 100.0, smoothing_alpha=1.0)
    guard.observe_heading(0.0)
    guard.accept(0.20, 0.0, 0.25)
    moving = guard.intent(now=100.01, dt=0.02)
    guard.stop(source="test_revoke")
    guard.observe_heading(1.0)
    stopped = guard.intent(now=100.02, dt=0.02)
    assert stopped.facing_angle_rad == pytest.approx(moving.facing_angle_rad)
    assert guard.command_facing == pytest.approx(moving.facing_angle_rad)

    guard.accept(0.20, 0.0, -0.25)
    restarted = guard.intent(now=100.03, dt=0.02)
    assert restarted.facing_angle_rad == pytest.approx(0.0)
    assert guard.command_facing == pytest.approx(0.0)


def test_controlled_stop_preserves_official_ros2_planner_facing():
    guard = CmdVelGuard(clock=lambda: 100.0, smoothing_alpha=1.0)
    guard.observe_heading(0.40)
    guard.accept(0.20, 0.0, 0.10)
    moving = guard.intent(now=100.01, dt=0.02)
    assert moving.facing_angle_rad == pytest.approx(0.002)

    # The body makes more yaw progress after the last policy intent.  NVIDIA's
    # ROS2 adapter nevertheless retains its admitted planner-facing goal when
    # angular velocity becomes zero; measured yaw must not rewrite that goal.
    guard.observe_heading(0.46)
    stopped_target = guard.stop_at_measured_heading(
        source="controlled_stop",
        facing_angle_rad=0.37,
    )
    assert stopped_target.source == "controlled_stop"
    assert guard.target_zero is True
    assert guard.command_zero is False

    idle = guard.intent(now=100.02, dt=0.02)
    assert idle.mode == "IDLE"
    assert idle.facing_angle_rad == pytest.approx(0.37)
    assert guard.command_facing == pytest.approx(0.37)
    assert guard.command_zero is True



def test_replan_gate_debounces_small_nav2_corrections_but_stops_immediately():
    now = [10.0]
    gate = ReplanIntentGate(clock=lambda: now[0])
    moving = MovementIntent("SLOW_WALK", 0.20, 0.0, 0.0, True, False)
    assert gate.admit(moving) is True

    now[0] = 10.40
    tiny_correction = MovementIntent("SLOW_WALK", 0.21, 0.01, 0.02, True, False)
    assert gate.admit(tiny_correction) is False

    significant = MovementIntent("SLOW_WALK", 0.25, 0.0, 0.20, True, False)
    assert gate.admit(significant) is True

    now[0] = 10.41
    stopped = MovementIntent("IDLE", 0.0, 0.20, 0.20, False, False)
    assert gate.admit(stopped) is True


def test_replan_gate_rate_limits_significant_in_motion_change():
    now = [20.0]
    gate = ReplanIntentGate(clock=lambda: now[0])
    assert gate.admit(MovementIntent("SLOW_WALK", 0.20, 0.0, 0.0, True, False))
    now[0] = 20.10
    assert not gate.admit(MovementIntent("SLOW_WALK", 0.25, 0.0, 0.20, True, False))
    now[0] = 20.36
    assert gate.admit(MovementIntent("SLOW_WALK", 0.25, 0.0, 0.20, True, False))


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_replan_gate_waits_for_verified_pure_rotation_request_size(sign):
    now = [30.0]
    gate = ReplanIntentGate(clock=lambda: now[0])
    idle = MovementIntent("IDLE", 0.0, 0.40, 0.40, False, False)
    assert gate.admit(idle)

    # Reproduce the live Spin incident: Nav2's acceleration ramp first exposed
    # a 0.0014 rad facing request.  That is below the locked planner's measured
    # 0.004016 rad noise floor and must not replace the proven IDLE policy or
    # start the required-reference clock.
    tiny = MovementIntent(
        "SLOW_WALK",
        0.0,
        0.40 + sign * 0.0014,
        0.40 + sign * 0.0014,
        False,
        False,
    )
    assert gate.admit(tiny, now=30.02) is False
    assert gate.last is idle

    verified = MovementIntent(
        "SLOW_WALK",
        0.0,
        0.40 + sign * 0.05,
        0.40 + sign * 0.05,
        False,
        False,
    )
    assert gate.admit(verified, now=30.50) is True
    assert gate.last is verified


def test_non_finite_command_is_rejected():
    guard = CmdVelGuard()
    with pytest.raises(ValueError):
        guard.accept(float("nan"), 0.0, 0.0)


def test_cmd_vel_guard_uses_wall_age_but_simulation_dt_for_facing_integration():
    wall_now = [10.0]
    guard = CmdVelGuard(
        timeout_s=0.25,
        smoothing_alpha=1.0,
        clock=lambda: wall_now[0],
    )
    guard.accept(0.2, 0.0, 0.2)

    fresh = guard.intent(dt=0.02)
    assert fresh.stale is False
    assert fresh.facing_angle_rad == pytest.approx(0.004)

    # A slow simulator must not extend a stale Nav2 command in wall time.  The
    # timeout stops it, while no unearned wall-time integration is performed.
    wall_now[0] += 0.26
    stale = guard.intent(dt=0.02)
    assert stale.stale is True
    assert stale.mode == "IDLE"
    assert stale.facing_angle_rad == pytest.approx(0.004)
    assert guard.command_zero is True


def test_navigation_arm_guard_passes_valid_policy_target_unchanged():
    guard = NavigationArmGuard(
        [0.2, 0.6],
        max_deviation_rad=[0.4, 0.5],
        max_velocity_rps=[1.0, 2.0],
    )

    first = guard.apply([0.18, 0.64], measured=[0.2, 0.6], dt=0.02)
    assert first.targets == pytest.approx((0.18, 0.64))
    assert first.max_step_rad == pytest.approx(0.04)
    assert first.constrained_joints == 0

    second = guard.apply([0.16, 0.68], measured=[99.0, 99.0], dt=0.02)
    assert second.targets == pytest.approx((0.16, 0.68))
    assert second.constrained_joints == 0


def test_navigation_arm_guard_projects_excursion_and_slew():
    guard = NavigationArmGuard(
        [0.2],
        max_deviation_rad=[0.4],
        max_velocity_rps=[1.0],
    )
    with pytest.raises(ValueError, match="finite"):
        guard.apply([float("nan")], measured=[0.2], dt=0.02)
    envelope_projected = guard.apply([0.7], measured=[0.2], dt=0.02)
    assert envelope_projected.targets == pytest.approx((0.22,))
    assert envelope_projected.constrained_joints == 1
    guard.anchor([0.2])
    projected = guard.apply([0.3], measured=[0.2], dt=0.02)
    assert projected.targets == pytest.approx((0.22,))
    assert projected.constrained_joints == 1
    assert projected.max_step_rad == pytest.approx(0.02)


def test_navigation_arm_guard_accepts_asymmetric_explicit_hard_limits():
    guard = NavigationArmGuard(
        [0.2],
        lower_limits_rad=[-3.0],
        upper_limits_rad=[2.6],
        max_velocity_rps=[200.0],
    )
    guard.anchor([0.2])

    assert guard.apply([2.5], dt=0.02).targets == pytest.approx((2.5,))
    guard.anchor([0.2])
    assert guard.apply([-2.9], dt=0.02).targets == pytest.approx((-2.9,))
    guard.anchor([0.2])
    upper = guard.apply([2.6000001], dt=0.02)
    assert upper.targets == pytest.approx((2.6,))
    assert upper.constrained_joint_indices == (0,)


@pytest.mark.parametrize(
    ("lower", "upper", "message"),
    [
        (None, [1.0], "provided together"),
        ([0.0], None, "provided together"),
        ([1.0], [0.0], "ordered"),
        ([0.3], [1.0], "nominal"),
    ],
)
def test_navigation_arm_guard_rejects_invalid_explicit_limits(
    lower,
    upper,
    message,
):
    with pytest.raises(ValueError, match=message):
        NavigationArmGuard(
            [0.2],
            lower_limits_rad=lower,
            upper_limits_rad=upper,
            max_velocity_rps=[1.0],
        )


def test_navigation_arm_guard_fails_closed_on_persistent_saturation():
    guard = NavigationArmGuard(
        [0.0],
        max_deviation_rad=[10.0],
        max_velocity_rps=[1.0],
        max_continuous_saturation_s=0.25,
    )
    guard.anchor([0.0])

    for _ in range(12):
        anchor = guard._last[0]
        projected = guard.apply([anchor + 0.10], dt=0.02)
        assert projected.targets == pytest.approx((anchor + 0.02,))
        assert projected.constrained_joints == 1

    anchor = guard._last[0]
    with pytest.raises(ValueError, match="persistently") as rejected:
        guard.apply([anchor + 0.10], dt=0.02)
    assert "allowed=0.02" in str(rejected.value)
    assert "saturated_for=0.25999999999999995s" in str(rejected.value)
    assert "budget=0.25s" in str(rejected.value)


def test_navigation_arm_guard_unconstrained_tick_resets_saturation_budget():
    guard = NavigationArmGuard(
        [0.0],
        max_deviation_rad=[10.0],
        max_velocity_rps=[1.0],
        max_continuous_saturation_s=0.04,
    )
    guard.anchor([0.0])

    first = guard.apply([0.10], dt=0.02)
    assert first.constrained_joints == 1
    cleared = guard.apply([0.03], dt=0.02)
    assert cleared.constrained_joints == 0
    assert cleared.targets == pytest.approx((0.03,))
    second = guard.apply([0.13], dt=0.02)
    assert second.constrained_joints == 1
    assert second.targets == pytest.approx((0.05,))


def test_navigation_arm_guard_can_record_saturation_without_invented_deadline():
    guard = NavigationArmGuard(
        [0.0],
        max_deviation_rad=[10.0],
        max_velocity_rps=[1.0],
        max_continuous_saturation_s=None,
    )
    guard.anchor([0.0])

    result = None
    for _ in range(20):
        anchor = guard._last[0]
        result = guard.apply([anchor + 0.10], dt=0.02)
    assert result is not None
    assert result.targets == pytest.approx((0.40,))
    assert result.continuous_saturation_s == pytest.approx((0.40,))


@pytest.mark.parametrize(
    "invalid_budget",
    [True, 0.0, -0.1, float("nan"), float("inf")],
)
def test_navigation_arm_guard_rejects_invalid_saturation_budget(
    invalid_budget,
):
    with pytest.raises(ValueError, match="saturation budget"):
        NavigationArmGuard(
            [0.0],
            max_deviation_rad=[1.0],
            max_velocity_rps=[1.0],
            max_continuous_saturation_s=invalid_budget,
        )


def test_navigation_arm_guard_rejects_boolean_control_dt():
    guard = NavigationArmGuard(
        [0.0],
        max_deviation_rad=[1.0],
        max_velocity_rps=[1.0],
    )
    with pytest.raises(ValueError, match="dt"):
        guard.apply([0.0], dt=True)


def test_navigation_arm_guard_anchors_exact_actuator_preload_before_slew():
    guard = NavigationArmGuard(
        [0.0, 0.0],
        max_deviation_rad=[0.4, 0.4],
        max_velocity_rps=[1.0, 1.0],
    )
    guard.anchor([0.30, -0.30])

    first = guard.apply(
        [0.30, -0.30],
        measured=[-0.20, 0.20],
        dt=0.02,
    )
    assert first.targets == pytest.approx((0.30, -0.30))

    # Even an out-of-envelope frozen target is not rewritten.  A policy-provided
    # recovery target is accepted unchanged only when it moves toward the
    # envelope at the configured slew cap.
    guard.anchor([0.60, -0.60])
    recovery = guard.apply([0.58, -0.58], measured=[0.0, 0.0], dt=0.02)
    assert recovery.targets == pytest.approx((0.58, -0.58))


def test_base_to_camera_preserves_ros_optical_axes():
    # Optical +X points base-right (-Y), +Y points down (-Z), and +Z points
    # base-forward (+X). The map-space camera quaternion below encodes exactly
    # that fixed optical convention for an identity-oriented base.
    expected_rotation = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    import mujoco

    camera_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(camera_quaternion, expected_rotation.reshape(-1))

    relative = MujocoObservationBridge._base_to_camera(
        (1.0, 2.0, 0.8, 1.0, 0.0, 0.0, 0.0),
        (1.2, 2.0, 1.95, *camera_quaternion),
    )

    assert relative[:3] == pytest.approx((0.2, 0.0, 1.15))
    qw, qx, qy, qz = relative[3:]
    actual_rotation = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    assert actual_rotation == pytest.approx(expected_rotation)
