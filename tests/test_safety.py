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


def test_plant_control_clock_accepts_long_running_4ms_mujoco_tick_stream():
    clock = PlantControlClock(control_hz=50)
    simulation_time_s = 0.0
    admitted = []

    # The bridge uses int(simulation_time * 1000).  Repeated binary floating
    # point addition occasionally produces a 3 ms or 5 ms tick increment even
    # though MuJoCo's configured physics step is exactly 4 ms.
    for _ in range(25_000):
        step = clock.observe(int(simulation_time_s * 1_000))
        if step is not None:
            admitted.append(step)
        simulation_time_s += 0.004

    assert len(admitted) == 5_000
    assert {step.dt_s for step in admitted} <= {0.02, 0.021, 0.022, 0.023, 0.024}


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


def test_static_joint_handoff_holds_idle_then_blends_policy():
    now = [1.0]
    handoff = StaticJointHandoff(joint_count=2, warmup_s=0.3, blend_s=0.4, clock=lambda: now[0])
    idle = handoff.decide(motion_active=False, measured=[1.0, 2.0], policy=[9.0, 9.0])
    assert idle.targets == pytest.approx((1.0, 2.0))
    assert idle.static_hold_active is True

    now[0] = 2.0
    start = handoff.decide(motion_active=True, measured=[1.1, 2.1], policy=[3.0, 4.0])
    assert start.targets == pytest.approx((1.1, 2.1))
    assert start.motion_handoff_active is True
    now[0] = 2.5
    blend = handoff.decide(motion_active=True, measured=[1.2, 2.2], policy=[3.0, 4.0])
    assert blend.handoff_ratio == pytest.approx(0.5)
    assert blend.targets == pytest.approx((2.05, 3.05))
    now[0] = 2.8
    active = handoff.decide(motion_active=True, measured=[1.3, 2.3], policy=[3.0, 4.0])
    assert active.targets == pytest.approx((3.0, 4.0))
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


def test_sub_sonic_terminal_velocity_is_held_idle():
    now = [100.0]
    guard = CmdVelGuard(
        max_linear_mps=0.35,
        min_walk_command_mps=0.10,
        timeout_s=0.25,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.accept(0.05, 0.0, 0.0)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.has_movement is False
    assert intent.mode == "IDLE"


def test_angular_only_nav2_command_selects_sonic_turning_plan():
    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    guard.accept(0.0, 0.0, 0.3)
    intent = guard.intent(now=100.01, dt=0.02)
    assert intent.mode == "SLOW_WALK"
    assert intent.has_movement is False
    assert intent.speed_mps == pytest.approx(0.0)
    assert intent.facing_angle_rad == pytest.approx(0.056)


def test_angular_command_starts_from_measured_heading_and_caps_open_loop_lead():
    now = [100.0]
    guard = CmdVelGuard(
        max_angular_rps=0.5,
        max_facing_lead_rad=0.35,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.6)
    guard.accept(0.0, 0.0, 0.5)
    initial = guard.intent(now=100.01, dt=0.2)
    assert initial.facing_angle_rad == pytest.approx(0.35)

    saturated = guard.intent(now=100.02, dt=2.0)
    assert saturated.facing_angle_rad == pytest.approx(0.35)

    guard.observe_heading(0.7)
    progressed = guard.intent(now=100.03, dt=0.02)
    assert progressed.facing_angle_rad == pytest.approx(0.36)


def test_reverse_measured_spin_cannot_reverse_the_nav2_facing_target():
    now = [100.0]
    guard = CmdVelGuard(
        max_angular_rps=0.25,
        max_facing_lead_rad=0.05,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.4)
    guard.accept(0.0, 0.0, 0.2)
    assert guard.intent(now=100.01, dt=0.02).facing_angle_rad == pytest.approx(0.05)

    # A physical response in the wrong direction is observation evidence for
    # the bridge interlock; it must never make a left Nav2 command become a
    # right-facing SONIC target while that interlock is being delivered.
    guard.observe_heading(0.3)
    assert guard.intent(now=100.02, dt=0.02).facing_angle_rad == pytest.approx(0.05)


def test_new_nav2_action_preserves_the_planner_canonical_facing_after_idle():
    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    guard.observe_heading(0.2)
    guard.accept(0.0, 0.0, 0.5)
    guard.intent(now=100.01, dt=0.2)
    guard.accept(0.0, 0.0, 0.0)
    guard.observe_heading(-0.4)
    guard.accept(0.2, 0.0, 0.0)
    forward = guard.intent(now=100.02, dt=0.02)
    assert forward.movement_angle_rad == pytest.approx(0.05)
    assert forward.facing_angle_rad == pytest.approx(0.05)


def test_sequential_spin_actions_accumulate_in_the_sonic_planner_frame():
    now = [100.0]
    guard = CmdVelGuard(
        max_facing_lead_rad=0.15,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.2)
    guard.accept(0.0, 0.0, 0.2)
    assert guard.intent(now=100.01, dt=0.02).facing_angle_rad == pytest.approx(0.15)
    guard.accept(0.0, 0.0, 0.0)
    guard.intent(now=100.02, dt=0.02)

    guard.observe_heading(0.35)
    guard.accept(0.0, 0.0, 0.2)
    assert guard.intent(now=100.03, dt=0.02).facing_angle_rad == pytest.approx(0.30)


def test_base_forward_command_tracks_accumulated_sonic_facing():
    now = [100.0]
    guard = CmdVelGuard(clock=lambda: now[0], smoothing_alpha=1.0)
    guard.accept(0.0, 0.0, 0.5)
    turn = guard.intent(now=100.01, dt=0.4)
    assert turn.facing_angle_rad == pytest.approx(0.25)
    guard.accept(0.2, 0.0, 0.0)
    forward = guard.intent(now=100.02, dt=0.02)
    assert forward.movement_angle_rad == pytest.approx(0.25)
    assert forward.facing_angle_rad == pytest.approx(0.25)


def test_translation_correction_does_not_follow_measured_yaw_drift():
    now = [100.0]
    guard = CmdVelGuard(
        max_facing_lead_rad=0.15,
        max_translation_facing_offset_rad=0.05,
        clock=lambda: now[0],
        smoothing_alpha=1.0,
    )
    guard.observe_heading(0.0)
    guard.accept(0.2, 0.0, 0.25)
    initial = guard.intent(now=100.01, dt=1.0)
    assert initial.facing_angle_rad == pytest.approx(0.05)

    # A large physical overshoot is evidence for a safety interlock; it must
    # not become a new movement/facing target and create positive feedback.
    guard.observe_heading(0.7)
    drifted = guard.intent(now=100.02, dt=1.0)
    assert drifted.facing_angle_rad == pytest.approx(0.05)
    assert drifted.movement_angle_rad == pytest.approx(0.05)

    guard.accept(0.0, 0.0, 0.0)
    idle = guard.intent(now=100.03, dt=0.02)
    assert idle.facing_angle_rad == pytest.approx(0.05)


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


def test_navigation_arm_guard_clamps_policy_excursion_and_slew():
    guard = NavigationArmGuard(
        [0.2, 0.6],
        max_deviation_rad=[0.4, 0.5],
        max_velocity_rps=[1.0, 2.0],
    )

    first = guard.apply([-3.0, 2.0], measured=[0.2, 0.6], dt=0.02)
    assert first.targets == pytest.approx((0.18, 0.64))
    assert first.max_step_rad == pytest.approx(0.04)
    assert first.constrained_joints == 2

    second = guard.apply([-3.0, 2.0], measured=[99.0, 99.0], dt=0.02)
    assert second.targets == pytest.approx((0.16, 0.68))
    assert second.constrained_joints == 2


def test_navigation_arm_guard_recovers_from_non_finite_target():
    guard = NavigationArmGuard(
        [0.2],
        max_deviation_rad=[0.4],
        max_velocity_rps=[1.0],
    )
    result = guard.apply([float("nan")], measured=[0.2], dt=0.02)
    assert result.targets == pytest.approx((0.2,))
    assert result.constrained_joints == 1


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
