from __future__ import annotations

import math

import pytest

from hide_mujoco_sonic.gait_adapter import (
    BaseObservation,
    GaitAdapter,
    GaitAdapterConfig,
    GaitState,
    Nav2Twist,
)


def observation(
    timestamp: float,
    *,
    generation: int = 1,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
) -> BaseObservation:
    return BaseObservation(
        generation=generation,
        timestamp_s=timestamp,
        x=x,
        y=y,
        yaw=yaw,
        roll=roll,
        pitch=pitch,
        vx_body=vx,
        vy_body=vy,
        wz=wz,
    )


def twist(
    timestamp: float,
    *,
    generation: int = 1,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
) -> Nav2Twist:
    return Nav2Twist(
        generation=generation,
        timestamp_s=timestamp,
        vx=vx,
        vy=vy,
        wz=wz,
    )


def armed_adapter(*, config: GaitAdapterConfig | None = None) -> GaitAdapter:
    adapter = GaitAdapter(config)
    assert adapter.ingest_observation(observation(10.0), now=10.0).state == "DISARMED"
    assert adapter.arm(1, now=10.01).state == "IDLE"
    return adapter


def acknowledge_stop(
    adapter: GaitAdapter,
    *,
    first: float,
    terminal: str,
    generation: int = 1,
) -> None:
    first_intent = adapter.ingest_observation(
        observation(first, generation=generation),
        now=first,
    )
    assert first_intent.state in {"STOPPING", "FAULT"}
    final = adapter.ingest_observation(
        observation(first + 0.16, generation=generation),
        now=first + 0.16,
    )
    assert final.state == terminal


def test_walk_requires_arm_and_measured_start_ack() -> None:
    adapter = armed_adapter()

    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    assert starting.state == "STARTING"
    assert starting.mode == "SLOW_WALK"
    assert starting.motion_enabled is True
    assert starting.velocity_x_mps == pytest.approx(0.2)
    assert starting.schema_version == "h-ide.sonic-intent/v1"
    assert starting.head_yaw_rad == 0.0
    assert starting.head_pitch_rad == 0.0

    walking = adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)
    assert walking.state == "WALKING"
    assert adapter.state is GaitState.WALKING


def test_turn_is_derived_only_from_nav2_and_has_fixed_head() -> None:
    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, wz=-0.5), now=10.02)
    assert starting.state == "STARTING"
    assert starting.linear_speed_mps == 0.0
    assert starting.yaw_rate_rps == pytest.approx(-0.5)
    assert starting.facing_delta_rad == pytest.approx(-0.1)
    assert starting.source == "nav2"
    assert starting.head_yaw_rad == starting.head_pitch_rad == 0.0

    turning = adapter.ingest_observation(observation(10.03, wz=-0.08), now=10.03)
    assert turning.state == "TURNING"


def test_nav2_zero_waits_for_multiple_measured_stop_samples() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)

    stopping = adapter.ingest_twist(twist(10.04), now=10.04)
    assert stopping.state == "STOPPING"
    assert stopping.motion_enabled is False
    assert stopping.stop_requested is True
    assert stopping.stop_acknowledged is False
    assert stopping.velocity_x_mps == 0.0

    acknowledge_stop(adapter, first=10.05, terminal="IDLE")
    assert adapter.state is GaitState.IDLE


def test_cancel_stops_then_requires_a_new_generation() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    adapter.ingest_observation(observation(10.03, vx=0.05), now=10.03)

    cancelled = adapter.cancel(1, 10.04, now=10.04)
    assert cancelled.state == "STOPPING"
    assert cancelled.velocity_x_mps == 0.0
    acknowledge_stop(adapter, first=10.05, terminal="DISARMED")

    rejected = adapter.arm(1, now=10.22)
    assert rejected.state == "DISARMED"
    assert "newer generation" in (rejected.reason or "")
    assert adapter.ingest_twist(twist(10.23, vx=0.2), now=10.23).motion_enabled is False

    adapter.ingest_observation(observation(10.24, generation=2), now=10.24)
    assert adapter.arm(2, now=10.25).state == "IDLE"


def test_stale_nav2_command_fails_closed_and_disarms_after_stop_ack() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)
    # Keep measured state fresh while allowing the command timestamp to expire.
    adapter.ingest_observation(observation(10.39), now=10.39)
    stale = adapter.tick(now=10.40)
    assert stale.state == "STOPPING"
    assert stale.stop_requested is True
    assert "stale" in (stale.reason or "")

    acknowledge_stop(adapter, first=10.41, terminal="DISARMED")


def test_idle_startup_stale_observation_can_rearm_same_generation_without_replay() -> None:
    config = GaitAdapterConfig(
        command_timeout_s=1.0,
        observation_timeout_s=0.20,
    )
    adapter = armed_adapter(config=config)

    # A below-deadband command remains IDLE but proves there is buffered input
    # to discard. It must never be replayed after observation recovery.
    buffered = adapter.ingest_twist(twist(10.02, vx=0.01), now=10.02)
    assert buffered.state == "IDLE"
    assert adapter._command is not None

    stale = adapter.tick(now=10.21)
    assert stale.state == "DISARMED"
    assert stale.stop_requested is True
    assert "explicit arm" in (stale.reason or "")
    assert adapter._requires_new_generation is False
    assert adapter._command is None

    # A fresh observation alone cannot arm or move. The explicit arm is a
    # distinct transition and emits a clean, zero-motion IDLE envelope.
    observed = adapter.ingest_observation(observation(10.22), now=10.22)
    assert observed.state == "DISARMED"
    assert observed.motion_enabled is False
    recovered = adapter.arm(1, now=10.22)
    assert recovered.state == "IDLE"
    assert recovered.reason is None
    assert recovered.motion_enabled is False
    assert recovered.stop_requested is False
    assert recovered.stop_acknowledged is True
    assert recovered.velocity_x_mps == recovered.velocity_y_mps == 0.0
    assert recovered.yaw_rate_rps == 0.0

    later = adapter.tick(now=10.23)
    assert later.state == "IDLE"
    assert later.motion_enabled is False


@pytest.mark.parametrize("active_state", ["STARTING", "WALKING", "TURNING"])
def test_stale_observation_during_motion_latches_fault(active_state: str) -> None:
    config = GaitAdapterConfig(command_timeout_s=1.0, observation_timeout_s=0.20)
    adapter = armed_adapter(config=config)
    if active_state == "TURNING":
        adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
        adapter.ingest_observation(observation(10.03, wz=0.06), now=10.03)
        stale_at = 10.24
    else:
        adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
        if active_state == "WALKING":
            adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)
            stale_at = 10.24
        else:
            stale_at = 10.21
    assert adapter.state.value == active_state

    fault = adapter.tick(now=stale_at)
    assert fault.state == "FAULT"
    assert fault.motion_enabled is False
    assert fault.stop_requested is True
    assert "observation" in (fault.reason or "")
    assert adapter._requires_new_generation is True

    # Neither fresh stationary data nor explicit arm relaxes a motion fault.
    fresh_at = stale_at + 0.01
    adapter.ingest_observation(observation(fresh_at), now=fresh_at)
    assert adapter.arm(1, now=fresh_at).state == "FAULT"


def test_idle_stale_after_prior_motion_still_requires_new_generation() -> None:
    config = GaitAdapterConfig(command_timeout_s=1.0, observation_timeout_s=0.20)
    adapter = armed_adapter(config=config)
    adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)
    assert adapter.ingest_twist(twist(10.04), now=10.04).state == "STOPPING"
    acknowledge_stop(adapter, first=10.05, terminal="IDLE")

    stale = adapter.tick(now=10.42)
    assert stale.state == "DISARMED"
    assert "newer generation" in (stale.reason or "")
    assert adapter._requires_new_generation is True

    adapter.ingest_observation(observation(10.43), now=10.43)
    rejected = adapter.arm(1, now=10.43)
    assert rejected.state == "DISARMED"
    assert "newer generation" in (rejected.reason or "")


def test_humanoid_gait_phase_reverse_velocity_does_not_latch_fault() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)
    # A G1 briefly rocks backward while loading the next foot.  Eight
    # millimetres of opposite displacement must not be treated as reversed
    # navigation merely because instantaneous base velocity is negative.
    transient = adapter.ingest_observation(
        observation(10.10, vx=-0.10, x=-0.008),
        now=10.10,
    )
    assert transient.state == "WALKING"


def test_cumulative_measured_reverse_translation_latches_fault() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    adapter.ingest_observation(observation(10.03, vx=0.06, x=0.01), now=10.03)
    fault = adapter.ingest_observation(
        observation(10.10, vx=-0.08, x=-0.07),
        now=10.10,
    )
    assert fault.state == "FAULT"
    assert fault.velocity_x_mps == 0.0
    assert "opposes Nav2" in (fault.reason or "")


def test_sustained_measured_reverse_turn_latches_fault() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, wz=0.3), now=10.02)
    adapter.ingest_observation(observation(10.03, wz=0.06, yaw=0.01), now=10.03)
    adapter.ingest_observation(observation(10.04, wz=-0.08, yaw=-0.02), now=10.04)
    fault = adapter.ingest_observation(
        observation(10.10, wz=-0.08, yaw=-0.06),
        now=10.10,
    )
    assert fault.state == "FAULT"
    assert "rotation opposes Nav2" in (fault.reason or "")


@pytest.mark.parametrize(
    "bad_event",
    [
        observation(10.02, roll=0.70),
        observation(10.02, pitch=-0.70),
        observation(10.02, vx=math.nan),
    ],
)
def test_tilt_and_non_finite_observations_fail_closed(bad_event: BaseObservation) -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.015, vx=0.2), now=10.015)
    fault = adapter.ingest_observation(bad_event, now=10.02)
    assert fault.state == "FAULT"
    assert fault.motion_enabled is False


def test_non_finite_or_non_nav2_command_fails_closed() -> None:
    adapter = armed_adapter()
    fault = adapter.ingest_twist(twist(10.02, vx=math.nan), now=10.02)
    assert fault.state == "FAULT"

    adapter = armed_adapter()
    foreign = Nav2Twist(1, 10.02, 0.2, 0.0, 0.0, source="teleop")
    assert adapter.ingest_twist(foreign, now=10.02).state == "FAULT"


def test_old_generation_is_ignored_and_new_generation_invalidates_motion() -> None:
    adapter = GaitAdapter()
    adapter.ingest_observation(observation(10.0, generation=2), now=10.0)
    adapter.arm(2, now=10.01)

    old = adapter.ingest_twist(twist(10.02, generation=1, vx=0.2), now=10.02)
    assert old.state == "IDLE"
    assert old.motion_enabled is False
    assert adapter.generation == 2

    adapter.ingest_twist(twist(10.03, generation=2, vx=0.2), now=10.03)
    adapter.ingest_observation(observation(10.04, generation=2, vx=0.06), now=10.04)
    future = adapter.ingest_twist(twist(10.05, generation=3, vx=0.2), now=10.05)
    assert future.state == "DISARMED"
    assert future.stop_requested is True
    assert future.motion_enabled is False
    assert adapter.generation == 3


def test_switching_between_walk_and_turn_requires_measured_stop() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)

    switching = adapter.ingest_twist(twist(10.04, wz=0.3), now=10.04)
    assert switching.state == "STOPPING"
    assert switching.motion_enabled is False
    acknowledge_stop(adapter, first=10.05, terminal="IDLE")


def test_large_command_is_norm_bounded_without_changing_direction() -> None:
    adapter = armed_adapter()
    intent = adapter.ingest_twist(twist(10.02, vx=3.0, vy=4.0, wz=9.0), now=10.02)
    assert intent.linear_speed_mps == pytest.approx(0.35)
    assert intent.velocity_x_mps == pytest.approx(0.21)
    assert intent.velocity_y_mps == pytest.approx(0.28)
    assert intent.movement_angle_rad == pytest.approx(math.atan2(4.0, 3.0))
    assert intent.yaw_rate_rps == pytest.approx(0.60)
    assert intent.facing_delta_rad == pytest.approx(0.12)


def test_fault_does_not_rearm_in_same_generation_even_after_stop_ack() -> None:
    adapter = armed_adapter()
    adapter.ingest_twist(twist(10.02, vx=math.nan), now=10.02)
    first = adapter.ingest_observation(observation(10.03), now=10.03)
    assert first.state == "FAULT"
    acknowledged = adapter.ingest_observation(observation(10.20), now=10.20)
    assert acknowledged.state == "FAULT"
    assert acknowledged.stop_acknowledged is True
    assert adapter.arm(1, now=10.21).state == "FAULT"
