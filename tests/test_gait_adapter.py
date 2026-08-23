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


def grant_reference(
    adapter: GaitAdapter,
    starting,
    *,
    granted_at: float = 10.025,
    reference_epoch: int = 1,
    dds_sequence: int = 1,
) -> None:
    assert starting.state == "STARTING"
    assert adapter.observe_reference_authority(
        generation=int(starting.generation),
        source_sequence=starting.sequence,
        reference_epoch=reference_epoch,
        dds_target_sequence=dds_sequence,
        granted_at_monotonic_s=granted_at,
        now=granted_at,
    ) is True


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
    grant_reference(adapter, starting)

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
    grant_reference(adapter, starting)

    turning = adapter.ingest_observation(observation(10.03, wz=-0.08), now=10.03)
    assert turning.state == "TURNING"


def test_planner_wait_does_not_consume_measured_start_timeout() -> None:
    adapter = armed_adapter(
        config=GaitAdapterConfig(
            command_timeout_s=5.0,
            observation_timeout_s=5.0,
            start_timeout_s=0.50,
        )
    )
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)

    assert adapter.tick(now=11.00).state == "STARTING"
    assert adapter._start_at is None
    assert adapter.observe_reference_authority(
        generation=1,
        source_sequence=starting.sequence,
        reference_epoch=2,
        dds_target_sequence=9,
        granted_at_monotonic_s=11.00,
        now=11.00,
    ) is True
    assert adapter.tick(now=11.50).state == "STARTING"
    fault = adapter.tick(now=11.501)
    assert fault.state == "FAULT"
    assert "acknowledge gait start" in (fault.reason or "")


def test_start_ack_requires_exact_first_dds_authority_identity_and_new_observation() -> None:
    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    pre_grant = adapter.ingest_observation(
        observation(10.03, vx=0.20, x=0.004),
        now=10.03,
    )
    assert pre_grant.state == "STARTING"

    common = dict(
        generation=1,
        reference_epoch=3,
        dds_target_sequence=12,
        granted_at_monotonic_s=10.03,
        now=10.03,
    )
    assert adapter.observe_reference_authority(
        source_sequence=starting.sequence - 1,
        **common,
    ) is False
    assert adapter.observe_reference_authority(
        source_sequence=starting.sequence + 1,
        **common,
    ) is False
    assert adapter.observe_reference_authority(
        source_sequence=starting.sequence,
        granted_at_monotonic_s=10.019,
        generation=1,
        reference_epoch=3,
        dds_target_sequence=12,
        now=10.03,
    ) is False
    with pytest.raises(ValueError, match="timestamp"):
        adapter.observe_reference_authority(
            source_sequence=starting.sequence,
            generation=1,
            reference_epoch=3,
            dds_target_sequence=12,
            granted_at_monotonic_s=10.20,
            now=10.03,
        )

    assert adapter.observe_reference_authority(
        source_sequence=starting.sequence,
        **common,
    ) is True
    assert adapter.observe_reference_authority(
        generation=1,
        source_sequence=starting.sequence,
        reference_epoch=3,
        dds_target_sequence=12,
        granted_at_monotonic_s=10.031,
        now=10.031,
    ) is False
    assert adapter.ingest_observation(
        observation(10.031, vx=0.06, x=0.005),
        now=10.031,
    ).state == "WALKING"


def test_repeated_starting_emissions_keep_first_admitted_authority_sequence() -> None:
    adapter = armed_adapter()
    admitted = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    later = adapter.tick(now=10.03)
    latest = adapter.tick(now=10.04)

    assert admitted.state == later.state == latest.state == "STARTING"
    assert admitted.sequence < later.sequence < latest.sequence
    assert adapter.observe_reference_authority(
        generation=1,
        source_sequence=latest.sequence,
        reference_epoch=1,
        dds_target_sequence=1,
        granted_at_monotonic_s=10.04,
        now=10.04,
    ) is False
    assert adapter.observe_reference_authority(
        generation=1,
        source_sequence=admitted.sequence,
        reference_epoch=1,
        dds_target_sequence=1,
        granted_at_monotonic_s=10.04,
        now=10.04,
    ) is True


def test_reference_authority_reanchors_after_preparation_drift() -> None:
    adapter = armed_adapter(config=GaitAdapterConfig(max_linear_mps=0.25))
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    # Policy INIT is not navigation motion.  Its base drift must be discarded
    # when the correlated first DDS authority fact is observed.
    adapter.ingest_observation(
        observation(10.05, vx=-0.08, x=-0.060),
        now=10.05,
    )
    assert adapter.observe_reference_authority(
        generation=1,
        source_sequence=starting.sequence,
        reference_epoch=1,
        dds_target_sequence=1,
        granted_at_monotonic_s=10.025,
        now=10.05,
    ) is True
    transient = adapter.ingest_observation(
        observation(10.28, vx=-0.08, x=-0.070),
        now=10.28,
    )
    assert transient.state == "STARTING"
    assert adapter._motion_anchor == pytest.approx((-0.060, 0.0, 0.0))


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


def test_normal_stop_accepts_bounded_humanoid_balance_window() -> None:
    """Learned IDLE may balance without being instantaneously motionless."""

    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
    grant_reference(adapter, starting)
    adapter.ingest_observation(
        observation(10.03, wz=0.12, yaw=0.01),
        now=10.03,
    )
    assert adapter.ingest_twist(twist(10.04), now=10.04).state == "STOPPING"

    # These samples reproduce the shape of the r50 learned-IDLE boundary:
    # instantaneous balance corrections exceed the strict 0.04 rad/s fast
    # path, while the base remains inside a small pose envelope and has almost
    # no net displacement over the complete observation window.
    # Exact rounded samples from the final 484 ms before the r50 timeout.
    samples = (
        (10.050, -6.445416, -0.094150, 1.515677, 0.037686, 0.105262),
        (10.170, -6.444872, -0.097000, 1.503301, 0.005644, 0.155183),
        (10.246, -6.444774, -0.096264, 1.490835, 0.021879, 0.095650),
        (10.330, -6.444611, -0.094390, 1.488481, 0.013553, 0.033814),
        (10.406, -6.444617, -0.094222, 1.494865, 0.005837, 0.134527),
        (10.534, -6.443927, -0.094676, 1.511140, 0.002879, 0.089670),
    )
    result = None
    for timestamp, x, y, yaw, vx, wz in samples:
        result = adapter.ingest_observation(
            observation(timestamp, x=x, y=y, yaw=yaw, vx=vx, wz=wz),
            now=timestamp,
        )

    assert result is not None
    assert result.state == "IDLE"
    assert result.stop_acknowledged is True


def test_normal_stop_balance_window_handles_wrapped_yaw() -> None:
    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
    grant_reference(adapter, starting)
    adapter.ingest_observation(
        observation(10.03, wz=0.12, yaw=3.12),
        now=10.03,
    )
    adapter.ingest_twist(twist(10.04), now=10.04)

    samples = (
        (10.05, 3.130, 0.12),
        (10.15, 3.139, 0.09),
        (10.25, -3.137, 0.07),
        (10.35, 3.139, -0.07),
        (10.45, 3.134, -0.06),
        (10.55, 3.132, -0.05),
    )
    result = None
    for timestamp, yaw, wz in samples:
        result = adapter.ingest_observation(
            observation(timestamp, yaw=yaw, vx=0.03, wz=wz),
            now=timestamp,
        )

    assert result is not None
    assert result.state == "IDLE"


@pytest.mark.parametrize(
    ("motion", "reason_fragment"),
    [
        ("translation", "stop before timeout"),
        ("rotation", "stop before timeout"),
    ],
)
def test_normal_stop_balance_window_rejects_continued_motion(
    motion: str,
    reason_fragment: str,
) -> None:
    config = GaitAdapterConfig(stop_timeout_s=0.80)
    adapter = armed_adapter(config=config)
    if motion == "translation":
        starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
        first_observation = observation(10.03, vx=0.06, x=0.001)
    else:
        starting = adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
        first_observation = observation(10.03, wz=0.08, yaw=0.001)
    grant_reference(adapter, starting)
    adapter.ingest_observation(first_observation, now=10.03)
    adapter.ingest_twist(twist(10.04), now=10.04)

    result = None
    for index in range(1, 8):
        timestamp = 10.04 + 0.10 * index
        if motion == "translation":
            measured = observation(
                timestamp,
                x=0.004 * index,
                vx=0.04,
            )
        else:
            measured = observation(
                timestamp,
                yaw=0.008 * index,
                wz=0.08,
            )
        result = adapter.ingest_observation(measured, now=timestamp)
        assert result.state == "STOPPING"

    fault = adapter.tick(now=10.85)
    assert fault.state == "FAULT"
    assert reason_fragment in (fault.reason or "")


def test_normal_stop_balance_window_restarts_after_gap_or_large_correction() -> None:
    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
    grant_reference(adapter, starting)
    adapter.ingest_observation(observation(10.03, wz=0.12), now=10.03)
    adapter.ingest_twist(twist(10.04), now=10.04)

    for timestamp in (10.05, 10.15, 10.25):
        adapter.ingest_observation(
            observation(timestamp, wz=0.08),
            now=timestamp,
        )
    # A >200 ms capture gap cannot be bridged, and a correction beyond the
    # bounded learned-balance speed clears the newly-started window again.
    assert adapter.ingest_observation(
        observation(10.50, wz=0.08),
        now=10.50,
    ).state == "STOPPING"
    assert adapter.ingest_observation(
        observation(10.60, wz=0.21),
        now=10.60,
    ).state == "STOPPING"

    result = None
    for index, timestamp in enumerate((10.70, 10.80, 10.90, 11.00, 11.10, 11.20)):
        result = adapter.ingest_observation(
            observation(
                timestamp,
                yaw=0.002 * (index % 3),
                vx=0.03,
                wz=0.08 if index % 2 == 0 else -0.08,
            ),
            now=timestamp,
        )
    assert result is not None
    assert result.state == "IDLE"


def test_safe_stop_does_not_use_normal_balanced_idle_window() -> None:
    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
    grant_reference(adapter, starting)
    adapter.ingest_observation(observation(10.03, wz=0.12), now=10.03)
    assert adapter.request_safe_stop(1, now=10.04).state == "STOPPING"

    result = None
    for index, timestamp in enumerate((10.05, 10.15, 10.25, 10.35, 10.45, 10.55)):
        result = adapter.ingest_observation(
            observation(
                timestamp,
                yaw=0.002 * (index % 3),
                vx=0.03,
                wz=0.08 if index % 2 == 0 else -0.08,
            ),
            now=timestamp,
        )
    assert result is not None
    assert result.state == "STOPPING"
    assert result.stop_acknowledged is False

    first_strict = adapter.ingest_observation(observation(10.65), now=10.65)
    assert first_strict.state == "STOPPING"
    confirmed = adapter.ingest_observation(observation(10.81), now=10.81)
    assert confirmed.state == "DISARMED"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"balanced_idle_stop_samples": True},
        {"balanced_idle_stop_window_s": 2.0},
        {"balanced_idle_stop_max_sample_gap_s": 0.45},
        {"balanced_idle_stop_max_linear_mps": 0.025},
        {"balanced_idle_stop_max_angular_rps": 0.60},
    ],
)
def test_balanced_idle_stop_config_rejects_unsafe_or_impossible_limits(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="balanced IDLE|balanced_idle"):
        GaitAdapterConfig(**kwargs)


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


def _adapter_in_nonfault_state(state: str) -> tuple[GaitAdapter, float]:
    if state == "DISARMED":
        adapter = GaitAdapter()
        adapter.advance_generation(1, now=10.0)
        adapter.ingest_observation(observation(10.01), now=10.01)
        return adapter, 10.02

    adapter = armed_adapter()
    if state == "IDLE":
        return adapter, 10.02
    if state == "STARTING":
        adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
        return adapter, 10.03
    if state == "WALKING":
        starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
        grant_reference(adapter, starting)
        adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)
        return adapter, 10.04
    if state == "TURNING":
        starting = adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
        grant_reference(adapter, starting)
        adapter.ingest_observation(observation(10.03, wz=0.06), now=10.03)
        return adapter, 10.04
    if state == "STOPPING":
        adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
        adapter.ingest_twist(twist(10.03), now=10.03)
        # This pre-transaction stationary sample must be discarded by the
        # correlated safe-stop request below.
        adapter.ingest_observation(observation(10.04), now=10.04)
        return adapter, 10.05
    raise AssertionError(f"unsupported test state {state}")


@pytest.mark.parametrize(
    "initial_state",
    ["DISARMED", "IDLE", "STARTING", "WALKING", "TURNING", "STOPPING"],
)
def test_request_safe_stop_requires_post_transaction_stationary_samples_from_every_nonfault_state(
    initial_state: str,
) -> None:
    adapter, requested_at = _adapter_in_nonfault_state(initial_state)

    requested = adapter.request_safe_stop(1, now=requested_at)
    assert requested.state == "STOPPING"
    assert requested.stop_requested is True
    assert requested.stop_acknowledged is False
    assert requested.motion_enabled is False

    first_at = requested_at + 0.01
    first = adapter.ingest_observation(observation(first_at), now=first_at)
    assert first.state == "STOPPING"
    assert first.stop_acknowledged is False

    final_at = first_at + 0.16
    final = adapter.ingest_observation(observation(final_at), now=final_at)
    assert final.state == "DISARMED"
    assert final.stop_acknowledged is True
    assert adapter.arm(1, now=final_at + 0.01).state == "DISARMED"


def test_request_safe_stop_retry_does_not_reset_stationary_evidence() -> None:
    adapter = armed_adapter()
    requested = adapter.request_safe_stop(1, now=10.02)
    first = adapter.ingest_observation(observation(10.03), now=10.03)
    assert first.state == "STOPPING"

    repeated = adapter.request_safe_stop(1, now=10.10)
    assert repeated.state == "STOPPING"
    assert repeated.sequence > requested.sequence
    final = adapter.ingest_observation(observation(10.19), now=10.19)
    assert final.state == "DISARMED"
    assert final.stop_acknowledged is True


def test_request_safe_stop_rejects_delayed_pre_request_capture_timestamps() -> None:
    adapter = armed_adapter()
    requested = adapter.request_safe_stop(1, now=10.20)
    assert requested.state == "STOPPING"

    delayed_first = adapter.ingest_observation(
        observation(10.05),
        now=10.21,
    )
    delayed_second = adapter.ingest_observation(
        observation(10.19),
        now=10.37,
    )
    assert delayed_first.state == delayed_second.state == "STOPPING"
    assert delayed_first.stop_acknowledged is False
    assert delayed_second.stop_acknowledged is False

    post_request_first = adapter.ingest_observation(
        observation(10.21),
        now=10.38,
    )
    assert post_request_first.state == "STOPPING"
    assert post_request_first.stop_acknowledged is False
    confirmed = adapter.ingest_observation(
        observation(10.37),
        now=10.54,
    )
    assert confirmed.state == "DISARMED"
    assert confirmed.stop_acknowledged is True


def test_request_safe_stop_uses_capture_span_not_callback_arrival_span() -> None:
    adapter = armed_adapter()
    adapter.request_safe_stop(1, now=10.20)

    first = adapter.ingest_observation(observation(10.201), now=10.21)
    delayed_arrival = adapter.ingest_observation(
        observation(10.202),
        now=10.37,
    )
    assert first.state == delayed_arrival.state == "STOPPING"
    assert delayed_arrival.stop_acknowledged is False

    confirmed = adapter.ingest_observation(
        observation(10.36),
        now=10.38,
    )
    assert confirmed.state == "DISARMED"
    assert confirmed.stop_acknowledged is True


def test_request_safe_stop_stale_captures_break_stationary_evidence() -> None:
    adapter = armed_adapter()
    adapter.request_safe_stop(1, now=10.20)

    stale_first = adapter.ingest_observation(observation(10.21), now=12.00)
    stale_second = adapter.ingest_observation(observation(10.37), now=12.10)
    assert stale_first.state == stale_second.state == "STOPPING"
    assert stale_second.stop_acknowledged is False

    fresh_first = adapter.ingest_observation(observation(12.11), now=12.11)
    assert fresh_first.state == "STOPPING"
    assert fresh_first.stop_acknowledged is False
    confirmed = adapter.ingest_observation(observation(12.27), now=12.27)
    assert confirmed.state == "DISARMED"
    assert confirmed.stop_acknowledged is True


def test_request_safe_stop_never_converts_fault_to_normal_stop() -> None:
    adapter = armed_adapter()
    fault = adapter.ingest_twist(twist(10.02, vx=math.nan), now=10.02)
    assert fault.state == "FAULT"
    fault_started_at = adapter._stop_started_at

    repeated = adapter.request_safe_stop(1, now=10.03)
    assert repeated.state == "FAULT"
    assert adapter._safe_stop_requested is False
    assert adapter._stop_started_at == fault_started_at


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
        starting = adapter.ingest_twist(twist(10.02, wz=0.2), now=10.02)
        grant_reference(adapter, starting)
        adapter.ingest_observation(observation(10.03, wz=0.06), now=10.03)
        stale_at = 10.24
    else:
        starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
        if active_state == "WALKING":
            grant_reference(adapter, starting)
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
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    grant_reference(adapter, starting)
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
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    grant_reference(adapter, starting)
    adapter.ingest_observation(observation(10.03, vx=0.06), now=10.03)
    # A single sample may be a centre-of-mass or foot-placement transient; its
    # cause is unknown without contact evidence.  Eight millimetres of opposite
    # displacement is not meaningful cumulative reverse navigation.
    transient = adapter.ingest_observation(
        observation(10.10, vx=-0.10, x=-0.008),
        now=10.10,
    )
    assert transient.state == "WALKING"


def test_cumulative_measured_reverse_translation_latches_fault() -> None:
    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    grant_reference(adapter, starting)
    adapter.ingest_observation(observation(10.03, vx=0.06, x=0.01), now=10.03)
    fault = adapter.ingest_observation(
        observation(10.10, vx=-0.08, x=-0.081),
        now=10.10,
    )
    assert fault.state == "FAULT"
    assert fault.velocity_x_mps == 0.0
    assert "opposes Nav2" in (fault.reason or "")


def test_reverse_startup_transient_must_end_within_one_second() -> None:
    adapter = armed_adapter(
        config=GaitAdapterConfig(command_timeout_s=5.0, observation_timeout_s=5.0)
    )
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    grant_reference(adapter, starting, granted_at=10.025)

    within_grace = adapter.ingest_observation(
        observation(11.00, vx=-0.03, x=-0.02),
        now=11.00,
    )
    assert within_grace.state == "STARTING"

    fault = adapter.ingest_observation(
        observation(11.03, vx=-0.03, x=-0.021),
        now=11.03,
    )
    assert fault.state == "FAULT"
    assert "startup grace" in (fault.reason or "")


def test_authorized_walk_requires_net_positive_progress_within_two_seconds() -> None:
    adapter = armed_adapter(
        config=GaitAdapterConfig(command_timeout_s=5.0, observation_timeout_s=5.0)
    )
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    grant_reference(adapter, starting, granted_at=10.025)
    # Positive velocity alone acknowledges the gait state, but millimetre-scale
    # pose noise cannot satisfy the independent progress deadline.
    walking = adapter.ingest_observation(
        observation(10.05, vx=0.06, x=0.001),
        now=10.05,
    )
    assert walking.state == "WALKING"
    fault = adapter.ingest_observation(
        observation(12.03, vx=0.03, x=0.005),
        now=12.03,
    )
    assert fault.state == "FAULT"
    assert "no net positive" in (fault.reason or "")


def test_sub_threshold_pose_noise_does_not_become_reverse_motion() -> None:
    adapter = armed_adapter(
        config=GaitAdapterConfig(command_timeout_s=5.0, observation_timeout_s=5.0)
    )
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    grant_reference(adapter, starting, granted_at=10.025)

    noisy = adapter.ingest_observation(
        observation(11.03, vx=0.0, x=-1e-6),
        now=11.03,
    )
    assert noisy.state == "STARTING"

    fault = adapter.ingest_observation(
        observation(12.03, vx=0.0, x=-1e-6),
        now=12.03,
    )
    assert fault.state == "FAULT"
    assert "no net positive" in (fault.reason or "")


def test_positive_progress_before_deadline_disarms_startup_progress_fault() -> None:
    adapter = armed_adapter(
        config=GaitAdapterConfig(command_timeout_s=5.0, observation_timeout_s=5.0)
    )
    starting = adapter.ingest_twist(twist(10.02, vx=0.2), now=10.02)
    grant_reference(adapter, starting, granted_at=10.025)
    assert adapter.ingest_observation(
        observation(10.20, vx=0.06, x=0.012),
        now=10.20,
    ).state == "WALKING"
    assert adapter.ingest_observation(
        observation(12.20, vx=0.03, x=0.013),
        now=12.20,
    ).state == "WALKING"


def test_sustained_measured_reverse_turn_latches_fault() -> None:
    adapter = armed_adapter()
    starting = adapter.ingest_twist(twist(10.02, wz=0.3), now=10.02)
    grant_reference(adapter, starting)
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
