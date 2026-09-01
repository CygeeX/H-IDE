from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import UUID

import pytest

from hide_mujoco_sonic.gait_adapter import GaitState
from hide_mujoco_sonic.motion_fault import (
    MOTION_FAULT_SCHEMA_VERSION,
    MotionFaultLatch,
    MotionSafetyFault,
    SonicHealthFaultGate,
    SonicHealthTransportGate,
    SonicHealth,
    classify_sonic_health,
    motion_fault_qos,
    require_fresh_motion_health,
    sonic_health_probe,
    validate_sonic_health_request,
)


EVENT_ID = UUID("10000000-0000-4000-8000-000000000001")
BRIDGE_BOOT_ID = UUID("20000000-0000-4000-8000-000000000002")
DRIVER_BOOT_ID = UUID("30000000-0000-4000-8000-000000000003")
REQUEST_ID = UUID("40000000-0000-4000-8000-000000000004")
NEW_DRIVER_BOOT_ID = UUID("50000000-0000-4000-8000-000000000005")


def event() -> MotionSafetyFault:
    return MotionSafetyFault(
        event_id=EVENT_ID,
        robot_id="g1-sim",
        localization_epoch_id="g1-sim-map-epoch-1",
        producer_boot_id=BRIDGE_BOOT_ID,
        source_sequence=1,
        control_generation=42,
        source="gait_adapter",
        code="GAIT_ADAPTER_FAULT",
        reason="measured reverse motion exceeded the safety bound",
        detected_at="2026-08-19T10:00:00Z",
        downstream_boot_id=DRIVER_BOOT_ID,
        probe_request_id=REQUEST_ID,
    )


def health_response(
    *,
    status: str = "ready",
    terminal_fault: bool = False,
    reason: str | None = None,
    motion_ready: bool = True,
    age: float | None = 0.02,
    control_age: float | None = 0.01,
    driver_boot_id: UUID = DRIVER_BOOT_ID,
    plant_support_mode: str = "idle",
    plant_support_safety_latched: bool = False,
    plant_support_stop_pending: bool = False,
    static_hold_active: bool = True,
    navigation_task_active: bool = False,
    motion_authorized: bool = True,
    nav2_command_zero: bool = True,
    controlled_policy_stop_active: bool = False,
    matched_reference_actuator_authority: bool = False,
    reference_authority_state: str = "REVOKED",
    reference_authority_generation: int | None = None,
    reference_authority_source_sequence: int | None = None,
    reference_authority_intent_generation: int | None = None,
    reference_authority_epoch: int = 0,
    reference_authority_granted_at_monotonic_s: float | None = None,
    reference_authority_granted_tick_ms: int | None = None,
    reference_authority_dds_target_sequence: int | None = None,
) -> dict[str, object]:
    return {
        "request_id": str(REQUEST_ID),
        "ok": True,
        "schema_version": "h-ide.sonic-health/v4",
        "robot_id": "g1-sim",
        "driver_boot_id": str(driver_boot_id),
        "driver_session_id": "31000000-0000-4000-8000-000000000003",
        "status": status,
        "terminal_fault": terminal_fault,
        "reason": reason,
        "motion_ready": motion_ready,
        "last_lowstate_age_s": age,
        "last_control_step_age_s": control_age,
        "plant_support_mode": plant_support_mode,
        "plant_support_safety_latched": plant_support_safety_latched,
        "plant_support_stop_pending": plant_support_stop_pending,
        "static_hold_active": static_hold_active,
        "navigation_task_active": navigation_task_active,
        "motion_authorized": motion_authorized,
        "nav2_command_zero": nav2_command_zero,
        "controlled_policy_stop_active": controlled_policy_stop_active,
        "matched_reference_actuator_authority": matched_reference_actuator_authority,
        "reference_authority_state": reference_authority_state,
        "reference_authority_generation": reference_authority_generation,
        "reference_authority_source_sequence": reference_authority_source_sequence,
        "reference_authority_intent_generation": reference_authority_intent_generation,
        "reference_authority_epoch": reference_authority_epoch,
        "reference_authority_granted_at_monotonic_s": (
            reference_authority_granted_at_monotonic_s
        ),
        "reference_authority_granted_tick_ms": reference_authority_granted_tick_ms,
        "reference_authority_dds_target_sequence": (
            reference_authority_dds_target_sequence
        ),
    }


def parse_health(**overrides) -> SonicHealth:
    return SonicHealth.from_response(
        health_response(**overrides),
        expected_request_id=REQUEST_ID,
        expected_robot_id="g1-sim",
    )


def health_fault_gate() -> SonicHealthFaultGate:
    return SonicHealthFaultGate(
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )


def health_transport_gate() -> SonicHealthTransportGate:
    return SonicHealthTransportGate(
        probe_period_s=0.025,
        probe_timeout_s=0.020,
    )


def observe(
    gate: SonicHealthFaultGate,
    health: SonicHealth,
    *,
    gait_state: GaitState = GaitState.IDLE,
):
    return gate.observe(health, gait_state=gait_state)


def safe_hold_health(**overrides) -> SonicHealth:
    values = {
        "plant_support_mode": "safety_stop",
        "plant_support_safety_latched": True,
        "plant_support_stop_pending": False,
        "static_hold_active": True,
        "navigation_task_active": False,
        "motion_authorized": False,
        "nav2_command_zero": True,
    }
    values.update(overrides)
    return parse_health(**values)


def test_motion_fault_v1_is_strict_and_round_trips() -> None:
    value = event()
    encoded = value.to_json()
    decoded = json.loads(encoded)

    assert decoded["schemaVersion"] == MOTION_FAULT_SCHEMA_VERSION
    assert decoded["latched"] is True
    assert decoded["eventId"] == str(EVENT_ID)
    assert decoded["producerBootId"] == str(BRIDGE_BOOT_ID)
    assert decoded["sourceSequence"] == 1
    assert decoded["controlGeneration"] == 42
    assert MotionSafetyFault.from_json(encoded) == value

    decoded["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        MotionSafetyFault.from_json(json.dumps(decoded))

    del decoded["unexpected"]
    decoded["latched"] = False
    with pytest.raises(ValueError, match="must be latched"):
        MotionSafetyFault.from_json(json.dumps(decoded))


def test_fault_latch_publishes_before_stop_and_deduplicates() -> None:
    order: list[str] = []
    published: list[MotionSafetyFault] = []
    latch = MotionFaultLatch(
        robot_id="g1-sim",
        localization_epoch_id="g1-sim-map-epoch-1",
        producer_boot_id=BRIDGE_BOOT_ID,
        clock=lambda: datetime(2026, 8, 19, 10, 0, tzinfo=timezone.utc),
        event_id_factory=lambda: EVENT_ID,
    )

    def publish(value: MotionSafetyFault) -> None:
        order.append("publish")
        published.append(value)

    def stop() -> None:
        order.append("stop")

    first = latch.trip(
        control_generation=42,
        source="gait_adapter",
        code="GAIT_ADAPTER_FAULT",
        reason="first cause",
        publish=publish,
        local_stop=stop,
    )
    repeated = latch.trip(
        control_generation=99,
        source="sonic_driver",
        code="SONIC_UNAVAILABLE",
        reason="later cause must not replace the first",
        publish=publish,
        local_stop=stop,
    )

    assert order == ["publish", "stop"]
    assert published == [first]
    assert repeated is first
    assert first.event_id == EVENT_ID
    assert first.producer_boot_id == BRIDGE_BOOT_ID
    assert first.source_sequence == 1
    assert first.control_generation == 42
    assert first.reason == "first cause"
    assert latch.complete is True


def test_fault_latch_reset_requires_exact_completed_event_and_keeps_sequence() -> None:
    event_ids = iter(
        [
            EVENT_ID,
            UUID("10000000-0000-4000-8000-000000000002"),
        ]
    )
    latch = MotionFaultLatch(
        robot_id="g1-sim",
        localization_epoch_id="epoch-1",
        producer_boot_id=BRIDGE_BOOT_ID,
        event_id_factory=lambda: next(event_ids),
    )
    first = latch.trip(
        control_generation=7,
        source="gait_adapter",
        code="GAIT_ADAPTER_FAULT",
        reason="first",
        publish=lambda _event: None,
        local_stop=lambda: None,
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        latch.reset(UUID("90000000-0000-4000-8000-000000000009"))
    assert latch.reset(first.event_id) is first
    assert latch.event is None
    assert latch.complete is False

    second = latch.trip(
        control_generation=8,
        source="navigation_guard",
        code="NAVIGATION_GUARD_FAULT",
        reason="second",
        publish=lambda _event: None,
        local_stop=lambda: None,
    )
    assert second.source_sequence == 2
    assert second.event_id != first.event_id


def test_failed_publication_still_stops_and_retries_the_same_event() -> None:
    attempts: list[MotionSafetyFault] = []
    stops = 0
    latch = MotionFaultLatch(
        robot_id="g1-sim",
        localization_epoch_id="epoch-1",
        producer_boot_id=BRIDGE_BOOT_ID,
        event_id_factory=lambda: EVENT_ID,
    )

    def fail_publish(value: MotionSafetyFault) -> None:
        attempts.append(value)
        raise RuntimeError("DDS unavailable")

    def stop() -> None:
        nonlocal stops
        stops += 1

    with pytest.raises(RuntimeError, match="publication failed"):
        latch.trip(
            control_generation=7,
            source="sonic_driver",
            code="SONIC_UNAVAILABLE",
            reason="command socket disconnected",
            publish=fail_publish,
            local_stop=stop,
        )
    assert stops == 1
    assert latch.complete is False

    delivered: list[MotionSafetyFault] = []
    retried = latch.trip(
        control_generation=8,
        source="mujoco_plant",
        code="PLANT_LOWSTATE_STALE",
        reason="later cause",
        publish=delivered.append,
        local_stop=stop,
    )
    assert stops == 1
    assert attempts[0] is retried
    assert delivered == [retried]
    assert retried.event_id == EVENT_ID
    assert retried.control_generation == 7
    assert retried.code == "SONIC_UNAVAILABLE"
    assert latch.complete is True


def test_failed_local_stop_retries_without_republishing() -> None:
    published: list[MotionSafetyFault] = []
    stop_attempts = 0
    latch = MotionFaultLatch(
        robot_id="g1-sim",
        localization_epoch_id="epoch-1",
        producer_boot_id=BRIDGE_BOOT_ID,
        event_id_factory=lambda: EVENT_ID,
    )

    def stop() -> None:
        nonlocal stop_attempts
        stop_attempts += 1
        if stop_attempts == 1:
            raise RuntimeError("Plant acknowledgement unavailable")

    with pytest.raises(RuntimeError, match="local stop failed"):
        latch.trip(
            control_generation=7,
            source="mujoco_plant",
            code="PLANT_LOWSTATE_STALE",
            reason="LowState stale",
            publish=published.append,
            local_stop=stop,
        )

    assert latch.complete is False
    retried = latch.trip(
        control_generation=8,
        source="sonic_driver",
        code="SONIC_UNAVAILABLE",
        reason="later cause",
        publish=published.append,
        local_stop=stop,
    )
    assert published == [retried]
    assert stop_attempts == 2
    assert retried.code == "PLANT_LOWSTATE_STALE"
    assert latch.complete is True


def test_sonic_probe_contract_and_health_classification_are_strict() -> None:
    probe = sonic_health_probe(
        robot_id="g1-sim",
        producer_boot_id=BRIDGE_BOOT_ID,
        request_id=REQUEST_ID,
    )
    assert validate_sonic_health_request(
        probe,
        expected_robot_id="g1-sim",
    ) == (REQUEST_ID, BRIDGE_BOOT_ID)
    with pytest.raises(ValueError, match="fields do not match"):
        validate_sonic_health_request(
            {**probe, "motion": True},
            expected_robot_id="g1-sim",
        )


def test_health_transport_timeout_gate_recovers_only_from_strict_success() -> None:
    gate = health_transport_gate()
    first_request = REQUEST_ID
    recovery_request = UUID("60000000-0000-4000-8000-000000000006")

    first = gate.observe_timeout(
        request_id=first_request,
        probe_started_at_s=10.000,
        probe_completed_at_s=10.020,
    )

    assert first.transition == "pending"
    assert first.confirmed_cause is None
    assert first.pending is not None
    assert first.pending.count == 1
    assert first.pending.confirmation_deadline_s == pytest.approx(10.070)
    assert first.total_timeout_count == 1
    assert first.recovered_timeout_count == 0
    assert first.confirmed_timeout_count == 0

    recovered = gate.observe_strict_success(
        request_id=recovery_request,
        probe_started_at_s=10.025,
        probe_completed_at_s=10.030,
    )

    assert recovered.transition == "recovered"
    assert recovered.confirmed_cause is None
    assert recovered.pending is None
    assert recovered.total_timeout_count == 1
    assert recovered.recovered_timeout_count == 1
    assert recovered.confirmed_timeout_count == 0


def test_health_transport_timeout_gate_second_timeout_is_terminal_once() -> None:
    gate = health_transport_gate()
    second_request = UUID("60000000-0000-4000-8000-000000000006")

    first = gate.observe_timeout(
        request_id=REQUEST_ID,
        probe_started_at_s=20.000,
        probe_completed_at_s=20.020,
    )
    second = gate.observe_timeout(
        request_id=second_request,
        probe_started_at_s=20.025,
        probe_completed_at_s=20.045,
    )
    repeated = gate.observe_timeout(
        request_id=UUID("70000000-0000-4000-8000-000000000007"),
        probe_started_at_s=20.050,
        probe_completed_at_s=20.070,
    )

    assert first.confirmation_budget_s == pytest.approx(0.070)
    assert second.transition == "confirmed"
    assert second.confirmed_cause is not None
    assert second.confirmed_cause.code == "SONIC_UNAVAILABLE"
    assert second.pending is not None
    assert second.pending.count == 2
    assert second.total_timeout_count == 2
    assert second.confirmed_timeout_count == 1
    assert repeated.transition == "latched"
    assert repeated.confirmed_cause is second.confirmed_cause
    assert repeated.total_timeout_count == 2
    assert repeated.confirmed_timeout_count == 1


def test_health_transport_timeout_gate_late_callback_cannot_recover() -> None:
    gate = health_transport_gate()
    gate.observe_timeout(
        request_id=REQUEST_ID,
        probe_started_at_s=30.000,
        probe_completed_at_s=30.020,
    )

    expired = gate.check_deadline(now_s=30.070)
    late_success = gate.observe_strict_success(
        request_id=UUID("60000000-0000-4000-8000-000000000006"),
        probe_started_at_s=30.075,
        probe_completed_at_s=30.080,
    )

    assert expired.transition == "confirmed"
    assert expired.confirmed_cause is not None
    assert "deadline expired" in expired.confirmed_cause.reason
    assert late_success.transition == "latched"
    assert late_success.confirmed_cause is expired.confirmed_cause
    assert late_success.pending is not None
    assert late_success.recovered_timeout_count == 0
    assert late_success.confirmed_timeout_count == 1


def test_overlapping_success_cannot_clear_health_transport_timeout() -> None:
    gate = health_transport_gate()
    gate.observe_timeout(
        request_id=REQUEST_ID,
        probe_started_at_s=40.000,
        probe_completed_at_s=40.020,
    )

    overlap = gate.observe_strict_success(
        request_id=UUID("60000000-0000-4000-8000-000000000006"),
        probe_started_at_s=40.005,
        probe_completed_at_s=40.025,
    )

    assert overlap.transition == "ignored_overlap"
    assert overlap.pending is not None
    assert overlap.pending.count == 1
    assert overlap.recovered_timeout_count == 0


def test_overlapping_timeout_does_not_count_as_confirmation_sample() -> None:
    gate = health_transport_gate()
    gate.observe_timeout(
        request_id=REQUEST_ID,
        probe_started_at_s=50.000,
        probe_completed_at_s=50.020,
    )

    overlap = gate.observe_timeout(
        request_id=UUID("60000000-0000-4000-8000-000000000006"),
        probe_started_at_s=50.005,
        probe_completed_at_s=50.025,
    )

    assert overlap.transition == "ignored_overlap"
    assert overlap.confirmed_cause is None
    assert overlap.pending is not None
    assert overlap.pending.count == 1
    assert overlap.total_timeout_count == 2


def test_replayed_request_timeout_does_not_confirm_transport_failure() -> None:
    gate = health_transport_gate()
    gate.observe_timeout(
        request_id=REQUEST_ID,
        probe_started_at_s=60.000,
        probe_completed_at_s=60.020,
    )

    replay = gate.observe_timeout(
        request_id=REQUEST_ID,
        probe_started_at_s=60.025,
        probe_completed_at_s=60.045,
    )

    assert replay.transition == "ignored_overlap"
    assert replay.confirmed_cause is None
    assert replay.pending is not None
    assert replay.pending.count == 1
    assert replay.total_timeout_count == 2


def test_health_transport_timeout_gate_rejects_unsafe_confirmation_budget() -> None:
    with pytest.raises(ValueError, match="cancellation-start budget"):
        SonicHealthTransportGate(
            probe_period_s=0.040,
            probe_timeout_s=0.020,
        )

    assert classify_sonic_health(
        parse_health(),
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    ) is None

    terminal = classify_sonic_health(
        parse_health(
            status="fault",
            terminal_fault=True,
            reason="plant tick rolled back",
            motion_ready=False,
        ),
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )
    assert terminal is not None
    assert (terminal.source, terminal.code) == (
        "sonic_driver",
        "SONIC_TERMINAL_FAULT",
    )

    unavailable = classify_sonic_health(
        parse_health(
            status="stopped",
            motion_ready=False,
        ),
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )
    assert unavailable is not None
    assert unavailable.code == "SONIC_UNAVAILABLE"

    stale = classify_sonic_health(
        parse_health(age=0.101),
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )
    assert stale is not None
    assert (stale.source, stale.code) == (
        "sonic_driver",
        "PLANT_LOWSTATE_STALE",
    )
    assert stale.reason.startswith("LowState reception age ")

    control_stale = classify_sonic_health(
        parse_health(control_age=0.101),
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )
    assert control_stale is not None
    assert (control_stale.source, control_stale.code) == (
        "sonic_driver",
        "SONIC_CONTROL_STALE",
    )

    with pytest.raises(ValueError, match="flags must be booleans"):
        SonicHealth.from_response(
            {**health_response(), "motion_ready": 1},
            expected_request_id=REQUEST_ID,
            expected_robot_id="g1-sim",
        )
    with pytest.raises(ValueError, match="fields do not match"):
        SonicHealth.from_response(
            {**health_response(), "legacy_guess": "SAFE_HOLD"},
            expected_request_id=REQUEST_ID,
            expected_robot_id="g1-sim",
        )
    with pytest.raises(ValueError, match="support mode is invalid"):
        SonicHealth.from_response(
            {**health_response(), "plant_support_mode": "looks_stopped"},
            expected_request_id=REQUEST_ID,
            expected_robot_id="g1-sim",
        )
    with pytest.raises(ValueError, match="authorize motion while support is latched"):
        SonicHealth.from_response(
            {
                **health_response(),
                "plant_support_mode": "safety_stop",
                "plant_support_safety_latched": True,
                "motion_authorized": True,
            },
            expected_request_id=REQUEST_ID,
            expected_robot_id="g1-sim",
        )


def test_health_exposes_only_first_dds_grant_as_reference_actuator_authority() -> None:
    primed = parse_health(
        navigation_task_active=True,
        motion_authorized=True,
        nav2_command_zero=False,
        reference_authority_state="PRIMED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
    )
    assert primed.matched_reference_actuator_authority is False

    granted = parse_health(
        navigation_task_active=True,
        motion_authorized=True,
        nav2_command_zero=False,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=100.25,
        reference_authority_granted_tick_ms=2500,
        reference_authority_dds_target_sequence=11,
    )
    assert granted.matched_reference_actuator_authority is True
    assert granted.reference_authority_state == "GRANTED"
    assert granted.reference_authority_dds_target_sequence == 11

    with pytest.raises(ValueError, match="flag and reference authority state"):
        parse_health(
            navigation_task_active=True,
            motion_authorized=True,
            nav2_command_zero=False,
            matched_reference_actuator_authority=True,
            reference_authority_state="PRIMED",
            reference_authority_generation=42,
            reference_authority_source_sequence=7,
            reference_authority_intent_generation=3,
            reference_authority_epoch=2,
            reference_authority_granted_at_monotonic_s=100.25,
            reference_authority_granted_tick_ms=2500,
            reference_authority_dds_target_sequence=11,
        )


def test_health_accepts_only_the_explicit_driver_owned_policy_braking_shape() -> None:
    braking = parse_health(
        plant_support_mode="locomotion",
        static_hold_active=False,
        navigation_task_active=False,
        motion_authorized=False,
        nav2_command_zero=True,
        controlled_policy_stop_active=True,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=100.25,
        reference_authority_granted_tick_ms=2500,
        reference_authority_dds_target_sequence=11,
    )

    assert braking.controlled_policy_stop_active is True
    assert braking.matched_reference_actuator_authority is True
    assert braking.confirmed_safe_hold is False

    balanced = parse_health(
        plant_support_mode="idle",
        static_hold_active=False,
        navigation_task_active=False,
        motion_authorized=True,
        nav2_command_zero=True,
        controlled_policy_stop_active=True,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=100.25,
        reference_authority_granted_tick_ms=2500,
        reference_authority_dds_target_sequence=11,
    )
    assert balanced.controlled_policy_stop_active is True
    assert balanced.motion_authorized is True
    assert balanced.navigation_task_active is False
    assert balanced.confirmed_safe_hold is False

    awaiting_plant_latch = parse_health(
        plant_support_mode="idle",
        plant_support_stop_pending=True,
        static_hold_active=False,
        navigation_task_active=False,
        motion_authorized=False,
        nav2_command_zero=True,
        controlled_policy_stop_active=True,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=100.25,
        reference_authority_granted_tick_ms=2500,
        reference_authority_dds_target_sequence=11,
    )
    assert awaiting_plant_latch.plant_support_stop_pending is True
    assert awaiting_plant_latch.controlled_policy_stop_active is True
    assert awaiting_plant_latch.confirmed_safe_hold is False

    with pytest.raises(ValueError, match="controlled SONIC policy stop"):
        parse_health(
            plant_support_mode="locomotion",
            plant_support_stop_pending=True,
            static_hold_active=False,
            navigation_task_active=False,
            motion_authorized=False,
            nav2_command_zero=True,
            controlled_policy_stop_active=True,
            matched_reference_actuator_authority=True,
            reference_authority_state="GRANTED",
            reference_authority_generation=42,
            reference_authority_source_sequence=7,
            reference_authority_intent_generation=3,
            reference_authority_epoch=2,
            reference_authority_granted_at_monotonic_s=100.25,
            reference_authority_granted_tick_ms=2500,
            reference_authority_dds_target_sequence=11,
        )

    with pytest.raises(ValueError, match="controlled SONIC policy stop"):
        parse_health(
            plant_support_mode="locomotion",
            static_hold_active=False,
            navigation_task_active=False,
            motion_authorized=True,
            nav2_command_zero=True,
            controlled_policy_stop_active=True,
            matched_reference_actuator_authority=True,
            reference_authority_state="GRANTED",
            reference_authority_generation=42,
            reference_authority_source_sequence=7,
            reference_authority_intent_generation=3,
            reference_authority_epoch=2,
            reference_authority_granted_at_monotonic_s=100.25,
            reference_authority_granted_tick_ms=2500,
            reference_authority_dds_target_sequence=11,
        )

    # A correlated controlled stop first withdraws the moving clip, then
    # requests/acquires a fresh measured IDLE reference.  Those planning-only
    # phases deliberately carry no actuator authority but remain an explicit
    # driver-owned braking transaction rather than a component failure.
    for phase in ("REVOKED", "REQUIRED", "MATCHED_INSTALLED", "PRIMED"):
        rebasing = parse_health(
            plant_support_mode="idle",
            static_hold_active=True,
            navigation_task_active=False,
            motion_authorized=False,
            nav2_command_zero=True,
            controlled_policy_stop_active=True,
            matched_reference_actuator_authority=False,
            reference_authority_state=phase,
            reference_authority_generation=42,
            reference_authority_source_sequence=7,
            reference_authority_intent_generation=(
                3 if phase in {"MATCHED_INSTALLED", "PRIMED"} else None
            ),
            reference_authority_epoch=2,
        )
        assert rebasing.controlled_policy_stop_active is True
        assert rebasing.matched_reference_actuator_authority is False
        assert rebasing.confirmed_safe_hold is False

    with pytest.raises(ValueError, match="controlled SONIC policy stop"):
        parse_health(
            plant_support_mode="idle",
            static_hold_active=False,
            navigation_task_active=False,
            motion_authorized=False,
            nav2_command_zero=True,
            controlled_policy_stop_active=True,
            matched_reference_actuator_authority=False,
            reference_authority_state="REQUIRED",
            reference_authority_generation=42,
            reference_authority_source_sequence=7,
            reference_authority_intent_generation=None,
            reference_authority_epoch=2,
        )

    with pytest.raises(ValueError, match="matched reference actuator authority"):
        parse_health(
            plant_support_mode="locomotion",
            static_hold_active=False,
            navigation_task_active=False,
            motion_authorized=False,
            nav2_command_zero=True,
            controlled_policy_stop_active=False,
            matched_reference_actuator_authority=True,
            reference_authority_state="GRANTED",
            reference_authority_generation=42,
            reference_authority_source_sequence=7,
            reference_authority_intent_generation=3,
            reference_authority_epoch=2,
            reference_authority_granted_at_monotonic_s=100.25,
            reference_authority_granted_tick_ms=2500,
            reference_authority_dds_target_sequence=11,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("static_hold_active", True),
        ("navigation_task_active", True),
        ("motion_authorized", True),
        ("nav2_command_zero", False),
        ("plant_support_stop_pending", True),
        ("plant_support_safety_latched", True),
    ],
)
def test_controlled_policy_braking_never_hides_an_unsafe_snapshot(
    field: str,
    value: bool,
) -> None:
    payload = {
        "plant_support_mode": "locomotion",
        "static_hold_active": False,
        "navigation_task_active": False,
        "motion_authorized": False,
        "nav2_command_zero": True,
        "controlled_policy_stop_active": True,
        "matched_reference_actuator_authority": True,
        "reference_authority_state": "GRANTED",
        "reference_authority_generation": 42,
        "reference_authority_source_sequence": 7,
        "reference_authority_intent_generation": 3,
        "reference_authority_epoch": 2,
        "reference_authority_granted_at_monotonic_s": 100.25,
        "reference_authority_granted_tick_ms": 2500,
        "reference_authority_dds_target_sequence": 11,
        field: value,
    }
    with pytest.raises(ValueError):
        parse_health(**payload)


def test_safe_hold_quarantines_age_only_faults_without_latching() -> None:
    gate = health_fault_gate()

    for count, age in enumerate((0.111029, 0.136, 0.451), start=1):
        decision = observe(gate, safe_hold_health(age=age))
        assert decision.confirmed_cause is None
        assert decision.pending is None
        assert decision.settled_safe_hold is True
        assert decision.quarantined_cause is not None
        assert decision.quarantined_cause.code == "PLANT_LOWSTATE_STALE"
        assert decision.quarantined_stale_count == count

    recovered = observe(gate, safe_hold_health(age=0.020))
    assert recovered.confirmed_cause is None
    assert recovered.pending is None
    assert recovered.quarantined_cause is None
    assert recovered.settled_safe_hold is False


@pytest.mark.parametrize(
    ("health", "gait_state"),
    [
        (
            safe_hold_health(age=0.111029, plant_support_stop_pending=True),
            GaitState.IDLE,
        ),
        (safe_hold_health(age=0.111029, static_hold_active=False), GaitState.IDLE),
        (safe_hold_health(age=0.111029, nav2_command_zero=False), GaitState.IDLE),
        (safe_hold_health(age=0.111029), GaitState.DISARMED),
        (safe_hold_health(age=0.111029), GaitState.STARTING),
        (safe_hold_health(age=0.111029), GaitState.WALKING),
        (safe_hold_health(age=0.111029), GaitState.TURNING),
    ],
)
def test_incomplete_or_non_idle_hold_never_quarantines_active_age_faults(
    health: SonicHealth,
    gait_state: GaitState,
) -> None:
    gate = health_fault_gate()

    first = observe(gate, health, gait_state=gait_state)
    second = observe(gate, health, gait_state=gait_state)

    assert first.quarantined_cause is None
    assert first.pending is not None
    assert first.pending.count == 1
    assert second.confirmed_cause is not None
    assert second.confirmed_cause.code == "PLANT_LOWSTATE_STALE"


def test_motion_authority_after_quarantine_restarts_strict_confirmation() -> None:
    gate = health_fault_gate()
    quarantined = observe(gate, safe_hold_health(age=0.111029))
    assert quarantined.quarantined_cause is not None

    first_active = observe(gate, parse_health(age=0.136))
    second_active = observe(gate, parse_health(age=0.151))

    assert first_active.pending is not None
    assert first_active.pending.count == 1
    assert first_active.confirmed_cause is None
    assert second_active.confirmed_cause is not None
    assert second_active.confirmed_cause.code == "PLANT_LOWSTATE_STALE"


def test_control_acquisition_requires_fresh_confirmed_safe_hold() -> None:
    require_fresh_motion_health(
        safe_hold_health(),
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )

    with pytest.raises(ValueError, match="PLANT_LOWSTATE_STALE"):
        require_fresh_motion_health(
            safe_hold_health(age=0.101),
            plant_lowstate_stale_s=0.10,
            sonic_control_step_stale_s=0.10,
        )
    balanced = parse_health(
        plant_support_mode="idle",
        static_hold_active=False,
        navigation_task_active=False,
        motion_authorized=True,
        nav2_command_zero=True,
        controlled_policy_stop_active=True,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=100.25,
        reference_authority_granted_tick_ms=2500,
        reference_authority_dds_target_sequence=11,
    )
    assert balanced.confirmed_balanced_policy_idle is True
    require_fresh_motion_health(
        balanced,
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
    )

    with pytest.raises(ValueError, match="resident balanced policy IDLE"):
        require_fresh_motion_health(
            parse_health(),
            plant_lowstate_stale_s=0.10,
            sonic_control_step_stale_s=0.10,
        )

    armed_static = parse_health()
    assert armed_static.confirmed_armed_static_idle is True
    require_fresh_motion_health(
        armed_static,
        plant_lowstate_stale_s=0.10,
        sonic_control_step_stale_s=0.10,
        allow_armed_static_idle=True,
    )


def test_terminal_and_boot_change_remain_immediate_in_safe_hold() -> None:
    terminal_gate = health_fault_gate()
    terminal = observe(
        terminal_gate,
        safe_hold_health(
            status="fault",
            terminal_fault=True,
            reason="control loop failed",
            motion_ready=False,
        ),
    )
    assert terminal.confirmed_cause is not None
    assert terminal.confirmed_cause.code == "SONIC_TERMINAL_FAULT"
    assert terminal.quarantined_cause is None

    boot_gate = health_fault_gate()
    assert observe(boot_gate, safe_hold_health()).confirmed_cause is None
    changed = observe(
        boot_gate,
        safe_hold_health(driver_boot_id=NEW_DRIVER_BOOT_ID),
    )
    assert changed.confirmed_cause is not None
    assert changed.confirmed_cause.code == "SONIC_BOOT_CHANGED"
    assert changed.quarantined_cause is None


def test_stale_health_gate_requires_two_consecutive_temporal_faults() -> None:
    gate = health_fault_gate()

    first = observe(gate, parse_health(age=0.111029))
    assert first.confirmed_cause is None
    assert first.required_count == 2
    assert first.pending is not None
    assert first.pending.driver_boot_id == DRIVER_BOOT_ID
    assert first.pending.source == "sonic_driver"
    assert first.pending.code == "PLANT_LOWSTATE_STALE"
    assert first.pending.first_source == "sonic_driver"
    assert first.pending.first_code == "PLANT_LOWSTATE_STALE"
    assert first.pending.latest_source == "sonic_driver"
    assert first.pending.latest_code == "PLANT_LOWSTATE_STALE"
    assert first.pending.count == 1
    assert first.pending.first_age_s == pytest.approx(0.111029)
    assert first.pending.latest_age_s == pytest.approx(0.111029)
    assert first.filtered_transient_count == 0

    second = observe(gate, parse_health(age=0.136))
    assert second.confirmed_cause is not None
    assert second.confirmed_cause.code == "PLANT_LOWSTATE_STALE"
    assert second.pending is not None
    assert second.pending.count == 2
    assert second.pending.first_code == "PLANT_LOWSTATE_STALE"
    assert second.pending.latest_code == "PLANT_LOWSTATE_STALE"
    assert second.pending.first_age_s == pytest.approx(0.111029)
    assert second.pending.latest_age_s == pytest.approx(0.136)


def test_stale_health_gate_healthy_probe_resets_pending() -> None:
    gate = health_fault_gate()

    assert observe(gate, parse_health(age=0.111029)).pending is not None
    healthy = observe(gate, parse_health(age=0.020))
    assert healthy.confirmed_cause is None
    assert healthy.pending is None
    assert healthy.filtered_transient_count == 1

    repeated = observe(gate, parse_health(age=0.111029))
    assert repeated.confirmed_cause is None
    assert repeated.pending is not None
    assert repeated.pending.count == 1


def test_stale_health_gate_alternating_codes_share_one_unhealthy_streak() -> None:
    gate = health_fault_gate()

    lowstate = observe(gate, parse_health(age=0.111029))
    assert lowstate.pending is not None
    assert lowstate.pending.first_code == "PLANT_LOWSTATE_STALE"
    assert lowstate.pending.latest_code == "PLANT_LOWSTATE_STALE"
    assert lowstate.pending.count == 1

    control = observe(gate, parse_health(age=0.020, control_age=0.111029))
    assert control.confirmed_cause is not None
    assert control.confirmed_cause.code == "SONIC_CONTROL_STALE"
    assert control.pending is not None
    assert control.pending.first_code == "PLANT_LOWSTATE_STALE"
    assert control.pending.latest_code == "SONIC_CONTROL_STALE"
    assert control.pending.count == 2
    assert control.pending.first_age_s == pytest.approx(0.111029)
    assert control.pending.latest_age_s == pytest.approx(0.111029)
    assert control.filtered_transient_count == 0

    lowstate_again = observe(gate, parse_health(age=0.136, control_age=0.020))
    assert lowstate_again.confirmed_cause is not None
    assert lowstate_again.confirmed_cause.code == "PLANT_LOWSTATE_STALE"
    assert lowstate_again.pending is not None
    assert lowstate_again.pending.first_code == "PLANT_LOWSTATE_STALE"
    assert lowstate_again.pending.latest_code == "PLANT_LOWSTATE_STALE"
    assert lowstate_again.pending.count == 3


def test_starting_static_pregrant_control_stale_remains_strict() -> None:
    gate = health_fault_gate()

    pregrant = parse_health(
        control_age=0.127784,
        static_hold_active=True,
        navigation_task_active=True,
        motion_authorized=True,
        nav2_command_zero=False,
        matched_reference_actuator_authority=False,
        reference_authority_state="REVOKED",
    )
    first_late = observe(gate, pregrant, gait_state=GaitState.STARTING)
    second_late = observe(gate, pregrant, gait_state=GaitState.STARTING)

    assert first_late.pending is not None
    assert first_late.confirmed_cause is None
    assert second_late.confirmed_cause is not None
    assert second_late.confirmed_cause.code == "SONIC_CONTROL_STALE"
    assert "0.100000s" in second_late.confirmed_cause.reason


def test_starting_lowstate_and_granted_control_stale_are_strict() -> None:
    pregrant_lowstate_stale = parse_health(
        age=0.101,
        control_age=0.127784,
        static_hold_active=True,
        navigation_task_active=True,
        motion_authorized=True,
        nav2_command_zero=False,
        matched_reference_actuator_authority=False,
        reference_authority_state="REQUIRED",
    )
    lowstate_gate = health_fault_gate()
    observe(
        lowstate_gate,
        pregrant_lowstate_stale,
        gait_state=GaitState.STARTING,
    )
    lowstate = observe(
        lowstate_gate,
        pregrant_lowstate_stale,
        gait_state=GaitState.STARTING,
    )
    assert lowstate.confirmed_cause is not None
    assert lowstate.confirmed_cause.code == "PLANT_LOWSTATE_STALE"

    granted = parse_health(
        control_age=0.127784,
        plant_support_mode="locomotion",
        static_hold_active=False,
        navigation_task_active=True,
        motion_authorized=True,
        nav2_command_zero=False,
        matched_reference_actuator_authority=True,
        reference_authority_state="GRANTED",
        reference_authority_generation=42,
        reference_authority_source_sequence=7,
        reference_authority_intent_generation=3,
        reference_authority_epoch=2,
        reference_authority_granted_at_monotonic_s=100.25,
        reference_authority_granted_tick_ms=2500,
        reference_authority_dds_target_sequence=11,
    )
    granted_gate = health_fault_gate()
    observe(granted_gate, granted, gait_state=GaitState.TURNING)
    strict = observe(granted_gate, granted, gait_state=GaitState.TURNING)
    assert strict.confirmed_cause is not None
    assert strict.confirmed_cause.code == "SONIC_CONTROL_STALE"


def test_safe_hold_quarantine_clears_an_active_temporal_streak() -> None:
    gate = health_fault_gate()

    active = observe(
        gate,
        parse_health(age=0.111029),
        gait_state=GaitState.WALKING,
    )
    assert active.pending is not None
    assert active.pending.count == 1

    quarantined = observe(
        gate,
        safe_hold_health(age=0.020, control_age=0.111029),
    )
    assert quarantined.confirmed_cause is None
    assert quarantined.pending is None
    assert quarantined.quarantined_cause is not None
    assert quarantined.quarantined_cause.code == "SONIC_CONTROL_STALE"
    assert quarantined.settled_safe_hold is True
    assert quarantined.filtered_transient_count == 1


def test_stale_health_gate_driver_boot_change_is_immediate() -> None:
    gate = health_fault_gate()

    assert observe(gate, parse_health(age=0.111029)).pending is not None
    changed = observe(
        gate,
        parse_health(age=0.020, driver_boot_id=NEW_DRIVER_BOOT_ID),
    )
    assert changed.pending is None
    assert changed.confirmed_cause is not None
    assert changed.confirmed_cause.code == "SONIC_BOOT_CHANGED"
    assert str(DRIVER_BOOT_ID) in changed.confirmed_cause.reason
    assert str(NEW_DRIVER_BOOT_ID) in changed.confirmed_cause.reason
    assert changed.driver_boot_id == NEW_DRIVER_BOOT_ID
    assert changed.filtered_transient_count == 1


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        (
            {
                "status": "fault",
                "terminal_fault": True,
                "reason": "plant tick rolled back",
                "motion_ready": False,
            },
            "SONIC_TERMINAL_FAULT",
        ),
        (
            {"status": "stopped", "motion_ready": False},
            "SONIC_UNAVAILABLE",
        ),
    ],
)
def test_stale_health_gate_non_temporal_faults_are_immediate(
    overrides: dict[str, object], expected_code: str
) -> None:
    decision = observe(health_fault_gate(), parse_health(**overrides))

    assert decision.pending is None
    assert decision.confirmed_cause is not None
    assert decision.confirmed_cause.code == expected_code


def test_confirmed_stale_health_trips_fault_latch_exactly_once() -> None:
    gate = health_fault_gate()
    latch = MotionFaultLatch(
        robot_id="g1-sim",
        localization_epoch_id="g1-sim-map-epoch-1",
        producer_boot_id=BRIDGE_BOOT_ID,
        event_id_factory=lambda: EVENT_ID,
    )
    published: list[MotionSafetyFault] = []
    stops = 0

    def stop() -> None:
        nonlocal stops
        stops += 1

    for age in (0.111029, 0.136, 0.151):
        decision = observe(gate, parse_health(age=age))
        cause = decision.confirmed_cause
        if cause is not None:
            latch.trip(
                control_generation=42,
                source=cause.source,
                code=cause.code,
                reason=cause.reason,
                publish=published.append,
                local_stop=stop,
                downstream_boot_id=decision.driver_boot_id,
            )

    assert len(published) == 1
    assert stops == 1
    assert published[0].code == "PLANT_LOWSTATE_STALE"
    assert latch.complete is True


def test_motion_fault_qos_is_reliable_transient_local_keep_last_one() -> None:
    qos_module = pytest.importorskip("rclpy.qos")
    profile = motion_fault_qos()
    assert profile.history == qos_module.HistoryPolicy.KEEP_LAST
    assert profile.depth == 1
    assert profile.reliability == qos_module.ReliabilityPolicy.RELIABLE
    assert profile.durability == qos_module.DurabilityPolicy.TRANSIENT_LOCAL
