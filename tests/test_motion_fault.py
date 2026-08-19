from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import UUID

import pytest

from hide_mujoco_sonic.motion_fault import (
    MOTION_FAULT_SCHEMA_VERSION,
    MotionFaultLatch,
    MotionSafetyFault,
    SonicHealthFaultGate,
    SonicHealth,
    classify_sonic_health,
    motion_fault_qos,
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
) -> dict[str, object]:
    return {
        "request_id": str(REQUEST_ID),
        "ok": True,
        "schema_version": "h-ide.sonic-health/v1",
        "robot_id": "g1-sim",
        "driver_boot_id": str(driver_boot_id),
        "status": status,
        "terminal_fault": terminal_fault,
        "reason": reason,
        "motion_ready": motion_ready,
        "last_lowstate_age_s": age,
        "last_control_step_age_s": control_age,
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


def test_stale_health_gate_requires_same_boot_and_code_twice() -> None:
    gate = health_fault_gate()

    first = gate.observe(parse_health(age=0.111029))
    assert first.confirmed_cause is None
    assert first.required_count == 2
    assert first.pending is not None
    assert first.pending.driver_boot_id == DRIVER_BOOT_ID
    assert first.pending.source == "sonic_driver"
    assert first.pending.code == "PLANT_LOWSTATE_STALE"
    assert first.pending.count == 1
    assert first.pending.first_age_s == pytest.approx(0.111029)
    assert first.pending.latest_age_s == pytest.approx(0.111029)
    assert first.filtered_transient_count == 0

    second = gate.observe(parse_health(age=0.136))
    assert second.confirmed_cause is not None
    assert second.confirmed_cause.code == "PLANT_LOWSTATE_STALE"
    assert second.pending is not None
    assert second.pending.count == 2
    assert second.pending.first_age_s == pytest.approx(0.111029)
    assert second.pending.latest_age_s == pytest.approx(0.136)


def test_stale_health_gate_healthy_probe_resets_pending() -> None:
    gate = health_fault_gate()

    assert gate.observe(parse_health(age=0.111029)).pending is not None
    healthy = gate.observe(parse_health(age=0.020))
    assert healthy.confirmed_cause is None
    assert healthy.pending is None
    assert healthy.filtered_transient_count == 1

    repeated = gate.observe(parse_health(age=0.111029))
    assert repeated.confirmed_cause is None
    assert repeated.pending is not None
    assert repeated.pending.count == 1


def test_stale_health_gate_different_code_starts_a_new_sequence() -> None:
    gate = health_fault_gate()

    lowstate = gate.observe(parse_health(age=0.111029))
    assert lowstate.pending is not None
    assert lowstate.pending.code == "PLANT_LOWSTATE_STALE"

    control = gate.observe(parse_health(age=0.020, control_age=0.111029))
    assert control.confirmed_cause is None
    assert control.pending is not None
    assert control.pending.code == "SONIC_CONTROL_STALE"
    assert control.pending.count == 1
    assert control.filtered_transient_count == 1


def test_stale_health_gate_driver_boot_change_is_immediate() -> None:
    gate = health_fault_gate()

    assert gate.observe(parse_health(age=0.111029)).pending is not None
    changed = gate.observe(parse_health(age=0.020, driver_boot_id=NEW_DRIVER_BOOT_ID))
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
    decision = health_fault_gate().observe(parse_health(**overrides))

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
        decision = gate.observe(parse_health(age=age))
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
