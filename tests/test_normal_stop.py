from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from hide_mujoco_sonic.normal_stop import (
    NORMAL_STOP_PHASES,
    NormalStopIdentity,
    NormalStopIdentityError,
    NormalStopSnapshot,
    NormalStopState,
    NormalStopTransitionError,
)


STOP_ID_1 = UUID("10000000-0000-4000-8000-000000000001")
STOP_ID_2 = UUID("20000000-0000-4000-8000-000000000002")
STOP_ID_3 = UUID("30000000-0000-4000-8000-000000000003")


def factory(*values: UUID):
    remaining = iter(values)
    return lambda: next(remaining)


def complete(state: NormalStopState):
    braking = state.begin(control_generation=10, gait_source_sequence=20)
    identity = braking.identity
    assert identity is not None
    state.acknowledge_stationary(identity)
    state.confirm_plant(identity)
    return state.confirm_driver(identity)


def test_happy_path_is_strictly_ordered_and_only_driver_confirms_safe_hold() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1))
    initial = state.snapshot
    assert initial.phase == "NONE"
    assert initial.identity is None
    assert initial.safe_hold_confirmed is False
    assert NORMAL_STOP_PHASES == {
        "NONE",
        "BRAKING",
        "STATIONARY_ACK",
        "PLANT_CONFIRMED",
        "DRIVER_CONFIRMED",
        "FAILED",
    }

    braking = state.begin(control_generation=10, gait_source_sequence=20)
    identity = braking.identity
    assert identity == NormalStopIdentity(STOP_ID_1, 10, 20, 1)
    assert braking.safe_hold_confirmed is False

    stationary = state.acknowledge_stationary(identity)
    assert stationary.phase == "STATIONARY_ACK"
    assert stationary.safe_hold_confirmed is False

    plant = state.confirm_plant(identity)
    assert plant.phase == "PLANT_CONFIRMED"
    assert plant.safe_hold_confirmed is False

    driver = state.confirm_driver(identity)
    assert driver.phase == "DRIVER_CONFIRMED"
    assert driver.safe_hold_confirmed is True
    with pytest.raises(FrozenInstanceError):
        driver.phase = "FAILED"  # type: ignore[misc]


def test_every_ack_requires_the_exact_stop_identity() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1))
    braking = state.begin(control_generation=10, gait_source_sequence=20)
    identity = braking.identity
    assert identity is not None

    mismatches = (
        NormalStopIdentity(STOP_ID_2, 10, 20, 1),
        NormalStopIdentity(STOP_ID_1, 11, 20, 1),
        NormalStopIdentity(STOP_ID_1, 10, 21, 1),
        NormalStopIdentity(STOP_ID_1, 10, 20, 2),
    )
    for mismatch in mismatches:
        with pytest.raises(NormalStopIdentityError, match="does not match"):
            state.acknowledge_stationary(mismatch)
        assert state.snapshot is braking

    with pytest.raises(NormalStopIdentityError, match="requires NormalStopIdentity"):
        state.acknowledge_stationary(object())  # type: ignore[arg-type]


def test_out_of_order_acknowledgements_are_rejected_without_state_change() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1))
    braking = state.begin(control_generation=10, gait_source_sequence=20)
    identity = braking.identity
    assert identity is not None

    with pytest.raises(NormalStopTransitionError, match="out of order"):
        state.confirm_plant(identity)
    with pytest.raises(NormalStopTransitionError, match="out of order"):
        state.confirm_driver(identity)
    assert state.snapshot is braking

    stationary = state.acknowledge_stationary(identity)
    with pytest.raises(NormalStopTransitionError, match="out of order"):
        state.confirm_driver(identity)
    assert state.snapshot is stationary


def test_exact_duplicate_events_and_begin_are_idempotent_without_regression() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1))
    braking = state.begin(control_generation=10, gait_source_sequence=20)
    assert state.begin(control_generation=10, gait_source_sequence=20) is braking
    identity = braking.identity
    assert identity is not None

    stationary = state.acknowledge_stationary(identity)
    assert state.acknowledge_stationary(identity) is stationary
    plant = state.confirm_plant(identity)
    assert state.acknowledge_stationary(identity) is plant
    assert state.confirm_plant(identity) is plant
    driver = state.confirm_driver(identity)
    assert state.acknowledge_stationary(identity) is driver
    assert state.confirm_plant(identity) is driver
    assert state.confirm_driver(identity) is driver
    assert state.begin(control_generation=10, gait_source_sequence=20) is driver


def test_old_ack_and_plant_ack_cannot_confirm_a_new_stop_epoch() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1, STOP_ID_2))
    first = complete(state)
    old_identity = first.identity
    assert old_identity is not None

    second = state.begin(control_generation=11, gait_source_sequence=21)
    new_identity = second.identity
    assert new_identity == NormalStopIdentity(STOP_ID_2, 11, 21, 2)
    assert second.stop_epoch > first.stop_epoch
    assert second.stop_id != first.stop_id

    with pytest.raises(NormalStopIdentityError, match="does not match"):
        state.acknowledge_stationary(old_identity)
    with pytest.raises(NormalStopIdentityError, match="does not match"):
        state.confirm_plant(old_identity)
    assert state.snapshot is second

    state.acknowledge_stationary(new_identity)
    assert state.confirm_plant(new_identity).phase == "PLANT_CONFIRMED"


def test_new_transaction_requires_new_generation_sequence_and_unique_uuid() -> None:
    state = NormalStopState(
        stop_id_factory=factory(STOP_ID_1, STOP_ID_2, STOP_ID_1)
    )
    complete(state)

    with pytest.raises(NormalStopTransitionError, match="newer control generation"):
        state.begin(control_generation=10, gait_source_sequence=21)
    with pytest.raises(NormalStopTransitionError, match="newer gait source sequence"):
        state.begin(control_generation=11, gait_source_sequence=20)

    second = state.begin(control_generation=11, gait_source_sequence=21)
    second_identity = second.identity
    assert second_identity is not None
    state.acknowledge_stationary(second_identity)
    state.confirm_plant(second_identity)
    state.confirm_driver(second_identity)

    with pytest.raises(ValueError, match="already-issued UUID"):
        state.begin(control_generation=12, gait_source_sequence=22)
    assert state.snapshot.phase == "DRIVER_CONFIRMED"
    assert state.snapshot.stop_epoch == 2


def test_failure_is_terminal_and_first_failure_snapshot_is_idempotent() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1, STOP_ID_2))
    braking = state.begin(control_generation=10, gait_source_sequence=20)
    identity = braking.identity
    assert identity is not None

    failed = state.fail(identity, reason="stationary acknowledgement timed out")
    assert failed.phase == "FAILED"
    assert failed.safe_hold_confirmed is False
    assert state.fail(identity, reason="later callback") is failed

    transitions = (
        state.acknowledge_stationary,
        state.confirm_plant,
        state.confirm_driver,
    )
    for transition in transitions:
        with pytest.raises(NormalStopTransitionError, match="terminal"):
            transition(identity)
    with pytest.raises(NormalStopTransitionError, match="terminal"):
        state.begin(control_generation=11, gait_source_sequence=21)


def test_failure_cannot_replace_a_confirmed_safe_hold() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1))
    confirmed = complete(state)
    identity = confirmed.identity
    assert identity is not None
    with pytest.raises(NormalStopTransitionError, match="cannot transition"):
        state.fail(identity, reason="late timeout")
    assert state.snapshot is confirmed
    assert state.snapshot.safe_hold_confirmed is True


def test_concurrent_duplicate_acknowledgements_commit_once_atomically() -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1))
    braking = state.begin(control_generation=10, gait_source_sequence=20)
    identity = braking.identity
    assert identity is not None

    with ThreadPoolExecutor(max_workers=16) as executor:
        stationary_results = list(
            executor.map(lambda _: state.acknowledge_stationary(identity), range(64))
        )
    assert all(value is state.snapshot for value in stationary_results)
    assert state.snapshot.phase == "STATIONARY_ACK"

    with ThreadPoolExecutor(max_workers=16) as executor:
        plant_results = list(
            executor.map(lambda _: state.confirm_plant(identity), range(64))
        )
    assert all(value is state.snapshot for value in plant_results)
    assert state.snapshot.phase == "PLANT_CONFIRMED"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"control_generation": True, "gait_source_sequence": 1}, "control_generation"),
        ({"control_generation": 1, "gait_source_sequence": 0}, "gait_source_sequence"),
    ),
)
def test_begin_rejects_invalid_correlation_values(kwargs, message: str) -> None:
    state = NormalStopState(stop_id_factory=factory(STOP_ID_1))
    with pytest.raises(ValueError, match=message):
        state.begin(**kwargs)


def test_snapshot_rejects_safe_hold_shaped_invalid_phases() -> None:
    with pytest.raises(ValueError, match="phase"):
        NormalStopSnapshot(phase="SAFE_HOLD")
    with pytest.raises(ValueError, match="NONE"):
        NormalStopSnapshot(stop_epoch=1)
