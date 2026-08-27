from __future__ import annotations

import pytest

from hide_mujoco_sonic.reference_authority import ReferenceAuthorityState


def test_reference_authority_requires_install_prime_and_successful_dds_grant() -> None:
    state = ReferenceAuthorityState()
    assert state.snapshot.phase == "REVOKED"
    assert state.snapshot.actuator_authority is False

    state.require(generation=42, source_sequence=7)
    state.mark_installed(
        generation=42,
        source_sequence=7,
        planner_intent_generation=3,
        reference_epoch=1,
    )
    state.prime(generation=42, source_sequence=7, reference_epoch=1)
    granted = state.grant(
        generation=42,
        source_sequence=7,
        reference_epoch=1,
        granted_at_monotonic_s=100.25,
        granted_at_tick_ms=2500,
        dds_target_sequence=11,
    )

    assert granted.phase == "GRANTED"
    assert granted.actuator_authority is True
    assert granted.planner_intent_generation == 3
    assert granted.granted_at_monotonic_s == pytest.approx(100.25)
    assert granted.granted_at_tick_ms == 2500
    assert granted.dds_target_sequence == 11


def test_reference_authority_revoke_clears_grant_but_preserves_diagnostic_identity() -> None:
    state = ReferenceAuthorityState()
    state.require(generation=5, source_sequence=9)
    state.mark_installed(
        generation=5,
        source_sequence=9,
        planner_intent_generation=2,
        reference_epoch=4,
    )
    state.prime(generation=5, source_sequence=9, reference_epoch=4)
    state.grant(
        generation=5,
        source_sequence=9,
        reference_epoch=4,
        granted_at_monotonic_s=2.0,
        granted_at_tick_ms=200,
        dds_target_sequence=8,
    )

    revoked = state.revoke("Nav2 cancelled")
    assert revoked.phase == "REVOKED"
    assert revoked.actuator_authority is False
    assert revoked.generation == 5
    assert revoked.source_sequence == 9
    assert revoked.planner_intent_generation == 2
    assert revoked.reference_epoch == 4
    assert revoked.granted_at_monotonic_s is None
    assert revoked.dds_target_sequence is None


def test_periodic_refresh_primes_new_epoch_without_exposing_old_grant() -> None:
    state = ReferenceAuthorityState()
    state.require(generation=5, source_sequence=9)
    state.mark_installed(
        generation=5,
        source_sequence=9,
        planner_intent_generation=2,
        reference_epoch=4,
    )
    state.prime(generation=5, source_sequence=9, reference_epoch=4)
    state.grant(
        generation=5,
        source_sequence=9,
        reference_epoch=4,
        granted_at_monotonic_s=2.0,
        granted_at_tick_ms=200,
        dds_target_sequence=8,
    )

    with pytest.raises(ValueError, match="intent generation"):
        state.refresh(
            generation=5,
            source_sequence=9,
            planner_intent_generation=3,
            reference_epoch=5,
        )
    refreshed = state.refresh(
        generation=5,
        source_sequence=9,
        planner_intent_generation=2,
        reference_epoch=5,
    )
    assert refreshed.phase == "PRIMED"
    assert refreshed.reference_epoch == 5
    assert refreshed.actuator_authority is False
    assert refreshed.granted_at_monotonic_s is None
    assert refreshed.dds_target_sequence is None


def test_newer_same_intent_epoch_may_supersede_only_an_ungranted_candidate() -> None:
    state = ReferenceAuthorityState()
    state.require(generation=5, source_sequence=9)
    state.mark_installed(
        generation=5,
        source_sequence=9,
        planner_intent_generation=2,
        reference_epoch=4,
    )
    state.prime(generation=5, source_sequence=9, reference_epoch=4)

    with pytest.raises(ValueError, match="intent generation"):
        state.supersede_primed(
            generation=5,
            source_sequence=9,
            planner_intent_generation=3,
            reference_epoch=5,
        )
    superseded = state.supersede_primed(
        generation=5,
        source_sequence=9,
        planner_intent_generation=2,
        reference_epoch=5,
    )
    assert superseded.phase == "PRIMED"
    assert superseded.actuator_authority is False
    assert superseded.reference_epoch == 5
    assert superseded.granted_at_monotonic_s is None
    assert superseded.dds_target_sequence is None

    with pytest.raises(ValueError, match="did not advance"):
        state.supersede_primed(
            generation=5,
            source_sequence=9,
            planner_intent_generation=2,
            reference_epoch=5,
        )


def test_many_same_intent_primed_refreshes_never_grant_before_dds() -> None:
    state = ReferenceAuthorityState()
    state.require(generation=7, source_sequence=11)
    state.mark_installed(
        generation=7,
        source_sequence=11,
        planner_intent_generation=3,
        reference_epoch=2,
    )
    state.prime(generation=7, source_sequence=11, reference_epoch=2)

    for reference_epoch in range(3, 203):
        superseded = state.supersede_primed(
            generation=7,
            source_sequence=11,
            planner_intent_generation=3,
            reference_epoch=reference_epoch,
        )
        assert superseded.phase == "PRIMED"
        assert superseded.actuator_authority is False
        assert superseded.dds_target_sequence is None

    granted = state.grant(
        generation=7,
        source_sequence=11,
        reference_epoch=202,
        granted_at_monotonic_s=4.0,
        granted_at_tick_ms=4_000,
        dds_target_sequence=19,
    )
    assert granted.phase == "GRANTED"
    assert granted.actuator_authority is True
    assert granted.reference_epoch == 202


def test_controlled_stop_transition_rebinds_new_intent_at_the_installed_epoch() -> None:
    state = ReferenceAuthorityState()
    state.require(generation=5, source_sequence=9)
    state.mark_installed(
        generation=5,
        source_sequence=9,
        planner_intent_generation=2,
        reference_epoch=4,
    )
    state.prime(generation=5, source_sequence=9, reference_epoch=4)
    state.grant(
        generation=5,
        source_sequence=9,
        reference_epoch=4,
        granted_at_monotonic_s=2.0,
        granted_at_tick_ms=200,
        dds_target_sequence=8,
    )

    transitioned = state.transition(
        generation=5,
        source_sequence=12,
        planner_intent_generation=3,
        reference_epoch=5,
    )
    assert transitioned.phase == "PRIMED"
    assert transitioned.actuator_authority is False
    assert transitioned.source_sequence == 12
    assert transitioned.planner_intent_generation == 3
    assert transitioned.reference_epoch == 5
    assert transitioned.granted_at_monotonic_s is None
    assert transitioned.dds_target_sequence is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"generation": 6},
        {"source_sequence": 9},
        {"planner_intent_generation": 2},
        {"reference_epoch": 4},
    ],
)
def test_controlled_stop_transition_rejects_stale_or_mismatched_identity(
    overrides: dict[str, int],
) -> None:
    state = ReferenceAuthorityState()
    state.require(generation=5, source_sequence=9)
    state.mark_installed(
        generation=5,
        source_sequence=9,
        planner_intent_generation=2,
        reference_epoch=4,
    )
    state.prime(generation=5, source_sequence=9, reference_epoch=4)
    state.grant(
        generation=5,
        source_sequence=9,
        reference_epoch=4,
        granted_at_monotonic_s=2.0,
        granted_at_tick_ms=200,
        dds_target_sequence=8,
    )
    values = {
        "generation": 5,
        "source_sequence": 12,
        "planner_intent_generation": 3,
        "reference_epoch": 5,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        state.transition(**values)
    assert state.snapshot.phase == "GRANTED"


def test_duplicate_require_cannot_withdraw_primed_or_granted_authority() -> None:
    state = ReferenceAuthorityState()
    state.require(generation=7, source_sequence=12)
    state.mark_installed(
        generation=7,
        source_sequence=12,
        planner_intent_generation=5,
        reference_epoch=2,
    )
    primed = state.prime(generation=7, source_sequence=12, reference_epoch=2)
    assert state.require(generation=7, source_sequence=12) is primed
    granted = state.grant(
        generation=7,
        source_sequence=12,
        reference_epoch=2,
        granted_at_monotonic_s=4.0,
        granted_at_tick_ms=400,
        dds_target_sequence=9,
    )
    assert state.require(generation=7, source_sequence=12) is granted
    assert state.snapshot.actuator_authority is True


def test_reference_authority_rejects_stale_identity_epoch_and_early_grant() -> None:
    state = ReferenceAuthorityState()
    state.require(generation=3, source_sequence=10)
    with pytest.raises(ValueError, match="correlation"):
        state.mark_installed(
            generation=3,
            source_sequence=9,
            planner_intent_generation=1,
            reference_epoch=1,
        )
    with pytest.raises(ValueError, match="correlation"):
        state.grant(
            generation=3,
            source_sequence=10,
            reference_epoch=1,
            granted_at_monotonic_s=1.0,
            granted_at_tick_ms=20,
            dds_target_sequence=1,
        )

    state.mark_installed(
        generation=3,
        source_sequence=10,
        planner_intent_generation=1,
        reference_epoch=1,
    )
    with pytest.raises(ValueError, match="epoch"):
        state.prime(generation=3, source_sequence=10, reference_epoch=2)

    state.revoke("new command")
    with pytest.raises(ValueError, match="backwards"):
        state.require(generation=3, source_sequence=9)
