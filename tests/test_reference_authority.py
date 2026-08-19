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
