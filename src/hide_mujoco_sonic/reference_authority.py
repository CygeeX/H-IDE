"""Transactional ownership of one matched SONIC planner reference.

Planner evidence proves that a returned clip matches a Nav2 intent.  Actuator
authority is stronger: the matching clip must be installed, primed from the
executed joint boundary, and have one complete target successfully cross DDS.
This module keeps that distinction explicit and exposes immutable snapshots for
the strict health contract.  Callers serialize mutations with their actuator
boundary lock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math


REFERENCE_AUTHORITY_PHASES = frozenset(
    {"REVOKED", "REQUIRED", "MATCHED_INSTALLED", "PRIMED", "GRANTED"}
)


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class ReferenceAuthoritySnapshot:
    phase: str = "REVOKED"
    generation: int | None = None
    source_sequence: int | None = None
    planner_intent_generation: int | None = None
    reference_epoch: int = 0
    actuator_authority: bool = False
    granted_at_monotonic_s: float | None = None
    granted_at_tick_ms: int | None = None
    dds_target_sequence: int | None = None
    reason: str = "no matched reference has actuator authority"

    def __post_init__(self) -> None:
        if self.phase not in REFERENCE_AUTHORITY_PHASES:
            raise ValueError("reference authority phase is invalid")
        if self.generation is not None:
            _integer(self.generation, name="generation", minimum=0)
        if self.source_sequence is not None:
            _integer(self.source_sequence, name="source_sequence", minimum=1)
        if self.planner_intent_generation is not None:
            _integer(
                self.planner_intent_generation,
                name="planner_intent_generation",
                minimum=1,
            )
        _integer(self.reference_epoch, name="reference_epoch", minimum=0)
        grant_values = (
            self.granted_at_monotonic_s,
            self.granted_at_tick_ms,
            self.dds_target_sequence,
        )
        if self.actuator_authority != (self.phase == "GRANTED"):
            raise ValueError("only GRANTED may expose actuator authority")
        if self.phase == "GRANTED":
            if (
                self.generation is None
                or self.source_sequence is None
                or self.planner_intent_generation is None
                or self.reference_epoch <= 0
                or any(value is None for value in grant_values)
            ):
                raise ValueError("GRANTED authority requires complete correlation evidence")
            granted_at = float(self.granted_at_monotonic_s)
            if not math.isfinite(granted_at) or granted_at < 0:
                raise ValueError("grant timestamp must be finite and non-negative")
            _integer(self.granted_at_tick_ms, name="granted_at_tick_ms", minimum=0)
            _integer(self.dds_target_sequence, name="dds_target_sequence", minimum=1)
        elif any(value is not None for value in grant_values):
            raise ValueError("non-GRANTED authority cannot expose DDS grant evidence")


class ReferenceAuthorityState:
    """Small externally-serialized state machine for planner/actuator ownership."""

    def __init__(self) -> None:
        self._snapshot = ReferenceAuthoritySnapshot()

    @property
    def snapshot(self) -> ReferenceAuthoritySnapshot:
        return self._snapshot

    @property
    def executable(self) -> bool:
        return self._snapshot.phase in {"PRIMED", "GRANTED"}

    def require(self, *, generation: int, source_sequence: int) -> ReferenceAuthoritySnapshot:
        generation = _integer(generation, name="generation", minimum=0)
        source_sequence = _integer(
            source_sequence, name="source_sequence", minimum=1
        )
        previous = self._snapshot
        if (
            previous.generation == generation
            and previous.source_sequence == source_sequence
            and previous.phase != "REVOKED"
        ):
            # Replayed/periodic envelopes for the same admitted intent are
            # idempotent.  They cannot withdraw an installed or already-granted
            # reference and restart its planner deadline.
            return previous
        if (
            previous.generation is not None
            and generation == previous.generation
            and previous.source_sequence is not None
            and source_sequence < previous.source_sequence
        ):
            raise ValueError("required reference source sequence moved backwards")
        self._snapshot = ReferenceAuthoritySnapshot(
            phase="REQUIRED",
            generation=generation,
            source_sequence=source_sequence,
            reference_epoch=previous.reference_epoch,
            reason="awaiting a matching installed SONIC reference",
        )
        return self._snapshot

    def mark_installed(
        self,
        *,
        generation: int,
        source_sequence: int,
        planner_intent_generation: int,
        reference_epoch: int,
    ) -> ReferenceAuthoritySnapshot:
        self._require_identity(generation, source_sequence, phase="REQUIRED")
        planner_intent_generation = _integer(
            planner_intent_generation,
            name="planner_intent_generation",
            minimum=1,
        )
        reference_epoch = _integer(
            reference_epoch, name="reference_epoch", minimum=1
        )
        if reference_epoch <= self._snapshot.reference_epoch:
            raise ValueError("installed reference epoch did not advance")
        self._snapshot = ReferenceAuthoritySnapshot(
            phase="MATCHED_INSTALLED",
            generation=generation,
            source_sequence=source_sequence,
            planner_intent_generation=planner_intent_generation,
            reference_epoch=reference_epoch,
            reason="matching planner reference installed without actuator authority",
        )
        return self._snapshot

    def prime(
        self,
        *,
        generation: int,
        source_sequence: int,
        reference_epoch: int,
    ) -> ReferenceAuthoritySnapshot:
        self._require_identity(
            generation, source_sequence, phase="MATCHED_INSTALLED"
        )
        if reference_epoch != self._snapshot.reference_epoch:
            raise ValueError("primed reference epoch mismatch")
        self._snapshot = replace(
            self._snapshot,
            phase="PRIMED",
            reason="matched reference primed; awaiting first successful DDS target",
        )
        return self._snapshot

    def grant(
        self,
        *,
        generation: int,
        source_sequence: int,
        reference_epoch: int,
        granted_at_monotonic_s: float,
        granted_at_tick_ms: int,
        dds_target_sequence: int,
    ) -> ReferenceAuthoritySnapshot:
        self._require_identity(generation, source_sequence, phase="PRIMED")
        if reference_epoch != self._snapshot.reference_epoch:
            raise ValueError("granted reference epoch mismatch")
        granted_at = float(granted_at_monotonic_s)
        if not math.isfinite(granted_at) or granted_at < 0:
            raise ValueError("grant timestamp must be finite and non-negative")
        granted_at_tick_ms = _integer(
            granted_at_tick_ms, name="granted_at_tick_ms", minimum=0
        )
        dds_target_sequence = _integer(
            dds_target_sequence, name="dds_target_sequence", minimum=1
        )
        self._snapshot = replace(
            self._snapshot,
            phase="GRANTED",
            actuator_authority=True,
            granted_at_monotonic_s=granted_at,
            granted_at_tick_ms=granted_at_tick_ms,
            dds_target_sequence=dds_target_sequence,
            reason="matched reference target crossed DDS successfully",
        )
        return self._snapshot

    def revoke(self, reason: str) -> ReferenceAuthoritySnapshot:
        detail = str(reason).strip() or "reference actuator authority revoked"
        previous = self._snapshot
        self._snapshot = ReferenceAuthoritySnapshot(
            phase="REVOKED",
            generation=previous.generation,
            source_sequence=previous.source_sequence,
            planner_intent_generation=previous.planner_intent_generation,
            reference_epoch=previous.reference_epoch,
            reason=detail,
        )
        return self._snapshot

    def _require_identity(
        self,
        generation: int,
        source_sequence: int,
        *,
        phase: str,
    ) -> None:
        generation = _integer(generation, name="generation", minimum=0)
        source_sequence = _integer(
            source_sequence, name="source_sequence", minimum=1
        )
        current = self._snapshot
        if (
            current.phase != phase
            or current.generation != generation
            or current.source_sequence != source_sequence
        ):
            raise ValueError("reference authority correlation mismatch")
