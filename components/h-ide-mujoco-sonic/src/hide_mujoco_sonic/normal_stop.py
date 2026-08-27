"""Correlated state machine for a recoverable, controlled locomotion stop.

Normal stop is deliberately separate from terminal gait and simulation faults.
It proves, in order, that the current command was braked, measured stationary,
latched by Plant, and finally reflected by the driver.  Every acknowledgement
must carry the exact transaction identity, so delayed callbacks from an older
stop cannot confirm a newer one.

The state machine performs no I/O.  Its mutations are internally serialized
and every value exposed to readers is an immutable snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Callable
from uuid import UUID, uuid4


NORMAL_STOP_PHASES = frozenset(
    {
        "NONE",
        "BRAKING",
        "STATIONARY_ACK",
        "PLANT_CONFIRMED",
        "DRIVER_CONFIRMED",
        "FAILED",
    }
)

_IN_FLIGHT_PHASES = frozenset(
    {"BRAKING", "STATIONARY_ACK", "PLANT_CONFIRMED"}
)


class NormalStopIdentityError(ValueError):
    """An acknowledgement does not belong to the active stop transaction."""


class NormalStopTransitionError(RuntimeError):
    """A normal-stop phase transition is invalid or out of order."""


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _reason(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError("reason must be a non-empty string of at most 512 characters")
    return value


@dataclass(frozen=True, slots=True)
class NormalStopIdentity:
    """Exact correlation token carried by every normal-stop acknowledgement."""

    stop_id: UUID
    control_generation: int
    gait_source_sequence: int
    stop_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.stop_id, UUID):
            raise ValueError("stop_id must be a UUID")
        _integer(
            self.control_generation,
            name="control_generation",
            minimum=0,
        )
        _integer(
            self.gait_source_sequence,
            name="gait_source_sequence",
            minimum=1,
        )
        _integer(self.stop_epoch, name="stop_epoch", minimum=1)


@dataclass(frozen=True, slots=True)
class NormalStopSnapshot:
    """Immutable public state for the current (or most recent) stop."""

    phase: str = "NONE"
    stop_id: UUID | None = None
    control_generation: int | None = None
    gait_source_sequence: int | None = None
    stop_epoch: int = 0
    reason: str = "no normal stop is active"

    def __post_init__(self) -> None:
        if self.phase not in NORMAL_STOP_PHASES:
            raise ValueError("normal-stop phase is invalid")
        _reason(self.reason)
        if self.phase == "NONE":
            if (
                self.stop_id is not None
                or self.control_generation is not None
                or self.gait_source_sequence is not None
                or self.stop_epoch != 0
            ):
                raise ValueError("NONE cannot carry a normal-stop identity")
            return
        NormalStopIdentity(
            stop_id=self.stop_id,  # type: ignore[arg-type]
            control_generation=self.control_generation,  # type: ignore[arg-type]
            gait_source_sequence=self.gait_source_sequence,  # type: ignore[arg-type]
            stop_epoch=self.stop_epoch,
        )

    @property
    def identity(self) -> NormalStopIdentity | None:
        if self.phase == "NONE":
            return None
        return NormalStopIdentity(
            stop_id=self.stop_id,  # type: ignore[arg-type]
            control_generation=self.control_generation,  # type: ignore[arg-type]
            gait_source_sequence=self.gait_source_sequence,  # type: ignore[arg-type]
            stop_epoch=self.stop_epoch,
        )

    @property
    def safe_hold_confirmed(self) -> bool:
        """True only after the final driver acknowledgement."""

        return self.phase == "DRIVER_CONFIRMED"


class NormalStopState:
    """Thread-safe transaction coordinator for non-terminal normal stops.

    ``begin`` owns UUID and epoch issuance.  A completed transaction may be
    followed only by a strictly newer control generation and gait sequence.
    ``FAILED`` is terminal for this coordinator; terminal gait faults belong to
    the separate motion-fault path and are intentionally not represented here.
    """

    def __init__(
        self,
        *,
        stop_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        if not callable(stop_id_factory):
            raise ValueError("stop_id_factory must be callable")
        self._stop_id_factory = stop_id_factory
        self._lock = threading.RLock()
        self._snapshot = NormalStopSnapshot()
        self._issued_stop_ids: set[UUID] = set()

    @property
    def snapshot(self) -> NormalStopSnapshot:
        with self._lock:
            return self._snapshot

    def begin(
        self,
        *,
        control_generation: int,
        gait_source_sequence: int,
        allow_source_sequence_reset: bool = False,
    ) -> NormalStopSnapshot:
        """Start braking, or return the exact already-active transaction.

        A bridge process restart may reset its source sequence while advancing
        the persisted control generation.  The caller must prove that restart
        separately, then opt in here.  The exception never relaxes generation,
        UUID, epoch, active-transaction, or failure monotonicity.
        """

        if not isinstance(allow_source_sequence_reset, bool):
            raise ValueError("allow_source_sequence_reset must be a boolean")

        generation = _integer(
            control_generation,
            name="control_generation",
            minimum=0,
        )
        source_sequence = _integer(
            gait_source_sequence,
            name="gait_source_sequence",
            minimum=1,
        )
        with self._lock:
            current = self._snapshot
            same_association = (
                current.control_generation == generation
                and current.gait_source_sequence == source_sequence
            )
            if allow_source_sequence_reset:
                if current.phase != "DRIVER_CONFIRMED":
                    raise NormalStopTransitionError(
                        "source sequence reset requires a confirmed normal stop"
                    )
                assert current.control_generation is not None
                if generation <= current.control_generation:
                    raise NormalStopTransitionError(
                        "source sequence reset requires a newer control generation"
                    )
            if current.phase == "FAILED":
                raise NormalStopTransitionError(
                    "normal-stop coordinator is terminal after failure"
                )
            if current.phase in _IN_FLIGHT_PHASES:
                if same_association:
                    return current
                raise NormalStopTransitionError(
                    "another normal-stop transaction is still active"
                )
            if current.phase == "DRIVER_CONFIRMED":
                if same_association:
                    return current
                assert current.control_generation is not None
                assert current.gait_source_sequence is not None
                if generation <= current.control_generation:
                    raise NormalStopTransitionError(
                        "a new normal stop requires a newer control generation"
                    )
                if (
                    source_sequence <= current.gait_source_sequence
                    and not allow_source_sequence_reset
                ):
                    raise NormalStopTransitionError(
                        "a new normal stop requires a newer gait source sequence"
                    )

            stop_id = self._stop_id_factory()
            if not isinstance(stop_id, UUID):
                raise ValueError("stop_id_factory must return a UUID")
            if stop_id in self._issued_stop_ids:
                raise ValueError("stop_id_factory returned an already-issued UUID")
            next_epoch = current.stop_epoch + 1
            self._issued_stop_ids.add(stop_id)
            self._snapshot = NormalStopSnapshot(
                phase="BRAKING",
                stop_id=stop_id,
                control_generation=generation,
                gait_source_sequence=source_sequence,
                stop_epoch=next_epoch,
                reason="controlled braking requested",
            )
            return self._snapshot

    def acknowledge_stationary(
        self,
        identity: NormalStopIdentity,
    ) -> NormalStopSnapshot:
        """Accept the measured-stationary acknowledgement after braking."""

        return self._advance(
            identity,
            predecessor="BRAKING",
            target="STATIONARY_ACK",
            later=frozenset({"PLANT_CONFIRMED", "DRIVER_CONFIRMED"}),
            reason="stationary measurement acknowledged",
        )

    def confirm_plant(
        self,
        identity: NormalStopIdentity,
    ) -> NormalStopSnapshot:
        """Accept Plant's stop-latch confirmation for the same transaction."""

        return self._advance(
            identity,
            predecessor="STATIONARY_ACK",
            target="PLANT_CONFIRMED",
            later=frozenset({"DRIVER_CONFIRMED"}),
            reason="Plant safety-stop latch confirmed",
        )

    def confirm_driver(
        self,
        identity: NormalStopIdentity,
    ) -> NormalStopSnapshot:
        """Accept the final driver confirmation and expose safe hold."""

        return self._advance(
            identity,
            predecessor="PLANT_CONFIRMED",
            target="DRIVER_CONFIRMED",
            later=frozenset(),
            reason="driver safe hold confirmed",
        )

    def fail(
        self,
        identity: NormalStopIdentity,
        *,
        reason: str,
    ) -> NormalStopSnapshot:
        """Latch an unrecoverable failure of the active normal-stop handshake."""

        detail = _reason(reason)
        with self._lock:
            self._require_exact_identity(identity)
            current = self._snapshot
            if current.phase == "FAILED":
                return current
            if current.phase == "DRIVER_CONFIRMED":
                raise NormalStopTransitionError(
                    "a confirmed normal stop cannot transition to FAILED"
                )
            if current.phase not in _IN_FLIGHT_PHASES:
                raise NormalStopTransitionError(
                    "normal-stop failure requires an active transaction"
                )
            self._snapshot = replace(current, phase="FAILED", reason=detail)
            return self._snapshot

    def _advance(
        self,
        identity: NormalStopIdentity,
        *,
        predecessor: str,
        target: str,
        later: frozenset[str],
        reason: str,
    ) -> NormalStopSnapshot:
        with self._lock:
            self._require_exact_identity(identity)
            current = self._snapshot
            if current.phase == "FAILED":
                raise NormalStopTransitionError(
                    "normal-stop failure is terminal"
                )
            if current.phase == target or current.phase in later:
                return current
            if current.phase != predecessor:
                raise NormalStopTransitionError(
                    f"{target} acknowledgement is out of order from {current.phase}"
                )
            self._snapshot = replace(current, phase=target, reason=reason)
            return self._snapshot

    def _require_exact_identity(self, identity: NormalStopIdentity) -> None:
        if not isinstance(identity, NormalStopIdentity):
            raise NormalStopIdentityError(
                "normal-stop acknowledgement requires NormalStopIdentity"
            )
        current_identity = self._snapshot.identity
        if current_identity is None or identity != current_identity:
            raise NormalStopIdentityError(
                "normal-stop acknowledgement identity does not match the active transaction"
            )
