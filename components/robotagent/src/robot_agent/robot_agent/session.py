"""Fail-closed controlling-session and application-lease state machine."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID


class ControlState(str, Enum):
    SAFE_HOLD = "SAFE_HOLD"
    ACTIVE = "ACTIVE"


class SafeStopState(str, Enum):
    """Truth state for the most recent physical safe-stop boundary."""

    REQUESTED = "requested"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class SessionStateError(RuntimeError):
    """Raised when an operation is invalid for the controlling session."""


class LeaseError(SessionStateError):
    """Raised when a lease renewal is invalid."""


class RetryableControlArmError(SessionStateError):
    """A bounded control-arm precondition failed without granting authority."""


@dataclass(frozen=True)
class SessionSnapshot:
    state: ControlState
    session_id: UUID | None
    operator_id: str | None
    lease_sequence: int | None
    lease_deadline: datetime | None
    reason: str
    can_acquire: bool
    authority_revoked: bool
    safe_stop_state: SafeStopState
    safe_stop_epoch: int


SafeStopCallback = Callable[[str], Awaitable[None]]
ControlArmCallback = Callable[[], Awaitable[None]]


class SessionController:
    """Own one remote session without conflating it with actuator shutdown.

    A lease, transport, or command lifecycle boundary revokes the remote
    session's authority, but it must not stop the resident locomotion policy.
    The physical safe-stop callback is reserved for an explicit developer
    boundary (and the process-start boundary invoked by the runtime).
    """

    def __init__(
        self,
        *,
        lease_timeout: timedelta,
        safe_stop: SafeStopCallback,
        initial_safe_stop_epoch: int = 0,
    ) -> None:
        if lease_timeout <= timedelta(0):
            raise ValueError("lease_timeout must be positive")
        if initial_safe_stop_epoch < 0:
            raise ValueError("initial_safe_stop_epoch must not be negative")
        self._lease_timeout = lease_timeout
        self._safe_stop = safe_stop
        self._lock = asyncio.Lock()
        self._state = ControlState.SAFE_HOLD
        self._session_id: UUID | None = None
        self._operator_id: str | None = None
        self._lease_sequence: int | None = None
        self._lease_deadline: datetime | None = None
        self._safety_fault: str | None = None
        # Standalone core users historically start in a confirmed logical hold.
        # The runtime refreshes this boundary against the physical adapter during
        # initialize(), using the next durable epoch before it advertises ready.
        self._authority_revoked = True
        self._safe_stop_state = SafeStopState.CONFIRMED
        self._safe_stop_epoch = initial_safe_stop_epoch
        self._reason = "agent startup requires explicit control.acquire"

    async def connected(self, session_id: UUID) -> SessionSnapshot:
        async with self._lock:
            if self._session_id is not None and self._session_id != session_id:
                raise SessionStateError("another controlling session is already connected")
            self._session_id = session_id
            self._operator_id = None
            self._lease_sequence = None
            self._lease_deadline = None
            self._state = ControlState.SAFE_HOLD
            self._reason = (
                self._latched_fault_reason()
                if self._safety_fault is not None
                else "new session requires explicit control.acquire"
            )
            return self.snapshot

    async def acquire(
        self,
        session_id: UUID,
        *,
        operator_id: str,
        at: datetime,
        before_activate: ControlArmCallback | None = None,
    ) -> SessionSnapshot:
        _require_aware(at)
        if not operator_id:
            raise SessionStateError("operator_id must not be empty")
        async with self._lock:
            self._require_session(session_id)
            if self._safety_fault is not None:
                raise SessionStateError(f"control blocked: {self._latched_fault_reason()}")
            if self._state is ControlState.ACTIVE:
                return self.snapshot
            if (
                not self._authority_revoked
                or self._safe_stop_state is not SafeStopState.CONFIRMED
            ):
                raise SessionStateError(
                    "control blocked: the latest physical safe-stop is not confirmed"
                )
            if before_activate is not None:
                try:
                    # Keep the session lock across this bounded local arm.  A
                    # simultaneous lease/disconnect stop therefore cannot
                    # race SAFE_HOLD -> ACTIVE between the decision to arm and
                    # the final state transition.
                    await before_activate()
                except RetryableControlArmError as exc:
                    # The callback explicitly proved that it did not grant
                    # actuator authority.  Keep the already-confirmed logical
                    # hold reusable so the gateway can retry the same acquire
                    # after the measured simulator state settles.
                    self._relinquish_control(
                        f"control activation deferred: {exc}"[:256]
                    )
                    raise
                except Exception as exc:
                    self._latch_control_fault(
                        f"control activation failed closed: {exc}"
                    )
                    raise
            self._state = ControlState.ACTIVE
            self._authority_revoked = False
            self._operator_id = operator_id
            self._lease_sequence = -1
            self._lease_deadline = at + self._lease_timeout
            self._reason = "control acquired"
            return self.snapshot

    async def renew(
        self,
        session_id: UUID,
        *,
        sequence: int,
        at: datetime,
    ) -> SessionSnapshot:
        _require_aware(at)
        if sequence < 0:
            raise LeaseError("lease sequence must not be negative")
        async with self._lock:
            self._require_session(session_id)
            if self._state is not ControlState.ACTIVE:
                raise LeaseError("lease cannot be renewed while control is not ACTIVE")
            assert self._lease_sequence is not None
            if sequence < self._lease_sequence:
                raise LeaseError(
                    f"lease sequence regressed from {self._lease_sequence} to {sequence}"
                )
            if sequence == self._lease_sequence:
                return self.snapshot
            self._lease_sequence = sequence
            self._lease_deadline = at + self._lease_timeout
            self._reason = "lease renewed"
            return self.snapshot

    async def check_lease(self, at: datetime) -> bool:
        """Relinquish remote authority once when its lease expires.

        Command cancellation and zero-motion quiescence are owned by the
        runtime.  Expiry never calls the physical safe-stop callback.
        """
        _require_aware(at)
        async with self._lock:
            if (
                self._state is not ControlState.ACTIVE
                or self._lease_deadline is None
                or at < self._lease_deadline
            ):
                return False
            self._relinquish_control("lease expired")
            return True

    async def disconnected(self, reason: str) -> SessionSnapshot:
        async with self._lock:
            had_session = self._session_id is not None
            try:
                if had_session:
                    self._relinquish_control(reason)
            finally:
                self._session_id = None
                self._operator_id = None
                self._lease_sequence = None
                self._lease_deadline = None
            if not had_session:
                self._reason = reason
            return self.snapshot

    async def force_safe_hold(self, reason: str) -> SessionSnapshot:
        return await self.ensure_safe_hold(reason, refresh=False)

    async def ensure_safe_hold(
        self,
        reason: str,
        *,
        refresh: bool,
    ) -> SessionSnapshot:
        """Revoke authority and confirm one physical safe-stop boundary.

        ``refresh`` is used for startup and other explicit fresh boundaries even
        when the logical control state is already SAFE_HOLD.
        """

        async with self._lock:
            needs_stop = (
                refresh
                or self._state is ControlState.ACTIVE
                or self._safe_stop_state is not SafeStopState.CONFIRMED
            )
            if not needs_stop:
                self._reason = (
                    self._latched_fault_reason() if self._safety_fault is not None else reason
                )
                return self.snapshot
            if self._safe_stop_state is not SafeStopState.REQUESTED:
                self._begin_safe_stop(reason)
            await self._confirm_requested_safe_stop(reason)
            return self.snapshot

    async def revoke_authority(self, reason: str) -> SessionSnapshot:
        """Immediately revoke only the remote command authority."""

        async with self._lock:
            self._relinquish_control(reason)
            return self.snapshot

    async def latch_safety_fault(self, reason: str) -> SessionSnapshot:
        """Block remote dispatch without stopping the resident policy."""
        async with self._lock:
            self._latch_control_fault(reason)
            return self.snapshot

    def _relinquish_control(self, reason: str) -> None:
        """Enter logical SAFE_HOLD while preserving physical stop truth."""

        self._state = ControlState.SAFE_HOLD
        self._authority_revoked = True
        self._operator_id = None
        self._lease_sequence = None
        self._lease_deadline = None
        self._reason = (
            self._latched_fault_reason()
            if self._safety_fault is not None
            else reason
        )

    def _latch_control_fault(self, reason: str) -> None:
        self._safety_fault = reason[:231]
        self._relinquish_control(self._latched_fault_reason())

    async def _enter_safe_hold(self, reason: str) -> None:
        self._begin_safe_stop(reason)
        await self._confirm_requested_safe_stop(reason)

    def _begin_safe_stop(self, reason: str) -> None:
        self._state = ControlState.SAFE_HOLD
        self._authority_revoked = True
        self._lease_deadline = None
        self._reason = reason
        self._safe_stop_epoch += 1
        self._safe_stop_state = SafeStopState.REQUESTED

    async def _confirm_requested_safe_stop(self, reason: str) -> None:
        try:
            await self._safe_stop(reason)
        except asyncio.CancelledError:
            self._safety_fault = "CancelledError: mandatory safe-stop operation was cancelled"
            self._safe_stop_state = SafeStopState.FAILED
            self._reason = self._latched_fault_reason()
            raise
        except Exception as exc:
            self._safety_fault = f"{type(exc).__name__}: {exc}"[:231]
            self._safe_stop_state = SafeStopState.FAILED
            self._reason = self._latched_fault_reason()
            raise
        self._safe_stop_state = SafeStopState.CONFIRMED
        self._reason = reason

    def _latched_fault_reason(self) -> str:
        assert self._safety_fault is not None
        return f"safety fault is latched: {self._safety_fault}"[:256]

    def assert_can_command(self, session_id: UUID) -> None:
        self._require_session(session_id)
        if self._state is not ControlState.ACTIVE:
            raise SessionStateError(
                "commands are blocked in SAFE_HOLD; explicit control.acquire is required"
            )

    def _require_session(self, session_id: UUID) -> None:
        if self._session_id != session_id:
            raise SessionStateError(
                f"session_id does not own control: received {session_id}, "
                f"current {self._session_id}"
            )

    @property
    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            state=self._state,
            session_id=self._session_id,
            operator_id=self._operator_id,
            lease_sequence=(
                None
                if self._lease_sequence is None or self._lease_sequence < 0
                else self._lease_sequence
            ),
            lease_deadline=self._lease_deadline,
            reason=self._reason,
            can_acquire=(
                self._session_id is not None
                and self._safety_fault is None
                and self._authority_revoked
                and self._safe_stop_state is SafeStopState.CONFIRMED
            ),
            authority_revoked=self._authority_revoked,
            safe_stop_state=self._safe_stop_state,
            safe_stop_epoch=self._safe_stop_epoch,
        )

    @property
    def state(self) -> ControlState:
        return self._state

    @property
    def session_id(self) -> UUID | None:
        return self._session_id

    @property
    def lease_deadline(self) -> datetime | None:
        return self._lease_deadline


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise SessionStateError("session timestamps must include a timezone")
