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


class SessionStateError(RuntimeError):
    """Raised when an operation is invalid for the controlling session."""


class LeaseError(SessionStateError):
    """Raised when a lease renewal is invalid."""


@dataclass(frozen=True)
class SessionSnapshot:
    state: ControlState
    session_id: UUID | None
    operator_id: str | None
    lease_sequence: int | None
    lease_deadline: datetime | None
    reason: str
    can_acquire: bool


SafeStopCallback = Callable[[str], Awaitable[None]]


class SessionController:
    """Own one session and latch SAFE_HOLD on disconnect or lease expiry."""

    def __init__(
        self,
        *,
        lease_timeout: timedelta,
        safe_stop: SafeStopCallback,
    ) -> None:
        if lease_timeout <= timedelta(0):
            raise ValueError("lease_timeout must be positive")
        self._lease_timeout = lease_timeout
        self._safe_stop = safe_stop
        self._lock = asyncio.Lock()
        self._state = ControlState.SAFE_HOLD
        self._session_id: UUID | None = None
        self._operator_id: str | None = None
        self._lease_sequence: int | None = None
        self._lease_deadline: datetime | None = None
        self._safety_fault: str | None = None
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
            self._state = ControlState.ACTIVE
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
        """Enter SAFE_HOLD once when the active lease deadline is reached."""
        _require_aware(at)
        async with self._lock:
            if (
                self._state is not ControlState.ACTIVE
                or self._lease_deadline is None
                or at < self._lease_deadline
            ):
                return False
            await self._enter_safe_hold("lease expired")
            return True

    async def disconnected(self, reason: str) -> SessionSnapshot:
        async with self._lock:
            had_session = self._session_id is not None
            try:
                if had_session:
                    await self._enter_safe_hold(reason)
            finally:
                self._session_id = None
                self._operator_id = None
                self._lease_sequence = None
                self._lease_deadline = None
            if not had_session:
                self._reason = reason
            return self.snapshot

    async def force_safe_hold(self, reason: str) -> SessionSnapshot:
        async with self._lock:
            if self._state is ControlState.ACTIVE:
                await self._enter_safe_hold(reason)
            else:
                self._reason = (
                    self._latched_fault_reason() if self._safety_fault is not None else reason
                )
            return self.snapshot

    async def latch_safety_fault(self, reason: str) -> SessionSnapshot:
        """Block all command dispatch immediately when execution state is uncertain."""
        async with self._lock:
            self._state = ControlState.SAFE_HOLD
            self._lease_deadline = None
            self._safety_fault = reason[:231]
            self._reason = self._latched_fault_reason()
            return self.snapshot

    async def _enter_safe_hold(self, reason: str) -> None:
        self._state = ControlState.SAFE_HOLD
        self._lease_deadline = None
        self._reason = reason
        try:
            await self._safe_stop(reason)
        except asyncio.CancelledError:
            self._safety_fault = "CancelledError: mandatory safe-stop operation was cancelled"
            self._reason = self._latched_fault_reason()
            raise
        except Exception as exc:
            self._safety_fault = f"{type(exc).__name__}: {exc}"[:231]
            self._reason = self._latched_fault_reason()
            raise

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
            can_acquire=self._session_id is not None and self._safety_fault is None,
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
