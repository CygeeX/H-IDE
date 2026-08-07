"""Command validation, idempotency, arbitration, execution, and cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID

from robot_agent.arbitration import ResourceArbiter
from robot_agent.config import (
    CommandConfig,
    CommandParametersError,
    validate_command_params,
)
from robot_agent.ledger import TERMINAL_STATES, CommandLedger, CommandRecord, CommandState
from robot_agent.protocol import CommandRequest
from robot_agent.session import SessionController, SessionStateError

_ADAPTER_CANCEL_TIMEOUT_SECONDS = 0.25
_EXECUTION_SETTLE_TIMEOUT_SECONDS = 0.1
_CANCEL_NOT_ACKNOWLEDGED = "adapter cancellation was not acknowledged"


@dataclass(frozen=True)
class ExecutionResult:
    succeeded: bool
    detail: str
    result: Any | None = None


class CommandCancellationError(RuntimeError):
    """Raised when a local command may still be executing after cancellation."""

    def __init__(self, record: CommandRecord) -> None:
        self.record = record
        super().__init__(
            f"command {record.command_id} {record.state.value.lower()}, "
            f"but {_CANCEL_NOT_ACKNOWLEDGED}"
        )


class CommandExecutor(Protocol):
    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult: ...

    async def cancel(self, command_id: UUID) -> None: ...


class CommandCoordinator:
    """Keep remote command intent inside the configured local trust boundary."""

    def __init__(
        self,
        *,
        commands: Mapping[str, CommandConfig],
        ledger: CommandLedger,
        arbiter: ResourceArbiter,
        session: SessionController,
        executor: CommandExecutor,
    ) -> None:
        self._commands = dict(commands)
        self._ledger = ledger
        self._arbiter = arbiter
        self._session = session
        self._executor = executor
        self._tasks: dict[UUID, asyncio.Task[Any]] = {}
        self._registration_locks: dict[UUID, asyncio.Lock] = {}
        self._executing: set[UUID] = set()
        self._cancel_reasons: dict[UUID, tuple[str, datetime]] = {}
        self._cancel_failures: set[UUID] = set()
        self._preempting: set[UUID] = set()

    async def submit(self, request: CommandRequest, *, at: datetime) -> CommandRecord:
        """Persist a request first, then execute it at most once."""
        self._session.assert_can_command(request.session_id)
        payload = request.payload
        current_task = asyncio.current_task()
        if current_task is None:  # pragma: no cover - asyncio always supplies one here.
            raise RuntimeError("command submission is not running in an asyncio task")
        owns_tracking = payload.command_id not in self._tasks
        if owns_tracking:
            self._tasks[payload.command_id] = current_task
            self._registration_locks[payload.command_id] = asyncio.Lock()
        registration_lock = self._registration_locks[payload.command_id]
        try:
            return await self._submit_tracked(
                request,
                at=at,
                observe_cancellation=owns_tracking,
                registration_lock=registration_lock,
            )
        finally:
            if owns_tracking:
                self._arbiter.release(payload.command_id)
                self._cancel_reasons.pop(payload.command_id, None)
                if self._tasks.get(payload.command_id) is current_task:
                    del self._tasks[payload.command_id]
                if self._registration_locks.get(payload.command_id) is registration_lock:
                    del self._registration_locks[payload.command_id]

    async def _submit_tracked(
        self,
        request: CommandRequest,
        *,
        at: datetime,
        observe_cancellation: bool,
        registration_lock: asyncio.Lock,
    ) -> CommandRecord:
        payload = request.payload
        loop = asyncio.get_running_loop()
        received_monotonic = loop.time()
        initial_remaining_seconds = (
            payload.issued_at.timestamp() + payload.ttl_ms / 1000.0 - at.timestamp()
        )
        dispatch_deadline = received_monotonic + initial_remaining_seconds

        def remaining_seconds() -> float:
            return dispatch_deadline - loop.time()

        async with registration_lock:
            registration = await asyncio.to_thread(
                self._ledger.register,
                command_id=payload.command_id,
                kind=payload.kind,
                payload=payload.model_dump(mode="json"),
                session_id=request.session_id,
                at=at,
            )
        if not registration.is_new:
            return registration.record
        if not observe_cancellation:
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.CANCELLED,
                detail="primary concurrent submission did not establish ledger ownership",
                at=_utc_now(),
            )
        if observe_cancellation:
            cancelled = await self._take_pre_dispatch_cancellation(payload.command_id)
            if cancelled is not None:
                return cancelled
        try:
            self._session.assert_can_command(request.session_id)
        except SessionStateError:
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.CANCELLED,
                detail="control entered SAFE_HOLD before command validation",
                at=_utc_now(),
            )

        command = self._commands.get(payload.kind)
        if command is None:
            return await self._reject(
                payload.command_id,
                f"unknown command kind {payload.kind!r}",
                at,
            )
        if payload.ttl_ms > command.max_ttl_ms:
            return await self._reject(
                payload.command_id,
                f"ttl_ms {payload.ttl_ms} exceeds registry maximum {command.max_ttl_ms}",
                at,
            )
        if remaining_seconds() <= 0:
            return await self._reject(
                payload.command_id,
                "command expired before dispatch",
                _utc_now(),
            )
        try:
            validate_command_params(command, payload.params)
        except CommandParametersError as exc:
            return await self._reject(payload.command_id, f"invalid command params: {exc}", at)

        claim = self._arbiter.claim(payload.command_id, command.resources)
        if not claim.acquired:
            conflicts = ", ".join(
                f"{resource}={owner}" for resource, owner in claim.conflicts.items()
            )
            return await self._reject(payload.command_id, f"resource conflict: {conflicts}", at)

        if remaining_seconds() <= 0:
            self._arbiter.release(payload.command_id)
            return await self._reject(
                payload.command_id,
                "command expired before local dispatch",
                _utc_now(),
            )

        await asyncio.to_thread(
            self._ledger.transition,
            payload.command_id,
            CommandState.RUNNING,
            detail="command dispatched to local adapter",
            at=_utc_now(),
        )
        if observe_cancellation:
            cancelled = await self._take_pre_dispatch_cancellation(payload.command_id)
            if cancelled is not None:
                return cancelled
        try:
            self._session.assert_can_command(request.session_id)
        except SessionStateError:
            self._arbiter.release(payload.command_id)
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.CANCELLED,
                detail="control entered SAFE_HOLD before local dispatch",
                at=_utc_now(),
            )
        if payload.command_id not in self._arbiter.active_commands:
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.CANCELLED,
                detail="safe stop preempted command before local dispatch",
                at=_utc_now(),
            )
        remaining_before_execution = remaining_seconds()
        if remaining_before_execution <= 0:
            self._arbiter.release(payload.command_id)
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.TIMED_OUT,
                detail="command expired while preparing local dispatch",
                at=_utc_now(),
            )
        timeout_seconds = min(command.timeout_ms / 1000.0, remaining_before_execution)
        self._executing.add(payload.command_id)
        execution_task = asyncio.create_task(
            self._executor.execute(payload.command_id, command, payload.params)
        )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(execution_task),
                timeout=timeout_seconds,
            )
            state = CommandState.SUCCEEDED if result.succeeded else CommandState.FAILED
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                state,
                detail=result.detail,
                result=result.result,
                at=_utc_now(),
            )
        except asyncio.TimeoutError:
            cancel_acknowledged = await self._stop_execution(
                payload.command_id,
                execution_task,
            )
            detail = f"local adapter exceeded {timeout_seconds:.3f}s deadline"
            if not cancel_acknowledged:
                detail = f"{detail}; {_CANCEL_NOT_ACKNOWLEDGED}"
            record = await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.TIMED_OUT,
                detail=detail,
                at=_utc_now(),
            )
            if not cancel_acknowledged:
                raise CommandCancellationError(record) from None
            return record
        except asyncio.CancelledError:
            cancel_acknowledged = await self._stop_execution(
                payload.command_id,
                execution_task,
            )
            record = await asyncio.to_thread(self._ledger.get, payload.command_id)
            if record.state in TERMINAL_STATES:
                return record
            detail, transition_at = self._cancel_reasons.pop(
                payload.command_id,
                ("command cancelled", _utc_now()),
            )
            if not cancel_acknowledged:
                detail = f"{detail}; {_CANCEL_NOT_ACKNOWLEDGED}"
                self._cancel_failures.add(payload.command_id)
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.CANCELLED,
                detail=detail,
                at=transition_at,
            )
        except Exception as exc:
            return await asyncio.to_thread(
                self._ledger.transition,
                payload.command_id,
                CommandState.FAILED,
                detail=f"local adapter failed: {type(exc).__name__}: {exc}",
                at=_utc_now(),
            )
        finally:
            if not execution_task.done():
                execution_task.cancel()
            execution_task.add_done_callback(_consume_task_outcome)
            self._arbiter.release(payload.command_id)
            self._executing.discard(payload.command_id)

    async def _take_pre_dispatch_cancellation(
        self,
        command_id: UUID,
    ) -> CommandRecord | None:
        cancellation = self._cancel_reasons.pop(command_id, None)
        if cancellation is None:
            return None
        reason, at = cancellation
        record = await asyncio.to_thread(self._ledger.get, command_id)
        if record.state in TERMINAL_STATES:
            return record
        return await asyncio.to_thread(
            self._ledger.transition,
            command_id,
            CommandState.CANCELLED,
            detail=reason,
            at=at,
        )

    async def _stop_execution(
        self,
        command_id: UUID,
        execution_task: asyncio.Task[ExecutionResult],
    ) -> bool:
        acknowledged = True
        failure_detail = _CANCEL_NOT_ACKNOWLEDGED
        try:
            await asyncio.wait_for(
                self._executor.cancel(command_id),
                timeout=_ADAPTER_CANCEL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            acknowledged = False
            failure_detail = f"{_CANCEL_NOT_ACKNOWLEDGED}: {type(exc).__name__}: {exc}"
        execution_stopped = await _stop_task_within(
            execution_task,
            timeout=_EXECUTION_SETTLE_TIMEOUT_SECONDS,
        )
        if not execution_stopped:
            acknowledged = False
            failure_detail = (
                f"{failure_detail}; local execution coroutine did not stop within "
                f"{_EXECUTION_SETTLE_TIMEOUT_SECONDS:.3f}s"
            )
        if not acknowledged and command_id not in self._preempting:
            await self._session.latch_safety_fault(failure_detail)
        return acknowledged

    async def _reject(self, command_id: UUID, detail: str, at: datetime) -> CommandRecord:
        return await asyncio.to_thread(
            self._ledger.transition,
            command_id,
            CommandState.REJECTED,
            detail=detail,
            at=at,
        )

    async def cancel(self, command_id: UUID, *, reason: str, at: datetime) -> bool:
        """Cancel one active command without disturbing disjoint resource groups."""
        task = self._tasks.get(command_id)
        if task is None or task.done():
            return False
        self._cancel_reasons[command_id] = (reason, at)
        if command_id in self._executing:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._cancel_reasons.pop(command_id, None)
        if command_id in self._cancel_failures:
            self._cancel_failures.discard(command_id)
            raise CommandCancellationError(await asyncio.to_thread(self._ledger.get, command_id))
        return True

    async def preempt_all(self, *, reason: str, at: datetime) -> tuple[UUID, ...]:
        """Bypass locks, request adapter cancellation, and terminalize all commands."""
        claimed_ids = self._arbiter.preempt_all()
        pending_ids = tuple(
            command_id
            for command_id, task in self._tasks.items()
            if not task.done()
        )
        command_ids = tuple(dict.fromkeys((*claimed_ids, *pending_ids)))
        self._preempting.update(command_ids)
        tasks: list[asyncio.Task[Any]] = []
        for command_id in command_ids:
            task = self._tasks.get(command_id)
            if task is not None and not task.done():
                self._cancel_reasons[command_id] = (reason, at)
                if command_id in self._executing:
                    task.cancel()
                tasks.append(task)
        try:
            if tasks:
                await asyncio.wait(tasks)
            failed = set(command_ids).intersection(self._cancel_failures)
        finally:
            self._preempting.difference_update(command_ids)
        self._cancel_failures.difference_update(command_ids)
        if failed:
            failed_id = sorted(failed, key=str)[0]
            raise CommandCancellationError(
                await asyncio.to_thread(self._ledger.get, failed_id)
            )
        return command_ids

    @property
    def active_commands(self) -> frozenset[UUID]:
        return self._arbiter.active_commands


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _stop_task_within(task: asyncio.Task[Any], *, timeout: float) -> bool:
    if not task.done():
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if task not in done:
            task.add_done_callback(_consume_task_outcome)
            return False
    _consume_task_outcome(task)
    return True


def _consume_task_outcome(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()
