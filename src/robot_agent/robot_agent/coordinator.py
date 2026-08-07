"""Command validation, idempotency, arbitration, execution, and cancellation."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
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
from robot_agent.session import SessionController


@dataclass(frozen=True)
class ExecutionResult:
    succeeded: bool
    detail: str
    result: Any | None = None


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
        self._tasks: dict[UUID, asyncio.Task[CommandRecord]] = {}
        self._cancel_reasons: dict[UUID, str] = {}

    async def submit(self, request: CommandRequest, *, at: datetime) -> CommandRecord:
        """Persist a request first, then execute it at most once."""
        self._session.assert_can_command(request.session_id)
        payload = request.payload
        registration = self._ledger.register(
            command_id=payload.command_id,
            kind=payload.kind,
            payload=payload.model_dump(mode="json"),
            session_id=request.session_id,
            at=at,
        )
        if not registration.is_new:
            return registration.record

        command = self._commands.get(payload.kind)
        if command is None:
            return self._reject(payload.command_id, f"unknown command kind {payload.kind!r}", at)
        if payload.ttl_ms > command.max_ttl_ms:
            return self._reject(
                payload.command_id,
                f"ttl_ms {payload.ttl_ms} exceeds registry maximum {command.max_ttl_ms}",
                at,
            )
        remaining_seconds = payload.issued_at.timestamp() + payload.ttl_ms / 1000.0 - at.timestamp()
        if remaining_seconds <= 0:
            return self._reject(payload.command_id, "command expired before dispatch", at)
        try:
            validate_command_params(command, payload.params)
        except CommandParametersError as exc:
            return self._reject(payload.command_id, f"invalid command params: {exc}", at)

        claim = self._arbiter.claim(payload.command_id, command.resources)
        if not claim.acquired:
            conflicts = ", ".join(
                f"{resource}={owner}" for resource, owner in claim.conflicts.items()
            )
            return self._reject(payload.command_id, f"resource conflict: {conflicts}", at)

        self._ledger.transition(
            payload.command_id,
            CommandState.RUNNING,
            detail="command dispatched to local adapter",
            at=at,
        )
        current_task = asyncio.current_task()
        if current_task is None:
            self._arbiter.release(payload.command_id)
            return self._ledger.transition(
                payload.command_id,
                CommandState.FAILED,
                detail="command submission is not running in an asyncio task",
                at=at,
            )
        task = current_task
        self._tasks[payload.command_id] = task
        timeout_seconds = min(command.timeout_ms / 1000.0, remaining_seconds)
        try:
            result = await asyncio.wait_for(
                self._executor.execute(payload.command_id, command, payload.params),
                timeout=timeout_seconds,
            )
            state = CommandState.SUCCEEDED if result.succeeded else CommandState.FAILED
            return self._ledger.transition(
                payload.command_id,
                state,
                detail=result.detail,
                result=result.result,
                at=at,
            )
        except asyncio.TimeoutError:
            await self._executor.cancel(payload.command_id)
            return self._ledger.transition(
                payload.command_id,
                CommandState.TIMED_OUT,
                detail=f"local adapter exceeded {timeout_seconds:.3f}s deadline",
                at=at,
            )
        except asyncio.CancelledError:
            record = self._ledger.get(payload.command_id)
            if record.state in TERMINAL_STATES:
                return record
            return self._ledger.transition(
                payload.command_id,
                CommandState.CANCELLED,
                detail=self._cancel_reasons.pop(payload.command_id, "command cancelled"),
                at=at,
            )
        except Exception as exc:
            return self._ledger.transition(
                payload.command_id,
                CommandState.FAILED,
                detail=f"local adapter failed: {type(exc).__name__}: {exc}",
                at=at,
            )
        finally:
            self._arbiter.release(payload.command_id)
            if self._tasks.get(payload.command_id) is task:
                del self._tasks[payload.command_id]

    def _reject(self, command_id: UUID, detail: str, at: datetime) -> CommandRecord:
        return self._ledger.transition(
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
        self._cancel_reasons[command_id] = reason
        with contextlib.suppress(Exception):
            await self._executor.cancel(command_id)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return True

    async def preempt_all(self, *, reason: str, at: datetime) -> tuple[UUID, ...]:
        """Bypass locks, request adapter cancellation, and terminalize all commands."""
        command_ids = self._arbiter.preempt_all()
        tasks: list[asyncio.Task[CommandRecord]] = []
        for command_id in command_ids:
            self._cancel_reasons[command_id] = reason
            with contextlib.suppress(Exception):
                await self._executor.cancel(command_id)
                # Safe-stop itself is owned by the caller and must continue even if
                # an individual adapter cannot acknowledge cancellation.
            task = self._tasks.get(command_id)
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return command_ids

    @property
    def active_commands(self) -> frozenset[UUID]:
        return self._arbiter.active_commands
