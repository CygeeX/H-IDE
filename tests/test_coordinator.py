from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from robot_agent.arbitration import ResourceArbiter
from robot_agent.coordinator import CommandCoordinator, ExecutionResult
from robot_agent.ledger import CommandIdConflict, CommandLedger, CommandState
from robot_agent.protocol import CommandRequest
from robot_agent.session import SessionController

from robot_agent.config import CommandConfig

NOW = datetime(2026, 8, 7, 15, 0, tzinfo=timezone.utc)


def command_config(resource: str, *, timeout_ms: int = 5000) -> CommandConfig:
    return CommandConfig.model_validate(
        {
            "adapter": "action",
            "ros_name": f"/{resource}/execute",
            "ros_type": "example_interfaces/action/Fibonacci",
            "request_schema": {
                "type": "object",
                "properties": {"order": {"type": "integer", "minimum": 1}},
                "required": ["order"],
                "additionalProperties": False,
            },
            "max_ttl_ms": 10000,
            "timeout_ms": timeout_ms,
            "resources": [resource],
        }
    )


def request(
    kind: str,
    session_id: UUID,
    *,
    command_id: UUID | None = None,
    order: int = 5,
    ttl_ms: int = 5000,
) -> CommandRequest:
    return CommandRequest.model_validate(
        {
            "version": "1.0",
            "type": "command.request",
            "message_id": uuid4(),
            "robot_id": "g1-dev",
            "session_id": session_id,
            "sent_at": NOW,
            "payload": {
                "command_id": command_id or uuid4(),
                "kind": kind,
                "issued_at": NOW,
                "ttl_ms": ttl_ms,
                "params": {"order": order},
            },
        }
    )


class ControlledExecutor:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self._calls_changed = asyncio.Condition()
        self.calls: list[UUID] = []
        self.cancelled: list[UUID] = []
        self.active = 0
        self.max_active = 0

    async def execute(
        self,
        command_id: UUID,
        command: CommandConfig,
        params: dict[str, Any],
    ) -> ExecutionResult:
        del command, params
        async with self._calls_changed:
            self.calls.append(command_id)
            self._calls_changed.notify_all()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            await self.release.wait()
            return ExecutionResult(succeeded=True, detail="done", result={"ok": True})
        finally:
            self.active -= 1

    async def cancel(self, command_id: UUID) -> None:
        self.cancelled.append(command_id)

    async def wait_for_calls(self, count: int) -> None:
        async with self._calls_changed:
            await self._calls_changed.wait_for(lambda: len(self.calls) >= count)


async def active_session() -> tuple[SessionController, UUID]:
    async def no_op_safe_stop(reason: str) -> None:
        del reason

    session = SessionController(lease_timeout=timedelta(seconds=2), safe_stop=no_op_safe_stop)
    session_id = uuid4()
    await session.connected(session_id)
    await session.acquire(session_id, operator_id="operator-7", at=NOW)
    return session, session_id


@pytest.mark.asyncio
async def test_coordinator_runs_disjoint_commands_and_rejects_overlap(
    tmp_path: Path,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        coordinator = CommandCoordinator(
            commands={
                "task.move": command_config("motion"),
                "camera.capture": command_config("camera"),
                "task.turn": command_config("motion"),
            },
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        move = request("task.move", session_id)
        camera = request("camera.capture", session_id)
        move_task = asyncio.create_task(coordinator.submit(move, at=NOW))
        camera_task = asyncio.create_task(coordinator.submit(camera, at=NOW))
        await executor.wait_for_calls(2)

        conflict = await coordinator.submit(request("task.turn", session_id), at=NOW)
        assert conflict.state is CommandState.REJECTED
        assert "resource conflict" in conflict.detail
        assert executor.max_active == 2

        executor.release.set()
        completed = await asyncio.gather(move_task, camera_task)
        assert all(record.state is CommandState.SUCCEEDED for record in completed)


@pytest.mark.asyncio
async def test_coordinator_replays_same_id_without_second_execution_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    command_id = uuid4()
    first_request = request("task.move", session_id, command_id=command_id)
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        task = asyncio.create_task(coordinator.submit(first_request, at=NOW))
        await executor.started.wait()
        in_flight_retry = await coordinator.submit(first_request, at=NOW)
        assert in_flight_retry.state is CommandState.RUNNING
        assert executor.calls == [command_id]

        executor.release.set()
        completed = await task
        replay = await coordinator.submit(first_request, at=NOW)
        assert replay == completed
        assert executor.calls == [command_id]

        changed = request("task.move", session_id, command_id=command_id, order=6)
        with pytest.raises(CommandIdConflict):
            await coordinator.submit(changed, at=NOW)


@pytest.mark.asyncio
async def test_coordinator_timeout_cancels_adapter_and_releases_resource(
    tmp_path: Path,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion", timeout_ms=20)},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        command = request("task.move", session_id)
        record = await coordinator.submit(command, at=NOW)

        assert record.state is CommandState.TIMED_OUT
        assert executor.cancelled == [command.payload.command_id]
        assert coordinator.active_commands == frozenset()


@pytest.mark.asyncio
async def test_safe_stop_preempts_and_terminalizes_active_commands(tmp_path: Path) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        coordinator = CommandCoordinator(
            commands={
                "task.move": command_config("motion"),
                "camera.capture": command_config("camera"),
            },
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        commands = [request("task.move", session_id), request("camera.capture", session_id)]
        tasks = [asyncio.create_task(coordinator.submit(item, at=NOW)) for item in commands]
        await executor.wait_for_calls(2)

        preempted = await coordinator.preempt_all(reason="lease expired", at=NOW)
        records = await asyncio.gather(*tasks)

        assert set(preempted) == {item.payload.command_id for item in commands}
        assert set(executor.cancelled) == set(preempted)
        assert all(record.state is CommandState.CANCELLED for record in records)
        assert coordinator.active_commands == frozenset()
