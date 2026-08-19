from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from robot_agent.arbitration import ResourceArbiter
from robot_agent.config import CommandConfig
from robot_agent.coordinator import (
    CommandCancellationError,
    CommandCoordinator,
    ExecutionResult,
)
from robot_agent.ledger import CommandIdConflict, CommandLedger, CommandState
from robot_agent.protocol import CommandRequest
from robot_agent.session import SessionController

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
        self.cancel_active_counts: list[int] = []
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
        self.cancel_active_counts.append(self.active)

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
async def test_concurrent_same_id_registration_cannot_release_running_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    command_id = uuid4()
    same_request = request("task.move", session_id, command_id=command_id)
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        original_register = ledger.register
        first_register_started = threading.Event()
        call_lock = threading.Lock()
        calls = 0

        def reordered_register(**kwargs: Any) -> Any:
            nonlocal calls
            with call_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_register_started.set()
                time.sleep(0.2)
            return original_register(**kwargs)

        monkeypatch.setattr(ledger, "register", reordered_register)
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        primary = asyncio.create_task(coordinator.submit(same_request, at=NOW))
        assert await asyncio.to_thread(first_register_started.wait, 0.5)
        retry = asyncio.create_task(coordinator.submit(same_request, at=NOW))
        await asyncio.wait_for(executor.started.wait(), timeout=1)
        done, _ = await asyncio.wait({primary, retry}, return_when=asyncio.FIRST_COMPLETED)

        assert len(done) == 1
        assert executor.active == 1
        assert coordinator.active_commands == frozenset({command_id})

        cancelled = await coordinator.cancel(
            command_id,
            reason="cancel concurrent owner",
            at=NOW,
        )
        records = await asyncio.gather(primary, retry)
        assert cancelled is True
        assert records[0].state is CommandState.CANCELLED
        assert executor.calls == [command_id]
        assert records[1].state in {
            CommandState.RECEIVED,
            CommandState.RUNNING,
            CommandState.CANCELLED,
        }
        assert coordinator.active_commands == frozenset()
        assert executor.active == 0


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
        assert executor.cancel_active_counts == [1]
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


@pytest.mark.asyncio
async def test_cancel_one_leaves_disjoint_command_running(tmp_path: Path) -> None:
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
        move = request("task.move", session_id)
        camera = request("camera.capture", session_id)
        move_task = asyncio.create_task(coordinator.submit(move, at=NOW))
        camera_task = asyncio.create_task(coordinator.submit(camera, at=NOW))
        await executor.wait_for_calls(2)

        cancelled = await coordinator.cancel(
            move.payload.command_id,
            reason="operator request",
            at=NOW,
        )
        move_record = await move_task
        assert cancelled is True
        assert move_record.state is CommandState.CANCELLED
        assert coordinator.active_commands == frozenset({camera.payload.command_id})

        executor.release.set()
        assert (await camera_task).state is CommandState.SUCCEEDED


@pytest.mark.asyncio
async def test_coordinator_policy_rejections_and_adapter_failures(tmp_path: Path) -> None:
    session, session_id = await active_session()

    class EdgeExecutor:
        async def execute(
            self,
            command_id: UUID,
            command: CommandConfig,
            params: dict[str, Any],
        ) -> ExecutionResult:
            del command_id, command
            if params["order"] == 2:
                return ExecutionResult(succeeded=False, detail="robot rejected")
            raise RuntimeError("adapter exploded")

        async def cancel(self, command_id: UUID) -> None:
            del command_id

    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=EdgeExecutor(),
        )
        unknown = await coordinator.submit(request("unknown.command", session_id), at=NOW)
        too_long = await coordinator.submit(request("task.move", session_id, ttl_ms=11000), at=NOW)
        expired = await coordinator.submit(
            request("task.move", session_id, ttl_ms=5000),
            at=NOW + timedelta(seconds=6),
        )
        invalid = await coordinator.submit(request("task.move", session_id, order=0), at=NOW)
        rejected = await coordinator.submit(request("task.move", session_id, order=2), at=NOW)
        failed = await coordinator.submit(request("task.move", session_id, order=3), at=NOW)

        assert all(
            record.state is CommandState.REJECTED
            for record in (unknown, too_long, expired, invalid)
        )
        assert rejected.state is CommandState.FAILED
        assert rejected.detail == "robot rejected"
        assert failed.state is CommandState.FAILED
        assert "adapter exploded" in failed.detail
        assert await coordinator.cancel(uuid4(), reason="missing", at=NOW) is False


@pytest.mark.asyncio
async def test_slow_ledger_write_does_not_block_lease_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        original_register = ledger.register

        def slow_register(**kwargs: Any) -> Any:
            time.sleep(0.2)
            return original_register(**kwargs)

        monkeypatch.setattr(ledger, "register", slow_register)
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        command_task = asyncio.create_task(
            coordinator.submit(request("task.move", session_id), at=NOW)
        )
        started = asyncio.get_running_loop().time()
        await asyncio.sleep(0.02)
        scheduling_delay = asyncio.get_running_loop().time() - started

        assert scheduling_delay < 0.1
        await executor.started.wait()
        executor.release.set()
        assert (await command_task).state is CommandState.SUCCEEDED


@pytest.mark.asyncio
async def test_command_expiring_during_slow_ledger_register_is_never_dispatched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        original_register = ledger.register

        def slow_register(**kwargs: Any) -> Any:
            time.sleep(0.25)
            return original_register(**kwargs)

        monkeypatch.setattr(ledger, "register", slow_register)
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )

        record = await coordinator.submit(
            request("task.move", session_id, ttl_ms=100),
            at=NOW,
        )

        assert record.state is CommandState.REJECTED
        assert "expired before dispatch" in record.detail
        assert executor.calls == []


@pytest.mark.asyncio
async def test_cancel_during_initial_ledger_write_prevents_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        original_register = ledger.register
        register_started = threading.Event()

        def slow_register(**kwargs: Any) -> Any:
            register_started.set()
            time.sleep(0.2)
            return original_register(**kwargs)

        monkeypatch.setattr(ledger, "register", slow_register)
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        command = request("task.move", session_id)
        submission = asyncio.create_task(coordinator.submit(command, at=NOW))
        assert await asyncio.to_thread(register_started.wait, 0.5)

        cancelled = await coordinator.cancel(
            command.payload.command_id,
            reason="operator cancelled before dispatch",
            at=NOW,
        )
        record = await submission

        assert cancelled is True
        assert record.state is CommandState.CANCELLED
        assert executor.calls == []


@pytest.mark.asyncio
async def test_safe_stop_preempts_submission_during_initial_ledger_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, session_id = await active_session()
    executor = ControlledExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        original_register = ledger.register
        register_started = threading.Event()

        def slow_register(**kwargs: Any) -> Any:
            register_started.set()
            time.sleep(0.2)
            return original_register(**kwargs)

        monkeypatch.setattr(ledger, "register", slow_register)
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        command = request("task.move", session_id)
        submission = asyncio.create_task(coordinator.submit(command, at=NOW))
        assert await asyncio.to_thread(register_started.wait, 0.5)

        preempted = await coordinator.preempt_all(reason="connection lost", at=NOW)
        record = await submission

        assert preempted == (command.payload.command_id,)
        assert record.state is CommandState.CANCELLED
        assert executor.calls == []


@pytest.mark.asyncio
async def test_cancel_fails_closed_if_execution_coroutine_ignores_cancellation(
    tmp_path: Path,
) -> None:
    session, session_id = await active_session()

    class UncooperativeExecutor:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.stopped = asyncio.Event()

        async def execute(
            self,
            command_id: UUID,
            command: CommandConfig,
            params: dict[str, Any],
        ) -> ExecutionResult:
            del command_id, command, params
            self.started.set()
            try:
                while not self.release.is_set():
                    try:
                        await asyncio.sleep(0.01)
                    except asyncio.CancelledError:
                        continue
                return ExecutionResult(succeeded=True, detail="released")
            finally:
                self.stopped.set()

        async def cancel(self, command_id: UUID) -> None:
            del command_id

    executor = UncooperativeExecutor()
    with CommandLedger(tmp_path / "ledger.sqlite3") as ledger:
        coordinator = CommandCoordinator(
            commands={"task.move": command_config("motion")},
            ledger=ledger,
            arbiter=ResourceArbiter(),
            session=session,
            executor=executor,
        )
        command = request("task.move", session_id)
        submission = asyncio.create_task(coordinator.submit(command, at=NOW))
        await executor.started.wait()
        cancellation = asyncio.create_task(
            coordinator.cancel(
                command.payload.command_id,
                reason="operator cancellation",
                at=NOW,
            )
        )

        try:
            done, _ = await asyncio.wait({cancellation}, timeout=0.75)
            assert cancellation in done
            with pytest.raises(CommandCancellationError):
                await cancellation
            record = await submission
            assert record.state is CommandState.FAILED
            assert "CANCEL_NOT_ACKNOWLEDGED" in record.detail
            assert "not acknowledged" in record.detail
        finally:
            executor.release.set()
            await asyncio.wait_for(executor.stopped.wait(), timeout=1)
