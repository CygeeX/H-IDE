from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from robot_agent.ledger import (
    CommandIdConflict,
    CommandLedger,
    CommandNotFound,
    CommandState,
    InvalidTransition,
    LedgerError,
    SafeStopAuditState,
)
from robot_agent.motion_fault import MotionSafetyFault

NOW = datetime(2026, 8, 7, 13, 0, tzinfo=timezone.utc)


def test_ledger_persists_transitions_and_same_payload_replay(tmp_path: Path) -> None:
    path = tmp_path / "commands.sqlite3"
    command_id = uuid4()
    payload = {
        "kind": "task.execute",
        "issued_at": NOW.isoformat(),
        "ttl_ms": 5000,
        "params": {"order": 5},
    }

    with CommandLedger(path) as ledger:
        registration = ledger.register(
            command_id=command_id,
            kind="task.execute",
            payload=payload,
            session_id=uuid4(),
            at=NOW,
        )
        assert registration.is_new is True
        assert registration.record.state is CommandState.RECEIVED

        running = ledger.transition(
            command_id,
            CommandState.RUNNING,
            detail="goal accepted",
            at=NOW + timedelta(milliseconds=10),
        )
        completed = ledger.transition(
            command_id,
            CommandState.SUCCEEDED,
            detail="done",
            result={"sequence": [0, 1, 1, 2, 3]},
            at=NOW + timedelta(seconds=1),
        )

        assert running.state is CommandState.RUNNING
        assert completed.result == {"sequence": [0, 1, 1, 2, 3]}

    with CommandLedger(path) as reopened:
        replay = reopened.register(
            command_id=command_id,
            kind="task.execute",
            payload=payload,
            session_id=uuid4(),
            at=NOW + timedelta(seconds=2),
        )
        assert replay.is_new is False
        assert replay.record.state is CommandState.SUCCEEDED
        assert len(reopened.transitions(command_id)) == 3


def test_ledger_rejects_same_id_with_different_payload(tmp_path: Path) -> None:
    command_id = uuid4()
    with CommandLedger(tmp_path / "commands.sqlite3") as ledger:
        ledger.register(
            command_id=command_id,
            kind="task.execute",
            payload={"kind": "task.execute", "params": {"order": 5}},
            session_id=uuid4(),
            at=NOW,
        )

        with pytest.raises(CommandIdConflict, match="different payload"):
            ledger.register(
                command_id=command_id,
                kind="task.execute",
                payload={"kind": "task.execute", "params": {"order": 6}},
                session_id=uuid4(),
                at=NOW,
            )


def test_restart_recovery_aborts_every_non_terminal_command(tmp_path: Path) -> None:
    path = tmp_path / "commands.sqlite3"
    received_id = uuid4()
    running_id = uuid4()
    succeeded_id = uuid4()
    with CommandLedger(path) as ledger:
        for command_id in (received_id, running_id, succeeded_id):
            ledger.register(
                command_id=command_id,
                kind="task.execute",
                payload={"kind": "task.execute", "params": {"order": 5}},
                session_id=uuid4(),
                at=NOW,
            )
        ledger.transition(running_id, CommandState.RUNNING, detail="running", at=NOW)
        ledger.transition(succeeded_id, CommandState.RUNNING, detail="running", at=NOW)
        ledger.transition(succeeded_id, CommandState.SUCCEEDED, detail="done", at=NOW)

    with CommandLedger(path) as reopened:
        recovered = reopened.abort_incomplete_after_restart(at=NOW + timedelta(seconds=1))

        assert {record.command_id for record in recovered} == {received_id, running_id}
        assert all(record.state is CommandState.ABORTED_AGENT_RESTART for record in recovered)
        assert reopened.get(succeeded_id).state is CommandState.SUCCEEDED
        assert reopened.abort_incomplete_after_restart(at=NOW + timedelta(seconds=2)) == []


def test_ledger_enforces_transition_graph_and_durable_pragmas(tmp_path: Path) -> None:
    path = tmp_path / "commands.sqlite3"
    command_id = uuid4()
    with CommandLedger(path) as ledger:
        ledger.register(
            command_id=command_id,
            kind="power.enable",
            payload={"kind": "power.enable", "params": {"data": True}},
            session_id=uuid4(),
            at=NOW,
        )
        with pytest.raises(InvalidTransition, match=r"RECEIVED.*SUCCEEDED"):
            ledger.transition(command_id, CommandState.SUCCEEDED, detail="skip", at=NOW)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_ledger_prunes_only_terminal_commands_older_than_retention_window(
    tmp_path: Path,
) -> None:
    old_terminal = uuid4()
    old_incomplete = uuid4()
    recent_terminal = uuid4()
    with CommandLedger(tmp_path / "commands.sqlite3") as ledger:
        for command_id, created_at in (
            (old_terminal, NOW - timedelta(days=40)),
            (old_incomplete, NOW - timedelta(days=40)),
            (recent_terminal, NOW - timedelta(days=2)),
        ):
            ledger.register(
                command_id=command_id,
                kind="task.execute",
                payload={"kind": "task.execute", "params": {"order": 5}},
                session_id=uuid4(),
                at=created_at,
            )
        for command_id, completed_at in (
            (old_terminal, NOW - timedelta(days=39)),
            (recent_terminal, NOW - timedelta(days=1)),
        ):
            ledger.transition(command_id, CommandState.RUNNING, detail="running", at=completed_at)
            ledger.transition(command_id, CommandState.SUCCEEDED, detail="done", at=completed_at)

        deleted = ledger.prune_terminal_before(before=NOW - timedelta(days=30))

        assert deleted == 1
        with pytest.raises(CommandNotFound):
            ledger.get(old_terminal)
        assert ledger.get(old_incomplete).state is CommandState.RECEIVED
        assert ledger.get(recent_terminal).state is CommandState.SUCCEEDED


def test_safe_stop_epochs_are_durable_monotonic_and_auditable(tmp_path: Path) -> None:
    path = tmp_path / "commands.sqlite3"
    with CommandLedger(path) as ledger:
        requested = ledger.begin_safe_stop(epoch=1, reason="agent startup", at=NOW)
        assert requested.state is SafeStopAuditState.REQUESTED
        confirmed = ledger.finish_safe_stop(
            epoch=1,
            state=SafeStopAuditState.CONFIRMED,
            detail="plant latched",
            at=NOW + timedelta(milliseconds=10),
        )
        assert confirmed.state is SafeStopAuditState.CONFIRMED
        ledger.begin_safe_stop(epoch=2, reason="lease expired", at=NOW)
        failed = ledger.finish_safe_stop(
            epoch=2,
            state=SafeStopAuditState.FAILED,
            detail="service unavailable",
            at=NOW + timedelta(milliseconds=20),
        )
        assert failed.state is SafeStopAuditState.FAILED
        with pytest.raises(LedgerError, match="already recorded"):
            ledger.begin_safe_stop(epoch=1, reason="reused", at=NOW)

    with CommandLedger(path) as reopened:
        assert reopened.latest_safe_stop_epoch == 2
        assert [event.state for event in reopened.safe_stop_events()] == [
            SafeStopAuditState.CONFIRMED,
            SafeStopAuditState.FAILED,
        ]


def test_v1_ledger_migrates_without_losing_command_history(tmp_path: Path) -> None:
    path = tmp_path / "commands.sqlite3"
    command_id = uuid4()
    with CommandLedger(path) as ledger:
        ledger.register(
            command_id=command_id,
            kind="task.execute",
            payload={"kind": "task.execute", "params": {"order": 5}},
            session_id=uuid4(),
            at=NOW,
        )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE safe_stop_events")
        connection.execute("PRAGMA user_version=1")

    with CommandLedger(path) as migrated:
        assert migrated.get(command_id).state is CommandState.RECEIVED
        assert migrated.latest_safe_stop_epoch == 0
        migrated.begin_safe_stop(epoch=1, reason="migration check", at=NOW)
        assert migrated.safe_stop_event(1).state is SafeStopAuditState.REQUESTED


def test_motion_fault_is_correlated_with_safe_stop_epoch_and_exact_command(
    tmp_path: Path,
) -> None:
    event_id = uuid4()
    producer_boot_id = uuid4()
    command_id = uuid4()
    fault = MotionSafetyFault.from_json(
        json.dumps(
            {
                "schemaVersion": "h-ide.motion-safety-fault/v1",
                "eventId": str(event_id),
                "robotId": "g1-sim",
                "localizationEpochId": "g1-sim-map-epoch-1",
                "producerBootId": str(producer_boot_id),
                "sourceSequence": 9,
                "controlGeneration": 4,
                "source": "sonic_driver",
                "code": "LOWSTATE_STALE",
                "reason": "low-state stream became stale",
                "latched": True,
                "detectedAt": NOW.isoformat(),
                "downstreamBootId": None,
                "probeRequestId": None,
            }
        )
    )
    path = tmp_path / "commands.sqlite3"
    with CommandLedger(path) as ledger:
        receipt = ledger.begin_motion_fault(
            fault=fault,
            received_monotonic_ns=1_000_000,
            at=NOW,
        )
        assert receipt.outcome == "received"
        ledger.begin_safe_stop(epoch=1, reason="fault", at=NOW)
        correlated = ledger.finish_motion_fault(
            receipt.handling_id,
            safe_stop_epoch=1,
            command_id=command_id,
            cancel_begin_monotonic_ns=1_010_000,
            ros_cancel_call_monotonic_ns=1_020_000,
            nav2_terminal_monotonic_ns=1_200_000,
            outcome="nav2_terminal",
            detail="exact goal cancelled",
            at=NOW + timedelta(milliseconds=1),
        )
        assert correlated.event_id == event_id
        assert correlated.source == "sonic_driver"
        assert correlated.code == "LOWSTATE_STALE"
        assert correlated.control_generation == 4
        assert correlated.safe_stop_epoch == 1
        assert correlated.command_id == command_id

    # Durable transient-local replay handling is intentionally append-only:
    # a fresh Robot Agent process must latch the same retained event again.
    with CommandLedger(path) as reopened:
        replay = reopened.begin_motion_fault(
            fault=fault,
            received_monotonic_ns=2_000_000,
            at=NOW + timedelta(seconds=1),
        )
        assert replay.handling_id != correlated.handling_id
        assert len(reopened.motion_fault_events()) == 2


def test_v2_ledger_migrates_motion_fault_audit_table(tmp_path: Path) -> None:
    path = tmp_path / "commands.sqlite3"
    with CommandLedger(path):
        pass
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE motion_fault_events")
        connection.execute("PRAGMA user_version=2")

    with CommandLedger(path) as migrated:
        assert migrated.motion_fault_events() == []
