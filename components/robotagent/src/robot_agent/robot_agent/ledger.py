"""Durable SQLite command ledger and restart recovery."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any
from uuid import UUID

from robot_agent.motion_fault import MotionSafetyFault


class CommandState(str, Enum):
    RECEIVED = "RECEIVED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"
    ABORTED_AGENT_RESTART = "ABORTED_AGENT_RESTART"


class SafeStopAuditState(str, Enum):
    REQUESTED = "requested"
    CONFIRMED = "confirmed"
    FAILED = "failed"


TERMINAL_STATES = frozenset(
    {
        CommandState.SUCCEEDED,
        CommandState.FAILED,
        CommandState.CANCELLED,
        CommandState.REJECTED,
        CommandState.TIMED_OUT,
        CommandState.ABORTED_AGENT_RESTART,
    }
)

_TRANSITIONS: dict[CommandState, frozenset[CommandState]] = {
    CommandState.RECEIVED: frozenset(
        {
            CommandState.RECEIVED,
            CommandState.RUNNING,
            CommandState.REJECTED,
            CommandState.CANCELLED,
            CommandState.TIMED_OUT,
            CommandState.ABORTED_AGENT_RESTART,
        }
    ),
    CommandState.RUNNING: frozenset(
        {
            CommandState.RUNNING,
            CommandState.SUCCEEDED,
            CommandState.FAILED,
            CommandState.CANCELLED,
            CommandState.TIMED_OUT,
            CommandState.ABORTED_AGENT_RESTART,
        }
    ),
}


class LedgerError(RuntimeError):
    """Base error for ledger operations."""


class CommandIdConflict(LedgerError):
    """Raised when a command ID is reused for a different canonical payload."""


class CommandNotFound(LedgerError):
    """Raised when a requested command row does not exist."""


class InvalidTransition(LedgerError):
    """Raised when a command state transition violates the state graph."""


class LedgerMigrationError(LedgerError):
    """Raised when a database schema cannot be migrated safely."""


@dataclass(frozen=True)
class CommandRecord:
    command_id: UUID
    kind: str
    payload_hash: str
    payload: dict[str, Any]
    state: CommandState
    detail: str
    result: Any | None
    session_id: UUID
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class LedgerRegistration:
    is_new: bool
    record: CommandRecord


@dataclass(frozen=True)
class TransitionRecord:
    command_id: UUID
    from_state: CommandState | None
    to_state: CommandState
    detail: str
    at: datetime


@dataclass(frozen=True)
class SafeStopAuditRecord:
    epoch: int
    state: SafeStopAuditState
    reason: str
    detail: str
    requested_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MotionFaultAuditRecord:
    handling_id: int
    event_id: UUID
    robot_id: str
    localization_epoch_id: str
    producer_boot_id: UUID
    source_sequence: int
    control_generation: int
    source: str
    code: str
    reason: str
    detected_at: str
    downstream_boot_id: UUID | None
    probe_request_id: UUID | None
    received_monotonic_ns: int
    safe_stop_epoch: int | None
    command_id: UUID | None
    cancel_begin_monotonic_ns: int | None
    ros_cancel_call_monotonic_ns: int | None
    nav2_terminal_monotonic_ns: int | None
    outcome: str
    detail: str
    received_at: datetime
    updated_at: datetime


class CommandLedger:
    """Thread-safe durable ledger with an explicit schema migration boundary."""

    SCHEMA_VERSION = 3

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if str(path) != ":memory:":
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path),
            timeout=5.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure_connection()
        self._migrate()

    def _configure_connection(self) -> None:
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")

    def _migrate(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > self.SCHEMA_VERSION:
            raise LedgerMigrationError(
                f"ledger schema {version} is newer than supported {self.SCHEMA_VERSION}"
            )
        if version == 0:
            states = ", ".join(f"'{state.value}'" for state in CommandState)
            with self._connection:
                self._connection.execute(
                    f"""
                    CREATE TABLE commands (
                        command_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        state TEXT NOT NULL CHECK (state IN ({states})),
                        detail TEXT NOT NULL,
                        result_json TEXT,
                        session_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                self._connection.execute(
                    f"""
                    CREATE TABLE command_transitions (
                        transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        command_id TEXT NOT NULL REFERENCES commands(command_id),
                        from_state TEXT CHECK (from_state IS NULL OR from_state IN ({states})),
                        to_state TEXT NOT NULL CHECK (to_state IN ({states})),
                        detail TEXT NOT NULL,
                        at TEXT NOT NULL
                    )
                    """
                )
                self._connection.execute(
                    "CREATE INDEX command_transitions_command_idx "
                    "ON command_transitions(command_id, transition_id)"
                )
                self._create_safe_stop_events_table()
                self._create_motion_fault_events_table()
                self._connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            return
        if version == 1:
            with self._connection:
                self._create_safe_stop_events_table()
                self._create_motion_fault_events_table()
                self._connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            return
        if version == 2:
            with self._connection:
                self._create_motion_fault_events_table()
                self._connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
            return
        if version == self.SCHEMA_VERSION:
            return
        raise LedgerMigrationError(f"no migration path from ledger schema {version}")

    def _create_safe_stop_events_table(self) -> None:
        states = ", ".join(f"'{state.value}'" for state in SafeStopAuditState)
        self._connection.execute(
            f"""
            CREATE TABLE safe_stop_events (
                epoch INTEGER PRIMARY KEY CHECK (epoch > 0),
                state TEXT NOT NULL CHECK (state IN ({states})),
                reason TEXT NOT NULL,
                detail TEXT NOT NULL,
                requested_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def _create_motion_fault_events_table(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS motion_fault_events (
                handling_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                robot_id TEXT NOT NULL,
                localization_epoch_id TEXT NOT NULL,
                producer_boot_id TEXT NOT NULL,
                source_sequence INTEGER NOT NULL CHECK (source_sequence > 0),
                control_generation INTEGER NOT NULL CHECK (control_generation >= 0),
                source TEXT NOT NULL,
                code TEXT NOT NULL,
                reason TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                downstream_boot_id TEXT,
                probe_request_id TEXT,
                received_monotonic_ns INTEGER NOT NULL CHECK (received_monotonic_ns >= 0),
                safe_stop_epoch INTEGER REFERENCES safe_stop_events(epoch),
                command_id TEXT,
                cancel_begin_monotonic_ns INTEGER,
                ros_cancel_call_monotonic_ns INTEGER,
                nav2_terminal_monotonic_ns INTEGER,
                outcome TEXT NOT NULL,
                detail TEXT NOT NULL,
                received_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS motion_fault_event_idx "
            "ON motion_fault_events(event_id, handling_id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS motion_fault_producer_idx "
            "ON motion_fault_events(producer_boot_id, source_sequence, handling_id)"
        )

    def begin_motion_fault(
        self,
        *,
        fault: MotionSafetyFault,
        received_monotonic_ns: int,
        at: datetime,
    ) -> MotionFaultAuditRecord:
        """Append receipt before associating the safe-stop/cancel outcome."""

        if received_monotonic_ns < 0:
            raise LedgerError("motion-fault monotonic receipt must be non-negative")
        timestamp = _aware_iso(at)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO motion_fault_events (
                    event_id, robot_id, localization_epoch_id,
                    producer_boot_id, source_sequence, control_generation,
                    source, code, reason, detected_at, downstream_boot_id,
                    probe_request_id, received_monotonic_ns, safe_stop_epoch,
                    command_id, cancel_begin_monotonic_ns,
                    ros_cancel_call_monotonic_ns, nav2_terminal_monotonic_ns,
                    outcome, detail, received_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL,
                          NULL, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    str(fault.event_id),
                    fault.robot_id,
                    fault.localization_epoch_id,
                    str(fault.producer_boot_id),
                    fault.source_sequence,
                    fault.control_generation,
                    fault.source,
                    fault.code,
                    fault.reason[:512],
                    fault.detected_at,
                    None if fault.downstream_boot_id is None else str(fault.downstream_boot_id),
                    None if fault.probe_request_id is None else str(fault.probe_request_id),
                    received_monotonic_ns,
                    "received",
                    "fault accepted; authority revocation pending",
                    timestamp,
                    timestamp,
                ),
            )
            handling_id = cursor.lastrowid
            if handling_id is None:  # pragma: no cover - SQLite INSERT invariant.
                raise LedgerError("motion-fault insert did not return a handling ID")
            return self.motion_fault_event(handling_id)

    def finish_motion_fault(
        self,
        handling_id: int,
        *,
        safe_stop_epoch: int | None,
        command_id: UUID | None,
        cancel_begin_monotonic_ns: int | None,
        ros_cancel_call_monotonic_ns: int | None,
        nav2_terminal_monotonic_ns: int | None,
        outcome: str,
        detail: str,
        at: datetime,
    ) -> MotionFaultAuditRecord:
        timestamp = _aware_iso(at)
        with self._lock, self._connection:
            if self._connection.execute(
                "SELECT 1 FROM motion_fault_events WHERE handling_id = ?",
                (handling_id,),
            ).fetchone() is None:
                raise LedgerError(f"unknown motion-fault handling {handling_id}")
            self._connection.execute(
                """
                UPDATE motion_fault_events
                SET safe_stop_epoch = ?, command_id = ?,
                    cancel_begin_monotonic_ns = ?,
                    ros_cancel_call_monotonic_ns = ?,
                    nav2_terminal_monotonic_ns = ?, outcome = ?, detail = ?,
                    updated_at = ?
                WHERE handling_id = ?
                """,
                (
                    safe_stop_epoch,
                    None if command_id is None else str(command_id),
                    cancel_begin_monotonic_ns,
                    ros_cancel_call_monotonic_ns,
                    nav2_terminal_monotonic_ns,
                    outcome[:128],
                    detail[:1024],
                    timestamp,
                    handling_id,
                ),
            )
            return self.motion_fault_event(handling_id)

    def motion_fault_event(self, handling_id: int) -> MotionFaultAuditRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM motion_fault_events WHERE handling_id = ?",
                (handling_id,),
            ).fetchone()
            if row is None:
                raise LedgerError(f"unknown motion-fault handling {handling_id}")
            return _row_to_motion_fault_record(row)

    def motion_fault_events(self) -> list[MotionFaultAuditRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM motion_fault_events ORDER BY handling_id"
            ).fetchall()
        return [_row_to_motion_fault_record(row) for row in rows]

    def begin_safe_stop(self, *, epoch: int, reason: str, at: datetime) -> SafeStopAuditRecord:
        """Durably record authority revocation before physical stop dispatch."""

        if epoch <= 0:
            raise LedgerError("safe-stop epoch must be positive")
        timestamp = _aware_iso(at)
        bounded_reason = reason[:512]
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT * FROM safe_stop_events WHERE epoch = ?", (epoch,)
            ).fetchone()
            if existing is not None:
                record = _row_to_safe_stop_record(existing)
                if (
                    record.state is SafeStopAuditState.REQUESTED
                    and record.reason == bounded_reason
                ):
                    return record
                raise LedgerError(f"safe-stop epoch {epoch} is already recorded")
            latest = self._connection.execute(
                "SELECT COALESCE(MAX(epoch), 0) FROM safe_stop_events"
            ).fetchone()[0]
            if epoch <= int(latest):
                raise LedgerError(
                    f"safe-stop epoch must increase beyond durable epoch {latest}"
                )
            self._connection.execute(
                """
                INSERT INTO safe_stop_events (
                    epoch, state, reason, detail, requested_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    epoch,
                    SafeStopAuditState.REQUESTED.value,
                    bounded_reason,
                    "physical safe-stop requested",
                    timestamp,
                    timestamp,
                ),
            )
            return self.safe_stop_event(epoch)

    def finish_safe_stop(
        self,
        *,
        epoch: int,
        state: SafeStopAuditState,
        detail: str,
        at: datetime,
    ) -> SafeStopAuditRecord:
        if state not in {SafeStopAuditState.CONFIRMED, SafeStopAuditState.FAILED}:
            raise LedgerError("safe-stop final state must be confirmed or failed")
        timestamp = _aware_iso(at)
        with self._lock, self._connection:
            current = self.safe_stop_event(epoch)
            if current.state is not SafeStopAuditState.REQUESTED:
                if current.state is state and current.detail == detail[:1024]:
                    return current
                raise LedgerError(
                    f"safe-stop epoch {epoch} is already {current.state.value}"
                )
            self._connection.execute(
                """
                UPDATE safe_stop_events
                SET state = ?, detail = ?, updated_at = ?
                WHERE epoch = ?
                """,
                (state.value, detail[:1024], timestamp, epoch),
            )
            return self.safe_stop_event(epoch)

    def safe_stop_event(self, epoch: int) -> SafeStopAuditRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM safe_stop_events WHERE epoch = ?", (epoch,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"unknown safe-stop epoch {epoch}")
            return _row_to_safe_stop_record(row)

    def safe_stop_events(self) -> list[SafeStopAuditRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM safe_stop_events ORDER BY epoch"
            ).fetchall()
        return [_row_to_safe_stop_record(row) for row in rows]

    @property
    def latest_safe_stop_epoch(self) -> int:
        with self._lock:
            return int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(epoch), 0) FROM safe_stop_events"
                ).fetchone()[0]
            )

    def register(
        self,
        *,
        command_id: UUID,
        kind: str,
        payload: Mapping[str, Any],
        session_id: UUID,
        at: datetime,
    ) -> LedgerRegistration:
        """Persist RECEIVED before dispatch, or return the exact prior command."""
        timestamp = _aware_iso(at)
        payload_json = _canonical_json(payload)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM commands WHERE command_id = ?", (str(command_id),)
            ).fetchone()
            if row is not None:
                record = _row_to_record(row)
                if (
                    record.kind != kind
                    or record.payload_hash != payload_hash
                    or _canonical_json(record.payload) != payload_json
                ):
                    raise CommandIdConflict(
                        f"command_id {command_id} was reused with a different payload"
                    )
                return LedgerRegistration(is_new=False, record=record)

            self._connection.execute(
                """
                INSERT INTO commands (
                    command_id, kind, payload_hash, payload_json, state, detail,
                    result_json, session_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    str(command_id),
                    kind,
                    payload_hash,
                    payload_json,
                    CommandState.RECEIVED.value,
                    "command received",
                    str(session_id),
                    timestamp,
                    timestamp,
                ),
            )
            self._insert_transition(
                command_id=command_id,
                from_state=None,
                to_state=CommandState.RECEIVED,
                detail="command received",
                at=timestamp,
            )
            return LedgerRegistration(is_new=True, record=self.get(command_id))

    def transition(
        self,
        command_id: UUID,
        state: CommandState,
        *,
        detail: str,
        at: datetime,
        result: Any | None = None,
    ) -> CommandRecord:
        """Atomically append one valid state transition and update the command row."""
        timestamp = _aware_iso(at)
        result_json = None if result is None else _canonical_json(result)
        with self._lock, self._connection:
            current = self.get(command_id)
            allowed = _TRANSITIONS.get(current.state, frozenset())
            if state not in allowed:
                raise InvalidTransition(
                    f"invalid command transition {current.state.value} -> {state.value}"
                )
            self._connection.execute(
                """
                UPDATE commands
                SET state = ?, detail = ?, result_json = ?, updated_at = ?
                WHERE command_id = ?
                """,
                (state.value, detail, result_json, timestamp, str(command_id)),
            )
            self._insert_transition(
                command_id=command_id,
                from_state=current.state,
                to_state=state,
                detail=detail,
                at=timestamp,
            )
            return self.get(command_id)

    def get(self, command_id: UUID) -> CommandRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM commands WHERE command_id = ?", (str(command_id),)
            ).fetchone()
            if row is None:
                raise CommandNotFound(f"unknown command_id {command_id}")
            return _row_to_record(row)

    def transitions(self, command_id: UUID) -> list[TransitionRecord]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT command_id, from_state, to_state, detail, at
                FROM command_transitions
                WHERE command_id = ?
                ORDER BY transition_id
                """,
                (str(command_id),),
            ).fetchall()
        return [
            TransitionRecord(
                command_id=UUID(row["command_id"]),
                from_state=(None if row["from_state"] is None else CommandState(row["from_state"])),
                to_state=CommandState(row["to_state"]),
                detail=row["detail"],
                at=datetime.fromisoformat(row["at"]),
            )
            for row in rows
        ]

    def abort_incomplete_after_restart(self, *, at: datetime) -> list[CommandRecord]:
        """Durably terminalize every command that could have been in flight."""
        timestamp = _aware_iso(at)
        recovered: list[CommandRecord] = []
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT * FROM commands WHERE state IN (?, ?) ORDER BY created_at",
                (CommandState.RECEIVED.value, CommandState.RUNNING.value),
            ).fetchall()
            for row in rows:
                current = _row_to_record(row)
                detail = "agent restarted before command reached a terminal state"
                self._connection.execute(
                    """
                    UPDATE commands
                    SET state = ?, detail = ?, result_json = NULL, updated_at = ?
                    WHERE command_id = ?
                    """,
                    (
                        CommandState.ABORTED_AGENT_RESTART.value,
                        detail,
                        timestamp,
                        str(current.command_id),
                    ),
                )
                self._insert_transition(
                    command_id=current.command_id,
                    from_state=current.state,
                    to_state=CommandState.ABORTED_AGENT_RESTART,
                    detail=detail,
                    at=timestamp,
                )
                recovered.append(self.get(current.command_id))
        return recovered

    def prune_terminal_before(self, *, before: datetime) -> int:
        """Delete terminal commands older than the configured idempotency window."""
        timestamp = _aware_iso(before)
        terminal_values = tuple(state.value for state in TERMINAL_STATES)
        placeholders = ", ".join("?" for _ in terminal_values)
        params = (*terminal_values, timestamp)
        old_terminal_commands = (
            "SELECT command_id FROM commands "
            f"WHERE state IN ({placeholders}) "
            "AND julianday(updated_at) < julianday(?)"
        )
        with self._lock, self._connection:
            self._connection.execute(
                f"DELETE FROM command_transitions WHERE command_id IN ({old_terminal_commands})",
                params,
            )
            cursor = self._connection.execute(
                f"DELETE FROM commands WHERE command_id IN ({old_terminal_commands})",
                params,
            )
        return cursor.rowcount

    def _insert_transition(
        self,
        *,
        command_id: UUID,
        from_state: CommandState | None,
        to_state: CommandState,
        detail: str,
        at: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO command_transitions (command_id, from_state, to_state, detail, at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(command_id),
                None if from_state is None else from_state.value,
                to_state.value,
                detail,
                at,
            ),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> CommandLedger:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"value is not canonical JSON: {exc}") from exc


def _aware_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LedgerError("ledger timestamps must include a timezone")
    return value.isoformat()


def _row_to_record(row: sqlite3.Row) -> CommandRecord:
    return CommandRecord(
        command_id=UUID(row["command_id"]),
        kind=row["kind"],
        payload_hash=row["payload_hash"],
        payload=json.loads(row["payload_json"]),
        state=CommandState(row["state"]),
        detail=row["detail"],
        result=None if row["result_json"] is None else json.loads(row["result_json"]),
        session_id=UUID(row["session_id"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_safe_stop_record(row: sqlite3.Row) -> SafeStopAuditRecord:
    return SafeStopAuditRecord(
        epoch=int(row["epoch"]),
        state=SafeStopAuditState(row["state"]),
        reason=row["reason"],
        detail=row["detail"],
        requested_at=datetime.fromisoformat(row["requested_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_motion_fault_record(row: sqlite3.Row) -> MotionFaultAuditRecord:
    return MotionFaultAuditRecord(
        handling_id=int(row["handling_id"]),
        event_id=UUID(row["event_id"]),
        robot_id=row["robot_id"],
        localization_epoch_id=row["localization_epoch_id"],
        producer_boot_id=UUID(row["producer_boot_id"]),
        source_sequence=int(row["source_sequence"]),
        control_generation=int(row["control_generation"]),
        source=row["source"],
        code=row["code"],
        reason=row["reason"],
        detected_at=row["detected_at"],
        downstream_boot_id=(
            None
            if row["downstream_boot_id"] is None
            else UUID(row["downstream_boot_id"])
        ),
        probe_request_id=(
            None
            if row["probe_request_id"] is None
            else UUID(row["probe_request_id"])
        ),
        received_monotonic_ns=int(row["received_monotonic_ns"]),
        safe_stop_epoch=(
            None if row["safe_stop_epoch"] is None else int(row["safe_stop_epoch"])
        ),
        command_id=(None if row["command_id"] is None else UUID(row["command_id"])),
        cancel_begin_monotonic_ns=(
            None
            if row["cancel_begin_monotonic_ns"] is None
            else int(row["cancel_begin_monotonic_ns"])
        ),
        ros_cancel_call_monotonic_ns=(
            None
            if row["ros_cancel_call_monotonic_ns"] is None
            else int(row["ros_cancel_call_monotonic_ns"])
        ),
        nav2_terminal_monotonic_ns=(
            None
            if row["nav2_terminal_monotonic_ns"] is None
            else int(row["nav2_terminal_monotonic_ns"])
        ),
        outcome=row["outcome"],
        detail=row["detail"],
        received_at=datetime.fromisoformat(row["received_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )
