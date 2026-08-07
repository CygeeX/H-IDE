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


class CommandState(str, Enum):
    RECEIVED = "RECEIVED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TIMED_OUT = "TIMED_OUT"
    ABORTED_AGENT_RESTART = "ABORTED_AGENT_RESTART"


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


class CommandLedger:
    """Thread-safe durable ledger with an explicit schema migration boundary."""

    SCHEMA_VERSION = 1

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
        if version == self.SCHEMA_VERSION:
            return
        if version != 0:
            raise LedgerMigrationError(f"no migration path from ledger schema {version}")
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
            self._connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")

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
