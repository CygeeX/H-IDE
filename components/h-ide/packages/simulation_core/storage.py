"""Small SQLite repository for simulation sessions and event history."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from .models import SimulationEvent, SimulationSession, SimulationSnapshot


def _encode(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _decode(value: str) -> Any:
    return json.loads(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SimulationStore:
    """Durable simulation repository.

    It intentionally has its own connection so the simulation layer can be
    enabled without changing the existing control-plane SQL repository.  Both
    repositories may point at the same SQLite file; transactions are short and
    guarded by a busy timeout.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS simulation_sessions (
                    session_id TEXT PRIMARY KEY,
                    session_json TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS simulation_events (
                    session_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence),
                    FOREIGN KEY(session_id) REFERENCES simulation_sessions(session_id)
                );
                CREATE INDEX IF NOT EXISTS simulation_events_created_idx
                    ON simulation_events(session_id, sequence);
                """
            )
            self._connection.commit()

    def save_session(self, session: SimulationSession, snapshot: SimulationSnapshot) -> None:
        now = _now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO simulation_sessions(session_id, session_json, snapshot_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    session_json=excluded.session_json,
                    snapshot_json=excluded.snapshot_json,
                    updated_at=excluded.updated_at
                """,
                (
                    session.sessionId,
                    _encode(session),
                    _encode(snapshot),
                    now,
                ),
            )
            self._connection.commit()

    def get_session(self, session_id: str) -> tuple[SimulationSession, SimulationSnapshot] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT session_json, snapshot_json FROM simulation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return (
            SimulationSession.model_validate(_decode(row["session_json"])),
            SimulationSnapshot.model_validate(_decode(row["snapshot_json"])),
        )

    def list_sessions(self, *, limit: int = 100) -> list[tuple[SimulationSession, SimulationSnapshot]]:
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._connection.execute(
                "SELECT session_json, snapshot_json FROM simulation_sessions "
                "ORDER BY updated_at DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [
            (
                SimulationSession.model_validate(_decode(row["session_json"])),
                SimulationSnapshot.model_validate(_decode(row["snapshot_json"])),
            )
            for row in rows
        ]

    def append_event(self, event: SimulationEvent) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO simulation_events(session_id, sequence, event_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    event.sessionId,
                    event.sequence,
                    _encode(event),
                    event.occurredAt.isoformat(),
                ),
            )
            self._connection.commit()

    def list_events(self, session_id: str, *, after: int = 0, limit: int = 500) -> list[SimulationEvent]:
        bounded = max(1, min(int(limit), 2_000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT event_json FROM simulation_events "
                "WHERE session_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (session_id, max(0, int(after)), bounded),
            ).fetchall()
        return [SimulationEvent.model_validate(_decode(row["event_json"])) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["SimulationStore"]
