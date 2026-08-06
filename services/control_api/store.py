"""Thread-safe SQLite persistence for the control-plane API.

The store intentionally exposes small repository-style methods rather than
leaking SQL into route handlers.  Each method uses the same connection under
an RLock, so ``:memory:`` databases are useful in tests while file databases
can be reopened by another app instance.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping

from pydantic import BaseModel

from packages.control_core.models import CapabilitySnapshot, CommandReceipt, CommandRequest, RobotState
from packages.control_core.registry import MappingRecord


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=False)
        if "schema_" in payload and "schema" not in payload:
            payload["schema"] = payload.pop("schema_")
        return payload
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo else value.replace(tzinfo=timezone.utc).isoformat()
    return value


def _encode(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _decode(value: str) -> Any:
    return json.loads(value)


class SQLiteStore:
    """A small repository boundary that can be replaced by PostgreSQL later."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            path = Path(self.db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS robots (
                    robot_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capabilities (
                    robot_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS commands (
                    command_id TEXT PRIMARY KEY,
                    request_json TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mappings (
                    mapping_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_id TEXT,
                    probe_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (mapping_id, version)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audits (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assistant_drafts (
                    draft_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _robot_payload(robot: RobotState | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(robot))

    @staticmethod
    def _capability_payload(snapshot: CapabilitySnapshot | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(snapshot))

    @staticmethod
    def _request_payload(request: CommandRequest | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(request))

    @staticmethod
    def _receipt_payload(receipt: CommandReceipt | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(receipt))

    def upsert_robot(self, robot: RobotState | Mapping[str, Any]) -> None:
        payload = self._robot_payload(robot)
        robot_id = payload.get("robotId") or payload.get("robot_id")
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robotId is required")
        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO robots(robot_id, payload_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(robot_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (robot_id, _encode(payload), now),
            )
            self._connection.commit()

    def list_robots(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT payload_json FROM robots ORDER BY robot_id").fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    def get_robot(self, robot_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT payload_json FROM robots WHERE robot_id = ?", (robot_id,)).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def upsert_capabilities(self, snapshot: CapabilitySnapshot | Mapping[str, Any], robot_id: str | None = None) -> None:
        payload = self._capability_payload(snapshot)
        target = robot_id or payload.get("robotId") or payload.get("robot_id")
        if not isinstance(target, str) or not target:
            raise ValueError("robotId is required for capabilities")
        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO capabilities(robot_id, payload_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(robot_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (target, _encode(payload), now),
            )
            self._connection.commit()

    def get_capabilities(self, robot_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT payload_json FROM capabilities WHERE robot_id = ?", (robot_id,)).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def save_command(self, request: CommandRequest | Mapping[str, Any], receipt: CommandReceipt | Mapping[str, Any]) -> None:
        request_payload = self._request_payload(request)
        receipt_payload = self._receipt_payload(receipt)
        command_id = receipt_payload.get("commandId") or receipt_payload.get("command_id") or request_payload.get("commandId")
        status = receipt_payload.get("status")
        if hasattr(status, "value"):
            status = status.value
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("commandId is required")
        if not isinstance(status, str):
            status = str(status)
        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO commands(command_id, request_json, receipt_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(command_id) DO UPDATE SET
                    request_json=excluded.request_json,
                    receipt_json=excluded.receipt_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (command_id, _encode(request_payload), _encode(receipt_payload), status, now, now),
            )
            self._connection.commit()

    def get_command_record(self, command_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json, receipt_json FROM commands WHERE command_id = ?", (command_id,)
            ).fetchone()
        if row is None:
            return None
        return {"request": _decode(row["request_json"]), "receipt": _decode(row["receipt_json"])}

    def get_command(self, command_id: str) -> dict[str, Any] | None:
        record = self.get_command_record(command_id)
        return None if record is None else record["receipt"]

    def list_commands(self, *, limit: int = 100) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT receipt_json FROM commands ORDER BY created_at DESC LIMIT ?", (bounded,)
            ).fetchall()
        return [_decode(row["receipt_json"]) for row in rows]

    @staticmethod
    def _mapping_payload(record: MappingRecord) -> tuple[dict[str, Any], str, str | None, dict[str, Any] | None]:
        mapping_payload = dict(_jsonable(record.mapping))
        probe_payload = None if record.probe is None else dict(_jsonable(record.probe))
        return mapping_payload, record.status, record.target_id, probe_payload

    def save_mapping(self, record: MappingRecord) -> None:
        mapping_payload, status, target_id, probe_payload = self._mapping_payload(record)
        mapping_id = mapping_payload.get("mappingId")
        version = mapping_payload.get("version")
        if not isinstance(mapping_id, str) or not isinstance(version, str):
            raise ValueError("mapping identity is required")
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO mappings(mapping_id, version, payload_json, status, target_id, probe_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mapping_id, version) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    status=excluded.status,
                    target_id=excluded.target_id,
                    probe_json=excluded.probe_json,
                    updated_at=excluded.updated_at
                """,
                (mapping_id, version, _encode(mapping_payload), status, target_id, None if probe_payload is None else _encode(probe_payload), self._now()),
            )
            self._connection.commit()

    def list_mappings(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json, status, target_id, probe_json, updated_at FROM mappings ORDER BY updated_at, mapping_id, version"
            ).fetchall()
        return [self._mapping_row(row) for row in rows]

    def get_mapping(self, mapping_id: str, version: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json, status, target_id, probe_json, updated_at FROM mappings WHERE mapping_id = ? AND version = ?",
                (mapping_id, version),
            ).fetchone()
        return None if row is None else self._mapping_row(row)

    @staticmethod
    def _mapping_row(row: sqlite3.Row) -> dict[str, Any]:
        probe = None if row["probe_json"] is None else _decode(row["probe_json"])
        return {
            "mapping": _decode(row["payload_json"]),
            "status": row["status"],
            "targetId": row["target_id"],
            "probe": probe,
            "updatedAt": row["updated_at"],
        }

    def record_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(_jsonable(event))
        with self._lock:
            created = self._now()
            cursor = self._connection.execute(
                "INSERT INTO events(payload_json, created_at) VALUES (?, ?)", ( _encode(payload), created)
            )
            self._connection.commit()
            event_id = int(cursor.lastrowid)
        payload.setdefault("eventId", event_id)
        payload.setdefault("createdAt", created)
        return payload

    def list_events(self, *, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 2000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT event_id, payload_json, created_at FROM events WHERE event_id > ? ORDER BY event_id LIMIT ?",
                (int(after_id), bounded),
            ).fetchall()
        result = []
        for row in rows:
            payload = _decode(row["payload_json"])
            payload.setdefault("eventId", int(row["event_id"]))
            payload.setdefault("createdAt", row["created_at"])
            result.append(payload)
        return result

    def record_audit(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(_jsonable(event))
        with self._lock:
            created = self._now()
            cursor = self._connection.execute(
                "INSERT INTO audits(payload_json, created_at) VALUES (?, ?)", ( _encode(payload), created)
            )
            self._connection.commit()
            audit_id = int(cursor.lastrowid)
        payload.setdefault("auditId", audit_id)
        payload.setdefault("createdAt", created)
        return payload

    def list_audits(self, *, limit: int = 500) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 2000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT audit_id, payload_json, created_at FROM audits ORDER BY audit_id LIMIT ?", (bounded,)
            ).fetchall()
        result = []
        for row in rows:
            payload = _decode(row["payload_json"])
            payload.setdefault("auditId", int(row["audit_id"]))
            payload.setdefault("createdAt", row["created_at"])
            result.append(payload)
        return result

    def save_assistant_draft(self, draft_id: str, payload: Mapping[str, Any], *, status: str = "draft") -> None:
        if not isinstance(draft_id, str) or not draft_id:
            raise ValueError("draft_id is required")
        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO assistant_drafts(draft_id, payload_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(draft_id) DO UPDATE SET payload_json=excluded.payload_json, status=excluded.status, updated_at=excluded.updated_at
                """,
                (draft_id, _encode(payload), status, now, now),
            )
            self._connection.commit()

    def get_assistant_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json, status, created_at, updated_at FROM assistant_drafts WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "draftId": draft_id,
            "plan": _decode(row["payload_json"]),
            "status": row["status"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def set_assistant_draft_status(self, draft_id: str, status: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE assistant_drafts SET status = ?, updated_at = ? WHERE draft_id = ?",
                (status, self._now(), draft_id),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def claim_assistant_draft(self, draft_id: str) -> bool:
        """Atomically move a draft into confirmation in progress."""
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE assistant_drafts SET status = 'confirming', updated_at = ? WHERE draft_id = ? AND status = 'draft'",
                (self._now(), draft_id),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def reset_confirming_drafts(self) -> None:
        """Recover drafts left mid-confirmation by a process restart."""
        with self._lock:
            self._connection.execute(
                "UPDATE assistant_drafts SET status = 'draft', updated_at = ? WHERE status = 'confirming'",
                (self._now(),),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["SQLiteStore"]
