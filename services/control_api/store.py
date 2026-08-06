"""Thread-safe SQLite persistence for the control-plane API.

The store intentionally exposes small repository-style methods rather than
leaking SQL into route handlers.  Each method uses the same connection under
an RLock, so ``:memory:`` databases are useful in tests while file databases
can be reopened by another app instance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Mapping

from pydantic import BaseModel

from packages.control_core.models import CapabilitySnapshot, CommandReceipt, CommandRequest, RobotState
from packages.control_core.registry import MappingRecord


@dataclass(frozen=True)
class CommandClaim:
    """Result of the durable command idempotency claim.

    ``owned`` is true only for the request that inserted the durable row.
    A caller seeing ``conflict`` supplied a command id already owned by a
    different idempotency key; neither case is allowed to dispatch again.
    """

    owned: bool
    conflict: bool
    record: dict[str, Any] | None


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
                    idempotency_key TEXT,
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
                    execution_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mapping_active (
                    command_key TEXT PRIMARY KEY,
                    mapping_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            # Existing installations predate the durable idempotency and
            # assistant execution columns.  Migrate in place rather than
            # requiring operators to delete a command database.
            self._ensure_column("commands", "idempotency_key", "TEXT")
            self._ensure_column("assistant_drafts", "execution_json", "TEXT NOT NULL DEFAULT '{}'" )
            self._ensure_column("assistant_drafts", "error", "TEXT")
            self._backfill_idempotency_keys()
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS commands_idempotency_key_uq "
                "ON commands(idempotency_key) WHERE idempotency_key IS NOT NULL"
            )
            self._backfill_mapping_pointers()
            self._connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _backfill_idempotency_keys(self) -> None:
        rows = self._connection.execute(
            "SELECT command_id, request_json, idempotency_key FROM commands ORDER BY created_at, command_id"
        ).fetchall()
        seen: set[str] = set()
        for row in rows:
            key = row["idempotency_key"]
            if not isinstance(key, str) or not key:
                try:
                    request = _decode(row["request_json"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    request = {}
                key = request.get("idempotencyKey") or request.get("idempotency_key")
            if not isinstance(key, str) or not key:
                key = f"legacy:{row['command_id']}"
            # A legacy database may contain duplicate keys because no unique
            # constraint existed.  Preserve every command while making the
            # first row's key authoritative for future retries.
            candidate = key
            if candidate in seen:
                candidate = f"{key}:legacy:{row['command_id']}"
                suffix = 1
                while candidate in seen:
                    suffix += 1
                    candidate = f"{key}:legacy:{row['command_id']}:{suffix}"
            seen.add(candidate)
            if row["idempotency_key"] != candidate:
                self._connection.execute(
                    "UPDATE commands SET idempotency_key = ? WHERE command_id = ?",
                    (candidate, row["command_id"]),
                )

    def _backfill_mapping_pointers(self) -> None:
        """Infer active pointers once for databases created before the table."""
        existing = self._connection.execute("SELECT COUNT(*) AS n FROM mapping_active").fetchone()["n"]
        if existing:
            return
        rows = self._connection.execute(
            "SELECT payload_json, mapping_id, version, updated_at FROM mappings "
            "WHERE status = 'active' ORDER BY updated_at, mapping_id, version"
        ).fetchall()
        for row in rows:
            try:
                payload = _decode(row["payload_json"])
                command_key = payload.get("commandKey") or payload.get("command_key")
            except (TypeError, ValueError, json.JSONDecodeError):
                command_key = None
            if isinstance(command_key, str) and command_key:
                self._connection.execute(
                    "INSERT INTO mapping_active(command_key, mapping_id, version, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(command_key) DO UPDATE SET "
                    "mapping_id=excluded.mapping_id, version=excluded.version, updated_at=excluded.updated_at",
                    (command_key, row["mapping_id"], row["version"], row["updated_at"]),
                )

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

    def _command_identity(self, request_payload: Mapping[str, Any]) -> tuple[str, str]:
        command_id = request_payload.get("commandId") or request_payload.get("command_id")
        idempotency_key = request_payload.get("idempotencyKey") or request_payload.get("idempotency_key")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("commandId is required")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotencyKey is required")
        return command_id, idempotency_key

    @staticmethod
    def _command_record_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "request": _decode(row["request_json"]),
            "receipt": _decode(row["receipt_json"]),
            "idempotencyKey": row["idempotency_key"],
        }

    def claim_command(
        self,
        request: CommandRequest | Mapping[str, Any],
        receipt: CommandReceipt | Mapping[str, Any] | None = None,
    ) -> CommandClaim:
        """Atomically claim a command idempotency key before adapter dispatch.

        The placeholder receipt makes a command visible to concurrent callers
        while the owner is talking to the robot.  SQLite's write transaction
        and unique index make this safe across threads and process restarts.
        """
        request_payload = self._request_payload(request)
        command_id, idempotency_key = self._command_identity(request_payload)
        if receipt is None:
            receipt_payload = {
                "commandId": command_id,
                "robotId": request_payload.get("robotId") or request_payload.get("robot_id"),
                "status": "accepted",
                "acceptedAt": self._now(),
                "startedAt": None,
                "completedAt": None,
                "result": None,
                "error": None,
            }
        else:
            receipt_payload = self._receipt_payload(receipt)
        status = receipt_payload.get("status")
        if hasattr(status, "value"):
            status = status.value
        if not isinstance(status, str) or not status:
            status = "accepted"
        receipt_payload["status"] = status
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                by_id = self._connection.execute(
                    "SELECT command_id, idempotency_key, request_json, receipt_json FROM commands WHERE command_id = ?",
                    (command_id,),
                ).fetchone()
                if by_id is not None:
                    self._connection.commit()
                    record = self._command_record_from_row(by_id)
                    return CommandClaim(False, record.get("idempotencyKey") != idempotency_key, record)
                by_key = self._connection.execute(
                    "SELECT command_id, idempotency_key, request_json, receipt_json FROM commands WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if by_key is not None:
                    self._connection.commit()
                    return CommandClaim(False, False, self._command_record_from_row(by_key))
                self._connection.execute(
                    "INSERT INTO commands(command_id, idempotency_key, request_json, receipt_json, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (command_id, idempotency_key, _encode(request_payload), _encode(receipt_payload), status, now, now),
                )
                self._connection.commit()
                return CommandClaim(True, False, {"request": request_payload, "receipt": receipt_payload, "idempotencyKey": idempotency_key})
            except sqlite3.IntegrityError:
                self._connection.rollback()
                row = self._connection.execute(
                    "SELECT command_id, idempotency_key, request_json, receipt_json FROM commands "
                    "WHERE command_id = ? OR idempotency_key = ? ORDER BY command_id LIMIT 1",
                    (command_id, idempotency_key),
                ).fetchone()
                if row is None:
                    raise
                record = self._command_record_from_row(row)
                return CommandClaim(False, row["command_id"] == command_id and row["idempotency_key"] != idempotency_key, record)
            except Exception:
                self._connection.rollback()
                raise

    def save_command(self, request: CommandRequest | Mapping[str, Any], receipt: CommandReceipt | Mapping[str, Any]) -> None:
        """Persist a receipt without replacing the immutable request identity."""
        request_payload = self._request_payload(request)
        receipt_payload = self._receipt_payload(receipt)
        command_id, idempotency_key = self._command_identity(request_payload)
        receipt_id = receipt_payload.get("commandId") or receipt_payload.get("command_id")
        if receipt_id != command_id:
            raise ValueError("receipt commandId does not match request")
        status = receipt_payload.get("status")
        if hasattr(status, "value"):
            status = status.value
        if not isinstance(status, str):
            status = str(status)
        now = self._now()
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json, idempotency_key FROM commands WHERE command_id = ?", (command_id,)
            ).fetchone()
            if row is None:
                # Keep direct repository users compatible, but still enforce
                # the same unique identity rules as the claim path.
                self._connection.execute(
                    "INSERT INTO commands(command_id, idempotency_key, request_json, receipt_json, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (command_id, idempotency_key, _encode(request_payload), _encode(receipt_payload), status, now, now),
                )
            else:
                if row["idempotency_key"] != idempotency_key:
                    raise ValueError("command identity conflict")
                self._connection.execute(
                    "UPDATE commands SET receipt_json = ?, status = ?, updated_at = ? WHERE command_id = ?",
                    (_encode(receipt_payload), status, now, command_id),
                )
            self._connection.commit()

    def get_command_record(self, command_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json, receipt_json, idempotency_key FROM commands WHERE command_id = ?", (command_id,)
            ).fetchone()
        if row is None:
            return None
        return self._command_record_from_row(row)

    def get_command_by_idempotency(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json, receipt_json, idempotency_key FROM commands WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return None if row is None else self._command_record_from_row(row)

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

    def _mapping_values(self, record: MappingRecord) -> tuple[str, str, str, str, str | None, str | None]:
        mapping_payload, status, target_id, probe_payload = self._mapping_payload(record)
        mapping_id = mapping_payload.get("mappingId")
        version = mapping_payload.get("version")
        if not isinstance(mapping_id, str) or not isinstance(version, str):
            raise ValueError("mapping identity is required")
        command_key = mapping_payload.get("commandKey") or mapping_payload.get("command_key")
        if not isinstance(command_key, str) or not command_key:
            raise ValueError("mapping commandKey is required")
        return (
            mapping_id,
            version,
            _encode(mapping_payload),
            status,
            target_id,
            None if probe_payload is None else _encode(probe_payload),
        )

    def save_mapping(self, record: MappingRecord) -> None:
        self.save_mapping_batch([record], active=record.mapping if record.status == "active" else None)

    def save_mapping_batch(self, records: list[MappingRecord] | tuple[MappingRecord, ...], *, active: Any = None) -> None:
        """Persist mapping snapshots and the active pointer in one transaction."""
        values = [self._mapping_values(record) for record in records]
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                for mapping_id, version, payload_json, status, target_id, probe_json in values:
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
                        (mapping_id, version, payload_json, status, target_id, probe_json, self._now()),
                    )
                if active is not None:
                    active_payload = dict(_jsonable(active))
                    command_key = active_payload.get("commandKey") or active_payload.get("command_key")
                    mapping_id = active_payload.get("mappingId") or active_payload.get("mapping_id")
                    version = active_payload.get("version")
                    if not all(isinstance(item, str) and item for item in (command_key, mapping_id, version)):
                        raise ValueError("active mapping identity is required")
                    self._connection.execute(
                        "INSERT INTO mapping_active(command_key, mapping_id, version, updated_at) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(command_key) DO UPDATE SET mapping_id=excluded.mapping_id, version=excluded.version, updated_at=excluded.updated_at",
                        (command_key, mapping_id, version, self._now()),
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

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

    def get_active_pointer(self, command_key: str) -> tuple[str, str] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT mapping_id, version FROM mapping_active WHERE command_key = ?", (command_key,)
            ).fetchone()
        return None if row is None else (str(row["mapping_id"]), str(row["version"]))

    def list_active_pointers(self) -> dict[str, tuple[str, str]]:
        with self._lock:
            rows = self._connection.execute("SELECT command_key, mapping_id, version FROM mapping_active").fetchall()
        return {str(row["command_key"]): (str(row["mapping_id"]), str(row["version"])) for row in rows}

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

    def save_assistant_draft(self, draft_id: str, payload: Mapping[str, Any], *, status: str = "draft") -> bool:
        """Insert an immutable draft; never overwrite an existing plan."""
        if not isinstance(draft_id, str) or not draft_id:
            raise ValueError("draft_id is required")
        now = self._now()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO assistant_drafts(draft_id, payload_json, status, execution_json, error, created_at, updated_at)
                VALUES (?, ?, ?, '{}', NULL, ?, ?)
                ON CONFLICT(draft_id) DO NOTHING
                """,
                (draft_id, _encode(payload), status, now, now),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def get_assistant_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json, status, execution_json, error, created_at, updated_at FROM assistant_drafts WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "draftId": draft_id,
            "plan": _decode(row["payload_json"]),
            "status": row["status"],
            "execution": _decode(row["execution_json"] or "{}"),
            "error": row["error"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def set_assistant_draft_status(
        self,
        draft_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        execution: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            clauses = ["status = ?", "updated_at = ?"]
            values: list[Any] = [status, self._now()]
            if execution is not None:
                clauses.append("execution_json = ?")
                values.append(_encode(execution))
            if error is not None:
                clauses.append("error = ?")
                values.append(error)
            elif status in {"confirmed", "draft", "unknown"}:
                clauses.append("error = NULL")
            values.append(draft_id)
            where = "draft_id = ?"
            if expected_status is not None:
                where += " AND status = ?"
                values.append(expected_status)
            cursor = self._connection.execute(
                f"UPDATE assistant_drafts SET {', '.join(clauses)} WHERE {where}", values
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
        """Compatibility alias; interrupted confirmations are not replayable."""
        self.recover_confirming_drafts()

    def recover_confirming_drafts(self) -> None:
        """Mark interrupted confirmations unknown rather than resetting them."""
        with self._lock:
            self._connection.execute(
                "UPDATE assistant_drafts SET status = 'unknown', error = ?, updated_at = ? WHERE status = 'confirming'",
                ("confirmation interrupted by process restart; reconciliation required", self._now()),
            )
            self._connection.commit()

    def update_assistant_execution(
        self,
        draft_id: str,
        execution: Mapping[str, Any],
        *,
        expected_status: str = "confirming",
    ) -> bool:
        """CAS update of durable per-step receipts while a plan is confirming."""
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE assistant_drafts SET execution_json = ?, updated_at = ? "
                "WHERE draft_id = ? AND status = ?",
                (_encode(execution), self._now(), draft_id, expected_status),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = ["CommandClaim", "SQLiteStore"]
