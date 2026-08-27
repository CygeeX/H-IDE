from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

from h_ide_vision.contracts import (
    ResolutionStatus,
    VisualResolution,
    VisualTaskResult,
)


class ResolutionNotFound(KeyError):
    pass


class ResolutionConflict(ValueError):
    pass


class ResolutionRepository:
    """Durable immutable-resolution store.

    SQLite is the local profile; PostgreSQL is the deployment target.
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS visual_resolutions (
              visual_resolution_id TEXT PRIMARY KEY,
              request_id TEXT NOT NULL UNIQUE,
              mission_id TEXT NOT NULL,
              stage_id TEXT NOT NULL,
              attempt INTEGER NOT NULL,
              status TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            DROP INDEX IF EXISTS uq_resolution_attempt_request;
            CREATE UNIQUE INDEX IF NOT EXISTS uq_resolution_attempt
              ON visual_resolutions(mission_id, stage_id, attempt);
            CREATE TABLE IF NOT EXISTS dispatch_bindings (
              visual_resolution_id TEXT PRIMARY KEY,
              command_id TEXT NOT NULL UNIQUE,
              bound_at TEXT NOT NULL,
              FOREIGN KEY(visual_resolution_id) REFERENCES visual_resolutions(visual_resolution_id)
            );
            CREATE TABLE IF NOT EXISTS visual_task_results (
              analysis_id TEXT PRIMARY KEY,
              request_id TEXT NOT NULL UNIQUE,
              mission_id TEXT NOT NULL,
              revision_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              attempt INTEGER NOT NULL,
              payload_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              UNIQUE(mission_id, revision_id, node_id, attempt)
            );
            """
        )

    def put_visual_task_result(self, result: VisualTaskResult) -> VisualTaskResult:
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO visual_task_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        result.analysisId,
                        result.requestId,
                        result.missionId,
                        result.revisionId,
                        result.nodeId,
                        result.attempt,
                        result.model_dump_json(),
                        result.createdAt.isoformat(),
                    ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                existing = self.get_visual_task_by_request(result.requestId)
                if existing is not None:
                    return existing
                raise ResolutionConflict(
                    "visual task mission revision/node attempt already exists"
                ) from exc
        return result

    def get_visual_task_result(self, analysis_id: str) -> VisualTaskResult:
        row = self._connection.execute(
            "SELECT payload_json FROM visual_task_results WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()
        if row is None:
            raise ResolutionNotFound(analysis_id)
        return VisualTaskResult.model_validate_json(row["payload_json"])

    def get_visual_task_by_request(self, request_id: str) -> VisualTaskResult | None:
        row = self._connection.execute(
            "SELECT payload_json FROM visual_task_results WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        return VisualTaskResult.model_validate_json(row["payload_json"])

    def insert(self, request_id: str, resolution: VisualResolution) -> VisualResolution:
        payload = resolution.model_dump_json()
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO visual_resolutions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        resolution.visualResolutionId,
                        request_id,
                        resolution.missionId,
                        resolution.stageId,
                        resolution.attempt,
                        resolution.status.value,
                        payload,
                        resolution.createdAt.isoformat(),
                    ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                existing = self.get_by_request(request_id)
                if existing is not None:
                    return existing
                raise ResolutionConflict("mission stage attempt already exists") from exc
        return resolution

    def get(self, resolution_id: str) -> VisualResolution:
        row = self._connection.execute(
            "SELECT payload_json FROM visual_resolutions WHERE visual_resolution_id = ?", (resolution_id,)
        ).fetchone()
        if row is None:
            raise ResolutionNotFound(resolution_id)
        return VisualResolution.model_validate_json(row["payload_json"])

    def get_by_request(self, request_id: str) -> VisualResolution | None:
        row = self._connection.execute(
            "SELECT payload_json FROM visual_resolutions WHERE request_id = ?", (request_id,)
        ).fetchone()
        return None if row is None else VisualResolution.model_validate_json(row["payload_json"])

    def replace_terminal(self, resolution: VisualResolution) -> VisualResolution:
        if resolution.status in {ResolutionStatus.QUEUED, ResolutionStatus.RESOLVING}:
            raise ResolutionConflict("replacement must be terminal")
        with self._lock:
            row = self._connection.execute(
                "SELECT status FROM visual_resolutions WHERE visual_resolution_id = ?",
                (resolution.visualResolutionId,),
            ).fetchone()
            if row is None:
                raise ResolutionNotFound(resolution.visualResolutionId)
            if row["status"] not in {ResolutionStatus.QUEUED.value, ResolutionStatus.RESOLVING.value}:
                return self.get(resolution.visualResolutionId)
            self._connection.execute(
                "UPDATE visual_resolutions SET status = ?, payload_json = ? WHERE visual_resolution_id = ?",
                (resolution.status.value, resolution.model_dump_json(), resolution.visualResolutionId),
            )
            self._connection.commit()
        return resolution

    def cancel(self, resolution_id: str) -> VisualResolution:
        current = self.get(resolution_id)
        if current.status not in {ResolutionStatus.QUEUED, ResolutionStatus.RESOLVING}:
            return current
        cancelled = current.model_copy(
            update={"status": ResolutionStatus.CANCELLED, "executable": False, "rejectionReason": "cancelled"}
        )
        return self.replace_terminal(cancelled)

    def fail_request(self, request_id: str, reason: str) -> VisualResolution | None:
        current = self.get_by_request(request_id)
        if current is None:
            return None
        if current.status not in {ResolutionStatus.QUEUED, ResolutionStatus.RESOLVING}:
            return current
        failed = current.model_copy(
            update={"status": ResolutionStatus.FAILED, "executable": False, "rejectionReason": reason}
        )
        return self.replace_terminal(failed)

    def recover_incomplete(self) -> int:
        rows = self._connection.execute(
            "SELECT request_id FROM visual_resolutions WHERE status IN (?, ?)",
            (ResolutionStatus.QUEUED.value, ResolutionStatus.RESOLVING.value),
        ).fetchall()
        for row in rows:
            self.fail_request(row["request_id"], "service_restarted_before_resolution_completed")
        return len(rows)

    def bind_dispatch(self, resolution_id: str, command_id: str) -> None:
        resolution = self.get(resolution_id)
        if not resolution.executable:
            raise ResolutionConflict("non-executable resolution cannot be dispatched")
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO dispatch_bindings VALUES (?, ?, ?)",
                    (resolution_id, command_id, datetime.now(UTC).isoformat()),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                row = self._connection.execute(
                    "SELECT command_id FROM dispatch_bindings WHERE visual_resolution_id = ?",
                    (resolution_id,),
                ).fetchone()
                if row is not None and row["command_id"] == command_id:
                    return
                raise ResolutionConflict("resolution or command is already bound") from exc
