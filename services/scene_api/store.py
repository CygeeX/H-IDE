from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    last_modified TEXT,
    status TEXT NOT NULL,
    offset INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    scene_id TEXT,
    asset_id TEXT,
    job_id TEXT,
    sha256 TEXT,
    data_encoding TEXT
);
CREATE TABLE IF NOT EXISTS scenes (
    id TEXT PRIMARY KEY,
    raw_asset_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assets (
    id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL REFERENCES scenes(id),
    kind TEXT NOT NULL,
    storage_key TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    data_encoding TEXT NOT NULL,
    decompressed_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL REFERENCES scenes(id),
    asset_id TEXT NOT NULL REFERENCES assets(id),
    kind TEXT NOT NULL DEFAULT 'import',
    revision INTEGER,
    input_json TEXT,
    input_hash TEXT,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    lease_expires_at TEXT,
    artifact_manifest TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_claim_order ON jobs(status, created_at, id);
CREATE TABLE IF NOT EXISTS scene_revisions (
    scene_id TEXT NOT NULL REFERENCES scenes(id),
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    base_revision INTEGER,
    import_job_id TEXT,
    calibration_json TEXT,
    annotations_json TEXT NOT NULL DEFAULT '[]',
    bounds_map_json TEXT,
    created_at TEXT NOT NULL,
    published_at TEXT,
    PRIMARY KEY (scene_id, revision)
);
CREATE TABLE IF NOT EXISTS robot_scene_bindings (
    robot_id TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (scene_id, revision) REFERENCES scene_revisions(scene_id, revision)
);
CREATE TABLE IF NOT EXISTS asset_sessions (
    token_hash TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (scene_id, revision) REFERENCES scene_revisions(scene_id, revision)
);
CREATE INDEX IF NOT EXISTS asset_sessions_expiry ON asset_sessions(expires_at);
"""


class SceneStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(assets)")}
            if "deleted_at" not in columns:
                connection.execute("ALTER TABLE assets ADD COLUMN deleted_at TEXT")
            job_columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "kind" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'import'")
            if "revision" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN revision INTEGER")
            if "input_json" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN input_json TEXT")
            if "input_hash" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN input_hash TEXT")
            revision_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(scene_revisions)")
            }
            if "import_job_id" not in revision_columns:
                connection.execute("ALTER TABLE scene_revisions ADD COLUMN import_job_id TEXT")
            # Pin legacy revisions to the newest successful import that existed
            # when the revision was created. Future reprocessing must not
            # silently change an immutable published revision.
            connection.execute(
                """UPDATE scene_revisions AS revision SET import_job_id = (
                     SELECT job.id FROM jobs AS job
                     WHERE job.scene_id = revision.scene_id AND job.kind = 'import'
                     AND job.status = 'succeeded' AND job.artifact_manifest IS NOT NULL
                     AND job.updated_at <= revision.created_at
                     ORDER BY job.updated_at DESC, job.id DESC LIMIT 1
                   ) WHERE import_job_id IS NULL"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS jobs_scene_kind_revision "
                "ON jobs(scene_id, kind, revision, updated_at, id)"
            )

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
