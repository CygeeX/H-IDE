from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from packages.scene_core import Calibration

from .contracts import ClaimRequest, CompleteJobRequest, FailJobRequest, UploadCreate, WorkerRequest
from .catalog import _queue_map_job, install_catalog_routes
from .pcd_header import PcdHeaderError, parse_pcd_header
from .storage import LocalSceneStorage, StorageError
from .store import SceneStore


DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024**3
DEFAULT_CHUNK_SIZE = 8 * 1024**2
DEFAULT_DISK_RESERVE_BYTES = 1024**3
MAX_ATTEMPTS = 2
bearer = HTTPBearer(auto_error=False)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _row(connection: sqlite3.Connection, table: str, record_id: str):
    # The table name is selected by callers, never by HTTP input.
    return connection.execute(f"SELECT * FROM {table} WHERE id = ?", (record_id,)).fetchone()


def _not_found(kind: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{kind} not found")


def _conflict(message: str) -> HTTPException:
    return HTTPException(status_code=409, detail=message)


def _sanitize_message(message: str) -> str:
    return "".join(character if character.isprintable() else " " for character in message).strip()[:500]


def create_app(
    db_path: str | os.PathLike[str],
    storage_root: str | os.PathLike[str],
    token: str,
    internal_token: str,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    disk_reserve_bytes: int = DEFAULT_DISK_RESERVE_BYTES,
    asset_cookie_secure: bool = False,
    target_clearance_meters: float = 0.35,
) -> FastAPI:
    if not token or not internal_token or hmac.compare_digest(token, internal_token):
        raise ValueError("public and internal bearer tokens must be non-empty and distinct")
    if max_upload_bytes <= 0 or chunk_size <= 0 or disk_reserve_bytes < 0:
        raise ValueError("upload and storage limits are invalid")
    if not math.isfinite(target_clearance_meters) or target_clearance_meters <= 0:
        raise ValueError("target clearance must be a positive finite value")

    store = SceneStore(db_path)
    storage = LocalSceneStorage(storage_root)
    application = FastAPI(title="H-IDE Scene API")
    application.state.store = store
    application.state.storage = storage

    def require(expected: str) -> Callable:
        def dependency(
            credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
        ) -> None:
            if credentials is None or credentials.scheme.lower() != "bearer" or not hmac.compare_digest(
                credentials.credentials, expected
            ):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="invalid bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return dependency

    public_auth = require(token)
    internal_auth = require(internal_token)

    @application.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @application.get("/readyz")
    def ready() -> dict:
        try:
            with store.connect() as connection:
                connection.execute("SELECT 1").fetchone()
            storage.root.stat()
        except (OSError, sqlite3.Error) as error:
            raise HTTPException(status_code=503, detail="scene service is not ready") from error
        return {"status": "ready"}

    def upload_result(row: sqlite3.Row) -> dict:
        return {
            "uploadId": row["id"],
            "status": row["status"],
            "offset": row["offset"],
            "sizeBytes": row["size_bytes"],
            "chunkSize": chunk_size,
            "expiresAt": row["expires_at"],
        }

    def completion_result(row: sqlite3.Row) -> dict:
        return {
            "sceneId": row["scene_id"],
            "assetId": row["asset_id"],
            "jobId": row["job_id"],
            "sha256": row["sha256"],
            "status": "queued",
            "dataEncoding": row["data_encoding"],
        }

    def job_result(row: sqlite3.Row) -> dict:
        manifest = json.loads(row["artifact_manifest"]) if row["artifact_manifest"] else None
        job_input = json.loads(row["input_json"]) if row["input_json"] else None
        return {
            "jobId": row["id"],
            "sceneId": row["scene_id"],
            "assetId": row["asset_id"],
            "jobKind": row["kind"],
            "revision": row["revision"],
            "input": job_input,
            "inputHash": row["input_hash"],
            "status": row["status"],
            "attempts": row["attempts"],
            "workerId": row["worker_id"],
            "leaseExpiresAt": row["lease_expires_at"],
            "artifactManifest": manifest,
            "errorCode": row["error_code"],
            "message": row["error_message"],
        }

    @application.post("/api/v1/scene-uploads", dependencies=[Depends(public_auth)])
    def create_upload(payload: UploadCreate) -> dict:
        file_name = payload.file_name
        if Path(file_name).name != file_name or "/" in file_name or "\\" in file_name:
            raise HTTPException(status_code=422, detail="fileName must be a basename")
        if Path(file_name).suffix.lower() != ".pcd":
            raise HTTPException(status_code=422, detail="only .pcd files are accepted")
        if payload.size_bytes > max_upload_bytes:
            raise HTTPException(status_code=422, detail="upload exceeds configured maximum")
        required = 4 * payload.size_bytes + disk_reserve_bytes
        if storage.free_bytes() < required:
            raise HTTPException(status_code=507, detail="insufficient storage")
        upload_id = str(uuid4())
        created = _now()
        expires = created + timedelta(hours=24)
        with store.transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO uploads
                   (id, file_name, size_bytes, last_modified, status, offset, expires_at, created_at)
                   VALUES (?, ?, ?, ?, 'active', 0, ?, ?)""",
                (
                    upload_id,
                    file_name,
                    payload.size_bytes,
                    None if payload.last_modified is None else str(payload.last_modified),
                    _iso(expires),
                    _iso(created),
                ),
            )
            row = _row(connection, "uploads", upload_id)
        return upload_result(row)

    @application.get("/api/v1/scene-uploads/{upload_id}", dependencies=[Depends(public_auth)])
    def get_upload(upload_id: str) -> dict:
        with store.connect() as connection:
            row = _row(connection, "uploads", upload_id)
        if row is None:
            raise _not_found("upload")
        return upload_result(row)

    @application.patch("/api/v1/scene-uploads/{upload_id}", dependencies=[Depends(public_auth)])
    async def append_upload(
        upload_id: str,
        request: Request,
        upload_offset: Annotated[str | None, Header(alias="Upload-Offset")] = None,
        upload_checksum: Annotated[str | None, Header(alias="Upload-Checksum")] = None,
    ) -> dict:
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/octet-stream":
            raise HTTPException(status_code=415, detail="Content-Type must be application/octet-stream")
        try:
            requested_offset = int(upload_offset or "")
            if requested_offset < 0:
                raise ValueError
        except ValueError as error:
            raise HTTPException(status_code=422, detail="Upload-Offset must be a non-negative integer") from error
        pieces = (upload_checksum or "").split()
        if len(pieces) != 2 or pieces[0].lower() != "sha256":
            raise HTTPException(status_code=422, detail="Upload-Checksum must use sha256")
        try:
            expected_digest = base64.b64decode(pieces[1], validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(status_code=422, detail="Upload-Checksum is not valid base64") from error
        if len(expected_digest) != hashlib.sha256().digest_size:
            raise HTTPException(status_code=422, detail="Upload-Checksum has the wrong length")

        body = bytearray()
        async for block in request.stream():
            body.extend(block)
            if len(body) > chunk_size:
                raise HTTPException(status_code=422, detail="chunk exceeds configured maximum")
        if not body:
            raise HTTPException(status_code=422, detail="chunk must not be empty")
        if not hmac.compare_digest(hashlib.sha256(body).digest(), expected_digest):
            raise HTTPException(status_code=422, detail="chunk checksum mismatch")

        offset_conflict: str | None = None
        new_offset: int | None = None
        with store.transaction(immediate=True) as connection:
            row = _row(connection, "uploads", upload_id)
            if row is None:
                raise _not_found("upload")
            if row["status"] != "active":
                raise _conflict("upload is not active")
            try:
                actual_offset = storage.part_size(upload_id)
            except StorageError as error:
                raise _conflict(str(error)) from error
            if actual_offset != row["offset"]:
                connection.execute("UPDATE uploads SET offset = ? WHERE id = ?", (actual_offset, upload_id))
            if requested_offset != actual_offset:
                offset_conflict = f"expected Upload-Offset {actual_offset}"
            elif actual_offset + len(body) > row["size_bytes"]:
                raise HTTPException(status_code=422, detail="chunk exceeds declared upload size")
            else:
                try:
                    new_offset = storage.append_part(upload_id, bytes(body))
                except (OSError, StorageError) as error:
                    raise HTTPException(status_code=500, detail="could not persist upload chunk") from error
                connection.execute("UPDATE uploads SET offset = ? WHERE id = ?", (new_offset, upload_id))
        if offset_conflict is not None:
            raise _conflict(offset_conflict)
        assert new_offset is not None
        return {"uploadId": upload_id, "status": "active", "offset": new_offset}

    @application.post("/api/v1/scene-uploads/{upload_id}/complete", dependencies=[Depends(public_auth)])
    def complete_upload(upload_id: str) -> dict:
        with store.transaction(immediate=True) as connection:
            row = _row(connection, "uploads", upload_id)
            if row is None:
                raise _not_found("upload")
            if row["status"] == "completed":
                return completion_result(row)
            if row["status"] != "active":
                raise _conflict("upload is not active")
            try:
                actual_size = storage.part_size(upload_id)
            except StorageError as error:
                raise _conflict(str(error)) from error
            if actual_size != row["size_bytes"]:
                if actual_size != row["offset"]:
                    connection.execute("UPDATE uploads SET offset = ? WHERE id = ?", (actual_size, upload_id))
                raise _conflict("upload size is incomplete")
            part_path = storage.part_path(upload_id)
            digest = hashlib.sha256()
            with part_path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            try:
                header = parse_pcd_header(part_path)
            except (OSError, PcdHeaderError) as error:
                raise HTTPException(status_code=422, detail=str(error)) from error

            scene_id, asset_id, job_id = str(uuid4()), str(uuid4()), str(uuid4())
            try:
                raw_path = storage.finalize_part(upload_id, asset_id)
                now = _iso(_now())
                sha256 = digest.hexdigest()
                connection.execute(
                    "INSERT INTO scenes (id, raw_asset_id, created_at) VALUES (?, ?, ?)",
                    (scene_id, asset_id, now),
                )
                connection.execute(
                    """INSERT INTO assets
                       (id, scene_id, kind, storage_key, sha256, size_bytes, data_encoding,
                        decompressed_bytes, created_at)
                       VALUES (?, ?, 'raw_pcd', ?, ?, ?, ?, ?, ?)""",
                    (
                        asset_id,
                        scene_id,
                        str(raw_path.relative_to(storage.root)),
                        sha256,
                        actual_size,
                        header.data_encoding,
                        header.decompressed_bytes,
                        now,
                    ),
                )
                connection.execute(
                    """INSERT INTO jobs
                       (id, scene_id, asset_id, kind, status, attempts, created_at, updated_at)
                       VALUES (?, ?, ?, 'import', 'queued', 0, ?, ?)""",
                    (job_id, scene_id, asset_id, now, now),
                )
                connection.execute(
                    """UPDATE uploads SET status = 'completed', completed_at = ?, scene_id = ?,
                       asset_id = ?, job_id = ?, sha256 = ?, data_encoding = ?, offset = ? WHERE id = ?""",
                    (now, scene_id, asset_id, job_id, sha256, header.data_encoding, actual_size, upload_id),
                )
            except BaseException:
                if 'raw_path' in locals() and raw_path.exists():
                    raw_path.chmod(0o600)
                    os.rename(raw_path, part_path)
                raise
            completed = _row(connection, "uploads", upload_id)
        return completion_result(completed)

    @application.post("/api/v1/scene-uploads/{upload_id}/cancel", dependencies=[Depends(public_auth)])
    def cancel_upload(upload_id: str) -> dict:
        with store.transaction(immediate=True) as connection:
            row = _row(connection, "uploads", upload_id)
            if row is None:
                raise _not_found("upload")
            if row["status"] != "active":
                raise _conflict("only active uploads can be cancelled")
            connection.execute("UPDATE uploads SET status = 'cancelled' WHERE id = ?", (upload_id,))
            storage.remove_part(upload_id)
            updated = _row(connection, "uploads", upload_id)
        return upload_result(updated)

    def cleanup_expired() -> int:
        expired_ids: list[str] = []
        with store.transaction(immediate=True) as connection:
            rows = connection.execute(
                "SELECT id FROM uploads WHERE status = 'active' AND expires_at <= ?", (_iso(_now()),)
            ).fetchall()
            expired_ids = [row["id"] for row in rows]
            if expired_ids:
                placeholders = ",".join("?" for _ in expired_ids)
                connection.execute(
                    f"UPDATE uploads SET status = 'expired' WHERE id IN ({placeholders})", expired_ids
                )
            for expired_id in expired_ids:
                storage.remove_part(expired_id)
        return len(expired_ids)

    application.state.cleanup_expired = cleanup_expired

    @application.post("/internal/v1/scene-uploads/cleanup", dependencies=[Depends(internal_auth)])
    def cleanup_uploads() -> dict:
        return {"expired": cleanup_expired()}

    @application.post("/internal/v1/scene-jobs/claim", dependencies=[Depends(internal_auth)])
    def claim_job(payload: ClaimRequest, response: Response):
        now = _now()
        now_text = _iso(now)
        lease_text = _iso(now + timedelta(seconds=payload.lease_seconds))
        with store.transaction(immediate=True) as connection:
            connection.execute(
                """UPDATE jobs SET status = 'queued', worker_id = NULL, lease_expires_at = NULL,
                   updated_at = ? WHERE status = 'running' AND lease_expires_at <= ? AND attempts < ?""",
                (now_text, now_text, MAX_ATTEMPTS),
            )
            connection.execute(
                """UPDATE jobs SET status = 'failed', worker_id = NULL, lease_expires_at = NULL,
                   error_code = 'lease_expired', error_message = 'worker lease expired', updated_at = ?
                   WHERE status = 'running' AND lease_expires_at <= ? AND attempts >= ?""",
                (now_text, now_text, MAX_ATTEMPTS),
            )
            queued = connection.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if queued is None:
                response.status_code = 204
                return None
            connection.execute(
                """UPDATE jobs SET status = 'running', attempts = attempts + 1, worker_id = ?,
                   lease_expires_at = ?, error_code = NULL, error_message = NULL, updated_at = ?
                   WHERE id = ? AND status = 'queued'""",
                (payload.worker_id, lease_text, now_text, queued["id"]),
            )
            claimed = _row(connection, "jobs", queued["id"])
        return job_result(claimed)

    def checked_running_job(connection: sqlite3.Connection, job_id: str, worker_id: str):
        row = _row(connection, "jobs", job_id)
        if row is None:
            raise _not_found("job")
        if (
            row["status"] != "running"
            or row["worker_id"] != worker_id
            or not row["lease_expires_at"]
            or row["lease_expires_at"] <= _iso(_now())
        ):
            raise _conflict("worker does not hold an active lease")
        return row

    @application.post("/internal/v1/scene-jobs/{job_id}/heartbeat", dependencies=[Depends(internal_auth)])
    def heartbeat_job(job_id: str, payload: WorkerRequest) -> dict:
        with store.transaction(immediate=True) as connection:
            checked_running_job(connection, job_id, payload.worker_id)
            lease = _iso(_now() + timedelta(seconds=payload.lease_seconds))
            connection.execute(
                "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
                (lease, _iso(_now()), job_id),
            )
            updated = _row(connection, "jobs", job_id)
        return job_result(updated)

    @application.post("/internal/v1/scene-jobs/{job_id}/complete", dependencies=[Depends(internal_auth)])
    def finish_job(job_id: str, payload: CompleteJobRequest) -> dict:
        with store.transaction(immediate=True) as connection:
            row = checked_running_job(connection, job_id, payload.worker_id)
            manifest = payload.artifact_manifest.model_dump(mode="json", by_alias=True)
            manifest_kind = manifest.get("jobKind", "import")
            if manifest_kind != row["kind"]:
                raise HTTPException(status_code=422, detail="artifact manifest does not match job kind")
            if row["kind"] == "map_derivation" and manifest.get("inputHash") != row["input_hash"]:
                raise HTTPException(status_code=422, detail="map artifact input hash does not match job")
            file_groups = (
                [manifest["files"]]
                if "files" in manifest
                else [manifest["publicFiles"], manifest["internalFiles"]]
            )
            for files in file_groups:
                for name, storage_key in files.items():
                    expected = str(
                        storage.artifact_path(row["scene_id"], row["id"], name).relative_to(storage.root)
                    )
                    if storage_key != expected:
                        raise HTTPException(status_code=422, detail="artifact key does not match claimed job")
                    try:
                        storage.resolve_artifact_key(storage_key)
                    except StorageError as error:
                        raise HTTPException(status_code=422, detail=str(error)) from error
            now = _iso(_now())
            connection.execute(
                """UPDATE jobs SET status = 'succeeded', artifact_manifest = ?, worker_id = NULL,
                   lease_expires_at = NULL, updated_at = ? WHERE id = ?""",
                (json.dumps(manifest, separators=(",", ":"), sort_keys=True), now, job_id),
            )
            if row["kind"] == "import":
                base_revision = connection.execute(
                    "SELECT * FROM scene_revisions WHERE scene_id = ? "
                    "ORDER BY revision DESC LIMIT 1",
                    (row["scene_id"],),
                ).fetchone()
                next_revision = 1 if base_revision is None else base_revision["revision"] + 1
                connection.execute(
                    """INSERT INTO scene_revisions
                       (scene_id, revision, status, base_revision, import_job_id,
                        calibration_json, annotations_json, bounds_map_json, created_at)
                       VALUES (?, ?, 'draft', ?, ?, ?, ?, ?, ?)""",
                    (
                        row["scene_id"],
                        next_revision,
                        None if base_revision is None else base_revision["revision"],
                        row["id"],
                        None if base_revision is None else base_revision["calibration_json"],
                        "[]" if base_revision is None else base_revision["annotations_json"],
                        None if base_revision is None else base_revision["bounds_map_json"],
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE jobs SET revision = ? WHERE id = ?",
                    (next_revision, row["id"]),
                )
                if base_revision is not None and base_revision["calibration_json"]:
                    _queue_map_job(
                        connection,
                        row["scene_id"],
                        next_revision,
                        Calibration.model_validate_json(base_revision["calibration_json"]),
                    )
            updated = _row(connection, "jobs", job_id)
        return job_result(updated)

    @application.post("/internal/v1/scene-jobs/{job_id}/fail", dependencies=[Depends(internal_auth)])
    def fail_job(job_id: str, payload: FailJobRequest) -> dict:
        with store.transaction(immediate=True) as connection:
            row = checked_running_job(connection, job_id, payload.worker_id)
            next_status = "failed" if payload.deterministic or row["attempts"] >= MAX_ATTEMPTS else "queued"
            connection.execute(
                """UPDATE jobs SET status = ?, worker_id = NULL, lease_expires_at = NULL,
                   error_code = ?, error_message = ?, updated_at = ? WHERE id = ?""",
                (next_status, payload.error_code, _sanitize_message(payload.message), _iso(_now()), job_id),
            )
            updated = _row(connection, "jobs", job_id)
        return job_result(updated)

    @application.get("/api/v1/scene-jobs/{job_id}", dependencies=[Depends(public_auth)])
    def get_job(job_id: str) -> dict:
        with store.connect() as connection:
            row = _row(connection, "jobs", job_id)
        if row is None:
            raise _not_found("job")
        return job_result(row)

    @application.post("/api/v1/scene-jobs/{job_id}/cancel", dependencies=[Depends(public_auth)])
    def cancel_job(job_id: str) -> dict:
        with store.transaction(immediate=True) as connection:
            row = _row(connection, "jobs", job_id)
            if row is None:
                raise _not_found("job")
            if row["status"] not in {"queued", "running"}:
                raise _conflict("job cannot be cancelled from its current state")
            connection.execute(
                """UPDATE jobs SET status = 'cancelled', worker_id = NULL, lease_expires_at = NULL,
                   updated_at = ? WHERE id = ?""",
                (_iso(_now()), job_id),
            )
            updated = _row(connection, "jobs", job_id)
        return job_result(updated)

    @application.post("/api/v1/scene-jobs/{job_id}/retry", dependencies=[Depends(public_auth)])
    def retry_job(job_id: str) -> dict:
        with store.transaction(immediate=True) as connection:
            row = _row(connection, "jobs", job_id)
            if row is None:
                raise _not_found("job")
            if row["status"] != "failed":
                raise _conflict("only failed jobs can be retried")
            connection.execute(
                """UPDATE jobs SET status = 'queued', attempts = 0, worker_id = NULL,
                   lease_expires_at = NULL, artifact_manifest = NULL, error_code = NULL,
                   error_message = NULL, updated_at = ? WHERE id = ?""",
                (_iso(_now()), job_id),
            )
            updated = _row(connection, "jobs", job_id)
        return job_result(updated)

    install_catalog_routes(
        application,
        store,
        storage,
        public_auth,
        internal_auth,
        asset_cookie_secure=asset_cookie_secure,
        target_clearance_meters=target_clearance_meters,
    )
    return application


app = create_app(
    os.environ.get("SCENE_API_DB", "/tmp/hide-scene-api.sqlite3"),
    os.environ.get("SCENE_STORAGE_ROOT", "/tmp/hide-scene-storage"),
    os.environ.get("SCENE_API_TOKEN", "development-public-token"),
    os.environ.get(
        "SCENE_API_INTERNAL_TOKEN",
        os.environ.get("SCENE_INTERNAL_TOKEN", "development-internal-token"),
    ),
    max_upload_bytes=int(os.environ.get("SCENE_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
    chunk_size=int(os.environ.get("SCENE_UPLOAD_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)),
    disk_reserve_bytes=int(os.environ.get("SCENE_DISK_RESERVE_BYTES", DEFAULT_DISK_RESERVE_BYTES)),
    asset_cookie_secure=os.environ.get("SCENE_ASSET_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"},
    target_clearance_meters=float(os.environ.get("SCENE_TARGET_CLEARANCE_METERS", "0.35")),
)
