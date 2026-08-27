from __future__ import annotations

import hashlib
import json
import math
import secrets
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse

from packages.scene_core import (
    Bounds3D,
    Calibration,
    Point3D,
    SceneMapSummary,
    SemanticRegion,
    TargetMapCheck,
    compute_interactive_similarity,
    semantic_label_collisions,
)

from .contracts import (
    AnnotationUpdate,
    CalibrationUpdate,
    InteractiveCalibrationUpdate,
    RevisionCreate,
    SceneBindingUpdate,
    TargetCheckRequest,
)
from .storage import LocalSceneStorage, StorageError
from .store import SceneStore


ASSET_COOKIE = "hide_scene_asset"
ASSET_SESSION_SECONDS = 15 * 60
MAX_ENTRY_DISTANCE_METERS = 2.0


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _json_value(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _json(value) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _scene_row(connection: sqlite3.Connection, scene_id: str):
    return connection.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()


def _revision_row(connection: sqlite3.Connection, scene_id: str, revision: int):
    return connection.execute(
        "SELECT * FROM scene_revisions WHERE scene_id = ? AND revision = ?",
        (scene_id, revision),
    ).fetchone()


def _job_row(connection: sqlite3.Connection, scene_id: str):
    return connection.execute(
        "SELECT * FROM jobs WHERE scene_id = ? AND kind = 'import' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (scene_id,),
    ).fetchone()


def _successful_import_job(connection: sqlite3.Connection, scene_id: str):
    return connection.execute(
        "SELECT * FROM jobs WHERE scene_id = ? AND kind = 'import' AND status = 'succeeded' "
        "AND artifact_manifest IS NOT NULL ORDER BY updated_at DESC, id DESC LIMIT 1",
        (scene_id,),
    ).fetchone()


def _import_job_by_id(connection: sqlite3.Connection, scene_id: str, job_id: str | None):
    if not job_id:
        return None
    return connection.execute(
        "SELECT * FROM jobs WHERE id = ? AND scene_id = ? AND kind = 'import' "
        "AND status = 'succeeded' AND artifact_manifest IS NOT NULL",
        (job_id, scene_id),
    ).fetchone()


def _revision_import_job(connection: sqlite3.Connection, revision_row: sqlite3.Row):
    if revision_row["import_job_id"]:
        return _import_job_by_id(
            connection,
            revision_row["scene_id"],
            revision_row["import_job_id"],
        )
    return _successful_import_job(connection, revision_row["scene_id"])


def _map_job_row(
    connection: sqlite3.Connection,
    scene_id: str,
    revision: int,
    input_hash: str | None = None,
):
    if input_hash is None:
        return connection.execute(
            "SELECT * FROM jobs WHERE scene_id = ? AND kind = 'map_derivation' AND revision = ? "
            "ORDER BY updated_at DESC, id DESC LIMIT 1",
            (scene_id, revision),
        ).fetchone()
    return connection.execute(
        "SELECT * FROM jobs WHERE scene_id = ? AND kind = 'map_derivation' AND revision = ? "
        "AND input_hash = ? ORDER BY updated_at DESC, id DESC LIMIT 1",
        (scene_id, revision, input_hash),
    ).fetchone()


def _require_scene(connection: sqlite3.Connection, scene_id: str):
    row = _scene_row(connection, scene_id)
    if row is None:
        raise HTTPException(status_code=404, detail="scene not found")
    return row


def _require_revision(connection: sqlite3.Connection, scene_id: str, revision: int):
    _require_scene(connection, scene_id)
    row = _revision_row(connection, scene_id, revision)
    if row is None:
        raise HTTPException(status_code=404, detail="scene revision not found")
    return row


def _calibration_hash(calibration: Calibration | dict) -> str:
    value = calibration if isinstance(calibration, Calibration) else Calibration.model_validate(calibration)
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _map_input_hash(import_job_id: str, calibration: Calibration | dict) -> str:
    calibration_hash = _calibration_hash(calibration)
    return hashlib.sha256(f"{import_job_id}:{calibration_hash}".encode("utf-8")).hexdigest()


def _revision_map_input_hash(
    connection: sqlite3.Connection,
    revision_row: sqlite3.Row,
    calibration: Calibration | dict,
) -> str | None:
    import_job = _revision_import_job(connection, revision_row)
    if import_job is None:
        return None
    return _map_input_hash(import_job["id"], calibration)


def _manifest_files(manifest: dict, *, public: bool) -> dict[str, str]:
    if manifest.get("schemaVersion") == "h-ide.scene-artifacts/2":
        return manifest.get("publicFiles" if public else "internalFiles", {})
    return manifest.get("files", {}) if public else {}


def _job_result(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    manifest = json.loads(row["artifact_manifest"]) if row["artifact_manifest"] else None
    result = {
        "jobId": row["id"],
        "jobKind": row["kind"],
        "revision": row["revision"],
        "status": row["status"],
        "attempts": row["attempts"],
        "errorCode": row["error_code"],
        "message": row["error_message"],
    }
    if manifest and row["kind"] == "import":
        result["previewAvailable"] = "preview/map.json" in _manifest_files(manifest, public=True)
        result["processingSummary"] = manifest.get("processingSummary")
    return result


def _revision_result(connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
    calibration = json.loads(row["calibration_json"]) if row["calibration_json"] else None
    import_job = _revision_import_job(connection, row)
    import_manifest = (
        json.loads(import_job["artifact_manifest"])
        if import_job is not None and import_job["artifact_manifest"]
        else None
    )
    map_input_hash = (
        _revision_map_input_hash(connection, row, calibration)
        if calibration
        else None
    )
    map_job = _map_job_row(connection, row["scene_id"], row["revision"], map_input_hash)
    map_manifest = (
        json.loads(map_job["artifact_manifest"])
        if map_job and map_job["status"] == "succeeded" and map_job["artifact_manifest"]
        else None
    )
    return {
        "sceneId": row["scene_id"],
        "revision": row["revision"],
        "status": row["status"],
        "baseRevision": row["base_revision"],
        "importJobId": import_job["id"] if import_job is not None else None,
        "artifactsAvailable": import_manifest is not None,
        "previewAvailable": bool(
            import_manifest
            and "preview/map.json" in _manifest_files(import_manifest, public=True)
        ),
        "processingSummary": import_manifest.get("processingSummary") if import_manifest else None,
        "calibration": calibration,
        "annotations": json.loads(row["annotations_json"]),
        "boundsMap": json.loads(row["bounds_map_json"]) if row["bounds_map_json"] else None,
        "createdAt": row["created_at"],
        "publishedAt": row["published_at"],
        "mapJob": _job_result(map_job),
        "mapSummary": map_manifest.get("mapSummary") if map_manifest else None,
    }


def _scene_result(connection: sqlite3.Connection, scene_id: str) -> dict:
    scene = _require_scene(connection, scene_id)
    asset = connection.execute("SELECT * FROM assets WHERE id = ?", (scene["raw_asset_id"],)).fetchone()
    upload = connection.execute(
        "SELECT file_name FROM uploads WHERE scene_id = ? ORDER BY completed_at DESC LIMIT 1",
        (scene_id,),
    ).fetchone()
    revisions = connection.execute(
        "SELECT * FROM scene_revisions WHERE scene_id = ? ORDER BY revision", (scene_id,)
    ).fetchall()
    successful_import = _successful_import_job(connection, scene_id)
    successful_manifest = (
        json.loads(successful_import["artifact_manifest"])
        if successful_import is not None and successful_import["artifact_manifest"]
        else None
    )
    return {
        "sceneId": scene_id,
        "sourceFileName": upload["file_name"] if upload else None,
        "rawAssetId": scene["raw_asset_id"],
        "sourceAvailable": bool(asset and not asset["deleted_at"]),
        "artifactsAvailable": successful_manifest is not None,
        "previewAvailable": bool(
            successful_manifest
            and "preview/map.json" in _manifest_files(successful_manifest, public=True)
        ),
        "processingSummary": successful_manifest.get("processingSummary") if successful_manifest else None,
        "job": _job_result(_job_row(connection, scene_id)),
        "revisions": [_revision_result(connection, row) for row in revisions],
        "createdAt": scene["created_at"],
    }


def _manifest(
    connection: sqlite3.Connection,
    scene_id: str,
    import_job_id: str | None = None,
) -> dict:
    job = (
        _import_job_by_id(connection, scene_id, import_job_id)
        if import_job_id
        else _successful_import_job(connection, scene_id)
    )
    if job is None or not job["artifact_manifest"]:
        raise HTTPException(status_code=409, detail="scene conversion is not ready")
    return json.loads(job["artifact_manifest"])


def _queue_map_job(
    connection: sqlite3.Connection,
    scene_id: str,
    revision: int,
    calibration: Calibration,
) -> sqlite3.Row | None:
    revision_row = _require_revision(connection, scene_id, revision)
    import_job = _revision_import_job(connection, revision_row)
    if import_job is None or not import_job["artifact_manifest"]:
        return None
    import_manifest = json.loads(import_job["artifact_manifest"])
    internal_files = _manifest_files(import_manifest, public=False)
    analysis_key = internal_files.get("analysis.ply")
    processing = import_manifest.get("processingSummary") or {}
    voxel_size = processing.get("voxelSize")
    if not isinstance(analysis_key, str) or isinstance(voxel_size, bool) or not isinstance(voxel_size, (int, float)):
        return None

    calibration_hash = _calibration_hash(calibration)
    input_hash = _map_input_hash(import_job["id"], calibration)
    current = _map_job_row(connection, scene_id, revision, input_hash)
    if current is not None and current["status"] not in {"superseded", "cancelled"}:
        return current
    now = _iso(_now())
    connection.execute(
        """UPDATE jobs SET status = 'superseded', worker_id = NULL, lease_expires_at = NULL,
           updated_at = ? WHERE scene_id = ? AND kind = 'map_derivation' AND revision = ?
           AND input_hash != ? AND status IN ('queued', 'running')""",
        (now, scene_id, revision, input_hash),
    )
    job_id = str(uuid4())
    job_input = {
        "revision": revision,
        "calibration": calibration.model_dump(mode="json"),
        "calibrationHash": calibration_hash,
        "inputHash": input_hash,
        "sourceImportJobId": import_job["id"],
        "analysisStorageKey": analysis_key,
        "voxelSize": float(voxel_size),
    }
    connection.execute(
        """INSERT INTO jobs
           (id, scene_id, asset_id, kind, revision, input_json, input_hash, status,
            attempts, created_at, updated_at)
           VALUES (?, ?, ?, 'map_derivation', ?, ?, ?, 'queued', 0, ?, ?)""",
        (job_id, scene_id, import_job["asset_id"], revision, _json(job_input), input_hash, now, now),
    )
    return connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def _merged_public_files(connection: sqlite3.Connection, scene_id: str, revision: int) -> dict[str, str]:
    revision_row = _require_revision(connection, scene_id, revision)
    import_manifest = _manifest(connection, scene_id, revision_row["import_job_id"])
    files = dict(_manifest_files(import_manifest, public=True))
    if revision_row["calibration_json"]:
        map_input_hash = _revision_map_input_hash(
            connection,
            revision_row,
            json.loads(revision_row["calibration_json"]),
        )
        map_job = _map_job_row(connection, scene_id, revision, map_input_hash)
        if map_job is not None and map_job["status"] == "succeeded" and map_job["artifact_manifest"]:
            files.update(_manifest_files(json.loads(map_job["artifact_manifest"]), public=True))
    return files


def _map_artifacts(
    connection: sqlite3.Connection,
    scene_id: str,
    revision: int,
) -> tuple[sqlite3.Row, dict, dict[str, str]]:
    revision_row = _require_revision(connection, scene_id, revision)
    if not revision_row["calibration_json"]:
        raise HTTPException(status_code=409, detail="scene revision is not calibrated")
    map_input_hash = _revision_map_input_hash(
        connection,
        revision_row,
        json.loads(revision_row["calibration_json"]),
    )
    job = _map_job_row(connection, scene_id, revision, map_input_hash)
    if job is None or job["status"] != "succeeded" or not job["artifact_manifest"]:
        raise HTTPException(status_code=409, detail="calibrated scene map is not ready")
    manifest = json.loads(job["artifact_manifest"])
    return revision_row, manifest, _manifest_files(manifest, public=False)


def _transform_bounds(source: Bounds3D, calibration: Calibration) -> Bounds3D:
    matrix = calibration.TMapFromScene
    transformed: list[Point3D] = []
    for x in (source.min.x, source.max.x):
        for y in (source.min.y, source.max.y):
            for z in (source.min.z, source.max.z):
                values = [x, y, z, 1.0]
                result = [sum(matrix[row][column] * values[column] for column in range(4)) for row in range(4)]
                if not math.isclose(result[3], 1.0, abs_tol=1e-7, rel_tol=0.0):
                    raise HTTPException(status_code=422, detail="calibration is not an affine map transform")
                transformed.append(Point3D(x=result[0], y=result[1], z=result[2]))
    return Bounds3D(
        min=Point3D(
            x=min(point.x for point in transformed),
            y=min(point.y for point in transformed),
            z=min(point.z for point in transformed),
        ),
        max=Point3D(
            x=max(point.x for point in transformed),
            y=max(point.y for point in transformed),
            z=max(point.z for point in transformed),
        ),
    )


def _point_in_polygon(x: float, y: float, region: SemanticRegion) -> bool:
    inside = False
    points = region.polygonXY
    previous = points[-1]
    for current in points:
        if (current.y > y) != (previous.y > y):
            crossing_x = (previous.x - current.x) * (y - current.y) / (previous.y - current.y) + current.x
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def _validate_annotations(annotations: list[SemanticRegion], bounds: Bounds3D) -> None:
    collisions = semantic_label_collisions(annotations)
    if collisions:
        raise HTTPException(status_code=422, detail={"semanticLabelCollisions": collisions})
    for region in annotations:
        if not bounds.contains(region.entryPose):
            raise HTTPException(status_code=422, detail=f"entry pose for {region.locationId} is outside scene bounds")
        if region.zMin < bounds.min.z or region.zMax > bounds.max.z:
            raise HTTPException(status_code=422, detail=f"height range for {region.locationId} is outside scene bounds")
        if any(
            point.x < bounds.min.x
            or point.x > bounds.max.x
            or point.y < bounds.min.y
            or point.y > bounds.max.y
            for point in region.polygonXY
        ):
            raise HTTPException(status_code=422, detail=f"polygon for {region.locationId} is outside scene bounds")
        if not _point_in_polygon(region.entryPose.x, region.entryPose.y, region):
            distances = [
                _point_segment_distance(
                    region.entryPose.x,
                    region.entryPose.y,
                    region.polygonXY[index].x,
                    region.polygonXY[index].y,
                    region.polygonXY[(index + 1) % len(region.polygonXY)].x,
                    region.polygonXY[(index + 1) % len(region.polygonXY)].y,
                )
                for index in range(len(region.polygonXY))
            ]
            if min(distances) > MAX_ENTRY_DISTANCE_METERS:
                raise HTTPException(status_code=422, detail=f"entry pose for {region.locationId} is too far from region")


def install_catalog_routes(
    application: FastAPI,
    store: SceneStore,
    storage: LocalSceneStorage,
    public_auth: Callable,
    internal_auth: Callable,
    *,
    asset_cookie_secure: bool = False,
    target_clearance_meters: float = 0.35,
) -> None:
    if not math.isfinite(target_clearance_meters) or target_clearance_meters <= 0:
        raise ValueError("target clearance must be a positive finite value")
    @application.get("/api/v1/scenes", dependencies=[Depends(public_auth)])
    def list_scenes() -> list[dict]:
        with store.connect() as connection:
            scene_ids = [row["id"] for row in connection.execute("SELECT id FROM scenes ORDER BY created_at DESC")]
            return [_scene_result(connection, scene_id) for scene_id in scene_ids]

    @application.get("/api/v1/scenes/{scene_id}", dependencies=[Depends(public_auth)])
    def get_scene(scene_id: str) -> dict:
        with store.connect() as connection:
            return _scene_result(connection, scene_id)

    @application.post("/api/v1/scenes/{scene_id}/reprocess", dependencies=[Depends(public_auth)])
    def reprocess_scene(scene_id: str) -> dict:
        """Create a new immutable artifact version from the retained raw PCD."""

        with store.transaction(immediate=True) as connection:
            scene = _require_scene(connection, scene_id)
            asset = connection.execute(
                "SELECT * FROM assets WHERE id = ?", (scene["raw_asset_id"],)
            ).fetchone()
            if asset is None or asset["deleted_at"]:
                raise HTTPException(status_code=409, detail="raw source is unavailable for reprocessing")
            active = connection.execute(
                "SELECT id FROM jobs WHERE scene_id = ? AND kind = 'import' "
                "AND status IN ('queued', 'running') LIMIT 1",
                (scene_id,),
            ).fetchone()
            if active is not None:
                raise HTTPException(status_code=409, detail="scene reprocessing is already active")
            now = _iso(_now())
            job_id = str(uuid4())
            connection.execute(
                """INSERT INTO jobs
                   (id, scene_id, asset_id, kind, status, attempts, created_at, updated_at)
                   VALUES (?, ?, ?, 'import', 'queued', 0, ?, ?)""",
                (job_id, scene_id, asset["id"], now, now),
            )
            return _job_result(connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    @application.post("/api/v1/scenes/{scene_id}/revisions", dependencies=[Depends(public_auth)])
    def create_revision(scene_id: str, payload: RevisionCreate) -> dict:
        with store.transaction(immediate=True) as connection:
            _require_scene(connection, scene_id)
            if payload.base_revision is None:
                base = connection.execute(
                    "SELECT * FROM scene_revisions WHERE scene_id = ? ORDER BY revision DESC LIMIT 1",
                    (scene_id,),
                ).fetchone()
            else:
                base = _require_revision(connection, scene_id, payload.base_revision)
            latest = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) AS value FROM scene_revisions WHERE scene_id = ?",
                (scene_id,),
            ).fetchone()["value"]
            revision = latest + 1
            now = _iso(_now())
            if base is None:
                import_job = _successful_import_job(connection, scene_id)
                pinned_import_job_id = import_job["id"] if import_job is not None else None
            else:
                import_job = _revision_import_job(connection, base)
                pinned_import_job_id = import_job["id"] if import_job is not None else None
            connection.execute(
                """INSERT INTO scene_revisions
                   (scene_id, revision, status, base_revision, import_job_id,
                    calibration_json, annotations_json, bounds_map_json, created_at)
                   VALUES (?, ?, 'draft', ?, ?, ?, ?, ?, ?)""",
                (
                    scene_id,
                    revision,
                    None if base is None else base["revision"],
                    pinned_import_job_id,
                    None if base is None else base["calibration_json"],
                    "[]" if base is None else base["annotations_json"],
                    None if base is None else base["bounds_map_json"],
                    now,
                ),
            )
            created = _revision_row(connection, scene_id, revision)
            if created["calibration_json"]:
                _queue_map_job(
                    connection,
                    scene_id,
                    revision,
                    Calibration.model_validate_json(created["calibration_json"]),
                )
            return _revision_result(connection, created)

    def editable_revision(connection: sqlite3.Connection, scene_id: str, revision: int):
        row = _require_revision(connection, scene_id, revision)
        if row["status"] != "draft":
            raise HTTPException(status_code=409, detail="published scene revisions are immutable")
        return row

    def persist_calibration(
        connection: sqlite3.Connection,
        scene_id: str,
        revision: int,
        calibration: Calibration,
    ) -> dict:
        row = editable_revision(connection, scene_id, revision)
        import_job = _revision_import_job(connection, row)
        if import_job is None:
            raise HTTPException(status_code=409, detail="scene conversion is not ready")
        if not row["import_job_id"]:
            connection.execute(
                "UPDATE scene_revisions SET import_job_id = ? WHERE scene_id = ? AND revision = ?",
                (import_job["id"], scene_id, revision),
            )
            row = _revision_row(connection, scene_id, revision)
        manifest = json.loads(import_job["artifact_manifest"])
        source_bounds = Bounds3D.model_validate(manifest["sourceBounds"])
        bounds_map = _transform_bounds(source_bounds, calibration)
        connection.execute(
            """UPDATE scene_revisions SET calibration_json = ?, bounds_map_json = ?,
               annotations_json = '[]' WHERE scene_id = ? AND revision = ?""",
            (_json(calibration), _json(bounds_map), scene_id, revision),
        )
        _queue_map_job(connection, scene_id, revision, calibration)
        return _revision_result(connection, _revision_row(connection, scene_id, revision))

    @application.put(
        "/api/v1/scenes/{scene_id}/revisions/{revision}/calibration",
        dependencies=[Depends(public_auth)],
    )
    def update_calibration(scene_id: str, revision: int, payload: CalibrationUpdate) -> dict:
        with store.transaction(immediate=True) as connection:
            return persist_calibration(connection, scene_id, revision, payload.calibration)

    @application.post(
        "/api/v1/scenes/{scene_id}/revisions/{revision}/calibration/fit",
        dependencies=[Depends(public_auth)],
    )
    def fit_calibration(scene_id: str, revision: int, payload: InteractiveCalibrationUpdate) -> dict:
        try:
            calibration = compute_interactive_similarity(
                payload.ground_points,
                payload.origin_scene,
                payload.x_axis_scene,
                payload.known_distance_m,
                payload.map_origin,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        with store.transaction(immediate=True) as connection:
            return persist_calibration(connection, scene_id, revision, calibration)

    @application.put(
        "/api/v1/scenes/{scene_id}/revisions/{revision}/annotations",
        dependencies=[Depends(public_auth)],
    )
    def update_annotations(scene_id: str, revision: int, payload: AnnotationUpdate) -> dict:
        with store.transaction(immediate=True) as connection:
            row = editable_revision(connection, scene_id, revision)
            if not row["bounds_map_json"]:
                raise HTTPException(status_code=409, detail="scene must be calibrated before annotation")
            bounds = Bounds3D.model_validate_json(row["bounds_map_json"])
            _validate_annotations(payload.annotations, bounds)
            connection.execute(
                "UPDATE scene_revisions SET annotations_json = ? WHERE scene_id = ? AND revision = ?",
                (_json(payload.annotations), scene_id, revision),
            )
            return _revision_result(connection, _revision_row(connection, scene_id, revision))

    @application.post(
        "/api/v1/scenes/{scene_id}/revisions/{revision}/publish",
        dependencies=[Depends(public_auth)],
    )
    def publish_revision(scene_id: str, revision: int) -> dict:
        with store.transaction(immediate=True) as connection:
            row = editable_revision(connection, scene_id, revision)
            if not row["calibration_json"] or not row["bounds_map_json"]:
                raise HTTPException(status_code=409, detail="scene calibration is required before publish")
            import_manifest = _manifest(connection, scene_id, row["import_job_id"])
            if import_manifest.get("schemaVersion") == "h-ide.scene-artifacts/2":
                _map_artifacts(connection, scene_id, revision)
            now = _iso(_now())
            connection.execute(
                "UPDATE scene_revisions SET status = 'published', published_at = ? "
                "WHERE scene_id = ? AND revision = ?",
                (now, scene_id, revision),
            )
            return _revision_result(connection, _revision_row(connection, scene_id, revision))

    @application.put("/api/v1/robots/{robot_id}/scene-binding", dependencies=[Depends(public_auth)])
    def bind_scene(robot_id: str, payload: SceneBindingUpdate) -> dict:
        if not robot_id.strip():
            raise HTTPException(status_code=422, detail="robotId must be non-empty")
        with store.transaction(immediate=True) as connection:
            revision = _require_revision(connection, payload.scene_id, payload.revision)
            if revision["status"] != "published":
                raise HTTPException(status_code=409, detail="only published scene revisions can be bound")
            now = _iso(_now())
            connection.execute(
                """INSERT INTO robot_scene_bindings (robot_id, scene_id, revision, updated_at)
                   VALUES (?, ?, ?, ?) ON CONFLICT(robot_id) DO UPDATE SET
                   scene_id = excluded.scene_id, revision = excluded.revision,
                   updated_at = excluded.updated_at""",
                (robot_id.strip(), payload.scene_id, payload.revision, now),
            )
            return {
                "robotId": robot_id.strip(),
                "sceneId": payload.scene_id,
                "revision": payload.revision,
                "updatedAt": now,
            }

    @application.get("/api/v1/robots/{robot_id}/scene-binding", dependencies=[Depends(public_auth)])
    def get_binding(robot_id: str) -> dict:
        with store.connect() as connection:
            row = connection.execute(
                "SELECT * FROM robot_scene_bindings WHERE robot_id = ?", (robot_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="scene binding not found")
        return {
            "robotId": row["robot_id"],
            "sceneId": row["scene_id"],
            "revision": row["revision"],
            "updatedAt": row["updated_at"],
        }

    @application.get(
        "/internal/v1/robots/{robot_id}/scene-context",
        dependencies=[Depends(internal_auth)],
    )
    def get_internal_scene_context(robot_id: str) -> dict:
        """Return only the immutable map facts that planning is allowed to see."""

        with store.connect() as connection:
            binding = connection.execute(
                "SELECT * FROM robot_scene_bindings WHERE robot_id = ?", (robot_id,)
            ).fetchone()
            if binding is None:
                raise HTTPException(status_code=404, detail="scene binding not found")
            revision = _require_revision(
                connection, binding["scene_id"], binding["revision"]
            )
            map_input_hash = (
                _revision_map_input_hash(
                    connection,
                    revision,
                    json.loads(revision["calibration_json"]),
                )
                if revision["calibration_json"]
                else None
            )
            map_job = (
                _map_job_row(connection, binding["scene_id"], binding["revision"], map_input_hash)
                if map_input_hash
                else None
            )
            map_manifest = (
                json.loads(map_job["artifact_manifest"])
                if map_job and map_job["status"] == "succeeded" and map_job["artifact_manifest"]
                else None
            )
        if revision["status"] != "published":
            raise HTTPException(status_code=409, detail="bound scene revision is not published")
        if not revision["bounds_map_json"]:
            raise HTTPException(status_code=409, detail="bound scene revision is not calibrated")
        return {
            "sceneRef": {
                "sceneId": binding["scene_id"],
                "revision": binding["revision"],
            },
            "bounds": json.loads(revision["bounds_map_json"]),
            "annotations": json.loads(revision["annotations_json"]),
            "mapSummary": map_manifest.get("mapSummary") if map_manifest else None,
            "currentPose": None,
            "selectedPoint": None,
        }

    @application.post(
        "/api/v1/scenes/{scene_id}/revisions/{revision}/asset-session",
        dependencies=[Depends(public_auth)],
    )
    def create_asset_session(scene_id: str, revision: int, response: Response) -> dict:
        with store.transaction(immediate=True) as connection:
            _require_revision(connection, scene_id, revision)
            _merged_public_files(connection, scene_id, revision)
            token = secrets.token_urlsafe(32)
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            created = _now()
            expires = created + timedelta(seconds=ASSET_SESSION_SECONDS)
            connection.execute("DELETE FROM asset_sessions WHERE expires_at <= ?", (_iso(created),))
            connection.execute(
                "INSERT INTO asset_sessions (token_hash, scene_id, revision, expires_at, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (token_hash, scene_id, revision, _iso(expires), _iso(created)),
            )
        cookie_path = f"/scene-assets/{scene_id}/{revision}"
        response.set_cookie(
            ASSET_COOKIE,
            token,
            max_age=ASSET_SESSION_SECONDS,
            expires=expires,
            path=cookie_path,
            secure=asset_cookie_secure,
            httponly=True,
            samesite="strict",
        )
        return {"assetBaseUrl": cookie_path, "expiresAt": _iso(expires)}

    @application.get("/scene-assets/{scene_id}/{revision}/{asset_path:path}")
    def read_asset(scene_id: str, revision: int, asset_path: str, request: Request):
        token = request.cookies.get(ASSET_COOKIE)
        if not token:
            raise HTTPException(status_code=401, detail="asset session required")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with store.connect() as connection:
            session = connection.execute(
                """SELECT * FROM asset_sessions WHERE token_hash = ? AND scene_id = ?
                   AND revision = ? AND expires_at > ?""",
                (token_hash, scene_id, revision, _iso(_now())),
            ).fetchone()
            if session is None:
                raise HTTPException(status_code=401, detail="asset session expired")
            public_files = _merged_public_files(connection, scene_id, revision)
        storage_key = public_files.get(asset_path)
        if storage_key is None:
            raise HTTPException(status_code=404, detail="scene asset not found")
        try:
            path = storage.resolve_artifact_key(storage_key)
        except StorageError as error:
            raise HTTPException(status_code=404, detail="scene asset unavailable") from error
        return FileResponse(path, headers={"Cache-Control": "private, max-age=900"})

    @application.post(
        "/internal/v1/scenes/{scene_id}/revisions/{revision}/target-check",
        dependencies=[Depends(internal_auth)],
    )
    def target_check(scene_id: str, revision: int, payload: TargetCheckRequest) -> dict:
        with store.connect() as connection:
            revision_row, manifest, internal_files = _map_artifacts(connection, scene_id, revision)
        map_key = _manifest_files(manifest, public=True).get("map/map.json")
        grid_key = internal_files.get("map/occupancy.bin")
        if not map_key or not grid_key:
            raise HTTPException(status_code=409, detail="calibrated scene map artifacts are incomplete")
        try:
            metadata_path = storage.resolve_artifact_key(map_key)
            grid_path = storage.resolve_artifact_key(grid_key)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            width = int(metadata["width"])
            height = int(metadata["height"])
            resolution_value = float(metadata["resolution"])
            origin_x = float(metadata["origin"]["x"])
            origin_y = float(metadata["origin"]["y"])
            map_summary = SceneMapSummary.model_validate(metadata["summary"])
            cells = grid_path.read_bytes()
        except (OSError, StorageError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=409, detail="calibrated scene map artifacts are invalid") from error
        if width <= 0 or height <= 0 or len(cells) != width * height or resolution_value <= 0:
            raise HTTPException(status_code=409, detail="calibrated scene grid dimensions are invalid")

        def evidence_warnings(primary: str) -> list[str]:
            values = [primary]
            if map_summary.status == "degraded" or map_summary.confidence == "low":
                values.append("正式地图质量处于降级状态，点云证据置信度较低")
            values.extend(map_summary.warnings)
            return list(dict.fromkeys(values))[:10]

        point = payload.point_map
        bounds = Bounds3D.model_validate_json(revision_row["bounds_map_json"])
        if not bounds.contains(point):
            result = TargetMapCheck(
                mapVersion=map_summary.mapVersion,
                classification="outside",
                clearanceMeters=target_clearance_meters,
                warnings=evidence_warnings("目标点位于已标定场景边界之外"),
            )
            return result.model_dump(mode="json")
        column = math.floor((point.x - origin_x) / resolution_value)
        row = math.floor((point.y - origin_y) / resolution_value)
        if column < 0 or column >= width or row < 0 or row >= height:
            result = TargetMapCheck(
                mapVersion=map_summary.mapVersion,
                classification="outside",
                clearanceMeters=target_clearance_meters,
                warnings=evidence_warnings("目标点位于派生栅格之外"),
            )
            return result.model_dump(mode="json")

        radius_cells = math.ceil(target_clearance_meters / resolution_value) + 1
        minimum_distance: float | None = None
        for candidate_row in range(max(0, row - radius_cells), min(height, row + radius_cells + 1)):
            for candidate_column in range(max(0, column - radius_cells), min(width, column + radius_cells + 1)):
                if cells[candidate_row * width + candidate_column] != 2:
                    continue
                cell_min_x = candidate_column * resolution_value + origin_x
                cell_max_x = cell_min_x + resolution_value
                cell_min_y = candidate_row * resolution_value + origin_y
                cell_max_y = cell_min_y + resolution_value
                distance = math.hypot(
                    max(cell_min_x - point.x, 0.0, point.x - cell_max_x),
                    max(cell_min_y - point.y, 0.0, point.y - cell_max_y),
                )
                if minimum_distance is None or distance < minimum_distance:
                    minimum_distance = distance
        center_class = cells[row * width + column]
        if center_class == 2 or (
            minimum_distance is not None and minimum_distance <= target_clearance_meters
        ):
            classification = "occupied"
            warnings = evidence_warnings("目标点净空范围内检测到点云障碍证据，请人工复核")
        elif center_class == 1:
            classification = "observed_free"
            warnings = evidence_warnings("仅表示派生点云未发现障碍，最终避障仍由真机代理负责")
        else:
            classification = "unknown"
            warnings = evidence_warnings("目标单元缺少充分地面观测，无法由工作站确认可通行")
        result = TargetMapCheck(
            mapVersion=map_summary.mapVersion,
            classification=classification,
            clearanceMeters=target_clearance_meters,
            distanceToObservedObstacleMeters=minimum_distance,
            warnings=warnings,
        )
        return result.model_dump(mode="json")

    @application.delete("/api/v1/scenes/{scene_id}/source", dependencies=[Depends(public_auth)])
    def delete_source(scene_id: str) -> dict:
        with store.transaction(immediate=True) as connection:
            scene = _require_scene(connection, scene_id)
            active_import = _job_row(connection, scene_id)
            if active_import is not None and active_import["status"] in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="raw source is in use by scene processing")
            job = _successful_import_job(connection, scene_id)
            if job is None:
                raise HTTPException(status_code=409, detail="raw source is retained until conversion succeeds")
            asset = connection.execute("SELECT * FROM assets WHERE id = ?", (scene["raw_asset_id"],)).fetchone()
            if asset is None or asset["deleted_at"]:
                raise HTTPException(status_code=409, detail="raw source is already deleted")
            try:
                storage.remove_raw(asset["id"])
            except StorageError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            deleted_at = _iso(_now())
            connection.execute("UPDATE assets SET deleted_at = ? WHERE id = ?", (deleted_at, asset["id"]))
            return {"sceneId": scene_id, "assetId": asset["id"], "status": "deleted", "deletedAt": deleted_at}
