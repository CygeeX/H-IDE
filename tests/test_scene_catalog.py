from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.scene_core import identity_calibration
from services.scene_api.main import create_app


PUBLIC = {"Authorization": "Bearer public-secret"}
INTERNAL = {"Authorization": "Bearer internal-secret"}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    app = create_app(
        tmp_path / "scene.sqlite3",
        tmp_path / "storage",
        "public-secret",
        "internal-secret",
        max_upload_bytes=1024 * 1024,
        chunk_size=1024,
        disk_reserve_bytes=0,
    )
    return TestClient(app)


def _pcd() -> bytes:
    return (
        b"VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        b"COUNT 1 1 1\nWIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA ascii\n0 0 0\n"
    )


def _upload(client: TestClient) -> dict:
    payload = _pcd()
    created = client.post(
        "/api/v1/scene-uploads",
        headers=PUBLIC,
        json={"fileName": "room.pcd", "sizeBytes": len(payload)},
    ).json()
    checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode()
    patched = client.patch(
        f"/api/v1/scene-uploads/{created['uploadId']}",
        headers={
            **PUBLIC,
            "Content-Type": "application/octet-stream",
            "Upload-Offset": "0",
            "Upload-Checksum": f"sha256 {checksum}",
        },
        content=payload,
    )
    assert patched.status_code == 200
    completed = client.post(
        f"/api/v1/scene-uploads/{created['uploadId']}/complete", headers=PUBLIC
    )
    assert completed.status_code == 200
    return completed.json()


def _finish_conversion(client: TestClient) -> dict:
    uploaded = _upload(client)
    job = client.post(
        "/internal/v1/scene-jobs/claim",
        headers=INTERNAL,
        json={"workerId": "worker", "leaseSeconds": 60},
    ).json()
    artifact_dir = (
        client.app.state.storage.artifact_root / uploaded["sceneId"] / uploaded["jobId"]
    )
    artifact_dir.mkdir(parents=True)
    contents = {
        "metadata.json": b'{"version":"2.0"}',
        "hierarchy.bin": b"hierarchy",
        "octree.bin": b"0123456789",
    }
    files: dict[str, str] = {}
    for name, value in contents.items():
        target = artifact_dir / name
        target.write_bytes(value)
        files[name] = str(target.relative_to(client.app.state.storage.root))
    response = client.post(
        f"/internal/v1/scene-jobs/{job['jobId']}/complete",
        headers=INTERNAL,
        json={
            "workerId": "worker",
            "artifactManifest": {
                "provider": "potree",
                "files": files,
                "sourceBounds": {
                    "min": {"x": -10, "y": -10, "z": -1},
                    "max": {"x": 10, "y": 10, "z": 3},
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    return uploaded


def _artifact_key(client: TestClient, scene_id: str, job_id: str, name: str, content: bytes) -> str:
    target = client.app.state.storage.artifact_path(scene_id, job_id, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target.relative_to(client.app.state.storage.root).as_posix()


def _v2_import_manifest(client: TestClient, scene_id: str, job_id: str) -> dict:
    public_contents = {
        "metadata.json": json.dumps({"version": "2.0", "job": job_id}).encode(),
        "hierarchy.bin": b"hierarchy",
        "octree.bin": b"octree",
        "scene-summary.json": b"{}",
        "preview/map.json": b"{}",
        "preview/occupancy.png": b"png",
        "preview/height.png": b"png",
    }
    internal_contents = {
        "analysis.ply": b"ply",
        "preview/occupancy.bin": b"\x01",
    }
    return {
        "schemaVersion": "h-ide.scene-artifacts/2",
        "provider": "h-ide-scene",
        "jobKind": "import",
        "publicFiles": {
            name: _artifact_key(client, scene_id, job_id, name, content)
            for name, content in public_contents.items()
        },
        "internalFiles": {
            name: _artifact_key(client, scene_id, job_id, name, content)
            for name, content in internal_contents.items()
        },
        "sourceBounds": {
            "min": {"x": 0, "y": 0, "z": -1},
            "max": {"x": 3, "y": 3, "z": 2},
        },
        "processingSummary": {"voxelSize": 0.05, "sourcePointCount": 100},
        "mapSummary": {
            "schemaVersion": "h-ide.scene-map/1",
            "mapVersion": f"preview-{job_id}",
            "frameId": "scene",
            "verified": False,
            "status": "degraded",
            "resolution": 0.1,
            "groundZ": 0,
            "observedCoverageRatio": 0.5,
            "confidence": "medium",
            "warnings": ["unverified"],
        },
    }


def _finish_v2_conversion(client: TestClient) -> dict:
    uploaded = _upload(client)
    job = client.post(
        "/internal/v1/scene-jobs/claim",
        headers=INTERNAL,
        json={"workerId": "worker-v2", "leaseSeconds": 60},
    ).json()
    response = client.post(
        f"/internal/v1/scene-jobs/{job['jobId']}/complete",
        headers=INTERNAL,
        json={
            "workerId": "worker-v2",
            "artifactManifest": _v2_import_manifest(
                client, uploaded["sceneId"], job["jobId"]
            ),
        },
    )
    assert response.status_code == 200, response.text
    return uploaded


def _kitchen(name: str = "厨房", aliases: list[str] | None = None) -> dict:
    return {
        "locationId": "kitchen",
        "name": name,
        "aliases": aliases or ["kitchen"],
        "polygonXY": [
            {"x": 1, "y": 1},
            {"x": 4, "y": 1},
            {"x": 4, "y": 4},
            {"x": 1, "y": 4},
        ],
        "zMin": 0,
        "zMax": 2.5,
        "entryPose": {"frameId": "map", "x": 1, "y": 2, "z": 0, "yaw": 0},
    }


def _publish_scene(client: TestClient) -> tuple[dict, dict]:
    uploaded = _finish_conversion(client)
    scene = client.get(f"/api/v1/scenes/{uploaded['sceneId']}", headers=PUBLIC)
    assert scene.status_code == 200, scene.text
    revision = scene.json()["revisions"][0]
    assert revision["status"] == "draft"

    calibration = identity_calibration().model_dump(mode="json")
    updated = client.put(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision['revision']}/calibration",
        headers=PUBLIC,
        json={"calibration": calibration},
    )
    assert updated.status_code == 200, updated.text
    annotated = client.put(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision['revision']}/annotations",
        headers=PUBLIC,
        json={"annotations": [_kitchen()]},
    )
    assert annotated.status_code == 200, annotated.text
    published = client.post(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision['revision']}/publish",
        headers=PUBLIC,
    )
    assert published.status_code == 200, published.text
    return uploaded, published.json()


def test_conversion_creates_editable_revision_and_published_versions_are_immutable(client: TestClient):
    uploaded, published = _publish_scene(client)
    assert published["status"] == "published"
    assert published["boundsMap"]["min"]["x"] == -10

    immutable = client.put(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{published['revision']}/annotations",
        headers=PUBLIC,
        json={"annotations": []},
    )
    assert immutable.status_code == 409

    cloned = client.post(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions",
        headers=PUBLIC,
        json={"baseRevision": published["revision"]},
    )
    assert cloned.status_code == 200, cloned.text
    assert cloned.json()["revision"] == published["revision"] + 1
    assert cloned.json()["annotations"][0]["locationId"] == "kitchen"


def test_annotation_collisions_and_out_of_bounds_entries_are_rejected(client: TestClient):
    uploaded = _finish_conversion(client)
    revision = client.get(
        f"/api/v1/scenes/{uploaded['sceneId']}", headers=PUBLIC
    ).json()["revisions"][0]["revision"]
    assert client.post(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision}/asset-session",
        headers=PUBLIC,
    ).status_code == 200
    client.put(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision}/calibration",
        headers=PUBLIC,
        json={"calibration": identity_calibration().model_dump(mode="json")},
    )
    collision = _kitchen()
    other = _kitchen(name="储藏室", aliases=["KITCHEN"])
    other["locationId"] = "storage"
    assert client.put(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision}/annotations",
        headers=PUBLIC,
        json={"annotations": [collision, other]},
    ).status_code == 422

    outside = _kitchen()
    outside["entryPose"]["x"] = 100
    assert client.put(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision}/annotations",
        headers=PUBLIC,
        json={"annotations": [outside]},
    ).status_code == 422


def test_interactive_calibration_fit_persists_similarity_transform(client: TestClient):
    uploaded = _finish_conversion(client)
    revision = client.get(
        f"/api/v1/scenes/{uploaded['sceneId']}", headers=PUBLIC
    ).json()["revisions"][0]["revision"]
    fitted = client.post(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{revision}/calibration/fit",
        headers=PUBLIC,
        json={
            "groundPoints": [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
            "originScene": [0, 0, 0],
            "xAxisScene": [2, 0, 0],
            "knownDistanceM": 4,
            "mapOrigin": [10, 20, 0],
        },
    )
    assert fitted.status_code == 200, fitted.text
    body = fitted.json()
    assert body["calibration"]["mode"] == "interactive_similarity"
    assert body["calibration"]["scaleMetersPerUnit"] == 2
    assert body["boundsMap"] == {
        "min": {"x": -10.0, "y": 0.0, "z": -2.0},
        "max": {"x": 30.0, "y": 40.0, "z": 6.0},
    }


def test_robot_binding_requires_published_revision_and_does_not_auto_switch(client: TestClient):
    uploaded, published = _publish_scene(client)
    bound = client.put(
        "/api/v1/robots/robot-1/scene-binding",
        headers=PUBLIC,
        json={"sceneId": uploaded["sceneId"], "revision": published["revision"]},
    )
    assert bound.status_code == 200
    cloned = client.post(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions",
        headers=PUBLIC,
        json={"baseRevision": published["revision"]},
    ).json()
    binding = client.get("/api/v1/robots/robot-1/scene-binding", headers=PUBLIC).json()
    assert binding["revision"] == published["revision"]
    assert binding["revision"] != cloned["revision"]
    assert client.get(
        "/internal/v1/robots/robot-1/scene-context", headers=PUBLIC
    ).status_code == 401
    context = client.get(
        "/internal/v1/robots/robot-1/scene-context", headers=INTERNAL
    )
    assert context.status_code == 200, context.text
    assert context.json()["sceneRef"] == {
        "sceneId": uploaded["sceneId"],
        "revision": published["revision"],
    }
    assert context.json()["annotations"][0]["locationId"] == "kitchen"
    assert "calibration" not in context.text
    assert "asset" not in context.text.lower()


def test_asset_session_protects_files_and_supports_range_requests(client: TestClient):
    uploaded, published = _publish_scene(client)
    anonymous = TestClient(client.app)
    asset_url = f"/scene-assets/{uploaded['sceneId']}/{published['revision']}/octree.bin"
    assert anonymous.get(asset_url).status_code == 401

    session = client.post(
        f"/api/v1/scenes/{uploaded['sceneId']}/revisions/{published['revision']}/asset-session",
        headers=PUBLIC,
    )
    assert session.status_code == 200, session.text
    assert session.json()["assetBaseUrl"].endswith(
        f"/{uploaded['sceneId']}/{published['revision']}"
    )
    ranged = client.get(asset_url, headers={"Range": "bytes=2-5"})
    assert ranged.status_code == 206
    assert ranged.content == b"2345"
    assert ranged.headers["content-range"] == "bytes 2-5/10"

    with client.app.state.store.transaction(immediate=True) as db:
        db.execute("UPDATE asset_sessions SET expires_at = '2000-01-01T00:00:00+00:00'")
    assert client.get(asset_url).status_code == 401


def test_raw_source_is_retained_until_explicit_delete(client: TestClient):
    uploaded, _ = _publish_scene(client)
    raw = client.app.state.storage.raw_path(uploaded["assetId"])
    assert raw.exists()
    deleted = client.delete(
        f"/api/v1/scenes/{uploaded['sceneId']}/source", headers=PUBLIC
    )
    assert deleted.status_code == 200, deleted.text
    assert not raw.exists()
    assert client.delete(
        f"/api/v1/scenes/{uploaded['sceneId']}/source", headers=PUBLIC
    ).status_code == 409


def test_v2_calibration_queues_formal_map_and_target_check_returns_review_evidence(
    client: TestClient,
):
    uploaded = _finish_v2_conversion(client)
    scene_id = uploaded["sceneId"]
    scene = client.get(f"/api/v1/scenes/{scene_id}", headers=PUBLIC).json()
    assert scene["previewAvailable"] is True
    revision = scene["revisions"][0]["revision"]
    calibrated = client.put(
        f"/api/v1/scenes/{scene_id}/revisions/{revision}/calibration",
        headers=PUBLIC,
        json={"calibration": identity_calibration().model_dump(mode="json")},
    )
    assert calibrated.status_code == 200, calibrated.text
    assert calibrated.json()["mapJob"]["status"] == "queued"

    map_job = client.post(
        "/internal/v1/scene-jobs/claim",
        headers=INTERNAL,
        json={"workerId": "map-worker", "leaseSeconds": 60},
    ).json()
    assert map_job["jobKind"] == "map_derivation"
    summary = {
        "schemaVersion": "h-ide.scene-map/1",
        "mapVersion": "formal-map-test",
        "frameId": "map",
        "verified": True,
        "status": "ready",
        "resolution": 1,
        "groundZ": 0,
        "observedCoverageRatio": 2 / 9,
        "confidence": "high",
        "warnings": [],
    }
    metadata = {
        "schemaVersion": "h-ide.scene-map/1",
        "mapVersion": summary["mapVersion"],
        "frameId": "map",
        "verified": True,
        "status": "ready",
        "resolution": 1,
        "origin": {"x": 0, "y": 0},
        "width": 3,
        "height": 3,
        "groundZ": 0,
        "summary": summary,
    }
    public = {
        "map/map.json": _artifact_key(
            client, scene_id, map_job["jobId"], "map/map.json", json.dumps(metadata).encode()
        ),
        "map/occupancy.png": _artifact_key(
            client, scene_id, map_job["jobId"], "map/occupancy.png", b"png"
        ),
        "map/height.png": _artifact_key(
            client, scene_id, map_job["jobId"], "map/height.png", b"png"
        ),
    }
    # Binary rows increase with map Y: free at (0,0), occupied at (1,1).
    internal = {
        "map/occupancy.bin": _artifact_key(
            client, scene_id, map_job["jobId"], "map/occupancy.bin", bytes([1, 0, 0, 0, 2, 0, 0, 0, 0])
        )
    }
    completed = client.post(
        f"/internal/v1/scene-jobs/{map_job['jobId']}/complete",
        headers=INTERNAL,
        json={
            "workerId": "map-worker",
            "artifactManifest": {
                "schemaVersion": "h-ide.scene-artifacts/2",
                "provider": "h-ide-map",
                "jobKind": "map_derivation",
                "publicFiles": public,
                "internalFiles": internal,
                "sourceBounds": {
                    "min": {"x": 0, "y": 0, "z": -1},
                    "max": {"x": 3, "y": 3, "z": 2},
                },
                "mapSummary": summary,
                "inputHash": map_job["inputHash"],
            },
        },
    )
    assert completed.status_code == 200, completed.text

    occupied = client.post(
        f"/internal/v1/scenes/{scene_id}/revisions/{revision}/target-check",
        headers=INTERNAL,
        json={"pointMap": {"x": 1.5, "y": 1.5, "z": 0}},
    )
    assert occupied.status_code == 200, occupied.text
    assert occupied.json()["classification"] == "occupied"
    free = client.post(
        f"/internal/v1/scenes/{scene_id}/revisions/{revision}/target-check",
        headers=INTERNAL,
        json={"pointMap": {"x": 0.25, "y": 0.25, "z": 0}},
    )
    assert free.json()["classification"] == "observed_free"


def test_reprocessing_creates_a_new_pinned_draft_without_mutating_old_revision(
    client: TestClient,
):
    uploaded = _finish_v2_conversion(client)
    scene_id = uploaded["sceneId"]
    before = client.get(f"/api/v1/scenes/{scene_id}", headers=PUBLIC).json()
    first = before["revisions"][0]
    first_import_job_id = first["importJobId"]

    queued = client.post(f"/api/v1/scenes/{scene_id}/reprocess", headers=PUBLIC)
    assert queued.status_code == 200, queued.text
    claimed = client.post(
        "/internal/v1/scene-jobs/claim",
        headers=INTERNAL,
        json={"workerId": "reprocess-worker", "leaseSeconds": 60},
    ).json()
    assert claimed["jobId"] == queued.json()["jobId"]
    completed = client.post(
        f"/internal/v1/scene-jobs/{claimed['jobId']}/complete",
        headers=INTERNAL,
        json={
            "workerId": "reprocess-worker",
            "artifactManifest": _v2_import_manifest(client, scene_id, claimed["jobId"]),
        },
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["revision"] == 2

    after = client.get(f"/api/v1/scenes/{scene_id}", headers=PUBLIC).json()
    old_revision, new_revision = after["revisions"]
    assert old_revision["importJobId"] == first_import_job_id
    assert new_revision["importJobId"] == claimed["jobId"]
    assert new_revision["baseRevision"] == old_revision["revision"]
    assert new_revision["status"] == "draft"

    assert client.post(
        f"/api/v1/scenes/{scene_id}/revisions/{old_revision['revision']}/asset-session",
        headers=PUBLIC,
    ).status_code == 200
    old_metadata = client.get(
        f"/scene-assets/{scene_id}/{old_revision['revision']}/metadata.json"
    ).json()
    assert client.post(
        f"/api/v1/scenes/{scene_id}/revisions/{new_revision['revision']}/asset-session",
        headers=PUBLIC,
    ).status_code == 200
    new_metadata = client.get(
        f"/scene-assets/{scene_id}/{new_revision['revision']}/metadata.json"
    ).json()
    assert old_metadata["job"] == first_import_job_id
    assert new_metadata["job"] == claimed["jobId"]
