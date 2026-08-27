from __future__ import annotations

import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
        chunk_size=128,
        disk_reserve_bytes=0,
    )
    return TestClient(app)


def pcd(encoding: str = "ascii", body: bytes = b"0 0 0\n") -> bytes:
    return (
        b"VERSION .7\n"
        b"FIELDS x y z intensity\n"
        b"SIZE 4 4 4 4\n"
        b"TYPE F F F F\n"
        b"COUNT 1 1 1 1\n"
        b"WIDTH 1\nHEIGHT 1\nPOINTS 1\n"
        + f"DATA {encoding}\n".encode()
        + body
    )


def create_upload(client: TestClient, payload: bytes, file_name: str = "scan.PCD") -> dict:
    response = client.post(
        "/api/v1/scene-uploads",
        headers=PUBLIC,
        json={"fileName": file_name, "sizeBytes": len(payload), "lastModified": 12},
    )
    assert response.status_code == 200, response.text
    return response.json()


def patch_upload(client: TestClient, upload_id: str, payload: bytes, offset: int = 0):
    checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode()
    return client.patch(
        f"/api/v1/scene-uploads/{upload_id}",
        headers={
            **PUBLIC,
            "Content-Type": "application/octet-stream",
            "Upload-Offset": str(offset),
            "Upload-Checksum": f"sha256 {checksum}",
        },
        content=payload,
    )


def complete_upload(client: TestClient, payload: bytes, encoding: str = "ascii") -> dict:
    upload = create_upload(client, payload, f"cloud-{encoding}.pcd")
    for offset in range(0, len(payload), 128):
        response = patch_upload(client, upload["uploadId"], payload[offset : offset + 128], offset)
        assert response.status_code == 200, response.text
    response = client.post(
        f"/api/v1/scene-uploads/{upload['uploadId']}/complete", headers=PUBLIC
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_public_and_internal_bearer_tokens_are_distinct(client: TestClient):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    assert client.post("/api/v1/scene-uploads", json={}).status_code == 401
    assert client.post("/api/v1/scene-uploads", headers=INTERNAL, json={}).status_code == 401
    assert client.post("/internal/v1/scene-jobs/claim", headers=PUBLIC, json={}).status_code == 401


@pytest.mark.parametrize(
    ("file_name", "size", "status"),
    [
        ("scan.txt", 10, 422),
        ("../scan.pcd", 10, 422),
        ("folder/scan.pcd", 10, 422),
        ("scan.pcd", 0, 422),
        ("scan.pcd", 1024 * 1024 + 1, 422),
    ],
)
def test_upload_creation_rejects_invalid_names_and_sizes(
    client: TestClient, file_name: str, size: int, status: int
):
    response = client.post(
        "/api/v1/scene-uploads",
        headers=PUBLIC,
        json={"fileName": file_name, "sizeBytes": size},
    )
    assert response.status_code == status


def test_upload_creation_reports_limits_and_checks_disk_space(client: TestClient, monkeypatch):
    monkeypatch.setattr(client.app.state.storage, "free_bytes", lambda: 39)
    response = client.post(
        "/api/v1/scene-uploads",
        headers=PUBLIC,
        json={"fileName": "scan.PcD", "sizeBytes": 10},
    )
    assert response.status_code == 507

    monkeypatch.setattr(client.app.state.storage, "free_bytes", lambda: 40)
    result = client.post(
        "/api/v1/scene-uploads",
        headers=PUBLIC,
        json={"fileName": "scan.PcD", "sizeBytes": 10},
    ).json()
    assert result["status"] == "active"
    assert result["offset"] == 0
    assert result["chunkSize"] == 128
    assert result["expiresAt"]


def test_upload_resume_enforces_offset_checksum_and_chunk_bounds(client: TestClient):
    upload = create_upload(client, b"abcdef")
    upload_id = upload["uploadId"]

    assert patch_upload(client, upload_id, b"abc", 1).status_code == 409
    wrong = client.patch(
        f"/api/v1/scene-uploads/{upload_id}",
        headers={
            **PUBLIC,
            "Content-Type": "application/octet-stream",
            "Upload-Offset": "0",
            "Upload-Checksum": "sha256 AAAA",
        },
        content=b"abc",
    )
    assert wrong.status_code == 422
    assert patch_upload(client, upload_id, b"").status_code == 422
    assert patch_upload(client, upload_id, b"x" * 129).status_code == 422

    assert patch_upload(client, upload_id, b"abc").json()["offset"] == 3
    assert client.get(f"/api/v1/scene-uploads/{upload_id}", headers=PUBLIC).json()["offset"] == 3
    assert patch_upload(client, upload_id, b"defg", 3).status_code == 422
    assert patch_upload(client, upload_id, b"def", 3).json()["offset"] == 6


def test_upload_reconciles_durable_offset_with_the_part_file(client: TestClient):
    upload = create_upload(client, b"abcdef")
    upload_id = upload["uploadId"]
    assert patch_upload(client, upload_id, b"abc").status_code == 200
    with client.app.state.store.transaction(immediate=True) as db:
        db.execute("UPDATE uploads SET offset = 0 WHERE id = ?", (upload_id,))

    assert patch_upload(client, upload_id, b"def", 0).status_code == 409
    status = client.get(f"/api/v1/scene-uploads/{upload_id}", headers=PUBLIC).json()
    assert status["offset"] == 3


@pytest.mark.parametrize("encoding", ["ascii", "binary", "binary_compressed"])
def test_completion_is_exact_idempotent_and_retains_raw(client: TestClient, encoding: str):
    payload = pcd(encoding, b"dummy compressed or binary bytes")
    upload = create_upload(client, payload)
    upload_id = upload["uploadId"]
    assert client.post(f"/api/v1/scene-uploads/{upload_id}/complete", headers=PUBLIC).status_code == 409
    assert patch_upload(client, upload_id, payload).status_code == 422  # over configured chunk
    for offset in range(0, len(payload), 128):
        assert patch_upload(client, upload_id, payload[offset : offset + 128], offset).status_code == 200

    first = client.post(f"/api/v1/scene-uploads/{upload_id}/complete", headers=PUBLIC)
    second = client.post(f"/api/v1/scene-uploads/{upload_id}/complete", headers=PUBLIC)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.json()["dataEncoding"] == encoding
    assert first.json()["status"] == "queued"
    raw = client.app.state.storage.raw_path(first.json()["assetId"])
    assert raw.read_bytes() == payload


@pytest.mark.parametrize(
    "payload",
    [
        b"FIELDS x y z\nSIZE 4 4 4\nCOUNT 1 1 1\nPOINTS 1\n",
        b"FIELDS x y z\nSIZE 4 4 4\nCOUNT 1 1 1\nPOINTS 1\nDATA ascii\nDATA binary\n",
        b"FIELDS x y\nSIZE 4 4\nCOUNT 1 1\nPOINTS 1\nDATA ascii\n",
        b"FIELDS x y z\nSIZE 4 4 4\nCOUNT 1 1 1\nPOINTS nope\nDATA ascii\n",
        b"FIELDS x y z\nSIZE 4 4 4\nCOUNT 1 1 1\nPOINTS 1\nDATA magical\n",
        b"#" * (64 * 1024 + 1),
    ],
)
def test_completion_rejects_malformed_or_bomb_headers(client: TestClient, payload: bytes):
    upload = create_upload(client, payload)
    for offset in range(0, len(payload), 128):
        assert patch_upload(client, upload["uploadId"], payload[offset : offset + 128], offset).status_code == 200
    response = client.post(f"/api/v1/scene-uploads/{upload['uploadId']}/complete", headers=PUBLIC)
    assert response.status_code == 422


def test_cancel_and_cleanup_remove_only_exact_regular_part_files(client: TestClient):
    first = create_upload(client, b"one")
    second = create_upload(client, b"two")
    assert patch_upload(client, first["uploadId"], b"one").status_code == 200
    assert patch_upload(client, second["uploadId"], b"two").status_code == 200
    first_part = client.app.state.storage.part_path(first["uploadId"])
    second_part = client.app.state.storage.part_path(second["uploadId"])

    assert client.post(f"/api/v1/scene-uploads/{first['uploadId']}/cancel", headers=PUBLIC).status_code == 200
    assert not first_part.exists()
    assert second_part.exists()
    assert client.post(f"/api/v1/scene-uploads/{first['uploadId']}/cancel", headers=PUBLIC).status_code == 409

    with client.app.state.store.transaction(immediate=True) as db:
        db.execute("UPDATE uploads SET expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (second["uploadId"],))
    result = client.post("/internal/v1/scene-uploads/cleanup", headers=INTERNAL)
    assert result.json() == {"expired": 1}
    assert not second_part.exists()


def test_job_claim_is_atomic_and_lease_transitions_are_durable(client: TestClient):
    completed = complete_upload(client, pcd())
    job_id = completed["jobId"]

    def claim():
        return client.post(
            "/internal/v1/scene-jobs/claim",
            headers=INTERNAL,
            json={"workerId": "worker-a", "leaseSeconds": 30},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: claim(), range(2)))
    assert sorted(response.status_code for response in responses) == [200, 204]
    claimed = next(response.json() for response in responses if response.status_code == 200)
    assert claimed["jobId"] == job_id
    assert claimed["attempts"] == 1

    assert client.post(
        f"/internal/v1/scene-jobs/{job_id}/heartbeat",
        headers=INTERNAL,
        json={"workerId": "wrong", "leaseSeconds": 30},
    ).status_code == 409
    assert client.post(
        f"/internal/v1/scene-jobs/{job_id}/heartbeat",
        headers=INTERNAL,
        json={"workerId": "worker-a", "leaseSeconds": 60},
    ).status_code == 200
    assert client.post(
        f"/internal/v1/scene-jobs/{job_id}/fail",
        headers=INTERNAL,
        json={"workerId": "worker-a", "deterministic": False, "errorCode": "tool_exit", "message": "safe"},
    ).json()["status"] == "queued"

    assert claim().json()["attempts"] == 2
    assert client.post(
        f"/internal/v1/scene-jobs/{job_id}/fail",
        headers=INTERNAL,
        json={"workerId": "worker-a", "deterministic": False, "errorCode": "tool_exit", "message": "safe"},
    ).json()["status"] == "failed"
    public = client.get(f"/api/v1/scene-jobs/{job_id}", headers=PUBLIC).json()
    assert public["status"] == "failed"
    assert client.post(f"/api/v1/scene-jobs/{job_id}/retry", headers=PUBLIC).json()["status"] == "queued"


def test_job_complete_cancel_retry_and_expired_lease(client: TestClient):
    first = complete_upload(client, pcd())
    first_id = first["jobId"]
    claimed = client.post(
        "/internal/v1/scene-jobs/claim",
        headers=INTERNAL,
        json={"workerId": "worker", "leaseSeconds": 30},
    ).json()
    assert claimed["jobId"] == first_id
    artifact_dir = client.app.state.storage.artifact_root / first["sceneId"] / first_id
    artifact_dir.mkdir(parents=True)
    files: dict[str, str] = {}
    for name, content in {
        "metadata.json": b'{"version":"2.0"}',
        "hierarchy.bin": b"hierarchy",
        "octree.bin": b"octree",
    }.items():
        target = artifact_dir / name
        target.write_bytes(content)
        files[name] = str(target.relative_to(client.app.state.storage.root))
    response = client.post(
        f"/internal/v1/scene-jobs/{first_id}/complete",
        headers=INTERNAL,
        json={
            "workerId": "worker",
            "artifactManifest": {
                "provider": "potree",
                "files": files,
                "sourceBounds": {
                    "min": {"x": 0, "y": 0, "z": 0},
                    "max": {"x": 1, "y": 1, "z": 1},
                },
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "succeeded"
    assert client.post(f"/api/v1/scene-jobs/{first_id}/retry", headers=PUBLIC).status_code == 409

    second = complete_upload(client, pcd(body=b"1 2 3\n"))
    assert client.post(f"/api/v1/scene-jobs/{second['jobId']}/cancel", headers=PUBLIC).json()["status"] == "cancelled"

    third = complete_upload(client, pcd(body=b"4 5 6\n"))
    claimed = client.post(
        "/internal/v1/scene-jobs/claim",
        headers=INTERNAL,
        json={"workerId": "old-worker", "leaseSeconds": 30},
    ).json()
    assert claimed["jobId"] == third["jobId"]
    with client.app.state.store.transaction(immediate=True) as db:
        db.execute("UPDATE jobs SET lease_expires_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (third["jobId"],))
    reclaimed = client.post(
        "/internal/v1/scene-jobs/claim",
        headers=INTERNAL,
        json={"workerId": "new-worker", "leaseSeconds": 30},
    ).json()
    assert reclaimed["jobId"] == third["jobId"]
    assert reclaimed["attempts"] == 2
