from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from h_ide_vision.api import build_service, create_app
from h_ide_vision.config import Settings
from h_ide_vision.runtime import ResolutionService

from .test_fixture_resolution import request


def settings(tmp_path: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        http_port=8092,
        grpc_host="127.0.0.1",
        grpc_port=50061,
        database_path=tmp_path / "db.sqlite3",
        evidence_dir=tmp_path / "evidence",
        sam2_checkpoint=None,
        sam2_model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_device="cuda",
        qwen_base_url="https://example.invalid/v1",
        qwen_model="qwen-test",
        qwen_api_key=None,
        stable_frame_wait_seconds=1.0,
        agent_token="agent-secret",
        control_token="control-secret",
        allow_insecure_loopback=True,
        tls_cert=None,
        tls_key=None,
        tls_ca=None,
    )


def test_scoped_create_and_read(service: ResolutionService, tmp_path: Path) -> None:
    client = TestClient(create_app(settings=settings(tmp_path), service=service))
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["releaseState"] == "pure_vlm_product_profile"
    capabilities = health.json()["capabilities"]
    assert capabilities["vlm"]["provider"] == "qwen"
    assert capabilities["segmentation"]["provider"] == "sam2"
    assert "foundationpose" not in capabilities
    assert capabilities["visualTaskV2"]["semanticProviderBoundary"] == "qwen_only"
    payload = request().model_dump(mode="json")
    assert client.post("/internal/v1/visual-resolutions", json=payload).status_code == 401
    created = client.post(
        "/internal/v1/visual-resolutions",
        json=payload,
        headers={"Authorization": "Bearer agent-secret"},
    )
    assert created.status_code == 202
    identifier = created.json()["visualResolutionId"]
    read = None
    for _ in range(50):
        read = client.get(
            f"/internal/v1/visual-resolutions/{identifier}",
            headers={"Authorization": "Bearer control-secret"},
        )
        if read.json()["status"] != "queued":
            break
        time.sleep(0.01)
    assert read is not None
    assert read.status_code == 200
    assert read.json()["executable"] is True
    evidence = read.json()["evidence"][0]
    fetched = client.get(
        f"/internal/v1/visual-resolutions/{identifier}/evidence/{evidence['sha256']}",
        headers={"Authorization": "Bearer control-secret"},
    )
    assert fetched.status_code == 200
    assert fetched.headers["content-type"] == evidence["mediaType"]
    assert fetched.headers["cache-control"] == "private, no-store, max-age=0"
    assert client.get(
        f"/internal/v1/visual-resolutions/{identifier}/evidence/{'0' * 64}",
        headers={"Authorization": "Bearer control-secret"},
    ).status_code == 404
    denied = client.post(
        "/internal/v1/visual-resolutions",
        json=payload,
        headers={"Authorization": "Bearer control-secret"},
    )
    assert denied.status_code == 401


def test_product_startup_fails_when_qwen_is_not_configured(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        build_service(settings(tmp_path))


def test_product_startup_fails_when_sam2_is_not_configured(tmp_path: Path) -> None:
    configured = replace(settings(tmp_path), qwen_api_key="configured-test-key")

    with pytest.raises(RuntimeError, match="SAM 2 is unavailable"):
        build_service(configured)


def test_settings_expose_no_runtime_provider_selector(tmp_path: Path) -> None:
    configured = settings(tmp_path)

    assert not hasattr(configured, "vlm_provider")
    assert not hasattr(configured, "segmentation_provider")
    assert not hasattr(configured, "mujoco_oracle_url")
