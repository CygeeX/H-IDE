from __future__ import annotations

import json
from pathlib import Path

import httpx
import yaml

from packages.simulation_core.mujoco_client import MujocoSonicClient


ROOT = Path(__file__).resolve().parents[1]


class _ReadOnlyFixture:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str) -> httpx.Response:
        self.urls.append(url)
        request = httpx.Request("GET", url)
        if url.endswith(":8792/healthz"):
            return httpx.Response(
                200,
                request=request,
                json={"status": "running", "motion_ready": True},
            )
        if url.endswith(":8794/healthz"):
            return httpx.Response(
                200,
                request=request,
                json={
                    "status": "running",
                    "sequence": 12,
                    "pose": {"x": 1.0, "y": 2.0, "z": 0.8, "qw": 1.0},
                },
            )
        if url.endswith(":8794/rgb.jpg"):
            return httpx.Response(
                200,
                request=request,
                content=b"\xff\xd8vlm-evidence\xff\xd9",
                headers={"content-type": "image/jpeg"},
            )
        raise AssertionError(f"unexpected non-observation URL: {url}")


def test_simulation_overlay_exposes_observations_but_no_command_port() -> None:
    overlay = yaml.safe_load(
        (ROOT / "infra/docker-compose.simulation.yml").read_text(encoding="utf-8")
    )
    control_api = overlay["services"]["control-api"]
    environment = control_api["environment"]

    assert "host.docker.internal:host-gateway" in control_api["extra_hosts"]
    assert "host.docker.internal:8792" in environment["HIDE_MUJOCO_SONIC_TELEMETRY_URL"]
    assert "host.docker.internal:8794" in environment["HIDE_MUJOCO_SONIC_PLANT_URL"]
    assert "127.0.0.1:8794" in environment["HIDE_MUJOCO_SONIC_PUBLIC_PLANT_URL"]
    assert "8793" not in json.dumps(control_api)
    assert "network_mode" not in control_api


def test_control_api_client_reads_backend_health_live_state_and_rgb_only() -> None:
    fixture = _ReadOnlyFixture()
    client = MujocoSonicClient(
        telemetry_url="http://host.docker.internal:8792",
        plant_url="http://host.docker.internal:8794",
        public_plant_url="http://127.0.0.1:8794",
        client=fixture,
    )

    live = client.live()
    rgb = client.rgb_frame()

    assert live["ready"] is True
    assert live["rgbUrl"] == "http://127.0.0.1:8794/rgb.jpg"
    assert live["pose"]["x"] == 1.0
    assert rgb.startswith(b"\xff\xd8")
    assert fixture.urls
    assert all(url.endswith(("/healthz", "/rgb.jpg")) for url in fixture.urls)
    assert all(":8793" not in url for url in fixture.urls)
