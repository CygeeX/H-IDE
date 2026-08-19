from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import threading

import pytest

from hide_mujoco_sonic.network import (
    ObservationBindError,
    require_loopback_bind_host,
    resolve_docker_observation_bind_host,
)
from hide_mujoco_sonic.config import SimConfig
from hide_mujoco_sonic.plant import PlantFrameStore, PlantSupportController
from hide_mujoco_sonic.sonic_driver import _bind_command_server
from hide_mujoco_sonic.telemetry import TelemetryState, TelemetryStore, serve_telemetry
from hide_mujoco_sonic.runtime_diagnostics import PlantRuntimeDiagnostics
from hide_mujoco_sonic.transport import request_json


ROOT = Path(__file__).resolve().parents[1]


def _docker_bridge(*, gateway: str = "172.31.0.1") -> dict[str, object]:
    return {
        "Name": "bridge",
        "Driver": "bridge",
        "Scope": "local",
        "Internal": False,
        "IPAM": {"Config": [{"Subnet": "172.31.0.0/16", "Gateway": gateway}]},
        "Options": {"com.docker.network.bridge.name": "docker-test"},
    }


def _dual_loopback_port() -> int:
    # Reserve the same candidate on two distinct loopback addresses before
    # handing it to the dual-listener code. No simulator or motion process is
    # involved in these HTTP-only tests.
    for _attempt in range(20):
        first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            first.bind(("127.0.0.1", 0))
            port = int(first.getsockname()[1])
            second.bind(("127.0.0.2", port))
            return port
        except OSError:
            continue
        finally:
            first.close()
            second.close()
    raise RuntimeError("could not reserve a dual-loopback test port")


def _request(host: str, port: int, method: str, path: str, body: bytes | None = None):
    connection = HTTPConnection(host, port, timeout=2)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_loopback_command_request_requires_a_matching_positive_ack() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])
    received: list[dict[str, object]] = []

    def respond() -> None:
        connection, _address = server.accept()
        with connection:
            raw = b""
            while b"\n" not in raw:
                raw += connection.recv(4096)
            payload = json.loads(raw.split(b"\n", 1)[0])
            received.append(payload)
            connection.sendall(
                (
                    json.dumps(
                        {
                            "request_id": payload["request_id"],
                            "ok": True,
                            "mode": "idle",
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )

    worker = threading.Thread(target=respond, daemon=True)
    worker.start()
    try:
        response = request_json(
            "127.0.0.1",
            port,
            {"type": "test", "request_id": "request-7"},
            timeout=1.0,
        )
    finally:
        worker.join(timeout=2)
        server.close()

    assert worker.is_alive() is False
    assert received == [{"type": "test", "request_id": "request-7"}]
    assert response == {"request_id": "request-7", "ok": True, "mode": "idle"}


def test_docker_gateway_resolution_is_exact_and_interface_bound() -> None:
    resolved = resolve_docker_observation_bind_host(
        "auto",
        inspect_bridge=_docker_bridge,
        interface_ipv4=lambda name: "172.31.0.1" if name == "docker-test" else "",
    )
    assert resolved == "172.31.0.1"

    with pytest.raises(ObservationBindError, match="does not match host-gateway"):
        resolve_docker_observation_bind_host(
            "192.168.1.20",
            inspect_bridge=_docker_bridge,
            interface_ipv4=lambda _name: "172.31.0.1",
        )
    with pytest.raises(ObservationBindError, match="owns"):
        resolve_docker_observation_bind_host(
            "auto",
            inspect_bridge=_docker_bridge,
            interface_ipv4=lambda _name: "172.31.0.2",
        )


def test_all_primary_binds_and_command_socket_remain_loopback() -> None:
    assert require_loopback_bind_host("127.0.0.1", field="plant") == "127.0.0.1"
    with pytest.raises(ObservationBindError, match="loopback-only"):
        require_loopback_bind_host("0.0.0.0", field="command")
    with pytest.raises(ObservationBindError, match="loopback-only"):
        _bind_command_server("172.31.0.1", 0)

    server = _bind_command_server("127.0.0.1", 0)
    try:
        port = int(server.getsockname()[1])
        with socket.create_connection(("127.0.0.1", port), timeout=1) as connection:
            assert connection.getpeername() == ("127.0.0.1", port)
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        try:
            assert probe.connect_ex(("127.0.0.2", port)) != 0
        finally:
            probe.close()
    finally:
        server.close()


def test_product_config_and_launcher_keep_the_dual_bind_boundary(monkeypatch) -> None:
    monkeypatch.delenv("HIDE_DOCKER_OBSERVATION_BIND_HOST", raising=False)
    config = SimConfig.from_file(ROOT / "config/ros-headless.yaml")
    assert config.telemetry_bind_host == "127.0.0.1"
    assert config.plant_bind_host == "127.0.0.1"
    assert config.command_bind_host == "127.0.0.1"
    assert config.docker_observation_bind_host is None

    launcher = (ROOT / "scripts/start-workstation-demo.sh").read_text(encoding="utf-8")
    assert "-m hide_mujoco_sonic.network" in launcher
    assert '--setenv="HIDE_DOCKER_OBSERVATION_BIND_HOST=' in launcher
    assert "172.17.0.1" not in launcher


def test_telemetry_health_is_available_on_loopback_and_docker_listener() -> None:
    port = _dual_loopback_port()
    store = TelemetryStore(TelemetryState(robot_id="g1-sim", status="running"))
    servers = serve_telemetry(
        store,
        "127.0.0.1",
        port,
        docker_observation_host="127.0.0.2",
    )
    try:
        for host in ("127.0.0.1", "127.0.0.2"):
            status, _headers, body = _request(host, port, "GET", "/healthz")
            assert status == 200
            assert json.loads(body)["status"] == "running"
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_docker_plant_listener_serves_live_rgb_but_rejects_support_write() -> None:
    port = _dual_loopback_port()
    support = PlantSupportController()
    support.finish_startup()
    runtime_diagnostics = PlantRuntimeDiagnostics(stall_threshold_s=0.1)
    runtime_diagnostics.observe_stage(
        "rgbRender",
        0.12,
        unix_ns=456,
    )
    store = PlantFrameStore(
        SimpleNamespace(
            robot_id="g1-sim",
            localization_epoch_id="test-epoch",
        ),
        model=None,
        data=None,
        support=support,
        runtime_diagnostics=runtime_diagnostics,
    )
    jpeg = b"\xff\xd8read-only-fixture\xff\xd9"
    with store._lock:
        store._sequence = 1
        store._captured_at_ns = 123
        store._rgb = jpeg
        store._depth = b"\x01\x00\x02\x00"
    store.serve(
        "127.0.0.1",
        port,
        docker_observation_host="127.0.0.2",
    )
    try:
        status, _headers, body = _request("127.0.0.2", port, "GET", "/healthz")
        assert status == 200
        state = json.loads(body)
        assert state["status"] == "running"
        assert state["runtimeTiming"]["lastSlowStage"] == "rgbRender"
        assert state["runtimeTiming"]["stages"]["rgbRender"][
            "overThresholdCount"
        ] == 1
        assert "oracle" not in json.dumps(state).lower()
        assert "segmentation" not in json.dumps(state).lower()
        status, _headers, body = _request("127.0.0.2", port, "GET", "/rgb.jpg")
        assert status == 200
        assert body == jpeg
        status, _headers, body = _request("127.0.0.2", port, "GET", "/depth.raw")
        assert status == 200
        assert body == b"\x01\x00\x02\x00"
        status, _headers, _body = _request(
            "127.0.0.2",
            port,
            "GET",
            "/oracle.json?rgbSha256=" + ("0" * 64),
        )
        assert status == 404

        payload = json.dumps({"mode": "locomotion"}).encode("utf-8")
        status, _headers, _body = _request("127.0.0.1", port, "POST", "/support", payload)
        assert status == 200
        assert support.state()["mode"] == "locomotion"

        safety = json.dumps({"mode": "safety_stop"}).encode("utf-8")
        status, _headers, _body = _request(
            "127.0.0.1", port, "POST", "/support", safety
        )
        assert status == 200
        ordinary_idle = json.dumps({"mode": "idle"}).encode("utf-8")
        status, _headers, _body = _request(
            "127.0.0.1", port, "POST", "/support", ordinary_idle
        )
        assert status == 422
        assert support.state()["mode"] == "safety_stop"
        explicit_rearm = json.dumps({"generation": 11}).encode("utf-8")
        status, _headers, body = _request(
            "127.0.0.1", port, "POST", "/support/rearm", explicit_rearm
        )
        assert status == 200
        assert json.loads(body)["mode"] == "idle"
        assert json.loads(body)["lastRearmGeneration"] == 11

        rejected = json.dumps({"mode": "safety_stop"}).encode("utf-8")
        status, headers, body = _request(
            "127.0.0.2", port, "POST", "/support", rejected
        )
        assert status == 405
        assert headers["Allow"] == "GET"
        assert body == b""
        assert support.state()["mode"] == "idle"
    finally:
        store.close()
