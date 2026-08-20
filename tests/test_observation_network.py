from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import threading
import time
import uuid

import pytest

from hide_mujoco_sonic.network import (
    ObservationBindError,
    require_loopback_bind_host,
    resolve_docker_observation_bind_host,
)
from hide_mujoco_sonic.config import SimConfig
from hide_mujoco_sonic.frames import BodyTwist
from hide_mujoco_sonic.plant import PlantFrameStore, PlantSupportController
from hide_mujoco_sonic.render_workers import RenderedFrame
from hide_mujoco_sonic.sonic_driver import _bind_command_server
from hide_mujoco_sonic.telemetry import TelemetryState, TelemetryStore, serve_telemetry
from hide_mujoco_sonic.runtime_diagnostics import PlantRuntimeDiagnostics
from hide_mujoco_sonic.transport import latch_plant_physics_fault, request_json


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


def _request_with_loopback_response(
    response: dict[str, object] | bytes,
    *,
    require_ok: bool = True,
) -> dict[str, object]:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = int(server.getsockname()[1])

    def respond() -> None:
        connection, _address = server.accept()
        with connection:
            raw = b""
            while b"\n" not in raw:
                raw += connection.recv(4096)
            encoded = (
                response
                if isinstance(response, bytes)
                else json.dumps(response).encode("utf-8")
            )
            connection.sendall(encoded + b"\n")

    worker = threading.Thread(target=respond, daemon=True)
    worker.start()
    try:
        return request_json(
            "127.0.0.1",
            port,
            {"type": "test", "request_id": "request-8"},
            timeout=1.0,
            require_ok=require_ok,
        )
    finally:
        worker.join(timeout=2.0)
        server.close()


def test_loopback_transport_can_return_strict_negative_truth_to_caller() -> None:
    negative = {
        "request_id": "request-8",
        "ok": False,
        "error": "strict state machine rejected the phase",
    }

    with pytest.raises(RuntimeError, match="rejected the phase"):
        _request_with_loopback_response(negative)
    assert _request_with_loopback_response(
        negative,
        require_ok=False,
    ) == negative


@pytest.mark.parametrize("ok", [None, 0, 1, "false"])
def test_negative_truth_mode_still_requires_an_exact_boolean(ok: object) -> None:
    with pytest.raises(RuntimeError, match="truth flag"):
        _request_with_loopback_response(
            {"request_id": "request-8", "ok": ok},
            require_ok=False,
        )


def test_negative_truth_mode_keeps_identity_and_protocol_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _request_with_loopback_response(
            {"request_id": "another-request", "ok": False},
            require_ok=False,
        )
    with pytest.raises(RuntimeError, match="invalid response"):
        _request_with_loopback_response(b"{not-json", require_ok=False)
    with pytest.raises(ValueError, match="require_ok"):
        request_json(
            "127.0.0.1",
            1,
            {"type": "test"},
            require_ok=0,  # type: ignore[arg-type]
        )


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
        support=support,
        runtime_diagnostics=runtime_diagnostics,
        worker_status_provider=lambda: {
            "status": "running",
            "ready": True,
            "alive": True,
        },
        dynamics_evidence_provider=lambda: {
            "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
            "status": "running",
            "diagnosticOnly": True,
            "authorityFeedbackEnabled": False,
            "supportBandEnabled": True,
            "attachedBody": "torso_link",
            "lastAppliedWrench": {
                "frameId": "world",
                "order": [
                    "force_x", "force_y", "force_z",
                    "torque_x", "torque_y", "torque_z",
                ],
                "values": [0.0, 0.0, 42.0, 1.0, 2.0, 0.0],
            },
        },
    )
    jpeg = b"\xff\xd8read-only-fixture\xff\xd9"
    captured_at = time.time_ns()
    captured_at_monotonic = time.monotonic_ns()
    producer_boot_id = str(uuid.uuid4())
    store.commit(
        RenderedFrame(
            producer_boot_id=producer_boot_id,
            sequence=1,
            captured_at_unix_ns=captured_at,
            captured_at_monotonic_ns=captured_at_monotonic,
            render_started_at_unix_ns=captured_at + 1,
            render_started_at_monotonic_ns=captured_at_monotonic + 1,
            rendered_at_unix_ns=captured_at + 2,
            rendered_at_monotonic_ns=captured_at_monotonic + 2,
            simulation_time_s=1.0,
            pose=(0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
            root_velocity=BodyTwist.zero(),
            camera_pose=(0.2, 0.0, 1.15, 1.0, 0.0, 0.0, 0.0),
            intrinsics=(("fx", 1.0), ("fy", 1.0), ("cx", 1.0), ("cy", 0.5)),
            width=2,
            height=1,
            rgb_jpeg=jpeg,
            depth_z16_le=b"\x01\x00\x02\x00",
            render_timing=(("totalMs", 0.001),),
        )
    )
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
        evidence = state["dynamicsEvidence"]
        assert evidence["diagnosticOnly"] is True
        assert evidence["authorityFeedbackEnabled"] is False
        assert evidence["supportBandEnabled"] is True
        assert evidence["attachedBody"] == "torso_link"
        assert evidence["lastAppliedWrench"]["values"][-1] == 0.0
        assert "oracle" not in json.dumps(state).lower()
        assert "segmentation" not in json.dumps(state).lower()
        # Image routes must use their selected immutable frame directly.  A
        # full state/diagnostics projection can be arbitrarily slow and is not
        # part of the RGB/depth response path.
        state_method = store.state
        store.state = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("image GET constructed full Plant state")
        )
        try:
            status, _headers, body = _request(
                "127.0.0.2", port, "GET", "/rgb.jpg"
            )
            assert status == 200
            assert body == jpeg
            status, _headers, body = _request(
                "127.0.0.2", port, "GET", "/depth.raw"
            )
            assert status == 200
            assert body == b"\x01\x00\x02\x00"
            exact = f"boot={producer_boot_id}&sequence=1"
            status, headers, body = _request(
                "127.0.0.2", port, "GET", f"/rgb.jpg?{exact}"
            )
        finally:
            store.state = state_method
        assert status == 200
        assert body == jpeg
        assert headers["X-Hide-Capture-Boot-Id"] == producer_boot_id
        assert headers["X-Hide-Capture-Sequence"] == "1"
        assert headers["X-Hide-Captured-At-Unix-Ns"] == str(captured_at)
        status, _headers, _body = _request(
            "127.0.0.2",
            port,
            "GET",
            f"/depth.raw?boot={producer_boot_id}&sequence=2",
        )
        assert status == 409
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

        store.set_worker_status_provider(
            lambda: {
                "status": "failed",
                "ready": True,
                "alive": False,
                "lastError": "injected renderer crash",
            }
        )
        status, _headers, body = _request(
            "127.0.0.2", port, "GET", "/healthz"
        )
        assert status == 503
        health = json.loads(body)
        assert health["physicsStatus"] == "running"
        assert health["observationStatus"] == "failed"
        assert health["status"] == "degraded"
        status, _headers, _body = _request(
            "127.0.0.2", port, "GET", "/rgb.jpg"
        )
        assert status == 503

        store.set_worker_status_provider(
            lambda: {"status": "running", "ready": True, "alive": True}
        )
        stale_at = time.time_ns() - 500_000_001
        stale_at_monotonic = time.monotonic_ns() - 500_000_001
        store.commit(
            RenderedFrame(
                producer_boot_id=producer_boot_id,
                sequence=2,
                captured_at_unix_ns=stale_at,
                captured_at_monotonic_ns=stale_at_monotonic,
                render_started_at_unix_ns=stale_at + 1,
                render_started_at_monotonic_ns=stale_at_monotonic + 1,
                rendered_at_unix_ns=stale_at + 2,
                rendered_at_monotonic_ns=stale_at_monotonic + 2,
                simulation_time_s=2.0,
                pose=(0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
                root_velocity=BodyTwist.zero(),
                camera_pose=(0.2, 0.0, 1.15, 1.0, 0.0, 0.0, 0.0),
                intrinsics=(("fx", 1.0), ("fy", 1.0), ("cx", 1.0), ("cy", 0.5)),
                width=2,
                height=1,
                rgb_jpeg=jpeg,
                depth_z16_le=b"\x01\x00\x02\x00",
                render_timing=(("totalMs", 0.001),),
            )
        )
        status, _headers, body = _request(
            "127.0.0.2", port, "GET", "/healthz"
        )
        assert status == 503
        assert json.loads(body)["observationStatus"] == "stale"
    finally:
        store.close()


def test_terminal_motion_fault_latches_plant_without_a_reset_path() -> None:
    port = _dual_loopback_port()
    support = PlantSupportController()
    support.finish_startup()
    store = PlantFrameStore(
        SimpleNamespace(
            robot_id="g1-sim",
            localization_epoch_id="test-epoch",
        ),
        support=support,
        runtime_diagnostics=PlantRuntimeDiagnostics(stall_threshold_s=0.1),
    )
    store.serve(
        "127.0.0.1",
        port,
        docker_observation_host="127.0.0.2",
    )
    try:
        reason = "gait_adapter/GAIT_ADAPTER_FAULT: start acknowledgement failed"
        state = latch_plant_physics_fault(
            "127.0.0.1",
            port,
            reason,
            timeout=1.0,
        )
        assert state["mode"] == "safety_stop"
        assert state["safetyStopLatched"] is True
        assert state["fault"] == reason

        # The Docker bridge listener is GET-only and cannot inject or clear a
        # terminal physics fault.
        payload = json.dumps({"reason": "replacement"}).encode("utf-8")
        status, headers, body = _request(
            "127.0.0.2",
            port,
            "POST",
            "/support/fault",
            payload,
        )
        assert status == 405
        assert headers["Allow"] == "GET"
        assert body == b""

        # Neither ordinary support traffic nor explicit rearm can recover a
        # physics fault; only a new Plant process can create a new state.
        status, _headers, _body = _request(
            "127.0.0.1",
            port,
            "POST",
            "/support/rearm",
            json.dumps({"generation": 99}).encode("utf-8"),
        )
        assert status == 422
        assert support.state()["fault"] == reason
    finally:
        store.close()
