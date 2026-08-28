from __future__ import annotations

import json
from http.client import HTTPConnection
import socket
import threading

from hide_mujoco_sonic.leader_demo_gesture_proxy import _build_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _request(port: int, method: str, path: str, *, body: object = None, token: str = "secret"):
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    payload = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(payload))})
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    result = (response.status, json.loads(response.read().decode("utf-8")))
    connection.close()
    return result


def test_proxy_allows_only_authenticated_g1_wave() -> None:
    port = _free_port()
    forwarded: list[dict] = []

    def forward(payload: bytes) -> dict:
        value = json.loads(payload.decode("utf-8"))
        forwarded.append(value)
        return {"ok": True, "status": "accepted", "request_id": value["request_id"]}

    server = _build_server(
        "127.0.0.1",
        port=port,
        token="secret",
        forward=forward,
        ready=lambda: True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result = _request(
            port,
            "POST",
            "/v1/gesture/wave",
            body={"robotId": "g1-sim"},
        )
        assert status == 200
        assert result["status"] == "accepted"
        assert forwarded and forwarded[0]["type"] == "gesture_wave"
        assert forwarded[0]["gesture"] == "wave"
        assert forwarded[0]["simulation_only"] is True

        unauthorized, _ = _request(
            port,
            "POST",
            "/v1/gesture/wave",
            body={"robotId": "g1-sim"},
            token="wrong",
        )
        assert unauthorized == 401

        wrong_robot, _ = _request(
            port,
            "POST",
            "/v1/gesture/wave",
            body={"robotId": "other"},
        )
        assert wrong_robot == 422
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_proxy_health_is_read_only_and_reports_upstream_state() -> None:
    port = _free_port()
    server = _build_server(
        "127.0.0.1",
        port=port,
        token="secret",
        forward=lambda payload: {
            "ok": True,
            "request_id": json.loads(payload.decode("utf-8"))["request_id"],
        },
        ready=lambda: True,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, result = _request(port, "GET", "/healthz")
        assert status == 200
        assert result["service"] == "leader-demo-gesture-proxy"
        assert _request(port, "GET", "/v1/gesture/wave")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
