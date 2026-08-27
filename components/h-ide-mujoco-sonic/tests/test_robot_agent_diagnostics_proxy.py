from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from hide_mujoco_sonic import robot_agent_diagnostics_proxy as proxy

ROOT = Path(__file__).resolve().parents[1]


def _start(server: ThreadingHTTPServer) -> threading.Thread:
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return worker


def _request(
    port: int,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_proxy_forwards_only_bodyless_gets_on_the_four_locked_paths() -> None:
    calls: list[str] = []

    def fetch(path: str) -> proxy.UpstreamResponse:
        calls.append(path)
        status = 503 if path == "/readyz" else 200
        return proxy.UpstreamResponse(
            status=status,
            body=json.dumps({"path": path}, separators=(",", ":")).encode(),
        )

    server = proxy._build_server("127.0.0.1", port=0, fetch=fetch)
    worker = _start(server)
    port = int(server.server_address[1])
    try:
        for path in sorted(proxy.ALLOWED_PATHS):
            status, headers, body = _request(port, "GET", path)
            assert status == (503 if path == "/readyz" else 200)
            assert json.loads(body) == {"path": path}
            assert headers["Cache-Control"] == "private, no-store, max-age=0"
            assert headers["X-Content-Type-Options"] == "nosniff"

        before_rejections = list(calls)
        status, _headers, body = _request(port, "GET", "/healthz?target=other")
        assert status == 404
        assert json.loads(body) == {"error": "not_found"}
        status, headers, body = _request(port, "POST", "/healthz", body=b"{}")
        assert status == 405
        assert headers["Allow"] == "GET"
        assert json.loads(body) == {"error": "method_not_allowed"}
        status, _headers, body = _request(port, "GET", "/healthz", body=b"{}")
        assert status == 400
        assert json.loads(body) == {"error": "request_body_forbidden"}
        assert calls == before_rejections
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class _UpstreamHandler(BaseHTTPRequestHandler):
    mode: ClassVar[str] = "ok"
    requests: ClassVar[list[tuple[str, str, int]]] = []

    def do_GET(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        self.__class__.requests.append((self.command, self.path, content_length))
        if self.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/diagnostics")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.mode == "unexpected_status":
            body = json.dumps({"error": "unexpected"}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.mode == "oversized":
            body = json.dumps({"padding": "x" * 2048}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.mode == "wrong_content_type":
            body = b"not-json"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = json.dumps({"status": "alive"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        return


def test_upstream_is_loopback_get_only_bounded_and_does_not_follow_redirects() -> None:
    upstream_server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    worker = _start(upstream_server)
    port = int(upstream_server.server_address[1])
    fixed_targets: list[tuple[str, int]] = []

    def connect(host: str, requested_port: int, *, timeout: float) -> HTTPConnection:
        fixed_targets.append((host, requested_port))
        return HTTPConnection(host, port, timeout=timeout)

    client = proxy.FixedLoopbackDiagnosticsUpstream(
        timeout_seconds=1.0,
        max_response_bytes=1024,
        _connection_factory=connect,
    )
    _UpstreamHandler.requests = []
    try:
        _UpstreamHandler.mode = "ok"
        response = client.fetch("/healthz")
        assert response.status == 200
        assert json.loads(response.body) == {"status": "alive"}
        assert fixed_targets == [("127.0.0.1", 8796)]
        assert _UpstreamHandler.requests == [("GET", "/healthz", 0)]

        with pytest.raises(proxy.DiagnosticsProxyError, match="path is not allowed"):
            client.fetch("/support")

        _UpstreamHandler.mode = "redirect"
        with pytest.raises(proxy.DiagnosticsProxyError, match="redirect is forbidden"):
            client.fetch("/v1/diagnostics")

        _UpstreamHandler.mode = "unexpected_status"
        with pytest.raises(proxy.DiagnosticsProxyError, match="unexpected status"):
            client.fetch("/healthz")

        _UpstreamHandler.mode = "oversized"
        with pytest.raises(proxy.DiagnosticsProxyError, match="size limit"):
            client.fetch("/v1/capabilities")

        _UpstreamHandler.mode = "wrong_content_type"
        with pytest.raises(proxy.DiagnosticsProxyError, match="not JSON"):
            client.fetch("/readyz")
    finally:
        upstream_server.shutdown()
        upstream_server.server_close()
        worker.join(timeout=2)


def test_main_revalidates_docker_bridge_and_exposes_no_upstream_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Server:
        def serve_forever(self, *, poll_interval: float) -> None:
            events.append(("serve", poll_interval))

        def server_close(self) -> None:
            events.append("closed")

    def resolve(requested: str) -> str:
        events.append(("resolve", requested))
        return "172.31.0.1"

    def build(bind_host: str) -> Server:
        events.append(("build", bind_host))
        return Server()

    monkeypatch.setattr(proxy, "resolve_docker_observation_bind_host", resolve)
    monkeypatch.setattr(proxy, "_build_server", build)

    assert proxy.main(["--bind-host", "172.31.0.1"]) == 0
    assert events == [
        ("resolve", "172.31.0.1"),
        ("build", "172.31.0.1"),
        ("serve", 0.25),
        "closed",
    ]
    with pytest.raises(SystemExit):
        proxy.main(["--upstream", "http://example.invalid"])


def test_launcher_routes_compose_to_proxy_and_starts_it_after_robot_agent() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )

    assert (
        'export SIMULATION_ROBOT_AGENT_DIAGNOSTICS_URL="http://host.docker.internal:8797"'
        in launcher
    )
    robot_agent = launcher.index("  robot-agent-g1-sim \\")
    loopback_wait = launcher.index("http://127.0.0.1:8796/healthz", robot_agent)
    proxy_start = launcher.index("  robot-agent-diagnostics-proxy \\", loopback_wait)
    proxy_ready = launcher.index(":8797/healthz", proxy_start)
    assert robot_agent < loopback_wait < proxy_start < proxy_ready
    assert "--upstream" not in launcher

    status_script = (ROOT / "scripts" / "workstation-demo-status.sh").read_text(
        encoding="utf-8"
    )
    assert "http://127.0.0.1:8796/healthz" in status_script
    assert "-m hide_mujoco_sonic.network --requested auto" in status_script
    assert '"http://${docker_diagnostics_host}:8797/healthz"' in status_script
