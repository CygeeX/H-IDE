"""Docker-only, allow-listed bridge for the leader-demo wave gesture.

The Control API runs in a container and must not receive a route to SONIC's
raw command socket.  This tiny host-side proxy is the only Docker-visible
write surface: it accepts exactly ``{"robotId": "g1-sim"}``, authenticates an
internal bearer token, and forwards a fixed ``gesture_wave`` frame to SONIC's
loopback socket.  It has no generic command forwarding capability.
"""

from __future__ import annotations

import argparse
from hmac import compare_digest
import json
import os
import socket
from collections.abc import Callable, Iterable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from uuid import uuid4

from .network import resolve_docker_observation_bind_host


LISTEN_PORT: Final = 8798
SONIC_COMMAND_HOST: Final = "127.0.0.1"
SONIC_COMMAND_PORT: Final = 8793
SONIC_TELEMETRY_HOST: Final = "127.0.0.1"
SONIC_TELEMETRY_PORT: Final = 8792
EXPECTED_ROBOT_ID: Final = "g1-sim"
MAX_REQUEST_BYTES: Final = 4096
MAX_RESPONSE_BYTES: Final = 64 * 1024
UPSTREAM_TIMEOUT_SECONDS: Final = 0.8
GESTURE_DURATION_SECONDS: Final = 2.4


class GestureProxyError(RuntimeError):
    """The constrained leader-demo proxy could not complete a request."""


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _read_sonic_ack(
    payload: bytes, *, timeout: float = UPSTREAM_TIMEOUT_SECONDS
) -> dict[str, Any]:
    try:
        with socket.create_connection(
            (SONIC_COMMAND_HOST, SONIC_COMMAND_PORT), timeout=timeout
        ) as connection:
            connection.settimeout(timeout)
            connection.sendall(payload + b"\n")
            connection.shutdown(socket.SHUT_WR)
            response = b""
            while b"\n" not in response:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                response += chunk
                if len(response) > MAX_RESPONSE_BYTES:
                    raise GestureProxyError("SONIC gesture response exceeded its bound")
    except (OSError, TimeoutError) as exc:
        raise GestureProxyError("SONIC gesture endpoint is unavailable") from exc
    if b"\n" not in response:
        raise GestureProxyError("SONIC gesture endpoint returned no complete ACK")
    try:
        value = json.loads(response.split(b"\n", 1)[0])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GestureProxyError("SONIC gesture endpoint returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise GestureProxyError("SONIC gesture endpoint returned a non-object ACK")
    if value.get("ok") is not True:
        raise GestureProxyError(str(value.get("error") or "SONIC rejected the gesture"))
    return value


def _sonic_ready(*, timeout: float = 0.5) -> bool:
    """Use SONIC's read-only health port for proxy readiness."""

    request = (
        b"GET /healthz HTTP/1.1\r\n"
        b"Host: 127.0.0.1\r\n"
        b"Connection: close\r\n\r\n"
    )
    try:
        with socket.create_connection(
            (SONIC_TELEMETRY_HOST, SONIC_TELEMETRY_PORT), timeout=timeout
        ) as connection:
            connection.settimeout(timeout)
            connection.sendall(request)
            raw = b""
            while len(raw) <= MAX_RESPONSE_BYTES:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                raw += chunk
                if len(raw) > MAX_RESPONSE_BYTES:
                    return False
                header_end = raw.find(b"\r\n\r\n")
                if header_end >= 0:
                    header = raw[:header_end]
                    declared = None
                    for line in header.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            try:
                                declared = int(line.split(b":", 1)[1].strip())
                            except ValueError:
                                return False
                            break
                    if declared is not None and len(raw) - header_end - 4 >= declared:
                        break
        header, _, body = raw.partition(b"\r\n\r\n")
        if b" 200 " not in header:
            return False
        value = json.loads(body.decode("utf-8"))
        return isinstance(value, Mapping) and value.get("status") == "running"
    except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError):
        return False


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16


def _handler_type(
    *,
    token: str,
    forward: Callable[[bytes], dict[str, Any]] = _read_sonic_ack,
    ready: Callable[[], bool] = _sonic_ready,
) -> type[BaseHTTPRequestHandler]:
    if not token:
        raise GestureProxyError("leader-demo gesture proxy token is required")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _respond(self, status: int, payload: Mapping[str, Any]) -> None:
            body = _json_bytes(payload)
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            if not supplied.startswith("Bearer "):
                return False
            return compare_digest(supplied[7:], token)

        def _reject_method(self) -> None:
            self._respond(405, {"error": "method_not_allowed"})

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._respond(404, {"error": "not_found"})
                return
            is_ready = ready()
            self._respond(
                200 if is_ready else 503,
                {
                    "status": "ok" if is_ready else "unavailable",
                    "service": "leader-demo-gesture-proxy",
                    "robotId": EXPECTED_ROBOT_ID,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/v1/gesture/wave":
                self._respond(404, {"error": "not_found"})
                return
            if not self._authorized():
                self._respond(401, {"error": "internal_authentication_required"})
                return
            if self.headers.get("Transfer-Encoding") is not None:
                self._respond(400, {"error": "request_body_forbidden"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_REQUEST_BYTES:
                self._respond(413, {"error": "request_body_exceeds_bound"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                self._respond(415, {"error": "application_json_required"})
                return
            try:
                value = json.loads(self.rfile.read(length))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                self._respond(400, {"error": "invalid_json"})
                return
            if not isinstance(value, Mapping) or set(value) != {"robotId"} or value.get("robotId") != EXPECTED_ROBOT_ID:
                self._respond(422, {"error": "only_g1_sim_wave_is_allowed"})
                return
            request_id = str(uuid4())
            payload = _json_bytes(
                {
                    "type": "gesture_wave",
                    "request_id": request_id,
                    "gesture": "wave",
                    "duration_s": GESTURE_DURATION_SECONDS,
                    "simulation_only": True,
                }
            )
            try:
                acknowledgement = forward(payload)
            except GestureProxyError:
                self._respond(503, {"error": "sonic_gesture_unavailable"})
                return
            if acknowledgement.get("request_id") != request_id:
                self._respond(503, {"error": "sonic_gesture_ack_identity_mismatch"})
                return
            self._respond(
                200,
                {
                    "status": "accepted",
                    "gesture": "wave",
                    "robotId": EXPECTED_ROBOT_ID,
                    "requestId": request_id,
                    "request_id": request_id,
                    "sonicAck": acknowledgement.get("status", "accepted"),
                },
            )

        do_PUT = _reject_method
        do_PATCH = _reject_method
        do_DELETE = _reject_method
        do_HEAD = _reject_method
        do_OPTIONS = _reject_method
        do_CONNECT = _reject_method
        do_TRACE = _reject_method

        def log_message(self, *_args: object) -> None:
            return

    return Handler


def _build_server(
    bind_host: str,
    *,
    port: int = LISTEN_PORT,
    token: str | None = None,
    forward: Callable[[bytes], dict[str, Any]] = _read_sonic_ack,
    ready: Callable[[], bool] = _sonic_ready,
) -> _ProxyServer:
    if not 1 <= int(port) <= 65535:
        raise ValueError("leader-demo gesture proxy listen port is invalid")
    resolved_token = token if token is not None else os.getenv("HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN", "").strip()
    return _ProxyServer(
        (bind_host, int(port)),
        _handler_type(token=resolved_token, forward=forward, ready=ready),
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind-host",
        default=os.getenv("HIDE_DOCKER_OBSERVATION_BIND_HOST", "auto"),
        help="'auto' or Docker's exact local default-bridge gateway",
    )
    parser.add_argument("--port", type=int, default=LISTEN_PORT)
    args = parser.parse_args(argv)
    bind_host = resolve_docker_observation_bind_host(args.bind_host)
    if bind_host is None:
        raise GestureProxyError("Docker gesture proxy bind is disabled")
    token = os.getenv("HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN", "").strip()
    server = _build_server(bind_host, port=args.port, token=token)
    print(
        f"Leader-demo gesture proxy listening on {bind_host}:{args.port}; "
        "only g1-sim wave is allowed",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = [
    "EXPECTED_ROBOT_ID",
    "GestureProxyError",
    "LISTEN_PORT",
    "_build_server",
    "_handler_type",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by launcher
    raise SystemExit(main())
