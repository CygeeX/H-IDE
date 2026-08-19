"""Docker-bridge-only proxy for Robot Agent's loopback diagnostics.

Robot Agent deliberately binds its diagnostics API to ``127.0.0.1``.  H-IDE's
containerized control API needs the same already-redacted, read-only view, but
must never gain a route to an arbitrary host service.  This proxy therefore
has two fixed boundaries:

* the listener must be Docker's validated local default-bridge gateway; and
* the upstream is always Robot Agent at ``127.0.0.1:8796``.

Only four bodyless GET resources are forwarded.  Redirects, oversized or
non-JSON responses, request bodies, query strings, and every other method fail
closed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from .network import resolve_docker_observation_bind_host

LISTEN_PORT: Final = 8797
UPSTREAM_HOST: Final = "127.0.0.1"
UPSTREAM_PORT: Final = 8796
UPSTREAM_TIMEOUT_SECONDS: Final = 1.0
MAX_RESPONSE_BYTES: Final = 1024 * 1024
ALLOWED_PATHS: Final = frozenset(
    {"/healthz", "/readyz", "/v1/diagnostics", "/v1/capabilities"}
)


class DiagnosticsProxyError(RuntimeError):
    """The fixed read-only proxy boundary could not be satisfied."""


@dataclass(frozen=True, slots=True)
class UpstreamResponse:
    status: int
    body: bytes


class FixedLoopbackDiagnosticsUpstream:
    """Bounded HTTP client whose host is not runtime-configurable."""

    def __init__(
        self,
        *,
        timeout_seconds: float = UPSTREAM_TIMEOUT_SECONDS,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        _connection_factory: Callable[..., HTTPConnection] = HTTPConnection,
    ) -> None:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 5.0:
            raise ValueError("diagnostics upstream timeout must be in (0, 5] seconds")
        if not 1024 <= max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ValueError(
                f"diagnostics response limit must be in [1024, {MAX_RESPONSE_BYTES}]"
            )
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self._connection_factory = _connection_factory

    def fetch(self, path: str) -> UpstreamResponse:
        if path not in ALLOWED_PATHS:
            raise DiagnosticsProxyError("diagnostics upstream path is not allowed")
        connection = self._connection_factory(
            UPSTREAM_HOST,
            UPSTREAM_PORT,
            timeout=self.timeout_seconds,
        )
        try:
            connection.request(
                "GET",
                path,
                body=None,
                headers={
                    "Accept": "application/json",
                    "Cache-Control": "no-store",
                    "Connection": "close",
                },
            )
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise DiagnosticsProxyError("diagnostics upstream redirect is forbidden")
            if response.status not in {200, 503}:
                raise DiagnosticsProxyError(
                    "diagnostics upstream returned an unexpected status"
                )
            content_type = response.getheader("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != "application/json":
                raise DiagnosticsProxyError("diagnostics upstream response is not JSON")
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError as exc:
                    raise DiagnosticsProxyError(
                        "diagnostics upstream Content-Length is invalid"
                    ) from exc
                if declared_length < 0 or declared_length > self.max_response_bytes:
                    raise DiagnosticsProxyError(
                        "diagnostics upstream response exceeds the size limit"
                    )
            body = response.read(self.max_response_bytes + 1)
            if len(body) > self.max_response_bytes:
                raise DiagnosticsProxyError(
                    "diagnostics upstream response exceeds the size limit"
                )
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DiagnosticsProxyError(
                    "diagnostics upstream response is invalid JSON"
                ) from exc
            if not isinstance(payload, dict):
                raise DiagnosticsProxyError(
                    "diagnostics upstream response must be a JSON object"
                )
            return UpstreamResponse(status=int(response.status), body=body)
        except (OSError, HTTPException) as exc:
            raise DiagnosticsProxyError("diagnostics upstream is unavailable") from exc
        finally:
            connection.close()


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 16


def _json_bytes(payload: dict[str, str]) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _handler_type(
    fetch: Callable[[str], UpstreamResponse],
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _respond(
            self,
            status: int,
            body: bytes,
            *,
            allow: str | None = None,
        ) -> None:
            self.close_connection = True
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "private, no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Connection", "close")
            if allow is not None:
                self.send_header("Allow", allow)
            self.end_headers()
            if body:
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _reject_method(self) -> None:
            self._respond(
                405,
                _json_bytes({"error": "method_not_allowed"}),
                allow="GET",
            )

        def handle_expect_100(self) -> bool:
            self._respond(417, _json_bytes({"error": "request_body_forbidden"}))
            return False

        def do_GET(self) -> None:
            if self.path not in ALLOWED_PATHS:
                self._respond(404, _json_bytes({"error": "not_found"}))
                return
            transfer_encoding = self.headers.get("Transfer-Encoding")
            content_length = self.headers.get("Content-Length")
            if transfer_encoding is not None:
                self._respond(400, _json_bytes({"error": "request_body_forbidden"}))
                return
            if content_length is not None:
                try:
                    parsed_length = int(content_length)
                except ValueError:
                    parsed_length = -1
                if parsed_length != 0:
                    self._respond(
                        400,
                        _json_bytes({"error": "request_body_forbidden"}),
                    )
                    return
            try:
                response = fetch(self.path)
            except DiagnosticsProxyError:
                self._respond(502, _json_bytes({"error": "upstream_unavailable"}))
                return
            self._respond(response.status, response.body)

        do_POST = _reject_method
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
    fetch: Callable[[str], UpstreamResponse] | None = None,
) -> _ProxyServer:
    if not 0 <= port <= 65535:
        raise ValueError("diagnostics proxy listen port is invalid")
    upstream = FixedLoopbackDiagnosticsUpstream()
    return _ProxyServer((bind_host, port), _handler_type(fetch or upstream.fetch))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Expose fixed Robot Agent diagnostics on Docker's local bridge"
    )
    parser.add_argument(
        "--bind-host",
        default=os.getenv("HIDE_DOCKER_OBSERVATION_BIND_HOST", "auto"),
        help="'auto' or Docker's exact local default-bridge gateway",
    )
    args = parser.parse_args(argv)
    bind_host = resolve_docker_observation_bind_host(args.bind_host)
    if bind_host is None:
        raise DiagnosticsProxyError("Docker diagnostics proxy bind is disabled")
    server = _build_server(bind_host)
    print(
        f"Robot Agent diagnostics proxy listening on {bind_host}:{LISTEN_PORT}; "
        f"fixed upstream={UPSTREAM_HOST}:{UPSTREAM_PORT}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - launcher entry point
    raise SystemExit(main())


__all__ = [
    "ALLOWED_PATHS",
    "LISTEN_PORT",
    "DiagnosticsProxyError",
    "FixedLoopbackDiagnosticsUpstream",
    "UpstreamResponse",
]
