"""Small loopback-only HTTP server for read-only Robot Agent diagnostics."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, Protocol

from robot_agent.config import DiagnosticsConfig

LOGGER = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 16 * 1024
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_STATUS_REASONS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Content Too Large",
    503: "Service Unavailable",
}


class DiagnosticsProvider(Protocol):
    async def diagnostics_snapshot(self) -> dict[str, Any]: ...

    async def capabilities_snapshot(self) -> dict[str, Any]: ...


class DiagnosticsApiServer:
    """Serve four bounded GET endpoints without a mutable application surface."""

    def __init__(
        self,
        config: DiagnosticsConfig,
        provider: DiagnosticsProvider,
    ) -> None:
        self._config = config
        self._provider = provider
        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        if not self._config.enabled:
            return
        if self._server is not None:
            raise RuntimeError("diagnostics API is already started")
        address = ipaddress.ip_address(self._config.host)
        if not address.is_loopback:
            raise RuntimeError("diagnostics API may bind only to loopback")
        self._server = await asyncio.start_server(
            self._accept,
            host=self._config.host,
            port=self._config.port,
            limit=_MAX_REQUEST_BYTES,
            start_serving=True,
        )

    def _accept(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._clients.add(task)
        task.add_done_callback(self._client_finished)

    def _client_finished(self, task: asyncio.Task[None]) -> None:
        self._clients.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            LOGGER.warning(
                "Robot Agent diagnostics client failed: %s",
                type(exception).__name__,
                exc_info=exception,
            )

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=self._config.request_timeout_ms / 1_000.0,
            )
            if len(request) > _MAX_REQUEST_BYTES:
                await self._write_json(writer, 413, {"error": "request_too_large"})
                return
            try:
                method, target, headers = _parse_request(request)
            except ValueError:
                await self._write_json(writer, 400, {"error": "invalid_request"})
                return
            if headers.get("transfer-encoding") is not None:
                await self._write_json(writer, 400, {"error": "request_body_not_allowed"})
                return
            content_length = headers.get("content-length")
            if content_length is not None:
                try:
                    body_size = int(content_length)
                except ValueError:
                    await self._write_json(writer, 400, {"error": "invalid_request"})
                    return
                if body_size != 0:
                    await self._write_json(
                        writer,
                        400,
                        {"error": "request_body_not_allowed"},
                    )
                    return
            if method != "GET":
                await self._write_json(
                    writer,
                    405,
                    {"error": "method_not_allowed"},
                    extra_headers={"Allow": "GET"},
                )
                return
            await self._route(writer, target)
        except (
            asyncio.IncompleteReadError,
            asyncio.LimitOverrunError,
            asyncio.TimeoutError,
        ):
            with contextlib.suppress(Exception):
                await self._write_json(writer, 400, {"error": "invalid_request"})
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _route(self, writer: asyncio.StreamWriter, target: str) -> None:
        if target == "/healthz":
            await self._write_json(writer, 200, {"status": "alive"})
            return
        try:
            if target in {"/readyz", "/v1/diagnostics"}:
                payload = await asyncio.wait_for(
                    self._provider.diagnostics_snapshot(),
                    timeout=self._config.request_timeout_ms / 1_000.0,
                )
                status = 200 if target == "/v1/diagnostics" or payload.get("ready") else 503
            elif target == "/v1/capabilities":
                payload = await asyncio.wait_for(
                    self._provider.capabilities_snapshot(),
                    timeout=self._config.request_timeout_ms / 1_000.0,
                )
                status = 200
            else:
                await self._write_json(writer, 404, {"error": "not_found"})
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.warning(
                "Robot Agent diagnostics snapshot failed: %s",
                type(exc).__name__,
                exc_info=exc,
            )
            await self._write_json(writer, 503, {"error": "diagnostics_unavailable"})
            return
        await self._write_json(writer, status, payload)

    async def _write_json(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        payload: Mapping[str, Any],
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        reason = _STATUS_REASONS[status]
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'",
            "X-Content-Type-Options": "nosniff",
            "Connection": "close",
            **(dict(extra_headers) if extra_headers is not None else {}),
        }
        head = [f"HTTP/1.1 {status} {reason}"]
        head.extend(f"{name}: {value}" for name, value in headers.items())
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode("ascii") + body)
        await writer.drain()

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        clients = tuple(self._clients)
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)

    @property
    def bound_port(self) -> int | None:
        server = self._server
        if server is None or not server.sockets:
            return None
        return int(server.sockets[0].getsockname()[1])


def _parse_request(request: bytes) -> tuple[str, str, dict[str, str]]:
    try:
        text = request.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("HTTP request must be ASCII") from exc
    lines = text.removesuffix("\r\n\r\n").split("\r\n")
    request_parts = lines[0].split(" ")
    if len(request_parts) != 3:
        raise ValueError("invalid HTTP request line")
    method, target, version = request_parts
    if version not in {"HTTP/1.0", "HTTP/1.1"}:
        raise ValueError("unsupported HTTP version")
    if not target.startswith("/") or "?" in target or "#" in target:
        raise ValueError("invalid request target")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            raise ValueError("invalid HTTP header")
        name, value = line.split(":", 1)
        normalized = name.lower()
        if _HEADER_NAME.fullmatch(name) is None or normalized in headers:
            raise ValueError("invalid or duplicate HTTP header")
        headers[normalized] = value.strip()
    return method, target, headers
