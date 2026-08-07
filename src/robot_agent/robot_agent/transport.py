"""Hardened WebSocket client/server transports sharing one connection contract."""

from __future__ import annotations

import asyncio
import contextlib
import ssl
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.typing import Subprotocol

from robot_agent.config import ClientTransportConfig, ServerTransportConfig, TLSConfig
from robot_agent.protocol import MAX_TEXT_FRAME_BYTES

SUBPROTOCOL: Subprotocol = Subprotocol("robot-agent.v1")


class TransportError(RuntimeError):
    """Base error for WebSocket transport failures."""


class TransportProtocolError(TransportError):
    """Raised when a peer violates framing or subprotocol requirements."""


WebSocketConnection = ClientConnection | ServerConnection
ConnectionHandler = Callable[["AgentConnection"], Awaitable[None]]
ErrorHandler = Callable[[Exception], Awaitable[None]]


class AgentConnection:
    """Text-only, size-bounded view of a websockets connection."""

    def __init__(self, websocket: WebSocketConnection, *, mode: str) -> None:
        self._websocket = websocket
        self.mode = mode

    async def recv_text(self) -> str:
        try:
            raw = await self._websocket.recv()
        except ConnectionClosed as exc:
            received_code = None if exc.rcvd is None else exc.rcvd.code
            sent_code = None if exc.sent is None else exc.sent.code
            if 1009 in {received_code, sent_code, self._websocket.close_code}:
                raise TransportProtocolError(
                    f"WebSocket frame exceeds {MAX_TEXT_FRAME_BYTES} bytes"
                ) from exc
            raise
        if isinstance(raw, bytes):
            await self._websocket.close(code=1003, reason="binary frames are not supported")
            raise TransportProtocolError("binary WebSocket frames are not supported")
        size = len(raw.encode("utf-8"))
        if size > MAX_TEXT_FRAME_BYTES:
            await self._websocket.close(code=1009, reason="text frame exceeds 65536 bytes")
            raise TransportProtocolError(
                f"WebSocket frame exceeds {MAX_TEXT_FRAME_BYTES} bytes: received {size}"
            )
        return raw

    async def send_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TransportProtocolError("outbound WebSocket frames must be text")
        size = len(text.encode("utf-8"))
        if size > MAX_TEXT_FRAME_BYTES:
            raise TransportProtocolError(
                f"outbound frame exceeds {MAX_TEXT_FRAME_BYTES} bytes: {size}"
            )
        await self._websocket.send(text)

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        await self._websocket.close(code=code, reason=reason)

    @property
    def peer_certificate(self) -> dict[str, Any] | None:
        transport = self._websocket.transport
        ssl_object = transport.get_extra_info("ssl_object")
        if ssl_object is None:
            return None
        certificate = ssl_object.getpeercert()
        return certificate if isinstance(certificate, dict) else None

    @property
    def closed(self) -> bool:
        return self._websocket.state.name == "CLOSED"


def build_client_ssl_context(config: TLSConfig) -> ssl.SSLContext | None:
    """Create a server-auth context that also presents the Agent certificate."""
    if not config.enabled:
        return None
    assert config.ca_file is not None
    assert config.cert_file is not None
    assert config.key_file is not None
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=config.ca_file)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.options |= ssl.OP_NO_COMPRESSION
    context.load_cert_chain(certfile=config.cert_file, keyfile=config.key_file)
    return context


def build_server_ssl_context(config: TLSConfig) -> ssl.SSLContext | None:
    """Create a client-auth context that rejects peers without a trusted cert."""
    if not config.enabled:
        return None
    assert config.ca_file is not None
    assert config.cert_file is not None
    assert config.key_file is not None
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=config.ca_file)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = False
    context.verify_mode = ssl.CERT_REQUIRED
    context.options |= ssl.OP_NO_COMPRESSION
    context.load_cert_chain(certfile=config.cert_file, keyfile=config.key_file)
    return context


class ServerTransport:
    """Inbound transport that permits exactly one controlling peer."""

    def __init__(self, config: ServerTransportConfig, tls: TLSConfig) -> None:
        self._config = config
        self._tls = tls
        self._server: Server | None = None
        self._active_lock = asyncio.Lock()
        self._active = False

    async def start(self, handler: ConnectionHandler) -> None:
        if self._server is not None:
            raise TransportError("server transport is already started")
        self._server = await serve(
            lambda websocket: self._accept(websocket, handler),
            self._config.host,
            self._config.port,
            ssl=build_server_ssl_context(self._tls),
            subprotocols=[SUBPROTOCOL],
            compression=None,
            open_timeout=self._config.handshake_timeout_ms / 1000.0,
            close_timeout=2.0,
            ping_interval=10.0,
            ping_timeout=5.0,
            max_size=MAX_TEXT_FRAME_BYTES,
            max_queue=16,
            server_header=None,
        )

    async def _accept(
        self,
        websocket: ServerConnection,
        handler: ConnectionHandler,
    ) -> None:
        if websocket.subprotocol != SUBPROTOCOL:
            await websocket.close(code=1002, reason=f"required subprotocol {SUBPROTOCOL}")
            return
        async with self._active_lock:
            if self._active:
                await websocket.close(code=1013, reason="controlling session already connected")
                return
            self._active = True
        connection = AgentConnection(websocket, mode="server")
        try:
            await handler(connection)
        finally:
            if not connection.closed:
                await connection.close()
            async with self._active_lock:
                self._active = False

    async def run(self, handler: ConnectionHandler, stop: asyncio.Event) -> None:
        await self.start(handler)
        try:
            await stop.wait()
        finally:
            await self.close()

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

    @property
    def bound_port(self) -> int | None:
        if self._server is None or not self._server.sockets:
            return None
        return int(self._server.sockets[0].getsockname()[1])


class ClientTransport:
    """Outbound transport with bounded exponential reconnect backoff."""

    def __init__(self, config: ClientTransportConfig, tls: TLSConfig) -> None:
        self._config = config
        self._tls = tls

    async def connect_once(self) -> AgentConnection:
        ssl_context = build_client_ssl_context(self._tls)
        kwargs: dict[str, Any] = {}
        if ssl_context is not None and self._tls.server_name is not None:
            kwargs["server_hostname"] = self._tls.server_name
        websocket = await connect(
            self._config.uri,
            ssl=ssl_context,
            subprotocols=[SUBPROTOCOL],
            compression=None,
            open_timeout=self._config.connect_timeout_ms / 1000.0,
            close_timeout=2.0,
            ping_interval=10.0,
            ping_timeout=5.0,
            max_size=MAX_TEXT_FRAME_BYTES,
            max_queue=16,
            user_agent_header=None,
            proxy=None,
            **kwargs,
        )
        if websocket.subprotocol != SUBPROTOCOL:
            await websocket.close(code=1002, reason=f"required subprotocol {SUBPROTOCOL}")
            raise TransportProtocolError(f"peer did not negotiate {SUBPROTOCOL}")
        return AgentConnection(websocket, mode="client")

    async def run(
        self,
        handler: ConnectionHandler,
        stop: asyncio.Event,
        *,
        on_error: ErrorHandler | None = None,
    ) -> None:
        backoff = self._config.reconnect_initial_ms / 1000.0
        maximum = self._config.reconnect_max_ms / 1000.0
        while not stop.is_set():
            connection: AgentConnection | None = None
            try:
                connection = await self.connect_once()
                backoff = self._config.reconnect_initial_ms / 1000.0
                handler_task: asyncio.Future[Any] = asyncio.ensure_future(handler(connection))
                stop_task: asyncio.Future[Any] = asyncio.ensure_future(stop.wait())
                done, _ = await asyncio.wait(
                    {handler_task, stop_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if stop_task in done and stop.is_set():
                    handler_task.cancel()
                    await asyncio.gather(handler_task, return_exceptions=True)
                else:
                    stop_task.cancel()
                    await asyncio.gather(stop_task, return_exceptions=True)
                    await handler_task
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if on_error is not None:
                    await on_error(exc)
            finally:
                if connection is not None and not connection.closed:
                    await connection.close()
            if stop.is_set():
                break
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=backoff)
            backoff = min(maximum, backoff * 2.0)
