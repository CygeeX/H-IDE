"""Small WebSocket transport seam used by realtime speech providers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol


class RealtimeTransport(Protocol):
    async def send_json(self, payload: Mapping[str, Any]) -> None: ...

    async def send_bytes(self, payload: bytes) -> None: ...

    async def receive_json(self) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


class RealtimeTransportFactory(Protocol):
    async def connect(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        open_timeout_seconds: float,
    ) -> RealtimeTransport: ...


class WebsocketsTransport:
    """Adapter around websockets' asyncio connection.

    The adapter never logs its URL or headers.  In particular, the bearer key
    remains in memory only for the upstream handshake.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    async def send_json(self, payload: Mapping[str, Any]) -> None:
        await self._connection.send(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    async def send_bytes(self, payload: bytes) -> None:
        await self._connection.send(payload)

    async def receive_json(self) -> Mapping[str, Any]:
        raw = await self._connection.recv()
        if not isinstance(raw, str):
            raise ValueError("realtime provider returned a binary control message")
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("realtime provider returned a non-object event")
        return payload

    async def close(self) -> None:
        await self._connection.close()


class WebsocketsTransportFactory:
    def __init__(
        self,
        *,
        ping_interval_seconds: float = 20.0,
        ping_timeout_seconds: float = 10.0,
        max_incoming_bytes: int = 1_048_576,
    ) -> None:
        self._ping_interval_seconds = ping_interval_seconds
        self._ping_timeout_seconds = ping_timeout_seconds
        self._max_incoming_bytes = max_incoming_bytes

    async def connect(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        open_timeout_seconds: float,
    ) -> RealtimeTransport:
        from websockets.asyncio.client import connect

        connection = await connect(
            url,
            additional_headers=dict(headers),
            open_timeout=open_timeout_seconds,
            ping_interval=self._ping_interval_seconds,
            ping_timeout=self._ping_timeout_seconds,
            max_size=self._max_incoming_bytes,
            compression=None,
        )
        return WebsocketsTransport(connection)


__all__ = [
    "RealtimeTransport",
    "RealtimeTransportFactory",
    "WebsocketsTransport",
    "WebsocketsTransportFactory",
]
