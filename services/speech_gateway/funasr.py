"""Explicit local FunASR 2-pass WebSocket provider.

This adapter speaks the protocol shipped in FunASR's
``runtime/python/websocket`` examples.  It is selected only with
``SPEECH_ASR_PROVIDER=funasr``; it is never an automatic fallback for a failed
cloud provider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse
from uuid import uuid4

from .contracts import AudioStreamConfig
from .errors import ProviderDisconnected, ProviderRejected, ProviderTimeout
from .providers import ASRProvider, ASRSession, ProviderEvent, UnavailableASRProvider
from .transport import (
    RealtimeTransport,
    RealtimeTransportFactory,
    WebsocketsTransportFactory,
)


def _valid_local_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme == "wss" and parsed.hostname:
        return True
    return parsed.scheme == "ws" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True, slots=True)
class FunASRRealtimeConfig:
    endpoint_url: str
    open_timeout_seconds: float = 5.0
    send_timeout_seconds: float = 2.0
    event_timeout_seconds: float = 120.0
    close_timeout_seconds: float = 2.0


class FunASRRealtimeProvider:
    name = "local-funasr-2pass"

    def __init__(
        self,
        config: FunASRRealtimeConfig,
        *,
        transport_factory: RealtimeTransportFactory | None = None,
    ) -> None:
        self._config = config
        self._transport_factory = transport_factory or WebsocketsTransportFactory()

    @classmethod
    def from_env(
        cls,
        values: Mapping[str, str],
        *,
        transport_factory: RealtimeTransportFactory | None = None,
    ) -> ASRProvider:
        endpoint = values.get("FUNASR_WS_URL", "").strip()
        if not endpoint:
            return UnavailableASRProvider("FUNASR_WS_URL is not configured")
        if not _valid_local_endpoint(endpoint):
            return UnavailableASRProvider(
                "FUNASR_WS_URL must use wss or plaintext loopback"
            )
        return cls(
            FunASRRealtimeConfig(endpoint_url=endpoint),
            transport_factory=transport_factory,
        )

    async def open(self, config: AudioStreamConfig) -> ASRSession:
        try:
            transport = await asyncio.wait_for(
                self._transport_factory.connect(
                    self._config.endpoint_url,
                    headers={},
                    open_timeout_seconds=self._config.open_timeout_seconds,
                ),
                timeout=self._config.open_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("connection") from exc
        except Exception as exc:
            raise ProviderDisconnected() from exc
        session = FunASRRealtimeSession(transport, self._config)
        try:
            await session.configure(config)
        except Exception:
            await session.close()
            raise
        return session


class FunASRRealtimeSession:
    def __init__(self, transport: RealtimeTransport, config: FunASRRealtimeConfig) -> None:
        self._transport = transport
        self._config = config
        self._send_lock = asyncio.Lock()
        self._closed = False
        self._finishing = False
        self._online_text = ""
        self._pending: list[ProviderEvent] = []

    async def _send_json(self, payload: Mapping[str, Any]) -> None:
        try:
            async with self._send_lock:
                await asyncio.wait_for(
                    self._transport.send_json(payload),
                    timeout=self._config.send_timeout_seconds,
                )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("audio upload") from exc
        except Exception as exc:
            raise ProviderDisconnected() from exc

    async def configure(self, stream: AudioStreamConfig) -> None:
        await self._send_json(
            {
                "mode": "2pass",
                "chunk_size": [5, 10, 5],
                "encoder_chunk_look_back": 4,
                "decoder_chunk_look_back": 1,
                "chunk_interval": 10,
                "wav_name": f"speech-{uuid4().hex}",
                "wav_format": "pcm",
                "audio_fs": stream.sample_rate_hz,
                "is_speaking": True,
                "itn": True,
            }
        )

    async def send_audio(self, frame: bytes) -> None:
        if self._finishing or self._closed:
            raise ProviderDisconnected()
        try:
            async with self._send_lock:
                await asyncio.wait_for(
                    self._transport.send_bytes(frame),
                    timeout=self._config.send_timeout_seconds,
                )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("audio upload") from exc
        except Exception as exc:
            raise ProviderDisconnected() from exc

    async def finish(self) -> None:
        if self._finishing or self._closed:
            return
        self._finishing = True
        await self._send_json({"is_speaking": False, "is_end": True})

    async def receive_event(self) -> ProviderEvent:
        if self._pending:
            return self._pending.pop(0)
        while not self._closed:
            try:
                payload = await asyncio.wait_for(
                    self._transport.receive_json(),
                    timeout=self._config.event_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise ProviderTimeout("provider event") from exc
            except Exception as exc:
                raise ProviderDisconnected() from exc
            if payload.get("error"):
                raise ProviderRejected()
            mode = payload.get("mode")
            text = payload.get("text")
            text = text if isinstance(text, str) else ""
            if text and mode in {"online", "2pass-online"}:
                self._online_text += text
                result = ProviderEvent(
                    "transcript.partial",
                    text=self._online_text,
                    confirmed_text="",
                )
                if payload.get("is_final") is True:
                    self._pending.append(
                        ProviderEvent(
                            "transcript.final",
                            text=self._online_text,
                            confirmed_text=self._online_text,
                        )
                    )
                    self._online_text = ""
                return result
            if text and mode in {"offline", "2pass-offline"}:
                self._online_text = ""
                result = ProviderEvent(
                    "transcript.final", text=text, confirmed_text=text
                )
                if payload.get("is_end") is True:
                    self._pending.append(ProviderEvent("session.finished"))
                return result
            if payload.get("is_end") is True:
                if payload.get("is_final") is not True:
                    raise ProviderRejected()
                return ProviderEvent("session.finished")
        raise ProviderDisconnected()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await asyncio.wait_for(
                self._transport.close(), timeout=self._config.close_timeout_seconds
            )
        except Exception:
            return


__all__ = ["FunASRRealtimeConfig", "FunASRRealtimeProvider"]
