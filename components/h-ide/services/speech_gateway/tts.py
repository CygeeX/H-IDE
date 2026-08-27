"""DashScope Qwen-TTS-Realtime streaming provider.

The implementation follows the official Realtime WebSocket lifecycle:
``session.created`` -> ``session.update`` -> append/commit text -> streamed
``response.audio.delta`` events -> ``session.finish``.  It never buffers a
complete audio file and never substitutes silence when the provider is absent.
"""

from __future__ import annotations

import asyncio
from base64 import b64decode
from binascii import Error as Base64Error
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from .errors import TTSDisconnected, TTSRejected, TTSTimeout, TTSUnavailable
from .providers import DisabledTTSProvider, StreamingTTSProvider, TTSChunk, TTSStream
from .transport import (
    RealtimeTransport,
    RealtimeTransportFactory,
    WebsocketsTransportFactory,
)


def _env_float(
    values: Mapping[str, str], key: str, default: float, *, minimum: float, maximum: float
) -> float:
    try:
        value = float(values.get(key, str(default)))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def _endpoint_with_model(endpoint: str, model: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "wss" or not parsed.hostname:
        raise ValueError("DashScope TTS endpoint must be a wss:// URL")
    if "{" in endpoint or "}" in endpoint:
        raise ValueError("DashScope TTS endpoint contains an unresolved placeholder")
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "model"]
    query.append(("model", model))
    return urlunparse(parsed._replace(query=urlencode(query)))


def _dashscope_tts_endpoint(values: Mapping[str, str]) -> str | None:
    explicit = values.get("DASHSCOPE_TTS_WS_URL", "").strip()
    if explicit:
        return explicit
    region = values.get("DASHSCOPE_REGION", "cn-beijing").strip().lower()
    if region == "cn-beijing":
        return "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    if region == "ap-southeast-1":
        return "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"
    return None


def _language_type(locale: str) -> str:
    language = locale.split("-", 1)[0].casefold()
    return {
        "zh": "Chinese",
        "en": "English",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "es": "Spanish",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "ru": "Russian",
    }.get(language, "Auto")


def _status_code(error: BaseException) -> int | None:
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(error, "response", None)
    nested = getattr(response, "status_code", None)
    return nested if isinstance(nested, int) else None


@dataclass(frozen=True, slots=True)
class DashScopeRealtimeTTSConfig:
    endpoint_url: str
    model: str = "qwen3-tts-flash-realtime"
    voice: str = "Cherry"
    workspace_id: str | None = None
    api_key: str = field(default="", repr=False)
    sample_rate_hz: int = 24_000
    open_timeout_seconds: float = 8.0
    configure_timeout_seconds: float = 8.0
    send_timeout_seconds: float = 2.0
    event_timeout_seconds: float = 60.0
    close_timeout_seconds: float = 3.0


class DashScopeRealtimeTTSProvider:
    name = "dashscope-qwen-tts-realtime"

    def __init__(
        self,
        config: DashScopeRealtimeTTSConfig,
        *,
        transport_factory: RealtimeTransportFactory | None = None,
    ) -> None:
        self._config = config
        self._transport_factory = transport_factory or WebsocketsTransportFactory()
        self._permission_unavailable = False

    @classmethod
    def from_env(
        cls,
        values: Mapping[str, str],
        *,
        transport_factory: RealtimeTransportFactory | None = None,
    ) -> StreamingTTSProvider:
        api_key = values.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            return DisabledTTSProvider()
        model = values.get(
            "DASHSCOPE_TTS_MODEL", "qwen3-tts-flash-realtime"
        ).strip()
        voice = values.get("DASHSCOPE_TTS_VOICE", "Cherry").strip()
        endpoint = _dashscope_tts_endpoint(values)
        if not model or not voice or endpoint is None:
            return DisabledTTSProvider()
        try:
            endpoint_url = _endpoint_with_model(endpoint, model)
        except ValueError:
            return DisabledTTSProvider()
        return cls(
            DashScopeRealtimeTTSConfig(
                api_key=api_key,
                endpoint_url=endpoint_url,
                model=model,
                voice=voice,
                workspace_id=values.get("DASHSCOPE_WORKSPACE_ID", "").strip() or None,
                open_timeout_seconds=_env_float(
                    values, "SPEECH_TTS_OPEN_TIMEOUT_SECONDS", 8.0, minimum=1, maximum=30
                ),
                configure_timeout_seconds=_env_float(
                    values,
                    "SPEECH_TTS_CONFIGURE_TIMEOUT_SECONDS",
                    8.0,
                    minimum=1,
                    maximum=30,
                ),
                send_timeout_seconds=_env_float(
                    values,
                    "SPEECH_TTS_SEND_TIMEOUT_SECONDS",
                    2.0,
                    minimum=0.1,
                    maximum=10,
                ),
                event_timeout_seconds=_env_float(
                    values,
                    "SPEECH_TTS_EVENT_TIMEOUT_SECONDS",
                    60.0,
                    minimum=5,
                    maximum=300,
                ),
                close_timeout_seconds=_env_float(
                    values,
                    "SPEECH_TTS_CLOSE_TIMEOUT_SECONDS",
                    3.0,
                    minimum=0.1,
                    maximum=10,
                ),
            ),
            transport_factory=transport_factory,
        )

    def capability(self) -> Mapping[str, Any]:
        if self._permission_unavailable:
            return {
                "provider": self.name,
                "model": self._config.model,
                "available": False,
                "reason": "The configured DashScope account cannot use the TTS model",
            }
        return {
            "provider": self.name,
            "model": self._config.model,
            "available": True,
            "voice": self._config.voice,
            "encoding": "pcm_s16le",
            "sampleRateHz": self._config.sample_rate_hz,
        }

    async def synthesize(self, text: str, *, locale: str) -> TTSStream:
        if self._permission_unavailable:
            raise TTSUnavailable(
                "The configured DashScope account cannot use the TTS model"
            )
        headers = {"Authorization": f"Bearer {self._config.api_key}"}
        if self._config.workspace_id is not None:
            headers["X-DashScope-WorkSpace"] = self._config.workspace_id
        try:
            transport = await asyncio.wait_for(
                self._transport_factory.connect(
                    self._config.endpoint_url,
                    headers=headers,
                    open_timeout_seconds=self._config.open_timeout_seconds,
                ),
                timeout=self._config.open_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise TTSTimeout("connection") from exc
        except Exception as exc:
            if _status_code(exc) in {401, 403}:
                self._permission_unavailable = True
                raise TTSUnavailable(
                    "The configured DashScope account cannot use the TTS model"
                ) from exc
            raise TTSDisconnected() from exc

        stream = DashScopeRealtimeTTSStream(transport, self._config)
        try:
            await stream.start(text, locale=locale)
        except TTSRejected as exc:
            # With a fixed, locally validated protocol/configuration, a
            # rejection during session creation/update means the configured
            # account/model/voice capability is unavailable.  Latch that
            # state so health/readiness does not continue claiming support.
            self._permission_unavailable = True
            await stream.close()
            raise TTSUnavailable(
                "The configured DashScope account cannot use the TTS model"
            ) from exc
        except Exception:
            await stream.close()
            raise
        return stream


class DashScopeRealtimeTTSStream:
    def __init__(
        self,
        transport: RealtimeTransport,
        config: DashScopeRealtimeTTSConfig,
    ) -> None:
        self.stream_id = f"tts-{uuid4().hex}"
        self._transport = transport
        self._config = config
        self._send_lock = asyncio.Lock()
        self._finish_sent = False
        self._session_finished = False
        self._closed = False

    async def _receive_json(self, *, phase: str, timeout: float) -> Mapping[str, Any]:
        try:
            return await asyncio.wait_for(self._transport.receive_json(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TTSTimeout(phase) from exc
        except Exception as exc:
            raise TTSDisconnected() from exc

    async def _send_json(self, payload: Mapping[str, Any]) -> None:
        try:
            async with self._send_lock:
                await asyncio.wait_for(
                    self._transport.send_json(payload),
                    timeout=self._config.send_timeout_seconds,
                )
        except asyncio.TimeoutError as exc:
            raise TTSTimeout("request upload") from exc
        except Exception as exc:
            raise TTSDisconnected() from exc

    async def start(self, text: str, *, locale: str) -> None:
        created = await self._receive_json(
            phase="session creation", timeout=self._config.configure_timeout_seconds
        )
        if created.get("type") == "error" or created.get("type") != "session.created":
            raise TTSRejected()
        await self._send_json(
            {
                "event_id": f"event_{uuid4().hex}",
                "type": "session.update",
                "session": {
                    "voice": self._config.voice,
                    "mode": "commit",
                    "language_type": _language_type(locale),
                    "response_format": "pcm",
                    "sample_rate": self._config.sample_rate_hz,
                },
            }
        )
        updated = await self._receive_json(
            phase="session configuration", timeout=self._config.configure_timeout_seconds
        )
        if updated.get("type") == "error" or updated.get("type") != "session.updated":
            raise TTSRejected()
        await self._send_json(
            {
                "event_id": f"event_{uuid4().hex}",
                "type": "input_text_buffer.append",
                "text": text,
            }
        )
        await self._send_json(
            {
                "event_id": f"event_{uuid4().hex}",
                "type": "input_text_buffer.commit",
            }
        )

    async def _send_finish(self) -> None:
        if self._finish_sent or self._closed:
            return
        self._finish_sent = True
        await self._send_json(
            {"event_id": f"event_{uuid4().hex}", "type": "session.finish"}
        )

    async def chunks(self) -> AsyncIterator[TTSChunk]:
        sequence = 0
        response_done = False
        try:
            while not self._closed and not self._session_finished:
                payload = await self._receive_json(
                    phase="audio generation",
                    timeout=self._config.event_timeout_seconds,
                )
                event_type = payload.get("type")
                if event_type == "error":
                    raise TTSRejected()
                if event_type == "response.audio.delta":
                    encoded = payload.get("delta")
                    if not isinstance(encoded, str) or not encoded:
                        raise TTSRejected()
                    try:
                        audio = b64decode(encoded, validate=True)
                    except (Base64Error, ValueError) as exc:
                        raise TTSRejected() from exc
                    if not audio:
                        raise TTSRejected()
                    yield TTSChunk(
                        sequence=sequence,
                        audio=audio,
                        sample_rate_hz=self._config.sample_rate_hz,
                    )
                    sequence += 1
                    continue
                if event_type == "response.done":
                    response = payload.get("response")
                    if not isinstance(response, Mapping) or response.get("status") != "completed":
                        raise TTSRejected()
                    response_done = True
                    await self._send_finish()
                    continue
                if event_type == "session.finished":
                    if not response_done:
                        raise TTSRejected()
                    self._session_finished = True
                    return
        finally:
            if not self._session_finished:
                await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        if not self._finish_sent:
            try:
                await self._send_finish()
            except Exception:
                pass
        self._closed = True
        try:
            await asyncio.wait_for(
                self._transport.close(), timeout=self._config.close_timeout_seconds
            )
        except Exception:
            return


__all__ = [
    "DashScopeRealtimeTTSConfig",
    "DashScopeRealtimeTTSProvider",
    "DashScopeRealtimeTTSStream",
]
