"""DashScope Qwen-ASR-Realtime provider.

Protocol implementation follows Alibaba Cloud's Qwen-ASR-Realtime WebSocket
events: one upstream connection is held for the complete browser speech session,
audio is appended as base64 PCM frames, and utterance boundaries come from
server VAD.
"""

from __future__ import annotations

import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
import re
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from .contracts import AudioStreamConfig
from .errors import (
    ProviderDisconnected,
    ProviderRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from .providers import ASRProvider, ASRSession, ProviderEvent, UnavailableASRProvider
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


def _env_int(
    values: Mapping[str, str], key: str, default: int, *, minimum: int, maximum: int
) -> int:
    try:
        value = int(values.get(key, str(default)))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def _endpoint_with_model(endpoint: str, model: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme != "wss" or not parsed.hostname:
        raise ValueError("DashScope realtime endpoint must be a wss:// URL")
    if "{" in endpoint or "}" in endpoint:
        raise ValueError("DashScope realtime endpoint contains an unresolved placeholder")
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "model"]
    query.append(("model", model))
    return urlunparse(parsed._replace(query=urlencode(query)))


def _dashscope_endpoint(values: Mapping[str, str]) -> str | None:
    explicit = values.get("DASHSCOPE_ASR_WS_URL", "").strip()
    if explicit:
        return explicit
    workspace = values.get("DASHSCOPE_WORKSPACE_ID", "").strip()
    if not workspace or re.fullmatch(r"[A-Za-z0-9-]+", workspace) is None:
        return None
    region = values.get("DASHSCOPE_REGION", "cn-beijing").strip().lower()
    domains = {
        "cn-beijing": "cn-beijing.maas.aliyuncs.com",
        "ap-southeast-1": "ap-southeast-1.maas.aliyuncs.com",
    }
    suffix = domains.get(region)
    if suffix is None:
        return None
    return f"wss://{workspace}.{suffix}/api-ws/v1/realtime"


@dataclass(frozen=True, slots=True)
class DashScopeRealtimeConfig:
    endpoint_url: str
    model: str = "qwen3-asr-flash-realtime"
    api_key: str = field(default="", repr=False)
    open_timeout_seconds: float = 8.0
    configure_timeout_seconds: float = 8.0
    send_timeout_seconds: float = 2.0
    event_timeout_seconds: float = 120.0
    close_timeout_seconds: float = 3.0
    vad_threshold: float = 0.0
    vad_silence_ms: int = 400


class DashScopeRealtimeProvider:
    name = "dashscope-qwen-asr-realtime"

    def __init__(
        self,
        config: DashScopeRealtimeConfig,
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
        api_key = values.get("DASHSCOPE_API_KEY", "").strip()
        endpoint = _dashscope_endpoint(values)
        if not api_key:
            return UnavailableASRProvider("DASHSCOPE_API_KEY is not configured")
        if not endpoint:
            return UnavailableASRProvider(
                "DASHSCOPE_ASR_WS_URL or DASHSCOPE_WORKSPACE_ID is not configured"
            )
        model = values.get(
            "DASHSCOPE_ASR_MODEL", "qwen3-asr-flash-realtime"
        ).strip()
        if not model:
            return UnavailableASRProvider("DASHSCOPE_ASR_MODEL is empty")
        try:
            endpoint_url = _endpoint_with_model(endpoint, model)
        except ValueError:
            return UnavailableASRProvider("DashScope realtime endpoint is invalid")
        return cls(
            DashScopeRealtimeConfig(
                api_key=api_key,
                endpoint_url=endpoint_url,
                model=model,
                open_timeout_seconds=_env_float(
                    values, "SPEECH_OPEN_TIMEOUT_SECONDS", 8.0, minimum=1, maximum=30
                ),
                configure_timeout_seconds=_env_float(
                    values,
                    "SPEECH_CONFIGURE_TIMEOUT_SECONDS",
                    8.0,
                    minimum=1,
                    maximum=30,
                ),
                send_timeout_seconds=_env_float(
                    values, "SPEECH_SEND_TIMEOUT_SECONDS", 2.0, minimum=0.1, maximum=10
                ),
                event_timeout_seconds=_env_float(
                    values, "SPEECH_EVENT_TIMEOUT_SECONDS", 120.0, minimum=10, maximum=600
                ),
                close_timeout_seconds=_env_float(
                    values, "SPEECH_CLOSE_TIMEOUT_SECONDS", 3.0, minimum=0.1, maximum=10
                ),
                vad_threshold=_env_float(
                    values, "DASHSCOPE_ASR_VAD_THRESHOLD", 0.0, minimum=0, maximum=1
                ),
                vad_silence_ms=_env_int(
                    values,
                    "DASHSCOPE_ASR_VAD_SILENCE_MS",
                    400,
                    minimum=200,
                    maximum=2_000,
                ),
            ),
            transport_factory=transport_factory,
        )

    async def open(self, config: AudioStreamConfig) -> ASRSession:
        try:
            transport = await asyncio.wait_for(
                self._transport_factory.connect(
                    self._config.endpoint_url,
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "OpenAI-Beta": "realtime=v1",
                    },
                    open_timeout_seconds=self._config.open_timeout_seconds,
                ),
                timeout=self._config.open_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout("connection") from exc
        except Exception as exc:
            raise ProviderDisconnected() from exc
        session = DashScopeRealtimeSession(transport, self._config)
        try:
            await session.configure(config)
        except Exception:
            await session.close()
            raise
        return session


class DashScopeRealtimeSession:
    def __init__(
        self, transport: RealtimeTransport, config: DashScopeRealtimeConfig
    ) -> None:
        self._transport = transport
        self._config = config
        self._send_lock = asyncio.Lock()
        self._finished = False
        self._closed = False

    async def _receive_json(self, *, phase: str, timeout: float) -> Mapping[str, Any]:
        try:
            return await asyncio.wait_for(self._transport.receive_json(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ProviderTimeout(phase) from exc
        except Exception as exc:
            raise ProviderDisconnected() from exc

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
        created = await self._receive_json(
            phase="session creation", timeout=self._config.configure_timeout_seconds
        )
        if created.get("type") == "error":
            raise ProviderRejected()
        if created.get("type") != "session.created":
            raise ProviderRejected()
        language = stream.locale.split("-", 1)[0].lower()
        transcription: dict[str, Any] = {}
        if language != "auto":
            transcription["language"] = language
        await self._send_json(
            {
                "event_id": f"event_{uuid4().hex}",
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "input_audio_format": "pcm",
                    "sample_rate": stream.sample_rate_hz,
                    "input_audio_transcription": transcription,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self._config.vad_threshold,
                        "silence_duration_ms": self._config.vad_silence_ms,
                    },
                },
            }
        )
        updated = await self._receive_json(
            phase="session configuration", timeout=self._config.configure_timeout_seconds
        )
        if updated.get("type") == "error":
            raise ProviderRejected()
        if updated.get("type") != "session.updated":
            raise ProviderRejected()

    async def send_audio(self, frame: bytes) -> None:
        if self._finished or self._closed:
            raise ProviderDisconnected()
        await self._send_json(
            {
                "event_id": f"event_{uuid4().hex}",
                "type": "input_audio_buffer.append",
                "audio": b64encode(frame).decode("ascii"),
            }
        )

    async def finish(self) -> None:
        if self._finished or self._closed:
            return
        self._finished = True
        await self._send_json(
            {"event_id": f"event_{uuid4().hex}", "type": "session.finish"}
        )

    async def receive_event(self) -> ProviderEvent:
        while not self._closed:
            payload = await self._receive_json(
                phase="provider event", timeout=self._config.event_timeout_seconds
            )
            event_type = payload.get("type")
            if event_type in {"error", "conversation.item.input_audio_transcription.failed"}:
                raise ProviderRejected()
            if event_type == "input_audio_buffer.speech_started":
                return ProviderEvent(
                    "speech.started",
                    utterance_id=_string(payload.get("item_id")),
                    offset_ms=_integer(payload.get("audio_start_ms")),
                )
            if event_type == "input_audio_buffer.speech_stopped":
                return ProviderEvent(
                    "speech.ended",
                    utterance_id=_string(payload.get("item_id")),
                    offset_ms=_integer(payload.get("audio_end_ms")),
                )
            if event_type == "conversation.item.input_audio_transcription.text":
                confirmed = _string(payload.get("text")) or ""
                stash = _string(payload.get("stash")) or ""
                preview = f"{confirmed}{stash}"
                if not preview:
                    continue
                return ProviderEvent(
                    "transcript.partial",
                    utterance_id=_string(payload.get("item_id")),
                    text=preview,
                    confirmed_text=confirmed,
                    language=_string(payload.get("language")),
                    emotion=_string(payload.get("emotion")),
                )
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = (_string(payload.get("transcript")) or "").strip()
                if not transcript:
                    continue
                return ProviderEvent(
                    "transcript.final",
                    utterance_id=_string(payload.get("item_id")),
                    text=transcript,
                    confirmed_text=transcript,
                    language=_string(payload.get("language")),
                    emotion=_string(payload.get("emotion")),
                )
            if event_type == "session.finished":
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
            # Closing is best effort; no key, URL, or provider payload is exposed.
            return


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


__all__ = [
    "DashScopeRealtimeConfig",
    "DashScopeRealtimeProvider",
    "DashScopeRealtimeSession",
]
