"""Provider contracts and fail-closed environment selection."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, AsyncIterator, Literal, Mapping, Protocol

from .contracts import AudioStreamConfig
from .errors import ProviderUnavailable, TTSUnavailable
from .transport import RealtimeTransportFactory


ProviderEventKind = Literal[
    "transcript.partial",
    "transcript.final",
    "speech.started",
    "speech.ended",
    "session.finished",
]


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    kind: ProviderEventKind
    utterance_id: str | None = None
    text: str | None = None
    confirmed_text: str | None = None
    language: str | None = None
    emotion: str | None = None
    offset_ms: int | None = None


class ASRSession(Protocol):
    async def send_audio(self, frame: bytes) -> None: ...

    async def finish(self) -> None: ...

    async def receive_event(self) -> ProviderEvent: ...

    async def close(self) -> None: ...


class ASRProvider(Protocol):
    name: str

    async def open(self, config: AudioStreamConfig) -> ASRSession: ...


class UnavailableASRProvider:
    name = "unavailable"

    def __init__(self, reason: str = "Speech recognition is not configured") -> None:
        self.reason = reason

    async def open(self, config: AudioStreamConfig) -> ASRSession:
        del config
        raise ProviderUnavailable(self.reason)

    def capability(self) -> Mapping[str, Any]:
        return {
            "provider": self.name,
            "available": False,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TTSChunk:
    sequence: int
    audio: bytes
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: int = 24_000
    channels: Literal[1] = 1


class TTSStream(Protocol):
    stream_id: str

    def chunks(self) -> AsyncIterator[TTSChunk]: ...

    async def close(self) -> None: ...


class StreamingTTSProvider(Protocol):
    name: str

    async def synthesize(self, text: str, *, locale: str) -> TTSStream: ...


class DisabledTTSProvider:
    """Honest default until a real streaming TTS provider is configured."""

    name = "unavailable"

    async def synthesize(self, text: str, *, locale: str) -> TTSStream:
        del text, locale
        raise TTSUnavailable()

    def capability(self) -> Mapping[str, Any]:
        return {
            "provider": self.name,
            "available": False,
            "reason": "Streaming voice replies are not configured",
        }


def provider_capability(provider: object) -> dict[str, Any]:
    """Return a key-safe, provider-neutral readiness snapshot.

    Providers may expose a richer ``capability`` method.  The fallback only
    reports their public name and configuration state; object representations
    and provider exception strings are intentionally never serialized.
    """

    probe = getattr(provider, "capability", None)
    if callable(probe):
        payload = probe()
        if isinstance(payload, Mapping):
            return {str(key): value for key, value in payload.items()}
    name = str(getattr(provider, "name", "unavailable"))
    return {"provider": name, "available": name != "unavailable"}


def asr_provider_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    transport_factory: RealtimeTransportFactory | None = None,
) -> ASRProvider:
    """Select one provider without implicit provider fallback.

    A DashScope configuration failure never silently switches to local FunASR;
    doing so would change accuracy/latency semantics without operator consent.
    """

    values = os.environ if environ is None else environ
    backend = values.get("SPEECH_ASR_PROVIDER", "dashscope").strip().lower()
    if backend == "dashscope":
        from .dashscope import DashScopeRealtimeProvider

        return DashScopeRealtimeProvider.from_env(
            values, transport_factory=transport_factory
        )
    if backend == "funasr":
        from .funasr import FunASRRealtimeProvider

        return FunASRRealtimeProvider.from_env(
            values, transport_factory=transport_factory
        )
    if backend in {"", "disabled", "unavailable", "none"}:
        return UnavailableASRProvider("Speech recognition is disabled")
    return UnavailableASRProvider("Configured speech recognition provider is unsupported")


def tts_provider_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    transport_factory: RealtimeTransportFactory | None = None,
) -> StreamingTTSProvider:
    """Select TTS explicitly and never fabricate audio or silently fall back."""

    values = os.environ if environ is None else environ
    backend = values.get("SPEECH_TTS_PROVIDER", "dashscope").strip().lower()
    if backend == "dashscope":
        from .tts import DashScopeRealtimeTTSProvider

        return DashScopeRealtimeTTSProvider.from_env(
            values, transport_factory=transport_factory
        )
    if backend in {"", "disabled", "unavailable", "none"}:
        return DisabledTTSProvider()
    return DisabledTTSProvider()


__all__ = [
    "ASRProvider",
    "ASRSession",
    "DisabledTTSProvider",
    "ProviderEvent",
    "provider_capability",
    "StreamingTTSProvider",
    "TTSChunk",
    "TTSStream",
    "UnavailableASRProvider",
    "asr_provider_from_env",
    "tts_provider_from_env",
]
