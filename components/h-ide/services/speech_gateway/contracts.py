"""Stable browser-facing speech-stream v2 contract."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal, Mapping
from uuid import uuid4

from .errors import InvalidAudioFrame, InvalidClientMessage


SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
MIN_FRAME_DURATION_MS = 20
MAX_FRAME_DURATION_MS = 60
MIN_FRAME_BYTES = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES * MIN_FRAME_DURATION_MS // 1000
MAX_FRAME_BYTES = SAMPLE_RATE_HZ * SAMPLE_WIDTH_BYTES * MAX_FRAME_DURATION_MS // 1000
_LOCALE_PATTERN = re.compile(r"^(?:auto|[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$")


@dataclass(frozen=True, slots=True)
class AudioStreamConfig:
    """Only one wire audio shape is accepted in v2.

    Keeping it fixed avoids per-provider transcoding and the latency of building
    complete WAV containers.  ``locale`` may be ``auto`` or a BCP-47-ish tag.
    """

    locale: str = "zh-CN"
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    sample_rate_hz: Literal[16000] = SAMPLE_RATE_HZ
    channels: Literal[1] = CHANNELS
    server_vad: Literal[True] = True

    @classmethod
    def from_start_message(cls, payload: Mapping[str, Any]) -> "AudioStreamConfig":
        if payload.get("type") != "session.start":
            raise InvalidClientMessage("The first message must be session.start")
        audio = payload.get("audio", {})
        if not isinstance(audio, Mapping):
            raise InvalidClientMessage("session.start.audio must be an object")
        expected = {
            "encoding": "pcm_s16le",
            "sampleRateHz": SAMPLE_RATE_HZ,
            "channels": CHANNELS,
        }
        for key, value in expected.items():
            if audio.get(key, value) != value:
                raise InvalidClientMessage(
                    "Audio must be mono PCM16 little-endian at 16000 Hz"
                )
        if payload.get("serverVad", True) is not True:
            raise InvalidClientMessage("speech-stream.v2 requires server VAD")
        locale = payload.get("locale", "zh-CN")
        if (
            not isinstance(locale, str)
            or not locale.strip()
            or len(locale) > 32
            or _LOCALE_PATTERN.fullmatch(locale.strip()) is None
        ):
            raise InvalidClientMessage("locale must be a short non-empty string")
        return cls(locale=locale.strip())


@dataclass(frozen=True, slots=True)
class ClientCommand:
    type: Literal["session.finish", "session.cancel", "tts.synthesize"]
    text: str | None = None

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> "ClientCommand":
        message_type = payload.get("type")
        if message_type in {"session.finish", "session.cancel"}:
            return cls(type=message_type)
        if message_type == "tts.synthesize":
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 4_000:
                raise InvalidClientMessage("tts.synthesize.text is invalid")
            return cls(type=message_type, text=text.strip())
        raise InvalidClientMessage("Unsupported speech-stream.v2 client message")


@dataclass(frozen=True, slots=True)
class SpeechEvent:
    type: str
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.payload}


def event(event_type: str, /, **payload: Any) -> SpeechEvent:
    return SpeechEvent(event_type, payload)


def new_session_id() -> str:
    return f"speech-{uuid4().hex}"


def validate_audio_frame(frame: bytes) -> None:
    if not isinstance(frame, bytes):
        raise InvalidAudioFrame("Audio frames must be binary WebSocket messages")
    size = len(frame)
    if size < MIN_FRAME_BYTES or size > MAX_FRAME_BYTES:
        raise InvalidAudioFrame(
            f"PCM frames must contain {MIN_FRAME_DURATION_MS}-{MAX_FRAME_DURATION_MS} ms of audio"
        )
    if size % SAMPLE_WIDTH_BYTES:
        raise InvalidAudioFrame("PCM16 frames must end on a complete sample")


__all__ = [
    "AudioStreamConfig",
    "CHANNELS",
    "ClientCommand",
    "MAX_FRAME_BYTES",
    "MAX_FRAME_DURATION_MS",
    "MIN_FRAME_BYTES",
    "MIN_FRAME_DURATION_MS",
    "SAMPLE_RATE_HZ",
    "SpeechEvent",
    "event",
    "new_session_id",
    "validate_audio_frame",
]
