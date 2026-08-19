"""Validated WAV boundary and local whisper.cpp HTTP adapter."""

from __future__ import annotations

from io import BytesIO
import os
from typing import Any, Mapping, Protocol
import wave

import httpx


MAX_WAV_BYTES = 10 * 1024 * 1024
MAX_WAV_SECONDS = 60.0
WAV_SAMPLE_RATE = 16_000
WAV_CHANNELS = 1
WAV_SAMPLE_WIDTH = 2


class SpeechClient(Protocol):
    def transcribe(
        self,
        wav_bytes: bytes,
        *,
        filename: str,
        locale: str,
    ) -> Mapping[str, Any]: ...


class SpeechServiceUnavailable(RuntimeError):
    pass


class UnavailableSpeechClient:
    def transcribe(self, wav_bytes: bytes, *, filename: str, locale: str):
        raise SpeechServiceUnavailable("whisper.cpp service is not configured")


class WhisperCppClient:
    def __init__(self, endpoint: str, *, timeout_seconds: float = 45.0):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def transcribe(self, wav_bytes: bytes, *, filename: str, locale: str):
        language = locale.split("-", 1)[0].lower() if locale else "auto"
        try:
            response = httpx.post(
                self.endpoint,
                files={"file": (filename, wav_bytes, "audio/wav")},
                data={"response_format": "json", "language": language},
                timeout=self.timeout_seconds,
                trust_env=False,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SpeechServiceUnavailable(f"speech transcription failed: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SpeechServiceUnavailable("speech service returned an invalid response")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise SpeechServiceUnavailable("speech service returned an empty transcript")
        return {"text": text.strip(), "language": payload.get("language") or language}


def validate_wav(wav_bytes: bytes) -> float:
    if not wav_bytes or len(wav_bytes) > MAX_WAV_BYTES:
        raise ValueError("WAV file must be non-empty and no larger than 10 MiB")
    if not wav_bytes.startswith(b"RIFF") or wav_bytes[8:12] != b"WAVE":
        raise ValueError("file is not a WAV container")
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            frame_rate = audio.getframerate()
            frames = audio.getnframes()
    except (wave.Error, EOFError) as exc:
        raise ValueError("file is not a valid PCM WAV") from exc
    if channels != WAV_CHANNELS:
        raise ValueError("WAV must be mono")
    if sample_width != WAV_SAMPLE_WIDTH:
        raise ValueError("WAV must use 16-bit PCM samples")
    if frame_rate != WAV_SAMPLE_RATE:
        raise ValueError("WAV must use a 16 kHz sample rate")
    duration = frames / frame_rate if frame_rate else 0
    if duration <= 0 or duration > MAX_WAV_SECONDS:
        raise ValueError("WAV duration must be between 0 and 60 seconds")
    return duration


def speech_client_from_env() -> SpeechClient:
    endpoint = os.getenv("WHISPER_ENDPOINT", "").strip()
    if not endpoint:
        return UnavailableSpeechClient()
    return WhisperCppClient(
        endpoint,
        timeout_seconds=float(os.getenv("WHISPER_TIMEOUT_SECONDS", "45")),
    )


__all__ = [
    "MAX_WAV_BYTES",
    "SpeechClient",
    "SpeechServiceUnavailable",
    "UnavailableSpeechClient",
    "WhisperCppClient",
    "speech_client_from_env",
    "validate_wav",
]
