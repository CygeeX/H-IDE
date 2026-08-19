"""Provider-neutral streaming session with bounded audio ingestion."""

from __future__ import annotations

import asyncio
from base64 import b64encode
from contextlib import suppress
from typing import Any

from .contracts import (
    AudioStreamConfig,
    SpeechEvent,
    event,
    new_session_id,
    validate_audio_frame,
)
from .errors import (
    AudioBackpressure,
    ProviderDisconnected,
    ProviderTimeout,
    SpeechGatewayError,
    TTSUnavailable,
)
from .providers import (
    ASRProvider,
    ASRSession,
    DisabledTTSProvider,
    ProviderEvent,
    StreamingTTSProvider,
)


_AUDIO_FINISH = object()
_EVENT_END = object()


class StreamingSpeechGateway:
    def __init__(
        self,
        asr_provider: ASRProvider,
        *,
        tts_provider: StreamingTTSProvider | None = None,
        audio_queue_frames: int = 24,
        event_queue_size: int = 128,
        finish_timeout_seconds: float = 12.0,
    ) -> None:
        if audio_queue_frames < 1:
            raise ValueError("audio_queue_frames must be positive")
        if event_queue_size < 8:
            raise ValueError("event_queue_size must be at least 8")
        self.asr_provider = asr_provider
        self.tts_provider = tts_provider or DisabledTTSProvider()
        self.audio_queue_frames = audio_queue_frames
        self.event_queue_size = event_queue_size
        self.finish_timeout_seconds = finish_timeout_seconds

    async def start(self, config: AudioStreamConfig) -> "GatewaySession":
        provider_session = await self.asr_provider.open(config)
        session = GatewaySession(
            session_id=new_session_id(),
            config=config,
            provider_name=self.asr_provider.name,
            provider_session=provider_session,
            tts_provider=self.tts_provider,
            audio_queue_frames=self.audio_queue_frames,
            event_queue_size=self.event_queue_size,
            finish_timeout_seconds=self.finish_timeout_seconds,
        )
        session.start()
        return session


class GatewaySession:
    def __init__(
        self,
        *,
        session_id: str,
        config: AudioStreamConfig,
        provider_name: str,
        provider_session: ASRSession,
        tts_provider: StreamingTTSProvider,
        audio_queue_frames: int,
        event_queue_size: int,
        finish_timeout_seconds: float,
    ) -> None:
        self.session_id = session_id
        self.config = config
        self.provider_name = provider_name
        self._provider_session = provider_session
        self._tts_provider = tts_provider
        self._audio_queue: asyncio.Queue[bytes | object] = asyncio.Queue(
            maxsize=audio_queue_frames
        )
        self._events: asyncio.Queue[SpeechEvent | object] = asyncio.Queue(
            maxsize=event_queue_size
        )
        self._finish_timeout_seconds = finish_timeout_seconds
        self._sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._finish_watch_task: asyncio.Task[None] | None = None
        self._tts_task: asyncio.Task[None] | None = None
        self._terminal_lock = asyncio.Lock()
        self._finishing = False
        self._terminal = False
        self._closed = False

    def start(self) -> None:
        if self._sender_task is not None:
            raise RuntimeError("speech session is already started")
        self._events.put_nowait(
            event(
                "session.started",
                sessionId=self.session_id,
                protocol="speech-stream.v2",
                provider=self.provider_name,
                audio={
                    "encoding": self.config.encoding,
                    "sampleRateHz": self.config.sample_rate_hz,
                    "channels": self.config.channels,
                    "frameDurationMs": {"min": 20, "max": 60},
                },
                serverVad=True,
                tts={"provider": self._tts_provider.name},
            )
        )
        self._sender_task = asyncio.create_task(
            self._send_audio_loop(), name=f"{self.session_id}-asr-send"
        )
        self._receiver_task = asyncio.create_task(
            self._receive_provider_loop(), name=f"{self.session_id}-asr-receive"
        )

    def submit_audio(self, frame: bytes) -> None:
        if self._finishing or self._terminal or self._closed:
            raise ProviderDisconnected()
        validate_audio_frame(frame)
        try:
            self._audio_queue.put_nowait(frame)
        except asyncio.QueueFull as exc:
            raise AudioBackpressure() from exc

    async def finish(self) -> None:
        if self._terminal or self._closed or self._finishing:
            return
        self._finishing = True
        try:
            self._audio_queue.put_nowait(_AUDIO_FINISH)
        except asyncio.QueueFull:
            # Waiting is safe here: after session.finish no new browser audio is
            # accepted, while the sender continues draining earlier frames.
            await self._audio_queue.put(_AUDIO_FINISH)
        self._finish_watch_task = asyncio.create_task(
            self._watch_finish_timeout(), name=f"{self.session_id}-finish-timeout"
        )

    async def request_tts(self, text: str) -> None:
        if self._terminal or self._closed:
            return
        if self._tts_task is not None and not self._tts_task.done():
            await self._emit(
                event(
                    "error",
                    code="TTS_BUSY",
                    message="A voice reply is already streaming",
                    retryable=True,
                    fatal=False,
                    scope="tts",
                )
            )
            return
        self._tts_task = asyncio.create_task(
            self._stream_tts(text), name=f"{self.session_id}-tts"
        )

    async def fail(self, error: SpeechGatewayError) -> None:
        await self._terminate(error)

    async def next_event(self) -> SpeechEvent | None:
        item = await self._events.get()
        if item is _EVENT_END:
            return None
        assert isinstance(item, SpeechEvent)
        return item

    async def wait_closed(self, *, timeout: float | None = None) -> None:
        tasks = [task for task in (self._sender_task, self._receiver_task) if task]
        if not tasks:
            return
        waiter = asyncio.gather(*tasks, return_exceptions=True)
        if timeout is None:
            await waiter
        else:
            await asyncio.wait_for(waiter, timeout=timeout)

    async def cancel(self) -> None:
        await self.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        current = asyncio.current_task()
        tasks = [
            task
            for task in (
                self._sender_task,
                self._receiver_task,
                self._finish_watch_task,
                self._tts_task,
            )
            if task is not None and task is not current and not task.done()
        ]
        for task in tasks:
            task.cancel()
        await self._provider_session.close()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_audio_loop(self) -> None:
        try:
            while True:
                item = await self._audio_queue.get()
                if item is _AUDIO_FINISH:
                    await self._provider_session.finish()
                    return
                assert isinstance(item, bytes)
                await self._provider_session.send_audio(item)
        except asyncio.CancelledError:
            raise
        except SpeechGatewayError as exc:
            await self._terminate(exc)
        except Exception as exc:
            await self._terminate(ProviderDisconnected(), cause=exc)

    async def _receive_provider_loop(self) -> None:
        try:
            while True:
                provider_event = await self._provider_session.receive_event()
                if provider_event.kind == "session.finished":
                    await self._complete()
                    return
                await self._emit(self._map_provider_event(provider_event))
        except asyncio.CancelledError:
            raise
        except SpeechGatewayError as exc:
            await self._terminate(exc)
        except Exception as exc:
            await self._terminate(ProviderDisconnected(), cause=exc)

    def _map_provider_event(self, provider_event: ProviderEvent) -> SpeechEvent:
        payload: dict[str, Any] = {"sessionId": self.session_id}
        if provider_event.utterance_id is not None:
            payload["utteranceId"] = provider_event.utterance_id
        if provider_event.text is not None:
            payload["text"] = provider_event.text
        if provider_event.confirmed_text is not None:
            payload["confirmedText"] = provider_event.confirmed_text
        if provider_event.language is not None:
            payload["language"] = provider_event.language
        if provider_event.emotion is not None:
            payload["emotion"] = provider_event.emotion
        if provider_event.offset_ms is not None:
            payload["offsetMs"] = provider_event.offset_ms
        return event(provider_event.kind, **payload)

    async def _watch_finish_timeout(self) -> None:
        try:
            await asyncio.sleep(self._finish_timeout_seconds)
            if not self._terminal:
                await self._terminate(ProviderTimeout("session finalization"))
        except asyncio.CancelledError:
            return

    async def _stream_tts(self, text: str) -> None:
        stream = None
        try:
            stream = await self._tts_provider.synthesize(text, locale=self.config.locale)
            await self._emit(
                event(
                    "tts.started",
                    sessionId=self.session_id,
                    streamId=stream.stream_id,
                    provider=self._tts_provider.name,
                )
            )
            expected_sequence = 0
            async for chunk in stream.chunks():
                if not chunk.audio or chunk.sequence != expected_sequence:
                    raise TTSUnavailable()
                expected_sequence += 1
                await self._emit(
                    event(
                        "tts.audio.chunk",
                        sessionId=self.session_id,
                        streamId=stream.stream_id,
                        sequence=chunk.sequence,
                        audio=b64encode(chunk.audio).decode("ascii"),
                        encoding=chunk.encoding,
                        sampleRateHz=chunk.sample_rate_hz,
                        channels=chunk.channels,
                    )
                )
            await self._emit(
                event(
                    "tts.ended",
                    sessionId=self.session_id,
                    streamId=stream.stream_id,
                )
            )
        except asyncio.CancelledError:
            raise
        except SpeechGatewayError as exc:
            await self._emit(
                event(
                    "error",
                    code=exc.code,
                    message=exc.message,
                    retryable=exc.retryable,
                    fatal=False,
                    scope="tts",
                )
            )
        except Exception:
            await self._emit(
                event(
                    "error",
                    code="TTS_FAILED",
                    message="Voice reply streaming failed",
                    retryable=True,
                    fatal=False,
                    scope="tts",
                )
            )
        finally:
            if stream is not None:
                with suppress(Exception):
                    await stream.close()

    async def _complete(self) -> None:
        async with self._terminal_lock:
            if self._terminal:
                return
            self._terminal = True
            if self._finish_watch_task is not None:
                self._finish_watch_task.cancel()
            await self._provider_session.close()
            await self._events.put(_EVENT_END)

    async def _terminate(
        self, error_value: SpeechGatewayError, *, cause: Exception | None = None
    ) -> None:
        del cause  # Intentionally never serialized or logged by this layer.
        async with self._terminal_lock:
            if self._terminal:
                return
            self._terminal = True
            await self._emit(
                event(
                    "error",
                    code=error_value.code,
                    message=error_value.message,
                    retryable=error_value.retryable,
                    fatal=True,
                    scope="asr",
                )
            )
            await self._provider_session.close()
            await self._events.put(_EVENT_END)

    async def _emit(self, item: SpeechEvent) -> None:
        await self._events.put(item)


__all__ = ["GatewaySession", "StreamingSpeechGateway"]
