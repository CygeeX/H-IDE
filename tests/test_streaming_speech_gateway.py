from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Mapping

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from services.speech_gateway.contracts import AudioStreamConfig
from services.speech_gateway.dashscope import (
    DashScopeRealtimeConfig,
    DashScopeRealtimeProvider,
)
from services.speech_gateway.errors import (
    AudioBackpressure,
    InvalidAudioFrame,
    ProviderDisconnected,
    ProviderTimeout,
    ProviderUnavailable,
)
from services.speech_gateway.funasr import FunASRRealtimeProvider
from services.speech_gateway.gateway import StreamingSpeechGateway
from services.speech_gateway.providers import (
    ProviderEvent,
    TTSChunk,
    asr_provider_from_env,
)
from services.speech_gateway.router import create_speech_router


PCM_20_MS = b"\x00\x00" * 320


class FakeTransport:
    def __init__(self, incoming: list[Mapping[str, Any] | Exception] | None = None):
        self.incoming: asyncio.Queue[Mapping[str, Any] | Exception] = asyncio.Queue()
        for item in incoming or []:
            self.incoming.put_nowait(item)
        self.sent_json: list[dict[str, Any]] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False

    async def send_json(self, payload: Mapping[str, Any]) -> None:
        self.sent_json.append(dict(payload))

    async def send_bytes(self, payload: bytes) -> None:
        self.sent_bytes.append(payload)

    async def receive_json(self) -> Mapping[str, Any]:
        item = await self.incoming.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


class FakeTransportFactory:
    def __init__(self, transport: FakeTransport):
        self.transport = transport
        self.calls: list[dict[str, Any]] = []

    async def connect(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        open_timeout_seconds: float,
    ) -> FakeTransport:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "open_timeout_seconds": open_timeout_seconds,
            }
        )
        return self.transport


class FakeASRSession:
    def __init__(self, *, events_on_finish: list[ProviderEvent] | None = None):
        self.events: asyncio.Queue[ProviderEvent] = asyncio.Queue()
        self.events_on_finish = events_on_finish or []
        self.frames: list[bytes] = []
        self.finished = False
        self.closed = False
        self.send_gate: asyncio.Event | None = None

    async def send_audio(self, frame: bytes) -> None:
        if self.send_gate is not None:
            await self.send_gate.wait()
        self.frames.append(frame)

    async def finish(self) -> None:
        self.finished = True
        for item in self.events_on_finish:
            self.events.put_nowait(item)

    async def receive_event(self) -> ProviderEvent:
        return await self.events.get()

    async def close(self) -> None:
        self.closed = True


class FakeASRProvider:
    name = "test-asr"

    def __init__(self, session: FakeASRSession):
        self.session = session
        self.open_count = 0

    async def open(self, config: AudioStreamConfig) -> FakeASRSession:
        assert config.sample_rate_hz == 16_000
        self.open_count += 1
        return self.session


def test_dashscope_reuses_one_connection_and_maps_realtime_events():
    async def scenario() -> None:
        transport = FakeTransport(
            [
                {"type": "session.created"},
                {"type": "session.updated"},
                {
                    "type": "input_audio_buffer.speech_started",
                    "item_id": "utterance-1",
                    "audio_start_ms": 20,
                },
                {
                    "type": "conversation.item.input_audio_transcription.text",
                    "item_id": "utterance-1",
                    "text": "你",
                    "stash": "好",
                    "language": "zh",
                    "emotion": "neutral",
                },
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "utterance-1",
                    "transcript": "你好",
                    "language": "zh",
                    "emotion": "neutral",
                },
                {
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": "utterance-1",
                    "audio_end_ms": 540,
                },
                {"type": "session.finished"},
            ]
        )
        factory = FakeTransportFactory(transport)
        provider = DashScopeRealtimeProvider(
            DashScopeRealtimeConfig(
                api_key="secret-key-must-not-leak",
                endpoint_url="wss://workspace.example/api-ws/v1/realtime?model=qwen",
            ),
            transport_factory=factory,
        )
        session = await provider.open(AudioStreamConfig())
        await session.send_audio(PCM_20_MS)
        await session.send_audio(PCM_20_MS)

        events = [await session.receive_event() for _ in range(4)]
        await session.finish()
        finished = await session.receive_event()

        assert len(factory.calls) == 1
        assert len([item for item in transport.sent_json if item["type"] == "session.update"]) == 1
        assert len(
            [item for item in transport.sent_json if item["type"] == "input_audio_buffer.append"]
        ) == 2
        assert [item.kind for item in events] == [
            "speech.started",
            "transcript.partial",
            "transcript.final",
            "speech.ended",
        ]
        assert events[1].text == "你好"
        assert events[1].confirmed_text == "你"
        assert events[2].text == "你好"
        assert finished.kind == "session.finished"
        assert transport.sent_json[-1]["type"] == "session.finish"

    asyncio.run(scenario())


def test_dashscope_disconnect_is_sanitized_and_key_repr_is_redacted():
    async def scenario() -> None:
        config = DashScopeRealtimeConfig(
            api_key="secret-key-must-not-leak",
            endpoint_url="wss://workspace.example/api-ws/v1/realtime?model=qwen",
        )
        assert "secret-key-must-not-leak" not in repr(config)
        transport = FakeTransport(
            [{"type": "session.created"}, {"type": "session.updated"}, RuntimeError("wire url")]
        )
        session = await DashScopeRealtimeProvider(
            config, transport_factory=FakeTransportFactory(transport)
        ).open(AudioStreamConfig())
        with pytest.raises(ProviderDisconnected) as error:
            await session.receive_event()
        assert "wire url" not in str(error.value)

    asyncio.run(scenario())


def test_dashscope_event_timeout_closes_as_a_retryable_provider_error():
    async def scenario() -> None:
        transport = FakeTransport(
            [{"type": "session.created"}, {"type": "session.updated"}]
        )
        session = await DashScopeRealtimeProvider(
            DashScopeRealtimeConfig(
                api_key="secret",
                endpoint_url="wss://workspace.example/api-ws/v1/realtime?model=qwen",
                event_timeout_seconds=0.01,
            ),
            transport_factory=FakeTransportFactory(transport),
        ).open(AudioStreamConfig())
        with pytest.raises(ProviderTimeout) as error:
            await session.receive_event()
        assert error.value.retryable is True
        await session.close()
        assert transport.closed is True

    asyncio.run(scenario())


def test_missing_key_fails_closed_without_opening_transport():
    async def scenario() -> None:
        transport = FakeTransport()
        factory = FakeTransportFactory(transport)
        provider = asr_provider_from_env(
            {
                "SPEECH_ASR_PROVIDER": "dashscope",
                "DASHSCOPE_ASR_WS_URL": "wss://workspace.example/api-ws/v1/realtime",
            },
            transport_factory=factory,
        )
        with pytest.raises(ProviderUnavailable, match="DASHSCOPE_API_KEY"):
            await provider.open(AudioStreamConfig())
        assert factory.calls == []

    asyncio.run(scenario())


def test_audio_queue_backpressure_never_drops_a_frame_silently():
    async def scenario() -> None:
        provider_session = FakeASRSession()
        provider_session.send_gate = asyncio.Event()
        session = await StreamingSpeechGateway(
            FakeASRProvider(provider_session), audio_queue_frames=1
        ).start(AudioStreamConfig())
        session.submit_audio(PCM_20_MS)
        with pytest.raises(AudioBackpressure):
            session.submit_audio(PCM_20_MS)
        await session.cancel()
        assert provider_session.closed is True

    asyncio.run(scenario())


def test_audio_frame_contract_accepts_20_to_60_ms_only():
    async def scenario() -> None:
        provider_session = FakeASRSession()
        session = await StreamingSpeechGateway(FakeASRProvider(provider_session)).start(
            AudioStreamConfig()
        )
        session.submit_audio(b"\x00\x00" * 320)
        session.submit_audio(b"\x00\x00" * 960)
        with pytest.raises(InvalidAudioFrame):
            session.submit_audio(b"\x00\x00" * 319)
        with pytest.raises(InvalidAudioFrame):
            session.submit_audio(b"\x00" * 641)
        await session.cancel()

    asyncio.run(scenario())


def test_gateway_emits_partial_final_and_closes_after_finish():
    async def scenario() -> None:
        provider_session = FakeASRSession(
            events_on_finish=[
                ProviderEvent("speech.started", utterance_id="u1", offset_ms=0),
                ProviderEvent(
                    "transcript.partial",
                    utterance_id="u1",
                    text="前进",
                    confirmed_text="前",
                ),
                ProviderEvent(
                    "transcript.final",
                    utterance_id="u1",
                    text="前进一米",
                    confirmed_text="前进一米",
                ),
                ProviderEvent("speech.ended", utterance_id="u1", offset_ms=700),
                ProviderEvent("session.finished"),
            ]
        )
        session = await StreamingSpeechGateway(FakeASRProvider(provider_session)).start(
            AudioStreamConfig()
        )
        assert (await session.next_event()).type == "session.started"
        session.submit_audio(PCM_20_MS)
        await session.finish()
        output = []
        while (item := await session.next_event()) is not None:
            output.append(item.as_dict())
        assert [item["type"] for item in output] == [
            "speech.started",
            "transcript.partial",
            "transcript.final",
            "speech.ended",
        ]
        assert output[2]["text"] == "前进一米"
        assert provider_session.finished is True
        assert provider_session.frames == [PCM_20_MS]

    asyncio.run(scenario())


def test_cancel_closes_provider_without_waiting_for_a_transcript():
    async def scenario() -> None:
        provider_session = FakeASRSession()
        session = await StreamingSpeechGateway(FakeASRProvider(provider_session)).start(
            AudioStreamConfig()
        )
        await session.cancel()
        assert provider_session.closed is True

    asyncio.run(scenario())


class FakeTTSStream:
    stream_id = "tts-1"

    def __init__(self):
        self.closed = False

    async def chunks(self) -> AsyncIterator[TTSChunk]:
        yield TTSChunk(sequence=0, audio=b"\x01\x02")
        yield TTSChunk(sequence=1, audio=b"\x03\x04")

    async def close(self) -> None:
        self.closed = True


class FakeTTSProvider:
    name = "test-tts"

    def __init__(self):
        self.stream = FakeTTSStream()

    async def synthesize(self, text: str, *, locale: str) -> FakeTTSStream:
        assert text == "你好"
        assert locale == "zh-CN"
        return self.stream


def test_tts_chunk_contract_and_disabled_state_are_honest():
    async def enabled() -> None:
        asr = FakeASRSession()
        tts = FakeTTSProvider()
        session = await StreamingSpeechGateway(
            FakeASRProvider(asr), tts_provider=tts
        ).start(AudioStreamConfig())
        await session.next_event()
        await session.request_tts("你好")
        output = [(await session.next_event()).as_dict() for _ in range(4)]
        assert [item["type"] for item in output] == [
            "tts.started",
            "tts.audio.chunk",
            "tts.audio.chunk",
            "tts.ended",
        ]
        assert output[1]["audio"] == "AQI="
        assert tts.stream.closed is True
        await session.cancel()

    async def disabled() -> None:
        session = await StreamingSpeechGateway(FakeASRProvider(FakeASRSession())).start(
            AudioStreamConfig()
        )
        await session.next_event()
        await session.request_tts("你好")
        error = (await session.next_event()).as_dict()
        assert error == {
            "type": "error",
            "code": "TTS_UNAVAILABLE",
            "message": "Streaming voice replies are not configured",
            "retryable": False,
            "fatal": False,
            "scope": "tts",
        }
        await session.cancel()

    asyncio.run(enabled())
    asyncio.run(disabled())


def test_funasr_is_explicit_and_maps_two_pass_output():
    async def scenario() -> None:
        missing = FunASRRealtimeProvider.from_env({}, transport_factory=None)
        with pytest.raises(ProviderUnavailable):
            await missing.open(AudioStreamConfig())

        transport = FakeTransport(
            [
                {"mode": "2pass-online", "text": "你", "is_final": False},
                {"mode": "2pass-offline", "text": "你好", "is_final": True},
                {"is_end": True, "is_final": True},
            ]
        )
        factory = FakeTransportFactory(transport)
        provider = FunASRRealtimeProvider.from_env(
            {"FUNASR_WS_URL": "ws://127.0.0.1:10095"}, transport_factory=factory
        )
        session = await provider.open(AudioStreamConfig())
        await session.send_audio(PCM_20_MS)
        await session.finish()
        assert (await session.receive_event()).kind == "transcript.partial"
        final = await session.receive_event()
        assert final.kind == "transcript.final"
        assert final.text == "你好"
        assert (await session.receive_event()).kind == "session.finished"
        assert len(factory.calls) == 1
        assert transport.sent_bytes == [PCM_20_MS]
        assert transport.sent_json[0]["mode"] == "2pass"
        assert transport.sent_json[-1] == {"is_speaking": False, "is_end": True}

    asyncio.run(scenario())


def test_fastapi_router_is_mountable_and_requires_injected_authorizer():
    provider_session = FakeASRSession(
        events_on_finish=[
            ProviderEvent("transcript.partial", text="你好", confirmed_text="你"),
            ProviderEvent("transcript.final", text="你好", confirmed_text="你好"),
            ProviderEvent("session.finished"),
        ]
    )
    authorized: list[bool] = []

    async def authorize(websocket: WebSocket, start: Mapping[str, Any]) -> None:
        del websocket
        authorized.append(start["type"] == "session.start")

    app = FastAPI()
    app.include_router(
        create_speech_router(
            StreamingSpeechGateway(FakeASRProvider(provider_session)),
            authorize=authorize,
        )
    )
    with TestClient(app).websocket_connect(
        "/v2/speech/stream", subprotocols=["speech-stream.v2"]
    ) as websocket:
        websocket.send_json(
            {
                "type": "session.start",
                "locale": "zh-CN",
                "audio": {
                    "encoding": "pcm_s16le",
                    "sampleRateHz": 16_000,
                    "channels": 1,
                },
                "serverVad": True,
            }
        )
        assert websocket.receive_json()["type"] == "session.started"
        websocket.send_bytes(PCM_20_MS)
        websocket.send_json({"type": "session.finish"})
        assert websocket.receive_json()["type"] == "transcript.partial"
        assert websocket.receive_json()["type"] == "transcript.final"
    assert authorized == [True]
    assert provider_session.frames == [PCM_20_MS]


def test_fastapi_router_surfaces_backpressure_before_closing():
    provider_session = FakeASRSession()
    provider_session.send_gate = asyncio.Event()

    async def authorize(websocket: WebSocket, start: Mapping[str, Any]) -> None:
        del websocket, start

    app = FastAPI()
    app.include_router(
        create_speech_router(
            StreamingSpeechGateway(
                FakeASRProvider(provider_session), audio_queue_frames=1
            ),
            authorize=authorize,
        )
    )
    with TestClient(app).websocket_connect("/v2/speech/stream") as websocket:
        websocket.send_json({"type": "session.start"})
        assert websocket.receive_json()["type"] == "session.started"
        for _ in range(10):
            websocket.send_bytes(PCM_20_MS)
        error = websocket.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "AUDIO_BACKPRESSURE"
        assert error["fatal"] is True
