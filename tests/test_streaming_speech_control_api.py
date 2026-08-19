from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from services.control_api.auth import CSRF_COOKIE
from services.control_api.main import create_app
from services.speech_gateway.contracts import AudioStreamConfig
from services.speech_gateway.gateway import StreamingSpeechGateway
from services.speech_gateway.providers import ProviderEvent, TTSChunk


TOKEN = "speech-control-api-test-token"
PASSWORD = "correct horse battery staple"
PCM_20_MS = b"\x00\x00" * 320


class StubASRSession:
    def __init__(self) -> None:
        self.events: asyncio.Queue[ProviderEvent] = asyncio.Queue()
        self.frames: list[bytes] = []
        self.closed = False

    async def send_audio(self, frame: bytes) -> None:
        self.frames.append(frame)

    async def finish(self) -> None:
        self.events.put_nowait(
            ProviderEvent(
                "transcript.final",
                utterance_id="utterance-1",
                text="你好",
                confirmed_text="你好",
            )
        )
        self.events.put_nowait(ProviderEvent("session.finished"))

    async def receive_event(self) -> ProviderEvent:
        return await self.events.get()

    async def close(self) -> None:
        self.closed = True


class StubASRProvider:
    name = "test-realtime-asr"

    def __init__(self) -> None:
        self.sessions: list[StubASRSession] = []

    async def open(self, config: AudioStreamConfig) -> StubASRSession:
        assert config.sample_rate_hz == 16_000
        session = StubASRSession()
        self.sessions.append(session)
        return session

    def capability(self):
        return {"provider": self.name, "available": True, "model": "test-asr"}


class StubTTSStream:
    stream_id = "tts-control-test"

    def __init__(self) -> None:
        self.closed = False

    async def chunks(self) -> AsyncIterator[TTSChunk]:
        yield TTSChunk(sequence=0, audio=b"\x01\x02")
        yield TTSChunk(sequence=1, audio=b"\x03\x04")

    async def close(self) -> None:
        self.closed = True


class StubTTSProvider:
    name = "test-realtime-tts"

    def __init__(self) -> None:
        self.streams: list[StubTTSStream] = []
        self.requests: list[tuple[str, str]] = []

    async def synthesize(self, text: str, *, locale: str) -> StubTTSStream:
        self.requests.append((text, locale))
        stream = StubTTSStream()
        self.streams.append(stream)
        return stream

    def capability(self):
        return {"provider": self.name, "available": True, "model": "test-tts"}


def gateway() -> tuple[StreamingSpeechGateway, StubASRProvider, StubTTSProvider]:
    asr = StubASRProvider()
    tts = StubTTSProvider()
    return StreamingSpeechGateway(asr, tts_provider=tts), asr, tts


def register_and_login(client: TestClient) -> str:
    registered = client.post(
        "/api/v1/auth/register",
        json={"username": "speech-user", "password": PASSWORD},
    )
    assert registered.status_code == 201, registered.text
    login = client.post(
        "/api/v1/auth/login",
        json={"username": "speech-user", "password": PASSWORD},
    )
    assert login.status_code == 200, login.text
    return login.json()["csrfToken"]


def test_control_api_mounts_cookie_csrf_and_bearer_speech_websocket() -> None:
    realtime, asr, _ = gateway()
    app = create_app(auth_token=TOKEN, streaming_speech_gateway=realtime)
    with TestClient(app) as client:
        csrf = register_and_login(client)

        with client.websocket_connect("/api/v2/speech/stream") as socket:
            socket.send_json({"type": "session.start", "locale": "zh-CN"})
            denied = socket.receive_json()
            assert denied["code"] == "SPEECH_AUTHORIZATION_DENIED"

        with client.websocket_connect(
            "/api/v2/speech/stream", subprotocols=["speech-stream.v2"]
        ) as socket:
            socket.send_json(
                {"type": "session.start", "locale": "zh-CN", "csrfToken": csrf}
            )
            assert socket.receive_json()["type"] == "session.started"
            socket.send_bytes(PCM_20_MS)
            socket.send_json({"type": "session.finish"})
            final = socket.receive_json()
            assert final["type"] == "transcript.final"
            assert final["text"] == "你好"

        # Explicit Bearer callers do not use browser CSRF, but credentials in
        # a URL are refused even if they match the legacy deployment token.
        with client.websocket_connect(
            "/api/v2/speech/stream",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as socket:
            socket.send_json({"type": "session.start", "locale": "zh-CN"})
            assert socket.receive_json()["type"] == "session.started"
            socket.send_json({"type": "session.cancel"})

        with client.websocket_connect(
            f"/api/v2/speech/stream?token={TOKEN}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as socket:
            socket.send_json({"type": "session.start", "locale": "zh-CN"})
            error = socket.receive_json()
            assert error["code"] == "INVALID_CLIENT_MESSAGE"
            assert "URL" in error["message"]

    assert asr.sessions[0].frames == [PCM_20_MS]
    assert all(session.closed for session in asr.sessions)


def test_control_api_speech_health_and_tts_are_authenticated_and_stream_real_bytes() -> None:
    realtime, _, tts = gateway()
    app = create_app(auth_token=TOKEN, streaming_speech_gateway=realtime)
    with TestClient(app) as client:
        assert client.get("/api/v2/speech/healthz").status_code == 401
        headers = {"Authorization": f"Bearer {TOKEN}"}
        health = client.get("/api/v2/speech/healthz", headers=headers)
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["asr"]["model"] == "test-asr"
        ready = client.get("/api/v2/speech/readyz", headers=headers)
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"

        response = client.post(
            "/api/v2/speech/tts",
            headers=headers,
            json={"text": "你好", "locale": "zh-CN"},
        )
        assert response.status_code == 200, response.text
        assert response.content == b"\x01\x02\x03\x04"
        assert response.headers["x-speech-audio-encoding"] == "pcm_s16le"
        assert response.headers["x-speech-sample-rate-hz"] == "24000"
        assert tts.requests == [("你好", "zh-CN")]
        assert tts.streams[0].closed is True


def test_cookie_tts_post_reuses_normal_csrf_boundary() -> None:
    realtime, _, tts = gateway()
    app = create_app(auth_token=TOKEN, streaming_speech_gateway=realtime)
    with TestClient(app) as client:
        csrf = register_and_login(client)
        denied = client.post("/api/v2/speech/tts", json={"text": "你好"})
        assert denied.status_code == 403
        response = client.post(
            "/api/v2/speech/tts",
            headers={"X-CSRF-Token": csrf},
            json={"text": "你好"},
        )
        assert response.status_code == 200
        assert client.cookies.get(CSRF_COOKIE) == csrf
        assert len(tts.streams) == 1
