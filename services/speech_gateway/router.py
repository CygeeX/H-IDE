"""FastAPI router factory; mounting and authorization stay with Control API."""

from __future__ import annotations

import asyncio
import inspect
import json
from contextlib import suppress
from typing import Any, Awaitable, Callable, Literal, Mapping

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .contracts import AudioStreamConfig, ClientCommand, event
from .errors import InvalidClientMessage, SpeechGatewayError
from .gateway import GatewaySession, StreamingSpeechGateway


SpeechAuthorizer = Callable[
    [WebSocket, Mapping[str, Any]], Awaitable[None] | None
]


def create_speech_router(
    gateway: StreamingSpeechGateway,
    *,
    authorize: SpeechAuthorizer,
    path: str = "/v2/speech/stream",
    initial_message_timeout_seconds: float = 5.0,
) -> APIRouter:
    """Create a router that must be mounted with an explicit authorizer."""

    router = APIRouter()

    @router.websocket(path)
    async def stream(websocket: WebSocket) -> None:
        offered = websocket.scope.get("subprotocols") or []
        await websocket.accept(
            subprotocol="speech-stream.v2" if "speech-stream.v2" in offered else None
        )
        session: GatewaySession | None = None
        try:
            initial = await asyncio.wait_for(
                websocket.receive(), timeout=initial_message_timeout_seconds
            )
            payload = _json_message(initial)
            result = authorize(websocket, payload)
            if inspect.isawaitable(result):
                await result
            config = AudioStreamConfig.from_start_message(payload)
            session = await gateway.start(config)
        except asyncio.TimeoutError:
            await _send_error(
                websocket,
                InvalidClientMessage("session.start was not received in time"),
            )
            await _safe_close(websocket, 1008)
            return
        except SpeechGatewayError as exc:
            await _send_error(websocket, exc)
            await _safe_close(websocket, exc.close_code)
            return
        except (ValueError, json.JSONDecodeError):
            error_value = InvalidClientMessage("The first message must be valid JSON")
            await _send_error(websocket, error_value)
            await _safe_close(websocket, error_value.close_code)
            return

        reader = asyncio.create_task(_read_browser(websocket, session))
        writer = asyncio.create_task(_write_browser(websocket, session))
        try:
            done, _ = await asyncio.wait(
                {reader, writer}, return_when=asyncio.FIRST_COMPLETED
            )
            if writer in done:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            else:
                outcome = await reader
                if outcome in {"finish", "error"}:
                    await writer
                else:
                    writer.cancel()
                    await asyncio.gather(writer, return_exceptions=True)
        finally:
            for task in (reader, writer):
                if not task.done():
                    task.cancel()
            await asyncio.gather(reader, writer, return_exceptions=True)
            await session.close()

    return router


async def _read_browser(
    websocket: WebSocket, session: GatewaySession
) -> Literal["finish", "cancel", "disconnect", "error"]:
    try:
        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                await session.cancel()
                return "disconnect"
            frame = message.get("bytes")
            if frame is not None:
                try:
                    session.submit_audio(frame)
                except SpeechGatewayError as exc:
                    await session.fail(exc)
                    return "error"
                continue
            try:
                command = ClientCommand.parse(_json_message(message))
            except (SpeechGatewayError, ValueError, json.JSONDecodeError) as exc:
                error_value = (
                    exc
                    if isinstance(exc, SpeechGatewayError)
                    else InvalidClientMessage("Client message must be valid JSON")
                )
                await session.fail(error_value)
                return "error"
            if command.type == "session.finish":
                await session.finish()
                return "finish"
            if command.type == "session.cancel":
                await session.cancel()
                await _safe_close(websocket, 1000)
                return "cancel"
            assert command.text is not None
            await session.request_tts(command.text)
    except WebSocketDisconnect:
        await session.cancel()
        return "disconnect"


async def _write_browser(websocket: WebSocket, session: GatewaySession) -> None:
    while True:
        item = await session.next_event()
        if item is None:
            await _safe_close(websocket, 1000)
            return
        await websocket.send_json(item.as_dict())


def _json_message(message: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = message.get("text")
    if not isinstance(raw, str):
        raise InvalidClientMessage("Control messages must be JSON text")
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise InvalidClientMessage("Control messages must be JSON objects")
    return payload


async def _send_error(websocket: WebSocket, error_value: SpeechGatewayError) -> None:
    with suppress(Exception):
        await websocket.send_json(
            event(
                "error",
                code=error_value.code,
                message=error_value.message,
                retryable=error_value.retryable,
                fatal=True,
                scope="asr",
            ).as_dict()
        )


async def _safe_close(websocket: WebSocket, code: int) -> None:
    with suppress(Exception):
        await websocket.close(code=code)


__all__ = ["SpeechAuthorizer", "create_speech_router"]
