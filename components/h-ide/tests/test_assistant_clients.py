from __future__ import annotations

import httpx

from packages.control_core.assistant_models import PlanningEnvelope
from packages.control_core.models import utc_now
from services.control_api.planning_client import MastraPlanningClient
from services.control_api.speech import WhisperCppClient


def planning_envelope() -> PlanningEnvelope:
    return PlanningEnvelope.model_validate(
        {
            "taskRequest": {
                "schemaVersion": "1.0",
                "taskId": "task-client-1",
                "conversationId": "conversation-client-1",
                "robotId": "robot-1",
                "requestedBy": "tester",
                "modality": "text",
                "text": "向前移动",
                "locale": "zh-CN",
                "createdAt": utc_now(),
            },
            "contextVersion": "ctx-client-1",
            "history": [],
            "robotState": {
                "robotId": "robot-1",
                "status": "online",
                "connected": True,
                "safetyMode": "normal",
                "details": {},
            },
            "capabilitySnapshot": {
                "agent": "robot-agent",
                "protocol": "fake",
                "robotId": "robot-1",
                "capabilities": ["move.forward"],
                "safetyMode": "normal",
                "schema": {"type": "object"},
            },
            "commandCatalog": [
                {
                    "commandKey": "move.forward",
                    "version": "1",
                    "kind": "motion",
                    "payloadSchema": {"type": "object"},
                }
            ],
        }
    )


def test_mastra_internal_client_ignores_workstation_proxy_environment(monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(
            202,
            json={"taskId": "task-client-1", "status": "needs_input", "assistantMessage": "距离？"},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", request)
    client = MastraPlanningClient("http://127.0.0.1:4111", "secret")

    result = client.run_turn(planning_envelope(), lambda: False)

    assert result["status"] == "needs_input"
    assert calls[0][2]["trust_env"] is False
    assert calls[0][2]["json"]["capabilitySnapshot"]["schema"] == {"type": "object"}


def test_mastra_internal_client_extracts_strict_result_from_terminal_turn(monkeypatch):
    nested_result = {
        "status": "needs_input",
        "assistantMessage": "请提供目标距离",
    }

    def request(method, url, **kwargs):
        return httpx.Response(
            202,
            json={
                "taskId": "task-client-1",
                "conversationId": "conversation-client-1",
                "status": "needs_input",
                "result": nested_result,
                "events": [],
                "lastEventId": 3,
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", request)
    client = MastraPlanningClient("http://127.0.0.1:4111", "secret")

    assert client.run_turn(planning_envelope(), lambda: False) == nested_result


def test_whisper_internal_client_ignores_workstation_proxy_environment(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200,
            json={"text": "向前移动", "language": "zh"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    result = WhisperCppClient("http://127.0.0.1:8081/inference").transcribe(
        b"wav", filename="voice.wav", locale="zh-CN"
    )

    assert result["text"] == "向前移动"
    assert calls[0][1]["trust_env"] is False
