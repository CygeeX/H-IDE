from __future__ import annotations

from typing import Any

from services.control_api.mission_client import MastraMissionPlanningClient


class _Response:
    status_code = 200
    text = "{}"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"plan": {"ok": True}}


class _Client:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None

    def post(self, _url: str, *, headers: dict[str, str], json: dict[str, Any]) -> _Response:
        assert headers["Content-Type"] == "application/json"
        self.payload = json
        return _Response()


def test_mission_draft_client_projects_h_ide_bookkeeping_at_agent_boundary() -> None:
    http = _Client()
    client = MastraMissionPlanningClient(
        "http://planning-agent:4111",
        "internal-token",
        client=http,
    )

    result = client.create_draft(
        {
            "requestId": "request-1",
            "missionId": "mission-1",
            "robotId": "g1-sim",
            "requestedBy": "operator",
            "objective": "巡检办公室",
            "locale": "zh-CN",
            "modality": "text",
            "contextVersion": "context-1",
            "envelope": {},
            "capabilities": [],
            "history": [],
            "worldRecords": [],
            "conversationId": "conversation-1",
            "sourceTaskId": "task-1",
            "semanticReferents": [{"referentId": "referent-1"}],
        }
    )

    assert result == {"plan": {"ok": True}}
    assert http.payload is not None
    assert "conversationId" not in http.payload
    assert "sourceTaskId" not in http.payload
    assert "semanticReferents" not in http.payload
    assert set(http.payload) == {
        "requestId",
        "missionId",
        "robotId",
        "requestedBy",
        "objective",
        "locale",
        "modality",
        "contextVersion",
        "envelope",
        "capabilities",
        "history",
        "worldRecords",
    }
