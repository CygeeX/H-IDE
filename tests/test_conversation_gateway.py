from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from packages.control_core.assistant_models import (
    ConversationGatewayResult,
    ConversationIntent,
    ConversationMemoryContext,
    ConversationTurnRequest,
    ConversationTurnResponse,
)
from services.control_api.conversation_client import (
    ConversationRoutingUnavailable,
    MastraConversationClient,
)
from services.control_api.conversation_gateway import ConversationGateway
from services.control_api.main import create_app
from services.control_api.mission_policy import MissionPolicyDenied


TOKEN = "conversation-v2-test-token"
PASSWORD = "correct horse battery staple"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


class RoutedConversationClient:
    configured = True

    def __init__(self, selector=None):
        self.selector = selector
        self.requests: list[ConversationTurnRequest] = []

    def route_turn(self, request: ConversationTurnRequest) -> ConversationTurnResponse:
        self.requests.append(request)
        if self.selector is not None:
            intent, message = self.selector(request)
        else:
            intent, message = ConversationIntent.SOCIAL_CHAT, "你好，很高兴见到你。"
        needs_robot = intent in {
            ConversationIntent.ROBOT_OBSERVE,
            ConversationIntent.ROBOT_ACTION,
            ConversationIntent.SCENE_TEACH,
        }
        return ConversationTurnResponse(
            schemaVersion="h-ide.conversation-result/2",
            conversationId=request.conversationId,
            robotId=request.robotId,
            intent=intent,
            assistantMessage=message,
            needsRobot=needs_robot,
            shouldCreateMission=intent is ConversationIntent.ROBOT_ACTION,
        )


class FailingConversationClient:
    configured = True

    def route_turn(self, request: ConversationTurnRequest) -> ConversationTurnResponse:
        del request
        raise AssertionError("deterministic office inspection must not call the router")


class VisualFixture:
    def __init__(self):
        self.requests: list[dict[str, Any]] = []

    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = dict(request)
        self.requests.append(payload)
        return {
            "schemaVersion": "h-ide.visual-result/2",
            "analysisId": "analysis-readonly-1",
            "requestId": payload["requestId"],
            "requestSha256": "a" * 64,
            "missionId": payload["missionId"],
            "revisionId": payload["revisionId"],
            "nodeId": payload["nodeId"],
            "attempt": 1,
            "mode": "scene_description",
            "status": "succeeded",
            "source": {
                "robotId": payload["robotId"],
                "robotBootId": "boot-readonly-1",
                "localizationEpochId": "epoch-readonly-1",
                "observationSessionId": "observation-readonly-1",
                "capturedAt": "2026-08-27T00:00:00Z",
            },
            "versions": {},
            "hypotheses": [
                {
                    "hypothesisId": "hypothesis-chair-readonly-1",
                    "source": "qwen",
                    "label": "chair",
                    "pixelBox": [10, 20, 110, 220],
                    "score": 0.95,
                },
                {
                    "hypothesisId": "hypothesis-shelf-readonly-1",
                    "source": "qwen",
                    "label": "shelf",
                    "pixelBox": [130, 20, 300, 240],
                    "score": 0.91,
                },
            ],
            "geometryCandidates": [],
            "observationSummary": "我看到一张会议桌和两把椅子。",
            "evidence": [
                {
                    "evidenceId": "evidence-rgb-1",
                    "kind": "input_rgb",
                    "sha256": "b" * 64,
                    "mediaType": "image/jpeg",
                }
            ],
            "warnings": [],
        }

    def get_evidence(self, analysis_id: str, sha256: str):
        del analysis_id, sha256
        return b"fixture", "image/jpeg"


def wait_turn(client: TestClient, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/assistant/turns/{task_id}", headers=auth())
        assert response.status_code == 200, response.text
        turn = response.json()
        if turn["status"] not in {"queued", "running"}:
            return turn
        time.sleep(0.01)
    raise AssertionError("assistant turn did not finish")


def test_typed_http_client_uses_internal_v2_endpoint_and_validates_flags(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(
            200,
            json={
                "schemaVersion": "h-ide.conversation-result/2",
                "conversationId": "conversation-1",
                "intent": "GENERAL_QA",
                "assistantMessage": "4",
                "needsRobot": False,
                "shouldCreateMission": False,
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    request = ConversationTurnRequest(
        conversationId="conversation-1",
        text="2+2 等于多少？",
    )
    response = MastraConversationClient("http://planning-agent:4111", "secret").route_turn(
        request
    )

    assert response.intent is ConversationIntent.GENERAL_QA
    assert calls[0][0].endswith("/internal/v2/conversation-turns")
    assert calls[0][1]["trust_env"] is False
    assert calls[0][1]["json"].get("robotId") is None


def test_exact_office_inspection_bypasses_remote_conversation_router() -> None:
    briefs: list[dict[str, Any]] = []

    def create_draft(brief: Mapping[str, Any]) -> Mapping[str, Any]:
        briefs.append(dict(brief))
        return {"draftId": "mission-office-deterministic-1"}

    result = ConversationGateway(
        FailingConversationClient(),
        mission_draft_creator=create_draft,
    ).handle(
        task_id="task-office-deterministic-1",
        request=ConversationTurnRequest(
            conversationId="conversation-office-deterministic-1",
            robotId="g1-sim",
            locale="zh-CN",
            text="走遍所有办公室，并把所有办公室里面的信息和路上的信息都给我",
        ),
        requested_by="operator-test",
        modality="text",
        mission_history=[],
    )

    assert result.intent is ConversationIntent.ROBOT_ACTION
    assert result.route.value == "mission_review"
    assert result.routeStatus == "review"
    assert result.missionDraftId == "mission-office-deterministic-1"
    assert briefs == [
        {
            "conversationId": "conversation-office-deterministic-1",
            "sourceTaskId": "task-office-deterministic-1",
            "robotId": "g1-sim",
            "requestedBy": "operator-test",
            "objective": "走遍所有办公室，并把所有办公室里面的信息和路上的信息都给我",
            "locale": "zh-CN",
            "modality": "text",
            "history": [],
        }
    ]


def test_ordered_room_inspection_phrases_bypass_remote_router_and_share_intent() -> None:
    briefs: list[dict[str, Any]] = []
    gateway = ConversationGateway(
        FailingConversationClient(),
        mission_draft_creator=lambda brief: briefs.append(dict(brief))
        or {"draftId": f"mission-office-sequence-{len(briefs)}"},
    )
    phrases = [
        "今天巡检开放办公室和会议室",
        "请依次进入开放办公区和会议室，分别在房间内部停稳后进行多方向观察，记录房间内及沿途看到的信息，最后按房间分别汇总",
    ]

    results = []
    for index, text in enumerate(phrases, start=1):
        results.append(
            gateway.handle(
                task_id=f"task-office-sequence-{index}",
                request=ConversationTurnRequest(
                    conversationId=f"conversation-office-sequence-{index}",
                    robotId="g1-sim",
                    locale="zh-CN",
                    text=text,
                ),
                requested_by="operator-test",
                modality="text",
                mission_history=[],
            )
        )

    assert all(item.intent is ConversationIntent.ROBOT_ACTION for item in results)
    assert all(item.routeStatus == "review" for item in results)
    assert [item["objective"] for item in briefs] == phrases


def test_selected_inspection_skill_routes_single_room_without_remote_llm() -> None:
    briefs: list[dict[str, Any]] = []
    result = ConversationGateway(
        FailingConversationClient(),
        mission_draft_creator=lambda brief: briefs.append(dict(brief))
        or {"draftId": "mission-inspection-skill-single-room"},
    ).handle(
        task_id="task-inspection-skill-single-room",
        request=ConversationTurnRequest(
            conversationId="conversation-inspection-skill-single-room",
            robotId="g1-sim",
            skillId="inspection.office",
            locale="zh-CN",
            text="今天巡检会议室",
        ),
        requested_by="operator-test",
        modality="text",
        mission_history=[],
    )

    assert result.routeStatus == "review"
    assert result.missionDraftId == "mission-inspection-skill-single-room"
    assert briefs[0]["skillId"] == "inspection.office"
    assert briefs[0]["objective"] == "今天巡检会议室"


def test_unbound_robot_returns_selection_guidance_instead_of_service_failure() -> None:
    def reject(_brief):
        raise MissionPolicyDenied("robot has no immutable site/execution-target binding")

    gateway = ConversationGateway(
        RoutedConversationClient(),
        mission_draft_creator=reject,
    )

    result = gateway.handle(
        task_id="task-unbound-robot",
        request=ConversationTurnRequest(
            conversationId="conversation-unbound-robot",
            robotId="g1-dev",
            locale="zh-CN",
            text="今天巡检开放办公室和会议室",
        ),
        requested_by="operator-test",
        modality="text",
        mission_history=[],
    )

    assert result.routeStatus == "needs_robot"
    assert "在线的仿真 G1" in result.assistantMessage


def test_known_object_and_exit_actions_override_a_misclassified_router_result() -> None:
    calls: list[dict[str, Any]] = []
    router = RoutedConversationClient(
        lambda request: (ConversationIntent.ROBOT_OBSERVE, "我只看看。")
    )
    gateway = ConversationGateway(
        router,
        mission_draft_creator=lambda brief: calls.append(dict(brief))
        or {"draftId": f"mission-known-{len(calls) + 1}"},
    )

    for index, text in enumerate(("去椅子旁边站着", "走出开放办公室"), start=1):
        result = gateway.handle(
            task_id=f"task-known-{index}",
            request=ConversationTurnRequest(
                conversationId=f"conversation-known-{index}",
                robotId="g1-sim",
                text=text,
            ),
            requested_by="operator-test",
            modality="text",
            mission_history=[],
        )
        assert result.intent is ConversationIntent.ROBOT_ACTION
        assert result.routeStatus == "review"

    assert [item["objective"] for item in calls] == [
        "去椅子旁边站着",
        "走出开放办公室",
    ]


def test_typed_http_client_falls_back_once_for_pre_memory_agent(monkeypatch):
    calls = []
    summary_text = "H-IDE 生成的旧轮次摘要"

    def post(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            return httpx.Response(
                422,
                json={"detail": "Unrecognized key: memoryContext"},
                request=httpx.Request("POST", url),
            )
        return httpx.Response(
            200,
            json={
                "schemaVersion": "h-ide.conversation-result/2",
                "conversationId": "conversation-legacy-agent",
                "intent": "GENERAL_QA",
                "assistantMessage": "兼容响应",
                "needsRobot": False,
                "shouldCreateMission": False,
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    request = ConversationTurnRequest(
        conversationId="conversation-legacy-agent",
        text="继续",
        memoryContext=ConversationMemoryContext.model_validate(
            {
                "resourceId": "principal-legacy",
                "revision": 3,
                "durableSummary": {
                    "summaryId": "summary-legacy",
                    "revision": 2,
                    "text": summary_text,
                    "coveredThroughTurn": 30,
                    "generatedAt": "2026-08-19T00:00:00+00:00",
                    "method": "SERVER_COMPACTION",
                    "contentSha256": hashlib.sha256(summary_text.encode()).hexdigest(),
                },
            }
        ),
    )
    result = MastraConversationClient(
        "http://planning-agent:4111", "secret"
    ).route_turn(request)

    assert result.assistantMessage == "兼容响应"
    assert "memoryContext" in calls[0][1]["json"]
    assert "memoryContext" not in calls[1][1]["json"]
    assert calls[1][1]["json"]["summary"] == summary_text
    assert all(call[1]["trust_env"] is False for call in calls)


def test_browser_cannot_forge_server_owned_memory_context():
    router = RoutedConversationClient()
    app = create_app(auth_token=TOKEN, conversation_client=router)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/assistant/conversations",
            headers=auth(),
            json={"locale": "zh-CN"},
        ).json()
        submitted = client.post(
            f"/api/v1/assistant/conversations/{created['conversationId']}/messages",
            headers=auth(),
            json={
                "text": "你好",
                "memoryContext": {
                    "providedBy": "BROWSER",
                    "authority": "PHYSICAL_TRUTH",
                    "resourceId": "attacker",
                    "revision": 999,
                },
            },
        )
        assert submitted.status_code == 202, submitted.text
        wait_turn(client, submitted.json()["taskId"])

        sent = router.requests[-1]
        assert sent.memoryContext is not None
        assert sent.memoryContext.providedBy == "H_IDE_SERVER"
        assert sent.memoryContext.authority == "CONVERSATIONAL_ONLY"
        assert sent.memoryContext.resourceId != "attacker"
        assert sent.memoryContext.revision == 1


def test_typed_http_client_strictly_rejects_escalated_new_agent_response(monkeypatch):
    def post(url, **kwargs):
        del kwargs
        return httpx.Response(
            200,
            json={
                "schemaVersion": "h-ide.conversation-result/2",
                "conversationId": "conversation-strict",
                "intent": "ROBOT_ACTION",
                "assistantMessage": "准备行动草案。",
                "needsRobot": True,
                "shouldCreateMission": True,
                "rememberedTargetPose": {"x": 8, "y": 9},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ConversationRoutingUnavailable, match="violated its contract"):
        MastraConversationClient(
            "http://planning-agent:4111", "secret"
        ).route_turn(
            ConversationTurnRequest(
                conversationId="conversation-strict",
                robotId="robot-1",
                text="去那里",
            )
        )


def test_gateway_only_robot_action_can_create_a_mission_draft():
    calls = []
    client = RoutedConversationClient(
        lambda request: (ConversationIntent.ROBOT_ACTION, "我已整理为待审核行动。")
    )
    gateway = ConversationGateway(
        client,
        mission_draft_creator=lambda brief: calls.append(dict(brief)) or {"draftId": "mission-1"},
    )

    result = gateway.handle(
        task_id="task-1",
        request=ConversationTurnRequest(
            conversationId="conversation-1",
            robotId="robot-1",
            text="向前走一米",
        ),
        requested_by="user-1",
        modality="text",
        mission_history=[],
    )

    assert result.route.value == "mission_review"
    assert result.routeStatus == "review"
    assert result.missionDraftId == "mission-1"
    assert calls == [
        {
            "conversationId": "conversation-1",
            "sourceTaskId": "task-1",
            "robotId": "robot-1",
            "requestedBy": "user-1",
            "objective": "向前走一米",
            "locale": "zh-CN",
            "modality": "text",
            "history": [],
        }
    ]

    with pytest.raises(ValidationError):
        ConversationGatewayResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "intent": "GENERAL_QA",
            }
        )


def test_unbound_conversation_supports_long_natural_chat_without_plan_drafts():
    router = RoutedConversationClient(
        lambda request: (
            (ConversationIntent.SOCIAL_CHAT, "你好，很高兴见到你。")
            if "你好" in request.text
            else (ConversationIntent.GENERAL_QA, "2 + 2 等于 4。")
        )
    )
    app = create_app(auth_token=TOKEN, conversation_client=router)
    with TestClient(app) as client:
        created = client.post(
            "/api/v1/assistant/conversations",
            headers=auth(),
            json={"locale": "zh-CN"},
        )
        assert created.status_code == 201, created.text
        conversation_id = created.json()["conversationId"]
        assert created.json()["robotId"] is None

        first = client.post(
            f"/api/v1/assistant/conversations/{conversation_id}/messages",
            headers=auth(),
            json={"text": "你好"},
        )
        first_turn = wait_turn(client, first.json()["taskId"])
        assert first_turn["status"] == "completed"
        assert first_turn["result"]["route"] == "reply"
        assert first_turn["result"].get("missionDraftId") is None

        second = client.post(
            f"/api/v1/assistant/conversations/{conversation_id}/messages",
            headers=auth(),
            json={"text": "2+2 等于多少？"},
        )
        second_turn = wait_turn(client, second.json()["taskId"])
        assert second_turn["result"]["intent"] == "GENERAL_QA"
        assert [item.content for item in router.requests[-1].history] == [
            "你好",
            "你好，很高兴见到你。",
        ]

        conversation = client.get(
            f"/api/v1/assistant/conversations/{conversation_id}", headers=auth()
        ).json()
        assert [message["content"] for message in conversation["messages"]] == [
            "你好",
            "你好，很高兴见到你。",
            "2+2 等于多少？",
            "2 + 2 等于 4。",
        ]
        assert conversation["drafts"] == []
        assert app.state.store.list_mission_drafts() == []

        events = client.get(
            f"/api/v1/assistant/turns/{second.json()['taskId']}/events",
            headers=auth(),
        ).json()
        assert [item["type"] for item in events["events"]] == [
            "turn.accepted",
            "turn.started",
            "route.selected",
            "turn.completed",
        ]


def test_observe_is_read_only_visual_task_and_scene_teach_stays_pending():
    router = RoutedConversationClient(
        lambda request: (
            (ConversationIntent.ROBOT_OBSERVE, "我会只观察当前画面。")
            if "观察" in request.text
            else (ConversationIntent.SCENE_TEACH, "我会先保存为待确认的场景教学。")
        )
    )
    visual = VisualFixture()
    app = create_app(
        auth_token=TOKEN,
        conversation_client=router,
        visual_task_client=visual,
    )
    with TestClient(app) as client:
        conversation = client.post(
            "/api/v1/assistant/conversations",
            headers=auth(),
            json={"robotId": "robot-1"},
        ).json()
        observe = client.post(
            f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
            headers=auth(),
            json={"text": "只观察当前画面，不要移动"},
        )
        observed = wait_turn(client, observe.json()["taskId"])
        assert observed["result"]["route"] == "visual_observation"
        assert observed["result"]["routeStatus"] == "completed"
        assert observed["result"]["assistantMessage"] == "我看到一张会议桌和两把椅子。"
        assert observed["result"]["evidence"][0]["kind"] == "input_rgb"
        evidence_url = observed["result"]["evidence"][0]["proxyUrl"]
        evidence_response = client.get(evidence_url, headers=auth())
        assert evidence_response.status_code == 200
        assert evidence_response.content == b"fixture"
        assert evidence_response.headers["cache-control"] == "private, no-store, max-age=0"
        assert visual.requests[0]["mode"] == "scene_description"
        assert visual.requests[0]["authorizationId"].startswith("assistant-readonly-")
        assert app.state.store.list_mission_drafts() == []
        assert client.get("/api/v1/commands", headers=auth()).json() == []
        referents = app.state.store.list_assistant_semantic_referents(
            conversation["conversationId"]
        )
        assert {item["label"] for item in referents} == {"chair", "shelf"}
        assert all(item["executionEligible"] is False for item in referents)

        teaching = client.post(
            f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
            headers=auth(),
            json={"text": "记住这里叫会议区"},
        )
        taught = wait_turn(client, teaching.json()["taskId"])
        assert taught["result"]["route"] == "scene_teaching"
        assert taught["result"]["routeStatus"] == "awaiting_confirmation"
        assert len(visual.requests) == 1
        assert app.state.store.list_mission_drafts() == []


def test_action_mission_draft_inherits_conversation_owner_and_site():
    router = RoutedConversationClient(
        lambda request: (ConversationIntent.ROBOT_ACTION, "行动方案已生成，等待你审核。")
    )
    app = create_app(auth_token=TOKEN, conversation_client=router)
    with TestClient(app) as client:
        registered = client.post(
            "/api/v1/auth/register",
            json={
                "username": "mission-owner",
                "displayName": "Mission Owner",
                "password": PASSWORD,
            },
        ).json()["user"]
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "mission-owner", "password": PASSWORD},
        ).json()
        csrf = login["csrfToken"]
        gateway = app.state.assistant.conversation_gateway
        gateway.mission_draft_creator = lambda brief: {"draftId": "mission-owned-1"}
        created = client.post(
            "/api/v1/assistant/conversations",
            headers={"X-CSRF-Token": csrf},
            json={"robotId": "robot-1"},
        )
        assert created.status_code == 201, created.text
        submitted = client.post(
            f"/api/v1/assistant/conversations/{created.json()['conversationId']}/messages",
            headers={"X-CSRF-Token": csrf},
            json={"text": "向前走一米"},
        )
        assert submitted.status_code == 202, submitted.text
        turn = wait_turn(client, submitted.json()["taskId"])
        assert turn["result"]["missionDraftId"] == "mission-owned-1"
        binding = app.state.store.get_resource_owner("mission_draft", "mission-owned-1")
        assert binding is not None
        assert binding["ownerUserId"] == registered["userId"]
        assert binding["siteId"] == "default"
