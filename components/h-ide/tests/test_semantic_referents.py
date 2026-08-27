from __future__ import annotations

from services.control_api.conversation_memory import ConversationMemoryContextBuilder
from services.control_api.main import create_app
from services.control_api.semantic_referents import extract_semantic_referents


def visual_result() -> dict:
    return {
        "analysisId": "analysis-office-objects-1",
        "source": {
            "robotId": "robot-1",
            "robotBootId": "boot-semantic-1",
            "localizationEpochId": "epoch-semantic-1",
            "observationSessionId": "observation-semantic-1",
            "capturedAt": "2026-08-27T00:00:00Z",
            "sceneId": "office-scene",
            "sceneRevision": 1,
        },
        "hypotheses": [
            {
                "hypothesisId": "hypothesis-chair-1",
                "label": "chair",
                "pixelBox": [10, 20, 110, 220],
                "score": 0.96,
            },
            {
                "hypothesisId": "hypothesis-shelf-1",
                "label": "shelf",
                "pixelBox": [130, 30, 300, 250],
                "score": 0.91,
            },
        ],
        "evidence": [
            {
                "kind": "input_rgb",
                "sha256": "a" * 64,
                "mediaType": "image/jpeg",
            }
        ],
    }


def test_visual_entities_are_coordinate_free_and_bilingual() -> None:
    referents = extract_semantic_referents(
        conversation_id="conversation-semantic-1",
        task_id="task-semantic-observe-1",
        robot_id="robot-1",
        visual_result=visual_result(),
        region_id="open_office",
    )

    assert [item.label for item in referents] == ["chair", "shelf"]
    assert "椅子" in referents[0].aliases
    assert "架子" in referents[1].aliases
    serialized = str([item.model_dump(mode="json") for item in referents])
    assert "targetPose" not in serialized
    assert "landmarkPose" not in serialized
    assert all(item.executionEligible is False for item in referents)


def test_referent_is_available_to_the_next_turn_by_chinese_alias() -> None:
    app = create_app(auth_token="semantic-referent-test-token")
    conversation = app.state.assistant.create_conversation(
        {
            "robotId": "robot-1",
            "requestedBy": "operator",
            "locale": "zh-CN",
        }
    )
    conversation_id = conversation["conversationId"]
    task_id = "task-semantic-observe-1"
    app.state.store.create_assistant_turn(
        {
            "taskId": task_id,
            "conversationId": conversation_id,
            "text": "你现在能看到什么",
            "modality": "text",
        },
        message_id="message-semantic-observe-user-1",
    )
    assert app.state.store.complete_assistant_turn(
        task_id,
        status="completed",
        result={"status": "completed"},
        assistant_message="我看到了椅子和架子。",
        message_id="message-semantic-observe-assistant-1",
        conversation_status="idle",
    )
    current_robot = app.state.store.get_robot("robot-1") or {}
    details = current_robot.get("details") or {}
    observed = visual_result()
    if isinstance(details.get("robotBootId"), str):
        observed["source"]["robotBootId"] = details["robotBootId"]
    if isinstance(details.get("localizationEpochId"), str):
        observed["source"]["localizationEpochId"] = details[
            "localizationEpochId"
        ]
    referents = extract_semantic_referents(
        conversation_id=conversation_id,
        task_id=task_id,
        robot_id="robot-1",
        visual_result=observed,
        region_id="open_office",
    )
    assert app.state.store.record_assistant_semantic_referents(
        conversation_id=conversation_id,
        robot_id="robot-1",
        referents=[item.model_dump(mode="json") for item in referents],
    ) == 2

    prepared = ConversationMemoryContextBuilder(app.state.store).prepare(
        conversation_id,
        "去椅子旁边站着",
    )

    assert [item.label for item in prepared.memory_context.semanticReferents] == [
        "chair"
    ]
    assert prepared.memory_context.semanticReferents[0].authority == "SEMANTIC_ONLY"
    assert "referentId=" in prepared.mission_history[-1]["content"]


def test_completed_mission_projects_report_and_visual_entities_to_source_turn() -> None:
    app = create_app(auth_token="semantic-projection-test-token")
    conversation = app.state.assistant.create_conversation(
        {
            "robotId": "robot-1",
            "requestedBy": "operator",
            "locale": "zh-CN",
        }
    )
    conversation_id = conversation["conversationId"]
    task_id = "task-semantic-mission-1"
    mission_id = "mission-semantic-projection-1"
    app.state.store.create_assistant_turn(
        {
            "taskId": task_id,
            "conversationId": conversation_id,
            "text": "去开放办公室观察",
            "modality": "text",
        },
        message_id="message-semantic-mission-user-1",
    )
    assert app.state.store.complete_assistant_turn(
        task_id,
        status="draft_ready",
        result={
            "schemaVersion": "h-ide.assistant-result/2",
            "intent": "ROBOT_ACTION",
            "route": "mission_review",
            "routeStatus": "review",
            "assistantMessage": "请审核任务。",
            "missionDraftId": mission_id,
        },
        assistant_message="请审核任务。",
        message_id="message-semantic-mission-assistant-1",
        conversation_status="review",
    )
    record = {
        "missionId": mission_id,
        "robotId": "robot-1",
        "snapshot": {"status": "completed", "detail": None},
        "skillExecutions": [
            {
                "request": {"input": {"regionId": "open_office"}},
                "result": {"result": {"visualResult": visual_result()}},
            },
            {
                "request": {"input": {"regionId": "open_office"}},
                "result": {
                    "result": {
                        "missionReport": {
                            "narrativeZhCN": "开放办公室观察报告：发现椅子和架子。"
                        }
                    }
                },
            },
        ],
    }

    app.state.mission._project_mission_conversation_artifacts(record)
    app.state.mission._project_mission_conversation_artifacts(record)

    messages = app.state.store.list_assistant_messages(conversation_id)
    assert [item["content"] for item in messages].count(
        "开放办公室观察报告：发现椅子和架子。"
    ) == 1
    referents = app.state.store.list_assistant_semantic_referents(
        conversation_id
    )
    assert {item["label"] for item in referents} == {"chair", "shelf"}
    assert all(item["sourceMissionId"] == mission_id for item in referents)
