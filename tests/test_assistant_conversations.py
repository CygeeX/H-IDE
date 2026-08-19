from __future__ import annotations

import asyncio
from datetime import datetime
from io import BytesIO
import threading
import time
import wave

from fastapi.testclient import TestClient

from packages.control_core.adapters import FakeAdapter
from packages.control_core.models import RobotState, utc_now
from packages.scene_core import (
    NavigationPayload,
    SceneContext,
    SceneMapSummary,
    SceneRef,
    SemanticSpatialIntent,
    RelativeSpatialIntent,
    TargetMapCheck,
    resolve_spatial_target,
    build_spatial_navigation_execution_payload,
)
from services.control_api.main import create_app
from services.control_api.scene_client import SceneContextConflict


TOKEN = "assistant-test-token"


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def proposal(summary="先移动，再打开夹爪"):
    return {
        "robotId": "robot-1",
        "summary": summary,
        "steps": [
            {
                "stepId": "move",
                "commandKey": "move.forward",
                "payload": {},
                "dependsOn": [],
                "rationale": "移动到工作位置",
                "contributedBy": "motion-worker",
            },
            {
                "stepId": "open",
                "commandKey": "gripper.open",
                "payload": {},
                "dependsOn": ["move"],
                "rationale": "打开夹爪",
                "contributedBy": "manipulation-worker",
            },
        ],
        "assumptions": [],
        "warnings": ["执行前确认周围无人"],
        "contributions": [
            {
                "agentId": "motion-worker",
                "contribution": "motion",
                "summary": "移动步骤",
                "steps": [
                    {
                        "stepId": "move",
                        "commandKey": "move.forward",
                        "payload": {},
                        "dependsOn": [],
                        "rationale": "移动到工作位置",
                    }
                ],
                "assumptions": [],
                "warnings": [],
            },
            {
                "agentId": "manipulation-worker",
                "contribution": "manipulation",
                "summary": "夹爪步骤",
                "steps": [
                    {
                        "stepId": "open",
                        "commandKey": "gripper.open",
                        "payload": {},
                        "dependsOn": ["move"],
                        "rationale": "打开夹爪",
                    }
                ],
                "assumptions": [],
                "warnings": [],
            },
        ],
        "contextVersion": "robot-1:test",
    }


class FakePlanningClient:
    def __init__(self, results=None, blocker: threading.Event | None = None):
        self.results = list(results or [])
        self.blocker = blocker
        self.envelopes = []
        self.cancelled = []

    def run_turn(self, envelope, is_cancelled):
        self.envelopes.append(envelope.model_dump(mode="json", by_alias=False))
        if self.blocker is not None:
            while not self.blocker.wait(0.01):
                if is_cancelled():
                    return {"status": "cancelled", "assistantMessage": "已取消"}
        if self.results:
            result = self.results.pop(0)
        else:
            result = {"status": "draft_ready", "assistantMessage": "草案已生成", "proposal": proposal()}
        if result.get("status") == "draft_ready":
            result["proposal"]["contextVersion"] = envelope.contextVersion
        return result

    def cancel_turn(self, task_id):
        self.cancelled.append(task_id)


class FakeSpeechClient:
    def __init__(self, text="向前移动后打开夹爪"):
        self.text = text
        self.calls = []
        self.ran_on_event_loop = None

    def transcribe(self, wav_bytes, *, filename, locale):
        try:
            asyncio.get_running_loop()
            self.ran_on_event_loop = True
        except RuntimeError:
            self.ran_on_event_loop = False
        self.calls.append((wav_bytes, filename, locale))
        return {"text": self.text, "language": "zh"}


def scene_context() -> SceneContext:
    return SceneContext.model_validate(
        {
            "sceneRef": {"sceneId": "scene-demo", "revision": 1},
            "bounds": {
                "min": {"x": -20, "y": -20, "z": -1},
                "max": {"x": 20, "y": 20, "z": 3},
            },
            "annotations": [
                {
                    "locationId": "kitchen",
                    "name": "厨房",
                    "aliases": ["kitchen"],
                    "polygonXY": [[2, 2], [6, 2], [6, 6], [2, 6]],
                    "zMin": 0,
                    "zMax": 2.5,
                    "entryPose": {"frameId": "map", "x": 2, "y": 3, "z": 0, "yaw": 0},
                }
            ],
        }
    )


def test_spatial_execution_payload_adds_live_binding_without_changing_plan_evidence():
    context = scene_context()
    resolution = resolve_spatial_target(
        context,
        SemanticSpatialIntent(locationId="kitchen"),
        "resolution-review-1",
        utc_now(),
    )
    planned = NavigationPayload(
        goalPose=resolution.goalPose,
        spatialResolution=resolution,
    )
    before = planned.model_dump(mode="json")

    execution = build_spatial_navigation_execution_payload(
        planned,
        {
            "observationSessionId": "observation-live-1",
            "robotBootId": "boot-live-1",
            "localizationEpochId": "epoch-live-1",
        },
        mission_id="draft-1",
        stage_id="navigate",
    )

    assert execution.payloadType == "spatial_resolution"
    assert execution.spatialResolutionId == "resolution-review-1"
    assert execution.goalPose == planned.goalPose
    assert execution.sceneRef == {"sceneId": "scene-demo", "revision": 1}
    assert planned.model_dump(mode="json") == before


class StaticSceneContextClient:
    def __init__(self):
        self.context = scene_context()

    def get_context(
        self,
        robot_id,
        *,
        current_pose,
        expected_ref=None,
        selected_point=None,
    ):
        if expected_ref is not None and expected_ref != self.context.sceneRef:
            raise SceneContextConflict("robot scene binding changed")
        return self.context.model_copy(
            update={"currentPose": current_pose, "selectedPoint": selected_point}
        )


class CheckedSceneContextClient(StaticSceneContextClient):
    def __init__(self):
        super().__init__()
        self.context = self.context.model_copy(update={
            "mapSummary": SceneMapSummary(
                mapVersion="scene-demo-r1-map",
                frameId="map",
                verified=True,
                status="ready",
                resolution=0.05,
                groundZ=0,
                observedCoverageRatio=0.8,
                confidence="high",
                warnings=[],
            ),
        })
        self.checked_points = []

    def check_target(self, scene_ref, point_map):
        self.checked_points.append((scene_ref, point_map))
        return TargetMapCheck(
            mapVersion="scene-demo-r1-map",
            classification="occupied",
            clearanceMeters=0.35,
            distanceToObservedObstacleMeters=0.1,
            warnings=["目标点净空范围内检测到点云障碍证据，请人工复核"],
        )


class SpatialPlanningClient:
    def __init__(self, *, relative=False):
        self.relative = relative
        self.envelopes = []

    def run_turn(self, envelope, is_cancelled):
        self.envelopes.append(envelope.model_dump(mode="json", by_alias=False))
        intent = (
            RelativeSpatialIntent(forwardMeters=5, lateralMeters=0, yawDeltaRad=0)
            if self.relative
            else SemanticSpatialIntent(locationId="kitchen")
        )
        resolution = resolve_spatial_target(
            envelope.sceneContext,
            intent,
            "resolution-test",
            utc_now(),
        )
        payload = NavigationPayload(
            goalPose=resolution.goalPose,
            spatialResolution=resolution,
        ).model_dump(mode="json")
        fragment = {
            "agentId": "motion-worker",
            "contribution": "navigation",
            "summary": "空间目标解析",
            "steps": [{
                "stepId": "navigate",
                "commandKey": "navigation.navigate_to_pose",
                "payload": payload,
                "dependsOn": [],
                "rationale": "导航到确定的 map 位姿",
            }],
            "assumptions": [],
            "warnings": [],
        }
        return {
            "status": "draft_ready",
            "assistantMessage": "空间计划已生成",
            "proposal": {
                "robotId": envelope.taskRequest.robotId,
                "summary": "导航到目标位置",
                "steps": [{**fragment["steps"][0], "contributedBy": "motion-worker"}],
                "assumptions": [],
                "warnings": [],
                "contributions": [fragment],
                "contextVersion": envelope.contextVersion,
            },
        }

    def cancel_turn(self, task_id):
        return None


def create_conversation(client):
    response = client.post(
        "/api/v1/assistant/conversations",
        headers=auth(),
        json={"robotId": "robot-1", "requestedBy": "tester", "locale": "zh-CN"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def wait_turn(client, task_id, expected=("draft_ready", "needs_input", "cannot_plan", "failed", "cancelled")):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/assistant/turns/{task_id}", headers=auth())
        assert response.status_code == 200, response.text
        turn = response.json()
        if turn["status"] in expected:
            return turn
        time.sleep(0.01)
    raise AssertionError(f"turn {task_id} did not finish")


def test_text_turn_builds_immutable_draft_without_dispatch_then_confirmation_executes():
    planning = FakePlanningClient()
    adapter = FakeAdapter()
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter}, planning_client=planning)
    client = TestClient(app)
    conversation = create_conversation(client)

    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "向前移动后打开夹爪", "modality": "text"},
    )
    assert accepted.status_code == 202, accepted.text
    turn = wait_turn(client, accepted.json()["taskId"])
    assert turn["status"] == "draft_ready"
    assert adapter.dispatches == []
    assert turn["draft"]["status"] == "draft"
    assert turn["draft"]["review"]["contributions"][0]["agentId"] == "motion-worker"

    envelope = planning.envelopes[0]
    assert envelope["taskRequest"]["conversationId"] == conversation["conversationId"]
    assert [item["commandKey"] for item in envelope["commandCatalog"]] == [
        "gripper.open",
        "move.forward",
        "navigation.navigate_to_pose",
        "stop",
    ]
    assert all("mapping" not in item and "endpoint" not in item for item in envelope["commandCatalog"])
    navigation = next(
        item for item in envelope["commandCatalog"]
        if item["commandKey"] == "navigation.navigate_to_pose"
    )
    assert navigation["payloadSchema"]["type"] == "object"
    assert set(navigation["payloadSchema"]["required"]) == {
        "goalPose", "spatialResolution"
    }
    assert "visualResolutionId" not in navigation["payloadSchema"].get("properties", {})

    confirmed = client.post(
        f"/api/v1/assistant/drafts/{turn['draft']['draftId']}/confirm",
        headers=auth(),
        json={"confirmed": True},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert [request.commandKey for request, _ in adapter.dispatches] == ["move.forward", "gripper.open"]
    refreshed = client.get(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}", headers=auth()
    ).json()
    assert refreshed["status"] == "completed"


def test_plan_draft_review_window_is_configurable():
    planning = FakePlanningClient()
    app = create_app(
        auth_token=TOKEN,
        planning_client=planning,
        assistant_draft_ttl_seconds=900,
    )
    client = TestClient(app)
    conversation = create_conversation(client)

    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "向前移动", "modality": "text"},
    )
    turn = wait_turn(client, accepted.json()["taskId"])
    plan = turn["draft"]["plan"]

    created_at = datetime.fromisoformat(plan["createdAt"])
    expires_at = datetime.fromisoformat(plan["expiresAt"])
    assert (expires_at - created_at).total_seconds() == 900


def test_published_scene_semantic_navigation_updates_fake_pose_only_after_confirmation():
    planning = SpatialPlanningClient()
    scenes = StaticSceneContextClient()
    adapter = FakeAdapter()
    app = create_app(
        auth_token=TOKEN,
        adapters={"fake": adapter},
        planning_client=planning,
        scene_context_client=scenes,
    )
    client = TestClient(app)
    conversation = create_conversation(client)
    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={
            "text": "去厨房",
            "modality": "text",
            "sceneSelection": {"sceneRef": {"sceneId": "scene-demo", "revision": 1}},
        },
    )
    turn = wait_turn(client, accepted.json()["taskId"])
    assert turn["status"] == "draft_ready"
    assert adapter.dispatches == []
    assert planning.envelopes[0]["sceneContext"]["annotations"][0]["locationId"] == "kitchen"

    confirmed = client.post(
        f"/api/v1/assistant/drafts/{turn['draft']['draftId']}/confirm",
        headers=auth(),
        json={"confirmed": True},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert [item.commandKey for item, _ in adapter.dispatches] == [
        "navigation.navigate_to_pose"
    ]
    robot = client.get("/api/v1/robots", headers=auth()).json()[0]
    assert (robot["pose"]["x"], robot["pose"]["y"]) == (2.0, 3.0)
    assert adapter.poses["robot-1"].x == 2


def test_formal_map_target_check_is_attached_as_review_warning_without_rewriting_plan():
    planning = SpatialPlanningClient()
    scenes = CheckedSceneContextClient()
    app = create_app(
        auth_token=TOKEN,
        planning_client=planning,
        scene_context_client=scenes,
    )
    client = TestClient(app)
    conversation = create_conversation(client)
    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "去厨房", "modality": "text"},
    )
    turn = wait_turn(client, accepted.json()["taskId"])

    assert turn["status"] == "draft_ready"
    review = turn["draft"]["review"]
    assert review["spatialChecks"][0]["check"]["classification"] == "occupied"
    assert any("障碍证据" in warning for warning in review["warnings"])
    assert turn["draft"]["plan"]["steps"][0]["payload"]["goalPose"]["x"] == 2
    assert scenes.checked_points[0][1].x == 2


def test_navigation_confirmation_rejects_pose_drift_before_fake_dispatch():
    planning = SpatialPlanningClient(relative=True)
    scenes = StaticSceneContextClient()
    adapter = FakeAdapter()
    app = create_app(
        auth_token=TOKEN,
        adapters={"fake": adapter},
        planning_client=planning,
        scene_context_client=scenes,
    )
    client = TestClient(app)
    conversation = create_conversation(client)
    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "向前 5 米", "modality": "text"},
    )
    turn = wait_turn(client, accepted.json()["taskId"])
    current = RobotState.model_validate(app.state.store.get_robot("robot-1"))
    app.state.store.upsert_robot(current.model_copy(update={
        "pose": current.pose.model_copy(update={"x": 2, "observedAt": utc_now()}),
    }))

    confirmed = client.post(
        f"/api/v1/assistant/drafts/{turn['draft']['draftId']}/confirm",
        headers=auth(),
        json={"confirmed": True},
    )
    assert confirmed.status_code == 409
    assert "drift" in confirmed.text.lower()
    assert adapter.dispatches == []


def test_confirmation_fails_before_dispatch_when_robot_capability_was_withdrawn():
    planning = FakePlanningClient()
    adapter = FakeAdapter()
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter}, planning_client=planning)
    client = TestClient(app)
    conversation = create_conversation(client)
    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "向前移动后打开夹爪", "modality": "text"},
    )
    turn = wait_turn(client, accepted.json()["taskId"])
    app.state.store.upsert_capabilities(
        {
            "agent": "fake-agent",
            "protocol": "fake",
            "robotId": "robot-1",
            "capabilities": ["stop"],
            "safetyMode": "normal",
            "schema": {"type": "object"},
        }
    )

    confirmed = client.post(
        f"/api/v1/assistant/drafts/{turn['draft']['draftId']}/confirm",
        headers=auth(),
        json={"confirmed": True},
    )

    assert confirmed.status_code == 409
    assert "capability" in confirmed.text.lower()
    assert adapter.dispatches == []


def test_needs_input_preserves_canonical_history_for_the_next_turn():
    planning = FakePlanningClient(
        [
            {"status": "needs_input", "assistantMessage": "请说明移动距离。"},
            {"status": "draft_ready", "assistantMessage": "已补全", "proposal": proposal("移动一米")},
        ]
    )
    client = TestClient(create_app(auth_token=TOKEN, planning_client=planning))
    conversation = create_conversation(client)
    first = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "向前移动", "modality": "text"},
    )
    assert wait_turn(client, first.json()["taskId"])["status"] == "needs_input"

    second = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "移动一米", "modality": "text"},
    )
    assert wait_turn(client, second.json()["taskId"])["status"] == "draft_ready"
    history = planning.envelopes[1]["history"]
    assert [(item["role"], item["content"]) for item in history] == [
        ("user", "向前移动"),
        ("assistant", "请说明移动距离。"),
        ("user", "移动一米"),
    ]


def test_only_one_active_turn_and_cancel_is_forwarded():
    release = threading.Event()
    planning = FakePlanningClient(blocker=release)
    client = TestClient(create_app(auth_token=TOKEN, planning_client=planning))
    conversation = create_conversation(client)
    first = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "持续任务", "modality": "text"},
    )
    assert first.status_code == 202
    second = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "并发任务", "modality": "text"},
    )
    assert second.status_code == 409

    cancelled = client.post(f"/api/v1/assistant/turns/{first.json()['taskId']}/cancel", headers=auth())
    assert cancelled.status_code == 200
    turn = wait_turn(client, first.json()["taskId"])
    assert turn["status"] == "cancelled"
    assert planning.cancelled == [first.json()["taskId"]]
    release.set()


def test_feedback_creates_new_turn_and_supersedes_but_never_mutates_original_draft():
    planning = FakePlanningClient()
    client = TestClient(create_app(auth_token=TOKEN, planning_client=planning))
    conversation = create_conversation(client)
    first = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "生成计划", "modality": "text"},
    )
    original = wait_turn(client, first.json()["taskId"])["draft"]
    original_steps = original["plan"]["steps"]

    revised = client.post(
        f"/api/v1/assistant/drafts/{original['draftId']}/revise",
        headers=auth(),
        json={"feedback": "先确认夹爪状态"},
    )
    assert revised.status_code == 202, revised.text
    replacement = wait_turn(client, revised.json()["taskId"])["draft"]
    assert replacement["draftId"] != original["draftId"]
    stored_original = client.get(
        f"/api/v1/assistant/drafts/{original['draftId']}", headers=auth()
    ).json()
    assert stored_original["status"] == "superseded"
    assert stored_original["plan"]["steps"] == original_steps
    assert planning.envelopes[-1]["taskRequest"]["revisionOfDraftId"] == original["draftId"]


def test_unsuccessful_revision_restores_original_draft_so_operator_can_retry():
    planning = FakePlanningClient(
        [
            {"status": "draft_ready", "assistantMessage": "初稿", "proposal": proposal()},
            {
                "status": "cannot_plan",
                "assistantMessage": "当前反馈无法形成计划",
                "reason": "missing detail",
                "missingCapabilities": [],
            },
            {"status": "draft_ready", "assistantMessage": "新稿", "proposal": proposal("修订稿")},
        ]
    )
    client = TestClient(create_app(auth_token=TOKEN, planning_client=planning))
    conversation = create_conversation(client)
    first = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "生成计划", "modality": "text"},
    )
    original = wait_turn(client, first.json()["taskId"])["draft"]

    unsuccessful = client.post(
        f"/api/v1/assistant/drafts/{original['draftId']}/revise",
        headers=auth(),
        json={"feedback": "采用当前不具备的能力"},
    )
    assert unsuccessful.status_code == 202
    assert wait_turn(client, unsuccessful.json()["taskId"])["status"] == "cannot_plan"
    restored = client.get(
        f"/api/v1/assistant/drafts/{original['draftId']}", headers=auth()
    ).json()
    assert restored["status"] == "draft"

    retried = client.post(
        f"/api/v1/assistant/drafts/{original['draftId']}/revise",
        headers=auth(),
        json={"feedback": "改为使用现有能力"},
    )
    assert retried.status_code == 202, retried.text
    replacement = wait_turn(client, retried.json()["taskId"])
    assert replacement["status"] == "draft_ready"


def test_reject_marks_draft_terminal_without_robot_dispatch():
    adapter = FakeAdapter()
    client = TestClient(
        create_app(auth_token=TOKEN, adapters={"fake": adapter}, planning_client=FakePlanningClient())
    )
    conversation = create_conversation(client)
    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={"text": "生成计划", "modality": "text"},
    )
    draft = wait_turn(client, accepted.json()["taskId"])["draft"]
    rejected = client.post(
        f"/api/v1/assistant/drafts/{draft['draftId']}/reject",
        headers=auth(),
        json={"reason": "现场条件不允许"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert adapter.dispatches == []


def wav_bytes(seconds=0.05, sample_rate=16000):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * int(seconds * sample_rate))
    return buffer.getvalue()


def test_voice_upload_returns_editable_transcript_and_does_not_start_planning():
    planning = FakePlanningClient()
    speech = FakeSpeechClient()
    client = TestClient(
        create_app(auth_token=TOKEN, planning_client=planning, speech_client=speech)
    )
    response = client.post(
        "/api/v1/speech/transcriptions",
        headers=auth(),
        files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        data={"locale": "zh-CN"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "向前移动后打开夹爪"
    assert response.json()["transcriptionId"].startswith("transcription-")
    assert planning.envelopes == []
    assert speech.calls[0][2] == "zh-CN"
    assert speech.ran_on_event_loop is False


def test_voice_task_requires_a_saved_transcription_and_uses_the_reviewed_text():
    planning = FakePlanningClient()
    speech = FakeSpeechClient()
    client = TestClient(
        create_app(auth_token=TOKEN, planning_client=planning, speech_client=speech)
    )
    conversation = create_conversation(client)
    missing = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={
            "text": "编辑后的任务",
            "modality": "voice",
            "transcriptionId": "transcription-does-not-exist",
        },
    )
    assert missing.status_code == 422

    transcript = client.post(
        "/api/v1/speech/transcriptions",
        headers=auth(),
        files={"file": ("command.wav", wav_bytes(), "audio/wav")},
        data={"locale": "zh-CN"},
    ).json()
    accepted = client.post(
        f"/api/v1/assistant/conversations/{conversation['conversationId']}/messages",
        headers=auth(),
        json={
            "text": "编辑后的任务",
            "modality": "voice",
            "transcriptionId": transcript["transcriptionId"],
        },
    )
    assert accepted.status_code == 202, accepted.text
    assert wait_turn(client, accepted.json()["taskId"])["status"] == "draft_ready"
    request = planning.envelopes[-1]["taskRequest"]
    assert request["modality"] == "voice"
    assert request["text"] == "编辑后的任务"
    assert request["transcriptionId"] == transcript["transcriptionId"]


def test_voice_upload_rejects_non_wav_and_wrong_audio_shape():
    client = TestClient(create_app(auth_token=TOKEN, speech_client=FakeSpeechClient()))
    not_wav = client.post(
        "/api/v1/speech/transcriptions",
        headers=auth(),
        files={"file": ("command.mp3", b"not wav", "audio/mpeg")},
    )
    assert not_wav.status_code == 415

    stereo = BytesIO()
    with wave.open(stereo, "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00\x00\x00" * 100)
    wrong_shape = client.post(
        "/api/v1/speech/transcriptions",
        headers=auth(),
        files={"file": ("command.wav", stereo.getvalue(), "audio/wav")},
    )
    assert wrong_shape.status_code == 422
