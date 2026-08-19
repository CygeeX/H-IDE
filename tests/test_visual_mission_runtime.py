from __future__ import annotations

from datetime import timedelta

from packages.control_core.adapters import AdapterResult, FakeAdapter
from packages.control_core.models import utc_now
from packages.control_core.runtime import ActionRuntime
from packages.visual_mission_core import parse_plan_draft
from packages.visual_mission_core.runtime import VisualMissionRuntime
from services.control_api.store import SQLiteStore

from .test_visual_mission_contracts import visual_plan


class StageClient:
    def __init__(self, resolution_id="vr-1"):
        self.resolution_id = resolution_id
        self.calls = []

    def resolve_stage(self, payload, is_cancelled):
        self.calls.append(payload)
        assert not is_cancelled()
        return {"visualResolutionId": self.resolution_id, "explanation": "固定目标已解析"}


class AmbiguousThenSelectedStageClient:
    def __init__(self):
        self.calls = []

    def resolve_stage(self, payload, is_cancelled):
        self.calls.append(payload)
        assert not is_cancelled()
        if payload["attempt"] == 1:
            return {
                "status": "needs_input",
                "visualResolutionId": "vr-ambiguous-1",
                "explanation": "检测到两个候选",
                "ambiguityCandidates": [
                    {"candidateId": "candidate-a", "label": "桌子", "box": [1, 2, 30, 40], "score": 0.9},
                    {"candidateId": "candidate-b", "label": "桌子", "box": [31, 2, 60, 40], "score": 0.8},
                ],
            }
        assert payload["candidateId"] == "candidate-a"
        assert payload["selectionSourceVisualResolutionId"] == "vr-ambiguous-1"
        return {
            "status": "succeeded",
            "visualResolutionId": "vr-selected-2",
            "explanation": "已解析用户选择的候选",
            "ambiguityCandidates": [],
        }


class VisionClient:
    def __init__(self, resolution):
        self.resolution = resolution
        self.reads = []

    def get_resolution(self, resolution_id):
        self.reads.append(resolution_id)
        return self.resolution

    def cancel_resolution(self, resolution_id):
        return {"visualResolutionId": resolution_id, "status": "cancelled"}

    def get_evidence(self, resolution_id, sha256):
        return b"fixture-evidence", "image/jpeg"


class AmbiguousThenSelectedVisionClient(VisionClient):
    def get_resolution(self, resolution_id):
        self.reads.append(resolution_id)
        if resolution_id == "vr-ambiguous-1":
            return resolution(
                visualResolutionId=resolution_id,
                status="needs_input",
                executable=False,
                landmarkPose=None,
                goalPose=None,
                safetyGates=[],
                rejectionReason="ambiguous_target",
                ambiguityCandidates=[
                    {"candidateId": "candidate-a", "label": "桌子", "box": [1, 2, 30, 40], "score": 0.9},
                    {"candidateId": "candidate-b", "label": "桌子", "box": [31, 2, 60, 40], "score": 0.8},
                ],
                evidence=[{"kind": "rgb_keyframe", "sha256": "b" * 64, "mediaType": "image/jpeg"}],
            )
        return resolution(
            visualResolutionId=resolution_id,
            attempt=2,
            selectedCandidateId="candidate-a",
            selectionSourceVisualResolutionId="vr-ambiguous-1",
        )


def resolution(**overrides):
    payload = {
        "visualResolutionId": "vr-1",
        "missionId": "visual-plan-1",
        "stageId": "approach-1",
        "attempt": 1,
        "status": "succeeded",
        "executable": True,
        "goalPose": {"frameId": "map", "x": 1.25, "y": 0, "z": 0, "yaw": 0},
        "targetPhrase": "yellow fixture target",
        "landmarkPose": {"frameId": "map", "x": 2.0, "y": 0, "z": 0.5, "yaw": 0},
        "evidence": [{"kind": "rgb_keyframe", "sha256": "b" * 64, "mediaType": "image/jpeg"}],
        "source": {
            "robotId": "robot-1",
            "robotBootId": "fixture-boot-1",
            "localizationEpochId": "lab-map-epoch-1",
            "sensorId": "head_rgbd",
            "sourceMode": "synthetic",
            "observationSessionId": "fixture-session-1",
        },
        "versions": {
            "vlmProvider": "FakeVlmProvider",
            "vlmModel": "fake-fixture-v1",
            "promptVersion": "visual-grounding-v1",
            "visualAlgorithmVersion": "rgbd-median-v1",
            "approachPolicyVersion": "v1",
            "approachPolicySha256": "a" * 64,
        },
        "safetyGates": [{"name": "stable_frames", "passed": True}],
        "createdAt": utc_now().isoformat(),
    }
    payload.update(overrides)
    return payload


def runtime(*, vision=None, adapter=None, mode="lab_execute", store=None):
    adapter = adapter or FakeAdapter()
    action = ActionRuntime(adapters={"fake": adapter})
    store = store or SQLiteStore()
    visual = VisualMissionRuntime(
        store=store,
        action_runtime=action,
        vision_client=VisionClient(vision or resolution()),
        stage_client=StageClient(),
        mode=mode,
    )
    plan = visual_plan()
    store.save_assistant_draft(plan["planId"], plan)
    return visual, adapter, store


def test_reviewed_fixture_resolution_dispatches_once_through_action_runtime() -> None:
    visual, adapter, store = runtime()
    plan = parse_plan_draft(visual_plan())
    visual.start(plan, draft_id=plan.planId)
    completed = visual.run(plan.planId)
    assert completed["status"] == "completed"
    assert completed["motion"] == "simulated"
    assert completed["releaseState"] == "code_complete_robot_agent_pending"
    assert len(adapter.dispatches) == 1


def test_running_robot_command_is_polled_to_terminal_before_mission_completion() -> None:
    class DelayedAdapter(FakeAdapter):
        def __init__(self):
            super().__init__(result=AdapterResult(status="running"))
            self.queries = 0

        def query(self, command_id):
            self.queries += 1
            return AdapterResult(
                status="running" if self.queries == 1 else "succeeded",
                result=None if self.queries == 1 else {"goalReached": True},
            )

    adapter = DelayedAdapter()
    action = ActionRuntime(adapters={"fake": adapter})
    store = SQLiteStore()
    visual = VisualMissionRuntime(
        store=store,
        action_runtime=action,
        vision_client=VisionClient(resolution()),
        stage_client=StageClient(),
        mode="lab_execute",
        command_poll_interval_seconds=0.01,
    )
    plan = parse_plan_draft(visual_plan())
    store.save_assistant_draft(plan.planId, plan.model_dump(mode="json"))
    visual.start(plan, draft_id=plan.planId)
    completed = visual.run(plan.planId)
    assert completed["status"] == "completed"
    assert completed["state"]["receipts"][-1]["status"] == "succeeded"
    assert adapter.queries == 2
    assert len(adapter.dispatches) == 1
    request = adapter.dispatches[0][0]
    assert request.commandKey == "navigation.navigate_to_pose"
    assert request.payload["visualResolutionId"] == "vr-1"
    assert request.payload["goalPose"]["frameId"] == "map"
    assert visual.run(plan.planId)["status"] == "completed"
    assert len(adapter.dispatches) == 1


def test_ambiguity_is_durable_and_selected_candidate_creates_a_new_attempt() -> None:
    adapter = FakeAdapter()
    action = ActionRuntime(adapters={"fake": adapter})
    store = SQLiteStore()
    stage = AmbiguousThenSelectedStageClient()
    visual = VisualMissionRuntime(
        store=store,
        action_runtime=action,
        vision_client=AmbiguousThenSelectedVisionClient(resolution()),
        stage_client=stage,
        mode="lab_execute",
    )
    plan = parse_plan_draft(visual_plan())
    store.save_assistant_draft(plan.planId, plan.model_dump(mode="json"))
    visual.start(plan, draft_id=plan.planId)

    needs_input = visual.run(plan.planId)

    assert needs_input["status"] == "needs_input"
    assert needs_input["stages"][0]["status"] == "needs_input"
    assert needs_input["stages"][0]["visualResolutionId"] == "vr-ambiguous-1"
    assert len(needs_input["stages"][0]["state"]["ambiguityCandidates"]) == 2
    assert adapter.dispatches == []

    approved = visual.select_candidate(plan.planId, "candidate-a")
    assert approved["status"] == "approved"
    assert approved["state"]["candidateSelection"] == {
        "stageId": "approach-1",
        "candidateId": "candidate-a",
        "selectionSourceVisualResolutionId": "vr-ambiguous-1",
    }

    completed = visual.run(plan.planId)

    assert completed["status"] == "completed"
    assert [item["attempt"] for item in completed["stages"]] == [1, 2]
    assert completed["stages"][1]["visualResolutionId"] == "vr-selected-2"
    assert len(adapter.dispatches) == 1
    assert adapter.dispatches[0][0].payload["visualResolutionId"] == "vr-selected-2"
    assert visual.run(plan.planId)["status"] == "completed"
    assert len(adapter.dispatches) == 1


def test_deterministic_resolution_rejection_is_durable_without_dispatch() -> None:
    adapter = FakeAdapter()
    rejected = resolution(
        status="rejected",
        executable=False,
        goalPose=None,
        rejectionReason="approach_rejected:target is too close",
        safetyGates=[
            {"name": "stable_frames", "passed": True},
            {"name": "approach_pose", "passed": False, "detail": "target is too close"},
        ],
    )
    visual, _, store = runtime(vision=rejected, adapter=adapter)
    plan = parse_plan_draft(visual_plan())
    visual.start(plan, draft_id=plan.planId)

    result = visual.run(plan.planId)

    assert result["status"] == "rejected"
    assert result["error"] == "approach_rejected:target is too close"
    assert result["stages"][0]["status"] == "rejected"
    assert result["stages"][0]["visualResolutionId"] == "vr-1"
    assert result["stages"][0]["state"]["rejectionReason"] == rejected["rejectionReason"]
    assert result["stages"][0]["state"]["evidence"] == rejected["evidence"]
    assert store.get_visual_mission(plan.planId)["status"] == "rejected"
    assert adapter.dispatches == []


def test_selected_candidate_resolution_cannot_change_candidate_binding() -> None:
    adapter = FakeAdapter()
    action = ActionRuntime(adapters={"fake": adapter})
    store = SQLiteStore()

    class TamperedVisionClient(AmbiguousThenSelectedVisionClient):
        def get_resolution(self, resolution_id):
            value = super().get_resolution(resolution_id)
            if resolution_id == "vr-selected-2":
                value = {**value, "selectedCandidateId": "candidate-b"}
            return value

    visual = VisualMissionRuntime(
        store=store,
        action_runtime=action,
        vision_client=TamperedVisionClient(resolution()),
        stage_client=AmbiguousThenSelectedStageClient(),
        mode="lab_execute",
    )
    plan = parse_plan_draft(visual_plan())
    store.save_assistant_draft(plan.planId, plan.model_dump(mode="json"))
    visual.start(plan, draft_id=plan.planId)
    assert visual.run(plan.planId)["status"] == "needs_input"
    visual.select_candidate(plan.planId, "candidate-a")

    try:
        visual.run(plan.planId)
    except ValueError as exc:
        assert "candidate" in str(exc)
    else:
        raise AssertionError("tampered candidate binding should fail closed")
    assert adapter.dispatches == []


def test_ambiguity_from_changed_observation_binding_is_not_offered_to_user() -> None:
    adapter = FakeAdapter()

    class TamperedAmbiguityClient(AmbiguousThenSelectedVisionClient):
        def get_resolution(self, resolution_id):
            value = super().get_resolution(resolution_id)
            if resolution_id == "vr-ambiguous-1":
                value = {
                    **value,
                    "source": {
                        **value["source"],
                        "localizationEpochId": "changed-epoch",
                    },
                }
            return value

    store = SQLiteStore()
    visual = VisualMissionRuntime(
        store=store,
        action_runtime=ActionRuntime(adapters={"fake": adapter}),
        vision_client=TamperedAmbiguityClient(resolution()),
        stage_client=AmbiguousThenSelectedStageClient(),
        mode="lab_execute",
    )
    plan = parse_plan_draft(visual_plan())
    store.save_assistant_draft(plan.planId, plan.model_dump(mode="json"))
    visual.start(plan, draft_id=plan.planId)

    try:
        visual.run(plan.planId)
    except ValueError as exc:
        assert "localizationEpochId mismatch" in str(exc)
    else:
        raise AssertionError("stale ambiguity source should fail closed")
    assert store.get_visual_mission(plan.planId)["status"] == "failed"
    assert adapter.dispatches == []


def test_observe_mode_and_changed_binding_produce_zero_dispatches() -> None:
    visual, adapter, _ = runtime(mode="observe")
    plan = parse_plan_draft(visual_plan())
    visual.start(plan, draft_id=plan.planId)
    try:
        visual.run(plan.planId)
    except ValueError as exc:
        assert "observe mode" in str(exc)
    assert adapter.dispatches == []

    stale = resolution()
    stale["source"] = {**stale["source"], "localizationEpochId": "changed"}
    visual, adapter, _ = runtime(vision=stale)
    plan = parse_plan_draft(visual_plan())
    visual.start(plan, draft_id=plan.planId)
    try:
        visual.run(plan.planId)
    except ValueError as exc:
        assert "localizationEpochId mismatch" in str(exc)
    assert adapter.dispatches == []


def test_restart_marks_inflight_mission_unknown_without_replay() -> None:
    visual, adapter, store = runtime()
    plan = parse_plan_draft(visual_plan())
    visual.start(plan, draft_id=plan.planId)
    store.transition_visual_mission(plan.planId, "navigating", expected_status="approved")
    store.recover_visual_missions()
    assert store.get_visual_mission(plan.planId)["status"] == "unknown"
    assert adapter.dispatches == []
