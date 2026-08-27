from __future__ import annotations

from datetime import timedelta
import hashlib
import math
import threading
import time
from typing import Any, Mapping
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from packages.control_core.adapters import AdapterResult, RobotAgentAdapter
from packages.control_core.capabilities import (
    CapabilityDescriptor,
    CapabilityKind,
    CapabilityManifest,
    RiskLevel,
    RobotProfile,
    SensorKind,
    SensorProfile,
)
from packages.control_core.validation import utc_now
from packages.mission_core import MissionConflict, MissionNode, PlanRevision
from packages.scene_core import (
    Bounds3D,
    NAVIGATION_COMMAND_KEY,
    PoseMap,
    SceneContext,
    SceneRef,
    SemanticRegion,
    TargetMapCheck,
    robot_navigation_command_schema,
)
from services.control_api.main import _minimum_travel_heading_order, create_app
from services.control_api.mission_client import MissionPlanningUnavailable
from services.control_api.mission_service import MissionService


TOKEN = "mission-v2-test-token"
INTERNAL_TOKEN = "mission-v2-internal-test-token"


@pytest.fixture(autouse=True)
def mission_internal_callback_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISSION_V2_INTERNAL_TOKEN", INTERNAL_TOKEN)
    # These legacy-flow tests explicitly exercise the pre-authenticated
    # developer compatibility path. Production defaults keep REAL disabled;
    # dedicated policy/ACL tests assert that fail-closed default.
    monkeypatch.setenv("MISSION_POLICY_ALLOW_LEGACY_REAL", "true")


def test_rotation_intent_preserves_explicit_degrees_instead_of_using_direction_default() -> None:
    assert MissionService._rotation_yaw_from_intent(
        {"motion": "rotate", "direction": "left", "angle_deg": 30}
    ) == pytest.approx(math.pi / 6)
    assert MissionService._rotation_yaw_from_intent(
        {"motion": "rotate", "direction": "right", "angleDeg": 45}
    ) == pytest.approx(-math.pi / 4)
    assert MissionService._rotation_yaw_from_intent(
        {"direction": "left"}
    ) == pytest.approx(math.pi / 2)
    assert MissionService._rotation_yaw_from_intent(
        {"direction": "right", "relativeYawRad": -0.2}
    ) == pytest.approx(-0.2)


def test_rotation_budget_accumulates_all_previously_issued_turns() -> None:
    record = {
        "skillExecutions": [
            {
                "request": {
                    "skillId": "navigation.rotate",
                    "input": {"targetYawRad": 0.3},
                }
            },
            {
                "request": {
                    "skillId": "vision.observe",
                    "input": {},
                }
            },
            {
                "request": {
                    "skillId": "navigation.rotate",
                    "input": {"targetYawRad": -0.2},
                }
            },
        ]
    }

    assert MissionService._estimated_cumulative_scan_yaw(record, 0.1) == pytest.approx(0.6)


def test_relative_translation_normalizes_common_llm_intent_fields() -> None:
    node = MissionNode.model_validate(
        {
            "nodeId": "move-forward",
            "skillId": NAVIGATION_COMMAND_KEY,
            "skillVersion": "2.0.0",
            "purpose": "Move the base forward by the reviewed distance",
            "intent": {"motion": "translate_forward", "distance": "0.3 meters"},
            "dependsOn": ["observe-forward"],
            "preconditionIds": [],
            "postconditionIds": [],
            "riskLevel": "medium",
        }
    )

    assert MissionService._relative_translation_intent(node.intent, node) == (
        "forward",
        0.3,
    )

    semantic_variant = node.model_copy(
        update={
            "intent": {
                "goal": "forward 0.3 meters from current robot location",
                "motion": "translation",
                "frame": "robot_base",
            }
        }
    )
    assert MissionService._relative_translation_intent(
        semantic_variant.intent,
        semantic_variant,
    ) == ("forward", 0.3)

    purpose_distance_variant = node.model_copy(
        update={
            "purpose": "向前走 0.4 米",
            "intent": {"description": "保持机器人头部固定"},
        }
    )
    assert MissionService._relative_translation_intent(
        purpose_distance_variant.intent,
        purpose_distance_variant,
    ) == ("forward", 0.4)


def test_arrival_task_alias_uses_pose_verification_after_successful_navigation() -> None:
    service = object.__new__(MissionService)
    navigation = MissionNode.model_validate(
        {
            "nodeId": "move-forward",
            "skillId": NAVIGATION_COMMAND_KEY,
            "skillVersion": "2.0.0",
            "purpose": "Move forward",
            "intent": {"motion": "translate_forward", "distance_m": 0.3},
            "dependsOn": [],
            "preconditionIds": [],
            "postconditionIds": ["moved"],
            "riskLevel": "medium",
        }
    )
    observe = MissionNode.model_validate(
        {
            "nodeId": "verify-arrival",
            "skillId": "vision.observe",
            "skillVersion": "2.0.0",
            "purpose": "Observe and confirm arrival",
            "intent": {
                "task": "observe",
                "analysis": "arrival_verification",
                "target": "current_position",
            },
            "dependsOn": ["move-forward"],
            "preconditionIds": ["moved"],
            "postconditionIds": ["verified"],
            "riskLevel": "low",
        }
    )
    record = {
        "activeRevision": {
            "schemaVersion": "h-ide.plan-revision/2",
            "revisionId": "revision-1",
            "missionId": "mission-1",
            "sequence": 1,
            "parentRevisionId": None,
            "objective": "move forward and confirm arrival",
            "nodes": [navigation.model_dump(mode="json"), observe.model_dump(mode="json")],
            "reasonCode": "initial_human_approval",
            "reason": "approved",
            "changeSummary": ["initial"],
            "disposition": "human_approved",
            "authorizationId": "authorization-1",
            "createdBy": "operator",
            "createdAt": utc_now().isoformat(),
        },
        "skillExecutions": [
            {
                "request": {
                    "skillId": NAVIGATION_COMMAND_KEY,
                    "input": {
                        "goalPose": {
                            "frameId": "map",
                            "x": 1.0,
                            "y": 2.0,
                            "z": 0.0,
                            "yaw": 0.1,
                            "observedAt": utc_now().isoformat(),
                        }
                    },
                },
                "result": {"status": "succeeded"},
            }
        ],
    }

    payload = service._build_visual_input(record, observe)

    assert payload["mode"] == "arrival_verification"
    assert payload["expectedTargetPose"]["x"] == 1.0
    assert "observedAt" not in payload["expectedTargetPose"]
    assert payload["verificationMaximumDistanceM"] == 0.15
    assert payload["verificationMaximumYawRad"] == 0.35


@pytest.mark.parametrize("task", ["describe_scene", "report_observation"])
def test_scene_observation_alias_does_not_require_a_semantic_target(task: str) -> None:
    service = object.__new__(MissionService)
    observe = MissionNode.model_validate(
        {
            "nodeId": "observe-scene",
            "skillId": "vision.observe",
            "skillVersion": "2.0.0",
            "purpose": "描述当前第一视角画面",
            "intent": {"task": task, "target": "current_view"},
            "dependsOn": [],
            "preconditionIds": [],
            "postconditionIds": ["scene-reported"],
            "riskLevel": "low",
        }
    )
    record = {
        "activeRevision": {
            "schemaVersion": "h-ide.plan-revision/2",
            "revisionId": "revision-scene-1",
            "missionId": "mission-scene-1",
            "sequence": 1,
            "parentRevisionId": None,
            "objective": "只观察并描述当前画面，不移动",
            "nodes": [observe.model_dump(mode="json")],
            "reasonCode": "initial_human_approval",
            "reason": "approved",
            "changeSummary": ["initial"],
            "disposition": "human_approved",
            "authorizationId": "authorization-scene-1",
            "createdBy": "operator",
            "createdAt": utc_now().isoformat(),
        },
        "skillExecutions": [],
    }

    assert service._build_visual_input(record, observe) == {
        "mode": "scene_description",
        "instruction": "描述当前第一视角画面",
    }


def test_grounding_alias_stays_vlm_grounding_before_relative_navigation() -> None:
    service = object.__new__(MissionService)
    observe = MissionNode.model_validate(
        {
            "nodeId": "observe-panel",
            "skillId": "vision.observe",
            "skillVersion": "2.0.0",
            "purpose": "确认中央灰色面板可见",
            "intent": {
                "task": "describe_and_ground",
                "target_description": "中央大型竖直灰色面板",
            },
            "dependsOn": [],
            "riskLevel": "low",
        }
    )
    move = MissionNode.model_validate(
        {
            "nodeId": "move-short",
            "skillId": "navigation.navigate_to_pose",
            "skillVersion": "2.0.0",
            "purpose": "保持当前朝向向前移动0.1米",
            "intent": {
                "motion": "translate_forward",
                "distance_m": 0.1,
                "reference": "base_link",
            },
            "dependsOn": ["observe-panel"],
            "riskLevel": "medium",
        }
    )
    record = {
        "activeRevision": {
            "schemaVersion": "h-ide.plan-revision/2",
            "revisionId": "revision-grounding-1",
            "missionId": "mission-grounding-1",
            "sequence": 1,
            "parentRevisionId": None,
            "objective": "观察中央灰色面板，然后向前移动0.1米",
            "nodes": [
                observe.model_dump(mode="json"),
                move.model_dump(mode="json"),
            ],
            "reasonCode": "initial_human_approval",
            "reason": "approved",
            "changeSummary": ["initial"],
            "disposition": "human_approved",
            "authorizationId": "authorization-grounding-1",
            "createdBy": "operator",
            "createdAt": utc_now().isoformat(),
        },
        "skillExecutions": [],
    }

    assert service._build_visual_input(record, observe) == {
        "mode": "landmark_observation",
        "instruction": "确认中央灰色面板可见",
        "targetDescription": "中央大型竖直灰色面板",
    }


class FakeMissionPlanningClient:
    def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]:
        now = utc_now()
        envelope = dict(brief["envelope"])
        assert NAVIGATION_COMMAND_KEY in envelope["allowedSkillIds"]
        navigation = next(
            item for item in brief["capabilities"] if item["skillId"] == NAVIGATION_COMMAND_KEY
        )
        return {
            "plan": {
                "planType": "mission",
                "schemaVersion": "h-ide.mission/2",
                "planId": brief["missionId"],
                "robotId": brief["robotId"],
                "objective": brief["objective"],
                "nodes": [
                    {
                        "nodeId": "navigate-kitchen",
                        "skillId": NAVIGATION_COMMAND_KEY,
                        "skillVersion": navigation["skillVersion"],
                        "purpose": "到达厨房附近",
                        "intent": {"targetDescription": "厨房"},
                        "dependsOn": [],
                        "preconditionIds": [],
                        "postconditionIds": ["arrival-verified"],
                        "riskLevel": navigation["riskLevel"],
                    }
                ],
                "envelope": envelope,
                "createdAt": now.isoformat(),
                "expiresAt": (now + timedelta(minutes=5)).isoformat(),
            }
        }

    def create_revision(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        raise AssertionError(f"unexpected revision request: {request}")

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del event, context
        return {"kind": "continue", "reason": "fixture continues"}

    def score_candidates(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "missionId": request["missionId"],
            "scores": [
                {
                    "candidateId": item["candidateId"],
                    "score": item.get("modelScore", 0.5),
                    "eligible": True,
                    "reasons": ["fixture score"],
                }
                for item in request["candidates"]
            ],
            "selectedCandidateId": request["candidates"][0]["candidateId"],
            "generatedAt": utc_now().isoformat(),
        }


class SameNodeTargetRevisionPlanningClient(FakeMissionPlanningClient):
    def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]:
        response = super().create_draft(brief)
        response["plan"]["nodes"][0]["intent"] = {
            "targetRecordId": "target-old",
        }
        return response

    def create_revision(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        active = request["activeRevision"]
        nodes = [dict(item) for item in active["nodes"]]
        nodes[0] = {
            **nodes[0],
            "purpose": "到达新审核目标点",
            "intent": {"targetRecordId": "target-new"},
        }
        return {
            "revision": {
                "schemaVersion": "h-ide.plan-revision/2",
                "revisionId": f"revision-{request['requestId']}",
                "missionId": request["missionId"],
                "sequence": active["sequence"] + 1,
                "parentRevisionId": active["revisionId"],
                "objective": request["objective"],
                "nodes": nodes,
                "reasonCode": "operator_intervention",
                "reason": "Operator selected a different reviewed target",
                "changeSummary": ["Changed the target record binding"],
                "disposition": "human_required",
                "authorizationId": None,
                "createdBy": "h-ide-agent",
                "createdAt": utc_now().isoformat(),
            }
        }


class VisualMissionPlanningClient(FakeMissionPlanningClient):
    def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]:
        now = utc_now()
        capabilities = {item["skillId"]: item for item in brief["capabilities"]}
        visual = capabilities["vision.observe"]
        navigation = capabilities[NAVIGATION_COMMAND_KEY]
        return {
            "plan": {
                "planType": "mission",
                "schemaVersion": "h-ide.mission/2",
                "planId": brief["missionId"],
                "robotId": brief["robotId"],
                "objective": brief["objective"],
                "nodes": [
                    {
                        "nodeId": "observe-target",
                        "skillId": "vision.observe",
                        "skillVersion": visual["skillVersion"],
                        "purpose": "定位银白色机器人旁边的安全目标点",
                        "intent": {
                            "visualTaskMode": "object_approach",
                            "targetDescription": "银白色机器人",
                        },
                        "dependsOn": [],
                        "preconditionIds": [],
                        "postconditionIds": ["target-pose-resolved"],
                        "riskLevel": visual["riskLevel"],
                    },
                    {
                        "nodeId": "navigate-target",
                        "skillId": NAVIGATION_COMMAND_KEY,
                        "skillVersion": navigation["skillVersion"],
                        "purpose": "到达视觉确认的目标点",
                        "intent": {"targetDescription": "银白色机器人旁边"},
                        "dependsOn": ["observe-target"],
                        "preconditionIds": ["target-pose-resolved"],
                        "postconditionIds": ["arrival-verified"],
                        "riskLevel": navigation["riskLevel"],
                    },
                ],
                "envelope": dict(brief["envelope"]),
                "createdAt": now.isoformat(),
                "expiresAt": (now + timedelta(minutes=5)).isoformat(),
            }
        }


class CandidateRetryPlanningClient(VisualMissionPlanningClient):
    def __init__(self) -> None:
        self.event_calls = 0

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del event, context
        self.event_calls += 1
        raise AssertionError("visual candidate answers must not call the planner")


class ContinueVisualMissionPlanningClient(VisualMissionPlanningClient):
    def __init__(self) -> None:
        self.event_calls = 0

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del event, context
        self.event_calls += 1
        return {"kind": "continue", "reason": "fixture visual mission continues"}


class RespondingMissionPlanningClient(FakeMissionPlanningClient):
    def __init__(self) -> None:
        self.event_calls = 0

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        self.event_calls += 1
        active = event["activeRevision"]
        nodes = [dict(item) for item in active["nodes"]]
        nodes[0] = {
            **nodes[0],
            "purpose": f"根据操作员补充重新规划：{event['payload']['text']}",
        }
        return {
            "kind": "replan",
            "request": {"requestId": f"replan-{event['correlationId']}"},
            "revision": {
                "schemaVersion": "h-ide.plan-revision/2",
                "revisionId": f"revision-{event['correlationId']}",
                "missionId": event["missionId"],
                "sequence": active["sequence"] + 1,
                "parentRevisionId": active["revisionId"],
                "objective": active["objective"],
                "nodes": nodes,
                "reasonCode": "operator_intervention",
                "reason": "Operator supplied new Mission context",
                "changeSummary": ["Updated the semantic navigation purpose"],
                "disposition": "human_required",
                "authorizationId": None,
                "createdBy": "h-ide-agent",
                "createdAt": utc_now().isoformat(),
            },
        }


class FailOnceMissionPlanningClient(FakeMissionPlanningClient):
    def __init__(self) -> None:
        self.event_calls = 0

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del event, context
        self.event_calls += 1
        if self.event_calls == 1:
            raise MissionPlanningUnavailable("fixture planner timeout")
        return {"kind": "continue", "reason": "fixture retry continues"}


class ContinueMissionPlanningClient(FakeMissionPlanningClient):
    def __init__(self) -> None:
        self.event_calls = 0

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del event, context
        self.event_calls += 1
        return {"kind": "continue", "reason": "fixture continues"}


class BlockingFailOnceRevisionPlanningClient(FakeMissionPlanningClient):
    def __init__(self) -> None:
        self.revision_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def create_revision(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.revision_calls += 1
        if self.revision_calls == 1:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("blocked revision planner was not released")
            raise MissionPlanningUnavailable("fixture revision planner timeout")
        active = request["activeRevision"]
        nodes = [dict(item) for item in active["nodes"]]
        nodes[0] = {
            **nodes[0],
            "purpose": f"修订后：{request['triggerDetail']}",
        }
        return {
            "revision": {
                "schemaVersion": "h-ide.plan-revision/2",
                "revisionId": f"revision-{request['requestId']}",
                "missionId": request["missionId"],
                "sequence": active["sequence"] + 1,
                "parentRevisionId": active["revisionId"],
                "objective": request["objective"],
                "nodes": nodes,
                "reasonCode": "operator_intervention",
                "reason": "operator requested a revision",
                "changeSummary": ["updated requested objective"],
                "disposition": "human_required",
                "authorizationId": None,
                "createdBy": "h-ide-agent",
                "createdAt": utc_now().isoformat(),
            }
        }


class BlockingEventPlanningClient(FakeMissionPlanningClient):
    def __init__(self) -> None:
        self.event_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del event, context
        self.event_calls += 1
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise AssertionError("blocked event planner was not released")
        return {"kind": "continue", "reason": "event accepted"}


class FakeVisualTaskClient:
    evidence_body = b"\xff\xd8h-ide-vision-fixture\xff\xd9"
    evidence_sha256 = hashlib.sha256(evidence_body).hexdigest()

    def __init__(self) -> None:
        self.requests: list[Mapping[str, Any]] = []
        self.evidence_requests: list[tuple[str, str]] = []

    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.requests.append(dict(request))
        now = utc_now().isoformat()
        return {
            "schemaVersion": "h-ide.visual-result/2",
            "analysisId": "analysis-silver-robot-1",
            "requestId": request["requestId"],
            "requestSha256": "a" * 64,
            "missionId": request["missionId"],
            "revisionId": request["revisionId"],
            "nodeId": request["nodeId"],
            "attempt": request["attempt"],
            "mode": request["mode"],
            "status": "succeeded",
            "source": {
                "robotId": request["robotId"],
                "robotBootId": request["robotBootId"],
                "gatewayInstanceId": "gateway-fixture-1",
                "sensorId": request["sensorId"],
                "observationSessionId": "observation-fixture-1",
                "localizationEpochId": request["localizationEpochId"],
                "sourceMode": request["requestedSourceMode"],
                "sourceId": "fixture-source-1",
                "bundleId": "fixture-bundle-1",
                "capturedAt": now,
                "calibrationSha256": "b" * 64,
            },
            "versions": {
                "vlmProvider": "qwen",
                "vlmModel": "qwen3-vl-plus",
                "promptVersion": "grounding-v2",
                "visualAlgorithmVersion": "rgbd-median-v1",
                "approachPolicyVersion": "v1",
                "approachPolicySha256": "c" * 64,
                "segmentationProvider": "sam2",
                "segmentationModel": "sam2.1_hiera_tiny",
            },
            "hypotheses": [
                {
                    "hypothesisId": "hypothesis-silver-robot-1",
                    "source": "qwen",
                    "label": "银白色机器人",
                    "pixelBox": [200, 100, 420, 460],
                    "score": 0.93,
                }
            ],
            "geometryCandidates": [
                {
                    "geometryCandidateId": "geometry-silver-robot-1",
                    "hypothesisId": "hypothesis-silver-robot-1",
                    "source": "deterministic_rgbd",
                    "role": "approach",
                    "landmarkPose": {
                        "frameId": "map",
                        "x": 2.0,
                        "y": 0.5,
                        "z": 0.0,
                        "yaw": 0.0,
                    },
                    "targetPose": {
                        "frameId": "map",
                        "x": 1.25,
                        "y": 0.5,
                        "z": 0.0,
                        "yaw": 0.0,
                    },
                    "verified": True,
                    "safetyGates": [
                        {"name": "stable_depth", "passed": True},
                        {"name": "map_tf", "passed": True},
                        {"name": "clearance", "passed": True},
                    ],
                    "evidenceIds": ["evidence-rgbd-1"],
                }
            ],
            "verification": None,
            "ambiguityCandidates": [],
            "warnings": [],
            "evidence": [
                {
                    "kind": "rgb_keyframe",
                    "sha256": self.evidence_sha256,
                    "mediaType": "image/jpeg",
                }
            ],
            "createdAt": now,
        }

    def get_evidence(self, analysis_id: str, sha256: str) -> tuple[bytes, str]:
        self.evidence_requests.append((analysis_id, sha256))
        return self.evidence_body, "image/jpeg"


class OfficeSceneDescriptionVisualTaskClient(FakeVisualTaskClient):
    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        result = dict(super().analyze(request))
        sequence = len(self.requests)
        source = dict(result["source"])
        source["observationSessionId"] = "observation-live-1"
        result.update(
            analysisId=f"analysis-office-heading-{sequence}",
            source=source,
            hypotheses=[
                {
                    "hypothesisId": f"office-visible-item-{sequence}",
                    "source": "qwen",
                    "label": "可见办公区域",
                    "pixelBox": [100, 80, 500, 420],
                    "score": 0.92,
                }
            ],
            geometryCandidates=[],
            observationSummary=f"方向 {sequence} 可见办公室门牌、家具和通道",
        )
        return result


class AmbiguousThenSelectedVisualTaskClient(FakeVisualTaskClient):
    source_analysis_id = "analysis-ambiguous-robot-1"
    first_candidate_id = "candidate-ambiguous-left"
    second_candidate_id = "candidate-ambiguous-right"

    def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        result = dict(super().analyze(request))
        if len(self.requests) == 1:
            result.update(
                analysisId=self.source_analysis_id,
                status="needs_input",
                hypotheses=[
                    {
                        "hypothesisId": self.first_candidate_id,
                        "source": "qwen",
                        "label": "左侧银白色机器人",
                        "pixelBox": [100, 100, 260, 440],
                        "score": 0.91,
                    },
                    {
                        "hypothesisId": self.second_candidate_id,
                        "source": "qwen",
                        "label": "右侧银白色机器人",
                        "pixelBox": [360, 100, 520, 440],
                        "score": 0.86,
                    },
                ],
                geometryCandidates=[],
                ambiguityCandidates=[
                    {
                        "candidateId": self.first_candidate_id,
                        "label": "左侧银白色机器人",
                        "box": [100, 100, 260, 440],
                        "score": 0.91,
                    },
                    {
                        "candidateId": self.second_candidate_id,
                        "label": "右侧银白色机器人",
                        "box": [360, 100, 520, 440],
                        "score": 0.86,
                    },
                ],
                selectedCandidateId=None,
                selectionSourceAnalysisId=None,
                rejectionReason="ambiguous_target",
            )
            return result
        assert request["attempt"] == 2
        assert request["candidateId"] == self.first_candidate_id
        assert request["selectionSourceAnalysisId"] == self.source_analysis_id
        result.update(
            analysisId="analysis-selected-robot-2",
            selectedCandidateId=self.first_candidate_id,
            selectionSourceAnalysisId=self.source_analysis_id,
            rejectionReason=None,
        )
        return result


class FakeSceneContextClient:
    scene_ref = SceneRef(sceneId="scene-lab", revision=3)

    def get_context(self, robot_id, *, current_pose, expected_ref=None, selected_point=None):
        assert robot_id == "robot-1"
        assert expected_ref == self.scene_ref
        assert selected_point is None
        return SceneContext(
            sceneRef=self.scene_ref,
            bounds=Bounds3D(
                min={"x": -2, "y": -2, "z": -0.2},
                max={"x": 6, "y": 6, "z": 3},
            ),
            currentPose=current_pose,
            annotations=[
                SemanticRegion(
                    locationId="kitchen",
                    name="厨房",
                    aliases=["茶水间"],
                    polygonXY=[
                        {"x": 1, "y": 0},
                        {"x": 3, "y": 0},
                        {"x": 3, "y": 2},
                        {"x": 1, "y": 2},
                    ],
                    zMin=0,
                    zMax=2.5,
                    entryPose=PoseMap(
                        frameId="map",
                        x=1.5,
                        y=0.5,
                        z=0,
                        yaw=0,
                    ),
                )
            ],
        )

    def check_target(self, scene_ref, point_map):
        assert scene_ref == self.scene_ref
        assert point_map.x == 1.5
        return TargetMapCheck(
            mapVersion="scene-map-3",
            classification="observed_free",
            clearanceMeters=0.7,
            distanceToObservedObstacleMeters=0.9,
            warnings=["PCD is a static prior; Nav2 must check current obstacles"],
        )


class MissionRobotDriver:
    def __init__(self) -> None:
        self.profile = RobotProfile.build(
            robotId="robot-1",
            baseType="differential_drive",
            headMotion="fixed",
            sensors=[
                SensorProfile(
                    sensorId="head_rgbd",
                    kind=SensorKind.RGBD,
                    frameId="camera_color_optical_frame",
                    fixedMount=True,
                    depthCapable=True,
                )
            ],
        )
        self.manifest = CapabilityManifest.build(
            robotId="robot-1",
            agentVersion="mission-test-1",
            robotProfileDigest=self.profile.profileDigest,
            capabilities=[
                CapabilityDescriptor(
                    skillId=NAVIGATION_COMMAND_KEY,
                    skillVersion="2.0.0",
                    description="Navigate one H-IDE-resolved pose through Nav2",
                    kind=CapabilityKind.MOTION,
                    riskLevel=RiskLevel.MEDIUM,
                    inputSchema=robot_navigation_command_schema(),
                    resources=["motion", "navigation", "observation"],
                    cancellable=True,
                    interruptible=True,
                    timeoutMs=30_000,
                    requiredSensorIds=["head_rgbd"],
                    executor="nav2",
                    executorProfile="nav2.navigate_to_pose.default_bt",
                )
            ],
        )
        self.dispatches: list[Any] = []
        self.cancellations: list[str] = []

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        return AdapterResult(
            status="available",
            result={
                "robotId": robot_id or "robot-1",
                "robotBootId": "boot-live-1",
                "capabilitiesDigest": self.manifest.manifestDigest,
                "capabilities": [NAVIGATION_COMMAND_KEY],
                "protocol": "robot-agent.v1",
                "executionBackend": "robot-agent",
                "robotProfile": self.profile.model_dump(mode="json"),
                "capabilityManifest": self.manifest.model_dump(mode="json"),
                "observation": {
                    "status": "USABLE",
                    "sessionId": "observation-live-1",
                    "localizationEpochId": "epoch-live-1",
                },
                "currentPose": {
                    "frameId": "map",
                    "x": 0.25,
                    "y": -0.1,
                    "z": 0.0,
                    "yaw": 0.0,
                    "observedAt": utc_now().isoformat(),
                },
            },
        )

    def probe(self, mapping) -> AdapterResult:
        del mapping
        return AdapterResult(status="probed")

    def dispatch(self, request, mapping) -> AdapterResult:
        self.dispatches.append((request, mapping))
        return AdapterResult(status="succeeded", result={"goalReached": True})

    def cancel(self, command_id: str) -> AdapterResult:
        self.cancellations.append(command_id)
        return AdapterResult(status="cancelled", result={"commandId": command_id})

    def query(self, command_id: str) -> AdapterResult:
        return AdapterResult(status="succeeded", result={"commandId": command_id})


class RunningMissionRobotDriver(MissionRobotDriver):
    def dispatch(self, request, mapping) -> AdapterResult:
        self.dispatches.append((request, mapping))
        return AdapterResult(status="running", result={"accepted": True})

    def query(self, command_id: str) -> AdapterResult:
        return AdapterResult(
            status="succeeded",
            result={"commandId": command_id, "goalReached": True},
        )


class OfficeInspectionMissionRobotDriver(MissionRobotDriver):
    def __init__(self) -> None:
        super().__init__()
        rotate = CapabilityDescriptor(
            skillId="navigation.rotate",
            skillVersion="2.0.0",
            description="Rotate through the strict Nav2 Spin adapter",
            kind=CapabilityKind.MOTION,
            riskLevel=RiskLevel.MEDIUM,
            inputSchema={
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "targetYawRad",
                    "timeAllowanceMs",
                    "missionId",
                    "revisionId",
                    "nodeId",
                    "authorizationId",
                    "skillRequestId",
                    "idempotencyKey",
                    "robotBootId",
                    "localizationEpochId",
                ],
                "properties": {
                    "targetYawRad": {"type": "number"},
                    "timeAllowanceMs": {"type": "integer"},
                    "missionId": {"type": "string"},
                    "revisionId": {"type": "string"},
                    "nodeId": {"type": "string"},
                    "authorizationId": {"type": "string"},
                    "skillRequestId": {"type": "string"},
                    "idempotencyKey": {"type": "string"},
                    "robotBootId": {"type": "string"},
                    "localizationEpochId": {"type": "string"},
                },
            },
            resources=["motion", "navigation"],
            cancellable=True,
            interruptible=True,
            timeoutMs=300_000,
            requiredSensorIds=["head_rgbd"],
            executor="nav2",
            executorProfile="nav2.strict_static_primitives",
        )
        self.manifest = CapabilityManifest.build(
            robotId="robot-1",
            agentVersion="mission-test-office-1",
            robotProfileDigest=self.profile.profileDigest,
            capabilities=[*self.manifest.capabilities, rotate],
        )

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        discovered = super().discover(robot_id)
        result = dict(discovered.result)
        result["capabilities"] = [
            NAVIGATION_COMMAND_KEY,
            "navigation.rotate",
        ]
        result["capabilitiesDigest"] = self.manifest.manifestDigest
        result["capabilityManifest"] = self.manifest.model_dump(mode="json")
        return AdapterResult(status=discovered.status, result=result)


class RunningOfficeInspectionMissionRobotDriver(OfficeInspectionMissionRobotDriver):
    def dispatch(self, request, mapping) -> AdapterResult:
        self.dispatches.append((request, mapping))
        if request.commandKey == "navigation.rotate":
            return AdapterResult(status="running", result={"accepted": True})
        return AdapterResult(status="succeeded", result={"goalReached": True})

    def query(self, command_id: str) -> AdapterResult:
        return AdapterResult(status="running", result={"commandId": command_id})


class NoPoseMissionRobotDriver(MissionRobotDriver):
    def discover(self, robot_id: str | None = None) -> AdapterResult:
        discovered = super().discover(robot_id)
        result = dict(discovered.result)
        result.pop("currentPose", None)
        return AdapterResult(status=discovered.status, result=result)


class PoseDropMissionRobotDriver(MissionRobotDriver):
    def __init__(self) -> None:
        super().__init__()
        self.drop_pose = False

    def discover(self, robot_id: str | None = None) -> AdapterResult:
        discovered = super().discover(robot_id)
        result = dict(discovered.result)
        if self.drop_pose:
            result.pop("currentPose", None)
        return AdapterResult(status=discovered.status, result=result)


def headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "X-H-IDE-Mission-Internal-Token": INTERNAL_TOKEN,
    }


def test_current_robot_pose_refreshes_after_human_review_delay(monkeypatch) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    driver = MissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    stale_robot = app.state.store.get_robot("robot-1")
    assert stale_robot is not None
    stale_robot["pose"] = {
        **stale_robot["pose"],
        "observedAt": (utc_now() - timedelta(seconds=10)).isoformat(),
    }
    app.state.store.upsert_robot(stale_robot)

    current = app.state.mission._current_robot_pose("robot-1")

    assert current.observedAt is not None
    assert (utc_now() - current.observedAt).total_seconds() < 2
    assert current.x == 0.25


def test_mission_v2_http_flow_is_approved_and_event_driven() -> None:
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        unauthorized = client.get("/api/v2/missions")
        assert unauthorized.status_code == 401

        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-kitchen-1",
                "missionId": "mission-kitchen-1",
                "robotId": "robot-1",
                "objective": "去厨房",
                "requestedBy": "operator-test",
            },
        )
        assert created.status_code == 201, created.text
        draft = created.json()
        assert draft["status"] == "draft"
        assert "goalPose" not in str(draft["plan"]["nodes"])

        draft_id = draft["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{draft_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["snapshot"]["status"] == "approved"

        started = client.post(
            f"/api/v2/missions/{draft_id}/start",
            headers=headers(),
        )
        assert started.status_code == 200, started.text
        assert started.json()["snapshot"]["status"] == "ready"

        # A semantic label is never sufficient authority for base motion.  The
        # happy-path fixture must bind the reviewed label to a trusted map
        # pose, exactly as scene teaching / RGB-D geometry would do in the
        # product, before the runtime may issue NavigateToPose.
        trusted_target = client.post(
            f"/api/v2/missions/{draft_id}/world-records",
            headers=headers(),
            json={
                "recordId": "record-kitchen-target-1",
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "厨房",
                "predicate": "target_pose",
                "value": {
                    "goalPose": {
                        "frameId": "map",
                        "x": 1.5,
                        "y": 0.5,
                        "z": 0.0,
                        "yaw": 0.0,
                    }
                },
                "source": "operator",
                "sourceRef": "operator-reviewed-kitchen-1",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
                "frameId": "map",
            },
        )
        assert trusted_target.status_code == 200, trusted_target.text

        issued = client.post(
            f"/api/v2/missions/{draft_id}/nodes/navigate-kitchen/skill-requests",
            headers=headers(),
        )
        assert issued.status_code == 202, issued.text
        skill_request = issued.json()["request"]
        assert skill_request["skillId"] == NAVIGATION_COMMAND_KEY
        assert issued.json()["dispatch"] == "pending_robot_agent_bridge"

        completed = client.post(
            f"/api/v2/missions/{draft_id}/skill-results",
            headers=headers(),
            json={
                "requestId": skill_request["requestId"],
                "status": "succeeded",
                "completedAt": utc_now().isoformat(),
                "result": {"goalReached": True},
                "evidenceIds": ["robot-agent-result-1"],
            },
        )
        assert completed.status_code == 200, completed.text
        mission = completed.json()
        assert mission["snapshot"]["status"] == "completed"
        assert [event["sequence"] for event in mission["events"]] == [1, 2, 3, 4, 5, 6]


def test_mission_v2_rejects_duplicate_approval_and_qwen_fact() -> None:
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-kitchen-2",
                "objective": "找到厨房",
            },
        ).json()
        draft_id = created["draftId"]
        first = client.post(
            f"/api/v2/missions/drafts/{draft_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert first.status_code == 200
        duplicate = client.post(
            f"/api/v2/missions/drafts/{draft_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert duplicate.status_code == 409

        qwen_fact = client.post(
            f"/api/v2/missions/{draft_id}/world-records",
            headers=headers(),
            json={
                "recordId": "record-qwen-fact-1",
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "kitchen",
                "predicate": "visible",
                "value": True,
                "source": "qwen",
                "sourceRef": "qwen-response-1",
                "confidence": 0.95,
                "observedAt": utc_now().isoformat(),
            },
        )
        assert qwen_fact.status_code == 422


def test_mission_v2_publishes_pcd_semantic_region_as_static_scene_target_fact() -> None:
    scene = FakeSceneContextClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=FakeMissionPlanningClient(),
        scene_context_client=scene,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-scene-kitchen-1",
                "objective": "去厨房",
                "sceneRefs": [scene.scene_ref.model_dump(mode="json")],
            },
        )
        assert created.status_code == 201, created.text
        mission_id = created.json()["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        scene_records = [
            item for item in approved.json()["worldRecords"] if item["source"] == "scene"
        ]
        assert {item["predicate"] for item in scene_records} == {
            "semantic_region",
            "target_pose",
        }

        advanced = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 4},
        )
        assert advanced.status_code == 200, (
            advanced.text,
            app.state.runtime.diagnostics[-5:],
            app.state.mission.runtime.get(mission_id),
        )
        result = advanced.json()
        assert result["stoppedBecause"] == "dispatching"
        request = result["mission"]["skillExecutions"][0]["request"]
        assert request["input"]["payloadType"] == "spatial_resolution"
        assert request["input"]["goalPose"]["x"] == 1.5
        assert request["input"]["sceneRef"] == {"sceneId": "scene-lab", "revision": 3}
        assert "path" not in request["input"]


def test_simulation_draft_inherits_the_current_published_scene_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AutoBoundScene(FakeSceneContextClient):
        def get_context(
            self,
            robot_id,
            *,
            current_pose,
            expected_ref=None,
            selected_point=None,
        ):
            assert expected_ref is None or expected_ref == self.scene_ref
            return super().get_context(
                robot_id,
                current_pose=current_pose,
                expected_ref=self.scene_ref,
                selected_point=selected_point,
            )

    class CapturingPlanner(FakeMissionPlanningClient):
        brief: Mapping[str, Any] | None = None

        def create_draft(self, brief: Mapping[str, Any]) -> Mapping[str, Any]:
            self.brief = brief
            return super().create_draft(brief)

    monkeypatch.setenv("MISSION_V2_AUTO_USE_BOUND_SCENE", "true")
    scene = AutoBoundScene()
    planner = CapturingPlanner()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        scene_context_client=scene,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-auto-bound-scene-1",
                "objective": "巡检所有区域",
            },
        )
    assert created.status_code == 201, created.text
    assert planner.brief is not None
    assert planner.brief["envelope"]["sceneRefs"] == [
        scene.scene_ref.model_dump(mode="json")
    ]
    assert planner.brief["envelope"]["allowedRegionIds"] == ["kitchen"]
    assert {item["predicate"] for item in planner.brief["worldRecords"]} == {
        "semantic_region",
        "target_pose",
    }


def test_mission_v2_operator_message_creates_a_reviewed_plan_revision() -> None:
    planner = RespondingMissionPlanningClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-follow-up-1",
                "objective": "去厨房",
            },
        ).json()
        mission_id = draft["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text

        intervention_id = "intervention-reviewed-replan-1"
        message = {
            "interventionId": intervention_id,
            "text": "改为去厨房门口，不要进入厨房",
            "kind": "modify_goal",
            "requestedBy": "operator-test",
        }
        response = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=message,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["decision"]["kind"] == "replan"
        assert payload["mission"]["snapshot"]["status"] == "awaiting_review"
        pending = payload["decision"]["revision"]
        assert pending["disposition"] == "human_required"
        assert planner.event_calls == 1
        generic_events = [
            item
            for item in payload["mission"]["events"]
            if item["correlationId"] == intervention_id
        ]
        assert [item["payload"]["phase"] for item in generic_events] == [
            "claimed",
            "completed",
        ]
        assert generic_events[1]["payload"]["decision"] == payload["decision"]
        assert generic_events[1]["causationId"] == generic_events[0]["eventId"]
        assert len(payload["mission"]["revisions"]) == 2

        replay = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=message,
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["decision"] == payload["decision"]
        assert replay.json()["mission"]["lastSequence"] == payload["mission"][
            "lastSequence"
        ]
        assert len(replay.json()["mission"]["revisions"]) == 2
        assert planner.event_calls == 1

        rebound = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={**message, "text": "同一个 ID 但不同内容"},
        )
        assert rebound.status_code == 409, rebound.text

        approved = client.post(
            f"/api/v2/missions/{mission_id}/revisions/{pending['revisionId']}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["snapshot"]["status"] == "ready"


def test_generic_message_planner_failure_stays_fenced_and_restarts_safely() -> None:
    planner = FailOnceMissionPlanningClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-generic-retry-1",
                "objective": "去厨房",
            },
        )
        assert draft.status_code == 201, draft.text
        mission_id = draft.json()["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        started = client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        )
        assert started.status_code == 200, started.text
        before = started.json()
        assert before["snapshot"]["status"] == "ready"

        # Bind the semantic target so a fence bypass would be capable of
        # issuing the old reviewed navigation node.
        target = client.post(
            f"/api/v2/missions/{mission_id}/world-records",
            headers=headers(),
            json={
                "recordId": "record-generic-retry-target-1",
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "厨房",
                "predicate": "target_pose",
                "value": {
                    "targetPose": {
                        "frameId": "map",
                        "x": 0.5,
                        "y": 0.0,
                        "z": 0.0,
                        "yaw": 0.0,
                    },
                },
                "source": "operator",
                "sourceRef": "generic-retry-target",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
                "frameId": "map",
            },
        )
        assert target.status_code == 200, target.text
        before = target.json()
        before_states = before["nodeStates"]

        intervention_id = "intervention-generic-retry-1"
        message = {
            "interventionId": intervention_id,
            "text": "仍然去厨房，但到门口停下",
            "kind": "append_requirement",
        }
        failed = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=message,
        )
        assert failed.status_code == 503, failed.text
        assert planner.event_calls == 1

        fenced = app.state.mission_runtime.get(mission_id)
        assert fenced["snapshot"]["status"] == "suspended"
        assert fenced["snapshot"]["errorCode"] == "PLANNER_EVALUATION_PENDING"
        assert fenced["snapshot"]["activeSkillRequestId"] is None
        assert fenced["nodeStates"] == before_states
        assert fenced["skillExecutions"] == []
        claim = next(
            item
            for item in fenced["events"]
            if item["correlationId"] == intervention_id
        )
        assert claim["payload"]["phase"] == "claimed"
        assert claim["payload"]["previousSnapshot"]["status"] == "ready"

        # Every public/direct path that could otherwise clear the fence or
        # dispatch the old goal stays fail-closed.
        assert client.post(
            f"/api/v2/missions/{mission_id}/start", headers=headers()
        ).status_code == 409
        advanced = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 20},
        )
        assert advanced.status_code == 200, advanced.text
        assert advanced.json()["actions"] == []
        assert advanced.json()["stoppedBecause"] == "suspended"
        resumed = client.post(
            f"/api/v2/missions/{mission_id}/interventions",
            headers=headers(),
            json={
                "interventionId": "resume-during-planner-fence",
                "kind": "resume",
                "text": "恢复",
            },
        )
        assert resumed.status_code == 409, resumed.text
        issued = client.post(
            f"/api/v2/missions/{mission_id}/nodes/navigate-kitchen/skill-requests",
            headers=headers(),
        )
        assert issued.status_code == 409, issued.text
        assert app.state.mission.converge_supervised_missions() == []

        runtime = app.state.mission_runtime
        with pytest.raises(MissionConflict, match="owns the Mission fence"):
            runtime.skip_nodes(
                mission_id,
                ["navigate-kitchen"],
                reason="must not bypass planner",
                causation_id="skip-during-planner-fence",
            )
        active = PlanRevision.model_validate(fenced["activeRevision"])
        ordinary_revision = active.model_copy(
            update={
                "revisionId": "revision-bypass-attempt",
                "sequence": active.sequence + 1,
                "parentRevisionId": active.revisionId,
                "reasonCode": "test_bypass",
                "reason": "must remain fenced",
                "changeSummary": ["attempted ordinary revision"],
                "disposition": "human_required",
                "authorizationId": None,
                "createdAt": utc_now(),
            }
        )
        with pytest.raises(MissionConflict, match="owns the Mission fence"):
            runtime.apply_revision(ordinary_revision)
        assert runtime.get(mission_id)["skillExecutions"] == []

        # A different key cannot steal the durable fence.
        stolen = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={**message, "interventionId": "intervention-generic-other"},
        )
        assert stolen.status_code == 409, stolen.text
        assert planner.event_calls == 1

        # Recreate the service (and therefore its in-process locks) over the
        # same store/runtime.  The persisted claim is sufficient to retry and
        # restore the first claim's READY snapshot.
        original_service = app.state.mission
        restarted_service = MissionService(
            app.state.store,
            app.state.mission_runtime,
            planner,
            allow_synthetic_profile=original_service.allow_synthetic_profile,
            policy=original_service.policy,
        )
        restarted_service.robot_refresher = original_service.robot_refresher
        restarted_service.scene_context_client = original_service.scene_context_client
        requested_by = claim["payload"]["requestedBy"]
        recovered = restarted_service.respond(
            mission_id,
            {**message, "requestedBy": requested_by},
        )
        assert recovered["decision"] == {
            "kind": "continue",
            "reason": "fixture retry continues",
        }
        assert recovered["mission"]["snapshot"]["status"] == "ready"
        assert recovered["mission"]["snapshot"]["errorCode"] is None
        assert recovered["mission"]["nodeStates"] == before_states
        assert planner.event_calls == 2

        replay = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=message,
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["decision"] == recovered["decision"]
        assert replay.json()["mission"]["lastSequence"] == recovered["mission"][
            "lastSequence"
        ]
        assert planner.event_calls == 2


def test_generic_message_fences_before_a_blocked_robot_refresh() -> None:
    planner = ContinueMissionPlanningClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-blocked-refresh-1",
                "objective": "去厨房",
            },
        )
        assert draft.status_code == 201, draft.text
        mission_id = draft.json()["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        started = client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        )
        assert started.status_code == 200, started.text
        before_states = started.json()["nodeStates"]

        service = app.state.mission
        original_refresher = service.robot_refresher
        entered = threading.Event()
        release = threading.Event()
        outcome: dict[str, Any] = {}
        errors: list[BaseException] = []

        def blocked_refresher(robot_id: str | None) -> None:
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("blocked refresher was not released")
            if original_refresher is not None:
                original_refresher(robot_id)

        def respond() -> None:
            try:
                outcome.update(
                    service.respond(
                        mission_id,
                        {
                            "interventionId": "intervention-blocked-refresh-1",
                            "text": "保持原目标并继续",
                            "kind": "append_requirement",
                            "requestedBy": "operator-test",
                        },
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        service.robot_refresher = blocked_refresher
        worker = threading.Thread(target=respond, name="blocked-planner-refresh-test")
        worker.start()
        world_record = {
            "recordId": "record-blocked-refresh-1",
            "robotId": "robot-1",
            "kind": "fact",
            "subject": "厨房",
            "predicate": "visible",
            "value": True,
            "source": "operator",
            "sourceRef": "blocked-refresh-test",
            "confidence": 1.0,
            "observedAt": utc_now().isoformat(),
        }
        try:
            assert entered.wait(timeout=2)
            fenced = app.state.mission_runtime.get(mission_id)
            assert fenced["snapshot"]["status"] == "suspended"
            assert fenced["snapshot"]["errorCode"] == "PLANNER_EVALUATION_PENDING"
            assert fenced["nodeStates"] == before_states
            assert fenced["skillExecutions"] == []
            assert service.converge_supervised_missions() == []
            assert app.state.mission_runtime.get(mission_id)["skillExecutions"] == []
            assert planner.event_calls == 0
            blocked_world = client.post(
                f"/api/v2/missions/{mission_id}/world-records",
                headers=headers(),
                json=world_record,
            )
            assert blocked_world.status_code == 409, blocked_world.text
        finally:
            release.set()
            worker.join(timeout=5)
            service.robot_refresher = original_refresher
        assert not worker.is_alive()
        assert errors == []
        assert outcome["decision"]["kind"] == "continue"
        assert outcome["mission"]["snapshot"]["status"] == "ready"
        assert outcome["mission"]["nodeStates"] == before_states
        assert planner.event_calls == 1
        accepted_world = client.post(
            f"/api/v2/missions/{mission_id}/world-records",
            headers=headers(),
            json=world_record,
        )
        assert accepted_world.status_code == 200, accepted_world.text


def test_cancel_aborts_a_pending_planner_fence_without_stale_error() -> None:
    planner = FailOnceMissionPlanningClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-cancel-planner-fence-1",
                "objective": "去厨房",
            },
        )
        assert draft.status_code == 201, draft.text
        mission_id = draft.json()["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        message = {
            "interventionId": "intervention-cancelled-planner-1",
            "kind": "append_requirement",
            "text": "到门口停下",
        }
        failed = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=message,
        )
        assert failed.status_code == 503, failed.text

        cancelled = client.post(
            f"/api/v2/missions/{mission_id}/cancel",
            headers=headers(),
            json={"reason": "不再执行这项任务"},
        )
        assert cancelled.status_code == 200, cancelled.text
        snapshot = cancelled.json()["mission"]["snapshot"]
        assert snapshot["status"] == "cancelled"
        assert snapshot["errorCode"] is None
        assert snapshot["detail"] is None
        inspected = app.state.mission_runtime.inspect_planned_intervention(
            {
                "interventionId": message["interventionId"],
                "missionId": mission_id,
                "kind": message["kind"],
                "text": message["text"],
                "requestedBy": next(
                    item["payload"]["requestedBy"]
                    for item in cancelled.json()["mission"]["events"]
                    if item["correlationId"] == message["interventionId"]
                ),
            },
            request_sha256=next(
                item["payload"]["requestSha256"]
                for item in cancelled.json()["mission"]["events"]
                if item["correlationId"] == message["interventionId"]
            ),
        )
        assert inspected["state"] == "aborted"

        retried = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=message,
        )
        assert retried.status_code == 409, retried.text
        assert planner.event_calls == 1


def test_revision_request_fences_before_planner_and_retries_atomically() -> None:
    planner = BlockingFailOnceRevisionPlanningClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-revision-saga-draft-1",
                "objective": "去厨房",
            },
        )
        assert draft.status_code == 201, draft.text
        mission_id = draft.json()["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        started = client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        )
        assert started.status_code == 200, started.text
        before = started.json()

        body = {
            "requestId": "request-revision-saga-1",
            "requestedBy": "operator-test",
            "objective": "去厨房门口",
            "trigger": "operator_intervention",
            "triggerDetail": "到门口停下，不进入厨房",
        }
        errors: list[BaseException] = []

        def request_revision() -> None:
            try:
                app.state.mission.request_revision(mission_id, body)
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        worker = threading.Thread(
            target=request_revision,
            name="blocked-revision-planner-test",
        )
        worker.start()
        try:
            assert planner.entered.wait(timeout=2)
            fenced = app.state.mission_runtime.get(mission_id)
            assert fenced["snapshot"]["status"] == "suspended"
            assert fenced["snapshot"]["errorCode"] == "PLANNER_EVALUATION_PENDING"
            assert fenced["snapshot"]["activeSkillRequestId"] is None
            assert fenced["nodeStates"] == before["nodeStates"]
            assert fenced["skillExecutions"] == []
            assert app.state.mission.converge_supervised_missions() == []
            assert app.state.mission_runtime.get(mission_id)["skillExecutions"] == []
        finally:
            planner.release.set()
            worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], MissionPlanningUnavailable)
        failed = app.state.mission_runtime.get(mission_id)
        assert failed["snapshot"]["status"] == "suspended"
        assert failed["snapshot"]["errorCode"] == "PLANNER_EVALUATION_PENDING"
        assert len(failed["revisions"]) == 1

        completed = app.state.mission.request_revision(mission_id, body)
        assert completed["snapshot"]["status"] == "awaiting_review"
        assert completed["snapshot"]["errorCode"] is None
        assert len(completed["revisions"]) == 2
        assert planner.revision_calls == 2
        events = [
            item
            for item in completed["events"]
            if item["correlationId"] == body["requestId"]
        ]
        assert [item["payload"]["phase"] for item in events] == [
            "claimed",
            "completed",
        ]

        replay = app.state.mission.request_revision(mission_id, body)
        assert replay["lastSequence"] == completed["lastSequence"]
        assert len(replay["revisions"]) == 2
        assert planner.revision_calls == 2
        with pytest.raises(MissionConflict):
            app.state.mission.request_revision(
                mission_id,
                {**body, "objective": "换一个不同目标"},
            )


def test_internal_event_evaluation_fences_before_blocked_planner() -> None:
    planner = BlockingEventPlanningClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-blocked-event-draft-1",
                "objective": "去厨房",
            },
        )
        assert draft.status_code == 201, draft.text
        mission_id = draft.json()["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        started = client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        )
        assert started.status_code == 200, started.text
        before = started.json()
        event = {
            "eventType": "observation.updated",
            "payload": {"detail": "目标仍然可见"},
            "correlationId": "observation-event-blocked-1",
            "occurredAt": utc_now().isoformat(),
        }
        outcome: dict[str, Any] = {}
        errors: list[BaseException] = []

        def evaluate() -> None:
            try:
                outcome.update(app.state.mission.evaluate_event(mission_id, event))
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        worker = threading.Thread(target=evaluate, name="blocked-event-planner-test")
        worker.start()
        try:
            assert planner.entered.wait(timeout=2)
            fenced = app.state.mission_runtime.get(mission_id)
            assert fenced["snapshot"]["status"] == "suspended"
            assert fenced["snapshot"]["errorCode"] == "PLANNER_EVALUATION_PENDING"
            assert fenced["nodeStates"] == before["nodeStates"]
            assert fenced["skillExecutions"] == []
            assert app.state.mission.converge_supervised_missions() == []
            assert app.state.mission_runtime.get(mission_id)["skillExecutions"] == []
        finally:
            planner.release.set()
            worker.join(timeout=5)
        assert not worker.is_alive()
        assert errors == []
        assert outcome["decision"] == {
            "kind": "continue",
            "reason": "event accepted",
        }
        assert outcome["mission"]["snapshot"]["status"] == "ready"
        assert outcome["mission"]["nodeStates"] == before["nodeStates"]
        assert planner.event_calls == 1

        replay = app.state.mission.evaluate_event(mission_id, event)
        assert replay["decision"] == outcome["decision"]
        assert replay["mission"]["lastSequence"] == outcome["mission"][
            "lastSequence"
        ]
        assert planner.event_calls == 1


def _create_ambiguous_visual_mission(client: TestClient) -> tuple[str, dict[str, Any]]:
    created = client.post(
        "/api/v2/missions/drafts",
        headers=headers(),
        json={
            "requestId": f"request-ambiguous-{uuid4().hex}",
            "objective": "找到银白色机器人并去它旁边",
        },
    )
    assert created.status_code == 201, created.text
    mission_id = created.json()["draftId"]
    approved = client.post(
        f"/api/v2/missions/drafts/{mission_id}/approve",
        headers=headers(),
        json={"approvedBy": "operator-test"},
    )
    assert approved.status_code == 200, approved.text
    started = client.post(
        f"/api/v2/missions/{mission_id}/start",
        headers=headers(),
    )
    assert started.status_code == 200, started.text
    issued = client.post(
        f"/api/v2/missions/{mission_id}/nodes/observe-target/skill-requests",
        headers=headers(),
    )
    assert issued.status_code == 202, issued.text
    mission = issued.json()["mission"]
    assert mission["snapshot"]["status"] == "needs_input"
    assert mission["snapshot"]["errorCode"] == "VISUAL_NEEDS_INPUT"
    return mission_id, mission


def test_completed_generic_answer_replays_before_later_visual_candidate_input(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    planner = ContinueVisualMissionPlanningClient()
    visual = AmbiguousThenSelectedVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        visual_task_client=visual,
        mission_supervisor_interval=3600,
    )
    intervention_id = "intervention-generic-before-visual-input"
    answer = {
        "interventionId": intervention_id,
        "kind": "answer",
        "text": "先保持当前观察计划",
    }
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-generic-before-visual-input",
                "objective": "找到银白色机器人并去它旁边",
            },
        )
        assert created.status_code == 201, created.text
        mission_id = created.json()["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        assert client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        ).status_code == 200

        first = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=answer,
        )
        assert first.status_code == 200, first.text
        first_payload = first.json()
        assert first_payload["decision"] == {
            "kind": "continue",
            "reason": "fixture visual mission continues",
        }
        assert planner.event_calls == 1

        # A later evidence writer cannot steal or poison the completed
        # intervention identity.
        collision = client.post(
            f"/api/v2/missions/{mission_id}/world-records",
            headers=headers(),
            json={
                "recordId": intervention_id,
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "silver robot",
                "predicate": "visible",
                "value": True,
                "source": "operator",
                "sourceRef": "generic-correlation-collision",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
            },
        )
        assert collision.status_code == 409, collision.text

        issued = client.post(
            f"/api/v2/missions/{mission_id}/nodes/observe-target/skill-requests",
            headers=headers(),
        )
        assert issued.status_code == 202, issued.text
        ambiguous = issued.json()["mission"]
        assert ambiguous["snapshot"]["status"] == "needs_input"
        assert ambiguous["snapshot"]["errorCode"] == "VISUAL_NEEDS_INPUT"
        assert len(visual.requests) == 1

        replay = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=answer,
        )
        assert replay.status_code == 200, replay.text
        replay_payload = replay.json()
        assert replay_payload["decision"] == first_payload["decision"]
        assert replay_payload["mission"]["lastSequence"] == ambiguous["lastSequence"]
        assert replay_payload["mission"]["snapshot"]["status"] == "needs_input"
        assert planner.event_calls == 1
        assert len(visual.requests) == 1
        node_state = next(
            item
            for item in replay_payload["mission"]["nodeStates"]
            if item["nodeId"] == "observe-target"
        )
        assert node_state["status"] == "rejected"
        assert "visualCandidateSelection" not in node_state["state"]
        assert not any(
            isinstance(item.get("payload"), Mapping)
            and "visualCandidateSelection" in item["payload"]
            for item in replay_payload["mission"]["events"]
            if item["correlationId"] == intervention_id
        )

        rebound = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={**answer, "text": "同一个 ID 但不同答案"},
        )
        assert rebound.status_code == 409, rebound.text
        assert planner.event_calls == 1
        assert len(visual.requests) == 1


@pytest.mark.parametrize(
    ("text", "candidate_id"),
    [
        (
            "选择第一个候选,继续当前mission",
            AmbiguousThenSelectedVisualTaskClient.first_candidate_id,
        ),
        (
            "选择候选：左侧银白色机器人",
            AmbiguousThenSelectedVisualTaskClient.first_candidate_id,
        ),
        (
            f"confirm {AmbiguousThenSelectedVisualTaskClient.second_candidate_id}",
            AmbiguousThenSelectedVisualTaskClient.second_candidate_id,
        ),
    ],
)
def test_visual_candidate_free_text_allowlist_accepts_only_complete_selection_forms(
    text: str,
    candidate_id: str,
) -> None:
    ambiguity = {
        "nodeId": "observe-target",
        "sourceRequestId": "skill-observe-1",
        "selectionSourceAnalysisId": (
            AmbiguousThenSelectedVisualTaskClient.source_analysis_id
        ),
        "candidates": [
            {
                "candidateId": AmbiguousThenSelectedVisualTaskClient.first_candidate_id,
                "label": "左侧银白色机器人",
            },
            {
                "candidateId": AmbiguousThenSelectedVisualTaskClient.second_candidate_id,
                "label": "右侧银白色机器人",
            },
        ],
    }

    resolved = MissionService._resolve_visual_candidate_answer({}, text, ambiguity)

    assert resolved["candidateId"] == candidate_id
    assert resolved["selectionSourceAnalysisId"] == (
        AmbiguousThenSelectedVisualTaskClient.source_analysis_id
    )


def test_mission_v2_visual_candidate_answer_retries_same_node_as_attempt_two(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    planner = CandidateRetryPlanningClient()
    visual = AmbiguousThenSelectedVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        visual_task_client=visual,
        mission_supervisor_interval=3600,
    )
    intervention_id = "intervention-select-first-candidate"
    answer = {
        "text": "选择第一个候选，继续当前任务",
        "kind": "answer",
        "interventionId": intervention_id,
    }
    with TestClient(app) as client:
        mission_id, before = _create_ambiguous_visual_mission(client)
        active_revision_id = before["activeRevisionId"]
        answered = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=answer,
        )
        assert answered.status_code == 200, answered.text
        payload = answered.json()
        assert payload["decision"]["kind"] == "continue"
        assert (
            payload["decision"]["candidateId"]
            == AmbiguousThenSelectedVisualTaskClient.first_candidate_id
        )
        selected = payload["mission"]
        assert selected["snapshot"]["status"] == "ready"
        assert selected["activeRevisionId"] == active_revision_id
        assert len(selected["revisions"]) == 1
        node_state = next(
            item for item in selected["nodeStates"] if item["nodeId"] == "observe-target"
        )
        assert node_state["status"] == "pending"
        assert node_state["attempt"] == 1
        selection = node_state["state"]["visualCandidateSelection"]
        assert selection["candidateId"] == (
            AmbiguousThenSelectedVisualTaskClient.first_candidate_id
        )
        assert selection["selectionSourceAnalysisId"] == (
            AmbiguousThenSelectedVisualTaskClient.source_analysis_id
        )
        assert selection["sourceRequestId"] == before["skillExecutions"][0][
            "requestId"
        ]
        assert selection["selectedBy"]
        assert selection["selectedAt"]
        assert len(visual.requests) == 1

        collision = client.post(
            f"/api/v2/missions/{mission_id}/world-records",
            headers=headers(),
            json={
                "recordId": intervention_id,
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "silver robot",
                "predicate": "selected",
                "value": True,
                "source": "operator",
                "sourceRef": "candidate-correlation-collision",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
            },
        )
        assert collision.status_code == 409, collision.text

        repeated = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=answer,
        )
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["mission"]["lastSequence"] == selected["lastSequence"]
        assert len(
            [
                item
                for item in repeated.json()["mission"]["events"]
                if item["eventType"] == "operator.answer"
                and item["correlationId"] == intervention_id
            ]
        ) == 1

        rebound = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={
                **answer,
                "candidateId": (
                    AmbiguousThenSelectedVisualTaskClient.second_candidate_id
                ),
                "selectionSourceAnalysisId": (
                    AmbiguousThenSelectedVisualTaskClient.source_analysis_id
                ),
            },
        )
        assert rebound.status_code == 409, rebound.text
        changed_kind = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={
                **answer,
                "kind": "modify_goal",
            },
        )
        assert changed_kind.status_code == 409, changed_kind.text

        retried = client.post(
            f"/api/v2/missions/{mission_id}/nodes/observe-target/skill-requests",
            headers=headers(),
        )
        assert retried.status_code == 202, retried.text
        request = retried.json()["request"]
        assert request["revisionId"] == active_revision_id
        assert request["nodeId"] == "observe-target"
        assert request["requestId"].endswith("-observe-target-2")
        assert (
            request["input"]["candidateId"]
            == AmbiguousThenSelectedVisualTaskClient.first_candidate_id
        )
        assert request["input"]["selectionSourceAnalysisId"] == (
            AmbiguousThenSelectedVisualTaskClient.source_analysis_id
        )

    assert planner.event_calls == 0
    assert len(visual.requests) == 2
    assert visual.requests[1]["attempt"] == 2


@pytest.mark.parametrize(
    ("answer", "expected_status"),
    [
        (
            {
                "text": "选择这个候选",
                "kind": "answer",
                "candidateId": "candidate-from-another-mission",
                "selectionSourceAnalysisId": (
                    AmbiguousThenSelectedVisualTaskClient.source_analysis_id
                ),
            },
            422,
        ),
        (
            {
                "text": "选择左侧候选",
                "kind": "answer",
                "candidateId": AmbiguousThenSelectedVisualTaskClient.first_candidate_id,
                "selectionSourceAnalysisId": "analysis-stale-result",
            },
            409,
        ),
        ({"text": "选择第三个候选", "kind": "answer"}, 422),
        ({"text": "选择第一个候选和第二个候选", "kind": "answer"}, 422),
        ({"text": "第一个还是第二个？", "kind": "answer"}, 422),
        ({"text": "候选1或候选2", "kind": "answer"}, 422),
        ({"text": "不要选择第一个候选", "kind": "answer"}, 422),
        ({"text": "是不是选择第一个候选？", "kind": "answer"}, 422),
        ({"text": "如果可以就选择第一个候选", "kind": "answer"}, 422),
        (
            {
                "text": "选择左侧候选",
                "kind": "answer",
                "candidateId": AmbiguousThenSelectedVisualTaskClient.first_candidate_id,
            },
            422,
        ),
    ],
)
def test_mission_v2_invalid_visual_candidate_stays_needs_input_without_dispatch(
    monkeypatch,
    answer: dict[str, Any],
    expected_status: int,
) -> None:
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    planner = CandidateRetryPlanningClient()
    visual = AmbiguousThenSelectedVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        visual_task_client=visual,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        mission_id, before = _create_ambiguous_visual_mission(client)
        rejected = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json=answer,
        )
        assert rejected.status_code == expected_status, rejected.text
        current = client.get(
            f"/api/v2/missions/{mission_id}",
            headers=headers(),
        )
        assert current.status_code == 200, current.text
        mission = current.json()
        assert mission["snapshot"]["status"] == "needs_input"
        assert mission["lastSequence"] == before["lastSequence"]
        node_state = next(
            item for item in mission["nodeStates"] if item["nodeId"] == "observe-target"
        )
        assert node_state["status"] == "rejected"
        assert node_state["attempt"] == 1
        assert not any(item["eventType"] == "operator.answer" for item in mission["events"])

    assert planner.event_calls == 0
    assert len(visual.requests) == 1


def test_mission_v2_expired_visual_candidate_stays_needs_input_without_dispatch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    planner = CandidateRetryPlanningClient()
    visual = AmbiguousThenSelectedVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=planner,
        visual_task_client=visual,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        mission_id, before = _create_ambiguous_visual_mission(client)
        monkeypatch.setattr(
            "packages.mission_core.runtime.utc_now",
            lambda: utc_now() + timedelta(hours=1),
        )
        rejected = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={"text": "选择第一个候选", "kind": "answer"},
        )
        assert rejected.status_code == 409, rejected.text
        current = client.get(
            f"/api/v2/missions/{mission_id}",
            headers=headers(),
        )
        assert current.status_code == 200, current.text
        mission = current.json()
        assert mission["snapshot"]["status"] == "needs_input"
        assert mission["lastSequence"] == before["lastSequence"]
        assert not any(item["eventType"] == "operator.answer" for item in mission["events"])

    assert planner.event_calls == 0
    assert len(visual.requests) == 1


def test_mission_v2_visual_skill_keeps_qwen_hypothesis_separate_from_geometry_fact(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    visual = FakeVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=VisualMissionPlanningClient(),
        visual_task_client=visual,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-visual-chain-1",
                "objective": "找到银白色机器人并去它旁边",
            },
        )
        assert created.status_code == 201, created.text
        draft_id = created.json()["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{draft_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        assert client.post(
            f"/api/v2/missions/{draft_id}/start",
            headers=headers(),
        ).status_code == 200

        issued = client.post(
            f"/api/v2/missions/{draft_id}/nodes/observe-target/skill-requests",
            headers=headers(),
        )
        assert issued.status_code == 202, issued.text
        response = issued.json()
        assert response["dispatch"]["mode"] == "h_ide_vision"
        assert response["mission"]["snapshot"]["status"] == "ready"
        assert response["mission"]["snapshot"]["activeNodeId"] == "navigate-target"
        records = response["mission"]["worldRecords"]
        visual_records = [item for item in records if item["predicate"] != "mission_origin"]
        assert {(item["kind"], item["source"]) for item in visual_records} == {
            ("hypothesis", "qwen"),
            ("fact", "geometry"),
        }
        target_fact = next(
            item for item in visual_records if item["kind"] == "fact"
        )
        assert target_fact["predicate"] == "target_pose"
        assert target_fact["value"]["targetPose"]["frameId"] == "map"
        assert target_fact["value"]["targetPose"]["x"] == 1.25

        evidence = client.get(
            f"/api/v2/missions/{draft_id}/visual-evidence/"
            f"analysis-silver-robot-1/{visual.evidence_sha256}",
            headers=headers(),
        )
        assert evidence.status_code == 200, evidence.text
        assert evidence.content == visual.evidence_body
        assert evidence.headers["content-type"] == "image/jpeg"
        assert evidence.headers["cache-control"] == "private, no-store, max-age=0"

        unbound = client.get(
            f"/api/v2/missions/{draft_id}/visual-evidence/"
            f"analysis-silver-robot-1/{'a' * 64}",
            headers=headers(),
        )
        assert unbound.status_code == 422

    assert len(visual.requests) == 1
    assert visual.evidence_requests == [
        ("analysis-silver-robot-1", visual.evidence_sha256)
    ]
    request = visual.requests[0]
    assert request["sensorId"] == "head_rgbd"
    assert request["mode"] == "object_approach"
    assert request["targetDescription"] == "银白色机器人"


def test_mission_v2_dispatches_reviewed_pose_only_through_robot_agent_nav2(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    driver = MissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-live-nav-1",
                "objective": "去已识别的厨房目标点",
            },
        )
        assert created.status_code == 201, created.text
        draft_id = created.json()["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{draft_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        assert client.post(
            f"/api/v2/missions/{draft_id}/start",
            headers=headers(),
        ).status_code == 200

        injected = client.post(
            f"/api/v2/missions/{draft_id}/nodes/navigate-kitchen/skill-requests",
            headers=headers(),
            json={
                "input": {
                    "goalPose": {
                        "frameId": "map",
                        "x": 99.0,
                        "y": 99.0,
                        "z": 0.0,
                        "yaw": 0.0,
                    }
                }
            },
        )
        assert injected.status_code == 422
        assert "server-owned" in injected.json()["detail"]
        assert driver.dispatches == []

        trusted_target = client.post(
            f"/api/v2/missions/{draft_id}/world-records",
            headers=headers(),
            json={
                "recordId": "record-live-kitchen-target-1",
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "厨房",
                "predicate": "target_pose",
                "value": {
                    "targetPose": {
                        "frameId": "map",
                        "x": 1.5,
                        "y": 0.25,
                        "z": 0,
                        "yaw": 0.1,
                    }
                },
                "source": "geometry",
                "sourceRef": "visual-resolution-1",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
                "frameId": "map",
                "evidenceIds": ["visual-evidence-1"],
            },
        )
        assert trusted_target.status_code == 200, trusted_target.text

        issued = client.post(
            f"/api/v2/missions/{draft_id}/nodes/navigate-kitchen/skill-requests",
            headers=headers(),
        )
        assert issued.status_code == 202, issued.text
        response = issued.json()
        assert response["dispatch"]["mode"] == "robot_agent"
        assert response["dispatch"]["receipt"]["status"] == "succeeded"
        assert response["mission"]["snapshot"]["status"] == "completed"

    assert len(driver.dispatches) == 1
    command, mapping = driver.dispatches[0]
    assert command.commandKey == NAVIGATION_COMMAND_KEY
    assert command.payload["observationSessionId"] == "observation-live-1"
    assert command.payload["robotBootId"] == "boot-live-1"
    assert command.payload["localizationEpochId"] == "epoch-live-1"
    assert command.payload["missionId"] == draft_id
    assert "path" not in command.payload
    assert "cmd_vel" not in command.payload
    assert mapping.transport.value == "robot_agent"


def test_skill_issue_rejects_stale_input_after_same_node_revision(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    driver = MissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=SameNodeTargetRevisionPlanningClient(),
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-stale-skill-input-1",
                "objective": "去审核过的目标点",
            },
        )
        assert draft.status_code == 201, draft.text
        mission_id = draft.json()["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text

        for record_id, x in (("target-old", 0.5), ("target-new", 1.5)):
            recorded = client.post(
                f"/api/v2/missions/{mission_id}/world-records",
                headers=headers(),
                json={
                    "recordId": record_id,
                    "robotId": "robot-1",
                    "kind": "fact",
                    "subject": "审核目标",
                    "predicate": "target_pose",
                    "value": {
                        "targetPose": {
                            "frameId": "map",
                            "x": x,
                            "y": 0.0,
                            "z": 0.0,
                            "yaw": 0.0,
                        }
                    },
                    "source": "operator",
                    "sourceRef": f"operator-{record_id}",
                    "confidence": 1.0,
                    "observedAt": utc_now().isoformat(),
                    "frameId": "map",
                },
            )
            assert recorded.status_code == 200, recorded.text
        started = client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        )
        assert started.status_code == 200, started.text

        service = app.state.mission
        original_refresher = service.robot_refresher
        original_authorize_skill = service.policy.authorize_skill
        entered = threading.Event()
        release = threading.Event()
        refresh_lock = threading.Lock()
        refresh_count = 0
        issue_responses: list[Any] = []
        issue_errors: list[BaseException] = []

        def blocked_second_refresher(robot_id: str | None) -> None:
            nonlocal refresh_count
            with refresh_lock:
                refresh_count += 1
                should_block = refresh_count == 2
            if should_block:
                entered.set()
                if not release.wait(timeout=5):
                    raise AssertionError("blocked skill refresher was not released")
            if original_refresher is not None:
                original_refresher(robot_id)

        def issue() -> None:
            try:
                issue_responses.append(
                    client.post(
                        f"/api/v2/missions/{mission_id}/nodes/"
                        "navigate-kitchen/skill-requests",
                        headers=headers(),
                    )
                )
            except BaseException as exc:  # pragma: no cover - asserted below
                issue_errors.append(exc)

        service.robot_refresher = blocked_second_refresher
        # Isolate the runtime compare-and-set contract from a coincidental
        # policy denial caused by revoking the old revision's grant. Without
        # the state binding below, the request would otherwise be constructed
        # with the new revision/authorization but the old resolved target.
        service.policy.authorize_skill = lambda **_: None
        worker = threading.Thread(target=issue, name="stale-skill-input-test")
        worker.start()
        try:
            assert entered.wait(timeout=2)
            revised = client.post(
                f"/api/v2/missions/{mission_id}/revisions",
                headers=headers(),
                json={
                    "requestId": "request-stale-skill-revision-1",
                    "objective": "改去新审核目标点",
                    "trigger": "operator_intervention",
                    "triggerDetail": "将同一导航节点绑定到新目标",
                },
            )
            assert revised.status_code == 200, revised.text
            pending = revised.json()["revisions"][-1]
            assert pending["nodes"][0]["nodeId"] == "navigate-kitchen"
            assert pending["nodes"][0]["intent"] == {
                "targetRecordId": "target-new"
            }
            newly_approved = client.post(
                f"/api/v2/missions/{mission_id}/revisions/"
                f"{pending['revisionId']}/approve",
                headers=headers(),
                json={"approvedBy": "operator-test"},
            )
            assert newly_approved.status_code == 200, newly_approved.text
            assert newly_approved.json()["snapshot"]["status"] == "ready"
            approved_revision_id = newly_approved.json()["activeRevision"][
                "revisionId"
            ]
        finally:
            release.set()
            worker.join(timeout=5)
            service.robot_refresher = original_refresher
            service.policy.authorize_skill = original_authorize_skill

        assert not worker.is_alive()
        assert issue_errors == []
        assert len(issue_responses) == 1
        rejected = issue_responses[0]
        assert rejected.status_code == 409, rejected.text
        assert "mission state changed before skill issue" in rejected.json()["detail"]

        current = client.get(
            f"/api/v2/missions/{mission_id}", headers=headers()
        )
        assert current.status_code == 200, current.text
        mission = current.json()
        assert mission["snapshot"]["status"] == "ready"
        assert mission["activeRevision"]["revisionId"] == approved_revision_id
        assert mission["activeRevision"]["nodes"][0]["intent"] == {
            "targetRecordId": "target-new"
        }
        assert mission["skillExecutions"] == []
        assert not any(
            event["eventType"] == "skill.requested" for event in mission["events"]
        )
    assert driver.dispatches == []


def test_mission_v2_refuses_navigation_approval_without_live_map_pose(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    driver = NoPoseMissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={"requestId": "request-no-live-pose-1", "objective": "去厨房"},
        )
        assert draft.status_code == 201, draft.text
        mission_id = draft.json()["draftId"]

        refused = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert refused.status_code == 409
        assert "map pose is unavailable" in refused.json()["detail"]

    assert app.state.store.get_mission(mission_id) is None
    assert driver.dispatches == []


def test_mission_v2_refuses_real_agent_draft_without_operator_bound_epoch(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.delenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", raising=False)
    driver = MissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={"requestId": "request-no-epoch-1", "objective": "去厨房"},
        )
        assert response.status_code == 422
        assert "localizationEpochId" in response.json()["detail"]


def test_mission_v2_rejects_approval_after_localization_epoch_changes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    driver = MissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={"requestId": "request-epoch-change-1", "objective": "去厨房"},
        )
        assert draft.status_code == 201, draft.text
        monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-2")
        approved = client.post(
            f"/api/v2/missions/drafts/{draft.json()['draftId']}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 409
        assert "epoch changed" in approved.json()["detail"]


def test_mission_v2_clears_live_pose_when_robot_state_stops_projecting_it(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    driver = PoseDropMissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
    )
    assert app.state.store.get_robot("robot-1")["pose"] is not None
    driver.drop_pose = True
    app.state.refresh_robot_agent_snapshot()
    assert app.state.store.get_robot("robot-1").get("pose") is None


def test_mission_v2_advance_runs_visual_geometry_to_nav2_as_one_continuous_mission(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    driver = MissionRobotDriver()
    visual = FakeVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=VisualMissionPlanningClient(),
        visual_task_client=visual,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-continuous-chain-1",
                "objective": "找到银白色机器人并去机器人旁边",
                "maximumGoalDistanceM": 3.0,
                "maximumCumulativeDistanceM": 5.0,
            },
        )
        assert created.status_code == 201, created.text
        mission_id = created.json()["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        origin = next(
            item
            for item in approved.json()["worldRecords"]
            if item["predicate"] == "mission_origin"
        )
        assert origin["source"] == "robot_agent"
        assert origin["value"]["x"] == 0.25

        advanced = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 10},
        )
        assert advanced.status_code == 200, advanced.text
        result = advanced.json()
        assert result["stoppedBecause"] == "completed"
        assert [item["kind"] for item in result["actions"]] == [
            "started",
            "skill_issued",
            "skill_issued",
        ]
        assert [item.get("skillId") for item in result["actions"]] == [
            None,
            "vision.observe",
            NAVIGATION_COMMAND_KEY,
        ]

    assert len(visual.requests) == 1
    assert len(driver.dispatches) == 1
    command, _ = driver.dispatches[0]
    assert command.payload["payloadType"] == "visual_resolution"
    assert command.payload["visualResolutionId"] == "analysis-silver-robot-1"
    assert command.payload["goalPose"]["x"] == 1.25
    assert "path" not in command.payload
    assert "cmd_vel" not in command.payload


def test_deterministic_object_search_skips_remaining_views_and_dispatches_nav2(
    monkeypatch,
) -> None:
    from services.control_api.mission_client import UnavailableMissionPlanningClient

    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    class SequentialVisualTaskClient(FakeVisualTaskClient):
        def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            result = dict(super().analyze(request))
            sequence = len(self.requests)
            hypothesis_id = f"hypothesis-chair-{sequence}"
            result["analysisId"] = f"analysis-chair-{sequence}"
            result["hypotheses"] = [
                {
                    **dict(result["hypotheses"][0]),
                    "hypothesisId": hypothesis_id,
                    "label": "chair",
                }
            ]
            result["geometryCandidates"] = [
                {
                    **dict(result["geometryCandidates"][0]),
                    "geometryCandidateId": f"geometry-chair-{sequence}",
                    "hypothesisId": hypothesis_id,
                }
            ]
            return result

    driver = OfficeInspectionMissionRobotDriver()
    visual = SequentialVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=UnavailableMissionPlanningClient(
            "object approach must not call the text planner"
        ),
        visual_task_client=visual,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-chair-search-1",
                "missionId": "mission-chair-search-1",
                "robotId": "robot-1",
                "objective": "去椅子旁边站着",
                "requestedBy": "operator-test",
                "maximumGoalDistanceM": 3.0,
                "maximumCumulativeDistanceM": 10.0,
            },
        )
        assert created.status_code == 201, created.text
        mission_id = created.json()["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text

        advanced = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 50},
        )
        assert advanced.status_code == 200, (
            advanced.text,
            app.state.runtime.diagnostics[-5:],
        )
        mission = advanced.json()["mission"]
        assert mission["snapshot"]["status"] == "completed"
        skipped = {
            item["nodeId"]
            for item in mission["nodeStates"]
            if item["status"] == "skipped"
        }
        assert any("view-02" in item for item in skipped)
        # The fake Robot Agent projects the achieved Nav2 goal exactly, so the
        # new post-approach facing checkpoint must be durably skipped instead
        # of dispatching an invalid zero Spin or pausing for user input.
        assert "object-approach-align" in skipped
        assert any(
            item.get("kind") == "skill_skipped"
            and item.get("nodeId") == "object-approach-align"
            for item in advanced.json()["actions"]
        )
        assert len(visual.requests) == 2  # target grounding + arrival verification
        assert visual.requests[0]["mode"] == "object_approach"
        assert visual.requests[1]["mode"] == "arrival_verification"

    navigation_dispatches = [
        request
        for request, _ in driver.dispatches
        if request.commandKey == NAVIGATION_COMMAND_KEY
    ]
    assert len(navigation_dispatches) == 1
    assert navigation_dispatches[0].payload["goalPose"]["x"] == 1.25


def test_deterministic_active_search_rotates_and_reports_not_found(
    monkeypatch,
) -> None:
    from services.control_api.mission_client import UnavailableMissionPlanningClient

    class MissingObjectVisualTaskClient(FakeVisualTaskClient):
        def analyze(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
            result = dict(super().analyze(request))
            sequence = len(self.requests)
            result.update(
                analysisId=f"analysis-missing-computer-{sequence}",
                status="rejected",
                hypotheses=[],
                geometryCandidates=[],
                ambiguityCandidates=[],
                rejectionReason="target_not_found",
            )
            return result

    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    driver = OfficeInspectionMissionRobotDriver()
    visual = MissingObjectVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=UnavailableMissionPlanningClient(
            "active search must not call the text planner"
        ),
        visual_task_client=visual,
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-computer-search-miss-1",
                "missionId": "mission-computer-search-miss-1",
                "robotId": "robot-1",
                "objective": "搜索并找到电脑",
                "requestedBy": "operator-test",
            },
        )
        assert created.status_code == 201, created.text
        mission_id = created.json()["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        advanced = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 50},
        )
        assert advanced.status_code == 200, advanced.text
        mission = advanced.json()["mission"]
        assert mission["snapshot"]["status"] == "completed"
        report = next(
            execution["result"]["result"]["missionReport"]
            for execution in mission["skillExecutions"]
            if execution["request"]["skillId"] == "mission.report"
        )
        assert report["found"] is False
        assert "没有找到" in report["narrativeZhCN"]

    assert len(visual.requests) == 4
    assert sum(
        1
        for request, _ in driver.dispatches
        if request.commandKey == "navigation.rotate"
    ) == 3


def test_second_robot_mission_stays_reviewable_but_cannot_start_concurrently() -> None:
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=FakeMissionPlanningClient(),
        mission_supervisor_interval=3600,
    )
    with TestClient(app) as client:
        first = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-queue-first",
                "missionId": "mission-queue-first",
                "robotId": "robot-1",
                "objective": "向前移动",
            },
        )
        assert first.status_code == 201, first.text
        first_id = first.json()["draftId"]
        approved = client.post(
            f"/api/v2/missions/drafts/{first_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text

        second = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-queue-second",
                "missionId": "mission-queue-second",
                "robotId": "robot-1",
                "objective": "向后移动",
            },
        )
        assert second.status_code == 201, second.text
        second_id = second.json()["draftId"]
        blocked = client.post(
            f"/api/v2/missions/drafts/{second_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert blocked.status_code == 422, blocked.text
        assert "ROBOT_MISSION_BUSY" in blocked.json()["detail"]
        assert client.get(
            f"/api/v2/missions/drafts/{second_id}", headers=headers()
        ).json()["status"] == "draft"


def test_mission_v2_advance_reconciles_running_robot_agent_without_replay(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    driver = RunningMissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=VisualMissionPlanningClient(),
        visual_task_client=FakeVisualTaskClient(),
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-running-reconcile-1",
                "objective": "找到银白色机器人并去机器人旁边",
            },
        ).json()
        mission_id = draft["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200

        advanced = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 10},
        )
        assert advanced.status_code == 200, advanced.text
        assert advanced.json()["stoppedBecause"] == "executing"
        command, _ = driver.dispatches[0]
        assert (command.expiresAt - command.issuedAt).total_seconds() == 30.0
        changed_while_moving = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={"text": "换一个目标", "kind": "modify_goal"},
        )
        assert changed_while_moving.status_code == 409
        reconciled = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 10},
        )
        assert reconciled.status_code == 200, reconciled.text
        result = reconciled.json()
        assert result["stoppedBecause"] == "completed"
        assert any(item["kind"] == "skill_reconciled" for item in result["actions"])

    # The accepted Nav2 goal is queried by its immutable command ID; it is
    # never dispatched a second time while the Mission coordinator waits.
    assert len(driver.dispatches) == 1
    robot = app.state.store.get_robot("robot-1")
    assert robot is not None
    assert robot["pose"]["x"] == 1.25
    assert robot["details"]["poseSource"] == "nav2_terminal_goal"


def test_mission_v2_cancel_stops_active_nav2_command_before_terminalizing_mission(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "synthetic")
    driver = RunningMissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=VisualMissionPlanningClient(),
        visual_task_client=FakeVisualTaskClient(),
    )
    with TestClient(app) as client:
        draft = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-running-cancel-1",
                "objective": "找到银白色机器人并去机器人旁边",
            },
        ).json()
        mission_id = draft["draftId"]
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        running = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 10},
        )
        assert running.status_code == 200, running.text
        active_request_id = running.json()["mission"]["snapshot"][
            "activeSkillRequestId"
        ]

        for kind in ("pause", "cancel"):
            bypass = client.post(
                f"/api/v2/missions/{mission_id}/interventions",
                headers=headers(),
                json={
                    "interventionId": f"direct-{kind}-active-request",
                    "kind": kind,
                    "text": f"direct {kind} must not orphan Robot Agent work",
                },
            )
            assert bypass.status_code == 409, bypass.text
            unchanged = app.state.mission_runtime.get(mission_id)
            assert unchanged["snapshot"]["activeSkillRequestId"] == active_request_id
            assert unchanged["snapshot"]["status"] in {"dispatching", "executing"}
            assert driver.cancellations == []

        cancelled = client.post(
            f"/api/v2/missions/{mission_id}/cancel",
            headers=headers(),
            json={"requestedBy": "operator-test", "reason": "停止本次联调"},
        )
        assert cancelled.status_code == 200, cancelled.text
        payload = cancelled.json()
        assert payload["receipt"]["status"] == "cancelled"
        assert payload["mission"]["snapshot"]["status"] == "cancelled"
        assert payload["mission"]["snapshot"]["activeSkillRequestId"] is None

        still_cancelled = client.post(
            f"/api/v2/missions/{mission_id}/advance",
            headers=headers(),
            json={"maximumSteps": 10},
        )
        assert still_cancelled.status_code == 200
        assert still_cancelled.json()["stoppedBecause"] == "cancelled"

    assert driver.cancellations == [active_request_id]
    assert len(driver.dispatches) == 1


def test_office_station_composite_rotates_observes_and_completes_parent(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "live")
    from services.control_api import mission_service as mission_service_module
    from services.control_api.mission_client import UnavailableMissionPlanningClient
    from services.control_api.office_inspection import (
        catalog as office_catalog,
        envelope_defaults as office_envelope_defaults,
    )

    fast_catalog = office_catalog()
    fast_catalog["capturePolicy"] = {
        **fast_catalog["capturePolicy"],
        "stopBeforeCaptureS": 0.0,
    }
    monkeypatch.setattr(
        mission_service_module,
        "office_inspection_catalog",
        lambda: fast_catalog,
    )
    monkeypatch.setattr(
        mission_service_module,
        "office_inspection_envelope_defaults",
        lambda objective: {
            **office_envelope_defaults(objective),
            "maximumGoalDistanceM": 10.0,
        },
    )
    driver = OfficeInspectionMissionRobotDriver()
    visual = OfficeSceneDescriptionVisualTaskClient()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=UnavailableMissionPlanningClient(
            "deterministic office inspection must not call the text planner"
        ),
        visual_task_client=visual,
        mission_supervisor_interval=3600,
    )
    mission_id = "mission-office-composite-1"
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-office-composite-1",
                "missionId": mission_id,
                "robotId": "robot-1",
                "objective": (
                    "走遍所有办公室，并把所有办公室里面的信息和路上的信息都给我"
                ),
                "requestedBy": "operator-test",
            },
        )
        assert created.status_code == 201, created.text
        assert len(created.json()["plan"]["nodes"]) == 41
        approved = client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        assert client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        ).status_code == 200
        target = client.post(
            f"/api/v2/missions/{mission_id}/world-records",
            headers=headers(),
            json={
                "recordId": (
                    f"world-{mission_id}-scene-target-corridor_west"
                ),
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "corridor_west",
                "predicate": "target_pose",
                "value": {
                    "goalPose": {
                        "frameId": "map",
                        "x": -6.15,
                        "y": 0.35,
                        "z": 0.0,
                        "yaw": 0.0,
                    }
                },
                "source": "operator",
                "sourceRef": "reviewed-office-catalog-fixture",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
                "frameId": "map",
            },
        )
        assert target.status_code == 200, target.text
        navigation = client.post(
            f"/api/v2/missions/{mission_id}/nodes/"
            "office-nav-01-corridor_west/skill-requests",
            headers=headers(),
        )
        assert navigation.status_code == 202, navigation.text
        assert navigation.json()["mission"]["snapshot"]["activeNodeId"] == (
            "office-scan-01-corridor_west"
        )
        scan = client.post(
            f"/api/v2/missions/{mission_id}/nodes/"
            "office-scan-01-corridor_west/skill-requests",
            headers=headers(),
        )
        assert scan.status_code == 202, scan.text
        assert scan.json()["dispatch"]["mode"] == "office_inspection_pending"

        deadline = time.monotonic() + 5.0
        mission: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            response = client.get(
                f"/api/v2/missions/{mission_id}",
                headers=headers(),
            )
            assert response.status_code == 200, response.text
            mission = response.json()
            if mission["snapshot"]["activeNodeId"] != (
                "office-scan-01-corridor_west"
            ):
                break
            time.sleep(0.02)
        assert mission is not None
        assert mission["snapshot"]["status"] == "ready"
        assert mission["snapshot"]["activeNodeId"] == (
            "office-nav-02-reception_entry"
        )
        execution = next(
            item
            for item in mission["skillExecutions"]
            if item["request"]["skillId"] == "inspection.scan_station"
        )
        inspection = execution["result"]["result"]["inspectionResult"]
        assert execution["result"]["status"] == "succeeded"
        assert inspection["stationId"] == "corridor_west"
        assert len(inspection["headings"]) == 4
        assert inspection["cumulativeYawRad"] == pytest.approx(math.tau)

    assert len(visual.requests) == 4
    assert [item["nodeId"] for item in visual.requests] == [
        "office-scan-01-corridor_west-view-01",
        "office-scan-01-corridor_west-view-02",
        "office-scan-01-corridor_west-view-03",
        "office-scan-01-corridor_west-view-04",
    ]
    assert len(
        {
            (item["missionId"], item["revisionId"], item["nodeId"], item["attempt"])
            for item in visual.requests
        }
    ) == 4
    assert [item[0].commandKey for item in driver.dispatches] == [
        NAVIGATION_COMMAND_KEY,
        "navigation.rotate",
        "navigation.rotate",
        "navigation.rotate",
    ]


def test_office_heading_order_minimizes_yaw_and_preserves_catalog_indexes() -> None:
    headings = [0.0, math.pi / 2, math.pi, -math.pi / 2]

    ordered = _minimum_travel_heading_order(
        headings,
        current_yaw=-math.pi / 2 - 0.02,
    )

    assert [index for index, _ in ordered] == [4, 1, 2, 3]
    assert sorted(ordered) == list(enumerate(headings, start=1))
    yaw = -math.pi / 2 - 0.02
    travel = 0.0
    for _, heading in ordered:
        travel += abs(math.atan2(math.sin(heading - yaw), math.cos(heading - yaw)))
        yaw = heading
    assert travel == pytest.approx(3 * math.pi / 2 + 0.02)
    assert travel < math.tau


def test_cancelling_office_station_composite_cancels_active_rotation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ROBOT_AGENT_ID", "robot-1")
    monkeypatch.setenv("ROBOT_AGENT_LOCALIZATION_EPOCH_ID", "epoch-live-1")
    monkeypatch.setenv("HIDE_VISION_SOURCE_MODE", "live")
    from services.control_api import mission_service as mission_service_module
    from services.control_api.office_inspection import (
        envelope_defaults as office_envelope_defaults,
    )

    monkeypatch.setattr(
        mission_service_module,
        "office_inspection_envelope_defaults",
        lambda objective: {
            **office_envelope_defaults(objective),
            "maximumGoalDistanceM": 10.0,
        },
    )
    driver = RunningOfficeInspectionMissionRobotDriver()
    app = create_app(
        auth_token=TOKEN,
        adapters={"robot_agent": RobotAgentAdapter(driver=driver)},
        mission_planning_client=FakeMissionPlanningClient(),
        visual_task_client=OfficeSceneDescriptionVisualTaskClient(),
        mission_supervisor_interval=3600,
    )
    mission_id = "mission-office-composite-cancel-1"
    with TestClient(app) as client:
        created = client.post(
            "/api/v2/missions/drafts",
            headers=headers(),
            json={
                "requestId": "request-office-composite-cancel-1",
                "missionId": mission_id,
                "robotId": "robot-1",
                "objective": (
                    "走遍所有办公室，并把所有办公室里面的信息和路上的信息都给我"
                ),
                "requestedBy": "operator-test",
            },
        )
        assert created.status_code == 201, created.text
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200
        assert client.post(
            f"/api/v2/missions/{mission_id}/start",
            headers=headers(),
        ).status_code == 200
        target = client.post(
            f"/api/v2/missions/{mission_id}/world-records",
            headers=headers(),
            json={
                "recordId": f"world-{mission_id}-scene-target-corridor_west",
                "robotId": "robot-1",
                "kind": "fact",
                "subject": "corridor_west",
                "predicate": "target_pose",
                "value": {
                    "goalPose": {
                        "frameId": "map",
                        "x": -6.15,
                        "y": 0.35,
                        "z": 0.0,
                        "yaw": 0.0,
                    }
                },
                "source": "operator",
                "sourceRef": "reviewed-office-catalog-fixture",
                "confidence": 1.0,
                "observedAt": utc_now().isoformat(),
                "frameId": "map",
            },
        )
        assert target.status_code == 200, target.text
        navigation = client.post(
            f"/api/v2/missions/{mission_id}/nodes/"
            "office-nav-01-corridor_west/skill-requests",
            headers=headers(),
        )
        assert navigation.status_code == 202, navigation.text
        scan = client.post(
            f"/api/v2/missions/{mission_id}/nodes/"
            "office-scan-01-corridor_west/skill-requests",
            headers=headers(),
        )
        assert scan.status_code == 202, scan.text

        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if len(driver.dispatches) >= 2:
                break
            time.sleep(0.01)
        assert len(driver.dispatches) == 2
        rotation_request = driver.dispatches[-1][0]
        assert rotation_request.commandKey == "navigation.rotate"

        cancelled = client.post(
            f"/api/v2/missions/{mission_id}/cancel",
            headers=headers(),
            json={"requestedBy": "operator-test", "reason": "停止办公室巡检"},
        )
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["mission"]["snapshot"]["status"] == "cancelled"
        assert driver.cancellations == [rotation_request.commandId]

        time.sleep(0.05)
        mission = app.state.mission_runtime.get(mission_id)
        assert mission["snapshot"]["activeSkillRequestId"] is None
