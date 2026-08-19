from __future__ import annotations

from datetime import timedelta
import hashlib
import math
from typing import Any, Mapping

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
from packages.mission_core import MissionNode
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
from services.control_api.main import create_app
from services.control_api.mission_service import MissionService


TOKEN = "mission-v2-test-token"


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


class RespondingMissionPlanningClient(FakeMissionPlanningClient):
    def evaluate_event(
        self,
        event: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
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
    return {"Authorization": f"Bearer {TOKEN}"}


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

        issued = client.post(
            f"/api/v2/missions/{draft_id}/nodes/navigate-kitchen/skill-requests",
            headers=headers(),
            json={"input": {"targetRef": "semantic:kitchen"}},
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
        assert [event["sequence"] for event in mission["events"]] == [1, 2, 3, 4, 5]


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
        assert advanced.status_code == 200, advanced.text
        result = advanced.json()
        assert result["stoppedBecause"] == "dispatching"
        request = result["mission"]["skillExecutions"][0]["request"]
        assert request["input"]["payloadType"] == "spatial_resolution"
        assert request["input"]["goalPose"]["x"] == 1.5
        assert request["input"]["sceneRef"] == {"sceneId": "scene-lab", "revision": 3}
        assert "path" not in request["input"]


def test_mission_v2_operator_message_creates_a_reviewed_plan_revision() -> None:
    app = create_app(
        auth_token=TOKEN,
        mission_planning_client=RespondingMissionPlanningClient(),
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
        assert client.post(
            f"/api/v2/missions/drafts/{mission_id}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        ).status_code == 200

        response = client.post(
            f"/api/v2/missions/{mission_id}/messages",
            headers=headers(),
            json={
                "text": "改为去厨房门口，不要进入厨房",
                "kind": "modify_goal",
                "requestedBy": "operator-test",
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["decision"]["kind"] == "replan"
        assert payload["mission"]["snapshot"]["status"] == "awaiting_review"
        pending = payload["decision"]["revision"]
        assert pending["disposition"] == "human_required"

        approved = client.post(
            f"/api/v2/missions/{mission_id}/revisions/{pending['revisionId']}/approve",
            headers=headers(),
            json={"approvedBy": "operator-test"},
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["snapshot"]["status"] == "ready"


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
            json={
                "input": {
                    "mode": "object_approach",
                    "instruction": "找到银白色机器人并计算旁边的安全目标点",
                    "targetDescription": "银白色机器人",
                }
            },
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

        issued = client.post(
            f"/api/v2/missions/{draft_id}/nodes/navigate-kitchen/skill-requests",
            headers=headers(),
            json={
                "input": {
                    "payloadType": "visual_resolution",
                    "goalPose": {
                        "frameId": "map",
                        "x": 1.5,
                        "y": 0.25,
                        "z": 0,
                        "yaw": 0.1,
                    },
                    "goalPurpose": "final",
                    "visualResolutionId": "visual-resolution-1",
                }
            },
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
