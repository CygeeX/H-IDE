from __future__ import annotations

from datetime import timedelta
import math

from packages.control_core.validation import utc_now
from packages.mission_core import MissionEnvelope
from packages.scene_core import PoseMap
from services.control_api.object_search_report import build_object_search_report
from services.control_api.object_semantic_navigation import (
    compile_plan,
    envelope_defaults,
    parse_object_semantic_request,
)
from services.control_api.office_inspection import MISSION_REPORT_SKILL, catalog, station_index
from services.control_api.office_semantic_navigation import SCENE_VERIFY_INSIDE_SKILL


SHA = "sha256:" + "3" * 64


def semantic_referent(label: str, alias: str, region_id: str) -> dict:
    return {
        "schemaVersion": "h-ide.semantic-referent/1",
        "referentId": f"referent-{label}-1",
        "sourceTaskId": "task-observe-1",
        "sourceMissionId": "mission-observe-1",
        "analysisId": f"analysis-{label}-1",
        "hypothesisId": f"hypothesis-{label}-1",
        "label": label,
        "aliases": [label, alias],
        "regionId": region_id,
        "capturedAt": "2026-08-27T00:00:00Z",
        "robotBootId": "boot-object-1",
        "localizationEpochId": "epoch-object-1",
        "authority": "SEMANTIC_ONLY",
        "executionEligible": False,
    }


def brief(objective: str, mission_id: str, referents: list[dict]) -> dict:
    now = utc_now()
    defaults = envelope_defaults(objective, semantic_referents=referents)
    envelope = MissionEnvelope(
        robotId="g1-sim",
        robotBootId="boot-object-1",
        localizationEpochId="epoch-object-1",
        capabilityManifestDigest=SHA,
        robotProfileDigest=SHA,
        workstationCapabilityDigest=SHA,
        allowedSkillIds=[
            "navigation.navigate_to_pose",
            "navigation.rotate",
            "vision.observe",
            SCENE_VERIFY_INSIDE_SKILL,
            MISSION_REPORT_SKILL,
        ],
        maximumGoalDistanceM=defaults["maximumGoalDistanceM"],
        maximumCumulativeDistanceM=defaults["maximumCumulativeDistanceM"],
        maximumSpeedMps=defaults["maximumSpeedMps"],
        maximumDurationSeconds=defaults["maximumDurationSeconds"],
        maximumAutoRevisions=defaults["maximumAutoRevisions"],
        maximumStageAttempts=defaults["maximumStageAttempts"],
        maximumScanYawRad=defaults["maximumScanYawRad"],
        modelPins=[
            {"role": "text_planner", "provider": "test", "model": "test"},
            {"role": "visual_grounder", "provider": "test", "model": "test"},
        ],
        expiresAt=now + timedelta(minutes=8),
    )
    return {
        "missionId": mission_id,
        "robotId": "g1-sim",
        "objective": objective,
        "locale": "zh-CN",
        "envelope": envelope.model_dump(mode="json"),
        "semanticReferents": referents,
        "worldRecords": [
            {
                "recordId": f"world-{mission_id}-scene-target-{station_id}",
                "kind": "fact",
                "predicate": "target_pose",
            }
            for station_id in station_index()
        ],
    }


def test_object_approach_uses_referent_and_promises_runtime_target_pose() -> None:
    referents = [semantic_referent("chair", "椅子", "open_office")]
    request = parse_object_semantic_request(
        "去椅子旁边站着",
        semantic_referents=referents,
    )
    assert request is not None
    assert request.task == "approach"
    assert request.semantic_referent_id == "referent-chair-1"
    plan = compile_plan(
        request,
        brief(request.objective, "mission-chair-approach", referents),
        current_pose=PoseMap(
            frameId="map", x=-4.5, y=2.4, z=0.78, yaw=0.0
        ),
        inspection_catalog=catalog(),
    )

    search_nodes = [
        node
        for node in plan.nodes
        if node.skillId == "vision.observe"
        and node.intent.get("searchOptional") is True
    ]
    assert len(search_nodes) >= 4
    assert all(node.intent["visualTaskMode"] == "object_approach" for node in search_nodes)
    assert all(node.intent["producesPredicate"] == "target_pose" for node in search_nodes)
    approach = next(node for node in plan.nodes if node.nodeId == "object-approach-nav")
    assert approach.intent["semanticReferentId"] == "referent-chair-1"
    assert "targetRecordId" not in approach.intent
    assert set(approach.dependsOn) == {node.nodeId for node in search_nodes}
    assert any(node.skillId == "navigation.rotate" for node in plan.nodes)
    alignment = next(node for node in plan.nodes if node.nodeId == "object-approach-align")
    assert alignment.skillId == "navigation.rotate"
    assert alignment.intent["targetYawFromExecutedGoal"] is True
    assert alignment.dependsOn == ["object-approach-nav"]
    assert plan.nodes[-1].nodeId == "object-approach-verify"
    assert plan.nodes[-1].dependsOn == ["object-approach-align"]


def test_active_computer_search_is_motion_capable_and_ends_with_report() -> None:
    request = parse_object_semantic_request("观察一下办公室找到电脑")
    assert request is not None
    assert request.task == "search"
    plan = compile_plan(
        request,
        brief(request.objective, "mission-computer-search", []),
        current_pose=PoseMap(
            frameId="map", x=-4.5, y=2.4, z=0.78, yaw=0.0
        ),
        inspection_catalog=catalog(),
    )

    assert any(node.skillId == "navigation.rotate" for node in plan.nodes)
    assert plan.nodes[-1].skillId == MISSION_REPORT_SKILL
    assert plan.nodes[-1].intent["reportScope"] == "object_search"
    assert plan.envelope.maximumScanYawRad == 4 * math.pi


def test_object_search_report_distinguishes_found_from_not_found() -> None:
    miss = build_object_search_report(
        {"missionId": "mission-search", "skillExecutions": []},
        target_description="computer",
        region_id="open_office",
    )
    assert miss["found"] is False
    assert "没有找到" in miss["narrativeZhCN"]

    found = build_object_search_report(
        {
            "missionId": "mission-search",
            "skillExecutions": [
                {
                    "request": {"skillId": "vision.observe"},
                    "result": {
                        "result": {
                            "visualResult": {
                                "analysisId": "analysis-computer",
                                "status": "succeeded",
                                "source": {"capturedAt": "2026-08-27T00:00:00Z"},
                                "geometryCandidates": [
                                    {
                                        "geometryCandidateId": "geometry-computer",
                                        "role": "approach",
                                        "verified": True,
                                        "targetPose": {"frameId": "map"},
                                    }
                                ],
                                "evidence": [],
                            }
                        }
                    },
                }
            ],
        },
        target_description="computer",
        region_id="open_office",
    )
    assert found["found"] is True
    assert "找到" in found["narrativeZhCN"]


def test_daaam_memory_is_only_a_region_search_prior() -> None:
    request = parse_object_semantic_request(
        "去椅子旁边站着",
        semantic_priors=[
            {
                "source": "daaam",
                "subject": "chair",
                "predicate": "semantic_memory_observation",
                "value": {
                    "regionId": "open_office",
                    "authority": "observation_only",
                    "executionEligible": False,
                },
            }
        ],
    )

    assert request is not None
    assert request.region_id == "open_office"
    assert request.semantic_referent_id is None
