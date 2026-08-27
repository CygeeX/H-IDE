from __future__ import annotations

from datetime import timedelta
import math

import pytest

from packages.control_core.validation import utc_now
from packages.control_core.capabilities import RiskLevel
from packages.mission_core import MissionEnvelope, MissionNode, SkillRequest, WorldRecord
from packages.scene_core import PoseMap
from services.control_api.mission_service import MissionInputRequired, MissionService
from services.control_api.office_inspection import (
    MISSION_REPORT_SKILL,
    OFFICE_INSPECTION_SKILL,
    catalog,
    station_index,
)
from services.control_api.office_semantic_navigation import (
    PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M,
    SCENE_VERIFY_INSIDE_SKILL,
    compile_plan,
    envelope_defaults,
    parse_office_place_request,
    parse_place_navigation_intent,
)


SHA = "sha256:" + "2" * 64


def brief(objective: str, mission_id: str = "mission-place") -> dict:
    now = utc_now()
    envelope = MissionEnvelope(
        robotId="g1-sim",
        robotBootId="boot-place",
        localizationEpochId="epoch-place",
        capabilityManifestDigest=SHA,
        robotProfileDigest=SHA,
        workstationCapabilityDigest=SHA,
        allowedSkillIds=[
            "navigation.navigate_to_pose",
            SCENE_VERIFY_INSIDE_SKILL,
            OFFICE_INSPECTION_SKILL,
            MISSION_REPORT_SKILL,
        ],
        maximumGoalDistanceM=20,
        maximumCumulativeDistanceM=50,
        maximumSpeedMps=0.2,
        maximumDurationSeconds=2400,
        maximumScanYawRad=2 * math.pi,
        modelPins=[
            {"role": "text_planner", "provider": "test", "model": "test"},
            {"role": "visual_grounder", "provider": "test", "model": "test"},
        ],
        expiresAt=now + timedelta(hours=1),
    )
    records = [
        {
            "recordId": f"world-{mission_id}-scene-target-{station_id}",
            "kind": "fact",
            "predicate": "target_pose",
        }
        for station_id in station_index()
    ]
    return {
        "missionId": mission_id,
        "robotId": "g1-sim",
        "objective": objective,
        "locale": "zh-CN",
        "envelope": envelope.model_dump(mode="json"),
        "worldRecords": records,
    }


@pytest.mark.parametrize(
    ("text", "task", "location_id"),
    [
        ("去会议室里观察,看到了什么东西告诉我", "observe_report", "meeting_room"),
        ("去会议室里看", "observe_report", "meeting_room"),
        ("去会议室里面站着", "stand", "meeting_room"),
        ("去open office里面站着", "stand", "open_office"),
        ("走到尽头左转进入办公室", "stand", "open_office"),
        ("走出开放办公室", "exit", "corridor"),
    ],
)
def test_original_user_phrases_resolve_to_typed_office_place_tasks(
    text: str,
    task: str,
    location_id: str,
) -> None:
    parsed = parse_office_place_request(text)

    assert parsed is not None
    assert parsed.task == task
    assert parsed.location_id == location_id
    assert parsed.requested_role == (
        "observation_vantage" if task == "observe_report" else "interior"
    )


def test_explicit_current_frame_observation_never_selects_place_navigation() -> None:
    assert parse_office_place_request(
        "只观察当前画面里有什么，不要移动"
    ) is None


def test_exit_open_office_compiles_portal_corridor_and_outside_verification() -> None:
    request = parse_office_place_request("走出开放办公室")
    assert request is not None
    assert request.source_location_id == "open_office"
    plan = compile_plan(
        request,
        brief(request.objective, "mission-exit-open-office"),
        current_pose=PoseMap(
            frameId="map",
            x=-4.5,
            y=2.45,
            z=0.78,
            yaw=0.0,
        ),
    )

    assert [node.skillId for node in plan.nodes] == [
        "navigation.navigate_to_pose",
        "navigation.navigate_to_pose",
        SCENE_VERIFY_INSIDE_SKILL,
    ]
    source = parse_place_navigation_intent(plan.nodes[0].intent)
    destination = parse_place_navigation_intent(plan.nodes[1].intent)
    verification = parse_place_navigation_intent(plan.nodes[2].intent)
    assert source.place.locationId == "open_office"
    assert source.place.stationId == "open_office_entry"
    assert destination.place.locationId == "corridor"
    assert destination.completion.predicate == "inside_region"
    assert destination.completion.maximumAnchorDistanceM == pytest.approx(
        PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M
    )
    assert verification.place.locationId == "corridor"
    assert "target_region" not in str(plan.model_dump(mode="json"))


def test_meeting_room_observation_compiles_entry_inside_verify_scan_report() -> None:
    request = parse_office_place_request("去会议室里观察，看到了什么告诉我")
    assert request is not None
    value = brief(request.objective)

    plan = compile_plan(
        request,
        value,
        current_pose=PoseMap(
            frameId="map",
            x=-6.45,
            y=0.0,
            z=0.78,
            yaw=0.0,
        ),
        inspection_catalog=catalog(),
    )

    assert [node.skillId for node in plan.nodes] == [
        "navigation.navigate_to_pose",
        "navigation.navigate_to_pose",
        SCENE_VERIFY_INSIDE_SKILL,
        OFFICE_INSPECTION_SKILL,
        MISSION_REPORT_SKILL,
    ]
    entry = parse_place_navigation_intent(plan.nodes[0].intent)
    interior = parse_place_navigation_intent(plan.nodes[1].intent)
    verification = parse_place_navigation_intent(plan.nodes[2].intent)
    assert entry.place.stationId == "meeting_entry"
    assert entry.place.anchorRole == "entry"
    assert interior.place.stationId == "meeting_south"
    assert interior.place.anchorRole == "observation_vantage"
    assert interior.place.targetRecordId.endswith("scene-target-meeting_south")
    assert interior.place.candidateTargetRecordIds == [
        "world-mission-place-scene-target-meeting_south",
        "world-mission-place-scene-target-meeting_north",
    ]
    assert verification.completion.predicate == "inside_region"
    assert verification.completion.maximumAnchorDistanceM == pytest.approx(
        PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M
    )
    assert plan.nodes[3].intent["stationId"] == "meeting_south"
    assert plan.nodes[4].intent == {
        "reportScope": "current_location",
        "regionId": "meeting_room",
        "stationId": "meeting_south",
        "locale": "zh-CN",
    }
    serialized = str(plan.model_dump(mode="json"))
    assert "goalPose" not in serialized
    assert "target_region" not in serialized


def test_open_office_stand_compiles_beyond_entry_and_has_no_report() -> None:
    request = parse_office_place_request("去open office里面站着")
    assert request is not None
    value = brief(request.objective, "mission-open-office")

    plan = compile_plan(
        request,
        value,
        current_pose=PoseMap(
            frameId="map",
            x=-6.45,
            y=0.0,
            z=0.78,
            yaw=0.0,
        ),
    )

    assert [node.skillId for node in plan.nodes] == [
        "navigation.navigate_to_pose",
        "navigation.navigate_to_pose",
        SCENE_VERIFY_INSIDE_SKILL,
    ]
    entry = parse_place_navigation_intent(plan.nodes[0].intent)
    interior = parse_place_navigation_intent(plan.nodes[1].intent)
    assert entry.place.stationId == "open_office_entry"
    assert interior.place.stationId == "open_office_west"
    assert interior.completion.maximumAnchorDistanceM == pytest.approx(
        PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M
    )


def test_generic_office_route_phrase_uses_the_reviewed_open_office_template() -> None:
    request = parse_office_place_request("走到尽头左转进入办公室")
    assert request is not None
    assert request.location_id == "open_office"
    assert request.task == "stand"
    plan = compile_plan(
        request,
        brief(request.objective, "mission-generic-office-route"),
        current_pose=PoseMap(
            frameId="map",
            x=-6.45,
            y=0.0,
            z=0.78,
            yaw=0.0,
        ),
    )

    # The relative-language phrase is compiled to the same bounded,
    # SceneRef-bound entry -> interior -> verification DAG as an explicit
    # “去开放办公室里面站着” request.  No planner-generated coordinates or
    # retired target_region fields are accepted.
    assert [node.skillId for node in plan.nodes] == [
        "navigation.navigate_to_pose",
        "navigation.navigate_to_pose",
        SCENE_VERIFY_INSIDE_SKILL,
    ]
    assert parse_place_navigation_intent(plan.nodes[0].intent).place.stationId == (
        "open_office_entry"
    )
    assert parse_place_navigation_intent(plan.nodes[1].intent).place.stationId == (
        "open_office_west"
    )
    assert "target_region" not in str(plan.model_dump(mode="json"))


@pytest.mark.parametrize(
    "text",
    [
        "沿走廊走到尽头左转进入办公室",
        "走到尽头右转进入会议室",
        "一直走然后进入开放办公区",
    ],
)
def test_route_wording_composes_with_published_place_aliases(text: str) -> None:
    request = parse_office_place_request(text)
    assert request is not None
    assert request.location_id in {"open_office", "meeting_room"}


def test_interior_anchor_is_selected_from_entry_not_pre_mission_pose() -> None:
    request = parse_office_place_request("去会议室里看")
    assert request is not None
    plan = compile_plan(
        request,
        brief(request.objective, "mission-entry-relative-anchor"),
        # This pose is nearer meeting_north, but the plan visits
        # meeting_entry first; meeting_south is nearer from that node.
        current_pose=PoseMap(
            frameId="map",
            x=2.6,
            y=4.4,
            z=0.78,
            yaw=0.0,
        ),
    )

    interior = parse_place_navigation_intent(plan.nodes[1].intent)
    assert interior.place.stationId == "meeting_south"
    assert interior.place.candidateTargetRecordIds[1].endswith(
        "scene-target-meeting_north"
    )
    assert interior.completion.predicate == "inside_region"


@pytest.mark.parametrize(
    "text",
    [
        "去前台里面站着",
        "去开放办公区里面站着",
        "去会议室里面站着",
        "去独立办公室里面站着",
        "去茶水间里面站着",
        "去储物间里面站着",
        "去走廊站着",
    ],
)
def test_every_published_office_region_has_a_deterministic_place_task(text: str) -> None:
    request = parse_office_place_request(text)
    assert request is not None
    value = brief(text, f"mission-{request.location_id}")
    plan = compile_plan(request, value, current_pose=None)

    assert plan.nodes
    assert any(node.skillId == SCENE_VERIFY_INSIDE_SKILL for node in plan.nodes)


def test_place_envelope_allows_cross_office_navigation_without_speed_guessing() -> None:
    assert envelope_defaults("去会议室里面站着") == {
        "maximumGoalDistanceM": 20.0,
        "maximumCumulativeDistanceM": 50.0,
        "maximumSpeedMps": 0.2,
        "maximumDurationSeconds": 2400,
        "maximumAutoRevisions": 3,
        "maximumStageAttempts": 3,
        "maximumScanYawRad": 2.0 * math.pi,
    }


def test_compiler_fails_closed_when_selected_scene_anchor_is_missing() -> None:
    request = parse_office_place_request("去会议室里面站着")
    assert request is not None
    value = brief(request.objective)
    value["worldRecords"] = [
        item
        for item in value["worldRecords"]
        if not item["recordId"].endswith("scene-target-meeting_south")
    ]

    with pytest.raises(ValueError, match="not published"):
        compile_plan(request, value, current_pose=None)


def _inside_verification_service(pose: PoseMap) -> tuple[MissionService, SkillRequest]:
    request_id = "skill-inside-meeting-1"
    mission = {
        "missionId": "mission-inside-meeting",
        "snapshot": {"activeSkillRequestId": request_id},
        "worldRecords": [
            {
                "recordId": "world-mission-inside-meeting-scene-meeting_room",
                "kind": "fact",
                "predicate": "semantic_region",
                "source": "scene",
                "value": {
                    "polygonXY": [
                        {"x": -1.14, "y": 1.41},
                        {"x": 3.04, "y": 1.41},
                        {"x": 3.04, "y": 5.82},
                        {"x": -1.14, "y": 5.82},
                    ]
                },
            },
            {
                "recordId": "world-mission-inside-meeting-scene-target-meeting_south",
                "kind": "fact",
                "predicate": "target_pose",
                "source": "scene",
                "value": {
                    "targetPose": {
                        "frameId": "map",
                        "x": 0.7,
                        "y": 2.18,
                        "z": 0.0,
                        "yaw": math.pi / 2,
                    }
                },
            },
        ],
    }

    class Runtime:
        def get(self, mission_id: str):
            assert mission_id == mission["missionId"]
            return mission

    service = object.__new__(MissionService)
    service.runtime = Runtime()
    service._current_robot_pose = lambda robot_id: pose  # type: ignore[method-assign]
    issued_at = utc_now()
    request = SkillRequest(
        requestId=request_id,
        missionId=mission["missionId"],
        revisionId="revision-inside-1",
        nodeId="verify-inside",
        robotId="g1-sim",
        skillId=SCENE_VERIFY_INSIDE_SKILL,
        skillVersion="1.0.0",
        input={
            "regionId": "meeting_room",
            "regionRecordId": "world-mission-inside-meeting-scene-meeting_room",
            "stationId": "meeting_south",
            "targetRecordId": "world-mission-inside-meeting-scene-target-meeting_south",
            "anchorRole": "interior",
            "minimumClearanceM": 0.25,
            "maximumAnchorDistanceM": 0.25,
        },
        capabilityManifestDigest=SHA,
        authorizationId="authorization-inside-1",
        idempotencyKey=request_id,
        issuedAt=issued_at,
        expiresAt=issued_at + timedelta(seconds=10),
    )
    return service, request


def test_inside_region_verifier_requires_polygon_and_selected_interior_anchor() -> None:
    inside_service, request = _inside_verification_service(
        PoseMap(
            frameId="map",
            x=0.7,
            y=2.18,
            z=0.78,
            yaw=math.pi / 2,
            observedAt=utc_now(),
        )
    )
    verified = inside_service.verify_inside_region(request)["insideRegionResult"]
    assert verified["inside"] is True
    assert verified["verified"] is True
    assert verified["measuredClearanceM"] >= 0.25
    assert verified["measuredAnchorDistanceM"] == pytest.approx(0.0)

    entry_service, entry_request = _inside_verification_service(
        PoseMap(
            frameId="map",
            x=-0.6,
            y=1.95,
            z=0.78,
            yaw=math.pi / 2,
            observedAt=utc_now(),
        )
    )
    rejected = entry_service.verify_inside_region(entry_request)["insideRegionResult"]
    assert rejected["inside"] is True
    assert rejected["verified"] is False
    assert rejected["measuredAnchorDistanceM"] > 0.25


def _target_world_record(record_id: str, station_id: str) -> dict:
    return WorldRecord(
        recordId=record_id,
        missionId="mission-place-retry",
        robotId="g1-sim",
        kind="fact",
        subject=station_id,
        predicate="target_pose",
        value={
            "targetPose": {
                "frameId": "map",
                "x": 0.7,
                "y": 2.18 if station_id == "meeting_south" else 4.7,
                "z": 0.0,
                "yaw": math.pi / 2,
            }
        },
        source="scene",
        sourceRef=f"scene:office:1:{station_id}:map-v1",
        confidence=1.0,
        observedAt=utc_now(),
        frameId="map",
        sceneRef={"sceneId": "office", "revision": 1},
    ).model_dump(mode="json")


def test_place_target_selection_fails_closed_and_uses_only_reviewed_fallbacks() -> None:
    target_ids = [
        "world-mission-place-retry-scene-target-meeting_south",
        "world-mission-place-retry-scene-target-meeting_north",
    ]
    intent = {
        "schemaVersion": "h-ide.place-intent/1",
        "task": "navigate_to_place",
        "place": {
            "schemaVersion": "h-ide.place-target/1",
            "locationId": "meeting_room",
            "locationName": "会议室",
            "anchorRole": "interior",
            "stationId": "meeting_south",
            "targetRecordId": target_ids[0],
            "candidateTargetRecordIds": target_ids,
        },
        "completion": {
            "schemaVersion": "h-ide.place-completion/1",
            "predicate": "inside_region",
            "regionId": "meeting_room",
            "minimumClearanceM": 0.25,
            "maximumAnchorDistanceM": 0.25,
        },
    }
    node = MissionNode(
        nodeId="navigate-meeting",
        skillId="navigation.navigate_to_pose",
        skillVersion="2.0.0",
        purpose="进入会议室",
        intent=intent,
        riskLevel=RiskLevel.MEDIUM,
    )
    record = {
        "worldRecords": [
            _target_world_record(target_ids[0], "meeting_south"),
            _target_world_record(target_ids[1], "meeting_north"),
        ],
        "skillExecutions": [],
    }
    service = object.__new__(MissionService)
    assert service._trusted_target_fact(record, node).recordId == target_ids[0]

    record["skillExecutions"] = [{
        "nodeId": node.nodeId,
        "result": {
            "status": "failed",
            "errorCode": "ROBOT_AGENT_FAILED",
            "result": {
                "commandReceipt": {
                    "status": "failed",
                    "error": "action finished with status aborted",
                }
            },
        },
    }]
    assert service._trusted_target_fact(record, node).recordId == target_ids[0]

    # One fresh-odometry retry stays on the primary anchor.  A second abort
    # advances to the reviewed alternate, which remains selected thereafter.
    record["skillExecutions"].append(dict(record["skillExecutions"][0]))
    assert service._trusted_target_fact(record, node).recordId == target_ids[1]

    single_candidate_intent = {
        **intent,
        "place": {
            **intent["place"],
            "candidateTargetRecordIds": [target_ids[0]],
        },
    }
    single_candidate_node = node.model_copy(
        update={"nodeId": "navigate-meeting-entry", "intent": single_candidate_intent}
    )
    single_candidate_record = {
        "worldRecords": [_target_world_record(target_ids[0], "meeting_south")],
        "skillExecutions": [
            {
                "nodeId": single_candidate_node.nodeId,
                "result": {
                    "status": "failed",
                    "errorCode": "ROBOT_AGENT_FAILED",
                    "result": {
                        "commandReceipt": {
                            "status": "failed",
                            "error": "action finished with status aborted",
                        }
                    },
                },
            }
        ],
    }
    assert (
        service._trusted_target_fact(single_candidate_record, single_candidate_node).recordId
        == target_ids[0]
    )

    unbound = MissionNode(
        nodeId="unbound",
        skillId="navigation.navigate_to_pose",
        skillVersion="2.0.0",
        purpose="没有目标的导航",
        intent={},
        riskLevel=RiskLevel.MEDIUM,
    )
    with pytest.raises(MissionInputRequired) as exc:
        service._trusted_target_fact(record, unbound)
    assert exc.value.code == "TRUSTED_TARGET_SELECTOR_REQUIRED"


def test_single_reviewed_place_anchor_remains_retryable_after_one_abort() -> None:
    request = parse_office_place_request("去open office里面站着")
    assert request is not None
    plan = compile_plan(
        request,
        brief(request.objective, "mission-single-anchor-retry"),
        current_pose=PoseMap(frameId="map", x=0.6, y=2.1, z=0.78, yaw=0.0),
    )
    node = plan.nodes[0]
    now = utc_now()
    revision_id = "revision-single-anchor-retry-1"
    request_id = "skill-single-anchor-retry-1"
    record = {
        "missionId": plan.planId,
        "plan": plan.model_dump(mode="json"),
        "activeRevision": {
            "revisionId": revision_id,
            "missionId": plan.planId,
            "sequence": 1,
            "objective": plan.objective,
            "nodes": [item.model_dump(mode="json") for item in plan.nodes],
            "reasonCode": "initial_plan",
            "reason": "reviewed",
            "changeSummary": ["initial reviewed plan"],
            "disposition": "human_approved",
            "authorizationId": "auth-single-anchor-retry",
            "createdBy": "operator",
            "createdAt": now.isoformat(),
        },
        "authorization": {
            "authorizationId": "auth-single-anchor-retry",
            "missionId": plan.planId,
            "planId": plan.planId,
            "envelopeSha256": SHA,
            "approvedBy": "operator",
            "approvedAt": now.isoformat(),
            "expiresAt": plan.expiresAt.isoformat(),
        },
        "snapshot": {
            "activeNodeId": node.nodeId,
            "activeSkillRequestId": None,
        },
        "nodeStates": [
            {
                "revisionId": revision_id,
                "nodeId": node.nodeId,
                "status": "failed",
                "attempt": 1,
                "state": {"skillRequestId": request_id},
            }
        ],
        "skillExecutions": [
            {
                "requestId": request_id,
                "nodeId": node.nodeId,
                "request": {
                    "missionId": plan.planId,
                    "revisionId": revision_id,
                    "nodeId": node.nodeId,
                    "skillId": "navigation.navigate_to_pose",
                },
                "result": {
                    "requestId": request_id,
                    "status": "failed",
                    "errorCode": "ROBOT_AGENT_FAILED",
                    "result": {
                        "commandReceipt": {
                            "status": "failed",
                            "error": "action finished with status ABORTED",
                        }
                    },
                },
            }
        ],
    }

    assert MissionService._place_navigation_retry_execution(record) is not None


def test_near_goal_nav_abort_is_accepted_only_with_fresh_scene_geometry() -> None:
    parsed = parse_office_place_request("去会议室里面站着")
    assert parsed is not None
    plan = compile_plan(
        parsed,
        brief(parsed.objective, "mission-semantic-arrival"),
        current_pose=PoseMap(frameId="map", x=-0.6, y=1.95, z=0.78, yaw=0.0),
    )
    node = plan.nodes[1]
    typed = parse_place_navigation_intent(node.intent)
    target_record_id = typed.place.targetRecordId
    source_ref = "scene:office:1:meeting_south:map-v1"
    now = utc_now()

    class FreshRobotStore:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

        def get_robot(self, robot_id: str) -> dict:
            assert robot_id == "g1-sim"
            return {
                "robotId": robot_id,
                "connected": True,
                "status": "online",
                "safetyMode": "normal",
                "pose": {
                    "frameId": "map",
                    "x": self.x,
                    "y": self.y,
                    "z": 0.78,
                    "yaw": 0.0,
                    "observedAt": now.isoformat(),
                },
                "details": {"adapter": "fixture"},
            }

    service = object.__new__(MissionService)
    service.store = FreshRobotStore(0.84, 2.18)
    service.robot_refresher = None
    revision_id = "revision-semantic-arrival-1"
    record = {
        "missionId": plan.planId,
        "activeRevision": {
            "revisionId": revision_id,
            "missionId": plan.planId,
            "sequence": 1,
            "objective": plan.objective,
            "nodes": [item.model_dump(mode="json") for item in plan.nodes],
            "reasonCode": "initial_plan",
            "reason": "reviewed",
            "changeSummary": ["initial reviewed plan"],
            "disposition": "human_approved",
            "authorizationId": "auth-semantic-arrival",
            "createdBy": "operator",
            "createdAt": now.isoformat(),
        },
        "worldRecords": [
            {
                "recordId": target_record_id,
                "kind": "fact",
                "predicate": "target_pose",
                "source": "scene",
                "sourceRef": source_ref,
                "value": {
                    "locationId": "meeting_south",
                    "targetPose": {
                        "frameId": "map",
                        "x": 0.7,
                        "y": 2.18,
                        "z": 0.0,
                        "yaw": math.pi / 2,
                    },
                },
            },
            {
                "recordId": f"world-{plan.planId}-scene-meeting_room",
                "kind": "fact",
                "predicate": "semantic_region",
                "source": "scene",
                "value": {
                    "polygonXY": [
                        {"x": 0.0, "y": 1.5},
                        {"x": 3.0, "y": 1.5},
                        {"x": 3.0, "y": 5.5},
                        {"x": 0.0, "y": 5.5},
                    ]
                },
            },
        ],
    }
    skill = SkillRequest(
        requestId="skill-semantic-arrival",
        missionId=plan.planId,
        revisionId=revision_id,
        nodeId=node.nodeId,
        robotId="g1-sim",
        skillId=node.skillId,
        skillVersion=node.skillVersion,
        input={
            "payloadType": "spatial_resolution",
            "spatialResolutionId": source_ref,
            "goalPose": {
                "frameId": "map",
                "x": 0.7,
                "y": 2.18,
                "z": 0.0,
                "yaw": math.pi / 2,
            },
        },
        capabilityManifestDigest=SHA,
        authorizationId="auth-semantic-arrival",
        idempotencyKey="skill-semantic-arrival",
        issuedAt=now,
        expiresAt=now + timedelta(minutes=10),
    )
    receipt = {
        "status": "failed",
        "error": "action finished with status ABORTED",
    }

    accepted = service._semantic_place_arrival_after_nav_abort(
        record,
        skill,
        receipt,
    )
    assert accepted is not None
    assert accepted["accepted"] is True
    assert accepted["inside"] is True
    assert accepted["stationId"] == "meeting_south"
    assert accepted["measuredAnchorDistanceM"] == pytest.approx(0.14)

    service.store = FreshRobotStore(1.1, 2.18)
    assert (
        service._semantic_place_arrival_after_nav_abort(record, skill, receipt)
        is None
    )
