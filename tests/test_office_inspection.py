from __future__ import annotations

from datetime import timedelta
import math

import pytest

from packages.control_core.capabilities import RiskLevel
from packages.control_core.validation import utc_now
from packages.mission_core import MissionEnvelope, MissionNode, MissionPlanDraft
from services.control_api.office_inspection import (
    MISSION_REPORT_SKILL,
    OFFICE_INSPECTION_SKILL,
    catalog,
    build_location_report,
    build_report,
    compile_plan,
    envelope_defaults,
    is_full_office_inspection,
    station_index,
)


SHA = "sha256:" + "1" * 64


def brief() -> dict:
    now = utc_now()
    envelope = MissionEnvelope(
        robotId="g1-sim",
        robotBootId="boot-1",
        localizationEpochId="epoch-1",
        capabilityManifestDigest=SHA,
        robotProfileDigest=SHA,
        workstationCapabilityDigest=SHA,
        allowedSkillIds=[
            "navigation.navigate_to_pose",
            OFFICE_INSPECTION_SKILL,
            MISSION_REPORT_SKILL,
        ],
        maximumGoalDistanceM=20,
        maximumCumulativeDistanceM=100,
        maximumSpeedMps=0.4,
        maximumDurationSeconds=7200,
        maximumScanYawRad=40 * math.pi,
        modelPins=[
            {"role": "text_planner", "provider": "test", "model": "test"},
            {"role": "visual_grounder", "provider": "test", "model": "test"},
        ],
        expiresAt=now + timedelta(hours=1),
    )
    return {
        "missionId": "mission-office-tour",
        "robotId": "g1-sim",
        "objective": "走遍所有办公室，并把所有办公室里面的信息和路上的信息都给我",
        "locale": "zh-CN",
        "envelope": envelope.model_dump(mode="json"),
    }


def planner_plan(value: dict) -> MissionPlanDraft:
    envelope = MissionEnvelope.model_validate(value["envelope"])
    return MissionPlanDraft(
        planId=value["missionId"],
        robotId=value["robotId"],
        objective=value["objective"],
        nodes=[
            MissionNode(
                nodeId="planner-placeholder",
                skillId=MISSION_REPORT_SKILL,
                skillVersion="1.0.0",
                purpose="placeholder",
                riskLevel=RiskLevel.LOW,
            )
        ],
        envelope=envelope,
        expiresAt=envelope.expiresAt,
    )


def test_exact_pwa_intent_selects_the_long_office_envelope() -> None:
    objective = "走遍所有办公室，并把所有办公室里面的信息和路上的信息都给我"

    assert is_full_office_inspection(objective) is True
    assert is_full_office_inspection("巡检全部办公室，把看到的信息汇总给我") is True
    assert is_full_office_inspection("遍历每个办公室并告诉我看到了什么") is True
    assert is_full_office_inspection("观察当前房间") is False
    assert envelope_defaults(objective) == {
        "maximumGoalDistanceM": 20.0,
        "maximumCumulativeDistanceM": 100.0,
        "maximumSpeedMps": 0.40,
        "maximumDurationSeconds": 7200,
        "maximumAutoRevisions": 3,
        "maximumStageAttempts": 3,
        "maximumScanYawRad": 40.0 * math.pi,
    }


def test_compiler_covers_every_safe_station_and_finishes_with_report() -> None:
    value = brief()
    source = catalog()
    compiled = compile_plan(planner_plan(value), value, inspection_catalog=source)
    stations = station_index(source)

    assert len(stations) == 20
    assert len(compiled.nodes) == 41
    assert [node.skillId for node in compiled.nodes].count(
        "navigation.navigate_to_pose"
    ) == 20
    assert [node.skillId for node in compiled.nodes].count(
        OFFICE_INSPECTION_SKILL
    ) == 20
    assert compiled.nodes[-1].skillId == MISSION_REPORT_SKILL
    assert compiled.nodes[-1].dependsOn == [compiled.nodes[-2].nodeId]
    assert {
        node.intent.get("stationId")
        for node in compiled.nodes
        if node.skillId == OFFICE_INSPECTION_SKILL
    } == set(stations)
    for index, node in enumerate(compiled.nodes[:-1]):
        if index == 0:
            assert node.dependsOn == []
        else:
            assert node.dependsOn == [compiled.nodes[index - 1].nodeId]


def test_compiler_builds_a_strict_suffix_continuation_with_report_lineage() -> None:
    value = brief()
    source = catalog()
    route = source["route"]
    value["officeInspectionContinuation"] = {
        "completedStationIds": route[:16],
        "sourceMissionIds": ["mission-office-source"],
    }

    compiled = compile_plan(None, value, inspection_catalog=source)

    assert len(compiled.nodes) == 9
    assert compiled.nodes[0].nodeId == "office-nav-17-storage_entry"
    assert compiled.nodes[1].intent["stationId"] == "storage_entry"
    assert compiled.nodes[-2].intent["stationId"] == "corridor_east"
    assert compiled.nodes[-1].intent["sourceMissionIds"] == [
        "mission-office-source"
    ]

    value["officeInspectionContinuation"]["completedStationIds"] = [
        route[0],
        route[2],
    ]
    with pytest.raises(ValueError, match="strict route prefix"):
        compile_plan(None, value, inspection_catalog=source)


def test_navigation_nodes_bind_scene_world_records_not_coordinates() -> None:
    value = brief()
    compiled = compile_plan(planner_plan(value), value)
    navigation = [
        node for node in compiled.nodes if node.skillId == "navigation.navigate_to_pose"
    ]

    assert navigation
    for node in navigation:
        dumped = node.model_dump(mode="json")
        text = str(dumped)
        assert "goalPose" not in text
        assert "waypoints" not in text
        assert node.intent["targetRecordId"].startswith(
            "world-mission-office-tour-scene-target-"
        )


def test_compiler_rejects_catalog_object_answers_and_route_gaps() -> None:
    value = brief()
    unsafe = catalog()
    unsafe["containsObjectAnswers"] = True
    with pytest.raises(ValueError, match="product boundary"):
        compile_plan(planner_plan(value), value, inspection_catalog=unsafe)

    incomplete = catalog()
    incomplete["route"] = incomplete["route"][:-1]
    with pytest.raises(ValueError, match="cover every station"):
        compile_plan(planner_plan(value), value, inspection_catalog=incomplete)


def test_non_office_plan_is_unchanged() -> None:
    value = brief()
    value["objective"] = "去前台"
    original = planner_plan(value)

    assert compile_plan(original, value) is original


def test_report_is_evidence_bound_and_marks_missing_stations() -> None:
    source = catalog()
    first = source["route"][0]
    first_station = station_index(source)[first]
    headings = [
        {
            "headingRad": heading,
            "visualResult": {
                "schemaVersion": "h-ide.visual-result/2",
                "status": "succeeded",
                "mode": "scene_description",
                "analysisId": f"analysis-{index}",
                "observationSummary": "看到清晰门牌和可通行走廊",
                "hypotheses": [],
                "warnings": [],
                "evidence": [
                    {
                        "sha256": "a" * 64,
                        "mediaType": "image/jpeg",
                    }
                ],
            },
        }
        for index, heading in enumerate(first_station["scanHeadingsRad"], start=1)
    ]
    report = build_report(
        {
            "missionId": "mission-office-tour",
            "skillExecutions": [
                {
                    "request": {
                        "skillId": OFFICE_INSPECTION_SKILL,
                    },
                    "result": {
                        "status": "succeeded",
                        "result": {
                            "inspectionResult": {
                                "schemaVersion": "h-ide.office-inspection/1",
                                "stationId": first,
                                "regionId": first_station["regionId"],
                                "headings": headings,
                            }
                        },
                    },
                }
            ],
        },
        inspection_catalog=source,
    )

    assert report["status"] == "completed_with_gaps"
    assert report["fullChainAccepted"] is False
    assert first not in report["missingStations"]
    assert len(report["missingStations"]) == 19
    assert len(report["evidence"]) == len(first_station["scanHeadingsRad"])
    assert report["evidence"][0] == {
        "stationId": first,
        "analysisId": "analysis-1",
        "sha256": "a" * 64,
        "mediaType": "image/jpeg",
    }
    assert "看到清晰门牌" in report["narrativeZhCN"]


def _location_report_mission(*, captured_before_arrival: bool = False) -> dict:
    station = station_index()["meeting_south"]
    verified_at = utc_now()
    captured_at = verified_at + timedelta(
        seconds=-1 if captured_before_arrival else 1
    )
    headings = [
        {
            "headingRad": heading,
            "visualResult": {
                "schemaVersion": "h-ide.visual-result/2",
                "status": "succeeded",
                "mode": "scene_description",
                "analysisId": f"analysis-meeting-{index}",
                "observationSummary": f"会议室方向 {index} 可见桌椅和显示设备",
                "source": {"capturedAt": captured_at.isoformat()},
                "evidence": [
                    {"sha256": f"{index:x}" * 64, "mediaType": "image/jpeg"}
                ],
            },
        }
        for index, heading in enumerate(station["scanHeadingsRad"], start=1)
    ]
    return {
        "missionId": "mission-meeting-observe",
        "skillExecutions": [
            {
                "request": {
                    "skillId": "scene.verify_inside_region",
                },
                "result": {
                    "status": "succeeded",
                    "result": {
                        "insideRegionResult": {
                            "schemaVersion": "h-ide.inside-region/1",
                            "verified": True,
                            "regionId": "meeting_room",
                            "stationId": "meeting_south",
                            "verifiedAt": verified_at.isoformat(),
                        }
                    },
                },
            },
            {
                "request": {
                    "skillId": OFFICE_INSPECTION_SKILL,
                    "input": {"stationId": "meeting_south"},
                },
                "result": {
                    "status": "succeeded",
                    "result": {
                        "inspectionResult": {
                            "schemaVersion": "h-ide.office-inspection/1",
                            "stationId": "meeting_south",
                            "regionId": "meeting_room",
                            "headings": headings,
                        }
                    },
                },
            },
        ],
    }


def test_location_report_requires_inside_verification_and_post_arrival_frames() -> None:
    report = build_location_report(
        _location_report_mission(),
        region_id="meeting_room",
        station_id="meeting_south",
    )

    assert report["reportScope"] == "current_location"
    assert report["locationId"] == "meeting_room"
    assert report["status"] == "completed"
    assert report["fullChainAccepted"] is True
    assert report["missingStations"] == []
    assert len(report["regions"]) == 1
    assert len(report["regions"][0]["summaries"]) == 4
    assert "会议室观察报告" in report["narrativeZhCN"]


def test_location_report_rejects_pre_arrival_visual_frames() -> None:
    with pytest.raises(ValueError, match="pre-arrival"):
        build_location_report(
            _location_report_mission(captured_before_arrival=True),
            region_id="meeting_room",
            station_id="meeting_south",
        )


def test_report_accepts_only_complete_catalog_bound_station_scans() -> None:
    source = catalog()
    stations = station_index(source)
    executions = []
    for station_id in source["route"]:
        station = stations[station_id]
        executions.append(
            {
                "request": {"skillId": OFFICE_INSPECTION_SKILL},
                "result": {
                    "status": "succeeded",
                    "result": {
                        "inspectionResult": {
                            "schemaVersion": "h-ide.office-inspection/1",
                            "stationId": station_id,
                            "regionId": station["regionId"],
                            "headings": [
                                {
                                    "headingRad": heading,
                                    "visualResult": {
                                        "schemaVersion": "h-ide.visual-result/2",
                                        "status": "succeeded",
                                        "mode": "scene_description",
                                        "analysisId": f"analysis-{station_id}-{index}",
                                        "observationSummary": (
                                            f"{station['regionName']} 的可见信息"
                                        ),
                                        "hypotheses": [],
                                        "warnings": [],
                                        "evidence": [
                                            {
                                                "sha256": f"{index:x}" * 64,
                                                "mediaType": "image/jpeg",
                                            }
                                        ],
                                    },
                                }
                                for index, heading in enumerate(
                                    station["scanHeadingsRad"], start=1
                                )
                            ],
                        }
                    },
                },
            }
        )

    report = build_report(
        {"missionId": "mission-office-complete", "skillExecutions": executions},
        inspection_catalog=source,
    )

    assert report["status"] == "completed"
    assert report["fullChainAccepted"] is True
    assert report["missingStations"] == []
    assert len(report["regions"]) == 7

    tampered = executions[0]["result"]["result"]["inspectionResult"]
    tampered["headings"] = tampered["headings"][:-1]
    rejected = build_report(
        {"missionId": "mission-office-incomplete", "skillExecutions": executions},
        inspection_catalog=source,
    )
    assert source["route"][0] in rejected["missingStations"]
    assert rejected["fullChainAccepted"] is False


def test_report_merges_bound_source_and_continuation_scans() -> None:
    source = catalog()
    stations = station_index(source)

    def execution(station_id: str) -> dict:
        station = stations[station_id]
        return {
            "request": {"skillId": OFFICE_INSPECTION_SKILL},
            "result": {
                "status": "succeeded",
                "result": {
                    "inspectionResult": {
                        "schemaVersion": "h-ide.office-inspection/1",
                        "stationId": station_id,
                        "regionId": station["regionId"],
                        "headings": [
                            {
                                "headingRad": heading,
                                "visualResult": {
                                    "schemaVersion": "h-ide.visual-result/2",
                                    "status": "succeeded",
                                    "mode": "scene_description",
                                    "analysisId": f"analysis-{station_id}-{index}",
                                    "observationSummary": f"{station_id} visible",
                                    "hypotheses": [],
                                    "warnings": [],
                                    "evidence": [
                                        {
                                            "sha256": f"{index:x}" * 64,
                                            "mediaType": "image/jpeg",
                                        }
                                    ],
                                },
                            }
                            for index, heading in enumerate(
                                station["scanHeadingsRad"], start=1
                            )
                        ],
                    }
                },
            },
        }

    split = 16
    report = build_report(
        {
            "missionId": "mission-office-continuation",
            "skillExecutions": [
                execution(item) for item in source["route"][split:]
            ],
        },
        inspection_catalog=source,
        source_missions=[
            {
                "missionId": "mission-office-source",
                "skillExecutions": [
                    execution(item) for item in source["route"][:split]
                ],
            }
        ],
    )

    assert report["missionId"] == "mission-office-continuation"
    assert report["status"] == "completed"
    assert report["fullChainAccepted"] is True
    assert report["missingStations"] == []
