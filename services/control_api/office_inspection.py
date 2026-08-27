"""Deterministic product compiler for the Office V2 inspection intent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import math
import re
from typing import Any

from packages.control_core.capabilities import RiskLevel
from packages.control_core.validation import utc_now
from packages.mission_core import MissionEnvelope, MissionNode, MissionPlanDraft
from services.scene_api.office_v2_bootstrap import load_bundle


OFFICE_INSPECTION_SKILL = "inspection.scan_station"
MISSION_REPORT_SKILL = "mission.report"
OFFICE_INSPECTION_MAXIMUM_GOAL_DISTANCE_M = 20.0
OFFICE_INSPECTION_MAXIMUM_CUMULATIVE_DISTANCE_M = 100.0
OFFICE_INSPECTION_MAXIMUM_DURATION_SECONDS = 7_200
OFFICE_INSPECTION_MAXIMUM_SCAN_YAW_RAD = 40.0 * math.pi


def is_full_office_inspection(objective: object) -> bool:
    if not isinstance(objective, str):
        return False
    normalized = re.sub(r"[\s，。！？、,.!?;；：:]+", "", objective).casefold()
    return all(
        token in normalized
        for token in ("走遍", "所有办公室", "信息", "路上")
    )


def envelope_defaults(objective: object) -> dict[str, Any]:
    if not is_full_office_inspection(objective):
        return {}
    return {
        "maximumGoalDistanceM": OFFICE_INSPECTION_MAXIMUM_GOAL_DISTANCE_M,
        "maximumCumulativeDistanceM": (
            OFFICE_INSPECTION_MAXIMUM_CUMULATIVE_DISTANCE_M
        ),
        "maximumSpeedMps": 0.40,
        "maximumDurationSeconds": OFFICE_INSPECTION_MAXIMUM_DURATION_SECONDS,
        "maximumAutoRevisions": 3,
        "maximumStageAttempts": 3,
        "maximumScanYawRad": OFFICE_INSPECTION_MAXIMUM_SCAN_YAW_RAD,
    }


def catalog() -> dict[str, Any]:
    return dict(load_bundle()["catalog"])


def station_index(value: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    source = catalog() if value is None else value
    result: dict[str, dict[str, Any]] = {}
    regions = source.get("regions")
    if not isinstance(regions, list):
        raise ValueError("Office inspection catalog has no regions")
    for region in regions:
        if not isinstance(region, Mapping):
            raise ValueError("Office inspection region is invalid")
        region_id = str(region.get("regionId", "")).strip()
        for station in region.get("stations", []):
            if not isinstance(station, Mapping):
                raise ValueError("Office inspection station is invalid")
            station_id = str(station.get("stationId", "")).strip()
            if not region_id or not station_id or station_id in result:
                raise ValueError("Office inspection station identity is invalid")
            result[station_id] = {
                **dict(station),
                "regionId": region_id,
                "regionName": str(region.get("name", region_id)),
            }
    return result


def compile_plan(
    planner_plan: MissionPlanDraft | None,
    brief: Mapping[str, Any],
    *,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> MissionPlanDraft:
    """Replace planner geometry with the reviewed, SceneRef-bound itinerary."""

    if not is_full_office_inspection(brief.get("objective")):
        if planner_plan is None:
            raise ValueError("a planner plan is required for non-inspection intent")
        return planner_plan
    source = catalog() if inspection_catalog is None else dict(inspection_catalog)
    if source.get("containsObjectAnswers") is not False:
        raise ValueError("Office inspection catalog violates the product boundary")
    route = source.get("route")
    if not isinstance(route, list) or not route:
        raise ValueError("Office inspection route is empty")
    stations = station_index(source)
    catalog_route_ids = [str(item).strip() for item in route]
    if any(not item for item in catalog_route_ids) or len(catalog_route_ids) != len(set(catalog_route_ids)):
        raise ValueError("Office inspection route identities are invalid")
    if set(catalog_route_ids) != set(stations):
        raise ValueError("Office inspection route does not cover every station")

    continuation = brief.get("officeInspectionContinuation")
    completed_station_ids: list[str] = []
    source_mission_ids: list[str] = []
    if continuation is not None:
        if not isinstance(continuation, Mapping) or set(continuation) != {
            "completedStationIds",
            "sourceMissionIds",
        }:
            raise ValueError("Office inspection continuation is invalid")
        raw_completed = continuation.get("completedStationIds")
        raw_sources = continuation.get("sourceMissionIds")
        if (
            not isinstance(raw_completed, list)
            or not isinstance(raw_sources, list)
            or not raw_sources
            or len(raw_sources) > 4
        ):
            raise ValueError("Office inspection continuation identities are invalid")
        completed_station_ids = [str(item).strip() for item in raw_completed]
        source_mission_ids = [str(item).strip() for item in raw_sources]
        if (
            any(not item for item in completed_station_ids)
            or any(not item for item in source_mission_ids)
            or len(completed_station_ids) != len(set(completed_station_ids))
            or len(source_mission_ids) != len(set(source_mission_ids))
            or completed_station_ids
            != catalog_route_ids[: len(completed_station_ids)]
            or len(completed_station_ids) >= len(catalog_route_ids)
        ):
            raise ValueError("Office inspection continuation must preserve a strict route prefix")

    route_ids = catalog_route_ids[len(completed_station_ids) :]

    envelope = MissionEnvelope.model_validate(brief["envelope"])
    required_skills = {
        "navigation.navigate_to_pose",
        OFFICE_INSPECTION_SKILL,
        MISSION_REPORT_SKILL,
    }
    missing = required_skills.difference(envelope.allowedSkillIds)
    if missing:
        raise ValueError(f"Office inspection capability is missing: {sorted(missing)[0]}")

    nodes: list[MissionNode] = []
    previous: str | None = None
    mission_id = str(brief["missionId"])
    for station_id in route_ids:
        sequence = catalog_route_ids.index(station_id) + 1
        station = stations[station_id]
        navigate_id = f"office-nav-{sequence:02d}-{station_id}"
        scan_id = f"office-scan-{sequence:02d}-{station_id}"
        nodes.append(
            MissionNode(
                nodeId=navigate_id,
                skillId="navigation.navigate_to_pose",
                skillVersion="2.0.0",
                purpose=f"前往巡检站 {station_id}",
                intent={
                    "targetRecordId": (
                        f"world-{mission_id}-scene-target-{station_id}"
                    ),
                    "targetDescription": station_id,
                    "regionId": station["regionId"],
                },
                dependsOn=[] if previous is None else [previous],
                riskLevel=RiskLevel.MEDIUM,
            )
        )
        nodes.append(
            MissionNode(
                nodeId=scan_id,
                skillId=OFFICE_INSPECTION_SKILL,
                skillVersion="1.0.0",
                purpose=f"停稳并扫描 {station_id}",
                intent={
                    "stationId": station_id,
                    "regionId": station["regionId"],
                    "regionName": station["regionName"],
                    "scanPurpose": station.get("purpose", "inspection"),
                },
                dependsOn=[navigate_id],
                riskLevel=RiskLevel.MEDIUM,
            )
        )
        previous = scan_id
    assert previous is not None
    nodes.append(
        MissionNode(
            nodeId="office-final-report",
            skillId=MISSION_REPORT_SKILL,
            skillVersion="1.0.0",
            purpose="汇总六个办公室与沿途观察信息",
            intent={
                "reportScope": "all_office_regions_and_corridor",
                "locale": str(brief.get("locale", "zh-CN")),
                **(
                    {"sourceMissionIds": source_mission_ids}
                    if source_mission_ids
                    else {}
                ),
            },
            dependsOn=[previous],
            riskLevel=RiskLevel.LOW,
        )
    )
    created = utc_now()
    return MissionPlanDraft(
        planId=mission_id,
        robotId=str(brief["robotId"]),
        objective=str(brief["objective"]),
        nodes=nodes,
        envelope=envelope,
        createdAt=created,
        expiresAt=envelope.expiresAt,
    )


def build_report(
    mission: Mapping[str, Any],
    *,
    inspection_catalog: Mapping[str, Any] | None = None,
    source_missions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate only immutable scan outputs; never consult object truth."""

    source = catalog() if inspection_catalog is None else dict(inspection_catalog)
    stations = station_index(source)
    observations: dict[str, dict[str, Any]] = {}
    for mission_record in [*(source_missions or ()), mission]:
        for execution in mission_record.get("skillExecutions", []):
            if not isinstance(execution, Mapping):
                continue
            request = execution.get("request")
            terminal = execution.get("result")
            if (
                not isinstance(request, Mapping)
                or request.get("skillId") != OFFICE_INSPECTION_SKILL
                or not isinstance(terminal, Mapping)
                or terminal.get("status") != "succeeded"
            ):
                continue
            result = terminal.get("result")
            scan = result.get("inspectionResult") if isinstance(result, Mapping) else None
            station_id = scan.get("stationId") if isinstance(scan, Mapping) else None
            if not isinstance(station_id, str) or station_id not in stations:
                continue
            expected_station = stations[station_id]
            raw_headings = scan.get("headings")
            expected_headings = expected_station.get("scanHeadingsRad")
            if (
                scan.get("schemaVersion") != "h-ide.office-inspection/1"
                or scan.get("regionId") != expected_station["regionId"]
                or not isinstance(raw_headings, list)
                or not isinstance(expected_headings, list)
                or len(raw_headings) != len(expected_headings)
            ):
                continue
            observed_headings: list[float] = []
            valid = True
            for heading in raw_headings:
                if not isinstance(heading, Mapping):
                    valid = False
                    break
                raw_yaw = heading.get("headingRad")
                visual = heading.get("visualResult")
                if (
                    isinstance(raw_yaw, bool)
                    or not isinstance(raw_yaw, (int, float))
                    or not math.isfinite(float(raw_yaw))
                    or not isinstance(visual, Mapping)
                    or visual.get("schemaVersion") != "h-ide.visual-result/2"
                    or visual.get("status") != "succeeded"
                    or visual.get("mode") != "scene_description"
                    or not isinstance(visual.get("observationSummary"), str)
                    or not str(visual["observationSummary"]).strip()
                ):
                    valid = False
                    break
                observed_headings.append(float(raw_yaw))
            if not valid or any(
                abs(actual - float(expected)) > 1e-9
                for actual, expected in zip(observed_headings, expected_headings, strict=True)
            ):
                continue
            observations[station_id] = dict(scan)

    regions: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    missing_stations: list[str] = []
    for region in source["regions"]:
        station_reports: list[dict[str, Any]] = []
        summaries: list[str] = []
        for station in region["stations"]:
            station_id = station["stationId"]
            scan = observations.get(station_id)
            if scan is None:
                missing_stations.append(station_id)
                station_reports.append(
                    {"stationId": station_id, "status": "missing", "headings": []}
                )
                continue
            heading_reports: list[dict[str, Any]] = []
            for heading in scan.get("headings", []):
                if not isinstance(heading, Mapping):
                    continue
                visual = heading.get("visualResult")
                if not isinstance(visual, Mapping):
                    continue
                summary = str(visual.get("observationSummary") or "").strip()
                if summary:
                    summaries.append(summary)
                analysis_id = visual.get("analysisId")
                bound = []
                for item in visual.get("evidence", []):
                    if not isinstance(item, Mapping):
                        continue
                    sha = item.get("sha256")
                    if isinstance(sha, str):
                        bound.append(sha)
                        evidence.append(
                            {
                                "stationId": station_id,
                                "analysisId": analysis_id,
                                "sha256": sha,
                                "mediaType": item.get("mediaType"),
                            }
                        )
                heading_reports.append(
                    {
                        "headingRad": heading.get("headingRad"),
                        "analysisId": analysis_id,
                        "summary": summary or None,
                        "hypotheses": list(visual.get("hypotheses", [])),
                        "warnings": list(visual.get("warnings", [])),
                        "evidenceSha256": bound,
                    }
                )
            station_reports.append(
                {
                    "stationId": station_id,
                    "status": "observed" if heading_reports else "missing",
                    "headings": heading_reports,
                }
            )
            if not heading_reports:
                missing_stations.append(station_id)
        regions.append(
            {
                "regionId": region["regionId"],
                "name": region["name"],
                "kind": region["kind"],
                "status": (
                    "observed"
                    if all(item["status"] == "observed" for item in station_reports)
                    else "partial"
                ),
                "summaries": summaries,
                "stations": station_reports,
            }
        )
    missing_stations = list(dict.fromkeys(missing_stations))
    status = "completed" if not missing_stations else "completed_with_gaps"
    lines = ["Office V2 巡检报告"]
    for region in regions:
        summaries = region["summaries"]
        lines.append(
            f"- {region['name']}："
            + ("；".join(summaries) if summaries else "未取得可靠视觉信息")
        )
    if missing_stations:
        lines.append("- 信息缺口：" + "、".join(missing_stations))
    return {
        "schemaVersion": "h-ide.office-report/1",
        "missionId": mission.get("missionId") or mission.get("snapshot", {}).get("missionId"),
        "status": status,
        "fullChainAccepted": status == "completed",
        "regions": regions,
        "missingStations": missing_stations,
        "evidence": evidence,
        "narrativeZhCN": "\n".join(lines),
    }


def build_location_report(
    mission: Mapping[str, Any],
    *,
    region_id: str,
    station_id: str,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one evidence-complete report from post-arrival fresh frames."""

    source = catalog() if inspection_catalog is None else dict(inspection_catalog)
    stations = station_index(source)
    station = stations.get(station_id)
    if station is None or station.get("regionId") != region_id:
        raise ValueError("location report station is not bound to the requested region")
    region = next(
        (
            item
            for item in source.get("regions", [])
            if isinstance(item, Mapping) and item.get("regionId") == region_id
        ),
        None,
    )
    if not isinstance(region, Mapping):
        raise ValueError("location report region is not in the Office V2 catalog")

    verified_at = None
    for execution in mission.get("skillExecutions", []):
        if not isinstance(execution, Mapping):
            continue
        request = execution.get("request")
        terminal = execution.get("result")
        if (
            not isinstance(request, Mapping)
            or request.get("skillId") != "scene.verify_inside_region"
            or not isinstance(terminal, Mapping)
            or terminal.get("status") != "succeeded"
        ):
            continue
        result = terminal.get("result")
        inside = (
            result.get("insideRegionResult")
            if isinstance(result, Mapping)
            else None
        )
        if (
            not isinstance(inside, Mapping)
            or inside.get("schemaVersion") != "h-ide.inside-region/1"
            or inside.get("verified") is not True
            or inside.get("regionId") != region_id
            or inside.get("stationId") != station_id
        ):
            continue
        raw_verified_at = inside.get("verifiedAt")
        if not isinstance(raw_verified_at, str):
            continue
        try:
            candidate = datetime.fromisoformat(raw_verified_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if verified_at is None or candidate > verified_at:
            verified_at = candidate
    if verified_at is None:
        raise ValueError("location report has no successful inside-region verification")

    scan = None
    for execution in mission.get("skillExecutions", []):
        if not isinstance(execution, Mapping):
            continue
        request = execution.get("request")
        terminal = execution.get("result")
        if (
            not isinstance(request, Mapping)
            or request.get("skillId") != OFFICE_INSPECTION_SKILL
            or request.get("input", {}).get("stationId") != station_id
            or not isinstance(terminal, Mapping)
            or terminal.get("status") != "succeeded"
        ):
            continue
        result = terminal.get("result")
        candidate_scan = (
            result.get("inspectionResult")
            if isinstance(result, Mapping)
            else None
        )
        if isinstance(candidate_scan, Mapping):
            scan = candidate_scan
    if not isinstance(scan, Mapping):
        raise ValueError("location report has no completed station observation")

    expected_headings = station.get("scanHeadingsRad")
    raw_headings = scan.get("headings")
    if (
        not isinstance(expected_headings, list)
        or not expected_headings
        or not isinstance(raw_headings, list)
        or len(raw_headings) != len(expected_headings)
    ):
        raise ValueError("location report observation headings are incomplete")
    summaries: list[str] = []
    evidence: list[dict[str, Any]] = []
    for expected_yaw, heading in zip(expected_headings, raw_headings, strict=True):
        if not isinstance(heading, Mapping):
            raise ValueError("location report heading is invalid")
        raw_yaw = heading.get("headingRad")
        visual = heading.get("visualResult")
        if (
            isinstance(raw_yaw, bool)
            or not isinstance(raw_yaw, (int, float))
            or abs(float(raw_yaw) - float(expected_yaw)) > 1e-9
            or not isinstance(visual, Mapping)
            or visual.get("schemaVersion") != "h-ide.visual-result/2"
            or visual.get("status") != "succeeded"
            or visual.get("mode") != "scene_description"
        ):
            raise ValueError("location report contains an invalid visual heading")
        source_payload = visual.get("source")
        captured_at = (
            source_payload.get("capturedAt")
            if isinstance(source_payload, Mapping)
            else None
        )
        if not isinstance(captured_at, str):
            raise ValueError("location report visual frame has no capture time")
        try:
            captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("location report visual capture time is invalid") from exc
        if captured <= verified_at:
            raise ValueError("location report attempted to reuse a pre-arrival frame")
        summary = str(visual.get("observationSummary") or "").strip()
        if not summary:
            raise ValueError("location report visual heading has no observation summary")
        summaries.append(summary)
        analysis_id = visual.get("analysisId")
        for item in visual.get("evidence", []):
            if not isinstance(item, Mapping) or not isinstance(item.get("sha256"), str):
                continue
            evidence.append(
                {
                    "stationId": station_id,
                    "analysisId": analysis_id,
                    "sha256": item["sha256"],
                    "mediaType": item.get("mediaType"),
                    "capturedAt": captured_at,
                }
            )
    name = str(region.get("name") or region_id)
    return {
        "schemaVersion": "h-ide.office-report/1",
        "missionId": mission.get("missionId")
        or mission.get("snapshot", {}).get("missionId"),
        "reportScope": "current_location",
        "locationId": region_id,
        "stationId": station_id,
        "insideVerifiedAt": verified_at.isoformat(),
        "status": "completed",
        "fullChainAccepted": True,
        "regions": [
            {
                "regionId": region_id,
                "name": name,
                "kind": str(region.get("kind") or "room"),
                "status": "observed",
                "summaries": summaries,
                "stations": [
                    {
                        "stationId": station_id,
                        "status": "observed",
                        "headings": list(raw_headings),
                    }
                ],
            }
        ],
        "missingStations": [],
        "evidence": evidence,
        "narrativeZhCN": f"{name}观察报告\n- " + "；".join(summaries),
    }


__all__ = [
    "MISSION_REPORT_SKILL",
    "OFFICE_INSPECTION_SKILL",
    "catalog",
    "build_report",
    "build_location_report",
    "compile_plan",
    "envelope_defaults",
    "is_full_office_inspection",
    "station_index",
]
