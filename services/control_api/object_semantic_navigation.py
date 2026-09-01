"""Deterministic Office object-approach and active-search Mission compiler."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal, Mapping

from packages.control_core.capabilities import RiskLevel
from packages.control_core.validation import utc_now
from packages.mission_core import MissionEnvelope, MissionNode, MissionPlanDraft
from packages.scene_core import PoseMap

from .office_inspection import (
    MISSION_REPORT_SKILL,
    catalog as office_catalog,
    station_index as office_station_index,
)
from .office_semantic_navigation import (
    OfficePlaceRequest,
    PlaceCompletion,
    PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M,
    PlaceNavigationIntent,
    PlaceTarget,
    compile_plan as compile_place_plan,
    parse_office_place_request,
    parse_place_navigation_intent,
)
from .semantic_referents import resolve_semantic_object


@dataclass(frozen=True)
class ObjectSemanticRequest:
    objective: str
    task: Literal["approach", "search"]
    target_label: str
    semantic_referent_id: str | None
    region_id: str | None


_APPROACH_PATTERN = re.compile(
    r"(?:去|走到|移动到|前往|靠近).{0,40}(?:旁边|前面|跟前|附近|边上)",
    re.IGNORECASE,
)
_SEARCH_PATTERN = re.compile(
    r"(?:寻找|找到|搜索|搜寻|环顾|巡视|全面观察|"
    r"\b(?:find|search|look\s+for)\b)",
    re.IGNORECASE,
)


def parse_object_semantic_request(
    objective: object,
    *,
    semantic_referents: list[Mapping[str, Any]] | None = None,
    semantic_priors: list[Mapping[str, Any]] | None = None,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> ObjectSemanticRequest | None:
    if not isinstance(objective, str) or not objective.strip():
        return None
    text = objective.strip()
    task: Literal["approach", "search"] | None = None
    if _APPROACH_PATTERN.search(text):
        task = "approach"
    elif _SEARCH_PATTERN.search(text):
        task = "search"
    if task is None:
        return None
    resolved = resolve_semantic_object(text, semantic_referents or [])
    if resolved is None:
        return None
    target_label, referent_id, referent_region = resolved
    prior_region: str | None = None
    if referent_region is None:
        for prior in semantic_priors or []:
            if (
                prior.get("source") != "daaam"
                or prior.get("predicate") != "semantic_memory_observation"
                or not isinstance(prior.get("subject"), str)
                or not isinstance(prior.get("value"), Mapping)
            ):
                continue
            prior_target = resolve_semantic_object(str(prior["subject"]))
            region = prior["value"].get("regionId")
            if (
                prior_target is not None
                and prior_target[0] == target_label
                and isinstance(region, str)
                and region
            ):
                prior_region = region
                break
    source = office_catalog() if inspection_catalog is None else inspection_catalog
    # Reuse the reviewed place alias parser by adding a non-executable language
    # prefix. It returns only a semantic region identity; no pose crosses this
    # parsing boundary.
    place = parse_office_place_request(
        f"去 {text}",
        inspection_catalog=source,
    )
    explicit_region = (
        place.location_id
        if place is not None and place.location_id != "corridor"
        else None
    )
    return ObjectSemanticRequest(
        objective=text,
        task=task,
        target_label=target_label,
        semantic_referent_id=referent_id,
        region_id=explicit_region or referent_region or prior_region,
    )


def envelope_defaults(
    objective: object,
    *,
    semantic_referents: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if parse_object_semantic_request(
        objective,
        semantic_referents=semantic_referents,
    ) is None:
        return {}
    return {
        "maximumGoalDistanceM": 20.0,
        "maximumCumulativeDistanceM": 30.0,
        "maximumSpeedMps": 0.20,
        "maximumDurationSeconds": 480,
        "maximumAutoRevisions": 1,
        "maximumStageAttempts": 3,
        "maximumScanYawRad": 4.0 * math.pi,
    }


def _point_in_polygon(x: float, y: float, polygon: list[Mapping[str, Any]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        current_x, current_y = float(current["x"]), float(current["y"])
        previous_x, previous_y = float(previous["x"]), float(previous["y"])
        if (current_y > y) != (previous_y > y):
            crossing_x = (
                (previous_x - current_x)
                * (y - current_y)
                / (previous_y - current_y)
                + current_x
            )
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _current_region_id(
    current_pose: PoseMap | None,
    inspection_catalog: Mapping[str, Any],
) -> str | None:
    if current_pose is None:
        return None
    matches: list[tuple[float, str]] = []
    for region in inspection_catalog.get("regions", []):
        if not isinstance(region, Mapping):
            continue
        region_id = region.get("regionId")
        polygon = region.get("polygonXY")
        if (
            not isinstance(region_id, str)
            or not isinstance(polygon, list)
            or len(polygon) < 3
            or any(not isinstance(item, Mapping) for item in polygon)
        ):
            continue
        try:
            if _point_in_polygon(current_pose.x, current_pose.y, polygon):
                area = abs(
                    sum(
                        float(left["x"]) * float(right["y"])
                        - float(right["x"]) * float(left["y"])
                        for left, right in zip(polygon, [*polygon[1:], polygon[0]])
                    )
                ) / 2.0
                matches.append((area, region_id))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    matches.sort()
    return matches[0][1] if matches else None


def _search_visual_node(
    *,
    node_id: str,
    purpose: str,
    target: str,
    referent_id: str | None,
    group_id: str,
    depends_on: list[str],
) -> MissionNode:
    return MissionNode(
        nodeId=node_id,
        skillId="vision.observe",
        skillVersion="2.0.0",
        purpose=purpose,
        intent={
            "visualTaskMode": "object_approach",
            "task": "search_and_ground",
            "targetDescription": target,
            "targetCategory": target,
            "producesPredicate": "target_pose",
            "searchOptional": True,
            "searchGroupId": group_id,
            **(
                {"semanticReferentId": referent_id}
                if referent_id is not None
                else {}
            ),
        },
        dependsOn=depends_on,
        riskLevel=RiskLevel.LOW,
    )


def _append_search_views(
    nodes: list[MissionNode],
    *,
    previous: str | None,
    target: str,
    referent_id: str | None,
    group_id: str,
    vantage_index: int,
) -> tuple[list[str], str]:
    visual_ids: list[str] = []
    current_previous = previous
    for view_index in range(1, 5):
        if view_index > 1:
            rotate_id = (
                f"object-search-{group_id}-p{vantage_index:02d}-"
                f"rotate-{view_index - 1:02d}"
            )
            nodes.append(
                MissionNode(
                    nodeId=rotate_id,
                    skillId="navigation.rotate",
                    skillVersion="2.0.0",
                    purpose=f"转向下一个方向继续寻找 {target}",
                    intent={
                        "relativeYawRad": math.pi / 2.0,
                        "searchGroupId": group_id,
                    },
                    dependsOn=[] if current_previous is None else [current_previous],
                    riskLevel=RiskLevel.MEDIUM,
                )
            )
            current_previous = rotate_id
        visual_id = (
            f"object-search-{group_id}-p{vantage_index:02d}-"
            f"view-{view_index:02d}"
        )
        nodes.append(
            _search_visual_node(
                node_id=visual_id,
                purpose=f"观察当前方向并寻找 {target}",
                target=target,
                referent_id=referent_id,
                group_id=group_id,
                depends_on=[] if current_previous is None else [current_previous],
            )
        )
        visual_ids.append(visual_id)
        current_previous = visual_id
    return visual_ids, str(current_previous)


def _secondary_vantage_node(
    *,
    request: ObjectSemanticRequest,
    mission_id: str,
    station: Mapping[str, Any],
    depends_on: str,
    group_id: str,
) -> MissionNode:
    station_id = str(station["stationId"])
    target_record_id = f"world-{mission_id}-scene-target-{station_id}"
    place = PlaceTarget(
        locationId=str(station["regionId"]),
        locationName=str(station.get("regionName") or station["regionId"]),
        anchorRole="observation_vantage",
        stationId=station_id,
        targetRecordId=target_record_id,
        candidateTargetRecordIds=[target_record_id],
    )
    return MissionNode(
        nodeId=f"object-search-{group_id}-vantage-02-nav",
        skillId="navigation.navigate_to_pose",
        skillVersion="2.0.0",
        purpose=f"前往同一区域的第二个观察点继续寻找 {request.target_label}",
        intent=PlaceNavigationIntent(
            task="navigate_to_place",
            place=place,
            completion=PlaceCompletion(
                predicate="inside_region",
                regionId=str(station["regionId"]),
                minimumClearanceM=0.25,
                maximumAnchorDistanceM=PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M,
            ),
        ).model_dump(mode="json"),
        dependsOn=[depends_on],
        riskLevel=RiskLevel.MEDIUM,
    )


def compile_plan(
    request: ObjectSemanticRequest,
    brief: Mapping[str, Any],
    *,
    current_pose: PoseMap | None,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> MissionPlanDraft:
    catalog = office_catalog() if inspection_catalog is None else dict(inspection_catalog)
    envelope = MissionEnvelope.model_validate(brief["envelope"])
    required = {"vision.observe", "navigation.rotate"}
    if request.task == "approach":
        required.add("navigation.navigate_to_pose")
    else:
        required.add(MISSION_REPORT_SKILL)
    missing = required.difference(envelope.allowedSkillIds)
    if missing:
        raise ValueError(
            f"Object semantic capability is missing: {sorted(missing)[0]}"
        )
    mission_id = str(brief["missionId"])
    region_id = request.region_id or _current_region_id(current_pose, catalog)
    nodes: list[MissionNode] = []
    previous: str | None = None
    primary_station_id: str | None = None
    if (
        region_id is not None
        and region_id != "corridor"
        and envelope.sceneRefs
    ):
        region = next(
            (
                item
                for item in catalog.get("regions", [])
                if isinstance(item, Mapping) and item.get("regionId") == region_id
            ),
            None,
        )
        if isinstance(region, Mapping):
            place_request = OfficePlaceRequest(
                objective=request.objective,
                location_id=region_id,
                location_name=str(region.get("name") or region_id),
                task="stand",
                requested_role="interior",
            )
            place_plan = compile_place_plan(
                place_request,
                brief,
                current_pose=current_pose,
                inspection_catalog=catalog,
            )
            nodes.extend(place_plan.nodes)
            previous = nodes[-1].nodeId
            navigation = [
                item
                for item in place_plan.nodes
                if item.skillId == "navigation.navigate_to_pose"
            ]
            if navigation:
                primary_station_id = parse_place_navigation_intent(
                    navigation[-1].intent
                ).place.stationId

    group_id = re.sub(r"[^a-z0-9]+", "-", request.target_label.casefold()).strip("-")
    group_id = (group_id or "target")[:40]
    all_visual_ids: list[str] = []
    first_views, previous = _append_search_views(
        nodes,
        previous=previous,
        target=request.target_label,
        referent_id=request.semantic_referent_id,
        group_id=group_id,
        vantage_index=1,
    )
    all_visual_ids.extend(first_views)

    stations = [
        item
        for item in office_station_index(catalog).values()
        if region_id is not None
        and item.get("regionId") == region_id
        and str(item.get("purpose") or "").casefold() == "interior"
        and item.get("stationId") != primary_station_id
    ]
    if stations:
        secondary = stations[0]
        secondary_record = (
            f"world-{mission_id}-scene-target-{secondary['stationId']}"
        )
        world_ids = {
            str(item.get("recordId"))
            for item in brief.get("worldRecords", [])
            if isinstance(item, Mapping)
        }
        if secondary_record in world_ids:
            secondary_node = _secondary_vantage_node(
                request=request,
                mission_id=mission_id,
                station=secondary,
                depends_on=previous,
                group_id=group_id,
            )
            nodes.append(secondary_node)
            second_views, previous = _append_search_views(
                nodes,
                previous=secondary_node.nodeId,
                target=request.target_label,
                referent_id=request.semantic_referent_id,
                group_id=group_id,
                vantage_index=2,
            )
            all_visual_ids.extend(second_views)

    if request.task == "approach":
        navigation_id = "object-approach-nav"
        nodes.append(
            MissionNode(
                nodeId=navigation_id,
                skillId="navigation.navigate_to_pose",
                skillVersion="2.0.0",
                purpose=f"通过 Nav2 前往 {request.target_label} 的安全接近点",
                intent={
                    "targetDescription": request.target_label,
                    "searchTargetDescription": request.target_label,
                    "requiresPredicate": "target_pose",
                    **(
                        {"semanticReferentId": request.semantic_referent_id}
                        if request.semantic_referent_id is not None
                        else {}
                    ),
                },
                dependsOn=all_visual_ids,
                riskLevel=RiskLevel.MEDIUM,
            )
        )
        # Nav2's learned whole-body execution can satisfy the XY endpoint
        # while retaining the path heading (occasionally the opposite
        # heading from the visual target). Make the final facing direction a
        # reviewed, closed-loop primitive instead of relying on the planner's
        # terminal orientation. MissionService resolves the relative delta
        # from the fresh Robot Agent pose and the just-executed goal; it never
        # accepts a browser-supplied yaw or a direct torque command.
        alignment_id = "object-approach-align"
        nodes.append(
            MissionNode(
                nodeId=alignment_id,
                skillId="navigation.rotate",
                skillVersion="2.0.0",
                purpose=f"到达后调整朝向面对 {request.target_label}",
                intent={
                    "targetYawFromExecutedGoal": True,
                    "targetDescription": request.target_label,
                    "requiresPredicate": "target_pose",
                    **(
                        {"semanticReferentId": request.semantic_referent_id}
                        if request.semantic_referent_id is not None
                        else {}
                    ),
                },
                dependsOn=[navigation_id],
                riskLevel=RiskLevel.MEDIUM,
            )
        )
        nodes.append(
            MissionNode(
                nodeId="object-approach-verify",
                skillId="vision.observe",
                skillVersion="2.0.0",
                purpose=f"到达后重新观察并确认 {request.target_label}",
                intent={
                    "visualTaskMode": "arrival_verification",
                    "task": "arrival_verification",
                    "targetDescription": request.target_label,
                    "verificationMaximumDistanceM": 0.15,
                    "verificationMaximumYawRad": 0.35,
                },
                dependsOn=[alignment_id],
                riskLevel=RiskLevel.LOW,
            )
        )
    else:
        nodes.append(
            MissionNode(
                nodeId="object-result-report",
                skillId=MISSION_REPORT_SKILL,
                skillVersion="1.0.0",
                purpose=f"汇总对 {request.target_label} 的主动搜索结果",
                intent={
                    "reportScope": "object_search",
                    "targetDescription": request.target_label,
                    "regionId": region_id,
                    "locale": str(brief.get("locale", "zh-CN")),
                },
                dependsOn=all_visual_ids,
                riskLevel=RiskLevel.LOW,
            )
        )
    created = utc_now()
    return MissionPlanDraft(
        planId=mission_id,
        robotId=str(brief["robotId"]),
        objective=request.objective,
        nodes=nodes,
        envelope=envelope,
        createdAt=created,
        expiresAt=envelope.expiresAt,
    )


__all__ = [
    "ObjectSemanticRequest",
    "compile_plan",
    "envelope_defaults",
    "parse_object_semantic_request",
]
