"""Typed, deterministic Office V2 place-task compiler.

The text planner may recognize a broad objective, but a published Office V2
place is not an open-ended language target.  This module maps known aliases to
the immutable product catalog and emits exact Scene WorldRecord identities for
every navigation node before a draft can be reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from packages.control_core.capabilities import RiskLevel
from packages.control_core.validation import utc_now
from packages.mission_core import MissionEnvelope, MissionNode, MissionPlanDraft
from packages.scene_core import PoseMap

from .office_inspection import (
    MISSION_REPORT_SKILL,
    OFFICE_INSPECTION_SKILL,
    catalog as office_catalog,
    station_index as office_station_index,
)


SCENE_VERIFY_INSIDE_SKILL = "scene.verify_inside_region"
PLACE_INTENT_SCHEMA_VERSION = "h-ide.place-intent/1"
PLACE_COMPLETION_SCHEMA_VERSION = "h-ide.place-completion/1"
PLACE_MINIMUM_CLEARANCE_M = 0.25
PLACE_MAXIMUM_ANCHOR_DISTANCE_M = 0.25
# Learned biped terminal settling can leave a small residual after Nav2
# reports success. Keep entry anchors tight, but permit an interior
# verification within a bounded 0.35 m envelope; polygon and clearance gates
# remain mandatory.
PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M = 0.35


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlaceTarget(_StrictModel):
    schemaVersion: Literal["h-ide.place-target/1"] = "h-ide.place-target/1"
    locationId: str = Field(min_length=1, max_length=200)
    locationName: str = Field(min_length=1, max_length=500)
    anchorRole: Literal["entry", "interior", "observation_vantage"]
    stationId: str = Field(min_length=1, max_length=200)
    targetRecordId: str = Field(min_length=1, max_length=500)
    candidateTargetRecordIds: list[str] = Field(min_length=1, max_length=20)

    @field_validator(
        "locationId",
        "locationName",
        "stationId",
        "targetRecordId",
    )
    @classmethod
    def _trimmed(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("place identity must be trimmed")
        return value

    @model_validator(mode="after")
    def _candidate_binding(self):
        if len(self.candidateTargetRecordIds) != len(
            set(self.candidateTargetRecordIds)
        ):
            raise ValueError("place candidate target records must be unique")
        if self.candidateTargetRecordIds[0] != self.targetRecordId:
            raise ValueError("selected place target must be the first candidate")
        return self


class PlaceCompletion(_StrictModel):
    schemaVersion: Literal["h-ide.place-completion/1"] = (
        PLACE_COMPLETION_SCHEMA_VERSION
    )
    predicate: Literal["at_anchor", "inside_region"]
    regionId: str = Field(min_length=1, max_length=200)
    minimumClearanceM: float = Field(ge=0.0, le=2.0)
    maximumAnchorDistanceM: float = Field(gt=0.0, le=2.0)


class PlaceNavigationIntent(_StrictModel):
    schemaVersion: Literal["h-ide.place-intent/1"] = PLACE_INTENT_SCHEMA_VERSION
    task: Literal["navigate_to_place", "verify_inside_region"]
    place: PlaceTarget
    completion: PlaceCompletion

    @model_validator(mode="after")
    def _roles_match_completion(self):
        if (
            self.completion.predicate == "inside_region"
            and self.place.anchorRole not in {"interior", "observation_vantage"}
        ):
            raise ValueError("inside-region completion requires an interior anchor")
        if self.completion.regionId != self.place.locationId:
            raise ValueError("place completion region does not match the target")
        return self


@dataclass(frozen=True)
class OfficePlaceRequest:
    objective: str
    location_id: str
    location_name: str
    task: Literal["stand", "observe_report", "exit"]
    requested_role: Literal["entry", "interior", "observation_vantage"]
    source_location_id: str | None = None


@dataclass(frozen=True)
class OfficeInspectionSequenceRequest:
    """An ordered, bounded list of Office V2 rooms to inspect.

    The request intentionally carries region identities only.  Stations and
    all executable navigation inputs are resolved later from the immutable
    Office V2 catalog by :func:`compile_inspection_sequence_plan`.
    """

    objective: str
    places: tuple[OfficePlaceRequest, ...]


_MOTION_PATTERN = re.compile(
    r"(?:去往|前往|进入|进去|走进|走到|走去|移动到|导航到|驶入|回到|返回到|靠近|到达|"
    r"巡检|巡查|检查|查看|观察|去|"
    r"\b(?:go|walk|move|navigate|enter|approach|return)\b)",
    re.IGNORECASE,
)
_EXIT_PATTERN = re.compile(
    r"(?:走出|离开|出去|退出|撤出|\b(?:leave|exit|walk\s+out\s+of)\b)",
    re.IGNORECASE,
)
_OBSERVE_PATTERN = re.compile(
    r"(?:观察|看看|看一看|去看|看到了|看见|看|有什么|告诉我|汇报|描述|"
    r"\b(?:observe|inspect|look|describe|report|what\s+is|what\s+can)\b)",
    re.IGNORECASE,
)
_ENTRY_PATTERN = re.compile(r"(?:门口|入口|\b(?:door|doorway|entrance|entry)\b)", re.IGNORECASE)
_STATIONARY_CURRENT_PATTERN = re.compile(
    r"(?:(?:只|仅)(?:观察|查看|看).{0,16}(?:当前|现在|眼前)|原地(?:观察|查看|看)|"
    r"(?:current\s+(?:view|scene)|without\s+moving|do\s+not\s+move))",
    re.IGNORECASE,
)
_INSPECTION_ACTION_PATTERN = re.compile(
    r"(?:巡检|巡查|检查|巡视|inspect(?:ion)?|patrol)",
    re.IGNORECASE,
)
_SEQUENCE_MARKER_PATTERN = re.compile(
    r"(?:依次|逐一|分别|按(?:房间|顺序)|先.{0,32}(?:再|然后|接着|随后))",
    re.IGNORECASE,
)
_MULTI_REGION_CONNECTOR_PATTERN = re.compile(
    r"(?:和|及|以及|、|,|，|然后|接着|随后)",
    re.IGNORECASE,
)


def _normalize_label(value: str) -> str:
    return re.sub(r"[\s_\-，。！？、,.!?;；：:()（）]+", "", value).casefold()


def _region_aliases(region: Mapping[str, Any]) -> set[str]:
    values = {
        str(region.get("regionId") or ""),
        str(region.get("name") or ""),
        *(str(item) for item in region.get("aliases", [])),
    }
    region_id = str(region.get("regionId") or "")
    supplemental = {
        "reception": {"前台", "接待区", "前台接待区"},
        # ``办公室`` is intentionally a vocabulary alias, not a sentence
        # template.  In Office V2 the only region whose product name is a
        # generic office is the open-office area; callers that mean another
        # room can use its explicit label (会议室/独立办公室/etc.).  Keeping
        # this mapping in the scene vocabulary means phrases such as
        # ``走到尽头左转进入办公室`` and ``去办公室里面站着`` share the same
        # typed place compiler instead of falling through to an unconstrained
        # LLM-generated Mission.
        "open_office": {"办公室", "开放办公区", "开放办公室", "open office", "office"},
        "meeting_room": {"会议室", "meeting room"},
        "focus_office": {"独立办公室", "独立办公区", "focus office"},
        "pantry": {"茶水间", "茶歇区", "pantry"},
        "storage": {"储物间", "仓储间", "storage"},
        "corridor": {"走廊", "过道", "corridor"},
    }
    values.update(supplemental.get(region_id, set()))
    return {normalized for item in values if (normalized := _normalize_label(item))}


def _ordered_region_matches(
    text: str,
    source: Mapping[str, Any],
) -> list[tuple[int, int, Mapping[str, Any]]]:
    """Return non-overlapping catalog region mentions in user-specified order."""

    normalized = _normalize_label(text)
    candidates: list[tuple[int, int, str, Mapping[str, Any]]] = []
    for region in source.get("regions", []):
        if not isinstance(region, Mapping):
            continue
        region_id = str(region.get("regionId") or "").strip()
        if not region_id:
            continue
        for alias in _region_aliases(region):
            if not alias:
                continue
            offset = normalized.find(alias)
            while offset >= 0:
                # Sort longer aliases first at the same offset.  This keeps
                # the supplemental generic alias "办公室" from shadowing
                # "开放办公室" or "独立办公室".
                candidates.append((offset, len(alias), region_id, region))
                offset = normalized.find(alias, offset + 1)
    candidates.sort(key=lambda item: (item[0], -item[1], item[2]))

    accepted: list[tuple[int, int, Mapping[str, Any]]] = []
    seen_regions: set[str] = set()
    occupied: list[tuple[int, int]] = []
    for offset, length, region_id, region in candidates:
        if region_id in seen_regions:
            continue
        end = offset + length
        if any(offset < right and end > left for left, right in occupied):
            continue
        accepted.append((offset, length, region))
        seen_regions.add(region_id)
        occupied.append((offset, end))
    accepted.sort(key=lambda item: item[0])
    return accepted


def parse_office_inspection_sequence_request(
    objective: object,
    *,
    inspection_catalog: Mapping[str, Any] | None = None,
    default_location_id: str | None = None,
    inspection_mode: bool = False,
) -> OfficeInspectionSequenceRequest | None:
    """Parse one or more requested rooms as a complete inspection workflow.

    Both concise commands such as ``请依次巡检开放办公区和会议室`` and
    explicit commands that spell out entering, settling, observing and
    reporting resolve to the same typed request.  A single-room command with an
    explicit inspection verb is intentionally included: ``巡检会议室`` means
    navigate, verify, scan and report, never merely navigate.  When the user has
    explicitly selected the inspection Skill, ``default_location_id`` may bind
    an ellipsis such as ``再观察一下`` to the robot's freshly measured current
    room; conversational memory never supplies coordinates.
    """

    if not isinstance(objective, str) or not objective.strip():
        return None
    text = objective.strip()
    if _EXIT_PATTERN.search(text) or _STATIONARY_CURRENT_PATTERN.search(text):
        return None
    has_sequence = _SEQUENCE_MARKER_PATTERN.search(text) is not None
    has_inspection = (
        _INSPECTION_ACTION_PATTERN.search(text) is not None or inspection_mode
    )
    has_motion = _MOTION_PATTERN.search(text) is not None
    has_observation = _OBSERVE_PATTERN.search(text) is not None
    # The long form may use "依次进入 ... 观察 ... 汇总" instead of the word
    # "巡检".  A short form without an order marker still needs an explicit
    # inspection verb and a room connector.
    if not has_motion or not (has_inspection or has_observation):
        return None

    source = office_catalog() if inspection_catalog is None else inspection_catalog
    matches = _ordered_region_matches(text, source)
    if not matches and inspection_mode and isinstance(default_location_id, str):
        fallback = next(
            (
                region
                for region in source.get("regions", [])
                if isinstance(region, Mapping)
                and str(region.get("regionId") or "").strip()
                == default_location_id.strip()
                and str(region.get("regionId") or "").strip() != "corridor"
            ),
            None,
        )
        if fallback is not None:
            matches = [(0, 0, fallback)]
    if len(matches) < 1 or len(matches) > 8:
        return None
    if len(matches) > 1 and not has_sequence and not (
        has_inspection and _MULTI_REGION_CONNECTOR_PATTERN.search(text)
    ):
        return None
    if len(matches) == 1 and not has_inspection:
        return None
    places: list[OfficePlaceRequest] = []
    for _offset, _length, region in matches:
        region_id = str(region.get("regionId") or "").strip()
        region_name = str(region.get("name") or region_id).strip()
        if not region_id or not region_name:
            return None
        places.append(
            OfficePlaceRequest(
                objective=text,
                location_id=region_id,
                location_name=region_name,
                task="observe_report",
                requested_role="observation_vantage",
            )
        )
    return OfficeInspectionSequenceRequest(objective=text, places=tuple(places))


def parse_office_place_request(
    objective: object,
    *,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> OfficePlaceRequest | None:
    if not isinstance(objective, str) or not objective.strip():
        return None
    text = objective.strip()
    # Do not silently downgrade an ordered multi-room inspection to the first
    # room mentioned in the sentence.  MissionService handles the sequence
    # through its dedicated compiler below.
    if parse_office_inspection_sequence_request(
        text,
        inspection_catalog=inspection_catalog,
    ) is not None:
        return None
    exiting = _EXIT_PATTERN.search(text) is not None
    if _STATIONARY_CURRENT_PATTERN.search(text) or (
        not exiting and not _MOTION_PATTERN.search(text)
    ):
        return None
    normalized = _normalize_label(text)
    source = office_catalog() if inspection_catalog is None else inspection_catalog
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for region in source.get("regions", []):
        if not isinstance(region, Mapping):
            continue
        lengths = [
            len(alias)
            for alias in _region_aliases(region)
            if alias and alias in normalized
        ]
        if lengths:
            matches.append((max(lengths), region))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    best_length = matches[0][0]
    best_regions = {
        str(region.get("regionId"))
        for length, region in matches
        if length == best_length
    }
    if len(best_regions) != 1:
        return None
    region = next(
        region for length, region in matches
        if length == best_length and str(region.get("regionId")) in best_regions
    )
    region_id = str(region.get("regionId") or "").strip()
    region_name = str(region.get("name") or region_id).strip()
    if not region_id or not region_name:
        return None
    if exiting:
        corridor = next(
            (
                item
                for item in source.get("regions", [])
                if isinstance(item, Mapping)
                and item.get("regionId") == "corridor"
            ),
            None,
        )
        if region_id == "corridor" or not isinstance(corridor, Mapping):
            return None
        return OfficePlaceRequest(
            objective=text,
            location_id="corridor",
            location_name=str(corridor.get("name") or "走廊"),
            task="exit",
            requested_role="interior",
            source_location_id=region_id,
        )
    task: Literal["stand", "observe_report", "exit"] = (
        "observe_report" if _OBSERVE_PATTERN.search(text) else "stand"
    )
    if _ENTRY_PATTERN.search(text):
        role: Literal["entry", "interior", "observation_vantage"] = "entry"
    elif task == "observe_report":
        role = "observation_vantage"
    else:
        role = "interior"
    return OfficePlaceRequest(
        objective=text,
        location_id=region_id,
        location_name=region_name,
        task=task,
        requested_role=role,
    )


def envelope_defaults(objective: object) -> dict[str, Any]:
    sequence = parse_office_inspection_sequence_request(objective)
    if sequence is not None:
        room_count = len(sequence.places)
        return {
            "maximumGoalDistanceM": 20.0,
            "maximumCumulativeDistanceM": 80.0,
            "maximumSpeedMps": 0.20,
            "maximumDurationSeconds": 4_800,
            "maximumAutoRevisions": 8,
            "maximumStageAttempts": 8,
            "maximumScanYawRad": min(40.0 * math.pi, room_count * math.tau),
        }
    if parse_office_place_request(objective) is None:
        return {}
    return {
        "maximumGoalDistanceM": 20.0,
        "maximumCumulativeDistanceM": 50.0,
        "maximumSpeedMps": 0.20,
        "maximumDurationSeconds": 2_400,
        "maximumAutoRevisions": 8,
        "maximumStageAttempts": 8,
        "maximumScanYawRad": 2.0 * math.pi,
    }


def _point_in_polygon(
    x: float,
    y: float,
    polygon: list[Mapping[str, Any]],
) -> bool:
    """Return whether one current map pose lies inside a published region."""

    if len(polygon) < 3:
        return False
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1 = float(previous.get("x", 0.0))
        y1 = float(previous.get("y", 0.0))
        x2 = float(current.get("x", 0.0))
        y2 = float(current.get("y", 0.0))
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _region_for_id(
    source: Mapping[str, Any],
    region_id: str,
) -> Mapping[str, Any] | None:
    return next(
        (
            region
            for region in source.get("regions", [])
            if isinstance(region, Mapping)
            and str(region.get("regionId") or "").strip() == region_id
        ),
        None,
    )


def _pose_inside_region(
    pose: PoseMap | None,
    region_id: str,
    source: Mapping[str, Any],
) -> bool:
    if pose is None or pose.frameId != "map":
        return False
    region = _region_for_id(source, region_id)
    polygon = region.get("polygonXY") if isinstance(region, Mapping) else None
    if not isinstance(polygon, list) or not all(
        isinstance(item, Mapping) for item in polygon
    ):
        return False
    return _point_in_polygon(pose.x, pose.y, polygon)


def current_office_region_id(
    pose: PoseMap | None,
    *,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve only the robot's fresh map pose to a published Office region."""

    source = office_catalog() if inspection_catalog is None else inspection_catalog
    if pose is None or pose.frameId != "map":
        return None
    for region in source.get("regions", []):
        if not isinstance(region, Mapping):
            continue
        region_id = str(region.get("regionId") or "").strip()
        if region_id and _pose_inside_region(pose, region_id, source):
            return region_id
    return None


def _pose_near_station(
    pose: PoseMap | None,
    station: Mapping[str, Any],
    *,
    maximum_distance_m: float = PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M,
) -> bool:
    if pose is None or pose.frameId != "map":
        return False
    target = station.get("pose")
    if not isinstance(target, Mapping):
        return False
    try:
        distance = math.hypot(
            float(target["x"]) - pose.x,
            float(target["y"]) - pose.y,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(distance) and distance <= maximum_distance_m


def _station_role(station: Mapping[str, Any]) -> str:
    purpose = str(station.get("purpose") or "").casefold()
    if purpose == "entrance":
        return "entry"
    if purpose in {"interior", "corridor"}:
        return "interior"
    return purpose


def _ordered_stations(
    request: OfficePlaceRequest,
    *,
    current_pose: PoseMap | None,
    inspection_catalog: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    stations = [
        item
        for item in office_station_index(inspection_catalog).values()
        if item.get("regionId") == request.location_id
    ]
    entry = next((item for item in stations if _station_role(item) == "entry"), None)
    interior = [item for item in stations if _station_role(item) == "interior"]
    if request.location_id == "corridor" and not interior:
        interior = stations
    if not interior and request.requested_role != "entry":
        raise ValueError(
            f"Office V2 location has no interior station: {request.location_id}"
        )

    catalog_order = {
        str(station.get("stationId")): index
        for index, station in enumerate(stations)
    }
    # Interior selection is evaluated from the preceding reviewed entry node,
    # not from the robot's pose when the whole Mission is drafted.  The robot
    # always visits ``entry`` first, so using the pre-Mission pose can select a
    # geometrically nearer station on the far side of furniture even though it
    # will no longer be the nearer station when this node executes.
    routing_origin = current_pose
    entry_pose = entry.get("pose") if isinstance(entry, Mapping) else None
    if isinstance(entry_pose, Mapping) and request.requested_role != "entry":
        try:
            routing_origin = PoseMap(
                frameId="map",
                x=float(entry_pose["x"]),
                y=float(entry_pose["y"]),
                z=float(entry_pose.get("z", 0.0)),
                yaw=float(entry_pose.get("yaw", 0.0)),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            routing_origin = current_pose

    def score(station: Mapping[str, Any]) -> tuple[float, float, int]:
        pose = station.get("pose")
        if not isinstance(pose, Mapping):
            return (math.inf, math.inf, catalog_order.get(str(station.get("stationId")), 10_000))
        headings = station.get("scanHeadingsRad")
        coverage = -float(len(headings)) if isinstance(headings, list) else 0.0
        if routing_origin is None:
            distance = 0.0
        else:
            distance = math.hypot(
                float(pose.get("x", 0.0)) - routing_origin.x,
                float(pose.get("y", 0.0)) - routing_origin.y,
            )
        return (
            coverage if request.task == "observe_report" else 0.0,
            distance,
            catalog_order.get(str(station.get("stationId")), 10_000),
        )

    return entry, sorted(interior, key=score)


def _record_id(mission_id: str, station_id: str) -> str:
    return f"world-{mission_id}-scene-target-{station_id}"


def _place_target(
    request: OfficePlaceRequest,
    *,
    mission_id: str,
    selected: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    role: Literal["entry", "interior", "observation_vantage"],
) -> PlaceTarget:
    selected_id = str(selected["stationId"])
    candidate_ids = [_record_id(mission_id, str(item["stationId"])) for item in candidates]
    selected_record = _record_id(mission_id, selected_id)
    if selected_record in candidate_ids:
        candidate_ids.remove(selected_record)
    return PlaceTarget(
        locationId=request.location_id,
        locationName=request.location_name,
        anchorRole=role,
        stationId=selected_id,
        targetRecordId=selected_record,
        candidateTargetRecordIds=[selected_record, *candidate_ids],
    )


def _completion(
    request: OfficePlaceRequest,
    *,
    predicate: Literal["at_anchor", "inside_region"],
) -> PlaceCompletion:
    return PlaceCompletion(
        predicate=predicate,
        regionId=request.location_id,
        minimumClearanceM=(
            PLACE_MINIMUM_CLEARANCE_M if predicate == "inside_region" else 0.0
        ),
        maximumAnchorDistanceM=(
            PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M
            if predicate == "inside_region"
            else PLACE_MAXIMUM_ANCHOR_DISTANCE_M
        ),
    )


def compile_plan(
    request: OfficePlaceRequest,
    brief: Mapping[str, Any],
    *,
    current_pose: PoseMap | None,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> MissionPlanDraft:
    source = office_catalog() if inspection_catalog is None else dict(inspection_catalog)
    if source.get("containsObjectAnswers") is not False:
        raise ValueError("Office semantic catalog violates the product boundary")
    mission_id = str(brief["missionId"])
    envelope = MissionEnvelope.model_validate(brief["envelope"])
    required = {"navigation.navigate_to_pose"}
    if request.requested_role != "entry":
        required.add(SCENE_VERIFY_INSIDE_SKILL)
    if request.task == "observe_report":
        required.update({OFFICE_INSPECTION_SKILL, MISSION_REPORT_SKILL})
    missing = required.difference(envelope.allowedSkillIds)
    if missing:
        raise ValueError(f"Office place capability is missing: {sorted(missing)[0]}")

    if request.task == "exit":
        return _compile_exit_plan(
            request,
            brief,
            envelope=envelope,
            inspection_catalog=source,
        )

    entry, interior_candidates = _ordered_stations(
        request,
        current_pose=current_pose,
        inspection_catalog=source,
    )
    world_ids = {
        str(item.get("recordId"))
        for item in brief.get("worldRecords", [])
        if isinstance(item, Mapping)
    }
    nodes: list[MissionNode] = []
    previous: str | None = None

    if request.requested_role == "entry":
        if entry is None:
            raise ValueError(
                f"Office V2 location has no entry station: {request.location_id}"
            )
        targets = [entry]
        final_target = _place_target(
            request,
            mission_id=mission_id,
            selected=entry,
            candidates=[entry],
            role="entry",
        )
        selected_station = entry
    else:
        if not interior_candidates:
            raise ValueError(
                f"Office V2 location has no interior station: {request.location_id}"
            )
        targets = ([entry] if entry is not None else []) + [interior_candidates[0]]
        final_target = _place_target(
            request,
            mission_id=mission_id,
            selected=interior_candidates[0],
            candidates=interior_candidates,
            role=(
                "observation_vantage"
                if request.task == "observe_report"
                else "interior"
            ),
        )
        selected_station = interior_candidates[0]

    if any(item not in world_ids for item in final_target.candidateTargetRecordIds):
        raise ValueError(
            "Office place target is not published in the bound Scene: "
            f"{final_target.stationId}"
        )

    for index, station in enumerate(targets, start=1):
        station_id = str(station["stationId"])
        is_final = station_id == str(selected_station["stationId"])
        role: Literal["entry", "interior", "observation_vantage"] = (
            final_target.anchorRole if is_final else "entry"
        )
        place = (
            final_target
            if is_final
            else _place_target(
                request,
                mission_id=mission_id,
                selected=station,
                candidates=[station],
                role="entry",
            )
        )
        completion = _completion(
            request,
            predicate=(
                "inside_region"
                if is_final and role != "entry"
                else "at_anchor"
            ),
        )
        if place.targetRecordId not in world_ids:
            raise ValueError(
                f"Office place target is not published in the bound Scene: {station_id}"
            )
        node_id = f"place-nav-{index:02d}-{station_id}"
        nodes.append(
            MissionNode(
                nodeId=node_id,
                skillId="navigation.navigate_to_pose",
                skillVersion="2.0.0",
                purpose=(
                    f"进入{request.location_name}内部"
                    if is_final and role != "entry"
                    else f"前往{request.location_name}入口"
                ),
                intent=PlaceNavigationIntent(
                    task="navigate_to_place",
                    place=place,
                    completion=completion,
                ).model_dump(mode="json"),
                dependsOn=[] if previous is None else [previous],
                riskLevel=RiskLevel.MEDIUM,
            )
        )
        previous = node_id

    if request.requested_role != "entry":
        verify_id = f"place-verify-{request.location_id}"
        nodes.append(
            MissionNode(
                nodeId=verify_id,
                skillId=SCENE_VERIFY_INSIDE_SKILL,
                skillVersion="1.0.0",
                purpose=f"确认机器人已完整进入{request.location_name}",
                intent=PlaceNavigationIntent(
                    task="verify_inside_region",
                    place=final_target,
                    completion=_completion(request, predicate="inside_region"),
                ).model_dump(mode="json"),
                dependsOn=[] if previous is None else [previous],
                riskLevel=RiskLevel.LOW,
            )
        )
        previous = verify_id

    if request.task == "observe_report":
        station_id = str(selected_station["stationId"])
        scan_id = f"place-scan-{station_id}"
        nodes.append(
            MissionNode(
                nodeId=scan_id,
                skillId=OFFICE_INSPECTION_SKILL,
                skillVersion="1.0.0",
                purpose=f"在{request.location_name}内部停稳并多方向观察",
                intent={
                    "stationId": station_id,
                    "regionId": request.location_id,
                    "regionName": request.location_name,
                    "scanPurpose": "location_observation",
                    "requiresInsideVerification": True,
                },
                dependsOn=[str(previous)],
                riskLevel=RiskLevel.MEDIUM,
            )
        )
        nodes.append(
            MissionNode(
                nodeId=f"place-report-{request.location_id}",
                skillId=MISSION_REPORT_SKILL,
                skillVersion="1.0.0",
                purpose=f"汇总{request.location_name}内实际观察到的信息",
                intent={
                    "reportScope": "current_location",
                    "regionId": request.location_id,
                    "stationId": station_id,
                    "locale": str(brief.get("locale", "zh-CN")),
                },
                dependsOn=[scan_id],
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


def _compile_observation_segment(
    request: OfficePlaceRequest,
    *,
    source: Mapping[str, Any],
    mission_id: str,
    world_ids: set[str],
    current_pose: PoseMap | None,
    segment_index: int,
    dependency: str | None,
) -> tuple[list[MissionNode], str, str]:
    """Compile one room's entry, interior verification and scan nodes."""

    entry, interior_candidates = _ordered_stations(
        request,
        current_pose=current_pose,
        inspection_catalog=source,
    )
    if not interior_candidates:
        raise ValueError(
            f"Office V2 location has no interior station: {request.location_id}"
        )

    selected_station = interior_candidates[0]
    final_target = _place_target(
        request,
        mission_id=mission_id,
        selected=selected_station,
        candidates=interior_candidates,
        role="observation_vantage",
    )
    current_inside = _pose_inside_region(
        current_pose,
        request.location_id,
        source,
    )
    already_at_selected = current_inside and _pose_near_station(
        current_pose,
        selected_station,
    )
    targets = (
        []
        if already_at_selected
        else [selected_station]
        if current_inside
        else ([entry] if entry is not None else []) + [selected_station]
    )
    if any(item not in world_ids for item in final_target.candidateTargetRecordIds):
        raise ValueError(
            "Office inspection target is not published in the bound Scene: "
            f"{final_target.stationId}"
        )
    nodes: list[MissionNode] = []
    previous = dependency
    for target_index, station in enumerate(targets, start=1):
        station_id = str(station["stationId"])
        is_final = target_index == len(targets)
        place = (
            final_target
            if is_final
            else _place_target(
                request,
                mission_id=mission_id,
                selected=station,
                candidates=[station],
                role="entry",
            )
        )
        if any(item not in world_ids for item in place.candidateTargetRecordIds):
            raise ValueError(
                f"Office inspection target is not published in the bound Scene: {station_id}"
            )
        node_id = (
            f"office-sequence-{segment_index:02d}-nav-"
            f"{target_index:02d}-{station_id}"
        )
        nodes.append(
            MissionNode(
                nodeId=node_id,
                skillId="navigation.navigate_to_pose",
                skillVersion="2.0.0",
                purpose=(
                    f"进入{request.location_name}内部"
                    if is_final
                    else f"前往{request.location_name}入口"
                ),
                intent=PlaceNavigationIntent(
                    task="navigate_to_place",
                    place=place,
                    completion=_completion(
                        request,
                        predicate="inside_region" if is_final else "at_anchor",
                    ),
                ).model_dump(mode="json"),
                dependsOn=[] if previous is None else [previous],
                riskLevel=RiskLevel.MEDIUM,
            )
        )
        previous = node_id

    verify_id = f"office-sequence-{segment_index:02d}-verify-{request.location_id}"
    nodes.append(
        MissionNode(
            nodeId=verify_id,
            skillId=SCENE_VERIFY_INSIDE_SKILL,
            skillVersion="1.0.0",
            purpose=f"确认机器人已完整进入{request.location_name}",
            intent=PlaceNavigationIntent(
                task="verify_inside_region",
                place=final_target,
                completion=_completion(request, predicate="inside_region"),
            ).model_dump(mode="json"),
            dependsOn=[] if previous is None else [previous],
            riskLevel=RiskLevel.LOW,
        )
    )
    scan_id = f"office-sequence-{segment_index:02d}-scan-{selected_station['stationId']}"
    nodes.append(
        MissionNode(
            nodeId=scan_id,
            skillId=OFFICE_INSPECTION_SKILL,
            skillVersion="1.0.0",
            purpose=f"在{request.location_name}内部停稳并多方向观察",
            intent={
                "stationId": str(selected_station["stationId"]),
                "regionId": request.location_id,
                "regionName": request.location_name,
                "scanPurpose": "location_observation",
                "requiresInsideVerification": True,
            },
            dependsOn=[verify_id],
            riskLevel=RiskLevel.MEDIUM,
        )
    )
    return nodes, scan_id, str(selected_station["stationId"])


def compile_inspection_sequence_plan(
    request: OfficeInspectionSequenceRequest,
    brief: Mapping[str, Any],
    *,
    current_pose: PoseMap | None,
    inspection_catalog: Mapping[str, Any] | None = None,
) -> MissionPlanDraft:
    """Compile one Mission that visits and observes each requested room."""

    source = office_catalog() if inspection_catalog is None else dict(inspection_catalog)
    if source.get("containsObjectAnswers") is not False:
        raise ValueError("Office semantic catalog violates the product boundary")
    if len(request.places) < 1 or len(request.places) > 8:
        raise ValueError("Office inspection sequence must contain one to eight rooms")
    location_ids = [item.location_id for item in request.places]
    if len(location_ids) != len(set(location_ids)):
        raise ValueError("Office inspection sequence contains duplicate rooms")

    mission_id = str(brief["missionId"])
    envelope = MissionEnvelope.model_validate(brief["envelope"])
    required = {
        "navigation.navigate_to_pose",
        SCENE_VERIFY_INSIDE_SKILL,
        OFFICE_INSPECTION_SKILL,
        MISSION_REPORT_SKILL,
    }
    missing = required.difference(envelope.allowedSkillIds)
    if missing:
        raise ValueError(f"Office inspection capability is missing: {sorted(missing)[0]}")
    allowed_regions = set(envelope.allowedRegionIds)
    if allowed_regions and any(item not in allowed_regions for item in location_ids):
        raise ValueError("Office inspection sequence contains a region outside the authorization envelope")

    world_ids = {
        str(item.get("recordId"))
        for item in brief.get("worldRecords", [])
        if isinstance(item, Mapping)
    }
    nodes: list[MissionNode] = []
    previous: str | None = None
    station_ids: list[str] = []
    for index, place in enumerate(request.places, start=1):
        segment, previous, station_id = _compile_observation_segment(
            place,
            source=source,
            mission_id=mission_id,
            world_ids=world_ids,
            current_pose=current_pose if index == 1 else None,
            segment_index=index,
            dependency=previous,
        )
        nodes.extend(segment)
        station_ids.append(station_id)

    assert previous is not None
    labels = "、".join(item.location_name for item in request.places)
    nodes.append(
        MissionNode(
            nodeId="office-sequence-final-report",
            skillId=MISSION_REPORT_SKILL,
            skillVersion="1.0.0",
            purpose=f"按房间汇总{labels}内及沿途观察信息",
            intent={
                "reportScope": "selected_locations",
                "regionIds": location_ids,
                "stationIds": station_ids,
                "locale": str(brief.get("locale", "zh-CN")),
            },
            dependsOn=[previous],
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


def _compile_exit_plan(
    request: OfficePlaceRequest,
    brief: Mapping[str, Any],
    *,
    envelope: MissionEnvelope,
    inspection_catalog: Mapping[str, Any],
) -> MissionPlanDraft:
    source_id = request.source_location_id
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Office exit task has no source region")
    regions = {
        str(item.get("regionId")): item
        for item in inspection_catalog.get("regions", [])
        if isinstance(item, Mapping)
    }
    source_region = regions.get(source_id)
    destination_region = regions.get(request.location_id)
    if not isinstance(source_region, Mapping) or not isinstance(
        destination_region, Mapping
    ):
        raise ValueError("Office exit regions are absent from the product catalog")
    stations = office_station_index(inspection_catalog)
    source_stations = [
        item for item in stations.values() if item.get("regionId") == source_id
    ]
    source_entry = next(
        (item for item in source_stations if _station_role(item) == "entry"),
        None,
    )
    destination_stations = [
        item
        for item in stations.values()
        if item.get("regionId") == request.location_id
        and _station_role(item) == "interior"
    ]
    if source_entry is None or not destination_stations:
        raise ValueError("Office exit route has no reviewed portal or destination")
    entry_pose = source_entry.get("pose")
    if not isinstance(entry_pose, Mapping):
        raise ValueError("Office exit portal pose is invalid")
    destination_stations.sort(
        key=lambda item: math.hypot(
            float(item.get("pose", {}).get("x", 0.0)) - float(entry_pose["x"]),
            float(item.get("pose", {}).get("y", 0.0)) - float(entry_pose["y"]),
        )
    )
    destination = destination_stations[0]
    mission_id = str(brief["missionId"])
    source_request = OfficePlaceRequest(
        objective=request.objective,
        location_id=source_id,
        location_name=str(source_region.get("name") or source_id),
        task="stand",
        requested_role="entry",
    )
    source_target = _place_target(
        source_request,
        mission_id=mission_id,
        selected=source_entry,
        candidates=[source_entry],
        role="entry",
    )
    destination_target = _place_target(
        request,
        mission_id=mission_id,
        selected=destination,
        candidates=destination_stations,
        role="interior",
    )
    world_ids = {
        str(item.get("recordId"))
        for item in brief.get("worldRecords", [])
        if isinstance(item, Mapping)
    }
    for target in (source_target, destination_target):
        if any(item not in world_ids for item in target.candidateTargetRecordIds):
            raise ValueError("Office exit target is absent from the bound Scene")
    entry_id = f"place-exit-01-{source_entry['stationId']}"
    destination_id = f"place-exit-02-{destination['stationId']}"
    verify_id = f"place-exit-verify-{request.location_id}"
    nodes = [
        MissionNode(
            nodeId=entry_id,
            skillId="navigation.navigate_to_pose",
            skillVersion="2.0.0",
            purpose=f"前往{source_request.location_name}出口",
            intent=PlaceNavigationIntent(
                task="navigate_to_place",
                place=source_target,
                completion=_completion(source_request, predicate="at_anchor"),
            ).model_dump(mode="json"),
            riskLevel=RiskLevel.MEDIUM,
        ),
        MissionNode(
            nodeId=destination_id,
            skillId="navigation.navigate_to_pose",
            skillVersion="2.0.0",
            purpose=f"走出{source_request.location_name}并进入{request.location_name}",
            intent=PlaceNavigationIntent(
                task="navigate_to_place",
                place=destination_target,
                completion=_completion(request, predicate="inside_region"),
            ).model_dump(mode="json"),
            dependsOn=[entry_id],
            riskLevel=RiskLevel.MEDIUM,
        ),
        MissionNode(
            nodeId=verify_id,
            skillId=SCENE_VERIFY_INSIDE_SKILL,
            skillVersion="1.0.0",
            purpose=f"确认机器人已离开{source_request.location_name}并进入{request.location_name}",
            intent=PlaceNavigationIntent(
                task="verify_inside_region",
                place=destination_target,
                completion=_completion(request, predicate="inside_region"),
            ).model_dump(mode="json"),
            dependsOn=[destination_id],
            riskLevel=RiskLevel.LOW,
        ),
    ]
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


def parse_place_navigation_intent(value: Mapping[str, Any]) -> PlaceNavigationIntent:
    return PlaceNavigationIntent.model_validate(dict(value))


__all__ = [
    "OfficeInspectionSequenceRequest",
    "OfficePlaceRequest",
    "PLACE_COMPLETION_SCHEMA_VERSION",
    "PLACE_INTENT_SCHEMA_VERSION",
    "PLACE_MAXIMUM_ANCHOR_DISTANCE_M",
    "PLACE_INSIDE_MAXIMUM_ANCHOR_DISTANCE_M",
    "PLACE_MINIMUM_CLEARANCE_M",
    "PlaceCompletion",
    "PlaceNavigationIntent",
    "PlaceTarget",
    "SCENE_VERIFY_INSIDE_SKILL",
    "compile_plan",
    "compile_inspection_sequence_plan",
    "current_office_region_id",
    "envelope_defaults",
    "parse_office_inspection_sequence_request",
    "parse_office_place_request",
    "parse_place_navigation_intent",
]
