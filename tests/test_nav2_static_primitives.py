from __future__ import annotations

from copy import deepcopy
from heapq import heappop, heappush
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

from jsonschema import Draft202012Validator, ValidationError
import pytest

from hide_mujoco_sonic.safety import CmdVelGuard


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "scenes" / "office_v2" / "generated"
MAP_YAML = GENERATED / "nav2" / "office_v2.yaml"
MAP_IMAGE = GENERATED / "nav2" / "office_v2.pgm"
RUNTIME_MANIFEST = GENERATED / "runtime_manifest.json"
MOTION_PROFILE_MANIFEST = ROOT / "docker" / "nav2" / "motion_profile_manifest.json"
SCHEMA_PATH = ROOT / "docker" / "nav2" / "static_primitive_plan.schema.json"
START = (-6.45, 0.0)
GOAL = (-7.30, -0.30)
CLEARANCE_RADIUS = 0.35 + 0.02 + 0.05
FIXTURE_PLANNING_CLEARANCE = 0.445


def _planner():
    path = ROOT / "docker" / "nav2" / "static_primitive_planner.py"
    spec = importlib.util.spec_from_file_location("static_primitive_planner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plan_kwargs() -> dict[str, object]:
    return {
        "map_yaml_path": MAP_YAML,
        "map_image_path": MAP_IMAGE,
        "runtime_manifest_path": RUNTIME_MANIFEST,
        "motion_profile_manifest_path": MOTION_PROFILE_MANIFEST,
        "robot_radius_m": 0.35,
        "footprint_padding_m": 0.02,
    }


def _astar_fixture_navfn_path(
    module,
    start: tuple[float, float],
    goal: tuple[float, float],
) -> dict[str, object]:
    """Generate offline test input; production planning remains NavFn-owned."""
    static_map = module.StaticMap.load(MAP_YAML, MAP_IMAGE)
    # Give the offline A* fixture additional room so it resembles NavFn's
    # inflation-cost preference instead of hugging the minimum legal contour.
    # Product code never runs this A*; the segmenter still revalidates against
    # the explicit 0.42 m contract below.
    blocked = static_map.inflated_blocked_cells(FIXTURE_PLANNING_CLEARANCE)
    start_cell = static_map.world_to_cell(*start)
    goal_cell = static_map.world_to_cell(*goal)
    assert start_cell not in blocked
    assert goal_cell not in blocked

    steps = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start_cell)]
    costs = {start_cell: 0.0}
    predecessors: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        _, current = heappop(frontier)
        if current == goal_cell:
            break
        for dr, dc, step_cost in steps:
            candidate = (current[0] + dr, current[1] + dc)
            if (
                not (
                    0 <= candidate[0] < static_map.height
                    and 0 <= candidate[1] < static_map.width
                )
                or candidate in blocked
            ):
                continue
            if dr and dc:
                side_cells = (
                    (current[0] + dr, current[1]),
                    (current[0], current[1] + dc),
                )
                if any(cell in blocked for cell in side_cells):
                    continue
            cost = costs[current] + step_cost
            if cost >= costs.get(candidate, math.inf):
                continue
            costs[candidate] = cost
            predecessors[candidate] = current
            heuristic = math.hypot(
                goal_cell[0] - candidate[0],
                goal_cell[1] - candidate[1],
            )
            heappush(frontier, (cost + heuristic, candidate))
    assert goal_cell in costs

    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(predecessors[cells[-1]])
    cells.reverse()
    points = [start]
    points.extend(static_map.cell_center(cell) for cell in cells[1:-1])
    points.append(goal)
    return {
        "frame_id": "map",
        "poses": [{"x": x, "y": y, "yaw": 0.0} for x, y in points],
    }


def _build(module, navfn_path: dict[str, object]) -> dict[str, object]:
    return module.build_primitive_plan(navfn_path, **_plan_kwargs())


def _assert_primitive_plan(module, navfn_path, plan) -> None:
    assert plan["schema_version"] == "h-ide.nav2-static-primitives/1"
    assert plan["motion_mode"] == "static_spin_drive_primitives"
    assert plan["planner_source"]["action_type"] == (
        "nav2_msgs/action/ComputePathToPose"
    )
    assert plan["planner_source"]["plugin"] == (
        "nav2_navfn_planner::NavfnPlanner"
    )
    assert plan["planner_source"]["navfn_path_sha256"] == (
        module._canonical_sha256(navfn_path)
    )
    assert plan["planner_source"]["navfn_path_pose_count"] == len(
        navfn_path["poses"]
    )
    assert plan["map_provenance"]["scene_source_sha256"] == (
        "76ee5cd945aab28f034075eab65e5e104085c904048bf2d7085495796b46c379"
    )
    assert all(
        len(plan["map_provenance"][name]) == 64
        for name in ("yaml_sha256", "image_sha256", "scene_source_sha256")
    )
    assert plan["runtime_provenance"]["runtime_manifest_sha256"] == (
        module._sha256(RUNTIME_MANIFEST)
    )
    assert plan["runtime_provenance"]["motion_profile_manifest_sha256"] == (
        module._sha256(MOTION_PROFILE_MANIFEST)
    )
    assert plan["runtime_provenance"]["follow_path_motion_accepted"] is False
    assert len(plan["segmenter"]["source_sha256"]) == 64
    assert plan["segmenter"]["extra_clearance_m"] == 0.05
    assert plan["segmenter"]["clearance_radius_m"] == pytest.approx(
        CLEARANCE_RADIUS
    )
    # Residuals within the measured whole-body braking envelope are complete;
    # any new motion segment must be longer than that endpoint envelope. Its
    # command speed remains NVIDIA's exact 0.30 m/s Slow Walk command.
    assert plan["segmenter"]["min_segment_m"] == 0.16
    assert plan["segmenter"]["max_segment_m"] == 2.0
    assert plan["execution_policy"] == {
        "fresh_odometry_required": True,
        "fresh_pose_frame": "map",
        "revalidate_line_of_sight_before_drive": True,
        "endpoint_tolerance_m": 0.15,
        "heading_tolerance_rad": 0.09,
        "drive_speed_mps": 0.30,
        "spin_min_angular_rps": 0.10,
        "spin_max_angular_rps": 0.20,
        "spin_angular_accel_rps2": 0.30,
        "zero_spin_policy": "skip_within_heading_tolerance",
        "final_heading_policy": (
            "omit_checkpoint_if_within_heading_tolerance"
        ),
        "short_residual_policy": (
            "skip_within_endpoint_tolerance_else_fail_below_min_segment"
        ),
    }

    endpoints = plan["segment_endpoint_indices"]
    assert endpoints[0] == 0
    assert endpoints[-1] == len(navfn_path["poses"]) - 1
    assert endpoints == sorted(set(endpoints))
    assert len(plan["segments"]) >= 2
    static_map = module.StaticMap.load(MAP_YAML, MAP_IMAGE)
    blocked = static_map.inflated_blocked_cells(CLEARANCE_RADIUS)
    previous_end = None
    for segment_id, (segment, start_index, end_index) in enumerate(
        zip(plan["segments"], endpoints, endpoints[1:])
    ):
        assert segment["id"] == f"segment-{segment_id:03d}"
        assert segment["path_start_index"] == start_index
        assert segment["path_end_index"] == end_index
        source_start = navfn_path["poses"][start_index]
        source_end = navfn_path["poses"][end_index]
        assert segment["start_map"] == {
            "x": source_start["x"],
            "y": source_start["y"],
        }
        assert segment["end_map"] == {
            "x": source_end["x"],
            "y": source_end["y"],
        }
        if previous_end is not None:
            assert segment["start_map"] == previous_end
        previous_end = segment["end_map"]
        assert 0.16 <= segment["distance_m"] <= 2.0
        assert static_map.segment_is_clear(
            (source_start["x"], source_start["y"]),
            (source_end["x"], source_end["y"]),
            blocked,
        )

    kinds = [primitive["kind"] for primitive in plan["primitives"]]
    assert kinds == [
        "spin" if index % 2 == 0 else "drive_on_heading"
        for index in range(len(kinds))
    ]
    assert [item["sequence"] for item in plan["primitives"]] == list(
        range(len(kinds))
    )
    assert plan["primitives"][-1]["checkpoint_role"] == "final_heading"
    for segment, spin, drive in zip(
        plan["segments"],
        plan["primitives"][0::2],
        plan["primitives"][1::2],
    ):
        assert spin["checkpoint_role"] == "align_segment"
        assert spin["segment_id"] == segment["id"]
        assert spin["reference_end_map"] == segment["end_map"]
        assert spin["path_index"] == segment["path_start_index"]
        assert drive["checkpoint_role"] == "reach_segment_endpoint"
        assert drive["segment_id"] == segment["id"]
        assert drive["end_map"] == segment["end_map"]
        assert drive["path_end_index"] == segment["path_end_index"]
        assert drive["speed_mps"] == 0.30
    assert all(item["requires_fresh_odometry"] for item in plan["primitives"])
    # Stored checkpoints contain only absolute references. Relative action
    # goals are created by resolve_live_primitive from fresh odometry.
    assert all("target_yaw_rad" not in item for item in plan["primitives"])
    assert all("target" not in item for item in plan["primitives"])
    unsigned = dict(plan)
    assert unsigned.pop("plan_sha256") == module._canonical_sha256(unsigned)


def test_locked_short_route_segments_bidirectionally_without_oracle_data() -> None:
    module = _planner()
    plans = []
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for start, goal in ((START, GOAL), (GOAL, START)):
        navfn_path = _astar_fixture_navfn_path(module, start, goal)
        plan = _build(module, navfn_path)
        _assert_primitive_plan(module, navfn_path, plan)
        validator.validate(plan)
        module.verify_primitive_plan(plan, navfn_path, **_plan_kwargs())
        plans.append(plan)

    assert plans[0]["planner_source"]["navfn_path_sha256"] != plans[1][
        "planner_source"
    ]["navfn_path_sha256"]
    # Both directions remain multi-segment detours at the reviewed goal; no
    # segment may be manufactured inside the 15 cm braking envelope.
    assert all(len(plan["segments"]) >= 2 for plan in plans)
    assert all(
        segment["distance_m"] >= 0.16
        for plan in plans
        for segment in plan["segments"]
    )
    source = (
        ROOT / "docker" / "nav2" / "static_primitive_planner.py"
    ).read_text(encoding="utf-8")
    assert "heappop" not in source
    assert "evaluation/" not in source
    assert "scenes/office_v2/scene.yaml" not in source


def test_navfn_relaxed_start_must_reenter_full_clearance_before_normal_segments(
) -> None:
    module = _planner()
    start = (-4.476, -2.389)
    reentry = (-4.475, -2.125)
    suffix = _astar_fixture_navfn_path(module, reentry, GOAL)
    navfn_path = {
        "frame_id": "map",
        "poses": [
            {"x": start[0], "y": start[1], "yaw": 0.0},
            {"x": -4.475, "y": -2.325, "yaw": 0.0},
            {"x": -4.475, "y": -2.275, "yaw": 0.0},
            {"x": -4.475, "y": -2.225, "yaw": 0.0},
            {"x": reentry[0], "y": reentry[1], "yaw": 0.0},
            *suffix["poses"][1:],
        ],
    }

    plan = _build(module, navfn_path)
    static_map = module.StaticMap.load(MAP_YAML, MAP_IMAGE)
    full = static_map.inflated_blocked_cells(CLEARANCE_RADIUS)
    relaxed = static_map.inflated_blocked_cells(
        module.RELAXED_LIVE_EXECUTION_CLEARANCE_RADIUS_M
    )

    assert static_map.world_to_cell(*start) in full
    assert static_map.world_to_cell(*start) not in relaxed
    assert plan["segment_endpoint_indices"][:2] == [0, 4]
    first = plan["segments"][0]
    assert first["end_map"] == {"x": reentry[0], "y": reentry[1]}
    assert static_map.segment_is_clear(start, reentry, relaxed)
    assert all(
        static_map.world_to_cell(pose["x"], pose["y"]) not in full
        for pose in navfn_path["poses"][4:]
    )
    module.verify_primitive_plan(plan, navfn_path, **_plan_kwargs())


def test_relaxed_start_recovery_may_finish_inside_endpoint_tolerance() -> None:
    """Regression for a live post-Spin pose beside the west waste bin."""

    module = _planner()
    start = (-6.75, -0.25)
    reentry = (-6.35, -0.15)
    goal = (-6.243011889573126, -0.1146368064599927)
    navfn_path = {
        "frame_id": "map",
        "poses": [
            {"x": start[0], "y": start[1], "yaw": 0.0},
            {"x": reentry[0], "y": reentry[1], "yaw": 0.0},
            {"x": goal[0], "y": goal[1], "yaw": 0.24566406970129795},
        ],
    }

    plan = _build(module, navfn_path)
    static_map = module.StaticMap.load(MAP_YAML, MAP_IMAGE)
    full = static_map.inflated_blocked_cells(CLEARANCE_RADIUS)

    assert static_map.world_to_cell(*start) in full
    assert static_map.world_to_cell(*reentry) not in full
    assert math.dist(reentry, goal) < plan["execution_policy"][
        "endpoint_tolerance_m"
    ]
    assert plan["segment_endpoint_indices"] == [0, 1]
    assert plan["segments"] == [
        {
            "id": "segment-000",
            "path_start_index": 0,
            "path_end_index": 1,
            "start_map": {"x": start[0], "y": start[1]},
            "end_map": {"x": reentry[0], "y": reentry[1]},
            "distance_m": pytest.approx(math.dist(start, reentry)),
            "heading_map_yaw_rad": pytest.approx(
                math.atan2(reentry[1] - start[1], reentry[0] - start[0])
            ),
        }
    ]
    module.verify_primitive_plan(plan, navfn_path, **_plan_kwargs())


def test_navfn_goal_in_extra_margin_stops_at_physical_clear_endpoint() -> None:
    module = _planner()
    start = (1.0, 5.1)
    endpoint = (0.75, 4.75)
    goal = (0.70, 4.70)
    navfn_path = {
        "frame_id": "map",
        "poses": [
            {"x": start[0], "y": start[1], "yaw": 0.0},
            {"x": endpoint[0], "y": endpoint[1], "yaw": 0.0},
            {"x": goal[0], "y": goal[1], "yaw": math.pi / 2.0},
        ],
    }

    plan = _build(module, navfn_path)
    static_map = module.StaticMap.load(MAP_YAML, MAP_IMAGE)
    full = static_map.inflated_blocked_cells(CLEARANCE_RADIUS)
    physical = static_map.inflated_blocked_cells(0.35 + 0.02)

    assert static_map.world_to_cell(*endpoint) not in full
    assert static_map.world_to_cell(*goal) in full
    assert static_map.world_to_cell(*goal) not in physical
    assert math.dist(endpoint, goal) < 0.15
    assert plan["segment_endpoint_indices"] == [0, 1]
    assert plan["segments"][-1]["end_map"] == {
        "x": endpoint[0],
        "y": endpoint[1],
    }
    module.verify_primitive_plan(plan, navfn_path, **_plan_kwargs())


def test_navfn_goal_inside_physical_footprint_clearance_is_rejected() -> None:
    module = _planner()
    navfn_path = {
        "frame_id": "map",
        "poses": [
            {"x": 1.0, "y": 5.1, "yaw": 0.0},
            {"x": 0.75, "y": 4.75, "yaw": 0.0},
            {"x": 0.75, "y": 4.55, "yaw": math.pi / 2.0},
        ],
    }

    with pytest.raises(ValueError, match="physical footprint clearance"):
        _build(module, navfn_path)


def test_start_clearance_recovery_moves_forward_monotonically_outward() -> None:
    module = _planner()
    current = {"x": 0.584, "y": 4.368, "yaw": 1.014}

    resolved = module.resolve_start_clearance_recovery(
        current,
        **_plan_kwargs(),
    )

    assert resolved["disposition"] == "send_action"
    assert resolved["action_type"] == "nav2_msgs/action/DriveOnHeading"
    assert 0.16 <= resolved["action_goal"]["target"]["x"] <= 0.50
    assert resolved["heading_error_rad"] <= 0.09
    assert resolved["end_clearance_m"] >= CLEARANCE_RADIUS
    assert resolved["end_clearance_m"] > resolved["start_clearance_m"]
    assert resolved["recovery_end_map"] == {
        "x": pytest.approx(0.825),
        "y": pytest.approx(4.725),
    }


def test_start_clearance_recovery_reverses_without_turning_inside_the_margin() -> None:
    module = _planner()

    resolved = module.resolve_start_clearance_recovery(
        {"x": 3.482, "y": 3.297, "yaw": 2.907},
        **_plan_kwargs(),
    )

    assert resolved["disposition"] == "send_action"
    assert resolved["recovery_direction"] == "reverse"
    assert -0.50 <= resolved["action_goal"]["target"]["x"] <= -0.16
    assert resolved["action_goal"]["speed_mps"] == pytest.approx(0.20)
    assert resolved["heading_error_rad"] <= 0.09
    assert resolved["end_clearance_m"] >= CLEARANCE_RADIUS
    assert resolved["end_clearance_m"] > resolved["start_clearance_m"]


def test_navfn_start_inside_relaxed_physical_clearance_is_rejected() -> None:
    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, START, GOAL)
    navfn_path["poses"][0] = {"x": -5.621, "y": -1.291, "yaw": 0.0}

    with pytest.raises(ValueError, match="relaxed physical clearance"):
        _build(module, navfn_path)


def test_live_resolver_uses_fresh_pose_and_never_emits_zero_or_subfloor_goals() -> None:
    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, START, GOAL)
    plan = _build(module, navfn_path)
    short_segment_index = min(
        range(len(plan["segments"])),
        key=lambda index: plan["segments"][index]["distance_m"],
    )
    segment = plan["segments"][short_segment_index]
    heading = segment["heading_map_yaw_rad"]
    start_pose = {**segment["start_map"], "yaw": heading}

    spin = module.resolve_live_primitive(
        plan,
        2 * short_segment_index,
        navfn_path,
        start_pose,
        **_plan_kwargs(),
    )
    assert spin["disposition"] == "skip_within_heading_tolerance"
    assert spin["action_goal"] is None

    drive = module.resolve_live_primitive(
        plan,
        2 * short_segment_index + 1,
        navfn_path,
        start_pose,
        **_plan_kwargs(),
    )
    assert drive["disposition"] == "send_action"
    assert drive["action_type"] == "nav2_msgs/action/DriveOnHeading"
    assert drive["action_goal"]["speed_mps"] == 0.30
    assert drive["action_goal"]["target"]["x"] == pytest.approx(
        segment["distance_m"]
    )
    assert drive["command_distance_m"] == pytest.approx(
        segment["distance_m"]
    )
    assert drive["endpoint_confirmation_required"] is True

    behind_pose = {
        "x": segment["start_map"]["x"] - 0.40 * math.cos(heading),
        "y": segment["start_map"]["y"] - 0.40 * math.sin(heading),
        "yaw": heading,
    }
    chunked = module.resolve_live_primitive(
        plan,
        2 * short_segment_index + 1,
        navfn_path,
        behind_pose,
        **_plan_kwargs(),
    )
    assert chunked["disposition"] == "send_action"
    assert chunked["fresh_distance_to_endpoint_m"] == pytest.approx(
        segment["distance_m"] + 0.40
    )
    assert chunked["command_distance_m"] == pytest.approx(
        segment["distance_m"] + 0.40
    )
    assert chunked["action_goal"]["target"]["x"] == pytest.approx(
        segment["distance_m"] + 0.40
    )

    misaligned_pose = {
        **segment["start_map"],
        "yaw": module._wrap_angle(heading + 0.20),
    }
    realignment = module.resolve_live_primitive(
        plan,
        2 * short_segment_index + 1,
        navfn_path,
        misaligned_pose,
        **_plan_kwargs(),
    )
    assert realignment == {
        "primitive_sequence": 2 * short_segment_index + 1,
        "plan_sha256": plan["plan_sha256"],
        "fresh_odometry_used": True,
        "planning_clearance_radius_m": pytest.approx(0.42),
        "live_execution_clearance_radius_m": pytest.approx(0.42),
        "disposition": "realign_spin_required",
        "action_goal": None,
        "fresh_distance_to_endpoint_m": pytest.approx(segment["distance_m"]),
        "resolved_target_map_yaw_rad": pytest.approx(heading),
        "heading_error_rad": pytest.approx(-0.20),
        "realignment_primitive_sequence": 2 * short_segment_index,
    }
    correction = module.resolve_live_primitive(
        plan,
        realignment["realignment_primitive_sequence"],
        navfn_path,
        misaligned_pose,
        **_plan_kwargs(),
    )
    assert correction["disposition"] == "send_action"
    assert correction["action_type"] == "nav2_msgs/action/Spin"
    assert correction["action_goal"]["target_yaw_rad"] == pytest.approx(-0.20)

    # A residual at the reviewed whole-body resolution is skipped instead of
    # asking the official >=0.10 rad/s Spin to execute a one-step correction.
    # Anything inside the measured action resolution retains the current
    # heading; anything just outside remains an explicit Spin.
    within_resolution = {
        **segment["start_map"],
        "yaw": heading - 0.09,
    }
    resolved_within = module.resolve_live_primitive(
        plan,
        2 * short_segment_index + 1,
        navfn_path,
        within_resolution,
        **_plan_kwargs(),
    )
    assert resolved_within["disposition"] == "send_action"
    assert resolved_within["action_type"] == "nav2_msgs/action/DriveOnHeading"

    outside_resolution = {
        **segment["start_map"],
        "yaw": heading - 0.0901,
    }
    resolved_outside = module.resolve_live_primitive(
        plan,
        2 * short_segment_index + 1,
        navfn_path,
        outside_resolution,
        **_plan_kwargs(),
    )
    assert resolved_outside["disposition"] == "realign_spin_required"

    endpoint = segment["end_map"]
    near_pose = {
        "x": endpoint["x"] - 0.02 * math.cos(heading),
        "y": endpoint["y"] - 0.02 * math.sin(heading),
        "yaw": heading,
    }
    skipped = module.resolve_live_primitive(
        plan,
        2 * short_segment_index + 1,
        navfn_path,
        near_pose,
        **_plan_kwargs(),
    )
    assert skipped["disposition"] == "skip_within_endpoint_tolerance"
    assert skipped["action_goal"] is None

    unsafe_residual_pose = {
        "x": endpoint["x"] - 0.155 * math.cos(heading),
        "y": endpoint["y"] - 0.155 * math.sin(heading),
        "yaw": heading,
    }
    with pytest.raises(ValueError, match="below minimum segment"):
        module.resolve_live_primitive(
            plan,
            2 * short_segment_index + 1,
            navfn_path,
            unsafe_residual_pose,
            **_plan_kwargs(),
        )

    final_index = len(plan["primitives"]) - 1
    final_pose = {"x": GOAL[0], "y": GOAL[1], "yaw": 0.0}
    final = module.resolve_live_primitive(
        plan,
        final_index,
        navfn_path,
        final_pose,
        **_plan_kwargs(),
    )
    assert final["disposition"] == "skip_within_heading_tolerance"
    assert final["action_goal"] is None


def test_long_segments_still_require_inflated_map_line_of_sight() -> None:
    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, START, GOAL)
    plan_kwargs = _plan_kwargs()
    plan_kwargs["max_segment_m"] = 6.0

    plan = module.build_primitive_plan(navfn_path, **plan_kwargs)
    static_map = module.StaticMap.load(MAP_YAML, MAP_IMAGE)
    blocked = static_map.inflated_blocked_cells(CLEARANCE_RADIUS)
    for segment in plan["segments"]:
        start = segment["start_map"]
        end = segment["end_map"]
        assert static_map.segment_is_clear(
            (start["x"], start["y"]),
            (end["x"], end["y"]),
            blocked,
        )


def test_live_revalidation_uses_physical_envelope_after_a_successful_action(
) -> None:
    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, (-6.45, 0.0), (6.1, 0.0))
    plan = _build(module, navfn_path)
    segment_index = 3
    segment = plan["segments"][segment_index]
    endpoint = segment["end_map"]
    current = {"x": -0.3772, "y": 0.5273}
    current["yaw"] = math.atan2(
        endpoint["y"] - current["y"],
        endpoint["x"] - current["x"],
    )
    static_map = module.StaticMap.load(MAP_YAML, MAP_IMAGE)
    start = (current["x"], current["y"])
    end = (endpoint["x"], endpoint["y"])

    assert not static_map.segment_is_clear(
        start,
        end,
        static_map.inflated_blocked_cells(CLEARANCE_RADIUS),
    )
    assert static_map.segment_is_clear(
        start,
        end,
        static_map.inflated_blocked_cells(
            module.RELAXED_LIVE_EXECUTION_CLEARANCE_RADIUS_M
        ),
    )

    resolved = module.resolve_live_primitive(
        plan,
        2 * segment_index + 1,
        navfn_path,
        current,
        live_execution_clearance_radius_m=(
            module.RELAXED_LIVE_EXECUTION_CLEARANCE_RADIUS_M
        ),
        **_plan_kwargs(),
    )

    assert resolved["disposition"] == "send_action"
    assert resolved["action_type"] == "nav2_msgs/action/DriveOnHeading"
    assert resolved["planning_clearance_radius_m"] == pytest.approx(0.42)
    assert resolved["live_execution_clearance_radius_m"] == pytest.approx(0.25)


def test_final_spin_is_omitted_when_last_segment_already_meets_goal_yaw() -> None:
    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, START, GOAL)
    baseline = _build(module, navfn_path)
    final_heading = baseline["segments"][-1]["heading_map_yaw_rad"]
    navfn_path["poses"][-1]["yaw"] = final_heading + 0.059

    plan = _build(module, navfn_path)

    assert len(plan["primitives"]) == 2 * len(plan["segments"])
    assert plan["primitives"][-1]["checkpoint_role"] == (
        "reach_segment_endpoint"
    )
    assert module.verify_primitive_plan(
        plan,
        navfn_path,
        **_plan_kwargs(),
    ) == plan


def test_live_resolver_rejects_fresh_pose_that_invalidates_the_planned_chord() -> None:
    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, GOAL, START)
    plan = _build(module, navfn_path)
    # The first 7.9 cm segment is necessary to clear the bin. If it were
    # falsely skipped, the actual start cannot safely connect to segment 1's
    # endpoint; fresh line-of-sight validation must stop before any action.
    second_segment_spin = 2
    with pytest.raises(ValueError, match="violates inflated clearance"):
        module.resolve_live_primitive(
            plan,
            second_segment_spin,
            navfn_path,
            {"x": GOAL[0], "y": GOAL[1], "yaw": 0.0},
            **_plan_kwargs(),
        )


def test_segmenter_rejects_a_navfn_path_that_violates_extra_clearance() -> None:
    module = _planner()
    direct_path = {
        "frame_id": "map",
        "poses": [
            {"x": START[0], "y": START[1], "yaw": 0.0},
            {"x": GOAL[0], "y": GOAL[1], "yaw": 0.0},
        ],
    }

    with pytest.raises(ValueError, match="violates inflated clearance"):
        _build(module, direct_path)


def test_map_loader_rejects_rotated_origin_and_same_basename_substitution(
    tmp_path: Path,
) -> None:
    module = _planner()
    copied_image = tmp_path / MAP_IMAGE.name
    shutil.copyfile(MAP_IMAGE, copied_image)
    with pytest.raises(ValueError, match="exact supplied PGM"):
        module.StaticMap.load(MAP_YAML, copied_image)

    copied_yaml = tmp_path / MAP_YAML.name
    text = MAP_YAML.read_text(encoding="utf-8").replace(
        "origin: [-8, -6, 0]",
        "origin: [-8, -6, 0.01]",
    )
    copied_yaml.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="rotated occupancy-map origins"):
        module.StaticMap.load(copied_yaml, copied_image)


def _copy_runtime_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    generated = tmp_path / "generated"
    nav2 = generated / "nav2"
    nav2.mkdir(parents=True)
    yaml_path = nav2 / MAP_YAML.name
    image_path = nav2 / MAP_IMAGE.name
    manifest_path = generated / RUNTIME_MANIFEST.name
    shutil.copyfile(MAP_YAML, yaml_path)
    shutil.copyfile(MAP_IMAGE, image_path)
    shutil.copyfile(RUNTIME_MANIFEST, manifest_path)
    return yaml_path, image_path, manifest_path


def test_plan_integrity_kills_path_plan_map_runtime_and_profile_tampering(
    tmp_path: Path,
) -> None:
    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, START, GOAL)
    plan = _build(module, navfn_path)

    changed_plan = deepcopy(plan)
    changed_plan["segments"][0]["end_map"]["x"] += 0.01
    unsigned = dict(changed_plan)
    unsigned.pop("plan_sha256")
    changed_plan["plan_sha256"] = module._canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="does not match trusted inputs"):
        module.verify_primitive_plan(
            changed_plan,
            navfn_path,
            **_plan_kwargs(),
        )

    changed_path = deepcopy(navfn_path)
    changed_path["poses"][-1]["yaw"] = 0.10
    with pytest.raises(ValueError, match="does not match trusted inputs"):
        module.verify_primitive_plan(plan, changed_path, **_plan_kwargs())

    yaml_path, image_path, runtime_path = _copy_runtime_bundle(tmp_path / "map")
    image_bytes = bytearray(image_path.read_bytes())
    image_bytes[-1] ^= 0x01
    image_path.write_bytes(image_bytes)
    with pytest.raises(ValueError, match="map image hash"):
        module.build_primitive_plan(
            navfn_path,
            **{
                **_plan_kwargs(),
                "map_yaml_path": yaml_path,
                "map_image_path": image_path,
                "runtime_manifest_path": runtime_path,
            },
        )

    yaml_path, image_path, runtime_path = _copy_runtime_bundle(tmp_path / "runtime")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["scene_source_sha256"] = "0" * 64
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError, match="scene source hash"):
        module.build_primitive_plan(
            navfn_path,
            **{
                **_plan_kwargs(),
                "map_yaml_path": yaml_path,
                "map_image_path": image_path,
                "runtime_manifest_path": runtime_path,
            },
        )

    changed_motion_path = tmp_path / "motion_profile_manifest.json"
    changed_motion = json.loads(MOTION_PROFILE_MANIFEST.read_text(encoding="utf-8"))
    changed_motion["follow_path_motion_accepted"] = True
    changed_motion_path.write_text(json.dumps(changed_motion), encoding="utf-8")
    with pytest.raises(ValueError, match="FollowPath motion"):
        module.build_primitive_plan(
            navfn_path,
            **{
                **_plan_kwargs(),
                "motion_profile_manifest_path": changed_motion_path,
            },
        )


def test_schema_and_motion_manifest_forbid_unaccepted_follow_path_motion() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MOTION_PROFILE_MANIFEST.read_text(encoding="utf-8"))

    assert schema["additionalProperties"] is False
    assert {
        "planner_source",
        "map_provenance",
        "runtime_provenance",
        "segmenter",
        "execution_policy",
        "segments",
        "primitives",
        "plan_sha256",
    }.issubset(schema["required"])
    assert all(
        definition.get("additionalProperties") is False
        for name, definition in schema["$defs"].items()
        if name not in {"sha256", "yaw"}
    )
    assert manifest["follow_path_motion_accepted"] is False
    assert manifest["cmd_vel_ownership"]["required_bridge_publisher_count"] == 1
    assert manifest["cmd_vel_ownership"]["strict_runner_node"] == (
        "/strict_motion_runner"
    )
    assert manifest["allowed_product_actions"] == [
        "nav2_msgs/action/ComputePathToPose",
        "nav2_msgs/action/Spin",
        "nav2_msgs/action/DriveOnHeading",
    ]
    assert manifest["primitive_action_contract"] == {
        "drive_on_heading_speed_mps": 0.3,
        "spin_min_angular_rps": 0.1,
        "spin_max_angular_rps": 0.2,
        "spin_angular_accel_rps2": 0.3,
    }
    assert {
        "nav2_msgs/action/NavigateToPose",
        "nav2_msgs/action/FollowPath",
        "nav2_msgs/action/BackUp",
    }.issubset(manifest["forbidden_product_actions"])

    module = _planner()
    navfn_path = _astar_fixture_navfn_path(module, START, GOAL)
    plan = _build(module, navfn_path)
    plan["segments"][0]["unexpected"] = True
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(plan)


def test_runtime_quantizes_nav2_ramp_to_official_forward_speed() -> None:
    guard = CmdVelGuard(
        max_linear_mps=0.40,
        min_walk_command_mps=0.20,
        max_angular_rps=0.25,
        smoothing_alpha=1.0,
    )

    intent = guard.accept(0.199, 0.0, 0.25)
    assert intent.vx == pytest.approx(0.30)
    assert intent.vy == pytest.approx(0.0)
