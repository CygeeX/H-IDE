from __future__ import annotations

from heapq import heappop, heappush
import importlib.util
import json
import math
from pathlib import Path
import shutil

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]

GRID_STEPS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _nav2_params() -> dict:
    return yaml.safe_load(
        (ROOT / "docker" / "nav2" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
    )


def _bundle_verifier():
    path = ROOT / "docker" / "nav2" / "verify_office_v2_bundle.py"
    spec = importlib.util.spec_from_file_location("verify_office_v2_bundle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_pgm(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    assert raw.startswith(b"P5\n")
    header, pixels = raw.split(b"\n255\n", 1)
    dimensions = [
        line for line in header.splitlines()[1:] if not line.startswith(b"#")
    ][-1]
    width, height = (int(value) for value in dimensions.split())
    assert len(pixels) == width * height
    return width, height, pixels


def _world_to_cell(
    pose: list[float],
    *,
    origin: list[float],
    resolution: float,
    height: int,
) -> tuple[int, int]:
    column = int((pose[0] - origin[0]) // resolution)
    row = height - 1 - int((pose[1] - origin[1]) // resolution)
    return row, column


def _traversable_neighbors(
    current: tuple[int, int],
    *,
    width: int,
    height: int,
    blocked: set[tuple[int, int]],
):
    for dr, dc, step_cost in GRID_STEPS:
        candidate = (current[0] + dr, current[1] + dc)
        if (
            not (0 <= candidate[0] < height and 0 <= candidate[1] < width)
            or candidate in blocked
        ):
            continue
        if dr and dc:
            # A diagonal segment sweeps through both adjoining grid edges. It
            # may not squeeze between two inflated lethal cells that only meet
            # at a corner.
            side_cells = (
                (current[0] + dr, current[1]),
                (current[0], current[1] + dc),
            )
            if any(cell in blocked for cell in side_cells):
                continue
        yield candidate, step_cost


def test_grid_detour_model_rejects_diagonal_corner_cutting() -> None:
    blocked = {(0, 1), (1, 0)}

    assert list(
        _traversable_neighbors(
            (0, 0),
            width=2,
            height=2,
            blocked=blocked,
        )
    ) == []


def test_both_nav2_costmaps_use_the_office_v2_static_geometry() -> None:
    params = _nav2_params()
    scene = yaml.safe_load(
        (ROOT / "scenes" / "office_v2" / "scene.yaml").read_text(
            encoding="utf-8"
        )
    )
    navigation = scene["navigation"]
    for costmap_name in ("local_costmap", "global_costmap"):
        costmap = params[costmap_name][costmap_name]["ros__parameters"]
        assert costmap["plugins"] == ["static_layer", "inflation_layer"]
        assert costmap["static_layer"] == {
            "plugin": "nav2_costmap_2d::StaticLayer",
            "map_subscribe_transient_local": True,
            "subscribe_to_updates": False,
        }
        assert costmap["resolution"] == scene["map"]["resolution"]
        assert costmap["robot_radius"] == navigation["robot_radius_m"]
        assert costmap["footprint_padding"] == navigation["footprint_padding_m"]


def test_evaluation_only_grid_proves_static_column_requires_a_forward_detour(
) -> None:
    """Prove map feasibility without exposing privileged coordinates at runtime."""
    generated = ROOT / "scenes" / "office_v2" / "generated"
    evaluation = json.loads(
        (ROOT / "scenes" / "office_v2" / "evaluation" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    map_yaml = yaml.safe_load(
        (generated / "nav2" / "office_v2.yaml").read_text(encoding="utf-8")
    )
    width, height, pixels = _read_pgm(generated / "nav2" / "office_v2.pgm")
    params = _nav2_params()
    global_costmap = params["global_costmap"]["global_costmap"]["ros__parameters"]

    case = next(
        item
        for item in evaluation["semantic_tests"]
        if item["id"] == "nav_avoid_corridor_column"
    )
    start_pose = next(item for item in evaluation["spawns"] if item["default"])[
        "pose"
    ]
    target_id = case["acceptable_target_ids"][0]
    goal_pose = next(
        item["pose"] for item in evaluation["targets"] if item["id"] == target_id
    )
    resolution = float(map_yaml["resolution"])
    origin = map_yaml["origin"]
    start = _world_to_cell(
        start_pose, origin=origin, resolution=resolution, height=height
    )
    goal = _world_to_cell(
        goal_pose, origin=origin, resolution=resolution, height=height
    )

    # Inflate lethal pixels by the configured circular footprint. Inflation
    # costs outside this radius remain traversable and are left for NavFn's
    # cost objective, matching the runtime static+inflation architecture.
    footprint_radius = (
        float(global_costmap["robot_radius"])
        + float(global_costmap["footprint_padding"])
    )
    radius_cells = math.ceil(footprint_radius / resolution)
    offsets = [
        (dr, dc)
        for dr in range(-radius_cells, radius_cells + 1)
        for dc in range(-radius_cells, radius_cells + 1)
        if math.hypot(dr, dc) * resolution <= footprint_radius
    ]
    lethal = {
        divmod(index, width)
        for index, value in enumerate(pixels)
        if (255 - value) / 255.0 >= float(map_yaml["occupied_thresh"])
    }
    blocked = {
        (row + dr, column + dc)
        for row, column in lethal
        for dr, dc in offsets
        if 0 <= row + dr < height and 0 <= column + dc < width
    }
    assert start not in blocked
    assert goal not in blocked

    # The approved endpoints share a row, and the static column blocks that
    # straight chord. A collision-free grid route must therefore turn around
    # fixed geometry instead of succeeding as a straight/no-op command.
    start_column, goal_column = sorted((start[1], goal[1]))
    assert any(
        (start[0], column) in blocked
        for column in range(start_column, goal_column + 1)
    )

    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    costs = {start: 0.0}
    predecessors: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        _, current = heappop(frontier)
        if current == goal:
            break
        for candidate, step_cost in _traversable_neighbors(
            current,
            width=width,
            height=height,
            blocked=blocked,
        ):
            cost = costs[current] + step_cost
            if cost >= costs.get(candidate, math.inf):
                continue
            costs[candidate] = cost
            predecessors[candidate] = current
            heuristic = math.hypot(goal[0] - candidate[0], goal[1] - candidate[1])
            heappush(frontier, (cost + heuristic, candidate))

    assert goal in costs
    assert costs[goal] > math.hypot(goal[0] - start[0], goal[1] - start[1])

    # Reconstruct the accepted route and re-check every swept grid edge. This
    # keeps a later search refactor from accidentally bypassing the no-corner-
    # cutting rule while still finding the same endpoints.
    route = [goal]
    while route[-1] != start:
        route.append(predecessors[route[-1]])
    route.reverse()
    for current, candidate in zip(route, route[1:]):
        assert candidate not in blocked
        dr = candidate[0] - current[0]
        dc = candidate[1] - current[1]
        if dr and dc:
            assert (current[0] + dr, current[1]) not in blocked
            assert (current[0], current[1] + dc) not in blocked


def test_nav2_runtime_has_no_second_office_map_source() -> None:
    dockerfile = (ROOT / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    launch = (ROOT / "docker" / "nav2" / "nav2_sim_launch.py").read_text(
        encoding="utf-8"
    )
    params = _nav2_params()

    assert "scenes/office_v2/generated/nav2/office_v2.yaml" in dockerfile
    assert "scenes/office_v2/generated/nav2/office_v2.pgm" in dockerfile
    assert "scenes/office_v2/generated/runtime_manifest.json" in dockerfile
    assert 'default_value="/opt/h-ide-nav2/nav2/office_v2.yaml"' in launch
    assert (
        params["map_server"]["ros__parameters"]["yaml_filename"]
        == "/opt/h-ide-nav2/nav2/office_v2.yaml"
    )
    assert dockerfile.count("/opt/h-ide-nav2/nav2/office_v2.yaml") == 2
    assert dockerfile.count("/opt/h-ide-nav2/nav2/office_v2.pgm") == 2
    assert "/opt/h-ide-nav2/maps/" not in dockerfile + launch + json.dumps(params)
    assert "office_floor" not in dockerfile + launch + json.dumps(params)
    assert not (ROOT / "docker" / "nav2" / "generate_office_map.py").exists()
    assert not (ROOT / "docker" / "nav2" / "office_floor.yaml").exists()
    assert not (ROOT / "docker" / "nav2" / "office_floor.pgm").exists()


def test_nav2_docker_context_is_small_and_excludes_privileged_scene_truth() -> None:
    rules = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert rules[0] == "**"
    included = {rule[1:] for rule in rules if rule.startswith("!")}
    assert {
        "docker/nav2/Dockerfile",
        "docker/nav2/behavior_trees/",
        "docker/nav2/behavior_trees/navigate_to_pose_no_reverse.xml",
        "docker/nav2/behavior_trees/navigate_through_poses_no_reverse.xml",
        "docker/nav2/motion_profile_manifest.json",
        "docker/nav2/nav2_params.yaml",
        "docker/nav2/nav2_sim_launch.py",
        "docker/nav2/static_primitive_plan.schema.json",
        "docker/nav2/static_primitive_planner.py",
        "docker/nav2/verify_nav2_runtime.py",
        "docker/nav2/verify_office_v2_bundle.py",
        "scenes/office_v2/generated/nav2/office_v2.yaml",
        "scenes/office_v2/generated/nav2/office_v2.pgm",
        "scenes/office_v2/generated/runtime_manifest.json",
    }.issubset(included)
    assert not any("evaluation" in path for path in included)
    assert "scenes/office_v2/scene.yaml" not in included
    assert not any(path.startswith("vendor/") for path in included)
    assert not any(path.startswith("models/") for path in included)


def test_office_v2_bundle_hash_verifier_fails_closed_on_map_drift(
    tmp_path: Path,
) -> None:
    generated = ROOT / "scenes" / "office_v2" / "generated"
    manifest_path = generated / "runtime_manifest.json"
    map_yaml = generated / "nav2" / "office_v2.yaml"
    map_image = generated / "nav2" / "office_v2.pgm"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verifier = _bundle_verifier()

    verifier.verify_bundle(
        manifest_path,
        map_yaml,
        map_image,
        manifest["scene_source_sha256"],
    )
    with pytest.raises(ValueError, match="does not match scene.yaml"):
        verifier.verify_bundle(
            manifest_path,
            map_yaml,
            map_image,
            "0" * 64,
        )

    drifted_image = tmp_path / "office_v2.pgm"
    body = bytearray(map_image.read_bytes())
    body[-1] ^= 0xFE
    drifted_image.write_bytes(body)
    with pytest.raises(ValueError, match="exact supplied map image"):
        verifier.verify_bundle(
            manifest_path,
            map_yaml,
            drifted_image,
            manifest["scene_source_sha256"],
        )

    copied_generated = tmp_path / "generated"
    copied_nav2 = copied_generated / "nav2"
    copied_nav2.mkdir(parents=True)
    copied_manifest = copied_generated / "runtime_manifest.json"
    copied_yaml = copied_nav2 / "office_v2.yaml"
    copied_image = copied_nav2 / "office_v2.pgm"
    shutil.copyfile(manifest_path, copied_manifest)
    shutil.copyfile(map_yaml, copied_yaml)
    copied_image.write_bytes(body)
    with pytest.raises(ValueError, match="map image does not match"):
        verifier.verify_bundle(
            copied_manifest,
            copied_yaml,
            copied_image,
            manifest["scene_source_sha256"],
        )


def test_workstation_builds_and_checks_the_exact_office_v2_scene_hash() -> None:
    launcher = (ROOT / "scripts" / "start-workstation-demo.sh").read_text(
        encoding="utf-8"
    )
    dockerfile = (ROOT / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert '"${HIDE_DEMO_SIM_ROOT}/tools/build_scene.py" --check' in launcher
    assert '--build-arg "OFFICE_V2_SOURCE_SHA256=${office_v2_source_sha256}"' in launcher
    assert '"${HIDE_DEMO_SIM_ROOT}"' in launcher
    assert "verify_office_v2_bundle.py" in dockerfile
    assert 'org.h-ide.scene.source-sha256="${OFFICE_V2_SOURCE_SHA256}"' in dockerfile
    assert "nav2_image_scene_sha256" in launcher
