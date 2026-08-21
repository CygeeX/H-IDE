#!/usr/bin/env python3
"""Convert a provenance-bound NavFn path into audited motion checkpoints.

The product path must come from Nav2's ComputePathToPose action.  This module
does not plan around obstacles: it only partitions that supplied path into
collision-checked straight segments using the same generated occupancy map.
At execution time every checkpoint must be resolved again from fresh map-frame
odometry; no precomputed relative Spin or DriveOnHeading goal is trusted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence


SCHEMA_VERSION = "h-ide.nav2-static-primitives/1"
RUNTIME_SCHEMA_VERSION = "h-ide.office-v2-runtime/1"
MOTION_PROFILE_SCHEMA_VERSION = "h-ide.nav2-motion-profile/1"
MOTION_MODE = "static_spin_drive_primitives"
COMPUTE_PATH_ACTION = "nav2_msgs/action/ComputePathToPose"
SPIN_ACTION = "nav2_msgs/action/Spin"
DRIVE_ACTION = "nav2_msgs/action/DriveOnHeading"
OFFICIAL_FORWARD_SPEED_MPS = 0.30
ALLOWED_PRODUCT_ACTIONS = [COMPUTE_PATH_ACTION, SPIN_ACTION, DRIVE_ACTION]
FORBIDDEN_PRODUCT_ACTIONS = {
    "nav2_msgs/action/NavigateToPose",
    "nav2_msgs/action/FollowPath",
    "nav2_msgs/action/BackUp",
}
SOURCE_SHA = re.compile(r"scene source sha256:\s*([0-9a-f]{64})")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _finite_number(value: object, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    _require(math.isfinite(result), f"{label} must be a finite number")
    return result


def _yaml_value(text: str, key: str) -> str:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+?)\s*$", text)
    if match is None:
        raise ValueError(f"map YAML is missing {key}")
    return match.group(1)


def _yaml_string(text: str, key: str) -> str:
    value = _yaml_value(text, key).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    _require(bool(value), f"map YAML {key} cannot be empty")
    return value


def _parse_pgm(path: Path) -> tuple[int, int, bytes]:
    raw = path.read_bytes()
    _require(raw.startswith(b"P5\n"), "occupancy image must be binary PGM (P5)")
    try:
        header, pixels = raw.split(b"\n255\n", 1)
        dimensions = [
            line for line in header.splitlines()[1:] if not line.startswith(b"#")
        ][-1]
        width, height = (int(value) for value in dimensions.split())
    except (IndexError, ValueError) as exc:
        raise ValueError("occupancy image has an invalid PGM header") from exc
    _require(width > 0 and height > 0, "occupancy image dimensions must be positive")
    _require(
        len(pixels) == width * height,
        "occupancy image pixel count does not match its dimensions",
    )
    return width, height, pixels


@dataclass(frozen=True)
class StaticMap:
    yaml_path: Path
    image_path: Path
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    base_blocked: frozenset[tuple[int, int]]
    yaml_sha256: str
    image_sha256: str
    scene_source_sha256: str

    @classmethod
    def load(cls, yaml_path: Path, image_path: Path) -> "StaticMap":
        yaml_path = yaml_path.resolve()
        image_path = image_path.resolve()
        text = yaml_path.read_text(encoding="utf-8")
        source_match = SOURCE_SHA.search(text)
        _require(source_match is not None, "map YAML has no scene source hash")

        image_reference = Path(_yaml_string(text, "image"))
        _require(
            not image_reference.is_absolute(),
            "map YAML image reference must be relative",
        )
        _require(
            (yaml_path.parent / image_reference).resolve() == image_path,
            "map YAML does not reference the exact supplied PGM",
        )

        resolution = _finite_number(_yaml_value(text, "resolution"), "map resolution")
        origin = json.loads(_yaml_value(text, "origin"))
        _require(
            isinstance(origin, list) and len(origin) == 3,
            "map YAML origin must contain x, y, and yaw",
        )
        origin_values = tuple(
            _finite_number(value, f"map origin[{index}]")
            for index, value in enumerate(origin)
        )
        _require(
            math.isclose(origin_values[2], 0.0, abs_tol=1e-12),
            "rotated occupancy-map origins are unsupported",
        )
        negate = int(_yaml_value(text, "negate"))
        free_threshold = _finite_number(
            _yaml_value(text, "free_thresh"), "map free threshold"
        )
        occupied_threshold = _finite_number(
            _yaml_value(text, "occupied_thresh"), "map occupied threshold"
        )
        _require(_yaml_string(text, "mode") == "trinary", "map mode must be trinary")
        _require(resolution > 0.0, "map resolution must be positive")
        _require(negate in (0, 1), "map negate must be zero or one")
        _require(
            0.0 <= free_threshold < occupied_threshold <= 1.0,
            "map occupancy thresholds are invalid",
        )

        width, height, pixels = _parse_pgm(image_path)
        blocked: set[tuple[int, int]] = set()
        for index, pixel in enumerate(pixels):
            occupancy = pixel / 255.0 if negate else (255 - pixel) / 255.0
            # NavFn is configured with allow_unknown=false. Treat occupied and
            # trinary/unknown cells as blocked; only definite free space may be
            # used by an audited primitive.
            if occupancy > free_threshold:
                blocked.add(divmod(index, width))
        return cls(
            yaml_path=yaml_path,
            image_path=image_path,
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_values[0],
            origin_y=origin_values[1],
            base_blocked=frozenset(blocked),
            yaml_sha256=_sha256(yaml_path),
            image_sha256=_sha256(image_path),
            scene_source_sha256=source_match.group(1),
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        column = math.floor((x - self.origin_x) / self.resolution)
        map_row = math.floor((y - self.origin_y) / self.resolution)
        row = self.height - 1 - map_row
        _require(
            0 <= row < self.height and 0 <= column < self.width,
            f"path point ({x}, {y}) is outside the occupancy map",
        )
        return row, column

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, column = cell
        return (
            self.origin_x + (column + 0.5) * self.resolution,
            self.origin_y + (self.height - 1 - row + 0.5) * self.resolution,
        )

    def inflated_blocked_cells(
        self,
        clearance_radius: float,
    ) -> frozenset[tuple[int, int]]:
        _require(clearance_radius > 0.0, "clearance radius must be positive")
        radius_cells = math.ceil(clearance_radius / self.resolution)
        offsets = [
            (dr, dc)
            for dr in range(-radius_cells, radius_cells + 1)
            for dc in range(-radius_cells, radius_cells + 1)
            if math.hypot(dr, dc) * self.resolution <= clearance_radius + 1e-12
        ]
        return frozenset(
            (row + dr, column + dc)
            for row, column in self.base_blocked
            for dr, dc in offsets
            if 0 <= row + dr < self.height
            and 0 <= column + dc < self.width
        )

    def segment_cells(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[tuple[int, int], ...]:
        distance = math.dist(start, end)
        sample_spacing = self.resolution / 4.0
        sample_count = max(1, math.ceil(distance / sample_spacing))
        cells: list[tuple[int, int]] = []
        previous: tuple[int, int] | None = None
        for index in range(sample_count + 1):
            ratio = index / sample_count
            cell = self.world_to_cell(
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
            if previous is not None:
                dr = cell[0] - previous[0]
                dc = cell[1] - previous[1]
                _require(
                    abs(dr) <= 1 and abs(dc) <= 1,
                    "segment rasterization skipped an occupancy cell",
                )
                if dr and dc:
                    # Supercover both sides of a diagonal transition so the
                    # swept footprint cannot squeeze through a blocked corner.
                    cells.extend(
                        (
                            (previous[0] + dr, previous[1]),
                            (previous[0], previous[1] + dc),
                        )
                    )
            cells.append(cell)
            previous = cell
        return tuple(dict.fromkeys(cells))

    def segment_is_clear(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        blocked: frozenset[tuple[int, int]],
    ) -> bool:
        return all(cell not in blocked for cell in self.segment_cells(start, end))


def _validate_runtime_bindings(
    static_map: StaticMap,
    runtime_manifest_path: Path,
    motion_profile_manifest_path: Path,
) -> dict[str, object]:
    runtime_manifest_path = runtime_manifest_path.resolve()
    runtime = _read_json_object(runtime_manifest_path, "runtime manifest")
    _require(
        runtime.get("schema_version") == RUNTIME_SCHEMA_VERSION,
        "unsupported Office runtime manifest",
    )
    _require(runtime.get("content_class") == "runtime_geometry", "runtime manifest is not runtime geometry")
    _require(runtime.get("scene_id") == "office_v2", "runtime manifest scene is not office_v2")
    _require(
        runtime.get("scene_source_sha256") == static_map.scene_source_sha256,
        "runtime manifest scene source hash does not match map YAML",
    )
    map_entry = _mapping(runtime.get("map"), "runtime manifest map")
    _require(map_entry.get("frame_id") == "map", "runtime map frame must be map")
    runtime_width = _finite_number(
        map_entry.get("width_cells"), "runtime map width"
    )
    runtime_height = _finite_number(
        map_entry.get("height_cells"), "runtime map height"
    )
    _require(
        runtime_width.is_integer()
        and runtime_height.is_integer()
        and int(runtime_width) == static_map.width
        and int(runtime_height) == static_map.height,
        "runtime map dimensions do not match the supplied PGM",
    )
    _require(
        math.isclose(
            _finite_number(map_entry.get("resolution"), "runtime map resolution"),
            static_map.resolution,
            abs_tol=1e-12,
        ),
        "runtime map resolution does not match map YAML",
    )
    runtime_origin = map_entry.get("origin")
    _require(
        isinstance(runtime_origin, list) and len(runtime_origin) == 3,
        "runtime map origin must contain x, y, and yaw",
    )
    runtime_origin_values = tuple(
        _finite_number(value, f"runtime map origin[{index}]")
        for index, value in enumerate(runtime_origin)
    )
    _require(
        all(
            math.isclose(actual, expected, abs_tol=1e-12)
            for actual, expected in zip(
                runtime_origin_values,
                (static_map.origin_x, static_map.origin_y, 0.0),
            )
        ),
        "runtime map origin does not match map YAML",
    )

    for key, supplied_path, supplied_sha in (
        ("yaml", static_map.yaml_path, static_map.yaml_sha256),
        ("image", static_map.image_path, static_map.image_sha256),
    ):
        artifact = _mapping(map_entry.get(key), f"runtime map {key}")
        artifact_path = artifact.get("path")
        _require(
            isinstance(artifact_path, str) and bool(artifact_path.strip()),
            f"runtime map {key} path must be a non-empty string",
        )
        artifact_relative = Path(artifact_path)
        _require(
            not artifact_relative.is_absolute(),
            f"runtime map {key} path must be relative",
        )
        _require(
            (runtime_manifest_path.parent / artifact_relative).resolve()
            == supplied_path,
            f"runtime manifest does not reference the exact supplied map {key}",
        )
        _require(
            artifact.get("sha256") == supplied_sha,
            f"runtime map {key} hash does not match supplied artifact",
        )

    motion_profile_manifest_path = motion_profile_manifest_path.resolve()
    motion = _read_json_object(
        motion_profile_manifest_path,
        "motion profile manifest",
    )
    _require(
        motion.get("schema_version") == MOTION_PROFILE_SCHEMA_VERSION,
        "unsupported motion profile manifest",
    )
    _require(
        motion.get("product_motion_mode") == MOTION_MODE,
        "motion profile does not select static primitives",
    )
    _require(
        motion.get("follow_path_motion_accepted") is False,
        "unvalidated FollowPath motion must remain disabled",
    )
    _require(
        motion.get("allowed_product_actions") == ALLOWED_PRODUCT_ACTIONS,
        "motion profile action allowlist is not the reviewed primitive chain",
    )
    primitive_contract = _mapping(
        motion.get("primitive_action_contract"),
        "motion profile primitive action contract",
    )
    expected_primitive_contract = {
        "drive_on_heading_speed_mps": OFFICIAL_FORWARD_SPEED_MPS,
        "spin_min_angular_rps": 0.10,
        "spin_max_angular_rps": 0.20,
        "spin_angular_accel_rps2": 0.30,
    }
    _require(
        dict(primitive_contract) == expected_primitive_contract,
        "motion profile primitive action limits do not match the reviewed envelope",
    )
    forbidden = motion.get("forbidden_product_actions")
    _require(isinstance(forbidden, list), "motion profile forbidden actions must be a list")
    _require(
        all(isinstance(action, str) for action in forbidden),
        "motion profile forbidden actions must contain only strings",
    )
    _require(
        FORBIDDEN_PRODUCT_ACTIONS.issubset(set(forbidden)),
        "motion profile does not forbid every unaccepted product action",
    )
    return {
        "runtime_manifest_sha256": _sha256(runtime_manifest_path),
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        "motion_profile_manifest_sha256": _sha256(motion_profile_manifest_path),
        "motion_profile_schema_version": MOTION_PROFILE_SCHEMA_VERSION,
        "follow_path_motion_accepted": False,
    }


def _normalise_path(
    navfn_path: Mapping[str, object],
) -> tuple[str, tuple[dict[str, float], ...]]:
    _require(
        set(navfn_path) == {"frame_id", "poses"},
        "NavFn path must contain only frame_id and poses",
    )
    frame_id = navfn_path.get("frame_id")
    _require(frame_id == "map", "NavFn path frame_id must be map")
    raw_poses = navfn_path.get("poses")
    _require(
        isinstance(raw_poses, Sequence) and not isinstance(raw_poses, (str, bytes)),
        "NavFn path poses must be a sequence",
    )
    _require(len(raw_poses) >= 2, "NavFn path must contain at least two poses")
    poses: list[dict[str, float]] = []
    for index, raw_pose in enumerate(raw_poses):
        _require(isinstance(raw_pose, Mapping), f"path pose {index} is not an object")
        _require(
            set(raw_pose) == {"x", "y", "yaw"},
            f"path pose {index} must contain only x, y, and yaw",
        )
        pose = {
            key: _finite_number(raw_pose[key], f"path pose {index} {key}")
            for key in ("x", "y", "yaw")
        }
        _require(
            -math.pi <= pose["yaw"] <= math.pi,
            f"path pose {index} yaw must be in [-pi, pi]",
        )
        poses.append(pose)
    return frame_id, tuple(poses)


def _normalise_pose(current_pose: Mapping[str, object]) -> dict[str, float]:
    _require(
        set(current_pose) == {"x", "y", "yaw"},
        "fresh odometry pose must contain only x, y, and yaw",
    )
    pose = {
        key: _finite_number(current_pose[key], f"fresh odometry {key}")
        for key in ("x", "y", "yaw")
    }
    _require(-math.pi <= pose["yaw"] <= math.pi, "fresh odometry yaw must be in [-pi, pi]")
    return pose


def _segment_indices(
    static_map: StaticMap,
    poses: tuple[dict[str, float], ...],
    blocked: frozenset[tuple[int, int]],
    *,
    min_segment_m: float,
    max_segment_m: float,
) -> tuple[int, ...]:
    _require(
        0.0 < min_segment_m <= max_segment_m,
        "segment distance bounds are invalid",
    )
    points = tuple((pose["x"], pose["y"]) for pose in poses)
    for index, point in enumerate(points):
        _require(
            static_map.world_to_cell(*point) not in blocked,
            f"NavFn path pose {index} violates inflated clearance",
        )
    for index, (start, end) in enumerate(zip(points, points[1:])):
        _require(
            static_map.segment_is_clear(start, end, blocked),
            f"NavFn path edge {index}->{index + 1} violates inflated clearance",
        )

    # Dynamic programming finds a deterministic minimum-count partition. Every
    # endpoint remains an original NavFn pose, preserving product provenance.
    best: list[tuple[int, ...] | None] = [None] * len(points)
    best[0] = (0,)
    for end_index in range(1, len(points)):
        for start_index in range(end_index):
            prefix = best[start_index]
            if prefix is None:
                continue
            distance = math.dist(points[start_index], points[end_index])
            if not min_segment_m - 1e-12 <= distance <= max_segment_m + 1e-12:
                continue
            if not static_map.segment_is_clear(
                points[start_index],
                points[end_index],
                blocked,
            ):
                continue
            candidate = (*prefix, end_index)
            current = best[end_index]
            if current is None or (len(candidate), candidate) < (
                len(current),
                current,
            ):
                best[end_index] = candidate
    result = best[-1]
    _require(
        result is not None,
        "NavFn path cannot be partitioned into collision-free distance-bounded segments",
    )
    return result


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def build_primitive_plan(
    navfn_path: Mapping[str, object],
    *,
    map_yaml_path: Path,
    map_image_path: Path,
    runtime_manifest_path: Path,
    motion_profile_manifest_path: Path,
    robot_radius_m: float,
    footprint_padding_m: float,
    extra_clearance_m: float = 0.05,
    min_segment_m: float = 0.05,
    max_segment_m: float = 0.75,
    drive_speed_mps: float = OFFICIAL_FORWARD_SPEED_MPS,
    endpoint_tolerance_m: float = 0.03,
    heading_tolerance_rad: float = 0.05,
) -> dict[str, object]:
    values = {
        "robot radius": robot_radius_m,
        "footprint padding": footprint_padding_m,
        "extra clearance": extra_clearance_m,
        "minimum segment": min_segment_m,
        "maximum segment": max_segment_m,
        "drive speed": drive_speed_mps,
        "endpoint tolerance": endpoint_tolerance_m,
        "heading tolerance": heading_tolerance_rad,
    }
    values = {label: _finite_number(value, label) for label, value in values.items()}
    robot_radius_m = values["robot radius"]
    footprint_padding_m = values["footprint padding"]
    extra_clearance_m = values["extra clearance"]
    min_segment_m = values["minimum segment"]
    max_segment_m = values["maximum segment"]
    drive_speed_mps = values["drive speed"]
    endpoint_tolerance_m = values["endpoint tolerance"]
    heading_tolerance_rad = values["heading tolerance"]
    _require(robot_radius_m > 0.0, "robot radius must be positive")
    _require(footprint_padding_m >= 0.0, "footprint padding cannot be negative")
    _require(extra_clearance_m >= 0.05, "extra clearance must be at least 0.05 m")
    _require(0.0 < endpoint_tolerance_m < min_segment_m, "endpoint tolerance must be below minimum segment distance")
    _require(0.0 < heading_tolerance_rad <= 0.05, "heading tolerance must be in (0, 0.05]")
    _require(
        math.isclose(
            drive_speed_mps,
            OFFICIAL_FORWARD_SPEED_MPS,
            abs_tol=1e-12,
        ),
        "DriveOnHeading speed must equal NVIDIA's forward Slow Walk 0.30 m/s",
    )
    frame_id, poses = _normalise_path(navfn_path)
    static_map = StaticMap.load(map_yaml_path, map_image_path)
    runtime_provenance = _validate_runtime_bindings(
        static_map,
        runtime_manifest_path,
        motion_profile_manifest_path,
    )
    clearance_radius = robot_radius_m + footprint_padding_m + extra_clearance_m
    blocked = static_map.inflated_blocked_cells(clearance_radius)
    endpoints = _segment_indices(
        static_map,
        poses,
        blocked,
        min_segment_m=min_segment_m,
        max_segment_m=max_segment_m,
    )

    segments: list[dict[str, object]] = []
    for segment_index, (start_index, end_index) in enumerate(
        zip(endpoints, endpoints[1:])
    ):
        start = poses[start_index]
        end = poses[end_index]
        heading = math.atan2(end["y"] - start["y"], end["x"] - start["x"])
        distance = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
        segments.append(
            {
                "id": f"segment-{segment_index:03d}",
                "path_start_index": start_index,
                "path_end_index": end_index,
                "start_map": {"x": start["x"], "y": start["y"]},
                "end_map": {"x": end["x"], "y": end["y"]},
                "distance_m": distance,
                "heading_map_yaw_rad": heading,
            }
        )

    # These are checkpoints, not executable relative goals. The live resolver
    # must use fresh odometry and the absolute endpoint immediately before each
    # official Nav2 action. This also ensures a zero Spin is represented as an
    # explicit skip rather than sent to the action server.
    primitives: list[dict[str, object]] = []
    for segment in segments:
        primitives.append(
            {
                "sequence": len(primitives),
                "kind": "spin",
                "checkpoint_role": "align_segment",
                "action_type": SPIN_ACTION,
                "segment_id": segment["id"],
                "path_index": segment["path_start_index"],
                "reference_end_map": dict(segment["end_map"]),
                "planned_target_map_yaw_rad": segment["heading_map_yaw_rad"],
                "requires_fresh_odometry": True,
            }
        )
        primitives.append(
            {
                "sequence": len(primitives),
                "kind": "drive_on_heading",
                "checkpoint_role": "reach_segment_endpoint",
                "action_type": DRIVE_ACTION,
                "segment_id": segment["id"],
                "path_end_index": segment["path_end_index"],
                "end_map": dict(segment["end_map"]),
                "speed_mps": drive_speed_mps,
                "requires_fresh_odometry": True,
            }
        )
    primitives.append(
        {
            "sequence": len(primitives),
            "kind": "spin",
            "checkpoint_role": "final_heading",
            "action_type": SPIN_ACTION,
            "path_index": len(poses) - 1,
            "reference_end_map": {"x": poses[-1]["x"], "y": poses[-1]["y"]},
            "planned_target_map_yaw_rad": poses[-1]["yaw"],
            "requires_fresh_odometry": True,
        }
    )

    normalised_path = {"frame_id": frame_id, "poses": list(poses)}
    plan: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "motion_mode": MOTION_MODE,
        "planner_source": {
            "action_type": COMPUTE_PATH_ACTION,
            "planner_id": "GridBased",
            "plugin": "nav2_navfn_planner::NavfnPlanner",
            "navfn_path_sha256": _canonical_sha256(normalised_path),
            "navfn_path_pose_count": len(poses),
        },
        "map_provenance": {
            "yaml_sha256": static_map.yaml_sha256,
            "image_sha256": static_map.image_sha256,
            "scene_source_sha256": static_map.scene_source_sha256,
        },
        "runtime_provenance": runtime_provenance,
        "segmenter": {
            "source_sha256": _sha256(Path(__file__).resolve()),
            "robot_radius_m": robot_radius_m,
            "footprint_padding_m": footprint_padding_m,
            "extra_clearance_m": extra_clearance_m,
            "clearance_radius_m": clearance_radius,
            "min_segment_m": min_segment_m,
            "max_segment_m": max_segment_m,
        },
        "execution_policy": {
            "fresh_odometry_required": True,
            "fresh_pose_frame": "map",
            "revalidate_line_of_sight_before_drive": True,
            "endpoint_tolerance_m": endpoint_tolerance_m,
            "heading_tolerance_rad": heading_tolerance_rad,
            "drive_speed_mps": drive_speed_mps,
            "spin_min_angular_rps": 0.10,
            "spin_max_angular_rps": 0.20,
            "spin_angular_accel_rps2": 0.30,
            "zero_spin_policy": "skip_within_heading_tolerance",
            "short_residual_policy": "skip_within_endpoint_tolerance_else_fail_below_min_segment",
        },
        "segment_endpoint_indices": list(endpoints),
        "segments": segments,
        "primitives": primitives,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def verify_primitive_plan(
    plan: Mapping[str, object],
    navfn_path: Mapping[str, object],
    *,
    map_yaml_path: Path,
    map_image_path: Path,
    runtime_manifest_path: Path,
    motion_profile_manifest_path: Path,
    robot_radius_m: float,
    footprint_padding_m: float,
    extra_clearance_m: float = 0.05,
    min_segment_m: float = 0.05,
    max_segment_m: float = 0.75,
    drive_speed_mps: float = OFFICIAL_FORWARD_SPEED_MPS,
    endpoint_tolerance_m: float = 0.03,
    heading_tolerance_rad: float = 0.05,
) -> dict[str, object]:
    """Rebuild a plan from trusted inputs and reject any drift or tampering."""
    _require(isinstance(plan, Mapping), "primitive plan must be an object")
    supplied = dict(plan)
    supplied_sha = supplied.pop("plan_sha256", None)
    _require(
        isinstance(supplied_sha, str) and supplied_sha == _canonical_sha256(supplied),
        "primitive plan SHA-256 is invalid",
    )
    expected = build_primitive_plan(
        navfn_path,
        map_yaml_path=map_yaml_path,
        map_image_path=map_image_path,
        runtime_manifest_path=runtime_manifest_path,
        motion_profile_manifest_path=motion_profile_manifest_path,
        robot_radius_m=robot_radius_m,
        footprint_padding_m=footprint_padding_m,
        extra_clearance_m=extra_clearance_m,
        min_segment_m=min_segment_m,
        max_segment_m=max_segment_m,
        drive_speed_mps=drive_speed_mps,
        endpoint_tolerance_m=endpoint_tolerance_m,
        heading_tolerance_rad=heading_tolerance_rad,
    )
    _require(dict(plan) == expected, "primitive plan does not match trusted inputs")
    return expected


def _segment_for_primitive(
    plan: Mapping[str, object],
    primitive: Mapping[str, object],
) -> Mapping[str, object]:
    segment_id = primitive.get("segment_id")
    segments = plan["segments"]
    _require(isinstance(segments, list), "primitive plan segments must be a list")
    matches = [
        segment
        for segment in segments
        if isinstance(segment, Mapping) and segment.get("id") == segment_id
    ]
    _require(len(matches) == 1, "primitive segment reference is invalid")
    return matches[0]


def _fresh_segment_geometry(
    static_map: StaticMap,
    blocked: frozenset[tuple[int, int]],
    segment: Mapping[str, object],
    current_pose: Mapping[str, float],
    *,
    endpoint_tolerance_m: float,
    min_segment_m: float,
    max_segment_m: float,
) -> tuple[float, float]:
    endpoint = _mapping(segment.get("end_map"), "segment endpoint")
    end = (
        _finite_number(endpoint.get("x"), "segment endpoint x"),
        _finite_number(endpoint.get("y"), "segment endpoint y"),
    )
    start = (current_pose["x"], current_pose["y"])
    _require(
        static_map.segment_is_clear(start, end, blocked),
        "fresh odometry to segment endpoint violates inflated clearance; replan required",
    )
    distance = math.dist(start, end)
    if distance <= endpoint_tolerance_m + 1e-12:
        return distance, current_pose["yaw"]
    _require(
        distance >= min_segment_m - 1e-12,
        "fresh residual is outside endpoint tolerance but below minimum segment; replan required",
    )
    _require(
        distance <= max_segment_m + 1e-12,
        "fresh residual exceeds maximum segment distance; replan required",
    )
    return distance, math.atan2(end[1] - start[1], end[0] - start[0])


def resolve_live_primitive(
    plan: Mapping[str, object],
    primitive_index: int,
    navfn_path: Mapping[str, object],
    current_pose: Mapping[str, object],
    *,
    map_yaml_path: Path,
    map_image_path: Path,
    runtime_manifest_path: Path,
    motion_profile_manifest_path: Path,
    robot_radius_m: float,
    footprint_padding_m: float,
    extra_clearance_m: float = 0.05,
    min_segment_m: float = 0.05,
    max_segment_m: float = 0.75,
    drive_speed_mps: float = OFFICIAL_FORWARD_SPEED_MPS,
    endpoint_tolerance_m: float = 0.03,
    heading_tolerance_rad: float = 0.05,
) -> dict[str, object]:
    """Resolve one checkpoint into an official action goal or an explicit skip.

    This function is intentionally pure and does not send a ROS action. A live
    runner must call it with a newly sampled map-frame odometry pose immediately
    before each checkpoint.
    """
    verified = verify_primitive_plan(
        plan,
        navfn_path,
        map_yaml_path=map_yaml_path,
        map_image_path=map_image_path,
        runtime_manifest_path=runtime_manifest_path,
        motion_profile_manifest_path=motion_profile_manifest_path,
        robot_radius_m=robot_radius_m,
        footprint_padding_m=footprint_padding_m,
        extra_clearance_m=extra_clearance_m,
        min_segment_m=min_segment_m,
        max_segment_m=max_segment_m,
        drive_speed_mps=drive_speed_mps,
        endpoint_tolerance_m=endpoint_tolerance_m,
        heading_tolerance_rad=heading_tolerance_rad,
    )
    _require(
        isinstance(primitive_index, int) and not isinstance(primitive_index, bool),
        "primitive index must be an integer",
    )
    primitives = verified["primitives"]
    _require(
        0 <= primitive_index < len(primitives),
        "primitive index is outside the plan",
    )
    primitive = primitives[primitive_index]
    _require(isinstance(primitive, Mapping), "primitive must be an object")
    pose = _normalise_pose(current_pose)
    static_map = StaticMap.load(map_yaml_path, map_image_path)
    blocked = static_map.inflated_blocked_cells(
        robot_radius_m + footprint_padding_m + extra_clearance_m
    )

    result: dict[str, object] = {
        "primitive_sequence": primitive_index,
        "plan_sha256": verified["plan_sha256"],
        "fresh_odometry_used": True,
    }
    role = primitive.get("checkpoint_role")
    if primitive.get("kind") == "spin" and role == "final_heading":
        reference = _mapping(primitive.get("reference_end_map"), "final reference endpoint")
        final_endpoint = (
            _finite_number(reference.get("x"), "final endpoint x"),
            _finite_number(reference.get("y"), "final endpoint y"),
        )
        current_xy = (pose["x"], pose["y"])
        _require(
            static_map.segment_is_clear(current_xy, final_endpoint, blocked),
            "fresh odometry at final checkpoint violates inflated clearance",
        )
        distance = math.hypot(
            final_endpoint[0] - pose["x"],
            final_endpoint[1] - pose["y"],
        )
        _require(
            distance <= endpoint_tolerance_m + 1e-12,
            "final heading checkpoint reached before endpoint; replan required",
        )
        target_yaw = _finite_number(
            primitive.get("planned_target_map_yaw_rad"),
            "final target map yaw",
        )
        delta = _wrap_angle(target_yaw - pose["yaw"])
        if abs(delta) <= heading_tolerance_rad + 1e-12:
            return {
                **result,
                "disposition": "skip_within_heading_tolerance",
                "action_goal": None,
            }
        return {
            **result,
            "disposition": "send_action",
            "action_type": SPIN_ACTION,
            "action_goal": {"target_yaw_rad": delta},
            "resolved_target_map_yaw_rad": target_yaw,
        }

    segment = _segment_for_primitive(verified, primitive)
    distance, target_heading = _fresh_segment_geometry(
        static_map,
        blocked,
        segment,
        pose,
        endpoint_tolerance_m=endpoint_tolerance_m,
        min_segment_m=min_segment_m,
        max_segment_m=max_segment_m,
    )
    if distance <= endpoint_tolerance_m + 1e-12:
        return {
            **result,
            "disposition": "skip_within_endpoint_tolerance",
            "action_goal": None,
            "fresh_distance_to_endpoint_m": distance,
        }

    heading_error = _wrap_angle(target_heading - pose["yaw"])
    if primitive.get("kind") == "spin" and role == "align_segment":
        if abs(heading_error) <= heading_tolerance_rad + 1e-12:
            return {
                **result,
                "disposition": "skip_within_heading_tolerance",
                "action_goal": None,
                "fresh_distance_to_endpoint_m": distance,
                "resolved_target_map_yaw_rad": target_heading,
            }
        return {
            **result,
            "disposition": "send_action",
            "action_type": SPIN_ACTION,
            "action_goal": {"target_yaw_rad": heading_error},
            "fresh_distance_to_endpoint_m": distance,
            "resolved_target_map_yaw_rad": target_heading,
        }

    _require(
        primitive.get("kind") == "drive_on_heading"
        and role == "reach_segment_endpoint",
        "unsupported primitive checkpoint",
    )
    _require(
        abs(heading_error) <= heading_tolerance_rad + 1e-12,
        "DriveOnHeading checkpoint is not aligned; resolve Spin again",
    )
    _require(
        math.isclose(
            _finite_number(primitive.get("speed_mps"), "DriveOnHeading speed"),
            OFFICIAL_FORWARD_SPEED_MPS,
            abs_tol=1e-12,
        ),
        "DriveOnHeading speed is not NVIDIA's forward Slow Walk command",
    )
    return {
        **result,
        "disposition": "send_action",
        "action_type": DRIVE_ACTION,
        "action_goal": {
            "target": {"x": distance, "y": 0.0, "z": 0.0},
            "speed_mps": OFFICIAL_FORWARD_SPEED_MPS,
        },
        "fresh_distance_to_endpoint_m": distance,
        "resolved_target_map_yaw_rad": target_heading,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navfn-path", type=Path, required=True)
    parser.add_argument("--map-yaml", type=Path, required=True)
    parser.add_argument("--map-image", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--motion-profile-manifest", type=Path, required=True)
    parser.add_argument("--robot-radius-m", type=float, required=True)
    parser.add_argument("--footprint-padding-m", type=float, required=True)
    parser.add_argument("--extra-clearance-m", type=float, default=0.05)
    parser.add_argument("--min-segment-m", type=float, default=0.05)
    parser.add_argument("--max-segment-m", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        navfn_path = json.loads(args.navfn_path.read_text(encoding="utf-8"))
        _require(isinstance(navfn_path, Mapping), "NavFn path JSON must be an object")
        plan = build_primitive_plan(
            navfn_path,
            map_yaml_path=args.map_yaml,
            map_image_path=args.map_image,
            runtime_manifest_path=args.runtime_manifest,
            motion_profile_manifest_path=args.motion_profile_manifest,
            robot_radius_m=args.robot_radius_m,
            footprint_padding_m=args.footprint_padding_m,
            extra_clearance_m=args.extra_clearance_m,
            min_segment_m=args.min_segment_m,
            max_segment_m=args.max_segment_m,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Static primitive planning failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(plan, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
