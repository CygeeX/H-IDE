#!/usr/bin/env python3
"""Offline, fail-closed acceptance for isolated locomotion primitives.

This evaluator has no network or control client.  It reads one explicit JSON
trace and either prints a result or writes it to an explicit output path.  It
never imports H-IDE/Mission code and cannot arm, move, stop, or reset a robot.

The trace contract intentionally separates collection from judgment:

* ``samples`` contain root pose/velocity normalized from the Plant's verbatim
  ``dynamicsEvidence.rootState`` at its own physics ``sampleSequence``, its
  authoritative COM and contact counters, confirmed support-off state,
  command/action state, and continuously advancing health for the motion core
  and both viewers.  The independent render ``plantSequence`` remains the
  first-person/native-viewer freshness boundary and is never conflated with
  the physics-step sequence.
  RobotAgent/simGateway are instead proven inactive, with a single stable
  observed ``/cmd_vel`` publisher, at preflight and every sample.
* ``armLimiterEvidence`` is a complete successful-session trace emitted by the
  authoritative SONIC 50 Hz control loop, with per-joint continuous saturation
  durations. Slow HTTP samples, interpolation, and repeated telemetry fail.
* ``action`` contains the issued primitive goal, activation provenance, and a
  canonical SHA-256.  Detours additionally contain a hashed, ordered sequence
  of official ``Spin``/``DriveOnHeading`` actions; ``NavigateToPose`` and
  ``FollowPath`` are not accepted as execution evidence.
* ``nav2Plan`` is mandatory only for ``detour`` and must be an observed
  ``nav_msgs/Path`` with collision geometry and canonical hashes.

Formal thresholds are code-owned.  The input cannot weaken them.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


SCHEMA_VERSION = "h-ide.strict-motion-trace/2"
RESULT_SCHEMA_VERSION = "h-ide.strict-motion-acceptance/1"
PLANT_DYNAMICS_SCHEMA_VERSION = "h-ide.mujoco-dynamics-evidence/1"
CONTACT_SCHEMA_VERSION = "h-ide.mujoco-contact-evidence/1"
ARM_LIMITER_SCHEMA_VERSION = "h-ide.sonic-arm-limiter-success-trace/1"
STATIC_PRIMITIVE_PLAN_SCHEMA_VERSION = "h-ide.nav2-static-primitives/1"
MAX_DRIVE_REALIGNMENT_SPINS = 4
MAX_DRIVE_REALIGNMENT_RETAINED_ERROR_RATIO = 0.90
ARM_JOINT_INDICES = tuple(range(15, 29))
MODES = (
    "forward_probe",
    "forward",
    "forward_3m",
    "reverse",
    "lateral_left",
    "lateral_right",
    "turn_left",
    "turn_right",
    "detour",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1e-9
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_STATIC_PRIMITIVE_PLANNER_PATH = (
    _REPOSITORY_ROOT / "docker" / "nav2" / "static_primitive_planner.py"
)
_OFFICE_MAP_YAML_PATH = (
    _REPOSITORY_ROOT
    / "scenes"
    / "office_v2"
    / "generated"
    / "nav2"
    / "office_v2.yaml"
)
_OFFICE_MAP_IMAGE_PATH = _OFFICE_MAP_YAML_PATH.with_suffix(".pgm")
_OFFICE_RUNTIME_MANIFEST_PATH = (
    _REPOSITORY_ROOT
    / "scenes"
    / "office_v2"
    / "generated"
    / "runtime_manifest.json"
)
_MOTION_PROFILE_MANIFEST_PATH = (
    _REPOSITORY_ROOT / "docker" / "nav2" / "motion_profile_manifest.json"
)
_STATIC_PLANNER_MODULE: Any | None = None
_FIRST_PERSON_VIEWER_HEALTH_SCHEMA_VERSION = (
    "h-ide.first-person-viewer-health/1"
)


def _production_static_planner() -> Any:
    """Load the production pure planner/resolver, never a copied evaluator model."""

    global _STATIC_PLANNER_MODULE
    if _STATIC_PLANNER_MODULE is not None:
        return _STATIC_PLANNER_MODULE
    spec = importlib.util.spec_from_file_location(
        "_hide_strict_acceptance_static_primitive_planner",
        _STATIC_PRIMITIVE_PLANNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise TraceContractError("production static primitive planner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _STATIC_PLANNER_MODULE = module
    return module


@dataclass(frozen=True, slots=True)
class FormalContract:
    expected_longitudinal_m: float | None = None
    expected_lateral_m: float | None = None
    expected_yaw_rad: float | None = None
    final_position_tolerance_m: float = 0.15
    # Nav2 DriveOnHeading emits the official constant-speed primitive until
    # odometry reaches its target.  SONIC then needs a bounded gait-braking
    # interval before the balanced IDLE settles.  Keep final accuracy at 15 cm
    # while allowing at most 18 cm transient longitudinal travel; collision,
    # cross-track, pose-jump, direction, and terminal gates remain independent.
    max_linear_overshoot_m: float = 0.18
    final_yaw_tolerance_rad: float = 0.35
    max_cross_track_m: float = 0.12
    max_heading_error_rad: float = 0.20
    # LateralStep hands the actuator owner from the commanded gait to learned
    # IDLE before the body has fully shed its lateral momentum.  Keep the
    # commanded phase and final settled pose on the strict 12 cm / 10 degree
    # contract, while admitting only the measured, short braking transient
    # seen between command-zero and the terminal settle window.
    max_lateral_braking_cross_track_m: float = 0.13
    max_lateral_braking_heading_error_rad: float = math.radians(12.0)
    max_opposite_motion_m: float = 0.04
    max_turn_translation_m: float = 0.15
    max_turn_reverse_rad: float = 0.05
    max_turn_overshoot_rad: float = 0.35
    # As with lateral gait, learned IDLE brakes the whole body after Nav2 Spin
    # withdraws its command.  The active and final contracts remain at the
    # existing 0.35 rad bound; only the measured command-zero transient gets
    # a separately reported 22 degree envelope.
    max_turn_braking_overshoot_rad: float = math.radians(22.0)
    progress_window_s: float = 15.0
    min_linear_progress_m: float = 0.10
    min_turn_progress_rad: float = 0.35
    min_root_z_m: float = 0.55
    max_root_z_m: float = 1.15
    max_roll_pitch_rad: float = 0.25
    max_position_jump_m: float = 0.08
    max_com_jump_m: float = 0.08
    max_yaw_jump_rad: float = 0.20
    max_root_com_delta_disagreement_m: float = 0.15
    terminal_settle_s: float = 1.0
    terminal_linear_speed_mps: float = 0.01
    # Dynamic SONIC IDLE retains actuator authority and makes small balance
    # corrections; the untethered live envelope peaks around 0.015 rad/s while
    # net pose drift remains millimetric.  This is an acceptance threshold,
    # not an actuator safety limit.
    terminal_angular_speed_rps: float = 0.04
    # The collector joins six independently scheduled health sources and the
    # 5 Hz first-person viewer.  Permit one 100 ms scheduling margin beyond its
    # nominal 200 ms frame period.  Every sample must still carry a strictly
    # newer viewer sequence and the actuator trace remains authoritative at
    # 50 Hz, so this does not admit repeated/stale visual evidence.
    max_terminal_sample_gap_s: float = 0.30
    max_arm_continuous_saturation_s: float = 0.24
    arm_must_clear_before_s: float = 0.25
    arm_control_period_s: float = 0.02
    command_zero_epsilon: float = 1e-6
    # ROS Twist uses float64, but several official Nav2 behavior parameters and
    # intermediate messages originate as float32.  Admit only serialization
    # noise around a code-owned cap (0.3f is 0.30000001192092896), not a
    # materially faster command.
    command_serialization_tolerance: float = 1e-6
    steady_target_tolerance: float = 0.01
    drive_on_heading_max_vx_mps: float = 0.30
    lateral_step_max_vy_mps: float = 0.40
    spin_max_wz_rps: float = 0.20
    max_fresh_odometry_age_s: float = 0.10
    max_first_person_capture_age_ms: float = 2000.0
    # The official constant-speed primitive hands off to learned IDLE before
    # the whole-body gait has fully braked. Accept the same 15 cm terminal
    # position envelope used by the motion contract, while the independent
    # 0.42 m inflated-clearance and contact gates remain strict.
    detour_endpoint_tolerance_m: float = 0.15
    # Nav2's official Spin has a 0.10 rad/s minimum command.  Live column-route
    # r142 showed that separate +/-0.07..0.08 rad learned whole-body Spins are
    # less precise than driving the already-clear short segment.  Pairing this
    # tolerance with 0.50 m segments keeps the drift geometrically bounded:
    # 0.50 * sin(0.09) = 0.04494 m, below the 0.05 m clearance reserve.
    detour_heading_tolerance_rad: float = 0.09
    robot_radius_m: float = 0.35
    footprint_padding_m: float = 0.02
    map_resolution_m: float = 0.05

    @property
    def required_obstacle_boundary_clearance_m(self) -> float:
        return self.robot_radius_m + self.footprint_padding_m + self.map_resolution_m

    @property
    def realized_obstacle_boundary_clearance_m(self) -> float:
        # Continuous root-vs-analytic-obstacle evidence has no occupancy-cell
        # quantization error.  Keep the full robot radius and configured
        # footprint padding; the extra map cell remains mandatory for the
        # raster NavFn plan, not for the measured physical trajectory.
        return self.robot_radius_m + self.footprint_padding_m


FORMAL_CONTRACTS: dict[str, FormalContract] = {
    "forward_probe": FormalContract(
        expected_longitudinal_m=0.20,
        final_position_tolerance_m=0.05,
        max_linear_overshoot_m=0.08,
        min_linear_progress_m=0.03,
    ),
    "forward": FormalContract(expected_longitudinal_m=1.0),
    "forward_3m": FormalContract(expected_longitudinal_m=3.0),
    "reverse": FormalContract(
        expected_longitudinal_m=-1.0,
        drive_on_heading_max_vx_mps=0.20,
    ),
    "lateral_left": FormalContract(
        expected_lateral_m=0.5,
        final_position_tolerance_m=0.10,
        max_heading_error_rad=math.radians(10.0),
    ),
    "lateral_right": FormalContract(
        expected_lateral_m=-0.5,
        final_position_tolerance_m=0.10,
        max_heading_error_rad=math.radians(10.0),
    ),
    "turn_left": FormalContract(expected_yaw_rad=math.pi / 2.0),
    "turn_right": FormalContract(expected_yaw_rad=-math.pi / 2.0),
    "detour": FormalContract(final_position_tolerance_m=0.18),
}

_EXPECTED_ACTION_TYPE = {
    "forward_probe": "DriveOnHeading",
    "forward": "DriveOnHeading",
    "forward_3m": "DriveOnHeading",
    "reverse": "DriveOnHeading",
    "lateral_left": "LateralStep",
    "lateral_right": "LateralStep",
    "turn_left": "Spin",
    "turn_right": "Spin",
    "detour": "StaticDetourPrimitives",
}
# ``motion_core`` deliberately excludes RobotAgent/simGateway command
# ownership.  They must be continuously proven inactive instead of being run
# in SAFE_HOLD and bypassed by the strict runner.
_REQUIRED_RUNTIME_COMPONENTS = (
    "plant",
    "sonic",
    "nav2",
    "rosBridge",
    "nativeViewer",
    "firstPersonViewer",
)
_ISOLATED_COMMAND_COMPONENTS = ("robotAgent", "simGateway")
_MOTION_CORE_TIER = "motion_core"
_STRICT_RUNNER_IDENTITY = "strict_motion_runner"


@dataclass(frozen=True, slots=True)
class Failure:
    code: str
    detail: str
    first_observed_s: float | None = None


class TraceContractError(ValueError):
    """The evidence document is incomplete or structurally invalid."""


def canonical_sha256(value: Any) -> str:
    """Return the canonical hash used for goals, plans, geometry, and traces."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _production_plan_kwargs_from_tuple(
    parameters: tuple[float, ...],
) -> dict[str, Any]:
    (
        robot_radius_m,
        footprint_padding_m,
        extra_clearance_m,
        min_segment_m,
        max_segment_m,
        drive_speed_mps,
        endpoint_tolerance_m,
        heading_tolerance_rad,
    ) = parameters
    return {
        "map_yaml_path": _OFFICE_MAP_YAML_PATH,
        "map_image_path": _OFFICE_MAP_IMAGE_PATH,
        "runtime_manifest_path": _OFFICE_RUNTIME_MANIFEST_PATH,
        "motion_profile_manifest_path": _MOTION_PROFILE_MANIFEST_PATH,
        "robot_radius_m": robot_radius_m,
        "footprint_padding_m": footprint_padding_m,
        "extra_clearance_m": extra_clearance_m,
        "min_segment_m": min_segment_m,
        "max_segment_m": max_segment_m,
        "drive_speed_mps": drive_speed_mps,
        "endpoint_tolerance_m": endpoint_tolerance_m,
        "heading_tolerance_rad": heading_tolerance_rad,
    }


@lru_cache(maxsize=64)
def _verify_production_static_plan_cached(
    plan_json: str,
    navfn_path_json: str,
    parameters: tuple[float, ...],
) -> Mapping[str, Any]:
    return _production_static_planner().verify_primitive_plan(
        json.loads(plan_json),
        json.loads(navfn_path_json),
        **_production_plan_kwargs_from_tuple(parameters),
    )


@lru_cache(maxsize=512)
def _resolve_production_static_primitive_cached(
    plan_json: str,
    navfn_path_json: str,
    primitive_sequence: int,
    fresh_pose_json: str,
    parameters: tuple[float, ...],
) -> Mapping[str, Any]:
    return _production_static_planner().resolve_live_primitive(
        json.loads(plan_json),
        primitive_sequence,
        json.loads(navfn_path_json),
        json.loads(fresh_pose_json),
        **_production_plan_kwargs_from_tuple(parameters),
    )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TraceContractError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TraceContractError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise TraceContractError(f"{label} must be a finite number")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceContractError(f"{label} must be a non-negative integer")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceContractError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TraceContractError(f"{label} must be an array")
    return value


def _xyz(value: Any, label: str) -> tuple[float, float, float]:
    obj = _mapping(value, label)
    return tuple(_finite(obj[name], f"{label}.{name}") for name in ("x", "y", "z"))  # type: ignore[return-value]


def _angle_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_to_rpy(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> tuple[float, float, float]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < _EPS:
        raise TraceContractError("Plant root quaternion cannot be zero")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return roll, pitch, yaw


def _point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= _EPS:
        return math.hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> bool:
    return (
        min(ax, bx) - _EPS <= px <= max(ax, bx) + _EPS
        and min(ay, by) - _EPS <= py <= max(ay, by) + _EPS
        and abs(_orientation(ax, ay, bx, by, px, py)) <= _EPS
    )


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _orientation(*a, *b, *c)
    o2 = _orientation(*a, *b, *d)
    o3 = _orientation(*c, *d, *a)
    o4 = _orientation(*c, *d, *b)
    if (o1 > _EPS and o2 < -_EPS or o1 < -_EPS and o2 > _EPS) and (
        o3 > _EPS and o4 < -_EPS or o3 < -_EPS and o4 > _EPS
    ):
        return True
    return any(
        (
            abs(value) <= _EPS
            and _on_segment(*first, *second, *point)
        )
        for value, first, second, point in (
            (o1, a, b, c),
            (o2, a, b, d),
            (o3, c, d, a),
            (o4, c, d, b),
        )
    )


def _segment_distance(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(*a, *c, *d),
        _point_segment_distance(*b, *c, *d),
        _point_segment_distance(*c, *a, *b),
        _point_segment_distance(*d, *a, *b),
    )


def _segment_obstacle_clearance(
    a: tuple[float, float],
    b: tuple[float, float],
    obstacle: Mapping[str, Any],
) -> float:
    center_values = _sequence(obstacle.get("center"), "obstacle.center")
    if len(center_values) != 2:
        raise TraceContractError("obstacle.center must contain two numbers")
    cx, cy = (_finite(center_values[i], f"obstacle.center[{i}]") for i in range(2))
    shape = obstacle.get("shape")
    if shape == "circle":
        radius = _finite(obstacle.get("radius"), "obstacle.radius")
        if radius <= 0:
            raise TraceContractError("obstacle.radius must be positive")
        return max(0.0, _point_segment_distance(cx, cy, *a, *b) - radius)
    if shape != "box":
        raise TraceContractError("obstacle.shape must be circle or box")
    size = _sequence(obstacle.get("size"), "obstacle.size")
    if len(size) != 2:
        raise TraceContractError("obstacle.size must contain two numbers")
    width, depth = (_finite(size[i], f"obstacle.size[{i}]") for i in range(2))
    if width <= 0 or depth <= 0:
        raise TraceContractError("obstacle.size values must be positive")
    yaw = _finite(obstacle.get("yaw", 0.0), "obstacle.yaw")
    cosine, sine = math.cos(yaw), math.sin(yaw)

    def local(point: tuple[float, float]) -> tuple[float, float]:
        dx, dy = point[0] - cx, point[1] - cy
        return cosine * dx + sine * dy, -sine * dx + cosine * dy

    la, lb = local(a), local(b)
    hx, hy = width / 2.0, depth / 2.0
    if (
        -hx <= la[0] <= hx and -hy <= la[1] <= hy
    ) or (-hx <= lb[0] <= hx and -hy <= lb[1] <= hy):
        return 0.0
    corners = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    edges = tuple(zip(corners, corners[1:] + corners[:1]))
    return min(_segment_distance(la, lb, edge[0], edge[1]) for edge in edges)


def _path_clearance(
    path: Sequence[tuple[float, float]],
    obstacles: Sequence[Mapping[str, Any]],
) -> float:
    if len(path) < 2:
        raise TraceContractError("a path must contain at least two points")
    if not obstacles:
        raise TraceContractError("detour collision geometry cannot be empty")
    return min(
        _segment_obstacle_clearance(a, b, obstacle)
        for a, b in zip(path, path[1:])
        for obstacle in obstacles
    )


def _path_length(path: Sequence[tuple[float, float]]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


class StrictMotionEvaluator:
    """Pure evaluator for one already-recorded primitive trace."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document = document
        self.failures: dict[str, Failure] = {}
        self.mode = str(document.get("mode", ""))
        self.contract = FORMAL_CONTRACTS.get(self.mode)
        self.samples: list[dict[str, Any]] = []
        self.arm_evidence: dict[str, Any] = {}
        self.arm_trace: list[dict[str, Any]] = []
        self.detour_segments: list[dict[str, Any]] = []
        self.isolation_manifest: dict[str, Any] = {}
        self.isolation_manifest_hash = ""
        self.metrics: dict[str, Any] = {}

    def _fail(self, code: str, detail: str, observed_s: float | None = None) -> None:
        self.failures.setdefault(code, Failure(code, detail, observed_s))

    def evaluate(self) -> dict[str, Any]:
        try:
            self._validate_document()
            self._evaluate_samples()
            self._evaluate_arm_trace()
            self._evaluate_action()
            self._evaluate_detour_plan()
            self._evaluate_terminal_settle()
        except TraceContractError as exc:
            self._fail("TRACE_SCHEMA_INVALID", str(exc))

        input_hash = canonical_sha256(self.document)
        evaluator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        failures = [asdict(item) for item in self.failures.values()]
        return {
            "schemaVersion": RESULT_SCHEMA_VERSION,
            "pass": not failures,
            "readOnly": True,
            "offline": True,
            "tier": self.document.get("tier"),
            "fullChainClaimed": False,
            "mode": self.mode,
            "caseId": self.document.get("caseId"),
            "formalContract": None if self.contract is None else asdict(self.contract),
            "inputSha256": input_hash,
            "evaluatorSha256": evaluator_hash,
            "sampleCount": len(self.samples),
            "armLimiterSampleCount": len(self.arm_trace),
            "metrics": self.metrics,
            "failures": failures,
        }

    def _validate_document(self) -> None:
        if self.document.get("schemaVersion") != SCHEMA_VERSION:
            raise TraceContractError(f"schemaVersion must be {SCHEMA_VERSION!r}")
        if self.document.get("tier") != _MOTION_CORE_TIER:
            raise TraceContractError(
                "tier must be 'motion_core'; this evaluator does not claim full-chain acceptance"
            )
        if self.contract is None:
            raise TraceContractError("mode must be one of " + ", ".join(MODES))
        case_id = self.document.get("caseId")
        if not isinstance(case_id, str) or not case_id.strip():
            raise TraceContractError("caseId must be a non-empty string")
        raw_samples = _sequence(self.document.get("samples"), "samples")
        if len(raw_samples) < 3:
            raise TraceContractError("samples must contain at least three observations")
        self.samples = [dict(_mapping(item, f"samples[{index}]")) for index, item in enumerate(raw_samples)]
        self.arm_evidence = dict(
            _mapping(self.document.get("armLimiterEvidence"), "armLimiterEvidence")
        )
        required_arm_fields = {
            "schemaVersion",
            "complete",
            "source",
            "interpolated",
            "repeatedTelemetry",
            "caseId",
            "policyActivationEpoch",
            "supportReleaseConfirmedEpoch",
            "plantBootId",
            "plantSessionId",
            "sonicDriverBootId",
            "sonicDriverSessionId",
            "sonicDriverSourceSha256",
            "taskStartControlTick",
            "taskEndControlTick",
            "records",
            "traceSha256",
        }
        if set(self.arm_evidence) != required_arm_fields:
            raise TraceContractError(
                "armLimiterEvidence v1 must contain exactly the authoritative success-trace fields"
            )
        if self.arm_evidence.get("schemaVersion") != ARM_LIMITER_SCHEMA_VERSION:
            raise TraceContractError(
                f"armLimiterEvidence.schemaVersion must be {ARM_LIMITER_SCHEMA_VERSION!r}"
            )
        if self.arm_evidence.get("complete") is not True:
            raise TraceContractError(
                "armLimiterEvidence.complete must prove a complete successful session"
            )
        driver_source_hash = self.arm_evidence.get("sonicDriverSourceSha256")
        if not isinstance(driver_source_hash, str) or not _SHA256.fullmatch(
            driver_source_hash
        ):
            raise TraceContractError(
                "armLimiterEvidence.sonicDriverSourceSha256 must be lowercase SHA-256"
            )
        claimed_trace_hash = self.arm_evidence.get("traceSha256")
        if not isinstance(claimed_trace_hash, str) or not _SHA256.fullmatch(
            claimed_trace_hash
        ):
            raise TraceContractError(
                "armLimiterEvidence.traceSha256 must be lowercase SHA-256"
            )
        trace_payload = {
            key: value
            for key, value in self.arm_evidence.items()
            if key != "traceSha256"
        }
        if claimed_trace_hash != canonical_sha256(trace_payload):
            self._fail(
                "ARM_TRACE_HASH_MISMATCH",
                "authoritative arm limiter trace hash does not match its complete payload",
            )
        raw_arm = _sequence(
            self.arm_evidence.get("records"), "armLimiterEvidence.records"
        )
        if len(raw_arm) < 2:
            raise TraceContractError(
                "armLimiterEvidence.records must contain complete 50 Hz evidence"
            )
        self.arm_trace = [
            dict(_mapping(item, f"armLimiterEvidence.records[{index}]"))
            for index, item in enumerate(raw_arm)
        ]
        required_record_fields = {
            "controlTick",
            "tS",
            "dtS",
            "constrainedJoints",
            "constrainedJointIndices",
            "continuousSaturationByJointS",
        }
        if any(set(item) != required_record_fields for item in self.arm_trace):
            raise TraceContractError(
                "armLimiterEvidence records v1 must contain exactly the six control-loop fields"
            )
        self.isolation_manifest = self._normalize_isolation_payload(
            self.document.get("isolationManifest"),
            "isolationManifest",
        )
        claimed_manifest_hash = self.document.get("isolationManifestSha256")
        if not isinstance(claimed_manifest_hash, str) or not _SHA256.fullmatch(
            claimed_manifest_hash
        ):
            raise TraceContractError(
                "isolationManifestSha256 must be lowercase SHA-256"
            )
        self.isolation_manifest_hash = canonical_sha256(self.isolation_manifest)
        if claimed_manifest_hash != self.isolation_manifest_hash:
            self._fail(
                "ISOLATION_MANIFEST_HASH_MISMATCH",
                "motion-core isolation manifest hash does not match",
            )

    def _normalize_isolation_payload(self, value: Any, label: str) -> dict[str, Any]:
        payload = _mapping(value, label)
        expected_fields = {
            "source",
            "actionInitiator",
            "cmdVelPublisher",
            *_ISOLATED_COMMAND_COMPONENTS,
        }
        if set(payload) != expected_fields:
            raise TraceContractError(f"{label} has unsupported/missing isolation fields")
        source = payload.get("source")
        if source != "strict_runner_preflight_and_continuous_observation":
            self._fail(
                "ISOLATION_SOURCE_INVALID",
                "isolation evidence must come from strict-runner preflight plus continuous observation",
            )
        initiator = payload.get("actionInitiator")
        if initiator != _STRICT_RUNNER_IDENTITY:
            self._fail(
                "ACTION_INITIATOR_INVALID",
                "motion-core actions must be initiated by strict_motion_runner",
            )
        publisher = _mapping(payload.get("cmdVelPublisher"), f"{label}.cmdVelPublisher")
        if set(publisher) != {"count", "nodeName", "publisherGid", "topic"}:
            raise TraceContractError(
                f"{label}.cmdVelPublisher must contain count/nodeName/publisherGid/topic"
            )
        publisher_count = _nonnegative_int(
            publisher.get("count"), f"{label}.cmdVelPublisher.count"
        )
        publisher_gid = publisher.get("publisherGid")
        publisher_node = publisher.get("nodeName")
        if not isinstance(publisher_gid, str) or not publisher_gid.strip():
            raise TraceContractError(
                f"{label}.cmdVelPublisher.publisherGid must be non-empty"
            )
        if not isinstance(publisher_node, str) or not publisher_node.strip():
            raise TraceContractError(
                f"{label}.cmdVelPublisher.nodeName must be non-empty"
            )
        if (
            publisher_count != 1
            or publisher.get("topic") != "/cmd_vel"
            or publisher_node not in {
                _STRICT_RUNNER_IDENTITY,
                f"/{_STRICT_RUNNER_IDENTITY}",
            }
        ):
            self._fail(
                "CMD_VEL_PUBLISHER_OWNERSHIP_INVALID",
                "strict_motion_runner must be the only observed /cmd_vel publisher",
            )
        normalized: dict[str, Any] = {
            "source": source,
            "actionInitiator": initiator,
            "cmdVelPublisher": {
                "count": publisher_count,
                "nodeName": publisher_node,
                "publisherGid": publisher_gid,
                "topic": publisher.get("topic"),
            },
        }
        for component_name in _ISOLATED_COMMAND_COMPONENTS:
            component = _mapping(payload.get(component_name), f"{label}.{component_name}")
            if set(component) != {"state", "leaseActive", "activeCommand"}:
                raise TraceContractError(
                    f"{label}.{component_name} must contain state/leaseActive/activeCommand"
                )
            state = component.get("state")
            lease_active = component.get("leaseActive")
            active_command = component.get("activeCommand")
            if not isinstance(lease_active, bool) or not isinstance(active_command, bool):
                raise TraceContractError(
                    f"{label}.{component_name} lease/command flags must be boolean"
                )
            if state not in {"stopped", "inactive"} or lease_active or active_command:
                self._fail(
                    "COMMAND_OWNER_NOT_ISOLATED",
                    f"{component_name} must be stopped/inactive with no lease or active command",
                )
            normalized[component_name] = {
                "state": state,
                "leaseActive": lease_active,
                "activeCommand": active_command,
            }
        return normalized

    def _sample_dynamics(
        self,
        sample: Mapping[str, Any],
        index: int,
        *,
        normalized_root: tuple[float, float, float],
        normalized_rpy: tuple[float, float, float],
        normalized_speeds: tuple[float, float],
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, ...],
        Mapping[str, Any],
        Mapping[str, int],
        int,
    ]:
        label = f"samples[{index}].plantDynamicsEvidence"
        plant_sequence = _nonnegative_int(
            sample.get("plantSequence"), f"samples[{index}].plantSequence"
        )
        if plant_sequence <= 0:
            raise TraceContractError(f"samples[{index}].plantSequence must be positive")
        dynamics = _mapping(sample.get("plantDynamicsEvidence"), label)
        if dynamics.get("schemaVersion") != PLANT_DYNAMICS_SCHEMA_VERSION:
            raise TraceContractError(f"{label}.schemaVersion is unsupported")
        dynamics_sequence = _nonnegative_int(
            dynamics.get("sampleSequence"), f"{label}.sampleSequence"
        )
        dynamics_time_s = _finite(
            dynamics.get("simulationTimeS"), f"{label}.simulationTimeS"
        )
        sample_time_s = _finite(sample.get("tS"), f"samples[{index}].tS")
        if abs(dynamics_time_s - sample_time_s) > _EPS:
            self._fail(
                "DYNAMICS_TIMESTAMP_MISMATCH",
                "normalized motion sample is not timestamped by its Plant physics snapshot",
                sample_time_s,
            )
        if dynamics.get("status") != "running" or dynamics.get("finite") is not True:
            self._fail("DYNAMICS_EVIDENCE_INVALID", "Plant dynamics evidence is not running and finite", _finite(sample.get("tS"), f"samples[{index}].tS"))
        root_state = _mapping(dynamics.get("rootState"), f"{label}.rootState")
        if root_state.get("body") != "pelvis":
            raise TraceContractError("Plant rootState.body must be pelvis")
        qpos_order = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        if root_state.get("qposOrder") != qpos_order:
            raise TraceContractError("Plant rootState.qposOrder is unsupported")
        qpos_raw = _sequence(root_state.get("qpos"), f"{label}.rootState.qpos")
        if len(qpos_raw) != 7:
            raise TraceContractError("Plant rootState.qpos must contain seven values")
        qpos = tuple(
            _finite(value, f"{label}.rootState.qpos[{position}]")
            for position, value in enumerate(qpos_raw)
        )
        qpos_rpy = _quaternion_to_rpy(qpos[3], qpos[4], qpos[5], qpos[6])
        if (
            math.dist(qpos[:3], normalized_root) > 1e-6
            or any(
                abs(_angle_delta(actual, normalized)) > 1e-6
                for actual, normalized in zip(qpos_rpy, normalized_rpy)
            )
        ):
            self._fail(
                "ROOT_STATE_MISMATCH",
                "normalized root pose does not match Plant dynamics rootState",
                _finite(sample.get("tS"), f"samples[{index}].tS"),
            )
        state_vectors: dict[str, tuple[float, ...]] = {}
        for field, prefix in (
            ("qvel", "qvel"),
            ("qacc", "qacc"),
            ("qaccWarmstart", "qacc"),
        ):
            if field != "qaccWarmstart" and root_state.get(f"{field}Order") != [
                f"{prefix}[{position}]" for position in range(6)
            ]:
                raise TraceContractError(f"Plant rootState.{field}Order is unsupported")
            raw_vector = _sequence(root_state.get(field), f"{label}.rootState.{field}")
            if len(raw_vector) != 6:
                raise TraceContractError(f"Plant rootState.{field} must contain six values")
            state_vectors[field] = tuple(
                _finite(value, f"{label}.rootState.{field}[{position}]")
                for position, value in enumerate(raw_vector)
            )
        qvel = state_vectors["qvel"]
        # Plant rootVelocity.linearSpeedMps is the documented horizontal
        # base_link magnitude (hypot(vx, vy)); vertical gait bob must not be
        # mistaken for an inconsistent normalized velocity.
        qvel_speeds = (
            math.hypot(qvel[0], qvel[1]),
            math.sqrt(sum(value * value for value in qvel[3:])),
        )
        if any(
            abs(actual - normalized) > 0.02 + _EPS
            for actual, normalized in zip(qvel_speeds, normalized_speeds)
        ):
            self._fail(
                "ROOT_VELOCITY_MISMATCH",
                "normalized root speed does not match Plant dynamics qvel",
                _finite(sample.get("tS"), f"samples[{index}].tS"),
            )
        com = _mapping(dynamics.get("robotCenterOfMass"), f"{label}.robotCenterOfMass")
        if com.get("frameId") != "world" or com.get("positionOrder") != ["x", "y", "z"]:
            raise TraceContractError("robotCenterOfMass must use world [x,y,z]")
        position = _sequence(com.get("positionM"), f"{label}.robotCenterOfMass.positionM")
        if len(position) != 3:
            raise TraceContractError("robotCenterOfMass.positionM must contain three values")
        com_xyz = tuple(_finite(position[i], f"{label}.robotCenterOfMass.positionM[{i}]") for i in range(3))
        contacts = _mapping(dynamics.get("contacts"), f"{label}.contacts")
        if contacts.get("schemaVersion") != CONTACT_SCHEMA_VERSION or contacts.get("valid") is not True:
            self._fail("CONTACT_EVIDENCE_INVALID", "Plant contact evidence is missing or invalid", _finite(sample.get("tS"), f"samples[{index}].tS"))
        contact_sequence = _nonnegative_int(
            contacts.get("sampleSequence"), f"{label}.contacts.sampleSequence"
        )
        if contact_sequence != dynamics_sequence:
            self._fail(
                "CONTACT_SEQUENCE_MISMATCH",
                "contact and dynamics evidence are not from the same physics snapshot",
                _finite(sample.get("tS"), f"samples[{index}].tS"),
            )
        foot_sides = _sequence(
            contacts.get("footGroundSides"), f"{label}.contacts.footGroundSides"
        )
        if any(side not in {"left", "right"} for side in foot_sides) or len(set(foot_sides)) != len(foot_sides):
            raise TraceContractError("footGroundSides must be a unique subset of left/right")
        foot_forces = _mapping(
            contacts.get("footGroundForceN"), f"{label}.contacts.footGroundForceN"
        )
        for side in ("left", "right"):
            if _finite(foot_forces.get(side), f"{label}.contacts.footGroundForceN.{side}") < 0:
                raise TraceContractError("foot ground forces cannot be negative")
        if _finite(
            contacts.get("footContactThresholdN"),
            f"{label}.contacts.footContactThresholdN",
        ) <= 0:
            raise TraceContractError("footContactThresholdN must be positive")
        if _nonnegative_int(
            contacts.get("footContactDebounceSamples"),
            f"{label}.contacts.footContactDebounceSamples",
        ) <= 0:
            raise TraceContractError("footContactDebounceSamples must be positive")
        warning_values = _mapping(dynamics.get("warningCounts"), f"{label}.warningCounts")
        warnings = {
            str(name): _nonnegative_int(value, f"{label}.warningCounts.{name}")
            for name, value in warning_values.items()
        }
        if not warnings:
            raise TraceContractError("Plant dynamics warningCounts cannot be empty")
        return com_xyz, qvel, contacts, warnings, dynamics_sequence

    def _evaluate_samples(self) -> None:
        assert self.contract is not None
        parsed: list[dict[str, Any]] = []
        previous_t: float | None = None
        previous_yaw: float | None = None
        previous_angular_speed: float | None = None
        unwrapped_yaw = 0.0
        previous_root: tuple[float, float, float] | None = None
        previous_com: tuple[float, float, float] | None = None
        previous_qvel: tuple[float, ...] | None = None
        max_raw_root_step_m = 0.0
        max_root_kinematic_residual_m = 0.0
        max_com_relative_step_m = 0.0
        baseline_counters: dict[str, int] | None = None
        previous_counters: dict[str, int] | None = None
        baseline_feet: dict[str, dict[str, int]] | None = None
        final_feet: dict[str, dict[str, int]] | None = None
        baseline_warnings: dict[str, int] | None = None
        previous_warnings: dict[str, int] | None = None
        final_warnings: dict[str, int] | None = None
        confirmed_release_epoch: int | None = None
        runtime_baseline: dict[str, dict[str, Any]] | None = None
        runtime_previous: dict[str, dict[str, Any]] | None = None
        baseline_dynamics_sequence: int | None = None
        previous_dynamics_sequence: int | None = None

        for index, sample in enumerate(self.samples):
            t_s = _finite(sample.get("tS"), f"samples[{index}].tS")
            if previous_t is not None and t_s <= previous_t:
                raise TraceContractError("sample times must be strictly increasing")
            root_obj = _mapping(sample.get("root"), f"samples[{index}].root")
            root = _xyz(root_obj, f"samples[{index}].root")
            roll = _finite(root_obj.get("roll"), f"samples[{index}].root.roll")
            pitch = _finite(root_obj.get("pitch"), f"samples[{index}].root.pitch")
            yaw = _finite(root_obj.get("yaw"), f"samples[{index}].root.yaw")
            linear_speed = _finite(root_obj.get("linearSpeedMps"), f"samples[{index}].root.linearSpeedMps")
            angular_speed = _finite(root_obj.get("angularSpeedRps"), f"samples[{index}].root.angularSpeedRps")
            if linear_speed < 0 or angular_speed < 0:
                raise TraceContractError("root speed magnitudes cannot be negative")
            if not self.contract.min_root_z_m <= root[2] <= self.contract.max_root_z_m:
                self._fail("ROOT_HEIGHT_OUT_OF_RANGE", f"root z is {root[2]:.4f}m", t_s)
            if max(abs(roll), abs(pitch)) > self.contract.max_roll_pitch_rad + _EPS:
                self._fail("ROOT_TILT_EXCEEDED", f"roll/pitch reached {max(abs(roll), abs(pitch)):.4f}rad", t_s)
            if previous_yaw is not None:
                yaw_step = _angle_delta(yaw, previous_yaw)
                unwrapped_yaw += yaw_step
                dt_s = t_s - previous_t if previous_t is not None else 0.0
                integrated_envelope = (
                    0.5 * (previous_angular_speed + angular_speed) * dt_s
                    + 0.05
                    if previous_angular_speed is not None
                    else 0.0
                )
                allowed_yaw_step = max(
                    self.contract.max_yaw_jump_rad,
                    integrated_envelope,
                )
                if abs(yaw_step) > allowed_yaw_step + _EPS:
                    self._fail("ROOT_YAW_JUMP", f"yaw jumped {abs(yaw_step):.4f}rad", t_s)
            com, qvel, contacts, warnings, dynamics_sequence = self._sample_dynamics(
                sample,
                index,
                normalized_root=root,
                normalized_rpy=(roll, pitch, yaw),
                normalized_speeds=(linear_speed, angular_speed),
            )
            if (
                previous_t is not None
                and previous_root is not None
                and previous_com is not None
                and previous_qvel is not None
            ):
                dt_s = t_s - previous_t
                root_step = tuple(
                    root[axis] - previous_root[axis] for axis in range(3)
                )
                expected_root_step = tuple(
                    0.5 * (previous_qvel[axis] + qvel[axis]) * dt_s
                    for axis in range(3)
                )
                root_kinematic_residual = math.dist(
                    root_step,
                    expected_root_step,
                )
                raw_root_step = math.sqrt(sum(value * value for value in root_step))
                max_raw_root_step_m = max(max_raw_root_step_m, raw_root_step)
                max_root_kinematic_residual_m = max(
                    max_root_kinematic_residual_m,
                    root_kinematic_residual,
                )
                if (
                    root_kinematic_residual
                    > self.contract.max_position_jump_m + _EPS
                ):
                    self._fail(
                        "ROOT_POSITION_JUMP",
                        "root motion disagrees with MuJoCo qvel integration by "
                        f"{root_kinematic_residual:.4f}m",
                        t_s,
                    )

                com_step = tuple(
                    com[axis] - previous_com[axis] for axis in range(3)
                )
                com_relative_step = math.dist(com_step, root_step)
                max_com_relative_step_m = max(
                    max_com_relative_step_m,
                    com_relative_step,
                )
                if com_relative_step > self.contract.max_com_jump_m + _EPS:
                    self._fail(
                        "COM_POSITION_JUMP",
                        "COM motion disagrees with the pelvis displacement by "
                        f"{com_relative_step:.4f}m",
                        t_s,
                    )
            if baseline_dynamics_sequence is None:
                baseline_dynamics_sequence = dynamics_sequence
            if (
                previous_dynamics_sequence is not None
                and dynamics_sequence <= previous_dynamics_sequence
            ):
                self._fail(
                    "DYNAMICS_SEQUENCE_NOT_ADVANCING",
                    "Plant physics evidence sequence did not advance",
                    t_s,
                )
            dynamics = _mapping(
                sample.get("plantDynamicsEvidence"),
                f"samples[{index}].plantDynamicsEvidence",
            )
            support = _mapping(sample.get("plantSupport"), f"samples[{index}].plantSupport")
            release_epoch = _nonnegative_int(
                support.get("fallArrestReleaseConfirmedEpoch"),
                f"samples[{index}].plantSupport.fallArrestReleaseConfirmedEpoch",
            )
            if release_epoch <= 0:
                self._fail(
                    "SUPPORT_RELEASE_NOT_CONFIRMED",
                    "fall-arrest release requires a positive confirmed activation epoch",
                    t_s,
                )
            if confirmed_release_epoch is None:
                confirmed_release_epoch = release_epoch
            elif release_epoch != confirmed_release_epoch:
                self._fail(
                    "SUPPORT_RELEASE_EPOCH_CHANGED",
                    "fall-arrest release epoch changed within the capability trace",
                    t_s,
                )
            if (
                dynamics.get("supportBandEnabled") is not False
                or support.get("fallArrestEnabled") is not False
            ):
                self._fail(
                    "SUPPORT_BAND_STILL_ENABLED",
                    "formal locomotion evidence requires confirmed support-off physics",
                    t_s,
                )
            rejected_epoch = support.get("fallArrestReleaseRejectedEpoch")
            if rejected_epoch is not None:
                self._fail(
                    "SUPPORT_RELEASE_REJECTED",
                    f"fall-arrest release rejection is present: {rejected_epoch!r}",
                    t_s,
                )
            counter_names = (
                "obstacleContactSampleCount",
                "obstacleContactEventCount",
                "nonFootGroundContactSampleCount",
                "nonFootGroundContactEventCount",
            )
            counters = {
                name: _nonnegative_int(contacts.get(name), f"samples[{index}].contacts.{name}")
                for name in counter_names
            }
            if contacts.get("obstaclePairs") != []:
                self._fail("ACTIVE_OBSTACLE_CONTACT", "active robot/static obstacle contact was observed", t_s)
            if contacts.get("nonFootGroundPairs") != []:
                self._fail("ACTIVE_NONFOOT_GROUND_CONTACT", "active non-foot/ground contact was observed", t_s)
            feet = {
                "touchdown": {
                    side: _nonnegative_int(
                        _mapping(contacts.get("footTouchdownCount"), "footTouchdownCount").get(side),
                        f"footTouchdownCount.{side}",
                    )
                    for side in ("left", "right")
                },
                "liftoff": {
                    side: _nonnegative_int(
                        _mapping(contacts.get("footLiftoffCount"), "footLiftoffCount").get(side),
                        f"footLiftoffCount.{side}",
                    )
                    for side in ("left", "right")
                },
            }
            if final_feet is not None and any(
                feet[event][side] < final_feet[event][side]
                for event in ("touchdown", "liftoff")
                for side in ("left", "right")
            ):
                self._fail(
                    "FOOT_COUNTER_ROLLBACK",
                    "Plant foot touchdown/liftoff counters rolled back",
                    t_s,
                )
            if baseline_counters is None:
                baseline_counters = counters
                baseline_feet = feet
                baseline_warnings = warnings
            if previous_counters is not None and any(counters[name] < previous_counters[name] for name in counter_names):
                self._fail("CONTACT_COUNTER_ROLLBACK", "Plant contact counter rolled back during the session", t_s)
            if previous_warnings is not None:
                if set(warnings) != set(previous_warnings):
                    self._fail("WARNING_COUNTER_SCHEMA_CHANGED", "MuJoCo warning counter keys changed", t_s)
                elif any(warnings[name] < previous_warnings[name] for name in warnings):
                    self._fail("WARNING_COUNTER_ROLLBACK", "MuJoCo warning counter rolled back", t_s)

            runtime_raw = _mapping(
                sample.get("runtimeHealth"), f"samples[{index}].runtimeHealth"
            )
            if set(runtime_raw) != set(_REQUIRED_RUNTIME_COMPONENTS):
                raise TraceContractError(
                    "motion_core runtimeHealth must contain exactly Plant/SONIC/Nav2/ROS bridge and both viewers"
                )
            runtime: dict[str, dict[str, Any]] = {}
            for component_name in _REQUIRED_RUNTIME_COMPONENTS:
                component = _mapping(
                    runtime_raw.get(component_name),
                    f"samples[{index}].runtimeHealth.{component_name}",
                )
                base_component_fields = {
                    "healthy",
                    "bootId",
                    "sequence",
                    "lossCount",
                    "restartCount",
                }
                if component_name in {"plant", "sonic"}:
                    expected_component_fields = {
                        *base_component_fields,
                        "sessionId",
                    }
                elif component_name == "firstPersonViewer":
                    expected_component_fields = {
                        *base_component_fields,
                        "schemaVersion",
                        "pid",
                        "status",
                        "mapped",
                        "plantBootId",
                        "plantSessionId",
                        "plantSequence",
                        "simulationTimeS",
                        "captureAgeMs",
                        "lastSuccessAtUnixNs",
                        "failureCount",
                        "lastError",
                    }
                else:
                    expected_component_fields = base_component_fields
                if set(component) != expected_component_fields:
                    raise TraceContractError(
                        f"runtimeHealth.{component_name} has unsupported/missing fields"
                    )
                boot_id = component.get("bootId")
                if not isinstance(boot_id, str) or not boot_id.strip():
                    raise TraceContractError(
                        f"runtimeHealth.{component_name}.bootId must be non-empty"
                    )
                session_id: str | None = None
                if component_name in {"plant", "sonic"}:
                    raw_session_id = component.get("sessionId")
                    if (
                        not isinstance(raw_session_id, str)
                        or not raw_session_id.strip()
                    ):
                        raise TraceContractError(
                            f"runtimeHealth.{component_name}.sessionId must be non-empty"
                        )
                    session_id = raw_session_id
                sequence_value = _nonnegative_int(
                    component.get("sequence"),
                    f"runtimeHealth.{component_name}.sequence",
                )
                loss_count = _nonnegative_int(
                    component.get("lossCount"),
                    f"runtimeHealth.{component_name}.lossCount",
                )
                restart_count = _nonnegative_int(
                    component.get("restartCount"),
                    f"runtimeHealth.{component_name}.restartCount",
                )
                if component.get("healthy") is not True:
                    self._fail(
                        "RUNTIME_COMPONENT_UNHEALTHY",
                        f"{component_name} was not continuously healthy",
                        t_s,
                    )
                if loss_count != 0 or restart_count != 0:
                    self._fail(
                        "RUNTIME_LOSS_OR_RESTART",
                        f"{component_name} loss/restart counters are "
                        f"{loss_count}/{restart_count}",
                        t_s,
                    )
                normalized_component: dict[str, Any] = {
                    "bootId": boot_id,
                    "sessionId": session_id,
                    "sequence": sequence_value,
                    "lossCount": loss_count,
                    "restartCount": restart_count,
                }
                if component_name == "firstPersonViewer":
                    viewer_pid = _nonnegative_int(
                        component.get("pid"),
                        "runtimeHealth.firstPersonViewer.pid",
                    )
                    viewer_plant_sequence = _nonnegative_int(
                        component.get("plantSequence"),
                        "runtimeHealth.firstPersonViewer.plantSequence",
                    )
                    viewer_simulation_time = _finite(
                        component.get("simulationTimeS"),
                        "runtimeHealth.firstPersonViewer.simulationTimeS",
                    )
                    viewer_capture_age = _finite(
                        component.get("captureAgeMs"),
                        "runtimeHealth.firstPersonViewer.captureAgeMs",
                    )
                    viewer_success_ns = _nonnegative_int(
                        component.get("lastSuccessAtUnixNs"),
                        "runtimeHealth.firstPersonViewer.lastSuccessAtUnixNs",
                    )
                    viewer_failure_count = _nonnegative_int(
                        component.get("failureCount"),
                        "runtimeHealth.firstPersonViewer.failureCount",
                    )
                    if (
                        component.get("schemaVersion")
                        != _FIRST_PERSON_VIEWER_HEALTH_SCHEMA_VERSION
                        or viewer_pid <= 0
                        or component.get("status") != "running"
                        or component.get("mapped") is not True
                        or viewer_plant_sequence <= 0
                        or viewer_simulation_time < 0.0
                        or viewer_success_ns <= 0
                        or component.get("lastError") is not None
                    ):
                        self._fail(
                            "FIRST_PERSON_VIEWER_STATUS_INVALID",
                            "first-person evidence must be a mapped/running authoritative success status",
                            t_s,
                        )
                    if (
                        viewer_capture_age < 0.0
                        or viewer_capture_age
                        > self.contract.max_first_person_capture_age_ms + _EPS
                    ):
                        self._fail(
                            "FIRST_PERSON_VIEWER_CAPTURE_STALE",
                            f"first-person Plant capture age is {viewer_capture_age:.1f}ms; <=2000ms required",
                            t_s,
                        )
                    normalized_component.update(
                        pid=viewer_pid,
                        plantBootId=component.get("plantBootId"),
                        plantSessionId=component.get("plantSessionId"),
                        plantSequence=viewer_plant_sequence,
                        simulationTimeS=viewer_simulation_time,
                        captureAgeMs=viewer_capture_age,
                        lastSuccessAtUnixNs=viewer_success_ns,
                        failureCount=viewer_failure_count,
                    )
                runtime[component_name] = normalized_component

            viewer = runtime["firstPersonViewer"]
            sample_plant_sequence = _nonnegative_int(
                sample.get("plantSequence"), f"samples[{index}].plantSequence"
            )
            if (
                viewer.get("plantBootId") != runtime["plant"]["bootId"]
                or viewer.get("plantSessionId") != runtime["plant"]["sessionId"]
                or int(viewer["plantSequence"]) > sample_plant_sequence
            ):
                self._fail(
                    "FIRST_PERSON_PLANT_BINDING_MISMATCH",
                    "first-person frame is not bound to the accepted Plant boot/session or precedes the trace snapshot",
                    t_s,
                )
            if runtime_baseline is None:
                runtime_baseline = runtime
            if runtime_previous is not None:
                for component_name in _REQUIRED_RUNTIME_COMPONENTS:
                    if (
                        runtime[component_name]["bootId"]
                        != runtime_previous[component_name]["bootId"]
                    ):
                        self._fail(
                            "RUNTIME_BOOT_ID_CHANGED",
                            f"{component_name} restarted during the trace",
                            t_s,
                        )
                    if (
                        runtime[component_name]["sessionId"]
                        != runtime_previous[component_name]["sessionId"]
                    ):
                        self._fail(
                            "RUNTIME_SESSION_ID_CHANGED",
                            f"{component_name} session changed during the trace",
                            t_s,
                        )
                    if (
                        runtime[component_name]["sequence"]
                        <= runtime_previous[component_name]["sequence"]
                    ):
                        self._fail(
                            "RUNTIME_COMPONENT_STALE",
                            f"{component_name} sequence did not advance",
                            t_s,
                        )
                previous_viewer = runtime_previous["firstPersonViewer"]
                if (
                    viewer["pid"] != previous_viewer["pid"]
                    or viewer["plantSequence"] <= previous_viewer["plantSequence"]
                    or viewer["simulationTimeS"]
                    <= previous_viewer["simulationTimeS"]
                    or viewer["lastSuccessAtUnixNs"]
                    <= previous_viewer["lastSuccessAtUnixNs"]
                ):
                    self._fail(
                        "FIRST_PERSON_VIEWER_NOT_ADVANCING",
                        "first-person PID must stay stable while its Plant frame/time/success sequence advances",
                        t_s,
                    )
                if viewer["failureCount"] != previous_viewer["failureCount"]:
                    self._fail(
                        "FIRST_PERSON_VIEWER_FAILURE_RECORDED",
                        "first-person authoritative failure counter changed during the trace",
                        t_s,
                    )

            isolation_raw = dict(
                _mapping(
                    sample.get("isolationProof"),
                    f"samples[{index}].isolationProof",
                )
            )
            claimed_sample_manifest_hash = isolation_raw.pop(
                "manifestSha256", None
            )
            if (
                not isinstance(claimed_sample_manifest_hash, str)
                or not _SHA256.fullmatch(claimed_sample_manifest_hash)
            ):
                raise TraceContractError(
                    f"samples[{index}].isolationProof.manifestSha256 must be lowercase SHA-256"
                )
            isolation = self._normalize_isolation_payload(
                isolation_raw,
                f"samples[{index}].isolationProof",
            )
            if (
                claimed_sample_manifest_hash != self.isolation_manifest_hash
                or isolation != self.isolation_manifest
            ):
                self._fail(
                    "ISOLATION_PROOF_CHANGED",
                    "continuous command-owner/publisher isolation differs from the preflight manifest",
                    t_s,
                )

            command = _mapping(sample.get("command"), f"samples[{index}].command")
            vx, vy, wz = (
                _finite(command.get(name), f"samples[{index}].command.{name}")
                for name in ("vx", "vy", "wz")
            )
            policy_idle = _mapping(
                sample.get("sonicPolicyIdle"),
                f"samples[{index}].sonicPolicyIdle",
            )
            expected_policy_idle_fields = {
                "active",
                "mode",
                "referenceAuthorityState",
                "actuatorAuthority",
                "staticHoldActive",
                "plantSupportMode",
                "plantSupportStopPending",
                "plantSupportSafetyLatched",
                "navigationTaskActive",
                "commandZero",
            }
            if set(policy_idle) != expected_policy_idle_fields:
                raise TraceContractError(
                    f"samples[{index}].sonicPolicyIdle must contain exactly "
                    "the dynamic balance authority fields"
                )
            for field in (
                "active",
                "actuatorAuthority",
                "staticHoldActive",
                "plantSupportStopPending",
                "plantSupportSafetyLatched",
                "navigationTaskActive",
                "commandZero",
            ):
                if not isinstance(policy_idle.get(field), bool):
                    raise TraceContractError(
                        f"samples[{index}].sonicPolicyIdle.{field} must be boolean"
                    )
            balanced_policy_idle = bool(
                policy_idle.get("active") is True
                and policy_idle.get("mode") == "IDLE"
                and policy_idle.get("referenceAuthorityState") == "GRANTED"
                and policy_idle.get("actuatorAuthority") is True
                and policy_idle.get("staticHoldActive") is False
                and policy_idle.get("plantSupportMode") == "idle"
                and policy_idle.get("plantSupportStopPending") is False
                and policy_idle.get("plantSupportSafetyLatched") is False
                and policy_idle.get("navigationTaskActive") is False
                and policy_idle.get("commandZero") is True
            )
            if bool(policy_idle["active"]) != balanced_policy_idle:
                self._fail(
                    "BALANCED_POLICY_IDLE_AUTHORITY_INVALID",
                    "SONIC balanced IDLE claim lacks executable policy/support authority",
                    t_s,
                )
            if not isinstance(sample.get("motionIntentActive"), bool):
                raise TraceContractError(f"samples[{index}].motionIntentActive must be boolean")
            intent = bool(sample["motionIntentActive"])
            actions = _mapping(sample.get("nav2Actions"), f"samples[{index}].nav2Actions")
            action_values = {}
            for key in (
                "driveOnHeadingActive",
                "navigateToPoseActive",
                "followPathActive",
                "spinActive",
            ):
                if not isinstance(actions.get(key), bool):
                    raise TraceContractError(f"samples[{index}].nav2Actions.{key} must be boolean")
                action_values[key] = bool(actions[key])
            sonic_control_tick: int | None = None
            detour_segment_index: int | None = None
            if self.mode == "detour":
                sonic_control_tick = _nonnegative_int(
                    sample.get("sonicControlTick"),
                    f"samples[{index}].sonicControlTick",
                )
                raw_segment_index = sample.get("detourSegmentIndex")
                if raw_segment_index is not None:
                    detour_segment_index = _nonnegative_int(
                        raw_segment_index,
                        f"samples[{index}].detourSegmentIndex",
                    )
            parsed.append(
                {
                    "t": t_s,
                    "root": root,
                    "com": com,
                    "roll": roll,
                    "pitch": pitch,
                    "yaw": yaw,
                    "unwrappedYaw": unwrapped_yaw,
                    "linearSpeed": linear_speed,
                    "angularSpeed": angular_speed,
                    "vx": vx,
                    "vy": vy,
                    "wz": wz,
                    "intent": intent,
                    "balancedPolicyIdle": balanced_policy_idle,
                    "footGroundSides": tuple(
                        str(side) for side in contacts["footGroundSides"]
                    ),
                    "actions": action_values,
                    "sonicControlTick": sonic_control_tick,
                    "detourSegmentIndex": detour_segment_index,
                }
            )
            previous_t, previous_yaw = t_s, yaw
            previous_angular_speed = angular_speed
            previous_root, previous_com, previous_qvel = root, com, qvel
            previous_counters, final_feet = counters, feet
            previous_warnings, final_warnings = warnings, warnings
            runtime_previous = runtime
            previous_dynamics_sequence = dynamics_sequence

        assert baseline_counters is not None and previous_counters is not None
        assert baseline_feet is not None and final_feet is not None
        assert baseline_warnings is not None and final_warnings is not None
        assert runtime_baseline is not None and runtime_previous is not None
        self.metrics["kinematicContinuity"] = {
            "maxRawRootStepM": max_raw_root_step_m,
            "maxRootQvelIntegrationResidualM": max_root_kinematic_residual_m,
            "maxComPelvisRelativeStepM": max_com_relative_step_m,
            "residualLimitM": self.contract.max_position_jump_m,
        }
        assert baseline_dynamics_sequence is not None
        assert previous_dynamics_sequence is not None
        collision_deltas = {
            name: previous_counters[name] - baseline_counters[name]
            for name in baseline_counters
        }
        self.metrics["contactCounterDeltas"] = collision_deltas
        if any(collision_deltas.values()):
            self._fail("CONTACT_COUNTER_INCREMENTED", f"contact counter deltas must all be zero: {collision_deltas}", parsed[-1]["t"])
        warning_deltas = {
            name: final_warnings[name] - baseline_warnings[name]
            for name in baseline_warnings
            if name in final_warnings
        }
        self.metrics["warningCounterDeltas"] = warning_deltas
        if any(value > 0 for value in warning_deltas.values()):
            self._fail("MUJOCO_WARNING_INCREMENTED", f"MuJoCo warning counters increased: {warning_deltas}", parsed[-1]["t"])
        foot_deltas = {
            event: {
                side: final_feet[event][side] - baseline_feet[event][side]
                for side in ("left", "right")
            }
            for event in ("touchdown", "liftoff")
        }
        self.metrics["footEventDeltas"] = foot_deltas
        completed_foot_cycles = sum(
            min(foot_deltas["touchdown"][side], foot_deltas["liftoff"][side])
            for side in ("left", "right")
        )
        self.metrics["footCycleEvidence"] = {
            "derivation": "authoritative-counter-delta",
            "orderingProven": False,
            "requiredForMode": self.mode not in {"turn_left", "turn_right"},
            "totalCompletedFootCyclesLowerBound": completed_foot_cycles,
        }
        self.metrics["supportReleaseConfirmedEpoch"] = confirmed_release_epoch
        self.metrics["plantDynamicsSequenceDelta"] = (
            previous_dynamics_sequence - baseline_dynamics_sequence
        )
        runtime_sequence_deltas = {
            name: runtime_previous[name]["sequence"]
            - runtime_baseline[name]["sequence"]
            for name in _REQUIRED_RUNTIME_COMPONENTS
        }
        self.metrics["runtimeHealth"] = {
            "bootIds": {
                name: runtime_baseline[name]["bootId"]
                for name in _REQUIRED_RUNTIME_COMPONENTS
            },
            "plantSessionId": runtime_baseline["plant"]["sessionId"],
            "sonicSessionId": runtime_baseline["sonic"]["sessionId"],
            "sequenceDeltas": runtime_sequence_deltas,
            "lossCount": 0,
            "restartCount": 0,
            "firstPersonViewer": {
                "schemaVersion": _FIRST_PERSON_VIEWER_HEALTH_SCHEMA_VERSION,
                "pid": runtime_baseline["firstPersonViewer"]["pid"],
                "plantBootId": runtime_baseline["firstPersonViewer"][
                    "plantBootId"
                ],
                "plantSessionId": runtime_baseline["firstPersonViewer"][
                    "plantSessionId"
                ],
                "plantSequenceDelta": runtime_previous["firstPersonViewer"][
                    "plantSequence"
                ]
                - runtime_baseline["firstPersonViewer"]["plantSequence"],
                "failureCountDelta": runtime_previous["firstPersonViewer"][
                    "failureCount"
                ]
                - runtime_baseline["firstPersonViewer"]["failureCount"],
                "maxAllowedCaptureAgeMs": self.contract.max_first_person_capture_age_ms,
            },
        }
        self.metrics["acceptanceTier"] = _MOTION_CORE_TIER
        self.metrics["isolationProof"] = {
            "manifestSha256": self.isolation_manifest_hash,
            "actionInitiator": _STRICT_RUNNER_IDENTITY,
            "cmdVelPublisher": self.isolation_manifest.get("cmdVelPublisher"),
            "isolatedComponents": list(_ISOLATED_COMMAND_COMPONENTS),
            "fullChainClaimed": False,
        }
        if any(value <= 0 for value in runtime_sequence_deltas.values()):
            self._fail(
                "RUNTIME_COMPONENT_STALE",
                f"all runtime/viewer sequences must advance: {runtime_sequence_deltas}",
                parsed[-1]["t"],
            )
        if self.mode in {"turn_left", "turn_right"}:
            active_spin_samples = [sample for sample in parsed if sample["intent"]]
            support_counts = [
                len(sample["footGroundSides"]) for sample in active_spin_samples
            ]
            observed_support_sides = sorted(
                {
                    side
                    for sample in active_spin_samples
                    for side in sample["footGroundSides"]
                }
            )
            minimum_support_count = min(support_counts, default=0)
            self.metrics["spinFootSupportEvidence"] = {
                "derivation": "authoritative-contact-snapshot",
                "activeSampleCount": len(active_spin_samples),
                "minimumSupportedFeet": minimum_support_count,
                "supportSidesObserved": observed_support_sides,
            }
            if minimum_support_count < 1 or observed_support_sides != ["left", "right"]:
                self._fail(
                    "TURN_FOOT_SUPPORT_LOST",
                    "Spin requires continuous support and observed contact from both feet",
                    parsed[-1]["t"],
                )
        elif any(
            foot_deltas[event][side] < 1
            for event in foot_deltas
            for side in ("left", "right")
        ):
            self._fail(
                "INCOMPLETE_FOOT_CYCLES",
                "both feet require touchdown and liftoff deltas >=1: "
                f"{foot_deltas}",
                parsed[-1]["t"],
            )

        self._evaluate_motion(parsed)
        self._parsed_samples = parsed

    def _evaluate_motion(self, samples: Sequence[Mapping[str, Any]]) -> None:
        assert self.contract is not None
        first, final = samples[0], samples[-1]
        yaw0 = float(first["yaw"])
        cosine, sine = math.cos(yaw0), math.sin(yaw0)

        def body_delta(point: Sequence[float], origin: Sequence[float]) -> tuple[float, float]:
            dx, dy = float(point[0]) - float(origin[0]), float(point[1]) - float(origin[1])
            return dx * cosine + dy * sine, -dx * sine + dy * cosine

        root_values = [body_delta(sample["root"], first["root"]) for sample in samples]
        com_values = [body_delta(sample["com"], first["com"]) for sample in samples]
        heading_values = [float(sample["unwrappedYaw"]) for sample in samples]
        self.metrics.update(
            rootStartM=list(first["root"]),
            rootFinalM=list(final["root"]),
            comStartM=list(first["com"]),
            comFinalM=list(final["com"]),
            rootTraceSha256=canonical_sha256([list(sample["root"]) for sample in samples]),
            comTraceSha256=canonical_sha256([list(sample["com"]) for sample in samples]),
            finalLongitudinalM=root_values[-1][0],
            finalLateralM=root_values[-1][1],
            finalComLongitudinalM=com_values[-1][0],
            finalComLateralM=com_values[-1][1],
            finalYawDeltaRad=heading_values[-1],
        )
        root_com_disagreements = [
            math.dist(root_value, com_value)
            for root_value, com_value in zip(root_values, com_values)
        ]
        root_com_disagreement = root_com_disagreements[-1]
        max_root_com_disagreement = max(root_com_disagreements)
        self.metrics["finalRootComDeltaDisagreementM"] = root_com_disagreement
        self.metrics["maxRootComDeltaDisagreementM"] = max_root_com_disagreement
        if max_root_com_disagreement > self.contract.max_root_com_delta_disagreement_m + _EPS:
            self._fail("ROOT_COM_DISAGREEMENT", f"root/COM displacement disagreement reached {max_root_com_disagreement:.4f}m", float(final["t"]))

        active = [sample for sample in samples if sample["intent"]]
        if not active:
            self._fail("NO_MOTION_INTENT", "no active motion-intent sample was recorded")
            return
        first_intent = float(active[0]["t"])
        required_action = {
            "forward_probe": "driveOnHeadingActive",
            "forward": "driveOnHeadingActive",
            "forward_3m": "driveOnHeadingActive",
            "reverse": "driveOnHeadingActive",
            "turn_left": "spinActive",
            "turn_right": "spinActive",
        }.get(self.mode)
        if required_action is not None:
            for sample in active:
                active_actions = [
                    name for name, enabled in sample["actions"].items() if enabled
                ]
                if active_actions != [required_action]:
                    self._fail(
                        "NAV2_ACTION_EVIDENCE_MISMATCH",
                        f"{self.mode} requires only {required_action} while intent is active",
                        float(sample["t"]),
                    )
        progress_time: float | None = None
        if self.contract.expected_longitudinal_m is not None:
            expected = self.contract.expected_longitudinal_m
            direction = 1.0 if expected > 0 else -1.0
            directional = [direction * value[0] for value in root_values]
            cross = [abs(value[1]) for value in root_values]
            heading = [abs(value) for value in heading_values]
            for sample, value in zip(samples, directional):
                if value >= self.contract.min_linear_progress_m:
                    progress_time = float(sample["t"])
                    break
            if min(directional) < -self.contract.max_opposite_motion_m - _EPS:
                self._fail("OPPOSITE_MOTION_EXCEEDED", f"opposite travel reached {-min(directional):.4f}m")
            if max(directional) > abs(expected) + self.contract.max_linear_overshoot_m + _EPS:
                self._fail("LINEAR_OVERSHOOT", f"directional travel reached {max(directional):.4f}m")
            if max(cross) > self.contract.max_cross_track_m + _EPS:
                self._fail("CROSS_TRACK_EXCEEDED", f"cross-track travel reached {max(cross):.4f}m")
            if max(heading) > self.contract.max_heading_error_rad + _EPS:
                self._fail("HEADING_ERROR_EXCEEDED", f"heading error reached {max(heading):.4f}rad")
            if abs(root_values[-1][0] - expected) > self.contract.final_position_tolerance_m + _EPS:
                self._fail("FINAL_POSITION_MISSED", f"final longitudinal travel is {root_values[-1][0]:.4f}m; expected {expected:.4f}m")
            if abs(com_values[-1][0] - expected) > self.contract.final_position_tolerance_m + self.contract.max_root_com_delta_disagreement_m + _EPS:
                self._fail("FINAL_COM_POSITION_MISSED", f"final COM longitudinal travel is {com_values[-1][0]:.4f}m")
            signed_commands = [direction * float(sample["vx"]) for sample in active]
            if min(signed_commands) < -self.contract.command_zero_epsilon - _EPS or max(signed_commands) <= self.contract.command_zero_epsilon:
                self._fail("LINEAR_COMMAND_DIRECTION_MISMATCH", "DriveOnHeading command direction contradicts the formal goal")
            for sample in active:
                if (
                    abs(float(sample["vy"])) > self.contract.command_zero_epsilon + _EPS
                    or abs(float(sample["wz"])) > self.contract.command_zero_epsilon + _EPS
                    or abs(float(sample["vx"]))
                    > self.contract.drive_on_heading_max_vx_mps
                    + self.contract.command_serialization_tolerance
                ):
                    self._fail(
                        "DRIVE_ON_HEADING_COMMAND_ENVELOPE",
                        "DriveOnHeading exceeds the NVIDIA direction speed or mixes axes",
                        float(sample["t"]),
                    )
            if max(signed_commands) < self.contract.drive_on_heading_max_vx_mps - self.contract.steady_target_tolerance - _EPS:
                self._fail(
                    "DRIVE_ON_HEADING_STEADY_TARGET_MISSING",
                    "no signed NVIDIA DriveOnHeading steady command was observed",
                )
        elif self.contract.expected_lateral_m is not None:
            expected = self.contract.expected_lateral_m
            direction = 1.0 if expected > 0 else -1.0
            directional = [direction * value[1] for value in root_values]
            cross = [abs(value[0]) for value in root_values]
            heading = [abs(value) for value in heading_values]
            active_indices = [
                index for index, sample in enumerate(samples) if sample["intent"]
            ]
            active_cross = [cross[index] for index in active_indices]
            active_heading = [heading[index] for index in active_indices]
            self.metrics["lateralTrackingEnvelope"] = {
                "activeMaxCrossTrackM": max(active_cross, default=0.0),
                "brakingMaxCrossTrackM": max(cross),
                "finalCrossTrackM": cross[-1],
                "activeMaxHeadingErrorRad": max(active_heading, default=0.0),
                "brakingMaxHeadingErrorRad": max(heading),
                "finalHeadingErrorRad": heading[-1],
            }
            for sample, value in zip(samples, directional):
                if value >= self.contract.min_linear_progress_m:
                    progress_time = float(sample["t"])
                    break
            if min(directional) < -self.contract.max_opposite_motion_m - _EPS:
                self._fail("OPPOSITE_MOTION_EXCEEDED", f"opposite lateral travel reached {-min(directional):.4f}m")
            if max(directional) > abs(expected) + self.contract.final_position_tolerance_m + _EPS:
                self._fail("LINEAR_OVERSHOOT", f"directional lateral travel reached {max(directional):.4f}m")
            if max(active_cross, default=0.0) > self.contract.max_cross_track_m + _EPS:
                self._fail("CROSS_TRACK_EXCEEDED", f"active lateral longitudinal drift reached {max(active_cross):.4f}m")
            if max(cross) > self.contract.max_lateral_braking_cross_track_m + _EPS:
                self._fail("BRAKING_CROSS_TRACK_EXCEEDED", f"lateral braking longitudinal drift reached {max(cross):.4f}m")
            if cross[-1] > self.contract.max_cross_track_m + _EPS:
                self._fail("FINAL_CROSS_TRACK_EXCEEDED", f"settled lateral longitudinal drift is {cross[-1]:.4f}m")
            if max(active_heading, default=0.0) > self.contract.max_heading_error_rad + _EPS:
                self._fail("HEADING_ERROR_EXCEEDED", f"active lateral yaw drift exceeded {self.contract.max_heading_error_rad:.4f}rad")
            if max(heading) > self.contract.max_lateral_braking_heading_error_rad + _EPS:
                self._fail("BRAKING_HEADING_ERROR_EXCEEDED", f"lateral braking yaw drift exceeded {self.contract.max_lateral_braking_heading_error_rad:.4f}rad")
            # Learned-IDLE performs the physical braking handoff after the
            # direct lateral command has ended.  Keep the active-motion bound
            # at 10 degrees, but apply the already-reviewed 12-degree braking
            # envelope to its settled residual as well.  Requiring the stop
            # policy to return from 12 to 10 degrees without a corrective Spin
            # rejected the otherwise clean live r137 right strafe.
            if (
                heading[-1]
                > self.contract.max_lateral_braking_heading_error_rad + _EPS
            ):
                self._fail(
                    "FINAL_HEADING_ERROR_EXCEEDED",
                    "settled lateral yaw drift exceeded the bounded learned-IDLE "
                    f"envelope: {heading[-1]:.4f}rad",
                )
            if abs(root_values[-1][1] - expected) > self.contract.final_position_tolerance_m + _EPS:
                self._fail("FINAL_POSITION_MISSED", f"final lateral travel is {root_values[-1][1]:.4f}m; expected {expected:.4f}m")
            if abs(com_values[-1][1] - expected) > self.contract.final_position_tolerance_m + self.contract.max_root_com_delta_disagreement_m + _EPS:
                self._fail("FINAL_COM_POSITION_MISSED", f"final COM lateral travel is {com_values[-1][1]:.4f}m")
            for sample in active:
                if abs(float(sample["vx"])) > self.contract.command_zero_epsilon + _EPS or abs(float(sample["wz"])) > self.contract.command_zero_epsilon + _EPS:
                    self._fail("LATERAL_COMMAND_ENVELOPE", "LateralStep requires |vx| and |wz| <= 1e-6", float(sample["t"]))
                if (
                    abs(float(sample["vy"]))
                    > self.contract.lateral_step_max_vy_mps
                    + self.contract.command_serialization_tolerance
                ):
                    self._fail("LATERAL_COMMAND_ENVELOPE", "LateralStep requires |vy| <= 0.40m/s", float(sample["t"]))
                if any(sample["actions"].values()):
                    self._fail("NAV2_ACTION_ACTIVE_DURING_LATERAL", "all Nav2 actions must be inactive during LateralStep", float(sample["t"]))
            signed_vy = [(1.0 if expected > 0 else -1.0) * float(sample["vy"]) for sample in active]
            if not signed_vy or max(signed_vy) < self.contract.lateral_step_max_vy_mps - self.contract.steady_target_tolerance - _EPS:
                self._fail("LATERAL_STEADY_TARGET_MISSING", "no signed 0.40m/s steady lateral command was observed")
        elif self.contract.expected_yaw_rad is not None:
            expected = self.contract.expected_yaw_rad
            direction = 1.0 if expected > 0 else -1.0
            directional = [direction * value for value in heading_values]
            translations = [math.hypot(*value) for value in root_values]
            active_indices = [
                index for index, sample in enumerate(samples) if sample["intent"]
            ]
            active_directional = [directional[index] for index in active_indices]
            self.metrics["turnTrackingEnvelope"] = {
                "activeMaxDirectionalYawRad": max(active_directional, default=0.0),
                "brakingMaxDirectionalYawRad": max(directional),
                "finalDirectionalYawRad": directional[-1],
                "targetMagnitudeRad": abs(expected),
            }
            for sample, value in zip(samples, directional):
                if value >= self.contract.min_turn_progress_rad:
                    progress_time = float(sample["t"])
                    break
            if min(directional) < -self.contract.max_turn_reverse_rad - _EPS:
                self._fail("TURN_REVERSED", f"turn reversed by {-min(directional):.4f}rad")
            if max(active_directional, default=0.0) > abs(expected) + self.contract.max_turn_overshoot_rad + _EPS:
                self._fail("TURN_OVERSHOOT", f"active turn reached {max(active_directional):.4f}rad")
            if max(directional) > abs(expected) + self.contract.max_turn_braking_overshoot_rad + _EPS:
                self._fail("TURN_BRAKING_OVERSHOOT", f"turn braking transient reached {max(directional):.4f}rad")
            if max(translations) > self.contract.max_turn_translation_m + _EPS:
                self._fail("TURN_TRANSLATION_EXCEEDED", f"turn translated {max(translations):.4f}m")
            if abs(heading_values[-1] - expected) > self.contract.final_yaw_tolerance_rad + _EPS:
                self._fail("FINAL_YAW_MISSED", f"final yaw delta is {heading_values[-1]:.4f}rad; expected {expected:.4f}rad")
            signed_commands = [direction * float(sample["wz"]) for sample in active]
            if min(signed_commands) < -self.contract.command_zero_epsilon - _EPS or max(signed_commands) <= self.contract.command_zero_epsilon:
                self._fail("TURN_COMMAND_DIRECTION_MISMATCH", "Spin command direction contradicts the formal goal")
            for sample in active:
                if (
                    abs(float(sample["vx"])) > self.contract.command_zero_epsilon + _EPS
                    or abs(float(sample["vy"])) > self.contract.command_zero_epsilon + _EPS
                    or abs(float(sample["wz"]))
                    > self.contract.spin_max_wz_rps
                    + self.contract.command_serialization_tolerance
                ):
                    self._fail(
                        "SPIN_COMMAND_ENVELOPE",
                        "Spin requires |vx|,|vy|<=1e-6 and |wz|<=0.20rad/s",
                        float(sample["t"]),
                    )
            if max(signed_commands) < self.contract.spin_max_wz_rps - self.contract.steady_target_tolerance - _EPS:
                self._fail(
                    "SPIN_STEADY_TARGET_MISSING",
                    "no signed 0.20rad/s Spin steady command was observed",
                )
        else:
            cumulative_translation = [0.0]
            cumulative_rotation = [0.0]
            for before, after in zip(samples, samples[1:]):
                cumulative_translation.append(
                    cumulative_translation[-1]
                    + math.dist(before["root"][:2], after["root"][:2])
                )
                cumulative_rotation.append(
                    cumulative_rotation[-1]
                    + abs(
                        float(after["unwrappedYaw"])
                        - float(before["unwrappedYaw"])
                    )
                )
            for sample, translation, rotation in zip(
                samples, cumulative_translation, cumulative_rotation
            ):
                if (
                    translation >= self.contract.min_linear_progress_m
                    or rotation >= self.contract.min_turn_progress_rad
                ):
                    progress_time = float(sample["t"])
                    break

        if progress_time is None:
            self._fail("MINIMUM_PROGRESS_NOT_REACHED", "minimum physical progress was not recorded")
        elif progress_time - first_intent > self.contract.progress_window_s + _EPS:
            self._fail("INSUFFICIENT_PROGRESS_15S", f"progress first appeared after {progress_time - first_intent:.3f}s")
        self.metrics["intentStartedS"] = first_intent - float(first["t"])
        self.metrics["progressReachedS"] = None if progress_time is None else progress_time - float(first["t"])

    def _evaluate_arm_trace(self) -> None:
        assert self.contract is not None
        action = _mapping(self.document.get("action"), "action")
        policy_activation_epoch = _nonnegative_int(
            self.arm_evidence.get("policyActivationEpoch"),
            "armLimiterEvidence.policyActivationEpoch",
        )
        support_release_epoch = _nonnegative_int(
            self.arm_evidence.get("supportReleaseConfirmedEpoch"),
            "armLimiterEvidence.supportReleaseConfirmedEpoch",
        )
        plant_boot_id = self.arm_evidence.get("plantBootId")
        plant_session_id = self.arm_evidence.get("plantSessionId")
        sonic_driver_boot_id = self.arm_evidence.get("sonicDriverBootId")
        sonic_driver_session_id = self.arm_evidence.get("sonicDriverSessionId")
        task_start_tick = _nonnegative_int(
            self.arm_evidence.get("taskStartControlTick"),
            "armLimiterEvidence.taskStartControlTick",
        )
        task_end_tick = _nonnegative_int(
            self.arm_evidence.get("taskEndControlTick"),
            "armLimiterEvidence.taskEndControlTick",
        )
        runtime = _mapping(self.metrics.get("runtimeHealth"), "metrics.runtimeHealth")
        runtime_boot_ids = _mapping(runtime.get("bootIds"), "metrics.runtimeHealth.bootIds")
        expected_policy_activation = _nonnegative_int(
            action.get("policyActivationEpoch"), "action.policyActivationEpoch"
        )
        expected_support_release = _nonnegative_int(
            action.get("supportReleaseConfirmedEpoch"),
            "action.supportReleaseConfirmedEpoch",
        )
        binding_mismatches: list[str] = []
        if policy_activation_epoch != expected_policy_activation:
            binding_mismatches.append("policyActivationEpoch")
        if support_release_epoch != expected_support_release:
            binding_mismatches.append("supportReleaseConfirmedEpoch")
        if plant_boot_id != runtime_boot_ids.get("plant"):
            binding_mismatches.append("plantBootId")
        if plant_session_id != runtime.get("plantSessionId"):
            binding_mismatches.append("plantSessionId")
        if sonic_driver_boot_id != runtime_boot_ids.get("sonic"):
            binding_mismatches.append("sonicDriverBootId")
        if sonic_driver_session_id != runtime.get("sonicSessionId"):
            binding_mismatches.append("sonicDriverSessionId")
        if self.arm_evidence.get("caseId") != self.document.get("caseId"):
            binding_mismatches.append("caseId")
        if task_end_tick <= task_start_tick:
            binding_mismatches.append("taskStart/EndControlTick")
        if binding_mismatches:
            self._fail(
                "ARM_TRACE_BINDING_MISMATCH",
                "arm limiter evidence is not bound to the accepted session: "
                + ", ".join(binding_mismatches),
            )
        if self.arm_evidence.get("source") != "sonic_authoritative_control_loop":
            self._fail(
                "ARM_TRACE_SOURCE_INVALID",
                "successful arm trace must come from the SONIC authoritative control loop",
            )
        if (
            self.arm_evidence.get("interpolated") is not False
            or self.arm_evidence.get("repeatedTelemetry") is not False
        ):
            self._fail(
                "ARM_TRACE_SOURCE_INVALID",
                "interpolated or repeated HTTP telemetry cannot prove 50 Hz arm safety",
            )

        previous_t: float | None = None
        previous_dt: float | None = None
        previous_tick: int | None = None
        previous_durations = {joint: 0.0 for joint in ARM_JOINT_INDICES}
        episode_starts: dict[int, float] = {}
        max_duration = 0.0
        episodes: list[dict[str, float | int]] = []
        final_constrained = 0
        for index, item in enumerate(self.arm_trace):
            label = f"armLimiterEvidence.records[{index}]"
            control_tick = _nonnegative_int(
                item.get("controlTick"), f"{label}.controlTick"
            )
            t_s = _finite(item.get("tS"), f"{label}.tS")
            dt_s = _finite(item.get("dtS"), f"{label}.dtS")
            if abs(dt_s - self.contract.arm_control_period_s) > 1e-6:
                self._fail("ARM_TRACE_INCOMPLETE", f"arm limiter dt is {dt_s:.6f}s, expected 0.020000s", t_s)
            if previous_t is not None:
                assert previous_dt is not None
                if abs((t_s - previous_t) - previous_dt) > 1e-6:
                    self._fail("ARM_TRACE_INCOMPLETE", "arm limiter control ticks are not contiguous", t_s)
            if previous_tick is not None and control_tick != previous_tick + 1:
                self._fail(
                    "ARM_TRACE_INCOMPLETE",
                    "arm limiter authoritative controlTick is not contiguous",
                    t_s,
                )
            constrained_raw = _sequence(
                item.get("constrainedJointIndices"),
                f"{label}.constrainedJointIndices",
            )
            constrained_indices = tuple(
                _nonnegative_int(value, f"{label}.constrainedJointIndices")
                for value in constrained_raw
            )
            if (
                len(set(constrained_indices)) != len(constrained_indices)
                or any(joint not in ARM_JOINT_INDICES for joint in constrained_indices)
            ):
                raise TraceContractError(
                    f"{label}.constrainedJointIndices must be unique G1 arm joints"
                )
            constrained = _nonnegative_int(
                item.get("constrainedJoints"), f"{label}.constrainedJoints"
            )
            if constrained != len(constrained_indices):
                self._fail(
                    "ARM_PER_JOINT_EVIDENCE_INVALID",
                    "constrainedJoints disagrees with constrainedJointIndices",
                    t_s,
                )
            durations_raw = _mapping(
                item.get("continuousSaturationByJointS"),
                f"{label}.continuousSaturationByJointS",
            )
            expected_keys = {str(joint) for joint in ARM_JOINT_INDICES}
            if set(durations_raw) != expected_keys:
                raise TraceContractError(
                    f"{label}.continuousSaturationByJointS must contain all 14 arm joints"
                )
            durations = {
                joint: _finite(
                    durations_raw[str(joint)],
                    f"{label}.continuousSaturationByJointS.{joint}",
                )
                for joint in ARM_JOINT_INDICES
            }
            for joint, duration in durations.items():
                is_constrained = joint in constrained_indices
                expected_duration = (
                    previous_durations[joint] + dt_s if is_constrained else 0.0
                )
                if duration < 0 or abs(duration - expected_duration) > 1e-6:
                    self._fail(
                        "ARM_PER_JOINT_EVIDENCE_INVALID",
                        f"joint {joint} duration {duration:.3f}s does not follow the 50 Hz recurrence",
                        t_s,
                    )
                if is_constrained and previous_durations[joint] <= _EPS:
                    episode_starts[joint] = t_s
                if not is_constrained and previous_durations[joint] > _EPS:
                    episode_start = episode_starts.pop(joint, None)
                    if episode_start is None:
                        self._fail(
                            "ARM_PER_JOINT_EVIDENCE_INVALID",
                            f"joint {joint} duration has no authoritative episode start",
                            t_s,
                        )
                        episode_start = t_s - previous_durations[joint]
                    clear_elapsed = t_s - episode_start
                    episodes.append(
                        {
                            "jointIndex": joint,
                            "startedAtS": episode_start,
                            "clearedAtS": t_s,
                            "durationS": previous_durations[joint],
                        }
                    )
                    if clear_elapsed >= self.contract.arm_must_clear_before_s - _EPS:
                        self._fail(
                            "ARM_SATURATION_NOT_CLEARED_BEFORE_250MS",
                            f"joint {joint} saturation cleared after {clear_elapsed:.3f}s",
                            t_s,
                        )
                max_duration = max(max_duration, duration)
                if duration > self.contract.max_arm_continuous_saturation_s + _EPS:
                    self._fail(
                        "ARM_SATURATION_TOO_LONG",
                        f"joint {joint} continuous saturation reached {duration:.3f}s",
                        t_s,
                    )
            previous_durations = durations
            previous_t, previous_dt = t_s, dt_s
            previous_tick = control_tick
            final_constrained = constrained
        if episode_starts:
            self._fail("ARM_SATURATION_NOT_CLEARED", "arm limiter remained constrained at trace end")
        first_control_tick = _nonnegative_int(
            self.arm_trace[0].get("controlTick"),
            "armLimiterEvidence.records[0].controlTick",
        )
        final_control_tick = _nonnegative_int(
            self.arm_trace[-1].get("controlTick"),
            "armLimiterEvidence.records[-1].controlTick",
        )
        if first_control_tick != task_start_tick or final_control_tick != task_end_tick:
            self._fail(
                "ARM_TRACE_BINDING_MISMATCH",
                "task start/end ticks do not match the complete arm record boundaries",
            )
        first_sample_t = _finite(self.samples[0].get("tS"), "samples[0].tS")
        final_sample_t = _finite(self.samples[-1].get("tS"), "samples[-1].tS")
        first_arm_t = _finite(self.arm_trace[0].get("tS"), "armLimiterEvidence.records[0].tS")
        final_arm_t = _finite(self.arm_trace[-1].get("tS"), "armLimiterEvidence.records[-1].tS")
        # ``evidence.begin`` commits before action admission, but the baseline
        # HTTP snapshot and the sole 50 Hz actuator owner are asynchronous. A
        # baseline captured between two control ticks may therefore precede
        # the first recorded actuator decision by less than one authoritative
        # period. No motion target is missing in that interval: the first
        # recorded decision is the first action-owned tick. Anything larger
        # than one period, or any terminal sample after the final record,
        # remains an incomplete trace.
        if (
            first_arm_t - first_sample_t
            > self.contract.arm_control_period_s + _EPS
            or final_arm_t < final_sample_t - _EPS
        ):
            self._fail(
                "ARM_TRACE_INCOMPLETE",
                "arm limiter trace does not cover the complete root/COM session",
            )
        if final_constrained != 0:
            self._fail("ARM_NOT_CLEAR_AT_END", "arm limiter is constrained at trace end")
        terminal_arm_start = final_sample_t - self.contract.terminal_settle_s
        if any(
            bool(item.get("constrainedJointIndices"))
            for item in self.arm_trace
            if _finite(item.get("tS"), "armLimiterEvidence.records.tS") >= terminal_arm_start - _EPS
        ):
            self._fail(
                "ARM_NOT_CLEAR_DURING_TERMINAL_SETTLE",
                "arm limiter was constrained during the final 1.0s settled interval",
            )
        self.metrics["armLimiter"] = {
            "maxContinuousSaturationS": max_duration,
            "episodes": episodes,
            "finalConstrainedJoints": final_constrained,
            "policyActivationEpoch": policy_activation_epoch,
            "supportReleaseConfirmedEpoch": support_release_epoch,
            "plantBootId": plant_boot_id,
            "plantSessionId": plant_session_id,
            "sonicDriverBootId": sonic_driver_boot_id,
            "sonicDriverSessionId": sonic_driver_session_id,
            "sonicDriverSourceSha256": self.arm_evidence.get(
                "sonicDriverSourceSha256"
            ),
            "traceSha256": self.arm_evidence.get("traceSha256"),
            "taskStartControlTick": task_start_tick,
            "taskEndControlTick": task_end_tick,
            "source": self.arm_evidence.get("source"),
        }

    def _evaluate_action(self) -> None:
        action = _mapping(self.document.get("action"), "action")
        expected_type = _EXPECTED_ACTION_TYPE[self.mode]
        if action.get("initiator") != _STRICT_RUNNER_IDENTITY:
            self._fail(
                "ACTION_INITIATOR_INVALID",
                "accepted motion actions must be initiated by strict_motion_runner",
            )
        if action.get("type") != expected_type:
            self._fail("ACTION_TYPE_MISMATCH", f"{self.mode} requires {expected_type}")
        if action.get("status") != "succeeded":
            self._fail("ACTION_NOT_SUCCEEDED", "action status is not succeeded")
        if action.get("balancedPolicyIdleConfirmed") is not True:
            self._fail(
                "BALANCED_POLICY_IDLE_NOT_CONFIRMED",
                "terminal zero-speed balanced-policy IDLE was not confirmed",
            )
        policy_activation_epoch = _nonnegative_int(
            action.get("policyActivationEpoch"), "action.policyActivationEpoch"
        )
        support_release_epoch = _nonnegative_int(
            action.get("supportReleaseConfirmedEpoch"),
            "action.supportReleaseConfirmedEpoch",
        )
        if policy_activation_epoch <= 0 or support_release_epoch <= 0:
            self._fail(
                "ACTIVATION_PROVENANCE_INVALID",
                "policy and support-release epochs must both be positive",
            )
        if support_release_epoch != self.metrics.get("supportReleaseConfirmedEpoch"):
            self._fail(
                "SUPPORT_RELEASE_EPOCH_MISMATCH",
                "action support-release epoch does not match Plant evidence",
            )
        provenance = action.get("activationProvenance")
        if provenance == "initial_support_release":
            provenance_valid = policy_activation_epoch == support_release_epoch
        elif provenance == "support_off_reinit":
            provenance_valid = support_release_epoch < policy_activation_epoch
        else:
            provenance_valid = False
        if not provenance_valid:
            self._fail(
                "ACTIVATION_PROVENANCE_INVALID",
                "initial activation requires equal epochs; support-off re-init requires release < policy activation",
            )
        runtime = _mapping(self.metrics.get("runtimeHealth"), "metrics.runtimeHealth")
        runtime_boot_ids = _mapping(runtime.get("bootIds"), "metrics.runtimeHealth.bootIds")
        if (
            action.get("plantBootId") != runtime_boot_ids.get("plant")
            or action.get("plantSessionId") != runtime.get("plantSessionId")
        ):
            self._fail(
                "ACTIVATION_SESSION_BINDING_MISMATCH",
                "activation provenance is not bound to the accepted Plant boot/session",
            )
        self.metrics["activationProvenance"] = {
            "policyActivationEpoch": policy_activation_epoch,
            "supportReleaseConfirmedEpoch": support_release_epoch,
            "kind": provenance,
            "plantBootId": action.get("plantBootId"),
            "plantSessionId": action.get("plantSessionId"),
        }
        goal = action.get("goal")
        claimed_hash = action.get("goalHashSha256")
        if not isinstance(claimed_hash, str) or not _SHA256.fullmatch(claimed_hash):
            raise TraceContractError("action.goalHashSha256 must be lowercase SHA-256")
        computed_hash = canonical_sha256(goal)
        self.metrics["actionGoalSha256"] = computed_hash
        if claimed_hash != computed_hash:
            self._fail("ACTION_GOAL_HASH_MISMATCH", "action goal hash does not match the canonical goal")
        if self.mode == "detour":
            self._evaluate_detour_sequence(action, computed_hash)

    def _evaluate_detour_sequence(
        self,
        action: Mapping[str, Any],
        computed_goal_hash: str,
    ) -> None:
        assert self.contract is not None
        plan_hash = action.get("plannerPlanHashSha256")
        if not isinstance(plan_hash, str) or not _SHA256.fullmatch(plan_hash):
            raise TraceContractError(
                "action.plannerPlanHashSha256 must be lowercase SHA-256"
            )
        raw_segments = _sequence(action.get("segments"), "action.segments")
        if not raw_segments:
            raise TraceContractError("detour action requires at least one segment")
        required_fields = {
            "sequenceIndex",
            "primitiveSequence",
            "type",
            "goal",
            "goalHashSha256",
            "resolverResult",
            "freshOdometryUsed",
            "freshOdometry",
            "lineOfSightRevalidated",
            "status",
            "balancedPolicyIdleConfirmed",
            "startedAtControlTick",
            "completedAtControlTick",
        }
        normalized: list[dict[str, Any]] = []
        previous_end_tick: int | None = None
        for index, raw_segment in enumerate(raw_segments):
            label = f"action.segments[{index}]"
            segment = _mapping(raw_segment, label)
            if set(segment) != required_fields:
                raise TraceContractError(
                    f"{label} must contain exactly the official segment evidence fields"
                )
            sequence_index = _nonnegative_int(
                segment.get("sequenceIndex"), f"{label}.sequenceIndex"
            )
            if sequence_index != index:
                self._fail(
                    "DETOUR_SEGMENT_ORDER_INVALID",
                    "detour segment indices must be contiguous and ordered",
                )
            primitive_sequence = _nonnegative_int(
                segment.get("primitiveSequence"), f"{label}.primitiveSequence"
            )
            segment_type = segment.get("type")
            if segment_type not in {"Spin", "DriveOnHeading"}:
                self._fail(
                    "DETOUR_FORBIDDEN_ACTION",
                    "detour execution may use only official Spin and DriveOnHeading actions",
                )
            if segment.get("status") != "succeeded":
                self._fail(
                    "DETOUR_SEGMENT_NOT_SUCCEEDED",
                    f"detour segment {index} did not succeed",
                )
            if segment.get("balancedPolicyIdleConfirmed") is not True:
                self._fail(
                    "DETOUR_SEGMENT_BALANCED_IDLE_MISSING",
                    f"detour segment {index} lacks confirmed balanced-policy IDLE",
                )
            started_tick = _nonnegative_int(
                segment.get("startedAtControlTick"),
                f"{label}.startedAtControlTick",
            )
            completed_tick = _nonnegative_int(
                segment.get("completedAtControlTick"),
                f"{label}.completedAtControlTick",
            )
            if completed_tick <= started_tick:
                self._fail(
                    "DETOUR_SEGMENT_TICK_RANGE_INVALID",
                    f"detour segment {index} has an empty/reversed control-tick range",
                )
            if previous_end_tick is not None and started_tick <= previous_end_tick:
                self._fail(
                    "DETOUR_SEGMENT_STOP_GAP_MISSING",
                    "successive detour segments require a non-overlapping stop gap",
                )
            previous_end_tick = completed_tick
            if segment.get("freshOdometryUsed") is not True:
                self._fail(
                    "DETOUR_FRESH_ODOMETRY_MISSING",
                    f"detour primitive {primitive_sequence} was not resolved from fresh odometry",
                )
            resolver_result = dict(
                _mapping(segment.get("resolverResult"), f"{label}.resolverResult")
            )
            fresh = _mapping(segment.get("freshOdometry"), f"{label}.freshOdometry")
            if set(fresh) != {
                "frameId",
                "x",
                "y",
                "yaw",
                "sampledAtControlTick",
                "ageS",
            }:
                raise TraceContractError(
                    f"{label}.freshOdometry must contain exact map-pose/tick/age fields"
                )
            fresh_tick = _nonnegative_int(
                fresh.get("sampledAtControlTick"),
                f"{label}.freshOdometry.sampledAtControlTick",
            )
            fresh_age = _finite(fresh.get("ageS"), f"{label}.freshOdometry.ageS")
            normalized_fresh = {
                "frameId": fresh.get("frameId"),
                "x": _finite(fresh.get("x"), f"{label}.freshOdometry.x"),
                "y": _finite(fresh.get("y"), f"{label}.freshOdometry.y"),
                "yaw": _finite(fresh.get("yaw"), f"{label}.freshOdometry.yaw"),
                "sampledAtControlTick": fresh_tick,
                "ageS": fresh_age,
            }
            if (
                normalized_fresh["frameId"] != "map"
                or fresh_tick != started_tick
                or fresh_age < 0
                or fresh_age > self.contract.max_fresh_odometry_age_s + _EPS
            ):
                self._fail(
                    "DETOUR_FRESH_ODOMETRY_INVALID",
                    "primitive resolution requires <=100ms map odometry sampled at its start tick",
                )
            line_of_sight = segment.get("lineOfSightRevalidated")
            if not isinstance(line_of_sight, bool):
                raise TraceContractError(
                    f"{label}.lineOfSightRevalidated must be boolean"
                )
            goal = _mapping(segment.get("goal"), f"{label}.goal")
            if segment_type == "DriveOnHeading":
                if set(goal) != {"distanceM", "speedMps"}:
                    raise TraceContractError(
                        f"{label}.goal must be exactly distanceM/speedMps"
                    )
                distance = _finite(goal.get("distanceM"), f"{label}.goal.distanceM")
                speed = _finite(goal.get("speedMps"), f"{label}.goal.speedMps")
                if distance <= 0 or abs(speed - self.contract.drive_on_heading_max_vx_mps) > _EPS:
                    self._fail(
                        "DETOUR_SEGMENT_GOAL_INVALID",
                        "DriveOnHeading detour goals require positive distance and locked 0.30m/s speed",
                    )
                if line_of_sight is not True:
                    self._fail(
                        "DETOUR_LINE_OF_SIGHT_NOT_REVALIDATED",
                        "DriveOnHeading requires fresh static-PGM line-of-sight revalidation",
                    )
                normalized_goal: dict[str, float] = {
                    "distanceM": distance,
                    "speedMps": speed,
                }
            else:
                if set(goal) != {"targetYawRad", "maxWzRps"}:
                    raise TraceContractError(
                        f"{label}.goal must be exactly targetYawRad/maxWzRps"
                    )
                target_yaw = _finite(
                    goal.get("targetYawRad"), f"{label}.goal.targetYawRad"
                )
                max_wz = _finite(goal.get("maxWzRps"), f"{label}.goal.maxWzRps")
                if abs(target_yaw) <= self.contract.command_zero_epsilon or abs(
                    max_wz - self.contract.spin_max_wz_rps
                ) > _EPS:
                    self._fail(
                        "DETOUR_SEGMENT_GOAL_INVALID",
                        "Spin detour goals require non-zero target yaw and locked 0.20rad/s speed",
                    )
                normalized_goal = {
                    "targetYawRad": target_yaw,
                    "maxWzRps": max_wz,
                }
            claimed_goal_hash = segment.get("goalHashSha256")
            if not isinstance(claimed_goal_hash, str) or not _SHA256.fullmatch(
                claimed_goal_hash
            ):
                raise TraceContractError(f"{label}.goalHashSha256 must be lowercase SHA-256")
            computed_segment_hash = canonical_sha256(normalized_goal)
            if claimed_goal_hash != computed_segment_hash:
                self._fail(
                    "DETOUR_SEGMENT_GOAL_HASH_MISMATCH",
                    f"detour segment {index} goal hash does not match",
                )
            normalized.append(
                {
                    "sequenceIndex": index,
                    "primitiveSequence": primitive_sequence,
                    "type": segment_type,
                    "goal": normalized_goal,
                    "goalHashSha256": computed_segment_hash,
                    "resolverResult": resolver_result,
                    "freshOdometryUsed": True,
                    "freshOdometry": normalized_fresh,
                    "lineOfSightRevalidated": line_of_sight,
                    "startedAtControlTick": started_tick,
                    "completedAtControlTick": completed_tick,
                }
            )

        sequence_payload = {
            "goalHashSha256": computed_goal_hash,
            "plannerPlanHashSha256": plan_hash,
            "segments": normalized,
        }
        computed_sequence_hash = canonical_sha256(sequence_payload)
        claimed_sequence_hash = action.get("sequenceHashSha256")
        if not isinstance(claimed_sequence_hash, str) or not _SHA256.fullmatch(
            claimed_sequence_hash
        ):
            raise TraceContractError("action.sequenceHashSha256 must be lowercase SHA-256")
        if claimed_sequence_hash != computed_sequence_hash:
            self._fail(
                "DETOUR_SEQUENCE_HASH_MISMATCH",
                "detour primitive sequence hash does not match order/goals/ticks/plan",
            )
        self.detour_segments = normalized
        self._evaluate_detour_segment_samples()
        self.metrics["detourPrimitiveSequence"] = {
            "segmentCount": len(normalized),
            "sequenceSha256": computed_sequence_hash,
            "plannerPlanSha256": plan_hash,
            "allowedTypes": ["Spin", "DriveOnHeading"],
        }

    def _evaluate_detour_segment_samples(self) -> None:
        assert self.contract is not None
        arm_start_tick = _nonnegative_int(
            self.arm_evidence.get("taskStartControlTick"),
            "armLimiterEvidence.taskStartControlTick",
        )
        arm_end_tick = _nonnegative_int(
            self.arm_evidence.get("taskEndControlTick"),
            "armLimiterEvidence.taskEndControlTick",
        )
        for index, segment in enumerate(self.detour_segments):
            if (
                int(segment["startedAtControlTick"]) < arm_start_tick
                or int(segment["completedAtControlTick"]) > arm_end_tick
            ):
                self._fail(
                    "DETOUR_SEGMENT_TICK_RANGE_INVALID",
                    f"detour segment {index} lies outside the authoritative task ticks",
                )
        sample_ticks = [int(sample["sonicControlTick"]) for sample in self._parsed_samples]
        if any(after <= before for before, after in zip(sample_ticks, sample_ticks[1:])):
            raise TraceContractError("detour sonicControlTick values must strictly advance")
        # The authoritative arm trace deliberately starts before root/COM samples:
        # it includes the measured-current activation and fall-arrest release
        # transaction.  Accepted motion samples begin only after support-off is
        # confirmed, so requiring their first tick to equal the arm start would
        # either discard the activation evidence or falsely label tethered frames
        # as support-off.  The sampled interval must still be a strict subset of
        # the arm trace. Finalizing the recorder after the last joined runtime
        # sample may append at most two real 50 Hz zero-command records.
        if (
            sample_ticks[0] < arm_start_tick
            or sample_ticks[-1] > arm_end_tick
            or arm_end_tick - sample_ticks[-1] > 2
        ):
            self._fail(
                "DETOUR_SAMPLE_TICK_BINDING_MISMATCH",
                "detour samples must lie inside the authoritative arm task range with at most two finalization ticks",
            )
        observed_by_segment: dict[int, list[int]] = {
            index: [] for index in range(len(self.detour_segments))
        }
        settled_by_segment = {
            index: False for index in range(len(self.detour_segments))
        }
        observed_order: list[int] = []
        previous_index: int | None = None
        for sample_index, sample in enumerate(self._parsed_samples):
            segment_index = sample["detourSegmentIndex"]
            active_actions = [
                name for name, enabled in sample["actions"].items() if enabled
            ]
            if not sample["intent"]:
                if segment_index is None:
                    if active_actions:
                        self._fail(
                            "NAV2_ACTION_EVIDENCE_MISMATCH",
                            "terminal detour samples must have every Nav2 action inactive",
                            float(sample["t"]),
                        )
                    if max(abs(float(sample[name])) for name in ("vx", "vy", "wz")) > self.contract.command_zero_epsilon + _EPS:
                        self._fail(
                            "DETOUR_STOP_COMMAND_NONZERO",
                            "terminal detour samples require zero commands",
                            float(sample["t"]),
                        )
                    if (
                        float(sample["linearSpeed"])
                        > self.contract.terminal_linear_speed_mps + _EPS
                        or float(sample["angularSpeed"])
                        > self.contract.terminal_angular_speed_rps + _EPS
                    ):
                        self._fail(
                            "DETOUR_STOP_NOT_SETTLED",
                            "terminal detour samples must prove physical zero-speed balanced-policy IDLE",
                            float(sample["t"]),
                        )
                    continue
            if segment_index is None or segment_index not in observed_by_segment:
                self._fail(
                    "DETOUR_SEGMENT_SAMPLE_BINDING_MISMATCH",
                    "each active detour sample must identify a valid segment",
                    float(sample["t"]),
                )
                continue
            segment = self.detour_segments[segment_index]
            expected_action = (
                "driveOnHeadingActive"
                if segment["type"] == "DriveOnHeading"
                else "spinActive"
            )
            forbidden = any(
                sample["actions"][name]
                for name in ("navigateToPoseActive", "followPathActive")
            )
            if forbidden or any(
                action != expected_action for action in active_actions
            ):
                self._fail(
                    "DETOUR_FORBIDDEN_ACTION" if forbidden else "NAV2_ACTION_EVIDENCE_MISMATCH",
                    "detour samples may contain only their declared Spin/DriveOnHeading action",
                    float(sample["t"]),
                )
            tick = int(sample["sonicControlTick"])
            if not segment["startedAtControlTick"] <= tick <= segment["completedAtControlTick"]:
                self._fail(
                    "DETOUR_SEGMENT_SAMPLE_BINDING_MISMATCH",
                    "segment sample tick lies outside its declared action/settle interval",
                    float(sample["t"]),
                )
            command_nonzero = (
                max(abs(float(sample[name])) for name in ("vx", "vy", "wz"))
                > self.contract.command_zero_epsilon + _EPS
            )
            if not command_nonzero:
                if (
                    not active_actions
                    and sample["balancedPolicyIdle"]
                    and float(sample["linearSpeed"])
                    <= self.contract.terminal_linear_speed_mps + _EPS
                    and float(sample["angularSpeed"])
                    <= self.contract.terminal_angular_speed_rps + _EPS
                ):
                    settled_by_segment[segment_index] = True
                continue
            if segment["type"] == "DriveOnHeading":
                if (
                    abs(float(sample["vy"])) > self.contract.command_zero_epsilon + _EPS
                    or abs(float(sample["wz"])) > self.contract.command_zero_epsilon + _EPS
                    or float(sample["vx"]) <= self.contract.command_zero_epsilon
                    or float(sample["vx"])
                    > self.contract.drive_on_heading_max_vx_mps
                    + self.contract.command_serialization_tolerance
                ):
                    self._fail(
                        "DRIVE_ON_HEADING_COMMAND_ENVELOPE",
                        "detour DriveOnHeading requires 0<vx<=0.30m/s and zero vy/wz",
                        float(sample["t"]),
                    )
            else:
                target = float(segment["goal"]["targetYawRad"])
                signed_wz = math.copysign(1.0, target) * float(sample["wz"])
                if (
                    abs(float(sample["vx"])) > self.contract.command_zero_epsilon + _EPS
                    or abs(float(sample["vy"])) > self.contract.command_zero_epsilon + _EPS
                    or signed_wz <= self.contract.command_zero_epsilon
                    or abs(float(sample["wz"]))
                    > self.contract.spin_max_wz_rps
                    + self.contract.command_serialization_tolerance
                ):
                    self._fail(
                        "SPIN_COMMAND_ENVELOPE",
                        "detour Spin requires target-signed 0<|wz|<=0.20rad/s and zero vx/vy",
                        float(sample["t"]),
                    )

            # At least one joined sample must prove the official action was
            # active while its signed command was non-zero. Transition samples
            # may briefly see one source clear before another; they remain
            # bound to the same segment and still undergo command/pose/contact
            # validation, but cannot establish action ownership by themselves.
            if active_actions == [expected_action]:
                if tick >= segment["completedAtControlTick"]:
                    self._fail(
                        "DETOUR_SEGMENT_SAMPLE_BINDING_MISMATCH",
                        "commanding sample reached or exceeded the completed action tick",
                        float(sample["t"]),
                    )
                observed_by_segment[segment_index].append(sample_index)
                if segment_index != previous_index:
                    observed_order.append(segment_index)
                    previous_index = segment_index

        expected_order = list(range(len(self.detour_segments)))
        if observed_order != expected_order:
            self._fail(
                "DETOUR_SEGMENT_ORDER_INVALID",
                f"observed segment order {observed_order} != {expected_order}",
            )
        for index, observed in observed_by_segment.items():
            if not observed:
                self._fail(
                    "DETOUR_SEGMENT_NOT_OBSERVED",
                    f"detour segment {index} has no active sample",
                )
                continue
            if not settled_by_segment[index]:
                self._fail(
                    "DETOUR_STOP_NOT_SETTLED",
                    f"detour segment {index} lacks a measured balanced-IDLE stop",
                )
                continue
            if any(
                self._parsed_samples[sample_index]["detourSegmentIndex"] != index
                for sample_index in range(observed[0], observed[-1] + 1)
            ):
                self._fail(
                    "DETOUR_SEGMENT_INTERRUPTED",
                    f"detour segment {index} was interleaved with another segment",
                )
            segment = self.detour_segments[index]
            active_samples = [self._parsed_samples[item] for item in observed]
            fresh = _mapping(segment.get("freshOdometry"), "detourSegment.freshOdometry")
            first_active = active_samples[0]
            if (
                math.dist(
                    (
                        _finite(fresh.get("x"), "detourSegment.freshOdometry.x"),
                        _finite(fresh.get("y"), "detourSegment.freshOdometry.y"),
                    ),
                    tuple(first_active["root"][:2]),
                )
                > self.contract.detour_endpoint_tolerance_m + _EPS
                or abs(
                    _angle_delta(
                        _finite(fresh.get("yaw"), "detourSegment.freshOdometry.yaw"),
                        float(first_active["yaw"]),
                    )
                )
                > self.contract.detour_heading_tolerance_rad + _EPS
            ):
                self._fail(
                    "DETOUR_FRESH_ODOMETRY_SAMPLE_MISMATCH",
                    f"primitive {index} fresh pose does not match its first authoritative sample",
                )
            if segment["type"] == "DriveOnHeading":
                steady = max(float(sample["vx"]) for sample in active_samples)
                target = self.contract.drive_on_heading_max_vx_mps
            else:
                sign = math.copysign(1.0, float(segment["goal"]["targetYawRad"]))
                steady = max(sign * float(sample["wz"]) for sample in active_samples)
                target = self.contract.spin_max_wz_rps
            if steady < target - self.contract.steady_target_tolerance - _EPS:
                self._fail(
                    "DETOUR_SEGMENT_STEADY_TARGET_MISSING",
                    f"detour segment {index} never observed its locked 0.30 steady target",
                )
            if index + 1 < len(self.detour_segments):
                next_observed = observed_by_segment[index + 1]
                if next_observed and next_observed[0] <= observed[-1] + 1:
                    self._fail(
                        "DETOUR_SEGMENT_STOP_GAP_MISSING",
                        f"detour segment {index} lacks an observed balanced-IDLE sample",
                    )

    def _evaluate_static_primitive_plan(
        self,
        navfn_path_payload: Mapping[str, Any],
        plan_poses: Sequence[tuple[float, float]],
        collision_source_hash: str,
    ) -> str:
        assert self.contract is not None
        plan = _mapping(self.document.get("staticPrimitivePlan"), "staticPrimitivePlan")
        required_top = {
            "schema_version",
            "motion_mode",
            "planner_source",
            "map_provenance",
            "runtime_provenance",
            "segmenter",
            "execution_policy",
            "segment_endpoint_indices",
            "segments",
            "primitives",
            "plan_sha256",
        }
        if set(plan) != required_top:
            raise TraceContractError(
                "staticPrimitivePlan must match the exact production v1 schema"
            )
        if (
            plan.get("schema_version") != STATIC_PRIMITIVE_PLAN_SCHEMA_VERSION
            or plan.get("motion_mode") != "static_spin_drive_primitives"
        ):
            self._fail(
                "STATIC_PRIMITIVE_PLAN_SCHEMA_INVALID",
                "detour requires the production static Spin/DriveOnHeading plan",
            )
        claimed_plan_hash = plan.get("plan_sha256")
        if not isinstance(claimed_plan_hash, str) or not _SHA256.fullmatch(
            claimed_plan_hash
        ):
            raise TraceContractError(
                "staticPrimitivePlan.plan_sha256 must be lowercase SHA-256"
            )
        computed_plan_hash = canonical_sha256(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        if claimed_plan_hash != computed_plan_hash:
            self._fail(
                "STATIC_PRIMITIVE_PLAN_HASH_MISMATCH",
                "static primitive plan SHA-256 does not match its complete payload",
            )
        production_parameters = (
            self.contract.robot_radius_m,
            self.contract.footprint_padding_m,
            self.contract.map_resolution_m,
            0.16,
            0.50,
            self.contract.drive_on_heading_max_vx_mps,
            self.contract.detour_endpoint_tolerance_m,
            self.contract.detour_heading_tolerance_rad,
        )
        plan_json = _canonical_json(plan)
        navfn_path_json = _canonical_json(navfn_path_payload)
        try:
            verified_plan = _verify_production_static_plan_cached(
                plan_json,
                navfn_path_json,
                production_parameters,
            )
        except (OSError, TypeError, ValueError) as exc:
            self._fail(
                "STATIC_PRIMITIVE_PRODUCTION_VERIFICATION_FAILED",
                f"production static primitive verifier rejected the artifact: {exc}",
            )
            self.metrics["staticPrimitivePlan"] = {
                "schemaVersion": plan.get("schema_version"),
                "planSha256": computed_plan_hash,
                "productionVerified": False,
            }
            return computed_plan_hash
        if dict(verified_plan) != dict(plan):
            self._fail(
                "STATIC_PRIMITIVE_PRODUCTION_VERIFICATION_FAILED",
                "production verifier did not return the exact supplied artifact",
            )

        planner = _mapping(plan.get("planner_source"), "staticPrimitivePlan.planner_source")
        if set(planner) != {
            "action_type",
            "planner_id",
            "plugin",
            "navfn_path_sha256",
            "navfn_path_pose_count",
        }:
            raise TraceContractError("static primitive planner_source fields are invalid")
        navfn_hash = canonical_sha256(navfn_path_payload)
        if (
            planner.get("action_type") != "nav2_msgs/action/ComputePathToPose"
            or planner.get("planner_id") != "GridBased"
            or planner.get("plugin") != "nav2_navfn_planner::NavfnPlanner"
            or planner.get("navfn_path_sha256") != navfn_hash
            or _nonnegative_int(
                planner.get("navfn_path_pose_count"),
                "staticPrimitivePlan.planner_source.navfn_path_pose_count",
            )
            != len(plan_poses)
        ):
            self._fail(
                "STATIC_PRIMITIVE_PLANNER_PROVENANCE_INVALID",
                "static plan is not bound to the accepted NavFn ComputePathToPose result",
            )

        map_provenance = _mapping(
            plan.get("map_provenance"), "staticPrimitivePlan.map_provenance"
        )
        if set(map_provenance) != {
            "yaml_sha256",
            "image_sha256",
            "scene_source_sha256",
        }:
            raise TraceContractError("static primitive map provenance fields are invalid")
        for key, value in map_provenance.items():
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise TraceContractError(f"staticPrimitivePlan.map_provenance.{key} must be SHA-256")
        if map_provenance.get("image_sha256") != collision_source_hash:
            self._fail(
                "STATIC_PRIMITIVE_MAP_PROVENANCE_MISMATCH",
                "static plan image hash differs from the accepted generated PGM",
            )

        runtime = _mapping(
            plan.get("runtime_provenance"), "staticPrimitivePlan.runtime_provenance"
        )
        if set(runtime) != {
            "runtime_manifest_sha256",
            "runtime_schema_version",
            "motion_profile_manifest_sha256",
            "motion_profile_schema_version",
            "follow_path_motion_accepted",
        }:
            raise TraceContractError("static primitive runtime provenance fields are invalid")
        for key in ("runtime_manifest_sha256", "motion_profile_manifest_sha256"):
            value = runtime.get(key)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise TraceContractError(f"staticPrimitivePlan.runtime_provenance.{key} must be SHA-256")
        if (
            runtime.get("runtime_schema_version") != "h-ide.office-v2-runtime/1"
            or runtime.get("motion_profile_schema_version")
            != "h-ide.nav2-motion-profile/1"
            or runtime.get("follow_path_motion_accepted") is not False
        ):
            self._fail(
                "STATIC_PRIMITIVE_RUNTIME_PROVENANCE_INVALID",
                "runtime/motion profile provenance does not lock FollowPath off",
            )

        segmenter = _mapping(plan.get("segmenter"), "staticPrimitivePlan.segmenter")
        expected_segmenter = {
            "robot_radius_m": self.contract.robot_radius_m,
            "footprint_padding_m": self.contract.footprint_padding_m,
            "extra_clearance_m": self.contract.map_resolution_m,
            "clearance_radius_m": self.contract.required_obstacle_boundary_clearance_m,
            "min_segment_m": 0.16,
            "max_segment_m": 0.5,
        }
        if set(segmenter) != {"source_sha256", *expected_segmenter}:
            raise TraceContractError("static primitive segmenter fields are invalid")
        source_hash = segmenter.get("source_sha256")
        if not isinstance(source_hash, str) or not _SHA256.fullmatch(source_hash):
            raise TraceContractError("staticPrimitivePlan.segmenter.source_sha256 must be SHA-256")
        if any(
            abs(_finite(segmenter.get(key), f"staticPrimitivePlan.segmenter.{key}") - expected)
            > _EPS
            for key, expected in expected_segmenter.items()
        ):
            self._fail(
                "STATIC_PRIMITIVE_SEGMENTER_CONTRACT_MISMATCH",
                "static segmenter radius/padding/clearance/distance contract drifted",
            )

        policy = _mapping(
            plan.get("execution_policy"), "staticPrimitivePlan.execution_policy"
        )
        expected_policy: dict[str, Any] = {
            "fresh_odometry_required": True,
            "fresh_pose_frame": "map",
            "revalidate_line_of_sight_before_drive": True,
            "endpoint_tolerance_m": self.contract.detour_endpoint_tolerance_m,
            "heading_tolerance_rad": self.contract.detour_heading_tolerance_rad,
            "drive_speed_mps": self.contract.drive_on_heading_max_vx_mps,
            "spin_min_angular_rps": 0.10,
            "spin_max_angular_rps": self.contract.spin_max_wz_rps,
            "spin_angular_accel_rps2": 0.30,
            "zero_spin_policy": "skip_within_heading_tolerance",
            "final_heading_policy": (
                "omit_checkpoint_if_within_heading_tolerance"
            ),
            "short_residual_policy": "skip_within_endpoint_tolerance_else_fail_below_min_segment",
        }
        if set(policy) != set(expected_policy):
            raise TraceContractError("static primitive execution policy fields are invalid")
        for key, expected in expected_policy.items():
            actual = policy.get(key)
            if isinstance(expected, float):
                valid = abs(_finite(actual, f"staticPrimitivePlan.execution_policy.{key}") - expected) <= _EPS
            else:
                valid = actual == expected and type(actual) is type(expected)
            if not valid:
                self._fail(
                    "STATIC_PRIMITIVE_EXECUTION_POLICY_MISMATCH",
                    f"static primitive execution policy {key} drifted",
                )

        raw_endpoints = _sequence(
            plan.get("segment_endpoint_indices"),
            "staticPrimitivePlan.segment_endpoint_indices",
        )
        endpoints = [
            _nonnegative_int(value, "staticPrimitivePlan.segment_endpoint_indices")
            for value in raw_endpoints
        ]
        if (
            len(endpoints) < 2
            or endpoints[0] != 0
            or endpoints[-1] != len(plan_poses) - 1
            or endpoints != sorted(set(endpoints))
        ):
            self._fail(
                "STATIC_PRIMITIVE_ENDPOINTS_INVALID",
                "segment endpoints must be ordered unique NavFn pose indices spanning the path",
            )
        raw_segments = _sequence(plan.get("segments"), "staticPrimitivePlan.segments")
        if len(raw_segments) != max(0, len(endpoints) - 1):
            self._fail(
                "STATIC_PRIMITIVE_SEGMENTS_INVALID",
                "segment count does not match endpoint partition",
            )
        normalized_segments: list[dict[str, Any]] = []
        segment_fields = {
            "id",
            "path_start_index",
            "path_end_index",
            "start_map",
            "end_map",
            "distance_m",
            "heading_map_yaw_rad",
        }
        for index, raw_segment in enumerate(raw_segments):
            segment = _mapping(raw_segment, f"staticPrimitivePlan.segments[{index}]")
            if set(segment) != segment_fields:
                raise TraceContractError(f"static primitive segment {index} fields are invalid")
            start_index = _nonnegative_int(segment.get("path_start_index"), "segment.path_start_index")
            end_index = _nonnegative_int(segment.get("path_end_index"), "segment.path_end_index")
            start_map = _mapping(segment.get("start_map"), "segment.start_map")
            end_map = _mapping(segment.get("end_map"), "segment.end_map")
            if set(start_map) != {"x", "y"} or set(end_map) != {"x", "y"}:
                raise TraceContractError("static primitive segment map points must be exact x/y")
            start = (_finite(start_map.get("x"), "segment.start_map.x"), _finite(start_map.get("y"), "segment.start_map.y"))
            end = (_finite(end_map.get("x"), "segment.end_map.x"), _finite(end_map.get("y"), "segment.end_map.y"))
            distance = _finite(segment.get("distance_m"), "segment.distance_m")
            heading = _finite(segment.get("heading_map_yaw_rad"), "segment.heading_map_yaw_rad")
            expected_start_index = endpoints[index] if index < len(endpoints) - 1 else -1
            expected_end_index = endpoints[index + 1] if index + 1 < len(endpoints) else -1
            mismatch = (
                segment.get("id") != f"segment-{index:03d}"
                or start_index != expected_start_index
                or end_index != expected_end_index
                or start_index >= len(plan_poses)
                or end_index >= len(plan_poses)
            )
            if not mismatch:
                expected_start = plan_poses[start_index]
                expected_end = plan_poses[end_index]
                expected_distance = math.dist(expected_start, expected_end)
                expected_heading = math.atan2(
                    expected_end[1] - expected_start[1],
                    expected_end[0] - expected_start[0],
                )
                mismatch = (
                    math.dist(start, expected_start) > 1e-6
                    or math.dist(end, expected_end) > 1e-6
                    or abs(distance - expected_distance) > 1e-6
                    or abs(_angle_delta(heading, expected_heading)) > 1e-6
                    or not 0.16 - _EPS <= distance <= 0.5 + _EPS
                )
            if mismatch:
                self._fail(
                    "STATIC_PRIMITIVE_SEGMENTS_INVALID",
                    f"segment {index} does not match its NavFn endpoint geometry",
                )
            normalized_segments.append(
                {
                    "id": segment.get("id"),
                    "startIndex": start_index,
                    "endIndex": end_index,
                    "start": start,
                    "end": end,
                    "distance": distance,
                    "heading": heading,
                }
            )

        primitives = _sequence(plan.get("primitives"), "staticPrimitivePlan.primitives")
        if len(primitives) not in {
            2 * len(normalized_segments),
            2 * len(normalized_segments) + 1,
        }:
            self._fail(
                "STATIC_PRIMITIVE_CHECKPOINTS_INVALID",
                "primitive checkpoints must alternate Spin/Drive with at most one final Spin",
            )
        expected_actions: list[tuple[int, str]] = []
        for segment_index, segment in enumerate(normalized_segments):
            spin_index = 2 * segment_index
            drive_index = spin_index + 1
            spin = _mapping(primitives[spin_index], f"staticPrimitivePlan.primitives[{spin_index}]")
            drive = _mapping(primitives[drive_index], f"staticPrimitivePlan.primitives[{drive_index}]")
            expected_spin_fields = {
                "sequence",
                "kind",
                "checkpoint_role",
                "action_type",
                "segment_id",
                "path_index",
                "reference_end_map",
                "planned_target_map_yaw_rad",
                "requires_fresh_odometry",
            }
            expected_drive_fields = {
                "sequence",
                "kind",
                "checkpoint_role",
                "action_type",
                "segment_id",
                "path_end_index",
                "end_map",
                "speed_mps",
                "requires_fresh_odometry",
            }
            spin_reference = _mapping(spin.get("reference_end_map"), "spin.reference_end_map")
            drive_end = _mapping(drive.get("end_map"), "drive.end_map")
            spin_valid = (
                set(spin) == expected_spin_fields
                and spin.get("sequence") == spin_index
                and spin.get("kind") == "spin"
                and spin.get("checkpoint_role") == "align_segment"
                and spin.get("action_type") == "nav2_msgs/action/Spin"
                and spin.get("segment_id") == segment["id"]
                and spin.get("path_index") == segment["startIndex"]
                and spin.get("requires_fresh_odometry") is True
                and set(spin_reference) == {"x", "y"}
                and math.dist(
                    (_finite(spin_reference.get("x"), "spin.reference.x"), _finite(spin_reference.get("y"), "spin.reference.y")),
                    segment["end"],
                )
                <= 1e-6
                and abs(_angle_delta(_finite(spin.get("planned_target_map_yaw_rad"), "spin.target_yaw"), segment["heading"])) <= 1e-6
            )
            drive_valid = (
                set(drive) == expected_drive_fields
                and drive.get("sequence") == drive_index
                and drive.get("kind") == "drive_on_heading"
                and drive.get("checkpoint_role") == "reach_segment_endpoint"
                and drive.get("action_type") == "nav2_msgs/action/DriveOnHeading"
                and drive.get("segment_id") == segment["id"]
                and drive.get("path_end_index") == segment["endIndex"]
                and drive.get("requires_fresh_odometry") is True
                and set(drive_end) == {"x", "y"}
                and math.dist(
                    (_finite(drive_end.get("x"), "drive.end.x"), _finite(drive_end.get("y"), "drive.end.y")),
                    segment["end"],
                )
                <= 1e-6
                and abs(_finite(drive.get("speed_mps"), "drive.speed_mps") - self.contract.drive_on_heading_max_vx_mps) <= _EPS
            )
            if not spin_valid or not drive_valid:
                self._fail(
                    "STATIC_PRIMITIVE_CHECKPOINTS_INVALID",
                    f"segment {segment_index} Spin/Drive checkpoints are not production schema",
                )
            expected_actions.append((spin_index, "Spin"))
            expected_actions.append((drive_index, "DriveOnHeading"))

        final_pose = _mapping(navfn_path_payload.get("poses", [{}])[-1], "navfn final pose")
        final_target_yaw = _finite(final_pose.get("yaw"), "navfn final yaw")
        terminal_heading_error = abs(
            _angle_delta(
                final_target_yaw,
                float(normalized_segments[-1]["heading"]),
            )
        )
        if len(primitives) == 2 * len(normalized_segments) + 1:
            final_index = len(primitives) - 1
            final = _mapping(primitives[final_index], f"staticPrimitivePlan.primitives[{final_index}]")
            final_fields = {
                "sequence",
                "kind",
                "checkpoint_role",
                "action_type",
                "path_index",
                "reference_end_map",
                "planned_target_map_yaw_rad",
                "requires_fresh_odometry",
            }
            final_reference = _mapping(final.get("reference_end_map"), "final.reference_end_map")
            final_valid = (
                terminal_heading_error
                > self.contract.detour_heading_tolerance_rad + _EPS
                and set(final) == final_fields
                and final.get("sequence") == final_index
                and final.get("kind") == "spin"
                and final.get("checkpoint_role") == "final_heading"
                and final.get("action_type") == "nav2_msgs/action/Spin"
                and final.get("path_index") == len(plan_poses) - 1
                and final.get("requires_fresh_odometry") is True
                and set(final_reference) == {"x", "y"}
                and math.dist(
                    (_finite(final_reference.get("x"), "final.reference.x"), _finite(final_reference.get("y"), "final.reference.y")),
                    plan_poses[-1],
                )
                <= 1e-6
                and abs(_angle_delta(_finite(final.get("planned_target_map_yaw_rad"), "final.target_yaw"), final_target_yaw)) <= 1e-6
            )
            if not final_valid:
                self._fail(
                    "STATIC_PRIMITIVE_CHECKPOINTS_INVALID",
                    "final heading checkpoint is not production schema",
                )
            expected_actions.append((final_index, "Spin"))
        elif terminal_heading_error > self.contract.detour_heading_tolerance_rad + _EPS:
            self._fail(
                "STATIC_PRIMITIVE_CHECKPOINTS_INVALID",
                "material final heading error requires an explicit final Spin",
            )

        # A learned-policy Spin may finish outside the 0.05 rad Drive gate
        # after its mandatory balanced-IDLE settle.  The live runner may then
        # repeat the immediately preceding, freshly-resolved Spin at most four
        # times.  Each repetition remains an ordinary official action with its
        # own pose, goal hash, resolver result, stop gap, and control ticks, and
        # each correction must materially contract the preceding heading
        # error so an oscillating controller cannot consume the larger budget.
        expanded_expected_actions: list[tuple[int, str]] = []
        skipped_resolver_results: list[dict[str, Any]] = []
        actual_cursor = 0
        realignment_spin_count = 0
        for primitive_sequence, expected_type in expected_actions:
            if expected_type == "Spin":
                candidate = (
                    self.detour_segments[actual_cursor]
                    if actual_cursor < len(self.detour_segments)
                    else None
                )
                if (
                    candidate is not None
                    and candidate.get("type") == "Spin"
                    and candidate.get("primitiveSequence") == primitive_sequence
                ):
                    expanded_expected_actions.append(
                        (primitive_sequence, expected_type)
                    )
                    actual_cursor += 1
                    continue

                # A planned alignment checkpoint is evidence-bearing even when
                # it emits no ROS action.  Re-evaluate it from the immediately
                # following DriveOnHeading's fresh pose; that same pose must
                # prove the Spin was within tolerance and the Drive executable.
                if (
                    candidate is not None
                    and candidate.get("type") == "DriveOnHeading"
                    and candidate.get("primitiveSequence")
                    == primitive_sequence + 1
                ):
                    fresh = _mapping(
                        candidate.get("freshOdometry"),
                        "detour skipped Spin freshOdometry",
                    )
                    fresh_pose = {
                        "x": _finite(fresh.get("x"), "detour skipped Spin fresh x"),
                        "y": _finite(fresh.get("y"), "detour skipped Spin fresh y"),
                        "yaw": _finite(
                            fresh.get("yaw"), "detour skipped Spin fresh yaw"
                        ),
                    }
                    try:
                        skipped = dict(
                            _resolve_production_static_primitive_cached(
                                plan_json,
                                navfn_path_json,
                                primitive_sequence,
                                _canonical_json(fresh_pose),
                                production_parameters,
                            )
                        )
                    except (OSError, TypeError, ValueError) as exc:
                        self._fail(
                            "DETOUR_PRODUCTION_RESOLUTION_FAILED",
                            f"production resolver rejected skipped primitive {primitive_sequence}: {exc}",
                        )
                    else:
                        if (
                            skipped.get("disposition")
                            == "skip_within_heading_tolerance"
                            and skipped.get("action_goal") is None
                        ):
                            skipped_resolver_results.append(skipped)
                            continue
                        self._fail(
                            "DETOUR_CHECKPOINT_NOT_EXECUTABLE",
                            f"omitted Spin primitive {primitive_sequence} was not within heading tolerance",
                        )

                # Keep the unmatched checkpoint in the expected sequence so
                # the ordinary count/order gates fail closed below.
                expanded_expected_actions.append(
                    (primitive_sequence, expected_type)
                )
                continue

            correction_count = 0
            previous_alignment_error_abs: float | None = None
            if actual_cursor > 0:
                prior = self.detour_segments[actual_cursor - 1]
                if (
                    prior.get("type") == "Spin"
                    and prior.get("primitiveSequence")
                    == primitive_sequence - 1
                ):
                    prior_goal = _mapping(
                        prior.get("goal"),
                        "detour realignment preceding Spin goal",
                    )
                    previous_alignment_error_abs = abs(
                        _finite(
                            prior_goal.get("targetYawRad"),
                            "detour realignment preceding heading error",
                        )
                    )
            while actual_cursor < len(self.detour_segments):
                candidate = self.detour_segments[actual_cursor]
                if not (
                    candidate.get("type") == "Spin"
                    and candidate.get("primitiveSequence")
                    == primitive_sequence - 1
                ):
                    break
                correction_count += 1
                realignment_spin_count += 1
                if correction_count > MAX_DRIVE_REALIGNMENT_SPINS:
                    self._fail(
                        "DETOUR_REALIGNMENT_BUDGET_EXCEEDED",
                        "DriveOnHeading used more than four corrective Spins",
                    )
                candidate_goal = _mapping(
                    candidate.get("goal"),
                    "detour corrective Spin goal",
                )
                correction_error_abs = abs(
                    _finite(
                        candidate_goal.get("targetYawRad"),
                        "detour corrective Spin heading error",
                    )
                )
                if (
                    previous_alignment_error_abs is not None
                    and correction_error_abs
                    > previous_alignment_error_abs
                    * MAX_DRIVE_REALIGNMENT_RETAINED_ERROR_RATIO
                ):
                    self._fail(
                        "DETOUR_REALIGNMENT_NOT_CONVERGING",
                        "DriveOnHeading corrective Spin did not contract heading error by at least 10%",
                    )
                previous_alignment_error_abs = correction_error_abs
                expanded_expected_actions.append(
                    (primitive_sequence - 1, "Spin")
                )
                actual_cursor += 1
            expanded_expected_actions.append((primitive_sequence, expected_type))
            if actual_cursor < len(self.detour_segments):
                candidate = self.detour_segments[actual_cursor]
                if (
                    candidate.get("type") == expected_type
                    and candidate.get("primitiveSequence") == primitive_sequence
                ):
                    actual_cursor += 1

        if actual_cursor != len(self.detour_segments):
            self._fail(
                "DETOUR_SEQUENCE_PLAN_MISMATCH",
                "executed actions contain an unbound static-plan checkpoint",
            )
        expected_actions = expanded_expected_actions
        self.metrics["detourDriveRealignmentSpinCount"] = realignment_spin_count
        if len(expected_actions) != len(self.detour_segments):
            self._fail(
                "DETOUR_SEQUENCE_PLAN_MISMATCH",
                "executed action count does not match static-plan checkpoints",
            )
        resolved_results: list[dict[str, Any]] = list(skipped_resolver_results)
        for index, (expected, actual) in enumerate(zip(expected_actions, self.detour_segments)):
            primitive_sequence, expected_type = expected
            fresh = _mapping(actual.get("freshOdometry"), "detourSegment.freshOdometry")
            fresh_pose = {
                "x": _finite(fresh.get("x"), "detour fresh x"),
                "y": _finite(fresh.get("y"), "detour fresh y"),
                "yaw": _finite(fresh.get("yaw"), "detour fresh yaw"),
            }
            try:
                resolved = _resolve_production_static_primitive_cached(
                    plan_json,
                    navfn_path_json,
                    primitive_sequence,
                    _canonical_json(fresh_pose),
                    production_parameters,
                )
            except (OSError, TypeError, ValueError) as exc:
                self._fail(
                    "DETOUR_PRODUCTION_RESOLUTION_FAILED",
                    f"production resolver rejected primitive {primitive_sequence}: {exc}",
                )
                continue
            resolved = dict(resolved)
            resolved_results.append(resolved)
            if resolved.get("disposition") != "send_action":
                self._fail(
                    "DETOUR_CHECKPOINT_NOT_EXECUTABLE",
                    f"locked acceptance primitive {primitive_sequence} resolved to {resolved.get('disposition')!r}",
                )
                continue
            resolved_action_type = {
                "nav2_msgs/action/Spin": "Spin",
                "nav2_msgs/action/DriveOnHeading": "DriveOnHeading",
            }.get(resolved.get("action_type"))
            resolved_action_goal = _mapping(
                resolved.get("action_goal"),
                f"resolved primitive {primitive_sequence} action_goal",
            )
            if resolved_action_type == "Spin":
                resolved_goal = {
                    "targetYawRad": _finite(
                        resolved_action_goal.get("target_yaw_rad"),
                        "resolved Spin target_yaw_rad",
                    ),
                    "maxWzRps": self.contract.spin_max_wz_rps,
                }
            elif resolved_action_type == "DriveOnHeading":
                target = _mapping(
                    resolved_action_goal.get("target"),
                    "resolved DriveOnHeading target",
                )
                if (
                    set(target) != {"x", "y", "z"}
                    or abs(_finite(target.get("y"), "resolved target.y")) > _EPS
                    or abs(_finite(target.get("z"), "resolved target.z")) > _EPS
                ):
                    self._fail(
                        "DETOUR_PRODUCTION_RESOLUTION_INVALID",
                        "production DriveOnHeading target must be body-frame +X with zero Y/Z",
                    )
                resolved_goal = {
                    "distanceM": _finite(
                        target.get("x"), "resolved DriveOnHeading target.x"
                    ),
                    "speedMps": _finite(
                        resolved_action_goal.get("speed_mps"),
                        "resolved DriveOnHeading speed_mps",
                    ),
                }
            else:
                self._fail(
                    "DETOUR_FORBIDDEN_ACTION",
                    f"production resolver emitted unsupported action {resolved.get('action_type')!r}",
                )
                continue
            actual_goal = _mapping(actual.get("goal"), f"detourSegments[{index}].goal")
            mismatch = (
                actual.get("primitiveSequence") != primitive_sequence
                or actual.get("type") != expected_type
                or actual.get("type") != resolved_action_type
                or actual.get("resolverResult") != resolved
                or set(actual_goal) != set(resolved_goal)
                or any(
                    abs(_finite(actual_goal[key], f"detourSegments[{index}].goal.{key}") - value) > 1e-6
                    for key, value in resolved_goal.items()
                )
            )
            if mismatch:
                self._fail(
                    "DETOUR_SEQUENCE_PLAN_MISMATCH",
                    f"executed primitive {index} is not the fresh-resolved static-plan checkpoint",
                )

        action = _mapping(self.document.get("action"), "action")
        if action.get("plannerPlanHashSha256") != computed_plan_hash:
            self._fail(
                "DETOUR_SEQUENCE_PLAN_HASH_MISMATCH",
                "executed primitive sequence is not bound to the full static plan",
            )
        self.metrics["staticPrimitivePlan"] = {
            "schemaVersion": STATIC_PRIMITIVE_PLAN_SCHEMA_VERSION,
            "planSha256": computed_plan_hash,
            "navfnPathSha256": navfn_hash,
            "segmenterSourceSha256": source_hash,
            "runtimeManifestSha256": runtime.get("runtime_manifest_sha256"),
            "motionProfileManifestSha256": runtime.get(
                "motion_profile_manifest_sha256"
            ),
            "segmentCount": len(normalized_segments),
            "primitiveCheckpointCount": len(primitives),
            "productionVerified": True,
            "resolverResultsSha256": canonical_sha256(resolved_results),
            "freshOdometryRequired": True,
            "followPathMotionAccepted": False,
        }
        return computed_plan_hash

    def _evaluate_detour_plan(self) -> None:
        if self.mode != "detour":
            self.metrics["nav2Plan"] = {"applicable": False}
            return
        assert self.contract is not None
        nav2 = _mapping(self.document.get("nav2Plan"), "nav2Plan")
        if nav2.get("source") != "observed_nav_msgs_path":
            self._fail("NAV2_PLAN_SOURCE_INVALID", "detour requires an observed nav_msgs/Path")
        if nav2.get("plannerSource") != "nav2_navfn_compute_path_to_pose":
            self._fail(
                "NAV2_PLANNER_SOURCE_INVALID",
                "detour path must be observed from Nav2 ComputePathToPose/NavFn",
            )
        if nav2.get("frameId") != "map":
            self._fail("NAV2_PLAN_FRAME_INVALID", "detour plan must use map frame")
        if nav2.get("collisionSource") != "generated_nav2_pgm":
            self._fail(
                "NAV2_COLLISION_SOURCE_INVALID",
                "detour clearance geometry must be derived from the generated Nav2 PGM",
            )
        collision_source_hash = nav2.get("collisionSourceSha256")
        if not isinstance(collision_source_hash, str) or not _SHA256.fullmatch(
            collision_source_hash
        ):
            raise TraceContractError(
                "nav2Plan.collisionSourceSha256 must be lowercase SHA-256"
            )
        raw_poses = _sequence(nav2.get("poses"), "nav2Plan.poses")
        if len(raw_poses) < 3:
            raise TraceContractError("detour Nav2 plan must contain at least three poses")
        poses: list[tuple[float, float]] = []
        plan_pose_payload: list[dict[str, float]] = []
        for index, pose in enumerate(raw_poses):
            point = _mapping(pose, f"nav2Plan.poses[{index}]")
            if set(point) != {"x", "y", "yaw"}:
                raise TraceContractError(
                    f"nav2Plan.poses[{index}] must contain exactly x/y/yaw"
                )
            x = _finite(point.get("x"), f"nav2Plan.poses[{index}].x")
            y = _finite(point.get("y"), f"nav2Plan.poses[{index}].y")
            yaw = _finite(point.get("yaw"), f"nav2Plan.poses[{index}].yaw")
            poses.append((x, y))
            plan_pose_payload.append({"x": x, "y": y, "yaw": yaw})
        plan_payload = {"frame_id": "map", "poses": plan_pose_payload}
        plan_hash = canonical_sha256(plan_payload)
        claimed_plan_hash = nav2.get("planHashSha256")
        if not isinstance(claimed_plan_hash, str) or not _SHA256.fullmatch(claimed_plan_hash):
            raise TraceContractError("nav2Plan.planHashSha256 must be lowercase SHA-256")
        if claimed_plan_hash != plan_hash:
            self._fail("NAV2_PLAN_HASH_MISMATCH", "Nav2 plan hash does not match canonical poses")
        action = _mapping(self.document.get("action"), "action")
        obstacles_raw = _sequence(nav2.get("obstacles"), "nav2Plan.obstacles")
        obstacles = [dict(_mapping(item, f"nav2Plan.obstacles[{index}]")) for index, item in enumerate(obstacles_raw)]
        obstacle_hash = canonical_sha256(obstacles)
        claimed_obstacle_hash = nav2.get("obstacleSetHashSha256")
        if not isinstance(claimed_obstacle_hash, str) or not _SHA256.fullmatch(claimed_obstacle_hash):
            raise TraceContractError("nav2Plan.obstacleSetHashSha256 must be lowercase SHA-256")
        if claimed_obstacle_hash != obstacle_hash:
            self._fail("OBSTACLE_SET_HASH_MISMATCH", "obstacle geometry hash does not match")
        static_plan_hash = self._evaluate_static_primitive_plan(
            plan_payload,
            poses,
            collision_source_hash,
        )
        for field, expected in (
            ("robotRadiusM", self.contract.robot_radius_m),
            ("footprintPaddingM", self.contract.footprint_padding_m),
            ("mapResolutionM", self.contract.map_resolution_m),
        ):
            actual = _finite(nav2.get(field), f"nav2Plan.{field}")
            if abs(actual - expected) > _EPS:
                self._fail("NAV2_CLEARANCE_CONTRACT_MISMATCH", f"{field} is {actual:.4f}, formal value is {expected:.4f}")
        required = self.contract.required_obstacle_boundary_clearance_m
        realized_required = (
            self.contract.realized_obstacle_boundary_clearance_m
        )
        plan_clearance = _path_clearance(poses, obstacles)
        realized_path = [tuple(sample["root"][:2]) for sample in self._parsed_samples]
        realized_clearance = _path_clearance(realized_path, obstacles)
        start, goal = realized_path[0], realized_path[-1]
        direct_clearance = _path_clearance((start, goal), obstacles)
        if plan_clearance + _EPS < required:
            self._fail("NAV2_PLAN_CLEARANCE_FAILED", f"plan boundary clearance is {plan_clearance:.4f}m; {required:.4f}m required")
        if realized_clearance + _EPS < realized_required:
            self._fail(
                "REALIZED_CLEARANCE_FAILED",
                "realized boundary clearance is "
                f"{realized_clearance:.4f}m; {realized_required:.4f}m "
                "physical footprint clearance required",
            )
        if direct_clearance + _EPS >= required:
            self._fail("DETOUR_NOT_PROVEN", "the direct start/goal segment is not blocked by the static obstacle set")
        goal_obj = _mapping(action.get("goal"), "action.goal")
        goal_xy = (_finite(goal_obj.get("x"), "action.goal.x"), _finite(goal_obj.get("y"), "action.goal.y"))
        goal_yaw = _finite(goal_obj.get("yaw"), "action.goal.yaw")
        if math.dist(goal, goal_xy) > self.contract.final_position_tolerance_m + _EPS:
            self._fail("FINAL_POSITION_MISSED", f"detour goal error is {math.dist(goal, goal_xy):.4f}m")
        final_yaw = float(self._parsed_samples[-1]["yaw"])
        if abs(_angle_delta(final_yaw, goal_yaw)) > self.contract.final_yaw_tolerance_rad + _EPS:
            self._fail("FINAL_YAW_MISSED", "detour final yaw missed the goal")
        if math.dist(poses[0], start) > self.contract.final_position_tolerance_m + _EPS or math.dist(poses[-1], goal_xy) > self.contract.final_position_tolerance_m + _EPS:
            self._fail("NAV2_PLAN_ENDPOINT_MISMATCH", "Nav2 plan endpoints do not match recorded start/goal")
        self.metrics["nav2Plan"] = {
            "applicable": True,
            "planSha256": plan_hash,
            "staticPrimitivePlanSha256": static_plan_hash,
            "obstacleSetSha256": obstacle_hash,
            "collisionSource": "generated_nav2_pgm",
            "collisionSourceSha256": collision_source_hash,
            "planLengthM": _path_length(poses),
            "realizedPathLengthM": _path_length(realized_path),
            "minimumPlanBoundaryClearanceM": plan_clearance,
            "minimumRealizedBoundaryClearanceM": realized_clearance,
            "directBoundaryClearanceM": direct_clearance,
            "requiredBoundaryClearanceM": required,
            "requiredRealizedBoundaryClearanceM": realized_required,
        }

    def _evaluate_terminal_settle(self) -> None:
        assert self.contract is not None
        samples = self._parsed_samples

        def settled(sample: Mapping[str, Any]) -> bool:
            return (
                not sample["intent"]
                and sample["balancedPolicyIdle"]
                and float(sample["linearSpeed"]) <= self.contract.terminal_linear_speed_mps + _EPS
                and float(sample["angularSpeed"]) <= self.contract.terminal_angular_speed_rps + _EPS
                and max(abs(float(sample[name])) for name in ("vx", "vy", "wz")) <= self.contract.command_zero_epsilon + _EPS
                and not any(sample["actions"].values())
            )

        settled_tail: list[Mapping[str, Any]] = []
        for sample in reversed(samples):
            if not settled(sample):
                break
            settled_tail.append(sample)
        settled_tail.reverse()
        duration = 0.0 if not settled_tail else float(settled_tail[-1]["t"]) - float(settled_tail[0]["t"])
        max_gap = 0.0
        if len(settled_tail) >= 2:
            max_gap = max(float(after["t"]) - float(before["t"]) for before, after in zip(settled_tail, settled_tail[1:]))
        self.metrics["terminalSettle"] = {
            "durationS": duration,
            "sampleCount": len(settled_tail),
            "maxSampleGapS": max_gap,
        }
        if duration + _EPS < self.contract.terminal_settle_s:
            self._fail("TERMINAL_SETTLE_TOO_SHORT", f"terminal settled interval is {duration:.3f}s; 1.000s required")
        if max_gap > self.contract.max_terminal_sample_gap_s + _EPS:
            self._fail("TERMINAL_SETTLE_SAMPLE_GAP", f"terminal evidence gap reached {max_gap:.3f}s")


def evaluate_trace(document: Mapping[str, Any]) -> dict[str, Any]:
    return StrictMotionEvaluator(document).evaluate()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an explicit offline locomotion trace. No network or robot "
            "control operations are implemented by this program."
        )
    )
    parser.add_argument("--input", type=Path, help="explicit JSON trace path")
    parser.add_argument("--output", type=Path, help="optional explicit JSON result path")
    parser.add_argument("--overwrite", action="store_true", help="replace the explicit result path")
    parser.add_argument("--describe-contract", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.describe_contract:
        print(json.dumps({mode: asdict(contract) for mode, contract in FORMAL_CONTRACTS.items()}, indent=2, sort_keys=True))
        return 0
    if args.input is None:
        raise SystemExit("--input is required unless --describe-contract is used")
    input_path = args.input.expanduser()
    if not input_path.is_file():
        raise SystemExit(f"input is not a regular file: {input_path}")
    document = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise SystemExit("input JSON must be an object")
    result = evaluate_trace(document)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        output_path = args.output.expanduser()
        if output_path.is_symlink():
            raise SystemExit("refusing to write through a symbolic link")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            stream.write(rendered)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
