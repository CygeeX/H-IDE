#!/usr/bin/env python3
"""Read-only MuJoCo/SONIC motion acceptance monitor.

This program deliberately has no control client.  Its only network operation is
HTTP GET against the Plant ``/state`` and SONIC ``/telemetry`` resources.  It
records every paired observation to a caller-selected JSONL file and evaluates
the evidence without attempting to stop, reset, rearm, or move the robot.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid


SCHEMA_VERSION = "h-ide.motion-acceptance-monitor/1"
_SAFE_HOLD_MARKERS = (
    "requires explicit control.acquire rearm",
    "safe_hold",
    "safe hold",
    "plant safety_stop is latched",
)
_RESET_FAULT_MARKERS = (
    "teleport",
    "rolled back",
    "rollback",
    "qacc",
    "non-finite",
    "numerical instability",
    "implicit reset",
    "physics time is invalid",
)


@dataclass(frozen=True, slots=True)
class AcceptanceLimits:
    phase: str
    min_root_z_m: float = 0.55
    max_root_z_m: float = 1.15
    max_roll_pitch_rad: float = 0.35
    max_position_jump_m: float = 0.30
    max_yaw_jump_rad: float = 0.50
    max_lateral_m: float = 0.25
    max_heading_error_rad: float = 0.35
    max_directional_travel_m: float = 0.80
    max_reverse_m: float = 0.05
    progress_window_s: float = 15.0
    min_progress_m: float = 0.10
    expected_forward_m: float = 0.50
    final_position_tolerance_m: float = 0.20
    expected_turn_rad: float = math.pi / 2.0
    final_yaw_tolerance_rad: float = 0.35
    min_turn_progress_rad: float = 0.35
    max_turn_overshoot_rad: float = 0.35
    max_turn_translation_m: float = 0.25
    max_turn_reverse_rad: float = 0.10
    max_idle_drift_m: float = 0.02
    max_idle_yaw_drift_rad: float = 0.035
    motion_position_epsilon_m: float = 0.01
    motion_yaw_epsilon_rad: float = 0.02
    command_epsilon: float = 0.01
    max_arm_constrained_joints: int = 0
    max_arm_step_rad: float = 0.15
    max_arm_raw_deviation_rad: float = 2.75
    max_arm_tracking_error_rad: float = 0.90
    max_arm_measured_deviation_rad: float = 2.75
    max_arm_velocity_rps: float = 7.0
    max_arm_estimated_torque_nm: float = 35.0
    require_settled: bool = True

    def __post_init__(self) -> None:
        if self.phase not in {"idle", "straight", "turn"}:
            raise ValueError("phase must be idle, straight, or turn")
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name
            not in {
                "phase",
                "expected_forward_m",
                "expected_turn_rad",
                "max_arm_constrained_joints",
                "require_settled",
            }
        }
        if any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in positive.values()
        ):
            raise ValueError("all acceptance thresholds must be finite and positive")
        if self.min_root_z_m >= self.max_root_z_m:
            raise ValueError("minimum root z must be below maximum root z")
        if not math.isfinite(self.expected_forward_m):
            raise ValueError("expected forward distance must be finite")
        if not math.isfinite(self.expected_turn_rad) or abs(self.expected_turn_rad) < 1e-6:
            raise ValueError("expected turn must be finite and non-zero")
        if self.max_arm_constrained_joints < 0:
            raise ValueError("maximum constrained arm joints cannot be negative")


@dataclass(frozen=True, slots=True)
class Failure:
    code: str
    detail: str
    first_observed_s: float


class ReadonlyHttpSampler:
    """Fetch the two fixed read-only resources using GET and no request body."""

    def __init__(
        self,
        plant_url: str,
        sonic_url: str,
        *,
        timeout_s: float = 1.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.plant_url = self._validate_url(plant_url, "/state", "Plant")
        self.sonic_url = self._validate_url(sonic_url, "/telemetry", "SONIC")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("HTTP timeout must be finite and positive")
        self.timeout_s = float(timeout_s)
        self._opener = opener

    @staticmethod
    def _validate_url(value: str, required_path: str, label: str) -> str:
        parsed = urlsplit(str(value))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{label} URL must be an absolute HTTP(S) URL")
        if parsed.path != required_path or parsed.query or parsed.fragment:
            raise ValueError(f"{label} URL must end exactly in {required_path}")
        return parsed.geturl()

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(
            url,
            method="GET",
            headers={"Accept": "application/json", "Cache-Control": "no-store"},
        )
        with self._opener(request, timeout=self.timeout_s) as response:
            status_value = getattr(response, "status", None)
            if status_value is None:
                status_value = response.getcode()
            status = int(status_value)
            if status != 200:
                raise RuntimeError(f"GET {url} returned HTTP {status}")
            body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise RuntimeError(f"GET {url} returned an oversized response")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise RuntimeError(f"GET {url} did not return a JSON object")
        return payload

    def sample(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._get_json(self.plant_url), self._get_json(self.sonic_url)


class JsonlEvidenceWriter:
    def __init__(self, path: str | Path, *, overwrite: bool = False) -> None:
        self.path = Path(path).expanduser()
        if self.path.is_symlink():
            raise ValueError("refusing to write evidence through a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open(
            "w" if overwrite else "x",
            encoding="utf-8",
        )

    def write(self, value: Mapping[str, Any]) -> None:
        serialized = json.dumps(
            _json_safe(value),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._stream.write(serialized + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> JsonlEvidenceWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _angular_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def _quaternion_to_rpy(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> tuple[float, float, float]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("root quaternion is zero or non-finite")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch_term = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, pitch_term)))
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return roll, pitch, yaw


def _nonfinite_paths(value: Any, path: str) -> list[str]:
    result: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        result.append(path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(_nonfinite_paths(item, f"{path}[{index}]"))
    return result


def _finite_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool):
        raise ValueError(f"{key} is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} is non-finite")
    return result


def _integer(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} is not an integer")
    return value


class MotionAcceptanceEvaluator:
    def __init__(self, limits: AcceptanceLimits) -> None:
        self.limits = limits
        self.failures: dict[str, Failure] = {}
        self.samples = 0
        self.first_observed_s: float | None = None
        self.last_observed_s: float | None = None
        self.baseline_pose: tuple[float, float, float, float] | None = None
        self.latest_pose: tuple[float, float, float, float] | None = None
        self.previous_pose: tuple[float, float, float, float] | None = None
        self.previous_yaw: float | None = None
        self.unwrapped_yaw = 0.0
        self.intent_started_s: float | None = None
        self.physical_motion_seen = False
        self.progress_reached_s: float | None = None
        self.settled_after_motion = False
        self.previous_counters: dict[str, float] = {}
        self.source_identity: tuple[Any, ...] | None = None
        self.maxima: dict[str, float] = {
            "abs_roll_rad": 0.0,
            "abs_pitch_rad": 0.0,
            "position_jump_m": 0.0,
            "yaw_jump_rad": 0.0,
            "abs_lateral_m": 0.0,
            "directional_travel_m": 0.0,
            "turn_translation_m": 0.0,
            "directional_turn_rad": 0.0,
            "arm_constrained_joints": 0.0,
            "arm_step_rad": 0.0,
            "arm_raw_deviation_rad": 0.0,
            "arm_tracking_error_rad": 0.0,
            "arm_measured_deviation_rad": 0.0,
            "arm_velocity_rps": 0.0,
            "arm_estimated_torque_nm": 0.0,
        }

    def _fail(self, code: str, detail: str, observed_s: float) -> None:
        if code not in self.failures:
            origin = self.first_observed_s if self.first_observed_s is not None else observed_s
            self.failures[code] = Failure(
                code=code,
                detail=detail,
                first_observed_s=max(0.0, observed_s - origin),
            )

    def observe_error(self, detail: str, *, observed_s: float) -> dict[str, Any]:
        if self.first_observed_s is None:
            self.first_observed_s = observed_s
        self.last_observed_s = observed_s
        self.samples += 1
        self._fail("SAMPLE_UNAVAILABLE", str(detail), observed_s)
        return {"sample": self.samples, "error": str(detail)}

    def observe(
        self,
        plant: Mapping[str, Any],
        sonic: Mapping[str, Any],
        *,
        observed_s: float,
    ) -> dict[str, Any]:
        if self.first_observed_s is None:
            self.first_observed_s = observed_s
        if self.last_observed_s is not None and observed_s <= self.last_observed_s:
            self._fail(
                "MONITOR_TIME_ROLLBACK",
                "monitor observation time did not advance",
                observed_s,
            )
        self.last_observed_s = observed_s
        self.samples += 1

        nonfinite = _nonfinite_paths(plant, "plant") + _nonfinite_paths(sonic, "sonic")
        if nonfinite:
            self._fail(
                "NONFINITE_TELEMETRY",
                "non-finite values at " + ", ".join(nonfinite[:8]),
                observed_s,
            )

        derived: dict[str, Any] = {"sample": self.samples}
        try:
            pose = plant["pose"]
            support = plant["support"]
            if not isinstance(pose, Mapping) or not isinstance(support, Mapping):
                raise ValueError("Plant pose/support are not objects")
            x, y, z = (_finite_number(pose, key) for key in ("x", "y", "z"))
            qw, qx, qy, qz = (
                _finite_number(pose, key) for key in ("qw", "qx", "qy", "qz")
            )
            roll, pitch, yaw = _quaternion_to_rpy(qw, qx, qy, qz)
            sequence = _integer(plant, "sequence")
            captured_at_ns = _integer(plant, "capturedAtUnixNs")
            updated_at = _finite_number(sonic, "updated_at")
            plant_tick = _integer(sonic, "plant_tick")
            control_tick = _integer(sonic, "control_tick_ms")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self._fail("SCHEMA_INVALID", f"required telemetry is invalid: {exc}", observed_s)
            return derived

        required_support_fields = {
            "mode",
            "fault",
            "updatedAt",
            "safetyStopLatched",
            "safetyStopEpoch",
            "lastRearmGeneration",
        }
        required_sonic_fields = {
            "status",
            "motion_ready",
            "nav2_state",
            "sonic_mode",
            "navigation_task_active",
            "plant_support_safety_latched",
            "fault_trace_path",
            "fault_trace_sample_count",
            "last_error",
        }
        missing = sorted(required_support_fields.difference(support)) + sorted(
            f"sonic.{name}" for name in required_sonic_fields.difference(sonic)
        )
        if missing:
            self._fail(
                "SCHEMA_INVALID",
                "required telemetry fields are missing: " + ", ".join(missing),
                observed_s,
            )

        derived.update(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            sequence=sequence,
            captured_at_unix_ns=captured_at_ns,
        )

        if captured_at_ns <= 0 or sequence <= 0:
            self._fail(
                "SCHEMA_INVALID",
                "running Plant sequence and capture time must be positive",
                observed_s,
            )
        if plant.get("schemaVersion") != "h-ide.mujoco-capture/1":
            self._fail(
                "SCHEMA_INVALID",
                f"unsupported Plant schemaVersion {plant.get('schemaVersion')!r}",
                observed_s,
            )
        if sonic.get("backend") not in {None, "mujoco_sonic"}:
            self._fail(
                "SCHEMA_INVALID",
                f"unexpected SONIC backend {sonic.get('backend')!r}",
                observed_s,
            )
        identity = (
            plant.get("robotId"),
            plant.get("sourceId"),
            plant.get("localizationEpochId"),
            sonic.get("robot_id"),
        )
        if self.source_identity is None:
            self.source_identity = identity
        elif identity != self.source_identity:
            self._fail(
                "SOURCE_IDENTITY_CHANGED",
                f"Plant/SONIC source identity changed from {self.source_identity!r} "
                f"to {identity!r}",
                observed_s,
            )

        if plant.get("status") != "running":
            self._fail(
                "PLANT_NOT_RUNNING",
                f"Plant status is {plant.get('status')!r}, expected 'running'",
                observed_s,
            )
        if sonic.get("status") != "running":
            self._fail(
                "SONIC_NOT_RUNNING",
                f"SONIC status is {sonic.get('status')!r}, expected 'running'",
                observed_s,
            )
        if sonic.get("motion_ready") is not True:
            self._fail("SONIC_NOT_MOTION_READY", "SONIC motion_ready is not true", observed_s)

        support_mode = str(support.get("mode", ""))
        support_latched = support.get("safetyStopLatched")
        if support_mode not in {"idle", "locomotion", "safety_stop"}:
            self._fail(
                "PLANT_SUPPORT_INVALID",
                f"running Plant support mode is {support_mode!r}",
                observed_s,
            )
        if (support_mode == "safety_stop") is not (support_latched is True):
            self._fail(
                "PLANT_SUPPORT_INVALID",
                "Plant support mode and safetyStopLatched disagree",
                observed_s,
            )

        fault = support.get("fault")
        if fault not in {None, ""}:
            self._fail("PLANT_FAULT", f"Plant fault is latched: {fault}", observed_s)
            normalized_fault = str(fault).lower()
            if any(marker in normalized_fault for marker in _RESET_FAULT_MARKERS):
                self._fail(
                    "RESET_OR_NUMERICAL_FAULT_CLUE",
                    f"Plant fault contains a reset/numerical-fault clue: {fault}",
                    observed_s,
                )
        if sonic.get("fault_trace_path") not in {None, ""} or sonic.get(
            "fault_trace_sample_count", 0
        ) not in {None, 0}:
            self._fail(
                "SONIC_FAULT_TRACE",
                "SONIC reported a persisted terminal fault trace",
                observed_s,
            )

        if not self.limits.min_root_z_m <= z <= self.limits.max_root_z_m:
            self._fail(
                "ROOT_HEIGHT_OUT_OF_RANGE",
                f"root z {z:.4f}m is outside [{self.limits.min_root_z_m:.3f}, "
                f"{self.limits.max_root_z_m:.3f}]m",
                observed_s,
            )
        self.maxima["abs_roll_rad"] = max(self.maxima["abs_roll_rad"], abs(roll))
        self.maxima["abs_pitch_rad"] = max(self.maxima["abs_pitch_rad"], abs(pitch))
        if abs(roll) > self.limits.max_roll_pitch_rad:
            self._fail("ROLL_LIMIT_EXCEEDED", f"absolute roll reached {abs(roll):.4f}rad", observed_s)
        if abs(pitch) > self.limits.max_roll_pitch_rad:
            self._fail("PITCH_LIMIT_EXCEEDED", f"absolute pitch reached {abs(pitch):.4f}rad", observed_s)

        counters = {
            "plant.sequence": float(sequence),
            "plant.capturedAtUnixNs": float(captured_at_ns),
            "plant.support.updatedAt": _optional_finite(support.get("updatedAt")),
            "plant.support.safetyStopEpoch": _optional_finite(support.get("safetyStopEpoch")),
            "plant.support.lastRearmGeneration": _optional_finite(
                support.get("lastRearmGeneration")
            ),
            "sonic.updated_at": updated_at,
            "sonic.plant_tick": float(plant_tick),
            "sonic.control_tick_ms": float(control_tick),
        }
        for name, current in counters.items():
            previous = self.previous_counters.get(name)
            if math.isfinite(current):
                if previous is not None and current < previous:
                    self._fail(
                        "SOURCE_TIME_OR_GENERATION_ROLLBACK",
                        f"{name} rolled back from {previous:g} to {current:g}",
                        observed_s,
                    )
                self.previous_counters[name] = current

        position = (x, y, z, yaw)
        if self.previous_pose is not None:
            jump = math.dist(position[:3], self.previous_pose[:3])
            yaw_jump = abs(_angular_delta(yaw, self.previous_yaw if self.previous_yaw is not None else yaw))
            self.maxima["position_jump_m"] = max(self.maxima["position_jump_m"], jump)
            self.maxima["yaw_jump_rad"] = max(self.maxima["yaw_jump_rad"], yaw_jump)
            if jump > self.limits.max_position_jump_m:
                self._fail(
                    "POSITION_JUMP_OR_TELEPORT",
                    f"root position jumped {jump:.4f}m between samples",
                    observed_s,
                )
            if yaw_jump > self.limits.max_yaw_jump_rad:
                self._fail(
                    "YAW_JUMP_OR_TELEPORT",
                    f"root yaw jumped {yaw_jump:.4f}rad between samples",
                    observed_s,
                )
            self.unwrapped_yaw += _angular_delta(yaw, self.previous_yaw if self.previous_yaw is not None else yaw)
        self.previous_pose = position
        self.previous_yaw = yaw
        self.latest_pose = position
        if self.baseline_pose is None:
            self.baseline_pose = position

        baseline_x, baseline_y, _baseline_z, baseline_yaw = self.baseline_pose
        dx, dy = x - baseline_x, y - baseline_y
        forward = dx * math.cos(baseline_yaw) + dy * math.sin(baseline_yaw)
        lateral = -dx * math.sin(baseline_yaw) + dy * math.cos(baseline_yaw)
        translation = math.hypot(dx, dy)
        heading_delta = self.unwrapped_yaw
        derived.update(
            forward_m=forward,
            lateral_m=lateral,
            translation_m=translation,
            unwrapped_yaw_rad=heading_delta,
        )

        command_values: list[float] = []
        for key in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz", "sonic_speed_mps"):
            try:
                command_values.append(abs(_finite_number(sonic, key)))
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self._fail("SCHEMA_INVALID", f"required SONIC field is invalid: {exc}", observed_s)
                command_values.append(0.0)
        nav2_state = str(sonic.get("nav2_state", ""))
        sonic_mode = str(sonic.get("sonic_mode", ""))
        active_state = nav2_state not in {
            "",
            "idle",
            "safety_stop",
            "stopped",
            "standby",
            "fault_latched",
            "startup_hold",
        }
        intent_active = bool(sonic.get("navigation_task_active")) or active_state or any(
            value > self.limits.command_epsilon for value in command_values
        ) or sonic_mode != "IDLE"
        if intent_active and self.intent_started_s is None:
            self.intent_started_s = observed_s

        if self.limits.phase != "idle" and (
            translation > self.limits.motion_position_epsilon_m
            or abs(heading_delta) > self.limits.motion_yaw_epsilon_rad
        ):
            self.physical_motion_seen = True

        safe_hold = (
            support_mode == "safety_stop"
            and nav2_state == "safety_stop"
            and sonic.get("plant_support_safety_latched") is True
        )
        last_error = sonic.get("last_error")
        if last_error not in {None, ""}:
            normalized = str(last_error).lower()
            pre_motion_safe_hold = (
                not self.physical_motion_seen
                and safe_hold
                and any(marker in normalized for marker in _SAFE_HOLD_MARKERS)
            )
            if not pre_motion_safe_hold:
                self._fail("SONIC_LAST_ERROR", f"SONIC last_error: {last_error}", observed_s)
            if any(marker in normalized for marker in _RESET_FAULT_MARKERS):
                self._fail(
                    "RESET_OR_NUMERICAL_FAULT_CLUE",
                    f"SONIC error contains a reset/numerical-fault clue: {last_error}",
                    observed_s,
                )

        if self.physical_motion_seen and safe_hold:
            self._fail(
                "SAFE_HOLD_AFTER_MOTION",
                "Plant/SONIC returned to SAFE_HOLD after physical motion began",
                observed_s,
            )
        if self.physical_motion_seen and intent_active and support_mode != "locomotion":
            self._fail(
                "PLANT_SUPPORT_INVALID",
                f"active physical motion observed while Plant support is {support_mode!r}",
                observed_s,
            )

        self._observe_arm_metrics(sonic, observed_s, after_intent=self.intent_started_s is not None)

        root_velocity = plant.get("rootVelocity")
        linear_speed = angular_speed = None
        if isinstance(root_velocity, Mapping):
            try:
                linear_speed = _finite_number(root_velocity, "linearSpeedMps")
                angular_speed = _finite_number(root_velocity, "angularSpeedRps")
                if linear_speed < 0 or angular_speed < 0:
                    raise ValueError("root speed magnitudes cannot be negative")
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                self._fail("SCHEMA_INVALID", f"rootVelocity is invalid: {exc}", observed_s)
        else:
            self._fail("SCHEMA_INVALID", "Plant rootVelocity is missing", observed_s)
        derived.update(
            intent_active=intent_active,
            physical_motion_seen=self.physical_motion_seen,
            linear_speed_mps=linear_speed,
            angular_speed_rps=angular_speed,
            safe_hold=safe_hold,
        )

        if self.limits.phase == "idle":
            self._evaluate_idle(translation, heading_delta, intent_active, observed_s)
        elif self.limits.phase == "straight":
            self._evaluate_straight(forward, lateral, heading_delta, observed_s)
        else:
            self._evaluate_turn(translation, heading_delta, observed_s)

        if (
            self.physical_motion_seen
            and not intent_active
            and linear_speed is not None
            and angular_speed is not None
            and linear_speed <= self.limits.command_epsilon
            and angular_speed <= self.limits.command_epsilon
            and support_mode == "idle"
        ):
            self.settled_after_motion = True
        elif self.physical_motion_seen and intent_active:
            self.settled_after_motion = False
        return derived

    def _observe_arm_metrics(
        self,
        sonic: Mapping[str, Any],
        observed_s: float,
        *,
        after_intent: bool,
    ) -> None:
        metrics = (
            (
                "navigation_arm_constrained_joints",
                "navigation_task_peak_arm_constrained_joints",
                "arm_constrained_joints",
                float(self.limits.max_arm_constrained_joints),
            ),
            (
                "navigation_arm_max_step_rad",
                None,
                "arm_step_rad",
                self.limits.max_arm_step_rad,
            ),
            (
                "navigation_arm_raw_max_deviation_rad",
                "navigation_task_peak_arm_raw_deviation_rad",
                "arm_raw_deviation_rad",
                self.limits.max_arm_raw_deviation_rad,
            ),
            (
                "navigation_arm_raw_max_error_from_measured_rad",
                "navigation_task_peak_arm_tracking_error_rad",
                "arm_tracking_error_rad",
                self.limits.max_arm_tracking_error_rad,
            ),
            (
                "navigation_arm_measured_max_deviation_rad",
                "navigation_task_peak_arm_measured_deviation_rad",
                "arm_measured_deviation_rad",
                self.limits.max_arm_measured_deviation_rad,
            ),
            (
                "navigation_arm_max_velocity_rps",
                "navigation_task_peak_arm_velocity_rps",
                "arm_velocity_rps",
                self.limits.max_arm_velocity_rps,
            ),
            (
                "navigation_arm_max_estimated_torque_nm",
                "navigation_task_peak_arm_estimated_torque_nm",
                "arm_estimated_torque_nm",
                self.limits.max_arm_estimated_torque_nm,
            ),
        )
        for live_key, peak_key, summary_key, limit in metrics:
            keys = [live_key]
            if after_intent and peak_key is not None:
                keys.append(peak_key)
            for key in keys:
                try:
                    value = _finite_number(sonic, key)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    self._fail("SCHEMA_INVALID", f"arm telemetry is invalid: {exc}", observed_s)
                    continue
                if value < 0:
                    self._fail(
                        "SCHEMA_INVALID",
                        f"{key} cannot be negative",
                        observed_s,
                    )
                    continue
                self.maxima[summary_key] = max(self.maxima[summary_key], value)
                if value > limit:
                    self._fail(
                        "ARM_TELEMETRY_SPIKE",
                        f"{key} reached {value:.4f}, above {limit:.4f}",
                        observed_s,
                    )

    def _evaluate_idle(
        self,
        translation: float,
        heading_delta: float,
        intent_active: bool,
        observed_s: float,
    ) -> None:
        if intent_active:
            self._fail("UNEXPECTED_MOTION_INTENT", "motion intent appeared during idle phase", observed_s)
        if translation > self.limits.max_idle_drift_m:
            self._fail(
                "IDLE_POSITION_DRIFT",
                f"idle root drift reached {translation:.4f}m",
                observed_s,
            )
        if abs(heading_delta) > self.limits.max_idle_yaw_drift_rad:
            self._fail(
                "IDLE_YAW_DRIFT",
                f"idle yaw drift reached {abs(heading_delta):.4f}rad",
                observed_s,
            )

    def _evaluate_straight(
        self,
        forward: float,
        lateral: float,
        heading_delta: float,
        observed_s: float,
    ) -> None:
        direction = 1.0 if self.limits.expected_forward_m >= 0 else -1.0
        directional = direction * forward
        self.maxima["abs_lateral_m"] = max(self.maxima["abs_lateral_m"], abs(lateral))
        self.maxima["directional_travel_m"] = max(
            self.maxima["directional_travel_m"], directional
        )
        if abs(lateral) > self.limits.max_lateral_m:
            self._fail("LATERAL_LIMIT_EXCEEDED", f"absolute lateral travel reached {abs(lateral):.4f}m", observed_s)
        if abs(heading_delta) > self.limits.max_heading_error_rad:
            self._fail("HEADING_LIMIT_EXCEEDED", f"straight-line heading error reached {abs(heading_delta):.4f}rad", observed_s)
        if directional > self.limits.max_directional_travel_m:
            self._fail("MAX_FORWARD_EXCEEDED", f"directional travel reached {directional:.4f}m", observed_s)
        if directional < -self.limits.max_reverse_m:
            self._fail("REVERSE_MOTION_EXCEEDED", f"opposite-direction travel reached {-directional:.4f}m", observed_s)
        if directional >= self.limits.min_progress_m and self.progress_reached_s is None:
            self.progress_reached_s = observed_s
        self._check_progress_deadline(observed_s)

    def _evaluate_turn(
        self,
        translation: float,
        heading_delta: float,
        observed_s: float,
    ) -> None:
        direction = 1.0 if self.limits.expected_turn_rad > 0 else -1.0
        directional = direction * heading_delta
        self.maxima["turn_translation_m"] = max(self.maxima["turn_translation_m"], translation)
        self.maxima["directional_turn_rad"] = max(
            self.maxima["directional_turn_rad"], directional
        )
        if translation > self.limits.max_turn_translation_m:
            self._fail("TURN_TRANSLATION_EXCEEDED", f"turn translated {translation:.4f}m", observed_s)
        if directional < -self.limits.max_turn_reverse_rad:
            self._fail("TURN_REVERSED", f"turn moved {abs(directional):.4f}rad in the wrong direction", observed_s)
        max_turn = abs(self.limits.expected_turn_rad) + self.limits.max_turn_overshoot_rad
        if directional > max_turn:
            self._fail("TURN_OVERSHOOT", f"directional turn reached {directional:.4f}rad", observed_s)
        if directional >= self.limits.min_turn_progress_rad and self.progress_reached_s is None:
            self.progress_reached_s = observed_s
        self._check_progress_deadline(observed_s)

    def _check_progress_deadline(self, observed_s: float) -> None:
        if (
            self.intent_started_s is not None
            and self.progress_reached_s is None
            and observed_s - self.intent_started_s >= self.limits.progress_window_s
        ):
            unit = "m" if self.limits.phase == "straight" else "rad"
            threshold = (
                self.limits.min_progress_m
                if self.limits.phase == "straight"
                else self.limits.min_turn_progress_rad
            )
            self._fail(
                "INSUFFICIENT_PROGRESS_15S",
                f"progress did not reach {threshold:.3f}{unit} within "
                f"{self.limits.progress_window_s:.1f}s of motion intent",
                observed_s,
            )

    def finalize(self) -> dict[str, Any]:
        final_time = self.last_observed_s if self.last_observed_s is not None else 0.0
        if self.samples < 2:
            self._fail("INSUFFICIENT_SAMPLES", "at least two paired samples are required", final_time)
        if self.limits.phase != "idle":
            if self.intent_started_s is None:
                self._fail("NO_MOTION_INTENT", "no Nav2/SONIC motion intent was observed", final_time)
            if not self.physical_motion_seen:
                self._fail("NO_PHYSICAL_MOTION", "no root translation or yaw motion was observed", final_time)
            if self.progress_reached_s is None:
                self._fail("MINIMUM_PROGRESS_NOT_REACHED", "required motion progress was not observed", final_time)
            if self.limits.require_settled and not self.settled_after_motion:
                self._fail(
                    "NOT_SETTLED_AT_END",
                    "motion did not finish in a stationary Plant idle state",
                    final_time,
                )
        if self.latest_pose is not None and self.baseline_pose is not None:
            x, y, _z, _yaw = self.latest_pose
            x0, y0, _z0, yaw0 = self.baseline_pose
            forward = (x - x0) * math.cos(yaw0) + (y - y0) * math.sin(yaw0)
            if self.limits.phase == "straight" and abs(
                forward - self.limits.expected_forward_m
            ) > self.limits.final_position_tolerance_m:
                self._fail(
                    "FINAL_POSITION_MISSED",
                    f"final forward travel {forward:.4f}m is not within "
                    f"{self.limits.final_position_tolerance_m:.3f}m of "
                    f"{self.limits.expected_forward_m:.3f}m",
                    final_time,
                )
            if self.limits.phase == "turn" and abs(
                self.unwrapped_yaw - self.limits.expected_turn_rad
            ) > self.limits.final_yaw_tolerance_rad:
                self._fail(
                    "FINAL_YAW_MISSED",
                    f"final yaw delta {self.unwrapped_yaw:.4f}rad is not within "
                    f"{self.limits.final_yaw_tolerance_rad:.3f}rad of "
                    f"{self.limits.expected_turn_rad:.3f}rad",
                    final_time,
                )
        elapsed = (
            0.0
            if self.first_observed_s is None or self.last_observed_s is None
            else max(0.0, self.last_observed_s - self.first_observed_s)
        )
        failures = [asdict(item) for item in self.failures.values()]
        return {
            "pass": not failures,
            "phase": self.limits.phase,
            "sampleCount": self.samples,
            "observedDurationS": elapsed,
            "physicalMotionSeen": self.physical_motion_seen,
            "intentStartedS": (
                None
                if self.intent_started_s is None or self.first_observed_s is None
                else self.intent_started_s - self.first_observed_s
            ),
            "progressReachedS": (
                None
                if self.progress_reached_s is None or self.first_observed_s is None
                else self.progress_reached_s - self.first_observed_s
            ),
            "settledAfterMotion": self.settled_after_motion,
            "finalUnwrappedYawRad": self.unwrapped_yaw,
            "maxima": dict(self.maxima),
            "failures": failures,
        }


def _optional_finite(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only MuJoCo/SONIC acceptance monitor; performs HTTP GET only "
            "and never sends a control, stop, rearm, Nav2, cmd_vel, or reset request."
        )
    )
    parser.add_argument("--phase", required=True, choices=("idle", "straight", "turn"))
    parser.add_argument("--output", required=True, help="explicit JSONL evidence path")
    parser.add_argument("--overwrite", action="store_true", help="replace the explicit output file if it exists")
    parser.add_argument("--duration", type=_positive_float, default=45.0)
    parser.add_argument("--interval", type=_positive_float, default=0.20)
    parser.add_argument("--http-timeout", type=_positive_float, default=1.0)
    parser.add_argument("--plant-url", default="http://127.0.0.1:8794/state")
    parser.add_argument("--sonic-url", default="http://127.0.0.1:8792/telemetry")
    parser.add_argument("--min-root-z", type=float, default=0.55)
    parser.add_argument("--max-root-z", type=float, default=1.15)
    parser.add_argument("--max-roll-pitch", type=_positive_float, default=0.35)
    parser.add_argument("--max-position-jump", type=_positive_float, default=0.30)
    parser.add_argument("--max-yaw-jump", type=_positive_float, default=0.50)
    parser.add_argument("--max-lateral", type=_positive_float, default=0.25)
    parser.add_argument("--max-heading-error", type=_positive_float, default=0.35)
    parser.add_argument("--max-forward", type=_positive_float, default=0.80)
    parser.add_argument("--max-reverse", type=_positive_float, default=0.05)
    parser.add_argument("--progress-window", type=_positive_float, default=15.0)
    parser.add_argument("--min-progress", type=_positive_float, default=0.10)
    parser.add_argument("--expected-forward", type=float, default=0.50)
    parser.add_argument("--final-position-tolerance", type=_positive_float, default=0.20)
    parser.add_argument("--expected-turn-deg", type=float, default=90.0)
    parser.add_argument("--final-yaw-tolerance-deg", type=_positive_float, default=20.0)
    parser.add_argument("--min-turn-progress-deg", type=_positive_float, default=20.0)
    parser.add_argument("--max-turn-overshoot-deg", type=_positive_float, default=20.0)
    parser.add_argument("--max-turn-translation", type=_positive_float, default=0.25)
    parser.add_argument("--max-turn-reverse-deg", type=_positive_float, default=6.0)
    parser.add_argument("--max-idle-drift", type=_positive_float, default=0.02)
    parser.add_argument("--max-idle-yaw-drift-deg", type=_positive_float, default=2.0)
    parser.add_argument("--max-arm-constrained-joints", type=_nonnegative_int, default=0)
    parser.add_argument("--max-arm-step", type=_positive_float, default=0.15)
    parser.add_argument("--max-arm-raw-deviation", type=_positive_float, default=2.75)
    parser.add_argument("--max-arm-tracking-error", type=_positive_float, default=0.90)
    parser.add_argument("--max-arm-measured-deviation", type=_positive_float, default=2.75)
    parser.add_argument("--max-arm-velocity", type=_positive_float, default=7.0)
    parser.add_argument("--max-arm-torque", type=_positive_float, default=35.0)
    parser.add_argument(
        "--require-settled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require a stationary Plant idle observation after motion",
    )
    return parser


def _limits_from_args(args: argparse.Namespace) -> AcceptanceLimits:
    expected_turn = math.radians(args.expected_turn_deg)
    if not math.isfinite(expected_turn) or abs(expected_turn) < 1e-6:
        raise ValueError("--expected-turn-deg must be finite and non-zero")
    return AcceptanceLimits(
        phase=args.phase,
        min_root_z_m=args.min_root_z,
        max_root_z_m=args.max_root_z,
        max_roll_pitch_rad=args.max_roll_pitch,
        max_position_jump_m=args.max_position_jump,
        max_yaw_jump_rad=args.max_yaw_jump,
        max_lateral_m=args.max_lateral,
        max_heading_error_rad=args.max_heading_error,
        max_directional_travel_m=args.max_forward,
        max_reverse_m=args.max_reverse,
        progress_window_s=args.progress_window,
        min_progress_m=args.min_progress,
        expected_forward_m=args.expected_forward,
        final_position_tolerance_m=args.final_position_tolerance,
        expected_turn_rad=expected_turn,
        final_yaw_tolerance_rad=math.radians(args.final_yaw_tolerance_deg),
        min_turn_progress_rad=math.radians(args.min_turn_progress_deg),
        max_turn_overshoot_rad=math.radians(args.max_turn_overshoot_deg),
        max_turn_translation_m=args.max_turn_translation,
        max_turn_reverse_rad=math.radians(args.max_turn_reverse_deg),
        max_idle_drift_m=args.max_idle_drift,
        max_idle_yaw_drift_rad=math.radians(args.max_idle_yaw_drift_deg),
        max_arm_constrained_joints=args.max_arm_constrained_joints,
        max_arm_step_rad=args.max_arm_step,
        max_arm_raw_deviation_rad=args.max_arm_raw_deviation,
        max_arm_tracking_error_rad=args.max_arm_tracking_error,
        max_arm_measured_deviation_rad=args.max_arm_measured_deviation,
        max_arm_velocity_rps=args.max_arm_velocity,
        max_arm_estimated_torque_nm=args.max_arm_torque,
        require_settled=args.require_settled,
    )


def run_monitor(
    *,
    sampler: ReadonlyHttpSampler,
    evaluator: MotionAcceptanceEvaluator,
    writer: JsonlEvidenceWriter,
    duration_s: float,
    interval_s: float,
    session_id: str,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = clock()
    writer.write(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "session_start",
            "sessionId": session_id,
            "recordedAt": _utc_now(),
            "readOnly": True,
            "networkMethods": ["GET"],
            "plantUrl": sampler.plant_url,
            "sonicUrl": sampler.sonic_url,
            "limits": asdict(evaluator.limits),
        }
    )
    next_sample = started
    interrupted = False
    try:
        while True:
            now = clock()
            if now - started >= duration_s:
                break
            if now < next_sample:
                sleeper(min(next_sample - now, interval_s))
                continue
            observed_at = clock()
            try:
                plant, sonic = sampler.sample()
            except Exception as exc:  # Network evidence is a failed sample, not a crash.
                derived = evaluator.observe_error(str(exc), observed_s=observed_at)
                writer.write(
                    {
                        "schemaVersion": SCHEMA_VERSION,
                        "kind": "sample",
                        "sessionId": session_id,
                        "recordedAt": _utc_now(),
                        "observedMonotonicS": observed_at - started,
                        "derived": derived,
                    }
                )
            else:
                derived = evaluator.observe(plant, sonic, observed_s=observed_at)
                writer.write(
                    {
                        "schemaVersion": SCHEMA_VERSION,
                        "kind": "sample",
                        "sessionId": session_id,
                        "recordedAt": _utc_now(),
                        "observedMonotonicS": observed_at - started,
                        "plant": plant,
                        "sonic": sonic,
                        "derived": derived,
                    }
                )
            next_sample += interval_s
            if next_sample < clock() - interval_s:
                # Sampling never catches up by issuing a burst of old reads.
                next_sample = clock() + interval_s
    except KeyboardInterrupt:
        interrupted = True
    result = evaluator.finalize()
    result["interrupted"] = interrupted
    writer.write(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "result",
            "sessionId": session_id,
            "recordedAt": _utc_now(),
            **result,
        }
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        limits = _limits_from_args(args)
        sampler = ReadonlyHttpSampler(
            args.plant_url,
            args.sonic_url,
            timeout_s=args.http_timeout,
        )
        evaluator = MotionAcceptanceEvaluator(limits)
        session_id = uuid.uuid4().hex
        with JsonlEvidenceWriter(args.output, overwrite=args.overwrite) as writer:
            result = run_monitor(
                sampler=sampler,
                evaluator=evaluator,
                writer=writer,
                duration_s=args.duration,
                interval_s=args.interval,
                session_id=session_id,
            )
    except (OSError, ValueError) as exc:
        print(f"monitor setup failed: {exc}", file=sys.stderr)
        return 2

    verdict = "PASS" if result["pass"] else "FAIL"
    print(
        f"{verdict} phase={result['phase']} samples={result['sampleCount']} "
        f"duration={result['observedDurationS']:.2f}s evidence={Path(args.output).expanduser()}"
    )
    for failure in result["failures"]:
        print(
            f"FAIL {failure['code']} at +{failure['first_observed_s']:.2f}s: "
            f"{failure['detail']}",
            file=sys.stderr,
        )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
