#!/usr/bin/env python3
"""Read-only long-running SAFE_HOLD soak evidence collector.

The monitor has no control client.  Its only network operations are HTTP GETs
to three exact loopback resources: the MuJoCo Plant state, SONIC telemetry and
Robot Agent diagnostics.  It reuses ``MotionAcceptanceEvaluator`` for physical
posture, motion-intent, reset/fault and arm checks, and adds baseline-relative
runtime-stall, Robot Agent and process-restart evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from urllib.request import Request, urlopen
import uuid

import motion_acceptance_monitor as acceptance


SCHEMA_VERSION = "h-ide.idle-soak-monitor/1"
PLANT_URL = "http://127.0.0.1:8794/state"
SONIC_URL = "http://127.0.0.1:8792/telemetry"
ROBOT_AGENT_URL = "http://127.0.0.1:8796/v1/diagnostics"
ALLOWED_URLS = frozenset({PLANT_URL, SONIC_URL, ROBOT_AGENT_URL})
SAMPLE_INTERVAL_S = 0.5
ROBOT_AGENT_INTERVAL_S = 5.0
RAW_CHECKPOINT_INTERVAL_S = 10.0
DEFAULT_DURATION_S = 605.0
EXPECTED_STALL_THRESHOLD_S = 0.1
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

_PLANT_STAGES = (
    "simStep",
    "snapshotPublish",
    "viewerSnapshotPublish",
    "viewerSync",
    "rgbRender",
    "jpegEncode",
    "depthRender",
    "capturePostprocess",
    "renderCaptureTotal",
)
_PLANT_WORKERS = ("observationWorker", "viewerWorker")
_SONIC_GAP_PREFIXES = (
    "lowstate_callback",
    "control_publish",
    "health_probe",
)
_CORE_PROCESS_NAMES = (
    "mujoco-plant",
    "sonic-driver",
    "ros-cmd-vel-bridge",
    "ros-observation-bridge",
    "ros-safe-stop",
    "sim-robot-agent-gateway",
    "observation-gateway",
    "robot-agent-g1-sim",
)
_LOG_NAMES = tuple(f"{name}.log" for name in _CORE_PROCESS_NAMES)
_HARD_LOG_MARKERS = re.compile(
    r"\bqacc\b|\bnan\b|non[- ]finite|numerical instability|"
    r"implicit reset|teleport|roll(?:ed)? back|rollback|"
    r"traceback \(most recent call last\)|segmentation fault|core dumped",
    re.IGNORECASE,
)
_TEMPORAL_STALE_MARKERS = re.compile(
    r"\b(?:PLANT_LOWSTATE_STALE|SONIC_CONTROL_STALE)\b"
)


@dataclass(frozen=True, slots=True)
class IdleSoakLimits:
    min_root_z_m: float = 0.55
    max_root_z_m: float = 1.15
    max_roll_pitch_rad: float = 0.25
    max_position_jump_m: float = 0.05
    max_yaw_jump_rad: float = 0.10
    max_idle_drift_m: float = 0.02
    max_idle_yaw_drift_rad: float = 0.035
    max_root_linear_speed_mps: float = 0.02
    max_root_angular_speed_rps: float = 0.05
    max_arm_step_rad: float = 0.05
    max_arm_raw_deviation_rad: float = 1.0
    max_arm_tracking_error_rad: float = 0.10
    max_arm_measured_deviation_rad: float = 1.0
    max_arm_velocity_rps: float = 0.25
    max_arm_estimated_torque_nm: float = 5.0
    max_lower_body_target_error_rad: float = 0.15
    max_lower_body_target_rms_error_rad: float = 0.08
    max_lower_body_velocity_rps: float = 0.25
    max_lower_body_estimated_torque_nm: float = 8.0

    def acceptance_limits(self) -> acceptance.AcceptanceLimits:
        return acceptance.AcceptanceLimits(
            phase="idle",
            min_root_z_m=self.min_root_z_m,
            max_root_z_m=self.max_root_z_m,
            max_roll_pitch_rad=self.max_roll_pitch_rad,
            max_position_jump_m=self.max_position_jump_m,
            max_yaw_jump_rad=self.max_yaw_jump_rad,
            max_idle_drift_m=self.max_idle_drift_m,
            max_idle_yaw_drift_rad=self.max_idle_yaw_drift_rad,
            max_arm_constrained_joints=0,
            max_arm_step_rad=self.max_arm_step_rad,
            max_arm_raw_deviation_rad=self.max_arm_raw_deviation_rad,
            max_arm_tracking_error_rad=self.max_arm_tracking_error_rad,
            max_arm_measured_deviation_rad=self.max_arm_measured_deviation_rad,
            max_arm_velocity_rps=self.max_arm_velocity_rps,
            max_arm_estimated_torque_nm=self.max_arm_estimated_torque_nm,
        )


@dataclass(frozen=True, slots=True)
class Failure:
    code: str
    detail: str
    first_observed_s: float


@dataclass(frozen=True, slots=True)
class EndpointBatch:
    payloads: dict[str, dict[str, Any]]
    latency_ms: dict[str, float]
    errors: dict[str, str]


@dataclass(frozen=True, slots=True)
class LogScan:
    hard_matches: tuple[dict[str, Any], ...]
    stale_matches: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]
    offsets: dict[str, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} is not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is non-finite")
    return number


def _counter(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} is not a non-negative integer")
    return value


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} is not an object")
    return value


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


class ReadonlyIdleSoakSampler:
    """Issue bodyless GETs only to the three compiled-in loopback URLs."""

    def __init__(
        self,
        *,
        plant_url: str = PLANT_URL,
        sonic_url: str = SONIC_URL,
        robot_agent_url: str = ROBOT_AGENT_URL,
        timeout_s: float = 0.5,
        robot_agent_timeout_s: float = 2.0,
        opener: Callable[..., Any] = urlopen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        urls = (str(plant_url), str(sonic_url), str(robot_agent_url))
        if urls != (PLANT_URL, SONIC_URL, ROBOT_AGENT_URL):
            raise ValueError("idle soak URLs must match the fixed loopback GET allowlist")
        for value, name in (
            (timeout_s, "timeout_s"),
            (robot_agent_timeout_s, "robot_agent_timeout_s"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.urls = {
            "plant": str(plant_url),
            "sonic": str(sonic_url),
            "robotAgent": str(robot_agent_url),
        }
        self.timeout_s = float(timeout_s)
        self.robot_agent_timeout_s = float(robot_agent_timeout_s)
        self._opener = opener
        self._clock = clock

    def _get_json(self, name: str) -> tuple[dict[str, Any], float]:
        url = self.urls[name]
        if url not in ALLOWED_URLS:
            raise RuntimeError("network target left the fixed GET allowlist")
        request = Request(
            url,
            method="GET",
            headers={"Accept": "application/json", "Cache-Control": "no-store"},
        )
        started = self._clock()
        timeout = (
            self.robot_agent_timeout_s if name == "robotAgent" else self.timeout_s
        )
        with self._opener(request, timeout=timeout) as response:
            status_value = getattr(response, "status", None)
            if status_value is None:
                status_value = response.getcode()
            if int(status_value) != 200:
                raise RuntimeError(f"GET {url} returned HTTP {status_value}")
            response_url = getattr(response, "geturl", lambda: url)()
            if response_url != url:
                raise RuntimeError(f"GET {url} was redirected")
            body = response.read(MAX_RESPONSE_BYTES + 1)
        latency_ms = max(0.0, self._clock() - started) * 1_000.0
        if len(body) > MAX_RESPONSE_BYTES:
            raise RuntimeError(f"GET {url} returned an oversized response")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise RuntimeError(f"GET {url} did not return a JSON object")
        return payload, latency_ms

    def sample(self, *, include_robot_agent: bool) -> EndpointBatch:
        names = ["plant", "sonic"]
        if include_robot_agent:
            names.append("robotAgent")
        payloads: dict[str, dict[str, Any]] = {}
        latency_ms: dict[str, float] = {}
        errors: dict[str, str] = {}
        for name in names:
            try:
                payload, elapsed = self._get_json(name)
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"[:512]
            else:
                payloads[name] = payload
                latency_ms[name] = round(elapsed, 3)
        return EndpointBatch(payloads, latency_ms, errors)


class SafeJsonlEvidenceWriter:
    """Reuse the acceptance writer after rejecting every symlink ancestor."""

    def __init__(self, path: str | Path) -> None:
        candidate = Path(path).expanduser().absolute()
        cursor = candidate
        while True:
            if cursor.is_symlink():
                raise ValueError(f"refusing evidence path with symlink: {cursor}")
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        self.path = candidate
        self._writer = acceptance.JsonlEvidenceWriter(candidate, overwrite=False)

    def write(self, value: Mapping[str, Any]) -> None:
        self._writer.write(value)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> SafeJsonlEvidenceWriter:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class ProcessIdentityProbe:
    """Read stable PID/start-time identities for the motion-critical workers."""

    def __init__(
        self,
        runtime_dir: str | Path,
        *,
        proc_root: str | Path = "/proc",
        names: Iterable[str] = _CORE_PROCESS_NAMES,
    ) -> None:
        self.runtime_dir = Path(runtime_dir).expanduser()
        self.proc_root = Path(proc_root)
        self.names = tuple(names)

    @staticmethod
    def _start_ticks(stat_text: str) -> int:
        _prefix, separator, suffix = stat_text.rpartition(")")
        if not separator:
            raise ValueError("/proc stat did not contain a process name")
        fields = suffix.strip().split()
        # suffix starts with field 3 (state), so field 22 is index 19.
        if len(fields) <= 19:
            raise ValueError("/proc stat did not contain process start time")
        return int(fields[19])

    def snapshot(self) -> tuple[dict[str, Any], tuple[str, ...]]:
        values: dict[str, Any] = {}
        errors: list[str] = []
        for name in self.names:
            pid_path = self.runtime_dir / f"{name}.pid"
            try:
                if pid_path.is_symlink():
                    raise ValueError("PID file is a symlink")
                pid_text = pid_path.read_text(encoding="ascii").strip()
                if not pid_text.isdecimal() or int(pid_text) <= 0:
                    raise ValueError("PID file is invalid")
                pid = int(pid_text)
                stat_path = self.proc_root / str(pid) / "stat"
                stat_text = stat_path.read_text(encoding="utf-8")
                values[name] = {
                    "pid": pid,
                    "startTimeTicks": self._start_ticks(stat_text),
                }
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}"[:512])
        return values, tuple(errors)


class RuntimeLogScanner:
    """Scan only newly appended core logs for numerical/reset failure clues."""

    def __init__(
        self,
        runtime_dir: str | Path,
        *,
        log_names: Iterable[str] = _LOG_NAMES,
        max_initial_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        self.runtime_dir = Path(runtime_dir).expanduser()
        self.log_names = tuple(log_names)
        self.max_initial_bytes = int(max_initial_bytes)
        self._offsets: dict[str, int] = {}

    @staticmethod
    def _match_record(name: str, offset: int, line: str) -> dict[str, Any]:
        return {
            "log": name,
            "offset": offset,
            "line": line.strip()[:512],
        }

    def scan(self) -> LogScan:
        hard: list[dict[str, Any]] = []
        stale: list[dict[str, Any]] = []
        errors: list[str] = []
        for name in self.log_names:
            path = self.runtime_dir / name
            previous = self._offsets.get(name, 0)
            try:
                if path.is_symlink():
                    raise ValueError("log file is a symlink")
                size = path.stat().st_size
                if name not in self._offsets and size > self.max_initial_bytes:
                    raise ValueError("initial log exceeds bounded scan size")
                if size < previous:
                    raise ValueError("log size rolled back during soak")
                unread = size - previous
                if unread > self.max_initial_bytes:
                    raise ValueError("new log data exceeds bounded scan size")
                with path.open("rb") as stream:
                    stream.seek(previous)
                    body = stream.read(unread)
                self._offsets[name] = size
                text = body.decode("utf-8", errors="replace")
                line_offset = previous
                for raw_line in text.splitlines(keepends=True):
                    if _HARD_LOG_MARKERS.search(raw_line):
                        hard.append(self._match_record(name, line_offset, raw_line))
                    if _TEMPORAL_STALE_MARKERS.search(raw_line):
                        stale.append(self._match_record(name, line_offset, raw_line))
                    line_offset += len(raw_line.encode("utf-8", errors="replace"))
            except (OSError, ValueError) as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}"[:512])
        return LogScan(tuple(hard), tuple(stale), tuple(errors), dict(self._offsets))


class RuntimeCounterTracker:
    """Compare cumulative Plant/SONIC timing counters with the first snapshot."""

    def __init__(self) -> None:
        self.baseline: dict[str, int] | None = None
        self.previous: dict[str, int] | None = None
        self.latest: dict[str, int] = {}
        self.stall_paths: set[str] = set()
        self.stall_deltas: dict[str, int] = {}

    def _extract(
        self,
        plant: Mapping[str, Any],
        sonic: Mapping[str, Any],
    ) -> tuple[dict[str, int], Mapping[str, Any], Mapping[str, Any]]:
        plant_timing = _mapping(plant.get("runtimeTiming"), "plant.runtimeTiming")
        sonic_timing = _mapping(
            sonic.get("runtime_diagnostics"), "sonic.runtime_diagnostics"
        )
        if "snapshot_error" in sonic_timing:
            raise ValueError(
                f"sonic.runtime_diagnostics failed: {sonic_timing['snapshot_error']}"
            )
        plant_threshold = _finite_number(
            plant_timing.get("stallThresholdS"),
            "plant.runtimeTiming.stallThresholdS",
        )
        sonic_threshold = _finite_number(
            sonic_timing.get("runtime_stall_threshold_s"),
            "sonic.runtime_diagnostics.runtime_stall_threshold_s",
        )
        if not math.isclose(
            plant_threshold,
            EXPECTED_STALL_THRESHOLD_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            sonic_threshold,
            EXPECTED_STALL_THRESHOLD_S,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("runtime stall threshold is not the reviewed 0.1s")
        if sonic_timing.get("python_gc_enabled") is not True:
            raise ValueError("SONIC cyclic GC is not enabled")
        if sonic_timing.get("python_gc_freeze_applied") is not True:
            raise ValueError("SONIC warm runtime freeze was not applied")
        if sonic_timing.get("python_gc_freeze_error") is not None:
            raise ValueError("SONIC warm runtime freeze reported an error")

        counters: dict[str, int] = {}
        loop_gap = _mapping(
            plant_timing.get("simLoopGap"), "plant.runtimeTiming.simLoopGap"
        )
        path = "plant.simLoopGap.overThresholdCount"
        counters[path] = _counter(loop_gap.get("overThresholdCount"), path)
        self.stall_paths.add(path)
        stages = _mapping(plant_timing.get("stages"), "plant.runtimeTiming.stages")
        for stage in _PLANT_STAGES:
            stage_value = _mapping(stages.get(stage), f"plant.runtimeTiming.{stage}")
            path = f"plant.stages.{stage}.overThresholdCount"
            counters[path] = _counter(stage_value.get("overThresholdCount"), path)
            self.stall_paths.add(path)

        for prefix in _SONIC_GAP_PREFIXES:
            for suffix in (
                "last_over_threshold_gap_s",
                "last_over_threshold_gc_pause_s",
                "last_over_threshold_at_unix_s",
            ):
                key = f"{prefix}_{suffix}"
                if key not in sonic_timing:
                    raise ValueError(
                        f"sonic.runtime_diagnostics.{key} is missing"
                    )
                if sonic_timing[key] is not None:
                    number = _finite_number(
                        sonic_timing[key], f"sonic.runtime_diagnostics.{key}"
                    )
                    if number < 0:
                        raise ValueError(
                            f"sonic.runtime_diagnostics.{key} is negative"
                        )
            total_path = f"sonic.{prefix}_over_threshold_count"
            counters[total_path] = _counter(
                sonic_timing.get(f"{prefix}_over_threshold_count"), total_path
            )
            self.stall_paths.add(total_path)
            for suffix in (
                "over_threshold_gc_overlap_count",
                "over_threshold_non_gc_residual_count",
            ):
                path = f"sonic.{prefix}_{suffix}"
                counters[path] = _counter(sonic_timing.get(f"{prefix}_{suffix}"), path)
        gc_path = "sonic.python_gc_collection_count"
        counters[gc_path] = _counter(
            sonic_timing.get("python_gc_collection_count"), gc_path
        )
        return counters, plant_timing, sonic_timing

    def observe(
        self,
        plant: Mapping[str, Any],
        sonic: Mapping[str, Any],
    ) -> dict[str, Any]:
        counters, plant_timing, sonic_timing = self._extract(plant, sonic)
        if self.baseline is None:
            self.baseline = dict(counters)
            self.previous = dict(counters)
            self.latest = dict(counters)
            self.stall_deltas = {path: 0 for path in sorted(self.stall_paths)}
            return {
                "baseline": True,
                "counterChanges": {},
                "counterRollbacks": {},
                "stallEvent": False,
                "deltasFromBaseline": dict(self.stall_deltas),
            }
        assert self.previous is not None
        changes: dict[str, dict[str, int]] = {}
        rollbacks: dict[str, dict[str, int]] = {}
        for path, current in counters.items():
            prior = self.previous[path]
            if current < prior:
                rollbacks[path] = {"previous": prior, "current": current}
            elif current > prior:
                changes[path] = {"previous": prior, "current": current}
        self.previous = dict(counters)
        self.latest = dict(counters)
        self.stall_deltas = {
            path: counters[path] - self.baseline[path]
            for path in sorted(self.stall_paths)
        }
        stall_event = any(path in self.stall_paths for path in changes)
        event_evidence: dict[str, Any] = {}
        if stall_event:
            event_evidence = {
                "plantLastSlowStage": plant_timing.get("lastSlowStage"),
                "plantLastSlowDurationS": plant_timing.get("lastSlowDurationS"),
                "plantLastSlowAtUnixNs": plant_timing.get("lastSlowAtUnixNs"),
                "sonicGapAttribution": {
                    prefix: {
                        "lastOverThresholdAtUnixS": sonic_timing.get(
                            f"{prefix}_last_over_threshold_at_unix_s"
                        ),
                        "lastOverThresholdGapS": sonic_timing.get(
                            f"{prefix}_last_over_threshold_gap_s"
                        ),
                        "lastOverThresholdGcPauseS": sonic_timing.get(
                            f"{prefix}_last_over_threshold_gc_pause_s"
                        ),
                    }
                    for prefix in _SONIC_GAP_PREFIXES
                },
            }
        return {
            "baseline": False,
            "counterChanges": changes,
            "counterRollbacks": rollbacks,
            "stallEvent": stall_event,
            "deltasFromBaseline": dict(self.stall_deltas),
            "eventEvidence": event_evidence,
        }


class PlantWorkerIdentityTracker:
    """Keep render-worker health, process identity and failure evidence stable."""

    def __init__(self) -> None:
        self.baseline: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _worker_pid(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} is not a positive integer")
        return value

    @staticmethod
    def _worker_boot_id(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} is not a UUID")
        try:
            uuid.UUID(value)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"{field} is not a UUID") from exc
        return value

    @staticmethod
    def _failure_snapshot(
        value: Mapping[str, Any],
        field: str,
        errors: list[str],
    ) -> dict[str, Any]:
        try:
            failure_count = _counter(
                value.get("failureCount"), f"{field}.failureCount"
            )
        except ValueError as exc:
            errors.append(str(exc))
            failure_count = value.get("failureCount")
        failure_at = value.get("lastFailureAtUnixNs")
        if failure_at is not None and (
            isinstance(failure_at, bool)
            or not isinstance(failure_at, int)
            or failure_at <= 0
        ):
            errors.append(
                f"{field}.lastFailureAtUnixNs must be null or a positive integer"
            )
        last_error = value.get("lastError")
        if last_error is not None and (
            not isinstance(last_error, str) or not last_error.strip()
        ):
            errors.append(f"{field}.lastError is not null or a non-empty string")
        return {
            "failureCount": failure_count,
            "lastFailureAtUnixNs": failure_at,
            "lastError": last_error,
        }

    @classmethod
    def _enabled_snapshot(
        cls,
        name: str,
        value: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        health_errors: list[str] = []
        identity_errors: list[str] = []
        if value.get("status") != "running":
            health_errors.append(
                f"plant.{name}.status={value.get('status')!r}, expected 'running'"
            )
        if value.get("ready") is not True:
            health_errors.append(f"plant.{name}.ready is not true")
        if value.get("alive") is not True:
            health_errors.append(f"plant.{name}.alive is not true")
        if value.get("exitCode") is not None:
            health_errors.append(f"plant.{name}.exitCode is not null")
        try:
            worker_pid: Any = cls._worker_pid(
                value.get("workerPid"), f"plant.{name}.workerPid"
            )
        except ValueError as exc:
            identity_errors.append(str(exc))
            worker_pid = value.get("workerPid")
        try:
            worker_boot_id: Any = cls._worker_boot_id(
                value.get("workerBootId"), f"plant.{name}.workerBootId"
            )
        except ValueError as exc:
            identity_errors.append(str(exc))
            worker_boot_id = value.get("workerBootId")
        failure = cls._failure_snapshot(value, f"plant.{name}", health_errors)
        if name == "viewerWorker":
            if value.get("authorityFeedbackEnabled") is not False:
                health_errors.append(
                    "plant.viewerWorker.authorityFeedbackEnabled is not false"
                )
            if value.get("keyCallbackRegistered") is not False:
                health_errors.append(
                    "plant.viewerWorker.keyCallbackRegistered is not false"
                )
        return (
            {
                "mode": "enabled",
                "identity": {
                    "workerPid": worker_pid,
                    "workerBootId": worker_boot_id,
                },
                "failure": failure,
                "health": {
                    "status": value.get("status"),
                    "ready": value.get("ready"),
                    "alive": value.get("alive"),
                    "exitCode": value.get("exitCode"),
                },
            },
            health_errors,
            identity_errors,
        )

    @classmethod
    def _disabled_viewer_snapshot(
        cls,
        value: Mapping[str, Any],
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        health_errors: list[str] = []
        identity_errors: list[str] = []
        for field in ("ready", "alive"):
            if value.get(field) not in (None, False):
                health_errors.append(
                    f"plant.viewerWorker.{field} must be absent or false when disabled"
                )
        if value.get("exitCode") is not None:
            health_errors.append(
                "plant.viewerWorker.exitCode must be null when disabled"
            )
        if value.get("authorityFeedbackEnabled") is not False:
            health_errors.append(
                "plant.viewerWorker.authorityFeedbackEnabled is not false"
            )
        if value.get("keyCallbackRegistered") is not False:
            health_errors.append(
                "plant.viewerWorker.keyCallbackRegistered is not false"
            )
        for field in ("workerPid", "workerBootId"):
            if value.get(field) is not None:
                identity_errors.append(
                    f"plant.viewerWorker.{field} must be null when disabled"
                )
        failure_count = value.get("failureCount", 0)
        if failure_count != 0 or isinstance(failure_count, bool):
            health_errors.append(
                "plant.viewerWorker.failureCount must be absent or zero when disabled"
            )
        for field in ("lastFailureAtUnixNs", "lastError"):
            if value.get(field) is not None:
                health_errors.append(
                    f"plant.viewerWorker.{field} must be null when disabled"
                )
        return (
            {
                "mode": "disabled",
                "identity": {"workerPid": None, "workerBootId": None},
                "failure": {
                    "failureCount": 0,
                    "lastFailureAtUnixNs": None,
                    "lastError": None,
                },
                "health": {
                    "status": "disabled",
                    "ready": value.get("ready"),
                    "alive": value.get("alive"),
                    "exitCode": value.get("exitCode"),
                },
            },
            health_errors,
            identity_errors,
        )

    def observe(
        self,
        plant: Mapping[str, Any],
        *,
        authority_pid: int | None,
    ) -> dict[str, Any]:
        workers: dict[str, dict[str, Any]] = {}
        health_errors: list[str] = []
        identity_errors: list[str] = []
        identity_changes: dict[str, dict[str, Any]] = {}
        failure_changes: dict[str, dict[str, Any]] = {}
        first_snapshot = not self.baseline

        for name in _PLANT_WORKERS:
            raw = plant.get(name)
            worker_health: list[str] = []
            worker_identity: list[str] = []
            if not isinstance(raw, Mapping):
                health_errors.append(f"plant.{name} is not an object")
                snapshot = {
                    "mode": "invalid",
                    "identity": {"workerPid": None, "workerBootId": None},
                    "failure": {
                        "failureCount": None,
                        "lastFailureAtUnixNs": None,
                        "lastError": None,
                    },
                    "health": {"status": None},
                }
            elif name == "viewerWorker" and raw.get("status") == "disabled":
                snapshot, worker_health, worker_identity = (
                    self._disabled_viewer_snapshot(raw)
                )
                health_errors.extend(worker_health)
                identity_errors.extend(worker_identity)
            else:
                snapshot, worker_health, worker_identity = self._enabled_snapshot(
                    name, raw
                )
                health_errors.extend(worker_health)
                identity_errors.extend(worker_identity)

            baseline = self.baseline.get(name)
            if baseline is None:
                self.baseline[name] = snapshot
            else:
                baseline_identity = {
                    "mode": baseline["mode"],
                    **baseline["identity"],
                }
                current_identity = {
                    "mode": snapshot["mode"],
                    **snapshot["identity"],
                }
                if current_identity != baseline_identity:
                    identity_changes[name] = {
                        "baseline": baseline_identity,
                        "current": current_identity,
                    }
                if snapshot["failure"] != baseline["failure"]:
                    failure_changes[name] = {
                        "baseline": baseline["failure"],
                        "current": snapshot["failure"],
                    }
            self.latest[name] = snapshot
            workers[name] = {
                **snapshot,
                "healthErrors": worker_health,
                "identityErrors": worker_identity,
            }

        observation_identity = workers["observationWorker"]["identity"]
        viewer_identity = workers["viewerWorker"]["identity"]
        observation_pid = observation_identity.get("workerPid")
        viewer_pid = viewer_identity.get("workerPid")
        observation_boot_id = observation_identity.get("workerBootId")
        viewer_boot_id = viewer_identity.get("workerBootId")
        if authority_pid is not None and observation_pid == authority_pid:
            identity_errors.append(
                "plant.observationWorker.workerPid equals the authoritative Plant PID"
            )
        if authority_pid is not None and viewer_pid == authority_pid:
            identity_errors.append(
                "plant.viewerWorker.workerPid equals the authoritative Plant PID"
            )
        if viewer_pid is not None and viewer_pid == observation_pid:
            identity_errors.append(
                "Plant observation and viewer workers report the same PID"
            )
        if viewer_boot_id is not None and viewer_boot_id == observation_boot_id:
            identity_errors.append(
                "Plant observation and viewer workers report the same boot ID"
            )

        return {
            "baseline": first_snapshot,
            "workers": workers,
            "healthErrors": health_errors,
            "identityErrors": identity_errors,
            "identityChanges": identity_changes,
            "failureChanges": failure_changes,
        }


class IdleSoakEvaluator:
    """Combine the existing physical evaluator with soak-only invariants."""

    def __init__(self, limits: IdleSoakLimits | None = None) -> None:
        self.limits = limits or IdleSoakLimits()
        self.motion = acceptance.MotionAcceptanceEvaluator(
            self.limits.acceptance_limits()
        )
        self.runtime = RuntimeCounterTracker()
        self.workers = PlantWorkerIdentityTracker()
        self.failures: dict[str, Failure] = {}
        self.first_observed_s: float | None = None
        self.last_observed_s: float | None = None
        self.samples = 0
        self.robot_agent_samples = 0
        self.robot_agent_all_ready = True
        self._robot_boot_id: str | None = None
        self._safe_stop_epoch: int | None = None
        self._motion_fault_counters: dict[str, int] | None = None
        self._process_baseline: dict[str, Any] | None = None
        self._process_latest: dict[str, Any] = {}
        self.runtime_stall_seen = False
        self.temporal_stale_log_seen = False

    def _fail(self, code: str, detail: str, observed_s: float) -> None:
        if code in self.failures:
            return
        origin = self.first_observed_s if self.first_observed_s is not None else observed_s
        self.failures[code] = Failure(
            code=code,
            detail=str(detail)[:1024],
            first_observed_s=max(0.0, observed_s - origin),
        )

    def _observe_robot_agent(
        self,
        value: Mapping[str, Any],
        observed_s: float,
    ) -> dict[str, Any]:
        self.robot_agent_samples += 1
        errors: list[str] = []
        if value.get("schemaVersion") != "robot-agent.diagnostics/1":
            errors.append("unsupported Robot Agent diagnostics schema")
        if value.get("robotId") != "g1-sim":
            errors.append("Robot Agent robotId is not g1-sim")
        if value.get("ready") is not True:
            errors.append("Robot Agent ready is not true")
        command = value.get("commandReadiness")
        control = value.get("controlState")
        fault = value.get("motionSafetyFault")
        checks = value.get("checks")
        if not isinstance(command, Mapping):
            errors.append("Robot Agent commandReadiness is missing")
            command = {}
        if not isinstance(control, Mapping):
            errors.append("Robot Agent controlState is missing")
            control = {}
        if not isinstance(fault, Mapping):
            errors.append("Robot Agent motionSafetyFault is missing")
            fault = {}
        if not isinstance(checks, Mapping):
            errors.append("Robot Agent checks are missing")
            checks = {}
        if command.get("ready") is not True or command.get("mode") != "STANDBY":
            errors.append("Robot Agent command readiness is not STANDBY")
        expected_control = {
            "state": "SAFE_HOLD",
            "active": False,
            "standby": True,
            "canAcquire": True,
            "authorityRevoked": True,
            "safeStopState": "confirmed",
            "leaseValid": False,
            "activeCommandCount": 0,
        }
        for field, expected in expected_control.items():
            if control.get(field) != expected:
                errors.append(
                    f"Robot Agent controlState.{field}={control.get(field)!r}, "
                    f"expected {expected!r}"
                )
        if control.get("leaseRemainingMs") is not None:
            errors.append("Robot Agent leaseRemainingMs is not null in SAFE_HOLD")
        if fault.get("ready") is not True or fault.get("latched") is not False:
            errors.append("Robot Agent motion-safety fault is not clear")
        for field in ("eventId", "source", "code"):
            if fault.get(field) is not None:
                errors.append(f"Robot Agent motionSafetyFault.{field} is not null")
        for name, check in checks.items():
            if (
                isinstance(check, Mapping)
                and check.get("required") is True
                and check.get("ready") is not True
            ):
                errors.append(f"Robot Agent required check {name} is not ready")
        required_check_names = {
            "runtime",
            "transportSession",
            "control",
            "safeStop",
            "safetyEndpoints",
            "rosExecutor",
            "nav2Actions",
            "localization",
            "observation",
            "motionSafetyFault",
        }
        missing_checks = sorted(required_check_names.difference(checks))
        if missing_checks:
            errors.append(
                "Robot Agent checks are missing: " + ", ".join(missing_checks)
            )

        boot_id = value.get("robotBootId")
        safe_stop_epoch = control.get("safeStopEpoch")
        if not isinstance(boot_id, str) or not boot_id:
            errors.append("Robot Agent robotBootId is invalid")
        if isinstance(safe_stop_epoch, bool) or not isinstance(safe_stop_epoch, int):
            errors.append("Robot Agent safeStopEpoch is invalid")
        counters_value = fault.get("counters")
        parsed_counters: dict[str, int] | None = None
        if isinstance(counters_value, Mapping):
            try:
                parsed_counters = {
                    str(name): _counter(counter_value, f"motion fault counter {name}")
                    for name, counter_value in counters_value.items()
                }
            except ValueError as exc:
                errors.append(str(exc))
        else:
            errors.append("Robot Agent motion-fault counters are missing")
        if self._robot_boot_id is None and isinstance(boot_id, str) and boot_id:
            self._robot_boot_id = boot_id
            self._safe_stop_epoch = (
                safe_stop_epoch if isinstance(safe_stop_epoch, int) else None
            )
            self._motion_fault_counters = parsed_counters
        else:
            if boot_id != self._robot_boot_id:
                errors.append("Robot Agent boot identity changed during soak")
            if safe_stop_epoch != self._safe_stop_epoch:
                errors.append("Robot Agent safe-stop epoch changed during soak")
            if parsed_counters != self._motion_fault_counters:
                errors.append("Robot Agent motion-fault counters changed during soak")
        nonfinite = _nonfinite_paths(value, "robotAgent")
        if nonfinite:
            errors.append("non-finite Robot Agent values: " + ", ".join(nonfinite[:8]))
        if errors:
            self.robot_agent_all_ready = False
            self._fail("ROBOT_AGENT_NOT_READY", "; ".join(errors), observed_s)
        return {
            "ready": not errors,
            "robotBootId": boot_id,
            "safeStopEpoch": safe_stop_epoch,
            "errors": errors,
        }

    def _observe_lower_body(
        self,
        sonic: Mapping[str, Any],
        observed_s: float,
    ) -> dict[str, float]:
        fields = {
            "lower_body_target_max_error_rad": self.limits.max_lower_body_target_error_rad,
            "lower_body_target_rms_error_rad": self.limits.max_lower_body_target_rms_error_rad,
            "lower_body_max_velocity_rps": self.limits.max_lower_body_velocity_rps,
            "lower_body_max_estimated_torque_nm": self.limits.max_lower_body_estimated_torque_nm,
        }
        observed: dict[str, float] = {}
        for field, limit in fields.items():
            try:
                value = _finite_number(sonic.get(field), f"sonic.{field}")
            except ValueError as exc:
                self._fail("LOWER_BODY_TELEMETRY_INVALID", str(exc), observed_s)
                continue
            observed[field] = value
            if value < 0 or value > limit:
                self._fail(
                    "LOWER_BODY_TELEMETRY_SPIKE",
                    f"{field} reached {value:.6f}, limit {limit:.6f}",
                    observed_s,
                )
        return observed

    def _observe_processes(
        self,
        snapshot: Mapping[str, Any],
        errors: Iterable[str],
        observed_s: float,
    ) -> bool:
        errors = tuple(errors)
        if errors:
            self._fail("PROCESS_IDENTITY_UNAVAILABLE", "; ".join(errors), observed_s)
        changed = False
        if self._process_baseline is None and not errors:
            self._process_baseline = dict(snapshot)
        elif self._process_baseline is not None and dict(snapshot) != self._process_baseline:
            changed = True
            self._fail(
                "PROCESS_RESTARTED",
                "motion-critical PID/start-time identity changed during soak",
                observed_s,
            )
        self._process_latest = dict(snapshot)
        return changed

    def observe(
        self,
        batch: EndpointBatch,
        *,
        process_snapshot: Mapping[str, Any],
        process_errors: Iterable[str],
        log_scan: LogScan | None,
        observed_s: float,
    ) -> dict[str, Any]:
        if self.first_observed_s is None:
            self.first_observed_s = observed_s
        self.last_observed_s = observed_s
        self.samples += 1
        for name, error in batch.errors.items():
            self._fail("ENDPOINT_UNAVAILABLE", f"{name}: {error}", observed_s)

        process_changed = self._observe_processes(
            process_snapshot, process_errors, observed_s
        )
        if log_scan is not None:
            if log_scan.errors:
                self._fail("RUNTIME_LOG_UNAVAILABLE", "; ".join(log_scan.errors), observed_s)
            if log_scan.hard_matches:
                detail = "; ".join(
                    f"{match['log']}:{match['offset']}: {match['line']}"
                    for match in log_scan.hard_matches[:4]
                )
                self._fail("NUMERICAL_RESET_OR_CRASH_LOG", detail, observed_s)
            if log_scan.stale_matches:
                self.temporal_stale_log_seen = True

        plant = batch.payloads.get("plant")
        sonic = batch.payloads.get("sonic")
        motion_derived: dict[str, Any] = {}
        runtime_derived: dict[str, Any] = {
            "stallEvent": False,
            "counterChanges": {},
            "counterRollbacks": {},
            "deltasFromBaseline": dict(self.runtime.stall_deltas),
        }
        lower_body: dict[str, float] = {}
        worker_derived: dict[str, Any] = {
            "baseline": False,
            "workers": {},
            "healthErrors": [],
            "identityErrors": [],
            "identityChanges": {},
            "failureChanges": {},
        }
        if plant is not None:
            authority_pid: int | None = None
            authority_process = process_snapshot.get("mujoco-plant")
            if isinstance(authority_process, Mapping):
                candidate_pid = authority_process.get("pid")
                if (
                    isinstance(candidate_pid, int)
                    and not isinstance(candidate_pid, bool)
                    and candidate_pid > 0
                ):
                    authority_pid = candidate_pid
            worker_derived = self.workers.observe(
                plant,
                authority_pid=authority_pid,
            )
            if worker_derived["healthErrors"]:
                self._fail(
                    "PLANT_WORKER_HEALTH_INVALID",
                    "; ".join(worker_derived["healthErrors"]),
                    observed_s,
                )
            if worker_derived["identityErrors"]:
                self._fail(
                    "PLANT_WORKER_IDENTITY_INVALID",
                    "; ".join(worker_derived["identityErrors"]),
                    observed_s,
                )
            if worker_derived["identityChanges"]:
                self._fail(
                    "PLANT_WORKER_IDENTITY_CHANGED",
                    json.dumps(
                        worker_derived["identityChanges"],
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    observed_s,
                )
            if worker_derived["failureChanges"]:
                self._fail(
                    "PLANT_WORKER_FAILURE_CHANGED",
                    json.dumps(
                        worker_derived["failureChanges"],
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    observed_s,
                )
        if plant is None or sonic is None:
            self.motion.observe_error(
                "Plant or SONIC sample unavailable", observed_s=observed_s
            )
        else:
            motion_derived = self.motion.observe(
                plant, sonic, observed_s=observed_s
            )
            linear_speed = motion_derived.get("linear_speed_mps")
            angular_speed = motion_derived.get("angular_speed_rps")
            if (
                isinstance(linear_speed, (int, float))
                and not isinstance(linear_speed, bool)
                and math.isfinite(float(linear_speed))
                and float(linear_speed) > self.limits.max_root_linear_speed_mps
            ):
                self._fail(
                    "IDLE_ROOT_SPEED_EXCEEDED",
                    f"root linear speed reached {float(linear_speed):.6f}m/s",
                    observed_s,
                )
            if (
                isinstance(angular_speed, (int, float))
                and not isinstance(angular_speed, bool)
                and math.isfinite(float(angular_speed))
                and float(angular_speed) > self.limits.max_root_angular_speed_rps
            ):
                self._fail(
                    "IDLE_ROOT_SPEED_EXCEEDED",
                    f"root angular speed reached {float(angular_speed):.6f}rad/s",
                    observed_s,
                )
            lower_body = self._observe_lower_body(sonic, observed_s)
            try:
                runtime_derived = self.runtime.observe(plant, sonic)
            except (KeyError, TypeError, ValueError) as exc:
                self._fail("RUNTIME_DIAGNOSTICS_INVALID", str(exc), observed_s)
            else:
                if runtime_derived["counterRollbacks"]:
                    self._fail(
                        "RUNTIME_COUNTER_ROLLBACK",
                        json.dumps(
                            runtime_derived["counterRollbacks"],
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        observed_s,
                    )
                if runtime_derived["stallEvent"]:
                    self.runtime_stall_seen = True

        robot_agent_derived: dict[str, Any] | None = None
        robot_agent = batch.payloads.get("robotAgent")
        if robot_agent is not None:
            robot_agent_derived = self._observe_robot_agent(
                robot_agent, observed_s
            )
        elif "robotAgent" in batch.errors:
            self.robot_agent_all_ready = False

        return {
            "sample": self.samples,
            "motion": motion_derived,
            "lowerBody": lower_body,
            "runtime": runtime_derived,
            "plantWorkers": worker_derived,
            "robotAgent": robot_agent_derived,
            "processIdentityChanged": process_changed,
        }

    def finalize(self) -> dict[str, Any]:
        motion_result = self.motion.finalize()
        if self.robot_agent_samples < 2:
            final_time = self.last_observed_s or 0.0
            self._fail(
                "INSUFFICIENT_ROBOT_AGENT_SAMPLES",
                "baseline and final Robot Agent diagnostics are required",
                final_time,
            )
        hard_failures = [asdict(failure) for failure in self.failures.values()]
        reported_failures = list(hard_failures)
        existing_codes = {failure["code"] for failure in reported_failures}
        for failure in motion_result["failures"]:
            if failure["code"] not in existing_codes:
                reported_failures.append(dict(failure))
                existing_codes.add(failure["code"])
        safe_hold_pass = (
            bool(motion_result["pass"])
            and not hard_failures
            and self.robot_agent_all_ready
        )
        quarantine_exercised = bool(
            (self.runtime_stall_seen or self.temporal_stale_log_seen)
            and safe_hold_pass
        )
        movement_release_allowed = bool(
            safe_hold_pass
            and not self.runtime_stall_seen
            and not self.temporal_stale_log_seen
            and all(delta == 0 for delta in self.runtime.stall_deltas.values())
        )
        elapsed = (
            0.0
            if self.first_observed_s is None or self.last_observed_s is None
            else max(0.0, self.last_observed_s - self.first_observed_s)
        )
        return {
            "safeHoldSafetyPass": safe_hold_pass,
            "movementReleaseAllowed": movement_release_allowed,
            "quarantineExercised": quarantine_exercised,
            "runtimeStallSeen": self.runtime_stall_seen,
            "temporalStaleLogSeen": self.temporal_stale_log_seen,
            "sampleCount": self.samples,
            "robotAgentSampleCount": self.robot_agent_samples,
            "observedDurationS": elapsed,
            "runtimeCounterDeltas": dict(self.runtime.stall_deltas),
            "processBaseline": self._process_baseline,
            "processFinal": self._process_latest,
            "plantWorkerBaseline": dict(self.workers.baseline),
            "plantWorkerFinal": dict(self.workers.latest),
            "motionAcceptance": motion_result,
            "failures": reported_failures,
        }


def _compact_payloads(batch: EndpointBatch) -> dict[str, Any]:
    plant = batch.payloads.get("plant", {})
    sonic = batch.payloads.get("sonic", {})
    robot_agent = batch.payloads.get("robotAgent")
    support = plant.get("support") if isinstance(plant.get("support"), Mapping) else {}
    result: dict[str, Any] = {
        "plant": {
            "status": plant.get("status"),
            "physicsStatus": plant.get("physicsStatus"),
            "observationStatus": plant.get("observationStatus"),
            "producerBootId": plant.get("producerBootId"),
            "sequence": plant.get("sequence"),
            "capturedAtUnixNs": plant.get("capturedAtUnixNs"),
            "pose": plant.get("pose"),
            "rootVelocity": plant.get("rootVelocity"),
            "support": support,
            "observationWorker": plant.get("observationWorker"),
            "viewerWorker": plant.get("viewerWorker"),
        },
        "sonic": {
            key: sonic.get(key)
            for key in (
                "status",
                "updated_at",
                "plant_tick",
                "control_tick_ms",
                "motion_ready",
                "control_clock_status",
                "nav2_state",
                "sonic_mode",
                "last_cmd_vx",
                "last_cmd_vy",
                "last_cmd_wz",
                "sonic_speed_mps",
                "sonic_policy_speed_mps",
                "navigation_task_active",
                "static_hold_active",
                "plant_support_mode",
                "plant_support_stop_pending",
                "plant_support_safety_latched",
                "navigation_arm_max_velocity_rps",
                "navigation_arm_max_estimated_torque_nm",
                "lower_body_max_velocity_rps",
                "lower_body_max_estimated_torque_nm",
                "fault_trace_path",
                "fault_trace_sample_count",
                "last_error",
            )
        },
        "latencyMs": batch.latency_ms,
        "errors": batch.errors,
    }
    if robot_agent is not None:
        result["robotAgent"] = {
            "ready": robot_agent.get("ready"),
            "robotBootId": robot_agent.get("robotBootId"),
            "commandReadiness": robot_agent.get("commandReadiness"),
            "controlState": robot_agent.get("controlState"),
            "motionSafetyFault": robot_agent.get("motionSafetyFault"),
        }
    return result


def _full_record(
    *,
    kind: str,
    session_id: str,
    elapsed_s: float,
    batch: EndpointBatch,
    processes: Mapping[str, Any],
    process_errors: Iterable[str],
    log_scan: LogScan | None,
    derived: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "kind": kind,
        "sessionId": session_id,
        "recordedAt": _utc_now(),
        "observedMonotonicS": elapsed_s,
        "payloads": batch.payloads,
        "latencyMs": batch.latency_ms,
        "endpointErrors": batch.errors,
        "processes": dict(processes),
        "processErrors": list(process_errors),
        "logScan": None if log_scan is None else asdict(log_scan),
        "derived": dict(derived),
    }


def run_soak(
    *,
    sampler: ReadonlyIdleSoakSampler,
    evaluator: IdleSoakEvaluator,
    writer: SafeJsonlEvidenceWriter,
    process_probe: ProcessIdentityProbe,
    log_scanner: RuntimeLogScanner,
    duration_s: float,
    session_id: str,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    started = clock()
    writer.write(
        {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "session_start",
            "sessionId": session_id,
            "recordedAt": _utc_now(),
            "readOnly": True,
            "networkMethods": ["GET"],
            "networkAllowlist": sorted(ALLOWED_URLS),
            "durationS": duration_s,
            "plantSonicIntervalS": SAMPLE_INTERVAL_S,
            "robotAgentIntervalS": ROBOT_AGENT_INTERVAL_S,
            "rawCheckpointIntervalS": RAW_CHECKPOINT_INTERVAL_S,
            "limits": asdict(evaluator.limits),
        }
    )

    next_sample = started
    next_robot_agent = started
    next_raw = started
    first = True
    interrupted = False

    def take_sample(*, force_robot_agent: bool, final: bool = False) -> None:
        nonlocal next_robot_agent, next_raw, first
        observed = clock()
        elapsed = max(0.0, observed - started)
        include_ra = force_robot_agent or observed >= next_robot_agent
        batch = sampler.sample(include_robot_agent=include_ra)
        processes, process_errors = process_probe.snapshot()
        scan_logs = include_ra or first or final
        log_scan = log_scanner.scan() if scan_logs else None
        derived = evaluator.observe(
            batch,
            process_snapshot=processes,
            process_errors=process_errors,
            log_scan=log_scan,
            observed_s=observed,
        )
        kind = "baseline" if first else "final_checkpoint" if final else "sample"
        if first or final:
            writer.write(
                _full_record(
                    kind=kind,
                    session_id=session_id,
                    elapsed_s=elapsed,
                    batch=batch,
                    processes=processes,
                    process_errors=process_errors,
                    log_scan=log_scan,
                    derived=derived,
                )
            )
        else:
            writer.write(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "kind": "sample",
                    "sessionId": session_id,
                    "recordedAt": _utc_now(),
                    "observedMonotonicS": elapsed,
                    **_compact_payloads(batch),
                    "derived": derived,
                }
            )
        runtime_event = bool(
            derived["runtime"].get("stallEvent")
            or derived["runtime"].get("counterRollbacks")
        )
        process_event = bool(derived.get("processIdentityChanged"))
        worker_evidence = derived.get("plantWorkers", {})
        worker_event = bool(
            worker_evidence.get("healthErrors")
            or worker_evidence.get("identityErrors")
            or worker_evidence.get("identityChanges")
            or worker_evidence.get("failureChanges")
        )
        failure_event = bool(
            batch.errors
            or tuple(process_errors)
            or (
                log_scan is not None
                and (log_scan.hard_matches or log_scan.errors)
            )
        )
        if not first and not final and (
            runtime_event
            or process_event
            or worker_event
            or failure_event
            or observed >= next_raw
        ):
            event_kind = (
                "runtime_event"
                if runtime_event
                else "process_event"
                if process_event
                else "worker_event"
                if worker_event
                else "failure_event"
                if failure_event
                else "raw_checkpoint"
            )
            writer.write(
                _full_record(
                    kind=event_kind,
                    session_id=session_id,
                    elapsed_s=elapsed,
                    batch=batch,
                    processes=processes,
                    process_errors=process_errors,
                    log_scan=log_scan,
                    derived=derived,
                )
            )
        if include_ra:
            while next_robot_agent <= observed:
                next_robot_agent += ROBOT_AGENT_INTERVAL_S
        if observed >= next_raw:
            while next_raw <= observed:
                next_raw += RAW_CHECKPOINT_INTERVAL_S
        first = False

    take_sample(force_robot_agent=True)
    next_sample = started + SAMPLE_INTERVAL_S
    try:
        while True:
            now = clock()
            if now - started >= duration_s:
                break
            if now < next_sample:
                sleeper(min(next_sample - now, SAMPLE_INTERVAL_S))
                continue
            take_sample(force_robot_agent=False)
            next_sample += SAMPLE_INTERVAL_S
            if next_sample < clock() - SAMPLE_INTERVAL_S:
                next_sample = clock() + SAMPLE_INTERVAL_S
    except KeyboardInterrupt:
        interrupted = True
    take_sample(force_robot_agent=True, final=True)
    result = evaluator.finalize()
    result["interrupted"] = interrupted
    if interrupted:
        result["safeHoldSafetyPass"] = False
        result["movementReleaseAllowed"] = False
        result["failures"].append(
            {
                "code": "MONITOR_INTERRUPTED",
                "detail": "idle soak monitor was interrupted",
                "first_observed_s": result["observedDurationS"],
            }
        )
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


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return number


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only 10-minute SAFE_HOLD soak monitor. Network access is "
            "limited to three fixed loopback HTTP GET resources."
        )
    )
    parser.add_argument("--output", required=True, help="new JSONL evidence path")
    parser.add_argument(
        "--duration",
        type=_positive_float,
        default=DEFAULT_DURATION_S,
        help="soak duration in seconds (default: 605)",
    )
    parser.add_argument(
        "--runtime-dir",
        default=f"/run/user/{os.getuid()}/h-ide-mujoco-demo-{os.getuid()}",
        help="read-only runtime PID/log directory",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        sampler = ReadonlyIdleSoakSampler()
        evaluator = IdleSoakEvaluator()
        process_probe = ProcessIdentityProbe(args.runtime_dir)
        log_scanner = RuntimeLogScanner(args.runtime_dir)
        with SafeJsonlEvidenceWriter(args.output) as writer:
            result = run_soak(
                sampler=sampler,
                evaluator=evaluator,
                writer=writer,
                process_probe=process_probe,
                log_scanner=log_scanner,
                duration_s=args.duration,
                session_id=uuid.uuid4().hex,
            )
    except (OSError, ValueError) as exc:
        print(f"idle soak monitor setup failed: {exc}", file=sys.stderr)
        return 2

    safety = "PASS" if result["safeHoldSafetyPass"] else "FAIL"
    release = "GO" if result["movementReleaseAllowed"] else "HOLD"
    print(
        f"SAFE_HOLD={safety} movement={release} "
        f"samples={result['sampleCount']} "
        f"duration={result['observedDurationS']:.2f}s "
        f"evidence={Path(args.output).expanduser().absolute()}"
    )
    for failure in result["failures"]:
        print(
            f"FAIL {failure['code']} at +{failure['first_observed_s']:.2f}s: "
            f"{failure['detail']}",
            file=sys.stderr,
        )
    return 0 if result["movementReleaseAllowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
