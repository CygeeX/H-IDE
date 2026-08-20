#!/usr/bin/env python3
"""Execute exactly one fail-closed 0.2 m Robot Agent navigation.

Importing this module is inert.  Network mutation is possible only from
``main`` (or an explicit ``Forward02mRunner.run`` call), and every mutating
operation has a single-attempt latch whose ambiguous outcome is never retried.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4


SCHEMA_VERSION = "h-ide.forward-02m-acceptance/1"
ENV_PATH = Path("/home/phi/learn-projects/h-ide/.env")
TOKEN_KEY = "SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN"
DEFAULT_GATEWAY_BASE = "http://127.0.0.1:8094"
PLANT_URL = "http://127.0.0.1:8794/state"
SONIC_URL = "http://127.0.0.1:8792/telemetry"
ROBOT_AGENT_URL = "http://127.0.0.1:8796/v1/diagnostics"
ROBOT_ID = "g1-sim"
COMMAND_KEY = "navigation.navigate_to_pose"
PROGRESS_M = 0.03
PROGRESS_DEADLINE_S = 15.0
RUN_DEADLINE_S = 25.0
SETTLE_S = 1.0
FILE_STALL_S = 2.0
POLL_INTERVAL_S = 0.1
MAX_JSON_BYTES = 2 * 1024 * 1024
MOTION_MONITOR_SCHEMA = "h-ide.motion-acceptance-monitor/1"
SIDECAR_MONITOR_SCHEMA = "h-ide.idle-soak-monitor/1"
_MOTION_RECORD_KINDS = frozenset({"session_start", "sample", "result"})
_SIDECAR_RECORD_KINDS = frozenset(
    {
        "session_start",
        "baseline",
        "sample",
        "raw_checkpoint",
        "runtime_event",
        "process_event",
        "worker_event",
        "failure_event",
        "final_checkpoint",
        "result",
    }
)
_CORE_PROCESS_NAMES = frozenset(
    {
        "mujoco-plant",
        "sonic-driver",
        "ros-cmd-vel-bridge",
        "ros-observation-bridge",
        "ros-safe-stop",
        "sim-robot-agent-gateway",
        "observation-gateway",
        "robot-agent-g1-sim",
    }
)
_EXPECTED_MOTION_LIMITS: Mapping[str, Any] = {
    "phase": "straight",
    "min_root_z_m": 0.55,
    "max_root_z_m": 1.15,
    "max_roll_pitch_rad": 0.25,
    "max_position_jump_m": 0.08,
    "max_yaw_jump_rad": 0.20,
    "max_lateral_m": 0.12,
    "max_heading_error_rad": 0.20,
    "max_directional_travel_m": 0.30,
    "max_reverse_m": 0.03,
    "progress_window_s": 15.0,
    "min_progress_m": 0.03,
    "expected_forward_m": 0.20,
    "final_position_tolerance_m": 0.05,
    "max_arm_constrained_joints": 0,
    "max_arm_step_rad": 0.12,
    "max_arm_raw_deviation_rad": 1.25,
    "max_arm_tracking_error_rad": 0.50,
    "max_arm_measured_deviation_rad": 1.0,
    "max_arm_velocity_rps": 2.0,
    "max_arm_estimated_torque_nm": 15.0,
    "require_settled": True,
}


class AcceptanceError(RuntimeError):
    """A fail-closed runner decision safe to include in evidence."""


class TransportUncertain(AcceptanceError):
    """The peer may have received the request but no response was proved."""


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    payload: Any


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> HttpResponse: ...


class JsonlSource(Protocol):
    def read_new(self) -> list[dict[str, Any]]: ...


def _decode_env_value(raw: str) -> str:
    """Decode one dotenv scalar without executing or expanding anything."""

    value = raw.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote_char = value[0]
        escaped = False
        end = None
        for index in range(1, len(value)):
            char = value[index]
            if quote_char == '"' and char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote_char and not escaped:
                end = index
                break
            escaped = False
        trailing = "" if end is None else value[end + 1 :].strip()
        if end is None or (trailing and not trailing.startswith("#")):
            raise AcceptanceError("gateway token assignment has invalid quoting")
        inner = value[1:end]
        if quote_char == "'":
            return inner
        # Dotenv's small, explicit double-quote escape set; no interpolation.
        return re.sub(
            r"\\([\\\"nrt])",
            lambda match: {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}[match.group(1)],
            inner,
        )
    # Unquoted comments begin only after whitespace, as in common dotenv parsers.
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    if any(char in value for char in "\r\n\x00"):
        raise AcceptanceError("gateway token assignment contains a forbidden character")
    return value


def load_gateway_token(path: str | Path = ENV_PATH) -> str:
    """Return the last exact token assignment without shell evaluation/output."""

    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise AcceptanceError("cannot securely open the gateway environment file") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 1024 * 1024:
            raise AcceptanceError("gateway environment path is not a small regular file")
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o022:
            raise AcceptanceError("gateway environment file ownership or write permissions are unsafe")
        chunks: list[bytes] = []
        remaining = metadata.st_size + 1
        while remaining > 0:
            part = os.read(fd, min(65536, remaining))
            if not part:
                break
            chunks.append(part)
            remaining -= len(part)
        if sum(map(len, chunks)) > 1024 * 1024:
            raise AcceptanceError("gateway environment file is oversized")
        after = os.fstat(fd)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ):
            raise AcceptanceError("gateway environment file changed while being read")
    finally:
        os.close(fd)
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AcceptanceError("gateway environment file is not UTF-8") from exc
    pattern = re.compile(rf"^(?:export[ \t]+)?{re.escape(TOKEN_KEY)}[ \t]*=(.*)$")
    found: str | None = None
    for line in text.splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            found = _decode_env_value(match.group(1))
    if (
        found is None
        or not found
        or any(ord(char) < 0x21 or ord(char) == 0x7F for char in found)
    ):
        raise AcceptanceError("gateway token is missing or invalid")
    return found


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class UrllibTransport:
    """Small JSON transport with proxies and redirects disabled."""

    def __init__(self) -> None:
        # The command dispatch runs in its own worker while the main thread
        # performs read-only safety polling.  OpenerDirector does not promise
        # a thread-safe shared instance, so each calling thread owns one.
        self._thread_local = threading.local()

    def _thread_opener(self) -> Any:
        opener = getattr(self._thread_local, "opener", None)
        if opener is None:
            opener = build_opener(ProxyHandler({}), _RejectRedirects())
            self._thread_local.opener = opener
        return opener

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json_body: Mapping[str, Any] | None = None,
        timeout_s: float = 1.0,
    ) -> HttpResponse:
        body = None
        request_headers = {"Accept": "application/json", "Cache-Control": "no-store"}
        if headers:
            request_headers.update(headers)
        if json_body is not None:
            body = json.dumps(json_body, separators=(",", ":"), allow_nan=False).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=request_headers, method=method)
        try:
            with self._thread_opener().open(request, timeout=timeout_s) as response:
                status_code = int(response.status)
                raw = response.read(MAX_JSON_BYTES + 1)
        except HTTPError as exc:
            status_code = int(exc.code)
            raw = exc.read(MAX_JSON_BYTES + 1)
        except Exception as exc:
            raise TransportUncertain(f"{method} transport outcome is uncertain") from exc
        if len(raw) > MAX_JSON_BYTES:
            raise AcceptanceError(f"{method} response is oversized")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcceptanceError(f"{method} response is not JSON") from exc
        return HttpResponse(status_code, payload)


class ExclusiveEvidence:
    """Canonical 0600 JSONL created with O_EXCL and O_NOFOLLOW."""

    def __init__(self, path: str | Path) -> None:
        supplied = Path(path).expanduser().absolute()
        parent = supplied.parent.resolve(strict=True)
        if supplied.parent != parent:
            raise AcceptanceError("evidence parent must be canonical and symlink-free")
        canonical = parent / supplied.name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self._fd = os.open(canonical, flags, 0o600)
        except OSError as exc:
            raise AcceptanceError("cannot exclusively create evidence file") from exc
        if not stat.S_ISREG(os.fstat(self._fd).st_mode):
            os.close(self._fd)
            raise AcceptanceError("evidence destination is not a regular file")
        self.path = canonical
        self.closed = False

    def write(self, value: Mapping[str, Any]) -> None:
        encoded = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        view = memoryview(encoded)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                raise AcceptanceError("failed to append evidence")
            view = view[written:]
        os.fsync(self._fd)

    def close(self) -> None:
        if not self.closed:
            os.close(self._fd)
            self.closed = True

    def __enter__(self) -> "ExclusiveEvidence":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class JsonlTail:
    """Incrementally read complete JSON objects from one immutable file identity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self._fd = os.open(self.path, flags)
        except OSError as exc:
            raise AcceptanceError(f"cannot securely open JSONL input {self.path.name}") from exc
        metadata = os.fstat(self._fd)
        if not stat.S_ISREG(metadata.st_mode):
            os.close(self._fd)
            raise AcceptanceError(f"JSONL input {self.path.name} is not a regular file")
        self._identity = (metadata.st_dev, metadata.st_ino)
        self._offset = 0
        self._buffer = b""
        self.closed = False

    def read_new(self) -> list[dict[str, Any]]:
        metadata = os.fstat(self._fd)
        if (metadata.st_dev, metadata.st_ino) != self._identity:
            raise AcceptanceError(f"JSONL input {self.path.name} identity changed")
        if metadata.st_size < self._offset:
            raise AcceptanceError(f"JSONL input {self.path.name} was truncated")
        records: list[dict[str, Any]] = []
        while self._offset < metadata.st_size:
            part = os.pread(self._fd, min(65536, metadata.st_size - self._offset), self._offset)
            if not part:
                break
            self._offset += len(part)
            self._buffer += part
            if len(self._buffer) > MAX_JSON_BYTES and b"\n" not in self._buffer:
                raise AcceptanceError(f"JSONL input {self.path.name} contains an oversized line")
            while b"\n" in self._buffer:
                raw, self._buffer = self._buffer.split(b"\n", 1)
                if not raw.strip():
                    continue
                if len(raw) > MAX_JSON_BYTES:
                    raise AcceptanceError(f"JSONL input {self.path.name} contains an oversized line")
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AcceptanceError(f"JSONL input {self.path.name} contains invalid JSON") from exc
                if not isinstance(value, dict):
                    raise AcceptanceError(f"JSONL input {self.path.name} record is not an object")
                records.append(value)
        return records

    def close(self) -> None:
        if not self.closed:
            os.close(self._fd)
            self.closed = True


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    gateway_base: str = DEFAULT_GATEWAY_BASE
    startup_timeout_s: float = 30.0
    progress_deadline_s: float = PROGRESS_DEADLINE_S
    run_deadline_s: float = RUN_DEADLINE_S
    file_stall_s: float = FILE_STALL_S
    settle_s: float = SETTLE_S
    poll_interval_s: float = POLL_INTERVAL_S

    def __post_init__(self) -> None:
        parsed = urlsplit(self.gateway_base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("gateway base must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("gateway base must not contain credentials, query, or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("gateway base must not contain a path")
        if self.gateway_base.rstrip("/") != DEFAULT_GATEWAY_BASE:
            raise ValueError("production gateway base is fixed to http://127.0.0.1:8094")
        for name in (
            "startup_timeout_s",
            "progress_deadline_s",
            "run_deadline_s",
            "file_stall_s",
            "settle_s",
            "poll_interval_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.progress_deadline_s >= self.run_deadline_s:
            raise ValueError("progress deadline must precede run deadline")

    @property
    def base(self) -> str:
        return self.gateway_base.rstrip("/")


@dataclass(frozen=True, slots=True)
class Snapshot:
    plant: Mapping[str, Any]
    sonic: Mapping[str, Any]
    robot_agent: Mapping[str, Any]
    gateway: Mapping[str, Any]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"{name} is missing or not an object")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcceptanceError(f"{name} is not a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AcceptanceError(f"{name} is not a finite number")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AcceptanceError(f"{name} is not an integer >= {minimum}")
    return value


def _exact_scalar(value: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return value is expected
    if isinstance(expected, int):
        return isinstance(value, int) and not isinstance(value, bool) and value == expected
    if isinstance(expected, float):
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) == expected
        )
    return type(value) is type(expected) and value == expected


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise AcceptanceError(f"{name} timestamp is missing")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AcceptanceError(f"{name} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AcceptanceError(f"{name} timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _rpy(pose: Mapping[str, Any]) -> tuple[float, float, float]:
    qw, qx, qy, qz = (_finite(pose.get(name), f"plant.pose.{name}") for name in ("qw", "qx", "qy", "qz"))
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1e-9:
        raise AcceptanceError("plant pose quaternion is zero")
    qw, qx, qy, qz = (item / norm for item in (qw, qx, qy, qz))
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def _motion_payloads(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
    plant = record.get("plant")
    sonic = record.get("sonic")
    if isinstance(plant, Mapping) and isinstance(sonic, Mapping):
        return plant, sonic
    payloads = record.get("payloads")
    if isinstance(payloads, Mapping):
        plant = payloads.get("plant")
        sonic = payloads.get("sonic")
        if isinstance(plant, Mapping) and isinstance(sonic, Mapping):
            return plant, sonic
    return None


def _motion_derived(record: Mapping[str, Any]) -> Mapping[str, Any]:
    derived = record.get("derived")
    if not isinstance(derived, Mapping):
        return {}
    nested = derived.get("motion")
    return nested if isinstance(nested, Mapping) else derived


def _safe_hold_plant_sonic(plant: Mapping[str, Any], sonic: Mapping[str, Any]) -> bool:
    try:
        support = _mapping(plant.get("support"), "plant.support")
        velocity = _mapping(plant.get("rootVelocity"), "plant.rootVelocity")
        support_epoch = _integer(support.get("safetyStopEpoch"), "plant.support.safetyStopEpoch", minimum=1)
        stop_generation = _integer(sonic.get("plant_support_stop_generation"), "sonic.plant_support_stop_generation")
        last_rearm = support.get("lastRearmGeneration")
        if last_rearm is not None:
            last_rearm = _integer(last_rearm, "plant.support.lastRearmGeneration")
        command_values = tuple(
            _finite(sonic.get(name), f"sonic.{name}")
            for name in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz", "sonic_speed_mps", "sonic_policy_speed_mps")
        )
        return bool(
            plant.get("status") == "running"
            and sonic.get("status") == "running"
            and plant.get("robotId") == ROBOT_ID
            and sonic.get("robot_id") == ROBOT_ID
            and sonic.get("motion_ready") is True
            and sonic.get("control_clock_status") == "simulation_time_50hz_motion_ready"
            and support.get("mode") == "safety_stop"
            and support.get("safetyStopLatched") is True
            and support.get("fault") in {None, ""}
            and sonic.get("nav2_state") == "safety_stop"
            and sonic.get("sonic_mode") == "IDLE"
            and sonic.get("navigation_task_active") is False
            and sonic.get("static_hold_active") is True
            and sonic.get("motion_handoff_active") is False
            and _finite(sonic.get("motion_handoff_ratio"), "sonic.motion_handoff_ratio") == 0.0
            and all(item == 0.0 for item in command_values)
            and 0.0 <= _finite(velocity.get("linearSpeedMps"), "plant.rootVelocity.linearSpeedMps") <= 0.01
            and 0.0 <= _finite(velocity.get("angularSpeedRps"), "plant.rootVelocity.angularSpeedRps") <= 0.01
            and sonic.get("plant_support_mode") == "safety_stop"
            and sonic.get("plant_support_requested_mode") == "safety_stop"
            and sonic.get("plant_support_confirmed_mode") == "safety_stop"
            and sonic.get("plant_support_stop_pending") is False
            and sonic.get("plant_support_safety_latched") is True
            and support_epoch >= 1
            and (last_rearm is None or last_rearm <= stop_generation)
            and sonic.get("sonic_planner_reference_ready") is False
            and sonic.get("sonic_planner_required_pending") is False
            and sonic.get("sonic_reference_authority_state") == "REVOKED"
            and sonic.get("sonic_reference_actuator_authority") is False
            and sonic.get("sonic_reference_authority_dds_target_sequence") is None
            and sonic.get("sonic_execution_feedback_pending") is False
            and sonic.get("fault_trace_path") in {None, ""}
            and sonic.get("fault_trace_sample_count") in {None, 0}
        )
    except AcceptanceError:
        return False


def _strict_idle_plant_sonic(plant: Mapping[str, Any], sonic: Mapping[str, Any]) -> bool:
    try:
        support = _mapping(plant.get("support"), "plant.support")
        velocity = _mapping(plant.get("rootVelocity"), "plant.rootVelocity")
        exact_zero = tuple(
            _finite(sonic.get(name), f"sonic.{name}")
            for name in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz", "sonic_speed_mps", "sonic_policy_speed_mps")
        )
        return bool(
            plant.get("status") == "running"
            and sonic.get("status") == "running"
            and plant.get("robotId") == ROBOT_ID
            and sonic.get("robot_id") == ROBOT_ID
            and sonic.get("motion_ready") is True
            and sonic.get("control_clock_status") == "simulation_time_50hz_motion_ready"
            and support.get("mode") == "idle"
            and support.get("safetyStopLatched") is False
            and support.get("fault") in {None, ""}
            and 0.0 <= _finite(velocity.get("linearSpeedMps"), "plant.rootVelocity.linearSpeedMps") <= 0.01
            and 0.0 <= _finite(velocity.get("angularSpeedRps"), "plant.rootVelocity.angularSpeedRps") <= 0.01
            and sonic.get("nav2_state") == "idle"
            and sonic.get("sonic_mode") == "IDLE"
            and sonic.get("navigation_task_active") is False
            and sonic.get("static_hold_active") is True
            and sonic.get("motion_handoff_active") is False
            and _finite(sonic.get("motion_handoff_ratio"), "sonic.motion_handoff_ratio") == 0.0
            and all(item == 0.0 for item in exact_zero)
            and sonic.get("plant_support_mode") == "idle"
            and sonic.get("plant_support_requested_mode") == "idle"
            and sonic.get("plant_support_confirmed_mode") == "idle"
            and sonic.get("plant_support_safety_latched") is False
            and sonic.get("plant_support_stop_pending") is False
            and sonic.get("sonic_planner_reference_ready") is False
            and sonic.get("sonic_planner_required_pending") is False
            and sonic.get("sonic_reference_authority_state") == "REVOKED"
            and sonic.get("sonic_reference_actuator_authority") is False
            and sonic.get("sonic_reference_authority_dds_target_sequence") is None
            and sonic.get("sonic_execution_feedback_pending") is False
            and sonic.get("fault_trace_path") in {None, ""}
            and sonic.get("fault_trace_sample_count") in {None, 0}
        )
    except AcceptanceError:
        return False


def _sonic_intent_active(sonic: Mapping[str, Any]) -> bool:
    try:
        commands = tuple(
            abs(_finite(sonic.get(name), f"sonic.{name}"))
            for name in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz", "sonic_speed_mps", "sonic_policy_speed_mps")
        )
    except AcceptanceError:
        return False
    return bool(
        sonic.get("navigation_task_active") is True
        or sonic.get("sonic_mode") != "IDLE"
        or sonic.get("nav2_state") not in {"idle", "safety_stop", "stopped", "standby"}
        or any(value > 0.0 for value in commands)
    )


def _ra_control(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(value.get("controlState"), "Robot Agent controlState")


_REQUIRED_RA_CHECKS = frozenset(
    {
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
)


def _ra_checks_ready(value: Mapping[str, Any]) -> bool:
    try:
        checks = _mapping(value.get("checks"), "Robot Agent checks")
    except AcceptanceError:
        return False
    if not _REQUIRED_RA_CHECKS.issubset(checks):
        return False
    return all(
        isinstance(checks[name], Mapping)
        and checks[name].get("required") is True
        and checks[name].get("ready") is True
        for name in _REQUIRED_RA_CHECKS
    )


def _ra_active_runtime(value: Mapping[str, Any], *, epoch: int) -> bool:
    try:
        control = _ra_control(value)
        command = _mapping(value.get("commandReadiness"), "Robot Agent commandReadiness")
        fault = _mapping(value.get("motionSafetyFault"), "Robot Agent motionSafetyFault")
        remaining = _finite(control.get("leaseRemainingMs"), "Robot Agent leaseRemainingMs")
        active_count = _integer(control.get("activeCommandCount"), "Robot Agent activeCommandCount")
        return bool(
            value.get("schemaVersion") == "robot-agent.diagnostics/1"
            and value.get("robotId") == ROBOT_ID
            and value.get("ready") is True
            and command.get("ready") is True
            and command.get("mode") == "ACTIVE"
            and control.get("state") == "ACTIVE"
            and control.get("active") is True
            and control.get("standby") is False
            and control.get("canAcquire") is False
            and control.get("authorityRevoked") is False
            and control.get("leaseValid") is True
            and remaining > 0.0
            and _integer(control.get("safeStopEpoch"), "Robot Agent safeStopEpoch", minimum=1) == epoch
            and active_count in {0, 1}
            and fault.get("ready") is True
            and fault.get("latched") is False
            and _ra_checks_ready(value)
        )
    except AcceptanceError:
        return False


def _ra_active_terminal(value: Mapping[str, Any], *, epoch: int) -> bool:
    if not _ra_active_runtime(value, epoch=epoch):
        return False
    try:
        return _integer(_ra_control(value).get("activeCommandCount"), "Robot Agent activeCommandCount") == 0
    except AcceptanceError:
        return False


def _ra_is_active(value: Mapping[str, Any]) -> bool:
    try:
        control = _ra_control(value)
    except AcceptanceError:
        return False
    return control.get("state") == "ACTIVE" or control.get("active") is True or control.get("leaseValid") is True


def _ra_safe_hold(value: Mapping[str, Any], *, epoch: int | None = None) -> bool:
    try:
        control = _ra_control(value)
        command = _mapping(value.get("commandReadiness"), "Robot Agent commandReadiness")
        fault = _mapping(value.get("motionSafetyFault"), "Robot Agent motionSafetyFault")
        actual_epoch = _integer(control.get("safeStopEpoch"), "Robot Agent safeStopEpoch", minimum=1)
        return bool(
            value.get("schemaVersion") == "robot-agent.diagnostics/1"
            and value.get("robotId") == ROBOT_ID
            and value.get("ready") is True
            and command.get("ready") is True
            and command.get("mode") == "STANDBY"
            and control.get("state") == "SAFE_HOLD"
            and control.get("active") is False
            and control.get("standby") is True
            and control.get("canAcquire") is True
            and control.get("authorityRevoked") is True
            and control.get("safeStopState") == "confirmed"
            and control.get("leaseValid") is False
            and control.get("leaseRemainingMs") is None
            and _integer(control.get("activeCommandCount"), "Robot Agent activeCommandCount") == 0
            and fault.get("ready") is True
            and fault.get("latched") is False
            and all(fault.get(field) is None for field in ("eventId", "source", "code"))
            and _ra_checks_ready(value)
            and (epoch is None or actual_epoch == epoch)
        )
    except AcceptanceError:
        return False


class SafetyTracker:
    """Reject physical, identity, fault, process, runtime, and worker changes."""

    _RESET_MARKERS = re.compile(
        r"qacc|\bnan\b|non[- ]finite|numerical instability|implicit reset|"
        r"teleport|roll(?:ed)? back|rollback|segmentation fault|core dumped",
        re.IGNORECASE,
    )
    _FAULT_REASON = re.compile(r"^[a-z][a-z0-9_]*/[A-Z][A-Z0-9_]*:")

    def __init__(self) -> None:
        self.identities: dict[str, Any] | None = None
        self.motion_identities: dict[str, Any] | None = None
        self.processes: dict[str, Any] | None = None
        self.previous_pose: tuple[float, float, float, float] | None = None
        self.motion_previous_pose: tuple[float, float, float, float] | None = None
        self.motion_plant_sequence: int | None = None
        self.baseline_pose: tuple[float, float, float] | None = None
        self.unwrapped_yaw = 0.0
        self.motion_unwrapped_yaw = 0.0

    @staticmethod
    def _identity(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AcceptanceError(f"{name} identity is missing")
        return value

    def _validate_physical(self, plant: Mapping[str, Any], sonic: Mapping[str, Any], *, track_pose: bool) -> float:
        if plant.get("status") != "running" or sonic.get("status") != "running" or sonic.get("motion_ready") is not True:
            raise AcceptanceError("Plant or SONIC is not running")
        if plant.get("robotId") != ROBOT_ID or sonic.get("robot_id") != ROBOT_ID:
            raise AcceptanceError("Plant/SONIC robot identity mismatch")
        if sonic.get("backend") != "mujoco_sonic":
            raise AcceptanceError("SONIC backend identity mismatch")
        pose = _mapping(plant.get("pose"), "plant.pose")
        x = _finite(pose.get("x"), "plant.pose.x")
        y = _finite(pose.get("y"), "plant.pose.y")
        z = _finite(pose.get("z"), "plant.pose.z")
        roll, pitch, yaw = _rpy(pose)
        if not 0.55 <= z <= 1.15:
            raise AcceptanceError("root-height hard threshold exceeded")
        if abs(roll) > 0.25 or abs(pitch) > 0.25:
            raise AcceptanceError("roll/pitch hard threshold exceeded")
        if track_pose and self.previous_pose is not None:
            px, py, pz, pyaw = self.previous_pose
            if math.dist((x, y, z), (px, py, pz)) > 0.08:
                raise AcceptanceError("position-jump hard threshold exceeded")
            yaw_jump = abs((yaw - pyaw + math.pi) % (2 * math.pi) - math.pi)
            if yaw_jump > 0.20:
                raise AcceptanceError("yaw-jump hard threshold exceeded")
            self.unwrapped_yaw += (yaw - pyaw + math.pi) % (2 * math.pi) - math.pi
            if abs(self.unwrapped_yaw) > 0.20:
                raise AcceptanceError("heading hard threshold exceeded")
        if track_pose:
            self.previous_pose = (x, y, z, yaw)
            if self.baseline_pose is None:
                self.baseline_pose = (x, y, yaw)
        support = _mapping(plant.get("support"), "plant.support")
        velocity = _mapping(plant.get("rootVelocity"), "plant.rootVelocity")
        _finite(velocity.get("linearSpeedMps"), "plant.rootVelocity.linearSpeedMps")
        _finite(velocity.get("angularSpeedRps"), "plant.rootVelocity.angularSpeedRps")
        for field in ("last_cmd_vx", "last_cmd_vy", "last_cmd_wz", "sonic_speed_mps", "sonic_policy_speed_mps"):
            _finite(sonic.get(field), f"sonic.{field}")
        mode = support.get("mode")
        if mode not in {"safety_stop", "idle", "locomotion"}:
            raise AcceptanceError("Plant support mode is invalid")
        if (mode == "safety_stop") != (support.get("safetyStopLatched") is True):
            raise AcceptanceError("Plant support latch contradicts its mode")
        if support.get("fault") not in {None, ""}:
            raise AcceptanceError("Plant fault is latched")
        if sonic.get("fault_trace_path") not in {None, ""} or sonic.get("fault_trace_sample_count") not in {None, 0}:
            raise AcceptanceError("SONIC terminal fault trace is present")
        last_error = sonic.get("last_error")
        if last_error not in {None, ""}:
            text = str(last_error)
            # A non-structured human reason is permitted only in a proved hold.
            if self._FAULT_REASON.match(text.strip()) or not _safe_hold_plant_sonic(plant, sonic):
                raise AcceptanceError("SONIC reports a motion error")
            if self._RESET_MARKERS.search(text):
                raise AcceptanceError("SONIC reports reset/numerical fault evidence")
        arm_limits = {
            "navigation_arm_constrained_joints": 0.0,
            "navigation_arm_max_step_rad": 0.12,
            "navigation_arm_raw_max_deviation_rad": 1.25,
            "navigation_arm_raw_max_error_from_measured_rad": 0.50,
            "navigation_arm_measured_max_deviation_rad": 1.0,
            "navigation_arm_max_velocity_rps": 2.0,
            "navigation_arm_max_estimated_torque_nm": 15.0,
            "navigation_task_peak_arm_constrained_joints": 0.0,
            "navigation_task_peak_arm_raw_deviation_rad": 1.25,
            "navigation_task_peak_arm_tracking_error_rad": 0.50,
            "navigation_task_peak_arm_measured_deviation_rad": 1.0,
            "navigation_task_peak_arm_velocity_rps": 2.0,
            "navigation_task_peak_arm_estimated_torque_nm": 15.0,
        }
        for field, limit in arm_limits.items():
            arm_value = _finite(sonic.get(field), f"sonic.{field}")
            if arm_value < 0.0 or arm_value > limit:
                raise AcceptanceError(f"{field} hard threshold exceeded")
        if self.baseline_pose is None:
            return 0.0
        bx, by, byaw = self.baseline_pose
        dx, dy = x - bx, y - by
        forward = dx * math.cos(byaw) + dy * math.sin(byaw)
        lateral = -dx * math.sin(byaw) + dy * math.cos(byaw)
        if forward < -0.03 or forward > 0.30 or abs(lateral) > 0.12:
            raise AcceptanceError("directional-travel hard threshold exceeded")
        return forward

    def observe_snapshot(self, snapshot: Snapshot) -> float:
        if snapshot.gateway.get("status") != "ok" or snapshot.gateway.get("protocol") != "robot-agent.v1":
            raise AcceptanceError("Gateway health is invalid")
        plant = snapshot.plant
        sonic = snapshot.sonic
        robot_agent = snapshot.robot_agent
        gateway_agent = _mapping(snapshot.gateway.get("robotAgent"), "Gateway robotAgent health")
        if robot_agent.get("schemaVersion") != "robot-agent.diagnostics/1":
            raise AcceptanceError("Robot Agent diagnostics schema is invalid")
        if robot_agent.get("robotId") != ROBOT_ID:
            raise AcceptanceError("Robot Agent robot identity mismatch")
        control = _ra_control(robot_agent)
        gateway_observation = _mapping(gateway_agent.get("observation"), "Gateway observation health")
        gateway_epoch = _integer(gateway_agent.get("safeStopEpoch"), "Gateway safeStopEpoch", minimum=1)
        direct_epoch = _integer(control.get("safeStopEpoch"), "Robot Agent safeStopEpoch", minimum=1)
        if (
            gateway_agent.get("connected") is not True
            or gateway_agent.get("robotId") != ROBOT_ID
            or gateway_agent.get("robotBootId") != robot_agent.get("robotBootId")
            or gateway_agent.get("controlState") != control.get("state")
            or gateway_agent.get("authorityRevoked") is not control.get("authorityRevoked")
            or gateway_agent.get("safeStopState") != control.get("safeStopState")
            or gateway_epoch != direct_epoch
            or gateway_agent.get("lastError") is not None
            or gateway_observation.get("status") != "USABLE"
            or gateway_observation.get("localizationEpochId") != plant.get("localizationEpochId")
        ):
            raise AcceptanceError("Gateway and Robot Agent health identities disagree")
        gateway_observation_session = self._identity(
            gateway_observation.get("sessionId"),
            "Gateway observation session",
        )
        gateway_observation_sensor = self._identity(
            gateway_observation.get("sensorId"),
            "Gateway observation sensor",
        )
        _integer(gateway_observation.get("sequence"), "Gateway observation sequence", minimum=0)
        for field in ("receivedAgeMs", "ackAgeMs", "captureAgeMs"):
            age_ms = _finite(gateway_observation.get(field), f"Gateway observation {field}")
            if age_ms < 0.0 or age_ms > 500.0:
                raise AcceptanceError("Gateway health observation is stale")
        fault = _mapping(robot_agent.get("motionSafetyFault"), "Robot Agent motionSafetyFault")
        if fault.get("latched") is not False or fault.get("ready") is not True:
            raise AcceptanceError("Robot Agent motion-safety fault is not clear")
        if any(fault.get(field) is not None for field in ("eventId", "source", "code")):
            raise AcceptanceError("Robot Agent motion-safety fault identity is present")
        if not _ra_checks_ready(robot_agent):
            raise AcceptanceError("Robot Agent required diagnostics check is not ready")
        fault_counters = _mapping(fault.get("counters"), "Robot Agent motionSafetyFault counters")
        normalized_fault_counters: tuple[tuple[str, int], ...] = tuple(
            sorted(
                (
                    str(name),
                    _integer(counter, f"Robot Agent motion fault counter {name}"),
                )
                for name, counter in fault_counters.items()
            )
        )
        observation_worker = _mapping(plant.get("observationWorker"), "Plant observationWorker")
        viewer_worker = _mapping(plant.get("viewerWorker"), "Plant viewerWorker")
        if (
            observation_worker.get("status") != "running"
            or observation_worker.get("ready") is not True
            or observation_worker.get("alive") is not True
            or observation_worker.get("exitCode") is not None
            or _integer(observation_worker.get("failureCount"), "observation worker failureCount") != 0
            or _integer(
                observation_worker.get("collectorFailureCount"),
                "observation worker collectorFailureCount",
            )
            != 0
        ):
            raise AcceptanceError("Plant observation worker health is invalid")
        viewer_failure_count = viewer_worker.get("failureCount", 0)
        if (
            viewer_worker.get("status") != "disabled"
            or viewer_worker.get("authorityFeedbackEnabled") is not False
            or viewer_worker.get("keyCallbackRegistered") is not False
            or (
                viewer_worker.get("ready") is not None
                and viewer_worker.get("ready") is not False
            )
            or (
                viewer_worker.get("alive") is not None
                and viewer_worker.get("alive") is not False
            )
            or viewer_worker.get("exitCode") is not None
            or viewer_worker.get("workerPid") is not None
            or viewer_worker.get("workerBootId") is not None
            or _integer(viewer_failure_count, "viewer worker failureCount") != 0
            or viewer_worker.get("lastFailureAtUnixNs") is not None
            or viewer_worker.get("lastError") is not None
        ):
            raise AcceptanceError("Plant viewer must remain canonically disabled")
        identities: dict[str, Any] = {
            "plant.robotId": plant.get("robotId"),
            "plant.sourceId": self._identity(plant.get("sourceId"), "Plant source"),
            "plant.producerBootId": self._identity(plant.get("producerBootId"), "Plant boot"),
            "plant.localizationEpochId": self._identity(plant.get("localizationEpochId"), "Plant localization epoch"),
            "sonic.robot_id": sonic.get("robot_id"),
            "robotAgent.robotId": robot_agent.get("robotId"),
            "robotAgent.robotBootId": self._identity(robot_agent.get("robotBootId"), "Robot Agent boot"),
            "robotAgent.motionFaultCounters": normalized_fault_counters,
            "gateway.robotBootId": self._identity(gateway_agent.get("robotBootId"), "Gateway Robot Agent boot"),
            "gateway.observationSessionId": gateway_observation_session,
            "gateway.observationSensorId": gateway_observation_sensor,
            "plant.observationWorkerPid": _integer(
                observation_worker.get("workerPid"), "observation worker pid", minimum=1
            ),
            "plant.observationWorkerBootId": self._identity(
                observation_worker.get("workerBootId"), "observation worker boot"
            ),
            "plant.viewerWorkerMode": "disabled",
            "plant.viewerWorkerPid": None,
            "plant.viewerWorkerBootId": None,
        }
        if "driver_boot_id" in sonic:
            identities["sonic.driver_boot_id"] = self._identity(sonic.get("driver_boot_id"), "SONIC boot")
        if self.identities is None:
            self.identities = identities
        elif identities != self.identities:
            raise AcceptanceError("runtime source identity changed")
        if self.motion_identities is not None:
            for name, value in self.motion_identities.items():
                if name in identities and identities[name] != value:
                    raise AcceptanceError("motion JSONL and live source identities differ")
        return self._validate_physical(plant, sonic, track_pose=True)

    def observe_motion_record(self, record: Mapping[str, Any]) -> None:
        if record.get("errors"):
            raise AcceptanceError("motion JSONL reports an endpoint error")
        derived = _motion_derived(record)
        if derived.get("error"):
            raise AcceptanceError("motion JSONL derived sample reports an error")
        payloads = _motion_payloads(record)
        if payloads is not None:
            plant, sonic = payloads
            self._validate_physical(plant, sonic, track_pose=False)
            plant_sequence = _integer(plant.get("sequence"), "motion Plant sequence", minimum=0)
            if self.motion_plant_sequence is not None and plant_sequence <= self.motion_plant_sequence:
                raise AcceptanceError("motion JSONL Plant sequence did not strictly advance")
            self.motion_plant_sequence = plant_sequence
            identities = {
                "plant.robotId": plant.get("robotId"),
                "plant.sourceId": self._identity(plant.get("sourceId"), "motion Plant source"),
                "plant.producerBootId": self._identity(plant.get("producerBootId"), "motion Plant boot"),
                "plant.localizationEpochId": self._identity(plant.get("localizationEpochId"), "motion Plant localization epoch"),
                "sonic.robot_id": sonic.get("robot_id"),
            }
            if "driver_boot_id" in sonic:
                identities["sonic.driver_boot_id"] = self._identity(sonic.get("driver_boot_id"), "motion SONIC boot")
            if self.motion_identities is None:
                self.motion_identities = identities
            elif identities != self.motion_identities:
                raise AcceptanceError("motion JSONL source identity changed")
            pose = _mapping(plant.get("pose"), "motion plant.pose")
            x, y, z = (_finite(pose.get(name), f"motion plant.pose.{name}") for name in ("x", "y", "z"))
            yaw = _rpy(pose)[2]
            if self.motion_previous_pose is not None:
                px, py, pz, pyaw = self.motion_previous_pose
                if math.dist((x, y, z), (px, py, pz)) > 0.08:
                    raise AcceptanceError("motion JSONL position-jump threshold exceeded")
                yaw_step = (yaw - pyaw + math.pi) % (2 * math.pi) - math.pi
                if abs(yaw_step) > 0.20:
                    raise AcceptanceError("motion JSONL yaw-jump threshold exceeded")
                self.motion_unwrapped_yaw += yaw_step
                if abs(self.motion_unwrapped_yaw) > 0.20:
                    raise AcceptanceError("motion JSONL heading threshold exceeded")
            self.motion_previous_pose = (x, y, z, yaw)
        failures = record.get("failures")
        if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)) and failures:
            raise AcceptanceError("motion JSONL reports a hard acceptance failure")

    def observe_sidecar_record(self, record: Mapping[str, Any]) -> None:
        if record.get("errors"):
            raise AcceptanceError("sidecar JSONL reports an endpoint error")
        endpoint_errors = record.get("endpointErrors")
        if isinstance(endpoint_errors, Mapping) and endpoint_errors:
            raise AcceptanceError("sidecar endpoint failure")
        process_errors = record.get("processErrors")
        if isinstance(process_errors, Sequence) and not isinstance(process_errors, (str, bytes)) and process_errors:
            raise AcceptanceError("sidecar process identity is unavailable")
        processes = record.get("processes")
        if isinstance(processes, Mapping):
            if not _CORE_PROCESS_NAMES.issubset(processes):
                raise AcceptanceError("sidecar core process baseline is incomplete")
            normalized: dict[str, dict[str, int]] = {}
            for name in sorted(_CORE_PROCESS_NAMES):
                process = _mapping(processes.get(name), f"sidecar process {name}")
                normalized[name] = {
                    "pid": _integer(process.get("pid"), f"sidecar process {name} pid", minimum=1),
                    "startTimeTicks": _integer(
                        process.get("startTimeTicks"),
                        f"sidecar process {name} startTimeTicks",
                        minimum=1,
                    ),
                }
            if self.processes is None:
                self.processes = normalized
            elif normalized != self.processes:
                raise AcceptanceError("sidecar process identity changed")
        log_scan = record.get("logScan")
        if isinstance(log_scan, Mapping):
            for key in (
                "hard_matches",
                "hardMatches",
                "stale_matches",
                "staleMatches",
                "transport_timeout_matches",
                "transportTimeoutMatches",
                "transport_timeout_counter_rollbacks",
                "transportTimeoutCounterRollbacks",
                "errors",
            ):
                value = log_scan.get(key)
                if isinstance(value, (list, tuple, dict)) and value:
                    raise AcceptanceError("sidecar runtime log fault")
        derived = record.get("derived")
        if not isinstance(derived, Mapping):
            return
        if derived.get("processIdentityChanged") is True:
            raise AcceptanceError("sidecar process restart event")
        runtime = derived.get("runtime")
        if isinstance(runtime, Mapping):
            if runtime.get("stallEvent") is True or runtime.get("counterRollbacks"):
                raise AcceptanceError("sidecar runtime stall event")
            deltas = runtime.get("deltasFromBaseline")
            if isinstance(deltas, Mapping) and any(value != 0 for value in deltas.values()):
                raise AcceptanceError("sidecar runtime stall counter changed")
        workers = derived.get("plantWorkers")
        if isinstance(workers, Mapping):
            for key in ("healthErrors", "identityErrors", "identityChanges", "failureChanges", "viewerProgressErrors"):
                if workers.get(key):
                    raise AcceptanceError("sidecar worker fault event")
            timing = workers.get("renderTiming")
            if isinstance(timing, Mapping):
                if timing.get("stallEvent") is True or timing.get("counterRollbacks"):
                    raise AcceptanceError("sidecar worker render stall event")
                deltas = timing.get("deltasFromBaseline")
                if isinstance(deltas, Mapping) and any(value != 0 for value in deltas.values()):
                    raise AcceptanceError("sidecar worker render counter changed")
        failures = record.get("failures")
        if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)) and failures:
            raise AcceptanceError("sidecar reports a hard failure")


class Forward02mRunner:
    """Single-use state machine; construct a new instance for another run."""

    def __init__(
        self,
        *,
        config: RunnerConfig,
        token: str,
        transport: Transport,
        motion_jsonl: JsonlSource,
        sidecar_jsonl: JsonlSource,
        evidence: Any,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        utcnow: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not token or any(ord(char) < 0x21 or ord(char) == 0x7F for char in token):
            raise ValueError("token is invalid")
        self.config = config
        self._token = token
        self.transport = transport
        self.motion_jsonl = motion_jsonl
        self.sidecar_jsonl = sidecar_jsonl
        self.evidence = evidence
        self.clock = clock
        self.sleeper = sleeper
        self.utcnow = utcnow
        self.safety = SafetyTracker()
        self.command_id: str | None = None
        self.command_post_attempted = False
        self.cancel_post_attempted = False
        self.command_get_attempted = False
        self.discover_get_attempted = False
        self._last_snapshot: Snapshot | None = None
        self._last_motion_record_s: float | None = None
        self._last_sidecar_record_s: float | None = None
        self._post_started_s: float | None = None
        # The one POST is a transaction, not a detached best-effort thread.
        # Abort proof must fence both the client worker and its observed
        # outcome before any SAFE_HOLD dwell can begin.
        self._post_completed = threading.Event()
        self._post_worker: threading.Thread | None = None
        self._post_outcome: dict[str, Any] = {}
        self._post_outcome_merged = False
        self._post_response_received = False
        self._post_dispatch_suppressed = False
        self._cancel_confirmed = False
        self._cancel_dispatch_suppressed = False
        self._motion_session_id: str | None = None
        self._sidecar_session_id: str | None = None
        self._baseline_epoch: int | None = None
        self._baseline_robot_boot_id: str | None = None
        self._abort_hold_evidence_clean = True
        self._latest_motion_healthy = False
        self._latest_motion_intent = True
        self._latest_motion_derived_s: float | None = None
        self._progress_seen = False
        self._intent_seen = False

    def _record(self, kind: str, **values: Any) -> None:
        # Deliberately curated: authorization headers and the token never enter
        # this object, even on transport failure.
        self.evidence.write(
            {
                "schemaVersion": SCHEMA_VERSION,
                "kind": kind,
                "observedMonotonicS": self.clock(),
                **values,
            }
        )

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _authorize_transport_call(
        self,
        method: str,
        url: str,
        *,
        authenticated: bool,
        body: Mapping[str, Any] | None,
    ) -> None:
        """Enforce the complete network contract at the transport boundary."""

        readonly = {PLANT_URL, SONIC_URL, ROBOT_AGENT_URL, f"{self.config.base}/healthz"}
        discover = f"{self.config.base}/internal/v1/discover?{urlencode({'robotId': ROBOT_ID})}"
        command_collection = f"{self.config.base}/internal/v1/commands"
        command_item = (
            None
            if self.command_id is None
            else f"{command_collection}/{quote(self.command_id, safe='')}"
        )
        allowed = False
        if method == "GET" and url in readonly:
            allowed = not authenticated and body is None
        elif method == "GET" and url == discover:
            allowed = authenticated and body is None
        elif method == "POST" and url == command_collection:
            request = body.get("request") if isinstance(body, Mapping) else None
            allowed = bool(
                authenticated
                and isinstance(request, Mapping)
                and self.command_id is not None
                and request.get("commandId") == self.command_id
            )
        elif command_item is not None and method == "GET" and url == command_item:
            allowed = authenticated and body is None
        elif command_item is not None and method == "POST" and url == f"{command_item}/cancel":
            allowed = authenticated and body is None
        if not allowed:
            raise AcceptanceError("transport request is outside the fixed method/URL/auth/body allowlist")

    def _json_response(self, method: str, url: str, *, authenticated: bool = False, body: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        self._authorize_transport_call(
            method,
            url,
            authenticated=authenticated,
            body=body,
        )
        path = urlsplit(url).path
        if method == "POST" and path.endswith("/cancel"):
            timeout_s = 8.0
        elif method == "POST" and path == "/internal/v1/commands":
            timeout_s = 10.0
        elif method == "GET" and path.startswith("/internal/v1/commands/"):
            timeout_s = 8.0
        else:
            timeout_s = 1.0
        response = self.transport.request(
            method,
            url,
            headers=self._auth() if authenticated else None,
            json_body=body,
            timeout_s=timeout_s,
        )
        if response.status != 200 or not isinstance(response.payload, Mapping):
            raise AcceptanceError(f"{method} {urlsplit(url).path} did not return HTTP 200 JSON")
        return response.payload

    def _snapshot(self) -> Snapshot:
        snapshot = Snapshot(
            plant=self._json_response("GET", PLANT_URL),
            sonic=self._json_response("GET", SONIC_URL),
            robot_agent=self._json_response("GET", ROBOT_AGENT_URL),
            gateway=self._json_response("GET", f"{self.config.base}/healthz"),
        )
        self._last_snapshot = snapshot
        return snapshot

    def _validate_stream_records(self, records: Sequence[Mapping[str, Any]], *, motion: bool) -> None:
        session_attr = "_motion_session_id" if motion else "_sidecar_session_id"
        session_id = getattr(self, session_attr)
        label = "motion" if motion else "sidecar"
        expected_schema = MOTION_MONITOR_SCHEMA if motion else SIDECAR_MONITOR_SCHEMA
        allowed_kinds = _MOTION_RECORD_KINDS if motion else _SIDECAR_RECORD_KINDS
        now = self.utcnow().astimezone(UTC)
        for record in records:
            kind = record.get("kind")
            if record.get("schemaVersion") != expected_schema:
                raise AcceptanceError(f"{label} JSONL schema is unknown")
            if kind not in allowed_kinds:
                raise AcceptanceError(f"{label} JSONL record kind is unknown")
            record_session = record.get("sessionId")
            if kind == "session_start":
                if session_id is not None:
                    raise AcceptanceError(f"{label} JSONL contains multiple session starts")
                if not isinstance(record_session, str) or not record_session:
                    raise AcceptanceError(f"{label} JSONL session identity is missing")
                if (
                    record.get("readOnly") is not True
                    or record.get("networkMethods") != ["GET"]
                    or not isinstance(record.get("limits"), Mapping)
                ):
                    raise AcceptanceError(f"{label} JSONL session is not a frozen read-only monitor")
                limits = _mapping(record.get("limits"), f"{label} JSONL limits")
                if motion:
                    if record.get("plantUrl") != PLANT_URL or record.get("sonicUrl") != SONIC_URL:
                        raise AcceptanceError("motion JSONL session endpoint contract is invalid")
                    if any(
                        key not in limits or not _exact_scalar(limits.get(key), expected)
                        for key, expected in _EXPECTED_MOTION_LIMITS.items()
                    ):
                        raise AcceptanceError("motion JSONL limits do not match the 0.2 m acceptance contract")
                else:
                    if record.get("networkAllowlist") != sorted({PLANT_URL, SONIC_URL, ROBOT_AGENT_URL}):
                        raise AcceptanceError("sidecar JSONL session endpoint allowlist is invalid")
                    duration_s = _finite(record.get("durationS"), "sidecar durationS")
                    if (
                        duration_s < 65.0
                        or not _exact_scalar(record.get("plantSonicIntervalS"), 0.5)
                        or not _exact_scalar(record.get("robotAgentIntervalS"), 5.0)
                        or not _exact_scalar(record.get("rawCheckpointIntervalS"), 10.0)
                    ):
                        raise AcceptanceError("sidecar JSONL timing contract is too weak")
                session_id = record_session
                setattr(self, session_attr, session_id)
            else:
                if session_id is None or record_session != session_id:
                    raise AcceptanceError(f"{label} JSONL record does not belong to the current session")
            if kind == "result" or record.get("interrupted") is True:
                raise AcceptanceError(f"{label} JSONL monitor is already finished/interrupted")
            if record_session is not None:
                recorded = _timestamp(record.get("recordedAt"), f"{label} recordedAt")
                age = (now - recorded).total_seconds()
                if age > 5.0 or age < -1.0:
                    raise AcceptanceError(f"{label} JSONL record is stale or future-dated")

    def _consume_jsonl(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        now = self.clock()
        motion = self.motion_jsonl.read_new()
        sidecar = self.sidecar_jsonl.read_new()
        self._validate_stream_records(motion, motion=True)
        self._validate_stream_records(sidecar, motion=False)
        if motion:
            self._last_motion_record_s = now
            for record in motion:
                derived = _motion_derived(record)
                if derived:
                    self._latest_motion_healthy = derived.get("healthy_settled_safe_hold") is True
                    self._latest_motion_intent = derived.get("intent_active") is not False
                    self._latest_motion_derived_s = now
                    if self._post_started_s is not None and derived.get("intent_active") is True:
                        self._intent_seen = True
        if sidecar:
            self._last_sidecar_record_s = now
        for record in motion:
            self.safety.observe_motion_record(record)
        for record in sidecar:
            self.safety.observe_sidecar_record(record)
        return motion, sidecar

    def _wait_for_first_records(self) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        started = self.clock()
        first_motion: Mapping[str, Any] | None = None
        first_sidecar: Mapping[str, Any] | None = None
        first_plant: Mapping[str, Any] | None = None
        baseline_batch_complete = False
        continued_motion = False
        continued_sidecar = False
        raw_checkpoint_seen = False
        while self.clock() - started < self.config.startup_timeout_s:
            motion, sidecar = self._consume_jsonl()
            if baseline_batch_complete:
                continued_motion = continued_motion or any(
                    record.get("kind") == "sample"
                    and bool(_motion_derived(record))
                    for record in motion
                )
                continued_sidecar = continued_sidecar or any(
                    record.get("kind") in {"sample", "raw_checkpoint"}
                    and isinstance(record.get("derived"), Mapping)
                    for record in sidecar
                )
                raw_checkpoint_seen = raw_checkpoint_seen or any(
                    record.get("kind") == "raw_checkpoint"
                    and isinstance(record.get("derived"), Mapping)
                    and isinstance(record.get("processes"), Mapping)
                    for record in sidecar
                )
            if first_motion is None:
                for record in motion:
                    payloads = _motion_payloads(record)
                    if payloads is None or not _motion_derived(record):
                        continue
                    first_motion = record
                    first_plant = payloads[0]
                    derived = _motion_derived(record)
                    if derived.get("healthy_settled_safe_hold") is not True or derived.get("intent_active") is not False:
                        raise AcceptanceError("motion first sample is not inactive healthy settled SAFE_HOLD")
                    break
            if first_sidecar is None:
                first_sidecar = next(
                    (
                        record
                        for record in sidecar
                        if record.get("kind") == "baseline"
                        and isinstance(record.get("derived"), Mapping)
                        and isinstance(record.get("processes"), Mapping)
                    ),
                    None,
                )
            if (
                first_motion is not None
                and first_sidecar is not None
                and first_plant is not None
                and baseline_batch_complete
                and continued_motion
                and continued_sidecar
                and raw_checkpoint_seen
            ):
                self._record("preflight_jsonl_ready")
                return first_motion, first_sidecar, first_plant
            if first_motion is not None and first_sidecar is not None and first_plant is not None:
                if not baseline_batch_complete:
                    baseline_batch_complete = True
                    # Drain once more immediately: an already-closed replay
                    # has no continuation, while live fake/file sources can
                    # prove a post-start sample without shifting command time.
                    continue
            self.sleeper(self.config.poll_interval_s)
        raise AcceptanceError("timed out waiting for the first motion and sidecar samples")

    def _discover(self, snapshot: Snapshot) -> Mapping[str, Any]:
        if self.discover_get_attempted:
            raise AcceptanceError("discover GET exactly-once latch is already consumed")
        self.discover_get_attempted = True
        url = f"{self.config.base}/internal/v1/discover?{urlencode({'robotId': ROBOT_ID})}"
        envelope = self._json_response("GET", url, authenticated=True)
        result = _mapping(envelope.get("result"), "Gateway discover result")
        capabilities = result.get("capabilities")
        skill_ids_are_valid = bool(
            isinstance(capabilities, list)
            and all(
                isinstance(capability, str)
                and capability
                and all(ord(char) >= 0x21 and ord(char) != 0x7F for char in capability)
                for capability in capabilities
            )
        )
        if (
            envelope.get("status") != "available"
            or result.get("available") is not True
            or result.get("protocol") != "robot-agent.v1"
            or result.get("robotId") != ROBOT_ID
            or not skill_ids_are_valid
            or len(capabilities) != len(set(capabilities))
            or COMMAND_KEY not in capabilities
        ):
            raise AcceptanceError("Gateway discover did not return one usable unique robot contract")
        observation = _mapping(result.get("observation"), "Gateway discovery observation")
        pose = _mapping(result.get("currentPose"), "Gateway discovery currentPose")
        if observation.get("status") != "USABLE" or pose.get("frameId") != "map":
            raise AcceptanceError("Gateway discovery observation/pose is not usable")
        if (
            self.safety.identities is None
            or observation.get("sessionId") != self.safety.identities.get("gateway.observationSessionId")
        ):
            raise AcceptanceError("Gateway discovery observation session changed from health preflight")
        for field in ("receivedAgeMs", "ackAgeMs", "captureAgeMs"):
            age_ms = _finite(observation.get(field), f"Gateway observation {field}")
            if age_ms < 0.0 or age_ms > 500.0:
                raise AcceptanceError("Gateway discovery observation is stale")
        if result.get("robotBootId") != snapshot.robot_agent.get("robotBootId"):
            raise AcceptanceError("Gateway and Robot Agent boot identities differ")
        if observation.get("localizationEpochId") != snapshot.plant.get("localizationEpochId"):
            raise AcceptanceError("Gateway and Plant localization epochs differ")
        observed_at = _timestamp(pose.get("observedAt"), "Gateway currentPose observedAt")
        pose_age_s = (self.utcnow().astimezone(UTC) - observed_at).total_seconds()
        if pose_age_s > 1.0 or pose_age_s < -0.2:
            raise AcceptanceError("Gateway currentPose is stale or future-dated")
        plant_pose = _mapping(snapshot.plant.get("pose"), "Plant pre-command pose")
        plant_x = _finite(plant_pose.get("x"), "Plant pre-command x")
        plant_y = _finite(plant_pose.get("y"), "Plant pre-command y")
        plant_yaw = _rpy(plant_pose)[2]
        gateway_x = _finite(pose.get("x"), "Gateway currentPose x")
        gateway_y = _finite(pose.get("y"), "Gateway currentPose y")
        gateway_yaw = _finite(pose.get("yaw"), "Gateway currentPose yaw")
        yaw_delta = abs((gateway_yaw - plant_yaw + math.pi) % (2 * math.pi) - math.pi)
        if math.hypot(gateway_x - plant_x, gateway_y - plant_y) > 0.03 or yaw_delta > 0.10:
            raise AcceptanceError("Gateway currentPose disagrees with Plant baseline")
        self._record("discover_validated", robotId=ROBOT_ID)
        return result

    def _build_command(self, discovery: Mapping[str, Any]) -> dict[str, Any]:
        pose = _mapping(discovery.get("currentPose"), "Gateway discovery currentPose")
        observation = _mapping(discovery.get("observation"), "Gateway discovery observation")
        x = _finite(pose.get("x"), "Gateway pose x")
        y = _finite(pose.get("y"), "Gateway pose y")
        yaw = _finite(pose.get("yaw"), "Gateway pose yaw")
        if not -math.pi <= yaw <= math.pi:
            raise AcceptanceError("Gateway pose yaw is outside [-pi, pi]")
        acceptance_uuid = str(uuid4())
        command_id = str(uuid4())
        issued = self.utcnow().astimezone(UTC)
        expires = issued + timedelta(seconds=30)
        self.command_id = command_id
        payload = {
            "payloadType": "relative_resolution",
            "goalPose": {
                "frameId": "map",
                "x": x + 0.2 * math.cos(yaw),
                "y": y + 0.2 * math.sin(yaw),
                "z": 0.0,
                "yaw": yaw,
                "observedAt": self.safety._identity(pose.get("observedAt"), "Gateway pose observedAt"),
            },
            "goalPurpose": "final",
            "missionId": acceptance_uuid,
            "stageId": "forward-0p2m",
            "relativeResolutionId": f"{acceptance_uuid}-forward-0p2m",
            "observationSessionId": self.safety._identity(observation.get("sessionId"), "observation session"),
            "robotBootId": self.safety._identity(discovery.get("robotBootId"), "discovered Robot Agent boot"),
            "localizationEpochId": self.safety._identity(observation.get("localizationEpochId"), "observation localization epoch"),
            "maximumSpeedMps": 0.2,
        }
        request = {
            "commandId": command_id,
            "robotId": ROBOT_ID,
            "source": "manual",
            "transport": "robot_agent",
            "commandKey": COMMAND_KEY,
            "commandVersion": "2.0.0",
            "payload": payload,
            "issuedAt": issued.isoformat().replace("+00:00", "Z"),
            "expiresAt": expires.isoformat().replace("+00:00", "Z"),
            "idempotencyKey": acceptance_uuid,
            "sessionId": acceptance_uuid,
            "traceId": acceptance_uuid,
        }
        mapping = {
            "mappingId": "acceptance-navigation-g1-sim",
            "version": "v1",
            "commandKey": COMMAND_KEY,
            "transport": "robot_agent",
            "mapping": {
                "targetId": ROBOT_ID,
                "adapterProfile": "robot-agent-gateway-simulation",
                "capabilityKey": COMMAND_KEY,
            },
        }
        return {"request": request, "mapping": mapping}

    def _validate_pending_phase(self, snapshot: Snapshot, forward: float) -> None:
        """Allow only the original hold or this command's strict ACTIVE phase."""

        if self._baseline_epoch is None:
            raise AcceptanceError("pending command has no SAFE_HOLD epoch baseline")
        active = _ra_active_runtime(snapshot.robot_agent, epoch=self._baseline_epoch)
        baseline_hold = bool(
            _ra_safe_hold(snapshot.robot_agent, epoch=self._baseline_epoch)
            and _safe_hold_plant_sonic(snapshot.plant, snapshot.sonic)
        )
        physical_intent = bool(
            _sonic_intent_active(snapshot.sonic)
            or _mapping(snapshot.plant.get("support"), "Plant support").get("mode") == "locomotion"
            or abs(forward) > 1e-6
        )
        if active:
            return
        if baseline_hold and not physical_intent:
            return
        raise AcceptanceError("pending POST has an incoherent Robot Agent command phase")

    def _is_suppressed_receipt(self, envelope: Mapping[str, Any]) -> bool:
        """Accept only the Gateway's command-bound tombstone receipt."""

        result = envelope.get("result")
        return bool(
            envelope.get("status") == "cancelled"
            and isinstance(result, Mapping)
            and result.get("robotAgentCommandId") == self.command_id
            and result.get("dispatchSuppressed") is True
        )

    def _merge_post_transaction_if_finished(self) -> bool:
        """Fence the POST worker and classify its immutable outcome once."""

        if self._post_outcome_merged:
            return True
        worker = self._post_worker
        if worker is None:
            return False
        if not self._post_completed.is_set():
            # A short bounded join yields to the client worker while the main
            # loop remains responsible for all live tails and snapshots.
            worker.join(timeout=0.01)
            if not self._post_completed.is_set():
                return False
        # completed is set in the worker's final action. Still require the
        # thread itself to terminate so no client-side POST work can outlive a
        # successful terminal proof.
        worker.join(timeout=0.01)
        if worker.is_alive():
            return False
        envelope = self._post_outcome.get("envelope")
        if isinstance(envelope, Mapping):
            result = envelope.get("result")
            self._post_response_received = bool(
                envelope.get("status")
                in {
                    "accepted",
                    "running",
                    "succeeded",
                    "failed",
                    "cancelled",
                    "expired",
                    "unknown",
                }
                and isinstance(result, Mapping)
                and result.get("robotAgentCommandId") == self.command_id
            )
            self._post_dispatch_suppressed = self._is_suppressed_receipt(envelope)
        self._post_outcome_merged = True
        return True

    def _post_command(self, body: Mapping[str, Any]) -> None:
        if self.command_post_attempted:
            raise AcceptanceError("command POST exactly-once latch is already consumed")
        self.command_post_attempted = True
        self._post_started_s = self.clock()  # Both 15 s and 25 s start before I/O.

        def issue_once() -> None:
            try:
                self._post_outcome["envelope"] = self._json_response(
                    "POST",
                    f"{self.config.base}/internal/v1/commands",
                    authenticated=True,
                    body=body,
                )
            except BaseException as exc:
                self._post_outcome["error"] = exc
            finally:
                self._post_completed.set()

        worker = threading.Thread(
            target=issue_once,
            name="forward-02m-command-post",
            daemon=True,
        )
        self._post_worker = worker
        worker.start()
        pending_failure: BaseException | None = None
        while not self._merge_post_transaction_if_finished():
            if self.clock() - self._post_started_s >= self.config.run_deadline_s:
                pending_failure = AcceptanceError("command remained unfinished for 25 seconds")
                self._cancel_once_if_active(self._last_snapshot, force_unknown=True)
                break
            try:
                snapshot, forward = self._poll()
                self._validate_pending_phase(snapshot, forward)
                if _sonic_intent_active(snapshot.sonic):
                    self._intent_seen = True
                observed_elapsed = self.clock() - self._post_started_s
                if forward >= PROGRESS_M and observed_elapsed <= self.config.progress_deadline_s:
                    self._progress_seen = True
                if (
                    not self._progress_seen
                    and observed_elapsed >= self.config.progress_deadline_s
                ):
                    raise AcceptanceError("command made less than 0.03 m progress in 15 seconds")
            except BaseException as exc:
                pending_failure = exc
                # The command UUID is already frozen. Gateway cancellation of
                # an as-yet unknown ID is side-effect free; if dispatch owns
                # the ledger lock, cancel serializes immediately behind it.
                self._cancel_once_if_active(self._last_snapshot, force_unknown=True)
                break
            self.sleeper(self.config.poll_interval_s)
        if pending_failure is not None:
            raise pending_failure
        if not self._post_outcome_merged:
            raise TransportUncertain("command POST completion is uncertain")
        error = self._post_outcome.get("error")
        if isinstance(error, BaseException):
            self._record("command_post_uncertain", commandId=self.command_id)
            raise error
        envelope = _mapping(self._post_outcome.get("envelope"), "command POST envelope")
        result = _mapping(envelope.get("result"), "command POST result")
        if envelope.get("status") != "accepted":
            raise AcceptanceError("command POST was not accepted")
        if result.get("robotAgentCommandId") != self.command_id:
            raise AcceptanceError("command POST receipt identity mismatch")
        self._record("command_post_accepted", commandId=self.command_id, status=envelope.get("status"))

    def _cancel_once_if_active(self, snapshot: Snapshot | None, *, force_unknown: bool = False) -> None:
        if self.command_id is None or self.cancel_post_attempted:
            return
        if not force_unknown and (snapshot is None or not _ra_is_active(snapshot.robot_agent)):
            return
        self.cancel_post_attempted = True
        url = f"{self.config.base}/internal/v1/commands/{quote(self.command_id, safe='')}/cancel"
        try:
            envelope = self._json_response("POST", url, authenticated=True)
        except BaseException:
            self._record("cancel_post_uncertain", commandId=self.command_id)
            return
        result = envelope.get("result")
        matching_id = bool(
            isinstance(result, Mapping)
            and result.get("robotAgentCommandId") == self.command_id
        )
        self._cancel_dispatch_suppressed = self._is_suppressed_receipt(envelope)
        self._cancel_confirmed = bool(
            envelope.get("status") == "cancelled" and matching_id
        )
        self._record(
            "cancel_post_completed",
            commandId=self.command_id,
            status=envelope.get("status"),
            cancelConfirmed=self._cancel_confirmed,
            dispatchSuppressed=self._cancel_dispatch_suppressed,
        )

    def _terminal_query(self, baseline_epoch: int) -> int:
        if self.command_get_attempted or self.command_id is None:
            raise AcceptanceError("terminal command GET exactly-once latch is unavailable")
        self.command_get_attempted = True
        url = f"{self.config.base}/internal/v1/commands/{quote(self.command_id, safe='')}"
        completed = threading.Event()
        outcome: dict[str, Any] = {}

        def query_once() -> None:
            try:
                outcome["envelope"] = self._json_response("GET", url, authenticated=True)
            except BaseException as exc:
                outcome["error"] = exc
            finally:
                completed.set()

        query_started = self.clock()
        threading.Thread(
            target=query_once,
            name="forward-02m-terminal-query",
            daemon=True,
        ).start()
        while not completed.is_set():
            if self.clock() - query_started >= 8.0:
                self._cancel_once_if_active(self._last_snapshot, force_unknown=True)
                raise TransportUncertain("terminal command GET completion timed out")
            try:
                snapshot, _forward = self._poll()
                phase_is_valid = bool(
                    _ra_active_runtime(snapshot.robot_agent, epoch=baseline_epoch)
                    or (
                        _ra_safe_hold(snapshot.robot_agent, epoch=baseline_epoch + 1)
                        and _safe_hold_plant_sonic(snapshot.plant, snapshot.sonic)
                    )
                )
                if not phase_is_valid:
                    raise AcceptanceError("terminal query entered an incoherent release phase")
            except BaseException:
                self._cancel_once_if_active(self._last_snapshot, force_unknown=True)
                raise
            self.sleeper(self.config.poll_interval_s)
        error = outcome.get("error")
        if isinstance(error, BaseException):
            raise error
        envelope = _mapping(outcome.get("envelope"), "terminal command envelope")
        result = _mapping(envelope.get("result"), "terminal command result")
        release = _mapping(result.get("controlRelease"), "terminal controlRelease")
        details = _mapping(envelope.get("details"), "terminal command details")
        details_release = _mapping(details.get("controlRelease"), "terminal details controlRelease")
        expected = {
            "state": "SAFE_HOLD",
            "authorityRevoked": True,
            "safeStopState": "confirmed",
            "safeStopEpoch": baseline_epoch + 1,
        }
        def exact_release(value: Mapping[str, Any]) -> bool:
            return bool(
                set(value) == set(expected)
                and value.get("state") == "SAFE_HOLD"
                and value.get("authorityRevoked") is True
                and value.get("safeStopState") == "confirmed"
                and isinstance(value.get("safeStopEpoch"), int)
                and not isinstance(value.get("safeStopEpoch"), bool)
                and value.get("safeStopEpoch") == baseline_epoch + 1
            )
        if (
            envelope.get("status") != "succeeded"
            or not exact_release(release)
            or not exact_release(details_release)
        ):
            raise AcceptanceError("terminal command receipt lacks exact succeeded SAFE_HOLD release truth")
        if result.get("robotAgentCommandId") != self.command_id:
            raise AcceptanceError("terminal command receipt identity mismatch")
        self._record("terminal_query_validated", commandId=self.command_id, safeStopEpoch=baseline_epoch + 1)
        return baseline_epoch + 1

    def _poll(self) -> tuple[Snapshot, float]:
        self._consume_jsonl()
        now = self.clock()
        if self._post_started_s is not None:
            if self._last_motion_record_s is None or now - self._last_motion_record_s > self.config.file_stall_s:
                raise AcceptanceError("motion JSONL file stalled")
            if self._last_sidecar_record_s is None or now - self._last_sidecar_record_s > self.config.file_stall_s:
                raise AcceptanceError("sidecar JSONL file stalled")
        snapshot = self._snapshot()
        progress = self.safety.observe_snapshot(snapshot)
        return snapshot, progress

    def _abort(self, reason: str) -> dict[str, Any]:
        snapshot = self._last_snapshot
        # Refresh authority once after an ambiguous/failing command boundary.
        # This is read-only and gives a just-activated command one chance to be
        # seen; a transport failure here never authorizes a blind cancellation.
        refresh_proved_safe = False
        refresh_failed = False
        if self.command_post_attempted:
            try:
                snapshot = self._snapshot()
                self.safety.observe_snapshot(snapshot)
                control = _ra_control(snapshot.robot_agent)
                epoch = control.get("safeStopEpoch")
                same_boot = snapshot.robot_agent.get("robotBootId") == self._baseline_robot_boot_id
                refresh_proved_safe = bool(
                    same_boot
                    and self._baseline_epoch is not None
                    and isinstance(epoch, int)
                    and not isinstance(epoch, bool)
                    and epoch > self._baseline_epoch
                    and _ra_safe_hold(snapshot.robot_agent)
                    and _safe_hold_plant_sonic(snapshot.plant, snapshot.sonic)
                )
            except BaseException:
                refresh_failed = True
                snapshot = self._last_snapshot
        if self.command_post_attempted and not refresh_proved_safe:
            # The Gateway's cancel ledger is safe for unknown IDs and is the
            # only fail-safe convergence action when POST/RA truth is unknown.
            self._cancel_once_if_active(
                snapshot,
                force_unknown=refresh_failed or snapshot is None or not _ra_is_active(snapshot.robot_agent),
            )
        terminal_hold_proved = False
        if self.command_post_attempted:
            try:
                terminal_hold_proved = self._prove_abort_hold()
            except BaseException:
                # A second interrupt cannot cause another mutation; preserve
                # the single cancel latch and report that proof was incomplete.
                terminal_hold_proved = False
        result = {
            "passed": False,
            "reason": reason[:512],
            "commandId": self.command_id,
            "discoverGetCount": int(self.discover_get_attempted),
            "commandPostCount": int(self.command_post_attempted),
            "cancelPostCount": int(self.cancel_post_attempted),
            "commandGetCount": int(self.command_get_attempted),
            "terminalSafeHoldProved": terminal_hold_proved,
            "terminalEvidenceClean": self._abort_hold_evidence_clean,
        }
        self._record("result", **result)
        return result


    def _prove_abort_hold(self) -> bool:
        """After cancellation/failure, tail inputs and prove 1 s physical hold."""

        started = self.clock()
        hold_since: float | None = None
        # A hard trajectory violation deliberately poisons the movement
        # tracker.  Terminal convergence starts a new trajectory baseline at
        # the stopped pose, while retaining every pre-command source/boot and
        # fault-counter identity exactly.
        hold_safety = SafetyTracker()
        hold_safety.identities = None if self.safety.identities is None else dict(self.safety.identities)
        hold_safety.motion_identities = (
            None if self.safety.motion_identities is None else dict(self.safety.motion_identities)
        )
        while self.clock() - started < 8.0:
            # Evaluate this before the direct snapshot. A newly merged accepted
            # response therefore forces its first hold sample after admission.
            post_fenced = self._merge_post_transaction_if_finished()
            baseline_hold_allowed = bool(
                post_fenced
                and (
                    self._post_dispatch_suppressed
                    or self._cancel_dispatch_suppressed
                )
            )
            advanced_hold_required = bool(
                post_fenced
                and not baseline_hold_allowed
                and (
                    self._post_response_received
                    or self._cancel_confirmed
                )
            )
            try:
                now = self.clock()
                motion = self.motion_jsonl.read_new()
                sidecar = self.sidecar_jsonl.read_new()
                self._validate_stream_records(motion, motion=True)
                self._validate_stream_records(sidecar, motion=False)
                for record in motion:
                    self.safety.observe_motion_record(record)
                    derived = _motion_derived(record)
                    if derived:
                        self._latest_motion_healthy = derived.get("healthy_settled_safe_hold") is True
                        self._latest_motion_intent = derived.get("intent_active") is not False
                        self._latest_motion_derived_s = now
                for record in sidecar:
                    self.safety.observe_sidecar_record(record)
                if motion:
                    self._last_motion_record_s = now
                if sidecar:
                    self._last_sidecar_record_s = now
                if (
                    self._last_motion_record_s is None
                    or now - self._last_motion_record_s > self.config.file_stall_s
                    or self._last_sidecar_record_s is None
                    or now - self._last_sidecar_record_s > self.config.file_stall_s
                ):
                    self._abort_hold_evidence_clean = False
            except BaseException:
                # The originating hard failure may be a malformed/stalled
                # evidence stream. Keep attempting both tails, but do not let
                # that prevent an independent physical SAFE_HOLD proof.
                self._abort_hold_evidence_clean = False
            try:
                snapshot = self._snapshot()
                hold_safety.observe_snapshot(snapshot)
                control = _ra_control(snapshot.robot_agent)
                epoch = control.get("safeStopEpoch")
                same_boot = snapshot.robot_agent.get("robotBootId") == self._baseline_robot_boot_id
                strict_physical_hold = bool(
                    same_boot
                    and _safe_hold_plant_sonic(snapshot.plant, snapshot.sonic)
                    and _ra_safe_hold(snapshot.robot_agent)
                )
                strict = bool(
                    strict_physical_hold
                    and (
                        baseline_hold_allowed
                        or (
                            advanced_hold_required
                            and self._baseline_epoch is not None
                            and isinstance(epoch, int)
                            and not isinstance(epoch, bool)
                            and epoch > self._baseline_epoch
                        )
                    )
                )
            except BaseException:
                strict = False
            if strict:
                hold_since = self.clock() if hold_since is None else hold_since
                if self.clock() - hold_since >= self.config.settle_s:
                    return True
            else:
                hold_since = None
            self.sleeper(self.config.poll_interval_s)
        return False

    def _observe_until_terminal_gate(self, baseline_epoch: int) -> int:
        assert self._post_started_s is not None
        progress_seen = self._progress_seen
        intent_seen = self._intent_seen
        idle_since: float | None = None
        while True:
            elapsed = self.clock() - self._post_started_s
            if elapsed >= self.config.run_deadline_s:
                raise AcceptanceError("command remained unfinished for 25 seconds")
            snapshot, forward = self._poll()
            if not _ra_active_runtime(snapshot.robot_agent, epoch=baseline_epoch):
                raise AcceptanceError("Robot Agent left the strict ACTIVE command phase")
            if _sonic_intent_active(snapshot.sonic):
                intent_seen = True
                self._intent_seen = True
            observed_elapsed = self.clock() - self._post_started_s
            if forward >= PROGRESS_M and observed_elapsed <= self.config.progress_deadline_s:
                progress_seen = True
                self._progress_seen = True
            if not progress_seen and observed_elapsed >= self.config.progress_deadline_s:
                raise AcceptanceError("command made less than 0.03 m progress in 15 seconds")
            if _strict_idle_plant_sonic(snapshot.plant, snapshot.sonic):
                idle_since = self.clock() if idle_since is None else idle_since
            else:
                idle_since = None
            settled = idle_since is not None and self.clock() - idle_since >= self.config.settle_s
            if progress_seen and settled and _ra_active_terminal(snapshot.robot_agent, epoch=baseline_epoch):
                # The four read-only requests above may themselves cross the
                # run deadline.  Never start the sole command query after the
                # 25 s command boundary.
                if self.clock() - self._post_started_s >= self.config.run_deadline_s:
                    raise AcceptanceError("command remained unfinished for 25 seconds")
                if not intent_seen:
                    raise AcceptanceError("physical progress lacked a witnessed SONIC intent")
                if abs(forward - 0.20) > 0.05:
                    raise AcceptanceError("terminal forward distance is outside 0.20 +/- 0.05 m")
                return self._terminal_query(baseline_epoch)
            self.sleeper(self.config.poll_interval_s)

    def _prove_final_hold(self, epoch: int) -> None:
        final_started = self.clock()
        hold_since: float | None = None
        hold_started = False
        while True:
            if self.clock() - final_started >= 8.0:
                raise AcceptanceError("final SAFE_HOLD was not proved within its 8 second budget")
            snapshot, _forward = self._poll()
            motion_truth = bool(
                self._latest_motion_healthy
                and not self._latest_motion_intent
                and self._latest_motion_derived_s is not None
                and self.clock() - self._latest_motion_derived_s <= self.config.file_stall_s
            )
            strict = (
                _safe_hold_plant_sonic(snapshot.plant, snapshot.sonic)
                and _ra_safe_hold(snapshot.robot_agent, epoch=epoch)
                and motion_truth
            )
            if strict:
                hold_since = self.clock() if hold_since is None else hold_since
                hold_started = True
                if self.clock() - hold_since >= self.config.settle_s:
                    return
            else:
                if hold_started:
                    raise AcceptanceError("terminal SAFE_HOLD continuity was lost")
                hold_since = None
            self.sleeper(self.config.poll_interval_s)

    def run(self) -> dict[str, Any]:
        if getattr(self, "_run_started", False):
            raise AcceptanceError("runner instances are single-use")
        self._run_started = True
        self._record(
            "session_start",
            gatewayBase=self.config.base,
            readOnlyPollUrls=[PLANT_URL, SONIC_URL, ROBOT_AGENT_URL, f"{self.config.base}/healthz"],
        )
        try:
            self._wait_for_first_records()
            snapshot = self._snapshot()
            self.safety.observe_snapshot(snapshot)
            if not _safe_hold_plant_sonic(snapshot.plant, snapshot.sonic) or not _ra_safe_hold(snapshot.robot_agent):
                raise AcceptanceError("live preflight is not strict SAFE_HOLD")
            baseline_epoch = _integer(_ra_control(snapshot.robot_agent).get("safeStopEpoch"), "baseline safeStopEpoch", minimum=1)
            self._baseline_epoch = baseline_epoch
            self._baseline_robot_boot_id = self.safety._identity(snapshot.robot_agent.get("robotBootId"), "baseline Robot Agent boot")
            discovery = self._discover(snapshot)
            command = self._build_command(discovery)
            self._record(
                "command_prepared",
                commandId=self.command_id,
                goalPose=command["request"]["payload"]["goalPose"],
                ttlSeconds=30,
                maximumSpeedMps=0.2,
                baselineSafeStopEpoch=baseline_epoch,
            )
            self._post_command(command)
            terminal_epoch = self._observe_until_terminal_gate(baseline_epoch)
            self._prove_final_hold(terminal_epoch)
        except BaseException as exc:
            # No exception text may originate from authorization data; all
            # built-in transport errors are deliberately generic.
            reason = (
                "interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else str(exc)
                if isinstance(exc, AcceptanceError)
                else type(exc).__name__
            )
            return self._abort(reason)
        result = {
            "passed": True,
            "reason": None,
            "commandId": self.command_id,
            "discoverGetCount": 1,
            "commandPostCount": 1,
            "cancelPostCount": 0,
            "commandGetCount": 1,
            "terminalSafeStopEpoch": terminal_epoch,
            "terminalSafeHoldProved": True,
        }
        self._record("result", **result)
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue one fail-closed 0.2 m forward Robot Agent acceptance command",
    )
    parser.add_argument("--motion-jsonl", required=True, help="live motion-monitor JSONL")
    parser.add_argument("--sidecar-jsonl", required=True, help="live idle-soak sidecar JSONL")
    parser.add_argument("--evidence", required=True, help="new exclusive runner evidence JSONL")
    parser.add_argument(
        "--gateway-base",
        default=DEFAULT_GATEWAY_BASE,
        help=f"fixed local Gateway base (default: {DEFAULT_GATEWAY_BASE})",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    motion: JsonlTail | None = None
    sidecar: JsonlTail | None = None
    try:
        config = RunnerConfig(gateway_base=args.gateway_base)
        token = load_gateway_token()
        motion = JsonlTail(args.motion_jsonl)
        sidecar = JsonlTail(args.sidecar_jsonl)
        with ExclusiveEvidence(args.evidence) as evidence:
            result = Forward02mRunner(
                config=config,
                token=token,
                transport=UrllibTransport(),
                motion_jsonl=motion,
                sidecar_jsonl=sidecar,
                evidence=evidence,
            ).run()
    except (AcceptanceError, OSError, ValueError) as exc:
        print(f"acceptance runner setup failed: {exc}", file=os.sys.stderr)
        return 2
    finally:
        if motion is not None:
            motion.close()
        if sidecar is not None:
            sidecar.close()
    if result["passed"]:
        print("0.2 m forward acceptance passed")
        return 0
    print(f"0.2 m forward acceptance failed: {result['reason']}", file=os.sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
