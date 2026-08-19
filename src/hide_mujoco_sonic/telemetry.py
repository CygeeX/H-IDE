from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
from pathlib import Path
import threading
import time
from typing import Any


@dataclass
class TelemetryState:
    robot_id: str
    backend: str = "mujoco_sonic"
    status: str = "starting"
    source_mode: str = "synthetic"
    ros_domain_id: int = 232
    nav2_state: str = "idle"
    sonic_mode: str = "IDLE"
    sonic_speed_mps: float = 0.0
    sonic_policy_speed_mps: float = 0.0
    sonic_movement_angle_rad: float = 0.0
    sonic_facing_angle_rad: float = 0.0
    sonic_decoder_heading_delta_rad: float = 0.0
    last_cmd_vx: float = 0.0
    last_cmd_vy: float = 0.0
    last_cmd_wz: float = 0.0
    cmd_vel_age_s: float | None = None
    plant_tick: int | None = None
    control_tick_ms: int | None = None
    control_dt_s: float | None = None
    control_clock_status: str = "waiting"
    controller_warmup_complete: bool = False
    controller_warmup_wall_s: float | None = None
    controller_resync_count: int = 0
    static_hold_active: bool = True
    motion_handoff_active: bool = False
    motion_handoff_ratio: float = 0.0
    plant_support_mode: str = "unknown"
    pose: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0})
    joint_target_nonzero: bool = False
    navigation_arm_guard_active: bool = True
    navigation_arm_constrained_joints: int = 0
    navigation_arm_max_step_rad: float = 0.0
    navigation_arm_raw_max_deviation_rad: float = 0.0
    navigation_arm_raw_max_error_from_measured_rad: float = 0.0
    navigation_arm_measured_max_deviation_rad: float = 0.0
    navigation_arm_max_velocity_rps: float = 0.0
    navigation_arm_max_estimated_torque_nm: float = 0.0
    navigation_arm_worst_velocity_joint: str | None = None
    navigation_task_active: bool = False
    navigation_task_peak_arm_constrained_joints: int = 0
    navigation_task_peak_arm_raw_deviation_rad: float = 0.0
    navigation_task_peak_arm_tracking_error_rad: float = 0.0
    navigation_task_peak_arm_measured_deviation_rad: float = 0.0
    navigation_task_peak_arm_velocity_rps: float = 0.0
    navigation_task_peak_arm_estimated_torque_nm: float = 0.0
    lower_body_target_max_error_rad: float = 0.0
    lower_body_target_rms_error_rad: float = 0.0
    lower_body_worst_tracking_joint: str | None = None
    lower_body_max_velocity_rps: float = 0.0
    lower_body_max_estimated_torque_nm: float = 0.0
    navigation_task_peak_lower_body_target_max_error_rad: float = 0.0
    navigation_task_peak_lower_body_target_rms_error_rad: float = 0.0
    rgb_frames: int = 0
    depth_frames: int = 0
    fault_trace_path: str | None = None
    fault_trace_sample_count: int = 0
    fault_joint_names: list[str] = field(default_factory=list)
    last_error: str | None = None
    updated_at: float = field(default_factory=time.time)

    def json(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryStore:
    def __init__(self, initial: TelemetryState):
        self._value = initial
        self._lock = threading.RLock()

    def update(self, **changes: Any) -> TelemetryState:
        with self._lock:
            for key, value in changes.items():
                if not hasattr(self._value, key):
                    raise KeyError(key)
                setattr(self._value, key, value)
            peak_sources = {
                "navigation_arm_constrained_joints": "navigation_task_peak_arm_constrained_joints",
                "navigation_arm_raw_max_deviation_rad": "navigation_task_peak_arm_raw_deviation_rad",
                "navigation_arm_raw_max_error_from_measured_rad": "navigation_task_peak_arm_tracking_error_rad",
                "navigation_arm_measured_max_deviation_rad": "navigation_task_peak_arm_measured_deviation_rad",
                "navigation_arm_max_velocity_rps": "navigation_task_peak_arm_velocity_rps",
                "navigation_arm_max_estimated_torque_nm": "navigation_task_peak_arm_estimated_torque_nm",
                "lower_body_target_max_error_rad": "navigation_task_peak_lower_body_target_max_error_rad",
                "lower_body_target_rms_error_rad": "navigation_task_peak_lower_body_target_rms_error_rad",
            }
            for source, peak in peak_sources.items():
                if not self._value.navigation_task_active:
                    break
                if source not in changes:
                    continue
                setattr(
                    self._value,
                    peak,
                    max(getattr(self._value, peak), changes[source]),
                )
            self._value.updated_at = time.time()
            return TelemetryState(**self._value.json())

    def reset_navigation_task_peaks(self) -> TelemetryState:
        """Start a fresh per-goal diagnostic window without changing control."""

        with self._lock:
            self._value.navigation_task_peak_arm_constrained_joints = 0
            self._value.navigation_task_peak_arm_raw_deviation_rad = 0.0
            self._value.navigation_task_peak_arm_tracking_error_rad = 0.0
            self._value.navigation_task_peak_arm_measured_deviation_rad = 0.0
            self._value.navigation_task_peak_arm_velocity_rps = 0.0
            self._value.navigation_task_peak_arm_estimated_torque_nm = 0.0
            self._value.navigation_task_peak_lower_body_target_max_error_rad = 0.0
            self._value.navigation_task_peak_lower_body_target_rms_error_rad = 0.0
            self._value.updated_at = time.time()
            return TelemetryState(**self._value.json())

    def get(self) -> TelemetryState:
        with self._lock:
            return TelemetryState(**self._value.json())


class JointFaultTraceBuffer:
    """Bounded pre-fault control evidence with one stable joint-name table.

    Normal telemetry exposes only scalar summaries.  This buffer retains the
    vectors needed to distinguish a policy-target discontinuity, a handoff or
    arm-guard rewrite, and a plant tracking failure.  It is frozen and written
    only after a terminal fault, so the health endpoint remains small during
    normal operation.
    """

    _VECTOR_FIELDS = (
        "measured_q_rad",
        "measured_dq_rps",
        "measured_ddq_rps2",
        "measured_tau_nm",
        "policy_target_q_rad",
        "handoff_target_q_rad",
        "published_target_q_rad",
    )

    def __init__(self, joint_names: Sequence[str], *, max_samples: int = 250) -> None:
        names = tuple(str(name) for name in joint_names)
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("fault trace joint names must be non-empty and unique")
        if isinstance(max_samples, bool) or int(max_samples) <= 0:
            raise ValueError("fault trace max_samples must be a positive integer")
        self.joint_names = names
        self.max_samples = int(max_samples)
        self._samples: deque[dict[str, Any]] = deque(maxlen=self.max_samples)
        self._lock = threading.RLock()

    @staticmethod
    def _json_number(value: object) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _vector(self, values: Sequence[object], field_name: str) -> list[float | None]:
        result = list(values)
        if len(result) != len(self.joint_names):
            raise ValueError(
                f"{field_name} must contain {len(self.joint_names)} joint values"
            )
        return [self._json_number(value) for value in result]

    def record(
        self,
        *,
        plant_tick_ms: int,
        control_dt_s: float,
        intent: Mapping[str, object],
        handoff_ratio: float,
        plant_support_mode: str,
        measured_q_rad: Sequence[object],
        measured_dq_rps: Sequence[object],
        measured_ddq_rps2: Sequence[object],
        measured_tau_nm: Sequence[object],
        policy_target_q_rad: Sequence[object],
        handoff_target_q_rad: Sequence[object],
        published_target_q_rad: Sequence[object],
    ) -> None:
        if isinstance(plant_tick_ms, bool) or not isinstance(plant_tick_ms, int):
            raise ValueError("fault trace plant tick must be an integer")
        dt = self._json_number(control_dt_s)
        if dt is None or dt <= 0:
            raise ValueError("fault trace control dt must be finite and positive")
        sample: dict[str, Any] = {
            "plant_tick_ms": plant_tick_ms,
            "control_dt_s": dt,
            "intent": {
                str(key): (
                    self._json_number(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else value
                )
                for key, value in intent.items()
            },
            "handoff_ratio": self._json_number(handoff_ratio),
            "plant_support_mode": str(plant_support_mode),
        }
        provided = {
            "measured_q_rad": measured_q_rad,
            "measured_dq_rps": measured_dq_rps,
            "measured_ddq_rps2": measured_ddq_rps2,
            "measured_tau_nm": measured_tau_nm,
            "policy_target_q_rad": policy_target_q_rad,
            "handoff_target_q_rad": handoff_target_q_rad,
            "published_target_q_rad": published_target_q_rad,
        }
        for field_name in self._VECTOR_FIELDS:
            sample[field_name] = self._vector(provided[field_name], field_name)
        with self._lock:
            self._samples.append(sample)

    def snapshot(
        self,
        *,
        reason: str,
        kp: Sequence[object] = (),
        kd: Sequence[object] = (),
    ) -> dict[str, Any]:
        gains: dict[str, list[float | None]] = {}
        if len(kp) > 0:
            gains["kp"] = self._vector(kp, "kp")
        if len(kd) > 0:
            gains["kd"] = self._vector(kd, "kd")
        with self._lock:
            samples = [dict(sample) for sample in self._samples]
        return {
            "schema_version": "h-ide.sonic-fault-trace/v1",
            "captured_at_unix_s": time.time(),
            "reason": str(reason),
            "joint_names": list(self.joint_names),
            "gains": gains,
            "sample_count": len(samples),
            "samples": samples,
        }

    @staticmethod
    def persist(path: str | Path, payload: Mapping[str, object]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        body = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.write_text(body + "\n", encoding="utf-8")
        temporary.replace(target)
        return target


def serve_telemetry(store: TelemetryStore, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path not in {"/healthz", "/telemetry"}:
                self.send_error(404)
                return
            state = store.get().json()
            body = json.dumps(state, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # Health pollers use strict timeouts while SONIC is loading its
                # ONNX sessions.  A disconnected reader is not a service fault
                # and should not flood the operator terminal with tracebacks.
                return

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="telemetry-http").start()
    return server
