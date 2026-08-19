from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
