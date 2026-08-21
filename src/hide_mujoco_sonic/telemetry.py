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
from typing import Any, Callable


@dataclass
class TelemetryState:
    robot_id: str
    driver_boot_id: str | None = None
    driver_session_id: str | None = None
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
    sonic_intent_generation: int | None = None
    sonic_intent_signature: str | None = None
    sonic_planner_request_generation: int | None = None
    sonic_planner_request_intent_generation: int | None = None
    sonic_planner_request_intent_signature: str | None = None
    sonic_planner_request_signature: str | None = None
    sonic_planner_request_gen_frame: int | None = None
    sonic_planner_request_matches_intent: bool | None = None
    sonic_planner_accepted_generation: int | None = None
    sonic_planner_accepted_intent_generation: int | None = None
    sonic_planner_accepted_intent_signature: str | None = None
    sonic_planner_accepted_signature: str | None = None
    sonic_planner_reference_ready: bool = False
    sonic_planner_required_pending: bool = False
    sonic_planner_required_started_at_s: float | None = None
    sonic_planner_required_timeout_s: float | None = None
    sonic_planner_required_deadline_s: float | None = None
    sonic_planner_ref_cursor: int | None = None
    sonic_planner_requested_root_direction_x: float | None = None
    sonic_planner_requested_root_direction_y: float | None = None
    sonic_planner_predicted_root_direction_x: float | None = None
    sonic_planner_predicted_root_direction_y: float | None = None
    sonic_planner_predicted_root_projection_m: float | None = None
    sonic_planner_root_prefix_projections_m: list[float] | None = None
    sonic_planner_root_prefix_min_projection_m: float | None = None
    sonic_planner_root_prefix_max_projection_m: float | None = None
    sonic_planner_root_first_positive_frame: int | None = None
    sonic_planner_root_first_positive_time_s: float | None = None
    sonic_planner_direction_consistent: bool | None = None
    sonic_planner_requested_yaw_error_rad: float | None = None
    sonic_planner_predicted_yaw_delta_rad: float | None = None
    sonic_planner_yaw_consistent: bool | None = None
    sonic_planner_evidence_reason: str = "no Nav2 motion intent"
    sonic_reference_authority_state: str = "REVOKED"
    sonic_reference_authority_generation: int | None = None
    sonic_reference_authority_source_sequence: int | None = None
    sonic_reference_authority_intent_generation: int | None = None
    sonic_reference_authority_epoch: int = 0
    sonic_reference_actuator_authority: bool = False
    sonic_reference_authority_dds_target_sequence: int | None = None
    sonic_execution_feedback_pending: bool = False
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
    controller_warmup_state_unchanged: bool = False
    startup_qualification_required_steps: int = 0
    startup_qualification_consecutive_steps: int = 0
    startup_qualification_dropped_samples: int = 0
    motion_ready: bool = False
    controller_resync_count: int = 0
    static_hold_active: bool = True
    motion_handoff_active: bool = False
    motion_handoff_ratio: float = 0.0
    policy_activation_epoch: int = 0
    policy_activation_path: str | None = None
    policy_activation_phase: str = "UNPREPARED"
    policy_activation_ratio: float = 0.0
    policy_activation_provenance: dict[str, Any] | None = None
    policy_activation_settle_stage: str | None = None
    policy_activation_settle_stable: bool | None = None
    policy_activation_settle_consecutive_samples: int = 0
    policy_activation_settle_required_consecutive_samples: int = 10
    policy_activation_settle_elapsed_s: float | None = None
    policy_activation_settle_observed_max_joint_error_rad: float | None = None
    policy_activation_settle_observed_max_joint_velocity_rps: float | None = None
    policy_activation_settle_observed_max_gyro_rps: float | None = None
    policy_activation_settle_observed_roll_rad: float | None = None
    policy_activation_settle_observed_pitch_rad: float | None = None
    policy_activation_settle_max_joint_error_rad: float = 0.10
    policy_activation_settle_max_joint_velocity_rps: float = 0.20
    policy_activation_settle_max_gyro_rps: float = 0.15
    policy_activation_settle_max_abs_roll_pitch_rad: float = 0.15
    policy_activation_settle_timeout_s: float = 3.0
    policy_activation_release_epoch: int | None = None
    policy_activation_release_confirmed_epoch: int | None = None
    policy_activation_release_pending: bool = False
    policy_activation_release_error: str | None = None
    policy_activation_prepared_tick_ms: int | None = None
    plant_support_mode: str = "unknown"
    plant_support_requested_mode: str = "unknown"
    plant_support_confirmed_mode: str = "unknown"
    plant_support_stop_pending: bool = False
    plant_support_safety_latched: bool = False
    plant_support_stop_generation: int | None = None
    plant_support_rearm_generation: int | None = None
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
    terminal_boundary_error: str | None = None
    plant_terminal_halt_confirmed: bool | None = None
    plant_terminal_halt_error: str | None = None
    last_error: str | None = None
    updated_at: float = field(default_factory=time.time)

    def json(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryStore:
    _PEAK_SOURCES = {
        "navigation_arm_constrained_joints": "navigation_task_peak_arm_constrained_joints",
        "navigation_arm_raw_max_deviation_rad": "navigation_task_peak_arm_raw_deviation_rad",
        "navigation_arm_raw_max_error_from_measured_rad": "navigation_task_peak_arm_tracking_error_rad",
        "navigation_arm_measured_max_deviation_rad": "navigation_task_peak_arm_measured_deviation_rad",
        "navigation_arm_max_velocity_rps": "navigation_task_peak_arm_velocity_rps",
        "navigation_arm_max_estimated_torque_nm": "navigation_task_peak_arm_estimated_torque_nm",
        "lower_body_target_max_error_rad": "navigation_task_peak_lower_body_target_max_error_rad",
        "lower_body_target_rms_error_rad": "navigation_task_peak_lower_body_target_rms_error_rad",
    }

    def __init__(self, initial: TelemetryState):
        self._value = initial
        self._lock = threading.RLock()
        self._runtime_diagnostics_provider: Callable[[], Mapping[str, Any]] | None = (
            None
        )

    def _apply(self, changes: Mapping[str, Any]) -> None:
        for key, value in changes.items():
            if not hasattr(self._value, key):
                raise KeyError(key)
            setattr(self._value, key, value)
        for source, peak in self._PEAK_SOURCES.items():
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

    def update(self, **changes: Any) -> TelemetryState:
        """Update and return the existing isolated state snapshot contract."""

        with self._lock:
            self._apply(changes)
            return TelemetryState(**self._value.json())

    def update_hot(self, **changes: Any) -> None:
        """Update without cloning the full state for a callback-rate writer.

        The LowState DDS callback is the only production caller.  Readers still
        receive isolated snapshots through :meth:`get` or
        :meth:`snapshot_json`, while ordinary ``update`` keeps its historical
        return value for compatibility.
        """

        with self._lock:
            self._apply(changes)

    def set_runtime_diagnostics_provider(
        self,
        provider: Callable[[], Mapping[str, Any]] | None,
    ) -> None:
        """Attach a read-only diagnostics snapshot to HTTP telemetry."""

        if provider is not None and not callable(provider):
            raise TypeError("runtime diagnostics provider must be callable")
        with self._lock:
            self._runtime_diagnostics_provider = provider

    def snapshot_json(self) -> dict[str, Any]:
        """Return one isolated JSON snapshot, including optional timing data."""

        with self._lock:
            state = self._value.json()
            provider = self._runtime_diagnostics_provider
        if provider is None:
            return state
        try:
            state["runtime_diagnostics"] = dict(provider())
        except Exception as exc:
            # Diagnostics must never turn the health surface into a process
            # failure; preserve a bounded, explicit indication instead.
            state["runtime_diagnostics"] = {
                "snapshot_error": f"{type(exc).__name__}: {exc}"[:256]
            }
        return state

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


def serve_telemetry(
    store: TelemetryStore,
    host: str,
    port: int,
    *,
    docker_observation_host: str | None = None,
) -> tuple[ThreadingHTTPServer, ...]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path not in {"/healthz", "/telemetry"}:
                self.send_error(404)
                return
            state = store.snapshot_json()
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

    hosts = [host]
    if docker_observation_host is not None:
        if docker_observation_host == host:
            raise ValueError("Docker observation host must differ from telemetry loopback")
        hosts.append(docker_observation_host)

    # Bind every socket before starting a serving thread. If the Docker bridge
    # listener cannot be established, SONIC must fail startup rather than leave
    # an apparently healthy loopback-only telemetry service behind.
    servers: list[ThreadingHTTPServer] = []
    try:
        for bind_host in hosts:
            servers.append(ThreadingHTTPServer((bind_host, port), Handler))
    except Exception:
        for server in servers:
            server.server_close()
        raise
    for index, server in enumerate(servers):
        suffix = "loopback" if index == 0 else "docker-readonly"
        threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name=f"telemetry-http-{suffix}",
        ).start()
    return tuple(servers)
