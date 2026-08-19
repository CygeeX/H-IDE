"""Run the official Unitree MuJoCo plant with an explicit H-IDE config.

Unitree DDS and ROS 2 use different CycloneDDS language bindings.  Keeping
them in separate processes prevents one runtime from changing the other's
global DDS state.  This plant therefore publishes pose/RGB-D through a
loopback-only observation endpoint; `ros_observation_bridge` republishes the
same capture atomically into the Robot Agent ROS contract.
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sys
import threading
import time
from typing import Iterable
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np

from .config import SimConfig
from .frames import BodyTwist, read_mujoco_local_body_twist
from .render_workers import (
    LatestStateMailbox,
    MAX_CAPTURE_AGE_NS,
    MujocoStateShape,
    PassiveViewerWorker,
    RenderedFrame,
    RenderWorkerConfig,
    RgbdRenderWorker,
    ViewerWorkerConfig,
)
from .runtime_diagnostics import PlantRuntimeDiagnostics
from .scene_runtime import RuntimeScene


class PlantPhysicsFault(RuntimeError):
    """A latched MuJoCo fault that requires an explicit process restart."""


class PlantSupportController:
    """Simulation-only fall-arrest state, never a navigation actuator.

    SONIC and MuJoCo remain connected exclusively through Unitree DDS for joint
    control.  This controller only changes the external safety wrench that the
    upstream simulator exposes for tethered testing.  It cannot create a goal,
    steer the base, or write qpos/qvel.
    """

    MODES = frozenset({"startup", "idle", "locomotion", "safety_stop"})

    def __init__(self) -> None:
        self._mode = "startup"
        self._updated_at = time.time()
        self._fault: str | None = None
        self._last_simulation_time: float | None = None
        self._last_rearm_generation = -1
        self._safety_stop_epoch = 0
        self._lock = threading.RLock()

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> str:
        value = str(mode)
        if value not in self.MODES:
            raise ValueError(
                f"support mode must be one of {', '.join(sorted(self.MODES))}"
            )
        with self._lock:
            if self._fault is not None and value != "safety_stop":
                raise ValueError(
                    f"plant physics fault is latched: {self._fault}"
                )
            if self._mode == "safety_stop" and value != "safety_stop":
                raise ValueError(
                    "plant safety_stop is latched; explicit control rearm is required"
                )
            if value == "safety_stop" and self._mode != "safety_stop":
                self._safety_stop_epoch += 1
            self._mode = value
            self._updated_at = time.time()
            return self._mode

    def rearm(self, generation: int) -> str:
        """Release a non-fault safety latch to IDLE for one newer authority.

        Normal support transitions can only strengthen a stop once the plant
        reaches ``safety_stop``.  The ROS bridge owns the fresh stationary
        observation and SONIC owns this rearm call; requiring a strictly newer
        generation prevents periodic IDLE traffic or a replayed Nav2 command
        from releasing the latch.  A physics fault remains process-restart
        only and is never recoverable through this method.
        """

        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise ValueError("rearm generation must be a positive integer")
        with self._lock:
            if self._fault is not None:
                raise ValueError(
                    f"plant physics fault is latched: {self._fault}"
                )
            if self._mode != "safety_stop":
                raise ValueError("plant rearm requires a latched safety_stop")
            if generation <= self._last_rearm_generation:
                raise ValueError(
                    "plant rearm generation must be strictly newer than the last release"
                )
            self._mode = "idle"
            self._last_rearm_generation = generation
            self._updated_at = time.time()
            return self._mode

    def finish_startup(self) -> str:
        with self._lock:
            if self._fault is None and self._mode == "startup":
                self._mode = "idle"
                self._updated_at = time.time()
            return self._mode

    def latch_fault(self, reason: str) -> bool:
        detail = str(reason).strip()
        if not detail:
            raise ValueError("plant physics fault requires a reason")
        with self._lock:
            newly_latched = self._fault is None
            if newly_latched:
                self._fault = detail
            if self._mode != "safety_stop":
                self._safety_stop_epoch += 1
            self._mode = "safety_stop"
            self._updated_at = time.time()
            return newly_latched

    def observe_simulation_time(self, value: float) -> bool:
        """Latch an unrecoverable fault when MuJoCo implicitly resets time."""

        current = float(value)
        if not math.isfinite(current) or current < 0:
            return self.latch_fault("MuJoCo simulation time is invalid")
        with self._lock:
            previous = self._last_simulation_time
            self._last_simulation_time = current
        if previous is not None and current + 1e-6 < previous:
            return self.latch_fault(
                "MuJoCo physics time rolled back after numerical instability "
                f"({previous:.6f}s -> {current:.6f}s)"
            )
        return False

    def fail(self, reason: str) -> None:
        """Latch one immutable fault and terminate the acceptance plant.

        Keeping the process alive after invalid physics lets downstream
        components consume an old pose and can make a later implicit reset look
        like continuous motion. The strict H-IDE profile therefore exits and
        requires an operator-visible restart.
        """

        self.latch_fault(reason)
        state = self.state()
        raise PlantPhysicsFault(str(state["fault"]))

    def require_healthy(self) -> None:
        state = self.state()
        if state["fault"] is not None:
            raise PlantPhysicsFault(str(state["fault"]))

    def state(self) -> dict[str, object]:
        with self._lock:
            return {
                "mode": self._mode,
                "updatedAt": self._updated_at,
                "fault": self._fault,
                "safetyStopLatched": self._mode == "safety_stop",
                "safetyStopEpoch": self._safety_stop_epoch,
                "lastRearmGeneration": (
                    None
                    if self._last_rearm_generation < 0
                    else self._last_rearm_generation
                ),
            }


def stabilized_support_wrench(
    raw_wrench,
    pose,
    mode: str,
    *,
    tilt_kp: float = 250.0,
    tilt_kd: float = 30.0,
):
    """Keep fall arrest independent from horizontal navigation authority.

    The upstream elastic band produces a 6D wrench.  Navigation must remain
    owned by Nav2/SONIC, so normal locomotion removes world-X/Y forces and yaw
    torque.  It must *not* remove vertical or roll/pitch support: doing so made
    a long G1 walk fall over, and a later Nav2 cancel could only stop commands,
    not the already tumbling rigid body.
    """

    wrench = np.asarray(raw_wrench, dtype=float).copy()
    state = np.asarray(pose, dtype=float)
    if wrench.shape != (6,) or state.shape != (13,):
        raise ValueError("support wrench requires a 6D wrench and 13D pose")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0
        for value in (tilt_kp, tilt_kd)
    ):
        raise ValueError("tilt support gains must be finite and non-negative")

    # The upstream band derives one axis-angle vector from the *complete*
    # orientation.  Its X/Y components are not roll/pitch-only: close to a
    # 180-degree yaw they change sign discontinuously.  Keeping those raw
    # components caused a large external-torque step exactly when a reviewed
    # pi Spin crossed the quaternion branch cut, eventually driving MuJoCo to
    # huge QACC and an implicit reset.  Align the pelvis up-axis with world Z
    # instead.  This tilt error is continuous and exactly invariant to yaw.
    qw, qx, qy, qz = (float(value) for value in state[3:7])
    quaternion_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(quaternion_norm) or quaternion_norm < 1e-9:
        raise ValueError("support pose quaternion must be finite and non-zero")
    qw, qx, qy, qz = (
        value / quaternion_norm for value in (qw, qx, qy, qz)
    )
    body_up_x = 2.0 * (qx * qz + qw * qy)
    body_up_y = 2.0 * (qy * qz - qw * qx)
    wrench[3] = float(tilt_kp) * body_up_y - float(tilt_kd) * float(state[10])
    wrench[4] = -float(tilt_kp) * body_up_x - float(tilt_kd) * float(state[11])

    if mode == "locomotion":
        wrench[:2] = 0.0
        wrench[5] = 0.0
    elif mode == "idle":
        wrench[0] = -120.0 * float(state[7])
        wrench[1] = -120.0 * float(state[8])
        wrench[5] = 0.0
    elif mode == "safety_stop":
        # Emergency braking is velocity damping, never a position target.  It
        # cannot drive toward the map origin; it only dissipates current
        # translation/yaw while retaining vertical and attitude fall arrest.
        wrench[0] = -180.0 * float(state[7])
        wrench[1] = -180.0 * float(state[8])
        wrench[5] = -80.0 * float(state[12])
    elif mode == "startup":
        wrench[5] = 0.0
    else:
        raise ValueError(f"unsupported support mode: {mode}")
    return wrench


def apply_process_start_spawn(data, model, scene: RuntimeScene) -> None:
    """Apply the manifest spawn exactly once during plant construction.

    This helper only initializes the newly-created ``MjData``. It is not used
    by any runtime fault/recovery path, preserving the explicit-restart reset
    contract.
    """

    if int(model.nq) < 7 or int(model.nv) < 6:
        raise RuntimeError("Office V2 runtime model must have a floating root")
    half_yaw = float(scene.spawn_yaw) / 2.0
    data.qpos[:3] = np.asarray(
        [scene.spawn_x, scene.spawn_y, float(model.qpos0[2])],
        dtype=float,
    )
    data.qpos[3:7] = np.asarray(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        dtype=float,
    )
    data.qvel[:] = 0.0


def configure_head_camera(mujoco, model) -> int:
    """Make the articulated G1 camera look level along the base +X axis."""

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "head_camera",
    )
    if camera_id < 0:
        raise RuntimeError("pinned G1 model does not define head_camera")
    camera_rotation = np.asarray(
        [
            0.0, 0.0, -1.0,
            -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ],
        dtype=float,
    )
    mujoco.mju_mat2Quat(model.cam_quat[camera_id], camera_rotation)
    model.cam_fovy[camera_id] = 75.0
    return int(camera_id)


def configure_tracking_viewer(mujoco, model, viewer, scene: RuntimeScene) -> None:
    """Start the operator viewer close to the spawned robot and track it."""

    if viewer is None:
        return
    pelvis_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    )
    if pelvis_id < 0:
        raise RuntimeError("pinned G1 model does not define pelvis")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = pelvis_id
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -22.0
    viewer.cam.distance = 3.8
    viewer.cam.lookat = np.asarray(
        [scene.spawn_x, scene.spawn_y, float(model.qpos0[2])],
        dtype=float,
    )


class PlantPhysicsMonitor:
    """Authority-thread checks that never depend on rendering availability."""

    def __init__(self, config: SimConfig, mujoco_api, model, data, support) -> None:
        self.config = config
        self.mujoco = mujoco_api
        self.model = model
        self.data = data
        self.support = support
        self.root_body_id = int(
            mujoco_api.mj_name2id(
                model,
                mujoco_api.mjtObj.mjOBJ_BODY,
                "pelvis",
            )
        )
        if self.root_body_id < 0:
            raise RuntimeError("pinned G1 model does not define pelvis")

    def observe(self) -> None:
        self.support.observe_simulation_time(float(self.data.time))
        self.support.require_healthy()
        arrays = (self.data.qpos, self.data.qvel, self.data.act)
        if not all(np.all(np.isfinite(values)) for values in arrays):
            self.support.fail("MuJoCo physics state contains non-finite values")
        root_pose = np.asarray(self.data.qpos[:7], dtype=float)
        try:
            read_mujoco_local_body_twist(
                self.mujoco,
                self.model,
                self.data,
                self.root_body_id,
            )
        except ValueError as exc:
            self.support.fail(f"MuJoCo root velocity is invalid: {exc}")
        x, y, z = (float(value) for value in root_pose[:3])
        bounds_min = self.config.runtime_scene.bounds_min
        bounds_max = self.config.runtime_scene.bounds_max
        if not (
            bounds_min[0] <= x <= bounds_max[0]
            and bounds_min[1] <= y <= bounds_max[1]
            and 0.20 <= z <= 1.50
        ):
            self.support.fail(
                "MuJoCo root pose left the validated office envelope "
                f"(x={x:.3f}, y={y:.3f}, z={z:.3f})"
            )


class PlantFrameStore:
    def __init__(
        self,
        config: SimConfig,
        support: PlantSupportController,
        runtime_diagnostics: PlantRuntimeDiagnostics | None = None,
        worker_status_provider=None,
        viewer_status_provider=None,
    ):
        self.config = config
        self.support = support
        self.runtime_diagnostics = runtime_diagnostics
        self._lock = threading.RLock()
        self._latest: RenderedFrame | None = None
        self._previous: RenderedFrame | None = None
        self._worker_status_provider = worker_status_provider
        self._viewer_status_provider = viewer_status_provider
        self._servers: list[ThreadingHTTPServer] = []

    def set_worker_status_provider(self, provider) -> None:
        self._worker_status_provider = provider

    def set_viewer_status_provider(self, provider) -> None:
        self._viewer_status_provider = provider

    def commit(self, frame: RenderedFrame) -> None:
        """Atomically install one complete RGB-D/state generation."""

        if frame.sequence <= 0 or frame.captured_at_unix_ns <= 0:
            raise ValueError("rendered frame identity is invalid")
        if not (
            frame.captured_at_unix_ns
            <= frame.render_started_at_unix_ns
            <= frame.rendered_at_unix_ns
        ):
            raise ValueError("rendered frame timestamps are not ordered")
        uuid.UUID(frame.producer_boot_id)
        if (
            frame.width <= 0
            or frame.height <= 0
            or not isinstance(frame.rgb_jpeg, bytes)
            or not frame.rgb_jpeg
        ):
            raise ValueError("rendered frame RGB payload is invalid")
        if (
            not isinstance(frame.depth_z16_le, bytes)
            or len(frame.depth_z16_le) != frame.width * frame.height * 2
        ):
            raise ValueError("rendered frame depth payload is invalid")
        if not isinstance(frame.pose, tuple) or len(frame.pose) != 7:
            raise ValueError("rendered frame root pose shape is invalid")
        if not isinstance(frame.camera_pose, tuple) or len(frame.camera_pose) != 7:
            raise ValueError("rendered frame camera pose shape is invalid")
        values = (*frame.pose, *frame.camera_pose, frame.simulation_time_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("rendered frame geometry contains non-finite values")
        intrinsics = dict(frame.intrinsics)
        if set(intrinsics) != {"fx", "fy", "cx", "cy"} or not all(
            math.isfinite(float(value)) for value in intrinsics.values()
        ):
            raise ValueError("rendered frame intrinsics are invalid")
        if float(intrinsics["fx"]) <= 0 or float(intrinsics["fy"]) <= 0:
            raise ValueError("rendered frame focal lengths must be positive")
        timings = dict(frame.render_timing)
        if not all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in timings.values()
        ):
            raise ValueError("rendered frame timing is invalid")
        with self._lock:
            latest = self._latest
            if latest is not None:
                if latest.producer_boot_id == frame.producer_boot_id:
                    if frame.sequence <= latest.sequence:
                        raise ValueError("rendered frame sequence did not increase")
                    self._previous = latest
                else:
                    self._previous = None
            self._latest = frame

    def _worker_status(self) -> dict[str, object]:
        provider = self._worker_status_provider
        if provider is None:
            return {
                "status": "starting",
                "ready": False,
                "alive": False,
                "lastError": "RGB-D worker is not configured",
            }
        try:
            value = provider()
        except Exception as exc:
            return {
                "status": "failed",
                "ready": False,
                "alive": False,
                "lastError": f"worker status unavailable: {type(exc).__name__}: {exc}"[:512],
            }
        return dict(value)

    def _viewer_status(self) -> dict[str, object]:
        provider = self._viewer_status_provider
        if provider is None:
            return {
                "status": "disabled",
                "authorityFeedbackEnabled": False,
                "keyCallbackRegistered": False,
            }
        try:
            return dict(provider())
        except Exception as exc:
            return {
                "status": "failed",
                "authorityFeedbackEnabled": False,
                "keyCallbackRegistered": False,
                "lastError": f"viewer status unavailable: {type(exc).__name__}: {exc}"[:512],
            }

    def _observation_status(
        self,
        frame: RenderedFrame | None,
        *,
        now_ns: int | None = None,
    ) -> tuple[str, float | None, dict[str, object]]:
        worker = self._worker_status()
        if worker.get("status") == "failed":
            status = "failed"
        elif worker.get("status") != "running" or frame is None:
            status = "starting"
        else:
            age_ms = max(
                0.0,
                ((time.time_ns() if now_ns is None else int(now_ns)) - frame.captured_at_unix_ns)
                / 1_000_000.0,
            )
            status = "stale" if age_ms > MAX_CAPTURE_AGE_NS / 1_000_000.0 else "running"
            return status, age_ms, worker
        age_ms = (
            None
            if frame is None
            else max(
                0.0,
                ((time.time_ns() if now_ns is None else int(now_ns)) - frame.captured_at_unix_ns)
                / 1_000_000.0,
            )
        )
        return status, age_ms, worker

    def state(self, frame: RenderedFrame | None = None) -> dict[str, object]:
        if frame is None:
            with self._lock:
                frame = self._latest
        support_state = self.support.state()
        observation_status, capture_age_ms, worker_status = self._observation_status(frame)
        physics_status = "failed" if support_state["fault"] is not None else "running"
        status = (
            "failed"
            if physics_status == "failed"
            else ("running" if observation_status == "running" else "degraded")
        )
        pose = frame.pose if frame is not None else (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        camera_pose = (
            frame.camera_pose
            if frame is not None
            else (0.2, 0.0, 1.15, 1.0, 0.0, 0.0, 0.0)
        )
        intrinsics = (
            dict(frame.intrinsics)
            if frame is not None
            else {"fx": 312.77, "fy": 312.77, "cx": 320.0, "cy": 240.0}
        )
        width = frame.width if frame is not None else 640
        height = frame.height if frame is not None else 480
        x, y, z, qw, qx, qy, qz = pose
        camera_x, camera_y, camera_z, camera_qw, camera_qx, camera_qy, camera_qz = camera_pose
        return {
            "status": status,
            "physicsStatus": physics_status,
            "observationStatus": observation_status,
            "schemaVersion": "h-ide.mujoco-capture/2",
            "robotId": self.config.robot_id,
            "sourceMode": "synthetic",
            "sourceId": f"mujoco:unitree-g1:{self.config.robot_id}",
            "localizationEpochId": self.config.localization_epoch_id,
            "producerBootId": frame.producer_boot_id if frame is not None else None,
            "sequence": frame.sequence if frame is not None else 0,
            "capturedAtUnixNs": frame.captured_at_unix_ns if frame is not None else 0,
            "renderStartedAtUnixNs": frame.render_started_at_unix_ns if frame is not None else 0,
            "renderedAtUnixNs": frame.rendered_at_unix_ns if frame is not None else 0,
            "simulationTimeS": frame.simulation_time_s if frame is not None else None,
            "captureAgeMs": capture_age_ms,
            "renderLatencyMs": (
                (frame.rendered_at_unix_ns - frame.captured_at_unix_ns) / 1_000_000.0
                if frame is not None
                else None
            ),
            "pose": {"x": x, "y": y, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz},
            "rootVelocity": (
                frame.root_velocity.capture_state()
                if frame is not None
                else BodyTwist.zero().capture_state()
            ),
            "support": support_state,
            "runtimeTiming": (
                self.runtime_diagnostics.snapshot()
                if self.runtime_diagnostics is not None
                else None
            ),
            "renderTiming": dict(frame.render_timing) if frame is not None else None,
            "observationWorker": worker_status,
            "viewerWorker": self._viewer_status(),
            "cameraPose": {
                "x": camera_x, "y": camera_y, "z": camera_z,
                "qw": camera_qw, "qx": camera_qx, "qy": camera_qy, "qz": camera_qz,
            },
            "intrinsics": intrinsics,
            "rgb": {"width": width, "height": height, "encoding": "jpeg"},
            "depth": {
                "width": width,
                "height": height,
                "encoding": "z16_le",
                "scaleMetersPerUnit": 0.001,
            },
        }

    def _select_frame(self, query: str) -> tuple[RenderedFrame | None, bool]:
        if not query:
            with self._lock:
                return self._latest, False
        values = parse_qs(query, keep_blank_values=True, strict_parsing=True)
        if set(values) != {"boot", "sequence"} or any(len(items) != 1 for items in values.values()):
            raise ValueError("boot and sequence must be supplied exactly once")
        boot = values["boot"][0]
        try:
            uuid.UUID(boot)
            sequence = int(values["sequence"][0])
        except (ValueError, TypeError) as exc:
            raise ValueError("capture identity is invalid") from exc
        if sequence <= 0:
            raise ValueError("capture sequence must be positive")
        with self._lock:
            for candidate in (self._latest, self._previous):
                if (
                    candidate is not None
                    and candidate.producer_boot_id == boot
                    and candidate.sequence == sequence
                ):
                    return candidate, True
        return None, True

    def serve(
        self,
        host: str,
        port: int,
        *,
        docker_observation_host: str | None = None,
    ) -> None:
        store = self

        def handler_type(*, read_only: bool):
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    parsed = urlparse(self.path)
                    frame = None
                    if parsed.path in {"/state", "/rgb.jpg", "/depth.raw"}:
                        try:
                            frame, exact = store._select_frame(parsed.query)
                        except ValueError as exc:
                            self.send_error(400, str(exc))
                            return
                        if exact and frame is None:
                            self.send_error(409, "requested capture is no longer retained")
                            return
                    elif parsed.path == "/healthz" and parsed.query:
                        self.send_error(400, "healthz does not accept a capture identity")
                        return

                    payload = None
                    if parsed.path in {"/healthz", "/state"}:
                        payload = store.state(frame)
                        if parsed.path == "/healthz":
                            payload = {
                                "backend": "mujoco_sonic",
                                **payload,
                            }
                        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                        content_type = "application/json"
                    elif parsed.path == "/rgb.jpg":
                        if frame is None:
                            self.send_error(503, "capture not ready")
                            return
                        if store.state(frame)["status"] != "running":
                            self.send_error(503, "capture is stale or worker failed")
                            return
                        body = frame.rgb_jpeg
                        content_type = "image/jpeg"
                    elif parsed.path == "/depth.raw":
                        if frame is None:
                            self.send_error(503, "capture not ready")
                            return
                        if store.state(frame)["status"] != "running":
                            self.send_error(503, "capture is stale or worker failed")
                            return
                        body = frame.depth_z16_le
                        content_type = "application/octet-stream"
                    elif (
                        parsed.path == "/support"
                        and not read_only
                        and self.client_address[0] in {"127.0.0.1", "::1"}
                    ):
                        body = json.dumps(
                            store.support.state(),
                            separators=(",", ":"),
                        ).encode("utf-8")
                        content_type = "application/json"
                    else:
                        self.send_error(404)
                        return
                    if not body:
                        self.send_error(503, "capture not ready")
                        return
                    status_code = 503 if (
                        parsed.path == "/healthz"
                        and payload is not None
                        and payload.get("status") != "running"
                    ) else 200
                    self.send_response(status_code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    if frame is not None:
                        self.send_header(
                            "X-Hide-Capture-Boot-Id",
                            frame.producer_boot_id,
                        )
                        self.send_header(
                            "X-Hide-Capture-Sequence",
                            str(frame.sequence),
                        )
                        self.send_header(
                            "X-Hide-Captured-At-Unix-Ns",
                            str(frame.captured_at_unix_ns),
                        )
                        self.send_header(
                            "X-Hide-Rendered-At-Unix-Ns",
                            str(frame.rendered_at_unix_ns),
                        )
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        # Pollers use short deadlines; abandoning a late response
                        # must not produce a simulator traceback or affect capture.
                        return

                def do_POST(self):  # noqa: N802
                    if read_only:
                        self.send_response(405)
                        self.send_header("Allow", "GET")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    parsed = urlparse(self.path)
                    if parsed.path not in {"/support", "/support/rearm"}:
                        self.send_error(404)
                        return
                    if self.client_address[0] not in {"127.0.0.1", "::1"}:
                        self.send_error(403, "support control is loopback-only")
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length <= 0 or length > 1024:
                            raise ValueError("bounded JSON body is required")
                        payload = json.loads(self.rfile.read(length))
                        if not isinstance(payload, dict):
                            raise ValueError("support request must be an object")
                        if parsed.path == "/support":
                            if set(payload) != {"mode"}:
                                raise ValueError(
                                    "support request must contain only mode"
                                )
                            store.support.set_mode(str(payload.get("mode", "")))
                        else:
                            if set(payload) != {"generation"}:
                                raise ValueError(
                                    "support rearm request must contain only generation"
                                )
                            store.support.rearm(payload.get("generation"))
                        body = json.dumps(
                            store.support.state(),
                            separators=(",", ":"),
                        ).encode("utf-8")
                    except (ValueError, json.JSONDecodeError) as exc:
                        self.send_error(422, str(exc))
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *_args):
                    return

            return Handler

        listeners = [(host, False, "loopback")]
        if docker_observation_host is not None:
            if docker_observation_host == host:
                raise ValueError("Docker observation host must differ from plant loopback")
            listeners.append((docker_observation_host, True, "docker-readonly"))
        servers: list[tuple[ThreadingHTTPServer, str]] = []
        try:
            for bind_host, read_only, label in listeners:
                server = ThreadingHTTPServer(
                    (bind_host, port),
                    handler_type(read_only=read_only),
                )
                servers.append((server, label))
        except Exception:
            for server, _label in servers:
                server.server_close()
            raise
        self._servers = [server for server, _label in servers]
        for server, label in servers:
            threading.Thread(
                target=server.serve_forever,
                daemon=True,
                name=f"mujoco-observation-http-{label}",
            ).start()

    def close(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        self._servers.clear()


def run_plant(config: SimConfig) -> int:
    repository_root = Path(__file__).resolve().parents[2]
    simulator_root = repository_root / "models" / "unitree-g1-mujoco"
    if not (simulator_root / "sim" / "simulator_factory.py").is_file():
        raise RuntimeError(f"pinned Unitree simulator is missing: {simulator_root}")
    sys.path.insert(0, str(simulator_root))
    from sim.simulator_factory import SimulatorFactory, init_channel

    model_config = {
        "ROBOT_TYPE": "g1_29dof",
        # ``Path / absolute`` in the pinned simulator preserves the verified
        # absolute path loaded from runtime_manifest.json.
        "ROBOT_SCENE": str(config.model_scene),
        "DOMAIN_ID": config.ros_domain_id,
        "INTERFACE": config.interface,
        "SIMULATOR": "mujoco",
        # The scene already has a floating root; its actuator vector contains
        # only the 43 body/hand motors.
        "FREE_BASE": False,
        "enable_waist": False,
        # Keep the official stabilizing tether enabled during startup.  It is
        # released by the sim profile once the standing controller is verified;
        # without an initial support force a policy handoff from zeroed joints can
        # fall before the first SONIC plan is blended in.
        "ENABLE_ELASTIC_BAND": True,
        "SIMULATE_DT": 0.004,
        "VIEWER_DT": 0.02,
        "IMAGE_DT": 1.0 / config.camera_hz,
        # Renderer and viewer ownership is delegated to independent spawn
        # processes below.  The authority must never create a GL context.
        "ENABLE_ONSCREEN": False,
        "ENABLE_OFFSCREEN": False,
        "MP_START_METHOD": "spawn",
        "USE_SENSOR": False,
        "PRINT_SCENE_INFORMATION": False,
        # Never hide a fall/numerical fault behind an in-process teleport.
        # A new spawn is permitted only by explicitly restarting the plant.
        "RESET_POLICY": config.reset_policy,
        "NUM_MOTORS": 29,
        "NUM_JOINTS": 29,
        # SONIC is the 29-DoF body controller.  The hand-containing scene is
        # explicitly marked unstable by the upstream MuJoCo model, so the
        # simulation uses the supported no-hand scene for navigation validation.
        "NUM_HAND_MOTORS": 0,
        "NUM_HAND_JOINTS": 0,
        "motor_effort_limit_list": [88, 88, 88, 139, 50, 50] * 2
        + [88, 50, 50]
        + [25, 25, 25, 25, 25, 5, 5] * 2,
    }
    init_channel(model_config)
    simulator = SimulatorFactory.create_simulator(
        config=model_config,
        env_name="default",
        onscreen=False,
        offscreen=False,
        camera_configs={},
    )

    # The upstream XML supplies the pelvis height but leaves all articulated
    # joints at zero.  Starting a floating humanoid from that pose makes it
    # collapse before a policy has had time to produce its first reference.  Use
    # SONIC's published nominal standing angles as the deterministic initial
    # state; this is still a normal lowstate/lowcmd handoff, not a kinematic
    # teleport during operation.
    import mujoco
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES_MUJOCO

    sim_env = simulator.sim_env
    # The upstream G1 viewer's elastic band is an operator aid: it applies a
    # very stiff spring toward a fixed world point.  That is useful while
    # manually tuning a standing pose, but it actively fights Nav2's horizontal
    # command in an autonomous workstation demo and can pull a walking robot
    # over.  Keep the band as a simulation-only safety support, limiting it to
    # vertical lift and roll/pitch stabilization.  SONIC still receives the
    # complete /cmd_vel and produces the official lowcmd joint targets; this
    # shim does not write qpos or motor commands.
    support_band = getattr(sim_env, "elastic_band", None)
    support_controller = PlantSupportController()
    if support_band is not None:
        original_band_advance = support_band.Advance

        def _fall_arrest_support(pose):
            return stabilized_support_wrench(
                original_band_advance(pose),
                pose,
                support_controller.mode,
                tilt_kp=250.0,
                tilt_kd=30.0,
            )

        support_band.Advance = _fall_arrest_support
    else:
        support_controller.finish_startup()

    apply_process_start_spawn(
        sim_env.mj_data,
        sim_env.mj_model,
        config.runtime_scene,
    )
    # ``body_joint_index`` follows the XML/Unitree lowstate deploy order
    # (left leg, right leg, waist, arms). Initialize from the same official
    # grouped standing pose used by the Unitree command interface. SONIC remains
    # responsible for all subsequent lowcmd updates.
    if len(sim_env.body_joint_index) != len(DEFAULT_ANGLES_MUJOCO):
        raise RuntimeError(
            f"G1 joint layout mismatch: model={len(sim_env.body_joint_index)} "
            f"nominal={len(DEFAULT_ANGLES_MUJOCO)}"
        )
    for i, joint_id in enumerate(sim_env.body_joint_index):
        sim_env.mj_data.qpos[int(joint_id) + 6] = float(DEFAULT_ANGLES_MUJOCO[i])
    sim_env.mj_data.qvel[:] = 0.0
    mujoco.mj_forward(sim_env.mj_model, sim_env.mj_data)
    if support_band is not None:
        # Upstream defaults target world z=1 with very stiff gains, which
        # visibly hoists the pelvis and makes contacting feet propel the robot
        # even while Nav2 is idle.  Anchor the simulation-only fall arrest at
        # the actual initialized body height and use compliant gains.  The
        # wrapper above still zeros every horizontal force and yaw torque.
        attached = int(sim_env.band_attached_link)
        support_band.point = np.asarray(
            [
                config.runtime_scene.spawn_x,
                config.runtime_scene.spawn_y,
                float(sim_env.mj_data.xpos[attached][2]),
            ],
            dtype=float,
        )
        support_band.length = 0.0
        # A fixed joint target is not itself a humanoid balance controller.
        # Preserve the upstream Unitree support stiffness only during the cold
        # start.  After the bounded handoff, the wrapper above removes every
        # component except vertical fall arrest, so it cannot mask or fight
        # SONIC's learned attitude control.
        support_band.kp_pos = 10000.0
        support_band.kd_pos = 1000.0
        support_band.kp_ang = 1000.0
        support_band.kd_ang = 50.0

    horizontal_support_release = None
    if support_band is not None and config.startup_horizontal_support_s > 0:
        def release_horizontal_support_after_control_handoff():
            bridge = getattr(sim_env, "unitree_bridge", None)
            while bridge is not None and not bridge.low_cmd_received:
                time.sleep(0.05)
            if bridge is None:
                return
            time.sleep(config.startup_horizontal_support_s)
            support_controller.finish_startup()

        horizontal_support_release = threading.Thread(
            target=release_horizontal_support_after_control_handoff,
            daemon=True,
            name="mujoco-horizontal-startup-hold-release",
        )
        horizontal_support_release.start()
    else:
        support_controller.finish_startup()

    # The official elastic band is useful only for the first few seconds of a
    # cold-start stand-up.  Release it before navigation so it cannot pull a
    # walking robot back to the origin.  A zero value keeps the support enabled
    # for the whole run, which is useful for isolated controller diagnostics.
    support_release = None
    if config.startup_support_s > 0 and support_band is not None:
        def release_support_after_control_handoff():
            # Model loading can take longer than the configured stand-up ramp.
            # Start that ramp from the first valid lowcmd, not from plant launch,
            # so the robot is never dropped before SONIC has taken control.
            bridge = getattr(sim_env, "unitree_bridge", None)
            while bridge is not None and not bridge.low_cmd_received:
                time.sleep(0.05)
            if bridge is None:
                return
            time.sleep(config.startup_support_s)
            support_band.enable = False

        support_release = threading.Thread(
            target=release_support_after_control_handoff,
            daemon=True,
            name="mujoco-startup-support-release",
        )
        support_release.start()
    frames: PlantFrameStore | None = None
    rgbd_worker: RgbdRenderWorker | None = None
    viewer_worker: PassiveViewerWorker | None = None
    runtime_diagnostics = PlantRuntimeDiagnostics(
        stall_threshold_s=config.plant_lowstate_stale_s,
    )
    physics_monitor = PlantPhysicsMonitor(
        config,
        mujoco,
        sim_env.mj_model,
        sim_env.mj_data,
        support_controller,
    )
    state_shape = MujocoStateShape.from_model(mujoco, sim_env.mj_model)
    producer_boot_id = str(uuid.uuid4())
    original_sim_step = simulator.sim_env.sim_step

    def timed_sim_step():
        started = time.monotonic()
        runtime_diagnostics.observe_loop_start(
            started,
            unix_ns=time.time_ns(),
        )
        try:
            result = original_sim_step()
            physics_monitor.observe()
            return result
        finally:
            runtime_diagnostics.observe_stage(
                "simStep",
                time.monotonic() - started,
                unix_ns=time.time_ns(),
            )

    simulator.sim_env.sim_step = timed_sim_step
    try:
        if config.offscreen:
            frames = PlantFrameStore(
                config,
                support_controller,
                runtime_diagnostics,
            )
            rgbd_mailbox = LatestStateMailbox(
                state_shape,
                producer_boot_id=producer_boot_id,
            )
            rgbd_worker = RgbdRenderWorker(
                mailbox=rgbd_mailbox,
                shape=state_shape,
                config=RenderWorkerConfig(
                    runtime_manifest=str(config.scene_runtime_manifest),
                    model_scene=str(config.model_scene),
                    reset_policy=config.reset_policy,
                    jpeg_quality=config.observation_jpeg_quality,
                ),
                on_frame=frames.commit,
            )
            frames.set_worker_status_provider(rgbd_worker.status)
            rgbd_worker.start()
            frames.serve(
                config.plant_bind_host,
                config.plant_port,
                docker_observation_host=config.docker_observation_bind_host,
            )

            def publish_rgbd_snapshot():
                started = time.monotonic()
                try:
                    rgbd_mailbox.try_publish(
                        mujoco,
                        sim_env.mj_model,
                        sim_env.mj_data,
                        captured_at_unix_ns=time.time_ns(),
                    )
                    return {}
                finally:
                    runtime_diagnostics.observe_stage(
                        "snapshotPublish",
                        time.monotonic() - started,
                        unix_ns=time.time_ns(),
                    )

            simulator.sim_env.update_render_caches = publish_rgbd_snapshot

        if config.onscreen:
            viewer_mailbox = LatestStateMailbox(
                state_shape,
                producer_boot_id=producer_boot_id,
            )
            viewer_worker = PassiveViewerWorker(
                mailbox=viewer_mailbox,
                shape=state_shape,
                config=ViewerWorkerConfig(
                    runtime_manifest=str(config.scene_runtime_manifest),
                    model_scene=str(config.model_scene),
                    reset_policy=config.reset_policy,
                    spawn_x=config.runtime_scene.spawn_x,
                    spawn_y=config.runtime_scene.spawn_y,
                ),
            )
            viewer_worker.start()
            if frames is not None:
                frames.set_viewer_status_provider(viewer_worker.status)

            def publish_viewer_snapshot():
                started = time.monotonic()
                try:
                    viewer_mailbox.try_publish(
                        mujoco,
                        sim_env.mj_model,
                        sim_env.mj_data,
                        captured_at_unix_ns=time.time_ns(),
                    )
                finally:
                    runtime_diagnostics.observe_stage(
                        "viewerSnapshotPublish",
                        time.monotonic() - started,
                        unix_ns=time.time_ns(),
                    )

            simulator.sim_env.update_viewer = publish_viewer_snapshot

        simulator.start()
    finally:
        try:
            if frames is not None:
                frames.close()
        finally:
            try:
                if rgbd_worker is not None:
                    rgbd_worker.close()
            finally:
                try:
                    if viewer_worker is not None:
                        viewer_worker.close()
                finally:
                    simulator.close()
    return 0


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    config = SimConfig.from_file(args.config)
    os.environ.setdefault("ROS_DOMAIN_ID", str(config.ros_domain_id))
    return run_plant(config)


if __name__ == "__main__":
    raise SystemExit(main())
