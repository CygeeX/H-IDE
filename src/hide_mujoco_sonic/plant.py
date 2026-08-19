"""Run the official Unitree MuJoCo plant with an explicit H-IDE config.

Unitree DDS and ROS 2 use different CycloneDDS language bindings.  Keeping
them in separate processes prevents one runtime from changing the other's
global DDS state.  This plant therefore publishes pose/RGB-D through a
loopback-only observation endpoint; `ros_observation_bridge` republishes the
same capture atomically into the Robot Agent ROS contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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


@dataclass(frozen=True, slots=True)
class _SupportCommandState:
    mode: str
    updated_at: float
    safety_stop_epoch: int
    last_rearm_generation: int


@dataclass(frozen=True, slots=True)
class _SupportPhysicsState:
    fault: str | None
    fault_updated_at: float
    last_simulation_time: float | None
    safety_stop_epoch_floor: int


@dataclass(frozen=True, slots=True)
class _FrameWindow:
    """One atomically published, immutable HTTP capture generation."""

    latest: RenderedFrame | None
    previous: RenderedFrame | None


@dataclass(frozen=True, slots=True)
class _PlantDynamicsSnapshot:
    """One immutable authority-thread sample for read-only fault evidence."""

    sequence: int
    captured_at_unix_ns: int
    simulation_time_s: float | None
    support_band_enabled: bool
    attached_body: str | None
    last_applied_wrench: tuple[float | None, ...]
    root_qpos: tuple[float | None, ...]
    root_qvel: tuple[float | None, ...]
    root_qacc: tuple[float | None, ...]
    root_qacc_warmstart: tuple[float | None, ...]
    potential_energy_j: float | None
    kinetic_energy_j: float | None
    warning_counts: tuple[tuple[str, int], ...]


_PHYSICS_WARNING_NAMES = (
    "mjWARN_BADQPOS",
    "mjWARN_BADQVEL",
    "mjWARN_BADQACC",
    "mjWARN_BADCTRL",
)


class PlantSupportController:
    """Simulation-only fall-arrest state, never a navigation actuator.

    SONIC and MuJoCo remain connected exclusively through Unitree DDS for joint
    control.  This controller only changes the external safety wrench that the
    upstream simulator exposes for tethered testing.  It cannot create a goal,
    steer the base, or write qpos/qvel.
    """

    MODES = frozenset({"startup", "idle", "locomotion", "safety_stop"})

    def __init__(self) -> None:
        if sys.implementation.name != "cpython":
            raise RuntimeError("lock-free Plant support snapshots require CPython")
        gil_enabled = getattr(sys, "_is_gil_enabled", None)
        if callable(gil_enabled) and not gil_enabled():
            raise RuntimeError("lock-free Plant support snapshots require the GIL")
        now = time.time()
        self._command_state = _SupportCommandState(
            mode="startup",
            updated_at=now,
            safety_stop_epoch=0,
            last_rearm_generation=-1,
        )
        self._physics_state = _SupportPhysicsState(
            fault=None,
            fault_updated_at=now,
            last_simulation_time=None,
            safety_stop_epoch_floor=0,
        )
        # Only HTTP/background command writers use this lock.  Physics reads
        # immutable state references and publishes its own immutable state, so
        # an arbitrarily slow request can never delay mj_step/LowState.
        self._write_lock = threading.Lock()

    @property
    def mode(self) -> str:
        command = self._command_state
        if self._physics_state.fault is not None:
            return "safety_stop"
        return command.mode

    @property
    def physics_fault(self) -> str | None:
        return self._physics_state.fault

    def set_mode(self, mode: str) -> str:
        value = str(mode)
        if value not in self.MODES:
            raise ValueError(
                f"support mode must be one of {', '.join(sorted(self.MODES))}"
            )
        with self._write_lock:
            physics = self._physics_state
            command = self._command_state
            effective_mode = "safety_stop" if physics.fault is not None else command.mode
            if physics.fault is not None and value != "safety_stop":
                raise ValueError(
                    f"plant physics fault is latched: {physics.fault}"
                )
            if effective_mode == "safety_stop" and value != "safety_stop":
                raise ValueError(
                    "plant safety_stop is latched; explicit control rearm is required"
                )
            epoch = max(
                command.safety_stop_epoch,
                physics.safety_stop_epoch_floor,
            )
            if value == "safety_stop" and effective_mode != "safety_stop":
                epoch += 1
            self._command_state = replace(
                command,
                mode=value,
                updated_at=time.time(),
                safety_stop_epoch=epoch,
            )
            return self.mode

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
        with self._write_lock:
            physics = self._physics_state
            command = self._command_state
            if physics.fault is not None:
                raise ValueError(
                    f"plant physics fault is latched: {physics.fault}"
                )
            if command.mode != "safety_stop":
                raise ValueError("plant rearm requires a latched safety_stop")
            if generation <= command.last_rearm_generation:
                raise ValueError(
                    "plant rearm generation must be strictly newer than the last release"
                )
            self._command_state = replace(
                command,
                mode="idle",
                updated_at=time.time(),
                safety_stop_epoch=max(
                    command.safety_stop_epoch,
                    physics.safety_stop_epoch_floor,
                ),
                last_rearm_generation=generation,
            )
            return self.mode

    def finish_startup(self) -> str:
        with self._write_lock:
            command = self._command_state
            if self._physics_state.fault is None and command.mode == "startup":
                self._command_state = replace(
                    command,
                    mode="idle",
                    updated_at=time.time(),
                )
            return self.mode

    def latch_fault(self, reason: str) -> bool:
        detail = str(reason).strip()
        if not detail:
            raise ValueError("plant physics fault requires a reason")
        physics = self._physics_state
        if physics.fault is not None:
            return False
        command = self._command_state
        current_epoch = max(
            command.safety_stop_epoch,
            physics.safety_stop_epoch_floor,
        )
        # A new physics fault is a distinct stop authority even when an HTTP
        # safety_stop/rearm races this lock-free publisher.  Advancing the
        # floor unconditionally prevents a concurrent rearm from hiding it.
        current_epoch += 1
        self._physics_state = replace(
            physics,
            fault=detail,
            fault_updated_at=time.time(),
            safety_stop_epoch_floor=current_epoch,
        )
        return True

    def observe_simulation_time(self, value: float) -> bool:
        """Latch an unrecoverable fault when MuJoCo implicitly resets time."""

        current = float(value)
        if not math.isfinite(current) or current < 0:
            return self.latch_fault("MuJoCo simulation time is invalid")
        # The authoritative simulator thread is the sole writer of physics
        # state.  HTTP request threads only publish command state under their
        # separate lock, so this copy-on-write update cannot wait for HTTP.
        physics = self._physics_state
        previous = physics.last_simulation_time
        if previous is not None and current + 1e-6 < previous:
            return self.latch_fault(
                "MuJoCo physics time rolled back after numerical instability "
                f"({previous:.6f}s -> {current:.6f}s)"
            )
        self._physics_state = replace(
            physics,
            last_simulation_time=current,
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
        raise PlantPhysicsFault(str(self._physics_state.fault))

    def require_healthy(self) -> None:
        fault = self._physics_state.fault
        if fault is not None:
            raise PlantPhysicsFault(str(fault))

    def state(self) -> dict[str, object]:
        # Both references are immutable and are replaced atomically under the
        # CPython GIL.  This projection never acquires the HTTP writer lock.
        command = self._command_state
        physics = self._physics_state
        fault = physics.fault
        mode = "safety_stop" if fault is not None else command.mode
        generation = command.last_rearm_generation
        return {
            "mode": mode,
            "updatedAt": max(command.updated_at, physics.fault_updated_at),
            "fault": fault,
            "safetyStopLatched": mode == "safety_stop",
            "safetyStopEpoch": max(
                command.safety_stop_epoch,
                physics.safety_stop_epoch_floor,
            ),
            "lastRearmGeneration": None if generation < 0 else generation,
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
        # cannot drive toward the map origin.  Only damp world X/Y here while
        # retaining vertical and attitude fall arrest.  Applying a yaw torque
        # directly to the tethered torso is not equivalent to actuator-space
        # damping: an accepted 0.2 m run showed that ``-80 * wz`` creates an
        # unbounded, numerically unstable sampled-data loop through the
        # waist/contact dynamics.  A
        # high-fidelity reconstructed replay left the validated envelope in
        # 64 ms and reached BADQACC in 76 ms, while the same frozen joint
        # targets with X/Y-only damping remained bounded for two seconds.
        # A controlled yaw stop must therefore come from the SONIC joint
        # controller; a terminal gait fault halts the Plant separately.
        wrench[0] = -180.0 * float(state[7])
        wrench[1] = -180.0 * float(state[8])
        wrench[5] = 0.0
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


class PlantDynamicsEvidence:
    """Publish complete, diagnostic-only dynamics without blocking physics.

    The authoritative MuJoCo thread is the sole writer.  HTTP readers project
    one immutable reference under the CPython GIL, exactly like the Plant
    support state.  These fields are never consumed by a controller or
    republished by the ROS observation bridge; they exist to make a numerical
    stop fault reconstructable after the process has exited.
    """

    _WRENCH_ORDER = (
        "force_x",
        "force_y",
        "force_z",
        "torque_x",
        "torque_y",
        "torque_z",
    )
    _ROOT_QPOS_ORDER = ("x", "y", "z", "qw", "qx", "qy", "qz")
    _ROOT_QVEL_ORDER = tuple(f"qvel[{index}]" for index in range(6))
    _ROOT_QACC_ORDER = tuple(f"qacc[{index}]" for index in range(6))

    def __init__(
        self,
        mujoco_api,
        model,
        data,
        *,
        support_band=None,
        attached_body_id: int | None = None,
    ) -> None:
        self.mujoco = mujoco_api
        self.model = model
        self.data = data
        self.support_band = support_band
        self.root_body_id = int(
            mujoco_api.mj_name2id(
                model,
                mujoco_api.mjtObj.mjOBJ_BODY,
                "pelvis",
            )
        )
        if self.root_body_id < 0:
            raise RuntimeError("pinned G1 model does not define pelvis")
        if attached_body_id is None:
            self.attached_body_id = None
            self.attached_body_name = None
        else:
            body_id = int(attached_body_id)
            if not (0 <= body_id < int(model.nbody)):
                raise ValueError("support attached body ID is invalid")
            body_name = mujoco_api.mj_id2name(
                model,
                mujoco_api.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if not body_name:
                raise ValueError("support attached body must have a stable name")
            self.attached_body_id = body_id
            self.attached_body_name = str(body_name)
        self._sequence = 0
        self._snapshot: _PlantDynamicsSnapshot | None = None

    @staticmethod
    def _finite(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _finite_tuple(cls, values) -> tuple[float | None, ...]:
        return tuple(cls._finite(value) for value in values)

    def _warning_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                name.removeprefix("mjWARN_"),
                int(
                    self.data.warning[
                        int(getattr(self.mujoco.mjtWarning, name))
                    ].number
                ),
            )
            for name in _PHYSICS_WARNING_NAMES
        )

    def observe(self, *, captured_at_unix_ns: int | None = None) -> None:
        """Capture the state after one authoritative step without any lock."""

        energy = (None, None)
        try:
            self.mujoco.mj_energyPos(self.model, self.data)
            self.mujoco.mj_energyVel(self.model, self.data)
            energy = tuple(self._finite(value) for value in self.data.energy[:2])
        except Exception:
            # Evidence must never become a new failure source.  The physics
            # monitor independently treats non-finite authoritative state as a
            # terminal fault; an unavailable energy projection remains explicit.
            energy = (None, None)
        if self.attached_body_id is None:
            wrench = (0.0,) * 6
        else:
            wrench = self._finite_tuple(
                self.data.xfrc_applied[self.attached_body_id]
            )
        self._sequence += 1
        self._snapshot = _PlantDynamicsSnapshot(
            sequence=self._sequence,
            captured_at_unix_ns=(
                time.time_ns()
                if captured_at_unix_ns is None
                else int(captured_at_unix_ns)
            ),
            simulation_time_s=self._finite(self.data.time),
            support_band_enabled=(
                self.support_band is not None
                and bool(getattr(self.support_band, "enable", False))
            ),
            attached_body=self.attached_body_name,
            last_applied_wrench=wrench,
            root_qpos=self._finite_tuple(self.data.qpos[:7]),
            root_qvel=self._finite_tuple(self.data.qvel[:6]),
            root_qacc=self._finite_tuple(self.data.qacc[:6]),
            root_qacc_warmstart=self._finite_tuple(
                self.data.qacc_warmstart[:6]
            ),
            potential_energy_j=energy[0],
            kinetic_energy_j=energy[1],
            warning_counts=self._warning_counts(),
        )

    def capture_state(self) -> dict[str, object]:
        """Return one JSON-safe projection without reading live ``MjData``."""

        snapshot = self._snapshot
        if snapshot is None:
            return {
                "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
                "status": "starting",
                "diagnosticOnly": True,
                "authorityFeedbackEnabled": False,
            }
        finite_values = (
            *snapshot.last_applied_wrench,
            *snapshot.root_qpos,
            *snapshot.root_qvel,
            *snapshot.root_qacc,
            *snapshot.root_qacc_warmstart,
            snapshot.simulation_time_s,
            snapshot.potential_energy_j,
            snapshot.kinetic_energy_j,
        )
        state_finite = all(value is not None for value in finite_values)
        total_energy = None
        if (
            snapshot.potential_energy_j is not None
            and snapshot.kinetic_energy_j is not None
        ):
            total_energy = (
                snapshot.potential_energy_j + snapshot.kinetic_energy_j
            )
        return {
            "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
            "status": "running" if state_finite else "invalid",
            "diagnosticOnly": True,
            "authorityFeedbackEnabled": False,
            "sampleSequence": snapshot.sequence,
            "capturedAtUnixNs": snapshot.captured_at_unix_ns,
            "simulationTimeS": snapshot.simulation_time_s,
            "supportBandEnabled": snapshot.support_band_enabled,
            "attachedBody": snapshot.attached_body,
            "lastAppliedWrench": {
                "frameId": "world",
                "order": list(self._WRENCH_ORDER),
                "values": list(snapshot.last_applied_wrench),
            },
            "rootState": {
                "body": "pelvis",
                "qposOrder": list(self._ROOT_QPOS_ORDER),
                "qpos": list(snapshot.root_qpos),
                "generalizedVelocityRepresentation": (
                    "raw MuJoCo free-joint qvel; preserve order for replay"
                ),
                "qvelOrder": list(self._ROOT_QVEL_ORDER),
                "qvel": list(snapshot.root_qvel),
                "qaccOrder": list(self._ROOT_QACC_ORDER),
                "qacc": list(snapshot.root_qacc),
                "qaccWarmstart": list(snapshot.root_qacc_warmstart),
            },
            "energy": {
                "potentialJ": snapshot.potential_energy_j,
                "kineticJ": snapshot.kinetic_energy_j,
                "totalJ": total_energy,
            },
            "warningCounts": dict(snapshot.warning_counts),
            "finite": state_finite,
        }


class PlantPhysicsMonitor:
    """Authority-thread checks that never depend on rendering availability."""

    _WARNING_NAMES = _PHYSICS_WARNING_NAMES

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
        self._warning_counts = self._read_warning_counts()
        active_warnings = {
            name: count
            for name, count in self._warning_counts.items()
            if count != 0
        }
        if active_warnings:
            self.support.fail(
                "MuJoCo warning counters are non-zero before the first Plant step: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in active_warnings.items()
                )
            )

    def _read_warning_counts(self) -> dict[str, int]:
        return {
            name.removeprefix("mjWARN_"): int(
                self.data.warning[
                    int(getattr(self.mujoco.mjtWarning, name))
                ].number
            )
            for name in self._WARNING_NAMES
        }

    def observe(self) -> None:
        self.support.observe_simulation_time(float(self.data.time))
        self.support.require_healthy()
        warning_counts = self._read_warning_counts()
        for name, current in warning_counts.items():
            previous = self._warning_counts[name]
            if current > previous:
                self.support.fail(
                    f"MuJoCo {name} warning count increased "
                    f"({previous} -> {current})"
                )
        self._warning_counts = warning_counts
        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.qacc,
            self.data.ctrl,
            self.data.act,
            self.data.actuator_force,
        )
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
        dynamics_evidence_provider=None,
    ):
        self.config = config
        self.support = support
        self.runtime_diagnostics = runtime_diagnostics
        # The collector is the sole writer.  HTTP readers load one immutable
        # window reference without touching this commit-only lock, so a slow
        # response can neither block a commit nor observe a torn latest/previous
        # pair.
        self._commit_lock = threading.Lock()
        self._window = _FrameWindow(latest=None, previous=None)
        self._worker_status_provider = worker_status_provider
        self._viewer_status_provider = viewer_status_provider
        self._dynamics_evidence_provider = dynamics_evidence_provider
        self._servers: list[ThreadingHTTPServer] = []

    def set_worker_status_provider(self, provider) -> None:
        self._worker_status_provider = provider

    def set_viewer_status_provider(self, provider) -> None:
        self._viewer_status_provider = provider

    def set_dynamics_evidence_provider(self, provider) -> None:
        self._dynamics_evidence_provider = provider

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
        with self._commit_lock:
            window = self._window
            latest = window.latest
            previous = window.previous
            if latest is not None:
                if latest.producer_boot_id == frame.producer_boot_id:
                    if frame.sequence <= latest.sequence:
                        raise ValueError("rendered frame sequence did not increase")
                    previous = latest
                else:
                    previous = None
            self._window = _FrameWindow(latest=frame, previous=previous)

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

    def _dynamics_evidence(self) -> dict[str, object]:
        provider = self._dynamics_evidence_provider
        if provider is None:
            return {
                "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
                "status": "starting",
                "diagnosticOnly": True,
                "authorityFeedbackEnabled": False,
            }
        try:
            return dict(provider())
        except Exception as exc:
            return {
                "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
                "status": "failed",
                "diagnosticOnly": True,
                "authorityFeedbackEnabled": False,
                "lastError": (
                    "dynamics evidence unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )[:512],
            }

    def _observation_status(
        self,
        frame: RenderedFrame | None,
        *,
        now_ns: int | None = None,
    ) -> tuple[str, float | None, dict[str, object]]:
        worker = self._worker_status()
        age_ns = None
        age_ms = None
        if frame is not None:
            age_ns = (
                (time.time_ns() if now_ns is None else int(now_ns))
                - frame.captured_at_unix_ns
            )
            age_ms = age_ns / 1_000_000.0
            if age_ns < 0:
                return "failed", age_ms, worker
        if worker.get("status") == "failed":
            status = "failed"
        elif worker.get("status") != "running" or frame is None:
            status = "starting"
        else:
            assert age_ns is not None
            status = "stale" if age_ns > MAX_CAPTURE_AGE_NS else "running"
            return status, age_ms, worker
        return status, age_ms, worker

    def state(self, frame: RenderedFrame | None = None) -> dict[str, object]:
        if frame is None:
            frame = self._window.latest
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
            "dynamicsEvidence": self._dynamics_evidence(),
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

    def _image_status(self, frame: RenderedFrame) -> str:
        if self.support.physics_fault is not None:
            return "failed"
        status, _age_ms, _worker = self._observation_status(frame)
        return status

    def _select_frame(self, query: str) -> tuple[RenderedFrame | None, bool]:
        window = self._window
        if not query:
            return window.latest, False
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
        for candidate in (window.latest, window.previous):
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
                        if store._image_status(frame) != "running":
                            self.send_error(503, "capture is stale or worker failed")
                            return
                        body = frame.rgb_jpeg
                        content_type = "image/jpeg"
                    elif parsed.path == "/depth.raw":
                        if frame is None:
                            self.send_error(503, "capture not ready")
                            return
                        if store._image_status(frame) != "running":
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
                    if parsed.path not in {
                        "/support",
                        "/support/rearm",
                        "/support/fault",
                    }:
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
                        elif parsed.path == "/support/rearm":
                            if set(payload) != {"generation"}:
                                raise ValueError(
                                    "support rearm request must contain only generation"
                                )
                            store.support.rearm(payload.get("generation"))
                        else:
                            if set(payload) != {"reason"}:
                                raise ValueError(
                                    "support fault request must contain only reason"
                                )
                            reason = str(payload.get("reason", "")).strip()
                            if not reason or len(reason) > 512:
                                raise ValueError(
                                    "support fault reason must contain 1..512 characters"
                                )
                            store.support.latch_fault(reason)
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
            # Snapshot the mode before checking the terminal latch.  If an
            # HTTP fault arrives between these two reads, require_healthy()
            # aborts.  If it arrives immediately afterwards, this one in-flight
            # step retains the previously observed support mode rather than
            # applying a newly selected emergency wrench to a dynamic pose.
            support_mode = support_controller.mode
            support_controller.require_healthy()
            return stabilized_support_wrench(
                original_band_advance(pose),
                pose,
                support_mode,
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
    dynamics_evidence = PlantDynamicsEvidence(
        mujoco,
        sim_env.mj_model,
        sim_env.mj_data,
        support_band=support_band,
        attached_body_id=(
            int(sim_env.band_attached_link)
            if support_band is not None
            else None
        ),
    )
    dynamics_evidence.observe()
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
            # A terminal motion fault is latched by the bridge before it asks
            # SONIC to freeze any joint target.  Check it before entering the
            # upstream step so the Plant exits instead of integrating a
            # dynamically unsupported posture.  The support callback repeats
            # the check at the last safe point before mj_step.
            support_controller.require_healthy()
            result = original_sim_step()
            # Capture the exact post-step evidence before the terminal monitor
            # can raise and close the process.  HTTP reads only the resulting
            # immutable projection; this data is never fed back into control.
            dynamics_evidence.observe()
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
                dynamics_evidence_provider=dynamics_evidence.capture_state,
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
