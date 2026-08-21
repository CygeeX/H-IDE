"""Process-isolated MuJoCo rendering from bounded authoritative snapshots.

The simulator process is the sole owner of the authoritative ``MjData``.  It
copies ``mjSTATE_FULLPHYSICS`` into a capacity-one shared-memory mailbox with a
non-blocking lock acquisition.  Rendering processes load the verified scene
again, own private ``MjData`` and OpenGL objects, and can therefore stall or
crash without delaying LowState publication or ``mj_step``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import multiprocessing
from pathlib import Path
import queue
import threading
import time
from typing import Callable
import uuid
import weakref

import numpy as np

from .frames import BodyTwist, read_mujoco_local_body_twist


CAPTURE_STATE_NAME = "mjSTATE_FULLPHYSICS"
MAX_CAPTURE_AGE_NS = 500_000_000
RGB_WIDTH = 640
RGB_HEIGHT = 480
RENDER_TIMING_STALL_THRESHOLD_MS = 100.0
VIEWER_PROGRESS_STALE_NS = 2_000_000_000
VIEWER_NATIVE_SHUTDOWN_TIMEOUT_S = 5.0
VIEWER_NATIVE_SHUTDOWN_PARENT_MARGIN_S = 1.0
VIEWER_NATIVE_SHUTDOWN_POLL_S = 0.01
_RENDER_TIMING_STAGE_NAMES = (
    "totalMs",
    "rgbRenderMs",
    "jpegEncodeMs",
    "depthRenderMs",
    "stateApplyMs",
    "mjForwardMs",
    "queueAgeAtStartMs",
)


@dataclass(frozen=True, slots=True)
class MujocoStateShape:
    """Dimensions that must match in the authority and every render worker."""

    nq: int
    nv: int
    na: int
    nhistory: int
    nmocap: int
    npluginstate: int
    nflex: int
    nskin: int
    fullphysics_size: int

    @classmethod
    def from_model(cls, mujoco_api, model) -> "MujocoStateShape":
        shape = cls(
            nq=int(model.nq),
            nv=int(model.nv),
            na=int(model.na),
            nhistory=int(getattr(model, "nhistory", 0)),
            nmocap=int(getattr(model, "nmocap", 0)),
            npluginstate=int(getattr(model, "npluginstate", 0)),
            nflex=int(getattr(model, "nflex", 0)),
            nskin=int(getattr(model, "nskin", 0)),
            fullphysics_size=int(
                mujoco_api.mj_stateSize(
                    model,
                    mujoco_api.mjtState.mjSTATE_FULLPHYSICS,
                )
            ),
        )
        shape.require_supported()
        return shape

    def require_supported(self) -> None:
        if self.nq < 7 or self.nv < 6 or self.fullphysics_size <= 0:
            raise RuntimeError(f"unsupported MuJoCo state shape: {self}")
        # FULLPHYSICS intentionally excludes mocap poses.  History/plugin and
        # deformable state need an explicitly versioned IPC contract before
        # they may be introduced; silently rendering an incomplete future
        # model would produce plausible but false observations.
        unsupported = {
            "nhistory": self.nhistory,
            "nmocap": self.nmocap,
            "npluginstate": self.npluginstate,
            "nflex": self.nflex,
            "nskin": self.nskin,
        }
        active = {name: value for name, value in unsupported.items() if value}
        if active:
            raise RuntimeError(
                "MuJoCo render snapshot does not support this state shape: "
                + ", ".join(f"{name}={value}" for name, value in active.items())
            )


@dataclass(frozen=True, slots=True)
class AuthoritativeSnapshot:
    producer_boot_id: str
    sequence: int
    captured_at_unix_ns: int
    captured_at_monotonic_ns: int
    state: np.ndarray


class LatestStateMailbox:
    """One-slot shared-memory mailbox whose authority-side write never waits."""

    def __init__(
        self,
        shape: MujocoStateShape,
        *,
        producer_boot_id: str | None = None,
        context=None,
    ) -> None:
        shape.require_supported()
        self.shape = shape
        self.producer_boot_id = str(producer_boot_id or uuid.uuid4())
        uuid.UUID(self.producer_boot_id)
        ctx = context or multiprocessing.get_context("spawn")
        self._lock = ctx.Lock()
        self._state = ctx.RawArray("d", shape.fullphysics_size)
        self._sequence = ctx.Value("Q", 0, lock=False)
        self._captured_at_ns = ctx.Value("Q", 0, lock=False)
        self._captured_at_monotonic_ns = ctx.Value("Q", 0, lock=False)
        self._published = ctx.Value("Q", 0, lock=False)
        self._dropped = ctx.Value("Q", 0, lock=False)

    @property
    def published_count(self) -> int:
        return int(self._published.value)

    @property
    def dropped_count(self) -> int:
        return int(self._dropped.value)

    def try_publish(
        self,
        mujoco_api,
        model,
        data,
        *,
        captured_at_unix_ns: int,
        captured_at_monotonic_ns: int | None = None,
    ) -> int | None:
        """Copy one state or drop it immediately when the worker is reading."""

        captured_at = int(captured_at_unix_ns)
        if captured_at <= 0:
            raise ValueError("captured_at_unix_ns must be positive")
        captured_monotonic = int(
            time.monotonic_ns()
            if captured_at_monotonic_ns is None
            else captured_at_monotonic_ns
        )
        if captured_monotonic <= 0:
            raise ValueError("captured_at_monotonic_ns must be positive")
        if not self._lock.acquire(False):
            self._dropped.value = int(self._dropped.value) + 1
            return None
        try:
            target = np.frombuffer(self._state, dtype=np.float64)
            mujoco_api.mj_getState(
                model,
                data,
                target,
                mujoco_api.mjtState.mjSTATE_FULLPHYSICS,
            )
            sequence = int(self._sequence.value) + 1
            self._captured_at_ns.value = captured_at
            self._captured_at_monotonic_ns.value = captured_monotonic
            # Sequence is written last while holding the lock, making it the
            # commit marker for the state and timestamp copied above.
            self._sequence.value = sequence
            self._published.value = int(self._published.value) + 1
            return sequence
        finally:
            self._lock.release()

    def copy_latest(self, *, after_sequence: int = 0) -> AuthoritativeSnapshot | None:
        """Copy the current slot into process-local memory, never returning a view."""

        if not self._lock.acquire(True, 0.05):
            return None
        try:
            sequence = int(self._sequence.value)
            if sequence <= int(after_sequence):
                return None
            captured_at = int(self._captured_at_ns.value)
            captured_at_monotonic = int(
                self._captured_at_monotonic_ns.value
            )
            state = np.frombuffer(self._state, dtype=np.float64).copy()
        finally:
            self._lock.release()
        return AuthoritativeSnapshot(
            producer_boot_id=self.producer_boot_id,
            sequence=sequence,
            captured_at_unix_ns=captured_at,
            captured_at_monotonic_ns=captured_at_monotonic,
            state=state,
        )


@dataclass(frozen=True, slots=True)
class RenderWorkerConfig:
    runtime_manifest: str
    model_scene: str
    reset_policy: str
    jpeg_quality: int
    width: int = RGB_WIDTH
    height: int = RGB_HEIGHT
    camera_name: str = "head_camera"
    gl_backend: str = "egl"


@dataclass(frozen=True, slots=True)
class ViewerWorkerConfig:
    runtime_manifest: str
    model_scene: str
    reset_policy: str
    spawn_x: float
    spawn_y: float
    viewer_period_s: float = 0.02
    gl_backend: str = "glfw"


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    producer_boot_id: str
    sequence: int
    captured_at_unix_ns: int
    captured_at_monotonic_ns: int
    render_started_at_unix_ns: int
    render_started_at_monotonic_ns: int
    rendered_at_unix_ns: int
    rendered_at_monotonic_ns: int
    simulation_time_s: float
    pose: tuple[float, float, float, float, float, float, float]
    root_velocity: BodyTwist
    camera_pose: tuple[float, float, float, float, float, float, float]
    intrinsics: tuple[tuple[str, float], ...]
    width: int
    height: int
    rgb_jpeg: bytes
    depth_z16_le: bytes
    render_timing: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    at_unix_ns: int
    message: str
    fatal: bool


def _empty_render_timing_snapshot() -> dict[str, object]:
    return {
        "thresholdMs": RENDER_TIMING_STALL_THRESHOLD_MS,
        "stages": {
            name: {
                "lastMs": None,
                "maxMs": 0.0,
                "overThresholdCount": 0,
                "lastOverThresholdMs": None,
                "lastOverThresholdAtUnixNs": None,
            }
            for name in _RENDER_TIMING_STAGE_NAMES
        },
    }


def _copy_render_timing_snapshot(value: dict[str, object]) -> dict[str, object]:
    stages = value["stages"]
    assert isinstance(stages, dict)
    return {
        "thresholdMs": value["thresholdMs"],
        "stages": {
            name: dict(stage)
            for name, stage in stages.items()
        },
    }


def _record_render_timing(stats, frame: RenderedFrame) -> None:
    """Publish every completed worker timing without losing slow evidence."""

    timings = dict(frame.render_timing)
    if set(timings) != set(_RENDER_TIMING_STAGE_NAMES):
        raise RuntimeError("render worker timing stages do not match the contract")
    values = tuple(float(timings[name]) for name in _RENDER_TIMING_STAGE_NAMES)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise RuntimeError("render worker timing contains an invalid duration")
    observed_at_ns = int(frame.rendered_at_unix_ns)
    if observed_at_ns <= 0:
        raise RuntimeError("render worker timing timestamp is invalid")

    # Only the render child writes.  HTTP uses a non-blocking read below, so a
    # dead child holding this lock can never hang Plant state/health requests.
    with stats.render_timing_lock:
        for index, duration_ms in enumerate(values):
            stats.render_timing_last_ms[index] = duration_ms
            stats.render_timing_max_ms[index] = max(
                float(stats.render_timing_max_ms[index]),
                duration_ms,
            )
            if duration_ms > RENDER_TIMING_STALL_THRESHOLD_MS:
                stats.render_timing_over_threshold_count[index] = (
                    int(stats.render_timing_over_threshold_count[index]) + 1
                )
                stats.render_timing_last_over_ms[index] = duration_ms
                stats.render_timing_last_over_at_ns[index] = observed_at_ns
        stats.render_timing_observed.value = 1


def _read_render_timing(stats) -> dict[str, object] | None:
    """Return one complete child generation or fail open to a parent cache."""

    if not stats.render_timing_lock.acquire(False):
        return None
    try:
        last_ms = tuple(float(value) for value in stats.render_timing_last_ms)
        max_ms = tuple(float(value) for value in stats.render_timing_max_ms)
        counts = tuple(
            int(value) for value in stats.render_timing_over_threshold_count
        )
        last_over_ms = tuple(
            float(value) for value in stats.render_timing_last_over_ms
        )
        last_over_at = tuple(
            int(value) for value in stats.render_timing_last_over_at_ns
        )
        observed = bool(stats.render_timing_observed.value)
    finally:
        stats.render_timing_lock.release()
    return {
        "thresholdMs": RENDER_TIMING_STALL_THRESHOLD_MS,
        "stages": {
            name: {
                "lastMs": last_ms[index] if observed else None,
                "maxMs": max_ms[index],
                "overThresholdCount": counts[index],
                "lastOverThresholdMs": (
                    last_over_ms[index] if last_over_at[index] else None
                ),
                "lastOverThresholdAtUnixNs": last_over_at[index] or None,
            }
            for index, name in enumerate(_RENDER_TIMING_STAGE_NAMES)
        },
    }


def _record_viewer_progress(stats, mailbox, snapshot) -> None:
    """Commit one internally consistent viewer progress generation."""

    with stats.viewer_progress_lock:
        stats.last_success_at_ns.value = time.time_ns()
        stats.last_success_sequence.value = snapshot.sequence
        stats.last_success_mailbox_dropped_count.value = mailbox.dropped_count
        # Monotonic is the commit marker and is written last.
        stats.last_success_monotonic_ns.value = time.monotonic_ns()


def _read_viewer_progress(stats) -> tuple[int, int, int, int] | None:
    if not stats.viewer_progress_lock.acquire(False):
        return None
    try:
        return (
            int(stats.last_success_at_ns.value),
            int(stats.last_success_monotonic_ns.value),
            int(stats.last_success_sequence.value),
            int(stats.last_success_mailbox_dropped_count.value),
        )
    finally:
        stats.viewer_progress_lock.release()


def _set_shared_error(stats, message: str) -> str:
    bounded = str(message)[:512]
    encoded = bounded.encode("utf-8", errors="replace")[:512]
    with stats.last_error.get_lock():
        stats.last_error.value = encoded
    return bounded


def _clear_shared_error(stats) -> None:
    with stats.last_error.get_lock():
        stats.last_error.value = b""


def _initialize_worker_identity(stats) -> None:
    import os

    worker_boot_id = str(uuid.uuid4()).encode("ascii")
    stats.worker_pid.value = os.getpid()
    with stats.worker_boot_id.get_lock():
        stats.worker_boot_id.value = worker_boot_id


def configure_head_camera(mujoco_api, model) -> int:
    """Make the articulated G1 camera look level along the base +X axis."""

    camera_id = mujoco_api.mj_name2id(
        model,
        mujoco_api.mjtObj.mjOBJ_CAMERA,
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
    mujoco_api.mju_mat2Quat(model.cam_quat[camera_id], camera_rotation)
    model.cam_fovy[camera_id] = 75.0
    return int(camera_id)


def configure_tracking_viewer(mujoco_api, model, viewer, scene) -> None:
    """Configure a worker-local passive viewer without input callbacks."""

    if viewer is None:
        return
    pelvis_id = int(
        mujoco_api.mj_name2id(model, mujoco_api.mjtObj.mjOBJ_BODY, "pelvis")
    )
    if pelvis_id < 0:
        raise RuntimeError("pinned G1 model does not define pelvis")
    viewer.cam.type = mujoco_api.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = pelvis_id
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -22.0
    viewer.cam.distance = 3.8
    viewer.cam.lookat = np.asarray(
        [scene.spawn_x, scene.spawn_y, float(model.qpos0[2])],
        dtype=float,
    )


def _load_verified_model(config, expected_shape: MujocoStateShape):
    import mujoco

    from .scene_runtime import load_runtime_scene

    runtime_scene = load_runtime_scene(
        config.runtime_manifest,
        configured_model_scene=config.model_scene,
        configured_reset_policy=config.reset_policy,
    )
    if Path(config.model_scene).resolve() != runtime_scene.model_scene:
        raise RuntimeError("render worker scene path differs from verified runtime scene")
    model = mujoco.MjModel.from_xml_path(str(runtime_scene.model_scene))
    actual_shape = MujocoStateShape.from_model(mujoco, model)
    if actual_shape != expected_shape:
        raise RuntimeError(
            f"render worker model shape differs: expected={expected_shape}, actual={actual_shape}"
        )
    return mujoco, runtime_scene, model


def _put_latest(result_queue, value) -> None:
    """Best-effort latest-only child output; the renderer must never wait."""

    try:
        result_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        result_queue.get_nowait()
    except queue.Empty:
        # multiprocessing.Queue can report full while its feeder has not made
        # the item readable yet.  Dropping this output is safer than waiting.
        return
    try:
        result_queue.put_nowait(value)
    except queue.Full:
        return


def _encode_rgb_jpeg(rgb: np.ndarray, *, quality: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError("render worker JPEG encoder rejected the RGB frame")
    return encoded.tobytes()


def _render_snapshot(
    *,
    mujoco_api,
    model,
    data,
    renderer,
    camera_id: int,
    root_body_id: int,
    snapshot: AuthoritativeSnapshot,
    shape: MujocoStateShape,
    config: RenderWorkerConfig,
) -> RenderedFrame:
    if snapshot.state.shape != (shape.fullphysics_size,):
        raise RuntimeError("render worker received an invalid state vector size")
    render_started_ns = time.time_ns()
    render_started_monotonic_ns = time.monotonic_ns()
    queue_age_ns = (
        render_started_monotonic_ns - snapshot.captured_at_monotonic_ns
    )
    if queue_age_ns < 0:
        raise RuntimeError("render worker monotonic capture age is negative")
    render_total_started = time.perf_counter()
    apply_started = time.perf_counter()
    mujoco_api.mj_setState(
        model,
        data,
        snapshot.state,
        mujoco_api.mjtState.mjSTATE_FULLPHYSICS,
    )
    state_apply_ms = (time.perf_counter() - apply_started) * 1000.0
    forward_started = time.perf_counter()
    mujoco_api.mj_forward(model, data)
    mj_forward_ms = (time.perf_counter() - forward_started) * 1000.0

    rgb_started = time.perf_counter()
    renderer.update_scene(data, camera=config.camera_name)
    rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
    rgb_render_ms = (time.perf_counter() - rgb_started) * 1000.0
    if rgb.shape != (config.height, config.width, 3):
        raise RuntimeError(f"render worker RGB shape is invalid: {rgb.shape}")

    depth_started = time.perf_counter()
    renderer.enable_depth_rendering()
    try:
        renderer.update_scene(data, camera=config.camera_name)
        depth_m = np.asarray(renderer.render(), dtype=np.float32).copy()
    finally:
        renderer.disable_depth_rendering()
    depth_render_ms = (time.perf_counter() - depth_started) * 1000.0
    if depth_m.shape != (config.height, config.width):
        raise RuntimeError(f"render worker depth shape is invalid: {depth_m.shape}")
    if not np.all(np.isfinite(depth_m)):
        raise RuntimeError("render worker depth contains non-finite values")

    jpeg_started = time.perf_counter()
    rgb_jpeg = _encode_rgb_jpeg(rgb, quality=config.jpeg_quality)
    jpeg_encode_ms = (time.perf_counter() - jpeg_started) * 1000.0
    depth = np.clip(depth_m * 1000.0, 0, 65535).astype("<u2")

    root_pose = tuple(float(value) for value in data.qpos[:7])
    if len(root_pose) != 7 or not all(math.isfinite(value) for value in root_pose):
        raise RuntimeError("render worker root pose is invalid")
    root_velocity = read_mujoco_local_body_twist(
        mujoco_api,
        model,
        data,
        root_body_id,
    )
    mujoco_camera_rotation = np.asarray(
        data.cam_xmat[camera_id],
        dtype=float,
    ).reshape(3, 3)
    optical_rotation = mujoco_camera_rotation @ np.diag([1.0, -1.0, -1.0])
    optical_quaternion = np.empty(4, dtype=float)
    mujoco_api.mju_mat2Quat(optical_quaternion, optical_rotation.reshape(-1))
    if optical_quaternion[0] < 0:
        optical_quaternion *= -1
    camera_pose = (
        *(float(value) for value in data.cam_xpos[camera_id]),
        *(float(value) for value in optical_quaternion),
    )
    focal_pixels = (config.height / 2.0) / math.tan(
        math.radians(float(model.cam_fovy[camera_id])) / 2.0
    )
    rendered_at_ns = time.time_ns()
    rendered_at_monotonic_ns = time.monotonic_ns()
    if rendered_at_monotonic_ns < render_started_monotonic_ns:
        raise RuntimeError("render worker monotonic timestamps are not ordered")
    total_ms = (time.perf_counter() - render_total_started) * 1000.0
    return RenderedFrame(
        producer_boot_id=snapshot.producer_boot_id,
        sequence=snapshot.sequence,
        captured_at_unix_ns=snapshot.captured_at_unix_ns,
        captured_at_monotonic_ns=snapshot.captured_at_monotonic_ns,
        render_started_at_unix_ns=render_started_ns,
        render_started_at_monotonic_ns=render_started_monotonic_ns,
        rendered_at_unix_ns=rendered_at_ns,
        rendered_at_monotonic_ns=rendered_at_monotonic_ns,
        simulation_time_s=float(data.time),
        pose=root_pose,  # type: ignore[arg-type]
        root_velocity=root_velocity,
        camera_pose=camera_pose,  # type: ignore[arg-type]
        intrinsics=(
            ("fx", focal_pixels),
            ("fy", focal_pixels),
            ("cx", config.width / 2.0),
            ("cy", config.height / 2.0),
        ),
        width=config.width,
        height=config.height,
        rgb_jpeg=rgb_jpeg,
        depth_z16_le=depth.tobytes(),
        render_timing=(
            ("stateApplyMs", state_apply_ms),
            ("rgbRenderMs", rgb_render_ms),
            ("depthRenderMs", depth_render_ms),
            ("jpegEncodeMs", jpeg_encode_ms),
            ("totalMs", total_ms),
            ("mjForwardMs", mj_forward_ms),
            (
                "queueAgeAtStartMs",
                queue_age_ns / 1_000_000.0,
            ),
        ),
    )


def _rgbd_worker_main(
    mailbox: LatestStateMailbox,
    expected_shape: MujocoStateShape,
    config: RenderWorkerConfig,
    result_queue,
    stop_event,
    stats,
) -> None:
    # MuJoCo chooses its GL backend at import time.  This fresh spawn process is
    # the only process allowed to create the offscreen Renderer.
    import os

    _initialize_worker_identity(stats)
    os.environ["MUJOCO_GL"] = config.gl_backend
    renderer = None
    fatal_failure_counted = False
    try:
        mujoco_api, _runtime_scene, model = _load_verified_model(
            config,
            expected_shape,
        )
        data = mujoco_api.MjData(model)
        camera_id = configure_head_camera(mujoco_api, model)
        root_body_id = int(
            mujoco_api.mj_name2id(
                model,
                mujoco_api.mjtObj.mjOBJ_BODY,
                "pelvis",
            )
        )
        if root_body_id < 0:
            raise RuntimeError("pinned G1 model does not define pelvis")
        renderer = mujoco_api.Renderer(
            model,
            height=config.height,
            width=config.width,
        )
        stats.ready.value = 1
        last_sequence = 0
        consecutive_failures = 0
        while not stop_event.is_set():
            snapshot = mailbox.copy_latest(after_sequence=last_sequence)
            if snapshot is None:
                stop_event.wait(0.001)
                continue
            last_sequence = snapshot.sequence
            try:
                frame = _render_snapshot(
                    mujoco_api=mujoco_api,
                    model=model,
                    data=data,
                    renderer=renderer,
                    camera_id=camera_id,
                    root_body_id=root_body_id,
                    snapshot=snapshot,
                    shape=expected_shape,
                    config=config,
                )
                _record_render_timing(stats, frame)
                consecutive_failures = 0
                _clear_shared_error(stats)
                stats.last_success_at_ns.value = frame.rendered_at_unix_ns
                stats.last_success_sequence.value = frame.sequence
                _put_latest(result_queue, frame)
            except Exception as exc:
                consecutive_failures += 1
                failure_at = time.time_ns()
                stats.failure_count.value = int(stats.failure_count.value) + 1
                stats.last_failure_at_ns.value = failure_at
                fatal = consecutive_failures >= 3
                message = _set_shared_error(
                    stats,
                    f"{type(exc).__name__}: {exc}",
                )
                _put_latest(
                    result_queue,
                    WorkerFailure(
                        at_unix_ns=failure_at,
                        message=message,
                        fatal=fatal,
                    ),
                )
                if fatal:
                    fatal_failure_counted = True
                    raise RuntimeError(
                        "RGB-D worker failed three consecutive frames: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
    except BaseException as exc:
        failure_at = time.time_ns()
        stats.fatal.value = 1
        if not fatal_failure_counted:
            stats.failure_count.value = int(stats.failure_count.value) + 1
        stats.last_failure_at_ns.value = failure_at
        message = _set_shared_error(
            stats,
            f"{type(exc).__name__}: {exc}",
        )
        _put_latest(
            result_queue,
            WorkerFailure(
                at_unix_ns=failure_at,
                message=message,
                fatal=True,
            ),
        )
    finally:
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass


def _require_native_lifetime_ref(viewer) -> weakref.ReferenceType:
    """Validate the MuJoCo lifetime barrier before the viewer becomes ready."""

    simulate_ref = getattr(viewer, "_sim", None)
    if not isinstance(simulate_ref, weakref.ReferenceType):
        raise RuntimeError(
            "passive viewer does not expose a weak native lifetime reference"
        )
    simulate = simulate_ref()
    if simulate is None or not callable(getattr(simulate, "exit", None)):
        raise RuntimeError("passive viewer native lifetime is already unavailable")
    return simulate_ref


def _close_and_wait_for_passive_viewer(
    viewer,
    *,
    simulate_ref: weakref.ReferenceType | None = None,
    timeout_s: float = VIEWER_NATIVE_SHUTDOWN_TIMEOUT_S,
    poll_s: float = VIEWER_NATIVE_SHUTDOWN_POLL_S,
    monotonic: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
) -> None:
    """Close a passive viewer only after its native render thread is gone.

    ``Handle.close()`` only submits ``Simulate.exit()``.  In MuJoCo's passive
    viewer implementation, ``Handle.is_running()`` becomes false as soon as
    that exit request is visible, before ``render_loop()``, the reload thread,
    and ``Simulate.destroy()`` have necessarily completed.  The handle's
    private weak reference is cleared only after ``_launch_internal`` really
    returns, so it is the lifetime barrier needed before releasing MjData and
    MjModel in this process.
    """

    if simulate_ref is None:
        simulate_ref = _require_native_lifetime_ref(viewer)
    timeout = max(0.0, float(timeout_s))
    poll = max(0.001, float(poll_s))
    deadline = monotonic() + timeout
    viewer.close()
    while simulate_ref() is not None:
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                "passive viewer native render thread did not stop within "
                f"{timeout:.3f}s"
            )
        wait(min(poll, remaining))


def _abort_viewer_worker_without_python_finalizers(stats, exc: BaseException) -> None:
    """Fail closed without destructing model/data under a live GL thread."""

    import os

    failure_at = time.time_ns()
    stats.fatal.value = 1
    stats.failure_count.value = int(stats.failure_count.value) + 1
    stats.last_failure_at_ns.value = failure_at
    _set_shared_error(stats, f"{type(exc).__name__}: {exc}")
    # Refcount-driven MjData/MjModel cleanup can race MuJoCo's still-running
    # native render thread.  A process-local immediate exit is the last-resort
    # fail-closed path: the OS tears down the isolated worker as one unit and
    # the parent never SIGTERMs it in the middle of Python finalization.
    os._exit(1)


def _viewer_shutdown_watchdog(
    stop_event,
    shutdown_requested: threading.Event,
    shutdown_complete: threading.Event,
    stats,
) -> None:
    """Self-abort if the main worker is stuck inside native viewer code."""

    while not stop_event.is_set():
        if shutdown_requested.wait(VIEWER_NATIVE_SHUTDOWN_POLL_S):
            break
    if shutdown_complete.wait(VIEWER_NATIVE_SHUTDOWN_TIMEOUT_S):
        return
    _abort_viewer_worker_without_python_finalizers(
        stats,
        TimeoutError("passive viewer native shutdown watchdog expired"),
    )


def _viewer_worker_main(
    mailbox: LatestStateMailbox,
    expected_shape: MujocoStateShape,
    config: ViewerWorkerConfig,
    stop_event,
    stats,
) -> None:
    """Own a passive viewer with no key callback and no authority feedback."""

    viewer = None
    model = None
    data = None
    native_shutdown_requested = threading.Event()
    native_shutdown_complete = threading.Event()
    shutdown_watchdog = None
    native_lifetime_ref = None
    try:
        import os

        _initialize_worker_identity(stats)
        os.environ.pop("MUJOCO_GL", None)
        os.environ["MUJOCO_GL"] = config.gl_backend
        mujoco_api, runtime_scene, model = _load_verified_model(
            config,
            expected_shape,
        )
        if (
            abs(float(runtime_scene.spawn_x) - float(config.spawn_x)) > 1e-9
            or abs(float(runtime_scene.spawn_y) - float(config.spawn_y)) > 1e-9
        ):
            raise RuntimeError("viewer spawn differs from verified runtime scene")
        data = mujoco_api.MjData(model)
        first = None
        while first is None and not stop_event.is_set():
            first = mailbox.copy_latest(after_sequence=0)
            if first is None:
                stop_event.wait(0.01)
        if first is None:
            return
        mujoco_api.mj_setState(
            model,
            data,
            first.state,
            mujoco_api.mjtState.mjSTATE_FULLPHYSICS,
        )
        mujoco_api.mj_forward(model, data)
        _record_viewer_progress(stats, mailbox, first)
        from mujoco import viewer as mujoco_viewer

        # No key_callback is registered.  Mouse perturbations, pause state and
        # all viewer-side edits apply only to this worker-private MjData.
        viewer = mujoco_viewer.launch_passive(model, data)
        try:
            native_lifetime_ref = _require_native_lifetime_ref(viewer)
        except RuntimeError as exc:
            raise RuntimeError(
                "unsupported MuJoCo passive-viewer lifetime contract "
                f"(version={getattr(mujoco_api, '__version__', 'unknown')}): {exc}"
            ) from exc
        stats.viewer_native_active.value = 1
        shutdown_watchdog = threading.Thread(
            target=_viewer_shutdown_watchdog,
            args=(
                stop_event,
                native_shutdown_requested,
                native_shutdown_complete,
                stats,
            ),
            daemon=True,
            name="mujoco-passive-viewer-shutdown-watchdog",
        )
        shutdown_watchdog.start()
        if not viewer.is_running():
            raise RuntimeError("passive viewer closed before initialization")
        with viewer.lock():
            if not viewer.is_running():
                raise RuntimeError(
                    "passive viewer closed before camera initialization"
                )
            configure_tracking_viewer(
                mujoco_api,
                model,
                viewer,
                runtime_scene,
            )
        # This is both the first coherent camera publication and a startup
        # feature probe for the state-only sync contract used by the loop.
        viewer.sync(state_only=True)
        if not viewer.is_running():
            raise RuntimeError("passive viewer closed during initialization")
        stats.ready.value = 1
        last_sequence = first.sequence
        while not stop_event.is_set() and viewer.is_running():
            snapshot = mailbox.copy_latest(after_sequence=last_sequence)
            if snapshot is not None:
                mujoco_api.mj_setState(
                    model,
                    data,
                    snapshot.state,
                    mujoco_api.mjtState.mjSTATE_FULLPHYSICS,
                )
                mujoco_api.mj_forward(model, data)
                last_sequence = snapshot.sequence
                _record_viewer_progress(stats, mailbox, snapshot)
            viewer.sync(state_only=True)
            stop_event.wait(config.viewer_period_s)
        if not stop_event.is_set():
            raise RuntimeError("passive viewer closed unexpectedly")
    except BaseException as exc:
        failure_at = time.time_ns()
        stats.fatal.value = 1
        stats.failure_count.value = int(stats.failure_count.value) + 1
        stats.last_failure_at_ns.value = failure_at
        _set_shared_error(stats, f"{type(exc).__name__}: {exc}")
    finally:
        if viewer is not None:
            stats.viewer_shutdown_started.value = 1
            native_shutdown_requested.set()
            try:
                _close_and_wait_for_passive_viewer(
                    viewer,
                    simulate_ref=native_lifetime_ref,
                )
                stats.viewer_native_shutdown_complete.value = 1
                stats.viewer_native_active.value = 0
                native_shutdown_complete.set()
            except BaseException as exc:
                _abort_viewer_worker_without_python_finalizers(stats, exc)
            if shutdown_watchdog is not None:
                shutdown_watchdog.join(timeout=0.1)
        # The native lifetime barrier above is deliberately before these
        # refcount releases.  Keep all GL/model/data finalization in the child
        # that created them.
        if viewer is None or bool(stats.viewer_native_shutdown_complete.value):
            viewer = None
            data = None
            model = None


class _SharedWorkerStats:
    def __init__(self, context) -> None:
        self.ready = context.Value("b", 0, lock=False)
        self.fatal = context.Value("b", 0, lock=False)
        self.failure_count = context.Value("Q", 0, lock=False)
        self.last_failure_at_ns = context.Value("Q", 0, lock=False)
        self.last_success_at_ns = context.Value("Q", 0, lock=False)
        self.last_success_monotonic_ns = context.Value("Q", 0, lock=False)
        self.last_success_sequence = context.Value("Q", 0, lock=False)
        self.last_success_mailbox_dropped_count = context.Value(
            "Q", 0, lock=False
        )
        self.last_error = context.Array("c", 513, lock=True)
        self.worker_pid = context.Value("q", 0, lock=False)
        self.worker_boot_id = context.Array("c", 37, lock=True)
        stage_count = len(_RENDER_TIMING_STAGE_NAMES)
        self.render_timing_lock = context.Lock()
        self.render_timing_observed = context.Value("b", 0, lock=False)
        self.render_timing_last_ms = context.RawArray("d", stage_count)
        self.render_timing_max_ms = context.RawArray("d", stage_count)
        self.render_timing_over_threshold_count = context.RawArray(
            "Q", stage_count
        )
        self.render_timing_last_over_ms = context.RawArray("d", stage_count)
        self.render_timing_last_over_at_ns = context.RawArray("Q", stage_count)
        self.viewer_progress_lock = context.Lock()
        self.viewer_native_active = context.Value("b", 0, lock=False)
        self.viewer_shutdown_started = context.Value("b", 0, lock=False)
        self.viewer_native_shutdown_complete = context.Value(
            "b", 0, lock=False
        )


class RgbdRenderWorker:
    """Spawn-process supervisor; result collection never touches ``MjData``."""

    def __init__(
        self,
        *,
        mailbox: LatestStateMailbox,
        shape: MujocoStateShape,
        config: RenderWorkerConfig,
        on_frame: Callable[[RenderedFrame], None],
        context=None,
        target=_rgbd_worker_main,
    ) -> None:
        self._context = context or multiprocessing.get_context("spawn")
        self.mailbox = mailbox
        self.shape = shape
        self.config = config
        self._on_frame = on_frame
        self._stop_event = self._context.Event()
        self._result_queue = self._context.Queue(maxsize=1)
        self._stats = _SharedWorkerStats(self._context)
        self._process = self._context.Process(
            target=target,
            args=(
                mailbox,
                shape,
                config,
                self._result_queue,
                self._stop_event,
                self._stats,
            ),
            name="mujoco-rgbd-render-worker",
        )
        self._collector_stop = threading.Event()
        self._collector: threading.Thread | None = None
        self._status_lock = threading.Lock()
        self._last_error: str | None = None
        self._last_error_at_ns = 0
        self._collector_failure_count = 0
        self._collector_current_error: str | None = None
        self._last_collector_failure_at_ns = 0
        self._last_collector_error: str | None = None
        self._last_render_timing = _empty_render_timing_snapshot()
        self._closed = False

    def start(self) -> None:
        self._process.start()
        self._collector = threading.Thread(
            target=self._collect,
            daemon=True,
            name="mujoco-rgbd-result-collector",
        )
        self._collector.start()

    def _collect(self) -> None:
        while not self._collector_stop.is_set():
            try:
                result = self._result_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError) as exc:
                if not self._collector_stop.is_set() and not self._closed:
                    self._record_collector_failure(
                        f"collector queue failed: {type(exc).__name__}: {exc}"
                    )
                break
            if isinstance(result, RenderedFrame):
                self._collect_frame(result)
            elif isinstance(result, WorkerFailure):
                with self._status_lock:
                    self._last_error = result.message
                    self._last_error_at_ns = result.at_unix_ns
            else:
                self._record_collector_failure(
                    "collector received an unknown worker result"
                )
                break

    def _record_collector_failure(self, message: str) -> None:
        error = str(message)[:512]
        failure_at_ns = time.time_ns()
        with self._status_lock:
            self._last_error = error
            self._last_error_at_ns = failure_at_ns
            self._collector_failure_count += 1
            self._collector_current_error = error
            self._last_collector_failure_at_ns = failure_at_ns
            self._last_collector_error = error

    def _collect_frame(self, frame: RenderedFrame) -> None:
        try:
            self._on_frame(frame)
        except Exception as exc:
            self._record_collector_failure(
                f"frame commit failed: {type(exc).__name__}: {exc}"
            )
            return
        with self._status_lock:
            render_timing = _read_render_timing(self._stats)
            self._last_error = None
            self._collector_current_error = None
            if render_timing is not None:
                self._last_render_timing = render_timing

    def status(self) -> dict[str, object]:
        alive = self._process.is_alive()
        exitcode = self._process.exitcode
        collector_started = self._collector is not None
        collector_alive = bool(
            self._collector is not None and self._collector.is_alive()
        )
        fatal = bool(self._stats.fatal.value) or (
            exitcode is not None and exitcode != 0
        ) or (self._process.pid is not None and not alive and not self._closed) or (
            collector_started and not collector_alive and not self._closed
        )
        last_success_at = int(self._stats.last_success_at_ns.value)
        with self._status_lock:
            render_timing = _read_render_timing(self._stats)
            local_error = self._last_error
            local_error_at_ns = self._last_error_at_ns
            collector_failure_count = self._collector_failure_count
            collector_current_error = self._collector_current_error
            last_collector_failure_at_ns = self._last_collector_failure_at_ns
            last_collector_error = self._last_collector_error
            if render_timing is not None:
                self._last_render_timing = render_timing
            render_timing_status = _copy_render_timing_snapshot(
                self._last_render_timing
            )
        last_failure_at = max(
            int(self._stats.last_failure_at_ns.value),
            local_error_at_ns,
        )
        with self._stats.last_error.get_lock():
            shared_error = self._stats.last_error.value.decode(
                "utf-8",
                errors="replace",
            )
        with self._stats.worker_boot_id.get_lock():
            worker_boot_id = self._stats.worker_boot_id.value.decode("ascii")
        active_error = local_error or shared_error or None
        if fatal or active_error is not None:
            status = "failed"
        elif last_success_at:
            status = "running"
        else:
            status = "starting"
        return {
            "status": status,
            "ready": bool(self._stats.ready.value),
            "alive": alive,
            "collectorAlive": collector_alive,
            "exitCode": exitcode,
            "workerPid": int(self._stats.worker_pid.value) or None,
            "workerBootId": worker_boot_id or None,
            "lastSuccessSequence": int(self._stats.last_success_sequence.value),
            "lastSuccessAtUnixNs": last_success_at,
            "failureCount": (
                int(self._stats.failure_count.value) + collector_failure_count
            ),
            "lastFailureAtUnixNs": last_failure_at or None,
            "lastError": active_error,
            "collectorFailureCount": collector_failure_count,
            "currentCollectorError": collector_current_error,
            "lastCollectorFailureAtUnixNs": (
                last_collector_failure_at_ns or None
            ),
            "lastCollectorError": last_collector_error,
            "mailboxPublishedCount": self.mailbox.published_count,
            "mailboxDroppedCount": self.mailbox.dropped_count,
            "renderTiming": render_timing_status,
        }

    def close(self, *, timeout_s: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._process.pid is not None:
            self._process.join(timeout=max(0.0, float(timeout_s)))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._collector_stop.set()
        if self._collector is not None:
            self._collector.join(timeout=1.0)
        self._result_queue.close()
        self._result_queue.cancel_join_thread()


class PassiveViewerWorker:
    """Optional display process with a private model/data pair and no callbacks."""

    def __init__(
        self,
        *,
        mailbox: LatestStateMailbox,
        shape: MujocoStateShape,
        config: ViewerWorkerConfig,
        context=None,
        target=_viewer_worker_main,
    ) -> None:
        self._context = context or multiprocessing.get_context("spawn")
        self.mailbox = mailbox
        self.shape = shape
        self.config = config
        self._stop_event = self._context.Event()
        self._stats = _SharedWorkerStats(self._context)
        self._process = self._context.Process(
            target=target,
            args=(mailbox, shape, config, self._stop_event, self._stats),
            name="mujoco-passive-viewer-worker",
        )
        self._started_at_monotonic_ns = 0
        self._progress_cache_lock = threading.Lock()
        self._last_viewer_progress = (0, 0, 0, 0)
        self._closed = False

    def start(self) -> None:
        self._started_at_monotonic_ns = time.monotonic_ns()
        self._process.start()

    def status(self) -> dict[str, object]:
        alive = self._process.is_alive()
        process_failed = bool(self._stats.fatal.value) or (
            self._process.pid is not None
            and not alive
            and self._process.exitcode not in (None, 0)
        ) or (
            self._process.pid is not None and not alive and not self._closed
        )
        with self._stats.last_error.get_lock():
            shared_error = self._stats.last_error.value.decode(
                "utf-8",
                errors="replace",
            )
        with self._stats.worker_boot_id.get_lock():
            worker_boot_id = self._stats.worker_boot_id.value.decode("ascii")
        with self._progress_cache_lock:
            progress = _read_viewer_progress(self._stats)
            if progress is not None:
                self._last_viewer_progress = progress
            (
                last_success_at,
                last_success_monotonic_ns,
                last_success_sequence,
                dropped_at_success,
            ) = self._last_viewer_progress
        # Sample after the complete progress generation.  The viewer child may
        # commit while this status call is running; sampling first could make a
        # legitimate fresh commit appear to come from the future.
        now_monotonic_ns = time.monotonic_ns()
        mailbox_published = self.mailbox.published_count
        mailbox_dropped = self.mailbox.dropped_count
        progress_anchor_ns = (
            last_success_monotonic_ns or self._started_at_monotonic_ns
        )
        progress_age_ns = (
            None
            if progress_anchor_ns <= 0
            else now_monotonic_ns - progress_anchor_ns
        )
        progress_invalid = bool(
            progress_age_ns is not None and progress_age_ns < 0
        )
        failed = process_failed or progress_invalid
        parent_has_advanced = (
            mailbox_published > last_success_sequence
            or mailbox_dropped > dropped_at_success
        )
        stale = bool(
            not failed
            and alive
            and parent_has_advanced
            and progress_age_ns is not None
            and progress_age_ns > VIEWER_PROGRESS_STALE_NS
        )
        initialized = bool(self._stats.ready.value)
        status = (
            "failed"
            if failed
            else "stopped"
            if self._closed and not alive and self._process.exitcode == 0
            else "stale"
            if stale
            else "running"
            if initialized and last_success_at and alive
            else "starting"
        )
        return {
            "status": status,
            "ready": initialized and alive and not failed and not stale,
            "alive": alive,
            "exitCode": self._process.exitcode,
            "workerPid": int(self._stats.worker_pid.value) or None,
            "workerBootId": worker_boot_id or None,
            "lastSuccessSequence": last_success_sequence,
            "lastSuccessAtUnixNs": last_success_at,
            "progressAgeMs": (
                None
                if progress_age_ns is None
                else progress_age_ns / 1_000_000.0
            ),
            "staleAfterMs": VIEWER_PROGRESS_STALE_NS / 1_000_000.0,
            "failureCount": int(self._stats.failure_count.value),
            "lastFailureAtUnixNs": (
                int(self._stats.last_failure_at_ns.value) or None
            ),
            "mailboxPublishedCount": mailbox_published,
            "mailboxDroppedCount": mailbox_dropped,
            "lastSuccessMailboxDroppedCount": dropped_at_success,
            "authorityFeedbackEnabled": False,
            "keyCallbackRegistered": False,
            "nativeViewerActive": bool(
                self._stats.viewer_native_active.value
            ),
            "shutdownStarted": bool(
                self._stats.viewer_shutdown_started.value
            ),
            "nativeShutdownComplete": bool(
                self._stats.viewer_native_shutdown_complete.value
            ),
            "lastError": (
                shared_error
                or (
                    "viewer monotonic progress age is negative"
                    if progress_invalid
                    else None
                )
            ),
        }

    def close(self, *, timeout_s: float = 3.0) -> None:
        if self._closed:
            return
        self._stop_event.set()
        if self._process.pid is not None:
            cooperative_timeout = max(
                0.0,
                float(timeout_s),
                VIEWER_NATIVE_SHUTDOWN_TIMEOUT_S
                + VIEWER_NATIVE_SHUTDOWN_PARENT_MARGIN_S,
            )
            self._process.join(timeout=cooperative_timeout)
            if self._process.is_alive():
                failure_at = time.time_ns()
                self._stats.fatal.value = 1
                self._stats.failure_count.value = (
                    int(self._stats.failure_count.value) + 1
                )
                self._stats.last_failure_at_ns.value = failure_at
                _set_shared_error(
                    self._stats,
                    "passive viewer exceeded its cooperative native shutdown "
                    f"deadline ({cooperative_timeout:.3f}s); forced SIGKILL",
                )
                # The viewer is process-isolated and has no control authority.
                # After the native grace period, SIGKILL is safer than SIGTERM:
                # it cannot enter Python, GLFW, or X11 finalizers while their
                # render thread is wedged.
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
                self._process.join(timeout=1.0)
                if self._process.is_alive():
                    raise RuntimeError(
                        "passive viewer remained alive after forced SIGKILL"
                    )
                self._stats.viewer_native_active.value = 0
        # Do not make a failed close permanently non-retryable.  In particular,
        # a live non-daemon multiprocessing child would otherwise outlive Plant
        # cleanup and can block interpreter shutdown indefinitely.
        self._closed = True


__all__ = [
    "AuthoritativeSnapshot",
    "CAPTURE_STATE_NAME",
    "LatestStateMailbox",
    "MAX_CAPTURE_AGE_NS",
    "MujocoStateShape",
    "PassiveViewerWorker",
    "RENDER_TIMING_STALL_THRESHOLD_MS",
    "RenderedFrame",
    "RenderWorkerConfig",
    "RgbdRenderWorker",
    "VIEWER_PROGRESS_STALE_NS",
    "VIEWER_NATIVE_SHUTDOWN_TIMEOUT_S",
    "ViewerWorkerConfig",
    "WorkerFailure",
    "configure_head_camera",
    "configure_tracking_viewer",
]
