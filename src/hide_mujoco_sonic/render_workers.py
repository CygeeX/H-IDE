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

import numpy as np

from .frames import BodyTwist, read_mujoco_local_body_twist


CAPTURE_STATE_NAME = "mjSTATE_FULLPHYSICS"
MAX_CAPTURE_AGE_NS = 500_000_000
RGB_WIDTH = 640
RGB_HEIGHT = 480


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
    ) -> int | None:
        """Copy one state or drop it immediately when the worker is reading."""

        captured_at = int(captured_at_unix_ns)
        if captured_at <= 0:
            raise ValueError("captured_at_unix_ns must be positive")
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
            state = np.frombuffer(self._state, dtype=np.float64).copy()
        finally:
            self._lock.release()
        return AuthoritativeSnapshot(
            producer_boot_id=self.producer_boot_id,
            sequence=sequence,
            captured_at_unix_ns=captured_at,
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
    render_started_at_unix_ns: int
    rendered_at_unix_ns: int
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
    apply_started = time.perf_counter()
    mujoco_api.mj_setState(
        model,
        data,
        snapshot.state,
        mujoco_api.mjtState.mjSTATE_FULLPHYSICS,
    )
    mujoco_api.mj_forward(model, data)
    state_apply_ms = (time.perf_counter() - apply_started) * 1000.0

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
    return RenderedFrame(
        producer_boot_id=snapshot.producer_boot_id,
        sequence=snapshot.sequence,
        captured_at_unix_ns=snapshot.captured_at_unix_ns,
        render_started_at_unix_ns=render_started_ns,
        rendered_at_unix_ns=rendered_at_ns,
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
            ("totalMs", (rendered_at_ns - render_started_ns) / 1_000_000.0),
            (
                "queueAgeAtStartMs",
                max(
                    0.0,
                    (render_started_ns - snapshot.captured_at_unix_ns) / 1_000_000.0,
                ),
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
        stats.last_success_at_ns.value = time.time_ns()
        stats.last_success_sequence.value = first.sequence
        from mujoco import viewer as mujoco_viewer

        # No key_callback is registered.  Mouse perturbations, pause state and
        # all viewer-side edits apply only to this worker-private MjData.
        viewer = mujoco_viewer.launch_passive(model, data)
        configure_tracking_viewer(mujoco_api, model, viewer, runtime_scene)
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
                stats.last_success_at_ns.value = time.time_ns()
                stats.last_success_sequence.value = snapshot.sequence
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
            try:
                viewer.close()
            except Exception:
                pass
        # Keep all GL/model/data finalization in the child that created it.
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
        self.last_success_sequence = context.Value("Q", 0, lock=False)
        self.last_error = context.Array("c", 513, lock=True)
        self.worker_pid = context.Value("q", 0, lock=False)
        self.worker_boot_id = context.Array("c", 37, lock=True)


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
            except (EOFError, OSError, ValueError):
                break
            if isinstance(result, RenderedFrame):
                try:
                    self._on_frame(result)
                    with self._status_lock:
                        self._last_error = None
                except Exception as exc:
                    with self._status_lock:
                        self._last_error = (
                            f"frame commit failed: {type(exc).__name__}: {exc}"[:512]
                        )
                        self._last_error_at_ns = time.time_ns()
            elif isinstance(result, WorkerFailure):
                with self._status_lock:
                    self._last_error = result.message
                    self._last_error_at_ns = result.at_unix_ns

    def status(self) -> dict[str, object]:
        alive = self._process.is_alive()
        exitcode = self._process.exitcode
        fatal = bool(self._stats.fatal.value) or (
            exitcode is not None and exitcode != 0
        ) or (self._process.pid is not None and not alive and not self._closed)
        last_success_at = int(self._stats.last_success_at_ns.value)
        with self._status_lock:
            local_error = self._last_error
            local_error_at_ns = self._last_error_at_ns
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
        if fatal or (last_failure_at > last_success_at):
            status = "failed"
        elif last_success_at:
            status = "running"
        else:
            status = "starting"
        return {
            "status": status,
            "ready": bool(self._stats.ready.value),
            "alive": alive,
            "exitCode": exitcode,
            "workerPid": int(self._stats.worker_pid.value) or None,
            "workerBootId": worker_boot_id or None,
            "lastSuccessSequence": int(self._stats.last_success_sequence.value),
            "lastSuccessAtUnixNs": last_success_at,
            "failureCount": int(self._stats.failure_count.value),
            "lastFailureAtUnixNs": last_failure_at or None,
            "lastError": local_error or shared_error or None,
            "mailboxPublishedCount": self.mailbox.published_count,
            "mailboxDroppedCount": self.mailbox.dropped_count,
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
        self._closed = False

    def start(self) -> None:
        self._process.start()

    def status(self) -> dict[str, object]:
        alive = self._process.is_alive()
        failed = bool(self._stats.fatal.value) or (
            self._process.pid is not None and not alive and not self._closed
        )
        with self._stats.last_error.get_lock():
            shared_error = self._stats.last_error.value.decode(
                "utf-8",
                errors="replace",
            )
        with self._stats.worker_boot_id.get_lock():
            worker_boot_id = self._stats.worker_boot_id.value.decode("ascii")
        return {
            "status": (
                "failed"
                if failed
                else (
                    "running"
                    if int(self._stats.last_success_at_ns.value)
                    else "starting"
                )
            ),
            "ready": bool(self._stats.ready.value),
            "alive": alive,
            "exitCode": self._process.exitcode,
            "workerPid": int(self._stats.worker_pid.value) or None,
            "workerBootId": worker_boot_id or None,
            "lastSuccessSequence": int(self._stats.last_success_sequence.value),
            "lastSuccessAtUnixNs": int(self._stats.last_success_at_ns.value),
            "failureCount": int(self._stats.failure_count.value),
            "lastFailureAtUnixNs": (
                int(self._stats.last_failure_at_ns.value) or None
            ),
            "mailboxPublishedCount": self.mailbox.published_count,
            "mailboxDroppedCount": self.mailbox.dropped_count,
            "authorityFeedbackEnabled": False,
            "keyCallbackRegistered": False,
            "lastError": shared_error or None,
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


__all__ = [
    "AuthoritativeSnapshot",
    "CAPTURE_STATE_NAME",
    "LatestStateMailbox",
    "MAX_CAPTURE_AGE_NS",
    "MujocoStateShape",
    "PassiveViewerWorker",
    "RenderedFrame",
    "RenderWorkerConfig",
    "RgbdRenderWorker",
    "ViewerWorkerConfig",
    "WorkerFailure",
    "configure_head_camera",
    "configure_tracking_viewer",
]
