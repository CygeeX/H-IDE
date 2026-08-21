from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import inspect
import math
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import uuid
import weakref

import numpy as np
import pytest

import hide_mujoco_sonic.render_workers as render_workers
from hide_mujoco_sonic.frames import BodyTwist
from hide_mujoco_sonic.plant import (
    PlantDynamicsEvidence,
    PlantFrameStore,
    PlantPhysicsFault,
    PlantPhysicsMonitor,
    PlantSupportController,
)
from hide_mujoco_sonic.render_workers import (
    LatestStateMailbox,
    MAX_CAPTURE_AGE_NS,
    MujocoStateShape,
    PassiveViewerWorker,
    RENDER_TIMING_STALL_THRESHOLD_MS,
    RenderedFrame,
    RenderWorkerConfig,
    RgbdRenderWorker,
    VIEWER_PROGRESS_STALE_NS,
    ViewerWorkerConfig,
    _SharedWorkerStats,
    _clear_shared_error,
    _encode_rgb_jpeg,
    _initialize_worker_identity,
    _read_render_timing,
    _record_render_timing,
    _record_viewer_progress,
    _set_shared_error,
    _viewer_worker_main,
)


MINIMAL_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <camera name="head_camera" pos="0.2 0 0.2"/>
    </body>
  </worldbody>
</mujoco>
"""
ACTUATED_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="link" pos="0 0 -0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
</mujoco>
"""
ROOT = Path(__file__).resolve().parents[1]


def _model_data():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(MINIMAL_XML)
    data = mujoco.MjData(model)
    data.qpos[:] = (0.25, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0)
    data.qvel[:] = (0.1, 0.2, -0.1, 0.01, 0.02, 0.03)
    data.time = 1.25
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_dynamics_evidence_is_complete_immutable_and_diagnostic_only() -> None:
    mujoco, model, data = _model_data()
    pelvis_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )
    data.xfrc_applied[pelvis_id] = (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)
    evidence = PlantDynamicsEvidence(
        mujoco,
        model,
        data,
        support_band=SimpleNamespace(enable=True),
        attached_body_id=pelvis_id,
    )

    evidence.observe(captured_at_unix_ns=123)
    state = evidence.capture_state()
    assert state["schemaVersion"] == "h-ide.mujoco-dynamics-evidence/1"
    assert state["status"] == "running"
    assert state["diagnosticOnly"] is True
    assert state["authorityFeedbackEnabled"] is False
    assert state["capturedAtUnixNs"] == 123
    assert state["simulationTimeS"] == 1.25
    assert state["supportBandEnabled"] is True
    assert state["attachedBody"] == "pelvis"
    assert state["lastAppliedWrench"]["frameId"] == "world"
    assert state["lastAppliedWrench"]["values"] == [
        1.0, -2.0, 3.0, -4.0, 5.0, -6.0,
    ]
    assert state["rootState"]["qpos"] == pytest.approx(
        [0.25, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0]
    )
    assert state["rootState"]["qvel"] == pytest.approx(
        [0.1, 0.2, -0.1, 0.01, 0.02, 0.03]
    )
    assert len(state["rootState"]["qacc"]) == 6
    assert len(state["rootState"]["qaccWarmstart"]) == 6
    assert math.isfinite(state["energy"]["potentialJ"])
    assert math.isfinite(state["energy"]["kineticJ"])
    assert state["energy"]["totalJ"] == pytest.approx(
        state["energy"]["potentialJ"] + state["energy"]["kineticJ"]
    )
    assert state["warningCounts"] == {
        "BADQPOS": 0,
        "BADQVEL": 0,
        "BADQACC": 0,
        "BADCTRL": 0,
    }
    assert "oracle" not in str(state).lower()

    # HTTP may read only the published immutable sample, never live MjData.
    data.qpos[0] = 99.0
    data.xfrc_applied[pelvis_id, 0] = 99.0
    unchanged = evidence.capture_state()
    assert unchanged["rootState"]["qpos"][0] == 0.25
    assert unchanged["lastAppliedWrench"]["values"][0] == 1.0


def _slow_reader(mailbox, ready, release, output) -> None:
    first = None
    while first is None:
        first = mailbox.copy_latest(after_sequence=0)
        if first is None:
            time.sleep(0.001)
    ready.set()
    release.wait(5.0)
    latest = mailbox.copy_latest(after_sequence=first.sequence)
    output.put(None if latest is None else latest.sequence)


def _crash_worker(
    _mailbox,
    _shape,
    _config,
    _result_queue,
    _stop_event,
    stats,
) -> None:
    stats.fatal.value = 1
    stats.last_failure_at_ns.value = time.time_ns()


def _cooperative_worker(
    _mailbox,
    _shape,
    _config,
    _result_queue,
    stop_event,
    stats,
) -> None:
    stats.ready.value = 1
    stop_event.wait(10.0)


def _stubborn_worker(
    _mailbox,
    _shape,
    _config,
    _result_queue,
    _stop_event,
    stats,
) -> None:
    stats.ready.value = 1
    while True:
        time.sleep(0.1)


def _healthy_viewer_identity_worker(
    _mailbox,
    _shape,
    _config,
    stop_event,
    stats,
) -> None:
    _initialize_worker_identity(stats)
    first = None
    while first is None and not stop_event.is_set():
        first = _mailbox.copy_latest(after_sequence=0)
        if first is None:
            stop_event.wait(0.001)
    if first is None:
        return
    _record_viewer_progress(stats, _mailbox, first)
    stats.ready.value = 1
    stop_event.wait(10.0)


class _FakeProcess:
    def __init__(
        self,
        *,
        exit_on_join: bool,
        exit_on_kill: bool = True,
    ) -> None:
        self.pid = 12345
        self.exitcode = None
        self._alive = True
        self._exit_on_join = exit_on_join
        self._exit_on_kill = exit_on_kill
        self.join_timeouts: list[float | None] = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)
        if self._exit_on_join:
            self._alive = False
            self.exitcode = 0

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False
        self.exitcode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        if self._exit_on_kill:
            self._alive = False
            self.exitcode = -9


class _FakeNativeViewer:
    @staticmethod
    def exit() -> None:
        return None


class _FakePassiveViewer:
    def __init__(self) -> None:
        self._native = _FakeNativeViewer()
        self._sim = weakref.ref(self._native)
        self.close_calls = 0

    @property
    def native_alive(self) -> bool:
        return self._sim() is not None

    @native_alive.setter
    def native_alive(self, value: bool) -> None:
        if value:
            if self._sim() is None:
                self._native = _FakeNativeViewer()
                self._sim = weakref.ref(self._native)
            return
        self._native = None

    def close(self) -> None:
        self.close_calls += 1


def _slow_then_normal_timing_worker(
    mailbox,
    _shape,
    _config,
    _result_queue,
    stop_event,
    stats,
) -> None:
    _initialize_worker_identity(stats)
    captured_at_ns = time.time_ns()
    _record_render_timing(
        stats,
        _frame(
            1,
            captured_at_ns=captured_at_ns,
            boot=mailbox.producer_boot_id,
            render_timing=_render_timing(total_ms=250.0, rgb_ms=125.0),
        ),
    )
    _record_render_timing(
        stats,
        _frame(
            2,
            captured_at_ns=captured_at_ns + 100,
            boot=mailbox.producer_boot_id,
            render_timing=_render_timing(total_ms=10.0, rgb_ms=5.0),
        ),
    )
    stats.last_success_at_ns.value = time.time_ns()
    stats.last_success_sequence.value = 2
    stats.ready.value = 1
    stop_event.wait(10.0)


def _render_timing(
    *,
    total_ms: float = 0.001,
    rgb_ms: float = 0.001,
    jpeg_ms: float = 0.001,
    depth_ms: float = 0.001,
    state_apply_ms: float = 0.001,
    mj_forward_ms: float = 0.001,
    queue_age_ms: float = 0.001,
) -> tuple[tuple[str, float], ...]:
    return (
        ("stateApplyMs", state_apply_ms),
        ("rgbRenderMs", rgb_ms),
        ("depthRenderMs", depth_ms),
        ("jpegEncodeMs", jpeg_ms),
        ("totalMs", total_ms),
        ("mjForwardMs", mj_forward_ms),
        ("queueAgeAtStartMs", queue_age_ms),
    )


def _frame(
    sequence: int,
    *,
    captured_at_ns: int,
    boot: str,
    captured_at_monotonic_ns: int | None = None,
    render_timing: tuple[tuple[str, float], ...] | None = None,
) -> RenderedFrame:
    captured_monotonic_ns = (
        time.monotonic_ns()
        if captured_at_monotonic_ns is None
        else captured_at_monotonic_ns
    )
    return RenderedFrame(
        producer_boot_id=boot,
        sequence=sequence,
        captured_at_unix_ns=captured_at_ns,
        captured_at_monotonic_ns=captured_monotonic_ns,
        render_started_at_unix_ns=captured_at_ns + 10,
        render_started_at_monotonic_ns=captured_monotonic_ns + 10,
        rendered_at_unix_ns=captured_at_ns + 20,
        rendered_at_monotonic_ns=captured_monotonic_ns + 20,
        simulation_time_s=float(sequence),
        pose=(0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
        root_velocity=BodyTwist.zero(),
        camera_pose=(0.2, 0.0, 1.1, 1.0, 0.0, 0.0, 0.0),
        intrinsics=(("fx", 100.0), ("fy", 100.0), ("cx", 1.0), ("cy", 0.5)),
        width=2,
        height=1,
        rgb_jpeg=f"jpeg-{sequence}".encode(),
        depth_z16_le=bytes((sequence, 0, sequence + 1, 0)),
        render_timing=render_timing or _render_timing(),
    )


def test_fullphysics_roundtrip_uses_an_independent_mjdata() -> None:
    mujoco, model, authority = _model_data()
    shape = MujocoStateShape.from_model(mujoco, model)
    mailbox = LatestStateMailbox(shape)

    assert mailbox.try_publish(
        mujoco,
        model,
        authority,
        captured_at_unix_ns=time.time_ns(),
        captured_at_monotonic_ns=123_456_789,
    ) == 1
    snapshot = mailbox.copy_latest()
    assert snapshot is not None
    assert snapshot.captured_at_monotonic_ns == 123_456_789

    replica = mujoco.MjData(model)
    mujoco.mj_setState(
        model,
        replica,
        snapshot.state,
        mujoco.mjtState.mjSTATE_FULLPHYSICS,
    )
    mujoco.mj_forward(model, replica)

    np.testing.assert_array_equal(replica.qpos, authority.qpos)
    np.testing.assert_array_equal(replica.qvel, authority.qvel)
    np.testing.assert_array_equal(replica.xpos, authority.xpos)
    np.testing.assert_array_equal(replica.xquat, authority.xquat)
    np.testing.assert_array_equal(replica.cam_xpos, authority.cam_xpos)
    np.testing.assert_array_equal(replica.cam_xmat, authority.cam_xmat)
    assert replica.time == authority.time
    replica.qpos[0] += 10.0
    assert authority.qpos[0] == pytest.approx(0.25)


def test_mailbox_is_latest_only_and_authority_never_waits_for_its_lock() -> None:
    mujoco, model, data = _model_data()
    mailbox = LatestStateMailbox(MujocoStateShape.from_model(mujoco, model))
    first = mailbox.try_publish(
        mujoco, model, data, captured_at_unix_ns=time.time_ns()
    )
    data.time = 2.0
    second = mailbox.try_publish(
        mujoco, model, data, captured_at_unix_ns=time.time_ns()
    )
    assert (first, second) == (1, 2)
    assert mailbox.copy_latest(after_sequence=0).sequence == 2  # type: ignore[union-attr]
    assert mailbox.copy_latest(after_sequence=2) is None

    assert mailbox._lock.acquire(False)
    started = time.perf_counter()
    try:
        assert mailbox.try_publish(
            mujoco, model, data, captured_at_unix_ns=time.time_ns()
        ) is None
    finally:
        mailbox._lock.release()
    assert time.perf_counter() - started < 0.05
    assert mailbox.dropped_count == 1


def test_slow_consumer_does_not_delay_snapshot_publisher_and_gets_latest() -> None:
    mujoco, model, data = _model_data()
    context = multiprocessing.get_context("spawn")
    mailbox = LatestStateMailbox(
        MujocoStateShape.from_model(mujoco, model),
        context=context,
    )
    mailbox.try_publish(mujoco, model, data, captured_at_unix_ns=time.time_ns())
    ready, release = context.Event(), context.Event()
    output = context.Queue(maxsize=1)
    process = context.Process(target=_slow_reader, args=(mailbox, ready, release, output))
    process.start()
    try:
        assert ready.wait(5.0)
        started = time.perf_counter()
        last_sequence = 0
        for index in range(200):
            data.time = 2.0 + index * 0.001
            published = mailbox.try_publish(
                mujoco,
                model,
                data,
                captured_at_unix_ns=time.time_ns(),
            )
            if published is not None:
                last_sequence = published
        assert time.perf_counter() - started < 0.5
        release.set()
        assert output.get(timeout=5.0) == last_sequence
    finally:
        release.set()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
        output.close()
    assert process.exitcode == 0


def test_unsupported_future_state_is_rejected_before_worker_start() -> None:
    with pytest.raises(RuntimeError, match="nmocap=1"):
        MujocoStateShape(
            nq=7,
            nv=6,
            na=0,
            nhistory=0,
            nmocap=1,
            npluginstate=0,
            nflex=0,
            nskin=0,
            fullphysics_size=14,
        ).require_supported()


def test_parent_physics_monitor_latches_nonfinite_state_without_rendering() -> None:
    mujoco, model, data = _model_data()
    support = PlantSupportController()
    support.finish_startup()
    monitor = PlantPhysicsMonitor(
        SimpleNamespace(
            runtime_scene=SimpleNamespace(
                bounds_min=(-1.0, -1.0, 0.0),
                bounds_max=(1.0, 1.0, 2.0),
            )
        ),
        mujoco,
        model,
        data,
        support,
    )
    monitor.observe()
    data.time += 0.004
    data.qvel[0] = float("nan")
    with pytest.raises(PlantPhysicsFault, match="non-finite"):
        monitor.observe()
    assert support.state()["fault"] == "MuJoCo physics state contains non-finite values"


def test_parent_physics_monitor_catches_warning_after_mj_step_self_heals() -> None:
    mujoco, model, data = _model_data()
    support = PlantSupportController()
    support.finish_startup()
    monitor = PlantPhysicsMonitor(
        SimpleNamespace(
            runtime_scene=SimpleNamespace(
                bounds_min=(-1.0, -1.0, 0.0),
                bounds_max=(1.0, 1.0, 2.0),
            )
        ),
        mujoco,
        model,
        data,
        support,
    )
    warning_index = int(mujoco.mjtWarning.mjWARN_BADQPOS)
    warning_before = int(data.warning[warning_index].number)
    data.qpos[0] = float("nan")

    mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert np.all(np.isfinite(data.qacc))
    assert int(data.warning[warning_index].number) == warning_before + 1
    with pytest.raises(PlantPhysicsFault, match="BADQPOS warning count increased"):
        monitor.observe()


@pytest.mark.parametrize(
    "warning_name",
    ["mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL"],
)
def test_parent_physics_monitor_rejects_every_nonzero_initial_bad_warning(
    warning_name: str,
) -> None:
    mujoco, model, data = _model_data()
    warning_index = int(getattr(mujoco.mjtWarning, warning_name))
    data.warning[warning_index].number = 7
    support = PlantSupportController()
    support.finish_startup()

    warning_label = warning_name.removeprefix("mjWARN_")
    with pytest.raises(
        PlantPhysicsFault,
        match=rf"non-zero before the first Plant step: .*{warning_label}=7",
    ):
        PlantPhysicsMonitor(
            SimpleNamespace(
                runtime_scene=SimpleNamespace(
                    bounds_min=(-1.0, -1.0, 0.0),
                    bounds_max=(1.0, 1.0, 2.0),
                )
            ),
            mujoco,
            model,
            data,
            support,
        )


@pytest.mark.parametrize(
    "warning_name",
    ["mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC", "mjWARN_BADCTRL"],
)
def test_parent_physics_monitor_detects_every_post_start_bad_warning(
    warning_name: str,
) -> None:
    mujoco, model, data = _model_data()
    support = PlantSupportController()
    support.finish_startup()
    monitor = PlantPhysicsMonitor(
        SimpleNamespace(
            runtime_scene=SimpleNamespace(
                bounds_min=(-1.0, -1.0, 0.0),
                bounds_max=(1.0, 1.0, 2.0),
            )
        ),
        mujoco,
        model,
        data,
        support,
    )
    monitor.observe()
    data.time += 0.004
    warning_index = int(getattr(mujoco.mjtWarning, warning_name))
    data.warning[warning_index].number = 1

    warning_label = warning_name.removeprefix("mjWARN_")
    with pytest.raises(
        PlantPhysicsFault,
        match=rf"{warning_label} warning count increased \(0 -> 1\)",
    ):
        monitor.observe()


@pytest.mark.parametrize("field", ["qacc", "ctrl", "actuator_force"])
def test_parent_physics_monitor_rejects_each_nonfinite_dynamic_array(
    field: str,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ACTUATED_XML)
    data = mujoco.MjData(model)
    data.qpos[:7] = (0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    support = PlantSupportController()
    support.finish_startup()
    monitor = PlantPhysicsMonitor(
        SimpleNamespace(
            runtime_scene=SimpleNamespace(
                bounds_min=(-1.0, -1.0, 0.0),
                bounds_max=(1.0, 1.0, 2.0),
            )
        ),
        mujoco,
        model,
        data,
        support,
    )
    getattr(data, field)[0] = float("nan")

    with pytest.raises(PlantPhysicsFault, match="non-finite"):
        monitor.observe()


def test_jpeg_failure_is_an_explicit_worker_error(monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    monkeypatch.setattr(cv2, "imencode", lambda *_args, **_kwargs: (False, None))
    with pytest.raises(RuntimeError, match="JPEG encoder"):
        _encode_rgb_jpeg(np.zeros((1, 2, 3), dtype=np.uint8), quality=85)


def test_worker_crash_does_not_disable_authoritative_snapshot_publication() -> None:
    mujoco, model, data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    mailbox = LatestStateMailbox(shape, context=context)
    committed: list[RenderedFrame] = []
    worker = RgbdRenderWorker(
        mailbox=mailbox,
        shape=shape,
        config=RenderWorkerConfig("missing", "missing", "explicit_process_restart", 85),
        on_frame=committed.append,
        context=context,
        target=_crash_worker,
    )
    worker.start()
    worker._process.join(timeout=5.0)
    assert worker.status()["status"] == "failed"
    assert mailbox.try_publish(
        mujoco, model, data, captured_at_unix_ns=time.time_ns()
    ) == 1
    assert committed == []
    worker.close()
    worker.close()


def test_gl_initialization_failure_is_confined_to_worker_process() -> None:
    mujoco, model, data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    mailbox = LatestStateMailbox(shape, context=context)
    worker = RgbdRenderWorker(
        mailbox=mailbox,
        shape=shape,
        config=RenderWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            85,
            gl_backend="not-a-mujoco-backend",
        ),
        on_frame=lambda _frame: None,
        context=context,
    )
    worker.start()
    worker._process.join(timeout=10.0)
    failed_status = worker.status()
    assert failed_status["status"] == "failed"
    assert failed_status["lastError"]
    assert failed_status["failureCount"] == 1
    assert failed_status["workerPid"] == worker._process.pid
    uuid.UUID(str(failed_status["workerBootId"]))
    assert mailbox.try_publish(
        mujoco, model, data, captured_at_unix_ns=time.time_ns()
    ) == 1
    worker.close()


def test_office_v2_worker_reloads_verified_scene_and_emits_one_atomic_rgbd() -> None:
    mujoco = pytest.importorskip("mujoco")
    from hide_mujoco_sonic.scene_runtime import load_runtime_scene

    runtime = load_runtime_scene(ROOT / "scenes/office_v2/generated/runtime_manifest.json")
    model = mujoco.MjModel.from_xml_path(str(runtime.model_scene))
    authority = mujoco.MjData(model)
    mujoco.mj_forward(model, authority)
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    boot = str(uuid.uuid4())
    mailbox = LatestStateMailbox(shape, producer_boot_id=boot, context=context)
    committed: list[RenderedFrame] = []
    committed_event = threading.Event()

    def commit(frame: RenderedFrame) -> None:
        committed.append(frame)
        committed_event.set()

    worker = RgbdRenderWorker(
        mailbox=mailbox,
        shape=shape,
        config=RenderWorkerConfig(
            runtime_manifest=str(runtime.manifest_path),
            model_scene=str(runtime.model_scene),
            reset_policy="explicit_process_restart",
            jpeg_quality=85,
        ),
        on_frame=commit,
        context=context,
    )
    worker.start()
    try:
        known_capture_monotonic_ns = time.monotonic_ns()
        assert mailbox.try_publish(
            mujoco,
            model,
            authority,
            captured_at_unix_ns=time.time_ns(),
            captured_at_monotonic_ns=known_capture_monotonic_ns,
        ) == 1
        assert committed_event.wait(30.0), worker.status()
        frame = committed[-1]
        assert (frame.producer_boot_id, frame.sequence) == (boot, 1)
        assert frame.captured_at_monotonic_ns == known_capture_monotonic_ns
        assert (frame.width, frame.height) == (640, 480)
        assert frame.rgb_jpeg.startswith(b"\xff\xd8")
        assert len(frame.depth_z16_le) == 640 * 480 * 2
        assert frame.pose == pytest.approx(tuple(float(value) for value in authority.qpos[:7]))
        assert frame.rendered_at_unix_ns >= frame.render_started_at_unix_ns
        assert (
            frame.captured_at_monotonic_ns
            <= frame.render_started_at_monotonic_ns
            <= frame.rendered_at_monotonic_ns
        )
        assert set(dict(frame.render_timing)) == {
            "stateApplyMs",
            "rgbRenderMs",
            "depthRenderMs",
            "jpegEncodeMs",
            "totalMs",
            "mjForwardMs",
            "queueAgeAtStartMs",
        }
        status = worker.status()
        assert status["status"] == "running"
        assert status["workerPid"] == worker._process.pid
        uuid.UUID(str(status["workerBootId"]))
        timing_status = status["renderTiming"]
        assert timing_status["thresholdMs"] == 100.0
        assert set(timing_status["stages"]) == set(dict(frame.render_timing))
        assert timing_status["stages"]["totalMs"]["lastMs"] == pytest.approx(
            dict(frame.render_timing)["totalMs"]
        )
        support = PlantSupportController()
        support.finish_startup()
        store = PlantFrameStore(
            SimpleNamespace(robot_id="g1-sim", localization_epoch_id="test-epoch"),
            support,
            worker_status_provider=lambda: {
                "status": "running",
                "ready": True,
                "alive": True,
            },
        )
        store.commit(frame)
        assert store.state()["capturedAtMonotonicNs"] == known_capture_monotonic_ns
    finally:
        worker.close(timeout_s=5.0)


def test_worker_timing_persists_slow_evidence_across_normal_frames() -> None:
    context = multiprocessing.get_context("spawn")
    stats = _SharedWorkerStats(context)
    boot = str(uuid.uuid4())
    captured_at_ns = time.time_ns()
    slow = _frame(
        1,
        captured_at_ns=captured_at_ns,
        boot=boot,
        render_timing=_render_timing(total_ms=250.0, rgb_ms=125.0),
    )
    normal = _frame(
        2,
        captured_at_ns=captured_at_ns + 100,
        boot=boot,
        render_timing=_render_timing(total_ms=10.0, rgb_ms=5.0),
    )

    _record_render_timing(stats, slow)
    slow_status = _read_render_timing(stats)
    assert slow_status is not None
    assert slow_status["stages"]["totalMs"] == {
        "lastMs": 250.0,
        "maxMs": 250.0,
        "overThresholdCount": 1,
        "lastOverThresholdMs": 250.0,
        "lastOverThresholdAtUnixNs": slow.rendered_at_unix_ns,
    }

    _record_render_timing(stats, normal)
    recovered_status = _read_render_timing(stats)
    assert recovered_status is not None
    assert recovered_status["stages"]["totalMs"] == {
        "lastMs": 10.0,
        "maxMs": 250.0,
        "overThresholdCount": 1,
        "lastOverThresholdMs": 250.0,
        "lastOverThresholdAtUnixNs": slow.rendered_at_unix_ns,
    }
    assert recovered_status["stages"]["rgbRenderMs"] == {
        "lastMs": 5.0,
        "maxMs": 125.0,
        "overThresholdCount": 1,
        "lastOverThresholdMs": 125.0,
        "lastOverThresholdAtUnixNs": slow.rendered_at_unix_ns,
    }


def test_worker_timing_exact_threshold_is_normal_and_new_worker_starts_clean() -> None:
    context = multiprocessing.get_context("spawn")
    first_worker_stats = _SharedWorkerStats(context)
    frame = _frame(
        1,
        captured_at_ns=time.time_ns(),
        boot=str(uuid.uuid4()),
        render_timing=_render_timing(total_ms=RENDER_TIMING_STALL_THRESHOLD_MS),
    )
    _record_render_timing(first_worker_stats, frame)
    first_status = _read_render_timing(first_worker_stats)
    assert first_status is not None
    assert first_status["stages"]["totalMs"] == {
        "lastMs": RENDER_TIMING_STALL_THRESHOLD_MS,
        "maxMs": RENDER_TIMING_STALL_THRESHOLD_MS,
        "overThresholdCount": 0,
        "lastOverThresholdMs": None,
        "lastOverThresholdAtUnixNs": None,
    }

    restarted_worker_stats = _SharedWorkerStats(context)
    restarted_status = _read_render_timing(restarted_worker_stats)
    assert restarted_status is not None
    assert restarted_status["stages"]["totalMs"] == {
        "lastMs": None,
        "maxMs": 0.0,
        "overThresholdCount": 0,
        "lastOverThresholdMs": None,
        "lastOverThresholdAtUnixNs": None,
    }


def test_worker_status_exposes_defensive_persistent_timing_snapshot() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = RgbdRenderWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=RenderWorkerConfig(
            "missing", "missing", "explicit_process_restart", 85
        ),
        on_frame=lambda _frame: None,
        context=context,
    )
    try:
        frame = _frame(
            1,
            captured_at_ns=time.time_ns(),
            boot=str(uuid.uuid4()),
            render_timing=_render_timing(total_ms=250.0),
        )
        _record_render_timing(worker._stats, frame)
        first = worker.status()["renderTiming"]
        assert first["thresholdMs"] == RENDER_TIMING_STALL_THRESHOLD_MS
        assert first["stages"]["totalMs"]["overThresholdCount"] == 1
        first["stages"]["totalMs"]["overThresholdCount"] = 999
        assert worker.status()["renderTiming"]["stages"]["totalMs"][
            "overThresholdCount"
        ] == 1
    finally:
        worker.close()


def test_parent_timing_cache_reads_and_publishes_under_one_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = RgbdRenderWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=RenderWorkerConfig(
            "missing", "missing", "explicit_process_restart", 85
        ),
        on_frame=lambda _frame: None,
        context=context,
    )
    first = render_workers._empty_render_timing_snapshot()
    second = render_workers._empty_render_timing_snapshot()
    first["stages"]["totalMs"]["overThresholdCount"] = 1
    second["stages"]["totalMs"]["overThresholdCount"] = 2
    snapshots = iter((first, second))
    lock_observations: list[bool] = []

    def read_timing(_stats):
        lock_observations.append(worker._status_lock.locked())
        return next(snapshots)

    monkeypatch.setattr(render_workers, "_read_render_timing", read_timing)
    try:
        worker._collect_frame(
            _frame(1, captured_at_ns=time.time_ns(), boot=str(uuid.uuid4()))
        )
        status = worker.status()
        assert lock_observations == [True, True]
        assert status["renderTiming"]["stages"]["totalMs"][
            "overThresholdCount"
        ] == 2
    finally:
        worker.close()


def test_spawn_worker_preserves_slow_timing_after_normal_recovery() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = RgbdRenderWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=RenderWorkerConfig(
            "missing", "missing", "explicit_process_restart", 85
        ),
        on_frame=lambda _frame: None,
        context=context,
        target=_slow_then_normal_timing_worker,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        status = worker.status()
        while not status["ready"] and time.monotonic() < deadline:
            time.sleep(0.01)
            status = worker.status()
        total = status["renderTiming"]["stages"]["totalMs"]
        assert status["status"] == "running"
        assert total["lastMs"] == 10.0
        assert total["maxMs"] == 250.0
        assert total["overThresholdCount"] == 1
        assert total["lastOverThresholdMs"] == 250.0
        assert isinstance(total["lastOverThresholdAtUnixNs"], int)
        assert total["lastOverThresholdAtUnixNs"] > 0
    finally:
        worker.close(timeout_s=1.0)


def test_viewer_status_exposes_child_identity_and_failure_counters() -> None:
    mujoco, model, data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    mailbox = LatestStateMailbox(shape, context=context)
    assert mailbox.try_publish(
        mujoco, model, data, captured_at_unix_ns=time.time_ns()
    ) == 1
    worker = PassiveViewerWorker(
        mailbox=mailbox,
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
        target=_healthy_viewer_identity_worker,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while not worker.status()["ready"] and time.monotonic() < deadline:
            time.sleep(0.01)
        status = worker.status()
        assert status["status"] == "running"
        assert status["workerPid"] == worker._process.pid
        uuid.UUID(str(status["workerBootId"]))
        assert status["failureCount"] == 0
        assert status["lastFailureAtUnixNs"] is None
        assert status["lastSuccessSequence"] == status["mailboxPublishedCount"]
        assert status["progressAgeMs"] < status["staleAfterMs"]
    finally:
        worker.close(timeout_s=1.0)


def test_viewer_alive_without_progress_becomes_stale_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mujoco, model, data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    mailbox = LatestStateMailbox(shape, context=context)
    mailbox.try_publish(mujoco, model, data, captured_at_unix_ns=time.time_ns())
    worker = PassiveViewerWorker(
        mailbox=mailbox,
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
        target=_healthy_viewer_identity_worker,
    )
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while not worker.status()["ready"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.status()["status"] == "running"
        data.time += 0.004
        assert mailbox.try_publish(
            mujoco, model, data, captured_at_unix_ns=time.time_ns()
        ) == 2
        progress_anchor_ns = 10_000_000_000
        worker._stats.last_success_monotonic_ns.value = progress_anchor_ns
        monkeypatch.setattr(
            time,
            "monotonic_ns",
            lambda: progress_anchor_ns + VIEWER_PROGRESS_STALE_NS,
        )
        boundary = worker.status()
        assert boundary["status"] == "running"
        assert boundary["ready"] is True
        monkeypatch.setattr(
            time,
            "monotonic_ns",
            lambda: progress_anchor_ns + VIEWER_PROGRESS_STALE_NS + 1,
        )

        stale = worker.status()

        assert stale["alive"] is True
        assert stale["status"] == "stale"
        assert stale["ready"] is False
        assert stale["lastSuccessSequence"] == 1
        assert stale["mailboxPublishedCount"] == 2
        assert stale["progressAgeMs"] > stale["staleAfterMs"]

        latest = mailbox.copy_latest(after_sequence=1)
        assert latest is not None
        _record_viewer_progress(worker._stats, mailbox, latest)
        recovered = worker.status()
        assert recovered["status"] == "running"
        assert recovered["ready"] is True
        assert recovered["lastSuccessSequence"] == 2
    finally:
        worker.close(timeout_s=1.0)


def test_viewer_progress_cache_reads_and_publishes_under_one_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = PassiveViewerWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
    )
    progress = iter(
        (
            (1_800_000_001_000_000_000, 1_000_000_000, 1, 0),
            (1_800_000_002_000_000_000, 2_000_000_000, 2, 0),
        )
    )
    lock_observations: list[bool] = []

    def read_progress(_stats):
        lock_observations.append(worker._progress_cache_lock.locked())
        return next(progress)

    monkeypatch.setattr(render_workers, "_read_viewer_progress", read_progress)
    monkeypatch.setattr(time, "monotonic_ns", lambda: 2_000_000_000)
    try:
        worker.status()
        status = worker.status()
        assert lock_observations == [True, True]
        assert status["lastSuccessSequence"] == 2
        assert worker._last_viewer_progress[2] == 2
    finally:
        worker.close()


def test_viewer_status_samples_clock_after_concurrent_progress_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = PassiveViewerWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
    )
    commit_monotonic_ns = 2_000_000_000
    progress_committed = threading.Event()

    def read_progress(_stats):
        progress_committed.set()
        return (
            1_800_000_001_000_000_000,
            commit_monotonic_ns,
            1,
            0,
        )

    def sample_now() -> int:
        assert progress_committed.is_set()
        return commit_monotonic_ns + 1

    monkeypatch.setattr(render_workers, "_read_viewer_progress", read_progress)
    monkeypatch.setattr(time, "monotonic_ns", sample_now)
    worker._stats.ready.value = 1
    try:
        # Exercise only the progress/clock ordering without launching X11.
        # A coherent running status also requires an alive child; restore the
        # real unstarted Process fact before close().
        with monkeypatch.context() as process_patch:
            process_patch.setattr(worker._process, "is_alive", lambda: True)
            status = worker.status()
        assert status["status"] == "running"
        assert status["ready"] is True
        assert status["progressAgeMs"] == pytest.approx(0.000001)
        assert status["lastError"] is None
    finally:
        worker.close()


def test_viewer_nonzero_exit_is_failed_even_after_parent_close() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = PassiveViewerWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
    )

    worker._closed = True
    worker._process._popen = SimpleNamespace(
        pid=1234,
        poll=lambda: -11,
        wait=lambda timeout=None: -11,
    )
    worker._stats.ready.value = 1
    worker._last_viewer_progress = (
        1_800_000_001_000_000_000,
        time.monotonic_ns(),
        1,
        0,
    )

    status = worker.status()

    assert status["status"] == "failed"
    assert status["ready"] is False
    assert status["alive"] is False
    assert status["exitCode"] == -11


def test_successful_worker_recovery_clears_shared_last_error() -> None:
    context = multiprocessing.get_context("spawn")
    stats = _SharedWorkerStats(context)
    _set_shared_error(stats, "transient JPEG failure")
    assert stats.last_error.value == b"transient JPEG failure"

    _clear_shared_error(stats)

    assert stats.last_error.value == b""


def test_collector_failure_is_durable_after_recovery_and_ignores_wall_order() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    commit_calls = 0

    def commit(_rendered: RenderedFrame) -> None:
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            raise RuntimeError("injected frame-store failure")

    worker = RgbdRenderWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=RenderWorkerConfig(
            "missing", "missing", "explicit_process_restart", 85
        ),
        on_frame=commit,
        context=context,
    )
    try:
        boot = str(uuid.uuid4())
        frame = _frame(1, captured_at_ns=time.time_ns(), boot=boot)
        # Simulate a wall-clock rollback: active errors must determine health,
        # not ordering two Unix timestamps from different clock generations.
        worker._stats.last_success_at_ns.value = 9_000_000_000_000_000_000
        worker._collect_frame(frame)

        failed = worker.status()
        assert failed["status"] == "failed"
        assert failed["failureCount"] == 1
        assert failed["collectorFailureCount"] == 1
        assert "injected frame-store failure" in failed["lastError"]
        assert failed["currentCollectorError"] == failed["lastError"]

        worker._collect_frame(
            _frame(2, captured_at_ns=time.time_ns(), boot=boot)
        )
        recovered = worker.status()
        assert recovered["status"] == "running"
        assert recovered["lastError"] is None
        assert recovered["currentCollectorError"] is None
        assert recovered["failureCount"] == 1
        assert recovered["collectorFailureCount"] == 1
        assert "injected frame-store failure" in recovered["lastCollectorError"]
        assert recovered["lastCollectorFailureAtUnixNs"] is not None
    finally:
        worker.close()


def test_worker_shutdown_is_bounded_and_idempotent() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = RgbdRenderWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=RenderWorkerConfig("missing", "missing", "explicit_process_restart", 85),
        on_frame=lambda _frame: None,
        context=context,
        target=_cooperative_worker,
    )
    worker.start()
    deadline = time.monotonic() + 5.0
    while not worker.status()["ready"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.status()["ready"] is True
    worker.close(timeout_s=1.0)
    assert not worker._process.is_alive()
    worker.close(timeout_s=1.0)


def test_worker_shutdown_terminates_an_unresponsive_child_within_bound() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = RgbdRenderWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=RenderWorkerConfig("missing", "missing", "explicit_process_restart", 85),
        on_frame=lambda _frame: None,
        context=context,
        target=_stubborn_worker,
    )
    worker.start()
    deadline = time.monotonic() + 5.0
    while not worker.status()["ready"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.status()["ready"] is True
    started = time.monotonic()
    worker.close(timeout_s=0.05)
    assert time.monotonic() - started < 2.0
    assert not worker._process.is_alive()


def test_passive_viewer_close_waits_for_native_lifetime_barrier() -> None:
    viewer = _FakePassiveViewer()
    now = 0.0
    waits: list[float] = []

    def monotonic() -> float:
        return now

    def wait(duration: float) -> None:
        nonlocal now
        assert viewer.close_calls == 1
        waits.append(duration)
        now += duration
        if len(waits) == 2:
            viewer.native_alive = False

    render_workers._close_and_wait_for_passive_viewer(
        viewer,
        timeout_s=1.0,
        poll_s=0.1,
        monotonic=monotonic,
        wait=wait,
    )

    assert viewer.close_calls == 1
    assert waits == pytest.approx([0.1, 0.1])
    assert viewer.native_alive is False


def test_passive_viewer_requires_a_weak_native_lifetime_contract() -> None:
    viewer = _FakePassiveViewer()
    assert render_workers._require_native_lifetime_ref(viewer) is viewer._sim

    viewer._sim = lambda: _FakeNativeViewer()
    with pytest.raises(RuntimeError, match="weak native lifetime reference"):
        render_workers._require_native_lifetime_ref(viewer)


def test_passive_viewer_close_timeout_is_bounded_and_diagnostic() -> None:
    viewer = _FakePassiveViewer()
    now = 0.0

    def monotonic() -> float:
        return now

    def wait(duration: float) -> None:
        nonlocal now
        now += duration

    with pytest.raises(
        TimeoutError,
        match="native render thread did not stop within 0.250s",
    ):
        render_workers._close_and_wait_for_passive_viewer(
            viewer,
            timeout_s=0.25,
            poll_s=0.1,
            monotonic=monotonic,
            wait=wait,
        )

    assert viewer.close_calls == 1
    assert now == pytest.approx(0.25)
    assert viewer.native_alive is True


def test_viewer_parent_reserves_native_cleanup_grace() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = PassiveViewerWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
    )
    process = _FakeProcess(exit_on_join=True)
    worker._process = process

    worker.close(timeout_s=0.01)

    assert process.join_timeouts == [
        render_workers.VIEWER_NATIVE_SHUTDOWN_TIMEOUT_S
        + render_workers.VIEWER_NATIVE_SHUTDOWN_PARENT_MARGIN_S
    ]
    assert process.terminate_calls == 0
    status = worker.status()
    assert status["status"] == "stopped"
    assert status["ready"] is False
    assert status["alive"] is False


def test_viewer_parent_sigkills_live_native_cleanup_after_full_grace() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = PassiveViewerWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
    )
    process = _FakeProcess(exit_on_join=False)
    worker._process = process
    worker._stats.viewer_native_active.value = 1
    worker._stats.viewer_shutdown_started.value = 1
    worker._stats.viewer_native_shutdown_complete.value = 0

    worker.close(timeout_s=0.01)

    assert process.terminate_calls == 0
    assert process.kill_calls == 1
    assert process.is_alive() is False
    assert process.join_timeouts == [
        render_workers.VIEWER_NATIVE_SHUTDOWN_TIMEOUT_S
        + render_workers.VIEWER_NATIVE_SHUTDOWN_PARENT_MARGIN_S,
        1.0,
    ]
    assert worker._closed is True
    status = worker.status()
    assert status["status"] == "failed"
    assert status["failureCount"] == 1
    assert status["exitCode"] == -9
    assert status["nativeViewerActive"] is False
    assert "forced SIGKILL" in status["lastError"]

    worker.close(timeout_s=0.01)
    assert process.kill_calls == 1


def test_viewer_failed_sigkill_keeps_close_retryable() -> None:
    mujoco, model, _data = _model_data()
    context = multiprocessing.get_context("spawn")
    shape = MujocoStateShape.from_model(mujoco, model)
    worker = PassiveViewerWorker(
        mailbox=LatestStateMailbox(shape, context=context),
        shape=shape,
        config=ViewerWorkerConfig(
            "missing",
            "missing",
            "explicit_process_restart",
            0.0,
            0.0,
        ),
        context=context,
    )
    process = _FakeProcess(exit_on_join=False, exit_on_kill=False)
    worker._process = process
    worker._stats.viewer_native_active.value = 1
    worker._stats.viewer_shutdown_started.value = 1

    with pytest.raises(
        RuntimeError,
        match="remained alive after forced SIGKILL",
    ):
        worker.close(timeout_s=0.01)

    assert worker._closed is False
    assert process.kill_calls == 1

    with pytest.raises(
        RuntimeError,
        match="remained alive after forced SIGKILL",
    ):
        worker.close(timeout_s=0.01)

    assert worker._closed is False
    assert process.kill_calls == 2


def test_frame_store_retains_only_latest_and_previous_and_uses_monotonic_age() -> None:
    support = PlantSupportController()
    support.finish_startup()
    store = PlantFrameStore(
        SimpleNamespace(robot_id="g1", localization_epoch_id="epoch"),
        support,
        worker_status_provider=lambda: {"status": "running", "alive": True},
    )
    boot = str(uuid.uuid4())
    now = time.time_ns()
    captured_monotonic_ns = 10_000_000_000
    for sequence in (1, 2, 3):
        store.commit(
            _frame(
                sequence,
                captured_at_ns=now + sequence,
                captured_at_monotonic_ns=captured_monotonic_ns + sequence,
                boot=boot,
            )
        )

    assert store._select_frame(f"boot={boot}&sequence=1") == (None, True)
    assert store._select_frame(f"boot={boot}&sequence=2")[0].sequence == 2  # type: ignore[union-attr]
    assert store._select_frame(f"boot={boot}&sequence=3")[0].sequence == 3  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        store._select_frame(f"boot={boot}")

    latest = store._select_frame("")[0]
    status, age_ms, _worker = store._observation_status(
        latest,
        now_monotonic_ns=(
            latest.captured_at_monotonic_ns + MAX_CAPTURE_AGE_NS + 1  # type: ignore[union-attr]
        ),
    )
    assert status == "stale"
    assert age_ms == pytest.approx(500.000001)

    status, age_ms, _worker = store._observation_status(
        latest,
        now_monotonic_ns=latest.captured_at_monotonic_ns - 1,  # type: ignore[union-attr]
    )
    assert status == "failed"
    assert age_ms == pytest.approx(-0.000001)


@pytest.mark.parametrize(
    ("render_started_wall_ns", "rendered_wall_ns"),
    [
        (20_000_000_000, 30_000_000_000),
        (9_000_000_000, 8_000_000_000),
    ],
    ids=("wall-forward", "wall-backward"),
)
def test_frame_store_wall_clock_corrections_do_not_change_freshness(
    render_started_wall_ns: int,
    rendered_wall_ns: int,
) -> None:
    support = PlantSupportController()
    support.finish_startup()
    store = PlantFrameStore(
        SimpleNamespace(robot_id="g1", localization_epoch_id="epoch"),
        support,
        worker_status_provider=lambda: {"status": "running", "alive": True},
    )
    captured_monotonic_ns = time.monotonic_ns() - 100_000_000
    frame = replace(
        _frame(
            1,
            captured_at_ns=10_000_000_000,
            captured_at_monotonic_ns=captured_monotonic_ns,
            boot=str(uuid.uuid4()),
        ),
        render_started_at_unix_ns=render_started_wall_ns,
        rendered_at_unix_ns=rendered_wall_ns,
        render_started_at_monotonic_ns=captured_monotonic_ns + 10_000_000,
        rendered_at_monotonic_ns=captured_monotonic_ns + 20_000_000,
    )

    store.commit(frame)
    state = store.state()

    assert state["observationStatus"] == "running"
    assert 0.0 <= state["captureAgeMs"] <= 500.0
    assert state["renderLatencyMs"] == pytest.approx(20.0)
    assert state["capturedAtUnixNs"] == 10_000_000_000
    assert state["capturedAtMonotonicNs"] == captured_monotonic_ns


def test_frame_store_http_reads_do_not_wait_for_commit_and_see_one_window() -> None:
    support = PlantSupportController()
    support.finish_startup()
    store = PlantFrameStore(
        SimpleNamespace(robot_id="g1", localization_epoch_id="epoch"),
        support,
        worker_status_provider=lambda: {"status": "running", "alive": True},
    )
    boot = str(uuid.uuid4())
    captured_at_ns = time.time_ns()
    first = _frame(1, captured_at_ns=captured_at_ns, boot=boot)
    second = _frame(2, captured_at_ns=captured_at_ns + 1, boot=boot)
    store.commit(first)
    original_window = store._window
    store._commit_lock.acquire()
    executor = ThreadPoolExecutor(max_workers=2)
    try:
        read_future = executor.submit(
            lambda: (
                store._select_frame("")[0],
                store.state()["sequence"],
            )
        )
        selected, state_sequence = read_future.result(timeout=1.0)
        assert selected is first
        assert state_sequence == 1

        commit_future = executor.submit(store.commit, second)
        assert not commit_future.done()
    finally:
        store._commit_lock.release()
        executor.shutdown(wait=True)

    assert store._window is not original_window
    assert original_window.latest is first
    assert original_window.previous is None
    assert store._window.latest is second
    assert store._window.previous is first


def test_viewer_worker_has_no_physics_step_or_input_callback_authority() -> None:
    source = inspect.getsource(_viewer_worker_main)
    assert "mj_step(" not in source
    assert "key_callback=" not in source
    assert "launch_passive(model, data)" in source
    assert 'os.environ.pop("MUJOCO_GL", None)' in source
    assert 'os.environ["MUJOCO_GL"] = config.gl_backend' in source
    assert "_close_and_wait_for_passive_viewer(" in source
    assert "simulate_ref=native_lifetime_ref" in source
    assert "with viewer.lock():" in source
    assert "viewer.sync(state_only=True)" in source
