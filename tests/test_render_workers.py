from __future__ import annotations

import inspect
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import uuid

import numpy as np
import pytest

from hide_mujoco_sonic.frames import BodyTwist
from hide_mujoco_sonic.plant import (
    PlantFrameStore,
    PlantPhysicsFault,
    PlantPhysicsMonitor,
    PlantSupportController,
)
from hide_mujoco_sonic.render_workers import (
    LatestStateMailbox,
    MAX_CAPTURE_AGE_NS,
    MujocoStateShape,
    RenderedFrame,
    RenderWorkerConfig,
    RgbdRenderWorker,
    _encode_rgb_jpeg,
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


def _frame(sequence: int, *, captured_at_ns: int, boot: str) -> RenderedFrame:
    return RenderedFrame(
        producer_boot_id=boot,
        sequence=sequence,
        captured_at_unix_ns=captured_at_ns,
        render_started_at_unix_ns=captured_at_ns + 10,
        rendered_at_unix_ns=captured_at_ns + 20,
        simulation_time_s=float(sequence),
        pose=(0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
        root_velocity=BodyTwist.zero(),
        camera_pose=(0.2, 0.0, 1.1, 1.0, 0.0, 0.0, 0.0),
        intrinsics=(("fx", 100.0), ("fy", 100.0), ("cx", 1.0), ("cy", 0.5)),
        width=2,
        height=1,
        rgb_jpeg=f"jpeg-{sequence}".encode(),
        depth_z16_le=bytes((sequence, 0, sequence + 1, 0)),
        render_timing=(("totalMs", 0.001),),
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
    ) == 1
    snapshot = mailbox.copy_latest()
    assert snapshot is not None

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
        assert mailbox.try_publish(
            mujoco,
            model,
            authority,
            captured_at_unix_ns=time.time_ns(),
        ) == 1
        assert committed_event.wait(30.0), worker.status()
        frame = committed[-1]
        assert (frame.producer_boot_id, frame.sequence) == (boot, 1)
        assert (frame.width, frame.height) == (640, 480)
        assert frame.rgb_jpeg.startswith(b"\xff\xd8")
        assert len(frame.depth_z16_le) == 640 * 480 * 2
        assert frame.pose == pytest.approx(tuple(float(value) for value in authority.qpos[:7]))
        assert frame.rendered_at_unix_ns >= frame.render_started_at_unix_ns
        assert set(dict(frame.render_timing)) == {
            "stateApplyMs",
            "rgbRenderMs",
            "depthRenderMs",
            "jpegEncodeMs",
            "totalMs",
            "queueAgeAtStartMs",
        }
    finally:
        worker.close(timeout_s=5.0)


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


def test_frame_store_retains_only_latest_and_previous_and_uses_source_age() -> None:
    support = PlantSupportController()
    support.finish_startup()
    store = PlantFrameStore(
        SimpleNamespace(robot_id="g1", localization_epoch_id="epoch"),
        support,
        worker_status_provider=lambda: {"status": "running", "alive": True},
    )
    boot = str(uuid.uuid4())
    now = time.time_ns()
    for sequence in (1, 2, 3):
        store.commit(_frame(sequence, captured_at_ns=now + sequence, boot=boot))

    assert store._select_frame(f"boot={boot}&sequence=1") == (None, True)
    assert store._select_frame(f"boot={boot}&sequence=2")[0].sequence == 2  # type: ignore[union-attr]
    assert store._select_frame(f"boot={boot}&sequence=3")[0].sequence == 3  # type: ignore[union-attr]
    with pytest.raises(ValueError):
        store._select_frame(f"boot={boot}")

    latest = store._select_frame("")[0]
    status, age_ms, _worker = store._observation_status(
        latest,
        now_ns=latest.captured_at_unix_ns + MAX_CAPTURE_AGE_NS + 1,  # type: ignore[union-attr]
    )
    assert status == "stale"
    assert age_ms == pytest.approx(500.000001)


def test_viewer_worker_has_no_physics_step_or_input_callback_authority() -> None:
    source = inspect.getsource(_viewer_worker_main)
    assert "mj_step(" not in source
    assert "key_callback=" not in source
    assert "launch_passive(model, data)" in source
