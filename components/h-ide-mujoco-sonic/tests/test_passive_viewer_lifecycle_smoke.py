from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import passive_viewer_lifecycle_smoke as smoke


class _FakeArray(list):
    def __setitem__(self, key, value) -> None:
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            replacement = (
                [value] * len(indices)
                if isinstance(value, (int, float))
                else list(value)
            )
            if key == slice(None, None, None):
                super().__setitem__(key, replacement)
                return
            for index, item in zip(indices, replacement, strict=True):
                super().__setitem__(index, item)
            return
        super().__setitem__(key, value)


class _FakeModel:
    nq = 8
    nv = 7
    qpos0 = _FakeArray([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.1])


class _FakeData:
    def __init__(self, _model) -> None:
        self.qpos = _FakeArray([0.0] * 8)
        self.qvel = _FakeArray([0.0] * 7)
        self.time = -1.0


class _FakeMujoco:
    __version__ = "test-version"
    forward_calls = 0

    class MjModel:
        @staticmethod
        def from_xml_path(_path: str):
            return _FakeModel()

    MjData = _FakeData

    @classmethod
    def mj_forward(cls, _model, _data) -> None:
        cls.forward_calls += 1


class _FakeMailbox:
    instances: list["_FakeMailbox"] = []

    def __init__(self, shape, *, context) -> None:
        self.shape = shape
        self.context = context
        self.published_count = 0
        self.instances.append(self)

    def try_publish(self, *_args, **_kwargs):
        self.published_count += 1
        return self.published_count


class _FakeWorker:
    instances: list["_FakeWorker"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        self.close_calls = 0
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def status(self) -> dict[str, object]:
        if self.closed:
            return {
                "status": "stopped",
                "ready": False,
                "alive": False,
                "exitCode": 0,
                "failureCount": 0,
                "lastError": None,
                "shutdownStarted": True,
                "nativeShutdownComplete": True,
                "nativeViewerActive": False,
                "workerPid": 1234,
                "workerBootId": "00000000-0000-0000-0000-000000000001",
            }
        return {
            "status": "running" if self.started else "starting",
            "ready": self.started,
            "alive": True,
            "exitCode": None,
            "failureCount": 0,
            "lastError": None,
        }

    def close(self, *, timeout_s: float) -> None:
        assert timeout_s == 6.0
        self.close_calls += 1
        self.closed = True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 10.0
        self.unix = 1_800_000_000_000_000_000

    def monotonic(self) -> float:
        return self.now

    def wait(self, seconds: float) -> None:
        self.now += seconds

    def unix_ns(self) -> int:
        self.unix += 1
        return self.unix


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeMujoco.forward_calls = 0
    _FakeMailbox.instances.clear()
    _FakeWorker.instances.clear()


def _config(tmp_path: Path):
    model = tmp_path / "scene.xml"
    model.write_text("<mujoco/>", encoding="utf-8")
    manifest = tmp_path / "runtime.json"
    manifest.write_text("{}", encoding="utf-8")
    return SimpleNamespace(
        model_scene=model,
        scene_runtime_manifest=manifest,
        reset_policy="explicit_process_restart",
        runtime_scene=SimpleNamespace(
            spawn_x=1.25,
            spawn_y=-0.5,
            spawn_yaw=0.4,
        ),
    )


def test_static_lifecycle_smoke_repeats_without_control_or_physics(
    tmp_path: Path,
) -> None:
    clock = _FakeClock()
    context = object()

    report = smoke.run_lifecycle_smoke(
        mujoco_api=_FakeMujoco,
        config=_config(tmp_path),
        display=":0",
        xauthority="/tmp/test-Xauthority",
        cycles=2,
        visible_seconds=0.04,
        startup_timeout_s=1.0,
        publish_period_s=0.02,
        close_timeout_s=6.0,
        context=context,
        worker_factory=_FakeWorker,
        mailbox_factory=_FakeMailbox,
        shape_factory=lambda _api, _model: "shape",
        monotonic=clock.monotonic,
        wait=clock.wait,
        unix_ns=clock.unix_ns,
    )

    assert report["status"] == "passed"
    assert report["cyclesCompleted"] == 2
    assert report["physicsSteps"] == 0
    assert report["controlAuthority"] is False
    assert report["networkListeners"] == []
    assert report["rosDdsEnabled"] is False
    assert report["staticSnapshotsPublished"] == 6
    assert all(
        cycle["finalStatus"] == "stopped"
        and cycle["readyAfterClose"] is False
        for cycle in report["cycles"]
    )
    assert _FakeMujoco.forward_calls == 1
    assert all(worker.started and worker.closed for worker in _FakeWorker.instances)
    assert all(worker.close_calls == 1 for worker in _FakeWorker.instances)
    assert all(
        worker.kwargs["context"] is context for worker in _FakeWorker.instances
    )
    assert all(
        mailbox.context is context for mailbox in _FakeMailbox.instances
    )


def test_failed_viewer_status_is_closed_before_failure_propagates(
    tmp_path: Path,
) -> None:
    class FailedWorker(_FakeWorker):
        def status(self) -> dict[str, object]:
            if self.closed:
                return super().status()
            return {
                "status": "failed",
                "ready": False,
                "alive": False,
                "exitCode": 1,
                "failureCount": 1,
                "lastError": "injected viewer failure",
            }

    clock = _FakeClock()
    with pytest.raises(
        smoke.ViewerLifecycleSmokeFailure,
        match="injected viewer failure",
    ):
        smoke.run_lifecycle_smoke(
            mujoco_api=_FakeMujoco,
            config=_config(tmp_path),
            display=":0",
            xauthority=None,
            cycles=1,
            visible_seconds=0.04,
            startup_timeout_s=1.0,
            publish_period_s=0.02,
            close_timeout_s=6.0,
            context=object(),
            worker_factory=FailedWorker,
            mailbox_factory=_FakeMailbox,
            shape_factory=lambda _api, _model: "shape",
            monotonic=clock.monotonic,
            wait=clock.wait,
            unix_ns=clock.unix_ns,
        )

    assert len(FailedWorker.instances) == 1
    assert FailedWorker.instances[0].closed is True
    assert FailedWorker.instances[0].close_calls == 1


def test_partial_worker_start_is_closed_before_failure_propagates(
    tmp_path: Path,
) -> None:
    class StartFailureWorker(_FakeWorker):
        def start(self) -> None:
            raise RuntimeError("injected partial start failure")

    clock = _FakeClock()
    with pytest.raises(RuntimeError, match="injected partial start failure"):
        smoke.run_lifecycle_smoke(
            mujoco_api=_FakeMujoco,
            config=_config(tmp_path),
            display=":0",
            xauthority=None,
            cycles=1,
            visible_seconds=0.04,
            startup_timeout_s=1.0,
            publish_period_s=0.02,
            close_timeout_s=6.0,
            context=object(),
            worker_factory=StartFailureWorker,
            mailbox_factory=_FakeMailbox,
            shape_factory=lambda _api, _model: "shape",
            monotonic=clock.monotonic,
            wait=clock.wait,
            unix_ns=clock.unix_ns,
        )

    assert len(StartFailureWorker.instances) == 1
    assert StartFailureWorker.instances[0].closed is True
    assert StartFailureWorker.instances[0].close_calls == 1


def test_x11_environment_fails_closed_without_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)

    with pytest.raises(
        smoke.ViewerLifecycleSmokeFailure,
        match="DISPLAY is required",
    ):
        smoke._configure_x11_environment(display=None, xauthority=None)


def test_x11_environment_uses_explicit_existing_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = tmp_path / "Xauthority"
    authority.write_bytes(b"test")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("XAUTHORITY", raising=False)

    display, resolved = smoke._configure_x11_environment(
        display=":99",
        xauthority=str(authority),
    )

    assert display == ":99"
    assert resolved == str(authority.resolve())
    assert smoke.os.environ["DISPLAY"] == ":99"
    assert smoke.os.environ["XAUTHORITY"] == str(authority.resolve())


def test_cli_rejects_non_positive_cycle_count() -> None:
    with pytest.raises(SystemExit):
        smoke._build_parser().parse_args(["--cycles", "0"])


def test_smoke_source_has_no_physics_or_external_transport_authority() -> None:
    source = inspect.getsource(smoke)
    assert "mj_step(" not in source
    assert "run_plant" not in source
    assert "run_sonic" not in source
    assert "rclpy" not in source
    assert "cyclonedds" not in source.lower()
    assert "requests." not in source
