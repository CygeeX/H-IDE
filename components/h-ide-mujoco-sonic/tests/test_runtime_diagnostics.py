from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from hide_mujoco_sonic.runtime_diagnostics import (
    PlantRuntimeDiagnostics,
    SonicRuntimeDiagnostics,
)


class FakeClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeGc:
    def __init__(self) -> None:
        self.callbacks = []
        self.enabled = True
        self.collected = 17
        self.frozen = False

    def isenabled(self) -> bool:
        return self.enabled

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def collect(self) -> int:
        return self.collected

    def freeze(self) -> None:
        self.frozen = True

    def get_freeze_count(self) -> int:
        return 123 if self.frozen else 0

    def emit(self, phase: str, *, generation: int = 2) -> None:
        for callback in tuple(self.callbacks):
            callback(phase, {"generation": generation})


def test_sonic_runtime_diagnostics_distinguishes_stream_gap_from_gc_pause():
    monotonic = FakeClock()
    unix = FakeClock(1_700_000_000.0)
    fake_gc = FakeGc()
    diagnostics = SonicRuntimeDiagnostics(
        stall_threshold_s=0.1,
        monotonic_clock=monotonic,
        unix_clock=unix,
        gc_module=fake_gc,
    )

    diagnostics.observe_lowstate()
    diagnostics.observe_health_probe()
    monotonic.advance(0.025)
    unix.advance(0.025)
    diagnostics.observe_health_probe()
    monotonic.advance(0.125)
    unix.advance(0.125)
    diagnostics.observe_lowstate()

    state = diagnostics.snapshot()
    assert state["lowstate_callback_last_gap_s"] == pytest.approx(0.15)
    assert state["lowstate_callback_over_threshold_count"] == 1
    assert state["lowstate_callback_last_gap_gc_pause_s"] == 0.0
    assert state["lowstate_callback_last_over_threshold_gap_s"] == pytest.approx(
        0.15
    )
    assert state["lowstate_callback_last_over_threshold_gc_pause_s"] == 0.0
    assert state["lowstate_callback_over_threshold_gc_overlap_count"] == 0
    assert state[
        "lowstate_callback_over_threshold_non_gc_residual_count"
    ] == 1
    assert state["health_probe_last_gap_s"] == pytest.approx(0.025)
    assert state["health_probe_over_threshold_count"] == 0
    assert state["python_gc_collection_count"] == 0

    diagnostics.close()
    assert fake_gc.callbacks == []


def test_sonic_runtime_diagnostics_correlates_completed_gc_with_control_gap():
    monotonic = FakeClock()
    unix = FakeClock(1_700_000_000.0)
    fake_gc = FakeGc()
    diagnostics = SonicRuntimeDiagnostics(
        stall_threshold_s=0.1,
        monotonic_clock=monotonic,
        unix_clock=unix,
        gc_module=fake_gc,
    )
    diagnostics.observe_control_publish()

    monotonic.advance(0.020)
    unix.advance(0.020)
    fake_gc.emit("start", generation=2)
    monotonic.advance(0.120)
    unix.advance(0.120)
    fake_gc.emit("stop", generation=2)
    monotonic.advance(0.010)
    unix.advance(0.010)
    diagnostics.observe_control_publish()

    state = diagnostics.snapshot()
    assert state["control_publish_last_gap_s"] == pytest.approx(0.15)
    assert state["control_publish_last_gap_gc_pause_s"] == pytest.approx(0.12)
    assert state["control_publish_last_over_threshold_gap_s"] == pytest.approx(
        0.15
    )
    assert state[
        "control_publish_last_over_threshold_gc_pause_s"
    ] == pytest.approx(0.12)
    assert state["control_publish_over_threshold_gc_overlap_count"] == 1
    assert state[
        "control_publish_over_threshold_non_gc_residual_count"
    ] == 0
    assert state["python_gc_collection_count"] == 1
    assert state["python_gc_last_generation"] == 2
    assert state["python_gc_last_pause_s"] == pytest.approx(0.12)
    assert state["python_gc_max_pause_s"] == pytest.approx(0.12)
    assert state["python_gc_total_pause_s"] == pytest.approx(0.12)
    assert state["python_gc_pause_active"] is False

    diagnostics.close()


def test_mixed_stall_persists_gc_and_non_gc_evidence_after_healthy_sample():
    monotonic = FakeClock()
    unix = FakeClock(1_700_000_000.0)
    fake_gc = FakeGc()
    diagnostics = SonicRuntimeDiagnostics(
        stall_threshold_s=0.1,
        monotonic_clock=monotonic,
        unix_clock=unix,
        gc_module=fake_gc,
    )
    diagnostics.observe_health_probe()

    monotonic.advance(0.020)
    unix.advance(0.020)
    fake_gc.emit("start", generation=1)
    monotonic.advance(0.120)
    unix.advance(0.120)
    fake_gc.emit("stop", generation=1)
    monotonic.advance(0.110)
    unix.advance(0.110)
    diagnostics.observe_health_probe()

    stalled = diagnostics.snapshot()
    assert stalled["health_probe_last_gap_s"] == pytest.approx(0.25)
    assert stalled["health_probe_last_gap_gc_pause_s"] == pytest.approx(0.12)
    assert stalled["health_probe_last_over_threshold_gap_s"] == pytest.approx(
        0.25
    )
    assert stalled[
        "health_probe_last_over_threshold_gc_pause_s"
    ] == pytest.approx(0.12)
    assert stalled["health_probe_over_threshold_gc_overlap_count"] == 1
    assert stalled["health_probe_over_threshold_non_gc_residual_count"] == 1
    stalled_at = stalled["health_probe_last_over_threshold_at_unix_s"]

    monotonic.advance(0.025)
    unix.advance(0.025)
    diagnostics.observe_health_probe()

    recovered = diagnostics.snapshot()
    assert recovered["health_probe_last_gap_s"] == pytest.approx(0.025)
    assert recovered["health_probe_last_gap_gc_pause_s"] == 0.0
    assert recovered["health_probe_last_over_threshold_gap_s"] == pytest.approx(
        0.25
    )
    assert recovered[
        "health_probe_last_over_threshold_gc_pause_s"
    ] == pytest.approx(0.12)
    assert recovered["health_probe_last_over_threshold_at_unix_s"] == stalled_at
    assert recovered["health_probe_over_threshold_gc_overlap_count"] == 1
    assert recovered["health_probe_over_threshold_non_gc_residual_count"] == 1

    diagnostics.close()


def test_health_snapshots_do_not_walk_the_frozen_object_graph(monkeypatch):
    fake_gc = FakeGc()
    diagnostics = SonicRuntimeDiagnostics(stall_threshold_s=0.1, gc_module=fake_gc)
    diagnostics.freeze_warm_runtime()

    def forbidden_walk():
        raise AssertionError("health polling must not walk frozen objects")

    monkeypatch.setattr(fake_gc, "get_freeze_count", forbidden_walk)
    for _ in range(20):
        assert diagnostics.snapshot()["python_gc_frozen_objects"] == 123
    diagnostics.close()


def test_freeze_warm_runtime_disables_automatic_gc_and_is_observable():
    monotonic = FakeClock()
    fake_gc = FakeGc()
    fake_gc.enabled = False
    diagnostics = SonicRuntimeDiagnostics(
        stall_threshold_s=0.1,
        monotonic_clock=monotonic,
        gc_module=fake_gc,
    )

    diagnostics.freeze_warm_runtime()

    state = diagnostics.snapshot()
    assert fake_gc.enabled is False
    assert fake_gc.frozen is True
    assert state["python_gc_enabled"] is False
    assert state["python_gc_automatic_disabled"] is True
    assert state["python_gc_warmup_collected_objects"] == 17
    assert state["python_gc_freeze_applied"] is True
    assert state["python_gc_frozen_objects"] == 123
    assert state["python_gc_freeze_error"] is None

    diagnostics.close()
    diagnostics.close()
    assert fake_gc.callbacks == []


def test_plant_runtime_diagnostics_attributes_slow_loop_stage():
    diagnostics = PlantRuntimeDiagnostics(stall_threshold_s=0.1)

    diagnostics.observe_loop_start(1.0, unix_ns=1_000_000_000)
    diagnostics.observe_stage("simStep", 0.004, unix_ns=1_004_000_000)
    diagnostics.observe_loop_start(1.15, unix_ns=1_150_000_000)
    diagnostics.observe_stage("rgbRender", 0.12, unix_ns=1_270_000_000)
    diagnostics.observe_stage(
        "capturePostprocess",
        0.025,
        unix_ns=1_295_000_000,
    )

    state = diagnostics.snapshot()
    assert state["simLoopGap"]["lastS"] == pytest.approx(0.15)
    assert state["simLoopGap"]["overThresholdCount"] == 1
    assert state["stages"]["simStep"]["maxS"] == pytest.approx(0.004)
    assert state["stages"]["rgbRender"]["overThresholdCount"] == 1
    assert state["lastSlowStage"] == "rgbRender"
    assert state["lastSlowDurationS"] == pytest.approx(0.12)
    assert state["lastSlowAtUnixNs"] == 1_270_000_000

    with pytest.raises(ValueError, match="unknown plant timing stage"):
        diagnostics.observe_stage("jpegMaybe", 0.2, unix_ns=2_000_000_000)


def test_plant_authority_publisher_never_waits_for_http_snapshot_reader() -> None:
    diagnostics = PlantRuntimeDiagnostics(stall_threshold_s=0.1)
    diagnostics._snapshot_reader_lock.acquire()

    def publish_authority_evidence() -> int:
        started_ns = time.perf_counter_ns()
        for index in range(100):
            at_s = 1.0 + index * 0.004
            diagnostics.observe_loop_start(at_s, unix_ns=1_000_000_000 + index)
            diagnostics.observe_stage(
                "simStep",
                0.004,
                unix_ns=1_000_000_000 + index,
            )
        return time.perf_counter_ns() - started_ns

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        elapsed_ns = executor.submit(publish_authority_evidence).result(timeout=1.0)
        assert elapsed_ns < 20_000_000
    finally:
        diagnostics._snapshot_reader_lock.release()
        executor.shutdown(wait=True)

    state = diagnostics.snapshot()
    assert state["simLoopGap"]["lastS"] == pytest.approx(0.004)
    assert state["stages"]["simStep"]["lastS"] == pytest.approx(0.004)


def test_plant_published_diagnostics_are_immutable_across_generations() -> None:
    diagnostics = PlantRuntimeDiagnostics(stall_threshold_s=0.1)
    original_generation = diagnostics._published_snapshot
    diagnostics.observe_stage("snapshotPublish", 0.02, unix_ns=123)

    assert original_generation["stages"]["snapshotPublish"]["lastS"] is None
    assert diagnostics._published_snapshot is not original_generation
    caller_copy = diagnostics.snapshot()
    caller_copy["simLoopGap"]["maxS"] = 999.0
    caller_copy["stages"]["snapshotPublish"]["maxS"] = 999.0

    unchanged = diagnostics.snapshot()
    assert unchanged["simLoopGap"]["maxS"] == 0.0
    assert unchanged["stages"]["snapshotPublish"]["maxS"] == pytest.approx(
        0.02
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_runtime_diagnostics_reject_invalid_threshold(value: float):
    with pytest.raises(ValueError, match="finite and positive"):
        PlantRuntimeDiagnostics(stall_threshold_s=value)
    with pytest.raises(ValueError, match="finite and positive"):
        SonicRuntimeDiagnostics(stall_threshold_s=value)
