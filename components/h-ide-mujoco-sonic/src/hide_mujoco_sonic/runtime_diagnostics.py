"""Low-overhead runtime timing evidence for the simulator processes.

The safety bridge must distinguish an upstream LowState publication gap from a
whole-process Python pause.  These observers retain scalar timing evidence only;
they never drive control, publish DDS, or perform I/O from a real-time callback.
"""

from __future__ import annotations

import gc
import math
import threading
import time
from typing import Any, Callable


class _GapProbe:
    def __init__(self, *, threshold_s: float) -> None:
        self.threshold_s = threshold_s
        self.previous_at_s: float | None = None
        self.previous_gc_total_s = 0.0
        self.last_gap_s: float | None = None
        self.max_gap_s = 0.0
        self.over_threshold_count = 0
        self.last_over_threshold_at_unix_s: float | None = None
        self.last_gap_gc_pause_s = 0.0
        self.last_over_threshold_gap_s: float | None = None
        self.last_over_threshold_gc_pause_s: float | None = None
        self.over_threshold_gc_overlap_count = 0
        self.over_threshold_non_gc_residual_count = 0

    def observe(
        self,
        at_s: float,
        *,
        unix_s: float,
        gc_total_pause_s: float,
    ) -> None:
        previous = self.previous_at_s
        previous_gc_total = self.previous_gc_total_s
        self.previous_at_s = at_s
        self.previous_gc_total_s = gc_total_pause_s
        if previous is None:
            return
        gap = max(0.0, at_s - previous)
        self.last_gap_s = gap
        self.max_gap_s = max(self.max_gap_s, gap)
        gc_pause = max(
            0.0,
            gc_total_pause_s - previous_gc_total,
        )
        self.last_gap_gc_pause_s = gc_pause
        if gap > self.threshold_s:
            self.over_threshold_count += 1
            self.last_over_threshold_at_unix_s = unix_s
            self.last_over_threshold_gap_s = gap
            self.last_over_threshold_gc_pause_s = gc_pause
            if gc_pause > 0.0:
                self.over_threshold_gc_overlap_count += 1
            non_gc_residual = max(0.0, gap - gc_pause)
            if non_gc_residual > self.threshold_s:
                self.over_threshold_non_gc_residual_count += 1

    def snapshot(self, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_last_gap_s": self.last_gap_s,
            f"{prefix}_max_gap_s": self.max_gap_s,
            f"{prefix}_over_threshold_count": self.over_threshold_count,
            f"{prefix}_last_over_threshold_at_unix_s": (
                self.last_over_threshold_at_unix_s
            ),
            f"{prefix}_last_gap_gc_pause_s": self.last_gap_gc_pause_s,
            f"{prefix}_last_over_threshold_gap_s": (
                self.last_over_threshold_gap_s
            ),
            f"{prefix}_last_over_threshold_gc_pause_s": (
                self.last_over_threshold_gc_pause_s
            ),
            f"{prefix}_over_threshold_gc_overlap_count": (
                self.over_threshold_gc_overlap_count
            ),
            f"{prefix}_over_threshold_non_gc_residual_count": (
                self.over_threshold_non_gc_residual_count
            ),
        }


class SonicRuntimeDiagnostics:
    """Correlate callback/control/probe gaps with CPython cyclic-GC pauses.

    GC callbacks deliberately update scalar attributes without taking a lock.
    Taking a contended Python lock from a stop-the-world callback can deadlock:
    the lock owner cannot run while the callback owns the GIL.  CPython executes
    the assignments below while holding the GIL, and a telemetry snapshot may
    tolerate observing two adjacent scalar generations during an update.
    """

    def __init__(
        self,
        *,
        stall_threshold_s: float,
        monotonic_clock: Callable[[], float] = time.monotonic,
        unix_clock: Callable[[], float] = time.time,
        gc_module=gc,
    ) -> None:
        threshold = float(stall_threshold_s)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("stall_threshold_s must be finite and positive")
        self.stall_threshold_s = threshold
        self._clock = monotonic_clock
        self._unix_clock = unix_clock
        self._gc = gc_module
        self._lowstate = _GapProbe(threshold_s=threshold)
        self._control = _GapProbe(threshold_s=threshold)
        self._health = _GapProbe(threshold_s=threshold)
        self._gc_started_at_s: float | None = None
        self._gc_pause_active = False
        self._gc_collection_count = 0
        self._gc_last_generation: int | None = None
        self._gc_last_pause_s: float | None = None
        self._gc_max_pause_s = 0.0
        self._gc_total_pause_s = 0.0
        self._gc_last_completed_at_unix_s: float | None = None
        self._gc_warmup_collected_objects = 0
        self._gc_freeze_applied = False
        self._gc_automatic_disabled = False
        self._gc_freeze_error: str | None = None
        self._gc_callback = self._observe_gc
        self._registered = False
        callbacks = getattr(self._gc, "callbacks", None)
        if callbacks is None:
            raise ValueError("gc module must expose callbacks")
        callbacks.append(self._gc_callback)
        self._registered = True

    def _observe_gc(self, phase: str, info: dict[str, Any]) -> None:
        now = self._clock()
        if phase == "start":
            self._gc_started_at_s = now
            self._gc_pause_active = True
            generation = info.get("generation")
            self._gc_last_generation = (
                int(generation) if isinstance(generation, int) else None
            )
            return
        if phase != "stop":
            return
        started = self._gc_started_at_s
        pause = 0.0 if started is None else max(0.0, now - started)
        self._gc_started_at_s = None
        self._gc_pause_active = False
        self._gc_collection_count += 1
        self._gc_last_pause_s = pause
        self._gc_max_pause_s = max(self._gc_max_pause_s, pause)
        self._gc_total_pause_s += pause
        self._gc_last_completed_at_unix_s = self._unix_clock()

    def observe_lowstate(self, at_s: float | None = None) -> None:
        now = self._clock() if at_s is None else float(at_s)
        self._lowstate.observe(
            now,
            unix_s=self._unix_clock(),
            gc_total_pause_s=self._gc_total_pause_s,
        )

    def observe_control_publish(self, at_s: float | None = None) -> None:
        now = self._clock() if at_s is None else float(at_s)
        self._control.observe(
            now,
            unix_s=self._unix_clock(),
            gc_total_pause_s=self._gc_total_pause_s,
        )

    def observe_health_probe(self, at_s: float | None = None) -> None:
        now = self._clock() if at_s is None else float(at_s)
        self._health.observe(
            now,
            unix_s=self._unix_clock(),
            gc_total_pause_s=self._gc_total_pause_s,
        )

    def freeze_warm_runtime(self) -> None:
        """Collect/freeze the warm graph, then disable automatic cyclic GC.

        CPython refcounting remains active.  Cyclic collection is deliberately
        manual-only after warm-up because it is process-global and therefore
        can stop the 50 Hz actuator owner even when a request/telemetry thread
        triggers it.  A live incident measured an 80.5 ms generation-1 pause
        inside a 154.6 ms control-publish gap.  The process is bounded-lived;
        deterministic control takes precedence over opportunistic cycle
        reclamation while it owns an actuator session.
        """

        try:
            if not self._gc.isenabled():
                self._gc.enable()
            self._gc_warmup_collected_objects = int(self._gc.collect())
            self._gc.freeze()
            self._gc.disable()
            if self._gc.isenabled():
                raise RuntimeError("automatic cyclic GC remained enabled")
            self._gc_freeze_applied = True
            self._gc_automatic_disabled = True
            self._gc_freeze_error = None
        except Exception as exc:
            self._gc_freeze_applied = False
            self._gc_automatic_disabled = False
            self._gc_freeze_error = f"{type(exc).__name__}: {exc}"[:256]

    def snapshot(self) -> dict[str, Any]:
        try:
            gc_enabled = bool(self._gc.isenabled())
        except Exception:
            gc_enabled = False
        try:
            frozen_objects = int(self._gc.get_freeze_count())
        except Exception:
            frozen_objects = 0
        result: dict[str, Any] = {
            "runtime_stall_threshold_s": self.stall_threshold_s,
            "python_gc_enabled": gc_enabled,
            "python_gc_pause_active": self._gc_pause_active,
            "python_gc_collection_count": self._gc_collection_count,
            "python_gc_last_generation": self._gc_last_generation,
            "python_gc_last_pause_s": self._gc_last_pause_s,
            "python_gc_max_pause_s": self._gc_max_pause_s,
            "python_gc_total_pause_s": self._gc_total_pause_s,
            "python_gc_last_completed_at_unix_s": (
                self._gc_last_completed_at_unix_s
            ),
            "python_gc_warmup_collected_objects": (
                self._gc_warmup_collected_objects
            ),
            "python_gc_freeze_applied": self._gc_freeze_applied,
            "python_gc_automatic_disabled": self._gc_automatic_disabled,
            "python_gc_frozen_objects": frozen_objects,
            "python_gc_freeze_error": self._gc_freeze_error,
        }
        result.update(self._lowstate.snapshot("lowstate_callback"))
        result.update(self._control.snapshot("control_publish"))
        result.update(self._health.snapshot("health_probe"))
        return result

    def close(self) -> None:
        if not self._registered:
            return
        callbacks = self._gc.callbacks
        try:
            callbacks.remove(self._gc_callback)
        except ValueError:
            pass
        self._registered = False


class _DurationProbe:
    def __init__(self, *, threshold_s: float) -> None:
        self.threshold_s = threshold_s
        self.last_s: float | None = None
        self.max_s = 0.0
        self.over_threshold_count = 0
        self.last_over_threshold_at_unix_ns: int | None = None

    def observe(self, duration_s: float, *, unix_ns: int) -> bool:
        duration = max(0.0, float(duration_s))
        self.last_s = duration
        self.max_s = max(self.max_s, duration)
        if duration <= self.threshold_s:
            return False
        self.over_threshold_count += 1
        self.last_over_threshold_at_unix_ns = int(unix_ns)
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "lastS": self.last_s,
            "maxS": self.max_s,
            "overThresholdCount": self.over_threshold_count,
            "lastOverThresholdAtUnixNs": self.last_over_threshold_at_unix_ns,
        }


class PlantRuntimeDiagnostics:
    """Attribute authority-loop stalls without executing render work there.

    ``observe_*`` has exactly one writer: the authoritative simulator thread.
    It publishes immutable dictionary generations with one CPython reference
    swap.  HTTP readers copy only the published generation and never touch the
    live probes or a lock used by the authority.
    """

    _STAGES = (
        "simStep",
        "viewerSync",
        "rgbRender",
        "jpegEncode",
        "depthRender",
        "capturePostprocess",
        "renderCaptureTotal",
        "snapshotPublish",
        "viewerSnapshotPublish",
    )

    def __init__(self, *, stall_threshold_s: float) -> None:
        threshold = float(stall_threshold_s)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("stall_threshold_s must be finite and positive")
        self.stall_threshold_s = threshold
        # HTTP readers may serialize their own defensive copies, but the
        # authority writer never touches this lock.  Published dictionaries
        # are never mutated after the atomic reference swap below.
        self._snapshot_reader_lock = threading.Lock()
        self._last_loop_start_s: float | None = None
        self._loop_gap = _DurationProbe(threshold_s=threshold)
        self._stages = {
            name: _DurationProbe(threshold_s=threshold) for name in self._STAGES
        }
        self._last_slow_stage: str | None = None
        self._last_slow_duration_s: float | None = None
        self._last_slow_at_unix_ns: int | None = None
        self._published_snapshot = self._build_snapshot()

    def observe_loop_start(self, at_s: float, *, unix_ns: int) -> None:
        current = float(at_s)
        previous = self._last_loop_start_s
        self._last_loop_start_s = current
        if previous is not None:
            slow = self._loop_gap.observe(current - previous, unix_ns=unix_ns)
            if slow:
                self._remember_slow(
                    "simLoopGap",
                    current - previous,
                    unix_ns,
                )
        self._publish_loop_snapshot()

    def observe_stage(
        self,
        stage: str,
        duration_s: float,
        *,
        unix_ns: int,
    ) -> None:
        if stage not in self._stages:
            raise ValueError(f"unknown plant timing stage: {stage}")
        if self._stages[stage].observe(duration_s, unix_ns=unix_ns):
            self._remember_slow(stage, duration_s, unix_ns)
        self._publish_stage_snapshot(stage)

    def _remember_slow(self, stage: str, duration_s: float, unix_ns: int) -> None:
        self._last_slow_stage = stage
        self._last_slow_duration_s = max(0.0, float(duration_s))
        self._last_slow_at_unix_ns = int(unix_ns)

    def _build_snapshot(self) -> dict[str, Any]:
        return {
            "stallThresholdS": self.stall_threshold_s,
            "simLoopGap": self._loop_gap.snapshot(),
            "stages": {
                name: probe.snapshot()
                for name, probe in self._stages.items()
            },
            "lastSlowStage": self._last_slow_stage,
            "lastSlowDurationS": self._last_slow_duration_s,
            "lastSlowAtUnixNs": self._last_slow_at_unix_ns,
        }

    def _published_common(self, current: dict[str, Any]) -> dict[str, Any]:
        return {
            **current,
            "lastSlowStage": self._last_slow_stage,
            "lastSlowDurationS": self._last_slow_duration_s,
            "lastSlowAtUnixNs": self._last_slow_at_unix_ns,
        }

    def _publish_loop_snapshot(self) -> None:
        current = self._published_snapshot
        self._published_snapshot = {
            **self._published_common(current),
            "simLoopGap": self._loop_gap.snapshot(),
        }

    def _publish_stage_snapshot(self, stage: str) -> None:
        current = self._published_snapshot
        stages = dict(current["stages"])
        stages[stage] = self._stages[stage].snapshot()
        self._published_snapshot = {
            **self._published_common(current),
            "stages": stages,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._snapshot_reader_lock:
            published = self._published_snapshot
            return {
                "stallThresholdS": published["stallThresholdS"],
                "simLoopGap": dict(published["simLoopGap"]),
                "stages": {
                    name: dict(value)
                    for name, value in published["stages"].items()
                },
                "lastSlowStage": published["lastSlowStage"],
                "lastSlowDurationS": published["lastSlowDurationS"],
                "lastSlowAtUnixNs": published["lastSlowAtUnixNs"],
            }
