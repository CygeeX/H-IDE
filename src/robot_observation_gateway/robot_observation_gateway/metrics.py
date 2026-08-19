"""Bounded counters and rolling latency statistics."""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass

from robot_observation_gateway.errors import DropReason


@dataclass(frozen=True)
class LatencySnapshot:
    current_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float


class RollingLatency:
    def __init__(self, *, max_samples: int = 512) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def observe(self, value_ms: float) -> None:
        if not math.isfinite(value_ms) or value_ms < 0:
            raise ValueError("latency must be finite and non-negative")
        with self._lock:
            self._samples.append(value_ms)

    def snapshot(self) -> LatencySnapshot:
        with self._lock:
            samples = sorted(self._samples)
        if not samples:
            return LatencySnapshot(0.0, 0.0, 0.0, 0.0)
        return LatencySnapshot(
            current_ms=samples[-1] if len(samples) == 1 else self.current,
            p50_ms=_percentile(samples, 0.50),
            p95_ms=_percentile(samples, 0.95),
            max_ms=samples[-1],
        )

    @property
    def current(self) -> float:
        with self._lock:
            return self._samples[-1] if self._samples else 0.0


class DropTracker:
    def __init__(self) -> None:
        self._counts: Counter[DropReason] = Counter()
        self._lock = threading.Lock()

    def increment(self, reason: DropReason, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("drop count increment must be positive")
        with self._lock:
            self._counts[reason] += count

    def count(self, reason: DropReason) -> int:
        with self._lock:
            return self._counts[reason]

    @property
    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    def snapshot(self) -> dict[DropReason, int]:
        with self._lock:
            return {reason: self._counts[reason] for reason in DropReason}


@dataclass(frozen=True)
class TrafficSnapshot:
    frames_per_second: float
    megabits_per_second: float


class TrafficWindow:
    def __init__(self, *, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0:
            raise ValueError("traffic window must be positive")
        self._window_ns = int(window_seconds * 1_000_000_000)
        self._samples: deque[tuple[int, int]] = deque()
        self._lock = threading.Lock()

    def observe(self, byte_count: int, *, monotonic_ns: int | None = None) -> None:
        if byte_count < 0:
            raise ValueError("byte count must be non-negative")
        now_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        with self._lock:
            self._samples.append((now_ns, byte_count))
            self._prune(now_ns)

    def snapshot(self, *, monotonic_ns: int | None = None) -> TrafficSnapshot:
        now_ns = time.monotonic_ns() if monotonic_ns is None else monotonic_ns
        with self._lock:
            self._prune(now_ns)
            samples = tuple(self._samples)
        if not samples:
            return TrafficSnapshot(0.0, 0.0)
        oldest_ns = samples[0][0]
        elapsed_seconds = max((now_ns - oldest_ns) / 1_000_000_000.0, 0.001)
        return TrafficSnapshot(
            frames_per_second=len(samples) / elapsed_seconds,
            megabits_per_second=(sum(sample[1] for sample in samples) * 8)
            / elapsed_seconds
            / 1_000_000.0,
        )

    def _prune(self, now_ns: int) -> None:
        cutoff = now_ns - self._window_ns
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()


def _percentile(sorted_samples: list[float], quantile: float) -> float:
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    index = (len(sorted_samples) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_samples[lower]
    weight = index - lower
    return sorted_samples[lower] * (1.0 - weight) + sorted_samples[upper] * weight
