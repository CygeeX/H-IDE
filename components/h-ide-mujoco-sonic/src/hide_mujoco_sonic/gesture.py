"""Simulation-only, single-owner gesture overlay for the G1 actuator stream.

The overlay is intentionally small and deterministic.  A request is accepted
by SONIC's existing command socket, then sampled by the same 50 Hz publisher
that owns ``rt/lowcmd``.  No HTTP thread, Plant thread, or second publisher
writes actuator messages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any
from uuid import UUID


class GestureError(ValueError):
    """A gesture request violates the bounded simulation contract."""


@dataclass(frozen=True, slots=True)
class GestureSnapshot:
    state: str
    gesture: str | None
    request_id: str | None
    progress: float
    started_at: float | None
    duration_s: float | None
    last_error: str | None


class GestureWaveController:
    """Thread-safe trajectory state; actuator sampling remains SONIC-owned."""

    _ARM_INDICES = (22, 23, 24, 25, 26, 27, 28)  # right shoulder/elbow/wrist

    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._state = "idle"
        self._gesture: str | None = None
        self._request_id: str | None = None
        self._started_at: float | None = None
        self._duration_s: float | None = None
        self._last_error: str | None = None
        self._replays: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _normalize_request_id(value: object) -> str:
        try:
            parsed = UUID(str(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise GestureError("gesture request_id must be a UUID") from exc
        if str(parsed) != str(value):
            raise GestureError("gesture request_id must be canonical")
        return str(value)

    def request_wave(
        self,
        *,
        request_id: object,
        gesture: object,
        duration_s: object,
        simulation_only: object,
        ready: bool,
    ) -> dict[str, Any]:
        request = self._normalize_request_id(request_id)
        if gesture != "wave":
            raise GestureError("only the wave gesture is available")
        if simulation_only is not True:
            raise GestureError("gesture requests must explicitly set simulation_only=true")
        try:
            duration = float(duration_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise GestureError("gesture duration must be finite") from exc
        if not math.isfinite(duration) or not 1.0 <= duration <= 4.0:
            raise GestureError("gesture duration must be between 1 and 4 seconds")
        with self._lock:
            replay = self._replays.get(request)
            if replay is not None:
                return dict(replay)
            if len(self._replays) >= 128:
                raise GestureError("gesture replay ledger is full")
            if not ready:
                self._state = "rejected"
                self._last_error = "wave is admitted only while the G1 is stationary"
                raise GestureError(self._last_error)
            if self._state == "running":
                raise GestureError("another gesture is already running")
            self._state = "running"
            self._gesture = "wave"
            self._request_id = request
            self._started_at = float(self._clock())
            self._duration_s = duration
            self._last_error = None
            response = {
                "status": "accepted",
                "gesture": "wave",
                "request_id": request,
                "duration_s": duration,
                "actuator_owner": "sonic",
                "simulation_only": True,
            }
            self._replays[request] = dict(response)
            return response

    def sample(self, targets: tuple[float, ...], *, now: float | None = None) -> tuple[tuple[float, ...], GestureSnapshot]:
        if len(targets) != 29 or not all(math.isfinite(float(value)) for value in targets):
            raise GestureError("gesture overlay requires 29 finite joint targets")
        with self._lock:
            if self._state != "running" or self._started_at is None or self._duration_s is None:
                return targets, self._snapshot_locked(progress=0.0)
            current = float(self._clock() if now is None else now)
            progress = max(0.0, min(1.0, (current - self._started_at) / self._duration_s))
            if progress >= 1.0:
                self._state = "completed"
                self._last_error = None
                return targets, self._snapshot_locked(progress=1.0)
            # Smooth envelope prevents a shoulder step at the start/end. Three
            # low-amplitude wrist cycles are visually readable without taking
            # the whole-body controller outside its nominal arm envelope.
            envelope = math.sin(math.pi * progress)
            phase = math.tau * 3.0 * progress
            values = list(targets)
            values[22] += -0.18 * envelope
            values[23] += -0.22 * envelope
            values[24] += 0.08 * envelope
            values[25] += -0.12 * envelope
            values[26] += 0.20 * envelope * math.sin(phase)
            values[27] += 0.12 * envelope * math.sin(phase + math.pi / 2)
            values[28] += 0.28 * envelope * math.sin(phase)
            result = tuple(float(value) for value in values)
            return result, self._snapshot_locked(progress=progress)

    def snapshot(self) -> GestureSnapshot:
        with self._lock:
            progress = 0.0
            if self._state == "running" and self._started_at is not None and self._duration_s:
                progress = max(0.0, min(1.0, (float(self._clock()) - self._started_at) / self._duration_s))
            return self._snapshot_locked(progress=progress)

    def cancel(self, reason: str) -> GestureSnapshot:
        detail = str(reason).strip()
        if not detail:
            detail = "gesture cancelled"
        with self._lock:
            if self._state == "running":
                self._state = "cancelled"
            self._last_error = detail[:512]
            return self._snapshot_locked(progress=0.0)

    def _snapshot_locked(self, *, progress: float) -> GestureSnapshot:
        return GestureSnapshot(
            state=self._state,
            gesture=self._gesture,
            request_id=self._request_id,
            progress=float(progress),
            started_at=self._started_at,
            duration_s=self._duration_s,
            last_error=self._last_error,
        )


__all__ = ["GestureError", "GestureSnapshot", "GestureWaveController"]
