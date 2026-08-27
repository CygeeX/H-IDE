"""Thread-safe normalized state storage and rate-limited change coalescing."""

from __future__ import annotations

import copy
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

_PATH_SEGMENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class StateEmission:
    kind: Literal["snapshot", "update"]
    sequence: int
    observed_at: datetime
    data: dict[str, Any]


class StateCoalescer:
    """Keep the latest normalized value and bound outbound update frequency."""

    def __init__(self, *, change_hz: float, snapshot_hz: float) -> None:
        if change_hz <= 0 or snapshot_hz <= 0:
            raise ValueError("state output rates must be positive")
        self._change_period = 1.0 / change_hz
        self._snapshot_period = 1.0 / snapshot_hz
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {}
        self._pending: dict[str, Any] = {}
        self._sequence = 0
        self._observed_at: datetime | None = None
        self._last_change: float | None = None
        self._last_snapshot: float | None = None
        self._last_poll: float | None = None

    def update(self, path: str, value: Any, *, observed_at: datetime) -> bool:
        """Apply one configured logical path; unchanged values are ignored."""
        _require_aware(observed_at)
        segments = path.split(".")
        if len(segments) < 2 or any(not _PATH_SEGMENT.fullmatch(part) for part in segments):
            raise ValueError(f"invalid normalized state path: {path!r}")
        _ensure_json(value)
        safe_value = copy.deepcopy(value)
        with self._lock:
            parent = self._state
            for segment in segments[:-1]:
                existing = parent.get(segment)
                if existing is None:
                    child: dict[str, Any] = {}
                    parent[segment] = child
                    parent = child
                elif isinstance(existing, dict):
                    parent = existing
                else:
                    raise ValueError(f"normalized state path {path!r} conflicts at {segment!r}")
            leaf = segments[-1]
            if leaf in parent and parent[leaf] == safe_value:
                return False
            parent[leaf] = safe_value
            self._pending[path] = copy.deepcopy(safe_value)
            self._observed_at = observed_at
            return True

    def poll(self, *, monotonic_now: float) -> list[StateEmission]:
        """Return due emissions, with a snapshot subsuming pending changes."""
        with self._lock:
            if self._last_poll is not None and monotonic_now < self._last_poll:
                raise ValueError("monotonic_now must not move backwards")
            self._last_poll = monotonic_now
            if self._observed_at is None:
                return []

            snapshot_due = (
                self._last_snapshot is None
                or monotonic_now - self._last_snapshot >= self._snapshot_period
            )
            if snapshot_due:
                self._last_snapshot = monotonic_now
                self._last_change = monotonic_now
                self._pending.clear()
                return [self._emission("snapshot", copy.deepcopy(self._state))]

            change_due = (
                bool(self._pending)
                and self._last_change is not None
                and monotonic_now - self._last_change >= self._change_period
            )
            if change_due:
                changes = copy.deepcopy(self._pending)
                self._pending.clear()
                self._last_change = monotonic_now
                return [self._emission("update", changes)]
            return []

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def _emission(self, kind: Literal["snapshot", "update"], data: dict[str, Any]) -> StateEmission:
        assert self._observed_at is not None
        self._sequence += 1
        return StateEmission(
            kind=kind,
            sequence=self._sequence,
            observed_at=self._observed_at,
            data=data,
        )


def _ensure_json(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"state value is not finite JSON: {exc}") from exc


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observed_at must include a timezone")
