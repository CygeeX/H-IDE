from __future__ import annotations

import threading
from collections import deque
from datetime import UTC, datetime

from h_ide_vision.contracts import ApproachPolicy

from .models import AcceptedObservation, ObservationBundle
from .validation import ObservationRejected, validate_bundle


class LatestFrameStore:
    def __init__(self, *, per_sensor_capacity: int = 3, policy: ApproachPolicy | None = None) -> None:
        if per_sensor_capacity < 1 or per_sensor_capacity > 10:
            raise ValueError("per_sensor_capacity must be between 1 and 10")
        self._capacity = per_sensor_capacity
        self._policy = policy or ApproachPolicy()
        self._frames: dict[tuple[str, str], deque[AcceptedObservation]] = {}
        self._last_sequence: dict[tuple[str, str, str], int] = {}
        self._active_session: dict[tuple[str, str], str] = {}
        self._revisions: dict[tuple[str, str], int] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def activate_session(self, robot_id: str, sensor_id: str, session_id: str) -> None:
        if not robot_id.strip() or not sensor_id.strip() or not session_id.strip():
            raise ValueError("observation session identity is incomplete")
        key = (robot_id, sensor_id)
        with self._lock:
            if self._active_session.get(key) == session_id:
                return
            self._active_session[key] = session_id
            self._frames.pop(key, None)
            self._last_sequence = {
                stored_key: sequence
                for stored_key, sequence in self._last_sequence.items()
                if stored_key[:2] != key
            }
            self._revisions[key] = self._revisions.get(key, 0) + 1
            self._condition.notify_all()

    def accept(self, bundle: ObservationBundle, *, now: datetime | None = None) -> AcceptedObservation:
        accepted_at = now or datetime.now(UTC)
        identifier = validate_bundle(bundle, self._policy, now=accepted_at)
        key = (bundle.identity.robot_id, bundle.identity.sensor_id)
        sequence_key = (*key, bundle.identity.observation_session_id)
        with self._lock:
            active_session = self._active_session.setdefault(
                key, bundle.identity.observation_session_id
            )
            if active_session != bundle.identity.observation_session_id:
                raise ObservationRejected(
                    "identity_mismatch", "observation session is not active"
                )
            current = self._frames.get(key)
            previous = self._last_sequence.get(sequence_key)
            if previous is not None and bundle.sequence <= previous:
                raise ObservationRejected("out_of_order", f"sequence {bundle.sequence} <= {previous}")
            if current:
                latest = current[-1].bundle
                if latest.identity.observation_session_id == bundle.identity.observation_session_id:
                    if latest.identity.robot_boot_id != bundle.identity.robot_boot_id:
                        raise ObservationRejected("identity_mismatch", "robot boot changed inside session")
                    if latest.identity.localization_epoch_id != bundle.identity.localization_epoch_id:
                        raise ObservationRejected(
                            "localization_mismatch",
                            "localization epoch changed inside session",
                        )
                    if latest.calibration_sha256 != bundle.calibration_sha256:
                        raise ObservationRejected(
                            "calibration_mismatch",
                            "calibration changed inside session",
                        )
            accepted = AcceptedObservation(bundle=bundle, accepted_at=accepted_at, bundle_id=identifier)
            queue = self._frames.setdefault(key, deque(maxlen=self._capacity))
            queue.append(accepted)
            self._last_sequence[sequence_key] = bundle.sequence
            self._revisions[key] = self._revisions.get(key, 0) + 1
            self._condition.notify_all()
            return accepted

    def latest(self, robot_id: str, sensor_id: str) -> tuple[AcceptedObservation, ...]:
        with self._lock:
            return tuple(self._frames.get((robot_id, sensor_id), ()))

    def latest_with_revision(
        self, robot_id: str, sensor_id: str
    ) -> tuple[tuple[AcceptedObservation, ...], int]:
        key = (robot_id, sensor_id)
        with self._lock:
            return tuple(self._frames.get(key, ())), self._revisions.get(key, 0)

    def wait_for_revision(
        self,
        robot_id: str,
        sensor_id: str,
        *,
        after_revision: int,
        timeout_seconds: float,
    ) -> bool:
        key = (robot_id, sensor_id)
        with self._condition:
            return self._condition.wait_for(
                lambda: self._revisions.get(key, 0) > after_revision,
                timeout=max(0.0, timeout_seconds),
            )
