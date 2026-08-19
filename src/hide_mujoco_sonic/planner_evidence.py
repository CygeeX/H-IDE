"""Generation-scoped evidence gate for SONIC's asynchronous planner.

The pinned LeRobot SONIC wrapper exposes the planner request boundary,
``gen_frame`` correlation value, returned ``PlannerMotion`` and live reference
cursor.  It does not expose a request id in the worker result.  This adapter
therefore gives every request a local generation, correlates the returned
motion by the upstream ``gen_frame``, and rejects ambiguous results.  It never
changes a velocity, planner input, reference, or actuator target.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import threading
from typing import Any

from .safety import MovementIntent


_MIN_DIRECTIONAL_PROJECTION_M = 1e-4


class PlannerReferenceDeadlineExceeded(RuntimeError):
    """A required Nav2 intent did not obtain a safe SONIC reference in time."""


def _signature(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class _IntentEvidence:
    generation: int
    signature: str
    has_movement: bool
    requested_direction: tuple[float, float] | None
    facing_direction: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _RequestEvidence:
    generation: int
    intent_generation: int | None
    intent_signature: str | None
    planner_signature: str
    gen_frame: int | None
    matches_intent: bool
    requested_direction: tuple[float, float] | None


class PlannerReferenceEvidenceGate:
    """Admit only a uniquely correlated, direction-consistent new reference.

    ``bind`` instruments the two public planner operations used by LeRobot's
    ``SonicRuntime``.  A returned plan is passed back to the runtime only when
    it maps to exactly one request for the currently required Nav2 intent.  For
    translational intents its predicted root displacement must also have a
    strictly positive projection on the requested direction.

    A pure Nav2 rotation has no requested root translation vector.  It still
    requires a generation-matched new plan, while its predicted root direction
    and displacement remain observable rather than being interpreted as a
    translation contract.
    """

    def __init__(self, *, required_timeout_s: float = 0.75) -> None:
        timeout = float(required_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("planner reference timeout must be finite and positive")
        self._lock = threading.RLock()
        self.required_timeout_s = timeout
        self._bound = False
        self._reference_owner: object | None = None
        self._intent_counter = 0
        self._request_counter = 0
        self._intent: _IntentEvidence | None = None
        self._intent_active = False
        self._reference_ready = False
        self._requests_by_frame: dict[int, list[_RequestEvidence]] = defaultdict(list)
        self._latest_request: _RequestEvidence | None = None
        self._accepted_request: _RequestEvidence | None = None
        self._predicted_direction: tuple[float, float] | None = None
        self._predicted_projection_m: float | None = None
        self._direction_consistent: bool | None = None
        self._required_started_at_s: float | None = None
        self._reason = "no Nav2 motion intent"

    @staticmethod
    def _intent_signature(intent: MovementIntent) -> str:
        return _signature(
            {
                "facing_angle_rad": float(intent.facing_angle_rad).hex(),
                "has_movement": bool(intent.has_movement),
                "mode": str(intent.mode),
                "movement_angle_rad": float(intent.movement_angle_rad).hex(),
                "source": str(intent.source),
                "speed_mps": float(intent.speed_mps).hex(),
                "stale": bool(intent.stale),
            }
        )

    @staticmethod
    def _planner_signature(ms: object) -> str:
        movement = tuple(float(value) for value in ms.movement_direction)
        facing = tuple(float(value) for value in ms.facing_direction)
        values = (
            float(ms.speed),
            float(ms.height),
            *movement,
            *facing,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SONIC planner request state must be finite")
        return _signature(
            {
                "facing_direction": [value.hex() for value in facing],
                "has_movement": bool(ms.has_movement),
                "height": float(ms.height).hex(),
                "mode": int(ms.mode),
                "movement_direction": [value.hex() for value in movement],
                "speed": float(ms.speed).hex(),
            }
        )

    @staticmethod
    def _planner_state_matches_intent(ms: object, intent: _IntentEvidence) -> bool:
        try:
            mode = int(ms.mode)
            has_movement = bool(ms.has_movement)
            movement = tuple(float(value) for value in ms.movement_direction)
            facing = tuple(float(value) for value in ms.facing_direction)
            speed = float(ms.speed)
        except (AttributeError, TypeError, ValueError):
            return False
        if (
            mode == 0
            or has_movement != intent.has_movement
            or len(movement) < 2
            or len(facing) < 2
            or not all(math.isfinite(value) for value in (*movement, *facing, speed))
        ):
            return False
        expected_facing = intent.facing_direction
        facing_dot = facing[0] * expected_facing[0] + facing[1] * expected_facing[1]
        if facing_dot < 1.0 - 1e-6:
            return False
        if intent.requested_direction is None:
            return math.hypot(movement[0], movement[1]) <= 1e-6
        expected = intent.requested_direction
        movement_dot = movement[0] * expected[0] + movement[1] * expected[1]
        return speed > 0.0 and movement_dot >= 1.0 - 1e-6

    def observe_intent(
        self,
        intent: MovementIntent,
        *,
        admitted: bool,
        simulation_now_s: float | None = None,
    ) -> None:
        """Observe one intent admitted (or debounced) by ``ReplanIntentGate``."""

        if not admitted:
            return
        if intent.mode == "IDLE":
            self.revoke("Nav2 intent is IDLE")
            return
        started_at: float | None = None
        if simulation_now_s is not None:
            started_at = float(simulation_now_s)
            if not math.isfinite(started_at) or started_at < 0:
                raise ValueError(
                    "planner reference generation requires finite non-negative MuJoCo time"
                )
        requested_direction = (
            (
                math.cos(float(intent.movement_angle_rad)),
                math.sin(float(intent.movement_angle_rad)),
            )
            if intent.has_movement
            else None
        )
        facing_direction = (
            math.cos(float(intent.facing_angle_rad)),
            math.sin(float(intent.facing_angle_rad)),
        )
        with self._lock:
            self._intent_counter += 1
            self._intent = _IntentEvidence(
                generation=self._intent_counter,
                signature=self._intent_signature(intent),
                has_movement=bool(intent.has_movement),
                requested_direction=requested_direction,
                facing_direction=facing_direction,
            )
            self._intent_active = True
            self._reference_ready = False
            self._accepted_request = None
            self._predicted_direction = None
            self._predicted_projection_m = None
            self._direction_consistent = None
            self._required_started_at_s = started_at
            self._reason = "awaiting a matching SONIC planner reference"

    def revoke(self, reason: str) -> None:
        """Revoke readiness without erasing the last evidence for diagnosis."""

        with self._lock:
            self._intent_active = False
            self._reference_ready = False
            self._required_started_at_s = None
            self._reason = str(reason)

    def _has_current_reference_locked(self) -> bool:
        intent = self._intent if self._intent_active else None
        accepted = self._accepted_request
        return bool(
            intent is not None
            and accepted is not None
            and accepted.matches_intent
            and accepted.intent_generation == intent.generation
            and accepted.intent_signature == intent.signature
        )

    def _reject_result_locked(self, reason: str) -> bool:
        """Reject one candidate without revoking a proven periodic predecessor."""

        if self._has_current_reference_locked():
            self._reference_ready = True
            self._reason = f"{reason}; retaining current accepted planner reference"
        else:
            self._reference_ready = False
            self._reason = reason
        return False

    @property
    def reference_ready(self) -> bool:
        with self._lock:
            return self._reference_ready and self._has_current_reference_locked()

    def enforce_required_deadline(self, *, simulation_now_s: float) -> None:
        """Fail closed when a required generation waits too long in MuJoCo time."""

        current = float(simulation_now_s)
        if not math.isfinite(current) or current < 0:
            raise ValueError(
                "planner reference deadline requires finite non-negative MuJoCo time"
            )
        failure: str | None = None
        with self._lock:
            if not self._intent_active or self._has_current_reference_locked():
                return
            started_at = self._required_started_at_s
            if started_at is None:
                failure = (
                    "required SONIC planner generation has no authoritative "
                    "MuJoCo start timestamp"
                )
            elif current < started_at:
                failure = (
                    "MuJoCo time moved backwards while awaiting a required "
                    "SONIC planner generation"
                )
            elif current - started_at >= self.required_timeout_s:
                intent_generation = (
                    None if self._intent is None else self._intent.generation
                )
                failure = (
                    "required SONIC planner reference timed out for intent "
                    f"generation {intent_generation} after "
                    f"{current - started_at:.3f}s MuJoCo time "
                    f"(limit={self.required_timeout_s:.3f}s)"
                )
            if failure is not None:
                self._reference_ready = False
                self._reason = failure
        if failure is not None:
            raise PlannerReferenceDeadlineExceeded(failure)

    def bind(self, planner: object, *, reference_owner: object) -> None:
        """Instrument one live LeRobot planner instance exactly once."""

        with self._lock:
            if self._bound:
                raise RuntimeError("planner evidence gate is already bound")
            request_replan = getattr(planner, "request_replan", None)
            try_get_new_motion = getattr(planner, "try_get_new_motion", None)
            if not callable(request_replan) or not callable(try_get_new_motion):
                raise RuntimeError("SONIC planner does not expose the required request/result API")
            raw_cursor = getattr(reference_owner, "ref_cursor", None)
            if isinstance(raw_cursor, bool):
                raise RuntimeError("SONIC reference owner does not expose ref_cursor")
            try:
                cursor = int(raw_cursor)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "SONIC reference owner does not expose ref_cursor"
                ) from exc
            if cursor < 0:
                raise RuntimeError("SONIC reference cursor must be non-negative")
            self._reference_owner = reference_owner
            self._bound = True

        def request_wrapper(cursor: object, ms: object) -> object:
            result = request_replan(cursor, ms)
            self._record_request(planner, ms)
            return result

        def result_wrapper() -> object:
            result = try_get_new_motion()
            if result is None:
                return None
            return result if self._accept_result(result) else None

        planner.request_replan = request_wrapper
        planner.try_get_new_motion = result_wrapper

    def _record_request(self, planner: object, ms: object) -> None:
        try:
            planner_signature = self._planner_signature(ms)
        except (AttributeError, TypeError, ValueError) as exc:
            planner_signature = _signature({"invalid": str(exc)})
        try:
            raw_gen_frame = getattr(planner, "gen_frame")
            gen_frame = int(raw_gen_frame)
            if isinstance(raw_gen_frame, bool) or gen_frame < 0:
                raise ValueError
        except (AttributeError, TypeError, ValueError):
            gen_frame = None

        with self._lock:
            self._request_counter += 1
            intent = self._intent if self._intent_active else None
            matches = bool(
                intent is not None
                and gen_frame is not None
                and self._planner_state_matches_intent(ms, intent)
            )
            request = _RequestEvidence(
                generation=self._request_counter,
                intent_generation=(None if intent is None else intent.generation),
                intent_signature=(None if intent is None else intent.signature),
                planner_signature=planner_signature,
                gen_frame=gen_frame,
                matches_intent=matches,
                requested_direction=(
                    None if intent is None else intent.requested_direction
                ),
            )
            self._latest_request = request
            if gen_frame is not None:
                self._requests_by_frame[gen_frame].append(request)
            self._prune_requests_locked()
            if intent is None:
                self._reason = "planner request has no active Nav2 motion intent"
            elif gen_frame is None:
                self._reason = "planner request did not expose a usable gen_frame"
            elif not matches:
                self._reason = "planner request state does not match the active Nav2 intent"
            elif len(self._requests_by_frame[gen_frame]) > 1:
                self._reason = f"planner gen_frame {gen_frame} is ambiguous"
            else:
                self._reason = "matching planner request is pending"

    def _prune_requests_locked(self) -> None:
        oldest = max(0, self._request_counter - 128)
        for frame, requests in tuple(self._requests_by_frame.items()):
            retained = [request for request in requests if request.generation >= oldest]
            if retained:
                self._requests_by_frame[frame] = retained
            else:
                del self._requests_by_frame[frame]

    @staticmethod
    def _predicted_root_motion(
        motion: object,
        requested_direction: tuple[float, float] | None,
    ) -> tuple[tuple[float, float] | None, float | None, bool]:
        try:
            timesteps = int(motion.timesteps)
            if timesteps < 2:
                raise ValueError("planner motion has fewer than two frames")
            first = motion.body_positions[0]
            last = motion.body_positions[timesteps - 1]
            dx = float(last[0]) - float(first[0])
            dy = float(last[1]) - float(first[1])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("planner motion has no usable root trajectory") from exc
        if not all(math.isfinite(value) for value in (dx, dy)):
            raise ValueError("planner root trajectory must be finite")
        distance = math.hypot(dx, dy)
        predicted_direction = (
            None if distance <= 1e-12 else (dx / distance, dy / distance)
        )
        if requested_direction is None:
            return predicted_direction, None, True
        projection = dx * requested_direction[0] + dy * requested_direction[1]
        return (
            predicted_direction,
            projection,
            projection > _MIN_DIRECTIONAL_PROJECTION_M,
        )

    def _accept_result(self, result: object) -> bool:
        try:
            motion, raw_gen_frame = result
            gen_frame = int(raw_gen_frame)
            if isinstance(raw_gen_frame, bool) or gen_frame < 0:
                raise ValueError
        except (TypeError, ValueError):
            with self._lock:
                self._reject_result_locked(
                    "planner result does not expose (motion, gen_frame)"
                )
            return False

        with self._lock:
            requests = self._requests_by_frame.pop(gen_frame, [])
            intent = self._intent if self._intent_active else None
            if len(requests) != 1:
                return self._reject_result_locked(
                    f"planner result gen_frame {gen_frame} has "
                    f"{len(requests)} matching requests"
                )
            request = requests[0]
            current_matches = bool(
                intent is not None
                and request.matches_intent
                and request.intent_generation == intent.generation
                and request.intent_signature == intent.signature
            )
            if not current_matches:
                return self._reject_result_locked(
                    "planner result belongs to a stale or mismatched intent"
                )
            reference_owner = self._reference_owner
            try:
                raw_cursor = getattr(reference_owner, "ref_cursor")
                ref_cursor = int(raw_cursor)
                timesteps = int(motion.timesteps)
                if (
                    isinstance(raw_cursor, bool)
                    or ref_cursor < 0
                    or timesteps < 2
                    or gen_frame - ref_cursor + timesteps <= 0
                ):
                    raise ValueError
            except (AttributeError, TypeError, ValueError):
                return self._reject_result_locked(
                    "planner result cannot be blended at the live reference cursor"
                )
            try:
                direction, projection, consistent = self._predicted_root_motion(
                    motion,
                    request.requested_direction,
                )
            except ValueError as exc:
                if not self._has_current_reference_locked():
                    self._predicted_direction = None
                    self._predicted_projection_m = None
                    self._direction_consistent = False
                return self._reject_result_locked(str(exc))
            if not consistent:
                if not self._has_current_reference_locked():
                    self._predicted_direction = direction
                    self._predicted_projection_m = projection
                    self._direction_consistent = False
                return self._reject_result_locked(
                    "planner root projection opposes or does not realize the Nav2 direction"
                )
            self._predicted_direction = direction
            self._predicted_projection_m = projection
            self._direction_consistent = True
            self._accepted_request = request
            self._reference_ready = True
            self._reason = "matching direction-consistent planner reference accepted"
            return True

    def telemetry_fields(self, *, ref_cursor: int | None) -> dict[str, Any]:
        with self._lock:
            intent = self._intent
            request = self._latest_request
            accepted = self._accepted_request
            requested = None if intent is None else intent.requested_direction
            predicted = self._predicted_direction
            return {
                "sonic_intent_generation": None if intent is None else intent.generation,
                "sonic_intent_signature": None if intent is None else intent.signature,
                "sonic_planner_request_generation": (
                    None if request is None else request.generation
                ),
                "sonic_planner_request_intent_generation": (
                    None if request is None else request.intent_generation
                ),
                "sonic_planner_request_intent_signature": (
                    None if request is None else request.intent_signature
                ),
                "sonic_planner_request_signature": (
                    None if request is None else request.planner_signature
                ),
                "sonic_planner_request_gen_frame": (
                    None if request is None else request.gen_frame
                ),
                "sonic_planner_request_matches_intent": (
                    None if request is None else request.matches_intent
                ),
                "sonic_planner_accepted_generation": (
                    None if accepted is None else accepted.generation
                ),
                "sonic_planner_accepted_intent_generation": (
                    None if accepted is None else accepted.intent_generation
                ),
                "sonic_planner_accepted_intent_signature": (
                    None if accepted is None else accepted.intent_signature
                ),
                "sonic_planner_accepted_signature": (
                    None if accepted is None else accepted.planner_signature
                ),
                "sonic_planner_reference_ready": (
                    self._reference_ready and self._has_current_reference_locked()
                ),
                "sonic_planner_required_pending": (
                    self._intent_active and not self._has_current_reference_locked()
                ),
                "sonic_planner_required_started_at_s": self._required_started_at_s,
                "sonic_planner_required_timeout_s": self.required_timeout_s,
                "sonic_planner_required_deadline_s": (
                    None
                    if self._required_started_at_s is None
                    else self._required_started_at_s + self.required_timeout_s
                ),
                "sonic_planner_ref_cursor": ref_cursor,
                "sonic_planner_requested_root_direction_x": (
                    None if requested is None else requested[0]
                ),
                "sonic_planner_requested_root_direction_y": (
                    None if requested is None else requested[1]
                ),
                "sonic_planner_predicted_root_direction_x": (
                    None if predicted is None else predicted[0]
                ),
                "sonic_planner_predicted_root_direction_y": (
                    None if predicted is None else predicted[1]
                ),
                "sonic_planner_predicted_root_projection_m": (
                    self._predicted_projection_m
                ),
                "sonic_planner_direction_consistent": self._direction_consistent,
                "sonic_planner_evidence_reason": self._reason,
            }

    def trace_fields(self, *, ref_cursor: int | None) -> dict[str, Any]:
        telemetry = self.telemetry_fields(ref_cursor=ref_cursor)
        return {
            key.removeprefix("sonic_"): value
            for key, value in telemetry.items()
        }


__all__ = [
    "PlannerReferenceDeadlineExceeded",
    "PlannerReferenceEvidenceGate",
]
