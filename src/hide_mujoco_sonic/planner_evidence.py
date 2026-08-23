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
_MAX_REFERENCE_PREFIX_REVERSE_M = 0.03
_REFERENCE_HZ = 50.0
# The locked planner_sonic.onnx revision 9c0ff22 nominal-context baseline
# produced 0.004016 rad of root-yaw drift for a zero facing request, while the
# smallest explicitly verified +/-0.05 rad requests produced
# -0.037168/+0.047419 rad. Keep a conservative deadband between those
# observations so planner noise cannot be mistaken for evidence that a pure
# Nav2 rotation is turning the requested way.
_MIN_MEANINGFUL_YAW_RAD = 0.01
# An official IDLE plan is a braking transition, not a frozen pose.  The
# pinned planner produces a 45.7 mm root envelope after a 3 m forward gait and
# the first live controlled stop produced 49.6 mm.  A later exact 0.2 m
# DriveOnHeading stops have produced 51.2 mm / 0.037945 rad (0.2 m task) and
# 62.13 mm / 0.060183 rad (1 m task) while their final 200 ms settled below the
# stationary-speed contract.  Requiring <=30 mm rejected valid official IDLE
# braking references and then misreported the local rejection as a planner
# timeout.  The continuous-history reference after a verified 90-degree Spin
# additionally reported -0.714 rad between its internal first/last root
# quaternions, while the cached NVIDIA controller plus MuJoCo retained the
# completed heading within 0.20 rad for five unsupported seconds.  That
# quaternion delta is therefore not an actuator-space stop-heading invariant
# for the vendor's continuous-IDLE/readaptation path.  Keep it as evidence, but
# gate the reference on bounded translation, per-frame speed and a quiet tail;
# measured heading/action completion and runtime tilt/collision gates remain
# authoritative for physical pose.
_MAX_STATIC_REFERENCE_TRANSLATION_M = 0.08
_MAX_STATIC_REFERENCE_FRAME_SPEED_MPS = 0.40
_MAX_STATIC_REFERENCE_TAIL_SPEED_MPS = 0.025
_STATIC_REFERENCE_SETTLE_FRAMES = 10


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
    mode: str
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
    requested_facing_angle_rad: float | None


@dataclass(frozen=True, slots=True)
class PlannerAuthorityEvidence:
    """Immutable correlation proof consumed by actuator-authority code."""

    reference_ready: bool
    intent_generation: int | None
    intent_signature: str | None
    accepted_request_generation: int | None
    accepted_intent_generation: int | None
    accepted_intent_signature: str | None


class PlannerReferenceEvidenceGate:
    """Admit only a uniquely correlated, direction-consistent new reference.

    ``bind`` instruments the two public planner operations used by LeRobot's
    ``SonicRuntime``.  A returned plan is passed back to the runtime only when
    it maps to exactly one request for the currently required Nav2 intent.  For
    translational intents its predicted root displacement must also have a
    strictly positive projection on the requested direction.

    A pure Nav2 rotation has no requested root translation vector.  It requires
    both a generation-matched plan and meaningful root-yaw motion whose shortest
    signed delta agrees with the requested facing change.  Translational plans
    retain their root-projection contract; yaw remains diagnostic for them and
    never replaces that projection gate. A controlled stop may explicitly ask
    for the official IDLE reference; that separate path accepts only a bounded
    static root envelope and is never enabled by an ordinary zero command.
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
        self._predicted_prefix_projections_m: tuple[float, ...] | None = None
        self._predicted_prefix_min_projection_m: float | None = None
        self._predicted_prefix_max_projection_m: float | None = None
        self._predicted_first_positive_frame: int | None = None
        self._predicted_first_positive_time_s: float | None = None
        self._direction_consistent: bool | None = None
        self._requested_yaw_error_rad: float | None = None
        self._predicted_yaw_delta_rad: float | None = None
        self._yaw_consistent: bool | None = None
        self._static_root_max_translation_m: float | None = None
        self._static_root_yaw_delta_rad: float | None = None
        self._static_root_max_speed_mps: float | None = None
        self._static_root_tail_max_speed_mps: float | None = None
        self._static_root_tail_translation_m: float | None = None
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
            has_movement != intent.has_movement
            or len(movement) < 2
            or len(facing) < 2
            or not all(math.isfinite(value) for value in (*movement, *facing, speed))
        ):
            return False
        expected_facing = intent.facing_direction
        facing_dot = facing[0] * expected_facing[0] + facing[1] * expected_facing[1]
        if facing_dot < 1.0 - 1e-6:
            return False
        if intent.mode == "IDLE":
            return bool(
                mode == 0
                and not has_movement
                and speed < 0.0
                and math.hypot(movement[0], movement[1]) <= 1e-6
            )
        if intent.requested_direction is None:
            # NVIDIA's pinned ROS2 input adapter represents a pure facing
            # change as an IDLE planner request with a zero movement vector.
            # The outer MovementIntent remains active SLOW_WALK so command
            # ownership, watchdogs and stop handling cannot mistake Spin for
            # an ordinary static hold.
            return bool(
                mode == 0
                and speed < 0.0
                and math.hypot(movement[0], movement[1]) <= 1e-6
            )
        if mode == 0:
            return False
        expected = intent.requested_direction
        movement_dot = movement[0] * expected[0] + movement[1] * expected[1]
        return speed > 0.0 and movement_dot >= 1.0 - 1e-6

    def observe_intent(
        self,
        intent: MovementIntent,
        *,
        admitted: bool,
        simulation_now_s: float | None = None,
        allow_static_reference: bool = False,
    ) -> None:
        """Observe one intent admitted (or debounced) by ``ReplanIntentGate``."""

        if not admitted:
            return
        if not isinstance(allow_static_reference, bool):
            raise TypeError("allow_static_reference must be a boolean")
        if intent.mode == "IDLE" and not allow_static_reference:
            self.revoke("Nav2 intent is IDLE")
            return
        if allow_static_reference and intent.mode != "IDLE":
            raise ValueError("static reference authority requires an IDLE intent")
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
                mode=str(intent.mode),
                has_movement=bool(intent.has_movement),
                requested_direction=requested_direction,
                facing_direction=facing_direction,
            )
            self._intent_active = True
            self._reference_ready = False
            self._accepted_request = None
            self._predicted_direction = None
            self._predicted_projection_m = None
            self._predicted_prefix_projections_m = None
            self._predicted_prefix_min_projection_m = None
            self._predicted_prefix_max_projection_m = None
            self._predicted_first_positive_frame = None
            self._predicted_first_positive_time_s = None
            self._direction_consistent = None
            self._requested_yaw_error_rad = None
            self._predicted_yaw_delta_rad = None
            self._yaw_consistent = None
            self._static_root_max_translation_m = None
            self._static_root_yaw_delta_rad = None
            self._static_root_max_speed_mps = None
            self._static_root_tail_max_speed_mps = None
            self._static_root_tail_translation_m = None
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

    @property
    def authority_evidence(self) -> PlannerAuthorityEvidence:
        """Return one typed snapshot of the currently matched reference proof."""

        with self._lock:
            intent = self._intent if self._intent_active else None
            accepted = self._accepted_request
            ready = self._reference_ready and self._has_current_reference_locked()
            return PlannerAuthorityEvidence(
                reference_ready=ready,
                intent_generation=None if intent is None else intent.generation,
                intent_signature=None if intent is None else intent.signature,
                accepted_request_generation=(
                    None if accepted is None else accepted.generation
                ),
                accepted_intent_generation=(
                    None if accepted is None else accepted.intent_generation
                ),
                accepted_intent_signature=(
                    None if accepted is None else accepted.intent_signature
                ),
            )

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
        try:
            facing = tuple(float(value) for value in ms.facing_direction)
            if len(facing) < 2 or not all(
                math.isfinite(value) for value in facing[:2]
            ):
                raise ValueError
            requested_facing_angle_rad = math.atan2(facing[1], facing[0])
        except (AttributeError, TypeError, ValueError):
            requested_facing_angle_rad = None

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
                requested_facing_angle_rad=requested_facing_angle_rad,
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
    ) -> tuple[
        tuple[float, float] | None,
        float | None,
        tuple[float, ...] | None,
        float | None,
        float | None,
        int | None,
        float | None,
        bool,
    ]:
        try:
            timesteps = int(motion.timesteps)
            if timesteps < 2:
                raise ValueError("planner motion has fewer than two frames")
            positions = tuple(
                (float(motion.body_positions[index][0]), float(motion.body_positions[index][1]))
                for index in range(timesteps)
            )
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("planner motion has no usable root trajectory") from exc
        if not all(
            math.isfinite(value)
            for position in positions
            for value in position
        ):
            raise ValueError("planner root trajectory must be finite")
        first_x, first_y = positions[0]
        dx = positions[-1][0] - first_x
        dy = positions[-1][1] - first_y
        distance = math.hypot(dx, dy)
        predicted_direction = (
            None if distance <= 1e-12 else (dx / distance, dy / distance)
        )
        if requested_direction is None:
            return predicted_direction, None, None, None, None, None, None, True
        prefix = tuple(
            (x - first_x) * requested_direction[0]
            + (y - first_y) * requested_direction[1]
            for x, y in positions
        )
        projection = prefix[-1]
        prefix_min = min(prefix)
        prefix_max = max(prefix)
        first_positive_frame = next(
            (
                index
                for index, value in enumerate(prefix)
                if value > _MIN_DIRECTIONAL_PROJECTION_M
            ),
            None,
        )
        first_positive_time_s = (
            None
            if first_positive_frame is None
            else first_positive_frame / _REFERENCE_HZ
        )
        return (
            predicted_direction,
            projection,
            prefix,
            prefix_min,
            prefix_max,
            first_positive_frame,
            first_positive_time_s,
            (
                projection > _MIN_DIRECTIONAL_PROJECTION_M
                and prefix_min >= -_MAX_REFERENCE_PREFIX_REVERSE_M
            ),
        )

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _quaternion_yaw(value: object) -> float:
        """Extract yaw from one finite, non-degenerate scalar-first quaternion."""

        try:
            quaternion = tuple(float(component) for component in value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "planner root quaternion is not a numeric wxyz vector"
            ) from exc
        if len(quaternion) != 4 or not all(
            math.isfinite(component) for component in quaternion
        ):
            raise ValueError("planner root quaternion must contain four finite values")
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm <= 1e-12:
            raise ValueError("planner root quaternion must be non-degenerate")
        w, x, y, z = (component / norm for component in quaternion)
        yaw = math.atan2(
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (y * y + z * z),
        )
        if not math.isfinite(yaw):
            raise ValueError("planner root quaternion produced a non-finite yaw")
        return yaw

    @classmethod
    def _predicted_root_yaw(
        cls,
        motion: object,
        requested_facing_angle_rad: float | None,
    ) -> tuple[float, float, bool | None]:
        """Return shortest requested/output yaw deltas and their direction match."""

        if requested_facing_angle_rad is None or not math.isfinite(
            requested_facing_angle_rad
        ):
            raise ValueError("planner request has no usable facing angle")
        try:
            timesteps = int(motion.timesteps)
            quaternions = motion.body_quaternions
            first_quaternion = quaternions[0]
            last_quaternion = quaternions[timesteps - 1]
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("planner motion has no usable root yaw trajectory") from exc
        if timesteps < 2:
            raise ValueError("planner motion has no usable root yaw trajectory")
        first_yaw = cls._quaternion_yaw(first_quaternion)
        last_yaw = cls._quaternion_yaw(last_quaternion)
        requested_error = cls._wrap_angle(
            requested_facing_angle_rad - first_yaw
        )
        predicted_delta = cls._wrap_angle(last_yaw - first_yaw)
        if abs(requested_error) <= _MIN_MEANINGFUL_YAW_RAD:
            consistent: bool | None = None
        elif abs(predicted_delta) <= _MIN_MEANINGFUL_YAW_RAD:
            consistent = False
        else:
            consistent = requested_error * predicted_delta > 0.0
        return requested_error, predicted_delta, consistent

    @classmethod
    def _predicted_static_root_motion(
        cls,
        motion: object,
    ) -> tuple[float, float, float, float, float]:
        """Measure one bounded, settling IDLE braking trajectory."""

        try:
            timesteps = int(motion.timesteps)
            if timesteps < _STATIC_REFERENCE_SETTLE_FRAMES + 1:
                raise ValueError
            positions = tuple(
                (float(motion.body_positions[index][0]), float(motion.body_positions[index][1]))
                for index in range(timesteps)
            )
            quaternions = motion.body_quaternions
            first_yaw = cls._quaternion_yaw(quaternions[0])
            last_yaw = cls._quaternion_yaw(quaternions[timesteps - 1])
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                "static planner motion has no usable root trajectory"
            ) from exc
        if not all(
            math.isfinite(value)
            for position in positions
            for value in position
        ):
            raise ValueError("static planner root trajectory must be finite")
        first_x, first_y = positions[0]
        max_translation = max(
            math.hypot(x - first_x, y - first_y) for x, y in positions
        )
        frame_speeds = tuple(
            math.hypot(x1 - x0, y1 - y0) * _REFERENCE_HZ
            for (x0, y0), (x1, y1) in zip(
                positions,
                positions[1:],
                strict=False,
            )
        )
        tail_speeds = frame_speeds[-_STATIC_REFERENCE_SETTLE_FRAMES:]
        tail_start = positions[-(_STATIC_REFERENCE_SETTLE_FRAMES + 1)]
        tail_end = positions[-1]
        tail_translation = math.hypot(
            tail_end[0] - tail_start[0],
            tail_end[1] - tail_start[1],
        )
        yaw_delta = cls._wrap_angle(last_yaw - first_yaw)
        return (
            max_translation,
            yaw_delta,
            max(frame_speeds),
            max(tail_speeds),
            tail_translation,
        )

    def _store_rejected_yaw_locked(
        self,
        requested_error: float | None,
        predicted_delta: float | None,
        consistent: bool | None,
    ) -> None:
        """Expose a rejected required candidate without erasing proven refresh evidence."""

        if self._has_current_reference_locked():
            return
        self._requested_yaw_error_rad = requested_error
        self._predicted_yaw_delta_rad = predicted_delta
        self._yaw_consistent = consistent

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
            static_intent = bool(intent is not None and intent.mode == "IDLE")
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
                (
                    direction,
                    projection,
                    prefix_projections,
                    prefix_min,
                    prefix_max,
                    first_positive_frame,
                    first_positive_time_s,
                    consistent,
                ) = self._predicted_root_motion(
                    motion,
                    request.requested_direction,
                )
            except ValueError as exc:
                if not self._has_current_reference_locked():
                    self._predicted_direction = None
                    self._predicted_projection_m = None
                    self._predicted_prefix_projections_m = None
                    self._predicted_prefix_min_projection_m = None
                    self._predicted_prefix_max_projection_m = None
                    self._predicted_first_positive_frame = None
                    self._predicted_first_positive_time_s = None
                    self._direction_consistent = False
                return self._reject_result_locked(str(exc))
            if not consistent:
                if not self._has_current_reference_locked():
                    self._predicted_direction = direction
                    self._predicted_projection_m = projection
                    self._predicted_prefix_projections_m = prefix_projections
                    self._predicted_prefix_min_projection_m = prefix_min
                    self._predicted_prefix_max_projection_m = prefix_max
                    self._predicted_first_positive_frame = first_positive_frame
                    self._predicted_first_positive_time_s = first_positive_time_s
                    self._direction_consistent = False
                if (
                    prefix_min is not None
                    and prefix_min < -_MAX_REFERENCE_PREFIX_REVERSE_M
                ):
                    return self._reject_result_locked(
                        "planner root prefix exceeds the 0.030m reverse "
                        "reference boundary"
                    )
                return self._reject_result_locked(
                    "planner root projection opposes or does not realize the Nav2 direction"
                )
            if static_intent:
                try:
                    (
                        static_translation,
                        static_yaw,
                        static_max_speed,
                        static_tail_max_speed,
                        static_tail_translation,
                    ) = (
                        self._predicted_static_root_motion(motion)
                    )
                except ValueError as exc:
                    return self._reject_result_locked(str(exc))
                self._static_root_max_translation_m = static_translation
                self._static_root_yaw_delta_rad = static_yaw
                self._static_root_max_speed_mps = static_max_speed
                self._static_root_tail_max_speed_mps = static_tail_max_speed
                self._static_root_tail_translation_m = static_tail_translation
                if static_translation > _MAX_STATIC_REFERENCE_TRANSLATION_M:
                    return self._reject_result_locked(
                        "IDLE planner braking translation exceeds the 0.080m envelope"
                    )
                if static_max_speed > _MAX_STATIC_REFERENCE_FRAME_SPEED_MPS:
                    return self._reject_result_locked(
                        "IDLE planner root speed exceeds the 0.400m/s braking envelope"
                    )
                if static_tail_max_speed > _MAX_STATIC_REFERENCE_TAIL_SPEED_MPS:
                    return self._reject_result_locked(
                        "IDLE planner tail does not settle below 0.025m/s"
                    )
            requested_yaw_error: float | None = None
            predicted_yaw_delta: float | None = None
            yaw_consistent: bool | None = None
            try:
                (
                    requested_yaw_error,
                    predicted_yaw_delta,
                    yaw_consistent,
                ) = self._predicted_root_yaw(
                    motion,
                    request.requested_facing_angle_rad,
                )
            except ValueError as exc:
                if request.requested_direction is None and not static_intent:
                    self._store_rejected_yaw_locked(None, None, False)
                    return self._reject_result_locked(str(exc))
                # Translation authority remains its positive root projection;
                # missing yaw is diagnostic only and cannot weaken that gate.
            if request.requested_direction is None and not static_intent:
                if (
                    requested_yaw_error is None
                    or abs(requested_yaw_error) <= _MIN_MEANINGFUL_YAW_RAD
                ):
                    self._store_rejected_yaw_locked(
                        requested_yaw_error,
                        predicted_yaw_delta,
                        False,
                    )
                    return self._reject_result_locked(
                        "pure-rotation planner request has no meaningful yaw error"
                    )
                if (
                    predicted_yaw_delta is None
                    or abs(predicted_yaw_delta) <= _MIN_MEANINGFUL_YAW_RAD
                ):
                    self._store_rejected_yaw_locked(
                        requested_yaw_error,
                        predicted_yaw_delta,
                        False,
                    )
                    return self._reject_result_locked(
                        "pure-rotation planner output has no meaningful root yaw"
                    )
                if yaw_consistent is not True:
                    self._store_rejected_yaw_locked(
                        requested_yaw_error,
                        predicted_yaw_delta,
                        False,
                    )
                    return self._reject_result_locked(
                        "planner root yaw opposes the pure-rotation request"
                    )
            self._predicted_direction = direction
            self._predicted_projection_m = projection
            self._predicted_prefix_projections_m = prefix_projections
            self._predicted_prefix_min_projection_m = prefix_min
            self._predicted_prefix_max_projection_m = prefix_max
            self._predicted_first_positive_frame = first_positive_frame
            self._predicted_first_positive_time_s = first_positive_time_s
            self._direction_consistent = True
            self._requested_yaw_error_rad = requested_yaw_error
            self._predicted_yaw_delta_rad = predicted_yaw_delta
            self._yaw_consistent = yaw_consistent
            self._accepted_request = request
            self._reference_ready = True
            self._reason = "matching planner reference evidence accepted"
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
                "sonic_planner_root_prefix_projections_m": (
                    None
                    if self._predicted_prefix_projections_m is None
                    else list(self._predicted_prefix_projections_m)
                ),
                "sonic_planner_root_prefix_min_projection_m": (
                    self._predicted_prefix_min_projection_m
                ),
                "sonic_planner_root_prefix_max_projection_m": (
                    self._predicted_prefix_max_projection_m
                ),
                "sonic_planner_root_first_positive_frame": (
                    self._predicted_first_positive_frame
                ),
                "sonic_planner_root_first_positive_time_s": (
                    self._predicted_first_positive_time_s
                ),
                "sonic_planner_direction_consistent": self._direction_consistent,
                "sonic_planner_requested_yaw_error_rad": (
                    self._requested_yaw_error_rad
                ),
                "sonic_planner_predicted_yaw_delta_rad": (
                    self._predicted_yaw_delta_rad
                ),
                "sonic_planner_yaw_consistent": self._yaw_consistent,
                "sonic_planner_static_root_max_translation_m": (
                    self._static_root_max_translation_m
                ),
                "sonic_planner_static_root_yaw_delta_rad": (
                    self._static_root_yaw_delta_rad
                ),
                "sonic_planner_static_root_max_speed_mps": (
                    self._static_root_max_speed_mps
                ),
                "sonic_planner_static_root_tail_max_speed_mps": (
                    self._static_root_tail_max_speed_mps
                ),
                "sonic_planner_static_root_tail_translation_m": (
                    self._static_root_tail_translation_m
                ),
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
