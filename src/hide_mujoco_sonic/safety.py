"""The only allowed conversion from Nav2 velocity to SONIC intent."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections.abc import Sequence


@dataclass(frozen=True)
class CmdVel:
    vx: float
    vy: float
    wz: float
    received_at: float
    source: str = "nav2"


@dataclass(frozen=True)
class MovementIntent:
    mode: str
    speed_mps: float
    movement_angle_rad: float
    facing_angle_rad: float
    has_movement: bool
    stale: bool
    source: str = "nav2"


@dataclass(frozen=True)
class LateralHeadingCorrection:
    """Bounded decoder correction for a pure lateral SONIC action."""

    heading_error_rad: float
    decoder_delta_heading_rad: float


class PlantTickGuard:
    """Detect a restarted MuJoCo publisher from Unitree ``LowState.tick``.

    The simulator publishes elapsed milliseconds in the unsigned tick field.
    A small backwards step may be a delayed DDS sample, while a large rollback
    means that the authoritative physics process restarted and every SONIC
    history/heading anchor is invalid.
    """

    def __init__(self, *, rollback_threshold_ms: int = 1_000) -> None:
        if isinstance(rollback_threshold_ms, bool) or rollback_threshold_ms <= 0:
            raise ValueError("rollback threshold must be a positive integer")
        self.rollback_threshold_ms = int(rollback_threshold_ms)
        self.last_tick: int | None = None
        self.restart_count = 0

    def observe(self, tick: int) -> bool:
        if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick <= 0xFFFFFFFF:
            raise ValueError("plant tick must be an unsigned 32-bit integer")
        previous = self.last_tick
        restarted = previous is not None and tick + self.rollback_threshold_ms < previous
        if restarted:
            self.restart_count += 1
            self.last_tick = tick
            return True
        if previous is None or tick >= previous:
            self.last_tick = tick
        return False


class PlantControlClockFault(RuntimeError):
    """The MuJoCo time source cannot safely advance the learned controller."""


@dataclass(frozen=True)
class PlantControlStep:
    """One SONIC step admitted by the authoritative MuJoCo clock."""

    tick_ms: int
    dt_s: float


class PlantControlClock:
    """Down-sample strict ``LowState.tick`` time to one SONIC step at 50 Hz.

    ``LowState.tick`` is elapsed MuJoCo time in milliseconds.  The simulation
    can run slower than wall time when RGB-D rendering is enabled, so a wall
    timer must never advance SONIC's reference cursor, history, handoff, or
    joint slew guards.  This gate admits exactly one control step after one
    simulation-time period and never emits catch-up steps.

    Duplicate/backwards observations, a missing source interval, or a control
    interval larger than the configured tolerance are terminal faults.  A
    caller must likewise fault if an admitted step is still pending when the
    next one arrives; buffering/catching up learned-policy steps would replay
    stale observations into the plant.
    """

    def __init__(
        self,
        *,
        control_hz: int = 50,
        max_source_gap_ms: int | None = None,
        control_tolerance_ms: int = 4,
    ) -> None:
        if isinstance(control_hz, bool) or int(control_hz) <= 0:
            raise ValueError("control_hz must be a positive integer")
        if 1_000 % int(control_hz) != 0:
            raise ValueError("control_hz must divide 1000 into an integer millisecond period")
        if isinstance(control_tolerance_ms, bool) or int(control_tolerance_ms) < 0:
            raise ValueError("control_tolerance_ms must be a non-negative integer")
        self.control_hz = int(control_hz)
        self.period_ms = 1_000 // self.control_hz
        source_gap = self.period_ms if max_source_gap_ms is None else max_source_gap_ms
        if isinstance(source_gap, bool) or int(source_gap) <= 0:
            raise ValueError("max_source_gap_ms must be a positive integer")
        self.max_source_gap_ms = int(source_gap)
        self.control_tolerance_ms = int(control_tolerance_ms)
        self.last_observed_tick_ms: int | None = None
        self.last_control_tick_ms: int | None = None

    @staticmethod
    def _tick(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
            raise PlantControlClockFault("LowState.tick must be an unsigned 32-bit integer")
        return value

    def reset(self) -> None:
        """Forget the startup stream before arming policy advancement."""

        self.last_observed_tick_ms = None
        self.last_control_tick_ms = None

    def rebase_stationary_qualification(self, tick_ms: int) -> None:
        """Rebase only a policy-free startup qualification on its latest tick.

        Model/GPU warmup can temporarily prevent the Python DDS callback from
        observing intermediate LowState samples while MuJoCo continues to
        publish a stationary held robot.  The caller is responsible for
        proving that motion is still disabled and invalidating every pending
        qualification sample.  Runtime policy control must always use
        :meth:`observe` and remains fail-closed on the same discontinuity.
        """

        current = self._tick(tick_ms)
        self.last_observed_tick_ms = current
        self.last_control_tick_ms = current

    def observe(self, tick_ms: int) -> PlantControlStep | None:
        current = self._tick(tick_ms)
        previous_observed = self.last_observed_tick_ms
        if previous_observed is not None:
            source_gap = current - previous_observed
            if source_gap <= 0:
                kind = "duplicate" if source_gap == 0 else "backwards"
                raise PlantControlClockFault(
                    f"MuJoCo control clock {kind}: {previous_observed}ms -> {current}ms"
                )
            if source_gap > self.max_source_gap_ms:
                raise PlantControlClockFault(
                    "MuJoCo control clock source jumped "
                    f"{source_gap}ms (limit {self.max_source_gap_ms}ms)"
                )
        self.last_observed_tick_ms = current

        previous_control = self.last_control_tick_ms
        if previous_control is None:
            self.last_control_tick_ms = current
            return PlantControlStep(current, self.period_ms / 1_000.0)

        elapsed_ms = current - previous_control
        if elapsed_ms < self.period_ms:
            return None
        if elapsed_ms > self.period_ms + self.control_tolerance_ms:
            raise PlantControlClockFault(
                "MuJoCo control interval jumped "
                f"{elapsed_ms}ms (expected {self.period_ms}ms, "
                f"tolerance {self.control_tolerance_ms}ms)"
            )
        self.last_control_tick_ms = current
        return PlantControlStep(current, elapsed_ms / 1_000.0)


class MotionDirectionGuard:
    """Latch a stop when measured travel contradicts Nav2's linear command.

    ``/cmd_vel`` is expressed in the robot base frame, while localization is in
    the map frame.  This guard rotates the commanded direction by the measured
    yaw, anchors it at the beginning of a sufficiently straight motion segment,
    and compares it with the actual map displacement.  It deliberately ignores
    in-place turns and small humanoid foot-placement transients.

    The guard does not create motion or a replacement planner.  It only rejects
    downstream forwarding after a clearly opposite displacement and therefore
    remains subordinate to Nav2's navigation authority.
    """

    def __init__(
        self,
        *,
        min_linear_mps: float = 0.10,
        max_angular_rps: float = 0.20,
        min_travel_m: float = 0.06,
        max_reverse_m: float = 0.04,
        direction_change_rad: float = math.radians(45.0),
        clear_zero_s: float = 0.50,
        clock=time.monotonic,
    ) -> None:
        positive = (
            min_linear_mps,
            max_angular_rps,
            min_travel_m,
            max_reverse_m,
            direction_change_rad,
            clear_zero_s,
        )
        if not all(math.isfinite(float(value)) and value > 0 for value in positive):
            raise ValueError("motion direction guard limits must be finite and positive")
        self.min_linear_mps = float(min_linear_mps)
        self.max_angular_rps = float(max_angular_rps)
        self.min_travel_m = float(min_travel_m)
        self.max_reverse_m = float(max_reverse_m)
        self.direction_change_rad = float(direction_change_rad)
        self.clear_zero_s = float(clear_zero_s)
        self._clock = clock
        self._pose: tuple[float, float, float] | None = None
        self._anchor: tuple[float, float] | None = None
        self._expected: tuple[float, float] | None = None
        self._zero_since: float | None = None
        self.latched = False
        self.reason: str | None = None

    def observe_pose(self, x: float, y: float, yaw: float) -> bool:
        values = tuple(float(value) for value in (x, y, yaw))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("localization pose must be finite")
        self._pose = values
        if self.latched or self._anchor is None or self._expected is None:
            return False

        dx = values[0] - self._anchor[0]
        dy = values[1] - self._anchor[1]
        travel = math.hypot(dx, dy)
        projection = dx * self._expected[0] + dy * self._expected[1]
        if travel < self.min_travel_m or projection > -self.max_reverse_m:
            return False

        self.latched = True
        self.reason = (
            "measured base motion opposes Nav2 command "
            f"(travel={travel:.3f}m, reverse_projection={-projection:.3f}m)"
        )
        self._zero_since = None
        return True

    def accept_command(self, vx: float, vy: float, wz: float) -> bool:
        values = tuple(float(value) for value in (vx, vy, wz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("velocity command must be finite")
        now = float(self._clock())
        linear = math.hypot(values[0], values[1])
        is_zero = linear < 1e-6 and abs(values[2]) < 1e-6

        if self.latched:
            if not is_zero:
                self._zero_since = None
                return False
            if self._zero_since is None:
                self._zero_since = now
                return False
            if now - self._zero_since < self.clear_zero_s:
                return False
            self.clear()
            return True

        self._zero_since = None
        if is_zero:
            # Zero velocity ends the current translational authority.  The
            # learned gait may need several balanced-policy frames to settle
            # after Nav2 cancels or completes a primitive; projecting that
            # deliberate braking motion onto the previous command direction
            # creates a false reverse-motion fault after authority has ended.
            # A later non-zero translation establishes a fresh anchor.
            self._anchor = None
            self._expected = None
            return True
        if self._pose is None:
            self._anchor = None
            self._expected = None
            return True
        if linear < 1e-6 and abs(values[2]) >= 1e-6:
            # A pure Spin owns no translational direction.  Retaining a
            # previous DriveOnHeading/lateral anchor makes ordinary humanoid
            # pelvis sway during the new action look like reverse travel.  The
            # dedicated RotationDirectionGuard already rejects wrong-way yaw
            # and persistent translation during Spin, so end the stale
            # translation transaction here.  A mixed turn with non-zero
            # linear intent deliberately keeps the existing anchor below.
            self._anchor = None
            self._expected = None
            return True
        if linear < self.min_linear_mps or abs(values[2]) > self.max_angular_rps:
            # A Nav2 controller commonly inserts a turn while following one
            # global goal.  Keep an existing straight-motion anchor through
            # that transient; clearing it here lets an opposite displacement
            # erase the only evidence that motion changed direction.  Do not
            # create a new anchor for a turn or a sub-threshold command.
            return True

        local_x = values[0] / linear
        local_y = values[1] / linear
        cos_yaw = math.cos(self._pose[2])
        sin_yaw = math.sin(self._pose[2])
        expected = (
            cos_yaw * local_x - sin_yaw * local_y,
            sin_yaw * local_x + cos_yaw * local_y,
        )
        if self._expected is not None:
            dot = max(-1.0, min(1.0, expected[0] * self._expected[0] + expected[1] * self._expected[1]))
            if math.acos(dot) <= self.direction_change_rad:
                return True
        self._anchor = (self._pose[0], self._pose[1])
        self._expected = expected
        return True

    def reanchor_at_actuator_grant(self) -> bool:
        """Discard pre-policy INIT drift at the first real motion boundary."""

        if self.latched or self._pose is None or self._expected is None:
            return False
        self._anchor = (self._pose[0], self._pose[1])
        return True

    def clear(self) -> None:
        self.latched = False
        self.reason = None
        self._anchor = None
        self._expected = None
        self._zero_since = None


class GoalProgressGuard:
    """Fail closed when a global navigation action regresses from its goal.

    This guard consumes the endpoint of Nav2's global ``Path`` and an
    independently published map pose.  It is not a planner and does not judge
    small detours: only a conservative regression beyond a configured distance
    latches the downstream SONIC stop interlock.  A stored path endpoint is not
    itself command authority: standalone Spin, DriveOnHeading, and lateral
    primitives must suspend this transaction so a stale global goal cannot
    reject their intentional motion.
    """

    def __init__(
        self,
        *,
        min_progress_m: float = 0.05,
        max_regression_m: float = 0.15,
        goal_change_m: float = 0.10,
        clear_zero_s: float = 0.50,
        clock=time.monotonic,
    ) -> None:
        limits = (min_progress_m, max_regression_m, goal_change_m, clear_zero_s)
        if not all(math.isfinite(float(value)) and value > 0 for value in limits):
            raise ValueError("goal progress guard limits must be finite and positive")
        self.min_progress_m = float(min_progress_m)
        self.max_regression_m = float(max_regression_m)
        self.goal_change_m = float(goal_change_m)
        self.clear_zero_s = float(clear_zero_s)
        self._clock = clock
        self._goal: tuple[float, float] | None = None
        self._pose: tuple[float, float] | None = None
        self._initial_distance: float | None = None
        self._best_distance: float | None = None
        self._translation_active = False
        self._zero_since: float | None = None
        self.latched = False
        self.reason: str | None = None

    @property
    def goal(self) -> tuple[float, float] | None:
        return self._goal

    def observe_goal(self, x: float, y: float) -> bool:
        goal = (float(x), float(y))
        if not all(math.isfinite(value) for value in goal):
            raise ValueError("Nav2 path goal must be finite")
        if self._goal is not None and math.dist(goal, self._goal) <= self.goal_change_m:
            return False
        self._goal = goal
        self._reset_progress()
        if self._pose is not None:
            distance = math.dist(self._pose, goal)
            self._initial_distance = distance
            self._best_distance = distance
        return True

    def observe_pose(self, x: float, y: float) -> bool:
        pose = (float(x), float(y))
        if not all(math.isfinite(value) for value in pose):
            raise ValueError("localization pose must be finite")
        self._pose = pose
        if self.latched or self._goal is None or not self._translation_active:
            return False

        distance = math.dist(pose, self._goal)
        if self._initial_distance is None or self._best_distance is None:
            self._initial_distance = distance
            self._best_distance = distance
            return False
        self._best_distance = min(self._best_distance, distance)
        improved = self._initial_distance - self._best_distance
        baseline = self._best_distance if improved >= self.min_progress_m else self._initial_distance
        regression = distance - baseline
        if regression <= self.max_regression_m:
            return False

        self.latched = True
        self.reason = (
            "measured base motion regresses from Nav2 goal "
            f"(distance={distance:.3f}m, baseline={baseline:.3f}m, regression={regression:.3f}m)"
        )
        self._zero_since = None
        return True

    def accept_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        *,
        tracking_enabled: bool = True,
    ) -> bool:
        values = tuple(float(value) for value in (vx, vy, wz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("velocity command must be finite")
        if not isinstance(tracking_enabled, bool):
            raise TypeError("tracking_enabled must be a boolean")
        if not self.latched:
            if not tracking_enabled:
                self.suspend_tracking()
                return True
            self._zero_since = None
            linear = math.hypot(values[0], values[1])
            if linear < 1e-6 and abs(values[2]) < 1e-6:
                # A completed/cancelled primitive no longer owns monotonic
                # progress toward its endpoint.  Stop-settle pelvis motion is
                # governed by the posture and finite-state guards, not by a
                # stale goal-distance transaction.
                self._translation_active = False
                self._reset_progress()
            elif linear < 1e-6 and abs(values[2]) >= 1e-6:
                # Pure Spin has no monotonic Euclidean-distance contract to a
                # global path endpoint.  Humanoid pelvis sway can temporarily
                # move away from that endpoint while yaw progresses correctly.
                # End the preceding translation transaction; the dedicated
                # rotation guard owns reverse-yaw and translation-envelope
                # safety until a new linear command establishes a fresh goal
                # progress baseline.
                self._translation_active = False
                self._reset_progress()
            elif linear >= 1e-6 and not self._translation_active:
                self._translation_active = True
                self._reset_progress()
                if self._pose is not None and self._goal is not None:
                    distance = math.dist(self._pose, self._goal)
                    self._initial_distance = distance
                    self._best_distance = distance
            return True

        now = float(self._clock())
        is_zero = all(abs(value) < 1e-6 for value in values)
        if not is_zero:
            self._zero_since = None
            return False
        if self._zero_since is None:
            self._zero_since = now
            return False
        if now - self._zero_since < self.clear_zero_s:
            return False
        self.clear()
        return True

    def suspend_tracking(self) -> None:
        """End only the current global-goal transaction, preserving its goal.

        A latched violation is deliberately not cleared here.  It still needs
        the existing sustained-zero/reset path; changing action type cannot
        erase a motion fault.
        """

        if self.latched:
            return
        self._translation_active = False
        self._zero_since = None
        self._reset_progress()

    def reanchor_at_actuator_grant(self) -> bool:
        """Start goal-progress accounting after policy authority is real."""

        if (
            self.latched
            or not self._translation_active
            or self._pose is None
            or self._goal is None
        ):
            return False
        distance = math.dist(self._pose, self._goal)
        self._initial_distance = distance
        self._best_distance = distance
        return True

    def clear(self) -> None:
        self.latched = False
        self.reason = None
        self._translation_active = False
        self._zero_since = None
        self._reset_progress()

    def _reset_progress(self) -> None:
        self._initial_distance = None
        self._best_distance = None


class TranslationHeadingGuard:
    """Fail closed when measured yaw diverges from Nav2 during translation."""

    def __init__(
        self,
        *,
        min_linear_mps: float = 0.10,
        max_heading_error_rad: float = 0.25,
        max_expected_delta_rad: float = 0.05,
        violation_s: float = 0.25,
        clear_zero_s: float = 0.50,
        clock=time.monotonic,
    ) -> None:
        limits = (
            min_linear_mps,
            max_heading_error_rad,
            max_expected_delta_rad,
            violation_s,
            clear_zero_s,
        )
        if not all(math.isfinite(float(value)) and value > 0 for value in limits):
            raise ValueError("translation heading guard limits must be finite and positive")
        self.min_linear_mps = float(min_linear_mps)
        self.max_heading_error_rad = float(max_heading_error_rad)
        self.max_expected_delta_rad = float(max_expected_delta_rad)
        self.violation_s = float(violation_s)
        self.clear_zero_s = float(clear_zero_s)
        self._clock = clock
        self._latest_yaw: float | None = None
        self._anchor_yaw: float | None = None
        self._expected_delta = 0.0
        self._last_command_at: float | None = None
        self._active = False
        self._violation_since: float | None = None
        self._zero_since: float | None = None
        self.latched = False
        self.reason: str | None = None

    def observe_pose(self, yaw: float) -> bool:
        value = float(yaw)
        if not math.isfinite(value):
            raise ValueError("translation heading pose must be finite")
        self._latest_yaw = self._wrap_angle(value)
        if self.latched or not self._active:
            return False
        if self._anchor_yaw is None:
            self._anchor_yaw = self._latest_yaw
            self._expected_delta = 0.0
            self._violation_since = None
            return False
        measured_delta = self._wrap_angle(self._latest_yaw - self._anchor_yaw)
        error = abs(self._wrap_angle(measured_delta - self._expected_delta))
        now = float(self._clock())
        if error <= self.max_heading_error_rad:
            self._violation_since = None
            return False
        if self._violation_since is None:
            self._violation_since = now
            return False
        if now - self._violation_since < self.violation_s:
            return False
        self.latched = True
        self.reason = (
            "measured translation heading diverges from Nav2 command "
            f"(error={error:.3f}rad, duration={self.violation_s:.2f}s)"
        )
        self._zero_since = None
        return True

    def accept_command(self, vx: float, vy: float, wz: float) -> bool:
        values = tuple(float(value) for value in (vx, vy, wz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("translation heading velocity command must be finite")
        now = float(self._clock())
        linear = math.hypot(values[0], values[1])
        is_zero = linear < 1e-6 and abs(values[2]) < 1e-6
        if self.latched:
            if not is_zero:
                self._zero_since = None
                return False
            if self._zero_since is None:
                self._zero_since = now
                return False
            if now - self._zero_since < self.clear_zero_s:
                return False
            self.clear()
            return True

        self._zero_since = None
        if linear >= self.min_linear_mps:
            if not self._active:
                self._active = True
                self._anchor_yaw = self._latest_yaw
                self._expected_delta = 0.0
                self._violation_since = None
            elif self._last_command_at is not None:
                dt = max(0.0, min(0.5, now - self._last_command_at))
                integrated = self._wrap_angle(
                    self._expected_delta + values[2] * dt
                )
                # The downstream CmdVelGuard intentionally constrains mixed
                # translation/rotation corrections to this same bounded band.
                # Comparing measured yaw with an unbounded integral of raw DWB
                # ``wz`` creates a false interlock even though SONIC was never
                # authorized to realize that accumulated turn.
                self._expected_delta = max(
                    -self.max_expected_delta_rad,
                    min(self.max_expected_delta_rad, integrated),
                )
            self._last_command_at = now
        else:
            self._reset_tracking()
        return True

    def clear(self) -> None:
        self.latched = False
        self.reason = None
        self._zero_since = None
        self._reset_tracking()

    def reanchor_at_actuator_grant(self) -> bool:
        """Exclude policy preparation from translation-heading evidence."""

        if self.latched or not self._active or self._latest_yaw is None:
            return False
        self._anchor_yaw = self._latest_yaw
        self._expected_delta = 0.0
        self._last_command_at = float(self._clock())
        self._violation_since = None
        return True

    def _reset_tracking(self) -> None:
        self._active = False
        self._anchor_yaw = None
        self._expected_delta = 0.0
        self._last_command_at = None
        self._violation_since = None

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class RotationDirectionGuard:
    """Fail closed when a Nav2 Spin turns backward or translates too far."""

    def __init__(
        self,
        *,
        min_angular_rps: float = 0.08,
        max_linear_mps: float = 0.05,
        max_reverse_rad: float = 0.05,
        max_translation_m: float = 0.15,
        translation_violation_s: float = 0.50,
        clear_zero_s: float = 0.50,
        clock=time.monotonic,
    ) -> None:
        limits = (
            min_angular_rps,
            max_linear_mps,
            max_reverse_rad,
            max_translation_m,
            translation_violation_s,
            clear_zero_s,
        )
        if not all(math.isfinite(float(value)) and value > 0 for value in limits):
            raise ValueError("rotation direction guard limits must be finite and positive")
        self.min_angular_rps = float(min_angular_rps)
        self.max_linear_mps = float(max_linear_mps)
        self.max_reverse_rad = float(max_reverse_rad)
        self.max_translation_m = float(max_translation_m)
        self.translation_violation_s = float(translation_violation_s)
        self.clear_zero_s = float(clear_zero_s)
        self._clock = clock
        self._pose: tuple[float, float, float] | None = None
        self._anchor: tuple[float, float, float] | None = None
        self._previous_yaw: float | None = None
        self._cumulative_yaw = 0.0
        self._sign = 0.0
        self._translation_violation_since: float | None = None
        self._zero_since: float | None = None
        self.latched = False
        self.reason: str | None = None

    def observe_pose(self, x: float, y: float, yaw: float) -> bool:
        pose = tuple(float(value) for value in (x, y, yaw))
        if not all(math.isfinite(value) for value in pose):
            raise ValueError("rotation localization pose must be finite")
        self._pose = pose
        if self.latched or self._anchor is None or not self._sign:
            return False
        # Integrate adjacent wrapped samples instead of wrapping the complete
        # anchor-to-current delta.  A valid +pi Spin necessarily crosses the
        # +pi/-pi representation boundary; wrapping the full displacement made
        # +3.49 rad look like -2.79 rad and falsely latched a reverse-spin stop.
        if self._previous_yaw is None:
            self._previous_yaw = pose[2]
        else:
            self._cumulative_yaw += self._wrap_angle(pose[2] - self._previous_yaw)
            self._previous_yaw = pose[2]
        signed_progress = self._sign * self._cumulative_yaw
        translation = math.hypot(pose[0] - self._anchor[0], pose[1] - self._anchor[1])
        if signed_progress < -self.max_reverse_rad:
            self._latch(
                "measured base yaw opposes Nav2 Spin "
                f"(reverse={-signed_progress:.3f}rad)"
            )
            return True
        if translation <= self.max_translation_m:
            self._translation_violation_since = None
        elif self._translation_violation_since is None:
            self._translation_violation_since = float(self._clock())
        elif float(self._clock()) - self._translation_violation_since >= self.translation_violation_s:
            self._latch(
                "in-place Nav2 Spin persistently exceeded translation envelope "
                f"(translation={translation:.3f}m, duration={self.translation_violation_s:.2f}s)"
            )
            return True
        return False

    def accept_command(self, vx: float, vy: float, wz: float) -> bool:
        values = tuple(float(value) for value in (vx, vy, wz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rotation velocity command must be finite")
        now = float(self._clock())
        linear = math.hypot(values[0], values[1])
        is_zero = linear < 1e-6 and abs(values[2]) < 1e-6
        if self.latched:
            if not is_zero:
                self._zero_since = None
                return False
            if self._zero_since is None:
                self._zero_since = now
                return False
            if now - self._zero_since < self.clear_zero_s:
                return False
            self.clear()
            return True

        self._zero_since = None
        pure_rotation = linear <= self.max_linear_mps and abs(values[2]) >= self.min_angular_rps
        if pure_rotation and self._pose is not None:
            sign = math.copysign(1.0, values[2])
            if self._anchor is None or sign != self._sign:
                self._anchor = self._pose
                self._previous_yaw = self._pose[2]
                self._cumulative_yaw = 0.0
                self._sign = sign
                self._translation_violation_since = None
        elif is_zero:
            self._anchor = None
            self._previous_yaw = None
            self._cumulative_yaw = 0.0
            self._sign = 0.0
            self._translation_violation_since = None
        else:
            self._anchor = None
            self._previous_yaw = None
            self._cumulative_yaw = 0.0
            self._sign = 0.0
            self._translation_violation_since = None
        return True

    def _latch(self, reason: str) -> None:
        self.latched = True
        self.reason = reason
        self._zero_since = None

    def reanchor_at_actuator_grant(self) -> bool:
        """Exclude policy preparation drift from Spin displacement evidence."""

        if self.latched or self._pose is None or not self._sign:
            return False
        self._anchor = self._pose
        self._previous_yaw = self._pose[2]
        self._cumulative_yaw = 0.0
        self._translation_violation_since = None
        return True

    def clear(self) -> None:
        self.latched = False
        self.reason = None
        self._anchor = None
        self._previous_yaw = None
        self._cumulative_yaw = 0.0
        self._sign = 0.0
        self._translation_violation_since = None
        self._zero_since = None

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class JointCommandDecision:
    targets: tuple[float, ...]
    static_hold_active: bool
    motion_handoff_active: bool
    handoff_ratio: float


@dataclass(frozen=True)
class NominalPolicyActivationDecision:
    """One immutable proposal in the policy activation ramp."""

    epoch: int
    tick_ms: int
    targets: tuple[float, ...]
    ratio: float
    complete: bool


@dataclass(frozen=True)
class NominalPolicySettleObservation:
    """Authoritative measured evidence for one post-INIT stability sample."""

    stage: str
    stable: bool
    consecutive_samples: int
    required_consecutive_samples: int
    elapsed_s: float
    max_joint_error_rad: float
    max_joint_velocity_rps: float
    max_gyro_rps: float
    roll_rad: float
    pitch_rad: float
    stage_complete: bool
    prepared: bool


class NominalPolicyActivationFault(RuntimeError):
    """Nominal INIT or measured stability evidence failed closed."""


class NominalPolicyActivation:
    """Prepare the official nominal standing pose before the first policy tick.

    NVIDIA's G1 deployment has a distinct INIT phase which moves the robot from
    its current joint pose to the policy's nominal standing pose before CONTROL
    begins.  H-IDE cannot perform that motion during process startup because its
    cold-start contract is an exact static hold, so the same transition is
    deferred until the first explicitly authorized Nav2 action.

    Proposals use authoritative MuJoCo milliseconds, never wall time.  State is
    committed only after the caller reports the exact target that successfully
    crossed DDS; a stop can therefore invalidate an in-flight proposal without
    manufacturing a completed activation.
    """

    _PHASES = frozenset(
        {
            "UNPREPARED",
            "UNPREPARED_SUPPORT_OFF",
            "PREPARING",
            "SETTLING_TETHERED",
            "WAITING_FOR_SUPPORT_RELEASE",
            "PREPARED",
        }
    )

    def __init__(
        self,
        *,
        joint_count: int = 29,
        duration_s: float = 3.0,
        release_fall_arrest: bool = True,
        max_settle_joint_error_rad: float = 0.10,
        max_settle_joint_velocity_rps: float = 0.20,
        max_settle_gyro_rps: float = 0.15,
        max_settle_roll_pitch_rad: float = 0.15,
        settle_consecutive_samples: int = 10,
        settle_timeout_s: float = 3.0,
    ) -> None:
        if (
            isinstance(joint_count, bool)
            or not isinstance(joint_count, int)
            or joint_count <= 0
        ):
            raise ValueError("activation joint_count must be a positive integer")
        if isinstance(duration_s, bool):
            raise ValueError("activation duration must be exactly 3s")
        duration = float(duration_s)
        if not math.isfinite(duration) or duration != 3.0:
            raise ValueError("activation duration must be exactly 3s")
        duration_ms = round(duration * 1_000.0)
        if duration_ms <= 0 or not math.isclose(
            duration_ms / 1_000.0,
            duration,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("activation duration must resolve to whole milliseconds")
        self.joint_count = joint_count
        self.duration_s = duration
        self.duration_ms = int(duration_ms)
        if not isinstance(release_fall_arrest, bool):
            raise ValueError("release_fall_arrest must be boolean")
        self.release_fall_arrest = release_fall_arrest
        limits = {
            "max_settle_joint_error_rad": max_settle_joint_error_rad,
            "max_settle_joint_velocity_rps": max_settle_joint_velocity_rps,
            "max_settle_gyro_rps": max_settle_gyro_rps,
            "max_settle_roll_pitch_rad": max_settle_roll_pitch_rad,
            "settle_timeout_s": settle_timeout_s,
        }
        normalized_limits: dict[str, float] = {}
        for label, value in limits.items():
            if isinstance(value, bool):
                raise ValueError(f"{label} must be finite and positive")
            normalized = float(value)
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{label} must be finite and positive")
            normalized_limits[label] = normalized
        if (
            isinstance(settle_consecutive_samples, bool)
            or not isinstance(settle_consecutive_samples, int)
            or settle_consecutive_samples <= 0
        ):
            raise ValueError("settle_consecutive_samples must be a positive integer")
        self.max_settle_joint_error_rad = normalized_limits[
            "max_settle_joint_error_rad"
        ]
        self.max_settle_joint_velocity_rps = normalized_limits[
            "max_settle_joint_velocity_rps"
        ]
        self.max_settle_gyro_rps = normalized_limits["max_settle_gyro_rps"]
        self.max_settle_roll_pitch_rad = normalized_limits[
            "max_settle_roll_pitch_rad"
        ]
        self.settle_consecutive_samples = settle_consecutive_samples
        self.settle_timeout_s = normalized_limits["settle_timeout_s"]
        self.settle_timeout_ms = round(self.settle_timeout_s * 1_000.0)
        self.settle_sample_period_ms = 20
        self._phase = "UNPREPARED"
        self._epoch = 0
        self._start_targets: tuple[float, ...] | None = None
        self._start_tick_ms: int | None = None
        self._ratio = 0.0
        self._settle_start_tick_ms: int | None = None
        self._settle_consecutive_samples = 0
        self._last_settle_observation: NominalPolicySettleObservation | None = None
        self._last_settle_tick_ms: int | None = None
        self._support_release_tick_ms: int | None = None
        self._support_release_epoch: int | None = None
        self._support_reengaged_safety_stop_epoch: int | None = None
        self._policy_takeover_tick_ms: int | None = None
        self._activation_path: str | None = None
        self._ready_targets: tuple[float, ...] | None = None

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def ratio(self) -> float:
        return self._ratio

    @property
    def preparing(self) -> bool:
        return self._phase == "PREPARING"

    @property
    def settling(self) -> bool:
        return self._phase == "SETTLING_TETHERED"

    @property
    def waiting_for_support_release(self) -> bool:
        return self._phase == "WAITING_FOR_SUPPORT_RELEASE"

    @property
    def prepared(self) -> bool:
        return self._phase == "PREPARED"

    @property
    def activation_path(self) -> str | None:
        return self._activation_path

    @property
    def ready_targets(self) -> tuple[float, ...] | None:
        """Exact actuator boundary used to seed the first policy reference."""

        return self._ready_targets

    @property
    def last_settle_observation(self) -> NominalPolicySettleObservation | None:
        return self._last_settle_observation

    @property
    def support_release_confirmed(self) -> bool:
        return self._support_release_epoch is not None

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "schema_version": "h-ide.sonic-policy-activation/v4",
            "source": "NVlabs/GR00T-WholeBodyControl::G1Deploy::InitControl",
            "init_interpolation": "measured_current_to_nominal_each_tick",
            "clock": "authoritative_mujoco_lowstate_tick_ms",
            "duration_s": self.duration_s,
            "activation_epoch": self._epoch,
            "activation_path": self._activation_path,
            "support_release": {
                "mode": (
                    "authority_epoch_release"
                    if self.release_fall_arrest
                    else "retained_vertical_attitude_fall_arrest"
                ),
                "one_way": False,
                "confirmed_epoch": self._support_release_epoch,
                "reengaged_safety_stop_epoch": (
                    self._support_reengaged_safety_stop_epoch
                ),
                "fault_recovery": (
                    "reengage_then_require_newer_activation_epoch"
                    if self.release_fall_arrest
                    else "fall_arrest_already_retained"
                ),
                "cancel_recovery": "resume_policy_from_exact_frozen_boundary",
            },
            "settle_gate": {
                "max_joint_error_rad": self.max_settle_joint_error_rad,
                "max_joint_velocity_rps": self.max_settle_joint_velocity_rps,
                "max_gyro_rps": self.max_settle_gyro_rps,
                "max_abs_roll_pitch_rad": self.max_settle_roll_pitch_rad,
                "required_consecutive_samples": self.settle_consecutive_samples,
                "sample_period_ms": self.settle_sample_period_ms,
                "timeout_s": self.settle_timeout_s,
                "stages": ["tethered"],
            },
            "policy_takeover": (
                "first_successful_policy_dds_then_exact_epoch_support_release"
                if self.release_fall_arrest
                else "first_successful_policy_dds_with_fall_arrest_retained"
            ),
            "policy_takeover_tick_ms": self._policy_takeover_tick_ms,
        }

    def begin(self, targets: Sequence[float], *, tick_ms: int) -> bool:
        """Begin once per process, preserving an already-running transition."""

        if self._phase not in {"UNPREPARED", "UNPREPARED_SUPPORT_OFF"}:
            return False
        support_off = self.support_release_confirmed
        boundary = self._validate_targets(targets, "activation boundary")
        start_tick = self._validate_tick(tick_ms)
        self._epoch += 1
        self._phase = "WAITING_FOR_SUPPORT_RELEASE" if support_off else "PREPARING"
        self._activation_path = (
            "support_off_reinit"
            if support_off
            else (
                "tethered_release"
                if self.release_fall_arrest
                else "tethered_retained"
            )
        )
        self._start_targets = None if support_off else boundary
        self._start_tick_ms = None if support_off else start_tick
        self._ratio = 1.0 if support_off else 0.0
        self._ready_targets = boundary if support_off else None
        self._settle_start_tick_ms = None
        self._settle_consecutive_samples = 0
        self._last_settle_observation = None
        self._last_settle_tick_ms = None
        self._policy_takeover_tick_ms = None
        if not support_off:
            self._support_release_tick_ms = None
        return True

    def propose(
        self,
        nominal: Sequence[float],
        *,
        measured: Sequence[float],
        tick_ms: int,
    ) -> NominalPolicyActivationDecision:
        if not self.preparing:
            raise RuntimeError("nominal policy activation is not preparing")
        assert self._start_tick_ms is not None
        current_tick = self._validate_tick(tick_ms)
        if current_tick < self._start_tick_ms:
            raise ValueError("activation MuJoCo tick moved backwards")
        nominal_targets = self._validate_targets(nominal, "nominal targets")
        measured_targets = self._validate_targets(measured, "measured targets")
        elapsed_ms = current_tick - self._start_tick_ms
        ratio = min(1.0, elapsed_ms / self.duration_ms)
        targets = tuple(
            current * (1.0 - ratio) + target * ratio
            for current, target in zip(
                measured_targets, nominal_targets, strict=True
            )
        )
        return NominalPolicyActivationDecision(
            epoch=self._epoch,
            tick_ms=current_tick,
            targets=targets,
            ratio=ratio,
            complete=ratio >= 1.0,
        )

    def is_current(self, decision: NominalPolicyActivationDecision) -> bool:
        return bool(
            isinstance(decision, NominalPolicyActivationDecision)
            and self.preparing
            and decision.epoch == self._epoch
        )

    def commit(
        self,
        decision: NominalPolicyActivationDecision,
        published_targets: Sequence[float],
    ) -> bool:
        """Commit one successful DDS fact; return True only on final commit."""

        if not self.is_current(decision):
            return False
        actual = self._validate_targets(published_targets, "published targets")
        if actual != decision.targets:
            return False
        if decision.ratio + 1e-12 < self._ratio:
            raise RuntimeError("activation ratio moved backwards")
        self._ratio = decision.ratio
        if decision.complete:
            self._phase = "SETTLING_TETHERED"
            self._start_targets = None
            self._start_tick_ms = None
            self._ratio = 1.0
            self._ready_targets = actual
            self._settle_start_tick_ms = decision.tick_ms
            self._settle_consecutive_samples = 0
            self._last_settle_observation = None
            self._last_settle_tick_ms = None
            return True
        return False

    def observe_settling(
        self,
        *,
        nominal: Sequence[float],
        measured: Sequence[float],
        velocities: Sequence[float],
        gyroscope: Sequence[float],
        roll_rad: float,
        pitch_rad: float,
        tick_ms: int,
    ) -> NominalPolicySettleObservation:
        if not self.settling:
            raise RuntimeError("nominal policy activation is not settling")
        assert self._settle_start_tick_ms is not None
        current_tick = self._validate_tick(tick_ms)
        if current_tick < self._settle_start_tick_ms:
            raise ValueError("activation settling MuJoCo tick moved backwards")
        elapsed_ms = current_tick - self._settle_start_tick_ms
        if elapsed_ms > self.settle_timeout_ms:
            raise NominalPolicyActivationFault(
                f"nominal policy activation settling timeout in {self._phase}"
            )
        nominal_targets = self._validate_targets(nominal, "nominal targets")
        measured_targets = self._validate_targets(measured, "measured targets")
        velocity_targets = self._validate_targets(
            velocities, "measured velocities"
        )
        gyro = tuple(float(value) for value in gyroscope)
        if len(gyro) != 3 or not all(math.isfinite(value) for value in gyro):
            raise ValueError("measured gyroscope must contain three finite values")
        roll = float(roll_rad)
        pitch = float(pitch_rad)
        if not all(math.isfinite(value) for value in (roll, pitch)):
            raise ValueError("measured roll/pitch must be finite")
        previous_tick = self._last_settle_tick_ms
        if previous_tick is not None:
            sample_gap_ms = current_tick - previous_tick
            if sample_gap_ms <= 0:
                raise NominalPolicyActivationFault(
                    "nominal policy activation requires a fresh settling sample"
                )
            if sample_gap_ms != self.settle_sample_period_ms:
                self._settle_consecutive_samples = 0
        self._last_settle_tick_ms = current_tick
        max_joint_error = max(
            abs(current - target)
            for current, target in zip(
                measured_targets, nominal_targets, strict=True
            )
        )
        max_joint_velocity = max(abs(value) for value in velocity_targets)
        max_gyro = max(abs(value) for value in gyro)
        stable = bool(
            max_joint_error <= self.max_settle_joint_error_rad
            and max_joint_velocity <= self.max_settle_joint_velocity_rps
            and max_gyro <= self.max_settle_gyro_rps
            and max(abs(roll), abs(pitch)) <= self.max_settle_roll_pitch_rad
        )
        self._settle_consecutive_samples = (
            self._settle_consecutive_samples + 1 if stable else 0
        )
        stage = "tethered"
        stage_complete = (
            self._settle_consecutive_samples >= self.settle_consecutive_samples
        )
        if stage_complete:
            self._phase = "WAITING_FOR_SUPPORT_RELEASE"
            self._settle_start_tick_ms = None
        observation = NominalPolicySettleObservation(
            stage=stage,
            stable=stable,
            consecutive_samples=self._settle_consecutive_samples,
            required_consecutive_samples=self.settle_consecutive_samples,
            elapsed_s=elapsed_ms / 1_000.0,
            max_joint_error_rad=max_joint_error,
            max_joint_velocity_rps=max_joint_velocity,
            max_gyro_rps=max_gyro,
            roll_rad=roll,
            pitch_rad=pitch,
            stage_complete=stage_complete,
            prepared=self.prepared,
        )
        self._last_settle_observation = observation
        return observation

    def confirm_support_release(self, *, tick_ms: int) -> bool:
        """Complete takeover only after policy authority and Plant proof.

        For the initial activation, the release epoch must equal this
        activation.  A cancellation that lost the one-way race may later
        resume from the exact frozen policy boundary; in that recovery path
        the already-confirmed release epoch must strictly precede this one.
        """

        if not self.waiting_for_support_release:
            return False
        current_tick = self._validate_tick(tick_ms)
        if (
            self._last_settle_tick_ms is not None
            and current_tick < self._last_settle_tick_ms
        ):
            raise ValueError("support release tick predates tethered evidence")
        release_epoch = self._support_release_epoch
        if release_epoch is None:
            raise NominalPolicyActivationFault(
                "policy takeover lacks authoritative Plant support-release proof"
            )
        if self._activation_path == "tethered_release":
            if release_epoch != self._epoch:
                raise NominalPolicyActivationFault(
                    "initial Plant support release does not match activation epoch"
                )
        elif self._activation_path == "support_off_reinit":
            if release_epoch >= self._epoch:
                raise NominalPolicyActivationFault(
                    "support-off recovery requires an earlier release epoch"
                )
        else:
            raise NominalPolicyActivationFault(
                "policy activation path is unavailable at support release"
            )
        self._support_release_tick_ms = current_tick
        self._policy_takeover_tick_ms = current_tick
        self._phase = "PREPARED"
        return True

    def confirm_supported_takeover(self, *, tick_ms: int) -> bool:
        """Complete CONTROL while retaining non-propulsive simulation support.

        ``startup_support_s: 0`` is the explicit simulation profile which
        keeps only vertical and roll/pitch fall arrest for the run.  That
        support has no world-X/Y force or yaw torque in locomotion, so it
        cannot manufacture translation or turning.  Requiring a successful
        first policy DDS grant still preserves the official WAIT -> CONTROL
        boundary without dropping a controller which has not yet demonstrated
        unsupported balance.
        """

        if not self.waiting_for_support_release:
            return False
        if self._activation_path != "tethered_retained":
            raise NominalPolicyActivationFault(
                "retained-support takeover requires its explicit activation path"
            )
        if self.release_fall_arrest or self._support_release_epoch is not None:
            raise NominalPolicyActivationFault(
                "retained-support takeover conflicts with Plant release evidence"
            )
        current_tick = self._validate_tick(tick_ms)
        if (
            self._last_settle_tick_ms is not None
            and current_tick < self._last_settle_tick_ms
        ):
            raise ValueError("policy takeover tick predates tethered evidence")
        self._policy_takeover_tick_ms = current_tick
        self._phase = "PREPARED"
        return True

    def record_support_release(self, activation_epoch: int) -> bool:
        """Persist one activation-bound Plant release proof."""

        if (
            isinstance(activation_epoch, bool)
            or not isinstance(activation_epoch, int)
            or activation_epoch <= 0
        ):
            raise ValueError("support release epoch must be a positive integer")
        previous = self._support_release_epoch
        if previous is not None and previous != activation_epoch:
            raise NominalPolicyActivationFault(
                "Plant fall arrest cannot be released by two activation epochs"
            )
        self._support_release_epoch = activation_epoch
        if self._phase == "UNPREPARED":
            self._phase = "UNPREPARED_SUPPORT_OFF"
        return previous is None

    def record_support_reengagement(self, safety_stop_epoch: int) -> bool:
        """Invalidate support-off authority after recoverable fault containment.

        Plant re-engages only its non-propulsive simulation fall arrest. The
        next explicit motion must nevertheless repeat measured INIT/settle and
        obtain a strictly newer release proof; a previously PREPARED policy may
        not silently run under newly restored support.
        """

        if (
            isinstance(safety_stop_epoch, bool)
            or not isinstance(safety_stop_epoch, int)
            or safety_stop_epoch <= 0
        ):
            raise ValueError("support re-engagement epoch must be positive")
        if not self.release_fall_arrest:
            return False
        previous_reengagement = self._support_reengaged_safety_stop_epoch
        if previous_reengagement is not None:
            if safety_stop_epoch < previous_reengagement:
                raise NominalPolicyActivationFault(
                    "Plant support re-engagement epoch moved backwards"
                )
            if safety_stop_epoch == previous_reengagement:
                return False
        self._support_reengaged_safety_stop_epoch = safety_stop_epoch
        self._support_release_epoch = None
        self._support_release_tick_ms = None
        self._phase = "UNPREPARED"
        self._activation_path = None
        self._start_targets = None
        self._start_tick_ms = None
        self._ratio = 0.0
        self._settle_start_tick_ms = None
        self._settle_consecutive_samples = 0
        self._last_settle_observation = None
        self._last_settle_tick_ms = None
        self._ready_targets = None
        self._policy_takeover_tick_ms = None
        return True

    def abort(self) -> bool:
        """Invalidate unfinished proposals while preserving process-level prep."""

        if self._phase in {
            "UNPREPARED",
            "UNPREPARED_SUPPORT_OFF",
            "PREPARED",
        }:
            return False
        if (
            self._phase == "WAITING_FOR_SUPPORT_RELEASE"
            and self._activation_path == "tethered_release"
            and self._support_release_epoch == self._epoch
        ):
            # The only writer of this proof starts after a successful first
            # policy DDS grant.  If a stop wins after the Plant consumed the
            # one-way release but before the owner tick consumed its HTTP
            # result, INIT is nevertheless complete and must not be repeated
            # unsupported on the next task.
            self._phase = "PREPARED"
            return True
        self._epoch += 1
        self._phase = (
            "UNPREPARED_SUPPORT_OFF"
            if self.support_release_confirmed
            else "UNPREPARED"
        )
        self._start_targets = None
        self._start_tick_ms = None
        self._ratio = 0.0
        self._settle_start_tick_ms = None
        self._settle_consecutive_samples = 0
        self._last_settle_observation = None
        self._last_settle_tick_ms = None
        self._ready_targets = None
        self._policy_takeover_tick_ms = None
        if not self.support_release_confirmed:
            self._support_release_tick_ms = None
        return True

    def _validate_targets(
        self,
        values: Sequence[float],
        label: str,
    ) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if len(result) != self.joint_count:
            raise ValueError(f"{label} must contain {self.joint_count} values")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{label} must be finite")
        return result

    @staticmethod
    def _validate_tick(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("activation tick_ms must be a non-negative integer")
        return value


class StaticJointHandoff:
    """Select either an exact static hold or an authority-owned policy target.

    SONIC's learned ``IDLE`` clip is a posture transition, not a zero-motion
    actuator command.  The simulation bridge therefore latches measured joints
    whenever Nav2 is idle.  A policy target is emitted only after the matching
    reference owns actuator authority, and is then emitted exactly: post-policy
    blending would create a target the learned controller neither predicted nor
    records in its action history.

    ``warmup_s`` and ``blend_s`` remain validated constructor inputs solely for
    configuration compatibility; binary authority deliberately does not use
    either duration.
    """

    def __init__(
        self,
        *,
        joint_count: int = 29,
        warmup_s: float = 0.35,
        blend_s: float = 0.40,
        clock=time.monotonic,
    ) -> None:
        if isinstance(joint_count, bool) or int(joint_count) <= 0:
            raise ValueError("joint_count must be a positive integer")
        if not math.isfinite(float(warmup_s)) or float(warmup_s) < 0:
            raise ValueError("warmup_s must be finite and non-negative")
        if not math.isfinite(float(blend_s)) or float(blend_s) <= 0:
            raise ValueError("blend_s must be finite and positive")
        self.joint_count = int(joint_count)
        self.warmup_s = float(warmup_s)
        self.blend_s = float(blend_s)
        self._clock = clock
        self._hold: tuple[float, ...] | None = None
        self._active = False

    def decide(
        self,
        *,
        motion_active: bool,
        reference_ready: bool = True,
        measured: Sequence[float],
        policy: Sequence[float],
        now: float | None = None,
    ) -> JointCommandDecision:
        if not isinstance(motion_active, bool) or not isinstance(reference_ready, bool):
            raise ValueError("handoff motion/reference readiness flags must be booleans")
        measured_values = self._validate(measured, "measured")
        policy_values = self._validate(policy, "policy")
        current = float(self._clock() if now is None else now)
        if not math.isfinite(current):
            raise ValueError("handoff time must be finite")

        # An asynchronous policy target is not motion authority.  A newly
        # admitted Nav2 intent must remain at an exact zero handoff ratio until
        # the planner evidence gate has accepted a generation-matched reference
        # whose predicted root displacement agrees with that intent.
        if not motion_active or not reference_ready:
            if self._active or self._hold is None:
                self._hold = measured_values
            self._active = False
            return JointCommandDecision(self._hold, True, False, 0.0)

        self._active = True
        return JointCommandDecision(policy_values, False, False, 1.0)

    def reset(self) -> None:
        self._hold = None
        self._active = False

    def freeze(self, targets: Sequence[float]) -> None:
        """Freeze an already-published actuator target without re-anchoring.

        A physical stop must be target-continuous.  Replacing the last
        published target with the current measured position removes the small
        PD error that supplies gravity-balancing torque and can destabilize a
        standing humanoid.  Callers therefore pass the exact last target that
        crossed the actuator boundary.
        """

        self._hold = self._validate(targets, "frozen")
        self._active = False

    def _validate(self, values: Sequence[float], label: str) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if len(result) != self.joint_count:
            raise ValueError(f"{label} joint vector must contain {self.joint_count} values")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{label} joint vector must be finite")
        return result


class ReplanIntentGate:
    """Latch Nav2 intent so an asynchronous SONIC planner is not flooded.

    Nav2 publishes continuously and makes tiny command corrections.  SONIC's
    planner, in contrast, produces motion clips asynchronously; requesting a
    new clip for every 50 Hz sample lets stale results overtake newer intent.
    Start/stop transitions remain immediate, while in-motion updates are
    admitted only when both meaningful and sufficiently spaced.
    """

    def __init__(
        self,
        *,
        min_update_interval_s: float = 0.35,
        min_speed_delta_mps: float = 0.05,
        min_angle_delta_rad: float = 0.30,
        min_rotation_start_angle_rad: float = 0.05,
        clock=time.monotonic,
    ) -> None:
        limits = (
            min_update_interval_s,
            min_speed_delta_mps,
            min_angle_delta_rad,
            min_rotation_start_angle_rad,
        )
        if not all(math.isfinite(float(value)) and value > 0 for value in limits):
            raise ValueError("replan gate limits must be finite and positive")
        self.min_update_interval_s = float(min_update_interval_s)
        self.min_speed_delta_mps = float(min_speed_delta_mps)
        self.min_angle_delta_rad = float(min_angle_delta_rad)
        self.min_rotation_start_angle_rad = float(min_rotation_start_angle_rad)
        self._clock = clock
        self._last: MovementIntent | None = None
        self._last_at = 0.0

    @property
    def last(self) -> MovementIntent | None:
        return self._last

    def reset(self) -> None:
        """Forget the prior task at an authoritative actuator stop boundary."""

        self._last = None
        self._last_at = 0.0

    def admit(self, intent: MovementIntent, *, now: float | None = None) -> bool:
        current = self._clock() if now is None else float(now)
        if not math.isfinite(current):
            raise ValueError("replan gate time must be finite")
        # Nav2's Spin behavior ramps angular velocity from zero.  Admitting the
        # first non-zero sample used to ask the asynchronous SONIC planner for a
        # 1--4 mrad pure-rotation clip.  The locked planner's measured noise
        # floor is 0.004016 rad and its smallest verified directional request is
        # +/-0.05 rad, so such a clip cannot provide meaningful same-sign yaw
        # evidence.  Keep executing the already-proven learned IDLE balance
        # policy until the command-only facing integral reaches that verified
        # request size; only then start the required-reference deadline.
        if self._defer_small_rotation_start(intent):
            return False
        if self._last is None:
            return self._remember(intent, current)

        # Safety transitions never wait for the planner debounce window.
        if (
            intent.mode != self._last.mode
            or intent.has_movement != self._last.has_movement
            or intent.stale != self._last.stale
        ):
            return self._remember(intent, current)
        if current - self._last_at < self.min_update_interval_s:
            return False

        changed = (
            abs(intent.speed_mps - self._last.speed_mps) >= self.min_speed_delta_mps
            or self._angular_distance(intent.movement_angle_rad, self._last.movement_angle_rad)
            >= self.min_angle_delta_rad
            or self._angular_distance(intent.facing_angle_rad, self._last.facing_angle_rad)
            >= self.min_angle_delta_rad
        )
        return self._remember(intent, current) if changed else False

    def _defer_small_rotation_start(self, intent: MovementIntent) -> bool:
        if intent.mode == "IDLE" or intent.has_movement:
            return False
        if self._last is not None and self._last.mode != "IDLE":
            return False
        # Every ordinary zero->motion transaction asks SONIC for a fresh
        # measured reference.  That request deliberately rebases the vendor
        # planner context to a heading-neutral local frame.  The preceding
        # learned-IDLE reference may retain a non-zero facing so the humanoid
        # can balance at the completed pose, but it is not the origin of the
        # next planner request.  Comparing a new Spin against that old facing
        # admitted milliradian ramps immediately and could invert a subsequent
        # turn after a completed action.
        origin = 0.0
        return (
            self._angular_distance(intent.facing_angle_rad, origin) + 1e-12
            < self.min_rotation_start_angle_rad
        )

    def _remember(self, intent: MovementIntent, current: float) -> bool:
        self._last = intent
        self._last_at = current
        return True

    @staticmethod
    def _angular_distance(left: float, right: float) -> float:
        return abs((left - right + math.pi) % (2 * math.pi) - math.pi)


class BodyStabilityGuard:
    """Latch motion when measured roll/pitch shows an incipient fall.

    This guard consumes localization only.  It cannot create an actuator
    command; the ROS bridge uses its latch solely to reject further Nav2
    velocity and engage the simulation emergency damping profile.
    """

    def __init__(
        self,
        *,
        max_tilt_rad: float = 0.30,
        violation_s: float = 0.05,
        recovery_s: float = 0.75,
        clock=time.monotonic,
    ) -> None:
        values = (max_tilt_rad, violation_s, recovery_s)
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ValueError("body stability limits must be finite and positive")
        if float(max_tilt_rad) >= math.pi / 2:
            raise ValueError("body stability tilt limit must be below pi/2")
        self.max_tilt_rad = float(max_tilt_rad)
        self.violation_s = float(violation_s)
        self.recovery_s = float(recovery_s)
        self._clock = clock
        self.latched = False
        self.reason: str | None = None
        self._violation_since: float | None = None
        self._stable_since: float | None = None
        self._zero_since: float | None = None

    def observe(self, roll: float, pitch: float, *, now: float | None = None) -> bool:
        values = (float(roll), float(pitch))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("body orientation must be finite")
        current = float(self._clock() if now is None else now)
        tilt = max(abs(values[0]), abs(values[1]))
        tripped = False
        if tilt > self.max_tilt_rad:
            self._stable_since = None
            if self._violation_since is None:
                self._violation_since = current
            if not self.latched and current - self._violation_since >= self.violation_s:
                self.latched = True
                self.reason = (
                    "measured G1 body tilt exceeded safety envelope "
                    f"({tilt:.3f} rad > {self.max_tilt_rad:.3f} rad)"
                )
                tripped = True
        else:
            self._violation_since = None
            if self.latched:
                if self._stable_since is None:
                    self._stable_since = current
        return tripped

    def admit_command(
        self,
        vx: float,
        vy: float,
        wz: float,
        *,
        now: float | None = None,
    ) -> bool:
        values = (float(vx), float(vy), float(wz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("body stability command must be finite")
        if not self.latched:
            return True
        current = float(self._clock() if now is None else now)
        nonzero = any(abs(value) > 1e-6 for value in values)
        if nonzero:
            self._zero_since = None
            return False
        if self._zero_since is None:
            self._zero_since = current
        if (
            self._stable_since is not None
            and current - max(self._stable_since, self._zero_since) >= self.recovery_s
        ):
            self.clear()
        return True

    def clear(self) -> None:
        self.latched = False
        self.reason = None
        self._violation_since = None
        self._stable_since = None
        self._zero_since = None


class CmdVelGuard:
    """Adapt, bound and expire Nav2 commands before they reach SONIC.

    The class contains no ROS or MuJoCo imports, making it independently
    testable and keeping the actuator boundary auditable.
    """

    _OFFICIAL_DIRECTION_BIN_RAD = math.pi / 4.0
    _OFFICIAL_TRANSLATION_DEADBAND_MPS = 0.01
    _OFFICIAL_SLOW_WALK_SPEED_BY_ABS_BIN = {
        0: 0.30,
        1: 0.30,
        2: 0.35,
        3: 0.25,
        4: 0.20,
    }
    # The default SONIC navigation model has a repeatable lateral torso-yaw
    # bias in untethered Office V2 MuJoCo.  A gain sweep against the pinned
    # cached model plus the r65 sequential live rejection showed that the
    # original gain=2/cap=.15 correction was under-damped: it recovered an
    # initial right-strafe yaw error, then overshot through the fixed heading.
    # Gain=.75/cap=.08 keeps both mirrored cached-model strafes below 5 degrees
    # while retaining margin inside the product's 10-degree endpoint contract.
    # It never alters Nav2 cmd_vel or the immutable facing goal.
    _LATERAL_HEADING_HOLD_GAIN = 0.75
    _LATERAL_HEADING_HOLD_MAX_RAD = 0.08

    def __init__(
        self,
        *,
        max_linear_mps: float = 0.40,
        min_walk_command_mps: float = 0.20,
        max_angular_rps: float = 0.5,
        max_facing_lead_rad: float = 0.05,
        max_translation_facing_offset_rad: float = 0.05,
        max_integration_dt_s: float = 0.024,
        timeout_s: float = 0.25,
        smoothing_alpha: float = 0.35,
        clock=time.monotonic,
    ) -> None:
        if (
            max_linear_mps <= 0
            or max_angular_rps <= 0
            or max_facing_lead_rad <= 0
            or max_translation_facing_offset_rad <= 0
            or max_integration_dt_s <= 0
            or timeout_s <= 0
        ):
            raise ValueError("guard limits must be positive")
        if max_translation_facing_offset_rad > max_facing_lead_rad:
            raise ValueError(
                "max_translation_facing_offset_rad must not exceed max_facing_lead_rad"
            )
        if not 0 < min_walk_command_mps < max_linear_mps:
            raise ValueError("min_walk_command_mps must be in (0, max_linear_mps)")
        if not 0 < smoothing_alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.max_linear_mps = float(max_linear_mps)
        self.min_walk_command_mps = float(min_walk_command_mps)
        self.max_angular_rps = float(max_angular_rps)
        self.max_facing_lead_rad = float(max_facing_lead_rad)
        self.max_translation_facing_offset_rad = float(
            max_translation_facing_offset_rad
        )
        self.max_integration_dt_s = float(max_integration_dt_s)
        self.timeout_s = float(timeout_s)
        self.smoothing_alpha = float(smoothing_alpha)
        self._clock = clock
        self._last = CmdVel(0.0, 0.0, 0.0, 0.0, source="boot")
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0
        self._facing = 0.0
        # This is the command-space goal.  It is advanced only by admitted
        # angular velocity and authoritative simulation dt.  The emitted
        # facing is a separately lead-limited view sent to SONIC; measured yaw
        # must never be fed back into this integral.
        self._command_facing = 0.0
        self._measured_heading: float | None = None
        self._heading_anchor: float | None = None
        self._action_facing_origin = 0.0

    @property
    def last_received_at(self) -> float | None:
        """Monotonic timestamp of the first accepted command, if any."""
        return None if self._last.source == "boot" else self._last.received_at

    @property
    def measured_heading(self) -> float | None:
        return self._measured_heading

    @property
    def command_facing(self) -> float:
        """Return the admitted command integral before measured-lead limiting."""

        return self._command_facing

    @property
    def command_zero(self) -> bool:
        """Whether both the admitted Nav2 target and filtered command are zero.

        This is an observational safety predicate. It never advances the
        watchdog or smoothing state, so a health probe cannot create motion or
        turn a pending stop into an acknowledgement.
        """

        return (
            not self._is_nonzero(self._last)
            and all(abs(value) <= 1e-9 for value in (self._vx, self._vy, self._wz))
        )

    @property
    def target_zero(self) -> bool:
        """Whether the latest admitted Nav2 target is exactly stopped.

        A controlled policy stop is owned as soon as the zero target crosses
        the command boundary.  The filtered values intentionally remain from
        the preceding 50 Hz policy tick until :meth:`intent` consumes that
        target.  Health must be able to prove the zero target during this
        bounded transition without misrepresenting the filtered state as
        already stationary; :attr:`command_zero` remains the stronger final
        hold predicate.
        """

        return not self._is_nonzero(self._last)

    def observe_heading(self, yaw: float) -> None:
        """Anchor SONIC's global facing frame to measured ``map→base`` yaw.

        Nav2 publishes angular velocity, not an absolute facing target.  Purely
        integrating that velocity lets the learned planner's facing frame drift
        without bound whenever the humanoid cannot immediately realize a turn.
        The measured heading remains observation-only: it cannot create motion,
        and a new facing target is still advanced solely by Nav2's ``wz``.
        """

        value = float(yaw)
        if not math.isfinite(value):
            raise ValueError("measured heading must be finite")
        self._measured_heading = self._wrap_angle(value)

    def accept(
        self,
        vx: float,
        vy: float,
        wz: float,
        *,
        source: str = "nav2",
        received_at: float | None = None,
    ) -> CmdVel:
        values = (float(vx), float(vy), float(wz))
        if not all(math.isfinite(value) for value in values):
            raise ValueError("cmd_vel values must be finite")
        norm = math.hypot(values[0], values[1])
        # Match NVIDIA's pinned ROS2 input adapter instead of interpreting a
        # continuous Nav2 magnitude as a learned-policy speed.  SONIC's
        # SLOW_WALK interface quantizes every active planar command to one of
        # eight directions and assigns a direction-specific reviewed speed:
        # forward/forward-diagonal=.30, lateral=.35,
        # backward-diagonal=.25 and backward=.20 m/s.  The upstream adapter
        # uses 0.01 m/s solely as its active/idle threshold.  Rejecting all
        # Nav2 ramp samples below .20 was an H-IDE invention which could stop a
        # valid controller before the official adapter would start walking.
        if norm <= self._OFFICIAL_TRANSLATION_DEADBAND_MPS:
            values = (0.0, 0.0, values[2])
            norm = 0.0
        else:
            binned_angle, official_speed = self._official_slow_walk_command(
                values[0], values[1]
            )
            reviewed_speed = min(official_speed, self.max_linear_mps)
            values = (
                reviewed_speed * math.cos(binned_angle),
                reviewed_speed * math.sin(binned_angle),
                values[2],
            )
            norm = reviewed_speed
        accepted_at = self._clock() if received_at is None else float(received_at)
        if not math.isfinite(accepted_at):
            raise ValueError("cmd_vel receive time must be finite")
        target = CmdVel(
            values[0],
            values[1],
            max(-self.max_angular_rps, min(self.max_angular_rps, values[2])),
            accepted_at,
            source=source,
        )
        starting = self._is_nonzero(target) and not self._is_nonzero(self._last)
        if starting:
            # An ordinary SONIC reference request calls
            # ``load_measured_reference()``, whose planner root quaternion is
            # explicitly heading-neutral.  Rebase every new Nav2 action to that
            # same local frame while retaining measured yaw only as the
            # observation anchor used by curve lead limiting.  Carrying the
            # preceding action's learned-IDLE facing into this new request made
            # a right turn following a left turn ask the planner to turn left.
            self._heading_anchor = self._measured_heading
            self._action_facing_origin = 0.0
            self._facing = 0.0
            self._command_facing = 0.0
        elif not self._is_nonzero(target):
            self._freeze_facing_action()
        self._last = target
        return target

    def stop(self, *, source: str = "watchdog") -> CmdVel:
        return self.accept(0.0, 0.0, 0.0, source=source)

    def stop_at_measured_heading(
        self,
        *,
        source: str = "gait_adapter_controlled_stop",
        facing_angle_rad: float | None = None,
    ) -> CmdVel:
        """Stop motion at the exact live planner-reference facing.

        The name is retained for the internal controlled-stop API, but the
        transaction now uses NVIDIA's continuous-history IDLE reference.  The
        driver supplies the yaw at the current reference cursor because H-IDE
        may have rebased that frame during a Nav2 action. Resetting to zero
        undoes the completed turn; reusing a stale global command angle can add
        the turn twice. ``accept`` then freezes that exact planner boundary.
        """

        if facing_angle_rad is not None:
            value = float(facing_angle_rad)
            if not math.isfinite(value):
                raise ValueError("controlled-stop reference facing must be finite")
            self._facing = self._wrap_angle(value)
            self._command_facing = self._facing
            self._action_facing_origin = self._facing
            self._heading_anchor = None
        return self.accept(0.0, 0.0, 0.0, source=source)

    def intent(self, now: float | None = None, dt: float = 0.02) -> MovementIntent:
        current = self._clock() if now is None else float(now)
        if not math.isfinite(current):
            raise ValueError("cmd_vel watchdog time must be finite")
        if isinstance(dt, bool):
            raise ValueError("authoritative cmd_vel integration dt must be finite")
        integration_dt = float(dt)
        if (
            not math.isfinite(integration_dt)
            or integration_dt <= 0.0
            or integration_dt > self.max_integration_dt_s
        ):
            raise ValueError(
                "authoritative cmd_vel integration dt must be in "
                f"(0, {self.max_integration_dt_s}]s"
            )
        if current < self._last.received_at:
            raise ValueError("cmd_vel watchdog time moved backwards")
        stale = current - self._last.received_at > self.timeout_s
        target = (
            CmdVel(0.0, 0.0, 0.0, self._last.received_at, source="watchdog")
            if stale
            else self._last
        )
        if stale:
            self._last = target
        # A zero from Nav2 is a safety/terminal transition, not another noisy
        # velocity sample.  Applying the normal low-pass tail here kept SONIC
        # in SLOW_WALK after Nav2 had already completed or cancelled Spin.
        alpha = 1.0 if stale or not self._is_nonzero(target) else self.smoothing_alpha
        target_speed = math.hypot(target.vx, target.vy)
        if target_speed >= self.min_walk_command_mps:
            # Filtering a legal 0.20m/s command from zero creates an illegal
            # sub-floor tail (0.07, 0.1155, ...). Translation is therefore
            # admitted atomically at the model's discrete floor.
            self._vx = target.vx
            self._vy = target.vy
        else:
            self._vx += (target.vx - self._vx) * alpha
            self._vy += (target.vy - self._vy) * alpha
        self._wz += (target.wz - self._wz) * alpha
        speed = math.hypot(self._vx, self._vy)
        moving = speed >= self.min_walk_command_mps and not stale
        rotating = abs(self._wz) >= 0.05 and not stale
        active = moving or rotating
        if active:
            # Only an admitted angular command and authoritative simulation dt
            # advance this goal. Measured yaw is used below solely to cap the
            # currently emitted lead; it is never fed back into the integral.
            self._command_facing = self._wrap_angle(
                self._command_facing + self._wz * integration_dt
            )
            if not moving:
                # Match NVIDIA's pinned ROS2 adapter: an angular-only Nav2
                # command advances ``planner_facing_angle_`` directly and the
                # resulting absolute facing vector is written to
                # ``MovementState``.  Capping a pure Spin to measured yaw +
                # 0.05 rad reduced every planner request to a short prefix;
                # once that prefix ended the robot could not make further
                # progress even though Nav2 continued to command a turn.
                # Rotation remains bounded by Nav2's signed Spin goal,
                # command timeout, direction/progress guards and the action
                # deadline.  Measured lead limiting is retained below for a
                # translating curve, where it protects path tracking.
                self._facing = self._command_facing
            elif abs(self._wz) <= 1e-9:
                # DriveOnHeading and a pure lateral step require an absolute
                # facing target for the complete action.  Following measured
                # yaw here creates positive feedback: a normal humanoid torso
                # oscillation moves the target with the body, so the policy is
                # never asked to recover the original heading.  NVIDIA's
                # planner contract keeps movement and facing directions
                # independent; a zero angular command therefore holds the
                # command-space facing exactly.
                self._facing = self._command_facing
            else:
                # During an explicitly commanded turn, bound how far the
                # emitted planner target may lead measured progress.  This is
                # a rate/lag guard for the authorized angular command, not a
                # mechanism for measured drift to redefine a straight goal.
                measured_facing = self._current_measured_facing()
                lead_limit = (
                    self.max_translation_facing_offset_rad
                    if moving
                    else self.max_facing_lead_rad
                )
                lead = self._wrap_angle(self._command_facing - measured_facing)
                lead = max(-lead_limit, min(lead_limit, lead))
                self._facing = self._wrap_angle(measured_facing + lead)
        else:
            # Explicit zero, watchdog expiry and revocation freeze both the
            # emitted target and command integral. The next action gets a
            # fresh measured anchor at this exact canonical boundary.
            self._freeze_facing_action()
        # Active sub-floor translation was rejected at admission, so speed is
        # either zero or inside this runtime's reviewed SLOW_WALK envelope.
        # Nav2 Twist linear velocity is expressed in ``base_link``.  SONIC's
        # movement angle is expressed in the controller's heading frame (the
        # same convention used by its joystick adapter), so a base-forward
        # command after a turn must follow the accumulated facing angle rather
        # than remain pinned to the original world +X direction.
        angle = self._facing + math.atan2(self._vy, self._vx) if moving else self._facing
        return MovementIntent(
            # SONIC represents an in-place facing change as a slow-walk plan
            # with a zero movement vector.  This is still initiated solely by
            # Nav2's angular cmd_vel and never bypasses the navigation owner.
            mode="SLOW_WALK" if moving or rotating else "IDLE",
            speed_mps=min(speed, self.max_linear_mps),
            movement_angle_rad=angle,
            facing_angle_rad=self._facing,
            has_movement=moving,
            stale=stale,
            source=target.source,
        )

    def lateral_heading_correction(
        self,
        intent: MovementIntent,
    ) -> LateralHeadingCorrection | None:
        """Return decoder feedback only for an explicitly pure strafe.

        NVIDIA exposes movement and facing as independent planner inputs.  A
        pure lateral command therefore has a fixed facing goal even while its
        movement direction is +/- pi/2.  This method compares measured yaw to
        the immutable action-start heading; measured drift never changes the
        command integral or authorizes rotation.  Every other motion type is
        left untouched.
        """

        pure_lateral = bool(
            intent.mode == "SLOW_WALK"
            and intent.has_movement
            and abs(self._last.vx) <= 1e-6
            and abs(self._last.vy) > 1e-6
            and abs(self._last.wz) <= 1e-9
            and abs(
                abs(
                    self._wrap_angle(
                        intent.movement_angle_rad - intent.facing_angle_rad
                    )
                )
                - math.pi / 2.0
            )
            <= 1e-6
        )
        if not pure_lateral:
            return None
        if self._measured_heading is None or self._heading_anchor is None:
            return LateralHeadingCorrection(
                heading_error_rad=0.0,
                decoder_delta_heading_rad=0.0,
            )
        measured_action_facing = self._current_measured_facing()
        error = self._wrap_angle(
            measured_action_facing - intent.facing_angle_rad
        )
        correction = max(
            -self._LATERAL_HEADING_HOLD_MAX_RAD,
            min(
                self._LATERAL_HEADING_HOLD_MAX_RAD,
                -self._LATERAL_HEADING_HOLD_GAIN * error,
            ),
        )
        return LateralHeadingCorrection(
            heading_error_rad=error,
            decoder_delta_heading_rad=correction,
        )

    @classmethod
    def _official_slow_walk_command(
        cls,
        vx: float,
        vy: float,
    ) -> tuple[float, float]:
        """Return NVIDIA's exact 8-way direction bin and SLOW_WALK speed."""

        angle = cls._wrap_angle(math.atan2(float(vy), float(vx)))
        scaled = angle / cls._OFFICIAL_DIRECTION_BIN_RAD
        bin_index = (
            math.floor(scaled + 0.5)
            if scaled >= 0.0
            else math.ceil(scaled - 0.5)
        )
        bin_index = max(-4, min(4, int(bin_index)))
        binned_angle = bin_index * cls._OFFICIAL_DIRECTION_BIN_RAD
        speed = cls._OFFICIAL_SLOW_WALK_SPEED_BY_ABS_BIN[abs(bin_index)]
        return binned_angle, speed

    @staticmethod
    def _is_nonzero(command: CmdVel) -> bool:
        return any(abs(value) > 1e-6 for value in (command.vx, command.vy, command.wz))

    def _freeze_facing_action(self) -> None:
        self._command_facing = self._facing
        self._heading_anchor = None
        self._action_facing_origin = self._facing

    def _current_measured_facing(self) -> float:
        """Return measured yaw expressed in the current SONIC action frame."""

        if self._measured_heading is None or self._heading_anchor is None:
            return self._facing
        return self._wrap_angle(
            self._action_facing_origin
            + self._wrap_angle(self._measured_heading - self._heading_anchor)
        )

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class JointGuardResult:
    targets: tuple[float, ...]
    constrained_joints: int
    max_step_rad: float
    # Arm-local immutable indices (0..13) and the complete per-joint running
    # saturation durations at this exact actuator decision.  The driver maps
    # indices to G1 global joints 15..28 when it records acceptance evidence.
    constrained_joint_indices: tuple[int, ...]
    continuous_saturation_s: tuple[float, ...]


class NavigationArmGuard:
    """Bound SONIC arm targets and preserve the exact executed history.

    SONIC is a whole-body policy, so even a Nav2 walking command can produce
    targets for all fourteen arm joints.  This guard rejects non-finite policy
    targets, projects finite targets into the authoritative physical joint
    ranges, and applies the configured per-joint slew bound.  The caller must
    feed the returned
    actuator target—not the raw candidate—through ``record_executed_target``
    so recurrent action history remains exactly equal to the DDS fact.

    Projection closes reference/intent seams without weakening the physical
    velocity limit.  Saturation duration remains authoritative evidence.  A
    caller may configure a fail-closed duration budget, but the production
    locomotion profile deliberately leaves it unset: no upstream SONIC or
    Unitree contract defines such a deadline, and the projected actuator fact
    is already bounded and fed back exactly to the recurrent policy.

    It deliberately runs *after* SONIC and before ``rt/lowcmd``.  The lower
    body and waist remain untouched, and this guard cannot create motion on its
    own.
    """

    def __init__(
        self,
        nominal: Sequence[float],
        *,
        max_deviation_rad: Sequence[float] | None = None,
        lower_limits_rad: Sequence[float] | None = None,
        upper_limits_rad: Sequence[float] | None = None,
        max_velocity_rps: Sequence[float],
        max_continuous_saturation_s: float | None = None,
    ) -> None:
        self.nominal = tuple(float(value) for value in nominal)
        self.max_velocity_rps = tuple(float(value) for value in max_velocity_rps)
        if not self.nominal:
            raise ValueError("arm guard requires at least one joint")
        if len(self.nominal) != len(self.max_velocity_rps):
            raise ValueError("arm guard vectors must have matching lengths")
        if not all(math.isfinite(value) for value in self.nominal):
            raise ValueError("arm nominal targets must be finite")
        explicit_limits = lower_limits_rad is not None or upper_limits_rad is not None
        if explicit_limits:
            if max_deviation_rad is not None:
                raise ValueError(
                    "arm guard accepts either deviations or explicit limits, not both"
                )
            if lower_limits_rad is None or upper_limits_rad is None:
                raise ValueError(
                    "arm guard explicit lower and upper limits must be provided together"
                )
            lower = tuple(float(value) for value in lower_limits_rad)
            upper = tuple(float(value) for value in upper_limits_rad)
            if len(lower) != len(self.nominal) or len(upper) != len(self.nominal):
                raise ValueError("arm guard vectors must have matching lengths")
            if not all(
                math.isfinite(low)
                and math.isfinite(high)
                and low < high
                for low, high in zip(lower, upper, strict=True)
            ):
                raise ValueError("arm hard limits must be finite and ordered")
            if not all(
                low <= center <= high
                for center, low, high in zip(
                    self.nominal,
                    lower,
                    upper,
                    strict=True,
                )
            ):
                raise ValueError("arm nominal targets must lie within hard limits")
            self.lower_limits_rad = lower
            self.upper_limits_rad = upper
            # Retain the legacy diagnostic attribute without using it for
            # enforcement.  Explicit limits are commonly asymmetric.
            self.max_deviation_rad = tuple(
                max(center - low, high - center)
                for center, low, high in zip(
                    self.nominal,
                    lower,
                    upper,
                    strict=True,
                )
            )
        else:
            if max_deviation_rad is None:
                raise ValueError(
                    "arm guard requires deviations or explicit hard limits"
                )
            deviations = tuple(float(value) for value in max_deviation_rad)
            if len(deviations) != len(self.nominal):
                raise ValueError("arm guard vectors must have matching lengths")
            if not all(
                math.isfinite(value) and value > 0 for value in deviations
            ):
                raise ValueError("arm deviations must be finite and positive")
            self.max_deviation_rad = deviations
            self.lower_limits_rad = tuple(
                center - radius
                for center, radius in zip(
                    self.nominal,
                    deviations,
                    strict=True,
                )
            )
            self.upper_limits_rad = tuple(
                center + radius
                for center, radius in zip(
                    self.nominal,
                    deviations,
                    strict=True,
                )
            )
        if not all(math.isfinite(value) and value > 0 for value in self.max_velocity_rps):
            raise ValueError("arm velocity limits must be finite and positive")
        saturation_budget: float | None
        if max_continuous_saturation_s is None:
            saturation_budget = None
        else:
            if isinstance(max_continuous_saturation_s, bool):
                raise ValueError(
                    "arm saturation budget must be finite and positive"
                )
            saturation_budget = float(max_continuous_saturation_s)
            if not math.isfinite(saturation_budget) or saturation_budget <= 0.0:
                raise ValueError(
                    "arm saturation budget must be finite and positive"
                )
        self.max_continuous_saturation_s = saturation_budget
        self._last: tuple[float, ...] | None = None
        self._continuous_saturation_s = tuple(0.0 for _ in self.nominal)

    def apply(
        self,
        targets: Sequence[float],
        *,
        dt: float,
        measured: Sequence[float] | None = None,
    ) -> JointGuardResult:
        if len(targets) != len(self.nominal):
            raise ValueError("arm target vector has an unexpected length")
        if measured is not None and len(measured) != len(self.nominal):
            raise ValueError("measured arm vector has an unexpected length")
        if isinstance(dt, bool) or not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("arm guard dt must be finite and positive")

        raw_targets = tuple(float(value) for value in targets)
        if not all(math.isfinite(value) for value in raw_targets):
            raise ValueError("arm policy targets must be finite")
        if self._last is None:
            initial = measured if measured is not None else self.nominal
            anchor = tuple(float(value) for value in initial)
            if not all(math.isfinite(value) for value in anchor):
                raise ValueError("arm verifier anchor must be finite")
        else:
            anchor = self._last

        projected_targets: list[float] = []
        saturation_durations: list[float] = []
        constrained_joint_indices: list[int] = []
        constrained_joints = 0
        max_step = 0.0
        for index, raw in enumerate(raw_targets):
            bounded = min(
                self.upper_limits_rad[index],
                max(self.lower_limits_rad[index], raw),
            )
            allowed_step = self.max_velocity_rps[index] * float(dt)
            step = bounded - anchor[index]
            hard_limit_constrained = bounded != raw
            slew_constrained = abs(step) > allowed_step + 1e-9
            if slew_constrained:
                projected = anchor[index] + math.copysign(
                    allowed_step,
                    step,
                )
            else:
                projected = bounded
            if hard_limit_constrained or slew_constrained:
                saturated_for = (
                    self._continuous_saturation_s[index] + float(dt)
                )
                if (
                    self.max_continuous_saturation_s is not None
                    and saturated_for
                    > self.max_continuous_saturation_s + 1e-12
                ):
                    raise ValueError(
                        "arm policy target "
                        f"{index}={raw:.17g} persistently exceeds the per-tick "
                        f"slew limit: anchor={anchor[index]:.17g}, "
                        f"delta={step:.17g}, allowed={allowed_step:.17g}, "
                        f"saturated_for={saturated_for:.17g}s, "
                        f"budget={self.max_continuous_saturation_s:.17g}s"
                    )
                constrained_joints += 1
                constrained_joint_indices.append(index)
            else:
                saturated_for = 0.0
            projected_targets.append(projected)
            saturation_durations.append(saturated_for)
            max_step = max(max_step, abs(projected - anchor[index]))

        self._last = tuple(projected_targets)
        self._continuous_saturation_s = tuple(saturation_durations)
        return JointGuardResult(
            self._last,
            constrained_joints,
            max_step,
            tuple(constrained_joint_indices),
            self._continuous_saturation_s,
        )

    def reset(self) -> None:
        self._last = None
        self._continuous_saturation_s = tuple(0.0 for _ in self.nominal)

    def anchor(self, targets: Sequence[float]) -> None:
        """Seed verification from the exact actuator-side arm targets.

        A planner handoff freezes the complete last-published 29-DoF command.
        The arm limiter must therefore start from those fourteen actuator
        targets as well, not from lagging measured angles.  An anchor already
        outside the nominal envelope may only stay fixed or recover toward it,
        within the configured per-tick velocity limit.
        """

        values = tuple(float(value) for value in targets)
        if len(values) != len(self.nominal):
            raise ValueError("arm anchor vector has an unexpected length")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("arm anchor vector must be finite")
        self._last = values
        self._continuous_saturation_s = tuple(0.0 for _ in self.nominal)
