"""Fail-closed, ROS-independent Nav2 to SONIC gait intent adapter.

The adapter deliberately contains no planner and no simulator integration.  It
accepts normalized, generation-scoped Nav2 ``Twist`` commands plus measured
base observations and emits a small versioned contract for the SONIC driver.
All translational and rotational authority therefore continues to originate in
Nav2.  Head targets are hard-coded to zero for the navigation-only profile.

The class is event driven so recorded traces can be replayed without ROS or
MuJoCo.  A typical integration performs the following steps::

    adapter.ingest_observation(observation)
    adapter.arm(generation=observation.generation)
    adapter.ingest_twist(nav2_twist)
    intent = adapter.tick()

An explicit Nav2 zero, cancel, stale input, generation change, or safety fault
always produces a zero-motion intent.  A normal stop does not become ``IDLE``
until fresh base observations either acknowledge strict instantaneous rest or
prove a bounded learned-IDLE balance window.  The latter is needed because an
untethered humanoid actively balances even when its commanded motion is zero.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import time
from typing import Any, Callable


class GaitState(str, Enum):
    """Lifecycle of one generation-scoped Nav2/SONIC handoff."""

    DISARMED = "DISARMED"
    IDLE = "IDLE"
    STARTING = "STARTING"
    WALKING = "WALKING"
    TURNING = "TURNING"
    STOPPING = "STOPPING"
    FAULT = "FAULT"


@dataclass(frozen=True, slots=True)
class Nav2Twist:
    """A normalized Nav2 velocity sample in ``base_link`` coordinates.

    ``timestamp_s`` must use the same monotonic clock domain as ``now`` passed
    to :class:`GaitAdapter`.  ``cancelled`` represents the action-level Nav2
    cancel event; its velocity fields are ignored and it always fails closed.
    """

    generation: int
    timestamp_s: float
    vx: float
    vy: float
    wz: float
    cancelled: bool = False
    source: str = "nav2"


@dataclass(frozen=True, slots=True)
class BaseObservation:
    """Measured base pose and body-frame velocity for one plant generation."""

    generation: int
    timestamp_s: float
    x: float
    y: float
    yaw: float
    roll: float
    pitch: float
    vx_body: float
    vy_body: float
    wz: float


@dataclass(frozen=True, slots=True)
class SonicIntent:
    """Versioned, wire-friendly navigation-only command for SONIC.

    ``facing_delta_rad`` is a bounded short-horizon value derived solely from
    the current Nav2 yaw rate.  It is not a locally planned heading.  A driver
    may map this contract to its model-specific representation, but must obey
    ``motion_enabled`` and ``stop_requested`` first.
    """

    sequence: int
    generation: int | None
    generated_at_s: float
    state: str
    mode: str
    velocity_x_mps: float
    velocity_y_mps: float
    linear_speed_mps: float
    movement_angle_rad: float
    yaw_rate_rps: float
    facing_delta_rad: float
    motion_enabled: bool
    stop_requested: bool
    stop_acknowledged: bool
    head_yaw_rad: float
    head_pitch_rad: float
    source: str
    reason: str | None
    schema_version: str = field(default="h-ide.sonic-intent/v1", init=False)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable contract without enum objects."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class GaitAdapterConfig:
    """Safety and handoff limits for :class:`GaitAdapter`."""

    command_timeout_s: float = 0.35
    observation_timeout_s: float = 0.30
    future_timestamp_tolerance_s: float = 0.05
    max_linear_mps: float = 0.35
    max_angular_rps: float = 0.60
    linear_deadband_mps: float = 0.03
    angular_deadband_rps: float = 0.04
    start_linear_ack_mps: float = 0.02
    start_angular_ack_rps: float = 0.03
    start_timeout_s: float = 1.50
    stop_linear_ack_mps: float = 0.025
    stop_angular_ack_rps: float = 0.04
    stop_ack_s: float = 0.15
    stop_ack_samples: int = 2
    stop_timeout_s: float = 2.0
    # The official learned IDLE remains an active whole-body policy. Permit a
    # normal Nav2 zero to complete when its instantaneous balance corrections
    # exceed the rigid-body thresholds above, but only while a continuous pose
    # window proves bounded oscillation and strict near-zero net motion. This
    # path is never used for cancel, safe-stop, or fault acknowledgement.
    balanced_idle_stop_timeout_s: float = 4.0
    balanced_idle_stop_window_s: float = 0.45
    balanced_idle_stop_max_sample_gap_s: float = 0.20
    balanced_idle_stop_linear_excursion_m: float = 0.015
    balanced_idle_stop_yaw_excursion_rad: float = 0.035
    balanced_idle_stop_max_linear_mps: float = 0.05
    balanced_idle_stop_max_angular_rps: float = 0.20
    balanced_idle_stop_samples: int = 5
    # Humanoid stand-to-walk may briefly shift its base opposite the commanded
    # direction.  The allowance is intentionally bounded in both distance and
    # time, and is measured only after the first correlated DDS authority fact.
    reverse_translation_m: float = 0.08
    reverse_min_travel_m: float = 0.08
    startup_reverse_grace_s: float = 1.0
    startup_progress_deadline_s: float = 2.0
    startup_min_progress_m: float = 0.01
    reverse_rotation_rad: float = 0.05
    max_tilt_rad: float = 0.55
    # Simulation experiment switch.  When disabled, an excessive but finite
    # roll/pitch observation remains visible to monitoring while the learned
    # whole-body policy keeps actuator authority.  Invalid/non-finite state,
    # stale input, cancellation and all other fault classes remain unchanged.
    trip_on_excessive_tilt: bool = True
    # Reversible simulation-only experiment.  When false, automatic adapter
    # observations remain diagnostic and the most recent valid SONIC command
    # retains authority.  Explicit Nav2 cancel and correlated safe-stop do not
    # use ``_trip_fault`` and therefore remain authoritative stop operations.
    trip_on_automatic_fault: bool = True
    direction_restart_rad: float = math.radians(100.0)
    turn_preview_s: float = 0.20
    max_facing_delta_rad: float = 0.15

    def __post_init__(self) -> None:
        positive = (
            self.command_timeout_s,
            self.observation_timeout_s,
            self.future_timestamp_tolerance_s,
            self.max_linear_mps,
            self.max_angular_rps,
            self.linear_deadband_mps,
            self.angular_deadband_rps,
            self.start_linear_ack_mps,
            self.start_angular_ack_rps,
            self.start_timeout_s,
            self.stop_linear_ack_mps,
            self.stop_angular_ack_rps,
            self.stop_ack_s,
            self.stop_timeout_s,
            self.balanced_idle_stop_timeout_s,
            self.balanced_idle_stop_window_s,
            self.balanced_idle_stop_max_sample_gap_s,
            self.balanced_idle_stop_linear_excursion_m,
            self.balanced_idle_stop_yaw_excursion_rad,
            self.balanced_idle_stop_max_linear_mps,
            self.balanced_idle_stop_max_angular_rps,
            self.reverse_translation_m,
            self.reverse_min_travel_m,
            self.startup_reverse_grace_s,
            self.startup_progress_deadline_s,
            self.startup_min_progress_m,
            self.reverse_rotation_rad,
            self.max_tilt_rad,
            self.direction_restart_rad,
            self.turn_preview_s,
            self.max_facing_delta_rad,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in positive):
            raise ValueError("gait adapter limits must be finite and positive")
        if not isinstance(self.trip_on_excessive_tilt, bool):
            raise ValueError("trip_on_excessive_tilt must be a boolean")
        if not isinstance(self.trip_on_automatic_fault, bool):
            raise ValueError("trip_on_automatic_fault must be a boolean")
        if (
            isinstance(self.stop_ack_samples, bool)
            or not isinstance(self.stop_ack_samples, int)
            or self.stop_ack_samples < 2
        ):
            raise ValueError("stop_ack_samples must be an integer of at least two")
        if (
            isinstance(self.balanced_idle_stop_samples, bool)
            or not isinstance(self.balanced_idle_stop_samples, int)
            or self.balanced_idle_stop_samples < 3
        ):
            raise ValueError(
                "balanced_idle_stop_samples must be an integer of at least three"
            )
        if self.linear_deadband_mps >= self.max_linear_mps:
            raise ValueError("linear deadband must be below the command limit")
        if self.angular_deadband_rps >= self.max_angular_rps:
            raise ValueError("angular deadband must be below the command limit")
        if self.start_linear_ack_mps >= self.max_linear_mps:
            raise ValueError("linear start acknowledgement must be below the command limit")
        if self.start_angular_ack_rps >= self.max_angular_rps:
            raise ValueError("angular start acknowledgement must be below the command limit")
        if self.stop_linear_ack_mps >= self.max_linear_mps:
            raise ValueError("linear stop acknowledgement must be below the command limit")
        if self.stop_angular_ack_rps >= self.max_angular_rps:
            raise ValueError("angular stop acknowledgement must be below the command limit")
        if not (
            self.stop_linear_ack_mps
            < self.balanced_idle_stop_max_linear_mps
            < self.max_linear_mps
        ):
            raise ValueError(
                "balanced IDLE linear speed must be between the strict stop and command limits"
            )
        if not (
            self.stop_angular_ack_rps
            < self.balanced_idle_stop_max_angular_rps
            < self.max_angular_rps
        ):
            raise ValueError(
                "balanced IDLE angular speed must be between the strict stop and command limits"
            )
        if self.balanced_idle_stop_timeout_s < self.stop_timeout_s:
            raise ValueError(
                "balanced IDLE stop timeout must not be shorter than emergency stop timeout"
            )
        if self.balanced_idle_stop_window_s >= self.balanced_idle_stop_timeout_s:
            raise ValueError(
                "balanced IDLE stop window must be shorter than its stop timeout"
            )
        if (
            self.balanced_idle_stop_max_sample_gap_s
            >= self.balanced_idle_stop_window_s
        ):
            raise ValueError(
                "balanced IDLE sample gap must be shorter than its stop window"
            )
        if (
            (self.balanced_idle_stop_samples - 1)
            * self.balanced_idle_stop_max_sample_gap_s
            < self.balanced_idle_stop_window_s
        ):
            raise ValueError(
                "balanced IDLE sample count and gap cannot span its stop window"
            )
        if self.direction_restart_rad > math.pi:
            raise ValueError("direction restart angle must not exceed pi")
        if self.startup_reverse_grace_s >= self.startup_progress_deadline_s:
            raise ValueError(
                "startup reverse grace must be shorter than the progress deadline"
            )


class GaitAdapter:
    """Generation-aware Nav2 to SONIC safety state machine.

    The adapter is intentionally single-threaded.  A ROS integration should
    serialize events before calling it, then publish the returned
    :class:`SonicIntent`.  Fault or post-motion recovery always requires a
    strictly newer plant generation.  The sole same-generation exception is a
    stale observation while still IDLE in a generation that has never granted
    motion: a fresh stationary observation plus an explicit :meth:`arm` may
    restore that startup/reconnect synchronization loss.
    """

    def __init__(
        self,
        config: GaitAdapterConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or GaitAdapterConfig()
        self._clock = clock
        self.state = GaitState.DISARMED
        self.generation: int | None = None
        self.reason: str | None = "awaiting plant generation and arm"
        self.automatic_fault_count = 0
        self.last_automatic_fault_reason: str | None = None

        self._sequence = 0
        self._last_now: float | None = None
        self._observation: BaseObservation | None = None
        self._command: Nav2Twist | None = None
        self._last_observation_timestamp: float | None = None
        self._last_command_timestamp: float | None = None
        self._requires_new_generation = False
        self._motion_started_in_generation = False

        self._start_at: float | None = None
        self._motion_authority_granted_at: float | None = None
        self._reference_wait_started_at: float | None = None
        self._reference_authority_source_sequence: int | None = None
        self._reference_authority_epoch: int | None = None
        self._reference_authority_dds_sequence: int | None = None
        self._active_kind: GaitState | None = None
        self._active_direction: tuple[float, float] | None = None
        self._active_turn_sign = 0.0
        self._motion_anchor: tuple[float, float, float] | None = None
        self._expected_map_direction: tuple[float, float] | None = None
        self._previous_yaw: float | None = None
        self._cumulative_yaw = 0.0
        self._startup_positive_progress_seen = False

        self._stop_started_at: float | None = None
        self._stop_terminal = GaitState.IDLE
        self._stationary_since: float | None = None
        self._stationary_samples = 0
        self._last_stop_observation_timestamp: float | None = None
        self._stop_acknowledged = False
        self._balanced_idle_stop_window: deque[
            tuple[float, float, float, float]
        ] = deque()
        self._balanced_idle_stop_last_raw_yaw: float | None = None
        self._balanced_idle_stop_last_unwrapped_yaw: float | None = None
        self._safe_stop_requested = False

    def advance_generation(self, generation: int, *, now: float | None = None) -> SonicIntent:
        """Invalidate all buffered motion and select a newer plant generation."""

        timestamp = self._resolve_now(now)
        if not self._valid_generation(generation):
            self._trip_fault("invalid plant generation", timestamp)
            return self._emit(timestamp)
        if self.generation is not None and generation < self.generation:
            return self._emit(timestamp, reason="old generation ignored")
        if generation == self.generation:
            return self._emit(timestamp)
        self._select_generation(generation)
        return self._emit(timestamp)

    def ingest_observation(
        self,
        observation: BaseObservation,
        *,
        now: float | None = None,
    ) -> SonicIntent:
        """Accept a measured pose/velocity sample and evaluate safety guards."""

        timestamp = self._resolve_now(now)
        if not self._valid_observation(observation, timestamp):
            self._trip_fault("invalid or non-finite base observation", timestamp)
            return self._emit(timestamp)

        if self.generation is None or observation.generation > self.generation:
            self._select_generation(observation.generation)
        elif observation.generation < self.generation:
            return self._emit(timestamp, reason="old generation observation ignored")

        if (
            self._last_observation_timestamp is not None
            and observation.timestamp_s <= self._last_observation_timestamp
        ):
            return self._emit(timestamp, reason="out-of-order base observation ignored")

        self._observation = observation
        self._last_observation_timestamp = observation.timestamp_s

        if (
            self.config.trip_on_excessive_tilt
            and (
                abs(observation.roll) > self.config.max_tilt_rad
                or abs(observation.pitch) > self.config.max_tilt_rad
            )
        ):
            self._trip_fault(
                "base tilt exceeds navigation gait envelope "
                f"(roll={observation.roll:.3f}, pitch={observation.pitch:.3f})",
                timestamp,
            )
            return self._emit(timestamp)

        if self.state in (GaitState.STARTING, GaitState.WALKING, GaitState.TURNING):
            if self.state is not GaitState.STARTING or self._start_at is not None:
                self._check_reverse_motion(observation, timestamp)
                if self.state is GaitState.STARTING:
                    self._check_start_ack(observation)

        if self.state in (GaitState.STOPPING, GaitState.FAULT):
            self._observe_stop_ack(observation, timestamp)

        self._evaluate(timestamp)
        return self._emit(timestamp)

    def arm(self, generation: int, *, now: float | None = None) -> SonicIntent:
        """Arm a selected generation after a fresh stationary measurement."""

        timestamp = self._resolve_now(now)
        if not self._valid_generation(generation):
            self._trip_fault("invalid arm generation", timestamp)
            return self._emit(timestamp)
        if generation != self.generation:
            return self._emit(timestamp, reason="arm rejected for inactive generation")
        if self._requires_new_generation or self.state is GaitState.FAULT:
            return self._emit(timestamp, reason="arm rejected until a newer generation is observed")
        if self.state is not GaitState.DISARMED:
            return self._emit(timestamp)
        if not self._observation_is_fresh(timestamp):
            return self._emit(timestamp, reason="arm rejected without a fresh base observation")
        assert self._observation is not None
        if not self._is_stationary(self._observation):
            return self._emit(timestamp, reason="arm rejected while measured base is moving")

        self.state = GaitState.IDLE
        self.reason = None
        self._command = None
        self._last_command_timestamp = None
        self._stop_acknowledged = True
        return self._emit(timestamp)

    def ingest_twist(self, command: Nav2Twist, *, now: float | None = None) -> SonicIntent:
        """Accept a Nav2 command; no other source is permitted to create motion."""

        timestamp = self._resolve_now(now)
        if not self._valid_command(command, timestamp):
            self._trip_fault("invalid, non-finite, or non-Nav2 velocity command", timestamp)
            return self._emit(timestamp)

        if self.generation is None or command.generation > self.generation:
            # A future command proves the old plant lifecycle is invalid, but
            # cannot arm a new lifecycle without a matching base observation.
            self._select_generation(command.generation)
            return self._emit(timestamp, reason="new generation command rejected until observation and arm")
        if command.generation < self.generation:
            return self._emit(timestamp, reason="old generation command ignored")
        if (
            self._last_command_timestamp is not None
            and command.timestamp_s <= self._last_command_timestamp
        ):
            return self._emit(timestamp, reason="out-of-order Nav2 command ignored")

        if command.cancelled:
            self._last_command_timestamp = command.timestamp_s
            self._command = command
            self._requires_new_generation = True
            if self.state is GaitState.DISARMED:
                self.reason = "Nav2 action cancelled; newer generation required"
            elif self.state is not GaitState.FAULT:
                self._request_stop(
                    "Nav2 action cancelled",
                    terminal=GaitState.DISARMED,
                    now=timestamp,
                )
            return self._emit(timestamp)

        if self.state in (GaitState.DISARMED, GaitState.FAULT):
            # Never buffer a pre-arm non-zero command: arming must not start a
            # robot from a command that arrived before the operator's action.
            return self._emit(timestamp, reason="Nav2 command rejected while gait adapter is not armed")

        self._last_command_timestamp = command.timestamp_s
        self._command = self._bounded(command)
        self._evaluate(timestamp)
        return self._emit(timestamp)

    def cancel(
        self,
        generation: int,
        timestamp_s: float,
        *,
        now: float | None = None,
    ) -> SonicIntent:
        """Convenience wrapper for a generation-scoped Nav2 cancel event."""

        return self.ingest_twist(
            Nav2Twist(
                generation=generation,
                timestamp_s=timestamp_s,
                vx=0.0,
                vy=0.0,
                wz=0.0,
                cancelled=True,
            ),
            now=now,
        )

    def request_safe_stop(
        self,
        generation: int,
        *,
        now: float | None = None,
    ) -> SonicIntent:
        """Begin one recoverable stop from fresh post-request measurements.

        Unlike a Nav2 cancel, this transition deliberately enters ``STOPPING``
        even when the gait is already IDLE or DISARMED.  The external
        safe-stop contract therefore cannot turn a pre-request zero-velocity
        sample into proof of a new physical stop.  Replays in the same
        generation are idempotent and do not reset evidence already collected
        for the active correlated stop.

        A gait ``FAULT`` is terminal and remains on the separate motion-fault
        path; this method never converts it into a recoverable normal stop.
        """

        timestamp = self._resolve_now(now)
        if not self._valid_generation(generation):
            self._trip_fault("invalid safe-stop generation", timestamp)
            return self._emit(timestamp)
        if generation != self.generation:
            return self._emit(
                timestamp,
                reason="safe-stop rejected for inactive generation",
            )
        if self.state is GaitState.FAULT:
            return self._emit(timestamp)
        if self._safe_stop_requested:
            return self._emit(timestamp)

        self._safe_stop_requested = True
        self._requires_new_generation = True
        self._command = None
        self._last_command_timestamp = None
        self._request_stop(
            "Robot Agent requested a correlated simulation safe-stop",
            terminal=GaitState.DISARMED,
            now=timestamp,
        )
        return self._emit(timestamp)

    def tick(self, *, now: float | None = None) -> SonicIntent:
        """Advance timeout/acknowledgement state without an incoming event."""

        timestamp = self._resolve_now(now)
        self._evaluate(timestamp)
        return self._emit(timestamp)

    def observe_reference_authority(
        self,
        *,
        generation: int,
        source_sequence: int,
        reference_epoch: int,
        dds_target_sequence: int,
        granted_at_monotonic_s: float,
        now: float | None = None,
    ) -> bool:
        """Start measured-motion timing from one correlated successful DDS grant."""

        timestamp = self._resolve_now(now)
        integer_values = (
            (generation, 0, "generation"),
            (source_sequence, 1, "source_sequence"),
            (reference_epoch, 1, "reference_epoch"),
            (dds_target_sequence, 1, "dds_target_sequence"),
        )
        for value, minimum, name in integer_values:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"reference authority {name} is invalid")
        granted_at = float(granted_at_monotonic_s)
        if (
            not math.isfinite(granted_at)
            or granted_at < 0
            or granted_at > timestamp + self.config.future_timestamp_tolerance_s
        ):
            raise ValueError("reference authority grant timestamp is invalid")
        if self.state is not GaitState.STARTING or generation != self.generation:
            return False
        required_sequence = self._reference_authority_source_sequence
        wait_started_at = self._reference_wait_started_at
        if (
            required_sequence is None
            or wait_started_at is None
            or source_sequence != required_sequence
            or granted_at < wait_started_at
        ):
            return False
        if self._start_at is not None:
            return bool(
                reference_epoch == self._reference_authority_epoch
                and dds_target_sequence == self._reference_authority_dds_sequence
                and granted_at == self._start_at
            )

        self._start_at = granted_at
        self._motion_authority_granted_at = granted_at
        self._reference_authority_epoch = reference_epoch
        self._reference_authority_dds_sequence = dds_target_sequence
        self._motion_started_in_generation = True
        self.reason = "waiting for measured gait start"
        observation = self._observation
        # Navigation motion begins at the first correlated DDS authority fact,
        # not when Nav2 first expresses intent.  In particular, the official
        # policy INIT/settle phase may move the floating base while actuator
        # authority is still REQUIRED.  Re-anchor unconditionally here so that
        # preparation drift can neither trip nor satisfy the motion guards.
        if observation is not None:
            self._motion_anchor = (
                observation.x,
                observation.y,
                observation.yaw,
            )
            self._previous_yaw = observation.yaw
            if self._active_direction is None:
                self._expected_map_direction = None
            else:
                direction_x, direction_y = self._active_direction
                cosine = math.cos(observation.yaw)
                sine = math.sin(observation.yaw)
                self._expected_map_direction = (
                    cosine * direction_x - sine * direction_y,
                    sine * direction_x + cosine * direction_y,
                )
        else:
            self._motion_anchor = None
            self._expected_map_direction = None
            self._previous_yaw = None
        self._startup_positive_progress_seen = False
        return True

    def _evaluate(self, now: float) -> None:
        if self.state is GaitState.FAULT:
            return

        if self.state is GaitState.STOPPING:
            if self._stop_acknowledged:
                self._complete_stop()
                return
            stop_timeout_s = (
                self.config.balanced_idle_stop_timeout_s
                if self._stop_terminal is GaitState.IDLE
                else self.config.stop_timeout_s
            )
            if (
                self._stop_started_at is not None
                and now - self._stop_started_at > stop_timeout_s
            ):
                self._trip_fault("measured base did not acknowledge stop before timeout", now)
            return

        if self.state is GaitState.DISARMED:
            return

        if not self._observation_is_fresh(now):
            if self.state is GaitState.IDLE:
                self.state = GaitState.DISARMED
                # Losing an observation before this generation has ever
                # granted motion is a startup/reconnect synchronization loss,
                # not evidence that a moving robot is in an unknown state.
                # Drop every command and require a new fresh stationary sample
                # plus an explicit arm.  Once STARTING was entered, even a
                # later IDLE observation loss retains the stronger new-
                # generation interlock.
                self._requires_new_generation = self._motion_started_in_generation
                self._command = None
                self._last_command_timestamp = None
                self.reason = (
                    "base observation became stale while idle; "
                    + (
                        "newer generation required after prior motion"
                        if self._requires_new_generation
                        else "fresh stationary observation and explicit arm required"
                    )
                )
            else:
                self._trip_fault("base observation became stale during motion", now)
            return

        if self.state is GaitState.IDLE:
            if self._command is None:
                return
            if self._command_is_stale(now):
                self._command = None
                self._last_command_timestamp = None
                return
            desired = self._desired_state(self._command)
            if desired is GaitState.IDLE:
                return
            self._begin_start(desired, self._command, now)
            return

        if self._command is None or self._command_is_stale(now):
            if not self.config.trip_on_automatic_fault:
                self._record_automatic_fault(
                    "Nav2 velocity command became stale",
                )
                return
            self._requires_new_generation = True
            self._request_stop(
                "Nav2 velocity command became stale",
                terminal=GaitState.DISARMED,
                now=now,
            )
            return

        desired = self._desired_state(self._command)
        if desired is GaitState.IDLE:
            self._request_stop("Nav2 requested zero velocity", terminal=GaitState.IDLE, now=now)
            return
        if self._motion_change_requires_stop(desired, self._command):
            self._request_stop("Nav2 gait mode or direction changed", terminal=GaitState.IDLE, now=now)
            return

        if self.state is GaitState.STARTING and self._start_at is not None:
            if now - self._start_at > self.config.start_timeout_s:
                self._trip_fault("measured base did not acknowledge gait start before timeout", now)

    def _begin_start(self, desired: GaitState, command: Nav2Twist, now: float) -> None:
        self.state = GaitState.STARTING
        self.reason = "waiting for matched SONIC reference actuator authority"
        self._start_at = None
        self._motion_authority_granted_at = None
        self._reference_wait_started_at = now
        self._reference_authority_source_sequence = None
        self._reference_authority_epoch = None
        self._reference_authority_dds_sequence = None
        self._active_kind = desired
        linear = math.hypot(command.vx, command.vy)
        if desired is GaitState.WALKING and linear > 0:
            self._active_direction = (command.vx / linear, command.vy / linear)
            self._active_turn_sign = 0.0
        else:
            self._active_direction = None
            self._active_turn_sign = math.copysign(1.0, command.wz)
        # This pre-authority state is deliberately not a motion baseline.  The
        # correlated first DDS grant installs the only accepted baseline.
        self._motion_anchor = None
        self._expected_map_direction = None
        self._previous_yaw = None
        self._cumulative_yaw = 0.0
        self._startup_positive_progress_seen = False
        self._stop_acknowledged = False

    def _check_start_ack(self, observation: BaseObservation) -> None:
        if self._start_at is None or observation.timestamp_s < self._start_at:
            return
        if self._active_kind is GaitState.WALKING and self._active_direction is not None:
            projection = (
                observation.vx_body * self._active_direction[0]
                + observation.vy_body * self._active_direction[1]
            )
            if projection >= self.config.start_linear_ack_mps:
                self.state = GaitState.WALKING
                self.reason = None
                self._start_at = None
        elif self._active_kind is GaitState.TURNING:
            if self._active_turn_sign * observation.wz >= self.config.start_angular_ack_rps:
                self.state = GaitState.TURNING
                self.reason = None
                self._start_at = None

    def _check_reverse_motion(self, observation: BaseObservation, now: float) -> None:
        """Reject cumulative opposite travel, not a humanoid gait-phase swing.

        A humanoid can show short centre-of-mass or foot-placement transients;
        the failed run contained eight millimetres of opposite displacement,
        but lacked foot-contact evidence to classify its cause.  Direction
        safety therefore does not infer reverse navigation from one velocity
        phase.  It uses meaningful pose displacement in the map/odom frame,
        matching the independent ROS navigation guard.  Turns likewise
        integrate adjacent wrapped yaw samples instead of testing one angular
        velocity sample.
        """

        anchor = self._motion_anchor
        if (
            self._active_kind is GaitState.WALKING
            and anchor is not None
            and self._expected_map_direction is not None
        ):
            dx = observation.x - anchor[0]
            dy = observation.y - anchor[1]
            travel = math.hypot(dx, dy)
            projection = (
                dx * self._expected_map_direction[0]
                + dy * self._expected_map_direction[1]
            )
            granted_at = self._motion_authority_granted_at
            elapsed = (
                0.0
                if granted_at is None
                else max(0.0, observation.timestamp_s - granted_at)
            )
            if projection >= self.config.startup_min_progress_m:
                self._startup_positive_progress_seen = True
            if (
                travel >= self.config.reverse_min_travel_m
                and projection <= -self.config.reverse_translation_m
            ):
                self._trip_fault(
                    "measured translation opposes Nav2 "
                    f"(travel={travel:.3f}m, reverse_projection={-projection:.3f}m)",
                    now,
                )
                return
            if (
                granted_at is not None
                and elapsed > self.config.startup_reverse_grace_s
                and projection <= -self.config.startup_min_progress_m
            ):
                self._trip_fault(
                    "measured translation remained opposite Nav2 beyond the "
                    f"startup grace (elapsed={elapsed:.3f}s, "
                    f"reverse_projection={-projection:.3f}m)",
                    now,
                )
                return
            if (
                granted_at is not None
                and elapsed > self.config.startup_progress_deadline_s
                and not self._startup_positive_progress_seen
            ):
                self._trip_fault(
                    "measured translation made no net positive Nav2 progress "
                    f"after authority grant (elapsed={elapsed:.3f}s, "
                    f"projection={projection:.3f}m)",
                    now,
                )
            return

        if self._active_kind is not GaitState.TURNING or not self._active_turn_sign:
            return
        if self._previous_yaw is None:
            self._previous_yaw = observation.yaw
            return
        self._cumulative_yaw += self._wrap_angle(
            observation.yaw - self._previous_yaw
        )
        self._previous_yaw = observation.yaw
        signed_progress = self._active_turn_sign * self._cumulative_yaw
        if signed_progress <= -self.config.reverse_rotation_rad:
            self._trip_fault(
                "measured rotation opposes Nav2 "
                f"(reverse={-signed_progress:.3f}rad)",
                now,
            )

    def _motion_change_requires_stop(self, desired: GaitState, command: Nav2Twist) -> bool:
        if self._active_kind is None or desired is not self._active_kind:
            return True
        if desired is GaitState.TURNING:
            return self._active_turn_sign * command.wz < 0
        if self._active_direction is None:
            return True
        linear = math.hypot(command.vx, command.vy)
        if linear <= 0:
            return True
        direction = (command.vx / linear, command.vy / linear)
        dot = max(-1.0, min(1.0, direction[0] * self._active_direction[0] + direction[1] * self._active_direction[1]))
        return math.acos(dot) > self.config.direction_restart_rad

    def _request_stop(self, reason: str, *, terminal: GaitState, now: float) -> None:
        if self.state is GaitState.FAULT:
            return
        self.state = GaitState.STOPPING
        self.reason = reason
        self._stop_started_at = now
        self._stop_terminal = terminal
        self._reference_wait_started_at = None
        self._reference_authority_source_sequence = None
        self._reference_authority_epoch = None
        self._reference_authority_dds_sequence = None
        self._reset_stop_ack()

    def _complete_stop(self) -> None:
        terminal = self._stop_terminal
        reason = self.reason
        self.state = terminal
        self._start_at = None
        self._motion_authority_granted_at = None
        self._reference_wait_started_at = None
        self._reference_authority_source_sequence = None
        self._reference_authority_epoch = None
        self._reference_authority_dds_sequence = None
        self._active_kind = None
        self._active_direction = None
        self._active_turn_sign = 0.0
        self._motion_anchor = None
        self._expected_map_direction = None
        self._previous_yaw = None
        self._cumulative_yaw = 0.0
        self._startup_positive_progress_seen = False
        self._stop_started_at = None
        if terminal is GaitState.IDLE:
            self.reason = None
            self._stop_acknowledged = True
        else:
            self.reason = f"{reason}; measured stop acknowledged" if reason else "measured stop acknowledged"
            self._command = None
            self._last_command_timestamp = None

    def _trip_fault(self, reason: str, now: float) -> None:
        if not self.config.trip_on_automatic_fault:
            self._record_automatic_fault(reason)
            return
        if self.state is not GaitState.FAULT:
            self._reset_stop_ack()
        self.state = GaitState.FAULT
        self.reason = reason
        self._requires_new_generation = True
        self._stop_started_at = now
        self._start_at = None
        self._motion_authority_granted_at = None
        self._reference_wait_started_at = None
        self._reference_authority_source_sequence = None
        self._reference_authority_epoch = None
        self._reference_authority_dds_sequence = None
        self._startup_positive_progress_seen = False
        self._safe_stop_requested = False

    def _record_automatic_fault(self, reason: str) -> None:
        """Retain the current valid policy while preserving diagnostics."""

        bounded = str(reason).strip()[:512] or "automatic gait fault"
        self.automatic_fault_count += 1
        self.last_automatic_fault_reason = bounded

    def _observe_stop_ack(self, observation: BaseObservation, now: float) -> None:
        # Callback arrival after a stop request is not enough: a queued ROS
        # sample may have been captured while the old motion command still had
        # authority.  Both values share the monotonic domain established by
        # the bridge, so only capture timestamps strictly after this stop began
        # may contribute evidence.
        if (
            self._stop_started_at is None
            or observation.timestamp_s <= self._stop_started_at
        ):
            return
        if now - observation.timestamp_s > self.config.observation_timeout_s:
            # A delayed sample cannot prove the robot's current state.  Break
            # the evidence window as well as rejecting this sample so a later
            # fresh capture cannot bridge an unobserved interval.
            self._stationary_since = None
            self._stationary_samples = 0
            self._last_stop_observation_timestamp = observation.timestamp_s
            self._stop_acknowledged = False
            self._reset_balanced_idle_stop_window()
            return
        if (
            self._last_stop_observation_timestamp is not None
            and observation.timestamp_s <= self._last_stop_observation_timestamp
        ):
            return
        self._last_stop_observation_timestamp = observation.timestamp_s
        balanced_idle_ack = self._observe_balanced_idle_stop(observation)
        if not self._is_stationary(observation):
            self._stationary_since = None
            self._stationary_samples = 0
            self._stop_acknowledged = balanced_idle_ack
            return
        if self._stationary_since is None:
            self._stationary_since = observation.timestamp_s
            self._stationary_samples = 1
        else:
            self._stationary_samples += 1
        strict_ack = (
            self._stationary_samples >= self.config.stop_ack_samples
            and observation.timestamp_s - self._stationary_since
            >= self.config.stop_ack_s
        )
        self._stop_acknowledged = strict_ack or balanced_idle_ack

    def _observe_balanced_idle_stop(self, observation: BaseObservation) -> bool:
        """Prove bounded learned-IDLE balance without accepting real motion."""

        if (
            self.state is not GaitState.STOPPING
            or self._stop_terminal is not GaitState.IDLE
        ):
            self._reset_balanced_idle_stop_window()
            return False

        linear_speed = math.hypot(observation.vx_body, observation.vy_body)
        if (
            linear_speed > self.config.balanced_idle_stop_max_linear_mps
            or abs(observation.wz)
            > self.config.balanced_idle_stop_max_angular_rps
        ):
            self._reset_balanced_idle_stop_window()
            return False

        window = self._balanced_idle_stop_window
        if (
            window
            and observation.timestamp_s - window[-1][0]
            > self.config.balanced_idle_stop_max_sample_gap_s
        ):
            self._reset_balanced_idle_stop_window()
            window = self._balanced_idle_stop_window

        if self._balanced_idle_stop_last_raw_yaw is None:
            unwrapped_yaw = observation.yaw
        else:
            assert self._balanced_idle_stop_last_unwrapped_yaw is not None
            unwrapped_yaw = (
                self._balanced_idle_stop_last_unwrapped_yaw
                + self._wrap_angle(
                    observation.yaw - self._balanced_idle_stop_last_raw_yaw
                )
            )
        self._balanced_idle_stop_last_raw_yaw = observation.yaw
        self._balanced_idle_stop_last_unwrapped_yaw = unwrapped_yaw
        window.append(
            (
                observation.timestamp_s,
                observation.x,
                observation.y,
                unwrapped_yaw,
            )
        )

        required_span = self.config.balanced_idle_stop_window_s
        # Retain one capture on or immediately before the left edge. This
        # proves the full span despite discrete 10 Hz odometry samples.
        while (
            len(window) >= 2
            and observation.timestamp_s - window[1][0] >= required_span
        ):
            window.popleft()
        if (
            len(window) < self.config.balanced_idle_stop_samples
            or observation.timestamp_s - window[0][0] < required_span
        ):
            return False

        samples = tuple(window)
        linear_diameter = max(
            math.hypot(left[1] - right[1], left[2] - right[2])
            for index, left in enumerate(samples)
            for right in samples[index:]
        )
        if linear_diameter > self.config.balanced_idle_stop_linear_excursion_m:
            return False
        yaw_values = tuple(sample[3] for sample in samples)
        if (
            max(yaw_values) - min(yaw_values)
            > self.config.balanced_idle_stop_yaw_excursion_rad
        ):
            return False

        elapsed = samples[-1][0] - samples[0][0]
        if elapsed <= 0.0:
            return False
        net_linear_speed = math.hypot(
            samples[-1][1] - samples[0][1],
            samples[-1][2] - samples[0][2],
        ) / elapsed
        net_angular_speed = abs(samples[-1][3] - samples[0][3]) / elapsed
        return (
            net_linear_speed <= self.config.stop_linear_ack_mps
            and net_angular_speed <= self.config.stop_angular_ack_rps
        )

    def _reset_stop_ack(self) -> None:
        self._stationary_since = None
        self._stationary_samples = 0
        self._last_stop_observation_timestamp = None
        self._stop_acknowledged = False
        self._reset_balanced_idle_stop_window()

    def _reset_balanced_idle_stop_window(self) -> None:
        self._balanced_idle_stop_window.clear()
        self._balanced_idle_stop_last_raw_yaw = None
        self._balanced_idle_stop_last_unwrapped_yaw = None

    def _select_generation(self, generation: int) -> None:
        self.generation = generation
        self.state = GaitState.DISARMED
        self.reason = "plant generation selected; awaiting fresh observation and arm"
        self._observation = None
        self._command = None
        self._last_observation_timestamp = None
        self._last_command_timestamp = None
        self._requires_new_generation = False
        self._motion_started_in_generation = False
        self._start_at = None
        self._motion_authority_granted_at = None
        self._reference_wait_started_at = None
        self._reference_authority_source_sequence = None
        self._reference_authority_epoch = None
        self._reference_authority_dds_sequence = None
        self._active_kind = None
        self._active_direction = None
        self._active_turn_sign = 0.0
        self._motion_anchor = None
        self._expected_map_direction = None
        self._previous_yaw = None
        self._cumulative_yaw = 0.0
        self._startup_positive_progress_seen = False
        self._stop_started_at = None
        self._stop_terminal = GaitState.IDLE
        self._safe_stop_requested = False
        self._reset_stop_ack()

    def _desired_state(self, command: Nav2Twist) -> GaitState:
        if math.hypot(command.vx, command.vy) >= self.config.linear_deadband_mps:
            return GaitState.WALKING
        if abs(command.wz) >= self.config.angular_deadband_rps:
            return GaitState.TURNING
        return GaitState.IDLE

    def _command_is_stale(self, now: float) -> bool:
        return self._command is None or now - self._command.timestamp_s > self.config.command_timeout_s

    def _observation_is_fresh(self, now: float) -> bool:
        return (
            self._observation is not None
            and now - self._observation.timestamp_s <= self.config.observation_timeout_s
        )

    def _is_stationary(self, observation: BaseObservation) -> bool:
        return (
            math.hypot(observation.vx_body, observation.vy_body) <= self.config.stop_linear_ack_mps
            and abs(observation.wz) <= self.config.stop_angular_ack_rps
        )

    def _bounded(self, command: Nav2Twist) -> Nav2Twist:
        vx = float(command.vx)
        vy = float(command.vy)
        linear = math.hypot(vx, vy)
        if linear > self.config.max_linear_mps:
            scale = self.config.max_linear_mps / linear
            vx *= scale
            vy *= scale
        wz = max(-self.config.max_angular_rps, min(self.config.max_angular_rps, float(command.wz)))
        return Nav2Twist(
            generation=command.generation,
            timestamp_s=command.timestamp_s,
            vx=vx,
            vy=vy,
            wz=wz,
            cancelled=False,
            source=command.source,
        )

    def _emit(self, now: float, *, reason: str | None = None) -> SonicIntent:
        self._sequence += 1
        if (
            self.state is GaitState.STARTING
            and self._reference_authority_source_sequence is None
        ):
            self._reference_authority_source_sequence = self._sequence
        moving = self.state in (
            GaitState.STARTING,
            GaitState.WALKING,
            GaitState.TURNING,
        )
        command = self._command if moving else None
        vx = command.vx if command is not None else 0.0
        vy = command.vy if command is not None else 0.0
        wz = command.wz if command is not None else 0.0
        linear = math.hypot(vx, vy)
        movement_angle = math.atan2(vy, vx) if linear > 0 else 0.0
        facing_delta = max(
            -self.config.max_facing_delta_rad,
            min(self.config.max_facing_delta_rad, wz * self.config.turn_preview_s),
        )
        return SonicIntent(
            sequence=self._sequence,
            generation=self.generation,
            generated_at_s=now,
            state=self.state.value,
            mode="SLOW_WALK" if moving else "IDLE",
            velocity_x_mps=vx,
            velocity_y_mps=vy,
            linear_speed_mps=linear,
            movement_angle_rad=movement_angle,
            yaw_rate_rps=wz,
            facing_delta_rad=facing_delta,
            motion_enabled=moving,
            stop_requested=self.state in (GaitState.DISARMED, GaitState.STOPPING, GaitState.FAULT),
            stop_acknowledged=(self.state is GaitState.IDLE or self._stop_acknowledged),
            head_yaw_rad=0.0,
            head_pitch_rad=0.0,
            source="nav2",
            reason=reason if reason is not None else self.reason,
        )

    def _resolve_now(self, now: float | None) -> float:
        raw = self._clock() if now is None else now
        try:
            timestamp = float(raw)
        except (TypeError, ValueError):
            timestamp = math.nan
        if not math.isfinite(timestamp):
            safe_now = self._last_now if self._last_now is not None else 0.0
            self._trip_fault("adapter clock is non-finite", safe_now)
            return safe_now
        if self._last_now is not None and timestamp < self._last_now:
            self._trip_fault("adapter clock moved backwards", self._last_now)
            return self._last_now
        self._last_now = timestamp
        return timestamp

    def _valid_command(self, command: Nav2Twist, now: float) -> bool:
        return (
            isinstance(command, Nav2Twist)
            and self._valid_generation(command.generation)
            and command.source == "nav2"
            and isinstance(command.cancelled, bool)
            and self._valid_timestamp(command.timestamp_s, now)
            and self._all_finite(command.vx, command.vy, command.wz)
        )

    def _valid_observation(self, observation: BaseObservation, now: float) -> bool:
        return (
            isinstance(observation, BaseObservation)
            and self._valid_generation(observation.generation)
            and self._valid_timestamp(observation.timestamp_s, now)
            and self._all_finite(
                observation.x,
                observation.y,
                observation.yaw,
                observation.roll,
                observation.pitch,
                observation.vx_body,
                observation.vy_body,
                observation.wz,
            )
        )

    def _valid_timestamp(self, timestamp: float, now: float) -> bool:
        try:
            value = float(timestamp)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value <= now + self.config.future_timestamp_tolerance_s

    @staticmethod
    def _valid_generation(generation: int) -> bool:
        return isinstance(generation, int) and not isinstance(generation, bool) and generation >= 0

    @staticmethod
    def _all_finite(*values: object) -> bool:
        try:
            return all(math.isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "BaseObservation",
    "GaitAdapter",
    "GaitAdapterConfig",
    "GaitState",
    "Nav2Twist",
    "SonicIntent",
]
