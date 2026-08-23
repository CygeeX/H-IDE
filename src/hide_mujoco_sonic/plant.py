"""Run the official Unitree MuJoCo plant with an explicit H-IDE config.

Unitree DDS and ROS 2 use different CycloneDDS language bindings.  Keeping
them in separate processes prevents one runtime from changing the other's
global DDS state.  This plant therefore publishes pose/RGB-D through a
loopback-only observation endpoint; `ros_observation_bridge` republishes the
same capture atomically into the Robot Agent ROS contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Callable, Iterable
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np

from .config import SimConfig
from .frames import BodyTwist, read_mujoco_local_body_twist
from .render_workers import (
    LatestStateMailbox,
    MAX_CAPTURE_AGE_NS,
    MujocoStateShape,
    PassiveViewerWorker,
    RenderedFrame,
    RenderWorkerConfig,
    RgbdRenderWorker,
    ViewerWorkerConfig,
)
from .runtime_diagnostics import PlantRuntimeDiagnostics
from .scene_runtime import RuntimeScene


class PlantPhysicsFault(RuntimeError):
    """A latched MuJoCo fault that requires an explicit process restart."""


_OPERATOR_VIEWER_ENV = "HIDE_MUJOCO_ONSCREEN_VIEWER"
_DEFERRED_OPERATOR_VIEWER_ENV = "HIDE_MUJOCO_DEFER_ONSCREEN_VIEWER"


def operator_viewer_enabled(config: SimConfig) -> bool:
    """Return whether the non-authoritative native viewer is requested.

    The reviewed workstation config stays byte-for-byte locked. Operators may
    explicitly add the private passive viewer for a local demonstration without
    changing observation cadence, support timing, model selectors, or any
    motion-capable process. Only the launcher-owned ``0``/``1`` values are
    accepted so a misspelled environment value cannot silently change mode.
    """

    override = os.environ.get(_OPERATOR_VIEWER_ENV, "0")
    if override not in {"0", "1"}:
        raise RuntimeError(f"{_OPERATOR_VIEWER_ENV} must be exactly 0 or 1")
    return config.onscreen or override == "1"


def operator_viewer_deferred(config: SimConfig) -> bool:
    """Return whether launcher-triggered viewer creation is requested."""

    override = os.environ.get(_DEFERRED_OPERATOR_VIEWER_ENV, "0")
    if override not in {"0", "1"}:
        raise RuntimeError(
            f"{_DEFERRED_OPERATOR_VIEWER_ENV} must be exactly 0 or 1"
        )
    if override == "1" and not operator_viewer_enabled(config):
        raise RuntimeError("deferred MuJoCo viewer requires an enabled viewer")
    return override == "1"


@dataclass(frozen=True, slots=True)
class _SupportCommandState:
    mode: str
    updated_at: float
    safety_stop_epoch: int
    last_rearm_generation: int
    motion_fault: str | None
    last_cleared_motion_fault: str | None
    fall_arrest_release_epoch_requested: int | None


@dataclass(frozen=True, slots=True)
class _SupportPhysicsState:
    fault: str | None
    fault_updated_at: float
    last_simulation_time: float | None
    safety_stop_epoch_floor: int


@dataclass(frozen=True, slots=True)
class _SupportBandState:
    enabled: bool
    release_epoch_confirmed: int | None
    release_epoch_rejected: int | None
    updated_at: float


@dataclass(frozen=True, slots=True)
class _FrameWindow:
    """One atomically published, immutable HTTP capture generation."""

    latest: RenderedFrame | None
    previous: RenderedFrame | None


@dataclass(frozen=True, slots=True)
class _PlantDynamicsSnapshot:
    """One immutable authority-thread sample for read-only fault evidence."""

    sequence: int
    captured_at_unix_ns: int
    simulation_time_s: float | None
    support_band_enabled: bool
    attached_body: str | None
    last_applied_wrench: tuple[float | None, ...]
    root_qpos: tuple[float | None, ...]
    root_qvel: tuple[float | None, ...]
    root_qacc: tuple[float | None, ...]
    root_qacc_warmstart: tuple[float | None, ...]
    robot_center_of_mass_world_m: tuple[float | None, ...]
    potential_energy_j: float | None
    kinetic_energy_j: float | None
    warning_counts: tuple[tuple[str, int], ...]
    foot_ground_sides: tuple[str, ...]
    foot_ground_forces_n: tuple[tuple[str, float], ...]
    foot_touchdown_counts: tuple[tuple[str, int], ...]
    foot_liftoff_counts: tuple[tuple[str, int], ...]
    obstacle_contact_pairs: tuple[tuple[str, str, float | None], ...]
    nonfoot_ground_pairs: tuple[tuple[str, str, float | None], ...]
    obstacle_contact_sample_count: int
    obstacle_contact_event_count: int
    nonfoot_ground_contact_sample_count: int
    nonfoot_ground_contact_event_count: int
    contact_evidence_valid: bool


_PHYSICS_WARNING_NAMES = (
    "mjWARN_BADQPOS",
    "mjWARN_BADQVEL",
    "mjWARN_BADQACC",
    "mjWARN_BADCTRL",
)


class PlantSupportController:
    """Simulation-only fall-arrest state, never a navigation actuator.

    SONIC and MuJoCo remain connected exclusively through Unitree DDS for joint
    control.  This controller only changes the external safety wrench that the
    upstream simulator exposes for tethered testing.  It cannot create a goal,
    steer the base, or write qpos/qvel.
    """

    MODES = frozenset({"startup", "idle", "locomotion", "safety_stop"})

    def __init__(self, *, fall_arrest_enabled: bool = True) -> None:
        if sys.implementation.name != "cpython":
            raise RuntimeError("lock-free Plant support snapshots require CPython")
        gil_enabled = getattr(sys, "_is_gil_enabled", None)
        if callable(gil_enabled) and not gil_enabled():
            raise RuntimeError("lock-free Plant support snapshots require the GIL")
        now = time.time()
        if not isinstance(fall_arrest_enabled, bool):
            raise TypeError("fall_arrest_enabled must be a boolean")
        self._command_state = _SupportCommandState(
            mode="startup",
            updated_at=now,
            safety_stop_epoch=0,
            last_rearm_generation=-1,
            motion_fault=None,
            last_cleared_motion_fault=None,
            fall_arrest_release_epoch_requested=None,
        )
        self._physics_state = _SupportPhysicsState(
            fault=None,
            fault_updated_at=now,
            last_simulation_time=None,
            safety_stop_epoch_floor=0,
        )
        self._band_state = _SupportBandState(
            enabled=fall_arrest_enabled,
            # Epoch zero means this Plant has no elastic band to release.
            # Positive epochs are reserved for an explicit SONIC activation.
            release_epoch_confirmed=None if fall_arrest_enabled else 0,
            release_epoch_rejected=None,
            updated_at=now,
        )
        # Serializes the rare one-way band release with every stop/fault
        # transition. It is deliberately separate from the HTTP writer lock:
        # an urgent physics fault must never wait behind a slow request.
        self._release_transition_lock = threading.Lock()
        # Only HTTP/background command writers use this lock.  Physics reads
        # immutable state references and publishes its own immutable state, so
        # an arbitrarily slow request can never delay mj_step/LowState.
        self._write_lock = threading.Lock()

    @property
    def mode(self) -> str:
        command = self._command_state
        if self._physics_state.fault is not None:
            return "safety_stop"
        return command.mode

    @property
    def physics_fault(self) -> str | None:
        return self._physics_state.fault

    def set_mode(self, mode: str) -> str:
        value = str(mode)
        if value not in self.MODES:
            raise ValueError(
                f"support mode must be one of {', '.join(sorted(self.MODES))}"
            )
        with self._write_lock, self._release_transition_lock:
            physics = self._physics_state
            command = self._command_state
            effective_mode = "safety_stop" if physics.fault is not None else command.mode
            if physics.fault is not None and value != "safety_stop":
                raise ValueError(
                    f"plant physics fault is latched: {physics.fault}"
                )
            if effective_mode == "safety_stop" and value != "safety_stop":
                raise ValueError(
                    "plant safety_stop is latched; explicit control rearm is required"
                )
            epoch = max(
                command.safety_stop_epoch,
                physics.safety_stop_epoch_floor,
            )
            if value == "safety_stop" and effective_mode != "safety_stop":
                epoch += 1
            self._command_state = replace(
                command,
                mode=value,
                updated_at=time.time(),
                safety_stop_epoch=epoch,
            )
            return self.mode

    def rearm(self, generation: int) -> str:
        """Release a non-fault safety latch to IDLE for one newer authority.

        Normal support transitions can only strengthen a stop once the plant
        reaches ``safety_stop``.  The ROS bridge owns the fresh stationary
        observation and SONIC owns this rearm call; requiring a strictly newer
        generation prevents periodic IDLE traffic or a replayed Nav2 command
        from releasing the latch.  A physics fault remains process-restart
        only and is never recoverable through this method.
        """

        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise ValueError("rearm generation must be a positive integer")
        with self._write_lock, self._release_transition_lock:
            physics = self._physics_state
            command = self._command_state
            if physics.fault is not None:
                raise ValueError(
                    f"plant physics fault is latched: {physics.fault}"
                )
            if command.motion_fault is not None:
                raise ValueError(
                    "plant motion fault must be explicitly reset before rearm"
                )
            if command.mode != "safety_stop":
                raise ValueError("plant rearm requires a latched safety_stop")
            if generation <= command.last_rearm_generation:
                raise ValueError(
                    "plant rearm generation must be strictly newer than the last release"
                )
            self._command_state = replace(
                command,
                mode="idle",
                updated_at=time.time(),
                safety_stop_epoch=max(
                    command.safety_stop_epoch,
                    physics.safety_stop_epoch_floor,
                ),
                last_rearm_generation=generation,
            )
            return self.mode

    def latch_motion_fault(self, reason: str) -> bool:
        """Hold a recoverable navigation fault without stopping MuJoCo."""

        detail = str(reason).strip()
        if not detail or len(detail) > 512:
            raise ValueError("plant motion fault reason must contain 1..512 characters")
        with self._write_lock, self._release_transition_lock:
            if self._physics_state.fault is not None:
                raise ValueError(
                    f"plant physics fault is latched: {self._physics_state.fault}"
                )
            command = self._command_state
            if command.motion_fault is not None:
                if command.motion_fault != detail:
                    raise ValueError("plant already carries another motion fault")
                return False
            self._command_state = replace(
                command,
                mode="safety_stop",
                updated_at=time.time(),
                safety_stop_epoch=command.safety_stop_epoch + 1,
                motion_fault=detail,
            )
            return True

    def clear_motion_fault(self, reason: str) -> bool:
        """Clear only the named recoverable fault, retaining safety stop."""

        detail = str(reason).strip()
        if not detail or len(detail) > 512:
            raise ValueError("plant motion fault reason must contain 1..512 characters")
        with self._write_lock, self._release_transition_lock:
            if self._physics_state.fault is not None:
                raise ValueError(
                    f"plant physics fault is latched: {self._physics_state.fault}"
                )
            command = self._command_state
            if command.mode != "safety_stop":
                raise ValueError("plant motion fault reset requires safety_stop")
            if command.motion_fault is None:
                if command.last_cleared_motion_fault == detail:
                    return False
                raise ValueError("plant has no recoverable motion fault")
            if command.motion_fault != detail:
                raise ValueError("plant motion fault identity mismatch")
            self._command_state = replace(
                command,
                updated_at=time.time(),
                motion_fault=None,
                last_cleared_motion_fault=detail,
            )
            return True

    def request_fall_arrest_release(self, activation_epoch: int) -> bool:
        """Request a one-way elastic-band release at a safe boundary."""

        if (
            isinstance(activation_epoch, bool)
            or not isinstance(activation_epoch, int)
            or activation_epoch < 0
        ):
            raise ValueError(
                "fall-arrest release activation epoch must be a non-negative integer"
            )
        with self._write_lock, self._release_transition_lock:
            physics = self._physics_state
            command = self._command_state
            band = self._band_state
            if physics.fault is not None:
                raise ValueError(
                    f"plant physics fault is latched: {physics.fault}"
                )
            if command.motion_fault is not None or command.mode == "safety_stop":
                raise ValueError(
                    "fall-arrest release is forbidden at a motion safety boundary"
                )
            # Epoch zero is reserved for the legacy bounded startup timer.
            # Explicit SONIC activation release must happen only from IDLE.
            if activation_epoch > 0 and command.mode != "idle":
                raise ValueError(
                    "fall-arrest release requires confirmed idle support"
                )
            if band.release_epoch_rejected is not None:
                if band.release_epoch_rejected == activation_epoch:
                    raise ValueError(
                        "fall-arrest release was rejected at the physics boundary"
                    )
                raise ValueError(
                    "fall-arrest release already belongs to another activation"
                )
            if not band.enabled:
                if band.release_epoch_confirmed not in {0, activation_epoch}:
                    raise ValueError(
                        "fall-arrest release already belongs to another activation"
                    )
                return False
            existing = command.fall_arrest_release_epoch_requested
            if existing is not None:
                if existing != activation_epoch:
                    raise ValueError(
                        "fall-arrest release already belongs to another activation"
                    )
                return False
            self._command_state = replace(
                command,
                updated_at=time.time(),
                fall_arrest_release_epoch_requested=activation_epoch,
            )
            return True

    def consume_fall_arrest_release(
        self,
        release_band: Callable[[], None],
    ) -> int | None:
        """Disable and confirm the band on the authoritative simulation thread."""

        if not callable(release_band):
            raise TypeError("fall-arrest release callback must be callable")
        with self._release_transition_lock:
            command = self._command_state
            physics = self._physics_state
            band = self._band_state
            epoch = command.fall_arrest_release_epoch_requested
            if epoch is None or not band.enabled:
                return None
            if band.release_epoch_rejected == epoch:
                return None
            if (
                physics.fault is not None
                or command.motion_fault is not None
                or command.mode == "safety_stop"
                or (epoch > 0 and command.mode != "idle")
            ):
                # A request is only an intent. The simulation thread owns the
                # final release boundary and permanently rejects an epoch that
                # races a stop/fault/mode transition, so it cannot be consumed
                # later after an unrelated reset.
                self._band_state = replace(
                    band,
                    release_epoch_rejected=epoch,
                    updated_at=time.time(),
                )
                return None
            try:
                release_band()
            except Exception:
                self._band_state = replace(
                    band,
                    release_epoch_rejected=epoch,
                    updated_at=time.time(),
                )
                raise
            self._band_state = replace(
                band,
                enabled=False,
                release_epoch_confirmed=epoch,
                updated_at=time.time(),
            )
            return epoch

    def finish_startup(self) -> str:
        with self._write_lock, self._release_transition_lock:
            command = self._command_state
            if self._physics_state.fault is None and command.mode == "startup":
                self._command_state = replace(
                    command,
                    mode="idle",
                    updated_at=time.time(),
                )
            return self.mode

    def latch_fault(self, reason: str) -> bool:
        detail = str(reason).strip()
        if not detail:
            raise ValueError("plant physics fault requires a reason")
        with self._release_transition_lock:
            physics = self._physics_state
            if physics.fault is not None:
                return False
            command = self._command_state
            current_epoch = max(
                command.safety_stop_epoch,
                physics.safety_stop_epoch_floor,
            )
            # A new physics fault is a distinct stop authority even when an
            # HTTP safety_stop/rearm races this lock-free publisher. Advancing
            # the floor unconditionally prevents a concurrent rearm from
            # hiding it.
            current_epoch += 1
            self._physics_state = replace(
                physics,
                fault=detail,
                fault_updated_at=time.time(),
                safety_stop_epoch_floor=current_epoch,
            )
            return True

    def observe_simulation_time(self, value: float) -> bool:
        """Latch an unrecoverable fault when MuJoCo implicitly resets time."""

        current = float(value)
        if not math.isfinite(current) or current < 0:
            return self.latch_fault("MuJoCo simulation time is invalid")
        # The authoritative simulator thread is the sole writer of physics
        # state.  HTTP request threads only publish command state under their
        # separate lock, so this copy-on-write update cannot wait for HTTP.
        physics = self._physics_state
        previous = physics.last_simulation_time
        if previous is not None and current + 1e-6 < previous:
            return self.latch_fault(
                "MuJoCo physics time rolled back after numerical instability "
                f"({previous:.6f}s -> {current:.6f}s)"
            )
        self._physics_state = replace(
            physics,
            last_simulation_time=current,
        )
        return False

    def fail(self, reason: str) -> None:
        """Latch one immutable fault and terminate the acceptance plant.

        Keeping the process alive after invalid physics lets downstream
        components consume an old pose and can make a later implicit reset look
        like continuous motion. The strict H-IDE profile therefore exits and
        requires an operator-visible restart.
        """

        self.latch_fault(reason)
        raise PlantPhysicsFault(str(self._physics_state.fault))

    def require_healthy(self) -> None:
        fault = self._physics_state.fault
        if fault is not None:
            raise PlantPhysicsFault(str(fault))

    def state(self) -> dict[str, object]:
        # Both references are immutable and are replaced atomically under the
        # CPython GIL.  This projection never acquires the HTTP writer lock.
        command = self._command_state
        physics = self._physics_state
        band = self._band_state
        fault = physics.fault
        mode = "safety_stop" if fault is not None else command.mode
        generation = command.last_rearm_generation
        return {
            "mode": mode,
            "updatedAt": max(
                command.updated_at,
                physics.fault_updated_at,
                band.updated_at,
            ),
            "fault": fault,
            "motionFault": command.motion_fault,
            "safetyStopLatched": mode == "safety_stop",
            "safetyStopEpoch": max(
                command.safety_stop_epoch,
                physics.safety_stop_epoch_floor,
            ),
            "lastRearmGeneration": None if generation < 0 else generation,
            "fallArrestEnabled": band.enabled,
            "fallArrestReleaseRequestedEpoch": (
                command.fall_arrest_release_epoch_requested
            ),
            "fallArrestReleaseConfirmedEpoch": band.release_epoch_confirmed,
            "fallArrestReleaseRejectedEpoch": band.release_epoch_rejected,
        }


class PlantControlSynchronizer:
    """Pause MuJoCo time until one fresh 50 Hz actuator command arrives.

    Hardware time cannot pause, so a real controller must treat a missed
    deadline as a fault.  MuJoCo *can* pause safely.  Once the policy-owned
    fall-arrest release proves that SONIC controls the untethered plant, this
    barrier turns the asynchronous DDS transport into the same observation ->
    controller -> actuator ordering used by the official synchronous
    simulator examples.  Host scheduling jitter slows wall time instead of
    dropping a learned-policy observation or terminating a healthy scene.

    The barrier is deliberately inactive during cold start and for
    support-always-on standalone demos.  It never catches up, repeats a policy
    step, or advances simulation without a newly received LowCmd.  A missing
    controller eventually latches the existing Plant physics fault.
    """

    def __init__(
        self,
        support: PlantSupportController,
        *,
        control_hz: int,
        command_identity: Callable[[], tuple[int, int | None, bool]],
        wait_for_command_for_tick: Callable[[int, int, float], bool],
        max_wall_wait_s: float = 2.0,
        wall_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(control_hz, bool) or int(control_hz) <= 0:
            raise ValueError("control synchronizer frequency must be positive")
        if 1_000 % int(control_hz) != 0:
            raise ValueError("control synchronizer frequency must divide 1000")
        if isinstance(max_wall_wait_s, bool):
            raise ValueError("control synchronizer timeout must be finite and positive")
        timeout = float(max_wall_wait_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("control synchronizer timeout must be finite and positive")
        if not callable(command_identity) or not callable(wait_for_command_for_tick):
            raise TypeError("control synchronizer requires command identity callbacks")
        self._support = support
        self._period_ms = 1_000 // int(control_hz)
        self._command_identity = command_identity
        self._wait_for_command_for_tick = wait_for_command_for_tick
        self._max_wall_wait_s = timeout
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._last_control_tick_ms: int | None = None
        self._wait_count = 0
        self._max_wait_s = 0.0
        self._last_wait_s = 0.0
        self._last_command_sequence: int | None = None
        self._active_release_epoch: int | None = None
        self._phase_ms: int | None = None
        self._phase_source_tick_ms: int | None = None

    @staticmethod
    def _sequence(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("LowCmd sequence must be a non-negative integer")
        return value

    @classmethod
    def _identity(
        cls,
        value: object,
    ) -> tuple[int, int | None, bool]:
        if not isinstance(value, tuple) or len(value) != 3:
            raise ValueError("LowCmd identity must be a (sequence, tick, async) tuple")
        sequence = cls._sequence(value[0])
        tick = value[1]
        asynchronous = value[2]
        if tick is not None and (
            isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
        ):
            raise ValueError("LowCmd control tick must be a non-negative integer")
        if not isinstance(asynchronous, bool):
            raise ValueError("LowCmd async-boundary marker must be boolean")
        if asynchronous and tick is not None:
            raise ValueError("async LowCmd must not claim a policy control tick")
        return sequence, tick, asynchronous

    def command_identity_before_publish(self) -> tuple[int, int | None, bool]:
        return self._identity(self._command_identity())

    def _release_epoch(self) -> int | None:
        state = self._support.state()
        epoch = state.get("fallArrestReleaseConfirmedEpoch")
        if (
            state.get("fallArrestEnabled") is False
            and isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and epoch > 0
        ):
            return epoch
        return None

    def wait_after_lowstate(
        self,
        *,
        tick_ms: int,
        command_identity_before_publish: tuple[int, int | None, bool],
    ) -> bool:
        if isinstance(tick_ms, bool) or not isinstance(tick_ms, int) or tick_ms < 0:
            raise ValueError("control synchronization tick must be non-negative")
        identity = self._identity(command_identity_before_publish)
        sequence, command_tick_ms, asynchronous = identity
        release_epoch = self._release_epoch()
        with self._lock:
            if release_epoch is None:
                self._active_release_epoch = None
                self._phase_ms = None
                self._phase_source_tick_ms = None
                return False
            if self._active_release_epoch != release_epoch:
                self._active_release_epoch = release_epoch
                self._phase_ms = None
                self._phase_source_tick_ms = None
                self._last_control_tick_ms = None
            if self._phase_ms is None:
                if asynchronous or command_tick_ms is None:
                    reason = (
                        "MuJoCo control synchronization has no SONIC phase provenance "
                        f"after support release epoch {release_epoch}"
                    )
                    self._support.latch_fault(reason)
                    raise PlantPhysicsFault(reason)
                command_age_ms = tick_ms - command_tick_ms
                if not 0 <= command_age_ms <= self._period_ms:
                    reason = (
                        "MuJoCo control synchronization received stale or future "
                        "SONIC phase provenance: "
                        f"command={command_tick_ms}ms plant={tick_ms}ms"
                    )
                    self._support.latch_fault(reason)
                    raise PlantPhysicsFault(reason)
                self._phase_ms = command_tick_ms % self._period_ms
                self._phase_source_tick_ms = command_tick_ms
            if tick_ms % self._period_ms != self._phase_ms:
                return False
            previous_tick = self._last_control_tick_ms
            if previous_tick is not None and tick_ms <= previous_tick:
                reason = (
                    "MuJoCo synchronized control tick did not advance: "
                    f"{tick_ms}ms after {previous_tick}ms"
                )
                self._support.latch_fault(reason)
                raise PlantPhysicsFault(reason)
            self._last_control_tick_ms = tick_ms

        started = self._wall_clock()
        deadline = started + self._max_wall_wait_s
        while True:
            self._support.require_healthy()
            remaining = deadline - self._wall_clock()
            if remaining <= 0.0:
                reason = (
                    "MuJoCo synchronized control timed out waiting for a fresh "
                    f"LowCmd after tick {tick_ms}ms"
                )
                self._support.latch_fault(reason)
                raise PlantPhysicsFault(reason)
            if self._wait_for_command_for_tick(
                sequence,
                tick_ms,
                min(0.05, remaining),
            ):
                break

        observed_sequence, observed_tick_ms, observed_async = self._identity(
            self._command_identity()
        )
        if observed_sequence <= sequence:
            reason = "MuJoCo control synchronization reported a stale LowCmd"
            self._support.latch_fault(reason)
            raise PlantPhysicsFault(reason)
        waited_s = max(0.0, self._wall_clock() - started)
        with self._lock:
            self._wait_count += 1
            self._last_wait_s = waited_s
            self._max_wait_s = max(self._max_wait_s, waited_s)
            self._last_command_sequence = observed_sequence
            if not observed_async and observed_tick_ms != tick_ms:
                reason = (
                    "MuJoCo control synchronization observed a mismatched LowCmd: "
                    f"expected={tick_ms}ms observed={observed_tick_ms}ms"
                )
                self._support.latch_fault(reason)
                raise PlantPhysicsFault(reason)
        return True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "schemaVersion": "h-ide.mujoco-control-synchronization/1",
                "active": self._release_epoch() is not None,
                "periodMs": self._period_ms,
                "phaseMs": self._phase_ms,
                "phaseSourceTickMs": self._phase_source_tick_ms,
                "releaseEpoch": self._active_release_epoch,
                "maxWallWaitS": self._max_wall_wait_s,
                "lastControlTickMs": self._last_control_tick_ms,
                "waitCount": self._wait_count,
                "lastWaitMs": self._last_wait_s * 1_000.0,
                "maxWaitMs": self._max_wait_s * 1_000.0,
                "lastCommandSequence": self._last_command_sequence,
            }


def stabilized_support_wrench(
    raw_wrench,
    pose,
    mode: str,
    *,
    tilt_kp: float = 250.0,
    tilt_kd: float = 30.0,
):
    """Keep fall arrest independent from horizontal navigation authority.

    The upstream elastic band produces a 6D wrench.  Navigation must remain
    owned by Nav2/SONIC, so normal locomotion removes world-X/Y forces and yaw
    torque.  It must *not* remove vertical or roll/pitch support: doing so made
    a long G1 walk fall over, and a later Nav2 cancel could only stop commands,
    not the already tumbling rigid body.
    """

    wrench = np.asarray(raw_wrench, dtype=float).copy()
    state = np.asarray(pose, dtype=float)
    if wrench.shape != (6,) or state.shape != (13,):
        raise ValueError("support wrench requires a 6D wrench and 13D pose")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0
        for value in (tilt_kp, tilt_kd)
    ):
        raise ValueError("tilt support gains must be finite and non-negative")

    # The upstream band derives one axis-angle vector from the *complete*
    # orientation.  Its X/Y components are not roll/pitch-only: close to a
    # 180-degree yaw they change sign discontinuously.  Keeping those raw
    # components caused a large external-torque step exactly when a reviewed
    # pi Spin crossed the quaternion branch cut, eventually driving MuJoCo to
    # huge QACC and an implicit reset.  Align the pelvis up-axis with world Z
    # instead.  This tilt error is continuous and exactly invariant to yaw.
    qw, qx, qy, qz = (float(value) for value in state[3:7])
    quaternion_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(quaternion_norm) or quaternion_norm < 1e-9:
        raise ValueError("support pose quaternion must be finite and non-zero")
    qw, qx, qy, qz = (
        value / quaternion_norm for value in (qw, qx, qy, qz)
    )
    body_up_x = 2.0 * (qx * qz + qw * qy)
    body_up_y = 2.0 * (qy * qz - qw * qx)
    wrench[3] = float(tilt_kp) * body_up_y - float(tilt_kd) * float(state[10])
    wrench[4] = -float(tilt_kp) * body_up_x - float(tilt_kd) * float(state[11])

    if mode == "locomotion":
        wrench[:2] = 0.0
        wrench[5] = 0.0
    elif mode == "idle":
        wrench[0] = -120.0 * float(state[7])
        wrench[1] = -120.0 * float(state[8])
        wrench[5] = 0.0
    elif mode == "safety_stop":
        # Emergency braking is velocity damping, never a position target.  It
        # cannot drive toward the map origin.  Only damp world X/Y here while
        # retaining vertical and attitude fall arrest.  Applying a yaw torque
        # directly to the tethered torso is not equivalent to actuator-space
        # damping: an accepted 0.2 m run showed that ``-80 * wz`` creates an
        # unbounded, numerically unstable sampled-data loop through the
        # waist/contact dynamics.  A
        # high-fidelity reconstructed replay left the validated envelope in
        # 64 ms and reached BADQACC in 76 ms, while the same frozen joint
        # targets with X/Y-only damping remained bounded for two seconds.
        # A controlled yaw stop must therefore come from the SONIC joint
        # controller; a terminal gait fault halts the Plant separately.
        wrench[0] = -180.0 * float(state[7])
        wrench[1] = -180.0 * float(state[8])
        wrench[5] = 0.0
    elif mode == "startup":
        wrench[5] = 0.0
    else:
        raise ValueError(f"unsupported support mode: {mode}")
    return wrench


def apply_process_start_spawn(data, model, scene: RuntimeScene) -> None:
    """Apply the manifest spawn exactly once during plant construction.

    This helper only initializes the newly-created ``MjData``. It is not used
    by any runtime fault/recovery path, preserving the explicit-restart reset
    contract.
    """

    if int(model.nq) < 7 or int(model.nv) < 6:
        raise RuntimeError("Office V2 runtime model must have a floating root")
    half_yaw = float(scene.spawn_yaw) / 2.0
    data.qpos[:3] = np.asarray(
        [scene.spawn_x, scene.spawn_y, float(model.qpos0[2])],
        dtype=float,
    )
    data.qpos[3:7] = np.asarray(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        dtype=float,
    )
    data.qvel[:] = 0.0


def configure_head_camera(mujoco, model) -> int:
    """Make the articulated G1 camera look level along the base +X axis."""

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "head_camera",
    )
    if camera_id < 0:
        raise RuntimeError("pinned G1 model does not define head_camera")
    camera_rotation = np.asarray(
        [
            0.0, 0.0, -1.0,
            -1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
        ],
        dtype=float,
    )
    mujoco.mju_mat2Quat(model.cam_quat[camera_id], camera_rotation)
    model.cam_fovy[camera_id] = 75.0
    return int(camera_id)


def configure_tracking_viewer(mujoco, model, viewer, scene: RuntimeScene) -> None:
    """Start the operator viewer close to the spawned robot and track it."""

    if viewer is None:
        return
    mujoco.mjv_defaultOption(viewer.opt)
    pelvis_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    )
    if pelvis_id < 0:
        raise RuntimeError("pinned G1 model does not define pelvis")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = pelvis_id
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -10.0
    viewer.cam.distance = 2.5
    viewer.cam.lookat = np.asarray(
        [scene.spawn_x, scene.spawn_y, float(model.qpos0[2])],
        dtype=float,
    )


class PlantDynamicsEvidence:
    """Publish complete, diagnostic-only dynamics without blocking physics.

    The authoritative MuJoCo thread is the sole writer.  HTTP readers project
    one immutable reference under the CPython GIL, exactly like the Plant
    support state.  These fields are never consumed by a controller or
    republished by the ROS observation bridge; they exist to make a numerical
    stop fault reconstructable after the process has exited.
    """

    _WRENCH_ORDER = (
        "force_x",
        "force_y",
        "force_z",
        "torque_x",
        "torque_y",
        "torque_z",
    )
    _ROOT_QPOS_ORDER = ("x", "y", "z", "qw", "qx", "qy", "qz")
    _ROOT_QVEL_ORDER = tuple(f"qvel[{index}]" for index in range(6))
    _ROOT_QACC_ORDER = tuple(f"qacc[{index}]" for index in range(6))
    _FOOT_CONTACT_FORCE_THRESHOLD_N = 5.0
    _FOOT_CONTACT_DEBOUNCE_SAMPLES = 5

    def __init__(
        self,
        mujoco_api,
        model,
        data,
        *,
        support_band=None,
        attached_body_id: int | None = None,
    ) -> None:
        self.mujoco = mujoco_api
        self.model = model
        self.data = data
        self.support_band = support_band
        self.root_body_id = int(
            mujoco_api.mj_name2id(
                model,
                mujoco_api.mjtObj.mjOBJ_BODY,
                "pelvis",
            )
        )
        if self.root_body_id < 0:
            raise RuntimeError("pinned G1 model does not define pelvis")
        if attached_body_id is None:
            self.attached_body_id = None
            self.attached_body_name = None
        else:
            body_id = int(attached_body_id)
            if not (0 <= body_id < int(model.nbody)):
                raise ValueError("support attached body ID is invalid")
            body_name = mujoco_api.mj_id2name(
                model,
                mujoco_api.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if not body_name:
                raise ValueError("support attached body must have a stable name")
            self.attached_body_id = body_id
            self.attached_body_name = str(body_name)
        body_parent_ids = tuple(int(value) for value in model.body_parentid)

        def body_descends_from(body_id: int, ancestor_id: int) -> bool:
            current = int(body_id)
            while current > 0:
                if current == ancestor_id:
                    return True
                current = body_parent_ids[current]
            return current == ancestor_id

        self._body_names = tuple(
            str(
                mujoco_api.mj_id2name(
                    model,
                    mujoco_api.mjtObj.mjOBJ_BODY,
                    body_id,
                )
                or f"body#{body_id}"
            )
            for body_id in range(int(model.nbody))
        )
        self._geom_names = tuple(
            str(
                mujoco_api.mj_id2name(
                    model,
                    mujoco_api.mjtObj.mjOBJ_GEOM,
                    geom_id,
                )
                or f"geom#{geom_id}"
            )
            for geom_id in range(int(model.ngeom))
        )
        self._robot_body_ids = frozenset(
            body_id
            for body_id in range(int(model.nbody))
            if body_descends_from(body_id, self.root_body_id)
        )
        self._robot_geom_ids = frozenset(
            geom_id
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) in self._robot_body_ids
        )
        foot_side_by_body: dict[int, str] = {}
        for side in ("left", "right"):
            ankle_id = int(
                mujoco_api.mj_name2id(
                    model,
                    mujoco_api.mjtObj.mjOBJ_BODY,
                    f"{side}_ankle_roll_link",
                )
            )
            if ankle_id < 0:
                continue
            for body_id in self._robot_body_ids:
                if body_descends_from(body_id, ankle_id):
                    foot_side_by_body[body_id] = side
        self._foot_side_by_body = foot_side_by_body
        plane_type = int(mujoco_api.mjtGeom.mjGEOM_PLANE)
        self._floor_geom_ids = frozenset(
            geom_id
            for geom_id in range(int(model.ngeom))
            if self._geom_names[geom_id] == "office_v2_floor"
            or (
                int(model.geom_type[geom_id]) == plane_type
                and int(model.geom_bodyid[geom_id]) == 0
            )
        )
        self._static_obstacle_geom_ids = frozenset(
            geom_id
            for geom_id in range(int(model.ngeom))
            if geom_id not in self._robot_geom_ids
            and geom_id not in self._floor_geom_ids
            and int(model.geom_contype[geom_id]) != 0
            and int(model.geom_conaffinity[geom_id]) != 0
        )
        self._sequence = 0
        self._snapshot: _PlantDynamicsSnapshot | None = None
        self._obstacle_contact_sample_count = 0
        self._obstacle_contact_event_count = 0
        self._nonfoot_ground_contact_sample_count = 0
        self._nonfoot_ground_contact_event_count = 0
        self._obstacle_contact_active = False
        self._nonfoot_ground_contact_active = False
        self._foot_contact_active = {"left": False, "right": False}
        self._foot_contact_candidate = {"left": False, "right": False}
        self._foot_contact_candidate_samples = {"left": 0, "right": 0}
        self._foot_touchdown_counts = {"left": 0, "right": 0}
        self._foot_liftoff_counts = {"left": 0, "right": 0}

    @staticmethod
    def _finite(value: object) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @classmethod
    def _finite_tuple(cls, values) -> tuple[float | None, ...]:
        return tuple(cls._finite(value) for value in values)

    def _warning_counts(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (
                name.removeprefix("mjWARN_"),
                int(
                    self.data.warning[
                        int(getattr(self.mujoco.mjtWarning, name))
                    ].number
                ),
            )
            for name in _PHYSICS_WARNING_NAMES
        )

    def _contact_evidence(
        self,
    ) -> tuple[
        tuple[tuple[str, float], ...],
        tuple[tuple[str, str, float | None], ...],
        tuple[tuple[str, str, float | None], ...],
    ]:
        """Classify active contacts without feeding them back into control."""

        foot_forces_n = {"left": 0.0, "right": 0.0}
        obstacle_pairs: dict[tuple[str, str], float | None] = {}
        nonfoot_ground_pairs: dict[tuple[str, str], float | None] = {}
        for contact_index, contact in enumerate(self.data.contact):
            first = int(contact.geom1)
            second = int(contact.geom2)
            if first in self._robot_geom_ids:
                robot_geom, environment_geom = first, second
            elif second in self._robot_geom_ids:
                robot_geom, environment_geom = second, first
            else:
                continue
            if environment_geom not in (
                self._floor_geom_ids | self._static_obstacle_geom_ids
            ):
                continue
            robot_body_id = int(self.model.geom_bodyid[robot_geom])
            robot_body = self._body_names[robot_body_id]
            environment_name = self._geom_names[environment_geom]
            distance = self._finite(contact.dist)
            penetration = (
                None if distance is None else max(0.0, -distance)
            )
            key = (robot_body, environment_name)
            if environment_geom in self._floor_geom_ids:
                side = self._foot_side_by_body.get(robot_body_id)
                if side is not None:
                    force = np.zeros(6, dtype=float)
                    self.mujoco.mj_contactForce(
                        self.model,
                        self.data,
                        contact_index,
                        force,
                    )
                    normal_force = self._finite(force[0])
                    if normal_force is None:
                        raise ValueError("foot contact force is non-finite")
                    foot_forces_n[side] += max(0.0, normal_force)
                else:
                    previous = nonfoot_ground_pairs.get(key)
                    nonfoot_ground_pairs[key] = (
                        penetration
                        if previous is None or penetration is None
                        else max(previous, penetration)
                    )
            else:
                previous = obstacle_pairs.get(key)
                obstacle_pairs[key] = (
                    penetration
                    if previous is None or penetration is None
                    else max(previous, penetration)
                )

        def frozen_pairs(
            values: dict[tuple[str, str], float | None],
        ) -> tuple[tuple[str, str, float | None], ...]:
            return tuple(
                (robot_body, environment_geom, penetration)
                for (robot_body, environment_geom), penetration in sorted(
                    values.items()
                )
            )

        return (
            tuple(sorted(foot_forces_n.items())),
            frozen_pairs(obstacle_pairs),
            frozen_pairs(nonfoot_ground_pairs),
        )

    def observe(self, *, captured_at_unix_ns: int | None = None) -> None:
        """Capture the state after one authoritative step without any lock."""

        energy = (None, None)
        try:
            self.mujoco.mj_energyPos(self.model, self.data)
            self.mujoco.mj_energyVel(self.model, self.data)
            energy = tuple(self._finite(value) for value in self.data.energy[:2])
        except Exception:
            # Evidence must never become a new failure source.  The physics
            # monitor independently treats non-finite authoritative state as a
            # terminal fault; an unavailable energy projection remains explicit.
            energy = (None, None)
        if self.attached_body_id is None:
            wrench = (0.0,) * 6
        else:
            wrench = self._finite_tuple(
                self.data.xfrc_applied[self.attached_body_id]
            )
        contact_evidence_valid = True
        try:
            (
                foot_ground_forces_n,
                obstacle_contact_pairs,
                nonfoot_ground_pairs,
            ) = self._contact_evidence()
        except Exception:
            # Contact evidence is diagnostic and must never become a new
            # physics failure source. Mark the sample invalid so acceptance
            # fails closed instead of reporting a fabricated zero-contact run.
            contact_evidence_valid = False
            foot_ground_forces_n = (("left", 0.0), ("right", 0.0))
            obstacle_contact_pairs = ()
            nonfoot_ground_pairs = ()
        foot_force_map = dict(foot_ground_forces_n)
        if contact_evidence_valid:
            for side in ("left", "right"):
                raw_active = (
                    float(foot_force_map.get(side, 0.0))
                    >= self._FOOT_CONTACT_FORCE_THRESHOLD_N
                )
                if raw_active == self._foot_contact_candidate[side]:
                    self._foot_contact_candidate_samples[side] += 1
                else:
                    self._foot_contact_candidate[side] = raw_active
                    self._foot_contact_candidate_samples[side] = 1
                if (
                    raw_active != self._foot_contact_active[side]
                    and self._foot_contact_candidate_samples[side]
                    >= self._FOOT_CONTACT_DEBOUNCE_SAMPLES
                ):
                    self._foot_contact_active[side] = raw_active
                    if raw_active:
                        self._foot_touchdown_counts[side] += 1
                    else:
                        self._foot_liftoff_counts[side] += 1
        foot_ground_sides = tuple(
            side
            for side in ("left", "right")
            if self._foot_contact_active[side]
        )
        obstacle_active = bool(obstacle_contact_pairs)
        nonfoot_ground_active = bool(nonfoot_ground_pairs)
        if obstacle_active:
            self._obstacle_contact_sample_count += 1
            if not self._obstacle_contact_active:
                self._obstacle_contact_event_count += 1
        if nonfoot_ground_active:
            self._nonfoot_ground_contact_sample_count += 1
            if not self._nonfoot_ground_contact_active:
                self._nonfoot_ground_contact_event_count += 1
        self._obstacle_contact_active = obstacle_active
        self._nonfoot_ground_contact_active = nonfoot_ground_active
        self._sequence += 1
        self._snapshot = _PlantDynamicsSnapshot(
            sequence=self._sequence,
            captured_at_unix_ns=(
                time.time_ns()
                if captured_at_unix_ns is None
                else int(captured_at_unix_ns)
            ),
            simulation_time_s=self._finite(self.data.time),
            support_band_enabled=(
                self.support_band is not None
                and bool(getattr(self.support_band, "enable", False))
            ),
            attached_body=self.attached_body_name,
            last_applied_wrench=wrench,
            root_qpos=self._finite_tuple(self.data.qpos[:7]),
            root_qvel=self._finite_tuple(self.data.qvel[:6]),
            root_qacc=self._finite_tuple(self.data.qacc[:6]),
            root_qacc_warmstart=self._finite_tuple(
                self.data.qacc_warmstart[:6]
            ),
            robot_center_of_mass_world_m=self._finite_tuple(
                self.data.subtree_com[self.root_body_id]
            ),
            potential_energy_j=energy[0],
            kinetic_energy_j=energy[1],
            warning_counts=self._warning_counts(),
            foot_ground_sides=foot_ground_sides,
            foot_ground_forces_n=foot_ground_forces_n,
            foot_touchdown_counts=tuple(
                sorted(self._foot_touchdown_counts.items())
            ),
            foot_liftoff_counts=tuple(
                sorted(self._foot_liftoff_counts.items())
            ),
            obstacle_contact_pairs=obstacle_contact_pairs,
            nonfoot_ground_pairs=nonfoot_ground_pairs,
            obstacle_contact_sample_count=self._obstacle_contact_sample_count,
            obstacle_contact_event_count=self._obstacle_contact_event_count,
            nonfoot_ground_contact_sample_count=(
                self._nonfoot_ground_contact_sample_count
            ),
            nonfoot_ground_contact_event_count=(
                self._nonfoot_ground_contact_event_count
            ),
            contact_evidence_valid=contact_evidence_valid,
        )

    def capture_state(self) -> dict[str, object]:
        """Return one JSON-safe projection without reading live ``MjData``."""

        snapshot = self._snapshot
        if snapshot is None:
            return {
                "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
                "status": "starting",
                "diagnosticOnly": True,
                "authorityFeedbackEnabled": False,
            }
        finite_values = (
            *snapshot.last_applied_wrench,
            *snapshot.root_qpos,
            *snapshot.root_qvel,
            *snapshot.root_qacc,
            *snapshot.root_qacc_warmstart,
            *snapshot.robot_center_of_mass_world_m,
            snapshot.simulation_time_s,
            snapshot.potential_energy_j,
            snapshot.kinetic_energy_j,
        )
        state_finite = bool(
            snapshot.contact_evidence_valid
            and all(value is not None for value in finite_values)
        )
        total_energy = None
        if (
            snapshot.potential_energy_j is not None
            and snapshot.kinetic_energy_j is not None
        ):
            total_energy = (
                snapshot.potential_energy_j + snapshot.kinetic_energy_j
            )
        return {
            "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
            "status": "running" if state_finite else "invalid",
            "diagnosticOnly": True,
            "authorityFeedbackEnabled": False,
            "sampleSequence": snapshot.sequence,
            "capturedAtUnixNs": snapshot.captured_at_unix_ns,
            "simulationTimeS": snapshot.simulation_time_s,
            "supportBandEnabled": snapshot.support_band_enabled,
            "attachedBody": snapshot.attached_body,
            "lastAppliedWrench": {
                "frameId": "world",
                "order": list(self._WRENCH_ORDER),
                "values": list(snapshot.last_applied_wrench),
            },
            "rootState": {
                "body": "pelvis",
                "qposOrder": list(self._ROOT_QPOS_ORDER),
                "qpos": list(snapshot.root_qpos),
                "generalizedVelocityRepresentation": (
                    "raw MuJoCo free-joint qvel; preserve order for replay"
                ),
                "qvelOrder": list(self._ROOT_QVEL_ORDER),
                "qvel": list(snapshot.root_qvel),
                "qaccOrder": list(self._ROOT_QACC_ORDER),
                "qacc": list(snapshot.root_qacc),
                "qaccWarmstart": list(snapshot.root_qacc_warmstart),
            },
            "robotCenterOfMass": {
                "frameId": "world",
                "positionOrder": ["x", "y", "z"],
                "positionM": list(snapshot.robot_center_of_mass_world_m),
            },
            "energy": {
                "potentialJ": snapshot.potential_energy_j,
                "kineticJ": snapshot.kinetic_energy_j,
                "totalJ": total_energy,
            },
            "contacts": {
                "schemaVersion": "h-ide.mujoco-contact-evidence/1",
                "sampleSequence": snapshot.sequence,
                "valid": snapshot.contact_evidence_valid,
                "footGroundSides": list(snapshot.foot_ground_sides),
                "footGroundForceN": dict(snapshot.foot_ground_forces_n),
                "footContactThresholdN": (
                    self._FOOT_CONTACT_FORCE_THRESHOLD_N
                ),
                "footContactDebounceSamples": (
                    self._FOOT_CONTACT_DEBOUNCE_SAMPLES
                ),
                "footTouchdownCount": dict(snapshot.foot_touchdown_counts),
                "footLiftoffCount": dict(snapshot.foot_liftoff_counts),
                "obstaclePairs": [
                    {
                        "robotBody": robot_body,
                        "staticGeom": static_geom,
                        "penetrationM": penetration,
                    }
                    for robot_body, static_geom, penetration
                    in snapshot.obstacle_contact_pairs
                ],
                "nonFootGroundPairs": [
                    {
                        "robotBody": robot_body,
                        "groundGeom": ground_geom,
                        "penetrationM": penetration,
                    }
                    for robot_body, ground_geom, penetration
                    in snapshot.nonfoot_ground_pairs
                ],
                # Cumulative counters are sampled at the authoritative 250 Hz
                # physics boundary, so a brief contact cannot disappear
                # between slower HTTP acceptance polls.
                "obstacleContactSampleCount": (
                    snapshot.obstacle_contact_sample_count
                ),
                "obstacleContactEventCount": (
                    snapshot.obstacle_contact_event_count
                ),
                "nonFootGroundContactSampleCount": (
                    snapshot.nonfoot_ground_contact_sample_count
                ),
                "nonFootGroundContactEventCount": (
                    snapshot.nonfoot_ground_contact_event_count
                ),
            },
            "warningCounts": dict(snapshot.warning_counts),
            "finite": state_finite,
        }


class PlantPhysicsMonitor:
    """Authority-thread checks that never depend on rendering availability."""

    _WARNING_NAMES = _PHYSICS_WARNING_NAMES

    def __init__(self, config: SimConfig, mujoco_api, model, data, support) -> None:
        self.config = config
        self.mujoco = mujoco_api
        self.model = model
        self.data = data
        self.support = support
        self.root_body_id = int(
            mujoco_api.mj_name2id(
                model,
                mujoco_api.mjtObj.mjOBJ_BODY,
                "pelvis",
            )
        )
        if self.root_body_id < 0:
            raise RuntimeError("pinned G1 model does not define pelvis")
        self._warning_counts = self._read_warning_counts()
        active_warnings = {
            name: count
            for name, count in self._warning_counts.items()
            if count != 0
        }
        if active_warnings:
            self.support.fail(
                "MuJoCo warning counters are non-zero before the first Plant step: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in active_warnings.items()
                )
            )

    def _read_warning_counts(self) -> dict[str, int]:
        return {
            name.removeprefix("mjWARN_"): int(
                self.data.warning[
                    int(getattr(self.mujoco.mjtWarning, name))
                ].number
            )
            for name in self._WARNING_NAMES
        }

    def observe(self) -> None:
        self.support.observe_simulation_time(float(self.data.time))
        self.support.require_healthy()
        warning_counts = self._read_warning_counts()
        for name, current in warning_counts.items():
            previous = self._warning_counts[name]
            if current > previous:
                self.support.fail(
                    f"MuJoCo {name} warning count increased "
                    f"({previous} -> {current})"
                )
        self._warning_counts = warning_counts
        arrays = (
            self.data.qpos,
            self.data.qvel,
            self.data.qacc,
            self.data.ctrl,
            self.data.act,
            self.data.actuator_force,
        )
        if not all(np.all(np.isfinite(values)) for values in arrays):
            self.support.fail("MuJoCo physics state contains non-finite values")
        root_pose = np.asarray(self.data.qpos[:7], dtype=float)
        try:
            read_mujoco_local_body_twist(
                self.mujoco,
                self.model,
                self.data,
                self.root_body_id,
            )
        except ValueError as exc:
            self.support.fail(f"MuJoCo root velocity is invalid: {exc}")
        x, y, z = (float(value) for value in root_pose[:3])
        bounds_min = self.config.runtime_scene.bounds_min
        bounds_max = self.config.runtime_scene.bounds_max
        if not (
            bounds_min[0] <= x <= bounds_max[0]
            and bounds_min[1] <= y <= bounds_max[1]
            and 0.20 <= z <= 1.50
        ):
            self.support.fail(
                "MuJoCo root pose left the validated office envelope "
                f"(x={x:.3f}, y={y:.3f}, z={z:.3f})"
            )


class PlantFrameStore:
    def __init__(
        self,
        config: SimConfig,
        support: PlantSupportController,
        runtime_diagnostics: PlantRuntimeDiagnostics | None = None,
        worker_status_provider=None,
        viewer_status_provider=None,
        dynamics_evidence_provider=None,
        control_synchronization_provider=None,
    ):
        self.config = config
        self.support = support
        self.runtime_diagnostics = runtime_diagnostics
        # The RGB-D producer boot identifies one render stream.  Keep a
        # separate process-local session identity for successful motion
        # evidence so a collector cannot splice samples across two Plant
        # lifetimes that happen to expose the same logical robot/configuration.
        self.session_id = str(uuid.uuid4())
        # The collector is the sole writer.  HTTP readers load one immutable
        # window reference without touching this commit-only lock, so a slow
        # response can neither block a commit nor observe a torn latest/previous
        # pair.
        self._commit_lock = threading.Lock()
        self._window = _FrameWindow(latest=None, previous=None)
        self._worker_status_provider = worker_status_provider
        self._viewer_status_provider = viewer_status_provider
        self._dynamics_evidence_provider = dynamics_evidence_provider
        self._control_synchronization_provider = control_synchronization_provider
        self._servers: list[ThreadingHTTPServer] = []

    def set_worker_status_provider(self, provider) -> None:
        self._worker_status_provider = provider

    def set_viewer_status_provider(self, provider) -> None:
        self._viewer_status_provider = provider

    def set_dynamics_evidence_provider(self, provider) -> None:
        self._dynamics_evidence_provider = provider

    def set_control_synchronization_provider(self, provider) -> None:
        self._control_synchronization_provider = provider

    def commit(self, frame: RenderedFrame) -> None:
        """Atomically install one complete RGB-D/state generation."""

        if (
            frame.sequence <= 0
            or frame.captured_at_unix_ns <= 0
            or frame.captured_at_monotonic_ns <= 0
        ):
            raise ValueError("rendered frame identity is invalid")
        if frame.render_started_at_unix_ns <= 0 or frame.rendered_at_unix_ns <= 0:
            raise ValueError("rendered frame wall timestamps are invalid")
        if not (
            frame.captured_at_monotonic_ns
            <= frame.render_started_at_monotonic_ns
            <= frame.rendered_at_monotonic_ns
        ):
            raise ValueError("rendered frame monotonic timestamps are not ordered")
        uuid.UUID(frame.producer_boot_id)
        if (
            frame.width <= 0
            or frame.height <= 0
            or not isinstance(frame.rgb_jpeg, bytes)
            or not frame.rgb_jpeg
        ):
            raise ValueError("rendered frame RGB payload is invalid")
        if (
            not isinstance(frame.depth_z16_le, bytes)
            or len(frame.depth_z16_le) != frame.width * frame.height * 2
        ):
            raise ValueError("rendered frame depth payload is invalid")
        if not isinstance(frame.pose, tuple) or len(frame.pose) != 7:
            raise ValueError("rendered frame root pose shape is invalid")
        if not isinstance(frame.camera_pose, tuple) or len(frame.camera_pose) != 7:
            raise ValueError("rendered frame camera pose shape is invalid")
        values = (*frame.pose, *frame.camera_pose, frame.simulation_time_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("rendered frame geometry contains non-finite values")
        intrinsics = dict(frame.intrinsics)
        if set(intrinsics) != {"fx", "fy", "cx", "cy"} or not all(
            math.isfinite(float(value)) for value in intrinsics.values()
        ):
            raise ValueError("rendered frame intrinsics are invalid")
        if float(intrinsics["fx"]) <= 0 or float(intrinsics["fy"]) <= 0:
            raise ValueError("rendered frame focal lengths must be positive")
        timings = dict(frame.render_timing)
        if not all(
            math.isfinite(float(value)) and float(value) >= 0
            for value in timings.values()
        ):
            raise ValueError("rendered frame timing is invalid")
        with self._commit_lock:
            window = self._window
            latest = window.latest
            previous = window.previous
            if latest is not None:
                if latest.producer_boot_id == frame.producer_boot_id:
                    if frame.sequence <= latest.sequence:
                        raise ValueError("rendered frame sequence did not increase")
                    previous = latest
                else:
                    previous = None
            self._window = _FrameWindow(latest=frame, previous=previous)

    def _worker_status(self) -> dict[str, object]:
        provider = self._worker_status_provider
        if provider is None:
            return {
                "status": "starting",
                "ready": False,
                "alive": False,
                "lastError": "RGB-D worker is not configured",
            }
        try:
            value = provider()
        except Exception as exc:
            return {
                "status": "failed",
                "ready": False,
                "alive": False,
                "lastError": f"worker status unavailable: {type(exc).__name__}: {exc}"[:512],
            }
        return dict(value)

    def _viewer_status(self) -> dict[str, object]:
        provider = self._viewer_status_provider
        if provider is None:
            return {
                "status": "disabled",
                "authorityFeedbackEnabled": False,
                "keyCallbackRegistered": False,
            }
        try:
            return dict(provider())
        except Exception as exc:
            return {
                "status": "failed",
                "authorityFeedbackEnabled": False,
                "keyCallbackRegistered": False,
                "lastError": f"viewer status unavailable: {type(exc).__name__}: {exc}"[:512],
            }

    def _dynamics_evidence(self) -> dict[str, object]:
        provider = self._dynamics_evidence_provider
        if provider is None:
            return {
                "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
                "status": "starting",
                "diagnosticOnly": True,
                "authorityFeedbackEnabled": False,
            }
        try:
            return dict(provider())
        except Exception as exc:
            return {
                "schemaVersion": "h-ide.mujoco-dynamics-evidence/1",
                "status": "failed",
                "diagnosticOnly": True,
                "authorityFeedbackEnabled": False,
                "lastError": (
                    "dynamics evidence unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )[:512],
            }

    def _control_synchronization(self) -> dict[str, object]:
        provider = self._control_synchronization_provider
        if provider is None:
            return {
                "schemaVersion": "h-ide.mujoco-control-synchronization/1",
                "active": False,
                "status": "unconfigured",
            }
        try:
            return dict(provider())
        except Exception as exc:
            return {
                "schemaVersion": "h-ide.mujoco-control-synchronization/1",
                "active": False,
                "status": "failed",
                "lastError": (
                    "control synchronization evidence unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )[:512],
            }

    def _observation_status(
        self,
        frame: RenderedFrame | None,
        *,
        now_monotonic_ns: int | None = None,
    ) -> tuple[str, float | None, dict[str, object]]:
        worker = self._worker_status()
        age_ns = None
        age_ms = None
        if frame is not None:
            age_ns = (
                (
                    time.monotonic_ns()
                    if now_monotonic_ns is None
                    else int(now_monotonic_ns)
                )
                - frame.captured_at_monotonic_ns
            )
            age_ms = age_ns / 1_000_000.0
            if age_ns < 0:
                return "failed", age_ms, worker
        if worker.get("status") == "failed":
            status = "failed"
        elif worker.get("status") != "running" or frame is None:
            status = "starting"
        else:
            assert age_ns is not None
            status = "stale" if age_ns > MAX_CAPTURE_AGE_NS else "running"
            return status, age_ms, worker
        return status, age_ms, worker

    def state(self, frame: RenderedFrame | None = None) -> dict[str, object]:
        if frame is None:
            frame = self._window.latest
        support_state = self.support.state()
        observation_status, capture_age_ms, worker_status = self._observation_status(frame)
        physics_status = "failed" if support_state["fault"] is not None else "running"
        status = (
            "failed"
            if physics_status == "failed"
            else ("running" if observation_status == "running" else "degraded")
        )
        pose = frame.pose if frame is not None else (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        camera_pose = (
            frame.camera_pose
            if frame is not None
            else (0.2, 0.0, 1.15, 1.0, 0.0, 0.0, 0.0)
        )
        intrinsics = (
            dict(frame.intrinsics)
            if frame is not None
            else {"fx": 312.77, "fy": 312.77, "cx": 320.0, "cy": 240.0}
        )
        width = frame.width if frame is not None else 640
        height = frame.height if frame is not None else 480
        x, y, z, qw, qx, qy, qz = pose
        camera_x, camera_y, camera_z, camera_qw, camera_qx, camera_qy, camera_qz = camera_pose
        return {
            "status": status,
            "physicsStatus": physics_status,
            "observationStatus": observation_status,
            "schemaVersion": "h-ide.mujoco-capture/2",
            "sessionId": self.session_id,
            "robotId": self.config.robot_id,
            "sourceMode": "synthetic",
            "sourceId": f"mujoco:unitree-g1:{self.config.robot_id}",
            "localizationEpochId": self.config.localization_epoch_id,
            "producerBootId": frame.producer_boot_id if frame is not None else None,
            "sequence": frame.sequence if frame is not None else 0,
            "capturedAtUnixNs": frame.captured_at_unix_ns if frame is not None else 0,
            "capturedAtMonotonicNs": (
                frame.captured_at_monotonic_ns if frame is not None else 0
            ),
            "renderStartedAtUnixNs": frame.render_started_at_unix_ns if frame is not None else 0,
            "renderStartedAtMonotonicNs": (
                frame.render_started_at_monotonic_ns if frame is not None else 0
            ),
            "renderedAtUnixNs": frame.rendered_at_unix_ns if frame is not None else 0,
            "renderedAtMonotonicNs": (
                frame.rendered_at_monotonic_ns if frame is not None else 0
            ),
            "simulationTimeS": frame.simulation_time_s if frame is not None else None,
            "captureAgeMs": capture_age_ms,
            "renderLatencyMs": (
                (
                    frame.rendered_at_monotonic_ns
                    - frame.captured_at_monotonic_ns
                )
                / 1_000_000.0
                if frame is not None
                else None
            ),
            "pose": {"x": x, "y": y, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz},
            "rootVelocity": (
                frame.root_velocity.capture_state()
                if frame is not None
                else BodyTwist.zero().capture_state()
            ),
            "support": support_state,
            "dynamicsEvidence": self._dynamics_evidence(),
            "controlSynchronization": self._control_synchronization(),
            "runtimeTiming": (
                self.runtime_diagnostics.snapshot()
                if self.runtime_diagnostics is not None
                else None
            ),
            "renderTiming": dict(frame.render_timing) if frame is not None else None,
            "observationWorker": worker_status,
            "viewerWorker": self._viewer_status(),
            "cameraPose": {
                "x": camera_x, "y": camera_y, "z": camera_z,
                "qw": camera_qw, "qx": camera_qx, "qy": camera_qy, "qz": camera_qz,
            },
            "intrinsics": intrinsics,
            "rgb": {"width": width, "height": height, "encoding": "jpeg"},
            "depth": {
                "width": width,
                "height": height,
                "encoding": "z16_le",
                "scaleMetersPerUnit": 0.001,
            },
        }

    def _image_status(self, frame: RenderedFrame) -> str:
        if self.support.physics_fault is not None:
            return "failed"
        status, _age_ms, _worker = self._observation_status(frame)
        return status

    def _select_frame(self, query: str) -> tuple[RenderedFrame | None, bool]:
        window = self._window
        if not query:
            return window.latest, False
        values = parse_qs(query, keep_blank_values=True, strict_parsing=True)
        if set(values) != {"boot", "sequence"} or any(len(items) != 1 for items in values.values()):
            raise ValueError("boot and sequence must be supplied exactly once")
        boot = values["boot"][0]
        try:
            uuid.UUID(boot)
            sequence = int(values["sequence"][0])
        except (ValueError, TypeError) as exc:
            raise ValueError("capture identity is invalid") from exc
        if sequence <= 0:
            raise ValueError("capture sequence must be positive")
        for candidate in (window.latest, window.previous):
            if (
                candidate is not None
                and candidate.producer_boot_id == boot
                and candidate.sequence == sequence
            ):
                return candidate, True
        return None, True

    def serve(
        self,
        host: str,
        port: int,
        *,
        docker_observation_host: str | None = None,
    ) -> None:
        store = self

        def handler_type(*, read_only: bool):
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):  # noqa: N802
                    parsed = urlparse(self.path)
                    frame = None
                    if parsed.path in {"/state", "/rgb.jpg", "/depth.raw"}:
                        try:
                            frame, exact = store._select_frame(parsed.query)
                        except ValueError as exc:
                            self.send_error(400, str(exc))
                            return
                        if exact and frame is None:
                            self.send_error(409, "requested capture is no longer retained")
                            return
                    elif parsed.path == "/healthz" and parsed.query:
                        self.send_error(400, "healthz does not accept a capture identity")
                        return

                    payload = None
                    if parsed.path in {"/healthz", "/state"}:
                        payload = store.state(frame)
                        if parsed.path == "/healthz":
                            payload = {
                                "backend": "mujoco_sonic",
                                **payload,
                            }
                        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                        content_type = "application/json"
                    elif parsed.path == "/rgb.jpg":
                        if frame is None:
                            self.send_error(503, "capture not ready")
                            return
                        if store._image_status(frame) != "running":
                            self.send_error(503, "capture is stale or worker failed")
                            return
                        body = frame.rgb_jpeg
                        content_type = "image/jpeg"
                    elif parsed.path == "/depth.raw":
                        if frame is None:
                            self.send_error(503, "capture not ready")
                            return
                        if store._image_status(frame) != "running":
                            self.send_error(503, "capture is stale or worker failed")
                            return
                        body = frame.depth_z16_le
                        content_type = "application/octet-stream"
                    elif (
                        parsed.path == "/support"
                        and not read_only
                        and self.client_address[0] in {"127.0.0.1", "::1"}
                    ):
                        body = json.dumps(
                            store.support.state(),
                            separators=(",", ":"),
                        ).encode("utf-8")
                        content_type = "application/json"
                    else:
                        self.send_error(404)
                        return
                    if not body:
                        self.send_error(503, "capture not ready")
                        return
                    status_code = 503 if (
                        parsed.path == "/healthz"
                        and payload is not None
                        and payload.get("status") != "running"
                    ) else 200
                    self.send_response(status_code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    if frame is not None:
                        self.send_header(
                            "X-Hide-Capture-Boot-Id",
                            frame.producer_boot_id,
                        )
                        self.send_header(
                            "X-Hide-Capture-Sequence",
                            str(frame.sequence),
                        )
                        self.send_header(
                            "X-Hide-Captured-At-Unix-Ns",
                            str(frame.captured_at_unix_ns),
                        )
                        self.send_header(
                            "X-Hide-Rendered-At-Unix-Ns",
                            str(frame.rendered_at_unix_ns),
                        )
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        # Pollers use short deadlines; abandoning a late response
                        # must not produce a simulator traceback or affect capture.
                        return

                def do_POST(self):  # noqa: N802
                    if read_only:
                        self.send_response(405)
                        self.send_header("Allow", "GET")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    parsed = urlparse(self.path)
                    if parsed.path not in {
                        "/support",
                        "/support/rearm",
                        "/support/fault",
                        "/support/motion-fault",
                        "/support/motion-fault/reset",
                        "/support/fall-arrest/release",
                    }:
                        self.send_error(404)
                        return
                    if self.client_address[0] not in {"127.0.0.1", "::1"}:
                        self.send_error(403, "support control is loopback-only")
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if length <= 0 or length > 1024:
                            raise ValueError("bounded JSON body is required")
                        payload = json.loads(self.rfile.read(length))
                        if not isinstance(payload, dict):
                            raise ValueError("support request must be an object")
                        if parsed.path == "/support":
                            if set(payload) != {"mode"}:
                                raise ValueError(
                                    "support request must contain only mode"
                                )
                            store.support.set_mode(str(payload.get("mode", "")))
                        elif parsed.path == "/support/rearm":
                            if set(payload) != {"generation"}:
                                raise ValueError(
                                    "support rearm request must contain only generation"
                                )
                            store.support.rearm(payload.get("generation"))
                        elif parsed.path == "/support/fault":
                            if set(payload) != {"reason"}:
                                raise ValueError(
                                    "support fault request must contain only reason"
                                )
                            reason = str(payload.get("reason", "")).strip()
                            if not reason or len(reason) > 512:
                                raise ValueError(
                                    "support fault reason must contain 1..512 characters"
                                )
                            store.support.latch_fault(reason)
                        elif parsed.path in {
                            "/support/motion-fault",
                            "/support/motion-fault/reset",
                        }:
                            if set(payload) != {"reason"}:
                                raise ValueError(
                                    "motion fault request must contain only reason"
                                )
                            reason = str(payload.get("reason", "")).strip()
                            if not reason or len(reason) > 512:
                                raise ValueError(
                                    "motion fault reason must contain 1..512 characters"
                                )
                            if parsed.path == "/support/motion-fault":
                                store.support.latch_motion_fault(reason)
                            else:
                                store.support.clear_motion_fault(reason)
                        else:
                            if set(payload) != {"activationEpoch"}:
                                raise ValueError(
                                    "fall-arrest release must contain only activationEpoch"
                                )
                            store.support.request_fall_arrest_release(
                                payload.get("activationEpoch")
                            )
                        body = json.dumps(
                            store.support.state(),
                            separators=(",", ":"),
                        ).encode("utf-8")
                    except (ValueError, json.JSONDecodeError) as exc:
                        self.send_error(422, str(exc))
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)

                def log_message(self, *_args):
                    return

            return Handler

        listeners = [(host, False, "loopback")]
        if docker_observation_host is not None:
            if docker_observation_host == host:
                raise ValueError("Docker observation host must differ from plant loopback")
            listeners.append((docker_observation_host, True, "docker-readonly"))
        servers: list[tuple[ThreadingHTTPServer, str]] = []
        try:
            for bind_host, read_only, label in listeners:
                server = ThreadingHTTPServer(
                    (bind_host, port),
                    handler_type(read_only=read_only),
                )
                servers.append((server, label))
        except Exception:
            for server, _label in servers:
                server.server_close()
            raise
        self._servers = [server for server, _label in servers]
        for server, label in servers:
            threading.Thread(
                target=server.serve_forever,
                daemon=True,
                name=f"mujoco-observation-http-{label}",
            ).start()

    def close(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        self._servers.clear()


def run_plant(config: SimConfig) -> int:
    onscreen_viewer_enabled = operator_viewer_enabled(config)
    defer_onscreen_viewer = operator_viewer_deferred(config)
    repository_root = Path(__file__).resolve().parents[2]
    simulator_root = repository_root / "models" / "unitree-g1-mujoco"
    if not (simulator_root / "sim" / "simulator_factory.py").is_file():
        raise RuntimeError(f"pinned Unitree simulator is missing: {simulator_root}")
    sys.path.insert(0, str(simulator_root))
    from sim.simulator_factory import SimulatorFactory, init_channel

    model_config = {
        "ROBOT_TYPE": "g1_29dof",
        # ``Path / absolute`` in the pinned simulator preserves the verified
        # absolute path loaded from runtime_manifest.json.
        "ROBOT_SCENE": str(config.model_scene),
        # Keep Unitree actuator/state DDS off the ROS 2 Nav2 graph.  Joining a
        # full Nav2 action graph to the old shared domain repeatedly delayed
        # the otherwise healthy LowState/LowCmd pair past its liveness bound.
        "DOMAIN_ID": config.unitree_domain_id,
        "INTERFACE": config.interface,
        "SIMULATOR": "mujoco",
        # The scene already has a floating root; its actuator vector contains
        # only the 43 body/hand motors.
        "FREE_BASE": False,
        "enable_waist": False,
        # Keep the official stabilizing tether enabled during startup.  It is
        # released by the sim profile once the standing controller is verified;
        # without an initial support force a policy handoff from zeroed joints can
        # fall before the first SONIC plan is blended in.
        "ENABLE_ELASTIC_BAND": True,
        "SIMULATE_DT": 0.004,
        "VIEWER_DT": 0.02,
        "IMAGE_DT": 1.0 / config.camera_hz,
        # Renderer and viewer ownership is delegated to independent spawn
        # processes below.  The authority must never create a GL context.
        "ENABLE_ONSCREEN": False,
        "ENABLE_OFFSCREEN": False,
        "MP_START_METHOD": "spawn",
        "USE_SENSOR": False,
        "PRINT_SCENE_INFORMATION": False,
        # Never hide a fall/numerical fault behind an in-process teleport.
        # A new spawn is permitted only by explicitly restarting the plant.
        "RESET_POLICY": config.reset_policy,
        "NUM_MOTORS": 29,
        "NUM_JOINTS": 29,
        # SONIC is the 29-DoF body controller.  The hand-containing scene is
        # explicitly marked unstable by the upstream MuJoCo model, so the
        # simulation uses the supported no-hand scene for navigation validation.
        "NUM_HAND_MOTORS": 0,
        "NUM_HAND_JOINTS": 0,
        "motor_effort_limit_list": [88, 88, 88, 139, 50, 50] * 2
        + [88, 50, 50]
        + [25, 25, 25, 25, 25, 5, 5] * 2,
    }
    init_channel(model_config)
    simulator = SimulatorFactory.create_simulator(
        config=model_config,
        env_name="default",
        onscreen=False,
        offscreen=False,
        camera_configs={},
    )

    # The upstream XML supplies the pelvis height but leaves all articulated
    # joints at zero.  Starting a floating humanoid from that pose makes it
    # collapse before a policy has had time to produce its first reference.  Use
    # SONIC's published nominal standing angles as the deterministic initial
    # state; this is still a normal lowstate/lowcmd handoff, not a kinematic
    # teleport during operation.
    import mujoco
    from lerobot.robots.unitree_g1.controllers.sonic_pipeline import DEFAULT_ANGLES_MUJOCO

    sim_env = simulator.sim_env
    # The upstream G1 viewer's elastic band is an operator aid: it applies a
    # very stiff spring toward a fixed world point.  That is useful while
    # manually tuning a standing pose, but it actively fights Nav2's horizontal
    # command in an autonomous workstation demo and can pull a walking robot
    # over.  Keep the band as a simulation-only safety support, limiting it to
    # vertical lift and roll/pitch stabilization.  SONIC still receives the
    # complete /cmd_vel and produces the official lowcmd joint targets; this
    # shim does not write qpos or motor commands.
    support_band = getattr(sim_env, "elastic_band", None)
    support_controller = PlantSupportController(
        fall_arrest_enabled=support_band is not None
    )
    if support_band is not None:
        original_band_advance = support_band.Advance

        def _fall_arrest_support(pose):
            if support_controller.consume_fall_arrest_release(
                lambda: setattr(support_band, "enable", False)
            ) is not None:
                return np.zeros(6, dtype=float)
            # Snapshot the mode before checking the terminal latch.  If an
            # HTTP fault arrives between these two reads, require_healthy()
            # aborts.  If it arrives immediately afterwards, this one in-flight
            # step retains the previously observed support mode rather than
            # applying a newly selected emergency wrench to a dynamic pose.
            support_mode = support_controller.mode
            support_controller.require_healthy()
            return stabilized_support_wrench(
                original_band_advance(pose),
                pose,
                support_mode,
                tilt_kp=250.0,
                tilt_kd=30.0,
            )

        support_band.Advance = _fall_arrest_support
    else:
        support_controller.finish_startup()

    unitree_bridge = sim_env.unitree_bridge
    control_synchronizer = PlantControlSynchronizer(
        support_controller,
        control_hz=config.control_hz,
        command_identity=unitree_bridge.LowCmdIdentity,
        wait_for_command_for_tick=unitree_bridge.WaitForLowCmdForTick,
    )
    original_publish_lowstate = unitree_bridge.PublishLowState

    def publish_synchronized_lowstate(observation):
        # Snapshot before publish so a fast DDS response cannot race past the
        # waiter.  Once support is released there is exactly one controller
        # write per admitted 50 Hz LowState; an asynchronous stop write after
        # this boundary is also a valid fresh actuator decision.
        command_identity = control_synchronizer.command_identity_before_publish()
        result = original_publish_lowstate(observation)
        control_synchronizer.wait_after_lowstate(
            tick_ms=int(unitree_bridge.low_state.tick),
            command_identity_before_publish=command_identity,
        )
        return result

    unitree_bridge.PublishLowState = publish_synchronized_lowstate

    apply_process_start_spawn(
        sim_env.mj_data,
        sim_env.mj_model,
        config.runtime_scene,
    )
    # ``body_joint_index`` follows the XML/Unitree lowstate deploy order
    # (left leg, right leg, waist, arms). Initialize from the same official
    # grouped standing pose used by the Unitree command interface. SONIC remains
    # responsible for all subsequent lowcmd updates.
    if len(sim_env.body_joint_index) != len(DEFAULT_ANGLES_MUJOCO):
        raise RuntimeError(
            f"G1 joint layout mismatch: model={len(sim_env.body_joint_index)} "
            f"nominal={len(DEFAULT_ANGLES_MUJOCO)}"
        )
    for i, joint_id in enumerate(sim_env.body_joint_index):
        sim_env.mj_data.qpos[int(joint_id) + 6] = float(DEFAULT_ANGLES_MUJOCO[i])
    sim_env.mj_data.qvel[:] = 0.0
    mujoco.mj_forward(sim_env.mj_model, sim_env.mj_data)
    if support_band is not None:
        # Upstream defaults target world z=1 with very stiff gains, which
        # visibly hoists the pelvis and makes contacting feet propel the robot
        # even while Nav2 is idle.  Anchor the simulation-only fall arrest at
        # the actual initialized body height and use compliant gains.  The
        # wrapper above still zeros every horizontal force and yaw torque.
        attached = int(sim_env.band_attached_link)
        support_band.point = np.asarray(
            [
                config.runtime_scene.spawn_x,
                config.runtime_scene.spawn_y,
                float(sim_env.mj_data.xpos[attached][2]),
            ],
            dtype=float,
        )
        support_band.length = 0.0
        # A fixed joint target is not itself a humanoid balance controller.
        # Preserve the upstream Unitree support stiffness only during the cold
        # start.  After the bounded handoff, the wrapper above removes every
        # component except vertical fall arrest, so it cannot mask or fight
        # SONIC's learned attitude control.
        support_band.kp_pos = 10000.0
        support_band.kd_pos = 1000.0
        support_band.kp_ang = 1000.0
        support_band.kd_ang = 50.0

    horizontal_support_release = None
    if support_band is not None and config.startup_horizontal_support_s > 0:
        def release_horizontal_support_after_control_handoff():
            bridge = getattr(sim_env, "unitree_bridge", None)
            while bridge is not None and not bridge.low_cmd_received:
                time.sleep(0.05)
            if bridge is None:
                return
            time.sleep(config.startup_horizontal_support_s)
            support_controller.finish_startup()

        horizontal_support_release = threading.Thread(
            target=release_horizontal_support_after_control_handoff,
            daemon=True,
            name="mujoco-horizontal-startup-hold-release",
        )
        horizontal_support_release.start()
    else:
        support_controller.finish_startup()

    # The legacy standalone Plant timer is mutually exclusive with SONIC's
    # exact-epoch policy handoff.  Product motion uses the latter: only the
    # actuator owner may release fall arrest after a successful learned-policy
    # DDS target.  The timer remains available solely for isolated Plant demos
    # which explicitly disable policy-owned release.
    support_release = None
    if (
        config.startup_support_s > 0
        and support_band is not None
        and not config.policy_release_fall_arrest
    ):
        def release_support_after_control_handoff():
            # Model loading can take longer than the configured stand-up ramp.
            # Start that ramp from the first valid lowcmd, not from plant launch,
            # so the robot is never dropped before SONIC has taken control.
            bridge = getattr(sim_env, "unitree_bridge", None)
            while bridge is not None and not bridge.low_cmd_received:
                time.sleep(0.05)
            if bridge is None:
                return
            time.sleep(config.startup_support_s)
            try:
                support_controller.request_fall_arrest_release(0)
            except ValueError:
                # An explicit positive activation epoch may have already
                # claimed or completed the one-way release. The legacy timer
                # must never overwrite that identity or emit a daemon-thread
                # traceback; any other rejection remains a real error.
                release_state = support_controller.state()
                if (
                    release_state["fallArrestEnabled"] is False
                    or release_state["fallArrestReleaseRequestedEpoch"] not in {
                        None,
                        0,
                    }
                ):
                    return
                raise

        support_release = threading.Thread(
            target=release_support_after_control_handoff,
            daemon=True,
            name="mujoco-startup-support-release",
        )
        support_release.start()
    frames: PlantFrameStore | None = None
    rgbd_worker: RgbdRenderWorker | None = None
    viewer_worker: PassiveViewerWorker | None = None
    viewer_lifecycle_lock = threading.Lock()
    viewer_start_event: threading.Event | None = None
    viewer_shutdown_event: threading.Event | None = None
    viewer_start_thread: threading.Thread | None = None
    previous_sigusr1_handler = None
    runtime_diagnostics = PlantRuntimeDiagnostics(
        stall_threshold_s=config.plant_lowstate_stale_s,
    )
    dynamics_evidence = PlantDynamicsEvidence(
        mujoco,
        sim_env.mj_model,
        sim_env.mj_data,
        support_band=support_band,
        attached_body_id=(
            int(sim_env.band_attached_link)
            if support_band is not None
            else None
        ),
    )
    dynamics_evidence.observe()
    physics_monitor = PlantPhysicsMonitor(
        config,
        mujoco,
        sim_env.mj_model,
        sim_env.mj_data,
        support_controller,
    )
    state_shape = MujocoStateShape.from_model(mujoco, sim_env.mj_model)
    producer_boot_id = str(uuid.uuid4())
    original_sim_step = simulator.sim_env.sim_step

    def timed_sim_step():
        started = time.monotonic()
        runtime_diagnostics.observe_loop_start(
            started,
            unix_ns=time.time_ns(),
        )
        try:
            # Only numerical/physics corruption terminates this authority
            # process. Recoverable motion faults keep stepping a frozen DDS
            # hold so both viewers and post-incident evidence remain alive.
            # The support callback repeats this physics-health check at the
            # last safe point before mj_step.
            support_controller.require_healthy()
            result = original_sim_step()
            # Capture the exact post-step evidence before the terminal monitor
            # can raise and close the process.  HTTP reads only the resulting
            # immutable projection; this data is never fed back into control.
            dynamics_evidence.observe()
            physics_monitor.observe()
            return result
        finally:
            runtime_diagnostics.observe_stage(
                "simStep",
                time.monotonic() - started,
                unix_ns=time.time_ns(),
            )

    simulator.sim_env.sim_step = timed_sim_step
    try:
        if config.offscreen:
            frames = PlantFrameStore(
                config,
                support_controller,
                runtime_diagnostics,
                dynamics_evidence_provider=dynamics_evidence.capture_state,
                control_synchronization_provider=control_synchronizer.snapshot,
            )
            rgbd_mailbox = LatestStateMailbox(
                state_shape,
                producer_boot_id=producer_boot_id,
            )
            rgbd_worker = RgbdRenderWorker(
                mailbox=rgbd_mailbox,
                shape=state_shape,
                config=RenderWorkerConfig(
                    runtime_manifest=str(config.scene_runtime_manifest),
                    model_scene=str(config.model_scene),
                    reset_policy=config.reset_policy,
                    jpeg_quality=config.observation_jpeg_quality,
                ),
                on_frame=frames.commit,
            )
            frames.set_worker_status_provider(rgbd_worker.status)
            rgbd_worker.start()
            frames.serve(
                config.plant_bind_host,
                config.plant_port,
                docker_observation_host=config.docker_observation_bind_host,
            )

            def publish_rgbd_snapshot():
                started = time.monotonic()
                try:
                    rgbd_mailbox.try_publish(
                        mujoco,
                        sim_env.mj_model,
                        sim_env.mj_data,
                        captured_at_unix_ns=time.time_ns(),
                    )
                    return {}
                finally:
                    runtime_diagnostics.observe_stage(
                        "snapshotPublish",
                        time.monotonic() - started,
                        unix_ns=time.time_ns(),
                    )

            simulator.sim_env.update_render_caches = publish_rgbd_snapshot

        if onscreen_viewer_enabled:
            viewer_mailbox = LatestStateMailbox(
                state_shape,
                producer_boot_id=producer_boot_id,
            )
            viewer_worker = PassiveViewerWorker(
                mailbox=viewer_mailbox,
                shape=state_shape,
                config=ViewerWorkerConfig(
                    runtime_manifest=str(config.scene_runtime_manifest),
                    model_scene=str(config.model_scene),
                    reset_policy=config.reset_policy,
                    spawn_x=config.runtime_scene.spawn_x,
                    spawn_y=config.runtime_scene.spawn_y,
                ),
            )

            def viewer_status():
                with viewer_lifecycle_lock:
                    return viewer_worker.status()

            if defer_onscreen_viewer:
                viewer_start_event = threading.Event()
                viewer_shutdown_event = threading.Event()

                def request_viewer_start(_signum, _frame):
                    viewer_start_event.set()

                def start_viewer_after_request():
                    viewer_start_event.wait()
                    if viewer_shutdown_event.is_set():
                        return
                    with viewer_lifecycle_lock:
                        if not viewer_shutdown_event.is_set():
                            viewer_worker.start()

                previous_sigusr1_handler = signal.getsignal(signal.SIGUSR1)
                signal.signal(signal.SIGUSR1, request_viewer_start)
                viewer_start_thread = threading.Thread(
                    target=start_viewer_after_request,
                    daemon=True,
                    name="mujoco-native-viewer-starter",
                )
                viewer_start_thread.start()
            else:
                with viewer_lifecycle_lock:
                    viewer_worker.start()
            if frames is not None:
                frames.set_viewer_status_provider(viewer_status)

            def publish_viewer_snapshot():
                started = time.monotonic()
                try:
                    viewer_mailbox.try_publish(
                        mujoco,
                        sim_env.mj_model,
                        sim_env.mj_data,
                        captured_at_unix_ns=time.time_ns(),
                    )
                finally:
                    runtime_diagnostics.observe_stage(
                        "viewerSnapshotPublish",
                        time.monotonic() - started,
                        unix_ns=time.time_ns(),
                    )

            simulator.sim_env.update_viewer = publish_viewer_snapshot

        simulator.start()
    finally:
        if previous_sigusr1_handler is not None:
            signal.signal(signal.SIGUSR1, previous_sigusr1_handler)
        if viewer_shutdown_event is not None:
            viewer_shutdown_event.set()
        if viewer_start_event is not None:
            viewer_start_event.set()
        if viewer_start_thread is not None:
            viewer_start_thread.join(timeout=1.0)
        try:
            if frames is not None:
                frames.close()
        finally:
            try:
                if rgbd_worker is not None:
                    rgbd_worker.close()
            finally:
                try:
                    if viewer_worker is not None:
                        with viewer_lifecycle_lock:
                            viewer_worker.close()
                finally:
                    simulator.close()
    return 0


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    config = SimConfig.from_file(args.config)
    os.environ.setdefault("ROS_DOMAIN_ID", str(config.ros_domain_id))
    return run_plant(config)


if __name__ == "__main__":
    raise SystemExit(main())
