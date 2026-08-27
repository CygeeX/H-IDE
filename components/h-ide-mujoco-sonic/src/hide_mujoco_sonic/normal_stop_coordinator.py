"""ROS-independent orchestration for one correlated simulation safe stop.

The ROS service callback is allowed to wait, but it must never call the
single-threaded :class:`~hide_mujoco_sonic.gait_adapter.GaitAdapter` directly.
This coordinator therefore splits a stop into two execution domains:

* :meth:`request_stop` runs in the normal-stop service callback.  It registers
  (or joins) one attempt and performs the blocking confirmation stages without
  retaining the coordinator lock around any I/O.
* :meth:`pump` and :meth:`observe_stationary_intent` run in the bridge's
  mutually-exclusive state callback group.  The pump is the only path that
  requests a GaitAdapter stop; odometry merely contributes the already-proved
  fresh stationary intent and wakes the waiter.

Terminal faults are deliberately outside the normal-stop state machine.  Once
``notify_terminal`` is called, waiters wake and no later normal acknowledgement
is sent.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time
from typing import Callable, Mapping
from uuid import UUID, uuid4

from .motion_fault import SonicHealth, require_fresh_motion_health
from .normal_stop import NormalStopIdentity
from .normal_stop_wire import (
    NormalStopDeadline,
    NormalStopResponse,
    normal_stop_begin_request,
    normal_stop_plant_request,
    normal_stop_stationary_request,
    parse_normal_stop_response,
)


_MAX_STAGE_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class NormalStopResult:
    """Outcome returned to one or more joined Trigger callbacks."""

    success: bool
    message: str
    t0_s: float
    control_generation: int
    identity: NormalStopIdentity | None = None
    plant_safety_stop_epoch: int | None = None
    terminal: bool = False


@dataclass(slots=True)
class _Attempt:
    t0_s: float
    deadline: NormalStopDeadline
    generation: int
    reason: str
    begin_payload: dict[str, object] | None = None
    begin_response: NormalStopResponse | None = None
    stationary_payload: dict[str, object] | None = None
    stationary_response: NormalStopResponse | None = None
    driver_boot_id: UUID | None = None
    identity: NormalStopIdentity | None = None
    plant_epoch: int | None = None
    pump_pending: bool = True
    pump_running: bool = False
    pump_complete: bool = False
    pump_error: str | None = None
    result: NormalStopResult | None = None


class BridgeNormalStopCoordinator:
    """Coordinate strict Driver, measured gait, Plant, and health evidence.

    The injected callbacks intentionally expose only narrow operations.  The
    class imports no ROS packages and is fully exercisable in unit tests.
    """

    def __init__(
        self,
        *,
        robot_id: str,
        bridge_boot_id: UUID,
        generation: Callable[[], int],
        request_gait_stop: Callable[[int, float], Mapping[str, object]],
        request_driver: Callable[[Mapping[str, object], float], Mapping[str, object]],
        stop_plant: Callable[[float], Mapping[str, object]],
        probe_health: Callable[[UUID, float], SonicHealth],
        wake_state_pump: Callable[[], None],
        terminal_reason: Callable[[], str | None],
        plant_lowstate_stale_s: float,
        sonic_control_step_stale_s: float,
        clock: Callable[[], float] = time.monotonic,
        request_id_factory: Callable[[], UUID] = uuid4,
        driver_timeout_s: float = 0.30,
        plant_timeout_s: float = 0.30,
        health_timeout_s: float = 0.30,
    ) -> None:
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_id must be non-empty")
        if not isinstance(bridge_boot_id, UUID):
            raise ValueError("bridge_boot_id must be a UUID")
        callbacks = (
            generation,
            request_gait_stop,
            request_driver,
            stop_plant,
            probe_health,
            wake_state_pump,
            terminal_reason,
            clock,
            request_id_factory,
        )
        if not all(callable(callback) for callback in callbacks):
            raise TypeError("normal-stop callbacks must be callable")
        for value, name in (
            (plant_lowstate_stale_s, "plant_lowstate_stale_s"),
            (sonic_control_step_stale_s, "sonic_control_step_stale_s"),
            (driver_timeout_s, "driver_timeout_s"),
            (plant_timeout_s, "plant_timeout_s"),
            (health_timeout_s, "health_timeout_s"),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")

        self._robot_id = robot_id
        self._bridge_boot_id = bridge_boot_id
        self._generation = generation
        self._request_gait_stop = request_gait_stop
        self._request_driver = request_driver
        self._stop_plant = stop_plant
        self._probe_health = probe_health
        self._wake_state_pump = wake_state_pump
        self._terminal_reason = terminal_reason
        self._plant_lowstate_stale_s = float(plant_lowstate_stale_s)
        self._sonic_control_step_stale_s = float(
            sonic_control_step_stale_s
        )
        self._clock = clock
        self._request_id_factory = request_id_factory
        self._driver_timeout_s = float(driver_timeout_s)
        self._plant_timeout_s = float(plant_timeout_s)
        self._health_timeout_s = float(health_timeout_s)

        self._condition = threading.Condition(threading.RLock())
        self._active: _Attempt | None = None
        self._resume: _Attempt | None = None
        self._confirmed: _Attempt | None = None
        self._last_t0_s: float | None = None
        self._terminal_override: str | None = None
        self._closed_reason: str | None = None
        self._control_arm_active = False

    @property
    def active(self) -> bool:
        with self._condition:
            return self._active is not None and self._active.result is None

    def blocks_generic_intent(self, generation: int | None) -> bool:
        """Whether ordinary gait envelopes must stay behind the strict path."""

        if generation is None:
            return False
        with self._condition:
            candidates = (self._active, self._resume, self._confirmed)
            return any(
                attempt is not None and attempt.generation == generation
                for attempt in candidates
            )

    def notify_terminal(self, reason: str) -> None:
        """Wake every waiter after the separate terminal path has claimed."""

        detail = str(reason).strip()[:512] or "terminal motion fault"
        with self._condition:
            if self._terminal_override is None:
                self._terminal_override = detail
            self._condition.notify_all()

    def shutdown(self, reason: str = "ROS bridge is shutting down") -> None:
        detail = str(reason).strip()[:512] or "ROS bridge is shutting down"
        with self._condition:
            if self._closed_reason is None:
                self._closed_reason = detail
            self._condition.notify_all()

    def reserve_control_arm(self) -> str | None:
        """Linearize control acquisition against Trigger registration.

        ``None`` means the state-group callback owns the short reservation
        until :meth:`release_control_arm`.  A text result means normal stop or
        terminal shutdown won and control acquisition must be rejected.
        """

        with self._condition:
            terminal = self._terminal_locked()
            if terminal is not None:
                return f"terminal stop path is active: {terminal}"
            if self._control_arm_active:
                return "another control acquisition is already active"
            active = self._active
            if active is not None and active.result is None:
                return "correlated normal stop is already active"
            self._control_arm_active = True
            return None

    def release_control_arm(self) -> None:
        """Release a reservation after the state-group callback returns."""

        with self._condition:
            if not self._control_arm_active:
                raise RuntimeError("control-arm reservation is not active")
            self._control_arm_active = False
            self._condition.notify_all()

    def request_stop(self, reason: str) -> NormalStopResult:
        """Register or join one stop and wait for the correlated proof chain."""

        bounded_reason = str(reason).strip()[:512]
        if not bounded_reason:
            raise ValueError("normal-stop reason must be non-empty")
        owner = False
        with self._condition:
            entry_t0_s = self._next_t0_locked()
            entry_deadline = NormalStopDeadline.start(entry_t0_s)
            while self._control_arm_active:
                terminal = self._terminal_locked()
                if terminal is not None:
                    generation = self._read_generation()
                    return self._terminal_result(
                        t0_s=entry_t0_s,
                        generation=generation,
                        reason=terminal,
                    )
                remaining = (
                    entry_deadline.transaction_deadline_s - self._now()
                )
                if remaining <= 0.0:
                    return NormalStopResult(
                        success=False,
                        message=(
                            "control acquisition did not release before the "
                            "normal-stop 3.5s deadline"
                        ),
                        t0_s=entry_t0_s,
                        control_generation=self._read_generation(),
                    )
                self._condition.wait(timeout=min(remaining, 0.05))
            # Generation selection and active-attempt publication share the
            # same boundary observed by reserve_control_arm().  Whichever path
            # wins cannot be overtaken between its check and mutation.
            generation = self._read_generation()
            terminal = self._terminal_locked()
            if terminal is not None:
                return self._terminal_result(
                    t0_s=entry_t0_s,
                    generation=generation,
                    reason=terminal,
                )

            active = self._active
            if (
                active is not None
                and active.result is None
                and active.generation == generation
            ):
                attempt = active
            else:
                if entry_deadline.gait_expired(self._now()):
                    return NormalStopResult(
                        success=False,
                        message=(
                            "control acquisition consumed the 2.5s normal-stop "
                            "gait deadline"
                        ),
                        t0_s=entry_t0_s,
                        control_generation=generation,
                    )
                source = self._retry_source_locked(generation)
                attempt = _Attempt(
                    t0_s=entry_t0_s,
                    deadline=entry_deadline,
                    generation=generation,
                    reason=bounded_reason,
                )
                if source is not None:
                    attempt.begin_payload = deepcopy(source.begin_payload)
                    attempt.stationary_payload = deepcopy(
                        source.stationary_payload
                    )
                    attempt.driver_boot_id = source.driver_boot_id
                    attempt.identity = source.identity
                    attempt.plant_epoch = source.plant_epoch
                self._active = attempt
                owner = True

        if owner:
            # Triggering the state callback is not I/O and occurs with no
            # coordinator lock retained.  Some unit tests intentionally invoke
            # the pump synchronously from this callback.
            try:
                self._wake_state_pump()
            except Exception as exc:
                self._set_pump_error(
                    attempt,
                    f"normal-stop state pump could not be triggered: {exc}",
                )
            result = self._run_owner(attempt)
            self._finish(attempt, result)

        return self._wait_result(attempt)

    def pump(self) -> bool:
        """Run the GaitAdapter/strict-begin boundary in the state callback group."""

        with self._condition:
            attempt = self._active
            if (
                attempt is None
                or attempt.result is not None
                or not attempt.pump_pending
                or attempt.pump_running
            ):
                return False
            attempt.pump_pending = False
            attempt.pump_running = True

        try:
            self._raise_if_terminal()
            now = self._now()
            intent = dict(
                self._request_gait_stop(attempt.generation, now)
            )
            self._raise_if_terminal()
            if intent.get("state") == "FAULT":
                raise RuntimeError(
                    "GaitAdapter entered terminal FAULT during normal stop"
                )

            payload = attempt.begin_payload
            if payload is None:
                self._require_begin_intent(intent, attempt.generation)
                request_id = self._new_request_id()
                payload = normal_stop_begin_request(
                    request_id=request_id,
                    robot_id=self._robot_id,
                    bridge_boot_id=self._bridge_boot_id,
                    reason=attempt.reason,
                    intent=intent,
                )
                attempt.begin_payload = deepcopy(payload)
            else:
                request_id = UUID(str(payload["request_id"]))

            response_payload = self._call_driver(
                payload,
                attempt.deadline.gait_deadline_s,
            )
            expected_identity = attempt.identity
            response = parse_normal_stop_response(
                response_payload,
                expected_request_id=request_id,
                expected_robot_id=self._robot_id,
                expected_driver_boot_id=attempt.driver_boot_id,
                expected_identity=expected_identity,
                expected_control_generation=attempt.generation,
                expected_gait_source_sequence=int(payload["intent"]["sequence"]),  # type: ignore[index]
                expected_phases=frozenset({"BRAKING"}),
            )
            identity = response.normal_stop.identity
            if identity is None:
                raise RuntimeError("normal-stop begin omitted its identity")
            if attempt.identity is not None and identity != attempt.identity:
                raise RuntimeError("normal-stop begin replay changed identity")
            if (
                attempt.driver_boot_id is not None
                and response.driver_boot_id != attempt.driver_boot_id
            ):
                raise RuntimeError("normal-stop begin replay changed driver boot")

            with self._condition:
                if self._active is not attempt:
                    return False
                attempt.begin_response = response
                attempt.driver_boot_id = response.driver_boot_id
                attempt.identity = identity
                attempt.pump_complete = True
                attempt.pump_running = False
                self._condition.notify_all()
            return True
        except Exception as exc:
            self._set_pump_error(attempt, str(exc))
            return False

    def observe_stationary_intent(
        self,
        intent: Mapping[str, object],
    ) -> bool:
        """Record one GaitAdapter-proved post-request stationary envelope.

        This method is intentionally side-effect free beyond local immutable
        evidence and a condition notification.  In particular, odometry never
        performs network or Plant I/O.
        """

        copied = deepcopy(dict(intent))
        with self._condition:
            attempt = self._active
            if (
                attempt is None
                or attempt.result is not None
                or not attempt.pump_complete
                or attempt.identity is None
                or attempt.driver_boot_id is None
                or attempt.stationary_payload is not None
            ):
                return False
            if not self._is_stationary_intent(
                copied,
                generation=attempt.generation,
                after_sequence=attempt.identity.gait_source_sequence,
            ):
                return False
            payload = normal_stop_stationary_request(
                request_id=self._new_request_id(),
                robot_id=self._robot_id,
                bridge_boot_id=self._bridge_boot_id,
                driver_boot_id=attempt.driver_boot_id,
                identity=attempt.identity,
                intent=copied,
            )
            attempt.stationary_payload = payload
            self._condition.notify_all()
            return True

    def _run_owner(self, attempt: _Attempt) -> NormalStopResult:
        try:
            self._wait_for_pump(attempt)
            self._raise_if_terminal()

            stationary_payload = self._wait_for_stationary(attempt)
            identity = attempt.identity
            driver_boot_id = attempt.driver_boot_id
            if identity is None or driver_boot_id is None:
                raise RuntimeError("normal-stop begin correlation is incomplete")
            stationary_request_id = UUID(
                str(stationary_payload["request_id"])
            )
            stationary_raw = self._call_driver(
                stationary_payload,
                attempt.deadline.gait_deadline_s,
            )
            stationary_response = parse_normal_stop_response(
                stationary_raw,
                expected_request_id=stationary_request_id,
                expected_robot_id=self._robot_id,
                expected_driver_boot_id=driver_boot_id,
                expected_identity=identity,
                expected_control_generation=attempt.generation,
                expected_gait_source_sequence=identity.gait_source_sequence,
                expected_phases=frozenset({"STATIONARY_ACK"}),
            )
            attempt.stationary_response = stationary_response
            self._raise_if_terminal()

            try:
                plant_state = self._call_plant(
                    attempt.deadline.transaction_deadline_s
                )
                plant_evidence = self._successful_plant_evidence(plant_state)
            except Exception as exc:
                terminal = self._terminal()
                if terminal is not None:
                    return self._terminal_result(
                        t0_s=attempt.t0_s,
                        generation=attempt.generation,
                        reason=terminal,
                        identity=identity,
                    )
                message = f"Plant safety-stop confirmation failed: {exc}"[:512]
                self._send_failed_plant_evidence(
                    attempt,
                    identity=identity,
                    driver_boot_id=driver_boot_id,
                    error=message,
                )
                return NormalStopResult(
                    success=False,
                    message=message,
                    t0_s=attempt.t0_s,
                    control_generation=attempt.generation,
                    identity=identity,
                )

            self._raise_if_terminal()
            plant_request_id = self._new_request_id()
            plant_payload = normal_stop_plant_request(
                request_id=plant_request_id,
                robot_id=self._robot_id,
                bridge_boot_id=self._bridge_boot_id,
                driver_boot_id=driver_boot_id,
                identity=identity,
                plant=plant_evidence,
            )
            plant_raw = self._call_driver(
                plant_payload,
                attempt.deadline.transaction_deadline_s,
            )
            plant_response = parse_normal_stop_response(
                plant_raw,
                expected_request_id=plant_request_id,
                expected_robot_id=self._robot_id,
                expected_driver_boot_id=driver_boot_id,
                expected_identity=identity,
                expected_control_generation=attempt.generation,
                expected_gait_source_sequence=identity.gait_source_sequence,
                expected_phases=frozenset({"DRIVER_CONFIRMED"}),
            )
            plant_epoch = plant_response.plant_safety_stop_epoch
            if plant_epoch != plant_evidence["safety_stop_epoch"]:
                raise RuntimeError(
                    "Driver and Plant safety-stop epochs do not match"
                )
            attempt.plant_epoch = plant_epoch
            self._raise_if_terminal()

            health = self._call_health(
                self._new_request_id(),
                attempt.deadline.transaction_deadline_s,
            )
            if health.driver_boot_id != driver_boot_id:
                raise RuntimeError("fresh health belongs to another Driver boot")
            require_fresh_motion_health(
                health,
                plant_lowstate_stale_s=self._plant_lowstate_stale_s,
                sonic_control_step_stale_s=self._sonic_control_step_stale_s,
            )
            self._raise_if_terminal()

            # Fresh health v1 proves the current physical hold and Driver boot,
            # but does not carry the normal-stop identity.  Re-read the exact
            # transaction after health with a new request ID.  The Driver's
            # new-ID path revalidates both the current local hold and the same
            # Plant epoch, closing a new-bridge/new-generation or rearm drift
            # window without trusting the earlier cached response.
            final_request_id = self._new_request_id()
            final_payload = normal_stop_plant_request(
                request_id=final_request_id,
                robot_id=self._robot_id,
                bridge_boot_id=self._bridge_boot_id,
                driver_boot_id=driver_boot_id,
                identity=identity,
                plant=plant_evidence,
            )
            final_raw = self._call_driver(
                final_payload,
                attempt.deadline.transaction_deadline_s,
            )
            final_response = parse_normal_stop_response(
                final_raw,
                expected_request_id=final_request_id,
                expected_robot_id=self._robot_id,
                expected_driver_boot_id=driver_boot_id,
                expected_identity=identity,
                expected_control_generation=attempt.generation,
                expected_gait_source_sequence=identity.gait_source_sequence,
                expected_phases=frozenset({"DRIVER_CONFIRMED"}),
            )
            if final_response.plant_safety_stop_epoch != plant_epoch:
                raise RuntimeError(
                    "final Driver revalidation changed the Plant stop epoch"
                )
            self._raise_if_terminal()
            return NormalStopResult(
                success=True,
                message=(
                    "correlated simulation safe-stop confirmed "
                    f"(epoch {plant_epoch})"
                ),
                t0_s=attempt.t0_s,
                control_generation=attempt.generation,
                identity=identity,
                plant_safety_stop_epoch=plant_epoch,
            )
        except Exception as exc:
            terminal = self._terminal()
            if terminal is not None:
                return self._terminal_result(
                    t0_s=attempt.t0_s,
                    generation=attempt.generation,
                    reason=terminal,
                    identity=attempt.identity,
                )
            return NormalStopResult(
                success=False,
                message=f"correlated simulation safe-stop failed: {exc}"[:512],
                t0_s=attempt.t0_s,
                control_generation=attempt.generation,
                identity=attempt.identity,
                plant_safety_stop_epoch=attempt.plant_epoch,
            )

    def _wait_for_pump(self, attempt: _Attempt) -> None:
        with self._condition:
            while not attempt.pump_complete and attempt.pump_error is None:
                terminal = self._terminal_locked()
                if terminal is not None:
                    raise RuntimeError(terminal)
                remaining = (
                    attempt.deadline.gait_deadline_s - self._now()
                )
                if remaining <= 0.0:
                    raise TimeoutError("normal-stop begin exceeded gait deadline")
                self._condition.wait(timeout=min(remaining, 0.05))
            if attempt.pump_error is not None:
                raise RuntimeError(attempt.pump_error)

    def _wait_for_stationary(
        self,
        attempt: _Attempt,
    ) -> dict[str, object]:
        with self._condition:
            while attempt.stationary_payload is None:
                terminal = self._terminal_locked()
                if terminal is not None:
                    raise RuntimeError(terminal)
                remaining = (
                    attempt.deadline.gait_deadline_s - self._now()
                )
                if remaining <= 0.0:
                    raise TimeoutError(
                        "fresh stationary gait evidence exceeded 2.5s deadline"
                    )
                self._condition.wait(timeout=min(remaining, 0.05))
            return deepcopy(attempt.stationary_payload)

    def _wait_result(self, attempt: _Attempt) -> NormalStopResult:
        with self._condition:
            while attempt.result is None:
                terminal = self._terminal_locked()
                if terminal is not None and self._active is not attempt:
                    return self._terminal_result(
                        t0_s=attempt.t0_s,
                        generation=attempt.generation,
                        reason=terminal,
                        identity=attempt.identity,
                    )
                remaining = (
                    attempt.deadline.transaction_deadline_s - self._now()
                )
                if remaining <= 0.0:
                    return NormalStopResult(
                        success=False,
                        message="correlated simulation safe-stop exceeded 3.5s deadline",
                        t0_s=attempt.t0_s,
                        control_generation=attempt.generation,
                        identity=attempt.identity,
                    )
                self._condition.wait(timeout=min(remaining, 0.05))
            return attempt.result

    def _finish(
        self,
        attempt: _Attempt,
        result: NormalStopResult,
    ) -> None:
        with self._condition:
            # This lock acquisition is the final result linearization.  A
            # terminal notifier that acquired the same Condition first must
            # replace any prepared non-terminal result; a result published
            # first remains a truthful earlier outcome and later terminal
            # handling proceeds on its separate path.
            if not result.terminal:
                terminal = self._terminal_locked()
                if terminal is not None:
                    result = self._terminal_result(
                        t0_s=attempt.t0_s,
                        generation=attempt.generation,
                        reason=terminal,
                        identity=attempt.identity,
                    )
                elif result.success and attempt.deadline.transaction_expired(
                    self._now()
                ):
                    result = NormalStopResult(
                        success=False,
                        message=(
                            "correlated simulation safe-stop crossed the 3.5s "
                            "deadline before final result publication"
                        ),
                        t0_s=attempt.t0_s,
                        control_generation=attempt.generation,
                        identity=attempt.identity,
                        plant_safety_stop_epoch=attempt.plant_epoch,
                    )
            attempt.result = result
            if self._active is attempt:
                self._active = None
            if result.success:
                self._confirmed = attempt
                self._resume = None
            elif not result.terminal:
                self._resume = attempt
            self._condition.notify_all()

    def _set_pump_error(self, attempt: _Attempt, error: str) -> None:
        with self._condition:
            if attempt.result is not None:
                return
            attempt.pump_error = str(error).strip()[:512] or "state pump failed"
            attempt.pump_running = False
            self._condition.notify_all()

    def _send_failed_plant_evidence(
        self,
        attempt: _Attempt,
        *,
        identity: NormalStopIdentity,
        driver_boot_id: UUID,
        error: str,
    ) -> None:
        # A terminal claim suppresses this normal failure transition.
        self._raise_if_terminal()
        request_id = self._new_request_id()
        payload = normal_stop_plant_request(
            request_id=request_id,
            robot_id=self._robot_id,
            bridge_boot_id=self._bridge_boot_id,
            driver_boot_id=driver_boot_id,
            identity=identity,
            plant={
                "confirmed": False,
                "mode": None,
                "safety_stop_latched": None,
                "safety_stop_epoch": None,
                "last_rearm_generation": None,
                "fault": None,
                "error": error[:512],
            },
        )
        raw = self._call_driver(
            payload,
            attempt.deadline.transaction_deadline_s,
        )
        parse_normal_stop_response(
            raw,
            expected_request_id=request_id,
            expected_robot_id=self._robot_id,
            expected_driver_boot_id=driver_boot_id,
            expected_identity=identity,
            expected_control_generation=attempt.generation,
            expected_gait_source_sequence=identity.gait_source_sequence,
            expected_phases=frozenset({"FAILED"}),
        )

    def _call_driver(
        self,
        payload: Mapping[str, object],
        deadline_s: float,
    ) -> Mapping[str, object]:
        request = deepcopy(dict(payload))
        last_error: Exception | None = None
        for _attempt in range(_MAX_STAGE_ATTEMPTS):
            self._raise_if_terminal()
            timeout = self._remaining_until(
                deadline_s,
                self._driver_timeout_s,
            )
            try:
                response = self._request_driver(
                    deepcopy(request),
                    timeout,
                )
            except Exception as exc:
                last_error = exc
                self._raise_if_terminal()
                continue
            self._raise_if_terminal()
            if self._now() >= deadline_s:
                raise TimeoutError(
                    "normal-stop Driver reply arrived after its deadline"
                )
            return response
        assert last_error is not None
        raise RuntimeError(
            "normal-stop Driver stage failed after bounded retries: "
            f"{last_error}"
        ) from last_error

    def _call_plant(self, deadline_s: float) -> Mapping[str, object]:
        last_error: Exception | None = None
        for _attempt in range(_MAX_STAGE_ATTEMPTS):
            self._raise_if_terminal()
            timeout = self._remaining_until(
                deadline_s,
                self._plant_timeout_s,
            )
            try:
                response = self._stop_plant(timeout)
            except Exception as exc:
                last_error = exc
                self._raise_if_terminal()
                continue
            self._raise_if_terminal()
            if self._now() >= deadline_s:
                raise TimeoutError(
                    "normal-stop Plant reply arrived after its deadline"
                )
            return response
        assert last_error is not None
        raise RuntimeError(
            "normal-stop Plant stage failed after bounded retries: "
            f"{last_error}"
        ) from last_error

    def _call_health(self, request_id: UUID, deadline_s: float) -> SonicHealth:
        last_error: Exception | None = None
        health: SonicHealth | None = None
        for _attempt in range(_MAX_STAGE_ATTEMPTS):
            self._raise_if_terminal()
            timeout = self._remaining_until(
                deadline_s,
                self._health_timeout_s,
            )
            try:
                health = self._probe_health(request_id, timeout)
            except Exception as exc:
                last_error = exc
                self._raise_if_terminal()
                continue
            self._raise_if_terminal()
            if self._now() >= deadline_s:
                raise TimeoutError(
                    "normal-stop health reply arrived after its deadline"
                )
            break
        if health is None:
            assert last_error is not None
            raise RuntimeError(
                "normal-stop health stage failed after bounded retries: "
                f"{last_error}"
            ) from last_error
        if not isinstance(health, SonicHealth):
            raise TypeError("normal-stop health probe did not return SonicHealth")
        if health.terminal_fault:
            self.notify_terminal(
                health.reason or "SONIC reported a terminal fault"
            )
            self._raise_if_terminal()
        return health

    def _remaining_until(self, deadline_s: float, cap_s: float) -> float:
        remaining = min(float(cap_s), float(deadline_s) - self._now())
        if remaining <= 0.0:
            raise TimeoutError("normal-stop phase deadline expired")
        return remaining

    def _terminal(self) -> str | None:
        with self._condition:
            return self._terminal_locked()

    def _terminal_locked(self) -> str | None:
        if self._closed_reason is not None:
            return self._closed_reason
        if self._terminal_override is not None:
            return self._terminal_override
        external = self._terminal_reason()
        if external is None:
            return None
        return str(external).strip()[:512] or "terminal motion fault"

    def _raise_if_terminal(self) -> None:
        terminal = self._terminal()
        if terminal is not None:
            raise RuntimeError(terminal)

    def _read_generation(self) -> int:
        generation = self._generation()
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("active control generation is invalid")
        return generation

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now) or now < 0.0:
            raise RuntimeError("normal-stop monotonic clock is invalid")
        return now

    def _next_t0_locked(self) -> float:
        now = self._now()
        if self._last_t0_s is not None and now <= self._last_t0_s:
            now = math.nextafter(self._last_t0_s, math.inf)
        self._last_t0_s = now
        return now

    def _new_request_id(self) -> UUID:
        request_id = self._request_id_factory()
        if not isinstance(request_id, UUID):
            raise ValueError("request_id_factory must return UUID")
        return request_id

    def _retry_source_locked(self, generation: int) -> _Attempt | None:
        for attempt in (self._confirmed, self._resume):
            if (
                attempt is not None
                and attempt.generation == generation
                and attempt.begin_payload is not None
            ):
                return attempt
        return None

    @staticmethod
    def _require_begin_intent(
        intent: Mapping[str, object],
        generation: int,
    ) -> None:
        if (
            intent.get("type") != "sonic_intent"
            or intent.get("state") != "STOPPING"
            or intent.get("generation") != generation
            or intent.get("motion_enabled") is not False
            or intent.get("stop_requested") is not True
            or intent.get("stop_acknowledged") is not False
        ):
            raise RuntimeError(
                "GaitAdapter did not produce the exact STOPPING begin intent"
            )

    @staticmethod
    def _is_stationary_intent(
        intent: Mapping[str, object],
        *,
        generation: int,
        after_sequence: int,
    ) -> bool:
        sequence = intent.get("sequence")
        return bool(
            intent.get("type") == "sonic_intent"
            and intent.get("state") == "DISARMED"
            and intent.get("generation") == generation
            and isinstance(sequence, int)
            and not isinstance(sequence, bool)
            and sequence > after_sequence
            and intent.get("motion_enabled") is False
            and intent.get("stop_requested") is True
            and intent.get("stop_acknowledged") is True
        )

    @staticmethod
    def _successful_plant_evidence(
        state: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(state, Mapping):
            raise RuntimeError("Plant safety-stop response is not an object")
        epoch = state.get("safetyStopEpoch")
        rearm = state.get("lastRearmGeneration")
        if (
            state.get("mode") != "safety_stop"
            or state.get("safetyStopLatched") is not True
            or state.get("fault") is not None
            or not isinstance(state.get("fallArrestAvailable"), bool)
            or (
                state.get("fallArrestAvailable") is True
                and (
                    state.get("fallArrestEnabled") is not True
                    or state.get("fallArrestReengagedSafetyStopEpoch") != epoch
                )
            )
            or isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 1
            or (
                rearm is not None
                and (
                    isinstance(rearm, bool)
                    or not isinstance(rearm, int)
                    or rearm < 0
                )
            )
        ):
            raise RuntimeError(
                "Plant did not return a correlated non-fault safety-stop latch"
            )
        return {
            "confirmed": True,
            "mode": "safety_stop",
            "safety_stop_latched": True,
            "safety_stop_epoch": epoch,
            "last_rearm_generation": rearm,
            "fault": None,
            "error": None,
        }

    @staticmethod
    def _terminal_result(
        *,
        t0_s: float,
        generation: int,
        reason: str,
        identity: NormalStopIdentity | None = None,
    ) -> NormalStopResult:
        return NormalStopResult(
            success=False,
            message=f"terminal stop path claimed: {reason}"[:512],
            t0_s=t0_s,
            control_generation=generation,
            identity=identity,
            terminal=True,
        )
