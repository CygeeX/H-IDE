#!/usr/bin/env python3
"""Expose RobotAgent-compatible Nav2 actions through reviewed primitives.

The public action contracts remain ``NavigateToPose`` and ``Spin`` so the
RobotAgent command registry does not become a second path planner.  Motion is
never delegated to Nav2 FollowPath: every map goal is planned by NavFn, bound
to the generated Office V2 occupancy map, and executed as fresh-odometry
``Spin`` / ``DriveOnHeading`` checkpoints through ``RosLiveBackend``.

Importing this module is inert.  ROS nodes, publishers, action servers, and
service clients are created only by :func:`main`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import threading
import time
from typing import Any, Protocol

import strict_motion_runner as runner


NAVIGATE_ACTION = "/h_ide/strict/navigate_to_pose"
ROTATE_ACTION = "/h_ide/strict/spin"
HEALTH_SERVICE = "/h_ide/strict_navigation/health"
ADAPTER_SCHEMA = "h-ide.strict-navigation-adapter/v1"
DEFAULT_COMMAND_TIMEOUT_S = 2_400.0
IDLE_SETTLE_TIMEOUT_S = 20.0
POLL_PERIOD_S = 0.02
MAX_PRODUCT_REPLANS = 2
# DriveOnHeading is distance-controlled, not an endpoint-tracking controller.
# The frozen integration profile deliberately caps each runtime checkpoint at
# 0.50 m: it re-samples map odometry and heading often enough that small learned
# gait yaw errors cannot accumulate across a two-metre static-plan segment.
#
# The opt-in continuous-handoff profile uses the static planner's reviewed
# maximum (2 m).  This is the important part of the experiment: a long, open
# straight segment is one official DriveOnHeading action instead of four short
# actions, so Nav2 does not request a stop/re-arm boundary every half metre.
MAX_RUNTIME_DRIVE_CHUNK_M = 0.50
CONTINUOUS_RUNTIME_DRIVE_CHUNK_M = 2.0
# Extra map clearance required before a plan is allowed to use the long
# continuous profile.  The reviewed planner clearance is 0.42 m; requiring a
# 0.50 m contour keeps the optimization away from door jambs, walls, and the
# column detour where learned yaw/cross-track error needs frequent checkpoints.
CONTINUOUS_RUNTIME_CLEARANCE_RADIUS_M = 0.50
# Full straight mode raises the planner's segment bound to the map's maximum
# useful span.  This is not an unbounded command: the static planner still
# proves the entire chord against the occupancy raster, and the runtime action
# timeout remains finite.  A 20 m diagonal covers the 16 m x 12 m Office V2 map.
FULL_STRAIGHT_RUNTIME_DRIVE_CHUNK_M = 20.0
FULL_STRAIGHT_RUNTIME_CLEARANCE_RADIUS_M = 0.60
MAX_RUNTIME_DRIVE_ACTIONS_PER_SEGMENT = 8


class StrictNavigationCancelled(RuntimeError):
    """The owning RobotAgent action requested cancellation."""


class StrictNavigationBackend(Protocol):
    def wait_ready(self, *, timeout_s: float = 10.0) -> None: ...
    def compute_path(self, goal_pose: runner.PlanarPose, *, timeout_s: float = 5.0) -> dict[str, Any]: ...
    def wait_fresh_map_pose(
        self,
        *,
        max_age_s: float = runner.MAP_LOCALIZATION_MAX_AGE_S,
        timeout_s: float = runner.MAP_LOCALIZATION_WAIT_TIMEOUT_S,
    ) -> runner.PoseObservation: ...
    def begin_relay(self, mode: str, *, primitive: str | None = None, direction: int | None = None) -> None: ...
    def send_drive(self, distance_m: float, *, timeout_s: float) -> runner.ActionExecution: ...
    def send_spin(self, target_yaw_rad: float, *, timeout_s: float) -> runner.ActionExecution: ...
    def action_done(self, execution: runner.ActionExecution) -> bool: ...
    def finish_action(self, execution: runner.ActionExecution) -> Any: ...
    def cancel_active(self, *, timeout_s: float = 2.0) -> bool: ...
    def stop_relay(self) -> None: ...
    def bridge_health(self) -> dict[str, Any]: ...
    def raise_if_faulted(self) -> None: ...


@dataclass(frozen=True, slots=True)
class NavigationProgress:
    plan_attempt: int
    primitive_index: int
    primitive_count: int
    current_pose: runner.PlanarPose


@dataclass(frozen=True, slots=True)
class NavigationResult:
    final_pose: runner.PlanarPose
    plan_sha256: str
    completed_primitives: int
    plan_attempts: int


class StrictNavigationExecutor:
    """Pure orchestration around the reviewed live backend and static planner."""

    def __init__(
        self,
        backend: StrictNavigationBackend,
        static_planner: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        live_clearance_radius_m: float | None = None,
        continuous_drive_handoff: bool = False,
        full_straight_drive: bool = False,
    ) -> None:
        timeout = float(command_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("command timeout must be finite and positive")
        if live_clearance_radius_m is None:
            live_clearance_radius_m = float(
                static_planner.RELAXED_LIVE_EXECUTION_CLEARANCE_RADIUS_M
            )
        clearance = float(live_clearance_radius_m)
        if not math.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("live clearance radius must be finite and positive")
        self.backend = backend
        self.static_planner = static_planner
        self.clock = clock
        self.sleep = sleep
        self.command_timeout_s = timeout
        self.live_clearance_radius_m = clearance
        if not isinstance(continuous_drive_handoff, bool):
            raise ValueError("continuous_drive_handoff must be a boolean")
        if not isinstance(full_straight_drive, bool):
            raise ValueError("full_straight_drive must be a boolean")
        self.continuous_drive_handoff = continuous_drive_handoff
        self.full_straight_drive = full_straight_drive
        self.runtime_drive_chunk_m = (
            FULL_STRAIGHT_RUNTIME_DRIVE_CHUNK_M
            if full_straight_drive
            else CONTINUOUS_RUNTIME_DRIVE_CHUNK_M
            if continuous_drive_handoff
            else MAX_RUNTIME_DRIVE_CHUNK_M
        )
        self._continuous_map = None
        self._continuous_blocked: dict[float, Any] = {}

    def _plan_kwargs(self, *, full_straight: bool | None = None) -> dict[str, Any]:
        kwargs = runner.StrictMotionCaseExecutor._static_plan_kwargs()
        if full_straight is None:
            full_straight = self.full_straight_drive
        if full_straight:
            kwargs["max_segment_m"] = FULL_STRAIGHT_RUNTIME_DRIVE_CHUNK_M
        return kwargs

    @staticmethod
    def _finite(value: object, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise runner.StrictMotionRunnerError(f"{field} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise runner.StrictMotionRunnerError(f"{field} must be finite")
        return result

    @staticmethod
    def _mapping(value: object, field: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise runner.StrictMotionRunnerError(f"{field} must be an object")
        return value

    def _continuous_segment_allowed(
        self,
        plan: Mapping[str, Any],
        primitive: Mapping[str, Any],
        *,
        full_straight: bool = False,
    ) -> bool:
        """Allow a long chunk only on one generously clear segment.

        The frozen profile deliberately revalidates every half metre.  The
        continuous profile is an optimization for open straight travel, not a
        blanket relaxation for turns or obstacle corridors.  The static
        planner is kept authoritative: if its map implementation is present,
        the selected segment must also be clear at a larger contour (0.50 m for
        the 2 m profile, 0.60 m for full-straight mode).
        Small fake planners used by unit tests do not expose ``StaticMap`` and
        retain the old test-only behavior.
        """

        if not (self.continuous_drive_handoff or self.full_straight_drive):
            return False
        raw_segments = plan.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            return not hasattr(self.static_planner, "StaticMap")
        segment_id = primitive.get("segment_id")
        selected: Mapping[str, Any] | None = None
        for index, raw in enumerate(raw_segments):
            if not isinstance(raw, Mapping):
                return False
            if segment_id is None or raw.get("id") == segment_id:
                selected = raw
                break
        if selected is None:
            return False
        try:
            start = self._mapping(selected.get("start_map"), "segment start")
            end = self._mapping(selected.get("end_map"), "segment end")
            self._finite(selected.get("heading_map_yaw_rad"), "segment heading")
            self._finite(start.get("x"), "segment start x")
            self._finite(start.get("y"), "segment start y")
            self._finite(end.get("x"), "segment end x")
            self._finite(end.get("y"), "segment end y")
        except runner.StrictMotionRunnerError:
            return False

        # Production plans expose the map raster.  Do not silently enable the
        # long profile if the larger-contour proof cannot be computed.
        map_type = getattr(self.static_planner, "StaticMap", None)
        if map_type is None:
            return True
        try:
            if self._continuous_map is None:
                kwargs = self._plan_kwargs()
                self._continuous_map = map_type.load(
                    kwargs["map_yaml_path"], kwargs["map_image_path"]
                )
            clearance_radius = (
                FULL_STRAIGHT_RUNTIME_CLEARANCE_RADIUS_M
                if full_straight
                else CONTINUOUS_RUNTIME_CLEARANCE_RADIUS_M
            )
            blocked = self._continuous_blocked.get(clearance_radius)
            if blocked is None:
                blocked = self._continuous_map.inflated_blocked_cells(
                    clearance_radius
                )
                self._continuous_blocked[clearance_radius] = blocked
            if not self._continuous_map.segment_is_clear(
                (
                    self._finite(start.get("x"), "segment start x"),
                    self._finite(start.get("y"), "segment start y"),
                ),
                (
                    self._finite(end.get("x"), "segment end x"),
                    self._finite(end.get("y"), "segment end y"),
                ),
                blocked,
            ):
                return False
        except Exception:
            return False
        return True

    def _build_verified_plan(
        self,
        navfn_path: Mapping[str, Any],
    ) -> tuple[Any, Mapping[str, Any], bool, dict[str, Any]]:
        """Build a plan, falling back to the reviewed 2 m partition if needed.

        ``full_straight_drive`` changes only the planner's maximum segment
        length.  A failure to construct or verify that wider plan must not turn
        a valid ordinary navigation request into an error; retry the same
        NavFn path with the frozen planner bound and let the existing 2 m/0.5 m
        runtime profiles take over.
        """

        requested_full = self.full_straight_drive
        plan_kwargs = self._plan_kwargs(full_straight=requested_full)
        try:
            plan = self.static_planner.build_primitive_plan(
                navfn_path,
                **plan_kwargs,
            )
            verified = self.static_planner.verify_primitive_plan(
                plan,
                navfn_path,
                **plan_kwargs,
            )
            return plan, verified, requested_full, plan_kwargs
        except Exception as full_error:
            if not requested_full:
                raise
            fallback_kwargs = self._plan_kwargs(full_straight=False)
            try:
                plan = self.static_planner.build_primitive_plan(
                    navfn_path,
                    **fallback_kwargs,
                )
                verified = self.static_planner.verify_primitive_plan(
                    plan,
                    navfn_path,
                    **fallback_kwargs,
                )
            except Exception as fallback_error:
                raise fallback_error from full_error
            return plan, verified, False, fallback_kwargs

    def _cancelled(self, cancelled: Callable[[], bool]) -> None:
        if cancelled():
            raise StrictNavigationCancelled("strict navigation was cancelled")

    def _wait_action(
        self,
        execution: runner.ActionExecution,
        *,
        timeout_s: float,
        cancelled: Callable[[], bool],
    ) -> None:
        deadline = self.clock() + min(float(timeout_s), self.command_timeout_s)
        while not self.backend.action_done(execution):
            self.backend.raise_if_faulted()
            self._cancelled(cancelled)
            if self.clock() >= deadline:
                raise runner.StrictMotionRunnerError(
                    f"{execution.kind} exceeded its bounded action timeout"
                )
            self.sleep(POLL_PERIOD_S)
        self.backend.finish_action(execution)

    def _wait_balanced_idle(self, *, cancelled: Callable[[], bool]) -> None:
        deadline = self.clock() + IDLE_SETTLE_TIMEOUT_S
        while self.clock() < deadline:
            self._cancelled(cancelled)
            self.backend.raise_if_faulted()
            try:
                health = self.backend.bridge_health()
            except runner.StrictMotionRunnerError:
                self.sleep(0.05)
                continue
            if (
                health.get("healthy") is True
                and str(health.get("gaitState", "")).lower() == "idle"
                and health.get("fault") is None
            ):
                return
            self.sleep(0.05)
        raise runner.StrictMotionRunnerError(
            "strict primitive did not return to healthy learned-policy IDLE"
        )

    def _stop_after_primitive(self, *, cancelled: Callable[[], bool]) -> None:
        self.backend.stop_relay()
        self._wait_balanced_idle(cancelled=cancelled)

    def _execute_resolved(
        self,
        resolved: Mapping[str, Any],
        *,
        cancelled: Callable[[], bool],
        wait_for_idle: bool = True,
        drive_chunk_m: float | None = None,
    ) -> str:
        if drive_chunk_m is None:
            drive_chunk_m = self.runtime_drive_chunk_m
        if (
            isinstance(drive_chunk_m, bool)
            or not isinstance(drive_chunk_m, (int, float))
            or not math.isfinite(float(drive_chunk_m))
            or float(drive_chunk_m) <= 0.0
        ):
            raise ValueError("drive chunk limit must be finite and positive")
        drive_chunk_m = float(drive_chunk_m)
        action_type = resolved.get("action_type")
        goal = self._mapping(resolved.get("action_goal"), "resolved action goal")
        if action_type == "nav2_msgs/action/DriveOnHeading":
            target = self._mapping(goal.get("target"), "DriveOnHeading target")
            resolved_distance = self._finite(
                target.get("x"), "DriveOnHeading distance"
            )
            distance = math.copysign(
                min(abs(resolved_distance), drive_chunk_m),
                resolved_distance,
            )
            timeout_s = runner.translation_action_timeout_s(abs(distance))
            self.backend.begin_relay(
                "detour",
                primitive="DriveOnHeading",
                direction=1 if distance > 0.0 else -1,
            )
            execution = self.backend.send_drive(distance, timeout_s=timeout_s)
            kind = "DriveOnHeading"
        elif action_type == "nav2_msgs/action/Spin":
            yaw = self._finite(goal.get("target_yaw_rad"), "Spin target yaw")
            timeout_s = (
                abs(yaw) / runner.OBSERVED_SPIN_PROGRESS_FLOOR_RPS
                + runner.SPIN_STARTUP_STOP_MARGIN_S
            )
            self.backend.begin_relay(
                "detour", primitive="Spin", direction=1 if yaw > 0.0 else -1
            )
            execution = self.backend.send_spin(yaw, timeout_s=timeout_s)
            kind = "Spin"
        else:
            raise runner.StrictMotionRunnerError(
                "static planner emitted a forbidden product action"
            )
        try:
            self._wait_action(
                execution,
                timeout_s=timeout_s,
                cancelled=cancelled,
            )
        finally:
            self.backend.stop_relay()
        if wait_for_idle:
            self._wait_balanced_idle(cancelled=cancelled)
        return kind

    def _cleanup(self) -> None:
        try:
            self.backend.cancel_active()
        except Exception:
            pass
        try:
            self.backend.stop_relay()
        except Exception:
            pass

    def _finish_relaxed_terminal_pose(
        self,
        goal: runner.PlanarPose,
        *,
        plan_sha256: str,
        plan_attempt: int,
        completed_primitives: int,
    ) -> NavigationResult | None:
        """Finish a previously reviewed route from its physical endpoint.

        Learned ``DriveOnHeading`` is intentionally accepted inside the
        reviewed 0.35 m physical endpoint after it has made real progress.  A
        subsequent NavFn recovery can produce a residual that is too short to
        partition even though the robot is already inside that same endpoint
        envelope.  In that case, do not backtrack or abort: keep the last
        verified plan provenance and accept the physical endpoint without
        another learned turn.  Office inspection owns its scan headings as
        separate ``Spin`` actions; a hidden terminal turn beside furniture
        only adds drift and cannot improve positional arrival.

        This path is unavailable before at least one static plan has been
        verified, so an arbitrary near-goal request cannot bypass planning.
        """

        if not plan_sha256:
            return None
        fresh = self.backend.wait_fresh_map_pose()
        distance = math.hypot(goal.x - fresh.pose.x, goal.y - fresh.pose.y)
        if distance > runner.DETOUR_COMPLETED_ENDPOINT_TOLERANCE_M:
            return None
        return NavigationResult(
            final_pose=fresh.pose,
            plan_sha256=plan_sha256,
            completed_primitives=completed_primitives,
            plan_attempts=plan_attempt,
        )

    @staticmethod
    def _terminal_path_sha256(
        navfn_path: Mapping[str, Any], goal: runner.PlanarPose
    ) -> str:
        payload = {
            "schemaVersion": "h-ide.strict-terminal-path/1",
            "goal": {"x": goal.x, "y": goal.y, "yaw": goal.yaw},
            "navfnPath": dict(navfn_path),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def navigate(
        self,
        goal: runner.PlanarPose,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        feedback: Callable[[NavigationProgress], None] | None = None,
    ) -> NavigationResult:
        if not all(math.isfinite(value) for value in (goal.x, goal.y, goal.yaw)):
            raise ValueError("navigation goal must be finite")
        started = self.clock()
        completed = 0
        final_plan_hash = ""
        pending_drive_handoff = False
        pending_drive_direction: int | None = None
        try:
            self.backend.wait_ready(timeout_s=10.0)
            recovery_pose = self.backend.wait_fresh_map_pose()
            recovery = self.static_planner.resolve_start_clearance_recovery(
                {
                    "x": recovery_pose.pose.x,
                    "y": recovery_pose.pose.y,
                    "yaw": recovery_pose.pose.yaw,
                },
                **self._plan_kwargs(full_straight=False),
            )
            if recovery.get("disposition") == "send_action":
                self._execute_resolved(recovery, cancelled=cancelled)
                completed += 1
            elif recovery.get("disposition") != "not_required":
                raise runner.StrictMotionRunnerError(
                    "static start-clearance recovery returned an invalid disposition"
                )
            for plan_attempt in range(1, MAX_PRODUCT_REPLANS + 2):
                self._cancelled(cancelled)
                if self.clock() - started >= self.command_timeout_s:
                    raise runner.StrictMotionRunnerError(
                        "strict navigation exceeded its command timeout"
                    )
                navfn_path = self.backend.compute_path(goal, timeout_s=10.0)
                fresh_before_plan = self.backend.wait_fresh_map_pose()
                if (
                    math.hypot(
                        goal.x - fresh_before_plan.pose.x,
                        goal.y - fresh_before_plan.pose.y,
                    )
                    <= runner.DETOUR_COMPLETED_ENDPOINT_TOLERANCE_M
                ):
                    if pending_drive_handoff:
                        self._wait_balanced_idle(cancelled=cancelled)
                        pending_drive_handoff = False
                        pending_drive_direction = None
                    return NavigationResult(
                        final_pose=fresh_before_plan.pose,
                        plan_sha256=self._terminal_path_sha256(navfn_path, goal),
                        completed_primitives=completed,
                        plan_attempts=plan_attempt,
                    )
                try:
                    (
                        plan,
                        verified,
                        full_plan_active,
                        plan_kwargs,
                    ) = self._build_verified_plan(navfn_path)
                except Exception:
                    terminal = self._finish_relaxed_terminal_pose(
                        goal,
                        plan_sha256=final_plan_hash,
                        plan_attempt=plan_attempt,
                        completed_primitives=completed,
                    )
                    if terminal is not None:
                        return terminal
                    raise
                primitives = verified.get("primitives")
                if not isinstance(primitives, list) or not primitives:
                    raise runner.StrictMotionRunnerError(
                        "static primitive plan is empty"
                    )
                final_plan_hash = str(verified.get("plan_sha256", ""))
                if not final_plan_hash:
                    raise runner.StrictMotionRunnerError(
                        "static primitive plan omitted its provenance hash"
                    )
                replan_required = False
                for primitive_index, primitive in enumerate(primitives):
                    if not isinstance(primitive, Mapping):
                        raise runner.StrictMotionRunnerError(
                            "static primitive entry is invalid"
                        )
                    full_segment_handoff = (
                        full_plan_active
                        and self._continuous_segment_allowed(
                            verified,
                            primitive,
                            full_straight=True,
                        )
                    )
                    continuous_segment_handoff = (
                        (
                            self.continuous_drive_handoff
                            or self.full_straight_drive
                            or full_plan_active
                        )
                        and self._continuous_segment_allowed(
                            verified,
                            primitive,
                            full_straight=False,
                        )
                    )
                    segment_handoff_enabled = (
                        full_segment_handoff or continuous_segment_handoff
                    )
                    plan_drive_chunk_m = (
                        FULL_STRAIGHT_RUNTIME_DRIVE_CHUNK_M
                        if full_segment_handoff
                        else CONTINUOUS_RUNTIME_DRIVE_CHUNK_M
                        if continuous_segment_handoff
                        else MAX_RUNTIME_DRIVE_CHUNK_M
                    )
                    realignments = 0
                    drive_actions = 0
                    previous_distance: float | None = None
                    while True:
                        self._cancelled(cancelled)
                        fresh = self.backend.wait_fresh_map_pose()
                        if feedback is not None:
                            feedback(
                                NavigationProgress(
                                    plan_attempt=plan_attempt,
                                    primitive_index=primitive_index,
                                    primitive_count=len(primitives),
                                    current_pose=fresh.pose,
                                )
                            )
                        pose = {
                            "x": fresh.pose.x,
                            "y": fresh.pose.y,
                            "yaw": fresh.pose.yaw,
                        }
                        try:
                            resolved = self.static_planner.resolve_live_primitive(
                                plan,
                                primitive_index,
                                navfn_path,
                                pose,
                                live_execution_clearance_radius_m=(
                                    self.live_clearance_radius_m
                                ),
                                **plan_kwargs,
                            )
                        except Exception:
                            if plan_attempt <= MAX_PRODUCT_REPLANS:
                                replan_required = True
                                break
                            raise
                        disposition = resolved.get("disposition")
                        if disposition in {
                            "skip_within_endpoint_tolerance",
                            "skip_within_heading_tolerance",
                        }:
                            if pending_drive_handoff:
                                self._wait_balanced_idle(cancelled=cancelled)
                                pending_drive_handoff = False
                                pending_drive_direction = None
                            break
                        if disposition == "realign_spin_required":
                            if pending_drive_handoff:
                                self._wait_balanced_idle(cancelled=cancelled)
                                pending_drive_handoff = False
                                pending_drive_direction = None
                            if realignments >= runner.MAX_DRIVE_REALIGNMENT_SPINS:
                                raise runner.StrictMotionRunnerError(
                                    "DriveOnHeading exceeded bounded realignment"
                                )
                            preceding = int(
                                resolved.get("realignment_primitive_sequence", -1)
                            )
                            if preceding != primitive_index - 1:
                                raise runner.StrictMotionRunnerError(
                                    "invalid DriveOnHeading realignment provenance"
                                )
                            resolved = self.static_planner.resolve_live_primitive(
                                plan,
                                preceding,
                                navfn_path,
                                pose,
                                live_execution_clearance_radius_m=(
                                    self.live_clearance_radius_m
                                ),
                                **plan_kwargs,
                            )
                            if resolved.get("disposition") != "send_action":
                                raise runner.StrictMotionRunnerError(
                                    "realignment did not resolve to a Spin"
                                )
                            realignments += 1
                            self._execute_resolved(resolved, cancelled=cancelled)
                            completed += 1
                            continue
                        if disposition != "send_action":
                            raise runner.StrictMotionRunnerError(
                                "static primitive could not be executed"
                            )
                        action_type = resolved.get("action_type")
                        drive_direction: int | None = None
                        if action_type == "nav2_msgs/action/DriveOnHeading":
                            raw_goal = self._mapping(
                                resolved.get("action_goal"),
                                "resolved action goal",
                            )
                            raw_target = self._mapping(
                                raw_goal.get("target"),
                                "resolved DriveOnHeading target",
                            )
                            raw_distance = self._finite(
                                raw_target.get("x"),
                                "resolved DriveOnHeading command distance",
                            )
                            if abs(raw_distance) <= 1e-9:
                                raise runner.StrictMotionRunnerError(
                                    "resolved DriveOnHeading command distance is zero"
                                )
                            drive_direction = 1 if raw_distance > 0.0 else -1
                        # A pending no-wait handoff is valid only for another
                        # DriveOnHeading in the same signed direction. Flush it
                        # before a Spin, a reverse/recovery drive, or any other
                        # transition so the learned policy gets its ordinary
                        # balanced-IDLE boundary.
                        if pending_drive_handoff and (
                            action_type != "nav2_msgs/action/DriveOnHeading"
                            or drive_direction != pending_drive_direction
                        ):
                            self._wait_balanced_idle(cancelled=cancelled)
                            pending_drive_handoff = False
                            pending_drive_direction = None
                        distance = None
                        if action_type == "nav2_msgs/action/DriveOnHeading":
                            distance = self._finite(
                                resolved.get("fresh_distance_to_endpoint_m"),
                                "fresh endpoint distance",
                            )
                            if previous_distance is not None:
                                # The following final-heading checkpoint is
                                # validated by the static planner against the
                                # reviewed endpoint tolerance (0.15 m).  Do
                                # not terminate progress at the looser 0.35 m
                                # motion-acceptance radius, or the very next
                                # checkpoint deterministically rejects the
                                # same pose as "before endpoint".
                                if distance <= runner.DETOUR_ENDPOINT_TOLERANCE_M:
                                    break
                                if (
                                    distance
                                    > previous_distance
                                    - runner.MIN_DRIVE_ENDPOINT_PROGRESS_M
                                ):
                                    raise runner.StrictMotionRunnerError(
                                        "DriveOnHeading made no bounded endpoint progress"
                                    )
                            if (
                                drive_actions
                                >= MAX_RUNTIME_DRIVE_ACTIONS_PER_SEGMENT
                            ):
                                raise runner.StrictMotionRunnerError(
                                    "DriveOnHeading exceeded bounded progress actions"
                                )
                        wait_for_idle = True
                        if (
                            segment_handoff_enabled
                            and action_type == "nav2_msgs/action/DriveOnHeading"
                            and distance is not None
                            and isinstance(
                                resolved.get("action_goal"), Mapping
                            )
                        ):
                            resolved_goal = self._mapping(
                                resolved.get("action_goal"),
                                "resolved action goal",
                            )
                            resolved_target = self._mapping(
                                resolved_goal.get("target"),
                                "resolved DriveOnHeading target",
                            )
                            command_distance = abs(
                                self._finite(
                                    resolved_target.get("x"),
                                    "resolved DriveOnHeading command distance",
                                )
                            )
                            admitted_distance = min(
                                command_distance,
                                plan_drive_chunk_m,
                            )
                            if (
                                distance
                                > admitted_distance
                                + runner.DETOUR_ENDPOINT_TOLERANCE_M
                            ):
                                # The action still publishes a zero at its own
                                # endpoint, but the next same-heading chunk is
                                # admitted immediately. If the fresh resolver
                                # asks for a Spin, a skip, or a replan, the
                                # pending handoff is flushed through the normal
                                # balanced-IDLE gate before that transition.
                                wait_for_idle = False
                        kind = self._execute_resolved(
                            resolved,
                            cancelled=cancelled,
                            wait_for_idle=wait_for_idle,
                            drive_chunk_m=plan_drive_chunk_m,
                        )
                        completed += 1
                        pending_drive_handoff = (
                            kind == "DriveOnHeading" and not wait_for_idle
                        )
                        pending_drive_direction = (
                            drive_direction if pending_drive_handoff else None
                        )
                        if kind == "DriveOnHeading":
                            drive_actions += 1
                            previous_distance = distance
                            continue
                        break
                    if replan_required:
                        if pending_drive_handoff:
                            self._wait_balanced_idle(cancelled=cancelled)
                            pending_drive_handoff = False
                            pending_drive_direction = None
                        break
                if replan_required:
                    continue
                if pending_drive_handoff:
                    self._wait_balanced_idle(cancelled=cancelled)
                    pending_drive_handoff = False
                    pending_drive_direction = None
                final = self.backend.wait_fresh_map_pose().pose
                return NavigationResult(
                    final_pose=final,
                    plan_sha256=final_plan_hash,
                    completed_primitives=completed,
                    plan_attempts=plan_attempt,
                )
            raise runner.StrictMotionRunnerError(
                "strict navigation exhausted its bounded replans"
            )
        except Exception:
            self._cleanup()
            raise

    def rotate(
        self,
        target_yaw_rad: float,
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        target = self._finite(target_yaw_rad, "rotation target")
        if abs(target) <= 1e-6 or abs(target) > math.tau:
            raise ValueError("rotation target must be in [-2pi, 2pi] and non-zero")
        try:
            self.backend.wait_ready(timeout_s=10.0)
            mode = "turn_left" if target > 0.0 else "turn_right"
            self.backend.begin_relay(mode)
            timeout_s = (
                abs(target) / runner.OBSERVED_SPIN_PROGRESS_FLOOR_RPS
                + runner.SPIN_STARTUP_STOP_MARGIN_S
            )
            execution = self.backend.send_spin(target, timeout_s=timeout_s)
            try:
                self._wait_action(
                    execution,
                    timeout_s=timeout_s,
                    cancelled=cancelled,
                )
            finally:
                self.backend.stop_relay()
            self._wait_balanced_idle(cancelled=cancelled)
        except Exception:
            self._cleanup()
            raise


class StrictNavigationActionServer:
    """ROS action facade; constructed only by the live CLI entry point."""

    def __init__(self, backend: runner.RosLiveBackend, executor: StrictNavigationExecutor) -> None:
        from nav2_msgs.action import NavigateToPose, Spin
        from rclpy.action import ActionServer, CancelResponse, GoalResponse
        from rclpy.callback_groups import ReentrantCallbackGroup
        from std_srvs.srv import Trigger

        self.backend = backend
        self.executor = executor
        self._NavigateToPose = NavigateToPose
        self._Spin = Spin
        self._CancelResponse = CancelResponse
        self._GoalResponse = GoalResponse
        self._lock = threading.RLock()
        self._reserved = False
        group = ReentrantCallbackGroup()
        self._navigate = ActionServer(
            backend.node,
            NavigateToPose,
            NAVIGATE_ACTION,
            execute_callback=self._execute_navigate,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=group,
        )
        self._rotate = ActionServer(
            backend.node,
            Spin,
            ROTATE_ACTION,
            execute_callback=self._execute_rotate,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=group,
        )
        self._health = backend.node.create_service(
            Trigger,
            HEALTH_SERVICE,
            self._health_callback,
            callback_group=group,
        )

    def _goal_callback(self, _request: Any) -> Any:
        with self._lock:
            if self._reserved:
                return self._GoalResponse.REJECT
            self._reserved = True
        return self._GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle: Any) -> Any:
        return self._CancelResponse.ACCEPT

    def _release(self) -> None:
        with self._lock:
            self._reserved = False

    def _health_callback(self, _request: Any, response: Any) -> Any:
        with self._lock:
            active = self._reserved
        try:
            self.backend.raise_if_faulted()
            bridge = self.backend.bridge_health()
            healthy = bridge.get("healthy") is True
            detail = None
        except Exception as exc:
            healthy = False
            bridge = None
            detail = (str(exc).strip() or type(exc).__name__)[:512]
        response.success = healthy
        response.message = json.dumps(
            {
                "schemaVersion": ADAPTER_SCHEMA,
                "healthy": healthy,
                "active": active,
                "continuousDriveHandoff": self.executor.continuous_drive_handoff,
                "fullStraightDrive": self.executor.full_straight_drive,
                "runtimeDriveChunkM": self.executor.runtime_drive_chunk_m,
                "bridge": bridge,
                "error": detail,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return response

    @staticmethod
    def _goal_pose(request: Any) -> runner.PlanarPose:
        pose = request.pose.pose
        quaternion_norm = math.sqrt(
            float(pose.orientation.x) ** 2
            + float(pose.orientation.y) ** 2
            + float(pose.orientation.z) ** 2
            + float(pose.orientation.w) ** 2
        )
        if not math.isfinite(quaternion_norm) or quaternion_norm <= 1e-9:
            raise ValueError("navigation goal quaternion is invalid")
        yaw = math.atan2(
            2.0
            * (
                float(pose.orientation.w) * float(pose.orientation.z)
                + float(pose.orientation.x) * float(pose.orientation.y)
            ),
            1.0
            - 2.0
            * (
                float(pose.orientation.y) ** 2
                + float(pose.orientation.z) ** 2
            ),
        )
        if str(request.pose.header.frame_id) != "map":
            raise ValueError("strict navigation accepts only map-frame goals")
        return runner.PlanarPose(
            float(pose.position.x), float(pose.position.y), yaw
        )

    def _execute_navigate(self, goal_handle: Any) -> Any:
        result = self._NavigateToPose.Result()
        try:
            goal = self._goal_pose(goal_handle.request)

            def publish(progress: NavigationProgress) -> None:
                message = self._NavigateToPose.Feedback()
                message.current_pose.header.frame_id = "map"
                message.current_pose.pose.position.x = progress.current_pose.x
                message.current_pose.pose.position.y = progress.current_pose.y
                message.current_pose.pose.orientation.z = math.sin(
                    progress.current_pose.yaw / 2.0
                )
                message.current_pose.pose.orientation.w = math.cos(
                    progress.current_pose.yaw / 2.0
                )
                goal_handle.publish_feedback(message)

            self.executor.navigate(
                goal,
                cancelled=lambda: bool(goal_handle.is_cancel_requested),
                feedback=publish,
            )
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.succeed()
            result.error_code = 0
            result.error_msg = ""
        except StrictNavigationCancelled:
            goal_handle.canceled()
            result.error_code = 0
            result.error_msg = "cancelled"
        except Exception as exc:
            self.backend.node.get_logger().error(
                "strict NavigateToPose aborted: %s"
                % (str(exc).strip() or type(exc).__name__)
            )
            goal_handle.abort()
            result.error_code = 100
            result.error_msg = (str(exc).strip() or type(exc).__name__)[:512]
        finally:
            self._release()
        return result

    def _execute_rotate(self, goal_handle: Any) -> Any:
        result = self._Spin.Result()
        try:
            self.executor.rotate(
                float(goal_handle.request.target_yaw),
                cancelled=lambda: bool(goal_handle.is_cancel_requested),
            )
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.succeed()
            result.error_code = 0
            result.error_msg = ""
        except StrictNavigationCancelled:
            goal_handle.canceled()
            result.error_code = 0
            result.error_msg = "cancelled"
        except Exception as exc:
            self.backend.node.get_logger().error(
                "strict Spin aborted: %s"
                % (str(exc).strip() or type(exc).__name__)
            )
            goal_handle.abort()
            result.error_code = 100
            result.error_msg = (str(exc).strip() or type(exc).__name__)[:512]
        finally:
            self._release()
        return result

    def destroy(self) -> None:
        self._navigate.destroy()
        self._rotate.destroy()
        self._health.destroy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_S,
    )
    parser.add_argument(
        "--continuous-drive-handoff",
        action="store_true",
        help=(
            "skip the balanced-IDLE evidence wait between contiguous "
            "DriveOnHeading chunks; final/direction-change stops remain gated"
        ),
    )
    parser.add_argument(
        "--full-straight-drive",
        action="store_true",
        help=(
            "allow one full map-verified straight segment (with automatic "
            "fallback to the 2 m/0.5 m profiles); implies continuous handoff"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    static_planner = runner._load_file_module(
        "h_ide_static_primitive_planner_adapter",
        runner.STATIC_PLANNER_PATH,
    )
    server: StrictNavigationActionServer | None = None
    with runner.RosLiveBackend() as backend:
        executor = StrictNavigationExecutor(
            backend,
            static_planner,
            command_timeout_s=args.command_timeout_seconds,
            continuous_drive_handoff=args.continuous_drive_handoff,
            full_straight_drive=args.full_straight_drive,
        )
        server = StrictNavigationActionServer(backend, executor)
        try:
            backend.wait_ready(timeout_s=30.0)
            while True:
                backend.raise_if_faulted()
                time.sleep(0.2)
        except KeyboardInterrupt:
            return 0
        finally:
            if server is not None:
                server.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
