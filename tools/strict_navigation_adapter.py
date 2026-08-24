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
DEFAULT_COMMAND_TIMEOUT_S = 300.0
IDLE_SETTLE_TIMEOUT_S = 20.0
POLL_PERIOD_S = 0.02
MAX_PRODUCT_REPLANS = 2


class StrictNavigationCancelled(RuntimeError):
    """The owning RobotAgent action requested cancellation."""


class StrictNavigationBackend(Protocol):
    def wait_ready(self, *, timeout_s: float = 10.0) -> None: ...
    def compute_path(self, goal_pose: runner.PlanarPose, *, timeout_s: float = 5.0) -> dict[str, Any]: ...
    def wait_fresh_map_pose(self, *, max_age_s: float = 0.10, timeout_s: float = 0.50) -> runner.PoseObservation: ...
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

    @staticmethod
    def _plan_kwargs() -> dict[str, Any]:
        return runner.StrictMotionCaseExecutor._static_plan_kwargs()

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
    ) -> str:
        action_type = resolved.get("action_type")
        goal = self._mapping(resolved.get("action_goal"), "resolved action goal")
        if action_type == "nav2_msgs/action/DriveOnHeading":
            target = self._mapping(goal.get("target"), "DriveOnHeading target")
            distance = self._finite(target.get("x"), "DriveOnHeading distance")
            timeout_s = runner.translation_action_timeout_s(distance)
            self.backend.begin_relay(
                "detour", primitive="DriveOnHeading", direction=1
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
        try:
            self.backend.wait_ready(timeout_s=10.0)
            for plan_attempt in range(1, MAX_PRODUCT_REPLANS + 2):
                self._cancelled(cancelled)
                if self.clock() - started >= self.command_timeout_s:
                    raise runner.StrictMotionRunnerError(
                        "strict navigation exceeded its command timeout"
                    )
                navfn_path = self.backend.compute_path(goal, timeout_s=10.0)
                plan = self.static_planner.build_primitive_plan(
                    navfn_path, **self._plan_kwargs()
                )
                verified = self.static_planner.verify_primitive_plan(
                    plan, navfn_path, **self._plan_kwargs()
                )
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
                                **self._plan_kwargs(),
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
                            break
                        if disposition == "realign_spin_required":
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
                                **self._plan_kwargs(),
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
                        distance = None
                        if action_type == "nav2_msgs/action/DriveOnHeading":
                            distance = self._finite(
                                resolved.get("fresh_distance_to_endpoint_m"),
                                "fresh endpoint distance",
                            )
                            if previous_distance is not None:
                                if distance <= runner.DETOUR_COMPLETED_ENDPOINT_TOLERANCE_M:
                                    break
                                if (
                                    distance
                                    > previous_distance
                                    - runner.MIN_DRIVE_ENDPOINT_PROGRESS_M
                                ):
                                    raise runner.StrictMotionRunnerError(
                                        "DriveOnHeading made no bounded endpoint progress"
                                    )
                            if drive_actions >= runner.MAX_DRIVE_PROGRESS_ACTIONS:
                                raise runner.StrictMotionRunnerError(
                                    "DriveOnHeading exceeded bounded progress actions"
                                )
                        kind = self._execute_resolved(resolved, cancelled=cancelled)
                        completed += 1
                        if kind == "DriveOnHeading":
                            drive_actions += 1
                            previous_distance = distance
                            continue
                        break
                    if replan_required:
                        break
                if replan_required:
                    continue
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
