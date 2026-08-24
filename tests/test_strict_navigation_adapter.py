from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strict_motion_runner as runner  # noqa: E402
import strict_navigation_adapter as adapter  # noqa: E402


class FakeBackend:
    def __init__(self, *, done: bool = True) -> None:
        self.pose = runner.PlanarPose(0.0, 0.0, 0.0)
        self.done = done
        self.ready_calls = 0
        self.compute_calls = 0
        self.started: list[tuple[str, str | None, int | None]] = []
        self.sent: list[tuple[str, float]] = []
        self.cancel_calls = 0
        self.stop_calls = 0

    def wait_ready(self, *, timeout_s: float = 10.0) -> None:
        assert timeout_s > 0.0
        self.ready_calls += 1

    def compute_path(self, goal_pose, *, timeout_s: float = 5.0):
        self.compute_calls += 1
        return {
            "frame_id": "map",
            "poses": [
                {"x": self.pose.x, "y": self.pose.y, "yaw": self.pose.yaw},
                {"x": goal_pose.x, "y": goal_pose.y, "yaw": goal_pose.yaw},
            ],
        }

    def wait_fresh_map_pose(self, **_kwargs):
        return SimpleNamespace(pose=self.pose)

    def begin_relay(self, mode, *, primitive=None, direction=None):
        self.started.append((mode, primitive, direction))

    def send_drive(self, distance_m, *, timeout_s):
        self.sent.append(("DriveOnHeading", distance_m))
        return SimpleNamespace(kind="DriveOnHeading", distance=distance_m)

    def send_spin(self, target_yaw_rad, *, timeout_s):
        self.sent.append(("Spin", target_yaw_rad))
        return SimpleNamespace(kind="Spin", distance=target_yaw_rad)

    def action_done(self, _execution):
        return self.done

    def finish_action(self, execution):
        if execution.kind == "DriveOnHeading":
            self.pose = runner.PlanarPose(
                self.pose.x + execution.distance,
                self.pose.y,
                self.pose.yaw,
            )
        else:
            self.pose = runner.PlanarPose(
                self.pose.x,
                self.pose.y,
                self.pose.yaw + execution.distance,
            )

    def cancel_active(self, *, timeout_s=2.0):
        self.cancel_calls += 1
        return True

    def stop_relay(self):
        self.stop_calls += 1

    def bridge_health(self):
        return {"healthy": True, "gaitState": "idle", "fault": None}

    def raise_if_faulted(self):
        return None


class FakePlanner:
    RELAXED_LIVE_EXECUTION_CLEARANCE_RADIUS_M = 0.25

    def __init__(self, *, fail_first_resolve: bool = False, forbidden: bool = False):
        self.fail_first_resolve = fail_first_resolve
        self.forbidden = forbidden
        self.resolve_calls: dict[tuple[int, int], int] = defaultdict(int)
        self.plan_number = 0

    def build_primitive_plan(self, navfn_path, **_kwargs):
        self.plan_number += 1
        return {
            "plan_sha256": f"plan-{self.plan_number}",
            "primitives": [
                {"kind": "spin"},
                {"kind": "drive_on_heading"},
            ],
        }

    def verify_primitive_plan(self, plan, navfn_path, **_kwargs):
        return plan

    def resolve_live_primitive(self, plan, index, navfn_path, pose, **_kwargs):
        key = (self.plan_number, index)
        self.resolve_calls[key] += 1
        if self.fail_first_resolve and self.plan_number == 1:
            raise RuntimeError("fresh chord is blocked")
        if index == 0:
            return {
                "disposition": "send_action",
                "action_type": "nav2_msgs/action/Spin",
                "action_goal": {"target_yaw_rad": 0.2},
            }
        remaining = max(0.0, 1.0 - float(pose["x"]))
        if remaining <= runner.DETOUR_ENDPOINT_TOLERANCE_M:
            return {"disposition": "skip_within_endpoint_tolerance"}
        return {
            "disposition": "send_action",
            "action_type": (
                "nav2_msgs/action/FollowPath"
                if self.forbidden
                else "nav2_msgs/action/DriveOnHeading"
            ),
            "action_goal": {
                "target": {"x": remaining, "y": 0.0, "z": 0.0},
                "speed_mps": runner.OFFICIAL_DRIVE_SPEED_MPS,
            },
            "fresh_distance_to_endpoint_m": remaining,
        }


class ResidualEndpointBackend(FakeBackend):
    """First learned drive stops 0.26 m short, as observed in MuJoCo."""

    def __init__(self) -> None:
        super().__init__()
        self.drive_finishes = 0

    def finish_action(self, execution):
        if execution.kind != "DriveOnHeading":
            return super().finish_action(execution)
        self.drive_finishes += 1
        progress = (
            max(0.0, execution.distance - 0.26)
            if self.drive_finishes == 1
            else execution.distance
        )
        self.pose = runner.PlanarPose(
            self.pose.x + progress,
            self.pose.y,
            self.pose.yaw,
        )


class FinalHeadingPlanner(FakePlanner):
    def build_primitive_plan(self, navfn_path, **_kwargs):
        self.plan_number += 1
        return {
            "plan_sha256": f"plan-{self.plan_number}",
            "primitives": [
                {"kind": "spin"},
                {"kind": "drive_on_heading"},
                {"kind": "spin"},
            ],
        }

    def resolve_live_primitive(self, plan, index, navfn_path, pose, **_kwargs):
        del plan, navfn_path, _kwargs
        if index == 0:
            return {
                "disposition": "send_action",
                "action_type": "nav2_msgs/action/Spin",
                "action_goal": {"target_yaw_rad": 0.2},
            }
        distance = max(0.0, 1.0 - float(pose["x"]))
        if index == 1:
            if distance <= runner.DETOUR_ENDPOINT_TOLERANCE_M:
                return {
                    "disposition": "skip_within_endpoint_tolerance",
                    "fresh_distance_to_endpoint_m": distance,
                }
            return {
                "disposition": "send_action",
                "action_type": "nav2_msgs/action/DriveOnHeading",
                "action_goal": {
                    "target": {"x": distance, "y": 0.0, "z": 0.0},
                    "speed_mps": runner.OFFICIAL_DRIVE_SPEED_MPS,
                },
                "fresh_distance_to_endpoint_m": distance,
            }
        if distance > runner.DETOUR_ENDPOINT_TOLERANCE_M:
            raise ValueError("final heading checkpoint reached before endpoint")
        return {
            "disposition": "send_action",
            "action_type": "nav2_msgs/action/Spin",
            "action_goal": {"target_yaw_rad": 0.3},
        }


def executor(backend: FakeBackend, planner: FakePlanner):
    return adapter.StrictNavigationExecutor(
        backend,
        planner,
        sleep=lambda _seconds: None,
    )


def test_navigate_uses_only_spin_and_drive_then_returns_idle() -> None:
    backend = FakeBackend()
    planner = FakePlanner()
    progress: list[adapter.NavigationProgress] = []

    result = executor(backend, planner).navigate(
        runner.PlanarPose(1.0, 0.0, 0.2),
        feedback=progress.append,
    )

    assert result.plan_sha256 == "plan-1"
    assert result.completed_primitives == 3
    assert backend.compute_calls == 1
    assert backend.sent == [
        ("Spin", 0.2),
        ("DriveOnHeading", 0.5),
        ("DriveOnHeading", 0.5),
    ]
    assert backend.started == [
        ("detour", "Spin", 1),
        ("detour", "DriveOnHeading", 1),
        ("detour", "DriveOnHeading", 1),
    ]
    assert backend.stop_calls == 3
    assert len(progress) >= 2


def test_navigate_closes_observed_residual_before_final_heading() -> None:
    backend = ResidualEndpointBackend()

    result = executor(backend, FinalHeadingPlanner()).navigate(
        runner.PlanarPose(1.0, 0.0, 0.5)
    )

    assert result.final_pose.x == pytest.approx(1.0)
    assert backend.sent == [
        ("Spin", 0.2),
        ("DriveOnHeading", 0.5),
        ("DriveOnHeading", 0.5),
        ("DriveOnHeading", pytest.approx(0.26)),
        ("Spin", 0.3),
    ]


def test_navigate_realigns_long_drive_at_half_metre_boundaries() -> None:
    backend = FakeBackend()

    result = executor(backend, FakePlanner()).navigate(
        runner.PlanarPose(1.0, 0.0, 0.2)
    )

    assert result.final_pose.x == pytest.approx(1.0)
    assert backend.sent == [
        ("Spin", 0.2),
        ("DriveOnHeading", 0.5),
        ("DriveOnHeading", 0.5),
    ]


def test_fresh_geometry_failure_replans_at_most_twice() -> None:
    backend = FakeBackend()
    planner = FakePlanner(fail_first_resolve=True)

    result = executor(backend, planner).navigate(
        runner.PlanarPose(1.0, 0.0, 0.2)
    )

    assert result.plan_attempts == 2
    assert backend.compute_calls == 2
    assert result.plan_sha256 == "plan-2"


def test_forbidden_follow_path_fails_closed_and_publishes_zero() -> None:
    backend = FakeBackend()
    planner = FakePlanner(forbidden=True)

    with pytest.raises(
        runner.StrictMotionRunnerError,
        match="forbidden product action",
    ):
        executor(backend, planner).navigate(
            runner.PlanarPose(1.0, 0.0, 0.2)
        )

    assert backend.cancel_calls == 1
    assert backend.stop_calls >= 2
    assert all(kind != "FollowPath" for kind, _value in backend.sent)


def test_cancel_propagates_to_active_primitive_and_zeroes_relay() -> None:
    backend = FakeBackend(done=False)
    planner = FakePlanner()
    checks = 0

    def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(adapter.StrictNavigationCancelled):
        executor(backend, planner).navigate(
            runner.PlanarPose(1.0, 0.0, 0.2),
            cancelled=cancelled,
        )

    assert backend.cancel_calls >= 1
    assert backend.stop_calls >= 1


@pytest.mark.parametrize(("target", "mode"), ((0.5, "turn_left"), (-0.5, "turn_right")))
def test_rotate_uses_the_reviewed_spin_relay(target: float, mode: str) -> None:
    backend = FakeBackend()
    planner = FakePlanner()

    executor(backend, planner).rotate(target)

    assert backend.started == [(mode, None, None)]
    assert backend.sent == [("Spin", target)]
    assert backend.stop_calls == 1


def test_adapter_import_has_no_ros_or_motion_side_effects() -> None:
    assert adapter.NAVIGATE_ACTION == "/h_ide/strict/navigate_to_pose"
    assert adapter.ROTATE_ACTION == "/h_ide/strict/spin"
    assert adapter.MAX_PRODUCT_REPLANS == 2
