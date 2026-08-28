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
        self.health_calls = 0

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
        self.health_calls += 1
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

    def resolve_start_clearance_recovery(self, _pose, **_kwargs):
        return {"disposition": "not_required", "action_goal": None}

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


def executor(
    backend: FakeBackend,
    planner: FakePlanner,
    *,
    continuous_drive_handoff: bool = False,
):
    return adapter.StrictNavigationExecutor(
        backend,
        planner,
        sleep=lambda _seconds: None,
        continuous_drive_handoff=continuous_drive_handoff,
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


def test_continuous_drive_handoff_skips_only_intermediate_idle_wait() -> None:
    backend = FakeBackend()
    planner = FakePlanner()

    result = executor(
        backend,
        planner,
        continuous_drive_handoff=True,
    ).navigate(runner.PlanarPose(1.0, 0.0, 0.2))

    assert result.final_pose.x == pytest.approx(1.0)
    assert backend.sent == [
        ("Spin", 0.2),
        ("DriveOnHeading", 1.0),
    ]
    # The opt-in profile takes the whole one-metre reviewed segment in one
    # official action.  The Spin and terminal Drive retain the ordinary
    # balanced-IDLE gate.
    assert backend.health_calls == 2


def test_continuous_drive_handoff_only_skips_an_intermediate_chunk() -> None:
    backend = FakeBackend()

    class ThreeMetrePlanner(FakePlanner):
        def resolve_live_primitive(self, plan, index, navfn_path, pose, **kwargs):
            if index == 0:
                return super().resolve_live_primitive(
                    plan, index, navfn_path, pose, **kwargs
                )
            remaining = max(0.0, 3.0 - float(pose["x"]))
            if remaining <= runner.DETOUR_ENDPOINT_TOLERANCE_M:
                return {"disposition": "skip_within_endpoint_tolerance"}
            return {
                "disposition": "send_action",
                "action_type": "nav2_msgs/action/DriveOnHeading",
                "action_goal": {
                    "target": {"x": remaining, "y": 0.0, "z": 0.0},
                    "speed_mps": runner.OFFICIAL_DRIVE_SPEED_MPS,
                },
                "fresh_distance_to_endpoint_m": remaining,
            }

    result = executor(
        backend,
        ThreeMetrePlanner(),
        continuous_drive_handoff=True,
    ).navigate(runner.PlanarPose(3.0, 0.0, 0.2))

    assert result.final_pose.x == pytest.approx(3.0)
    assert backend.sent == [
        ("Spin", 0.2),
        ("DriveOnHeading", 2.0),
        ("DriveOnHeading", 1.0),
    ]
    # Spin + terminal Drive only.  The first 2 m action is handed directly to
    # the next same-direction chunk without a blocking learned-IDLE wait.
    assert backend.health_calls == 2


@pytest.mark.parametrize("clear", [True, False])
def test_continuous_profile_uses_extra_map_clearance_gate(clear: bool) -> None:
    class FakeMap:
        @classmethod
        def load(cls, _yaml_path, _image_path):
            return cls()

        def inflated_blocked_cells(self, _radius):
            return frozenset()

        def segment_is_clear(self, _start, _end, _blocked):
            return clear

    class ProductionLikePlanner(FakePlanner):
        StaticMap = FakeMap

    backend = FakeBackend()
    planner = ProductionLikePlanner()
    subject = executor(backend, planner, continuous_drive_handoff=True)
    plan = {
        "segments": [
            {
                "id": "segment-000",
                "start_map": {"x": 0.0, "y": 0.0},
                "end_map": {"x": 1.8, "y": 0.0},
                "heading_map_yaw_rad": 0.0,
            }
        ]
    }

    allowed = subject._continuous_segment_allowed(
        plan,
        {"segment_id": "segment-000"},
    )

    assert allowed is clear


def test_fresh_geometry_failure_replans_at_most_twice() -> None:
    backend = FakeBackend()
    planner = FakePlanner(fail_first_resolve=True)

    result = executor(backend, planner).navigate(
        runner.PlanarPose(1.0, 0.0, 0.2)
    )

    assert result.plan_attempts == 2
    assert backend.compute_calls == 2
    assert result.plan_sha256 == "plan-2"


def test_replan_finishes_inside_endpoint_before_unpartitionable_micro_route() -> None:
    backend = FakeBackend()

    class NearGoalReplanPlanner(FakePlanner):
        def build_primitive_plan(self, navfn_path, **kwargs):
            if self.plan_number == 1:
                raise ValueError(
                    "NavFn path cannot be partitioned into collision-free "
                    "distance-bounded segments"
                )
            return super().build_primitive_plan(navfn_path, **kwargs)

        def resolve_live_primitive(self, plan, index, navfn_path, pose, **kwargs):
            backend.pose = runner.PlanarPose(0.82, 0.0, 0.0)
            raise RuntimeError("fresh chord requires a bounded replan")

    result = executor(backend, NearGoalReplanPlanner()).navigate(
        runner.PlanarPose(1.0, 0.0, 0.5)
    )

    assert len(result.plan_sha256) == 64
    assert result.plan_attempts == 2
    assert result.completed_primitives == 0
    assert result.final_pose == runner.PlanarPose(0.82, 0.0, 0.0)
    assert backend.compute_calls == 2
    assert backend.sent == []


def test_navigate_already_inside_physical_endpoint_sends_no_micro_motion() -> None:
    backend = FakeBackend()
    backend.pose = runner.PlanarPose(0.82, 0.0, -1.0)

    result = executor(backend, FakePlanner()).navigate(
        runner.PlanarPose(1.0, 0.0, 1.5)
    )

    assert len(result.plan_sha256) == 64
    assert result.plan_attempts == 1
    assert result.completed_primitives == 0
    assert result.final_pose == backend.pose
    assert backend.compute_calls == 1
    assert backend.sent == []


def test_navigate_executes_forward_only_start_clearance_recovery_first() -> None:
    backend = FakeBackend()

    class RecoveringPlanner(FakePlanner):
        def resolve_start_clearance_recovery(self, _pose, **_kwargs):
            return {
                "disposition": "send_action",
                "action_type": "nav2_msgs/action/DriveOnHeading",
                "action_goal": {
                    "target": {"x": 0.30, "y": 0.0, "z": 0.0},
                    "speed_mps": runner.OFFICIAL_DRIVE_SPEED_MPS,
                },
            }

    result = executor(backend, RecoveringPlanner()).navigate(
        runner.PlanarPose(1.0, 0.0, 0.0)
    )

    assert result.completed_primitives == 4
    assert backend.sent == [
        ("DriveOnHeading", pytest.approx(0.30)),
        ("Spin", pytest.approx(0.20)),
        ("DriveOnHeading", pytest.approx(0.50)),
        ("DriveOnHeading", pytest.approx(0.20)),
    ]


def test_execute_resolved_preserves_bounded_reverse_drive_direction() -> None:
    backend = FakeBackend()
    subject = executor(backend, FakePlanner())

    kind = subject._execute_resolved(
        {
            "action_type": "nav2_msgs/action/DriveOnHeading",
            "action_goal": {
                "target": {"x": -0.70, "y": 0.0, "z": 0.0},
                "speed_mps": runner.OFFICIAL_REVERSE_SPEED_MPS,
            },
        },
        cancelled=lambda: False,
    )

    assert kind == "DriveOnHeading"
    assert backend.sent == [("DriveOnHeading", pytest.approx(-0.50))]
    assert backend.started == [("detour", "DriveOnHeading", -1)]


def test_replan_partition_failure_outside_physical_endpoint_still_aborts() -> None:
    backend = FakeBackend()

    class FarReplanPlanner(FakePlanner):
        def build_primitive_plan(self, navfn_path, **kwargs):
            if self.plan_number == 1:
                raise ValueError("route remains unpartitionable")
            return super().build_primitive_plan(navfn_path, **kwargs)

        def resolve_live_primitive(self, plan, index, navfn_path, pose, **kwargs):
            backend.pose = runner.PlanarPose(0.40, 0.0, 0.0)
            raise RuntimeError("fresh chord requires a bounded replan")

    with pytest.raises(ValueError, match="route remains unpartitionable"):
        executor(backend, FarReplanPlanner()).navigate(
            runner.PlanarPose(1.0, 0.0, 0.5)
        )

    assert backend.compute_calls == 2
    assert backend.sent == []
    assert backend.cancel_calls == 1


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
    assert adapter.DEFAULT_COMMAND_TIMEOUT_S == 2_400.0
