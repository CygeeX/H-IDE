from __future__ import annotations

from pathlib import Path

import yaml


def test_goal_checker_revalidates_xy_for_every_mission_action() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "docker" / "nav2" / "nav2_params.yaml"
    )
    params = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    controller = params["controller_server"]["ros__parameters"]

    assert controller["goal_checker_plugins"] == ["general_goal_checker"]
    assert controller["general_goal_checker"]["stateful"] is False
    assert controller["general_goal_checker"]["xy_goal_tolerance"] == 0.15
    # This is not a declared DWB parameter and must not be mistaken for the
    # actual SimpleGoalChecker latch above.
    assert "stateful" not in controller["FollowPath"]


def test_controller_backports_fail_closed_goal_transform() -> None:
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    patch = (
        root / "docker" / "nav2" / "controller_server_tf_fail_closed.patch"
    ).read_text(encoding="utf-8")

    # Pin the exact Jazzy source that matches the installed 1.3.12 packages.
    assert "6be3614013ec586051b86c97b919b293281490fe" in dockerfile
    assert "git -C /opt/navigation2 apply --check" in dockerfile
    # A TF failure must throw before an all-zero PoseStamped reaches the goal
    # checker and becomes a false success near the map origin.
    assert "if (!nav_2d_utils::transformPose(" in patch
    assert "throw nav2_core::ControllerTFError" in patch


def test_planner_never_silently_substitutes_an_approved_goal() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "docker" / "nav2" / "nav2_params.yaml"
    )
    params = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    planner = params["planner_server"]["ros__parameters"]["GridBased"]

    assert planner["tolerance"] == 0.0


def test_controller_speed_does_not_exceed_mission_envelope() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "docker" / "nav2" / "nav2_params.yaml"
    )
    params = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    follow_path = params["controller_server"]["ros__parameters"]["FollowPath"]

    assert follow_path["max_vel_x"] == 0.20
    assert follow_path["max_speed_xy"] == 0.20
    assert follow_path["min_vel_x"] == 0.0
    assert follow_path["min_speed_xy"] == 0.10
    assert follow_path["min_speed_theta"] == 0.05
    assert follow_path["max_vel_theta"] == 0.25
