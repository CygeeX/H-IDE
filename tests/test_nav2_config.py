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
