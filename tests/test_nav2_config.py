from __future__ import annotations

import json
import math
from pathlib import Path
import importlib.util
import xml.etree.ElementTree as ET

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _params() -> dict:
    return yaml.safe_load(
        (ROOT / "docker" / "nav2" / "nav2_params.yaml").read_text(
            encoding="utf-8"
        )
    )


def _runtime_verifier():
    path = ROOT / "docker" / "nav2" / "verify_nav2_runtime.py"
    spec = importlib.util.spec_from_file_location("verify_nav2_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_goal_checker_revalidates_xy_for_every_mission_action() -> None:
    params = _params()
    controller = params["controller_server"]["ros__parameters"]

    assert controller["goal_checker_plugins"] == ["general_goal_checker"]
    assert controller["general_goal_checker"]["stateful"] is False
    assert controller["general_goal_checker"]["xy_goal_tolerance"] == 0.15


def test_controller_backports_fail_closed_goal_transform() -> None:
    dockerfile = (ROOT / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    patch = (
        ROOT / "docker" / "nav2" / "controller_server_tf_fail_closed.patch"
    ).read_text(encoding="utf-8")

    # Pin the exact Jazzy source that matches the installed 1.3.12 packages.
    assert "6be3614013ec586051b86c97b919b293281490fe" in dockerfile
    assert "git -C /opt/navigation2 apply --check" in dockerfile
    # A TF failure must throw before an all-zero PoseStamped reaches the goal
    # checker and becomes a false success near the map origin.
    assert "if (!nav_2d_utils::transformPose(" in patch
    assert "throw nav2_core::ControllerTFError" in patch


def test_planner_never_silently_substitutes_an_approved_goal() -> None:
    params = _params()
    planner = params["planner_server"]["ros__parameters"]["GridBased"]

    assert planner["tolerance"] == 0.0


def test_controller_uses_forward_only_rotation_shim_and_graceful_profile() -> None:
    params = _params()
    follow_path = params["controller_server"]["ros__parameters"]["FollowPath"]

    assert follow_path["plugin"] == (
        "nav2_rotation_shim_controller::RotationShimController"
    )
    assert follow_path["primary_controller"] == (
        "nav2_graceful_controller::GracefulController"
    )
    assert follow_path["closed_loop"] is True
    assert follow_path["rotate_to_goal_heading"] is True
    assert follow_path["initial_rotation"] is False
    assert follow_path["allow_backward"] is False
    assert math.isclose(follow_path["angular_dist_threshold"], math.radians(20.0))
    assert math.isclose(
        follow_path["angular_disengage_threshold"], math.radians(10.0)
    )
    assert follow_path["forward_sampling_distance"] == 0.6

    # This is the one admissible regular walking envelope. Graceful may lower
    # vx to preserve its angular cap on a tighter curve, so the runtime guard
    # must fail closed on any nonzero sub-floor command.
    assert follow_path["v_linear_min"] == 0.20
    assert follow_path["v_linear_max"] == 0.20
    assert follow_path["v_angular_max"] == 0.25
    assert follow_path["min_lookahead"] == 0.25
    assert follow_path["max_lookahead"] == 0.70
    assert follow_path["k_phi"] == 1.5
    assert follow_path["k_delta"] == 0.5
    assert follow_path["prefer_final_rotation"] is True
    assert follow_path["rotation_scaling_factor"] * math.pi * follow_path[
        "v_angular_max"
    ] <= follow_path["v_angular_max"]
    assert follow_path["v_angular_min_in_place"] <= follow_path["v_angular_max"]

    dwb_only_parameters = {
        "max_vel_x",
        "max_vel_y",
        "vx_samples",
        "vy_samples",
        "vtheta_samples",
        "critics",
    }
    assert dwb_only_parameters.isdisjoint(follow_path)

    rpp_only_parameters = {
        "desired_linear_vel",
        "lookahead_dist",
        "use_regulated_linear_velocity_scaling",
        "regulated_linear_scaling_min_speed",
        "allow_reversing",
    }
    assert rpp_only_parameters.isdisjoint(follow_path)


def test_default_navigation_trees_and_behavior_server_forbid_reverse_recovery(
) -> None:
    params = _params()
    navigator = params["bt_navigator"]["ros__parameters"]
    behavior_server = params["behavior_server"]["ros__parameters"]
    tree_dir = ROOT / "docker" / "nav2" / "behavior_trees"
    expected = {
        "default_nav_to_pose_bt_xml": (
            "/opt/h-ide-nav2/behavior_trees/navigate_to_pose_no_reverse.xml",
            tree_dir / "navigate_to_pose_no_reverse.xml",
            "ComputePathToPose",
        ),
        "default_nav_through_poses_bt_xml": (
            "/opt/h-ide-nav2/behavior_trees/navigate_through_poses_no_reverse.xml",
            tree_dir / "navigate_through_poses_no_reverse.xml",
            "ComputePathThroughPoses",
        ),
    }

    for parameter, (container_path, source_path, planner_node) in expected.items():
        assert navigator[parameter] == container_path
        root = ET.parse(source_path).getroot()
        tags = {node.tag for node in root.iter()}
        assert planner_node in tags
        assert "FollowPath" in tags
        assert "BackUp" not in tags
        assert all(
            "backup_" not in value
            for node in root.iter()
            for value in node.attrib.values()
        )

    assert behavior_server["behavior_plugins"] == [
        "spin",
        "drive_on_heading",
        "wait",
    ]
    assert "backup" not in behavior_server

    # Exercise the same BT verifier that runs in the image build. Plugin
    # registration and shared-library closure are checked against /opt/ros in
    # that build layer, where the selected binary packages are installed.
    _runtime_verifier().verify_behavior_trees(tree_dir)


def test_robot_agent_readiness_nodes_remain_in_launch_and_lifecycle() -> None:
    params = _params()
    lifecycle_nodes = params["lifecycle_manager_navigation"]["ros__parameters"][
        "node_names"
    ]
    launch = (ROOT / "docker" / "nav2" / "nav2_sim_launch.py").read_text(
        encoding="utf-8"
    )

    assert lifecycle_nodes == [
        "map_server",
        "planner_server",
        "controller_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
    ]
    assert 'package="nav2_bt_navigator"' in launch
    assert 'executable="bt_navigator"' in launch
    assert 'package="nav2_waypoint_follower"' in launch
    assert 'executable="waypoint_follower"' in launch


def test_strict_runner_is_the_only_possible_bridge_cmd_vel_publisher() -> None:
    launch = (ROOT / "docker" / "nav2" / "nav2_sim_launch.py").read_text(
        encoding="utf-8"
    )
    manifest = json.loads(
        (ROOT / "docker" / "nav2" / "motion_profile_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert '("cmd_vel", "/h_ide/strict_motion/input")' in launch
    assert '("cmd_vel", "/h_ide/unaccepted_follow_path/cmd_vel")' in launch
    assert manifest["cmd_vel_ownership"] == {
        "bridge_topic": "/cmd_vel",
        "strict_runner_node": "/strict_motion_runner",
        "nav2_behavior_input_topic": "/h_ide/strict_motion/input",
        "unaccepted_follow_path_topic": "/h_ide/unaccepted_follow_path/cmd_vel",
        "required_bridge_publisher_count": 1,
    }


def test_nav2_image_installs_and_copies_the_selected_controller_contract() -> None:
    dockerfile = (ROOT / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "ros-jazzy-nav2-graceful-controller" in dockerfile
    assert "ros-jazzy-nav2-rotation-shim-controller" in dockerfile
    assert "ros-jazzy-nav2-waypoint-follower" in dockerfile
    # The bringup meta-package pulls simulation, RViz, SLAM, and every stock
    # controller into an otherwise isolated runtime. Every launched node is
    # listed explicitly instead.
    assert "ros-jazzy-nav2-bringup" not in dockerfile
    assert "ros-jazzy-nav2-dwb-controller" not in dockerfile
    assert "ros-jazzy-dwb-plugins" not in dockerfile
    assert "ros-jazzy-nav2-regulated-pure-pursuit-controller" not in dockerfile
    assert (
        "COPY docker/nav2/behavior_trees/ /opt/h-ide-nav2/behavior_trees/"
        in dockerfile
    )
    assert "ros2 pkg prefix nav2_rotation_shim_controller" in dockerfile
    assert "ros2 pkg prefix nav2_graceful_controller" in dockerfile
    assert "python3 /opt/h-ide-nav2/verify_nav2_runtime.py" in dockerfile
    assert "--motion-profile-manifest" in dockerfile


def test_official_primitive_actions_share_the_reviewed_sonic_envelope() -> None:
    params = _params()
    behavior = params["behavior_server"]["ros__parameters"]
    manifest = json.loads(
        (ROOT / "docker" / "nav2" / "motion_profile_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["primitive_action_contract"] == {
        "drive_on_heading_speed_mps": 0.2,
        "spin_min_angular_rps": 0.1,
        "spin_max_angular_rps": 0.2,
        "spin_angular_accel_rps2": 0.3,
    }
    assert behavior["min_rotational_vel"] == 0.1
    assert behavior["max_rotational_vel"] == 0.2
    assert behavior["rotational_acc_lim"] == 0.3
