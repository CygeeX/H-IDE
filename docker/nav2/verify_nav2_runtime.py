#!/usr/bin/env python3
"""Fail closed if the selected Nav2 plugins or behavior trees are not loadable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


PLUGINS = {
    "nav2_rotation_shim_controller": (
        "nav2_rotation_shim_controller.xml",
        "nav2_rotation_shim_controller",
        "nav2_rotation_shim_controller::RotationShimController",
    ),
    "nav2_graceful_controller": (
        "graceful_controller_plugin.xml",
        "nav2_graceful_controller",
        "nav2_graceful_controller::GracefulController",
    ),
}

BEHAVIOR_TREES = {
    "navigate_to_pose_no_reverse.xml": "ComputePathToPose",
    "navigate_through_poses_no_reverse.xml": "ComputePathThroughPoses",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_plugin(
    ros_prefix: Path,
    package: str,
    descriptor_filename: str,
    library_name: str,
    class_name: str,
) -> None:
    descriptor_relative = Path("share") / package / descriptor_filename
    descriptor = ros_prefix / descriptor_relative
    marker = (
        ros_prefix
        / "share"
        / "ament_index"
        / "resource_index"
        / "nav2_core__pluginlib__plugin"
        / package
    )

    _require(descriptor.is_file(), f"missing plugin descriptor: {descriptor}")
    _require(marker.is_file(), f"missing pluginlib index marker: {marker}")
    indexed_descriptors = {
        Path(line.strip())
        for line in marker.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    _require(
        descriptor_relative in indexed_descriptors,
        f"{descriptor_relative} is not registered in {marker}",
    )

    root = ET.parse(descriptor).getroot()
    libraries = [
        element
        for element in root.findall("library")
        if element.attrib.get("path") == library_name
    ]
    _require(libraries, f"{descriptor} does not export library {library_name}")
    exported_classes = [
        element
        for library in libraries
        for element in library.findall("class")
        if element.attrib.get("type") == class_name
        and element.attrib.get("base_class_type") == "nav2_core::Controller"
    ]
    _require(exported_classes, f"{descriptor} does not export {class_name}")

    library = ros_prefix / "lib" / f"lib{library_name}.so"
    _require(library.is_file(), f"missing plugin library: {library}")
    result = subprocess.run(
        ["ldd", str(library)],
        check=False,
        capture_output=True,
        text=True,
    )
    dependencies = result.stdout + result.stderr
    _require(result.returncode == 0, f"ldd failed for {library}: {dependencies}")
    _require("not found" not in dependencies, f"unresolved dependency: {dependencies}")


def verify_behavior_trees(tree_dir: Path) -> None:
    for filename, planner_node in BEHAVIOR_TREES.items():
        path = tree_dir / filename
        _require(path.is_file(), f"missing behavior tree: {path}")
        root = ET.parse(path).getroot()
        tags = {node.tag for node in root.iter()}
        _require(planner_node in tags, f"{path} is missing {planner_node}")
        _require("FollowPath" in tags, f"{path} is missing FollowPath")
        _require("BackUp" not in tags, f"{path} enables BackUp")
        _require(
            all(
                "backup_" not in value
                for node in root.iter()
                for value in node.attrib.values()
            ),
            f"{path} references a backup recovery port",
        )


def verify_motion_profile(path: Path) -> None:
    profile = json.loads(path.read_text(encoding="utf-8"))
    _require(
        profile.get("schema_version") == "h-ide.nav2-motion-profile/1",
        "unsupported motion profile manifest",
    )
    _require(
        profile.get("follow_path_motion_accepted") is False,
        "unvalidated FollowPath motion must remain disabled",
    )
    _require(
        profile.get("cmd_vel_ownership")
        == {
            "bridge_topic": "/cmd_vel",
            "strict_runner_node": "/strict_motion_runner",
            "nav2_behavior_input_topic": "/h_ide/strict_motion/input",
            "unaccepted_follow_path_topic": (
                "/h_ide/unaccepted_follow_path/cmd_vel"
            ),
            "required_bridge_publisher_count": 1,
        },
        "strict runner is not the sole reviewed /cmd_vel owner",
    )
    _require(
        profile.get("allowed_product_actions")
        == [
            "nav2_msgs/action/ComputePathToPose",
            "nav2_msgs/action/Spin",
            "nav2_msgs/action/DriveOnHeading",
        ],
        "product action allowlist is not the reviewed primitive chain",
    )
    _require(
        profile.get("primitive_action_contract")
        == {
            "drive_on_heading_speed_mps": 0.3,
            "spin_min_angular_rps": 0.1,
            "spin_max_angular_rps": 0.2,
            "spin_angular_accel_rps2": 0.3,
        },
        "official primitive action limits do not match the reviewed envelope",
    )
    forbidden = set(profile.get("forbidden_product_actions", []))
    _require(
        {
            "nav2_msgs/action/NavigateToPose",
            "nav2_msgs/action/FollowPath",
            "nav2_msgs/action/BackUp",
        }.issubset(forbidden),
        "unaccepted product actions are not fail-closed",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ros-prefix", type=Path, required=True)
    parser.add_argument("--behavior-tree-dir", type=Path, required=True)
    parser.add_argument("--motion-profile-manifest", type=Path, required=True)
    args = parser.parse_args()

    for package, values in PLUGINS.items():
        verify_plugin(args.ros_prefix, package, *values)
    verify_behavior_trees(args.behavior_tree_dir)
    verify_motion_profile(args.motion_profile_manifest)
    print(
        "Nav2 plugins, behavior trees, and primitive-only motion profile verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
