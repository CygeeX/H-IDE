"""Launch the official RealSense RGB-D pipeline and the independent gateway sidecar."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config = LaunchConfiguration("gateway_config")
    serial = LaunchConfiguration("serial_no")
    start_gateway = LaunchConfiguration("start_gateway")
    realsense_launch = GroupAction(
        actions=(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
                    )
                ),
                launch_arguments={
                    "camera_namespace": "camera",
                    "camera_name": "camera",
                    "serial_no": serial,
                    "enable_color": "true",
                    "enable_depth": "true",
                    "enable_rgbd": "true",
                    "enable_sync": "true",
                    "align_depth.enable": "true",
                    "rgb_camera.color_profile": "640x480x30",
                    "depth_module.depth_profile": "640x480x30",
                }.items(),
            ),
        ),
        forwarding=False,
        launch_configurations={"serial_no": serial},
    )
    gateway = Node(
        package="robot_observation_gateway",
        executable="robot-observation-gateway",
        name="robot_observation_gateway_process",
        output="screen",
        arguments=("--config", config),
        condition=IfCondition(start_gateway),
    )
    return LaunchDescription(
        (
            DeclareLaunchArgument(
                "gateway_config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("robot_observation_gateway"),
                        "config",
                        "robot_observation_gateway.local.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("serial_no", default_value=""),
            DeclareLaunchArgument(
                "start_gateway",
                default_value="true",
                description=(
                    "Start the observation Gateway. Set false when the visual demo owns "
                    "the local workstation and Gateway."
                ),
            ),
            realsense_launch,
            gateway,
        )
    )
