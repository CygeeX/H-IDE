"""Launch the Robot Agent CLI without exposing arbitrary ROS parameters."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import FindExecutable, LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    config = LaunchConfiguration("config")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value="/etc/robot-agent/agent.yaml",
                description="Absolute path to the strict Robot Agent YAML configuration",
            ),
            ExecuteProcess(
                cmd=[
                    FindExecutable(name="robot-agent"),
                    "run",
                    "--config",
                    config,
                ],
                output="screen",
                respawn=False,
            ),
        ]
    )
