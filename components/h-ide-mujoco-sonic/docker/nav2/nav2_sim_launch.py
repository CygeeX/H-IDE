"""Small, explicit Nav2 stack for the isolated MuJoCo domain.

The simulator publishes perfect map/odom/base_link state and a static office
map.  No fake action server is involved: NavigateToPose and Spin are provided
by the stock Nav2 BT navigator and behavior server.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = LaunchConfiguration("params_file")
    map_file = LaunchConfiguration("map")
    autostart = LaunchConfiguration("autostart")
    use_sim_time = LaunchConfiguration("use_sim_time")
    lifecycle_nodes = [
        "map_server",
        "planner_server",
        "controller_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
    ]
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value="/opt/h-ide-nav2/nav2_params.yaml"),
        DeclareLaunchArgument("map", default_value="/opt/h-ide-nav2/nav2/office_v2.yaml"),
        DeclareLaunchArgument("autostart", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="map_server",
            output="screen",
            parameters=[params, {"yaml_filename": map_file, "use_sim_time": use_sim_time}],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[params, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[params, {"use_sim_time": use_sim_time}],
            # FollowPath is an unaccepted calibration candidate.  Even while
            # its local costmap stays active, it must not own the production
            # bridge topic or become a second /cmd_vel publisher.
            remappings=[("cmd_vel", "/h_ide/unaccepted_follow_path/cmd_vel")],
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[params, {"use_sim_time": use_sim_time}],
            # The strict motion runner validates and relays official
            # Spin/DriveOnHeading output.  It is the sole /cmd_vel publisher
            # observed by the SONIC bridge during motion-core acceptance.
            remappings=[("cmd_vel", "/h_ide/strict_motion/input")],
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            parameters=[params, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="nav2_waypoint_follower",
            executable="waypoint_follower",
            name="waypoint_follower",
            output="screen",
            parameters=[params, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[
                params,
                {"use_sim_time": use_sim_time, "autostart": autostart, "node_names": lifecycle_nodes},
            ],
        ),
        ExecuteProcess(
            cmd=["python3", "/opt/h-ide-nav2/nav2_runtime_health.py"],
            name="h_ide_nav2_runtime_health",
            output="screen",
        ),
    ])
