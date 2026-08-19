#!/usr/bin/env python3
"""Local ROS 2 graph used to exercise Robot Agent without robot hardware."""

from __future__ import annotations

import argparse
import threading
import time
from typing import Any

import rclpy
from example_interfaces.action import Fibonacci
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger


def reliable_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class MockRobot(Node):
    def __init__(self, *, context: Context | None = None) -> None:
        super().__init__("robot_agent_mock_robot", context=context)
        group = ReentrantCallbackGroup()
        self._enabled = False
        self.navigation_goal_count = 0
        self.last_navigation_goal: NavigateToPose.Goal | None = None
        self.navigation_goal_event = threading.Event()
        self._mode_publisher = self.create_publisher(
            String,
            "/robot/mode",
            reliable_qos(),
            callback_group=group,
        )
        self.create_timer(0.2, self._publish_mode, callback_group=group)
        self.create_service(
            SetBool,
            "/robot/power_enable",
            self._power_enable,
            qos_profile=reliable_qos(depth=10),
            callback_group=group,
        )
        self.create_service(
            Trigger,
            "/robot/safe_stop",
            self._safe_stop,
            qos_profile=reliable_qos(depth=10),
            callback_group=group,
        )
        self._action_server = ActionServer(
            self,
            Fibonacci,
            "/robot/task_execute",
            execute_callback=self._execute,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self._navigation_action_server = ActionServer(
            self,
            NavigateToPose,
            "/navigate_to_pose",
            execute_callback=self._navigate_to_pose,
            goal_callback=self._accept_navigation_goal,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.get_logger().warning(
            "Mock Nav2 enabled: goals are recorded and simulated; no motion command is published"
        )

    def _power_enable(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self._enabled = request.data
        response.success = True
        response.message = "power enabled" if self._enabled else "power disabled"
        self.get_logger().info(response.message)
        return response

    def _safe_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        self._enabled = False
        response.success = True
        response.message = "mock robot entered safe stop"
        self.get_logger().warning(response.message)
        return response

    def _execute(self, goal_handle: Any) -> Fibonacci.Result:
        sequence = [0, 1]
        for index in range(2, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = Fibonacci.Result()
                result.sequence = sequence
                return result
            sequence.append(sequence[index - 1] + sequence[index - 2])
            feedback = Fibonacci.Feedback()
            feedback.sequence = sequence
            goal_handle.publish_feedback(feedback)
            time.sleep(0.05)
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence
        return result

    def _accept_navigation_goal(self, goal: NavigateToPose.Goal) -> GoalResponse:
        pose = goal.pose
        if pose.header.frame_id != "map" or goal.behavior_tree:
            self.get_logger().error(
                "rejected non-map or planner-mutating NavigateToPose goal"
            )
            return GoalResponse.REJECT
        self.navigation_goal_count += 1
        self.last_navigation_goal = goal
        self.navigation_goal_event.set()
        self.get_logger().info(
            "SIMULATED NavigateToPose received: "
            f"x={pose.pose.position.x:.3f} y={pose.pose.position.y:.3f} "
            f"z={pose.pose.position.z:.3f}"
        )
        return GoalResponse.ACCEPT

    def _navigate_to_pose(self, goal_handle: Any) -> NavigateToPose.Result:
        """Exercise the real Nav2 action type without publishing robot motion."""
        started = self.get_clock().now()
        for step in range(5):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NavigateToPose.Result()
            feedback = NavigateToPose.Feedback()
            feedback.current_pose = goal_handle.request.pose
            feedback.navigation_time = (self.get_clock().now() - started).to_msg()
            feedback.estimated_time_remaining.sec = max(0, 1 - step // 3)
            feedback.number_of_recoveries = 0
            feedback.distance_remaining = float(4 - step) * 0.25
            goal_handle.publish_feedback(feedback)
            time.sleep(0.1)
        goal_handle.succeed()
        return NavigateToPose.Result()

    def _publish_mode(self) -> None:
        message = String()
        message.data = "IDLE" if self._enabled else "SAFE_HOLD"
        self._mode_publisher.publish(message)

    def close(self) -> None:
        self._action_server.destroy()
        self._navigation_action_server.destroy()
        self.destroy_node()


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rclpy.init()
    robot = MockRobot()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(robot)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        robot.close()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
