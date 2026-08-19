from __future__ import annotations

import asyncio
import math
import threading
from pathlib import Path
from uuid import uuid4

import pytest
import rclpy

pytest.importorskip("nav2_msgs.action")

from rclpy.context import Context
from rclpy.executors import MultiThreadedExecutor

from robot_agent.config import load_config
from robot_agent.observation import adapter_params
from robot_agent.ros_adapter import RosAdapter
from tools.mock_robot import MockRobot

WORKSPACE = Path(__file__).resolve().parents[1]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_h_ide_map_target_reaches_real_navigate_to_pose_mock() -> None:
    """No physical command publisher exists in this graph; the Nav2 goal is simulated."""
    context = Context()
    rclpy.init(context=context)
    mock = MockRobot(context=context)
    executor = MultiThreadedExecutor(num_threads=4, context=context)
    executor.add_node(mock)
    running = threading.Event()
    running.set()

    def spin() -> None:
        while running.is_set():
            executor.spin_once(timeout_sec=0.1)

    thread = threading.Thread(target=spin, name="mock-nav2-executor", daemon=True)
    thread.start()
    feedback: list[dict[str, object]] = []
    adapter = RosAdapter(
        feedback_callback=lambda _command_id, value: feedback.append(value)
    )
    adapter.start()
    try:
        config = load_config(
            WORKSPACE / "src/robot_agent/config/robot_agent.local.yaml"
        )
        command = config.commands["navigation.navigate_to_pose"]
        params = {
            "payloadType": "visual_resolution",
            "goalPose": {
                "frameId": "map",
                "x": 1.25,
                "y": -0.5,
                "z": 0.0,
                "yaw": math.pi / 2,
                "observedAt": "2026-08-12T12:00:00Z",
            },
            "goalPurpose": "approach",
            "missionId": "mission-audit-only",
            "stageId": "stage-audit-only",
            "visualResolutionId": "resolution-audit-only",
            "observationSessionId": "observation-audit-only",
            "robotBootId": "boot-audit-only",
            "localizationEpochId": "epoch-audit-only",
            "maximumSpeedMps": 0.3,
            "sceneRef": None,
        }
        local_params = adapter_params(command, params)
        assert local_params == {"goalPose": params["goalPose"]}

        result = await asyncio.wait_for(
            adapter.execute(uuid4(), command, local_params),
            timeout=5.0,
        )

        assert result.succeeded is True
        assert result.result is not None
        assert result.result["rosGoalId"] is not None
        assert isinstance(result.result["actionResult"], dict)
        assert mock.navigation_goal_count == 1
        assert mock.last_navigation_goal is not None
        goal = mock.last_navigation_goal
        assert goal.pose.header.frame_id == "map"
        assert goal.pose.header.stamp.sec > 0
        assert goal.pose.pose.position.x == pytest.approx(1.25)
        assert goal.pose.pose.position.y == pytest.approx(-0.5)
        assert goal.pose.pose.position.z == pytest.approx(0.0)
        assert goal.pose.pose.orientation.z == pytest.approx(math.sqrt(0.5))
        assert goal.pose.pose.orientation.w == pytest.approx(math.sqrt(0.5))
        assert goal.behavior_tree == ""
        assert feedback
        assert feedback[-1]["distance_remaining"] == pytest.approx(0.0)
    finally:
        await adapter.close()
        running.clear()
        executor.wake()
        thread.join(timeout=2.0)
        executor.shutdown(timeout_sec=2.0)
        mock.close()
        if context.ok():
            context.shutdown()
