"""Loopback ROS safe-stop service for the isolated simulation profile.

Robot Agent requires a local safe-stop endpoint during startup and whenever a
lease expires.  A successful response means both layers have been commanded:
SONIC intent is zeroed and the MuJoCo plant has engaged bounded translational
and attitude support.  The service fails closed when either operation is
rejected.  A separately latched motion-safety fault uses the terminal Plant
halt endpoint and can never be reported as a successful recoverable stop.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

from .config import SimConfig
from .transport import request_json, send_json, set_plant_support


def execute_safe_stop(
    config: SimConfig,
    *,
    send_command=send_json,
    confirm_command=request_json,
    set_support=set_plant_support,
) -> str:
    """Stop SONIC intent and engage bounded MuJoCo emergency support."""

    send_command(
        config.command_host,
        config.command_port,
        {
            "type": "stop",
            "source": "robot_agent_safe_stop",
            "reason": "Robot Agent requested simulation safe-stop",
        },
        timeout=0.5,
    )
    set_support("127.0.0.1", config.plant_port, "safety_stop", timeout=0.5)
    confirm_command(
        config.command_host,
        config.command_port,
        {
            "type": "plant_support_state",
            "schema_version": "h-ide.plant-support-state/v1",
            "source": "robot_agent_safe_stop",
            "mode": "safety_stop",
            "reason": "Robot Agent requested simulation safe-stop",
        },
        timeout=0.5,
    )
    return "SONIC stopped and MuJoCo emergency support engaged"


def run(config: SimConfig) -> None:
    os.environ["ROS_DOMAIN_ID"] = str(config.ros_domain_id)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    if config.interface == "lo":
        os.environ["CYCLONEDDS_URI"] = (
            "<CycloneDDS><Domain><General>"
            '<Interfaces><NetworkInterface name="lo"/></Interfaces>'
            "<AllowMulticast>true</AllowMulticast>"
            "</General></Domain></CycloneDDS>"
        )
    import rclpy
    from std_srvs.srv import Trigger

    rclpy.init(args=None)
    node = rclpy.create_node("h_ide_mujoco_safe_stop")

    def stop(_request, response):
        try:
            response.message = execute_safe_stop(config)
            response.success = True
        except Exception as exc:
            response.success = False
            response.message = f"simulation safe-stop failed: {exc}"
        return response

    node.create_service(Trigger, "/robot/safe_stop", stop)
    node.get_logger().info(
        f"simulation safe-stop service ready on ROS domain {config.ros_domain_id}"
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/ros-headless.yaml")
    args = parser.parse_args(argv)
    run(SimConfig.from_file(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
