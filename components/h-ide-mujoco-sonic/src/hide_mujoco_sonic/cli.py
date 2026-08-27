from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from .config import SimConfig


def check(config: SimConfig) -> int:
    errors: list[str] = []
    if config.schema_version != "h-ide.mujoco-sonic-config/1":
        errors.append(f"unsupported config schema: {config.schema_version}")
    if not config.model_scene.exists():
        errors.append(f"MuJoCo scene does not exist: {config.model_scene}")
    vendor = Path(__file__).resolve().parents[2] / "vendor"
    lerobot = vendor / "lerobot"
    if not lerobot.exists():
        errors.append(f"LeRobot vendor directory is missing: {lerobot}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    print("config: ok")
    print(f"robot: {config.robot_id} ({config.robot_profile})")
    print(f"ros domain: {config.ros_domain_id} (isolated from live)")
    print(
        "unitree control domain: "
        f"{config.unitree_domain_id} (isolated from ROS/Nav2)"
    )
    print(f"scene: {config.model_scene}")
    print(f"cmd_vel guard: <= {config.max_linear_mps} m/s, <= {config.max_angular_rps} rad/s, timeout {config.command_timeout_s}s")
    for module in ("numpy", "yaml"):
        try:
            __import__(module)
        except Exception as exc:
            print(f"dependency {module}: missing ({exc})")
        else:
            print(f"dependency {module}: ok")
    for module in ("mujoco", "onnxruntime", "unitree_sdk2py"):
        try:
            __import__(module)
        except Exception as exc:
            print(f"runtime {module}: unavailable ({exc})")
        else:
            print(f"runtime {module}: ok")
    print("No physical actuator is selected by this check.")
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hide-mujoco-sonic")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "plant", "simulator", "sonic", "ros-bridge", "ros-observation-bridge", "safe-stop-service"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="config/sim.yaml")
    args = parser.parse_args(argv)
    config = SimConfig.from_file(args.config)
    if args.command == "check":
        return check(config)
    if args.command in {"plant", "simulator"}:
        # ``simulator`` is retained as a compatibility spelling, but both
        # names now run the one authoritative plant.  The old subprocess
        # wrapper also tried to own the SONIC telemetry port and generated an
        # incomplete temporary config, so keeping two launch paths made the
        # demo dependent on which command an operator happened to remember.
        from .plant import run_plant
        return run_plant(config)
    if args.command == "sonic":
        from .sonic_driver import run_driver
        run_driver(config)
        return 0
    if args.command == "ros-observation-bridge":
        from .ros_observation_bridge import run
        run(config)
        return 0
    if args.command == "safe-stop-service":
        from .safe_stop_service import run
        run(config)
        return 0
    from .ros_bridge import run_ros_bridge
    run_ros_bridge(config)
    return 0
