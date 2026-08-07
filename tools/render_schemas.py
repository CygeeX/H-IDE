#!/usr/bin/env python3
"""Render canonical Pydantic schemas into the ROS package data directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_agent.protocol import protocol_json_schema

from robot_agent.config import config_json_schema


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/robot_agent/schemas"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schemas = {
        "robot-agent.v1.schema.json": protocol_json_schema(),
        "robot-agent-config.v1.schema.json": config_json_schema(),
    }
    for filename, schema in schemas.items():
        target = args.output / filename
        target.write_text(
            json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
