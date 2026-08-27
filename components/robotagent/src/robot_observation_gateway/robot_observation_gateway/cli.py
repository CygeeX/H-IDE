"""Command-line entry point for the observation gateway sidecar."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from collections.abc import Sequence

from robot_observation_gateway.config import load_config
from robot_observation_gateway.runtime import GatewayRuntime


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Robot RGB-D observation gateway")
    value.add_argument("--config", required=True, help="Path to strict gateway YAML")
    value.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return value


async def _serve(config_path: str) -> None:
    runtime = GatewayRuntime(load_config(config_path))
    loop = asyncio.get_running_loop()
    stop_task: asyncio.Task[None] | None = None

    def request_stop() -> None:
        nonlocal stop_task
        if stop_task is None:
            stop_task = asyncio.create_task(runtime.stop())

    for name in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(name, request_stop)
    await runtime.run()


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_serve(args.config))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
