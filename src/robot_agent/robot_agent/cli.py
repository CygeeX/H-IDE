"""Robot Agent command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from robot_agent.config import ConfigLoadError, load_config
from robot_agent.doctor import collect_diagnostics
from robot_agent.logging_utils import configure_logging
from robot_agent.runtime import RobotAgentRuntime
from robot_agent.systemd_notify import notify, watchdog_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-agent",
        description="Restricted robot-side WebSocket-to-ROS 2 Agent",
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "validate-config", "doctor"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        config = load_config(args.config)
    except ConfigLoadError as exc:
        _print_json({"valid": False, "error": str(exc)}, stream=sys.stderr)
        return 2

    if args.command == "validate-config":
        _print_json(
            {
                "valid": True,
                "robot_id": config.robot_id,
                "transport_mode": config.transport.mode,
                "capabilities_digest": config.capabilities_digest,
                "command_count": len(config.commands),
                "state_source_count": len(config.state_sources),
            }
        )
        return 0
    if args.command == "doctor":
        report = collect_diagnostics(config)
        _print_json(report)
        return 0 if report["healthy"] else 1
    try:
        return asyncio.run(_run(config))
    except KeyboardInterrupt:
        return 130
    except Exception:
        logging.getLogger(__name__).exception("Robot Agent terminated with an error")
        return 1


async def _run(config: Any) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signum, stop.set)

    runtime = RobotAgentRuntime(config)
    runtime_task = asyncio.create_task(runtime.run(stop))
    ready_task = asyncio.create_task(runtime.ready.wait())
    done, _ = await asyncio.wait(
        {runtime_task, ready_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if runtime_task in done:
        ready_task.cancel()
        await asyncio.gather(ready_task, return_exceptions=True)
        await runtime_task
        return 0

    notify("READY=1\nSTATUS=Robot Agent safe core ready")
    watchdog = asyncio.create_task(watchdog_loop(stop))
    try:
        await runtime_task
    finally:
        stop.set()
        watchdog.cancel()
        await asyncio.gather(watchdog, return_exceptions=True)
        notify("STOPPING=1\nSTATUS=Robot Agent stopping")
    return 0


def _print_json(payload: dict[str, Any], *, stream: Any | None = None) -> None:
    output = sys.stdout if stream is None else stream
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=output)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
