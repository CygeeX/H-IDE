#!/usr/bin/env python3
"""Minimal BFF peer for local Robot Agent protocol and lease testing."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from websockets.asyncio.client import connect
from websockets.asyncio.server import ServerConnection, serve

from robot_agent.protocol import (
    CommandRequest,
    CommandRequestPayload,
    CommandStatus,
    ControlAcquire,
    ControlAcquirePayload,
    LeaseRenew,
    LeaseRenewPayload,
    SessionHello,
    SessionWelcome,
    SessionWelcomePayload,
    parse_wire_message,
    serialize_wire_message,
)
from robot_agent.transport import SUBPROTOCOL

TERMINAL_COMMAND_STATES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
        "REJECTED",
        "TIMED_OUT",
        "ABORTED_AGENT_RESTART",
    }
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def envelope_fields(robot_id: str, session_id: UUID) -> dict[str, Any]:
    return {
        "version": "1.0",
        "message_id": uuid4(),
        "robot_id": robot_id,
        "session_id": session_id,
        "sent_at": now(),
    }


def command_reached_terminal_state(message: Any, command_id: UUID) -> bool:
    return (
        isinstance(message, CommandStatus)
        and message.payload.command_id == command_id
        and message.payload.state in TERMINAL_COMMAND_STATES
    )


async def run_session(websocket: Any, args: argparse.Namespace) -> bool:
    hello = parse_wire_message(await websocket.recv(), expected_robot_id=args.robot_id)
    if not isinstance(hello, SessionHello):
        raise RuntimeError("Agent must initiate with session.hello")
    session_id = uuid4()
    welcome = SessionWelcome(
        **envelope_fields(args.robot_id, session_id),
        type="session.welcome",
        payload=SessionWelcomePayload(
            accepted_version="1.0",
            lease_interval_ms=args.lease_interval_ms,
            lease_timeout_ms=args.lease_timeout_ms,
            control_state="SAFE_HOLD",
        ),
    )
    await websocket.send(serialize_wire_message(welcome))
    acquire = ControlAcquire(
        **envelope_fields(args.robot_id, session_id),
        type="control.acquire",
        payload=ControlAcquirePayload(operator_id=args.operator_id),
    )
    await websocket.send(serialize_wire_message(acquire))

    command_id: UUID | None = None
    if args.command_kind:
        params = json.loads(args.params)
        command_id = uuid4()
        command = CommandRequest(
            **envelope_fields(args.robot_id, session_id),
            type="command.request",
            payload=CommandRequestPayload(
                command_id=command_id,
                kind=args.command_kind,
                issued_at=now(),
                ttl_ms=args.command_ttl_ms,
                params=params,
            ),
        )
        await websocket.send(serialize_wire_message(command))

    async def renew_lease() -> None:
        sequence = 0
        while True:
            await asyncio.sleep(args.lease_interval_ms / 1000.0)
            renewal = LeaseRenew(
                **envelope_fields(args.robot_id, session_id),
                type="lease.renew",
                payload=LeaseRenewPayload(sequence=sequence),
            )
            sequence += 1
            await websocket.send(serialize_wire_message(renewal))

    renewer = asyncio.create_task(renew_lease())
    try:
        async for raw in websocket:
            parsed = parse_wire_message(raw, expected_robot_id=args.robot_id)
            print(serialize_wire_message(parsed), flush=True)
            if (
                args.exit_on_command_terminal
                and command_id is not None
                and command_reached_terminal_state(parsed, command_id)
            ):
                return True
    finally:
        renewer.cancel()
        await asyncio.gather(renewer, return_exceptions=True)
    return False


def client_context(args: argparse.Namespace) -> ssl.SSLContext | None:
    if not args.ca_file:
        return None
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=args.ca_file)
    context.load_cert_chain(args.cert_file, args.key_file)
    return context


def server_context(args: argparse.Namespace) -> ssl.SSLContext | None:
    if not args.ca_file:
        return None
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=args.ca_file)
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(args.cert_file, args.key_file)
    return context


async def run(args: argparse.Namespace) -> None:
    if args.mode == "client":
        async with connect(
            args.uri,
            ssl=client_context(args),
            server_hostname=args.server_name,
            subprotocols=[SUBPROTOCOL],
            compression=None,
            proxy=None,
        ) as websocket:
            await run_session(websocket, args)
        return

    terminal_received = asyncio.Event()

    async def handler(websocket: ServerConnection) -> None:
        if await run_session(websocket, args):
            terminal_received.set()

    async with serve(
        handler,
        args.host,
        args.port,
        ssl=server_context(args),
        subprotocols=[SUBPROTOCOL],
        compression=None,
    ):
        if args.exit_on_command_terminal:
            await terminal_received.wait()
        else:
            await asyncio.Future()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("client", "server"), default="server")
    result.add_argument("--uri", default="wss://localhost:8765/agent")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--robot-id", default="g1-dev")
    result.add_argument("--operator-id", default="mock-operator")
    result.add_argument("--lease-interval-ms", type=int, default=500)
    result.add_argument("--lease-timeout-ms", type=int, default=2000)
    result.add_argument("--command-kind")
    result.add_argument("--command-ttl-ms", type=int, default=5000)
    result.add_argument("--params", default="{}")
    result.add_argument(
        "--exit-on-command-terminal",
        action="store_true",
        help="exit after the configured command reaches a terminal status",
    )
    result.add_argument("--ca-file", type=Path)
    result.add_argument("--cert-file", type=Path)
    result.add_argument("--key-file", type=Path)
    result.add_argument("--server-name")
    return result


def main() -> int:
    args = parser().parse_args()
    if bool(args.ca_file) != bool(args.cert_file and args.key_file):
        raise SystemExit("--ca-file, --cert-file, and --key-file must be provided together")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
