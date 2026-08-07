"""Dependency-free subset of the systemd notification protocol."""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Mapping


def notify(message: str, *, environment: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environment is None else environment
    address = env.get("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]
    payload = message.encode("utf-8")
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notifier:
        notifier.connect(address)
        notifier.sendall(payload)
    return True


def watchdog_interval(
    *, environment: Mapping[str, str] | None = None, process_id: int | None = None
) -> float | None:
    env = os.environ if environment is None else environment
    raw_usec = env.get("WATCHDOG_USEC")
    if not raw_usec:
        return None
    pid = os.getpid() if process_id is None else process_id
    raw_pid = env.get("WATCHDOG_PID")
    if raw_pid:
        try:
            if int(raw_pid) != pid:
                return None
        except ValueError:
            return None
    try:
        seconds = int(raw_usec) / 1_000_000.0
    except ValueError:
        return None
    if seconds <= 0:
        return None
    return seconds / 2.0


async def watchdog_loop(stop: asyncio.Event) -> None:
    interval = watchdog_interval()
    if interval is None:
        return
    while not stop.is_set():
        notify("WATCHDOG=1")
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            continue
