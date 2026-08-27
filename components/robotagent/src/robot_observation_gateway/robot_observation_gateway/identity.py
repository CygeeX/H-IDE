"""Stable robot boot identity and per-process Gateway instance identity."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

LINUX_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def read_robot_boot_id(path: Path = LINUX_BOOT_ID_PATH) -> str:
    raw = path.read_text(encoding="ascii").strip()
    return str(UUID(raw))


def new_gateway_instance_id() -> str:
    return str(uuid4())
