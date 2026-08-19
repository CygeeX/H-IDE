"""Robot identity shared across Agent processes on one Linux boot."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

LINUX_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def read_robot_boot_id(path: Path = LINUX_BOOT_ID_PATH) -> UUID:
    return UUID(path.read_text(encoding="ascii").strip())
