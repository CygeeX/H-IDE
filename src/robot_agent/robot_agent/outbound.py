"""Bounded priority queue with latest-state coalescing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class OutboundPriority(IntEnum):
    SAFETY = 0
    CONTROL = 10
    COMMAND = 20
    SNAPSHOT = 30
    STATE = 40


class QueueSaturated(RuntimeError):
    """Raised when a critical event cannot be enqueued without losing one."""


@dataclass(frozen=True)
class OutboundItem:
    priority: OutboundPriority
    kind: str
    payload: Any
    coalesce_key: str | None = None


class PriorityOutboundQueue:
    """A small asyncio queue where safety always overtakes telemetry."""

    def __init__(self, *, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self._maxsize = maxsize
        self._condition = asyncio.Condition()
        self._items: list[tuple[int, OutboundItem]] = []
        self._sequence = 0
        self._dropped_count = 0
        self._coalesced_count = 0

    async def put(self, item: OutboundItem) -> bool:
        async with self._condition:
            if item.coalesce_key is not None:
                for index, (_, existing) in enumerate(self._items):
                    if existing.coalesce_key == item.coalesce_key:
                        self._sequence += 1
                        self._items[index] = (self._sequence, item)
                        self._coalesced_count += 1
                        self._condition.notify()
                        return True

            if len(self._items) >= self._maxsize:
                candidates = [
                    (index, existing.priority)
                    for index, (_, existing) in enumerate(self._items)
                    if existing.priority > item.priority
                ]
                if candidates:
                    evict_index = max(candidates, key=lambda candidate: candidate[1])[0]
                    self._items.pop(evict_index)
                    self._dropped_count += 1
                elif item.priority <= OutboundPriority.COMMAND:
                    raise QueueSaturated(
                        f"outbound queue cannot admit critical {item.kind}; "
                        "safety/control/command delivery is saturated"
                    )
                else:
                    self._dropped_count += 1
                    return False

            self._sequence += 1
            self._items.append((self._sequence, item))
            self._condition.notify()
            return True

    async def get(self) -> OutboundItem:
        async with self._condition:
            await self._condition.wait_for(lambda: bool(self._items))
            index = min(
                range(len(self._items)),
                key=lambda item_index: (
                    self._items[item_index][1].priority,
                    self._items[item_index][0],
                ),
            )
            _, item = self._items.pop(index)
            return item

    @property
    def qsize(self) -> int:
        return len(self._items)

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def coalesced_count(self) -> int:
        return self._coalesced_count
