"""Capacity-one newest-only slots with explicit overwrite accounting."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    """Thread-safe capacity-one slot; a producer never waits behind stale data."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._item: T | None = None
        self._closed = False
        self._overwritten = 0

    def put(self, item: T) -> bool:
        with self._condition:
            if self._closed:
                raise RuntimeError("latest slot is closed")
            overwritten = self._item is not None
            if overwritten:
                self._overwritten += 1
            self._item = item
            self._condition.notify(1)
            return overwritten

    def pop(self) -> T | None:
        with self._condition:
            item = self._item
            self._item = None
            return item

    def wait_pop(self, timeout: float | None = None) -> T | None:
        """Wait for an item, timeout, or close without ever accumulating a backlog."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._item is None and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            item = self._item
            self._item = None
            return item

    def clear(self) -> bool:
        return self.pop() is not None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._item = None
            self._condition.notify_all()

    @property
    def qsize(self) -> int:
        with self._condition:
            return int(self._item is not None)

    @property
    def overwritten_count(self) -> int:
        with self._condition:
            return self._overwritten


class AsyncLatestSlot(Generic[T]):
    """Async capacity-one slot used between the encoder and the single gRPC writer."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._item: T | None = None
        self._closed = False
        self._overwritten = 0

    async def put(self, item: T) -> bool:
        async with self._condition:
            if self._closed:
                raise RuntimeError("latest slot is closed")
            overwritten = self._item is not None
            if overwritten:
                self._overwritten += 1
            self._item = item
            self._condition.notify(1)
            return overwritten

    async def get(self) -> T:
        async with self._condition:
            await self._condition.wait_for(lambda: self._item is not None or self._closed)
            if self._item is None:
                raise EOFError("latest slot is closed")
            item = self._item
            self._item = None
            return item

    async def clear(self) -> bool:
        async with self._condition:
            present = self._item is not None
            self._item = None
            return present

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            self._item = None
            self._condition.notify_all()

    @property
    def qsize(self) -> int:
        return int(self._item is not None)

    @property
    def overwritten_count(self) -> int:
        return self._overwritten
