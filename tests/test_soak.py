from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from robot_agent.outbound import OutboundItem, OutboundPriority, PriorityOutboundQueue
from robot_agent.state import StateCoalescer


def resident_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    return int(fields[1]) * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.soak
@pytest.mark.asyncio
async def test_state_and_queue_remain_bounded_for_30_minutes() -> None:
    if os.environ.get("ROBOT_AGENT_RUN_SOAK") != "1":
        pytest.skip("set ROBOT_AGENT_RUN_SOAK=1 for the 30-minute acceptance soak")
    duration = float(os.environ.get("ROBOT_AGENT_SOAK_SECONDS", "1800"))
    queue = PriorityOutboundQueue(maxsize=64)
    state = StateCoalescer(change_hz=5.0, snapshot_hz=1.0)
    loop = asyncio.get_running_loop()
    started = loop.time()
    next_tick = started
    start_rss = resident_bytes()
    max_rss = start_rss
    max_queue = 0
    max_lag = 0.0
    iterations = 0

    while loop.time() - started < duration:
        iterations += 1
        observed = datetime.now(timezone.utc)
        state.update("robot.counter", iterations, observed_at=observed)
        for emission in state.poll(monotonic_now=loop.time()):
            await queue.put(
                OutboundItem(
                    priority=(
                        OutboundPriority.SNAPSHOT
                        if emission.kind == "snapshot"
                        else OutboundPriority.STATE
                    ),
                    kind=f"state.{emission.kind}",
                    payload=emission.data,
                    coalesce_key=f"state.{emission.kind}",
                )
            )
        if iterations % 100 == 0:
            await queue.put(
                OutboundItem(
                    priority=OutboundPriority.SAFETY,
                    kind="safety.event",
                    payload={"code": "SOAK_HEARTBEAT"},
                )
            )
        max_queue = max(max_queue, queue.qsize)
        while queue.qsize:
            await queue.get()
        if iterations % 100 == 0:
            max_rss = max(max_rss, resident_bytes())

        next_tick += 0.01
        await asyncio.sleep(max(0.0, next_tick - loop.time()))
        max_lag = max(max_lag, loop.time() - next_tick)

    assert iterations >= int(duration * 50)
    assert max_queue <= 64
    assert queue.dropped_count == 0
    assert max_rss - start_rss < 64 * 1024 * 1024
    assert max_lag < 0.5
