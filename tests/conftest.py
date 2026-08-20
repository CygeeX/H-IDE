from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

import hide_mujoco_sonic.sonic_driver as sonic_driver


PlantWriteCall = tuple[str, tuple[Any, ...], dict[str, Any]]


@pytest.fixture(autouse=True)
def block_live_driver_plant_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[PlantWriteCall]]:
    """Fail closed unless a test explicitly supplies its Plant transport.

    Driver unit/integration fixtures are intentionally incomplete and must
    never inherit the production loopback Plant endpoint.  Tests that exercise
    a Plant transition replace the relevant module-level transport with a
    deterministic fake after this autouse fixture is installed.
    """

    calls: list[PlantWriteCall] = []

    def blocked(name: str) -> Callable[..., Any]:
        def reject(*args: Any, **kwargs: Any) -> Any:
            calls.append((name, args, kwargs))
            raise AssertionError(
                f"test attempted unmocked live Plant write via {name}"
            )

        return reject

    for name in (
        "latch_plant_physics_fault",
        "rearm_plant_support",
        "set_plant_support",
    ):
        monkeypatch.setattr(sonic_driver, name, blocked(name))
    yield calls
    if calls:
        names = ", ".join(call[0] for call in calls)
        pytest.fail(f"test left unacknowledged blocked Plant writes: {names}")
