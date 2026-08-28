from __future__ import annotations

from uuid import uuid4

import pytest

from hide_mujoco_sonic.gesture import GestureError, GestureWaveController


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


def test_wave_is_bounded_and_returns_to_base() -> None:
    clock = _Clock()
    controller = GestureWaveController(clock=clock)
    request_id = str(uuid4())
    controller.request_wave(
        request_id=request_id,
        gesture="wave",
        duration_s=2.4,
        simulation_only=True,
        ready=True,
    )
    base = tuple([0.0] * 29)
    clock.value = 0.8
    sampled, snapshot = controller.sample(base)
    assert snapshot.state == "running"
    assert snapshot.progress == pytest.approx(1 / 3)
    assert sampled != base
    assert max(abs(sampled[index]) for index in range(29)) <= 0.30
    clock.value = 3.0
    sampled, snapshot = controller.sample(base)
    assert sampled == base
    assert snapshot.state == "completed"
    assert snapshot.progress == 1.0


def test_wave_requires_stationary_simulation_and_replays_request() -> None:
    clock = _Clock()
    controller = GestureWaveController(clock=clock)
    request_id = str(uuid4())
    with pytest.raises(GestureError, match="stationary"):
        controller.request_wave(
            request_id=request_id,
            gesture="wave",
            duration_s=2.4,
            simulation_only=True,
            ready=False,
        )
    with pytest.raises(GestureError, match="simulation_only"):
        controller.request_wave(
            request_id=str(uuid4()),
            gesture="wave",
            duration_s=2.4,
            simulation_only=False,
            ready=True,
        )
    accepted = controller.request_wave(
        request_id=request_id,
        gesture="wave",
        duration_s=2.4,
        simulation_only=True,
        ready=True,
    )
    replay = controller.request_wave(
        request_id=request_id,
        gesture="wave",
        duration_s=2.4,
        simulation_only=True,
        ready=True,
    )
    assert replay == accepted
    with pytest.raises(GestureError, match="another gesture"):
        controller.request_wave(
            request_id=str(uuid4()),
            gesture="wave",
            duration_s=2.4,
            simulation_only=True,
            ready=True,
        )
