from __future__ import annotations

import pytest

from services.control_api import leader_demo


def test_trigger_wave_uses_the_allow_listed_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"status": "accepted", "requestId": "wave-1"}

    def post(url, *, json, headers, timeout, trust_env):
        calls.append((url, json, headers))
        assert timeout == pytest.approx(0.5)
        assert trust_env is False
        return Response()

    monkeypatch.setenv("HIDE_LEADER_DEMO_GESTURES_ENABLED", "true")
    monkeypatch.setenv("HIDE_LEADER_DEMO_GESTURE_URL", "http://host.docker.internal:8798")
    monkeypatch.setenv("HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN", "secret")
    monkeypatch.setattr(leader_demo.httpx, "post", post)

    result = leader_demo.trigger_wave(robot_id="g1-sim", expected_robot_id="g1-sim")

    assert result["status"] == "accepted"
    assert calls == [
        (
            "http://host.docker.internal:8798/v1/gesture/wave",
            {"robotId": "g1-sim"},
            {"Authorization": "Bearer secret"},
        )
    ]


def test_trigger_wave_fails_closed_for_non_simulation_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIDE_LEADER_DEMO_GESTURES_ENABLED", "true")
    with pytest.raises(leader_demo.LeaderDemoGestureUnavailable):
        leader_demo.trigger_wave(robot_id="real-g1", expected_robot_id="g1-sim")
