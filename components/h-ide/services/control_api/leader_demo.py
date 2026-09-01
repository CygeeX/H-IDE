"""Scoped client for the simulation-only leader-demo gesture proxy.

Control API is containerized and must not reach SONIC's raw command socket.
It calls a host-side allow-listed HTTP proxy instead; SONIC remains the single
actuator owner and arbitrates the bounded arm overlay with its locomotion
policy.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx


class LeaderDemoGestureUnavailable(RuntimeError):
    """The local simulation gesture endpoint is not configured or reachable."""


def _enabled() -> bool:
    value = os.getenv("HIDE_LEADER_DEMO_GESTURES_ENABLED", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def trigger_wave(
    *,
    robot_id: str,
    expected_robot_id: str,
    timeout_s: float = 0.5,
) -> dict[str, Any]:
    """Request one bounded simulation wave and return SONIC's ACK.

    ``expected_robot_id`` is server configuration, not browser input.  This
    prevents a public HTTP caller from turning the endpoint into an arbitrary
    network socket or a real-robot action bridge.
    """

    if not _enabled():
        raise LeaderDemoGestureUnavailable("leader-demo gestures are disabled")
    if not isinstance(robot_id, str) or not robot_id.strip():
        raise ValueError("robot_id is required")
    if robot_id != expected_robot_id:
        raise LeaderDemoGestureUnavailable("leader-demo gesture is simulation-only")
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise LeaderDemoGestureUnavailable("SONIC gesture transport configuration is invalid") from exc
    endpoint = os.getenv(
        "HIDE_LEADER_DEMO_GESTURE_URL",
        "http://127.0.0.1:8798",
    ).strip().rstrip("/") + "/v1/gesture/wave"
    token = os.getenv("HIDE_LEADER_DEMO_GESTURE_PROXY_TOKEN", "").strip()
    if not endpoint or not token or not 0.05 <= timeout <= 2.0:
        raise LeaderDemoGestureUnavailable("SONIC gesture transport configuration is invalid")
    try:
        response = httpx.post(
            endpoint,
            json={"robotId": robot_id},
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            trust_env=False,
        )
        response.raise_for_status()
        result = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise LeaderDemoGestureUnavailable(
            f"SONIC gesture proxy unavailable: {type(exc).__name__}"
        ) from exc
    if not isinstance(result, Mapping) or result.get("status") != "accepted":
        raise LeaderDemoGestureUnavailable("SONIC gesture proxy returned an invalid ACK")
    return dict(result)


__all__ = ["LeaderDemoGestureUnavailable", "trigger_wave"]
