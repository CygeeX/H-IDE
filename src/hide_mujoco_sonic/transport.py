"""Small localhost JSON-lines transport between ROS bridge and SONIC driver."""

from __future__ import annotations

import json
import socket
from typing import Any
from uuid import uuid4
from urllib import request


def send_json(host: str, port: int, payload: dict[str, Any], *, timeout: float = 0.2) -> None:
    data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(data)


def request_json(
    host: str,
    port: int,
    payload: dict[str, Any],
    *,
    timeout: float = 0.5,
    require_ok: bool = True,
) -> dict[str, Any]:
    """Send one bounded loopback command and require its matching ACK."""

    if not isinstance(require_ok, bool):
        raise ValueError("require_ok must be a boolean")

    request_id = str(payload.get("request_id") or uuid4())
    message = {**payload, "request_id": request_id}
    data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(data)
        sock.shutdown(socket.SHUT_WR)
        response = b""
        while b"\n" not in response:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if len(response) > 4096:
                raise RuntimeError("SONIC command response exceeds the bounded limit")
    try:
        value = json.loads(response.split(b"\n", 1)[0])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("SONIC command endpoint returned an invalid response") from exc
    if not isinstance(value, dict) or value.get("request_id") != request_id:
        raise RuntimeError("SONIC command response identity mismatch")
    ok = value.get("ok")
    if not isinstance(ok, bool):
        raise RuntimeError("SONIC command response truth flag is invalid")
    if require_ok and not ok:
        raise RuntimeError(str(value.get("error") or "SONIC command was rejected"))
    return value


def set_plant_support(
    host: str,
    port: int,
    mode: str,
    *,
    timeout: float = 0.3,
    urlopen=request.urlopen,
) -> dict[str, Any]:
    if mode not in {"startup", "idle", "locomotion", "safety_stop"}:
        raise ValueError("unsupported plant support mode")
    body = json.dumps({"mode": mode}, separators=(",", ":")).encode("utf-8")
    http_request = request.Request(
        f"http://{host}:{int(port)}/support",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(http_request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"plant support endpoint returned {response.status}")
        payload = json.loads(response.read(1024))
    if (
        not isinstance(payload, dict)
        or payload.get("mode") != mode
        or payload.get("fault") is not None
    ):
        raise RuntimeError("plant support endpoint did not confirm requested mode")
    return payload


def latch_plant_physics_fault(
    host: str,
    port: int,
    reason: str,
    *,
    timeout: float = 0.3,
    urlopen=request.urlopen,
) -> dict[str, Any]:
    """Latch a terminal Plant fault before another MuJoCo integration step.

    This endpoint is reserved for an already-latched motion safety event.  It
    never resets state or asks the support tether to stabilize a dynamic pose;
    the authoritative Plant loop observes the fault and exits fail-closed.
    """

    detail = str(reason).strip()
    if not detail or len(detail) > 512:
        raise ValueError("plant physics fault reason must contain 1..512 characters")
    body = json.dumps({"reason": detail}, separators=(",", ":")).encode("utf-8")
    http_request = request.Request(
        f"http://{host}:{int(port)}/support/fault",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(http_request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(
                f"plant physics fault endpoint returned {response.status}"
            )
        payload = json.loads(response.read(1024))
    if (
        not isinstance(payload, dict)
        or payload.get("mode") != "safety_stop"
        or payload.get("safetyStopLatched") is not True
        or payload.get("fault") != detail
    ):
        raise RuntimeError("plant did not confirm the terminal physics fault")
    return payload


def rearm_plant_support(
    host: str,
    port: int,
    generation: int,
    *,
    timeout: float = 0.3,
    urlopen=request.urlopen,
) -> dict[str, Any]:
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation <= 0
    ):
        raise ValueError("plant rearm generation must be a positive integer")
    body = json.dumps(
        {"generation": generation}, separators=(",", ":")
    ).encode("utf-8")
    http_request = request.Request(
        f"http://{host}:{int(port)}/support/rearm",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(http_request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(
                f"plant support rearm endpoint returned {response.status}"
            )
        payload = json.loads(response.read(1024))
    if (
        not isinstance(payload, dict)
        or payload.get("mode") != "idle"
        or payload.get("lastRearmGeneration") != generation
        or payload.get("safetyStopLatched") is not False
    ):
        raise RuntimeError("plant support endpoint did not confirm explicit rearm")
    return payload


def parse_lines(buffer: bytes) -> tuple[list[dict[str, Any]], bytes]:
    messages: list[dict[str, Any]] = []
    for raw in buffer.split(b"\n")[:-1]:
        if not raw:
            continue
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("transport message must be an object")
        messages.append(value)
    return messages, buffer.rsplit(b"\n", 1)[-1]
