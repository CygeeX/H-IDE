"""Small localhost JSON-lines transport between ROS bridge and SONIC driver."""

from __future__ import annotations

import json
import socket
from typing import Any
from urllib import request


def send_json(host: str, port: int, payload: dict[str, Any], *, timeout: float = 0.2) -> None:
    data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.sendall(data)


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
    if not isinstance(payload, dict) or payload.get("mode") != mode:
        raise RuntimeError("plant support endpoint did not confirm requested mode")
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
