"""Small localhost JSON-lines transport between ROS bridge and SONIC driver."""

from __future__ import annotations

import json
import math
import socket
import time
from typing import Any
from uuid import UUID, uuid4
from urllib import request


class JsonResponseUnavailableError(ConnectionError):
    """The peer closed before returning one complete JSON-lines response."""


class JsonResponseContractError(RuntimeError):
    """The peer returned bytes that violate the correlated response contract."""


class JsonResponseTimeoutError(TimeoutError):
    """A socket operation exceeded the caller's bounded response deadline."""


def read_plant_identity(
    host: str,
    port: int,
    *,
    timeout: float = 0.3,
    urlopen=request.urlopen,
) -> dict[str, str]:
    """Read the live Plant process/capture identities without mutating it."""

    http_request = request.Request(
        f"http://{host}:{int(port)}/state",
        method="GET",
    )
    with urlopen(http_request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(
                f"plant state endpoint returned {response.status}"
            )
        payload = json.loads(response.read(2 * 1024 * 1024))
    if not isinstance(payload, dict):
        raise RuntimeError("plant state endpoint did not return an object")
    values: dict[str, str] = {}
    for key, label in (
        ("producerBootId", "Plant producer boot ID"),
        ("sessionId", "Plant session ID"),
    ):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{label} is missing")
        try:
            parsed = UUID(value)
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"{label} is not a UUID") from exc
        if str(parsed) != value:
            raise RuntimeError(f"{label} is not canonical")
        values[key] = value
    return values


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
    max_response_bytes: int = 4096,
) -> dict[str, Any]:
    """Send one bounded loopback command and require its matching ACK."""

    if not isinstance(require_ok, bool):
        raise ValueError("require_ok must be a boolean")
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or not 1024 <= max_response_bytes <= 32 * 1024 * 1024
    ):
        raise ValueError(
            "max_response_bytes must be an integer between 1024 and 33554432"
        )

    request_id = str(payload.get("request_id") or uuid4())
    message = {**payload, "request_id": request_id}
    data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
    try:
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
                if len(response) > max_response_bytes:
                    raise JsonResponseContractError(
                        "SONIC command response exceeds the bounded limit"
                    )
    except TimeoutError as exc:
        raise JsonResponseTimeoutError(
            "SONIC command response timed out"
        ) from exc
    if b"\n" not in response:
        raise JsonResponseUnavailableError(
            "SONIC command endpoint closed before a complete response"
        )
    try:
        value = json.loads(response.split(b"\n", 1)[0])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise JsonResponseContractError(
            "SONIC command endpoint returned an invalid response"
        ) from exc
    if not isinstance(value, dict) or value.get("request_id") != request_id:
        raise JsonResponseContractError(
            "SONIC command response identity mismatch"
        )
    ok = value.get("ok")
    if not isinstance(ok, bool):
        raise JsonResponseContractError(
            "SONIC command response truth flag is invalid"
        )
    if require_ok and not ok:
        raise JsonResponseContractError(
            str(value.get("error") or "SONIC command was rejected")
        )
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


def _plant_motion_fault_request(
    host: str,
    port: int,
    reason: str,
    *,
    reset: bool,
    timeout: float,
    urlopen,
) -> dict[str, Any]:
    detail = str(reason).strip()
    if not detail or len(detail) > 512:
        raise ValueError("plant motion fault reason must contain 1..512 characters")
    suffix = "/reset" if reset else ""
    body = json.dumps({"reason": detail}, separators=(",", ":")).encode("utf-8")
    http_request = request.Request(
        f"http://{host}:{int(port)}/support/motion-fault{suffix}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(http_request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(
                f"plant motion fault endpoint returned {response.status}"
            )
        payload = json.loads(response.read(1024))
    expected_fault = None if reset else detail
    if (
        not isinstance(payload, dict)
        or payload.get("mode") != "safety_stop"
        or payload.get("safetyStopLatched") is not True
        or payload.get("fault") is not None
        or payload.get("motionFault") != expected_fault
        or not isinstance(payload.get("fallArrestAvailable"), bool)
        or (
            payload.get("fallArrestAvailable") is True
            and (
                payload.get("fallArrestEnabled") is not True
                or isinstance(payload.get("safetyStopEpoch"), bool)
                or not isinstance(payload.get("safetyStopEpoch"), int)
                or payload.get("safetyStopEpoch") <= 0
                or payload.get("fallArrestReengagedSafetyStopEpoch")
                != payload.get("safetyStopEpoch")
            )
        )
    ):
        raise RuntimeError("plant did not confirm the recoverable motion fault state")
    return payload


def latch_plant_motion_fault(
    host: str,
    port: int,
    reason: str,
    *,
    timeout: float = 0.3,
    urlopen=request.urlopen,
) -> dict[str, Any]:
    """Latch safety stop while keeping valid MuJoCo physics alive."""

    return _plant_motion_fault_request(
        host,
        port,
        reason,
        reset=False,
        timeout=timeout,
        urlopen=urlopen,
    )


def clear_plant_motion_fault(
    host: str,
    port: int,
    reason: str,
    *,
    timeout: float = 0.3,
    urlopen=request.urlopen,
) -> dict[str, Any]:
    """Clear the named motion fault without rearming support or motion."""

    return _plant_motion_fault_request(
        host,
        port,
        reason,
        reset=True,
        timeout=timeout,
        urlopen=urlopen,
    )


def release_plant_fall_arrest(
    host: str,
    port: int,
    activation_epoch: int,
    *,
    timeout: float = 0.75,
    poll_interval: float = 0.01,
    urlopen=request.urlopen,
    clock=time.monotonic,
    sleeper=time.sleep,
) -> dict[str, Any]:
    """Request and prove one authority-bound simulation fall-arrest release."""

    if (
        isinstance(activation_epoch, bool)
        or not isinstance(activation_epoch, int)
        or activation_epoch <= 0
    ):
        raise ValueError("fall-arrest activation epoch must be a positive integer")
    timeout_s = float(timeout)
    interval_s = float(poll_interval)
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("fall-arrest release timeout must be finite and positive")
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("fall-arrest poll interval must be finite and positive")
    body = json.dumps(
        {"activationEpoch": activation_epoch},
        separators=(",", ":"),
    ).encode("utf-8")
    release_request = request.Request(
        f"http://{host}:{int(port)}/support/fall-arrest/release",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = float(clock()) + timeout_s
    with urlopen(release_request, timeout=timeout_s) as response:
        if response.status != 200:
            raise RuntimeError(
                f"plant fall-arrest release endpoint returned {response.status}"
            )
        first = json.loads(response.read(2048))

    def confirmed(payload: object) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        support = payload.get("support", payload)
        if not isinstance(support, dict):
            return None
        rejected_epoch = support.get("fallArrestReleaseRejectedEpoch")
        if rejected_epoch == activation_epoch:
            raise RuntimeError(
                "Plant rejected fall-arrest release at the physics boundary"
            )
        confirmed_epoch = support.get("fallArrestReleaseConfirmedEpoch")
        if (
            support.get("fallArrestEnabled") is False
            and confirmed_epoch == activation_epoch
            and support.get("fault") is None
            and support.get("motionFault") is None
        ):
            return dict(support)
        return None

    result = confirmed(first)
    while result is None:
        remaining = deadline - float(clock())
        if remaining <= 0.0:
            raise RuntimeError(
                "Plant did not confirm fall-arrest release before the deadline"
            )
        sleeper(min(interval_s, remaining))
        state_request = request.Request(
            f"http://{host}:{int(port)}/state",
            method="GET",
        )
        with urlopen(state_request, timeout=max(0.01, remaining)) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"plant state endpoint returned {response.status}"
                )
            result = confirmed(json.loads(response.read(2 * 1024 * 1024)))
    return result


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
