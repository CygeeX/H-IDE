from __future__ import annotations

from dataclasses import replace
import time

import pytest
import requests

import hide_mujoco_sonic.ros_observation_bridge as bridge_module
from hide_mujoco_sonic.ros_observation_bridge import (
    MAX_CAPTURE_AGE_NS,
    CaptureContractError,
    CaptureIdentity,
    MujocoObservationBridge,
    is_new_capture,
)


BOOT_A = "9d8ea4f8-ff59-42df-ac77-70994be3d788"
BOOT_B = "dbce2764-236e-4648-b13e-d6bdd1029366"


def _headers(capture: CaptureIdentity) -> dict[str, str]:
    return {
        "X-Hide-Capture-Boot-Id": capture.producer_boot_id,
        "X-Hide-Capture-Sequence": str(capture.sequence),
        "X-Hide-Captured-At-Unix-Ns": str(capture.captured_at_unix_ns),
    }


class _Response:
    def __init__(
        self,
        *,
        state: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        status_code: int = 200,
    ) -> None:
        self._state = state
        self.headers = headers or {}
        self.status_code = status_code
        self.content = b"must-not-be-decoded"

    def json(self) -> dict[str, object]:
        assert self._state is not None
        return self._state

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger


def _uninitialized_bridge() -> MujocoObservationBridge:
    bridge = object.__new__(MujocoObservationBridge)
    bridge._base_url = "http://127.0.0.1:19080"
    bridge._last_producer_boot_id = None
    bridge._last_sequence = 0
    bridge._last_captured_at_ns = 0
    bridge.node = _Node()
    return bridge


def _state(capture: CaptureIdentity) -> dict[str, object]:
    return {
        "status": "running",
        "producerBootId": capture.producer_boot_id,
        "sequence": capture.sequence,
        "capturedAtUnixNs": capture.captured_at_unix_ns,
    }


def test_capture_identity_builds_exact_boot_and_sequence_urls() -> None:
    capture = CaptureIdentity.from_state(
        {
            "producerBootId": BOOT_A,
            "sequence": 17,
            "capturedAtUnixNs": 1_900_000_000,
        }
    )

    assert capture.exact_resource_url("http://plant", "/rgb.jpg") == (
        "http://plant/rgb.jpg?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788"
        "&sequence=17"
    )
    assert capture.exact_resource_url("http://plant", "/depth.raw") == (
        "http://plant/depth.raw?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788"
        "&sequence=17"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("sequence", True), ("sequence", 1.5), ("capturedAtUnixNs", "1900000000")],
)
def test_state_capture_identity_rejects_coerced_numbers(field: str, value) -> None:
    state = {
        "producerBootId": BOOT_A,
        "sequence": 1,
        "capturedAtUnixNs": 1_900_000_000,
    }
    state[field] = value
    with pytest.raises(CaptureContractError, match="invalid capture"):
        CaptureIdentity.from_state(state)


def test_capture_headers_are_case_insensitive_but_all_three_must_match() -> None:
    capture = CaptureIdentity(BOOT_A, 8, 1_900_000_000)
    lowercase_headers = {
        key.lower(): value for key, value in _headers(capture).items()
    }

    capture.validate_response_headers(lowercase_headers, "/rgb.jpg")

    mismatches = [
        {**_headers(capture), "X-Hide-Capture-Boot-Id": BOOT_B},
        {**_headers(capture), "X-Hide-Capture-Sequence": "9"},
        {**_headers(capture), "X-Hide-Captured-At-Unix-Ns": "1900000001"},
        {
            key: value
            for key, value in _headers(capture).items()
            if key != "X-Hide-Capture-Sequence"
        },
    ]
    for headers in mismatches:
        with pytest.raises(CaptureContractError):
            capture.validate_response_headers(headers, "/rgb.jpg")


def test_sequence_is_strict_within_boot_and_resets_only_on_boot_change() -> None:
    previous_sequence = 42

    assert not is_new_capture(
        CaptureIdentity(BOOT_A, 42, 2_000_000_000), BOOT_A, previous_sequence
    )
    assert not is_new_capture(
        CaptureIdentity(BOOT_A, 41, 2_000_000_000), BOOT_A, previous_sequence
    )
    assert is_new_capture(
        CaptureIdentity(BOOT_A, 43, 2_000_000_000), BOOT_A, previous_sequence
    )
    assert is_new_capture(
        CaptureIdentity(BOOT_B, 1, 2_000_000_000), BOOT_A, previous_sequence
    )


def test_capture_age_gate_includes_exact_500ms_boundary() -> None:
    capture = CaptureIdentity(BOOT_A, 1, 10_000_000_000)

    assert capture.validate_age(capture.captured_at_unix_ns) == 0
    assert (
        capture.validate_age(capture.captured_at_unix_ns + MAX_CAPTURE_AGE_NS)
        == MAX_CAPTURE_AGE_NS
    )
    with pytest.raises(CaptureContractError, match="stale"):
        capture.validate_age(
            capture.captured_at_unix_ns + MAX_CAPTURE_AGE_NS + 1
        )
    with pytest.raises(CaptureContractError, match="future"):
        capture.validate_age(capture.captured_at_unix_ns - 1)


@pytest.mark.parametrize("failure", ["rgb-mismatch", "depth-mismatch", "rgb-409"])
def test_mismatch_or_409_returns_before_decode_or_publish(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_at_ns = time.time_ns()
    capture = CaptureIdentity(BOOT_A, 12, captured_at_ns)
    calls: list[str] = []
    responses = [
        _Response(state=_state(capture), headers=_headers(capture)),
        _Response(
            headers=(
                _headers(replace(capture, sequence=13))
                if failure == "rgb-mismatch"
                else _headers(capture)
            ),
            status_code=409 if failure == "rgb-409" else 200,
        ),
    ]
    if failure == "depth-mismatch":
        responses.append(
            _Response(headers=_headers(replace(capture, producer_boot_id=BOOT_B)))
        )

    def fake_get(url: str, *, timeout: float) -> _Response:
        calls.append(url)
        return responses.pop(0)

    decoded = 0

    def forbidden_decode(*_args, **_kwargs):
        nonlocal decoded
        decoded += 1
        raise AssertionError("a rejected capture must not be decoded")

    import cv2

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(cv2, "imdecode", forbidden_decode)
    bridge = _uninitialized_bridge()

    bridge.poll()

    assert decoded == 0
    assert bridge._last_producer_boot_id is None
    assert bridge._last_sequence == 0
    assert bridge.node.logger.warnings
    assert calls[0].endswith("/state")
    assert calls[1].endswith(
        "/rgb.jpg?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788&sequence=12"
    )
    if failure == "depth-mismatch":
        assert calls[2].endswith(
            "/depth.raw?boot=9d8ea4f8-ff59-42df-ac77-70994be3d788&sequence=12"
        )


def test_capture_age_is_rechecked_after_both_http_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_at_ns = 10_000_000_000
    capture = CaptureIdentity(BOOT_A, 3, captured_at_ns)
    responses = [
        _Response(state=_state(capture), headers=_headers(capture)),
        _Response(headers=_headers(capture)),
        _Response(headers=_headers(capture)),
    ]
    now_values = iter(
        [
            captured_at_ns + 100_000_000,
            captured_at_ns + MAX_CAPTURE_AGE_NS + 1,
        ]
    )

    monkeypatch.setattr(requests, "get", lambda *_args, **_kwargs: responses.pop(0))
    monkeypatch.setattr(bridge_module.time, "time_ns", lambda: next(now_values))
    bridge = _uninitialized_bridge()

    bridge.poll()

    assert not responses
    assert bridge._last_producer_boot_id is None
    assert bridge._last_sequence == 0
    assert "stale" in bridge.node.logger.warnings[-1]
