from __future__ import annotations

import json

import httpx
import pytest

from h_ide_vision.fixtures import fixture_rgb
from h_ide_vision.providers.vlm.qwen import QwenVlmProvider


class _Response:
    def __init__(self, content: dict[str, object]):
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._content


def test_qwen_normalized_box_is_converted_to_image_pixels(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "payload": json, "timeout": timeout})
        content = {
            "candidates": [
                {
                    "label": "黄色方框",
                    "box": [375, 300, 625, 700],
                    "score": 0.98,
                }
            ]
        }
        return _Response(
            {"choices": [{"message": {"content": json_module.dumps(content)}}]}
        )

    json_module = json
    monkeypatch.setattr(httpx, "post", post)
    provider = QwenVlmProvider(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="qwen3-vl-plus",
    )
    candidates = provider.ground(image_jpeg=fixture_rgb(), target="黄色方框")
    assert candidates[0].box == (24, 14, 40, 34)
    assert candidates[0].candidate_id.startswith("candidate-")
    schema = captured["payload"]["response_format"]["json_schema"]["schema"]
    box_items = schema["properties"]["candidates"]["items"]["properties"]["box"]["items"]
    assert box_items == {"type": "integer", "minimum": 0, "maximum": 1000}


@pytest.mark.parametrize(
    "box",
    ([500, 500, 500, 700], [-1, 0, 10, 10], [0, 0, 1001, 10]),
)
def test_qwen_rejects_invalid_normalized_boxes(monkeypatch, box) -> None:
    def post(*args, **kwargs):
        del args, kwargs
        content = {
            "candidates": [
                {"label": "bad", "box": box, "score": 0.1}
            ]
        }
        return _Response(
            {"choices": [{"message": {"content": json.dumps(content)}}]}
        )

    monkeypatch.setattr(httpx, "post", post)
    provider = QwenVlmProvider(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="qwen3-vl-plus",
    )
    with pytest.raises(RuntimeError, match="normalized box"):
        provider.ground(image_jpeg=fixture_rgb(), target="bad")


def test_qwen_candidate_ids_are_deterministic_and_duplicate_boxes_are_collapsed(
    monkeypatch,
) -> None:
    content = {
        "candidates": [
            {"label": "桌子", "box": [100, 200, 300, 400], "score": 0.9},
            {"label": "桌子", "box": [100, 200, 300, 400], "score": 0.8},
            {"label": "桌子", "box": [500, 200, 700, 400], "score": 0.7},
        ]
    }

    def post(*args, **kwargs):
        del args, kwargs
        return _Response(
            {"choices": [{"message": {"content": json.dumps(content)}}]}
        )

    monkeypatch.setattr(httpx, "post", post)
    provider = QwenVlmProvider(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="qwen3-vl-plus",
    )

    first = provider.ground(image_jpeg=fixture_rgb(), target="桌子")
    second = provider.ground(image_jpeg=fixture_rgb(), target="桌子")

    assert len(first) == 2
    assert [item.candidate_id for item in first] == [
        item.candidate_id for item in second
    ]
    assert first[0].candidate_id != first[1].candidate_id


def test_qwen_retries_transient_transport_failure(monkeypatch) -> None:
    calls = 0

    def post(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("temporary TLS connect failure")
        content = {"candidates": [{"label": "机器人", "box": [100, 100, 500, 900]}]}
        return _Response(
            {"choices": [{"message": {"content": json.dumps(content)}}]}
        )

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("h_ide_vision.providers.vlm.qwen.time.sleep", lambda _: None)
    provider = QwenVlmProvider(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="qwen3-vl-plus",
    )

    assert len(provider.ground(image_jpeg=fixture_rgb(), target="机器人")) == 1
    assert calls == 2


def test_qwen_does_not_retry_non_retryable_http_status(monkeypatch) -> None:
    calls = 0
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")

    def post(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return httpx.Response(422, request=request)

    monkeypatch.setattr(httpx, "post", post)
    provider = QwenVlmProvider(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="qwen3-vl-plus",
    )

    with pytest.raises(httpx.HTTPStatusError):
        provider.ground(image_jpeg=fixture_rgb(), target="机器人")
    assert calls == 1
