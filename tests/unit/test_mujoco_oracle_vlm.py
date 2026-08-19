from __future__ import annotations

import hashlib
from io import BytesIO

import httpx
import pytest
from PIL import Image

from h_ide_vision.providers.vlm import MujocoOracleVlmProvider


def _jpeg() -> bytes:
    body = BytesIO()
    Image.new("RGB", (64, 48), color=(20, 30, 40)).save(body, format="JPEG")
    return body.getvalue()


def test_oracle_returns_only_matching_pixel_hypotheses_for_exact_frame() -> None:
    image = _jpeg()
    digest = hashlib.sha256(image).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["rgbSha256"] == digest
        return httpx.Response(
            200,
            json={
                "schemaVersion": "h-ide.mujoco-oracle/1",
                "sourceMode": "synthetic",
                "rgbSha256": digest,
                "width": 64,
                "height": 48,
                "annotations": [
                    {
                        "annotationId": "geom-1",
                        "label": "显示器",
                        "aliases": ["显示器", "屏幕", "monitor"],
                        "pixelBox": [10, 8, 30, 24],
                    },
                    {
                        "annotationId": "geom-2",
                        "label": "杯子",
                        "aliases": ["杯子", "cup"],
                        "pixelBox": [40, 20, 50, 35],
                    },
                ],
            },
        )

    provider = MujocoOracleVlmProvider(
        base_url="http://mujoco.invalid",
        transport=httpx.MockTransport(handler),
    )

    candidates = provider.ground(image_jpeg=image, target="只识别当前画面的显示器")

    assert len(candidates) == 1
    assert candidates[0].label == "显示器"
    assert candidates[0].box == (10, 8, 30, 24)
    assert candidates[0].source == "mujoco_oracle"
    assert provider.ground(image_jpeg=image, target="寻找沙发") == ()

    visible = provider.observe(image_jpeg=image)
    assert [candidate.label for candidate in visible] == ["显示器", "杯子"]


def test_oracle_rejects_a_response_for_a_different_frame() -> None:
    image = _jpeg()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "schemaVersion": "h-ide.mujoco-oracle/1",
                "sourceMode": "synthetic",
                "rgbSha256": "0" * 64,
                "width": 64,
                "height": 48,
                "annotations": [],
            },
        )

    provider = MujocoOracleVlmProvider(
        base_url="http://mujoco.invalid",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(RuntimeError, match="does not match"):
        provider.ground(image_jpeg=image, target="显示器")
