from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from .base import GroundingCandidate


class QwenVlmProvider:
    def __init__(self, *, api_key: str | None, base_url: str, model: str, timeout_s: float = 20.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def model_name(self) -> str:
        return self._model

    def ground(self, *, image_jpeg: bytes, target: str) -> tuple[GroundingCandidate, ...]:
        if not self._api_key:
            raise RuntimeError("Qwen VLM is unavailable: DASHSCOPE_API_KEY is missing")
        try:
            with Image.open(BytesIO(image_jpeg)) as image:
                width, height = image.size
        except Exception as exc:
            raise RuntimeError("Qwen VLM input is not a decodable image") from exc
        if width < 2 or height < 2:
            raise RuntimeError("Qwen VLM input dimensions are too small")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["candidates"],
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "box"],
                        "properties": {
                            "label": {"type": "string"},
                            "box": {
                                "type": "array",
                                "minItems": 4,
                                "maxItems": 4,
                                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                            },
                            "score": {"type": "number"},
                        },
                    },
                }
            },
        }
        data_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return only semantic image candidates. Never return coordinates "
                        "outside pixel boxes, paths, robot commands, ROS names, velocities, "
                        "or safety decisions. Every box must be [x_min, y_min, x_max, y_max] "
                        "in a normalized 0-1000 image coordinate system."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {
                            "type": "text",
                            "text": (
                                f"Locate visible candidates matching: {target}. Return boxes "
                                "only in the normalized 0-1000 coordinate system."
                            ),
                        },
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "visual_candidates",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 0,
        }
        response = self._post_with_retry(payload)
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        candidates = []
        seen_candidates: set[tuple[str, tuple[int, int, int, int]]] = set()
        for item in parsed["candidates"]:
            raw_box = item["box"]
            if (
                not isinstance(raw_box, list)
                or len(raw_box) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_box)
                or any(value < 0 or value > 1000 for value in raw_box)
            ):
                raise RuntimeError("Qwen VLM returned an invalid normalized box")
            x_min, y_min, x_max, y_max = raw_box
            if x_min >= x_max or y_min >= y_max:
                raise RuntimeError("Qwen VLM returned an empty normalized box")
            pixel_box = (
                max(0, min(width - 1, math.floor(x_min * width / 1000))),
                max(0, min(height - 1, math.floor(y_min * height / 1000))),
                max(1, min(width - 1, math.ceil(x_max * width / 1000))),
                max(1, min(height - 1, math.ceil(y_max * height / 1000))),
            )
            if pixel_box[0] >= pixel_box[2] or pixel_box[1] >= pixel_box[3]:
                raise RuntimeError("Qwen VLM box collapses at the input resolution")
            label = str(item["label"]).strip()
            if not label:
                raise RuntimeError("Qwen VLM returned an empty candidate label")
            identity = (label, pixel_box)
            if identity in seen_candidates:
                continue
            seen_candidates.add(identity)
            digest_source = json.dumps(
                {"box": pixel_box, "label": label},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            candidates.append(
                GroundingCandidate(
                    candidate_id=f"candidate-{hashlib.sha256(digest_source).hexdigest()[:16]}",
                    label=label,
                    box=pixel_box,
                    score=None if item.get("score") is None else float(item["score"]),
                    source="qwen",
                )
            )
        return tuple(candidates)

    def observe(self, *, image_jpeg: bytes) -> tuple[GroundingCandidate, ...]:
        """Return salient visible objects without inventing map geometry.

        Scene prose is assembled locally from these bounded pixel hypotheses;
        Qwen remains a semantic grounder and never becomes a motion authority.
        """

        return self.ground(
            image_jpeg=image_jpeg,
            target=(
                "all clearly visible salient objects, room fixtures, landmarks, "
                "doorways and obstacles in the current robot camera view"
            ),
        )

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        delays = (0.25, 0.75)
        for attempt in range(3):
            try:
                response = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                    timeout=self._timeout_s,
                )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if attempt == 2 or (status != 429 and not 500 <= status <= 599):
                    raise
            except httpx.TransportError:
                if attempt == 2:
                    raise
            time.sleep(delays[attempt])
        raise RuntimeError("unreachable Qwen retry state")
