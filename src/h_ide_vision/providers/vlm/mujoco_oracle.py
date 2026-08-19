"""Simulation-only semantic grounding backed by MuJoCo segmentation.

The oracle returns pixel-space hypotheses for the exact JPEG submitted by the
Robot Agent observation stream.  It intentionally exposes no world pose: depth,
camera calibration, and TF remain the only geometry authority.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from .base import GroundingCandidate


class MujocoOracleVlmProvider:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float = 2.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    @property
    def available(self) -> bool:
        return bool(self._base_url)

    @property
    def model_name(self) -> str:
        return "mujoco-segmentation-oracle-v1"

    def ground(
        self, *, image_jpeg: bytes, target: str
    ) -> tuple[GroundingCandidate, ...]:
        if not image_jpeg or not target.strip():
            return ()
        rgb_sha256, annotations = self._annotations_for_image(image_jpeg)
        normalized_target = self._normalize(target)
        return self._candidates(
            annotations,
            rgb_sha256=rgb_sha256,
            predicate=lambda item: any(
                self._term_matches(normalized_target, term)
                for term in (item["label"], *item["aliases"])
            ),
        )

    def observe(self, *, image_jpeg: bytes) -> tuple[GroundingCandidate, ...]:
        if not image_jpeg:
            return ()
        rgb_sha256, annotations = self._annotations_for_image(image_jpeg)
        return self._candidates(
            annotations,
            rgb_sha256=rgb_sha256,
            predicate=lambda _item: True,
        )

    def _annotations_for_image(
        self, image_jpeg: bytes
    ) -> tuple[str, tuple[dict[str, Any], ...]]:
        try:
            with Image.open(BytesIO(image_jpeg)) as image:
                width, height = image.size
        except Exception as exc:
            raise RuntimeError("MuJoCo Oracle input is not a decodable image") from exc
        rgb_sha256 = hashlib.sha256(image_jpeg).hexdigest()
        try:
            with httpx.Client(
                transport=self._transport,
                timeout=self._timeout_s,
            ) as client:
                response = client.get(
                    f"{self._base_url}/oracle.json",
                    params={"rgbSha256": rgb_sha256},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"MuJoCo Oracle request failed: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("MuJoCo Oracle returned invalid JSON") from exc
        annotations = self._validated_annotations(
            payload,
            rgb_sha256=rgb_sha256,
            width=width,
            height=height,
        )
        return rgb_sha256, annotations

    @staticmethod
    def _candidates(
        annotations: tuple[dict[str, Any], ...],
        *,
        rgb_sha256: str,
        predicate: Any,
    ) -> tuple[GroundingCandidate, ...]:
        candidates: list[GroundingCandidate] = []
        for item in annotations:
            if not predicate(item):
                continue
            candidates.append(
                GroundingCandidate(
                    candidate_id=(
                        f"oracle-{item['annotationId']}-{rgb_sha256[:12]}"
                    ),
                    label=item["label"],
                    box=item["pixelBox"],
                    score=1.0,
                    source="mujoco_oracle",
                )
            )
        return tuple(candidates)

    @classmethod
    def _term_matches(cls, normalized_target: str, term: str) -> bool:
        normalized_term = cls._normalize(term)
        return bool(normalized_term) and normalized_term in normalized_target

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(character for character in value.casefold() if character.isalnum())

    @staticmethod
    def _validated_annotations(
        payload: Any,
        *,
        rgb_sha256: str,
        width: int,
        height: int,
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(payload, dict):
            raise RuntimeError("MuJoCo Oracle response must be an object")
        if payload.get("schemaVersion") != "h-ide.mujoco-oracle/1":
            raise RuntimeError("MuJoCo Oracle schema version mismatch")
        if payload.get("sourceMode") != "synthetic":
            raise RuntimeError("MuJoCo Oracle refused a non-synthetic source")
        if payload.get("rgbSha256") != rgb_sha256:
            raise RuntimeError("MuJoCo Oracle response does not match the input frame")
        if payload.get("width") != width or payload.get("height") != height:
            raise RuntimeError("MuJoCo Oracle image dimensions do not match")
        raw_annotations = payload.get("annotations")
        if not isinstance(raw_annotations, list) or len(raw_annotations) > 128:
            raise RuntimeError("MuJoCo Oracle annotations are invalid")
        validated: list[dict[str, Any]] = []
        for raw in raw_annotations:
            if not isinstance(raw, dict):
                raise RuntimeError("MuJoCo Oracle annotation must be an object")
            annotation_id = raw.get("annotationId")
            label = raw.get("label")
            aliases = raw.get("aliases")
            box = raw.get("pixelBox")
            if not isinstance(annotation_id, str) or not annotation_id.strip():
                raise RuntimeError("MuJoCo Oracle annotation ID is invalid")
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError("MuJoCo Oracle annotation label is invalid")
            if (
                not isinstance(aliases, list)
                or len(aliases) > 32
                or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
            ):
                raise RuntimeError("MuJoCo Oracle annotation aliases are invalid")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
            ):
                raise RuntimeError("MuJoCo Oracle annotation box is invalid")
            x_min, y_min, x_max, y_max = box
            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                raise RuntimeError("MuJoCo Oracle annotation box is outside the image")
            validated.append(
                {
                    "annotationId": annotation_id.strip(),
                    "label": label.strip(),
                    "aliases": tuple(alias.strip() for alias in aliases),
                    "pixelBox": (x_min, y_min, x_max, y_max),
                }
            )
        return tuple(validated)
