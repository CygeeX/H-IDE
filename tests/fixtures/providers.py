from __future__ import annotations

import numpy as np

from h_ide_vision.providers.vlm.base import GroundingCandidate


class StubQwenProvider:
    """Deterministic Qwen protocol double that exists only in tests."""

    provider_name = "qwen"

    def __init__(self, candidates: tuple[GroundingCandidate, ...] | None = None) -> None:
        self._candidates = candidates or (
            GroundingCandidate(
                candidate_id="qwen-test-target-1",
                label="fixture target",
                box=(24, 16, 40, 32),
                score=1.0,
            ),
        )

    @property
    def available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "qwen-test-double"

    def ensure_ready(self) -> None:
        return None

    def ground(
        self, *, image_jpeg: bytes, target: str
    ) -> tuple[GroundingCandidate, ...]:
        return self._candidates if image_jpeg and target.strip() else ()

    def observe(self, *, image_jpeg: bytes) -> tuple[GroundingCandidate, ...]:
        return self._candidates if image_jpeg else ()


class StubSam2Provider:
    """Box-shaped SAM 2 protocol double; never part of the installed package."""

    name = "sam2"
    available = True

    @property
    def model_name(self) -> str:
        return "sam2-test-double"

    def ensure_ready(self) -> None:
        return None

    def segment(
        self,
        *,
        image_jpeg: bytes,
        box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> np.ndarray:
        del image_jpeg
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("test segmentation box is outside image")
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask
