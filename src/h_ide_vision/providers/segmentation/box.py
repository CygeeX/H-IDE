from __future__ import annotations

import numpy as np


class BoxSegmentationProvider:
    """Bounding-box mask for fixtures and trusted synthetic oracles only."""

    name = "box-mask-v1"
    available = True

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
            raise ValueError("segmentation box is outside image")
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask
