from __future__ import annotations

from typing import Protocol

import numpy as np


class SegmentationProvider(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def segment(
        self,
        *,
        image_jpeg: bytes,
        box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> np.ndarray: ...
