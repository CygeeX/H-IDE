from __future__ import annotations

from typing import Any

import numpy as np


class FoundationPoseAdapter:
    """Lazy adapter around the separately licensed FoundationPose inference subset."""

    name = "foundationpose"

    def __init__(self, estimator: Any | None = None) -> None:
        self._estimator = estimator

    @property
    def available(self) -> bool:
        return self._estimator is not None

    def estimate(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("FoundationPose is unavailable: CUDA extensions/model are not configured")
        pose = self._estimator.register(K=intrinsics, rgb=rgb, depth=depth_m, ob_mask=mask, iteration=5)
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise RuntimeError("FoundationPose returned an invalid transform")
        return value
