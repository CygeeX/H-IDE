"""Retained offline FoundationPose spike; not installed or product-registered."""

from __future__ import annotations

from typing import Any

import numpy as np


class FoundationPoseExperiment:
    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    def estimate(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        pose = self._estimator.register(
            K=intrinsics,
            rgb=rgb,
            depth=depth_m,
            ob_mask=mask,
            iteration=5,
        )
        value = np.asarray(pose, dtype=np.float64)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise RuntimeError("FoundationPose experiment returned an invalid transform")
        return value
