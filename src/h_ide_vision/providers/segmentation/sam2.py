from __future__ import annotations

import threading
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class Sam2SegmentationProvider:
    """Lazy SAM 2 adapter. Missing optional dependencies remain an explicit capability failure."""

    name = "sam2"

    def __init__(
        self,
        predictor: Any | None = None,
        *,
        checkpoint: Path | None = None,
        model_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        device: str = "cuda",
    ) -> None:
        self._predictor = predictor
        self._checkpoint = checkpoint
        self._model_config = model_config
        self._device = device
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        if self._predictor is not None:
            return True
        if not (
            self._checkpoint is not None
            and self._checkpoint.is_file()
            and find_spec("sam2") is not None
            and find_spec("torch") is not None
        ):
            return False
        if self._device != "cuda":
            return True
        import torch

        return bool(torch.cuda.is_available())

    def segment(
        self,
        *,
        image_jpeg: bytes,
        box: tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> np.ndarray:
        with self._lock:
            predictor = self._predictor or self._load_predictor()
            image_rgb = np.array(Image.open(BytesIO(image_jpeg)).convert("RGB"), copy=True)
            if image_rgb.shape[:2] != (height, width):
                raise RuntimeError("SAM 2 keyframe dimensions do not match observation metadata")
            import torch

            with torch.inference_mode():
                predictor.set_image(image_rgb)
                masks, scores, _ = predictor.predict(
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=True,
                )
            if len(masks) == 0:
                raise RuntimeError("SAM 2 returned no masks")
            mask = np.asarray(masks[int(np.argmax(scores))], dtype=bool)
            if mask.shape != (height, width) or not mask.any():
                raise RuntimeError("SAM 2 returned an invalid or empty mask")
            return mask

    def _load_predictor(self) -> Any:
        if not self.available or self._checkpoint is None:
            raise RuntimeError("SAM 2 is unavailable: GPU environment/model is not configured")
        try:
            import torch
            from sam2.build_sam import build_sam2  # type: ignore[import-untyped]
            from sam2.sam2_image_predictor import (  # type: ignore[import-untyped]
                SAM2ImagePredictor,
            )
        except ImportError as exc:
            raise RuntimeError("SAM 2 optional GPU dependencies are unavailable") from exc
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM 2 CUDA device is unavailable")
        self._predictor = SAM2ImagePredictor(
            build_sam2(self._model_config, str(self._checkpoint), device=self._device)
        )
        return self._predictor
