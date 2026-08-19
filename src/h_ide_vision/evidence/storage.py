from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from h_ide_vision.contracts import EvidenceRef


class EvidenceStorage:
    def __init__(self, root: Path, retention_hours: int = 24) -> None:
        self.root = root
        self.retention = timedelta(hours=retention_hours)

    def put(
        self,
        *,
        kind: str,
        media_type: str,
        data: bytes,
        producer: Literal[
            "observation_gateway",
            "qwen",
            "sam2",
            "deterministic_rgbd",
            "legacy_unverified",
        ],
        model: str | None = None,
    ) -> EvidenceRef:
        digest = hashlib.sha256(data).hexdigest()
        path = self.root / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return EvidenceRef(
            kind=kind,
            sha256=digest,
            mediaType=media_type,
            producer=producer,
            model=model,
            expiresAt=datetime.now(UTC) + self.retention,
        )

    def put_rgb_input(self, image_jpeg: bytes) -> tuple[EvidenceRef, EvidenceRef]:
        """Persist the exact input plus a bounded operator-facing thumbnail."""

        exact = self.put(
            kind="input_rgb",
            media_type="image/jpeg",
            data=image_jpeg,
            producer="observation_gateway",
        )
        try:
            with Image.open(BytesIO(image_jpeg)) as source:
                image = source.convert("RGB")
                image.thumbnail((320, 240), Image.Resampling.LANCZOS)
                output = BytesIO()
                image.save(output, format="JPEG", quality=82, optimize=True)
        except Exception as exc:
            raise ValueError("evidence RGB input is not a decodable image") from exc
        thumbnail = self.put(
            kind="input_rgb_thumbnail",
            media_type="image/jpeg",
            data=output.getvalue(),
            producer="observation_gateway",
        )
        return exact, thumbnail

    def put_mask(self, mask: np.ndarray, *, model: str) -> EvidenceRef:
        value = np.asarray(mask, dtype=bool)
        if value.ndim != 2 or not value.any():
            raise ValueError("SAM 2 evidence mask must be a non-empty 2D mask")
        output = BytesIO()
        Image.fromarray(value.astype(np.uint8) * 255).save(
            output, format="PNG", optimize=True
        )
        return self.put(
            kind="sam2_mask",
            media_type="image/png",
            data=output.getvalue(),
            producer="sam2",
            model=model,
        )

    def get(self, sha256: str) -> bytes:
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError("invalid evidence hash")
        return (self.root / sha256[:2] / sha256).read_bytes()
