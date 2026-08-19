from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("var/models/sam2.1_hiera_tiny.pt"))
    args = parser.parse_args()
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise SystemExit(
            "SAM 2 capability unavailable: install the locked optional GPU dependencies"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("SAM 2 capability unavailable: CUDA is not available")
    if not args.checkpoint.is_file():
        raise SystemExit(f"SAM 2 capability unavailable: checkpoint not found at {args.checkpoint}")
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[64:192, 80:176] = (245, 196, 40)
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(args.checkpoint),
        device="cuda",
    )
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            box=np.asarray([72, 56, 184, 200], dtype=np.float32),
            multimask_output=True,
        )
    selected = np.asarray(masks[int(np.argmax(scores))], dtype=bool)
    mask_pixels = int(selected.sum())
    if selected.shape != image.shape[:2] or mask_pixels == 0:
        raise SystemExit("SAM 2 capability unavailable: fixed box prompt returned an empty mask")
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    peak = torch.cuda.max_memory_allocated()
    del predictor, model, masks, scores, selected
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "status": "available",
        "checkpoint": str(args.checkpoint),
        "checkpointSha256": checkpoint_hash,
        "maskPixels": mask_pixels,
        "peakCudaBytes": peak,
    }, indent=2))


if __name__ == "__main__":
    main()
