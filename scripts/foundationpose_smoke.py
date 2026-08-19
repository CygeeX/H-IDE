from __future__ import annotations

from pathlib import Path


def main() -> None:
    source = Path("/home/phi/learn-projects/FoundationPose-main")
    required = (
        source / "LICENSE",
        source / "weights/2023-10-28-18-33-37/model_best.pth",
        source / "weights/2024-01-11-20-02-45/model_best.pth",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"FoundationPose capability unavailable: missing {missing}")
    weights = required[1:]
    lfs_pointers = [
        str(path)
        for path in weights
        if path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    ]
    if lfs_pointers:
        raise SystemExit(
            "FoundationPose capability unavailable: model weights are Git LFS pointer files, "
            f"not checkpoint bytes: {lfs_pointers}"
        )
    try:
        import nvdiffrast.torch  # noqa: F401
        import pytorch3d  # noqa: F401
        import torch
    except ImportError as exc:
        raise SystemExit(
            f"FoundationPose capability unavailable: optional GPU dependency is missing: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("FoundationPose capability unavailable: CUDA is not available")
    print("FoundationPose dependencies and real checkpoint files are present; run the sample next")


if __name__ == "__main__":
    main()
