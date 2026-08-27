from __future__ import annotations

import json


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "GPU capability unavailable: install the locked Blackwell-capable PyTorch environment"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("GPU capability unavailable: torch.cuda.is_available() is false")
    properties = torch.cuda.get_device_properties(0)
    value = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    result = value @ value
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "device": properties.name,
                "computeCapability": f"{properties.major}.{properties.minor}",
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "checksum": float(result.sum().cpu()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
