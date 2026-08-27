# GPU Compatibility Lock

## Detected hardware

- GPU: NVIDIA GeForce RTX 5060 Ti, 16,311 MiB.
- Driver: 595.71.05.
- Reported compute capability: 12.0 (Blackwell generation).
- Current independent Python 3.11 environment initially contains no PyTorch, SAM 2, or FoundationPose CUDA dependencies.

## Rule

Use a current PyTorch build with CUDA 12.8 or newer and native support for the detected compute capability. Do not reuse the FoundationPose snapshot's older `cu124` environment instructions without a successful compatibility spike.

The API/Ingress contract tests do not require CUDA because they inject doubles from `tests/fixtures`. Product startup does require the locked SAM 2 GPU stack and never routes to those doubles.

## Locked-version procedure

1. Create a dedicated Python 3.11 GPU environment.
2. Install a Blackwell-capable PyTorch/CUDA build and record exact wheel hashes.
3. Run `scripts/gpu_smoke.py`.
4. Install official SAM 2 at a pinned commit/checkpoint and run a real box-prompt segmentation.
5. Record Python, PyTorch, CUDA, driver, source commit/checkpoint hashes, peak memory and results before enabling the product profile.

## Compatibility result — 2026-08-12

- Python: 3.11; GPU: RTX 5060 Ti (compute capability 12.0); driver: 595.71.05; system NVCC: 13.0.88.
- PyTorch: `2.12.1+cu130`; torchvision: `0.27.1+cu130`. The CUDA matrix smoke passed twice.
- SAM 2: official commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`.
- SAM 2.1 Hiera Tiny checkpoint SHA-256: `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
- Fixed box-prompt inference returned a non-empty 12,288-pixel mask; peak allocated CUDA memory was 549,676,544 bytes.
- NVDiffRast: official commit `253ac4fcea7de5f396371124af597e6cc957bfae` built successfully.
- Qwen3-VL: adapter implemented but **unavailable** until a DashScope secret is injected and a live privacy-approved request passes.

The product runtime starts only when Qwen configuration and SAM 2 are both ready. FoundationPose is excluded from the installed package and retained only under `experiments/` for a future offline spike.
