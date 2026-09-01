#!/usr/bin/env bash
set -euo pipefail

# Build every source/runtime environment from this checkout. External G1,
# SONIC, SAM2 and cloud credentials remain the responsibility of bootstrap and
# the operator; this script installs only package dependencies and ROS builds.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
COMPONENTS="${ROOT}/components"
PYTHON_RUNTIME_MANIFEST="${ROOT}/manifests/python/simulation-runtime.json"

# First-time CUDA wheels are hundreds of megabytes and the upstream NVIDIA
# endpoint can pause longer than pip's short default read timeout. Keep exact
# versions/indexes unchanged while making transient transport stalls retryable.
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-180}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"

for command_name in python3.12 uv npm; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "${command_name}" >&2
    exit 1
  }
done
[[ -f /opt/ros/jazzy/setup.bash ]] || {
  printf '%s\n' 'ROS 2 Jazzy is required at /opt/ros/jazzy' >&2
  exit 1
}

readarray -t simulation_versions < <(python3.12 - "${PYTHON_RUNTIME_MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schemaVersion") != "skillstore.python-runtime/1":
    raise SystemExit("unsupported Python runtime manifest")
packages = payload["packages"]
print(payload["pytorchIndexUrl"])
print(packages["torch"])
print(packages["torchvision"])
print(packages["cyclonedds"])
PY
)
[[ "${#simulation_versions[@]}" -eq 4 ]]
pytorch_index="${simulation_versions[0]}"
torch_version="${simulation_versions[1]}"
torchvision_version="${simulation_versions[2]}"
cyclonedds_version="${simulation_versions[3]}"

ensure_venv() {
  local path="$1"
  if [[ ! -x "${path}/bin/python" ]]; then
    python3.12 -m venv "${path}"
  fi
}

hide="${COMPONENTS}/h-ide"
ensure_venv "${hide}/.venv"
"${hide}/.venv/bin/python" -m pip install --disable-pip-version-check \
  -e "${hide}[test,daaam]"

sim="${COMPONENTS}/h-ide-mujoco-sonic"
ensure_venv "${sim}/.venv"
"${sim}/.venv/bin/python" -m pip install --disable-pip-version-check \
  --index-url "${pytorch_index}" \
  "torch==${torch_version}" "torchvision==${torchvision_version}"
"${sim}/.venv/bin/python" -m pip install --disable-pip-version-check \
  -e "${sim}[runtime,test]"
"${sim}/.venv/bin/python" -m pip install --disable-pip-version-check \
  --no-deps -e "${ROOT}/third_party/unitree_sdk2_python"
"${sim}/.venv/bin/python" -m pip install --disable-pip-version-check \
  -e "${ROOT}/third_party/lerobot"
"${sim}/.venv/bin/python" - \
  "${torch_version}" "${torchvision_version}" "${cyclonedds_version}" <<'PY'
import importlib.metadata
import sys
import torch
import torchvision
import cyclonedds  # noqa: F401
import lerobot  # noqa: F401
import unitree_sdk2py  # noqa: F401

expected_torch, expected_vision, expected_dds = sys.argv[1:]
actual = {
    "torch": importlib.metadata.version("torch"),
    "torchvision": importlib.metadata.version("torchvision"),
    "cyclonedds": importlib.metadata.version("cyclonedds"),
}
expected = {
    "torch": expected_torch,
    "torchvision": expected_vision,
    "cyclonedds": expected_dds,
}
if actual != expected or torch.version.cuda != "12.8":
    raise SystemExit(f"simulation runtime mismatch: {actual}, CUDA={torch.version.cuda}")
print("simulation Python/CUDA runtime: OK")
PY

(
  cd "${COMPONENTS}/h-ide-vision"
  uv sync --locked --extra dev
  # gpu.lock intentionally mixes the official PyTorch CUDA index with PyPI.
  # Every package is exact-version locked; allow resolution across both
  # reviewed indexes instead of treating the first matching project name as
  # the only source.
  uv pip install --python .venv/bin/python \
    --index-strategy unsafe-best-match \
    -r requirements/gpu.lock
)

robot="${COMPONENTS}/robotagent"
if [[ ! -x "${robot}/.venv/bin/python" ]]; then
  (cd "${robot}" && uv venv --python /usr/bin/python3.12 --system-site-packages .venv)
fi
(cd "${robot}" && uv sync --locked --python .venv/bin/python --all-groups)
(
  cd "${robot}"
  # ROS generated setup files read optional variables before defining them and
  # are not nounset-safe. Restore the release script's strict mode immediately
  # after sourcing the three reviewed setup files.
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  # shellcheck disable=SC1091
  source tools/activate-local-nav2-jazzy.bash
  # shellcheck disable=SC1091
  source tools/activate-local-realsense-jazzy.bash
  set -u
  /usr/bin/python3 /usr/bin/colcon build --symlink-install
)

(cd "${COMPONENTS}/h-ide-agent" && npm ci)
(cd "${COMPONENTS}/h-ide/apps/control-pwa" && npm ci)

printf '%s\n' 'All Skillstore component environments are installed.'
