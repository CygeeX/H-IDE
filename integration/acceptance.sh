#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
LEVEL=L0
EXECUTE_MOTION=false

usage() {
  cat <<'EOF'
usage: integration/acceptance.sh [--level L0|L1|L2] [--execute-motion]

L0 checks repository hygiene, manifests, shell syntax and portable selectors.
L1 adds deterministic component contract/unit tests and Compose/scene checks.
L2 adds live service health and model/ROS/GPU checks; motion cases require the
   explicit --execute-motion switch and are never sent implicitly.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --level) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; LEVEL="$2"; shift 2 ;;
    --execute-motion) EXECUTE_MOTION=true; shift ;;
    --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
case "${LEVEL}" in L0|L1|L2) ;; *) printf 'unsupported level: %s\n' "${LEVEL}" >&2; exit 2 ;; esac

"${SCRIPT_DIR}/preflight.sh" --level "${LEVEL}"

printf '%s\n' 'Acceptance: shell and manifest syntax'
for script in "${SCRIPT_DIR}"/*.sh "${ROOT}/components/h-ide-mujoco-sonic/scripts"/*.sh; do
  bash -n "${script}"
done
python3 - "${ROOT}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for path in [
    root / "components/h-ide-mujoco-sonic/config/sonic-models.lock.json",
    root / "manifests/vendor/g1-model.json",
    root / "manifests/vendor/groot-wbc.json",
    root / "manifests/vendor/lerobot.json",
    root / "manifests/vendor/unitree-sdk2-python.json",
]:
    json.loads(path.read_text(encoding="utf-8"))
print("JSON manifests: OK")
PY

if [[ "${LEVEL}" == L0 ]]; then
  printf '%s\n' 'acceptance L0: OK'
  exit 0
fi

run_pytest_if_ready() {
  local dir="$1"; shift
  local python_bin="${dir}/.venv/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    printf 'missing %s; run the environment setup in docs/reproduction/README.md\n' "${python_bin}" >&2
    return 1
  fi
  (cd "${dir}" && "${python_bin}" -m pytest -q "$@")
}

printf '%s\n' 'Acceptance: deterministic component contracts'
run_pytest_if_ready "${ROOT}/components/h-ide" \
  tests/test_end_to_end.py tests/test_visual_mission_runtime.py tests/test_daaam_memory.py
run_pytest_if_ready "${ROOT}/components/h-ide-vision" \
  tests/contracts/test_contract_lock.py tests/unit/test_product_provider_boundary.py \
  tests/unit/test_geometry.py tests/unit/test_visual_task_v2.py
run_pytest_if_ready "${ROOT}/components/robotagent" \
  tests/test_config.py tests/test_capabilities_v2.py tests/test_protocol_families.py
run_pytest_if_ready "${ROOT}/components/h-ide-mujoco-sonic" \
  tests/test_model_lock.py tests/test_nav2_config.py tests/test_office_v2_scene.py \
  tests/test_reset_policy.py

if command -v npm >/dev/null 2>&1 && [[ -d "${ROOT}/components/h-ide-agent/node_modules" ]]; then
  (cd "${ROOT}/components/h-ide-agent" && npm test)
  (cd "${ROOT}/components/h-ide-agent" && npm run typecheck)
else
  printf '%s\n' 'missing h-ide-agent/node_modules; run npm ci before L1 acceptance' >&2
  exit 1
fi

if [[ "${LEVEL}" == L1 ]]; then
  printf '%s\n' 'acceptance L1: OK'
  exit 0
fi

printf '%s\n' 'Acceptance: live service health'
curl -fsS --max-time 5 http://127.0.0.1:8794/healthz >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8792/healthz >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8092/healthz >/dev/null
curl -fsS --max-time 5 http://127.0.0.1:8080/readyz >/dev/null

if [[ "${EXECUTE_MOTION}" == true ]]; then
  sim="${ROOT}/components/h-ide-mujoco-sonic"
  [[ -x "${sim}/.venv/bin/python" ]] || { printf '%s\n' 'simulator venv is required for motion acceptance' >&2; exit 1; }
  (cd "${sim}" && .venv/bin/python tools/strict_motion_runner.py --help >/dev/null)
  printf '%s\n' 'Motion execution is intentionally a separate operator-reviewed step.'
  printf '%s\n' 'Run one named case from docs/acceptance/README.md; this script does not guess a route.'
fi

printf '%s\n' 'acceptance L2: OK'
