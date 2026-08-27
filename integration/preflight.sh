#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
COMPONENTS="${ROOT}/components"
ENV_FILE="${SKILLSTORE_ENV_FILE:-${ROOT}/.env}"
LEVEL=L0

usage() {
  printf '%s\n' 'usage: integration/preflight.sh [--level L0|L1|L2]'
  printf '%s\n' '  L0: repository, secret and manifest checks (no runtime assets required)'
  printf '%s\n' '  L1: additionally validate Docker/scene/build configuration'
  printf '%s\n' '  L2: additionally require ROS/GPU and every external runtime artifact'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --level)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      LEVEL="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done
case "${LEVEL}" in
  L0|L1|L2) ;;
  *) printf 'unsupported level: %s\n' "${LEVEL}" >&2; exit 2 ;;
esac

failures=0
warnings=0
fail() { printf 'FAIL: %s\n' "$*" >&2; failures=$((failures + 1)); }
warn() { printf 'WARN: %s\n' "$*" >&2; warnings=$((warnings + 1)); }
need_cmd() { command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"; }

for component in h-ide h-ide-agent h-ide-vision robotagent h-ide-mujoco-sonic; do
  [[ -d "${COMPONENTS}/${component}" ]] || fail "missing component: ${component}"
done

if [[ ! -d "${ROOT}/.git" ]]; then
  fail 'repository is not a Git worktree'
else
  if git -C "${ROOT}" ls-files -s | awk '$1 == 160000 {print; found=1} END {exit found ? 0 : 1}'; then
    fail 'Git gitlink entries remain; dependency migration is incomplete'
  fi
  if ! python3 - "${ROOT}" <<'PY'
import subprocess
import sys

root = sys.argv[1]
paths = subprocess.check_output(["git", "-C", root, "ls-files"], text=True).splitlines()
bad = []
allowed_example_env = {
    "third_party/lerobot/examples/isaac_teleop_to_so101/default.env",
}
for path in paths:
    low = path.casefold()
    is_example_env = path in allowed_example_env
    top_level_runtime_dir = path.split("/", 1)[0] in {
        "var", "data", "logs", "log", "runs", "evidence"
    }
    if (
        top_level_runtime_dir
        or (low.endswith(".env") and not is_example_env)
        or low.endswith((".db", ".sqlite", ".sqlite3", ".onnx", ".pt", ".pth", ".safetensors"))
    ):
        bad.append(path)
if bad:
    print("tracked runtime/secrets/artifacts:", *bad, sep="\n- ")
    raise SystemExit(1)
PY
  then
    fail 'forbidden runtime or secret file is tracked'
  fi
fi

if [[ -e "${ENV_FILE}" || -L "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_FILE}" || -L "${ENV_FILE}" ]]; then
    fail "environment file is not a regular file: ${ENV_FILE}"
  else
    mode="$(stat -c '%a' "${ENV_FILE}" 2>/dev/null || printf unknown)"
    [[ "${mode}" == 600 ]] || fail "environment file must have mode 0600 (got ${mode})"
    for key in CONTROL_API_TOKEN SCENE_API_INTERNAL_TOKEN MASTRA_INTERNAL_TOKEN MISSION_V2_INTERNAL_TOKEN; do
      value="$(sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1)"
      [[ -n "${value}" && "${value}" != replace-with-* ]] || fail "${key} is empty or still a placeholder"
    done
    for key in DEEPSEEK_API_KEY DASHSCOPE_API_KEY; do
      value="$(sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1)"
      [[ -n "${value}" ]] || warn "${key} is empty; model-backed L2 tasks will be unavailable"
    done
  fi
elif [[ "${LEVEL}" == L0 ]]; then
  warn "${ENV_FILE} is absent; run integration/bootstrap.sh"
else
  fail "${ENV_FILE} is absent; run integration/bootstrap.sh"
fi

sim="${COMPONENTS}/h-ide-mujoco-sonic"
critical=(
  "${sim}/config/sim.yaml"
  "${sim}/config/headless.yaml"
  "${sim}/config/ros-headless.yaml"
  "${sim}/config/sonic-models.lock.json"
  "${sim}/scripts/start-workstation-demo.sh"
)
for path in "${critical[@]}"; do
  [[ -f "${path}" ]] || fail "missing critical file: ${path}"
done

if rg -n '/home/phi/learn-projects|systemed_study' \
  "${sim}/config" \
  "${sim}/scripts/start-workstation-demo.sh" \
  "${sim}/scripts/workstation-demo-status.sh" \
  "${COMPONENTS}/h-ide/infra/docker-compose.yml" \
  >/tmp/skillstore-absolute-paths; then
  fail 'critical runtime files still contain developer-machine absolute paths'
fi

if ! python3 - "${ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = [
    root / "manifests/vendor/g1-model.json",
    root / "manifests/vendor/groot-wbc.json",
    root / "manifests/vendor/lerobot.json",
    root / "manifests/vendor/unitree-sdk2-python.json",
    root / "components/h-ide-mujoco-sonic/config/sonic-models.lock.json",
]
for path in paths:
    json.loads(path.read_text(encoding="utf-8"))
lock = json.loads(paths[-1].read_text(encoding="utf-8"))
if lock.get("schema_version") != "h-ide.sonic-model-lock/1":
    raise SystemExit("unexpected SONIC lock schema")
patch = root / "third_party/g1-model-office.patch"
manifest = json.loads((root / "manifests/vendor/g1-model.json").read_text(encoding="utf-8"))
if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest.get("overlay_sha256"):
    raise SystemExit("G1 overlay SHA-256 does not match its manifest")
for name in ("lerobot.json", "groot-wbc.json", "unitree-sdk2-python.json"):
    value = json.loads((root / "manifests/vendor" / name).read_text(encoding="utf-8"))
    if not value.get("upstream_revision"):
        raise SystemExit(f"vendor manifest has no upstream_revision: {name}")
print("JSON/vendor manifests: OK")
PY
then
  fail 'JSON/vendor manifest validation failed'
fi

if [[ "${LEVEL}" != L0 ]]; then
  need_cmd docker
  if [[ -f "${ENV_FILE}" ]]; then
    docker compose --env-file "${ENV_FILE}" \
      -f "${COMPONENTS}/h-ide/infra/docker-compose.yml" \
      -f "${COMPONENTS}/h-ide/infra/docker-compose.simulation.yml" \
      config --quiet || fail 'Docker Compose configuration is invalid'
  fi
  if [[ -x "${sim}/.venv/bin/python" ]]; then
    (cd "${sim}" && .venv/bin/python tools/build_scene.py --check) \
      || fail 'Office V2 generated scene artifacts do not match scene.yaml'
  else
    warn 'MuJoCo Python environment is absent; scene check deferred'
  fi
fi

g1_path="${sim}/models/unitree-g1-mujoco"
groot_path="${sim}/external/GR00T-WholeBodyControl"
if [[ "${LEVEL}" == L2 ]]; then
  need_cmd git
  need_cmd ros2
  need_cmd nvidia-smi
  [[ -L "${g1_path}" && -f "${g1_path}/assets/h_ide_office_29dof.xml" ]] \
    || fail 'G1 model overlay is not installed; run bootstrap.sh --fetch-g1-model'
  if [[ ! -L "${groot_path}" || ! -d "${groot_path}" ]]; then
    fail 'GR00T source is not installed; run bootstrap.sh --fetch-groot-source'
  else
    groot_revision="$(git -C "${groot_path}" rev-parse HEAD 2>/dev/null || true)"
    [[ "${groot_revision}" == c374bae5b9039cd0ee71377e654d11ce1bc69e1d ]] \
      || fail 'GR00T source revision does not match the frozen manifest'
    if [[ -n "$(git -C "${groot_path}" status --porcelain=v1 --untracked-files=all 2>/dev/null || true)" ]]; then
      fail 'GR00T source checkout is dirty'
    fi
  fi
  sam2=""
  [[ -f "${ENV_FILE}" ]] && sam2="$(sed -n 's/^HIDE_VISION_SAM2_CHECKPOINT=//p' "${ENV_FILE}" | tail -n 1)"
  [[ -n "${sam2}" && -f "${sam2}" ]] || fail 'HIDE_VISION_SAM2_CHECKPOINT is missing'
  if [[ -n "${sam2}" && -f "${sam2}" ]]; then
    actual="$(sha256sum "${sam2}" | awk '{print $1}')"
    [[ "${actual}" == 7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69 ]] \
      || fail 'SAM2 checkpoint SHA-256 does not match the frozen manifest'
  fi
  if [[ -x "${sim}/.venv/bin/python" ]]; then
    (cd "${sim}" && .venv/bin/python tools/verify_model_lock.py \
      --lock config/sonic-models.lock.json --config config/ros-headless.yaml) \
      || fail 'SONIC runtime model lock verification failed'
  else
    warn 'SONIC model lock runtime check deferred until the simulator environment is installed'
  fi
else
  [[ -L "${g1_path}" ]] || warn 'G1 model is optional for L0/L1 and is not installed'
fi

if (( failures > 0 )); then
  printf 'preflight %s: FAILED (%s failures, %s warnings)\n' "${LEVEL}" "${failures}" "${warnings}" >&2
  exit 1
fi
printf 'preflight %s: OK (%s warnings)\n' "${LEVEL}" "${warnings}"
