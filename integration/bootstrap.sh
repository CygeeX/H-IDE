#!/usr/bin/env bash
set -euo pipefail

# Prepare an operator-local Skillstore workstation. The script is idempotent
# and fail-closed: it never overwrites a dependency whose identity differs
# from the release manifest, and it never writes a secret to Git.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
COMPONENTS="${ROOT}/components"
ENV_FILE="${SKILLSTORE_ENV_FILE:-${ROOT}/.env}"
DEFAULT_EXTERNAL_ROOT="${XDG_CACHE_HOME:-${HOME}/.cache}/skillstore"
G1_URL="https://huggingface.co/lerobot/unitree-g1-mujoco"
G1_BASE_REVISION="a38dc8617f0fca51b38e9354dc58ee35ad850fb5"
G1_PATCH="${ROOT}/third_party/g1-model-office.patch"
GROOT_URL="https://github.com/NVlabs/GR00T-WholeBodyControl.git"
GROOT_REVISION="c374bae5b9039cd0ee71377e654d11ce1bc69e1d"
SONIC_REPO="nvidia/GEAR-SONIC"
SONIC_REVISION="9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"
HUGGINGFACE_HUB_VERSION="1.29.0"

FETCH_SONIC=false
FETCH_G1=false
FETCH_GROOT=false
INSTALL_AGENT=false
PROVISION_ROS_PREFIXES=false

usage() {
  cat <<'EOF'
usage: integration/bootstrap.sh [options]

Options:
  --fetch-sonic-model  Download the locked GEAR-SONIC bundle (about 865 MB).
  --fetch-g1-model     Download the G1 simulator repository and apply the
                       reviewed Office overlay. Requires explicit terms acceptance.
  --fetch-groot-source Download the pinned NVIDIA GR00T/SONIC deployment source.
  --install-agent      Run npm ci for the vendored planning agent.
  --provision-ros-prefixes
                       Download, verify, and extract the frozen Nav2/RealSense
                       ROS packages into RobotAgent's user-space prefixes.
  --help               Show this help.

The default action only creates a local .env, component links, and reports
missing external assets. It does not download large files silently.
EOF
}

for argument in "$@"; do
  case "$argument" in
    --fetch-sonic-model) FETCH_SONIC=true ;;
    --fetch-g1-model) FETCH_G1=true ;;
    --fetch-groot-source) FETCH_GROOT=true ;;
    --install-agent) INSTALL_AGENT=true ;;
    --provision-ros-prefixes) PROVISION_ROS_PREFIXES=true ;;
    --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

test -d "${COMPONENTS}/h-ide"
test -d "${COMPONENTS}/h-ide-agent"
test -d "${COMPONENTS}/h-ide-vision"
test -d "${COMPONENTS}/robotagent"
test -d "${COMPONENTS}/h-ide-mujoco-sonic"

if [[ ! -e "${ENV_FILE}" ]]; then
  install -m 0600 "${SCRIPT_DIR}/.env.simulation.example" "${ENV_FILE}"
fi
if [[ ! -f "${ENV_FILE}" || -L "${ENV_FILE}" ]]; then
  printf 'refusing non-regular environment file: %s\n' "${ENV_FILE}" >&2
  exit 1
fi
chmod 0600 "${ENV_FILE}"

env_value() {
  local key="$1"
  local line=""
  line="$(sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1)"
  line="${line%$'\r'}"
  printf '%s' "${line}"
}

configured_external_root="$(env_value SKILLSTORE_EXTERNAL_ROOT)"
EXTERNAL_ROOT="${SKILLSTORE_EXTERNAL_ROOT:-${configured_external_root:-${DEFAULT_EXTERNAL_ROOT}}}"
if [[ "${EXTERNAL_ROOT}" != /* ]]; then
  EXTERNAL_ROOT="${ROOT}/${EXTERNAL_ROOT}"
fi
mkdir -p "$(dirname -- "${EXTERNAL_ROOT}")"
EXTERNAL_ROOT="$(cd -- "$(dirname -- "${EXTERNAL_ROOT}")" && pwd -P)/$(basename -- "${EXTERNAL_ROOT}")"

random_hex() {
  python3 -c 'import secrets; print(secrets.token_hex(32))'
}

set_env_value() {
  local key="$1"
  local value="$2"
  if grep -qE "^${key}=" "${ENV_FILE}"; then
    sed -i "s|^${key}=.*$|${key}=${value}|" "${ENV_FILE}"
  else
    printf '%s=%s\n' "${key}" "${value}" >>"${ENV_FILE}"
  fi
}

for key in CONTROL_API_TOKEN SCENE_API_INTERNAL_TOKEN MASTRA_INTERNAL_TOKEN MISSION_V2_INTERNAL_TOKEN \
  HIDE_VISION_AGENT_TOKEN HIDE_VISION_CONTROL_TOKEN \
  SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN ROBOT_AGENT_GATEWAY_TOKEN; do
  current="$(sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1)"
  if [[ -z "${current}" || "${current}" == replace-with-* ]]; then
    set_env_value "${key}" "$(random_hex)"
  fi
done

if ! grep -q '^SKILLSTORE_EXTERNAL_ROOT=' "${ENV_FILE}"; then
  printf 'SKILLSTORE_EXTERNAL_ROOT=%s\n' "${EXTERNAL_ROOT}" >>"${ENV_FILE}"
fi

# The host launcher historically reads the component-local .env path. Keep a
# relative link so the monorepo wrapper and legacy launcher share one secret
# file without copying it.
component_env="${COMPONENTS}/h-ide/.env"
if [[ -e "${component_env}" || -L "${component_env}" ]]; then
  if [[ ! -L "${component_env}" || "$(readlink "${component_env}")" != "../../.env" ]]; then
    printf 'component .env path is occupied by a different file/link: %s\n' "${component_env}" >&2
    exit 1
  fi
else
  ln -s ../../.env "${component_env}"
fi

mkdir -p "${EXTERNAL_ROOT}"

configured_g1_root="$(env_value HIDE_G1_MODEL_ROOT)"
g1_target="${HIDE_G1_MODEL_ROOT:-${configured_g1_root:-${EXTERNAL_ROOT}/unitree-g1-mujoco}}"
if [[ "${g1_target}" != /* ]]; then
  g1_target="${ROOT}/${g1_target}"
fi
mkdir -p "$(dirname -- "${g1_target}")"
g1_target="$(cd -- "$(dirname -- "${g1_target}")" && pwd -P)/$(basename -- "${g1_target}")"
g1_component_path="${COMPONENTS}/h-ide-mujoco-sonic/models/unitree-g1-mujoco"
mkdir -p "$(dirname -- "${g1_component_path}")"

configured_groot_root="$(env_value HIDE_GROOT_ROOT)"
groot_target="${HIDE_GROOT_ROOT:-${configured_groot_root:-${EXTERNAL_ROOT}/GR00T-WholeBodyControl}}"
if [[ "${groot_target}" != /* ]]; then
  groot_target="${ROOT}/${groot_target}"
fi
mkdir -p "$(dirname -- "${groot_target}")"
groot_target="$(cd -- "$(dirname -- "${groot_target}")" && pwd -P)/$(basename -- "${groot_target}")"
groot_component_path="${COMPONENTS}/h-ide-mujoco-sonic/external/GR00T-WholeBodyControl"

if [[ "${FETCH_GROOT}" == true ]]; then
  if [[ -e "${groot_target}" || -L "${groot_target}" ]]; then
    test -d "${groot_target}"
  else
    # This release needs the pinned controller source for provenance only.
    # The upstream repository also references a large LFS training corpus that
    # is neither executed nor licensed for redistribution by this workspace.
    GIT_LFS_SKIP_SMUDGE=1 git clone "${GROOT_URL}" "${groot_target}"
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "${groot_target}" fetch --depth 1 origin "${GROOT_REVISION}"
  GIT_LFS_SKIP_SMUDGE=1 git -C "${groot_target}" checkout --detach "${GROOT_REVISION}"
  test "$(git -C "${groot_target}" rev-parse HEAD)" = "${GROOT_REVISION}"
fi

if [[ -e "${groot_component_path}" || -L "${groot_component_path}" ]]; then
  [[ -L "${groot_component_path}" ]] || {
    printf 'GR00T component path must be a link to an external checkout: %s\n' "${groot_component_path}" >&2
    exit 1
  }
  resolved_groot="$(readlink -f -- "${groot_component_path}" 2>/dev/null || true)"
  [[ -n "${resolved_groot}" && "${resolved_groot}" == "${groot_target}" ]] || {
    printf 'GR00T component link does not match HIDE_GROOT_ROOT: %s\n' "${groot_component_path}" >&2
    exit 1
  }
elif [[ -d "${groot_target}" ]]; then
  mkdir -p "$(dirname -- "${groot_component_path}")"
  ln -s "${groot_target}" "${groot_component_path}"
else
  printf '%s\n' 'GR00T source is not installed; official SONIC profiles remain unavailable until --fetch-groot-source.'
fi

if [[ "${FETCH_G1}" == true ]]; then
  accepted="${SKILLSTORE_ACCEPT_G1_MODEL_TERMS:-$(env_value SKILLSTORE_ACCEPT_G1_MODEL_TERMS)}"
  if [[ "${accepted}" != 1 ]]; then
    printf '%s\n' 'G1 model download requires SKILLSTORE_ACCEPT_G1_MODEL_TERMS=1 after reviewing the upstream license.' >&2
    exit 2
  fi
  if [[ -e "${g1_target}" || -L "${g1_target}" ]]; then
    test -d "${g1_target}"
  else
    git clone "${G1_URL}" "${g1_target}"
  fi
  git -C "${g1_target}" fetch --depth 1 origin "${G1_BASE_REVISION}"
  git -C "${g1_target}" checkout --detach "${G1_BASE_REVISION}"
  command -v git-lfs >/dev/null 2>&1 || {
    printf '%s\n' 'git-lfs is required to obtain the G1 mesh assets.' >&2
    exit 1
  }
  git -C "${g1_target}" lfs pull
  if git -C "${g1_target}" apply --check "${G1_PATCH}"; then
    git -C "${g1_target}" apply "${G1_PATCH}"
  elif git -C "${g1_target}" apply --reverse --check "${G1_PATCH}"; then
    printf '%s\n' 'G1 Office overlay is already applied; leaving the clean overlay intact.'
  else
    printf '%s\n' 'G1 checkout is neither a clean base nor the reviewed Office overlay.' >&2
    exit 1
  fi
  test -f "${g1_target}/assets/h_ide_office_29dof.xml"
fi

if [[ -e "${g1_component_path}" || -L "${g1_component_path}" ]]; then
  if [[ ! -L "${g1_component_path}" ]]; then
    printf 'G1 component path must be a link to an external, license-accepted checkout: %s\n' "${g1_component_path}" >&2
    exit 1
  fi
  resolved_g1="$(readlink -f -- "${g1_component_path}" 2>/dev/null || true)"
  [[ -n "${resolved_g1}" && "${resolved_g1}" == "${g1_target}" ]] || {
    printf 'G1 component link does not match HIDE_G1_MODEL_ROOT: %s\n' "${g1_component_path}" >&2
    exit 1
  }
elif [[ -d "${g1_target}" ]]; then
  ln -s "${g1_target}" "${g1_component_path}"
else
  printf '%s\n' 'G1 model is not installed; rerun with --fetch-g1-model and explicit terms acceptance for L2 simulation.'
fi

if [[ "${FETCH_SONIC}" == true ]]; then
  configured_sonic_cache="$(env_value SONIC_HF_CACHE)"
  sonic_cache="${SONIC_HF_CACHE:-${configured_sonic_cache:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface/hub}}"
  mkdir -p "${sonic_cache}"
  bootstrap_tools="${EXTERNAL_ROOT}/bootstrap-tools"
  bootstrap_python="${bootstrap_tools}/bin/python"
  if [[ ! -x "${bootstrap_python}" ]]; then
    python3 -m venv "${bootstrap_tools}"
  fi
  if ! "${bootstrap_python}" - "${HUGGINGFACE_HUB_VERSION}" <<'PY'
import importlib.metadata
import sys

expected = sys.argv[1]
try:
    actual = importlib.metadata.version("huggingface-hub")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if actual == expected else 1)
PY
  then
    "${bootstrap_python}" -m pip install --disable-pip-version-check \
      "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"
  fi
  "${bootstrap_python}" - \
    "${sonic_cache}" "${SONIC_REPO}" "${SONIC_REVISION}" \
    "${COMPONENTS}/h-ide-mujoco-sonic/config/sonic-models.lock.json" <<'PY'
import json
from pathlib import Path
import sys
try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    raise SystemExit(f"huggingface_hub is required for --fetch-sonic-model: {exc}")
cache, repo, revision, lock_path = sys.argv[1:]
lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
profile = lock["profiles"]["sonic-python-experimental"]
marker = f"/snapshots/{revision}/"
patterns = sorted(
    {
        entry["path"].split(marker, 1)[1]
        for entry in profile["files"]
        if marker in entry.get("path", "")
    }
)
if not patterns:
    raise SystemExit("SONIC runtime lock contains no model files to fetch")
snapshot_download(
    repo_id=repo,
    revision=revision,
    cache_dir=cache,
    allow_patterns=patterns,
)
PY
fi

if [[ "${INSTALL_AGENT}" == true ]]; then
  command -v npm >/dev/null 2>&1 || {
    printf '%s\n' 'npm is required for --install-agent' >&2
    exit 1
  }
  (cd "${COMPONENTS}/h-ide-agent" && npm ci)
fi

if [[ "${PROVISION_ROS_PREFIXES}" == true ]]; then
  "${SCRIPT_DIR}/provision-ros-prefixes.sh"
fi

printf '%s\n' 'Skillstore bootstrap checks:'
printf '%s\n' "- environment: ${ENV_FILE} (0600)"
printf '%s\n' "- external root: ${EXTERNAL_ROOT}"
printf '%s\n' '- LeRobot: vendored in third_party/lerobot'
printf '%s\n' '- Unitree SDK: vendored in third_party/unitree_sdk2_python'
if [[ -d "${COMPONENTS}/robotagent/.local/ros-jazzy-nav2/root/opt/ros/jazzy/share/nav2_msgs" \
   && -d "${COMPONENTS}/robotagent/.local/ros-jazzy-realsense/root/opt/ros/jazzy/share/realsense2_camera" ]]; then
  printf '%s\n' '- ROS user prefixes: installed and manifest-locked'
else
  printf '%s\n' '- ROS user prefixes: missing (run with --provision-ros-prefixes before L2)'
fi
if [[ -e "${g1_component_path}" ]]; then
  printf '%s\n' "- G1 model: linked ${g1_component_path}"
else
  printf '%s\n' '- G1 model: missing (required for L2)'
fi
printf '%s\n' 'Next: run integration/preflight.sh --level L0, then configure external keys and assets for L1/L2.'
