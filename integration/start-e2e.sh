#!/usr/bin/env bash
set -euo pipefail

# Monorepo entry point. It supplies absolute paths derived from the clone root
# and then delegates all motion-specific ordering/gates to the reviewed
# simulator launcher.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ENV_FILE="${SKILLSTORE_ENV_FILE:-${ROOT}/.env}"

if [[ "${1:-}" == "--help" ]]; then
  cat <<'EOF'
usage: integration/start-e2e.sh [--rebuild] [--viewer] [--motion-core]
  Start the frozen Office V2 H-IDE/Mission/Vision/Robot Agent/Nav2/SONIC chain.
EOF
  exit 0
fi

if [[ ! -f "${ENV_FILE}" || -L "${ENV_FILE}" ]]; then
  printf '%s\n' "Missing regular ${ENV_FILE}; run integration/bootstrap.sh first." >&2
  exit 2
fi

# Values in .env are operator-owned configuration. Export them for Compose,
# while the launcher receives explicit paths so it does not depend on cwd.
set -a
# shellcheck disable=SC1090
. "${ENV_FILE}"
set +a

export SKILLSTORE_ROOT="${ROOT}"
export HIDE_DEMO_H_IDE_ROOT="${ROOT}/components/h-ide"
export HIDE_DEMO_AGENT_ROOT="${ROOT}/components/h-ide-agent"
export HIDE_DEMO_VISION_ROOT="${ROOT}/components/h-ide-vision"
export HIDE_DEMO_ROBOT_ROOT="${ROOT}/components/robotagent"
export HIDE_DEMO_SIM_ROOT="${ROOT}/components/h-ide-mujoco-sonic"
export HIDE_DEMO_CONFIG="${HIDE_DEMO_CONFIG:-${HIDE_DEMO_SIM_ROOT}/config/ros-headless.yaml}"
export HIDE_DEMO_ROBOT_CONFIG="${HIDE_DEMO_ROBOT_CONFIG:-${HIDE_DEMO_ROBOT_ROOT}/src/robot_agent/config/robot_agent.mujoco.yaml}"
export HIDE_DEMO_OBSERVATION_CONFIG="${HIDE_DEMO_OBSERVATION_CONFIG:-${HIDE_DEMO_ROBOT_ROOT}/src/robot_observation_gateway/config/robot_observation_gateway.mujoco.yaml}"

"${SCRIPT_DIR}/preflight.sh" --level L1
exec "${HIDE_DEMO_SIM_ROOT}/scripts/start-workstation-demo.sh" "$@"
