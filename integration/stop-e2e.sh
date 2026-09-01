#!/usr/bin/env bash
set -euo pipefail

# Stop only the named Skillstore motion units and Compose project. No volumes,
# databases, logs, or external model checkouts are deleted by this command.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ENV_FILE="${SKILLSTORE_ENV_FILE:-${ROOT}/.env}"
PREFIX="${HIDE_DEMO_SYSTEMD_PREFIX:-h-ide-mujoco-demo}"

units=(
  robot-agent-diagnostics-proxy
  robot-agent-g1-sim
  observation-gateway
  sim-robot-agent-gateway
  strict-navigation-adapter
  ros-safe-stop
  ros-observation-bridge
  ros-cmd-vel-bridge
  first-person-viewer
  leader-demo-gesture-proxy
  h-ide-vision-qwen
  sonic-driver
  mujoco-plant
)

for suffix in "${units[@]}"; do
  systemctl --user stop "${PREFIX}-${suffix}.service" >/dev/null 2>&1 || true
done

if docker container inspect h-ide-nav2-sim >/dev/null 2>&1; then
  docker stop --time 10 h-ide-nav2-sim >/dev/null 2>&1 || true
fi

if command -v docker >/dev/null 2>&1 && [[ -f "${ENV_FILE}" ]]; then
  docker compose \
    --project-name "${SKILLSTORE_COMPOSE_PROJECT:-skillstore-e2e}" \
    --env-file "${ENV_FILE}" \
    -f "${ROOT}/components/h-ide/infra/docker-compose.yml" \
    -f "${ROOT}/components/h-ide/infra/docker-compose.simulation.yml" \
    down --remove-orphans
fi

printf '%s\n' 'Skillstore E2E services stopped. Persistent volumes and external assets were kept.'
