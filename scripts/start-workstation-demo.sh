#!/usr/bin/env bash
set -euo pipefail

# Non-interactive launchers do not export the graphical user's bus variables.
# Establish them before deriving the runtime path so logs/PIDs never alternate
# between /tmp and /run/user across otherwise identical starts.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/${UID}}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=${XDG_RUNTIME_DIR}/bus}"

HIDE_DEMO_H_IDE_ROOT="/home/phi/learn-projects/h-ide"
HIDE_DEMO_SIM_ROOT="/home/phi/learn-projects/h-ide-mujoco-sonic"
HIDE_DEMO_ROBOT_ROOT="/home/phi/learn-projects/robotagent"
HIDE_DEMO_VISION_ROOT="/home/phi/learn-projects/h-ide-vision"
HIDE_DEMO_RUN_ROOT="${XDG_RUNTIME_DIR:-/tmp}/h-ide-mujoco-demo-${UID}"
HIDE_DEMO_CONFIG="${HIDE_DEMO_SIM_ROOT}/config/ros-headless.yaml"
HIDE_DEMO_ROBOT_CONFIG="${HIDE_DEMO_ROBOT_ROOT}/src/robot_agent/config/robot_agent.mujoco.yaml"
HIDE_DEMO_OBSERVATION_CONFIG="${HIDE_DEMO_ROBOT_ROOT}/src/robot_observation_gateway/config/robot_observation_gateway.mujoco.yaml"
HIDE_DEMO_CYCLONE='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
HIDE_DEMO_SYSTEMD_PREFIX="h-ide-mujoco-demo"

# Codex/CI terminal scopes reap their complete process group when the command
# finishes, including children started through nohup.  Put every long-lived
# simulation worker in the logged-in user's systemd manager instead.  The
# explicit bus variables also make this work from non-interactive shells.

export HIDE_DEMO_H_IDE_ROOT HIDE_DEMO_SIM_ROOT HIDE_DEMO_ROBOT_ROOT HIDE_DEMO_VISION_ROOT
export HIDE_DEMO_CONFIG HIDE_DEMO_ROBOT_CONFIG HIDE_DEMO_OBSERVATION_CONFIG
export HIDE_DEMO_CYCLONE DISPLAY="${DISPLAY:-:0}"

for required in \
  "${HIDE_DEMO_H_IDE_ROOT}/.env" \
  "${HIDE_DEMO_CONFIG}" \
  "${HIDE_DEMO_ROBOT_CONFIG}" \
  "${HIDE_DEMO_OBSERVATION_CONFIG}" \
  "${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" \
  "${HIDE_DEMO_VISION_ROOT}/.venv/bin/h-ide-vision" \
  "${HIDE_DEMO_ROBOT_ROOT}/.venv/bin/python"; do
  if [[ ! -f "${required}" ]]; then
    printf 'missing required file: %s\n' "${required}" >&2
    exit 2
  fi
done

mkdir -p "${HIDE_DEMO_RUN_ROOT}"

env_value() {
  local key="$1"
  local fallback="$2"
  local line=""
  line="$(sed -n "s/^${key}=//p" "${HIDE_DEMO_H_IDE_ROOT}/.env" | tail -n 1)"
  line="${line%$'\r'}"
  if [[ -n "${line}" ]]; then
    printf '%s' "${line}"
  else
    printf '%s' "${fallback}"
  fi
}

HIDE_DEMO_GATEWAY_TOKEN="$(env_value SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN dev-sim-gateway-token)"
export HIDE_DEMO_GATEWAY_TOKEN

start_process() {
  local name="$1"
  local marker="$2"
  local working_directory="$3"
  local command="$4"
  local unit="${HIDE_DEMO_SYSTEMD_PREFIX}-${name}.service"
  local log="${HIDE_DEMO_RUN_ROOT}/${name}.log"
  local existing=""
  local main_pid=""

  if systemctl --user is-active --quiet "${unit}"; then
    main_pid="$(systemctl --user show -p MainPID --value "${unit}")"
    printf '%-28s already running (pid=%s)\n' "${name}" "${main_pid}"
    printf '%s\n' "${main_pid}" >"${HIDE_DEMO_RUN_ROOT}/${name}.pid"
    return
  fi
  # ``pgrep -f`` also matches the parent shell because this script's command
  # line can contain every launch marker.  Restrict discovery to the actual
  # Python workers so a stopped service is never reported as already running.
  existing="$(
    ps -eo pid=,comm=,args= \
      | awk -v marker="${marker}" '$2 ~ /^(python|uvicorn)/ && $0 ~ marker { print $1; exit }'
  )"
  if [[ -n "${existing}" ]]; then
    printf '%-28s already running (pid=%s)\n' "${name}" "${existing}"
    return
  fi

  # A failed/stopped transient unit may still be loaded for a moment.  Stop it
  # and reset its state before recreating the stable unit name.
  systemctl --user stop "${unit}" >/dev/null 2>&1 || true
  systemctl --user reset-failed "${unit}" >/dev/null 2>&1 || true
  truncate -s 0 "${log}"
  systemd-run --user \
    --unit="${unit%.service}" \
    --collect \
    --working-directory="${working_directory}" \
    --property=Restart=no \
    --property=TimeoutStopSec=10 \
    --property="StandardOutput=append:${log}" \
    --property="StandardError=append:${log}" \
    --setenv="DISPLAY=${DISPLAY}" \
    --setenv="HIDE_DEMO_H_IDE_ROOT=${HIDE_DEMO_H_IDE_ROOT}" \
    --setenv="HIDE_DEMO_SIM_ROOT=${HIDE_DEMO_SIM_ROOT}" \
    --setenv="HIDE_DEMO_ROBOT_ROOT=${HIDE_DEMO_ROBOT_ROOT}" \
    --setenv="HIDE_DEMO_VISION_ROOT=${HIDE_DEMO_VISION_ROOT}" \
    --setenv="HIDE_DEMO_CONFIG=${HIDE_DEMO_CONFIG}" \
    --setenv="HIDE_DEMO_ROBOT_CONFIG=${HIDE_DEMO_ROBOT_CONFIG}" \
    --setenv="HIDE_DEMO_OBSERVATION_CONFIG=${HIDE_DEMO_OBSERVATION_CONFIG}" \
    --setenv="HIDE_DEMO_CYCLONE=${HIDE_DEMO_CYCLONE}" \
    --setenv="HIDE_DEMO_GATEWAY_TOKEN=${HIDE_DEMO_GATEWAY_TOKEN}" \
    /bin/bash -lc "${command}" >/dev/null

  for _ in {1..30}; do
    main_pid="$(systemctl --user show -p MainPID --value "${unit}" 2>/dev/null || true)"
    if [[ "${main_pid}" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s\n' "${main_pid}" >"${HIDE_DEMO_RUN_ROOT}/${name}.pid"
      printf '%-28s started (pid=%s)\n' "${name}" "${main_pid}"
      return
    fi
    sleep 0.1
  done
  printf '%s did not remain active; inspect %s\n' "${name}" "${log}" >&2
  return 1
}

wait_for_http() {
  local label="$1"
  local url="$2"
  local timeout_seconds="${3:-45}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 2 "${url}" >/dev/null 2>&1; then
      printf '%-28s ready\n' "${label}"
      return 0
    fi
    sleep 1
  done
  printf '%s did not become ready within %ss: %s\n' "${label}" "${timeout_seconds}" "${url}" >&2
  return 1
}

wait_for_qwen_vision() {
  local timeout_seconds="${1:-120}"
  local deadline=$((SECONDS + timeout_seconds))
  local health=""
  while (( SECONDS < deadline )); do
    health="$(curl -fsS --max-time 3 http://127.0.0.1:8092/healthz 2>/dev/null || true)"
    if [[ "${health}" == *'"provider":"QwenVlmProvider"'* \
      && "${health}" == *'"provider":"sam2","available":true'* ]]; then
      printf '%-28s ready\n' 'H-IDE Vision Qwen + SAM2'
      return 0
    fi
    sleep 1
  done
  printf 'H-IDE Vision did not expose Qwen + SAM 2 readiness within %ss\n' "${timeout_seconds}" >&2
  return 1
}

demo_ros_command() {
  bash -lc '
    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=232
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI="$HIDE_DEMO_CYCLONE"
    exec "$@"
  ' -- "$@"
}

wait_for_simulation_tf() {
  local timeout_seconds="${1:-45}"
  local deadline=$((SECONDS + timeout_seconds))
  local output=""
  while (( SECONDS < deadline )); do
    output="$(timeout 3 bash -lc '
      source /opt/ros/jazzy/setup.bash
      export ROS_DOMAIN_ID=232
      export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      export CYCLONEDDS_URI="$HIDE_DEMO_CYCLONE"
      ros2 run tf2_ros tf2_echo map base_link
    ' 2>&1 || true)"
    if grep -q -- '- Translation:' <<<"${output}"; then
      printf '%-28s ready\n' 'MuJoCo map -> base_link TF'
      return 0
    fi
    sleep 1
  done
  printf 'MuJoCo localization TF did not become ready within %ss\n' "${timeout_seconds}" >&2
  return 1
}

nav2_lifecycle_active() {
  local node=""
  local state=""
  for node in map_server planner_server controller_server behavior_server bt_navigator waypoint_follower; do
    state="$(demo_ros_command timeout 4 ros2 lifecycle get "/${node}" 2>&1 || true)"
    if [[ "${state}" != *'active [3]'* ]]; then
      return 1
    fi
  done
}

wait_for_nav2_active() {
  local timeout_seconds="${1:-60}"
  local deadline=$((SECONDS + timeout_seconds))
  while (( SECONDS < deadline )); do
    if nav2_lifecycle_active; then
      printf '%-28s ready\n' 'Nav2 lifecycle'
      return 0
    fi
    sleep 1
  done
  return 1
}

compose=(
  docker compose
  --env-file "${HIDE_DEMO_H_IDE_ROOT}/.env"
  -f "${HIDE_DEMO_H_IDE_ROOT}/infra/docker-compose.yml"
  -f "${HIDE_DEMO_H_IDE_ROOT}/infra/docker-compose.simulation.yml"
)

if [[ "${1:-}" == "--rebuild" ]]; then
  "${compose[@]}" up -d --build
else
  "${compose[@]}" up -d
fi

# The Compose image intentionally remains the CPU Oracle fixture. A pinned
# host GPU service owns the real simulation RGB-D ingress and Qwen/SAM 2 path.
# The Docker fixture is published on alternate ports by .env, while containers
# reach this host process through host.docker.internal:8092.
mkdir -p "${HIDE_DEMO_VISION_ROOT}/var/evidence" "${HIDE_DEMO_VISION_ROOT}/var/models"
start_process \
  h-ide-vision-qwen \
  'h-ide-vision serve' \
  "${HIDE_DEMO_VISION_ROOT}" \
  'env_get() { sed -n "s/^$1=//p" "$HIDE_DEMO_H_IDE_ROOT/.env" | tail -n 1; }; agent_token="$(env_get HIDE_VISION_AGENT_TOKEN)"; control_token="$(env_get HIDE_VISION_CONTROL_TOKEN)"; export DASHSCOPE_API_KEY="$(env_get DASHSCOPE_API_KEY)" HIDE_VISION_AGENT_TOKEN="${agent_token:-dev-agent-token}" HIDE_VISION_CONTROL_TOKEN="${control_token:-dev-control-token}" HIDE_VISION_SAM2_CHECKPOINT="$(env_get HIDE_VISION_SAM2_CHECKPOINT)"; test -n "$DASHSCOPE_API_KEY"; test -f "$HIDE_VISION_SAM2_CHECKPOINT"; export HIDE_VISION_HOST=0.0.0.0 HIDE_VISION_HTTP_PORT=8092 HIDE_VISION_GRPC_HOST=127.0.0.1 HIDE_VISION_GRPC_PORT=50061 HIDE_VISION_DATABASE_PATH="$HIDE_DEMO_VISION_ROOT/var/h-ide-vision-qwen.sqlite3" HIDE_VISION_EVIDENCE_DIR="$HIDE_DEMO_VISION_ROOT/var/evidence" HIDE_VISION_VLM_PROVIDER=qwen HIDE_VISION_SEGMENTATION_PROVIDER=sam2 HIDE_VISION_SAM2_DEVICE=cuda HIDE_VISION_QWEN_MODEL=qwen3-vl-plus HIDE_VISION_ALLOW_INSECURE_LOOPBACK=true; exec "$HIDE_DEMO_VISION_ROOT/.venv/bin/h-ide-vision" serve'

wait_for_qwen_vision 120

nav2_image="h-ide-nav2-sim:jazzy"
if ! docker image inspect "${nav2_image}" >/dev/null 2>&1; then
  docker build \
    -t "${nav2_image}" \
    -f "${HIDE_DEMO_SIM_ROOT}/docker/nav2/Dockerfile" \
    "${HIDE_DEMO_SIM_ROOT}/docker/nav2"
fi
# Nav2's lifecycle manager only makes one autostart attempt.  Keep it stopped
# until the simulation publishes map -> base_link; otherwise a cold start can
# permanently leave planner/controller/BT navigator inactive.
nav2_needs_start=true
if docker container inspect h-ide-nav2-sim >/dev/null 2>&1 \
  && [[ "$(docker inspect h-ide-nav2-sim --format '{{.State.Running}}')" == "true" ]]; then
  if nav2_lifecycle_active; then
    nav2_needs_start=false
    printf '%-28s already active\n' nav2
  else
    docker stop --time 10 h-ide-nav2-sim >/dev/null
  fi
fi

start_process \
  mujoco-plant \
  '-m hide_mujoco_sonic(\.plant)?( plant)? --config' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'exec "$HIDE_DEMO_SIM_ROOT/.venv/bin/python" -u -m hide_mujoco_sonic plant --config "$HIDE_DEMO_CONFIG"'

start_process \
  sonic-driver \
  '-m hide_mujoco_sonic sonic --config' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'exec "$HIDE_DEMO_SIM_ROOT/.venv/bin/python" -u -m hide_mujoco_sonic sonic --config "$HIDE_DEMO_CONFIG"'

start_process \
  ros-cmd-vel-bridge \
  '-m hide_mujoco_sonic ros-bridge --config' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=232 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="$HIDE_DEMO_CYCLONE"; export PYTHONPATH="$HIDE_DEMO_SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"; exec /usr/bin/python3 -u -m hide_mujoco_sonic ros-bridge --config "$HIDE_DEMO_CONFIG"'

start_process \
  ros-observation-bridge \
  '-m hide_mujoco_sonic ros-observation-bridge --config' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'source /opt/ros/jazzy/setup.bash; source "$HIDE_DEMO_ROBOT_ROOT/tools/activate-local-realsense-jazzy.bash"; export ROS_DOMAIN_ID=232 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="$HIDE_DEMO_CYCLONE"; export PYTHONPATH="$HIDE_DEMO_SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"; exec /usr/bin/python3 -u -m hide_mujoco_sonic ros-observation-bridge --config "$HIDE_DEMO_CONFIG"'

start_process \
  ros-safe-stop \
  '-m hide_mujoco_sonic(\.safe_stop_service| safe-stop-service) --config' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'source /opt/ros/jazzy/setup.bash; export ROS_DOMAIN_ID=232 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="$HIDE_DEMO_CYCLONE"; export PYTHONPATH="$HIDE_DEMO_SIM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"; exec /usr/bin/python3 -u -m hide_mujoco_sonic safe-stop-service --config "$HIDE_DEMO_CONFIG"'

wait_for_http 'MuJoCo plant + RGB-D' 'http://127.0.0.1:8794/healthz' 60
wait_for_simulation_tf 60

if [[ "${nav2_needs_start}" == "true" ]]; then
  if docker container inspect h-ide-nav2-sim >/dev/null 2>&1; then
    docker start h-ide-nav2-sim >/dev/null
  else
    docker run -d \
      --name h-ide-nav2-sim \
      --network host \
      --restart unless-stopped \
      -e ROS_DOMAIN_ID=232 \
      -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
      -e "CYCLONEDDS_URI=${HIDE_DEMO_CYCLONE}" \
      "${nav2_image}" >/dev/null
  fi
  printf '%-28s started after TF readiness\n' nav2
fi

if ! wait_for_nav2_active 60; then
  printf 'Nav2 activation failed; retrying once while simulation TF is available\n' >&2
  docker logs --tail 80 h-ide-nav2-sim >&2 || true
  wait_for_simulation_tf 15
  docker restart h-ide-nav2-sim >/dev/null
  if ! wait_for_nav2_active 60; then
    docker logs --tail 120 h-ide-nav2-sim >&2 || true
    printf 'Nav2 lifecycle did not become active after the recovery attempt\n' >&2
    exit 1
  fi
fi

start_process \
  sim-robot-agent-gateway \
  'uvicorn services.robot_agent_gateway.main:app --host 0.0.0.0 --port 8094' \
  "${HIDE_DEMO_H_IDE_ROOT}" \
  'export ROBOT_AGENT_GATEWAY_BACKEND=websocket_inbound ROBOT_AGENT_ID=g1-sim ROBOT_AGENT_ALLOW_PLAINTEXT_LOOPBACK=true ROBOT_AGENT_GATEWAY_TOKEN="$HIDE_DEMO_GATEWAY_TOKEN" ROBOT_AGENT_OBSERVATION_PREFLIGHT_TIMEOUT_SECONDS=2 ROBOT_AGENT_OBSERVATION_PREFLIGHT_MAX_AGE_MS=500; unset ROBOT_AGENT_CAPABILITIES_DIGEST ROBOT_AGENT_CAPABILITIES; exec "$HIDE_DEMO_H_IDE_ROOT/.venv/bin/uvicorn" services.robot_agent_gateway.main:app --host 0.0.0.0 --port 8094 --workers 1'

start_process \
  observation-gateway \
  '-m robot_observation_gateway.cli --config.*robot_observation_gateway.mujoco.yaml' \
  "${HIDE_DEMO_ROBOT_ROOT}" \
  'source /opt/ros/jazzy/setup.bash; source "$HIDE_DEMO_ROBOT_ROOT/install/setup.bash"; source "$HIDE_DEMO_ROBOT_ROOT/tools/activate-local-realsense-jazzy.bash"; export ROS_DOMAIN_ID=232 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="$HIDE_DEMO_CYCLONE"; export PYTHONPATH="$HIDE_DEMO_ROBOT_ROOT/src/robot_observation_gateway${PYTHONPATH:+:$PYTHONPATH}"; exec "$HIDE_DEMO_ROBOT_ROOT/.venv/bin/python" -u -m robot_observation_gateway.cli --config "$HIDE_DEMO_OBSERVATION_CONFIG"'

start_process \
  robot-agent-g1-sim \
  '-m robot_agent.cli run --config.*robot_agent.mujoco.yaml' \
  "${HIDE_DEMO_ROBOT_ROOT}" \
  'source /opt/ros/jazzy/setup.bash; source "$HIDE_DEMO_ROBOT_ROOT/tools/activate-local-nav2-jazzy.bash"; source "$HIDE_DEMO_ROBOT_ROOT/install/setup.bash"; export ROS_DOMAIN_ID=232 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI="$HIDE_DEMO_CYCLONE"; export PYTHONPATH="$HIDE_DEMO_ROBOT_ROOT/src/robot_agent${PYTHONPATH:+:$PYTHONPATH}"; exec "$HIDE_DEMO_ROBOT_ROOT/.venv/bin/python" -u -m robot_agent.cli run --config "$HIDE_DEMO_ROBOT_CONFIG"'

printf '\nRuntime logs: %s\n' "${HIDE_DEMO_RUN_ROOT}"
printf 'MuJoCo viewer uses DISPLAY=%s. Open http://127.0.0.1:8080/simulation for RGB-D and mission evidence.\n' "${DISPLAY}"
"${HIDE_DEMO_SIM_ROOT}/scripts/workstation-demo-status.sh"
