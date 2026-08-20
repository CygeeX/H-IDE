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
HIDE_DOCKER_OBSERVATION_BIND_REQUEST="${HIDE_DOCKER_OBSERVATION_BIND_HOST:-auto}"
HIDE_DEMO_REBUILD=false
HIDE_MUJOCO_ONSCREEN_VIEWER=0

for argument in "$@"; do
  case "${argument}" in
    --rebuild)
      HIDE_DEMO_REBUILD=true
      ;;
    --viewer)
      HIDE_MUJOCO_ONSCREEN_VIEWER=1
      ;;
    *)
      printf 'usage: %s [--rebuild] [--viewer]\n' "$0" >&2
      exit 2
      ;;
  esac
done

# Codex/CI terminal scopes reap their complete process group when the command
# finishes, including children started through nohup.  Put every long-lived
# simulation worker in the logged-in user's systemd manager instead.  The
# explicit bus variables also make this work from non-interactive shells.

export HIDE_DEMO_H_IDE_ROOT HIDE_DEMO_SIM_ROOT HIDE_DEMO_ROBOT_ROOT HIDE_DEMO_VISION_ROOT
export HIDE_DEMO_CONFIG HIDE_DEMO_ROBOT_CONFIG HIDE_DEMO_OBSERVATION_CONFIG
export HIDE_DEMO_CYCLONE DISPLAY="${DISPLAY:-:0}"
HIDE_DEMO_XAUTHORITY="${XAUTHORITY:-${XDG_RUNTIME_DIR}/gdm/Xauthority}"
if [[ "${HIDE_MUJOCO_ONSCREEN_VIEWER}" == "1" ]]; then
  HIDE_DEMO_XAUTHORITY_RESOLVED="$(
    readlink -f -- "${HIDE_DEMO_XAUTHORITY}" 2>/dev/null || true
  )"
  if [[ ! -f "${HIDE_DEMO_XAUTHORITY_RESOLVED}" \
    || ! -r "${HIDE_DEMO_XAUTHORITY_RESOLVED}" ]]; then
    printf 'MuJoCo viewer Xauthority is not a readable regular file: %s\n' \
      "${HIDE_DEMO_XAUTHORITY}" >&2
    exit 2
  fi
  if [[ "$(stat -c '%u' "${HIDE_DEMO_XAUTHORITY_RESOLVED}")" != "${UID}" ]]; then
    printf 'MuJoCo viewer Xauthority is not owned by uid %s: %s\n' \
      "${UID}" "${HIDE_DEMO_XAUTHORITY_RESOLVED}" >&2
    exit 2
  fi
  HIDE_DEMO_XAUTHORITY="${HIDE_DEMO_XAUTHORITY_RESOLVED}"
  export XAUTHORITY="${HIDE_DEMO_XAUTHORITY}"
  if ! xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1; then
    printf 'MuJoCo viewer cannot connect to display %s\n' "${DISPLAY}" >&2
    exit 2
  fi
fi

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

if [[ "${HIDE_MUJOCO_ONSCREEN_VIEWER}" == "1" ]] \
  && systemctl --user is-active --quiet \
    "${HIDE_DEMO_SYSTEMD_PREFIX}-mujoco-plant.service"; then
  current_plant_state="$(
    curl -fsS --max-time 2 http://127.0.0.1:8794/state 2>/dev/null || true
  )"
  if [[ -z "${current_plant_state}" ]] || ! printf '%s' "${current_plant_state}" \
    | "${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" -c '
import json
import sys

try:
    viewer = json.load(sys.stdin).get("viewerWorker")
except (AttributeError, json.JSONDecodeError, TypeError):
    raise SystemExit(1)
ready = (
    isinstance(viewer, dict)
    and viewer.get("status") == "running"
    and viewer.get("ready") is True
    and viewer.get("alive") is True
)
raise SystemExit(0 if ready else 1)
'; then
    printf '%s\n' \
      'MuJoCo Plant is already running without a ready native viewer.' \
      'Stop the complete simulation chain before selecting --viewer.' >&2
    exit 3
  fi
fi

# Resolve Docker's current host-gateway instead of assuming a fixed subnet. The
# resolver also proves that the address belongs to Docker's local default
# bridge. Both plant and SONIC independently repeat this validation when they
# load SimConfig, so a changed daemon/network cannot silently widen the bind.
HIDE_DOCKER_OBSERVATION_BIND_HOST="$(
  "${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" \
    -m hide_mujoco_sonic.network \
    --requested "${HIDE_DOCKER_OBSERVATION_BIND_REQUEST}"
)"
: "${HIDE_DOCKER_OBSERVATION_BIND_HOST:?Docker observation gateway resolution returned empty}"
export HIDE_DOCKER_OBSERVATION_BIND_HOST
# Robot Agent keeps its diagnostic listener on loopback.  The separately
# constrained proxy below is the only Docker-visible route, so Compose must
# never point control-api directly at the loopback-only 8796 endpoint.
export SIMULATION_ROBOT_AGENT_DIAGNOSTICS_URL="http://host.docker.internal:8797"

mkdir -p "${HIDE_DEMO_RUN_ROOT}"

# The controller and its observation contract are one indivisible model bundle.
# Refuse to start any motion-capable process when the config, source revision,
# or cached model bytes differ from the reviewed lock.
"${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" \
  "${HIDE_DEMO_SIM_ROOT}/tools/verify_model_lock.py" \
  --lock "${HIDE_DEMO_SIM_ROOT}/config/sonic-models.lock.json" \
  --config "${HIDE_DEMO_CONFIG}"

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
  local operator_viewer_env_args=()

  if [[ "${name}" == "mujoco-plant" ]]; then
    operator_viewer_env_args=(
      --setenv="HIDE_MUJOCO_ONSCREEN_VIEWER=${HIDE_MUJOCO_ONSCREEN_VIEWER}"
    )
    if [[ "${HIDE_MUJOCO_ONSCREEN_VIEWER}" == "1" ]]; then
      operator_viewer_env_args+=(
        --setenv="XAUTHORITY=${HIDE_DEMO_XAUTHORITY}"
      )
    fi
  fi

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
    "${operator_viewer_env_args[@]}" \
    --setenv="HIDE_DEMO_H_IDE_ROOT=${HIDE_DEMO_H_IDE_ROOT}" \
    --setenv="HIDE_DEMO_SIM_ROOT=${HIDE_DEMO_SIM_ROOT}" \
    --setenv="HIDE_DEMO_ROBOT_ROOT=${HIDE_DEMO_ROBOT_ROOT}" \
    --setenv="HIDE_DEMO_VISION_ROOT=${HIDE_DEMO_VISION_ROOT}" \
    --setenv="HIDE_DEMO_CONFIG=${HIDE_DEMO_CONFIG}" \
    --setenv="HIDE_DEMO_ROBOT_CONFIG=${HIDE_DEMO_ROBOT_CONFIG}" \
    --setenv="HIDE_DEMO_OBSERVATION_CONFIG=${HIDE_DEMO_OBSERVATION_CONFIG}" \
    --setenv="HIDE_DEMO_CYCLONE=${HIDE_DEMO_CYCLONE}" \
    --setenv="HIDE_DEMO_GATEWAY_TOKEN=${HIDE_DEMO_GATEWAY_TOKEN}" \
    --setenv="HIDE_DOCKER_OBSERVATION_BIND_HOST=${HIDE_DOCKER_OBSERVATION_BIND_HOST}" \
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

wait_for_native_viewer() {
  if [[ "${HIDE_MUJOCO_ONSCREEN_VIEWER}" != "1" ]]; then
    return 0
  fi
  local deadline=$((SECONDS + 30))
  local state=""
  while (( SECONDS < deadline )); do
    state="$(curl -fsS --max-time 2 http://127.0.0.1:8794/state 2>/dev/null || true)"
    if [[ -n "${state}" ]] && printf '%s' "${state}" \
      | "${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" -c '
import json
import sys

try:
    state = json.load(sys.stdin)
except (json.JSONDecodeError, TypeError):
    raise SystemExit(1)
viewer = state.get("viewerWorker")
ready = (
    isinstance(viewer, dict)
    and viewer.get("status") == "running"
    and viewer.get("ready") is True
    and viewer.get("alive") is True
    and viewer.get("authorityFeedbackEnabled") is False
    and viewer.get("keyCallbackRegistered") is False
    and viewer.get("lastError") is None
)
if ready:
    raise SystemExit(0)
raise SystemExit(1)
'; then
      printf '%-28s ready\n' 'MuJoCo native viewer'
      return 0
    fi
    sleep 0.25
  done
  printf 'MuJoCo native viewer did not become ready within 30s\n' >&2
  return 1
}

wait_for_qwen_vision() {
  local timeout_seconds="${1:-120}"
  local deadline=$((SECONDS + timeout_seconds))
  local health=""
  while (( SECONDS < deadline )); do
    health="$(curl -fsS --max-time 3 http://127.0.0.1:8092/healthz 2>/dev/null || true)"
    if [[ "${health}" == *'"releaseState":"pure_vlm_product_profile"'* \
      && "${health}" == *'"provider":"qwen","model":"qwen3-vl-plus","available":true'* \
      && "${health}" == *'"provider":"sam2","available":true'* ]]; then
      printf '%-28s ready\n' 'H-IDE Vision Qwen + SAM2'
      return 0
    fi
    sleep 1
  done
  printf 'H-IDE Vision did not expose Qwen + SAM 2 readiness within %ss\n' "${timeout_seconds}" >&2
  return 1
}

wait_for_sonic_motion_ready() {
  local timeout_seconds="${1:-120}"
  local deadline=$((SECONDS + timeout_seconds))
  local telemetry=""
  while (( SECONDS < deadline )); do
    telemetry="$(curl -fsS --max-time 3 http://127.0.0.1:8792/telemetry 2>/dev/null || true)"
    if [[ -n "${telemetry}" ]] && printf '%s' "${telemetry}" \
      | "${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" -c '
import json
import re
import sys

try:
    value = json.load(sys.stdin)
except (json.JSONDecodeError, TypeError):
    raise SystemExit(1)

base_ready = (
    value.get("status") == "running"
    and value.get("motion_ready") is True
    and value.get("control_clock_status") == "simulation_time_50hz_motion_ready"
)
last_error = value.get("last_error")
structured_motion_fault = bool(
    last_error
    and re.match(r"^[a-z][a-z0-9_]*/[A-Z][A-Z0-9_]*:", str(last_error).strip())
)
settled_safe_hold = (
    value.get("nav2_state") == "safety_stop"
    and value.get("sonic_mode") == "IDLE"
    and abs(float(value.get("sonic_speed_mps", 1.0))) == 0.0
    and abs(float(value.get("sonic_policy_speed_mps", 1.0))) == 0.0
    and value.get("static_hold_active") is True
    and value.get("motion_handoff_active") is False
    and abs(float(value.get("motion_handoff_ratio", 1.0))) == 0.0
    and value.get("navigation_task_active") is False
    and all(abs(float(value.get(name, 1.0))) == 0.0 for name in (
        "last_cmd_vx", "last_cmd_vy", "last_cmd_wz"
    ))
    and value.get("sonic_planner_reference_ready") is False
    and value.get("sonic_planner_required_pending") is False
    and value.get("sonic_reference_authority_state") == "REVOKED"
    and value.get("sonic_reference_actuator_authority") is False
    and value.get("sonic_reference_authority_dds_target_sequence") is None
    and value.get("sonic_execution_feedback_pending") is False
    and value.get("plant_support_mode") == "safety_stop"
    and value.get("plant_support_requested_mode") == "safety_stop"
    and value.get("plant_support_confirmed_mode") == "safety_stop"
    and value.get("plant_support_stop_pending") is False
    and value.get("plant_support_safety_latched") is True
    and isinstance(value.get("plant_support_stop_generation"), int)
    and not isinstance(value.get("plant_support_stop_generation"), bool)
    and value.get("plant_support_stop_generation") >= 0
    and value.get("fault_trace_path") in {None, ""}
    and value.get("fault_trace_sample_count") in {None, 0}
)
support_safety_clue = (
    value.get("nav2_state") == "safety_stop"
    or any(value.get(name) == "safety_stop" for name in (
        "plant_support_mode",
        "plant_support_requested_mode",
        "plant_support_confirmed_mode",
    ))
    or value.get("plant_support_stop_pending") is True
    or value.get("plant_support_safety_latched") is True
    or value.get("plant_support_stop_generation") is not None
)
settled_idle = (
    value.get("nav2_state") == "idle"
    and value.get("sonic_mode") == "IDLE"
    and abs(float(value.get("sonic_speed_mps", 1.0))) == 0.0
    and abs(float(value.get("sonic_policy_speed_mps", 1.0))) == 0.0
    and value.get("static_hold_active") is True
    and value.get("motion_handoff_active") is False
    and abs(float(value.get("motion_handoff_ratio", 1.0))) == 0.0
    and value.get("navigation_task_active") is False
    and all(abs(float(value.get(name, 1.0))) == 0.0 for name in (
        "last_cmd_vx", "last_cmd_vy", "last_cmd_wz"
    ))
    and value.get("sonic_planner_reference_ready") is False
    and value.get("sonic_planner_required_pending") is False
    and value.get("sonic_reference_authority_state") == "REVOKED"
    and value.get("sonic_reference_actuator_authority") is False
    and value.get("sonic_reference_authority_dds_target_sequence") is None
    and value.get("sonic_execution_feedback_pending") is False
    and value.get("plant_support_mode") == "idle"
    and value.get("plant_support_requested_mode") == "idle"
    and value.get("plant_support_confirmed_mode") == "idle"
    and value.get("plant_support_stop_pending") is False
    and value.get("plant_support_safety_latched") is False
    and value.get("plant_support_stop_generation") is None
    and value.get("fault_trace_path") in {None, ""}
    and value.get("fault_trace_sample_count") in {None, 0}
)
ready = base_ready and not structured_motion_fault and (
    settled_safe_hold
    if support_safety_clue
    else settled_idle and last_error in {None, ""}
)
raise SystemExit(0 if ready else 1)
'; then
      printf '%-28s ready\n' 'SONIC 50 Hz motion gate'
      return 0
    fi
    sleep 1
  done
  printf 'SONIC did not qualify its simulation-time motion gate within %ss\n' "${timeout_seconds}" >&2
  if [[ -n "${telemetry}" ]]; then
    printf '%s\n' "${telemetry}" >&2
  fi
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

if [[ "${HIDE_DEMO_REBUILD}" == "true" ]]; then
  "${compose[@]}" up -d --build
else
  "${compose[@]}" up -d
fi

# A pinned host GPU service owns the only simulation RGB-D ingress and
# Qwen/SAM 2 path. The default Compose profile contains no competing Vision
# inference container; containers reach this product service through
# host.docker.internal:8092.
mkdir -p "${HIDE_DEMO_VISION_ROOT}/var/evidence" "${HIDE_DEMO_VISION_ROOT}/var/models"
start_process \
  h-ide-vision-qwen \
  'h-ide-vision serve' \
  "${HIDE_DEMO_VISION_ROOT}" \
  'env_get() { sed -n "s/^$1=//p" "$HIDE_DEMO_H_IDE_ROOT/.env" | tail -n 1; }; agent_token="$(env_get HIDE_VISION_AGENT_TOKEN)"; control_token="$(env_get HIDE_VISION_CONTROL_TOKEN)"; export DASHSCOPE_API_KEY="$(env_get DASHSCOPE_API_KEY)" HIDE_VISION_AGENT_TOKEN="${agent_token:-dev-agent-token}" HIDE_VISION_CONTROL_TOKEN="${control_token:-dev-control-token}" HIDE_VISION_SAM2_CHECKPOINT="$(env_get HIDE_VISION_SAM2_CHECKPOINT)"; test -n "$DASHSCOPE_API_KEY"; test -f "$HIDE_VISION_SAM2_CHECKPOINT"; export HIDE_VISION_HOST=0.0.0.0 HIDE_VISION_HTTP_PORT=8092 HIDE_VISION_GRPC_HOST=127.0.0.1 HIDE_VISION_GRPC_PORT=50061 HIDE_VISION_DATABASE_PATH="$HIDE_DEMO_VISION_ROOT/var/h-ide-vision-qwen.sqlite3" HIDE_VISION_EVIDENCE_DIR="$HIDE_DEMO_VISION_ROOT/var/evidence" HIDE_VISION_VLM_PROVIDER=qwen HIDE_VISION_SEGMENTATION_PROVIDER=sam2 HIDE_VISION_SAM2_DEVICE=cuda HIDE_VISION_QWEN_MODEL=qwen3-vl-plus HIDE_VISION_ALLOW_INSECURE_LOOPBACK=true; exec "$HIDE_DEMO_VISION_ROOT/.venv/bin/h-ide-vision" serve'

wait_for_qwen_vision 120

nav2_image="h-ide-nav2-sim:jazzy"
# Office V2's MuJoCo collision model and Nav2 occupancy map are generated from
# one scene.yaml. Refuse to build or reuse a navigation image unless every
# committed runtime artifact exactly matches a clean deterministic build.
"${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" \
  "${HIDE_DEMO_SIM_ROOT}/tools/build_scene.py" --check
office_v2_runtime_manifest="${HIDE_DEMO_SIM_ROOT}/scenes/office_v2/generated/runtime_manifest.json"
office_v2_source_sha256="$(
  "${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" -c \
    'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["scene_source_sha256"])' \
    "${office_v2_runtime_manifest}"
)"
if [[ ! "${office_v2_source_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'Office V2 runtime manifest has an invalid scene source SHA-256\n' >&2
  exit 1
fi
# Re-run the build on every launch. Docker's layer cache makes an unchanged
# context cheap, while a tag-exists check would silently keep an old controller
# after a fail-closed Nav2 patch or parameter update.
docker build \
  -t "${nav2_image}" \
  --build-arg "OFFICE_V2_SOURCE_SHA256=${office_v2_source_sha256}" \
  -f "${HIDE_DEMO_SIM_ROOT}/docker/nav2/Dockerfile" \
  "${HIDE_DEMO_SIM_ROOT}"
nav2_image_id="$(docker image inspect "${nav2_image}" --format '{{.Id}}')"
test -n "${nav2_image_id:?}"
nav2_image_scene_sha256="$(
  docker image inspect "${nav2_image}" \
    --format '{{ index .Config.Labels "org.h-ide.scene.source-sha256" }}'
)"
if [[ "${nav2_image_scene_sha256}" != "${office_v2_source_sha256}" ]]; then
  printf 'Nav2 image Office V2 source hash mismatch: expected %s, got %s\n' \
    "${office_v2_source_sha256}" "${nav2_image_scene_sha256}" >&2
  exit 1
fi
nav2_container_recreate=false
if docker container inspect h-ide-nav2-sim >/dev/null 2>&1; then
  nav2_container_image_id="$(docker container inspect h-ide-nav2-sim --format '{{.Image}}')"
  test -n "${nav2_container_image_id:?}"
  if [[ "${nav2_container_image_id}" != "${nav2_image_id}" ]]; then
    nav2_container_recreate=true
  fi
fi
# Nav2's lifecycle manager only makes one autostart attempt.  Keep it stopped
# until the simulation publishes map -> base_link; otherwise a cold start can
# permanently leave planner/controller/BT navigator inactive.
nav2_needs_start=true
if docker container inspect h-ide-nav2-sim >/dev/null 2>&1 \
  && [[ "$(docker inspect h-ide-nav2-sim --format '{{.State.Running}}')" == "true" ]]; then
  if [[ "${nav2_container_recreate}" == "true" ]]; then
    docker stop --timeout 10 h-ide-nav2-sim >/dev/null
  elif nav2_lifecycle_active; then
    nav2_needs_start=false
    printf '%-28s already active\n' nav2
  else
    docker stop --timeout 10 h-ide-nav2-sim >/dev/null
  fi
fi
if [[ "${nav2_container_recreate}" == "true" ]]; then
  if [[ "$(docker container inspect h-ide-nav2-sim --format '{{.State.Running}}')" == "true" ]]; then
    printf 'refusing to replace the Nav2 container while it is still running\n' >&2
    exit 1
  fi
  # The container is a stateless process wrapper: maps/parameters live in the
  # image and no user data or volumes are attached.
  docker container rm h-ide-nav2-sim >/dev/null
fi

start_process \
  mujoco-plant \
  '-m hide_mujoco_sonic(\.plant)?( plant)? --config' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'exec "$HIDE_DEMO_SIM_ROOT/.venv/bin/python" -u -m hide_mujoco_sonic plant --config "$HIDE_DEMO_CONFIG"'

wait_for_http 'MuJoCo plant + RGB-D' 'http://127.0.0.1:8794/healthz' 60
wait_for_native_viewer

# Native viewer/X11 initialization and SONIC's ONNX model load are both heavy
# cold-start operations. Complete the passive viewer gate first so they cannot
# combine into a false Plant control-clock discontinuity. This does not relax
# the runtime gap threshold; the Plant remains fail-closed throughout.
start_process \
  sonic-driver \
  '-m hide_mujoco_sonic sonic --config' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'exec "$HIDE_DEMO_SIM_ROOT/.venv/bin/python" -u -m hide_mujoco_sonic sonic --config "$HIDE_DEMO_CONFIG"'

# A listening telemetry port only proves that the Python process exists.  Keep
# Nav2 and Robot Agent offline until the full controller path has warmed, the
# IDLE consumer has sustained the authoritative 50 Hz simulation cadence, and
# the driver explicitly opens its motion gate.
wait_for_sonic_motion_ready 120

# Start ROS ingress only after the actuator-side controller is ready. This
# avoids creating an already-armed IDLE lifecycle whose observation stream can
# expire while the cold SONIC process is still loading its models. Runtime
# same-generation recovery remains implemented in GaitAdapter for real bridge
# reconnects; launch ordering is only race reduction, not the safety mechanism.
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

wait_for_http \
  'Robot Agent loopback diagnostics' \
  'http://127.0.0.1:8796/healthz' \
  30

start_process \
  robot-agent-diagnostics-proxy \
  '-m hide_mujoco_sonic.robot_agent_diagnostics_proxy' \
  "${HIDE_DEMO_SIM_ROOT}" \
  'exec "$HIDE_DEMO_SIM_ROOT/.venv/bin/python" -u -m hide_mujoco_sonic.robot_agent_diagnostics_proxy --bind-host "$HIDE_DOCKER_OBSERVATION_BIND_HOST"'

wait_for_http \
  'Docker Robot Agent diagnostics' \
  "http://${HIDE_DOCKER_OBSERVATION_BIND_HOST}:8797/healthz" \
  30

# A listening Robot Agent endpoint is not authority evidence.  Before reporting
# a successful workstation launch, require two consecutive, cross-identified
# read-only snapshots of the exact Plant/SONIC/Robot-Agent SAFE_HOLD boundary.
# The gate has fixed loopback GET endpoints and a hard 30-second total deadline.
"${HIDE_DEMO_SIM_ROOT}/.venv/bin/python" \
  "${HIDE_DEMO_SIM_ROOT}/tools/startup_safe_hold_gate.py" \
  --deadline-seconds 30 \
  --interval-seconds 0.5 \
  --http-timeout-seconds 1

printf '\nRuntime logs: %s\n' "${HIDE_DEMO_RUN_ROOT}"
if [[ "${HIDE_MUJOCO_ONSCREEN_VIEWER}" == "1" ]]; then
  printf 'MuJoCo native third-person viewer is enabled on DISPLAY=%s.\n' "${DISPLAY}"
else
  printf 'Optional MuJoCo viewer is disabled. Open http://127.0.0.1:8080/developer/simulation for RGB-D and mission evidence.\n'
fi
"${HIDE_DEMO_SIM_ROOT}/scripts/workstation-demo-status.sh"
