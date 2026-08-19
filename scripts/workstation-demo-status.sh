#!/usr/bin/env bash
set -u

HIDE_STATUS_H_IDE_ROOT="/home/phi/learn-projects/h-ide"
HIDE_STATUS_ENV="${HIDE_STATUS_H_IDE_ROOT}/.env"

env_value() {
  local key="$1"
  local fallback="$2"
  local line=""
  if [[ -f "${HIDE_STATUS_ENV}" ]]; then
    line="$(sed -n "s/^${key}=//p" "${HIDE_STATUS_ENV}" | tail -n 1)"
    line="${line%$'\r'}"
  fi
  printf '%s' "${line:-$fallback}"
}

gateway_token="$(env_value SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN dev-sim-gateway-token)"
control_token="$(env_value CONTROL_API_TOKEN dev-token)"

http_status() {
  local label="$1"
  local url="$2"
  shift 2
  local code=""
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$@" "${url}" 2>/dev/null || true)"
  if [[ "${code}" == "200" ]]; then
    printf '%-28s ok\n' "${label}"
  else
    printf '%-28s unavailable (HTTP %s)\n' "${label}" "${code:-none}"
  fi
}

http_status 'H-IDE web/control API' 'http://127.0.0.1:8080/api/v1/robots' -H "Authorization: Bearer ${control_token}"
vision_health="$(curl -fsS --max-time 3 http://127.0.0.1:8092/healthz 2>/dev/null || true)"
if [[ "${vision_health}" == *'"releaseState":"pure_vlm_product_profile"'* \
  && "${vision_health}" == *'"provider":"qwen","model":"qwen3-vl-plus","available":true'* \
  && "${vision_health}" == *'"provider":"sam2","available":true'* ]]; then
  printf '%-28s ok (qwen3-vl-plus + sam2)\n' 'H-IDE Vision'
else
  printf '%-28s unavailable or wrong provider\n' 'H-IDE Vision'
fi
http_status 'MuJoCo plant + RGB-D' 'http://127.0.0.1:8794/healthz'
http_status 'SONIC telemetry' 'http://127.0.0.1:8792/healthz'
http_status 'g1-sim Robot Agent gateway' 'http://127.0.0.1:8094/internal/v1/discover?robotId=g1-sim' -H "Authorization: Bearer ${gateway_token}"

nav2_state=""
if docker inspect h-ide-nav2-sim >/dev/null 2>&1 \
  && [[ "$(docker inspect h-ide-nav2-sim --format '{{.State.Running}}')" == "true" ]]; then
  nav2_state="$(timeout 4 bash -lc '
    source /opt/ros/jazzy/setup.bash
    export ROS_DOMAIN_ID=232 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI='"'"'<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'"'"'
    ros2 lifecycle get /bt_navigator
  ' 2>/dev/null || true)"
fi
if [[ "${nav2_state}" == *'active [3]'* ]]; then
  printf '%-28s ok (active)\n' 'Nav2 lifecycle'
elif [[ -n "${nav2_state}" ]]; then
  printf '%-28s unavailable (%s)\n' 'Nav2 lifecycle' "${nav2_state//$'\n'/ }"
else
  printf '%-28s unavailable\n' 'Nav2 lifecycle'
fi

printf '%-28s %s\n' 'Web UI' 'http://127.0.0.1:8080/'
printf '%-28s %s\n' 'Simulation observatory' 'http://127.0.0.1:8080/simulation'
