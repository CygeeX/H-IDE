#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ENV_FILE="${SKILLSTORE_ENV_FILE:-${ROOT}/.env}"
RELEASE_ID=skillstore-office-v2-e2e-20260901-r1
LEVEL=L0
EXECUTE_MOTION=false
MISSION_ID=""
OPERATOR_APPROVAL=""
RESULT_DIR="${SKILLSTORE_ACCEPTANCE_RESULT_DIR:-${ROOT}/var/acceptance/${RELEASE_ID}}"
CONTROL_API_BASE_URL="${SKILLSTORE_CONTROL_API_BASE_URL:-http://127.0.0.1:8080}"
COOKIE_JAR="${SKILLSTORE_L3_COOKIE_JAR:-}"

usage() {
  cat <<'EOF'
usage: integration/acceptance.sh [options]

  --level L0|L1|L2|L3
  --execute-motion                 required for L3; invalid for L0-L2
  --mission-id ID                  existing operator-reviewed PWA Mission
  --operator-approval TEXT         must equal AUTHORIZE_L3:<mission-id>
  --cookie-jar PATH                mode-0600 PWA session cookie jar
  --result-dir PATH                ignored runtime evidence destination

L0 checks repository hygiene, release manifests, shell syntax and selectors.
L1 adds deterministic component contract/unit tests and Compose/scene checks.
L2 adds GPU/ROS/model readiness and live service health. It sends no motion.
L3 records one real PWA Mission after an explicit flag, exact approval phrase
   and interactive operator confirmation. This script never invents a route,
   approves a draft, starts a Mission or sends a motion command automatically.

For L3, create and approve an inspection Mission in the PWA, execute it to a
terminal state, then provide a mode-0600 curl cookie jar for the same PWA user.
Evidence is written mode 0600 beneath var/ by default and must not be committed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --level)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      LEVEL="$2"
      shift 2
      ;;
    --execute-motion)
      EXECUTE_MOTION=true
      shift
      ;;
    --mission-id)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      MISSION_ID="$2"
      shift 2
      ;;
    --operator-approval)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      OPERATOR_APPROVAL="$2"
      shift 2
      ;;
    --cookie-jar)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      COOKIE_JAR="$2"
      shift 2
      ;;
    --result-dir)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      RESULT_DIR="$2"
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
  L0|L1|L2|L3) ;;
  *) printf 'unsupported level: %s\n' "${LEVEL}" >&2; exit 2 ;;
esac

if [[ "${LEVEL}" != L3 && "${EXECUTE_MOTION}" == true ]]; then
  printf '%s\n' '--execute-motion is valid only with --level L3' >&2
  exit 2
fi
if [[ "${LEVEL}" != L3 && ( -n "${MISSION_ID}" || -n "${OPERATOR_APPROVAL}" || -n "${COOKIE_JAR}" ) ]]; then
  printf '%s\n' '--mission-id, --operator-approval and --cookie-jar are valid only for L3' >&2
  exit 2
fi

preflight_level="${LEVEL}"
[[ "${LEVEL}" == L3 ]] && preflight_level=L2
"${SCRIPT_DIR}/preflight.sh" --level "${preflight_level}"

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
    root / "manifests/release/skillstore-office-v2-e2e-20260901-r1.json",
    root / "manifests/release/l3-mission-result.schema.json",
    root / "manifests/sbom/skillstore-office-v2-e2e-20260901-r1.spdx.json",
    root / "manifests/ros/jazzy-user-prefix-amd64.json",
    root / "manifests/python/simulation-runtime.json",
    root / "manifests/python/vision-runtime.json",
    root / "components/h-ide-mujoco-sonic/config/sonic-models.lock.json",
    root / "manifests/vendor/g1-model.json",
    root / "manifests/vendor/groot-wbc.json",
    root / "manifests/vendor/lerobot.json",
    root / "manifests/vendor/sam2-checkpoint.json",
    root / "manifests/vendor/unitree-sdk2-python.json",
]:
    json.loads(path.read_text(encoding="utf-8"))
print("JSON manifests: OK")
PY

if [[ "${LEVEL}" == L0 ]]; then
  printf '%s\n' 'acceptance L0: OK (heuristic repository scan; not complete secret clearance)'
  exit 0
fi

run_pytest_if_ready() {
  local dir="$1"
  shift
  local python_bin="${dir}/.venv/bin/python"
  if [[ ! -x "${python_bin}" ]]; then
    printf 'missing %s; run the environment setup in docs/reproduction/README.md\n' "${python_bin}" >&2
    return 1
  fi
  # ROS installs optional pytest entry points into the system interpreter. They
  # can import packages absent from a component venv before collection starts;
  # contract tests must only load plugins declared by that component.
  (cd "${dir}" && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${python_bin}" -m pytest -q "$@")
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
  tests/test_model_lock.py tests/test_nav2_config.py tests/test_reset_policy.py

if command -v npm >/dev/null 2>&1 \
  && [[ -d "${ROOT}/components/h-ide-agent/node_modules" ]] \
  && [[ -d "${ROOT}/components/h-ide/apps/control-pwa/node_modules" ]]; then
  (cd "${ROOT}/components/h-ide-agent" && npm test)
  (cd "${ROOT}/components/h-ide-agent" && npm run typecheck)
  (cd "${ROOT}/components/h-ide-agent" && npm run build)
  (cd "${ROOT}/components/h-ide/apps/control-pwa" && npm test -- --run)
  (cd "${ROOT}/components/h-ide/apps/control-pwa" && npm run build)
else
  printf '%s\n' \
    'missing npm or Agent/PWA node_modules; run integration/setup-environments.sh before L1 acceptance' >&2
  exit 1
fi

if [[ "${LEVEL}" == L1 ]]; then
  printf '%s\n' 'acceptance L1: OK'
  exit 0
fi

# This suite loads the operator-fetched G1 display mesh. L2 preflight has
# already verified that the licensed external asset and its manifest exist.
run_pytest_if_ready "${ROOT}/components/h-ide-mujoco-sonic" \
  tests/test_office_v2_scene.py

command -v curl >/dev/null 2>&1 || { printf '%s\n' 'missing command: curl' >&2; exit 1; }
printf '%s\n' 'Acceptance: live service health (read-only; no motion command is sent)'
[[ -f "${ENV_FILE}" && ! -L "${ENV_FILE}" ]] || {
  printf '%s\n' 'L2 requires the regular Skillstore environment file' >&2
  exit 1
}

env_value() {
  local key="$1"
  sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1
}

control_token="$(env_value CONTROL_API_TOKEN)"
gateway_token="$(env_value SIMULATION_ROBOT_AGENT_GATEWAY_TOKEN)"
scene_internal_token="$(env_value SCENE_API_INTERNAL_TOKEN)"
[[ -n "${control_token}" && -n "${gateway_token}" && -n "${scene_internal_token}" ]] || {
  printf '%s\n' 'L2 requires Control, Robot Agent gateway and Scene internal credentials' >&2
  exit 1
}

compose_services="$({
  docker compose \
    --project-name "${SKILLSTORE_COMPOSE_PROJECT:-skillstore-e2e}" \
    --env-file "${ENV_FILE}" \
    -f "${ROOT}/components/h-ide/infra/docker-compose.yml" \
    -f "${ROOT}/components/h-ide/infra/docker-compose.simulation.yml" \
    ps --services --filter status=running
} | sort)"
expected_services="$({
  docker compose \
    --project-name "${SKILLSTORE_COMPOSE_PROJECT:-skillstore-e2e}" \
    --env-file "${ENV_FILE}" \
    -f "${ROOT}/components/h-ide/infra/docker-compose.yml" \
    -f "${ROOT}/components/h-ide/infra/docker-compose.simulation.yml" \
    config --services
} | sort)"
[[ "${compose_services}" == "${expected_services}" ]] || {
  printf '%s\n' 'L2 Compose service set is not completely running' >&2
  exit 1
}

plant_health="$(curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8794/healthz)"
sonic_health="$(curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8792/telemetry)"
vision_health="$(curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8092/healthz)"
robot_diagnostics="$(curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8796/v1/diagnostics)"
gateway_discovery="$(curl --noproxy '*' -fsS --max-time 5 \
  -H "Authorization: Bearer ${gateway_token}" \
  'http://127.0.0.1:8094/internal/v1/discover?robotId=g1-sim')"
control_robots="$(curl --noproxy '*' -fsS --max-time 5 \
  -H "Authorization: Bearer ${control_token}" \
  http://127.0.0.1:8080/api/v1/robots)"
scene_context="$(curl --noproxy '*' -fsS --max-time 5 \
  -H "Authorization: Bearer ${scene_internal_token}" \
  http://127.0.0.1:8090/internal/v1/robots/g1-sim/scene-context)"

python3 - "${plant_health}" "${sonic_health}" "${vision_health}" \
  "${robot_diagnostics}" "${gateway_discovery}" "${control_robots}" \
  "${scene_context}" <<'PY'
import json
import sys

plant, sonic, vision, diagnostics, discovery, robots, scene = map(json.loads, sys.argv[1:])
if plant.get("status") != "running":
    raise SystemExit("MuJoCo Plant is not running")
if not all(isinstance(plant.get(key), str) and plant[key] for key in (
    "producerBootId", "sessionId", "localizationEpochId"
)):
    raise SystemExit("MuJoCo Plant identity is incomplete")
support = plant.get("support") if isinstance(plant.get("support"), dict) else {}
if support.get("fault") is not None or support.get("motionFault") is not None:
    raise SystemExit("MuJoCo Plant retains a physics or motion fault")
if sonic.get("status") != "running" or sonic.get("gesture_state") == "running":
    raise SystemExit("SONIC is not in a read-only-ready state")
if sonic.get("navigation_task_active") is not False:
    raise SystemExit("SONIC still owns an active navigation task")
capabilities = vision.get("capabilities") if isinstance(vision.get("capabilities"), dict) else {}
vlm = capabilities.get("vlm") if isinstance(capabilities.get("vlm"), dict) else {}
segmenter = capabilities.get("segmentation") if isinstance(capabilities.get("segmentation"), dict) else {}
if (
    vision.get("status") != "ok"
    or vision.get("releaseState") != "pure_vlm_product_profile"
    or vlm.get("provider") != "qwen"
    or vlm.get("model") != "qwen3-vl-plus"
    or vlm.get("available") is not True
    or segmenter.get("provider") != "sam2"
    or segmenter.get("available") is not True
):
    raise SystemExit("Vision is not the frozen Qwen + SAM2 product profile")
if diagnostics.get("ready") is not True or not diagnostics.get("robotBootId"):
    raise SystemExit("Robot Agent diagnostics are not ready")
if diagnostics.get("commandReadiness", {}).get("ready") is not True:
    raise SystemExit("Robot Agent command boundary is not ready")
result = discovery.get("result") if isinstance(discovery.get("result"), dict) else {}
observation = result.get("observation") if isinstance(result.get("observation"), dict) else {}
if (
    discovery.get("status") != "available"
    or result.get("available") is not True
    or result.get("robotId") != "g1-sim"
    or result.get("robotBootId") != diagnostics.get("robotBootId")
    or observation.get("status") != "USABLE"
    or not observation.get("sessionId")
    or not observation.get("localizationEpochId")
    or observation.get("localizationEpochId") != plant.get("localizationEpochId")
):
    raise SystemExit("Robot Agent gateway discovery/observation identity is not ready")
robot = next((item for item in robots if item.get("robotId") == "g1-sim"), None)
details = robot.get("details") if isinstance(robot, dict) and isinstance(robot.get("details"), dict) else {}
manifest = details.get("capabilityManifest") if isinstance(details.get("capabilityManifest"), dict) else {}
skill_ids = {
    item.get("skillId")
    for item in manifest.get("capabilities", [])
    if isinstance(item, dict)
}
if (
    not isinstance(robot, dict)
    or robot.get("status") != "online"
    or robot.get("connected") is not True
    or details.get("robotBootId") != diagnostics.get("robotBootId")
    or details.get("observationSessionId") != observation.get("sessionId")
    or details.get("localizationEpochId") != observation.get("localizationEpochId")
    or not {"navigation.navigate_to_pose", "navigation.rotate"}.issubset(skill_ids)
):
    raise SystemExit("Control API robot projection does not match Robot Agent identity")
scene_ref = scene.get("sceneRef") if isinstance(scene.get("sceneRef"), dict) else {}
annotations = scene.get("annotations") if isinstance(scene.get("annotations"), list) else []
location_ids = {item.get("locationId") for item in annotations if isinstance(item, dict)}
required = {
    "corridor", "corridor_west", "corridor_center", "corridor_east",
    "reception", "reception_entry", "reception_main",
    "open_office", "open_office_entry", "open_office_west", "open_office_east",
    "meeting_room", "meeting_entry", "meeting_south", "meeting_north",
    "focus_office", "focus_entry", "focus_main", "focus_east_wall",
    "pantry", "pantry_entry", "pantry_west", "pantry_center",
    "storage", "storage_entry", "storage_main", "storage_deep",
}
if (
    not scene_ref.get("sceneId")
    or not isinstance(scene_ref.get("revision"), int)
    or location_ids != required
):
    raise SystemExit("Office V2 published Scene binding is incomplete")
print(
    "Live identities: Plant/SONIC, Robot Agent, observation session, Control API "
    "and Office V2 Scene binding agree"
)
PY

# Control API serves the built PWA at root and currently has no generic
# /readyz route. Compose already gates its Postgres/Redis dependencies; a 200
# root response proves the same process and gateway path used by operators.
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8080/ >/dev/null

ros_runtime() (
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  # shellcheck disable=SC1091
  source "${ROOT}/components/robotagent/tools/activate-local-nav2-jazzy.bash"
  set -u
  export ROS_DOMAIN_ID=232
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
  exec "$@"
)
for node in map_server planner_server controller_server behavior_server bt_navigator waypoint_follower; do
  lifecycle="$(ros_runtime timeout 8 ros2 lifecycle get "/${node}")"
  [[ "${lifecycle}" == *'active [3]'* ]] || {
    printf 'Nav2 lifecycle node is not active: %s\n' "${node}" >&2
    exit 1
  }
done
strict_health="$(ros_runtime timeout 10 ros2 service call \
  /h_ide/strict_navigation/health std_srvs/srv/Trigger '{}')"
python3 - "${strict_health}" <<'PY'
import ast
import json
import re
import sys

match = re.search(r"success=(True|False), message=(?P<message>.+)\)\s*$", sys.argv[1], re.DOTALL)
if match is None or match.group(1) != "True":
    raise SystemExit("strict navigation health service did not succeed")
payload = json.loads(ast.literal_eval(match.group("message")))
bridge = payload.get("bridge") if isinstance(payload.get("bridge"), dict) else {}
if (
    payload.get("healthy") is not True
    or payload.get("active") is not False
    or bridge.get("healthy") is not True
    or bridge.get("fault") is not None
    or bridge.get("sonicStatus") != "ready"
):
    raise SystemExit("strict navigation adapter/ROS bridge is not idle and healthy")
print(
    "Strict adapter: healthy, idle, "
    f"fullStraightDrive={payload.get('fullStraightDrive')}, "
    f"continuousDriveHandoff={payload.get('continuousDriveHandoff')}, "
    f"runtimeDriveChunkM={payload.get('runtimeDriveChunkM')}"
)
PY

if [[ "${LEVEL}" == L2 ]]; then
  printf '%s\n' 'acceptance L2: OK (runtime health only; no Mission or motion was executed)'
  exit 0
fi

# L3 is intentionally a recorder for an operator-reviewed PWA Mission. It does
# not call a motion runner, create/approve/start a Mission or issue a route.
[[ "${EXECUTE_MOTION}" == true ]] || {
  printf '%s\n' 'L3 requires the explicit --execute-motion acknowledgement' >&2
  exit 2
}
[[ "${MISSION_ID}" =~ ^[A-Za-z0-9._:-]{1,256}$ ]] || {
  printf '%s\n' 'L3 requires a bounded --mission-id containing only A-Z, a-z, 0-9, dot, underscore, colon or dash' >&2
  exit 2
}
expected_approval="AUTHORIZE_L3:${MISSION_ID}"
[[ "${OPERATOR_APPROVAL}" == "${expected_approval}" ]] || {
  printf 'L3 --operator-approval must exactly equal %s\n' "${expected_approval}" >&2
  exit 2
}
[[ -t 0 ]] || {
  printf '%s\n' 'L3 requires an interactive operator TTY and is forbidden in CI/non-interactive jobs' >&2
  exit 2
}
[[ "${CONTROL_API_BASE_URL}" =~ ^http://(127\.0\.0\.1|localhost):[0-9]{1,5}$ ]] || {
  printf '%s\n' 'L3 session cookies may only be sent to an explicit loopback host and numeric port' >&2
  exit 2
}

printf '%s\n' 'L3 sends no command. In the PWA, verify this exact Mission was reviewed, executed and reached a terminal state.'
printf 'Type the Mission ID (%s) to authorize local result capture: ' "${MISSION_ID}" >&2
IFS= read -r typed_mission_id
[[ "${typed_mission_id}" == "${MISSION_ID}" ]] || {
  printf '%s\n' 'operator confirmation did not match the Mission ID' >&2
  exit 2
}

if [[ -z "${COOKIE_JAR}" ]]; then
  printf '%s' 'PWA curl cookie-jar path: ' >&2
  IFS= read -r COOKIE_JAR
fi
[[ -f "${COOKIE_JAR}" && ! -L "${COOKIE_JAR}" ]] || {
  printf '%s\n' 'L3 requires a regular, non-symlink PWA cookie jar' >&2
  exit 2
}
cookie_mode="$(stat -c '%a' "${COOKIE_JAR}" 2>/dev/null || printf unknown)"
[[ "${cookie_mode}" == 600 ]] || {
  printf 'L3 cookie jar must have mode 0600 (got %s)\n' "${cookie_mode}" >&2
  exit 2
}

umask 077
mkdir -p -- "${RESULT_DIR}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
base="${RESULT_DIR}/${timestamp}-${MISSION_ID}"
raw_pending="${base}.mission.json.pending"
raw_result="${base}.mission.json"
summary_pending="${base}.acceptance.json.pending"
summary_result="${base}.acceptance.json"
cleanup_pending() {
  rm -f -- "${raw_pending}" "${summary_pending}"
}
trap cleanup_pending EXIT

curl --noproxy '*' -fsS --max-time 15 \
  --cookie "${COOKIE_JAR}" \
  "${CONTROL_API_BASE_URL}/api/v2/missions/${MISSION_ID}" \
  -o "${raw_pending}"
mv -- "${raw_pending}" "${raw_result}"

repository_commit="$(git -C "${ROOT}" rev-parse HEAD)"
mission_validation_status=0
python3 - \
  "${raw_result}" "${summary_pending}" "${MISSION_ID}" \
  "${repository_commit}" "${RELEASE_ID}" <<'PY' || mission_validation_status=$?
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

raw_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
expected_mission_id = sys.argv[3]
repository_commit = sys.argv[4]
release_id = sys.argv[5]
raw = raw_path.read_bytes()
mission = json.loads(raw)
mission_id = mission.get("missionId")
if mission_id != expected_mission_id:
    raise SystemExit("Control API returned a different Mission identity")
snapshot = mission.get("snapshot") if isinstance(mission.get("snapshot"), dict) else {}
mission_status = str(snapshot.get("status") or "unknown")
executions = mission.get("skillExecutions")
if not isinstance(executions, list):
    executions = []

def execution_identity(item):
    if not isinstance(item, dict):
        return "", ""
    request = item.get("request") if isinstance(item.get("request"), dict) else {}
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    skill_id = str(item.get("skillId") or request.get("skillId") or "")
    status = str(item.get("status") or result.get("status") or "")
    return skill_id, status

identities = [execution_identity(item) for item in executions]
navigation_succeeded = any(
    skill_id == "navigation.navigate_to_pose" and status == "succeeded"
    for skill_id, status in identities
)
inspection_succeeded = any(
    skill_id == "inspection.scan_station" and status == "succeeded"
    for skill_id, status in identities
)
expected_nodes = [
    ("office-sequence-01-nav-01-open_office_entry", "navigation.navigate_to_pose"),
    ("office-sequence-01-nav-02-open_office_west", "navigation.navigate_to_pose"),
    ("office-sequence-01-verify-open_office", "scene.verify_inside_region"),
    ("office-sequence-01-scan-open_office_west", "inspection.scan_station"),
    ("office-sequence-02-nav-01-meeting_entry", "navigation.navigate_to_pose"),
    ("office-sequence-02-nav-02-meeting_south", "navigation.navigate_to_pose"),
    ("office-sequence-02-verify-meeting_room", "scene.verify_inside_region"),
    ("office-sequence-02-scan-meeting_south", "inspection.scan_station"),
    ("office-sequence-final-report", "mission.report"),
]
expected_contracts = {
    "office-sequence-01-nav-01-open_office_entry": {
        "dependsOn": [], "locationId": "open_office", "stationId": "open_office_entry", "anchorRole": "entry",
    },
    "office-sequence-01-nav-02-open_office_west": {
        "dependsOn": ["office-sequence-01-nav-01-open_office_entry"], "locationId": "open_office", "stationId": "open_office_west", "anchorRole": "observation_vantage",
    },
    "office-sequence-01-verify-open_office": {
        "dependsOn": ["office-sequence-01-nav-02-open_office_west"], "locationId": "open_office", "stationId": "open_office_west", "anchorRole": "observation_vantage",
    },
    "office-sequence-01-scan-open_office_west": {
        "dependsOn": ["office-sequence-01-verify-open_office"], "regionId": "open_office", "stationId": "open_office_west",
    },
    "office-sequence-02-nav-01-meeting_entry": {
        "dependsOn": ["office-sequence-01-scan-open_office_west"], "locationId": "meeting_room", "stationId": "meeting_entry", "anchorRole": "entry",
    },
    "office-sequence-02-nav-02-meeting_south": {
        "dependsOn": ["office-sequence-02-nav-01-meeting_entry"], "locationId": "meeting_room", "stationId": "meeting_south", "anchorRole": "observation_vantage",
    },
    "office-sequence-02-verify-meeting_room": {
        "dependsOn": ["office-sequence-02-nav-02-meeting_south"], "locationId": "meeting_room", "stationId": "meeting_south", "anchorRole": "observation_vantage",
    },
    "office-sequence-02-scan-meeting_south": {
        "dependsOn": ["office-sequence-02-verify-meeting_room"], "regionId": "meeting_room", "stationId": "meeting_south",
    },
    "office-sequence-final-report": {
        "dependsOn": ["office-sequence-02-scan-meeting_south"], "reportScope": "selected_locations",
    },
}
plan = mission.get("plan") if isinstance(mission.get("plan"), dict) else {}
plan_nodes = plan.get("nodes") if isinstance(plan.get("nodes"), list) else []
actual_nodes = [
    (str(item.get("nodeId") or ""), str(item.get("skillId") or ""))
    for item in plan_nodes
    if isinstance(item, dict)
]
node_by_id = {
    str(item.get("nodeId") or ""): item
    for item in plan_nodes
    if isinstance(item, dict)
}

def node_contract_matches(node_id, skill_id):
    node = node_by_id.get(node_id)
    expected = expected_contracts[node_id]
    if not isinstance(node, dict) or node.get("dependsOn") != expected["dependsOn"]:
        return False
    intent = node.get("intent") if isinstance(node.get("intent"), dict) else {}
    if skill_id in {"navigation.navigate_to_pose", "scene.verify_inside_region"}:
        place = intent.get("place") if isinstance(intent.get("place"), dict) else {}
        return all(place.get(key) == expected[key] for key in ("locationId", "stationId", "anchorRole"))
    if skill_id == "inspection.scan_station":
        return (
            intent.get("regionId") == expected["regionId"]
            and intent.get("stationId") == expected["stationId"]
            and intent.get("requiresInsideVerification") is True
        )
    if skill_id == "mission.report":
        return (
            intent.get("reportScope") == "selected_locations"
            and intent.get("regionIds") == ["open_office", "meeting_room"]
            and intent.get("stationIds") == ["open_office_west", "meeting_south"]
            and intent.get("locale") == "zh-CN"
        )
    return False

active_revision = mission.get("activeRevision") if isinstance(mission.get("activeRevision"), dict) else {}
revisions = mission.get("revisions") if isinstance(mission.get("revisions"), list) else []
active_revision_nodes = active_revision.get("nodes") if isinstance(active_revision.get("nodes"), list) else []
revision_stable = bool(
    len(revisions) == 1
    and active_revision.get("sequence") == 1
    and active_revision.get("revisionId") == mission.get("activeRevisionId")
    and active_revision_nodes == plan_nodes
)
office_sequence_validated = bool(
    plan.get("objective") == "今天巡检开放办公室和会议室"
    and actual_nodes == expected_nodes
    and all(node_contract_matches(node_id, skill_id) for node_id, skill_id in expected_nodes)
    and revision_stable
)
node_states = mission.get("nodeStates") if isinstance(mission.get("nodeStates"), list) else []
state_by_node = {
    str(item.get("nodeId") or ""): str(item.get("status") or "")
    for item in node_states
    if isinstance(item, dict)
}
all_required_nodes_succeeded = all(
    state_by_node.get(node_id) == "succeeded" for node_id, _skill_id in expected_nodes
)

def execution_for(node_id, skill_id):
    for item in reversed(executions):
        if not isinstance(item, dict) or item.get("nodeId") != node_id:
            continue
        request = item.get("request") if isinstance(item.get("request"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        if (
            request.get("skillId") == skill_id
            and item.get("status") == "succeeded"
            and result.get("status") == "succeeded"
        ):
            return item
    return None

scan_complete = True
scan_evidence_count = 0
seen_scan_evidence = set()
expected_scans = {
    "office-sequence-01-scan-open_office_west": ("open_office_west", "open_office"),
    "office-sequence-02-scan-meeting_south": ("meeting_south", "meeting_room"),
}
expected_heading_radians = {1: 0.0, 2: math.pi / 2.0, 3: math.pi, 4: -math.pi / 2.0}
for node_id, (station_id, region_id) in expected_scans.items():
    execution = execution_for(node_id, "inspection.scan_station")
    outer_result = execution.get("result") if isinstance(execution, dict) else {}
    result = outer_result.get("result") if isinstance(outer_result, dict) else {}
    inspection = result.get("inspectionResult") if isinstance(result, dict) else {}
    headings = inspection.get("headings") if isinstance(inspection, dict) else None
    heading_gaps = inspection.get("headingGaps", []) if isinstance(inspection, dict) else None
    headings_by_index = {
        item.get("headingIndex"): item
        for item in headings
        if isinstance(item, dict) and isinstance(item.get("headingIndex"), int) and not isinstance(item.get("headingIndex"), bool)
    } if isinstance(headings, list) else {}
    valid_headings = [
        item for index, expected_heading in expected_heading_radians.items()
        for item in [headings_by_index.get(index)]
        if isinstance(item, dict)
        and isinstance(item.get("headingRad"), (int, float))
        and not isinstance(item.get("headingRad"), bool)
        and math.isclose(float(item["headingRad"]), expected_heading, abs_tol=1e-9)
        and item.get("status") == "observed"
        and isinstance(item.get("visualResult"), dict)
        and item["visualResult"].get("status") == "succeeded"
        and item["visualResult"].get("missionId") == mission_id
        and item["visualResult"].get("revisionId") == mission.get("activeRevisionId")
        and str(item["visualResult"].get("nodeId") or "").startswith(node_id + "-view-")
        and isinstance(item["visualResult"].get("evidence"), list)
        and bool(item["visualResult"]["evidence"])
    ]
    evidence_hashes = [
        evidence.get("sha256")
        for item in valid_headings
        for evidence in item["visualResult"]["evidence"]
        if isinstance(evidence, dict)
    ]
    evidence_valid = bool(
        evidence_hashes
        and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in evidence_hashes)
        and len(evidence_hashes) == len(set(evidence_hashes))
        and not (set(evidence_hashes) & seen_scan_evidence)
    )
    seen_scan_evidence.update(value for value in evidence_hashes if isinstance(value, str))
    scan_evidence_count += len(evidence_hashes)
    if (
        inspection.get("stationId") != station_id
        or inspection.get("regionId") != region_id
        or not isinstance(headings, list)
        or len(headings) != 4
        or set(headings_by_index) != set(expected_heading_radians)
        or len(valid_headings) != 4
        or not isinstance(heading_gaps, list)
        or heading_gaps
        or not evidence_valid
    ):
        scan_complete = False

report_execution = execution_for("office-sequence-final-report", "mission.report")
outer_report = report_execution.get("result") if isinstance(report_execution, dict) else {}
report_result = outer_report.get("result") if isinstance(outer_report, dict) else {}
report = report_result.get("missionReport") if isinstance(report_result, dict) else {}
expected_regions = ["open_office", "meeting_room"]
expected_stations = ["open_office_west", "meeting_south"]
report_evidence = report.get("evidence") if isinstance(report, dict) else []
report_heading_gaps = report.get("headingGaps") if isinstance(report, dict) else None
report_missing_stations = report.get("missingStations") if isinstance(report, dict) else None
report_evidence_hashes = [
    item.get("sha256")
    for item in report_evidence
    if isinstance(item, dict)
] if isinstance(report_evidence, list) else []
report_evidence_valid = bool(
    report_evidence_hashes
    and len(report_evidence_hashes) == len(report_evidence)
    and len(report_evidence_hashes) == len(set(report_evidence_hashes))
    and all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in report_evidence_hashes
    )
    and set(report_evidence_hashes).issubset(seen_scan_evidence)
)
report_full_chain_accepted = bool(
    isinstance(report, dict)
    and report.get("reportScope") == "selected_locations"
    and report.get("status") == "completed"
    and report.get("fullChainAccepted") is True
    and report.get("regionIds") == expected_regions
    and report.get("stationIds") == expected_stations
    and isinstance(report_heading_gaps, list)
    and not report_heading_gaps
    and isinstance(report_missing_stations, list)
    and not report_missing_stations
    and isinstance(report_evidence, list)
    and bool(report_evidence)
    and report_evidence_valid
    and "开放办公区" in str(report.get("narrativeZhCN") or "")
    and "会议室" in str(report.get("narrativeZhCN") or "")
)
events = mission.get("events") if isinstance(mission.get("events"), list) else []
world_records = (
    mission.get("worldRecords") if isinstance(mission.get("worldRecords"), list) else []
)
accepted = (
    mission_status == "completed"
    and navigation_succeeded
    and inspection_succeeded
    and bool(world_records)
    and office_sequence_validated
    and all_required_nodes_succeeded
    and scan_complete
    and report_full_chain_accepted
)
summary = {
    "schemaVersion": "skillstore.l3-mission-result/1",
    "releaseId": release_id,
    "repositoryCommit": repository_commit,
    "recordedAt": datetime.now(timezone.utc).isoformat(),
    "missionId": mission_id,
    "operatorAuthorized": True,
    "missionStatus": mission_status,
    "navigationSucceeded": navigation_succeeded,
    "inspectionSucceeded": inspection_succeeded,
    "officeSequenceValidated": office_sequence_validated,
    "allRequiredNodesSucceeded": all_required_nodes_succeeded,
    "allRequiredHeadingsObserved": scan_complete,
    "reportFullChainAccepted": report_full_chain_accepted,
    "regionIds": expected_regions if report_full_chain_accepted else [],
    "stationIds": expected_stations if report_full_chain_accepted else [],
    "headingGapCount": len(report_heading_gaps) if isinstance(report_heading_gaps, list) else 0,
    "reportEvidenceCount": len(report_evidence) if isinstance(report_evidence, list) else 0,
    "scanEvidenceCount": scan_evidence_count,
    "resultAccepted": accepted,
    "eventCount": len(events),
    "worldRecordCount": len(world_records),
    "skillExecutionCount": len(executions),
    "rawMissionSha256": hashlib.sha256(raw).hexdigest(),
    "rawMissionPath": str(raw_path),
}
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
if not accepted:
    raise SystemExit(
        "L3 rejected: expected the exact reviewed two-room DAG, every required "
        "node and four-heading scan to succeed, a gap-free full-chain selected-"
        "locations report, and at least one durable world record"
    )
PY
mv -- "${summary_pending}" "${summary_result}"
schema_validation_status=0
"${ROOT}/components/h-ide/.venv/bin/python" - \
  "${ROOT}/manifests/release/l3-mission-result.schema.json" \
  "${summary_result}" <<'PY' || schema_validation_status=$?
import json
from pathlib import Path
import sys

from jsonschema import Draft202012Validator

schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda item: list(item.path))
if errors:
    detail = "; ".join(error.message for error in errors[:5])
    raise SystemExit(f"L3 structured result does not match its release schema: {detail}")
PY
trap - EXIT
if (( mission_validation_status != 0 || schema_validation_status != 0 )); then
  printf 'acceptance L3: FAILED; structured result: %s\n' "${summary_result}" >&2
  exit 1
fi
printf 'acceptance L3: OK; structured result: %s\n' "${summary_result}"
