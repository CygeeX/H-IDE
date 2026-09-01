#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
RELEASE_ID=skillstore-office-v2-e2e-20260901-r1
LEVEL=L0
EXECUTE_MOTION=false
MISSION_ID=""
OPERATOR_APPROVAL=""
RESULT_DIR="${SKILLSTORE_ACCEPTANCE_RESULT_DIR:-${ROOT}/var/acceptance/${RELEASE_ID}}"
CONTROL_API_BASE_URL="${SKILLSTORE_CONTROL_API_BASE_URL:-http://127.0.0.1:8080}"

usage() {
  cat <<'EOF'
usage: integration/acceptance.sh [options]

  --level L0|L1|L2|L3
  --execute-motion                 required for L3; invalid for L0-L2
  --mission-id ID                  existing operator-reviewed PWA Mission
  --operator-approval TEXT         must equal AUTHORIZE_L3:<mission-id>
  --result-dir PATH                ignored runtime evidence destination

L0 checks repository hygiene, release manifests, shell syntax and selectors.
L1 adds deterministic component contract/unit tests and Compose/scene checks.
L2 adds GPU/ROS/model readiness and live service health. It sends no motion.
L3 records one real PWA Mission after an explicit flag, exact approval phrase
   and interactive operator confirmation. This script never invents a route,
   approves a draft, starts a Mission or sends a motion command automatically.

For L3, create and approve an inspection Mission in the PWA, execute it to a
terminal state, then provide a short-lived PWA bearer token through the hidden
prompt or SKILLSTORE_L3_BEARER_TOKEN. Evidence is written mode 0600 beneath
var/ by default and must not be committed.
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
if [[ "${LEVEL}" != L3 && ( -n "${MISSION_ID}" || -n "${OPERATOR_APPROVAL}" ) ]]; then
  printf '%s\n' '--mission-id and --operator-approval are valid only for L3' >&2
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

command -v curl >/dev/null 2>&1 || { printf '%s\n' 'missing command: curl' >&2; exit 1; }
printf '%s\n' 'Acceptance: live service health (read-only; no motion command is sent)'
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8794/healthz >/dev/null
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8792/healthz >/dev/null
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8092/healthz >/dev/null
curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8080/readyz >/dev/null

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
  printf '%s\n' 'L3 bearer tokens may only be sent to an explicit loopback host and numeric port' >&2
  exit 2
}

printf '%s\n' 'L3 sends no command. In the PWA, verify this exact Mission was reviewed, executed and reached a terminal state.'
printf 'Type the Mission ID (%s) to authorize local result capture: ' "${MISSION_ID}" >&2
IFS= read -r typed_mission_id
[[ "${typed_mission_id}" == "${MISSION_ID}" ]] || {
  printf '%s\n' 'operator confirmation did not match the Mission ID' >&2
  exit 2
}

bearer_token="${SKILLSTORE_L3_BEARER_TOKEN:-}"
if [[ -z "${bearer_token}" ]]; then
  printf '%s' 'PWA bearer token (hidden, never persisted in the result): ' >&2
  IFS= read -r -s bearer_token
  printf '\n' >&2
fi
[[ -n "${bearer_token}" ]] || { printf '%s\n' 'a short-lived PWA bearer token is required' >&2; exit 2; }

umask 077
mkdir -p -- "${RESULT_DIR}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
base="${RESULT_DIR}/${timestamp}-${MISSION_ID}"
raw_pending="${base}.mission.json.pending"
raw_result="${base}.mission.json"
summary_pending="${base}.acceptance.json.pending"
summary_result="${base}.acceptance.json"
header_pending="${base}.headers.pending"
cleanup_pending() {
  rm -f -- "${raw_pending}" "${summary_pending}" "${header_pending}"
}
trap cleanup_pending EXIT

printf 'Authorization: Bearer %s\n' "${bearer_token}" >"${header_pending}"
unset bearer_token
curl --noproxy '*' -fsS --max-time 15 \
  -H "@${header_pending}" \
  "${CONTROL_API_BASE_URL}/api/v2/missions/${MISSION_ID}" \
  -o "${raw_pending}"
rm -f -- "${header_pending}"
mv -- "${raw_pending}" "${raw_result}"

repository_commit="$(git -C "${ROOT}" rev-parse HEAD)"
mission_validation_status=0
python3 - \
  "${raw_result}" "${summary_pending}" "${MISSION_ID}" \
  "${repository_commit}" "${RELEASE_ID}" <<'PY' || mission_validation_status=$?
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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
events = mission.get("events") if isinstance(mission.get("events"), list) else []
world_records = (
    mission.get("worldRecords") if isinstance(mission.get("worldRecords"), list) else []
)
accepted = (
    mission_status == "completed"
    and navigation_succeeded
    and inspection_succeeded
    and bool(world_records)
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
        "L3 rejected: expected completed Mission with succeeded navigation, "
        "succeeded inspection and at least one durable world record"
    )
PY
mv -- "${summary_pending}" "${summary_result}"
trap - EXIT
if (( mission_validation_status != 0 )); then
  printf 'acceptance L3: FAILED; structured result: %s\n' "${summary_result}" >&2
  exit 1
fi
printf 'acceptance L3: OK; structured result: %s\n' "${summary_result}"
