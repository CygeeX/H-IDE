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
  printf '%s\n' '  L2: additionally require ROS/GPU, runtime health prerequisites and external artifacts'
  printf '%s\n' '  Real Mission execution belongs to acceptance L3, never preflight.'
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

need_cmd git
need_cmd python3
need_cmd rg

for component in h-ide h-ide-agent h-ide-vision robotagent h-ide-mujoco-sonic; do
  [[ -d "${COMPONENTS}/${component}" ]] || fail "missing component: ${component}"
done

if [[ ! -d "${ROOT}/.git" ]]; then
  fail 'repository is not a Git worktree'
else
  if git -C "${ROOT}" ls-files -s | awk '$1 == 160000 {print; found=1} END {exit found ? 0 : 1}'; then
    fail 'Git gitlink entries remain; dependency migration is incomplete'
  fi
  repository_risk_status=0
  python3 - "${ROOT}" <<'PY' || repository_risk_status=$?
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from urllib.parse import urlsplit

root = Path(sys.argv[1]).resolve()
git = ["git", "-C", str(root)]
tracked = subprocess.check_output(git + ["ls-files", "-z"]).decode().split("\0")
untracked = subprocess.check_output(
    git + ["ls-files", "--others", "--exclude-standard", "-z"]
).decode().split("\0")
paths = sorted({path for path in tracked + untracked if path})
bad_paths = []
large_current = []
unsafe_symlinks = []
content_hits = []
allowed_example_env = {
    "third_party/lerobot/examples/isaac_teleop_to_so101/default.env",
}

secret_patterns = {
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider-token": re.compile(
        rb"\b(?:ghp_|github_pat_|glpat-|gitea_)[A-Za-z0-9_-]{20,}"
    ),
    "openai-shaped-token": re.compile(rb"\bsk-[A-Za-z0-9]{20,}"),
    "aws-access-key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "credential-in-http-url": re.compile(
        rb"https?://[^/@\s:]+:[^/@\s]+@", re.IGNORECASE
    ),
}
audited_url_templates = {
    ("docs/security/README.md", b"http://user:token@"),
    (
        "third_party/lerobot/.github/workflows/latest_deps_tests.yml",
        b"https://x-access-token:${GH_TOKEN}@",
    ),
    ("integration/preflight.sh", b"http://user:token@"),
    ("integration/preflight.sh", b"https://x-access-token:${GH_TOKEN}@"),
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
        bad_paths.append(path)
    absolute = root / path
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        continue
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(absolute)
        resolved = (absolute.parent / target).resolve(strict=False)
        if os.path.isabs(target) or not resolved.is_relative_to(root):
            unsafe_symlinks.append(path)
        continue
    if not stat.S_ISREG(metadata.st_mode):
        continue
    if metadata.st_size >= 10 * 1024 * 1024:
        large_current.append((metadata.st_size, path))
    if metadata.st_size <= 5 * 1024 * 1024:
        data = absolute.read_bytes()
        for label, pattern in secret_patterns.items():
            for match in pattern.finditer(data):
                matched_value = match.group(0)
                if (
                    label == "credential-in-http-url"
                    and (path, matched_value) in audited_url_templates
                ):
                    # Deliberately documented negative example and a vendored
                    # CI environment-variable reference; neither has a value.
                    continue
                content_hits.append((label, path))
                break

remote_urls = subprocess.check_output(
    git + ["remote", "-v"], text=True, stderr=subprocess.DEVNULL
).splitlines()
unsafe_remotes = []
for line in remote_urls:
    fields = line.split()
    if len(fields) < 2 or not fields[1].startswith(("http://", "https://")):
        continue
    parsed = urlsplit(fields[1])
    if parsed.username is not None or parsed.password is not None:
        unsafe_remotes.append(fields[0])

objects = subprocess.check_output(git + ["rev-list", "--objects", "--all"], text=True)
object_paths = {}
object_ids = []
for line in objects.splitlines():
    oid, separator, path = line.partition(" ")
    object_ids.append(oid)
    if separator and path:
        object_paths.setdefault(oid, path)
checks = subprocess.check_output(
    git + ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
    input="\n".join(object_ids) + "\n",
    text=True,
)
historical_large = []
historical_artifact_paths = set()
artifact_suffixes = (
    ".db", ".sqlite", ".sqlite3", ".onnx", ".pt", ".pth", ".safetensors",
    ".ckpt", ".mcap", ".bag", ".mp4", ".log",
)
for line in checks.splitlines():
    fields = line.split()
    if len(fields) != 3 or fields[1] != "blob":
        continue
    oid, _, raw_size = fields
    size = int(raw_size)
    path = object_paths.get(oid, "<path-unavailable>")
    if size >= 10 * 1024 * 1024:
        historical_large.append((size, oid, path))
    low = path.casefold()
    if low.endswith(artifact_suffixes):
        historical_artifact_paths.add(path)

failed = bool(
    bad_paths or large_current or unsafe_symlinks or content_hits or unsafe_remotes
)
if bad_paths:
    print("FAIL: current tree contains forbidden runtime/secret/artifact paths:")
    for path in bad_paths[:20]:
        print(f"- {path}")
if large_current:
    print("FAIL: current tree contains files at or above the 10 MiB Git limit:")
    for size, path in sorted(large_current, reverse=True)[:20]:
        print(f"- {path} ({size} bytes)")
if unsafe_symlinks:
    print("FAIL: current tree contains absolute or repository-escaping symlinks:")
    for path in unsafe_symlinks[:20]:
        print(f"- {path}")
if content_hits:
    print("FAIL: high-confidence credential patterns found; values are intentionally redacted:")
    for label, path in sorted(set(content_hits))[:20]:
        print(f"- {path} [{label}]")
if unsafe_remotes:
    print("FAIL: one or more HTTP Git remotes embed user information; values are redacted")

history_warning = bool(historical_large or historical_artifact_paths)
if historical_large:
    total = sum(item[0] for item in historical_large)
    largest = max(historical_large)
    print(
        "WARN: Git history retains "
        f"{len(historical_large)} blobs >=10 MiB ({total} bytes total); "
        f"largest is {largest[0]} bytes at {largest[2]}"
    )
if historical_artifact_paths:
    print(
        "WARN: Git history contains "
        f"{len(historical_artifact_paths)} model/data/runtime-like paths; removed files "
        "remain retrievable until an explicitly reviewed history rewrite"
    )
print(
    "Repository risk scan: current tree filenames/sizes/symlinks and selected "
    "high-confidence content patterns checked; Git history filenames/blob sizes checked."
)
print(
    "Repository risk scan limitation: this is not complete secret clearance and does "
    "not replace dedicated server-side scanning or push protection."
)
if failed:
    raise SystemExit(2)
if history_warning:
    raise SystemExit(3)
PY
  case "${repository_risk_status}" in
    0) ;;
    3) warn 'known Git-history artifact/large-blob risk is recorded in the release manifest' ;;
    *) fail 'repository risk scan found a blocking current-tree or remote issue' ;;
  esac
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
  "${ROOT}/manifests/ros/jazzy-user-prefix-amd64.json"
  "${ROOT}/integration/provision-ros-prefixes.sh"
)
for path in "${critical[@]}"; do
  [[ -f "${path}" ]] || fail "missing critical file: ${path}"
done

if absolute_path_matches="$(rg -n '/home/phi/learn-projects|systemed_study' \
  "${sim}/config" \
  "${sim}/scripts/start-workstation-demo.sh" \
  "${sim}/scripts/workstation-demo-status.sh" \
  "${COMPONENTS}/h-ide/infra/docker-compose.yml" || true)" && \
  [[ -n "${absolute_path_matches}" ]]; then
  printf '%s\n' "${absolute_path_matches}" >&2
  fail 'critical runtime files still contain developer-machine absolute paths'
fi

if ! python3 - "${ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(sys.argv[1])
paths = [
    root / "manifests/release/skillstore-office-v2-e2e-20260901-r1.json",
    root / "manifests/release/l3-mission-result.schema.json",
    root / "manifests/sbom/skillstore-office-v2-e2e-20260901-r1.spdx.json",
    root / "manifests/ros/jazzy-user-prefix-amd64.json",
    root / "manifests/vendor/g1-model.json",
    root / "manifests/vendor/groot-wbc.json",
    root / "manifests/vendor/lerobot.json",
    root / "manifests/vendor/unitree-sdk2-python.json",
    root / "components/h-ide-mujoco-sonic/config/sonic-models.lock.json",
]
for path in paths:
    json.loads(path.read_text(encoding="utf-8"))
release = json.loads(paths[0].read_text(encoding="utf-8"))
for component in release.get("components", []):
    component_id = component.get("id")
    source_commit = component.get("sourceCommit")
    component_path = component.get("path")
    distribution_tree = component.get("distributionTree")
    if not all(
        isinstance(value, str) and value
        for value in (component_id, source_commit, component_path, distribution_tree)
    ):
        raise SystemExit("release manifest component identity is incomplete")
    subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{source_commit}^{{commit}}"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    actual_tree = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"HEAD:{component_path}"],
        text=True,
    ).strip()
    if actual_tree != distribution_tree:
        raise SystemExit(
            f"distribution tree mismatch for {component_id}: update the release lock"
        )
ros_prefix = json.loads(paths[3].read_text(encoding="utf-8"))
if ros_prefix.get("schemaVersion") != "skillstore.ros-user-prefix/1":
    raise SystemExit("unexpected ROS user-prefix manifest schema")
if (
    ros_prefix.get("os") != "ubuntu-24.04"
    or ros_prefix.get("architecture") != "amd64"
    or ros_prefix.get("rosDistribution") != "jazzy"
):
    raise SystemExit("unexpected ROS user-prefix platform contract")
prefix_ids = set()
for prefix in ros_prefix.get("prefixes", []):
    prefix_id = prefix.get("id")
    relative_target = prefix.get("relativeTarget")
    required_path = prefix.get("requiredPath")
    if not all(isinstance(value, str) and value for value in (
        prefix_id, relative_target, required_path
    )):
        raise SystemExit("ROS user-prefix identity is incomplete")
    if prefix_id in prefix_ids:
        raise SystemExit(f"duplicate ROS user-prefix id: {prefix_id}")
    prefix_ids.add(prefix_id)
    for value in (relative_target, required_path):
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise SystemExit(f"unsafe ROS user-prefix relative path: {value}")
    packages = prefix.get("packages")
    if not isinstance(packages, list) or not packages:
        raise SystemExit(f"ROS user-prefix has no packages: {prefix_id}")
    for package in packages:
        if not all(
            isinstance(package.get(key), str) and package.get(key)
            for key in ("name", "version", "sha256")
        ) or not re.fullmatch(r"[0-9a-f]{64}", package["sha256"]):
            raise SystemExit(f"invalid ROS package lock in prefix: {prefix_id}")
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
  need_cmd dpkg
  [[ -x "${ROOT}/integration/provision-ros-prefixes.sh" ]] \
    || fail 'ROS user-prefix provisioner is not executable'
  if ! python3 - "${ROOT}" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
manifest_path = root / "manifests/ros/jazzy-user-prefix-amd64.json"
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
architecture = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()
if architecture != manifest["architecture"]:
    raise SystemExit(
        f"ROS prefix architecture mismatch: expected {manifest['architecture']}, got {architecture}"
    )
os_release = {}
for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
    key, separator, value = line.partition("=")
    if separator:
        os_release[key] = value.strip().strip('"')
expected_os = f"{os_release.get('ID', '')}-{os_release.get('VERSION_ID', '')}"
if expected_os != manifest["os"]:
    raise SystemExit(
        f"ROS prefix OS mismatch: expected {manifest['os']}, got {expected_os}"
    )
for prefix in manifest["prefixes"]:
    target = root / prefix["relativeTarget"]
    required = target / prefix["requiredPath"]
    copied_manifest = target / "release-manifest.json"
    if not target.is_dir() or not required.is_dir():
        raise SystemExit(
            f"ROS prefix {prefix['id']} is absent/incomplete; "
            "run integration/bootstrap.sh --provision-ros-prefixes"
        )
    if not copied_manifest.is_file() or copied_manifest.read_bytes() != manifest_bytes:
        raise SystemExit(f"ROS prefix {prefix['id']} manifest copy does not match release lock")
    package_dir = target / "packages"
    actual_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in package_dir.glob("*.deb")
        if path.is_file()
    }
    missing_hashes = {
        package["sha256"] for package in prefix["packages"]
    } - actual_hashes
    if missing_hashes:
        raise SystemExit(f"ROS prefix {prefix['id']} is missing locked package payloads")
print("ROS Jazzy user prefixes: manifest, platform, payloads and required paths OK")
PY
  then
    fail 'ROS Jazzy user-prefix validation failed'
  fi
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
