#!/usr/bin/env bash
set -euo pipefail

# Reproduce the user-space ROS packages consumed by the simulation launcher.
# Packages are downloaded from the immutable ROS buildfarm snapshot named in
# the manifest, verified by SHA-256, and extracted without sudo.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
MANIFEST="${ROOT}/manifests/ros/jazzy-user-prefix-amd64.json"

for command_name in curl dpkg dpkg-deb python3 sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "${command_name}" >&2
    exit 1
  }
done

architecture="$(dpkg --print-architecture)"
[[ "${architecture}" == amd64 ]] || {
  printf 'the frozen ROS prefix supports amd64 only (found %s)\n' "${architecture}" >&2
  exit 1
}

source_base_url="$(python3 - "${MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
source = payload.get("source")
if not isinstance(source, dict) or source.get("type") != "ros-buildfarm-snapshot":
    raise SystemExit("ROS prefix manifest does not name a buildfarm snapshot")
base_url = source.get("baseUrl")
if not isinstance(base_url, str) or not base_url:
    raise SystemExit("ROS prefix manifest snapshot URL is missing")
print(base_url.rstrip("/"))
PY
)"

mapfile -t prefix_rows < <(python3 - "${MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("schemaVersion") != "skillstore.ros-user-prefix/1":
    raise SystemExit("unsupported ROS prefix manifest")
for prefix in payload["prefixes"]:
    print("\t".join((prefix["id"], prefix["relativeTarget"], prefix["requiredPath"])))
PY
)

[[ "${#prefix_rows[@]}" -gt 0 ]] || {
  printf '%s\n' 'ROS prefix manifest contains no prefixes' >&2
  exit 1
}

prefix_matches_manifest() {
  local target="$1"
  local prefix_id="$2"
  local required_path="$3"
  python3 - "${MANIFEST}" "${target}" "${prefix_id}" "${required_path}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
target = Path(sys.argv[2])
prefix_id = sys.argv[3]
required_path = sys.argv[4]
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
prefix = next(item for item in manifest["prefixes"] if item["id"] == prefix_id)

if target.is_symlink() or not (target / required_path).is_dir():
    raise SystemExit(1)
copied_manifest = target / "release-manifest.json"
if not copied_manifest.is_file() or copied_manifest.read_bytes() != manifest_bytes:
    raise SystemExit(1)
package_dir = target / "packages"
actual_hashes = {
    hashlib.sha256(path.read_bytes()).hexdigest()
    for path in package_dir.glob("*.deb")
    if path.is_file()
}
expected_hashes = {package["sha256"] for package in prefix["packages"]}
if actual_hashes != expected_hashes:
    raise SystemExit(1)
PY
}

for prefix_row in "${prefix_rows[@]}"; do
  IFS=$'\t' read -r prefix_id relative_target required_path <<<"${prefix_row}"
  target="${ROOT}/${relative_target}"
  if prefix_matches_manifest "${target}" "${prefix_id}" "${required_path}"; then
    printf 'ROS prefix %s: verified at %s\n' "${prefix_id}" "${target}"
    continue
  fi
  refresh_target=0
  if [[ -L "${target}" || ( -e "${target}" && ! -d "${target}" ) ]]; then
    printf 'refusing non-directory ROS prefix: %s\n' "${target}" >&2
    exit 1
  elif [[ -d "${target}" ]]; then
    refresh_target=1
  fi

  parent="$(dirname -- "${target}")"
  mkdir -p "${parent}"
  staging="$(mktemp -d "${parent}/.${prefix_id}.staging.XXXXXX")"
  cleanup_staging() {
    if [[ -n "${staging:-}" && -d "${staging}" ]]; then
      find "${staging}" -depth -mindepth 1 -delete
      rmdir "${staging}"
    fi
  }
  trap cleanup_staging EXIT
  mkdir -p "${staging}/root" "${staging}/packages"

  mapfile -t package_rows < <(python3 - "${MANIFEST}" "${prefix_id}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prefix_id = sys.argv[2]
prefix = next(item for item in payload["prefixes"] if item["id"] == prefix_id)
for package in prefix["packages"]:
    print("\t".join((package["name"], package["version"], package["sha256"])))
PY
  )

  for package_row in "${package_rows[@]}"; do
    IFS=$'\t' read -r package_name package_version expected_sha256 <<<"${package_row}"
    download_dir="$(mktemp -d "${staging}/download.XXXXXX")"
    package_filename="${package_name}_${package_version}_${architecture}.deb"
    package_url="${source_base_url}/pool/main/r/${package_name}/${package_filename}"
    curl --fail --location --silent --show-error \
      --output "${download_dir}/${package_filename}" "${package_url}"
    mapfile -t debs < <(find "${download_dir}" -maxdepth 1 -type f -name '*.deb' -print)
    [[ "${#debs[@]}" -eq 1 ]] || {
      printf 'expected one deb for %s, found %s\n' "${package_name}" "${#debs[@]}" >&2
      exit 1
    }
    actual_sha256="$(sha256sum "${debs[0]}" | awk '{print $1}')"
    [[ "${actual_sha256}" == "${expected_sha256}" ]] || {
      printf 'SHA-256 mismatch for %s=%s\n' "${package_name}" "${package_version}" >&2
      exit 1
    }
    dpkg-deb -x "${debs[0]}" "${staging}/root"
    mv "${debs[0]}" "${staging}/packages/"
    rmdir "${download_dir}"
  done

  [[ -d "${staging}/${required_path}" ]] || {
    printf 'provisioned prefix is missing %s\n' "${required_path}" >&2
    exit 1
  }
  cp "${MANIFEST}" "${staging}/release-manifest.json"
  if [[ "${refresh_target}" -eq 1 ]]; then
    backup="$(mktemp -d "${parent}/.${prefix_id}.previous.XXXXXX")"
    rmdir "${backup}"
    mv "${target}" "${backup}"
    if ! mv "${staging}" "${target}"; then
      mv "${backup}" "${target}"
      printf 'failed to replace ROS prefix %s; restored the previous prefix\n' \
        "${prefix_id}" >&2
      exit 1
    fi
    staging=""
    find "${backup}" -depth -mindepth 1 -delete
    rmdir "${backup}"
    printf 'ROS prefix %s: refreshed at %s\n' "${prefix_id}" "${target}"
  else
    mv "${staging}" "${target}"
    staging=""
    printf 'ROS prefix %s: provisioned at %s\n' "${prefix_id}" "${target}"
  fi
  trap - EXIT
done

printf '%s\n' 'ROS Jazzy user prefixes: OK'
