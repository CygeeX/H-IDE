#!/usr/bin/env bash
set -euo pipefail

# Reproduce the user-space ROS packages consumed by the simulation launcher.
# Packages are downloaded from the operator-configured ROS 2 apt repository,
# verified against the release manifest, and extracted without sudo.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
MANIFEST="${ROOT}/manifests/ros/jazzy-user-prefix-amd64.json"

for command_name in apt-get dpkg-deb python3 sha256sum; do
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

for prefix_row in "${prefix_rows[@]}"; do
  IFS=$'\t' read -r prefix_id relative_target required_path <<<"${prefix_row}"
  target="${ROOT}/${relative_target}"
  if [[ -d "${target}/${required_path}" ]]; then
    printf 'ROS prefix %s: already present at %s\n' "${prefix_id}" "${target}"
    continue
  fi
  if [[ -e "${target}" || -L "${target}" ]]; then
    printf 'refusing incomplete or non-directory ROS prefix: %s\n' "${target}" >&2
    exit 1
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
    (
      cd "${download_dir}"
      apt-get download "${package_name}=${package_version}"
    )
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
  mv "${staging}" "${target}"
  staging=""
  trap - EXIT
  printf 'ROS prefix %s: provisioned at %s\n' "${prefix_id}" "${target}"
done

printf '%s\n' 'ROS Jazzy user prefixes: OK'
