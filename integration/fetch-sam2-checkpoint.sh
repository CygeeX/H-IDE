#!/usr/bin/env bash
set -euo pipefail

# Fetch exactly one operator-owned SAM2 checkpoint. Runtime services never call
# this script; it only materializes a manifest-locked file outside the checkout.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
MANIFEST="${ROOT}/manifests/vendor/sam2-checkpoint.json"
TARGET=""
ENV_FILE=""

usage() {
  cat <<'EOF'
usage: integration/fetch-sam2-checkpoint.sh --target ABSOLUTE_PATH [--env-file PATH]

Download the exact SAM 2.1 checkpoint recorded in the release manifest. The
target parent must already exist, be owned by the current user, and not be
group/other writable. Existing correct targets are verified idempotently;
existing incorrect targets are never overwritten.

--env-file safely updates only HIDE_VISION_SAM2_CHECKPOINT in an existing
mode-0600, current-user-owned regular dotenv file. The file is never sourced.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
      TARGET="$2"
      shift 2
      ;;
    --env-file)
      [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
      ENV_FILE="$2"
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

[[ -n "${TARGET}" ]] || { usage >&2; exit 2; }
for command_name in curl mktemp python3 sha256sum stat; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "${command_name}" >&2
    exit 1
  }
done

mapfile -t manifest_values < <(python3 - "${MANIFEST}" <<'PY'
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("schema_version") != "skillstore.external-model/1":
    raise SystemExit("unsupported SAM2 checkpoint manifest schema")
filename = payload.get("filename")
url = payload.get("source_url")
digest = payload.get("sha256")
size = payload.get("byte_size")
if (
    not isinstance(filename, str)
    or not filename
    or Path(filename).name != filename
    or filename in {".", ".."}
):
    raise SystemExit("SAM2 manifest filename must be one safe path component")
if not isinstance(url, str):
    raise SystemExit("SAM2 manifest source URL is missing")
parsed = urlsplit(url)
if (
    parsed.scheme != "https"
    or not parsed.netloc
    or parsed.username is not None
    or parsed.password is not None
    or parsed.query
    or parsed.fragment
):
    raise SystemExit("SAM2 manifest source must be a credential-free HTTPS URL")
if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
    raise SystemExit("SAM2 manifest SHA-256 is invalid")
if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
    raise SystemExit("SAM2 manifest byte size is invalid")
if payload.get("redistribution") != "operator_fetch":
    raise SystemExit("SAM2 manifest must retain the operator-fetch boundary")
print(url)
print(filename)
print(size)
print(digest)
PY
)
[[ "${#manifest_values[@]}" -eq 4 ]] || {
  printf '%s\n' 'SAM2 manifest did not produce one complete asset identity' >&2
  exit 1
}
SOURCE_URL="${manifest_values[0]}"
EXPECTED_FILENAME="${manifest_values[1]}"
EXPECTED_SIZE="${manifest_values[2]}"
EXPECTED_SHA256="${manifest_values[3]}"

mapfile -t target_values < <(python3 - \
  "${ROOT}" "${TARGET}" "${EXPECTED_FILENAME}" <<'PY'
from pathlib import Path
import os
import stat
import sys

root = Path(sys.argv[1]).resolve(strict=True)
raw_target = sys.argv[2]
expected_filename = sys.argv[3]
if "\x00" in raw_target or "\n" in raw_target or "\r" in raw_target:
    raise SystemExit("SAM2 target contains a forbidden control character")
target_input = Path(raw_target)
if not target_input.is_absolute():
    raise SystemExit("SAM2 --target must be absolute")
if target_input.name != expected_filename:
    raise SystemExit(f"SAM2 target filename must be exactly {expected_filename}")
parent_input = target_input.parent
if parent_input.is_symlink():
    raise SystemExit("SAM2 target parent must not be a symbolic link")
try:
    parent = parent_input.resolve(strict=True)
except FileNotFoundError as exc:
    raise SystemExit("SAM2 target parent must already exist") from exc
metadata = parent.stat()
if not stat.S_ISDIR(metadata.st_mode):
    raise SystemExit("SAM2 target parent must be a directory")
if metadata.st_uid != os.getuid():
    raise SystemExit("SAM2 target parent must be owned by the current user")
if stat.S_IMODE(metadata.st_mode) & 0o022:
    raise SystemExit("SAM2 target parent must not be group/other writable")
target = parent / expected_filename
if target == root or target.is_relative_to(root):
    raise SystemExit("SAM2 target must stay outside the Git checkout")
if target.is_symlink():
    raise SystemExit("SAM2 target must not be a symbolic link")
if target.exists():
    target_metadata = target.stat()
    if not stat.S_ISREG(target_metadata.st_mode):
        raise SystemExit("SAM2 target must be a regular file")
    if target_metadata.st_uid != os.getuid():
        raise SystemExit("SAM2 target must be owned by the current user")
print(target)
print(parent)
print("1" if target.exists() else "0")
PY
)
[[ "${#target_values[@]}" -eq 3 ]] || {
  printf '%s\n' 'SAM2 target validation did not produce one canonical path' >&2
  exit 1
}
TARGET="${target_values[0]}"
TARGET_PARENT="${target_values[1]}"
TARGET_EXISTS="${target_values[2]}"

verify_checkpoint() {
  local path="$1"
  local actual_size=""
  local actual_sha256=""
  [[ -f "${path}" && ! -L "${path}" ]] || return 1
  actual_size="$(stat -c '%s' -- "${path}")"
  [[ "${actual_size}" == "${EXPECTED_SIZE}" ]] || return 1
  actual_sha256="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${actual_sha256}" == "${EXPECTED_SHA256}" ]]
}

update_env_file() {
  [[ -n "${ENV_FILE}" ]] || return 0
  python3 - "${ENV_FILE}" "${TARGET}" <<'PY'
from pathlib import Path
import os
import stat
import sys
import tempfile

path = Path(sys.argv[1])
target = sys.argv[2]
if "\x00" in target or "\n" in target or "\r" in target:
    raise SystemExit("SAM2 target cannot be represented in the environment file")
if path.is_symlink():
    raise SystemExit("SAM2 environment file must not be a symbolic link")
try:
    metadata = path.stat()
except FileNotFoundError as exc:
    raise SystemExit("SAM2 environment file must already exist") from exc
if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
    raise SystemExit("SAM2 environment file must be a current-user-owned regular file")
if stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit("SAM2 environment file must have mode 0600")
lines = path.read_text(encoding="utf-8").splitlines()
key = "HIDE_VISION_SAM2_CHECKPOINT"
matches = [index for index, line in enumerate(lines) if line.startswith(f"{key}=")]
if len(matches) > 1:
    raise SystemExit(f"environment file contains duplicate key: {key}")
replacement = f"{key}={target}"
if matches:
    lines[matches[0]] = replacement
else:
    lines.append(replacement)
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_name, 0o600)
    os.replace(temporary_name, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
PY
}

if [[ "${TARGET_EXISTS}" == 1 ]]; then
  if ! verify_checkpoint "${TARGET}"; then
    printf 'existing SAM2 target does not match the release manifest: %s\n' \
      "${TARGET}" >&2
    exit 1
  fi
  update_env_file
  printf 'SAM2 checkpoint verified: %s\n' "${TARGET}"
  exit 0
fi

temporary="$(mktemp --tmpdir="${TARGET_PARENT}" ".${EXPECTED_FILENAME}.download.XXXXXX")"
cleanup_temporary() {
  if [[ -n "${temporary:-}" && -e "${temporary}" && ! -L "${temporary}" ]]; then
    rm -f -- "${temporary}"
  fi
}
trap cleanup_temporary EXIT
chmod 0600 -- "${temporary}"

curl --fail --location --silent --show-error \
  --proto '=https' --proto-redir '=https' --tlsv1.2 \
  --connect-timeout 20 --max-time 1800 --retry 3 \
  --output "${temporary}" \
  "${SOURCE_URL}"

if ! verify_checkpoint "${temporary}"; then
  printf '%s\n' 'downloaded SAM2 checkpoint does not match manifest byte size/SHA-256' >&2
  exit 1
fi

# Hard-linking a same-directory temporary file provides atomic no-clobber
# publication: os.link fails if another process or symlink already owns target.
python3 - "${temporary}" "${TARGET}" <<'PY'
from pathlib import Path
import os
import sys

temporary = Path(sys.argv[1])
target = Path(sys.argv[2])
with temporary.open("rb") as stream:
    os.fsync(stream.fileno())
try:
    os.link(temporary, target, follow_symlinks=False)
except FileExistsError as exc:
    raise SystemExit("SAM2 target appeared during download; refusing to overwrite it") from exc
temporary.unlink()
directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
temporary=""
trap - EXIT

if ! verify_checkpoint "${TARGET}"; then
  printf '%s\n' 'installed SAM2 checkpoint failed post-install verification' >&2
  exit 1
fi
update_env_file
printf 'SAM2 checkpoint installed and verified: %s\n' "${TARGET}"
