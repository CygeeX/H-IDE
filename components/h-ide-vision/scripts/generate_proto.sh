#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
proto_root="${repo_root}/contracts"
proto_file="robot_observation/v1/observation.proto"
generated_root="${repo_root}/src/h_ide_vision/generated"
descriptor="${repo_root}/contracts/robot_observation/v1/descriptor.pb"
lock_file="${repo_root}/contracts/robot_observation/PROTOCOL.lock"

mkdir -p "${generated_root}"
touch "${generated_root}/__init__.py"

python_bin="${PYTHON:-${repo_root}/.venv/bin/python}"
include_root="$("${python_bin}" -c 'import pathlib, grpc_tools; print(pathlib.Path(grpc_tools.__file__).parent / "_proto")')"
"${python_bin}" -m grpc_tools.protoc \
  -I "${proto_root}" \
  -I "${include_root}" \
  --python_out="${generated_root}" \
  --grpc_python_out="${generated_root}" \
  --descriptor_set_out="${descriptor}" \
  --include_imports \
  "${proto_root}/${proto_file}"

hash="$(sha256sum "${descriptor}" | cut -d ' ' -f 1)"
printf 'major=1\nminor=0\nschema_sha256=%s\n' "${hash}" > "${lock_file}"

# grpc_tools emits absolute package imports; make them package-relative.
sed -i 's/^from robot_observation/from h_ide_vision.generated.robot_observation/' \
  "${generated_root}/robot_observation/v1/observation_pb2_grpc.py"

find "${generated_root}/robot_observation" -type d -exec sh -c 'touch "$1/__init__.py"' _ {} \;
printf 'generated %s\n' "${hash}"
