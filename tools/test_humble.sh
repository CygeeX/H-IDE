#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="robot-agent-humble:test"
base_digest="sha256:09da889006b4d4f120ada9b788394d566818f9e451f59a3a9246a1f9eecf849b"
base_config="sha256:90c99f6aa2ad179162a267eb881058cb7e518fca9db73110927bd3950e2903f4"
base_reference="ros:humble-ros-base-jammy@${base_digest}"

# A user-space registry import may leave the exact image content without a
# RepoDigest in Docker. Reuse it only after verifying the immutable config ID.
if local_id="$(docker image inspect ros:humble-ros-base-jammy --format '{{.Id}}' 2>/dev/null)" \
  && [[ "${local_id}" == "${base_config}" ]]; then
  base_reference="ros:humble-ros-base-jammy"
fi

docker build \
  --build-arg "ROS_HUMBLE_BASE=${base_reference}" \
  --file "${workspace_dir}/docker/Dockerfile.humble" \
  --tag "${image_name}" \
  "${workspace_dir}"
docker run --rm --network host "${image_name}"
