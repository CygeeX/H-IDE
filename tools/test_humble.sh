#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="robot-agent-humble:test"

docker build --file "${workspace_dir}/docker/Dockerfile.humble" --tag "${image_name}" "${workspace_dir}"
docker run --rm --network host "${image_name}"
