#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source /opt/h-ide-nav2-overlay/setup.bash
set -u

exec "$@"
