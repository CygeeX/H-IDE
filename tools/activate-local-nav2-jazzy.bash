#!/usr/bin/env bash
# Source after /opt/ros/jazzy/setup.bash to use a user-space nav2_msgs package.

robotagent_workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robotagent_nav2_prefix="${robotagent_workspace_root}/.local/ros-jazzy-nav2/root/opt/ros/jazzy"

if [[ ! -d "${robotagent_nav2_prefix}/share/nav2_msgs" ]]; then
  echo "local Jazzy nav2_msgs is unavailable: ${robotagent_nav2_prefix}" >&2
  unset robotagent_workspace_root robotagent_nav2_prefix
  return 1
fi

export AMENT_PREFIX_PATH="${robotagent_nav2_prefix}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${robotagent_nav2_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${robotagent_nav2_prefix}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${robotagent_nav2_prefix}/bin${PATH:+:${PATH}}"

unset robotagent_workspace_root robotagent_nav2_prefix
