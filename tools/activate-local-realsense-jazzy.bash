#!/usr/bin/env bash
# Source this file only on the current Jazzy development workstation. It exposes
# locally extracted RealSense ROS packages without installing or copying them.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this file instead of executing it" >&2
  exit 2
fi

robotagent_workspace_root="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
robotagent_realsense_prefix="${robotagent_workspace_root}/.local/ros-jazzy-realsense/root/opt/ros/jazzy"

if [[ ! -d "${robotagent_realsense_prefix}/share/realsense2_camera" ]]; then
  echo "local RealSense Jazzy packages are unavailable: ${robotagent_realsense_prefix}" >&2
  return 1
fi

export AMENT_PREFIX_PATH="${robotagent_realsense_prefix}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${robotagent_realsense_prefix}/lib:${robotagent_realsense_prefix}/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${robotagent_realsense_prefix}/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${robotagent_realsense_prefix}/bin:${robotagent_realsense_prefix}/lib/realsense2_camera:${PATH}"

unset robotagent_realsense_prefix
unset robotagent_workspace_root
