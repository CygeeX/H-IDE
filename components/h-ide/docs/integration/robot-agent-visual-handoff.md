# Robot Agent Visual Integration Handoff

This document specifies what H-IDE will deliver to the independently maintained Robot Agent project. The normal ownership rule is that H-IDE does not edit `/home/phi/learn-projects/robotagent`; an explicitly authorized 2026-08-17 exception added only the localization state-source wiring and ROS dependency declarations. Motion adapters, Nav2 execution, watchdogs and safety remain Robot Agent-owned and were not changed by H-IDE.

## Observation boundary

H-IDE Vision owns and distributes:

- `robot_observation/v1/observation.proto`.
- A deterministic descriptor set and `PROTOCOL.lock` SHA-256.
- Valid and invalid binary fixtures plus a conformance runner.
- Required session identity, source identity, timing, calibration, aligned-depth, capture-time TF, localization and flow-control semantics.

Robot Agent must regenerate its bindings and report the exact supported descriptor hash. A major/minor/hash mismatch is an unavailable observation service, not a best-effort compatibility mode.

The Robot Observation Gateway must:

- Use the RealSense ROS driver as the only D435i USB owner.
- Send synchronized JPEG RGB and aligned Z16 depth with calibration and capture-time transforms.
- Distinguish `live`, `replay` and `synthetic` source modes.
- Use newest-only bounded buffering, ACK/reject reasons, rate hints, keepalive and mTLS identity binding.
- Expose no robot command RPC.

## Navigation boundary

H-IDE sends the logical command `navigation.navigate_to_pose` with:

```json
{
  "goalPose": {"frameId": "map", "x": 0, "y": 0, "z": 0, "yaw": 0},
  "goalPurpose": "approach",
  "missionId": "...",
  "stageId": "...",
  "visualResolutionId": "...",
  "observationSessionId": "...",
  "robotBootId": "...",
  "localizationEpochId": "...",
  "maximumSpeedMps": 0.2
}
```

Robot Agent must validate the bindings and map the command to `nav2_msgs/action/NavigateToPose`. It owns path planning, obstacle avoidance, execution, watchdog and local safety. Fibonacci or a multi-waypoint VLN action is not an acceptance substitute.

Normalized lifecycle must include `accepted|active|succeeded|failed|cancelled|unknown`, Robot Agent command ID, ROS goal ID, feedback age and terminal detail. Protocol-level `command.cancel` cancels the active Nav2 goal. Duplicate idempotency keys must not create duplicate ROS goals.

## Deferred validation

The exact readiness gate, test order, measurements and evidence layout are frozen in [robot-agent-deferred-validation.md](robot-agent-deferred-validation.md). Passing H-IDE's protocol simulator proves only H-IDE's side of the boundary.
