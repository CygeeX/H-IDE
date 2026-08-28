# Continuous Drive Handoff

## Purpose

The frozen Office V2 integration profile used a `0.50 m` runtime
`DriveOnHeading` checkpoint.  After each checkpoint the adapter published zero
velocity and waited for a measured learned-policy `IDLE` boundary.  That
boundary is correct for safety, but it made an open straight corridor look like
“walk, stop, re-arm” every half metre.

This document describes the opt-in continuous profile.  It does not remove
fault handling, cancellation, endpoint checks, fresh localization, or the
normal stop boundary at a turn.

## Behaviour

The profile keeps the official NVIDIA/SONIC forward speed (`0.40 m/s`).  It
allows a runtime drive command up to the static planner's reviewed `2.0 m`
limit when all of the following are true:

1. The selected static segment is collision-free with an additional `0.50 m`
   occupancy inflation contour.
2. The next action is a `DriveOnHeading` in the same signed direction.
3. The current action has a residual endpoint distance beyond the admitted
   chunk.

The adapter still publishes the action-end zero command.  It only avoids the
blocking balanced-IDLE wait when another same-direction chunk is immediately
needed.  A `Spin`, reverse/recovery drive, skip, replan, cancellation, fault,
or final endpoint flushes the pending handoff through the ordinary IDLE gate.
Segments near walls, doorways, the column, or other obstacles automatically use
the frozen `0.50 m` checkpoint policy.

## Start

From the simulator repository:

```bash
./scripts/start-workstation-demo.sh --viewer --continuous-drive-handoff
```

The launcher's default remains the frozen profile.  The equivalent environment
switch is:

```bash
HIDE_CONTINUOUS_DRIVE_HANDOFF=1 ./scripts/start-workstation-demo.sh --viewer
```

The strict adapter health service exposes the active choice:

```text
/h_ide/strict_navigation/health
continuousDriveHandoff: true
runtimeDriveChunkM: 2.0
```

## Rollback

The stable branch and tag are unchanged:

```bash
git switch feat/assistant-v2
./scripts/start-workstation-demo.sh --viewer
```

Omitting `--continuous-drive-handoff` also disables the profile when running
the experiment branch.  Restart the complete coupled simulation chain after a
branch or profile change so Plant, SONIC, ROS bridges, Nav2, and the adapter
share one boot/session identity.

## Verification record

- Adapter regression tests: `18 passed`.
- Formal launch checks: `bash -n scripts/start-workstation-demo.sh` and Python
  byte-compilation pass.
- Live open-corridor A/B runs: approximately `1.99 m` and `1.98 m` traversed
  as one continuous forward action, with terminal `IDLE/GRANTED` and no Plant
  or SONIC fault.
- A multi-segment column route retained short checkpoints near the obstacle;
  its independent endpoint/heading convergence issue is not hidden by this
  optimization and remains a separate navigation task.

## Non-goals

This profile is not a license to bypass the safety interlock or to make a
humanoid accept an unbounded open-loop command.  It does not change the SONIC
model, gait limits, official speed, map provenance, collision model, or final
heading policy.
