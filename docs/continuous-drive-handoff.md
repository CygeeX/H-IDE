# Continuous Drive Handoff and Full Straight Drive

## Purpose

The frozen Office V2 integration profile used a `0.50 m` runtime
`DriveOnHeading` checkpoint.  After each checkpoint the adapter published zero
velocity and waited for a measured learned-policy `IDLE` boundary.  That
boundary is correct for safety, but it made an open straight corridor look like
“walk, stop, re-arm” every half metre.

This document describes the opt-in continuous profile.  It does not remove
fault handling, cancellation, endpoint checks, fresh localization, or the
normal stop boundary at a turn.

## Full straight mode

`full_straight_drive` is a separate, more aggressive profile for a verified
straight segment.  It asks the static planner to use a map-bounded `4 m`
maximum segment, so an open 3–4 m chord can be sent as one
`DriveOnHeading` action.  The action contract
itself has no 2 m limit; the runtime currently caps one action at `4.0 m` so
the conservative progress timeout remains within Nav2's 120-second bound.

Full mode is admitted only when the selected segment remains collision-free at
an additional `0.60 m` occupancy inflation contour.  The single-action bound
is `4.0 m`, because the current conservative progress watchdog would otherwise
produce a `time_allowance` over Nav2's 120-second contract.  Longer clear
paths are divided into a small number of 4 m actions.  If the wider plan
cannot be built/verified, execution automatically falls back to the 2 m
continuous profile and then to the frozen 0.50 m profile.  Turns, doorways,
walls, columns, reverse/recovery actions, and final heading checks remain
segmented.

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

To enable the independent full straight profile:

```bash
./scripts/start-workstation-demo.sh --viewer --full-straight-drive
```

Full mode implies the continuous handoff behavior for any segment that does
not qualify for the full-length contour.  The equivalent environment switch
is `HIDE_FULL_STRAIGHT_DRIVE=1`.

The launcher's default remains the frozen profile.  The equivalent environment
switch is:

```bash
HIDE_CONTINUOUS_DRIVE_HANDOFF=1 ./scripts/start-workstation-demo.sh --viewer
```

The strict adapter health service exposes the active choice:

```text
/h_ide/strict_navigation/health
continuousDriveHandoff: true
fullStraightDrive: false
runtimeDriveChunkM: 2.0
```

## Rollback

The stable branch and tag are unchanged:

```bash
git switch feat/assistant-v2
./scripts/start-workstation-demo.sh --viewer
```

Omitting both profile flags disables the optimization when running the
experiment branch.  Restart the complete coupled simulation chain after a
branch or profile change so Plant, SONIC, ROS bridges, Nav2, and the adapter
share one boot/session identity.

## Verification record

- Adapter regression tests: `20 passed` (including full-straight and fallback
  cases).
- Formal launch checks: `bash -n scripts/start-workstation-demo.sh` and Python
  byte-compilation pass.
- Live open-corridor A/B runs: approximately `1.99 m` and `1.98 m` traversed
  as one continuous forward action, with terminal `IDLE/GRANTED` and no Plant
  or SONIC fault.
- Offline full-straight planner proof: a clear 3 m corridor is partitioned into
  one segment and passes production verification with the full-mode segment
  bound.
- A multi-segment column route retained short checkpoints near the obstacle;
  its independent endpoint/heading convergence issue is not hidden by this
  optimization and remains a separate navigation task.

## Non-goals

This profile is not a license to bypass the safety interlock or to make a
humanoid accept an unbounded open-loop command.  It does not change the SONIC
model, gait limits, official speed, map provenance, collision model, or final
heading policy.
