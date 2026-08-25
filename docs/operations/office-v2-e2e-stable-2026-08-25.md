# Office V2 end-to-end stable baseline — 2026-08-25

This baseline captures the locally validated PWA → Mission → RobotAgent →
Nav2 → SONIC → MuJoCo Office V2 integration. Use the same annotated tag in
every repository:

`stable/office-v2-e2e-20260825`

## Repository revisions

- `h-ide`: resolve the tag above (this document is committed at that tag).
- `h-ide-mujoco-sonic`: `d8ee5bdcb713380b0830bc89d77b43b67b777f37`
- `robotagent`: `624dd747a8227daa68ace4b4d17d8a8fc01f5e57`
- `h-ide-vision`: `072a49f9741bed54805656c0358dde54fc998a0f`

## Validation evidence

- A clean workstation cold start passed two consecutive startup SAFE_HOLD
  samples with Plant, SONIC, TF, Nav2, RobotAgent, Gateway, Vision, and both
  MuJoCo viewers online.
- A PWA Office V2 Mission navigated from the spawn point to corridor west,
  completed multiple scan rotations, reached reception, and continued toward
  open office without Plant faults, RobotAgent motion faults, MuJoCo non-finite
  values, or Gateway command failures.
- The previously failing transition from a completed navigation command to a
  subsequent scan command successfully reused the exact resident balanced
  SONIC policy generation. Repeated acquire → move/rotate → settle → release
  cycles completed without the former Gateway HTTP 500.
- The final demonstration Mission was cancelled normally before freezing the
  baseline; no command or lease remained active.

## Local rollback

Stop the coupled workstation runtime first. Preserve or commit any later work,
then run the following in each repository:

```bash
git switch --detach stable/office-v2-e2e-20260825
```

To resume branch development from the baseline instead of detached HEAD:

```bash
git switch -c recovery/office-v2-e2e-20260825 \
  stable/office-v2-e2e-20260825
```

The tag records source, configuration, generated scene assets, and tests. It
does not include runtime databases, logs, credentials, model caches, or local
temporary files.
