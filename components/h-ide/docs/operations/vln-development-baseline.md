# VLN Development Baseline

Recorded before implementation on 2026-08-12 (Asia/Shanghai).

## Repository state

- `h-ide`: commit `1b30e0a`; substantial pre-existing modified and untracked PCD, Assistant and PWA work is user-owned.
- `h-ide-agent`: independent repository with no initial commit; all current files are user-owned and untracked.
- `robotagent`: commit `2d3d4b5`; substantial active Observation Gateway and command work; read-only for this implementation.
- None of the three repositories currently contains `.codegraph/`.

## Test baseline

- H-IDE selected Python tests pass when third-party pytest plugin autoload is disabled: 96 passed.
- Without `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, the ROS Jazzy `launch_testing` plugin fails to import because `lark` is missing. This is an environment/plugin discovery problem, not an H-IDE test failure.
- PWA selected tests: 10 passed; production build passes.
- H-IDE Agent selected tests: 4 passed; TypeScript typecheck passes.

## Implementation rules

- Preserve all pre-existing work and make narrow additive edits.
- Capture changed-boundary tests at every phase exit.
- Do not modify Robot Agent.
- Stop current implementation at P8 with release state `code_complete_robot_agent_pending`.
- Keep real Robot Agent/Nav2 validation frozen until its explicit readiness gate is met.

