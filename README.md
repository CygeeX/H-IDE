# H-IDE MuJoCo + SONIC simulation adapter

This is an independent simulation service for the H-IDE workstation.  It is
deliberately not part of the H-IDE control API and it never talks to a physical
motor.  The intended execution boundary is:

```text
H-IDE -> Robot Agent (g1-sim) -> Nav2 NavigateToPose/Spin
      -> Nav2 /cmd_vel -> cmd_vel_guard -> SONIC MovementState
      -> Unitree SDK2 DDS (rt/lowcmd) -> MuJoCo G1
      -> DDS lowstate + synthetic RGB-D/pose -> Observation Gateway/H-IDE
```

Nav2 remains the only navigation owner.  SONIC is a locomotion controller; it
does not receive map goals and it does not perform global/local planning.  A
command is accepted only from the isolated simulation ROS domain (`232`), is
bounded to a slow-walk envelope, and is set to idle when the Nav2 command is
stale or cancelled.

## What is vendored and what is downloaded

* `vendor/lerobot` is pinned to `be49454734c8526340fbd997c374aaa1eb9e83cb`.
  The source is Apache-2.0.
* `models/unitree-g1-mujoco` is the public `lerobot/unitree-g1-mujoco`
  repository.  Its scene/assets are used as the MuJoCo plant.
* SONIC ONNX weights are downloaded at runtime from `nvidia/GEAR-SONIC` and
  must not be committed or baked into an image.  The weights are covered by
  NVIDIA's Open Model License, which is separate from the LeRobot license.

The first install can download several gigabytes of PyTorch/CUDA wheels and
about 865 MB of SONIC weights.  Set `SONIC_FORCE_CPU=1` when checking the
contracts on a host without a compatible CUDA runtime.  CPU mode is suitable
for contract tests, not for a smooth 50 Hz demonstration.

## Office V2 scene contract

`scenes/office_v2/scene.yaml` is the sole authored office inventory. Run
`python tools/build_scene.py` after changing it; the MuJoCo include/runtime
model, textures, Nav2 PGM/YAML and their integrity manifests are generated
together. `python tools/build_scene.py --check` is read-only and fails when a
committed artifact has drifted.

The plant loads `generated/runtime_manifest.json` before constructing MuJoCo.
It verifies the runtime MJCF, office collision include, render textures and
Nav2 map hashes, then uses the manifest's west-corridor spawn
(`x=-6.45, y=0, yaw=0`) only while creating a new process. A physics fault
never invokes an in-process reset or teleport. The native viewer tracks this
spawned G1; the head camera continues to publish RGB-D.

Privileged labels, targets and complete object/spawn inventories exist only in
`scenes/office_v2/evaluation/manifest.json`. That file and `scene.yaml` are
excluded from the Nav2 Docker context and are never exposed by the plant,
H-IDE or Vision runtime. Nav2 receives only the generated occupancy map and a
non-semantic integrity manifest.

## Safety and isolation

* Use `ROS_DOMAIN_ID=232` for every simulation process.  The real Robot Agent
  currently uses domain 231 and must not be mixed with this service.
* The adapter never selects a physical Robot Agent and never exposes a
  physical-execution flag.
* Every mission-level replan is still held at H-IDE's human approval gate.
* `cmd_vel` is clamped to `0.25 m/s` linear and `0.5 rad/s` angular, smoothed,
  and expires after `250 ms`. Linear commands below `0.10 m/s` are held idle
  because SONIC's slow-walk model cannot execute Nav2-style creeping reliably.
* The current navigation-only profile bounds SONIC's arm targets around the
  nominal stance and rate-limits them before `rt/lowcmd`; manipulation-sized
  whole-body output is not permitted until H-IDE exposes an explicit arm skill.
* Continuous Nav2 corrections are latched before SONIC replanning: start/stop
  transitions are immediate, while in-motion clip updates are rate-limited so
  asynchronous planner results cannot overtake newer commands.
* A Nav2 angular-only `cmd_vel` selects SONIC's zero-translation facing-change
  plan.  Rotation is therefore still owned by Nav2 and never by a direct motor
  or head command.
* The workstation profile keeps MuJoCo's simulation-only vertical and
  roll/pitch support enabled as a fall-arrest harness.  Horizontal force and
  yaw torque are explicitly zeroed after a six-second controller startup hold,
  so all accepted map motion still comes from SONIC.
* A Nav2 cancel, bridge disconnect, or stale command publishes an idle SONIC
  intent and sends a zero/hold command to the MuJoCo plant.

## One-command workstation start

The launcher starts the H-IDE Docker services, the isolated Nav2 container,
the MuJoCo plant/viewer, SONIC, both ROS bridges, the simulation Robot Agent
Gateway, `g1-sim`, and its Observation Gateway. It is idempotent: already
running components are reported and left alone.

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
scripts/start-workstation-demo.sh

# Rebuild changed H-IDE container images first when source changed:
scripts/start-workstation-demo.sh --rebuild

# Read-only health summary:
scripts/workstation-demo-status.sh
```

Runtime logs are kept outside the repositories under
`${XDG_RUNTIME_DIR:-/tmp}/h-ide-mujoco-demo-$UID`. The native MuJoCo window is
the third-person physics view; `http://127.0.0.1:8080/simulation` shows the
robot's first-person RGB-D view, root pose, Nav2/SONIC state and trajectory.

## Manual component start (for debugging)

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
source .venv/bin/activate

# Validate imports, pins, model and safety configuration (no simulator start)
python -m hide_mujoco_sonic check --config config/sim.yaml

# Start the authoritative Unitree MuJoCo plant in the isolated DDS domain.
# `simulator` remains an alias for compatibility; it runs this same code path.
python -m hide_mujoco_sonic plant --config config/ros-headless.yaml

# In another terminal, start the SONIC DDS driver.  It listens for guarded
# commands from the ROS bridge and publishes rt/lowcmd at 50 Hz.
python -m hide_mujoco_sonic sonic --config config/ros-headless.yaml

# In the robotagent ROS environment, start the Nav2/Robot-Agent bridge.  A
# full Nav2 Jazzy stack is required for real movement; this process refuses to
# claim readiness when only the mock action server is present.
python -m hide_mujoco_sonic ros-bridge --config config/ros-headless.yaml
```

The ROS commands also require the Jazzy environment, domain 232 and the Robot
Agent overlays. Use the launcher as the executable reference instead of
copying partial `PYTHONPATH` values into a terminal.

The service exposes a small local telemetry endpoint for the H-IDE adapter at
`http://127.0.0.1:8792/healthz` and `.../telemetry`.  It is intentionally a
read-only observation surface.  H-IDE's simulation API can proxy this status
and show whether the real MuJoCo/SONIC chain is running.

## Acceptance gate before deleting the old deterministic backend

The deterministic/FakeAdapter code is retained only as a migration fixture.
It may be removed after the written acceptance record proves all of the
following in the isolated domain: a real Nav2 action server accepts goals,
`/cmd_vel` reaches the guard, SONIC emits non-zero joint targets, MuJoCo walks
and publishes changing root pose, RGB-D reaches the Observation Gateway,
cancel/stale watchdog stops the plant, and ten consecutive replans each wait
for human approval.  Until then, removing the fixture would make failures
ambiguous rather than clean the repository.
