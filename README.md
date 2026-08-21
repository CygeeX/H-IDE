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

* `vendor/lerobot` is pinned to `cab2c37a172c0101391d95f754123ed054f1b6e6`.
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
spawned G1 when the optional viewer is enabled; the head camera continues to
publish RGB-D.

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
* Every admitted motion intent and asynchronous planner request has a local
  generation/signature lineage.  Reference authority is binary: the exact last
  successful 29-DoF DDS target remains held until a uniquely correlated
  reference is installed and primed, and only its first successful DDS write
  grants motion.  There is no post-policy target blend.  The exact published
  target is converted back into executed-action history, and the reference
  cursor advances only for a candidate from the same install epoch and cursor.
  Translational references must also predict positive root displacement in the
  requested direction.  Ambiguous, stale and opposite-direction results are
  dropped rather than guessed; an absent matching reference times out into a
  terminal exact-target hold.  A bad periodic refresh is discarded without
  revoking the currently proven reference.
* Pure rotations additionally require the planned root quaternion trajectory
  to contain a meaningful shortest yaw delta with the same sign as the
  requested facing error. Translation continues to use root projection as its
  authority gate; its yaw evidence is recorded for diagnosis only.
* A Nav2 angular-only `cmd_vel` selects SONIC's zero-translation facing-change
  plan.  Rotation is therefore still owned by Nav2 and never by a direct motor
  or head command.
* The workstation profile keeps MuJoCo's simulation-only vertical and
  roll/pitch support enabled as a fall-arrest harness.  Horizontal force and
  yaw torque are explicitly zeroed after a six-second controller startup hold,
  so all accepted map motion still comes from SONIC.
* A Nav2 cancel, bridge disconnect, or stale command publishes an idle SONIC
  intent and sends a zero/hold command to the MuJoCo plant.
* A normal Robot Agent safe-stop uses a correlated transaction rather than the
  terminal-fault path: SONIC first freezes and revokes authority, GaitAdapter
  requires fresh post-request stationary odometry, Plant then latches
  `safety_stop`, and Driver identity/epoch plus fresh health are revalidated
  before success is exposed.  The bridge transaction is bounded to 3.5 s and
  its loopback proxy to 3.65 s; failures remain failures and never synthesize a
  confirmed hold.
* GaitAdapter faults, SONIC loss/terminal faults, and confirmed stale LowState
  reception or SONIC control steps publish one strict
  `h-ide.motion-safety-fault/v1` JSON event on
  `/h_ide/g1_sim/motion_safety_fault` with reliable, transient-local QoS. The
  event is published before the local Plant/SONIC stop is attempted. It carries
  robot/epoch, bridge and driver boot IDs, source sequence, control generation,
  event ID and health-probe request ID for isolation and de-duplication.
  Robot Agent remains the sole Nav2 action owner and cancels only its correlated
  goal handle; the simulator never uses a global `CancelGoal(all)` operation.
* The existing loopback-only port 8793 also accepts a strict, read-only
  `h-ide.sonic-health/v3` probe. In addition to LowState/control-step ages and
  the immutable driver boot ID, it carries typed Plant-support latch/pending,
  static-hold, Nav2-task, motion-authority and zero-command facts. Age-only
  faults require two consecutive 25 ms temporal-unhealthy probes with the same
  driver boot ID while motion is authorized or the gait is not settled at
  IDLE; alternating LowState/control-step codes share that one streak.
  A fully confirmed `safety_stop + IDLE + static hold + authority revoked +
  zero command + no task` quarantines age-only degradation instead of emitting
  an irreversible motion fault; a fresh probe recovers it, and `control.acquire`
  remains blocked until both ages are fresh. Terminal faults, unavailability,
  probe failures, and driver boot changes remain immediate. The 100 ms stale
  bound plus two probe periods and the 40 ms timeout yields a 190 ms worst-case
  detection budget. The 90 ms confirmation handoff leaves 10 ms for the
  latched ROS event to begin Robot Agent cancellation within 100 ms; Robot
  Agent retains the 300 ms Nav2 terminal deadline, without placing ROS code
  inside SONIC.
* Robot Agent diagnostics remain loopback-only on `127.0.0.1:8796`. The
  launcher starts a separate proxy on Docker's dynamically verified default
  bridge at port `8797`; it can forward only bodyless GET requests for
  `/healthz`, `/readyz`, `/v1/diagnostics`, and `/v1/capabilities` to that fixed
  loopback upstream. It rejects redirects, query strings, request bodies,
  non-JSON/oversized responses, and every write method.

## One-command workstation start

The launcher starts the H-IDE Docker services, the isolated Nav2 container,
the MuJoCo plant and offscreen RGB-D worker, SONIC, both ROS bridges, the
simulation Robot Agent Gateway, `g1-sim`, and its Observation Gateway. It is
idempotent: already running components are reported and left alone.

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
scripts/start-workstation-demo.sh

# Rebuild changed H-IDE container images first when source changed:
scripts/start-workstation-demo.sh --rebuild

# Add the local third-person window and live first-person companion:
scripts/start-workstation-demo.sh --viewer

# Cold-start only the isolated motion core used by strict locomotion tests.
# RobotAgent, its gateways, H-IDE Compose and Vision are not started:
scripts/start-workstation-demo.sh --motion-core --viewer

# Read-only health summary:
scripts/workstation-demo-status.sh
```

Runtime logs are kept outside the repositories under
`${XDG_RUNTIME_DIR:-/tmp}/h-ide-mujoco-demo-$UID`. The ordinary workstation
profile may run without the native viewer; the isolated strict motion profile
requires both viewers and binds them to the same Plant boot/session.
`http://127.0.0.1:8080/developer/simulation` remains the full-workstation view
when H-IDE Compose is running.

`--viewer` preserves that exact reviewed configuration and adds a passive
third-person viewer with a private model/data copy plus a read-only first-person
RGB companion window. It requires the logged-in X11 session and can be selected
only during a cold Plant start; neither window can write controller state.

`--motion-core` is intentionally not an idempotent partial restart. It rejects
surviving RobotAgent/gateway units and listening ports 8796/8094, then starts
only Plant, SONIC, the ROS motion/observation/safe-stop bridges, Nav2, and the
two passive viewers. It exits with the robot stationary and sends no motion.
Run exactly one explicitly authorized case from a second terminal:

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
source /opt/ros/jazzy/setup.bash
source /home/phi/learn-projects/robotagent/tools/activate-local-nav2-jazzy.bash
export ROS_DOMAIN_ID=232
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/></Interfaces><AllowMulticast>true</AllowMulticast></General></Domain></CycloneDDS>'
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p var/acceptance

/usr/bin/python3 tools/strict_motion_runner.py \
  --mode forward \
  --case-id forward-1m-run-1 \
  --output "var/acceptance/forward-1m-run-1.json" \
  --execute
```

The accepted standalone modes are `forward` (1 m), `forward_3m`, `reverse`
(0.5 m), `lateral_left`, `lateral_right` (0.5 m), `turn_left`, and
`turn_right` (90 degrees). `detour` additionally requires one of the two
locked Office V2 waste-bin routes. DriveOnHeading uses NVIDIA/Nav2's reviewed
0.20 m/s, true lateral uses NVIDIA's 0.40 m/s, and Spin stays within
0.10–0.20 rad/s. A trace path is created only after its matching evaluated
result is durable; any fault stops, cancels, aborts evidence, and never retries
automatically. This proves the isolated `motion_core` tier only—not the H-IDE
RobotAgent full chain.

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
and show whether the real MuJoCo/SONIC chain is running.  Planner-evidence
fields expose the current intent generation/signature, request and accepted
generations, reference cursor, predicted root direction/projection, and the
requested/predicted root-yaw deltas. The same fields are copied into every
bounded pre-fault control-trace sample.

The telemetry JSON also includes a read-only `runtime_diagnostics` snapshot.
It records LowState-callback, successful-control-publish, and health-probe
gaps, together with CPython cyclic-GC count, last/max/total pause, and how much
completed GC time overlapped each last gap.  For every stream it also preserves
the latest over-threshold gap and GC overlap after healthy samples resume, plus
counters for over-threshold gaps with any GC overlap and those whose remaining
non-GC time independently exceeds the same threshold. These durable fields use
the `*_last_over_threshold_gap_s`,
`*_last_over_threshold_gc_pause_s`,
`*_over_threshold_gc_overlap_count`, and
`*_over_threshold_non_gc_residual_count` suffixes. SONIC performs one full
collection and freezes the warmed long-lived Python object graph before arming
its strict control clock; automatic GC remains enabled for runtime request and
control objects, and the applied/error state is explicit in telemetry. The
LowState-only hot update mutates scalar telemetry in place; the ordinary update
API retains its isolated-snapshot return contract, and HTTP readers also
receive isolated copies.

MuJoCo `/state` and `/healthz` expose `runtimeTiming` for the authoritative
simulation-loop gap, physics step, and non-blocking state-snapshot publication.
They also expose a separately identified `dynamicsEvidence` sample captured by
the MuJoCo authority thread after each step.  It records
`supportBandEnabled`, the `attachedBody`, the world-frame
`lastAppliedWrench`, the complete floating-root qpos/qvel/qacc and solver
warm-start acceleration, MuJoCo potential/kinetic energy, and numerical-warning
counts.  The sample is an immutable, read-only diagnostic projection:
`diagnosticOnly=true` and `authorityFeedbackEnabled=false`; it is neither
consumed by a controller nor forwarded through the ROS observation bridge.
RGB, depth and JPEG work runs only in a `spawn` worker with its own verified
OfficeV2 model, `MjData` and Renderer; its per-frame costs are reported as
`renderTiming`.  A worker failure or a frame older than 500 ms changes only
`observationStatus`, makes `/healthz` return 503, and never enables a synchronous
render fallback in the physics process.  The optional operator viewer follows
the same private-model rule and has no control/input callback into the plant.

`/rgb.jpg` and `/depth.raw` remain compatible as latest-frame routes.  Atomic
consumers first read `/state`, then request both resources with the exact
`?boot=<producerBootId>&sequence=<sequence>` identity.  The latest and previous
complete captures are retained; a missing generation returns 409, and all
three responses carry matching capture identity headers.  The ROS observation
bridge enforces those identities and rechecks the original capture timestamp
against the same 500-ms limit before publishing.  These changes do not widen
the 100-ms physics safety limits or replay stale samples.

`tests/test_stop_dynamics_regression.py` loads the real OfficeV2 runtime XML
and runs a deterministic terminal-stop A/B at the configured 4 ms timestep.
Its small fixture reconstructs the last retained finite root/29-DoF state,
frozen SONIC target, and deployed gains; it is explicitly not bit-exact because
the retained Plant and SONIC samples were about one tick apart and no contact
solver warm start was recorded.  The regression requires the production
zero-yaw `safety_stop` wrench to remain finite, warning-free, and inside the
validated office envelope for two seconds, while the historical tethered-torso
`-80 * wz` counterfactual must exceed acceleration/speed limits and leave the
envelope within 80 ms.  The multi-megabyte raw trace is intentionally not part
of the repository.

### Read-only SAFE_HOLD soak evidence

After a coordinated cold start has reached Robot Agent `STANDBY`, collect the
ten-minute pre-motion record without sending a control, Nav2, reset, stop, or
rearm request:

```bash
cd /home/phi/learn-projects/h-ide-mujoco-sonic
.venv/bin/python tools/idle_soak_monitor.py \
  --output var/acceptance/$(date -u +%Y%m%dT%H%M%SZ)-safe-hold/idle-soak.jsonl
```

The default duration is 605 seconds so the recorded first-to-last observation
spans at least ten minutes. Plant and SONIC are sampled at 2 Hz; the more
expensive Robot Agent diagnostics probe is sampled every five seconds. Network
access is compiled down to bodyless GET requests against exactly
`127.0.0.1:8794/state`, `127.0.0.1:8792/telemetry`, and
`127.0.0.1:8796/v1/diagnostics`. There is no URL override or mutable client.

The JSONL path is create-exclusive: an existing file, target symlink, or
symlinked ancestor is rejected. It contains a complete baseline and final
checkpoint, compact 2-Hz samples, full raw checkpoints every ten seconds, and
an immediate full checkpoint whenever a cumulative runtime-stall counter or
motion-critical PID/start-time identity changes. The monitor also scans only
new data in the motion-critical runtime logs for QACC, NaN/non-finite,
numerical-instability, implicit-reset, teleport/rollback, traceback, and crash
clues. The existing `MotionAcceptanceEvaluator` remains the single source of
truth for SAFE_HOLD structure, pose drift, fall/tilt, unexpected intent, fault
trace, reset clues, and arm limits; the soak monitor adds stricter stationary
root-speed and lower-body limits.

The result deliberately has two verdicts:

- `safeHoldSafetyPass` means the robot remained structurally stationary and
  Robot Agent remained ready in authority-revoked `STANDBY`, with no numerical,
  process-identity, counter-rollback, or terminal-fault evidence.
- `movementReleaseAllowed` additionally requires zero increase from the
  baseline in every Plant and SONIC over-threshold runtime counter and no stale
  marker in the runtime logs. Any `SONIC_HEALTH_PROBE_TIMEOUT` marker recorded
  in the window also keeps this verdict false, including a single timeout that
  later recovered and a timeout from the separately reserved normal-stop path.

An isolated age/stage stall may therefore leave `safeHoldSafetyPass=true` while
forcing `movementReleaseAllowed=false`. If Robot Agent remained ready, the
record also sets `quarantineExercised=true`: this proves the safe-hold
quarantine did not permanently fault the agent, but the observed performance
stall still blocks a motion attempt. With no natural stall,
`quarantineExercised=false`; the injected offline safety tests remain the
evidence for that unexercised branch. Transport uncertainty is reported
separately as `healthProbeTransportTimeoutLogSeen` and
`healthProbeTransportTimeoutExercised`; it is never folded into the age-only
quarantine verdict. Exit status is zero only when movement is released, one
for a completed HOLD/failed record, and two for setup errors.
`--duration` exists for offline and short diagnostic tests; reducing it does
not produce ten-minute acceptance evidence.

## Acceptance gate before deleting the old deterministic backend

The deterministic/FakeAdapter code is retained only as a migration fixture.
It may be removed after the written acceptance record proves all of the
following in the isolated domain: a real Nav2 action server accepts goals,
`/cmd_vel` reaches the guard, SONIC emits non-zero joint targets, MuJoCo walks
and publishes changing root pose, RGB-D reaches the Observation Gateway,
cancel/stale watchdog stops the plant, and ten consecutive replans each wait
for human approval.  Until then, removing the fixture would make failures
ambiguous rather than clean the repository.
