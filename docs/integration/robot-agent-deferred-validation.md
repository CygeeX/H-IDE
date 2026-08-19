# Robot Agent Deferred Integration and Acceptance Package

**Status:** `FROZEN`

**Reason:** H-IDE and Robot Agent are being developed concurrently. The local
Robot Agent configuration and user-space Jazzy `nav2_msgs` interfaces now pass
their focused checks, but the Robot Agent/Observation Gateway/vision worktrees
are not version-frozen and the complete RA evidence matrix has not been rerun.
Real behavior therefore cannot yet be treated as a stable release dependency.

**Current release label:** `code_complete_robot_agent_pending`

This document preserves the tests that must be run later. It is not evidence that those tests have passed. Until every required row is completed with real artifacts, H-IDE must not claim `robot_agent_integrated`, Nav2 acceptance, or physical robot success.

**2026-08-13 note:** a development pre-integration exercised real local
components and is recorded in
[local-preintegration-2026-08-13.md](local-preintegration-2026-08-13.md).
The development run completed one text-to-Mock-Nav2 happy path, but did not
satisfy the version-frozen unfreeze checklist or complete RA matrix. This
package therefore remains `FROZEN`, and no checkbox below is implicitly
completed.

---

## 1. What may be completed before unfreezing

H-IDE development may finish independently using frozen contracts and protocol-conformant simulators:

- H-IDE Vision production gRPC ingress and deterministic RGB-D replay/synthetic fixture sender.
- Visual target inference, deterministic depth/TF geometry, immutable `VisualResolution`, and safety gates.
- H-IDE Agent Supervisor and vision-worker integration.
- Reviewable `VisualMissionPlanDraft` and durable `VisualMissionRuntime`.
- Production H-IDE Robot Agent Gateway client/session/driver.
- A protocol-level Robot Agent simulator exercising lease, command, feedback, query, cancel, reconnect, idempotency, and recovery behavior.
- PWA review, evidence, progress, cancel, stop, and simulated-state presentation.

The simulator must identify itself as synthetic, cannot advertise physical movement, and cannot enable `VISUAL_MISSION_MODE=robot_execute`.

The following remain unverified until this package is unfrozen:

- Real Robot Observation Gateway interoperability.
- Live synchronized/aligned D435i RGB-D transport.
- Real Robot Agent control lease and command lifecycle.
- `navigation.navigate_to_pose` mapping to `nav2_msgs/action/NavigateToPose`.
- Mock Nav2 behavior behind the real Robot Agent.
- Robot-side watchdog, cancel, stop, reconnect, and restart behavior.
- End-to-end timing through the real processes.
- Physical motion and physical post-arrival visual verification.

---

## 2. Ownership boundary

| Boundary | H-IDE team | Robot Agent owner |
|---|---|---|
| Observation protocol | Own frozen proto, lock hash, fixtures, conformance runner, Vision ingress | Generate bindings and implement compatible Gateway |
| Vision | Own inference, geometry, evidence, target-resolution API | Supply synchronized observations only |
| Navigation request | Own reviewed target pose, ActionRuntime, gateway protocol client | Validate command and delegate pose to Nav2 |
| Path and motion | Must not generate path, velocity, or ROS action details | Own Nav2 path planning, avoidance, execution, watchdog and local safety |
| Repository changes | Normally must not modify `robotagent`; the 2026-08-17 pose handoff is the explicitly authorized narrow exception and must be owner-reviewed | Applies and reviews all Robot Agent changes |
| Acceptance evidence | Own H-IDE traces, resolution and command audit | Own Robot Agent/ROS/Nav2 logs and test results |

---

## 3. Unfreeze readiness gate

Do not schedule integration until the Robot Agent owner supplies all of the following:

- [ ] Cleanly identified Robot Agent commit SHA and build instructions.
- [ ] Frozen observation descriptor hash exactly matching H-IDE Vision `PROTOCOL.lock`.
- [ ] Passing Robot Agent observation protocol conformance report.
- [ ] `source_mode`, immutable fixture/live sensor identity, `robot_boot_id`, and `localization_epoch_id` support.
- [ ] Live Gateway health distinguishes USB, RGB, depth, synchronized RGB-D, calibration, TF, localization, and observation usability.
- [ ] `navigation.navigate_to_pose` capability with the agreed strict payload schema.
- [ ] Real `nav2_msgs/action/NavigateToPose` mapping; no Fibonacci acceptance substitute.
- [ ] Protocol-level query and cancel for active navigation.
- [ ] Normalized feedback and terminal statuses with Robot Agent command ID and ROS goal ID.
- [ ] Control lease, idempotency, watchdog, safe-stop, reconnect, and restart behavior documented.
- [ ] Mock Nav2 based on the real NavigateToPose action contract.
- [ ] A lab `map -> base_link -> camera` TF/localization setup with a defined localization epoch.
- [ ] Named Robot Agent owner and a shared integration window.

H-IDE must also supply:

- [ ] Passing P1–P8 results from the main implementation plan.
- [ ] H-IDE, H-IDE Agent, and H-IDE Vision commit SHAs.
- [ ] Observation proto/descriptor/lock and navigation schema artifacts.
- [ ] Passing protocol-simulator integration report.
- [ ] Deterministic fixture manifest and hashes.
- [ ] Valid Qwen3-VL credential supplied through secret management, not committed files.
- [ ] Locked GPU/model/Prompt/ApproachPolicy versions.
- [ ] `lab_execute` enabled and `robot_execute` disabled for the first integration run.

If any box is unchecked, keep this package frozen. A partial implementation may be inspected, but it cannot generate acceptance evidence.

---

## 4. Version manifest to fill at unfreeze

```text
integration_run_id:
date/time/timezone:
operator:
reviewer:

h_ide_commit:
h_ide_agent_commit:
h_ide_vision_commit:
robotagent_commit:

observation_protocol_major:
observation_protocol_minor:
observation_schema_sha256:
navigation_schema_sha256:

robot_id:
robot_boot_id:
gateway_instance_id:
sensor_id_and_serial:
localization_epoch_id:
scene_id_revision_or_none:

python_pytorch_cuda_versions:
qwen_provider_model:
sam2_model_hash:
foundationpose_code_and_weight_hashes:
prompt_version:
visual_algorithm_version:
approach_policy_hash:
```

Any change to a listed value invalidates prior results affected by that boundary and requires a new run ID.

---

## 5. Frozen test sequence

Execute in order. Stop at the first failed gate, retain evidence, and do not weaken a threshold merely to continue.

### RA-0 — Protect both worktrees

1. Record `git status --short` and commit SHA in all four repositories.
2. Confirm no integration script writes into another repository.
3. Confirm secrets, camera frames, evidence, and certificates are ignored by Git.
4. Confirm the H-IDE release label is still `code_complete_robot_agent_pending`.

**Pass:** Reproducible versions are recorded and no user-owned change is overwritten.

### RA-1 — Robot Agent owner test suite

Run from the Robot Agent worktree after its owner has reviewed the command:

```bash
python -m pytest \
  tests/test_observation_core.py \
  tests/test_observation_grpc_integration.py \
  tests/test_observation_mtls.py \
  tests/test_observation_ros_bridge.py \
  tests/test_agent_observation.py \
  tests/test_ros_integration.py -q
```

Also run the Robot Agent owner's ROS 2 build/test command for the real observation and navigation packages.

**Pass:** All selected tests and ROS builds pass at the recorded commit. Synthetic-only results do not satisfy later live gates.

### RA-2 — Observation protocol conformance

1. Run valid, malformed, version-mismatch, hash-mismatch, CRC-failure, stale, out-of-order, calibration-change, boot-change, and localization-epoch-change fixtures against the real Gateway.
2. Verify mTLS identity is bound to the declared robot/gateway identity.
3. Verify newest-only flow control, bounded memory, ACK/reject reasons, rate hint, keepalive, and reconnect.
4. Verify no observation RPC can issue a robot command.

**Pass:** Valid fixtures interoperate with the production H-IDE Vision ingress; every incompatible or unsafe fixture fails closed with a specific reason.

### RA-3 — Live D435i observation

1. Start the RealSense ROS driver as the sole D435i USB owner.
2. Verify synchronized color, aligned Z16 depth, CameraInfo, depth scale, and capture timestamps.
3. Start the real Robot Observation Gateway and connect it to H-IDE Vision.
4. Verify the health ladder and record at least one bounded diagnostic sample.
5. Disconnect/reconnect the camera and Vision endpoint; verify safe recovery and a new session identity where required.

**Pass:** At least 80% of requested frames are usable, RGB/depth delta is at most 30 ms, TF delta is at most 50 ms, queues remain bounded, and a three-frame static target is stable in `map` within the V1 threshold.

### RA-4 — Navigation contract conformance

1. Connect the production H-IDE Robot Agent Gateway to the real local Robot Agent.
2. Acquire and renew one control lease.
3. Submit `navigation.navigate_to_pose` with a reviewed `map` pose and all audit bindings.
4. Verify normalized accepted/active/terminal feedback, Robot Agent command ID, ROS goal ID, query reconciliation, and TTL handling.
5. Verify `command.cancel` cancels the active NavigateToPose goal.
6. Verify duplicate idempotency key produces no duplicate ROS goal.
7. Verify unsupported command, expired request, wrong boot/epoch/session, and lost lease produce zero navigation dispatches.

**Pass:** One H-IDE command maps to at most one real NavigateToPose goal, and all rejection/cancel outcomes are observable and auditable.

### RA-5 — End-to-end visual mission with Mock Nav2

1. Run real D435i, real Observation Gateway, H-IDE Vision, H-IDE Agent, H-IDE services, production H-IDE Robot Agent Gateway, real local Robot Agent, and Mock Nav2.
2. Submit a V1 static visible-target instruction by text, then repeat once through voice.
3. Review and confirm the VisualMission PlanDraft.
4. Verify one immutable VisualResolution is reread and revalidated by Control API.
5. Verify exactly one `navigation.navigate_to_pose` child command reaches Robot Agent/Mock Nav2.
6. Verify progress and terminal state propagate to the PWA.
7. Keep the UI label `运动为模拟`; do not claim physical arrival from Mock Nav2.

**Pass:** One trace ID joins TaskRequest, draft, mission, resolution, child command, Robot Agent command, ROS goal, and terminal receipt. No `FakeAdapter` or protocol simulator is present in the chain.

### RA-6 — Safety and recovery matrix

For each case below, record expected and actual state plus dispatch count:

| Case | Required result |
|---|---|
| Stale or desynchronized RGB-D | Resolution rejected; zero dispatches |
| Missing/old TF or high localization covariance | Resolution non-executable; zero dispatches |
| Localization epoch or robot boot changes | Mission pauses/fails; old goal not replayed |
| Ambiguous target or insufficient valid depth | Needs input/rejected; zero dispatches |
| Scene, calibration, model, Prompt, algorithm, or policy drift | Confirmation/resolution invalidated |
| Duplicate confirmation/request | At most one effective command and ROS goal |
| Vision or Agent process dies before dispatch | Mission recoverable/failed; zero dispatches |
| Control API dies while command may be active | State becomes `unknown`; no automatic replay |
| Gateway/Robot Agent disconnects while active | Query/reconciliation required; no automatic replay |
| User cancels | Current Nav2 goal cancelled and no next stage starts |
| User stops/E-stop | Highest-priority stop remains available independent of vision |

**Pass:** Every unsafe case fails closed and produces the required audit evidence.

### RA-7 — Accuracy and performance

Run each accepted scenario at least ten times unless the integration owner records a stronger sample plan:

- Registered CAD object: position error <= 10 cm, orientation error <= 15 degrees.
- Generic static chair/table: position error <= 30 cm.
- Deterministic target fixture: map-coordinate error <= 5 cm.
- Mock Nav2 reported final pose: error <= 25 cm and visibly labeled simulated.
- Observation request to first usable RGB-D: <= 2 s.
- Visual resolution P95: <= 15 s.
- Post-arrival verification: <= 8 s where the camera movement is physically meaningful.
- UI state propagation: <= 500 ms.
- Observation Gateway: sustained 2 FPS baseline/5 FPS burst without unbounded queues.

Record raw measurements, sample count, median, P95, maximum, failures, and test geometry. Do not report only averages.

### RA-8 — Physical robot validation (separate authorization)

This is not authorized merely by unfreezing RA-0 through RA-7. It requires an on-site operator, cleared low-speed area, physical E-stop, reviewed robot allowlist, and explicit `robot_execute` enablement.

Progress in order: observe-only, generate-without-execute, short motion, cancel, network loss, E-stop, then one bounded static-target mission. A Mock Nav2 result cannot satisfy this gate.

---

## 6. Evidence bundle

Store an immutable manifest containing:

```text
validation/<integration_run_id>/
  manifest.json
  versions.txt
  test-results/
  protocol-conformance/
  ros-logs/
  gateway-logs/
  traces/
  metrics/
  visual-resolutions/
  command-receipts/
  screenshots/
  hashes.sha256
  result.md
```

Do not commit raw secrets or unrestricted camera recordings. Evidence references must obey the configured retention and privacy policy.

---

## 7. Result record

```text
run_id:
status: PASS | FAIL | BLOCKED
highest_completed_gate: RA-0 .. RA-8
failed_or_blocked_gate:
reason:
robot_dispatch_count_for_rejection_tests:
open_issues:
evidence_manifest_sha256:
h_ide_reviewer:
robot_agent_reviewer:
signed_at:
```

Release-state rules:

- P1–P8 complete, this package not passed: `code_complete_robot_agent_pending`.
- RA-0 through RA-7 passed: `robot_agent_integrated` and Mock Nav2 behavior remains labeled simulated.
- RA-8 passed with retained evidence: `physical_robot_validated` for the exact recorded hardware/software configuration only.
- Any material protocol, safety, localization, calibration, model, algorithm, or robot configuration change returns the affected boundary to pending.

---

## 8. How to unfreeze

1. Robot Agent owner announces readiness with commit SHA and completed readiness checklist.
2. H-IDE owner verifies all P1–P8 artifacts and creates an integration run ID.
3. Both owners review contract hashes and commands; neither side edits the other repository during the run.
4. Change this document's status from `FROZEN` to `ACTIVE`, record the reviewers, and execute RA-0 onward.
5. On failure, record evidence and return to `FROZEN` if either project must change its contract or implementation.
6. Change release state only after the applicable gate has a complete evidence bundle and both owners approve it.
