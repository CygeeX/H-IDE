# H-IDE VLN Visual Target Resolution V1 Implementation Plan

> **Execution rule:** Implement in the phase order below and stop at every exit gate. Preserve all pre-existing uncommitted work. Do not edit `/home/phi/learn-projects/robotagent`; deliver contracts, conformance tests, reproducible findings, and requested changes to its owner instead.

**Goal:** Add a visual target-resolution capability that turns natural-language spatial intent plus synchronized head-camera RGB-D into one safe `map`-frame target pose at a time, keeps PCD scene support, requires a reviewable `PlanDraft`, and delegates path planning, obstacle avoidance, and motion to Robot Agent/Nav2.

**Current code-complete outcome:** A protocol-conformant observation sender streams deterministic RGB-D fixtures through the real `h-ide-vision` gRPC ingress. The Qwen3-VL adapter and production SAM 2 provider are implemented; SAM 2 has passed a real GPU smoke test, while live Qwen remains unavailable until a DashScope secret is injected. FoundationPose remains disabled because the local checkpoints are Git LFS pointers and PyTorch3D is unavailable. Deterministic geometry creates a safe approach pose, H-IDE presents a visual mission for human review, and the production H-IDE Robot Agent Gateway path dispatches that pose to a protocol-conformant Robot Agent simulator. This milestone is labeled `code_complete_robot_agent_pending`; it is not real Robot Agent, Nav2, or physical-motion acceptance.

**Deferred integrated outcome:** After the independently developed Robot Agent declares its observation and navigation contracts ready, run the frozen validation package in [`docs/integration/robot-agent-deferred-validation.md`](../../integration/robot-agent-deferred-validation.md). Only that evidence can advance the system to `robot_agent_integrated`; physical robot validation remains a separate later milestone.

**Tech stack:** Python 3.11, FastAPI, `grpc.aio`, protobuf, PostgreSQL, Redis, local content-addressed evidence storage, PyTorch with a Blackwell-capable CUDA build, SAM 2, FoundationPose, Qwen3-VL API, Mastra/TypeScript, existing FastAPI/Pydantic/SQLite H-IDE control plane, React/Vite PWA, Robot Agent WebSocket protocol, ROS 2/Nav2.

---

## 1. Locked architecture and safety boundary

```text
PCD / text / voice
        |
        v
unified TaskRequest
        |
        v
H-IDE Agent (Mastra Supervisor)
        |
        +--> motion/manipulation workers
        |
        +--> vision-worker --> h-ide-vision
                                ^
                                |
                   reverse gRPC+mTLS RGB-D
                                |
      contract fixture sender (current development)
           or Robot Observation Gateway (deferred)
                                |
                 fixture/replay or RealSense ROS driver

Supervisor output
        |
        v
Action PlanDraft OR VisualMission PlanDraft
        |
        v
human review
        |
        v
VisualMissionRuntime (deterministic state machine)
        |
        +--> one immutable VisualResolution per stage
        |
        +--> one navigation.navigate_to_pose child command
                                |
                                v
ActionRuntime --> H-IDE Robot Agent Gateway --> protocol simulator (current)
                                           `-> real Robot Agent -> Nav2 (deferred)
```

The following rules are non-negotiable:

- H-IDE and `h-ide-vision` never calculate or transmit a path or velocity sequence. They produce one `PoseStamped`-equivalent goal (`x/y/z/yaw`, `frame=map`) at a time.
- Only Robot Agent/Nav2 plans a path, avoids obstacles, and executes motion.
- The VLM identifies and relates objects; it cannot produce trusted coordinates, safety distances, command names, or executable payloads.
- Three-dimensional coordinates come only from synchronized RGB/aligned depth, camera calibration, capture-time TF, localization, and deterministic geometry.
- `h-ide-agent` is the only caller allowed to create a visual resolution. Control API may read/cancel the canonical resolution but cannot invent or edit it.
- Every robot motion remains behind immutable `PlanDraft` review and `ActionRuntime`. Browser, Agent, video, scene, and vision services have no direct ROS or robot-control credentials.
- PCD remains optional. `scene-only`, `vision-only`, and `fused` modes are supported; PCD is never uploaded to the VLM.
- A visual failure may disable a visual mission, but must not block manual stop/cancel, ordinary manual commands, PCD workflows, or non-visual planning.
- Simulators must report `source_mode=synthetic|replay`, use unmistakable simulator identities, and can never enable `robot_execute` or produce a real-integration/physical-success receipt.

### V1 scope

- A static target visible from the current stopped camera pose.
- One final or approach target pose per mission stage.
- Registered CAD object via FoundationPose.
- Generic visible object via Qwen3-VL rough grounding followed by SAM 2 mask refinement.
- Optional PCD region/free-space cross-check.
- Frozen Observation Gateway and navigation contracts, conformance fixtures, and owner handoff artifacts.
- Protocol-conformant observation/control simulators exercising the production ingress and H-IDE gateway drivers; `FakeAdapter` is allowed only below that boundary for unit tests.
- Evidence snapshots at 2–5 FPS when observing; continuous H.264/WebRTC preview is deferred.

### Explicitly deferred to V1.1

- `OBSERVE` exploration goals when the target is outside the current view.
- Multi-stage instructions such as “pass the water dispenser, turn right, enter the second room”.
- Continuous moving-camera VLN and video tracking while the base moves.
- MediaMTX/WebRTC browser video.
- Dynamic targets such as people, pets, and moving carts.
- Head/neck actuation, manipulation, elevators, automatic doors, and cross-floor missions.
- Real D435i through the independently developed Robot Observation Gateway, real local Robot Agent, and Mock/real Nav2 integration; these are frozen in the deferred validation package rather than blocking current code completion.

---

## 2. Current baseline and known integration gaps

Treat these as preflight facts, not as implementation assumptions:

| Area | Current state | Required resolution |
|---|---|---|
| H-IDE | Existing control plane, PCD workflow, Mastra planning client, immutable action drafts, ActionRuntime, and chat PWA; worktree contains substantial uncommitted user work | Make narrow additive edits and never reformat unrelated files |
| H-IDE Agent | Independent Mastra project exists but its whole initial implementation is currently uncommitted | Extend its real contracts and worker routing without relocating it into H-IDE |
| H-IDE Vision | `/home/phi/learn-projects/h-ide-vision` does not exist | Create a new independent private Git repository |
| Robot Observation Gateway | Robot Agent worktree now contains a substantial uncommitted candidate Gateway, reverse gRPC protocol, synthetic source, ROS bridge, mTLS tests, and systemd files | Reconcile and freeze the candidate protocol; do not write into this repository |
| Robot navigation command | H-IDE already uses `navigation.navigate_to_pose`; Robot Agent candidate currently declares `navigation.vln.execute` backed by a Fibonacci mock action | Robot Agent owner must expose `navigation.navigate_to_pose` backed by Nav2 `NavigateToPose`; H-IDE operation `navigation.cancel` maps to protocol-level `command.cancel` |
| RealSense | D435i is visible over USB; color frames are live; aligned depth and `/camera/camera/rgbd` produced no frame during the latest sample | Fix RealSense RGB-D publication before live 3-D acceptance |
| Global coordinates | Current local ROS graph has no `map` frame | Add a measured `lab_test_map -> base_link -> camera` chain and a localization epoch fixture for lab execution |
| GPU | RTX 5060 Ti 16 GB and driver 595.71.05 are available; Docker has no NVIDIA runtime configured | Run V1 GPU Worker in a host Python 3.11 environment and complete a Blackwell compatibility spike before API integration |
| VLM credentials | No Qwen/DashScope credential is available in the current process | Use FakeVlmProvider for development; mark real Qwen capability unavailable until a secret is injected |
| FoundationPose source | Local snapshot exists but has no Git metadata; both `.pth` files are Git LFS pointer text rather than model data, and PyTorch3D is unavailable | Keep the capability disabled until the owner supplies real checkpoints and a compatible environment; then repeat the GPU spike and register a real box/CAD model for acceptance |

Known local FoundationPose artifact hashes:

```text
d0a0528ec6f5b2411a294780b5ef76bd271dc37d654abb6b928ab7ab6859b06b  weights/2023-10-28-18-33-37/model_best.pth
8537f5892d0d42b3357847a0d51fcf6ed4724472c336fb895b829c13106eda46  weights/2024-01-11-20-02-45/model_best.pth
485267ed87d8784b1fceb77fef2af2ecd7cc35ac04d31056cda72bf7d94e682a  cylinder.obj
```

---

## 3. Authoritative contracts

### 3.1 Robot observation protocol

The authoritative source will be:

```text
/home/phi/learn-projects/h-ide-vision/contracts/robot_observation/v1/observation.proto
/home/phi/learn-projects/h-ide-vision/contracts/robot_observation/PROTOCOL.lock
```

Start from the current Robot Agent candidate proto, review it once, then freeze it. The frozen V1 contract must include:

- `major`, `minor`, and runtime `schema_sha256` negotiation.
- `robot_id`, `robot_boot_id`, `gateway_instance_id`, `sensor_id`, and `observation_session_id`.
- `localization_epoch_id` so relocalization invalidates old goals.
- `source_mode = live|replay|synthetic` plus a serial number or immutable fixture hash.
- Optional scene reference; PCD/scene binding must not be mandatory for `vision-only` mode.
- JPEG RGB and Zstd-compressed little-endian Z16 aligned depth.
- RGB/depth frame numbers and sensor timestamps, capture time, clock domain, CRC32C, dimensions, intrinsics, depth scale, `calibration_sha256`, capture-time `T_map_camera`, `T_base_camera`, localization timestamp, and 36-value covariance.
- Newest-only flow control, per-frame ACK/reject reason, rate hint, keepalive, session start/stop, telemetry, and bounded message size.
- No robot command RPC of any kind.

Generated bindings are build artifacts. CI regenerates them and fails if the descriptor-set hash does not equal `PROTOCOL.lock`. The Robot Agent handoff bundle contains the proto, descriptor set, hash, generated language bindings or generation command, and conformance fixtures.

### 3.2 Visual mission contract

Add a discriminated draft union while retaining legacy action drafts:

```text
PlanDraft = ActionPlanDraft | VisualMissionPlanDraft

ActionPlanDraft:
  planType: action
  steps: existing immutable PlanStep DAG

VisualMissionPlanDraft:
  planType: visual_mission
  mission: immutable VisualMissionSpec
```

Legacy stored plans without `planType` migrate/read as `action`. `VisualMissionSpec` must pin:

- Robot, instruction, locale, requested source/fusion mode, target description, goal purpose.
- Ordered finite stages (`LOCATE`, `APPROACH`, `VERIFY`, `COMPLETE` in V1; `OBSERVE` enabled in V1.1).
- Same-map/same-floor constraint, maximum 6 stages, 10-minute duration, 5 m per goal, and 20 m cumulative travel.
- Scene ID/revision when used, `robotBootId`, `localizationEpochId`, and allowed sensor IDs.
- Allowed target categories and explicit unsupported capabilities.
- VLM provider/model, prompt version, visual algorithm version, ApproachPolicy version/hash, and review expiry.

### 3.3 Immutable visual resolution

`VisualResolution` is canonical authority owned by `h-ide-vision`. It includes:

- `visualResolutionId`, `missionId`, `stageId`, `attempt`, `traceId`.
- Robot/source/session/bundle identity and capture time.
- Target phrase, VLM candidate IDs, selected region, SAM 2 mask hash, optional FoundationPose object registration/version.
- `landmarkPose` and separately derived `goalPose`, both with frames and timestamps.
- `goalPurpose = observe|approach|final`.
- Deterministic quality metrics, every safety gate and its measured value, ambiguity candidates, warnings, and rejection reason.
- Pinned calibration, localization epoch, scene, model, prompt, algorithm, and ApproachPolicy versions.
- `executable` boolean calculated only by deterministic code.

One resolution may be bound to at most one effective command dispatch. User retry creates a new attempt and new resolution; no stored resolution is edited in place.

### 3.4 Robot navigation contract

H-IDE logical command:

```text
navigation.navigate_to_pose
```

Required payload contains a `map`-frame goal pose and audit/safety bindings:

- `goalPose {frameId, x, y, z, yaw}`.
- `goalPurpose`, `missionId`, `stageId`, `visualResolutionId`.
- `observationSessionId`, `robotBootId`, `localizationEpochId`.
- Optional immutable scene reference.
- Maximum speed and expiry constrained by active mapping and Robot Agent local configuration.

Robot Agent must map it to `nav2_msgs/action/NavigateToPose` (or the reviewed production navigation action), publish normalized feedback, and use its existing protocol-level command cancellation for `navigation.cancel`. A Fibonacci action is never an acceptance substitute.

---

## 4. Phase plan and exit gates

### Phase 0 — Protect the baseline and establish ownership

**Repositories:** `h-ide`, `h-ide-agent`, `robotagent` (read-only)

**Files:**

- Add: `docs/architecture/vln-visual-target-resolution.md`
- Add: `docs/integration/robot-agent-visual-handoff.md`
- Add later in the new repository: `docs/decisions/*.md`

**Work:**

1. Capture `git status`, current test commands, service ports, and schema hashes before every phase.
2. Treat all existing modifications as user-owned. Never clean, reset, overwrite, or commit unrelated changes.
3. Record ownership:
   - `h-ide`: control/review/runtime/UI.
   - `h-ide-agent`: Supervisor and vision-worker reasoning.
   - `h-ide-vision`: observation ingress, visual inference, geometry, evidence.
   - `robotagent`: robot-side Gateway, navigation adapter, local safety; maintained by its other owner.
4. Record feature levels: `off`, `observe`, `lab_execute`, `robot_execute`; default is `observe`.

**Exit gate P0:** Every downstream file has one owning repository, and no task requires direct edits to Robot Agent.

### Phase 1 — Create `h-ide-vision` and freeze contracts

**Repository:** create `/home/phi/learn-projects/h-ide-vision`

**Initial files:**

```text
h-ide-vision/
  pyproject.toml
  README.md
  .env.example
  .gitignore
  contracts/robot_observation/v1/observation.proto
  contracts/robot_observation/PROTOCOL.lock
  contracts/visual_resolution.v1.schema.json
  scripts/generate_proto.sh
  scripts/check_contract_lock.py
  src/h_ide_vision/contracts/
  tests/contracts/
  docs/architecture.md
  docs/third-party-notices.md
```

**Work:**

1. Initialize an independent private Git repository; use Python 3.11.
2. Copy the current candidate observation proto for review only, make the V1 additions from section 3.1, generate a deterministic descriptor set, and create `PROTOCOL.lock`.
3. Define strict Pydantic contracts for Observation identity, VisualMissionSpec, VisualResolution, evidence, ambiguity, safety gates, and ApproachPolicy.
4. Produce malformed/valid binary protocol fixtures and cross-version rejection fixtures.
5. Export a read-only handoff archive for the Robot Agent owner; do not copy generated files back into their worktree.

**Necessary tests:**

- Proto regeneration is clean and deterministic.
- Major/minor/hash mismatch rejects the stream before an observation session starts.
- Unknown enum values and missing identity/calibration/localization data fail closed.
- PCD-optional `vision-only` session validates without a fabricated scene version.

**Exit gate P1:** H-IDE Vision freezes and tests one deterministic V1 descriptor hash, exports the owner handoff bundle, and refuses every stream that does not report that exact hash. Robot Agent owner approval is required only to unfreeze P9.

### Phase 2 — Complete the RTX 5060 Ti GPU compatibility spike

**Repository:** `h-ide-vision`

**Files:**

```text
scripts/gpu_smoke.py
scripts/sam2_smoke.py
scripts/foundationpose_smoke.py
requirements/api.lock
requirements/gpu.lock
docs/gpu-compatibility.md
third_party/foundationpose/LICENSE
third_party/foundationpose/NOTICE.md
third_party/foundationpose/MANIFEST.sha256
```

**Work:**

1. Use a host Python 3.11 GPU environment for V1. Do not add CUDA to the H-IDE control environment.
2. Select a currently supported PyTorch/CUDA build with native Blackwell support (CUDA 12.8+); do not copy the old `cu124` instruction blindly.
3. Verify CUDA tensor execution, device capability, and deterministic memory cleanup.
4. Run SAM 2 box-prompt segmentation on one image.
5. Build and run the required FoundationPose CUDA extensions (`nvdiffrast`, PyTorch3D, and only the inference dependencies actually imported).
6. Run one FoundationPose sample with the local snapshot and weights.
7. Load SAM 2 and FoundationPose sequentially under the 16 GB limit; GPU concurrency remains one.
8. Copy only the fixed inference subset later needed by `KnownObjectPoseAdapter`; retain NVIDIA notices. Because the source snapshot lacks Git metadata, hash every imported source file and weight.

**Necessary tests:**

- `gpu_smoke.py` exits zero and reports the expected RTX 5060 Ti.
- SAM 2 returns a non-empty mask from a fixed box prompt.
- FoundationPose returns a finite 6-DoF pose for its sample.
- A second sequential inference succeeds after releasing the first model's memory.

**Exit gate P2:** Exact Python, PyTorch, CUDA, extension, SAM 2, and FoundationPose versions are locked. If FoundationPose fails, continue the generic-object V1 path but report `foundationpose=unavailable`; do not hide the failed capability.

### Phase 3 — Build the fixture-based `h-ide-vision` closed loop

**Repository:** `h-ide-vision`

**Core files:**

```text
src/h_ide_vision/api/main.py
src/h_ide_vision/api/routes/resolutions.py
src/h_ide_vision/api/routes/capabilities.py
src/h_ide_vision/ingress/server.py
src/h_ide_vision/ingress/session.py
src/h_ide_vision/ingress/frame_store.py
src/h_ide_vision/jobs/queue.py
src/h_ide_vision/jobs/worker.py
src/h_ide_vision/persistence/models.py
src/h_ide_vision/persistence/repository.py
src/h_ide_vision/evidence/storage.py
src/h_ide_vision/providers/vlm/base.py
src/h_ide_vision/providers/vlm/fake.py
src/h_ide_vision/providers/vlm/qwen.py
src/h_ide_vision/providers/segmentation/sam2.py
src/h_ide_vision/providers/pose/foundationpose.py
src/h_ide_vision/geometry/deprojection.py
src/h_ide_vision/geometry/transforms.py
src/h_ide_vision/geometry/mask_depth.py
src/h_ide_vision/geometry/approach.py
src/h_ide_vision/policies/approach.py
src/h_ide_vision/runtime/resolution_service.py
infra/docker-compose.yml
infra/Dockerfile.api
infra/Dockerfile.ingress
```

**Work:**

1. Run FastAPI, `grpc.aio` ingress, and GPU Worker as separate processes. Containerize API/Ingress/PostgreSQL/Redis; keep the GPU Worker on the host for V1.
2. Implement reverse gRPC session negotiation, mTLS identity binding, loopback-only plaintext development mode, keepalive, bounded latest-frame storage, ACK/reject, and backpressure. Never queue historical video frames.
3. Accept frames only when CRC, dimensions, encoding, identity, session, calibration, clock, sequence, and localization checks pass.
4. Implement the asynchronous API:
   - `POST /internal/v1/visual-resolutions`
   - `GET /internal/v1/visual-resolutions/{id}`
   - `POST /internal/v1/visual-resolutions/{id}/cancel`
   - `GET /internal/v1/capabilities`
5. Enforce separate scoped credentials: Agent can create; Control API can read/cancel; browser has no direct access.
6. Implement `FakeVlmProvider` and deterministic fixture results first. Implement `QwenVlmProvider` with strict JSON Schema, one formatting repair attempt, cropped RGB keyframes only, and no depth/PCD/robot ID/map coordinates in cloud requests.
7. Implement VLM rough box/point -> SAM 2 mask -> masked depth -> robust 3-D landmark. Remove invalid depth and robustly reject outliers/background; never use the raw box center as a goal.
8. Implement registered-object FoundationPose as an optional `KnownObjectPoseAdapter`, not as the universal VLN engine.
9. Implement versioned ApproachPolicy. Produce a landmark pose and separate candidate approach poses; filter by ground support, robot radius, clearance, class-specific distance, and requested yaw. This is geometric goal selection, not path planning.
10. Persist immutable metadata in PostgreSQL, use Redis only for jobs, and store evidence by SHA-256 in a bounded local volume.
11. Apply retention: raw stream in memory only, keyframe/mask/overlay 24 hours, numeric resolution metadata 30 days, command/safety audit according to H-IDE policy.

**Deterministic execution gates:**

- Frame age at binding <= 1 s.
- RGB/depth timestamp delta <= 30 ms.
- TF delta from capture <= 50 ms.
- Cross-machine clock uncertainty <= 100 ms when cross-machine age is used.
- Valid calibration, positive depth scale, finite transforms, and acceptable localization covariance.
- Target mask valid-depth ratio >= 60%.
- At least three fresh frames within 2 seconds; `map` landmark cluster radius <= 0.25 m.
- Registered FoundationPose orientation spread <= 15 degrees.
- Target and candidate goal are not within deterministic local obstacles; optional PCD check occurs again in H-IDE.
- VLM self-reported confidence is evidence only and never satisfies a gate.

**Necessary tests:**

- Unit: depth deprojection, quaternion/TF composition, robust mask statistics, candidate goal/yaw generation.
- Contract: bad schema, stale frame, bad CRC, out-of-order frame, identity mismatch, localization epoch mismatch.
- State: idempotent create, cancel, crash recovery, one resolution-to-dispatch binding.
- Fixture E2E: synthetic known coordinates resolve within 5 cm and every invalid evidence case produces zero executable resolution.
- Provider: Fake VLM always available; Qwen capability is explicitly unavailable without its secret.

**Exit gate P3:** A deterministic protobuf fixture travels through real gRPC ingress, async job processing, fake semantic grounding, depth/TF geometry, and persistence to an immutable executable VisualResolution. All fail-closed cases are proven.

### Phase 4 — Build the observation conformance simulator and freeze the live handoff

**Repositories:** `h-ide-vision`; `robotagent` read-only for compatibility inspection only

**H-IDE Vision files:**

```text
tools/fixture_recorder.py
tools/observation_diagnostics.py
tools/observation_fixture_sender.py
tests/fixtures/manifests/
tests/integration/test_observation_fixture_sender.py
docs/realsense-lab-setup.md
```

**Current implementation work:**

1. Build a contract-level fixture sender that connects to the production `grpc.aio` ingress and implements the frozen reverse-stream handshake, frame flow control, ACK, keepalive, reconnect, and session-stop behavior.
2. Support only `source_mode=synthetic|replay`; bind every session to an immutable fixture manifest/hash and use an obvious simulator identity. It must be impossible to configure this tool as `live`.
3. Record deterministic synchronized RGB/aligned-depth/calibration/TF/localization bundles, plus stale, corrupt, out-of-order, relocalized, and protocol-mismatch variants.
4. Exercise the same ingress/session/frame store used by a future real Gateway—do not add a fixture-only ingestion API or bypass mTLS/schema negotiation in integration tests.
5. Keep `tools/observation_diagnostics.py` able to inspect a future live stream without changing or controlling Robot Agent.
6. Write the Robot Agent handoff requests below into the frozen integration package; do not wait for or edit the other repository.

**Deferred Robot Agent owner requests:**

1. Consume the frozen observation descriptor/hash and regenerate its bindings.
2. Keep RealSense ROS driver as the sole USB owner; Gateway subscribes to synchronized ROS RGB-D and never opens D435i through `pyrealsense2` a second time.
3. Make `/camera/camera/rgbd` publish synchronized color, aligned depth, and CameraInfo. Fix the current color-only state before integration.
4. Add the accepted `source_mode`, `localization_epoch_id`, optional scene binding, and schema-hash fields.
5. Preserve the current capacity-one pipeline, explicit drop metrics, ACK watchdog, mTLS, and independent process/systemd boundary.
6. Publish the real Gateway health ladder: USB, RGB, depth, synchronized RGB-D, calibration, TF, localization, observation usable.

**Deferred lab setup:**

1. Rigidly mount the D435i on a tripod or measured mock robot head. Handheld mode is preview-only and cannot create executable coordinates.
2. Define `lab_test_map -> base_link -> camera_link -> camera_color_optical_frame` with measured transforms.
3. Publish a deterministic local localization pose/covariance plus `localizationEpochId`; AprilTag may be used as a lab measurement aid, not as production localization.
4. Confirm RGB-D >= 80% of requested rate, matching dimensions, valid depth scale/intrinsics, and timestamp delta <= 30 ms.
5. Record both `rosbag2` source data and normalized protobuf fixtures with manifests and SHA-256.
6. Exercise `sourceMode=live|replay|synthetic`; a session cannot change sources in place.

**Current necessary tests:**

- Fixture sender connects to the production H-IDE Vision ingress using loopback development security and test mTLS.
- Valid replay/synthetic RGB-D produces accepted ACKs and usable simulated observation health.
- Disconnect/reconnect, bad schema hash, stale frame, missing map TF, corrupt depth, and localization epoch change all stop or reject the session safely.
- A deterministic known target produces camera-frame and map-frame coordinates within 5 cm.
- No fixture/simulator receipt can be serialized as `source_mode=live` or unlock `robot_execute`.

**Exit gate P4:** A protocol-conformant replay/synthetic sender reaches the production `h-ide-vision` ingress, produces a three-frame stable map target, and proves all fail-closed protocol cases. The real D435i/Gateway checks are present in the frozen validation document and do not block current implementation.

### Phase 5 — Add H-IDE visual drafts, mission runtime, and Robot Agent gateway

**Repository:** `h-ide`

**Files:**

```text
packages/control_core/models.py
packages/control_core/assistant_models.py
packages/control_core/validation.py
packages/control_core/runtime.py
packages/control_core/adapters.py
packages/visual_mission_core/__init__.py
packages/visual_mission_core/models.py
packages/visual_mission_core/runtime.py
packages/visual_mission_core/validation.py
services/control_api/assistant_service.py
services/control_api/main.py
services/control_api/store.py
services/control_api/vision_client.py
services/control_api/robot_agent_client.py
services/robot_agent_gateway/main.py
services/robot_agent_gateway/session.py
services/robot_agent_gateway/driver.py
infra/Dockerfile.robot-agent-gateway
infra/docker-compose.yml
.env.example
tests/test_visual_mission_contracts.py
tests/test_visual_mission_runtime.py
tests/test_vision_client.py
tests/test_robot_agent_gateway.py
tests/test_visual_mission_api.py
```

**Work:**

1. Introduce the discriminated PlanDraft union from section 3.2. Preserve action-plan JSON and migrate legacy rows with `planType=action` on read/write.
2. Add durable tables for visual missions, stages, attempts, resolution references, child commands, mission events, and review constraints. Use CAS state changes:

   ```text
   approved -> observing -> resolving -> dispatching -> navigating -> verifying
            -> completed | paused | failed | cancelled | unknown
   ```

3. Build `VisualMissionRuntime` as deterministic Control API-owned orchestration. It does not run inside Mastra and does not directly call a robot.
4. On confirmation, return a parent mission receipt immediately; update progress over the existing event stream. A process restart while a child motion may be active transitions the parent to `unknown` and never replays it.
5. Add per-robot motion-authority reservation above individual ActionRuntime leases. A visual mission blocks ordinary manual motion but never blocks stop/cancel or non-motion reads.
6. For each V1 stage:
   - request structured stage resolution from H-IDE Agent;
   - receive only `visualResolutionId` and structured explanation;
   - read the canonical resolution directly from H-IDE Vision with read-only credentials;
   - revalidate identity, source, calibration, robot boot, localization epoch, scene, policy/model versions, mission limits, expiry, and executable gates;
   - optionally reuse existing Scene API deterministic free-space/region validation;
   - create one child `navigation.navigate_to_pose` request through ActionRuntime;
   - wait for normalized Robot Agent terminal status;
   - capture a new stopped observation and verify the semantic relation before completion.
7. Distinguish cancel from stop. Mission cancel stops future stages and cancels the current navigation; stop remains the existing highest-priority safety path.
8. Permit visual-network/no-frame retries at most twice using new frames. Never automatically redispatch a failed, unknown, expired, or safety-rejected navigation command.
9. Add four feature levels (`off`, `observe`, `lab_execute`, `robot_execute`) and robot allowlists. Default to `observe`.
10. Build the independent H-IDE Robot Agent Gateway. It owns `robot-agent.v1` WebSocket/mTLS session, control lease, state/capability stream, command query/cancel, and reconciliation. Control API uses only its scoped internal API. H-IDE Vision has no control credential.
11. Replace the `RobotAgentAdapter` skeleton only through a driver that calls the H-IDE Robot Agent Gateway; preserve FakeAdapter for existing non-visual tests, never for V1 success claims.

**Necessary tests:**

- Existing action PlanDraft deserializes, confirms, and executes unchanged.
- Visual mission confirmation never attempts to resolve `visual.execute_mission` as a Robot Agent command.
- Draft/mode/robot mismatch, expiry, scene drift, boot change, localization epoch change, calibration change, policy/model change, stale resolution, and duplicated resolution all produce zero dispatches.
- Manual motion conflicts while a mission owns motion; stop/cancel remain available.
- Restart produces `unknown`, retains child receipts, and never replays.
- One resolution causes at most one Robot Agent dispatch under repeat HTTP requests.

**Exit gate P5:** A reviewed VisualMission PlanDraft starts a durable mission, consumes a fixture VisualResolution, and creates exactly one idempotent navigation child command. All existing action-plan and PCD tests relevant to changed files remain green.

### Phase 6 — Add the Mastra vision-worker and stage resolution path

**Repository:** `h-ide-agent`

**Files:**

```text
src/contracts.ts
src/config.ts
src/http/routes.ts
src/mastra/agents.ts
src/planning/engine.ts
src/vision/client.ts
src/vision/worker.ts
test/vision-worker.test.ts
test/visual-mission-contracts.test.ts
test/visual-stage-route.test.ts
.env.example
README.md
```

**Work:**

1. Extend planning output to a strict union: action proposal or VisualMissionSpec proposal. The Supervisor selects the vision-worker when current visual evidence or a registered object is required.
2. Initial planning creates only the reviewable semantic mission and bounds; it cannot invent future coordinates.
3. Add an authenticated internal stage-resolution route used only by VisualMissionRuntime.
4. The vision-worker creates/polls/cancels the async H-IDE Vision resolution. It returns the immutable `visualResolutionId`, ambiguity information, and explanation—not a trusted coordinate copied from the LLM.
5. Treat OCR/image text as untrusted scene content. It may label a room but cannot alter the user task, system prompt, tools, command catalog, or safety constraints.
6. Keep DeepSeek V4 Flash for text planning/structuring as configured. Use Qwen3-VL only through H-IDE Vision's provider abstraction.
7. If multiple candidates are ambiguous, return `needs_input` with stable candidate IDs. User selection creates a new attempt/resolution instead of editing the old record.

**Necessary tests:**

- Visual language routes to vision-worker; ordinary action requests do not.
- Worker cannot output a path, `cmd_vel`, ROS type/name, or arbitrary command key.
- Malformed VLM/vision results fail closed.
- Cancel propagates to the resolution job.
- Fake vision fixtures produce deterministic mission-stage results.

**Exit gate P6:** A text or voice request creates a strict VisualMission PlanDraft, and a confirmed stage can obtain only a canonical fixture `visualResolutionId` through the vision-worker.

### Phase 7 — Add the compact chat visual-mission experience

**Repository:** `h-ide`

**Files:**

```text
apps/control-pwa/src/contracts.ts
apps/control-pwa/src/main.tsx
apps/control-pwa/src/chat-shell.css
apps/control-pwa/src/visual-mission-card.tsx
apps/control-pwa/src/visual-evidence-drawer.tsx
apps/control-pwa/src/main.test.tsx
apps/control-pwa/src/visual-mission.test.tsx
apps/control-pwa/public/sw.js
```

**Work:**

1. Keep the home screen as the simple dark chat interface.
2. Branch PlanDraft rendering by `planType`. Visual review shows semantic stages, bounds, mode, robot/map/source, model/policy pins, warnings, and unsupported requested actions.
3. Add a compact live task card with stage, state, one keyframe thumbnail, target, proposed approach pose, and pause/cancel/evidence actions.
4. Put camera snapshot, VLM candidates, SAM 2 mask, FoundationPose overlay, depth statistics, landmark pose, goal pose, safety gates, PCD check, and trace IDs in an expandable desktop side panel/mobile full-screen drawer.
5. Ambiguity UI displays numbered immutable candidates. Selection creates a new attempt. Manual point adjustment creates a separate human resolution and must be revalidated/reconfirmed.
6. Allow mobile view/review/long-press confirm/pause/cancel/stop only for operator/admin roles and only when deployment policy permits remote execution.
7. Display the health ladder rather than a single “camera online” label.
8. Display `运动为模拟` in `lab_execute`; only `robot_execute` after physical validation may show physical completion.
9. Serve bounded evidence through H-IDE's same-origin authenticated proxy. Never let the Service Worker cache observation frames, evidence URLs, uploads, or model responses.

**Necessary tests:**

- Action and visual PlanDraft cards both render and confirm correctly.
- Long chat history remains scrollable on desktop/mobile.
- Evidence drawer and ambiguity selection are keyboard/touch accessible.
- Stop/cancel remain available when vision, scene, or Agent requests fail.
- `observe` mode never enables confirmation; `lab_execute` always shows the simulated-motion label.

**Exit gate P7:** The complete fixture mission is understandable and controllable from desktop and mobile without exposing engineering panels on the main chat timeline.

### Phase 8 — Complete the Robot Agent boundary with a protocol simulator

**Repository:** no direct writes to `robotagent`

**Deliver to Robot Agent owner:**

```text
docs/integration/robot-agent-visual-handoff.md
artifacts/robot-observation-v1/
  observation.proto
  descriptor.pb
  PROTOCOL.lock
  fixtures/
  conformance-runner instructions
artifacts/robot-navigation-v1/
  navigation.navigate_to_pose.schema.json
  normalized-feedback.schema.json
  mock-nav2 acceptance steps
```

**Current H-IDE implementation:**

1. Implement a protocol-conformant Robot Agent simulator outside the Robot Agent repository. It must use the frozen `robot-agent.v1` WebSocket messages, control leases, capability snapshots, command lifecycle, query, cancel, heartbeat, and reconnect semantics.
2. Advertise `navigation.navigate_to_pose`, accept one map-frame pose, and return scriptable normalized `accepted|active|succeeded|failed|cancelled|unknown` results. It must never implement path planning or claim Nav2 execution.
3. Use unmistakable simulator IDs and receipts (`executionBackend=protocol_simulator`, `motion=simulated`). Reject attempts to start it under `robot_execute`.
4. Drive the production H-IDE Robot Agent Gateway client/session/driver. Do not let VisualMissionRuntime or its integration tests call `FakeAdapter` directly.
5. Cover lease loss, duplicate request/idempotency, command query reconciliation, cancel, disconnect while active, boot change, timeout, and restart-to-unknown.
6. Produce the observation/navigation handoff archive and deferred validation document as versioned artifacts.

**Deferred requested Robot Agent changes:**

1. Replace the temporary `navigation.vln.execute`/Fibonacci mapping with `navigation.navigate_to_pose` and real `nav2_msgs/action/NavigateToPose` semantics.
2. Return normalized `accepted|active|succeeded|failed|cancelled|unknown`, Robot Agent command ID, ROS goal ID, current/final map pose, feedback age, remaining distance, final error, and Nav2 error code.
3. Bind commands to observation session, boot, localization epoch, optional scene, TTL, and visual resolution; require usable observation when the mapping says so.
4. Use protocol-level `command.cancel` to cancel the active Nav2 goal and retain safe-stop/watchdog behavior.
5. Provide Mock Nav2 using the real NavigateToPose Action contract, not Fibonacci. Mock movement must remain visibly simulated.
6. Pass frozen observation protocol conformance, RGB-D Gateway, mTLS, observation watchdog, navigation cancellation, and safe-stop tests.

**Exit gate P8 / current code complete:** A fixture observation travels through production vision ingress; H-IDE Agent creates a reviewable visual mission; confirmation produces one canonical resolution and one idempotent `navigation.navigate_to_pose`; the production H-IDE Robot Agent Gateway exchanges the full command lifecycle with the protocol simulator. Failure/recovery tests pass, all receipts remain visibly simulated, and no `FakeAdapter` reports integration success.

### Phase 9 — Frozen real Robot Agent/Nav2 integration and system acceptance

**Status:** Deferred and non-blocking for current code completion. Do not execute until the readiness conditions in [`docs/integration/robot-agent-deferred-validation.md`](../../integration/robot-agent-deferred-validation.md) are satisfied.

**Repositories:** `h-ide`, `h-ide-agent`, `h-ide-vision`; Robot Agent evidence supplied by its owner; no unreviewed cross-repository edits

**Acceptance scenarios:**

1. **Registered object:** Real box plus accurate CAD/mesh -> FoundationPose -> pose error <= 10 cm and orientation error <= 15 degrees.
2. **Generic object:** Static chair/table -> Qwen3-VL + SAM 2 + depth -> position error <= 30 cm.
3. **PCD fusion:** Door or published room region -> live visual evidence plus immutable PCD region/free-space check.
4. **Deterministic fixture:** Known target -> final map coordinate error <= 5 cm.
5. **Navigation:** Mock Nav2 final reported pose error <= 25 cm, explicitly labeled simulated.
6. **Fail closed:** Stale frame, RGB-D desynchronization, bad CRC, missing TF, unstable localization, epoch change, protocol mismatch, ambiguous target, insufficient target depth, duplicate resolution, and scene/policy drift each produce zero robot dispatches.
7. **Recovery:** Kill API, worker, Agent, Gateway, and Control API at relevant states. Motion ownership becomes `unknown`/paused and never replays automatically; stop remains available.

**Performance goals:**

- Observation request to first usable RGB-D <= 2 s.
- Visual resolution P95 <= 15 s.
- Post-arrival visual verification <= 8 s.
- UI state propagation <= 500 ms.
- Gateway sustains requested 2 FPS baseline/5 FPS burst without unbounded queues.

**Required documentation:**

- Overall architecture and trust boundaries.
- `h-ide-vision` development/runbook and GPU compatibility lock.
- Robot Agent protocol/navigation handoff.
- D435i + lab map + real local Robot Agent + Mock Nav2 acceptance guide.
- Evidence retention, cloud VLM privacy, secrets, model/Prompt/policy upgrade procedure, and third-party notices.
- First physical robot test checklist: cleared low-speed area, on-site operator, physical E-stop, observe-only, generate-without-execute, short motion, cancel, network loss, E-stop, then staged missions.

**Exit gate P9 / integrated V1 complete:** Every acceptance scenario, performance goal, and required document in this phase has real-system evidence; the three accuracy paths are recorded; safety rejection tests produce zero dispatches; and the UI clearly distinguishes fixture, live camera, simulated motion, and physically validated motion. Until then, the honest release state remains `code_complete_robot_agent_pending`.

---

## 5. Minimal verification matrix

Run only the tests relevant to each changed boundary before moving to the next gate.

### H-IDE Vision

```bash
cd /home/phi/learn-projects/h-ide-vision
python -m pytest tests/contracts tests/unit tests/integration/test_fixture_resolution.py -q
python -m ruff check src tests
python -m mypy src
python scripts/gpu_smoke.py
python -m pytest -m gpu tests/gpu -q
```

### H-IDE control plane

```bash
cd /home/phi/learn-projects/h-ide
.venv/bin/python -m pytest \
  tests/test_visual_mission_contracts.py \
  tests/test_visual_mission_runtime.py \
  tests/test_visual_mission_api.py \
  tests/test_robot_agent_gateway.py \
  tests/test_assistant_conversations.py \
  tests/test_end_to_end.py -q
```

### H-IDE PWA

```bash
cd /home/phi/learn-projects/h-ide/apps/control-pwa
npm test -- src/visual-mission.test.tsx src/main.test.tsx
npm run build
```

### H-IDE Agent

```bash
cd /home/phi/learn-projects/h-ide-agent
npm test -- \
  test/vision-worker.test.ts \
  test/visual-mission-contracts.test.ts \
  test/visual-stage-route.test.ts \
  test/planning-engine.test.ts
npm run typecheck
```

### Protocol simulator acceptance (current)

Run the H-IDE-owned contract simulator and production gateway tests before declaring current code completion:

```bash
cd /home/phi/learn-projects/h-ide
.venv/bin/python -m pytest \
  tests/test_robot_agent_protocol_simulator.py \
  tests/test_robot_agent_gateway.py \
  tests/test_visual_mission_runtime.py \
  tests/test_visual_mission_fixture_e2e.py -q
```

These tests prove H-IDE's side of the frozen boundary, not Robot Agent or Nav2 behavior.

### Robot Agent owner acceptance (deferred)

Run in the Robot Agent worktree only after its owner applies the handoff:

```bash
python -m pytest \
  tests/test_observation_core.py \
  tests/test_observation_grpc_integration.py \
  tests/test_observation_mtls.py \
  tests/test_observation_ros_bridge.py \
  tests/test_agent_observation.py \
  tests/test_ros_integration.py -q
```

These commands and the live D435i plus Mock Nav2 run are frozen in the deferred validation document. They are not part of the current code-complete gate, and their absence must remain visible in release status.

---

## 6. Critical path and stop conditions

```text
P1 protocol freeze
   +--> P3 fixture ingress/runtime
   +--> Robot Agent conformance handoff

P2 GPU spike
   +--> P3 SAM 2/FoundationPose adapters

P3 fixture resolution
   +--> P5 H-IDE mission runtime
   +--> P6 Mastra vision-worker

P1 + observation fixture sender
   +--> P4 protocol-conformant observation boundary

P4 + P5 + P6 + P7 + Robot Agent protocol simulator
   +--> P8 current code complete

P8 + Robot Agent owner readiness + healthy D435i + lab map + Mock Nav2
   +--> P9 deferred integrated V1 acceptance
```

Current development stops successfully at P8 with status `code_complete_robot_agent_pending`; unavailable Robot Agent implementation is not a blocker before that gate. It becomes a blocker only after P9 is explicitly unfrozen.

Stop a phase and report the exact blocker when any of these occur:

- Before P8, the H-IDE-owned proto cannot express the required observation identity, timing, calibration, localization, or flow-control semantics. After P9 is unfrozen, the Robot Agent owner cannot accept that frozen contract without incompatible semantics.
- RGB-D is not synchronized/aligned or map TF/localization is unavailable.
- GPU extensions cannot run reliably on the locked Blackwell environment.
- A visual or language model output would need to be trusted for coordinates or safety.
- A mission could dispatch without immutable review, canonical resolution reread, or ActionRuntime.
- After P9 is unfrozen, Robot Agent still exposes only Fibonacci or a multi-waypoint VLN command.
- A restart/timeout path can replay a motion command.
- Tests require overwriting unrelated uncommitted user work.

---

## 7. Rollback and compatibility strategy

- Ship all new behavior behind `VISUAL_MISSION_MODE=off|observe|lab_execute|robot_execute`; default `observe`.
- Keep action PlanDraft and existing PCD spatial behavior unchanged and covered by regression tests.
- Treat H-IDE Vision as optional in health/readiness aggregation; never make the main gateway depend on its readiness.
- Roll model, Prompt, algorithm, ApproachPolicy, or calibration changes back by selecting a prior immutable version. Never mutate an active version.
- A protocol major/minor/hash mismatch disables observation rather than enabling a compatibility guess.
- Disable visual mission creation if Qwen is unavailable; keep Fake provider restricted to fixture/test profiles.
- If FoundationPose is unavailable, expose that capability as disabled while retaining generic VLM + SAM 2 operation.
- If live RGB-D or map localization fails, fall back only to display/`observe` mode or an explicitly reviewed PCD/manual-point plan—never to monocular distance guessing.
