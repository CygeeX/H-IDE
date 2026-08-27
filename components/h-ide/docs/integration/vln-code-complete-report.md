# VLN Visual Target Resolution Code-Complete Report

**Recorded:** 2026-08-12  
**Release label:** `code_complete_robot_agent_pending`  
**Scope:** P1–P8 implementation with deterministic fixtures and protocol simulators. Real Robot Agent, D435i Gateway, Nav2, and physical-robot acceptance are excluded and remain frozen.

## Delivered

- `h-ide-vision`: strict observation/visual-resolution contracts, version/hash negotiation, reverse gRPC ingress with loopback or mTLS policy, bounded RGB-D frame storage, reconnect/session isolation, execution-time freshness checks, deterministic depth/TF geometry and safety gates, evidence storage/API, asynchronous jobs, Fake/Qwen grounding adapters, box/SAM 2 segmentation providers, and a continuous fixture sender.
- `h-ide-agent`: Mastra Supervisor plus dedicated vision worker, strict VisualMission draft contracts, model/policy version pinning, and canonical VisualResolution references. LLM output never supplies executable coordinates.
- `h-ide`: VisualMission draft review, immutable confirmation/runtime state, evidence proxy/drawer, cancel/reject/revise flows, ActionRuntime dispatch, production Robot Agent WebSocket client, protocol simulator, navigation/evidence handoff artifacts, and dark chat-first PWA integration.
- Deployment: CPU fixture Compose profile wires H-IDE Vision into the workstation stack while keeping plaintext observation traffic on loopback. Real GPU inference remains a separate host profile.

## Verification evidence

| Repository/layer | Result |
|---|---|
| H-IDE Vision | Ruff passed; strict mypy passed over 35 source files; 21 tests passed |
| H-IDE backend | 97 selected backend tests passed |
| H-IDE PWA | 22 selected tests passed; production build passed |
| H-IDE Agent | 39 tests passed; TypeScript typecheck passed |
| Compose | `docker compose -f infra/docker-compose.yml config --quiet` passed |
| CPU fixture image | Final H-IDE Vision image built; container `/healthz` returned `ok` with explicit fake/box capabilities |
| Observation protocol | Descriptor SHA-256 `d5055d5dbd81f2e098209bf9cba9db5739b74a3be62dae172958bd536648bb14` is pinned in the handoff artifacts |
| SAM 2 | Real RTX 5060 Ti/CUDA smoke passed with a non-empty mask |

The protocol simulator demonstrates H-IDE-owned behavior only. It is not evidence that the independent Robot Agent or Nav2 is integrated.

## Known unavailable capabilities at the recorded 2026-08-12 baseline

- Live Qwen3-VL inference: adapter exists, but no DashScope secret/live privacy-approved request was available for acceptance.
- FoundationPose: disabled because the local `.pth` files are Git LFS pointer files and PyTorch3D did not install. No FoundationPose success is claimed.
- Real D435i observation, Robot Agent command lifecycle, Nav2 goal mapping, robot-side watchdog/recovery, timing, accuracy, and physical motion: not yet tested.

## Frozen continuation

Run [robot-agent-deferred-validation.md](robot-agent-deferred-validation.md) only after the Robot Agent owner meets every unfreeze prerequisite. Until then, keep `robot_execute` disabled; use fixture/protocol-simulator modes and label motion as simulated. No files in `/home/phi/learn-projects/robotagent` were changed by this implementation.

## 2026-08-13 local pre-integration addendum

Qwen `qwen3.8-max`, `qwen3-vl-plus`, real D435i/Observation Gateway, the real
local Robot Agent and Mock Nav2 were exercised after this baseline. A fresh
natural-language TaskRequest produced a reviewed visual PlanDraft, immutable
VisualResolution, one real Robot Agent command and one Mock Nav2 ROS goal, all
ending successfully. Earlier stale/unavailable-observation runs also verified
zero-dispatch fail-closed behavior. H-IDE added workstation observation
preflight, nested state projection, bounded stable-frame collection, bounded
model-transport retries and auditable terminal rejection. Full details and
identifiers are in
[local-preintegration-2026-08-13.md](local-preintegration-2026-08-13.md).

The addendum does not change the release label: dirty/uncommitted worktrees,
the Robot Agent safe-stop timeout mismatch, the voice/safety/performance
matrix, physical motion, and formal owner review remain outstanding.

## 2026-08-17 Mission V2 hardening addendum

The continuous Mission V2 path now also rejects a real-agent draft when the
operator-bound localization epoch is missing, rejects approval/revision/skill
issuance after Robot Agent boot or epoch identity changes, and refuses a first
navigation without a fresh live `map` pose. The Gateway projects localization
fields from Robot Agent state and uses the UTC state timestamp associated with
actual localization changes; ROS header time remains provenance so simulated
ROS clocks do not create false expiry. A safe Mission cancel first cancels the
same physical command ID and only then terminalizes the Mission. These changes
are covered by the focused H-IDE Mission/Gateway tests; real Robot Agent/Nav2
acceptance remains frozen.

The 2026-08-17 Robot Agent-side change is limited to the two profile YAML files,
the `geometry_msgs` runtime dependency (with the already-present navigation /
observation dependencies), and its configuration expectation test. It does
not alter command execution or motion logic and still requires owner review.
