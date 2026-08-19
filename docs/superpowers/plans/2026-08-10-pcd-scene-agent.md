# H-IDE PCD Scene and Spatial Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use subagent-driven-development and test-driven-development; do not overwrite pre-existing uncommitted work.

**Goal:** Deliver a resumable PCD-to-Potree scene workflow with map calibration, semantic annotation, deterministic Agent target resolution, human review, and FakeAdapter pose feedback.

**Architecture:** A separate scene-api owns metadata and local storage; a leased scene-worker runs fixed PCL/PDAL/PotreeConverter commands. The PWA uses Potree through a same-origin gateway. Mastra invokes a deterministic H-IDE spatial tool, while Control API remains the execution authority.

**Tech Stack:** FastAPI/Pydantic/SQLite, React/Vite/PWA, Mastra/TypeScript, Caddy, PCL 1.15.1, PDAL 2.10.0, PotreeConverter 2.1.3, Potree 1.8.2.

---

### Task 1: Scene contracts and deterministic spatial resolver

- Add immutable scene, map pose, calibration, annotation and spatial intent/result contracts.
- Add pure identity/similarity calibration and relative/semantic/selected-point resolution.
- Add failing tests first for validation, coordinate math, bounds and ambiguity.

### Task 2: Scene API, local storage and resumable uploads

- Add a separate FastAPI app with owned SQLite metadata and LocalSceneStorage.
- Add 8 MiB offset/checksum upload sessions, completion, jobs, scene revisions, bindings and asset sessions.
- Add recovery, traversal, expiry, conflict and Range tests.

### Task 3: Worker and deployment boundary

- Add leased job claim/heartbeat/complete/fail flow and fixed command runner.
- Normalize compressed PCD with PCL, convert with PDAL and PotreeConverter, validate and atomically publish.
- Add fixed build inputs, gateway routes, health checks and cleanup behavior.

### Task 4: PWA scene experience

- Add scene library/import/progress, Potree wrapper, two calibration wizards and semantic polygon/entry editing.
- Add desktop and touch interaction modes plus review overlays.
- Add component and browser tests; exclude uploads and range assets from service-worker caching.

### Task 5: Agent and ActionRuntime integration

- Extend TaskRequest/PlanningEnvelope with compact scene context and typed pose.
- Give motion-worker the deterministic resolver tool; add navigation.navigate_to_pose.
- Verify stored resolutions at draft and confirmation, reject stale/tampered/drifted plans, update Fake pose only on success.

### Task 6: Integration and acceptance

- Run full Python/PWA/Agent suites, compose validation and actual tiny PCD conversion.
- Exercise import, calibration, kitchen annotation, semantic navigation and relative navigation on desktop/mobile.
- Document local development, storage, limits, failure recovery and out-of-scope path planning.
