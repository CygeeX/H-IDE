# Robot Action Console Implementation Plan

> **For agentic workers:** Execute tasks in order with tests first. Each task is independently reviewable.

**Goal:** Build a runnable workstation control plane that supports high-level manual commands, confirmed Agent plans, versioned interface mappings, multi-robot state, and a browser PWA.

**Architecture:** A FastAPI BFF fronts a Python action runtime and adapter registry. Logical commands are transport-independent; Robot Agent, HTTP, MQTT, and ROS 2 adapters translate immutable mapping versions. A React/TypeScript PWA consumes REST/WebSocket contracts; scene/video/assistant integrations are isolated from the safety path.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, pytest, React, TypeScript, Vite, WebSocket, SQLite for the runnable local profile (PostgreSQL/Redis deployment configuration documented), Docker Compose, systemd.

---

### Task 1: Domain contracts and validation

**Files:**
- Create: `packages/control_core/models.py`
- Create: `packages/control_core/validation.py`
- Create: `tests/test_contracts.py`

- [ ] Define `CommandDefinition`, `CommandRequest`, `PlanDraft`, `CommandReceipt`, `RobotState`, `CapabilitySnapshot`, `TargetProfile`, and `MappingVersion` with Pydantic validation.
- [ ] Enforce command IDs, robot IDs, expiry timestamps, payload schemas, and safe transport names.
- [ ] Test valid requests, expired requests, unknown transports, malformed plans, and immutable mapping versions.

### Task 2: Action runtime and adapter registry

**Files:**
- Create: `packages/control_core/runtime.py`
- Create: `packages/control_core/adapters.py`
- Create: `packages/control_core/registry.py`
- Create: `tests/test_runtime.py`
- Create: `tests/test_registry.py`

- [ ] Implement the command lifecycle, per-robot motion lock, idempotency, lease expiry, cancellation, and event publication.
- [ ] Implement Fake, Robot Agent, HTTP, MQTT, and ROS 2 adapter interfaces; live transports may initially return an explicit unavailable result.
- [ ] Implement draft → validate → probe → activate → rollback mapping lifecycle with atomic active-version pointers.
- [ ] Test duplicate commands, stale leases, in-flight version pinning, probe failures, rollback, and capability mismatch.

### Task 3: BFF API and persistence

**Files:**
- Create: `services/control-api/main.py`
- Create: `services/control-api/store.py`
- Create: `services/control-api/requirements.txt`
- Create: `tests/test_api.py`

- [ ] Expose robot, capability, command, event, mapping validation/probe/activation/rollback, and assistant-draft endpoints.
- [ ] Persist commands, receipts, mappings, and audit events in SQLite for local execution; keep repository interfaces compatible with PostgreSQL.
- [ ] Serve the PWA static bundle and a WebSocket event stream.
- [ ] Test manual command submission, Agent draft confirmation, cancellation, config activation, rollback, and unauthorized/expired requests.

### Task 4: Responsive PWA

**Files:**
- Create: `apps/control-pwa/package.json`
- Create: `apps/control-pwa/index.html`
- Create: `apps/control-pwa/src/main.tsx`
- Create: `apps/control-pwa/src/contracts.ts`
- Create: `apps/control-pwa/src/app.css`
- Create: `apps/control-pwa/src/main.test.tsx`

- [ ] Build Fleet, Robot Workspace, command controls, Agent plan confirmation, command timeline, and Interface Center views.
- [ ] Keep all motion controls high-level; show stale state and preserve stop/cancel when scene/video services fail.
- [ ] Render command forms from logical schemas; mapping edits show validation/probe results and version history.
- [ ] Test confirmation gating, rejected/expired receipts, WebSocket reconnect state, mapping activation, and 375px/1280px layouts.

### Task 5: Isolated scene/video/assistant adapters and deployment

**Files:**
- Create: `services/scene-api/README.md`
- Create: `services/scene-worker/README.md`
- Create: `services/video-gateway/README.md`
- Create: `infra/docker-compose.yml`
- Create: `infra/systemd/robot-action-console.service`
- Create: `.env.example`

- [ ] Document PDAL/Potree job boundaries, ZMQ→WebRTC boundary, and Agent/ASR gateway contracts.
- [ ] Provide Docker Compose for API, database, Redis, and static serving; keep ROS 2 adapter host-network compatible.
- [ ] Provide systemd dependency ordering, restart, watchdog, and journal logging configuration.

### Task 6: End-to-end and safety verification

**Files:**
- Create: `tests/test_end_to_end.py`
- Create: `docs/operations/safety-and-weak-network.md`
- Create: `README.md`

- [ ] Verify manual and multi-step Agent commands use the same runtime.
- [ ] Verify v1→v2 interface changes through the UI without frontend code changes and verify rollback.
- [ ] Verify duplicate/late/expired messages, adapter restarts, VPN loss, and no automatic replay.
- [ ] Verify scene/video/assistant failures do not block stop/cancel and document that physical emergency stop remains authoritative.

