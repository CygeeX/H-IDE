# Control API

`services.control_api.main:create_app` builds an isolated FastAPI control
plane, and the module-level `services.control_api.main:app` is the single
process entry point used by Docker/systemd. Every route requires
`Authorization: Bearer <token>`; the websocket
also accepts the token as `?token=` for browser clients.

Interactive OpenAPI, Swagger, and ReDoc endpoints are disabled by default
(`/openapi.json`, `/docs`, and `/redoc` return 404) so robot capability
metadata is not exposed anonymously. If a deployment publishes a schema it
must put it behind the same authentication boundary. A built PWA bundle, when
present under `apps/control-pwa/dist`, is mounted as public static assets; it
contains no command authority without the authenticated API token.

The default development token is `dev-token`.  Deployments should pass
`auth_token=` or set `CONTROL_API_TOKEN`. When `create_app()` is called without
an explicit `db_path`, it reads `CONTROL_DB_PATH` (falling back to `:memory:`),
so service restarts retain commands, mappings and audits when a file path is
configured.

## Routes

- Mission V2 is the primary Agent task API. `POST /api/v2/missions/drafts`
  creates a review-only semantic DAG; approval and bounded continuous
  execution use `POST .../drafts/{draftId}/approve` and
  `POST .../missions/{missionId}/advance`. Follow-up messages produce an
  immutable PlanRevision. `POST .../missions/{missionId}/cancel` first
  cancels the active ActionRuntime / Robot Agent command under the same
  request ID, then terminalizes the Mission. `GET /api/v2/missions` and
  `GET .../{missionId}` expose the durable event, fact and execution record.
  Product visual results are accepted only when their immutable version block
  identifies `qwen / qwen3-vl-plus / sam2`, and every semantic hypothesis has
  `source=qwen`. Retired semantic records remain available through history and
  evidence reads with `legacyReadOnly=true` and `executionEligible=false`; the
  runtime never validates them as a new `WorldRecord` or resumes their Mission.
- `GET /api/v1/robots` and
  `GET /api/v1/robots/{robotId}/capabilities` return the seeded fake robot
  and capability snapshots.
- `POST /api/v1/leader-demo/gestures/wave` is a simulation-only voice-assistant
  presentation hook. It accepts only the bound `g1-sim` ID and calls the
  host-side allow-listed gesture proxy; the Control API container never has
  access to SONIC's raw command socket.
- `POST /api/v1/commands` accepts a manual command body.  `commandId`,
  `issuedAt`, `expiresAt`, `idempotencyKey`, `sessionId`, and `traceId` may be
  omitted; the server fills safe defaults.  The response is a camelCase
  `CommandReceipt`.  `GET /api/v1/commands` lists receipts,
  `GET /api/v1/commands/{commandId}` reads one, and
  `POST /api/v1/commands/{commandId}/cancel` calls the same runtime cancel
  path. `idempotencyKey` is durably unique in SQLite: retries return the
  original immutable receipt (including after restart), concurrent requests
  claim one owner, and a reused `commandId` with a different key returns
  `409`.
- `POST /api/v1/mappings/drafts` (also `POST /api/v1/mappings`) accepts a
  `MappingVersion` plus optional `targetId`.  Lifecycle calls are
  `POST /api/v1/mappings/{mappingId}/{version}/{validate|probe|activate}`;
  rollback is `POST /api/v1/mappings/{commandKey}/rollback` with optional
  `{"version": "..."}`.  `GET /api/v1/mappings` lists persisted records.
- The `/api/v1/assistant/*` routes are migration compatibility for historical
  records and pre-Mission workflows. `POST /api/v1/assistant/drafts`
  (singular alias supported) stores a
  `PlanDraft` only.  No adapter is called until
  `POST /api/v1/assistant/drafts/{draftId}/confirm`; confirmation returns the
  receipts produced by `ActionRuntime`. Confirmation is a durable
  compare-and-set state machine (`draft -> confirming ->
  confirmed|failed|partial|expired|unknown`); a restart marks an in-flight
  confirmation `unknown` and it is never replayed. Per-step receipts remain
  persisted for reconciliation, and a second confirmation is rejected.
- `POST /api/v1/assistant/conversations` creates an owner/site-scoped
  conversation. `robotId` is optional so ordinary chat and QA do not require a
  robot binding;
  `POST /api/v1/assistant/conversations/{conversationId}/messages` accepts a
  natural-language text or reviewed voice transcript and returns `202` with a
  turn `taskId`. The Assistant V2 gateway calls Mastra's typed conversation
  router first. Social chat, QA and admin explanations return text without a
  draft; observation calls a fixed-head `scene_description` VisualTask and can
  never dispatch motion; scene teaching remains pending confirmation. Only an
  unambiguous `ROBOT_ACTION` may call Mission V2, which still produces a
  reviewable draft and cannot bypass policy or human confirmation. `GET
  /api/v1/assistant/turns/{taskId}` polls the durable result, `GET .../events`
  returns ordered event-compatible output, and the per-turn visual-evidence
  proxy keeps the Vision credential out of the browser. Full message history is
  the H-IDE business source of truth. Mastra receives only a server-assembled
  window of at most 20 messages, an append-only digest-checked summary and at
  most eight relevant conversational excerpts. Every context selection has a
  durable revision/audit row; browser `memoryContext` values are ignored and
  remembered map/pose coordinates are redacted before routing or Mission
  history. This context may resolve semantic pronouns but has no scene,
  localization, reachability, or safety authority. Conversation display names and recoverable
  lifecycle state are managed with `PATCH
  /api/v1/assistant/conversations/{conversationId}` (`title`, `archived`, and
  `deleted`). `DELETE` writes the same deletion tombstone and never physically
  deletes messages, turns, Mission records, or evidence. Lists omit archived
  and deleted records by default; authenticated recovery tools may opt in with
  `includeArchived=true` and `includeDeleted=true`, then clear the flags through
  `PATCH`. The product history uses cursor-paged `GET
  /api/v2/assistant/conversations` (`lifecycle`, `q`, `limit`, `cursor`) and
  returns one root row per conversation with its durable Mission draft/Mission
  links; Mission records are never projected as duplicate conversations.
  `POST /api/v2/assistant/conversations/bulk-delete` soft-deletes up to 100
  authorized terminal conversations and reports active turns or active Missions
  as skipped. A lifecycle mutation is rejected while either kind of work is
  active. Every read and mutation remains owner/site scoped and Cookie writes
  require CSRF validation.
- Generated drafts support `POST /api/v1/assistant/drafts/{draftId}/revise`
  with feedback and `POST .../reject` with a reason. Revision creates a new
  planning turn and never mutates the original plan payload.
- Conversation detail returns server-assigned `turnSequence` values and
  `missionLinks[]` (`taskId -> missionDraftId -> missionId`). Product clients
  render messages, evidence, plans, Mission progress and terminal reports
  under that originating turn instead of selecting one global latest card.
- Qwen visual hypotheses are projected into coordinate-free
  `SemanticReferent` records for immediate same-conversation names/pronouns.
  They are `SEMANTIC_ONLY`, filtered by Robot Agent boot/localization epoch and
  must be freshly re-grounded before object navigation.
- Known Office object approach, room exit and active object search use
  deterministic Mission templates. Malformed planner geometry or invented
  record IDs are not accepted as a fallback.
- Scene upload, conversion, calibration, annotations, publishing and robot
  bindings are owned by the separate Scene API and exposed through the
  same-origin gateway. For planning, Control API reads only the internal
  compact context for the robot. A browser may submit a matching `sceneRef`
  and one selected map point, but cannot replace canonical bounds or semantic
  annotations.
- `POST /api/v1/speech/transcriptions` accepts at most 60 seconds / 10 MiB of
  16 kHz mono 16-bit PCM WAV and delegates transcription to the configured
  local whisper.cpp endpoint. Transcription never submits a task by itself.
  The returned server-issued record is bound to the authenticated owner/site;
  a different user cannot reuse its identifier in a voice message.
- `WS /api/v2/speech/stream` accepts 20-60 ms PCM16 frames and returns server-VAD
  partial/final transcripts. Cookie clients echo the CSRF cookie value in
  `session.start`; Bearer credentials and session tokens are never accepted in
  query strings. Before each final event is released to the browser, Control API
  durably issues an owner/session/utterance-bound `transcriptionId`. Duplicate
  provider finals for one utterance return the same identifier. A persistence
  failure closes the stream without exposing an unsigned final.
- `POST /api/v2/speech/tts` streams 24 kHz mono PCM16 from the configured real
  provider. `/api/v2/speech/healthz` and `/api/v2/speech/readyz` expose
  authenticated ASR/TTS capability state without exposing provider keys.
- `GET /api/v1/events` reads persisted runtime events.  The websocket at the
  same path replays stored events and polls for new ones; disconnects stop the
  polling task without retaining a subscription.

All command execution is routed through immutable `CommandRequest`,
`CommandDefinition`, and `MappingVersion` objects and then `ActionRuntime`.
The API additionally rejects an embedded mapping target that does not match
the request robot.  Non-safety actions fail closed unless the persisted robot
state is explicitly connected and in an online/degraded, normal/degraded
safety state; `stop` remains a `SAFETY` command. Direct in-process runtime
callers remain responsible for that binding and robot-state check.

`navigation.navigate_to_pose` is the logical command used by spatial Agent
plans. Its payload contains both a ROS `map` goal and the complete deterministic
`spatialResolution`. Control API resolves it once before persisting a draft and
again on confirmation. A changed scene binding, altered resolution, or robot
pose drift beyond the configured thresholds is rejected before dispatch.
Path planning is deliberately outside this API and remains a Robot Agent/Nav2
responsibility.

## Robot Agent execution profile

The default profile is an isolated `FakeAdapter` demo. A configured Gateway
URL/token does not silently change command authority. To use the real internal
Gateway, set all three values:

```dotenv
ROBOT_AGENT_GATEWAY_ENABLED=true
ROBOT_AGENT_GATEWAY_URL=http://127.0.0.1:8093
ROBOT_AGENT_GATEWAY_TOKEN=<internal-token>
```

For the inbound Robot Agent profile,
`ROBOT_AGENT_OBSERVATION_PREFLIGHT_TIMEOUT_SECONDS` (default `2`) bounds how
long the workstation waits for a fresh Robot Agent observation state, and
`ROBOT_AGENT_OBSERVATION_PREFLIGHT_MAX_AGE_MS` (default `400`) limits both ACK
and capture age. Navigation is rejected in the Gateway ledger with zero
`command.request` messages when status, session, epoch, or age is unsafe.

Real mode does not auto-install a Fake fallback. Robot state and command
buttons are projected from live Gateway discovery, and every manual or planned
command must be present in the current capability snapshot. A missing or
disconnected Robot Agent therefore fails closed.

## Assistant service configuration

- `MASTRA_PLANNING_URL` and `MASTRA_INTERNAL_TOKEN` enable both the internal
  Assistant V2 router and Mission V2 planner. If either is missing, legacy V1
  embedded callers remain compatible and configured V2 routes fail closed;
  manual control remains available.
- `MASTRA_CONVERSATION_TIMEOUT_SECONDS` defaults to 30 seconds.
- `MASTRA_PLANNING_TIMEOUT_SECONDS` defaults to 120 seconds.
- `WHISPER_ENDPOINT` enables local speech transcription and normally ends in
  `/inference`; `WHISPER_TIMEOUT_SECONDS` defaults to 45 seconds.
- `SCENE_API_INTERNAL_URL` and `SCENE_API_INTERNAL_TOKEN` enable canonical
  robot scene context. The token must be distinct from browser and Mastra
  tokens; `SCENE_API_TIMEOUT_SECONDS` defaults to 3 seconds.
- `SPATIAL_CONFIRM_TRANSLATION_DRIFT_METERS` and
  `SPATIAL_CONFIRM_YAW_DRIFT_RADIANS` default to `0.5` and `0.35`. They only
  allow bounded motion between planning and confirmation; they do not weaken
  the exact scene revision or deterministic goal checks.

The browser never receives the Mastra token or Whisper endpoint. See
`docs/architecture/assistant-supervisor-worker.md` and
`docs/architecture/pcd-scene-spatial-agent.md` for the full trust boundary.
