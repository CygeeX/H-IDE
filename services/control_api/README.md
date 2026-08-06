# Control API

`services.control_api.main:create_app` builds an isolated FastAPI control
plane.  Every route requires `Authorization: Bearer <token>`; the websocket
also accepts the token as `?token=` for browser clients.

Interactive OpenAPI, Swagger, and ReDoc endpoints are disabled by default
(`/openapi.json`, `/docs`, and `/redoc` return 404) so robot capability
metadata is not exposed anonymously. If a deployment publishes a schema it
must put it behind the same authentication boundary. A built PWA bundle, when
present under `apps/control-pwa/dist`, is mounted as public static assets; it
contains no command authority without the authenticated API token.

The default development token is `dev-token`.  Deployments should pass
`auth_token=` or set `CONTROL_API_TOKEN`.

## Routes

- `GET /api/v1/robots` and
  `GET /api/v1/robots/{robotId}/capabilities` return the seeded fake robot
  and capability snapshots.
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
- `POST /api/v1/assistant/drafts` (singular alias supported) stores a
  `PlanDraft` only.  No adapter is called until
  `POST /api/v1/assistant/drafts/{draftId}/confirm`; confirmation returns the
  receipts produced by `ActionRuntime`. Confirmation is a durable
  compare-and-set state machine (`draft -> confirming ->
  confirmed|failed|partial|expired|unknown`); a restart marks an in-flight
  confirmation `unknown` and it is never replayed. Per-step receipts remain
  persisted for reconciliation, and a second confirmation is rejected.
- `GET /api/v1/events` reads persisted runtime events.  The websocket at the
  same path replays stored events and polls for new ones; disconnects stop the
  polling task without retaining a subscription.

All command execution is routed through immutable `CommandRequest`,
`CommandDefinition`, and `MappingVersion` objects and then `ActionRuntime`.
The API additionally rejects an embedded mapping target that does not match
the request robot.  Direct in-process runtime callers remain responsible for
that binding check.
