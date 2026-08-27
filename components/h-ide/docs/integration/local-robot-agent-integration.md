# Local Robot Agent integration

## 2026-08-17 preflight

The D435i is currently visible on USB (`Intel RealSense Depth Camera 435i`,
video nodes `/dev/video0`–`/dev/video5`). No RealSense ROS node or Nav2 node is
running in the checked ROS domain at the time of this check, so this is a
hardware-presence check only—not a live observation or motion result.

The Robot Agent checkout is at `2d3d4b5` with concurrent uncommitted changes;
H-IDE is at `1b30e0a` with uncommitted changes. H-IDE Vision and h-ide-agent
are present as uncommitted worktrees without a commit SHA. Formal RA-0 version
freezing therefore remains pending even though the local Nav2 interface and
configuration checks now pass.

This profile lets the real local Robot Agent remain a WebSocket **client** and
connect outbound to the workstation-owned H-IDE Robot Agent Gateway. It does
not authorize physical motion; keep `VISUAL_MISSION_MODE=lab_execute` and use a
Mock Nav2 `NavigateToPose` server for the first run.

## H-IDE Gateway configuration

Use the exact values printed by Robot Agent `validate-config`. Do not advertise
a command that is absent from that Robot Agent configuration.

Last read-only compatibility check: **2026-08-13 CST**. The concurrent
Robot Agent worktree reported:

- client URI `ws://127.0.0.1:8093/agent`;
- capabilities digest
  `sha256:038575b99105c456bd9a890fa389d0b70187be848ac1c241483ae731968b16a6`;
- commands `navigation.navigate_to_pose`, `power.enable`, and `task.execute`;
- navigation target `/navigate_to_pose` with
  `nav2_msgs/action/NavigateToPose` and `navigate_to_pose_v1`;
- Observation descriptor lock
  `d5055d5dbd81f2e098209bf9cba9db5739b74a3be62dae172958bd536648bb14`,
  matching H-IDE Vision.

These are a time-stamped handoff, not permanent constants. Recompute the
digest and compare both protocol locks immediately before every integration
run; a concurrent Robot Agent edit invalidates this snapshot.

The older snapshot listed only `navigation.navigate_to_pose`, `power.enable`
and `task.execute`; the current profile also advertises
`navigation.rotate`. Treat the live manifest as authoritative.

## 2026-08-17 pose handoff delta

For the next local integration window, H-IDE was allowed to make a narrow
Robot Agent-side handoff change. Only the local and production example YAML
profiles now project the `map`-frame `/localization/pose`
(`geometry_msgs/msg/PoseWithCovarianceStamped`) fields through the existing
state channel, and `geometry_msgs`, `nav2_msgs` and
`robot_observation_interfaces` are declared as runtime ROS dependencies. No
Robot Agent command adapter, Nav2 action implementation, watchdog, lease or
motion code was changed by H-IDE. Re-run `validate-config`, regenerate the
capability digest and have the Robot Agent owner review these files before
using the profile.

The H-IDE Gateway treats ROS header stamps as provenance only. Freshness for
Mission admission uses the UTC state-envelope time associated with changed
localization fields (and ignores unrelated telemetry), which also works when a
local Mock Nav2 domain uses simulated ROS time. A first navigation is rejected
unless this live pose is present, bound to the current boot/epoch and no older
than the configured two-second window.

The current local profile was structurally validated on 2026-08-17 with
`validate-config` in the available ROS environment:

```text
robot_id: g1-dev
capabilities_digest: sha256:1c3c0510fbab0efd69be9f59a945825a6085962001cbf155c6cd2efb4c3fcab2
command_count: 4
state_source_count: 11
```

This digest supersedes the older 2026-08-13 snapshot for a run using the
current profile. It is still a handoff value: copy the digest and the exact
advertised capability list from the Robot Agent owner immediately before
connecting H-IDE.

The system ROS installation does not expose `nav2_msgs` by default on this
workstation, but the Robot Agent checkout contains a user-space Jazzy prefix.
Source it after the system ROS setup and before running the Agent or its
verification commands:

```bash
source /opt/ros/jazzy/setup.bash
source $SKILLSTORE_ROOT/components/robotagent/tools/activate-local-nav2-jazzy.bash
```

With that prefix active, the read-only `validate-config` and `doctor` checks
were rerun on 2026-08-17. Both `navigation.navigate_to_pose` and
`navigation.rotate` resolve to their real `nav2_msgs` action types and
`doctor` reports `healthy: true`. The system-only check still reports
`Package not found`; that is expected until the user-space prefix is sourced,
and must not be confused with a missing protocol mapping. The local Nav2
prefix provides interfaces for the Mock Nav2/pre-integration domain; it is not
evidence that a physical Nav2 stack or a physical base is running.

```dotenv
ROBOT_AGENT_GATEWAY_ENABLED=true
ROBOT_AGENT_GATEWAY_BACKEND=websocket_inbound
ROBOT_AGENT_ID=g1-dev
ROBOT_AGENT_CAPABILITIES_DIGEST=sha256:1c3c0510fbab0efd69be9f59a945825a6085962001cbf155c6cd2efb4c3fcab2
ROBOT_AGENT_CAPABILITIES=navigation.navigate_to_pose,navigation.rotate,power.enable,task.execute
ROBOT_AGENT_LOCALIZATION_EPOCH_ID=q40-map-epoch-dev-1
ROBOT_AGENT_ALLOW_PLAINTEXT_LOOPBACK=true
VISUAL_MISSION_MODE=lab_execute
HIDE_VISION_SOURCE_MODE=live
```

`ROBOT_AGENT_GATEWAY_ENABLED` is the Control API execution-profile switch.
Leaving it false keeps the self-contained FakeAdapter demonstration even when
Gateway URL/token values are present. Setting it true removes the automatic
Fake fallback: only capabilities reported by the live Robot Agent can be
planned, enabled in the UI, or dispatched.

`ROBOT_AGENT_LOCALIZATION_EPOCH_ID` is an explicit observation identity
binding, not a planner hint. It must equal the value published by the current
Robot Agent localization/Observation Gateway profile. The control WebSocket
does not transport this value; when the binding is absent or a reviewed visual
mission contains another epoch, H-IDE rejects execution before resolution or
dispatch.

Start the Gateway on loopback port 8093. The Robot Agent client URI is:

```text
ws://127.0.0.1:8093/agent
```

Only one Robot Agent controlling connection is accepted. `/healthz` reports
`robotAgent.connected`, robot identity, boot identity, control state, projected
observation status/session/epoch/age and the last protocol error. The Control API refreshes its robot/capability snapshot
from this authenticated internal Gateway and does not claim the Agent is
online before the session is available.

## Command lifecycle

H-IDE sends one idempotent `command.request`. `RECEIVED` and `RUNNING` remain
non-terminal. VisualMissionRuntime polls the Gateway ledger until
`SUCCEEDED`, `FAILED`, `CANCELLED`, `REJECTED`, `TIMED_OUT`, or
`ABORTED_AGENT_RESTART`. It never replays the physical operation. A disconnect
marks unresolved commands unknown and latches an H-IDE motion safety barrier.
Reconnect alone does not release that barrier; an explicit safety/recovery
operation and operator confirmation are required.

Mission V2 keeps its readable `skill-...` request ID for audit and durable
correlation. At the Gateway transport boundary, that namespace is converted
to a deterministic UUID5 required by `robot-agent.v1`; UUID request IDs pass
through unchanged. Query and cancel apply the same conversion, so a retry or
Gateway restart cannot create a second physical command identity (a restart
still requires the frozen reconciliation procedure). Receipts retain the
Robot Agent wire command ID as `robotAgentCommandId`.

## Local pre-integration status

The real local Robot Agent, real Robot Observation Gateway, D435i, Qwen/SAM 2
and Mock Nav2 have now been exercised in a development pre-integration. This
includes a successful natural-language-to-Mock-Nav2 chain, but does not
unfreeze formal acceptance. See the dated evidence in
[local-preintegration-2026-08-13.md](local-preintegration-2026-08-13.md).

## Remaining external prerequisites

- Robot Agent declares the frozen Observation descriptor hash and the
  `navigation.navigate_to_pose` → `nav2_msgs/action/NavigateToPose` mapping;
  local runtime interoperability with Mock Nav2 has been observed, but the
  formal matrix and owner-reviewed evidence are still pending.
- Resolve the Robot Agent local safe-stop timeout mismatch; the stock 1000 ms
  profile did not complete its startup stop inside the configured deadline.
- Start H-IDE Vision gRPC ingress on `127.0.0.1:50061`, and point the Robot
  Agent Observation Gateway there.
- Run the full contract/camera/Mock Nav2 safety, recovery, accuracy, timing,
  voice, and version-frozen acceptance matrix before considering real movement.
- For cross-machine or production deployment replace plaintext loopback with
  an mTLS reverse proxy or an mTLS-capable inbound Gateway transport.
