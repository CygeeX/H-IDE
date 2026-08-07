# Robot Agent v1 Design

## Goal

Build the robot-side trust boundary between a workstation WebSocket session and local ROS 2 interfaces without exposing a generic rosbridge surface.

## Architecture

The deployable unit is one `ament_python` process. An asyncio event loop owns WebSocket transport, protocol/session state, command policy, persistence, and outbound state. A dedicated rclpy executor thread owns all ROS entities. Thread-safe request and event queues connect the two domains.

The same binary supports outbound WebSocket client mode and inbound server mode; configuration enables exactly one. After mTLS, the Agent always initiates the application handshake. One controlling session is allowed. Loss of the session or its 2-second application lease invokes the mandatory safe-stop adapter and latches `SAFE_HOLD` until a new session explicitly acquires control.

## Wire Contract

`robot-agent.v1` uses strict JSON text frames. The envelope fields are `version`, `type`, `message_id`, `robot_id`, `session_id`, `sent_at`, and `payload`. Session, lease, control, command, state, safety, and error messages are represented by discriminated models. Frames larger than 64 KiB, unknown fields, binary frames, invalid targets, expired commands, and reused command IDs with different payloads are rejected.

Remote callers use logical command kinds only. A local registry maps each kind to a ROS Service or Action, JSON Schema, maximum TTL, timeout, and required resources. Topic input is normalized into a stable `RobotState`; no generic remote Topic publication or subscription is exposed.

## Safety and Persistence

SQLite records canonical payload hashes and command state transitions before ROS execution. Same-ID retries return recorded state. Non-terminal rows found after restart become `ABORTED_AGENT_RESTART` before any control can be acquired. Commands with overlapping resources are rejected; safe stop bypasses all resource locks, cancels active actions, and latches safe state.

ROS safety heartbeat is published every 250 ms with explicit reliability, deadline, and lifespan so a future independent monitor can stop the robot if the Agent process dies.

## Verification

Pure core behavior is unit tested without ROS. Mock ROS services, actions, and topics exercise the adapter in Jazzy and an official Humble container. Both transport modes receive mTLS integration coverage. Failure tests cover invalid identity, stale commands, duplicate IDs, resource conflicts, ROS timeout/cancel behavior, disconnect, lease expiry, queue saturation, and restart recovery. A 30-minute soak verifies bounded queues and event-loop responsiveness.

