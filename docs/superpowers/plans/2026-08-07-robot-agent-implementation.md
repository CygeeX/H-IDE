# Robot Agent v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and locally verify a production-shaped, fail-closed WebSocket-to-ROS 2 Robot Agent core.

**Architecture:** One ament_python process separates asyncio transport/session logic from a dedicated rclpy executor thread. Strict external models and a local command registry prevent remote access to arbitrary ROS graph resources.

**Tech Stack:** Python 3.10/3.12, ROS 2 Humble/Jazzy, rclpy, websockets, Pydantic v2, jsonschema, SQLite, pytest, launch_testing, systemd.

---

### Task 1: Package and test foundation

- [x] Create the ament_python package, dependency locks, schemas/config directories, launch skeleton, and pytest configuration.
- [x] Verify pure pytest collection and colcon package discovery.

### Task 2: Protocol and configuration

- [x] Write failing tests for strict envelope parsing, all message families, time/target/frame validation, TLS configuration, and local command registry validation.
- [x] Implement the minimal Pydantic models and configuration loader; export JSON Schemas.
- [x] Run focused tests and static checks.

### Task 3: Persistence, arbitration, and session safety

- [x] Write failing tests for durable command transitions, same-ID retry, ID mismatch, restart recovery, resource overlap, lease expiry, explicit acquire, and safe-stop latching.
- [x] Implement SQLite ledger, resource coordinator, session state machine, priority events, and state coalescing.
- [x] Run focused tests and verify the database migration path from an empty file.

### Task 4: WebSocket transports

- [x] Write failing client/server integration tests with generated CA/server/client certificates.
- [x] Implement shared connection handling, mTLS contexts, subprotocol enforcement, single-session server policy, client reconnect backoff, frame limits, and queue backpressure.
- [x] Verify rejected CA/SAN, binary/oversized frames, disconnect, and reconnect-unarmed behavior.

### Task 5: ROS adapter and runtime

- [x] Write failing tests around mock Trigger/SetBool services, Fibonacci action feedback/cancel, state topics, QoS, ROS timeout, and safety heartbeat.
- [x] Implement the rclpy node, executor bridge, dynamic local type loading/conversion, Service/Action handlers, state sources, diagnostics, and safe shutdown.
- [x] Verify no synchronous ROS wait occurs in callbacks and every entity has explicit QoS/callback group configuration.

### Task 6: Operations and documentation

- [x] Add `run`, `validate-config`, and read-only `doctor` commands, launch file, Mock BFF/ROS tools, systemd notify/watchdog support, hardened Unit, example configuration, and operator README.
- [x] Validate config/Schema examples and run `systemd-analyze verify`.

### Task 7: Full acceptance

- [x] Run Ruff, mypy, unit/integration tests, coverage, and colcon build/test on Jazzy.
- [x] Build the pinned Humble test image and run the same suite.
- [x] Run fault tests and a 30-minute soak; record timing, memory, queue, and replay results.
- [x] Re-read the design and close every requirement before completion.
