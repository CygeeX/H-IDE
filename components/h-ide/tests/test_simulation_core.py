from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from packages.simulation_core import SimulationConflict, SimulationRuntime


class _MujocoClientStub:
    def __init__(self) -> None:
        self.pose = {
            "frameId": "map",
            "x": 0.31,
            "y": -0.02,
            "z": 0.77,
            "yaw": 0.03,
            "observedAt": None,
        }

    def ready(self) -> bool:
        return True

    def live(self) -> dict:
        return {
            "backend": "mujoco_sonic",
            "ready": True,
            "pose": dict(self.pose),
            "telemetry": {},
        }


def test_deterministic_session_moves_and_records_simulated_events() -> None:
    runtime = SimulationRuntime()
    created = runtime.create({"robotId": "sim-robot-1"})
    session_id = created["session"]["sessionId"]

    runtime.start(session_id)
    goal = runtime.set_goal(
        session_id,
        {
            "goalPose": {
                "frameId": "map",
                "x": 0.5,
                "y": 0.0,
                "z": 0.0,
                "yaw": 0.0,
            },
            "missionId": "mission-sim-1",
            "revisionId": "revision-sim-1",
        },
    )
    assert goal["receipt"]["motion"] == "simulated"
    stepped = runtime.step(session_id, {"distanceM": 0.5})
    assert stepped["snapshot"]["navigationState"] == "arrived"
    assert stepped["snapshot"]["pose"]["x"] == pytest.approx(0.5)

    events = runtime.events(session_id)
    assert [event["eventType"] for event in events] == [
        "simulation.reset",
        "simulation.started",
        "navigation.goal_received",
        "navigation.succeeded",
    ]
    assert all(event["source"] in {"simulator", "nav2"} for event in events)
    runtime.close()


def test_simulation_rejects_isaac_until_an_explicit_backend_is_implemented() -> None:
    runtime = SimulationRuntime()
    with pytest.raises(SimulationConflict, match="isaac backend is not enabled"):
        runtime.create({"backend": "isaac"})
    runtime.close()


def test_gazebo_session_fails_closed_when_host_backend_is_missing(monkeypatch) -> None:
    runtime = SimulationRuntime()
    created = runtime.create({"backend": "gazebo"})
    monkeypatch.setattr("packages.simulation_core.runtime.shutil.which", lambda _: None)
    with pytest.raises(SimulationConflict, match="gazebo backend is staged but not enabled"):
        runtime.start(created["session"]["sessionId"])
    runtime.close()


def test_gazebo_binary_alone_does_not_claim_full_robot_agent_readiness(monkeypatch) -> None:
    runtime = SimulationRuntime()
    monkeypatch.setattr("packages.simulation_core.runtime.shutil.which", lambda _: "/usr/bin/gz")
    gazebo = next(item for item in runtime.backends() if item["backend"] == "gazebo")
    assert gazebo["installed"] is True
    assert gazebo["available"] is False
    runtime.close()


def test_static_obstacle_gets_a_bounded_detour_without_physical_motion() -> None:
    runtime = SimulationRuntime()
    created = runtime.create({})
    session_id = created["session"]["sessionId"]
    runtime.start(session_id)
    runtime.set_goal(
        session_id,
        {
            "goalPose": {
                "frameId": "map",
                "x": 4.0,
                "y": 1.0,
                "z": 0.0,
                "yaw": 0.0,
            }
        },
    )
    planned = runtime.get(session_id)["snapshot"]["plannedWaypoints"]
    assert len(planned) >= 2
    stepped = None
    for _ in range(30):
        stepped = runtime.step(session_id)
        if stepped["receipt"]["status"] == "succeeded":
            break
    assert stepped is not None
    assert stepped["receipt"]["status"] == "succeeded"
    assert stepped["receipt"]["motion"] == "simulated"
    assert stepped["snapshot"]["navigationState"] == "arrived"
    assert stepped["snapshot"]["pose"]["x"] == pytest.approx(4.0)
    runtime.close()


def test_reset_returns_to_the_bound_start_pose() -> None:
    runtime = SimulationRuntime()
    created = runtime.create(
        {
            "startPose": {
                "frameId": "map",
                "x": 1.0,
                "y": -1.0,
                "z": 0.0,
                "yaw": 1.57,
            }
        }
    )
    session_id = created["session"]["sessionId"]
    runtime.start(session_id)
    reset = runtime.reset(session_id)
    assert reset["session"]["status"] == "created"
    assert reset["snapshot"]["pose"]["x"] == pytest.approx(1.0)
    assert reset["snapshot"]["pose"]["y"] == pytest.approx(-1.0)
    runtime.close()


def test_anomaly_creates_a_durable_human_review_marker_without_mission_binding() -> None:
    runtime = SimulationRuntime()
    created = runtime.create({})
    session_id = created["session"]["sessionId"]

    result = runtime.inject_event(
        session_id,
        {"eventType": "vision.target_lost", "payload": {"target": "桌子"}},
    )

    review = result["snapshot"]["pendingReview"]
    assert review["status"] == "pending"
    assert review["decisionKind"] == "needs_review"
    assert review["missionId"] is None
    assert review["triggerEventType"] == "vision.target_lost"
    runtime.close()


def test_mission_binding_is_explicit_and_does_not_authorize_execution() -> None:
    runtime = SimulationRuntime()
    created = runtime.create({})
    session_id = created["session"]["sessionId"]
    bound = runtime.bind_mission(session_id, "mission-sim-1")
    assert bound["session"]["missionId"] == "mission-sim-1"
    assert bound["snapshot"]["activeMissionId"] == "mission-sim-1"
    assert bound["session"]["executionProfile"] == "simulation"
    assert bound["snapshot"]["pendingReview"] is None
    runtime.close()


def test_pending_review_blocks_a_new_navigation_goal_until_reset() -> None:
    runtime = SimulationRuntime()
    created = runtime.create({})
    session_id = created["session"]["sessionId"]
    runtime.start(session_id)
    runtime.inject_event(
        session_id,
        {"eventType": "navigation.blocked", "payload": {"reason": "fixture"}},
    )
    with pytest.raises(SimulationConflict, match="pending Mission review"):
        runtime.set_goal(
            session_id,
            {"goalPose": {"frameId": "map", "x": 0.5, "y": 0, "z": 0, "yaw": 0}},
        )
    runtime.close()


def test_mujoco_session_uses_robot_agent_current_pose_as_start_authority() -> None:
    runtime = SimulationRuntime()
    runtime.set_identity_provider(
        lambda _robot_id: {
            "robotBootId": "sim-boot-1",
            "localizationEpochId": "sim-map-1",
            "observationSessionId": "sim-observation-1",
            "currentPose": {
                "frameId": "map",
                "x": 1.25,
                "y": -0.4,
                "z": 0.78,
                "yaw": 0.2,
                "observedAt": "2026-08-19T00:00:00Z",
            },
        }
    )
    created = runtime.create({"backend": "mujoco_sonic", "robotId": "g1-sim"})
    assert created["snapshot"]["pose"]["x"] == pytest.approx(1.25)
    assert created["snapshot"]["pose"]["y"] == pytest.approx(-0.4)
    assert created["session"]["metadata"]["startPose"]["x"] == pytest.approx(1.25)
    runtime.close()


def test_mujoco_product_snapshot_has_no_semantic_map_or_label_navigation_shortcut() -> None:
    runtime = SimulationRuntime(mujoco_client=_MujocoClientStub())
    dispatched: list[object] = []
    runtime.set_identity_provider(
        lambda _robot_id: {
            "robotBootId": "sim-boot-vlm",
            "localizationEpochId": "sim-map-vlm",
            "currentPose": {
                "frameId": "map",
                "x": 0.0,
                "y": 0.0,
                "z": 0.77,
                "yaw": 0.0,
                "observedAt": "2026-08-19T00:00:00Z",
            },
        }
    )
    runtime.set_navigation_dispatcher(
        lambda session, request: dispatched.append((session, request)) or {}
    )
    created = runtime.create({"backend": "mujoco_sonic", "robotId": "g1-sim"})
    session_id = created["session"]["sessionId"]
    snapshot = created["snapshot"]
    assert snapshot["landmarks"] == []
    assert snapshot["obstacles"] == []
    assert snapshot["plannedWaypoints"] == []

    runtime.start(session_id)
    with pytest.raises(ValidationError):
        runtime.set_goal(session_id, {"semanticTarget": "厨房"})
    assert dispatched == []

    live = runtime.live(session_id)
    assert live["snapshot"]["landmarks"] == []
    assert live["snapshot"]["obstacles"] == []
    assert live["snapshot"]["plannedWaypoints"] == []
    runtime.close()


def test_mujoco_session_rejects_missing_robot_agent_pose() -> None:
    runtime = SimulationRuntime()
    runtime.set_identity_provider(
        lambda _robot_id: {
            "robotBootId": "sim-boot-1",
            "localizationEpochId": "sim-map-1",
        }
    )
    with pytest.raises(SimulationConflict, match="current pose is unavailable"):
        runtime.create({"backend": "mujoco_sonic", "robotId": "g1-sim"})
    runtime.close()


def _configured_mujoco_runtime(status: dict[str, str]) -> tuple[SimulationRuntime, str]:
    runtime = SimulationRuntime(mujoco_client=_MujocoClientStub())
    runtime.set_identity_provider(
        lambda _robot_id: {
            "robotBootId": "sim-boot-1",
            "localizationEpochId": "sim-map-1",
            "currentPose": {
                "frameId": "map",
                "x": 0.0,
                "y": 0.0,
                "z": 0.77,
                "yaw": 0.0,
                "observedAt": "2026-08-19T00:00:00Z",
            },
        }
    )
    runtime.set_navigation_dispatcher(
        lambda _session, _request: {
            "status": "accepted",
            "commandId": "robot-agent-command-1",
            "executionBackend": "robot_agent_nav2_sonic",
        }
    )
    runtime.set_navigation_status_provider(
        lambda command_id: {
            "status": status["value"],
            "commandId": command_id,
            "detail": f"Robot Agent reported {status['value']}",
            "result": {
                "robotAgentCommandId": command_id,
                "finalPose": {
                    "frameId": "map",
                    "x": 0.29,
                    "y": -0.01,
                    "z": 0.77,
                    "yaw": 0.02,
                    "observedAt": "2026-08-19T00:00:03Z",
                },
            },
        }
    )
    created = runtime.create({"backend": "mujoco_sonic", "robotId": "g1-sim"})
    session_id = created["session"]["sessionId"]
    runtime.start(session_id)
    runtime.set_goal(
        session_id,
        {"goalPose": {"frameId": "map", "x": 0.3, "y": 0.0, "z": 0.0, "yaw": 0.0}},
    )
    return runtime, session_id


def test_mujoco_live_reconciles_success_once_into_the_durable_event_ledger() -> None:
    status = {"value": "succeeded"}
    runtime, session_id = _configured_mujoco_runtime(status)

    first = runtime.live(session_id)
    assert first["snapshot"]["navigationState"] == "arrived"
    assert first["snapshot"]["goalPose"] is None
    assert first["snapshot"]["pose"]["x"] == pytest.approx(0.29)
    assert first["session"]["metadata"].get("activeNavigationCommandId") is None
    assert first["session"]["metadata"]["lastNavigationCommandId"] == "robot-agent-command-1"
    assert runtime.get(session_id)["snapshot"]["navigationState"] == "arrived"

    second = runtime.live(session_id)
    assert second["live"]["navigation"]["status"] == "succeeded"
    events = runtime.events(session_id)
    assert [item["eventType"] for item in events].count("navigation.succeeded") == 1
    succeeded = next(item for item in events if item["eventType"] == "navigation.succeeded")
    assert succeeded["payload"]["poseSource"] == "robot_agent_terminal_receipt"
    runtime.close()


def test_mujoco_live_reconciles_failure_once_and_requires_human_review() -> None:
    status = {"value": "failed"}
    runtime, session_id = _configured_mujoco_runtime(status)

    failed = runtime.live(session_id)
    assert failed["snapshot"]["navigationState"] == "failed"
    assert failed["snapshot"]["pendingReview"]["status"] == "pending"
    runtime.live(session_id)
    events = runtime.events(session_id)
    assert [item["eventType"] for item in events].count("navigation.failed") == 1
    runtime.close()


def test_mujoco_live_waits_for_the_first_post_terminal_pose_before_persisting() -> None:
    status = {"value": "succeeded"}
    runtime, session_id = _configured_mujoco_runtime(status)
    receipt = {
        "status": "succeeded",
        "commandId": "robot-agent-command-1",
        "detail": "terminal received; pose handoff pending",
        "result": {
            "terminalReceivedAt": datetime.now(UTC).isoformat(),
            "finalPoseRelation": "latest_before_terminal",
            "finalPose": {
                "frameId": "map",
                "x": 0.22,
                "y": 0.0,
                "z": 0.77,
                "yaw": 0.0,
                "observedAt": datetime.now(UTC).isoformat(),
            },
        },
    }
    runtime.set_navigation_status_provider(lambda _command_id: receipt)

    pending = runtime.live(session_id)
    assert pending["snapshot"]["navigationState"] == "moving"
    assert not any(
        item["eventType"] == "navigation.succeeded"
        for item in runtime.events(session_id)
    )

    receipt["result"]["finalPoseRelation"] = "stable_after_terminal"
    receipt["result"]["finalPose"]["x"] = 0.3
    completed = runtime.live(session_id)
    assert completed["snapshot"]["navigationState"] == "arrived"
    assert completed["snapshot"]["pose"]["x"] == pytest.approx(0.3)
    runtime.close()
