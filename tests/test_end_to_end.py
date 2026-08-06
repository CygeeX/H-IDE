"""Cross-layer acceptance tests for the workstation control plane.

These tests deliberately exercise the public BFF contract rather than calling
the runtime directly.  They model the paths used by the manual controls and
the PWA's Agent/interface views, while keeping the adapter deterministic.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time
import warnings

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
from fastapi.testclient import TestClient

from packages.control_core.adapters import AdapterResult, FakeAdapter
from services.control_api.main import create_app


TOKEN = "e2e-token"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


def command_body(**overrides):
    body = {
        "commandId": "e2e-command",
        "robotId": "robot-1",
        "commandKey": "move.forward",
        "commandVersion": "1",
        "transport": "fake",
        "payload": {},
        "idempotencyKey": "e2e-idempotency",
    }
    body.update(overrides)
    return body


def mapping_body(mapping_id: str, version: str, topic: str):
    return {
        "mappingId": mapping_id,
        "version": version,
        "commandKey": "move.forward",
        "transport": "fake",
        "targetId": "robot-1",
        "mapping": {"targetId": "robot-1", "topic": topic},
    }


class RecordingAdapter(FakeAdapter):
    """A fake transport that exposes the logical requests it receives."""

    def __init__(self, *results: AdapterResult):
        super().__init__()
        self.results = list(results)

    def dispatch(self, request, mapping):
        self.dispatches.append((request, mapping))
        return self.results.pop(0) if self.results else AdapterResult(status="succeeded")


class RunningAdapter(FakeAdapter):
    """Leaves motion in flight so lease, restart, and cancellation paths run."""

    def dispatch(self, request, mapping):
        self.dispatches.append((request, mapping))
        return AdapterResult(status="running")


class VpnLostAdapter(FakeAdapter):
    def dispatch(self, request, mapping):
        self.dispatches.append((request, mapping))
        return AdapterResult(status="unavailable", error="VPN link unavailable")


class ChangingProbeAdapter(RecordingAdapter):
    def __init__(self):
        super().__init__()
        self.probes = [
            AdapterResult(status="probed", details={"capability": "v1"}),
            AdapterResult(status="probed", details={"capability": "v2"}),
        ]

    def probe(self, mapping):
        return self.probes.pop(0) if self.probes else AdapterResult(status="probed")


class SafetyIndependentAdapter(FakeAdapter):
    """A failed task or hanging task must not remove the stop/cancel path."""

    def __init__(self, *, motion_running: bool):
        super().__init__()
        self.motion_running = motion_running

    def dispatch(self, request, mapping):
        self.dispatches.append((request, mapping))
        if request.commandKey == "stop":
            return AdapterResult(status="succeeded", result={"safety": "accepted"})
        if self.motion_running:
            return AdapterResult(status="running")
        return AdapterResult(status="failed", error="scene/video service unavailable")

    def cancel(self, command_id):
        self.cancellations.append(command_id)
        return AdapterResult(status="cancelled")


def complete_mapping(client: TestClient, mapping_id: str, version: str, topic: str):
    created = client.post(
        "/api/v1/mappings/drafts",
        headers=auth(),
        json=mapping_body(mapping_id, version, topic),
    )
    assert created.status_code == 200, created.text
    for action in ("validate", "probe", "activate"):
        response = client.post(
            f"/api/v1/mappings/{mapping_id}/{version}/{action}",
            headers=auth(),
        )
        assert response.status_code == 200, response.text


def test_manual_and_agent_plan_share_one_runtime_and_adapter():
    adapter = RecordingAdapter()
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter})
    client = TestClient(app)

    manual = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="manual-e2e", idempotencyKey="manual-e2e-key"),
    )
    assert manual.status_code == 200, manual.text
    assert manual.json()["status"] == "succeeded"

    draft = client.post(
        "/api/v1/assistant/drafts",
        headers=auth(),
        json={
            "draftId": "agent-e2e-plan",
            "robotId": "robot-1",
            "steps": [
                {"stepId": "move", "commandKey": "move.forward", "payload": {}},
                {
                    "stepId": "gripper",
                    "commandKey": "gripper.open",
                    "payload": {},
                    "dependsOn": ["move"],
                },
            ],
        },
    )
    assert draft.status_code == 200, draft.text
    assert len(adapter.dispatches) == 1
    assert client.get("/api/v1/assistant/drafts/agent-e2e-plan", headers=auth()).json()["status"] == "draft"

    confirmed = client.post(
        "/api/v1/assistant/drafts/agent-e2e-plan/confirm",
        headers=auth(),
        json={"confirmed": True},
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["status"] == "confirmed"
    assert [request.commandKey for request, _ in adapter.dispatches] == [
        "move.forward",
        "move.forward",
        "gripper.open",
    ]
    # Both ingress paths leave their requests and receipts in the same runtime
    # owned by the app; there is no second execution path for Agent commands.
    runtime_ids = set(app.state.runtime._requests)
    assert manual.json()["commandId"] in runtime_ids
    assert {item[0].commandId for item in adapter.dispatches[1:]} <= runtime_ids


def test_mapping_v1_v2_can_change_without_frontend_code_and_roll_back():
    adapter = RecordingAdapter()
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    complete_mapping(client, "ui-contract-map", "v1", "/move/v1")
    complete_mapping(client, "ui-contract-map", "v2", "/move/v2")

    v2_command = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="mapping-v2-command", idempotencyKey="mapping-v2-key"),
    )
    assert v2_command.status_code == 200, v2_command.text
    assert adapter.dispatches[-1][1].version == "v2"
    assert adapter.dispatches[-1][1].mapping["topic"] == "/move/v2"

    rolled_back = client.post(
        "/api/v1/mappings/move.forward/rollback",
        headers=auth(),
        json={"version": "v1"},
    )
    assert rolled_back.status_code == 200, rolled_back.text
    assert rolled_back.json()["mapping"]["version"] == "v1"

    v1_command = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="mapping-v1-command", idempotencyKey="mapping-v1-key"),
    )
    assert v1_command.status_code == 200, v1_command.text
    assert adapter.dispatches[-1][1].version == "v1"
    assert adapter.dispatches[-1][1].mapping["topic"] == "/move/v1"


def test_sequential_capability_probe_refresh_uses_snapshot_cas():
    adapter = ChangingProbeAdapter()
    app = create_app(auth_token=TOKEN)
    # Do not consume the deterministic probe sequence while the app seeds its
    # built-in mappings; this adapter is specifically for the user mapping.
    app.state.control.registry.adapters["fake"] = adapter
    client = TestClient(app)
    created = client.post(
        "/api/v1/mappings/drafts",
        headers=auth(),
        json=mapping_body("probe-refresh", "v1", "/move/refresh"),
    )
    assert created.status_code == 200
    assert client.post("/api/v1/mappings/probe-refresh/v1/validate", headers=auth()).status_code == 200
    first = client.post("/api/v1/mappings/probe-refresh/v1/probe", headers=auth())
    assert first.status_code == 200, first.text
    assert first.json()["probe"]["details"]["capability"] == "v1"
    second = client.post("/api/v1/mappings/probe-refresh/v1/probe", headers=auth())
    assert second.status_code == 200, second.text
    assert second.json()["probe"]["details"]["capability"] == "v2"


def test_malformed_robot_state_fails_closed_but_stop_remains_available():
    adapter = RecordingAdapter()
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter})
    app.state.store.upsert_robot(
        {
            "robotId": "robot-1",
            "connected": "yes",
            "status": "online",
            "safetyMode": "normal",
        }
    )
    client = TestClient(app)
    motion = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="malformed-motion", idempotencyKey="malformed-motion-key"),
    )
    assert motion.status_code == 409
    assert adapter.dispatches == []
    stop = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="malformed-stop", idempotencyKey="malformed-stop-key", commandKey="stop"),
    )
    assert stop.status_code == 200
    assert stop.json()["status"] == "succeeded"


def test_corrupt_active_pointer_never_promotes_a_draft_after_restart(tmp_path):
    db_path = str(tmp_path / "pointer-corruption.db")
    first_app = create_app(db_path=db_path, auth_token=TOKEN)
    first = TestClient(first_app)
    draft = first.post(
        "/api/v1/mappings/drafts",
        headers=auth(),
        json=mapping_body("unsafe-pointer", "v1", "/unsafe"),
    )
    assert draft.status_code == 200
    first_app.state.store._connection.execute(
        "UPDATE mapping_active SET mapping_id = ?, version = ? WHERE command_key = ?",
        ("unsafe-pointer", "v1", "move.forward"),
    )
    first_app.state.store._connection.commit()

    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN))
    active = restarted.get("/api/v1/mappings/active/move.forward", headers=auth())
    assert active.status_code == 200
    assert active.json()["mappingId"] == "seed-move-forward"
    assert active.json()["mapping"]["topic"] == "/move.forward"


def test_expiry_duplicate_and_late_messages_never_replay():
    adapter = RunningAdapter()
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter}, watchdog_interval=0.01)
    app.state.runtime.lease_seconds = 0.04
    client = TestClient(app)
    original_body = command_body(commandId="expiry-e2e", idempotencyKey="expiry-e2e-key")
    first = client.post("/api/v1/commands", headers=auth(), json=original_body)
    assert first.status_code == 200, first.text
    assert first.json()["status"] == "running"

    deadline = time.monotonic() + 1.5
    expired = None
    while time.monotonic() < deadline:
        expired = client.get("/api/v1/commands/expiry-e2e", headers=auth()).json()
        if expired["status"] == "expired":
            break
        time.sleep(0.02)
    assert expired is not None and expired["status"] == "expired"

    duplicate = client.post(
        "/api/v1/commands",
        headers=auth(),
        json={**original_body, "payload": {"lateRetry": True}},
    )
    assert duplicate.status_code == 200
    assert duplicate.json() == expired
    assert len(adapter.dispatches) == 1

    late = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(
            commandId="late-new-command",
            idempotencyKey="late-new-key",
            expiresAt=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        ),
    )
    assert late.status_code in {400, 422}
    assert len(adapter.dispatches) == 1


def test_adapter_restart_and_vpn_loss_do_not_trigger_automatic_replay(tmp_path):
    db_path = str(tmp_path / "reconciliation-e2e.db")
    first_adapter = RunningAdapter()
    first = TestClient(create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": first_adapter}))
    body = command_body(commandId="restart-e2e", idempotencyKey="restart-e2e-key")
    submitted = first.post("/api/v1/commands", headers=auth(), json=body)
    assert submitted.status_code == 200
    assert submitted.json()["status"] == "running"

    restarted_adapter = RecordingAdapter()
    restarted_app = create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": restarted_adapter})
    restarted = TestClient(restarted_app)
    persisted = restarted.get("/api/v1/commands/restart-e2e", headers=auth()).json()
    assert persisted["status"] == "running"
    assert "reconciliation" in (persisted.get("error") or "").lower()
    retry = restarted.post("/api/v1/commands", headers=auth(), json={**body, "payload": {"replayed": True}})
    assert retry.status_code == 200
    assert retry.json() == persisted
    assert restarted_adapter.dispatches == []
    cancel = restarted.post("/api/v1/commands/restart-e2e/cancel", headers=auth())
    assert cancel.status_code == 409
    assert "reconciliation" in cancel.text.lower()

    vpn_adapter = VpnLostAdapter()
    vpn_client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": vpn_adapter}))
    vpn_body = command_body(commandId="vpn-e2e", idempotencyKey="vpn-e2e-key")
    failed = vpn_client.post("/api/v1/commands", headers=auth(), json=vpn_body)
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    retried = vpn_client.post("/api/v1/commands", headers=auth(), json={**vpn_body, "payload": {"replayed": True}})
    assert retried.status_code == 200
    assert retried.json() == failed.json()
    assert len(vpn_adapter.dispatches) == 1


def test_failed_optional_work_does_not_block_stop_or_cancel():
    failing_task_adapter = SafetyIndependentAdapter(motion_running=False)
    failed_client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": failing_task_adapter}))
    failed = failed_client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="scene-failure", idempotencyKey="scene-failure-key"),
    )
    assert failed.status_code == 200
    assert failed.json()["status"] == "failed"
    stop = failed_client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="scene-failure-stop", idempotencyKey="scene-failure-stop-key", commandKey="stop"),
    )
    assert stop.status_code == 200
    assert stop.json()["status"] == "succeeded"

    running_adapter = SafetyIndependentAdapter(motion_running=True)
    running_client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": running_adapter}))
    running = running_client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="cancel-e2e", idempotencyKey="cancel-e2e-key"),
    )
    assert running.status_code == 200
    assert running.json()["status"] == "running"
    cancelled = running_client.post("/api/v1/commands/cancel-e2e/cancel", headers=auth())
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert running_adapter.cancellations == ["cancel-e2e"]
    # Stop remains an independent safety command even after cancellation.
    stop_after_cancel = running_client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="cancel-e2e-stop", idempotencyKey="cancel-e2e-stop-key", commandKey="stop"),
    )
    assert stop_after_cancel.status_code == 200
    assert stop_after_cancel.json()["status"] == "succeeded"


def test_factory_uses_control_db_path_for_deployment(monkeypatch, tmp_path):
    db_path = str(tmp_path / "from-env.db")
    monkeypatch.setenv("CONTROL_DB_PATH", db_path)
    app = create_app(auth_token=TOKEN)
    assert app.state.store.db_path == db_path
    client = TestClient(app)
    assert client.get("/api/v1/robots", headers=auth()).status_code == 200
