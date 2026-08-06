from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import warnings

import pytest
from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
from fastapi.testclient import TestClient

from packages.control_core.adapters import AdapterResult, FakeAdapter
from services.control_api.main import create_app
from services.control_api.store import SQLiteStore


TOKEN = "test-token"


class _SequenceAdapter(FakeAdapter):
    """Deterministic adapter used to make a plan fail after one step."""

    def __init__(self, results):
        super().__init__()
        self._results = list(results)
        self._lock = threading.Lock()

    def dispatch(self, request, mapping):
        self.dispatches.append((request, mapping))
        with self._lock:
            if self._results:
                return self._results.pop(0)
        return AdapterResult(status="succeeded")


class _SlowCountingAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def dispatch(self, request, mapping):
        with self._lock:
            self.dispatches.append((request, mapping))
        time.sleep(0.08)
        return AdapterResult(status="succeeded")


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def command_body(**overrides):
    body = {
        "robotId": "robot-1",
        "commandKey": "move.forward",
        "commandVersion": "1",
        "transport": "fake",
        "payload": {},
        "idempotencyKey": "api-idem-1",
    }
    body.update(overrides)
    return body


def mapping_body(mapping_id="api-map", version="v1", command_key="move.forward"):
    return {
        "mappingId": mapping_id,
        "version": version,
        "commandKey": command_key,
        "transport": "fake",
        "mapping": {"topic": f"/{command_key}"},
    }


def test_bearer_auth_and_seeded_robot():
    api_app = create_app(auth_token=TOKEN)
    assert api_app.state.store is api_app.state.control.store
    assert api_app.state.runtime is api_app.state.control.runtime
    client = TestClient(api_app)
    assert client.get("/api/v1/robots").status_code == 401
    assert client.get("/api/v1/robots", headers={"Authorization": "Bearer wrong"}).status_code == 401
    response = client.get("/api/v1/robots", headers=auth())
    assert response.status_code == 200
    assert any(robot["robotId"] == "robot-1" for robot in response.json())
    capabilities = client.get("/api/v1/robots/robot-1/capabilities", headers=auth()).json()
    assert capabilities["schema"] == {"type": "object"}


def test_manual_command_receipt_and_restart_persistence(tmp_path):
    db_path = str(tmp_path / "control.db")
    app = create_app(db_path=db_path, auth_token=TOKEN)
    client = TestClient(app)
    response = client.post("/api/v1/commands", headers=auth(), json=command_body())
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["commandId"]
    assert receipt["status"] == "succeeded"
    command_id = receipt["commandId"]
    fetched = client.get(f"/api/v1/commands/{command_id}", headers=auth())
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "succeeded"
    assert client.get("/api/v1/commands", headers=auth()).json()[0]["commandId"] == command_id

    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN))
    persisted = restarted.get(f"/api/v1/commands/{command_id}", headers=auth())
    assert persisted.status_code == 200
    assert persisted.json()["status"] == "succeeded"


def test_idempotency_retry_keeps_original_persisted_request():
    adapter = FakeAdapter()
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter})
    client = TestClient(app)
    first = client.post("/api/v1/commands", headers=auth(), json=command_body(idempotencyKey="same-key"))
    second = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="same-key", payload={"changed": True}),
    )
    assert first.status_code == second.status_code == 200
    assert second.json()["commandId"] == first.json()["commandId"]
    assert len(adapter.dispatches) == 1
    record = app.state.control.store.get_command_record(first.json()["commandId"])
    assert record["request"]["commandId"] == first.json()["commandId"]


def test_expired_manual_command_is_controlled_client_error():
    client = TestClient(create_app(auth_token=TOKEN))
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    response = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="expired", expiresAt=expired),
    )
    assert response.status_code in {400, 422}
    assert "expires" in response.text.lower()


def test_cancel_uses_runtime_and_persists_terminal_receipt():
    adapter = FakeAdapter(result=AdapterResult(status="running"))
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    submitted = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="cancel-me"),
    )
    assert submitted.status_code == 200
    command_id = submitted.json()["commandId"]
    cancelled = client.post(f"/api/v1/commands/{command_id}/cancel", headers=auth())
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert adapter.cancellations == [command_id]


def test_assistant_draft_requires_explicit_confirmation_before_dispatch():
    adapter = FakeAdapter()
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter})
    client = TestClient(app)
    draft = client.post(
        "/api/v1/assistant/drafts",
        headers=auth(),
        json={
            "draftId": "plan-1",
            "robotId": "robot-1",
            "steps": [{"stepId": "step-1", "commandKey": "move.forward", "payload": {}}],
        },
    )
    assert draft.status_code == 200
    assert draft.json()["status"] == "draft"
    assert adapter.dispatches == []
    confirmed = client.post("/api/v1/assistant/drafts/plan-1/confirm", headers=auth())
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "confirmed"
    assert len(adapter.dispatches) == 1
    assert confirmed.json()["receipts"][0]["status"] == "succeeded"
    assert client.post("/api/v1/assistant/drafts/plan-1/confirm", headers=auth()).status_code == 409


def test_assistant_draft_id_cannot_be_overwritten():
    client = TestClient(create_app(auth_token=TOKEN))
    body = {
        "draftId": "immutable-plan",
        "robotId": "robot-1",
        "steps": [{"stepId": "step-1", "commandKey": "move.forward", "payload": {}}],
    }
    assert client.post("/api/v1/assistant/drafts", headers=auth(), json=body).status_code == 200
    replacement = dict(body)
    replacement["steps"] = [{"stepId": "different", "commandKey": "stop", "payload": {}}]
    assert client.post("/api/v1/assistant/drafts", headers=auth(), json=replacement).status_code == 409


def test_assistant_draft_claim_is_compare_and_set_under_competing_confirmations():
    store = SQLiteStore()
    store.save_assistant_draft("cas-plan", {"planId": "cas-plan", "steps": []})
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait(timeout=2)
        return store.claim_assistant_draft("cas-plan")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sorted(results) == [False, True]
    assert store.get_assistant_draft("cas-plan")["status"] == "confirming"
    store.close()


def test_manual_endpoint_cannot_label_unconfirmed_agent_execution():
    adapter = FakeAdapter()
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    response = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="unconfirmed-agent", source="agent"),
    )
    assert response.status_code == 422
    assert adapter.dispatches == []


def test_mapping_lifecycle_and_latest_version_rollback():
    client = TestClient(create_app(auth_token=TOKEN))
    for mapping_id in ("api-m1", "api-m2"):
        body = mapping_body(mapping_id=mapping_id)
        body["targetId"] = "robot-1"
        assert client.post("/api/v1/mappings/drafts", headers=auth(), json=body).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/validate", headers=auth()).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/probe", headers=auth()).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/activate", headers=auth()).status_code == 200
    active = client.get("/api/v1/mappings/active/move.forward", headers=auth())
    assert active.status_code == 200
    assert active.json()["mappingId"] == "api-m2"
    rows = {row["mapping"]["mappingId"]: row for row in client.get("/api/v1/mappings", headers=auth()).json()}
    assert rows["api-m1"]["status"] == "superseded"
    rolled = client.post(
        "/api/v1/mappings/move.forward/rollback",
        headers=auth(),
        json={"version": "v1"},
    )
    assert rolled.status_code == 200
    assert rolled.json()["mapping"]["mappingId"] == "api-m1"


def test_command_cannot_dispatch_against_non_active_explicit_mapping():
    adapter = FakeAdapter()
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    body = mapping_body(mapping_id="draft-only")
    assert client.post("/api/v1/mappings/drafts", headers=auth(), json=body).status_code == 200
    response = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="draft-only-command", mappingId="draft-only", mappingVersion="v1"),
    )
    assert response.status_code in {409, 422}
    assert adapter.dispatches == []


def test_mapping_active_pointer_and_rollback_survive_restart(tmp_path):
    db_path = str(tmp_path / "mappings.db")
    client = TestClient(create_app(db_path=db_path, auth_token=TOKEN))
    for mapping_id in ("z-old", "a-new"):
        body = mapping_body(mapping_id=mapping_id)
        assert client.post("/api/v1/mappings/drafts", headers=auth(), json=body).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/validate", headers=auth()).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/probe", headers=auth()).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/activate", headers=auth()).status_code == 200
    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN))
    assert restarted.get("/api/v1/mappings/active/move.forward", headers=auth()).json()["mappingId"] == "a-new"
    rollback = restarted.post(
        "/api/v1/mappings/move.forward/rollback",
        headers=auth(),
        json={"version": "v1"},
    )
    assert rollback.status_code == 200
    assert rollback.json()["mapping"]["mappingId"] == "z-old"


def test_concurrent_mapping_activate_keeps_registry_and_sqlite_pointer_consistent(monkeypatch):
    client = TestClient(create_app(auth_token=TOKEN))
    for mapping_id in ("race-b", "race-c"):
        body = mapping_body(mapping_id=mapping_id)
        assert client.post("/api/v1/mappings/drafts", headers=auth(), json=body).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/validate", headers=auth()).status_code == 200
        assert client.post(f"/api/v1/mappings/{mapping_id}/v1/probe", headers=auth()).status_code == 200

    context = client.app.state.control
    original_activate = context.registry.activate
    entered_b = threading.Event()
    release_b = threading.Event()

    def activate(mapping_id, version):
        record = original_activate(mapping_id, version)
        if mapping_id == "race-b":
            entered_b.set()
            assert release_b.wait(timeout=3)
        return record

    monkeypatch.setattr(context.registry, "activate", activate)
    responses = {}

    def activate_one(mapping_id):
        responses[mapping_id] = client.post(
            f"/api/v1/mappings/{mapping_id}/v1/activate", headers=auth()
        )

    first = threading.Thread(target=activate_one, args=("race-b",))
    first.start()
    assert entered_b.wait(timeout=3)
    second = threading.Thread(target=activate_one, args=("race-c",))
    second.start()
    # Let C finish its transition and pointer commit before B is allowed to
    # persist its stale ``previous`` snapshot.
    time.sleep(0.05)
    release_b.set()
    first.join(timeout=3)
    second.join(timeout=3)
    assert not first.is_alive() and not second.is_alive()
    assert all(response.status_code == 200 for response in responses.values())
    memory_active = context.registry.active("move.forward").mappingId
    sqlite_active = context.store.get_active_pointer("move.forward")[0]
    assert memory_active == sqlite_active


def test_command_rejects_mapping_target_bound_to_another_robot():
    adapter = FakeAdapter()
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    body = mapping_body(mapping_id="other-target")
    body["mapping"] = {"targetId": "robot-2", "topic": "/move"}
    assert client.post("/api/v1/mappings/drafts", headers=auth(), json=body).status_code == 200
    assert client.post("/api/v1/mappings/other-target/v1/validate", headers=auth()).status_code == 200
    assert client.post("/api/v1/mappings/other-target/v1/probe", headers=auth()).status_code == 200
    assert client.post("/api/v1/mappings/other-target/v1/activate", headers=auth()).status_code == 200
    response = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="wrong-target", mappingId="other-target", mappingVersion="v1"),
    )
    assert response.status_code == 422
    assert adapter.dispatches == []


def test_command_rejects_explicit_mapping_target_bound_to_another_robot():
    adapter = FakeAdapter()
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    body = mapping_body(mapping_id="explicit-other-target")
    body["targetId"] = "robot-2"
    assert client.post("/api/v1/mappings/drafts", headers=auth(), json=body).status_code == 200
    for action in ("validate", "probe", "activate"):
        assert client.post(f"/api/v1/mappings/explicit-other-target/v1/{action}", headers=auth()).status_code == 200
    response = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="wrong-explicit-target", mappingId="explicit-other-target", mappingVersion="v1"),
    )
    assert response.status_code == 422
    assert adapter.dispatches == []


def test_websocket_requires_auth_and_replays_persisted_events():
    app = create_app(auth_token=TOKEN)
    client = TestClient(app)
    response = client.post("/api/v1/commands", headers=auth(), json=command_body())
    assert response.status_code == 200
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/events?token=wrong"):
            pass
    with client.websocket_connect("/api/v1/events?token=test-token") as websocket:
        event = websocket.receive_json()
        assert event["commandId"] == response.json()["commandId"]


def test_store_events_are_written_during_concurrent_requests(tmp_path):
    app = create_app(db_path=str(tmp_path / "events.db"), auth_token=TOKEN)
    client = TestClient(app)
    responses = []

    def submit(index):
        responses.append(
            client.post(
                "/api/v1/commands",
                headers=auth(),
                json=command_body(idempotencyKey=f"parallel-{index}"),
            )
        )

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)
    assert all(response.status_code == 200 for response in responses)
    events = client.get("/api/v1/events", headers=auth()).json()
    assert len(events) >= 5
    assert len({event["commandId"] for event in events}) == 3


def test_idempotency_is_durable_across_restart(tmp_path):
    db_path = str(tmp_path / "idempotency.db")
    first_adapter = FakeAdapter()
    first = TestClient(create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": first_adapter}))
    original = first.post("/api/v1/commands", headers=auth(), json=command_body(idempotencyKey="restart-key"))
    assert original.status_code == 200

    second_adapter = FakeAdapter()
    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": second_adapter}))
    retry = restarted.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="restart-key", payload={"mustNotRunAgain": True}),
    )
    assert retry.status_code == 200
    assert retry.json() == original.json()
    assert second_adapter.dispatches == []


def test_same_idempotency_key_is_claimed_once_under_concurrency(tmp_path):
    adapter = _SlowCountingAdapter()
    app = create_app(db_path=str(tmp_path / "concurrent-idem.db"), auth_token=TOKEN, adapters={"fake": adapter})
    client = TestClient(app)
    body = command_body(idempotencyKey="one-owner")

    def submit():
        return client.post("/api/v1/commands", headers=auth(), json=body)

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: submit(), range(2)))
    assert [response.status_code for response in responses] == [200, 200]
    assert len(adapter.dispatches) == 1
    assert len({response.json()["commandId"] for response in responses}) == 1


def test_command_id_conflict_never_dispatches_twice():
    adapter = FakeAdapter()
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    first = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="fixed-command", idempotencyKey="fixed-key"),
    )
    assert first.status_code == 200
    conflict = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandId="fixed-command", idempotencyKey="different-key"),
    )
    assert conflict.status_code == 409
    assert len(adapter.dispatches) == 1


def test_assistant_partial_failure_is_durable_and_not_replayed():
    adapter = _SequenceAdapter(
        [AdapterResult(status="succeeded", result={"step": 1}), AdapterResult(status="failed", error="step failed")]
    )
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter})
    client = TestClient(app)
    draft = client.post(
        "/api/v1/assistant/drafts",
        headers=auth(),
        json={
            "draftId": "partial-plan",
            "robotId": "robot-1",
            "steps": [
                {"stepId": "first", "commandKey": "move.forward", "payload": {}},
                {"stepId": "second", "commandKey": "gripper.open", "payload": {}},
            ],
        },
    )
    assert draft.status_code == 200
    failed = client.post("/api/v1/assistant/drafts/partial-plan/confirm", headers=auth())
    assert failed.status_code == 409
    stored = client.get("/api/v1/assistant/drafts/partial-plan", headers=auth()).json()
    assert stored["status"] == "partial"
    assert len(stored["execution"]["steps"]) == 2
    assert len(adapter.dispatches) == 2
    assert client.post("/api/v1/assistant/drafts/partial-plan/confirm", headers=auth()).status_code == 409
    assert len(adapter.dispatches) == 2


def test_assistant_resolution_failure_is_terminal_not_replayable():
    client = TestClient(create_app(auth_token=TOKEN))
    created = client.post(
        "/api/v1/assistant/drafts",
        headers=auth(),
        json={
            "draftId": "unresolvable-plan",
            "robotId": "robot-1",
            "steps": [{"stepId": "missing", "commandKey": "does.not.exist", "payload": {}}],
        },
    )
    assert created.status_code == 200
    assert client.post("/api/v1/assistant/drafts/unresolvable-plan/confirm", headers=auth()).status_code == 409
    stored = client.get("/api/v1/assistant/drafts/unresolvable-plan", headers=auth()).json()
    assert stored["status"] == "failed"
    assert client.post("/api/v1/assistant/drafts/unresolvable-plan/confirm", headers=auth()).status_code == 409


def test_assistant_non_terminal_receipt_stops_plan_as_unknown():
    adapter = _SequenceAdapter([AdapterResult(status="running"), AdapterResult(status="succeeded")])
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    created = client.post(
        "/api/v1/assistant/drafts",
        headers=auth(),
        json={
            "draftId": "running-plan",
            "robotId": "robot-1",
            "steps": [
                {"stepId": "first", "commandKey": "move.forward", "payload": {}},
                {"stepId": "second", "commandKey": "gripper.open", "payload": {}, "dependsOn": ["first"]},
            ],
        },
    )
    assert created.status_code == 200
    response = client.post("/api/v1/assistant/drafts/running-plan/confirm", headers=auth())
    assert response.status_code == 409
    stored = client.get("/api/v1/assistant/drafts/running-plan", headers=auth()).json()
    assert stored["status"] == "unknown"
    assert [step["stepId"] for step in stored["execution"]["steps"]] == ["first"]
    assert len(adapter.dispatches) == 1
    assert client.post("/api/v1/assistant/drafts/running-plan/confirm", headers=auth()).status_code == 409
    assert len(adapter.dispatches) == 1


def test_restart_skips_corrupt_mapping_payload_rows(tmp_path):
    db_path = str(tmp_path / "corrupt-mapping.db")
    first = create_app(db_path=db_path, auth_token=TOKEN)
    store = first.state.store
    store._connection.execute("DELETE FROM mappings")
    store._connection.execute(
        "INSERT INTO mappings(mapping_id, version, payload_json, status, target_id, probe_json, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("corrupt-list", "v1", "[]", "active", None, None, store._now()),
    )
    store._connection.execute(
        "INSERT INTO mappings(mapping_id, version, payload_json, status, target_id, probe_json, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("corrupt-object", "v1", '{"mappingId":"corrupt-object"}', "active", None, None, store._now()),
    )
    # Exercise the pointer backfill path used by an older database.
    store._connection.execute("DELETE FROM mapping_active")
    store._connection.commit()
    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN))
    active = restarted.get("/api/v1/mappings/active/move.forward", headers=auth())
    assert active.status_code == 200
    assert active.json()["mappingId"].startswith("seed-")


def test_restart_recovery_marks_confirming_plan_unknown(tmp_path):
    db_path = str(tmp_path / "confirming.db")
    first = create_app(db_path=db_path, auth_token=TOKEN)
    first.state.store.save_assistant_draft(
        "interrupted-plan",
        {
            "planId": "interrupted-plan",
            "robotId": "robot-1",
            "steps": [{"stepId": "step", "commandKey": "move.forward", "payload": {}}],
        },
    )
    assert first.state.store.claim_assistant_draft("interrupted-plan")
    restarted_adapter = FakeAdapter()
    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": restarted_adapter}))
    recovered = restarted.get("/api/v1/assistant/drafts/interrupted-plan", headers=auth()).json()
    assert recovered["status"] == "unknown"
    assert restarted.post("/api/v1/assistant/drafts/interrupted-plan/confirm", headers=auth()).status_code == 409
    assert restarted_adapter.dispatches == []


def test_restart_running_command_cancel_is_explicitly_unavailable(tmp_path):
    db_path = str(tmp_path / "running.db")
    running_adapter = FakeAdapter(result=AdapterResult(status="running"))
    first = TestClient(create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": running_adapter}))
    submitted = first.post("/api/v1/commands", headers=auth(), json=command_body(idempotencyKey="running-restart"))
    assert submitted.status_code == 200
    command_id = submitted.json()["commandId"]
    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": FakeAdapter()}))
    cancelled = restarted.post(f"/api/v1/commands/{command_id}/cancel", headers=auth())
    assert cancelled.status_code in {409, 503}
    persisted = restarted.get(f"/api/v1/commands/{command_id}", headers=auth()).json()
    assert persisted["status"] == "running"
    assert "reconciliation" in (persisted.get("error") or "").lower()


def test_restart_preserves_estop_robot_state_and_rejects_motion_but_allows_stop(tmp_path):
    db_path = str(tmp_path / "robot-state.db")
    adapter = FakeAdapter()
    first = create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": adapter})
    first.state.store.upsert_robot(
        {
            "robotId": "robot-1",
            "status": "estop",
            "connected": False,
            "safetyMode": "estop",
            "details": {"reason": "operator"},
        }
    )
    restarted_adapter = FakeAdapter()
    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN, adapters={"fake": restarted_adapter}))
    robot = restarted.get("/api/v1/robots", headers=auth()).json()[0]
    assert robot["status"] == "estop"
    assert robot["connected"] is False
    motion = restarted.post("/api/v1/commands", headers=auth(), json=command_body(idempotencyKey="estop-motion"))
    assert motion.status_code in {409, 422}
    stop = restarted.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(commandKey="stop", idempotencyKey="estop-stop"),
    )
    assert stop.status_code == 200
    assert [request.commandKey for request, _ in restarted_adapter.dispatches] == ["stop"]


def test_restart_marks_claimed_accepted_command_for_reconciliation(tmp_path):
    db_path = str(tmp_path / "accepted-command.db")
    first = create_app(db_path=db_path, auth_token=TOKEN)
    first.state.store.claim_command(
        {
            "commandId": "accepted-before-crash",
            "robotId": "robot-1",
            "idempotencyKey": "accepted-before-crash-key",
        }
    )
    restarted = TestClient(create_app(db_path=db_path, auth_token=TOKEN))
    receipt = restarted.get("/api/v1/commands/accepted-before-crash", headers=auth()).json()
    assert receipt["status"] == "accepted"
    assert "reconciliation" in (receipt.get("error") or "").lower()


def test_runtime_watchdog_persists_expired_lease_without_new_command():
    adapter = FakeAdapter(result=AdapterResult(status="running"))
    app = create_app(auth_token=TOKEN, adapters={"fake": adapter})
    app.state.control.runtime.lease_seconds = 0.03
    client = TestClient(app)
    submitted = client.post("/api/v1/commands", headers=auth(), json=command_body(idempotencyKey="watchdog-expiry"))
    assert submitted.status_code == 200
    command_id = submitted.json()["commandId"]
    deadline = time.monotonic() + 1.0
    persisted = None
    while time.monotonic() < deadline:
        persisted = client.get(f"/api/v1/commands/{command_id}", headers=auth()).json()
        if persisted["status"] == "expired":
            break
        time.sleep(0.02)
    assert persisted is not None
    assert persisted["status"] == "expired"
    assert adapter.cancellations == [command_id]


def test_nested_mapping_target_alias_is_bound_and_checked():
    adapter = FakeAdapter()
    client = TestClient(create_app(auth_token=TOKEN, adapters={"fake": adapter}))
    body = mapping_body(mapping_id="nested-target")
    body["mapping"] = {"transport": {"target_id": "robot-2", "topic": "/move"}}
    assert client.post("/api/v1/mappings/drafts", headers=auth(), json=body).status_code == 200
    for action in ("validate", "probe", "activate"):
        assert client.post(f"/api/v1/mappings/nested-target/v1/{action}", headers=auth()).status_code == 200
    response = client.post(
        "/api/v1/commands",
        headers=auth(),
        json=command_body(idempotencyKey="nested-target-command", mappingId="nested-target", mappingVersion="v1"),
    )
    assert response.status_code == 422
    assert adapter.dispatches == []


def test_openapi_and_docs_are_not_public():
    client = TestClient(create_app(auth_token=TOKEN))
    assert client.get("/openapi.json").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
