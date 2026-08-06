from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import threading
import warnings

import pytest
from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
from fastapi.testclient import TestClient

from packages.control_core.adapters import AdapterResult, FakeAdapter
from services.control_api.main import create_app
from services.control_api.store import SQLiteStore


TOKEN = "test-token"


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
