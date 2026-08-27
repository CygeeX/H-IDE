from __future__ import annotations

from fastapi.testclient import TestClient

from services.control_api.main import create_app


TOKEN = "assistant-history-v2-test-token"
PASSWORD = "correct horse battery staple"


def register_and_login(client: TestClient, username: str) -> str:
    registered = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": PASSWORD},
    )
    assert registered.status_code == 201, registered.text
    logged_in = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": PASSWORD},
    )
    assert logged_in.status_code == 200, logged_in.text
    return str(logged_in.json()["csrfToken"])


def create_conversation(
    client: TestClient,
    csrf: str,
    title: str,
) -> dict:
    response = client.post(
        "/api/v1/assistant/conversations",
        headers={"X-CSRF-Token": csrf},
        json={"robotId": "robot-1", "title": title},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_conversation_history_v2_is_cursor_paged_searchable_and_tenant_scoped() -> None:
    app = create_app(auth_token=TOKEN)
    alice = TestClient(app)
    bob = TestClient(app)
    try:
        alice_csrf = register_and_login(alice, "history-v2-alice")
        bob_csrf = register_and_login(bob, "history-v2-bob")
        alice_rows = [
            create_conversation(alice, alice_csrf, title)
            for title in ("会议室巡检", "前台问答", "办公室导航", "视觉观察")
        ]
        bob_row = create_conversation(bob, bob_csrf, "另一个用户的会话")

        # Make ordering deterministic while preserving a cross-owner row in
        # the global scan. The V2 cursor must still fill Alice's pages without
        # leaking Bob's resource.
        ordered = [alice_rows[0], bob_row, alice_rows[1], alice_rows[2], alice_rows[3]]
        for index, row in enumerate(ordered):
            app.state.store._connection.execute(
                "UPDATE assistant_conversations SET updated_at = ? "
                "WHERE conversation_id = ?",
                (f"2026-08-20T00:00:0{9 - index}+00:00", row["conversationId"]),
            )
        app.state.store._connection.commit()

        first = alice.get("/api/v2/assistant/conversations?limit=2")
        assert first.status_code == 200, first.text
        assert len(first.json()["items"]) == 2
        assert first.json()["nextCursor"]
        second = alice.get(
            "/api/v2/assistant/conversations?limit=2&cursor="
            + first.json()["nextCursor"]
        )
        assert second.status_code == 200, second.text
        ids = [item["conversationId"] for item in first.json()["items"]]
        ids.extend(item["conversationId"] for item in second.json()["items"])
        assert len(ids) == 4
        assert len(set(ids)) == 4
        assert bob_row["conversationId"] not in ids
        assert second.json()["nextCursor"] is None

        searched = alice.get(
            "/api/v2/assistant/conversations?limit=30&q=%E4%BC%9A%E8%AE%AE"
        )
        assert [item["title"] for item in searched.json()["items"]] == ["会议室巡检"]

        archived_id = alice_rows[1]["conversationId"]
        deleted_id = alice_rows[2]["conversationId"]
        assert alice.patch(
            f"/api/v1/assistant/conversations/{archived_id}",
            headers={"X-CSRF-Token": alice_csrf},
            json={"archived": True},
        ).status_code == 200
        assert alice.delete(
            f"/api/v1/assistant/conversations/{deleted_id}",
            headers={"X-CSRF-Token": alice_csrf},
        ).status_code == 200
        archived = alice.get(
            "/api/v2/assistant/conversations?lifecycle=archived"
        ).json()["items"]
        deleted = alice.get(
            "/api/v2/assistant/conversations?lifecycle=deleted"
        ).json()["items"]
        assert [item["conversationId"] for item in archived] == [archived_id]
        assert [item["conversationId"] for item in deleted] == [deleted_id]

        remaining_ids = [alice_rows[0]["conversationId"], alice_rows[3]["conversationId"]]
        bulk = alice.post(
            "/api/v2/assistant/conversations/bulk-delete",
            headers={"X-CSRF-Token": alice_csrf},
            json={"conversationIds": remaining_ids},
        )
        assert bulk.status_code == 200, bulk.text
        assert set(bulk.json()["deletedConversationIds"]) == set(remaining_ids)
        assert bulk.json()["skipped"] == []
    finally:
        alice.close()
        bob.close()


def test_linked_active_mission_blocks_single_and_bulk_conversation_deletion() -> None:
    app = create_app(auth_token=TOKEN)
    client = TestClient(app)
    try:
        csrf = register_and_login(client, "history-v2-mission-owner")
        conversation = create_conversation(client, csrf, "连续办公室任务")
        conversation_id = conversation["conversationId"]
        task_id = "task-history-mission-1"
        mission_id = "mission-history-1"
        app.state.store.create_assistant_turn(
            {
                "taskId": task_id,
                "conversationId": conversation_id,
                "text": "走遍所有办公室",
                "modality": "text",
            },
            message_id="message-history-mission-user-1",
        )
        assert app.state.store.complete_assistant_turn(
            task_id,
            status="draft_ready",
            result={
                "schemaVersion": "h-ide.assistant-result/2",
                "status": "review",
                "intent": "ROBOT_ACTION",
                "route": "mission_review",
                "routeStatus": "review",
                "assistantMessage": "请审核 Mission",
                "missionDraftId": mission_id,
            },
            assistant_message="请审核 Mission",
            message_id="message-history-mission-assistant-1",
            conversation_status="review",
        )
        # Migration recovery reconstructs the relation from the immutable turn
        # result instead of depending on a frontend-generated synthetic row.
        app.state.store._connection.execute(
            "DELETE FROM assistant_conversation_missions WHERE mission_draft_id = ?",
            (mission_id,),
        )
        app.state.store._backfill_assistant_conversation_mission_links()
        app.state.store._connection.commit()
        assert app.state.store.get_assistant_conversation_mission_links(
            conversation_id
        )[0]["missionDraftId"] == mission_id
        now = "2026-08-20T01:00:00+00:00"
        app.state.store._connection.execute(
            "INSERT INTO missions(mission_id, robot_id, status, plan_json, "
            "authorization_json, snapshot_json, active_revision_id, last_sequence, "
            "error, created_at, updated_at) VALUES (?, 'robot-1', 'executing', "
            "'{}', '{}', '{}', 'revision-history-1', 1, NULL, ?, ?)",
            (mission_id, now, now),
        )
        app.state.store._connection.commit()
        assert app.state.store.bind_approved_mission_to_conversation(
            mission_id, mission_id
        )

        page = client.get("/api/v2/assistant/conversations").json()["items"]
        assert len(page) == 1
        assert page[0]["missionDraftIds"] == [mission_id]
        assert page[0]["missionIds"] == [mission_id]
        assert page[0]["activeMissionId"] == mission_id
        assert page[0]["lifecycleMutable"] is False
        assert page[0]["lifecycleBlockedReason"] == "active_mission"

        blocked = client.delete(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": csrf},
        )
        assert blocked.status_code == 409
        assert "linked Mission is active" in blocked.json()["detail"]
        bulk = client.post(
            "/api/v2/assistant/conversations/bulk-delete",
            headers={"X-CSRF-Token": csrf},
            json={"conversationIds": [conversation_id]},
        )
        assert bulk.status_code == 200, bulk.text
        assert bulk.json()["deletedConversationIds"] == []
        assert bulk.json()["skipped"] == [
            {"conversationId": conversation_id, "reason": "active_mission"}
        ]

        app.state.store._connection.execute(
            "UPDATE missions SET status = 'completed' WHERE mission_id = ?",
            (mission_id,),
        )
        app.state.store._connection.commit()
        deleted = client.delete(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": csrf},
        )
        assert deleted.status_code == 200, deleted.text
        detail = client.get(
            f"/api/v1/assistant/conversations/{conversation_id}"
        ).json()
        assert detail["missionIds"] == [mission_id]
        assert detail["deleted"] is True
    finally:
        client.close()


def test_conversation_detail_preserves_turn_sequence_and_mission_links() -> None:
    app = create_app(auth_token=TOKEN)
    client = TestClient(app)
    try:
        csrf = register_and_login(client, "history-v2-turn-order")
        conversation = create_conversation(client, csrf, "严格轮次顺序")
        conversation_id = conversation["conversationId"]

        for sequence in (1, 2):
            task_id = f"task-order-{sequence}"
            mission_id = f"mission-order-{sequence}"
            created = app.state.store.create_assistant_turn(
                {
                    "taskId": task_id,
                    "conversationId": conversation_id,
                    "text": f"第 {sequence} 个任务",
                    "modality": "text",
                },
                message_id=f"message-order-user-{sequence}",
            )
            assert created["turnSequence"] == sequence
            assert app.state.store.complete_assistant_turn(
                task_id,
                status="draft_ready",
                result={
                    "schemaVersion": "h-ide.assistant-result/2",
                    "intent": "ROBOT_ACTION",
                    "route": "mission_review",
                    "routeStatus": "review",
                    "assistantMessage": f"第 {sequence} 个计划",
                    "missionDraftId": mission_id,
                },
                assistant_message=f"第 {sequence} 个计划",
                message_id=f"message-order-assistant-{sequence}",
                conversation_status="review",
            )

        # Updating the older link later must never move it below turn 2.
        app.state.store._connection.execute(
            "UPDATE assistant_conversation_missions SET updated_at = ? "
            "WHERE task_id = 'task-order-1'",
            ("2099-01-01T00:00:00+00:00",),
        )
        app.state.store._connection.commit()

        detail = app.state.assistant.get_conversation(conversation_id)
        assert [item["taskId"] for item in detail["turns"]] == [
            "task-order-1",
            "task-order-2",
        ]
        assert [item["turnSequence"] for item in detail["turns"]] == [1, 2]
        assert [item["taskId"] for item in detail["missionLinks"]] == [
            "task-order-1",
            "task-order-2",
        ]
        assert [item["turnSequence"] for item in detail["missionLinks"]] == [1, 2]
        assert [item["turnSequence"] for item in detail["messages"]] == [
            1,
            1,
            2,
            2,
        ]
    finally:
        client.close()


def test_failed_turn_projects_specific_recoverable_error_instead_of_outage() -> None:
    app = create_app(auth_token=TOKEN)
    client = TestClient(app)
    try:
        csrf = register_and_login(client, "history-v2-error-projection")
        conversation = create_conversation(client, csrf, "错误分类")
        task_id = "task-target-resolution-error"
        app.state.store.create_assistant_turn(
            {
                "taskId": task_id,
                "conversationId": conversation["conversationId"],
                "text": "去椅子旁边站着",
                "modality": "text",
            },
            message_id="message-target-resolution-error",
        )
        assert app.state.store.fail_assistant_turn(
            task_id,
            "Mission navigation node navigate_to_chair has no unique trusted target",
        )

        turn = app.state.assistant.get_turn(task_id)
        assert turn["failure"] == {
            "errorCode": "TARGET_RESOLUTION_REQUIRED",
            "category": "target_resolution",
            "recoverable": True,
            "dependencyUnavailable": False,
            "userMessage": "目标还没有形成可导航的可信位置；我会重新观察或展示候选，不会发送猜测坐标。",
        }
    finally:
        client.close()
