from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.control_api.main import _mount_control_pwa, create_app
from services.control_api.store import AssistantConversationConflict


TOKEN = "conversation-lifecycle-test-token"
PASSWORD = "correct horse battery staple"


def register(client: TestClient, username: str) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={"username": username, "password": PASSWORD},
    )
    assert response.status_code == 201, response.text
    return response.json()["user"]


def login(client: TestClient, username: str) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": PASSWORD},
    )
    assert response.status_code == 200, response.text
    return response.json()["csrfToken"]


def test_conversation_lifecycle_is_recoverable_preserves_children_and_enforces_acl() -> None:
    app = create_app(auth_token=TOKEN)
    alice = TestClient(app)
    bob = TestClient(app)
    try:
        alice_user = register(alice, "lifecycle-alice")
        bob_user = register(bob, "lifecycle-bob")
        alice_csrf = login(alice, "lifecycle-alice")
        bob_csrf = login(bob, "lifecycle-bob")

        created = alice.post(
            "/api/v1/assistant/conversations",
            headers={"X-CSRF-Token": alice_csrf},
            json={"robotId": "robot-1", "title": "  初始会话  "},
        )
        assert created.status_code == 201, created.text
        conversation_id = created.json()["conversationId"]
        assert created.json()["title"] == "初始会话"
        assert created.json()["archived"] is False
        assert created.json()["deleted"] is False

        # The tombstone must not cascade into any child/audit relation.
        app.state.store._connection.execute(
            "INSERT INTO assistant_messages("
            "message_id, conversation_id, task_id, role, content, modality, "
            "transcription_id, created_at) VALUES (?, ?, NULL, 'user', ?, 'text', NULL, ?)",
            (
                "message-lifecycle-1",
                conversation_id,
                "需要永久保留的消息",
                "2026-08-19T00:00:00+00:00",
            ),
        )
        app.state.store._connection.commit()

        missing_csrf = alice.patch(
            f"/api/v1/assistant/conversations/{conversation_id}",
            json={"title": "新标题"},
        )
        assert missing_csrf.status_code == 403

        renamed = alice.patch(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": alice_csrf},
            json={"title": "  新标题  "},
        )
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()["title"] == "新标题"

        # Ordinary users cannot manage another owner's same-site resource.
        denied = bob.patch(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": bob_csrf},
            json={"archived": True},
        )
        assert denied.status_code == 404

        # Developers are site-scoped, never globally privileged.
        app.state.store.update_user_access(bob_user["userId"], role="DEVELOPER")
        app.state.store.update_user_access(bob_user["userId"], site_id="other-site")
        denied_cross_site = bob.delete(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": bob_csrf},
        )
        assert denied_cross_site.status_code == 404

        archived = alice.patch(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": alice_csrf},
            json={"archived": True},
        )
        assert archived.status_code == 200, archived.text
        assert archived.json()["archived"] is True
        assert alice.get("/api/v1/assistant/conversations").json() == []
        visible_archived = alice.get(
            "/api/v1/assistant/conversations?includeArchived=true"
        ).json()
        assert [item["conversationId"] for item in visible_archived] == [conversation_id]
        with pytest.raises(AssistantConversationConflict):
            app.state.store.create_assistant_turn(
                {
                    "taskId": "task-after-archive",
                    "conversationId": conversation_id,
                    "text": "不能与归档并发写入",
                },
                message_id="message-after-archive",
            )

        deleted = alice.delete(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": alice_csrf},
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["deleted"] is True
        assert alice.get(
            "/api/v1/assistant/conversations?includeArchived=true"
        ).json() == []
        visible_deleted = alice.get(
            "/api/v1/assistant/conversations?includeArchived=true&includeDeleted=true"
        ).json()
        assert [item["conversationId"] for item in visible_deleted] == [conversation_id]

        detail = alice.get(
            f"/api/v1/assistant/conversations/{conversation_id}"
        )
        assert detail.status_code == 200, detail.text
        assert [item["content"] for item in detail.json()["messages"]] == [
            "需要永久保留的消息"
        ]
        assert app.state.store.get_resource_owner(
            "assistant_conversation", conversation_id
        )["ownerUserId"] == alice_user["userId"]

        blocked_message = alice.post(
            f"/api/v1/assistant/conversations/{conversation_id}/messages",
            headers={"X-CSRF-Token": alice_csrf},
            json={"text": "删除状态不能继续对话"},
        )
        assert blocked_message.status_code == 409

        restored = alice.patch(
            f"/api/v1/assistant/conversations/{conversation_id}",
            headers={"X-CSRF-Token": alice_csrf},
            json={"deleted": False, "archived": False},
        )
        assert restored.status_code == 200, restored.text
        assert restored.json()["deleted"] is False
        assert restored.json()["archived"] is False
        assert [item["conversationId"] for item in alice.get(
            "/api/v1/assistant/conversations"
        ).json()] == [conversation_id]
    finally:
        alice.close()
        bob.close()


def test_control_pwa_production_deep_links_fall_back_to_index(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<main>assistant-v2-shell</main>", encoding="utf-8")
    (dist / "asset.txt").write_text("static-asset", encoding="utf-8")
    app = FastAPI()
    assert _mount_control_pwa(app, dist) is True

    with TestClient(app) as client:
        for path in (
            "/app",
            "/app/scene-memory",
            "/app/tasks/conversation-1",
            "/developer",
            "/developer/tasks/conversation-2",
            "/developer/robots",
            "/developer/diagnostics",
            "/developer/simulation",
            "/developer/interfaces",
            "/developer/scene-memory",
        ):
            response = client.get(path)
            assert response.status_code == 200, (path, response.text)
            assert response.text == "<main>assistant-v2-shell</main>"
            assert response.headers["cache-control"] == "no-cache"

        # Legacy paths remain compatibility fallbacks but do not reintroduce
        # a separate scenes product route or mutate archived scene data.
        assert client.get("/scenes").text == "<main>assistant-v2-shell</main>"
        assert client.get("/asset.txt").text == "static-asset"
