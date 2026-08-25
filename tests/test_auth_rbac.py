from __future__ import annotations

import hashlib
import sqlite3

from fastapi.testclient import TestClient

from services.control_api.auth import CSRF_COOKIE, SESSION_COOKIE
from services.control_api.main import create_app
from services.control_api.store import SQLiteStore


LEGACY_TOKEN = "legacy-developer-test-token"
PASSWORD = "correct horse battery staple"


def _register(client: TestClient, username: str) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "displayName": username.title(),
            "password": PASSWORD,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["user"]


def _login(client: TestClient, username: str, password: str = PASSWORD) -> tuple[dict, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.text
    return response.json()["principal"], response.json()["csrfToken"]


def test_auth_migration_is_expand_only_versioned_and_idempotent(tmp_path) -> None:
    db_path = tmp_path / "existing-control.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE robots(robot_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO robots(robot_id, payload_json, updated_at) VALUES (?, ?, ?)",
        ("legacy-robot", '{"robotId":"legacy-robot","sentinel":"preserved"}', "2026-01-01T00:00:00+00:00"),
    )
    connection.execute(
        "CREATE TABLE assistant_conversations("
        "conversation_id TEXT PRIMARY KEY, robot_id TEXT NOT NULL, requested_by TEXT NOT NULL, "
        "locale TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO assistant_conversations("
        "conversation_id, robot_id, requested_by, locale, status, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "legacy-conversation",
            "legacy-robot",
            "legacy-operator",
            "zh-CN",
            "idle",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    first = SQLiteStore(str(db_path))
    assert first.get_robot("legacy-robot")["sentinel"] == "preserved"
    legacy_conversation = first.get_assistant_conversation("legacy-conversation")
    assert legacy_conversation is not None
    assert legacy_conversation["robotId"] == "legacy-robot"
    assert legacy_conversation["title"] is None
    assert legacy_conversation["archived"] is False
    assert legacy_conversation["deleted"] is False
    migrations = first.schema_migrations()
    assert migrations == [
        {
            "version": 1,
            "name": "assistant_v2_auth_rbac",
            "appliedAt": migrations[0]["appliedAt"],
        },
        {
            "version": 2,
            "name": "assistant_v2_conversation_gateway",
            "appliedAt": migrations[1]["appliedAt"],
        },
        {
            "version": 3,
            "name": "assistant_v2_conversation_lifecycle",
            "appliedAt": migrations[2]["appliedAt"],
        },
        {
            "version": 4,
            "name": "assistant_v2_server_owned_conversation_memory",
            "appliedAt": migrations[3]["appliedAt"],
        },
        {
            "version": 5,
            "name": "assistant_v2_owned_speech_transcriptions",
            "appliedAt": migrations[4]["appliedAt"],
        },
        {
            "version": 6,
            "name": "assistant_v2_typed_scene_memory",
            "appliedAt": migrations[5]["appliedAt"],
        },
        {
            "version": 7,
            "name": "mission_v2_autonomy_policy",
            "appliedAt": migrations[6]["appliedAt"],
        },
        {
            "version": 8,
            "name": "daaam_observation_only_semantic_memory",
            "appliedAt": migrations[7]["appliedAt"],
        },
    ]
    first.close()

    reopened = SQLiteStore(str(db_path))
    assert len(reopened.schema_migrations()) == 8
    assert reopened.get_robot("legacy-robot")["sentinel"] == "preserved"
    tables = {
        row[0]
        for row in reopened._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "schema_migrations",
        "users",
        "sessions",
        "login_audits",
        "resource_acl",
        "assistant_turn_events",
        "assistant_conversation_summaries",
        "assistant_conversation_memory_audits",
        "scene_inspection_sessions",
        "scene_inspection_checkpoints",
        "scene_landmarks",
        "scene_landmark_observations",
        "scene_landmark_aliases",
        "autonomy_preferences",
        "autonomy_policy_decisions",
        "autonomy_grants",
        "robot_site_bindings",
        "daaam_memory_imports",
        "daaam_memory_entities",
    } <= tables
    reopened.close()


def test_session_secrets_are_hashed_cookie_flags_and_csrf_are_enforced() -> None:
    app = create_app(auth_token=LEGACY_TOKEN)
    with TestClient(app) as client:
        user = _register(client, "alice")
        denied = client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "wrong-password-value"},
        )
        assert denied.status_code == 401
        assert app.state.store.list_login_audits(limit=1)[0]["outcome"] == "denied"
        principal, _ = _login(client, "alice")
        assert principal["userId"] == user["userId"]
        assert principal["role"] == "USER"

        set_cookie = client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": PASSWORD},
        ).headers.get_list("set-cookie")
        session_cookie = next(value for value in set_cookie if value.startswith(f"{SESSION_COOKIE}="))
        assert "HttpOnly" in session_cookie
        assert "SameSite=lax" in session_cookie
        csrf_cookie = next(value for value in set_cookie if value.startswith(f"{CSRF_COOKIE}="))
        assert "HttpOnly" not in csrf_cookie
        assert "SameSite=lax" in csrf_cookie

        raw_session = client.cookies.get(SESSION_COOKIE)
        assert raw_session
        stored = app.state.store._connection.execute(
            "SELECT token_hash, csrf_token_hash FROM sessions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert stored["token_hash"] == hashlib.sha256(raw_session.encode()).hexdigest()
        assert stored["token_hash"] != raw_session
        assert stored["csrf_token_hash"] != client.cookies.get(CSRF_COOKIE)
        password_row = app.state.store._connection.execute(
            "SELECT password_hash FROM users WHERE user_id = ?",
            (user["userId"],),
        ).fetchone()
        assert password_row["password_hash"].startswith("scrypt$")
        assert PASSWORD not in password_row["password_hash"]

        assert client.get("/api/v1/auth/me").status_code == 200
        without_csrf = client.post(
            "/api/v1/assistant/conversations",
            json={"robotId": "robot-1"},
        )
        assert without_csrf.status_code == 403
        created = client.post(
            "/api/v1/assistant/conversations",
            headers={"X-CSRF-Token": client.cookies.get(CSRF_COOKIE)},
            json={"robotId": "robot-1", "requestedBy": "spoofed-admin"},
        )
        assert created.status_code == 201, created.text
        assert created.json()["requestedBy"] == user["userId"]

        logout = client.post(
            "/api/v1/auth/logout",
            headers={"X-CSRF-Token": client.cookies.get(CSRF_COOKIE)},
        )
        assert logout.status_code == 204
        revoked = app.state.store._connection.execute(
            "SELECT revoked_at FROM sessions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        assert revoked["revoked_at"] is not None
        assert client.get("/api/v1/auth/me").status_code == 401


def test_rbac_ignores_frontend_role_claim_and_preserves_legacy_developer() -> None:
    app = create_app(auth_token=LEGACY_TOKEN)
    with TestClient(app) as client:
        forged_registration = client.post(
            "/api/v1/auth/register",
            json={"username": "forged-admin", "password": PASSWORD, "role": "ADMIN"},
        )
        assert forged_registration.status_code == 422
        _register(client, "normal-user")
        _, csrf = _login(client, "normal-user")
        assert client.get("/api/v1/robots").status_code == 200
        assert client.get("/api/v2/missions").status_code == 200
        assert client.get("/api/v2/simulations").status_code == 403
        assert client.get("/api/v1/diagnostics/auth").status_code == 403
        denied = client.get("/api/v1/commands", headers={"X-Role": "ADMIN"})
        assert denied.status_code == 403
        denied_write = client.post(
            "/api/v1/commands",
            headers={"X-CSRF-Token": csrf, "X-Role": "ADMIN"},
            json={},
        )
        assert denied_write.status_code == 403

        legacy = client.get(
            "/api/v1/commands",
            headers={"Authorization": f"Bearer {LEGACY_TOKEN}"},
        )
        assert legacy.status_code == 200
        assert client.get(
            "/api/v2/simulations",
            headers={"Authorization": f"Bearer {LEGACY_TOKEN}"},
        ).status_code == 200
        diagnostics = client.get(
            "/api/v1/diagnostics/auth",
            headers={"Authorization": f"Bearer {LEGACY_TOKEN}"},
        )
        assert diagnostics.status_code == 200
        assert diagnostics.json()["policyEngine"] == "deterministic-casbin-equivalent"
        legacy_me = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {LEGACY_TOKEN}"},
        ).json()["principal"]
        assert legacy_me["role"] == "DEVELOPER"
        assert legacy_me["authenticationMethod"] == "legacy_bearer"
        assert "users:manage" not in legacy_me["scopes"]
        assert client.get(
            "/api/v1/admin/users",
            headers={"Authorization": f"Bearer {LEGACY_TOKEN}"},
        ).status_code == 403


def test_assistant_resources_are_owner_isolated_and_developer_is_site_scoped() -> None:
    app = create_app(auth_token=LEGACY_TOKEN)
    alice = TestClient(app)
    bob = TestClient(app)
    try:
        alice_user = _register(alice, "alice-owner")
        bob_user = _register(bob, "bob-reader")
        password_hashes = {
            row["password_hash"]
            for row in app.state.store._connection.execute(
                "SELECT password_hash FROM users WHERE user_id IN (?, ?)",
                (alice_user["userId"], bob_user["userId"]),
            ).fetchall()
        }
        assert len(password_hashes) == 2
        _, alice_csrf = _login(alice, "alice-owner")
        _, _ = _login(bob, "bob-reader")

        created = alice.post(
            "/api/v1/assistant/conversations",
            headers={"X-CSRF-Token": alice_csrf},
            json={"robotId": "robot-1"},
        )
        assert created.status_code == 201, created.text
        conversation_id = created.json()["conversationId"]
        assert [item["conversationId"] for item in alice.get("/api/v1/assistant/conversations").json()] == [
            conversation_id
        ]
        assert bob.get("/api/v1/assistant/conversations").json() == []
        assert bob.get(f"/api/v1/assistant/conversations/{conversation_id}").status_code == 404

        app.state.store.update_user_access(bob_user["userId"], role="DEVELOPER")
        assert bob.get(f"/api/v1/assistant/conversations/{conversation_id}").status_code == 200

        app.state.store.update_user_access(bob_user["userId"], site_id="another-site")
        assert bob.get(f"/api/v1/assistant/conversations/{conversation_id}").status_code == 404
        binding = app.state.store.get_resource_owner(
            "assistant_conversation", conversation_id
        )
        assert binding is not None
        assert binding["ownerUserId"] == alice_user["userId"]
        assert binding["siteId"] == "default"
    finally:
        alice.close()
        bob.close()


def test_bootstrap_admin_is_explicit_and_user_admin_boundary_is_enforced(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_BOOTSTRAP_ADMIN_USERNAME", "root-admin")
    monkeypatch.setenv("AUTH_BOOTSTRAP_ADMIN_PASSWORD", PASSWORD)
    app = create_app(auth_token=LEGACY_TOKEN)
    admin = TestClient(app)
    user = TestClient(app)
    try:
        _, admin_csrf = _login(admin, "root-admin")
        registered = _register(user, "plain-user")
        _login(user, "plain-user")
        assert user.get("/api/v1/admin/users").status_code == 403
        listed = admin.get("/api/v1/admin/users")
        assert listed.status_code == 200
        assert {item["username"] for item in listed.json()} == {"root-admin", "plain-user"}

        promoted = admin.patch(
            f"/api/v1/admin/users/{registered['userId']}",
            headers={"X-CSRF-Token": admin_csrf},
            json={"role": "DEVELOPER"},
        )
        assert promoted.status_code == 200
        assert promoted.json()["role"] == "DEVELOPER"
        audits = admin.get("/api/v1/admin/login-audits")
        assert audits.status_code == 200
        assert all(item["outcome"] == "succeeded" for item in audits.json())
    finally:
        admin.close()
        user.close()
