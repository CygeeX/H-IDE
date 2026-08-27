"""Thread-safe SQLite persistence for the control-plane API.

The store intentionally exposes small repository-style methods rather than
leaking SQL into route handlers.  Each method uses the same connection under
an RLock, so ``:memory:`` databases are useful in tests while file databases
can be reopened by another app instance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterable, Mapping
import uuid

from pydantic import BaseModel

from packages.control_core.models import CapabilitySnapshot, CommandReceipt, CommandRequest, RobotState
from packages.control_core.registry import MappingRecord


@dataclass(frozen=True)
class CommandClaim:
    """Result of the durable command idempotency claim.

    ``owned`` is true only for the request that inserted the durable row.
    A caller seeing ``conflict`` supplied a command id already owned by a
    different idempotency key; neither case is allowed to dispatch again.
    """

    owned: bool
    conflict: bool
    record: dict[str, Any] | None


class MappingPointerConflict(ValueError):
    """The durable active pointer changed since a lifecycle operation began."""

    def __init__(
        self,
        command_key: str,
        expected: tuple[str, str] | None,
        actual: tuple[str, str] | None,
    ) -> None:
        self.command_key = command_key
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"active mapping pointer changed for {command_key}: "
            f"expected {expected!r}, found {actual!r}"
        )


class MappingStateConflict(ValueError):
    """A stale snapshot attempted to overwrite a newer mapping row."""

    def __init__(self, mapping_id: str, version: str, existing: str, incoming: str) -> None:
        self.mapping_id = mapping_id
        self.version = version
        self.existing = existing
        self.incoming = incoming
        super().__init__(
            f"mapping {mapping_id}/{version} changed concurrently "
            f"(stored={existing}, incoming={incoming})"
        )


class AssistantTurnConflict(ValueError):
    """A conversation already owns an active planning turn."""


class AssistantConversationConflict(ValueError):
    """A lifecycle mutation conflicts with the conversation's current state."""


_MAPPING_POINTER_UNSET = object()
_MAPPING_PROBE_UNSET = object()
_ASSISTANT_TITLE_UNSET = object()


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=False)
        if "schema_" in payload and "schema" not in payload:
            payload["schema"] = payload.pop("schema_")
        return payload
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo else value.replace(tzinfo=timezone.utc).isoformat()
    return value


def _encode(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _decode(value: str) -> Any:
    return json.loads(value)


def _legacy_world_record_read_projection(value: Any) -> Any:
    """Expose retired semantic records for audit without making them writable.

    The durable JSON is never rewritten.  New Mission contracts reject the
    retired source, and these output-only flags make that boundary explicit to
    recovery/export clients.
    """

    if not isinstance(value, dict) or value.get("source") != "mujoco_oracle":
        return value
    return {**value, "legacyReadOnly": True, "executionEligible": False}


def _legacy_skill_result_read_projection(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = value.get("result")
    if not isinstance(result, dict):
        return value
    visual = result.get("visualResult")
    if not isinstance(visual, dict):
        return value
    versions = visual.get("versions")
    provider = versions.get("vlmProvider") if isinstance(versions, dict) else None
    hypotheses = visual.get("hypotheses")
    retired_source = (
        provider != "qwen"
        or (isinstance(versions, dict) and versions.get("poseModel") not in {None, ""})
        or (
            isinstance(hypotheses, list)
            and any(
                isinstance(item, dict) and item.get("source") != "qwen"
                for item in hypotheses
            )
        )
    )
    if not retired_source:
        return value
    return {
        **value,
        "result": {
            **result,
            "visualResult": {
                **visual,
                "legacyReadOnly": True,
                "executionEligible": False,
            },
        },
    }


class SQLiteStore:
    """A small repository boundary that can be replaced by PostgreSQL later."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            path = Path(self.db_path)
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS robots (
                    robot_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capabilities (
                    robot_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS commands (
                    command_id TEXT PRIMARY KEY,
                    idempotency_key TEXT,
                    request_json TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mappings (
                    mapping_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_id TEXT,
                    probe_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (mapping_id, version)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audits (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assistant_drafts (
                    draft_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assistant_conversations (
                    conversation_id TEXT PRIMARY KEY,
                    robot_id TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    locale TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assistant_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    task_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    transcription_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES assistant_conversations(conversation_id)
                );
                CREATE TABLE IF NOT EXISTS assistant_turns (
                    task_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES assistant_conversations(conversation_id)
                );
                CREATE TABLE IF NOT EXISTS assistant_draft_metadata (
                    draft_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    review_json TEXT NOT NULL,
                    revision_of_draft_id TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(draft_id) REFERENCES assistant_drafts(draft_id),
                    FOREIGN KEY(conversation_id) REFERENCES assistant_conversations(conversation_id),
                    FOREIGN KEY(task_id) REFERENCES assistant_turns(task_id)
                );
                CREATE TABLE IF NOT EXISTS speech_transcriptions (
                    transcription_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    language TEXT,
                    locale TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mapping_active (
                    command_key TEXT PRIMARY KEY,
                    mapping_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS visual_missions (
                    mission_id TEXT PRIMARY KEY,
                    draft_id TEXT NOT NULL UNIQUE,
                    robot_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(draft_id) REFERENCES assistant_drafts(draft_id)
                );
                CREATE TABLE IF NOT EXISTS visual_mission_stages (
                    mission_id TEXT NOT NULL,
                    stage_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    visual_resolution_id TEXT,
                    command_id TEXT,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(mission_id, stage_id, attempt),
                    FOREIGN KEY(mission_id) REFERENCES visual_missions(mission_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS visual_resolution_one_command_uq
                    ON visual_mission_stages(visual_resolution_id)
                    WHERE visual_resolution_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS missions (
                    mission_id TEXT PRIMARY KEY,
                    robot_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    authorization_json TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    active_revision_id TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mission_drafts (
                    draft_id TEXT PRIMARY KEY,
                    robot_id TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mission_revisions (
                    revision_id TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(mission_id, sequence),
                    FOREIGN KEY(mission_id) REFERENCES missions(mission_id)
                );
                CREATE TABLE IF NOT EXISTS mission_authorizations (
                    authorization_id TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(mission_id) REFERENCES missions(mission_id)
                );
                CREATE TABLE IF NOT EXISTS mission_events (
                    mission_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(mission_id, sequence),
                    FOREIGN KEY(mission_id) REFERENCES missions(mission_id)
                );
                CREATE TABLE IF NOT EXISTS mission_world_records (
                    record_id TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL,
                    robot_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    FOREIGN KEY(mission_id) REFERENCES missions(mission_id)
                );
                CREATE TABLE IF NOT EXISTS mission_node_states (
                    mission_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(mission_id, revision_id, node_id),
                    FOREIGN KEY(mission_id) REFERENCES missions(mission_id),
                    FOREIGN KEY(revision_id) REFERENCES mission_revisions(revision_id)
                );
                CREATE TABLE IF NOT EXISTS mission_skill_executions (
                    request_id TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(mission_id) REFERENCES missions(mission_id),
                    FOREIGN KEY(revision_id) REFERENCES mission_revisions(revision_id)
                );
                CREATE INDEX IF NOT EXISTS mission_events_sequence_idx
                    ON mission_events(mission_id, sequence);
                CREATE INDEX IF NOT EXISTS mission_world_subject_idx
                    ON mission_world_records(mission_id, subject, predicate, observed_at);
                CREATE INDEX IF NOT EXISTS mission_skill_node_idx
                    ON mission_skill_executions(mission_id, revision_id, node_id, updated_at);
                """
            )
            # Existing installations predate the durable idempotency and
            # assistant execution columns.  Migrate in place rather than
            # requiring operators to delete a command database.
            self._ensure_column("commands", "idempotency_key", "TEXT")
            self._ensure_column("assistant_drafts", "execution_json", "TEXT NOT NULL DEFAULT '{}'" )
            self._ensure_column("assistant_drafts", "error", "TEXT")
            self._backfill_idempotency_keys()
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS commands_idempotency_key_uq "
                "ON commands(idempotency_key) WHERE idempotency_key IS NOT NULL"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS assistant_messages_conversation_idx "
                "ON assistant_messages(conversation_id, created_at, message_id)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS assistant_turns_conversation_idx "
                "ON assistant_turns(conversation_id, created_at, task_id)"
            )
            self._connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS assistant_turns_one_active_idx "
                "ON assistant_turns(conversation_id) WHERE status IN ('queued', 'running')"
            )
            self._backfill_mapping_pointers()
            self._apply_schema_migrations()
            self._backfill_assistant_conversation_mission_links()
            self._connection.commit()

    def _apply_schema_migrations(self) -> None:
        """Apply explicit, expand-only schema migrations.

        The original control-plane schema predates a migration ledger and is
        still created above with ``CREATE TABLE IF NOT EXISTS`` for backwards
        compatibility.  Assistant V2 additions are deliberately kept in
        numbered transactions: an existing database is never rebuilt and a
        partially-applied migration cannot be recorded as successful.
        """

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                applied_at TEXT NOT NULL
            )
            """
        )
        migrations: tuple[tuple[int, str, tuple[str, ...]], ...] = (
            (
                1,
                "assistant_v2_auth_rbac",
                (
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                        display_name TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('USER', 'DEVELOPER', 'ADMIN')),
                        site_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active'
                            CHECK(status IN ('active', 'disabled')),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        token_hash TEXT NOT NULL UNIQUE,
                        csrf_token_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        revoked_at TEXT,
                        user_agent_hash TEXT,
                        FOREIGN KEY(user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS login_audits (
                        login_audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        username TEXT NOT NULL,
                        outcome TEXT NOT NULL,
                        remote_address TEXT,
                        user_agent_hash TEXT,
                        detail_json TEXT NOT NULL DEFAULT '{}',
                        occurred_at TEXT NOT NULL,
                        FOREIGN KEY(user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE TABLE IF NOT EXISTS resource_acl (
                        resource_type TEXT NOT NULL,
                        resource_id TEXT NOT NULL,
                        owner_user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(resource_type, resource_id),
                        FOREIGN KEY(owner_user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS users_site_role_idx
                        ON users(site_id, role, status)
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS sessions_user_active_idx
                        ON sessions(user_id, expires_at, revoked_at)
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS login_audits_user_time_idx
                        ON login_audits(user_id, occurred_at)
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS login_audits_username_time_idx
                        ON login_audits(username, occurred_at)
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS resource_acl_owner_idx
                        ON resource_acl(owner_user_id, resource_type, updated_at)
                    """,
                    """
                    CREATE INDEX IF NOT EXISTS resource_acl_site_idx
                        ON resource_acl(site_id, resource_type, updated_at)
                    """,
                ),
            ),
            (
                2,
                "assistant_v2_conversation_gateway",
                (
                    "ALTER TABLE assistant_conversations ADD COLUMN robot_id_v2 TEXT",
                    "ALTER TABLE assistant_conversations ADD COLUMN summary TEXT",
                    "UPDATE assistant_conversations SET robot_id_v2 = robot_id "
                    "WHERE robot_id_v2 IS NULL AND robot_id <> '__assistant_unbound__'",
                    """
                    CREATE TABLE assistant_turn_events (
                        task_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        event_id TEXT NOT NULL UNIQUE,
                        conversation_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        data_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(task_id, sequence),
                        FOREIGN KEY(task_id) REFERENCES assistant_turns(task_id),
                        FOREIGN KEY(conversation_id)
                            REFERENCES assistant_conversations(conversation_id)
                    )
                    """,
                    """
                    CREATE INDEX assistant_turn_events_conversation_idx
                        ON assistant_turn_events(conversation_id, created_at, event_id)
                    """,
                ),
            ),
            (
                3,
                "assistant_v2_conversation_lifecycle",
                (
                    "ALTER TABLE assistant_conversations ADD COLUMN title TEXT",
                    "ALTER TABLE assistant_conversations ADD COLUMN archived_at TEXT",
                    "ALTER TABLE assistant_conversations ADD COLUMN deleted_at TEXT",
                    """
                    CREATE INDEX assistant_conversations_lifecycle_updated_idx
                        ON assistant_conversations(deleted_at, archived_at, updated_at DESC)
                    """,
                ),
            ),
            (
                4,
                "assistant_v2_server_owned_conversation_memory",
                (
                    """
                    CREATE TABLE assistant_conversation_summaries (
                        conversation_id TEXT NOT NULL,
                        summary_revision INTEGER NOT NULL,
                        summary_id TEXT NOT NULL,
                        summary_text TEXT NOT NULL,
                        covered_through_turn INTEGER NOT NULL,
                        generated_at TEXT NOT NULL,
                        method TEXT NOT NULL
                            CHECK(method IN ('SERVER_COMPACTION', 'OPERATOR_APPROVED')),
                        content_sha256 TEXT NOT NULL,
                        source_sha256 TEXT NOT NULL,
                        source_message_ids_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(conversation_id, summary_revision),
                        FOREIGN KEY(conversation_id)
                            REFERENCES assistant_conversations(conversation_id)
                    )
                    """,
                    """
                    CREATE UNIQUE INDEX assistant_conversation_summary_identity_uq
                        ON assistant_conversation_summaries(conversation_id, summary_id,
                            summary_revision)
                    """,
                    """
                    CREATE TABLE assistant_conversation_memory_audits (
                        audit_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        context_revision INTEGER NOT NULL,
                        resource_id TEXT NOT NULL,
                        summary_revision INTEGER,
                        recent_message_ids_json TEXT NOT NULL DEFAULT '[]',
                        selected_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                        query_sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(conversation_id, context_revision),
                        FOREIGN KEY(conversation_id)
                            REFERENCES assistant_conversations(conversation_id)
                    )
                    """,
                    """
                    CREATE INDEX assistant_conversation_memory_audit_time_idx
                        ON assistant_conversation_memory_audits(
                            conversation_id, created_at, audit_id)
                    """,
                ),
            ),
            (
                5,
                "assistant_v2_owned_speech_transcriptions",
                (
                    "ALTER TABLE speech_transcriptions ADD COLUMN owner_user_id TEXT",
                    "ALTER TABLE speech_transcriptions ADD COLUMN site_id TEXT",
                    "ALTER TABLE speech_transcriptions ADD COLUMN auth_session_id TEXT",
                    "ALTER TABLE speech_transcriptions ADD COLUMN speech_session_id TEXT",
                    "ALTER TABLE speech_transcriptions ADD COLUMN utterance_id TEXT",
                    "ALTER TABLE speech_transcriptions ADD COLUMN source TEXT NOT NULL "
                    "DEFAULT 'legacy_unbound'",
                    """
                    CREATE UNIQUE INDEX speech_transcriptions_stream_utterance_uq
                        ON speech_transcriptions(speech_session_id, utterance_id)
                        WHERE source = 'streaming_asr'
                            AND speech_session_id IS NOT NULL
                            AND utterance_id IS NOT NULL
                    """,
                    """
                    CREATE INDEX speech_transcriptions_owner_site_idx
                        ON speech_transcriptions(owner_user_id, site_id, created_at)
                    """,
                ),
            ),
            (
                6,
                "assistant_v2_typed_scene_memory",
                (
                    """
                    CREATE TABLE scene_inspection_sessions (
                        inspection_id TEXT PRIMARY KEY,
                        owner_user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        robot_id TEXT NOT NULL,
                        purpose TEXT,
                        map_frame_id TEXT NOT NULL CHECK(map_frame_id = 'map'),
                        localization_epoch_id TEXT NOT NULL,
                        status TEXT NOT NULL
                            CHECK(status IN ('open', 'completed', 'cancelled')),
                        record_version INTEGER NOT NULL DEFAULT 1
                            CHECK(record_version >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        cancelled_at TEXT,
                        FOREIGN KEY(owner_user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE TABLE scene_inspection_checkpoints (
                        checkpoint_id TEXT PRIMARY KEY,
                        inspection_id TEXT NOT NULL,
                        owner_user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        sequence INTEGER NOT NULL CHECK(sequence >= 1),
                        status TEXT NOT NULL CHECK(status IN (
                            'stopped', 'captured', 'named', 'confirmed',
                            'superseded', 'cancelled'
                        )),
                        motion_state TEXT NOT NULL CHECK(motion_state = 'STOPPED'),
                        head_state TEXT NOT NULL CHECK(head_state = 'FIXED'),
                        stationary_duration_ms INTEGER NOT NULL
                            CHECK(stationary_duration_ms >= 500),
                        linear_speed_mps REAL NOT NULL CHECK(linear_speed_mps >= 0),
                        angular_speed_radps REAL NOT NULL CHECK(angular_speed_radps >= 0),
                        robot_pose_json TEXT NOT NULL,
                        localization_epoch_id TEXT NOT NULL,
                        observed_at TEXT NOT NULL,
                        frame_id TEXT,
                        frame_sha256 TEXT,
                        evidence_sha256 TEXT,
                        evidence_ref TEXT,
                        vlm_provider TEXT CHECK(vlm_provider IS NULL OR vlm_provider = 'qwen'),
                        vlm_model TEXT,
                        bbox_json TEXT,
                        depth_json TEXT,
                        tf_provenance_json TEXT,
                        confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                        support_count INTEGER CHECK(support_count IS NULL OR support_count >= 1),
                        captured_at TEXT,
                        named_at TEXT,
                        confirmed_at TEXT,
                        record_version INTEGER NOT NULL DEFAULT 1
                            CHECK(record_version >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(inspection_id, sequence),
                        FOREIGN KEY(inspection_id)
                            REFERENCES scene_inspection_sessions(inspection_id),
                        FOREIGN KEY(owner_user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE TABLE scene_landmarks (
                        landmark_id TEXT PRIMARY KEY,
                        root_landmark_id TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK(revision >= 1),
                        owner_user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        inspection_id TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        canonical_name TEXT NOT NULL,
                        normalized_name TEXT NOT NULL,
                        description TEXT,
                        status TEXT NOT NULL
                            CHECK(status IN ('candidate', 'confirmed', 'superseded')),
                        source_kind TEXT NOT NULL DEFAULT 'SCENE_TEACH'
                            CHECK(source_kind = 'SCENE_TEACH'),
                        execution_eligible INTEGER NOT NULL DEFAULT 0
                            CHECK(execution_eligible = 0),
                        current_observation_id TEXT NOT NULL,
                        supersedes_landmark_id TEXT,
                        superseded_by_landmark_id TEXT,
                        superseded_reason TEXT,
                        confirmed_by_user_id TEXT,
                        confirmed_at TEXT,
                        invalidated_by_user_id TEXT,
                        invalidated_at TEXT,
                        record_version INTEGER NOT NULL DEFAULT 1
                            CHECK(record_version >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(root_landmark_id, revision),
                        FOREIGN KEY(owner_user_id) REFERENCES users(user_id),
                        FOREIGN KEY(inspection_id)
                            REFERENCES scene_inspection_sessions(inspection_id),
                        FOREIGN KEY(checkpoint_id)
                            REFERENCES scene_inspection_checkpoints(checkpoint_id),
                        FOREIGN KEY(supersedes_landmark_id)
                            REFERENCES scene_landmarks(landmark_id),
                        FOREIGN KEY(superseded_by_landmark_id)
                            REFERENCES scene_landmarks(landmark_id),
                        FOREIGN KEY(confirmed_by_user_id) REFERENCES users(user_id),
                        FOREIGN KEY(invalidated_by_user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE TABLE scene_landmark_observations (
                        observation_id TEXT PRIMARY KEY,
                        landmark_id TEXT NOT NULL,
                        owner_user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        inspection_id TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        status TEXT NOT NULL
                            CHECK(status IN ('candidate', 'confirmed', 'superseded')),
                        map_pose_json TEXT NOT NULL,
                        covariance_json TEXT NOT NULL,
                        localization_epoch_id TEXT NOT NULL,
                        robot_pose_json TEXT NOT NULL,
                        frame_id TEXT NOT NULL,
                        frame_sha256 TEXT NOT NULL,
                        evidence_sha256 TEXT NOT NULL,
                        evidence_ref TEXT NOT NULL,
                        vlm_provider TEXT NOT NULL CHECK(vlm_provider = 'qwen'),
                        vlm_model TEXT NOT NULL,
                        bbox_json TEXT NOT NULL,
                        depth_json TEXT NOT NULL,
                        tf_provenance_json TEXT NOT NULL,
                        confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
                        support_count INTEGER NOT NULL CHECK(support_count >= 1),
                        authority TEXT NOT NULL DEFAULT 'observation_only'
                            CHECK(authority = 'observation_only'),
                        execution_eligible INTEGER NOT NULL DEFAULT 0
                            CHECK(execution_eligible = 0),
                        confirmed_by_user_id TEXT,
                        confirmed_at TEXT,
                        superseded_by_observation_id TEXT,
                        superseded_reason TEXT,
                        record_version INTEGER NOT NULL DEFAULT 1
                            CHECK(record_version >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(landmark_id) REFERENCES scene_landmarks(landmark_id),
                        FOREIGN KEY(owner_user_id) REFERENCES users(user_id),
                        FOREIGN KEY(inspection_id)
                            REFERENCES scene_inspection_sessions(inspection_id),
                        FOREIGN KEY(checkpoint_id)
                            REFERENCES scene_inspection_checkpoints(checkpoint_id),
                        FOREIGN KEY(confirmed_by_user_id) REFERENCES users(user_id),
                        FOREIGN KEY(superseded_by_observation_id)
                            REFERENCES scene_landmark_observations(observation_id)
                    )
                    """,
                    """
                    CREATE TABLE scene_landmark_aliases (
                        alias_id TEXT PRIMARY KEY,
                        landmark_id TEXT NOT NULL,
                        owner_user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        alias TEXT NOT NULL,
                        normalized_alias TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
                        superseded_by_alias_id TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(landmark_id) REFERENCES scene_landmarks(landmark_id),
                        FOREIGN KEY(owner_user_id) REFERENCES users(user_id),
                        FOREIGN KEY(superseded_by_alias_id)
                            REFERENCES scene_landmark_aliases(alias_id)
                    )
                    """,
                    """
                    CREATE INDEX scene_inspections_site_owner_time_idx
                        ON scene_inspection_sessions(site_id, owner_user_id, updated_at DESC)
                    """,
                    """
                    CREATE INDEX scene_checkpoints_inspection_sequence_idx
                        ON scene_inspection_checkpoints(inspection_id, sequence)
                    """,
                    """
                    CREATE INDEX scene_landmarks_site_status_name_idx
                        ON scene_landmarks(site_id, status, normalized_name, updated_at DESC)
                    """,
                    """
                    CREATE INDEX scene_landmark_root_revision_idx
                        ON scene_landmarks(root_landmark_id, revision DESC)
                    """,
                    """
                    CREATE INDEX scene_observations_landmark_time_idx
                        ON scene_landmark_observations(landmark_id, created_at DESC)
                    """,
                    """
                    CREATE INDEX scene_aliases_site_lookup_idx
                        ON scene_landmark_aliases(site_id, normalized_alias, status)
                    """,
                ),
            ),
            (
                7,
                "mission_v2_autonomy_policy",
                (
                    """
                    CREATE TABLE autonomy_preferences (
                        user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        execution_target TEXT NOT NULL
                            CHECK(execution_target IN ('sim', 'real')),
                        mode TEXT NOT NULL CHECK(mode IN (
                            'ASK_EVERY_ACTION',
                            'ASK_RISKY_OR_COMPLEX',
                            'AUTO_WITHIN_LIMITS'
                        )),
                        version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(user_id, site_id, execution_target),
                        FOREIGN KEY(user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE TABLE autonomy_policy_decisions (
                        decision_id TEXT PRIMARY KEY,
                        operation TEXT NOT NULL,
                        principal_user_id TEXT NOT NULL,
                        principal_role TEXT NOT NULL
                            CHECK(principal_role IN ('USER', 'DEVELOPER', 'ADMIN')),
                        site_id TEXT NOT NULL,
                        robot_id TEXT,
                        execution_target TEXT NOT NULL
                            CHECK(execution_target IN ('sim', 'real')),
                        plan_id TEXT,
                        plan_revision INTEGER CHECK(
                            plan_revision IS NULL OR plan_revision >= 0
                        ),
                        node_id TEXT,
                        skill_id TEXT NOT NULL,
                        effect TEXT NOT NULL
                            CHECK(effect IN ('ALLOW', 'REQUIRE_CONFIRM', 'DENY')),
                        code TEXT NOT NULL,
                        explanation TEXT NOT NULL,
                        effective_risk TEXT,
                        failed_checks_json TEXT NOT NULL DEFAULT '[]',
                        request_json TEXT NOT NULL,
                        request_sha256 TEXT NOT NULL,
                        grant_id TEXT,
                        created_at TEXT NOT NULL
                    )
                    """,
                    """
                    CREATE INDEX autonomy_policy_decisions_plan_time_idx
                        ON autonomy_policy_decisions(
                            plan_id, plan_revision, created_at, decision_id
                        )
                    """,
                    """
                    CREATE INDEX autonomy_policy_decisions_subject_time_idx
                        ON autonomy_policy_decisions(
                            principal_user_id, site_id, created_at, decision_id
                        )
                    """,
                    """
                    CREATE TABLE autonomy_grants (
                        grant_id TEXT PRIMARY KEY,
                        authorization_id TEXT NOT NULL UNIQUE,
                        subject_user_id TEXT NOT NULL,
                        subject_role TEXT NOT NULL
                            CHECK(subject_role IN ('USER', 'DEVELOPER', 'ADMIN')),
                        site_id TEXT NOT NULL,
                        robot_id TEXT NOT NULL,
                        execution_target TEXT NOT NULL
                            CHECK(execution_target IN ('sim', 'real')),
                        plan_id TEXT NOT NULL,
                        plan_revision INTEGER NOT NULL CHECK(plan_revision >= 1),
                        action_scope_json TEXT NOT NULL,
                        max_risk TEXT NOT NULL
                            CHECK(max_risk IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                        max_distance_m REAL NOT NULL CHECK(max_distance_m >= 0),
                        max_speed_mps REAL NOT NULL CHECK(max_speed_mps >= 0),
                        max_rotation_rad REAL NOT NULL CHECK(max_rotation_rad >= 0),
                        max_angular_speed_rad_s REAL NOT NULL
                            CHECK(max_angular_speed_rad_s >= 0),
                        issued_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        revoked_at TEXT,
                        created_by_decision_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(
                            subject_user_id, site_id, robot_id,
                            execution_target, plan_id, plan_revision
                        ),
                        FOREIGN KEY(created_by_decision_id)
                            REFERENCES autonomy_policy_decisions(decision_id)
                    )
                    """,
                    """
                    CREATE INDEX autonomy_grants_plan_revision_idx
                        ON autonomy_grants(plan_id, plan_revision, revoked_at, expires_at)
                    """,
                    """
                    CREATE TABLE robot_site_bindings (
                        robot_id TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        execution_target TEXT NOT NULL
                            CHECK(execution_target IN ('sim', 'real')),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """,
                    """
                    CREATE INDEX robot_site_bindings_site_target_idx
                        ON robot_site_bindings(site_id, execution_target, robot_id)
                    """,
                ),
            ),
            (
                8,
                "daaam_observation_only_semantic_memory",
                (
                    """
                    CREATE TABLE daaam_memory_imports (
                        import_id TEXT PRIMARY KEY,
                        mission_id TEXT NOT NULL,
                        robot_id TEXT NOT NULL,
                        owner_user_id TEXT NOT NULL,
                        site_id TEXT NOT NULL,
                        robot_boot_id TEXT NOT NULL,
                        localization_epoch_id TEXT NOT NULL,
                        observation_session_id TEXT NOT NULL,
                        scene_id TEXT NOT NULL,
                        scene_revision TEXT NOT NULL,
                        captured_from TEXT NOT NULL,
                        captured_through TEXT NOT NULL,
                        payload_sha256 TEXT NOT NULL UNIQUE,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(mission_id) REFERENCES missions(mission_id),
                        FOREIGN KEY(owner_user_id) REFERENCES users(user_id)
                    )
                    """,
                    """
                    CREATE TABLE daaam_memory_entities (
                        import_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        semantic_id INTEGER NOT NULL CHECK(semantic_id >= 0),
                        normalized_label TEXT NOT NULL,
                        region_id TEXT,
                        last_observed_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY(import_id, entity_id),
                        UNIQUE(import_id, semantic_id),
                        FOREIGN KEY(import_id)
                            REFERENCES daaam_memory_imports(import_id)
                    )
                    """,
                    """
                    CREATE INDEX daaam_memory_owner_scene_time_idx
                        ON daaam_memory_imports(
                            owner_user_id, site_id, robot_id,
                            scene_id, scene_revision, captured_through DESC
                        )
                    """,
                    """
                    CREATE INDEX daaam_memory_entity_region_label_idx
                        ON daaam_memory_entities(
                            region_id, normalized_label, last_observed_at DESC
                        )
                    """,
                ),
            ),
            (
                9,
                "assistant_v2_conversation_mission_history",
                (
                    """
                    CREATE TABLE assistant_conversation_missions (
                        conversation_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        mission_draft_id TEXT NOT NULL UNIQUE,
                        mission_id TEXT UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(conversation_id, mission_draft_id),
                        FOREIGN KEY(conversation_id)
                            REFERENCES assistant_conversations(conversation_id),
                        FOREIGN KEY(task_id) REFERENCES assistant_turns(task_id)
                    )
                    """,
                    """
                    CREATE INDEX assistant_conversation_missions_conversation_idx
                        ON assistant_conversation_missions(
                            conversation_id, updated_at DESC, mission_draft_id
                        )
                    """,
                ),
            ),
            (
                10,
                "assistant_v2_turn_ordering",
                (
                    "ALTER TABLE assistant_turns ADD COLUMN turn_sequence INTEGER",
                    """
                    UPDATE assistant_turns AS current
                    SET turn_sequence = (
                        SELECT COUNT(*)
                        FROM assistant_turns AS prior
                        WHERE prior.conversation_id = current.conversation_id
                          AND (
                            prior.created_at < current.created_at
                            OR (
                                prior.created_at = current.created_at
                                AND prior.task_id <= current.task_id
                            )
                          )
                    )
                    """,
                    """
                    CREATE UNIQUE INDEX assistant_turns_conversation_sequence_uq
                        ON assistant_turns(conversation_id, turn_sequence)
                        WHERE turn_sequence IS NOT NULL
                    """,
                    """
                    CREATE INDEX assistant_messages_task_idx
                        ON assistant_messages(conversation_id, task_id, created_at)
                    """,
                ),
            ),
            (
                11,
                "assistant_v2_semantic_referents",
                (
                    """
                    CREATE TABLE assistant_semantic_referents (
                        referent_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        mission_id TEXT,
                        robot_id TEXT NOT NULL,
                        analysis_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL,
                        normalized_label TEXT NOT NULL,
                        region_id TEXT,
                        captured_at TEXT NOT NULL,
                        robot_boot_id TEXT NOT NULL,
                        localization_epoch_id TEXT NOT NULL,
                        scene_id TEXT,
                        scene_revision TEXT,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(conversation_id, analysis_id, hypothesis_id),
                        FOREIGN KEY(conversation_id)
                            REFERENCES assistant_conversations(conversation_id),
                        FOREIGN KEY(task_id) REFERENCES assistant_turns(task_id)
                    )
                    """,
                    """
                    CREATE INDEX assistant_semantic_referents_recent_idx
                        ON assistant_semantic_referents(
                            conversation_id, captured_at DESC, referent_id)
                    """,
                    """
                    CREATE INDEX assistant_semantic_referents_label_idx
                        ON assistant_semantic_referents(
                            conversation_id, normalized_label, captured_at DESC)
                    """,
                ),
            ),
        )
        for version, name, statements in migrations:
            applied = self._connection.execute(
                "SELECT name FROM schema_migrations WHERE version = ?",
                (version,),
            ).fetchone()
            if applied is not None:
                if str(applied["name"]) != name:
                    raise RuntimeError(
                        f"schema migration {version} name mismatch: "
                        f"stored={applied['name']!r}, expected={name!r}"
                    )
                continue
            # _initialize is serialized and the surrounding connection has no
            # pending write at this point.  A SAVEPOINT keeps DDL and the
            # ledger row atomic without committing unrelated initialization.
            savepoint = f"schema_migration_{version}"
            self._connection.execute(f"SAVEPOINT {savepoint}")
            try:
                for statement in statements:
                    self._connection.execute(statement)
                self._connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) "
                    "VALUES (?, ?, ?)",
                    (version, name, self._now()),
                )
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception:
                self._connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                self._connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _backfill_idempotency_keys(self) -> None:
        rows = self._connection.execute(
            "SELECT command_id, request_json, idempotency_key FROM commands ORDER BY created_at, command_id"
        ).fetchall()
        seen: set[str] = set()
        for row in rows:
            key = row["idempotency_key"]
            if not isinstance(key, str) or not key:
                try:
                    request = _decode(row["request_json"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    request = {}
                if isinstance(request, Mapping):
                    key = request.get("idempotencyKey") or request.get("idempotency_key")
                else:
                    key = None
            if not isinstance(key, str) or not key:
                key = f"legacy:{row['command_id']}"
            # A legacy database may contain duplicate keys because no unique
            # constraint existed.  Preserve every command while making the
            # first row's key authoritative for future retries.
            candidate = key
            if candidate in seen:
                candidate = f"{key}:legacy:{row['command_id']}"
                suffix = 1
                while candidate in seen:
                    suffix += 1
                    candidate = f"{key}:legacy:{row['command_id']}:{suffix}"
            seen.add(candidate)
            if row["idempotency_key"] != candidate:
                self._connection.execute(
                    "UPDATE commands SET idempotency_key = ? WHERE command_id = ?",
                    (candidate, row["command_id"]),
                )

    def _backfill_mapping_pointers(self) -> None:
        """Infer active pointers once for databases created before the table."""
        existing = self._connection.execute("SELECT COUNT(*) AS n FROM mapping_active").fetchone()["n"]
        if existing:
            return
        rows = self._connection.execute(
            "SELECT payload_json, mapping_id, version, updated_at FROM mappings "
            "WHERE status = 'active' ORDER BY updated_at, mapping_id, version"
        ).fetchall()
        for row in rows:
            try:
                payload = _decode(row["payload_json"])
                if not isinstance(payload, Mapping):
                    continue
                command_key = payload.get("commandKey") or payload.get("command_key")
            except (TypeError, ValueError, json.JSONDecodeError):
                command_key = None
            if isinstance(command_key, str) and command_key:
                self._connection.execute(
                    "INSERT INTO mapping_active(command_key, mapping_id, version, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(command_key) DO UPDATE SET "
                    "mapping_id=excluded.mapping_id, version=excluded.version, updated_at=excluded.updated_at",
                    (command_key, row["mapping_id"], row["version"], row["updated_at"]),
                )

    def _backfill_assistant_conversation_mission_links(self) -> None:
        """Recover durable Mission ownership for pre-migration Assistant turns."""

        rows = self._connection.execute(
            "SELECT task_id, conversation_id, result_json, created_at, updated_at "
            "FROM assistant_turns WHERE result_json IS NOT NULL "
            "ORDER BY created_at, task_id"
        ).fetchall()
        for row in rows:
            try:
                result = _decode(row["result_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(result, Mapping):
                continue
            mission_draft_id = result.get("missionDraftId")
            if not isinstance(mission_draft_id, str) or not mission_draft_id:
                continue
            existing = self._connection.execute(
                "SELECT conversation_id, task_id FROM assistant_conversation_missions "
                "WHERE mission_draft_id = ?",
                (mission_draft_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["conversation_id"]) != str(row["conversation_id"])
                    or str(existing["task_id"]) != str(row["task_id"])
                ):
                    raise RuntimeError(
                        "Mission draft history is bound to conflicting Assistant turns"
                    )
                continue
            mission = self._connection.execute(
                "SELECT mission_id FROM missions WHERE mission_id = ?",
                (mission_draft_id,),
            ).fetchone()
            self._connection.execute(
                "INSERT INTO assistant_conversation_missions("
                "conversation_id, task_id, mission_draft_id, mission_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["conversation_id"],
                    row["task_id"],
                    mission_draft_id,
                    None if mission is None else mission_draft_id,
                    row["created_at"],
                    row["updated_at"],
                ),
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _robot_payload(robot: RobotState | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(robot))

    @staticmethod
    def _capability_payload(snapshot: CapabilitySnapshot | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(snapshot))

    @staticmethod
    def _request_payload(request: CommandRequest | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(request))

    @staticmethod
    def _receipt_payload(receipt: CommandReceipt | Mapping[str, Any]) -> dict[str, Any]:
        return dict(_jsonable(receipt))

    def upsert_robot(self, robot: RobotState | Mapping[str, Any]) -> None:
        payload = self._robot_payload(robot)
        robot_id = payload.get("robotId") or payload.get("robot_id")
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robotId is required")
        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO robots(robot_id, payload_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(robot_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (robot_id, _encode(payload), now),
            )
            self._connection.commit()

    def list_robots(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute("SELECT payload_json FROM robots ORDER BY robot_id").fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    def get_robot(self, robot_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT payload_json FROM robots WHERE robot_id = ?", (robot_id,)).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def upsert_capabilities(self, snapshot: CapabilitySnapshot | Mapping[str, Any], robot_id: str | None = None) -> None:
        payload = self._capability_payload(snapshot)
        target = robot_id or payload.get("robotId") or payload.get("robot_id")
        if not isinstance(target, str) or not target:
            raise ValueError("robotId is required for capabilities")
        now = self._now()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO capabilities(robot_id, payload_json, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(robot_id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at
                """,
                (target, _encode(payload), now),
            )
            self._connection.commit()

    def get_capabilities(self, robot_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT payload_json FROM capabilities WHERE robot_id = ?", (robot_id,)).fetchone()
        return None if row is None else _decode(row["payload_json"])

    def _command_identity(self, request_payload: Mapping[str, Any]) -> tuple[str, str]:
        command_id = request_payload.get("commandId") or request_payload.get("command_id")
        idempotency_key = request_payload.get("idempotencyKey") or request_payload.get("idempotency_key")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("commandId is required")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotencyKey is required")
        return command_id, idempotency_key

    @staticmethod
    def _command_record_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "request": _decode(row["request_json"]),
            "receipt": _decode(row["receipt_json"]),
            "idempotencyKey": row["idempotency_key"],
        }

    def claim_command(
        self,
        request: CommandRequest | Mapping[str, Any],
        receipt: CommandReceipt | Mapping[str, Any] | None = None,
    ) -> CommandClaim:
        """Atomically claim a command idempotency key before adapter dispatch.

        The placeholder receipt makes a command visible to concurrent callers
        while the owner is talking to the robot.  SQLite's write transaction
        and unique index make this safe across threads and process restarts.
        """
        request_payload = self._request_payload(request)
        command_id, idempotency_key = self._command_identity(request_payload)
        if receipt is None:
            receipt_payload = {
                "commandId": command_id,
                "robotId": request_payload.get("robotId") or request_payload.get("robot_id"),
                "status": "accepted",
                "acceptedAt": self._now(),
                "startedAt": None,
                "completedAt": None,
                "result": None,
                "error": None,
            }
        else:
            receipt_payload = self._receipt_payload(receipt)
        status = receipt_payload.get("status")
        if hasattr(status, "value"):
            status = status.value
        if not isinstance(status, str) or not status:
            status = "accepted"
        receipt_payload["status"] = status
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                by_id = self._connection.execute(
                    "SELECT command_id, idempotency_key, request_json, receipt_json FROM commands WHERE command_id = ?",
                    (command_id,),
                ).fetchone()
                if by_id is not None:
                    self._connection.commit()
                    record = self._command_record_from_row(by_id)
                    return CommandClaim(False, record.get("idempotencyKey") != idempotency_key, record)
                by_key = self._connection.execute(
                    "SELECT command_id, idempotency_key, request_json, receipt_json FROM commands WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if by_key is not None:
                    self._connection.commit()
                    return CommandClaim(False, False, self._command_record_from_row(by_key))
                self._connection.execute(
                    "INSERT INTO commands(command_id, idempotency_key, request_json, receipt_json, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (command_id, idempotency_key, _encode(request_payload), _encode(receipt_payload), status, now, now),
                )
                self._connection.commit()
                return CommandClaim(True, False, {"request": request_payload, "receipt": receipt_payload, "idempotencyKey": idempotency_key})
            except sqlite3.IntegrityError:
                self._connection.rollback()
                row = self._connection.execute(
                    "SELECT command_id, idempotency_key, request_json, receipt_json FROM commands "
                    "WHERE command_id = ? OR idempotency_key = ? ORDER BY command_id LIMIT 1",
                    (command_id, idempotency_key),
                ).fetchone()
                if row is None:
                    raise
                record = self._command_record_from_row(row)
                return CommandClaim(False, row["command_id"] == command_id and row["idempotency_key"] != idempotency_key, record)
            except Exception:
                self._connection.rollback()
                raise

    def save_command(self, request: CommandRequest | Mapping[str, Any], receipt: CommandReceipt | Mapping[str, Any]) -> None:
        """Persist a receipt without replacing the immutable request identity."""
        request_payload = self._request_payload(request)
        receipt_payload = self._receipt_payload(receipt)
        command_id, idempotency_key = self._command_identity(request_payload)
        receipt_id = receipt_payload.get("commandId") or receipt_payload.get("command_id")
        if receipt_id != command_id:
            raise ValueError("receipt commandId does not match request")
        status = receipt_payload.get("status")
        if hasattr(status, "value"):
            status = status.value
        if not isinstance(status, str):
            status = str(status)
        now = self._now()
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json, idempotency_key FROM commands WHERE command_id = ?", (command_id,)
            ).fetchone()
            if row is None:
                # Keep direct repository users compatible, but still enforce
                # the same unique identity rules as the claim path.
                self._connection.execute(
                    "INSERT INTO commands(command_id, idempotency_key, request_json, receipt_json, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (command_id, idempotency_key, _encode(request_payload), _encode(receipt_payload), status, now, now),
                )
            else:
                if row["idempotency_key"] != idempotency_key:
                    raise ValueError("command identity conflict")
                self._connection.execute(
                    "UPDATE commands SET receipt_json = ?, status = ?, updated_at = ? WHERE command_id = ?",
                    (_encode(receipt_payload), status, now, command_id),
                )
            self._connection.commit()

    def get_command_record(self, command_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json, receipt_json, idempotency_key FROM commands WHERE command_id = ?", (command_id,)
            ).fetchone()
        if row is None:
            return None
        return self._command_record_from_row(row)

    def get_command_by_idempotency(self, idempotency_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json, receipt_json, idempotency_key FROM commands WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return None if row is None else self._command_record_from_row(row)

    def get_command(self, command_id: str) -> dict[str, Any] | None:
        record = self.get_command_record(command_id)
        return None if record is None else record["receipt"]

    def list_commands(self, *, limit: int = 100) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 1000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT receipt_json FROM commands ORDER BY created_at DESC LIMIT ?", (bounded,)
            ).fetchall()
        return [_decode(row["receipt_json"]) for row in rows]

    def mark_unowned_commands_for_reconciliation(self) -> int:
        """Annotate non-terminal commands after a process restart.

        The status is intentionally preserved (``accepted``/``running`` are
        useful facts for operators), but the error makes clear that this
        process cannot claim or replay the adapter operation.
        """
        message = "runtime ownership unavailable after process restart; reconciliation required"
        count = 0
        with self._lock:
            rows = self._connection.execute(
                "SELECT command_id, receipt_json, status FROM commands "
                "WHERE lower(status) IN ('accepted', 'running', 'pending')"
            ).fetchall()
            for row in rows:
                try:
                    receipt = _decode(row["receipt_json"])
                except (TypeError, ValueError, json.JSONDecodeError):
                    receipt = None
                if not isinstance(receipt, Mapping):
                    receipt = {"commandId": row["command_id"], "status": row["status"]}
                old_error = receipt.get("error")
                if isinstance(old_error, str) and "reconciliation" in old_error.lower():
                    continue
                receipt["error"] = f"{old_error}; {message}" if old_error else message
                self._connection.execute(
                    "UPDATE commands SET receipt_json = ?, updated_at = ? WHERE command_id = ?",
                    (_encode(receipt), self._now(), row["command_id"]),
                )
                count += 1
            if rows:
                self._connection.commit()
        return count

    @staticmethod
    def _mapping_payload(record: MappingRecord) -> tuple[dict[str, Any], str, str | None, dict[str, Any] | None]:
        mapping_payload = dict(_jsonable(record.mapping))
        probe_payload = None if record.probe is None else dict(_jsonable(record.probe))
        return mapping_payload, record.status, record.target_id, probe_payload

    def _mapping_values(self, record: MappingRecord) -> tuple[str, str, str, str, str | None, str | None]:
        mapping_payload, status, target_id, probe_payload = self._mapping_payload(record)
        mapping_id = mapping_payload.get("mappingId")
        version = mapping_payload.get("version")
        if not isinstance(mapping_id, str) or not isinstance(version, str):
            raise ValueError("mapping identity is required")
        command_key = mapping_payload.get("commandKey") or mapping_payload.get("command_key")
        if not isinstance(command_key, str) or not command_key:
            raise ValueError("mapping commandKey is required")
        return (
            mapping_id,
            version,
            _encode(mapping_payload),
            status,
            target_id,
            None if probe_payload is None else _encode(probe_payload),
        )

    def save_mapping(self, record: MappingRecord) -> None:
        self.save_mapping_batch([record], active=record.mapping if record.status == "active" else None)

    @staticmethod
    def _pointer_value(value: Any) -> tuple[str, str] | None:
        """Normalize a mapping/version pointer supplied by an API caller."""
        if value is None:
            return None
        if isinstance(value, MappingRecord):
            value = value.mapping
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("mapping pointer must contain mapping id and version")
            mapping_id, version = value
        else:
            payload = dict(_jsonable(value))
            mapping_id = payload.get("mappingId") or payload.get("mapping_id")
            version = payload.get("version")
        if not all(isinstance(item, str) and item for item in (mapping_id, version)):
            raise ValueError("mapping pointer requires mapping id and version")
        return str(mapping_id), str(version)

    @staticmethod
    def _mapping_command_key(value: Any) -> str:
        payload = dict(_jsonable(value.mapping if isinstance(value, MappingRecord) else value))
        command_key = payload.get("commandKey") or payload.get("command_key")
        if not isinstance(command_key, str) or not command_key:
            raise ValueError("mapping commandKey is required")
        return command_key

    def _check_mapping_row_transition(
        self,
        mapping_id: str,
        version: str,
        payload_json: str,
        status: str,
        target_id: str | None,
        probe_json: str | None,
        expected_probe: Any = _MAPPING_PROBE_UNSET,
    ) -> None:
        """Reject stale or backwards snapshots before an upsert can overwrite data."""
        row = self._connection.execute(
            "SELECT payload_json, status, target_id, probe_json FROM mappings WHERE mapping_id = ? AND version = ?",
            (mapping_id, version),
        ).fetchone()
        if row is None:
            return
        try:
            old_payload = _decode(row["payload_json"])
            new_payload = _decode(payload_json)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MappingStateConflict(mapping_id, version, str(row["status"]), status) from exc
        # MappingVersion is immutable once a version is published.  In
        # particular, a stale process must never replace an active row with a
        # draft carrying a different topic or target.
        if old_payload != new_payload or row["target_id"] != target_id:
            raise MappingStateConflict(mapping_id, version, str(row["status"]), status)
        old_status = str(row["status"])
        old_probe_json = row["probe_json"]

        def probe_value(raw: Any) -> Any:
            if raw is None:
                return None
            if isinstance(raw, str):
                try:
                    return _decode(raw)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return raw
            return _jsonable(raw)

        old_probe = probe_value(old_probe_json)
        if expected_probe is not _MAPPING_PROBE_UNSET:
            # Probe is an optimistic-CAS field.  A caller may intentionally
            # refresh an already-probed mapping, but only if the snapshot it
            # read before probing is still the durable snapshot.  This lets a
            # sequential capability refresh succeed while preventing an older
            # concurrent result from overwriting a newer one.
            if old_probe != probe_value(expected_probe):
                raise MappingStateConflict(mapping_id, version, str(row["status"]), status)
        else:
            probe_equal = old_probe == probe_value(probe_json)
            if not probe_equal:
                # Without an explicit probe CAS token, retain the conservative
                # repository behaviour for direct callers.
                if not (old_probe is None and probe_json is not None and old_status in {"draft", "validated"}):
                    raise MappingStateConflict(mapping_id, version, str(row["status"]), status)
        if old_status == status:
            # Draft creation is insert-only at the API boundary.  A second app
            # presenting the same draft is still a conflict, while idempotent
            # lifecycle persistence (validated/probed/active/superseded) is
            # harmless.
            if status == "draft":
                raise MappingStateConflict(mapping_id, version, old_status, status)
            return
        allowed = {
            "draft": {"validated"},
            "validated": {"probed"},
            "probed": {"active"},
            "active": {"superseded"},
            "superseded": {"active"},
        }
        if status not in allowed.get(old_status, set()):
            raise MappingStateConflict(mapping_id, version, old_status, status)

    def save_mapping_batch(
        self,
        records: list[MappingRecord] | tuple[MappingRecord, ...],
        *,
        active: Any = None,
        expected_active: Any = _MAPPING_POINTER_UNSET,
        audit_events: Iterable[Mapping[str, Any]] | None = None,
        expected_probes: Mapping[tuple[str, str], Any] | None = None,
    ) -> None:
        """Persist mapping snapshots, pointer CAS, and staged audits atomically.

        ``expected_active`` is intentionally distinct from ``None``: omitted
        means a caller does not participate in pointer CAS, while ``None``
        means it expects no pointer for the command being activated.  The
        compare and all row/pointer/audit writes happen under ``BEGIN
        IMMEDIATE`` so separate app processes cannot publish stale transitions.
        """
        values = [self._mapping_values(record) for record in records]
        active_pointer = None if active is None else self._pointer_value(active)
        command_key: str | None = None
        if active is not None:
            command_key = self._mapping_command_key(active)
        elif expected_active is not _MAPPING_POINTER_UNSET:
            if not values:
                raise ValueError("command key is required for mapping pointer CAS")
            # A non-activation CAS is uncommon, but deriving the key keeps the
            # repository API safe for direct callers as well.
            command_key = self._mapping_command_key(records[0].mapping)
        expected_pointer = (
            None if expected_active is None else self._pointer_value(expected_active)
        ) if expected_active is not _MAPPING_POINTER_UNSET else _MAPPING_POINTER_UNSET
        staged_audits = list(audit_events or ())
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                if expected_pointer is not _MAPPING_POINTER_UNSET:
                    row = self._connection.execute(
                        "SELECT mapping_id, version FROM mapping_active WHERE command_key = ?",
                        (command_key,),
                    ).fetchone()
                    actual_pointer = None if row is None else (str(row["mapping_id"]), str(row["version"]))
                    if actual_pointer != expected_pointer:
                        raise MappingPointerConflict(str(command_key), expected_pointer, actual_pointer)
                for mapping_id, version, payload_json, status, target_id, probe_json in values:
                    self._check_mapping_row_transition(
                        mapping_id,
                        version,
                        payload_json,
                        status,
                        target_id,
                        probe_json,
                        _MAPPING_PROBE_UNSET
                        if expected_probes is None or (mapping_id, version) not in expected_probes
                        else expected_probes[(mapping_id, version)],
                    )
                    self._connection.execute(
                        """
                        INSERT INTO mappings(mapping_id, version, payload_json, status, target_id, probe_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(mapping_id, version) DO UPDATE SET
                            payload_json=excluded.payload_json,
                            status=excluded.status,
                            target_id=excluded.target_id,
                            probe_json=excluded.probe_json,
                            updated_at=excluded.updated_at
                        """,
                        (mapping_id, version, payload_json, status, target_id, probe_json, self._now()),
                    )
                if active_pointer is not None:
                    mapping_id, version = active_pointer
                    if not all(isinstance(item, str) and item for item in (command_key, mapping_id, version)):
                        raise ValueError("active mapping identity is required")
                    self._connection.execute(
                        "INSERT INTO mapping_active(command_key, mapping_id, version, updated_at) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(command_key) DO UPDATE SET mapping_id=excluded.mapping_id, version=excluded.version, updated_at=excluded.updated_at",
                        (command_key, mapping_id, version, self._now()),
                    )
                for event in staged_audits:
                    payload = dict(_jsonable(event))
                    self._connection.execute(
                        "INSERT INTO audits(payload_json, created_at) VALUES (?, ?)",
                        (_encode(payload), self._now()),
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def list_mappings(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json, status, target_id, probe_json, updated_at FROM mappings ORDER BY updated_at, mapping_id, version"
            ).fetchall()
        return [self._mapping_row(row) for row in rows]

    def get_mapping(self, mapping_id: str, version: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json, status, target_id, probe_json, updated_at FROM mappings WHERE mapping_id = ? AND version = ?",
                (mapping_id, version),
            ).fetchone()
        return None if row is None else self._mapping_row(row)

    @staticmethod
    def _mapping_row(row: sqlite3.Row) -> dict[str, Any]:
        try:
            mapping = _decode(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            mapping = None
        try:
            probe = None if row["probe_json"] is None else _decode(row["probe_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            probe = None
        return {
            "mapping": mapping,
            "status": row["status"],
            "targetId": row["target_id"],
            "probe": probe,
            "updatedAt": row["updated_at"],
        }

    def get_active_pointer(self, command_key: str) -> tuple[str, str] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT mapping_id, version FROM mapping_active WHERE command_key = ?", (command_key,)
            ).fetchone()
        return None if row is None else (str(row["mapping_id"]), str(row["version"]))

    def list_active_pointers(self) -> dict[str, tuple[str, str]]:
        with self._lock:
            rows = self._connection.execute("SELECT command_key, mapping_id, version FROM mapping_active").fetchall()
        return {str(row["command_key"]): (str(row["mapping_id"]), str(row["version"])) for row in rows}

    def record_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(_jsonable(event))
        with self._lock:
            created = self._now()
            cursor = self._connection.execute(
                "INSERT INTO events(payload_json, created_at) VALUES (?, ?)", ( _encode(payload), created)
            )
            self._connection.commit()
            event_id = int(cursor.lastrowid)
        payload.setdefault("eventId", event_id)
        payload.setdefault("createdAt", created)
        return payload

    def list_events(self, *, after_id: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 2000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT event_id, payload_json, created_at FROM events WHERE event_id > ? ORDER BY event_id LIMIT ?",
                (int(after_id), bounded),
            ).fetchall()
        result = []
        for row in rows:
            payload = _decode(row["payload_json"])
            payload.setdefault("eventId", int(row["event_id"]))
            payload.setdefault("createdAt", row["created_at"])
            result.append(payload)
        return result

    def record_audit(self, event: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(_jsonable(event))
        with self._lock:
            created = self._now()
            cursor = self._connection.execute(
                "INSERT INTO audits(payload_json, created_at) VALUES (?, ?)", ( _encode(payload), created)
            )
            self._connection.commit()
            audit_id = int(cursor.lastrowid)
        payload.setdefault("auditId", audit_id)
        payload.setdefault("createdAt", created)
        return payload

    def list_audits(self, *, limit: int = 500) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 2000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT audit_id, payload_json, created_at FROM audits ORDER BY audit_id LIMIT ?", (bounded,)
            ).fetchall()
        result = []
        for row in rows:
            payload = _decode(row["payload_json"])
            payload.setdefault("auditId", int(row["audit_id"]))
            payload.setdefault("createdAt", row["created_at"])
            result.append(payload)
        return result

    def schema_migrations(self) -> list[dict[str, Any]]:
        """Return the durable migration ledger for diagnostics and tests."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT version, name, applied_at FROM schema_migrations ORDER BY version"
            ).fetchall()
        return [
            {
                "version": int(row["version"]),
                "name": str(row["name"]),
                "appliedAt": str(row["applied_at"]),
            }
            for row in rows
        ]

    @staticmethod
    def _user_record(row: sqlite3.Row, *, include_password: bool = False) -> dict[str, Any]:
        value = {
            "userId": str(row["user_id"]),
            "username": str(row["username"]),
            "displayName": str(row["display_name"]),
            "role": str(row["role"]),
            "siteId": str(row["site_id"]),
            "status": str(row["status"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }
        if include_password:
            value["passwordHash"] = str(row["password_hash"])
        return value

    def create_user(
        self,
        *,
        user_id: str,
        username: str,
        display_name: str,
        password_hash: str,
        role: str,
        site_id: str,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO users(user_id, username, display_name, password_hash, role, "
                    "site_id, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)",
                    (
                        user_id,
                        username,
                        display_name,
                        password_hash,
                        role,
                        site_id,
                        now,
                        now,
                    ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                raise ValueError("username is already registered") from exc
        created = self.get_user(user_id)
        if created is None:
            raise RuntimeError("created user disappeared")
        return created

    def get_user(self, user_id: str, *, include_password: bool = False) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return None if row is None else self._user_record(row, include_password=include_password)

    def get_user_by_username(
        self,
        username: str,
        *,
        include_password: bool = False,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
        return None if row is None else self._user_record(row, include_password=include_password)

    def list_users(self, *, limit: int = 200) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 1_000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM users ORDER BY username COLLATE NOCASE LIMIT ?",
                (bounded,),
            ).fetchall()
        return [self._user_record(row) for row in rows]

    def count_users(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) AS count FROM users").fetchone()
        return int(row["count"])

    def update_user_access(
        self,
        user_id: str,
        *,
        role: str | None = None,
        status: str | None = None,
        site_id: str | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        values: list[Any] = []
        if role is not None:
            assignments.append("role = ?")
            values.append(role)
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if site_id is not None:
            assignments.append("site_id = ?")
            values.append(site_id)
        if not assignments:
            return self.get_user(user_id)
        assignments.append("updated_at = ?")
        values.append(self._now())
        values.append(user_id)
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE users SET {', '.join(assignments)} WHERE user_id = ?",
                values,
            )
            self._connection.commit()
        return None if cursor.rowcount != 1 else self.get_user(user_id)

    def create_session(
        self,
        *,
        session_id: str,
        user_id: str,
        token_hash: str,
        csrf_token_hash: str,
        expires_at: str,
        user_agent_hash: str | None,
    ) -> None:
        now = self._now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO sessions(session_id, user_id, token_hash, csrf_token_hash, "
                "created_at, expires_at, last_seen_at, revoked_at, user_agent_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    session_id,
                    user_id,
                    token_hash,
                    csrf_token_hash,
                    now,
                    expires_at,
                    now,
                    user_agent_hash,
                ),
            )
            self._connection.commit()

    def get_session_by_token_hash(self, token_hash: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT s.*, u.username, u.display_name, u.role, u.site_id, u.status "
                "FROM sessions AS s JOIN users AS u ON u.user_id = s.user_id "
                "WHERE s.token_hash = ?",
                (token_hash,),
            ).fetchone()
        if row is None:
            return None
        return {
            "sessionId": str(row["session_id"]),
            "userId": str(row["user_id"]),
            "tokenHash": str(row["token_hash"]),
            "csrfTokenHash": str(row["csrf_token_hash"]),
            "createdAt": str(row["created_at"]),
            "expiresAt": str(row["expires_at"]),
            "lastSeenAt": str(row["last_seen_at"]),
            "revokedAt": row["revoked_at"],
            "userAgentHash": row["user_agent_hash"],
            "username": str(row["username"]),
            "displayName": str(row["display_name"]),
            "role": str(row["role"]),
            "siteId": str(row["site_id"]),
            "userStatus": str(row["status"]),
        }

    def touch_session(self, session_id: str, *, observed_at: str | None = None) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE session_id = ? AND revoked_at IS NULL",
                (observed_at or self._now(), session_id),
            )
            self._connection.commit()

    def revoke_session(self, session_id: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE sessions SET revoked_at = COALESCE(revoked_at, ?) WHERE session_id = ?",
                (self._now(), session_id),
            )
            self._connection.commit()
        return cursor.rowcount == 1

    def record_login_audit(
        self,
        *,
        username: str,
        outcome: str,
        user_id: str | None = None,
        remote_address: str | None = None,
        user_agent_hash: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        occurred_at = self._now()
        with self._lock:
            cursor = self._connection.execute(
                "INSERT INTO login_audits(user_id, username, outcome, remote_address, "
                "user_agent_hash, detail_json, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    username,
                    outcome,
                    remote_address,
                    user_agent_hash,
                    _encode(dict(detail or {})),
                    occurred_at,
                ),
            )
            self._connection.commit()
            audit_id = int(cursor.lastrowid)
        return {
            "loginAuditId": audit_id,
            "userId": user_id,
            "username": username,
            "outcome": outcome,
            "occurredAt": occurred_at,
        }

    def list_login_audits(self, *, limit: int = 200) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 1_000))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM login_audits ORDER BY login_audit_id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [
            {
                "loginAuditId": int(row["login_audit_id"]),
                "userId": row["user_id"],
                "username": str(row["username"]),
                "outcome": str(row["outcome"]),
                "remoteAddress": row["remote_address"],
                "userAgentHash": row["user_agent_hash"],
                "detail": _decode(row["detail_json"]),
                "occurredAt": str(row["occurred_at"]),
            }
            for row in rows
        ]

    def _bind_resource_owner_locked(
        self,
        resource_type: str,
        resource_id: str,
        *,
        owner_user_id: str,
        site_id: str,
        now: str,
    ) -> None:
        existing = self._connection.execute(
            "SELECT owner_user_id, site_id FROM resource_acl "
            "WHERE resource_type = ? AND resource_id = ?",
            (resource_type, resource_id),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["owner_user_id"]) != owner_user_id
                or str(existing["site_id"]) != site_id
            ):
                raise ValueError("resource owner binding is immutable")
            return
        self._connection.execute(
            "INSERT INTO resource_acl(resource_type, resource_id, owner_user_id, site_id, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (resource_type, resource_id, owner_user_id, site_id, now, now),
        )

    def bind_resource_owner(
        self,
        resource_type: str,
        resource_id: str,
        *,
        owner_user_id: str,
        site_id: str,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                self._bind_resource_owner_locked(
                    resource_type,
                    resource_id,
                    owner_user_id=owner_user_id,
                    site_id=site_id,
                    now=now,
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {
            "resourceType": resource_type,
            "resourceId": resource_id,
            "ownerUserId": owner_user_id,
            "siteId": site_id,
            "updatedAt": now,
        }

    def get_resource_owner(self, resource_type: str, resource_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM resource_acl WHERE resource_type = ? AND resource_id = ?",
                (resource_type, resource_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "resourceType": str(row["resource_type"]),
            "resourceId": str(row["resource_id"]),
            "ownerUserId": str(row["owner_user_id"]),
            "siteId": str(row["site_id"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    # ------------------------------------------------------------------
    # Mission V2 deterministic autonomy policy
    # ------------------------------------------------------------------

    def get_autonomy_preference(
        self,
        user_id: str,
        site_id: str,
        execution_target: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM autonomy_preferences "
                "WHERE user_id = ? AND site_id = ? AND execution_target = ?",
                (user_id, site_id, execution_target),
            ).fetchone()
        if row is None:
            return None
        return {
            "userId": str(row["user_id"]),
            "siteId": str(row["site_id"]),
            "executionTarget": str(row["execution_target"]),
            "mode": str(row["mode"]),
            "version": int(row["version"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def set_autonomy_preference(
        self,
        *,
        user_id: str,
        site_id: str,
        execution_target: str,
        mode: str,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT version FROM autonomy_preferences "
                    "WHERE user_id = ? AND site_id = ? AND execution_target = ?",
                    (user_id, site_id, execution_target),
                ).fetchone()
                if row is None:
                    if expected_version not in {None, 0}:
                        raise ValueError("autonomy preference version conflict")
                    self._connection.execute(
                        "INSERT INTO autonomy_preferences("
                        "user_id, site_id, execution_target, mode, version, created_at, updated_at"
                        ") VALUES (?, ?, ?, ?, 1, ?, ?)",
                        (user_id, site_id, execution_target, mode, now, now),
                    )
                else:
                    current = int(row["version"])
                    if expected_version is not None and expected_version != current:
                        raise ValueError("autonomy preference version conflict")
                    cursor = self._connection.execute(
                        "UPDATE autonomy_preferences SET mode = ?, version = ?, updated_at = ? "
                        "WHERE user_id = ? AND site_id = ? AND execution_target = ? "
                        "AND version = ?",
                        (
                            mode,
                            current + 1,
                            now,
                            user_id,
                            site_id,
                            execution_target,
                            current,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("autonomy preference version conflict")
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        value = self.get_autonomy_preference(user_id, site_id, execution_target)
        assert value is not None
        return value

    @staticmethod
    def _policy_decision_values(decision: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            decision["decisionId"],
            decision["operation"],
            decision["principalUserId"],
            decision["principalRole"],
            decision["siteId"],
            decision.get("robotId"),
            decision["executionTarget"],
            decision.get("planId"),
            decision.get("planRevision"),
            decision.get("nodeId"),
            decision["skillId"],
            decision["effect"],
            decision["code"],
            decision["explanation"],
            decision.get("effectiveRisk"),
            _encode(decision.get("failedChecks", [])),
            _encode(decision["request"]),
            decision["requestSha256"],
            decision.get("grantId"),
            decision["createdAt"],
        )

    def _insert_policy_decision_locked(self, decision: Mapping[str, Any]) -> None:
        values = self._policy_decision_values(decision)
        existing = self._connection.execute(
            "SELECT request_sha256, effect, code FROM autonomy_policy_decisions "
            "WHERE decision_id = ?",
            (decision["decisionId"],),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["request_sha256"]) != decision["requestSha256"]
                or str(existing["effect"]) != decision["effect"]
                or str(existing["code"]) != decision["code"]
            ):
                raise ValueError("policy decision ID is bound to another decision")
            return
        self._connection.execute(
            "INSERT INTO autonomy_policy_decisions("
            "decision_id, operation, principal_user_id, principal_role, site_id, "
            "robot_id, execution_target, plan_id, plan_revision, node_id, skill_id, "
            "effect, code, explanation, effective_risk, failed_checks_json, request_json, "
            "request_sha256, grant_id, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

    def record_autonomy_policy_decision(
        self,
        decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            try:
                self._insert_policy_decision_locked(decision)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return dict(decision)

    def list_autonomy_policy_decisions(
        self,
        *,
        plan_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        bounded = max(1, min(int(limit), 1_000))
        query = "SELECT * FROM autonomy_policy_decisions"
        values: tuple[Any, ...]
        if plan_id is None:
            values = (bounded,)
        else:
            query += " WHERE plan_id = ?"
            values = (plan_id, bounded)
        query += " ORDER BY created_at DESC, decision_id DESC LIMIT ?"
        with self._lock:
            rows = self._connection.execute(query, values).fetchall()
        return [
            {
                "decisionId": str(row["decision_id"]),
                "operation": str(row["operation"]),
                "principalUserId": str(row["principal_user_id"]),
                "principalRole": str(row["principal_role"]),
                "siteId": str(row["site_id"]),
                "robotId": row["robot_id"],
                "executionTarget": str(row["execution_target"]),
                "planId": row["plan_id"],
                "planRevision": row["plan_revision"],
                "nodeId": row["node_id"],
                "skillId": str(row["skill_id"]),
                "effect": str(row["effect"]),
                "code": str(row["code"]),
                "explanation": str(row["explanation"]),
                "effectiveRisk": row["effective_risk"],
                "failedChecks": _decode(row["failed_checks_json"]),
                "request": _decode(row["request_json"]),
                "requestSha256": str(row["request_sha256"]),
                "grantId": row["grant_id"],
                "createdAt": str(row["created_at"]),
            }
            for row in rows
        ]

    @staticmethod
    def _autonomy_grant_values(grant: Mapping[str, Any], now: str) -> tuple[Any, ...]:
        return (
            grant["grantId"],
            grant["authorizationId"],
            grant["subjectUserId"],
            grant["subjectRole"],
            grant["siteId"],
            grant["robotId"],
            grant["executionTarget"],
            grant["planId"],
            grant["planRevision"],
            _encode(grant["actionScope"]),
            grant["maxRisk"],
            grant["maxDistanceM"],
            grant["maxSpeedMps"],
            grant["maxRotationRad"],
            grant["maxAngularSpeedRadS"],
            grant["issuedAt"],
            grant["expiresAt"],
            grant.get("revokedAt"),
            grant["createdByDecisionId"],
            now,
            now,
        )

    def _insert_autonomy_grant_locked(
        self,
        grant: Mapping[str, Any],
        *,
        now: str,
    ) -> None:
        existing = self._connection.execute(
            "SELECT * FROM autonomy_grants WHERE grant_id = ? OR authorization_id = ?",
            (grant["grantId"], grant["authorizationId"]),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["grant_id"]) != grant["grantId"]
                or str(existing["authorization_id"]) != grant["authorizationId"]
                or str(existing["subject_user_id"]) != grant["subjectUserId"]
                or str(existing["site_id"]) != grant["siteId"]
                or str(existing["robot_id"]) != grant["robotId"]
                or str(existing["execution_target"]) != grant["executionTarget"]
                or str(existing["plan_id"]) != grant["planId"]
                or int(existing["plan_revision"]) != int(grant["planRevision"])
            ):
                raise ValueError("autonomy grant identity conflict")
            return
        self._connection.execute(
            "INSERT INTO autonomy_grants("
            "grant_id, authorization_id, subject_user_id, subject_role, site_id, robot_id, "
            "execution_target, plan_id, plan_revision, action_scope_json, max_risk, "
            "max_distance_m, max_speed_mps, max_rotation_rad, max_angular_speed_rad_s, "
            "issued_at, expires_at, revoked_at, created_by_decision_id, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._autonomy_grant_values(grant, now),
        )

    def save_autonomy_grant(self, grant: Mapping[str, Any]) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                self._insert_autonomy_grant_locked(grant, now=now)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        value = self.get_autonomy_grant(str(grant["grantId"]))
        assert value is not None
        return value

    @staticmethod
    def _autonomy_grant_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "grantId": str(row["grant_id"]),
            "authorizationId": str(row["authorization_id"]),
            "subjectUserId": str(row["subject_user_id"]),
            "subjectRole": str(row["subject_role"]),
            "siteId": str(row["site_id"]),
            "robotId": str(row["robot_id"]),
            "executionTarget": str(row["execution_target"]),
            "planId": str(row["plan_id"]),
            "planRevision": int(row["plan_revision"]),
            "actionScope": _decode(row["action_scope_json"]),
            "maxRisk": str(row["max_risk"]),
            "maxDistanceM": float(row["max_distance_m"]),
            "maxSpeedMps": float(row["max_speed_mps"]),
            "maxRotationRad": float(row["max_rotation_rad"]),
            "maxAngularSpeedRadS": float(row["max_angular_speed_rad_s"]),
            "issuedAt": str(row["issued_at"]),
            "expiresAt": str(row["expires_at"]),
            "revokedAt": row["revoked_at"],
            "createdByDecisionId": str(row["created_by_decision_id"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def get_autonomy_grant(self, grant_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM autonomy_grants WHERE grant_id = ?",
                (grant_id,),
            ).fetchone()
        return None if row is None else self._autonomy_grant_record(row)

    def get_autonomy_grant_by_authorization(
        self,
        authorization_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM autonomy_grants WHERE authorization_id = ?",
                (authorization_id,),
            ).fetchone()
        return None if row is None else self._autonomy_grant_record(row)

    def revoke_autonomy_grants(
        self,
        plan_id: str,
        *,
        except_revision: int | None = None,
    ) -> int:
        now = self._now()
        where = "plan_id = ? AND revoked_at IS NULL"
        values: list[Any] = [now, now, plan_id]
        if except_revision is not None:
            where += " AND plan_revision <> ?"
            values.append(except_revision)
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE autonomy_grants SET revoked_at = ?, updated_at = ? WHERE {where}",
                values,
            )
            self._connection.commit()
        return int(cursor.rowcount)

    def bind_robot_site(
        self,
        robot_id: str,
        *,
        site_id: str,
        execution_target: str,
    ) -> dict[str, Any]:
        now = self._now()
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT site_id, execution_target, created_at FROM robot_site_bindings "
                    "WHERE robot_id = ?",
                    (robot_id,),
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        "INSERT INTO robot_site_bindings("
                        "robot_id, site_id, execution_target, created_at, updated_at"
                        ") VALUES (?, ?, ?, ?, ?)",
                        (robot_id, site_id, execution_target, now, now),
                    )
                    created_at = now
                else:
                    if (
                        str(row["site_id"]) != site_id
                        or str(row["execution_target"]) != execution_target
                    ):
                        raise ValueError("robot site binding is immutable")
                    created_at = str(row["created_at"])
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {
            "robotId": robot_id,
            "siteId": site_id,
            "executionTarget": execution_target,
            "createdAt": created_at,
            "updatedAt": now,
        }

    def get_robot_site_binding(self, robot_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM robot_site_bindings WHERE robot_id = ?",
                (robot_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "robotId": str(row["robot_id"]),
            "siteId": str(row["site_id"]),
            "executionTarget": str(row["execution_target"]),
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def create_assistant_conversation(
        self,
        conversation_id: str,
        robot_id: str | None,
        requested_by: str,
        locale: str,
        summary: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        now = self._now()
        legacy_robot_id = robot_id or "__assistant_unbound__"
        with self._lock:
            self._connection.execute(
                "INSERT INTO assistant_conversations("
                "conversation_id, robot_id, robot_id_v2, requested_by, locale, summary, title, "
                "status, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, 'idle', ?, ?)",
                (
                    conversation_id,
                    legacy_robot_id,
                    robot_id,
                    requested_by,
                    locale,
                    summary,
                    title,
                    now,
                    now,
                ),
            )
            self._connection.commit()
        return {
            "conversationId": conversation_id,
            "robotId": robot_id,
            "requestedBy": requested_by,
            "locale": locale,
            "summary": summary,
            "title": title,
            "status": "idle",
            "archived": False,
            "archivedAt": None,
            "deleted": False,
            "deletedAt": None,
            "createdAt": now,
            "updatedAt": now,
        }

    @staticmethod
    def _assistant_conversation_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "conversationId": row["conversation_id"],
            "robotId": row["robot_id_v2"],
            "requestedBy": row["requested_by"],
            "locale": row["locale"],
            "summary": row["summary"],
            "title": row["title"],
            "status": row["status"],
            "archived": row["archived_at"] is not None,
            "archivedAt": row["archived_at"],
            "deleted": row["deleted_at"] is not None,
            "deletedAt": row["deleted_at"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def list_assistant_conversations(
        self,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        predicates: list[str] = []
        if not include_archived:
            predicates.append("archived_at IS NULL")
        if not include_deleted:
            predicates.append("deleted_at IS NULL")
        where = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM assistant_conversations"
                f"{where} ORDER BY updated_at DESC, conversation_id"
            ).fetchall()
        return [self._assistant_conversation_row(row) for row in rows]

    @staticmethod
    def _assistant_history_search_pattern(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    def list_assistant_conversation_page(
        self,
        *,
        lifecycle: str = "active",
        query: str | None = None,
        before_updated_at: str | None = None,
        before_conversation_id: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Return one stable, cursor-addressable page of conversation roots.

        Mission drafts and Mission executions are deliberately not projected as
        synthetic conversations.  Callers may enrich each root through
        :meth:`get_assistant_conversation_mission_links`.
        """

        lifecycle_predicates = {
            "active": ("deleted_at IS NULL", "archived_at IS NULL"),
            "archived": ("deleted_at IS NULL", "archived_at IS NOT NULL"),
            "deleted": ("deleted_at IS NOT NULL",),
        }
        if lifecycle not in lifecycle_predicates:
            raise ValueError("conversation lifecycle must be active, archived, or deleted")
        if (before_updated_at is None) != (before_conversation_id is None):
            raise ValueError("conversation cursor requires updatedAt and conversationId")
        bounded_limit = max(1, min(int(limit), 100))
        predicates = list(lifecycle_predicates[lifecycle])
        parameters: list[Any] = []
        normalized_query = None if query is None else query.strip()
        if normalized_query:
            pattern = self._assistant_history_search_pattern(normalized_query)
            predicates.append(
                "(COALESCE(title, '') LIKE ? ESCAPE '\\' OR EXISTS ("
                "SELECT 1 FROM assistant_messages message "
                "WHERE message.conversation_id = assistant_conversations.conversation_id "
                "AND message.role = 'user' AND message.content LIKE ? ESCAPE '\\'))"
            )
            parameters.extend((pattern, pattern))
        if before_updated_at is not None and before_conversation_id is not None:
            predicates.append(
                "(updated_at < ? OR (updated_at = ? AND conversation_id < ?))"
            )
            parameters.extend(
                (before_updated_at, before_updated_at, before_conversation_id)
            )
        where = " AND ".join(predicates)
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM assistant_conversations "
                f"WHERE {where} ORDER BY updated_at DESC, conversation_id DESC LIMIT ?",
                (*parameters, bounded_limit),
            ).fetchall()
        return [self._assistant_conversation_row(row) for row in rows]

    def get_assistant_conversation_mission_links(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT link.task_id, link.mission_draft_id, link.mission_id, "
                "draft.status AS draft_status, mission.status AS mission_status, "
                "turn.turn_sequence, link.created_at, link.updated_at "
                "FROM assistant_conversation_missions link "
                "JOIN assistant_turns turn ON turn.task_id = link.task_id "
                "LEFT JOIN mission_drafts draft "
                "ON draft.draft_id = link.mission_draft_id "
                "LEFT JOIN missions mission "
                "ON mission.mission_id = COALESCE(link.mission_id, link.mission_draft_id) "
                "WHERE link.conversation_id = ? "
                "ORDER BY COALESCE(turn.turn_sequence, 2147483647), "
                "link.created_at, link.mission_draft_id",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "taskId": row["task_id"],
                "turnSequence": row["turn_sequence"],
                "missionDraftId": row["mission_draft_id"],
                "missionId": (
                    row["mission_id"]
                    if row["mission_id"] is not None
                    else row["mission_draft_id"]
                    if row["mission_status"] is not None
                    else None
                ),
                "draftStatus": row["draft_status"],
                "missionStatus": row["mission_status"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
            }
            for row in rows
        ]

    def bind_assistant_conversation_mission(
        self,
        *,
        conversation_id: str,
        task_id: str,
        mission_draft_id: str,
        mission_id: str | None = None,
    ) -> None:
        """Idempotently bind one Mission identity to its conversation turn."""

        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    "SELECT conversation_id, task_id, mission_id "
                    "FROM assistant_conversation_missions WHERE mission_draft_id = ?",
                    (mission_draft_id,),
                ).fetchone()
                if existing is None:
                    self._connection.execute(
                        "INSERT INTO assistant_conversation_missions("
                        "conversation_id, task_id, mission_draft_id, mission_id, "
                        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            conversation_id,
                            task_id,
                            mission_draft_id,
                            mission_id,
                            now,
                            now,
                        ),
                    )
                else:
                    if (
                        str(existing["conversation_id"]) != conversation_id
                        or str(existing["task_id"]) != task_id
                    ):
                        raise AssistantConversationConflict(
                            "Mission draft is already bound to another conversation turn"
                        )
                    existing_mission_id = existing["mission_id"]
                    if (
                        existing_mission_id is not None
                        and mission_id is not None
                        and str(existing_mission_id) != mission_id
                    ):
                        raise AssistantConversationConflict(
                            "Mission draft is already bound to another Mission"
                        )
                    if existing_mission_id is None and mission_id is not None:
                        self._connection.execute(
                            "UPDATE assistant_conversation_missions "
                            "SET mission_id = ?, updated_at = ? "
                            "WHERE mission_draft_id = ?",
                            (mission_id, now, mission_draft_id),
                        )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def bind_approved_mission_to_conversation(
        self,
        mission_draft_id: str,
        mission_id: str,
    ) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE assistant_conversation_missions "
                "SET mission_id = ?, updated_at = ? "
                "WHERE mission_draft_id = ? "
                "AND (mission_id IS NULL OR mission_id = ?)",
                (mission_id, self._now(), mission_draft_id, mission_id),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def _assistant_conversation_lifecycle_blocker_locked(
        self,
        conversation_id: str,
    ) -> str | None:
        active_turn = self._connection.execute(
            "SELECT task_id FROM assistant_turns WHERE conversation_id = ? "
            "AND status IN ('queued', 'running') LIMIT 1",
            (conversation_id,),
        ).fetchone()
        if active_turn is not None:
            return "active_turn"
        active_mission = self._connection.execute(
            "SELECT mission.mission_id FROM assistant_conversation_missions link "
            "JOIN missions mission "
            "ON mission.mission_id = COALESCE(link.mission_id, link.mission_draft_id) "
            "WHERE link.conversation_id = ? "
            "AND mission.status NOT IN ('completed', 'failed', 'cancelled') LIMIT 1",
            (conversation_id,),
        ).fetchone()
        return "active_mission" if active_mission is not None else None

    def assistant_conversation_lifecycle_blocker(
        self,
        conversation_id: str,
    ) -> str | None:
        with self._lock:
            return self._assistant_conversation_lifecycle_blocker_locked(
                conversation_id
            )

    def delete_assistant_conversations(
        self,
        conversation_ids: Iterable[str],
    ) -> dict[str, Any]:
        """Soft-delete an authorized batch while skipping active work."""

        normalized = list(dict.fromkeys(str(item) for item in conversation_ids))
        now = self._now()
        deleted: list[str] = []
        skipped: list[dict[str, str]] = []
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                for conversation_id in normalized:
                    row = self._connection.execute(
                        "SELECT deleted_at FROM assistant_conversations "
                        "WHERE conversation_id = ?",
                        (conversation_id,),
                    ).fetchone()
                    if row is None:
                        skipped.append(
                            {"conversationId": conversation_id, "reason": "not_found"}
                        )
                        continue
                    blocker = self._assistant_conversation_lifecycle_blocker_locked(
                        conversation_id
                    )
                    if blocker is not None:
                        skipped.append(
                            {"conversationId": conversation_id, "reason": blocker}
                        )
                        continue
                    if row["deleted_at"] is None:
                        self._connection.execute(
                            "UPDATE assistant_conversations "
                            "SET deleted_at = ?, updated_at = ? WHERE conversation_id = ?",
                            (now, now, conversation_id),
                        )
                    deleted.append(conversation_id)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {"deletedConversationIds": deleted, "skipped": skipped}

    def get_assistant_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM assistant_conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return None if row is None else self._assistant_conversation_row(row)

    def update_assistant_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None | object = _ASSISTANT_TITLE_UNSET,
        archived: bool | None = None,
        deleted: bool | None = None,
    ) -> dict[str, Any] | None:
        """Atomically update display metadata and the recoverable lifecycle.

        Messages, turns, drafts, Missions, and evidence are deliberately not
        cascaded.  ``deleted_at`` is a tombstone that can be cleared through
        the same authenticated PATCH boundary.
        """

        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT * FROM assistant_conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if row is None:
                    self._connection.rollback()
                    return None

                current_deleted = row["deleted_at"] is not None
                restoring = deleted is False and current_deleted
                lifecycle_requested = archived is not None or deleted is not None
                title_requested = title is not _ASSISTANT_TITLE_UNSET
                if current_deleted and not restoring and (title_requested or archived is not None):
                    raise AssistantConversationConflict(
                        "deleted conversation must be restored before it can be changed"
                    )

                lifecycle_blocker = (
                    self._assistant_conversation_lifecycle_blocker_locked(
                        conversation_id
                    )
                    if lifecycle_requested
                    else None
                )
                desired_archived_at = row["archived_at"]
                desired_deleted_at = row["deleted_at"]
                if archived is True and desired_archived_at is None:
                    desired_archived_at = now
                elif archived is False:
                    desired_archived_at = None
                if deleted is True and desired_deleted_at is None:
                    desired_deleted_at = now
                elif deleted is False:
                    desired_deleted_at = None

                lifecycle_changed = (
                    desired_archived_at != row["archived_at"]
                    or desired_deleted_at != row["deleted_at"]
                )
                hiding_conversation = (
                    desired_archived_at is not None
                    and desired_archived_at != row["archived_at"]
                ) or (
                    desired_deleted_at is not None
                    and desired_deleted_at != row["deleted_at"]
                )
                if hiding_conversation and lifecycle_blocker == "active_turn":
                    raise AssistantConversationConflict(
                        "conversation lifecycle cannot change while a turn is active"
                    )
                if hiding_conversation and lifecycle_blocker == "active_mission":
                    raise AssistantConversationConflict(
                        "conversation lifecycle cannot change while a linked Mission is active"
                    )

                desired_title = row["title"] if title is _ASSISTANT_TITLE_UNSET else title
                if (
                    desired_title == row["title"]
                    and not lifecycle_changed
                ):
                    self._connection.commit()
                    return self._assistant_conversation_row(row)

                self._connection.execute(
                    "UPDATE assistant_conversations SET title = ?, archived_at = ?, "
                    "deleted_at = ?, updated_at = ? WHERE conversation_id = ?",
                    (
                        desired_title,
                        desired_archived_at,
                        desired_deleted_at,
                        now,
                        conversation_id,
                    ),
                )
                updated = self._connection.execute(
                    "SELECT * FROM assistant_conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return None if updated is None else self._assistant_conversation_row(updated)

    def list_assistant_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT message.message_id, message.task_id, message.role, "
                "message.content, message.modality, message.transcription_id, "
                "message.created_at, turn.turn_sequence "
                "FROM assistant_messages AS message "
                "LEFT JOIN assistant_turns AS turn ON turn.task_id = message.task_id "
                "WHERE message.conversation_id = ? "
                "ORDER BY COALESCE(turn.turn_sequence, 2147483647), "
                "message.created_at, message.message_id",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "messageId": row["message_id"],
                "taskId": row["task_id"],
                "turnSequence": row["turn_sequence"],
                "role": row["role"],
                "content": row["content"],
                "modality": row["modality"],
                "transcriptionId": row["transcription_id"],
                "createdAt": row["created_at"],
            }
            for row in rows
        ]

    def record_assistant_semantic_referents(
        self,
        *,
        conversation_id: str,
        robot_id: str,
        referents: Iterable[Mapping[str, Any]],
    ) -> int:
        """Idempotently persist coordinate-free visual entities for one turn."""

        payloads = [dict(item) for item in referents]
        if not payloads:
            return 0
        now = self._now()
        inserted = 0
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                conversation = self._connection.execute(
                    "SELECT robot_id FROM assistant_conversations "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if conversation is None or str(conversation["robot_id"]) != robot_id:
                    raise ValueError("semantic referent conversation binding is invalid")
                for payload in payloads:
                    referent_id = str(payload.get("referentId") or "")
                    task_id = str(payload.get("sourceTaskId") or "")
                    analysis_id = str(payload.get("analysisId") or "")
                    hypothesis_id = str(payload.get("hypothesisId") or "")
                    label = str(payload.get("label") or "")
                    captured_at = str(payload.get("capturedAt") or "")
                    robot_boot_id = str(payload.get("robotBootId") or "")
                    localization_epoch_id = str(
                        payload.get("localizationEpochId") or ""
                    )
                    if not all(
                        value and value == value.strip()
                        for value in (
                            referent_id,
                            task_id,
                            analysis_id,
                            hypothesis_id,
                            label,
                            captured_at,
                            robot_boot_id,
                            localization_epoch_id,
                        )
                    ):
                        raise ValueError("semantic referent identity is incomplete")
                    turn = self._connection.execute(
                        "SELECT conversation_id FROM assistant_turns WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    if turn is None or str(turn["conversation_id"]) != conversation_id:
                        raise ValueError("semantic referent task binding is invalid")
                    existing = self._connection.execute(
                        "SELECT referent_id, payload_json "
                        "FROM assistant_semantic_referents "
                        "WHERE conversation_id = ? AND analysis_id = ? "
                        "AND hypothesis_id = ?",
                        (conversation_id, analysis_id, hypothesis_id),
                    ).fetchone()
                    encoded = _encode(payload)
                    if existing is not None:
                        if (
                            str(existing["referent_id"]) != referent_id
                            or str(existing["payload_json"]) != encoded
                        ):
                            raise ValueError(
                                "semantic referent identity is bound to different evidence"
                            )
                        continue
                    scene_ref = payload.get("sceneRef")
                    scene_id = (
                        scene_ref.get("sceneId")
                        if isinstance(scene_ref, Mapping)
                        else None
                    )
                    scene_revision = (
                        _encode(scene_ref.get("revision"))
                        if isinstance(scene_ref, Mapping)
                        and scene_ref.get("revision") is not None
                        else None
                    )
                    self._connection.execute(
                        "INSERT INTO assistant_semantic_referents("
                        "referent_id, conversation_id, task_id, mission_id, robot_id, "
                        "analysis_id, hypothesis_id, normalized_label, region_id, "
                        "captured_at, robot_boot_id, localization_epoch_id, scene_id, "
                        "scene_revision, payload_json, created_at, updated_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            referent_id,
                            conversation_id,
                            task_id,
                            payload.get("sourceMissionId"),
                            robot_id,
                            analysis_id,
                            hypothesis_id,
                            " ".join(label.split()).casefold(),
                            payload.get("regionId"),
                            captured_at,
                            robot_boot_id,
                            localization_epoch_id,
                            scene_id,
                            scene_revision,
                            encoded,
                            now,
                            now,
                        ),
                    )
                    inserted += 1
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return inserted

    def list_assistant_semantic_referents(
        self,
        conversation_id: str,
        *,
        robot_boot_id: str | None = None,
        localization_epoch_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        predicates = ["conversation_id = ?"]
        parameters: list[Any] = [conversation_id]
        if robot_boot_id is not None:
            predicates.append("robot_boot_id = ?")
            parameters.append(robot_boot_id)
        if localization_epoch_id is not None:
            predicates.append("localization_epoch_id = ?")
            parameters.append(localization_epoch_id)
        parameters.append(max(1, min(int(limit), 100)))
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM assistant_semantic_referents WHERE "
                + " AND ".join(predicates)
                + " ORDER BY captured_at DESC, referent_id DESC LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return [_decode(row["payload_json"]) for row in rows]

    def get_assistant_conversation_mission_link_by_mission(
        self,
        mission_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT conversation_id, task_id, mission_draft_id, mission_id, "
                "created_at, updated_at FROM assistant_conversation_missions "
                "WHERE mission_id = ? OR mission_draft_id = ? LIMIT 1",
                (mission_id, mission_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "conversationId": str(row["conversation_id"]),
            "taskId": str(row["task_id"]),
            "missionDraftId": str(row["mission_draft_id"]),
            "missionId": row["mission_id"],
            "createdAt": str(row["created_at"]),
            "updatedAt": str(row["updated_at"]),
        }

    def append_assistant_mission_message(
        self,
        *,
        mission_id: str,
        message_id: str,
        content: str,
        conversation_status: str = "idle",
    ) -> bool:
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                link = self._connection.execute(
                    "SELECT conversation_id, task_id FROM "
                    "assistant_conversation_missions "
                    "WHERE mission_id = ? OR mission_draft_id = ? LIMIT 1",
                    (mission_id, mission_id),
                ).fetchone()
                if link is None:
                    self._connection.rollback()
                    return False
                cursor = self._connection.execute(
                    "INSERT OR IGNORE INTO assistant_messages("
                    "message_id, conversation_id, task_id, role, content, modality, "
                    "transcription_id, created_at) VALUES (?, ?, ?, 'assistant', ?, "
                    "'text', NULL, ?)",
                    (
                        message_id,
                        str(link["conversation_id"]),
                        str(link["task_id"]),
                        content,
                        now,
                    ),
                )
                if cursor.rowcount == 1:
                    self._connection.execute(
                        "UPDATE assistant_conversations SET status = ?, updated_at = ? "
                        "WHERE conversation_id = ?",
                        (conversation_status, now, str(link["conversation_id"])),
                    )
                self._connection.commit()
                return cursor.rowcount == 1
            except Exception:
                self._connection.rollback()
                raise

    @staticmethod
    def _assistant_conversation_summary_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "summaryId": str(row["summary_id"]),
            "revision": int(row["summary_revision"]),
            "text": str(row["summary_text"]),
            "coveredThroughTurn": int(row["covered_through_turn"]),
            "generatedAt": str(row["generated_at"]),
            "method": str(row["method"]),
            "contentSha256": str(row["content_sha256"]),
            "sourceSha256": str(row["source_sha256"]),
            "sourceMessageIds": _decode(row["source_message_ids_json"]),
            "createdAt": str(row["created_at"]),
        }

    def get_assistant_conversation_summary(
        self,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        """Return the newest append-only H-IDE summary revision."""

        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM assistant_conversation_summaries "
                "WHERE conversation_id = ? ORDER BY summary_revision DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return None if row is None else self._assistant_conversation_summary_row(row)

    def record_assistant_conversation_memory_context(
        self,
        conversation_id: str,
        *,
        resource_id: str,
        query_sha256: str,
        recent_message_ids: Iterable[str],
        selected_memory_ids: Iterable[str],
        summary: Mapping[str, Any] | None,
        source_sha256: str | None = None,
        source_message_ids: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Append one summary revision when needed and one context-use audit.

        Raw messages are never rewritten or deleted.  A summary is another
        append-only audit projection of those messages, while every request to
        the internal conversation router receives a monotonically increasing
        context revision scoped to this H-IDE conversation.
        """

        recent_ids = [str(value) for value in recent_message_ids]
        selected_ids = [str(value) for value in selected_memory_ids]
        source_ids = [str(value) for value in source_message_ids]
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                conversation = self._connection.execute(
                    "SELECT conversation_id FROM assistant_conversations "
                    "WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if conversation is None:
                    raise ValueError("assistant conversation not found")

                latest = self._connection.execute(
                    "SELECT * FROM assistant_conversation_summaries "
                    "WHERE conversation_id = ? "
                    "ORDER BY summary_revision DESC LIMIT 1",
                    (conversation_id,),
                ).fetchone()
                if summary is not None:
                    candidate_coverage = int(summary["coveredThroughTurn"])
                    candidate_source = str(source_sha256 or "")
                    should_append = latest is None or (
                        candidate_coverage >= int(latest["covered_through_turn"])
                        and (
                            candidate_coverage > int(latest["covered_through_turn"])
                            or candidate_source != str(latest["source_sha256"])
                        )
                    )
                    if should_append:
                        summary_revision = (
                            1 if latest is None else int(latest["summary_revision"]) + 1
                        )
                        self._connection.execute(
                            "INSERT INTO assistant_conversation_summaries("
                            "conversation_id, summary_revision, summary_id, summary_text, "
                            "covered_through_turn, generated_at, method, content_sha256, "
                            "source_sha256, source_message_ids_json, created_at"
                            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                conversation_id,
                                summary_revision,
                                str(summary["summaryId"]),
                                str(summary["text"]),
                                candidate_coverage,
                                str(summary["generatedAt"]),
                                str(summary["method"]),
                                str(summary["contentSha256"]),
                                candidate_source,
                                _encode(source_ids),
                                now,
                            ),
                        )
                        latest = self._connection.execute(
                            "SELECT * FROM assistant_conversation_summaries "
                            "WHERE conversation_id = ? AND summary_revision = ?",
                            (conversation_id, summary_revision),
                        ).fetchone()

                revision_row = self._connection.execute(
                    "SELECT COALESCE(MAX(context_revision), 0) AS revision "
                    "FROM assistant_conversation_memory_audits WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                context_revision = int(revision_row["revision"]) + 1
                audit_id = f"conversation-memory-audit-{uuid.uuid4().hex}"
                self._connection.execute(
                    "INSERT INTO assistant_conversation_memory_audits("
                    "audit_id, conversation_id, context_revision, resource_id, "
                    "summary_revision, recent_message_ids_json, "
                    "selected_memory_ids_json, query_sha256, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        audit_id,
                        conversation_id,
                        context_revision,
                        resource_id,
                        None if latest is None else int(latest["summary_revision"]),
                        _encode(recent_ids),
                        _encode(selected_ids),
                        query_sha256,
                        now,
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return {
            "auditId": audit_id,
            "contextRevision": context_revision,
            "summary": (
                None
                if latest is None
                else self._assistant_conversation_summary_row(latest)
            ),
        }

    def list_assistant_conversation_memory_audits(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM assistant_conversation_memory_audits "
                "WHERE conversation_id = ? ORDER BY context_revision",
                (conversation_id,),
            ).fetchall()
        return [
            {
                "auditId": str(row["audit_id"]),
                "conversationId": str(row["conversation_id"]),
                "contextRevision": int(row["context_revision"]),
                "resourceId": str(row["resource_id"]),
                "summaryRevision": (
                    None
                    if row["summary_revision"] is None
                    else int(row["summary_revision"])
                ),
                "recentMessageIds": _decode(row["recent_message_ids_json"]),
                "selectedMemoryIds": _decode(row["selected_memory_ids_json"]),
                "querySha256": str(row["query_sha256"]),
                "createdAt": str(row["created_at"]),
            }
            for row in rows
        ]

    @staticmethod
    def _assistant_turn_row(row: sqlite3.Row) -> dict[str, Any]:
        result = _decode(row["result_json"]) if row["result_json"] else None
        return {
            "taskId": row["task_id"],
            "conversationId": row["conversation_id"],
            "turnSequence": row["turn_sequence"],
            "status": row["status"],
            "request": _decode(row["request_json"]),
            "result": result,
            "error": row["error"],
            "cancellationRequested": bool(row["cancellation_requested"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def list_assistant_turns(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM assistant_turns WHERE conversation_id = ? "
                "ORDER BY COALESCE(turn_sequence, 2147483647), created_at, task_id",
                (conversation_id,),
            ).fetchall()
        return [self._assistant_turn_row(row) for row in rows]

    def latest_visual_diagnostic(self, *, site_id: str | None) -> dict[str, Any] | None:
        """Return bounded visual evidence metadata for developer diagnostics.

        This intentionally projects no prompt, image, pose, model response, or
        raw evidence reference.  A site-scoped developer can only learn that a
        visual analysis exists, how long its persisted turn took, and which
        immutable evidence kinds were recorded.
        """

        parameters: tuple[Any, ...]
        site_predicate = ""
        if site_id is None:
            parameters = ()
        else:
            site_predicate = "AND acl.site_id = ? "
            parameters = (site_id,)
        with self._lock:
            rows = self._connection.execute(
                "SELECT turn.task_id, turn.created_at, turn.updated_at, turn.result_json "
                "FROM assistant_turns AS turn "
                "JOIN resource_acl AS acl ON acl.resource_type = 'assistant_conversation' "
                "AND acl.resource_id = turn.conversation_id "
                "WHERE turn.result_json IS NOT NULL "
                f"{site_predicate}"
                "ORDER BY turn.updated_at DESC, turn.task_id DESC LIMIT 100",
                parameters,
            ).fetchall()
        for row in rows:
            result = _decode(row["result_json"])
            if not isinstance(result, Mapping):
                continue
            visual = result.get("visualResult")
            evidence = result.get("evidence")
            if not isinstance(visual, Mapping) and not isinstance(evidence, list):
                continue
            visual_mapping = visual if isinstance(visual, Mapping) else {}
            raw_evidence = (
                evidence
                if isinstance(evidence, list)
                else visual_mapping.get("evidence", [])
            )
            kinds = sorted(
                {
                    str(item.get("kind"))[:80]
                    for item in raw_evidence
                    if isinstance(item, Mapping) and isinstance(item.get("kind"), str)
                }
            )[:20]
            try:
                created = datetime.fromisoformat(str(row["created_at"]).replace("Z", "+00:00"))
                updated = datetime.fromisoformat(str(row["updated_at"]).replace("Z", "+00:00"))
                latency_ms = max(0.0, round((updated - created).total_seconds() * 1_000, 3))
            except (TypeError, ValueError, OverflowError):
                latency_ms = None
            return {
                "recordedAt": str(row["updated_at"]),
                "latencyMs": latency_ms,
                "status": str(visual_mapping.get("status") or result.get("routeStatus") or "recorded")[:80],
                "evidenceCount": len(raw_evidence) if isinstance(raw_evidence, list) else 0,
                "evidenceKinds": kinds,
            }
        return None

    def _append_assistant_event_locked(
        self,
        task_id: str,
        conversation_id: str,
        event_type: str,
        data: Mapping[str, Any],
        *,
        created_at: str | None = None,
    ) -> None:
        """Append one event while the caller owns ``self._lock`` and transaction."""

        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS last_sequence "
            "FROM assistant_turn_events WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        sequence = int(row["last_sequence"]) + 1
        event_id = f"assistant-event-{task_id}-{sequence}"
        self._connection.execute(
            "INSERT INTO assistant_turn_events("
            "task_id, sequence, event_id, conversation_id, event_type, data_json, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                sequence,
                event_id,
                conversation_id,
                event_type,
                _encode(data),
                created_at or self._now(),
            ),
        )

    def list_assistant_turn_events(
        self,
        task_id: str,
        *,
        after_sequence: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT sequence, event_id, conversation_id, event_type, data_json, created_at "
                "FROM assistant_turn_events WHERE task_id = ? AND sequence > ? "
                "ORDER BY sequence",
                (task_id, max(0, int(after_sequence))),
            ).fetchall()
        return [
            {
                "schemaVersion": "h-ide.assistant-event/2",
                "eventId": str(row["event_id"]),
                "taskId": task_id,
                "conversationId": str(row["conversation_id"]),
                "sequence": int(row["sequence"]),
                "type": str(row["event_type"]),
                "data": _decode(row["data_json"]),
                "createdAt": str(row["created_at"]),
            }
            for row in rows
        ]

    def get_assistant_turn(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM assistant_turns WHERE task_id = ?", (task_id,)
            ).fetchone()
        return None if row is None else self._assistant_turn_row(row)

    def _restore_revision_source(self, request_json: str, now: str) -> None:
        """Make the immutable source draft reviewable when no replacement was produced.

        The caller holds ``self._lock`` and an open transaction. Only the
        lifecycle status changes; the stored plan payload is never rewritten.
        """
        request = _decode(request_json)
        revision_of = request.get("revisionOfDraftId") if isinstance(request, Mapping) else None
        if isinstance(revision_of, str) and revision_of:
            self._connection.execute(
                "UPDATE assistant_drafts SET status = 'draft', updated_at = ? "
                "WHERE draft_id = ? AND status = 'revision_requested'",
                (now, revision_of),
            )

    def create_assistant_turn(
        self,
        task_request: Mapping[str, Any],
        *,
        message_id: str,
    ) -> dict[str, Any]:
        task_id = task_request.get("taskId")
        conversation_id = task_request.get("conversationId")
        text = task_request.get("text")
        modality = task_request.get("modality", "text")
        transcription_id = task_request.get("transcriptionId")
        revision_of = task_request.get("revisionOfDraftId")
        if not all(isinstance(value, str) and value for value in (task_id, conversation_id, text)):
            raise ValueError("taskId, conversationId, and text are required")
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                conversation = self._connection.execute(
                    "SELECT robot_id, archived_at, deleted_at "
                    "FROM assistant_conversations WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                if conversation is None:
                    raise ValueError("assistant conversation not found")
                if conversation["deleted_at"] is not None:
                    raise AssistantConversationConflict(
                        "deleted conversation must be restored before accepting messages"
                    )
                if conversation["archived_at"] is not None:
                    raise AssistantConversationConflict(
                        "archived conversation must be restored before accepting messages"
                    )
                active = self._connection.execute(
                    "SELECT task_id FROM assistant_turns WHERE conversation_id = ? "
                    "AND status IN ('queued', 'running') LIMIT 1",
                    (conversation_id,),
                ).fetchone()
                if active is not None:
                    raise AssistantTurnConflict("assistant conversation already has an active turn")
                sequence_row = self._connection.execute(
                    "SELECT COALESCE(MAX(turn_sequence), 0) + 1 AS next_sequence "
                    "FROM assistant_turns WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                turn_sequence = int(sequence_row["next_sequence"])
                if isinstance(revision_of, str) and revision_of:
                    draft = self._connection.execute(
                        "SELECT d.status, m.conversation_id FROM assistant_drafts d "
                        "JOIN assistant_draft_metadata m ON m.draft_id = d.draft_id "
                        "WHERE d.draft_id = ?",
                        (revision_of,),
                    ).fetchone()
                    if draft is None or draft["conversation_id"] != conversation_id:
                        raise ValueError("assistant draft does not belong to conversation")
                    if draft["status"] != "draft":
                        raise AssistantTurnConflict(
                            f"assistant draft is not revisable (status={draft['status']})"
                        )
                    self._connection.execute(
                        "UPDATE assistant_drafts SET status = 'revision_requested', updated_at = ? "
                        "WHERE draft_id = ? AND status = 'draft'",
                        (now, revision_of),
                    )
                self._connection.execute(
                    "INSERT INTO assistant_messages("
                    "message_id, conversation_id, task_id, role, content, modality, transcription_id, created_at"
                    ") VALUES (?, ?, ?, 'user', ?, ?, ?, ?)",
                    (message_id, conversation_id, task_id, text, modality, transcription_id, now),
                )
                self._connection.execute(
                    "INSERT INTO assistant_turns("
                    "task_id, conversation_id, request_json, status, result_json, error, "
                    "cancellation_requested, created_at, updated_at, turn_sequence"
                    ") VALUES (?, ?, ?, 'queued', NULL, NULL, 0, ?, ?, ?)",
                    (
                        task_id,
                        conversation_id,
                        _encode(task_request),
                        now,
                        now,
                        turn_sequence,
                    ),
                )
                self._connection.execute(
                    "UPDATE assistant_conversations SET status = 'planning', updated_at = ? "
                    "WHERE conversation_id = ?",
                    (now, conversation_id),
                )
                self._append_assistant_event_locked(
                    task_id,
                    conversation_id,
                    "turn.accepted",
                    {"status": "queued"},
                    created_at=now,
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                raise AssistantTurnConflict("assistant conversation already has an active turn") from exc
            except Exception:
                self._connection.rollback()
                raise
        return self.get_assistant_turn(task_id) or {}

    def mark_assistant_turn_running(self, task_id: str) -> bool:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._now()
                cursor = self._connection.execute(
                    "UPDATE assistant_turns SET status = 'running', updated_at = ? "
                    "WHERE task_id = ? AND status = 'queued' AND cancellation_requested = 0",
                    (now, task_id),
                )
                if cursor.rowcount == 1:
                    row = self._connection.execute(
                        "SELECT conversation_id FROM assistant_turns WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    assert row is not None
                    self._append_assistant_event_locked(
                        task_id,
                        str(row["conversation_id"]),
                        "turn.started",
                        {"status": "running"},
                        created_at=now,
                    )
                self._connection.commit()
                return cursor.rowcount == 1
            except Exception:
                self._connection.rollback()
                raise

    def is_assistant_turn_cancelled(self, task_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT status, cancellation_requested FROM assistant_turns WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        return row is None or bool(row["cancellation_requested"]) or row["status"] == "cancelled"

    def cancel_assistant_turn(self, task_id: str) -> dict[str, Any] | None:
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT conversation_id, status, request_json FROM assistant_turns WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    self._connection.rollback()
                    return None
                if row["status"] in {"queued", "running"}:
                    self._restore_revision_source(row["request_json"], now)
                    self._connection.execute(
                        "UPDATE assistant_turns SET status = 'cancelled', cancellation_requested = 1, "
                        "updated_at = ? WHERE task_id = ?",
                        (now, task_id),
                    )
                    self._connection.execute(
                        "UPDATE assistant_conversations SET status = 'idle', updated_at = ? "
                        "WHERE conversation_id = ?",
                        (now, row["conversation_id"]),
                    )
                    self._append_assistant_event_locked(
                        task_id,
                        str(row["conversation_id"]),
                        "turn.cancelled",
                        {"status": "cancelled"},
                        created_at=now,
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_assistant_turn(task_id)

    def complete_assistant_turn(
        self,
        task_id: str,
        *,
        status: str,
        result: Mapping[str, Any],
        assistant_message: str,
        message_id: str,
        conversation_status: str,
        draft: Mapping[str, Any] | None = None,
        review: Mapping[str, Any] | None = None,
    ) -> bool:
        """Atomically persist a terminal turn, reply, and optional immutable draft."""
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT conversation_id, request_json, status FROM assistant_turns WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None or row["status"] not in {"queued", "running"}:
                    self._connection.rollback()
                    return False
                conversation_id = row["conversation_id"]
                request = _decode(row["request_json"])
                revision_of = request.get("revisionOfDraftId") if isinstance(request, Mapping) else None
                if draft is not None:
                    draft_id = draft.get("draftId")
                    plan = draft.get("plan")
                    if not isinstance(draft_id, str) or not isinstance(plan, Mapping):
                        raise ValueError("draftId and plan are required")
                    self._connection.execute(
                        "INSERT INTO assistant_drafts("
                        "draft_id, payload_json, status, execution_json, error, created_at, updated_at"
                        ") VALUES (?, ?, 'draft', '{}', NULL, ?, ?)",
                        (draft_id, _encode(plan), now, now),
                    )
                    self._connection.execute(
                        "INSERT INTO assistant_draft_metadata("
                        "draft_id, conversation_id, task_id, review_json, revision_of_draft_id, created_at"
                        ") VALUES (?, ?, ?, ?, ?, ?)",
                        (draft_id, conversation_id, task_id, _encode(review or {}), revision_of, now),
                    )
                    if isinstance(revision_of, str) and revision_of:
                        updated = self._connection.execute(
                            "UPDATE assistant_drafts SET status = 'superseded', updated_at = ? "
                            "WHERE draft_id = ? AND status = 'revision_requested'",
                            (now, revision_of),
                        )
                        if updated.rowcount != 1:
                            raise AssistantTurnConflict("revision source is no longer supersedable")
                elif isinstance(revision_of, str) and revision_of:
                    self._restore_revision_source(row["request_json"], now)
                self._connection.execute(
                    "UPDATE assistant_turns SET status = ?, result_json = ?, error = NULL, updated_at = ? "
                    "WHERE task_id = ?",
                    (status, _encode(result), now, task_id),
                )
                self._connection.execute(
                    "INSERT INTO assistant_messages("
                    "message_id, conversation_id, task_id, role, content, modality, transcription_id, created_at"
                    ") VALUES (?, ?, ?, 'assistant', ?, 'text', NULL, ?)",
                    (message_id, conversation_id, task_id, assistant_message, now),
                )
                self._connection.execute(
                    "UPDATE assistant_conversations SET status = ?, updated_at = ? WHERE conversation_id = ?",
                    (conversation_status, now, conversation_id),
                )
                if result.get("schemaVersion") == "h-ide.assistant-result/2":
                    mission_draft_id = result.get("missionDraftId")
                    if isinstance(mission_draft_id, str) and mission_draft_id:
                        existing_link = self._connection.execute(
                            "SELECT conversation_id, task_id "
                            "FROM assistant_conversation_missions "
                            "WHERE mission_draft_id = ?",
                            (mission_draft_id,),
                        ).fetchone()
                        if existing_link is None:
                            self._connection.execute(
                                "INSERT INTO assistant_conversation_missions("
                                "conversation_id, task_id, mission_draft_id, mission_id, "
                                "created_at, updated_at) VALUES (?, ?, ?, NULL, ?, ?)",
                                (
                                    conversation_id,
                                    task_id,
                                    mission_draft_id,
                                    now,
                                    now,
                                ),
                            )
                        elif (
                            str(existing_link["conversation_id"]) != conversation_id
                            or str(existing_link["task_id"]) != task_id
                        ):
                            raise AssistantConversationConflict(
                                "Mission draft is already bound to another conversation turn"
                            )
                    route_event = {
                        key: result[key]
                        for key in (
                            "intent",
                            "route",
                            "routeStatus",
                            "missionDraftId",
                        )
                        if result.get(key) is not None
                    }
                    self._append_assistant_event_locked(
                        task_id,
                        str(conversation_id),
                        "route.selected",
                        route_event,
                        created_at=now,
                    )
                self._append_assistant_event_locked(
                    task_id,
                    str(conversation_id),
                    "turn.completed",
                    {"status": status},
                    created_at=now,
                )
                self._connection.commit()
                return True
            except Exception:
                self._connection.rollback()
                raise

    def fail_assistant_turn(self, task_id: str, error: str) -> bool:
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT conversation_id, status, request_json FROM assistant_turns WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None or row["status"] not in {"queued", "running"}:
                    self._connection.rollback()
                    return False
                self._restore_revision_source(row["request_json"], now)
                self._connection.execute(
                    "UPDATE assistant_turns SET status = 'failed', error = ?, updated_at = ? WHERE task_id = ?",
                    (error, now, task_id),
                )
                self._connection.execute(
                    "UPDATE assistant_conversations SET status = 'failed', updated_at = ? WHERE conversation_id = ?",
                    (now, row["conversation_id"]),
                )
                self._append_assistant_event_locked(
                    task_id,
                    str(row["conversation_id"]),
                    "turn.failed",
                    {"status": "failed", "error": error[:2000]},
                    created_at=now,
                )
                self._connection.commit()
                return True
            except Exception:
                self._connection.rollback()
                raise

    def recover_active_assistant_turns(self) -> None:
        """Never replay an LLM planning run after a process restart."""
        now = self._now()
        message = "planning turn interrupted by process restart; submit a new turn"
        with self._lock:
            rows = self._connection.execute(
                "SELECT task_id, conversation_id, request_json FROM assistant_turns "
                "WHERE status IN ('queued', 'running')"
            ).fetchall()
            for row in rows:
                self._restore_revision_source(row["request_json"], now)
                self._append_assistant_event_locked(
                    str(row["task_id"]),
                    str(row["conversation_id"]),
                    "turn.failed",
                    {"status": "unknown", "error": message},
                    created_at=now,
                )
            self._connection.execute(
                "UPDATE assistant_turns SET status = 'unknown', error = ?, updated_at = ? "
                "WHERE status IN ('queued', 'running')",
                (message, now),
            )
            self._connection.execute(
                "UPDATE assistant_conversations SET status = 'failed', updated_at = ? "
                "WHERE status = 'planning'",
                (now,),
            )
            self._connection.commit()

    def get_assistant_draft_review(self, draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT conversation_id, task_id, review_json, revision_of_draft_id, created_at "
                "FROM assistant_draft_metadata WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "conversationId": row["conversation_id"],
            "taskId": row["task_id"],
            "review": _decode(row["review_json"]),
            "revisionOfDraftId": row["revision_of_draft_id"],
            "createdAt": row["created_at"],
        }

    def list_assistant_conversation_drafts(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT m.draft_id FROM assistant_draft_metadata m "
                "JOIN assistant_drafts d ON d.draft_id = m.draft_id "
                "WHERE m.conversation_id = ? ORDER BY m.created_at, m.draft_id",
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            draft = self.get_assistant_draft(row["draft_id"])
            if draft is not None:
                metadata = self.get_assistant_draft_review(row["draft_id"])
                draft["taskId"] = (metadata or {}).get("taskId")
                draft["review"] = (metadata or {}).get("review", {})
                draft["revisionOfDraftId"] = (metadata or {}).get("revisionOfDraftId")
                result.append(draft)
        return result

    def get_assistant_draft_for_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT draft_id FROM assistant_draft_metadata WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        draft = self.get_assistant_draft(row["draft_id"])
        if draft is None:
            return None
        metadata = self.get_assistant_draft_review(row["draft_id"])
        draft["taskId"] = (metadata or {}).get("taskId")
        draft["review"] = (metadata or {}).get("review", {})
        draft["revisionOfDraftId"] = (metadata or {}).get("revisionOfDraftId")
        return draft

    def reject_assistant_draft(self, draft_id: str, reason: str, *, message_id: str) -> dict[str, Any] | None:
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT d.status, m.conversation_id FROM assistant_drafts d "
                    "JOIN assistant_draft_metadata m ON m.draft_id = d.draft_id "
                    "WHERE d.draft_id = ?",
                    (draft_id,),
                ).fetchone()
                if row is None:
                    self._connection.rollback()
                    return None
                if row["status"] != "draft":
                    raise AssistantTurnConflict(
                        f"assistant draft is not rejectable (status={row['status']})"
                    )
                self._connection.execute(
                    "UPDATE assistant_drafts SET status = 'rejected', error = ?, updated_at = ? "
                    "WHERE draft_id = ?",
                    (reason, now, draft_id),
                )
                self._connection.execute(
                    "UPDATE assistant_conversations SET status = 'rejected', updated_at = ? "
                    "WHERE conversation_id = ?",
                    (now, row["conversation_id"]),
                )
                self._connection.execute(
                    "INSERT INTO assistant_messages("
                    "message_id, conversation_id, task_id, role, content, modality, transcription_id, created_at"
                    ") VALUES (?, ?, NULL, 'system', ?, 'text', NULL, ?)",
                    (message_id, row["conversation_id"], f"计划已拒绝：{reason}", now),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_assistant_draft(draft_id)

    def set_assistant_conversation_status_for_draft(self, draft_id: str, status: str) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE assistant_conversations SET status = ?, updated_at = ? WHERE conversation_id = ("
                "SELECT conversation_id FROM assistant_draft_metadata WHERE draft_id = ?)",
                (status, self._now(), draft_id),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def save_transcription(
        self,
        transcription_id: str,
        text: str,
        *,
        language: str | None,
        locale: str,
        filename: str,
        owner_user_id: str | None = None,
        site_id: str | None = None,
        auth_session_id: str | None = None,
        source: str = "batch_wav",
    ) -> dict[str, Any]:
        created_at = self._now()
        with self._lock:
            self._connection.execute(
                "INSERT INTO speech_transcriptions("
                "transcription_id, text, language, locale, filename, created_at, "
                "owner_user_id, site_id, auth_session_id, speech_session_id, "
                "utterance_id, source"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)",
                (
                    transcription_id,
                    text,
                    language,
                    locale,
                    filename,
                    created_at,
                    owner_user_id,
                    site_id,
                    auth_session_id,
                    source,
                ),
            )
            self._connection.commit()
        result = self.get_transcription(transcription_id)
        assert result is not None
        return result

    def save_streaming_transcription(
        self,
        *,
        text: str,
        language: str | None,
        locale: str,
        owner_user_id: str,
        site_id: str,
        auth_session_id: str | None,
        speech_session_id: str,
        utterance_id: str,
    ) -> dict[str, Any]:
        """Persist a server-issued ASR final exactly once per utterance.

        The browser supplies none of these identity fields. A repeated provider
        final returns the original transcription ID, while a changed text or
        principal for the same server session/utterance fails closed.
        """

        values = {
            "text": str(text).strip(),
            "locale": str(locale).strip(),
            "owner_user_id": str(owner_user_id).strip(),
            "site_id": str(site_id).strip(),
            "speech_session_id": str(speech_session_id).strip(),
            "utterance_id": str(utterance_id).strip(),
        }
        if any(not value for value in values.values()):
            raise ValueError("streaming transcription identity is incomplete")
        transcription_id = f"transcription-{uuid.uuid4().hex}"
        created_at = self._now()
        with self._lock:
            try:
                self._connection.execute(
                    "INSERT INTO speech_transcriptions("
                    "transcription_id, text, language, locale, filename, created_at, "
                    "owner_user_id, site_id, auth_session_id, speech_session_id, "
                    "utterance_id, source"
                    ") VALUES (?, ?, ?, ?, 'speech-stream.v2', ?, ?, ?, ?, ?, ?, "
                    "'streaming_asr') ON CONFLICT(speech_session_id, utterance_id) "
                    "WHERE source = 'streaming_asr' AND speech_session_id IS NOT NULL "
                    "AND utterance_id IS NOT NULL DO NOTHING",
                    (
                        transcription_id,
                        values["text"],
                        language,
                        values["locale"],
                        created_at,
                        values["owner_user_id"],
                        values["site_id"],
                        auth_session_id,
                        values["speech_session_id"],
                        values["utterance_id"],
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM speech_transcriptions WHERE speech_session_id = ? "
                    "AND utterance_id = ? AND source = 'streaming_asr'",
                    (values["speech_session_id"], values["utterance_id"]),
                ).fetchone()
                if row is None:
                    raise RuntimeError("streaming transcription persistence failed")
                expected = (
                    values["text"],
                    language,
                    values["locale"],
                    values["owner_user_id"],
                    values["site_id"],
                    auth_session_id,
                )
                actual = (
                    str(row["text"]),
                    row["language"],
                    str(row["locale"]),
                    row["owner_user_id"],
                    row["site_id"],
                    row["auth_session_id"],
                )
                if actual != expected:
                    raise ValueError(
                        "streaming transcription identity conflicts with persisted final"
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        result = self.get_transcription(str(row["transcription_id"]))
        assert result is not None
        return result

    def get_transcription(self, transcription_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT transcription_id, text, language, locale, filename, created_at, "
                "owner_user_id, site_id, auth_session_id, speech_session_id, "
                "utterance_id, source "
                "FROM speech_transcriptions WHERE transcription_id = ?",
                (transcription_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "transcriptionId": row["transcription_id"],
            "text": row["text"],
            "language": row["language"],
            "locale": row["locale"],
            "filename": row["filename"],
            "createdAt": row["created_at"],
            "ownerUserId": row["owner_user_id"],
            "siteId": row["site_id"],
            "authSessionId": row["auth_session_id"],
            "speechSessionId": row["speech_session_id"],
            "utteranceId": row["utterance_id"],
            "source": row["source"],
        }

    def save_assistant_draft(self, draft_id: str, payload: Mapping[str, Any], *, status: str = "draft") -> bool:
        """Insert an immutable draft; never overwrite an existing plan."""
        if not isinstance(draft_id, str) or not draft_id:
            raise ValueError("draft_id is required")
        now = self._now()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO assistant_drafts(draft_id, payload_json, status, execution_json, error, created_at, updated_at)
                VALUES (?, ?, ?, '{}', NULL, ?, ?)
                ON CONFLICT(draft_id) DO NOTHING
                """,
                (draft_id, _encode(payload), status, now, now),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def create_visual_mission(
        self,
        mission_id: str,
        draft_id: str,
        robot_id: str,
        plan: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Idempotently create a durable parent mission after draft confirmation."""
        now = self._now()
        state = {"currentStageIndex": 0, "cancelRequested": False, "receipts": []}
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT draft_id, plan_json FROM visual_missions WHERE mission_id = ?",
                    (mission_id,),
                ).fetchone()
                if row is not None:
                    if row["draft_id"] != draft_id or _decode(row["plan_json"]) != _jsonable(plan):
                        raise ValueError("visual mission ID is already bound to another immutable plan")
                    self._connection.rollback()
                    result = self.get_visual_mission(mission_id)
                    assert result is not None
                    return result
                self._connection.execute(
                    "INSERT INTO visual_missions VALUES (?, ?, ?, 'approved', ?, ?, NULL, ?, ?)",
                    (mission_id, draft_id, robot_id, _encode(plan), _encode(state), now, now),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        result = self.get_visual_mission(mission_id)
        assert result is not None
        return result

    def get_visual_mission(self, mission_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM visual_missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
            stages = self._connection.execute(
                "SELECT * FROM visual_mission_stages WHERE mission_id = ? ORDER BY rowid",
                (mission_id,),
            ).fetchall()
        if row is None:
            return None
        return {
            "missionId": mission_id,
            "draftId": row["draft_id"],
            "robotId": row["robot_id"],
            "status": row["status"],
            "plan": _decode(row["plan_json"]),
            "state": _decode(row["state_json"]),
            "error": row["error"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "stages": [
                {
                    "stageId": item["stage_id"],
                    "attempt": item["attempt"],
                    "status": item["status"],
                    "visualResolutionId": item["visual_resolution_id"],
                    "commandId": item["command_id"],
                    "state": _decode(item["state_json"]),
                    "updatedAt": item["updated_at"],
                }
                for item in stages
            ],
        }

    def transition_visual_mission(
        self,
        mission_id: str,
        status: str,
        *,
        expected_status: str | tuple[str, ...] | None = None,
        state: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        expected = (expected_status,) if isinstance(expected_status, str) else expected_status
        with self._lock:
            row = self._connection.execute(
                "SELECT status, state_json FROM visual_missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
            if row is None or expected is not None and row["status"] not in expected:
                return False
            next_state = _decode(row["state_json"]) if state is None else dict(state)
            cursor = self._connection.execute(
                "UPDATE visual_missions SET status = ?, state_json = ?, error = ?, updated_at = ? "
                "WHERE mission_id = ? AND status = ?",
                (status, _encode(next_state), error, self._now(), mission_id, row["status"]),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def save_visual_stage(
        self,
        mission_id: str,
        stage_id: str,
        attempt: int,
        status: str,
        *,
        visual_resolution_id: str | None = None,
        command_id: str | None = None,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO visual_mission_stages VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(mission_id, stage_id, attempt) DO UPDATE SET "
                "status=excluded.status, "
                "visual_resolution_id=COALESCE(visual_mission_stages.visual_resolution_id, excluded.visual_resolution_id), "
                "command_id=COALESCE(visual_mission_stages.command_id, excluded.command_id), "
                "state_json=excluded.state_json, updated_at=excluded.updated_at",
                (
                    mission_id,
                    stage_id,
                    attempt,
                    status,
                    visual_resolution_id,
                    command_id,
                    _encode(state or {}),
                    self._now(),
                ),
            )
            self._connection.commit()

    def request_visual_mission_cancel(self, mission_id: str) -> bool:
        mission = self.get_visual_mission(mission_id)
        if mission is None or mission["status"] in {
            "completed",
            "rejected",
            "failed",
            "cancelled",
            "unknown",
        }:
            return False
        state = dict(mission["state"])
        state["cancelRequested"] = True
        return self.transition_visual_mission(
            mission_id,
            "cancelled",
            expected_status=mission["status"],
            state=state,
            error="cancelled by user",
        )

    def recover_visual_missions(self) -> None:
        """Never replay a motion whose terminal state was lost across restart."""
        with self._lock:
            self._connection.execute(
                "UPDATE visual_missions SET status='unknown', error=?, updated_at=? "
                "WHERE status IN ('observing','resolving','dispatching','navigating','verifying')",
                ("visual mission interrupted by process restart; reconcile before retry", self._now()),
            )
            self._connection.commit()

    def get_assistant_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json, status, execution_json, error, created_at, updated_at FROM assistant_drafts WHERE draft_id = ?",
                (draft_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "draftId": draft_id,
            "plan": _decode(row["payload_json"]),
            "status": row["status"],
            "execution": _decode(row["execution_json"] or "{}"),
            "error": row["error"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def set_assistant_draft_status(
        self,
        draft_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        execution: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            clauses = ["status = ?", "updated_at = ?"]
            values: list[Any] = [status, self._now()]
            if execution is not None:
                clauses.append("execution_json = ?")
                values.append(_encode(execution))
            if error is not None:
                clauses.append("error = ?")
                values.append(error)
            elif status in {"confirmed", "draft", "unknown"}:
                clauses.append("error = NULL")
            values.append(draft_id)
            where = "draft_id = ?"
            if expected_status is not None:
                where += " AND status = ?"
                values.append(expected_status)
            cursor = self._connection.execute(
                f"UPDATE assistant_drafts SET {', '.join(clauses)} WHERE {where}", values
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def claim_assistant_draft(self, draft_id: str) -> bool:
        """Atomically move a draft into confirmation in progress."""
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE assistant_drafts SET status = 'confirming', updated_at = ? WHERE draft_id = ? AND status = 'draft'",
                (self._now(), draft_id),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def reset_confirming_drafts(self) -> None:
        """Compatibility alias; interrupted confirmations are not replayable."""
        self.recover_confirming_drafts()

    def recover_confirming_drafts(self) -> None:
        """Mark interrupted confirmations unknown rather than resetting them."""
        with self._lock:
            self._connection.execute(
                "UPDATE assistant_drafts SET status = 'unknown', error = ?, updated_at = ? WHERE status = 'confirming'",
                ("confirmation interrupted by process restart; reconciliation required", self._now()),
            )
            self._connection.commit()

    def update_assistant_execution(
        self,
        draft_id: str,
        execution: Mapping[str, Any],
        *,
        expected_status: str = "confirming",
    ) -> bool:
        """CAS update of durable per-step receipts while a plan is confirming."""
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE assistant_drafts SET execution_json = ?, updated_at = ? "
                "WHERE draft_id = ? AND status = ?",
                (_encode(execution), self._now(), draft_id, expected_status),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    # ------------------------------------------------------------------
    # Mission V2 durable event store
    # ------------------------------------------------------------------

    @staticmethod
    def _mission_identity(payload: Mapping[str, Any], field: str = "missionId") -> str:
        mission_id = payload.get(field)
        if not isinstance(mission_id, str) or not mission_id:
            raise ValueError(f"{field} is required")
        return mission_id

    def create_mission(
        self,
        plan: Mapping[str, Any],
        authorization: Mapping[str, Any],
        revision: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        event: Mapping[str, Any],
        *,
        owner_user_id: str | None = None,
        site_id: str | None = None,
        autonomy_decision: Mapping[str, Any] | None = None,
        autonomy_grant: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Atomically create an approved immutable Mission and event 1.

        A retry with the same immutable plan returns the existing mission.  A
        reused mission ID with different content is rejected before any state
        is changed.
        """

        plan_payload = dict(_jsonable(plan))
        authorization_payload = dict(_jsonable(authorization))
        revision_payload = dict(_jsonable(revision))
        snapshot_payload = dict(_jsonable(snapshot))
        event_payload = dict(_jsonable(event))
        mission_id = self._mission_identity(revision_payload)
        if snapshot_payload.get("missionId") != mission_id or event_payload.get("missionId") != mission_id:
            raise ValueError("mission creation identity mismatch")
        if int(snapshot_payload.get("lastSequence", -1)) != 1 or int(event_payload.get("sequence", -1)) != 1:
            raise ValueError("mission creation must commit sequence 1")
        robot_id = snapshot_payload.get("robotId")
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("mission robotId is required")
        revision_id = revision_payload.get("revisionId")
        if snapshot_payload.get("activeRevisionId") != revision_id:
            raise ValueError("initial mission revision binding mismatch")
        if (owner_user_id is None) != (site_id is None):
            raise ValueError("mission owner and site must be provided together")
        if autonomy_grant is not None and autonomy_decision is None:
            raise ValueError("autonomy grant requires its policy decision")
        if autonomy_grant is not None and (
            autonomy_grant.get("authorizationId")
            != authorization_payload.get("authorizationId")
            or autonomy_grant.get("planId") != mission_id
            or int(autonomy_grant.get("planRevision", -1))
            != int(revision_payload.get("sequence", -2))
        ):
            raise ValueError("initial autonomy grant binding mismatch")
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                existing = self._connection.execute(
                    "SELECT plan_json FROM missions WHERE mission_id = ?", (mission_id,)
                ).fetchone()
                if existing is not None:
                    if _decode(existing["plan_json"]) != plan_payload:
                        raise ValueError("mission ID is already bound to another immutable plan")
                    self._connection.rollback()
                    result = self.get_mission(mission_id)
                    assert result is not None
                    return result
                self._connection.execute(
                    "INSERT INTO missions(mission_id, robot_id, status, plan_json, authorization_json, "
                    "snapshot_json, active_revision_id, last_sequence, error, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?)",
                    (
                        mission_id,
                        robot_id,
                        snapshot_payload["status"],
                        _encode(plan_payload),
                        _encode(authorization_payload),
                        _encode(snapshot_payload),
                        revision_id,
                        now,
                        now,
                    ),
                )
                self._connection.execute(
                    "INSERT INTO mission_revisions(revision_id, mission_id, sequence, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        revision_id,
                        mission_id,
                        int(revision_payload["sequence"]),
                        _encode(revision_payload),
                        revision_payload.get("createdAt", now),
                    ),
                )
                self._connection.execute(
                    "INSERT INTO mission_authorizations(authorization_id, mission_id, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        authorization_payload["authorizationId"],
                        mission_id,
                        _encode(authorization_payload),
                        authorization_payload.get("approvedAt", now),
                    ),
                )
                if owner_user_id is not None and site_id is not None:
                    self._bind_resource_owner_locked(
                        "mission",
                        mission_id,
                        owner_user_id=owner_user_id,
                        site_id=site_id,
                        now=now,
                    )
                    self._bind_resource_owner_locked(
                        "mission_revision",
                        str(revision_id),
                        owner_user_id=owner_user_id,
                        site_id=site_id,
                        now=now,
                    )
                if autonomy_decision is not None:
                    self._insert_policy_decision_locked(autonomy_decision)
                if autonomy_grant is not None:
                    self._insert_autonomy_grant_locked(autonomy_grant, now=now)
                self._connection.execute(
                    "INSERT INTO mission_events(mission_id, sequence, event_id, event_type, payload_json, created_at) "
                    "VALUES (?, 1, ?, ?, ?, ?)",
                    (
                        mission_id,
                        event_payload["eventId"],
                        event_payload["eventType"],
                        _encode(event_payload),
                        event_payload.get("occurredAt", now),
                    ),
                )
                for node in revision_payload.get("nodes", []):
                    self._connection.execute(
                        "INSERT INTO mission_node_states(mission_id, revision_id, node_id, status, attempt, state_json, updated_at) "
                        "VALUES (?, ?, ?, 'pending', 0, '{}', ?)",
                        (mission_id, revision_id, node["nodeId"], now),
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        result = self.get_mission(mission_id)
        assert result is not None
        return result

    def save_mission_draft(
        self,
        draft_id: str,
        robot_id: str,
        plan: Mapping[str, Any],
        *,
        status: str = "draft",
        owner_user_id: str | None = None,
        site_id: str | None = None,
    ) -> bool:
        if (owner_user_id is None) != (site_id is None):
            raise ValueError("mission draft owner and site must be provided together")
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                cursor = self._connection.execute(
                    "INSERT INTO mission_drafts(draft_id, robot_id, plan_json, status, error, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, NULL, ?, ?) ON CONFLICT(draft_id) DO NOTHING",
                    (draft_id, robot_id, _encode(plan), status, now, now),
                )
                if owner_user_id is not None and site_id is not None:
                    self._bind_resource_owner_locked(
                        "mission_draft",
                        draft_id,
                        owner_user_id=owner_user_id,
                        site_id=site_id,
                        now=now,
                    )
                self._connection.commit()
                return cursor.rowcount == 1
            except Exception:
                self._connection.rollback()
                raise

    def get_mission_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM mission_drafts WHERE draft_id = ?", (draft_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "draftId": row["draft_id"],
            "robotId": row["robot_id"],
            "plan": _decode(row["plan_json"]),
            "status": row["status"],
            "error": row["error"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def set_mission_draft_status(
        self,
        draft_id: str,
        status: str,
        *,
        expected_status: str | None = None,
        error: str | None = None,
    ) -> bool:
        with self._lock:
            where = "draft_id = ?"
            values: list[Any] = [status, error, self._now(), draft_id]
            if expected_status is not None:
                where += " AND status = ?"
                values.append(expected_status)
            cursor = self._connection.execute(
                f"UPDATE mission_drafts SET status=?, error=?, updated_at=? WHERE {where}",
                values,
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def recover_mission_draft_approvals(self) -> int:
        """Resolve only the local CAS claim after an interrupted approval.

        If the atomic Mission row exists, approval authority already exists
        and the draft is finalized. Otherwise it is safe to return the draft
        to review. No physical skill is replayed.
        """

        now = self._now()
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE mission_drafts SET status = CASE "
                "WHEN EXISTS(SELECT 1 FROM missions WHERE mission_id = draft_id) "
                "THEN 'approved' ELSE 'draft' END, updated_at = ? "
                "WHERE status = 'approving'",
                (now,),
            )
            self._connection.commit()
        return int(cursor.rowcount)

    def list_mission_drafts(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT draft_id FROM mission_drafts ORDER BY updated_at DESC, draft_id LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [draft for row in rows if (draft := self.get_mission_draft(row["draft_id"])) is not None]

    def get_mission(self, mission_id: str, *, after_sequence: int = 0) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
            if row is None:
                return None
            revisions = self._connection.execute(
                "SELECT payload_json FROM mission_revisions WHERE mission_id = ? ORDER BY sequence",
                (mission_id,),
            ).fetchall()
            authorizations = self._connection.execute(
                "SELECT payload_json FROM mission_authorizations WHERE mission_id = ? ORDER BY created_at, authorization_id",
                (mission_id,),
            ).fetchall()
            events = self._connection.execute(
                "SELECT payload_json FROM mission_events WHERE mission_id = ? AND sequence > ? ORDER BY sequence",
                (mission_id, max(0, int(after_sequence))),
            ).fetchall()
            world = self._connection.execute(
                "SELECT payload_json FROM mission_world_records WHERE mission_id = ? ORDER BY observed_at, record_id",
                (mission_id,),
            ).fetchall()
            nodes = self._connection.execute(
                "SELECT * FROM mission_node_states WHERE mission_id = ? ORDER BY revision_id, rowid",
                (mission_id,),
            ).fetchall()
            skills = self._connection.execute(
                "SELECT * FROM mission_skill_executions WHERE mission_id = ? ORDER BY updated_at, request_id",
                (mission_id,),
            ).fetchall()
        revisions_payload = [_decode(item["payload_json"]) for item in revisions]
        return {
            "missionId": mission_id,
            "robotId": row["robot_id"],
            "status": row["status"],
            "plan": _decode(row["plan_json"]),
            "authorization": _decode(row["authorization_json"]),
            "authorizations": [_decode(item["payload_json"]) for item in authorizations],
            "snapshot": _decode(row["snapshot_json"]),
            "activeRevisionId": row["active_revision_id"],
            "activeRevision": next(
                (
                    item
                    for item in reversed(revisions_payload)
                    if item.get("revisionId") == row["active_revision_id"]
                ),
                None,
            ),
            "revisions": revisions_payload,
            "events": [_decode(item["payload_json"]) for item in events],
            "worldRecords": [
                _legacy_world_record_read_projection(_decode(item["payload_json"]))
                for item in world
            ],
            "nodeStates": [
                {
                    "missionId": item["mission_id"],
                    "revisionId": item["revision_id"],
                    "nodeId": item["node_id"],
                    "status": item["status"],
                    "attempt": item["attempt"],
                    "state": _decode(item["state_json"]),
                    "updatedAt": item["updated_at"],
                }
                for item in nodes
            ],
            "skillExecutions": [
                {
                    "requestId": item["request_id"],
                    "missionId": item["mission_id"],
                    "revisionId": item["revision_id"],
                    "nodeId": item["node_id"],
                    "status": item["status"],
                    "request": _decode(item["request_json"]),
                    "progress": _decode(item["progress_json"]),
                    "result": (
                        None
                        if item["result_json"] is None
                        else _legacy_skill_result_read_projection(_decode(item["result_json"]))
                    ),
                    "updatedAt": item["updated_at"],
                }
                for item in skills
            ],
            "lastSequence": row["last_sequence"],
            "error": row["error"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def list_missions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT mission_id FROM missions ORDER BY updated_at DESC, mission_id LIMIT ?",
                (max(1, min(int(limit), 500)),),
            ).fetchall()
        return [mission for row in rows if (mission := self.get_mission(row["mission_id"])) is not None]

    def commit_mission_transition(
        self,
        mission_id: str,
        *,
        expected_last_sequence: int,
        snapshot: Mapping[str, Any],
        event: Mapping[str, Any],
        revision: Mapping[str, Any] | None = None,
        authorization: Mapping[str, Any] | None = None,
        autonomy_decision: Mapping[str, Any] | None = None,
        autonomy_grant: Mapping[str, Any] | None = None,
        world_record: Mapping[str, Any] | None = None,
        skill_request: Mapping[str, Any] | None = None,
        skill_progress: Mapping[str, Any] | None = None,
        skill_result: Mapping[str, Any] | None = None,
        node_states: Iterable[Mapping[str, Any]] = (),
        error: str | None = None,
    ) -> bool:
        """Commit one Mission transition and all of its evidence atomically."""

        snapshot_payload = dict(_jsonable(snapshot))
        event_payload = dict(_jsonable(event))
        next_sequence = expected_last_sequence + 1
        if snapshot_payload.get("missionId") != mission_id or event_payload.get("missionId") != mission_id:
            raise ValueError("mission transition identity mismatch")
        if snapshot_payload.get("lastSequence") != next_sequence or event_payload.get("sequence") != next_sequence:
            raise ValueError("mission event and snapshot sequence mismatch")
        if autonomy_grant is not None and (
            authorization is None or autonomy_decision is None
        ):
            raise ValueError(
                "autonomy grant requires an authorization and policy decision"
            )
        if autonomy_grant is not None:
            grant_payload = dict(_jsonable(autonomy_grant))
            authorization_payload = dict(_jsonable(authorization or {}))
            revision_payload = dict(_jsonable(revision or {}))
            if (
                grant_payload.get("authorizationId")
                != authorization_payload.get("authorizationId")
                or grant_payload.get("planId") != mission_id
                or int(grant_payload.get("planRevision", -1))
                != int(revision_payload.get("sequence", -2))
            ):
                raise ValueError("autonomy grant transition binding mismatch")
        now = self._now()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT last_sequence, active_revision_id FROM missions WHERE mission_id = ?",
                    (mission_id,),
                ).fetchone()
                if row is None:
                    self._connection.rollback()
                    return False
                if int(row["last_sequence"]) != expected_last_sequence:
                    self._connection.rollback()
                    return False
                if revision is not None:
                    revision_payload = dict(_jsonable(revision))
                    if revision_payload.get("missionId") != mission_id:
                        raise ValueError("mission revision identity mismatch")
                    self._connection.execute(
                        "INSERT INTO mission_revisions(revision_id, mission_id, sequence, payload_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            revision_payload["revisionId"],
                            mission_id,
                            int(revision_payload["sequence"]),
                            _encode(revision_payload),
                            revision_payload.get("createdAt", now),
                        ),
                    )
                    mission_binding = self._connection.execute(
                        "SELECT owner_user_id, site_id FROM resource_acl "
                        "WHERE resource_type = 'mission' AND resource_id = ?",
                        (mission_id,),
                    ).fetchone()
                    if mission_binding is not None:
                        self._bind_resource_owner_locked(
                            "mission_revision",
                            str(revision_payload["revisionId"]),
                            owner_user_id=str(mission_binding["owner_user_id"]),
                            site_id=str(mission_binding["site_id"]),
                            now=now,
                        )
                if authorization is not None:
                    authorization_payload = dict(_jsonable(authorization))
                    if authorization_payload.get("missionId") != mission_id:
                        raise ValueError("mission authorization identity mismatch")
                    self._connection.execute(
                        "INSERT INTO mission_authorizations(authorization_id, mission_id, payload_json, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            authorization_payload["authorizationId"],
                            mission_id,
                            _encode(authorization_payload),
                            authorization_payload.get("approvedAt", now),
                        ),
                    )
                if autonomy_decision is not None:
                    self._insert_policy_decision_locked(autonomy_decision)
                if autonomy_grant is not None:
                    self._connection.execute(
                        "UPDATE autonomy_grants SET revoked_at = ?, updated_at = ? "
                        "WHERE plan_id = ? AND revoked_at IS NULL",
                        (now, now, mission_id),
                    )
                    self._insert_autonomy_grant_locked(autonomy_grant, now=now)
                if world_record is not None:
                    record_payload = dict(_jsonable(world_record))
                    if record_payload.get("missionId") != mission_id:
                        raise ValueError("world record mission identity mismatch")
                    existing = self._connection.execute(
                        "SELECT payload_json FROM mission_world_records WHERE record_id = ?",
                        (record_payload["recordId"],),
                    ).fetchone()
                    if existing is not None and _decode(existing["payload_json"]) != record_payload:
                        raise ValueError("world record ID is bound to different evidence")
                    if existing is None:
                        self._connection.execute(
                            "INSERT INTO mission_world_records(record_id, mission_id, robot_id, kind, subject, predicate, payload_json, observed_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                record_payload["recordId"],
                                mission_id,
                                record_payload["robotId"],
                                record_payload["kind"],
                                record_payload["subject"],
                                record_payload["predicate"],
                                _encode(record_payload),
                                record_payload["observedAt"],
                            ),
                        )
                if skill_request is not None:
                    request_payload = dict(_jsonable(skill_request))
                    if request_payload.get("missionId") != mission_id:
                        raise ValueError("skill request mission identity mismatch")
                    self._connection.execute(
                        "INSERT INTO mission_skill_executions(request_id, mission_id, revision_id, node_id, status, request_json, progress_json, result_json, updated_at) "
                        "VALUES (?, ?, ?, ?, 'requested', ?, '{}', NULL, ?)",
                        (
                            request_payload["requestId"],
                            mission_id,
                            request_payload["revisionId"],
                            request_payload["nodeId"],
                            _encode(request_payload),
                            now,
                        ),
                    )
                if skill_progress is not None:
                    progress_payload = dict(_jsonable(skill_progress))
                    cursor = self._connection.execute(
                        "UPDATE mission_skill_executions SET status='running', progress_json=?, updated_at=? "
                        "WHERE request_id=? AND mission_id=? AND result_json IS NULL",
                        (_encode(progress_payload), now, progress_payload["requestId"], mission_id),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("skill progress is not bound to an active request")
                if skill_result is not None:
                    result_payload = dict(_jsonable(skill_result))
                    cursor = self._connection.execute(
                        "UPDATE mission_skill_executions SET status=?, result_json=?, updated_at=? "
                        "WHERE request_id=? AND mission_id=? AND result_json IS NULL",
                        (
                            result_payload["status"],
                            _encode(result_payload),
                            now,
                            result_payload["requestId"],
                            mission_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("skill result is not bound to an active request")
                active_revision_id = snapshot_payload.get("activeRevisionId")
                for item in node_states:
                    state = dict(_jsonable(item))
                    revision_id = state.get("revisionId") or active_revision_id
                    self._connection.execute(
                        "INSERT INTO mission_node_states(mission_id, revision_id, node_id, status, attempt, state_json, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(mission_id, revision_id, node_id) DO UPDATE SET "
                        "status=excluded.status, attempt=excluded.attempt, state_json=excluded.state_json, updated_at=excluded.updated_at",
                        (
                            mission_id,
                            revision_id,
                            state["nodeId"],
                            state["status"],
                            int(state.get("attempt", 0)),
                            _encode(state.get("state", {})),
                            now,
                        ),
                    )
                self._connection.execute(
                    "INSERT INTO mission_events(mission_id, sequence, event_id, event_type, payload_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        mission_id,
                        next_sequence,
                        event_payload["eventId"],
                        event_payload["eventType"],
                        _encode(event_payload),
                        event_payload.get("occurredAt", now),
                    ),
                )
                current_authorization = (
                    _encode(dict(_jsonable(authorization)))
                    if authorization is not None
                    else None
                )
                cursor = self._connection.execute(
                    "UPDATE missions SET status=?, snapshot_json=?, active_revision_id=?, last_sequence=?, "
                    "authorization_json=COALESCE(?, authorization_json), error=?, updated_at=? "
                    "WHERE mission_id=? AND last_sequence=?",
                    (
                        snapshot_payload["status"],
                        _encode(snapshot_payload),
                        active_revision_id,
                        next_sequence,
                        current_authorization,
                        error,
                        now,
                        mission_id,
                        expected_last_sequence,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("mission transition lost compare-and-set ownership")
                self._connection.commit()
                return True
            except Exception:
                self._connection.rollback()
                raise

    def recover_active_missions(self) -> int:
        """Recover interrupted Missions without replaying a durable request.

        Vision is a synchronous, workstation-owned observation boundary, so an
        interrupted request can be atomically closed as failed.  Every other
        in-flight request is treated as potentially external/physical: keep its
        original request ID active and force the Mission through reconciliation
        instead of issuing a replacement command.
        """

        active = {
            "planning", "observing", "dispatching", "executing", "verifying"
        }
        recovered = 0
        with self._lock:
            rows = self._connection.execute(
                "SELECT mission_id, snapshot_json, last_sequence FROM missions"
            ).fetchall()
        for row in rows:
            snapshot = _decode(row["snapshot_json"])
            if snapshot.get("status") not in active:
                continue
            sequence = int(row["last_sequence"]) + 1
            now = self._now()
            request_id = snapshot.get("activeSkillRequestId")
            execution = None
            if isinstance(request_id, str) and request_id:
                with self._lock:
                    execution = self._connection.execute(
                        "SELECT revision_id, node_id, request_json, result_json "
                        "FROM mission_skill_executions "
                        "WHERE mission_id = ? AND request_id = ?",
                        (row["mission_id"], request_id),
                    ).fetchone()
            request_payload = (
                _decode(execution["request_json"])
                if execution is not None and execution["result_json"] is None
                else None
            )
            skill_id = (
                request_payload.get("skillId")
                if isinstance(request_payload, Mapping)
                else None
            )
            workstation_vision = (
                isinstance(skill_id, str) and skill_id.startswith("vision.")
            )
            external_reconciliation = (
                execution is not None
                and execution["result_json"] is None
                and isinstance(request_id, str)
                and bool(request_id)
                and not workstation_vision
            )
            skill_result = None
            node_states: list[dict[str, Any]] = []
            if workstation_vision:
                detail = (
                    "Workstation observation was interrupted by process restart; "
                    "a reviewed retry is required"
                )
                error_code = "WORKSTATION_PROCESS_RESTART"
                snapshot.update(
                    {
                        "status": "planning",
                        "lastSequence": sequence,
                        "activeSkillRequestId": None,
                        "updatedAt": now,
                        "errorCode": error_code,
                        "detail": detail,
                    }
                )
                skill_result = {
                    "requestId": request_id,
                    "status": "failed",
                    "completedAt": now,
                    "result": {
                        "dispatchAuthority": "workstation",
                        "recoveredAfterRestart": True,
                    },
                    "evidenceIds": [],
                    "errorCode": error_code,
                    "detail": detail,
                }
                with self._lock:
                    state = self._connection.execute(
                        "SELECT attempt, state_json FROM mission_node_states "
                        "WHERE mission_id = ? AND revision_id = ? AND node_id = ?",
                        (
                            row["mission_id"],
                            execution["revision_id"],
                            execution["node_id"],
                        ),
                    ).fetchone()
                previous_state = (
                    _decode(state["state_json"]) if state is not None else {}
                )
                node_states.append(
                    {
                        "revisionId": execution["revision_id"],
                        "nodeId": execution["node_id"],
                        "status": "failed",
                        "attempt": int(state["attempt"]) if state is not None else 1,
                        "state": {
                            **previous_state,
                            "skillRequestId": request_id,
                            "resultStatus": "failed",
                            "recoveredAfterRestart": True,
                        },
                    }
                )
                event_type = "skill.failed_after_restart"
                correlation_id = request_id
                payload = {
                    "reason": "process_restart",
                    "requestId": request_id,
                    "errorCode": error_code,
                    "dispatchAuthority": "workstation",
                }
            elif external_reconciliation:
                detail = (
                    "External skill was interrupted by process restart; reconcile "
                    "the original request before any further command"
                )
                error_code = "ROBOT_AGENT_RECONCILIATION_REQUIRED"
                snapshot.update(
                    {
                        "status": "dispatching",
                        "lastSequence": sequence,
                        "activeSkillRequestId": request_id,
                        "updatedAt": now,
                        "errorCode": error_code,
                        "detail": detail,
                    }
                )
                event_type = "mission.reconciliation_required_after_restart"
                correlation_id = request_id
                payload = {
                    "reason": "process_restart",
                    "requestId": request_id,
                    "errorCode": error_code,
                    "recoveryRequired": True,
                }
            else:
                detail = (
                    "Mission interrupted by process restart; reconcile before resume"
                )
                error_code = "PROCESS_RESTART"
                snapshot.update(
                    {
                        "status": "suspended",
                        "lastSequence": sequence,
                        "activeSkillRequestId": None,
                        "updatedAt": now,
                        "errorCode": error_code,
                        "detail": detail,
                    }
                )
                event_type = "mission.suspended_after_restart"
                correlation_id = f"recovery-{row['mission_id']}"
                payload = {"reason": "process_restart"}
            event = {
                "schemaVersion": "h-ide.mission-event/2",
                "eventId": f"recovery-{row['mission_id']}-{sequence}",
                "missionId": row["mission_id"],
                "sequence": sequence,
                "eventType": event_type,
                "occurredAt": now,
                "causationId": None,
                "correlationId": correlation_id,
                "revisionId": snapshot.get("activeRevisionId"),
                "nodeId": snapshot.get("activeNodeId"),
                "payload": payload,
            }
            if self.commit_mission_transition(
                row["mission_id"],
                expected_last_sequence=int(row["last_sequence"]),
                snapshot=snapshot,
                event=event,
                skill_result=skill_result,
                node_states=node_states,
                error=detail,
            ):
                recovered += 1
        return recovered

    def close(self) -> None:
        with self._lock:
            self._connection.close()


__all__ = [
    "AssistantTurnConflict",
    "CommandClaim",
    "MappingPointerConflict",
    "MappingStateConflict",
    "SQLiteStore",
]
