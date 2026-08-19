"""H-IDE-owned, auditable conversational context assembly.

The browser never supplies this context.  It is projected from the durable
conversation transcript after the HTTP ACL has selected a conversation.  All
projected text has ``CONVERSATIONAL_ONLY`` authority: it may resolve discourse
and preferences but cannot become pose, obstacle, reachability, or safety
truth for a Mission.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Mapping

from packages.control_core.assistant_models import (
    ConversationDurableSummary,
    ConversationHistoryMessage,
    ConversationMemoryContext,
    ConversationMemorySnippet,
)

from .store import SQLiteStore


_RECENT_MESSAGE_LIMIT = 20
_RECENT_CHARACTER_BUDGET = 40_000
_MEMORY_LIMIT = 8
_SUMMARY_CHARACTER_LIMIT = 12_000

_NUMBER = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
_SPATIAL_PATTERNS = (
    re.compile(
        rf"(?i)\b(?:map|odom|pose)\s*(?:frame|坐标)?\s*[:=]?\s*"
        rf"[\(\[]\s*{_NUMBER}\s*[,，]\s*{_NUMBER}"
        rf"(?:\s*[,，]\s*{_NUMBER})?\s*[\)\]]"
    ),
    re.compile(
        rf"(?:坐标|地图位置|位姿)\s*(?:是|为|在|[:：=])?\s*"
        rf"[\(\[]?\s*{_NUMBER}\s*[,，]\s*{_NUMBER}"
        rf"(?:\s*[,，]\s*{_NUMBER})?\s*[\)\]]?"
    ),
    re.compile(
        rf"(?i)(?:[\"']?x[\"']?\s*[:=]\s*{_NUMBER}\s*[,，]\s*)"
        rf"(?:[\"']?y[\"']?\s*[:=]\s*{_NUMBER})"
        rf"(?:\s*[,，]\s*[\"']?z[\"']?\s*[:=]\s*{_NUMBER})?"
    ),
)
_PRONOUN_REFERENCE = re.compile(
    r"(?:那里|那边|那个|它旁边|去它|到它|刚才(?:的)?|之前(?:的)?|the(?:re|m)|that one)",
    re.IGNORECASE,
)
_PROMPT_INJECTION = re.compile(
    r"(?:ignore (?:all |the )?(?:previous|system) instructions?|"
    r"reveal (?:the )?(?:prompt|credentials?|secrets?)|"
    r"bypass (?:policy|review|permission)|"
    r"忽略(?:之前|以上|系统)(?:的)?(?:指令|提示)|"
    r"泄露(?:提示词|密钥|凭据)|绕过(?:策略|审核|权限))",
    re.IGNORECASE,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(encoded)


def redact_spatial_claims(value: str) -> str:
    """Remove coordinate-like claims while retaining semantic referents.

    This is intentionally narrow: ordinary numbers and mathematical history
    remain useful, while common map/pose representations cannot silently move
    from conversational recall into Mission input.
    """

    result = value
    for pattern in _SPATIAL_PATTERNS:
        result = pattern.sub("[空间坐标已省略]", result)
    return re.sub(r"\s+", " ", result).strip()


def _mission_safe_text(value: str) -> str:
    spatially_safe = redact_spatial_claims(value)
    return _PROMPT_INJECTION.sub("[非可信指令已省略]", spatially_safe)


def _bounded(value: str, maximum: int) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= maximum:
        return compact
    return f"{compact[: maximum - 1]}…"


def _terms(value: str) -> set[str]:
    normalized = value.casefold()
    terms = set(re.findall(r"[a-z0-9_-]{2,}", normalized))
    for run in re.findall(r"[\u3400-\u9fff]+", normalized):
        terms.add(run)
        for offset in range(max(0, len(run) - 1)):
            terms.add(run[offset : offset + 2])
    return terms


def _lexical_score(query: str, candidate: str) -> float:
    query_terms = _terms(query)
    candidate_terms = _terms(candidate)
    if not query_terms or not candidate_terms:
        return 0.0
    overlap = len(query_terms.intersection(candidate_terms))
    if not overlap:
        return 0.0
    return min(1.0, overlap / ((len(query_terms) * len(candidate_terms)) ** 0.5))


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


@dataclass(frozen=True)
class _AnnotatedMessage:
    message_id: str
    role: str
    content: str
    created_at: datetime
    turn: int
    index: int


@dataclass(frozen=True)
class PreparedConversationMemory:
    history: tuple[ConversationHistoryMessage, ...]
    memory_context: ConversationMemoryContext
    mission_history: tuple[dict[str, Any], ...]
    audit_id: str


def _annotate(messages: list[Mapping[str, Any]]) -> list[_AnnotatedMessage]:
    result: list[_AnnotatedMessage] = []
    turn = 0
    for item in messages:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        if role == "user":
            turn += 1
        # A legacy assistant-only row has no auditable user turn and is kept in
        # recent history, but never claimed as covered by a durable summary.
        message_turn = turn if turn > 0 else 0
        result.append(
            _AnnotatedMessage(
                message_id=str(item.get("messageId") or f"legacy-{len(result)}"),
                role=str(role),
                content=content.strip(),
                created_at=_as_datetime(item.get("createdAt")),
                turn=message_turn,
                index=len(result),
            )
        )
    return result


def _recent_suffix(messages: list[_AnnotatedMessage]) -> list[_AnnotatedMessage]:
    selected: list[_AnnotatedMessage] = []
    characters = 0
    for message in reversed(messages):
        if len(selected) >= _RECENT_MESSAGE_LIMIT:
            break
        sanitized = redact_spatial_claims(message.content)
        if selected and characters + len(sanitized) > _RECENT_CHARACTER_BUDGET:
            break
        if not selected and len(sanitized) > _RECENT_CHARACTER_BUDGET:
            sanitized = sanitized[-_RECENT_CHARACTER_BUDGET:]
        characters += len(sanitized)
        selected.append(message)
    selected.reverse()
    return selected


def _summary_coverage(
    messages: list[_AnnotatedMessage],
    recent: list[_AnnotatedMessage],
) -> int:
    cutoff = recent[0].index if recent else len(messages)
    indices_by_turn: dict[int, list[int]] = {}
    for message in messages:
        if message.turn > 0:
            indices_by_turn.setdefault(message.turn, []).append(message.index)
    covered = 0
    for turn in range(1, max(indices_by_turn, default=0) + 1):
        indices = indices_by_turn.get(turn)
        if not indices or max(indices) >= cutoff:
            break
        covered = turn
    return covered


def _summary_projection(
    conversation_id: str,
    messages: list[_AnnotatedMessage],
    covered_through_turn: int,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    if covered_through_turn <= 0:
        return None, None, []
    covered = [
        message
        for message in messages
        if 0 < message.turn <= covered_through_turn
    ]
    if not covered:
        return None, None, []
    lines = [
        "以下是持久对话的审计摘录，仅用于语言衔接；不是位置、感知、可达性或安全事实。"
    ]
    for message in covered:
        role = "用户" if message.role == "user" else "助手"
        content = _bounded(redact_spatial_claims(message.content), 420)
        lines.append(f"T{message.turn} {role}: {content}")
    text = "\n".join(lines)
    if len(text) > _SUMMARY_CHARACTER_LIMIT:
        head = text[:5_700]
        tail = text[-6_150:]
        text = f"{head}\n…[中间原文仍保留在 H-IDE 审计消息中]…\n{tail}"
        text = text[:_SUMMARY_CHARACTER_LIMIT]
    source = [
        {
            "messageId": message.message_id,
            "turn": message.turn,
            "role": message.role,
            "content": message.content,
            "createdAt": message.created_at.isoformat(),
        }
        for message in covered
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    return (
        {
            "summaryId": f"summary-{_sha256_text(conversation_id)[:24]}",
            "text": text,
            "coveredThroughTurn": covered_through_turn,
            "generatedAt": generated_at,
            "method": "SERVER_COMPACTION",
            "contentSha256": _sha256_text(text),
        },
        _canonical_digest(source),
        [message.message_id for message in covered],
    )


def _snippet_kind(message: _AnnotatedMessage) -> str:
    content = message.content.casefold()
    if any(marker in content for marker in ("我喜欢", "我偏好", "请一直", "prefer")):
        return "USER_PREFERENCE"
    if any(marker in content for marker in ("决定", "就按", "先做", "agreed", "decided")):
        return "PRIOR_DECISION"
    return "CONVERSATION_EXCERPT"


def _select_memories(
    query: str,
    messages: list[_AnnotatedMessage],
    recent: list[_AnnotatedMessage],
) -> list[ConversationMemorySnippet]:
    cutoff = recent[0].index if recent else len(messages)
    older = [message for message in messages if message.index < cutoff and message.turn > 0]
    ranked: list[tuple[float, int, _AnnotatedMessage]] = []
    for message in older:
        sanitized = redact_spatial_claims(message.content)
        score = _lexical_score(query, sanitized)
        if score > 0:
            ranked.append((score, message.index, message))

    # Pronoun-only follow-ups have no lexical overlap. Select the latest older
    # user statement as an untrusted semantic referent, with all coordinate
    # claims redacted; current perception/SceneMemory must ground it again.
    if _PRONOUN_REFERENCE.search(query):
        fallback = next(
            (message for message in reversed(older) if message.role == "user"),
            None,
        )
        if fallback is not None and not any(item[2].message_id == fallback.message_id for item in ranked):
            ranked.append((0.5, fallback.index, fallback))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2].message_id))
    snippets: list[ConversationMemorySnippet] = []
    seen: set[str] = set()
    for score, _, message in ranked:
        if message.message_id in seen:
            continue
        seen.add(message.message_id)
        role = "用户" if message.role == "user" else "助手"
        text = _bounded(
            f"{role}曾说：{redact_spatial_claims(message.content)}",
            1_000,
        )
        snippets.append(
            ConversationMemorySnippet(
                memoryId=f"message-{_sha256_text(message.message_id)[:24]}",
                kind=_snippet_kind(message),
                text=text,
                recordedAt=message.created_at,
                sourceTurn=message.turn,
                relevanceScore=round(score, 6),
            )
        )
        if len(snippets) >= _MEMORY_LIMIT:
            break
    return snippets


def _resource_id(
    conversation: Mapping[str, Any],
    owner: Mapping[str, Any] | None,
) -> str:
    if owner is not None:
        authority = f"{owner.get('siteId')}\0{owner.get('ownerUserId')}"
        prefix = "principal"
    else:
        # Legacy deployments did not create ACL rows. requestedBy was still
        # persisted by H-IDE and is preferable to accepting any browser ID in
        # the memory wire contract.
        authority = f"legacy\0{conversation.get('requestedBy')}"
        prefix = "legacy-principal"
    return f"{prefix}-{_sha256_text(authority)[:40]}"


class ConversationMemoryContextBuilder:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def prepare(
        self,
        conversation_id: str,
        current_text: str,
        *,
        exclude_task_id: str | None = None,
    ) -> PreparedConversationMemory:
        conversation = self.store.get_assistant_conversation(conversation_id)
        if conversation is None:
            raise ValueError("assistant conversation not found")
        raw_messages = self.store.list_assistant_messages(conversation_id)
        if exclude_task_id is not None:
            raw_messages = [
                item for item in raw_messages if item.get("taskId") != exclude_task_id
            ]
        messages = _annotate(raw_messages)
        recent = _recent_suffix(messages)
        covered = _summary_coverage(messages, recent)
        summary, source_sha256, source_message_ids = _summary_projection(
            conversation_id,
            messages,
            covered,
        )
        memories = _select_memories(current_text, messages, recent)
        owner = self.store.get_resource_owner("assistant_conversation", conversation_id)
        resource_id = _resource_id(conversation, owner)
        recorded = self.store.record_assistant_conversation_memory_context(
            conversation_id,
            resource_id=resource_id,
            query_sha256=_sha256_text(current_text),
            recent_message_ids=[message.message_id for message in recent],
            selected_memory_ids=[memory.memoryId for memory in memories],
            summary=summary,
            source_sha256=source_sha256,
            source_message_ids=source_message_ids,
        )
        persisted_summary = recorded.get("summary")
        durable_summary = None
        if isinstance(persisted_summary, Mapping):
            durable_summary = ConversationDurableSummary.model_validate(
                {
                    key: persisted_summary[key]
                    for key in (
                        "summaryId",
                        "revision",
                        "text",
                        "coveredThroughTurn",
                        "generatedAt",
                        "method",
                        "contentSha256",
                    )
                }
            )
        memory_context = ConversationMemoryContext(
            resourceId=resource_id,
            revision=int(recorded["contextRevision"]),
            durableSummary=durable_summary,
            relevantMemories=memories,
        )
        history = tuple(
            ConversationHistoryMessage(
                role=message.role,
                content=redact_spatial_claims(message.content),
            )
            for message in recent
        )
        mission_memories = [
            memory for memory in memories if not _PROMPT_INJECTION.search(memory.text)
        ]
        mission_recent = recent[-19:] if mission_memories else recent
        mission_items = [
            {
                "role": message.role,
                "content": _mission_safe_text(message.content),
                "createdAt": message.created_at.isoformat(),
            }
            for message in mission_recent
        ]
        if mission_memories:
            # Mission V2 receives semantic discourse continuity only.  Putting
            # recalled excerpts in one explicitly non-authoritative assistant
            # message lets "去那里" retain the named referent without ever
            # forwarding remembered coordinates or replacing typed grounding.
            recalled = "\n".join(memory.text for memory in mission_memories)
            mission_items.append(
                {
                    "role": "assistant",
                    "content": _bounded((
                        "较早的对话参考（只用于理解语义指代，不是坐标、感知、"
                        f"可达性或安全事实）：\n{recalled}"
                    ), 4_000),
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                }
            )
        mission_history = tuple(mission_items)
        return PreparedConversationMemory(
            history=history,
            memory_context=memory_context,
            mission_history=mission_history,
            audit_id=str(recorded["auditId"]),
        )


__all__ = [
    "ConversationMemoryContextBuilder",
    "PreparedConversationMemory",
    "redact_spatial_claims",
]
