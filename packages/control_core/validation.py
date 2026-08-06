"""Reusable validation helpers for control-plane domain contracts.

The helpers in this module deliberately do not know about persistence or
transport implementations.  They are small pure functions so contracts can
be used by the API, runtime, and tests without dependency injection concerns.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Mapping


CLOCK_SKEW = timedelta(seconds=5)


class FrozenDict(dict):
    """A JSON-serializable dict that rejects all in-place mutations."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("mapping version values are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


class FrozenList(tuple):
    """Tuple-backed JSON array exposing mutation names with clear errors."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("mapping version values are immutable")

    append = extend = insert = pop = remove = reverse = sort = _immutable


def _json_value(value: Any, *, freeze: bool = False) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        converted = {key: _json_value(item, freeze=freeze) for key, item in value.items()}
        return FrozenDict(converted) if freeze else converted
    if isinstance(value, (list, tuple)):
        converted = tuple(_json_value(item, freeze=freeze) for item in value)
        return FrozenList(converted) if freeze else list(converted)
    raise ValueError("value is not JSON-compatible")


def non_empty_id(value: Any, field_name: str = "id") -> str:
    """Return a trimmed identifier, rejecting blank and non-string values."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """Normalize a datetime to UTC.

    Naive values are interpreted as UTC.  This keeps the contract friendly to
    callers that use ``datetime.utcnow()`` while ensuring comparisons are
    deterministic once validated.
    """

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def validate_expiry(issued_at: datetime, expires_at: datetime) -> tuple[datetime, datetime]:
    """Normalize and validate request lifetime and reject already-expired data."""

    issued = as_utc(issued_at)
    expires = as_utc(expires_at)
    if issued > utc_now() + CLOCK_SKEW:
        raise ValueError("issuedAt cannot be in the future")
    if expires <= issued:
        raise ValueError("expiresAt must be later than issuedAt")
    if expires <= utc_now():
        raise ValueError("expiresAt is in the past")
    return issued, expires


def validate_command_request(request: Any) -> Any:
    """Re-check a request's lifetime at dispatch time.

    Construction validates the initial expiry, while this helper is useful at
    the runtime boundary where a previously accepted request may have expired
    while waiting in a queue.  It intentionally returns the original object.
    """

    expires = as_utc(request.expiresAt)
    if expires <= utc_now():
        raise ValueError("expiresAt is in the past")
    return request


def require_json_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    """Make a defensive JSON-compatible schema copy.

    Schema semantics are intentionally left to a schema engine; contracts only
    guarantee that a JSON object can be stored and transported.
    """

    if not isinstance(value, Mapping):
        raise ValueError("payload schema must be a JSON object")
    return _json_value(value)


def freeze_json_object(value: Mapping[str, Any]) -> FrozenDict:
    """Validate and recursively freeze a JSON object for immutable models."""

    if not isinstance(value, Mapping):
        raise ValueError("value must be a JSON object")
    return _json_value(value, freeze=True)


def ensure_acyclic_steps(steps: list[Any]) -> None:
    """Ensure step IDs are unique and dependencies form a finite DAG."""

    ids = [step.stepId for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("plan step IDs must be unique")
    known = set(ids)
    graph: dict[str, set[str]] = {}
    for step in steps:
        deps = set(step.dependsOn)
        if step.stepId in deps:
            raise ValueError("plan dependency cycle detected")
        unknown = deps - known
        if unknown:
            raise ValueError(f"unknown plan dependency: {sorted(unknown)[0]}")
        graph[step.stepId] = deps

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("plan dependency cycle detected")
        if node in visited:
            return
        visiting.add(node)
        for parent in graph[node]:
            visit(parent)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
