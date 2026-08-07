"""In-process resource-group arbitration for logical commands."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID


class ResourceClaimError(ValueError):
    """Raised when a resource claim is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ClaimResult:
    acquired: bool
    conflicts: dict[str, UUID]


class ResourceArbiter:
    """Atomically grants commands exclusive ownership of named resources."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._owners: dict[str, UUID] = {}
        self._claims: dict[UUID, frozenset[str]] = {}

    def claim(self, command_id: UUID, resources: Iterable[str]) -> ClaimResult:
        requested = frozenset(resources)
        if not requested:
            raise ResourceClaimError("a command must claim at least one resource")
        if any(not resource for resource in requested):
            raise ResourceClaimError("resource names must not be empty")
        with self._lock:
            prior = self._claims.get(command_id)
            if prior is not None:
                if prior != requested:
                    raise ResourceClaimError(
                        f"command_id {command_id} already claimed a different resource set"
                    )
                return ClaimResult(acquired=True, conflicts={})

            conflicts = {
                resource: owner
                for resource in sorted(requested)
                if (owner := self._owners.get(resource)) is not None and owner != command_id
            }
            if conflicts:
                return ClaimResult(acquired=False, conflicts=conflicts)
            self._claims[command_id] = requested
            for resource in requested:
                self._owners[resource] = command_id
            return ClaimResult(acquired=True, conflicts={})

    def release(self, command_id: UUID) -> frozenset[str]:
        with self._lock:
            resources = self._claims.pop(command_id, frozenset())
            for resource in resources:
                if self._owners.get(resource) == command_id:
                    del self._owners[resource]
            return resources

    def preempt_all(self) -> tuple[UUID, ...]:
        """Release all locks and return command IDs that must be cancelled."""
        with self._lock:
            command_ids = tuple(self._claims)
            self._claims.clear()
            self._owners.clear()
            return command_ids

    @property
    def active_commands(self) -> frozenset[UUID]:
        with self._lock:
            return frozenset(self._claims)

    @property
    def owners(self) -> dict[str, UUID]:
        with self._lock:
            return dict(self._owners)
