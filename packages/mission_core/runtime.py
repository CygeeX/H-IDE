"""Durable event-driven Mission V2 authority for the H-IDE control plane.

The runtime owns approval, monotonic events, revisions, node state and skill
issuance.  It deliberately does not plan paths, infer coordinates, or talk to
ROS directly.  Skill-specific bridges resolve semantic intent and hand the
result to ActionRuntime/Robot Agent, where Nav2 remains the only base-motion
executor.
"""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Protocol

from packages.control_core.capabilities import RiskLevel
from packages.control_core.validation import as_utc, non_empty_id, utc_now
from packages.scene_core import NAVIGATION_COMMAND_KEY

from .models import (
    ApprovalDisposition,
    MissionAuthorization,
    MissionEvent,
    MissionIntervention,
    MissionInterventionKind,
    MissionNode,
    MissionPlanDraft,
    MissionSnapshot,
    MissionStatus,
    PlanRevision,
    SkillProgress,
    SkillRequest,
    SkillResult,
    SkillResultStatus,
    WorldRecord,
)


class MissionStore(Protocol):
    def create_mission(
        self,
        plan: Mapping[str, Any],
        authorization: Mapping[str, Any],
        revision: Mapping[str, Any],
        snapshot: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def get_mission(self, mission_id: str, *, after_sequence: int = 0) -> dict[str, Any] | None: ...

    def list_missions(self, *, limit: int = 100) -> list[dict[str, Any]]: ...

    def commit_mission_transition(self, mission_id: str, **values: Any) -> bool: ...

    def recover_active_missions(self) -> int: ...


class MissionNotFound(ValueError):
    pass


class MissionConflict(ValueError):
    pass


_TERMINAL = {
    MissionStatus.COMPLETED,
    MissionStatus.FAILED,
    MissionStatus.CANCELLED,
}

_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


def _plain(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=False)
    return dict(value)


def _digest(value: Any) -> str:
    payload = value.model_dump(mode="json", by_alias=False) if hasattr(value, "model_dump") else value
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class MissionRuntime:
    """Pure control-plane state machine backed by an atomic MissionStore."""

    def __init__(self, store: MissionStore, *, recover: bool = True) -> None:
        self.store = store
        if recover:
            self.store.recover_active_missions()

    def approve(
        self,
        draft: MissionPlanDraft | Mapping[str, Any],
        *,
        approved_by: str,
        mission_id: str | None = None,
    ) -> dict[str, Any]:
        plan = draft if isinstance(draft, MissionPlanDraft) else MissionPlanDraft.model_validate(draft)
        owner = non_empty_id(approved_by, "approvedBy")
        identifier = non_empty_id(mission_id or plan.planId, "missionId")
        now = utc_now()
        expiry = min(as_utc(plan.expiresAt), as_utc(plan.envelope.expiresAt))
        if expiry <= now:
            raise MissionConflict("mission draft or envelope has expired")
        authorization = MissionAuthorization(
            authorizationId=f"auth-{identifier}-1",
            missionId=identifier,
            planId=plan.planId,
            envelopeSha256=_digest(plan.envelope),
            approvedBy=owner,
            approvedAt=now,
            expiresAt=expiry,
        )
        revision = PlanRevision(
            revisionId=f"revision-{identifier}-1",
            missionId=identifier,
            sequence=1,
            parentRevisionId=None,
            objective=plan.objective,
            nodes=plan.nodes,
            reasonCode="initial_human_approval",
            reason="Human approved the initial Mission envelope and semantic DAG.",
            changeSummary=["Initial Mission DAG approved"],
            disposition=ApprovalDisposition.HUMAN_APPROVED,
            authorizationId=authorization.authorizationId,
            createdBy=owner,
            createdAt=now,
        )
        snapshot = MissionSnapshot(
            missionId=identifier,
            robotId=plan.robotId,
            status=MissionStatus.APPROVED,
            lastSequence=1,
            activeRevisionId=revision.revisionId,
            updatedAt=now,
        )
        event = MissionEvent(
            eventId=f"event-{identifier}-1",
            missionId=identifier,
            sequence=1,
            eventType="mission.approved",
            occurredAt=now,
            correlationId=plan.planId,
            revisionId=revision.revisionId,
            payload={
                "authorizationId": authorization.authorizationId,
                "approvedBy": owner,
                "envelopeSha256": authorization.envelopeSha256,
            },
        )
        return self.store.create_mission(
            _plain(plan),
            _plain(authorization),
            _plain(revision),
            _plain(snapshot),
            _plain(event),
        )

    def get(self, mission_id: str, *, after_sequence: int = 0) -> dict[str, Any]:
        record = self.store.get_mission(non_empty_id(mission_id, "missionId"), after_sequence=after_sequence)
        if record is None:
            raise MissionNotFound("mission not found")
        return record

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_missions(limit=limit)

    @staticmethod
    def _snapshot(record: Mapping[str, Any]) -> MissionSnapshot:
        return MissionSnapshot.model_validate(record["snapshot"])

    @staticmethod
    def _revision(record: Mapping[str, Any]) -> PlanRevision:
        value = record.get("activeRevision")
        if not isinstance(value, Mapping):
            raise MissionConflict("mission has no active revision")
        return PlanRevision.model_validate(value)

    @staticmethod
    def _node_states(record: Mapping[str, Any], revision_id: str) -> dict[str, dict[str, Any]]:
        return {
            str(item["nodeId"]): dict(item)
            for item in record.get("nodeStates", [])
            if item.get("revisionId") == revision_id
        }

    @staticmethod
    def _budget_xy(value: Any, field: str) -> tuple[float, float]:
        if not isinstance(value, Mapping):
            raise MissionConflict(f"{field} must be an object")
        coordinates: list[float] = []
        for key in ("x", "y"):
            raw = value.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise MissionConflict(f"{field}.{key} must be a finite number")
            coordinate = float(raw)
            if not math.isfinite(coordinate):
                raise MissionConflict(f"{field}.{key} must be a finite number")
            coordinates.append(coordinate)
        return coordinates[0], coordinates[1]

    @classmethod
    def _reserved_navigation_distance(
        cls,
        record: Mapping[str, Any],
        next_input: Mapping[str, Any],
    ) -> float:
        origins = [
            item
            for item in record.get("worldRecords", [])
            if isinstance(item, Mapping)
            and item.get("kind") == "fact"
            and item.get("predicate") == "mission_origin"
        ]
        if not origins:
            raise MissionConflict("navigation budget requires a reviewed mission origin")
        points = [cls._budget_xy(origins[-1].get("value"), "missionOrigin")]
        for execution in record.get("skillExecutions", []):
            if not isinstance(execution, Mapping):
                continue
            request = execution.get("request")
            if not isinstance(request, Mapping) or request.get("skillId") != NAVIGATION_COMMAND_KEY:
                continue
            skill_input = request.get("input")
            goal = skill_input.get("goalPose") if isinstance(skill_input, Mapping) else None
            points.append(cls._budget_xy(goal, "reservedGoal"))
        points.append(cls._budget_xy(next_input.get("goalPose"), "goalPose"))
        return sum(math.dist(left, right) for left, right in zip(points, points[1:]))

    @staticmethod
    def _reserved_scan_yaw(
        record: Mapping[str, Any],
        next_input: Mapping[str, Any],
    ) -> float:
        values: list[Any] = []
        for execution in record.get("skillExecutions", []):
            if not isinstance(execution, Mapping):
                continue
            request = execution.get("request")
            if not isinstance(request, Mapping) or request.get("skillId") != "navigation.rotate":
                continue
            skill_input = request.get("input")
            values.append(
                skill_input.get("targetYawRad")
                if isinstance(skill_input, Mapping)
                else None
            )
        values.append(next_input.get("targetYawRad"))
        total = 0.0
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MissionConflict("rotation budget requires a finite targetYawRad")
            yaw = float(value)
            if not math.isfinite(yaw) or yaw == 0:
                raise MissionConflict("rotation budget requires a non-zero finite targetYawRad")
            total += abs(yaw)
        return total

    @staticmethod
    def _event(
        snapshot: MissionSnapshot,
        *,
        event_type: str,
        correlation_id: str,
        payload: Mapping[str, Any] | None = None,
        causation_id: str | None = None,
        node_id: str | None = None,
        revision_id: str | None = None,
    ) -> MissionEvent:
        sequence = snapshot.lastSequence + 1
        return MissionEvent(
            eventId=f"event-{snapshot.missionId}-{sequence}",
            missionId=snapshot.missionId,
            sequence=sequence,
            eventType=event_type,
            occurredAt=utc_now(),
            causationId=causation_id,
            correlationId=correlation_id,
            revisionId=revision_id or snapshot.activeRevisionId,
            nodeId=node_id,
            payload=dict(payload or {}),
        )

    def _commit(
        self,
        record: Mapping[str, Any],
        snapshot: MissionSnapshot,
        event: MissionEvent,
        **extras: Any,
    ) -> dict[str, Any]:
        previous = self._snapshot(record)
        if not self.store.commit_mission_transition(
            snapshot.missionId,
            expected_last_sequence=previous.lastSequence,
            snapshot=_plain(snapshot),
            event=_plain(event),
            **extras,
        ):
            raise MissionConflict("mission state changed concurrently")
        return self.get(snapshot.missionId)

    @staticmethod
    def _ready_nodes(
        revision: PlanRevision,
        states: Mapping[str, Mapping[str, Any]],
    ) -> list[MissionNode]:
        completed = {
            node_id
            for node_id, state in states.items()
            if state.get("status") in {"succeeded", "skipped"}
        }
        return [
            node
            for node in revision.nodes
            if states.get(node.nodeId, {}).get("status", "pending") in {"pending", "ready"}
            and set(node.dependsOn).issubset(completed)
        ]

    def start(self, mission_id: str) -> dict[str, Any]:
        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        if snapshot.status in _TERMINAL:
            raise MissionConflict(f"mission is terminal ({snapshot.status.value})")
        if snapshot.status not in {MissionStatus.APPROVED, MissionStatus.SUSPENDED, MissionStatus.READY}:
            raise MissionConflict(f"mission cannot start from {snapshot.status.value}")
        authorization = MissionAuthorization.model_validate(record["authorization"])
        if authorization.expiresAt <= utc_now():
            raise MissionConflict("mission authorization has expired")
        revision = self._revision(record)
        states = self._node_states(record, revision.revisionId)
        ready = self._ready_nodes(revision, states)
        if not ready:
            raise MissionConflict("mission has no dependency-ready node")
        node_updates = [
            {
                "revisionId": revision.revisionId,
                "nodeId": node.nodeId,
                "status": "ready",
                "attempt": int(states.get(node.nodeId, {}).get("attempt", 0)),
                "state": states.get(node.nodeId, {}).get("state", {}),
            }
            for node in ready
        ]
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.READY,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": ready[0].nodeId,
                "activeSkillRequestId": None,
                "updatedAt": utc_now(),
                "errorCode": None,
                "detail": None,
            }
        )
        event = self._event(
            snapshot,
            event_type="mission.started",
            correlation_id=mission_id,
            payload={"readyNodeIds": [node.nodeId for node in ready]},
            node_id=ready[0].nodeId,
        )
        return self._commit(record, next_snapshot, event, node_states=node_updates)

    def request_input(
        self,
        mission_id: str,
        *,
        node_id: str,
        error_code: str,
        detail: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Durably stop automatic execution when trusted input is missing.

        This transition never fabricates a failed SkillRequest.  It records
        that the currently ready/planning node needs new evidence, an operator
        answer, or a reviewed revision before execution can continue.
        """

        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        if snapshot.status not in {MissionStatus.READY, MissionStatus.PLANNING}:
            raise MissionConflict(
                f"mission cannot request input from {snapshot.status.value}"
            )
        revision = self._revision(record)
        if not any(item.nodeId == node_id for item in revision.nodes):
            raise MissionConflict("input request node is not in the active revision")
        now = utc_now()
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.NEEDS_INPUT,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": node_id,
                "activeSkillRequestId": None,
                "updatedAt": now,
                "errorCode": non_empty_id(error_code, "errorCode"),
                "detail": non_empty_id(detail, "detail"),
            }
        )
        event = self._event(
            snapshot,
            event_type="mission.needs_input",
            correlation_id=f"input-{mission_id}-{snapshot.lastSequence + 1}",
            payload={
                "errorCode": next_snapshot.errorCode,
                "detail": next_snapshot.detail,
                **dict(payload or {}),
            },
            node_id=node_id,
        )
        return self._commit(record, next_snapshot, event)

    def issue_skill(
        self,
        mission_id: str,
        node_id: str,
        input: Mapping[str, Any],
        *,
        expires_in_seconds: float = 60,
    ) -> SkillRequest:
        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        if snapshot.status not in {MissionStatus.READY, MissionStatus.PLANNING}:
            raise MissionConflict(f"mission cannot issue a skill from {snapshot.status.value}")
        revision = self._revision(record)
        node = next((item for item in revision.nodes if item.nodeId == node_id), None)
        if node is None:
            raise MissionConflict("mission node not found in active revision")
        states = self._node_states(record, revision.revisionId)
        state = states.get(node_id, {"status": "pending", "attempt": 0, "state": {}})
        if state.get("status") not in {"pending", "ready"}:
            raise MissionConflict(f"mission node is not issuable ({state.get('status')})")
        completed = {
            identifier
            for identifier, item in states.items()
            if item.get("status") in {"succeeded", "skipped"}
        }
        if not set(node.dependsOn).issubset(completed):
            raise MissionConflict("mission node dependencies are not complete")
        plan = MissionPlanDraft.model_validate(record["plan"])
        authorization = MissionAuthorization.model_validate(record["authorization"])
        if authorization.expiresAt <= utc_now():
            raise MissionConflict("mission authorization has expired")
        attempt = int(state.get("attempt", 0)) + 1
        if attempt > plan.envelope.maximumStageAttempts:
            raise MissionConflict("mission node attempt budget is exhausted")
        now = utc_now()
        try:
            lifetime_seconds = float(expires_in_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MissionConflict("skill request lifetime must be finite and positive") from exc
        if not math.isfinite(lifetime_seconds) or lifetime_seconds <= 0:
            raise MissionConflict("skill request lifetime must be finite and positive")
        expires = min(
            authorization.expiresAt,
            now + timedelta(seconds=min(lifetime_seconds, 600.0)),
        )
        request_id = f"skill-{mission_id}-{revision.sequence}-{node_id}-{attempt}"
        bound_input = dict(input)

        def bind_control(field: str, value: Any) -> None:
            if field in bound_input and bound_input[field] != value:
                raise MissionConflict(f"skill input attempts to override {field}")
            bound_input[field] = value

        if node.skillId == NAVIGATION_COMMAND_KEY:
            bind_control("missionId", mission_id)
            bind_control("stageId", node.nodeId)
            bind_control("robotBootId", plan.envelope.robotBootId)
            bind_control(
                "localizationEpochId",
                plan.envelope.localizationEpochId,
            )
            bind_control(
                "maximumSpeedMps",
                plan.envelope.maximumSpeedMps,
            )
        elif node.skillId == "navigation.rotate":
            bind_control("missionId", mission_id)
            bind_control("revisionId", revision.revisionId)
            bind_control("nodeId", node.nodeId)
            bind_control("authorizationId", authorization.authorizationId)
            bind_control("skillRequestId", request_id)
            bind_control("idempotencyKey", request_id)
            bind_control("robotBootId", plan.envelope.robotBootId)
            bind_control(
                "localizationEpochId",
                plan.envelope.localizationEpochId,
            )
        cumulative_distance = snapshot.cumulativeDistanceM
        cumulative_scan_yaw = snapshot.cumulativeScanYawRad
        if node.skillId == NAVIGATION_COMMAND_KEY:
            cumulative_distance = self._reserved_navigation_distance(record, bound_input)
            if cumulative_distance > plan.envelope.maximumCumulativeDistanceM + 1e-9:
                raise MissionConflict("navigation command exceeds the cumulative distance envelope")
        elif node.skillId == "navigation.rotate":
            cumulative_scan_yaw = self._reserved_scan_yaw(record, bound_input)
            if cumulative_scan_yaw > plan.envelope.maximumScanYawRad + 1e-9:
                raise MissionConflict("rotation command exceeds the cumulative scan-yaw envelope")
        request = SkillRequest(
            requestId=request_id,
            missionId=mission_id,
            revisionId=revision.revisionId,
            nodeId=node.nodeId,
            robotId=plan.robotId,
            skillId=node.skillId,
            skillVersion=node.skillVersion,
            input=bound_input,
            capabilityManifestDigest=plan.envelope.capabilityManifestDigest,
            authorizationId=authorization.authorizationId,
            idempotencyKey=request_id,
            issuedAt=now,
            expiresAt=expires,
        )
        status = (
            MissionStatus.OBSERVING
            if node.skillId.startswith("vision.")
            else MissionStatus.DISPATCHING
        )
        next_snapshot = snapshot.model_copy(
            update={
                "status": status,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": node.nodeId,
                "activeSkillRequestId": request.requestId,
                # These are conservative command reservations, not odometry.
                # They therefore survive failures and reviewed replans instead
                # of allowing a new revision to reset the authorization budget.
                "cumulativeDistanceM": cumulative_distance,
                "cumulativeScanYawRad": cumulative_scan_yaw,
                "updatedAt": now,
            }
        )
        event = self._event(
            snapshot,
            event_type="skill.requested",
            correlation_id=request.requestId,
            payload={
                "skillId": request.skillId,
                "skillVersion": request.skillVersion,
                "attempt": attempt,
                "idempotencyKey": request.idempotencyKey,
            },
            node_id=node.nodeId,
        )
        self._commit(
            record,
            next_snapshot,
            event,
            skill_request=_plain(request),
            node_states=[{
                "revisionId": revision.revisionId,
                "nodeId": node.nodeId,
                "status": "running",
                "attempt": attempt,
                "state": {"skillRequestId": request.requestId},
            }],
        )
        return request

    def record_progress(
        self,
        mission_id: str,
        progress: SkillProgress | Mapping[str, Any],
    ) -> dict[str, Any]:
        value = progress if isinstance(progress, SkillProgress) else SkillProgress.model_validate(progress)
        record = self.get(mission_id)
        execution = next(
            (item for item in record["skillExecutions"] if item["requestId"] == value.requestId),
            None,
        )
        if execution is None or execution.get("result") is not None:
            raise MissionConflict("skill progress is not bound to an active request")
        previous_progress = execution.get("progress") or {}
        if isinstance(previous_progress.get("sequence"), int) and value.sequence <= previous_progress["sequence"]:
            raise MissionConflict("skill progress sequence is stale")
        snapshot = self._snapshot(record)
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.EXECUTING,
                "lastSequence": snapshot.lastSequence + 1,
                "updatedAt": value.observedAt,
            }
        )
        event = self._event(
            snapshot,
            event_type="skill.progress",
            correlation_id=value.requestId,
            payload=_plain(value),
            node_id=execution["nodeId"],
        )
        return self._commit(record, next_snapshot, event, skill_progress=_plain(value))

    def record_result(
        self,
        mission_id: str,
        result: SkillResult | Mapping[str, Any],
    ) -> dict[str, Any]:
        value = result if isinstance(result, SkillResult) else SkillResult.model_validate(result)
        record = self.get(mission_id)
        execution = next(
            (item for item in record["skillExecutions"] if item["requestId"] == value.requestId),
            None,
        )
        if execution is None or execution.get("result") is not None:
            raise MissionConflict("skill result is not bound to an active request")
        snapshot = self._snapshot(record)
        revision = self._revision(record)
        states = self._node_states(record, revision.revisionId)
        succeeded = value.status is SkillResultStatus.SUCCEEDED
        node_status = "succeeded" if succeeded else value.status.value
        current_state = states.get(execution["nodeId"], {})
        node_updates: list[dict[str, Any]] = [{
            "revisionId": revision.revisionId,
            "nodeId": execution["nodeId"],
            "status": node_status,
            "attempt": int(current_state.get("attempt", 1)),
            "state": {
                **dict(current_state.get("state", {})),
                "skillRequestId": value.requestId,
                "resultStatus": value.status.value,
                "evidenceIds": list(value.evidenceIds),
            },
        }]
        next_active: str | None = None
        if succeeded:
            projected = {key: dict(item) for key, item in states.items()}
            projected[execution["nodeId"]] = node_updates[0]
            ready = self._ready_nodes(revision, projected)
            for node in ready:
                if node.nodeId == execution["nodeId"]:
                    continue
                existing = projected.get(node.nodeId, {})
                node_updates.append({
                    "revisionId": revision.revisionId,
                    "nodeId": node.nodeId,
                    "status": "ready",
                    "attempt": int(existing.get("attempt", 0)),
                    "state": existing.get("state", {}),
                })
            next_active = next((node.nodeId for node in ready if node.nodeId != execution["nodeId"]), None)
            all_completed = all(
                node.nodeId == execution["nodeId"]
                or projected.get(node.nodeId, {}).get("status")
                in {"succeeded", "skipped"}
                for node in revision.nodes
            )
            next_status = MissionStatus.COMPLETED if all_completed else MissionStatus.READY
        else:
            # A visual observation can legitimately need another view or an
            # operator-provided reference.  Keep that distinct from an
            # execution failure so the coordinator/UI can request input
            # instead of blindly entering the replanning loop.
            next_status = (
                MissionStatus.NEEDS_INPUT
                if value.errorCode == "VISUAL_NEEDS_INPUT"
                else MissionStatus.PLANNING
            )
        next_snapshot = snapshot.model_copy(
            update={
                "status": next_status,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": next_active if succeeded else execution["nodeId"],
                "activeSkillRequestId": None,
                "updatedAt": value.completedAt,
                "errorCode": None if succeeded else (value.errorCode or value.status.value),
                "detail": None if succeeded else (value.detail or "Skill failed; replanning required"),
            }
        )
        event = self._event(
            snapshot,
            event_type="skill.succeeded" if succeeded else "skill.failed",
            correlation_id=value.requestId,
            payload=_plain(value),
            node_id=execution["nodeId"],
        )
        return self._commit(
            record,
            next_snapshot,
            event,
            skill_result=_plain(value),
            node_states=node_updates,
            error=None if succeeded else next_snapshot.detail,
        )

    def skip_nodes(
        self,
        mission_id: str,
        node_ids: Iterable[str],
        *,
        reason: str,
        causation_id: str,
    ) -> dict[str, Any]:
        """Durably skip a reviewed conditional branch without claiming execution.

        A skip is terminal for dependency ordering, but remains distinct from
        success in the audit log and node state.  It is only allowed while no
        physical skill is active.
        """

        identifiers = tuple(dict.fromkeys(non_empty_id(item, "nodeId") for item in node_ids))
        if not identifiers:
            return self.get(mission_id)
        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        if snapshot.status in _TERMINAL:
            raise MissionConflict(f"mission is terminal ({snapshot.status.value})")
        if snapshot.activeSkillRequestId is not None:
            raise MissionConflict("cannot skip nodes while a skill request is active")
        revision = self._revision(record)
        nodes = {node.nodeId: node for node in revision.nodes}
        missing = [node_id for node_id in identifiers if node_id not in nodes]
        if missing:
            raise MissionConflict(f"skip node is not in the active revision: {missing[0]}")
        states = self._node_states(record, revision.revisionId)
        projected = {key: dict(value) for key, value in states.items()}
        now = utc_now()
        node_updates: list[dict[str, Any]] = []
        for node_id in identifiers:
            current = states.get(
                node_id,
                {"status": "pending", "attempt": 0, "state": {}},
            )
            if current.get("status") not in {"pending", "ready"}:
                raise MissionConflict(
                    f"mission node cannot be skipped ({current.get('status')})"
                )
            update = {
                "revisionId": revision.revisionId,
                "nodeId": node_id,
                "status": "skipped",
                "attempt": int(current.get("attempt", 0)),
                "state": {
                    **dict(current.get("state", {})),
                    "skipReason": non_empty_id(reason, "reason"),
                    "skipCausationId": non_empty_id(causation_id, "causationId"),
                    "skippedAt": now.isoformat(),
                },
            }
            projected[node_id] = update
            node_updates.append(update)

        ready = self._ready_nodes(revision, projected)
        for node in ready:
            existing = projected.get(node.nodeId, {})
            update = {
                "revisionId": revision.revisionId,
                "nodeId": node.nodeId,
                "status": "ready",
                "attempt": int(existing.get("attempt", 0)),
                "state": existing.get("state", {}),
            }
            projected[node.nodeId] = update
            node_updates.append(update)
        all_completed = all(
            projected.get(node.nodeId, {}).get("status") in {"succeeded", "skipped"}
            for node in revision.nodes
        )
        next_status = MissionStatus.COMPLETED if all_completed else MissionStatus.READY
        next_active = None if all_completed else next(
            (node.nodeId for node in ready),
            None,
        )
        next_snapshot = snapshot.model_copy(
            update={
                "status": next_status,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": next_active,
                "activeSkillRequestId": None,
                "updatedAt": now,
                "errorCode": None,
                "detail": None,
            }
        )
        event = self._event(
            snapshot,
            event_type="mission.nodes_skipped",
            correlation_id=causation_id,
            causation_id=causation_id,
            node_id=identifiers[0],
            payload={
                "nodeIds": list(identifiers),
                "reason": reason,
                "readyNodeIds": [node.nodeId for node in ready],
            },
        )
        return self._commit(
            record,
            next_snapshot,
            event,
            node_states=node_updates,
        )

    def record_world(self, record: WorldRecord | Mapping[str, Any]) -> dict[str, Any]:
        value = record if isinstance(record, WorldRecord) else WorldRecord.model_validate(record)
        mission = self.get(value.missionId)
        snapshot = self._snapshot(mission)
        if value.robotId != snapshot.robotId:
            raise MissionConflict("world record robot identity mismatch")
        next_snapshot = snapshot.model_copy(
            update={
                "lastSequence": snapshot.lastSequence + 1,
                "updatedAt": max(snapshot.updatedAt, value.observedAt),
            }
        )
        event = self._event(
            snapshot,
            event_type="world.recorded",
            correlation_id=value.recordId,
            payload={
                "recordId": value.recordId,
                "kind": value.kind.value,
                "subject": value.subject,
                "predicate": value.predicate,
                "source": value.source,
                "confidence": value.confidence,
            },
        )
        return self._commit(
            mission,
            next_snapshot,
            event,
            world_record=_plain(value),
        )

    def apply_revision(
        self,
        revision: PlanRevision | Mapping[str, Any],
    ) -> dict[str, Any]:
        value = revision if isinstance(revision, PlanRevision) else PlanRevision.model_validate(revision)
        record = self.get(value.missionId)
        snapshot = self._snapshot(record)
        active = self._revision(record)
        plan = MissionPlanDraft.model_validate(record["plan"])
        if value.sequence != active.sequence + 1 or value.parentRevisionId != active.revisionId:
            raise MissionConflict("mission revision sequence or parent is stale")
        allowed = set(plan.envelope.allowedSkillIds)
        for node in value.nodes:
            if node.skillId not in allowed:
                raise MissionConflict(f"mission revision uses skill outside envelope: {node.skillId}")
            if _RISK_ORDER[node.riskLevel] > _RISK_ORDER[plan.envelope.maximumRiskLevel]:
                raise MissionConflict(f"mission revision exceeds envelope risk: {node.nodeId}")
        pending_review = value.disposition is ApprovalDisposition.HUMAN_REQUIRED
        if value.disposition is ApprovalDisposition.POLICY_INHERITED:
            authorization = MissionAuthorization.model_validate(record["authorization"])
            if value.authorizationId != authorization.authorizationId:
                raise MissionConflict("policy-inherited revision references another authorization")
            if authorization.expiresAt <= utc_now():
                raise MissionConflict("mission authorization has expired")
            if snapshot.autoRevisionCount >= plan.envelope.maximumAutoRevisions:
                raise MissionConflict("automatic revision budget is exhausted")
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.AWAITING_REVIEW if pending_review else MissionStatus.READY,
                "lastSequence": snapshot.lastSequence + 1,
                "activeRevisionId": snapshot.activeRevisionId if pending_review else value.revisionId,
                "activeNodeId": None,
                "activeSkillRequestId": None,
                "autoRevisionCount": snapshot.autoRevisionCount + (
                    1 if value.disposition is ApprovalDisposition.POLICY_INHERITED else 0
                ),
                "updatedAt": value.createdAt,
                "errorCode": None,
                "detail": "Revision requires human review" if pending_review else None,
            }
        )
        node_states: list[dict[str, Any]] = []
        if not pending_review:
            old_states = self._node_states(record, active.revisionId)
            old_nodes = {node.nodeId: node for node in active.nodes}
            for node in value.nodes:
                old = old_nodes.get(node.nodeId)
                old_state = old_states.get(node.nodeId, {})
                preserve = old == node and old_state.get("status") == "succeeded"
                node_states.append({
                    "revisionId": value.revisionId,
                    "nodeId": node.nodeId,
                    "status": "succeeded" if preserve else "pending",
                    "attempt": int(old_state.get("attempt", 0)) if preserve else 0,
                    "state": old_state.get("state", {}) if preserve else {},
                })
        event = self._event(
            snapshot,
            event_type="revision.awaiting_review" if pending_review else "revision.applied",
            correlation_id=value.revisionId,
            payload={
                "revisionId": value.revisionId,
                "sequence": value.sequence,
                "disposition": value.disposition.value,
                "changeSummary": list(value.changeSummary),
            },
            revision_id=value.revisionId,
        )
        return self._commit(
            record,
            next_snapshot,
            event,
            revision=_plain(value),
            node_states=node_states,
        )

    def approve_pending_revision(
        self,
        mission_id: str,
        revision_id: str,
        *,
        approved_by: str,
    ) -> dict[str, Any]:
        """Approve a stored human-required proposal as a new immutable revision."""

        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        if snapshot.status is not MissionStatus.AWAITING_REVIEW:
            raise MissionConflict("mission is not awaiting revision review")
        pending_payload = next(
            (
                item
                for item in record.get("revisions", [])
                if item.get("revisionId") == revision_id
            ),
            None,
        )
        if pending_payload is None:
            raise MissionConflict("pending mission revision not found")
        pending = PlanRevision.model_validate(pending_payload)
        if pending.disposition is not ApprovalDisposition.HUMAN_REQUIRED:
            raise MissionConflict("mission revision does not require human approval")
        latest_sequence = max(int(item["sequence"]) for item in record["revisions"])
        if pending.sequence != latest_sequence:
            raise MissionConflict("only the latest pending revision can be approved")
        plan = MissionPlanDraft.model_validate(record["plan"])
        now = utc_now()
        expiry = min(plan.expiresAt, plan.envelope.expiresAt)
        if expiry <= now:
            raise MissionConflict("mission draft or envelope has expired")
        authorization = MissionAuthorization(
            authorizationId=f"auth-{mission_id}-{pending.sequence + 1}",
            missionId=mission_id,
            planId=pending.revisionId,
            envelopeSha256=_digest(plan.envelope),
            approvedBy=non_empty_id(approved_by, "approvedBy"),
            approvedAt=now,
            expiresAt=expiry,
        )
        approved = PlanRevision(
            revisionId=f"revision-{mission_id}-{pending.sequence + 1}",
            missionId=mission_id,
            sequence=pending.sequence + 1,
            parentRevisionId=pending.revisionId,
            objective=pending.objective,
            nodes=pending.nodes,
            reasonCode="human_approved_revision",
            reason=f"Human approved proposed revision {pending.revisionId}.",
            changeSummary=[
                *pending.changeSummary,
                f"Approved by {authorization.approvedBy}",
            ],
            disposition=ApprovalDisposition.HUMAN_APPROVED,
            authorizationId=authorization.authorizationId,
            createdBy=authorization.approvedBy,
            createdAt=now,
        )
        active = self._revision(record)
        old_states = self._node_states(record, active.revisionId)
        old_nodes = {node.nodeId: node for node in active.nodes}
        node_states: list[dict[str, Any]] = []
        for node in approved.nodes:
            old = old_nodes.get(node.nodeId)
            old_state = old_states.get(node.nodeId, {})
            preserve = old == node and old_state.get("status") == "succeeded"
            node_states.append(
                {
                    "revisionId": approved.revisionId,
                    "nodeId": node.nodeId,
                    "status": "succeeded" if preserve else "pending",
                    "attempt": int(old_state.get("attempt", 0)) if preserve else 0,
                    "state": old_state.get("state", {}) if preserve else {},
                }
            )
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.READY,
                "lastSequence": snapshot.lastSequence + 1,
                "activeRevisionId": approved.revisionId,
                "activeNodeId": None,
                "activeSkillRequestId": None,
                "updatedAt": now,
                "errorCode": None,
                "detail": None,
            }
        )
        event = self._event(
            snapshot,
            event_type="revision.human_approved",
            correlation_id=approved.revisionId,
            causation_id=pending.revisionId,
            revision_id=approved.revisionId,
            payload={
                "pendingRevisionId": pending.revisionId,
                "authorizationId": authorization.authorizationId,
                "approvedBy": authorization.approvedBy,
            },
        )
        return self._commit(
            record,
            next_snapshot,
            event,
            revision=_plain(approved),
            authorization=_plain(authorization),
            node_states=node_states,
        )

    def intervene(
        self,
        intervention: MissionIntervention | Mapping[str, Any],
    ) -> dict[str, Any]:
        value = (
            intervention
            if isinstance(intervention, MissionIntervention)
            else MissionIntervention.model_validate(intervention)
        )
        record = self.get(value.missionId)
        snapshot = self._snapshot(record)
        if snapshot.status in _TERMINAL:
            raise MissionConflict(f"mission is terminal ({snapshot.status.value})")
        if value.kind is MissionInterventionKind.CANCEL:
            status = MissionStatus.CANCELLED
        elif value.kind is MissionInterventionKind.PAUSE:
            status = MissionStatus.SUSPENDED
        elif value.kind is MissionInterventionKind.RESUME:
            if snapshot.activeSkillRequestId:
                raise MissionConflict("cannot resume while an unresolved skill request is active")
            status = MissionStatus.READY
        else:
            status = MissionStatus.AWAITING_REVIEW
        next_snapshot = snapshot.model_copy(
            update={
                "status": status,
                "lastSequence": snapshot.lastSequence + 1,
                "activeSkillRequestId": None if status in {MissionStatus.CANCELLED, MissionStatus.SUSPENDED} else snapshot.activeSkillRequestId,
                "updatedAt": value.requestedAt,
                "detail": value.text,
            }
        )
        event = self._event(
            snapshot,
            event_type=f"operator.{value.kind.value}",
            correlation_id=value.interventionId,
            payload=_plain(value),
        )
        return self._commit(
            record,
            next_snapshot,
            event,
            error="cancelled by operator" if status is MissionStatus.CANCELLED else None,
        )


__all__ = [
    "MissionConflict",
    "MissionNotFound",
    "MissionRuntime",
    "MissionStore",
]
