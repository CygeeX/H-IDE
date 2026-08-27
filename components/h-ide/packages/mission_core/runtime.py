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
        **values: Any,
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
_PHYSICAL_SKILLS = frozenset(
    {"navigation.navigate_to_pose", "navigation.rotate"}
)
_PLANNED_INTERVENTION_OPERATION = "generic_planner_response"
_PLANNED_INTERVENTION_PENDING = "PLANNER_EVALUATION_PENDING"


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
        owner_user_id: str | None = None,
        site_id: str | None = None,
        autonomy_decision: Mapping[str, Any] | None = None,
        autonomy_grant: Mapping[str, Any] | None = None,
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
            owner_user_id=owner_user_id,
            site_id=site_id,
            autonomy_decision=autonomy_decision,
            autonomy_grant=autonomy_grant,
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
        *,
        next_skill_id: str,
    ) -> float:
        values: list[tuple[str, Any]] = []
        for execution in record.get("skillExecutions", []):
            if not isinstance(execution, Mapping):
                continue
            request = execution.get("request")
            if not isinstance(request, Mapping):
                continue
            skill_id = request.get("skillId")
            if skill_id not in {"navigation.rotate", "inspection.scan_station"}:
                continue
            skill_input = request.get("input")
            field = (
                "targetYawRad"
                if skill_id == "navigation.rotate"
                else "maximumCumulativeYawRad"
            )
            values.append(
                (
                    str(skill_id),
                    skill_input.get(field) if isinstance(skill_input, Mapping) else None,
                )
            )
        next_field = (
            "targetYawRad"
            if next_skill_id == "navigation.rotate"
            else "maximumCumulativeYawRad"
        )
        values.append((next_skill_id, next_input.get(next_field)))
        total = 0.0
        for skill_id, value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MissionConflict(
                    f"rotation budget requires a finite value for {skill_id}"
                )
            yaw = float(value)
            if not math.isfinite(yaw) or yaw == 0:
                raise MissionConflict(
                    f"rotation budget requires a non-zero finite value for {skill_id}"
                )
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

    @staticmethod
    def _assert_no_planned_intervention_fence(
        snapshot: MissionSnapshot,
    ) -> None:
        if snapshot.errorCode == _PLANNED_INTERVENTION_PENDING:
            raise MissionConflict(
                "a planned operator response still owns the Mission fence"
            )

    def start(self, mission_id: str) -> dict[str, Any]:
        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        if snapshot.status in _TERMINAL:
            raise MissionConflict(f"mission is terminal ({snapshot.status.value})")
        self._assert_no_planned_intervention_fence(snapshot)
        if snapshot.status not in {MissionStatus.APPROVED, MissionStatus.READY}:
            raise MissionConflict(f"mission cannot start from {snapshot.status.value}")
        if snapshot.errorCode is not None:
            raise MissionConflict("mission cannot start while an error is unresolved")
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

    def retry_failed_skill(
        self,
        mission_id: str,
        *,
        request_id: str,
        reason: str,
        source_intervention: MissionIntervention | Mapping[str, Any] | None = None,
        source_request_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Ready one unchanged failed node for its next bounded attempt.

        This transition is intentionally narrower than replanning: it cannot
        change the active revision, authorization, node, or input.  The next
        immutable SkillRequest is rebuilt from trusted Mission state, so a
        long-running simulator route may continue from the robot's fresh pose
        without asking a text planner to rewrite reviewed geometry.

        A failed-skill planner claim may already own the durable suspension
        fence.  Supplying that exact claim completes it and schedules the
        retry in the same transaction; there is never an intermediate state
        in which the failed revision can be dispatched accidentally.
        """

        identifier = non_empty_id(request_id, "requestId")
        retry_reason = non_empty_id(reason, "reason")
        supplied_source = (
            source_intervention is not None,
            source_request_sha256 is not None,
        )
        if any(supplied_source) and not all(supplied_source):
            raise MissionConflict("failed skill retry source binding is incomplete")

        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        source_value: MissionIntervention | None = None
        source_digest: str | None = None
        source_state: dict[str, Any] | None = None
        base_snapshot = snapshot
        if source_intervention is not None:
            source_value = (
                source_intervention
                if isinstance(source_intervention, MissionIntervention)
                else MissionIntervention.model_validate(source_intervention)
            )
            if source_value.missionId != mission_id:
                raise MissionConflict("failed skill retry source mission is stale")
            source_digest = non_empty_id(
                source_request_sha256, "sourceRequestSha256"
            )
            source_state = self._planned_intervention_state(
                record, source_value, source_digest
            )
            if source_state["state"] == "completed":
                current_state = next(
                    (
                        item
                        for item in record.get("nodeStates", [])
                        if isinstance(item, Mapping)
                        and isinstance(item.get("state"), Mapping)
                        and item["state"].get("retryScheduledFromRequestId")
                        == identifier
                    ),
                    None,
                )
                if current_state is not None:
                    return record
                raise MissionConflict("failed skill retry source is already completed")
            if source_state["state"] != "pending":
                raise MissionConflict("failed skill retry source was not claimed")
            self._assert_planned_intervention_fence(snapshot, source_value)
            base_snapshot = self._restore_planned_intervention_snapshot(
                snapshot,
                source_value,
                source_state.get("claimEvent"),
            )
        else:
            self._assert_no_planned_intervention_fence(snapshot)

        if base_snapshot.status is not MissionStatus.PLANNING:
            raise MissionConflict(
                f"mission cannot retry a failed skill from {base_snapshot.status.value}"
            )
        if base_snapshot.activeSkillRequestId is not None:
            raise MissionConflict("failed skill retry cannot replace active work")

        revision = self._revision(record)
        execution = next(
            (
                item
                for item in record.get("skillExecutions", [])
                if isinstance(item, Mapping) and item.get("requestId") == identifier
            ),
            None,
        )
        if execution is None:
            raise MissionConflict("failed skill request is unavailable")
        request = execution.get("request")
        result = execution.get("result")
        if not isinstance(request, Mapping) or not isinstance(result, Mapping):
            raise MissionConflict("failed skill request has no terminal result")
        node_id = request.get("nodeId")
        if (
            request.get("missionId") != mission_id
            or request.get("revisionId") != revision.revisionId
            or not isinstance(node_id, str)
            or node_id != execution.get("nodeId")
            or node_id != base_snapshot.activeNodeId
            or result.get("requestId") != identifier
            or result.get("status") == SkillResultStatus.SUCCEEDED.value
        ):
            raise MissionConflict("failed skill retry identity is stale")
        node = next((item for item in revision.nodes if item.nodeId == node_id), None)
        if node is None or node.skillId != request.get("skillId"):
            raise MissionConflict("failed skill node is unavailable")

        states = self._node_states(record, revision.revisionId)
        current = states.get(node_id)
        current_payload = current.get("state") if isinstance(current, Mapping) else None
        if (
            not isinstance(current, Mapping)
            or current.get("status") != result.get("status")
            or not isinstance(current_payload, Mapping)
            or current_payload.get("skillRequestId") != identifier
            or current_payload.get("resultStatus") != result.get("status")
        ):
            raise MissionConflict("failed skill node state is stale")
        attempt = int(current.get("attempt", 0))
        plan = MissionPlanDraft.model_validate(record["plan"])
        if attempt < 1 or attempt >= plan.envelope.maximumStageAttempts:
            raise MissionConflict("mission node attempt budget is exhausted")
        completed = {
            item_id
            for item_id, item in states.items()
            if item.get("status") in {"succeeded", "skipped"}
        }
        if not set(node.dependsOn).issubset(completed):
            raise MissionConflict("failed skill dependencies are no longer complete")

        now = utc_now()
        next_snapshot = base_snapshot.model_copy(
            update={
                "status": MissionStatus.READY,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": node_id,
                "activeSkillRequestId": None,
                "updatedAt": now,
                "errorCode": None,
                "detail": None,
            }
        )
        retry_payload = {
            "requestId": identifier,
            "nodeId": node_id,
            "attempt": attempt,
            "nextAttempt": attempt + 1,
            "reason": retry_reason,
        }
        if source_value is not None and source_digest is not None:
            decision = {
                "kind": "continue",
                "reason": retry_reason,
                "deterministicRetry": retry_payload,
            }
            event = self._event(
                snapshot,
                event_type="operator.intervention.completed",
                correlation_id=source_value.interventionId,
                causation_id=(
                    source_state["claimEvent"].get("eventId")
                    if source_state is not None
                    and isinstance(source_state.get("claimEvent"), Mapping)
                    else None
                ),
                node_id=node_id,
                payload={
                    **_plain(source_value),
                    "operationType": _PLANNED_INTERVENTION_OPERATION,
                    "phase": "completed",
                    "requestSha256": source_digest,
                    "decision": decision,
                },
            )
        else:
            event = self._event(
                snapshot,
                event_type="skill.retry_scheduled",
                correlation_id=f"retry:{identifier}:{attempt + 1}",
                causation_id=identifier,
                node_id=node_id,
                payload=retry_payload,
            )
        return self._commit(
            record,
            next_snapshot,
            event,
            node_states=[
                {
                    "revisionId": revision.revisionId,
                    "nodeId": node_id,
                    "status": "ready",
                    "attempt": attempt,
                    "state": {
                        **dict(current_payload),
                        "retryScheduledFromRequestId": identifier,
                        "retryReason": retry_reason,
                        "nextAttempt": attempt + 1,
                    },
                }
            ],
        )

    def retry_visual_candidate(
        self,
        intervention: MissionIntervention | Mapping[str, Any],
        *,
        node_id: str,
        source_request_id: str,
        candidate_id: str,
        selection_source_analysis_id: str,
    ) -> dict[str, Any]:
        """Atomically bind one validated visual candidate and ready its retry.

        Candidate selection is evidence for the existing low-risk observation
        node, not a semantic plan change.  Keeping it in trusted node state
        preserves the reviewed revision and the first attempt count, so the
        next immutable SkillRequest is attempt 2.  The application service
        resolves natural-language answers, while this runtime boundary still
        re-validates every durable Mission/result identity before committing.
        """

        value = (
            intervention
            if isinstance(intervention, MissionIntervention)
            else MissionIntervention.model_validate(intervention)
        )
        if value.kind is not MissionInterventionKind.ANSWER:
            raise MissionConflict("visual candidate retry requires an operator answer")
        resolved_node_id = non_empty_id(node_id, "nodeId")
        resolved_request_id = non_empty_id(source_request_id, "sourceRequestId")
        resolved_candidate_id = non_empty_id(candidate_id, "candidateId")
        resolved_analysis_id = non_empty_id(
            selection_source_analysis_id,
            "selectionSourceAnalysisId",
        )
        record = self.get(value.missionId)

        selection = {
            "candidateId": resolved_candidate_id,
            "selectionSourceAnalysisId": resolved_analysis_id,
            "sourceRequestId": resolved_request_id,
            "selectedBy": value.requestedBy,
        }
        existing_events = [
            item
            for item in record.get("events", [])
            if isinstance(item, Mapping)
            and item.get("correlationId") == value.interventionId
        ]
        if existing_events:
            if len(existing_events) != 1:
                raise MissionConflict(
                    "interventionId is bound to an invalid event history"
                )
            existing = existing_events[0]
            payload = existing.get("payload")
            previous = (
                payload.get("visualCandidateSelection")
                if isinstance(payload, Mapping)
                else None
            )
            if (
                existing.get("eventType") == "operator.answer"
                and isinstance(payload, Mapping)
                and payload.get("interventionId") == value.interventionId
                and payload.get("kind") == value.kind.value
                and payload.get("text") == value.text
                and payload.get("requestedBy") == value.requestedBy
                and isinstance(previous, Mapping)
                and all(previous.get(key) == item for key, item in selection.items())
            ):
                return record
            raise MissionConflict("interventionId already belongs to another answer")

        snapshot = self._snapshot(record)
        if snapshot.status is not MissionStatus.NEEDS_INPUT:
            raise MissionConflict(
                f"mission cannot select a visual candidate from {snapshot.status.value}"
            )
        if snapshot.activeSkillRequestId:
            raise MissionConflict("visual candidate retry cannot replace an active skill")
        if snapshot.errorCode != "VISUAL_NEEDS_INPUT":
            raise MissionConflict("mission is not waiting for a visual candidate")
        if snapshot.activeNodeId != resolved_node_id:
            raise MissionConflict("visual candidate node is not the active input request")

        plan = MissionPlanDraft.model_validate(record["plan"])
        authorization = MissionAuthorization.model_validate(record["authorization"])
        now = utc_now()
        if min(plan.expiresAt, plan.envelope.expiresAt, authorization.expiresAt) <= now:
            raise MissionConflict("mission authorization has expired")
        revision = self._revision(record)
        node = next(
            (item for item in revision.nodes if item.nodeId == resolved_node_id),
            None,
        )
        if node is None or node.skillId != "vision.observe":
            raise MissionConflict("visual candidate node is not in the active revision")

        states = self._node_states(record, revision.revisionId)
        current_state = states.get(resolved_node_id)
        if not isinstance(current_state, Mapping) or current_state.get("status") != "rejected":
            raise MissionConflict("visual candidate source node is not rejected")
        attempt = int(current_state.get("attempt", 0))
        if attempt < 1:
            raise MissionConflict("visual candidate source attempt is invalid")
        if attempt >= plan.envelope.maximumStageAttempts:
            raise MissionConflict("mission node attempt budget is exhausted")

        current_state_payload = current_state.get("state")
        current_request_id = (
            current_state_payload.get("skillRequestId")
            if isinstance(current_state_payload, Mapping)
            else None
        )
        if current_request_id != resolved_request_id:
            raise MissionConflict("visual candidate source request is stale")

        execution = next(
            (
                item
                for item in record.get("skillExecutions", [])
                if isinstance(item, Mapping)
                and item.get("requestId") == current_request_id
            ),
            None,
        )
        if execution is None:
            raise MissionConflict("visual candidate source request is unavailable")
        terminal = execution.get("result")
        terminal_payload = terminal.get("result") if isinstance(terminal, Mapping) else None
        visual = (
            terminal_payload.get("visualResult")
            if isinstance(terminal_payload, Mapping)
            else None
        )
        request = execution["request"]
        if (
            not isinstance(terminal, Mapping)
            or terminal.get("status") != "rejected"
            or terminal.get("errorCode") != "VISUAL_NEEDS_INPUT"
            or not isinstance(visual, Mapping)
            or visual.get("status") != "needs_input"
            or visual.get("rejectionReason") != "ambiguous_target"
            or visual.get("analysisId") != resolved_analysis_id
            or visual.get("requestId") != resolved_request_id
            or visual.get("missionId") != value.missionId
            or visual.get("revisionId") != revision.revisionId
            or visual.get("nodeId") != resolved_node_id
            or visual.get("attempt") != attempt
            or request.get("missionId") != value.missionId
            or request.get("revisionId") != revision.revisionId
            or request.get("nodeId") != resolved_node_id
        ):
            raise MissionConflict("visual candidate source identity is invalid")
        candidates = visual.get("ambiguityCandidates")
        if (
            not isinstance(candidates, list)
            or not any(
                isinstance(item, Mapping)
                and item.get("candidateId") == resolved_candidate_id
                for item in candidates
            )
        ):
            raise MissionConflict("visual candidate is not in the latest ambiguity result")

        selection["selectedAt"] = value.requestedAt.isoformat()
        state_payload = {
            **dict(current_state.get("state", {})),
            "visualCandidateSelection": selection,
        }
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.READY,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": resolved_node_id,
                "activeSkillRequestId": None,
                "updatedAt": value.requestedAt,
                "errorCode": None,
                "detail": None,
            }
        )
        event = self._event(
            snapshot,
            event_type="operator.answer",
            correlation_id=value.interventionId,
            payload={
                **_plain(value),
                "visualCandidateSelection": selection,
            },
            node_id=resolved_node_id,
            revision_id=revision.revisionId,
        )
        try:
            return self._commit(
                record,
                next_snapshot,
                event,
                node_states=[
                    {
                        "revisionId": revision.revisionId,
                        "nodeId": resolved_node_id,
                        "status": "pending",
                        "attempt": attempt,
                        "state": state_payload,
                    }
                ],
            )
        except MissionConflict as exc:
            if str(exc) != "mission state changed concurrently":
                raise
            refreshed = self.get(value.missionId)
            matching = [
                item
                for item in refreshed.get("events", [])
                if isinstance(item, Mapping)
                and item.get("correlationId") == value.interventionId
            ]
            if len(matching) == 1 and matching[0].get("eventType") == "operator.answer":
                payload = matching[0].get("payload")
                previous = (
                    payload.get("visualCandidateSelection")
                    if isinstance(payload, Mapping)
                    else None
                )
                if (
                    isinstance(payload, Mapping)
                    and payload.get("interventionId") == value.interventionId
                    and payload.get("kind") == value.kind.value
                    and payload.get("text") == value.text
                    and payload.get("requestedBy") == value.requestedBy
                    and isinstance(previous, Mapping)
                    and all(
                        previous.get(key) == item
                        for key, item in selection.items()
                        if key != "selectedAt"
                    )
                ):
                    return refreshed
            raise

    @staticmethod
    def _planned_intervention_state(
        record: Mapping[str, Any],
        value: MissionIntervention,
        request_sha256: str,
    ) -> dict[str, Any]:
        generic: list[Mapping[str, Any]] = []
        collisions: list[Mapping[str, Any]] = []
        for item in record.get("events", []):
            if not isinstance(item, Mapping):
                continue
            payload = item.get("payload")
            planned = (
                isinstance(payload, Mapping)
                and payload.get("operationType") == _PLANNED_INTERVENTION_OPERATION
                and payload.get("interventionId") == value.interventionId
            )
            if planned:
                generic.append(item)
            elif item.get("correlationId") == value.interventionId:
                collisions.append(item)
        if not generic:
            if collisions:
                raise MissionConflict(
                    "interventionId already belongs to another operation"
                )
            return {
                "state": "none",
                "mission": record,
                "decision": None,
                "createdAt": value.requestedAt.isoformat(),
                "claimEvent": None,
            }
        for item in generic:
            payload = item.get("payload")
            if (
                not isinstance(payload, Mapping)
                or payload.get("requestSha256") != request_sha256
                or payload.get("interventionId") != value.interventionId
                or payload.get("kind") != value.kind.value
                or payload.get("text") != value.text
                or payload.get("requestedBy") != value.requestedBy
            ):
                raise MissionConflict("interventionId already belongs to another message")
        ordered = sorted(generic, key=lambda item: int(item.get("sequence", 0)))
        claim = next(
            (
                item
                for item in reversed(ordered)
                if isinstance(item.get("payload"), Mapping)
                and item["payload"].get("phase") == "claimed"
            ),
            None,
        )
        completed = next(
            (
                item
                for item in reversed(ordered)
                if isinstance(item.get("payload"), Mapping)
                and item["payload"].get("phase") == "completed"
            ),
            None,
        )
        first_payload = ordered[0].get("payload")
        created_at = (
            first_payload.get("requestedAt")
            if isinstance(first_payload, Mapping)
            else None
        ) or value.requestedAt.isoformat()
        if completed is not None:
            payload = completed.get("payload")
            decision = payload.get("decision") if isinstance(payload, Mapping) else None
            if not isinstance(decision, Mapping):
                raise MissionConflict("completed intervention has no durable decision")
            return {
                "state": "completed",
                "mission": record,
                "decision": dict(decision),
                "createdAt": created_at,
                "claimEvent": claim,
            }
        if collisions:
            raise MissionConflict("interventionId already belongs to another operation")
        snapshot_payload = record.get("snapshot")
        snapshot_status = (
            snapshot_payload.get("status")
            if isinstance(snapshot_payload, Mapping)
            else None
        )
        if snapshot_status in {item.value for item in _TERMINAL}:
            return {
                "state": "aborted",
                "mission": record,
                "decision": None,
                "createdAt": created_at,
                "claimEvent": claim,
            }
        latest_payload = ordered[-1].get("payload")
        phase = latest_payload.get("phase") if isinstance(latest_payload, Mapping) else None
        if phase not in {"claimed", "retryable"}:
            raise MissionConflict("interventionId is bound to an invalid event history")
        return {
            "state": "pending" if phase == "claimed" else "retryable",
            "mission": record,
            "decision": None,
            "createdAt": created_at,
            "claimEvent": claim,
        }

    def inspect_planned_intervention(
        self,
        intervention: MissionIntervention | Mapping[str, Any],
        *,
        request_sha256: str,
    ) -> dict[str, Any]:
        value = (
            intervention
            if isinstance(intervention, MissionIntervention)
            else MissionIntervention.model_validate(intervention)
        )
        digest = non_empty_id(request_sha256, "requestSha256")
        return self._planned_intervention_state(
            self.get(value.missionId), value, digest
        )

    @staticmethod
    def _assert_planned_intervention_fence(
        snapshot: MissionSnapshot,
        value: MissionIntervention,
    ) -> None:
        if (
            snapshot.status is not MissionStatus.SUSPENDED
            or snapshot.errorCode != _PLANNED_INTERVENTION_PENDING
            or snapshot.detail
            != f"Planner evaluation pending for intervention {value.interventionId}"
            or snapshot.activeSkillRequestId is not None
        ):
            raise MissionConflict("planned intervention no longer owns the Mission fence")

    def claim_planned_intervention(
        self,
        intervention: MissionIntervention | Mapping[str, Any],
        *,
        request_sha256: str,
        expected_last_sequence: int,
    ) -> dict[str, Any]:
        """Durably fence a generic planner call without fabricating review state."""

        value = (
            intervention
            if isinstance(intervention, MissionIntervention)
            else MissionIntervention.model_validate(intervention)
        )
        digest = non_empty_id(request_sha256, "requestSha256")
        record = self.get(value.missionId)
        existing = self._planned_intervention_state(record, value, digest)
        if existing["state"] == "completed":
            return {**existing, "claimedByCaller": False}
        snapshot = self._snapshot(record)
        if existing["state"] == "aborted":
            raise MissionConflict(
                f"mission is terminal ({snapshot.status.value})"
            )
        if existing["state"] == "pending":
            self._assert_planned_intervention_fence(snapshot, value)
            # A call that observes a previously durable pending claim is an
            # explicit retry (the service serializes calls per Mission).
            return {**existing, "claimedByCaller": True}
        if snapshot.status in _TERMINAL:
            raise MissionConflict(f"mission is terminal ({snapshot.status.value})")
        if snapshot.activeSkillRequestId:
            raise MissionConflict(
                "an active skill must finish or be cancelled before changing the Mission"
            )
        if snapshot.errorCode == _PLANNED_INTERVENTION_PENDING:
            raise MissionConflict("another planned intervention owns the Mission fence")
        if snapshot.lastSequence != int(expected_last_sequence):
            raise MissionConflict("mission state changed concurrently")
        previous_snapshot = _plain(snapshot)
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.SUSPENDED,
                "lastSequence": snapshot.lastSequence + 1,
                "activeSkillRequestId": None,
                "updatedAt": value.requestedAt,
                "errorCode": _PLANNED_INTERVENTION_PENDING,
                "detail": (
                    f"Planner evaluation pending for intervention "
                    f"{value.interventionId}"
                ),
            }
        )
        event = self._event(
            snapshot,
            event_type=f"operator.{value.kind.value}",
            correlation_id=value.interventionId,
            payload={
                **_plain(value),
                "operationType": _PLANNED_INTERVENTION_OPERATION,
                "phase": "claimed",
                "requestSha256": digest,
                "previousSnapshot": previous_snapshot,
            },
        )
        try:
            committed = self._commit(record, next_snapshot, event)
        except MissionConflict as exc:
            if str(exc) != "mission state changed concurrently":
                raise
            refreshed = self._planned_intervention_state(
                self.get(value.missionId), value, digest
            )
            if refreshed["state"] in {"pending", "completed"}:
                # This caller lost the authoritative CAS.  It must never run
                # the planner in parallel with the winner.
                return {**refreshed, "claimedByCaller": False}
            raise
        return {
            **self._planned_intervention_state(committed, value, digest),
            "claimedByCaller": True,
        }

    def _restore_planned_intervention_snapshot(
        self,
        snapshot: MissionSnapshot,
        value: MissionIntervention,
        claim_event: Mapping[str, Any] | None,
    ) -> MissionSnapshot:
        self._assert_planned_intervention_fence(snapshot, value)
        payload = claim_event.get("payload") if isinstance(claim_event, Mapping) else None
        previous_payload = (
            payload.get("previousSnapshot") if isinstance(payload, Mapping) else None
        )
        if not isinstance(previous_payload, Mapping):
            raise MissionConflict("planned intervention has no recovery snapshot")
        previous = MissionSnapshot.model_validate(previous_payload)
        if (
            previous.missionId != snapshot.missionId
            or previous.robotId != snapshot.robotId
            or previous.activeRevisionId != snapshot.activeRevisionId
            or previous.activeSkillRequestId is not None
        ):
            raise MissionConflict("planned intervention recovery snapshot is stale")
        return previous.model_copy(
            update={
                "lastSequence": snapshot.lastSequence + 1,
                "updatedAt": utc_now(),
            }
        )

    def finalize_planned_intervention(
        self,
        intervention: MissionIntervention | Mapping[str, Any],
        *,
        request_sha256: str,
        decision: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Commit one planner outcome; revision and review state stay atomic."""

        value = (
            intervention
            if isinstance(intervention, MissionIntervention)
            else MissionIntervention.model_validate(intervention)
        )
        digest = non_empty_id(request_sha256, "requestSha256")
        normalized = dict(decision)
        record = self.get(value.missionId)
        state = self._planned_intervention_state(record, value, digest)
        if state["state"] == "completed":
            return state
        if state["state"] == "aborted":
            snapshot = self._snapshot(record)
            raise MissionConflict(
                f"mission is terminal ({snapshot.status.value})"
            )
        kind = normalized.get("kind")
        if kind == "replan" and isinstance(normalized.get("revision"), Mapping):
            self.apply_revision(
                normalized["revision"],
                source_intervention=value,
                source_request_sha256=digest,
                source_decision=normalized,
            )
            return self.inspect_planned_intervention(
                value, request_sha256=digest
            )
        if kind != "continue":
            raise MissionConflict(
                "generic operator response requires continue or a valid replan"
            )
        if state["state"] != "pending":
            raise MissionConflict("planned intervention was not claimed")
        snapshot = self._snapshot(record)
        next_snapshot = self._restore_planned_intervention_snapshot(
            snapshot, value, state.get("claimEvent")
        )
        event = self._event(
            snapshot,
            event_type="operator.intervention.completed",
            correlation_id=value.interventionId,
            causation_id=(
                state["claimEvent"].get("eventId")
                if isinstance(state.get("claimEvent"), Mapping)
                else None
            ),
            payload={
                **_plain(value),
                "operationType": _PLANNED_INTERVENTION_OPERATION,
                "phase": "completed",
                "requestSha256": digest,
                "decision": normalized,
            },
        )
        try:
            completed = self._commit(record, next_snapshot, event)
        except MissionConflict as exc:
            if str(exc) != "mission state changed concurrently":
                raise
            refreshed = self._planned_intervention_state(
                self.get(value.missionId), value, digest
            )
            if refreshed["state"] == "completed":
                return refreshed
            raise
        return self._planned_intervention_state(completed, value, digest)

    def issue_skill(
        self,
        mission_id: str,
        node_id: str,
        input: Mapping[str, Any],
        *,
        expected_last_sequence: int,
        expected_revision_id: str,
        expected_authorization_id: str,
        expires_in_seconds: float = 60,
    ) -> SkillRequest:
        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        revision = self._revision(record)
        authorization = MissionAuthorization.model_validate(record["authorization"])
        if (
            snapshot.lastSequence != expected_last_sequence
            or snapshot.activeRevisionId != expected_revision_id
            or revision.revisionId != expected_revision_id
            or authorization.authorizationId != expected_authorization_id
        ):
            raise MissionConflict("mission state changed before skill issue")
        if snapshot.status is not MissionStatus.READY:
            raise MissionConflict(f"mission cannot issue a skill from {snapshot.status.value}")
        if snapshot.errorCode is not None:
            raise MissionConflict("mission cannot issue a skill while an error is unresolved")
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
            now + timedelta(seconds=min(lifetime_seconds, 2_400.0)),
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
        elif node.skillId in {"navigation.rotate", "inspection.scan_station"}:
            cumulative_scan_yaw = self._reserved_scan_yaw(
                record,
                bound_input,
                next_skill_id=node.skillId,
            )
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

    def record_recovery_required(
        self,
        mission_id: str,
        *,
        request_id: str,
        error_code: str,
        detail: str,
    ) -> dict[str, Any]:
        """Atomically close a non-physical request whose normal result write failed.

        This is deliberately a separate transition from :meth:`record_result`.
        The application service uses it only as the last durable fallback when
        recording the original workstation failure raised.  The skill gets an
        ``unknown`` terminal result, while the Mission is suspended so neither
        ``advance`` nor a process restart can silently issue another attempt.
        An operator can still inspect or cancel the Mission.
        """

        identifier = non_empty_id(request_id, "requestId")
        code = non_empty_id(error_code, "errorCode")
        safe_detail = non_empty_id(detail, "detail")
        record = self.get(mission_id)
        execution = next(
            (
                item
                for item in record["skillExecutions"]
                if item["requestId"] == identifier
            ),
            None,
        )
        if execution is None or execution.get("result") is not None:
            raise MissionConflict(
                "skill recovery result is not bound to an active request"
            )
        snapshot = self._snapshot(record)
        if snapshot.activeSkillRequestId != identifier:
            raise MissionConflict("skill recovery request is not active")
        revision = self._revision(record)
        states = self._node_states(record, revision.revisionId)
        current_state = states.get(execution["nodeId"], {})
        now = utc_now()
        value = SkillResult(
            requestId=identifier,
            status=SkillResultStatus.UNKNOWN,
            completedAt=now,
            result={"recoveryRequired": True},
            errorCode=code,
            detail=safe_detail,
        )
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.SUSPENDED,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": execution["nodeId"],
                "activeSkillRequestId": None,
                "updatedAt": now,
                "errorCode": code,
                "detail": safe_detail,
            }
        )
        event = self._event(
            snapshot,
            event_type="skill.recovery_required",
            correlation_id=identifier,
            payload={
                "requestId": identifier,
                "status": SkillResultStatus.UNKNOWN.value,
                "errorCode": code,
                "recoveryRequired": True,
            },
            node_id=execution["nodeId"],
        )
        return self._commit(
            record,
            next_snapshot,
            event,
            skill_result=_plain(value),
            node_states=[
                {
                    "revisionId": revision.revisionId,
                    "nodeId": execution["nodeId"],
                    "status": SkillResultStatus.UNKNOWN.value,
                    "attempt": int(current_state.get("attempt", 1)),
                    "state": {
                        **dict(current_state.get("state", {})),
                        "skillRequestId": identifier,
                        "resultStatus": SkillResultStatus.UNKNOWN.value,
                        "recoveryRequired": True,
                    },
                }
            ],
            error=safe_detail,
        )

    def mark_reconciliation_required(
        self,
        mission_id: str,
        *,
        request_id: str,
        error_code: str,
        detail: str,
    ) -> dict[str, Any]:
        """Keep an externally dispatched request active until it is reconciled.

        A Robot Agent call can fail after the remote side has accepted the
        idempotent command.  Therefore this transition records uncertainty but
        does *not* invent a terminal SkillResult and does not clear the active
        request.  Subsequent ``advance`` calls take the reconciliation branch
        and cannot dispatch the physical command again.
        """

        identifier = non_empty_id(request_id, "requestId")
        code = non_empty_id(error_code, "errorCode")
        safe_detail = non_empty_id(detail, "detail")
        record = self.get(mission_id)
        execution = next(
            (
                item
                for item in record["skillExecutions"]
                if item["requestId"] == identifier
            ),
            None,
        )
        if execution is None or execution.get("result") is not None:
            raise MissionConflict(
                "skill reconciliation marker is not bound to an active request"
            )
        snapshot = self._snapshot(record)
        if snapshot.activeSkillRequestId != identifier:
            raise MissionConflict("skill reconciliation request is not active")
        revision = self._revision(record)
        states = self._node_states(record, revision.revisionId)
        current_state = states.get(execution["nodeId"], {})
        now = utc_now()
        next_snapshot = snapshot.model_copy(
            update={
                "status": MissionStatus.DISPATCHING,
                "lastSequence": snapshot.lastSequence + 1,
                "activeNodeId": execution["nodeId"],
                "activeSkillRequestId": identifier,
                "updatedAt": now,
                "errorCode": code,
                "detail": safe_detail,
            }
        )
        event = self._event(
            snapshot,
            event_type="skill.reconciliation_required",
            correlation_id=identifier,
            payload={
                "requestId": identifier,
                "errorCode": code,
                "recoveryRequired": True,
            },
            node_id=execution["nodeId"],
        )
        return self._commit(
            record,
            next_snapshot,
            event,
            node_states=[
                {
                    "revisionId": revision.revisionId,
                    "nodeId": execution["nodeId"],
                    "status": "running",
                    "attempt": int(current_state.get("attempt", 1)),
                    "state": {
                        **dict(current_state.get("state", {})),
                        "skillRequestId": identifier,
                        "dispatchState": "unknown_reconcile_required",
                    },
                }
            ],
            error=safe_detail,
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
        self._assert_no_planned_intervention_fence(snapshot)
        if snapshot.activeSkillRequestId is not None:
            raise MissionConflict("cannot skip nodes while a skill request is active")
        if snapshot.status is not MissionStatus.READY:
            raise MissionConflict(
                f"mission cannot skip nodes from {snapshot.status.value}"
            )
        if snapshot.errorCode is not None:
            raise MissionConflict("mission cannot skip nodes while an error is unresolved")
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

    @staticmethod
    def _world_record_is_exact_replay(
        mission: Mapping[str, Any],
        value: WorldRecord,
    ) -> bool:
        matching_events = [
            item
            for item in mission.get("events", [])
            if isinstance(item, Mapping)
            and item.get("correlationId") == value.recordId
        ]
        matching_records = [
            item
            for item in mission.get("worldRecords", [])
            if isinstance(item, Mapping)
            and item.get("recordId") == value.recordId
        ]
        if not matching_events and not matching_records:
            return False
        if (
            len(matching_events) == 1
            and matching_events[0].get("eventType") == "world.recorded"
            and len(matching_records) == 1
            and dict(matching_records[0]) == _plain(value)
        ):
            return True
        raise MissionConflict("recordId already belongs to another operation")

    def record_world(self, record: WorldRecord | Mapping[str, Any]) -> dict[str, Any]:
        value = (
            record
            if isinstance(record, WorldRecord)
            else WorldRecord.model_validate(record)
        )
        mission = self.get(value.missionId)
        if self._world_record_is_exact_replay(mission, value):
            return mission
        snapshot = self._snapshot(mission)
        self._assert_no_planned_intervention_fence(snapshot)
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
        try:
            return self._commit(
                mission,
                next_snapshot,
                event,
                world_record=_plain(value),
            )
        except MissionConflict as exc:
            if str(exc) != "mission state changed concurrently":
                raise
            refreshed = self.get(value.missionId)
            if self._world_record_is_exact_replay(refreshed, value):
                return refreshed
            raise

    def apply_revision(
        self,
        revision: PlanRevision | Mapping[str, Any],
        *,
        source_intervention: MissionIntervention | Mapping[str, Any] | None = None,
        source_request_sha256: str | None = None,
        source_decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        value = revision if isinstance(revision, PlanRevision) else PlanRevision.model_validate(revision)
        record = self.get(value.missionId)
        snapshot = self._snapshot(record)
        if snapshot.activeSkillRequestId is not None:
            raise MissionConflict(
                "cannot apply a revision while a skill request is active"
            )
        source_value: MissionIntervention | None = None
        source_digest: str | None = None
        source_state: dict[str, Any] | None = None
        supplied_source_fields = (
            source_intervention is not None,
            source_request_sha256 is not None,
            source_decision is not None,
        )
        if any(supplied_source_fields) and not all(supplied_source_fields):
            raise MissionConflict("planned revision source binding is incomplete")
        if source_intervention is not None:
            source_value = (
                source_intervention
                if isinstance(source_intervention, MissionIntervention)
                else MissionIntervention.model_validate(source_intervention)
            )
            if source_value.missionId != value.missionId:
                raise MissionConflict("planned revision source mission is stale")
            source_digest = non_empty_id(
                source_request_sha256, "sourceRequestSha256"
            )
            source_state = self._planned_intervention_state(
                record, source_value, source_digest
            )
            if source_state["state"] == "completed":
                return record
            if source_state["state"] != "pending":
                raise MissionConflict("planned intervention was not claimed")
            self._assert_planned_intervention_fence(snapshot, source_value)
        else:
            self._assert_no_planned_intervention_fence(snapshot)
        active = self._revision(record)
        plan = MissionPlanDraft.model_validate(record["plan"])
        if value.disposition is ApprovalDisposition.HUMAN_APPROVED:
            raise MissionConflict(
                "planner revisions cannot claim human approval; use the explicit approval endpoint"
            )
        if (
            value.disposition is ApprovalDisposition.POLICY_INHERITED
            and any(node.skillId in _PHYSICAL_SKILLS for node in value.nodes)
        ):
            raise MissionConflict(
                "physical revisions require explicit human approval in this release"
            )
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
                preserve_attempt = (
                    old is not None
                    and old.skillId == node.skillId
                    and old.skillVersion == node.skillVersion
                )
                node_states.append({
                    "revisionId": value.revisionId,
                    "nodeId": node.nodeId,
                    "status": "succeeded" if preserve else "pending",
                    "attempt": (
                        int(old_state.get("attempt", 0))
                        if preserve_attempt
                        else 0
                    ),
                    "state": old_state.get("state", {}) if preserve else {},
                })
        event_payload: dict[str, Any] = {
            "revisionId": value.revisionId,
            "sequence": value.sequence,
            "disposition": value.disposition.value,
            "changeSummary": list(value.changeSummary),
        }
        if source_value is not None and source_digest is not None:
            event_payload.update(
                {
                    **_plain(source_value),
                    "operationType": _PLANNED_INTERVENTION_OPERATION,
                    "phase": "completed",
                    "requestSha256": source_digest,
                    "decision": dict(source_decision or {}),
                }
            )
        event = self._event(
            snapshot,
            event_type="revision.awaiting_review" if pending_review else "revision.applied",
            correlation_id=(
                source_value.interventionId
                if source_value is not None
                else value.revisionId
            ),
            causation_id=(
                source_state["claimEvent"].get("eventId")
                if source_state is not None
                and isinstance(source_state.get("claimEvent"), Mapping)
                else None
            ),
            payload=event_payload,
            revision_id=value.revisionId,
        )
        try:
            return self._commit(
                record,
                next_snapshot,
                event,
                revision=_plain(value),
                node_states=node_states,
            )
        except MissionConflict as exc:
            if (
                source_value is None
                or source_digest is None
                or str(exc) != "mission state changed concurrently"
            ):
                raise
            refreshed = self._planned_intervention_state(
                self.get(value.missionId), source_value, source_digest
            )
            if refreshed["state"] == "completed":
                return refreshed["mission"]
            raise

    def approve_pending_revision(
        self,
        mission_id: str,
        revision_id: str,
        *,
        approved_by: str,
        autonomy_decision: Mapping[str, Any] | None = None,
        autonomy_grant: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Approve a stored human-required proposal as a new immutable revision."""

        record = self.get(mission_id)
        snapshot = self._snapshot(record)
        self._assert_no_planned_intervention_fence(snapshot)
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
            preserve_attempt = (
                old is not None
                and old.skillId == node.skillId
                and old.skillVersion == node.skillVersion
            )
            node_states.append(
                {
                    "revisionId": approved.revisionId,
                    "nodeId": node.nodeId,
                    "status": "succeeded" if preserve else "pending",
                    "attempt": (
                        int(old_state.get("attempt", 0))
                        if preserve_attempt
                        else 0
                    ),
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
            autonomy_decision=autonomy_decision,
            autonomy_grant=autonomy_grant,
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
        matching = [
            item
            for item in record.get("events", [])
            if isinstance(item, Mapping)
            and item.get("correlationId") == value.interventionId
        ]
        if matching:
            if len(matching) == 1:
                event = matching[0]
                payload = event.get("payload")
                if (
                    event.get("eventType") == f"operator.{value.kind.value}"
                    and isinstance(payload, Mapping)
                    and payload.get("operationType") is None
                    and payload.get("interventionId") == value.interventionId
                    and payload.get("kind") == value.kind.value
                    and payload.get("text") == value.text
                    and payload.get("requestedBy") == value.requestedBy
                ):
                    return record
            raise MissionConflict("interventionId already belongs to another operation")
        snapshot = self._snapshot(record)
        if snapshot.status in _TERMINAL:
            raise MissionConflict(f"mission is terminal ({snapshot.status.value})")
        if snapshot.activeSkillRequestId:
            raise MissionConflict(
                "an active skill must be cancelled through the Mission cancellation endpoint"
            )
        if value.kind is not MissionInterventionKind.CANCEL:
            self._assert_no_planned_intervention_fence(snapshot)
        if value.kind is MissionInterventionKind.CANCEL:
            status = MissionStatus.CANCELLED
        elif value.kind is MissionInterventionKind.PAUSE:
            if snapshot.status is not MissionStatus.READY:
                raise MissionConflict(
                    f"mission cannot pause from {snapshot.status.value}"
                )
            if snapshot.errorCode is not None:
                raise MissionConflict("mission cannot pause while an error is unresolved")
            status = MissionStatus.SUSPENDED
        elif value.kind is MissionInterventionKind.RESUME:
            def latest_direct_intervention_sequence(
                kind: MissionInterventionKind,
            ) -> int:
                event_type = f"operator.{kind.value}"
                sequences = []
                for item in record.get("events", []):
                    if not isinstance(item, Mapping):
                        continue
                    sequence = item.get("sequence")
                    payload = item.get("payload")
                    if (
                        type(sequence) is not int
                        or sequence > snapshot.lastSequence
                        or item.get("eventType") != event_type
                        or not isinstance(payload, Mapping)
                        or payload.get("operationType") is not None
                        or payload.get("kind") != kind.value
                        or payload.get("interventionId")
                        != item.get("correlationId")
                    ):
                        continue
                    sequences.append(sequence)
                return max(sequences, default=-1)

            latest_pause_sequence = latest_direct_intervention_sequence(
                MissionInterventionKind.PAUSE
            )
            latest_resume_sequence = latest_direct_intervention_sequence(
                MissionInterventionKind.RESUME
            )
            if (
                snapshot.status is not MissionStatus.SUSPENDED
                or snapshot.errorCode is not None
                or snapshot.activeSkillRequestId is not None
                or latest_pause_sequence <= latest_resume_sequence
            ):
                raise MissionConflict(
                    "mission can resume only from an unconsumed explicit operator pause"
                )
            status = MissionStatus.READY
        else:
            status = MissionStatus.AWAITING_REVIEW
        next_snapshot = snapshot.model_copy(
            update={
                "status": status,
                "lastSequence": snapshot.lastSequence + 1,
                "activeSkillRequestId": None if status in {MissionStatus.CANCELLED, MissionStatus.SUSPENDED} else snapshot.activeSkillRequestId,
                "updatedAt": value.requestedAt,
                "errorCode": (
                    None
                    if status in {MissionStatus.CANCELLED, MissionStatus.READY}
                    else snapshot.errorCode
                ),
                "detail": (
                    None
                    if status in {MissionStatus.CANCELLED, MissionStatus.READY}
                    else value.text
                ),
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
