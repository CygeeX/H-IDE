from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import uuid4

from packages.control_core.models import (
    CommandDefinition,
    CommandKind,
    CommandRequest,
    CommandSource,
    MappingVersion,
    Transport,
    utc_now,
)
from packages.control_core.runtime import ActionRuntime
from packages.scene_core import NAVIGATION_COMMAND_KEY, visual_navigation_command_schema

from .clients import VisionClient, VisualStageClient
from .models import VisualMissionMode, VisualMissionPlanDraft
from .validation import validate_visual_resolution, validate_visual_resolution_binding

if TYPE_CHECKING:
    from services.control_api.store import SQLiteStore


class VisualMissionRuntime:
    """Durable deterministic mission orchestrator above ActionRuntime."""

    _motion_lock = threading.RLock()
    _motion_owners: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        *,
        store: SQLiteStore,
        action_runtime: ActionRuntime,
        vision_client: VisionClient,
        stage_client: VisualStageClient,
        mode: str | VisualMissionMode = VisualMissionMode.OBSERVE,
        mapping_resolver: Callable[[str], MappingVersion] | None = None,
        command_poll_interval_seconds: float = 0.25,
    ) -> None:
        self.store = store
        self.action_runtime = action_runtime
        self.vision_client = vision_client
        self.stage_client = stage_client
        self.mode = mode if isinstance(mode, VisualMissionMode) else VisualMissionMode(mode)
        self.mapping_resolver = mapping_resolver
        self.command_poll_interval_seconds = float(command_poll_interval_seconds)
        if not 0.01 <= self.command_poll_interval_seconds <= 5.0:
            raise ValueError("command poll interval must be between 0.01 and 5 seconds")
        self.definition = CommandDefinition(
            commandKey=NAVIGATION_COMMAND_KEY,
            commandVersion="1",
            transports=[Transport.FAKE, Transport.ROBOT_AGENT],
            kind=CommandKind.MOTION,
            payloadSchema=visual_navigation_command_schema(),
        )

    def start(self, draft: VisualMissionPlanDraft, *, draft_id: str) -> dict[str, Any]:
        if self.mode is VisualMissionMode.OFF:
            raise ValueError("visual missions are disabled")
        if draft_id != draft.planId:
            raise ValueError("visual mission draft identity mismatch")
        mission = self.store.create_visual_mission(
            draft.mission.missionId,
            draft_id,
            draft.robotId,
            draft.model_dump(mode="json"),
        )
        return self._public(mission)

    def get(self, mission_id: str) -> dict[str, Any] | None:
        mission = self.store.get_visual_mission(mission_id)
        return None if mission is None else self._public(mission)

    @classmethod
    def motion_owner(cls, robot_id: str) -> str | None:
        with cls._motion_lock:
            return cls._motion_owners.get(robot_id)

    def run(self, mission_id: str) -> dict[str, Any]:
        mission = self.store.get_visual_mission(mission_id)
        if mission is None:
            raise ValueError("visual mission not found")
        if self.mode is VisualMissionMode.OBSERVE:
            raise ValueError("observe mode cannot dispatch a visual mission")
        if mission["status"] == "completed":
            return self._public(mission)
        if mission["status"] != "approved":
            raise ValueError(f"visual mission is not runnable (status={mission['status']})")
        draft = VisualMissionPlanDraft.model_validate(mission["plan"])
        self._claim_motion(draft.robotId, mission_id)
        try:
            return self._run_claimed(draft, mission)
        except Exception as exc:
            current = self.store.get_visual_mission(mission_id)
            if current is not None and current["status"] not in {
                "completed",
                "cancelled",
                "unknown",
            }:
                self.store.transition_visual_mission(
                    mission_id,
                    "failed",
                    expected_status=current["status"],
                    state=current["state"],
                    error=str(exc),
                )
            raise
        finally:
            self._release_motion(draft.robotId, mission_id)

    def select_candidate(self, mission_id: str, candidate_id: str) -> dict[str, Any]:
        """Approve one immutable ambiguity candidate for a fresh resolution attempt."""

        candidate_id = candidate_id.strip()
        if not candidate_id:
            raise ValueError("visual candidateId must not be empty")
        mission = self.store.get_visual_mission(mission_id)
        if mission is None:
            raise ValueError("visual mission not found")
        if mission["status"] != "needs_input":
            raise ValueError(
                f"visual mission is not awaiting a candidate (status={mission['status']})"
            )
        state = dict(mission["state"])
        draft = VisualMissionPlanDraft.model_validate(mission["plan"])
        stage_index = int(state.get("currentStageIndex", 0))
        if stage_index < 0 or stage_index >= len(draft.mission.stages):
            raise ValueError("visual mission current stage is invalid")
        stage = draft.mission.stages[stage_index]
        attempts = [
            item
            for item in mission["stages"]
            if item.get("stageId") == stage.stageId and item.get("status") == "needs_input"
        ]
        if not attempts:
            raise ValueError("visual mission has no durable ambiguity result")
        source = max(attempts, key=lambda item: int(item.get("attempt", 0)))
        candidates = source.get("state", {}).get("ambiguityCandidates", [])
        if not any(
            isinstance(item, Mapping) and item.get("candidateId") == candidate_id
            for item in candidates
        ):
            raise ValueError("visual candidate is not part of the immutable ambiguity result")
        resolution_id = source.get("visualResolutionId")
        if not isinstance(resolution_id, str) or not resolution_id:
            raise ValueError("visual ambiguity result has no canonical resolution ID")
        state["candidateSelection"] = {
            "stageId": stage.stageId,
            "candidateId": candidate_id,
            "selectionSourceVisualResolutionId": resolution_id,
        }
        if not self.store.transition_visual_mission(
            mission_id,
            "approved",
            expected_status="needs_input",
            state=state,
            error=None,
        ):
            raise ValueError("visual mission state changed concurrently")
        selected = self.store.get_visual_mission(mission_id)
        assert selected is not None
        return self._public(selected)

    def _run_claimed(self, draft: VisualMissionPlanDraft, mission: Mapping[str, Any]) -> dict[str, Any]:
        mission_id = draft.mission.missionId
        state = dict(mission["state"])
        start_index = int(state.get("currentStageIndex", 0))
        for index, stage in enumerate(draft.mission.stages[start_index:], start=start_index):
            if self._cancelled(mission_id):
                raise ValueError("visual mission cancelled")
            prior_attempts = [
                item
                for item in mission.get("stages", [])
                if item.get("stageId") == stage.stageId
            ]
            selection = state.get("candidateSelection")
            selected_candidate_id = None
            selection_source_id = None
            if isinstance(selection, Mapping) and selection.get("stageId") == stage.stageId:
                selected_candidate_id = selection.get("candidateId")
                selection_source_id = selection.get("selectionSourceVisualResolutionId")
            attempt = max(
                (int(item.get("attempt", 0)) for item in prior_attempts),
                default=0,
            ) + 1
            self._transition(mission_id, "resolving", state)
            self.store.save_visual_stage(mission_id, stage.stageId, attempt, "resolving")
            stage_result = self.stage_client.resolve_stage(
                {
                    "mission": draft.mission.model_dump(mode="json"),
                    "stage": stage.model_dump(mode="json"),
                    "attempt": attempt,
                    **(
                        {
                            "candidateId": selected_candidate_id,
                            "selectionSourceVisualResolutionId": selection_source_id,
                        }
                        if isinstance(selected_candidate_id, str)
                        and isinstance(selection_source_id, str)
                        else {}
                    ),
                },
                lambda: self._cancelled(mission_id),
            )
            resolution_id = stage_result.get("visualResolutionId")
            if not isinstance(resolution_id, str) or not resolution_id:
                raise ValueError("vision-worker did not return visualResolutionId")
            canonical = self.vision_client.get_resolution(resolution_id)
            validate_visual_resolution_binding(
                draft,
                stage,
                canonical,
                expected_attempt=attempt,
                selected_candidate_id=(
                    selected_candidate_id
                    if isinstance(selected_candidate_id, str)
                    else None
                ),
                selection_source_visual_resolution_id=(
                    selection_source_id
                    if isinstance(selection_source_id, str)
                    else None
                ),
            )
            if canonical.get("status") == "needs_input":
                canonical_candidates = canonical.get("ambiguityCandidates", [])
                if (
                    canonical.get("missionId") != draft.mission.missionId
                    or canonical.get("stageId") != stage.stageId
                    or canonical.get("attempt", attempt) != attempt
                    or not isinstance(canonical_candidates, list)
                    or not canonical_candidates
                ):
                    raise ValueError("vision-worker returned an invalid ambiguity result")
                self.store.save_visual_stage(
                    mission_id,
                    stage.stageId,
                    attempt,
                    "needs_input",
                    visual_resolution_id=resolution_id,
                    state=self._resolution_state(canonical, stage_result),
                )
                state.pop("candidateSelection", None)
                self._transition(mission_id, "needs_input", state, "ambiguous_target")
                result = self.store.get_visual_mission(mission_id)
                assert result is not None
                return self._public(result)
            resolution_status = canonical.get("status")
            if resolution_status in {"rejected", "cancelled", "failed"}:
                rejection_reason = str(
                    canonical.get("rejectionReason")
                    or stage_result.get("explanation")
                    or resolution_status
                )
                self.store.save_visual_stage(
                    mission_id,
                    stage.stageId,
                    attempt,
                    str(resolution_status),
                    visual_resolution_id=resolution_id,
                    state=self._resolution_state(canonical, stage_result),
                )
                state.pop("candidateSelection", None)
                mission_status = (
                    "rejected"
                    if resolution_status == "rejected"
                    else "cancelled"
                    if resolution_status == "cancelled"
                    else "failed"
                )
                self._transition(mission_id, mission_status, state, rejection_reason)
                result = self.store.get_visual_mission(mission_id)
                assert result is not None
                return self._public(result)
            payload = validate_visual_resolution(
                draft,
                stage,
                canonical,
                expected_attempt=attempt,
                selected_candidate_id=(
                    selected_candidate_id
                    if isinstance(selected_candidate_id, str)
                    else None
                ),
                selection_source_visual_resolution_id=(
                    selection_source_id
                    if isinstance(selection_source_id, str)
                    else None
                ),
            )
            self.store.save_visual_stage(
                mission_id,
                stage.stageId,
                attempt,
                "resolved",
                visual_resolution_id=resolution_id,
                state={
                    "explanation": stage_result.get("explanation"),
                    "targetPhrase": canonical.get("targetPhrase"),
                    "landmarkPose": canonical.get("landmarkPose"),
                    "goalPose": canonical.get("goalPose"),
                    "safetyGates": canonical.get("safetyGates", []),
                    "ambiguityCandidates": canonical.get("ambiguityCandidates", []),
                    "evidence": canonical.get("evidence", []),
                    "source": canonical.get("source"),
                    "versions": canonical.get("versions"),
                },
            )
            if stage.goalPurpose == "observe":
                state["currentStageIndex"] = index + 1
                continue
            self._transition(mission_id, "dispatching", state)
            mapping = self._mapping()
            command_id = str(uuid4())
            request = CommandRequest(
                commandId=command_id,
                robotId=draft.robotId,
                source=CommandSource.AGENT,
                transport=mapping.transport,
                commandKey=NAVIGATION_COMMAND_KEY,
                commandVersion="1",
                payload=payload.model_dump(mode="json"),
                issuedAt=utc_now(),
                expiresAt=min(draft.expiresAt or utc_now() + timedelta(seconds=30), utc_now() + timedelta(seconds=30)),
                idempotencyKey=f"{mission_id}:{stage.stageId}:{attempt}:{resolution_id}",
                sessionId=mission_id,
                traceId=f"visual-mission:{mission_id}",
            )
            receipt = self.action_runtime.dispatch(request, self.definition, mapping)
            self.store.save_command(request, receipt)
            if receipt.status.value in {"accepted", "running"}:
                self._transition(mission_id, "navigating", state)
                receipt = self._await_terminal_receipt(
                    mission_id,
                    stage.stageId,
                    attempt,
                    resolution_id,
                    request,
                    receipt,
                    canonical,
                    stage_result,
                )
            self.store.save_visual_stage(
                mission_id,
                stage.stageId,
                attempt,
                receipt.status.value,
                visual_resolution_id=resolution_id,
                command_id=receipt.commandId,
                state={
                    **self._resolution_state(canonical, stage_result),
                    "receipt": receipt.model_dump(mode="json"),
                },
            )
            state.setdefault("receipts", []).append(receipt.model_dump(mode="json"))
            state["currentStageIndex"] = index + 1
            state.pop("candidateSelection", None)
            if receipt.status.value != "succeeded":
                current = self.store.get_visual_mission(mission_id)
                if current is not None and current["status"] == "cancelled":
                    return self._public(current)
                terminal = "failed" if receipt.status.value in {"failed", "rejected", "expired", "cancelled"} else "unknown"
                self._transition(mission_id, terminal, state, receipt.error)
                result = self.store.get_visual_mission(mission_id)
                assert result is not None
                return self._public(result)
            self._transition(mission_id, "verifying", state)
        self._transition(mission_id, "completed", state)
        result = self.store.get_visual_mission(mission_id)
        assert result is not None
        return self._public(result)

    def _await_terminal_receipt(
        self,
        mission_id: str,
        stage_id: str,
        attempt: int,
        resolution_id: str,
        request: CommandRequest,
        receipt: Any,
        canonical: Mapping[str, Any],
        stage_result: Mapping[str, Any],
    ) -> Any:
        """Reconcile a long-running Robot Agent command without replaying it."""

        latest = receipt
        while latest.status.value in {"accepted", "running"}:
            if self._cancelled(mission_id):
                try:
                    latest = self.action_runtime.cancel(request.commandId)
                except ValueError:
                    pass
                break
            time.sleep(self.command_poll_interval_seconds)
            latest = self.action_runtime.reconcile(request.commandId)
            self.store.save_command(request, latest)
            self.store.save_visual_stage(
                mission_id,
                stage_id,
                attempt,
                latest.status.value,
                visual_resolution_id=resolution_id,
                command_id=request.commandId,
                state={
                    **self._resolution_state(canonical, stage_result),
                    "receipt": latest.model_dump(mode="json"),
                },
            )
        return latest

    @staticmethod
    def _resolution_state(
        canonical: Mapping[str, Any], stage_result: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "status": canonical.get("status"),
            "executable": canonical.get("executable"),
            "explanation": stage_result.get("explanation"),
            "rejectionReason": canonical.get("rejectionReason"),
            "targetPhrase": canonical.get("targetPhrase"),
            "landmarkPose": canonical.get("landmarkPose"),
            "goalPose": canonical.get("goalPose"),
            "safetyGates": canonical.get("safetyGates", []),
            "ambiguityCandidates": canonical.get("ambiguityCandidates", []),
            "evidence": canonical.get("evidence", []),
            "source": canonical.get("source"),
            "versions": canonical.get("versions"),
            "selectedCandidateId": canonical.get("selectedCandidateId"),
            "selectionSourceVisualResolutionId": canonical.get(
                "selectionSourceVisualResolutionId"
            ),
        }

    def cancel(self, mission_id: str) -> dict[str, Any]:
        mission = self.store.get_visual_mission(mission_id)
        if mission is None:
            raise ValueError("visual mission not found")
        active = next(
            (
                stage
                for stage in reversed(mission["stages"])
                if stage.get("commandId") and stage["status"] in {"pending", "accepted", "running"}
            ),
            None,
        )
        if active is not None:
            try:
                receipt = self.action_runtime.cancel(active["commandId"])
                self.store.save_visual_stage(
                    mission_id,
                    active["stageId"],
                    active["attempt"],
                    receipt.status.value,
                    visual_resolution_id=active.get("visualResolutionId"),
                    command_id=active["commandId"],
                    state={"receipt": receipt.model_dump(mode="json")},
                )
            except ValueError:
                pass
        if not self.store.request_visual_mission_cancel(mission_id):
            raise ValueError(f"visual mission is not cancellable (status={mission['status']})")
        result = self.store.get_visual_mission(mission_id)
        assert result is not None
        return self._public(result)

    def _mapping(self) -> MappingVersion:
        if self.mapping_resolver is not None:
            return self.mapping_resolver(NAVIGATION_COMMAND_KEY)
        transport = Transport.ROBOT_AGENT if "robot_agent" in self.action_runtime.adapters else Transport.FAKE
        return MappingVersion(
            mappingId="visual-navigation-runtime",
            version="v1",
            commandKey=NAVIGATION_COMMAND_KEY,
            transport=transport,
            mapping={"targetId": "robot-1", "topic": "/navigation.navigate_to_pose"},
            payloadSchema=visual_navigation_command_schema(),
        )

    def _transition(
        self,
        mission_id: str,
        status: str,
        state: Mapping[str, Any],
        error: str | None = None,
    ) -> None:
        current = self.store.get_visual_mission(mission_id)
        if current is None or not self.store.transition_visual_mission(
            mission_id,
            status,
            expected_status=current["status"],
            state=state,
            error=error,
        ):
            raise ValueError("visual mission state changed concurrently")

    def _cancelled(self, mission_id: str) -> bool:
        mission = self.store.get_visual_mission(mission_id)
        return mission is None or mission["status"] == "cancelled" or bool(mission["state"].get("cancelRequested"))

    @classmethod
    def _claim_motion(cls, robot_id: str, mission_id: str) -> None:
        with cls._motion_lock:
            owner = cls._motion_owners.get(robot_id)
            if owner not in {None, mission_id}:
                raise ValueError(f"robot motion is already owned by visual mission {owner}")
            cls._motion_owners[robot_id] = mission_id

    @classmethod
    def _release_motion(cls, robot_id: str, mission_id: str) -> None:
        with cls._motion_lock:
            if cls._motion_owners.get(robot_id) == mission_id:
                cls._motion_owners.pop(robot_id, None)

    def _public(self, mission: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(mission)
        result["mode"] = self.mode.value
        result["releaseState"] = "code_complete_robot_agent_pending"
        result["motion"] = "simulated" if self.mode is VisualMissionMode.LAB_EXECUTE else "not_executed"
        return result
