"""Conversation orchestration for planning-only Supervisor–Worker turns."""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import threading
import uuid
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as JsonSchemaValidationError
from pydantic import TypeAdapter, ValidationError

from packages.scene_core import (
    NAVIGATION_COMMAND_KEY,
    NavigationPayload,
    Point3D,
    SceneRef,
    validate_navigation_payload,
)
from packages.control_core.assistant_models import (
    AssistantTurnEvent,
    ConversationTurnRequest,
    DraftReadyResult,
    LogicalCommand,
    PlanningEnvelope,
    PlanningTurnResult,
    TaskRequest,
)
from packages.control_core.models import CapabilitySnapshot, CommandDefinition, PlanDraft, RobotState, utc_now
from packages.visual_mission_core import VisualMissionPlanDraft
from packages.control_core.validation import non_empty_id

from .planning_client import PlanningClient
from .conversation_gateway import ConversationGateway
from .conversation_memory import ConversationMemoryContextBuilder
from .scene_client import SceneContextClient, SceneContextUnavailable
from .store import (
    AssistantConversationConflict,
    AssistantTurnConflict,
    SQLiteStore,
    _jsonable,
)


_TURN_RESULT_ADAPTER = TypeAdapter(PlanningTurnResult)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class AssistantNotFound(ValueError):
    pass


def _conversation_title(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("title must be a string or null")
    title = value.strip()
    if not title:
        raise ValueError("title cannot be blank")
    if len(title) > 120:
        raise ValueError("title is too long")
    return title


class AssistantService:
    def __init__(
        self,
        store: SQLiteStore,
        definitions: Mapping[tuple[str, str], CommandDefinition],
        planning_client: PlanningClient,
        scene_context_client: SceneContextClient,
        *,
        draft_ttl_seconds: float = 900.0,
        conversation_gateway: ConversationGateway | None = None,
    ):
        self.store = store
        self.definitions = definitions
        self.planning_client = planning_client
        self.scene_context_client = scene_context_client
        self.draft_ttl_seconds = draft_ttl_seconds
        self.conversation_gateway = conversation_gateway
        self.conversation_memory = ConversationMemoryContextBuilder(store)
        self._threads: dict[str, threading.Thread] = {}
        self._threads_lock = threading.Lock()
        self.store.recover_active_assistant_turns()

    def create_conversation(self, body: Mapping[str, Any]) -> dict[str, Any]:
        raw_robot_id = body.get("robotId")
        robot_id = None if raw_robot_id is None else non_empty_id(raw_robot_id, "robotId")
        requested_by = non_empty_id(body.get("requestedBy", "operator"), "requestedBy")
        locale = non_empty_id(body.get("locale", "zh-CN"), "locale")
        raw_summary = body.get("summary")
        summary = None if raw_summary is None else non_empty_id(raw_summary, "summary")
        title = _conversation_title(body.get("title"))
        if len(locale) > 35:
            raise ValueError("locale is too long")
        if summary is not None and len(summary) > 12000:
            raise ValueError("summary is too long")
        if robot_id is not None and self.store.get_robot(robot_id) is None:
            raise AssistantNotFound("robot not found")
        return self.store.create_assistant_conversation(
            _new_id("conversation"), robot_id, requested_by, locale, summary, title
        )

    def list_conversations(
        self,
        *,
        include_archived: bool = False,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        return self.store.list_assistant_conversations(
            include_archived=include_archived,
            include_deleted=include_deleted,
        )

    def update_conversation(
        self,
        conversation_id: str,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        supported = {"title", "archived", "deleted"}
        unknown = set(body) - supported
        if unknown:
            raise ValueError(
                "conversation patch contains unsupported fields: "
                + ", ".join(sorted(str(item) for item in unknown))
            )
        if not body:
            raise ValueError("conversation patch must change at least one field")
        kwargs: dict[str, Any] = {}
        if "title" in body:
            kwargs["title"] = _conversation_title(body.get("title"))
        for field in ("archived", "deleted"):
            if field not in body:
                continue
            value = body[field]
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be a boolean")
            kwargs[field] = value
        updated = self.store.update_assistant_conversation(conversation_id, **kwargs)
        if updated is None:
            raise AssistantNotFound("assistant conversation not found")
        return updated

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        updated = self.store.update_assistant_conversation(
            conversation_id,
            deleted=True,
        )
        if updated is None:
            raise AssistantNotFound("assistant conversation not found")
        return updated

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.store.get_assistant_conversation(conversation_id)
        if conversation is None:
            raise AssistantNotFound("assistant conversation not found")
        conversation["messages"] = self.store.list_assistant_messages(conversation_id)
        conversation["turns"] = self.store.list_assistant_turns(conversation_id)
        conversation["drafts"] = self.store.list_assistant_conversation_drafts(conversation_id)
        return conversation

    def submit_message(
        self,
        conversation_id: str,
        body: Mapping[str, Any],
        *,
        revision_of_draft_id: str | None = None,
    ) -> dict[str, Any]:
        conversation = self.store.get_assistant_conversation(conversation_id)
        if conversation is None:
            raise AssistantNotFound("assistant conversation not found")
        if conversation.get("deleted"):
            raise AssistantConversationConflict(
                "deleted conversation must be restored before accepting messages"
            )
        if conversation.get("archived"):
            raise AssistantConversationConflict(
                "archived conversation must be restored before accepting messages"
            )
        modality = body.get("modality", "text")
        transcription_id = body.get("transcriptionId")
        if modality == "voice":
            if not isinstance(transcription_id, str):
                raise ValueError("voice messages require transcriptionId")
            if self.store.get_transcription(transcription_id) is None:
                raise ValueError("voice message transcriptionId was not issued by this control plane")
        elif transcription_id is not None:
            raise ValueError("transcriptionId is only valid for voice messages")
        text = body.get("text")
        if revision_of_draft_id is not None and isinstance(text, str) and len(text) > 2000:
            raise ValueError("revision feedback cannot exceed 2000 characters")
        if self.conversation_gateway is not None and revision_of_draft_id is None:
            route_request = ConversationTurnRequest.model_validate(
                {
                    "conversationId": conversation_id,
                    "robotId": conversation.get("robotId"),
                    "locale": body.get("locale", conversation["locale"]),
                    "text": text,
                    "history": [],
                }
            )
            task_id = _new_id("task")
            stored_request = {
                **route_request.model_dump(mode="json", by_alias=False, exclude_none=True),
                "taskId": task_id,
                "requestedBy": conversation["requestedBy"],
                "modality": modality,
                "transcriptionId": transcription_id,
            }
            self.store.create_assistant_turn(
                stored_request,
                message_id=_new_id("message"),
            )
            thread = threading.Thread(
                target=self._execute_conversation_turn,
                args=(task_id,),
                name=f"assistant-v2-{task_id[-8:]}",
                daemon=True,
            )
            with self._threads_lock:
                self._threads[task_id] = thread
            thread.start()
            return {
                "taskId": task_id,
                "conversationId": conversation_id,
                "status": "queued",
                "eventsUrl": f"/api/v1/assistant/turns/{task_id}/events",
            }
        selection = body.get("sceneSelection")
        expected_ref = None
        selected_point = None
        if selection is not None:
            if not isinstance(selection, Mapping):
                raise ValueError("sceneSelection must be an object")
            unknown = set(selection) - {"sceneRef", "selectedPoint"}
            if unknown:
                raise ValueError("sceneSelection contains unsupported fields")
            expected_ref = SceneRef.model_validate(selection.get("sceneRef"))
            if selection.get("selectedPoint") is not None:
                selected_point = Point3D.model_validate(selection["selectedPoint"])
        robot_payload = self.store.get_robot(conversation["robotId"])
        if robot_payload is None:
            raise AssistantNotFound("robot planning context is unavailable")
        robot_state = RobotState.model_validate(robot_payload)
        scene_context = self.scene_context_client.get_context(
            conversation["robotId"],
            current_pose=robot_state.pose,
            expected_ref=expected_ref,
            selected_point=selected_point,
        )
        request = TaskRequest.model_validate(
            {
                "schemaVersion": "1.0",
                "taskId": _new_id("task"),
                "conversationId": conversation_id,
                "robotId": conversation["robotId"],
                "requestedBy": conversation["requestedBy"],
                "modality": modality,
                "text": text,
                "locale": body.get("locale", conversation["locale"]),
                "transcriptionId": transcription_id,
                "revisionOfDraftId": revision_of_draft_id,
                "sceneContext": scene_context,
            }
        )
        self.store.create_assistant_turn(
            request.model_dump(mode="json", by_alias=False), message_id=_new_id("message")
        )
        thread = threading.Thread(
            target=self._execute_turn,
            args=(request.taskId,),
            name=f"assistant-turn-{request.taskId[-8:]}",
            daemon=True,
        )
        with self._threads_lock:
            self._threads[request.taskId] = thread
        thread.start()
        return {"taskId": request.taskId, "conversationId": conversation_id, "status": "queued"}

    def get_turn(self, task_id: str) -> dict[str, Any]:
        turn = self.store.get_assistant_turn(task_id)
        if turn is None:
            raise AssistantNotFound("assistant turn not found")
        draft = self.store.get_assistant_draft_for_task(task_id)
        if draft is not None:
            turn["draft"] = draft
        return turn

    def get_turn_events(self, task_id: str, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        if self.store.get_assistant_turn(task_id) is None:
            raise AssistantNotFound("assistant turn not found")
        return [
            AssistantTurnEvent.model_validate(item).model_dump(mode="json", by_alias=False)
            for item in self.store.list_assistant_turn_events(
                task_id,
                after_sequence=after_sequence,
            )
        ]

    def cancel_turn(self, task_id: str) -> dict[str, Any]:
        turn = self.store.cancel_assistant_turn(task_id)
        if turn is None:
            raise AssistantNotFound("assistant turn not found")
        request = turn.get("request")
        is_v2 = (
            isinstance(request, Mapping)
            and request.get("schemaVersion") == "h-ide.conversation-turn/2"
        )
        if turn["status"] == "cancelled" and not is_v2:
            self.planning_client.cancel_turn(task_id)
        return turn

    def _execute_conversation_turn(self, task_id: str) -> None:
        """Run routing, then cross exactly one downstream authority boundary."""

        try:
            if not self.store.mark_assistant_turn_running(task_id):
                return
            turn = self.store.get_assistant_turn(task_id)
            if turn is None:
                raise AssistantNotFound("assistant turn not found")
            raw_request = turn.get("request")
            if not isinstance(raw_request, Mapping):
                raise ValueError("Assistant V2 turn request is unavailable")
            conversation_id = str(turn["conversationId"])
            conversation = self.store.get_assistant_conversation(conversation_id)
            if conversation is None:
                raise AssistantNotFound("assistant conversation not found")
            prepared_memory = self.conversation_memory.prepare(
                conversation_id,
                str(raw_request.get("text") or ""),
                exclude_task_id=task_id,
            )
            route_request = ConversationTurnRequest.model_validate(
                {
                    "conversationId": conversation_id,
                    "robotId": conversation.get("robotId"),
                    "locale": raw_request.get("locale", conversation["locale"]),
                    "text": raw_request.get("text"),
                    "history": list(prepared_memory.history),
                    "memoryContext": prepared_memory.memory_context,
                }
            )
            gateway = self.conversation_gateway
            if gateway is None:
                raise RuntimeError("Assistant V2 conversation gateway is unavailable")
            result = gateway.handle(
                task_id=task_id,
                request=route_request,
                requested_by=str(conversation["requestedBy"]),
                modality=str(raw_request.get("modality") or "text"),
                mission_history=list(prepared_memory.mission_history),
            )
            if self.store.is_assistant_turn_cancelled(task_id):
                return
            if result.missionDraftId is not None:
                binding = self.store.get_resource_owner(
                    "assistant_conversation", conversation_id
                )
                if binding is not None:
                    self.store.bind_resource_owner(
                        "mission_draft",
                        result.missionDraftId,
                        owner_user_id=str(binding["ownerUserId"]),
                        site_id=str(binding["siteId"]),
                    )
            terminal_status = {
                "review": "draft_ready",
                "needs_input": "needs_input",
                "needs_robot": "needs_input",
                "unavailable": "cannot_plan",
            }.get(result.routeStatus, "completed")
            conversation_status = {
                "review": "review",
                "needs_input": "awaiting_user",
                "needs_robot": "awaiting_user",
                "awaiting_confirmation": "teaching",
            }.get(result.routeStatus, "idle")
            self.store.complete_assistant_turn(
                task_id,
                status=terminal_status,
                result=result.model_dump(mode="json", by_alias=False, exclude_none=True),
                assistant_message=result.assistantMessage,
                message_id=_new_id("message"),
                conversation_status=conversation_status,
            )
        except Exception as exc:
            self.store.fail_assistant_turn(task_id, str(exc)[:2000] or type(exc).__name__)
        finally:
            with self._threads_lock:
                self._threads.pop(task_id, None)

    def reject_draft(self, draft_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        reason = non_empty_id(body.get("reason", "operator rejected the plan"), "reason")
        if len(reason) > 2000:
            raise ValueError("rejection reason cannot exceed 2000 characters")
        draft = self.store.reject_assistant_draft(
            draft_id, reason, message_id=_new_id("message")
        )
        if draft is None:
            raise AssistantNotFound("assistant draft not found")
        return draft

    def revise_draft(self, draft_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
        metadata = self.store.get_assistant_draft_review(draft_id)
        if metadata is None:
            raise AssistantNotFound("assistant draft is not owned by a conversation")
        feedback = non_empty_id(body.get("feedback"), "feedback")
        return self.submit_message(
            metadata["conversationId"],
            {"text": feedback, "modality": "text"},
            revision_of_draft_id=draft_id,
        )

    def _logical_catalog(self) -> list[LogicalCommand]:
        result = []
        for (_, _), definition in sorted(
            self.definitions.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            result.append(
                LogicalCommand(
                    commandKey=definition.commandKey,
                    version=definition.commandVersion,
                    kind=definition.kind,
                    description=definition.description,
                    # Planning sees the PCD target-candidate contract. The
                    # Robot Agent execution contract also carries fresh
                    # observation bindings and is intentionally not an LLM
                    # output format.
                    payloadSchema=(
                        NavigationPayload.model_json_schema(by_alias=True)
                        if definition.commandKey == NAVIGATION_COMMAND_KEY
                        else definition.payloadSchema
                    ),
                )
            )
        return result

    @staticmethod
    def _context_version(
        robot: Mapping[str, Any],
        capability: Mapping[str, Any],
        catalog: list[LogicalCommand],
        scene_context,
    ) -> str:
        data = {
            "robot": robot,
            "capability": capability,
            "catalog": [item.model_dump(mode="json", by_alias=False) for item in catalog],
            "sceneContext": scene_context,
        }
        canonical = json.dumps(_jsonable(data), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"ctx-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:20]}"

    def _build_envelope(self, task_id: str) -> PlanningEnvelope:
        turn = self.store.get_assistant_turn(task_id)
        if turn is None:
            raise AssistantNotFound("assistant turn not found")
        request = TaskRequest.model_validate(turn["request"])
        robot_payload = self.store.get_robot(request.robotId)
        capability_payload = self.store.get_capabilities(request.robotId)
        if robot_payload is None or capability_payload is None:
            raise AssistantNotFound("robot planning context is unavailable")
        messages = self.store.list_assistant_messages(request.conversationId)[-20:]
        history = [
            {"role": item["role"], "content": item["content"], "createdAt": item["createdAt"]}
            for item in messages
        ]
        catalog = self._logical_catalog()
        return PlanningEnvelope(
            taskRequest=request,
            contextVersion=self._context_version(
                robot_payload,
                capability_payload,
                catalog,
                request.sceneContext,
            ),
            history=history,
            robotState=RobotState.model_validate(robot_payload),
            capabilitySnapshot=CapabilitySnapshot.model_validate(capability_payload),
            commandCatalog=catalog,
            sceneContext=request.sceneContext,
        )

    @staticmethod
    def _validate_payload(definition: CommandDefinition, payload: Mapping[str, Any]) -> None:
        try:
            validator = Draft202012Validator(definition.payloadSchema)
            validator.validate(dict(payload))
        except SchemaError as exc:
            raise ValueError(f"command schema is invalid for {definition.commandKey}: {exc.message}") from exc
        except JsonSchemaValidationError as exc:
            raise ValueError(f"invalid payload for {definition.commandKey}: {exc.message}") from exc

    def _validate_proposal(self, result: DraftReadyResult, envelope: PlanningEnvelope) -> None:
        proposal = result.proposal
        if proposal.robotId != envelope.taskRequest.robotId:
            raise ValueError("planning proposal robotId does not match the task")
        if proposal.contextVersion != envelope.contextVersion:
            raise ValueError("planning proposal was produced from stale robot context")
        if result.visualMission is not None:
            mission = result.visualMission
            current_boot_id = envelope.robotState.details.get("robotBootId")
            if not isinstance(current_boot_id, str) or not current_boot_id:
                raise ValueError("current robotBootId is unavailable")
            current_epoch_id = envelope.robotState.details.get("localizationEpochId")
            if not isinstance(current_epoch_id, str) or not current_epoch_id:
                raise ValueError("current localizationEpochId is unavailable")
            if mission.robotBootId != current_boot_id:
                raise ValueError("visual mission robotBootId does not match current robot state")
            if mission.localizationEpochId != current_epoch_id:
                raise ValueError(
                    "visual mission localizationEpochId does not match current robot state"
                )
            if mission.sceneRef is not None:
                if envelope.sceneContext is None or mission.sceneRef != envelope.sceneContext.sceneRef:
                    raise ValueError("visual mission sceneRef does not match planning context")
            return
        available = set(envelope.capabilitySnapshot.capabilities)
        for step in proposal.steps:
            definition = self.definitions.get((step.commandKey, "1"))
            if definition is None:
                raise ValueError(f"planning proposal uses unknown command: {step.commandKey}")
            if step.commandKey not in available:
                raise ValueError(f"robot capability is missing: {step.commandKey}")
            if step.commandKey == NAVIGATION_COMMAND_KEY:
                if envelope.sceneContext is None:
                    raise ValueError("navigation plan requires a published robot scene binding")
                validate_navigation_payload(envelope.sceneContext, step.payload)
            else:
                self._validate_payload(definition, step.payload)

    def _spatial_review(self, result: DraftReadyResult, envelope: PlanningEnvelope) -> tuple[list[dict], list[str]]:
        """Collect non-authoritative map evidence without changing the proposed plan."""

        context = envelope.sceneContext
        navigation_steps = [
            step for step in result.proposal.steps if step.commandKey == NAVIGATION_COMMAND_KEY
        ]
        if not navigation_steps or context is None:
            return [], []
        if context.mapSummary is None or not context.mapSummary.verified:
            return [], ["正式场景地图尚未就绪，本次导航目标未进行点云占用校验"]

        checks: list[dict] = []
        warnings: list[str] = []
        target_checker = getattr(self.scene_context_client, "check_target", None)
        if target_checker is None:
            return [], ["场景客户端尚未提供地图目标校验，请人工复核导航目标"]
        for step in navigation_steps:
            navigation = validate_navigation_payload(context, step.payload)
            point = Point3D(
                x=navigation.goalPose.x,
                y=navigation.goalPose.y,
                z=navigation.goalPose.z,
            )
            try:
                check = target_checker(context.sceneRef, point)
            except SceneContextUnavailable:
                warnings.append(f"步骤 {step.stepId} 的地图目标校验暂不可用，请人工复核")
                continue
            if check is None:
                warnings.append(f"步骤 {step.stepId} 没有可用的正式栅格证据，请人工复核")
                continue
            checks.append({
                "stepId": step.stepId,
                "goalPose": navigation.goalPose.model_dump(mode="json"),
                "check": check.model_dump(mode="json"),
            })
            if check.classification != "observed_free":
                label = {
                    "occupied": "净空范围内存在障碍证据",
                    "unknown": "目标区域观测不足",
                    "outside": "目标位于派生地图之外",
                }[check.classification]
                warnings.append(f"步骤 {step.stepId}：{label}；地图结果只作为审核提示")
            warnings.extend(f"步骤 {step.stepId}：{warning}" for warning in check.warnings)
        return checks, list(dict.fromkeys(warnings))

    def _execute_turn(self, task_id: str) -> None:
        try:
            if not self.store.mark_assistant_turn_running(task_id):
                return
            envelope = self._build_envelope(task_id)
            raw_result = self.planning_client.run_turn(
                envelope, lambda: self.store.is_assistant_turn_cancelled(task_id)
            )
            if raw_result.get("status") == "cancelled":
                self.store.cancel_assistant_turn(task_id)
                return
            result = _TURN_RESULT_ADAPTER.validate_python(raw_result)
            result_payload = result.model_dump(mode="json", by_alias=False)
            draft_payload = None
            review = None
            conversation_status = "awaiting_user"
            if isinstance(result, DraftReadyResult):
                self._validate_proposal(result, envelope)
                now = utc_now()
                draft_id = _new_id("draft")
                if result.visualMission is not None:
                    mission = result.visualMission.model_copy(
                        update={
                            "missionId": draft_id,
                            "reviewExpiresAt": now + timedelta(seconds=self.draft_ttl_seconds),
                        }
                    )
                    plan = VisualMissionPlanDraft(
                        planId=draft_id,
                        robotId=result.proposal.robotId,
                        mission=mission,
                        createdAt=now,
                        expiresAt=now + timedelta(seconds=self.draft_ttl_seconds),
                    )
                else:
                    plan = PlanDraft(
                        planId=draft_id,
                        robotId=result.proposal.robotId,
                        steps=[
                            {
                                "stepId": step.stepId,
                                "commandKey": step.commandKey,
                                "payload": step.payload,
                                "dependsOn": step.dependsOn,
                            }
                            for step in result.proposal.steps
                        ],
                        createdAt=now,
                        expiresAt=now + timedelta(seconds=self.draft_ttl_seconds),
                    )
                draft_payload = {
                    "draftId": draft_id,
                    "plan": plan.model_dump(mode="json", by_alias=False),
                }
                review = result.proposal.model_dump(mode="json", by_alias=False)
                spatial_checks, spatial_warnings = self._spatial_review(result, envelope)
                review["spatialChecks"] = spatial_checks
                review["warnings"] = list(dict.fromkeys([*review["warnings"], *spatial_warnings]))
                result_payload = {
                    "status": result.status,
                    "assistantMessage": result.assistantMessage,
                    "draftId": draft_id,
                }
                conversation_status = "review"
            self.store.complete_assistant_turn(
                task_id,
                status=result.status,
                result=result_payload,
                assistant_message=result.assistantMessage,
                message_id=_new_id("message"),
                conversation_status=conversation_status,
                draft=draft_payload,
                review=review,
            )
        except (ValidationError, ValueError, AssistantTurnConflict) as exc:
            self.store.fail_assistant_turn(task_id, f"planning result rejected: {exc}")
        except Exception as exc:
            self.store.fail_assistant_turn(task_id, str(exc)[:2000] or type(exc).__name__)
        finally:
            with self._threads_lock:
                self._threads.pop(task_id, None)

    def close(self) -> None:
        with self._threads_lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join(timeout=0.5)


__all__ = ["AssistantNotFound", "AssistantService"]
