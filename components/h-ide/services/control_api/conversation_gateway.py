"""Assistant V2 intent gateway and its non-escalating authority boundaries."""

from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any, Mapping

from packages.control_core.assistant_models import (
    ConversationGatewayResult,
    ConversationIntent,
    ConversationRoute,
    ConversationTurnRequest,
)

from .conversation_client import ConversationClient
from .conversation_client import ConversationRoutingUnavailable
from .mission_policy import PolicyActor
from .office_inspection import is_full_office_inspection
from .object_semantic_navigation import parse_object_semantic_request
from .office_semantic_navigation import parse_office_place_request


MissionDraftCreator = Callable[[Mapping[str, Any]], Mapping[str, Any]]
VisualObserver = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class ConversationGatewayError(RuntimeError):
    pass


class ConversationGateway:
    """Route one classified turn without granting the classifier execution power.

    The Mastra endpoint may classify and write a natural-language response. H-IDE
    derives every downstream call from that validated classification: observations
    are read-only VisualTasks, scene teaching only creates a pending memory state,
    and physical action can only create a reviewable Mission V2 draft.
    """

    def __init__(
        self,
        client: ConversationClient,
        *,
        mission_draft_creator: MissionDraftCreator | None = None,
        visual_observer: VisualObserver | None = None,
    ) -> None:
        self.client = client
        self.mission_draft_creator = mission_draft_creator
        self.visual_observer = visual_observer

    @staticmethod
    def _result(
        *,
        task_id: str,
        request: ConversationTurnRequest,
        intent: ConversationIntent,
        route: ConversationRoute,
        route_status: str,
        assistant_message: str,
        clarification: str | None = None,
        mission_draft_id: str | None = None,
        visual_result: Mapping[str, Any] | None = None,
    ) -> ConversationGatewayResult:
        evidence = []
        if isinstance(visual_result, Mapping):
            raw_evidence = visual_result.get("evidence")
            analysis_id = visual_result.get("analysisId")
            if isinstance(raw_evidence, list):
                for item in raw_evidence:
                    if not isinstance(item, Mapping):
                        continue
                    projected = dict(item)
                    sha256 = projected.get("sha256")
                    if (
                        isinstance(analysis_id, str)
                        and analysis_id
                        and isinstance(sha256, str)
                        and sha256
                    ):
                        projected["proxyUrl"] = (
                            f"/api/v1/assistant/turns/{task_id}/visual-evidence/"
                            f"{analysis_id}/{sha256}"
                        )
                    evidence.append(projected)
        return ConversationGatewayResult.model_validate(
            {
                "taskId": task_id,
                "conversationId": request.conversationId,
                "robotId": request.robotId,
                "intent": intent,
                "route": route,
                "routeStatus": route_status,
                "assistantMessage": assistant_message,
                "clarification": clarification,
                "missionDraftId": mission_draft_id,
                "visualResult": None if visual_result is None else dict(visual_result),
                "evidence": evidence,
            }
        )

    def handle(
        self,
        *,
        task_id: str,
        request: ConversationTurnRequest,
        requested_by: str,
        modality: str,
        mission_history: list[Mapping[str, Any]],
        mission_actor: PolicyActor | None = None,
    ) -> ConversationGatewayResult:
        # This reviewed product intent has a deterministic, SceneRef-bound
        # compiler downstream.  Routing it through a remote text classifier
        # adds no authority check and makes the fixed acceptance flow depend
        # on an unrelated model/network timeout.  The shortcut still creates
        # only a reviewable Mission draft; it cannot dispatch motion.
        if is_full_office_inspection(request.text):
            if request.robotId is None:
                return self._result(
                    task_id=task_id,
                    request=request,
                    intent=ConversationIntent.ROBOT_ACTION,
                    route=ConversationRoute.MISSION_REVIEW,
                    route_status="needs_robot",
                    assistant_message=(
                        "请先选择要巡检的机器人；未绑定机器人时不会创建行动方案。"
                    ),
                )
            return self._create_mission_review(
                task_id=task_id,
                request=request,
                requested_by=requested_by,
                modality=modality,
                mission_history=mission_history,
                mission_actor=mission_actor,
                assistant_message="办公室巡检方案已生成，等待你审核后执行。",
            )

        semantic_referents = (
            [item.model_dump(mode="json") for item in request.memoryContext.semanticReferents]
            if request.memoryContext is not None
            else []
        )
        known_place = parse_office_place_request(request.text)
        known_object = parse_object_semantic_request(
            request.text,
            semantic_referents=semantic_referents,
        )
        known_physical_action = known_place is not None or known_object is not None
        try:
            routed = self.client.route_turn(request)
        except ConversationRoutingUnavailable:
            if not known_physical_action:
                raise
            # A known product template can still produce a reviewable Mission
            # when the conversational classifier is temporarily unavailable;
            # geometry and execution remain behind the normal Mission gates.
            routed = None
        if known_physical_action and routed is None:
            if request.robotId is None:
                return self._result(
                    task_id=task_id,
                    request=request,
                    intent=ConversationIntent.ROBOT_ACTION,
                    route=ConversationRoute.MISSION_REVIEW,
                    route_status="needs_robot",
                    assistant_message="请先选择要使用的机器人；已识别为行动任务。",
                )
            return self._create_mission_review(
                task_id=task_id,
                request=request,
                requested_by=requested_by,
                modality=modality,
                mission_history=mission_history,
                mission_actor=mission_actor,
                assistant_message="行动方案已按当前场景模板生成，等待你审核。",
            )
        if routed.conversationId != request.conversationId:
            raise ConversationGatewayError("conversation router changed conversationId")
        if routed.robotId != request.robotId:
            raise ConversationGatewayError("conversation router changed robotId")

        # The classifier supplies language, not execution semantics.  Once a
        # known Office place/object action is recognized deterministically,
        # preserve the action boundary even if the model overweights words such
        # as “看/观察” or asks an irrelevant clarification.
        if known_physical_action:
            if request.robotId is None:
                return self._result(
                    task_id=task_id,
                    request=request,
                    intent=ConversationIntent.ROBOT_ACTION,
                    route=ConversationRoute.MISSION_REVIEW,
                    route_status="needs_robot",
                    assistant_message="请先选择要使用的机器人；已识别为行动任务。",
                )
            return self._create_mission_review(
                task_id=task_id,
                request=request,
                requested_by=requested_by,
                modality=modality,
                mission_history=mission_history,
                mission_actor=mission_actor,
                assistant_message=(
                    routed.assistantMessage
                    if routed.assistantMessage and routed.clarification is None
                    else "行动方案已按当前场景模板生成，等待你审核。"
                ),
            )

        route = {
            ConversationIntent.SOCIAL_CHAT: ConversationRoute.REPLY,
            ConversationIntent.GENERAL_QA: ConversationRoute.REPLY,
            ConversationIntent.ADMIN_DEV: ConversationRoute.REPLY,
            ConversationIntent.ROBOT_OBSERVE: ConversationRoute.VISUAL_OBSERVATION,
            ConversationIntent.SCENE_TEACH: ConversationRoute.SCENE_TEACHING,
            ConversationIntent.ROBOT_ACTION: ConversationRoute.MISSION_REVIEW,
        }[routed.intent]

        if routed.clarification is not None:
            return self._result(
                task_id=task_id,
                request=request,
                intent=routed.intent,
                route=route,
                route_status="needs_input",
                assistant_message=routed.assistantMessage,
                clarification=routed.clarification,
            )

        if routed.needsRobot and request.robotId is None:
            message = (
                "请先选择要使用的机器人，我不会在未绑定机器人时发起观察或行动。"
                if request.locale.casefold().startswith("zh")
                else "Select a robot first; no observation or action is started while unbound."
            )
            return self._result(
                task_id=task_id,
                request=request,
                intent=routed.intent,
                route=route,
                route_status="needs_robot",
                assistant_message=message,
            )

        if route is ConversationRoute.REPLY:
            return self._result(
                task_id=task_id,
                request=request,
                intent=routed.intent,
                route=route,
                route_status="completed",
                assistant_message=routed.assistantMessage,
            )

        if route is ConversationRoute.SCENE_TEACHING:
            # This is intentionally only a pending, non-motion teaching state.
            # A later scene-memory module may confirm and persist it, but this
            # route cannot dispatch Robot Agent, Nav2, or ActionRuntime calls.
            return self._result(
                task_id=task_id,
                request=request,
                intent=routed.intent,
                route=route,
                route_status="awaiting_confirmation",
                assistant_message=routed.assistantMessage,
            )

        if route is ConversationRoute.VISUAL_OBSERVATION:
            if self.visual_observer is None:
                return self._result(
                    task_id=task_id,
                    request=request,
                    intent=routed.intent,
                    route=route,
                    route_status="unavailable",
                    assistant_message="视觉观察服务暂不可用；机器人没有移动。",
                )
            try:
                visual = self.visual_observer(
                    {
                        "taskId": task_id,
                        "conversationId": request.conversationId,
                        "robotId": request.robotId,
                        "instruction": request.text,
                        "locale": request.locale,
                    }
                )
            except Exception:
                # Observation failure is terminal for this read-only turn. It
                # must never fall through into Mission planning or a motion
                # fallback merely because the camera/VLM path is unavailable.
                return self._result(
                    task_id=task_id,
                    request=request,
                    intent=routed.intent,
                    route=route,
                    route_status="unavailable",
                    assistant_message="视觉观察没有完成；机器人没有移动，请检查视觉服务后重试。",
                )
            visual_status = str(visual.get("status") or "failed")
            observation = visual.get("observationSummary")
            assistant_message = (
                str(observation).strip()
                if isinstance(observation, str) and observation.strip()
                else routed.assistantMessage
            )
            route_status = {
                "succeeded": "completed",
                "needs_input": "needs_input",
            }.get(visual_status, "unavailable")
            return self._result(
                task_id=task_id,
                request=request,
                intent=routed.intent,
                route=route,
                route_status=route_status,
                assistant_message=assistant_message,
                visual_result=visual,
            )

        if not routed.shouldCreateMission:
            raise ConversationGatewayError(
                "conversation router did not authorize Mission draft creation"
            )
        return self._create_mission_review(
            task_id=task_id,
            request=request,
            requested_by=requested_by,
            modality=modality,
            mission_history=mission_history,
            mission_actor=mission_actor,
            assistant_message=routed.assistantMessage,
        )

    def _create_mission_review(
        self,
        *,
        task_id: str,
        request: ConversationTurnRequest,
        requested_by: str,
        modality: str,
        mission_history: list[Mapping[str, Any]],
        mission_actor: PolicyActor | None,
        assistant_message: str,
    ) -> ConversationGatewayResult:
        if self.mission_draft_creator is None:
            raise ConversationGatewayError("Mission V2 draft service is unavailable")
        if request.robotId is None:
            raise ConversationGatewayError("Mission V2 requires a bound robot")
        brief = {
            "conversationId": request.conversationId,
            "sourceTaskId": task_id,
            "robotId": request.robotId,
            "requestedBy": requested_by,
            "objective": request.text,
            "locale": request.locale,
            "modality": modality,
            "history": [dict(item) for item in mission_history],
        }
        creator_parameters = inspect.signature(
            self.mission_draft_creator
        ).parameters.values()
        accepts_actor = any(
            item.name == "actor" or item.kind is inspect.Parameter.VAR_KEYWORD
            for item in creator_parameters
        )
        draft = (
            self.mission_draft_creator(brief, actor=mission_actor)
            if mission_actor is not None and accepts_actor
            else self.mission_draft_creator(brief)
        )
        draft_id = draft.get("draftId")
        if not isinstance(draft_id, str) or not draft_id:
            raise ConversationGatewayError("Mission V2 returned no durable draftId")
        return self._result(
            task_id=task_id,
            request=request,
            intent=ConversationIntent.ROBOT_ACTION,
            route=ConversationRoute.MISSION_REVIEW,
            route_status="review",
            assistant_message=assistant_message,
            mission_draft_id=draft_id,
        )


__all__ = [
    "ConversationGateway",
    "ConversationGatewayError",
    "MissionDraftCreator",
    "VisualObserver",
]
