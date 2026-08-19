"""Deterministic, safety-labelled simulation runtime.

The runtime models feedback and state transitions so H-IDE can be tested
without a simulator installation.  Its bounded geometric detour is only a
demo aid; it deliberately does not claim to be a production path planner.
Gazebo/Nav2 remains the integration backend for real navigation semantics.
"""

from __future__ import annotations

from datetime import datetime
import math
import shutil
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4

from packages.scene_core import Point2D, PoseMap
from packages.control_core.validation import utc_now

from .models import (
    NavigationGoalInput,
    SimulationBackend,
    SimulationEvent,
    SimulationEventInput,
    SimulationEventType,
    SimulationLandmark,
    SimulationReceipt,
    SimulationReview,
    SimulationSession,
    SimulationSessionCreate,
    SimulationSnapshot,
    SimulationStatus,
    SimulationStepInput,
    distance_xy,
)
from .storage import SimulationStore
from .mujoco_client import MujocoSonicClient, MujocoSonicUnavailable


class SimulationNotFound(LookupError):
    pass


class SimulationConflict(ValueError):
    pass


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _line_circle_distance(start: PoseMap, end: PoseMap, center: Point2D) -> float:
    """Distance from a 2D line segment to a circular obstacle."""

    dx = end.x - start.x
    dy = end.y - start.y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(center.x - start.x, center.y - start.y)
    t = ((center.x - start.x) * dx + (center.y - start.y) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    nearest_x = start.x + t * dx
    nearest_y = start.y + t * dy
    return math.hypot(center.x - nearest_x, center.y - nearest_y)


def _segment_clear(
    start: PoseMap,
    end: PoseMap,
    obstacles: list[Any],
    *,
    clearance_m: float = 0.08,
) -> bool:
    return all(
        _line_circle_distance(start, end, obstacle.center)
        > obstacle.radiusM + clearance_m
        for obstacle in obstacles
        if obstacle.static
    )


def _detour_waypoints(
    start: PoseMap,
    goal: PoseMap,
    obstacles: list[Any],
) -> list[PoseMap] | None:
    """Build a tiny deterministic detour around static circular fixtures.

    This is intentionally bounded and geometric: it is a visual/demo aid for
    the contract simulator, not a replacement for Nav2's costmaps or planner.
    """

    current = start
    waypoints: list[PoseMap] = []
    remaining = [item for item in obstacles if item.static]
    for _ in range(4):
        blockers = [
            obstacle
            for obstacle in remaining
            if _line_circle_distance(current, goal, obstacle.center)
            <= obstacle.radiusM + 0.08
        ]
        if not blockers:
            return [*waypoints, goal]
        obstacle = min(
            blockers,
            key=lambda item: _line_circle_distance(current, goal, item.center),
        )
        radius = obstacle.radiusM + 0.22
        candidates: list[PoseMap] = []
        for index in range(16):
            angle = math.tau * index / 16.0
            candidate = PoseMap(
                frameId="map",
                x=obstacle.center.x + math.cos(angle) * radius,
                y=obstacle.center.y + math.sin(angle) * radius,
                z=goal.z,
                yaw=0.0,
            )
            # Set a useful heading without putting geometry into the planner
            # intent; this remains a simulator-internal waypoint.
            candidate = candidate.model_copy(
                update={"yaw": math.atan2(goal.y - candidate.y, goal.x - candidate.x)}
            )
            if _segment_clear(current, candidate, obstacles) and _segment_clear(candidate, goal, obstacles):
                candidates.append(candidate)
        if not candidates:
            return None
        chosen = min(
            candidates,
            key=lambda item: distance_xy(current, item) + distance_xy(item, goal),
        )
        waypoints.append(chosen)
        current = chosen
        remaining = [item for item in remaining if item.obstacleId != obstacle.obstacleId]
    return None


class SimulationRuntime:
    """Own simulation sessions and their append-only event streams."""

    def __init__(
        self,
        store: SimulationStore | None = None,
        *,
        db_path: str = ":memory:",
        mujoco_client: MujocoSonicClient | None = None,
    ) -> None:
        self.store = store or SimulationStore(db_path)
        self._owns_store = store is None
        self._lock = threading.RLock()
        self._snapshot_sink: Callable[[SimulationSession, SimulationSnapshot], None] | None = None
        self._event_sink: Callable[[SimulationSession, SimulationSnapshot, SimulationEvent], None] | None = None
        # The client is read-only. A separate navigation dispatcher may be
        # attached by the service composition root once the simulation Robot
        # Agent gateway is configured.
        self.mujoco_client = mujoco_client or MujocoSonicClient()
        self._navigation_dispatcher: Callable[[SimulationSession, NavigationGoalInput], Mapping[str, Any]] | None = None
        self._navigation_status_provider: Callable[[str], Mapping[str, Any]] | None = None
        self._identity_provider: Callable[[str], Mapping[str, Any]] | None = None

    def set_navigation_dispatcher(
        self,
        dispatcher: Callable[[SimulationSession, NavigationGoalInput], Mapping[str, Any]] | None,
    ) -> None:
        """Attach the simulation Robot Agent/Nav2 dispatcher.

        Keeping this callback outside the simulation package prevents the
        simulator from becoming a second command transport. It is normally
        supplied by ``services.control_api.main`` and points at the isolated
        Robot Agent Gateway.
        """

        with self._lock:
            self._navigation_dispatcher = dispatcher

    def set_identity_provider(
        self,
        provider: Callable[[str], Mapping[str, Any]] | None,
    ) -> None:
        """Attach a provider for the live simulation Robot Agent identity."""

        with self._lock:
            self._identity_provider = provider

    def set_navigation_status_provider(
        self,
        provider: Callable[[str], Mapping[str, Any]] | None,
    ) -> None:
        """Attach a read-only Robot Agent command-status provider for the UI."""

        with self._lock:
            self._navigation_status_provider = provider

    def set_snapshot_sink(
        self,
        sink: Callable[[SimulationSession, SimulationSnapshot], None] | None,
    ) -> None:
        """Project simulation pose into an owning control-plane snapshot.

        The sink is deliberately one-way: simulation state may update the
        workstation's observation view, but it cannot dispatch a physical
        command or mutate Mission authorization.
        """

        with self._lock:
            self._snapshot_sink = sink

    def set_event_sink(
        self,
        sink: Callable[[SimulationSession, SimulationSnapshot, SimulationEvent], None] | None,
    ) -> None:
        """Register a best-effort consumer for mission-facing simulation events.

        The durable simulation event is committed before this callback runs.
        A bridge may therefore create a Mission review proposal, but a bridge
        failure can never turn into a simulated or physical command.
        """

        with self._lock:
            self._event_sink = sink

    @staticmethod
    def _deterministic_office_obstacles():
        from .models import SimulationObstacle

        return [
            SimulationObstacle(
                obstacleId="office-column-1",
                label="固定柱",
                center=Point2D(x=2.0, y=0.8),
                radiusM=0.38,
            ),
            SimulationObstacle(
                obstacleId="office-table-1",
                label="桌子",
                center=Point2D(x=3.8, y=-0.65),
                radiusM=0.55,
            ),
        ]

    @staticmethod
    def _deterministic_office_landmarks():
        return [
            SimulationLandmark(
                landmarkId="kitchen",
                label="厨房",
                aliases=["茶水间", "饮水区"],
                pose=PoseMap(x=4.5, y=1.3, z=0.0, yaw=math.pi / 2),
            ),
            SimulationLandmark(
                landmarkId="meeting-room",
                label="会议室",
                aliases=["会议室入口"],
                pose=PoseMap(x=4.5, y=-1.3, z=0.0, yaw=-math.pi / 2),
            ),
            SimulationLandmark(
                landmarkId="desk",
                label="桌子",
                aliases=["办公桌"],
                pose=PoseMap(x=5.1, y=1.3, z=0.0, yaw=math.pi),
            ),
        ]

    def _get(self, session_id: str) -> tuple[SimulationSession, SimulationSnapshot]:
        value = self.store.get_session(session_id)
        if value is None:
            raise SimulationNotFound("simulation session not found")
        return value

    @staticmethod
    def _require_product_session(session: SimulationSession) -> None:
        if session.mode.value != "qwen":
            raise SimulationConflict(
                "historical simulator-truth session is read-only and cannot execute"
            )

    @staticmethod
    def _result(session: SimulationSession, snapshot: SimulationSnapshot) -> dict[str, Any]:
        # Historical deterministic fixtures may keep their authored semantic
        # map. A product MuJoCo session must not project those coordinates into
        # the UI: doing so would let a label such as "厨房" bypass VLM grounding.
        # Nav2/Robot Agent command truth and the observed robot trajectory stay
        # visible, while semantic landmarks/obstacles and manufactured
        # waypoints remain absent.
        if (
            session.mode.value == "qwen"
            and session.backend is SimulationBackend.MUJOCO_SONIC
        ):
            snapshot = snapshot.model_copy(
                update={"obstacles": [], "landmarks": [], "plannedWaypoints": []}
            )
        result = {
            "session": session.model_dump(mode="json"),
            "snapshot": snapshot.model_dump(mode="json"),
        }
        if session.mode.value != "qwen":
            result["legacyReadOnly"] = True
        return result

    def _persist_event(
        self,
        session: SimulationSession,
        snapshot: SimulationSnapshot,
        event_type: SimulationEventType,
        payload: Mapping[str, Any] | None = None,
        *,
        source: str = "simulator",
        session_status: SimulationStatus | None = None,
    ) -> tuple[SimulationSession, SimulationSnapshot, SimulationEvent]:
        now = utc_now()
        sequence = snapshot.lastEventSequence + 1
        event = SimulationEvent(
            eventId=_id("sim-event"),
            sessionId=session.sessionId,
            sequence=sequence,
            eventType=event_type,
            source=source,  # type: ignore[arg-type]
            payload=dict(payload or {}),
            occurredAt=now,
        )
        next_status = session_status or session.status
        pending_review = snapshot.pendingReview
        if event_type in {
            SimulationEventType.NAVIGATION_BLOCKED,
            SimulationEventType.NAVIGATION_FAILED,
            SimulationEventType.TARGET_LOST,
        }:
            detail_by_type = {
                SimulationEventType.NAVIGATION_BLOCKED: "仿真 Nav2 反馈被静态障碍阻塞，需要人工审核是否重规划",
                SimulationEventType.NAVIGATION_FAILED: "仿真导航失败，需要人工审核是否重规划",
                SimulationEventType.TARGET_LOST: "仿真视觉目标丢失，需要人工审核是否重规划",
            }
            pending_review = SimulationReview(
                triggerEventId=event.eventId,
                triggerEventType=event.eventType.value,
                missionId=session.missionId or snapshot.activeMissionId,
                decisionKind="needs_review",
                detail=detail_by_type[event_type],
                createdAt=now,
                updatedAt=now,
            )
        elif event_type is SimulationEventType.RESET:
            pending_review = None
        next_session = session.model_copy(update={"status": next_status, "updatedAt": now})
        next_snapshot = snapshot.model_copy(
            update={
                "status": next_status,
                "lastEventSequence": sequence,
                "updatedAt": now,
                "pendingReview": pending_review,
            }
        )
        self.store.append_event(event)
        self.store.save_session(next_session, next_snapshot)
        sink = self._snapshot_sink
        if sink is not None:
            try:
                sink(next_session, next_snapshot)
            except Exception:
                # A UI projection must not make the durable simulation event
                # disappear. The next snapshot poll can repair the view.
                pass
        event_sink = self._event_sink
        if event_sink is not None:
            try:
                event_sink(next_session, next_snapshot, event)
            except Exception:
                # Review generation is advisory. The event and its generic
                # human-review marker remain durable even if the planner is
                # unavailable or returns an invalid proposal.
                pass
        # A synchronous Mission bridge may enrich ``pendingReview`` after the
        # event commit. Return that latest durable snapshot so the API response
        # and the next browser poll agree on the same review state.
        latest = self.store.get_session(session.sessionId)
        if latest is not None:
            latest_session, latest_snapshot = latest
            return latest_session, latest_snapshot, event
        return next_session, next_snapshot, event

    def create(self, payload: Mapping[str, Any] | SimulationSessionCreate) -> dict[str, Any]:
        request = payload if isinstance(payload, SimulationSessionCreate) else SimulationSessionCreate.model_validate(payload)
        if request.backend is SimulationBackend.ISAAC:
            raise SimulationConflict("isaac backend is not enabled; use deterministic or gazebo")
        now = utc_now()
        session_id = _id("sim")
        live_identity: Mapping[str, Any] = {}
        if request.backend is SimulationBackend.MUJOCO_SONIC and self._identity_provider is not None:
            try:
                candidate = self._identity_provider(request.robotId)
                if isinstance(candidate, Mapping):
                    live_identity = candidate
            except Exception as exc:
                raise SimulationConflict(f"simulation Robot Agent identity is unavailable: {exc}") from exc
        boot_id = request.robotBootId or str(live_identity.get("robotBootId") or "") or (
            "fixture-boot-1" if request.robotId == "robot-1" else _id("sim-boot")
        )
        epoch_id = request.localizationEpochId or str(live_identity.get("localizationEpochId") or "") or (
            "lab-map-epoch-1" if request.robotId == "robot-1" else _id("sim-epoch")
        )
        start_pose = request.startPose.model_copy(update={"observedAt": now})
        if request.backend is SimulationBackend.MUJOCO_SONIC:
            raw_live_pose = live_identity.get("currentPose")
            if not isinstance(raw_live_pose, Mapping):
                raise SimulationConflict(
                    "simulation Robot Agent current pose is unavailable"
                )
            try:
                start_pose = PoseMap.model_validate(raw_live_pose)
            except (TypeError, ValueError) as exc:
                raise SimulationConflict(
                    "simulation Robot Agent current pose is invalid"
                ) from exc
            if start_pose.observedAt is None:
                raise SimulationConflict(
                    "simulation Robot Agent current pose is stale"
                )
        metadata = dict(request.metadata)
        metadata["startPose"] = start_pose.model_dump(mode="json")
        if live_identity.get("observationSessionId"):
            metadata["observationSessionId"] = str(live_identity["observationSessionId"])
        if live_identity.get("capabilitiesDigest"):
            metadata["capabilitiesDigest"] = str(live_identity["capabilitiesDigest"])
        session = SimulationSession(
            sessionId=session_id,
            backend=request.backend,
            sourceMode=request.sourceMode,
            robotId=request.robotId,
            robotBootId=boot_id,
            localizationEpochId=epoch_id,
            missionId=request.missionId,
            sceneRef=request.sceneRef,
            mode=request.mode,
            status=SimulationStatus.CREATED,
            policy=request.policy,
            metadata=metadata,
            createdAt=now,
            updatedAt=now,
        )
        live_backend = request.backend is SimulationBackend.MUJOCO_SONIC
        snapshot = SimulationSnapshot(
            sessionId=session_id,
            status=SimulationStatus.CREATED,
            pose=start_pose,
            trajectory=[start_pose],
            obstacles=([] if live_backend else self._deterministic_office_obstacles()),
            landmarks=([] if live_backend else self._deterministic_office_landmarks()),
            activeMissionId=request.missionId,
            updatedAt=now,
        )
        with self._lock:
            self.store.save_session(session, snapshot)
            session, snapshot, _ = self._persist_event(
                session,
                snapshot,
                SimulationEventType.RESET,
                {"reason": "session_created", "backend": request.backend.value},
            )
        return self._result(session, snapshot)

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return self._result(*self._get(session_id))

    def live(self, session_id: str) -> dict[str, Any]:
        """Return a browser-safe live MuJoCo observation for one session."""

        with self._lock:
            session, snapshot = self._get(session_id)
        if session.mode.value != "qwen":
            return {**self._result(session, snapshot), "live": None}
        if session.backend is not SimulationBackend.MUJOCO_SONIC:
            return {**self._result(session, snapshot), "live": None}
        try:
            live = self.mujoco_client.live()
        except MujocoSonicUnavailable as exc:
            raise SimulationConflict(str(exc)) from exc
        command_id = session.metadata.get("activeNavigationCommandId")
        navigation: dict[str, Any] | None = None
        provider = self._navigation_status_provider
        if isinstance(command_id, str) and command_id and provider is not None:
            try:
                navigation = dict(provider(command_id))
            except Exception as exc:
                navigation = {
                    "status": "unknown",
                    "detail": f"Robot Agent status unavailable: {type(exc).__name__}",
                }
        elif isinstance(session.metadata.get("lastNavigationReceipt"), Mapping):
            navigation = dict(session.metadata["lastNavigationReceipt"])
        live["navigation"] = navigation

        # In-flight observations remain a read-only projection. A terminal
        # Robot Agent receipt is different: it is durable command truth and
        # must be reconciled exactly once into the simulation event ledger so
        # the observatory, session detail, and audit history cannot disagree.
        live_pose = live.get("pose")
        status_value = str((navigation or {}).get("status") or "").lower()
        terminal_event = {
            "succeeded": SimulationEventType.NAVIGATION_SUCCEEDED,
            "blocked": SimulationEventType.NAVIGATION_BLOCKED,
            "failed": SimulationEventType.NAVIGATION_FAILED,
            "cancelled": SimulationEventType.NAVIGATION_FAILED,
            "expired": SimulationEventType.NAVIGATION_FAILED,
            "rejected": SimulationEventType.NAVIGATION_FAILED,
        }.get(status_value)
        navigation_result = (
            navigation.get("result")
            if isinstance(navigation, Mapping)
            else None
        )
        pose_relation = (
            str(navigation_result.get("finalPoseRelation") or "")
            if isinstance(navigation_result, Mapping)
            else ""
        )
        terminal_pose_pending = pose_relation in {
            "latest_before_terminal",
            "awaiting_post_terminal",
            "settling_after_terminal",
        }
        if terminal_pose_pending and isinstance(navigation_result, Mapping):
            raw_received_at = navigation_result.get("terminalReceivedAt")
            try:
                received_at = datetime.fromisoformat(
                    str(raw_received_at).replace("Z", "+00:00")
                )
                if received_at.tzinfo is None:
                    raise ValueError("terminalReceivedAt must be timezone-aware")
                # A walking body may need a short bounded interval to become
                # physically stationary after Nav2 reports success. Do not
                # hide a terminal state forever if localization is lost: the
                # Gateway normally resolves this earlier, while five seconds
                # remains the explicit workstation fallback ceiling.
                terminal_pose_pending = (utc_now() - received_at).total_seconds() < 5.0
            except (TypeError, ValueError):
                terminal_pose_pending = False
        if (
            terminal_event is not None
            and not terminal_pose_pending
            and isinstance(command_id, str)
            and command_id
            and navigation is not None
        ):
            with self._lock:
                current_session, current_snapshot = self._get(session_id)
                if current_session.metadata.get("activeNavigationCommandId") == command_id:
                    reconciled_pose: PoseMap | None = None
                    pose_source: str | None = None
                    terminal_pose = (
                        navigation_result.get("finalPose")
                        if isinstance(navigation_result, Mapping)
                        else None
                    )
                    if (
                        isinstance(terminal_pose, Mapping)
                        and pose_relation not in {
                            "latest_before_terminal",
                            "awaiting_post_terminal",
                            "settling_after_terminal",
                        }
                    ):
                        try:
                            candidate = PoseMap.model_validate(terminal_pose)
                        except (TypeError, ValueError):
                            candidate = None
                        if candidate is not None and candidate.observedAt is not None:
                            reconciled_pose = candidate
                        pose_source = "robot_agent_terminal_receipt"
                    if reconciled_pose is None and isinstance(live_pose, Mapping):
                        reconciled_pose = PoseMap.model_validate(live_pose).model_copy(
                            update={"observedAt": utc_now()}
                        )
                        pose_source = "mujoco_live_reconciliation"
                    trajectory = list(current_snapshot.trajectory)
                    if (
                        reconciled_pose is not None
                        and (not trajectory or distance_xy(trajectory[-1], reconciled_pose) >= 0.02)
                    ):
                        trajectory = [*trajectory, reconciled_pose][-10_000:]
                    next_state = "arrived" if terminal_event is SimulationEventType.NAVIGATION_SUCCEEDED else (
                        "blocked" if terminal_event is SimulationEventType.NAVIGATION_BLOCKED else "failed"
                    )
                    snapshot_updates: dict[str, Any] = {
                        "navigationState": next_state,
                        "plannedWaypoints": [],
                        "trajectory": trajectory,
                    }
                    if reconciled_pose is not None:
                        snapshot_updates["pose"] = reconciled_pose
                    if terminal_event is SimulationEventType.NAVIGATION_SUCCEEDED:
                        snapshot_updates["goalPose"] = None
                    current_snapshot = current_snapshot.model_copy(update=snapshot_updates)

                    receipt = dict(navigation)
                    receipt["commandId"] = command_id
                    metadata = dict(current_session.metadata)
                    metadata.pop("activeNavigationCommandId", None)
                    metadata["lastNavigationCommandId"] = command_id
                    metadata["lastNavigationReceipt"] = receipt
                    current_session = current_session.model_copy(update={"metadata": metadata})
                    event_payload: dict[str, Any] = {
                        "commandId": command_id,
                        "status": status_value,
                        "detail": navigation.get("detail"),
                        "result": navigation.get("result"),
                    }
                    if reconciled_pose is not None:
                        event_payload["pose"] = reconciled_pose.model_dump(mode="json")
                        event_payload["poseSource"] = pose_source
                    session, snapshot, _ = self._persist_event(
                        current_session,
                        current_snapshot,
                        terminal_event,
                        event_payload,
                        source="robot_agent",
                    )

        projected_state = {
            "succeeded": "arrived",
            "blocked": "blocked",
            "failed": "failed",
            "cancelled": "failed",
            "expired": "failed",
            "rejected": "failed",
            "accepted": "moving",
            "running": "moving",
        }.get(status_value, snapshot.navigationState)
        if isinstance(live_pose, Mapping) and terminal_event is None:
            projected_pose = PoseMap.model_validate(live_pose).model_copy(update={"observedAt": utc_now()})
            trajectory = list(snapshot.trajectory)
            if not trajectory or distance_xy(trajectory[-1], projected_pose) >= 0.02:
                trajectory = [*trajectory, projected_pose][-10_000:]
            snapshot = snapshot.model_copy(
                update={
                    "navigationState": projected_state,
                    "pose": projected_pose,
                    "trajectory": trajectory,
                    "updatedAt": utc_now(),
                }
            )
        elif terminal_event is None:
            snapshot = snapshot.model_copy(update={"navigationState": projected_state})
        return {**self._result(session, snapshot), "live": live}

    def rgb_frame(self, session_id: str) -> bytes:
        """Return one synthetic camera frame for an authenticated UI client."""

        with self._lock:
            session, _ = self._get(session_id)
        if session.backend is not SimulationBackend.MUJOCO_SONIC:
            raise SimulationConflict("RGB camera is only available for MuJoCo+SONIC sessions")
        try:
            return self.mujoco_client.rgb_frame()
        except MujocoSonicUnavailable as exc:
            raise SimulationConflict(str(exc)) from exc

    def bind_mission(self, session_id: str, mission_id: str) -> dict[str, Any]:
        """Bind one existing Mission identity without authorizing execution."""

        if not isinstance(mission_id, str) or not mission_id.strip():
            raise SimulationConflict("missionId is required")
        mission_id = mission_id.strip()
        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            if session.status in {SimulationStatus.STOPPED, SimulationStatus.FAILED}:
                raise SimulationConflict("stopped or failed simulation cannot bind a mission")
            if session.missionId and session.missionId != mission_id:
                raise SimulationConflict("simulation is already bound to another mission")
            if session.missionId == mission_id and snapshot.activeMissionId == mission_id:
                return self._result(session, snapshot)
            next_snapshot = snapshot.model_copy(update={"activeMissionId": mission_id})
            next_session, next_snapshot, _ = self._persist_event(
                session.model_copy(update={"missionId": mission_id}),
                next_snapshot,
                SimulationEventType.MISSION_BOUND,
                {"missionId": mission_id},
                source="operator",
            )
            return self._result(next_session, next_snapshot)

    def set_review(self, session_id: str, review: Mapping[str, Any] | SimulationReview) -> dict[str, Any]:
        """Persist a Mission bridge decision without changing authorization."""

        value = review if isinstance(review, SimulationReview) else SimulationReview.model_validate(review)
        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            if value.missionId and session.missionId and value.missionId != session.missionId:
                raise SimulationConflict("review mission does not match simulation binding")
            updated = snapshot.model_copy(update={"pendingReview": value, "updatedAt": utc_now()})
            self.store.save_session(session.model_copy(update={"updatedAt": updated.updatedAt}), updated)
            sink = self._snapshot_sink
            if sink is not None:
                try:
                    sink(session, updated)
                except Exception:
                    pass
            return self._result(session, updated)

    def resolve_review(
        self,
        session_id: str,
        *,
        revision_id: str | None = None,
        detail: str = "Mission 人工审核已完成",
    ) -> dict[str, Any]:
        """Mark a review resolved after an external Mission approval check."""

        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            review = snapshot.pendingReview
            if review is None:
                return self._result(session, snapshot)
            if revision_id is not None and review.revisionId != revision_id:
                raise SimulationConflict("review revision does not match the pending proposal")
            now = utc_now()
            updated_review = review.model_copy(
                update={"status": "resolved", "detail": detail, "updatedAt": now}
            )
            updated = snapshot.model_copy(update={"pendingReview": updated_review, "updatedAt": now})
            next_session = session.model_copy(update={"updatedAt": now})
            self.store.save_session(next_session, updated)
            sink = self._snapshot_sink
            if sink is not None:
                try:
                    sink(next_session, updated)
                except Exception:
                    pass
            return self._result(next_session, updated)

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [self._result(session, snapshot) for session, snapshot in self.store.list_sessions(limit=limit)]

    def backends(self) -> list[dict[str, Any]]:
        """Return explicit backend readiness without attempting to launch one."""

        gazebo_installed = shutil.which("gz") is not None
        try:
            mujoco_health = self.mujoco_client.health()
            mujoco_available = self.mujoco_client.ready()
            mujoco_reason = (
                "MuJoCo plant and SONIC telemetry are ready"
                if mujoco_available
                else "MuJoCo+SONIC processes are present but not ready"
            )
        except MujocoSonicUnavailable as exc:
            mujoco_health = None
            mujoco_available = False
            mujoco_reason = str(exc)
        return [
            {
                "backend": SimulationBackend.DETERMINISTIC.value,
                "available": True,
                "reason": "H-IDE deterministic contract simulator",
            },
            {
                "backend": SimulationBackend.MUJOCO_SONIC.value,
                "available": mujoco_available,
                "installed": mujoco_health is not None,
                "reason": mujoco_reason,
                **({"health": mujoco_health} if mujoco_health is not None else {}),
            },
            {
                "backend": SimulationBackend.GAZEBO.value,
                # A binary alone is not a safe integration signal. The
                # Robot Agent/Gazebo observation + Nav2 adapter is still a
                # separate, not-yet-enabled boundary.
                "available": False,
                "installed": gazebo_installed,
                "reason": (
                    "Gazebo executable detected, but Robot Agent/Gazebo adapter is not wired"
                    if gazebo_installed
                    else "gz executable is not installed"
                ),
            },
            {
                "backend": SimulationBackend.ISAAC.value,
                "available": False,
                "reason": "Isaac is intentionally not part of the mainline profile",
            },
        ]

    def events(self, session_id: str, *, after: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        with self._lock:
            self._get(session_id)
            return [event.model_dump(mode="json") for event in self.store.list_events(session_id, after=after, limit=limit)]

    def start(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            if session.status not in {SimulationStatus.CREATED, SimulationStatus.PAUSED}:
                raise SimulationConflict(f"simulation cannot start from {session.status.value}")
            if session.backend is SimulationBackend.GAZEBO:
                raise SimulationConflict(
                    "gazebo backend is staged but not enabled: connect the Robot Agent/Gazebo adapter or use deterministic"
                )
            if session.backend is SimulationBackend.MUJOCO_SONIC and not self.mujoco_client.ready():
                raise SimulationConflict(
                    "mujoco_sonic backend is not ready: start the MuJoCo plant, SONIC driver, and ROS bridges first"
                )
            if session.backend is SimulationBackend.MUJOCO_SONIC and self._navigation_dispatcher is None:
                raise SimulationConflict(
                    "mujoco_sonic Robot Agent/Nav2 dispatcher is not configured"
                )
            session, snapshot, _ = self._persist_event(
                session,
                snapshot,
                SimulationEventType.STARTED,
                {"executionProfile": "simulation"},
                session_status=SimulationStatus.RUNNING,
            )
            return self._result(session, snapshot)

    def pause(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session, snapshot = self._get(session_id)
            if session.status is not SimulationStatus.RUNNING:
                raise SimulationConflict(f"simulation cannot pause from {session.status.value}")
            session, snapshot, _ = self._persist_event(
                session, snapshot, SimulationEventType.PAUSED, {}, session_status=SimulationStatus.PAUSED
            )
            return self._result(session, snapshot)

    def stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session, snapshot = self._get(session_id)
            if session.status in {SimulationStatus.STOPPED, SimulationStatus.FAILED}:
                return self._result(session, snapshot)
            session, snapshot, _ = self._persist_event(
                session, snapshot, SimulationEventType.STOPPED, {}, session_status=SimulationStatus.STOPPED
            )
            return self._result(session, snapshot)

    def reset(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            raw_start = session.metadata.get("startPose")
            if not isinstance(raw_start, Mapping):
                raise SimulationConflict("simulation start pose is unavailable")
            start_pose = PoseMap.model_validate(raw_start).model_copy(update={"observedAt": utc_now()})
            base = snapshot.model_copy(
                update={
                    "status": SimulationStatus.CREATED,
                    "navigationState": "idle",
                    "pose": start_pose,
                    "goalPose": None,
                    "plannedWaypoints": [],
                    "trajectory": [start_pose],
                    "activeMissionId": session.missionId,
                    "activeRevisionId": None,
                }
            )
            session = session.model_copy(update={"status": SimulationStatus.CREATED})
            session, snapshot, _ = self._persist_event(
                session, base, SimulationEventType.RESET, {"reason": "operator_reset"}, session_status=SimulationStatus.CREATED
            )
            return self._result(session, snapshot)

    def set_goal(self, session_id: str, payload: Mapping[str, Any] | NavigationGoalInput) -> dict[str, Any]:
        request = payload if isinstance(payload, NavigationGoalInput) else NavigationGoalInput.model_validate(payload)
        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            if session.status is not SimulationStatus.RUNNING:
                raise SimulationConflict("simulation must be running before a navigation goal is accepted")
            if snapshot.pendingReview is not None and snapshot.pendingReview.status != "resolved":
                raise SimulationConflict(
                    "simulation has a pending Mission review; reset or resolve the review before sending another goal"
                )
            if request.missionId and session.missionId and request.missionId != session.missionId:
                raise SimulationConflict("navigation goal mission does not match simulation binding")
            distance = distance_xy(snapshot.pose, request.goalPose)
            if distance > session.policy.maximumGoalDistanceM:
                raise SimulationConflict("navigation goal exceeds the simulation policy distance bound")
            if session.backend is SimulationBackend.MUJOCO_SONIC:
                dispatcher = self._navigation_dispatcher
                if dispatcher is None:
                    raise SimulationConflict(
                        "mujoco_sonic navigation dispatcher is not configured; no command was sent"
                    )
                # The external Nav2 action owns path planning and feedback. Do
                # not manufacture geometric waypoints for the live backend.
                dispatched = dispatcher(session, request)
                if not isinstance(dispatched, Mapping):
                    raise SimulationConflict("simulation Robot Agent dispatcher returned an invalid result")
                dispatched_result = dispatched.get("result")
                command_id = dispatched.get("commandId")
                if not command_id and isinstance(dispatched_result, Mapping):
                    command_id = dispatched_result.get("robotAgentCommandId")
                if command_id:
                    metadata = dict(session.metadata)
                    metadata["activeNavigationCommandId"] = str(command_id)
                    session = session.model_copy(update={"metadata": metadata})
                next_snapshot = snapshot.model_copy(
                    update={
                        "navigationState": "moving",
                        "goalPose": request.goalPose.model_copy(update={"observedAt": utc_now()}),
                        "plannedWaypoints": [],
                        "activeMissionId": request.missionId or snapshot.activeMissionId or session.missionId,
                        "activeRevisionId": request.revisionId or snapshot.activeRevisionId,
                    }
                )
                session, next_snapshot, _ = self._persist_event(
                    session,
                    next_snapshot,
                    SimulationEventType.NAVIGATION_GOAL,
                    {
                        "goalPose": request.goalPose.model_dump(mode="json"),
                        "missionId": request.missionId,
                        "revisionId": request.revisionId,
                        "nodeId": request.nodeId,
                        "distanceM": distance,
                        "executionBackend": "robot_agent_nav2_sonic",
                        "dispatcher": dict(dispatched),
                    },
                    source="robot_agent",
                )
                receipt_status = str(dispatched.get("status") or "running").lower()
                if receipt_status not in {"accepted", "running", "succeeded", "blocked", "failed", "cancelled"}:
                    receipt_status = "running"
                return {
                    **self._result(session, next_snapshot),
                    "receipt": SimulationReceipt(
                        receiptId=_id("sim-receipt"),
                        sessionId=session_id,
                        commandId=str(command_id or _id("sim-command")),
                        status=receipt_status,  # type: ignore[arg-type]
                        executionBackend=str(dispatched.get("executionBackend") or "robot_agent_nav2_sonic"),
                        motion="mujoco",
                        goalPose=request.goalPose,
                        detail=str(dispatched.get("detail") or "NavigateToPose dispatched through Robot Agent and Nav2"),
                        createdAt=utc_now(),
                    ).model_dump(mode="json"),
                }
            planned = _detour_waypoints(
                snapshot.pose,
                request.goalPose,
                snapshot.obstacles,
            )
            next_snapshot = snapshot.model_copy(
                update={
                    "navigationState": "moving",
                    "goalPose": request.goalPose.model_copy(update={"observedAt": utc_now()}),
                    "plannedWaypoints": planned or [],
                    "activeMissionId": request.missionId or snapshot.activeMissionId or session.missionId,
                    "activeRevisionId": request.revisionId or snapshot.activeRevisionId,
                }
            )
            session, next_snapshot, _ = self._persist_event(
                session,
                next_snapshot,
                SimulationEventType.NAVIGATION_GOAL,
                {
                    "goalPose": request.goalPose.model_dump(mode="json"),
                    "missionId": request.missionId,
                    "revisionId": request.revisionId,
                    "nodeId": request.nodeId,
                    "distanceM": distance,
                    "plannedWaypointCount": len(planned or []),
                },
                source="nav2",
            )
            return {
                **self._result(session, next_snapshot),
                "receipt": SimulationReceipt(
                    receiptId=_id("sim-receipt"),
                    sessionId=session_id,
                    commandId=_id("sim-command"),
                    status="running",
                    executionBackend=f"{session.backend.value}_simulator",
                    goalPose=request.goalPose,
                    detail="simulated NavigateToPose accepted",
                    createdAt=utc_now(),
                ).model_dump(mode="json"),
            }

    def step(self, session_id: str, payload: Mapping[str, Any] | SimulationStepInput | None = None) -> dict[str, Any]:
        request = payload if isinstance(payload, SimulationStepInput) else SimulationStepInput.model_validate(payload or {})
        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            if session.status is not SimulationStatus.RUNNING:
                raise SimulationConflict("simulation must be running before stepping")
            goal = snapshot.goalPose
            if goal is None:
                return {**self._result(session, snapshot), "receipt": None}
            waypoints = list(snapshot.plannedWaypoints)
            target = waypoints[0] if waypoints else goal
            if not snapshot.plannedWaypoints and not _segment_clear(snapshot.pose, goal, snapshot.obstacles):
                blocked = snapshot.model_copy(update={"navigationState": "blocked"})
                session, blocked, _ = self._persist_event(
                    session,
                    blocked,
                    SimulationEventType.NAVIGATION_BLOCKED,
                    {"goalPose": goal.model_dump(mode="json"), "reason": "static_obstacle_no_detour"},
                    source="nav2",
                )
                return {
                    **self._result(session, blocked),
                    "receipt": SimulationReceipt(
                        receiptId=_id("sim-receipt"),
                        sessionId=session_id,
                        commandId=_id("sim-command"),
                        status="blocked",
                        executionBackend=f"{session.backend.value}_simulator",
                        goalPose=goal,
                        detail="deterministic simulator detected a static obstacle",
                        createdAt=utc_now(),
                    ).model_dump(mode="json"),
                }
            remaining = distance_xy(snapshot.pose, target)
            travel = min(request.distanceM, remaining)
            ratio = 0.0 if remaining <= 1e-9 else travel / remaining
            x = snapshot.pose.x + (target.x - snapshot.pose.x) * ratio
            y = snapshot.pose.y + (target.y - snapshot.pose.y) * ratio
            yaw = math.atan2(target.y - snapshot.pose.y, target.x - snapshot.pose.x)
            pose = PoseMap(x=x, y=y, z=target.z, yaw=yaw, observedAt=utc_now())
            trajectory = list(snapshot.trajectory)
            trajectory.append(pose)
            reached_waypoint = remaining <= request.distanceM + 1e-9
            if reached_waypoint and waypoints:
                waypoints.pop(0)
            arrived = reached_waypoint and not waypoints
            next_snapshot = snapshot.model_copy(
                update={
                    "pose": target.model_copy(update={"observedAt": utc_now()}) if reached_waypoint else pose,
                    "goalPose": None if arrived else goal,
                    "plannedWaypoints": waypoints,
                    "trajectory": trajectory,
                    "navigationState": "arrived" if arrived else "moving",
                }
            )
            event_type = SimulationEventType.NAVIGATION_SUCCEEDED if arrived else SimulationEventType.NAVIGATION_FEEDBACK
            session, next_snapshot, _ = self._persist_event(
                session,
                next_snapshot,
                event_type,
                {
                    "currentPose": next_snapshot.pose.model_dump(mode="json"),
                    "distanceRemainingM": 0.0 if arrived else distance_xy(next_snapshot.pose, goal),
                    "waypointsRemaining": len(waypoints),
                },
                source="nav2",
            )
            status = "succeeded" if arrived else "running"
            return {
                **self._result(session, next_snapshot),
                "receipt": SimulationReceipt(
                    receiptId=_id("sim-receipt"),
                    sessionId=session_id,
                    commandId=_id("sim-command"),
                    status=status,
                    executionBackend=f"{session.backend.value}_simulator",
                    goalPose=goal,
                    detail="simulated navigation reached the goal" if arrived else "simulated Nav2 feedback",
                    createdAt=utc_now(),
                ).model_dump(mode="json"),
            }

    def inject_event(self, session_id: str, payload: Mapping[str, Any] | SimulationEventInput) -> dict[str, Any]:
        request = payload if isinstance(payload, SimulationEventInput) else SimulationEventInput.model_validate(payload)
        with self._lock:
            session, snapshot = self._get(session_id)
            self._require_product_session(session)
            event_payload = dict(request.payload)
            next_snapshot = snapshot
            if request.eventType is SimulationEventType.POSE_UPDATED:
                raw_pose = event_payload.get("pose")
                if not isinstance(raw_pose, Mapping):
                    raise SimulationConflict("pose_updated event requires payload.pose")
                pose = PoseMap.model_validate(raw_pose).model_copy(update={"observedAt": utc_now()})
                next_snapshot = snapshot.model_copy(
                    update={"pose": pose, "trajectory": [*snapshot.trajectory, pose]}
                )
                event_payload["pose"] = pose.model_dump(mode="json")
            elif request.eventType is SimulationEventType.NAVIGATION_SUCCEEDED:
                raw_pose = event_payload.get("pose")
                if isinstance(raw_pose, Mapping):
                    pose = PoseMap.model_validate(raw_pose).model_copy(update={"observedAt": utc_now()})
                    next_snapshot = snapshot.model_copy(
                        update={
                            "pose": pose,
                            "goalPose": None,
                            "plannedWaypoints": [],
                            "navigationState": "arrived",
                            "trajectory": [*snapshot.trajectory, pose],
                        }
                    )
                    event_payload["pose"] = pose.model_dump(mode="json")
            elif request.eventType is SimulationEventType.NAVIGATION_BLOCKED:
                next_snapshot = snapshot.model_copy(update={"navigationState": "blocked"})
            elif request.eventType is SimulationEventType.NAVIGATION_FAILED:
                next_snapshot = snapshot.model_copy(update={"navigationState": "failed"})
            session, next_snapshot, _ = self._persist_event(
                session,
                next_snapshot,
                request.eventType,
                event_payload,
                source="operator",
            )
            return self._result(session, next_snapshot)

    def close(self) -> None:
        if self._owns_store:
            self.store.close()


__all__ = ["SimulationConflict", "SimulationNotFound", "SimulationRuntime"]
