from __future__ import annotations

from hmac import compare_digest
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status

from h_ide_vision.config import Settings, load_settings
from h_ide_vision.contracts import (
    VisualResolution,
    VisualResolutionRequest,
    VisualTaskRequest,
    VisualTaskResult,
)
from h_ide_vision.evidence import EvidenceStorage
from h_ide_vision.ingress import LatestFrameStore
from h_ide_vision.jobs import ResolutionJobRunner
from h_ide_vision.persistence import ResolutionNotFound, ResolutionRepository
from h_ide_vision.providers.segmentation import Sam2SegmentationProvider
from h_ide_vision.providers.vlm import (
    QwenVlmProvider,
)
from h_ide_vision.runtime import ResolutionService, VisualTaskService


def _bearer(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, separator, token = authorization.partition(" ")
    return token if separator and scheme.lower() == "bearer" and token else None


def build_service(settings: Settings, frames: LatestFrameStore | None = None) -> ResolutionService:
    # The product registry is deliberately closed: Qwen owns semantics and
    # SAM 2 may only refine a Qwen pixel box. Tests inject protocol stubs into
    # ``ResolutionService`` directly and cannot select a runtime provider.
    vlm = QwenVlmProvider(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
        model=settings.qwen_model,
    )
    segmentation = Sam2SegmentationProvider(
        checkpoint=settings.sam2_checkpoint,
        model_config=settings.sam2_model_config,
        device=settings.sam2_device,
    )
    vlm.ensure_ready()
    segmentation.ensure_ready()
    return ResolutionService(
        repository=ResolutionRepository(settings.database_path),
        frames=frames or LatestFrameStore(),
        evidence=EvidenceStorage(settings.evidence_dir),
        vlm=vlm,
        segmentation=segmentation,
        stable_frame_wait_seconds=settings.stable_frame_wait_seconds,
    )


def create_app(
    *,
    settings: Settings | None = None,
    service: ResolutionService | None = None,
    jobs: ResolutionJobRunner | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    service = service or build_service(settings)
    app = FastAPI(title="H-IDE Vision", version="0.1.0")
    app.state.settings = settings
    app.state.resolution_service = service
    visual_tasks = VisualTaskService(service)
    app.state.visual_task_service = visual_tasks
    jobs = jobs or ResolutionJobRunner(service)
    app.state.resolution_jobs = jobs
    app.router.add_event_handler("shutdown", jobs.close)

    def require_agent(authorization: Annotated[str | None, Header()] = None) -> None:
        token = _bearer(authorization)
        if token is None or not compare_digest(token, settings.agent_token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="agent credential required")

    def require_reader(authorization: Annotated[str | None, Header()] = None) -> None:
        token = _bearer(authorization)
        if token is None or not any(
            compare_digest(token, expected) for expected in (settings.agent_token, settings.control_token)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="vision read credential required",
            )

    @app.get("/healthz")
    def health() -> dict[str, object]:
        provider = service.vlm
        return {
            "status": "ok",
            "releaseState": "pure_vlm_product_profile",
            "capabilities": {
                "vlm": {
                    "provider": provider.provider_name,
                    "model": provider.model_name,
                    "available": provider.available,
                },
                "segmentation": {
                    "provider": service.segmentation.name,
                    "available": service.segmentation.available,
                },
                # The service may be fed by a live Robot Agent gateway or by a
                # deterministic fixture. Health only promises that the ingress
                # is ready; it must not mislabel a live camera as fixture data.
                "observation": "ingress_ready",
                "robotControl": False,
                "visualTaskV2": {
                    "available": provider.available and service.segmentation.available,
                    "modes": [
                        "object_approach",
                        "landmark_observation",
                        "directional_free_space",
                        "arrival_verification",
                        "scene_description",
                    ],
                    "semanticProviderBoundary": "qwen_only",
                    "maskRefinementBoundary": "sam2_from_qwen_box_only",
                    "semanticGroundingOutputAuthority": "hypothesis_only",
                    "geometryAuthority": "rgbd_calibration_tf",
                },
            },
        }

    @app.get("/internal/v1/capabilities", dependencies=[Depends(require_reader)])
    def capabilities() -> dict[str, object]:
        return health()["capabilities"]  # type: ignore[return-value]

    @app.post(
        "/internal/v1/visual-resolutions",
        response_model=VisualResolution,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(require_agent)],
    )
    def create_resolution(request: VisualResolutionRequest) -> VisualResolution:
        queued = service.create(request)
        jobs.submit(request)
        return queued

    @app.get(
        "/internal/v1/visual-resolutions/{resolution_id}",
        response_model=VisualResolution,
        dependencies=[Depends(require_reader)],
    )
    def get_resolution(resolution_id: str) -> VisualResolution:
        try:
            return service.repository.get(resolution_id)
        except ResolutionNotFound as exc:
            raise HTTPException(status_code=404, detail="visual resolution not found") from exc

    @app.get(
        "/internal/v1/visual-resolutions/{resolution_id}/evidence/{sha256}",
        dependencies=[Depends(require_reader)],
        response_class=Response,
    )
    def get_evidence(resolution_id: str, sha256: str) -> Response:
        try:
            resolution = service.repository.get(resolution_id)
        except ResolutionNotFound as exc:
            raise HTTPException(status_code=404, detail="visual resolution not found") from exc
        evidence = next((item for item in resolution.evidence if item.sha256 == sha256), None)
        if evidence is None:
            raise HTTPException(status_code=404, detail="evidence is not bound to visual resolution")
        try:
            body = service.evidence.get(sha256)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="visual evidence is unavailable") from exc
        return Response(
            content=body,
            media_type=evidence.mediaType,
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "Content-Security-Policy": "default-src 'none'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.post(
        "/internal/v1/visual-resolutions/{resolution_id}/cancel",
        response_model=VisualResolution,
        dependencies=[Depends(require_agent)],
    )
    def cancel_resolution(resolution_id: str) -> VisualResolution:
        try:
            return service.repository.cancel(resolution_id)
        except ResolutionNotFound as exc:
            raise HTTPException(status_code=404, detail="visual resolution not found") from exc

    @app.post(
        "/internal/v2/visual-tasks:analyze",
        response_model=VisualTaskResult,
        dependencies=[Depends(require_agent)],
    )
    def analyze_visual_task(request: VisualTaskRequest) -> VisualTaskResult:
        try:
            return visual_tasks.analyze(request)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get(
        "/internal/v2/visual-tasks/{analysis_id}",
        response_model=VisualTaskResult,
        dependencies=[Depends(require_reader)],
    )
    def get_visual_task(analysis_id: str) -> VisualTaskResult:
        try:
            return service.repository.get_visual_task_result(analysis_id)
        except ResolutionNotFound as exc:
            raise HTTPException(status_code=404, detail="visual task result not found") from exc

    @app.get(
        "/internal/v2/visual-tasks/{analysis_id}/evidence/{sha256}",
        dependencies=[Depends(require_reader)],
        response_class=Response,
    )
    def get_visual_task_evidence(analysis_id: str, sha256: str) -> Response:
        try:
            result = service.repository.get_visual_task_result(analysis_id)
        except ResolutionNotFound as exc:
            raise HTTPException(status_code=404, detail="visual task result not found") from exc
        evidence = next((item for item in result.evidence if item.sha256 == sha256), None)
        if evidence is None:
            raise HTTPException(status_code=404, detail="evidence is not bound to visual task")
        try:
            body = service.evidence.get(sha256)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="visual evidence is unavailable") from exc
        return Response(
            content=body,
            media_type=evidence.mediaType,
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "Content-Security-Policy": "default-src 'none'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app
