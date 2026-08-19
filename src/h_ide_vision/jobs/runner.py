from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from h_ide_vision.contracts import ResolutionStatus, VisualResolutionRequest
from h_ide_vision.runtime import ResolutionService


class ResolutionJobRunner:
    """Bounded local worker profile; production can replace it with a durable queue."""

    def __init__(self, service: ResolutionService, maximum_workers: int = 1) -> None:
        if maximum_workers != 1:
            raise ValueError("V1 GPU inference concurrency is fixed at one")
        self.service = service
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-resolution")
        self._lock = threading.RLock()
        self._jobs: dict[str, Future[Any]] = {}

    def submit(self, request: VisualResolutionRequest) -> None:
        current = self.service.repository.get_by_request(request.requestId)
        if current is not None and current.status not in {
            ResolutionStatus.QUEUED,
            ResolutionStatus.RESOLVING,
        }:
            return
        with self._lock:
            existing = self._jobs.get(request.requestId)
            if existing is not None and not existing.done():
                return
            future = self._executor.submit(self._execute, request)
            self._jobs[request.requestId] = future
            future.add_done_callback(lambda completed: self._finished(request.requestId, completed))

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _execute(self, request: VisualResolutionRequest) -> object:
        try:
            return self.service.resolve(request)
        except Exception as exc:
            reason = f"inference_failed:{type(exc).__name__}"
            self.service.repository.fail_request(request.requestId, reason)
            return reason

    def _finished(self, request_id: str, _future: Future[Any]) -> None:
        with self._lock:
            self._jobs.pop(request_id, None)
