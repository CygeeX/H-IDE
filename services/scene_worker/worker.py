"""Lease lifecycle and one-job orchestration."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from .client import LeaseLost, SceneApiUnavailable
from .pipeline import ConversionCancelled, SceneConversionPipeline, WorkerError


logger = logging.getLogger(__name__)


class SceneClient(Protocol):
    def claim(self, worker_id: str, lease_seconds: int) -> dict | None: ...
    def heartbeat(self, job_id: str, worker_id: str, lease_seconds: int) -> None: ...
    def complete(self, job_id: str, worker_id: str, manifest: dict) -> None: ...
    def fail(
        self,
        job_id: str,
        worker_id: str,
        deterministic: bool,
        error_code: str,
        message: str,
    ) -> None: ...


class LeaseKeeper:
    def __init__(self, client: SceneClient, job_id: str, worker_id: str, lease_seconds: int):
        self.client = client
        self.job_id = job_id
        self.worker_id = worker_id
        self.lease_seconds = lease_seconds
        self.stop_event = threading.Event()
        self.lost_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"scene-lease-{job_id}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        interval = max(1.0, min(30.0, self.lease_seconds / 3))
        while not self.stop_event.wait(interval):
            try:
                self.client.heartbeat(self.job_id, self.worker_id, self.lease_seconds)
            except (LeaseLost, SceneApiUnavailable):
                self.lost_event.set()
                return


def process_one(
    client: SceneClient,
    pipeline: SceneConversionPipeline,
    worker_id: str,
    lease_seconds: int,
) -> bool:
    """Claim and process at most one job; return False when the queue is empty."""

    job = client.claim(worker_id, lease_seconds)
    if job is None:
        return False
    job_id = job.get("jobId")
    if not isinstance(job_id, str):
        raise SceneApiUnavailable("Scene API returned a malformed job")

    keeper = LeaseKeeper(client, job_id, worker_id, lease_seconds)
    keeper.start()
    manifest: dict | None = None
    failure: WorkerError | None = None
    try:
        manifest = pipeline.convert(job, keeper.lost_event)
    except ConversionCancelled:
        keeper.lost_event.set()
    except WorkerError as error:
        failure = error
    except Exception:
        logger.exception("unexpected scene conversion failure for job %s", job_id)
        failure = WorkerError(
            "internal_error",
            "scene conversion failed unexpectedly",
            deterministic=False,
        )
    finally:
        keeper.stop()

    if keeper.lost_event.is_set():
        return True
    try:
        if failure is not None:
            client.fail(
                job_id,
                worker_id,
                failure.deterministic,
                failure.code,
                failure.message,
            )
        else:
            assert manifest is not None
            client.complete(job_id, worker_id, manifest)
    except LeaseLost:
        return True
    return True
