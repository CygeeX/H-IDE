"""Entrypoint for the isolated Scene Worker service."""

from __future__ import annotations

import logging
import os
import signal
import socket
import threading

from .client import SceneApiClient, SceneApiUnavailable
from .pipeline import SceneConversionPipeline
from .worker import process_one


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger("scene-worker")
    internal_token = os.environ.get("SCENE_API_INTERNAL_TOKEN", "")
    if not internal_token:
        raise RuntimeError("SCENE_API_INTERNAL_TOKEN is required")
    worker_id = os.getenv("SCENE_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")[:128]
    lease_seconds = int(os.getenv("SCENE_JOB_LEASE_SECONDS", "300"))
    poll_seconds = float(os.getenv("SCENE_JOB_POLL_SECONDS", "2"))
    if lease_seconds < 10 or poll_seconds <= 0:
        raise RuntimeError("scene worker lease and poll settings are invalid")

    client = SceneApiClient(
        os.getenv("SCENE_API_INTERNAL_URL", "http://scene-api:8090"),
        internal_token,
    )
    pipeline = SceneConversionPipeline(
        os.getenv("SCENE_STORAGE_ROOT", "/scene-data"),
        pcl_executable=os.getenv("PCL_CONVERT_BIN", "pcl_convert_pcd_ascii_binary"),
        pdal_executable=os.getenv("PDAL_BIN", "pdal"),
        potree_executable=os.getenv("POTREE_CONVERTER_BIN", "PotreeConverter"),
        command_timeout_seconds=int(os.getenv("SCENE_COMMAND_TIMEOUT_SECONDS", "3600")),
    )
    stopped = threading.Event()

    def stop(_signum, _frame) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopped.is_set():
            try:
                had_job = process_one(client, pipeline, worker_id, lease_seconds)
            except SceneApiUnavailable as error:
                logger.warning("Scene API unavailable: %s", error)
                had_job = False
            if not had_job:
                stopped.wait(poll_seconds)
    finally:
        client.close()


if __name__ == "__main__":
    main()
