#!/usr/bin/env python3
"""Exercise only the process-isolated passive-viewer lifecycle on a real X11.

This tool never advances physics, starts Plant or SONIC, opens a network
listener, or creates a ROS/DDS participant.  It repeatedly publishes one
static FULLPHYSICS snapshot through the in-process shared-memory mailbox,
starts the production ``PassiveViewerWorker``, and requires a clean native
shutdown before moving to the next cycle.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Callable, Iterable

from hide_mujoco_sonic.config import SimConfig
from hide_mujoco_sonic.render_workers import (
    LatestStateMailbox,
    MujocoStateShape,
    PassiveViewerWorker,
    ViewerWorkerConfig,
)


SCHEMA_VERSION = "h-ide.passive-viewer-lifecycle-smoke/1"


class ViewerLifecycleSmokeFailure(RuntimeError):
    """One isolated viewer cycle did not meet its lifecycle contract."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _configure_x11_environment(
    *,
    display: str | None,
    xauthority: str | None,
) -> tuple[str, str | None]:
    resolved_display = str(display or os.getenv("DISPLAY", "")).strip()
    if not resolved_display:
        raise ViewerLifecycleSmokeFailure(
            "DISPLAY is required for the passive-viewer lifecycle smoke"
        )
    os.environ["DISPLAY"] = resolved_display

    resolved_xauthority = str(
        xauthority or os.getenv("XAUTHORITY", "")
    ).strip()
    if resolved_xauthority:
        authority_path = Path(resolved_xauthority).expanduser().resolve()
        if not authority_path.is_file():
            raise ViewerLifecycleSmokeFailure(
                f"XAUTHORITY is not a file: {authority_path}"
            )
        resolved_xauthority = str(authority_path)
        os.environ["XAUTHORITY"] = resolved_xauthority
    return resolved_display, resolved_xauthority or None


def _initialize_static_state(mujoco_api, config: SimConfig, model, data) -> None:
    """Place the model at its configured spawn without taking a physics step."""

    if int(model.nq) < 7 or int(model.nv) < 6:
        raise ViewerLifecycleSmokeFailure(
            "viewer lifecycle model must have a floating root"
        )
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    half_yaw = float(config.runtime_scene.spawn_yaw) / 2.0
    data.qpos[:3] = (
        float(config.runtime_scene.spawn_x),
        float(config.runtime_scene.spawn_y),
        float(model.qpos0[2]),
    )
    data.qpos[3:7] = (
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    data.time = 0.0
    mujoco_api.mj_forward(model, data)


def _failure_detail(cycle: int, status: dict[str, object]) -> str:
    return (
        f"viewer cycle {cycle} failed: status={status.get('status')}, "
        f"alive={status.get('alive')}, exitCode={status.get('exitCode')}, "
        f"lastError={status.get('lastError')}"
    )


def _require_running_status(cycle: int, status: dict[str, object]) -> None:
    if status.get("status") in {"failed", "stale"}:
        raise ViewerLifecycleSmokeFailure(_failure_detail(cycle, status))
    if not bool(status.get("alive")):
        raise ViewerLifecycleSmokeFailure(_failure_detail(cycle, status))


def _require_clean_shutdown(cycle: int, status: dict[str, object]) -> None:
    failures: list[str] = []
    if status.get("status") != "stopped":
        failures.append(f"status={status.get('status')}")
    if bool(status.get("ready")):
        failures.append("worker remains ready after close")
    if bool(status.get("alive")):
        failures.append("worker is still alive")
    if status.get("exitCode") != 0:
        failures.append(f"exitCode={status.get('exitCode')}")
    if int(status.get("failureCount") or 0) != 0:
        failures.append(f"failureCount={status.get('failureCount')}")
    if status.get("lastError"):
        failures.append(f"lastError={status.get('lastError')}")
    if not bool(status.get("shutdownStarted")):
        failures.append("shutdown did not start")
    if not bool(status.get("nativeShutdownComplete")):
        failures.append("native shutdown did not complete")
    if bool(status.get("nativeViewerActive")):
        failures.append("native viewer remains active")
    if failures:
        raise ViewerLifecycleSmokeFailure(
            f"viewer cycle {cycle} did not close cleanly: " + "; ".join(failures)
        )


def run_lifecycle_smoke(
    *,
    mujoco_api,
    config: SimConfig,
    display: str,
    xauthority: str | None,
    cycles: int,
    visible_seconds: float,
    startup_timeout_s: float,
    publish_period_s: float,
    close_timeout_s: float,
    context=None,
    worker_factory=PassiveViewerWorker,
    mailbox_factory=LatestStateMailbox,
    shape_factory=MujocoStateShape.from_model,
    monotonic: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
    unix_ns: Callable[[], int] = time.time_ns,
) -> dict[str, object]:
    """Run repeated static viewer cycles and return machine-readable evidence."""

    if cycles <= 0:
        raise ValueError("cycles must be positive")
    for name, value in (
        ("visible_seconds", visible_seconds),
        ("startup_timeout_s", startup_timeout_s),
        ("publish_period_s", publish_period_s),
        ("close_timeout_s", close_timeout_s),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

    process_context = context or multiprocessing.get_context("spawn")
    model = mujoco_api.MjModel.from_xml_path(str(config.model_scene))
    data = mujoco_api.MjData(model)
    _initialize_static_state(mujoco_api, config, model, data)
    shape = shape_factory(mujoco_api, model)
    started_at_ns = unix_ns()
    total_published = 0
    results: list[dict[str, object]] = []

    for cycle in range(1, cycles + 1):
        mailbox = mailbox_factory(shape, context=process_context)
        sequence = mailbox.try_publish(
            mujoco_api,
            model,
            data,
            captured_at_unix_ns=unix_ns(),
        )
        if sequence is None:
            raise ViewerLifecycleSmokeFailure(
                f"viewer cycle {cycle} could not publish its initial snapshot"
            )
        cycle_published = 1
        total_published += 1
        worker = worker_factory(
            mailbox=mailbox,
            shape=shape,
            config=ViewerWorkerConfig(
                runtime_manifest=str(config.scene_runtime_manifest),
                model_scene=str(config.model_scene),
                reset_policy=config.reset_policy,
                spawn_x=float(config.runtime_scene.spawn_x),
                spawn_y=float(config.runtime_scene.spawn_y),
            ),
            context=process_context,
        )
        cycle_started = monotonic()
        worker_started = False
        try:
            # Mark cleanup as required before start: multiprocessing may create
            # part of a child and then raise, and close() is safe with pid=None.
            worker_started = True
            worker.start()
            startup_deadline = cycle_started + startup_timeout_s
            status = worker.status()
            while not bool(status.get("ready")):
                _require_running_status(cycle, status)
                remaining = startup_deadline - monotonic()
                if remaining <= 0.0:
                    raise ViewerLifecycleSmokeFailure(
                        f"viewer cycle {cycle} did not become ready within "
                        f"{startup_timeout_s:.3f}s"
                    )
                wait(min(publish_period_s, remaining))
                status = worker.status()

            ready_at = monotonic()
            visible_deadline = ready_at + visible_seconds
            while monotonic() < visible_deadline:
                _require_running_status(cycle, worker.status())
                published = mailbox.try_publish(
                    mujoco_api,
                    model,
                    data,
                    captured_at_unix_ns=unix_ns(),
                )
                if published is not None:
                    cycle_published += 1
                    total_published += 1
                remaining = visible_deadline - monotonic()
                if remaining > 0.0:
                    wait(min(publish_period_s, remaining))

            worker.close(timeout_s=close_timeout_s)
            final_status = worker.status()
            _require_clean_shutdown(cycle, final_status)
            results.append(
                {
                    "cycle": cycle,
                    "workerPid": final_status.get("workerPid"),
                    "workerBootId": final_status.get("workerBootId"),
                    "startupMs": (ready_at - cycle_started) * 1000.0,
                    "visibleMs": visible_seconds * 1000.0,
                    "snapshotsPublished": cycle_published,
                    "finalStatus": final_status.get("status"),
                    "readyAfterClose": final_status.get("ready"),
                    "exitCode": final_status.get("exitCode"),
                    "nativeShutdownComplete": True,
                    "forcedKill": False,
                }
            )
        except BaseException:
            if worker_started:
                worker.close(timeout_s=close_timeout_s)
            raise

    finished_at_ns = unix_ns()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "status": "passed",
        "startedAtUnixNs": started_at_ns,
        "finishedAtUnixNs": finished_at_ns,
        "mujocoVersion": str(getattr(mujoco_api, "__version__", "unknown")),
        "display": display,
        "xauthority": xauthority,
        "modelScene": str(Path(config.model_scene).resolve()),
        "cyclesRequested": cycles,
        "cyclesCompleted": len(results),
        "staticSnapshotsPublished": total_published,
        "physicsSteps": 0,
        "controlAuthority": False,
        "networkListeners": [],
        "rosDdsEnabled": False,
        "cycles": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/sim.yaml")
    parser.add_argument("--cycles", type=_positive_int, default=20)
    parser.add_argument("--visible-seconds", type=_positive_float, default=0.5)
    parser.add_argument("--startup-timeout-s", type=_positive_float, default=15.0)
    parser.add_argument("--publish-period-s", type=_positive_float, default=0.02)
    parser.add_argument("--close-timeout-s", type=_positive_float, default=6.0)
    parser.add_argument("--display")
    parser.add_argument("--xauthority")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        display, xauthority = _configure_x11_environment(
            display=args.display,
            xauthority=args.xauthority,
        )
        import mujoco

        report = run_lifecycle_smoke(
            mujoco_api=mujoco,
            config=SimConfig.from_file(args.config),
            display=display,
            xauthority=xauthority,
            cycles=args.cycles,
            visible_seconds=args.visible_seconds,
            startup_timeout_s=args.startup_timeout_s,
            publish_period_s=args.publish_period_s,
            close_timeout_s=args.close_timeout_s,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
