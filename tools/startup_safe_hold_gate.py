#!/usr/bin/env python3
"""Fail-closed, read-only startup proof for the simulation SAFE_HOLD.

The gate has no control client and no configurable network destination.  It
performs HTTP GET requests only against the Plant, SONIC, and Robot Agent
loopback diagnostics resources, then requires two consecutive coherent
snapshots before the workstation launcher may report success.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import math
import sys
import time
from typing import Any
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import motion_acceptance_monitor as acceptance


PLANT_URL = "http://127.0.0.1:8794/state"
SONIC_URL = "http://127.0.0.1:8792/telemetry"
ROBOT_AGENT_URL = "http://127.0.0.1:8796/v1/diagnostics"
ENDPOINTS = (PLANT_URL, SONIC_URL, ROBOT_AGENT_URL)
MAX_DEADLINE_S = 30.0
REQUIRED_CONSECUTIVE_SAMPLES = 2
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_SOURCE_AGE_S = 5.0
MAX_SOURCE_FUTURE_S = 1.0

_REQUIRED_ROBOT_AGENT_CHECKS = frozenset(
    {
        "runtime",
        "transportSession",
        "control",
        "safeStop",
        "safetyEndpoints",
        "rosExecutor",
        "nav2Actions",
        "localization",
        "observation",
        "motionSafetyFault",
    }
)

StabilityKey = tuple[int, int, str, int]
StartupSample = tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    float,
    float,
]


@dataclass(frozen=True, slots=True)
class StartupSafeHoldVerdict:
    ready: bool
    sample_count: int
    stability_key: StabilityKey | None
    healthy_settled_safe_hold: bool
    motion_failure_codes: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StartupGateResult:
    passed: bool
    attempts: int
    consecutive_samples: int
    elapsed_s: float
    stability_key: StabilityKey | None
    errors: tuple[str, ...]


class _RejectRedirects(HTTPRedirectHandler):
    """Keep a fixed loopback GET from being redirected elsewhere."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _append_once(errors: list[str], detail: str) -> None:
    if detail not in errors:
        errors.append(detail)


def _nonfinite_paths(value: Any, path: str) -> list[str]:
    result: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        result.append(path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(_nonfinite_paths(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(_nonfinite_paths(item, f"{path}[{index}]"))
    return result


def _integer_marker(value: Any, name: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{name} is not a positive integer")
        return None
    return value


def _finite_marker(value: Any, name: str, errors: list[str]) -> float | None:
    if isinstance(value, bool):
        errors.append(f"{name} is not finite")
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{name} is not finite")
        return None
    if not math.isfinite(result):
        errors.append(f"{name} is not finite")
        return None
    return result


def _observed_at_marker(value: Any, errors: list[str]) -> float | None:
    if not isinstance(value, str) or not value.strip():
        errors.append("Robot Agent observedAt is invalid")
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        observed = datetime.fromisoformat(normalized)
        if observed.tzinfo is None or observed.utcoffset() is None:
            raise ValueError("timezone is missing")
        result = observed.timestamp()
    except (ValueError, OverflowError, OSError):
        errors.append("Robot Agent observedAt is invalid")
        return None
    if not math.isfinite(result):
        errors.append("Robot Agent observedAt is invalid")
        return None
    return result


def _source_markers(
    plant: Mapping[str, Any],
    sonic: Mapping[str, Any],
    robot_agent: Mapping[str, Any],
    *,
    observed_wall_s: float,
) -> tuple[dict[str, int | float | str], tuple[str, ...]]:
    """Validate absolute freshness and return cross-sample progress markers."""

    errors: list[str] = []
    wall_s = _finite_marker(observed_wall_s, "gate wall observation time", errors)
    producer_boot_id = plant.get("producerBootId")
    if not isinstance(producer_boot_id, str) or not producer_boot_id.strip():
        errors.append("Plant producerBootId is invalid")

    sequence = _integer_marker(plant.get("sequence"), "Plant sequence", errors)
    captured_at_ns = _integer_marker(
        plant.get("capturedAtUnixNs"),
        "Plant capturedAtUnixNs",
        errors,
    )
    sonic_updated_at = _finite_marker(
        sonic.get("updated_at"),
        "SONIC updated_at",
        errors,
    )
    plant_tick = _integer_marker(sonic.get("plant_tick"), "SONIC plant_tick", errors)
    control_tick = _integer_marker(
        sonic.get("control_tick_ms"),
        "SONIC control_tick_ms",
        errors,
    )
    agent_observed_at = _observed_at_marker(robot_agent.get("observedAt"), errors)

    source_times = {
        "Plant capturedAtUnixNs": (
            None if captured_at_ns is None else captured_at_ns / 1_000_000_000.0
        ),
        "SONIC updated_at": sonic_updated_at,
        "Robot Agent observedAt": agent_observed_at,
    }
    if wall_s is not None:
        for name, source_s in source_times.items():
            if source_s is None:
                continue
            age_s = wall_s - source_s
            if age_s > MAX_SOURCE_AGE_S:
                errors.append(
                    f"{name} is stale by {age_s:.3f}s; maximum is "
                    f"{MAX_SOURCE_AGE_S:.1f}s"
                )
            elif age_s < -MAX_SOURCE_FUTURE_S:
                errors.append(
                    f"{name} is {-age_s:.3f}s in the future; maximum is "
                    f"{MAX_SOURCE_FUTURE_S:.1f}s"
                )

    if errors:
        return {}, tuple(errors)
    assert isinstance(producer_boot_id, str)
    assert sequence is not None
    assert captured_at_ns is not None
    assert sonic_updated_at is not None
    assert plant_tick is not None
    assert control_tick is not None
    assert agent_observed_at is not None
    return {
        "plant.producerBootId": producer_boot_id,
        "plant.sequence": sequence,
        "plant.capturedAtUnixNs": captured_at_ns,
        "sonic.updated_at": sonic_updated_at,
        "sonic.plant_tick": plant_tick,
        "sonic.control_tick_ms": control_tick,
        "robotAgent.observedAt": agent_observed_at,
    }, ()


def _robot_agent_errors(value: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if value.get("schemaVersion") != "robot-agent.diagnostics/1":
        errors.append("unsupported Robot Agent diagnostics schema")
    robot_id = value.get("robotId")
    if not isinstance(robot_id, str) or not robot_id.strip():
        errors.append("Robot Agent robotId is not a non-empty string")
    if value.get("ready") is not True:
        errors.append("Robot Agent ready is not true")

    command = value.get("commandReadiness")
    control = value.get("controlState")
    fault = value.get("motionSafetyFault")
    checks = value.get("checks")
    if not isinstance(command, Mapping):
        errors.append("Robot Agent commandReadiness is missing")
        command = {}
    if not isinstance(control, Mapping):
        errors.append("Robot Agent controlState is missing")
        control = {}
    if not isinstance(fault, Mapping):
        errors.append("Robot Agent motionSafetyFault is missing")
        fault = {}
    if not isinstance(checks, Mapping):
        errors.append("Robot Agent checks are missing")
        checks = {}

    if command.get("ready") is not True or command.get("mode") != "STANDBY":
        errors.append("Robot Agent command readiness is not ready STANDBY")

    expected_control = {
        "state": "SAFE_HOLD",
        "active": False,
        "standby": True,
        "canAcquire": True,
        "authorityRevoked": True,
        "safeStopState": "confirmed",
        "leaseValid": False,
    }
    for field, expected in expected_control.items():
        actual = control.get(field)
        if actual != expected or (
            isinstance(expected, bool) and actual is not expected
        ):
            errors.append(
                f"Robot Agent controlState.{field}={actual!r}, "
                f"expected {expected!r}"
            )
    if control.get("leaseRemainingMs") is not None:
        errors.append("Robot Agent leaseRemainingMs is not null in SAFE_HOLD")
    active_commands = control.get("activeCommandCount")
    if (
        isinstance(active_commands, bool)
        or not isinstance(active_commands, int)
        or active_commands != 0
    ):
        errors.append("Robot Agent activeCommandCount is not integer zero")

    boot_id = value.get("robotBootId")
    if not isinstance(boot_id, str) or not boot_id.strip():
        errors.append("Robot Agent robotBootId is invalid")
    safe_stop_epoch = control.get("safeStopEpoch")
    if (
        isinstance(safe_stop_epoch, bool)
        or not isinstance(safe_stop_epoch, int)
        or safe_stop_epoch <= 0
    ):
        errors.append("Robot Agent safeStopEpoch is invalid")

    if fault.get("ready") is not True or fault.get("latched") is not False:
        errors.append("Robot Agent motion-safety fault is not clear")
    for field in ("eventId", "source", "code"):
        if fault.get(field) is not None:
            errors.append(f"Robot Agent motionSafetyFault.{field} is not null")

    missing_checks = sorted(_REQUIRED_ROBOT_AGENT_CHECKS.difference(checks))
    if missing_checks:
        errors.append("Robot Agent checks are missing: " + ", ".join(missing_checks))
    for name in sorted(_REQUIRED_ROBOT_AGENT_CHECKS.intersection(checks)):
        check = checks[name]
        if not isinstance(check, Mapping):
            errors.append(f"Robot Agent required profile check {name} is not an object")
        elif check.get("required") is not True or check.get("ready") is not True:
            errors.append(
                f"Robot Agent required profile check {name} is not required and ready"
            )
    for name, check in checks.items():
        if not isinstance(check, Mapping):
            errors.append(f"Robot Agent check {name} is not an object")
        elif check.get("required") is True and check.get("ready") is not True:
            errors.append(f"Robot Agent required check {name} is not ready")

    nonfinite = _nonfinite_paths(value, "robotAgent")
    if nonfinite:
        errors.append("non-finite Robot Agent values: " + ", ".join(nonfinite[:8]))
    return tuple(errors)


def _stability_key(
    plant: Mapping[str, Any],
    sonic: Mapping[str, Any],
    robot_agent: Mapping[str, Any],
) -> tuple[StabilityKey | None, tuple[str, ...]]:
    errors: list[str] = []
    support = plant.get("support")
    control = robot_agent.get("controlState")
    if not isinstance(support, Mapping):
        errors.append("Plant support is missing")
        support = {}
    if not isinstance(control, Mapping):
        errors.append("Robot Agent controlState is missing")
        control = {}

    plant_robot_id = plant.get("robotId")
    sonic_robot_id = sonic.get("robot_id")
    agent_robot_id = robot_agent.get("robotId")
    identities = (plant_robot_id, sonic_robot_id, agent_robot_id)
    if not all(isinstance(value, str) and bool(value.strip()) for value in identities):
        errors.append("Plant, SONIC, and Robot Agent robot IDs must be non-empty strings")
    elif not (plant_robot_id == sonic_robot_id == agent_robot_id):
        errors.append(
            "Plant, SONIC, and Robot Agent robot IDs do not match: "
            f"{identities!r}"
        )

    plant_epoch = support.get("safetyStopEpoch")
    sonic_generation = sonic.get("plant_support_stop_generation")
    robot_boot_id = robot_agent.get("robotBootId")
    agent_epoch = control.get("safeStopEpoch")
    if (
        isinstance(plant_epoch, bool)
        or not isinstance(plant_epoch, int)
        or plant_epoch <= 0
    ):
        errors.append("Plant safetyStopEpoch is not a positive integer")
    if (
        isinstance(sonic_generation, bool)
        or not isinstance(sonic_generation, int)
        or sonic_generation < 0
    ):
        errors.append("SONIC plant_support_stop_generation is invalid")
    if not isinstance(robot_boot_id, str) or not robot_boot_id.strip():
        errors.append("Robot Agent robotBootId is invalid")
    if (
        isinstance(agent_epoch, bool)
        or not isinstance(agent_epoch, int)
        or agent_epoch <= 0
    ):
        errors.append("Robot Agent safeStopEpoch is invalid")
    if errors:
        return None, tuple(errors)
    assert isinstance(plant_epoch, int)
    assert isinstance(sonic_generation, int)
    assert isinstance(robot_boot_id, str)
    assert isinstance(agent_epoch, int)
    return (
        plant_epoch,
        sonic_generation,
        robot_boot_id,
        agent_epoch,
    ), ()


def _motion_observation(
    evaluator: acceptance.MotionAcceptanceEvaluator,
    plant: Mapping[str, Any],
    sonic: Mapping[str, Any],
    *,
    observed_s: float,
) -> tuple[bool, tuple[str, ...]]:
    try:
        derived = evaluator.observe(plant, sonic, observed_s=observed_s)
    except Exception as exc:  # A validator bug or alien payload must fail closed.
        return False, (f"MotionAcceptanceEvaluator raised: {exc}",)
    healthy = derived.get("healthy_settled_safe_hold") is True
    errors = [
        f"{failure.code}: {failure.detail}"
        for failure in evaluator.failures.values()
    ]
    if not healthy:
        _append_once(
            errors,
            "MotionAcceptanceEvaluator did not derive healthy_settled_safe_hold",
        )
    return healthy, tuple(errors)


def evaluate_startup_snapshot(
    plant: Mapping[str, Any],
    sonic: Mapping[str, Any],
    robot_agent: Mapping[str, Any],
    *,
    observed_s: float = 0.0,
    observed_wall_s: float,
) -> StartupSafeHoldVerdict:
    """Purely validate one three-endpoint observation group."""

    evaluator = acceptance.MotionAcceptanceEvaluator(
        acceptance.AcceptanceLimits(phase="idle")
    )
    healthy, motion_errors = _motion_observation(
        evaluator,
        plant,
        sonic,
        observed_s=observed_s,
    )
    agent_errors = _robot_agent_errors(robot_agent)
    key, key_errors = _stability_key(plant, sonic, robot_agent)
    _markers, source_errors = _source_markers(
        plant,
        sonic,
        robot_agent,
        observed_wall_s=observed_wall_s,
    )
    errors = tuple(
        (*motion_errors, *agent_errors, *key_errors, *source_errors)
    )
    return StartupSafeHoldVerdict(
        ready=healthy and not evaluator.failures and not errors and key is not None,
        sample_count=1,
        stability_key=key,
        healthy_settled_safe_hold=healthy,
        motion_failure_codes=tuple(evaluator.failures),
        errors=errors,
    )


def evaluate_startup_sequence(
    samples: Sequence[StartupSample],
) -> StartupSafeHoldVerdict:
    """Validate a consecutive candidate sequence with one complete evaluator."""

    evaluator = acceptance.MotionAcceptanceEvaluator(
        acceptance.AcceptanceLimits(phase="idle")
    )
    errors: list[str] = []
    keys: list[StabilityKey] = []
    source_markers: list[dict[str, int | float | str]] = []
    all_healthy = True
    for plant, sonic, robot_agent, observed_s, observed_wall_s in samples:
        healthy, motion_errors = _motion_observation(
            evaluator,
            plant,
            sonic,
            observed_s=observed_s,
        )
        all_healthy = all_healthy and healthy
        for detail in motion_errors:
            _append_once(errors, detail)
        for detail in _robot_agent_errors(robot_agent):
            _append_once(errors, detail)
        key, key_errors = _stability_key(plant, sonic, robot_agent)
        for detail in key_errors:
            _append_once(errors, detail)
        if key is not None:
            keys.append(key)
        markers, source_errors = _source_markers(
            plant,
            sonic,
            robot_agent,
            observed_wall_s=observed_wall_s,
        )
        for detail in source_errors:
            _append_once(errors, detail)
        if markers:
            source_markers.append(markers)

    if len(samples) < REQUIRED_CONSECUTIVE_SAMPLES:
        errors.append(
            f"at least {REQUIRED_CONSECUTIVE_SAMPLES} consecutive samples are required"
        )
    if len(keys) != len(samples) or (keys and any(key != keys[0] for key in keys[1:])):
        errors.append("startup SAFE_HOLD stability identity changed between samples")
    if len(source_markers) == len(samples) and source_markers:
        producer_boot_id = source_markers[0]["plant.producerBootId"]
        if any(
            markers["plant.producerBootId"] != producer_boot_id
            for markers in source_markers[1:]
        ):
            errors.append("Plant producerBootId changed between startup samples")
        for name in (
            "plant.sequence",
            "plant.capturedAtUnixNs",
            "sonic.updated_at",
            "sonic.plant_tick",
            "sonic.control_tick_ms",
            "robotAgent.observedAt",
        ):
            for previous, current in zip(source_markers, source_markers[1:]):
                if current[name] <= previous[name]:
                    errors.append(
                        f"{name} did not strictly advance between startup samples"
                    )
                    break
    if len(samples) >= REQUIRED_CONSECUTIVE_SAMPLES:
        finalized = evaluator.finalize()
        for failure in finalized["failures"]:
            _append_once(
                errors,
                f"{failure['code']}: {failure['detail']}",
            )

    stable_key = keys[0] if len(keys) == len(samples) and keys else None
    return StartupSafeHoldVerdict(
        ready=(
            len(samples) >= REQUIRED_CONSECUTIVE_SAMPLES
            and all_healthy
            and not evaluator.failures
            and not errors
            and stable_key is not None
        ),
        sample_count=len(samples),
        stability_key=stable_key,
        healthy_settled_safe_hold=all_healthy,
        motion_failure_codes=tuple(evaluator.failures),
        errors=tuple(errors),
    )


class ReadonlyStartupSampler:
    """Fetch the three fixed loopback resources using GET and no body."""

    def __init__(
        self,
        *,
        timeout_s: float = 1.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if (
            isinstance(timeout_s, bool)
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0.0
        ):
            raise ValueError("HTTP timeout must be finite and positive")
        self.timeout_s = float(timeout_s)
        self._opener = opener or build_opener(
            ProxyHandler({}),
            _RejectRedirects(),
        ).open

    def _get_json(self, url: str, *, timeout_s: float) -> dict[str, Any]:
        request = Request(
            url,
            method="GET",
            headers={"Accept": "application/json", "Cache-Control": "no-store"},
        )
        with self._opener(request, timeout=timeout_s) as response:
            final_url_reader = getattr(response, "geturl", None)
            final_url = final_url_reader() if callable(final_url_reader) else url
            if final_url != url:
                raise RuntimeError(
                    f"GET {url} was redirected to forbidden target {final_url}"
                )
            status_value = getattr(response, "status", None)
            if status_value is None:
                status_value = response.getcode()
            if int(status_value) != 200:
                raise RuntimeError(f"GET {url} returned HTTP {status_value}")
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise RuntimeError(f"GET {url} returned an oversized response")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise RuntimeError(f"GET {url} did not return a JSON object")
        return payload

    def sample(
        self,
        *,
        deadline_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for url in ENDPOINTS:
            remaining_s = deadline_s - clock()
            if remaining_s <= 0.0:
                raise TimeoutError("startup SAFE_HOLD deadline expired")
            payloads.append(
                self._get_json(
                    url,
                    timeout_s=min(self.timeout_s, remaining_s),
                )
            )
        return payloads[0], payloads[1], payloads[2]


def _bounded_positive(value: float, *, name: str, maximum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must not exceed {maximum:g}s")
    return result


def wait_for_startup_safe_hold(
    *,
    sampler: ReadonlyStartupSampler,
    deadline_s: float = MAX_DEADLINE_S,
    interval_s: float = 0.5,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
) -> StartupGateResult:
    """Poll until two coherent groups pass, without exceeding 30 seconds."""

    deadline_s = _bounded_positive(
        deadline_s,
        name="deadline_s",
        maximum=MAX_DEADLINE_S,
    )
    interval_s = _bounded_positive(interval_s, name="interval_s")
    started_s = clock()
    hard_deadline_s = started_s + deadline_s
    attempts = 0
    candidate_key: StabilityKey | None = None
    candidates: list[StartupSample] = []
    last_errors: tuple[str, ...] = ("no endpoint group was observed",)

    while clock() < hard_deadline_s:
        attempts += 1
        try:
            plant, sonic, robot_agent = sampler.sample(
                deadline_s=hard_deadline_s,
                clock=clock,
            )
        except Exception as exc:
            candidate_key = None
            candidates.clear()
            last_errors = (f"read-only endpoint sample failed: {exc}",)
        else:
            observed_s = clock()
            observed_wall_s = wall_clock()
            if observed_s > hard_deadline_s:
                last_errors = ("startup SAFE_HOLD deadline expired during sampling",)
                break
            point = evaluate_startup_snapshot(
                plant,
                sonic,
                robot_agent,
                observed_s=observed_s,
                observed_wall_s=observed_wall_s,
            )
            if point.ready and point.stability_key is not None:
                if point.stability_key != candidate_key:
                    candidate_key = point.stability_key
                    candidates.clear()
                candidates.append(
                    (
                        plant,
                        sonic,
                        robot_agent,
                        observed_s,
                        observed_wall_s,
                    )
                )
                candidates[:] = candidates[-REQUIRED_CONSECUTIVE_SAMPLES:]
                last_errors = point.errors
                if len(candidates) == REQUIRED_CONSECUTIVE_SAMPLES:
                    sequence = evaluate_startup_sequence(candidates)
                    if sequence.ready and clock() <= hard_deadline_s:
                        return StartupGateResult(
                            passed=True,
                            attempts=attempts,
                            consecutive_samples=len(candidates),
                            elapsed_s=max(0.0, clock() - started_s),
                            stability_key=sequence.stability_key,
                            errors=(),
                        )
                    # The current point remains a valid first candidate even if
                    # the preceding observation rolled back or jumped.
                    candidates[:] = candidates[-1:]
                    last_errors = sequence.errors
            else:
                candidate_key = None
                candidates.clear()
                last_errors = point.errors

        remaining_s = hard_deadline_s - clock()
        if remaining_s <= 0.0:
            break
        sleeper(min(interval_s, remaining_s))

    return StartupGateResult(
        passed=False,
        attempts=attempts,
        consecutive_samples=len(candidates),
        elapsed_s=max(0.0, clock() - started_s),
        stability_key=candidate_key,
        errors=last_errors,
    )


def _positive_cli(value: str) -> float:
    try:
        return _bounded_positive(float(value), name="value")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _deadline_cli(value: str) -> float:
    try:
        return _bounded_positive(
            float(value),
            name="deadline",
            maximum=MAX_DEADLINE_S,
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only startup SAFE_HOLD gate. Uses fixed loopback HTTP GET "
            "resources only; never sends control, stop, rearm, motion, or reset."
        )
    )
    parser.add_argument("--deadline-seconds", type=_deadline_cli, default=30.0)
    parser.add_argument("--interval-seconds", type=_positive_cli, default=0.5)
    parser.add_argument("--http-timeout-seconds", type=_positive_cli, default=1.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        sampler = ReadonlyStartupSampler(timeout_s=args.http_timeout_seconds)
        result = wait_for_startup_safe_hold(
            sampler=sampler,
            deadline_s=args.deadline_seconds,
            interval_s=args.interval_seconds,
        )
    except (OSError, ValueError) as exc:
        print(f"startup SAFE_HOLD gate setup failed: {exc}", file=sys.stderr)
        return 2

    key = None if result.stability_key is None else list(result.stability_key)
    if result.passed:
        print(
            "PASS startup SAFE_HOLD "
            f"samples={result.consecutive_samples} attempts={result.attempts} "
            f"elapsed={result.elapsed_s:.2f}s stabilityKey={json.dumps(key)}"
        )
        return 0
    print(
        "FAIL startup SAFE_HOLD "
        f"attempts={result.attempts} elapsed={result.elapsed_s:.2f}s "
        f"lastErrors={json.dumps(list(result.errors))}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
