from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache

from hide_mujoco_sonic.sonic_driver import SonicDriver
from lerobot.robots.unitree_g1.controllers import sonic_whole_body
from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
    CONTROL_DT,
    DEFAULT_ANGLES,
    LM,
    hardware_target_to_normalized_action,
)
from lerobot.robots.unitree_g1.controllers.sonic_whole_body import (
    SonicWholeBodyController,
)
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex


_MODEL_REPO = "nvidia/GEAR-SONIC"
_MODEL_REVISION = "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"
_MODEL_FILES = (
    "planner_sonic.onnx",
    "model_encoder.onnx",
    "model_decoder.onnx",
)


def _cached_model_paths() -> dict[str, Path]:
    """Resolve a complete local SONIC bundle without permitting a download."""

    paths: dict[str, Path] = {}
    for filename in _MODEL_FILES:
        cached = try_to_load_from_cache(
            _MODEL_REPO,
            filename,
            revision=_MODEL_REVISION,
        )
        if not isinstance(cached, str):
            pytest.skip(f"cached {_MODEL_REPO}/{filename} is unavailable")
        path = Path(cached)
        if not path.is_file():
            pytest.skip(f"cached {_MODEL_REPO}/{filename} is incomplete")
        paths[filename] = path
    return paths


def _nominal_lowstate() -> SimpleNamespace:
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=float(target), dq=0.0, tau_est=0.0)
            for target in DEFAULT_ANGLES
        ],
        imu_state=SimpleNamespace(
            gyroscope=[0.0, 0.0, 0.0],
            accelerometer=[0.0, 0.0, -9.81],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            rpy=[0.0, 0.0, 0.0],
        ),
        wireless_remote=b"",
    )


def _targets(action: dict[str, float]) -> tuple[float, ...]:
    return tuple(
        float(action[f"{joint.name}.q"])
        for joint in G1_29_JointIndex
    )


def _project_complete_target(
    raw: tuple[float, ...],
    projected_arm: tuple[float, ...],
) -> tuple[float, ...]:
    assert len(raw) == 29
    assert len(projected_arm) == 14
    return raw[:15] + projected_arm


def _track_target(
    lowstate: SimpleNamespace,
    targets: tuple[float, ...],
    *,
    alpha: float,
) -> None:
    """Apply one conservative first-order measured tracking observation."""

    for state, target in zip(lowstate.motor_state, targets, strict=True):
        previous = float(state.q)
        measured = previous + alpha * (float(target) - previous)
        state.q = measured
        state.dq = (measured - previous) / CONTROL_DT


def _wait_for_matching_reference(
    controller: SonicWholeBodyController,
) -> None:
    deadline = time.monotonic() + 20.0
    while not controller.poll_reference():
        if time.monotonic() >= deadline:
            raise TimeoutError("cached SONIC planner did not return within 20s")
        time.sleep(0.005)


def test_cached_real_sonic_first_policy_target_is_slew_projected_and_rebased(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real cached planner/encoder/decoder without network I/O."""

    pytest.importorskip("onnxruntime")
    cached_paths = _cached_model_paths()

    def cache_only_download(*, repo_id: str, filename: str, **_kwargs) -> str:
        assert repo_id == _MODEL_REPO
        try:
            return str(cached_paths[filename])
        except KeyError as exc:
            raise AssertionError(
                f"unexpected SONIC model download request: {filename}"
            ) from exc

    # SonicRuntime normally calls hf_hub_download during construction. Replace
    # that symbol with a complete cache resolver so this acceptance test can
    # neither contact the network nor mutate the local Hugging Face cache.
    monkeypatch.setattr(
        sonic_whole_body,
        "hf_hub_download",
        cache_only_download,
    )

    cases = (
        ("straight", 0.25, 0.0, 0.0, True, False),
        ("turn", -1.0, 0.0, 0.15, False, False),
        ("diagonal_30deg", 0.20, math.radians(30.0), 0.0, True, True),
        ("lateral_90deg", 0.20, math.pi / 2.0, 0.0, True, True),
    )
    nominal = tuple(float(value) for value in DEFAULT_ANGLES)
    guard_owner = SonicDriver.__new__(SonicDriver)
    controller = SonicWholeBodyController(
        force_cpu=True,
        auto_reference_authority=False,
        enable_startup_blend=False,
        allow_operator_intent=False,
    )
    try:
        for case_index, (
            label,
            speed,
            movement_angle,
            facing_angle,
            has_movement,
            expects_projection,
        ) in enumerate(cases):
            if case_index:
                controller.reset()
            lowstate = _nominal_lowstate()
            assert controller.initialize_from_lowstate(lowstate) is True
            controller.ms.mode = LM.SLOW_WALK
            controller.ms.speed = speed
            controller.ms.movement_angle = movement_angle
            controller.ms.facing_angle = facing_angle
            controller.ms.has_movement = has_movement
            controller.ms.needs_replan = False
            controller.request_reference(lowstate)
            _wait_for_matching_reference(controller)
            controller.acquire_reference(
                lowstate,
                executed_target=nominal,
                expected_install_epoch=controller.reference_install_epoch,
            )

            # The guard factory depends only on the checked-in navigation
            # profile constants; avoid constructing a live DDS driver.
            guard = guard_owner._new_arm_guard()
            guard.anchor(nominal[15:29])
            raw = _targets(controller.run_step({}, lowstate))
            first = guard.apply(
                raw[15:29],
                measured=nominal[15:29],
                dt=CONTROL_DT,
            )
            projected = _project_complete_target(raw, first.targets)

            if expects_projection:
                assert first.constrained_joints > 0, label
            else:
                assert first.constrained_joints == 0, label
            for offset, (actual, anchor, velocity) in enumerate(
                zip(
                    first.targets,
                    nominal[15:29],
                    guard.max_velocity_rps,
                    strict=True,
                )
            ):
                assert abs(actual - anchor) <= velocity * CONTROL_DT + 1e-12, (
                    label,
                    offset,
                )

            executed = controller.record_executed_target(projected)
            np.testing.assert_array_equal(
                executed,
                hardware_target_to_normalized_action(projected),
            )
            assert controller.execution_feedback_pending is False

            if not expects_projection:
                continue

            # A reasonable first-order plant response makes the real policy
            # consume its exact projected history and leave saturation long
            # before the guard's 0.25 s fail-closed budget is exhausted.
            _track_target(lowstate, projected, alpha=0.10)
            cleared_after_frames: int | None = None
            maximum_safe_frames = math.floor(
                guard.max_continuous_saturation_s / CONTROL_DT
            )
            for frame in range(1, maximum_safe_frames):
                raw = _targets(controller.run_step({}, lowstate))
                decision = guard.apply(
                    raw[15:29],
                    measured=tuple(
                        float(state.q) for state in lowstate.motor_state[15:29]
                    ),
                    dt=CONTROL_DT,
                )
                projected = _project_complete_target(raw, decision.targets)
                controller.record_executed_target(projected)
                _track_target(lowstate, projected, alpha=0.10)
                if decision.constrained_joints == 0:
                    cleared_after_frames = frame
                    break

            assert cleared_after_frames is not None, label
            assert (
                (cleared_after_frames + 1) * CONTROL_DT
                < guard.max_continuous_saturation_s
            ), label
    finally:
        controller.shutdown()
