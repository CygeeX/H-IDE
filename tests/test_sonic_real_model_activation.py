from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
from huggingface_hub import try_to_load_from_cache

from hide_mujoco_sonic.plant import stabilized_support_wrench
from hide_mujoco_sonic.safety import CmdVelGuard
from hide_mujoco_sonic.sonic_driver import SonicDriver
from lerobot.robots.unitree_g1.controllers import sonic_whole_body
from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
    CONTROL_DT,
    DEFAULT_ANGLES,
    LM,
    compute_kp_kd,
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


def _measured_lowstate(
    q: np.ndarray,
    dq: np.ndarray,
    quaternion: np.ndarray,
    *,
    gyroscope: np.ndarray | None = None,
) -> SimpleNamespace:
    gyro = np.zeros(3, dtype=np.float64) if gyroscope is None else gyroscope
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=float(position), dq=float(velocity), tau_est=0.0)
            for position, velocity in zip(q, dq, strict=True)
        ],
        imu_state=SimpleNamespace(
            gyroscope=[float(value) for value in gyro],
            accelerometer=[0.0, 0.0, -9.81],
            quaternion=[float(value) for value in quaternion],
            # The pinned MuJoCo Unitree bridge does not populate this legacy
            # field.  Production stability checks derive roll/pitch from the
            # authoritative quaternion instead.
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
        # Official first-CONTROL history is zero padded.  Its cold decoder
        # output can require the existing bounded arm projection for every
        # motion class; exact executed feedback must make that transient clear.
        ("straight", 0.25, 0.0, 0.0, True, True),
        ("turn", -1.0, 0.0, 0.15, False, True),
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
            # consume its exact projected history and leave saturation.  The
            # production guard records duration but does not invent a vendor-
            # unsupported time-to-fault.
            _track_target(lowstate, projected, alpha=0.10)
            cleared_after_frames: int | None = None
            for frame in range(1, 50):
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
            assert cleared_after_frames < 49, label
    finally:
        controller.shutdown()


def _yaw_from_wxyz(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = (float(value) for value in quaternion)
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def _roll_pitch_from_wxyz(quaternion: np.ndarray) -> tuple[float, float]:
    qw, qx, qy, qz = (float(value) for value in quaternion)
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    )
    return roll, pitch


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@pytest.mark.parametrize(
    (
        "label",
        "speed",
        "movement_angle",
        "facing_angle",
        "has_movement",
        "expected_translation",
        "expected_yaw_sign",
        "target_magnitude",
        "motion_timeout_s",
    ),
    (
        # Exact direction-dependent SLOW_WALK speeds from NVIDIA's pinned
        # ROS2 input handler: forward=.30, lateral=.35, backward=.20 m/s.
        ("forward_1m", 0.30, 0.0, 0.0, True, (1.0, 0.0), 0, 1.0, 30.0),
        ("forward_3m", 0.30, 0.0, 0.0, True, (1.0, 0.0), 0, 3.0, 60.0),
        ("reverse", 0.20, math.pi, 0.0, True, (-1.0, 0.0), 0, 0.5, 30.0),
        (
            "reverse_early_stop",
            0.20,
            math.pi,
            0.0,
            True,
            (-1.0, 0.0),
            0,
            0.20,
            15.0,
        ),
        (
            "left_turn",
            -1.0,
            0.0,
            math.pi / 2.0,
            False,
            (0.0, 0.0),
            1,
            math.pi / 2.0,
            20.0,
        ),
        (
            "right_turn",
            -1.0,
            0.0,
            -math.pi / 2.0,
            False,
            (0.0, 0.0),
            -1,
            math.pi / 2.0,
            20.0,
        ),
        (
            "heading_hold_left_strafe",
            0.40,
            math.pi / 2.0,
            0.0,
            True,
            (0.0, 1.0),
            0,
            0.5,
            30.0,
        ),
        (
            "heading_hold_right_strafe",
            0.40,
            -math.pi / 2.0,
            0.0,
            True,
            (0.0, -1.0),
            0,
            0.5,
            30.0,
        ),
        (
            "heading_hold_forward_1m",
            0.30,
            0.0,
            0.0,
            True,
            (1.0, 0.0),
            0,
            1.0,
            30.0,
        ),
    ),
)
def test_official_cold_boot_sequence_moves_in_requested_direction(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    speed: float,
    movement_angle: float,
    facing_angle: float,
    has_movement: bool,
    expected_translation: tuple[float, float],
    expected_yaw_sign: int,
    target_magnitude: float,
    motion_timeout_s: float,
) -> None:
    """Prove the official INIT -> WAIT -> CONTROL flow in real MuJoCo.

    The elastic band is present only during NVIDIA's 3 s INIT. It is fully
    released before reference acquisition and CONTROL, matching the official
    quickstart rather than using vertical or attitude assistance as evidence.
    """

    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    cached_paths = _cached_model_paths()

    def cache_only_download(*, repo_id: str, filename: str, **_kwargs) -> str:
        assert repo_id == _MODEL_REPO
        return str(cached_paths[filename])

    monkeypatch.setattr(
        sonic_whole_body,
        "hf_hub_download",
        cache_only_download,
    )

    model_path = (
        Path(__file__).resolve().parents[1]
        / "scenes/office_v2/generated/mujoco/office_v2.runtime.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    measured_q = np.asarray(DEFAULT_ANGLES, dtype=np.float64)
    data.qpos[:3] = (-6.45, 0.0, float(model.qpos0[2]))
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    data.qpos[7:] = measured_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    torso_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "torso_link",
    )
    support_anchor = np.asarray(data.xpos[torso_id], dtype=np.float64).copy()

    def lowstate() -> SimpleNamespace:
        return _measured_lowstate(
            np.asarray(data.qpos[7:], dtype=np.float64),
            np.asarray(data.qvel[6:], dtype=np.float64),
            np.asarray(data.qpos[3:7], dtype=np.float64),
            gyroscope=np.asarray(data.qvel[3:6], dtype=np.float64),
        )

    nominal = tuple(float(value) for value in DEFAULT_ANGLES)
    kp, kd = (
        np.asarray(values, dtype=np.float64)
        for values in compute_kp_kd()
    )
    guard = SonicDriver.__new__(SonicDriver)._new_arm_guard()
    guard.anchor(tuple(float(value) for value in measured_q[15:29]))
    published = measured_q.copy()
    controller = SonicWholeBodyController(
        force_cpu=True,
        auto_reference_authority=False,
        enable_startup_blend=False,
        allow_operator_intent=False,
    )
    heading_hold_enabled = label.startswith("heading_hold_")
    heading_guard: CmdVelGuard | None = None
    heading_intent = None
    heading_hold_applied_frames = 0
    try:
        assert controller.initialize_from_lowstate(lowstate()) is True
        controller.ms.mode = LM.IDLE
        controller.ms.speed = -1.0
        controller.ms.movement_angle = 0.0
        controller.ms.facing_angle = 0.0
        controller.ms.has_movement = False
        controller.ms.needs_replan = False
        command_sent = False
        reference_requested = False
        reference_acquired = False
        expected_reference_epoch = controller.reference_install_epoch + 1
        motion_initial_xy: np.ndarray | None = None
        motion_initial_yaw: float | None = None
        reference_installed_s: float | None = None
        requested_projections: list[float] = []
        target_reached = False
        turn_reference_refreshes = 0
        minimum_z = float(data.qpos[2])
        peak_saturation_s = 0.0
        constrained_control_frames = 0
        last_constrained_control_frame = -1
        physics_steps = round(
            (3.5 + motion_timeout_s) / float(model.opt.timestep)
        )
        control_every_float = CONTROL_DT / float(model.opt.timestep)
        assert control_every_float == pytest.approx(round(control_every_float))
        control_every = round(control_every_float)
        assert control_every > 0
        for physics_tick in range(physics_steps):
            if physics_tick % control_every == 0:
                simulation_now_s = float(data.time)
                if simulation_now_s <= 3.0 + 1e-12:
                    ratio = min(1.0, simulation_now_s / 3.0)
                    raw = tuple(
                        float(current) * (1.0 - ratio) + target * ratio
                        for current, target in zip(
                            data.qpos[7:],
                            nominal,
                            strict=True,
                        )
                    )
                    arm = guard.apply(
                        raw[15:29],
                        measured=tuple(float(value) for value in data.qpos[22:36]),
                        dt=CONTROL_DT,
                    )
                    published = np.asarray(
                        _project_complete_target(raw, arm.targets),
                        dtype=np.float64,
                    )
                elif not command_sent and simulation_now_s >= 3.20:
                    # NVIDIA WAIT_FOR_CONTROL keeps publishing nominal without
                    # running the policy or mutating its recurrent history.
                    command_sent = True
                    controller.ms.mode = (
                        LM.IDLE if expected_yaw_sign else LM.SLOW_WALK
                    )
                    controller.ms.speed = speed
                    controller.ms.movement_angle = movement_angle
                    controller.ms.facing_angle = facing_angle
                    controller.ms.has_movement = has_movement
                    controller.ms.needs_replan = False
                if command_sent and not reference_requested:
                    controller.request_reference(lowstate())
                    reference_requested = True
                    # Planning is a wall-clock worker. Freeze this deterministic
                    # offline plant at the exact post-release settled LowState
                    # instead of letting CPU scheduling decide how many seconds
                    # of unsupported MuJoCo time pass before installation.
                    _wait_for_matching_reference(controller)
                    controller.acquire_reference(
                        lowstate(),
                        executed_target=tuple(published),
                        expected_install_epoch=controller.reference_install_epoch,
                    )
                    reference_acquired = True
                if reference_acquired and motion_initial_xy is None:
                    assert controller.reference_install_epoch >= expected_reference_epoch
                    motion_initial_xy = np.asarray(data.qpos[:2], dtype=np.float64).copy()
                    motion_initial_yaw = _yaw_from_wxyz(
                        np.asarray(data.qpos[3:7], dtype=np.float64)
                    )
                    reference_installed_s = float(data.time)
                    if heading_hold_enabled:
                        heading_guard = CmdVelGuard(
                            clock=lambda: float(data.time),
                            smoothing_alpha=1.0,
                        )
                        heading_guard.observe_heading(motion_initial_yaw)
                        heading_guard.accept(
                            math.cos(movement_angle) if has_movement else 0.0,
                            math.sin(movement_angle) if has_movement else 0.0,
                            0.0,
                            received_at=float(data.time),
                        )
                        heading_intent = heading_guard.intent(
                            now=float(data.time),
                            dt=CONTROL_DT,
                        )
                if (
                    reference_acquired
                    and expected_yaw_sign
                    and reference_installed_s is not None
                    and turn_reference_refreshes < 2
                    and float(data.time) - reference_installed_s
                    >= 8.0 * (turn_reference_refreshes + 1)
                ):
                    # A planner clip is a bounded prefix, not a complete Spin
                    # action. Rebase from the fresh measured LowState exactly
                    # as the production intent transaction does while Nav2 is
                    # still commanding the same unresolved yaw goal.
                    controller.request_reference(lowstate())
                    _wait_for_matching_reference(controller)
                    controller.acquire_reference(
                        lowstate(),
                        executed_target=tuple(published),
                        expected_install_epoch=controller.reference_install_epoch,
                    )
                    turn_reference_refreshes += 1
                if heading_guard is not None and heading_intent is not None:
                    heading_guard.observe_heading(
                        _yaw_from_wxyz(np.asarray(data.qpos[3:7]))
                    )
                    heading_correction = heading_guard.lateral_heading_correction(
                        heading_intent
                    )
                    if heading_correction is not None:
                        controller.controller.delta_heading = (
                            heading_correction.decoder_delta_heading_rad
                        )
                        heading_hold_applied_frames += 1
                policy_action = (
                    controller.run_step({}, lowstate())
                    if reference_acquired
                    else {}
                )
                if policy_action:
                    raw = _targets(policy_action)
                    arm = guard.apply(
                        raw[15:29],
                        measured=tuple(float(value) for value in data.qpos[22:36]),
                        dt=CONTROL_DT,
                    )
                    published = np.asarray(
                        _project_complete_target(raw, arm.targets),
                        dtype=np.float64,
                    )
                    peak_saturation_s = max(
                        peak_saturation_s,
                        max(arm.continuous_saturation_s),
                    )
                    if arm.constrained_joints:
                        constrained_control_frames += 1
                        last_constrained_control_frame = physics_tick // control_every
                    controller.record_executed_target(tuple(published))

            raw_control = kp * (published - data.qpos[7:]) - kd * data.qvel[6:]
            data.ctrl[:] = np.clip(
                raw_control,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            data.xfrc_applied[:] = 0.0
            pose = np.concatenate(
                (
                    data.xpos[torso_id],
                    data.xquat[torso_id],
                    np.zeros(6, dtype=np.float64),
                )
            )
            mujoco.mj_objectVelocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                torso_id,
                pose[7:13],
                0,
            )
            angular = pose[7:10].copy()
            pose[7:10] = pose[10:13]
            pose[10:13] = angular
            raw_wrench = np.zeros(6, dtype=np.float64)
            raw_wrench[:3] = (
                10_000.0 * (support_anchor - pose[:3])
                - 1_000.0 * pose[7:10]
            )
            if float(data.time) <= 3.0 + 1e-12:
                data.xfrc_applied[torso_id] = stabilized_support_wrench(
                    raw_wrench,
                    pose,
                    "idle",
                )
            mujoco.mj_step(model, data)
            assert np.all(np.isfinite(data.qpos))
            if motion_initial_xy is not None and expected_yaw_sign == 0:
                displacement = np.asarray(data.qpos[:2]) - motion_initial_xy
                requested_projections.append(
                    float(np.dot(displacement, np.asarray(expected_translation)))
                )
                target_reached = requested_projections[-1] >= target_magnitude
            elif motion_initial_yaw is not None:
                yaw_delta = _wrap_angle(
                    _yaw_from_wxyz(np.asarray(data.qpos[3:7]))
                    - motion_initial_yaw
                )
                target_reached = (
                    expected_yaw_sign * yaw_delta >= target_magnitude
                )
            minimum_z = min(minimum_z, float(data.qpos[2]))
            if target_reached:
                break
            if (
                reference_installed_s is not None
                and float(data.time) - reference_installed_s >= motion_timeout_s
            ):
                break

        assert motion_initial_xy is not None
        assert motion_initial_yaw is not None
        displacement = np.asarray(data.qpos[:2]) - motion_initial_xy
        final_yaw_delta = _wrap_angle(
            _yaw_from_wxyz(np.asarray(data.qpos[3:7])) - motion_initial_yaw
        )
        minimum_projection_index = (
            int(np.argmin(requested_projections)) if requested_projections else None
        )
        positive_progress_index = next(
            (
                index
                for index, projection in enumerate(requested_projections)
                if projection >= 0.01
            ),
            None,
        )
        diagnostic = {
            "label": label,
            "displacement_xy_m": tuple(float(value) for value in displacement),
            "yaw_delta_rad": final_yaw_delta,
            "minimum_z_m": minimum_z,
            "peak_saturation_s": peak_saturation_s,
            "constrained_frames": constrained_control_frames,
            "last_constrained_frame": last_constrained_control_frame,
            "target_magnitude": target_magnitude,
            "target_reached": target_reached,
            "motion_elapsed_s": (
                None
                if reference_installed_s is None
                else float(data.time) - reference_installed_s
            ),
            "opposite_transient_m": (
                None if not requested_projections else -min(requested_projections)
            ),
            "opposite_transient_at_s": (
                None
                if minimum_projection_index is None
                else minimum_projection_index * float(model.opt.timestep)
            ),
            "positive_progress_at_s": (
                None
                if positive_progress_index is None
                else positive_progress_index * float(model.opt.timestep)
            ),
            "heading_hold_applied_frames": heading_hold_applied_frames,
        }
        assert target_reached, diagnostic
        if expected_yaw_sign:
            assert expected_yaw_sign * final_yaw_delta >= target_magnitude, diagnostic
            assert float(np.linalg.norm(displacement)) < 0.30, diagnostic
        else:
            requested = np.asarray(expected_translation, dtype=np.float64)
            forward = float(np.dot(displacement, requested))
            cross = float(
                abs(requested[0] * displacement[1] - requested[1] * displacement[0])
            )
            assert forward >= target_magnitude, diagnostic
            # This fixture exercises the learned gait open-loop, without a
            # Nav2 lateral/yaw correction loop. Keep the historical 0.30 m
            # capability envelope here; strict live path tracking is evaluated
            # separately against the commanded trajectory.
            assert cross < 0.30, diagnostic
            max_open_loop_yaw = 15.0 if label == "forward_3m" else 10.0
            assert abs(final_yaw_delta) < math.radians(max_open_loop_yaw), diagnostic
            if label in {
                "heading_hold_left_strafe",
                "heading_hold_right_strafe",
            }:
                # Keep a two-times margin inside the product's 10-degree
                # lateral endpoint contract.  The pinned ONNX/MuJoCo path has
                # small CPU-kernel variation; tuning harder for a cosmetic
                # 2-degree target over-corrects the mirrored direction without
                # improving path safety.
                assert abs(final_yaw_delta) < math.radians(5.0), diagnostic
                assert heading_hold_applied_frames > 0, diagnostic
            if label == "heading_hold_forward_1m":
                assert heading_hold_applied_frames == 0, diagnostic
            # The official learned gait may shift its centre of mass before
            # the first requested step. Bound that startup transient without
            # weakening the sustained direction or final-target checks.
            assert min(requested_projections) > -0.04, diagnostic
        assert minimum_z >= 0.55, diagnostic
        assert reference_installed_s is not None
        assert peak_saturation_s < 2.0, diagnostic
        assert constrained_control_frames > 0, diagnostic
        assert last_constrained_control_frame >= 0, diagnostic

        # A successful motion action is not complete until the learned
        # controller has transitioned to NVIDIA's bounded IDLE reference.
        # Preserve the acknowledged recurrent history and cursor across that
        # transition: rebuilding all history from one measured sample passed a
        # forward stop but destabilized the live reverse stop.
        stop_started_xy = np.asarray(data.qpos[:2], dtype=np.float64).copy()
        stop_started_yaw = _yaw_from_wxyz(
            np.asarray(data.qpos[3:7], dtype=np.float64)
        )
        stop_started_epoch = controller.reference_install_epoch
        controller.ms.mode = LM.IDLE
        controller.ms.speed = -1.0
        controller.ms.movement_angle = 0.0
        # H-IDE may install several measured-frame references during one Nav2
        # action. Continuous-IDLE must therefore hold the exact yaw at the live
        # planner cursor: zero undoes a turn, while a stale global goal can add
        # it twice.
        stop_reference_facing = controller.current_reference_facing_angle()
        controller.ms.facing_angle = stop_reference_facing
        controller.ms.has_movement = False
        controller.ms.needs_replan = False
        # Reference planning is a wall-clock worker while this fixture can run
        # MuJoCo much faster than real time.  Resolve and install the stop
        # reference against one frozen LowState so CPU scheduling cannot invent
        # an unreviewed actuator dwell before IDLE starts.
        controller.request_continuous_idle_reference()
        _wait_for_matching_reference(controller)
        controller.acquire_continuous_reference(
            executed_target=tuple(published),
            expected_install_epoch=controller.reference_install_epoch,
        )
        stop_minimum_z = float(data.qpos[2])
        stop_peak_tilt = 0.0
        stop_peak_idle_readapt_error = 0.0
        stop_idle_readapt_states: set[str] = set()
        stop_started_s = float(data.time)
        while float(data.time) - stop_started_s < 5.0:
            physics_tick += 1
            if physics_tick % control_every == 0:
                policy_action = controller.run_step({}, lowstate())
                assert policy_action
                raw = _targets(policy_action)
                arm = guard.apply(
                    raw[15:29],
                    measured=tuple(float(value) for value in data.qpos[22:36]),
                    dt=CONTROL_DT,
                )
                published = np.asarray(
                    _project_complete_target(raw, arm.targets),
                    dtype=np.float64,
                )
                controller.record_executed_target(tuple(published))
                stop_peak_idle_readapt_error = max(
                    stop_peak_idle_readapt_error,
                    float(controller.controller.idle_readapt_average_error_rad),
                )
                stop_idle_readapt_states.add(
                    controller.controller.idle_readapt_state.name
                )

            raw_control = kp * (published - data.qpos[7:]) - kd * data.qvel[6:]
            data.ctrl[:] = np.clip(
                raw_control,
                model.actuator_ctrlrange[:, 0],
                model.actuator_ctrlrange[:, 1],
            )
            data.xfrc_applied[:] = 0.0
            mujoco.mj_step(model, data)
            assert np.all(np.isfinite(data.qpos))
            stop_minimum_z = min(stop_minimum_z, float(data.qpos[2]))
            roll, pitch = _roll_pitch_from_wxyz(
                np.asarray(data.qpos[3:7], dtype=np.float64)
            )
            stop_peak_tilt = max(stop_peak_tilt, abs(roll), abs(pitch))

        stop_displacement = np.asarray(data.qpos[:2]) - stop_started_xy
        stop_yaw_delta = _wrap_angle(
            _yaw_from_wxyz(np.asarray(data.qpos[3:7], dtype=np.float64))
            - stop_started_yaw
        )
        stop_linear_speed = float(np.linalg.norm(data.qvel[:2]))
        stop_angular_speed = float(np.linalg.norm(data.qvel[3:6]))
        stop_diagnostic = {
            **diagnostic,
            "stop_displacement_xy_m": tuple(
                float(value) for value in stop_displacement
            ),
            "stop_yaw_delta_rad": stop_yaw_delta,
            "stop_reference_facing_rad": stop_reference_facing,
            "stop_minimum_z_m": stop_minimum_z,
            "stop_peak_tilt_rad": stop_peak_tilt,
            "stop_linear_speed_mps": stop_linear_speed,
            "stop_angular_speed_rps": stop_angular_speed,
            "stop_reference_epoch": controller.reference_install_epoch,
            "stop_idle_readapt_stored": (
                controller.controller.idle_readapt_stored
            ),
            "stop_idle_readapt_states": sorted(stop_idle_readapt_states),
            "stop_peak_idle_readapt_error_rad": stop_peak_idle_readapt_error,
        }
        assert controller.reference_install_epoch > stop_started_epoch, stop_diagnostic
        assert stop_minimum_z >= 0.55, stop_diagnostic
        assert stop_peak_tilt < 0.30, stop_diagnostic
        assert float(np.linalg.norm(stop_displacement)) < 0.60, stop_diagnostic
        assert abs(stop_yaw_delta) < 0.20, stop_diagnostic
        assert stop_linear_speed < 0.08, stop_diagnostic
        assert stop_angular_speed < 0.20, stop_diagnostic
        assert controller.controller.idle_readapt_stored is True, stop_diagnostic
    finally:
        controller.shutdown()
