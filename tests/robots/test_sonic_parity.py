#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Golden invariants for the SONIC deploy-order closed loop."""

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.robots.unitree_g1.controllers.sonic_pipeline import (
    ACTION_SCALE,
    DEFAULT_ANGLES,
    DEFAULT_ANGLES_ISAACLAB,
    ENCODER_UPDATE_EVERY,
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    MovementState,
    PlannerController,
    PlannerMotion,
    StandingEncoderDecoder,
    hardware_target_to_normalized_action,
    normalized_action_to_hardware_target,
    observation_state,
    snapshot_ms,
)
from lerobot.robots.unitree_g1.controllers.sonic_whole_body import (
    SonicRuntime,
    SonicWholeBodyController,
)
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex, get_gravity_orientation


class _FakeSession:
    def __init__(self, input_dim: int, output: np.ndarray):
        self.input = SimpleNamespace(name=f"input_{input_dim}", shape=(1, input_dim))
        self.output = np.asarray(output, np.float32)
        self.inputs: list[np.ndarray] = []

    def get_inputs(self):
        return [self.input]

    def run(self, _outputs, feed):
        self.inputs.append(np.asarray(feed[self.input.name]).copy())
        return [self.output.reshape(1, -1).copy()]


def _motion_from_measured(q_mujoco: np.ndarray, *, frames: int = 40) -> PlannerMotion:
    motion = PlannerMotion(max_frames=max(frames, 50))
    motion.timesteps = frames
    q_il = q_mujoco[MUJOCO_TO_ISAACLAB]
    motion.joint_positions[:frames] = q_il
    motion.joint_velocities[:frames] = 0.0
    motion.body_positions[:frames, 2] = 0.788740
    motion.body_positions[:frames, 0] = np.linspace(0.0, 0.2, frames)
    motion.body_quaternions[:frames, 0] = 1.0
    return motion


class _FakePlanner:
    def __init__(self):
        self.gen_frame = 0
        self.initialize_inputs: list[np.ndarray] = []
        self.requests: list[tuple[int, MovementState]] = []
        self.results: list[tuple[PlannerMotion, int]] = []
        self.started = 0
        self.stopped = 0

    def initialize(self, q_mujoco, _ms):
        q = np.asarray(q_mujoco, np.float32).copy()
        self.initialize_inputs.append(q)
        return _motion_from_measured(q)

    def start_subprocess(self, controller, *, use_gpu=False):
        self.controller = controller
        self.use_gpu = use_gpu
        self.started += 1

    def stop_subprocess(self):
        self.stopped += 1

    def request_replan(self, cursor, ms):
        self.gen_frame = int(cursor) + 2
        self.requests.append((int(cursor), ms))

    def try_get_new_motion(self):
        return self.results.pop(0) if self.results else None


def _complete_obs(*, offset: float = 0.0) -> dict[str, float]:
    obs: dict[str, float] = {}
    for motor in G1_29_JointIndex:
        obs[f"{motor.name}.q"] = float(DEFAULT_ANGLES[motor.value] + offset + motor.value * 1e-3)
        obs[f"{motor.name}.dq"] = float(-0.2 + motor.value * 0.01)
    obs.update(
        {
            "imu.quat.w": 1.0,
            "imu.quat.x": 0.0,
            "imu.quat.y": 0.0,
            "imu.quat.z": 0.0,
            "imu.gyro.x": 0.11,
            "imu.gyro.y": -0.22,
            "imu.gyro.z": 0.33,
        }
    )
    return obs


def _target_vector(target: dict[str, float]) -> np.ndarray:
    return np.array([target[f"{motor.name}.q"] for motor in G1_29_JointIndex], np.float32)


def _runtime(*, auto_reference_authority: bool = False):
    encoder = _FakeSession(1762, np.arange(64, dtype=np.float32))
    raw_action = np.linspace(-0.4, 0.4, 29, dtype=np.float32)
    decoder = _FakeSession(994, raw_action)
    planner = _FakePlanner()
    controller = PlannerController(planner, encoder, decoder)

    runtime = SonicRuntime.__new__(SonicRuntime)
    runtime.kp = runtime.kd = np.ones(29, np.float32)
    runtime.ms = MovementState()
    runtime.planner = planner
    runtime.controller = controller
    runtime.use_gpu = False
    runtime.step = 0
    runtime.replan_timer = 0.0
    runtime.last_ms = snapshot_ms(runtime.ms)
    runtime.auto_reference_authority = auto_reference_authority
    runtime._initialized = False
    runtime._planner_worker_started = False
    runtime._matching_reference_loaded = False
    runtime._reference_install_epoch = 0
    runtime._minimum_acquirable_epoch = None
    runtime._policy_step_pending = False
    runtime._pending_reference_install_epoch = None
    runtime._pending_ref_cursor = None
    return runtime, planner, controller, encoder, decoder, raw_action


def _initialize_and_acquire(runtime, planner, obs):
    runtime.initialize_from_observation(obs)
    boundary = np.array([obs[f"{motor.name}.q"] for motor in G1_29_JointIndex], np.float32)
    runtime.request_reference(obs)
    planner.results.append((_motion_from_measured(boundary), planner.gen_frame))
    assert runtime.poll_reference()
    runtime.acquire_reference(
        obs,
        executed_target=boundary,
        expected_install_epoch=runtime.reference_install_epoch,
    )
    return boundary


def test_action_mapping_is_a_29_axis_bijection():
    assert ENCODER_UPDATE_EVERY == 1
    for policy_index in range(29):
        action = np.zeros(29, np.float32)
        action[policy_index] = 1.0
        target = normalized_action_to_hardware_target(action)
        changed = np.flatnonzero(np.abs(target - DEFAULT_ANGLES) > 1e-7)
        expected_hardware_index = int(MUJOCO_TO_ISAACLAB[policy_index])
        np.testing.assert_array_equal(changed, [expected_hardware_index])
        assert target[expected_hardware_index] - DEFAULT_ANGLES[expected_hardware_index] == pytest.approx(
            float(ACTION_SCALE[expected_hardware_index])
        )
        np.testing.assert_allclose(
            hardware_target_to_normalized_action(target),
            action,
            atol=2e-6,
            rtol=0.0,
        )

    np.testing.assert_array_equal(
        np.arange(29, dtype=np.int32)[ISAACLAB_TO_MUJOCO][MUJOCO_TO_ISAACLAB],
        np.arange(29, dtype=np.int32),
    )


def test_decoder_history_tensor_matches_golden_oldest_to_newest_layout():
    controller = StandingEncoderDecoder(
        _FakeSession(1762, np.zeros(64, np.float32)),
        _FakeSession(994, np.zeros(29, np.float32)),
    )
    controller.token = np.arange(64, dtype=np.float32) + 0.5
    controller.h_ang = [np.full(3, 100 + frame, np.float32) for frame in range(10)]
    controller.h_q_mj = [np.full(29, 200 + frame, np.float32) for frame in range(10)]
    controller.h_dq_mj = [np.full(29, 300 + frame, np.float32) for frame in range(10)]
    controller.h_act_mj = [np.full(29, 400 + frame, np.float32) for frame in range(10)]
    controller.h_quat = []
    for frame in range(10):
        angle = 0.02 * frame
        controller.h_quat.append(np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0], np.float32))

    expected_parts = [controller.token]
    for history in (
        controller.h_ang,
        controller.h_q_mj,
        controller.h_dq_mj,
        controller.h_act_mj,
    ):
        expected_parts.extend(reversed(history))
    expected_parts.extend(get_gravity_orientation(quat) for quat in reversed(controller.h_quat))
    expected = np.concatenate(expected_parts).astype(np.float32)

    actual = controller.build_decoder_obs()
    assert actual.shape == (994,)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "missing_key",
    [
        f"{G1_29_JointIndex.kLeftHipPitch.name}.q",
        f"{G1_29_JointIndex.kRightWristYaw.name}.dq",
        "imu.quat.w",
        "imu.gyro.z",
    ],
)
def test_authoritative_observation_rejects_every_partial_lowstate_class(missing_key):
    obs = _complete_obs()
    del obs[missing_key]
    with pytest.raises(ValueError, match="complete measured LowState"):
        observation_state(obs, require_complete=True)


def test_authoritative_observation_rejects_nonfinite_and_unnormalized_values():
    obs = _complete_obs()
    obs[f"{G1_29_JointIndex.kLeftKnee.name}.dq"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        observation_state(obs, require_complete=True)

    obs = _complete_obs()
    obs["imu.quat.w"] = 2.0
    with pytest.raises(ValueError, match="normalized"):
        observation_state(obs, require_complete=True)


def test_initialization_is_measured_lazy_idempotent_and_policy_free():
    runtime, planner, controller, encoder, decoder, _raw_action = _runtime()
    obs = _complete_obs(offset=0.07)
    measured = np.array([obs[f"{motor.name}.q"] for motor in G1_29_JointIndex], np.float32)

    assert not runtime.initialized
    assert not planner.initialize_inputs
    assert runtime.initialize_from_observation(obs)
    assert runtime.initialized
    assert runtime.step == 0
    assert controller.ref_cursor == 0
    assert runtime.reference_install_epoch == 1
    np.testing.assert_array_equal(planner.initialize_inputs[0], measured)
    assert planner.started == 1
    assert len(encoder.inputs) == len(decoder.inputs) == 0

    measured_residual = measured[MUJOCO_TO_ISAACLAB] - DEFAULT_ANGLES_ISAACLAB
    executed = hardware_target_to_normalized_action(measured)
    for history in controller.h_q_mj:
        np.testing.assert_allclose(history, measured_residual, atol=1e-7)
    for history in controller.h_act_mj:
        np.testing.assert_allclose(history, executed, atol=2e-6)

    assert not runtime.initialize_from_observation(obs)
    assert len(planner.initialize_inputs) == 1
    assert planner.started == 1


def test_publish_feedback_is_required_and_is_the_only_cursor_commit():
    runtime, planner, controller, encoder, decoder, _raw_action = _runtime()
    obs = _complete_obs()
    runtime.initialize_from_observation(obs)
    boundary = np.array([obs[f"{motor.name}.q"] for motor in G1_29_JointIndex], np.float32)
    with pytest.raises(RuntimeError, match="exact reference install epoch"):
        runtime.acquire_reference(obs, executed_target=boundary)

    runtime.request_reference(obs)
    planner.results.append((_motion_from_measured(boundary), planner.gen_frame))
    assert runtime.poll_reference()
    runtime.acquire_reference(
        obs,
        executed_target=boundary,
        expected_install_epoch=runtime.reference_install_epoch,
    )

    first = runtime.tick(obs, debug=False, use_joystick=False)
    assert len(first) == 29
    assert runtime.execution_feedback_pending
    assert controller.ref_cursor == 0
    assert len(encoder.inputs) == len(decoder.inputs) == 1
    with pytest.raises(RuntimeError, match="not acknowledged"):
        runtime.tick(obs, debug=False, use_joystick=False)

    published = _target_vector(first)
    published[0] += 0.025  # emulate a final actuator-side guard rewrite
    expected_executed = hardware_target_to_normalized_action(published)
    np.testing.assert_allclose(runtime.record_executed_target(published), expected_executed, atol=2e-6)
    np.testing.assert_allclose(controller.last_executed_action_il, expected_executed, atol=2e-6)
    assert controller.ref_cursor == 1

    runtime.tick(obs, debug=False, use_joystick=False)
    assert len(encoder.inputs) == len(decoder.inputs) == 2
    assert controller.ref_cursor == 1

    # A stop racing this pending candidate keeps the feedback transaction open.
    runtime.revoke_reference_authority()
    assert runtime.execution_feedback_pending
    with pytest.raises(RuntimeError, match="not acknowledged"):
        runtime.tick(obs, debug=False, use_joystick=False)
    runtime.record_executed_target(published)
    assert controller.ref_cursor == 1
    assert not runtime.execution_feedback_pending
    with pytest.raises(RuntimeError, match="no policy step"):
        runtime.record_executed_target(published)


def test_periodic_refresh_installs_before_decode_and_binds_candidate_to_new_epoch():
    runtime, planner, controller, _encoder, _decoder, _raw_action = _runtime()
    obs = _complete_obs()
    _initialize_and_acquire(runtime, planner, obs)

    first = runtime.tick(obs, debug=False, use_joystick=False)
    runtime.record_executed_target(_target_vector(first))
    old_epoch = runtime.reference_install_epoch
    assert controller.ref_cursor == 1

    refreshed_q = np.array(
        [obs[f"{motor.name}.q"] + 0.05 for motor in G1_29_JointIndex],
        np.float32,
    )
    planner.results.append((_motion_from_measured(refreshed_q), planner.gen_frame))
    runtime.tick(obs, debug=False, use_joystick=False)

    assert runtime.reference_install_epoch == old_epoch + 1
    assert runtime._pending_reference_install_epoch == runtime.reference_install_epoch
    assert runtime._pending_ref_cursor == controller.ref_cursor == 0


@pytest.mark.parametrize("identity_change", ["epoch", "cursor"])
def test_stale_candidate_cannot_commit_across_reference_identity(identity_change):
    runtime, planner, controller, _encoder, _decoder, _raw_action = _runtime()
    obs = _complete_obs()
    _initialize_and_acquire(runtime, planner, obs)
    candidate = runtime.tick(obs, debug=False, use_joystick=False)
    published = _target_vector(candidate)
    executed_before = controller.last_executed_action_il.copy()
    cursor_before = controller.ref_cursor

    if identity_change == "epoch":
        runtime._reference_install_epoch += 1
    else:
        controller.ref_cursor += 1

    with pytest.raises(RuntimeError, match="no longer matches"):
        runtime.record_executed_target(published)
    np.testing.assert_array_equal(controller.last_executed_action_il, executed_before)
    assert controller.execution_feedback_pending
    assert runtime.execution_feedback_pending
    assert controller.ref_cursor == cursor_before + (identity_change == "cursor")


def test_poll_reference_rejects_install_while_candidate_feedback_is_pending():
    runtime, planner, _controller, _encoder, _decoder, _raw_action = _runtime()
    obs = _complete_obs()
    boundary = _initialize_and_acquire(runtime, planner, obs)
    runtime.tick(obs, debug=False, use_joystick=False)
    epoch_before = runtime.reference_install_epoch
    planner.results.append((_motion_from_measured(boundary), planner.gen_frame))

    with pytest.raises(RuntimeError, match="feedback is pending"):
        runtime.poll_reference()
    assert runtime.reference_install_epoch == epoch_before
    assert len(planner.results) == 1


def test_planning_only_request_and_poll_do_not_advance_policy_state():
    runtime, planner, controller, encoder, decoder, _raw_action = _runtime()
    initial_obs = _complete_obs()
    runtime.initialize_from_observation(initial_obs)
    before = {
        "token": controller.token.copy(),
        "raw": controller.last_raw_action_il.copy(),
        "executed": controller.last_executed_action_il.copy(),
        "q_history": np.asarray(controller.h_q_mj).copy(),
        "cursor": controller.ref_cursor,
        "step": runtime.step,
    }

    fresh_obs = _complete_obs(offset=0.03)
    runtime.request_reference(fresh_obs)
    assert len(planner.requests) == 1
    assert not runtime.reference_authority
    assert not runtime.matching_reference_installed
    assert runtime.reference_install_epoch == 1
    assert controller.ref_cursor == before["cursor"] == 0
    assert runtime.step == before["step"] == 0
    assert len(encoder.inputs) == len(decoder.inputs) == 0
    np.testing.assert_array_equal(controller.token, before["token"])
    np.testing.assert_array_equal(controller.last_raw_action_il, before["raw"])
    np.testing.assert_array_equal(controller.last_executed_action_il, before["executed"])
    np.testing.assert_array_equal(np.asarray(controller.h_q_mj), before["q_history"])

    measured = np.array([fresh_obs[f"{motor.name}.q"] for motor in G1_29_JointIndex], np.float32)
    planner.results.append((_motion_from_measured(measured), planner.gen_frame))
    assert runtime.poll_reference()
    assert runtime.matching_reference_installed
    assert runtime.reference_install_epoch == 2
    assert controller.ref_cursor == runtime.step == 0
    assert len(encoder.inputs) == len(decoder.inputs) == 0


def test_reset_preserves_movement_alias_and_restarts_planner_after_measurement():
    runtime, planner, _controller, _encoder, _decoder, _raw_action = _runtime()
    obs = _complete_obs()
    runtime.initialize_from_observation(obs)
    movement_alias = runtime.ms
    runtime.ms.speed = 0.7
    runtime.ms.has_movement = True

    runtime.reset()
    assert runtime.ms is movement_alias
    assert runtime.ms.speed == -1.0
    assert not runtime.ms.has_movement
    assert not runtime.initialized
    assert planner.stopped == 1
    assert not runtime._planner_worker_started

    assert runtime.initialize_from_observation(obs)
    assert planner.started == 2


def test_manual_authority_disables_startup_blend_and_operator_intent():
    wrapper = SonicWholeBodyController.__new__(SonicWholeBodyController)
    wrapper._startup_blend_enabled = False
    wrapper._init_step = 0
    wrapper._init_ramp_steps = 150
    wrapper._start_pose = {}
    wrapper._allow_operator_intent = False
    wrapper.ms = MovementState()
    wrapper.controller = SimpleNamespace()

    obs = _complete_obs()
    raw = {f"{motor.name}.q": float(DEFAULT_ANGLES[motor.value] + 0.2) for motor in G1_29_JointIndex}
    assert wrapper._startup_blend(obs, raw) is raw
    wrapper._process_keyboard({"keyboard.keys": {"w", "q", "0"}})
    assert wrapper.ms.mode == 0
    assert wrapper.ms.speed == -1.0
    assert not wrapper.ms.has_movement
