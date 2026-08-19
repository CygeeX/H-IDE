#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""SONIC full-body controller for Unitree G1."""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import hf_hub_download

from lerobot.teleoperators.pico_headset.smpl_constants import (
    LOCO_AXES_PREFIX,
    LOCO_BTN_PREFIX,
    LOCO_N_AXES,
    LOCO_N_BTN,
    ROOT_ACTION_DIM,
    ROOT_ACTION_PREFIX,
    SMPL_ACTION_PREFIX,
    SMPL_OBS_DIM as SMPL_ACTION_DIM,
    VR3_ORN_DIM,
    VR3_ORN_PREFIX,
    VR3_POS_DIM,
    VR3_POS_PREFIX,
    WB_ACTION_DIM,
    wb_action_key,
)
from lerobot.utils.import_utils import _onnxruntime_available, require_package

from ..g1_utils import KEYBOARD_KEYS_FIELD, MUJOCO_TO_ISAACLAB, G1_29_JointIndex, lowstate_to_obs
from .sonic_pipeline import (
    CONTROL_DT,
    DEBUG_PRINT_EVERY,
    DEFAULT_ANGLES,
    DEFAULT_HEIGHT,
    LM,
    MOTION_SETS,
    MovementState,
    PlannerController,
    SonicPlanner,
    apply_pico_loco_axes,
    clamp_mode_params,
    compute_kp_kd,
    hardware_target_vector,
    make_ort_session_options,
    observation_state,
    ort_providers,
    process_joystick,
    should_replan_request,
    snapshot_ms,
)

if TYPE_CHECKING or _onnxruntime_available:
    import onnxruntime as ort
else:
    ort = None

logger = logging.getLogger(__name__)

# Startup blend duration: over the first control ticks, linearly interpolate every joint
# from the robot's initial measured pose into the policy's commanded target, so control
# eases in without a snap on the first command.
INIT_RAMP_S = 3.0


def _extract_smpl_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a (720,) SMPL window from ``smpl.{i}`` action keys, or None.

    The pico_headset teleoperator emits the whole-body reference as flat floats so
    it flows unchanged through the standard lerobot action pipeline.
    """
    # The keys are smpl.0 .. smpl.719; presence of the first element (smpl.0) is the
    # sentinel that a full SMPL window was sent this tick. If it's absent, there's no
    # whole-body reference, so bail out.
    if not action or f"{SMPL_ACTION_PREFIX}0" not in action:
        return None
    arr = np.fromiter(
        (float(action.get(f"{SMPL_ACTION_PREFIX}{i}", 0.0)) for i in range(SMPL_ACTION_DIM)),
        dtype=np.float32,
        count=SMPL_ACTION_DIM,
    )
    return arr


def _extract_root_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a (4,) SMPL root quaternion (wxyz) from ``root.{i}`` keys, or None."""
    if not action or f"{ROOT_ACTION_PREFIX}0" not in action:
        return None
    q = np.fromiter(
        (float(action.get(f"{ROOT_ACTION_PREFIX}{i}", 0.0)) for i in range(ROOT_ACTION_DIM)),
        dtype=np.float32,
        count=ROOT_ACTION_DIM,
    )
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return None
    return q / n


def _extract_vr3_from_action(action: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Reassemble the 3-point VR targets from ``vr3_pos.{i}`` / ``vr3_orn.{i}`` keys.

    Returns ``(pos (9,), orn (12,))`` for the [l-wrist, r-wrist, neck] keypoints, or
    None when no VR3 reference was sent this tick. Presence of ``vr3_pos.0`` is the
    sentinel that a full 3-point frame is available (mirrors the SMPL sentinel).
    """
    if not action or f"{VR3_POS_PREFIX}0" not in action:
        return None
    pos = np.fromiter(
        (float(action.get(f"{VR3_POS_PREFIX}{i}", 0.0)) for i in range(VR3_POS_DIM)),
        dtype=np.float32,
        count=VR3_POS_DIM,
    )
    orn = np.fromiter(
        (float(action.get(f"{VR3_ORN_PREFIX}{i}", 0.0)) for i in range(VR3_ORN_DIM)),
        dtype=np.float32,
        count=VR3_ORN_DIM,
    )
    return pos, orn


def _extract_wb34_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a dense (34,) whole-body command from ``wb.{i}.pos`` keys, or None.

    This is the OpenHLM / pi0.5 joint-based interface: one 34-D vector per tick
    (sentinel: presence of ``wb.0.pos``) carrying absolute joint targets in real
    units. The ``.pos`` suffix lets these flow through ``lerobot-rollout`` as normal
    joint-position action features.
    """
    if not action or wb_action_key(0) not in action:
        return None
    return np.fromiter(
        (float(action.get(wb_action_key(i), 0.0)) for i in range(WB_ACTION_DIM)),
        dtype=np.float32,
        count=WB_ACTION_DIM,
    )


def _wb34_to_reference(wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a 34-D OpenHLM whole-body command to a SONIC mode-0 reference.

    Returns ``(ref29, anchor_quat)`` where ``ref29`` is the 29 joint targets in
    IsaacLab order (what SONIC's ``motion_joint_positions`` expects) and
    ``anchor_quat`` (wxyz) encodes the root roll/pitch (yaw=0).

    OpenHLM layout : [L-arm 0:7, L-grip 7, R-arm 8:15, R-grip 15,
                      L-leg 16:22, R-leg 22:28, waist 28:31, root rp+yaw 31:34]
    The 29 joints are first assembled in MuJoCo / Unitree-SDK order
    ([L-leg 0:6, R-leg 6:12, waist 12:15, L-arm 15:22, R-arm 22:29] — the
    ``G1_29_JointIndex`` grouping OpenHLM uses), then permuted to IsaacLab order via
    ``MUJOCO_TO_ISAACLAB``. Grippers (7, 15) and yaw-rate (33) are not part of the
    29-DoF SONIC reference.
    """
    ref_mj = np.zeros(29, np.float32)  # MuJoCo / Unitree-SDK grouped order
    ref_mj[0:6] = wb[16:22]  # left leg
    ref_mj[6:12] = wb[22:28]  # right leg
    ref_mj[12:15] = wb[28:31]  # waist
    ref_mj[15:22] = wb[0:7]  # left arm
    ref_mj[22:29] = wb[8:15]  # right arm
    ref = ref_mj[MUJOCO_TO_ISAACLAB].astype(np.float32)  # -> IsaacLab order for SONIC
    roll, pitch = float(wb[31]), float(wb[32])
    cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
    anchor = np.array([cr * cp, sr * cp, cr * sp, sr * sp], np.float32)  # Rx(roll)·Ry(pitch)
    return ref, anchor


def _extract_loco_from_action(action: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Reassemble controller-stick locomotion from ``loco_axes.{i}`` / ``loco_btn.{i}``.

    Returns ``(axes (4,) = [lx, ly, rx, ry], buttons (4,) = [A, B, X, Y])`` or None
    when no locomotion state was sent this tick (sentinel: ``loco_axes.0``).
    """
    if not action or f"{LOCO_AXES_PREFIX}0" not in action:
        return None
    axes = np.fromiter(
        (float(action.get(f"{LOCO_AXES_PREFIX}{i}", 0.0)) for i in range(LOCO_N_AXES)),
        dtype=np.float32,
        count=LOCO_N_AXES,
    )
    buttons = np.fromiter(
        (float(action.get(f"{LOCO_BTN_PREFIX}{i}", 0.0)) for i in range(LOCO_N_BTN)),
        dtype=np.float32,
        count=LOCO_N_BTN,
    )
    return axes, buttons


class SonicRuntime:
    """Shared SONIC loop with explicit planner and actuator transactions."""

    def __init__(self, force_cpu: bool = False, *, auto_reference_authority: bool = True):
        require_package("onnxruntime", extra="unitree_g1")
        planner_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="planner_sonic.onnx")
        encoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_encoder.onnx")
        decoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_decoder.onnx")

        providers = ort_providers(force_cpu=force_cpu)
        self.use_gpu = providers[0] == "CUDAExecutionProvider"
        so = make_ort_session_options()

        planner_sess = ort.InferenceSession(planner_path, sess_options=so, providers=providers)
        encoder_sess = ort.InferenceSession(encoder_path, sess_options=so, providers=providers)
        decoder_sess = ort.InferenceSession(decoder_path, sess_options=so, providers=providers)

        self.kp, self.kd = compute_kp_kd()
        self.ms = MovementState()
        self.planner = SonicPlanner(planner_sess, planner_path)
        self.controller = PlannerController(self.planner, encoder_sess, decoder_sess)
        self.step = 0
        self.replan_timer = 0.0
        self.last_ms = snapshot_ms(self.ms)
        self.auto_reference_authority = bool(auto_reference_authority)
        self._initialized = False
        self._planner_worker_started = False
        self._matching_reference_loaded = False
        self._reference_install_epoch = 0
        self._minimum_acquirable_epoch: int | None = None
        self._policy_step_pending = False

    @property
    def pipeline(self):
        return self.controller

    @property
    def initialized(self):
        """Whether a complete measured LowState seeded the live runtime."""

        return self._initialized

    @property
    def reference_authority(self):
        return self.controller.reference_authority

    @property
    def execution_feedback_pending(self):
        return self._policy_step_pending

    @property
    def reference_install_epoch(self):
        """Monotonic count of planner motions installed in the live buffer."""

        return self._reference_install_epoch

    @property
    def matching_reference_installed(self):
        """Whether the latest required reference was accepted and installed."""

        return self._matching_reference_loaded

    def _start_planner_worker(self):
        if self._planner_worker_started:
            return
        self.planner.start_subprocess(self.controller, use_gpu=self.use_gpu)
        self._planner_worker_started = True

    def initialize_from_observation(self, obs: dict) -> bool:
        """Idempotently initialize planner/history from the first real LowState.

        This synchronous method is safe to call during a startup preparation
        phase before a real-time clock is armed. It never advances ``step`` or
        ``ref_cursor`` and construction itself performs no planner inference.
        """

        q, _dq, _ang, _quat = observation_state(obs, require_complete=True)
        if self._initialized:
            return False
        motion = self.planner.initialize(q, self.ms)
        self.controller.load_initial_motion(motion)
        self.controller.seed_history(obs, executed_target=q)
        self._matching_reference_loaded = True
        self._reference_install_epoch += 1
        self._start_planner_worker()
        self._initialized = True
        if self.auto_reference_authority:
            self.controller.set_reference_authority(
                actuator_authority=True,
                matching_reference_ready=True,
            )
        return True

    def revoke_reference_authority(
        self,
        *,
        discard_reference: bool = True,
        discard_pending_execution: bool = False,
    ):
        """Stop playback while preserving an in-flight publish transaction.

        A stop may win after inference but before DDS. By default the pending
        candidate remains open so the publisher can acknowledge the frozen
        target it actually writes. Cursor advancement is independently blocked
        by revoked authority. Only a terminal abort with no possible publish
        should set ``discard_pending_execution=True``.
        """

        if discard_pending_execution and self.controller.execution_feedback_pending:
            self.controller.discard_pending_execution()
            self._policy_step_pending = False
        self.controller.set_reference_authority(
            actuator_authority=False,
            matching_reference_ready=False,
        )
        self.controller.playing = False
        if discard_reference:
            self._matching_reference_loaded = False

    def request_reference(self, obs: dict):
        """Request a plan from fresh measurement without running encoder/decoder.

        The async planner request/result API remains visible to generation-scoped
        evidence adapters. No history, token, action, runtime step, or playback
        cursor is advanced on this planning-only path.
        """

        if not self._initialized:
            raise RuntimeError("SONIC must be initialized from LowState before planning-only requests")
        if self._policy_step_pending or self.controller.execution_feedback_pending:
            raise RuntimeError("cannot rebase SONIC planning with unpublished policy output")
        self.revoke_reference_authority(
            discard_reference=True,
            discard_pending_execution=True,
        )
        self.controller.load_measured_reference(obs)
        self._minimum_acquirable_epoch = self._reference_install_epoch + 1
        self.planner.request_replan(self.controller.ref_cursor, self.ms)
        self.replan_timer = 0.0
        self.ms.needs_replan = False
        self.last_ms = snapshot_ms(self.ms)

    def poll_reference(self) -> bool:
        """Poll and blend one accepted async plan without policy-state progress."""

        result = self.planner.try_get_new_motion()
        if result is None:
            return False
        self.controller.blend_new_motion(*result)
        self.controller.reinit_heading = True
        self._matching_reference_loaded = True
        self._reference_install_epoch += 1
        return True

    def acquire_reference(self, obs: dict, *, executed_target, expected_install_epoch: int | None = None):
        """Acquire a loaded matching plan at the actual actuator boundary.

        Manual-authority integrations must pass the exact epoch they observed
        after their evidence-filtered ``poll_reference``. Thus the synchronous
        startup/IDLE motion cannot be mistaken for a new intent's accepted plan.
        """

        if not self._initialized:
            raise RuntimeError("SONIC must be initialized from LowState before authority acquisition")
        if not self._matching_reference_loaded:
            raise RuntimeError("SONIC cannot acquire authority without a loaded matching reference")
        if not self.auto_reference_authority:
            minimum = self._minimum_acquirable_epoch
            if expected_install_epoch is None or isinstance(expected_install_epoch, bool):
                raise RuntimeError("manual SONIC authority requires an exact reference install epoch")
            if (
                minimum is None
                or self._reference_install_epoch < minimum
                or int(expected_install_epoch) != self._reference_install_epoch
            ):
                raise RuntimeError("manual SONIC authority epoch does not identify the requested plan")
        if self._policy_step_pending or self.controller.execution_feedback_pending:
            raise RuntimeError("SONIC cannot acquire authority with unpublished policy output")
        # Validate the boundary before mutating any recurrent state.
        boundary = hardware_target_vector(executed_target)
        self.controller.seed_history(obs, executed_target=boundary)
        self.controller.reinit_heading = True
        self.controller.first_motion = True
        self.controller.playing = True
        self.controller.set_reference_authority(
            actuator_authority=True,
            matching_reference_ready=True,
        )
        self.step = 0
        self.replan_timer = 0.0
        self.last_ms = snapshot_ms(self.ms)
        self._minimum_acquirable_epoch = None

    def record_executed_target(self, target):
        """Commit a successful publish, then advance exactly one ref frame."""

        if not self._policy_step_pending:
            raise RuntimeError("SONIC runtime has no policy step awaiting publish feedback")
        executed = self.controller.record_executed_target(target)
        self.controller.advance_cursor()
        self._policy_step_pending = False
        return executed

    def tick(self, obs: dict, *, debug: bool | None = None, use_joystick: bool = True) -> dict:
        if not obs:
            return {}

        if not self._initialized:
            self.initialize_from_observation(obs)
        if not self.reference_authority:
            if not self.auto_reference_authority:
                return {}
            if not self._matching_reference_loaded:
                raise RuntimeError("automatic SONIC authority has no initialized reference")
            q, _dq, _ang, _quat = observation_state(obs, require_complete=True)
            self.acquire_reference(obs, executed_target=q)

        if use_joystick:
            process_joystick(obs, self.ms, self.controller)
        clamp_mode_params(self.ms)

        if self.step > 0:
            self.replan_timer += CONTROL_DT
        if should_replan_request(self.ms, self.last_ms, self.replan_timer, self.step):
            self.planner.request_replan(self.controller.ref_cursor, self.ms)
            self.replan_timer = 0.0
            self.ms.needs_replan = False
            self.last_ms = snapshot_ms(self.ms)

        # Official deploy parity: the encoder token is fresh on every 50 Hz
        # policy tick; cadence must not depend on a mutable/debug counter.
        do_enc = True
        if debug is None:
            debug = self.step % DEBUG_PRINT_EVERY == 0
        action = self.controller.step(obs, update_encoder=do_enc, debug=debug)
        self._policy_step_pending = True

        self.poll_reference()

        self.step += 1
        return action

    def reset(self):
        self.revoke_reference_authority(
            discard_reference=True,
            discard_pending_execution=True,
        )
        # Do not run synchronous re-initialization concurrently with an old
        # async planner worker. A later measured initialization starts a fresh
        # worker after its initial motion is installed.
        if self._planner_worker_started:
            self.planner.stop_subprocess()
            self._planner_worker_started = False
        # Preserve the wrapper/runtime alias rather than replacing ``self.ms``.
        self.ms.__dict__.clear()
        self.ms.__dict__.update(vars(MovementState()))
        self.controller.reinit_heading = True
        self.step = 0
        self.replan_timer = 0.0
        self.last_ms = snapshot_ms(self.ms)
        self._initialized = False
        self._minimum_acquirable_epoch = None

    def shutdown(self):
        if self._planner_worker_started:
            self.planner.stop_subprocess()
            self._planner_worker_started = False


class SonicWholeBodyController:
    """Full-body SONIC controller for UnitreeG1's background controller thread.

    Integrations that arbitrate the actuator externally must construct with
    ``auto_reference_authority=False`` and follow this transaction protocol::

        initialize_from_lowstate(lowstate)
        revoke_reference_authority()
        request_reference(fresh_lowstate)
        while not poll_reference():
            ...  # hold the actuator; no policy tick or cursor progress
        acquire_reference(
            fresh_lowstate,
            executed_target=last_published_29,
            expected_install_epoch=reference_install_epoch,
        )
        candidate = run_step({}, lowstate)
        publish(candidate)
        record_executed_target(final_published_29)

    ``poll_reference`` must be observed through the owner's matching-reference
    evidence gate. A stop racing ``run_step`` revokes authority but preserves
    pending feedback: publish the frozen target, then record it; the cursor does
    not advance while authority is revoked. Only a terminal path that cannot
    publish anything may discard pending execution explicitly.
    """

    control_dt = CONTROL_DT
    full_body = True
    # Advertise a dense 34-D whole-body action space (OpenHLM / pi0.5) so the robot
    # exposes ``wb.{i}.pos`` action features and ``lerobot-rollout`` can drive it
    # directly with a 34-D VLA policy.
    wb_action = True

    def __init__(
        self,
        force_cpu: bool = False,
        *,
        auto_reference_authority: bool = True,
        enable_startup_blend: bool | None = None,
        allow_operator_intent: bool | None = None,
        enable_smpl_root: bool = False,
        root_smoothing_alpha: float = 0.15,
        enable_smpl_stream: bool = False,
        smpl_host: str | None = None,
        smpl_port: int | None = None,
    ):
        logger.info("Loading SONIC whole-body controller...")
        self._runtime = SonicRuntime(
            force_cpu=force_cpu,
            auto_reference_authority=auto_reference_authority,
        )
        self.kp = self._runtime.kp
        self.kd = self._runtime.kd
        self.controller = self._runtime.controller
        self.ms = self._runtime.ms
        if enable_startup_blend is None:
            enable_startup_blend = auto_reference_authority
        if enable_startup_blend and not auto_reference_authority:
            raise ValueError("manual SONIC authority cannot enable post-policy startup blending")
        self._startup_blend_enabled = bool(enable_startup_blend)
        if allow_operator_intent is None:
            allow_operator_intent = auto_reference_authority
        if allow_operator_intent and not auto_reference_authority:
            raise ValueError("manual SONIC authority accepts intent only from its owning integration")
        self._allow_operator_intent = bool(allow_operator_intent)

        # When True, the per-frame SMPL root quaternion steers the mode-2 anchor.
        # Off by default: even with smoothing this changes the anchor/heading and is
        # untested on hardware, so it stays opt-in. When enabled, the raw per-frame
        # root quat (from a 30 Hz dataset resampled to a 50 Hz loop) is spherically
        # smoothed by :meth:`_smooth_root_quat` before it reaches the anchor, which
        # removes the root-acceleration spikes (NaN QACC at DOF 0) the unsmoothed
        # trajectory caused. ``root_smoothing_alpha`` in (0, 1] is the per-tick blend
        # toward the incoming quat (smaller = smoother/laggier, 1 = no smoothing).
        self.enable_smpl_root = enable_smpl_root
        self._root_smoothing_alpha = float(np.clip(root_smoothing_alpha, 1e-3, 1.0))
        self._smoothed_root_quat: np.ndarray | None = None

        # Tracks the previous keyboard held-key set so discrete controls (mode,
        # motion set, replan, e-stop, WASD direction) fire once per physical press
        # instead of every 50 Hz tick while the key is held.
        self._prev_keys: set[str] = set()
        # Edge state for the PICO A+B / X+Y locomotion-mode cycle (3-point teleop).
        self._prev_loco_mode_pair: tuple[bool, bool] = (False, False)

        # Startup blend: ease from the robot's initial pose into the first commanded
        # policy targets over INIT_RAMP_S (captured on the first control tick).
        self._init_ramp_steps = max(1, round(INIT_RAMP_S / CONTROL_DT))
        self._init_step = 0
        self._start_pose: dict[str, float] = {}

        # Tick counter for the dense whole-body (OpenHLM, mode-0) path's encoder cadence.
        self._wb_step = 0
        # Rolling 50-frame reference trajectory (ref29 + anchor quat) built from the
        # stream of per-tick whole-body commands, fed to the encoder as a batch.
        self._wb_traj: deque[np.ndarray] = deque(maxlen=50)
        self._wb_quat_traj: deque[np.ndarray] = deque(maxlen=50)

        # Optional: subscribe directly to the rt/smpl headset stream so full-body
        # teleop works with ANY teleoperator (e.g. --teleop.type=unitree_g1 for the
        # estop/joystick) before the dedicated pico_headset teleop exists.
        self._smpl_host = smpl_host
        self._smpl_port = smpl_port
        self._smpl_stream = None
        if enable_smpl_stream:
            self._init_smpl_stream()

        logger.info(
            "SONIC ready: %s (default mode: %s, smpl_stream=%s)",
            MOTION_SETS[0][0],
            LM(self.ms.mode).name,
            self._smpl_stream is not None,
        )

    @property
    def initialized(self) -> bool:
        return self._runtime.initialized

    @property
    def reference_authority(self) -> bool:
        return self._runtime.reference_authority

    @property
    def execution_feedback_pending(self) -> bool:
        return self._runtime.execution_feedback_pending or self.controller.execution_feedback_pending

    @property
    def reference_install_epoch(self) -> int:
        return self._runtime.reference_install_epoch

    @property
    def matching_reference_installed(self) -> bool:
        return self._runtime.matching_reference_installed

    def initialize_from_lowstate(self, lowstate) -> bool:
        """Initialize planner/history from LowState without a policy tick."""

        if lowstate is None:
            raise ValueError("SONIC initialization requires a LowState")
        return self._runtime.initialize_from_observation(lowstate_to_obs(lowstate))

    def request_reference(self, lowstate) -> None:
        """Begin a planning-only measured rebase for a new intent."""

        if lowstate is None:
            raise ValueError("SONIC reference request requires a LowState")
        self._runtime.request_reference(lowstate_to_obs(lowstate))

    def poll_reference(self) -> bool:
        """Poll/blend an async reference without running the policy."""

        return self._runtime.poll_reference()

    def acquire_reference(
        self,
        lowstate,
        *,
        executed_target,
        expected_install_epoch: int | None = None,
    ) -> None:
        """Rebase policy history and acquire a matching planner reference."""

        if lowstate is None:
            raise ValueError("SONIC authority acquisition requires a LowState")
        self._runtime.acquire_reference(
            lowstate_to_obs(lowstate),
            executed_target=executed_target,
            expected_install_epoch=expected_install_epoch,
        )

    def revoke_reference_authority(
        self,
        *,
        discard_reference: bool = True,
        discard_pending_execution: bool = False,
    ) -> None:
        self._runtime.revoke_reference_authority(
            discard_reference=discard_reference,
            discard_pending_execution=discard_pending_execution,
        )

    def record_executed_target(self, target):
        """Record a target only after its complete actuator publish succeeds."""

        if self._runtime.execution_feedback_pending:
            return self._runtime.record_executed_target(target)
        return self.controller.record_executed_target(target)

    def _init_smpl_stream(self) -> None:
        # Lazy import so the zmq dependency is only required when streaming is on.
        from lerobot.teleoperators.pico_headset.smpl_stream import (
            DEFAULT_SMPL_HOST,
            DEFAULT_SMPL_PORT,
            SmplStream,
        )

        host = self._smpl_host or DEFAULT_SMPL_HOST
        port = self._smpl_port or DEFAULT_SMPL_PORT
        self._smpl_stream = SmplStream(host=host, port=port)
        logger.info("SONIC subscribed to rt/smpl @ tcp://%s:%d", host, port)

    def _enter_wholebody(self) -> None:
        """Switch into SMPL whole-body tracking (encode_mode 2)."""
        self.controller.encode_mode = 2
        self.controller.reinit_heading = True
        logger.info("SONIC: SMPL stream active -> whole-body tracking (mode 2)")

    def _enter_3point(self) -> None:
        """Switch into 3-point VR upper-body teleop (encode_mode 1).

        The upper body tracks the VR wrist/neck targets while the lower body /
        locomotion keeps running off the planner (joystick/keyboard-driven).
        """
        self.controller.encode_mode = 1
        self.controller.playing = True
        self.controller.reinit_heading = True
        self.ms.needs_replan = True
        logger.info("SONIC: 3-point VR active -> upper-body tracking + planner locomotion (mode 1)")

    def _exit_wholebody(self) -> None:
        """Revert to locomotion/standing (encode_mode 0) after a teleop reference is lost.

        Mirrors the 'M' toggle in sonic.py so the handoff is clean: the robot holds
        a standing reference and (if a joystick teleop is attached) can be driven.
        """
        self.controller.encode_mode = 0
        self.controller.playing = True
        self.controller.reinit_heading = True
        self.ms.needs_replan = True
        logger.warning("SONIC: teleop reference lost/stale -> reverting to locomotion (standing)")

    def _process_keyboard(self, action: dict | None) -> None:
        """Translate a native KeyboardTeleop's held-key set into MovementState.

        Mirrors the standalone SONIC demo's keyboard mapping so locomotion (mode 0/1)
        can be driven with ``--teleop.type=keyboard`` instead of the PICO SMPL stream.
        Discrete controls act on newly-pressed keys (edge-detected against the previous
        tick); inherently-continuous controls (facing turn, height, speed) integrate a
        small per-tick delta while the key is held so they feel smooth at 50 Hz.

        Controls: WASD move, Q/E turn, 1-8 select mode, 9/0 speed down/up,
        -/= height down/up, R replan, Space emergency-stop -> IDLE.
        """
        if not self._allow_operator_intent or action is None:
            return
        keys = action.get(KEYBOARD_KEYS_FIELD)
        if keys is None:
            return  # No KeyboardTeleop attached; leave joystick/SMPL paths untouched.

        ms, controller = self.ms, self.controller
        held = {k.lower() if isinstance(k, str) and len(k) == 1 else k for k in keys}
        prev = self._prev_keys
        pressed = held - prev  # newly-pressed this tick (edge)
        self._prev_keys = held

        # ── Discrete: fire once per press ────────────────────────────────────
        if "space" in pressed:
            ms.mode = LM.IDLE
            ms.speed = ms.height = -1.0
            ms.has_movement = False
            ms.needs_replan = True
            controller.playing = False
            controller.reinit_heading = True
            logger.info("SONIC keyboard: EMERGENCY STOP -> IDLE")
        if "r" in pressed:
            ms.needs_replan = True
        if "n" in pressed or "p" in pressed:
            step = 1 if "n" in pressed else -1
            ms.motion_set_idx = (ms.motion_set_idx + step) % len(MOTION_SETS)
            logger.info("SONIC keyboard: motion set -> %s", MOTION_SETS[ms.motion_set_idx][0])
        for digit in ("1", "2", "3", "4", "5", "6", "7", "8"):
            if digit in pressed:
                idx = int(digit) - 1
                modes = MOTION_SETS[ms.motion_set_idx][1]
                if 0 <= idx < len(modes):
                    ms.mode = modes[idx]
                    ms.has_movement = False
                    ms.needs_replan = True
                    controller.playing = True
                    controller.reinit_heading = True
                    logger.info("SONIC keyboard: mode -> %s", LM(ms.mode).name)
        # WASD sets the movement direction relative to current facing (press to set,
        # Space to stop) to match the standalone demo.
        if "w" in pressed:
            ms.movement_angle = ms.facing_angle
        elif "s" in pressed:
            ms.movement_angle = ms.facing_angle + math.pi
        elif "a" in pressed:
            ms.movement_angle = ms.facing_angle + math.pi / 2
        elif "d" in pressed:
            ms.movement_angle = ms.facing_angle - math.pi / 2
        if pressed & {"w", "a", "s", "d"}:
            ms.has_movement = True
            ms.needs_replan = True

        # ── Continuous: integrate a small delta while held ───────────────────
        if "q" in held:
            ms.facing_angle += 0.02
            controller.delta_heading += 0.02
        if "e" in held:
            ms.facing_angle -= 0.02
            controller.delta_heading -= 0.02
        if "0" in held:
            ms.speed = min(5.0, (ms.speed if ms.speed >= 0 else 1.0) + 0.02)
        if "9" in held:
            ms.speed = max(0.0, (ms.speed if ms.speed >= 0 else 1.0) - 0.02)
        if "=" in held:
            ms.height = min(1.0, (ms.height if ms.height >= 0 else DEFAULT_HEIGHT) + 0.005)
        if "-" in held:
            ms.height = max(0.1, (ms.height if ms.height >= 0 else DEFAULT_HEIGHT) - 0.005)

    def _process_pico_loco(self, axes: np.ndarray, buttons: np.ndarray) -> None:
        """Drive locomotion from the PICO controller sticks/buttons (encode_mode 1).

        Mirrors gear_sonic's ``PlannerLoop`` VR-3PT tick: left/right sticks steer
        movement/facing/speed via :func:`apply_pico_loco_axes` (the faithful gear_sonic
        yaw-accumulator + mode-dependent speed curves, not the keyboard-parity map), and
        A+B / X+Y edge-cycle the locomotion mode within the current motion set.
        """
        if not self._allow_operator_intent:
            return
        lx, ly, rx, ry = (float(v) for v in axes)
        apply_pico_loco_axes(lx, ly, rx, ry, self.ms)

        # Mode cycling: step linearly through the LocomotionMode enum (A+B = next,
        # X+Y = previous), exactly like gear_sonic's PlannerLoop — so the operator can
        # reach squat/kneel/crawl, not just the modes in one UI motion set.
        a, b, x, y = (v > 0.5 for v in buttons)
        ab_now, xy_now = (a and b), (x and y)
        ab_prev, xy_prev = self._prev_loco_mode_pair
        mode = int(self.ms.mode)
        if ab_now and not ab_prev:
            mode = min(int(LM.INJURED_WALK), mode + 1)
        elif xy_now and not xy_prev:
            mode = max(int(LM.IDLE), mode - 1)
        if mode != int(self.ms.mode):
            self.ms.mode = LM(mode)
            self.ms.needs_replan = True
            self.controller.playing = True
            logger.info("SONIC 3-point: locomotion mode -> %s", LM(self.ms.mode).name)
        self._prev_loco_mode_pair = (ab_now, xy_now)

    def _run_wholebody34(self, obs: dict, wb: np.ndarray) -> dict:
        """Feed a dense 34-D OpenHLM whole-body command as the mode-0 encoder reference.

        The 29 joint targets are held across the encoder lookahead window (zero
        velocity) and the root roll/pitch set the anchor orientation, then the
        encoder/decoder run directly (planner bypassed). One command per tick, so the
        VLA's commanded pose is what SONIC tracks.
        """
        if not self._runtime.initialized:
            self._runtime.initialize_from_observation(obs)
        if not self._runtime.reference_authority:
            if not self._runtime.auto_reference_authority:
                return {}
            q, _dq, _ang, _quat = observation_state(obs, require_complete=True)
            self._runtime.acquire_reference(obs, executed_target=q)

        ref, anchor = _wb34_to_reference(wb)
        c = self.controller
        if c.encode_mode != 0:
            c.encode_mode = 0
            c.reinit_heading = True
        # Capture the heading/anchor reference on the first whole-body tick. The
        # controller only latches ``init_ref_quat`` (and the base heading) inside
        # ``step()`` when ``first_motion or reinit_heading`` — but it already boots in
        # mode 0, so the mode-switch guard above misses the very first command and the
        # anchor would stay identity. This mirrors the GEAR reference, which seeds
        # ``init_ref_quat`` from the first anchor. Must run before the buffers below so
        # ``step()`` latches ``motion_body_quats[0]`` = this tick's anchor.
        if self._wb_step == 0:
            c.reinit_heading = True

        # Accumulate the per-tick commands into a rolling 50-frame reference
        # trajectory so the encoder's 10-frame, step-5 lookahead sees an actual
        # motion sequence (with velocities) instead of one repeated pose. 50 frames
        # == chunk horizon == 10 lookahead frames × step 5.
        self._wb_traj.append(ref)
        self._wb_quat_traj.append(anchor)
        traj = np.asarray(self._wb_traj, np.float32)  # (L, 29), oldest -> newest
        quats = np.asarray(self._wb_quat_traj, np.float32)  # (L, 4)
        n = len(traj)
        # Per-frame velocities from finite differences (rad/s at the control rate).
        vel = np.zeros_like(traj)
        if n > 1:
            vel[1:] = (traj[1:] - traj[:-1]) / CONTROL_DT
            vel[0] = vel[1]
        with c.motion_lock:
            c.motion_joint_positions[:n] = traj
            c.motion_joint_velocities[:n] = vel
            c.motion_body_quats[:n] = quats
            c.motion_body_pos[:n] = 0.0
            c.motion_timesteps = n
            c.ref_cursor = 0
        c.playing = True
        do_enc = True
        out = c.step(obs, update_encoder=do_enc, debug=False)
        if self._wb_step % 25 == 0:
            tgt = np.array([out[f"{m.name}.q"] for m in G1_29_JointIndex], np.float32)
            logger.info(
                "[WB34] step=%d |ref|mean=%.3f |target|mean=%.3f target_std=%.3f init_ref_quat=%s",
                self._wb_step,
                float(np.abs(ref).mean()),
                float(np.abs(tgt).mean()),
                float(tgt.std()),
                np.round(c.init_ref_quat, 3).tolist(),
            )
        self._wb_step += 1
        return out

    def _smooth_root_quat(self, root_quat: np.ndarray | None) -> np.ndarray | None:
        """Spherically smooth the per-frame SMPL root quaternion (mode-2 anchor).

        The reference root trajectory is authored at ~30 Hz and consumed at 50 Hz, so
        the raw per-tick quat steps unevenly and injects root-acceleration spikes into
        the anchor. This keeps a persistent estimate and shortest-path nlerp-slerps it
        toward each incoming (unit) quat by ``root_smoothing_alpha``, yielding a
        continuous, rate-matched heading. Quaternions are scalar-first (w, x, y, z).
        Returns ``None`` (leaving the anchor self-driven) for an invalid/zero input.
        """
        if root_quat is None:
            self._smoothed_root_quat = None
            return None
        q = np.asarray(root_quat, np.float64)
        n = np.linalg.norm(q)
        if n < 1e-8:
            return self._smoothed_root_quat
        q = q / n
        if self._smoothed_root_quat is None:
            self._smoothed_root_quat = q
        else:
            prev = self._smoothed_root_quat
            if np.dot(prev, q) < 0.0:  # shortest-path: quats double-cover SO(3)
                q = -q
            blended = prev + self._root_smoothing_alpha * (q - prev)
            self._smoothed_root_quat = blended / (np.linalg.norm(blended) + 1e-12)
        return self._smoothed_root_quat.astype(np.float32)

    def _startup_blend(self, obs: dict, out: dict) -> dict:
        """Ease into policy control at startup: for the first ``INIT_RAMP_S`` seconds,
        interpolate between the robot's pose captured on the first tick and the policy's
        live commanded target, so the handoff has no snap.

        ``out`` is the policy's ``<joint>.q`` target dict for this tick; the blend ratio
        climbs 0->1 over the ramp, after which the raw policy target passes through.
        """
        if not self._startup_blend_enabled or self._init_step >= self._init_ramp_steps or not out:
            return out
        if self._init_step == 0:
            # Capture the robot's actual pose as the interpolation start point.
            self._start_pose = {
                f"{m.name}.q": float(obs.get(f"{m.name}.q", DEFAULT_ANGLES[m.value]))
                for m in G1_29_JointIndex
            }
        self._init_step += 1
        ratio = min(1.0, self._init_step / self._init_ramp_steps)
        blended = {
            k: self._start_pose.get(k, float(tgt)) * (1.0 - ratio) + float(tgt) * ratio
            for k, tgt in out.items()
        }
        if self._init_step >= self._init_ramp_steps:
            logger.info("SONIC startup blend complete -> full policy control")
        return blended

    def run_step(self, action: dict, lowstate) -> dict:
        if lowstate is None:
            return {}
        obs = lowstate_to_obs(lowstate)

        # Keyboard teleop (native KeyboardTeleop) drives the same locomotion intent
        # the joystick does; applied before the SMPL check so whole-body tracking
        # still takes priority when a headset stream is present.
        self._process_keyboard(action)

        # Prefer SMPL delivered via the teleop action (pico_headset). Fall back to a
        # direct rt/smpl subscription when enabled (enable_smpl_stream). A stale
        # stream (headset silent past its timeout) is treated as "no SMPL" so the
        # robot doesn't stay frozen tracking the last pose.
        # Dense whole-body command (OpenHLM / pi0.5 joint interface) takes priority:
        # a single 34-D vector drives the mode-0 joint reference directly.
        wb = _extract_wb34_from_action(action)
        if wb is not None:
            return self._startup_blend(obs, self._run_wholebody34(obs, wb))
        self._wb_miss = getattr(self, "_wb_miss", 0) + 1
        if self._wb_miss % 50 == 1:
            akeys = [k for k in action if isinstance(k, str)]
            logger.info(
                "[WB34] no wb.*.pos in action this tick (miss=%d). action keys sample: %s",
                self._wb_miss,
                akeys[:8],
            )

        smpl = _extract_smpl_from_action(action)
        root_quat = _extract_root_from_action(action)
        vr3 = _extract_vr3_from_action(action)
        loco = _extract_loco_from_action(action)
        if smpl is None and vr3 is None and self._smpl_stream is not None:
            window = self._smpl_stream.step()
            if self._smpl_stream.has_data and not self._smpl_stream.is_stale:
                smpl = window
                root_quat = np.asarray(self._smpl_stream.root_quat, np.float32)
            # VR3 is independent of the SMPL window: the controller-state source
            # (head + controllers only) sends 3-point targets with no SMPL frame.
            elif self._smpl_stream.has_fresh_vr3:
                vr3 = (self._smpl_stream.vr3_pos, self._smpl_stream.vr3_orn)
                if self._smpl_stream.has_fresh_loco:
                    loco = (self._smpl_stream.loco_axes, self._smpl_stream.loco_buttons)

        if smpl is not None:
            # Full-body whole-body tracking: SMPL drives the reference, not joystick.
            if self.controller.encode_mode != 2:
                self._enter_wholebody()
            self.controller.smpl_joints_10frame_step1 = smpl
            # Root orientation steers the mode-2 anchor/heading, but only when
            # explicitly enabled (see enable_smpl_root); the raw per-frame quat is
            # spherically smoothed first so the 30->50 Hz resample doesn't spike the
            # anchor. Disabled -> anchor stays self-driven.
            self.controller.smpl_root_quat = (
                self._smooth_root_quat(root_quat) if self.enable_smpl_root else None
            )
            out = self._runtime.tick(obs, debug=False, use_joystick=False)
        elif vr3 is not None:
            # 3-point VR teleop: upper body tracks the wrist/neck targets; the lower
            # body / locomotion keeps running off the planner, so the joystick (and
            # keyboard) still steer walking/turning underneath.
            if self.controller.encode_mode != 1:
                self._enter_3point()
            self.controller.vr_3point_local_target = vr3[0]
            self.controller.vr_3point_local_orn_target = vr3[1]
            # Replicate the original encode_mode-1 handling: when the PICO controller
            # sticks are forwarded, drive locomotion from them directly (and skip the
            # wireless-remote joystick read). Otherwise leave the remote/keyboard path.
            if loco is not None and self._allow_operator_intent:
                self._process_pico_loco(loco[0], loco[1])
                out = self._runtime.tick(obs, debug=False, use_joystick=False)
            else:
                out = self._runtime.tick(
                    obs,
                    debug=False,
                    use_joystick=self._allow_operator_intent,
                )
        else:
            # No (or stale) teleop reference: fall back to locomotion so the robot stays balanced.
            if self.controller.encode_mode != 0:
                self.controller.smpl_root_quat = None
                self._smoothed_root_quat = None
                self._exit_wholebody()
            out = self._runtime.tick(
                obs,
                debug=False,
                use_joystick=self._allow_operator_intent,
            )

        # Startup interpolation: blend from the robot's initial pose into the policy's
        # commanded target over INIT_RAMP_S, regardless of mode.
        return self._startup_blend(obs, out)

    def reset(self):
        self._runtime.reset()
        self._init_step = 0  # re-run the startup blend after a reset
        self._start_pose = {}
        self._smoothed_root_quat = None
        self._wb_step = 0
        self._wb_traj.clear()
        self._wb_quat_traj.clear()

    def shutdown(self):
        if self._smpl_stream is not None:
            self._smpl_stream.close()
        self._runtime.shutdown()
