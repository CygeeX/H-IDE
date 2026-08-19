"""Real OfficeV2 MuJoCo regression for the terminal-stop support wrench.

The fixture is a compact reconstruction of one accepted 0.2 m run at its
stop boundary.  It preserves the measured free-root state, 29-DoF state,
frozen SONIC target, and deployed gains, but is deliberately not bit-exact:
the retained Plant root sample and SONIC control sample were about one 4 ms
physics tick apart, and the original contact-solver warm start was not part of
the acceptance record.  This is a deterministic causal A/B, not a substitute
for the signed end-to-end acceptance artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pytest

from hide_mujoco_sonic.plant import stabilized_support_wrench


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "scenes/office_v2/generated/mujoco/office_v2.runtime.xml"

# Last retained, finite stop-boundary state.  Joint arrays follow the XML's
# 29-DoF deploy order; the target is held constant in both A/B branches.
ROOT_QPOS = (
    -6.413516161038488,
    0.006743863908396299,
    0.7833954366893704,
    0.9998416670533792,
    0.0037146310001334192,
    -0.011980106792902373,
    -0.012622178167091484,
)
ROOT_QVEL = (
    -0.0036499502984473848,
    0.07039297357335768,
    -0.011591232986190907,
    0.10450159532584427,
    -0.09171548059110642,
    -0.020151148091213943,
)
JOINT_Q = (
    -0.019397882744669914, -0.0007715715328231454, 0.06552442163228989,
    0.2559010088443756, -0.21296913921833038, 0.005933523178100586,
    -0.06624133139848709, -0.06744789332151413, -0.1350374072790146,
    0.32024723291397095, -0.22167673707008362, 0.02227115072309971,
    -0.0030453873332589865, 0.0002693877322599292, 0.02680952288210392,
    -0.0007745528127998114, 0.2694564759731293, -0.5493819117546082,
    1.1341665983200073, 0.3195675015449524, 0.10635481774806976,
    0.14290033280849457, -0.006499279756098986, -0.27602618932724,
    0.549492597579956, 1.1986243724822998, -0.21708960831165314,
    0.04870866239070892, -0.21500009298324585,
)
JOINT_DQ = (
    0.14358872175216675, -0.20740167796611786, 0.12162496149539948,
    -0.08216063678264618, 0.017877614125609398, 0.11263154447078705,
    -0.1712772250175476, -0.26763781905174255, 0.017201833426952362,
    0.5249475836753845, -0.2032359093427658, 0.12091915309429169,
    -0.24997764825820923, -0.10184374451637268, 0.004532312508672476,
    0.24279402196407318, 0.005403302609920502, -0.025740329176187515,
    0.016802143305540085, 0.20820839703083038, 0.11299790441989899,
    -0.02983187697827816, -0.2053186446428299, 0.014728032983839512,
    0.5094354152679443, 0.0011295088334009051, -0.35006436705589294,
    0.336726576089859, 0.4392004907131195,
)
FROZEN_TARGET = (
    -0.0007225573062896729, -0.02970697544515133, 0.11027891933917999,
    0.23131152987480164, -0.10946926474571228, 0.0047332774847745895,
    -0.0650702565908432, -0.06384848803281784, -0.21357733011245728,
    0.37294718623161316, 0.12767991423606873, 0.047672536224126816,
    0.007464476395398378, 0.016711954027414322, -0.07711179554462433,
    -0.06878675520420074, 0.3602832555770874, -0.5504120588302612,
    1.1256427764892578, 0.3465593457221985, 0.11799628287553787,
    0.13740848004817963, -0.047354236245155334, -0.37139028310775757,
    0.5765565633773804, 1.1788954734802246, -0.2521865665912628,
    0.07033932954072952, -0.18971338868141174,
)
KP = (
    99.09842681884766, 99.09842681884766, 40.179237365722656,
    99.09842681884766, 28.501245498657227, 28.501245498657227,
    99.09842681884766, 99.09842681884766, 40.179237365722656,
    99.09842681884766, 28.501245498657227, 28.501245498657227,
    40.179237365722656, 28.501245498657227, 28.501245498657227,
    14.250622749328613, 14.250622749328613, 14.250622749328613,
    14.250622749328613, 14.250622749328613, 16.77832794189453,
    16.77832794189453, 14.250622749328613, 14.250622749328613,
    14.250622749328613, 14.250622749328613, 14.250622749328613,
    16.77832794189453, 16.77832794189453,
)
KD = (
    6.308801651000977, 6.308801651000977, 2.557889699935913,
    6.308801651000977, 1.8144457340240479, 1.8144457340240479,
    6.308801651000977, 6.308801651000977, 2.557889699935913,
    6.308801651000977, 1.8144457340240479, 1.8144457340240479,
    2.557889699935913, 1.8144457340240479, 1.8144457340240479,
    0.9072228670120239, 0.9072228670120239, 0.9072228670120239,
    0.9072228670120239, 0.9072228670120239, 1.0681414604187012,
    1.0681414604187012, 0.9072228670120239, 0.9072228670120239,
    0.9072228670120239, 0.9072228670120239, 0.9072228670120239,
    1.0681414604187012, 1.0681414604187012,
)

WARNING_NAMES = (
    "mjWARN_BADQPOS",
    "mjWARN_BADQVEL",
    "mjWARN_BADQACC",
    "mjWARN_BADCTRL",
)


@dataclass(frozen=True)
class ReplayResult:
    duration_s: float
    first_qacc_over_1000_s: float | None
    first_joint_speed_over_10_s: float | None
    first_envelope_exit_s: float | None
    warning_counts: tuple[int, ...]
    max_abs_qacc: float
    max_abs_joint_speed: float
    max_abs_yaw_torque_nm: float
    max_kinetic_energy_j: float
    finite: bool


def _torso_pose(mujoco, model, data, torso_id: int) -> np.ndarray:
    pose = np.concatenate((data.xpos[torso_id], data.xquat[torso_id], np.zeros(6)))
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        torso_id,
        pose[7:13],
        0,
    )
    # mj_objectVelocity returns angular then linear velocity.  Upstream's
    # elastic-band pose contract stores world linear then angular velocity.
    angular = pose[7:10].copy()
    pose[7:10] = pose[10:13]
    pose[10:13] = angular
    return pose


def _replay(*, legacy_torso_yaw_feedback: bool, steps: int) -> ReplayResult:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    assert (model.nq, model.nv, model.nu, model.opt.timestep) == (36, 35, 29, 0.004)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert torso_id >= 0

    data.qpos[:7] = ROOT_QPOS
    data.qpos[7:] = JOINT_Q
    data.qvel[:6] = ROOT_QVEL
    data.qvel[6:] = JOINT_DQ
    mujoco.mj_forward(model, data)

    target = np.asarray(FROZEN_TARGET)
    kp = np.asarray(KP)
    kd = np.asarray(KD)
    # Exact process-start support anchor retained from the accepted OfficeV2
    # run.  It is an external fall-arrest reference, never a navigation goal.
    anchor = np.asarray((-6.45, 0.0, 0.847))
    warning_baseline = tuple(
        int(data.warning[int(getattr(mujoco.mjtWarning, name))].number)
        for name in WARNING_NAMES
    )
    first_qacc = None
    first_joint_speed = None
    first_envelope_exit = None
    max_abs_qacc = 0.0
    max_abs_joint_speed = 0.0
    max_abs_yaw_torque = 0.0
    max_kinetic_energy = 0.0
    finite = True

    for _ in range(steps):
        q = data.qpos[7:]
        dq = data.qvel[6:]
        data.ctrl[:] = np.clip(
            kp * (target - q) - kd * dq,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        pose = _torso_pose(mujoco, model, data, int(torso_id))
        raw_wrench = np.zeros(6)
        raw_wrench[:3] = 10000.0 * (anchor - pose[:3]) - 1000.0 * pose[7:10]
        wrench = stabilized_support_wrench(raw_wrench, pose, "safety_stop")
        if legacy_torso_yaw_feedback:
            # Historical counterfactual.  The production helper is not
            # modified: only this A/B branch reinstates the removed equation.
            wrench[5] = -80.0 * pose[12]
        max_abs_yaw_torque = max(max_abs_yaw_torque, abs(float(wrench[5])))
        data.xfrc_applied[int(torso_id)] = wrench
        mujoco.mj_step(model, data)

        elapsed = float(data.time)
        max_abs_qacc = max(max_abs_qacc, float(np.max(np.abs(data.qacc))))
        max_abs_joint_speed = max(
            max_abs_joint_speed,
            float(np.max(np.abs(data.qvel[6:]))),
        )
        mujoco.mj_energyVel(model, data)
        max_kinetic_energy = max(max_kinetic_energy, float(data.energy[1]))
        if first_qacc is None and max_abs_qacc > 1000.0:
            first_qacc = elapsed
        if first_joint_speed is None and max_abs_joint_speed > 10.0:
            first_joint_speed = elapsed
        x, y, z = (float(value) for value in data.qpos[:3])
        if first_envelope_exit is None and not (
            -8.0 <= x <= 8.0 and -6.0 <= y <= 6.0 and 0.20 <= z <= 1.50
        ):
            first_envelope_exit = elapsed
        finite = finite and all(
            np.all(np.isfinite(values))
            for values in (data.qpos, data.qvel, data.qacc, data.ctrl)
        )
        # Stop the known-bad branch before MuJoCo's warning recovery mutates
        # the state.  The envelope exit itself is already a terminal fault.
        if legacy_torso_yaw_feedback and first_envelope_exit is not None:
            break

    warnings = tuple(
        int(data.warning[int(getattr(mujoco.mjtWarning, name))].number) - baseline
        for name, baseline in zip(WARNING_NAMES, warning_baseline, strict=True)
    )
    return ReplayResult(
        duration_s=float(data.time),
        first_qacc_over_1000_s=first_qacc,
        first_joint_speed_over_10_s=first_joint_speed,
        first_envelope_exit_s=first_envelope_exit,
        warning_counts=warnings,
        max_abs_qacc=max_abs_qacc,
        max_abs_joint_speed=max_abs_joint_speed,
        max_abs_yaw_torque_nm=max_abs_yaw_torque,
        max_kinetic_energy_j=max_kinetic_energy,
        finite=finite,
    )


def test_zero_yaw_safety_wrench_is_bounded_but_legacy_torso_feedback_diverges() -> None:
    safe = _replay(legacy_torso_yaw_feedback=False, steps=500)
    assert math.isclose(safe.duration_s, 2.0, abs_tol=1e-12)
    assert safe.finite
    assert safe.warning_counts == (0, 0, 0, 0)
    assert safe.first_qacc_over_1000_s is None
    assert safe.first_joint_speed_over_10_s is None
    assert safe.first_envelope_exit_s is None
    assert safe.max_abs_qacc < 250.0
    assert safe.max_abs_joint_speed < 2.0
    assert safe.max_abs_yaw_torque_nm == 0.0
    assert safe.max_kinetic_energy_j < 1.0

    legacy = _replay(legacy_torso_yaw_feedback=True, steps=25)
    assert legacy.finite
    assert legacy.warning_counts == (0, 0, 0, 0)
    assert legacy.first_qacc_over_1000_s is not None
    assert legacy.first_qacc_over_1000_s <= 0.028
    assert legacy.first_joint_speed_over_10_s is not None
    assert legacy.first_joint_speed_over_10_s <= 0.028
    assert legacy.first_envelope_exit_s is not None
    assert legacy.first_envelope_exit_s <= 0.080
    assert legacy.max_abs_yaw_torque_nm > 20.0
