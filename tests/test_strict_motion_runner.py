from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "strict_motion_runner.py"
SPEC = importlib.util.spec_from_file_location("strict_motion_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def command(vx: float = 0.0, vy: float = 0.0, wz: float = 0.0):
    return runner.TwistCommand(vx, vy, wz)


@pytest.mark.parametrize(
    ("mode", "vx"),
    (
        ("forward_probe", 0.2),
        ("forward", 0.2),
        ("forward_3m", 0.2),
        ("reverse", -0.2),
    ),
)
def test_drive_on_heading_relays_only_the_exact_official_speed(mode: str, vx: float) -> None:
    relay = runner.StrictCommandRelay()
    relay.begin(mode)
    relay.observe_nav2(command(vx=vx), observed_at_s=10.0)

    assert relay.nav2_output(now_s=10.1) == command(vx=vx)
    assert relay.source_sequence == 1

    with pytest.raises(runner.StrictMotionRunnerError, match="official 0.20"):
        relay.observe_nav2(command(vx=math.copysign(0.199, vx)), observed_at_s=10.11)


def test_forward_probe_is_a_fixed_short_official_drive() -> None:
    assert runner.motion_goal("forward_probe") == {
        "distanceM": 0.20,
        "speedMps": runner.OFFICIAL_DRIVE_SPEED_MPS,
    }
    assert runner.action_timeout_s(
        "forward_probe", runner.motion_goal("forward_probe")
    ) == pytest.approx(11.0)


def test_drive_on_heading_rejects_wrong_sign_and_mixed_axes() -> None:
    relay = runner.StrictCommandRelay()
    relay.begin("forward")
    with pytest.raises(runner.StrictMotionRunnerError, match="official 0.20"):
        relay.observe_nav2(command(vx=-0.2), observed_at_s=10.0)
    with pytest.raises(runner.StrictMotionRunnerError, match="mixed-axis"):
        relay.observe_nav2(command(vx=0.2, wz=0.01), observed_at_s=10.0)


@pytest.mark.parametrize(
    ("mode", "wz"),
    (("turn_left", 0.1), ("turn_left", 0.2), ("turn_right", -0.1), ("turn_right", -0.2)),
)
def test_spin_relays_the_official_acceleration_envelope(mode: str, wz: float) -> None:
    relay = runner.StrictCommandRelay()
    relay.begin(mode)
    relay.observe_nav2(command(wz=wz), observed_at_s=1.0)
    assert relay.nav2_output(now_s=1.1) == command(wz=wz)


def test_spin_rejects_wrong_sign_subfloor_and_translation() -> None:
    relay = runner.StrictCommandRelay()
    relay.begin("turn_left")
    for invalid in (command(wz=-0.2), command(wz=0.099), command(vx=0.01, wz=0.2)):
        with pytest.raises(runner.StrictMotionRunnerError):
            relay.observe_nav2(invalid, observed_at_s=1.0)


def test_nav2_output_fails_closed_when_source_stalls() -> None:
    relay = runner.StrictCommandRelay()
    relay.begin("forward")
    relay.observe_nav2(command(vx=0.2), observed_at_s=2.0)
    with pytest.raises(runner.StrictMotionRunnerError, match="stale"):
        relay.nav2_output(now_s=2.151)


def test_nav2_action_contracts_are_version_and_field_pinned() -> None:
    def action(fields: dict[str, str]):
        return SimpleNamespace(
            Goal=SimpleNamespace(get_fields_and_field_types=lambda: fields)
        )

    drive = action(
        {
            "target": "geometry_msgs/Point",
            "speed": "float",
            "time_allowance": "builtin_interfaces/Duration",
        }
    )
    spin = action(
        {
            "target_yaw": "float",
            "time_allowance": "builtin_interfaces/Duration",
        }
    )
    compute = action(
        {
            "goal": "geometry_msgs/PoseStamped",
            "start": "geometry_msgs/PoseStamped",
            "planner_id": "string",
            "use_start": "boolean",
        }
    )
    runner._require_nav2_action_contracts(
        drive,
        spin,
        compute,
        installed_version="1.3.12",
    )

    with pytest.raises(runner.StrictMotionRunnerError, match="not pinned"):
        runner._require_nav2_action_contracts(
            drive,
            spin,
            compute,
            installed_version="1.3.13",
        )
    broken = action({"target": "geometry_msgs/Point"})
    with pytest.raises(runner.StrictMotionRunnerError, match="contracts drifted"):
        runner._require_nav2_action_contracts(
            broken,
            spin,
            compute,
            installed_version="1.3.12",
        )


@pytest.mark.parametrize(
    ("mode", "sign"),
    (("lateral_left", 1.0), ("lateral_right", -1.0)),
)
def test_lateral_step_is_measured_closed_loop_and_mirrored(mode: str, sign: float) -> None:
    relay = runner.StrictCommandRelay()
    relay.begin(mode, start_pose=runner.PlanarPose(1.0, 2.0, math.pi / 2.0))

    assert relay.lateral_output(runner.PlanarPose(1.0, 2.0, math.pi / 2.0)) == command(
        vy=sign * 0.4
    )
    # At yaw=+pi/2, body-left is world -X.
    near_goal = runner.PlanarPose(1.0 - sign * 0.48, 2.0, math.pi / 2.0)
    assert relay.lateral_output(near_goal).is_zero
    assert relay.lateral_complete is True


def test_lateral_step_rejects_opposite_drift_yaw_and_overshoot() -> None:
    start = runner.PlanarPose(0.0, 0.0, 0.0)

    opposite = runner.StrictCommandRelay()
    opposite.begin("lateral_left", start_pose=start)
    with pytest.raises(runner.StrictMotionRunnerError, match="opposite"):
        opposite.lateral_output(runner.PlanarPose(0.0, -0.041, 0.0))

    cross = runner.StrictCommandRelay()
    cross.begin("lateral_left", start_pose=start)
    with pytest.raises(runner.StrictMotionRunnerError, match="longitudinal"):
        cross.lateral_output(runner.PlanarPose(0.121, 0.1, 0.0))

    yaw = runner.StrictCommandRelay()
    yaw.begin("lateral_left", start_pose=start)
    with pytest.raises(runner.StrictMotionRunnerError, match="yaw"):
        yaw.lateral_output(
            runner.PlanarPose(0.0, 0.1, runner.LATERAL_MAX_YAW_DRIFT_RAD + 1e-5)
        )

    overshoot = runner.StrictCommandRelay()
    overshoot.begin("lateral_left", start_pose=start)
    with pytest.raises(runner.StrictMotionRunnerError, match="overshot"):
        overshoot.lateral_output(runner.PlanarPose(0.0, 0.531, 0.0))


def test_nav2_motion_without_an_active_case_is_rejected_but_zero_is_safe() -> None:
    relay = runner.StrictCommandRelay()
    relay.observe_nav2(command(), observed_at_s=1.0)
    assert relay.nav2_output(now_s=1.0).is_zero
    with pytest.raises(runner.StrictMotionRunnerError, match="without an active"):
        relay.observe_nav2(command(vx=0.2), observed_at_s=1.1)


def test_import_and_non_execute_cli_are_inert(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist.json"
    result = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--mode",
            "forward",
            "--case-id",
            "inert",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "refusing live motion" in result.stderr
    assert not output.exists()


def test_success_persists_result_before_trace_commit_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeBackend:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    class FakeExecutor:
        def __init__(self, *_args) -> None:
            pass

        def run_standalone(self, *, mode: str, case_id: str):
            assert mode == "forward"
            assert case_id == "atomic-order"
            return {"trace": True}, {"pass": True}

    written: list[tuple[Path, object]] = []
    monkeypatch.setattr(runner, "RosLiveBackend", FakeBackend)
    monkeypatch.setattr(runner, "RuntimeSnapshotCollector", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "SonicEvidenceClient", lambda: object())
    monkeypatch.setattr(runner, "StrictMotionCaseExecutor", FakeExecutor)
    monkeypatch.setattr(
        runner,
        "_write_exclusive_json",
        lambda path, value: written.append((Path(path), value)),
    )
    trace_path = tmp_path / "trace.json"

    assert runner.main(
        [
            "--mode",
            "forward",
            "--case-id",
            "atomic-order",
            "--output",
            str(trace_path),
            "--execute",
        ]
    ) == 0
    assert written == [
        (tmp_path / "trace.json.result.json", {"pass": True}),
        (trace_path, {"trace": True}),
    ]


class FakeReader:
    def __init__(self, plant: dict, sonic: dict) -> None:
        self.plant = plant
        self.sonic = sonic

    def read(self, url: str, *, timeout_s: float = 0.5) -> dict:
        del timeout_s
        return self.plant if url == runner.PLANT_STATE_URL else self.sonic


class FakeRos:
    def __init__(self) -> None:
        self.command = command(vx=0.2)

    def bridge_health(self) -> dict:
        return {
            "schemaVersion": runner.BRIDGE_HEALTH_SCHEMA,
            "healthy": True,
            "bootId": "41000000-0000-4000-8000-000000000004",
            "sessionId": "51000000-0000-4000-8000-000000000005",
            "sequence": 50,
            "sonicDriverBootId": "21000000-0000-4000-8000-000000000002",
            "sonicDriverSessionId": "31000000-0000-4000-8000-000000000003",
            "fault": None,
        }

    def nav2_health(self) -> dict:
        return {
            "schemaVersion": runner.NAV2_HEALTH_SCHEMA,
            "healthy": True,
            "bootId": "61000000-0000-4000-8000-000000000006",
            "sessionId": "71000000-0000-4000-8000-000000000007",
            "sequence": 80,
            "fault": None,
        }

    def cmd_vel_publisher_identity(self) -> dict:
        return {
            "count": 1,
            "nodeName": "/strict_motion_runner",
            "publisherGid": "01020304",
            "topic": "/cmd_vel",
        }

    def last_published_command(self):
        return self.command

    def action_statuses(self) -> dict[str, bool]:
        return {
            "driveOnHeadingActive": True,
            "navigateToPoseActive": False,
            "followPathActive": False,
            "spinActive": False,
        }


def plant_state() -> dict:
    return {
        "status": "running",
        "physicsStatus": "running",
        "observationStatus": "running",
        "producerBootId": "11000000-0000-4000-8000-000000000001",
        "sessionId": "12000000-0000-4000-8000-000000000001",
        "sequence": 100,
        "simulationTimeS": 12.5,
        "pose": {
            "x": 1.0,
            "y": 2.0,
            "z": 0.78,
            "qw": 1.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
        },
        "rootVelocity": {"linearSpeedMps": 0.2, "angularSpeedRps": 0.0},
        "support": {
            "fallArrestEnabled": False,
            "fallArrestReleaseConfirmedEpoch": 17,
            "fallArrestReleaseRejectedEpoch": None,
        },
        "viewerWorker": {
            "status": "running",
            "ready": True,
            "alive": True,
            "workerBootId": "81000000-0000-4000-8000-000000000008",
            "lastSuccessSequence": 90,
            "authorityFeedbackEnabled": False,
            "keyCallbackRegistered": False,
            "lastError": None,
        },
        "dynamicsEvidence": {"schemaVersion": "fixture"},
    }


def sonic_state() -> dict:
    return {
        "status": "running",
        "motion_ready": True,
        "last_error": None,
        "driver_boot_id": "21000000-0000-4000-8000-000000000002",
        "driver_session_id": "31000000-0000-4000-8000-000000000003",
        "control_tick_ms": 12_500,
        "navigation_task_active": True,
    }


def first_person_status(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": runner.FIRST_PERSON_STATUS_SCHEMA,
                "bootId": "91000000-0000-4000-8000-000000000009",
                "pid": 4242,
                "status": "running",
                "healthy": True,
                "mapped": True,
                "sequence": 40,
                "plantBootId": "11000000-0000-4000-8000-000000000001",
                "plantSessionId": "12000000-0000-4000-8000-000000000001",
                "plantSequence": 99,
                "simulationTimeS": 12.4,
                "captureAgeMs": 5.0,
                "lastSuccessAtUnixNs": 1_800_000_000_000_000_000,
                "failureCount": 0,
                "lastError": None,
            }
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def test_snapshot_collector_binds_every_runtime_identity_and_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = tmp_path / "first-person.json"
    first_person_status(status)
    monkeypatch.setattr(runner, "_inactive_user_unit", lambda _unit: "inactive")
    monkeypatch.setattr(runner, "_require_port_closed", lambda _port: None)
    collector = runner.RuntimeSnapshotCollector(
        FakeRos(),
        reader=FakeReader(plant_state(), sonic_state()),
        first_person_status=status,
    )

    sample = collector.collect_once()

    assert sample["runtimeHealth"]["plant"]["sessionId"] == (
        "12000000-0000-4000-8000-000000000001"
    )
    assert sample["runtimeHealth"]["sonic"]["sessionId"] == (
        "31000000-0000-4000-8000-000000000003"
    )
    assert sample["runtimeHealth"]["rosBridge"]["sequence"] == 50
    assert sample["runtimeHealth"]["nav2"]["sequence"] == 80
    assert sample["runtimeHealth"]["nativeViewer"]["sequence"] == 90
    assert sample["runtimeHealth"]["firstPersonViewer"]["sequence"] == 40
    assert sample["isolationProof"]["cmdVelPublisher"]["nodeName"] == (
        "/strict_motion_runner"
    )
    assert sample["command"] == {"vx": 0.2, "vy": 0.0, "wz": 0.0}


def test_snapshot_collector_rejects_bridge_sonic_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status = tmp_path / "first-person.json"
    first_person_status(status)
    monkeypatch.setattr(runner, "_inactive_user_unit", lambda _unit: "inactive")
    monkeypatch.setattr(runner, "_require_port_closed", lambda _port: None)
    ros = FakeRos()
    original = ros.bridge_health
    ros.bridge_health = lambda: {  # type: ignore[method-assign]
        **original(),
        "sonicDriverSessionId": "32000000-0000-4000-8000-000000000003",
    }
    collector = runner.RuntimeSnapshotCollector(
        ros,
        reader=FakeReader(plant_state(), sonic_state()),
        first_person_status=status,
    )

    with pytest.raises(runner.StrictMotionRunnerError, match="bridge/SONIC"):
        collector.collect_once()


def test_secure_first_person_status_rejects_group_writable_file(tmp_path: Path) -> None:
    status = tmp_path / "first-person.json"
    first_person_status(status)
    os.chmod(status, 0o620)
    with pytest.raises(runner.StrictMotionRunnerError, match="unsafe"):
        runner.read_secure_status(status)


def _executor_sample(t_s: float) -> dict:
    return {
        "tS": t_s,
        "motionIntentActive": False,
        "command": {"vx": 0.0, "vy": 0.0, "wz": 0.0},
        "root": {"linearSpeedMps": 0.0, "angularSpeedRps": 0.0},
        "nav2Actions": {
            "driveOnHeadingActive": False,
            "navigateToPoseActive": False,
            "followPathActive": False,
            "spinActive": False,
        },
        "plantSupport": {
            "fallArrestEnabled": False,
            "fallArrestReleaseConfirmedEpoch": 17,
            "fallArrestReleaseRejectedEpoch": None,
        },
        "plantDynamicsEvidence": {"supportBandEnabled": False},
    }


class FakeExecutorBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.result_checks = 0
        self.stop_future = SimpleNamespace(done=lambda: True)

    def wait_ready(self) -> None:
        self.calls.append("ready")

    def control_arm(self) -> str:
        self.calls.append("arm")
        return "armed"

    def begin_relay(self, mode: str) -> None:
        self.calls.append(f"begin:{mode}")

    def send_drive(self, distance_m: float, *, timeout_s: float):
        assert distance_m == 1.0
        assert timeout_s == 15.0
        self.calls.append("send_drive")
        return SimpleNamespace(kind="DriveOnHeading")

    def action_done(self, _action) -> bool:
        self.result_checks += 1
        return self.result_checks >= 2

    def finish_action(self, _action) -> None:
        self.calls.append("finish")

    def raise_if_faulted(self) -> None:
        return None

    def stop_relay(self) -> None:
        self.calls.append("zero")

    def normal_stop(self) -> str:
        self.calls.append("normal_stop")
        return "confirmed"

    def begin_normal_stop(self):
        self.calls.append("normal_stop_begin")
        return self.stop_future

    def normal_stop_done(self, future) -> bool:
        return future.done()

    def finish_normal_stop(self, _future) -> None:
        self.calls.append("normal_stop_finish")

    def cancel_active(self) -> bool:
        self.calls.append("cancel")
        return True


class FakeExecutorCollector:
    def __init__(self, samples: list[dict]) -> None:
        self.samples = iter(samples)

    def collect_once(self) -> dict:
        return {
            "runtimeHealth": {
                "plant": {
                    "bootId": "11000000-0000-4000-8000-000000000001",
                    "sessionId": "12000000-0000-4000-8000-000000000001",
                }
            }
        }

    def isolation_manifest(self) -> dict:
        return {"source": "fixture"}

    def wait_next(self) -> dict:
        return next(self.samples)


class FakeExecutorEvidence:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def begin(self, **_kwargs) -> dict:
        self.calls.append("begin")
        return {
            "support_release_confirmed_epoch": 17,
            "policy_activation_epoch": 17,
        }

    def finalize(self, **_kwargs) -> dict:
        self.calls.append("finalize")
        return {
            "policyActivationEpoch": 17,
            "supportReleaseConfirmedEpoch": 17,
        }

    def abort(self, **_kwargs) -> bool:
        self.calls.append("abort")
        return True


class FakeDetourBackend(FakeExecutorBackend):
    def __init__(self) -> None:
        super().__init__()
        self.result_checks = 999

    def latest_map_pose(self):
        return runner.PoseObservation(
            runner.PlanarPose(-6.45, 0.0, 0.0),
            "map",
            1.0,
        )

    def compute_path(self, _goal) -> dict:
        self.calls.append("compute_path")
        return {
            "frame_id": "map",
            "poses": [
                {"x": -6.45, "y": 0.0, "yaw": 0.0},
                {"x": -7.0, "y": -0.3, "yaw": -2.6},
                {"x": -7.4, "y": -0.45, "yaw": 0.0},
            ],
        }

    def begin_relay(self, mode: str, **kwargs) -> None:
        self.calls.append(f"begin:{mode}:{kwargs['primitive']}")

    def send_spin(self, target_yaw_rad: float, *, timeout_s: float):
        assert target_yaw_rad < 0.0
        assert timeout_s > 10.0
        self.calls.append("send_spin")
        return SimpleNamespace(kind="Spin")

    def send_drive(self, distance_m: float, *, timeout_s: float):
        assert distance_m == 0.5
        assert timeout_s == 12.5
        self.calls.append("send_drive")
        return SimpleNamespace(kind="DriveOnHeading")

    def action_done(self, _action) -> bool:
        return True

    def latest_odom(self):
        return runner.PoseObservation(runner.PlanarPose(0.0, 0.0, 0.0), "odom", 1.0)


class FakeStaticPlanner:
    def __init__(self, *, skip: bool = False) -> None:
        self.skip = skip

    def build_primitive_plan(self, _path, **_kwargs) -> dict:
        return {
            "primitives": [{"sequence": 0}, {"sequence": 1}],
            "plan_sha256": "a" * 64,
            "map_provenance": {"image_sha256": "b" * 64},
        }

    def verify_primitive_plan(self, plan, _path, **_kwargs):
        return plan

    def resolve_live_primitive(self, _plan, index, _path, _pose, **_kwargs):
        if self.skip:
            return {
                "primitive_sequence": index,
                "plan_sha256": "a" * 64,
                "fresh_odometry_used": True,
                "disposition": "skip_within_heading_tolerance",
                "action_goal": None,
            }
        if index == 0:
            return {
                "primitive_sequence": 0,
                "plan_sha256": "a" * 64,
                "fresh_odometry_used": True,
                "disposition": "send_action",
                "action_type": "nav2_msgs/action/Spin",
                "action_goal": {"target_yaw_rad": -0.5},
            }
        return {
            "primitive_sequence": 1,
            "plan_sha256": "a" * 64,
            "fresh_odometry_used": True,
            "disposition": "send_action",
            "action_type": "nav2_msgs/action/DriveOnHeading",
            "action_goal": {
                "target": {"x": 0.5, "y": 0.0, "z": 0.0},
                "speed_mps": 0.2,
            },
        }


class FakeDetourEvidence(FakeExecutorEvidence):
    def finalize(self, **_kwargs) -> dict:
        self.calls.append("finalize")
        records = [
            {
                "controlTick": 1000 + index,
                "tS": round(index * 0.1, 10),
            }
            for index in range(30)
        ]
        return {
            "policyActivationEpoch": 17,
            "supportReleaseConfirmedEpoch": 17,
            "records": records,
        }


def test_standalone_executor_orders_authority_evidence_action_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeExecutorBackend()
    samples = [_executor_sample(value / 10.0) for value in range(13)]
    collector = FakeExecutorCollector(samples)
    evidence = FakeExecutorEvidence()
    monkeypatch.setattr(
        runner,
        "_load_file_module",
        lambda *_args: SimpleNamespace(
            evaluate_trace=lambda trace: {"pass": trace["mode"] == "forward"}
        ),
    )
    executor = runner.StrictMotionCaseExecutor(
        backend,
        collector,
        evidence,
        clock=lambda: 1.0,
    )

    trace, result = executor.run_standalone(mode="forward", case_id="forward-run-1")

    assert result == {"pass": True}
    assert trace["action"]["goal"] == {"distanceM": 1.0, "speedMps": 0.2}
    assert trace["action"]["activationProvenance"] == "initial_support_release"
    assert backend.calls == [
        "ready",
        "normal_stop",
        "arm",
        "begin:forward",
        "send_drive",
        "finish",
        "zero",
        "normal_stop_begin",
        "normal_stop_finish",
    ]
    assert evidence.calls == ["begin", "finalize"]


def test_standalone_executor_failure_cancels_stops_and_aborts() -> None:
    backend = FakeExecutorBackend()

    class BrokenCollector(FakeExecutorCollector):
        def wait_next(self) -> dict:
            raise runner.StrictMotionRunnerError("evidence lost")

    evidence = FakeExecutorEvidence()
    executor = runner.StrictMotionCaseExecutor(
        backend,
        BrokenCollector([]),
        evidence,
        clock=lambda: 1.0,
    )

    with pytest.raises(runner.StrictMotionRunnerError, match="evidence lost"):
        executor.run_standalone(mode="forward", case_id="forward-run-failed")

    assert backend.calls[-3:] == ["cancel", "zero", "normal_stop"]
    assert evidence.calls == ["begin", "abort"]


def test_detour_executor_uses_only_fresh_resolved_official_primitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeDetourBackend()
    samples = [_executor_sample(index / 10.0) for index in range(17)]
    # Each primitive has one active sample; the remaining samples are stops.
    samples[1]["motionIntentActive"] = True
    samples[1]["nav2Actions"]["spinActive"] = True
    samples[1]["command"]["wz"] = -0.2
    samples[4]["motionIntentActive"] = True
    samples[4]["nav2Actions"]["driveOnHeadingActive"] = True
    samples[4]["command"]["vx"] = 0.2
    collector = FakeExecutorCollector(samples)
    evidence = FakeDetourEvidence()
    planner = FakeStaticPlanner()

    def load(name: str, _path: Path):
        if "planner" in name:
            return planner
        return SimpleNamespace(evaluate_trace=lambda trace: {"pass": len(trace["action"]["segments"]) == 2})

    monkeypatch.setattr(runner, "_load_file_module", load)
    executor = runner.StrictMotionCaseExecutor(
        backend,
        collector,
        evidence,
        clock=lambda: 1.0,
    )

    trace, result = executor.run_detour(
        case_id="detour-run-1",
        route_id="office_waste_bin_s_to_g",
    )

    assert result == {"pass": True}
    assert [item["type"] for item in trace["action"]["segments"]] == [
        "Spin",
        "DriveOnHeading",
    ]
    assert all(
        item["freshOdometry"]["sampledAtControlTick"]
        == item["startedAtControlTick"]
        for item in trace["action"]["segments"]
    )
    assert "send_spin" in backend.calls and "send_drive" in backend.calls
    assert backend.calls[:4] == ["ready", "compute_path", "normal_stop", "arm"]
    assert evidence.calls == ["begin", "finalize"]


def test_detour_executor_rejects_resolver_skip_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeDetourBackend()
    collector = FakeExecutorCollector([_executor_sample(0.0)])
    evidence = FakeDetourEvidence()
    planner = FakeStaticPlanner(skip=True)
    monkeypatch.setattr(runner, "_load_file_module", lambda *_args: planner)
    executor = runner.StrictMotionCaseExecutor(
        backend,
        collector,
        evidence,
        clock=lambda: 1.0,
    )

    with pytest.raises(runner.StrictMotionRunnerError, match="resolved to a skip"):
        executor.run_detour(
            case_id="detour-skip",
            route_id="office_waste_bin_s_to_g",
        )

    assert evidence.calls == ["begin", "abort"]
    assert backend.calls[-3:] == ["cancel", "zero", "normal_stop"]
