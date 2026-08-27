from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from hide_mujoco_sonic.arm_limiter_evidence import (
    ARM_JOINT_INDICES,
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_MAX_RECORDS,
    ArmLimiterEvidenceError,
    ArmLimiterEvidenceRecorder,
    canonical_sha256,
    load_arm_limiter_evidence_artifact,
    source_bundle_sha256,
)


SOURCE_HASH = "a" * 64


def _recorder(
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
    artifact_directory: Path | None = None,
) -> ArmLimiterEvidenceRecorder:
    return ArmLimiterEvidenceRecorder(
        sonic_driver_boot_id="sonic-boot-1",
        sonic_driver_session_id="sonic-session-1",
        sonic_driver_source_sha256=SOURCE_HASH,
        max_records=max_records,
        artifact_directory=artifact_directory,
    )


def _durations(*, joint: int | None = None, duration: float = 0.0):
    return {
        joint_index: duration if joint_index == joint else 0.0
        for joint_index in ARM_JOINT_INDICES
    }


def _begin(recorder: ArmLimiterEvidenceRecorder, *, case_id: str = "case-1") -> None:
    recorder.begin(
        case_id=case_id,
        policy_activation_epoch=7,
        support_release_confirmed_epoch=5,
        plant_boot_id="plant-boot-1",
        plant_session_id="plant-session-1",
    )


def test_active_identity_is_an_immutable_projection_of_the_session_binding() -> None:
    recorder = _recorder()
    _begin(recorder)
    identity = recorder.active_identity
    assert identity == {
        "caseId": "case-1",
        "policyActivationEpoch": 7,
        "supportReleaseConfirmedEpoch": 5,
        "plantBootId": "plant-boot-1",
        "plantSessionId": "plant-session-1",
    }
    assert identity is not None
    identity["caseId"] = "poison"
    assert recorder.active_identity["caseId"] == "case-1"  # type: ignore[index]


def _record(
    recorder: ArmLimiterEvidenceRecorder,
    tick: int,
    *,
    joint: int | None = None,
    duration: float = 0.0,
) -> None:
    recorder.record(
        control_tick=tick,
        t_s=(tick - 100) * 0.02,
        dt_s=0.02,
        constrained_joint_indices=[] if joint is None else [joint],
        continuous_saturation_by_joint_s=_durations(
            joint=joint, duration=duration
        ),
    )


def test_success_trace_is_exact_bound_hashed_and_idempotently_readable() -> None:
    recorder = _recorder()
    _begin(recorder)
    _record(recorder, 100, joint=15, duration=0.02)
    _record(recorder, 101, joint=15, duration=0.04)
    _record(recorder, 102)

    result = recorder.finalize(case_id="case-1")
    expected_fields = {
        "schemaVersion",
        "caseId",
        "source",
        "complete",
        "interpolated",
        "repeatedTelemetry",
        "policyActivationEpoch",
        "supportReleaseConfirmedEpoch",
        "plantBootId",
        "plantSessionId",
        "sonicDriverBootId",
        "sonicDriverSessionId",
        "sonicDriverSourceSha256",
        "taskStartControlTick",
        "taskEndControlTick",
        "records",
        "traceSha256",
    }
    assert set(result) == expected_fields
    asserted_hash = result.pop("traceSha256")
    assert asserted_hash == canonical_sha256(result)
    result["traceSha256"] = asserted_hash
    assert result["taskStartControlTick"] == 100
    assert result["taskEndControlTick"] == 102
    assert result["records"][0]["constrainedJointIndices"] == [15]
    assert result["records"][1]["continuousSaturationByJointS"]["15"] == 0.04
    assert result["records"][2]["constrainedJoints"] == 0

    replay = recorder.finalize(case_id="case-1")
    assert replay == result
    replay["records"].clear()
    assert recorder.completed(case_id="case-1") == result


def test_large_artifact_is_private_atomic_and_transport_descriptor_is_bounded(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "arm-limiter-evidence"
    recorder = _recorder(artifact_directory=directory)
    _begin(recorder)
    for tick in range(100, 600):
        _record(recorder, tick)

    expected = recorder.finalize(case_id="case-1")
    descriptor = recorder._completed_transport_descriptor(case_id="case-1")

    assert descriptor is not None
    assert descriptor["schemaVersion"] == ARTIFACT_SCHEMA_VERSION
    assert descriptor["caseId"] == "case-1"
    assert descriptor["traceSha256"] == expected["traceSha256"]
    assert len(json.dumps({"status": "complete", "artifact": descriptor})) < 4096
    artifact = Path(descriptor["path"])
    assert artifact.parent == directory
    assert artifact.stat().st_mode & 0o777 == 0o600
    assert directory.stat().st_mode & 0o777 == 0o700
    assert not list(directory.glob("*.tmp-*"))
    assert load_arm_limiter_evidence_artifact(
        descriptor,
        case_id="case-1",
        directory=directory,
    ) == expected


def test_artifact_loader_rejects_tampering_unsafe_mode_and_path_escape(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "arm-limiter-evidence"
    recorder = _recorder(artifact_directory=directory)
    _begin(recorder)
    _record(recorder, 100)
    _record(recorder, 101)
    recorder.finalize(case_id="case-1")
    descriptor = recorder._completed_transport_descriptor(case_id="case-1")
    assert descriptor is not None
    artifact = Path(descriptor["path"])

    escaped = dict(descriptor)
    escaped["path"] = str(tmp_path / artifact.name)
    with pytest.raises(ArmLimiterEvidenceError, match="escapes"):
        load_arm_limiter_evidence_artifact(
            escaped,
            case_id="case-1",
            directory=directory,
        )

    original = artifact.read_bytes()
    symlink_target = tmp_path / "untrusted-copy.json"
    symlink_target.write_bytes(original)
    artifact.unlink()
    artifact.symlink_to(symlink_target)
    with pytest.raises(ArmLimiterEvidenceError, match="cannot be opened"):
        load_arm_limiter_evidence_artifact(
            descriptor,
            case_id="case-1",
            directory=directory,
        )
    artifact.unlink()
    artifact.write_bytes(original)
    os.chmod(artifact, 0o600)

    os.chmod(artifact, 0o644)
    with pytest.raises(ArmLimiterEvidenceError, match="unsafe"):
        load_arm_limiter_evidence_artifact(
            descriptor,
            case_id="case-1",
            directory=directory,
        )
    os.chmod(artifact, 0o600)

    raw = bytearray(artifact.read_bytes())
    raw[-2] = ord("0") if raw[-2] != ord("0") else ord("1")
    artifact.write_bytes(raw)
    os.chmod(artifact, 0o600)
    with pytest.raises(ArmLimiterEvidenceError, match="hash"):
        load_arm_limiter_evidence_artifact(
            descriptor,
            case_id="case-1",
            directory=directory,
        )


def test_seal_detaches_the_live_boundary_before_json_materialization() -> None:
    recorder = _recorder()
    _begin(recorder)
    _record(recorder, 100)
    _record(recorder, 101)

    recorder.seal(case_id="case-1")

    assert recorder.active is False
    assert recorder.sealed_case_id == "case-1"
    assert recorder.completed_case_id is None
    # A later control tick cannot extend the already sealed actuator fact.
    _record(recorder, 102)
    completed = recorder.finalize(case_id="case-1")
    assert completed["taskEndControlTick"] == 101
    assert recorder.sealed_case_id is None
    assert recorder.completed_case_id == "case-1"


def test_fault_while_sealed_invalidates_materialization_and_allows_recovery() -> None:
    recorder = _recorder()
    _begin(recorder)
    _record(recorder, 100)
    _record(recorder, 101)
    recorder.seal(case_id="case-1")

    assert recorder.close_failed_active("balance lost after seal") == "case-1"
    assert recorder.sealed_case_id is None
    with pytest.raises(ArmLimiterEvidenceError, match="no matching"):
        recorder.finalize(case_id="case-1")

    _begin(recorder, case_id="case-2")
    _record(recorder, 200)
    _record(recorder, 201)
    assert recorder.finalize(case_id="case-2")["caseId"] == "case-2"


@pytest.mark.parametrize(
    ("tick", "t_s", "indices", "durations", "match"),
    [
        (102, 0.02, [], _durations(), "control ticks"),
        (101, 0.03, [], _durations(), "timestamps"),
        (101, 0.02, [15], _durations(joint=15, duration=0.04), "durations"),
        (101, 0.02, [15, 15], _durations(joint=15, duration=0.02), "unique"),
    ],
)
def test_gap_duplicate_or_fabricated_duration_permanently_fails_active_session(
    tick, t_s, indices, durations, match
) -> None:
    recorder = _recorder()
    _begin(recorder)
    _record(recorder, 100)

    with pytest.raises(ArmLimiterEvidenceError, match=match):
        recorder.record(
            control_tick=tick,
            t_s=t_s,
            dt_s=0.02,
            constrained_joint_indices=indices,
            continuous_saturation_by_joint_s=durations,
        )
    with pytest.raises(ArmLimiterEvidenceError, match=match):
        _record(recorder, 101)
    with pytest.raises(ArmLimiterEvidenceError, match=match):
        recorder.finalize(case_id="case-1")


def test_record_capacity_is_bounded_and_overflow_is_not_truncated() -> None:
    recorder = _recorder(max_records=2)
    _begin(recorder)
    _record(recorder, 100)
    _record(recorder, 101)
    with pytest.raises(ArmLimiterEvidenceError, match="bounded record capacity"):
        _record(recorder, 102)


def test_default_capacity_covers_more_than_the_previous_400_second_limit() -> None:
    recorder = _recorder()
    _begin(recorder)

    for tick in range(100, 20_101):
        _record(recorder, tick)

    assert recorder.max_records == 120_000
    assert recorder.active is True
    assert recorder.failed_reason is None


def test_finalize_rejects_constrained_tail_and_case_collision() -> None:
    recorder = _recorder()
    _begin(recorder)
    _record(recorder, 100, joint=28, duration=0.02)
    _record(recorder, 101, joint=28, duration=0.04)
    with pytest.raises(ArmLimiterEvidenceError, match="case identity"):
        recorder.finalize(case_id="different")
    with pytest.raises(ArmLimiterEvidenceError, match="remains constrained"):
        recorder.finalize(case_id="case-1")
    assert recorder.abort(case_id="case-1") is True
    assert recorder.active is False


def test_one_way_support_release_epoch_may_precede_recovery_activation() -> None:
    recorder = _recorder()
    _begin(recorder)
    assert recorder.active_case_id == "case-1"
    assert recorder.abort(case_id="case-1") is True

    with pytest.raises(ValueError, match="cannot follow"):
        recorder.begin(
            case_id="bad",
            policy_activation_epoch=7,
            support_release_confirmed_epoch=8,
            plant_boot_id="plant-boot-1",
            plant_session_id="plant-session-1",
        )


def test_source_bundle_hash_binds_names_lengths_and_content(tmp_path: Path) -> None:
    left = tmp_path / "left.py"
    right = tmp_path / "right.py"
    left.write_text("ab", encoding="utf-8")
    right.write_text("c", encoding="utf-8")
    first = source_bundle_sha256((left, right))
    assert len(first) == 64
    right.write_text("d", encoding="utf-8")
    assert source_bundle_sha256((left, right)) != first

    with pytest.raises(ValueError, match="unique"):
        source_bundle_sha256((left, left))


def test_duration_keys_reject_float_and_noncanonical_string_aliases() -> None:
    recorder = _recorder()
    _begin(recorder)
    float_keyed = _durations()
    float_keyed[15.0] = float_keyed.pop(15)
    with pytest.raises(ArmLimiterEvidenceError, match="keys are invalid"):
        recorder.record(
            control_tick=100,
            t_s=0.0,
            dt_s=0.02,
            constrained_joint_indices=[],
            continuous_saturation_by_joint_s=float_keyed,
        )


def test_completed_payload_is_not_mutable_through_returned_objects() -> None:
    recorder = _recorder()
    _begin(recorder)
    _record(recorder, 100)
    _record(recorder, 101)
    first = recorder.finalize(case_id="case-1")
    mutated = copy.deepcopy(first)
    mutated["caseId"] = "poison"
    mutated["records"][0]["controlTick"] = 999
    assert recorder.completed(case_id="case-1") == first


def test_explicit_authority_loss_permanently_fails_only_an_active_session() -> None:
    recorder = _recorder()
    recorder.fail_active("ignored while disabled")
    _begin(recorder)

    with pytest.raises(ArmLimiterEvidenceError, match="DDS context was missing"):
        recorder.fail_active("DDS context was missing")
    with pytest.raises(ArmLimiterEvidenceError, match="DDS context was missing"):
        recorder.finalize(case_id="case-1")
