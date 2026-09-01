from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads(
    (ROOT / "manifests/release/l3-mission-result.schema.json").read_text(
        encoding="utf-8"
    )
)


def result(*, accepted: bool) -> dict[str, object]:
    value: dict[str, object] = {
        "schemaVersion": "skillstore.l3-mission-result/1",
        "releaseId": "skillstore-office-v2-e2e-20260901-r1",
        "repositoryCommit": "a" * 40,
        "recordedAt": "2026-09-01T00:00:00+00:00",
        "missionId": "mission-test",
        "operatorAuthorized": True,
        "missionStatus": "completed" if accepted else "failed",
        "navigationSucceeded": accepted,
        "inspectionSucceeded": accepted,
        "officeSequenceValidated": accepted,
        "allRequiredNodesSucceeded": accepted,
        "allRequiredHeadingsObserved": accepted,
        "reportFullChainAccepted": accepted,
        "regionIds": ["open_office", "meeting_room"] if accepted else [],
        "stationIds": ["open_office_west", "meeting_south"] if accepted else [],
        "headingGapCount": 0 if accepted else 2,
        "reportEvidenceCount": 24 if accepted else 0,
        "scanEvidenceCount": 24 if accepted else 0,
        "resultAccepted": accepted,
        "eventCount": 1,
        "worldRecordCount": 1 if accepted else 0,
        "skillExecutionCount": 1,
        "rawMissionSha256": "b" * 64,
        "rawMissionPath": "var/acceptance/mission-test.json",
    }
    return value


def errors(value: dict[str, object]) -> list[object]:
    return list(Draft202012Validator(SCHEMA).iter_errors(value))


def test_schema_accepts_both_a_structured_failure_and_the_exact_success_contract() -> None:
    assert errors(result(accepted=False)) == []
    assert errors(result(accepted=True)) == []


def test_success_contract_cannot_claim_acceptance_with_gaps_or_wrong_room_order() -> None:
    with_gap = deepcopy(result(accepted=True))
    with_gap["headingGapCount"] = 1
    assert errors(with_gap)

    wrong_order = deepcopy(result(accepted=True))
    wrong_order["regionIds"] = ["meeting_room", "open_office"]
    assert errors(wrong_order)


def test_acceptance_recorder_validates_the_written_result_schema() -> None:
    source = (ROOT / "integration/acceptance.sh").read_text(encoding="utf-8")
    assert "Draft202012Validator" in source
    assert "schema_validation_status" in source
    assert 'isinstance(report_heading_gaps, list)' in source
    assert 'isinstance(report_missing_stations, list)' in source
