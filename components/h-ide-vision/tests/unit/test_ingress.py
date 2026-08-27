from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from h_ide_vision.ingress import LatestFrameStore, ObservationRejected
from tests.fixtures import fixture_bundle


def test_store_is_bounded_and_rejects_out_of_order() -> None:
    store = LatestFrameStore(per_sensor_capacity=3)
    for sequence in range(1, 5):
        store.accept(fixture_bundle(sequence=sequence))
    assert [frame.bundle.sequence for frame in store.latest("fixture-robot", "head_rgbd")] == [2, 3, 4]
    with pytest.raises(ObservationRejected, match="sequence") as error:
        store.accept(fixture_bundle(sequence=4))
    assert error.value.reason == "out_of_order"


def test_new_observation_session_resets_sequence_and_drops_old_frames() -> None:
    store = LatestFrameStore(per_sensor_capacity=3)
    store.activate_session("fixture-robot", "head_rgbd", "session-1")
    store.accept(fixture_bundle(sequence=1, session_id="session-1"))
    store.accept(fixture_bundle(sequence=2, session_id="session-1"))

    store.activate_session("fixture-robot", "head_rgbd", "session-2")
    store.accept(fixture_bundle(sequence=1, session_id="session-2"))

    frames = store.latest("fixture-robot", "head_rgbd")
    assert [frame.bundle.sequence for frame in frames] == [1]
    assert frames[0].bundle.identity.observation_session_id == "session-2"
    with pytest.raises(ObservationRejected) as error:
        store.accept(fixture_bundle(sequence=3, session_id="session-1"))
    assert error.value.reason == "identity_mismatch"


def test_stale_bad_crc_and_epoch_change_fail_closed() -> None:
    store = LatestFrameStore()
    stale = fixture_bundle(sequence=1, captured_at=datetime.now(UTC) - timedelta(seconds=3))
    with pytest.raises(ObservationRejected) as error:
        store.accept(stale)
    assert error.value.reason == "stale"
    current = fixture_bundle(sequence=1)
    with pytest.raises(ObservationRejected) as error:
        store.accept(replace(current, rgb_crc32c=0))
    assert error.value.reason == "bad_checksum"
    store.accept(current)
    changed = fixture_bundle(sequence=2, localization_epoch_id="epoch-2")
    with pytest.raises(ObservationRejected) as error:
        store.accept(changed)
    assert error.value.reason == "localization_mismatch"
