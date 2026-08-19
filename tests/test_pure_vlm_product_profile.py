from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from packages.simulation_core import SimulationConflict, SimulationRuntime, SimulationSession
from services.control_api.store import (
    _legacy_skill_result_read_projection,
    _legacy_world_record_read_projection,
)


ROOT = Path(__file__).resolve().parents[1]


def test_product_compose_has_one_host_qwen_sam2_boundary() -> None:
    base = (ROOT / "infra/docker-compose.yml").read_text()
    simulation = (ROOT / "infra/docker-compose.simulation.yml").read_text()
    example = (ROOT / ".env.example").read_text()
    product_configuration = "\n".join((base, simulation, example))

    assert "http://host.docker.internal:8092" in base
    assert "HIDE_VISION_VLM_PROVIDER: qwen" in base
    assert "HIDE_VISION_VLM_MODEL: qwen3-vl-plus" in base
    assert "HIDE_MISSION_VISION_PROVIDER: qwen" in base
    assert "HIDE_MISSION_VISION_MODEL: qwen3-vl-plus" in base
    assert "\n  h-ide-vision:\n" not in base
    assert 'profiles: ["visual-fixture"]' in base
    assert "HIDE_VISION_SAM2_DEVICE=cuda" in example

    for retired_setting in (
        "mujoco_oracle",
        "MujocoOracle",
        "HIDE_VISION_RUNTIME_VLM_PROVIDER",
        "HIDE_VISION_SEGMENTATION_PROVIDER",
        "HIDE_VISION_MUJOCO_ORACLE_URL",
        "FoundationPose",
    ):
        assert retired_setting not in product_configuration


def test_new_simulation_sessions_are_qwen_only() -> None:
    runtime = SimulationRuntime()
    try:
        created = runtime.create({"robotId": "sim-robot-pure-vlm"})
        assert created["session"]["mode"] == "qwen"
        with pytest.raises(ValidationError):
            runtime.create({"robotId": "sim-robot-retired", "mode": "oracle"})
    finally:
        runtime.close()


def test_legacy_simulation_session_remains_readable_but_cannot_restart() -> None:
    runtime = SimulationRuntime()
    try:
        created = runtime.create({"robotId": "sim-robot-history"})
        session_id = created["session"]["sessionId"]
        stored = runtime.store.get_session(session_id)
        assert stored is not None
        session, snapshot = stored
        legacy = SimulationSession.model_validate(
            {**session.model_dump(mode="json"), "mode": "oracle"}
        )
        runtime.store.save_session(legacy, snapshot)

        restored = runtime.get(session_id)
        assert restored["legacyReadOnly"] is True
        with pytest.raises(SimulationConflict, match="read-only"):
            runtime.start(session_id)
    finally:
        runtime.close()


def test_retired_visual_records_are_projected_read_only_without_mutation() -> None:
    world = {"recordId": "legacy-world-1", "source": "mujoco_oracle"}
    projected_world = _legacy_world_record_read_projection(world)
    assert world == {"recordId": "legacy-world-1", "source": "mujoco_oracle"}
    assert projected_world["legacyReadOnly"] is True
    assert projected_world["executionEligible"] is False

    skill = {
        "result": {
            "visualResult": {
                "analysisId": "legacy-analysis-1",
                "versions": {"vlmProvider": "retired-simulator-provider"},
                "hypotheses": [],
            }
        }
    }
    projected_skill = _legacy_skill_result_read_projection(skill)
    assert "legacyReadOnly" not in skill["result"]["visualResult"]
    assert projected_skill["result"]["visualResult"]["legacyReadOnly"] is True
    assert projected_skill["result"]["visualResult"]["executionEligible"] is False
