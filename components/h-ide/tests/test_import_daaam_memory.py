from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.import_daaam_memory import DaaamExportError, build_import, main


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    data = tmp_path / "daaam-output"
    data.mkdir()
    background = {
        "total_background_objects": 2,
        "total_in_hydra": 1,
        "total_filtered": 1,
        "objects": [
            {
                "semantic_id": 7,
                "label": "display",
                "position_world": [0.5, 0.5, 1.1],
                "position_camera": [0.1, 0.1, 2.0],
                "centroid_pixel": [100, 120],
                "median_depth": 2.0,
                "observations": 3,
                "in_hydra": True,
                "first_observed": 1_750_000_000.0,
                "last_observed": 1_750_000_000.2,
            },
            {
                "semantic_id": 9,
                "label": "unmapped object",
                "position_world": [20.0, 20.0, 0.4],
                "position_camera": [0.2, 0.2, 3.0],
                "centroid_pixel": [200, 220],
                "median_depth": 3.0,
                "observations": 2,
                "in_hydra": False,
                "first_observed": 1_750_000_001.0,
                "last_observed": 1_750_000_001.1,
            },
        ],
    }
    corrections = {
        "label_names": [
            {
                "label": 7,
                "name": "display",
                "temporal_history": {
                    "frame_ids": [10, 11, 12],
                    "timestamps": [
                        1_750_000_000.0,
                        1_750_000_000.1,
                        1_750_000_000.2,
                    ],
                    "observation_count": 3,
                    "first_observed": 1_750_000_000.0,
                    "last_observed": 1_750_000_000.2,
                },
            },
            {
                "label": 9,
                "name": "unmapped object",
                "temporal_history": {
                    "frame_ids": [20, 21],
                    "timestamps": [1_750_000_001.0, 1_750_000_001.1],
                    "observation_count": 2,
                    "first_observed": 1_750_000_001.0,
                    "last_observed": 1_750_000_001.1,
                },
            },
        ]
    }
    (data / "background_objects.yaml").write_text(
        yaml.safe_dump(background), encoding="utf-8"
    )
    (data / "corrections.yaml").write_text(
        yaml.safe_dump(corrections), encoding="utf-8"
    )
    config = data / "pipeline_config.yaml"
    config.write_text("grounding:\n  agent_model_name: gpt-5-mini\n", encoding="utf-8")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "regionId": "meeting_room",
                        "polygonXY": [
                            {"x": 0, "y": 0},
                            {"x": 1, "y": 0},
                            {"x": 1, "y": 1},
                            {"x": 0, "y": 1},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return data, config, catalog


def _build(tmp_path: Path):
    data, config, catalog = _inputs(tmp_path)
    return build_import(
        data_dir=data,
        config_path=config,
        region_catalog_path=catalog,
        mission_id="mission-1",
        robot_id="robot-1",
        robot_boot_id="boot-1",
        localization_epoch_id="epoch-1",
        observation_session_id="observation-1",
        scene_id="office-v2",
        scene_revision="published-1",
    )


def test_exporter_hashes_artifacts_maps_regions_and_preserves_unmapped_provenance(tmp_path) -> None:
    result = _build(tmp_path)
    assert result.frameRange.firstFrameId == 10
    assert result.frameRange.lastFrameId == 21
    assert result.frameRange.frameCount == 5
    assert [item.kind for item in result.source.artifacts] == [
        "background_objects",
        "corrections",
    ]
    assert all(item.sha256.startswith("sha256:") for item in result.source.artifacts)
    assert result.entities[0].regionId == "meeting_room"
    assert result.entities[1].regionId is None
    assert result.entities[0].confidence is None
    assert result.payloadSha256.startswith("sha256:")


def test_exporter_fails_closed_on_temporal_or_cross_artifact_disagreement(tmp_path) -> None:
    data, config, catalog = _inputs(tmp_path)
    background_path = data / "background_objects.yaml"
    background = yaml.safe_load(background_path.read_text(encoding="utf-8"))
    background["objects"][0]["observations"] = 2
    background_path.write_text(yaml.safe_dump(background), encoding="utf-8")
    with pytest.raises(DaaamExportError, match="observation counts disagree"):
        build_import(
            data_dir=data,
            config_path=config,
            region_catalog_path=catalog,
            mission_id="mission-1",
            robot_id="robot-1",
            robot_boot_id="boot-1",
            localization_epoch_id="epoch-1",
            observation_session_id="observation-1",
            scene_id="office-v2",
            scene_revision="1",
        )


def test_exporter_rejects_stringified_boolean_provenance(tmp_path) -> None:
    data, config, catalog = _inputs(tmp_path)
    background_path = data / "background_objects.yaml"
    background = yaml.safe_load(background_path.read_text(encoding="utf-8"))
    background["objects"][0]["in_hydra"] = "false"
    background_path.write_text(yaml.safe_dump(background), encoding="utf-8")
    with pytest.raises(DaaamExportError, match="must be a boolean"):
        build_import(
            data_dir=data,
            config_path=config,
            region_catalog_path=catalog,
            mission_id="mission-1",
            robot_id="robot-1",
            robot_boot_id="boot-1",
            localization_epoch_id="epoch-1",
            observation_session_id="observation-1",
            scene_id="office-v2",
            scene_revision="1",
        )


def test_cli_writes_a_valid_offline_import_without_posting(tmp_path, capsys) -> None:
    data, _config, catalog = _inputs(tmp_path)
    output = tmp_path / "import.json"
    exit_code = main(
        [
            "--data-dir",
            str(data),
            "--region-catalog",
            str(catalog),
            "--mission-id",
            "mission-1",
            "--robot-id",
            "robot-1",
            "--robot-boot-id",
            "boot-1",
            "--localization-epoch-id",
            "epoch-1",
            "--observation-session-id",
            "observation-1",
            "--scene-id",
            "office-v2",
            "--scene-revision",
            "published-1",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == str(output)
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["schemaVersion"] == "h-ide.daaam-semantic-memory/1"
    assert parsed["source"]["license"] == "BSD-3-Clause"
