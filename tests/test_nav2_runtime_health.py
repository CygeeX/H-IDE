from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from uuid import UUID


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "docker" / "nav2" / "nav2_runtime_health.py"
SPEC = importlib.util.spec_from_file_location("nav2_runtime_health", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
health = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = health
SPEC.loader.exec_module(health)


def active_states() -> dict[str, str]:
    return {name: "active" for name in health.MANAGED_NODES}


def evidence() -> object:
    return health.Nav2RuntimeEvidence(
        boot_id=UUID("61000000-0000-4000-8000-000000000006"),
        session_id=UUID("71000000-0000-4000-8000-000000000007"),
    )


def test_nav2_heartbeat_requires_lifecycle_and_local_costmap_progress() -> None:
    runtime = evidence()
    empty = runtime.publish_snapshot(now_s=10.0)
    assert empty["healthy"] is False
    assert empty["sequence"] == 1

    runtime.observe_lifecycle(active_states(), observed_at_s=10.0)
    no_costmap = runtime.publish_snapshot(now_s=10.01)
    assert no_costmap["healthy"] is False

    runtime.observe_local_costmap(observed_at_s=10.02)
    ready = runtime.publish_snapshot(now_s=10.03)
    assert ready["healthy"] is True
    assert ready["sequence"] == 3
    assert ready["lifecycleValidationSequence"] == 1
    assert ready["localCostmapSequence"] == 1
    assert ready["bootId"] == "61000000-0000-4000-8000-000000000006"
    assert ready["sessionId"] == "71000000-0000-4000-8000-000000000007"


def test_nav2_heartbeat_fails_closed_for_inactive_stale_or_failed_scan() -> None:
    runtime = evidence()
    states = active_states()
    states["behavior_server"] = "inactive"
    runtime.observe_lifecycle(states, observed_at_s=10.0)
    runtime.observe_local_costmap(observed_at_s=10.0)
    assert runtime.publish_snapshot(now_s=10.01)["healthy"] is False

    runtime.observe_lifecycle(active_states(), observed_at_s=10.1)
    assert runtime.publish_snapshot(now_s=11.100001)["healthy"] is False

    runtime.observe_lifecycle(active_states(), observed_at_s=12.0)
    runtime.observe_local_costmap(observed_at_s=12.0)
    runtime.observe_lifecycle_failure("planner state timeout")
    failed = runtime.publish_snapshot(now_s=12.01)
    assert failed["healthy"] is False
    assert failed["fault"] == "planner state timeout"


def test_nav2_health_process_has_no_motion_publisher_or_action_client() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "/h_ide/nav2/runtime_health" in string_literals
    assert "/local_costmap/costmap_raw" in string_literals
    assert "/cmd_vel" not in string_literals
    assert "ActionClient" not in source


def test_nav2_image_launches_and_build_checks_the_health_process() -> None:
    dockerfile = (ROOT / "docker" / "nav2" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    launch = (ROOT / "docker" / "nav2" / "nav2_sim_launch.py").read_text(
        encoding="utf-8"
    )

    assert "COPY docker/nav2/nav2_runtime_health.py" in dockerfile
    assert "/opt/h-ide-nav2/nav2_runtime_health.py" in dockerfile
    assert 'ExecuteProcess(' in launch
    assert '"/opt/h-ide-nav2/nav2_runtime_health.py"' in launch
