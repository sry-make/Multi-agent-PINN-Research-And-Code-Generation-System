from __future__ import annotations

import pytest

from memory.project_store import (
    default_project_memory,
    format_project_memory,
    load_project_memory,
    record_project_decision,
    record_rejected_option,
    save_project_memory,
)


pytestmark = pytest.mark.unit


def test_default_project_memory_has_version_and_decisions() -> None:
    project_memory = default_project_memory()

    assert project_memory["memory_version"] >= 2
    assert project_memory["updated_at"]
    assert project_memory["decisions"]
    assert project_memory["rejected_options"]


def test_record_project_decision_and_rejected_option_can_roundtrip(tmp_path) -> None:
    project_path = tmp_path / "project_memory.json"
    project_memory = load_project_memory(path=project_path)

    project_memory = record_project_decision(
        project_memory,
        "Version project memory with decisions and rejected options.",
        rationale="This keeps long-term project facts auditable and easier to explain in interviews.",
        decision_id="project-memory-versioning",
    )
    project_memory = record_rejected_option(
        project_memory,
        "Auto-write arbitrary project facts from the LLM.",
        reason="Project memory should stay curated to avoid hallucinated long-term facts.",
    )
    save_project_memory(project_memory, path=project_path)
    reloaded = load_project_memory(path=project_path)

    assert reloaded["memory_version"] >= 2
    assert any(item["id"] == "project-memory-versioning" for item in reloaded["decisions"])
    assert any(
        "Auto-write arbitrary project facts from the LLM." in item["option"]
        for item in reloaded["rejected_options"]
    )

    formatted = format_project_memory(reloaded)
    assert "Project memory version" in formatted
    assert "Key decisions" in formatted
    assert "Rejected options" in formatted
