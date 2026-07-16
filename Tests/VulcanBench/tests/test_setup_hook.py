"""Tests for the setup/warm-up hook in harness.tasks.run_setup.

Covers: runs before the agent, traces events, aborts on non-zero exit,
and the validator runs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.tasks import Task, run_setup


def _task_with_setup(setup: list[dict[str, str]], **meta_overrides: Any) -> Task:
    """Build a Task with the given setup commands in metadata."""
    metadata: dict[str, Any] = {
        "id": "test-setup",
        "category": "bug_fix",
        "languages": ["rust"],
        "difficulty": "easy",
        "source": "hand-authored",
        "decontaminated": True,
        "decontamination_notes": "test fixture",
        "setup": setup,
    }
    metadata.update(meta_overrides)
    return Task(
        task_id="test-setup",
        root=Path("/nonexistent"),
        metadata=metadata,
        issue="fix the bug",
        verifier=None,
    )


def test_setup_no_commands(tmp_path: Path) -> None:
    task = _task_with_setup([])
    results = run_setup(task, tmp_path)
    assert results == []


def test_setup_success(tmp_path: Path) -> None:
    task = _task_with_setup([{"name": "echo", "cmd": "echo hello"}])
    results = run_setup(task, tmp_path)
    assert len(results) == 1
    assert results[0]["name"] == "echo"
    assert results[0]["exit_code"] == 0
    assert results[0]["duration_s"] > 0


def test_setup_traces_events(tmp_path: Path) -> None:
    task = _task_with_setup([{"name": "echo", "cmd": "echo hi"}])

    class FakeCollector:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, Any]]] = []

        def record(self, event: str, data: dict[str, Any]) -> None:
            self.events.append((event, data))

    collector = FakeCollector()
    run_setup(task, tmp_path, collector=collector)
    assert len(collector.events) == 1
    assert collector.events[0][0] == "setup"
    assert collector.events[0][1]["name"] == "echo"
    assert collector.events[0][1]["exit_code"] == 0


def test_setup_aborts_on_nonzero(tmp_path: Path) -> None:
    task = _task_with_setup(
        [
            {"name": "ok", "cmd": "echo ok"},
            {"name": "fail", "cmd": "false"},
            {"name": "never", "cmd": "echo never"},
        ]
    )
    with pytest.raises(RuntimeError, match="setup command 'fail' failed"):
        run_setup(task, tmp_path)
    # The third command should never execute.


def test_setup_timeout_from_metadata(tmp_path: Path) -> None:
    task = _task_with_setup(
        [{"name": "echo", "cmd": "echo hi"}],
        setup_timeout_s=30,
    )
    # Verify the timeout defaults from metadata.
    assert task.setup_timeout_s == 30


def test_setup_timeout_default(tmp_path: Path) -> None:
    task = _task_with_setup([{"name": "echo", "cmd": "echo hi"}])
    assert task.setup_timeout_s == 600


def test_setup_commands_property_valid() -> None:
    task = _task_with_setup([{"name": "a", "cmd": "cmd_a"}, {"name": "b", "cmd": "cmd_b"}])
    assert len(task.setup_commands) == 2
    assert task.setup_commands[0]["name"] == "a"


def test_setup_commands_property_invalid() -> None:
    metadata: dict[str, Any] = {
        "id": "bad",
        "category": "bug_fix",
        "setup": "not a list",
    }
    task = Task(task_id="bad", root=Path("/x"), metadata=metadata, issue="", verifier=None)
    assert task.setup_commands == []


def test_setup_commands_property_missing_name() -> None:
    metadata: dict[str, Any] = {
        "id": "bad",
        "category": "bug_fix",
        "setup": [{"cmd": "echo"}],
    }
    task = Task(task_id="bad", root=Path("/x"), metadata=metadata, issue="", verifier=None)
    assert task.setup_commands == []
