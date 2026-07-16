"""Tests for task loading and workspace preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.tasks import list_task_ids, load_task, prepare_workspace


def _make_task(root: Path, task_id: str) -> Path:
    d = root / task_id
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": task_id, "category": "bug"}))
    (d / "issue.md").write_text("Fix the bug.")
    (d / "verifier.py").write_text('print("{}")')
    return d


def test_load_task(tmp_path: Path) -> None:
    _make_task(tmp_path, "swe-001")
    task = load_task("swe-001", tmp_path)
    assert task.task_id == "swe-001"
    assert task.metadata["category"] == "bug"
    assert task.issue == "Fix the bug."
    assert task.verifier is not None
    assert task.snapshot is None


def test_load_task_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_task("nope", tmp_path)


def test_list_task_ids(tmp_path: Path) -> None:
    _make_task(tmp_path, "b")
    _make_task(tmp_path, "a")
    assert list_task_ids(tmp_path) == ["a", "b"]


def test_list_task_ids_empty(tmp_path: Path) -> None:
    assert list_task_ids(tmp_path / "missing") == []


def test_prepare_workspace_writes_issue(tmp_path: Path) -> None:
    _make_task(tmp_path, "swe-001")
    task = load_task("swe-001", tmp_path)
    ws = prepare_workspace(task, tmp_path / "ws")
    assert (ws / "issue.md").read_text() == "Fix the bug."


def test_load_real_hello_world() -> None:
    """The shipped demo task loads and has the expected shape."""
    task = load_task("hello-world", Path("tasks/v1"))
    assert task.verifier is not None
    assert "hello from vulcanbench" in task.issue
