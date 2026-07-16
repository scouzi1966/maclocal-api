"""Tests for per-task Docker sandbox image selection."""

from __future__ import annotations

import json
from pathlib import Path

from harness.sandbox.docker_executor import DEFAULT_IMAGE
from harness.sandbox.images import DEFAULT_RUST_IMAGE, resolve_sandbox_image
from harness.tasks import load_task


def test_resolve_rust_task_uses_rust_image(tmp_path: Path) -> None:
    task_dir = tmp_path / "rs-demo"
    (task_dir / "repo").mkdir(parents=True)
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": "rs-demo",
                "category": "bug_fix",
                "languages": ["rust"],
                "difficulty": "medium",
                "source": "hand-authored",
                "decontaminated": True,
                "decontamination_notes": "fixture",
                "created": "2026-06-01",
                "tests": {"fail_to_pass": [{"name": "t", "cmd": "true"}], "pass_to_pass": []},
            }
        )
    )
    (task_dir / "issue.md").write_text("fix")
    (task_dir / "gold_patch.diff").write_text("diff --git a/x b/x\n")
    task = load_task("rs-demo", tmp_path)
    assert resolve_sandbox_image(task) == DEFAULT_RUST_IMAGE


def test_cli_image_overrides_language_tier() -> None:
    task = load_task("rs-borrow-split", Path("tasks/v1"))
    assert resolve_sandbox_image(task, "custom:img") == "custom:img"


def test_python_task_uses_base_image() -> None:
    task = load_task("py-topo-sort-cycle", Path("tasks/v1"))
    assert resolve_sandbox_image(task) == DEFAULT_IMAGE
