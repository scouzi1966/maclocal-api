"""Task content hashing + drift detection.

Claims ledger (promise -> proving test):
- "task_hash is deterministic; same task -> same hash"      -> test_deterministic
- "distinct tasks have distinct hashes"                      -> test_distinct_tasks_differ
- "changing a scoring file (repo/tests/tests-spec/gold)
   changes the hash"                                         -> test_* _changes_hash
- "cosmetic metadata edits do NOT change the hash"           -> test_cosmetic_metadata_stable
- "the leaderboard flags runs scored against a task version
   that no longer matches the current definition"           -> test_mark_stale_*
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.leaderboard import mark_stale
from harness.tasks import load_task, task_hash


def _make_task(root: Path, task_id: str = "t") -> Path:
    d = root / task_id
    (d / "repo").mkdir(parents=True)
    (d / "tests").mkdir(parents=True)
    (d / "repo" / "m.py").write_text("def f():\n    return 1\n")
    (d / "tests" / "t_x.py").write_text("from m import f\n\ndef test():\n    assert f() == 2\n")
    (d / "gold_patch.diff").write_text("diff --git a/m.py b/m.py\n")
    (d / "issue.md").write_text("fix f")
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "category": "bug_fix",
                "languages": ["python"],
                "difficulty": "easy",
                "created": "2026-05-30",
                "source": "hand-authored",
                "decontamination_notes": "original",
                "tests": {
                    "fail_to_pass": [{"name": "x", "cmd": "pytest t_x.py"}],
                    "pass_to_pass": [],
                },
            }
        )
    )
    return d


def _h(root: Path, task_id: str = "t") -> str:
    return task_hash(load_task(task_id, root))


def test_deterministic(tmp_path: Path) -> None:
    _make_task(tmp_path)
    assert _h(tmp_path) == _h(tmp_path)
    assert len(_h(tmp_path)) == 64


def test_distinct_tasks_differ(tmp_path: Path) -> None:
    _make_task(tmp_path, "a")
    b = _make_task(tmp_path, "b")
    (b / "repo" / "m.py").write_text("def f():\n    return 99\n")
    assert _h(tmp_path, "a") != _h(tmp_path, "b")


def test_repo_change_changes_hash(tmp_path: Path) -> None:
    d = _make_task(tmp_path)
    before = _h(tmp_path)
    (d / "repo" / "m.py").write_text("def f():\n    return 2\n")
    assert _h(tmp_path) != before


def test_hidden_test_change_changes_hash(tmp_path: Path) -> None:
    d = _make_task(tmp_path)
    before = _h(tmp_path)
    (d / "tests" / "t_x.py").write_text("from m import f\n\ndef test():\n    assert f() == 3\n")
    assert _h(tmp_path) != before


def test_tests_spec_change_changes_hash(tmp_path: Path) -> None:
    d = _make_task(tmp_path)
    before = _h(tmp_path)
    meta = json.loads((d / "metadata.json").read_text())
    meta["tests"]["fail_to_pass"][0]["cmd"] = "pytest t_x.py -v"
    (d / "metadata.json").write_text(json.dumps(meta))
    assert _h(tmp_path) != before


def test_gold_patch_change_changes_hash(tmp_path: Path) -> None:
    d = _make_task(tmp_path)
    before = _h(tmp_path)
    (d / "gold_patch.diff").write_text("diff --git a/m.py b/m.py\n+changed\n")
    assert _h(tmp_path) != before


def test_issue_change_changes_hash(tmp_path: Path) -> None:
    """The prompt (issue.md) is part of the task version."""
    d = _make_task(tmp_path)
    before = _h(tmp_path)
    (d / "issue.md").write_text("a different prompt entirely")
    assert _h(tmp_path) != before


def test_verifier_change_changes_hash(tmp_path: Path) -> None:
    """Legacy verifier.py is scoring logic and must affect the hash."""
    d = _make_task(tmp_path)
    before = _h(tmp_path)
    (d / "verifier.py").write_text("import json; print(json.dumps({'functional': 1.0}))")
    after_add = _h(tmp_path)
    assert after_add != before
    (d / "verifier.py").write_text("import json; print(json.dumps({'functional': 0.0}))")
    assert _h(tmp_path) != after_add


def test_cosmetic_metadata_stable(tmp_path: Path) -> None:
    """Editing non-scoring metadata must NOT register as drift."""
    d = _make_task(tmp_path)
    before = _h(tmp_path)
    meta = json.loads((d / "metadata.json").read_text())
    meta["decontamination_notes"] = "rewritten note, totally different wording"
    meta["difficulty"] = "hard"
    meta["created"] = "2027-01-01"
    (d / "metadata.json").write_text(json.dumps(meta))
    assert _h(tmp_path) == before


def test_mark_stale_matching_and_mismatched(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    current = _h(tmp_path, "t")
    rows = [
        {"run_id": "r_ok", "task_id": "t", "task_hash": current},
        {"run_id": "r_old", "task_id": "t", "task_hash": "0" * 64},
        {"run_id": "r_pre", "task_id": "t", "task_hash": None},  # predates hashing
        {"run_id": "r_gone", "task_id": "missing", "task_hash": current},
    ]
    mark_stale(rows, tasks_root=tmp_path)
    flags = {r["run_id"]: r["task_stale"] for r in rows}
    assert flags["r_ok"] is False
    assert flags["r_old"] is True
    assert flags["r_pre"] is None
    assert flags["r_gone"] is None  # task no longer exists -> unknown, not flagged
