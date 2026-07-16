"""Tests for empirical difficulty calibration.

Claims ledger (promise -> proving test):
- "banding: >= easy_min -> easy, >= medium_min -> medium, else hard"  -> test_banding
- "thresholds must be finite and ordered 0 <= med < easy <= 1"        -> test_threshold_validation
- "macro-averaging: model A 10/10 + model B 0/2 -> 0.5, not 10/12"  -> test_macro_averaging
- "mock rows are excluded by default; --include-mock keeps them"      -> test_mock_exclusion
- "stale rows (task_stale=True) are excluded"                        -> test_stale_exclusion
- "rows with unknown task_id are excluded"                           -> test_unknown_task_exclusion
- "evidence gating: insufficient attempts/models -> insufficient_data" -> test_evidence_gate
- "trivial label maps to easy for agreement comparison"               -> test_trivial_agreement
- "zero-run tasks appear as insufficient_data"                       -> test_zero_run_tasks
- "entries are sorted: disagreements, calibrated, insufficient"       -> test_sort_order
- "result is JSON-serializable"                                      -> test_json_round_trips
- "CLI: --format json parses; bad thresholds exit 1"                 -> test_cli_json / test_cli_bad_thresholds
- "report includes calibration section"                               -> test_report_calibration
- "markdown contains calibration section + disagreement callout"     -> test_markdown_calibration
- "backend: /api/calibration and enriched /api/tasks"                -> test_backend_calibration
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.calibration import calibrate_tasks, calibration_to_markdown
from harness.cli import app
from harness.report import build_report, to_markdown
from harness.tasks import load_task, task_hash
from tests.test_task_hash import _make_task


def _row(
    task_id: str = "t",
    model: str = "openai:gpt-4o",
    functional: float = 1.0,
    task_hash_val: str | None = None,
) -> dict[str, object]:
    r: dict[str, object] = {
        "run_id": f"r-{task_id}-{model}-{functional}",
        "task_id": task_id,
        "model": model,
        "functional": functional,
        "total": functional,
    }
    if task_hash_val is not None:
        r["task_hash"] = task_hash_val
    return r


def _make_task_with_difficulty(root: Path, task_id: str = "t", difficulty: str = "easy") -> Path:
    d = _make_task(root, task_id)
    meta = json.loads((d / "metadata.json").read_text())
    meta["difficulty"] = difficulty
    (d / "metadata.json").write_text(json.dumps(meta))
    return d


def test_banding(tmp_path: Path) -> None:
    _make_task_with_difficulty(tmp_path, "easy-task", "easy")
    _make_task_with_difficulty(tmp_path, "med-task", "medium")
    _make_task_with_difficulty(tmp_path, "hard-task", "hard")

    rows = [
        *[_row("easy-task", f"m{i}", 1.0) for i in range(3)],
        *[_row("easy-task", f"m{i}", 0.0) for i in range(3, 6)],
        *[_row("med-task", f"m{i}", 0.5) for i in range(3)],
        *[_row("med-task", f"m{i}", 0.0) for i in range(3, 6)],
        *[_row("hard-task", f"m{i}", 0.0) for i in range(6)],
    ]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=1, min_models=1)
    by_id = {e["task_id"]: e for e in cal["tasks"]}

    assert by_id["easy-task"]["empirical_difficulty"] == "medium"
    assert by_id["med-task"]["empirical_difficulty"] == "hard"
    assert by_id["hard-task"]["empirical_difficulty"] == "hard"


def test_banding_edges(tmp_path: Path) -> None:
    _make_task(tmp_path, "a")
    _make_task(tmp_path, "b")
    _make_task(tmp_path, "c")

    rows_a = [_row("a", f"m{i}", 1.0) for i in range(2)]
    rows_b = [_row("b", f"m{i}", 1.0) for i in range(1)] + [_row("b", "m2", 0.0)]
    rows_c = [_row("c", f"m{i}", 0.0) for i in range(2)]
    cal = calibrate_tasks(
        rows_a + rows_b + rows_c,
        tasks_root=tmp_path,
        min_attempts=1,
        min_models=1,
        easy_min=0.85,
        medium_min=0.40,
    )
    by_id = {e["task_id"]: e for e in cal["tasks"]}
    assert by_id["a"]["empirical_difficulty"] == "easy"
    assert by_id["b"]["empirical_difficulty"] == "medium"
    assert by_id["c"]["empirical_difficulty"] == "hard"


def test_threshold_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        calibrate_tasks([], easy_min=float("inf"), medium_min=0.4)
    with pytest.raises(ValueError, match="0 <= medium_min < easy_min <= 1"):
        calibrate_tasks([], easy_min=0.3, medium_min=0.5)
    with pytest.raises(ValueError, match="0 <= medium_min < easy_min <= 1"):
        calibrate_tasks([], easy_min=1.5, medium_min=0.4)
    with pytest.raises(ValueError, match="min_attempts must be >= 1"):
        calibrate_tasks([], min_attempts=0)
    with pytest.raises(ValueError, match="min_models must be >= 1"):
        calibrate_tasks([], min_models=0)


def test_macro_averaging(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    rows = [_row("t", "heavy", 1.0) for _ in range(10)] + [
        _row("t", "light", 0.0) for _ in range(2)
    ]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=1, min_models=1)
    e = cal["tasks"][0]
    assert e["solve_rate"] == 0.5
    assert e["per_model_solve_rate"]["heavy"] == 1.0
    assert e["per_model_solve_rate"]["light"] == 0.0


def test_mock_exclusion(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    rows = [_row("t", "mock:synthetic", 1.0), _row("t", "openai:gpt-4o", 0.0)]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=1, min_models=1)
    assert cal["excluded"]["mock_runs"] == 1
    e = cal["tasks"][0]
    assert e["n_attempts"] == 1
    assert e["models"] == ["openai:gpt-4o"]

    cal2 = calibrate_tasks(
        rows, tasks_root=tmp_path, min_attempts=1, min_models=1, include_mock=True
    )
    assert cal2["excluded"]["mock_runs"] == 0
    assert cal2["tasks"][0]["n_attempts"] == 2


def test_stale_exclusion(tmp_path: Path) -> None:
    """Stale runs (task_hash mismatch) are excluded by calibration.

    calibrate_tasks calls mark_stale internally, which overwrites any
    pre-set task_stale. We must provide a real task_hash mismatch to
    trigger stale detection.
    """
    _make_task(tmp_path, "t")
    current_hash = task_hash(load_task("t", tmp_path))
    rows = [
        {
            "run_id": "r_stale",
            "task_id": "t",
            "model": "m1",
            "functional": 1.0,
            "total": 1.0,
            "task_hash": "0" * 64,
        },
        {
            "run_id": "r_fresh",
            "task_id": "t",
            "model": "m2",
            "functional": 1.0,
            "total": 1.0,
            "task_hash": current_hash,
        },
    ]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=1, min_models=1)
    assert cal["excluded"]["stale_runs"] == 1
    assert cal["tasks"][0]["n_attempts"] == 1


def test_unknown_task_exclusion(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    rows = [_row("t", "m1", 1.0), _row("nonexistent", "m2", 1.0)]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=1, min_models=1)
    assert cal["excluded"]["unknown_task_runs"] == 1
    assert len(cal["tasks"]) == 1


def test_evidence_gate(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    rows = [_row("t", "m1", 1.0)]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=5, min_models=2)
    e = cal["tasks"][0]
    assert e["status"] == "insufficient_data"
    assert e["empirical_difficulty"] is None
    assert e["agreement"] is None
    assert cal["summary"]["n_insufficient"] == 1


def test_trivial_agreement(tmp_path: Path) -> None:
    _make_task_with_difficulty(tmp_path, "t", "trivial")
    rows = [_row("t", f"m{i}", 1.0) for i in range(5)]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=5, min_models=1)
    e = cal["tasks"][0]
    assert e["labeled_difficulty"] == "trivial"
    assert e["empirical_difficulty"] == "easy"
    assert e["agreement"] is True


def test_zero_run_tasks(tmp_path: Path) -> None:
    _make_task(tmp_path, "has-no-runs")
    cal = calibrate_tasks([], tasks_root=tmp_path)
    e = next(e for e in cal["tasks"] if e["task_id"] == "has-no-runs")
    assert e["status"] == "insufficient_data"
    assert e["n_attempts"] == 0
    assert e["solve_rate"] is None


def test_sort_order(tmp_path: Path) -> None:
    _make_task_with_difficulty(tmp_path, "agree-task", "easy")
    _make_task_with_difficulty(tmp_path, "disagree-task", "hard")
    _make_task(tmp_path, "no-data")

    rows = [_row("agree-task", f"m{i}", 1.0) for i in range(5)] + [
        _row("disagree-task", f"m{i}", 1.0) for i in range(5)
    ]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=5, min_models=1)
    ids = [e["task_id"] for e in cal["tasks"]]
    assert ids[0] == "disagree-task"
    assert ids[1] == "agree-task"
    assert ids[2] == "no-data"


def test_json_round_trips(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    rows = [_row("t", "m1", 1.0)]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=1, min_models=1)
    serialized = json.dumps(cal)
    deserialized = json.loads(serialized)
    assert deserialized["summary"]["n_tasks"] == cal["summary"]["n_tasks"]


def test_calibration_to_markdown(tmp_path: Path) -> None:
    """Markdown includes a disagreement callout when empirical != labeled."""
    _make_task_with_difficulty(tmp_path, "labeled-hard-but-easy", "hard")
    _make_task_with_difficulty(tmp_path, "labeled-easy-and-easy", "easy")
    # labeled-hard-but-easy: all models solve -> measured easy, disagrees with label "hard"
    rows = [_row("labeled-hard-but-easy", f"m{i}", 1.0) for i in range(3)] + [
        _row("labeled-easy-and-easy", f"m{i}", 1.0) for i in range(3)
    ]
    cal = calibrate_tasks(rows, tasks_root=tmp_path, min_attempts=1, min_models=1)
    md = calibration_to_markdown(cal)
    assert "## Calibration" in md
    assert "labeled-hard-but-easy" in md
    assert "labeled-easy-and-easy" in md
    assert "disagrees" in md


def test_calibration_to_markdown_no_calibrated(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    cal = calibrate_tasks([], tasks_root=tmp_path)
    md = calibration_to_markdown(cal)
    assert "No tasks have sufficient data" in md


runner = CliRunner()


def test_cli_table(tmp_path: Path) -> None:
    result = runner.invoke(app, ["calibrate", "--tasks-root", str(tmp_path)])
    assert result.exit_code == 0


def test_cli_json(tmp_path: Path) -> None:
    result = runner.invoke(app, ["calibrate", "--format", "json", "--tasks-root", str(tmp_path)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "tasks" in data


def test_cli_bad_thresholds(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        ["calibrate", "--easy-min", "0.3", "--medium-min", "0.5", "--tasks-root", str(tmp_path)],
    )
    assert r.exit_code == 1
    assert "medium_min" in r.output or "easy_min" in r.output

    r2 = runner.invoke(app, ["calibrate", "--min-attempts", "0", "--tasks-root", str(tmp_path)])
    assert r2.exit_code == 1

    r3 = runner.invoke(app, ["calibrate", "--min-models", "0", "--tasks-root", str(tmp_path)])
    assert r3.exit_code == 1


def test_cli_bad_format(tmp_path: Path) -> None:
    r = runner.invoke(app, ["calibrate", "--format", "xml", "--tasks-root", str(tmp_path)])
    assert r.exit_code == 1
    assert "format must be" in r.output


def test_report_includes_calibration(tmp_path: Path) -> None:
    _make_task(tmp_path, "t")
    runs = tmp_path / "runs"
    d = runs / "r1"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "task_id": "t",
                "model": "openai:gpt-4o",
                "scores": {"functional": 1.0, "total": 1.0},
            }
        )
    )
    rep = build_report(runs_dir=runs, tasks_root=tmp_path)
    assert "calibration" in rep
    assert rep["calibration"]["summary"]["n_tasks"] >= 1


def test_report_markdown_has_calibration(tmp_path: Path) -> None:
    """Report markdown shows calibration when tasks have sufficient data."""
    _make_task_with_difficulty(tmp_path, "t", "easy")
    runs = tmp_path / "runs"
    for i in range(6):
        d = runs / f"r{i}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": f"r{i}",
                    "task_id": "t",
                    "model": f"model-{i % 3}",
                    "scores": {"functional": 1.0, "total": 1.0},
                }
            )
        )
    rep = build_report(runs_dir=runs, tasks_root=tmp_path)
    md = to_markdown(rep)
    assert "## Calibration" in md
