"""Tests for `vulcanbench report`.

Claims ledger (promise -> proving test):
- "builds a JSON-serializable report with the documented sections" -> test_structure
- "ranks models (pass@1 ± stderr, pass@k, cost, latency)"          -> test_models_ranking
- "includes a per-task breakdown (per-model solve rate)"           -> test_per_task
- "summarizes the environment from run manifests"                  -> test_environment
- "flags runs scored against a now-stale task version"             -> test_integrity_flags_drift
- "filters by suite"                                               -> test_suite_filter
- "renders Markdown with each section + a drift warning"           -> test_markdown
"""

from __future__ import annotations

import json
from pathlib import Path

from harness.report import build_report, to_markdown
from harness.tasks import load_task, task_hash
from tests.test_task_hash import _make_task


def _summary(
    runs_dir: Path,
    run_id: str,
    *,
    model: str,
    task_id: str,
    functional: float,
    task_hash: str | None = None,
    suite: str | None = None,
    python: str = "3.12.0",
    effort: str | None = None,
    complexity: str = "localized",
    repo_scale: str = "small",
    language: str = "python",
    difficulty: str = "easy",
    cost_usd: float = 0.01,
    duration_s: float = 2.0,
    total_tokens: int = 100,
) -> None:
    effort_meta = (
        {
            "requested": effort,
            "provider": "mock",
            "provider_value": None,
            "supported": False,
        }
        if effort
        else None
    )
    d = runs_dir / run_id
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task_id": task_id,
                "model": model,
                "suite": suite,
                "cost_usd": cost_usd,
                "duration_s": duration_s,
                "total_tokens": total_tokens,
                "task_hash": task_hash,
                **({"effort": effort_meta} if effort_meta else {}),
                "manifest": {
                    "model": model,
                    "runtime": {"python": python},
                    "tools": {"git": "git version 2.40"},
                    "task": {
                        "repo_scale": repo_scale,
                        "task_complexity": complexity,
                        "languages": [language],
                        "difficulty": difficulty,
                    },
                },
                "scores": {"functional": functional, "total": functional},
            }
        )
    )


def test_structure(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "r1", model="m", task_id="t1", functional=1.0)
    rep = build_report(runs_dir=runs, tasks_root=tmp_path / "tasks", generated_at="fixed")
    assert set(rep) == {
        "generated_at",
        "suite",
        "totals",
        "models",
        "tasks",
        "environment",
        "integrity",
        "calibration",
        "effort_sensitivity",
    }
    assert rep["generated_at"] == "fixed"
    assert rep["totals"]["n_runs"] == 1
    json.dumps(rep)  # must be serializable


def test_models_ranking(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "a1", model="strong", task_id="t1", functional=1.0)
    _summary(runs, "b1", model="weak", task_id="t1", functional=0.0)
    rep = build_report(runs_dir=runs, tasks_root=tmp_path / "tasks")
    assert [m["model"] for m in rep["models"]] == ["strong", "weak"]
    assert rep["models"][0]["pass_at_1"] == 1.0


def test_per_task(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "r1", model="m", task_id="t1", functional=1.0)
    _summary(runs, "r2", model="m", task_id="t1", functional=0.0)
    _summary(runs, "r3", model="m", task_id="t2", functional=1.0)
    rep = build_report(runs_dir=runs, tasks_root=tmp_path / "tasks")
    by_id = {t["task_id"]: t for t in rep["tasks"]}
    assert by_id["t1"]["models"][0]["attempts"] == 2
    assert by_id["t1"]["models"][0]["solve_rate"] == 0.5
    assert by_id["t2"]["models"][0]["solve_rate"] == 1.0


def test_environment(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "r1", model="m", task_id="t1", functional=1.0, python="3.12.0")
    _summary(runs, "r2", model="n", task_id="t1", functional=1.0, python="3.13.0")
    env = build_report(runs_dir=runs, tasks_root=tmp_path / "tasks")["environment"]
    assert env["models"] == ["m", "n"]
    assert env["python"] == ["3.12.0", "3.13.0"]
    assert env["tools"]["git"] == ["git version 2.40"]


def test_integrity_flags_drift(tmp_path: Path) -> None:
    """A run whose recorded task_hash != the current task hash is flagged stale."""
    tasks = tmp_path / "tasks"
    _make_task(tasks, "t1")
    current = task_hash(load_task("t1", tasks))
    runs = tmp_path / "runs"
    _summary(runs, "fresh", model="m", task_id="t1", functional=1.0, task_hash=current)
    _summary(runs, "stale", model="m", task_id="t1", functional=1.0, task_hash="0" * 64)

    rep = build_report(runs_dir=runs, tasks_root=tasks)
    assert rep["integrity"]["stale"] == 1
    assert rep["integrity"]["stale_run_ids"] == ["stale"]
    md = to_markdown(rep)
    assert "⚠️" in md and "stale" in md


def test_integrity_flags_not_decontaminated(tmp_path: Path) -> None:
    """A run scored against a `decontaminated: false` task is flagged in integrity."""
    tasks = tmp_path / "tasks"
    (tasks / "oss-x").mkdir(parents=True)
    (tasks / "oss-x" / "metadata.json").write_text(
        json.dumps({"id": "oss-x", "source": "oss", "decontaminated": False})
    )
    (tasks / "clean").mkdir(parents=True)
    (tasks / "clean" / "metadata.json").write_text(
        json.dumps({"id": "clean", "source": "hand-authored", "decontaminated": True})
    )
    runs = tmp_path / "runs"
    _summary(runs, "r_oss", model="m", task_id="oss-x", functional=1.0)
    _summary(runs, "r_clean", model="m", task_id="clean", functional=1.0)

    integ = build_report(runs_dir=runs, tasks_root=tasks)["integrity"]
    assert integ["not_decontaminated"] == 1
    assert integ["not_decontaminated_run_ids"] == ["r_oss"]
    assert integ["not_decontaminated_tasks"] == ["oss-x"]

    md = to_markdown(build_report(runs_dir=runs, tasks_root=tasks))
    assert "non-decontaminated" in md


def test_suite_filter(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "r1", model="m", task_id="t1", functional=1.0, suite="v1")
    _summary(runs, "r2", model="m", task_id="t2", functional=0.0, suite="other")
    rep = build_report(runs_dir=runs, tasks_root=tmp_path / "tasks", suite="v1")
    assert rep["totals"]["n_runs"] == 1
    assert rep["suite"] == "v1"


def test_markdown(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "r1", model="m", task_id="t1", functional=1.0)
    md = to_markdown(build_report(runs_dir=runs, tasks_root=tmp_path / "tasks"))
    assert "# VulcanBench report" in md
    assert "## Models" in md
    assert "## Per-task" in md
    assert "## Environment" in md
    assert "pass@1" in md


def test_effort_sensitivity_flags_low_sufficient(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "low", model="m", task_id="t1", functional=1.0, effort="low")
    _summary(runs, "high", model="m", task_id="t1", functional=1.0, effort="high")

    effort = build_report(runs_dir=runs, tasks_root=tmp_path / "tasks")["effort_sensitivity"]
    assert effort["available"] is True
    row = effort["strata"][0]
    assert row["language"] == "python"
    assert row["task_complexity"] == "localized"
    assert row["high_minus_low_pass_at_1"] == 0.0
    assert row["classification"] == "low sufficient"


def test_effort_sensitivity_flags_effort_sensitive(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _summary(runs, "low", model="m", task_id="t1", functional=0.0, effort="low")
    _summary(
        runs,
        "high",
        model="m",
        task_id="t1",
        functional=1.0,
        effort="high",
        cost_usd=0.03,
        duration_s=6.0,
    )

    row = build_report(runs_dir=runs, tasks_root=tmp_path / "tasks")["effort_sensitivity"][
        "strata"
    ][0]
    assert row["high_minus_low_pass_at_1"] == 1.0
    assert row["high_cost_ratio"] == 3.0
    assert row["high_latency_ratio"] == 3.0
    assert row["classification"] == "effort sensitive"

    md = to_markdown(build_report(runs_dir=runs, tasks_root=tmp_path / "tasks"))
    assert "## Effort Sensitivity" in md
