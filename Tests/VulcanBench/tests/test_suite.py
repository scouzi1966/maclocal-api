"""Tests for suite loading and running."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import harness.suite as suite_mod
from harness.suite import load_suite, run_suite


def _make_task(suite_dir: Path, task_id: str, functional: float = 1.0) -> None:
    d = suite_dir / task_id
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"id": task_id, "languages": ["python"]}))
    (d / "issue.md").write_text(f"Create hello.py that prints '{task_id}'.")
    # Legacy verifier returns a fixed functional score for deterministic tests.
    (d / "verifier.py").write_text(
        f"import json; print(json.dumps({{'functional': {functional}}}))"
    )


def test_load_suite_lists_tasks(tmp_path: Path) -> None:
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a")
    _make_task(base / "demo", "task-b")
    suite = load_suite("demo", tasks_base=base)
    assert suite.name == "demo"
    assert suite.task_ids == ["task-a", "task-b"]


def test_load_suite_manifest_pins_subset(tmp_path: Path) -> None:
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a")
    _make_task(base / "demo", "task-b")
    (base / "demo" / "suite.json").write_text(json.dumps({"tasks": ["task-b"]}))
    assert load_suite("demo", tasks_base=base).task_ids == ["task-b"]


def test_load_suite_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_suite("nope", tasks_base=tmp_path / "tasks")


def test_run_suite_runs_all_tasks(tmp_path: Path) -> None:
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a")
    _make_task(base / "demo", "task-b")
    out = tmp_path / "runs"

    result = run_suite("demo", "mock:synthetic", output_dir=out, tasks_base=base, judges=False)
    assert result["n_tasks"] == 2
    assert {t["task_id"] for t in result["tasks"]} == {"task-a", "task-b"}
    # Aggregate present for the one model, both tasks tagged with the suite.
    assert len(result["aggregate"]) == 1
    agg = result["aggregate"][0]
    assert agg["model"] == "mock:synthetic"
    assert agg["n_tasks"] == 2
    # suite.json written under the suite_id dir.
    assert (out / result["suite_id"] / "suite.json").exists()


def test_run_suite_repeat(tmp_path: Path) -> None:
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a")
    _make_task(base / "demo", "task-b")
    out = tmp_path / "runs"

    result = run_suite(
        "demo", "mock:synthetic", output_dir=out, tasks_base=base, judges=False, repeat=3
    )
    assert result["repeat"] == 3
    assert result["n_runs"] == 6  # 2 tasks x 3 attempts
    assert len(result["tasks"]) == 6
    agg = result["aggregate"][0]
    assert agg["n_tasks"] == 2  # distinct tasks
    assert agg["n_runs"] == 6  # attempts
    assert agg["repeats"] == 3.0
    # The suite aggregate must only include THIS invocation's runs (suite_id match):
    # a second invocation into the same output_dir should not inflate it.
    result2 = run_suite(
        "demo", "mock:synthetic", output_dir=out, tasks_base=base, judges=False, repeat=1
    )
    assert result2["aggregate"][0]["n_runs"] == 2


def test_run_suite_passes_effort_and_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a")
    seen: list[dict[str, object]] = []

    def fake_run_agent(**kwargs):  # type: ignore[no-untyped-def]
        seen.append(kwargs)
        return {
            "run_id": "task-a-x",
            "replay": "",
            "summary": {
                "run_id": "task-a-x",
                "task_id": "task-a",
                "model": kwargs["model"],
                "effort": {"requested": kwargs["effort"]},
                "experiment_id": kwargs["experiment_id"],
                "scores": {"functional": 1.0, "total": 1.0},
                "cost_usd": 0.0,
                "duration_s": 1.0,
            },
        }

    monkeypatch.setattr(suite_mod, "run_agent", fake_run_agent)
    res = run_suite(
        "demo",
        "mock:synthetic",
        output_dir=tmp_path / "runs",
        tasks_base=base,
        effort="low",
        experiment_id="experiment-1",
    )
    assert seen[0]["effort"] == "low"
    assert seen[0]["experiment_id"] == "experiment-1"
    assert res["effort"] == "low"
    assert res["experiment_id"] == "experiment-1"
    assert res["tasks"][0]["effort"]["requested"] == "low"


def _fake_run_agent_costing(cost: float, calls: list[str]):  # type: ignore[no-untyped-def]
    """A run_agent stand-in that returns a fixed per-run cost (no real work)."""

    def fake(*, task_id: str, **k):  # type: ignore[no-untyped-def]
        calls.append(task_id)
        return {
            "run_id": f"{task_id}-x{len(calls)}",
            "replay": "",
            "summary": {
                "run_id": f"{task_id}-x{len(calls)}",
                "task_id": task_id,
                "model": "mock:synthetic",
                "scores": {"functional": 1.0, "total": 1.0},
                "cost_usd": cost,
                "duration_s": 1.0,
            },
        }

    return fake


def test_max_cost_stops_launching_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once accumulated cost >= max_cost, no new runs launch; the rest are skipped."""
    base = tmp_path / "tasks"
    for t in ("a", "b", "c", "d"):
        _make_task(base / "demo", t)
    calls: list[str] = []
    monkeypatch.setattr(suite_mod, "run_agent", _fake_run_agent_costing(0.5, calls))

    res = run_suite(
        "demo", "mock:synthetic", output_dir=tmp_path / "runs", tasks_base=base, max_cost=1.0
    )
    # 0.5 + 0.5 = 1.0 >= cap after 2 runs -> 2 run, 2 skipped; only 2 launched.
    assert res["n_runs"] == 2
    assert res["n_skipped"] == 2
    assert res["spent_usd"] == 1.0
    assert len(calls) == 2  # remaining units were never launched
    assert len(res["skipped"]) == 2


def test_max_cost_overshoot_bounded_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "tasks"
    for t in ("a", "b", "c", "d", "e", "f"):
        _make_task(base / "demo", t)
    calls: list[str] = []
    monkeypatch.setattr(suite_mod, "run_agent", _fake_run_agent_costing(0.5, calls))

    res = run_suite(
        "demo",
        "mock:synthetic",
        output_dir=tmp_path / "runs",
        tasks_base=base,
        max_cost=1.0,
        max_concurrency=2,
    )
    # Budget is enforced; some runs skipped; launched count == completed count;
    # overshoot bounded (never launches the whole grid of 6).
    assert res["n_skipped"] >= 1
    assert res["n_runs"] + res["n_skipped"] == 6
    assert len(calls) == res["n_runs"]
    assert res["spent_usd"] >= 1.0


def test_no_max_cost_runs_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "tasks"
    for t in ("a", "b", "c"):
        _make_task(base / "demo", t)
    calls: list[str] = []
    monkeypatch.setattr(suite_mod, "run_agent", _fake_run_agent_costing(99.0, calls))
    res = run_suite("demo", "mock:synthetic", output_dir=tmp_path / "runs", tasks_base=base)
    assert res["n_runs"] == 3 and res["n_skipped"] == 0  # no cap -> all run


def test_max_cost_fails_closed_on_unknown_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a completed run reports no cost while a budget is set, stop launching
    (fail closed) instead of silently running the whole suite."""
    base = tmp_path / "tasks"
    for t in ("a", "b", "c", "d"):
        _make_task(base / "demo", t)
    calls: list[str] = []

    def fake(*, task_id: str, **k):  # type: ignore[no-untyped-def]
        calls.append(task_id)
        return {
            "run_id": f"{task_id}-x",
            "replay": "",
            "summary": {
                "run_id": f"{task_id}-x",
                "task_id": task_id,
                "model": "mock:synthetic",
                "scores": {"functional": 1.0, "total": 1.0},
                "cost_usd": None,  # unknown cost
                "duration_s": 1.0,
            },
        }

    monkeypatch.setattr(suite_mod, "run_agent", fake)
    res = run_suite(
        "demo", "mock:synthetic", output_dir=tmp_path / "runs", tasks_base=base, max_cost=1.0
    )
    assert res["cost_unknown"] is True
    assert res["n_skipped"] >= 1  # did NOT run the whole suite
    assert len(calls) < 4  # stopped launching after the first unknown-cost run


def test_max_cost_non_positive_rejected(tmp_path: Path) -> None:
    base = tmp_path / "tasks"
    _make_task(base / "demo", "a")
    with pytest.raises(ValueError):
        run_suite(
            "demo", "mock:synthetic", output_dir=tmp_path / "runs", tasks_base=base, max_cost=0
        )


def test_run_suite_repeat_zero_rejected(tmp_path: Path) -> None:
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a")
    with pytest.raises(ValueError):
        run_suite("demo", "mock:synthetic", output_dir=tmp_path / "runs", tasks_base=base, repeat=0)


def _agg_signature(result: dict) -> dict:  # type: ignore[type-arg]
    a = result["aggregate"][0]
    return {k: a[k] for k in ("n_tasks", "n_runs", "pass_at_1", "pass_at_k", "avg_total")}


def test_parallel_matches_sequential(tmp_path: Path) -> None:
    """max_concurrency must not change results — only wall-clock."""
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a", functional=1.0)
    _make_task(base / "demo", "task-b", functional=0.0)  # discriminating case
    _make_task(base / "demo", "task-c", functional=1.0)

    seq = run_suite(
        "demo",
        "mock:synthetic",
        output_dir=tmp_path / "seq",
        tasks_base=base,
        judges=False,
        repeat=2,
        max_concurrency=1,
    )
    par = run_suite(
        "demo",
        "mock:synthetic",
        output_dir=tmp_path / "par",
        tasks_base=base,
        judges=False,
        repeat=2,
        max_concurrency=4,
    )
    assert _agg_signature(seq) == _agg_signature(par)
    # pass@1 = mean(1.0, 0.0, 1.0) = 2/3 regardless of concurrency.
    assert par["aggregate"][0]["pass_at_1"] == round(2 / 3, 4)
    assert par["aggregate"][0]["n_runs"] == 6
    # Per-task: each task ran exactly `repeat` times, in isolated run dirs.
    runs_by_task: dict[str, set[str]] = {}
    for t in par["tasks"]:
        runs_by_task.setdefault(t["task_id"], set()).add(t["run_id"])
    assert all(len(ids) == 2 for ids in runs_by_task.values())


def test_parallel_error_containment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One unit raising is recorded; the rest of the suite still completes."""
    base = tmp_path / "tasks"
    _make_task(base / "demo", "task-a")
    _make_task(base / "demo", "task-b")

    real = suite_mod.run_agent

    def flaky(**kwargs):  # type: ignore[no-untyped-def]
        if kwargs["task_id"] == "task-b":
            raise RuntimeError("boom in task-b")
        return real(**kwargs)

    monkeypatch.setattr(suite_mod, "run_agent", flaky)
    result = run_suite(
        "demo",
        "mock:synthetic",
        output_dir=tmp_path / "runs",
        tasks_base=base,
        judges=False,
        max_concurrency=2,
    )
    assert len(result["errors"]) == 1
    assert result["errors"][0]["task_id"] == "task-b"
    assert {t["task_id"] for t in result["tasks"]} == {"task-a"}
    assert result["aggregate"][0]["n_tasks"] == 1  # only task-a produced a run
