"""Tests for per-model leaderboard aggregation and repeat statistics."""

from __future__ import annotations

import math

from harness.leaderboard import _stderr, aggregate_by_model


def _row(model, task_id, functional, total=None, cost=None, duration=None, suite=None):  # type: ignore[no-untyped-def]
    return {
        "model": model,
        "task_id": task_id,
        "total": total if total is not None else functional,
        "functional": functional,
        "quality": 1.0,
        "security": 1.0,
        "efficiency": 0.9,
        "human_like": 0.8,
        "cost_usd": cost,
        "duration_s": duration,
        "total_tokens": 100,
        "suite": suite,
    }


def test_distinct_tasks_vs_runs() -> None:
    # 2 attempts of one task, 1 of another -> 2 tasks, 3 runs.
    rows = [
        _row("m", "t1", 1.0),
        _row("m", "t1", 0.0),
        _row("m", "t2", 1.0),
    ]
    agg = aggregate_by_model(rows)[0]
    assert agg["n_tasks"] == 2
    assert agg["n_runs"] == 3
    assert agg["repeats"] == 1.5


def test_pass_at_1_and_pass_at_k() -> None:
    # t1: solved 2/3 attempts; t2: solved 0/3 attempts.
    rows = [
        _row("m", "t1", 1.0),
        _row("m", "t1", 0.0),
        _row("m", "t1", 1.0),
        _row("m", "t2", 0.0),
        _row("m", "t2", 0.0),
        _row("m", "t2", 0.0),
    ]
    agg = aggregate_by_model(rows)[0]
    # pass@1 = mean(2/3, 0/3) = 1/3
    assert agg["pass_at_1"] == round(1 / 3, 4)
    # pass@k = mean(solved-at-least-once) = mean(1, 0) = 0.5
    assert agg["pass_at_k"] == 0.5
    # stderr across the two per-task rates {0.6667, 0.0}
    assert agg["pass_at_1_stderr"] == _stderr([2 / 3, 0.0])


def test_stderr_zero_with_single_task() -> None:
    agg = aggregate_by_model([_row("m", "t1", 1.0), _row("m", "t1", 0.0)])[0]
    assert agg["n_tasks"] == 1
    assert agg["pass_at_1_stderr"] == 0.0  # only one task -> no across-task spread
    assert agg["pass_at_1"] == 0.5  # 1/2 attempts solved


def test_stderr_formula() -> None:
    vals = [1.0, 0.0, 1.0, 0.0]
    expected = round((__import__("statistics").stdev(vals)) / math.sqrt(len(vals)), 4)
    assert _stderr(vals) == expected
    assert _stderr([0.5]) == 0.0
    assert _stderr([]) == 0.0


def test_sorted_by_pass_at_1_desc() -> None:
    rows = [_row("weak", "t1", 0.0), _row("strong", "t1", 1.0)]
    assert [a["model"] for a in aggregate_by_model(rows)] == ["strong", "weak"]


def test_total_cost_known_flag() -> None:
    rows = [_row("m", "t1", 1.0, cost=0.01), _row("m", "t2", 1.0, cost=None)]
    agg = aggregate_by_model(rows)[0]
    assert agg["cost_known"] is False  # one run unpriced
    assert agg["total_cost"] == 0.01


def test_suite_filter() -> None:
    rows = [_row("m", "t1", 1.0, suite="v1"), _row("m", "t2", 0.0, suite="other")]
    aggs = aggregate_by_model(rows, suite="v1")
    assert len(aggs) == 1 and aggs[0]["n_tasks"] == 1 and aggs[0]["pass_at_1"] == 1.0
