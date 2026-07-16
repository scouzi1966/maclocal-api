"""Tests for pre-run cost estimation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli import app
from harness.cost_estimate import estimate_plan
from harness.cost_priors import load_cost_priors, reset_cache

runner = CliRunner()
_FIXTURE_PRIORS = Path(__file__).resolve().parent / "fixtures" / "cost_priors_min.json"


def _write_summary(runs_dir: Path, run_id: str, payload: dict) -> None:
    d = runs_dir / run_id
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_estimate_exact_history(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_summary(
        runs,
        "a",
        {
            "task_id": "hello-world",
            "model": "openai:gpt-5.5",
            "cost_usd": 0.02,
            "manifest": {"task": {"repo_scale": "micro"}},
        },
    )
    plan = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["hello-world"],
        judges=False,
        runs_dir=runs,
    )
    assert plan.mid_usd == 0.02
    assert plan.models[0].confidence == "high"


def test_estimate_scales_across_models(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_summary(
        runs,
        "a",
        {
            "task_id": "hello-world",
            "model": "zai:glm-5.2",
            "cost_usd": 0.01,
            "manifest": {"task": {"repo_scale": "micro"}},
        },
    )
    plan = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["hello-world"],
        judges=False,
        runs_dir=runs,
    )
    assert plan.mid_usd > 0
    assert plan.models[0].per_task[0].source == "task_scaled"


def test_estimate_judges_multiplier(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_summary(
        runs,
        "a",
        {
            "task_id": "hello-world",
            "model": "openai:gpt-5.5",
            "cost_usd": 0.03,
            "manifest": {"task": {"repo_scale": "micro"}},
        },
    )
    off = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["hello-world"],
        judges=False,
        runs_dir=runs,
        use_priors=False,
    )
    on = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["hello-world"],
        judges=True,
        runs_dir=runs,
        use_priors=False,
    )
    assert on.mid_usd == pytest.approx(off.mid_usd * 3)


def test_estimate_bundled_priors_cold_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("VULCANBENCH_COST_PRIORS", str(_FIXTURE_PRIORS))
    reset_cache()
    plan = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["hello-world"],
        judges=True,
        runs_dir=runs,
    )
    assert plan.mid_usd == 0.04
    assert plan.models[0].per_task[0].source == "prior_exact"
    assert plan.models[0].confidence == "medium"
    assert any("bundled cost priors" in n for n in plan.models[0].notes)


def test_estimate_local_overrides_priors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setenv("VULCANBENCH_COST_PRIORS", str(_FIXTURE_PRIORS))
    reset_cache()
    _write_summary(
        runs,
        "a",
        {
            "task_id": "hello-world",
            "model": "openai:gpt-5.5",
            "cost_usd": 0.02,
            "manifest": {"task": {"repo_scale": "micro"}},
        },
    )
    plan = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["hello-world"],
        judges=False,
        runs_dir=runs,
    )
    assert plan.mid_usd == 0.02
    assert plan.models[0].per_task[0].source == "exact"
    assert plan.models[0].confidence == "high"


def test_estimate_no_priors_uses_defaults(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    plan = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["hello-world"],
        judges=False,
        runs_dir=runs,
        use_priors=False,
    )
    assert plan.models[0].per_task[0].source == "default"
    assert plan.models[0].confidence == "low"


def test_bundled_priors_load() -> None:
    reset_cache()
    priors = load_cost_priors()
    assert priors.by_model_task.get(("openai:gpt-5.5", "py-topo-sort-cycle")) is not None
    assert priors.by_model_scale.get(("openai:gpt-5.5", "micro")) is not None


def test_cli_estimate_command() -> None:
    result = runner.invoke(
        app,
        [
            "estimate",
            "--task",
            "hello-world",
            "--model",
            "openai:gpt-5.5",
            "--no-judges",
        ],
    )
    assert result.exit_code == 0
    assert "Cost estimate" in result.output
    assert "OPENAI_API_KEY" in result.output


def test_run_dry_run_includes_estimate() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "openai:gpt-5.5",
            "--no-judges",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "dry-run" in result.output
    assert "Cost estimate" in result.output


def test_estimate_v1_compare_cold_start(tmp_path: Path) -> None:
    """Fresh install: v1-compare uses bundled priors, not flat defaults."""
    runs = tmp_path / "runs"
    reset_cache()
    plan = estimate_plan(
        models=["openai:gpt-5.5"],
        task_ids=["py-topo-sort-cycle", "go-stack-pop-bug"],
        judges=True,
        runs_dir=runs,
    )
    assert all(t.source == "prior_exact" for t in plan.models[0].per_task)
    assert plan.models[0].confidence == "medium"
    assert plan.mid_usd > 0.07
