"""Tests for the harness CLI."""

from __future__ import annotations

import json
from pathlib import Path

import docker
import pytest
from typer.testing import CliRunner

import harness.cli as cli_mod
from harness.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.5.1" in result.output


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vulcanbench" in result.output.lower()
    assert "run" in result.output


def test_run_unknown_task() -> None:
    result = runner.invoke(app, ["run", "--task", "does-not-exist", "--model", "mock:synthetic"])
    assert result.exit_code == 1
    assert "unknown task" in result.output


def test_run_dry_run() -> None:
    result = runner.invoke(
        app, ["run", "--task", "hello-world", "--model", "mock:synthetic", "--dry-run"]
    )
    assert result.exit_code == 0
    assert "dry-run" in result.output


def test_run_end_to_end(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--sandbox",
            "local",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run complete" in result.output
    assert "functional=1.0" in result.output
    assert "human_like=0.8" in result.output


def test_run_no_judges(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--sandbox",
            "local",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "human_like=None" in result.output


def test_run_bad_model_spec(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "--task", "hello-world", "--model", "garbage", "--output-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_run_invalid_sandbox(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--sandbox",
            "bogus",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "sandbox must be" in result.output


def test_run_docker_daemon_down(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def boom():  # type: ignore[no-untyped-def]
        raise ConnectionError("no daemon")

    monkeypatch.setattr(docker, "from_env", boom)
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--sandbox",
            "docker",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 3
    assert "sandbox error" in result.output.lower()


def test_run_auto_refuses_host_fallback_without_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom():  # type: ignore[no-untyped-def]
        raise ConnectionError("no daemon")

    monkeypatch.setattr(docker, "from_env", boom)
    monkeypatch.delenv("VULCANBENCH_ALLOW_HOST_EXEC", raising=False)
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--sandbox",
            "auto",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 3
    assert "VULCANBENCH_ALLOW_HOST_EXEC" in result.output


def test_run_auto_falls_back_with_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom():  # type: ignore[no-untyped-def]
        raise ConnectionError("no daemon")

    monkeypatch.setattr(docker, "from_env", boom)
    monkeypatch.setenv("VULCANBENCH_ALLOW_HOST_EXEC", "1")
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--sandbox",
            "auto",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "run complete" in result.output


def test_run_requires_exactly_one_target(tmp_path: Path) -> None:
    # Neither --task nor --suite.
    r1 = runner.invoke(app, ["run", "--model", "mock:synthetic", "--output-dir", str(tmp_path)])
    assert r1.exit_code == 1
    assert "exactly one" in r1.output
    # Both --task and --suite.
    r2 = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--suite",
            "v1",
            "--model",
            "mock:synthetic",
            "-o",
            str(tmp_path),
        ],
    )
    assert r2.exit_code == 1


def test_run_suite_end_to_end(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "v1",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--sandbox",
            "local",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "running suite" in result.output
    assert "pass@1" in result.output


def test_run_task_repeat(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--repeat",
            "3",
            "--sandbox",
            "local",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "pass@1=1.0" in result.output
    assert "pass@3=1.0" in result.output
    # 3 run directories were created.
    assert sum(1 for p in tmp_path.iterdir() if p.is_dir()) == 3


def test_effort_sweep_writes_experiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_run_suite(name, model, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs["effort"])
        return {
            "suite_id": f"suite-{kwargs['effort']}",
            "suite": name,
            "model": model,
            "effort": kwargs["effort"],
            "experiment_id": kwargs["experiment_id"],
            "repeat": kwargs["repeat"],
            "max_concurrency": kwargs["max_concurrency"],
            "tasks": [],
            "errors": [],
            "aggregate": [],
        }

    monkeypatch.setattr(cli_mod, "run_suite", fake_run_suite)
    result = runner.invoke(
        app,
        [
            "effort-sweep",
            "--suite",
            "v1",
            "--model",
            "mock:synthetic",
            "--efforts",
            "low,high",
            "--repeat",
            "2",
            "--max-concurrency",
            "3",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["low", "high"]
    experiment_paths = list(tmp_path.glob("experiment-*/experiment.json"))
    assert len(experiment_paths) == 1
    experiment = json.loads(experiment_paths[0].read_text())
    assert experiment["efforts"] == ["low", "high"]
    assert experiment["repeat"] == 2
    assert {s["effort"] for s in experiment["suites"]} == {"low", "high"}


def test_run_repeat_zero_rejected(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--repeat",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "repeat" in result.output


def test_report_writes_md_and_json(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--sandbox",
            "local",
            "--output-dir",
            str(runs),
        ],
    )
    md_path = tmp_path / "report.md"
    r1 = runner.invoke(app, ["report", "--runs-dir", str(runs), "-o", str(md_path)])
    assert r1.exit_code == 0
    assert md_path.exists() and "# VulcanBench report" in md_path.read_text()

    json_path = tmp_path / "report.json"
    r2 = runner.invoke(
        app, ["report", "--runs-dir", str(runs), "--format", "json", "-o", str(json_path)]
    )
    assert r2.exit_code == 0
    rep = json.loads(json_path.read_text())
    assert rep["totals"]["n_runs"] == 1

    # stdout path (no -o) prints the report.
    r3 = runner.invoke(app, ["report", "--runs-dir", str(runs)])
    assert r3.exit_code == 0
    assert "## Models" in r3.output


def test_report_bad_format(tmp_path: Path) -> None:
    r = runner.invoke(app, ["report", "--runs-dir", str(tmp_path), "--format", "xml"])
    assert r.exit_code == 1
    assert "format must be" in r.output


def _run_fail_under(task: str, threshold: str, tmp_path: Path):  # type: ignore[no-untyped-def]
    return runner.invoke(
        app,
        [
            "run",
            "--task",
            task,
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--fail-under",
            threshold,
            "--sandbox",
            "local",
            "--output-dir",
            str(tmp_path),
        ],
    )


def test_fail_under_below_threshold_exits_4(tmp_path: Path) -> None:
    # mock can't solve this -> pass@1 = 0.0, below 0.5 -> gate fails (exit 4).
    r = _run_fail_under("py-ttl-cache-expiry", "0.5", tmp_path)
    assert r.exit_code == 4
    assert "FAIL" in r.output


def test_fail_under_at_boundary_passes(tmp_path: Path) -> None:
    # hello-world pass@1 = 1.0; threshold 1.0 -> 1.0 >= 1.0 -> pass (exit 0).
    r = _run_fail_under("hello-world", "1.0", tmp_path)
    assert r.exit_code == 0
    assert "PASS" in r.output


def test_fail_under_zero_boundary_passes(tmp_path: Path) -> None:
    # pass@1 = 0.0; threshold 0.0 -> 0.0 >= 0.0 -> pass (exit 0).
    r = _run_fail_under("py-ttl-cache-expiry", "0.0", tmp_path)
    assert r.exit_code == 0
    assert "PASS" in r.output


def test_fail_under_above_threshold_passes(tmp_path: Path) -> None:
    r = _run_fail_under("hello-world", "0.5", tmp_path)
    assert r.exit_code == 0


def test_fail_under_fails_closed_on_suite_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate must NOT go green when a suite unit errored, even if the
    surviving subset has pass@1 == 1.0 (reviewer's P1 repro)."""

    def fake_run_suite(*a, **k):  # type: ignore[no-untyped-def]
        return {
            "repeat": 1,
            "errors": [{"task_id": "boom", "error": "kaboom"}],
            "aggregate": [
                {
                    "model": "mock:synthetic",
                    "pass_at_1": 1.0,
                    "pass_at_1_stderr": 0.0,
                    "pass_at_k": 1.0,
                    "n_tasks": 1,
                    "n_runs": 1,
                    "avg_total": 1.0,
                    "total_cost": 0.0,
                }
            ],
        }

    monkeypatch.setattr(cli_mod, "run_suite", fake_run_suite)
    r = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "v1",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--fail-under",
            "0.9",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 4
    assert "errored" in r.output.lower()


def test_fail_under_rejects_non_finite_and_out_of_range(tmp_path: Path) -> None:
    """nan/inf and values outside [0,1] are rejected before running (reviewer's P1)."""
    for bad in ("nan", "inf", "-0.1", "1.5"):
        r = runner.invoke(
            app,
            [
                "run",
                "--task",
                "hello-world",
                "--model",
                "mock:synthetic",
                "--no-judges",
                "--fail-under",
                bad,
                "--output-dir",
                str(tmp_path / bad),
            ],
        )
        assert r.exit_code == 1, f"{bad}: {r.output}"
        assert "fail-under must be" in r.output


def test_max_cost_requires_priced_model(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "v1",
            "--model",
            "openai:nonexistent-9000",
            "--max-cost",
            "1.0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 1
    assert "priced" in r.output


def test_max_cost_unpriced_judge_is_named(tmp_path: Path) -> None:
    # Run model is priced (mock); only the judge model is unpriced -> the error
    # must name the judge, not the (priced) run model.
    r = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "v1",
            "--model",
            "mock:synthetic",
            "--judge-model",
            "openai:nonexistent-9000",
            "--max-cost",
            "1.0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 1
    assert "openai:nonexistent-9000" in r.output
    assert "mock:synthetic" not in r.output  # the priced model isn't blamed


def test_max_cost_requires_suite(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--max-cost",
            "1.0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 1
    assert "suite" in r.output


def test_max_cost_must_be_positive(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "v1",
            "--model",
            "mock:synthetic",
            "--max-cost",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 1


def test_fail_under_fails_closed_on_budget_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget-truncated suite (runs skipped) must not pass a CI gate, even with
    pass@1 == 1.0 on the subset that ran."""

    def fake_run_suite(*a, **k):  # type: ignore[no-untyped-def]
        return {
            "repeat": 1,
            "errors": [],
            "skipped": ["d", "e"],
            "n_skipped": 2,
            "max_cost": 1.0,
            "spent_usd": 1.2,
            "aggregate": [
                {
                    "model": "mock:synthetic",
                    "pass_at_1": 1.0,
                    "pass_at_1_stderr": 0.0,
                    "pass_at_k": 1.0,
                    "n_tasks": 1,
                    "n_runs": 1,
                    "avg_total": 1.0,
                    "total_cost": 1.2,
                }
            ],
        }

    monkeypatch.setattr(cli_mod, "run_suite", fake_run_suite)
    r = runner.invoke(
        app,
        [
            "run",
            "--suite",
            "v1",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--fail-under",
            "0.9",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 4
    assert "skipped" in r.output.lower()


def test_no_fail_under_exits_zero_even_when_unsolved(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "run",
            "--task",
            "py-ttl-cache-expiry",
            "--model",
            "mock:synthetic",
            "--no-judges",
            "--sandbox",
            "local",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert r.exit_code == 0  # no gate -> always 0 on a successful run


def test_leaderboard_by_model_and_run() -> None:
    assert runner.invoke(app, ["leaderboard", "--by", "model"]).exit_code == 0
    assert runner.invoke(app, ["leaderboard", "--by", "run"]).exit_code == 0


def test_list_tasks() -> None:
    result = runner.invoke(app, ["list-tasks"])
    assert result.exit_code == 0
    assert "hello-world" in result.output


def test_leaderboard_json(tmp_path: Path) -> None:
    runner.invoke(
        app,
        [
            "run",
            "--task",
            "hello-world",
            "--model",
            "mock:synthetic",
            "--sandbox",
            "local",
            "--output-dir",
            str(tmp_path),
        ],
    )
    result = runner.invoke(app, ["leaderboard", "--format", "json"])
    assert result.exit_code == 0
