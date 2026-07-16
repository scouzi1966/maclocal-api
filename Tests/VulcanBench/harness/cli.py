"""VulcanBench CLI entrypoint (Typer + Rich).

Commands: run, estimate, effort-sweep, leaderboard, report, calibrate, replay,
validate-task, list-tasks. See ``vulcanbench --help``.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from harness import __version__
from harness.agent.loop import run_agent
from harness.agent.providers import ProviderError
from harness.calibration import calibrate_tasks, calibration_to_markdown
from harness.cost_estimate import estimate_plan
from harness.effort import DEFAULT_SWEEP_EFFORTS, parse_efforts
from harness.leaderboard import aggregate_by_model, scan_leaderboard
from harness.pricing import is_priced
from harness.report import build_report, to_markdown
from harness.sandbox.docker_executor import SandboxError
from harness.suite import SUITE_ALIASES, load_suite, run_suite
from harness.tasks import list_task_ids
from harness.validate import main as validate_main

app = typer.Typer(
    name="vulcanbench",
    help="VulcanBench v1 - reproducible LLM benchmarking harness for SWE tasks",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]vulcanbench[/bold] {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    """VulcanBench CLI."""
    pass


@app.command()
def run(  # noqa: PLR0912, PLR0915 — CLI entry: option declarations + linear guards + dispatch
    task: str | None = typer.Option(None, "--task", "-t", help="Task ID e.g. swe-001"),
    suite: str | None = typer.Option(None, "--suite", help="Run a whole suite, e.g. v1 (tasks/v1)"),
    model: str = typer.Option(..., "--model", "-m", help="provider:model e.g. openai:gpt-4o"),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("./runs"), "--output-dir", "-o", help="Where to write trace/replay"
    ),
    max_steps: int | None = typer.Option(
        None, "--max-steps", help="Cap agent steps (default: per-task metadata / scale tier)"
    ),
    judges: bool = typer.Option(
        True, "--judges/--no-judges", help="Run the human_like LLM judge ensemble"
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Model for judges (default: same as --model)"
    ),
    sandbox: str = typer.Option(
        "docker",
        "--sandbox",
        help="Where tools run: local|docker|auto (default docker; agents execute "
        "model-written shell commands, so opt into 'local' deliberately)",
    ),
    image: str | None = typer.Option(
        None,
        "--image",
        help="Docker image for the sandbox (default: per task or vulcanbench/sandbox:base)",
    ),
    network: bool = typer.Option(
        False, "--network/--no-network", help="Allow network in the docker sandbox"
    ),
    repeat: int = typer.Option(1, "--repeat", help="Run each task N times (for pass@k / stderr)"),
    max_concurrency: int = typer.Option(
        1, "--max-concurrency", help="Run suite tasks in parallel (suite runs only)"
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="USD spend cap for a suite run (stops launching new runs)"
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Per-run wall-clock budget in seconds (abort if exceeded)"
    ),
    fail_under: float | None = typer.Option(
        None, "--fail-under", help="Exit non-zero (4) if pass@1 is below this threshold (CI gate)"
    ),
    effort: str | None = typer.Option(
        None,
        "--effort",
        help="Normalized reasoning effort: low|medium|high|extra-high",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, do not launch sandbox"),
    use_priors: bool = typer.Option(
        True,
        "--priors/--no-priors",
        help="Use bundled cost priors when local ./runs history is missing (dry-run estimate)",
    ),
) -> None:
    """Run an agent against a task or a whole suite, recording full traces."""
    if (task is None) == (suite is None):
        console.print("[red]error[/red] pass exactly one of --task or --suite")
        raise typer.Exit(code=1)
    if repeat < 1:
        console.print("[red]error[/red] --repeat must be >= 1")
        raise typer.Exit(code=1)
    if max_concurrency < 1:
        console.print("[red]error[/red] --max-concurrency must be >= 1")
        raise typer.Exit(code=1)
    if fail_under is not None and not (math.isfinite(fail_under) and 0.0 <= fail_under <= 1.0):
        console.print(
            f"[red]error[/red] --fail-under must be a finite number in [0, 1], got {fail_under}"
        )
        raise typer.Exit(code=1)
    if max_cost is not None:
        if suite is None:
            console.print("[red]error[/red] --max-cost applies to suite runs (use --suite)")
            raise typer.Exit(code=1)
        if not (math.isfinite(max_cost) and max_cost > 0):
            console.print(
                f"[red]error[/red] --max-cost must be a finite number > 0, got {max_cost}"
            )
            raise typer.Exit(code=1)
        judge = judge_model or model
        unpriced = [m for m, used in ((model, True), (judge, judges)) if used and not is_priced(m)]
        if unpriced:
            names = ", ".join(sorted(set(unpriced)))
            console.print(f"[red]error[/red] --max-cost needs priced models; no price for {names}")
            raise typer.Exit(code=1)
    if sandbox not in {"local", "docker", "auto"}:
        console.print(f"[red]error[/red] --sandbox must be local|docker|auto, got {sandbox!r}")
        raise typer.Exit(code=1)
    if task is not None and task not in list_task_ids():
        available = ", ".join(list_task_ids()) or "(none)"
        console.print(f"[red]unknown task[/red] {task!r}. Available: {available}")
        raise typer.Exit(code=1)
    if dry_run:
        target = f"suite={suite}" if suite else f"task={task}"
        effort_note = f" effort={effort}" if effort else ""
        console.print(
            f"[yellow]dry-run[/yellow] would run {target} model={model} "
            f"sandbox={sandbox}{effort_note}"
        )
        if is_priced(model):
            try:
                task_ids = (
                    load_suite(suite).task_ids if suite is not None else [task]  # type: ignore[list-item]
                )
                plan = estimate_plan(
                    models=[model],
                    task_ids=task_ids,
                    repeat=repeat,
                    judges=judges,
                    runs_dir=output_dir,
                    use_priors=use_priors,
                )
                _print_cost_estimate(plan)
            except ValueError as e:
                console.print(f"[yellow]cost estimate skipped[/yellow]: {e}")
        raise typer.Exit()

    run_kwargs = {
        "max_steps": max_steps,
        "judges": judges,
        "judge_model": judge_model,
        "sandbox": sandbox,
        "image": image,
        "network": network,
        "timeout_s": timeout,
        "effort": effort,
    }
    pass_at_1: float | None = None
    n_incomplete = 0  # errored + skipped runs (an incomplete suite)
    try:
        if suite is not None:
            pass_at_1, n_incomplete = _run_suite(
                suite, model, output_dir, run_kwargs, repeat, max_concurrency, max_cost
            )
        else:
            pass_at_1, n_incomplete = _run_single(task, model, output_dir, run_kwargs, repeat)  # type: ignore[arg-type]
    except SandboxError as e:
        console.print(f"[red]sandbox error[/red] {e}")
        raise typer.Exit(code=3) from e
    except ProviderError as e:
        console.print(f"[red]provider error[/red] {e}")
        raise typer.Exit(code=2) from e
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e

    if fail_under is not None:
        # Gate semantics: pass iff every unit ran AND pass@1 >= threshold.
        # An errored OR budget-skipped run, or an unavailable pass@1, fails
        # closed — a CI gate must never go green on a partial/unknown result.
        if n_incomplete:
            console.print(
                f"[red]FAIL[/red] {n_incomplete} run(s) errored or skipped — gate fails closed"
            )
            raise typer.Exit(code=4)
        if pass_at_1 is None or pass_at_1 < fail_under:
            console.print(f"[red]FAIL[/red] pass@1={pass_at_1} < --fail-under {fail_under}")
            raise typer.Exit(code=4)
        console.print(f"[green]PASS[/green] pass@1={pass_at_1} >= --fail-under {fail_under}")


def _print_cost_estimate(plan: Any, *, json_output: bool = False) -> None:
    if json_output:
        typer.echo(json.dumps(plan.to_dict(), indent=2))
        return

    n_runs = sum(m.n_runs for m in plan.models)
    console.print(
        f"\n[bold]Cost estimate[/bold]  {len(plan.task_ids)} tasks x {plan.repeat} repeat(s)"
        f" = {n_runs} run(s) per model" + ("  [dim](judges on)[/dim]" if plan.judges else "")
    )
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Provider / env")
    table.add_column("Low", justify="right")
    table.add_column("Mid", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Load ≥", justify="right", style="bold green")
    table.add_column("Conf.")
    for m in plan.models:
        table.add_row(
            m.model,
            f"{m.provider}\n[dim]{m.env_var}[/dim]",
            f"${m.low_usd:.2f}",
            f"${m.mid_usd:.2f}",
            f"${m.high_usd:.2f}",
            f"${m.recommended_usd:.2f}",
            m.confidence,
        )
    if len(plan.models) > 1:
        table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]${plan.low_usd:.2f}[/bold]",
            f"[bold]${plan.mid_usd:.2f}[/bold]",
            f"[bold]${plan.high_usd:.2f}[/bold]",
            f"[bold]${plan.recommended_usd:.2f}[/bold]",
            "",
        )
    console.print(table)
    for m in plan.models:
        for note in m.notes:
            console.print(f"  [dim]• {m.model}:[/dim] {note}")
    console.print(
        "\n[dim]Estimates prefer local ./runs history; bundled priors fill gaps on fresh "
        "installs. Load the recommended amount per provider before starting.[/dim]"
    )


@app.command()
def estimate(
    task: str | None = typer.Option(None, "--task", "-t", help="Single task ID"),
    suite: str | None = typer.Option(None, "--suite", help="Suite name, e.g. v1-compare"),
    model: list[str] = typer.Option(  # noqa: B008
        ..., "--model", "-m", help="provider:model (repeat for multiple models)"
    ),
    repeat: int = typer.Option(1, "--repeat", help="Planned repeats per task"),
    judges: bool = typer.Option(
        True, "--judges/--no-judges", help="Include judge ensemble cost (~3x agent tokens)"
    ),
    runs_dir: Path = typer.Option(  # noqa: B008
        Path("./runs"), "--runs-dir", help="Local runs dir for historical costs"
    ),
    use_priors: bool = typer.Option(
        True,
        "--priors/--no-priors",
        help="Use bundled cost priors when local ./runs history is missing",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Estimate API spend before running a benchmark."""
    if (task is None) == (suite is None):
        console.print("[red]error[/red] pass exactly one of --task or --suite")
        raise typer.Exit(code=1)
    if repeat < 1:
        console.print("[red]error[/red] --repeat must be >= 1")
        raise typer.Exit(code=1)
    if task is not None and task not in list_task_ids():
        console.print(f"[red]unknown task[/red] {task!r}")
        raise typer.Exit(code=1)

    try:
        task_ids = load_suite(suite).task_ids if suite is not None else [task]  # type: ignore[list-item]
        plan = estimate_plan(
            models=model,
            task_ids=task_ids,
            repeat=repeat,
            judges=judges,
            runs_dir=runs_dir,
            use_priors=use_priors,
        )
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e

    _print_cost_estimate(plan, json_output=json_output)


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Build a leaderboard-shaped row from a run summary (for in-memory aggregation)."""
    sc = summary.get("scores", {})
    return {
        "run_id": summary.get("run_id"),
        "task_id": summary.get("task_id"),
        "model": summary.get("model"),
        "total": sc.get("total"),
        "functional": sc.get("functional"),
        "cost_usd": summary.get("cost_usd"),
        "duration_s": summary.get("duration_s"),
        "total_tokens": summary.get("total_tokens"),
        "effort": summary.get("effort"),
    }


def _run_single(
    task: str, model: str, output_dir: Path, run_kwargs: dict[str, Any], repeat: int
) -> tuple[float | None, int]:
    """Run a single task ``repeat`` times; returns (pass@1, n_errors).

    A single run that fails raises out of here (caught by ``run``), so the
    error count is always 0 on this path.
    """
    summaries = []
    for _ in range(repeat):
        res = run_agent(task_id=task, model=model, output_dir=output_dir, **run_kwargs)
        summaries.append(res["summary"])

    agg = aggregate_by_model([_summary_row(s) for s in summaries])[0]
    if repeat == 1:
        scores = summaries[0].get("scores", {})
        console.print(f"[green]run complete[/green] {res['run_id']}")
        console.print(
            f"functional={scores.get('functional')} quality={scores.get('quality')} "
            f"security={scores.get('security')} human_like={scores.get('human_like')} "
            f"total={scores.get('total')} cost=${summaries[0].get('cost_usd')}"
        )
        console.print(f"replay: {res['replay']}")
    else:
        console.print(f"[green]{task}[/green] x{repeat} with {model}")
        console.print(
            f"pass@1={agg['pass_at_1']}±{agg['pass_at_1_stderr']}  "
            f"pass@{repeat}={agg['pass_at_k']}  "
            f"avg_total={agg['avg_total']}±{agg['avg_total_stderr']}  cost=${agg['total_cost']}"
        )
    p1 = agg["pass_at_1"]
    return (float(p1) if p1 is not None else None), 0


def _run_suite(
    suite: str,
    model: str,
    output_dir: Path,
    run_kwargs: dict[str, Any],
    repeat: int,
    max_concurrency: int,
    max_cost: float | None = None,
) -> tuple[float | None, int]:
    """Run a suite; returns (pass@1, n_incomplete) where n_incomplete counts
    errored + budget-skipped runs."""
    suffix = f" x{repeat}" if repeat > 1 else ""
    par = f" ({max_concurrency}-way parallel)" if max_concurrency > 1 else ""
    cap = f" (budget ${max_cost})" if max_cost is not None else ""
    console.print(f"[cyan]running suite[/cyan] {suite}{suffix}{par}{cap} with {model} ...")
    result = run_suite(
        suite,
        model,
        output_dir=output_dir,
        repeat=repeat,
        max_concurrency=max_concurrency,
        max_cost=max_cost,
        **run_kwargs,
    )
    n_errors = len(result.get("errors") or [])
    n_skipped = result.get("n_skipped", 0)
    if n_errors:
        console.print(f"[yellow]{n_errors} run(s) errored[/yellow]")
    if n_skipped:
        console.print(
            f"[yellow]budget ${result.get('max_cost')} reached after ${result.get('spent_usd')}; "
            f"skipped {n_skipped} run(s)[/yellow]"
        )
    for agg in result["aggregate"]:
        console.print(
            f"[green]{agg['model']}[/green]: pass@1={agg['pass_at_1']}±{agg['pass_at_1_stderr']} "
            f"pass@{result['repeat']}={agg['pass_at_k']} over {agg['n_tasks']} tasks "
            f"({agg['n_runs']} runs)  avg_total={agg['avg_total']} cost=${agg['total_cost']}"
        )
    aggregate = result["aggregate"]
    p1 = aggregate[0]["pass_at_1"] if aggregate else None
    return (float(p1) if p1 is not None else None), n_errors + n_skipped


@app.command("effort-sweep")
def effort_sweep(
    suite: str = typer.Option(..., "--suite", help="Suite to sweep, e.g. v1"),
    model: str = typer.Option(..., "--model", "-m", help="provider:model e.g. openai:gpt-5.1"),
    efforts: str = typer.Option(
        ",".join(DEFAULT_SWEEP_EFFORTS),
        "--efforts",
        help="Comma-separated normalized efforts; default low,medium,high",
    ),
    repeat: int = typer.Option(1, "--repeat", help="Run each task N times per effort"),
    output_dir: Path = typer.Option(  # noqa: B008
        Path("./runs"), "--output-dir", "-o", help="Where to write traces and experiment.json"
    ),
    max_steps: int | None = typer.Option(
        None, "--max-steps", help="Cap agent steps (default: per-task metadata / scale tier)"
    ),
    judges: bool = typer.Option(
        True, "--judges/--no-judges", help="Run the human_like LLM judge ensemble"
    ),
    judge_model: str | None = typer.Option(
        None, "--judge-model", help="Model for judges (default: same as --model)"
    ),
    sandbox: str = typer.Option(
        "docker",
        "--sandbox",
        help="Where tools run: local|docker|auto (default docker; agents execute "
        "model-written shell commands, so opt into 'local' deliberately)",
    ),
    image: str | None = typer.Option(
        None,
        "--image",
        help="Docker image for the sandbox (default: per task or vulcanbench/sandbox:base)",
    ),
    network: bool = typer.Option(
        False, "--network/--no-network", help="Allow network in the docker sandbox"
    ),
    max_concurrency: int = typer.Option(
        1, "--max-concurrency", help="Run suite tasks in parallel for each effort"
    ),
    max_cost: float | None = typer.Option(
        None, "--max-cost", help="USD spend cap per effort (same semantics as run --suite)"
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Per-run wall-clock budget in seconds (abort if exceeded)"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, do not launch sandbox"),
) -> None:
    """Run a suite across normalized reasoning-effort levels."""
    try:
        effort_list = parse_efforts(efforts)
    except ValueError as e:
        console.print(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e

    if repeat < 1:
        console.print("[red]error[/red] --repeat must be >= 1")
        raise typer.Exit(code=1)
    if max_concurrency < 1:
        console.print("[red]error[/red] --max-concurrency must be >= 1")
        raise typer.Exit(code=1)
    if sandbox not in {"local", "docker", "auto"}:
        console.print(f"[red]error[/red] --sandbox must be local|docker|auto, got {sandbox!r}")
        raise typer.Exit(code=1)
    if max_cost is not None and not (math.isfinite(max_cost) and max_cost > 0):
        console.print(f"[red]error[/red] --max-cost must be a finite number > 0, got {max_cost}")
        raise typer.Exit(code=1)

    experiment_id = f"experiment-{uuid.uuid4().hex[:8]}"
    console.print(
        f"[cyan]effort sweep[/cyan] {suite} with {model}: {', '.join(effort_list)} "
        f"x{repeat} ({experiment_id})"
    )
    if dry_run:
        console.print("[yellow]dry-run[/yellow] would run one suite invocation per effort")
        raise typer.Exit()

    suites: list[dict[str, Any]] = []
    started_at = datetime.now(UTC)
    try:
        for effort in effort_list:
            console.print(f"[cyan]effort[/cyan] {effort}")
            suites.append(
                run_suite(
                    suite,
                    model,
                    output_dir=output_dir,
                    repeat=repeat,
                    max_concurrency=max_concurrency,
                    max_cost=max_cost,
                    effort=effort,
                    max_steps=max_steps,
                    judges=judges,
                    judge_model=judge_model,
                    sandbox=sandbox,
                    image=image,
                    network=network,
                    timeout_s=timeout,
                    experiment_id=experiment_id,
                )
            )
    except SandboxError as e:
        console.print(f"[red]sandbox error[/red] {e}")
        raise typer.Exit(code=3) from e
    except ProviderError as e:
        console.print(f"[red]provider error[/red] {e}")
        raise typer.Exit(code=2) from e
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]error[/red] {e}")
        raise typer.Exit(code=1) from e

    experiment = {
        "experiment_id": experiment_id,
        "suite": suite,
        "model": model,
        "efforts": effort_list,
        "repeat": repeat,
        "max_concurrency": max_concurrency,
        "max_cost_per_effort": max_cost,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "suites": suites,
    }
    experiment_dir = output_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2), encoding="utf-8"
    )
    console.print(f"[green]wrote[/green] {experiment_dir / 'experiment.json'}")


@app.command()
def leaderboard(  # noqa: PLR0912
    by: str = typer.Option("model", "--by", help="Aggregate view: model|run"),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown|json"),
    task: str | None = typer.Option(None, "--task", help="Filter by task id (run view)"),
    suite: str | None = typer.Option(None, "--suite", help="Filter by suite"),
) -> None:
    """Show the leaderboard: per-model aggregate (default) or per-run (--by run)."""
    rows = scan_leaderboard()
    if suite in SUITE_ALIASES:
        allowed = set(load_suite(suite).task_ids)
        rows = [r for r in rows if r.get("task_id") in allowed]
    elif suite:
        rows = [r for r in rows if r.get("suite") == suite]
    if by == "model":
        data: list[dict] = aggregate_by_model(rows)  # type: ignore[type-arg]
    else:
        data = rows
        if task:
            data = [r for r in data if r.get("task_id") == task]

    if format == "json":
        console.print(json.dumps(data, indent=2))
        return
    if not data:
        console.print(
            "no runs found in ./runs/ (try: vulcanbench run --task hello-world --model mock:synthetic)"
        )
        return

    if by == "model":
        table = Table(title="VulcanBench Leaderboard — by model")
        for col in (
            "Model",
            "Tasks",
            "Runs",
            "pass@1 ± se",
            "pass@k",
            "AvgTotal",
            "Qual",
            "Sec",
            "Human",
            "Cost $",
            "AvgTime",
        ):
            table.add_column(col)
        for a in data:
            cost = "?" if not a.get("cost_known") else f"{a.get('total_cost')}"
            table.add_row(
                a["model"],
                str(a["n_tasks"]),
                str(a["n_runs"]),
                f"{a['pass_at_1']} ± {a['pass_at_1_stderr']}",
                str(a["pass_at_k"]),
                str(a["avg_total"]),
                str(a["avg_quality"]),
                str(a["avg_security"]),
                str(a["avg_human_like"]),
                cost,
                str(a["avg_duration_s"]),
            )
    else:
        table = Table(title="VulcanBench Leaderboard — by run")
        for col in ("Run ID", "Task", "Model", "Total", "Functional", "Cost $", "Time s"):
            table.add_column(col)
        for r in data:
            table.add_row(
                r["run_id"],
                r.get("task_id", "?"),
                r.get("model", "?"),
                str(r.get("total")),
                str(r.get("functional")),
                str(r.get("cost_usd")),
                str(r.get("duration_s")),
            )
    console.print(table)


@app.command()
def report(
    suite: str | None = typer.Option(None, "--suite", help="Limit the report to one suite"),
    format: str = typer.Option("md", "--format", "-f", help="md|json"),
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Write to a file (default: stdout)"
    ),
    runs_dir: Path = typer.Option(Path("./runs"), "--runs-dir", help="Where runs live"),  # noqa: B008
    tasks_root: Path = typer.Option(Path("tasks/v1"), "--tasks-root", help="Task definitions"),  # noqa: B008
) -> None:
    """Generate a shareable results report (Markdown or JSON) from recorded runs."""
    if format not in {"md", "json"}:
        console.print(f"[red]error[/red] --format must be md|json, got {format!r}")
        raise typer.Exit(code=1)
    rep = build_report(runs_dir=runs_dir, tasks_root=tasks_root, suite=suite)
    text = json.dumps(rep, indent=2) if format == "json" else to_markdown(rep)
    if output is not None:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]wrote[/green] {output} ({rep['totals']['n_runs']} runs)")
    else:
        # Report goes to stdout so it can be piped/redirected.
        typer.echo(text)


@app.command()
def calibrate(  # noqa: PLR0912
    suite: str | None = typer.Option(None, "--suite", help="Filter by suite"),
    format: str = typer.Option("table", "--format", "-f", help="table|md|json"),
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Write to a file (default: stdout)"
    ),
    min_attempts: int = typer.Option(5, "--min-attempts", help="Minimum attempts for calibration"),
    min_models: int = typer.Option(
        2, "--min-models", help="Minimum distinct models for calibration"
    ),
    easy_min: float = typer.Option(0.85, "--easy-min", help="Solve rate >= this is easy"),
    medium_min: float = typer.Option(0.40, "--medium-min", help="Solve rate >= this is medium"),
    runs_dir: Path = typer.Option(Path("./runs"), "--runs-dir", help="Where runs live"),  # noqa: B008
    tasks_root: Path = typer.Option(Path("tasks/v1"), "--tasks-root", help="Task definitions"),  # noqa: B008
    include_mock: bool = typer.Option(False, "--include-mock", help="Include mock:model runs"),
) -> None:
    """Calibrate empirical difficulty from recorded runs."""
    if format not in {"table", "md", "json"}:
        console.print(f"[red]error[/red] --format must be table|md|json, got {format!r}")
        raise typer.Exit(code=1)
    if not (math.isfinite(easy_min) and math.isfinite(medium_min)):
        console.print("[red]error[/red] --easy-min and --medium-min must be finite")
        raise typer.Exit(code=1)
    if not (0 <= medium_min < easy_min <= 1):
        console.print(
            f"[red]error[/red] require 0 <= medium_min < easy_min <= 1, "
            f"got medium_min={medium_min} easy_min={easy_min}"
        )
        raise typer.Exit(code=1)
    if min_attempts < 1:
        console.print(f"[red]error[/red] --min-attempts must be >= 1, got {min_attempts}")
        raise typer.Exit(code=1)
    if min_models < 1:
        console.print(f"[red]error[/red] --min-models must be >= 1, got {min_models}")
        raise typer.Exit(code=1)

    rows = scan_leaderboard(runs_dir)
    if suite in SUITE_ALIASES:
        allowed = set(load_suite(suite).task_ids)
        rows = [r for r in rows if r.get("task_id") in allowed]
    elif suite:
        rows = [r for r in rows if r.get("suite") == suite]

    if not rows and not include_mock and format == "table":
        console.print(
            f"no runs found in {runs_dir} "
            "(try: vulcanbench run --task hello-world --model openai:gpt-4o)"
        )
        return

    cal = calibrate_tasks(
        rows,
        tasks_root=tasks_root,
        min_attempts=min_attempts,
        min_models=min_models,
        easy_min=easy_min,
        medium_min=medium_min,
        include_mock=include_mock,
    )

    if format == "json":
        text = json.dumps(cal, indent=2)
    elif format == "md":
        text = calibration_to_markdown(cal)
    else:
        text = ""

    if format == "table":
        table = Table(title="VulcanBench Calibration")
        for col in (
            "Task",
            "Label",
            "Attempts",
            "Models",
            "Solve rate ± se",
            "Empirical",
            "Status",
        ):
            table.add_column(col)
        for e in cal["tasks"]:
            sr = e["solve_rate"]
            se = e["solve_rate_stderr"]
            sr_str = f"{sr:.4f} ± {se:.4f}" if sr is not None else "—"
            label = e["labeled_difficulty"] or "—"
            emp = e["empirical_difficulty"] or "—"
            style = "[bold red]" if e["agreement"] is False else ""
            table.add_row(
                e["task_id"],
                label,
                str(e["n_attempts"]),
                str(e["n_models"]),
                sr_str,
                f"{style}{emp}{'[/]' if style else ''}",
                e["status"],
            )
        console.print(table)
        s = cal["summary"]
        console.print(
            f"{s['n_calibrated']} calibrated, {s['n_disagree']} disagreements, "
            f"{s['n_insufficient']} insufficient data (of {s['n_tasks']} tasks)"
        )
    elif output is not None:
        output.write_text(text, encoding="utf-8")
        console.print(f"[green]wrote[/green] {output}")
    else:
        typer.echo(text)


@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Run ID from ./runs/"),
    output: str = typer.Option("html", "--output", "-o", help="html|json"),
) -> None:
    p = Path("./runs") / run_id
    if output == "html":
        h = p / "replay.html"
        if h.exists():
            console.print(f"replay ready: {h}")
            console.print(h.read_text()[:500] + "... (full in file)")
        else:
            console.print(f"no replay.html for {run_id}")
    else:
        s = p / "summary.json"
        console.print(s.read_text() if s.exists() else "no summary")


@app.command()
def validate_task(
    path: Path = typer.Argument(  # noqa: B008
        ..., exists=True, file_okay=False, dir_okay=True, help="Path to a task dir or tasks/v1"
    ),
    sandbox: str = typer.Option(
        "local",
        "--sandbox",
        help="Where to run verifiers: local|docker (docker matches benchmark runs)",
    ),
    image: str | None = typer.Option(
        None, "--image", help="Docker sandbox image when --sandbox docker"
    ),
) -> None:
    """Validate task(s): schema, gold-solves, fail-to-pass-real, determinism."""
    if sandbox not in {"local", "docker"}:
        console.print(f"[red]error[/red] --sandbox must be local|docker, got {sandbox!r}")
        raise typer.Exit(code=1)
    argv = [str(path)]
    if sandbox != "local":
        argv.extend(["--sandbox", sandbox])
    if image is not None:
        argv.extend(["--image", image])
    raise typer.Exit(code=validate_main(argv))


@app.command()
def list_tasks(
    category: str | None = typer.Option(None, "--category", help="Filter category"),
) -> None:
    tasks_dir = Path("tasks/v1")
    ids = [d.name for d in tasks_dir.iterdir() if d.is_dir()] if tasks_dir.exists() else []
    if category:
        ids = [i for i in ids if category in i]
    for i in ids:
        console.print(i)
    if not ids:
        console.print("no tasks in tasks/v1/ (hello-world synthetic is the demo seed)")


if __name__ == "__main__":
    app()
