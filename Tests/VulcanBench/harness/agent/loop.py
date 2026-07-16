"""The VulcanBench agent loop.

``run_agent`` is a real tool-calling ReAct loop: it prompts a model with the
task issue and the standardized tool schemas, executes whatever tools the model
requests through the sandbox executor, records every step to the trace, and then
verifies + scores the result. It works with any provider (mock/OpenAI/Anthropic).

``run_synthetic_hello`` is kept as a thin backward-compatible wrapper that drives
the same loop with the deterministic ``mock`` provider.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.agent.local_executor import LocalToolExecutor
from harness.agent.protocol import RunCommandArgs, ToolCall, ToolProtocol, get_openai_tool_schemas
from harness.agent.providers import LLMProvider, get_provider
from harness.effort import effort_config
from harness.evaluator.evaluate import evaluate_run
from harness.evaluator.scorer import run_verifier, score_run
from harness.persistence import maybe_post_run_summary
from harness.pricing import cost_usd, is_priced
from harness.redaction import sanitize
from harness.sandbox.docker_executor import (
    DockerToolExecutor,
    SandboxError,
    _docker_available,
)
from harness.sandbox.images import resolve_sandbox_image
from harness.task_metadata import (
    measure_repo_path,
    repo_scale,
    resolve_agent_timeout_s,
    resolve_max_steps,
    resolve_verifier_timeout_s,
    system_prompt_for_task,
    task_complexity,
)
from harness.tasks import Task, load_task, prepare_workspace, run_setup, task_hash
from harness.tracer.collector import TraceCollector, generate_replay_html
from harness.verifier import DEFAULT_TIMEOUT, Runner, run_declarative_verifier

SYSTEM_PROMPT = (
    "You are an autonomous software engineering agent. Solve the task described "
    "in the issue by using the provided tools to inspect and edit files and run "
    "tests. Make the smallest correct change. When the task is complete and tests "
    "pass, reply with a final message beginning with 'FINISH:' and stop calling "
    "tools."
)

_VALID_TOOLS = {t["function"]["name"] for t in get_openai_tool_schemas()}
_FINISH_MARKER = "FINISH:"

# Cap on the serialized tool observation fed back to the model; larger outputs
# are cut with an explicit marker so the model knows it saw a partial result.
_MAX_OBSERVATION_CHARS = 8_000


@dataclass
class _RunDeadline:
    started_mono: float
    timeout_s: float | None
    exceeded: bool = False

    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started_mono)

    def remaining_s(self) -> float | None:
        if self.timeout_s is None:
            return None
        return max(0.0, self.timeout_s - self.elapsed_s())

    def is_expired(self) -> bool:
        remaining = self.remaining_s()
        return remaining is not None and remaining <= 0

    def timeout_for(self, default: int) -> int | None:
        remaining = self.remaining_s()
        if remaining is None:
            return default
        if remaining <= 0:
            return None
        return max(1, min(default, math.ceil(remaining)))

    def ensure_time(self, collector: TraceCollector, stage: str, step: int | None = None) -> bool:
        if not self.is_expired():
            return True
        self.record_exceeded(collector, stage, step)
        return False

    def record_exceeded(
        self, collector: TraceCollector, stage: str, step: int | None = None
    ) -> None:
        if self.exceeded:
            return
        self.exceeded = True
        payload: dict[str, Any] = {
            "timeout_s": self.timeout_s,
            "elapsed_s": round(self.elapsed_s(), 3),
            "stage": stage,
        }
        if step is not None:
            payload["step"] = step
        collector.record("budget_exceeded", payload)


def run_agent(
    task_id: str,
    model: str,
    output_dir: Path = Path("./runs"),
    max_steps: int | None = None,
    provider: LLMProvider | None = None,
    tasks_root: Path = Path("tasks/v1"),
    judges: bool = True,
    judge_model: str | None = None,
    sandbox: str = "local",
    image: str | None = None,
    network: bool = False,
    suite: str | None = None,
    suite_id: str | None = None,
    timeout_s: float | None = None,
    effort: str | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Run one evaluation: agent solves ``task_id`` with ``model``.

    ``sandbox`` selects where the agent's tools run: ``local`` (host), ``docker``
    (isolated container), or ``auto`` (docker if available, else local). When
    ``judges`` is set, the ``human_like`` metric is computed by an LLM judge
    ensemble using ``judge_model`` (defaulting to ``model``). ``suite``/
    ``suite_id`` tag the run so the leaderboard can group it. Returns a summary
    dict (also persisted to ``<run_dir>/summary.json``).
    """
    task = load_task(task_id, tasks_root)
    provider = provider or get_provider(model)
    effort_meta = effort_config(provider.name, effort)
    effective_max_steps = resolve_max_steps(task.metadata, max_steps)
    effective_timeout = resolve_agent_timeout_s(task.metadata, timeout_s)

    run_id = f"{task_id}-{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / run_id
    collector = TraceCollector(run_dir, run_id, task_id, model)
    collector.record("task_start", {"task_id": task_id, "metadata": task.metadata})

    workspace = run_dir / "workspace"
    prepare_workspace(task, workspace)
    _git_init(workspace)
    executor = _make_executor(sandbox, workspace, image, network, task, collector)

    # Run setup commands BEFORE the agent's wall-clock budget starts.
    # This is used by Rust tasks (cargo build --tests warm-up) and any task
    # that declares a metadata "setup" key.
    _run_task_setup(task, workspace, executor, collector)

    started_at = datetime.now(UTC)
    started_mono = time.monotonic()
    deadline = _RunDeadline(started_mono=started_mono, timeout_s=effective_timeout)

    tools = get_openai_tool_schemas()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt_for_task(task.metadata, SYSTEM_PROMPT)},
        {"role": "user", "content": f"# Issue\n\n{task.issue}"},
    ]

    try:
        prompt_tokens, completion_tokens, finished = _run_model_loop(
            provider,
            tools,
            messages,
            effective_max_steps,
            collector,
            executor,
            run_id,
            deadline,
            effort_meta.as_summary() if effort_meta else None,
        )
        patch = _git_diff(workspace)
        changed_files = _git_changed_files(workspace)
        # Sanitize the published patch artifact (defense-in-depth; trace is sanitized too).
        (run_dir / "final.patch").write_text(str(sanitize(patch)), encoding="utf-8")
        collector.record("diff", {"patch": patch})

        functional, verifier_payload = _verify_with_budget(
            task, workspace, collector, executor, deadline
        )
        manifest = _collect_manifest_with_budget(
            model, judge_model, network, executor, collector, deadline, task, workspace
        )
    finally:
        # Quality/security/judges run host-side over the mounted workspace and
        # don't need the container, so tear it down now.
        close = getattr(executor, "close", None)
        if callable(close):
            close()

    total_tokens = prompt_tokens + completion_tokens
    judge_provider = _build_judge_provider(judges, judge_model, model, provider, collector)
    scores = _evaluate_with_budget(
        functional=functional,
        total_tokens=total_tokens,
        steps=collector.step,
        workspace=workspace,
        patch=patch,
        changed_files=changed_files,
        issue=task.issue,
        verifier_payload=verifier_payload,
        judges=judges,
        judge_provider=judge_provider,
        collector=collector,
        deadline=deadline,
    )
    collector.record("metric_computed", scores)

    cost = _compute_cost(model, prompt_tokens, completion_tokens, judge_model, scores)
    duration_s = round(time.monotonic() - started_mono, 3)
    summary = collector.finalize(
        scores,
        {
            "steps": collector.step,
            "total_tokens": total_tokens,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "cost_usd": cost["total"],
            "cost_detail": cost,
            "duration_s": duration_s,
            "started_at": started_at.isoformat(),
            "finished": finished,
            "manifest": manifest,
            "task_hash": task_hash(task),
            "suite": suite,
            "suite_id": suite_id,
            **({"effort": effort_meta.as_summary()} if effort_meta else {}),
            **({"experiment_id": experiment_id} if experiment_id else {}),
            "verifier": verifier_payload,
            "replay_command": f"vulcanbench replay {run_id}",
        },
    )
    generate_replay_html(collector.trace_path, run_dir / "replay.html")
    maybe_post_run_summary(summary)
    return {"run_id": run_id, "summary": summary, "replay": str(run_dir / "replay.html")}


def run_synthetic_hello(
    task_dir: Path,
    output_dir: Path = Path("./runs"),
    max_steps: int = 20,
) -> dict[str, Any]:
    """Backward-compatible wrapper: run the hello-world demo via the mock provider."""
    tasks_root = task_dir.parent
    return run_agent(
        task_id=task_dir.name,
        model="mock:synthetic",
        output_dir=output_dir,
        max_steps=max_steps,
        tasks_root=tasks_root,
    )


def _compute_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    judge_model: str | None,
    scores: dict[str, Any],
) -> dict[str, Any]:
    """Agent + judge cost in USD. ``total`` is ``None`` when the model is unpriced."""
    agent = cost_usd(model, prompt_tokens, completion_tokens)
    human = scores.get("metric_details", {}).get("human_like", {})
    jp = int(human.get("judge_prompt_tokens", 0))
    jc = int(human.get("judge_completion_tokens", 0))
    judges = cost_usd(judge_model or model, jp, jc) if (jp or jc) else 0.0
    total = round(agent + judges, 6) if (agent is not None and judges is not None) else None
    return {"agent": agent, "judges": judges, "total": total, "model_priced": is_priced(model)}


def _verify_with_budget(
    task: Task,
    workspace: Path,
    collector: TraceCollector,
    executor: ToolProtocol,
    deadline: _RunDeadline,
) -> tuple[float, dict[str, Any]]:
    # Verify through the executor so tests run wherever the agent did: inside
    # the Docker container under --sandbox docker, on the host under local.
    if not deadline.ensure_time(collector, "verify"):
        return 0.0, _budget_exceeded_verifier_payload()
    verifier_budget = resolve_verifier_timeout_s(task.metadata, DEFAULT_TIMEOUT)
    functional, verifier_payload = _verify(
        task,
        workspace,
        collector,
        executor,
        timeout=deadline.timeout_for(verifier_budget) or 1,
    )
    if deadline.is_expired():
        deadline.record_exceeded(collector, "verify")
    return functional, verifier_payload


def _collect_manifest_with_budget(
    model: str,
    judge_model: str | None,
    network: bool,
    executor: ToolProtocol,
    collector: TraceCollector,
    deadline: _RunDeadline,
    task: Task,
    workspace: Path,
) -> dict[str, Any]:
    if not deadline.ensure_time(collector, "manifest"):
        return _minimal_manifest(model, judge_model, network, executor) | _task_workspace_stats(
            task, workspace
        )
    manifest = _collect_manifest(
        model, judge_model, network, executor, remaining_s=deadline.remaining_s
    )
    manifest |= _task_workspace_stats(task, workspace)
    if deadline.is_expired():
        deadline.record_exceeded(collector, "manifest")
    return manifest


def _evaluate_with_budget(
    *,
    functional: float,
    total_tokens: int,
    steps: int,
    workspace: Path,
    patch: str,
    changed_files: list[str],
    issue: str,
    verifier_payload: dict[str, Any],
    judges: bool,
    judge_provider: LLMProvider | None,
    collector: TraceCollector,
    deadline: _RunDeadline,
) -> dict[str, Any]:
    if not deadline.ensure_time(collector, "evaluate"):
        return _budget_exceeded_scores(functional, total_tokens, steps)
    scores = evaluate_run(
        functional=functional,
        total_tokens=total_tokens,
        steps=steps,
        workspace=workspace,
        patch=patch,
        changed_files=changed_files,
        issue=issue,
        verifier_payload=verifier_payload,
        judges_enabled=judges,
        judge_provider=judge_provider,
        collector=collector,
        remaining_s=deadline.remaining_s,
        budget_exceeded=lambda stage: deadline.record_exceeded(collector, stage),
    )
    if deadline.exceeded:
        scores["budget_exceeded"] = True
    return scores


def _budget_exceeded_verifier_payload() -> dict[str, Any]:
    return {
        "scores": {"functional": 0.0, "error": "run budget exceeded before verification"},
        "budget_exceeded": True,
    }


def _budget_exceeded_scores(functional: float, total_tokens: int, steps: int) -> dict[str, Any]:
    return score_run(
        functional=functional,
        total_tokens=total_tokens,
        steps=steps,
        quality=None,
        security=None,
        human_like=None,
        extra={
            "budget_exceeded": True,
            "metric_details": {
                "quality": {"reason": "run budget exceeded"},
                "security": {"reason": "run budget exceeded"},
                "human_like": {"reason": "run budget exceeded"},
            },
        },
    )


def _run_model_loop(
    provider: LLMProvider,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    max_steps: int,
    collector: TraceCollector,
    executor: ToolProtocol,
    run_id: str,
    deadline: _RunDeadline,
    effort: dict[str, Any] | None = None,
) -> tuple[int, int, bool]:
    prompt_tokens = 0
    completion_tokens = 0
    finished = False

    for step in range(1, max_steps + 1):
        if not deadline.ensure_time(collector, "llm_request", step):
            break
        request_data: dict[str, Any] = {"step": step, "messages": len(messages)}
        if effort is not None:
            request_data["effort"] = effort
        collector.record("llm_request", request_data)
        try:
            resp = _complete_provider(provider, messages, tools, deadline.remaining_s(), effort)
        except Exception:
            if deadline.is_expired():
                deadline.record_exceeded(collector, "llm_request", step)
                break
            raise
        prompt_tokens += resp.usage.prompt_tokens
        completion_tokens += resp.usage.completion_tokens
        collector.record(
            "llm_response",
            {
                "content": resp.content,
                "tool_calls": [tc.model_dump() for tc in resp.tool_calls],
                "usage": resp.usage.model_dump(),
            },
        )
        if not deadline.ensure_time(collector, "llm_response", step):
            break

        messages.append(_assistant_message(resp))

        if not resp.wants_tools:
            finished = bool(resp.content and _FINISH_MARKER in resp.content)
            break

        for tc in resp.tool_calls:
            if not deadline.ensure_time(collector, f"tool:{tc.name}", step):
                break
            observation = _execute_tool(executor, tc, run_id, step, deadline.remaining_s())
            collector.record("tool_call", {"tool": tc.name, "args": tc.arguments, "id": tc.id})
            collector.record(
                "tool_observation",
                {
                    "tool": tc.name,
                    "result": observation.get("result"),
                    "error": observation.get("error"),
                },
            )
            content = json.dumps(observation, default=str)
            if len(content) > _MAX_OBSERVATION_CHARS:
                omitted = len(content) - _MAX_OBSERVATION_CHARS
                content = (
                    f"{content[:_MAX_OBSERVATION_CHARS]}"
                    f"...[output truncated: {omitted} chars omitted]"
                )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )
            if not deadline.ensure_time(collector, f"tool:{tc.name}", step):
                break

    return prompt_tokens, completion_tokens, finished


_MANIFEST_TOOLS = ("git", "ruff", "bandit", "radon", "go", "node")


def _first_output_line(output: str) -> str | None:
    lines = output.strip().splitlines()
    return lines[0].strip() if lines else None


def _host_command_first_line(args: list[str], timeout: int = 10) -> str | None:
    """Run a trusted local version probe and return its first output line."""
    if shutil.which(args[0]) is None:
        return None
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return _first_output_line(proc.stdout or proc.stderr)


def _executor_command_first_line(executor: ToolProtocol, cmd: str, timeout: int = 10) -> str | None:
    """Run a trusted probe in the execution environment and return one line."""
    try:
        res = executor.run_command(RunCommandArgs(cmd=cmd, timeout=timeout))
    except Exception:
        return None
    if int(res.get("exit_code", 1)) != 0:
        return None
    return _first_output_line(str(res.get("stdout") or res.get("stderr") or ""))


def _tool_version(
    cmd: str, executor: ToolProtocol, sandbox_mode: str, timeout: int = 10
) -> str | None:
    """First line of the tool's version output in the run environment."""
    version_cmd = f"{cmd} version" if cmd == "go" else f"{cmd} --version"
    if sandbox_mode == "docker":
        return _executor_command_first_line(
            executor,
            f"command -v {cmd} >/dev/null 2>&1 && {version_cmd}",
            timeout=timeout,
        )
    # Host subprocess keeps local manifests independent of workspace shell state.
    args = [cmd, "version"] if cmd == "go" else [cmd, "--version"]
    return _host_command_first_line(args, timeout=timeout)


def _runtime_python(executor: ToolProtocol, sandbox_mode: str, timeout: int = 10) -> str:
    if sandbox_mode == "docker":
        return (
            _executor_command_first_line(executor, "python --version", timeout=timeout)
            or _executor_command_first_line(executor, "python3 --version", timeout=timeout)
            or platform.python_version()
        )
    return platform.python_version()


def _runtime_platform(executor: ToolProtocol, sandbox_mode: str, timeout: int = 10) -> str:
    if sandbox_mode == "docker":
        return _executor_command_first_line(executor, "uname -srm", timeout=timeout) or sys.platform
    return sys.platform


def _task_workspace_stats(task: Task, workspace: Path) -> dict[str, Any]:
    stats = measure_repo_path(workspace)
    return {
        "task": {
            "repo_scale": repo_scale(task.metadata),
            "task_complexity": task_complexity(task.metadata),
            "languages": task.metadata.get("languages", []),
            "difficulty": task.metadata.get("difficulty"),
            "file_count": stats["file_count"],
            "loc": stats["loc"],
        }
    }


def _collect_manifest(
    model: str,
    judge_model: str | None,
    network: bool,
    executor: ToolProtocol,
    remaining_s: Callable[[], float | None] | None = None,
) -> dict[str, Any]:
    """Record the environment a run executed in, for audit + reproducibility."""
    sandbox_mode = "docker" if isinstance(executor, DockerToolExecutor) else "local"
    runtime_timeout = _remaining_timeout(10, remaining_s)
    if runtime_timeout is None:
        return _minimal_manifest(model, judge_model, network, executor) | {"budget_exceeded": True}
    tools: dict[str, str | None] = {}
    for tool in _MANIFEST_TOOLS:
        timeout = _remaining_timeout(10, remaining_s)
        tools[tool] = (
            None if timeout is None else _tool_version(tool, executor, sandbox_mode, timeout)
        )
    return {
        "model": model,
        "judge_model": judge_model or model,
        "runtime": {
            "python": _runtime_python(executor, sandbox_mode, runtime_timeout),
            "platform": _runtime_platform(executor, sandbox_mode, runtime_timeout),
            "host_python": platform.python_version(),
            "host_platform": sys.platform,
        },
        "sandbox": {
            "mode": sandbox_mode,
            "image": getattr(executor, "image", None),
            "network": network,
        },
        "tools": tools,
    }


def _minimal_manifest(
    model: str, judge_model: str | None, network: bool, executor: ToolProtocol
) -> dict[str, Any]:
    sandbox_mode = "docker" if isinstance(executor, DockerToolExecutor) else "local"
    return {
        "model": model,
        "judge_model": judge_model or model,
        "runtime": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "host_python": platform.python_version(),
            "host_platform": sys.platform,
        },
        "sandbox": {
            "mode": sandbox_mode,
            "image": getattr(executor, "image", None),
            "network": network,
        },
        "tools": {tool: None for tool in _MANIFEST_TOOLS},
    }


def _remaining_timeout(default: int, remaining_s: Callable[[], float | None] | None) -> int | None:
    if remaining_s is None:
        return default
    remaining = remaining_s()
    if remaining is None:
        return default
    if remaining <= 0:
        return None
    return max(1, min(default, math.ceil(remaining)))


def _make_executor(
    sandbox: str,
    workspace: Path,
    image: str | None,
    network: bool,
    task: Task,
    collector: TraceCollector,
) -> ToolProtocol:
    """Build the tool executor for the requested sandbox mode.

    ``docker`` errors out if the daemon is unavailable (never silently runs on
    the host). ``auto`` uses Docker when available; falling back to local (host)
    execution requires the explicit ``VULCANBENCH_ALLOW_HOST_EXEC=1`` opt-in,
    and is always recorded with a loud warning.
    """
    resolved_image = resolve_sandbox_image(task, image)
    if sandbox == "local":
        collector.record("sandbox", {"mode": "local"})
        return LocalToolExecutor(workspace)
    if sandbox == "docker":
        collector.record("sandbox", {"mode": "docker", "image": resolved_image, "network": network})
        return DockerToolExecutor(workspace, image=resolved_image, network=network)
    if sandbox == "auto":
        if _docker_available():
            collector.record(
                "sandbox", {"mode": "docker", "image": resolved_image, "network": network}
            )
            return DockerToolExecutor(workspace, image=resolved_image, network=network)
        if os.environ.get("VULCANBENCH_ALLOW_HOST_EXEC") != "1":
            raise SandboxError(
                "sandbox 'auto': Docker daemon unavailable. Refusing to run "
                "model-written commands on the host. Start Docker, pass "
                "--sandbox local, or set VULCANBENCH_ALLOW_HOST_EXEC=1 to allow "
                "the fallback."
            )
        collector.record(
            "sandbox_fallback",
            {"requested": "auto", "using": "local", "reason": "docker daemon unavailable"},
        )
        print(
            "[vulcanbench] WARNING: Docker daemon unavailable; falling back to local (host) execution.",
            file=sys.stderr,
        )
        return LocalToolExecutor(workspace)
    raise ValueError(f"unknown sandbox mode {sandbox!r}; expected local|docker|auto")


def _assistant_message(resp: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": resp.content or ""}
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id or f"call_{i}",
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for i, tc in enumerate(resp.tool_calls)
        ]
    return msg


def _complete_provider(
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout_s: float | None,
    effort: dict[str, Any] | None = None,
) -> Any:
    params = inspect.signature(provider.complete).parameters
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepts_timeout = "timeout_s" in params or accepts_kwargs
    accepts_effort = "effort" in params or accepts_kwargs
    kwargs: dict[str, Any] = {}
    if accepts_timeout:
        kwargs["timeout_s"] = timeout_s
    if effort is not None and effort.get("supported") and accepts_effort:
        kwargs["effort"] = effort.get("provider_value")
    if kwargs:
        return provider.complete(messages, tools, **kwargs)
    return provider.complete(messages, tools)


def _execute_tool(
    executor: ToolProtocol, tc: Any, run_id: str, step: int, timeout_s: float | None = None
) -> dict[str, Any]:
    if tc.name not in _VALID_TOOLS:
        observation = {"result": None, "error": f"unknown tool {tc.name!r}"}
    elif _tool_budget_exhausted(timeout_s):
        observation = {"result": None, "error": "run budget exceeded"}
    elif tc.name == "run_command":
        observation = _execute_run_command_tool(
            executor, tc, run_id, step, timeout=_tool_timeout(timeout_s)
        )
    elif tc.name == "search_code":
        observation = _execute_search_code_tool(
            executor, tc, run_id, step, _tool_timeout(timeout_s)
        )
    elif tc.name == "run_tests":
        observation = _run_command_tool(
            executor,
            RunCommandArgs(
                cmd="python -m pytest -q --tb=no || true", timeout=_tool_timeout(timeout_s)
            ),
        )
    elif tc.name == "run_lint":
        observation = _run_command_tool(
            executor, RunCommandArgs(cmd="ruff check . || true", timeout=_tool_timeout(timeout_s))
        )
    elif tc.name == "security_scan":
        observation = _security_scan_tool(executor, timeout_s)
    else:
        call = ToolCall(tool=tc.name, args=tc.arguments, step=step, run_id=run_id)
        obs = executor.execute(call)
        observation = {"result": obs.result, "error": obs.error}
    return observation


def _execute_run_command_tool(
    executor: ToolProtocol, tc: Any, run_id: str, step: int, timeout: int | None
) -> dict[str, Any]:
    args = dict(tc.arguments)
    if timeout is not None:
        requested = args.get("timeout")
        try:
            requested_timeout = timeout if requested is None else int(requested)
        except (TypeError, ValueError):
            requested_timeout = timeout
        args["timeout"] = min(requested_timeout, timeout)
    call = ToolCall(tool=tc.name, args=args, step=step, run_id=run_id)
    obs = executor.execute(call)
    return {"result": obs.result, "error": obs.error}


def _execute_search_code_tool(
    executor: ToolProtocol, tc: Any, run_id: str, step: int, timeout: int | None
) -> dict[str, Any]:
    args = dict(tc.arguments)
    if timeout is not None:
        requested = args.get("timeout")
        try:
            requested_timeout = timeout if requested is None else int(requested)
        except (TypeError, ValueError):
            requested_timeout = timeout
        args["timeout"] = min(requested_timeout, timeout)
    call = ToolCall(tool=tc.name, args=args, step=step, run_id=run_id)
    obs = executor.execute(call)
    return {"result": obs.result, "error": obs.error}


def _security_scan_tool(executor: ToolProtocol, timeout_s: float | None) -> dict[str, Any]:
    try:
        return {"result": executor.security_scan(timeout_s=timeout_s), "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _tool_timeout(timeout_s: float | None) -> int | None:
    if timeout_s is None:
        return None
    return max(1, min(120, math.ceil(timeout_s)))


def _tool_budget_exhausted(timeout_s: float | None) -> bool:
    return timeout_s is not None and timeout_s <= 0


def _run_command_tool(executor: ToolProtocol, args: RunCommandArgs) -> dict[str, Any]:
    try:
        return {"result": executor.run_command(args), "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _run_task_setup(
    task: Task,
    workspace: Path,
    executor: ToolProtocol,
    collector: TraceCollector,
) -> None:
    """Run the task's optional setup commands before the agent loop starts.

    Uses the executor's runner so setup runs in-sandbox under ``--sandbox
    docker``. A non-zero setup exit raises ``RuntimeError`` so the calling
    suite/loop can handle it as a run error without crashing the process.
    """
    if not task.setup_commands:
        return
    runner = _executor_runner(executor)
    run_setup(task, workspace, runner=runner, collector=collector)


def _executor_runner(executor: ToolProtocol) -> Runner:
    """A verifier runner that executes test commands through the tool executor.

    For the local executor this runs on the host; for the Docker executor it
    ``exec``s inside the container, so verification happens in the sandbox.
    """

    def run(cmd: str, _workspace: Path, timeout: int) -> int:
        try:
            res = executor.run_command(RunCommandArgs(cmd=cmd, timeout=timeout))
        except Exception:
            return 124
        return int(res.get("exit_code", 1))

    return run


def _verify(
    task: Task,
    workspace: Path,
    collector: TraceCollector,
    executor: ToolProtocol,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[float, dict[str, Any]]:
    if task.tests_spec is not None:
        payload = run_declarative_verifier(
            task, workspace, runner=_executor_runner(executor), timeout=timeout
        )
    elif task.verifier is not None:
        payload = run_verifier(task.verifier, workspace, timeout=timeout)
    else:
        return 0.0, {"error": "no verifier for task"}
    collector.record("test_result", payload)
    functional = float(payload.get("scores", {}).get("functional", 0.0))
    return functional, payload


_WORKSPACE_GITIGNORE = ".coverage\n__pycache__/\n.pytest_cache/\n.ruff_cache/\n*.pyc\n"


def _git_init(workspace: Path) -> None:
    """Initialize a throwaway git repo so we can diff the agent's changes.

    A default ``.gitignore`` keeps test/build artifacts (e.g. ``.coverage``) out
    of the captured patch so ``final.patch`` reflects only the agent's edits.
    """
    (workspace / ".gitignore").write_text(_WORKSPACE_GITIGNORE, encoding="utf-8")
    env = {"GIT_AUTHOR_NAME": "vulcanbench", "GIT_AUTHOR_EMAIL": "bot@vulcanbench"}
    env |= {"GIT_COMMITTER_NAME": "vulcanbench", "GIT_COMMITTER_EMAIL": "bot@vulcanbench"}
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["commit", "-q", "--allow-empty", "-m", "base"],
    ):
        subprocess.run(
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            env=_git_env(env),
        )


def _git_diff(workspace: Path) -> str:
    subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, text=True, check=False)
    proc = subprocess.run(
        ["git", "diff", "--cached"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def _git_changed_files(workspace: Path) -> list[str]:
    """List files the agent changed (staged vs the base commit), relative paths."""
    subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, text=True, check=False)
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    return [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]


# Env var each provider needs before judging is worthwhile (mock needs none).
_JUDGE_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "zai": "ZAI_API_KEY",
}


def _build_judge_provider(
    judges: bool,
    judge_model: str | None,
    model: str,
    run_provider: LLMProvider | None,
    collector: TraceCollector,
) -> LLMProvider | None:
    """Resolve the provider for the judge ensemble, or ``None`` if unusable.

    Reuses the run's provider instance when judging with the same spec (keeps the
    mock deterministic and avoids reconstruction). Returns ``None`` up front when
    the spec is invalid or the required API key is absent, so we don't fire
    retry-wrapped calls that are guaranteed to fail.
    """
    if not judges:
        return None
    spec = judge_model or model
    if run_provider is not None and spec == model:
        provider: LLMProvider = run_provider
    else:
        try:
            provider = get_provider(spec)
        except ValueError:
            return None
    need = _JUDGE_KEY_ENV.get(provider.name)
    if need is not None and not os.environ.get(need):
        return None
    return provider


def _git_env(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(extra)
    return env
