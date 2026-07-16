"""Declarative task verifier.

A task declares its tests in ``metadata.tests`` as two lists of
``{"name", "cmd"}`` entries; each ``cmd`` runs in the workspace and **exit code
0 means the test passed**:

- ``fail_to_pass``: must fail on the starting repo and pass after the fix — the
  real signal. ``functional`` is the fraction of these that pass.
- ``pass_to_pass``: must keep passing (regression guard). If any of these fail,
  ``functional`` is gated to 0.0 regardless of the fail-to-pass results.

This runs at scoring time, after copying the task's hidden ``tests/`` into the
workspace (so the agent never saw them while solving).

Test commands are dispatched through a ``Runner`` — a callable that takes
``(cmd, workspace, timeout)`` and returns an exit code. The default runs on the
host; the agent loop passes a runner that ``exec``s inside the Docker sandbox so
verification happens in the same isolated, reproducible environment as the run.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from harness.tasks import Task, install_hidden_tests

DEFAULT_TIMEOUT = 120

# (cmd, workspace, timeout) -> exit code (0 == pass).
Runner = Callable[[str, Path, int], int]


def host_runner(cmd: str, workspace: Path, timeout: int) -> int:
    """Run a test command on the host, in the workspace; returns its exit code."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124  # conventional timeout exit code
    return proc.returncode


def _run_group(
    entries: list[dict[str, Any]], workspace: Path, timeout: int, runner: Runner
) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for i, entry in enumerate(entries):
        name = str(entry.get("name") or f"test_{i}")
        cmd = entry.get("cmd")
        results[name] = bool(cmd) and runner(str(cmd), workspace, timeout) == 0
    return results


def run_declarative_verifier(
    task: Task,
    workspace: Path,
    runner: Runner | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run a task's declarative tests and return a scores payload.

    ``runner`` defaults to running tests on the host; pass a sandbox runner to
    verify inside a container. Either way, hidden tests are copied into the
    workspace first (the workspace is the container's bind mount under Docker).
    """
    runner = runner or host_runner
    install_hidden_tests(task, workspace)
    spec = task.tests_spec or {}
    fail_to_pass = list(spec.get("fail_to_pass") or [])
    pass_to_pass = list(spec.get("pass_to_pass") or [])

    if not fail_to_pass:
        return {"scores": {"functional": 0.0, "error": "no fail_to_pass tests declared"}}

    p2p_results = _run_group(pass_to_pass, workspace, timeout, runner)
    f2p_results = _run_group(fail_to_pass, workspace, timeout, runner)

    p2p_ok = all(p2p_results.values())
    f2p_passing = sum(1 for ok in f2p_results.values() if ok)
    functional = 0.0 if not p2p_ok else round(f2p_passing / len(f2p_results), 4)

    details = []
    if not p2p_ok:
        broke = [n for n, ok in p2p_results.items() if not ok]
        details.append(f"regression: pass_to_pass failing: {broke}")
    details.append(f"fail_to_pass {f2p_passing}/{len(f2p_results)} passing")

    return {
        "scores": {"functional": functional},
        "fail_to_pass": f2p_results,
        "pass_to_pass": p2p_results,
        "pass_to_pass_ok": p2p_ok,
        "details": details,
    }
