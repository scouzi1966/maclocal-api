"""Multi-metric scorer.

Combines the five VulcanBench metrics into a weighted ``total``:

- ``functional``  - from the task verifier (0.0-1.0).
- ``quality``     - linting + complexity + maintainability (``evaluator.quality``).
- ``security``    - static-analysis scanners (``evaluator.security``).
- ``efficiency``  - token usage and step count (lower is better).
- ``human_like``  - LLM judge ensemble (``evaluator.judges``).

Any metric may be ``None`` (e.g. no analyzer for the changed language, or judges
disabled); the weighted total re-normalizes over whichever metrics are present
so the score stays honest. ``evaluator.evaluate.evaluate_run`` orchestrates
computing the metrics and calling :func:`score_run`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Metric weights, re-normalized over whichever metrics are non-None so the total
# is always on a 0-1 scale.
_WEIGHTS = {
    "functional": 0.50,
    "quality": 0.15,
    "security": 0.15,
    "efficiency": 0.10,
    "human_like": 0.10,
}

# Token budget past which efficiency starts decaying toward 0.
_EFFICIENCY_TOKEN_BUDGET = 50_000


def run_verifier(verifier: Path, workspace: Path, timeout: int = 300) -> dict[str, Any]:
    """Run a task verifier in the workspace and parse its JSON stdout.

    The verifier contract: print a single JSON object containing at least
    ``functional`` (0.0-1.0). Anything else it prints to stderr is captured for
    the trace.
    """
    try:
        proc = subprocess.run(
            [sys.executable, str(verifier.resolve())],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=max(1, timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "exit_code": 124,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "scores": {"functional": 0.0, "error": f"verifier timed out after {timeout}s"},
        }
    result: dict[str, Any] = {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    try:
        result["scores"] = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result["scores"] = {"functional": 0.0, "error": "verifier did not emit valid JSON"}
    return result


def efficiency_score(total_tokens: int, steps: int) -> float:
    """Map resource usage to a 0-1 score (1.0 = frugal, →0 = profligate)."""
    if total_tokens <= 0 and steps <= 0:
        return 1.0
    token_factor = max(0.0, 1.0 - total_tokens / _EFFICIENCY_TOKEN_BUDGET)
    step_factor = max(0.0, 1.0 - steps / 100.0)
    return round(0.7 * token_factor + 0.3 * step_factor, 4)


def score_run(
    functional: float,
    total_tokens: int,
    steps: int,
    quality: float | None = None,
    security: float | None = None,
    human_like: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine the five metrics into a score dict with a weighted total.

    ``quality``/``security``/``human_like`` may be ``None`` when not computed;
    they are then excluded from the weighted average.
    """
    metrics: dict[str, Any] = {
        "functional": round(float(functional), 4),
        "quality": round(quality, 4) if quality is not None else None,
        "security": round(security, 4) if security is not None else None,
        "efficiency": efficiency_score(total_tokens, steps),
        "human_like": round(human_like, 4) if human_like is not None else None,
    }
    present = {k: v for k, v in metrics.items() if k in _WEIGHTS and v is not None}
    weight_sum = sum(_WEIGHTS[k] for k in present) or 1.0
    metrics["total"] = round(sum(_WEIGHTS[k] * v for k, v in present.items()) / weight_sum, 4)
    if extra:
        metrics.update(extra)
    return metrics
