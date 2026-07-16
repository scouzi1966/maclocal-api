"""``human_like`` metric: a small ensemble of LLM judges scoring the solution.

Each judge is the same provider/model under a distinct reviewer persona, so the
ensemble captures correctness, readability, and maintainability perspectives.
Judges return JSON ``{"score": 0-100, "rationale": "..."}``; the metric is the
mean of valid votes, normalized to 0-1. Judges degrade gracefully: a provider
error or unparsable reply drops that vote, and zero valid votes yields a
``None`` score with a reason -- never a fabricated number.

Judge token usage is reported in ``details.judge_tokens`` but is intentionally
kept out of the run's efficiency accounting (efficiency measures the agent, not
the evaluation).
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from typing import Any

from harness.agent.providers import LLMProvider, LLMResponse, ProviderError
from harness.evaluator.langs import MetricResult

# Sentinel embedded in every judge prompt so the deterministic MockProvider can
# recognize a judge request and return a fixed score for offline tests/demos.
JUDGE_SENTINEL = "VULCANBENCH_JUDGE"
RemainingSeconds = Callable[[], float | None]

_PERSONAS: list[tuple[str, str]] = [
    (
        "correctness",
        "You are a senior engineer reviewing a code change strictly for correctness "
        "and whether it actually resolves the issue.",
    ),
    (
        "readability",
        "You are a code reviewer focused on readability, naming, and clarity -- "
        "would this pass review as clean, idiomatic code?",
    ),
    (
        "maintainability",
        "You are a maintainer judging long-term maintainability: structure, "
        "minimalism of the change, and absence of foot-guns.",
    ),
]

_INSTRUCTION = (
    "Rate how human-like and high-quality this solution is on a scale of 0-100. "
    f'Respond with ONLY a JSON object: {{"score": <0-100>, "rationale": "<one sentence>"}}. '
    f"Include the token {JUDGE_SENTINEL} nowhere in your answer. ({JUDGE_SENTINEL})"
)


def assess_human_like(
    issue: str,
    patch: str,
    verifier_summary: str,
    provider: LLMProvider,
    personas: list[tuple[str, str]] | None = None,
    remaining_s: RemainingSeconds | None = None,
) -> MetricResult:
    """Run the judge ensemble and return the ``human_like`` metric."""
    personas = personas or _PERSONAS
    if not patch.strip():
        return MetricResult(score=None, details={"reason": "no patch to judge"})

    votes: list[dict[str, Any]] = []
    judge_prompt_tokens = 0
    judge_completion_tokens = 0
    failures: list[str] = []
    for name, persona in personas:
        timeout_s = remaining_s() if remaining_s is not None else None
        if timeout_s is not None and timeout_s <= 0:
            failures.append(f"{name}: run budget exceeded")
            break
        messages = _build_messages(persona, issue, patch, verifier_summary)
        try:
            resp = _complete_with_optional_timeout(provider, messages, [], timeout_s)
        except ProviderError as e:
            failures.append(f"{name}: {e}")
            continue
        if _budget_exhausted(remaining_s):
            failures.append(f"{name}: run budget exceeded after response")
            break
        judge_prompt_tokens += resp.usage.prompt_tokens
        judge_completion_tokens += resp.usage.completion_tokens
        parsed = _extract_score(resp.content)
        if parsed is None:
            failures.append(f"{name}: unparsable response")
            continue
        score, rationale = parsed
        votes.append(
            {"persona": name, "model": provider.spec, "score": score, "rationale": rationale}
        )

    details: dict[str, Any] = {
        "model": provider.spec,
        "votes": votes,
        "judge_tokens": judge_prompt_tokens + judge_completion_tokens,
        "judge_prompt_tokens": judge_prompt_tokens,
        "judge_completion_tokens": judge_completion_tokens,
    }
    if failures:
        details["failures"] = failures
    if not votes:
        details["reason"] = "no valid judge votes"
        return MetricResult(score=None, details=details)

    mean_0_100 = sum(v["score"] for v in votes) / len(votes)
    return MetricResult(score=round(mean_0_100 / 100.0, 4), details=details)


_MAX_PATCH_CHARS = 12_000


def _build_messages(
    persona: str, issue: str, patch: str, verifier_summary: str
) -> list[dict[str, Any]]:
    shown = patch
    if len(patch) > _MAX_PATCH_CHARS:
        omitted = len(patch) - _MAX_PATCH_CHARS
        shown = f"{patch[:_MAX_PATCH_CHARS]}\n...[patch truncated: {omitted} chars omitted]"
    return [
        {"role": "system", "content": persona},
        {
            "role": "user",
            "content": (
                f"# Issue\n{issue}\n\n"
                f"# Test outcome\n{verifier_summary}\n\n"
                f"# Candidate patch\n```diff\n{shown}\n```\n\n{_INSTRUCTION}"
            ),
        },
    ]


def _complete_with_optional_timeout(
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout_s: float | None,
) -> LLMResponse:
    params = inspect.signature(provider.complete).parameters
    accepts_timeout = "timeout_s" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if accepts_timeout:
        return provider.complete(messages, tools, timeout_s=timeout_s)
    return provider.complete(messages, tools)


def _budget_exhausted(remaining_s: RemainingSeconds | None) -> bool:
    remaining = remaining_s() if remaining_s is not None else None
    return remaining is not None and remaining <= 0


def _extract_score(content: str | None) -> tuple[float, str] | None:
    """Pull ``score``/``rationale`` from a model reply, tolerating surrounding prose."""
    if not content:
        return None
    obj = _first_json_object(content)
    if obj is not None and "score" in obj:
        try:
            score = float(obj["score"])
        except (TypeError, ValueError):
            return None
        return _clamp_score(score), str(obj.get("rationale", ""))
    # Fallback: a bare "score": N somewhere in the text.
    m = re.search(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', content)
    if m:
        return _clamp_score(float(m.group(1))), ""
    return None


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced ``{...}`` JSON object in ``text``, if any."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _clamp_score(score: float) -> float:
    return max(0.0, min(100.0, score))
