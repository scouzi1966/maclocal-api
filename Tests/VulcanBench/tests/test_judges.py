"""Tests for the human_like judge ensemble."""

from __future__ import annotations

from typing import Any

from harness.agent.providers import LLMProvider, LLMResponse, MockProvider, TokenUsage
from harness.evaluator.judges import _extract_score, _first_json_object, assess_human_like


def test_extract_score_clean_json() -> None:
    assert _extract_score('{"score": 90, "rationale": "good"}') == (90.0, "good")


def test_extract_score_with_surrounding_prose() -> None:
    text = 'Sure! Here is my verdict:\n```json\n{"score": 75, "rationale": "ok"}\n```\nThanks.'
    assert _extract_score(text) == (75.0, "ok")


def test_extract_score_regex_fallback() -> None:
    score, rationale = _extract_score("I would rate score: 60 out of 100")  # type: ignore[misc]
    assert score == 60.0
    assert rationale == ""


def test_extract_score_clamps_and_handles_garbage() -> None:
    assert _extract_score('{"score": 150}')[0] == 100.0
    assert _extract_score("no number here") is None
    assert _extract_score(None) is None


def test_first_json_object_ignores_non_objects() -> None:
    assert _first_json_object("[1,2,3]") is None
    assert _first_json_object('prefix {"a": 1} suffix') == {"a": 1}


def test_assess_human_like_with_mock_is_deterministic() -> None:
    result = assess_human_like(
        issue="do x",
        patch="diff --git a/x b/x\n+x",
        verifier_summary="passed",
        provider=MockProvider("synthetic"),
    )
    assert result.score == 0.8  # mock judge returns 80/100 for all 3 personas
    assert len(result.details["votes"]) == 3
    assert result.details["judge_tokens"] > 0


def test_assess_human_like_no_patch() -> None:
    result = assess_human_like("i", "", "passed", MockProvider("synthetic"))
    assert result.score is None
    assert result.details["reason"] == "no patch to judge"


def test_assess_human_like_averages_varied_votes() -> None:
    class VaryingProvider(LLMProvider):
        def __init__(self) -> None:
            super().__init__("varying")
            self.calls = 0

        @property
        def name(self) -> str:
            return "mock"

        def complete(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> LLMResponse:
            scores = [60, 90, 30]
            s = scores[self.calls % 3]
            self.calls += 1
            return LLMResponse(content=f'{{"score": {s}}}', usage=TokenUsage(prompt_tokens=1))

    result = assess_human_like("i", "patch", "passed", VaryingProvider())
    assert result.score == round((60 + 90 + 30) / 3 / 100, 4)


def test_assess_human_like_drops_unparsable_votes() -> None:
    class FlakyProvider(LLMProvider):
        def __init__(self) -> None:
            super().__init__("flaky")
            self.calls = 0

        @property
        def name(self) -> str:
            return "mock"

        def complete(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
        ) -> LLMResponse:
            self.calls += 1
            content = '{"score": 70}' if self.calls == 1 else "sorry, I cannot comply"
            return LLMResponse(content=content)

    result = assess_human_like("i", "patch", "passed", FlakyProvider())
    assert result.score == 0.7  # only the one valid vote counts
    assert len(result.details["votes"]) == 1
    assert len(result.details["failures"]) == 2


def test_assess_human_like_drops_late_vote_after_budget_expires() -> None:
    ticks = iter([1.0, 0.0])

    class LateProvider(LLMProvider):
        @property
        def name(self) -> str:
            return "mock"

        def complete(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            timeout_s: float | None = None,
        ) -> LLMResponse:
            return LLMResponse(content='{"score": 90}', usage=TokenUsage(prompt_tokens=1))

    result = assess_human_like(
        "i",
        "patch",
        "passed",
        LateProvider("late"),
        personas=[("correctness", "review")],
        remaining_s=lambda: next(ticks, 0.0),
    )

    assert result.score is None
    assert result.details["votes"] == []
    assert "run budget exceeded after response" in result.details["failures"][0]
