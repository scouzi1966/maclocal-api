"""Tests for the evaluate_run orchestrator."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from harness.agent.providers import MockProvider
from harness.evaluator.evaluate import evaluate_run

_PATCH = "diff --git a/hello.py b/hello.py\n+print('hi')\n"


def _ws(tmp_path: Path) -> Path:
    (tmp_path / "hello.py").write_text("print('hi')\n")
    return tmp_path


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed")
def test_evaluate_run_full(tmp_path: Path) -> None:
    scores = evaluate_run(
        functional=1.0,
        total_tokens=100,
        steps=4,
        workspace=_ws(tmp_path),
        patch=_PATCH,
        changed_files=["hello.py"],
        issue="print hi",
        verifier_payload={"scores": {"functional": 1.0}},
        judges_enabled=True,
        judge_provider=MockProvider("synthetic"),
    )
    assert scores["functional"] == 1.0
    assert scores["quality"] is not None
    assert scores["security"] == 1.0
    assert scores["human_like"] == 0.8
    assert 0.0 < scores["total"] <= 1.0
    assert "metric_details" in scores


def test_evaluate_run_judges_disabled(tmp_path: Path) -> None:
    scores = evaluate_run(
        functional=1.0,
        total_tokens=0,
        steps=1,
        workspace=_ws(tmp_path),
        patch=_PATCH,
        changed_files=["hello.py"],
        issue="x",
        verifier_payload={"scores": {"functional": 1.0}},
        judges_enabled=False,
        judge_provider=None,
    )
    assert scores["human_like"] is None
    assert scores["metric_details"]["human_like"]["reason"] == "judges disabled"


def test_evaluate_run_no_judge_provider(tmp_path: Path) -> None:
    scores = evaluate_run(
        functional=1.0,
        total_tokens=0,
        steps=1,
        workspace=_ws(tmp_path),
        patch=_PATCH,
        changed_files=["hello.py"],
        issue="x",
        verifier_payload={"scores": {"functional": 1.0}},
        judges_enabled=True,
        judge_provider=None,
    )
    assert scores["human_like"] is None
    assert "no judge provider" in scores["metric_details"]["human_like"]["reason"]
