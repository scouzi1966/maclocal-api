"""Tests for task scale metadata helpers."""

from __future__ import annotations

from harness.task_metadata import (
    infer_task_complexity_from_gold_patch,
    repo_scale,
    resolve_agent_timeout_s,
    resolve_max_steps,
    resolve_verifier_timeout_s,
    task_complexity,
    validate_scale_fields,
)


def test_repo_scale_default() -> None:
    assert repo_scale({}) == "micro"


def test_task_complexity_default_and_valid_values() -> None:
    assert task_complexity({}) == "localized"
    assert task_complexity({"task_complexity": "system"}) == "system"


def test_infer_task_complexity_from_gold_patch() -> None:
    assert (
        infer_task_complexity_from_gold_patch(
            "diff --git a/a.py b/a.py\n"
            "diff --git a/b.py b/b.py\n"
            "diff --git a/README.md b/README.md\n"
        )
        == "multi_file"
    )
    assert (
        infer_task_complexity_from_gold_patch(
            "diff --git a/a.py b/a.py\ndiff --git a/b.ts b/b.ts\ndiff --git a/c.go b/c.go\n"
        )
        == "system"
    )


def test_resolve_max_steps_from_hints() -> None:
    meta = {"repo_scale": "large", "agent_hints": {"suggested_max_steps": 120}}
    assert resolve_max_steps(meta) == 120


def test_resolve_max_steps_cli_caps_hints() -> None:
    meta = {"agent_hints": {"suggested_max_steps": 100}}
    assert resolve_max_steps(meta, cli_max_steps=30) == 30
    assert resolve_max_steps(meta) == 100


def test_resolve_agent_timeout_ignores_test_timeout_s() -> None:
    meta = {"repo_scale": "medium", "test_timeout_s": 120}
    assert resolve_agent_timeout_s(meta) == 1200.0
    assert resolve_verifier_timeout_s(meta) == 120


def test_resolve_agent_timeout_cli_cap() -> None:
    meta = {"repo_scale": "medium", "test_timeout_s": 120}
    assert resolve_agent_timeout_s(meta, cli_timeout=100.0) == 100.0


def test_resolve_verifier_timeout_default() -> None:
    assert resolve_verifier_timeout_s({}) == 120


def test_validate_scale_oss_requires_base_commit() -> None:
    reasons = validate_scale_fields(
        __import__("pathlib").Path("."),
        {"source": "oss", "upstream": {"url": "https://example.com"}},
    )
    assert any("base_commit" in r for r in reasons)


def test_validate_scale_rejects_placeholder_commit() -> None:
    reasons = validate_scale_fields(
        __import__("pathlib").Path("."),
        {
            "source": "oss",
            "base_commit": "0000000000000000000000000000000000000001",
            "upstream": {"url": "https://example.com"},
        },
    )
    assert any("placeholder" in r for r in reasons)


def test_validate_scale_rejects_bad_task_complexity() -> None:
    reasons = validate_scale_fields(
        __import__("pathlib").Path("."),
        {"task_complexity": "giant"},
    )
    assert any("task_complexity" in r for r in reasons)
