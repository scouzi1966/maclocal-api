"""Suite alias loading (v1-micro / v1-large / v1-rust)."""

from __future__ import annotations

from pathlib import Path

from harness.suite import load_suite


def test_v1_large_suite_non_empty() -> None:
    suite = load_suite("v1-large", Path("tasks"))
    assert suite.task_ids
    assert all(tid.startswith("oss-") or tid.startswith("py-") for tid in suite.task_ids[:3])


def test_v1_micro_excludes_large_scale_tasks() -> None:
    micro = set(load_suite("v1-micro", Path("tasks")).task_ids)
    large = set(load_suite("v1-large", Path("tasks")).task_ids)
    assert not micro & large


def test_v1_suite_excludes_hello_world() -> None:
    suite = load_suite("v1", Path("tasks"))
    assert "hello-world" not in suite.task_ids
    assert len(suite.task_ids) >= 50


def test_v1_rust_suite_contains_rust_tasks() -> None:
    suite = load_suite("v1-rust", Path("tasks"))
    assert suite.task_ids
    assert "rs-borrow-split" in suite.task_ids
    assert "rs-feature-gate" in suite.task_ids
    # Should NOT contain non-Rust tasks.
    for tid in suite.task_ids:
        assert tid.startswith("rs-"), f"v1-rust suite contains non-Rust task: {tid}"
