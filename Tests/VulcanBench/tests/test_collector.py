"""The tracer must never write secrets or unbounded fields to disk."""

from __future__ import annotations

import json
from pathlib import Path

from harness.redaction import MAX_FIELD_CHARS
from harness.tracer.collector import TraceCollector


def test_record_redacts_and_caps(tmp_path: Path) -> None:
    c = TraceCollector(tmp_path / "run", "rid", "task", "mock:synthetic")
    c.record(
        "tool_observation",
        {"result": {"stdout": "leaked sk-abcdefghijklmnopqrstuvwxyz0123", "big": "y" * 99999}},
    )
    raw = (tmp_path / "run" / "trace.jsonl").read_text()
    assert "sk-abcdefghij" not in raw  # secret never hits disk
    ev = json.loads(raw.splitlines()[0])
    assert "[REDACTED]" in ev["data"]["result"]["stdout"]
    assert len(ev["data"]["result"]["big"]) <= MAX_FIELD_CHARS + 40  # capped + marker


def test_finalize_redacts_summary(tmp_path: Path) -> None:
    c = TraceCollector(tmp_path / "run", "rid", "task", "mock:synthetic")
    summary = c.finalize(
        {"functional": 1.0},
        {"verifier": {"stdout": "ghp_0123456789abcdefghijklmnopqrstuvwx"}},
    )
    assert "[REDACTED]" in summary["verifier"]["stdout"]
    on_disk = (tmp_path / "run" / "summary.json").read_text()
    assert "ghp_0123456789" not in on_disk
