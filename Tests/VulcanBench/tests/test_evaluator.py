"""Tests for the scorer."""

from __future__ import annotations

from pathlib import Path

from harness.evaluator.scorer import efficiency_score, run_verifier, score_run


def test_efficiency_full_when_free() -> None:
    assert efficiency_score(0, 0) == 1.0


def test_efficiency_decreases_with_tokens() -> None:
    cheap = efficiency_score(1_000, 2)
    pricey = efficiency_score(40_000, 80)
    assert cheap > pricey
    assert 0.0 <= pricey <= 1.0


def test_score_run_excludes_none_metrics() -> None:
    # Only functional + efficiency present -> total re-normalizes over their weights.
    s = score_run(functional=1.0, total_tokens=0, steps=0)
    assert s["functional"] == 1.0
    assert s["total"] == 1.0
    assert s["quality"] is None
    assert s["security"] is None
    assert s["human_like"] is None


def test_score_run_partial_functional() -> None:
    s = score_run(functional=0.0, total_tokens=0, steps=0)
    # functional 0.5 * 0 + efficiency 0.1 * 1.0, normalized over (0.5 + 0.1) -> 0.1667
    assert s["total"] == round(0.10 / 0.60, 4)


def test_score_run_all_five_metrics() -> None:
    s = score_run(
        functional=1.0,
        total_tokens=0,
        steps=0,
        quality=0.5,
        security=1.0,
        human_like=0.8,
    )
    assert s["quality"] == 0.5
    assert s["security"] == 1.0
    assert s["human_like"] == 0.8
    # Weighted: .5*1 + .15*.5 + .15*1 + .1*1 + .1*.8 = 0.905 over full weight 1.0
    assert s["total"] == 0.905


def test_run_verifier_parses_json(tmp_path: Path) -> None:
    verifier = tmp_path / "v.py"
    verifier.write_text('import json; print(json.dumps({"functional": 0.5, "details": ["ok"]}))')
    out = run_verifier(verifier, tmp_path)
    assert out["scores"]["functional"] == 0.5
    assert out["exit_code"] == 0


def test_run_verifier_handles_non_json(tmp_path: Path) -> None:
    verifier = tmp_path / "v.py"
    verifier.write_text('print("not json")')
    out = run_verifier(verifier, tmp_path)
    assert out["scores"]["functional"] == 0.0
    assert "error" in out["scores"]
