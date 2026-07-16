"""Tests for the optional database store. Skipped unless deps are installed."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from backend import db
from backend.app import app
from harness.leaderboard import aggregate_by_model


def _summary(run_id: str, model: str = "openai:gpt-4o", total: float = 0.9) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "task_id": "py-ttl-cache-expiry",
        "model": model,
        "suite": "v1",
        "suite_id": "suite-1",
        "effort": {
            "requested": "low",
            "provider": "openai",
            "provider_value": "low",
            "supported": True,
        },
        "experiment_id": "experiment-1",
        "steps": 5,
        "total_tokens": 1000,
        "cost_usd": 0.01,
        "duration_s": 3.2,
        "finished_at": "2026-05-30T00:00:00+00:00",
        "manifest": {"task": {"repo_scale": "small", "task_complexity": "localized"}},
        "scores": {"functional": 1.0, "total": total, "quality": 0.8},
    }


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Iterator[None]:
    db.configure(f"sqlite:///{tmp_path / 't.db'}")
    db.init_db()
    try:
        yield
    finally:
        db.configure(None)  # restore filesystem mode for other tests


def test_disabled_by_default() -> None:
    db.configure(None)
    assert db.enabled() is False
    assert db.leaderboard_rows() == []
    assert db.get_summary("x") is None


def test_upsert_and_rows(sqlite_db: None) -> None:
    db.upsert_run(_summary("r1", total=0.9))
    db.upsert_run(_summary("r2", total=0.5))
    rows = db.leaderboard_rows()
    assert {r["run_id"] for r in rows} == {"r1", "r2"}
    r = next(r for r in rows if r["run_id"] == "r1")
    assert r["total"] == 0.9 and r["cost_usd"] == 0.01 and r["suite"] == "v1"
    assert r["effort_requested"] == "low"
    assert r["task_complexity"] == "localized"
    assert db.get_summary("r1")["scores"]["quality"] == 0.8


def test_upsert_is_idempotent(sqlite_db: None) -> None:
    db.upsert_run(_summary("r1", total=0.5))
    db.upsert_run(_summary("r1", total=0.9))  # replace
    rows = db.leaderboard_rows()
    assert len(rows) == 1 and rows[0]["total"] == 0.9


def test_aggregate_from_db_rows(sqlite_db: None) -> None:
    db.upsert_run(_summary("r1", model="m", total=1.0))
    db.upsert_run(_summary("r2", model="m", total=0.0))
    aggs = aggregate_by_model(db.leaderboard_rows())
    assert aggs[0]["model"] == "m"
    # Same task_id across both runs -> 1 distinct task, 2 runs.
    assert aggs[0]["n_tasks"] == 1
    assert aggs[0]["n_runs"] == 2
    assert aggs[0]["avg_total"] == 0.5


def test_feedback(sqlite_db: None) -> None:
    out = db.add_feedback(task_id="py-ttl-cache-expiry", run_id="r1", rating=4, comment="nice")
    assert out["id"] is not None


def test_post_run_and_feedback_via_api(sqlite_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VULCANBENCH_API_TOKEN", "s3cret")
    headers = {"Authorization": "Bearer s3cret"}
    c = TestClient(app)
    assert c.get("/api/health").json()["store"] == "db"
    assert c.post("/api/runs", json=_summary("r9"), headers=headers).json()["ok"] is True
    # The leaderboard now reflects the DB row.
    aggs = c.get("/api/leaderboard").json()
    assert any(a["model"] == "openai:gpt-4o" for a in aggs)
    assert c.get("/api/run/r9").json()["scores"]["total"] == 0.9
    fb = c.post("/api/feedback", json={"run_id": "r9", "rating": 5}, headers=headers)
    assert fb.json()["id"] is not None


def test_post_run_requires_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VULCANBENCH_API_TOKEN", "s3cret")
    db.configure(None)
    c = TestClient(app)
    headers = {"Authorization": "Bearer s3cret"}
    assert c.post("/api/runs", json={"run_id": "x"}, headers=headers).status_code == 400
