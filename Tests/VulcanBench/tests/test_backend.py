"""Tests for the read API. Skipped unless the optional `backend` deps are installed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by Starlette's TestClient

from fastapi.testclient import TestClient

import backend.app as appmod
from backend import db


def _client(runs_dir: Path) -> TestClient:
    # Point the API at a temp runs dir for isolation (endpoints read it lazily).
    appmod.RUNS_DIR = runs_dir
    return TestClient(appmod.app)


def _seed(runs: Path, run_id: str, total: float) -> None:
    d = runs / run_id
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "task_id": "hello-world",
                "model": "mock:synthetic",
                "steps": 4,
                "effort": {
                    "requested": "low" if total < 0.7 else "high",
                    "provider": "mock",
                    "provider_value": None,
                    "supported": False,
                },
                "manifest": {
                    "task": {
                        "repo_scale": "micro",
                        "task_complexity": "localized",
                        "languages": ["python"],
                        "difficulty": "trivial",
                    }
                },
                "scores": {"functional": 1.0, "total": total},
            }
        )
    )
    (d / "trace.jsonl").write_text(
        json.dumps({"step": 1, "ts": "t", "type": "task_start", "data": {}}) + "\n"
    )


def test_health(tmp_path: Path) -> None:
    body = _client(tmp_path).get("/api/health").json()
    assert body["status"] == "ok"
    assert body["store"] in {"filesystem", "db"}


def test_leaderboard_by_run_sorted(tmp_path: Path) -> None:
    _seed(tmp_path, "low", 0.5)
    _seed(tmp_path, "high", 0.9)
    rows = _client(tmp_path).get("/api/leaderboard?by=run").json()
    assert [r["run_id"] for r in rows] == ["high", "low"]


def test_leaderboard_by_model_default(tmp_path: Path) -> None:
    _seed(tmp_path, "low", 0.5)
    _seed(tmp_path, "high", 0.9)
    # Default view aggregates by model (both runs are mock:synthetic).
    aggs = _client(tmp_path).get("/api/leaderboard").json()
    assert len(aggs) == 1
    assert aggs[0]["model"] == "mock:synthetic"
    # Both seeded runs are the same task_id -> 1 distinct task, 2 runs.
    assert aggs[0]["n_tasks"] == 1
    assert aggs[0]["n_runs"] == 2


def test_leaderboard_filters_effort_and_complexity(tmp_path: Path) -> None:
    _seed(tmp_path, "low", 0.5)
    _seed(tmp_path, "high", 0.9)
    rows = _client(tmp_path).get("/api/leaderboard?by=run&effort=low").json()
    assert [r["run_id"] for r in rows] == ["low"]
    rows = _client(tmp_path).get("/api/leaderboard?by=run&task_complexity=localized").json()
    assert {r["run_id"] for r in rows} == {"low", "high"}


def test_effort_sensitivity_endpoint(tmp_path: Path) -> None:
    _seed(tmp_path, "low", 0.5)
    _seed(tmp_path, "high", 0.9)
    body = _client(tmp_path).get("/api/effort-sensitivity").json()
    assert body["available"] is True
    assert body["strata"][0]["classification"] == "effort sensitive"


def test_run_and_trace(tmp_path: Path) -> None:
    _seed(tmp_path, "r1", 0.9)
    c = _client(tmp_path)
    assert c.get("/api/run/r1").json()["scores"]["total"] == 0.9
    assert len(c.get("/api/run/r1/trace").json()) == 1


def test_run_not_found(tmp_path: Path) -> None:
    assert _client(tmp_path).get("/api/run/missing").status_code == 404


def test_tasks_list_and_detail(tmp_path: Path) -> None:
    # Uses the real tasks/v1 dir (TASKS_ROOT default).
    c = _client(tmp_path)
    tasks = c.get("/api/tasks").json()
    ids = {t["id"] for t in tasks}
    assert "py-ttl-cache-expiry" in ids
    assert all("task_complexity" in t for t in tasks)
    detail = c.get("/api/task/py-ttl-cache-expiry").json()
    assert detail["id"] == "py-ttl-cache-expiry"
    assert "TTLCache" in detail["issue"]
    assert "runs" in detail
    assert c.get("/api/task/nope").status_code == 404


def test_run_patch_endpoint(tmp_path: Path) -> None:
    d = tmp_path / "r1"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"run_id": "r1", "scores": {}}))
    (d / "final.patch").write_text("diff --git a/x b/x\n+hi\n")
    c = _client(tmp_path)
    assert "diff --git" in c.get("/api/run/r1/patch").json()["patch"]
    assert c.get("/api/run/missing/patch").json()["patch"] == ""


def test_write_endpoints_fail_closed_without_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VULCANBENCH_API_TOKEN", raising=False)
    c = _client(tmp_path)
    assert c.post("/api/runs", json={"run_id": "r1"}).status_code == 503
    assert c.post("/api/feedback", json={"comment": "hi"}).status_code == 503


def test_write_endpoints_reject_bad_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VULCANBENCH_API_TOKEN", "s3cret")
    c = _client(tmp_path)
    assert c.post("/api/runs", json={"run_id": "r1"}).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert c.post("/api/runs", json={"run_id": "r1"}, headers=bad).status_code == 401


def test_write_endpoints_accept_valid_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VULCANBENCH_API_TOKEN", "s3cret")
    db.configure(f"sqlite:///{tmp_path / 't.db'}")
    try:
        db.init_db()
        c = _client(tmp_path)
        headers = {"Authorization": "Bearer s3cret"}
        resp = c.post("/api/runs", json={"run_id": "r1", "scores": {}}, headers=headers)
        assert resp.status_code == 200 and resp.json()["ok"] is True
        resp = c.post("/api/feedback", json={"comment": "hi"}, headers=headers)
        assert resp.status_code == 200 and resp.json()["id"] is not None
    finally:
        db.configure(None)
