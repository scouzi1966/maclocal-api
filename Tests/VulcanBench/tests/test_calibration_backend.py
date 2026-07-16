"""Backend API tests for the calibration feature.

Skipped unless the optional `backend` deps are installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

import backend.app as appmod
from backend import db


def test_backend_calibration_endpoint(tmp_path: Path) -> None:
    appmod.RUNS_DIR = tmp_path
    c = TestClient(appmod.app)
    resp = c.get("/api/calibration")
    assert resp.status_code == 200
    data = resp.json()
    assert "tasks" in data
    assert "summary" in data


def test_backend_tasks_enriched(tmp_path: Path) -> None:
    appmod.RUNS_DIR = tmp_path
    c = TestClient(appmod.app)
    tasks = c.get("/api/tasks").json()
    assert len(tasks) > 0
    t = tasks[0]
    assert "empirical_difficulty" in t
    assert "solve_rate" in t
    assert "calibration_status" in t


def test_db_rows_include_task_hash(tmp_path: Path) -> None:
    db.configure(f"sqlite:///{tmp_path / 't.db'}")
    db.init_db()
    try:
        db.upsert_run(
            {
                "run_id": "r1",
                "task_id": "t",
                "model": "m",
                "scores": {"functional": 1.0},
                "task_hash": "abc123",
            }
        )
        rows = db.leaderboard_rows()
        assert rows[0]["task_hash"] == "abc123"
    finally:
        db.configure(None)
