#!/usr/bin/env python3
"""Ingest local ``./runs/`` summaries into the configured database.

Set ``DATABASE_URL`` first (e.g. a Postgres URL, or
``sqlite:///vulcanbench.db``), then:

    DATABASE_URL=... python scripts/ingest_runs.py [./runs]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from backend import db


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    runs_dir = Path(argv[0]) if argv else Path(os.environ.get("VULCANBENCH_RUNS_DIR", "./runs"))

    if not db.enabled():
        print("error: set DATABASE_URL before ingesting", file=sys.stderr)
        return 2
    db.init_db()

    if not runs_dir.exists():
        print(f"no runs dir at {runs_dir}")
        return 0

    count = 0
    for d in sorted(runs_dir.iterdir()):
        summary = d / "summary.json"
        if not summary.exists():
            continue
        try:
            db.upsert_run(json.loads(summary.read_text(encoding="utf-8")))
            count += 1
        except (OSError, json.JSONDecodeError):
            continue
    print(f"ingested {count} runs into the database")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
