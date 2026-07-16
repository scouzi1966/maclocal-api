#!/usr/bin/env python3
"""CLI shim for the task validator — the logic lives in ``harness.validate``.

Usage:
    python scripts/validate_tasks.py [tasks/v1]          # all tasks (host)
    python scripts/validate_tasks.py tasks/v1/<task-id>  # one task
    python scripts/validate_tasks.py tasks/v1 --sandbox docker  # in-container
"""

from __future__ import annotations

from harness.validate import main

if __name__ == "__main__":
    raise SystemExit(main())
