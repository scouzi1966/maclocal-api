"""Post-run persistence helpers (optional API write-through)."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def _api_base() -> str | None:
    base = os.environ.get("VULCANBENCH_API_BASE", "").strip().rstrip("/")
    return base or None


def _api_token() -> str | None:
    token = os.environ.get("VULCANBENCH_API_TOKEN", "").strip()
    return token or None


def maybe_post_run_summary(summary: dict[str, Any]) -> bool:
    """POST ``summary`` to ``/api/runs`` when API env vars are configured.

    Returns True on success, False when skipped or on failure (logged, non-fatal).
    """
    base = _api_base()
    if base is None:
        return False
    token = _api_token()
    if token is None:
        logger.warning(
            "VULCANBENCH_API_BASE is set but VULCANBENCH_API_TOKEN is missing; "
            "skipping write-through"
        )
        return False

    url = f"{base}/api/runs"
    payload = json.dumps(summary).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status >= 400:
                logger.warning("write-through failed: HTTP %s", resp.status)
                return False
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.warning("write-through failed: HTTP %s — %s", e.code, body[:500])
        return False
    except OSError as e:
        logger.warning("write-through failed: %s", e)
        return False
    logger.info("posted run %s to %s", summary.get("run_id"), url)
    return True
