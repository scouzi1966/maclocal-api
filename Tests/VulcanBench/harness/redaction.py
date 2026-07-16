"""Sanitize text that gets persisted to published run artifacts.

Traces, summaries, and patches are meant to be shared, so they must not leak
secrets and must not balloon to multi-megabyte blobs. :func:`redact` scrubs
common credential shapes plus the literal values of known secret env vars;
:func:`sanitize` walks an arbitrary JSON-able structure applying redaction and a
per-string length cap.

This is defense-in-depth: best-effort pattern matching, not a guarantee. It
errs toward over-redaction (a code line that looks like a key gets masked) —
acceptable for an artifact whose point is to be publishable.
"""

from __future__ import annotations

import os
import re
from typing import Any

REDACTED = "[REDACTED]"
MAX_FIELD_CHARS = 20_000

# Credential shapes. Order matters: more specific prefixes (sk-ant-) first.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),  # Anthropic
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),  # OpenAI
    re.compile(r"gh[opsru]_[A-Za-z0-9]{16,}"),  # GitHub tokens (ghp_/gho_/...)
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),  # GitHub fine-grained PAT
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),  # Slack
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),  # Google API key
)

# Env vars whose literal values are scrubbed wherever they appear.
_SECRET_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ZAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "HF_TOKEN",
    "VULCANBENCH_API_TOKEN",
    "VULCANBENCH_API_KEY",  # legacy alias
)

_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._-]{16,}")


def redact(text: str) -> str:
    """Mask credential-shaped substrings and known secret env values in ``text``."""
    if not text:
        return text
    # Concrete env values first (most reliable signal).
    for var in _SECRET_ENV:
        val = os.environ.get(var)
        if val and len(val) >= 8:
            text = text.replace(val, REDACTED)
    text = _BEARER.sub(rf"\1{REDACTED}", text)
    for pat in _PATTERNS:
        text = pat.sub(REDACTED, text)
    return text


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...[truncated {len(text) - max_chars} chars]"


def sanitize(obj: Any, max_chars: int = MAX_FIELD_CHARS) -> Any:
    """Recursively redact secrets and cap string length in a JSON-able value."""
    if isinstance(obj, str):
        return _cap(redact(obj), max_chars)
    if isinstance(obj, dict):
        return {k: sanitize(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v, max_chars) for v in obj]
    return obj
