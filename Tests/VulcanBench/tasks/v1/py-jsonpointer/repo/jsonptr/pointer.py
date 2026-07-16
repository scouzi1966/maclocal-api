"""Resolve a JSON Pointer (RFC 6901) against a decoded JSON document.

NOTE: this implementation is incomplete — see ``issue.md``. It walks object
members but does not handle escaping, arrays, or the error cases the RFC
requires.
"""
from __future__ import annotations

from typing import Any

from jsonptr.errors import JSONPointerError


def resolve(document: Any, pointer: str) -> Any:
    """Return the value in ``document`` referenced by ``pointer``."""
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise JSONPointerError(f"invalid JSON Pointer {pointer!r}: must be empty or start with '/'")
    current = document
    for token in pointer.split("/")[1:]:
        current = current[token]
    return current
