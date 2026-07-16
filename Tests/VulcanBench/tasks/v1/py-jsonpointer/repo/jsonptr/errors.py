"""Exceptions raised by :mod:`jsonptr`."""
from __future__ import annotations


class JSONPointerError(Exception):
    """Raised for a malformed pointer or a pointer that does not resolve."""
