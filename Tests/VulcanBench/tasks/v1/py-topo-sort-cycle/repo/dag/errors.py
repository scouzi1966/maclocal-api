"""Exceptions raised by the :mod:`dag` library."""
from __future__ import annotations


class CycleError(Exception):
    """Raised when a topological order is requested for a graph that has a cycle."""
