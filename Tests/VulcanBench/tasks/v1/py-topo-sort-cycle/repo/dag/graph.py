"""A directed graph with a topological ordering.

An edge ``u -> v`` means *u must come before v* in the output order.
"""
from __future__ import annotations

from typing import Hashable

from dag.errors import CycleError


class Graph:
    """A directed graph over hashable, mutually-comparable nodes."""

    def __init__(self) -> None:
        self._nodes: list[Hashable] = []  # insertion order
        self._succ: dict[Hashable, set[Hashable]] = {}

    def add_node(self, node: Hashable) -> None:
        if node not in self._succ:
            self._nodes.append(node)
            self._succ[node] = set()

    def add_edge(self, src: Hashable, dst: Hashable) -> None:
        """Add an edge ``src -> dst`` (creating the nodes if needed)."""
        self.add_node(src)
        self.add_node(dst)
        self._succ[src].add(dst)

    def topological_order(self) -> list[Hashable]:
        """Return the nodes in dependency order (Kahn's algorithm).

        See ``issue.md`` — this implementation is buggy.
        """
        indegree = {n: 0 for n in self._nodes}
        for src in self._nodes:
            for dst in self._succ[src]:
                indegree[dst] += 1

        ready = [n for n in self._nodes if indegree[n] == 0]
        order: list[Hashable] = []
        while ready:
            node = ready.pop()  # take the most-recently-added ready node
            order.append(node)
            for dst in self._succ[node]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    ready.append(dst)
        return order
