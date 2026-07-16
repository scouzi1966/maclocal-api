"""dag — a tiny directed-acyclic-graph library with a topological sort."""
from dag.errors import CycleError
from dag.graph import Graph

__all__ = ["Graph", "CycleError"]
