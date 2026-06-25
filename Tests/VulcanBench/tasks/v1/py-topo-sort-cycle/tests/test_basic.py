"""Invariants that hold both before and after the fix (regression guard)."""
from dag import Graph


def test_empty_graph_is_empty_order():
    assert Graph().topological_order() == []


def test_single_node():
    g = Graph()
    g.add_node("only")
    assert g.topological_order() == ["only"]


def test_add_edge_creates_nodes_idempotently():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("a", "b")  # duplicate edge must not duplicate nodes
    assert sorted(g.topological_order()) == ["a", "b"]
