"""Topological-order behavior — fails on the starting repo, passes after the fix."""
import pytest

from dag import CycleError, Graph


def test_linear_chain():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    assert g.topological_order() == ["a", "b", "c"]


def test_ready_nodes_emitted_smallest_first():
    # a, b, c are all ready (no incoming edges); each points at d.
    g = Graph()
    g.add_edge("a", "d")
    g.add_edge("b", "d")
    g.add_edge("c", "d")
    assert g.topological_order() == ["a", "b", "c", "d"]


def test_diamond_is_deterministic():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", "d")
    g.add_edge("c", "d")
    assert g.topological_order() == ["a", "b", "c", "d"]


def test_two_cycle_raises():
    g = Graph()
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    with pytest.raises(CycleError):
        g.topological_order()


def test_self_loop_raises():
    g = Graph()
    g.add_edge("x", "x")
    with pytest.raises(CycleError):
        g.topological_order()


def test_cycle_with_tail_raises():
    # A clean chain that feeds into a 3-cycle must still be detected.
    g = Graph()
    g.add_edge("start", "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "a")
    with pytest.raises(CycleError):
        g.topological_order()
