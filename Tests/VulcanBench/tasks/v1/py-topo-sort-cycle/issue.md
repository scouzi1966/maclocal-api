# Topological sort returns wrong orders and never detects cycles

The `dag` library sorts a directed graph so that for every edge `u -> v`, `u`
comes before `v` in the result (`dag.Graph.topological_order`). Two things are
broken:

1. **Cycles are silently ignored.** If the graph contains a cycle there is no
   valid topological order, but `topological_order()` currently returns a
   *partial* list (the nodes it managed to emit) instead of failing. It should
   raise `dag.CycleError` when no valid ordering exists.

2. **The order is not deterministic / not stable.** When several nodes are
   ready to be emitted at the same time (all their predecessors are already
   placed), the algorithm should emit the **smallest** ready node first, so the
   output is reproducible across runs and machines. Today it emits them in a
   different, implementation-dependent order.

Fix `topological_order()` so that:

- among the nodes that are currently ready (in-degree 0), the smallest one is
  emitted next (nodes are mutually comparable, e.g. strings), and
- a `CycleError` is raised if the graph contains a cycle (i.e. not every node
  can be placed).

`CycleError` already exists in `dag/errors.py`. The algorithm lives in
`dag/graph.py`.
