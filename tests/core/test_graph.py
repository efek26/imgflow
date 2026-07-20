import pytest

from imgflow.core.errors import CycleError
from imgflow.core.graph import Edge, Graph, Node


def linear_graph() -> Graph:
    g = Graph()
    g.add_node(Node("a", "op.a"))
    g.add_node(Node("b", "op.b"))
    g.add_node(Node("c", "op.c"))
    g.add_edge(Edge(("a", "out"), ("b", "in")))
    g.add_edge(Edge(("b", "out"), ("c", "in")))
    return g


def test_successors_and_predecessors():
    g = linear_graph()
    assert g.successors("a") == ["b"]
    assert g.predecessors("c") == ["b"]
    assert g.predecessors("a") == []
    assert g.successors("c") == []


def test_ancestors_closure():
    g = linear_graph()
    assert g.ancestors_closure("c") == {"a", "b"}
    assert g.ancestors_closure("a") == set()


def test_topo_order_linear():
    g = linear_graph()
    assert g.topo_order() == ["a", "b", "c"]


def test_topo_order_branch_merge():
    g = Graph()
    for nid in ("src", "t1", "t2", "merge"):
        g.add_node(Node(nid, f"op.{nid}"))
    g.add_edge(Edge(("src", "out"), ("t1", "in")))
    g.add_edge(Edge(("src", "out"), ("t2", "in")))
    g.add_edge(Edge(("t1", "out"), ("merge", "a")))
    g.add_edge(Edge(("t2", "out"), ("merge", "b")))

    order = g.topo_order()
    assert order.index("src") < order.index("t1")
    assert order.index("src") < order.index("t2")
    assert order.index("t1") < order.index("merge")
    assert order.index("t2") < order.index("merge")
    assert g.ancestors_closure("merge") == {"src", "t1", "t2"}


def test_topo_order_raises_on_cycle():
    g = Graph()
    g.add_node(Node("a", "op.a"))
    g.add_node(Node("b", "op.b"))
    g.add_edge(Edge(("a", "out"), ("b", "in")))
    g.add_edge(Edge(("b", "out"), ("a", "in")))
    with pytest.raises(CycleError):
        g.topo_order()


def test_add_node_duplicate_id_raises():
    g = Graph()
    g.add_node(Node("a", "op.a"))
    with pytest.raises(ValueError):
        g.add_node(Node("a", "op.b"))


def test_add_edge_unknown_node_raises():
    g = Graph()
    g.add_node(Node("a", "op.a"))
    with pytest.raises(KeyError):
        g.add_edge(Edge(("a", "out"), ("missing", "in")))


def test_remove_node_cleans_up_edges():
    g = linear_graph()
    g.remove_node("b")
    assert "b" not in g.nodes
    assert all(e.src[0] != "b" and e.dst[0] != "b" for e in g.edges)
