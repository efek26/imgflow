import pytest

from imgflow.core.graph import Node
from imgflow.core.linear_pipeline import LinearPipeline
from imgflow.operators import registry


def test_append_chains_nodes_via_primary_ports():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("src", "io.image_source"))
    pipeline.append(Node("gray", "color.convert"))
    pipeline.append(Node("th", "segment.threshold"))

    assert pipeline.order == ["src", "gray", "th"]
    assert len(pipeline.graph.edges) == 2
    assert pipeline.graph.edges[0].src == ("src", "image")
    assert pipeline.graph.edges[0].dst == ("gray", "image")
    assert pipeline.graph.edges[1].src == ("gray", "image")
    assert pipeline.graph.edges[1].dst == ("th", "image")


def test_first_node_without_inputs_gets_no_incoming_edge():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("src", "io.image_source"))
    assert pipeline.graph.edges == []


def test_set_order_reorders_and_rewires():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("a", "morphology.erode"))
    pipeline.append(Node("b", "morphology.dilate"))
    pipeline.append(Node("c", "morphology.open"))

    pipeline.set_order(["c", "a", "b"])

    assert pipeline.order == ["c", "a", "b"]
    assert pipeline.graph.edges[0].src[0] == "c"
    assert pipeline.graph.edges[0].dst[0] == "a"
    assert pipeline.graph.edges[1].src[0] == "a"
    assert pipeline.graph.edges[1].dst[0] == "b"


def test_set_order_with_mismatched_nodes_raises():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("a", "morphology.erode"))
    with pytest.raises(ValueError):
        pipeline.set_order(["a", "missing"])


def test_remove_rewires_remaining_chain():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("src", "io.image_source"))
    pipeline.append(Node("gray", "color.convert"))
    pipeline.append(Node("th", "segment.threshold"))

    pipeline.remove("gray")

    assert pipeline.order == ["src", "th"]
    assert len(pipeline.graph.edges) == 1
    assert pipeline.graph.edges[0].src == ("src", "image")
    assert pipeline.graph.edges[0].dst == ("th", "image")


def test_existing_graph_and_order_are_rewired_on_construction():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("a", "morphology.erode"))
    pipeline.append(Node("b", "morphology.dilate"))

    rebuilt = LinearPipeline(registry, graph=pipeline.graph, order=list(pipeline.order))
    assert rebuilt.graph.edges[0].src[0] == "a"
    assert rebuilt.graph.edges[0].dst[0] == "b"
