import pytest

from imgflow.core.graph import Graph, Node
from imgflow.core.linear_pipeline import LinearPipeline
from imgflow.operators import registry


def test_append_chains_nodes_via_primary_ports():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("src", "io.image_source"))
    pipeline.append(Node("gray", "color.grayscale"))
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
    pipeline.append(Node("gray", "color.grayscale"))
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


def test_generate_node_id_avoids_collisions():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("image_source", "io.image_source"))
    assert pipeline.generate_node_id("io.image_source") == "image_source_2"


def test_generate_node_id_uses_base_name_when_free():
    pipeline = LinearPipeline(registry)
    assert pipeline.generate_node_id("color.grayscale") == "grayscale"


def test_rewire_skips_edge_when_primary_port_types_mismatch():
    # analysis.region_props'un birincil çıktısı MEASUREMENTS tipinde; segment.threshold
    # girdisi IMAGE bekliyor. Körlemesine bağlanırsa threshold bir ölçüm listesi alır ve
    # çöker; bunun yerine hiç bağlanmamalı ki engine net bir "girdi bağlı değil" hatası versin.
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("props", "analysis.region_props"))
    pipeline.append(Node("th", "segment.threshold"))

    assert pipeline.graph.edges == []


def test_rewire_connects_when_primary_port_types_match():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("cc", "segment.connected_components"))
    pipeline.append(Node("props", "analysis.region_props"))

    assert len(pipeline.graph.edges) == 1
    assert pipeline.graph.edges[0].src == ("cc", "labels")
    assert pipeline.graph.edges[0].dst == ("props", "labels")


def test_load_replaces_content_and_preserves_graph_identity():
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("old", "io.image_source"))
    original_graph = pipeline.graph

    new_graph = Graph()
    new_graph.add_node(Node("src", "io.image_source"))
    new_graph.add_node(Node("gray", "color.grayscale"))

    pipeline.load(new_graph)

    assert pipeline.graph is original_graph
    assert pipeline.order == ["src", "gray"]
    assert pipeline.graph.edges[0].src == ("src", "image")
    assert pipeline.graph.edges[0].dst == ("gray", "image")
    assert "old" not in pipeline.graph.nodes
