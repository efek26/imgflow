import pytest

from imgflow.core.errors import RecipeVersionError
from imgflow.core.graph import Edge, Graph, Node
from imgflow.io_utils.recipe import (
    RECIPE_SCHEMA_VERSION,
    graph_from_dict,
    graph_to_dict,
    load_recipe,
    save_recipe,
)


def _sample_graph() -> Graph:
    g = Graph()
    g.add_node(Node("src", "io.image_source", params={"path": "a.png"}, position=(1.0, 2.0)))
    g.add_node(Node("gray", "color.convert", params={"mode": "BGR2GRAY"}))
    g.add_edge(Edge(("src", "image"), ("gray", "image")))
    return g


def test_roundtrip_via_dict():
    data = graph_to_dict(_sample_graph())
    restored = graph_from_dict(data)

    assert set(restored.nodes) == {"src", "gray"}
    assert restored.nodes["src"].op_id == "io.image_source"
    assert restored.nodes["src"].params == {"path": "a.png"}
    assert restored.nodes["src"].position == (1.0, 2.0)
    assert len(restored.edges) == 1
    assert restored.edges[0].src == ("src", "image")
    assert restored.edges[0].dst == ("gray", "image")


def test_roundtrip_via_file(tmp_path):
    g = _sample_graph()
    path = tmp_path / "recipe.json"

    save_recipe(path, g)
    restored = load_recipe(path)

    assert restored.topo_order() == g.topo_order()


def test_future_schema_version_raises():
    data = graph_to_dict(_sample_graph())
    data["schema_version"] = RECIPE_SCHEMA_VERSION + 1
    with pytest.raises(RecipeVersionError):
        graph_from_dict(data)


def test_missing_schema_version_raises():
    data = graph_to_dict(_sample_graph())
    del data["schema_version"]
    with pytest.raises(RecipeVersionError):
        graph_from_dict(data)
