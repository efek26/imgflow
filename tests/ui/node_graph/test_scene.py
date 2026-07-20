from PySide6.QtCore import QPointF

from imgflow.core.graph import Edge, Graph, Node
from imgflow.operators import registry
from imgflow.ui.node_graph.scene import NodeGraphScene


def _scene() -> NodeGraphScene:
    return NodeGraphScene(Graph(), registry)


def test_add_node_registers_in_graph_and_scene(qtbot):
    scene = _scene()
    item = scene.add_node(Node("src", "io.image_source"))
    assert "src" in scene.graph.nodes
    assert scene.node_items["src"] is item


def test_connect_ports_creates_edge(qtbot):
    scene = _scene()
    src_item = scene.add_node(Node("src", "io.image_source"))
    dst_item = scene.add_node(Node("gray", "color.convert"))

    ok = scene.connect_ports(src_item.output_ports["image"], dst_item.input_ports["image"])

    assert ok is True
    assert len(scene.graph.edges) == 1
    assert scene.graph.edges[0].src == ("src", "image")
    assert scene.graph.edges[0].dst == ("gray", "image")
    assert len(scene.edge_items) == 1


def test_connect_ports_accepts_reversed_argument_order(qtbot):
    scene = _scene()
    src_item = scene.add_node(Node("src", "io.image_source"))
    dst_item = scene.add_node(Node("gray", "color.convert"))

    ok = scene.connect_ports(dst_item.input_ports["image"], src_item.output_ports["image"])

    assert ok is True
    assert scene.graph.edges[0].src == ("src", "image")


def test_connect_ports_rejects_type_mismatch(qtbot):
    scene = _scene()
    cc_item = scene.add_node(Node("cc", "segment.connected_components"))
    props_item = scene.add_node(Node("props", "analysis.region_props"))

    ok = scene.connect_ports(cc_item.output_ports["count"], props_item.input_ports["labels"])

    assert ok is False
    assert scene.graph.edges == []


def test_connect_ports_rejects_output_to_output(qtbot):
    scene = _scene()
    a = scene.add_node(Node("a", "io.image_source"))
    b = scene.add_node(Node("b", "io.image_source"))

    assert scene.connect_ports(a.output_ports["image"], b.output_ports["image"]) is False


def test_connect_ports_rejects_self_loop(qtbot):
    scene = _scene()
    a = scene.add_node(Node("a", "color.convert"))
    assert scene.connect_ports(a.output_ports["image"], a.input_ports["image"]) is False


def test_connect_ports_rejects_cycle(qtbot):
    scene = _scene()
    a = scene.add_node(Node("a", "color.convert"))
    b = scene.add_node(Node("b", "color.convert"))
    scene.connect_ports(a.output_ports["image"], b.input_ports["image"])

    ok = scene.connect_ports(b.output_ports["image"], a.input_ports["image"])

    assert ok is False
    assert len(scene.graph.edges) == 1


def test_connecting_new_edge_replaces_existing_on_same_input(qtbot):
    scene = _scene()
    src1 = scene.add_node(Node("src1", "io.image_source"))
    src2 = scene.add_node(Node("src2", "io.image_source"))
    dst = scene.add_node(Node("gray", "color.convert"))

    scene.connect_ports(src1.output_ports["image"], dst.input_ports["image"])
    scene.connect_ports(src2.output_ports["image"], dst.input_ports["image"])

    assert len(scene.graph.edges) == 1
    assert scene.graph.edges[0].src == ("src2", "image")


def test_remove_node_cleans_up_edges_and_items(qtbot):
    scene = _scene()
    src_item = scene.add_node(Node("src", "io.image_source"))
    dst_item = scene.add_node(Node("gray", "color.convert"))
    scene.connect_ports(src_item.output_ports["image"], dst_item.input_ports["image"])

    scene.remove_node("src")

    assert "src" not in scene.graph.nodes
    assert scene.graph.edges == []
    assert scene.edge_items == []
    assert "src" not in scene.node_items


def test_disconnect_edge_item_removes_edge(qtbot):
    scene = _scene()
    src_item = scene.add_node(Node("src", "io.image_source"))
    dst_item = scene.add_node(Node("gray", "color.convert"))
    scene.connect_ports(src_item.output_ports["image"], dst_item.input_ports["image"])

    scene.disconnect_edge_item(scene.edge_items[0])

    assert scene.graph.edges == []
    assert scene.edge_items == []


def test_moving_node_updates_graph_position(qtbot):
    scene = _scene()
    src_item = scene.add_node(Node("src", "io.image_source", position=(0, 0)))
    dst_item = scene.add_node(Node("gray", "color.convert", position=(200, 0)))
    scene.connect_ports(src_item.output_ports["image"], dst_item.input_ports["image"])

    dst_item.setPos(300, 50)

    assert scene.graph.nodes["gray"].position == (300.0, 50.0)


def test_generate_node_id_avoids_collisions(qtbot):
    scene = _scene()
    scene.add_node(Node("image_source", "io.image_source"))
    assert scene.generate_node_id("io.image_source") == "image_source_2"


def test_begin_update_end_connection_creates_edge_when_released_over_compatible_port(qtbot):
    scene = _scene()
    src_item = scene.add_node(Node("src", "io.image_source"))
    dst_item = scene.add_node(Node("gray", "color.convert"))
    src_port = src_item.output_ports["image"]
    dst_port = dst_item.input_ports["image"]

    scene.begin_connection(src_port, src_port.scene_center())
    scene.update_connection(dst_port.scene_center())
    scene.end_connection(dst_port.scene_center())

    assert len(scene.graph.edges) == 1
    assert scene._temp_line is None


def test_load_graph_replaces_content_and_preserves_graph_identity(qtbot):
    scene = _scene()
    scene.add_node(Node("old", "io.image_source"))
    original_graph = scene.graph

    new_graph = Graph()
    new_graph.add_node(Node("src", "io.image_source", position=(5.0, 5.0)))
    new_graph.add_node(Node("gray", "color.convert"))
    new_graph.add_edge(Edge(("src", "image"), ("gray", "image")))

    scene.load_graph(new_graph)

    assert scene.graph is original_graph
    assert set(scene.graph.nodes) == {"src", "gray"}
    assert set(scene.node_items) == {"src", "gray"}
    assert len(scene.edge_items) == 1
    assert "old" not in scene.graph.nodes


def test_end_connection_without_target_creates_no_edge(qtbot):
    scene = _scene()
    src_item = scene.add_node(Node("src", "io.image_source"))
    src_port = src_item.output_ports["image"]

    scene.begin_connection(src_port, src_port.scene_center())
    scene.end_connection(QPointF(5000, 5000))

    assert scene.graph.edges == []
