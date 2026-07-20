from imgflow.core.graph import Node
from imgflow.operators import registry
from imgflow.ui.node_graph.node_item import NodeItem


def test_node_item_creates_ports_matching_op_spec(qtbot):
    op_cls = registry.get("segment.threshold")
    node = Node("th", "segment.threshold", position=(10.0, 20.0))

    item = NodeItem(node, op_cls)

    assert set(item.input_ports) == {p.name for p in op_cls.inputs}
    assert set(item.output_ports) == {p.name for p in op_cls.outputs}
    assert item.pos().x() == 10.0
    assert item.pos().y() == 20.0


def test_node_item_without_scene_does_not_error_on_move(qtbot):
    op_cls = registry.get("io.image_source")
    node = Node("src", "io.image_source")
    item = NodeItem(node, op_cls)

    item.setPos(5.0, 5.0)  # scene() is None; itemChange must not raise
    assert item.pos().x() == 5.0
