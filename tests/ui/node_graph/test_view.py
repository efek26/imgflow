from PySide6.QtCore import QPointF

from imgflow.core.graph import Graph
from imgflow.operators import registry
from imgflow.ui.node_graph.scene import NodeGraphScene
from imgflow.ui.node_graph.view import NodeGraphView


def _view(qtbot) -> NodeGraphView:
    scene = NodeGraphScene(Graph(), registry)
    view = NodeGraphView(scene)
    qtbot.addWidget(view)
    return view


def test_add_node_at_creates_node_at_given_position(qtbot):
    view = _view(qtbot)
    item = view.add_node_at("io.image_source", QPointF(50, 60))

    assert item.node_id in view.node_graph_scene.graph.nodes
    assert view.node_graph_scene.graph.nodes[item.node_id].position == (50.0, 60.0)


def test_add_node_at_generates_unique_ids(qtbot):
    view = _view(qtbot)
    first = view.add_node_at("io.image_source", QPointF(0, 0))
    second = view.add_node_at("io.image_source", QPointF(100, 0))
    assert first.node_id != second.node_id


def test_delete_selected_removes_selected_nodes(qtbot):
    view = _view(qtbot)
    item = view.add_node_at("io.image_source", QPointF(0, 0))
    item.setSelected(True)

    view.delete_selected()

    assert item.node_id not in view.node_graph_scene.graph.nodes


def test_delete_selected_leaves_unselected_nodes(qtbot):
    view = _view(qtbot)
    kept_item = view.add_node_at("io.image_source", QPointF(0, 0))
    removed_item = view.add_node_at("color.convert", QPointF(200, 0))
    removed_item.setSelected(True)

    view.delete_selected()

    assert kept_item.node_id in view.node_graph_scene.graph.nodes
    assert removed_item.node_id not in view.node_graph_scene.graph.nodes


def test_wheel_event_zooms_in_within_bounds(qtbot):
    view = _view(qtbot)
    initial_scale = view._scale

    class _FakeAngleDelta:
        def y(self) -> int:
            return 120

    class _FakeWheelEvent:
        def angleDelta(self) -> _FakeAngleDelta:
            return _FakeAngleDelta()

    view.wheelEvent(_FakeWheelEvent())

    assert view._scale > initial_scale
