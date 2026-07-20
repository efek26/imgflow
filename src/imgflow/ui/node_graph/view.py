"""Node-graph sahnesini gösteren, yakınlaştırma ve sağ tık menüsü destekleyen görünüm."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsView, QMenu

from imgflow.core.graph import Node
from imgflow.core.params import defaults_for
from imgflow.ui.node_graph.node_item import NodeItem
from imgflow.ui.node_graph.scene import NodeGraphScene

_ZOOM_STEP = 1.15
_MIN_SCALE = 0.2
_MAX_SCALE = 3.0


class NodeGraphView(QGraphicsView):
    def __init__(self, scene: NodeGraphScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.node_graph_scene = scene
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self._scale = 1.0

    def add_node_at(self, op_id: str, scene_pos: QPointF) -> NodeItem:
        op_cls = self.node_graph_scene.registry.get(op_id)
        node_id = self.node_graph_scene.generate_node_id(op_id)
        node = Node(
            node_id,
            op_id,
            params=defaults_for(op_cls.params),
            position=(scene_pos.x(), scene_pos.y()),
        )
        return self.node_graph_scene.add_node(node)

    def selected_node_ids(self) -> list[str]:
        return [
            item.node_id
            for item in self.node_graph_scene.selectedItems()
            if isinstance(item, NodeItem)
        ]

    def delete_selected(self) -> None:
        for node_id in self.selected_node_ids():
            self.node_graph_scene.remove_node(node_id)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        factor = _ZOOM_STEP if event.angleDelta().y() > 0 else 1 / _ZOOM_STEP
        new_scale = self._scale * factor
        if _MIN_SCALE <= new_scale <= _MAX_SCALE:
            self.scale(factor, factor)
            self._scale = new_scale

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override
        menu = QMenu(self)
        scene_pos = self.mapToScene(event.pos())
        for op_id in self.node_graph_scene.registry.ids():
            action = menu.addAction(op_id)
            action.triggered.connect(
                lambda checked=False, oid=op_id, pos=scene_pos: self.add_node_at(oid, pos)
            )
        menu.exec(event.globalPos())
