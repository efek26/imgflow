"""Görsel node-graph sahnesi: Graph modelini QGraphicsScene ile senkronize tutar.

Bu sahne, altındaki Graph'ın node/edge kümesine tam sahiptir: add_node/remove_node/
connect_ports/disconnect_edge_item dışında Graph'a doğrudan müdahale beklenmez.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, Signal
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsScene

from imgflow.core.graph import Edge, Graph, Node
from imgflow.core.types import PortSpec
from imgflow.ui.node_graph.edge_item import EdgeItem
from imgflow.ui.node_graph.node_item import NodeItem
from imgflow.ui.node_graph.port_item import PortItem

_TEMP_LINE_COLOR = QColor("#dddddd")


def ports_compatible(a: PortSpec, b: PortSpec) -> bool:
    return a.type is b.type


class NodeGraphScene(QGraphicsScene):
    node_selected = Signal(str)
    graph_changed = Signal()

    def __init__(self, graph: Graph, registry: Any, parent=None) -> None:
        super().__init__(parent)
        self.graph = graph
        self.registry = registry
        self.node_items: dict[str, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []
        self._pending_src_port: PortItem | None = None
        self._temp_line: QGraphicsPathItem | None = None
        self.selectionChanged.connect(self._on_selection_changed)

        for node in graph.nodes.values():
            self._create_node_item(node)
        for edge in graph.edges:
            self._create_edge_item(edge)

    # -- node/edge yönetimi -------------------------------------------------

    def generate_node_id(self, op_id: str) -> str:
        base = op_id.split(".")[-1]
        existing = set(self.graph.nodes)
        if base not in existing:
            return base
        counter = 2
        while f"{base}_{counter}" in existing:
            counter += 1
        return f"{base}_{counter}"

    def add_node(self, node: Node) -> NodeItem:
        self.graph.add_node(node)
        item = self._create_node_item(node)
        self.graph_changed.emit()
        return item

    def load_graph(self, new_graph: Graph) -> None:
        """Sahnenin tamamını new_graph'ın içeriğiyle değiştirir.

        Altındaki Graph nesnesinin kimliği korunur (nodes/edges yerinde değiştirilir),
        böylece bu sahneyle aynı Graph'ı paylaşan ExecutionEngine referansı geçersiz olmaz.
        """
        for item in list(self.node_items.values()):
            self.removeItem(item)
        for edge_item in list(self.edge_items):
            self.removeItem(edge_item)
        self.node_items.clear()
        self.edge_items.clear()

        self.graph.nodes = dict(new_graph.nodes)
        self.graph.edges = list(new_graph.edges)

        for node in self.graph.nodes.values():
            self._create_node_item(node)
        for edge in self.graph.edges:
            self._create_edge_item(edge)

        self.graph_changed.emit()

    def remove_node(self, node_id: str) -> None:
        item = self.node_items.pop(node_id, None)
        if item is None:
            return
        for edge_item in [
            e for e in self.edge_items if e.src_port.node_id == node_id or e.dst_port.node_id == node_id
        ]:
            self._destroy_edge_item(edge_item)
        self.removeItem(item)
        self.graph.remove_node(node_id)
        self.graph_changed.emit()

    def connect_ports(self, a: PortItem, b: PortItem) -> bool:
        if a.node_id == b.node_id or a.is_output == b.is_output:
            return False
        src, dst = (a, b) if a.is_output else (b, a)
        if not ports_compatible(src.spec, dst.spec):
            return False
        if dst.node_id in self.graph.ancestors_closure(src.node_id):
            return False

        for edge_item in [e for e in self.edge_items if e.dst_port is dst]:
            self._destroy_edge_item(edge_item)

        edge = Edge((src.node_id, src.spec.name), (dst.node_id, dst.spec.name))
        self.graph.add_edge(edge)
        self._create_edge_item(edge)
        self.graph_changed.emit()
        return True

    def disconnect_edge_item(self, edge_item: EdgeItem) -> None:
        self._destroy_edge_item(edge_item)
        self.graph_changed.emit()

    def on_node_moved(self, node_item: NodeItem) -> None:
        node = self.graph.nodes.get(node_item.node_id)
        if node is not None:
            pos = node_item.pos()
            node.position = (pos.x(), pos.y())
        for port in list(node_item.input_ports.values()) + list(node_item.output_ports.values()):
            for edge_item in list(port.edges):
                edge_item.update_path()

    # -- fare ile bağlantı sürükleme -----------------------------------------

    def begin_connection(self, port: PortItem, scene_pos: QPointF) -> None:
        self._pending_src_port = port
        self._temp_line = QGraphicsPathItem()
        self._temp_line.setPen(QPen(_TEMP_LINE_COLOR, 2))
        self._temp_line.setZValue(-1)
        self.addItem(self._temp_line)
        self._update_temp_line(scene_pos)

    def update_connection(self, scene_pos: QPointF) -> None:
        if self._temp_line is not None:
            self._update_temp_line(scene_pos)

    def end_connection(self, scene_pos: QPointF) -> None:
        if self._temp_line is not None:
            self.removeItem(self._temp_line)
            self._temp_line = None

        src_port = self._pending_src_port
        self._pending_src_port = None
        if src_port is None:
            return

        target = self._port_at(scene_pos)
        if target is not None and target is not src_port:
            self.connect_ports(src_port, target)

    def _update_temp_line(self, scene_pos: QPointF) -> None:
        start = self._pending_src_port.scene_center()
        path = QPainterPath(start)
        path.lineTo(scene_pos)
        self._temp_line.setPath(path)

    def _port_at(self, scene_pos: QPointF) -> PortItem | None:
        for item in self.items(scene_pos):
            if isinstance(item, PortItem):
                return item
        return None

    # -- yardımcılar ----------------------------------------------------

    def _create_node_item(self, node: Node) -> NodeItem:
        op_cls = self.registry.get(node.op_id)
        item = NodeItem(node, op_cls)
        self.addItem(item)
        self.node_items[node.node_id] = item
        return item

    def _create_edge_item(self, edge: Edge) -> EdgeItem:
        src_item = self.node_items[edge.src[0]]
        dst_item = self.node_items[edge.dst[0]]
        src_port = src_item.output_ports[edge.src[1]]
        dst_port = dst_item.input_ports[edge.dst[1]]
        edge_item = EdgeItem(src_port, dst_port)
        self.addItem(edge_item)
        self.edge_items.append(edge_item)
        return edge_item

    def _destroy_edge_item(self, edge_item: EdgeItem) -> None:
        edge_item.detach()
        if edge_item in self.edge_items:
            self.edge_items.remove(edge_item)
        self.removeItem(edge_item)
        target = (
            (edge_item.src_port.node_id, edge_item.src_port.spec.name),
            (edge_item.dst_port.node_id, edge_item.dst_port.spec.name),
        )
        self.graph.edges = [e for e in self.graph.edges if (e.src, e.dst) != target]

    def _on_selection_changed(self) -> None:
        selected = [item for item in self.selectedItems() if isinstance(item, NodeItem)]
        if selected:
            self.node_selected.emit(selected[0].node_id)
