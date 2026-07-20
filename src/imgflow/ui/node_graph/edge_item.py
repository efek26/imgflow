"""İki port arasındaki kalıcı bağlantıyı bezier eğrisi olarak çizen görsel öğe."""

from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem

from imgflow.ui.node_graph.port_item import PortItem

_EDGE_COLOR = QColor("#dddddd")
_EDGE_WIDTH = 2.0


class EdgeItem(QGraphicsPathItem):
    def __init__(self, src_port: PortItem, dst_port: PortItem) -> None:
        super().__init__()
        self.src_port = src_port
        self.dst_port = dst_port
        self.setPen(QPen(_EDGE_COLOR, _EDGE_WIDTH))
        self.setZValue(-1)
        src_port.edges.append(self)
        dst_port.edges.append(self)
        self.update_path()

    def update_path(self) -> None:
        start = self.src_port.scene_center()
        end = self.dst_port.scene_center()
        dx = max(abs(end.x() - start.x()) * 0.5, 40.0)
        path = QPainterPath(start)
        path.cubicTo(QPointF(start.x() + dx, start.y()), QPointF(end.x() - dx, end.y()), end)
        self.setPath(path)

    def detach(self) -> None:
        if self in self.src_port.edges:
            self.src_port.edges.remove(self)
        if self in self.dst_port.edges:
            self.dst_port.edges.remove(self)
