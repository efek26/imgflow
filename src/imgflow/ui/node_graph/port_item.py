"""Node üzerindeki tek bir giriş/çıkış portunu temsil eden görsel öğe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem

from imgflow.core.types import PortSpec, PortType

if TYPE_CHECKING:
    from imgflow.ui.node_graph.edge_item import EdgeItem
    from imgflow.ui.node_graph.node_item import NodeItem

PORT_RADIUS = 6.0

_PORT_COLORS = {
    PortType.IMAGE: QColor("#4fa3f7"),
    PortType.ROI: QColor("#f7b84f"),
    PortType.SCALAR: QColor("#7fe08a"),
    PortType.SCALAR_LIST: QColor("#7fe0c8"),
    PortType.MEASUREMENTS: QColor("#d17ff0"),
}


class PortItem(QGraphicsEllipseItem):
    def __init__(self, node_item: "NodeItem", spec: PortSpec, is_output: bool) -> None:
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, PORT_RADIUS * 2, PORT_RADIUS * 2, node_item)
        self.node_item = node_item
        self.spec = spec
        self.is_output = is_output
        self.edges: list["EdgeItem"] = []

        color = _PORT_COLORS.get(spec.type, QColor("#cccccc"))
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor("#1a1a1a"), 1))
        self.setToolTip(f"{spec.name} ({spec.type.value})")

    @property
    def node_id(self) -> str:
        return self.node_item.node_id

    def scene_center(self):
        return self.mapToScene(self.rect().center())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        scene = self.scene()
        if scene is not None:
            scene.begin_connection(self, event.scenePos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        scene = self.scene()
        if scene is not None:
            scene.update_connection(event.scenePos())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        scene = self.scene()
        if scene is not None:
            scene.end_connection(event.scenePos())
        event.accept()
