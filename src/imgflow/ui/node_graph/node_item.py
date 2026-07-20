"""Bir pipeline node'unu temsil eden sürüklenebilir kutu; portlarını op_cls'e göre üretir."""

from __future__ import annotations

from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsSimpleTextItem

from imgflow.core.graph import Node
from imgflow.ui.node_graph.port_item import PortItem

WIDTH = 160.0
_TITLE_HEIGHT = 28.0
_ROW_HEIGHT = 20.0
_PADDING = 10.0


class NodeItem(QGraphicsRectItem):
    def __init__(self, node: Node, op_cls: type) -> None:
        rows = max(len(op_cls.inputs), len(op_cls.outputs), 1)
        height = _TITLE_HEIGHT + rows * _ROW_HEIGHT + _PADDING
        super().__init__(0, 0, WIDTH, height)

        self.node_id = node.node_id
        self.op_id = node.op_id

        self.setBrush(QBrush(QColor("#2d2d2d")))
        self.setPen(QPen(QColor("#555555"), 1))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setPos(*node.position)

        title = QGraphicsSimpleTextItem(f"{node.node_id}\n({node.op_id})", self)
        title.setBrush(QBrush(QColor("#eeeeee")))
        font = QFont()
        font.setPointSize(8)
        title.setFont(font)
        title.setPos(_PADDING * 0.5, 2)

        self.input_ports: dict[str, PortItem] = {}
        self.output_ports: dict[str, PortItem] = {}

        for i, spec in enumerate(op_cls.inputs):
            port = PortItem(self, spec, is_output=False)
            port.setPos(0, _TITLE_HEIGHT + i * _ROW_HEIGHT + _ROW_HEIGHT / 2)
            self.input_ports[spec.name] = port

        for i, spec in enumerate(op_cls.outputs):
            port = PortItem(self, spec, is_output=True)
            port.setPos(WIDTH, _TITLE_HEIGHT + i * _ROW_HEIGHT + _ROW_HEIGHT / 2)
            self.output_ports[spec.name] = port

    def itemChange(self, change, value):  # noqa: N802 - Qt override
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            scene = self.scene()
            if scene is not None:
                scene.on_node_moved(self)
        return super().itemChange(change, value)
