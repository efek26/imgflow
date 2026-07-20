"""imgflow ana penceresi: görsel node-graph canvası + parametre paneli + önizleme.

Canvas, Graph modelini doğrudan (LinearPipeline'ın zorladığı doğrusal zincir kısıtı
olmadan) düzenler: node'lar sürüklenip taşınabilir, portlar fare ile birbirine
bağlanabilir. Parametre değişiklikleri motorun dirty/cache mekanizmasıyla yalnızca
gerekli node'ları yeniden hesaplattırır.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.engine import ExecutionEngine
from imgflow.core.graph import Graph, Node
from imgflow.core.params import defaults_for
from imgflow.operators import registry as default_registry
from imgflow.ui.node_graph.scene import NodeGraphScene
from imgflow.ui.node_graph.view import NodeGraphView
from imgflow.ui.widgets.image_view import ImageView, extract_preview_image
from imgflow.ui.widgets.param_form import ParamForm

_NEW_NODE_STEP_X = 190.0
_NEW_NODE_STEP_Y = 130.0
_NEW_NODE_COLUMNS = 4


class MainWindow(QMainWindow):
    def __init__(self, registry: Any = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("imgflow")
        self.registry = registry or default_registry

        self.graph = Graph()
        self.scene = NodeGraphScene(self.graph, self.registry)
        self.engine = ExecutionEngine(self.graph, registry=self.registry)
        self._selected_node_id: str | None = None

        self.node_graph_view = NodeGraphView(self.scene)
        self.param_form = ParamForm()
        self.image_view = ImageView()
        self.status_label = QLabel("")

        self.scene.node_selected.connect(self._on_node_selected)
        self.scene.graph_changed.connect(self._on_graph_changed)
        self.param_form.params_changed.connect(self._on_params_changed)

        self._build_layout()

    def _build_layout(self) -> None:
        add_button = QPushButton("Operatör Ekle...")
        add_button.clicked.connect(self._on_add_operator)
        remove_button = QPushButton("Seçileni Sil")
        remove_button.clicked.connect(self._on_remove_operator)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.node_graph_view)
        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        left_layout.addLayout(button_row)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Parametreler"))
        right_layout.addWidget(self.param_form)
        right_layout.addWidget(QLabel("Önizleme"))
        right_layout.addWidget(self.image_view, stretch=1)
        right_layout.addWidget(self.status_label)

        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.addWidget(left, stretch=2)
        central_layout.addWidget(right, stretch=1)
        self.setCentralWidget(central)

    # -- public API (dialogsuz, testlerden de çağrılabilir) -----------------

    def add_operator(self, op_id: str) -> str:
        op_cls = self.registry.get(op_id)
        node_id = self.scene.generate_node_id(op_id)
        index = len(self.graph.nodes)
        position = (
            (index % _NEW_NODE_COLUMNS) * _NEW_NODE_STEP_X + 40,
            (index // _NEW_NODE_COLUMNS) * _NEW_NODE_STEP_Y + 40,
        )
        item = self.scene.add_node(
            Node(node_id, op_id, params=defaults_for(op_cls.params), position=position)
        )
        item.setSelected(True)
        return node_id

    def remove_operator(self, node_id: str) -> None:
        self.scene.remove_node(node_id)
        if self._selected_node_id == node_id:
            self._selected_node_id = None
            self.param_form.set_params([], {})
            self._refresh_preview()

    def selected_node_id(self) -> str | None:
        ids = self.node_graph_view.selected_node_ids()
        return ids[0] if ids else None

    # -- Qt slot'ları -----------------------------------------------------

    def _on_add_operator(self) -> None:
        op_ids = self.registry.ids()
        if not op_ids:
            return
        op_id, ok = QInputDialog.getItem(self, "Operatör Ekle", "Operatör:", op_ids, editable=False)
        if ok and op_id:
            self.add_operator(op_id)

    def _on_remove_operator(self) -> None:
        node_id = self.selected_node_id()
        if node_id:
            self.remove_operator(node_id)

    def _on_graph_changed(self) -> None:
        self.engine.mark_all_dirty()
        self._refresh_preview()

    def _on_node_selected(self, node_id: str) -> None:
        self._selected_node_id = node_id
        node = self.graph.nodes[node_id]
        op_cls = self.registry.get(node.op_id)
        self.param_form.set_params(op_cls.params, node.params)
        self._refresh_preview()

    def _on_params_changed(self, values: dict[str, Any]) -> None:
        if self._selected_node_id is None:
            return
        self.graph.nodes[self._selected_node_id].params = values
        self.engine.mark_dirty(self._selected_node_id)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            self.image_view.set_image(None)
            self.status_label.setText("")
            return

        result = self.engine.evaluate(node_id)
        if not result.ok:
            self.image_view.set_image(None)
            self.status_label.setText(f"Hata: {result.error}")
            return

        op_cls = self.registry.get(self.graph.nodes[node_id].op_id)
        image = extract_preview_image(op_cls, result.outputs)
        self.image_view.set_image(image)
        self.status_label.setText("" if image is not None else "Bu node için görsel çıktı yok.")
