"""imgflow ana penceresi: pipeline listesi + parametre paneli + önizleme.

v1 arayüzü kutu-tel bir node-graph editörü değil; LinearPipeline'ın sıralı listesini
gösterir (bkz. core/linear_pipeline.py). Parametre değişiklikleri motorun dirty/cache
mekanizmasıyla yalnızca gerekli node'ları yeniden hesaplattırır.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
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
from imgflow.core.graph import Node
from imgflow.core.linear_pipeline import LinearPipeline
from imgflow.core.params import defaults_for
from imgflow.operators import registry as default_registry
from imgflow.ui.panels.pipeline_panel import PipelinePanel
from imgflow.ui.widgets.image_view import ImageView, extract_preview_image
from imgflow.ui.widgets.param_form import ParamForm


class MainWindow(QMainWindow):
    def __init__(self, registry: Any = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("imgflow")
        self.registry = registry or default_registry

        self.pipeline = LinearPipeline(self.registry)
        self.engine = ExecutionEngine(self.pipeline.graph, registry=self.registry)
        self._selected_node_id: str | None = None

        self.pipeline_panel = PipelinePanel(self.pipeline)
        self.param_form = ParamForm()
        self.image_view = ImageView()
        self.status_label = QLabel("")

        self.pipeline_panel.node_selected.connect(self._on_node_selected)
        self.pipeline_panel.order_changed.connect(self._on_pipeline_changed)
        self.param_form.params_changed.connect(self._on_params_changed)

        self._build_layout()

    def _build_layout(self) -> None:
        add_button = QPushButton("Operatör Ekle...")
        add_button.clicked.connect(self._on_add_operator)
        remove_button = QPushButton("Seçileni Sil")
        remove_button.clicked.connect(self._on_remove_operator)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.pipeline_panel)
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
        central_layout.addWidget(left, stretch=1)
        central_layout.addWidget(right, stretch=2)
        self.setCentralWidget(central)

    # -- public API (dialogsuz, testlerden de çağrılabilir) -----------------

    def add_operator(self, op_id: str) -> str:
        op_cls = self.registry.get(op_id)
        node_id = self._generate_node_id(op_id)
        self.pipeline.append(Node(node_id, op_id, params=defaults_for(op_cls.params)))
        self.pipeline_panel.refresh()
        self.engine.mark_all_dirty()
        self._select_row_for(node_id)
        return node_id

    def remove_operator(self, node_id: str) -> None:
        self.pipeline.remove(node_id)
        self.pipeline_panel.refresh()
        self.engine.mark_all_dirty()
        if self._selected_node_id == node_id:
            self._selected_node_id = None
            self.param_form.set_params([], {})
            self._refresh_preview()
        if self.pipeline.order:
            self._select_row_for(self.pipeline.order[-1])

    # -- yardımcılar ----------------------------------------------------

    def _generate_node_id(self, op_id: str) -> str:
        base = op_id.split(".")[-1]
        existing = set(self.pipeline.graph.nodes)
        if base not in existing:
            return base
        counter = 2
        while f"{base}_{counter}" in existing:
            counter += 1
        return f"{base}_{counter}"

    def _select_row_for(self, node_id: str) -> None:
        for i in range(self.pipeline_panel.count()):
            if self.pipeline_panel.item(i).data(Qt.ItemDataRole.UserRole) == node_id:
                self.pipeline_panel.setCurrentRow(i)
                return

    # -- Qt slot'ları -----------------------------------------------------

    def _on_add_operator(self) -> None:
        op_ids = self.registry.ids()
        if not op_ids:
            return
        op_id, ok = QInputDialog.getItem(self, "Operatör Ekle", "Operatör:", op_ids, editable=False)
        if ok and op_id:
            self.add_operator(op_id)

    def _on_remove_operator(self) -> None:
        node_id = self.pipeline_panel.selected_node_id()
        if node_id:
            self.remove_operator(node_id)

    def _on_pipeline_changed(self) -> None:
        self.engine.mark_all_dirty()
        self._refresh_preview()

    def _on_node_selected(self, node_id: str) -> None:
        self._selected_node_id = node_id
        node = self.pipeline.graph.nodes[node_id]
        op_cls = self.registry.get(node.op_id)
        self.param_form.set_params(op_cls.params, node.params)
        self._refresh_preview()

    def _on_params_changed(self, values: dict[str, Any]) -> None:
        if self._selected_node_id is None:
            return
        self.pipeline.graph.nodes[self._selected_node_id].params = values
        self.engine.mark_dirty(self._selected_node_id)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.pipeline.graph.nodes:
            self.image_view.set_image(None)
            self.status_label.setText("")
            return

        result = self.engine.evaluate(node_id)
        if not result.ok:
            self.image_view.set_image(None)
            self.status_label.setText(f"Hata: {result.error}")
            return

        op_cls = self.registry.get(self.pipeline.graph.nodes[node_id].op_id)
        image = extract_preview_image(op_cls, result.outputs)
        self.image_view.set_image(image)
        self.status_label.setText("" if image is not None else "Bu node için görsel çıktı yok.")
