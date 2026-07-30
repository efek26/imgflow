"""Aktif pipeline adımlarını sıralı liste olarak gösteren panel; sürükle-bırak ile yeniden sıralanabilir."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QAbstractItemView, QListWidget

from imgflow.core.linear_pipeline import LinearPipeline

_NODE_ID_ROLE = Qt.ItemDataRole.UserRole


class PipelineStepsPanel(QListWidget):
    step_selected = Signal(str)
    order_changed = Signal()
    reordered = Signal(list)  # previous_order (str listesi) - sürükle-bırak ile yeniden sıralamadan HEMEN önceki sıra

    def __init__(self, pipeline: LinearPipeline, parent=None) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.currentItemChanged.connect(self._on_current_changed)
        self.refresh()

    def refresh(self) -> None:
        self.blockSignals(True)
        self.clear()
        for index, node_id in enumerate(self.pipeline.order, start=1):
            node = self.pipeline.graph.nodes[node_id]
            self.addItem(f"{index}. {node.op_id}")
            self.item(self.count() - 1).setData(_NODE_ID_ROLE, node_id)
        self.blockSignals(False)

    def selected_node_id(self) -> str | None:
        item = self.currentItem()
        return item.data(_NODE_ID_ROLE) if item is not None else None

    def select_node(self, node_id: str) -> None:
        for i in range(self.count()):
            if self.item(i).data(_NODE_ID_ROLE) == node_id:
                self.setCurrentRow(i)
                return

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        previous_order = list(self.pipeline.order)
        super().dropEvent(event)  # Qt satırları taşır
        self._apply_dropped_row_order(previous_order)

    def _apply_dropped_row_order(self, previous_order: list[str]) -> None:
        """Qt satırları yeniden sıraladıktan SONRA çalışan kısım: yeni satır sırasını
        pipeline'a yazar ve GERÇEKTEN değiştiyse sinyalleri yayar. `dropEvent`'ten ayrı bir
        metot olmasının sebebi, gerçek bir `QDropEvent` üretmenin headless testte zahmetli
        olması -- testler bu kısmı doğrudan çağırabilir (bkz. `tests/ui/test_pipeline_steps.py`).
        Sıra DEĞİŞMEDİYSE (kullanıcı öğeyi aynı yere bıraktı) hiçbir sinyal yayılmaz, aksi
        halde boş bir undo adımı kaydedilir ve pipeline gereksiz yere baştan hesaplanırdı."""
        new_order = [self.item(i).data(_NODE_ID_ROLE) for i in range(self.count())]
        self.pipeline.set_order(new_order)
        self.refresh()
        if new_order == previous_order:
            return
        self.reordered.emit(previous_order)
        self.order_changed.emit()

    def _on_current_changed(self, current, previous) -> None:
        if current is None:
            return
        self.step_selected.emit(current.data(_NODE_ID_ROLE))
