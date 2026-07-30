from PySide6.QtCore import Qt

from imgflow.core.graph import Node
from imgflow.core.linear_pipeline import LinearPipeline
from imgflow.operators import registry
from imgflow.ui.panels.pipeline_steps import PipelineStepsPanel


def _pipeline() -> LinearPipeline:
    pipeline = LinearPipeline(registry)
    pipeline.append(Node("src", "io.image_source"))
    pipeline.append(Node("gray", "color.grayscale"))
    pipeline.append(Node("th", "segment.threshold"))
    return pipeline


def test_refresh_populates_items_in_order(qtbot):
    panel = PipelineStepsPanel(_pipeline())
    qtbot.addWidget(panel)

    assert panel.count() == 3
    node_ids = [panel.item(i).data(Qt.ItemDataRole.UserRole) for i in range(3)]
    assert node_ids == ["src", "gray", "th"]


def test_selecting_item_emits_step_selected(qtbot):
    panel = PipelineStepsPanel(_pipeline())
    qtbot.addWidget(panel)

    with qtbot.waitSignal(panel.step_selected, timeout=1000) as blocker:
        panel.setCurrentRow(1)

    assert blocker.args == ["gray"]
    assert panel.selected_node_id() == "gray"


def test_select_node_sets_current_row(qtbot):
    panel = PipelineStepsPanel(_pipeline())
    qtbot.addWidget(panel)

    panel.select_node("th")

    assert panel.selected_node_id() == "th"


def test_refresh_reflects_pipeline_set_order(qtbot):
    pipeline = _pipeline()
    panel = PipelineStepsPanel(pipeline)
    qtbot.addWidget(panel)

    pipeline.set_order(["th", "src", "gray"])
    panel.refresh()

    node_ids = [panel.item(i).data(Qt.ItemDataRole.UserRole) for i in range(3)]
    assert node_ids == ["th", "src", "gray"]


def test_apply_dropped_row_order_emits_reordered_with_previous_order(qtbot):
    """Gerçek bir sürükle-bırak (`QDropEvent`) headless testte üretmesi zahmetli olduğu için
    `_apply_dropped_row_order` (Qt zaten satırları yeniden sıraladıktan SONRAKİ kısım)
    doğrudan çağrılır — bir satır elle taşınıp (Qt'nin gerçek dropEvent'inin bıraktığı son
    durumu simüle eder) yeni sıra ÖNCEKİ sıradan farklıysa `reordered` sinyali önceki
    sırayla yayınlanmalı."""
    pipeline = _pipeline()
    panel = PipelineStepsPanel(pipeline)
    qtbot.addWidget(panel)
    previous_order = list(pipeline.order)

    item = panel.takeItem(0)
    panel.insertItem(2, item)

    with qtbot.waitSignal(panel.reordered, timeout=1000) as blocker:
        panel._apply_dropped_row_order(previous_order)

    assert blocker.args == [previous_order]
    assert pipeline.order == ["gray", "th", "src"]


def test_apply_dropped_row_order_does_not_emit_reordered_when_order_unchanged(qtbot):
    pipeline = _pipeline()
    panel = PipelineStepsPanel(pipeline)
    qtbot.addWidget(panel)
    previous_order = list(pipeline.order)

    calls = []
    panel.reordered.connect(lambda order: calls.append(order))
    panel._apply_dropped_row_order(previous_order)

    assert calls == []
