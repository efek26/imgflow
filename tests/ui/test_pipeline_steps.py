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
