import cv2
import numpy as np

from imgflow.ui.main_window import MainWindow


def _sample_image_path(tmp_path):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return path


def test_add_operator_appends_and_selects_node(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    node_id = window.add_operator("io.image_source")

    assert node_id in window.pipeline.graph.nodes
    assert window.pipeline.order == [node_id]
    assert window.pipeline_panel.selected_node_id() == node_id


def test_full_pipeline_updates_preview_on_param_change(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    gray_id = window.add_operator("color.convert")
    th_id = window.add_operator("segment.threshold")

    window._on_node_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})

    window._on_node_selected(th_id)
    window.param_form.params_changed.emit(
        {"value": 100, "max_value": 255, "mode": "BINARY"}
    )

    assert window.image_view._pixmap is not None
    assert window.status_label.text() == ""
    assert gray_id in window.pipeline.graph.nodes


def test_remove_operator_clears_preview_when_selected_node_removed(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window.param_form.params_changed.emit({"path": str(path)})

    window.remove_operator(src_id)

    assert src_id not in window.pipeline.graph.nodes
    assert window.image_view.text() in ("Önizleme yok", "")


def test_invalid_source_path_reports_error_in_status(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    src_id = window.add_operator("io.image_source")
    window._on_node_selected(src_id)
    window.param_form.params_changed.emit({"path": str(tmp_path / "yok.png")})

    assert window.status_label.text().startswith("Hata:")
