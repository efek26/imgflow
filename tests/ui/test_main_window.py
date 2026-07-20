import cv2
import numpy as np
import pytest

from imgflow.core.errors import ImgflowError
from imgflow.ui.main_window import MainWindow


def _sample_image_path(tmp_path):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return path


def _connect(window, src_id, src_port, dst_id, dst_port):
    src = window.scene.node_items[src_id].output_ports[src_port]
    dst = window.scene.node_items[dst_id].input_ports[dst_port]
    assert window.scene.connect_ports(src, dst)


def test_add_operator_appends_and_selects_node(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    node_id = window.add_operator("io.image_source")

    assert node_id in window.graph.nodes
    assert window.selected_node_id() == node_id


def test_full_pipeline_updates_preview_on_param_change(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    gray_id = window.add_operator("color.convert")
    th_id = window.add_operator("segment.threshold")
    _connect(window, src_id, "image", gray_id, "image")
    _connect(window, gray_id, "image", th_id, "image")

    window._on_node_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})

    window._on_node_selected(th_id)
    window.param_form.params_changed.emit({"value": 100, "max_value": 255, "mode": "BINARY"})

    assert window.image_view._pixmap is not None
    assert window.status_label.text() == ""
    assert gray_id in window.graph.nodes
    assert len(window.graph.edges) == 2


def test_remove_operator_clears_preview_when_selected_node_removed(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window.param_form.params_changed.emit({"path": str(path)})

    window.remove_operator(src_id)

    assert src_id not in window.graph.nodes
    assert window.image_view.text() in ("Önizleme yok", "")


def test_invalid_source_path_reports_error_in_status(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    src_id = window.add_operator("io.image_source")
    window._on_node_selected(src_id)
    window.param_form.params_changed.emit({"path": str(tmp_path / "yok.png")})

    assert window.status_label.text().startswith("Hata:")


def test_connecting_incompatible_ports_is_rejected(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    src_id = window.add_operator("segment.connected_components")
    dst_id = window.add_operator("analysis.region_props")

    count_port = window.scene.node_items[src_id].output_ports["count"]
    labels_in = window.scene.node_items[dst_id].input_ports["labels"]

    assert window.scene.connect_ports(count_port, labels_in) is False
    assert window.graph.edges == []


def test_save_and_load_recipe_roundtrip_preserves_pipeline(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    src_id = window.add_operator("io.image_source")
    gray_id = window.add_operator("color.convert")
    _connect(window, src_id, "image", gray_id, "image")
    window._on_node_selected(gray_id)
    window.param_form.params_changed.emit({"mode": "BGR2HSV"})

    recipe_path = tmp_path / "recipe.json"
    window.save_recipe_to(str(recipe_path))

    reloaded = MainWindow()
    qtbot.addWidget(reloaded)
    reloaded.load_recipe_from(str(recipe_path))

    assert set(reloaded.graph.nodes) == {src_id, gray_id}
    assert reloaded.graph.nodes[gray_id].params == {"mode": "BGR2HSV"}
    assert len(reloaded.graph.edges) == 1
    # motorun bağlı olduğu Graph referansı korunuyor mu
    assert reloaded.engine.graph is reloaded.graph


def test_load_recipe_with_unknown_operator_raises(qtbot, tmp_path):
    recipe_path = tmp_path / "bad_recipe.json"
    recipe_path.write_text(
        '{"schema_version": 1, "nodes": '
        '[{"id": "x", "op_id": "no.such.operator", "params": {}, "position": [0, 0]}], "edges": []}',
        encoding="utf-8",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    with pytest.raises(ImgflowError):
        window.load_recipe_from(str(recipe_path))


def test_run_batch_process_writes_csv(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    src_id = window.add_operator("io.image_source")
    gray_id = window.add_operator("color.convert")
    th_id = window.add_operator("segment.threshold")
    cc_id = window.add_operator("segment.connected_components")
    props_id = window.add_operator("analysis.region_props")
    _connect(window, src_id, "image", gray_id, "image")
    _connect(window, gray_id, "image", th_id, "image")
    _connect(window, th_id, "image", cc_id, "image")
    _connect(window, cc_id, "labels", props_id, "labels")

    window._on_node_selected(th_id)
    window.param_form.params_changed.emit({"value": 100, "max_value": 255, "mode": "BINARY"})

    input_dir = tmp_path / "images"
    input_dir.mkdir()
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    cv2.imwrite(str(input_dir / "a.png"), img)

    output_csv = tmp_path / "out.csv"
    rows = window.run_batch_process(props_id, str(input_dir), str(output_csv))

    assert len(rows) == 1
    assert rows[0]["image"] == "a.png"
    assert output_csv.exists()


def test_roi_pipeline_masks_outside_region_end_to_end(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[1:3, 1:3] = 255  # ROI dışında gürültü
    img[10:14, 10:14] = 255  # ROI içinde asıl bölge
    path = tmp_path / "roi_sample.png"
    cv2.imwrite(str(path), img)

    src_id = window.add_operator("io.image_source")
    roi_id = window.add_operator("roi.rectangle")
    mask_id = window.add_operator("roi.mask")
    th_id = window.add_operator("segment.threshold")
    cc_id = window.add_operator("segment.connected_components")
    props_id = window.add_operator("analysis.region_props")

    # src.image iki tüketiciye dallanıyor (roi.rectangle'ın opsiyonel clamp girdisi + roi.mask'ın asıl görüntüsü)
    _connect(window, src_id, "image", roi_id, "image")
    _connect(window, src_id, "image", mask_id, "image")
    _connect(window, roi_id, "roi", mask_id, "roi")
    _connect(window, mask_id, "image", th_id, "image")
    _connect(window, th_id, "image", cc_id, "image")
    _connect(window, cc_id, "labels", props_id, "labels")

    window._on_node_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})

    window._on_node_selected(roi_id)
    window.param_form.params_changed.emit({"x": 8, "y": 8, "w": 10, "h": 10})

    window._on_node_selected(th_id)
    window.param_form.params_changed.emit({"value": 100, "max_value": 255, "mode": "BINARY"})

    result = window.engine.evaluate(props_id)

    assert result.ok
    measurements = result.outputs["measurements"]
    assert len(measurements) == 1
    assert measurements[0]["bbox_x"] == 10
    assert measurements[0]["bbox_y"] == 10
