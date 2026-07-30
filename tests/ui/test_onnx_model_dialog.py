import onnx
from onnx import TensorProto, helper
from PySide6.QtWidgets import QMessageBox

from imgflow.io_utils import onnx_model_store
from imgflow.ui.dialogs.onnx_model_dialog import _GUIDE_TEXT, OnnxModelDialog


def _fake_onnx_file(tmp_path):
    path = tmp_path / "source.onnx"
    path.write_bytes(b"fake-onnx-bytes")
    return path


def _real_onnx_file_with_metadata(tmp_path, input_size=320):
    """`core/onnx_detection.py::inspect_onnx_model`'in okuyabileceği, Ultralytics tarzı
    `"names"` metadata'sı gömülü GERÇEK (ama minik) bir ONNX dosyası -- diyalogun otomatik
    doldurma davranışını gerçekten çalıştırarak test etmek için."""
    path = tmp_path / "real_source.onnx"
    input_tensor = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, input_size, input_size]
    )
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 7])
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output"],
        value=helper.make_tensor(
            name="const_val", data_type=TensorProto.FLOAT, dims=[1, 1, 7], vals=[0.0] * 7
        ),
    )
    graph = helper.make_graph([const_node], "test_graph", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="imgflow-test")
    model.opset_import[0].version = 13
    model.metadata_props.append(onnx.StringStringEntryProto(key="names", value="{0: 'kusur', 1: 'saglam'}"))
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))
    return path


def test_guide_label_present_and_nonempty(qtbot):
    dialog = OnnxModelDialog(directory=None, parent=None)
    qtbot.addWidget(dialog)

    labels = [w for w in dialog.findChildren(type(dialog._status_label)) if w.text() == _GUIDE_TEXT]
    assert len(labels) == 1


def test_default_task_type_is_yolo(qtbot, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    assert dialog._task_type_combo.currentText() == "YOLO"


def test_save_without_file_is_noop(qtbot, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)

    dialog._on_save()

    assert dialog._list_combo.count() == 0


def test_save_without_name_shows_warning(qtbot, monkeypatch, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._selected_onnx_path = _fake_onnx_file(tmp_path)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_save()

    assert len(warnings) == 1
    assert dialog._list_combo.count() == 0


def test_save_without_class_labels_shows_warning(qtbot, monkeypatch, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._selected_onnx_path = _fake_onnx_file(tmp_path)
    dialog._name_edit.setText("model1")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_save()

    assert len(warnings) == 1


def test_save_and_delete_round_trip(qtbot, monkeypatch, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._selected_onnx_path = _fake_onnx_file(tmp_path)
    dialog._name_edit.setText("kusur_dedektoru")
    dialog._class_labels_edit.setText("kusur, saglam")
    dialog._input_size_spin.setValue(320)
    changed = []
    dialog.models_changed.connect(lambda: changed.append(True))

    dialog._on_save()

    assert dialog._list_combo.count() == 1
    assert dialog._list_combo.itemText(0) == "kusur_dedektoru"
    assert len(changed) == 1

    meta = onnx_model_store.load_model_meta("kusur_dedektoru", directory=tmp_path)
    assert meta["task_type"] == "yolo"
    assert meta["input_size"] == 320
    assert meta["class_labels"] == ["kusur", "saglam"]

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog._on_delete()

    assert dialog._list_combo.count() == 0
    assert len(changed) == 2


def test_classification_task_type_can_be_saved(qtbot, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._selected_onnx_path = _fake_onnx_file(tmp_path)
    dialog._name_edit.setText("siniflandirici")
    dialog._class_labels_edit.setText("ok, ng")
    dialog._task_type_combo.setCurrentText("Sınıflandırma / Anomali")

    dialog._on_save()

    meta = onnx_model_store.load_model_meta("siniflandirici", directory=tmp_path)
    assert meta["task_type"] == "classification"


def test_segmentation_task_type_can_be_saved(qtbot, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._selected_onnx_path = _fake_onnx_file(tmp_path)
    dialog._name_edit.setText("segmentleyici")
    dialog._class_labels_edit.setText("arka_plan, kusur")
    dialog._task_type_combo.setCurrentText("Segmentasyon")

    dialog._on_save()

    meta = onnx_model_store.load_model_meta("segmentleyici", directory=tmp_path)
    assert meta["task_type"] == "segmentation"


def test_delete_without_selection_is_noop(qtbot, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)

    dialog._on_delete()  # combo boş, hiçbir şey olmamalı — exception fırlatmamalı


def test_choose_file_autofills_class_labels_and_input_size_from_real_model(qtbot, monkeypatch, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    real_path = _real_onnx_file_with_metadata(tmp_path, input_size=320)
    monkeypatch.setattr(
        "imgflow.ui.dialogs.onnx_model_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(real_path), ""),
    )

    dialog._on_choose_file()

    assert dialog._class_labels_edit.text() == "kusur, saglam"
    assert dialog._input_size_spin.value() == 320
    assert "otomatik" in dialog._status_label.text().lower()
    assert dialog._save_button.isEnabled()


def test_choose_file_with_fake_bytes_leaves_fields_for_manual_entry(qtbot, monkeypatch, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    fake_path = _fake_onnx_file(tmp_path)
    monkeypatch.setattr(
        "imgflow.ui.dialogs.onnx_model_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(fake_path), ""),
    )

    dialog._on_choose_file()  # bozuk/gerçek olmayan dosyada ÇÖKMEMELİ

    assert dialog._class_labels_edit.text() == ""
    assert dialog._input_size_spin.value() == 640  # varsayılan, değişmedi
    assert "elle girin" in dialog._status_label.text().lower()
    assert dialog._save_button.isEnabled()
