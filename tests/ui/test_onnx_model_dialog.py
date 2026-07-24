from PySide6.QtWidgets import QMessageBox

from imgflow.io_utils import onnx_model_store
from imgflow.ui.dialogs.onnx_model_dialog import _GUIDE_TEXT, OnnxModelDialog


def _fake_onnx_file(tmp_path):
    path = tmp_path / "source.onnx"
    path.write_bytes(b"fake-onnx-bytes")
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


def test_placeholder_task_type_can_be_saved(qtbot, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._selected_onnx_path = _fake_onnx_file(tmp_path)
    dialog._name_edit.setText("siniflandirici")
    dialog._class_labels_edit.setText("ok, ng")
    dialog._task_type_combo.setCurrentText("Sınıflandırma / Anomali (yakında)")

    dialog._on_save()

    meta = onnx_model_store.load_model_meta("siniflandirici", directory=tmp_path)
    assert meta["task_type"] == "classification"


def test_delete_without_selection_is_noop(qtbot, tmp_path):
    dialog = OnnxModelDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)

    dialog._on_delete()  # combo boş, hiçbir şey olmamalı — exception fırlatmamalı
