import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox

from imgflow.ui.dialogs.flat_field_dialog import _GUIDE_TEXT, FlatFieldDialog


def _reference_image() -> np.ndarray:
    return np.full((40, 60), 180, dtype=np.uint8)


def test_guide_label_present_and_nonempty(qtbot):
    dialog = FlatFieldDialog(directory=None, parent=None)
    qtbot.addWidget(dialog)

    labels = [w for w in dialog.findChildren(type(dialog._status_label)) if w.text() == _GUIDE_TEXT]
    assert len(labels) == 1


def test_save_without_image_is_noop(qtbot, tmp_path):
    dialog = FlatFieldDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)

    dialog._on_save()  # görüntü yok, hiçbir şey olmamalı

    assert dialog._list_combo.count() == 0


def test_save_without_name_shows_warning(qtbot, monkeypatch, tmp_path):
    dialog = FlatFieldDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._image = _reference_image()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_save()

    assert len(warnings) == 1


def test_file_dropped_loads_image_same_as_open_dialog(qtbot, tmp_path):
    dialog = FlatFieldDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = _reference_image()
    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), image)

    dialog._on_file_dropped(str(path))

    assert dialog._image is not None
    assert np.array_equal(dialog._image, image)
    assert dialog._save_button.isEnabled()


def test_file_dropped_with_missing_file_shows_error(qtbot, monkeypatch, tmp_path):
    dialog = FlatFieldDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    dialog._on_file_dropped(str(tmp_path / "does_not_exist.png"))

    assert len(errors) == 1
    assert dialog._image is None


def test_save_and_delete_round_trip(qtbot, monkeypatch, tmp_path):
    dialog = FlatFieldDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._image = _reference_image()
    dialog._name_edit.setText("hat1")
    changed = []
    dialog.references_changed.connect(lambda: changed.append(True))

    dialog._on_save()

    assert dialog._list_combo.count() == 1
    assert dialog._list_combo.itemText(0) == "hat1"
    assert len(changed) == 1

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dialog._on_delete()

    assert dialog._list_combo.count() == 0
    assert len(changed) == 2


def test_delete_without_selection_is_noop(qtbot, tmp_path):
    dialog = FlatFieldDialog(directory=tmp_path, parent=None)
    qtbot.addWidget(dialog)

    dialog._on_delete()  # combo boş, hiçbir şey olmamalı — exception fırlatmamalı
