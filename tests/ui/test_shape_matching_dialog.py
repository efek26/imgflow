import cv2
import numpy as np
from PySide6.QtWidgets import QInputDialog, QMessageBox

from imgflow.ui.dialogs.shape_matching_dialog import _GUIDE_TEXT, ShapeMatchingDialog

_BASE_TRIANGLE = np.array([[0, -40], [35, 25], [-20, 30]], dtype=np.float64)


def _draw_triangle(image: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (_BASE_TRIANGLE @ rot.T) + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)


def _reference_image() -> np.ndarray:
    image = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(image, (100, 100), 0.0)
    return image


def _load_reference_directly(dialog: ShapeMatchingDialog, image: np.ndarray, roi=(50, 50, 100, 100)) -> None:
    """`QFileDialog.getOpenFileNames` gerçek bir dosya seçici açar, headless testte kullanılamaz
    — `_on_load_references`'ın diskten yükleme kısmını atlayıp aynı state'i doğrudan kuruyoruz."""
    dialog._reference_image = image
    dialog._canvas.set_image(image)
    dialog._roi = roi
    dialog._canvas.set_roi(*roi)


def test_guide_label_present_and_nonempty(qtbot):
    dialog = ShapeMatchingDialog(model_dir=None, parent=None)
    qtbot.addWidget(dialog)

    labels = [w for w in dialog.findChildren(type(dialog._train_status_label)) if w.text() == _GUIDE_TEXT]
    assert len(labels) == 1


def test_train_without_reference_image_shows_warning(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_train()

    assert len(warnings) == 1
    assert dialog.model is None


def test_train_produces_model_and_enables_save(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())

    dialog._on_train()

    assert dialog.model is not None
    assert dialog._save_button.isEnabled()


def test_train_on_blank_roi_shows_error_status(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    blank = np.full((200, 200), 255, dtype=np.uint8)
    _load_reference_directly(dialog, blank)

    dialog._on_train()

    assert dialog.model is None
    assert "başarısız" in dialog._train_status_label.text()


def test_reference_file_dropped_loads_image_same_as_open_dialog(qtbot, tmp_path):
    """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) canvas'a sürükleyip
    bırakmak `_on_load_references` (dosya seçici) ile AYNI sonucu üretmeli."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = _reference_image()
    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), image)

    dialog._on_reference_file_dropped(str(path))

    assert dialog._reference_image is not None
    assert np.array_equal(dialog._reference_image, image)


def test_reference_file_dropped_with_missing_file_shows_error(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    dialog._on_reference_file_dropped(str(tmp_path / "does_not_exist.png"))

    assert len(errors) == 1
    assert dialog._reference_image is None


def test_load_references_adds_multiple_files_to_gallery(qtbot, monkeypatch, tmp_path):
    """Kullanıcı isteği: 'model eğit kısmına daha fazla fotoğraf yüklenebilmeli' — birden
    fazla dosya TEK diyalog çağrısında galeriye eklenebilmeli, en son yüklenen otomatik aktif
    referans olmalı."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image_a, image_b = _reference_image(), _reference_image()
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(path_a), image_a)
    cv2.imwrite(str(path_b), image_b)
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getOpenFileNames",
        lambda *a, **k: ([str(path_a), str(path_b)], ""),
    )

    dialog._on_load_references()

    assert dialog._reference_list.count() == 2
    assert len(dialog._reference_images) == 2
    assert np.array_equal(dialog._reference_image, image_b)  # sonuncusu aktif referans olur


def test_active_reference_is_marked_unambiguously_in_gallery(qtbot, tmp_path):
    """Kullanıcı isteği: 'model eğitme kısmında açının doğru alınması için bir tane referans
    seçelim' — birden fazla foto arasında hangisinin şu an eğitim referansı (dolayısıyla
    modelin 'açı=0' pozu) olduğu galeride '★' öneki ve ayrı bir etiketle AÇIKÇA görünmeli."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._add_reference_to_gallery(str(tmp_path / "first.png"), _reference_image())
    dialog._add_reference_to_gallery(str(tmp_path / "second.png"), _reference_image())

    dialog._activate_reference(0)

    assert dialog._reference_list.item(0).text().startswith("★ ")
    assert not dialog._reference_list.item(1).text().startswith("★ ")
    assert "first.png" in dialog._active_reference_label.text()

    dialog._activate_reference(1)

    assert not dialog._reference_list.item(0).text().startswith("★ ")
    assert dialog._reference_list.item(1).text().startswith("★ ")
    assert "second.png" in dialog._active_reference_label.text()


def test_clicking_gallery_item_switches_active_reference(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    first = _reference_image()
    second = np.full((200, 200), 0, dtype=np.uint8)
    dialog._add_reference_to_gallery(str(tmp_path / "first.png"), first)
    dialog._add_reference_to_gallery(str(tmp_path / "second.png"), second)
    dialog._use_reference_image(second)  # ikincisi yüklendikten sonra aktif oldu

    dialog._on_reference_gallery_item_clicked(dialog._reference_list.item(0))

    assert np.array_equal(dialog._reference_image, first)


def test_load_references_with_missing_file_shows_error_and_skips(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getOpenFileNames",
        lambda *a, **k: ([str(tmp_path / "does_not_exist.png")], ""),
    )

    dialog._on_load_references()

    assert len(errors) == 1
    assert dialog._reference_list.count() == 0
    assert dialog._reference_image is None


def test_save_without_name_shows_warning(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._on_train()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_save()

    assert len(warnings) == 1


def test_save_and_load_round_trip_via_combo(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._on_train()
    dialog._name_edit.setText("ucgen_modeli")

    dialog._on_save()

    assert "Kaydedildi" in dialog._save_status_label.text()

    other_dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(other_dialog)
    assert other_dialog._load_combo.findText("ucgen_modeli") >= 0

    other_dialog._load_combo.setCurrentText("ucgen_modeli")
    other_dialog._on_load_model()

    assert other_dialog.model is not None
    assert "Yüklendi" in other_dialog._save_status_label.text()


def _train_and_save(dialog, name, monkeypatch=None):
    _load_reference_directly(dialog, _reference_image())
    dialog._on_train()
    dialog._name_edit.setText(name)
    dialog._on_save()


def test_delete_model_removes_it_from_combo_and_emits_models_changed(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "silinecek_model")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    changed = []
    dialog.models_changed.connect(lambda: changed.append(1))

    dialog._load_combo.setCurrentText("silinecek_model")
    dialog._on_delete_model()

    assert dialog._load_combo.findText("silinecek_model") == -1
    assert "Silindi" in dialog._save_status_label.text()
    assert changed == [1]
    assert dialog.model is None
    assert not dialog._save_button.isEnabled()


def test_delete_model_cancelled_keeps_it(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "kalacak_model")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    dialog._load_combo.setCurrentText("kalacak_model")
    dialog._on_delete_model()

    assert dialog._load_combo.findText("kalacak_model") >= 0


def test_rename_model_updates_combo_and_emits_models_changed(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "eski_isim")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("yeni_isim", True))
    changed = []
    dialog.models_changed.connect(lambda: changed.append(1))

    dialog._load_combo.setCurrentText("eski_isim")
    dialog._on_rename_model()

    assert dialog._load_combo.findText("yeni_isim") >= 0
    assert dialog._load_combo.findText("eski_isim") == -1
    assert changed == [1]
    assert dialog._name_edit.text() == "yeni_isim"


def test_rename_model_to_existing_name_shows_error(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "model_a")
    dialog._name_edit.clear()
    _train_and_save(dialog, "model_b")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("model_b", True))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    dialog._load_combo.setCurrentText("model_a")
    dialog._on_rename_model()

    assert len(errors) == 1
    assert dialog._load_combo.findText("model_a") >= 0
    assert dialog._load_combo.findText("model_b") >= 0
