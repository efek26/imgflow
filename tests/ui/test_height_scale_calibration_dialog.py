import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMessageBox

from imgflow.core import capture_store
from imgflow.core.lens_calibration import LensProfile
from imgflow.io_utils import calibration_store
from imgflow.ui.dialogs.height_scale_calibration_dialog import _GUIDE_TEXT, HeightScaleCalibrationDialog
from tests.core.test_lens_calibration import _charuco_board, _draw_checkerboard


@pytest.fixture(autouse=True)
def _isolated_capture_dir(tmp_path, monkeypatch):
    """`_on_capture_frame()` artık `capture_store.save_capture()` tetikliyor — gerçek
    `~/.imgflow/captures` dizinine ASLA dokunulmamalı (bkz. test_lens_calibration_dialog.py
    ile aynı desen)."""
    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")

_CHARUCO_BOARD = _charuco_board(squares_x=5, squares_y=7)
_CHARUCO_FRAME = _CHARUCO_BOARD.generateImage((600, 500))

_FRAME = np.zeros((100, 100, 3), dtype=np.uint8)
_BOARD_FRAME = _draw_checkerboard(9, 6, square_px=30, margin=30)
_SYNTHETIC_LENS_PROFILE = LensProfile(
    camera_matrix=np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]),
    dist_coeffs=np.zeros((5, 1)),
    image_size=(_BOARD_FRAME.shape[1], _BOARD_FRAME.shape[0]),
    rms_error=0.1,
)


def _add_point(dialog, qtbot, height_mm, real_mm, p1, p2, widget_size=(200, 200)):
    dialog._canvas.resize(*widget_size)
    dialog._on_capture_frame()
    dialog._height_spin.setValue(height_mm)
    dialog._real_mm_spin.setValue(real_mm)
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(*p1))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(*p2))
    dialog._on_add_point()


def test_capture_frame_loads_into_canvas(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    dialog._on_capture_frame()

    assert dialog._canvas._image_size == (100, 100)


def test_capture_persists_frame_and_emits_signal(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    received = []
    dialog.frame_captured.connect(lambda: received.append(1))
    dialog._on_capture_frame()

    assert received == [1]
    records = capture_store.list_captures(source="height_scale")
    assert len(records) == 1


def test_frame_file_dropped_loads_into_canvas_same_as_capture(qtbot, tmp_path):
    """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) canvas'a sürükleyip
    bırakmak `_on_capture_frame` (canlı kamera) ile AYNI sonucu üretmeli."""
    import cv2

    dialog = HeightScaleCalibrationDialog(lambda: None)
    qtbot.addWidget(dialog)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _FRAME)

    dialog._on_frame_file_dropped(str(path))

    assert dialog._canvas._image_size == (100, 100)
    assert dialog._captured_frame is not None
    records = capture_store.list_captures(source="height_scale")
    assert len(records) == 1


def test_frame_file_dropped_with_missing_file_shows_inline_status(qtbot, tmp_path):
    dialog = HeightScaleCalibrationDialog(lambda: None)
    qtbot.addWidget(dialog)

    dialog._on_frame_file_dropped(str(tmp_path / "does_not_exist.png"))

    assert dialog._captured_frame is None
    assert dialog._board_status_label.text() != ""


def test_guide_label_present_and_nonempty(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    labels = [w for w in dialog.findChildren(type(dialog._pixel_label)) if w.text() == _GUIDE_TEXT]
    assert len(labels) == 1


def test_measurement_enables_add_point_button(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)
    dialog._canvas.resize(200, 200)
    dialog._on_capture_frame()

    assert dialog._add_point_button.isEnabled() is False
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))

    assert dialog._add_point_button.isEnabled() is True


def test_add_point_appends_to_model_and_table(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    _add_point(dialog, qtbot, height_mm=50.0, real_mm=20.0, p1=(20, 20), p2=(60, 20))

    assert len(dialog.model.points) == 1
    assert dialog._points_table.rowCount() == 1
    assert dialog._add_point_button.isEnabled() is False


def test_fit_with_two_heights_emits_model_updated(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    # Fizikte yükseklik arttıkça (kameraya yaklaştıkça) ölçek (px/mm) BÜYÜMELİ; bu yüzden
    # h=100 için h=50'ye göre daha büyük piksel mesafesi (dolayısıyla daha büyük ölçek)
    # kullanılıyor — aksi halde HeightScaleModel.fit() bunu tutarsız veri olarak reddeder.
    _add_point(dialog, qtbot, height_mm=50.0, real_mm=20.0, p1=(20, 20), p2=(50, 20))  # scale=0.75
    _add_point(dialog, qtbot, height_mm=100.0, real_mm=20.0, p1=(20, 40), p2=(60, 40))  # scale=1.0

    received = []
    dialog.model_updated.connect(received.append)
    dialog._on_fit()

    assert len(received) == 1
    assert dialog.model.f_eff is not None
    assert "f_eff" in dialog._fit_result_label.text()


def test_mount_mode_defaults_to_vertical_and_toggles_height_label(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    assert dialog._mount_vertical_radio.isChecked() is True
    assert dialog._height_label.text() == "Yükseklik (mm):"

    dialog._mount_angled_radio.setChecked(True)
    assert dialog._height_label.text() == "Kameradan Mesafe (mm):"


def test_detect_board_without_capture_shows_message(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    dialog._on_detect_board()

    assert "Önce" in dialog._board_status_label.text()
    assert len(dialog.model.points) == 0


def test_detect_board_not_found_shows_message(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)
    dialog._on_capture_frame()  # düz siyah kare, tahta yok

    dialog._on_detect_board()

    assert "bulunamadı" in dialog._board_status_label.text()
    assert len(dialog.model.points) == 0


def test_detect_board_vertical_without_lens_profile_uses_pixel_spacing_fallback(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _BOARD_FRAME)
    qtbot.addWidget(dialog)
    dialog._board_cols_spin.setValue(9)
    dialog._board_rows_spin.setValue(6)
    dialog._board_square_spin.setValue(25.0)
    dialog._height_spin.setValue(80.0)
    dialog._on_capture_frame()

    dialog._on_detect_board()

    assert len(dialog.model.points) == 1
    point = dialog.model.points[0]
    assert point.height_mm == 80.0
    assert point.real_mm == 25.0
    assert point.pixel_distance > 0
    assert "Piksel aralığı" in dialog._board_status_label.text()


def test_detect_board_angled_without_lens_profile_shows_error_and_adds_no_point(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _BOARD_FRAME)
    qtbot.addWidget(dialog)
    dialog._mount_angled_radio.setChecked(True)
    dialog._on_capture_frame()

    dialog._on_detect_board()

    assert "Açılı montaj" in dialog._board_status_label.text()
    assert len(dialog.model.points) == 0


def test_detect_board_with_lens_profile_uses_solvepnp_regardless_of_mount_mode(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _BOARD_FRAME, lens_profile_provider=lambda: _SYNTHETIC_LENS_PROFILE)
    qtbot.addWidget(dialog)
    dialog._mount_angled_radio.setChecked(True)
    dialog._board_cols_spin.setValue(9)
    dialog._board_rows_spin.setValue(6)
    dialog._board_square_spin.setValue(25.0)
    dialog._on_capture_frame()

    dialog._on_detect_board()

    assert len(dialog.model.points) == 1
    point = dialog.model.points[0]
    assert point.height_mm > 0  # solvePnP mesafesi
    assert point.real_mm == 1.0  # back-computed scale trick: pixel_distance zaten ölçeğin kendisi
    assert "solvePnP" in dialog._board_status_label.text()


def test_board_type_defaults_to_checkerboard_and_toggles_visibility(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog._board_checkerboard_radio.isChecked() is True
    assert dialog._checkerboard_widget.isVisible() is True
    assert dialog._charuco_widget.isVisible() is False

    dialog._board_charuco_radio.setChecked(True)

    assert dialog._checkerboard_widget.isVisible() is False
    assert dialog._charuco_widget.isVisible() is True


def test_detect_board_charuco_without_lens_profile_shows_error(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _CHARUCO_FRAME)
    qtbot.addWidget(dialog)
    dialog._board_charuco_radio.setChecked(True)
    dialog._on_capture_frame()

    dialog._on_detect_board()

    assert "ChArUco modu için önce Lens Kalibrasyonu" in dialog._board_status_label.text()
    assert len(dialog.model.points) == 0


def test_detect_board_charuco_with_lens_profile_uses_solvepnp(qtbot):
    lens_profile = LensProfile(
        camera_matrix=np.array([[1000.0, 0.0, 300.0], [0.0, 1000.0, 250.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros((5, 1)),
        image_size=(_CHARUCO_FRAME.shape[1], _CHARUCO_FRAME.shape[0]),
        rms_error=0.1,
    )
    dialog = HeightScaleCalibrationDialog(lambda: _CHARUCO_FRAME, lens_profile_provider=lambda: lens_profile)
    qtbot.addWidget(dialog)
    dialog._board_charuco_radio.setChecked(True)
    dialog._on_capture_frame()

    dialog._on_detect_board()

    assert len(dialog.model.points) == 1
    point = dialog.model.points[0]
    assert point.height_mm > 0
    assert point.real_mm == 1.0
    assert "ChArUco mesafesi (solvePnP)" in dialog._board_status_label.text()


def test_save_profile_requires_name(qtbot, monkeypatch):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)
    _add_point(dialog, qtbot, height_mm=50.0, real_mm=20.0, p1=(20, 20), p2=(50, 20))
    _add_point(dialog, qtbot, height_mm=100.0, real_mm=20.0, p1=(20, 40), p2=(60, 40))
    dialog._on_fit()

    warned = []
    monkeypatch.setattr(
        "imgflow.ui.dialogs.height_scale_calibration_dialog.QMessageBox.warning",
        lambda *args, **kwargs: warned.append(1),
    )
    dialog._profile_name_edit.setText("")
    dialog._on_save_profile()

    assert dialog._save_status_label.text() == ""
    assert len(warned) == 1


def test_save_profile_persists_to_calibration_store(qtbot, tmp_path):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    _add_point(dialog, qtbot, height_mm=50.0, real_mm=20.0, p1=(20, 20), p2=(50, 20))
    _add_point(dialog, qtbot, height_mm=100.0, real_mm=20.0, p1=(20, 40), p2=(60, 40))
    dialog._on_fit()

    dialog._profile_name_edit.setText("hat1")
    dialog._on_save_profile()

    assert "hat1" in dialog._save_status_label.text()
    data = calibration_store.load_profile("hat1", directory=tmp_path)
    assert data.height_model is not None


def test_save_profile_warns_when_quality_regresses(qtbot, tmp_path, monkeypatch):
    from imgflow.core.height_scale_calibration import HeightScaleModel as HSM
    from imgflow.core.height_scale_calibration import TeachPoint as TP

    good_model = HSM()
    good_model.add_point(TP(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))  # scale=0.75
    good_model.add_point(TP(height_mm=75.0, pixel_distance=17.0, real_mm=20.0))  # scale=0.85 (hafif sapma)
    good_model.add_point(TP(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))  # scale=1.0
    good_model.fit()
    calibration_store.save_profile("hat2", height_model=good_model, directory=tmp_path)

    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    _add_point(dialog, qtbot, height_mm=50.0, real_mm=20.0, p1=(20, 20), p2=(50, 20))
    _add_point(dialog, qtbot, height_mm=100.0, real_mm=20.0, p1=(20, 40), p2=(60, 40))
    dialog._on_fit()

    # Yeni modelin residuals_mm_per_px()'ini yapay olarak kötüleştir
    monkeypatch.setattr(dialog.model, "residuals_mm_per_px", lambda: [1.0, 1.0])

    asked = []
    monkeypatch.setattr(
        "imgflow.ui.dialogs.height_scale_calibration_dialog.QMessageBox.question",
        lambda *args, **kwargs: asked.append(1) or QMessageBox.StandardButton.No,
    )

    dialog._profile_name_edit.setText("hat2")
    dialog._on_save_profile()

    assert len(asked) == 1
    data = calibration_store.load_profile("hat2", directory=tmp_path)
    assert data.height_model.residuals_mm_per_px() != [1.0, 1.0]  # üzerine yazılmadı


def test_profile_combo_lists_saved_profiles(qtbot, tmp_path):
    from imgflow.core.height_scale_calibration import HeightScaleModel as HSM
    from imgflow.core.height_scale_calibration import TeachPoint as TP

    model = HSM()
    model.add_point(TP(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TP(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    calibration_store.save_profile("hat9", height_model=model, directory=tmp_path)

    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)

    items = [dialog._load_profile_combo.itemText(i) for i in range(dialog._load_profile_combo.count())]
    assert items == ["hat9"]


def test_load_profile_applies_height_model_and_emits_signal(qtbot, tmp_path):
    from imgflow.core.height_scale_calibration import HeightScaleModel as HSM
    from imgflow.core.height_scale_calibration import TeachPoint as TP

    model = HSM()
    model.add_point(TP(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TP(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    calibration_store.save_profile("hat10", height_model=model, directory=tmp_path)

    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)

    received = []
    dialog.model_updated.connect(received.append)
    dialog._load_profile_combo.setCurrentText("hat10")
    dialog._on_load_profile()

    assert len(received) == 1
    assert dialog.model.f_eff == model.f_eff
    assert dialog._points_table.rowCount() == 2
    assert dialog._save_button.isEnabled() is True
    assert dialog._profile_name_edit.text() == "hat10"
    assert "hat10" in dialog._save_status_label.text()


def test_capture_frame_appends_to_gallery_instead_of_replacing(qtbot):
    """Gerçek kullanıcı raporu: "çektiğim fotoğraflar aşağıdaki [galeriye] gelmiyor" --
    eskiden 'Kare Yakala' tek bir kareyi ÜZERİNE YAZIYORDU (galeri yoktu). Artık her yakalama
    galeriye EKLENMELİ."""
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    dialog._on_capture_frame()
    dialog._on_capture_frame()
    dialog._on_capture_frame()

    assert len(dialog._gallery_frames) == 3
    assert dialog._gallery_list.count() == 3


def test_capturing_marks_newest_frame_as_active_with_star(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    dialog._on_capture_frame()
    dialog._on_capture_frame()

    assert dialog._gallery_list.item(0).text() == "#1"
    assert dialog._gallery_list.item(1).text() == "★ #2"


def test_clicking_gallery_item_activates_that_frame_in_canvas(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)
    other_frame = np.full((50, 60, 3), 200, dtype=np.uint8)

    dialog._on_capture_frame()  # index 0: _FRAME
    dialog._frame_provider = lambda: other_frame
    dialog._on_capture_frame()  # index 1: other_frame, aktif olur

    assert dialog._captured_frame.shape == other_frame.shape

    dialog._on_gallery_item_clicked(dialog._gallery_list.item(0))

    assert dialog._captured_frame.shape == _FRAME.shape
    assert dialog._gallery_list.item(0).text() == "★ #1"
    assert dialog._gallery_list.item(1).text() == "#2"


def test_gallery_files_dropped_appends_multiple_frames(qtbot, tmp_path):
    """Yakalananlar (yan panel) galerisinden bu diyalogun galerisine doğrudan sürükleyip
    bırakma -- gerçek kullanıcı isteği: "yandaki daha önceden yakaladığım kareleri de
    kullanabilmeliyim"."""
    import cv2

    dialog = HeightScaleCalibrationDialog(lambda: None)
    qtbot.addWidget(dialog)
    path1 = tmp_path / "a.png"
    path2 = tmp_path / "b.png"
    cv2.imwrite(str(path1), _FRAME)
    cv2.imwrite(str(path2), _FRAME)

    dialog._on_gallery_files_dropped([str(path1), str(path2)])

    assert len(dialog._gallery_frames) == 2
    assert dialog._captured_frame is not None


def test_detect_board_batch_without_checked_frames_shows_message(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _BOARD_FRAME)
    qtbot.addWidget(dialog)

    dialog._on_detect_board_batch()

    assert "işaretleyin" in dialog._board_status_label.text()
    assert len(dialog.model.points) == 0


def test_detect_board_batch_processes_all_checked_frames(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _BOARD_FRAME)
    qtbot.addWidget(dialog)
    dialog._board_cols_spin.setValue(9)
    dialog._board_rows_spin.setValue(6)
    dialog._board_square_spin.setValue(25.0)
    dialog._height_spin.setValue(80.0)

    dialog._on_capture_frame()  # galeriye #1 (tahta var)
    dialog._frame_provider = lambda: _FRAME
    dialog._on_capture_frame()  # galeriye #2 (düz kare, tahta YOK)

    dialog._on_detect_board_batch()

    assert len(dialog.model.points) == 1  # sadece #1'de tahta bulundu
    assert "2 işaretli kareden 1 tanesinde" in dialog._board_status_label.text()


def test_detect_board_batch_skips_unchecked_frames(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _BOARD_FRAME)
    qtbot.addWidget(dialog)
    dialog._board_cols_spin.setValue(9)
    dialog._board_rows_spin.setValue(6)
    dialog._board_square_spin.setValue(25.0)
    dialog._height_spin.setValue(80.0)

    dialog._on_capture_frame()
    dialog._gallery_list.item(0).setCheckState(Qt.CheckState.Unchecked)

    dialog._on_detect_board_batch()

    assert "işaretleyin" in dialog._board_status_label.text()
    assert len(dialog.model.points) == 0


def test_angled_mount_without_lens_profile_shows_persistent_warning(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)

    assert dialog._angled_warning_label.text() == ""

    dialog._mount_angled_radio.setChecked(True)

    assert "Açılı montaj seçili" in dialog._angled_warning_label.text()

    dialog._mount_vertical_radio.setChecked(True)

    assert dialog._angled_warning_label.text() == ""


def test_angled_mount_with_lens_profile_shows_no_warning(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, lens_profile_provider=lambda: _SYNTHETIC_LENS_PROFILE)
    qtbot.addWidget(dialog)

    dialog._mount_angled_radio.setChecked(True)

    assert dialog._angled_warning_label.text() == ""


def test_show_event_refreshes_angled_warning(qtbot):
    """Kullanıcı 'Lens Kalibrasyonunu Aç...' ile araya girip Lens Kalibrasyonu'nu
    tamamlamış olabilir -- pencere tekrar öne geldiğinde uyarı güncellenmeli."""
    profile_holder = {"profile": None}
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, lens_profile_provider=lambda: profile_holder["profile"])
    qtbot.addWidget(dialog)
    dialog._mount_angled_radio.setChecked(True)
    assert dialog._angled_warning_label.text() != ""

    profile_holder["profile"] = _SYNTHETIC_LENS_PROFILE
    dialog.show()

    assert dialog._angled_warning_label.text() == ""


def test_open_lens_calibration_button_hidden_without_callback(qtbot):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog._open_lens_button.isVisible() is False


def test_open_lens_calibration_button_visible_and_calls_callback(qtbot):
    calls = []
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, open_lens_calibration=lambda: calls.append(1))
    qtbot.addWidget(dialog)
    dialog.show()

    assert dialog._open_lens_button.isVisible() is True

    dialog._on_open_lens_calibration_clicked()

    assert calls == [1]


def test_load_profile_with_unknown_name_shows_inline_error(qtbot, tmp_path):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    dialog._load_profile_combo.addItem("does_not_exist")
    dialog._load_profile_combo.setCurrentText("does_not_exist")

    received = []
    dialog.model_updated.connect(received.append)
    dialog._on_load_profile()

    assert received == []
    assert dialog._save_status_label.text() != ""


def test_fit_with_single_height_shows_error_and_does_not_emit(qtbot, monkeypatch):
    dialog = HeightScaleCalibrationDialog(lambda: _FRAME)
    qtbot.addWidget(dialog)
    _add_point(dialog, qtbot, height_mm=50.0, real_mm=20.0, p1=(20, 20), p2=(60, 20))

    critical_calls = []
    monkeypatch.setattr(
        "imgflow.ui.dialogs.height_scale_calibration_dialog.QMessageBox.critical",
        lambda *args, **kwargs: critical_calls.append(args),
    )
    received = []
    dialog.model_updated.connect(received.append)

    dialog._on_fit()

    assert received == []
    assert len(critical_calls) == 1
