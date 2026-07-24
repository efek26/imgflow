import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from imgflow.core import capture_store
from imgflow.core.lens_calibration import LensProfile
from imgflow.io_utils import calibration_store
from imgflow.ui.dialogs.lens_calibration_dialog import _GUIDE_TEXT, LensCalibrationDialog
from tests.core.test_lens_calibration import _charuco_board, _draw_checkerboard, _warped_views


def _board_on_fixed_canvas(inner_cols, inner_rows, square_px, canvas_size=(700, 600)):
    """Checkerboard'ı SABİT boyutlu bir tuvale, farklı `square_px` (görünür ölçek) ile
    çizer — `LensCalibrator.calibrate()` tüm karelerin AYNI `image_size`'a sahip olmasını
    varsayar, bu yüzden gerçek mesafe farkını simüle ederken frame boyutunu SABİT tutmak
    gerekir (daha büyük görünür kare = solvePnP'ye göre daha YAKIN, daha küçük = daha UZAK)."""
    board = _draw_checkerboard(inner_cols, inner_rows, square_px=square_px, margin=40)
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    h, w = board.shape[:2]
    y0, x0 = (canvas_size[1] - h) // 2, (canvas_size[0] - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = board
    return canvas


@pytest.fixture(autouse=True)
def _isolated_capture_dir(tmp_path, monkeypatch):
    """Neredeyse her test burada `_on_capture()` çağırıyor — bu artık her karede
    `capture_store.save_capture()` tetikliyor. Gerçek `~/.imgflow/captures` dizinine ASLA
    dokunulmamalı, bu yüzden dosya autouse olarak izole ediliyor (bkz. `test_custom_filters.py`
    ile aynı desen)."""
    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")


def _configure_checkerboard(dialog, cols=5, rows=4, square=25.0):
    dialog._cb_cols_spin.setValue(cols)
    dialog._cb_rows_spin.setValue(rows)
    dialog._cb_square_spin.setValue(square)
    dialog._reset_calibrator()


def test_capture_button_flow_increments_count(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 3))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)

    dialog._on_capture()
    dialog._on_capture()

    assert len(dialog._captures) == 2
    assert dialog._capture_list.count() == 2
    assert "2" in dialog._count_label.text()


def test_captured_frames_are_checked_by_default_only_when_board_found(qtbot):
    board = _draw_checkerboard(5, 4)
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    frames = iter([board, blank])
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)

    dialog._on_capture()
    dialog._on_capture()

    assert dialog._capture_list.item(0).checkState() == Qt.CheckState.Checked
    assert dialog._capture_list.item(1).checkState() == Qt.CheckState.Unchecked
    assert dialog._checked_capture_indices() == [0]


def test_unchecking_captured_frame_excludes_it_from_calibration(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()

    dialog._capture_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    calibrator, used_frames, _reference_position = dialog._build_calibrator_from_selection()

    assert calibrator.captured_count == 11
    assert len(used_frames) == 11


def test_removing_a_capture_drops_it_from_selection(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 3))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(3):
        dialog._on_capture()

    dialog._capture_list.setCurrentRow(1)
    dialog._on_remove_selected_capture()

    assert len(dialog._captures) == 2
    assert dialog._capture_list.count() == 2


def test_removing_multiple_selected_captures_at_once(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 5))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(5):
        dialog._on_capture()

    dialog._capture_list.item(0).setSelected(True)
    dialog._capture_list.item(2).setSelected(True)
    dialog._capture_list.item(4).setSelected(True)
    dialog._on_remove_selected_capture()

    assert len(dialog._captures) == 2
    assert dialog._capture_list.count() == 2


def test_capture_persists_frame_and_emits_signal(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 3))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)

    received = []
    dialog.frame_captured.connect(lambda: received.append(1))
    dialog._on_capture()

    assert received == [1]
    records = capture_store.list_captures(source="lens")
    assert len(records) == 1


def test_guide_label_present_and_nonempty(qtbot):
    dialog = LensCalibrationDialog(lambda: None)
    qtbot.addWidget(dialog)

    labels = [w for w in dialog.findChildren(type(dialog._count_label)) if w.text() == _GUIDE_TEXT]
    assert len(labels) == 1


def test_captures_dropped_adds_frames_same_as_capture_button(qtbot, tmp_path):
    """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) kare listesine sürükleyip
    bırakmak `_on_capture` (canlı kamera) ile AYNI şekilde kalibrasyon setine ekler — canlı
    kamera olmadan da daha önce yakalanmış zor bir kareyi kalibrasyona katmayı sağlar."""
    board = _draw_checkerboard(5, 4)
    dialog = LensCalibrationDialog(lambda: None)
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    path = tmp_path / "board.png"
    cv2.imwrite(str(path), board)

    dialog._on_captures_dropped([str(path)])

    assert len(dialog._captures) == 1
    assert dialog._capture_list.count() == 1
    records = capture_store.list_captures(source="lens")
    assert len(records) == 1


def test_captures_dropped_with_missing_file_shows_warning_and_skips(qtbot, monkeypatch, tmp_path):
    dialog = LensCalibrationDialog(lambda: None)
    qtbot.addWidget(dialog)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_captures_dropped([str(tmp_path / "does_not_exist.png")])

    assert len(warnings) == 1
    assert len(dialog._captures) == 0


def test_frame_provider_returning_none_is_ignored(qtbot):
    dialog = LensCalibrationDialog(lambda: None)
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)

    dialog._on_capture()

    assert dialog._calibrator.captured_count == 0


def test_calibrate_with_enough_frames_emits_profile(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)

    for _ in range(12):
        dialog._on_capture()

    received = []
    dialog.calibrated.connect(received.append)
    dialog._on_calibrate()

    assert len(received) == 1
    assert isinstance(received[0], LensProfile)
    assert "RMS" in dialog._rms_label.text()
    assert dialog._save_button.isEnabled() is True


def test_charuco_mode_capture_and_calibrate(qtbot):
    board = _charuco_board()
    base_image = board.generateImage((600, 500))
    frames = iter(_warped_views(base_image, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    dialog._charuco_radio.setChecked(True)

    for _ in range(12):
        dialog._on_capture()

    assert len(dialog._captures) == 12
    dialog._on_calibrate()
    assert dialog._profile is not None
    assert dialog._calibrator.captured_count == 12


def test_switching_board_type_resets_captured_count(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 3))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    dialog.show()
    _configure_checkerboard(dialog)
    dialog._on_capture()
    assert len(dialog._captures) == 1

    dialog._charuco_radio.setChecked(True)

    assert len(dialog._captures) == 0
    assert dialog._capture_list.count() == 0
    assert dialog._charuco_widget.isVisible() is True
    assert dialog._checkerboard_widget.isVisible() is False


def test_save_profile_requires_name(qtbot, monkeypatch):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()
    dialog._on_calibrate()

    warned = []
    monkeypatch.setattr(
        "imgflow.ui.dialogs.lens_calibration_dialog.QMessageBox.warning",
        lambda *args, **kwargs: warned.append(1),
    )
    dialog._profile_name_edit.setText("")
    dialog._on_save_profile()

    assert dialog._save_status_label.text() == ""
    assert len(warned) == 1


def test_save_profile_persists_to_calibration_store(qtbot, tmp_path):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None), calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()
    dialog._on_calibrate()

    dialog._profile_name_edit.setText("hat1")
    dialog._on_save_profile()

    assert "hat1" in dialog._save_status_label.text()
    data = calibration_store.load_profile("hat1", directory=tmp_path)
    assert data.lens_profile is not None


def test_profile_combo_lists_saved_profiles(qtbot, tmp_path):
    good_profile = LensProfile(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        image_size=(640, 480),
        rms_error=0.1,
    )
    calibration_store.save_profile("hat9", lens_profile=good_profile, directory=tmp_path)

    dialog = LensCalibrationDialog(lambda: None, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)

    items = [dialog._load_profile_combo.itemText(i) for i in range(dialog._load_profile_combo.count())]
    assert items == ["hat9"]


def test_load_profile_applies_lens_profile_and_emits_signal(qtbot, tmp_path):
    saved_profile = LensProfile(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        image_size=(640, 480),
        rms_error=0.3,
    )
    calibration_store.save_profile("hat10", lens_profile=saved_profile, directory=tmp_path)

    dialog = LensCalibrationDialog(lambda: None, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)

    received = []
    dialog.calibrated.connect(received.append)
    dialog._load_profile_combo.setCurrentText("hat10")
    dialog._on_load_profile()

    assert len(received) == 1
    assert np.allclose(received[0].camera_matrix, saved_profile.camera_matrix)
    assert dialog._profile is not None
    assert dialog._save_button.isEnabled() is True
    assert dialog._profile_name_edit.text() == "hat10"
    assert "hat10" in dialog._save_status_label.text()


def test_load_profile_with_unknown_name_shows_inline_error(qtbot, tmp_path):
    dialog = LensCalibrationDialog(lambda: None, calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    dialog._load_profile_combo.addItem("does_not_exist")
    dialog._load_profile_combo.setCurrentText("does_not_exist")

    received = []
    dialog.calibrated.connect(received.append)
    dialog._on_load_profile()

    assert received == []
    assert dialog._save_status_label.text() != ""


def test_save_profile_warns_when_quality_regresses(qtbot, tmp_path, monkeypatch):
    good_profile = LensProfile(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        image_size=(640, 480),
        rms_error=0.05,
    )
    calibration_store.save_profile("hat2", lens_profile=good_profile, directory=tmp_path)

    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None), calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()
    dialog._on_calibrate()
    dialog._profile.rms_error = 5.0  # kötü bir yeni kalibrasyonu simüle et

    asked = []
    monkeypatch.setattr(
        "imgflow.ui.dialogs.lens_calibration_dialog.QMessageBox.question",
        lambda *args, **kwargs: asked.append(1) or QMessageBox.StandardButton.No,
    )

    dialog._profile_name_edit.setText("hat2")
    dialog._on_save_profile()

    assert len(asked) == 1
    # Kullanıcı "Hayır" dedi -> üzerine yazılmamalı, eski RMS korunmalı
    data = calibration_store.load_profile("hat2", directory=tmp_path)
    assert data.lens_profile.rms_error == 0.05


def test_calibrate_builds_experimental_focus_model_when_placements_vary(qtbot):
    """Checkerboard'ı GERÇEKTEN farklı görünür ölçekte (yakın=büyük&keskin, uzak=küçük&bulanık)
    gösterirsek, solvePnP mesafesi ile netlik ölçüsü birlikte değişir — deneysel netlik→mesafe
    modeli bunu fit edebilmeli."""
    near = _board_on_fixed_canvas(5, 4, square_px=70)
    far = cv2.GaussianBlur(_board_on_fixed_canvas(5, 4, square_px=35), (9, 9), sigmaX=3.0)

    frames = iter(_warped_views(near, 6) + _warped_views(far, 6))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()

    dialog._on_calibrate()

    assert dialog._focus_model is not None
    assert dialog._focus_model.correlation is not None
    assert "Netlik" in dialog._focus_status_label.text()


def test_calibrate_reports_failure_when_focus_signal_is_degenerate(qtbot, monkeypatch):
    """Tüm kareler AYNI netlik ölçüsüne sahipse (ör. hep aynı, bulanıklaştırılmamış görüntü)
    fit degenerdir — deneysel model sessizce yanlış bir şey üretmek yerine None kalmalı ve
    durum etiketinde açıkça belirtilmeli."""
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()

    monkeypatch.setattr("imgflow.core.focus_distance.focus_measure", lambda frame: 42.0)
    dialog._on_calibrate()

    assert dialog._focus_model is None
    assert "hesaplanamadı" in dialog._focus_status_label.text()


def test_marking_reference_capture_computes_distance_on_calibrate(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()

    dialog._capture_list.item(3).setSelected(True)
    dialog._on_mark_reference_capture()
    assert dialog._captures[3].is_reference is True
    assert "★" in dialog._capture_list.item(3).text()

    dialog._on_calibrate()

    assert dialog._reference_distance_mm is not None
    assert dialog._reference_distance_mm > 0
    assert "Referans mesafe" in dialog._reference_status_label.text()
    assert dialog._plane_rectification is not None
    assert dialog._plane_rectification.mm_per_px > 0
    assert dialog._plane_rectification.output_size[0] > 0
    assert dialog._plane_rectification.output_size[1] > 0


def test_reference_marked_but_unchecked_clears_plane_rectification_too(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()

    dialog._capture_list.item(5).setSelected(True)
    dialog._on_mark_reference_capture()
    dialog._capture_list.item(5).setCheckState(Qt.CheckState.Unchecked)  # kalibrasyona dahil etme

    dialog._on_calibrate()

    assert dialog._reference_distance_mm is None
    assert dialog._plane_rectification is None


def test_plane_rectification_saved_and_reloaded_with_profile(qtbot, tmp_path):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None), calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()
    dialog._capture_list.item(0).setSelected(True)
    dialog._on_mark_reference_capture()
    dialog._on_calibrate()
    saved_mm_per_px = dialog._plane_rectification.mm_per_px

    dialog._profile_name_edit.setText("hat_rect")
    dialog._on_save_profile()

    data = calibration_store.load_profile("hat_rect", directory=tmp_path)
    assert data.plane_rectification is not None
    assert data.plane_rectification.mm_per_px == pytest.approx(saved_mm_per_px)

    new_dialog = LensCalibrationDialog(lambda: None, calibration_dir=tmp_path)
    qtbot.addWidget(new_dialog)
    new_dialog._load_profile_combo.setCurrentText("hat_rect")
    new_dialog._on_load_profile()

    assert new_dialog._plane_rectification is not None
    assert new_dialog._plane_rectification.mm_per_px == pytest.approx(saved_mm_per_px)


def test_marking_a_new_reference_clears_the_previous_one(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 3))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(3):
        dialog._on_capture()

    dialog._capture_list.item(0).setSelected(True)
    dialog._on_mark_reference_capture()
    dialog._capture_list.item(2).setSelected(True)
    dialog._on_mark_reference_capture()

    assert dialog._captures[0].is_reference is False
    assert dialog._captures[2].is_reference is True


def test_reference_marked_but_unchecked_leaves_distance_none(qtbot):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None))
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()

    dialog._capture_list.item(5).setSelected(True)
    dialog._on_mark_reference_capture()
    dialog._capture_list.item(5).setCheckState(Qt.CheckState.Unchecked)  # kalibrasyona dahil etme

    dialog._on_calibrate()

    assert dialog._reference_distance_mm is None
    assert "dahil değil" in dialog._reference_status_label.text()


def test_reference_distance_saved_and_reloaded_with_profile(qtbot, tmp_path):
    board = _draw_checkerboard(5, 4)
    frames = iter(_warped_views(board, 12))
    dialog = LensCalibrationDialog(lambda: next(frames, None), calibration_dir=tmp_path)
    qtbot.addWidget(dialog)
    _configure_checkerboard(dialog)
    for _ in range(12):
        dialog._on_capture()
    dialog._capture_list.item(0).setSelected(True)
    dialog._on_mark_reference_capture()
    dialog._on_calibrate()
    saved_distance = dialog._reference_distance_mm
    assert saved_distance is not None

    dialog._profile_name_edit.setText("hat_ref")
    dialog._on_save_profile()

    data = calibration_store.load_profile("hat_ref", directory=tmp_path)
    assert data.reference_distance_mm == pytest.approx(saved_distance)

    new_dialog = LensCalibrationDialog(lambda: None, calibration_dir=tmp_path)
    qtbot.addWidget(new_dialog)
    new_dialog._load_profile_combo.setCurrentText("hat_ref")
    new_dialog._on_load_profile()

    assert new_dialog._reference_distance_mm == pytest.approx(saved_distance)
    assert new_dialog._plane_rectification is not None
    assert new_dialog._reference_status_label.text() != ""
