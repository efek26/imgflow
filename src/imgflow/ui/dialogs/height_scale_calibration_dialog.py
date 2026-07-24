"""Yükseklik→ölçek "öğretme" penceresi — bağımsız non-modal QDialog.

İki öğretme yöntemi desteklenir:
1. **İki noktaya tıklayarak** (klasik): "Kare Yakala" ile kare dondurulur, `MeasureCanvas`
   üzerinde bilinen gerçek uzunluktaki bir referansın iki ucuna tıklanır, "Yükseklik (mm)"/
   "Gerçek Mesafe (mm)" girilip "Nokta Ekle" ile ölçüm bir `TeachPoint` olarak eklenir.
2. **Satranç tahtası ile** (daha hızlı/hassas, elle tıklama gerektirmez): dondurulmuş karede
   bir tahta aranır; bulunursa köşelerin bilinen gerçek aralığından piksel ölçeği hesaplanır.
   Tahta türü seçilebilir: **Klasik Checkerboard** (varsayılan) ya da **ChArUco** (kısmi
   görünürlükte/kırpılmada bile çalışır, ama lens kalibrasyonu ZORUNLUDUR — adjacency
   varsayımı yapan piksel-aralığı yedek yöntemi ChArUco için güvenilir değildir).

**Kamera Montajı** seçimi (Dik/Açılı) hangi mesafenin nasıl elde edileceğini belirler:
- Aktif bir lens kalibrasyonu (intrinsics) varsa, HER İKİ montaj tipinde de tahta yöntemi
  `cv2.solvePnP` (bkz. `core/board_pose.py`) ile tahtanın kameraya olan TAM 3D mesafesini
  çözer — bu, kamera açılı monte edilmiş olsa bile doğru sonuç verir (basit dikey mesafe
  varsayımı YAPMAZ). "Açılı" seçiliyken lens kalibrasyonu ZORUNLUDUR.
- "Dik" seçiliyse VE lens kalibrasyonu henüz yoksa VE tahta türü Klasik Checkerboard'sa,
  basit bir yedek yönteme düşülür: köşeler arası piksel mesafesi / bilinen kare kenarından
  ölçek hesaplanır, yükseklik kullanıcı tarafından elle girilir.

En az 2 farklı yükseklikte nokta toplandıktan sonra "Modeli Hesapla" `HeightScaleModel.fit()`'i
çalıştırır ve kalan (residual) değerlerini gösterir. Model isteğe bağlı olarak isimli bir
profile kaydedilebilir (`io_utils/calibration_store.py`) — aynı isimle önceden kaydedilmiş
bir model varsa ve yeni modelin maks. kalanı belirgin ölçüde kötüyse kullanıcı onaylamadan
üzerine yazılmaz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imgflow.core import capture_store
from imgflow.core.board_pose import estimate_distance_mm
from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint
from imgflow.core.lens_calibration import (
    CharucoBoardConfig,
    LensProfile,
    build_charuco_board,
    build_object_points,
    detect_charuco_corners,
    detect_checkerboard_corners,
)
from imgflow.io_utils import calibration_store
from imgflow.io_utils.image_io import load_image
from imgflow.ui.widgets.measure_canvas import MeasureCanvas

_MOUNT_VERTICAL = "Dik"
_MOUNT_ANGLED = "Açılı"
_RESIDUAL_WARNING_FACTOR = 2.0
"""Yeni modelin maks. kalanı, eskisinin bu kat üzerindeyse "kalite düştü" uyarısı gösterilir."""
_GUIDE_TEXT = (
    "Kamera Montajı seçin (açılıysa önce Lens Kalibrasyonu gerekir) → 'Kare Yakala' → nokta "
    "ekle ya da 'Tahtayı Algıla' → en az 2 farklı yükseklikte toplayın → 'Modeli Hesapla' → "
    "isterseniz 'Profili Kaydet'."
)


class HeightScaleCalibrationDialog(QDialog):
    model_updated = Signal(object)  # HeightScaleModel
    frame_captured = Signal()

    def __init__(
        self,
        frame_provider: Callable[[], np.ndarray | None],
        lens_profile_provider: Callable[[], LensProfile | None] | None = None,
        calibration_dir: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Yükseklik-Ölçek Kalibrasyonu")
        self.setModal(False)
        self._model = HeightScaleModel()
        self._frame_provider = frame_provider
        self._lens_profile_provider = lens_profile_provider or (lambda: None)
        self._calibration_dir = calibration_dir if calibration_dir is not None else calibration_store.DEFAULT_CALIBRATION_DIR
        self._pending_pixel_distance: float | None = None
        self._captured_frame: np.ndarray | None = None

        self._canvas = MeasureCanvas()
        self._canvas.set_editing_enabled(True)
        self._canvas.measurement_made.connect(self._on_measurement_made)
        self._canvas.setAcceptDrops(True)
        self._canvas.image_file_dropped.connect(self._on_frame_file_dropped)

        capture_button = QPushButton("Kare Yakala")
        capture_button.clicked.connect(self._on_capture_frame)

        self._mount_vertical_radio = QRadioButton(_MOUNT_VERTICAL)
        self._mount_vertical_radio.setChecked(True)
        self._mount_angled_radio = QRadioButton(_MOUNT_ANGLED)
        mount_group = QButtonGroup(self)
        mount_group.addButton(self._mount_vertical_radio)
        mount_group.addButton(self._mount_angled_radio)
        self._mount_vertical_radio.toggled.connect(self._update_height_label)

        self._height_label = QLabel("Yükseklik (mm):")
        self._height_spin = QDoubleSpinBox()
        self._height_spin.setRange(0.0, 10000.0)
        self._height_spin.setSuffix(" mm")
        self._real_mm_spin = QDoubleSpinBox()
        self._real_mm_spin.setRange(0.01, 10000.0)
        self._real_mm_spin.setSuffix(" mm")
        self._real_mm_spin.setValue(10.0)

        self._pixel_label = QLabel("Piksel mesafesi: -")
        self._add_point_button = QPushButton("Nokta Ekle")
        self._add_point_button.setEnabled(False)
        self._add_point_button.clicked.connect(self._on_add_point)

        # -- tahta türü seçici (Klasik Checkerboard / ChArUco) --
        self._board_checkerboard_radio = QRadioButton("Klasik Checkerboard")
        self._board_checkerboard_radio.setChecked(True)
        self._board_charuco_radio = QRadioButton("ChArUco")
        board_type_group = QButtonGroup(self)
        board_type_group.addButton(self._board_checkerboard_radio)
        board_type_group.addButton(self._board_charuco_radio)
        self._board_checkerboard_radio.toggled.connect(self._on_board_type_changed)

        self._board_cols_spin = QSpinBox()
        self._board_cols_spin.setRange(3, 30)
        self._board_cols_spin.setValue(9)
        self._board_rows_spin = QSpinBox()
        self._board_rows_spin.setRange(3, 30)
        self._board_rows_spin.setValue(6)
        self._board_square_spin = QDoubleSpinBox()
        self._board_square_spin.setRange(0.1, 1000.0)
        self._board_square_spin.setValue(25.0)
        self._board_square_spin.setSuffix(" mm")

        self._ch_squares_x_spin = QSpinBox()
        self._ch_squares_x_spin.setRange(3, 30)
        self._ch_squares_x_spin.setValue(5)
        self._ch_squares_y_spin = QSpinBox()
        self._ch_squares_y_spin.setRange(3, 30)
        self._ch_squares_y_spin.setValue(7)
        self._ch_square_len_spin = QDoubleSpinBox()
        self._ch_square_len_spin.setRange(0.1, 1000.0)
        self._ch_square_len_spin.setValue(25.0)
        self._ch_square_len_spin.setSuffix(" mm")
        self._ch_marker_len_spin = QDoubleSpinBox()
        self._ch_marker_len_spin.setRange(0.1, 1000.0)
        self._ch_marker_len_spin.setValue(18.0)
        self._ch_marker_len_spin.setSuffix(" mm")

        self._checkerboard_widget = self._row_widget(
            QLabel("Sütun:"), self._board_cols_spin,
            QLabel("Satır:"), self._board_rows_spin,
            QLabel("Kare Kenarı:"), self._board_square_spin,
        )
        self._charuco_widget = self._row_widget(
            QLabel("Sütun:"), self._ch_squares_x_spin,
            QLabel("Satır:"), self._ch_squares_y_spin,
            QLabel("Kare Kenarı:"), self._ch_square_len_spin,
            QLabel("Marker Kenarı:"), self._ch_marker_len_spin,
        )
        self._charuco_widget.setVisible(False)

        detect_board_button = QPushButton("Tahtayı Algıla")
        detect_board_button.clicked.connect(self._on_detect_board)
        self._board_status_label = QLabel("")

        self._points_table = QTableWidget(0, 3)
        self._points_table.setHorizontalHeaderLabels(["Yükseklik/Mesafe (mm)", "Piksel/Ölçek", "Gerçek (mm)"])

        fit_button = QPushButton("Modeli Hesapla")
        fit_button.clicked.connect(self._on_fit)
        self._fit_result_label = QLabel("")

        self._profile_name_edit = QLineEdit()
        self._profile_name_edit.setPlaceholderText("Profil adı (ör. hat1)")
        self._save_button = QPushButton("Profili Kaydet")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save_profile)
        self._save_status_label = QLabel("")

        self._load_profile_combo = QComboBox()
        self._load_button = QPushButton("Profili Yükle")
        self._load_button.clicked.connect(self._on_load_profile)

        mount_row = QHBoxLayout()
        mount_row.addWidget(QLabel("Kamera Montajı:"))
        mount_row.addWidget(self._mount_vertical_radio)
        mount_row.addWidget(self._mount_angled_radio)
        mount_row.addStretch(1)

        form_row = QHBoxLayout()
        form_row.addWidget(self._height_label)
        form_row.addWidget(self._height_spin)
        form_row.addWidget(QLabel("Gerçek Mesafe (mm):"))
        form_row.addWidget(self._real_mm_spin)

        board_type_row = QHBoxLayout()
        board_type_row.addWidget(QLabel("Tahta Türü:"))
        board_type_row.addWidget(self._board_checkerboard_radio)
        board_type_row.addWidget(self._board_charuco_radio)
        board_type_row.addStretch(1)

        board_row = QHBoxLayout()
        board_row.addWidget(self._checkerboard_widget)
        board_row.addWidget(self._charuco_widget)
        board_row.addWidget(detect_board_button)

        save_row = QHBoxLayout()
        save_row.addWidget(self._profile_name_edit, 1)
        save_row.addWidget(self._save_button)

        load_row = QHBoxLayout()
        load_row.addWidget(self._load_profile_combo, 1)
        load_row.addWidget(self._load_button)

        guide_label = QLabel(_GUIDE_TEXT)
        guide_label.setWordWrap(True)
        guide_label.setStyleSheet("color: #666; font-style: italic;")

        layout = QVBoxLayout(self)
        layout.addWidget(guide_label)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(capture_button)
        layout.addLayout(mount_row)
        layout.addLayout(form_row)
        layout.addWidget(self._pixel_label)
        layout.addWidget(self._add_point_button)
        layout.addLayout(board_type_row)
        layout.addLayout(board_row)
        layout.addWidget(self._board_status_label)
        layout.addWidget(self._points_table)
        layout.addWidget(fit_button)
        layout.addWidget(self._fit_result_label)
        layout.addLayout(save_row)
        layout.addWidget(self._save_status_label)
        layout.addLayout(load_row)

        self._refresh_profile_combo()

    @property
    def model(self) -> HeightScaleModel:
        return self._model

    @staticmethod
    def _row_widget(*widgets: QWidget) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            row_layout.addWidget(widget)
        return row

    def _update_height_label(self) -> None:
        if self._mount_vertical_radio.isChecked():
            self._height_label.setText("Yükseklik (mm):")
        else:
            self._height_label.setText("Kameradan Mesafe (mm):")

    def _on_board_type_changed(self, _checked: bool) -> None:
        self._checkerboard_widget.setVisible(self._board_checkerboard_radio.isChecked())
        self._charuco_widget.setVisible(self._board_charuco_radio.isChecked())

    def _current_charuco_board(self):
        config = CharucoBoardConfig(
            squares_x=self._ch_squares_x_spin.value(),
            squares_y=self._ch_squares_y_spin.value(),
            square_length_mm=self._ch_square_len_spin.value(),
            marker_length_mm=self._ch_marker_len_spin.value(),
        )
        return build_charuco_board(config)

    def _on_capture_frame(self) -> None:
        frame = self._frame_provider()
        if frame is None:
            return
        self._use_frame(frame)

    def _on_frame_file_dropped(self, path: str) -> None:
        """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) canvas'a sürükleyip
        bırakılan bir kareyi `_on_capture_frame` ile AYNI şekilde kullanır — canlı kamera
        gerektirmeden, daha önce yakalanmış bir kareyle öğretim noktası eklenebilir."""
        try:
            frame = load_image(path)
        except FileNotFoundError as exc:
            self._board_status_label.setText(f"Görüntü yüklenemedi: {exc}")
            return
        self._use_frame(frame)

    def _use_frame(self, frame: np.ndarray) -> None:
        self._captured_frame = frame
        self._canvas.set_image(frame)
        self._canvas.clear_points()
        self._pending_pixel_distance = None
        self._add_point_button.setEnabled(False)
        self._pixel_label.setText("Piksel mesafesi: -")
        self._board_status_label.setText("")
        capture_store.save_capture(frame, source="height_scale")
        self.frame_captured.emit()

    def _on_measurement_made(self, _x1: float, _y1: float, _x2: float, _y2: float, pixel_distance: float) -> None:
        self._pending_pixel_distance = pixel_distance
        self._pixel_label.setText(f"Piksel mesafesi: {pixel_distance:.1f}")
        self._add_point_button.setEnabled(True)

    def _on_add_point(self) -> None:
        if self._pending_pixel_distance is None:
            return
        height_mm = self._height_spin.value()
        real_mm = self._real_mm_spin.value()
        point = TeachPoint(height_mm=height_mm, pixel_distance=self._pending_pixel_distance, real_mm=real_mm)
        self._register_point(point, height_mm, self._pending_pixel_distance, real_mm)
        self._pending_pixel_distance = None
        self._add_point_button.setEnabled(False)
        self._canvas.clear_points()

    def _on_detect_board(self) -> None:
        if self._captured_frame is None:
            self._board_status_label.setText("Önce 'Kare Yakala' ile bir kare yakalayın.")
            return

        lens_profile = self._lens_profile_provider()
        angled = self._mount_angled_radio.isChecked()

        if self._board_charuco_radio.isChecked():
            board = self._current_charuco_board()
            found, charuco_corners, charuco_ids, _preview = detect_charuco_corners(self._captured_frame, board)
            if not found:
                self._board_status_label.setText("ChArUco tahtası bulunamadı.")
                return
            if lens_profile is None:
                self._board_status_label.setText(
                    "ChArUco modu için önce Lens Kalibrasyonu (Kamera menüsü) tamamlanmalı."
                )
                return
            object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
            self._register_solvepnp_point(object_points, image_points, lens_profile, "ChArUco")
            return

        pattern_size = (self._board_cols_spin.value(), self._board_rows_spin.value())
        found, corners, _preview = detect_checkerboard_corners(self._captured_frame, pattern_size)
        if not found:
            self._board_status_label.setText("Satranç tahtası bulunamadı.")
            return

        square_size_mm = self._board_square_spin.value()
        if lens_profile is not None:
            object_points = build_object_points(pattern_size[0], pattern_size[1], square_size_mm)
            self._register_solvepnp_point(object_points, corners, lens_profile, "Satranç tahtası")
            return

        if angled:
            self._board_status_label.setText(
                "Açılı montaj için önce Lens Kalibrasyonu (Kamera menüsü) tamamlanmalı."
            )
            return

        # Dik montaj + lens kalibrasyonu yok: köşe aralığından basit ölçek hesabı (yedek yol)
        pts = corners.reshape(-1, 2)
        pixel_distance = float(np.linalg.norm(pts[1] - pts[0]))
        height_mm = self._height_spin.value()
        point = TeachPoint(height_mm=height_mm, pixel_distance=pixel_distance, real_mm=square_size_mm)
        self._register_point(point, height_mm, pixel_distance, square_size_mm)
        self._board_status_label.setText(
            f"Piksel aralığı: {pixel_distance:.1f}px (kare kenarı={square_size_mm} mm). Uyarı: Lens "
            "kalibrasyonu yapılmadığı için basitleştirilmiş (yaklaşık) yöntem kullanıldı — daha "
            "hassas ölçüm için önce Lens Kalibrasyonu yapmanız önerilir."
        )

    def _register_solvepnp_point(
        self, object_points: np.ndarray, image_points: np.ndarray, lens_profile: LensProfile, method_label: str
    ) -> None:
        try:
            distance = estimate_distance_mm(object_points, image_points, lens_profile)
        except ValueError as exc:
            self._board_status_label.setText(str(exc))
            return
        fx = float(lens_profile.camera_matrix[0, 0])
        scale = fx / distance
        point = TeachPoint(height_mm=distance, pixel_distance=scale, real_mm=1.0)
        self._register_point(point, distance, scale, 1.0)
        self._board_status_label.setText(f"{method_label} mesafesi (solvePnP): {distance:.1f} mm")

    def _register_point(self, point: TeachPoint, height_col: float, scale_col: float, real_col: float) -> None:
        self._model.add_point(point)
        self._register_point_row(height_col, scale_col, real_col)

    def _register_point_row(self, height_col: float, scale_col: float, real_col: float) -> None:
        row = self._points_table.rowCount()
        self._points_table.insertRow(row)
        self._points_table.setItem(row, 0, QTableWidgetItem(f"{height_col:.1f}"))
        self._points_table.setItem(row, 1, QTableWidgetItem(f"{scale_col:.3f}"))
        self._points_table.setItem(row, 2, QTableWidgetItem(f"{real_col:.2f}"))

    def _on_fit(self) -> None:
        try:
            self._model.fit()
        except ValueError as exc:
            QMessageBox.critical(self, "Kalibrasyon Hatası", str(exc))
            return

        residuals = self._model.residuals_mm_per_px()
        max_residual = max((abs(r) for r in residuals), default=0.0)
        self._fit_result_label.setText(
            f"f_eff={self._model.f_eff:.2f}, H_camera={self._model.h_camera:.2f} mm, "
            f"maks. kalan={max_residual:.5f} mm/px"
        )
        self._save_button.setEnabled(True)
        self.model_updated.emit(self._model)

    def _on_save_profile(self) -> None:
        if self._model.f_eff is None:
            return
        name = self._profile_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Profil Adı Gerekli", "Kaydetmek için bir profil adı girin.")
            return

        new_residuals = self._model.residuals_mm_per_px()
        new_max_residual = max((abs(r) for r in new_residuals), default=0.0)

        existing_max_residual = None
        try:
            existing = calibration_store.load_profile(name, directory=self._calibration_dir)
            if existing.height_model is not None and existing.height_model.f_eff is not None:
                old_residuals = existing.height_model.residuals_mm_per_px()
                existing_max_residual = max((abs(r) for r in old_residuals), default=0.0)
        except (OSError, ValueError, KeyError, RuntimeError):
            existing_max_residual = None

        if (
            existing_max_residual is not None
            and existing_max_residual > 0
            and new_max_residual > existing_max_residual * _RESIDUAL_WARNING_FACTOR
        ):
            reply = QMessageBox.question(
                self,
                "Kalite Uyarısı",
                "Yeni yükseklik-ölçek modeli öncekinden belirgin ölçüde kötü görünüyor "
                f"(önceki maks. kalan={existing_max_residual:.5f}, yeni={new_max_residual:.5f} mm/px).\n"
                "Yine de kaydedilsin mi?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        calibration_store.save_profile(name, height_model=self._model, directory=self._calibration_dir)
        self._save_status_label.setText(f"Kaydedildi: '{name}'")
        self._refresh_profile_combo()

    def _refresh_profile_combo(self) -> None:
        current = self._load_profile_combo.currentText()
        self._load_profile_combo.clear()
        names = calibration_store.list_profiles(self._calibration_dir)
        self._load_profile_combo.addItems(names)
        if current in names:
            self._load_profile_combo.setCurrentText(current)

    def _on_load_profile(self) -> None:
        name = self._load_profile_combo.currentText().strip()
        if not name:
            return
        try:
            profile_data = calibration_store.load_profile(name, directory=self._calibration_dir)
        except (OSError, ValueError, KeyError) as exc:
            self._save_status_label.setText(f"Profil yüklenemedi: {exc}")
            return
        model = profile_data.height_model
        if model is None or model.f_eff is None:
            self._save_status_label.setText(f"'{name}' profilinde yükseklik-ölçek modeli yok.")
            return
        self._model = model
        self._points_table.setRowCount(0)
        for point in model.points:
            self._register_point_row(point.height_mm, point.pixel_distance, point.real_mm)
        residuals = model.residuals_mm_per_px()
        max_residual = max((abs(r) for r in residuals), default=0.0)
        self._fit_result_label.setText(
            f"f_eff={model.f_eff:.2f}, H_camera={model.h_camera:.2f} mm, "
            f"maks. kalan={max_residual:.5f} mm/px"
        )
        self._save_button.setEnabled(True)
        self._profile_name_edit.setText(name)
        self._save_status_label.setText(f"Yüklendi: '{name}'")
        self.model_updated.emit(self._model)
