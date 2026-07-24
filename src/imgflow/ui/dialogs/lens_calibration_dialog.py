"""Lens (intrinsic) kalibrasyonu — bağımsız non-modal pencere.

Kameraya doğrudan bağlı değildir; bir `frame_provider` callback'i alır (mevcut kamera
karesini döner ya da yoksa None) — bu sayede testte gerçek kameraya ihtiyaç duymadan sahte
bir sağlayıcıyla tam kapsamlı test edilebilir.

İki tahta türü desteklenir (radio ile seçilir): **Klasik Checkerboard** (varsayılan, geriye
uyumlu) ve **ChArUco** (her karede benzersiz ID'li ArUco marker'ı olan checkerboard — kısmi
görünürlükte/kırpılmada bile kalibre edilebilir, dar konveyör bantlarında düz checkerboard'a
göre çok daha sağlam). Tahta türü değiştirildiğinde toplanan kareler sıfırlanır (iki türü
aynı anda biriktirmek anlamlı değildir).

Kalibrasyon tamamlandıktan sonra isteğe bağlı olarak isimli bir profile kaydedilebilir
(`io_utils/calibration_store.py`) — kaydetmeden önce, aynı isimle daha önce kaydedilmiş bir
profil varsa RMS karşılaştırılır; yeni kalibrasyon belirgin ölçüde kötüyse kullanıcı
onaylamadan üzerine yazılmaz.

`Kalibre Et` AYRICA, aynı checkerboard karelerinden DENEYSEL bir "netlik→mesafe" modeli
(`core/focus_distance.py`) üretir: `cv2.calibrateCamera`'ın zaten hesapladığı her karenin
kameraya gerçek mesafesini (`LensCalibrator.tvecs`), o karenin netlik/bulanıklık ölçüsüyle
eşleştirip fit eder. KESİN DEĞİLDİR ve gerçek kullanımda genelde ÇALIŞMADI (korelasyon
~0'a yakın çıktı — kameranın odak derinliği bu sinyali taşımıyor); bu yüzden fit kalitesi
UI'da açıkça gösterilir, ama ASIL ÖNERİLEN yöntem aşağıdaki sabit referans mesafesidir.

Galeride bir kareyi "Referans (Bant Seviyesi)" olarak işaretlemek MÜMKÜN — bu, ölçülecek
gerçek cisimlerin HER ZAMAN checkerboard'ın o kare yakalanırkenki AYNI düzleminde
duracağı varsayımıyla, o karenin `solvePnP`'den çözülen TAM pozundan (rotasyon + öteleme)
iki şey üretir: (1) SABİT bir `reference_distance_mm` (`mm_per_px = reference_distance_mm /
fx`), (2) bir `PlaneRectification` düzlem homografisi (`core/plane_rectification.py`).
İkincisi ÖNCELİKLİDİR — kamera referans düzleme tam dik olmasa bile (gerçek kullanıcı
raporu: aynı cisim ekranın farklı yerlerinde 6mm/5.5mm ölçülüyordu) KONUM-BAĞIMSIZ doğru
ölçüm verir; `reference_distance_mm` sadece homografi hesaplanamazsa (ör. eski profil) bir
yedektir. İkisi de netlik-tabanlı deneysel modelden ÖNCELİKLİDİR (bkz.
`main_window._compute_auto_mm_per_px`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from imgflow.core import capture_store
from imgflow.core.focus_distance import FocusDistanceModel, model_from_calibration_frames
from imgflow.core.lens_calibration import (
    CharucoBoardConfig,
    CheckerboardConfig,
    LensCalibrator,
    LensProfile,
    build_charuco_board,
    detect_charuco_corners,
    quick_checkerboard_found,
)
from imgflow.core.plane_rectification import PlaneRectification, compute_plane_rectification
from imgflow.io_utils import calibration_store
from imgflow.io_utils.image_io import load_image
from imgflow.ui.widgets.image_view import ImageView, normalize_to_uint8, numpy_to_qimage


class _CaptureDropListWidget(QListWidget):
    """Yakalananlar galerisinden (`CaptureGalleryPanel`) ya da herhangi bir yerel dosyadan
    sürüklenip bırakılan kareleri kabul eder — canlı kamera olmadan da (ör. bant durduğunda
    daha önce yakalanmış zor bir kareyi) kalibrasyon setine eklemeyi sağlar. `files_dropped`
    bırakılan tüm yerel dosya yollarını taşır; iç öğe taşıma davranışı DOKUNULMADAN kalır."""

    files_dropped = Signal(list)  # list[str]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

_GUIDE_TEXT = (
    "1) Tahta türünü seçin  2) Tahtayı, ölçülecek cisimlerin duracağı GERÇEK düzlemde (ör. bant "
    "üzerinde) tutarak 'Kare Yakala'ya basın, o kareyi seçip 'Referans (Bant Seviyesi) Yap'a "
    "basın  3) İsterseniz farklı açı/mesafelerden birkaç kare daha ekleyin (lens kalibrasyonu "
    "için en az 2-3 kare)  4) 'Kalibre Et' — lens profili + sabit referans mesafesini üretir "
    "5) İsterseniz isim vererek 'Profili Kaydet'."
)

_THUMB_SIZE = 96
_CAPTURE_INDEX_ROLE = Qt.ItemDataRole.UserRole


@dataclass
class _CaptureRecord:
    """`Kare Yakala` ile yakalanmış ham kare + son önizlemede tahtanın bulunup bulunmadığı.

    Kalibrasyon setine dahil olup olmayacağı bu kayıttan DEĞİL, kullanıcının galeri listesinde
    bıraktığı checkbox durumundan belirlenir — `found` sadece varsayılan checkbox durumunu ve
    galerideki etiketi belirler (bkz. modül docstring'i: kullanıcı hangi karelerin eğitime
    gireceğini seçebilmeli, tüm bulunan kareler otomatik/sessizce eğitilmemeli)."""

    frame: np.ndarray
    preview: np.ndarray
    found: bool
    included: bool
    is_reference: bool = False
    """Kullanıcının "Referans (Bant Seviyesi) Yap" ile işaretlediği TEK kare — bkz. modül
    docstring'i. Aynı anda sadece bir kayıtta `True` olabilir (`_on_mark_reference_capture`
    diğerlerini temizler)."""


_RMS_WARNING_FACTOR = 2.0
"""Yeni RMS, eskisinin bu kat üzerindeyse "kalite düştü" uyarısı gösterilir."""


class LensCalibrationDialog(QDialog):
    calibrated = Signal(object)  # LensProfile
    frame_captured = Signal()

    @property
    def focus_model(self) -> FocusDistanceModel | None:
        """`calibrated` sinyali SADECE `LensProfile`'ı taşır (geriye uyumluluk için — bkz.
        `tests/ui/test_lens_calibration_dialog.py`); bu ve aşağıdaki iki property, aynı
        `_on_calibrate`/`_on_load_profile` çağrısında hesaplanan/yüklenen referans mesafesi ve
        düzlem düzeltmesini `main_window._on_lens_calibrated`'ın "Kalibrasyon Profili Yükle..."
        akışıyla AYNI şekilde uygulayabilmesi için dışa açar — aksi halde kullanıcı `calibrated`
        sinyalini dinleyip SADECE lens profilini alır, referans/düzlem-düzeltme dialog içinde
        kilitli kalır ve mm/px hiç canlı oturuma yansımaz (gerçek kullanıcı raporu: "lens
        kalibrasyonu yükleyince de mm/cm gözükmüyor")."""
        return self._focus_model

    @property
    def reference_distance_mm(self) -> float | None:
        return self._reference_distance_mm

    @property
    def plane_rectification(self) -> PlaneRectification | None:
        return self._plane_rectification

    def __init__(
        self,
        frame_provider: Callable[[], np.ndarray | None],
        calibration_dir: str | Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lens Kalibrasyonu")
        self.setModal(False)
        self._frame_provider = frame_provider
        self._calibration_dir = calibration_dir if calibration_dir is not None else calibration_store.DEFAULT_CALIBRATION_DIR
        self._profile: LensProfile | None = None
        self._focus_model: FocusDistanceModel | None = None
        self._reference_distance_mm: float | None = None
        self._plane_rectification: PlaneRectification | None = None
        self._captures: list[_CaptureRecord] = []

        self._preview = ImageView()
        self._count_label = QLabel("Yakalanan kare: 0")
        self._rms_label = QLabel("")
        self._focus_status_label = QLabel("")
        self._reference_status_label = QLabel("")

        self._checkerboard_radio = QRadioButton("Klasik Checkerboard")
        self._checkerboard_radio.setChecked(True)
        self._charuco_radio = QRadioButton("ChArUco")
        board_type_group = QButtonGroup(self)
        board_type_group.addButton(self._checkerboard_radio)
        board_type_group.addButton(self._charuco_radio)
        self._checkerboard_radio.toggled.connect(self._on_board_type_changed)

        self._cb_cols_spin = QSpinBox()
        self._cb_cols_spin.setRange(3, 30)
        self._cb_cols_spin.setValue(9)
        self._cb_rows_spin = QSpinBox()
        self._cb_rows_spin.setRange(3, 30)
        self._cb_rows_spin.setValue(6)
        self._cb_square_spin = QDoubleSpinBox()
        self._cb_square_spin.setRange(0.1, 1000.0)
        self._cb_square_spin.setValue(25.0)
        self._cb_square_spin.setSuffix(" mm")

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

        self._checkerboard_widget = self._build_checkerboard_row()
        self._charuco_widget = self._build_charuco_row()
        self._charuco_widget.setVisible(False)

        capture_button = QPushButton("Kare Yakala")
        capture_button.clicked.connect(self._on_capture)
        self._calibrate_button = QPushButton("Kalibre Et")
        self._calibrate_button.clicked.connect(self._on_calibrate)
        self._calibrate_button.setEnabled(False)

        button_row = QHBoxLayout()
        button_row.addWidget(capture_button)
        button_row.addWidget(self._calibrate_button)

        self._capture_list = _CaptureDropListWidget()
        self._capture_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._capture_list.setIconSize(QPixmap(_THUMB_SIZE, _THUMB_SIZE).size())
        self._capture_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._capture_list.setFixedHeight(_THUMB_SIZE + 48)
        self._capture_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._capture_list.itemChanged.connect(self._on_capture_item_changed)
        self._capture_list.files_dropped.connect(self._on_captures_dropped)
        capture_list_label = QLabel(
            "Yakalanan kareler (kalibrasyona dahil etmek için işaretleyin — Yakalananlar "
            "galerisinden buraya sürükleyip bırakabilirsiniz):"
        )
        remove_capture_button = QPushButton("Seçilenleri Listeden Kaldır")
        remove_capture_button.clicked.connect(self._on_remove_selected_capture)
        mark_reference_button = QPushButton("Referans (Bant Seviyesi) Yap")
        mark_reference_button.clicked.connect(self._on_mark_reference_capture)

        capture_list_button_row = QHBoxLayout()
        capture_list_button_row.addWidget(remove_capture_button)
        capture_list_button_row.addWidget(mark_reference_button)

        self._profile_name_edit = QLineEdit()
        self._profile_name_edit.setPlaceholderText("Profil adı (ör. hat1)")
        self._save_button = QPushButton("Profili Kaydet")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save_profile)
        self._save_status_label = QLabel("")

        save_row = QHBoxLayout()
        save_row.addWidget(self._profile_name_edit, 1)
        save_row.addWidget(self._save_button)

        self._load_profile_combo = QComboBox()
        self._load_button = QPushButton("Profili Yükle")
        self._load_button.clicked.connect(self._on_load_profile)

        load_row = QHBoxLayout()
        load_row.addWidget(self._load_profile_combo, 1)
        load_row.addWidget(self._load_button)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Tahta Türü:"))
        type_row.addWidget(self._checkerboard_radio)
        type_row.addWidget(self._charuco_radio)
        type_row.addStretch(1)

        guide_label = QLabel(_GUIDE_TEXT)
        guide_label.setWordWrap(True)
        guide_label.setStyleSheet("color: #666; font-style: italic;")

        layout = QVBoxLayout(self)
        layout.addWidget(guide_label)
        layout.addWidget(self._preview, 1)
        layout.addLayout(type_row)
        layout.addWidget(self._checkerboard_widget)
        layout.addWidget(self._charuco_widget)
        layout.addWidget(self._count_label)
        layout.addLayout(button_row)
        layout.addWidget(capture_list_label)
        layout.addWidget(self._capture_list)
        layout.addLayout(capture_list_button_row)
        layout.addWidget(self._reference_status_label)
        layout.addWidget(self._rms_label)
        layout.addWidget(self._focus_status_label)
        layout.addLayout(save_row)
        layout.addWidget(self._save_status_label)
        layout.addLayout(load_row)

        self._reset_calibrator()
        self._refresh_profile_combo()

    @staticmethod
    def _row_widget(*widgets: QWidget) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            row_layout.addWidget(widget)
        return row

    def _build_checkerboard_row(self) -> QWidget:
        return self._row_widget(
            QLabel("Sütun:"), self._cb_cols_spin,
            QLabel("Satır:"), self._cb_rows_spin,
            QLabel("Kare Kenarı:"), self._cb_square_spin,
        )

    def _build_charuco_row(self) -> QWidget:
        return self._row_widget(
            QLabel("Sütun:"), self._ch_squares_x_spin,
            QLabel("Satır:"), self._ch_squares_y_spin,
            QLabel("Kare Kenarı:"), self._ch_square_len_spin,
            QLabel("Marker Kenarı:"), self._ch_marker_len_spin,
        )

    def _current_checkerboard_config(self) -> CheckerboardConfig:
        return CheckerboardConfig(
            inner_cols=self._cb_cols_spin.value(),
            inner_rows=self._cb_rows_spin.value(),
            square_size_mm=self._cb_square_spin.value(),
        )

    def _current_charuco_board(self):
        config = CharucoBoardConfig(
            squares_x=self._ch_squares_x_spin.value(),
            squares_y=self._ch_squares_y_spin.value(),
            square_length_mm=self._ch_square_len_spin.value(),
            marker_length_mm=self._ch_marker_len_spin.value(),
        )
        return build_charuco_board(config)

    def _on_board_type_changed(self, _checked: bool) -> None:
        self._checkerboard_widget.setVisible(self._checkerboard_radio.isChecked())
        self._charuco_widget.setVisible(self._charuco_radio.isChecked())
        self._reset_calibrator()

    def _reset_calibrator(self) -> None:
        if self._checkerboard_radio.isChecked():
            self._calibrator = LensCalibrator(self._current_checkerboard_config())
        else:
            self._calibrator = LensCalibrator()
        self._captures = []
        self._capture_list.clear()
        self._count_label.setText("Yakalanan kare: 0")
        self._calibrate_button.setEnabled(False)
        self._focus_model = None
        self._focus_status_label.setText("")
        self._reference_distance_mm = None
        self._plane_rectification = None
        self._reference_status_label.setText("")

    def _detect_preview(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Sadece galeri önizlemesi/varsayılan checkbox durumu için HIZLI bir "bulundu mu"
        kontrolü — `self._calibrator`'ı DEĞİŞTİRMEZ, alt-piksel iyileştirme/SB fallback
        YAPMAZ. Karenin gerçekten kalibrasyona dahil edilip edilmeyeceği `_on_calibrate`
        sırasında, kullanıcının işaretlediği karelerle (tam çözünürlükte, SB fallback dahil)
        YENİDEN belirlenir — bu yüzden burada hız için doğruluktan ödün vermek güvenlidir.

        Bunun tam `detect_checkerboard_corners` yerine `quick_checkerboard_found` kullanması
        KASITLI: gerçek kamera çözünürlüğünde (ör. 1280x1024) tahta bulunamadığında klasik
        dedektör SANİYELER sürebilir (tüm eşikleme seviyelerini dener) — her 'Kare Yakala'
        tıklamasında bunu tam çözünürlükte iki kez (klasik + SB fallback) çalıştırmak
        arayüzü gözle görülür şekilde kilitler (gerçek bir kullanıcı raporuyla doğrulandı)."""
        if self._charuco_radio.isChecked():
            found, _corners, _ids, preview = detect_charuco_corners(frame, self._current_charuco_board())
            return found, preview

        pattern_size = (self._cb_cols_spin.value(), self._cb_rows_spin.value())
        found = quick_checkerboard_found(frame, pattern_size)
        preview = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return found, preview

    def _thumbnail_pixmap(self, image: np.ndarray) -> QPixmap:
        qimage = numpy_to_qimage(normalize_to_uint8(image))
        return QPixmap.fromImage(qimage).scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _append_capture_list_item(self, index: int) -> None:
        record = self._captures[index]
        found_label = "Bulundu" if record.found else "Bulunamadı"
        prefix = "★ REFERANS\n" if record.is_reference else ""
        item = QListWidgetItem(f"{prefix}#{index + 1}\n{found_label}")
        item.setData(_CAPTURE_INDEX_ROLE, index)
        item.setIcon(QIcon(self._thumbnail_pixmap(record.preview)))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if record.included else Qt.CheckState.Unchecked)
        self._capture_list.addItem(item)

    def _rebuild_capture_list(self) -> None:
        self._capture_list.blockSignals(True)
        self._capture_list.clear()
        for index in range(len(self._captures)):
            self._append_capture_list_item(index)
        self._capture_list.blockSignals(False)
        self._update_count_label()

    def _checked_capture_indices(self) -> list[int]:
        indices = []
        for i in range(self._capture_list.count()):
            item = self._capture_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                indices.append(item.data(_CAPTURE_INDEX_ROLE))
        return indices

    def _update_count_label(self) -> None:
        total = len(self._captures)
        selected = len(self._checked_capture_indices())
        self._count_label.setText(f"Yakalanan kare: {total} (kalibrasyona seçili: {selected})")

    def _on_capture_item_changed(self, item: QListWidgetItem) -> None:
        index = item.data(_CAPTURE_INDEX_ROLE)
        self._captures[index].included = item.checkState() == Qt.CheckState.Checked
        self._update_count_label()

    def _on_remove_selected_capture(self) -> None:
        items = self._capture_list.selectedItems()
        if not items:
            return
        indices = sorted((item.data(_CAPTURE_INDEX_ROLE) for item in items), reverse=True)
        for index in indices:
            del self._captures[index]
        self._rebuild_capture_list()

    def _on_mark_reference_capture(self) -> None:
        """Galeride TEK bir kareyi "ölçülecek cisimlerin duracağı gerçek düzlem" (bant
        seviyesi) referansı olarak işaretler — aynı anda sadece bir kare referans olabilir,
        önceki işaret otomatik temizlenir. Sadece işaretlemek yeterli, `_on_calibrate`
        çalıştığında bu karenin solvePnP mesafesi `_reference_distance_mm`'e dönüşür."""
        items = self._capture_list.selectedItems()
        if not items:
            return
        selected_index = items[0].data(_CAPTURE_INDEX_ROLE)
        for i, record in enumerate(self._captures):
            record.is_reference = i == selected_index
        self._rebuild_capture_list()
        self._reference_distance_mm = None
        self._plane_rectification = None
        self._reference_status_label.setText(
            f"Referans: #{selected_index + 1} işaretlendi — mesafesi 'Kalibre Et' ile hesaplanacak."
        )

    def _on_capture(self) -> None:
        frame = self._frame_provider()
        if frame is None:
            return
        self._add_capture(frame)

    def _on_captures_dropped(self, paths: list[str]) -> None:
        """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) `_capture_list`'e
        sürüklenip bırakılan kareleri `_on_capture` ile AYNI şekilde ekler — canlı kamera
        gerektirmeden, daha önce yakalanmış zor bir kareyi kalibrasyon setine katmayı sağlar."""
        for path in paths:
            try:
                frame = load_image(path)
            except FileNotFoundError as exc:
                QMessageBox.warning(self, "Görüntü Yüklenemedi", str(exc))
                continue
            self._add_capture(frame)

    def _add_capture(self, frame: np.ndarray) -> None:
        found, preview = self._detect_preview(frame)
        self._captures.append(_CaptureRecord(frame=frame, preview=preview, found=found, included=found))
        self._preview.set_image(preview)
        self._append_capture_list_item(len(self._captures) - 1)
        self._update_count_label()
        self._calibrate_button.setEnabled(len(self._captures) >= 2)
        capture_store.save_capture(frame, source="lens")
        self.frame_captured.emit()

    def _build_calibrator_from_selection(self) -> tuple[LensCalibrator, list[np.ndarray], int | None]:
        """Kalibrasyona SADECE galeride işaretli kareler dahil edilir — `Kare Yakala` anındaki
        tahta bulundu/bulunamadı durumu yalnızca varsayılan checkbox durumunu belirler, kullanıcı
        onu değiştirebilir. Köşe tespiti burada, o anki tahta türü/ayarlarıyla YENİDEN yapılır;
        böylece kullanıcı yakaladıktan sonra sütun/satır sayısını düzeltip yeniden kalibre
        edebilir, kareleri tekrar yakalamasına gerek kalmaz.

        İkinci dönüş değeri, kalibratöre GERÇEKTEN eklenmiş (tahtası bulunmuş) ham kareler —
        `calibrator.tvecs` ile AYNI SIRADA — deneysel netlik→mesafe modelini bunlardan kurmak
        için. Üçüncü dönüş değeri, "Referans (Bant Seviyesi)" işaretli karenin bu listedeki
        KONUMU — işaretli kare seçili değilse ya da tahtası bulunamadıysa `None` (bkz.
        `_on_calibrate` → `_compute_reference_distance`)."""
        used_frames: list[np.ndarray] = []
        reference_position: int | None = None
        if self._charuco_radio.isChecked():
            calibrator = LensCalibrator()
            board = self._current_charuco_board()
            for index in self._checked_capture_indices():
                record = self._captures[index]
                found, _preview = calibrator.add_charuco_frame(record.frame, board)
                if found:
                    if record.is_reference:
                        reference_position = len(used_frames)
                    used_frames.append(record.frame)
        else:
            calibrator = LensCalibrator(self._current_checkerboard_config())
            for index in self._checked_capture_indices():
                record = self._captures[index]
                found, _preview = calibrator.add_frame(record.frame)
                if found:
                    if record.is_reference:
                        reference_position = len(used_frames)
                    used_frames.append(record.frame)
        return calibrator, used_frames, reference_position

    def _on_calibrate(self) -> None:
        calibrator, used_frames, reference_position = self._build_calibrator_from_selection()
        selected = len(self._checked_capture_indices())
        self._count_label.setText(
            f"Yakalanan kare: {len(self._captures)} (kalibrasyona seçili: {selected}, "
            f"kullanılabilir: {calibrator.captured_count})"
        )
        try:
            profile = calibrator.calibrate()
        except ValueError as exc:
            if calibrator.captured_count < 2:
                QMessageBox.critical(self, "Kalibrasyon Hatası", str(exc))
                return
            reply = QMessageBox.question(self, "Az Kare", f"{exc}\nYine de devam edilsin mi?")
            if reply != QMessageBox.StandardButton.Yes:
                return
            profile = calibrator.calibrate(force=True)

        self._calibrator = calibrator
        self._profile = profile
        self._rms_label.setText(f"RMS hata: {profile.rms_error:.4f}")
        self._save_button.setEnabled(True)
        self._fit_focus_model(used_frames, calibrator.tvecs or [])
        self._compute_reference_distance(profile, calibrator.rvecs or [], calibrator.tvecs or [], reference_position)
        self.calibrated.emit(profile)

    def _compute_reference_distance(
        self,
        profile: LensProfile,
        rvecs: list[np.ndarray],
        tvecs: list[np.ndarray],
        reference_position: int | None,
    ) -> None:
        """`reference_position` (`used_frames`/`tvecs`/`rvecs` içindeki konum) verilmişse, o
        karenin solvePnP mesafesini SABİT `_reference_distance_mm` olarak, TAM pozundan
        (rotasyon dahil) da bir düzlem homografisini (`_plane_rectification`) hesaplar.

        İkisi de çalışma zamanında (bkz. `main_window._compute_auto_mm_per_px`) netlik-tabanlı
        deneysel tahminden ÖNCELİKLİDİR — `_plane_rectification` (kamera düzleme dik olmasa
        bile KONUM-BAĞIMSIZ doğru) ikisinin arasında da en yüksek önceliğe sahiptir, bkz.
        `core/plane_rectification.py` modül docstring'i (gerçek kullanıcı raporu: aynı cisim
        ekranın farklı yerlerinde 6mm/5.5mm ölçülüyordu)."""
        if reference_position is None:
            if any(record.is_reference for record in self._captures):
                self._reference_distance_mm = None
                self._plane_rectification = None
                self._reference_status_label.setText(
                    "Referans olarak işaretlenen kare kalibrasyona dahil değil (işaretsiz) "
                    "ya da tahtası bulunamadı — işaretli kareyi seçip tekrar kalibre edin."
                )
            return
        distance_mm = float(np.linalg.norm(tvecs[reference_position]))
        self._reference_distance_mm = distance_mm
        self._plane_rectification = compute_plane_rectification(
            profile.camera_matrix, rvecs[reference_position], tvecs[reference_position], profile.image_size
        )
        self._reference_status_label.setText(
            f"Referans mesafe (bant seviyesi): {distance_mm:.1f} mm — konum-bağımsız düzlem "
            f"düzeltmesi hesaplandı (mm/px={self._plane_rectification.mm_per_px:.4f})"
        )

    def _fit_focus_model(self, used_frames: list[np.ndarray], tvecs: list[np.ndarray]) -> None:
        """DENEYSEL netlik→mesafe modelini aynı checkerboard karelerinden kurar — kullanıcı
        checkerboard'ı farklı yüksekliklerde/mesafelerde gösterdiyse (`tvecs`'ten en az 2
        FARKLI mesafe çıkarsa) fit edilir. Sinyal zayıfsa (`correlation` küçük) bunu UI'da
        AÇIKÇA gösterir — sessizce güvenilmez bir model üretip kullanıcıyı yanıltmaz."""
        try:
            model = model_from_calibration_frames(used_frames, tvecs)
            model.fit()
        except ValueError as exc:
            self._focus_model = None
            self._focus_status_label.setText(
                f"Netlik→Mesafe modeli (deneysel) hesaplanamadı: {exc} — checkerboard'ı "
                "farklı yüksekliklerde/mesafelerde göstererek tekrar deneyin."
            )
            return

        self._focus_model = model
        r = model.correlation or 0.0
        if abs(r) >= 0.7:
            quality = "güçlü sinyal"
        elif abs(r) >= 0.4:
            quality = "orta sinyal"
        else:
            quality = "ZAYIF sinyal, güvenilmeyebilir"
        self._focus_status_label.setText(
            f"Netlik→Mesafe modeli (deneysel): r={r:.2f} ({quality}, {len(model.points)} kare)"
        )

    def _on_save_profile(self) -> None:
        if self._profile is None:
            return
        name = self._profile_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Profil Adı Gerekli", "Kaydetmek için bir profil adı girin.")
            return

        existing_rms = None
        try:
            existing = calibration_store.load_profile(name, directory=self._calibration_dir)
            if existing.lens_profile is not None:
                existing_rms = existing.lens_profile.rms_error
        except (OSError, ValueError, KeyError):
            existing_rms = None

        if existing_rms is not None and self._profile.rms_error > existing_rms * _RMS_WARNING_FACTOR:
            reply = QMessageBox.question(
                self,
                "Kalite Uyarısı",
                f"Yeni kalibrasyon öncekinden belirgin ölçüde kötü görünüyor "
                f"(önceki RMS={existing_rms:.4f}, yeni RMS={self._profile.rms_error:.4f}).\n"
                "Yine de kaydedilsin mi?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        calibration_store.save_profile(
            name,
            lens_profile=self._profile,
            focus_model=self._focus_model,
            reference_distance_mm=self._reference_distance_mm,
            plane_rectification=self._plane_rectification,
            directory=self._calibration_dir,
        )
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
        if profile_data.lens_profile is None:
            self._save_status_label.setText(f"'{name}' profilinde lens kalibrasyonu yok.")
            return
        self._profile = profile_data.lens_profile
        self._rms_label.setText(f"RMS hata: {self._profile.rms_error:.4f}")
        self._focus_model = profile_data.focus_model
        if self._focus_model is not None and self._focus_model.correlation is not None:
            r = self._focus_model.correlation
            self._focus_status_label.setText(f"Netlik→Mesafe modeli (deneysel) yüklendi: r={r:.2f}")
        else:
            self._focus_status_label.setText("")
        self._reference_distance_mm = profile_data.reference_distance_mm
        self._plane_rectification = profile_data.plane_rectification
        if self._plane_rectification is not None:
            self._reference_status_label.setText(
                f"Referans yüklendi (konum-bağımsız düzlem düzeltmesi): "
                f"mm/px={self._plane_rectification.mm_per_px:.4f}"
            )
        elif self._reference_distance_mm is not None:
            self._reference_status_label.setText(f"Referans mesafe (bant seviyesi) yüklendi: {self._reference_distance_mm:.1f} mm")
        else:
            self._reference_status_label.setText("")
        self._save_button.setEnabled(True)
        self._profile_name_edit.setText(name)
        self._save_status_label.setText(f"Yüklendi: '{name}'")
        self.calibrated.emit(self._profile)
