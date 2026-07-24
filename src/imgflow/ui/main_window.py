"""imgflow ana penceresi: checkbox'lı operatör kütüphanesi + doğrusal pipeline şeridi
+ görüntü önizleme (ana alan) + parametre paneli.

Bir operatör kütüphaneden işaretlendiğinde LinearPipeline'a eklenir ve otomatik olarak
zincire (birincil portlar üzerinden) bağlanır; işaret kaldırılınca çıkarılır. Parametre
değişiklikleri motorun dirty/cache mekanizmasıyla yalnızca gerekli node'ları yeniden
hesaplattırır.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.batch import run_batch
from imgflow.core.camera_params import CameraParameterController
from imgflow.core.camera_source import BaslerCameraSource, CameraSource, UsbCameraSource, VideoFileSource
from imgflow.core.engine import ExecutionEngine
from imgflow.core.errors import ImgflowError
from imgflow.core.focus_distance import FocusDistanceModel
from imgflow.core.focus_metric import focus_measure
from imgflow.core.graph import Node
from imgflow.core.height_scale_calibration import HeightScaleModel
from imgflow.core.lens_calibration import LensProfile, undistort
from imgflow.core.linear_pipeline import LinearPipeline
from imgflow.core.plane_rectification import PlaneRectification
from imgflow.core.params import ParamType, defaults_for
from imgflow.io_utils import calibration_store, camera_settings_store
from imgflow.io_utils.recipe import load_recipe, save_recipe
from imgflow.operators import registry as default_registry
from imgflow.operators.builtin.region_props import draw_measurements_overlay
from imgflow.ui.dialogs.custom_filter_dialog import CustomFilterDialog
from imgflow.ui.dialogs.flat_field_dialog import FlatFieldDialog
from imgflow.ui.dialogs.height_scale_calibration_dialog import HeightScaleCalibrationDialog
from imgflow.ui.dialogs.help_dialog import HelpDialog
from imgflow.ui.dialogs.lens_calibration_dialog import LensCalibrationDialog
from imgflow.ui.dialogs.measurement_tool_dialog import MeasurementToolDialog
from imgflow.ui.dialogs.onnx_model_dialog import OnnxModelDialog
from imgflow.ui.dialogs.shape_matching_dialog import ShapeMatchingDialog
from imgflow.ui.error_dialog import show_error
from imgflow.ui.panels.camera_settings_panel import CameraSettingsPanel
from imgflow.ui.panels.capture_gallery_panel import CaptureGalleryPanel
from imgflow.ui.panels.operator_library import OperatorLibrary, description_for
from imgflow.ui.panels.pipeline_steps import PipelineStepsPanel
from imgflow.ui.widgets.enum_gallery import EnumGallery
from imgflow.ui.widgets.image_view import extract_preview_image, normalize_to_uint8
from imgflow.ui.widgets.measurements_summary import MeasurementsSummaryPanel
from imgflow.ui.widgets.param_form import ParamForm
from imgflow.ui.widgets.roi_canvas import RoiCanvas

_IMAGE_FILE_FILTER = "Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
_VIDEO_FILE_FILTER = "Videolar (*.mp4 *.avi *.mov *.mkv)"
_ROI_OP_ID = "roi.region"
_CAMERA_TICK_MS = 100
_AUTO_HEIGHT_TICK_STRIDE = 5
"""Otomatik netlik-tabanlı yükseklik tahmini her tick'te değil, her 5 tick'te bir (10Hz'lik
tick'te ~2Hz) hesaplanır — fiziksel yükseklik bu kadar hızlı değişmez, ve `focus_measure`'ı
(Laplacian) + olası dirty-marking'i her 100ms'de bir yapmak gereksiz CPU/FPS maliyeti."""
_MAX_PREVIEW_DIM = 1600
"""`image_view` zaten pencereye "sığdır" ile ölçekleyip gösterdiği için (bkz.
`ImageView._rescale`), "Normal"/"İkisi Bir Arada" görünüm modunda bunun ötesinde bir
çözünürlükte overlay çizip/birleştirip Qt'ye aktarmanın hiçbir görsel faydası yok -- sadece
10Hz kamera tick'inde gereksiz kopyalama/dönüştürme maliyeti (gerçek kullanıcı raporu: "Bölge
Ölçümü" adımının parametrelerini değiştirirken belirgin kasma). Zoom butonları/Ctrl+tekerlek
hâlâ kullanılabilir -- bu SADECE taban çözünürlüğü sınırlar, kullanıcı yine de yakınlaştırıp
detaya bakabilir. "Filtrelenmiş" (eski/varsayılan) yol ve ROI düzenleme bundan ETKİLENMEZ."""


def _cap_preview_size_with_scale(
    image: np.ndarray, max_dim: int = _MAX_PREVIEW_DIM
) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / longest
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def _cap_preview_size(image: np.ndarray, max_dim: int = _MAX_PREVIEW_DIM) -> np.ndarray:
    return _cap_preview_size_with_scale(image, max_dim)[0]


def _scale_measurements_for_display(
    measurements: list[dict[str, Any]] | None, scale: float
) -> list[dict[str, Any]] | None:
    """Üzerine-gelme hit-testinin, GÖSTERİLEN (muhtemelen `_cap_preview_size_with_scale` ile
    küçültülmüş) görüntünün koordinat sisteminde çalışabilmesi için `bbox_x/y/w/h`'yi
    ölçekler -- diğer alanlar (cm/tolerans/açı gibi FİZİKSEL değerler) çözünürlükten bağımsız
    olduğu için DOKUNULMAZ."""
    if not measurements or scale == 1.0:
        return measurements
    scaled = []
    for m in measurements:
        m2 = dict(m)
        for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            if key in m2:
                m2[key] = m2[key] * scale
        scaled.append(m2)
    return scaled


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """"İkisi Bir Arada" görünümde `np.hstack` iki görüntünün aynı dtype/kanal sayısında
    olmasını gerektirir -- filtrelenmiş taraf tek kanallı (ör. eşikleme/gri) olabilirken
    normal taraf her zaman BGR'dir, bu yüzden birleştirmeden önce ikisi de 3 kanallı uint8'e getirilir."""
    normalized = normalize_to_uint8(image)
    if normalized.ndim == 2:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return normalized


_PARAM_DEBOUNCE_MS = 120
"""Parametre paneli (slider/spinbox) değişikliklerini toplayıp TEK bir yeniden hesaplamaya
indirger — pahalı operatörlerde (ör. geom.shape_match) slider sürüklenirken her piksel
hareketinde tam pipeline'ın yeniden çalıştırılması UI'ı kilitler ('kasma'); ısınmış
`camera_settings_panel._flush_pending_changes` ile aynı debounce deseni."""


class MainWindow(QMainWindow):
    def __init__(
        self,
        registry: Any = None,
        parent=None,
        camera_settings_dir: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("imgflow")
        self.registry = registry or default_registry

        self.pipeline = LinearPipeline(self.registry)
        self.graph = self.pipeline.graph
        self.engine = ExecutionEngine(self.graph, registry=self.registry)
        self._selected_node_id: str | None = None
        self._camera_source: CameraSource | None = None
        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._on_camera_tick)
        self._param_debounce_timer = QTimer(self)
        self._param_debounce_timer.setSingleShot(True)
        self._param_debounce_timer.timeout.connect(self._flush_pending_params)
        self._lens_calibration_dialog: LensCalibrationDialog | None = None
        self._active_lens_profile: LensProfile | None = None
        self._height_scale_model: HeightScaleModel | None = None
        self._active_height_mm: float | None = None
        self._focus_distance_model: FocusDistanceModel | None = None
        self._reference_distance_mm: float | None = None
        """Sabit referans mesafe (mm) — `LensCalibrationDialog`'da "Referans (Bant Seviyesi)"
        işaretli karenin solvePnP mesafesi. Mevcutsa `_compute_auto_mm_per_px` bunu
        `_focus_distance_model`'in (gerçek kullanımda genelde zayıf çıkan) netlik-tabanlı
        DENEYSEL tahmininden ÖNCELİKLİ kullanır — tam solvePnP doğruluğunda, ama SABİT
        (ölçülen cisim kalibrasyon düzleminden farklı bir yükseklikteyse hatalıdır).
        `_plane_rectification` mevcutsa bunun ÜZERİNDE önceliklidir (bkz. aşağı)."""
        self._plane_rectification: PlaneRectification | None = None
        """Referans karenin TAM pozundan (rotasyon + öteleme) çözülen düzlem homografisi —
        bkz. `core/plane_rectification.py`. `reference_distance_mm`'in aksine (tek skaler,
        kamera düzleme tam dik değilse konuma göre hatalı — gerçek kullanıcı raporu: aynı
        cisim ekranın farklı yerlerinde 6mm/5.5mm ölçülüyordu) bu, `_on_camera_tick`'te ham
        kareyi doğrudan "kuşbakışı" bir görünüme çevirir (`cv2.warpPerspective`) — sonrasında
        `mm_per_px` SABİT bir sayı olur, kareye/konuma bakılmaz. Mevcutsa EN YÜKSEK
        önceliğe sahiptir."""
        self._last_auto_mm_per_px: float | None = None
        self._auto_height_tick_counter: int = 0
        self._last_camera_frame: np.ndarray | None = None
        """Otomatik netlik-tabanlı `mm_per_px` tekrar tekrar aynı değere yakınsa (kamera
        tick'i başına, ör. 10Hz) her seferinde node'ları dirty işaretleyip yeniden hesaplama
        yapmamak için — bkz. `_maybe_apply_auto_mm_per_px`."""
        self._view_mode: str = "filtered"
        """Canlı önizleme görünüm modu: "filtered" (seçili adımın çıktısı, eski/varsayılan
        davranış), "normal" (kalibrasyon düzeltmeleri uygulanmış ama pipeline'dan geçmemiş ham
        kare, ölçüm varsa üzerine çizilir) veya "both" (ikisi yan yana) — bkz. `_compose_display_image`.
        Otomatik ölçüm (segment.connected_components + analysis.region_props, model öğretmeden)
        kullanıcının kendi eşiklemesiyle ürünü zeminden ayırdığı senaryoda, ürünü hem
        filtrelenmiş hem de normal kare üzerinde boyutlarıyla görebilmek için eklendi."""
        self._measurement_tool_dialog: MeasurementToolDialog | None = None
        self._height_scale_dialog: HeightScaleCalibrationDialog | None = None
        self._help_dialog: HelpDialog | None = None
        self._custom_filter_dialog: CustomFilterDialog | None = None
        self._shape_matching_dialog: ShapeMatchingDialog | None = None
        self._flat_field_dialog: FlatFieldDialog | None = None
        self._onnx_model_dialog: OnnxModelDialog | None = None

        self.operator_library = OperatorLibrary(self.registry)
        self.steps_panel = PipelineStepsPanel(self.pipeline)
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color: #666; font-style: italic;")
        self.param_form = ParamForm()
        self.enum_gallery = EnumGallery()
        self.camera_settings_panel = CameraSettingsPanel(camera_settings_dir=camera_settings_dir)
        self.capture_gallery_panel = CaptureGalleryPanel()
        self.results_panel = MeasurementsSummaryPanel()
        self.image_view = RoiCanvas()
        self.image_view.setAcceptDrops(True)
        self.status_label = QLabel("")
        self.hover_info_label = QLabel("Fareyi bir nesnenin üzerine getirin.")
        self.hover_info_label.setWordWrap(True)
        self.hover_info_label.setStyleSheet("color: #ccc; background-color: #2a2a2a; padding: 4px;")

        self.operator_library.operator_checked.connect(self.add_operator)
        self.operator_library.operator_unchecked.connect(self._on_operator_unchecked)
        self.operator_library.custom_filter_editor_requested.connect(self._on_open_custom_filter_editor)
        self.steps_panel.step_selected.connect(self._on_step_selected)
        self.steps_panel.order_changed.connect(self._on_pipeline_changed)
        self.image_view.hover_measurement_changed.connect(self._on_hover_measurement_changed)
        self.image_view.image_file_dropped.connect(self._on_image_file_dropped)
        self.param_form.params_changed.connect(self._on_params_changed)
        self.enum_gallery.choice_selected.connect(self._on_enum_choice_selected)
        self.image_view.roi_changed.connect(self._on_roi_canvas_changed)
        self.image_view.roi_circle_changed.connect(self._on_roi_circle_canvas_changed)

        self._build_layout()
        self._build_menu()
        self._build_toolbar()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Dosya")
        file_menu.addAction("Görüntü Aç...", self._on_open_image)
        file_menu.addSeparator()
        file_menu.addAction("Reçeteyi Kaydet...", self._on_save_recipe)
        file_menu.addAction("Reçete Yükle...", self._on_load_recipe)
        file_menu.addSeparator()
        file_menu.addAction("Toplu İşlem...", self._on_run_batch)

        camera_menu = self.menuBar().addMenu("Kamera")
        camera_menu.addAction("USB Kamera Aç...", self._on_open_usb_camera)
        camera_menu.addAction("GigE/Basler Kamera Aç...", self._on_open_gige_camera)
        camera_menu.addAction("Video Dosyası Aç...", self._on_open_video_file)
        camera_menu.addSeparator()
        self._camera_settings_action = camera_menu.addAction(
            "Kamera Ayarları...", self._on_open_camera_settings
        )
        self._camera_settings_action.setEnabled(False)
        self._camera_settings_action.setToolTip(
            "Sadece GigE/Basler USB3 kameralarda kullanılabilir."
        )
        self._lens_calibration_action = camera_menu.addAction(
            "Lens Kalibrasyonu...", self._on_open_lens_calibration
        )
        self._lens_calibration_action.setEnabled(False)
        self._lens_calibration_action.setToolTip(
            "Checkerboard'ı farklı yükseklik/mesafelerde göstererek TEK geçişte hem lens "
            "kalibrasyonunu hem de (deneysel) otomatik yükseklik tahminini öğretir — hiçbir "
            "yükseklik/mesafe elle girilmez. Yeni çalışmalar için BURADAN başlayın."
        )
        self._load_calibration_profile_action = camera_menu.addAction(
            "Kalibrasyon Profili Yükle...", self._on_load_calibration_profile
        )
        self._load_calibration_profile_action.setEnabled(False)
        self._load_calibration_profile_action.setToolTip(
            "Daha önce kaydedilmiş bir kalibrasyon profilini (lens + yükseklik-ölçek) "
            "yeniden kalibre etmeden bu oturuma uygular."
        )
        camera_menu.addSeparator()
        camera_menu.addAction("Kamerayı Durdur", self._on_stop_camera)

        tools_menu = self.menuBar().addMenu("Araçlar")
        tools_menu.addAction("Ölçüm Aracı...", self._on_open_measurement_tool)
        active_height_action = tools_menu.addAction("Aktif Yükseklik Ayarla...", self._on_set_active_height)
        active_height_action.setToolTip(
            "ESKİ/manuel yöntem — Kamera > Lens Kalibrasyonu'nda otomatik netlik->mesafe "
            "modeli varsa bu artık gerekmez, yükseklik her karede otomatik hesaplanır."
        )
        self._height_scale_action = tools_menu.addAction(
            "Yükseklik Kalibrasyonu (Öğretme, elle)...", self._on_open_height_scale_calibration
        )
        self._height_scale_action.setEnabled(False)
        self._height_scale_action.setToolTip(
            "ESKİ/manuel yöntem: 2 noktaya tıklayarak ya da tek tek checkerboard karesi "
            "yakalayarak, her nokta için yükseklik ELLE girilir. Checkerboard'ı farklı "
            "yüksekliklerde göstererek hiçbir şey elle girmeden kalibre etmek için bunun "
            "yerine Kamera > Lens Kalibrasyonu'nu kullanın."
        )
        tools_menu.addAction("Şekil Eşleştirme (Model Öğret)...", self._on_open_shape_matching)
        tools_menu.addAction("Aydınlatma Referansı Kaydet...", self._on_open_flat_field)
        tools_menu.addAction("ONNX Model Kaydet...", self._on_open_onnx_model)

        help_menu = self.menuBar().addMenu("Yardım")
        help_menu.addAction("Kalibrasyon Kılavuzu...", self._on_open_help)

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Ana Araç Çubuğu")
        toolbar.addAction("Görüntü Aç...", self._on_open_image)
        toolbar.addSeparator()
        toolbar.addAction("Reçeteyi Kaydet...", self._on_save_recipe)
        toolbar.addAction("Reçete Yükle...", self._on_load_recipe)
        toolbar.addSeparator()
        toolbar.addAction("Toplu İşlem...", self._on_run_batch)

    def _build_layout(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Operatör Kütüphanesi"))
        left_layout.addWidget(self.operator_library, stretch=2)
        left_layout.addWidget(QLabel("Pipeline Adımları"))
        left_layout.addWidget(self.steps_panel, stretch=1)
        left_layout.addWidget(QLabel("Nesne Bilgisi (üzerine gelin):"))
        left_layout.addWidget(self.hover_info_label)

        self.image_scroll_area = QScrollArea()
        self.image_scroll_area.setWidget(self.image_view)
        self.image_scroll_area.setWidgetResizable(True)
        self.image_view.set_scroll_host(self.image_scroll_area)

        zoom_out_button = QPushButton("−")
        zoom_out_button.setToolTip("Uzaklaştır (Ctrl+tekerlek de kullanılabilir)")
        zoom_out_button.setFixedWidth(28)
        zoom_out_button.clicked.connect(self.image_view.zoom_out)
        zoom_in_button = QPushButton("+")
        zoom_in_button.setToolTip("Yakınlaştır (Ctrl+tekerlek de kullanılabilir)")
        zoom_in_button.setFixedWidth(28)
        zoom_in_button.clicked.connect(self.image_view.zoom_in)
        zoom_fit_button = QPushButton("Sığdır")
        zoom_fit_button.setToolTip("Önizlemeyi pencereye sığdır (%100 sığdırma)")
        zoom_fit_button.clicked.connect(self.image_view.zoom_reset)
        self.zoom_percent_label = QLabel("%100")
        self.zoom_percent_label.setFixedWidth(60)
        self.image_view.zoom_changed.connect(self._on_zoom_changed)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.setToolTip(
            "Filtrelenmiş: seçili adımın çıktısı. Normal: kalibrasyonu uygulanmış ham kare "
            "(ölçüm varsa üzerine çizilir). İkisi Bir Arada: ikisi yan yana."
        )
        for label, mode in [
            ("Filtrelenmiş", "filtered"),
            ("Normal", "normal"),
            ("İkisi Bir Arada", "both"),
        ]:
            self.view_mode_combo.addItem(label, mode)
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)

        zoom_bar = QHBoxLayout()
        zoom_bar.addWidget(QLabel("Yakınlaştırma:"))
        zoom_bar.addWidget(zoom_out_button)
        zoom_bar.addWidget(zoom_in_button)
        zoom_bar.addWidget(zoom_fit_button)
        zoom_bar.addWidget(self.zoom_percent_label)
        zoom_bar.addStretch(1)
        zoom_bar.addWidget(QLabel("Görünüm:"))
        zoom_bar.addWidget(self.view_mode_combo)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addLayout(zoom_bar)
        center_layout.addWidget(self.image_scroll_area, stretch=1)
        center_layout.addWidget(self.status_label)

        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        params_layout.addWidget(QLabel("Parametreler"))
        params_layout.addWidget(self.description_label)
        params_layout.addWidget(self.param_form)
        params_layout.addWidget(QLabel("Seçenek Önizlemeleri"))
        params_layout.addWidget(self.enum_gallery)
        params_layout.addStretch(1)

        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(params_tab, "Parametreler")
        self.right_tabs.addTab(self.camera_settings_panel, "Kamera Ayarları")
        self.right_tabs.addTab(self.results_panel, "Sonuçlar")

        # QSplitter: kullanıcı bölmeler arası sınırları sürükleyip önizleme alanını
        # büyütebilsin diye (sabit stretch oranlı QHBoxLayout'ta bu mümkün değildi).
        self.central_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.central_splitter.addWidget(left)
        self.central_splitter.addWidget(center)
        self.central_splitter.addWidget(self.right_tabs)
        self.central_splitter.setStretchFactor(0, 1)
        self.central_splitter.setStretchFactor(1, 2)
        self.central_splitter.setStretchFactor(2, 1)
        self.central_splitter.setSizes([260, 640, 260])
        self.setCentralWidget(self.central_splitter)

        self.capture_gallery_dock = QDockWidget("Yakalanan Kareler", self)
        self.capture_gallery_dock.setWidget(self.capture_gallery_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.capture_gallery_dock)

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_percent_label.setText(f"%{int(zoom * 100)}")

    def _on_view_mode_changed(self, index: int) -> None:
        self._view_mode = self.view_mode_combo.itemData(index)
        self._refresh_preview()

    def _on_hover_measurement_changed(self, measurement: dict[str, Any] | None) -> None:
        """Sol paneldeki `hover_info_label`'ı, fare `image_view` üzerinde bir ölçüm kutusunun
        üstündeyken günceller (bkz. `RoiCanvas.hover_measurement_changed`). Hem
        `analysis.region_props` (boyut/alan/açı/tolerans) hem `geom.shape_match` (model/skor/
        açı) tarzı ölçüm sözlüklerini -- alan adlarına bakarak -- ele alır."""
        if not measurement:
            self.hover_info_label.setText("Fareyi bir nesnenin üzerine getirin.")
            return

        lines: list[str] = []
        label = measurement.get("label")
        if label is not None:
            lines.append(f"Etiket: {label}")

        if "model" in measurement:
            lines.append(f"Model: {measurement['model']}")
            if "x" in measurement and "y" in measurement:
                lines.append(f"Konum: x={measurement['x']:.1f} y={measurement['y']:.1f}")
            if "score" in measurement:
                lines.append(f"Skor: {measurement['score']:.2f}")
            if "angle" in measurement:
                lines.append(f"Açı (alpha): {measurement['angle']:.1f}°")
            self.hover_info_label.setText("\n".join(lines))
            return

        if "obb_mm_w" in measurement and "obb_mm_h" in measurement:
            lines.append(
                f"Boyut: {measurement['obb_mm_w'] / 10:.2f} x {measurement['obb_mm_h'] / 10:.2f} cm"
            )
        elif "obb_w" in measurement and "obb_h" in measurement:
            lines.append(f"Boyut: {measurement['obb_w']:.0f} x {measurement['obb_h']:.0f} px")

        if "area_mm2" in measurement:
            lines.append(f"Alan: {measurement['area_mm2'] / 100:.2f} cm²")
        elif "area" in measurement:
            lines.append(f"Alan: {measurement['area']:.0f} px²")

        if "obb_angle" in measurement:
            lines.append(f"Açı: {measurement['obb_angle']:.1f}°")

        if "tolerance_ok" in measurement:
            lines.append("Tolerans: OK" if measurement["tolerance_ok"] else "Tolerans: NG")

        self.hover_info_label.setText("\n".join(lines) if lines else "")

    # -- public API (dialogsuz, testlerden de çağrılabilir) -----------------

    def add_operator(self, op_id: str) -> str:
        op_cls = self.registry.get(op_id)
        node_id = self.pipeline.generate_node_id(op_id)
        self.pipeline.append(Node(node_id, op_id, params=defaults_for(op_cls.params)))
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self.operator_library.set_checked(op_id, True)
        self._select_step(node_id)
        return node_id

    def remove_operator(self, node_id: str) -> None:
        op_id = self.graph.nodes[node_id].op_id
        was_selected = self._selected_node_id == node_id
        removed_position = self.pipeline.order.index(node_id) if node_id in self.pipeline.order else None

        self.pipeline.remove(node_id)
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self.operator_library.set_checked(op_id, False)

        if was_selected:
            self._selected_node_id = None
            self.description_label.setText("")
            self.param_form.set_params([], {})
            self.enum_gallery.clear()
            fallback_id = self._fallback_step_id(removed_position)
            if fallback_id is not None:
                self._select_step(fallback_id)
                self._on_step_selected(fallback_id)
                return
        self._refresh_preview()

    def _fallback_step_id(self, removed_position: int | None) -> str | None:
        """Kaldırılan adım seçiliyken, zincirde onun yerini alan (ya da son) adımı döner."""
        order = self.pipeline.order
        if not order:
            return None
        if removed_position is None:
            return order[-1]
        index = min(removed_position, len(order) - 1)
        return order[index]

    def selected_node_id(self) -> str | None:
        return self._selected_node_id

    def save_recipe_to(self, path: str) -> None:
        save_recipe(path, self.graph)

    def load_recipe_from(self, path: str) -> None:
        new_graph = load_recipe(path)
        for node in new_graph.nodes.values():
            self.registry.get(node.op_id)  # bilinmeyen operatör varsa erken hata ver
        self._selected_node_id = None
        self.description_label.setText("")
        self.param_form.set_params([], {})
        self.enum_gallery.clear()
        self.pipeline.load(new_graph)
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self._sync_library_checkboxes()
        if new_graph.calibration_profile:
            self._load_calibration_profile(new_graph.calibration_profile)
            if new_graph.calibration_height_mm is not None:
                self._set_active_height_mm(new_graph.calibration_height_mm)
        self._refresh_preview()

    def _load_calibration_profile(self, name: str) -> bool:
        """İsimle bir kalibrasyon profilini (lens + yükseklik-ölçek modeli) yükler.

        İki çağrı yolu var: (1) reçete yüklenirken otomatik (`load_recipe_from` — kullanıcı
        elle hiçbir şey yapmadan uygulanır, bu yol dönüş değerini yok sayar); (2) Kamera
        menüsünden elle "Kalibrasyon Profili Yükle..." (bu yolda çağıran, `False` dönerse
        kullanıcıya inline bir hata göstermeli — otomatik yoldaki sessiz davranış burada
        uygun değil). Profil bulunamazsa/okunamazsa `False` döner, mevcut aktif kalibrasyon
        DEĞİŞTİRİLMEZ."""
        try:
            profile_data = calibration_store.load_profile(name)
        except (OSError, ValueError, KeyError):
            return False
        self._active_lens_profile = profile_data.lens_profile
        self._height_scale_model = profile_data.height_model
        self._focus_distance_model = profile_data.focus_model
        self._reference_distance_mm = profile_data.reference_distance_mm
        self._plane_rectification = profile_data.plane_rectification
        self._last_auto_mm_per_px = None
        # Sabit referans mesafe/düzlem düzeltmesi varsa kameradan bir kare beklemeye gerek
        # yok — hemen uygula (bkz. `_compute_auto_mm_per_px`, kareye bağımlı değil).
        mm_per_px = self._compute_auto_mm_per_px()
        if mm_per_px is not None:
            self._last_auto_mm_per_px = mm_per_px
            self._push_mm_per_px(mm_per_px, refresh=False)
        return True

    def _push_mm_per_px(self, mm_per_px: float, refresh: bool = True) -> None:
        """`refresh=False`, `_on_camera_tick` gibi zaten kendi sonunda bir kez
        `_refresh_preview()` çağıracak bir çağrıdan geldiğinde kullanılır — aksi halde her
        tick'te pipeline'ı GEREKSİZ YERE İKİ KEZ çalıştırıp (bir kez burada, bir kez tick'in
        kendi sonunda) FPS'i belirgin şekilde düşürür (gerçek bir kullanıcı raporuyla
        doğrulandı: otomatik yükseklik tahmini eklendikten sonra kamera akışı yavaşladı)."""
        for node_id, node in self.graph.nodes.items():
            if node.op_id == "analysis.region_props":
                node.params = {**node.params, "mm_per_px": mm_per_px}
                self.engine.mark_dirty(node_id)
        if refresh:
            self._refresh_preview()

    def _set_active_height_mm(self, height_mm: float) -> None:
        """Aktif yüksekliği ayarlar ve varsa yükseklik-ölçek modelinden hesaplanan mm/px
        değerini tüm `analysis.region_props` node'larına uygular. Hem reçeteden (otomatik)
        hem de "Araçlar" menüsünden (manuel) çağrılan tek ortak yol.

        Bu, ESKİ (2 noktaya tıklayarak öğretilen `HeightScaleModel`) akış içindir. Checkerboard-
        tabanlı yeni akışta (`_focus_distance_model` mevcutsa) `mm_per_px` bunun yerine HER
        kamera karesinde otomatik hesaplanır (bkz. `_maybe_apply_auto_mm_per_px`) — kullanıcı
        hiçbir yükseklik değeri elle girmez."""
        self._active_height_mm = height_mm
        mm_per_px = self._active_mm_per_px()
        if mm_per_px is None:
            return
        self._push_mm_per_px(mm_per_px)

    def _maybe_apply_auto_mm_per_px(self, frame: np.ndarray) -> None:
        """Checkerboard-tabanlı DENEYSEL akış: lens kalibrasyonundan bilinen `fx` ile, o anki
        karenin netlik ölçüsünden (`focus_measure`) `FocusDistanceModel` üzerinden tahmin
        edilen mesafeyi birleştirip `mm_per_px`'i canlı hesaplar (bkz. `_compute_auto_mm_per_px`)
        — hiçbir yükseklik/mesafe elle girilmez. `_active_lens_profile`/`_focus_distance_model`
        yoksa (eski akış ya da henüz kalibre edilmemiş) hiçbir şey yapmaz, mevcut davranış
        DEĞİŞMEZ.

        `self._last_camera_frame` HER çağrıda (throttle'dan ÖNCE) güncellenir — `_active_mm_per_px`
        (Ölçüm Aracı gibi anlık tüketiciler) her zaman en güncel kareyi kullansın diye.

        Tahmin, `FocusDistanceModel.predict_distance` tarafından zaten kalibre edilen mesafe
        aralığına kenetlenir (bkz. `core/focus_distance.py`) — zayıf/gürültülü bir fit çılgın
        bir ölçeğe savrulamaz. `_on_camera_tick` her 100ms'de bir çağırdığı için maliyeti iki
        yönden sınırlıyoruz: (1) sadece her `_AUTO_HEIGHT_TICK_STRIDE` tick'te bir hesaplanır
        (fiziksel yükseklik 10Hz'de değişmez, ~2Hz yeterli), (2) `_push_mm_per_px`'e
        `refresh=False` verilir çünkü tick zaten kendi sonunda bir kez yenileniyor —
        aksi halde pipeline tick başına iki kez çalışıp FPS'i düşürür.

        '%0.5'ten az değişince atla' kısayolu SADECE hesaplanan değerin kendisini
        `_last_auto_mm_per_px`'e karşı kontrol ediyordu — pipeline'a kalibrasyon zaten
        stabilize olduktan SONRA yeni bir `analysis.region_props` adımı eklenirse (varsayılan
        `mm_per_px=0.0` ile başlar), hesaplanan değer "değişmedi" sayılıp bu YENİ node hiç
        güncellenmeden sonsuza dek 0'da kalıyordu (gerçek kullanıcı raporu: "kalibrasyon
        profilini yükleyince de çalışmıyordu sadece pixel yazıyordu"). Bu yüzden atlama
        koşuluna `_region_props_needs_mm_per_px` kontrolü de eklendi — herhangi bir node'un
        GÜNCEL parametresi hedeften farklıysa değer aynı olsa bile push tekrar çalışır."""
        self._last_camera_frame = frame
        self._auto_height_tick_counter += 1
        if (self._auto_height_tick_counter - 1) % _AUTO_HEIGHT_TICK_STRIDE != 0:
            return
        mm_per_px = self._compute_auto_mm_per_px()
        if mm_per_px is None:
            return
        if (
            self._last_auto_mm_per_px is not None
            and abs(mm_per_px - self._last_auto_mm_per_px) <= 0.005 * self._last_auto_mm_per_px
            and not self._region_props_needs_mm_per_px(mm_per_px)
        ):
            return
        self._last_auto_mm_per_px = mm_per_px
        self._push_mm_per_px(mm_per_px, refresh=False)

    def _region_props_needs_mm_per_px(self, mm_per_px: float) -> bool:
        for node in self.graph.nodes.values():
            if node.op_id != "analysis.region_props":
                continue
            current = float(node.params.get("mm_per_px", 0.0))
            if abs(current - mm_per_px) > 0.005 * mm_per_px:
                return True
        return False

    def run_batch_process(self, node_id: str, input_folder: str, output_csv: str) -> list[dict[str, Any]]:
        return run_batch(self.graph, node_id, input_folder, output_csv, registry=self.registry)

    def open_image(self, path: str) -> str:
        """Var olan io.image_source node'unu bulup path'ini günceller, yoksa bir tane ekler.

        Kamera aktif akıyorsa ÖNCE durdurulur — aksi halde `_on_camera_tick` en geç
        `_CAMERA_TICK_MS` (100ms) içinde `engine.inject_result` ile canlı kareyi tekrar
        cache'e yazıp burada açılan statik görüntünün üzerine yazardı (yakalanan bir kareyi
        sürükleyip pipeline'a bırakınca ya da 'Görüntü Aç...' ile açınca kalıcı görünmesi
        beklenir, 100ms sonra sessizce kaybolması değil)."""
        if self._camera_timer.isActive():
            self.stop_camera()
        node_id = self._find_or_create_image_source()
        self.graph.nodes[node_id].params["path"] = path
        self.engine.mark_dirty(node_id)
        self._select_step(node_id)
        self._on_step_selected(node_id)  # zaten seçiliyse bile formu/önizlemeyi taze path ile senkronla
        return node_id

    def _on_image_file_dropped(self, path: str) -> None:
        """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan — Windows Gezgini'nden
        sürüklemek de aynı şekilde çalışır, bkz. `ImageView.dropEvent`) ana pipeline
        önizlemesine sürükle-bırak: `open_image` ile AYNI yolu kullanır, pipeline'ın girdisini
        o dosyaya çevirir."""
        self.open_image(path)
        self.status_label.setText(f"Görüntü bırakıldı: '{path}'")

    def start_camera(self, source: CameraSource) -> None:
        """Verilen kaynaktan (USB/GigE/video) periyodik kare çekmeye başlar.

        Her tick'te io.image_source node'unun cache'ine doğrudan kare enjekte edilir
        (bkz. ExecutionEngine.inject_result) — statik dosya ile canlı akış aynı pipeline'ı
        ve aynı io.image_source node'unu paylaşır.
        """
        self.stop_camera()
        self._camera_source = source
        self._camera_timer.start(_CAMERA_TICK_MS)
        is_basler = isinstance(source, BaslerCameraSource)
        self._camera_settings_action.setEnabled(is_basler)
        self._lens_calibration_action.setEnabled(True)
        self._height_scale_action.setEnabled(True)
        self._load_calibration_profile_action.setEnabled(True)
        if is_basler:
            controller = CameraParameterController(source.node_map)
            restored = self._apply_last_used_settings(controller)
            status = f"🟢 Bağlı ({type(source).__name__})"
            if restored:
                status += " — önceki ayarlar geri yüklendi"
            self.camera_settings_panel.set_controller(controller, status_text=status)
        else:
            self.camera_settings_panel.set_controller(None)

    def _apply_last_used_settings(self, controller: CameraParameterController) -> bool:
        """Bu kameranın (aynı ön ayar dizininin) en son bağlantı kesildiğinde otomatik
        kaydedilen ayarlarını varsa geri yükler. Farklı bir kamera modeli takılıysa bazı
        node'lar reddedebilir/mevcut olmayabilir — `_on_load_settings` ile aynı desen:
        alan bazlı try/except, tek bir alanın başarısızlığı diğerlerini engellemez."""
        data = camera_settings_store.load_last_used(directory=self.camera_settings_panel.settings_dir)
        if data is None or not data.values:
            return False
        applied = 0
        for name, value in data.values.items():
            try:
                controller.set(name, value)
            except Exception:
                continue
            applied += 1
        return applied > 0

    def stop_camera(self) -> None:
        if self._camera_timer.isActive():
            self._camera_timer.stop()
        if isinstance(self._camera_source, BaslerCameraSource):
            try:
                values = self.camera_settings_panel.current_values()
                if values:
                    camera_settings_store.save_last_used(values, directory=self.camera_settings_panel.settings_dir)
            except Exception:
                # Kamera bağlantısı zaten kopmuş olabilir (ör. kablo çekildi) — bu durumda
                # son ayarları okuyamayız, ama bu YİNE DE `stop_camera()`'ı (kapanış/kamera
                # değiştirme akışında çağrılıyor) engellememeli.
                pass
        if self._camera_source is not None:
            self._camera_source.release()
            self._camera_source = None
        self._camera_settings_action.setEnabled(False)
        self._lens_calibration_action.setEnabled(False)
        self._height_scale_action.setEnabled(False)
        self._load_calibration_profile_action.setEnabled(False)
        self.camera_settings_panel.set_controller(None)
        if self._lens_calibration_dialog is not None:
            self._lens_calibration_dialog.close()
            self._lens_calibration_dialog = None
        if self._height_scale_dialog is not None:
            self._height_scale_dialog.close()
            self._height_scale_dialog = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.stop_camera()
        super().closeEvent(event)

    # -- yardımcılar ----------------------------------------------------

    def _find_or_create_image_source(self) -> str:
        for node_id, node in self.graph.nodes.items():
            if node.op_id == "io.image_source":
                return node_id
        return self.add_operator("io.image_source")

    def _select_step(self, node_id: str) -> None:
        self.steps_panel.select_node(node_id)

    def _sync_library_checkboxes(self) -> None:
        present_op_ids = {n.op_id for n in self.graph.nodes.values()}
        for op_id in self.registry.ids():
            self.operator_library.set_checked(op_id, op_id in present_op_ids)

    def _render_enum_choice(self, node_id: str, param_name: str, choice: str) -> np.ndarray | None:
        node = self.graph.nodes[node_id]
        trial_params = dict(node.params)
        trial_params[param_name] = choice
        result = self.engine.trial_run(node_id, trial_params)
        if not result.ok:
            return None
        op_cls = self.registry.get(node.op_id)
        return extract_preview_image(op_cls, result.outputs)

    def _refresh_enum_gallery(self) -> None:
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            self.enum_gallery.clear()
            return
        node = self.graph.nodes[node_id]
        op_cls = self.registry.get(node.op_id)
        enum_param = next((p for p in op_cls.params if p.type is ParamType.ENUM), None)
        if enum_param is None:
            self.enum_gallery.clear()
            return
        current = node.params.get(enum_param.name, enum_param.default)
        self.enum_gallery.show_choices(
            enum_param.name,
            list(enum_param.choices or []),
            current,
            lambda choice: self._render_enum_choice(node_id, enum_param.name, choice),
        )

    # -- Qt slot'ları -----------------------------------------------------

    def _on_open_image(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Görüntü Aç", "", _IMAGE_FILE_FILTER)
        if path:
            self.open_image(path)

    def _on_open_usb_camera(self) -> None:
        index, ok = QInputDialog.getInt(self, "USB Kamera", "Cihaz index'i:", 0, 0, 10)
        if not ok:
            return
        try:
            self.start_camera(UsbCameraSource(index))
        except RuntimeError as exc:
            show_error(self, "Kamera Açılamadı", exc)

    def _on_open_gige_camera(self) -> None:
        index, ok = QInputDialog.getInt(self, "GigE/Basler Kamera", "Kamera index'i:", 0, 0, 10)
        if not ok:
            return
        try:
            self.start_camera(BaslerCameraSource(index))
        except RuntimeError as exc:
            show_error(self, "Kamera Açılamadı", exc)

    def _on_open_camera_settings(self) -> None:
        self.right_tabs.setCurrentWidget(self.camera_settings_panel)

    def _on_open_lens_calibration(self) -> None:
        if self._camera_source is None:
            return
        if self._lens_calibration_dialog is not None:
            self._lens_calibration_dialog.show()
            self._lens_calibration_dialog.raise_()
            self._lens_calibration_dialog.activateWindow()
            return
        dialog = LensCalibrationDialog(self._camera_source.read, parent=self)
        dialog.calibrated.connect(self._on_lens_calibrated)
        dialog.frame_captured.connect(self.capture_gallery_panel.refresh)
        dialog.destroyed.connect(lambda: setattr(self, "_lens_calibration_dialog", None))
        self._lens_calibration_dialog = dialog
        dialog.show()

    def _on_lens_calibrated(self, profile: LensProfile) -> None:
        """`LensCalibrationDialog.calibrated` SADECE `LensProfile`'ı taşır (dialog testlerinin
        bekledi imza budur) — ama aynı kalibrasyon/yükleme çağrısında dialog içinde hesaplanan
        referans mesafe/düzlem düzeltmesi/netlik modeli de vardır (`dialog.reference_distance_mm`
        / `plane_rectification` / `focus_model` property'leri, bkz. o dosyadaki docstring).
        Bunları burada da `_load_calibration_profile` ile AYNI şekilde uygulamazsak, kullanıcı
        "Lens Kalibrasyonu..." akışını (referansı işaretleyip 'Kalibre Et'e basarak) TEK
        geçişte tamamlasa bile mm/px hiçbir zaman canlı oturuma yansımaz — sadece kaydedip
        ayrıca "Kalibrasyon Profili Yükle..." ile yeniden yüklerse çalışırdı (gerçek kullanıcı
        raporu: "lens kalibrasyonu yükleyince de mm/cm gözükmüyor")."""
        self._active_lens_profile = profile
        dialog = self._lens_calibration_dialog
        if dialog is not None:
            self._focus_distance_model = dialog.focus_model
            self._reference_distance_mm = dialog.reference_distance_mm
            self._plane_rectification = dialog.plane_rectification
        self._last_auto_mm_per_px = None
        mm_per_px = self._compute_auto_mm_per_px()
        if mm_per_px is not None:
            self._last_auto_mm_per_px = mm_per_px
            self._push_mm_per_px(mm_per_px, refresh=False)
        self._refresh_preview()

    def _on_load_calibration_profile(self) -> None:
        names = calibration_store.list_profiles()
        if not names:
            self.status_label.setText("Kayıtlı kalibrasyon profili yok.")
            return
        # Her profilin SABİT mm/px değeri (varsa) isminin yanına parantez içinde eklenir --
        # kullanıcı hangi profilin hangi ölçeğe karşılık geldiğini isme bakmadan görebilsin.
        labels = []
        for name in names:
            try:
                profile_data = calibration_store.load_profile(name)
            except (OSError, ValueError, KeyError):
                labels.append(name)
                continue
            labels.append(calibration_store.format_profile_label(name, profile_data))
        label, ok = QInputDialog.getItem(
            self, "Kalibrasyon Profili Yükle", "Profil:", labels, 0, False
        )
        if not ok or not label:
            return
        name = calibration_store.profile_name_from_label(label)
        if self._load_calibration_profile(name):
            self.graph.calibration_profile = name
            self._refresh_preview()
            self.status_label.setText(f"Kalibrasyon profili yüklendi: '{name}'")
        else:
            self.status_label.setText(f"Kalibrasyon profili yüklenemedi: '{name}'")

    def _on_open_height_scale_calibration(self) -> None:
        if self._camera_source is None:
            return
        if self._height_scale_dialog is not None:
            self._height_scale_dialog.show()
            self._height_scale_dialog.raise_()
            self._height_scale_dialog.activateWindow()
            return
        dialog = HeightScaleCalibrationDialog(
            self._camera_source.read,
            lens_profile_provider=lambda: self._active_lens_profile,
            parent=self,
        )
        dialog.model_updated.connect(self._on_height_scale_model_updated)
        dialog.frame_captured.connect(self.capture_gallery_panel.refresh)
        dialog.destroyed.connect(lambda: setattr(self, "_height_scale_dialog", None))
        self._height_scale_dialog = dialog
        dialog.show()

    def _on_height_scale_model_updated(self, model: HeightScaleModel) -> None:
        self._height_scale_model = model

    def _on_open_shape_matching(self) -> None:
        if self._shape_matching_dialog is not None:
            self._shape_matching_dialog.show()
            self._shape_matching_dialog.raise_()
            self._shape_matching_dialog.activateWindow()
            return
        dialog = ShapeMatchingDialog(parent=self)
        dialog.models_changed.connect(self._on_shape_models_changed)
        dialog.destroyed.connect(lambda: setattr(self, "_shape_matching_dialog", None))
        self._shape_matching_dialog = dialog
        dialog.show()

    def _on_shape_models_changed(self) -> None:
        """Şekil Eşleştirme aracında bir model kaydedilir/silinir/yeniden adlandırılırsa,
        `geom.shape_match` düğümü o an seçiliyse parametre panelindeki 'Model Adı' açılır
        listesi hemen günceli göstersin diye formu yeniden kurar."""
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            return
        node = self.graph.nodes[node_id]
        if node.op_id != "geom.shape_match":
            return
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)

    def _on_open_flat_field(self) -> None:
        if self._flat_field_dialog is not None:
            self._flat_field_dialog.show()
            self._flat_field_dialog.raise_()
            self._flat_field_dialog.activateWindow()
            return
        dialog = FlatFieldDialog(parent=self)
        dialog.references_changed.connect(self._on_flat_field_references_changed)
        dialog.destroyed.connect(lambda: setattr(self, "_flat_field_dialog", None))
        self._flat_field_dialog = dialog
        dialog.show()

    def _on_flat_field_references_changed(self) -> None:
        """`_on_shape_models_changed` ile AYNI desen: bir referans kaydedilir/silinirse,
        `correction.flat_field` düğümü o an seçiliyse parametre panelindeki 'Aydınlatma
        Referansı' açılır listesi hemen günceli göstersin diye formu yeniden kurar."""
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            return
        node = self.graph.nodes[node_id]
        if node.op_id != "correction.flat_field":
            return
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)

    def _on_open_onnx_model(self) -> None:
        if self._onnx_model_dialog is not None:
            self._onnx_model_dialog.show()
            self._onnx_model_dialog.raise_()
            self._onnx_model_dialog.activateWindow()
            return
        dialog = OnnxModelDialog(parent=self)
        dialog.models_changed.connect(self._on_onnx_models_changed)
        dialog.destroyed.connect(lambda: setattr(self, "_onnx_model_dialog", None))
        self._onnx_model_dialog = dialog
        dialog.show()

    def _on_onnx_models_changed(self) -> None:
        """`_on_shape_models_changed`/`_on_flat_field_references_changed` ile AYNI desen:
        bir ONNX modeli kaydedilir/silinirse, `ml.onnx_detect` düğümü o an seçiliyse
        parametre panelindeki 'Model(ler)' açılır listesi hemen günceli göstersin diye
        formu yeniden kurar."""
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            return
        node = self.graph.nodes[node_id]
        if node.op_id != "ml.onnx_detect":
            return
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)

    def _on_open_help(self) -> None:
        if self._help_dialog is not None:
            self._help_dialog.show()
            self._help_dialog.raise_()
            self._help_dialog.activateWindow()
            return
        dialog = HelpDialog(parent=self)
        dialog.destroyed.connect(lambda: setattr(self, "_help_dialog", None))
        self._help_dialog = dialog
        dialog.show()

    def _on_open_custom_filter_editor(self) -> None:
        if self._custom_filter_dialog is not None:
            self._custom_filter_dialog.show()
            self._custom_filter_dialog.raise_()
            self._custom_filter_dialog.activateWindow()
            return
        dialog = CustomFilterDialog(
            frame_provider=self._current_preview_image,
            is_op_in_use=lambda op_id: any(n.op_id == op_id for n in self.graph.nodes.values()),
            parent=self,
        )
        dialog.filters_changed.connect(self.operator_library.refresh)
        dialog.destroyed.connect(lambda: setattr(self, "_custom_filter_dialog", None))
        self._custom_filter_dialog = dialog
        dialog.show()

    def _current_preview_image(self) -> np.ndarray | None:
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            return None
        node = self.graph.nodes[node_id]
        result = self.engine.evaluate(node_id)
        if not result.ok:
            return None
        op_cls = self.registry.get(node.op_id)
        return extract_preview_image(op_cls, result.outputs)

    def _active_mm_per_px(self) -> float | None:
        """Checkerboard-tabanlı DENEYSEL otomatik akış (lens profili + netlik->mesafe modeli +
        son kamera karesi) varsa ONU kullanır — `_maybe_apply_auto_mm_per_px` (tick-driven,
        `region_props`'a otomatik uygulanan) ile AYNI hesaplamayı (`_compute_auto_mm_per_px`)
        paylaşır. Yoksa ESKİ yönteme (`HeightScaleModel` + elle ayarlanmış `_active_height_mm`)
        düşer. Bu paylaşım olmadan 'Ölçüm Aracı' gibi tek seferlik tüketiciler otomatik akışı
        hiç göremiyor, sadece eski (kullanılmayan) modele bakıp 'kalibrasyon yok' diyordu —
        gerçek bir kullanıcı raporuyla doğrulandı."""
        auto_mm_per_px = self._compute_auto_mm_per_px()
        if auto_mm_per_px is not None:
            return auto_mm_per_px

        if self._active_height_mm is None or self._height_scale_model is None:
            return None
        try:
            scale = self._height_scale_model.predict_scale(self._active_height_mm)
        except (RuntimeError, ValueError):
            return None
        return 1.0 / scale

    def _compute_auto_mm_per_px(self) -> float | None:
        """`_maybe_apply_auto_mm_per_px`'in içindeki DENEYSEL mm/px hesabının, throttling/
        dirty-marking YAPMAYAN saf hâli — hem tick döngüsü hem de `_active_mm_per_px` (Ölçüm
        Aracı gibi anlık tüketiciler) tarafından paylaşılır, böylece iki yer birbirinden
        FARKLI/tutarsız bir değer vermez.

        Üç kaynak sırayla denenir: (1) `_plane_rectification` (EN YÜKSEK öncelik — kare zaten
        `_on_camera_tick`'te bu homografiyle rektifiye edildiği için `mm_per_px` orada SABİT
        ve konum-bağımsızdır, `fx`'e bile gerek yok); (2) yoksa `_reference_distance_mm`
        (SABİT, `LensCalibrationDialog` galerisinde işaretlenen "bant seviyesi" karesinin
        solvePnP mesafesi — tam doğrulukta ama kamera düzleme tam dik değilse konuma göre
        hatalı); (3) yoksa `_focus_distance_model` (DENEYSEL, kareden canlı netlik ölçümü) —
        gerçek kullanımda genelde ~0 korelasyon çıktığı için SADECE ikisi de yokken bir
        yedek olarak denenir."""
        if self._plane_rectification is not None:
            return self._plane_rectification.mm_per_px

        if self._active_lens_profile is None:
            return None
        fx = float(self._active_lens_profile.camera_matrix[0, 0])
        if fx <= 0:
            return None

        if self._reference_distance_mm is not None:
            return self._reference_distance_mm / fx

        if self._last_camera_frame is None or self._focus_distance_model is None:
            return None
        if self._focus_distance_model.slope is None:
            return None
        try:
            distance_mm = self._focus_distance_model.predict_distance(focus_measure(self._last_camera_frame))
        except RuntimeError:
            return None
        return distance_mm / fx

    def _on_open_measurement_tool(self) -> None:
        if self._measurement_tool_dialog is not None:
            self._measurement_tool_dialog.close()
        image = self._current_preview_image()
        dialog = MeasurementToolDialog(image, self._active_mm_per_px(), parent=self)
        dialog.destroyed.connect(lambda: setattr(self, "_measurement_tool_dialog", None))
        self._measurement_tool_dialog = dialog
        dialog.show()

    def _on_set_active_height(self) -> None:
        current = self._active_height_mm if self._active_height_mm is not None else 0.0
        height, ok = QInputDialog.getDouble(
            self, "Aktif Yükseklik", "Yükseklik (mm):", current, 0.0, 100000.0, 2
        )
        if not ok:
            return
        self._set_active_height_mm(height)

    def _on_open_video_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Video Dosyası Aç", "", _VIDEO_FILE_FILTER)
        if not path:
            return
        try:
            self.start_camera(VideoFileSource(path))
        except RuntimeError as exc:
            show_error(self, "Video Açılamadı", exc)

    def _on_stop_camera(self) -> None:
        self.stop_camera()

    def _on_camera_tick(self) -> None:
        if self._camera_source is None:
            return
        try:
            frame = self._camera_source.read()
        except Exception as exc:
            # Bu 100ms'de bir çalışan bir zamanlayıcı — burada modal bir hata diyaloğu ASLA
            # gösterilmemeli: TriggerMode gibi bir ayar kamerayı geçici olarak kare üretemez
            # duruma soksa (ör. tetik beklerken) bu tick'te sürekli aynı istisna tekrar
            # tekrar fırlar, her 100ms'de bir yeni modal açılıp kullanıcının uygulamayı
            # kapatmasını bile engelleyen bir "diyalog fırtınası"na yol açardı (gerçek bir
            # kullanıcı raporuyla doğrulandı). Bunun yerine sadece durum etiketinde göster.
            self.status_label.setText(f"Kare okunamadı: {exc}")
            return
        if frame is None:
            return
        if self._active_lens_profile is not None:
            frame = undistort(frame, self._active_lens_profile)
        if self._plane_rectification is not None:
            # Kamera referans düzleme tam dik olmasa bile konum-bağımsız doğru ölçüm için:
            # ham kareyi doğrudan "kuşbakışı" (top-down) bir görünüme çevirir — bundan sonra
            # HER piksel sabit `_plane_rectification.mm_per_px` kadar gerçek mm'i temsil eder
            # (bkz. `core/plane_rectification.py`, gerçek kullanıcı raporu: aynı cisim
            # ekranın farklı yerlerinde 6mm/5.5mm ölçülüyordu).
            frame = self._plane_rectification.rectify(frame)
        self._maybe_apply_auto_mm_per_px(frame)
        node_id = self._find_or_create_image_source()
        self.engine.inject_result(node_id, {"image": frame})
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _on_operator_unchecked(self, op_id: str) -> None:
        node_id = next((nid for nid, n in self.graph.nodes.items() if n.op_id == op_id), None)
        if node_id is not None:
            self.remove_operator(node_id)

    def _on_save_recipe(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Reçeteyi Kaydet", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.save_recipe_to(path)
        except OSError as exc:
            show_error(self, "Reçete Kaydedilemedi", exc, hint="Dosya yolunu ve yazma izinlerini kontrol edin.")

    def _on_load_recipe(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Reçete Yükle", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.load_recipe_from(path)
        except (ImgflowError, OSError, ValueError, KeyError) as exc:
            show_error(
                self,
                "Reçete Yüklenemedi",
                exc,
                hint=(
                    "Reçete dosyası bozuk veya eksik görünüyor — elle düzenlediyseniz kontrol "
                    "edin ya da orijinal reçeteyi tekrar kaydedin."
                ),
            )

    def _on_run_batch(self) -> None:
        node_id = self.selected_node_id()
        if node_id is None:
            QMessageBox.information(self, "Toplu İşlem", "Önce ölçümlerin alınacağı adımı seçin.")
            return
        input_dir = QFileDialog.getExistingDirectory(self, "Görüntü Klasörü Seç")
        if not input_dir:
            return
        output_csv, _filter = QFileDialog.getSaveFileName(self, "CSV Dosyası", "", "CSV (*.csv)")
        if not output_csv:
            return
        try:
            rows = self.run_batch_process(node_id, input_dir, output_csv)
        except (ValueError, OSError) as exc:
            show_error(self, "Toplu İşlem Hatası", exc)
            return
        error_count = sum(1 for row in rows if "error" in row)
        QMessageBox.information(
            self, "Toplu İşlem Tamamlandı", f"{len(rows)} satır yazıldı ({error_count} hata)."
        )

    def _on_pipeline_changed(self) -> None:
        self.engine.mark_all_dirty()
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _on_step_selected(self, node_id: str) -> None:
        self._selected_node_id = node_id
        node = self.graph.nodes[node_id]
        op_cls = self.registry.get(node.op_id)
        self.description_label.setText(description_for(node.op_id))
        self.param_form.set_params(op_cls.params, node.params)
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _on_params_changed(self, values: dict[str, Any]) -> None:
        if self._selected_node_id is None:
            return
        # Node/graph durumu HER zaman senkron güncellenir (ör. bir test/çağıran hemen ardından
        # engine.evaluate() çağırdığında güncel parametreleri görsün) — sadece PAHALI kısım
        # (gerçek yeniden hesaplama + önizleme çizimi) debounce edilir, aksi halde slider
        # sürüklenirken her piksel hareketinde tam pipeline yeniden çalışır ve UI kilitlenir.
        self.graph.nodes[self._selected_node_id].params = values
        self.engine.mark_dirty(self._selected_node_id)
        self._param_debounce_timer.start(_PARAM_DEBOUNCE_MS)

    def _flush_pending_params(self) -> None:
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _on_enum_choice_selected(self, param_name: str, choice: str) -> None:
        if self._selected_node_id is None:
            return
        node = self.graph.nodes[self._selected_node_id]
        node.params[param_name] = choice
        self.engine.mark_dirty(self._selected_node_id)
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _on_roi_canvas_changed(self, x: int, y: int, w: int, h: int) -> None:
        node_id = self._selected_node_id
        if node_id is None:
            return
        node = self.graph.nodes[node_id]
        if node.op_id != _ROI_OP_ID:
            return
        node.params = {**node.params, "enabled": True, "shape": "RECT", "x": x, "y": y, "w": w, "h": h}
        self.engine.mark_dirty(node_id)
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)
        self._refresh_preview()

    def _on_roi_circle_canvas_changed(self, cx: int, cy: int, r: int) -> None:
        node_id = self._selected_node_id
        if node_id is None:
            return
        node = self.graph.nodes[node_id]
        if node.op_id != _ROI_OP_ID:
            return
        node.params = {**node.params, "enabled": True, "shape": "CIRCLE", "cx": cx, "cy": cy, "r": r}
        self.engine.mark_dirty(node_id)
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            self.image_view.set_image(None)
            self.image_view.set_measurements(None)
            self.image_view.set_editing_enabled(False)
            self.status_label.setText("")
            self.results_panel.set_measurements([])
            return

        node = self.graph.nodes[node_id]
        result = self.engine.evaluate(node_id)
        if not result.ok:
            self.image_view.set_image(None)
            self.image_view.set_measurements(None)
            self.image_view.set_editing_enabled(False)
            self.status_label.setText(f"Hata: {result.error}")
            self.results_panel.set_measurements([])
            return

        self.results_panel.set_measurements((result.outputs or {}).get("measurements"))

        is_roi_step = node.op_id == _ROI_OP_ID
        if is_roi_step:
            # ROI'yi tam (kırpılmamış) kare üzerinde çizebilmek için enabled=False ile bir kerelik
            # deneme çalıştırması yapılır; başarısız olursa normal (muhtemelen kırpılmış) sonuca dönülür.
            preview_result = self.engine.trial_run(node_id, {**node.params, "enabled": False})
            if preview_result.ok:
                result = preview_result

        op_cls = self.registry.get(node.op_id)
        image = extract_preview_image(op_cls, result.outputs)
        if is_roi_step:
            # ROI dikdörtgen/daire çizim overlay'i, tam bu görüntünün piksel koordinatlarına
            # göre konumlanır (bkz. aşağıdaki `set_roi`/`set_roi_circle`) — "Normal"/"İkisi Bir
            # Arada" görünüm modu bunu farklı bir taban görüntüye ya da yan yana bileşik bir
            # görüntüye taşırsa koordinatlar kayar. Bu yüzden ROI adımı seçiliyken görünüm
            # modundan bağımsız olarak her zaman filtrelenmiş (kırpılmamış deneme) görüntü kullanılır.
            display_image = image
            hover_measurements: list[dict[str, Any]] | None = None
        else:
            measurements = (result.outputs or {}).get("measurements")
            mm_per_px = float(node.params.get("mm_per_px", 0.0))
            display_image, hover_measurements = self._compose_display_image(image, measurements, mm_per_px)
        self.image_view.set_image(display_image)
        self.image_view.set_measurements(hover_measurements)
        self.image_view.set_editing_enabled(is_roi_step)
        if is_roi_step:
            shape = node.params.get("shape", "RECT")
            self.image_view.set_shape(shape)
            if shape == "CIRCLE":
                self.image_view.set_roi_circle(
                    int(node.params.get("cx", 100)),
                    int(node.params.get("cy", 100)),
                    int(node.params.get("r", 50)),
                )
            else:
                self.image_view.set_roi(
                    int(node.params.get("x", 0)),
                    int(node.params.get("y", 0)),
                    int(node.params.get("w", 100)),
                    int(node.params.get("h", 100)),
                )
        self.status_label.setText("" if image is not None else "Bu adım için görsel çıktı yok.")

    def _compose_display_image(
        self,
        filtered_image: np.ndarray | None,
        measurements: list[dict[str, Any]] | None,
        mm_per_px: float,
    ) -> tuple[np.ndarray | None, list[dict[str, Any]] | None]:
        """`_view_mode`'a göre canlı önizlemede gösterilecek (görüntü, üzerine-gelme-ölçümleri)
        çiftini üretir. İkinci eleman `image_view.set_measurements()`'e verilir -- fare bir
        nesnenin üzerine geldiğinde sol paneldeki özellik etiketini güncellemek için "GÖSTERİLEN
        görüntünün koordinat sistemine" ölçeklenmiş `bbox_x/y/w/h` içerir (bkz.
        `_scale_measurements_for_display`); "İkisi Bir Arada" modunda iki farklı ölçek/offset'i
        (sol/sağ yarı + kırpma ölçeği) tek bir tutarlı koordinat sistemine indirmenin karmaşıklığı
        bu özelliğin değerine değmediği için o modda `None` döner (üzerine gelme desteklenmez).

        "filtered" seçili adımın (ör. `analysis.region_props`'un labels tabanlı overlay'i)
        çıktısıdır -- eski/varsayılan davranış, değişmedi. "normal"/"both" ise kalibrasyon
        düzeltmeleri (undistort/plane rectification) uygulanmış ama pipeline'ın kendi
        filtresinden (eşikleme/HSV vb.) geçmemiş ham kareyi kullanır; ölçüm varsa
        `draw_measurements_overlay` ile bunun üzerine de çizilir -- kullanıcı ürünü
        filtrelenmiş görüntüde ayırıp ölçse bile boyutları normal (renkli) kare üzerinde de
        görebilsin diye (ölçüm hesaplaması hangi görüntü üzerinde gösterildiğinden bağımsız).

        Her iki taraf da ekrana aktarılmadan ÖNCE `_MAX_PREVIEW_DIM`'e küçültülür (bkz. orada
        ki not) -- gerçek kullanıcı raporu: yüksek çözünürlüklü canlı kamera akışında "Bölge
        Ölçümü" adımının parametrelerini (tolerans vb.) değiştirirken belirgin kasma oluyordu."""
        if self._view_mode == "filtered":
            if filtered_image is None:
                return None, None
            capped, scale = _cap_preview_size_with_scale(filtered_image)
            return capped, _scale_measurements_for_display(measurements, scale)

        normal_base = self._get_normal_base_image()
        normal_image = normal_base.copy() if normal_base is not None else None
        if normal_image is not None and measurements:
            # Overlay, koordinatların (bbox_x/y vb.) hesaplandığı TAM çözünürlükte çizilir --
            # boyut sınırlaması SADECE bundan sonra (görüntüleme amaçlı) uygulanır, aksi halde
            # kutular/yazılar küçültülmüş görüntüde yanlış yere düşer.
            normal_image = draw_measurements_overlay(normal_image, measurements, mm_per_px)
        normal_capped, normal_scale = (
            _cap_preview_size_with_scale(normal_image) if normal_image is not None else (None, 1.0)
        )

        if self._view_mode == "normal":
            if normal_capped is not None:
                return normal_capped, _scale_measurements_for_display(measurements, normal_scale)
            if filtered_image is None:
                return None, None
            filtered_capped, filtered_scale = _cap_preview_size_with_scale(filtered_image)
            return filtered_capped, _scale_measurements_for_display(measurements, filtered_scale)

        # "both" -- bkz. yukarıdaki docstring, üzerine gelme bilgisi desteklenmez.
        filtered_capped = _cap_preview_size(filtered_image) if filtered_image is not None else None
        if filtered_capped is None:
            return normal_capped, None
        if normal_capped is None:
            return filtered_capped, None
        return self._compose_side_by_side(filtered_capped, normal_capped), None

    def _get_normal_base_image(self) -> np.ndarray | None:
        """Pipeline'ın kendi filtresinden geçmemiş, ama kalibrasyon düzeltmeleri (varsa
        undistort + plane rectification) uygulanmış "ham" kareyi döner. Kamera aktifse
        `_on_camera_tick`'te zaten üretilmiş `_last_camera_frame` kullanılır (yeniden
        hesaplamaya gerek yok); değilse (statik görüntü modu) pipeline'daki `io.image_source`
        node'u -- varsa -- doğrudan değerlendirilir."""
        if self._camera_source is not None and self._last_camera_frame is not None:
            return self._last_camera_frame
        source_id = next(
            (nid for nid, n in self.graph.nodes.items() if n.op_id == "io.image_source"), None
        )
        if source_id is None:
            return None
        result = self.engine.evaluate(source_id)
        if not result.ok:
            return None
        op_cls = self.registry.get(self.graph.nodes[source_id].op_id)
        return extract_preview_image(op_cls, result.outputs)

    @staticmethod
    def _compose_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_bgr = _to_bgr_uint8(left)
        right_bgr = _to_bgr_uint8(right)
        height = max(left_bgr.shape[0], right_bgr.shape[0])

        def _scale_to_height(img: np.ndarray) -> np.ndarray:
            if img.shape[0] == height:
                return img
            scale = height / img.shape[0]
            new_w = max(1, int(round(img.shape[1] * scale)))
            return cv2.resize(img, (new_w, height))

        divider = np.full((height, 4, 3), 96, dtype=np.uint8)
        return np.hstack([_scale_to_height(left_bgr), divider, _scale_to_height(right_bgr)])
