"""imgflow ana penceresi: checkbox'lı operatör kütüphanesi + doğrusal pipeline şeridi
+ görüntü önizleme (ana alan) + parametre paneli.

Bir operatör kütüphaneden işaretlendiğinde LinearPipeline'a eklenir ve otomatik olarak
zincire (birincil portlar üzerinden) bağlanır; işaret kaldırılınca çıkarılır. Parametre
değişiklikleri motorun dirty/cache mekanizmasıyla yalnızca gerekli node'ları yeniden
hesaplattırır.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from typing import Any, Callable

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.batch import iter_images, run_batch
from imgflow.core.camera_params import CameraParameterController
from imgflow.core.camera_source import BaslerCameraSource, CameraSource, UsbCameraSource, VideoFileSource
from imgflow.core.engine import ExecutionEngine
from imgflow.core.errors import ImgflowError
from imgflow.core.focus_distance import FocusDistanceModel
from imgflow.core.focus_metric import focus_measure
from imgflow.core.graph import Graph, Node
from imgflow.core.height_scale_calibration import HeightScaleModel
from imgflow.core.lens_calibration import LensProfile, undistort
from imgflow.core.linear_pipeline import LinearPipeline
from imgflow.core.plane_rectification import PlaneRectification
from imgflow.core.roi import parse_roi_list
from imgflow.core.params import ParamType, defaults_for
from imgflow.core import capture_store
from imgflow.io_utils import calibration_store, camera_settings_store
from imgflow.io_utils.app_log import get_logger
from imgflow.io_utils.image_io import save_image
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
from imgflow.ui.panels.operator_library import OperatorLibrary, description_for, label_for
from imgflow.ui.panels.pipeline_steps import PipelineStepsPanel
from imgflow.ui.widgets.enum_gallery import EnumGallery
from imgflow.ui.widgets.image_view import extract_preview_image, normalize_to_uint8
from imgflow.ui.widgets.measurements_summary import MeasurementsSummaryPanel
from imgflow.ui.widgets.param_form import ParamForm
from imgflow.ui.widgets.roi_canvas import RoiCanvas

_IMAGE_FILE_FILTER = "Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
_VIDEO_FILE_FILTER = "Videolar (*.mp4 *.avi *.mov *.mkv)"
_ROI_OP_ID = "roi.region"
_MANUAL_ROI_OP_IDS = {"analysis.color_props", "analysis.texture_props"}
"""`manual_roi_enabled` parametresi olan operatörler -- seçiliyken ve bu parametre açıkken
`image_view` çoklu-ROI çizim moduna geçer (bkz. `RoiCanvas.set_multi_mode`)."""
_LABEL_MEASUREMENT_OP_IDS = {"analysis.region_props"}
"""Çıktısı bir `labels` (etiket haritası) girdisinden türeyen ölçüm operatörleri -- kendi
overlay'lerini zorunlu olarak İKİLİ (siyah/beyaz) bir siluet üzerine çizerler, çünkü
`LinearPipeline`'ın tek-girdi zincirinde kaynak görüntüye erişimleri YOKTUR (bkz. CLAUDE.md).
Gerçek kullanıcı raporu: "hsv ile bölge ölçümünde siyah beyaz yapıyor" -- renkli bir HSV
maskesinin ardından Bölge Ölçümü seçilince önizleme birden siyah/beyaza dönüyordu. Bu adımlar
seçiliyken önizleme tabanı `_measurement_overlay_base` ile zincirdeki en yakın GERÇEK
(uint8) görüntüye taşınır; segmentasyonun kendisini görmek isteyen kullanıcı zaten bir önceki
adımı (Bağlı Bileşenler) seçebilir."""
_MM_PER_PX_OP_IDS = ("analysis.region_props", "geom.shape_match")
"""Kendi `mm_per_px` parametresi olan, otomatik kalibrasyon akışının (`_push_mm_per_px`/
`_region_props_needs_mm_per_px`) doldurduğu operatörler."""
_CAMERA_TICK_MS = 100
_CAMERA_DISCONNECT_THRESHOLD_TICKS = 20
"""Art arda ~2 saniye (bu kadar tick) kare okunamazsa kamera koptu sayılıp otomatik yeniden
bağlanma denenir — bkz. `_register_camera_failure`."""
_CAMERA_RECONNECT_COOLDOWN_TICKS = 30
"""Başarısız bir yeniden bağlanma denemesinden sonra bir sonrakini denemeden önce beklenecek
tick sayısı (~3 saniye) — kablo hâlâ takılı değilken cihaz numaralandırmayı (`Enumerate
Devices` gibi pahalı olabilecek çağrıları) her 100ms'de bir tekrar tekrar denememek için."""
_ENUM_GALLERY_TICK_STRIDE = 5
"""Seçili adımda bir enum parametre varsa (ör. eşikleme modu), galerinin HER seçeneği için
`engine.trial_run` (önbelleksiz, tam operatör çalıştırma) + pixmap render yapılır — bunu her
kamera tick'inde (100ms/saniyede 10 kez) çalıştırmak, kullanıcı galeriye hiç bakmıyor olsa
bile gereksiz CPU harcar. `_AUTO_HEIGHT_TICK_STRIDE` ile AYNI desende sadece her 5 tick'te
bir (~2Hz) yenilenir — canlı galeri hâlâ günceller, sadece insan gözünün ayırt edemeyeceği
kadar sık değil. `_refresh_enum_gallery`'nin DİĞER çağrı noktaları (adım seçimi, parametre
debounce flush'ı, enum seçimi, ROI değişikliği) GERÇEK değişiklik olayları olduğu için
DOKUNULMADAN kalır, sadece bu tick-güdümlü çağrı noktası kısıtlanır."""
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


_ROI_SHIFT_X_KEYS = ("bbox_x", "centroid_x", "obb_cx")
_ROI_SHIFT_Y_KEYS = ("bbox_y", "centroid_y", "obb_cy")
_ROI_SHIFT_MM_X_KEYS = ("bbox_mm_x", "centroid_mm_x")
_ROI_SHIFT_MM_Y_KEYS = ("bbox_mm_y", "centroid_mm_y")


def _shift_measurements_for_roi_offset(
    measurements: list[dict[str, Any]], offset_x: int, offset_y: int, mm_per_px: float
) -> list[dict[str, Any]]:
    """Upstream'de aktif bir `roi.region` varsa, `analysis.region_props`/`analysis.color_props`/
    `analysis.texture_props` gibi operatörlerin `bbox_x/y`, `centroid_x/y`, `obb_cx/cy`
    (ve mm karşılıkları) alanları KIRPILMIŞ görüntüye görecedir -- bu, TAM çözünürlüklü ham
    kare üzerine çizildiğinde (bkz. `_compose_display_image`'in "normal"/"both" dalı) ROI'nin
    sol-üst köşesi kadar KAYMIŞ kutulara yol açar. Sadece BİLİNEN piksel-konumu alanları
    kaydırılır -- `bbox_w/h`, `obb_w/h`, `area`, `*_mm2` gibi boyut/alan alanları bir kırpmadan
    ETKİLENMEDİĞİ için dokunulmaz; `shape_match`'in `x`/`y` poz alanları gibi FARKLI anlam
    taşıyan alanlar da (anahtar adı eşleşmediği için) kazara kaydırılmaz."""
    shifted = []
    for m in measurements:
        m2 = dict(m)
        for key in _ROI_SHIFT_X_KEYS:
            if key in m2:
                m2[key] = m2[key] + offset_x
        for key in _ROI_SHIFT_Y_KEYS:
            if key in m2:
                m2[key] = m2[key] + offset_y
        if mm_per_px > 0:
            for key in _ROI_SHIFT_MM_X_KEYS:
                if key in m2:
                    m2[key] = m2[key] + offset_x * mm_per_px
            for key in _ROI_SHIFT_MM_Y_KEYS:
                if key in m2:
                    m2[key] = m2[key] + offset_y * mm_per_px
        shifted.append(m2)
    return shifted


def _to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """"İkisi Bir Arada" görünümde `np.hstack` iki görüntünün aynı dtype/kanal sayısında
    olmasını gerektirir -- filtrelenmiş taraf tek kanallı (ör. eşikleme/gri) olabilirken
    normal taraf her zaman BGR'dir, bu yüzden birleştirmeden önce ikisi de 3 kanallı uint8'e getirilir."""
    normalized = normalize_to_uint8(image)
    if normalized.ndim == 2:
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    return normalized


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


def _get_normal_base_image(
    graph: Graph,
    engine: ExecutionEngine,
    registry: Any,
    camera_active: bool,
    last_camera_frame: np.ndarray | None,
) -> np.ndarray | None:
    """Pipeline'ın kendi filtresinden geçmemiş, ama kalibrasyon düzeltmeleri (varsa
    undistort + plane rectification) uygulanmış "ham" kareyi döner. Kamera aktifse
    `_on_camera_tick`'te zaten üretilmiş `last_camera_frame` kullanılır (yeniden
    hesaplamaya gerek yok); değilse (statik görüntü modu) pipeline'daki `io.image_source`
    node'u -- varsa -- doğrudan değerlendirilir."""
    if camera_active and last_camera_frame is not None:
        return last_camera_frame
    source_id = next((nid for nid, n in graph.nodes.items() if n.op_id == "io.image_source"), None)
    if source_id is None:
        return None
    result = engine.evaluate(source_id)
    if not result.ok:
        return None
    op_cls = registry.get(graph.nodes[source_id].op_id)
    return extract_preview_image(op_cls, result.outputs)


def _measurement_overlay_base(
    graph: Graph,
    pipeline_order: list[str],
    node_id: str,
    engine: ExecutionEngine,
    registry: Any,
) -> np.ndarray | None:
    """`node_id`'den ÖNCEKİ zincirde, gösterilebilir GERÇEK bir görüntü üreten en yakın adımın
    çıktısını döner (yoksa `None`).

    "Gerçek görüntü" ölçütü `dtype == uint8`: `segment.connected_components`'ın `labels`
    çıktısı int32 bir etiket haritasıdır, bu ölçüt onu doğal olarak atlayıp bir üstteki
    (ör. `color.hsv`'nin renkli maskeli) çıktısına iner. Adımlar bu noktada `engine`'de zaten
    değerlendirilmiş/önbelleklenmiş olduğundan buradaki `evaluate` çağrıları cache-hit'tir."""
    if node_id not in pipeline_order:
        return None
    for prev_id in reversed(pipeline_order[: pipeline_order.index(node_id)]):
        node = graph.nodes.get(prev_id)
        if node is None:
            continue
        result = engine.evaluate(prev_id)
        if not result.ok:
            return None
        image = extract_preview_image(registry.get(node.op_id), result.outputs)
        if image is not None and image.dtype == np.uint8:
            return image
    return None


def _cumulative_roi_offset(graph: Graph, pipeline_order: list[str], target_node_id: str) -> tuple[int, int]:
    """`target_node_id`'den ÖNCEKİ (upstream) zincirde aktif bir/birden fazla `roi.region`
    varsa, toplam kırpma ofsetini (x,y) döner -- `_shift_measurements_for_roi_offset`
    bunu "normal"/"both" görünüm modunda ölçüm koordinatlarını düzeltmek için kullanır.
    Dairesel ROI'nin ofseti, `RoiCircle.bounding_rect()`'in ürettiği sınırlayıcı kutunun
    sol-üst köşesidir (`core/roi.py` ile AYNI hesap: `cx-r, cy-r`, negatif olamaz).
    Kasıtlı olarak `RoiRect`/`RoiCircle.clamp()`'i burada TEKRAR ÜRETMİYORUZ (görüntü
    boyutu bu noktada bilinmiyor/gereksiz karmaşıklık) -- `roi_canvas.py` zaten kullanıcının
    ROI'yi sürüklerken görüntü sınırları İÇİNDE tutmasını sağlıyor, tıpkı `image_view.
    set_roi`/`set_roi_circle`'ın da ham parametreleri doğrudan kullanması gibi."""
    offset_x = offset_y = 0
    if target_node_id not in pipeline_order:
        return 0, 0
    target_index = pipeline_order.index(target_node_id)
    for node_id in pipeline_order[:target_index]:
        node = graph.nodes.get(node_id)
        if node is None or node.op_id != _ROI_OP_ID or not node.params.get("enabled", False):
            continue
        if node.params.get("shape", "RECT") == "CIRCLE":
            r = int(node.params.get("r", 50))
            offset_x += max(0, int(node.params.get("cx", 100)) - r)
            offset_y += max(0, int(node.params.get("cy", 100)) - r)
        else:
            offset_x += int(node.params.get("x", 0))
            offset_y += int(node.params.get("y", 0))
    return offset_x, offset_y


_ROI_CONTEXT_BORDER_COLOR = (0, 200, 255)
""""ROI Bağlamda" görünümünde işlenmiş bölgenin sınırını gösteren çerçeve rengi (turuncu-sarı)
-- ölçüm overlay'lerinin yeşil/kırmızısıyla karışmasın diye ayrı bir ton."""
_ROI_CONTEXT_BORDER_PX = 2


def _last_active_roi_is_circle(graph: Graph, pipeline_order: list[str], target_node_id: str) -> bool:
    """Hedef adımdan ÖNCEKİ zincirdeki SON aktif `roi.region` adımı DAİRE modunda mı?

    `roi.region`'ın daire modu (bkz. `operators/builtin/roi.py::_run_circle`) çıktıyı dairenin
    KARE sınırlayıcı kutusuna kırpar ve kutu içinde dairenin DIŞINDA kalan pikselleri siyaha
    boyar. "ROI Bağlamda" görünümü bu kareyi olduğu gibi ham karenin üzerine yapıştırdığında
    o siyah köşeler gerçek görüntü içeriğini SİLİYORDU -- gerçek kullanıcı raporu: "circle roi
    aldıktan sonra... circle'ın etrafında bir kutu oluşturdu ve içini siyah yaptı"."""
    if target_node_id not in pipeline_order:
        return False
    for node_id in reversed(pipeline_order[: pipeline_order.index(target_node_id)]):
        node = graph.nodes.get(node_id)
        if node is None or node.op_id != _ROI_OP_ID or not node.params.get("enabled", False):
            continue
        return node.params.get("shape", "RECT") == "CIRCLE"
    return False


def _paste_filtered_into_frame(
    normal_base: np.ndarray | None,
    filtered_image: np.ndarray | None,
    graph: Graph,
    pipeline_order: list[str],
    node_id: str | None,
) -> np.ndarray | None:
    """Filtrelenmiş (ROI ile KIRPILMIŞ) sonucu, ham tam karenin İÇİNE kendi yerine yapıştırır.

    Gerçek kullanıcı isteği: "ROI uygulayınca başka filtreye geçince sadece ROI alanı
    görünüyor, ben ROI dışında kalan alanı da görmek istiyorum -- işlemli ve işlemsiz bölgeyi
    görmüş olurum böylece." "Filtrelenmiş" modu yalnızca kırpımı, "Normal" modu yalnızca ham
    kareyi gösteriyordu; bu mod ikisini TEK karede birleştirir.

    Zincirdeki `roi.region` adımlarının toplam ofseti (`_cumulative_roi_offset`, MEVCUT)
    yapıştırma konumunu verir. Filtrelenmiş görüntü ham kareyle AYNI boyuttaysa (zincirde ROI
    yok ya da kırpma kapalı) yapıştırmanın anlamı kalmaz ve `None` dönülür -- çağıran taraf
    normal "filtrelenmiş" davranışına düşer. Tek kanallı (ör. eşikleme) çıktı, renkli tam
    kareye yapıştırılabilmesi için BGR'ye çevrilir."""
    if normal_base is None or filtered_image is None or node_id is None:
        return None
    offset_x, offset_y = _cumulative_roi_offset(graph, pipeline_order, node_id)
    base = np.ascontiguousarray(normal_base).copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    patch = filtered_image
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    if patch.shape[:2] == base.shape[:2] and offset_x == 0 and offset_y == 0:
        return None  # kırpma yok -- "Filtrelenmiş" ile aynı olurdu

    h, w = patch.shape[:2]
    y0, x0 = max(0, offset_y), max(0, offset_x)
    y1, x1 = min(base.shape[0], y0 + h), min(base.shape[1], x0 + w)
    if y1 <= y0 or x1 <= x0:
        return None
    visible = patch[: y1 - y0, : x1 - x0]
    if _last_active_roi_is_circle(graph, pipeline_order, node_id) and h == w:
        # Daire ROI: kırpım dairenin KARE kutusudur, köşeleri `roi.region` tarafından zaten
        # siyaha boyanmıştır (bkz. `_last_active_roi_is_circle`). Bu kareyi olduğu gibi
        # yapıştırmak o siyah köşelerle ham karenin gerçek içeriğini SİLERDİ -- sadece daire
        # İÇİ yapıştırılır, sınır da dikdörtgen yerine DAİRE olarak çizilir. `h == w` koşulu
        # savunma amaçlı: daireden sonra zincirde başka bir kırpma varsa (kare olmayan yama)
        # maskenin geometrisi artık dairenin kendisi olmadığından eski davranışa düşülür.
        radius = h // 2
        mask = np.zeros((h, w), dtype=bool)
        yy, xx = np.ogrid[:h, :w]
        # `roi.region::_run_circle` ile BİREBİR aynı ölçüt (orada "dışarısı" `> r**2`).
        mask[(xx - radius) ** 2 + (yy - radius) ** 2 <= radius**2] = True
        region = base[y0:y1, x0:x1]
        visible_mask = mask[: y1 - y0, : x1 - x0]
        region[visible_mask] = visible[visible_mask]
        cv2.circle(
            base,
            (x0 + radius, y0 + radius),
            radius,
            _ROI_CONTEXT_BORDER_COLOR,
            _ROI_CONTEXT_BORDER_PX,
        )
        return base

    base[y0:y1, x0:x1] = visible
    cv2.rectangle(
        base, (x0, y0), (x1 - 1, y1 - 1), _ROI_CONTEXT_BORDER_COLOR, _ROI_CONTEXT_BORDER_PX
    )
    return base


def _compose_display_image(
    filtered_image: np.ndarray | None,
    measurements: list[dict[str, Any]] | None,
    mm_per_px: float,
    node_id: str | None,
    *,
    view_mode: str,
    graph: Graph,
    pipeline_order: list[str],
    engine: ExecutionEngine,
    registry: Any,
    camera_active: bool,
    last_camera_frame: np.ndarray | None,
) -> tuple[np.ndarray | None, list[dict[str, Any]] | None]:
    """`view_mode`'a göre canlı önizlemede gösterilecek (görüntü, üzerine-gelme-ölçümleri)
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
    Ölçümü" adımının parametrelerini (tolerans vb.) değiştirirken belirgin kasma oluyordu.

    **ROI ofseti:** zincirde bu adımdan ÖNCE aktif bir `roi.region` varsa (ör. "önce bir
    bölgeye kırp, sonra o bölgede Bağlı Bileşenler/Bölge Ölçümü uygula" -- kalibrasyonu
    BOZMADAN tek bir ürüne/bölgeye odaklanmanın yolu, bkz. CLAUDE.md FAZ3 notu), ölçüm
    koordinatları (`bbox_x/y`, `centroid_x/y`, `obb_cx/cy`) KIRPILMIŞ görüntüye görecedir.
    "filtered" modda sorun yok (o zaten kırpılmış görüntüyü kendi boyutunda gösteriyor),
    ama "normal"/"both" TAM ÇÖZÜNÜRLÜKLÜ ham kareyi kullandığından, ofset eklenmezse
    kutular ROI'nin sol-üst köşesi kadar KAYMIŞ görünür. `_cumulative_roi_offset` ile bu
    ofset hesaplanıp SADECE bu (normal/both) dalda ölçümlere eklenir -- alan/çevre/mm
    ölçümlerinin kendisi zaten etkilenmiyordu (mm_per_px bir kırpmadan bağımsız, piksel
    başına fiziksel boyut oranı), sadece EKRANDAKİ KUTU KONUMU kayıyordu."""
    if view_mode == "filtered":
        if filtered_image is None:
            return None, None
        capped, scale = _cap_preview_size_with_scale(filtered_image)
        return capped, _scale_measurements_for_display(measurements, scale)

    if measurements and node_id is not None:
        offset_x, offset_y = _cumulative_roi_offset(graph, pipeline_order, node_id)
        if offset_x or offset_y:
            measurements = _shift_measurements_for_roi_offset(measurements, offset_x, offset_y, mm_per_px)

    normal_base = _get_normal_base_image(graph, engine, registry, camera_active, last_camera_frame)
    normal_image = normal_base.copy() if normal_base is not None else None
    if normal_image is not None and measurements:
        # Overlay, koordinatların (bbox_x/y vb.) hesaplandığı TAM çözünürlükte çizilir --
        # boyut sınırlaması SADECE bundan sonra (görüntüleme amaçlı) uygulanır, aksi halde
        # kutular/yazılar küçültülmüş görüntüde yanlış yere düşer.
        normal_image = draw_measurements_overlay(normal_image, measurements, mm_per_px)
    normal_capped, normal_scale = (
        _cap_preview_size_with_scale(normal_image) if normal_image is not None else (None, 1.0)
    )

    if view_mode == "normal":
        if normal_capped is not None:
            return normal_capped, _scale_measurements_for_display(measurements, normal_scale)
        if filtered_image is None:
            return None, None
        filtered_capped, filtered_scale = _cap_preview_size_with_scale(filtered_image)
        return filtered_capped, _scale_measurements_for_display(measurements, filtered_scale)

    if view_mode == "in_context":
        composed = _paste_filtered_into_frame(
            normal_base, filtered_image, graph, pipeline_order, node_id
        )
        if composed is None:
            if filtered_image is None:
                return normal_capped, _scale_measurements_for_display(measurements, normal_scale)
            capped, scale = _cap_preview_size_with_scale(filtered_image)
            return capped, _scale_measurements_for_display(measurements, scale)
        if measurements:
            composed = draw_measurements_overlay(composed, measurements, mm_per_px)
        capped, scale = _cap_preview_size_with_scale(composed)
        return capped, _scale_measurements_for_display(measurements, scale)

    # "both" -- bkz. yukarıdaki docstring, üzerine gelme bilgisi desteklenmez.
    filtered_capped = _cap_preview_size(filtered_image) if filtered_image is not None else None
    if filtered_capped is None:
        return normal_capped, None
    if normal_capped is None:
        return filtered_capped, None
    return _compose_side_by_side(filtered_capped, normal_capped), None


@dataclass
class PreviewFrameResult:
    """`_build_preview_frame`'in SAF (Qt widget'larına hiç dokunmayan) çıktısı -- hem senkron
    çağrı yolu (`MainWindow._refresh_preview`) hem de arka plan `_LiveTickWorker` tarafından
    üretilir. UI thread'deki `MainWindow._apply_preview_frame_result` SADECE bu alanları
    widget'lara uygular, hiçbir yeniden hesaplama yapmaz -- bu simetri, worker'ın hiçbir Qt
    çağrısı yapmadan (thread güvenliği) TAMAMEN aynı sonucu üretebilmesini sağlar."""

    ok: bool
    error: str | None
    display_image: np.ndarray | None
    hover_measurements: list[dict[str, Any]] | None
    measurements: list[dict[str, Any]] | None
    status_text: str
    is_roi_step: bool
    manual_roi_active: bool
    roi_shape: str
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int
    roi_cx: int
    roi_cy: int
    roi_r: int
    manual_rois: list[tuple[int, int, int, int]] | None
    step_durations: list[tuple[str, float]] | None
    """(Türkçe adım etiketi, saniye) çiftleri, pipeline sırasıyla -- SADECE bu `evaluate()`
    çağrısında gerçekten çalıştırılan (bkz. `core/engine.py::ExecutionEngine._durations`)
    adımları içerir. Gerçek kullanıcı isteği: "her işlemin sonucunun süresi sonuçlar
    kısmında yazmalı" -- `MeasurementsSummaryPanel.set_step_durations`'a geçirilir."""


def _collect_step_durations(
    engine: ExecutionEngine, graph: Graph, pipeline_order: list[str]
) -> list[tuple[str, float]]:
    """`engine.durations`'ı (bkz. `core/engine.py`) pipeline SIRASIYLA, Türkçe adım
    etiketleriyle (`label_for`) bir listeye çevirir -- SADECE bu `evaluate()` çağrısında
    gerçekten çalıştırılan adımlar listede yer alır (cache-hit/üst-akış-hatası nedeniyle
    atlanan adımlar `engine.durations`'ta hiç yoktur, bkz. o modüldeki not)."""
    durations = engine.durations
    return [
        (label_for(graph.nodes[nid].op_id), durations[nid])
        for nid in pipeline_order
        if nid in durations and nid in graph.nodes
    ]


def _build_preview_frame(
    *,
    engine: ExecutionEngine,
    registry: Any,
    graph: Graph,
    pipeline_order: list[str],
    node_id: str | None,
    view_mode: str,
    camera_active: bool,
    last_camera_frame: np.ndarray | None,
) -> PreviewFrameResult:
    """`MainWindow._refresh_preview`'in eski gövdesinin SAF (hiçbir `self.*` widget'ına
    dokunmayan) hesaplama kısmı -- `_LiveTickWorker.run()` tarafından arka plan thread'inde,
    `MainWindow._refresh_preview` tarafından da UI thread'inde (tek-seferlik/on-demand
    çağrılar için) AYNEN çağrılır, kod tekrarı YOK. `engine`/`graph` çağırana göre ya canlı
    `self.engine`/`self.graph` (senkron yol) ya da worker'ın kendi `copy.deepcopy` kopyası
    üzerinde kurduğu taze bir `ExecutionEngine` (arka plan yolu) olabilir -- fonksiyonun
    kendisi ikisini de bilmez, sadece verilen `engine`/`graph` ile çalışır."""
    if node_id is None or node_id not in graph.nodes:
        return PreviewFrameResult(
            ok=True,
            error=None,
            display_image=None,
            hover_measurements=None,
            measurements=[],
            status_text="",
            is_roi_step=False,
            manual_roi_active=False,
            roi_shape="RECT",
            roi_x=0,
            roi_y=0,
            roi_w=100,
            roi_h=100,
            roi_cx=100,
            roi_cy=100,
            roi_r=50,
            manual_rois=None,
            step_durations=None,
        )

    node = graph.nodes[node_id]
    result = engine.evaluate(node_id)
    if not result.ok:
        return PreviewFrameResult(
            ok=False,
            error=str(result.error),
            display_image=None,
            hover_measurements=None,
            measurements=[],
            status_text=f"Hata: {result.error}",
            is_roi_step=False,
            manual_roi_active=False,
            roi_shape="RECT",
            roi_x=0,
            roi_y=0,
            roi_w=100,
            roi_h=100,
            roi_cx=100,
            roi_cy=100,
            roi_r=50,
            manual_rois=None,
            step_durations=_collect_step_durations(engine, graph, pipeline_order),
        )

    result_measurements = (result.outputs or {}).get("measurements")

    is_roi_step = node.op_id == _ROI_OP_ID
    manual_roi_active = node.op_id in _MANUAL_ROI_OP_IDS and bool(node.params.get("manual_roi_enabled", False))
    if is_roi_step:
        # ROI'yi tam (kırpılmamış) kare üzerinde çizebilmek için enabled=False ile bir kerelik
        # deneme çalıştırması yapılır; başarısız olursa normal (muhtemelen kırpılmış) sonuca dönülür.
        preview_result = engine.trial_run(node_id, {**node.params, "enabled": False})
        if preview_result.ok:
            result = preview_result

    op_cls = registry.get(node.op_id)
    image = extract_preview_image(op_cls, result.outputs)
    manual_rois: list[tuple[int, int, int, int]] | None = None
    if is_roi_step or manual_roi_active:
        # ROI/çoklu-ROI çizim overlay'i, tam bu görüntünün piksel koordinatlarına göre
        # konumlanır (bkz. aşağıdaki `set_roi`/`set_roi_circle`/`set_rois`) -- "Normal"/
        # "İkisi Bir Arada" görünüm modu ya da `_cap_preview_size_with_scale` küçültmesi
        # farklı bir taban görüntüye/ölçeğe taşırsa koordinatlar kayar. Bu yüzden ROI
        # düzenleme adımı seçiliyken görünüm modundan/küçültmeden bağımsız olarak her
        # zaman filtrelenmiş (kırpılmamış, tam çözünürlüklü) görüntü kullanılır.
        display_image = image
        hover_measurements: list[dict[str, Any]] | None = None
        if manual_roi_active and image is not None:
            rois = parse_roi_list(str(node.params.get("manual_rois", "[]")), image.shape[1], image.shape[0])
            manual_rois = [(r.x, r.y, r.w, r.h) for r in rois]
    else:
        measurements = (result.outputs or {}).get("measurements")
        mm_per_px = float(node.params.get("mm_per_px", 0.0))
        if node.op_id in _LABEL_MEASUREMENT_OP_IDS:
            # bkz. `_LABEL_MEASUREMENT_OP_IDS` -- ikili siluet yerine zincirdeki son gerçek
            # (ör. HSV'nin RENKLİ) görüntü üzerine çizilir. Kaynak bulunamazsa operatörün
            # kendi overlay'iyle (eski davranış) devam edilir.
            base = _measurement_overlay_base(graph, pipeline_order, node_id, engine, registry)
            if base is not None:
                image = draw_measurements_overlay(base, measurements or [], mm_per_px)
        display_image, hover_measurements = _compose_display_image(
            image,
            measurements,
            mm_per_px,
            node_id,
            view_mode=view_mode,
            graph=graph,
            pipeline_order=pipeline_order,
            engine=engine,
            registry=registry,
            camera_active=camera_active,
            last_camera_frame=last_camera_frame,
        )

    if manual_roi_active and image is not None:
        status_text = (
            "Sürükleyerek yeni ROI çizin · içine tıklayıp taşıyın · köşeden yeniden "
            "boyutlandırın · sağ tık ile silin."
        )
    else:
        status_text = "" if image is not None else "Bu adım için görsel çıktı yok."

    return PreviewFrameResult(
        ok=True,
        error=None,
        display_image=display_image,
        hover_measurements=hover_measurements,
        measurements=result_measurements,
        status_text=status_text,
        is_roi_step=is_roi_step,
        manual_roi_active=manual_roi_active,
        roi_shape=str(node.params.get("shape", "RECT")),
        roi_x=int(node.params.get("x", 0)),
        roi_y=int(node.params.get("y", 0)),
        roi_w=int(node.params.get("w", 100)),
        roi_h=int(node.params.get("h", 100)),
        roi_cx=int(node.params.get("cx", 100)),
        roi_cy=int(node.params.get("cy", 100)),
        roi_r=int(node.params.get("r", 50)),
        manual_rois=manual_rois,
        step_durations=_collect_step_durations(engine, graph, pipeline_order),
    )


_PARAM_DEBOUNCE_MS = 120
"""Parametre paneli (slider/spinbox) değişikliklerini toplayıp TEK bir yeniden hesaplamaya
indirger — pahalı operatörlerde (ör. geom.shape_match) slider sürüklenirken her piksel
hareketinde tam pipeline'ın yeniden çalıştırılması UI'ı kilitler ('kasma'); ısınmış
`camera_settings_panel._flush_pending_changes` ile aynı debounce deseni."""
_UNDO_STACK_LIMIT = 50
"""Geri al yığınının üst sınırı — sınırsız büyümesin diye en eski adım aşılınca atılır."""


class _BatchWorker(QThread):
    """Toplu işlemi (bkz. `core/batch.py::run_batch`) UI thread'inin DIŞINDA çalıştırır —
    kod tabanında İLK arka-plan thread kullanımı. `graph`, çağıran tarafından (bkz.
    `MainWindow._on_run_batch`) `copy.deepcopy` ile İZOLE edilmiş olmalı: `run_batch` bu
    nesneyi mutasyona uğratır (`params["path"]` günceller), canlı `self.graph`'ı DOĞRUDAN
    paylaşmak kamera tick'i/parametre düzenlemesiyle aynı anda çalışırsa veri yarışına yol
    açardı — deepcopy ile hiçbir mutable state UI thread'iyle paylaşılmaz, bu yüzden arayüzü
    düzenlemeyi engellemeye (disable) gerek YOKTUR."""

    progress = Signal(int, int)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(self, graph, node_id: str, input_folder: str, output_csv: str, registry) -> None:
        super().__init__()
        self._graph = graph
        self._node_id = node_id
        self._input_folder = input_folder
        self._output_csv = output_csv
        self._registry = registry
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:  # noqa: N802 - Qt override
        try:
            rows = run_batch(
                self._graph,
                self._node_id,
                self._input_folder,
                self._output_csv,
                registry=self._registry,
                progress_callback=lambda done, total: self.progress.emit(done, total),
                should_cancel=lambda: self._cancel_requested,
            )
        except (ValueError, OSError) as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(rows)


class _LiveTickWorker(QThread):
    """Canlı kamera tick'inin AĞIR kısmını (`_build_preview_frame` -- pipeline `evaluate()`,
    overlay bileşimi, ROI trial-run) UI thread'inin DIŞINDA çalıştırır. Gerçek kullanıcı
    raporu: "şekil bul özelliği çok kasıyor" -- kök neden, `geom.shape_match` gibi ağır bir
    adım ne kadar sürerse tüm arayüzün (pencere sürükleme, diğer panellere tıklama dahil) o
    kadar donmasıydı, çünkü `_on_camera_tick` her 100ms'de bir TAMAMEN senkron/UI-thread'de
    çalışıyordu.

    `_BatchWorker` ile AYNI izolasyon ilkesi: `graph` çağıran tarafından (bkz.
    `MainWindow._on_camera_tick`) `copy.deepcopy` ile İZOLE edilmiş olmalı -- worker kendi
    TAZE `ExecutionEngine`'ini bu kopya üzerinde kurar, canlı `self.engine`/`self.graph`'a
    HİÇ dokunmaz. Bu güvenli: `inject_result` zaten HER tick'te downstream'in tamamını dirty
    işaretlediğinden (yeni kare = yeniden hesapla), tick'ler arası engine cache'i canlı kamera
    zincirinde ZATEN hiçbir fayda sağlamıyordu -- taze bir engine kurmak hiçbir şey
    KAYBETTİRMEZ. `self.engine`/`self.graph`'ın kendisi tek seferlik/on-demand çağrılar
    (Görüntüyü Dışa Aktar, Filtrelenmiş Kareyi Yakala, Özel Filtre diyalogu vb.) için AYNEN
    senkron kalır, bu worker'dan tamamen bağımsızdır.

    `generation`, dispatch anında `MainWindow._live_tick_generation`'dan alınan bir bayatlık
    (staleness) token'ıdır -- sonuç geldiğinde seçili adım/görünüm modu değişmişse (bkz.
    `MainWindow._on_live_tick_result`) sonuç sessizce atılır."""

    result_ready = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self,
        graph: Graph,
        pipeline_order: list[str],
        registry: Any,
        source_node_id: str,
        frame: np.ndarray,
        target_node_id: str | None,
        view_mode: str,
        generation: int,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._pipeline_order = pipeline_order
        self._registry = registry
        self._source_node_id = source_node_id
        self._frame = frame
        self._target_node_id = target_node_id
        self._view_mode = view_mode
        self._generation = generation

    def run(self) -> None:  # noqa: N802 - Qt override
        try:
            engine = ExecutionEngine(self._graph, registry=self._registry)
            engine.inject_result(self._source_node_id, {"image": self._frame})
            result = _build_preview_frame(
                engine=engine,
                registry=self._registry,
                graph=self._graph,
                pipeline_order=self._pipeline_order,
                node_id=self._target_node_id,
                view_mode=self._view_mode,
                camera_active=True,
                last_camera_frame=self._frame,
            )
        except Exception as exc:  # noqa: BLE001 - worker asla sessizce ölmemeli/UI'ı düşürmemeli
            self.failed.emit(self._generation, str(exc))
            return
        self.result_ready.emit(self._generation, result)


class MainWindow(QMainWindow):
    def __init__(
        self,
        registry: Any = None,
        parent=None,
        camera_settings_dir: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("EK")
        self.registry = registry or default_registry

        self.pipeline = LinearPipeline(self.registry)
        self.graph = self.pipeline.graph
        self.engine = ExecutionEngine(self.graph, registry=self.registry)
        self._selected_node_id: str | None = None
        self._camera_source: CameraSource | None = None
        self._camera_timer = QTimer(self)
        self._camera_timer.timeout.connect(self._on_camera_tick)
        self._camera_reconnect_factory: Callable[[], CameraSource] | None = None
        """Son açılan USB/GigE kameranın (aynı tür+index ile) sıfırdan nasıl açılacağını
        bilen sıfır-argümanlı bir fabrika — bağlantı koptuğunda `_attempt_camera_reconnect`
        bunu çağırıp `start_camera` ile yeniden kurar. Video dosyası akışında (kopma diye bir
        şey yok) ve elle "Kamerayı Durdur" sonrası `None`'dır."""
        self._camera_fail_streak = 0
        self._camera_reconnect_cooldown_ticks = 0
        self._live_worker: _LiveTickWorker | None = None
        self._live_worker_busy: bool = False
        self._live_tick_generation: int = 0
        """`_LiveTickWorker` dispatch edilirken artırılan bir bayatlık (staleness) token'ı --
        `_on_live_tick_result` sonuç geldiğinde bunu o anki değerle karşılaştırır, farklıysa
        (seçili adım/görünüm modu dispatch'ten SONRA değiştiyse) sonucu sessizce atar. Sadece
        `_on_camera_tick`'teki her yeni dispatch'te VE `_set_selected_node`/`_on_view_mode_
        changed`'de artırılır -- parametre düzenlemelerinde BİLİNÇLİ olarak artırılMAZ (en kötü
        ihtimalle bir sonraki tick'te düzelen, eski parametrelerle hesaplanmış TEK bir bayat
        kare görünür -- bu, zaten var olan ~100ms'lik doğal gecikmeden görsel olarak ayırt
        edilemez)."""
        self._batch_worker: _BatchWorker | None = None
        self._batch_progress_dialog: QProgressDialog | None = None
        self._undo_stack: list[dict[str, Any]] = []
        self._redo_stack: list[dict[str, Any]] = []
        self._pending_param_undo_snapshot: dict[str, Any] | None = None
        """`_on_params_changed`'in ilk çağrısında (bir "yazma turunun" başında) yakalanan,
        henüz geri-al yığınına PUSH edilmemiş anlık durum — art arda gelen tuş vuruşları TEK
        bir undo adımına birleşsin diye `_flush_pending_params` (mevcut debounce, bkz. aşağı)
        tetiklenene kadar bekletilir."""
        self._param_debounce_timer = QTimer(self)
        self._param_debounce_timer.setSingleShot(True)
        self._param_debounce_timer.timeout.connect(self._flush_pending_params)
        # `_on_right_tabs_changed`'in "bir sonraki olay turunda bölme boyutlarını geri uygula"
        # gecikmesi. Eskiden `QTimer.singleShot(0, self, lambda ...)` (bağlam nesnesi
        # aşırı yüklemesi) kullanılıyordu; PySide bu bağlantıyı kendi dahili "global
        # receiver" nesnesi üzerinden kurduğundan, bağlam (MainWindow) yok edildiğinde
        # BEKLEYEN atış her zaman güvenilir biçimde iptal EDİLMİYOR ve daha sonra ölü
        # pencerede slot aranıp "AttributeError: Slot 'MainWindow::' not found" üretiliyordu.
        # `self`'in ÇOCUĞU olan gerçek bir QTimer, pencere yok edilirken Qt tarafından
        # deterministik olarak yok edilir -- bekleyen atış kesin olarak düşer.
        self._splitter_restore_timer = QTimer(self)
        self._splitter_restore_timer.setSingleShot(True)
        self._splitter_restore_timer.timeout.connect(self._restore_splitter_sizes)
        self._pending_splitter_sizes: list[int] | None = None
        self._tracked_dialogs: list[Any] = []
        """`destroyed` sinyaline bağlanmış TÜM dialoglar (yalnızca o an izlenen tekil
        referanslar değil, `deleteLater()` ile yok edilmeyi BEKLEYEN eskiler de). Bkz.
        `_connect_destroyed`/`closeEvent`."""
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
        self._enum_gallery_tick_counter: int = 0
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
        self.capture_gallery_panel.open_requested.connect(self.open_image)
        self.results_panel = MeasurementsSummaryPanel()
        self.image_view = RoiCanvas()
        self.image_view.setAcceptDrops(True)
        self.status_label = QLabel("")
        # `camera_settings_panel._status_label` ile AYNI gerekçe/desen: bu etiket uzun
        # hata/durum metinleri gösteriyor (ör. "Hata: Node 'flat_field' ... çalıştırılırken
        # ...") ve sarmasız bir QLabel'ın minimumSizeHint'i TÜM metnin genişliğine eşit
        # oluyordu -- ölçüldü: tek bir hata mesajı ORTA panelin minimum genişliğini
        # 1010px'ten 1314px'e, pencerenin talebini de aynı oranda büyütüyordu (gerçek
        # kullanıcı raporu: "gereksiz sayfa büyümeleri"). `setWordWrap` TEK BAŞINA yetmez
        # (boşluksuz uzun tek bir "kelime" -- ör. bir dosya yolu -- bölünemez), bu yüzden
        # yatay `QSizePolicy.Ignored`: layout bu widget'ın genişlik talebini TAMAMEN yok
        # sayar, metin mevcut alana göre sarar ama panelin minimumunu ASLA büyütmez.
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.hover_info_label = QLabel("Fareyi bir nesnenin üzerine getirin.")
        self.hover_info_label.setWordWrap(True)
        self.hover_info_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.hover_info_label.setStyleSheet("color: #ccc; background-color: #2a2a2a; padding: 4px;")

        self.operator_library.operator_checked.connect(self.add_operator)
        self.operator_library.operator_unchecked.connect(self._on_operator_unchecked)
        self.operator_library.custom_filter_editor_requested.connect(self._on_open_custom_filter_editor)
        self.steps_panel.step_selected.connect(self._on_step_selected)
        self.steps_panel.order_changed.connect(self._on_pipeline_changed)
        self.steps_panel.reordered.connect(self._on_pipeline_reordered)
        self.image_view.hover_measurement_changed.connect(self._on_hover_measurement_changed)
        self.image_view.image_file_dropped.connect(self._on_image_file_dropped)
        self.param_form.params_changed.connect(self._on_params_changed)
        self.enum_gallery.choice_selected.connect(self._on_enum_choice_selected)
        self.image_view.roi_changed.connect(self._on_roi_canvas_changed)
        self.image_view.roi_circle_changed.connect(self._on_roi_circle_canvas_changed)
        self.image_view.rois_changed.connect(self._on_manual_rois_canvas_changed)

        self._build_layout()
        self._build_menu()
        self._build_toolbar()

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Dosya")
        file_menu.addAction("Görüntü Aç...", self._on_open_image)
        export_image_action = file_menu.addAction(
            "Görüntüyü Dışa Aktar (PNG/JPG)...", self._on_export_current_image
        )
        export_image_action.setToolTip(
            "Seçili adımın o anki çıktısını bir görüntü dosyasına kaydeder — ör. filtrelerle "
            "ayrıştırdığınız bir nesneyi Şekil Eşleştirme'de referans olarak kullanmak için. "
            "DİKKAT: reçete (.json) dosyası PİKSEL VERİSİ İÇERMEZ, sadece pipeline adımlarını "
            "saklar — bir reçeteyi başka bir yerde 'görüntü' olarak yükleyemezsiniz, önce "
            "burada gerçek bir resim dosyasına dışa aktarmanız gerekir."
        )
        file_menu.addSeparator()
        file_menu.addAction("Reçeteyi Kaydet...", self._on_save_recipe)
        file_menu.addAction("Reçete Yükle...", self._on_load_recipe)
        file_menu.addSeparator()
        file_menu.addAction("Toplu İşlem...", self._on_run_batch)

        edit_menu = self.menuBar().addMenu("Düzen")
        self._undo_action = edit_menu.addAction("Geri Al", self._on_undo)
        self._undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_action.setEnabled(False)
        self._redo_action = edit_menu.addAction("Yinele", self._on_redo)
        self._redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_action.setEnabled(False)

        camera_menu = self.menuBar().addMenu("Kamera")
        camera_menu.addAction("USB Kamera Aç...", self._on_open_usb_camera)
        camera_menu.addAction("GigE/Basler Kamera Aç...", self._on_open_gige_camera)
        camera_menu.addAction("Video Dosyası Aç...", self._on_open_video_file)
        camera_menu.addSeparator()
        self._capture_photo_action = camera_menu.addAction(
            "Fotoğraf Çek (Kare Yakala)", self._on_capture_camera_photo
        )
        self._capture_photo_action.setShortcut(QKeySequence("Ctrl+Shift+K"))
        self._capture_photo_action.setEnabled(False)
        self._capture_photo_action.setToolTip(
            "O anki ham kamera karesini 'Yakalananlar' galerisine kaydeder — kalibrasyon "
            "diyaloglarındaki 'Kare Yakala' ile AYNI galeriye yazar, oradan da sürükleyip "
            "kullanılabilir."
        )
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
        self._lens_calibration_action.setToolTip(
            "Checkerboard'ı farklı yükseklik/mesafelerde göstererek TEK geçişte hem lens "
            "kalibrasyonunu hem de (deneysel) otomatik yükseklik tahminini öğretir — hiçbir "
            "yükseklik/mesafe elle girilmez. Yeni çalışmalar için BURADAN başlayın. Kamera "
            "aktif olmasa da açılabilir — 'Kare Yakala' yerine sürükle-bırak/galeri karesi "
            "kullanabilirsiniz (gerçek kullanıcı isteği: 'yaptığımız kalibrasyonu canlı "
            "görüntü olmadan da kullanabilmeliyim')."
        )
        self._load_calibration_profile_action = camera_menu.addAction(
            "Kalibrasyon Profili Yükle...", self._on_load_calibration_profile
        )
        self._load_calibration_profile_action.setToolTip(
            "Daha önce kaydedilmiş bir kalibrasyon profilini (lens + yükseklik-ölçek) "
            "yeniden kalibre etmeden bu oturuma uygular — kamera aktif olmasa da kullanılabilir."
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
        adjust_height_delta_action = tools_menu.addAction(
            "Bant/Ürün Yüksekliği Değişti...", self._on_adjust_height_delta
        )
        adjust_height_delta_action.setToolTip(
            "Mevcut kalibrasyonu SIFIRDAN tekrarlamadan, bant/ürün yüzeyinin kameraya ne kadar "
            "yaklaştığını/uzaklaştığını (mm) girerek mm/px'i anında günceller."
        )
        self._height_scale_action = tools_menu.addAction(
            "Yükseklik Kalibrasyonu (Öğretme, elle)...", self._on_open_height_scale_calibration
        )
        self._height_scale_action.setToolTip(
            "ESKİ/manuel yöntem: 2 noktaya tıklayarak ya da tek tek checkerboard karesi "
            "yakalayarak, her nokta için yükseklik ELLE girilir. Checkerboard'ı farklı "
            "yüksekliklerde göstererek hiçbir şey elle girmeden kalibre etmek için bunun "
            "yerine Kamera > Lens Kalibrasyonu'nu kullanın. Kamera aktif olmasa da açılabilir "
            "— 'Kare Yakala' yerine sürükle-bırak/galeri karesi kullanabilirsiniz."
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
        toolbar.addSeparator()
        toolbar.addAction(self._undo_action)
        toolbar.addAction(self._redo_action)

    def _build_layout(self) -> None:
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Operatör Kütüphanesi"))
        left_layout.addWidget(self.operator_library, stretch=2)
        left_layout.addWidget(QLabel("Pipeline Adımları"))
        left_layout.addWidget(self.steps_panel, stretch=1)
        step_buttons_layout = QHBoxLayout()
        move_step_up_button = QPushButton("▲ Yukarı")
        move_step_up_button.setToolTip("Seçili adımı bir yukarı taşır.")
        move_step_up_button.clicked.connect(self._on_move_step_up_clicked)
        move_step_down_button = QPushButton("▼ Aşağı")
        move_step_down_button.setToolTip("Seçili adımı bir aşağı taşır.")
        move_step_down_button.clicked.connect(self._on_move_step_down_clicked)
        duplicate_step_button = QPushButton("⎘ Kopyala")
        duplicate_step_button.setToolTip("Seçili adımı aynı parametrelerle hemen altına kopyalar.")
        duplicate_step_button.clicked.connect(self._on_duplicate_step_clicked)
        step_buttons_layout.addWidget(move_step_up_button)
        step_buttons_layout.addWidget(move_step_down_button)
        step_buttons_layout.addWidget(duplicate_step_button)
        left_layout.addLayout(step_buttons_layout)
        left_layout.addWidget(QLabel("Nesne Bilgisi (üzerine gelin):"))
        left_layout.addWidget(self.hover_info_label)

        # Kamera hızlı-erişim çubuğu: "Kamera" menüsündeki en sık kullanılan 3 aksiyonun
        # (aç/fotoğraf çek/durdur) görüntü panelinin HEMEN ÜSTÜNDE, menüye gitmeden tek
        # tıkla erişilebilir kopyası — `imgflow: speed over guidance` ilkesiyle tutarlı
        # (bkz. `[[imgflow-speed-over-guidance]]`). Menüdeki asıl aksiyonlar (`_on_open_*`,
        # `_on_capture_camera_photo`, `_on_stop_camera`) DEĞİŞMEDEN aynen çağrılır — bu sadece
        # ikinci bir giriş noktası, ayrı bir mantık DEĞİL. `_capture_photo_button`'ın etkin/
        # devre dışı durumu `start_camera`/`stop_camera`'da `_capture_photo_action` ile TAM
        # eşzamanlı tutulur (bkz. o iki metod).
        open_camera_button = QPushButton("Kamera Aç ▾")
        open_camera_button.setToolTip("USB, GigE/Basler kamera ya da video dosyası açar.")
        open_camera_menu = QMenu(open_camera_button)
        open_camera_menu.addAction("USB Kamera...", self._on_open_usb_camera)
        open_camera_menu.addAction("GigE/Basler Kamera...", self._on_open_gige_camera)
        open_camera_menu.addAction("Video Dosyası...", self._on_open_video_file)
        open_camera_button.setMenu(open_camera_menu)

        self._capture_photo_button = QPushButton("📷 Fotoğraf Çek")
        self._capture_photo_button.setToolTip(
            "O anki ham kamera karesini 'Yakalananlar' galerisine kaydeder (Kamera menüsündeki "
            "'Fotoğraf Çek' ile aynı, kısayol Ctrl+Shift+K)."
        )
        self._capture_photo_button.setEnabled(False)
        self._capture_photo_button.clicked.connect(self._on_capture_camera_photo)

        # `_capture_photo_button`'ın AKSİNE kameraya bağlı değil: ham kare yerine seçili
        # pipeline adımının FİLTRELENMİŞ çıktısını (`_current_preview_image()`) yakalar --
        # gerçek kullanıcı isteği "filtrelediğim fotoğrafı da yakalayıp sağ taraftaki panele
        # atmak istiyorum". Durağan (io.image_source ile yüklenmiş) bir görüntüde de çalışır,
        # bu yüzden kamera aktif olmasa da her zaman etkin.
        self._capture_filtered_button = QPushButton("🖼️ Filtrelenmiş Kareyi Yakala")
        self._capture_filtered_button.setToolTip(
            "Seçili pipeline adımının o anki FİLTRELENMİŞ çıktısını 'Yakalananlar' galerisine "
            "kaydeder (ham kamera karesi değil, uygulanan filtrelerden sonraki görüntü)."
        )
        self._capture_filtered_button.clicked.connect(self._on_capture_filtered_frame)

        stop_camera_button = QPushButton("Kamerayı Durdur")
        stop_camera_button.clicked.connect(self._on_stop_camera)

        camera_bar = QHBoxLayout()
        camera_bar.addWidget(QLabel("Kamera:"))
        camera_bar.addWidget(open_camera_button)
        camera_bar.addWidget(self._capture_photo_button)
        camera_bar.addWidget(self._capture_filtered_button)
        camera_bar.addWidget(stop_camera_button)
        camera_bar.addStretch(1)

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
            "(ölçüm varsa üzerine çizilir). İkisi Bir Arada: ikisi yan yana. "
            "ROI Bağlamda: filtrelenmiş sonuç, ham karenin İÇİNDE kendi yerine yerleştirilir — "
            "işlenen bölge ile işlenmeyen alanı aynı karede birlikte görürsünüz."
        )
        for label, mode in [
            ("Filtrelenmiş", "filtered"),
            ("Normal", "normal"),
            ("İkisi Bir Arada", "both"),
            ("ROI Bağlamda", "in_context"),
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
        center_layout.addLayout(camera_bar)
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

        # `camera_settings_panel`'in `QToolBox`'ında ve `ShapeMatchingDialog`'un kontrol
        # panelinde olduğu GİBİ: parametre sekmesinin içeriği (operatöre göre değişen
        # sayıda alan + seçenek önizleme galerisi) sekmeye doğrudan konulursa, kendi
        # sizeHint'i sekme -> `central_splitter` -> ana pencere zincirine YUKARI doğru
        # taşınıp pencereyi büyütüyor/panoları yeniden dağıtıyordu. Kaydırma alanı bu
        # talebi ÖZÜMSER: sığmayan içerik panel İÇİNDE kaydırılır, pencere boyutu
        # operatör seçimine göre DEĞİŞMEZ.
        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_tab)

        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(params_scroll, "Parametreler")
        self.right_tabs.addTab(self.camera_settings_panel, "Kamera Ayarları")
        self.right_tabs.addTab(self.results_panel, "Sonuçlar")
        self.right_tabs.currentChanged.connect(self._on_right_tabs_changed)

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
        self._live_tick_generation += 1
        self._refresh_preview()

    def _on_hover_measurement_changed(self, measurement: dict[str, Any] | None) -> None:
        """Sol paneldeki `hover_info_label`'ı, fare `image_view` üzerinde bir ölçüm kutusunun
        üstündeyken günceller (bkz. `RoiCanvas.hover_measurement_changed`). Hem
        `analysis.region_props` (boyut/alan/açı/tolerans) hem `geom.shape_match` (model/skor/
        açı) hem `ml.onnx_detect` (model/sınıf/güven/tolerans) tarzı ölçüm sözlüklerini --
        alan adlarına bakarak -- ele alır."""
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
            if "scale_percent" in measurement:
                lines.append(f"Ölçek: %{measurement['scale_percent']:.0f}")
            if "width_mm" in measurement and "height_mm" in measurement:
                # Kalibrasyon aktifken px değeri de yanına yazılır -- gerçek kullanıcı isteği:
                # "kalibrasyon seçili olduğu her senaryoda pixelin yanında mm de yazsın veya cm".
                lines.append(
                    f"Boyut: {measurement['width_px']:.0f} x {measurement['height_px']:.0f} px "
                    f"({measurement['width_mm']:.1f} x {measurement['height_mm']:.1f} mm)"
                )
            elif "width_px" in measurement and "height_px" in measurement:
                lines.append(f"Boyut: {measurement['width_px']:.0f} x {measurement['height_px']:.0f} px")
            if "displacement_x_cm" in measurement:
                lines.append(
                    f"Öteleme: x={measurement['displacement_x_cm']:+.2f} "
                    f"y={measurement['displacement_y_cm']:+.2f} cm"
                )
            elif "displacement_x_px" in measurement:
                lines.append(
                    f"Öteleme: x={measurement['displacement_x_px']:+.1f} "
                    f"y={measurement['displacement_y_px']:+.1f} px"
                )
            if "class_name" in measurement:
                lines.append(f"Sınıf: {measurement['class_name']}")
            if "confidence" in measurement:
                lines.append(f"Güven: %{measurement['confidence'] * 100:.0f}")
            if "area_percent" in measurement:
                lines.append(f"Alan: %{measurement['area_percent']:.1f} ({measurement.get('area_px', 0)}px²)")
            if "tolerance_ok" in measurement:
                lines.append("Durum: OK" if measurement["tolerance_ok"] else "Durum: NG")
            self.hover_info_label.setText("\n".join(lines))
            return

        # Gerçek kullanıcı isteği: "kalibrasyon varsa pixelin yanında cm veya mm de yazsın."
        # `geom.shape_match` dalı (yukarıda) bunu zaten yapıyordu ama `analysis.region_props`
        # dalı kalibrasyon aktifken px değerini TAMAMEN gizleyip sadece cm gösteriyordu --
        # aynı bilgi burada da px + cm birlikte yazılıyor (overlay metni `region_props.py`'de
        # zaten bu formatta).
        if "obb_mm_w" in measurement and "obb_mm_h" in measurement:
            lines.append(
                f"Boyut: {measurement['obb_w']:.0f} x {measurement['obb_h']:.0f} px "
                f"({measurement['obb_mm_w'] / 10:.2f} x {measurement['obb_mm_h'] / 10:.2f} cm)"
            )
        elif "obb_w" in measurement and "obb_h" in measurement:
            lines.append(f"Boyut: {measurement['obb_w']:.0f} x {measurement['obb_h']:.0f} px")

        if "area_mm2" in measurement:
            lines.append(
                f"Alan: {measurement['area']:.0f} px² ({measurement['area_mm2'] / 100:.2f} cm²)"
            )
        elif "area" in measurement:
            lines.append(f"Alan: {measurement['area']:.0f} px²")

        if "perimeter_mm" in measurement:
            lines.append(
                f"Çevre: {measurement['perimeter']:.0f} px ({measurement['perimeter_mm']:.1f} mm)"
            )
        elif "perimeter" in measurement:
            lines.append(f"Çevre: {measurement['perimeter']:.0f} px")

        if "obb_angle" in measurement:
            lines.append(f"Açı: {measurement['obb_angle']:.1f}°")

        if "tolerance_ok" in measurement:
            lines.append("Tolerans: OK" if measurement["tolerance_ok"] else "Tolerans: NG")

        self.hover_info_label.setText("\n".join(lines) if lines else "")

    # -- public API (dialogsuz, testlerden de çağrılabilir) -----------------

    def add_operator(self, op_id: str) -> str:
        self._capture_undo_checkpoint()
        op_cls = self.registry.get(op_id)
        node_id = self.pipeline.generate_node_id(op_id)
        self.pipeline.append(Node(node_id, op_id, params=defaults_for(op_cls.params)))
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self.operator_library.set_checked(op_id, True)
        self._select_step(node_id)
        return node_id

    def remove_operator(self, node_id: str) -> None:
        self._capture_undo_checkpoint()
        op_id = self.graph.nodes[node_id].op_id
        was_selected = self._selected_node_id == node_id
        removed_position = self.pipeline.order.index(node_id) if node_id in self.pipeline.order else None

        self.pipeline.remove(node_id)
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        # Sadece bu op_id'den BAŞKA örnek kalmadıysa checkbox'ı kapat — "adım kopyala" ile
        # aynı op_id'den birden fazla node oluşabiliyor artık; ikisinden birini silmek
        # diğeri hâlâ pipeline'dayken checkbox'ı yanlışlıkla kapatıp kafa karıştırmamalı.
        # Bugüne kadar 0-veya-1-örnek senaryosunda davranış AYNI kalır.
        if not any(n.op_id == op_id for n in self.graph.nodes.values()):
            self.operator_library.set_checked(op_id, False)

        if was_selected:
            self._set_selected_node(None)
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
        self._set_selected_node(None)
        self.description_label.setText("")
        self.param_form.set_params([], {})
        self.enum_gallery.clear()
        self.pipeline.load(new_graph)
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self._sync_library_checkboxes()
        # Yeni bir reçete = yeni bir geçmiş; önceki reçeteye ait geri al/yinele adımlarını
        # yeni pipeline'a uygulamak anlamsız/tehlikeli olurdu.
        self._pending_param_undo_snapshot = None
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_redo_actions()
        if new_graph.calibration_profile:
            self._load_calibration_profile(new_graph.calibration_profile)
            if new_graph.calibration_height_mm is not None:
                self._set_active_height_mm(new_graph.calibration_height_mm)
        # Reçete KENDİ kalibrasyon profilini taşımıyorsa (yukarıdaki dal çalışmadıysa) oturumda
        # aktif olan kalibrasyon yine de korunur -- reçetedeki boş `mm_per_px` alanı onu
        # sessizce silmesin (bkz. `_reapply_active_calibration`).
        self._reapply_active_calibration()
        self._refresh_preview()
        get_logger().info("Reçete yüklendi: %s", path)

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
        get_logger().info("Kalibrasyon profili uygulandı: %s", name)
        return True

    def _reapply_active_calibration(self) -> None:
        """Aktif kalibrasyonu (varsa) düğüm parametrelerine YENİDEN yazar.

        Gerçek kullanıcı raporu: "kalibrasyon ayarı kendi kendine kaybolabiliyor". Bir önceki
        tur `_push_mm_per_px`'in `ParamForm` önbelleğiyle senkron olmama sorununu çözmüştü;
        geriye İKİNCİ bir kayıp yolu kalmıştı: `mm_per_px` düğümün `params`'ında YAŞADIĞI için,
        düğüm parametrelerini TOPLUCA geri yükleyen her işlem (Geri Al/Yinele -- snapshot
        kalibrasyondan ÖNCE alınmışsa; ve reçete yükleme -- reçete kendi kalibrasyon profilini
        taşımıyorsa) alanı sessizce ESKİ/BOŞ değerine döndürüyordu. Oysa kalibrasyon KAYNAĞI
        (`_active_lens_profile`/`_plane_rectification`/`_reference_distance_mm`/
        `HeightScaleModel`) bu işlemlerden hiç etkilenmiyor -- yani veri duruyor, sadece
        düğüme yazılmış kopyası siliniyordu.

        `_compute_auto_mm_per_px()` (kareye bağımlı netlik-tabanlı yol henüz bir kare
        görmediyse `None` dönebilir) başarısız olursa en son hesaplanan değere
        (`_last_auto_mm_per_px`) düşülür. Hiçbir OTOMATİK kalibrasyon kaynağı yoksa hiçbir şey
        yapılmaz -- kullanıcının ELLE girdiği bir `mm_per_px`'in geri alınması meşru bir
        Geri Al'dır, ona dokunulmaz."""
        mm_per_px = self._compute_auto_mm_per_px()
        if mm_per_px is None:
            mm_per_px = self._last_auto_mm_per_px
        if mm_per_px is None or mm_per_px <= 0:
            return
        self._push_mm_per_px(mm_per_px, refresh=False)

    def _push_mm_per_px(self, mm_per_px: float, refresh: bool = True) -> None:
        """`refresh=False`, `_on_camera_tick` gibi zaten kendi sonunda bir kez
        `_refresh_preview()` çağıracak bir çağrıdan geldiğinde kullanılır — aksi halde her
        tick'te pipeline'ı GEREKSİZ YERE İKİ KEZ çalıştırıp (bir kez burada, bir kez tick'in
        kendi sonunda) FPS'i belirgin şekilde düşürür (gerçek bir kullanıcı raporuyla
        doğrulandı: otomatik yükseklik tahmini eklendikten sonra kamera akışı yavaşladı).

        `_MM_PER_PX_OP_IDS`'teki HER op_id güncellenir -- `geom.shape_match` da (gerçek
        kullanıcı isteği: "geometrik eşlemede scale boyutu ... da yazmalı", mevcut kalibrasyon
        kullanılarak gerçek boyut/öteleme mesafesi mm cinsinden gösterilir) `analysis.
        region_props` ile AYNI otomatik doldurma akışından faydalanır.

        **Kritik:** `self.graph.nodes[node_id].params` burada `ParamForm`'un HABERİ OLMADAN
        doğrudan güncelleniyor -- güncellenen node o an seçili/panelde gösteriliyorsa
        `ParamForm._values` (formun kendi önbelleği) bunu YANSITMAZ, ESKİ `mm_per_px`'te
        kalır. Kullanıcı sonra o node'da BAŞKA bir parametreyi değiştirirse (ör. 'Min. Alan'),
        `ParamForm._on_change` formun TÜM `_values`'ini (ESKİ mm_per_px dahil)
        `params_changed` ile yayınlar, `_on_params_changed` bunu `node.params`'ın ÜZERİNE
        KOŞULSUZ yazar (bkz. o metot) -- otomatik hesaplanan kalibrasyon SESSİZCE kaybolur,
        kullanıcı kalibrasyona hiç dokunmamış olsa bile (gerçek kullanıcı raporu: "kalibrasyon
        ayarı kendi kendine kaybolabiliyor"). Düzeltme: node o an panelde gösteriliyorsa
        `ParamForm.set_value()` (donanım-clamp senkronizasyonuyla AYNI mekanizma,
        `camera_settings_panel.py`'de zaten kullanılıyor) ile formun önbelleği de HEMEN
        güncellenir -- sinyal TEKRAR yaymaz, gereksiz bir döngü/yeniden hesaplama tetiklemez."""
        for node_id, node in self.graph.nodes.items():
            if node.op_id in _MM_PER_PX_OP_IDS:
                node.params = {**node.params, "mm_per_px": mm_per_px}
                self.engine.mark_dirty(node_id)
                if node_id == self._selected_node_id:
                    self.param_form.set_value("mm_per_px", mm_per_px)
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
        """`_MM_PER_PX_OP_IDS`'teki HER op_id kontrol edilir -- aksi halde YENİ eklenen bir
        `geom.shape_match` düğümü (varsayılan `mm_per_px=0.0` ile), otomatik hesaplanan değer
        değişmediği için bu "neredeyse değişmedi, atla" kısayoluna takılıp asla ilk değerini
        alamayabilirdi (CLAUDE.md'de `analysis.region_props` için ÖNCEDEN düzeltilmiş AYNI
        bug sınıfı: "kalibrasyon profilini yükledim ama hâlâ px yazıyor")."""
        for node in self.graph.nodes.values():
            if node.op_id not in _MM_PER_PX_OP_IDS:
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
        self._capture_photo_action.setEnabled(True)
        self._capture_photo_button.setEnabled(True)
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
        self._capture_photo_action.setEnabled(False)
        self._capture_photo_button.setEnabled(False)
        self.camera_settings_panel.set_controller(None)
        if self._lens_calibration_dialog is not None:
            self._lens_calibration_dialog.close()
            self._lens_calibration_dialog = None
        if self._height_scale_dialog is not None:
            self._height_scale_dialog.close()
            self._height_scale_dialog = None

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self.stop_camera()
        # `_param_debounce_timer` (bkz. `_on_params_changed`, 120ms) pencere kapatılırken HÂLÂ
        # bekliyor olabilir -- bir parametre değişikliğinden hemen sonra kapatmak yeterli.
        # Zamanlayıcı `self`'in çocuğu olsa da, ateşlenmesi ile Python nesnesinin yok edilmesi
        # ÇAKIŞIRSA PySide bağlı slotu (bir Python bound method'u) artık çözemeyip
        # "AttributeError: Slot 'MainWindow::' not found" üretiyordu -- test paketinde
        # `-p no:randomly` ile HER ÇALIŞTIRMADA aynı iki testte (ikisi de kapanmadan hemen
        # önce `params_changed` yayınlıyor) tekrar üretilebiliyordu. Açıkça durdurmak, bekleyen
        # tek seferlik atışı tamamen ortadan kaldırır.
        self._param_debounce_timer.stop()
        self._splitter_restore_timer.stop()
        # Tüm tekil dialoglar `destroyed.connect(...)` ile, `self`'i (MainWindow) yakalayan bir
        # Python callable'ına bağlı (izleyen referansı temizlemek için). `deleteLater()` ile
        # yok edilmeyi BEKLEYEN bir dialog varsa (ör. Ölçüm Aracı yeniden açıldığında eskisi),
        # onun `destroyed`'ı ana pencere çoktan yok edildikten SONRA ateşlenebiliyor -- PySide
        # o anda callable'ın alıcısını (ölü MainWindow) artık çözemeyip "AttributeError: Slot
        # 'MainWindow::' not found" üretiyordu. Bağlantıyı burada kesmek bunu tamamen ortadan
        # kaldırır: pencere kapanırken bu geri çağrıların (sadece kendi alanlarını `None`
        # yapıyorlar) yapacak hiçbir anlamlı işi zaten kalmamıştır.
        # İzlenen tekil referanslar YETMEZ: `deleteLater()` ile yok edilmeyi bekleyen ESKİ
        # örnekler (ör. Ölçüm Aracı her açılışta yeniden kurulur) artık hiçbir alanda
        # tutulmuyor ama `destroyed` bağlantıları hâlâ canlı. Bu yüzden `_connect_destroyed`
        # ile kaydedilen TÜM dialoglar geziliyor.
        for dialog in self._tracked_dialogs:
            try:
                dialog.destroyed.disconnect()
            except (RuntimeError, TypeError):
                # C++ nesnesi çoktan yok edilmiş ya da bağlantı zaten kesilmiş -- zararsız.
                pass
        self._tracked_dialogs.clear()
        # `_LiveTickWorker`/`_BatchWorker` (bkz. yukarıdaki sınıflar) arka planda çalışırken
        # pencere kapatılırsa (ör. testlerin `qtbot.addWidget` ile tetiklediği otomatik
        # teardown -- `_on_camera_tick` çağrılıp worker'ın bitmesi HİÇ beklenmeden pencere
        # hemen kapatılabiliyor) thread hâlâ CANLI iken bu nesne yok edilebiliyordu. Thread
        # bittiğinde `result_ready`/`finished` sinyalleri (kuyruklu, çapraz-thread bağlantı)
        # artık yok edilmiş/yok edilmekte olan `self`'e ulaşmaya çalışıyor -- bu, thread'in
        # yok etmeyle TAM OLARAK aynı anda çalışmasına bağlı bir yarış durumu (Qt'nin
        # otomatik bağlantı-kesme mekanizması bunu HER ZAMAN güvenli engelleyemiyor). `wait()`
        # thread tamamen bitene kadar (bu tek seferlik/sınırlı bir hesaplama, sonsuz döngü
        # DEĞİL) engeller -- normal kullanımda bu neredeyse anlık sürer, sadece pencere
        # ağır bir hesaplama ortasında kapatılırsa fark edilir bir gecikme olur.
        #
        # `wait()` TEK BAŞINA yetmiyordu: worker `result_ready`/`finished`'ı emit ettikten
        # SONRA (thread biter, `wait()` anında döner) bu sinyallerin KUYRUKLU teslimatı hâlâ
        # ana thread'in olay kuyruğunda bekliyor olabilir. PySide bir Python callable'ına
        # yapılan bağlantıda alıcı olarak MainWindow'u DEĞİL kendi dahili "global receiver"
        # nesnesini kullandığından, MainWindow yok edilince Qt'nin "alıcı yok edildi, bekleyen
        # olayları sil" mekanizması bu çağrıyı TEMİZLEMİYOR -- olay daha sonra (bambaşka bir
        # zamanda) teslim edilip ölü pencerede slot aranıyor ve "AttributeError: Slot
        # 'MainWindow::' not found" üretiyordu (test paketinde ~3 çalıştırmada 1 rastgele bir
        # testin teardown'ında görülen kalıntı hata buydu). Bağlantıları `wait()`'ten ÖNCE
        # açıkça kesmek bekleyen teslimatı da geçersiz kılar.
        for worker in (self._live_worker, self._batch_worker):
            if worker is None:
                continue
            for signal in (
                getattr(worker, "result_ready", None),
                getattr(worker, "failed", None),
                getattr(worker, "progress", None),
                getattr(worker, "finished_ok", None),
                worker.finished,
            ):
                if signal is None:
                    continue
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    # Zaten bağlantısı yoksa Qt/PySide hata veriyor -- zararsız.
                    pass
            worker.wait()
        self._live_worker = None
        self._batch_worker = None
        self._live_worker_busy = False
        # NOT: burada `QApplication.processEvents()` ile kuyruğu boşaltmak DENENDİ ve
        # ÖLÇÜLDÜ -- bekleyen "Slot 'MainWindow::' not found" çağrılarını azaltmadı, hatta
        # tekrar-giriş nedeniyle biraz artırdı; bu yüzden eklenmedi.
        super().closeEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Gerçek kullanıcı raporu: "uygulama kendi kendine tam ekrandan çıkıyor ve ekran
        kayıyor, bazı panellere erişemiyorum, o yüzden tam ekrandan hiç çıkmasın." Pencere
        durumu Qt/işletim sistemi tarafında (ör. başlık çubuğunu çift tıklama, kenara
        sürükleyip bırakma/"Aero Snap", ya da dock panel yeniden düzenlemesinin dolaylı bir
        yan etkisi) BEKLENMEDİK şekilde "geri yüklenmiş" (`Qt.WindowState.WindowNoState` --
        ne büyütülmüş ne küçültülmüş ne tam ekran) hale dönebiliyor; bu durumda dock
        panellerin konumu/boyutu da bu küçük pencereye göre yeniden hesaplanıp bazı
        panellerin ekran dışına/erişilemez hale gelmesine yol açabiliyor. Pencere
        KÜÇÜLTÜLMÜŞSE (görev çubuğuna atılmışsa) dokunulmaz -- SADECE düz "geri yüklenmiş"
        duruma dönüldüğünde büyütülmüş hale HEMEN geri döndürülür. `showMaximized()`'ın
        kendisi de bir `WindowStateChange` üretir, ama o zaman `windowState()` artık
        `WindowNoState` OLMADIĞINDAN aşağıdaki koşul bir sonraki olayda kendiliğinden
        durur -- sonsuz döngü YOK."""
        super().changeEvent(event)
        if (
            event.type() == QEvent.Type.WindowStateChange
            and self.isVisible()
            and self.windowState() == Qt.WindowState.WindowNoState
        ):
            self.showMaximized()

    def _on_right_tabs_changed(self, _index: int) -> None:
        """Gerçek kullanıcı raporu: "roi seçtikten sonra başka bir sekmeye geçince
        zoomluyor" -- `right_tabs`'ın (Parametreler/Kamera Ayarları/Sonuçlar) aktif sekmesi
        değişince, o sekmenin içeriği FARKLI bir `sizeHint()` doğurabilir (ör. Kamera
        Ayarları'ndaki `QToolBox` sayfaları boşken/doluyken farklı boy ister) -- Qt bunu
        `central_splitter`'a bir yeniden-düzenleme isteği olarak iletip panolar arası alanı
        SESSİZCE yeniden dağıtabiliyor; bu da orta (görüntü) panelinin genişliğini
        değiştirip `ImageView._rescale()`'in "sığdır" ölçeğini kaydırıyor -- kullanıcıya
        istenmeyen bir "zoom" gibi görünüyor. Sekme değişiminden HEMEN ÖNCEKİ boyutları
        yakalayıp Qt'nin kendi otomatik yeniden-düzenlemesi bu olay turunda tamamlandıktan
        SONRA (`QTimer.singleShot(0, ...)`) tekrar uyguluyoruz -- görüntü paneli sekme
        geçişlerinden TAMAMEN bağımsız kalır, kullanıcı SADECE bölme sınırını elle
        sürükleyerek boyutları değiştirebilir."""
        self._pending_splitter_sizes = self.central_splitter.sizes()
        self._splitter_restore_timer.start(0)

    def _connect_destroyed(self, dialog: Any, handler) -> None:
        """Bir dialog'un `destroyed` sinyalini `self`'i (MainWindow) yakalayan bir geri
        çağrıya bağlar VE dialog'u `_tracked_dialogs`'a kaydeder.

        Kayıt şart: `destroyed`, C++ nesnesi yok edilirken ateşlenir ve alıcı MainWindow'dur.
        Bir dialog `deleteLater()` ile yok edilmeyi beklerken (ya da GC'nin onu toplaması
        gecikirken) ana pencere ÖNCE yok edilirse, sinyal daha sonra ÖLÜ bir pencerede slot
        aramaya çalışıp "AttributeError: Slot 'MainWindow::' not found" üretiyor -- bu hata
        test paketinde, onu YARATAN testten bambaşka bir testte (o an hangi olay döngüsü
        dönüyorsa orada) rastgele patlıyordu. `closeEvent` bu listeyi gezip bağlantıları
        kesiyor; "artık izlenmeyen" (ör. Ölçüm Aracı'nın bir önceki örneği) dialoglar da
        böylece kapsanıyor -- SADECE tekil referanslara bakmak yetmiyordu."""
        dialog.destroyed.connect(handler)
        self._tracked_dialogs.append(dialog)

    def _restore_splitter_sizes(self) -> None:
        if self._pending_splitter_sizes is not None:
            self.central_splitter.setSizes(self._pending_splitter_sizes)
            self._pending_splitter_sizes = None

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

    # -- geri al / yinele -------------------------------------------------
    #
    # Kapsam BİLİNÇLİ olarak SADECE pipeline adımları (ekleme/silme/taşıma/kopyalama) ve
    # parametre değişiklikleridir — ROI çizimi ve kalibrasyon durumu (mm/px, lens profili)
    # KAPSAM DIŞI (kullanıcı onayı). Snapshot'lar `LinearPipeline.load()` (mevcut, tek
    # metotta node+order birlikte değiştirme ilkeli) üzerinden geri yüklenir.

    def _snapshot_dict(self, order: list[str]) -> dict[str, Any]:
        return {
            "nodes": copy.deepcopy(self.graph.nodes),
            "order": list(order),
            "selected_node_id": self._selected_node_id,
        }

    def _push_undo_snapshot(self, snapshot: dict[str, Any]) -> None:
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > _UNDO_STACK_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_actions()

    def _flush_pending_param_undo(self) -> None:
        if self._pending_param_undo_snapshot is not None:
            self._push_undo_snapshot(self._pending_param_undo_snapshot)
            self._pending_param_undo_snapshot = None

    def _capture_undo_checkpoint(self) -> None:
        """Yapısal bir pipeline değişikliğinden (adım ekleme/silme/taşıma/kopyalama) HEMEN
        ÖNCE çağrılır — mevcut (henüz değişmemiş) durumu geri-al yığınına iter. Önce bekleyen
        bir parametre-düzenleme snapshot'ı varsa (kronolojik sırayı korumak için) onu iter."""
        self._flush_pending_param_undo()
        self._push_undo_snapshot(self._snapshot_dict(self.pipeline.order))

    def _restore_pipeline_snapshot(self, snapshot: dict[str, Any]) -> None:
        temp_graph = Graph()
        temp_graph.nodes = copy.deepcopy(snapshot["nodes"])
        self.pipeline.load(temp_graph, order=list(snapshot["order"]))
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self._sync_library_checkboxes()
        selected = snapshot.get("selected_node_id")
        if selected is not None and selected in self.graph.nodes:
            self._set_selected_node(None)
            self._select_step(selected)
            self._on_step_selected(selected)
        else:
            self._set_selected_node(None)
            self.description_label.setText("")
            self.param_form.set_params([], {})
            self.enum_gallery.clear()
        # Snapshot kalibrasyondan ÖNCE alınmış olabilir -- düğümlerin `mm_per_px` alanı geri
        # yüklenirken sessizce sıfırlanmasın (bkz. `_reapply_active_calibration`).
        self._reapply_active_calibration()
        self._refresh_preview()

    def _update_undo_redo_actions(self) -> None:
        self._undo_action.setEnabled(bool(self._undo_stack))
        self._redo_action.setEnabled(bool(self._redo_stack))

    def _on_undo(self) -> None:
        self._flush_pending_param_undo()
        if not self._undo_stack:
            return
        current = self._snapshot_dict(self.pipeline.order)
        snapshot = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore_pipeline_snapshot(snapshot)
        self._update_undo_redo_actions()

    def _on_redo(self) -> None:
        if not self._redo_stack:
            return
        current = self._snapshot_dict(self.pipeline.order)
        snapshot = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._restore_pipeline_snapshot(snapshot)
        self._update_undo_redo_actions()

    def _on_pipeline_reordered(self, previous_order: list[str]) -> None:
        self._flush_pending_param_undo()
        self._push_undo_snapshot(self._snapshot_dict(previous_order))

    def _move_step(self, node_id: str, direction: int) -> None:
        order = self.pipeline.order
        if node_id not in order:
            return
        idx = order.index(node_id)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(order):
            return
        self._capture_undo_checkpoint()
        new_order = list(order)
        new_order[idx], new_order[new_idx] = new_order[new_idx], new_order[idx]
        self.pipeline.set_order(new_order)
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self._select_step(node_id)
        self._refresh_preview()

    def _on_move_step_up_clicked(self) -> None:
        node_id = self.steps_panel.selected_node_id()
        if node_id is not None:
            self._move_step(node_id, -1)

    def _on_move_step_down_clicked(self) -> None:
        node_id = self.steps_panel.selected_node_id()
        if node_id is not None:
            self._move_step(node_id, 1)

    def _on_duplicate_step_clicked(self) -> None:
        node_id = self.steps_panel.selected_node_id()
        if node_id is None:
            return
        self._capture_undo_checkpoint()
        node = self.graph.nodes[node_id]
        new_id = self.pipeline.generate_node_id(node.op_id)
        new_node = Node(new_id, node.op_id, params=copy.deepcopy(node.params))
        original_index = self.pipeline.order.index(node_id)
        self.pipeline.append(new_node)
        new_order = list(self.pipeline.order)
        new_order.remove(new_id)
        new_order.insert(original_index + 1, new_id)
        self.pipeline.set_order(new_order)
        self.engine.mark_all_dirty()
        self.steps_panel.refresh()
        self._select_step(new_id)
        self._on_step_selected(new_id)
        self._refresh_preview()

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
        # `_render_enum_choice`'ın kullandığı `trial_run`, hedef node'un ATA zincirinin
        # `self.engine`'in cache'inde HAZIR olmasını bekler (bkz. `ExecutionEngine.trial_run`
        # docstring'i). Canlı kamera tick'i artık AĞIR `evaluate()`'i `self.engine` üzerinde
        # DEĞİL, `_LiveTickWorker`'ın kendi TAZE/izole kopyası üzerinde çalıştırdığından
        # (bkz. `_on_camera_tick`), `self.engine`'in cache'i bu ata zinciri için artık
        # sürekli ısınmış OLMAYABİLİR -- burada AÇIKÇA bir `evaluate()` ile ısıtılır. Zaten
        # `_ENUM_GALLERY_TICK_STRIDE` ile throttle edilen tick-güdümlü çağrı noktası dışında
        # (adım seçimi, parametre debounce flush'ı, enum seçimi, ROI değişikliği) bu ZATEN
        # gerekli bir adımdı, sadece önceden `_refresh_preview()`'in AYNI tick'te önce
        # çalışması sayesinde ÖRTÜK olarak sağlanıyordu.
        self.engine.evaluate(node_id)
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
            source = UsbCameraSource(index)
        except RuntimeError as exc:
            show_error(self, "Kamera Açılamadı", exc)
            return
        self._camera_reconnect_factory = lambda: UsbCameraSource(index)
        self.start_camera(source)

    def _on_open_gige_camera(self) -> None:
        index, ok = QInputDialog.getInt(self, "GigE/Basler Kamera", "Kamera index'i:", 0, 0, 10)
        if not ok:
            return
        try:
            source = BaslerCameraSource(index)
        except RuntimeError as exc:
            show_error(self, "Kamera Açılamadı", exc)
            return
        self._camera_reconnect_factory = lambda: BaslerCameraSource(index)
        self.start_camera(source)

    def _on_open_camera_settings(self) -> None:
        self.right_tabs.setCurrentWidget(self.camera_settings_panel)

    def _camera_frame_provider(self) -> np.ndarray | None:
        """Kalibrasyon dialoglarına (Lens/Yükseklik) verilen `frame_provider` — kamera aktif
        değilken `None` döner. Gerçek kullanıcı isteği: "yaptığımız kalibrasyonu canlı
        görüntü olmadan da kullanabilmeliyim" -- bu dialoglar artık kamera olmadan da
        açılabiliyor (bkz. `_lens_calibration_action` vb.'nin artık HER ZAMAN etkin olması),
        bu yüzden 'Kare Yakala' butonu kamerasızken ÇÖKMEMELİ, sadece hiçbir şey yapmamalı --
        `LensCalibrationDialog._on_capture`/`HeightScaleCalibrationDialog` zaten `None`
        dönüşünü sessizce no-op olarak ele alıyor (canlı kameranın geçici bir tick'te kare
        üretememesiyle AYNI durum). Sürükle-bırak/galeri girdisi bu durumdan ETKİLENMEZ,
        tam çalışır."""
        if self._camera_source is None:
            return None
        return self._camera_source.read()

    def _on_open_lens_calibration(self) -> None:
        if self._lens_calibration_dialog is not None:
            self._lens_calibration_dialog.show()
            self._lens_calibration_dialog.raise_()
            self._lens_calibration_dialog.activateWindow()
            return
        dialog = LensCalibrationDialog(self._camera_frame_provider, parent=self)
        dialog.calibrated.connect(self._on_lens_calibrated)
        dialog.frame_captured.connect(self.capture_gallery_panel.refresh)
        self._connect_destroyed(dialog, lambda: setattr(self, "_lens_calibration_dialog", None))
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
        get_logger().info("Lens kalibrasyonu uygulandı.")

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
        if self._height_scale_dialog is not None:
            self._height_scale_dialog.show()
            self._height_scale_dialog.raise_()
            self._height_scale_dialog.activateWindow()
            return
        dialog = HeightScaleCalibrationDialog(
            self._camera_frame_provider,
            lens_profile_provider=lambda: self._active_lens_profile,
            open_lens_calibration=self._on_open_lens_calibration,
            parent=self,
        )
        dialog.model_updated.connect(self._on_height_scale_model_updated)
        dialog.frame_captured.connect(self.capture_gallery_panel.refresh)
        self._connect_destroyed(dialog, lambda: setattr(self, "_height_scale_dialog", None))
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
        dialog.frame_captured.connect(self.capture_gallery_panel.refresh)
        self._connect_destroyed(dialog, lambda: setattr(self, "_shape_matching_dialog", None))
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
        dialog.frame_captured.connect(self.capture_gallery_panel.refresh)
        self._connect_destroyed(dialog, lambda: setattr(self, "_flat_field_dialog", None))
        self._flat_field_dialog = dialog
        dialog.show()

    def _on_flat_field_references_changed(self, saved_name: str = "") -> None:
        """`_on_shape_models_changed` ile AYNI desen: bir referans kaydedilir/silinirse,
        `correction.flat_field` düğümü o an seçiliyse parametre panelindeki 'Aydınlatma
        Referansı' açılır listesi hemen günceli göstersin diye formu yeniden kurar.

        Gerçek kullanıcı raporu: referans kaydetmek `reference_name` parametresini OTOMATİK
        seçmiyordu -- kullanıcı referansı kaydettikten sonra filtreyi uyguluyor ama alan hâlâ
        boş kaldığı için "'reference_name' parametresi boş olamaz" hatası alıyordu. Artık bir
        KAYDETME'den geliyorsa (`saved_name` dolu) VE alan hâlâ BOŞSA yeni referans otomatik
        seçilir -- doluysa (kullanıcı başka bir referans üzerinde çalışıyorsa) sessizce
        DEĞİŞTİRİLMEZ, silmede (`saved_name=""`) hiç dokunulmaz."""
        node_id = self._selected_node_id
        if node_id is None or node_id not in self.graph.nodes:
            return
        node = self.graph.nodes[node_id]
        if node.op_id != "correction.flat_field":
            return
        if saved_name and not node.params.get("reference_name"):
            self._capture_undo_checkpoint()
            node.params["reference_name"] = saved_name
            self.engine.mark_dirty(node_id)
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)
        self._refresh_preview()

    def _on_open_onnx_model(self) -> None:
        if self._onnx_model_dialog is not None:
            self._onnx_model_dialog.show()
            self._onnx_model_dialog.raise_()
            self._onnx_model_dialog.activateWindow()
            return
        dialog = OnnxModelDialog(parent=self)
        dialog.models_changed.connect(self._on_onnx_models_changed)
        self._connect_destroyed(dialog, lambda: setattr(self, "_onnx_model_dialog", None))
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
        self._connect_destroyed(dialog, lambda: setattr(self, "_help_dialog", None))
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
        self._connect_destroyed(dialog, lambda: setattr(self, "_custom_filter_dialog", None))
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
            # `WA_DeleteOnClose` YOK (bkz. tekil-dialog yeniden kullanım deseni) — bu yüzden
            # `close()` C++ nesnesini YOK ETMEZ, sadece gizler. Bu dialog (aksine) her açılışta
            # YENİDEN kurulduğundan (taze önizleme görüntüsü için) eskisini burada açıkça
            # `deleteLater()` ETMEZSEK her "Ölçüm Aracı" açılışında bir önceki dialog (ve
            # içindeki tam çözünürlüklü görüntü) MainWindow'un çocuğu olarak sonsuza kadar
            # bellekte kalır — gerçek bir sızıntı, tekrarlı aç/kapa testiyle doğrulandı.
            self._measurement_tool_dialog.close()
            self._measurement_tool_dialog.deleteLater()
        image = self._current_preview_image()
        dialog = MeasurementToolDialog(image, self._active_mm_per_px(), parent=self)
        dialog.frame_captured.connect(self.capture_gallery_panel.refresh)

        def _clear_if_current(dialog: MeasurementToolDialog = dialog) -> None:
            # `deleteLater()`'ı yukarıda ÇAĞIRDIĞIMIZ eski dialog'un `destroyed` sinyali, YENİ
            # dialog zaten `self._measurement_tool_dialog`'a atandıktan SONRA (olay döngüsü
            # döndüğünde) gecikmeli ateşlenir — kimlik kontrolü OLMADAN kayıtsızca `None` atamak
            # o an takip edilen GÜNCEL dialog referansını sessizce siler (dialog ekranda hâlâ
            # açıkken bir sonraki açılış onu artık kapatamaz/takip edemez hâle gelirdi).
            if self._measurement_tool_dialog is dialog:
                self._measurement_tool_dialog = None

        self._connect_destroyed(dialog, _clear_if_current)
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

    def _on_adjust_height_delta(self) -> None:
        """Tam kalibrasyonu tekrarlamadan, bant/ürün yüzeyinin fiziksel olarak kameraya ne
        kadar yaklaştığını/uzaklaştığını (mm) girdirip aktif mm/px'i buna göre günceller.
        `_compute_auto_mm_per_px` ile AYNI öncelik sırasını (plane_rectification >
        reference_distance_mm > HeightScaleModel) izler; üçünün de İÇİNİ değiştirmez, sadece
        OKUYUP yeni bir değerle üzerine yazar — mevcut (çalışan) kalibrasyon akışlarına
        dokunulmaz. Homografiyi/pozu YENİDEN KURMAZ, sadece mesafeye bağlı mm/px skalerini
        günceller (`_plane_rectification` yolunda kullanıcının kabul ettiği bir yaklaşıklık —
        bkz. plan notu)."""
        delta, ok = QInputDialog.getDouble(
            self,
            "Bant/Ürün Yüksekliği Değişti",
            "Yüzey kameraya ne kadar yaklaştı (mm)?\nUzaklaştıysa negatif girin.",
            0.0,
            -100000.0,
            100000.0,
            2,
        )
        if not ok or delta == 0.0:
            return

        if self._plane_rectification is not None:
            if self._active_lens_profile is None:
                self.status_label.setText("Lens profili eksik, mm/px güncellenemedi.")
                return
            fx = float(self._active_lens_profile.camera_matrix[0, 0])
            if fx <= 0:
                self.status_label.setText("Geçersiz lens profili (fx<=0), mm/px güncellenemedi.")
                return
            old_mm_per_px = self._plane_rectification.mm_per_px
            old_distance = old_mm_per_px * fx
            new_distance = old_distance - delta
            if new_distance <= 0:
                self.status_label.setText(
                    f"Geçersiz değişim: yeni mesafe ({new_distance:.1f}mm) sıfır veya negatif olamaz."
                )
                return
            new_mm_per_px = new_distance / fx
            # Gerçek kullanıcı raporu: "bant yüksekliği değişti diyip ölçüm alıyorum fakat çok
            # yanlış ölçüyor" -- kök neden burada SADECE `mm_per_px` alanı değiştiriliyordu,
            # `homography`/`output_size` DOKUNULMADAN kalıyordu. Ama `rectify()`'ın kullandığı
            # `homography` matrisinin KENDİSİ, `compute_plane_rectification`'da `1/mm_per_px`
            # ölçeğini (bkz. `world_to_output`) zaten İÇİNE gömüyor -- yani `_on_camera_tick`
            # HER karede hâlâ ESKİ ölçekle rektifiye ediyordu, ama ölçüm operatörleri (ör.
            # `region_props`) piksel->mm dönüşümünde YENİ `mm_per_px`'i kullanıyordu. Bu
            # tutarsızlık, gerçek ölçümde tam olarak (eski_mm_per_px/yeni_mm_per_px) oranında
            # sistematik bir hataya yol açıyordu (ör. yükseklik %50 değiştiyse ölçümler 2x
            # yanlış çıkıyordu). Düzeltme: `world_to_output @ (eski world_to_output)^-1`
            # matematiksel olarak SADECE düzgün bir `(eski_mm_per_px/yeni_mm_per_px)` skaler
            # matrisine indirgeniyor (bkz. plan/analiz notu) -- `homography`'ye SOLDAN bu
            # skaler matrisi çarpmak, orijinal `rvec`/`tvec`/`camera_matrix` hiç saklanmasa
            # bile (bu dataclass sadece bileşik `homography`'yi tutuyor) matrisi doğru şekilde
            # yeni ölçeğe taşır. `output_size` de AYNI oranla ölçeklenir ki rektifiye görüntü
            # hâlâ kaynağın TAMAMINI kapsasın (kırpma/taşma olmasın).
            scale_ratio = old_mm_per_px / new_mm_per_px
            rescale = np.diag([scale_ratio, scale_ratio, 1.0])
            new_homography = rescale @ self._plane_rectification.homography
            old_w, old_h = self._plane_rectification.output_size
            new_output_size = (
                max(1, round(old_w * scale_ratio)),
                max(1, round(old_h * scale_ratio)),
            )
            self._plane_rectification = replace(
                self._plane_rectification,
                homography=new_homography,
                output_size=new_output_size,
                mm_per_px=new_mm_per_px,
            )
        elif self._reference_distance_mm is not None:
            new_distance = self._reference_distance_mm - delta
            if new_distance <= 0:
                self.status_label.setText(
                    f"Geçersiz değişim: yeni mesafe ({new_distance:.1f}mm) sıfır veya negatif olamaz."
                )
                return
            self._reference_distance_mm = new_distance
        elif self._height_scale_model is not None and self._active_height_mm is not None:
            new_height = self._active_height_mm + delta
            try:
                self._height_scale_model.predict_scale(new_height)
            except (RuntimeError, ValueError) as exc:
                self.status_label.setText(f"Geçersiz değişim: {exc}")
                return
            self._set_active_height_mm(new_height)
            mm_per_px = self._active_mm_per_px()
            if mm_per_px is not None:
                self.status_label.setText(f"Kalibrasyon güncellendi: yeni mm/px = {mm_per_px:.5f}")
                get_logger().info("Bant/ürün yüksekliği %.2f mm değişti, yeni mm/px=%.5f.", delta, mm_per_px)
            return
        else:
            self.status_label.setText(
                "Önce bir kalibrasyon kurulmalı (Lens Kalibrasyonu veya Aktif Yükseklik Ayarla)."
            )
            return

        self._last_auto_mm_per_px = None
        mm_per_px = self._compute_auto_mm_per_px()
        if mm_per_px is not None:
            self._last_auto_mm_per_px = mm_per_px
            self._push_mm_per_px(mm_per_px, refresh=False)
            self.status_label.setText(f"Kalibrasyon güncellendi: yeni mm/px = {mm_per_px:.5f}")
            get_logger().info("Bant/ürün yüksekliği %.2f mm değişti, yeni mm/px=%.5f.", delta, mm_per_px)
        self._refresh_preview()

    def _on_open_video_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Video Dosyası Aç", "", _VIDEO_FILE_FILTER)
        if not path:
            return
        try:
            source = VideoFileSource(path)
        except RuntimeError as exc:
            show_error(self, "Video Açılamadı", exc)
            return
        self._camera_reconnect_factory = None
        self.start_camera(source)

    def _on_capture_camera_photo(self) -> None:
        """Kalibrasyon diyaloglarındaki 'Kare Yakala' ile AYNI depoya (`capture_store`, dolayısıyla
        AYNI 'Yakalananlar' galerisine) o anki ham kamera karesini kaydeder — kullanıcı canlı
        akıştan kalibrasyon/analiz için ayrıca bir diyalog açmadan tek tıkla fotoğraf çekebilsin
        diye. `_last_camera_frame` HER tick'te (throttle'dan önce) güncellenir, bu yüzden
        piksel-tazeliği `_current_preview_image()`'daki (pipeline'dan geçmiş) görüntüyle AYNI
        kaynağı (ham kare) paylaşır — `LensCalibrationDialog._on_capture`'ın `frame_provider()`
        çağırdığı kareyle BİREBİR aynı anlam."""
        if self._camera_source is None:
            self.status_label.setText("Fotoğraf çekilemedi: aktif kamera yok.")
            return
        if self._last_camera_frame is None:
            self.status_label.setText("Fotoğraf çekilemedi: henüz kameradan kare alınmadı.")
            return
        capture_store.save_capture(self._last_camera_frame, source="live")
        self.capture_gallery_panel.refresh()
        self.status_label.setText("Fotoğraf 'Yakalananlar' galerisine kaydedildi.")
        get_logger().info("Kameradan fotoğraf yakalandı (Kamera > Fotoğraf Çek).")

    def _on_capture_filtered_frame(self) -> None:
        """`_on_capture_camera_photo`'nun aksine ham kamera karesini DEĞİL, seçili pipeline
        adımının o anki FİLTRELENMİŞ çıktısını ('Görüntüyü Dışa Aktar'ın kullandığı AYNI
        `_current_preview_image()` yardımcısı) 'Yakalananlar' galerisine kaydeder -- kamera
        aktif olmasa da (durağan/dosyadan yüklenmiş bir görüntüde de) çalışır, çünkü kaynak
        ham kare değil ENGINE'in o adım için ürettiği sonuçtur."""
        image = self._current_preview_image()
        if image is None:
            self.status_label.setText(
                "Filtrelenmiş kare yakalanamadı: önce bir pipeline adımı seçin."
            )
            return
        capture_store.save_capture(image, source="filtered")
        self.capture_gallery_panel.refresh()
        self.status_label.setText("Filtrelenmiş kare 'Yakalananlar' galerisine kaydedildi.")
        get_logger().info("Filtrelenmiş kare yakalandı (Filtrelenmiş Kareyi Yakala).")

    def _on_stop_camera(self) -> None:
        self.stop_camera()
        self._camera_reconnect_factory = None
        self._camera_fail_streak = 0
        self._camera_reconnect_cooldown_ticks = 0

    def _register_camera_failure(self, message: str) -> None:
        """Kare okunamadığında (istisna veya kalıcı `None`) çağrılır — SADECE durum
        etiketinde gösterir (bkz. `_on_camera_tick`'teki "diyalog fırtınası" notu, burada da
        AYNI kural geçerli). `_camera_reconnect_factory` varsa (bir USB/GigE kamera açıkken),
        art arda `_CAMERA_DISCONNECT_THRESHOLD_TICKS` başarısızlıktan sonra otomatik yeniden
        bağlanmayı dener; başarısız bir denemeden sonra `_CAMERA_RECONNECT_COOLDOWN_TICKS`
        kadar bekleyip tekrar dener — kablo tekrar takılana kadar SÜREKLİ, asla vazgeçmeden."""
        self.status_label.setText(message)
        if self._camera_reconnect_factory is None:
            return
        if self._camera_reconnect_cooldown_ticks > 0:
            self._camera_reconnect_cooldown_ticks -= 1
            return
        self._camera_fail_streak += 1
        if self._camera_fail_streak >= _CAMERA_DISCONNECT_THRESHOLD_TICKS:
            self._attempt_camera_reconnect()

    def _attempt_camera_reconnect(self) -> None:
        self._camera_fail_streak = 0
        self._camera_reconnect_cooldown_ticks = _CAMERA_RECONNECT_COOLDOWN_TICKS
        factory = self._camera_reconnect_factory
        if factory is None:
            return
        try:
            new_source = factory()
        except RuntimeError as exc:
            self.status_label.setText(f"Kamera yeniden bağlanamadı, tekrar denenecek: {exc}")
            get_logger().warning("Kamera yeniden bağlanma denemesi başarısız: %s", exc)
            return
        self.start_camera(new_source)
        self.status_label.setText("Kamera yeniden bağlandı.")
        get_logger().info("Kamera yeniden bağlandı.")

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
            self._register_camera_failure(f"Kare okunamadı: {exc}")
            return
        if frame is None:
            # Basler/GigE tetikleyici modda tetik gelmemesi NORMALDİR (bkz. `BaslerCameraSource
            # .read()` docstring'i) — bu, bağlantı kopması DEĞİLDİR, bu yüzden sadece o türde
            # bir kamera DEĞİLKEN başarısızlık sayacına dahil edilir (USB/video dosyası kaynağı
            # `None` döndüğünde bu her zaman gerçek bir okuma hatasıdır).
            if not isinstance(self._camera_source, BaslerCameraSource):
                self._register_camera_failure("Kare okunamadı (kamera yanıt vermiyor).")
            return
        self._camera_fail_streak = 0
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
        # `self.engine`'in cache'i (SADECE kaynak node'un cache girdisi -- ucuz bir dict yazımı,
        # `evaluate()`'in kendisi DEĞİL) her tick'te güncel tutulur -- `_current_preview_image()`
        # gibi tek seferlik/on-demand çağrılar (Görüntüyü Dışa Aktar, Filtrelenmiş Kareyi
        # Yakala, Özel Filtre diyalogu) hep GÜNCEL bir kareyi görsün diye.
        self.engine.inject_result(node_id, {"image": frame})

        # `_maybe_refresh_enum_gallery` (kendi 5-tick throttle'ı var) worker'ın meşgul olup
        # olmamasından BAĞIMSIZ her tick'te çağrılır -- `self.engine` üzerinde çalışır, worker'ın
        # ayrı/izole kopyasını hiç etkilemez/etkilenmez.
        self._maybe_refresh_enum_gallery()

        if self._live_worker_busy:
            # Önceki tick'in ağır hesaplaması (ör. şekil bulma) hâlâ arka planda sürüyor --
            # bu kare İŞLENMEK üzere kuyruğa ALINMAZ, doğrudan atlanır. Gerçek kullanıcı raporu:
            # "şekil bul özelliği çok kasıyor ve kamerayı siyah-beyaza çeviriyor [arayüzü
            # dondurup UI'ı bloklaması]" -- kuyruklama yerine atlama, sistemin HER ZAMAN en
            # hızlı ulaşabildiği kadar taze bir kareyi işlemesini sağlar (birikmiş, gecikmeli
            # kareler asla işlenmez).
            return
        self._live_worker_busy = True
        self._live_tick_generation += 1
        generation = self._live_tick_generation
        worker = _LiveTickWorker(
            copy.deepcopy(self.graph),
            list(self.pipeline.order),
            self.registry,
            node_id,
            frame,
            self._selected_node_id,
            self._view_mode,
            generation,
        )
        worker.result_ready.connect(self._on_live_tick_result)
        worker.failed.connect(self._on_live_tick_failed)
        worker.finished.connect(lambda: self._on_live_worker_finished(worker))
        self._live_worker = worker
        worker.start()

    def _on_live_tick_result(self, generation: int, result: PreviewFrameResult) -> None:
        """`_LiveTickWorker.result_ready` sinyalinin UI-thread slotu -- SADECE burada widget'lara
        dokunulur (worker'ın kendisi hiçbir Qt çağrısı yapmaz, bkz. `_LiveTickWorker.run`).
        `generation` dispatch anındakiyle uyuşmuyorsa (seçili adım/görünüm modu bu arada
        değişti, bkz. `_set_selected_node`/`_on_view_mode_changed`) sonuç BAYAT sayılıp
        sessizce atılır -- yanlış adım/moda ait bir görüntünün ekrana yanlışlıkla
        yansımasını önler."""
        if generation != self._live_tick_generation:
            return
        self._apply_preview_frame_result(result)

    def _on_live_tick_failed(self, generation: int, message: str) -> None:
        """`_LiveTickWorker.failed` -- `_build_preview_frame`'in kendisi zaten per-node
        hataları yutup `PreviewFrameResult.error`'da taşıdığından bu pratikte pek
        tetiklenmez; yine de worker'ın THREAD'i BEKLENMEDİK şekilde patlarsa (savunma amaçlı)
        sessizce yutulmak yerine durum etiketinde görünür olsun diye."""
        if generation != self._live_tick_generation:
            return
        self.status_label.setText(f"Canlı önizleme hesaplanamadı: {message}")

    def _on_live_worker_finished(self, worker: "_LiveTickWorker") -> None:
        """GERÇEK ÇÖKME KAYNAĞI (Windows olay günlüğüyle doğrulandı: `pythonw.exe` içinde
        `shiboken6.abi3.dll`/`pyside6.abi3.dll` modülünde `0xc0000005` erişim ihlali, kullanıcı
        raporu "kare yakalarken uygulama kendiliğinden kapandı" — Python tarafında HİÇ traceback
        yoktu, çünkü çökme C++ katmanındaydı).

        `worker.wait()` neden ŞART: `QThread::finished` thread'in OS düzeyinde HENÜZ ÇIKMADIĞI
        bir anda yayılır. `self._live_worker = None`, worker'a kalan TEK güçlü Python
        referansını bırakır — `finished`'e bağlanan lambda worker'ı yakalıyor olsa da o lambda
        worker'ın KENDİ sinyal bağlantısına ait olduğundan ortaya sadece bir REFERANS DÖNGÜSÜ
        çıkar. Döngüyü çöp toplayıcı RASTGELE bir anda toplar; o an thread henüz sonlanmamışsa
        shiboken alttaki C++ `QThread`'ini "çalışırken yok edilmiş" halde siler ve süreç
        sessizce ölür. Yakalama sırasında tetiklenmesi tesadüf değil: 'Filtrelenmiş Kareyi
        Yakala' ana thread'de senkron `engine.evaluate()` + galeri küçük-resim yenilemesi
        çalıştırır, bu da olay teslimini geciktirip GC zamanlamasını tam bu tehlikeli pencereye
        kaydırır (kamera tick'leri bu sırada worker kurup yıkmaya devam eder).

        `wait()` bırakmayı DETERMİNİSTİK bir join'in arkasına alır: referans düştükten sonra
        GC nesneyi ne zaman toplarsa toplasın thread KESİN olarak çıkmış olur. Slot ana
        thread'de (kuyruklu bağlantı) çalıştığından kilitlenme riski yok; thread çoktan
        bitmişse `wait()` anında döner."""
        worker.wait()
        if self._live_worker is worker:
            self._live_worker = None
        self._live_worker_busy = False

    def _maybe_refresh_enum_gallery(self) -> None:
        """`_ENUM_GALLERY_TICK_STRIDE` docstring'inde açıklanan gerekçeyle `_on_camera_tick`'ten
        SADECE bu tick-güdümlü sarmalayıcı üzerinden çağrılır; diğer (gerçek değişiklik
        olaylarına bağlı) çağrı noktaları doğrudan `_refresh_enum_gallery()`'i kullanmaya
        devam eder."""
        self._enum_gallery_tick_counter += 1
        if (self._enum_gallery_tick_counter - 1) % _ENUM_GALLERY_TICK_STRIDE != 0:
            return
        self._refresh_enum_gallery()

    def _on_operator_unchecked(self, op_id: str) -> None:
        node_id = next((nid for nid, n in self.graph.nodes.items() if n.op_id == op_id), None)
        if node_id is not None:
            self.remove_operator(node_id)

    def _on_export_current_image(self) -> None:
        """Seçili adımın o anki çıktısını gerçek bir görüntü dosyasına (.png/.jpg) yazar.

        Gerçek kullanıcı raporu: kameradan görüntü alıp filtrelerle bir nesneyi (ör. bir
        kulaklığı) ayrıştırdıktan sonra "Reçeteyi Kaydet..." ile kaydedip o `.json` dosyasını
        Şekil Eşleştirme'nin 'Referans Görüntüler Yükle...'/'İçe Aktar (JSON)' düğmelerine
        vermeye çalışmış — `KeyError: 'levels'` ile başarısız oldu, çünkü reçete
        (`io_utils/recipe.py::graph_to_dict`) SADECE pipeline node/edge/param grafiğini
        saklar, HİÇ piksel verisi içermez (`ShapeModel.from_dict`'in beklediği "levels"/
        "corners" anahtarlarıyla uzaktan yakından ilgisi yok). Bu aksiyon o boşluğu kapatır:
        `_current_preview_image()` (Ölçüm Aracı/Özel Filtre'nin de kullandığı AYNI yardımcı)
        ile seçili adımın çıktısını alıp diske yazar — kullanıcı bu dosyayı Şekil
        Eşleştirme'nin 'Referans Görüntüler Yükle...' düğmesine (JPG/PNG zaten kabul
        ediyordu) verebilir."""
        image = self._current_preview_image()
        if image is None:
            QMessageBox.information(
                self,
                "Görüntüyü Dışa Aktar",
                "Dışa aktarılacak bir önizleme yok — önce bir pipeline adımı seçin.",
            )
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Görüntüyü Dışa Aktar", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not path:
            return
        try:
            save_image(path, image)
        except OSError as exc:
            show_error(self, "Görüntü Dışa Aktarılamadı", exc, hint="Dosya yolunu ve yazma izinlerini kontrol edin.")
            return
        self.status_label.setText(f"Görüntü dışa aktarıldı: '{path}'")
        get_logger().info("Görüntü dışa aktarıldı: %s", path)

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
            total = len(iter_images(input_dir))
        except OSError as exc:
            show_error(self, "Toplu İşlem Hatası", exc)
            return

        # Worker'a canlı `self.graph`'ın DERİN KOPYASI verilir (bkz. `_BatchWorker` docstring'i)
        # — arka plan thread'i ile UI thread'i arasında hiçbir mutable state paylaşılmaz.
        worker = _BatchWorker(copy.deepcopy(self.graph), node_id, input_dir, output_csv, self.registry)
        progress_dialog = QProgressDialog("Toplu işlem sürüyor...", "İptal", 0, total, self)
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.canceled.connect(worker.request_cancel)
        worker.progress.connect(lambda done, _total: progress_dialog.setValue(done))
        worker.finished_ok.connect(self._on_batch_finished)
        worker.failed.connect(self._on_batch_failed)
        worker.finished.connect(progress_dialog.close)
        # `_on_open_measurement_tool`'daki AYNI sızıntı: bu dialog her toplu işlem
        # çalıştırmasında YENİDEN kurulur, `close()` C++ nesnesini yok etmediğinden
        # `deleteLater()` olmadan her çalıştırma bir hayalet `QProgressDialog` bırakır.
        worker.finished.connect(progress_dialog.deleteLater)
        self._batch_worker = worker
        self._batch_progress_dialog = progress_dialog
        progress_dialog.show()
        worker.start()

    def _on_batch_finished(self, rows: list[dict[str, Any]]) -> None:
        self._join_batch_worker()
        error_count = sum(1 for row in rows if "error" in row)
        QMessageBox.information(
            self, "Toplu İşlem Tamamlandı", f"{len(rows)} satır yazıldı ({error_count} hata)."
        )
        get_logger().info("Toplu işlem tamamlandı: %d satır, %d hata.", len(rows), error_count)
        self._batch_worker = None
        self._batch_progress_dialog = None

    def _join_batch_worker(self) -> None:
        """`_on_live_worker_finished`'deki AYNI çökme gerekçesi (bkz. oradaki uzun not):
        `_BatchWorker`'ın `finished_ok`/`failed` sinyalleri `run()`'ın İÇİNDEN yayılır, yani
        yayıldıkları anda thread KESİNLİKLE hâlâ çalışıyordur. `self._batch_worker = None`
        son güçlü referansı bırakıp C++ `QThread`'in çalışırken yok edilmesine yol
        açabiliyordu. (Toplu işlem dalında modal `QMessageBox` nested bir olay döngüsü
        çalıştırdığı için thread'e çıkma şansı veriyordu — ama bu ŞANS, garanti değil.)"""
        worker = self._batch_worker
        if worker is not None:
            worker.wait()

    def _on_batch_failed(self, message: str) -> None:
        self._join_batch_worker()
        # `_BatchWorker` sadece str yayınlıyor (Qt sinyali sınırı) — `show_error` bir
        # Exception bekliyor, `ValueError`/`OSError` de zaten `_ACTIONABLE_TYPES` içinde
        # olduğu için RuntimeError'a sarmak aynı "girdi/ayar sorunu" çerçevesini korur.
        show_error(self, "Toplu İşlem Hatası", RuntimeError(message))
        get_logger().error("Toplu işlem başarısız: %s", message)
        self._batch_worker = None
        self._batch_progress_dialog = None

    def _on_pipeline_changed(self) -> None:
        self.engine.mark_all_dirty()
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _set_selected_node(self, node_id: str | None) -> None:
        """`self._selected_node_id`'ye yapılan TÜM atamaların TEK geçiş noktası -- ayrıca
        `_live_tick_generation`'ı artırır ki seçili adım değiştiğinde `_LiveTickWorker`'dan
        gelecek ESKİ (farklı bir node için hesaplanmış) bir sonuç `_on_live_tick_result`
        tarafından sessizce atılsın (bkz. `__init__`'teki `_live_tick_generation` notu)."""
        self._selected_node_id = node_id
        self._live_tick_generation += 1

    def _on_step_selected(self, node_id: str) -> None:
        self._set_selected_node(node_id)
        node = self.graph.nodes[node_id]
        op_cls = self.registry.get(node.op_id)
        self.description_label.setText(description_for(node.op_id))
        self.param_form.set_params(op_cls.params, node.params)
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _on_params_changed(self, values: dict[str, Any]) -> None:
        if self._selected_node_id is None:
            return
        # Art arda gelen tuş vuruşlarının/slider hareketlerinin HER biri ayrı bir undo adımı
        # olmasın diye, bir "yazma turunun" İLK değişikliğinde (henüz bekleyen bir snapshot
        # yokken) anlık durum yakalanır; asıl PUSH mevcut debounce (`_flush_pending_params`)
        # tetiklenince olur — böylece tek bir Ctrl+Z tüm turu geri alır.
        if self._pending_param_undo_snapshot is None:
            self._pending_param_undo_snapshot = self._snapshot_dict(self.pipeline.order)
        # Node/graph durumu HER zaman senkron güncellenir (ör. bir test/çağıran hemen ardından
        # engine.evaluate() çağırdığında güncel parametreleri görsün) — sadece PAHALI kısım
        # (gerçek yeniden hesaplama + önizleme çizimi) debounce edilir, aksi halde slider
        # sürüklenirken her piksel hareketinde tam pipeline yeniden çalışır ve UI kilitlenir.
        self.graph.nodes[self._selected_node_id].params = values
        self.engine.mark_dirty(self._selected_node_id)
        self._param_debounce_timer.start(_PARAM_DEBOUNCE_MS)

    def _flush_pending_params(self) -> None:
        self._flush_pending_param_undo()
        self._refresh_preview()
        self._refresh_enum_gallery()

    def _on_enum_choice_selected(self, param_name: str, choice: str) -> None:
        if self._selected_node_id is None:
            return
        self._capture_undo_checkpoint()
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

    def _on_manual_rois_canvas_changed(self, rois: list[tuple[int, int, int, int]]) -> None:
        node_id = self._selected_node_id
        if node_id is None:
            return
        node = self.graph.nodes[node_id]
        if node.op_id not in _MANUAL_ROI_OP_IDS:
            return
        node.params = {**node.params, "manual_rois": json.dumps([list(r) for r in rois])}
        self.engine.mark_dirty(node_id)
        self.param_form.set_params(self.registry.get(node.op_id).params, node.params)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """Senkron/on-demand yol: `self.engine`/`self.graph` (canlı, UI-thread'e ait) ile
        `_build_preview_frame`'i doğrudan çağırıp sonucu widget'lara uygular. Canlı kamera
        tick'i bunun YERİNE `_LiveTickWorker` üzerinden AYNI `_build_preview_frame`'i arka
        plan thread'inde çalıştırır (bkz. `_on_camera_tick`/`_on_live_tick_result`) -- ikisi de
        TEK bir saf fonksiyonu paylaşır, kod tekrarı yok."""
        result = _build_preview_frame(
            engine=self.engine,
            registry=self.registry,
            graph=self.graph,
            pipeline_order=self.pipeline.order,
            node_id=self._selected_node_id,
            view_mode=self._view_mode,
            camera_active=self._camera_source is not None,
            last_camera_frame=self._last_camera_frame,
        )
        self._apply_preview_frame_result(result)

    def _apply_preview_frame_result(self, result: PreviewFrameResult) -> None:
        """`_build_preview_frame`'in ürettiği SAF sonucu widget'lara uygulayan tek yer --
        hem `_refresh_preview` (senkron yol) hem de `_on_live_tick_result` (arka plan worker
        sonucu, bkz. `_LiveTickWorker`) tarafından çağrılır."""
        self.results_panel.set_measurements(result.measurements)
        self.results_panel.set_step_durations(result.step_durations)
        self.image_view.set_image(result.display_image)
        self.image_view.set_measurements(result.hover_measurements)
        self.image_view.set_editing_enabled(result.is_roi_step or result.manual_roi_active)
        self.image_view.set_multi_mode(result.manual_roi_active)
        if result.is_roi_step:
            self.image_view.set_shape(result.roi_shape)
            if result.roi_shape == "CIRCLE":
                self.image_view.set_roi_circle(result.roi_cx, result.roi_cy, result.roi_r)
            else:
                self.image_view.set_roi(result.roi_x, result.roi_y, result.roi_w, result.roi_h)
        elif result.manual_rois is not None:
            self.image_view.set_rois(result.manual_rois)
        self.status_label.setText(result.status_text)
