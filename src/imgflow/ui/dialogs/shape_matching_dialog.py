"""Şekil Eşleştirme (Model Öğret) — bağımsız non-modal QDialog.

Referans bir görüntü yüklenir, `RoiCanvas` üzerinde (dikdörtgen) model bölgesi seçilir,
"Modeli Eğit" ile `core/shape_matching.train_shape_model()` çağrılır, ve model isimle
kaydedilir (`io_utils/shape_model_store.py`) — daha sonra pipeline'daki "Geometrik
Eşleştirme" operatöründen (`geom.shape_match`) `model_names` parametresiyle (birden fazla model
aynı adımda seçilebilir) kullanılabilir.

Lens/Yükseklik-Ölçek kalibrasyon dialoglarıyla (bkz. `height_scale_calibration_dialog.py`)
AYNI isimli-profil deseni: eğit → isim ver → Kaydet; kayıtlı modeller açılır listeden
Yükle/Sil/Yeniden Adlandır. Pipeline'ın kendisinden TAMAMEN bağımsızdır — model burada bir
kere eğitilir, operatör onu sadece isimle referans alır (ikinci bir zorunlu girdiye ihtiyaç
duymadan).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from imgflow.core import capture_store
from imgflow.core.auto_objects import detect_objects
from imgflow.core.roi import RoiRect
from imgflow.core.shape_matching import ShapeMatchingError, ShapeModel, train_shape_model
from imgflow.io_utils import shape_model_store
from imgflow.io_utils.image_io import load_image
from imgflow.ui.widgets.image_view import normalize_to_uint8, numpy_to_qimage
from imgflow.ui.widgets.roi_canvas import RoiCanvas

_GUIDE_TEXT = (
    "Referans Görüntüler Yükle (birden fazla seçilebilir) → galeriden birini seçin → 'Çizim "
    "Modu'nda Dikdörtgen, Daire ya da Serbest Kontur seçip model bölgesini fare ile çizin → "
    "'Modeli Eğit' → isim verip 'Kaydet'. 'Konturu Otomatik Algıla' HER ÜÇ modda da "
    "işaretlenebilir — çizilen alan İÇİNDE nesnenin gerçek dış hattı bulunup sadece bu hat "
    "üzerindeki kenarlar modele alınır (Dikdörtgen/Daire/Poligon'un dışına asla taşmaz) — "
    "'ROI'yi Nesneye Daralt' ile bulunan nesnenin sınırlayıcı kutusu YENİ ROI/çember olarak "
    "da atanabilir (gereksiz kenar boşluğunu eğitim öncesi eler). "
    "Serbest Kontur modunda sol tık nokta ekler, sağ tık son noktayı geri alır, 'Poligonu "
    "Kapat' (en az 3 nokta) konturu tamamlar, kapandıktan sonra bir köşeyi sürükleyerek "
    "düzeltebilirsiniz. Pipeline'da 'Geometrik Eşleştirme' kategorisindeki operatörde bu isim "
    "'Model Adı' açılır listesinden seçilebilir. Kayıtlı modelleri aşağıdaki listeden Yükle/"
    "Sil/Yeniden Adlandır ile yönetebilirsiniz."
)
_AUTO_CONTOUR_HELP = (
    "Konturu Otomatik Algıla: çizilen ROI/çember/poligon İÇİNDE eşikleme+bağlı bileşen "
    "analiziyle (Otomatik Nesne Tespiti ile AYNI yöntem) nesnenin gerçek dış hattı bulunur, "
    "sadece bu hat üzerindeki kenarlar modele alınır — çizilen alanın dışına asla taşmaz. "
    "Dikdörtgen modda elenen bölge canvas üzerinde canlı olarak yarı saydam KIRMIZI ile "
    "işaretlenir, böylece eğitmeden önce neyin dışlandığını görebilirsiniz. Nesne bulunamazsa "
    "(parlak/karmaşık sahne) otomatik olarak tüm çizilen alan kullanılır, eğitim ÇÖKMEZ."
)
_INVERT_MASK_HELP = (
    "Ters Çevir: normalde (bu kapalıyken) tespit edilen/çizilen konturun İÇİ kullanılır, "
    "DIŞI (kırmızı önizleme) elenir. İşaretlenirse bunun tam tersi olur — konturun İÇİ "
    "elenir, DIŞI (ROI/çember/poligon sınırları içinde) kullanılır. Örn. nesnenin kendisi "
    "değil de üzerindeki bir deliği/logoyu ya da konturun etrafındaki dokuyu öğretmek "
    "isterseniz kullanışlıdır."
)
_OUTER_ONLY_HELP = (
    "Sadece Dış Kontur: modele YALNIZCA nesnenin en dıştaki kenar noktaları alınır; iç yapı "
    "(logo, kabartma, kapak halkası, parlama izi) tamamen dışlanır. Gerçek bir vakada aynı "
    "üründen bazıları bulunurken bazıları bulunamıyordu: modelin ~%22'si iç halkadaydı ve o "
    "halka gölgede kalan ürünlerde görünmediği için skor eşiğin hemen altına düşüyordu. Dış "
    "hat aydınlatmadan çok daha az etkilendiğinden bu seçenek ürünler arası tutarlılığı "
    "artırır. UYARI: model saf silüete yaklaştığı için benzer siluetli başka şeyler de "
    "eşleşebilir — operatördeki 'İç Bölge Doğrulaması' ile BİRLİKTE kullanın."
)
_NEGLIGIBLE_EXCLUDED_RATIO = 0.01
"""Kontur önizlemesinde "elenecek arka plan pratikte yok" sayılan oran. Bunun altında kırmızı
alan gözle görülmez; kullanıcı bunu "tespit çalışmıyor" sanmasın diye ayrı bir mesaj yazılır
(gerçek kullanıcı raporu: "poligon ve çember ROI'nin içinde kontür bulamıyor")."""
_DEFAULT_MIN_OBJECT_AREA_PERCENT = 1.0
"""Otomatik kontur tespitinde bir bileşenin ROI alanının en az yüzde kaçını kaplaması
gerektiği. Gerçek kullanıcı isteği: "minimum nesne alanı sıfırdan başlamasın" -- 0'da tek
piksellik gürültü lekeleri de "nesne" sayılıp `_MULTI_PART_AREA_RATIO` karşılaştırmasını
bozabiliyordu."""
_DEFAULT_MAX_OBJECT_AREA_PERCENT = 90.0
"""Bir bileşenin en fazla ROI alanının yüzde kaçını kaplayabileceği. ROI'nin neredeyse
tamamını kaplayan bileşen pratikte ARKA PLANDIR (gerçek kullanıcı raporu: "bazen arka planı
ölçüyor"); 90 bu bileşeni eler ama ROI'ye sıkıca oturan gerçek nesneleri korur."""
_DEFAULT_ROI = (40, 40, 120, 120)
_DEFAULT_ROI_CIRCLE = (100, 100, 60)  # cx, cy, r -- _DEFAULT_ROI'nin merkezine yakın
_DRAW_MODE_SHAPES = ("RECT", "POLYGON", "CIRCLE")
"""`_draw_mode_combo`'nun sırasıyla eşleştiği `RoiCanvas.set_shape()` değerleri -- kombo kutuya
yeni bir çizim modu eklenirse burası da GÜNCELLENMELİ (sıra combo'daki `addItems` sırasıyla
BİREBİR aynı olmalı)."""
_MASK_DILATION_PX = 2
_MULTI_PART_AREA_RATIO = 0.15
"""`_build_auto_contour_mask`, ROI içinde Otsu ile ayrılan TÜM bağlı bileşenlerden sadece EN
BÜYÜĞÜNÜ değil, alanı en büyük bileşenin en az bu oranı kadar olan HEPSİNİ "nesne" sayar --
gerçek kullanıcı raporu: "eti yazısını öğrettim, sadece o yazıyı görmek istiyorum" -- yazı gibi
harfleri birbirinden AYRIK (bağlı olmayan) çok-parçalı nesnelerde eskiden sadece EN BÜYÜK harf
"nesne" sayılıp diğer harfler arka plan gibi elenip eğitim nokta filtresinden düşüyordu. Bu
nokta filtresi ARTIK overlay'in KENDİSİ olduğundan (bkz. `core/shape_matching.py::render_
match_overlay`, HALCON tarzı nokta-bulutu çizimi) bu ayrım eskisinden de önemli. Küçük gürültü
benekleri (en büyük parçanın %15'inden küçük) hâlâ elenir."""
_DEFAULT_DIALOG_SIZE = (1100, 1000)
_CONTROLS_PANEL_MIN_HEIGHT = 160
"""Kontrol paneli (yükle/çizim modu/eğit/kayıtlı modeller vb.) kendi `QScrollArea`'sına
sarılıp bu KÜÇÜK minimum yüksekliğe sabitlenir -- gerçek kullanıcı raporu: "model eğit
menüsünün boyutlandırması kötü olduğu için aşağıdaki sil butonlarını göremiyormuşum...
istersem pencereyi büyültüp küçültebileyim". Eskiden TÜM kontroller ana `QVBoxLayout`'a
doğrudan ekleniyordu -- pencere küçük bir ekranda/varsayılan boyuttan küçük açılırsa (ya da
kullanıcı daraltmaya çalışırsa) dialog'un kendi minimum yükseklik sınırı TÜM widget'ların
toplam `sizeHint`'ine dayanıyordu, bu da "Sil" gibi alttaki butonların ekran dışına
itilmesine/pencerenin daha fazla küçültülememesine yol açıyordu. Artık kontrol paneli KENDİ
İÇİNDE kaydırılabilir olduğundan dialog'un tamamı bu küçük değere kadar SERBESTÇE küçültülüp
büyütülebilir, "Sil" dahil HER buton her zaman (gerekirse panel içi kaydırarak) erişilebilir
kalır.

**Devam -- gerçek kullanıcı raporu: "aşağıdaki menüler gözükmüyor" (canvas/kontrol paneli
arasındaki `QVBoxLayout`'ta kontrol paneli stretch=0 olduğundan sadece `QScrollArea.
sizeHint()`'ini alıyordu -- bu, Qt'nin QScrollArea için içerik boyutundan BAĞIMSIZ olarak
küçük bir üst sınıra kadar SIKIŞTIRDIĞI bir değer (ölçülen: 432x288), oysa gerçek içerik
587px yükseklik istiyordu -- yani panel varsayılan pencere boyutunda bile HER ZAMAN kendi
içinde ~300px'lik gizli bir kaydırmaya ihtiyaç duyuyordu, "Sil" gibi alt satırlar bu yüzden
görünmüyordu). Düzeltme: canvas/kontrol paneli artık bir `QSplitter` (bkz. `content_splitter`)
içinde -- kullanıcı ARADAKİ tutamacı sürükleyerek her ikisine de istediği kadar alan
ayırabilir, varsayılan bölünme kontrol panelinin TAM içeriğini (587px) gösterecek şekilde
seçildi. Pencere boyutu da (1000x850 -> 1100x1000) bunu varsayılanda kaydırmadan
göstermeye yetecek şekilde büyütüldü."""
_THUMB_SIZE = 96
_TRAINED_PREVIEW_SIZE = 160
_REFERENCE_INDEX_ROLE = Qt.ItemDataRole.UserRole
_NUM_LEVELS_HELP = (
    "Piramit Seviyesi: aramanın kaç çözünürlük kademesinde (tam boy → yarı boy → çeyrek boy ...) "
    "yapılacağı. Önce EN KABA (en küçük) seviyede hızlıca birkaç aday konum/açı bulunur, sonra "
    "SADECE bu adaylar daha ince seviyelerde iyileştirilir — bu yüzden yüksek değer aramayı asıl "
    "yavaşlatmaz, aksine daha kaba bir ön-elemeyle genelde hızlandırır. Çok küçük/detaysız "
    "modellerde (küçük ROI) 1-2 seviye yeterlidir; büyük/detaylı modellerde 3-4 önerilir."
)
_MIN_GRADIENT_HELP = (
    "Min. Gradyan Oranı: bir pikselin model için 'kenar noktası' sayılması için gereken gradyan "
    "(parlaklık değişimi) büyüklüğü eşiği — ROI içindeki EN GÜÇLÜ kenara ORANLA (mutlak bir "
    "piksel değeri değil). Düşürürseniz daha SOLUK/zayıf kenarlar da modele dahil edilir (daha "
    "fazla nokta, gürültüye karşı daha az sağlam); yükseltirseniz sadece EN BELİRGİN kenarlar "
    "kullanılır (daha az ama daha güvenilir nokta). ROI'de kenar bulunamazsa eğitim hata verir — "
    "bu durumda bu değeri düşürmeyi deneyin."
)


class ShapeMatchingDialog(QDialog):
    model_trained = Signal(object)  # ShapeModel
    models_changed = Signal()
    """Kayıtlı model KÜMESİ değiştiğinde (kaydet/sil/yeniden adlandır) yayınlanır — ana
    pencere, `geom.shape_match` düğümü seçiliyse parametre panelindeki 'Model Adı' açılır
    listesini bu sinyalle canlı olarak yeniler."""
    frame_captured = Signal()
    """Aktif referans görüntü galeriye eklendiğinde yayınlanır -- `LensCalibrationDialog`/
    `HeightScaleCalibrationDialog`'daki AYNI sinyal, ana pencere `capture_gallery_panel.
    refresh`'e bağlar."""

    def __init__(self, model_dir: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Şekil Eşleştirme (Model Öğret)")
        self.setModal(False)
        # Varsayılan bir QDialog'da (Windows'ta) büyüt/tam ekran düğmesi YOKTUR -- gerçek
        # kullanıcı raporu: "onu da tam ekrana alabilelim". Sistem menüsü + büyüt/küçült
        # düğmeleri açıkça eklenip pencere başlık çubuğundaki çift tıklama/sürükleyerek
        # maksimize etme de bu sayede çalışır hale gelir.
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        # Varsayılan Qt boyutlandırması (içerik sığdırma) canvas'ı çok küçük bırakıyordu --
        # gerçek kullanıcı raporu: "görüntü çok küçük ROI seçmekte zorlanıyorum". Sabit büyük
        # bir başlangıç boyutu + aşağıdaki yakınlaştırma çubuğu (bkz. `_canvas`/`_scroll_area`),
        # `ui/main_window.py`'nin ana önizleme panelindeki AYNI ikili çözüm.
        self.resize(*_DEFAULT_DIALOG_SIZE)
        self._model_dir = model_dir if model_dir is not None else shape_model_store.SHAPE_MODEL_DIR
        self._reference_image: np.ndarray | None = None
        self._reference_images: list[np.ndarray] = []
        """`_reference_list` galerisindeki her thumbnail'e karşılık gelen tam-çözünürlük
        görüntü, sırasıyla — kullanıcı "daha fazla fotoğraf yüklenebilmeli, daha yüksek
        doğruluk payımız olsun" istedi: birden fazla referans fotoğrafı aynı anda yüklenip
        galeride bir arada tutulabilir, aralarında tıklayarak en temiz/net olanı seçip o
        pozdan model eğitilebilir (fotoğraflar BİRLEŞTİRİLMEZ — kullanıcı bilinçli olarak bu
        basit galeri yaklaşımını, kenar noktalarını birleştiren daha karmaşık alternatife
        tercih etti)."""
        self._reference_names: list[str] = []
        self._active_reference_index: int | None = None
        """Galerideki HANGİ görüntünün şu an eğitim referansı (dolayısıyla modelin 'açı=0'
        baz aldığı poz) olduğu, listede '★' öneki + `_active_reference_label` ile AÇIKÇA
        gösterilir — kullanıcı isteği: "model eğitme kısmında açının doğru alınması için bir
        tane referans seçelim". Birden fazla foto arasında hangisinin eğitime gireceği belirsiz
        kalırsa, farklı fotoğraflardaki küçük poz farkları modelin açı referansını kaydırabilir."""
        self._model: ShapeModel | None = None
        self._last_train_mask: np.ndarray | None = None
        """Son başarılı `_on_train()` çağrısında `train_shape_model`'e FİİLEN geçirilen `mask` --
        `_trained_filter_preview_label`'ın gösterdiği görüntüyü YENİDEN üretebilmek (ör. testte)
        için saklanır."""
        self._roi = _DEFAULT_ROI
        self._roi_circle = _DEFAULT_ROI_CIRCLE

        self._canvas = RoiCanvas()
        self._canvas.set_shape("RECT")
        self._canvas.set_editing_enabled(True)
        self._canvas.roi_changed.connect(self._on_roi_changed)
        self._canvas.roi_circle_changed.connect(self._on_roi_circle_changed)
        self._canvas.polygon_changed.connect(self._on_polygon_changed)
        self._canvas.setAcceptDrops(True)
        self._canvas.image_file_dropped.connect(self._on_reference_file_dropped)

        # `ui/main_window.py`'nin ana önizleme panelindeki AYNI yakınlaştırma deseni --
        # gerçek kullanıcı raporu: "görüntü çok küçük ROI seçmekte zorlanıyorum, zoom in/out
        # lazım". `RoiCanvas` (bkz. `image_view.py`) zaten zoom API'sine sahip, sadece bu
        # dialog'da hiç bağlanmamıştı.
        self._canvas_scroll_area = QScrollArea()
        self._canvas_scroll_area.setWidget(self._canvas)
        self._canvas_scroll_area.setWidgetResizable(True)
        self._canvas.set_scroll_host(self._canvas_scroll_area)

        zoom_out_button = QPushButton("−")
        zoom_out_button.setToolTip("Uzaklaştır (Ctrl+tekerlek de kullanılabilir)")
        zoom_out_button.setFixedWidth(28)
        zoom_out_button.clicked.connect(self._canvas.zoom_out)
        zoom_in_button = QPushButton("+")
        zoom_in_button.setToolTip("Yakınlaştır (Ctrl+tekerlek de kullanılabilir)")
        zoom_in_button.setFixedWidth(28)
        zoom_in_button.clicked.connect(self._canvas.zoom_in)
        zoom_fit_button = QPushButton("Sığdır")
        zoom_fit_button.setToolTip("Görüntüyü pencereye sığdır (%100 sığdırma)")
        zoom_fit_button.clicked.connect(self._canvas.zoom_reset)
        self._zoom_percent_label = QLabel("%100")
        self._zoom_percent_label.setFixedWidth(50)
        self._canvas.zoom_changed.connect(
            lambda zoom: self._zoom_percent_label.setText(f"%{round(zoom * 100)}")
        )

        # Gerçek kullanıcı isteği: "ölçüm de dahil her alanda kare yakalayıp yan ekrana
        # atabilmek istiyorum" -- Lens/Yükseklik-Ölçek kalibrasyon diyaloglarının AYNI
        # `capture_store` deposuna (`source="shape_match"`) yazan bir buton; bu diyalogda
        # canlı kamera akışı YOK (sadece yüklenmiş/sürüklenmiş referans görüntüler), bu yüzden
        # `self._reference_image` (o an aktif referans) galeriye eklenir.
        self._capture_button = QPushButton("Referansı Galeriye Ekle")
        self._capture_button.setToolTip(
            "Şu an aktif referans görüntüyü 'Yakalanan Kareler' galerisine (yan panel) ekler."
        )
        self._capture_button.setEnabled(False)
        self._capture_button.clicked.connect(self._on_capture_reference)

        zoom_bar = QHBoxLayout()
        zoom_bar.addWidget(QLabel("Yakınlaştırma:"))
        zoom_bar.addWidget(zoom_out_button)
        zoom_bar.addWidget(zoom_in_button)
        zoom_bar.addWidget(zoom_fit_button)
        zoom_bar.addWidget(self._zoom_percent_label)
        zoom_bar.addStretch(1)
        zoom_bar.addWidget(self._capture_button)

        load_button = QPushButton("Referans Görüntüler Yükle...")
        load_button.setToolTip("Birden fazla görüntü seçebilirsiniz — hepsi aşağıdaki galeriye eklenir.")
        load_button.clicked.connect(self._on_load_references)

        self._reference_list = QListWidget()
        self._reference_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._reference_list.setIconSize(QPixmap(_THUMB_SIZE, _THUMB_SIZE).size())
        self._reference_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._reference_list.setFixedHeight(_THUMB_SIZE + 48)
        self._reference_list.setToolTip("Bir referansa tıklayarak canvas'ta aktif hale getirin.")
        self._reference_list.itemClicked.connect(self._on_reference_gallery_item_clicked)
        self._active_reference_label = QLabel("Aktif referans: (henüz yok)")

        self._draw_mode_combo = QComboBox()
        self._draw_mode_combo.addItems(["Dikdörtgen (ROI)", "Serbest Kontur (Poligon)", "Daire (Çember)"])
        self._draw_mode_combo.setToolTip(
            "Dikdörtgen: köşelerden sürükleyerek boyutlandırılan klasik ROI. Serbest Kontur: "
            "nesnenin gerçek dış hattını elle nokta nokta çizersiniz — ROI dikdörtgeninin "
            "içine kaçabilecek arka plan/komşu nesne kenarları modele hiç girmez. Daire: "
            "yuvarlak nesneler için merkezden sürükleyip kenardan yarıçapı ayarlayabileceğiniz "
            "bir ROI — çemberin kendisi kontur olarak kullanılır, 'Konturu Otomatik Algıla' "
            "işaretliyse çemberin İÇİNDE nesne daha da daraltılabilir."
        )
        self._draw_mode_combo.currentIndexChanged.connect(self._on_draw_mode_changed)

        self._auto_contour_checkbox = QCheckBox("Konturu Otomatik Algıla (arka planı ele)")
        self._auto_contour_checkbox.setToolTip(_AUTO_CONTOUR_HELP)
        self._auto_contour_checkbox.toggled.connect(self._on_auto_contour_toggled)

        self._invert_mask_checkbox = QCheckBox("Kontur Dışını Kullan (ters çevir)")
        self._invert_mask_checkbox.setToolTip(_INVERT_MASK_HELP)
        self._invert_mask_checkbox.toggled.connect(self._on_auto_contour_toggled)

        self._outer_only_checkbox = QCheckBox("Sadece Dış Kontur")
        self._outer_only_checkbox.setToolTip(_OUTER_ONLY_HELP)

        # Gerçek kullanıcı isteği: "minimum nesne alanı sıfırdan başlamasın, kendi otomatik
        # belirlesin" + "max alan da seçelim, bazen arka planı ölçüyor". Değerler ROI'nin
        # YÜZDESİ olarak verilir -- ROI büyüklüğü değiştiğinde elle yeniden ayarlamak
        # gerekmez.
        self._min_object_area_spin = QDoubleSpinBox()
        self._min_object_area_spin.setRange(0.0, 100.0)
        self._min_object_area_spin.setSingleStep(0.5)
        self._min_object_area_spin.setSuffix(" %")
        self._min_object_area_spin.setValue(_DEFAULT_MIN_OBJECT_AREA_PERCENT)
        self._min_object_area_spin.setToolTip(
            "Otomatik kontur tespitinde bir bileşenin 'nesne' sayılması için ROI alanının en az "
            "yüzde kaçını kaplaması gerektiği. Varsayılan 0 DEĞİL: sıfırdan başlarsa tek piksellik "
            "gürültü lekeleri de nesne sayılır."
        )
        self._min_object_area_spin.valueChanged.connect(lambda _v: self._refresh_contour_preview())

        self._max_object_area_spin = QDoubleSpinBox()
        self._max_object_area_spin.setRange(1.0, 100.0)
        self._max_object_area_spin.setSingleStep(1.0)
        self._max_object_area_spin.setSuffix(" %")
        self._max_object_area_spin.setValue(_DEFAULT_MAX_OBJECT_AREA_PERCENT)
        self._max_object_area_spin.setToolTip(
            "Bir bileşenin en fazla ROI alanının yüzde kaçını kaplayabileceği. ROI'nin neredeyse "
            "tamamını kaplayan bileşen genelde ARKA PLANDIR; bu üst sınır onu eler. 100 = sınır yok."
        )
        self._max_object_area_spin.valueChanged.connect(lambda _v: self._refresh_contour_preview())

        self._polygon_close_button = QPushButton("Poligonu Kapat")
        self._polygon_close_button.setToolTip("En az 3 nokta çizdikten sonra konturu tamamlar.")
        self._polygon_close_button.setEnabled(False)
        self._polygon_close_button.clicked.connect(self._canvas.close_polygon)
        self._polygon_reset_button = QPushButton("Poligonu Sıfırla")
        self._polygon_reset_button.setToolTip("Çizilen tüm noktaları silip yeniden başlar.")
        self._polygon_reset_button.clicked.connect(self._on_polygon_reset)
        self._polygon_status_label = QLabel("")

        self._shrink_roi_button = QPushButton("ROI'yi Nesneye Daralt")
        self._shrink_roi_button.setToolTip(
            "Çizilen ROI/çember içinde otomatik nesne tespiti çalıştırıp, bulunan nesnenin "
            "sınırlayıcı kutusunu YENİ (daraltılmış) ROI/çember olarak belirler — gereksiz kenar "
            "boşluğunu/uzak gürültüyü eğitim ÖNCESİNDE eler, hem eğitimi hızlandırır hem "
            "kontur bulunamadığında düşülen dikdörtgen overlay'i daha sıkı yapar. Mevcut kontur "
            "çizimiyle (varsa) BİRLİKTE çalışır, onun yerine geçmez."
        )
        self._shrink_roi_button.clicked.connect(self._on_shrink_roi_to_contour)

        draw_mode_row = QHBoxLayout()
        draw_mode_row.addWidget(QLabel("Çizim Modu:"))
        draw_mode_row.addWidget(self._draw_mode_combo)
        draw_mode_row.addWidget(self._auto_contour_checkbox)
        draw_mode_row.addWidget(self._invert_mask_checkbox)
        draw_mode_row.addWidget(self._outer_only_checkbox)
        draw_mode_row.addWidget(QLabel("Min. Nesne:"))
        draw_mode_row.addWidget(self._min_object_area_spin)
        draw_mode_row.addWidget(QLabel("Maks. Nesne:"))
        draw_mode_row.addWidget(self._max_object_area_spin)
        draw_mode_row.addWidget(self._shrink_roi_button)
        draw_mode_row.addWidget(self._polygon_close_button)
        draw_mode_row.addWidget(self._polygon_reset_button)
        draw_mode_row.addWidget(self._polygon_status_label, 1)
        self._set_draw_mode_widgets_visible(_DRAW_MODE_SHAPES[0])

        self._contour_preview_label = QLabel("")
        self._contour_preview_label.setStyleSheet("color: #a33;")

        self._num_levels_spin = QSpinBox()
        self._num_levels_spin.setRange(1, 6)
        self._num_levels_spin.setValue(3)
        self._num_levels_spin.setToolTip(_NUM_LEVELS_HELP)
        num_levels_label = QLabel("Piramit Seviyesi:")
        num_levels_label.setToolTip(_NUM_LEVELS_HELP)

        self._auto_num_levels_checkbox = QCheckBox("Otomatik (HALCON tarzı)")
        self._auto_num_levels_checkbox.setToolTip(
            "HALCON'un inspect_shape_model'indeki gibi: piramit derinliği bir sonraki (daha "
            "kaba) seviyenin kenar noktası sayısı çok azalana kadar otomatik artırılır. "
            "İşaretliyken yukarıdaki sayı yoksayılır (devre dışı gösterilir)."
        )
        self._auto_num_levels_checkbox.toggled.connect(
            lambda checked: self._num_levels_spin.setEnabled(not checked)
        )

        self._min_gradient_spin = QDoubleSpinBox()
        self._min_gradient_spin.setRange(0.01, 0.95)
        self._min_gradient_spin.setSingleStep(0.05)
        self._min_gradient_spin.setValue(0.2)
        self._min_gradient_spin.setToolTip(_MIN_GRADIENT_HELP)
        min_gradient_label = QLabel("Min. Gradyan Oranı:")
        min_gradient_label.setToolTip(_MIN_GRADIENT_HELP)

        train_button = QPushButton("Modeli Eğit")
        train_button.setToolTip(
            "Seçili bölgedeki (dikdörtgen ya da serbest kontur) kenarlardan bir şekil modeli "
            "çıkarır. Modelin kaydedilmesi için önce bu adımın başarıyla tamamlanması gerekir."
        )
        train_button.clicked.connect(self._on_train)
        self._train_status_label = QLabel("")

        self._trained_filter_preview_label = QLabel("(eğitim sonrası burada gösterilir)")
        self._trained_filter_preview_label.setFixedSize(_TRAINED_PREVIEW_SIZE, _TRAINED_PREVIEW_SIZE)
        self._trained_filter_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._trained_filter_preview_label.setStyleSheet("border: 1px solid #999; color: #666;")
        self._trained_filter_preview_label.setToolTip(
            "Eğitimde FİİLEN kullanılan (mask uygulanmış) ROI kırpımı -- 'Konturu Otomatik Algıla' "
            "işaretliyken elenen arka plan burada SİYAH görünür; gerçekten filtrelenip "
            "filtrelenmediğini doğrudan gözle doğrulamak için."
        )
        self._trained_filter_preview_caption = QLabel("")
        self._trained_filter_preview_caption.setWordWrap(True)
        self._trained_filter_preview_caption.setFixedWidth(_TRAINED_PREVIEW_SIZE)
        self._trained_filter_preview_caption.setStyleSheet("font-size: 10px;")

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Model adı (ör. kapak_logo)")
        self._save_button = QPushButton("Kaydet")
        self._save_button.setToolTip("Son eğitilen/yüklenen modeli yukarıdaki isimle diske kaydeder.")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save)
        self._save_status_label = QLabel("")

        self._load_combo = QComboBox()
        self._load_combo.setToolTip("Daha önce kaydedilmiş şekil modelleri.")
        load_model_button = QPushButton("Yükle")
        load_model_button.setToolTip("Seçili modeli belleğe yükler (pipeline'daki operatörden bağımsız).")
        load_model_button.clicked.connect(self._on_load_model)
        rename_button = QPushButton("Yeniden Adlandır...")
        rename_button.setToolTip("Seçili modelin adını değiştirir.")
        rename_button.clicked.connect(self._on_rename_model)
        delete_button = QPushButton("Sil")
        delete_button.setToolTip("Seçili modeli kalıcı olarak siler.")
        delete_button.clicked.connect(self._on_delete_model)
        export_button = QPushButton("Dışa Aktar (JSON)...")
        export_button.setToolTip(
            "Seçili modeli bir .json dosyasına kaydeder — başka bir bilgisayara taşımak/"
            "yedeklemek için."
        )
        export_button.clicked.connect(self._on_export_model)
        import_button = QPushButton("İçe Aktar (JSON)...")
        import_button.setToolTip(
            "Daha önce 'Dışa Aktar' ile kaydedilmiş bir model .json dosyasını yükleyip "
            "isimli listeye ekler."
        )
        import_button.clicked.connect(self._on_import_model)

        params_row = QHBoxLayout()
        params_row.addWidget(num_levels_label)
        params_row.addWidget(self._num_levels_spin)
        params_row.addWidget(self._auto_num_levels_checkbox)
        params_row.addWidget(min_gradient_label)
        params_row.addWidget(self._min_gradient_spin)

        train_row = QHBoxLayout()
        train_status_column = QVBoxLayout()
        train_status_column.addWidget(train_button)
        train_status_column.addWidget(self._train_status_label)
        train_row.addLayout(train_status_column, 1)
        preview_column = QVBoxLayout()
        preview_column.addWidget(self._trained_filter_preview_label)
        preview_column.addWidget(self._trained_filter_preview_caption)
        train_row.addLayout(preview_column)

        save_row = QHBoxLayout()
        save_row.addWidget(self._name_edit, 1)
        save_row.addWidget(self._save_button)

        # İKİ satıra bölünmüş -- gerçek kullanıcı raporu: "eğittiğim modelleri silemiyorum...
        # silme menüsünü göremedim direkt". Eskiden 6 widget (combo + 5 buton, bazıları uzun
        # metinli: "Yeniden Adlandır...", "Dışa Aktar (JSON)...") TEK bir `QHBoxLayout`'ta idi --
        # dar bir pencerede "Sil" butonu görünür alanın dışına itilip gözden kaçabiliyordu. En
        # sık kullanılan aksiyonlar (Yükle/Sil) artık İLK satırda, daha az kullanılanlar ikincide.
        manage_row = QHBoxLayout()
        manage_row.addWidget(self._load_combo, 1)
        manage_row.addWidget(load_model_button)
        manage_row.addWidget(delete_button)

        manage_row_2 = QHBoxLayout()
        manage_row_2.addWidget(rename_button)
        manage_row_2.addWidget(export_button)
        manage_row_2.addWidget(import_button)

        guide_label = QLabel(_GUIDE_TEXT)
        guide_label.setWordWrap(True)
        guide_label.setStyleSheet("color: #666; font-style: italic;")

        # Kontrol paneli (yükle/çizim modu/eğit/kayıtlı modeller) KENDİ `QScrollArea`'sına
        # sarılıyor -- gerçek kullanıcı raporu: "menünün boyutlandırması kötü olduğu için
        # aşağıdaki sil butonlarını göremiyormuşum... istersem pencereyi büyültüp
        # küçültebileyim" (bkz. `_CONTROLS_PANEL_MIN_HEIGHT`). Canvas (yukarıda, stretch=1)
        # büyüme önceliğini korur; bu panel küçük bir minimum yüksekliğe kadar serbestçe
        # küçülüp gerekirse İÇİNDE kaydırma sunar.
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(load_button)
        controls_layout.addWidget(QLabel("Referans Galerisi:"))
        controls_layout.addWidget(self._reference_list)
        controls_layout.addWidget(self._active_reference_label)
        controls_layout.addLayout(draw_mode_row)
        controls_layout.addWidget(self._contour_preview_label)
        controls_layout.addLayout(params_row)
        controls_layout.addLayout(train_row)
        controls_layout.addLayout(save_row)
        controls_layout.addWidget(self._save_status_label)
        controls_layout.addWidget(QLabel("Kayıtlı Modeller:"))
        controls_layout.addLayout(manage_row)
        controls_layout.addLayout(manage_row_2)

        controls_scroll_area = QScrollArea()
        controls_scroll_area.setWidget(controls_widget)
        controls_scroll_area.setWidgetResizable(True)
        controls_scroll_area.setMinimumHeight(_CONTROLS_PANEL_MIN_HEIGHT)

        # `QVBoxLayout`'ta stretch=0 bir widget SADECE kendi `QScrollArea.sizeHint()`'ini alır
        # -- bu, Qt'nin içerik boyutundan bağımsız küçük bir üst sınıra kadar sıkıştırdığı bir
        # değerdir (bkz. `_CONTROLS_PANEL_MIN_HEIGHT` docstring'i), gerçek içeriğin (587px)
        # ihtiyacından çok daha az. `QSplitter` kullanmak hem varsayılanda TÜM kontrolleri
        # kaydırmadan gösterebilmemizi (aşağıdaki `setSizes`) hem de kullanıcının tutamacı
        # sürükleyerek alanı istediği gibi yeniden dağıtabilmesini sağlar.
        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.addWidget(self._canvas_scroll_area)
        content_splitter.addWidget(controls_scroll_area)
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 0)
        content_splitter.setSizes([400, 620])

        layout = QVBoxLayout(self)
        layout.addWidget(guide_label)
        layout.addLayout(zoom_bar)
        layout.addWidget(content_splitter, 1)

        self._refresh_model_combo()

    @property
    def model(self) -> ShapeModel | None:
        return self._model

    def _on_roi_changed(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = (x, y, w, h)
        self._refresh_contour_preview()

    def _on_roi_circle_changed(self, cx: int, cy: int, r: int) -> None:
        self._roi_circle = (cx, cy, r)
        # Çember taşınınca/boyutlanınca kontur önizlemesi de tazelenir (RECT'teki
        # `_on_roi_changed` ile AYNI desen) -- eskiden çemberde önizleme HİÇ güncellenmiyordu.
        self._refresh_contour_preview()

    def _set_draw_mode_widgets_visible(self, shape: str) -> None:
        is_polygon = shape == "POLYGON"
        # "Konturu Otomatik Algıla" ARTIK her 3 modda da görünür/kullanılabilir -- gerçek
        # kullanıcı isteği: "bu seçenekler kalsın ama hepsi üst üste kullanılabilsin". Poligon
        # modunda işaretlenince, çizilen poligonun İÇİNDE tespit edilen nesneyle kesişim
        # alınarak daha da daraltılır (bkz. `_on_train`, Çember'in AYNI deseni).
        self._polygon_close_button.setVisible(is_polygon)
        self._polygon_reset_button.setVisible(is_polygon)
        self._polygon_status_label.setVisible(is_polygon)
        # "ROI'yi Nesneye Daralt" SADECE RECT/CIRCLE'da anlamlı -- Poligon zaten elle çizilir,
        # daraltmaya gerek yok.
        self._shrink_roi_button.setVisible(not is_polygon)

    def _on_draw_mode_changed(self, index: int) -> None:
        shape = _DRAW_MODE_SHAPES[index]
        self._canvas.set_shape(shape)
        self._set_draw_mode_widgets_visible(shape)
        if shape == "POLYGON":
            self._on_polygon_changed(self._canvas.polygon_points())
        self._refresh_contour_preview()

    def _on_auto_contour_toggled(self, _checked: bool) -> None:
        self._refresh_contour_preview()

    def _refresh_contour_preview(self) -> None:
        """Konturu Otomatik Algıla işaretliyken, seçili ROI'nin (Dikdörtgen, Poligon ya da
        Çember) içinde HANGİ piksellerin modelden elendiğini canvas üzerinde canlı olarak
        kırmızıyla gösterir — gerçek kullanıcı raporu: "elediği arka planı göremiyorum, onu da
        görmek istiyorum".

        Gerçek kullanıcı raporu (2. tur): "poligon ve çember ROI'nin içinde kontür bulamıyor."
        Kontur aslında BULUNUYORDU (eğitim doğru çalışıyordu) ama önizleme SADECE Dikdörtgen
        modunda hesaplanıp çiziliyordu; diğer iki modda hiçbir görsel geri bildirim olmadığı
        için tespit hiç çalışmıyor gibi görünüyordu. Artık üç mod da `_on_train`'in
        kullandığıyla BİREBİR AYNI maskeyi (`_contour_region_and_mask`) gösteriyor."""
        if self._reference_image is None or not self._auto_contour_checkbox.isChecked():
            self._canvas.set_contour_preview(None)
            self._contour_preview_label.setText("")
            return

        region, mask, note = self._contour_region_and_mask()
        if mask is None or region is None:
            self._canvas.set_contour_preview(None)
            self._contour_preview_label.setText(note)
            return

        excluded = region & ~mask
        self._canvas.set_contour_preview(excluded)
        region_area = int(region.sum())
        excluded_ratio = (int(excluded.sum()) / region_area) if region_area else 0.0
        side = "dışı" if self._invert_mask_checkbox.isChecked() else "içi"
        if excluded_ratio < _NEGLIGIBLE_EXCLUDED_RATIO:
            # Kırmızı alan yok diye kullanıcı "tespit çalışmıyor" sanmasın: çizim nesneye
            # sıkıca oturduğunda elenecek arka plan GERÇEKTEN kalmaz, bu beklenen durumdur.
            self._contour_preview_label.setText(
                "Elenecek arka plan yok — çizdiğiniz alan zaten nesneye oturuyor "
                "(tespit çalıştı, elenecek bir şey bulamadı)."
            )
        else:
            self._contour_preview_label.setText(
                f"Kırmızı alan modelden elenecek (%{excluded_ratio * 100:.0f}) — "
                f"yalnızca kontur {side} kullanılacak."
            )

    def _contour_region_and_mask(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """`(çizilen ROI bölgesi, eğitimde kullanılacak maske, not)` üçlüsü.

        `_on_train`'in üç dalıyla AYNI mantığı paylaşır (aynı ROI geometrisi, aynı
        `_build_auto_contour_mask` + `&` kesişimi, aynı `_finalize_mask`) — böylece canlı
        önizleme ile gerçek eğitim asla birbirinden sapmaz. Poligon kapatılmamışsa ya da
        nesne bulunamazsa maske `None` döner."""
        gray_shape = self._reference_image.shape[:2]
        index = self._draw_mode_combo.currentIndex()

        if index == 1:  # POLYGON
            points = self._canvas.polygon_points()
            if not self._canvas.is_polygon_closed() or len(points) < 3:
                return None, None, "Poligonu kapatın (en az 3 nokta) — sonra kontur gösterilir."
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x0, y0 = min(xs), min(ys)
            roi = RoiRect(x=x0, y=y0, w=max(1, max(xs) - x0), h=max(1, max(ys) - y0))
            region_canvas = np.zeros(gray_shape, dtype=np.uint8)
            cv2.fillPoly(region_canvas, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)], 255)
            region = region_canvas.astype(bool)
        elif index == 2:  # CIRCLE
            cx, cy, r = self._roi_circle
            roi = RoiRect(x=cx - r, y=cy - r, w=2 * r, h=2 * r)
            region_canvas = np.zeros(gray_shape, dtype=np.uint8)
            cv2.circle(region_canvas, (cx, cy), r, 255, thickness=-1)
            region = region_canvas.astype(bool)
        else:  # RECT
            x, y, w, h = self._roi
            roi = RoiRect(x=x, y=y, w=w, h=h)
            clamped = roi.clamp(gray_shape[1], gray_shape[0])
            region = np.zeros(gray_shape, dtype=bool)
            region[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w] = True

        roi = roi.clamp(gray_shape[1], gray_shape[0])
        inside_mask, note = self._build_auto_contour_mask(roi, gray_shape)
        if inside_mask is None:
            return region, None, note
        # RECT'te maske doğrudan tespit sonucudur; Poligon/Çember'de çizilen şeklin İÇİYLE
        # kesiştirilir (şeklin dışına asla taşmaz) -- `_on_train` ile AYNI kural.
        base_mask = inside_mask if index == 0 else (region & inside_mask)
        return region, self._finalize_mask(base_mask), note

    def _on_polygon_changed(self, points: list) -> None:
        count = len(points)
        if self._canvas.is_polygon_closed():
            self._polygon_status_label.setText(f"Poligon kapatıldı ({count} nokta) — bir köşeyi sürükleyip düzeltebilirsiniz.")
        else:
            self._polygon_status_label.setText(f"{count} nokta çizildi (kapatmak için en az 3 gerekli).")
        self._polygon_close_button.setEnabled(not self._canvas.is_polygon_closed() and count >= 3)
        # Poligon kapatılınca/köşesi taşınınca kontur önizlemesi tazelenir (çizim sürerken
        # `_contour_region_and_mask` "poligonu kapatın" notunu döndürür, boşa iş yapılmaz).
        self._refresh_contour_preview()

    def _on_polygon_reset(self) -> None:
        self._canvas.clear_polygon()
        self._on_polygon_changed([])

    def _on_load_references(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(
            self, "Referans Görüntüler Seç", "", "Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        last_index: int | None = None
        for path in paths:
            try:
                image = load_image(path)
            except FileNotFoundError as exc:
                QMessageBox.critical(self, "Görüntü Yüklenemedi", f"{path}: {exc}")
                continue
            last_index = self._add_reference_to_gallery(path, image)
        if last_index is not None:
            self._activate_reference(last_index)

    def _on_reference_file_dropped(self, path: str) -> None:
        """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) canvas'a sürükleyip
        bırakılan bir görüntüyü `_on_load_references` ile AYNI şekilde galeriye ekler ve
        aktif referans yapar."""
        try:
            image = load_image(path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Görüntü Yüklenemedi", str(exc))
            return
        index = self._add_reference_to_gallery(path, image)
        self._activate_reference(index)

    def _add_reference_to_gallery(self, path: str, image: np.ndarray) -> int:
        index = len(self._reference_images)
        self._reference_images.append(image)
        self._reference_names.append(Path(path).name)
        item = QListWidgetItem(self._reference_names[index])
        item.setData(_REFERENCE_INDEX_ROLE, index)
        item.setIcon(QIcon(self._reference_thumbnail(image)))
        self._reference_list.addItem(item)
        return index

    def _reference_thumbnail(self, image: np.ndarray) -> QPixmap:
        qimage = numpy_to_qimage(normalize_to_uint8(image))
        return QPixmap.fromImage(qimage).scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _build_filtered_training_preview(self, roi: RoiRect, mask: np.ndarray | None) -> np.ndarray:
        """Eğitimde `train_shape_model`'e FİİLEN geçirilen `mask` ile ROI kırpımını üretir --
        `mask` doluysa elenen (dışlanan) pikseller SİYAHA boyanır, böylece "Konturu Otomatik
        Algıla"nın gerçekten uygulanıp uygulanmadığı gözle doğrudan doğrulanabilir (gerçek
        kullanıcı raporu: "bence eğitmiyor" -- eğitim ÖNCESİ canvas önizlemesi eğitim SONRASI
        gerçekten NEYİN kullanıldığını göstermiyordu). `mask` `None`'sa (checkbox kapalı/kontur
        bulunamadı) ham kırpım DEĞİŞTİRİLMEDEN döner -- kullanıcı checkbox'ı açıp kapatarak
        farkı doğrudan karşılaştırabilir."""
        clamped = roi.clamp(self._reference_image.shape[1], self._reference_image.shape[0])
        crop = self._reference_image[
            clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w
        ].copy()
        if mask is not None:
            local_mask = mask[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w]
            crop[~local_mask] = 0
        return crop

    def _update_trained_filter_preview(self, roi: RoiRect, mask: np.ndarray | None) -> None:
        preview = self._build_filtered_training_preview(roi, mask)
        qimage = numpy_to_qimage(normalize_to_uint8(preview))
        pixmap = QPixmap.fromImage(qimage).scaled(
            _TRAINED_PREVIEW_SIZE,
            _TRAINED_PREVIEW_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._trained_filter_preview_label.setPixmap(pixmap)
        # Gerçek kullanıcı raporu: "kırpılmış bölge siyah gözüküyor ama hâlâ şekil bulurken o
        # bölgeyi de seçiyor" -- kök neden 'Ters Çevir' işaretliyken (kullanıcıda DAHA ÖNCE de
        # olmuş bir yanlış tıklama) önizlemede SİYAH görünenin aslında NESNENİN KENDİSİ olması
        # (arka plan DEĞİL) ve overlay'in bu modda BİLİNÇLİ olarak dikdörtgene düşmesiydi (bkz.
        # `_on_train`/`_finalize_mask`) -- kullanıcıya hiç AÇIKLANMIYORDU. Bu üçüncü satır bunu
        # açıkça söyler.
        if mask is None:
            caption, style = "(filtre uygulanmadı)", "font-size: 10px; color: #666;"
        elif self._invert_mask_checkbox.isChecked():
            caption = (
                "⚠ Korunan: ARKA PLAN, nesne elendi ('Ters Çevir' işaretli) — "
                "overlay dikdörtgene düşecek"
            )
            style = "font-size: 10px; color: #b35900; font-weight: bold;"
        else:
            caption, style = "Korunan: Nesne (arka plan elendi)", "font-size: 10px; color: #2a7a2a;"
        self._trained_filter_preview_caption.setText(caption)
        self._trained_filter_preview_caption.setStyleSheet(style)

    def _on_reference_gallery_item_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(_REFERENCE_INDEX_ROLE)
        if index is None or index >= len(self._reference_images):
            return
        self._activate_reference(index)

    def _activate_reference(self, index: int) -> None:
        """Galeride TEK bir görüntüyü aktif eğitim referansı yapar ve bunu listede AÇIKÇA
        ('★' öneki + `_active_reference_label`) işaretler — kullanıcı isteği: "model eğitme
        kısmında açının doğru alınması için bir tane referans seçelim". `LensCalibrationDialog`
        'daki '★ REFERANS' işaretleme deseniyle AYNI: birden fazla foto arasında hangisinin
        şu an modelin 'açı=0' baz aldığı poz olduğu belirsiz kalmasın diye."""
        self._active_reference_index = index
        self._use_reference_image(self._reference_images[index])
        self._active_reference_label.setText(f"Aktif referans: {self._reference_names[index]}")
        for i in range(self._reference_list.count()):
            item = self._reference_list.item(i)
            item_index = item.data(_REFERENCE_INDEX_ROLE)
            prefix = "★ " if item_index == index else ""
            item.setText(f"{prefix}{self._reference_names[item_index]}")

    def _use_reference_image(self, image: np.ndarray) -> None:
        self._reference_image = image
        self._capture_button.setEnabled(True)
        self._canvas.set_image(image)
        # Gerçek kullanıcı raporu: "yüklediğim görüntü çok küçük kalabiliyor" -- kök neden bir
        # ÖNCEKİ referansta yakınlaştırma/uzaklaştırma kullanılmışsa `_zoom`'un sıfırlanmadan
        # kalması: yeni bir referans/galeri öğesi seçildiğinde eski (ör. uzaklaştırılmış) zoom
        # seviyesi YENİ görüntüye de uygulanıyordu. `zoom_reset()` (zoom=1.0 -- görüntüyü
        # canvas viewport'una SIĞDIRIR, bkz. `image_view.py::_rescale`) her referans
        # değişiminde HER ZAMAN çağrılarak baştan sığdırılmış bir görünümle başlanır.
        self._canvas.zoom_reset()
        x, y, w, h = self._roi
        self._canvas.set_roi(x, y, w, h)
        cx, cy, r = self._roi_circle
        self._canvas.set_roi_circle(cx, cy, r)
        self._canvas.clear_polygon()
        self._on_polygon_changed([])
        self._train_status_label.setText("")
        self._refresh_contour_preview()

    def _on_capture_reference(self) -> None:
        if self._reference_image is None:
            return
        capture_store.save_capture(self._reference_image, source="shape_match")
        self.frame_captured.emit()

    @staticmethod
    def _dilate_mask(mask: np.ndarray) -> np.ndarray:
        """Otomatik kontur/poligon maskesini birkaç piksel genişletir -- kullanıcının elle
        çizdiği kontur ya da Otsu'nun bulduğu nesne sınırı, kenarın gerçek Sobel gradyan tepe
        noktasından 1-2 piksel içeride/dışında kalabilir; bu pay olmadan nesnenin KENDİ dış
        hattı bile maskeden dışlanıp eğitim gereksiz yere ('yeterli kenar noktası yok')
        başarısız olabilir."""
        kernel = np.ones((2 * _MASK_DILATION_PX + 1, 2 * _MASK_DILATION_PX + 1), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    def _build_auto_contour_mask(
        self, roi: RoiRect, gray_shape: tuple[int, int]
    ) -> tuple[np.ndarray | None, str]:
        """`roi` dikdörtgeninin görüntü kırpımında `core/auto_objects.py::detect_objects()`
        (Otomatik Nesne Tespiti ile AYNI fonksiyon, `robust=True` -- eğitim TEK SEFERLİK
        olduğundan performans kısıtı yok, kenar-destekli daha sağlam tespit kullanılır) çalıştırıp
        EN BÜYÜK maskeli nesneyi tam-görüntü boyutunda bir bool maskeye yerleştirir. Nesne
        bulunamazsa (ör. parlak/karmaşık sahnede tespit başarısız) `None` + kullanıcıya
        gösterilecek inline bir not döner — eğitim eskisi gibi tüm ROI'yi kullanarak devam eder,
        ÇÖKMEZ. Dönen maske HENÜZ ne ters çevrilmiş (bkz. `_finalize_mask`) ne de
        genişletilmiştir — konturun İÇİ `True`'dur."""
        clamped = roi.clamp(gray_shape[1], gray_shape[0])
        crop = self._reference_image[clamped.y : clamped.y + clamped.h, clamped.x : clamped.x + clamped.w]
        # Alan sınırları ROI'nin YÜZDESİ olarak verilir: mutlak piksel değeri ROI büyüklüğüne
        # göre her seferinde elle ayarlanmak zorunda kalırdı. Üst sınır ayrıca "tüm ROI'yi
        # kaplayan arka plan bileşeni"ni doğrudan eler (gerçek kullanıcı raporu: "bazen arka
        # planı ölçüyor").
        crop_area = float(crop.shape[0] * crop.shape[1])
        min_area = crop_area * self._min_object_area_spin.value() / 100.0
        max_area = crop_area * self._max_object_area_spin.value() / 100.0
        objects = detect_objects(
            crop,
            min_area=min_area,
            max_area=max_area if self._max_object_area_spin.value() < 100 else 0.0,
            robust=True,
            polarity="auto",
        )
        if not objects:
            # Gerçek kullanıcı raporu: "poligon ve çember ROI'nin içinde kontür bulamıyor."
            # En sık nedeni bir HATA değil, ROI'nin nesneye ÇOK SIKI oturması: kırpımın içinde
            # arka plan kalmadığında ayırt edilecek bir "nesne/zemin" ayrımı da kalmaz (ve
            # tüm kırpımı kaplayan tek bileşen 'Maks. Nesne' sınırına takılır). Mesaj bu yüzden
            # ne yapılacağını söyler; eğitim zaten tüm ROI ile GÜVENLE devam eder.
            return None, (
                "Bu ROI'de ayrı bir nesne ayırt edilemedi — tüm ROI kullanılacak. "
                "ROI'yi nesnenin çevresinde biraz GENİŞ çizin (kırpımda bir miktar zemin "
                "görünmeli) ya da 'Maks. Nesne' yüzdesini yükseltin."
            )
        largest_area = max(int(o.mask.sum()) for o in objects)
        significant = [o for o in objects if int(o.mask.sum()) >= largest_area * _MULTI_PART_AREA_RATIO]
        mask = np.zeros(gray_shape, dtype=bool)
        for obj in significant:
            mask[
                clamped.y + obj.y : clamped.y + obj.y + obj.h,
                clamped.x + obj.x : clamped.x + obj.x + obj.w,
            ] |= obj.mask
        return mask, ""

    def _finalize_mask(self, inside_mask: np.ndarray) -> np.ndarray:
        """`inside_mask`: konturun/poligonun İÇİ `True` olan ham (genişletilmemiş) maske.
        `_invert_mask_checkbox` işaretliyse tutulacak taraf DIŞ olacak şekilde önce ters
        çevrilir, SONRA genişletme uygulanır — sıralama önemli: genişletme her zaman
        TUTULACAK tarafın sınırını öteki tarafa doğru birkaç piksel genişletir (kullanıcının
        elle çizdiği/Otsu'nun bulduğu sınır, gerçek Sobel kenarından 1-2px sapmış olsa bile
        o sınıra en yakın kenar noktaları dışlanmasın diye) — ters çevirmeden önce genişletilse
        bu pay ters tarafa da sızar ve gerçek kontur sınırındaki noktalar öteki modda
        dışlanabilirdi."""
        kept = ~inside_mask if self._invert_mask_checkbox.isChecked() else inside_mask
        return self._dilate_mask(kept)

    def _on_shrink_roi_to_contour(self) -> None:
        """Yeni önerilen yöntem — gerçek kullanıcı isteği: "roi elle olsun şekille olsun
        seçildikten sonra otomatik kontür özelliği çalışmalı. elediğimiz kontürü hesaba katıp
        cismin etrafını çizebilir ve yeni roi olarak belirleyebiliriz." Çizilen ROI/çember
        içinde `_build_auto_contour_mask` (MEVCUT, DEĞİŞMEDEN) ile nesne tespit edilip, bulunan
        maskenin sınırlayıcı kutusu YENİ (daraltılmış) ROI/çember olarak atanır — gereksiz kenar
        boşluğunu/uzak gürültüyü eğitim ÖNCESİNDE eler. `_on_train` hâlâ AYNI akıştan geçer,
        sadece ROI artık daha sıkı olduğundan eğitilen nokta bulutu (overlay'in KENDİSİ, bkz.
        `core/shape_matching.py::render_match_overlay`) daha az gereksiz kenar boşluğu içerir.
        Nesne bulunamazsa (ör. parlak/karmaşık sahne) ROI/çember DEĞİŞMEDEN kalır, sessizce bir
        not gösterilir — çökme YOK."""
        if self._reference_image is None:
            return
        gray_shape = self._reference_image.shape[:2]
        is_circle = self._draw_mode_combo.currentIndex() == 2
        if is_circle:
            cx, cy, r = self._roi_circle
            roi = RoiRect(x=cx - r, y=cy - r, w=2 * r, h=2 * r).clamp(gray_shape[1], gray_shape[0])
        else:
            x, y, w, h = self._roi
            roi = RoiRect(x=x, y=y, w=w, h=h).clamp(gray_shape[1], gray_shape[0])

        inside_mask, note = self._build_auto_contour_mask(roi, gray_shape)
        if inside_mask is None:
            self._train_status_label.setText(note or "Nesne bulunamadı, ROI değiştirilmedi.")
            return
        ys, xs = np.nonzero(inside_mask)
        if ys.size == 0:
            self._train_status_label.setText("Nesne bulunamadı, ROI değiştirilmedi.")
            return
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())

        if is_circle:
            new_cx = (x0 + x1) // 2
            new_cy = (y0 + y1) // 2
            new_r = max(1, max(x1 - x0, y1 - y0) // 2)
            self._roi_circle = (new_cx, new_cy, new_r)
            self._canvas.set_roi_circle(new_cx, new_cy, new_r)
        else:
            new_w = max(1, x1 - x0 + 1)
            new_h = max(1, y1 - y0 + 1)
            self._roi = (x0, y0, new_w, new_h)
            self._canvas.set_roi(x0, y0, new_w, new_h)
        self._train_status_label.setText("ROI nesnenin sınırlayıcı kutusuna daraltıldı.")
        self._refresh_contour_preview()

    def _on_train(self) -> None:
        if self._reference_image is None:
            QMessageBox.warning(self, "Referans Görüntü Gerekli", "Önce bir referans görüntü yükleyin.")
            return

        gray_shape = self._reference_image.shape[:2]
        mask: np.ndarray | None = None
        note = ""

        if self._draw_mode_combo.currentIndex() == 1:
            points = self._canvas.polygon_points()
            if not self._canvas.is_polygon_closed() or len(points) < 3:
                self._train_status_label.setText(
                    "Poligon kapatılmadı: en az 3 nokta çizip 'Poligonu Kapat'a basın."
                )
                return
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x0, y0 = min(xs), min(ys)
            roi = RoiRect(x=x0, y=y0, w=max(1, max(xs) - x0), h=max(1, max(ys) - y0))
            roi = roi.clamp(gray_shape[1], gray_shape[0])
            mask_canvas = np.zeros(gray_shape, dtype=np.uint8)
            cv2.fillPoly(mask_canvas, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)], 255)
            base_mask = mask_canvas.astype(bool)
            # Poligonun çizilmesi kendi başına ZATEN bir kontur -- "Konturu Otomatik Algıla"
            # işaretliyse bu, poligonun İÇİNDE nesneyi daha da daraltmak için EK bir arıtma
            # katmanı olur (`&` kesişimi sayesinde poligonun dışına ASLA taşmaz) -- gerçek
            # kullanıcı isteği: "bu seçenekler kalsın ama hepsi üst üste kullanılabilsin".
            if self._auto_contour_checkbox.isChecked():
                inside_mask, detect_note = self._build_auto_contour_mask(roi, gray_shape)
                if inside_mask is not None:
                    base_mask = base_mask & inside_mask
                    note = detect_note
            mask = self._finalize_mask(base_mask)
        elif self._draw_mode_combo.currentIndex() == 2:
            cx, cy, r = self._roi_circle
            roi = RoiRect(x=cx - r, y=cy - r, w=2 * r, h=2 * r)
            roi = roi.clamp(gray_shape[1], gray_shape[0])
            circle_canvas = np.zeros(gray_shape, dtype=np.uint8)
            cv2.circle(circle_canvas, (cx, cy), r, 255, thickness=-1)
            base_mask = circle_canvas.astype(bool)
            # Çemberin çizilmesi kendi başına ZATEN bir kontur (Poligon'daki `mask_canvas` ile
            # AYNI desen) -- "Konturu Otomatik Algıla" işaretliyse bu, çemberin İÇİNDE nesneyi
            # daha da daraltmak için EK bir arıtma katmanı olur (`&` kesişimi sayesinde çemberin
            # dışına ASLA taşmaz).
            if self._auto_contour_checkbox.isChecked():
                inside_mask, detect_note = self._build_auto_contour_mask(roi, gray_shape)
                if inside_mask is not None:
                    base_mask = base_mask & inside_mask
                    note = detect_note
            mask = self._finalize_mask(base_mask)
        else:
            x, y, w, h = self._roi
            roi = RoiRect(x=x, y=y, w=w, h=h)
            # Eğitim/eşleştirmede KULLANILAN nokta filtresi (`mask`) SADECE checkbox
            # açıkken ayarlanır -- kapalıyken tüm ROI dikdörtgeni kullanılır (davranış eskisiyle
            # BİREBİR aynı). Bu nokta filtresi ARTIK overlay'in KENDİSİ olduğundan (bkz.
            # `core/shape_matching.py::render_match_overlay`), checkbox'ın etkisi hem eğitim
            # kalitesine hem overlay'in görünümüne DOĞRUDAN yansır.
            if self._auto_contour_checkbox.isChecked():
                inside_mask, detect_note = self._build_auto_contour_mask(roi, gray_shape)
                note = detect_note
                if inside_mask is not None:
                    mask = self._finalize_mask(inside_mask)

        num_levels = None if self._auto_num_levels_checkbox.isChecked() else self._num_levels_spin.value()
        try:
            model = train_shape_model(
                self._reference_image,
                roi,
                num_levels=num_levels,
                min_gradient_fraction=self._min_gradient_spin.value(),
                mask=mask,
                outer_contour_only=self._outer_only_checkbox.isChecked(),
            )
        except ShapeMatchingError as exc:
            self._train_status_label.setText(f"Eğitim başarısız: {exc}")
            return

        self._model = model
        point_counts = ", ".join(str(lvl.points.shape[0]) for lvl in model.levels)
        status = f"Model eğitildi (seviye başına nokta sayısı: {point_counts})."
        if note:
            status += f" {note}"
        self._train_status_label.setText(status)
        self._last_train_mask = mask
        self._update_trained_filter_preview(roi, mask)
        self._save_button.setEnabled(True)
        self.model_trained.emit(model)

    def _on_save(self) -> None:
        if self._model is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Model Adı Gerekli", "Kaydetmek için bir model adı girin.")
            return

        shape_model_store.save_shape_model(name, self._model, directory=self._model_dir)
        self._save_status_label.setText(f"Kaydedildi: '{name}'")
        self._refresh_model_combo()
        self.models_changed.emit()

    def _refresh_model_combo(self) -> None:
        current = self._load_combo.currentText()
        self._load_combo.clear()
        names = shape_model_store.list_shape_models(self._model_dir)
        self._load_combo.addItems(names)
        if current in names:
            self._load_combo.setCurrentText(current)

    def _on_load_model(self) -> None:
        name = self._load_combo.currentText().strip()
        if not name:
            return
        try:
            model = shape_model_store.load_shape_model(name, directory=self._model_dir)
        except shape_model_store.ShapeModelNotFoundError as exc:
            self._save_status_label.setText(str(exc))
            return

        self._model = model
        self._save_button.setEnabled(True)
        self._name_edit.setText(name)
        self._save_status_label.setText(f"Yüklendi: '{name}'")
        self.model_trained.emit(model)

    def _on_delete_model(self) -> None:
        name = self._load_combo.currentText().strip()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Modeli Sil", f"'{name}' adlı model kalıcı olarak silinsin mi?"
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            shape_model_store.delete_shape_model(name, directory=self._model_dir)
        except OSError as exc:
            # Gerçek kullanıcı raporu: "öğrettiğim şekilleri silemiyorum" -- `unlink()`
            # önceden burada hiç YAKALANMIYORDU: dosya salt-okunur/başka bir işlem
            # tarafından kilitli olduğunda (ör. bir antivirüs/yedekleme aracı) silme sessizce
            # BAŞARISIZ oluyor, kullanıcıya HİÇBİR geri bildirim gitmiyordu -- ekranda "hiçbir
            # şey olmuyormuş" gibi görünüyordu. Artık `_on_rename_model`'daki hata gösterimiyle
            # AYNI desende, ne olduğu açıkça söyleniyor; model listeden/bellekten SİLİNMEDİ
            # sayılır (aşağıdaki başarı-yolu koşulsuz ATLANIR).
            QMessageBox.critical(
                self,
                "Silme Başarısız",
                f"'{name}' silinemedi: {exc}\n\nDosya başka bir programda açık/kilitli "
                "olabilir ya da salt-okunur olabilir.",
            )
            return

        if self._name_edit.text().strip() == name:
            self._name_edit.clear()
            self._model = None
            self._save_button.setEnabled(False)
        self._save_status_label.setText(f"Silindi: '{name}'")
        self._refresh_model_combo()
        self.models_changed.emit()

    def _on_rename_model(self) -> None:
        old_name = self._load_combo.currentText().strip()
        if not old_name:
            return
        new_name, ok = QInputDialog.getText(self, "Yeniden Adlandır", "Yeni ad:", text=old_name)
        new_name = new_name.strip()
        if not ok or not new_name or new_name == old_name:
            return

        try:
            shape_model_store.rename_shape_model(old_name, new_name, directory=self._model_dir)
        except (shape_model_store.ShapeModelNotFoundError, FileExistsError) as exc:
            QMessageBox.critical(self, "Yeniden Adlandırma Başarısız", str(exc))
            return

        if self._name_edit.text().strip() == old_name:
            self._name_edit.setText(new_name)
        self._save_status_label.setText(f"'{old_name}' -> '{new_name}' olarak yeniden adlandırıldı.")
        self._refresh_model_combo()
        self.models_changed.emit()

    def _on_export_model(self) -> None:
        """Seçili (isimli) modeli, uygulamanın kendi `~/.imgflow/shape_models` deposunun
        DIŞINA, kullanıcının seçtiği herhangi bir konuma `.json` olarak yazar — başka bir
        makineye taşımak/yedeklemek/paylaşmak için. `save_shape_model`'in yazdığıyla AYNI
        `{"name": ..., "model": ...}` zarfı kullanılır ki `_on_import_model` (ve elle
        `shape_model_store` deposuna kopyalanan bir dosya) sorunsuz geri okunabilsin."""
        name = self._load_combo.currentText().strip()
        if not name:
            return
        try:
            model = shape_model_store.load_shape_model(name, directory=self._model_dir)
        except shape_model_store.ShapeModelNotFoundError as exc:
            self._save_status_label.setText(str(exc))
            return

        path, _filter = QFileDialog.getSaveFileName(
            self, "Modeli Dışa Aktar", f"{name}.json", "JSON (*.json)"
        )
        if not path:
            return
        payload = {"name": name, "model": model.to_dict()}
        try:
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Dışa Aktarma Başarısız", str(exc))
            return
        self._save_status_label.setText(f"'{name}' dışa aktarıldı: '{path}'")

    def _on_import_model(self) -> None:
        """`_on_export_model`'in yazdığı (ya da elle aynı `{"name", "model"}` zarfında
        hazırlanmış) bir `.json` dosyasını okuyup HEM belleğe (`self._model`, diğer model
        kaynaklarıyla — eğitim/`_on_load_model` — AYNI şekilde) HEM DE isimli depoya
        (`shape_model_store.save_shape_model`) kaydeder — kullanıcı ayrıca 'Kaydet'e basmak
        zorunda kalmadan içe aktarılan model hemen 'Kayıtlı Modeller' listesinde ve
        `geom.shape_match` operatörünün 'Model Adı' listesinde görünür. Bozuk/uyumsuz bir
        dosya sessizce YUTULMAZ, ama uygulamayı da ÇÖKERTMEZ (`json.JSONDecodeError`/
        `KeyError`/`ValueError`/`TypeError` hepsi kullanıcıya okunur bir mesajla gösterilir).

        Gerçek kullanıcı raporu: bir PİPELİNE REÇETESİNİ (`Dosya > Reçeteyi Kaydet...`, bkz.
        `io_utils/recipe.py`) buraya sürükleyip `KeyError: 'levels'` almış — reçete SADECE
        node/edge/param grafiğini saklar, hiç piksel verisi İÇERMEZ, bu yüzden ne kadar
        geçerli bir JSON olursa olsun bir şekil modeli DEĞİLDİR. Bu durum aşağıda AYRI ve
        ÖNCE kontrol edilip kullanıcıya ham `KeyError` yerine ne yapması gerektiğini
        (`Dosya > Görüntüyü Dışa Aktar...` ile önce gerçek bir resme çevirmesi) söyleyen
        özel bir mesaj gösterilir."""
        path, _filter = QFileDialog.getOpenFileName(self, "Model İçe Aktar", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "İçe Aktarma Başarısız", f"'{path}' okunamadı: {exc}")
            return

        if isinstance(data, dict) and "nodes" in data and "edges" in data:
            QMessageBox.critical(
                self,
                "İçe Aktarma Başarısız",
                f"'{path}' bir ŞEKİL MODELİ değil, bir PİPELİNE REÇETESİ (Dosya > Reçeteyi "
                "Kaydet...) gibi görünüyor — reçeteler sadece pipeline adımlarını saklar, "
                "piksel/görüntü verisi İÇERMEZ. Önce reçeteyi yükleyip Dosya menüsündeki "
                "'Görüntüyü Dışa Aktar (PNG/JPG)...' ile o adımın çıktısını bir resim "
                "dosyasına kaydedin, sonra o resmi yukarıdaki 'Referans Görüntüler "
                "Yükle...' ile buraya yükleyip modeli burada eğitin.",
            )
            return

        try:
            model_dict = data["model"] if isinstance(data, dict) and "model" in data else data
            model = ShapeModel.from_dict(model_dict)
        except (KeyError, ValueError, TypeError) as exc:
            QMessageBox.critical(
                self,
                "İçe Aktarma Başarısız",
                f"'{path}' geçerli bir şekil modeli JSON'u olarak okunamadı: {exc}",
            )
            return

        default_name = data.get("name") if isinstance(data, dict) else None
        name = str(default_name).strip() if default_name else Path(path).stem
        if not name:
            name = Path(path).stem

        self._model = model
        self._name_edit.setText(name)
        self._save_button.setEnabled(True)
        shape_model_store.save_shape_model(name, model, directory=self._model_dir)
        self._save_status_label.setText(f"İçe aktarıldı ve '{name}' olarak kaydedildi.")
        self._refresh_model_combo()
        self.models_changed.emit()
