"""Kategorilere ayrılmış, aranabilir, checkbox'lı operatör kütüphanesi paneli.

Bir operatör işaretlendiğinde pipeline'a eklenir, işareti kaldırıldığında çıkarılır.
Burada kategorisi tanımlanmamış bir operatör "Diğer" altında listelenir, hiçbiri kaybolmaz.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.custom_filters import OP_ID_PREFIX, list_custom_filters

_CATEGORY_BY_OP_ID = {
    "io.image_source": "Girdi/Çıktı (I/O)",
    "correction.flat_field": "Aydınlatma Düzeltme",
    "color.grayscale": "Görüntü / Renk",
    "color.rgb": "Görüntü / Renk",
    "color.hsv": "Görüntü / Renk",
    "segment.threshold": "Görüntü / Renk",
    "filter.gaussian_blur": "Filtreler",
    "filter.median_blur": "Filtreler",
    "filter.bilateral": "Filtreler",
    "filter.canny_edge": "Filtreler",
    "filter.sobel_edge": "Filtreler",
    "filter.laplacian_edge": "Filtreler",
    "filter.sharpen": "Filtreler",
    "filter.histogram_equalize": "Filtreler",
    "morphology.erode": "Morfoloji",
    "morphology.dilate": "Morfoloji",
    "morphology.open": "Morfoloji",
    "morphology.close": "Morfoloji",
    "segment.connected_components": "Bölgeler / Ölçüm",
    "analysis.region_props": "Bölgeler / Ölçüm",
    "roi.region": "ROI",
    "geom.shape_match": "Geometrik Eşleştirme",
    "analysis.color_props": "Renk / Doku Analizi",
    "analysis.texture_props": "Renk / Doku Analizi",
    "ml.onnx_detect": "Yapay Zeka (ONNX)",
}
UNCATEGORIZED_LABEL = "Diğer"
_CATEGORY_ORDER = [
    "Girdi/Çıktı (I/O)",
    "Aydınlatma Düzeltme",
    "Görüntü / Renk",
    "Filtreler",
    "Morfoloji",
    "Bölgeler / Ölçüm",
    "ROI",
    "Geometrik Eşleştirme",
    "Renk / Doku Analizi",
    "Yapay Zeka (ONNX)",
    UNCATEGORIZED_LABEL,
]

_LABEL_BY_OP_ID = {
    "io.image_source": "Görüntü Yükle",
    "correction.flat_field": "Aydınlatma Düzeltme (Flat-Field)",
    "color.grayscale": "Siyah-Beyaz",
    "color.rgb": "RGB",
    "color.hsv": "HSV",
    "segment.threshold": "Eşikleme (Threshold)",
    "filter.gaussian_blur": "Gaussian Bulanıklaştırma",
    "filter.median_blur": "Median Bulanıklaştırma",
    "filter.bilateral": "Bilateral Filtre",
    "filter.canny_edge": "Kenar Tespiti (Canny)",
    "filter.sobel_edge": "Kenar Tespiti (Sobel)",
    "filter.laplacian_edge": "Kenar Tespiti (Laplacian)",
    "filter.sharpen": "Keskinleştirme",
    "filter.histogram_equalize": "Kontrast Artırma (Histogram Eşitleme)",
    "morphology.erode": "Aşındırma (Erosion)",
    "morphology.dilate": "Yayma (Dilation)",
    "morphology.open": "Açma (Opening)",
    "morphology.close": "Kapama (Closing)",
    "segment.connected_components": "Bağlı Bileşenler",
    "analysis.region_props": "Bölge Ölçümü",
    "roi.region": "İlgi Alanı (ROI)",
    "geom.shape_match": "Şekil Eşleştirme (Bul)",
    "analysis.color_props": "Renk Analizi (LAB)",
    "analysis.texture_props": "Doku Analizi (GLCM)",
    "ml.onnx_detect": "ONNX Nesne Tespiti (YOLO)",
}

_DESCRIPTION_BY_OP_ID = {
    "io.image_source": (
        "Diskten bir görüntü dosyası yükler; pipeline'ın başlangıç adımıdır. Tek seferlik statik "
        "görüntü analizinde (numune fotoğrafı, arşiv incelemesi) kullanılır — canlı kamera akışı "
        "için Kamera menüsündeki USB/GigE/Video seçeneklerini kullanın."
    ),
    "correction.flat_field": (
        "Araçlar > Aydınlatma Referansı Kaydet... ile önceden kaydedilmiş BOŞ/düz aydınlatılmış "
        "bir referans kareye (ör. boş bant) göre kenarlara doğru kararan (vignetting) ya da eşit "
        "olmayan aydınlatmayı düzeltir (HALCON'daki div_image mantığına benzer). Pipeline'ın en "
        "başına, eşikleme/renk analizi gibi aydınlatmaya duyarlı adımlardan ÖNCE eklenmelidir."
    ),
    "color.grayscale": (
        "Görüntüyü gri tonlamaya çevirir; CLAHE, eşikleme (OTSU/Adaptive/Manuel) ve tersini alma "
        "içerir. Şekil/boyut/kusur analizinde (parça sayma, alan ölçümü, kontur çıkarma) renk "
        "bilgisine ihtiyaç olmayan durumların standart ilk adımıdır — PCB lehim noktası "
        "kontrolünden gıda ambalaj kusuruna kadar en sık kullanılan moddur."
    ),
    "color.rgb": (
        "RGB renk kanallarıyla çalışır: kanal kazançları, otomatik beyaz dengesi, tek kanal "
        "izleme. Baskı/etiket renk doğrulama gibi renk TUTARLILIĞI denetiminde ve farklı LED/lamba "
        "renklerinden kaynaklanan sapmayı dengelemede kullanılır."
    ),
    "color.hsv": (
        "HSV renk uzayına çevirir; belirli bir renk aralığını maskeleyebilir, gölge giderme "
        "uygulayabilir. Aydınlatma değişse bile belirli bir rengi (ör. kırmızı kapak, yeşil PCB) "
        "güvenilir şekilde ayıklamak istediğinizde RGB'den daha kararlıdır — meyve/sebze "
        "ayrıştırma, renkli parça sıralama gibi işlerde tercih edilir."
    ),
    "segment.threshold": (
        "Görüntüyü verilen eşik değerine göre ikili (siyah/beyaz) görüntüye çevirir. "
        "Nesne/arka plan ayrımının temelidir; Bağlı Bileşenler ve Bölge Ölçümü adımlarından önce "
        "mutlaka uygulanmalıdır."
    ),
    "filter.gaussian_blur": (
        "Gauss çekirdeğiyle yumuşatarak sensör/aydınlatma gürültüsünü azaltır. Kenar tespiti veya "
        "eşiklemeden ÖNCE uygulanır — aksi halde gürültü sahte kenar/gürültülü bölge olarak algılanır."
    ),
    "filter.median_blur": (
        "Medyan filtre ile özellikle tuz-biber (nokta) gürültüsünü temizler; Gaussian'a göre "
        "kenarları daha iyi korur. Toz/kir parçacıklarının küçük beyaz noktalar olarak göründüğü "
        "kameralarda tercih edilir."
    ),
    "filter.bilateral": (
        "Kenarları koruyarak yumuşatma yapar. Yüzey kusuru tespiti gibi kenar keskinliğinin kritik "
        "olduğu ama gürültünün de azaltılması gereken durumlarda (metal/plastik yüzey) kullanılır — "
        "Gaussian'dan daha yavaş ama daha kaliteli sonuç verir."
    ),
    "filter.canny_edge": (
        "Nesnelerin dış hatlarını bulur; endüstride en yaygın kullanılan kenar tespit algoritmasıdır "
        "(hız + tutarlılık dengesi). Parça kontur çıkarma, ölçüm noktası belirleme, kusur/çatlak "
        "sınırlarını bulmada kullanılır. En iyi sonuç için önce bir bulanıklaştırma filtresi ekleyin."
    ),
    "filter.sobel_edge": (
        "Sobel gradyan tabanlı kenar tespiti. X/Y yönünü ayrı verebildiği için YÖNLÜ kusur "
        "tespitinde (ör. sadece yatay çizikler) kullanışlıdır; Canny'ye göre daha basit ve "
        "gürültüye daha açıktır."
    ),
    "filter.laplacian_edge": (
        "Laplacian (ikinci türev) tabanlı kenar tespiti; ince çatlak/çizik gibi keskin geçişlere "
        "Sobel'den daha duyarlıdır. Gürültüyü de güçlendirdiği için önce bulanıklaştırma önerilir."
    ),
    "filter.sharpen": (
        "Keskinleştirme (unsharp mask). Odak dışı/hafif bulanık kameralarda görsel netliği "
        "artırır — ölçüm hassasiyeti gereken uygulamalarda kenar tespitinden önce eklenebilir."
    ),
    "filter.histogram_equalize": (
        "Global histogram eşitleme; düşük kontrastlı (sisli/soluk aydınlatılmış) görüntülerde "
        "kontrastı otomatik artırır. Aydınlatma homojen değilse bunun yerine Siyah-Beyaz "
        "modundaki CLAHE seçeneği tercih edilmelidir."
    ),
    "morphology.erode": (
        "Beyaz bölgeleri küçültür; ince çıkıntıları ve küçük gürültü noktalarını siler. "
        "Eşiklemeden sonra kalan gürültüyü (tuz-biber benzeri küçük beyaz noktalar) temizlemek "
        "için kullanılır."
    ),
    "morphology.dilate": (
        "Beyaz bölgeleri büyütür; kopuk parçaları birleştirir. Gölge veya kesintiden dolayı "
        "parçalı görünen bir nesnenin konturunu tek parça haline getirmek için kullanılır."
    ),
    "morphology.open": (
        "Önce aşındırıp sonra yayar (erosion+dilation); nesne boyutunu büyük ölçüde korurken "
        "küçük gürültü noktalarını temizler. Eşikleme sonrası standart temizlik adımıdır."
    ),
    "morphology.close": (
        "Önce yayıp sonra aşındırır (dilation+erosion); nesne içindeki küçük boşlukları/delikleri "
        "doldurur. Bağlı Bileşenler'den önce nesnenin bütünlüğünü sağlamlaştırmak için kullanılır."
    ),
    "segment.connected_components": (
        "Birbirine bağlı beyaz piksel gruplarını (nesneleri) etiketler ve sayar. Üründe kaç "
        "parça/kusur olduğunu saymanın temel adımıdır — konveyör bandındaki nesne sayımında kullanılır."
    ),
    "analysis.region_props": (
        "Etiketlenmiş her bölge için alan, çevre, merkez, sınırlayıcı kutu ve döndürülmüş "
        "cisimlerde de doğru en/boy (minAreaRect) ölçümleri üretir; sonuçları görsel olarak da "
        "(kutu + numara + boyut etiketi) gösterir. Boyut/şekil bazlı kalite kontrolde (ör. 'alanı "
        "500 pikselden küçük parçaları ret et') kullanılır. Anlamlı sonuç için önce Bağlı "
        "Bileşenler adımını eklemeniz gerekir. Piksel Ölçeği (mm/px) parametresi doluysa (Araçlar "
        "> Aktif Yükseklik Ayarla ile otomatik doldurulur) piksel ölçümlerinin yanına gerçek dünya "
        "(mm/mm²) karşılıkları da eklenir. Tolerans Kontrolü isteğe bağlıdır: kapalıyken (varsayılan) "
        "referans girmeden tüm cisimler yeşil çerçeveyle ölçülür; açılıp kısa/uzun kenar için "
        "min/max girilirse spec içindeki cisimler yeşil, dışındakiler kırmızı çerçeve ve OK/NG "
        "etiketiyle işaretlenir."
    ),
    "roi.region": (
        "Sadece belirlediğiniz dikdörtgen ya da dairesel bölgeyi işler; geri kalanı yok sayılır "
        "(Şekil parametresiyle seçilir, önizlemede fare ile elle taşınıp yeniden boyutlandırılabilir). "
        "İşlem hızını artırmak (küçük alanda daha hızlı analiz) veya görüntünün yalnızca ilgili "
        "kısmına (ör. konveyör üzerindeki sabit inceleme penceresi ya da dairesel bir kap ağzı) "
        "odaklanmak için kullanılır."
    ),
    "geom.shape_match": (
        "Araçlar > Şekil Eşleştirme (Model Öğret) ile önceden isimle kaydedilmiş bir veya BİRDEN "
        "FAZLA modeli (ör. 'cıvata' ve 'somun' aynı anda) bu görüntü üzerinde arar; bulunan her "
        "eşleşmenin konumunu, açısını ve skorunu (x, y, alpha, score) hangi modelden geldiğiyle "
        "birlikte üretir, sonucu görsel olarak da (döndürülmüş kontur + açı ekseni + 'cıvata1' "
        "gibi etiket) gösterir; sağdaki Sonuçlar sekmesinde model başına sayım görünür. "
        "HALCON'daki find_shape_model'e yakın, piramit tabanlı kabadan-inceye arama kullanır — "
        "ürünün bant üzerinde döndüğü/kaydığı konumlandırma, hizalama kontrolü ve parça sayımında "
        "kullanılır."
    ),
    "analysis.color_props": (
        "Tüm görüntünün (önceki bir ROI adımıyla daraltılmışsa sadece o alanın) L*a*b* "
        "ortalama/std değerlerini hesaplar; opsiyonel Tolerans Kontrolü açılırsa girilen bir "
        "referans renkten ΔE (renk farkı) sapmasına göre OK/NG işaretler. Renk sapması/soluk "
        "ürün/yanlış pişme gibi tondan kaynaklanan kusurların tespitinde, tek bir ürüne "
        "odaklanmak için önce bir ROI adımı eklenmesi önerilir."
    ),
    "analysis.texture_props": (
        "Tüm görüntünün (önceki bir ROI adımıyla daraltılmışsa sadece o alanın) yüzey "
        "dokusunu gri-seviye eş-oluşum matrisi (GLCM/Haralick: contrast, homogeneity, "
        "energy, correlation) ile sayısallaştırır. Çizik/pürüzlülük/yüzey kusuru gibi renk "
        "yerine DOKU farkına dayanan kalite kontrolde kullanılır (HALCON'daki "
        "gen_cooc_matrix/cooc_feature'a benzer)."
    ),
    "ml.onnx_detect": (
        "Araçlar > ONNX Model Kaydet... ile önceden isimle kaydedilmiş bir veya BİRDEN FAZLA "
        "YOLO modelini bu görüntü üzerinde çalıştırır; bulunan her nesnenin konumunu (kutu), "
        "sınıfını ve güven skorunu hangi modelden geldiğiyle birlikte üretir, sonucu görsel "
        "olarak da (kutu + sınıf adı + skor etiketi) gösterir. Kural tabanlı Şekil "
        "Eşleştirme'nin aksine, imgflow DIŞINDA (ör. Ultralytics YOLO ile) eğitilmiş bir "
        "modeli çalıştırır — kural yazarak yakalanamayacak ince kusurlar/varyasyonlar için "
        "kullanılır. Şu an SADECE YOLO türündeki modeller çalışır; kaydederken "
        "'Sınıflandırma'/'Segmentasyon' seçilirse çalıştırıldığında net bir 'henüz "
        "desteklenmiyor' hatası verir."
    ),
}

_OP_ID_ROLE = Qt.ItemDataRole.UserRole


def _custom_filter_name(op_id: str) -> str | None:
    """`custom.*` op_id'leri derleme zamanında değil ÇALIŞMA ZAMANINDA (kullanıcı ekler/
    siler) var olduğu için statik sözlüklerde yer almazlar — adı diskteki kayıttan okunur."""
    from imgflow.core.custom_filters import op_id_for

    return next((d.name for d in list_custom_filters() if op_id_for(d.name) == op_id), None)


def category_for(op_id: str) -> str:
    if op_id.startswith(OP_ID_PREFIX):
        return "Filtreler"
    return _CATEGORY_BY_OP_ID.get(op_id, UNCATEGORIZED_LABEL)


def label_for(op_id: str) -> str:
    if op_id.startswith(OP_ID_PREFIX):
        return _custom_filter_name(op_id) or op_id
    return _LABEL_BY_OP_ID.get(op_id, op_id)


def description_for(op_id: str) -> str:
    if op_id.startswith(OP_ID_PREFIX):
        return "Kullanıcı tanımlı özel filtre (OpenCV kodu) — Filtreler panelindeki '+' ile eklenir/düzenlenir."
    return _DESCRIPTION_BY_OP_ID.get(op_id, op_id)


class OperatorLibrary(QWidget):
    operator_checked = Signal(str)
    operator_unchecked = Signal(str)
    custom_filter_editor_requested = Signal()

    def __init__(self, registry: Any, parent=None) -> None:
        super().__init__(parent)
        self.registry = registry
        self._items_by_op_id: dict[str, QTreeWidgetItem] = {}
        self._suppress_signal = False

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Ara...")
        self.search_box.textChanged.connect(self._apply_filter)

        custom_filter_button = QPushButton("+ Özel Filtre Ekle/Düzenle...")
        custom_filter_button.clicked.connect(self.custom_filter_editor_requested.emit)

        search_row = QHBoxLayout()
        search_row.addWidget(self.search_box, 1)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemChanged.connect(self._on_item_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(search_row)
        layout.addWidget(self.tree)
        layout.addWidget(custom_filter_button)

        self.refresh()

    def refresh(self) -> None:
        self.tree.clear()
        self._items_by_op_id.clear()
        by_category: dict[str, list[str]] = {}
        for op_id in self.registry.ids():
            by_category.setdefault(category_for(op_id), []).append(op_id)

        ordered_categories = [c for c in _CATEGORY_ORDER if c in by_category]
        ordered_categories += sorted(c for c in by_category if c not in _CATEGORY_ORDER)

        self._suppress_signal = True
        for category in ordered_categories:
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(category_item)
            for op_id in sorted(by_category[category]):
                op_item = QTreeWidgetItem([label_for(op_id)])
                op_item.setData(0, _OP_ID_ROLE, op_id)
                op_item.setToolTip(0, description_for(op_id))
                op_item.setFlags(op_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                op_item.setCheckState(0, Qt.CheckState.Unchecked)
                category_item.addChild(op_item)
                self._items_by_op_id[op_id] = op_item
        self._suppress_signal = False
        self.tree.expandAll()

    def set_checked(self, op_id: str, checked: bool) -> None:
        item = self._items_by_op_id.get(op_id)
        if item is None:
            return
        self._suppress_signal = True
        item.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._suppress_signal = False

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            category_item = self.tree.topLevelItem(i)
            visible_children = 0
            for j in range(category_item.childCount()):
                child = category_item.child(j)
                op_id = child.data(0, _OP_ID_ROLE) or ""
                match = needle in child.text(0).lower() or needle in op_id.lower()
                child.setHidden(not match)
                visible_children += match
            category_item.setHidden(visible_children == 0)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._suppress_signal or column != 0:
            return
        op_id = item.data(0, _OP_ID_ROLE)
        if not op_id:
            return
        if item.checkState(0) == Qt.CheckState.Checked:
            self.operator_checked.emit(op_id)
        else:
            self.operator_unchecked.emit(op_id)
