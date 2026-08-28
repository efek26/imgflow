"""HALCON'daki create_shape_model/find_shape_model mantığına yakın, piramit (çözünürlük
kademeleri) tabanlı kenar-gradyan yönü eşleştirmesi.

Asıl hız sırrı KABADAN İNCEYE arama: naif yaklaşım her açıda tam çözünürlükte tarama yapar
(çok yavaş); burada önce en kaba (en küçük) piramit seviyesinde az sayıda aday (x, y, açı)
bulunur, sonra SADECE bu adaylar bir alt (daha ince) seviyeye taşınıp dar bir pencerede
iyileştirilir — tam çözünürlükte hiçbir zaman tüm görüntü × tüm açı taranmaz.

Puanlama, model kenar noktalarının gradyan yönü ile hedef görüntüdeki gradyan yönü arasındaki
kosinüs benzerliğinin (yön uyumu) ortalamasıdır; mutlak gradyan büyüklüğü puanlamaya
katılmaz — bu, HALCON'un kontrast/aydınlatma değişimine karşı sağlam olan yön-tabanlı
eşleştirmesiyle aynı prensiptir. Model noktalarının ofset+açılarına göre küçük bir "yön
çekirdeği" kurup `cv2.filter2D` ile tüm konum adaylarını TEK SEFERDE (görüntü genelinde
vektörize) puanlamak, "her piksel için Python döngüsü" yaklaşımına göre çok daha hızlıdır.

Dönme + öteleme (x, y, alpha) yanında OPSİYONEL ölçek (scale) araması da desteklenir
(`scale_min`/`scale_max`/`scale_step_coarse`, bkz. `find_shape_model`) — varsayılan
`scale_min=scale_max=1.0` iken tek bir ölçek (1.0) taranır, davranış/performans eskisiyle
BİREBİR aynı kalır (gerçek kullanıcı isteği: "farklı boyutlarda aynı cisimden varsa scale
boyutu yazmalı" -- bant yüksekliği/kamera mesafesi kalibrasyonsuz değiştiğinde ya da fiziksel
olarak farklı boyutlu aynı ürün varyantlarında öğretilen boyuttan sapan nesneleri de bulabilmek
için).

**Overlay çizimi HALCON'un `get_generic_shape_model_result_object(..., 'contours')` deseniyle
AYNI:** bir eşleşme bulununca modelin AYRI bir siluet/kontur çıkarımı DEĞİL, EĞİTİMDE zaten
çıkarılmış ve EŞLEŞTİRMEDE zaten kullanılan GERÇEK kenar noktaları (`ShapeLevel.points`,
`levels[0]` -- tam çözünürlük) bulunan pozisyona (x,y,açı,ölçek) dönüştürülüp DOĞRUDAN
noktalar/noktacıklar olarak çizilir (bkz. `render_match_overlay`). Bu, overlay'in HERHANGİ bir
Otsu/siluet-tespitinin başarılı olmasına bağımlı olmamasını sağlar -- nokta bulutu HER ZAMAN
gerçek eğitilen kenarları yansıtır (tek/çok parçalı nesne, ters-çevrilmiş eğitim modu fark
etmez) -- gerçek kullanıcı raporu (ve paylaştığı HALCON HDevelop örneği): "sadece logonun dış
hattı çizilsin, kare içine alınmasın" sorununun KÖKÜ, overlay'in eşleştirmede kullanılandan
AYRI bir siluet-kontur kaynağından türetilmeye çalışılmasıydı, Otsu'nun kalitesi değil.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from imgflow.core.roi import RoiRect

_DEFAULT_NUM_LEVELS = 3
_AUTO_LEVEL_MIN_POINTS = 15
"""`train_shape_model(num_levels=None)` (HALCON'un `inspect_shape_model`'indeki otomatik
`NumLevels` seçimiyle AYNI mantık) iken bir sonraki (daha kaba) piramit seviyesinin nokta
sayısı bu eşiğin ALTINA düşerse o seviye modele DAHİL EDİLMEZ, kurulum durur -- HALCON'un
literal '15' alan eşiğinin bizdeki (piksel alanı değil nokta sayısı) karşılığı."""
_AUTO_LEVEL_MAX_LEVELS = 6
"""Otomatik modda denenecek üst sınır -- UI'daki "Piramit Seviyesi" spinbox'ının mevcut üst
sınırıyla (1-6) AYNI."""
_DEFAULT_MIN_GRADIENT_FRACTION = 0.2
"""Bir pikselin 'kenar noktası' sayılması için gereken gradyan büyüklüğü eşiği, ROI
içindeki MAKSİMUM büyüklüğe ORANLA (mutlak eşik değil) — böylece farklı kontrast/aydınlatma
koşullarında elle eşik ayarlamaya gerek kalmaz."""
_DEFAULT_MAX_POINTS_PER_LEVEL = 300
_DEFAULT_ANGLE_STEP_COARSE = 3.0
"""Kaba arama açı adımı — eskiden 5.0 idi; iki adım arasına düşen (ör. eski adımda tam
2.5°'lik sapma) gerçek nesneler hiç aday üretmeden kayboluyordu (gerçek kullanıcı raporu:
"nesne bulunamıyor/kaçırılıyor"). 3.0 örnekleme boşluğunu azaltır (kaba geçiş ~1.67x daha
yavaş olur ama `carry_limit` aday sayısını zaten sınırladığı için tipik pipeline
gecikmesine etkisi ihmal edilebilir)."""
_MIN_COARSE_LEVEL_POINTS = 100
"""Kaba (aday üretme) aramasının kullanacağı piramit seviyesinde bulunması gereken EN AZ
nokta sayısı; bu sayının altındaki (daha kaba) seviyeler aday üretmek için ATLANIR.

**Neden (ölçüldü):** çok kaba seviyelerde nokta sayısı düştükçe skor dağılımı düzleşir --
1280x1024'lük gürültülü (σ=8) bir sahnede 4 gerçek nesnenin kaba skoru seviye 4'te (21 nokta)
0.79-0.86 iken ARKA PLAN GÜRÜLTÜSÜNÜN 99.9 yüzdeliği 0.77'ye çıkıyordu. Eşik (0.32) her
ikisinin de altında kaldığından binlerce gürültü adayı üretiliyor, `carry_limit` budamasında
gerçek eşleşmeler listeden taşıyor ve nesne KAÇIRILIYORDU (ölçüm: 5 seviyeli modelde 0/4,
4 seviyelide 2/4, 3 seviyelide 4/4 eşleşme). Seviye başına nokta sayısı yeterli olduğunda
gürültü tabanı belirgin şekilde düşüyor (seviye 2'de 0.39).

Bu kontrol ARAMA tarafında olduğundan, DAHA ÖNCE eğitilmiş (fazla derin) modeller de yeniden
eğitilmeye gerek kalmadan düzelir. Model hiçbir seviyede bu sayıya ulaşmıyorsa (küçük/az
kenarlı nesneler) en kaba seviye yine kullanılır -- davranış eskisi gibi olur, kimse
"model çok küçük" diye eşleşmesiz kalmaz."""
_COARSE_ACCEPT_LOOSENING = 0.6
"""Kaba (en bulanık) piramit seviyesinde bir adayın üretilmesi için gereken eşiği, nihai
`min_score` kabul eşiğinden AYRIŞTIRAN çarpan. En kaba seviye `cv2.pyrDown` ile bulanıklaşmış
ve daha az kenar noktası içerdiğinden GERÇEK bir eşleşme bile bu seviyede nihai (tam
çözünürlük) skorundan sistematik olarak daha düşük skorlanır — eskiden kaba eşik doğrudan
`min_score * greediness` idi, bu da düşük-ama-gerçek adayları daha rafine olma şansı
bulamadan siliyordu (gerçek kullanıcı raporu). Nihai kabul hâlâ tam çözünürlükte `min_score`
ile yapılıyor (bkz. `find_shape_model` sonu) — bu sabit SADECE hangi adayların iyileştirmeye
taşınacağını gevşetir, nihai hassasiyeti/false-positive oranını değiştirmez."""
_REFINE_XY_RADIUS = 5
"""Kaba seviyeden bir alt (daha ince) seviyeye geçerken konum ikiye katlanır (`x*2, y*2`);
piramit örnekleme + `_local_maxima`'daki 3x3 dilatasyon nedeniyle kaba seviyedeki konum hatası
birkaç piksel olabilir, bu da 2x büyütülünce eski dar pencereyi (3px) aşıp iyileştirmenin
gerçek konuma hiç ulaşamamasına yol açabiliyordu (gerçek kullanıcı raporu: "kaçırılıyor").
Eskiden 3 idi."""
_REFINE_ANGLE_DIVISOR = 4.0
_DEFAULT_SCALE_MIN = 1.0
_DEFAULT_SCALE_MAX = 1.0
"""`scale_min == scale_max` (varsayılan) iken taranan tek ölçek değeri `scale_min`'dir --
`_score_map`'e verilen çarpan 1.0 olduğunda kernel eskisiyle BİREBİR aynı üretilir, bu yüzden
varsayılan davranış/performans hiç DEĞİŞMEZ (bkz. modül docstring'i)."""
_DEFAULT_SCALE_STEP_COARSE = 0.1
_REFINE_SCALE_DIVISOR = 2.0
_MIN_SCALE_STEP = 0.01
"""İyileştirme sırasında ölçek penceresi (`angle_window` ile AYNI desen) her seviyede
`_REFINE_SCALE_DIVISOR`'a bölünerek daralır -- bu taban, adım sıfıra çökmesin diye."""
_DEFAULT_MIN_CONTRAST = 10.0
"""HALCON'un `create_shape_model(..., MinContrast, ...)` parametresinin karşılığı (gri seviye
cinsinden). Puanlama SADECE gradyan YÖNÜNE baktığından (büyüklük normalize edilir), gradyanı
neredeyse SIFIR olan düz/gürültülü bir arka plan pikseli bile rastgele ama TAM BİRİM uzunlukta
bir yön vektörü üretir ve skora ±1 katkı verir -- yani gürültü, gerçek bir kenar kadar
"güçlü" sayılır. HALCON bu yüzden MinContrast altındaki pikselleri puanlamadan tamamen
dışlar; burada da eşiğin altındaki piksellerin cos/sin haritası SIFIRLANIR (katkısı 0 olur,
yanlış YÖNE sahip bir kenar gibi CEZALANDIRILMAZ). İki etkisi var: (1) dokulu/gürültülü
arka planda yanlış eşleşmeler belirgin azalır, (2) skor gerçekten "modelin kenarlarının yüzde
kaçı görüntüde BULUNDU" anlamına gelir, `min_score` sezgisel hale gelir."""

_SOBEL_GRADIENT_SCALE = 4.0
"""`min_contrast` gri-seviye cinsinden verilir ama karşılaştırma `cv2.Sobel(ksize=3)`
büyüklüğü üzerinde yapılır: genliği Δ olan ideal bir basamak kenarı bu çekirdekle ~4Δ
büyüklük üretir, bu yüzden eşik `min_contrast * 4`'tür."""

_OUTER_CONTOUR_BINS = 360
"""`train_shape_model(outer_contour_only=True)`: merkez etrafında kaç açısal kutuya bölünüp
her kutuda en dıştaki noktanın tutulacağı. 360 (derece başına bir örnek) tipik nesne
boyutlarında dış hattı seyrekleştirmeden temsil eder."""
_OUTER_CONTOUR_SMOOTH_BINS = 3
"""Silüet sınırı hesaplanırken her açısal kutunun kaç KOMŞU kutuyla birlikte (maksimum
alınarak) değerlendirileceği (±3 kutu = ±3°). Bkz. `_keep_outer_points`."""
_OUTER_CONTOUR_RADIUS_TOLERANCE = 0.85
"""Bir noktanın "dış hatta ait" sayılması için, kendi yönündeki silüet yarıçapının en az bu
oranı kadar uzakta olması gerekir. 1.0 (yalnızca tam en dıştaki nokta) dış hattı gereksiz
seyrekleştirirdi; 0.85 kalın/pürüzlü kenarları korurken iç yapıyı dışarıda bırakır."""
_INTERIOR_PROBE_SAMPLES = 120
"""İç bölge doğrulaması için iç/dış bölgeden örneklenen nokta sayısı (her biri). Aday başına
~240 piksel okuması demek -- vektörize `numpy` indekslemeyle maliyeti ihmal edilebilir."""
_INTERIOR_PROBE_MARGIN = 0.15
"""Dış (arka plan) halkasının, nesnenin sınırlayıcı kutusuna oranla ne kadar dışarı taşacağı."""
_MIN_PROBE_SAMPLES = 10
"""Bir adayın iç/dış ortalaması için görüntü SINIRLARI İÇİNDE kalması gereken en az örnek
sayısı; altında kalırsa ölçüm güvenilmez sayılıp doğrulama o aday için ATLANIR."""
_DEFAULT_VERIFY_TOLERANCE = 0.35
"""Aday kontrastının, eğitimdekinin en az bu KATI olması beklenir. Düşük tutulmasının
nedeni: gölge/parlama gerçek nesnenin kontrastını da bir miktar düşürür; kriterin amacı
"nesne mi değil mi" ayrımı (düz zeminde oran ~0 çıkar), ince ayar değil."""

_DIAGNOSTIC_REJECT_LIMIT = 10
"""Teşhis çıktısında raporlanacak, eşiğin altında kalmış en iyi aday sayısı."""

_NMS_DIST_THRESH = 2.0
"""Küçük modeller için ALT SINIR (piksel) — asıl kullanılan eşik `_NMS_DIST_FRACTION * model
yarıçapı`'dır, bkz. `_nms_dist_thresh_for`."""
_NMS_DIST_FRACTION = 0.25
"""Modelin sınırlayıcı yarıçapına (`corners`ın merkeze uzaklığının maksimumu) ORANLA NMS mesafe
eşiği. Sabit (piksel cinsinden) bir eşik büyük modellerde işe yaramaz: aynı fiziksel nesnenin
etrafında piramit/açı adımlarından kaynaklanan birkaç piksellik farklarla oluşan adaylar
birleştirilemez ve aynı nesne birden fazla 'eşleşme' olarak döner. 0.25 deneysel olarak hem bu
tekrarları temizliyor hem de ~1.5x model-yarıçapı kadar ayrık gerçek nesneleri yanlışlıkla
birleştirmiyor (0.4-0.5'te gerçek ayrı nesneler de birleşmeye başlıyor)."""
_CANDIDATE_CARRY_MULTIPLIER = 5
_MIN_CANDIDATES_CARRIED = 10
_AUTO_CANDIDATE_CARRY_LIMIT = 60
"""`max_matches=None` (otomatik: eşiği geçen TÜM eşleşmeleri döndür) modunda, aday budama
aşamasında kullanılan sabit güvenlik tavanı — `max_matches * _CANDIDATE_CARRY_MULTIPLIER`
hesaplanamaz çünkü hedeflenen eşleşme sayısı yok."""

_OVERLAY_COLOR = (0, 255, 0)
_REJECTED_COLOR = (0, 0, 255)
"""Teşhis modunda (bkz. `render_match_overlay`'in `rejected` argümanı) eşiğin altında kalmış
adayların rengi -- kabul edilenlerin yeşiliyle karışmasın diye kırmızı."""
_TEXT_SCALE_REFERENCE_DIM = 1000.0
"""Font/çizgi kalınlığı bu referans boyuta (px, görüntünün uzun kenarı) göre ölçeklenir —
`operators/builtin/region_props.py` `draw_measurements_overlay`'deki AYNI desen (gerçek
kullanıcı raporu: gerçek endüstriyel kamera çözünürlüğünde (ör. 1920x1200) eski SABİT
(0.45 fontScale, 1px kalınlık) yazılar okunamayacak kadar küçük kalıyordu). `max(..., taban)`
ile küçük görüntülerde (testler dahil) eski sabit değerlerle TAM aynı sonucu üretir."""


class ShapeMatchingError(Exception):
    """Model eğitimi/eşleştirme sırasında anlaşılır bir hata için (dialog'da doğrudan gösterilir)."""


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ShapeMatchingError(f"Desteklenmeyen görüntü şekli: {image.shape}")


def _normalize_angle(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


@dataclass
class ShapeLevel:
    """Bir piramit seviyesindeki kenar noktaları: model merkezine göre (x, y) ofseti ve
    o noktadaki gradyan yönü (radyan)."""

    points: np.ndarray  # (N, 2) float64 — [ofset_x, ofset_y]
    angles: np.ndarray  # (N,) float64 — radyan

    def to_dict(self) -> dict[str, Any]:
        return {"points": self.points.tolist(), "angles": self.angles.tolist()}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ShapeLevel:
        return ShapeLevel(
            points=np.array(data["points"], dtype=np.float64).reshape(-1, 2),
            angles=np.array(data["angles"], dtype=np.float64),
        )


@dataclass
class ShapeModel:
    """Eğitilmiş şekil modeli: en ince (index 0, tam çözünürlük) seviyeden en kaba
    (son eleman) seviyeye kadar `ShapeLevel` listesi, artı görselleştirme için modelin
    tam-çözünürlükteki sınırlayıcı kutu köşeleri (merkeze göre)."""

    levels: list[ShapeLevel]
    corners: np.ndarray  # (4, 2) float64 — tam çözünürlükte, merkeze göre köşe ofsetleri
    reference_center: tuple[float, float] | None = None
    """Eğitim ROI'sinin MUTLAK merkezi (referans görüntüdeki piksel konumu) — `None` ise
    (eski/`.json`'dan yüklenmiş bir model, bu alandan ÖNCE kaydedilmiş) "bilinmiyor" anlamına
    gelir, YANLIŞ bir (0,0) DEĞİL. `operators/builtin/shape_match.py::run()` bunu, bulunan
    nesnenin öğretildiği pozisyondan ne kadar ÖTELENDİĞİNİ (mesafe) hesaplamak için kullanır
    — gerçek kullanıcı isteği: "geometrik eşlemede ... cismin ötelenme uzaklığı da yazmalı"."""
    interior_points: np.ndarray | None = None
    """(M, 2) — modelin İÇİNDEN (level-0 noktalarının dışbükey zarfının içinden) örneklenmiş,
    merkeze göre ofsetler. `exterior_points`/`interior_contrast` ile birlikte "iç bölge
    doğrulaması" için kullanılır (bkz. `verify_interior`, `_measure_interior_contrast`)."""
    exterior_points: np.ndarray | None = None
    """(M, 2) — aynı zarfın DIŞINDAKİ halkadan örneklenmiş ofsetler (arka plan referansı)."""
    interior_contrast: float | None = None
    """Eğitim sırasında ölçülen, AYDINLATMADAN BAĞIMSIZ iç/dış kontrast oranı
    `(ort_iç - ort_dış) / (ort_iç + ort_dış)`. İşareti nesnenin zemine göre KOYU mu AÇIK mı
    olduğunu taşır; bu yüzden "koyu nesne/açık zemin" gibi sahneye özel bir varsayım YOK --
    her iki yön de kendi işaretiyle doğru çalışır. `None` ise (bu alandan ÖNCE kaydedilmiş
    modeller) doğrulama sessizce ATLANIR."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": [lvl.to_dict() for lvl in self.levels],
            "corners": self.corners.tolist(),
            "reference_center": list(self.reference_center) if self.reference_center is not None else None,
            "interior_points": self.interior_points.tolist() if self.interior_points is not None else None,
            "exterior_points": self.exterior_points.tolist() if self.exterior_points is not None else None,
            "interior_contrast": self.interior_contrast,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ShapeModel:
        # Eski (bu özellikten ÖNCE kaydedilmiş) `.json` dosyalarında hâlâ bir "contour" anahtarı
        # olabilir -- burada HİÇ okunmuyor (dataclass'ta artık alan yok), fazladan anahtar
        # sessizce YOKSAYILIR, ÇÖKME yok (bkz. modül docstring'i: overlay artık `levels[0].
        # points`'i kullanıyor, eski modeller YENİDEN EĞİTİLMEDEN bundan otomatik faydalanır).
        reference_center_data = data.get("reference_center")

        def _points(key: str) -> np.ndarray | None:
            raw = data.get(key)
            if not raw:
                return None
            return np.array(raw, dtype=np.float64).reshape(-1, 2)

        return ShapeModel(
            levels=[ShapeLevel.from_dict(d) for d in data["levels"]],
            corners=np.array(data["corners"], dtype=np.float64).reshape(-1, 2),
            reference_center=tuple(reference_center_data) if reference_center_data else None,
            interior_points=_points("interior_points"),
            exterior_points=_points("exterior_points"),
            interior_contrast=data.get("interior_contrast"),
        )

    def supports_interior_verification(self) -> bool:
        return (
            self.interior_points is not None
            and self.exterior_points is not None
            and self.interior_contrast is not None
            and self.interior_points.size > 0
            and self.exterior_points.size > 0
        )


@dataclass(frozen=True)
class MatchResult:
    x: float
    y: float
    angle: float
    """Derece cinsinden, [-180, 180) aralığına normalize edilmiş."""
    score: float
    """[0, 1] aralığına yakın, 1'e ne kadar yakınsa yön uyumu o kadar iyi (kosinüs benzerliği ortalaması)."""
    scale: float = 1.0
    """Öğretilen modele göre bulunan nesnenin ölçeği (1.0 = öğretildiği boyutta). Ölçek araması
    kapalıyken (`find_shape_model`'in varsayılan `scale_min=scale_max=1.0`'ı) HER ZAMAN 1.0'dır
    -- gerçek kullanıcı isteği: "farklı boyutlarda aynı cisimden varsa scale boyutu yazmalı"."""


def _extract_level_points(
    image: np.ndarray,
    roi: RoiRect,
    cx: float,
    cy: float,
    min_gradient_fraction: float,
    max_points: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`mask` (verilmişse): `image` ile AYNI boyutta bool dizi, True olmayan pikseller
    gradyan eşiğini geçse bile modele ALINMAZ — "Konturu Otomatik Algıla"/elle poligon
    çizim yollarının ROI dikdörtgeni içindeki arka planı/gürültüyü elemesi için."""
    row0, row1 = max(0, roi.y), min(image.shape[0], roi.y + roi.h)
    col0, col1 = max(0, roi.x), min(image.shape[1], roi.x + roi.w)
    if row1 <= row0 or col1 <= col0:
        return np.empty((0, 2)), np.empty((0,))

    patch = image[row0:row1, col0:col1].astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    max_mag = float(mag.max()) if mag.size else 0.0
    if max_mag <= 0.0:
        return np.empty((0, 2)), np.empty((0,))

    point_mask = mag >= (min_gradient_fraction * max_mag)
    if mask is not None:
        point_mask = point_mask & mask[row0:row1, col0:col1]
    ys, xs = np.nonzero(point_mask)
    if ys.size == 0:
        return np.empty((0, 2)), np.empty((0,))

    angles = np.arctan2(gy[ys, xs], gx[ys, xs]).astype(np.float64)
    abs_x = xs.astype(np.float64) + col0
    abs_y = ys.astype(np.float64) + row0
    points = np.stack([abs_x - cx, abs_y - cy], axis=1)

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
        points = points[idx]
        angles = angles[idx]
    return points, angles


def train_shape_model(
    reference_image: np.ndarray,
    roi: RoiRect,
    num_levels: int | None = _DEFAULT_NUM_LEVELS,
    min_gradient_fraction: float = _DEFAULT_MIN_GRADIENT_FRACTION,
    max_points_per_level: int = _DEFAULT_MAX_POINTS_PER_LEVEL,
    mask: np.ndarray | None = None,
    outer_contour_only: bool = False,
) -> ShapeModel:
    """`mask` (opsiyonel): `reference_image` ile AYNI (yükseklik, genişlik) boyutunda bool
    dizi — True olan pikseller model noktası olabilir, False olanlar ROI içinde bile olsa
    ELENİR. `None` ise davranış eskisiyle BİREBİR aynıdır (tüm ROI dikdörtgeni kullanılır).
    "Konturu Otomatik Algıla" (auto_objects tabanlı) ve elle çizilen serbest kontur/poligon/
    çember ikisi de bu tek mekanizmayı kullanır — eşleştirme/serileştirme kodu (ShapeModel/
    find_shape_model) sadece düz nokta listesi gördüğünden bundan habersizdir. Bu noktalar
    AYNI ZAMANDA `render_match_overlay`'in çizdiği "dış hat"tır (bkz. modül docstring'i,
    HALCON'daki `get_generic_shape_model_result_object` deseni) -- yani `mask` artık sadece
    eşleştirme kalitesini değil, overlay'in GÖRÜNÜMÜNÜ de doğrudan belirler.

    `num_levels` (opsiyonel): piramit derinliği. `None` verilirse HALCON'un `inspect_shape_
    model`'indeki gibi OTOMATİK seçilir -- seviye seviye kurulur, bir sonraki (daha kaba)
    seviyenin nokta sayısı `_AUTO_LEVEL_MIN_POINTS`'in altına düşer düşmez o seviye modele
    DAHİL EDİLMEDEN durulur (en fazla `_AUTO_LEVEL_MAX_LEVELS`'e kadar dener). Tam çözünürlük
    (ilk, `level==0`) seviye HER ZAMAN dahil edilmeye çalışılır (0 nokta bulunursa hâlâ hata
    verir). Bir TAM SAYI verilirse (varsayılan/manuel davranış) DAVRANIŞ DEĞİŞMEZ.

    `outer_contour_only` (opsiyonel): her seviyede SADECE en dıştaki kenar noktaları tutulur
    (bkz. `_keep_outer_points`). Nesnenin İÇ yapısı (logo, kabartma, parlama/gölgeyle
    kaybolabilen iç halka) modele hiç girmez -- gerçek kullanıcı sorunu: aynı üründen
    bazıları bulunurken bazıları bulunamıyordu, çünkü modelin ~%22'si iç halkadaydı ve o
    halka gölgede kalan ürünlerde görünmüyordu (skor eşiğin hemen altına düşüyordu).
    Filtre tamamen geometriktir -- daire, yazı, dişli, herhangi bir şekil için çalışır."""
    gray = _to_gray(reference_image)
    roi = roi.clamp(gray.shape[1], gray.shape[0])
    if roi.w < 4 or roi.h < 4:
        raise ShapeMatchingError("ROI çok küçük (en az 4x4 piksel gerekli).")
    if mask is not None and mask.shape != gray.shape:
        raise ShapeMatchingError("Kontur maskesi boyutu referans görüntüyle uyuşmuyor.")

    cx = roi.x + roi.w / 2.0
    cy = roi.y + roi.h / 2.0
    auto_levels = num_levels is None
    max_levels = _AUTO_LEVEL_MAX_LEVELS if auto_levels else num_levels

    levels: list[ShapeLevel] = []
    level_image = gray
    level_roi = roi
    level_cx, level_cy = cx, cy
    level_mask = mask
    for level in range(max_levels):
        points, angles = _extract_level_points(
            level_image,
            level_roi,
            level_cx,
            level_cy,
            min_gradient_fraction,
            max_points_per_level,
            mask=level_mask,
        )
        if points.shape[0] == 0:
            if auto_levels and level > 0:
                break  # bu seviye ATLANIR (dahil edilmez), önceki seviyelerle yetinilir
            reason = (
                "; kontur/maskeyi genişletin ya da 'Konturu Otomatik Algıla'/poligon "
                "seçeneğini kapatıp tüm ROI'yi kullanın"
                if mask is not None
                else ""
            )
            raise ShapeMatchingError(
                f"Piramit seviyesi {level}'de yeterli kenar noktası bulunamadı; ROI'de daha "
                f"belirgin bir kenar/kontur olmalı ya da 'Min. Gradyan Oranı' düşürülmeli{reason}."
            )
        if outer_contour_only:
            points, angles = _keep_outer_points(points, angles)
        if auto_levels and level > 0 and points.shape[0] < _AUTO_LEVEL_MIN_POINTS:
            break  # bu seviye çok seyrek -- HALCON'daki gibi modele DAHİL EDİLMEZ, durulur
        levels.append(ShapeLevel(points=points, angles=angles))
        if level < max_levels - 1:
            level_image = cv2.pyrDown(level_image)
            level_cx /= 2.0
            level_cy /= 2.0
            level_roi = RoiRect(
                x=int(level_roi.x // 2),
                y=int(level_roi.y // 2),
                w=max(1, level_roi.w // 2),
                h=max(1, level_roi.h // 2),
            )
            if level_mask is not None:
                level_mask = cv2.resize(
                    level_mask.astype(np.uint8),
                    (level_image.shape[1], level_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

    half_w, half_h = roi.w / 2.0, roi.h / 2.0
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]], dtype=np.float64
    )
    interior, exterior, contrast = _build_interior_probe(gray, levels[0].points, cx, cy)
    return ShapeModel(
        levels=levels,
        corners=corners,
        reference_center=(cx, cy),
        interior_points=interior,
        exterior_points=exterior,
        interior_contrast=contrast,
    )


def _keep_outer_points(points: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merkez etrafındaki her açısal kutuda SADECE en büyük yarıçaplı noktayı tutar.

    Nesnenin dış hattı, iç yapısının aksine aydınlatmadan/parlamadan bağımsız olarak
    kararlıdır; iç kenarlar ise sahneye göre görünüp kaybolabilir ve modelin o oranda
    puanını düşürür. Filtre şekil varsayımı YAPMAZ (dairesellik vb. aranmaz), sadece
    "merkezden bakınca en dıştaki nokta" kuralını uygular.

    **Bilinen ödünleşim:** DERİN girintili şekillerde (yıldız/dişli gibi, girinti yarıçapı
    dış yarıçapın %85'inin altına inen) girintideki MEŞRU dış-hat noktaları da elenir --
    o yöndeki sınır, komşu çıkıntıların yarıçapından hesaplanır. Böyle şekillerde bu
    seçenek KAPALI bırakılmalıdır; asıl hedefi "iç yapıyı (logo/kabartma/halka) at" olan
    dışbükeye yakın ürünlerdir."""
    if points.shape[0] == 0:
        return points, angles
    radius = np.linalg.norm(points, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    bins = np.clip(
        np.floor((theta + np.pi) / (2 * np.pi) * _OUTER_CONTOUR_BINS).astype(int),
        0,
        _OUTER_CONTOUR_BINS - 1,
    )

    # Her açısal kutudaki EN BÜYÜK yarıçap = o yöndeki silüet sınırı.
    boundary = np.full(_OUTER_CONTOUR_BINS, -np.inf)
    np.maximum.at(boundary, bins, radius)

    # Kutu başına maksimum TEK BAŞINA yetmiyor: dış hattın hiç örnek düşürmediği bir kutuda
    # SADECE iç yapıya ait bir nokta varsa, o nokta kendi kutusunun "sınırı" olup kendini
    # geçerli sayıyordu (ölçüldü: iç kareli test nesnesinde noktaların %17'si hâlâ içeriden
    # geliyordu). Bu yüzden sınır, KOMŞU kutuların maksimumuyla birlikte değerlendirilir --
    # penceresi dar tutulur ki yıldız/dişli gibi gerçekten girintili şekillerde girintideki
    # meşru dış-hat noktaları elenmesin.
    window = _OUTER_CONTOUR_SMOOTH_BINS
    padded = np.concatenate([boundary[-window:], boundary, boundary[:window]])
    boundary = np.array(
        [padded[i : i + 2 * window + 1].max() for i in range(_OUTER_CONTOUR_BINS)]
    )

    # Hâlâ tamamen BOŞ kalan kutular (yakınında hiç örnek yok) komşulardan interpolasyonla
    # doldurulur.
    known = np.isfinite(boundary)
    if not known.any():
        return points, angles
    idx = np.arange(_OUTER_CONTOUR_BINS)
    # Dairesel (2*pi'de kapanan) interpolasyon: bilinen kutular üç kopya halinde uzatılır.
    known_idx = idx[known]
    known_val = boundary[known]
    extended_idx = np.concatenate(
        [known_idx - _OUTER_CONTOUR_BINS, known_idx, known_idx + _OUTER_CONTOUR_BINS]
    )
    extended_val = np.concatenate([known_val, known_val, known_val])
    boundary = np.interp(idx, extended_idx, extended_val)

    keep = radius >= _OUTER_CONTOUR_RADIUS_TOLERANCE * boundary[bins]
    if not keep.any():
        return points, angles
    return points[keep], angles[keep]


def _build_interior_probe(
    gray: np.ndarray, points: np.ndarray, cx: float, cy: float
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """İç bölge doğrulaması için örnekleme noktalarını ve eğitimdeki kontrast oranını üretir.

    Model noktalarının DIŞBÜKEY ZARFI kullanılır: bu, şekilden bağımsız genel bir "nesnenin
    kapladığı alan" tanımıdır (daire, çok parçalı yazı, dişli -- hepsinde çalışır). İç
    örnekler zarfın içinden, dış örnekler zarfın `_INTERIOR_PROBE_MARGIN` oranında
    büyütülmüş halinin DIŞINDA kalan halkadan alınır. Ölçüt, mutlak gri seviye DEĞİL
    aydınlatmadan bağımsız `(iç-dış)/(iç+dış)` oranıdır."""
    if points.shape[0] < 3:
        return None, None, None
    hull = cv2.convexHull((points + np.array([cx, cy])).astype(np.float32))
    if hull is None or len(hull) < 3:
        return None, None, None

    x0, y0, w, h = cv2.boundingRect(hull)
    pad = int(max(w, h) * _INTERIOR_PROBE_MARGIN) + 2
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1 = min(gray.shape[1], x0 + w + 2 * pad)
    y1 = min(gray.shape[0], y0 + h + 2 * pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None, None, None

    inside = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillConvexPoly(inside, (hull - np.array([[x0, y0]], dtype=np.float32)).astype(np.int32), 1)
    grown = cv2.dilate(inside, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8))
    shrunk = cv2.erode(inside, np.ones((3, 3), np.uint8))
    ring = (grown > 0) & (inside == 0)

    interior = _sample_offsets(shrunk > 0, x0, y0, cx, cy)
    exterior = _sample_offsets(ring, x0, y0, cx, cy)
    if interior is None or exterior is None:
        return None, None, None

    gray_f = gray.astype(np.float32)
    inner_mean = float(gray_f[y0:y1, x0:x1][shrunk > 0].mean())
    outer_mean = float(gray_f[y0:y1, x0:x1][ring].mean())
    return interior, exterior, _contrast_ratio(inner_mean, outer_mean)


def _sample_offsets(
    mask: np.ndarray, x0: int, y0: int, cx: float, cy: float
) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    if ys.size > _INTERIOR_PROBE_SAMPLES:
        idx = np.linspace(0, ys.size - 1, _INTERIOR_PROBE_SAMPLES).astype(int)
        ys, xs = ys[idx], xs[idx]
    return np.stack([xs + x0 - cx, ys + y0 - cy], axis=1).astype(np.float64)


def _contrast_ratio(inner_mean: float, outer_mean: float) -> float:
    """Michelson benzeri, aydınlatma kazancından BAĞIMSIZ oran. Görüntünün tamamı 2x
    parlarsa hem pay hem payda 2x olur, oran DEĞİŞMEZ -- bu yüzden eğitimdeki ve aramadaki
    aydınlatma farklı olsa bile karşılaştırılabilir."""
    denominator = abs(inner_mean) + abs(outer_mean)
    if denominator < 1e-6:
        return 0.0
    return (inner_mean - outer_mean) / denominator


def _rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def _build_direction_kernels(
    points: np.ndarray, angles: np.ndarray, angle_offset_deg: float, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """`scale` model noktalarının merkeze göre ofsetlerini büyütür/küçültür (üniform ölçek) --
    gradyan YÖNLERİNİ etkilemez (bir kenarın açısı ölçekle değişmez, sadece konumu). Bu, hedef
    görüntüde öğretilen boyuttan FARKLI boyuttaki bir nesneyi arayabilmenin çekirdek mekanizması
    (bkz. `find_shape_model`'in `scale_min`/`scale_max`'ı) -- `scale=1.0` (varsayılan) eskisiyle
    BİREBİR aynı çekirdeği üretir."""
    rotated = _rotate_points(points, angle_offset_deg) * scale
    rotated_angles = angles + np.radians(angle_offset_deg)
    offsets = np.round(rotated).astype(int)

    min_dx, min_dy = offsets.min(axis=0)
    max_dx, max_dy = offsets.max(axis=0)
    # Ofset (0,0) -- model MERKEZİNİN kendisi -- çekirdeğin sınırlarına HER ZAMAN dahil edilmeli:
    # `anchor` (aşağıda `center_col`/`center_row`) "ofset 0,0 çekirdekte hangi hücreye denk
    # geliyor" anlamına gelir ve `cv2.filter2D`'e geçirilir; OpenCV anchor'ın çekirdek sınırları
    # İÇİNDE olmasını ZORUNLU kılar. Eskiden sınırlar SADECE gerçek noktaların min/max'ından
    # hesaplanıyordu -- eğer (belirli bir döndürme açısında) TÜM noktalar merkeze göre kesinlikle
    # aynı taraftaysa (ör. `min_dx > 0`), 0 aralığa hiç GİRMİYORDU ve `center_col = -min_dx`
    # NEGATİF (sınırın DIŞINDA) çıkıp `cv2.filter2D` "anchor.inside(...)" OpenCV assertion'ıyla
    # ÇÖKÜYORDU -- gerçek kullanıcı raporu: elle çizilen (Poligon) bir kontur gibi ROI'nin naif
    # merkezine göre ASİMETRİK nokta bulutlarında (özellikle en kaba piramit seviyesinde az nokta
    # kalınca) 360° tam açı taramasının bir noktasında bu koşul oluşup node'u çökertiyordu. Sentetik
    # test şekilleri her zaman ROI merkezine göre dengeli seçildiğinden bu daha önce hiç
    # YAKALANMAMIŞTI.
    min_dx, max_dx = min(min_dx, 0), max(max_dx, 0)
    min_dy, max_dy = min(min_dy, 0), max(max_dy, 0)
    width = int(max_dx - min_dx + 1)
    height = int(max_dy - min_dy + 1)
    center_col = int(-min_dx)
    center_row = int(-min_dy)

    kernel_cos = np.zeros((height, width), dtype=np.float32)
    kernel_sin = np.zeros((height, width), dtype=np.float32)
    rows = offsets[:, 1] + center_row
    cols = offsets[:, 0] + center_col
    np.add.at(kernel_cos, (rows, cols), np.cos(rotated_angles).astype(np.float32))
    np.add.at(kernel_sin, (rows, cols), np.sin(rotated_angles).astype(np.float32))
    return kernel_cos, kernel_sin, (center_col, center_row)


_MAX_KERNEL_CACHE_ENTRIES = 2000
"""Bir `ShapeLevel`'ın açı-çekirdek önbelleğindeki üst sınır — kaba arama sabit bir açı
ızgarası kullandığı için (bkz. `_kernel_cache_for_level` docstring'i) pratikte birkaç yüz
girdiyi asla aşmaz; bu sadece savunma amaçlı bir tavan (aşılırsa önbellek sıfırlanır)."""


def _kernel_cache_for_level(
    level: ShapeLevel,
) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    """`level` nesnesine bağlı, çalışma zamanında oluşturulan bir önbellek — `ShapeModel`/
    `ShapeLevel` dataclass ALANI DEĞİL (bilerek): `to_dict()`/eşitlik karşılaştırması bu
    fazladan niteliğe hiç dokunmaz. `geom.shape_match` operatörü artık `shape_model_store`
    sayesinde (bkz. o modüldeki mtime-tabanlı önbellek) canlı kamerada tick'ler arasında AYNI
    `ShapeModel`/`ShapeLevel` nesnelerini yeniden kullandığından, bu önbellek de tick'ler
    arasında KALICI olur — kaba arama SABİT bir açı ızgarası taradığı için (`angle_step_
    coarse`) aynı yön çekirdekleri eskiden HER tick'te sıfırdan kuruluyordu, artık sadece
    İLK tick'te."""
    cache = getattr(level, "_kernel_cache", None)
    if cache is None:
        cache = {}
        level._kernel_cache = cache
    return cache


def _score_map(
    cos_map: np.ndarray,
    sin_map: np.ndarray,
    level: ShapeLevel,
    angle_offset_deg: float,
    scale: float = 1.0,
    ignore_polarity: bool = False,
) -> np.ndarray:
    """`ignore_polarity`, HALCON'un `create_shape_model(..., Metric='ignore_global_polarity')`
    karşılığıdır: skor haritasının MUTLAK değeri alınır. Kontrast TÜMÜYLE ters döndüğünde
    (koyu bant üzerinde açık ürün <-> açık bant üzerinde koyu ürün; ıslak bant, farklı ürün
    rengi, arkadan aydınlatma) her model noktasının gradyan yönü tam 180° döner ve skor +1
    yerine -1'e gider -- yani nesne MÜKEMMEL eşleştiği halde tamamen kaçırılır. Mutlak değer
    bu iki durumu tek bir eşleşme olarak kabul eder. HALCON'un 'ignore_local_polarity'si
    (her kenarın AYRI AYRI ters dönebilmesi) bilinçli olarak KAPSAM DIŞI -- o, nokta başına
    mutlak değer gerektirir ve tek bir `filter2D` ile hesaplanamaz (nesne başına yüzlerce
    ayrı korelasyon demektir)."""
    cache = _kernel_cache_for_level(level)
    key = (round(angle_offset_deg, 6), round(scale, 6))
    cached = cache.get(key)
    if cached is None:
        if len(cache) >= _MAX_KERNEL_CACHE_ENTRIES:
            cache.clear()
        cached = _build_direction_kernels(level.points, level.angles, angle_offset_deg, scale)
        cache[key] = cached
    kernel_cos, kernel_sin, anchor = cached
    dst_cos = cv2.filter2D(cos_map, -1, kernel_cos, anchor=anchor, borderType=cv2.BORDER_CONSTANT)
    dst_sin = cv2.filter2D(sin_map, -1, kernel_sin, anchor=anchor, borderType=cv2.BORDER_CONSTANT)
    result = (dst_cos + dst_sin) / float(level.points.shape[0])
    return np.abs(result) if ignore_polarity else result


def _sparse_cache_for_level(
    level: ShapeLevel,
) -> dict[tuple[float, float], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """`_kernel_cache_for_level` ile AYNI desendeki ikinci bir çalışma-zamanı önbelleği, ama
    YOĞUN çekirdek yerine `_score_window`'un kullandığı SEYREK biçimi (ofsetler + ağırlıklar)
    tutar."""
    cache = getattr(level, "_sparse_kernel_cache", None)
    if cache is None:
        cache = {}
        level._sparse_kernel_cache = cache
    return cache


def _sparse_kernel(
    level: ShapeLevel, angle_offset_deg: float, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`_build_direction_kernels`'in ürettiği yoğun çekirdeği `(ox, oy, w_cos, w_sin)` seyrek
    biçimine çevirir: `dst(x,y) = Σ_i cos_map(y+oy_i, x+ox_i)*w_cos_i + sin_map(...)*w_sin_i`
    -- `cv2.filter2D`'in (KORELASYON, konvolüsyon değil) anchor'lı davranışının BİREBİR aynısı.
    Çekirdek hücresine düşen birden fazla model noktası zaten `np.add.at` ile toplandığından
    seyrek biçim de aynı toplamı taşır; hem cos hem sin ağırlığı tam olarak 0 olan hücreler
    (matematiksel katkısı 0) atlanır."""
    cache = _sparse_cache_for_level(level)
    key = (round(angle_offset_deg, 6), round(scale, 6))
    cached = cache.get(key)
    if cached is None:
        if len(cache) >= _MAX_KERNEL_CACHE_ENTRIES:
            cache.clear()
        kernel_cos, kernel_sin, anchor = _build_direction_kernels(
            level.points, level.angles, angle_offset_deg, scale
        )
        mask = (kernel_cos != 0) | (kernel_sin != 0)
        rows, cols = np.nonzero(mask)
        cached = (
            (cols - anchor[0]).astype(np.intp),
            (rows - anchor[1]).astype(np.intp),
            kernel_cos[mask],
            kernel_sin[mask],
        )
        cache[key] = cached
    return cached


def _score_window(
    cos_map: np.ndarray,
    sin_map: np.ndarray,
    level: ShapeLevel,
    angle_offset_deg: float,
    scale: float,
    ignore_polarity: bool,
    y_lo: int,
    y_hi: int,
    x_lo: int,
    x_hi: int,
) -> np.ndarray:
    """`_score_map` ile AYNI skoru, ama SADECE `[y_lo,y_hi) x [x_lo,x_hi)` penceresi için
    hesaplar (sayısal olarak özdeş, ölçüldü: maks fark ~1e-18).

    Neden: `_refine_candidate` bir adayı iyileştirirken skor haritasının yalnızca ±`xy_radius`
    kutusunu okur, ama `cv2.filter2D` çıktı boyutunu girdi boyutuna EŞİTLER ve girdinin model
    ayak izini de kapsaması ZORUNLUDUR -- yani tam çözünürlük seviyesinde ~280x280'lik bir
    skor haritası hesaplanıp içinden 11x11'lik bir kutu okunuyordu. Seyrek/doğrudan hesap
    sadece gereken konumları üretir (ölçüldü, 300 noktalı model: 4.11 ms -> 0.27 ms; ~15x).
    Kaba arama (`_coarse_search`) TÜM görüntüyü taradığı için orada hâlâ `_score_map`/filter2D
    kullanılır -- orada seyrek biçim daha yavaş olurdu."""
    ox, oy, w_cos, w_sin = _sparse_kernel(level, angle_offset_deg, scale)
    height, width = cos_map.shape
    rows = np.arange(y_lo, y_hi, dtype=np.intp)[:, None, None] + oy[None, None, :]
    cols = np.arange(x_lo, x_hi, dtype=np.intp)[None, :, None] + ox[None, None, :]

    if rows.min() >= 0 and rows.max() < height and cols.min() >= 0 and cols.max() < width:
        # Pencerenin model ayak iziyle birlikte TAMAMI görüntünün içinde (tipik durum) --
        # kırpma/maskeleme adımları atlanır.
        acc = cos_map[rows, cols] * w_cos + sin_map[rows, cols] * w_sin
    else:
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        clipped_rows = np.clip(rows, 0, height - 1)
        clipped_cols = np.clip(cols, 0, width - 1)
        acc = cos_map[clipped_rows, clipped_cols] * w_cos + sin_map[clipped_rows, clipped_cols] * w_sin
        acc[~inside] = 0.0  # `cv2.BORDER_CONSTANT` (0) ile aynı sınır davranışı
    result = acc.sum(axis=2) / float(level.points.shape[0])
    return np.abs(result) if ignore_polarity else result


def _local_maxima(score_map: np.ndarray, threshold: float, neighborhood: int = 3) -> list[list[float]]:
    dilated = cv2.dilate(score_map, np.ones((neighborhood, neighborhood), dtype=np.uint8))
    mask = (score_map >= threshold) & (score_map >= dilated - 1e-6)
    ys, xs = np.nonzero(mask)
    return [[float(x), float(y), float(score_map[y, x])] for x, y in zip(xs, ys)]


def _nms(
    candidates: list[list[float]],
    dist_thresh: float,
    box_size: tuple[float, float] | None = None,
    max_overlap: float | None = None,
) -> list[list[float]]:
    """Konum tabanlı (yalnızca x/y) non-max suppression: `dist_thresh` içindeki adaylardan
    sadece en yüksek skorlu tutulur.

    `max_overlap` VE `box_size` birlikte verilirse bunun yerine HALCON'un `MaxOverlap`
    kuralı uygulanır (bkz. `_nms_by_overlap`); verilmezse (varsayılan) aşağıdaki mesafe
    tabanlı davranış BİREBİR korunur.

    Açı KASITLI OLARAK bu karşılaştırmaya dahil edilmiyor — eskiden hem konum hem açı yakınlığı
    birlikte aranıyordu, ama kaba/ince arama adımlarının ürettiği küçük açı sapmalarıyla AYNI
    fiziksel nesne için birden fazla aday (ör. 45° ve 48°'de, aynı x/y'de) farklı açı eşiğini
    aşıp 'ayrı nesne' sayılıyor, aynı cisim tekrar tekrar sonuçta görünüyordu (gerçek kullanıcı
    raporu: "aynı cismi farklı açılardan tespit edip tekrar tekrar gösteriyor"). Rijit bir
    nesnenin TEK bir gerçek açısı vardır; konum tek başına yeterli bir ayraçtır — iki GERÇEKTEN
    farklı nesnenin merkezinin `dist_thresh` (modelin yarıçapına göre ölçeklenir, bkz.
    `_nms_dist_thresh_for`) kadar yakın olması pratikte imkansızdır. Bu değişiklik ayrıca
    performansı da iyileştirir: kaba arama aşamasında aynı nesne için üretilen fazladan
    adaylar artık daha agresif birleştiği için, pahalı çok-seviyeli iyileştirme aşamasına
    (`_refine_candidate`) daha AZ aday taşınır.

    **Performans (mekansal ızgara):** gerçek kullanıcı raporu "şekil bul çok kasıyor" ile
    profillenirken, YOĞUN/gürültülü bir arka planda (gerçek bir kamera karesi gibi) kaba
    aramanın `min_score`'u gevşeten `_COARSE_ACCEPT_LOOSENING` eşiği (bkz. yukarısı --
    KASITLI OLARAK dokunulmadı, "kaçırma" regresyonuna yol açabilir) binlerce (ör. 8000+) ham
    aday üretebiliyor -- eski düz `for cand in sorted(...): any(... for r in result)` yaklaşımı
    HER adayı `result`'taki TÜM önceki kabul edilmiş adaylarla (insertion sırasına göre)
    karşılaştırıyordu; bu, `result` büyüdükçe pahalılaşan bir Python-seviyesi döngüydü (ölçülen:
    8505 adayda ~15ms, TEK bir `find_shape_model` çağrısının toplam süresinin önemli bir
    kısmı). Şimdi adaylar `dist_thresh` boyutunda hücrelere ızgaralanıyor -- her aday sadece
    kendi hücresi + 8 komşu hücredeki (aynı `dist_thresh` kuralına göre) kabul edilmiş
    adaylarla karşılaştırılıyor, TÜM `result` listesiyle DEĞİL. Karar mantığı/sıralaması
    BİREBİR AYNI (aynı en-yüksek-skor-önce sıralama, aynı `abs(dx)<=thresh and abs(dy)<=thresh`
    kuralı) -- SADECE hangi noktalarla karşılaştırıldığı ızgarayla daraltılıyor, bu yüzden
    sonuç kümesi HER durumda eskisiyle özdeş (aynı sentetik aday kümesiyle doğrulandı). Ölçülen
    kazanç: 8505 adayda ~15ms → ~7ms (~2.2x); tipik temiz sahnede (10-20 aday) ızgara
    ek yükü ihmal edilebilir (~5 mikrosaniye fark)."""
    if not candidates:
        return []
    if max_overlap is not None and box_size is not None:
        return _nms_by_overlap(candidates, box_size, max_overlap)
    cell = dist_thresh if dist_thresh > 0 else 1.0
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    result: list[list[float]] = []
    for cand in sorted(candidates, key=lambda c: -c[4]):
        x, y = cand[0], cand[1]
        cell_x, cell_y = int(x // cell), int(y // cell)
        if any(
            abs(x - rx) <= dist_thresh and abs(y - ry) <= dist_thresh
            for gx in (cell_x - 1, cell_x, cell_x + 1)
            for gy in (cell_y - 1, cell_y, cell_y + 1)
            for rx, ry in grid.get((gx, gy), ())
        ):
            continue
        result.append(cand)
        grid.setdefault((cell_x, cell_y), []).append((x, y))
    return result


def _nms_by_overlap(
    candidates: list[list[float]], box_size: tuple[float, float], max_overlap: float
) -> list[list[float]]:
    """HALCON'un `find_shape_model(..., MaxOverlap, ...)` karşılığı: iki eşleşmenin
    (eksene paralel) sınırlayıcı kutularının örtüşme oranı `max_overlap`'i AŞIYORSA düşük
    skorlu olan elenir.

    Sabit mesafe eşiğinden (`_nms`'in varsayılan yolu, modelin yarıçapının `_NMS_DIST_FRACTION`
    katı) farkı: eşik modelin GERÇEK boyutuna ve eşleşmenin ÖLÇEĞİNE göre hesaplandığı için
    yan yana duran/temas eden iki AYRI ürün (ör. bantta bitişik iki paket) artık tek eşleşmeye
    indirgenmiyor -- kullanıcı `max_overlap` ile "ne kadar üst üste binebilirler" kararını
    kendisi veriyor. Örtüşme, HALCON gibi KÜÇÜK olan kutunun alanına oranlanır (IoU değil):
    küçük bir eşleşme büyük bir eşleşmenin İÇİNDE kalıyorsa oran 1.0 olur ve elenir.

    Izgara hızlandırması `_nms`'inkiyle AYNI mantıkta, sadece hücre boyu kutu boyutudur."""
    box_w, box_h = max(box_size[0], 1e-6), max(box_size[1], 1e-6)
    cell = max(box_w, box_h)
    grid: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    result: list[list[float]] = []
    for cand in sorted(candidates, key=lambda c: -c[4]):
        x, y, scale = cand[0], cand[1], cand[3]
        w, h = box_w * scale, box_h * scale
        cell_x, cell_y = int(x // cell), int(y // cell)
        suppressed = False
        for gx in (cell_x - 1, cell_x, cell_x + 1):
            for gy in (cell_y - 1, cell_y, cell_y + 1):
                for rx, ry, r_scale in grid.get((gx, gy), ()):
                    rw, rh = box_w * r_scale, box_h * r_scale
                    inter_w = min(x + w / 2, rx + rw / 2) - max(x - w / 2, rx - rw / 2)
                    inter_h = min(y + h / 2, ry + rh / 2) - max(y - h / 2, ry - rh / 2)
                    if inter_w <= 0 or inter_h <= 0:
                        continue
                    smaller_area = min(w * h, rw * rh)
                    if smaller_area > 0 and (inter_w * inter_h) / smaller_area > max_overlap:
                        suppressed = True
                        break
                if suppressed:
                    break
            if suppressed:
                break
        if suppressed:
            continue
        result.append(cand)
        grid.setdefault((cell_x, cell_y), []).append((x, y, scale))
    return result


def _frange(lo: float, hi: float, step: float) -> list[float]:
    """`lo`'dan `hi`'a (dahil) EŞİT aralıklı örnekler -- `step` sadece kaç örnek alınacağını
    belirler (`round((hi-lo)/step)`), üretilen adım bu sayıya göre YENİDEN hesaplanır ki `hi`
    uç noktası her zaman tam olarak örneklensin (`np.arange`'in kayan nokta birikim hatasıyla
    uç noktayı atlayabilmesinin aksine)."""
    if step <= 0 or hi <= lo:
        return [lo]
    n = max(1, int(round((hi - lo) / step)))
    return [lo + i * (hi - lo) / n for i in range(n + 1)]


def _coarse_search(
    cos_map: np.ndarray,
    sin_map: np.ndarray,
    level: ShapeLevel,
    angle_start: float,
    angle_extent: float,
    angle_step: float,
    relaxed_min: float,
    scale_values: list[float],
    ignore_polarity: bool = False,
) -> list[list[float]]:
    candidates: list[list[float]] = []
    n_steps = max(1, int(round(angle_extent / angle_step)))
    for scale in scale_values:
        for i in range(n_steps + 1):
            angle = angle_start + i * angle_step
            if angle > angle_start + angle_extent + 1e-6:
                break
            score_map = _score_map(cos_map, sin_map, level, angle, scale, ignore_polarity)
            for x, y, score in _local_maxima(score_map, relaxed_min):
                candidates.append([x, y, angle, scale, score])
    return candidates


def _refine_candidate(
    cos_map: np.ndarray,
    sin_map: np.ndarray,
    level: ShapeLevel,
    x0: float,
    y0: float,
    angle0: float,
    scale0: float,
    xy_radius: int,
    angle_window: float,
    angle_step: float,
    scale_window: float = 0.0,
    scale_step: float = 0.0,
    ignore_polarity: bool = False,
    subpixel: bool = False,
) -> list[float]:
    """`scale_window <= 0` (ölçek araması kapalı, varsayılan) iken SADECE `scale0` denenir --
    eskisiyle BİREBİR aynı maliyet/davranış. Açıksa (bkz. `find_shape_model`'in `scale_min`/
    `scale_max`'ı) `angle_window`'un narrowlanma deseninin AYNISI ölçek için de uygulanır: her
    piramit seviyesinde pencere `_REFINE_SCALE_DIVISOR`'a bölünerek daralır.

    Skor, eskiden model ayak izini de kapsayan BÜYÜK bir kırpım üzerinde `_score_map`
    (`cv2.filter2D`) ile hesaplanıp içinden ±`xy_radius` kutusu okunuyordu; artık `_score_
    window` doğrudan sadece o kutuyu üretiyor (sayısal olarak ÖZDEŞ, bkz. orada ki not)."""
    h, w = cos_map.shape
    best = [float(x0), float(y0), float(angle0), float(scale0), -1.0]

    # Okunan konum kutusu (mutlak koordinat) -- eski kod bunu kırpımın içinde
    # `int(local_0 ∓ xy_radius)` ile hesaplıyordu; kırpımın başlangıcı tam sayı olduğundan
    # sonuç birebir aynıdır.
    ry_lo, ry_hi = max(0, int(y0 - xy_radius)), min(h, int(y0 + xy_radius) + 1)
    rx_lo, rx_hi = max(0, int(x0 - xy_radius)), min(w, int(x0 + xy_radius) + 1)
    if ry_hi <= ry_lo or rx_hi <= rx_lo:
        return best
    # Subpiksel parabolü tepe noktasının KOMŞULARINI ister -- pencere her yönde 1 piksel
    # geniş hesaplanır (eski büyük kırpımda bu doğal olarak sağlanıyordu); görüntü kenarında
    # kırpılır, orada parabol zaten 0 döner (`_parabolic_peak_offset`).
    wy_lo, wy_hi = max(0, ry_lo - 1), min(h, ry_hi + 1)
    wx_lo, wx_hi = max(0, rx_lo - 1), min(w, rx_hi + 1)

    scale_values = (
        [scale0] if scale_window <= 0 else _frange(scale0 - scale_window, scale0 + scale_window, scale_step)
    )

    # Subpiksel/subadım iyileştirme için: en iyi sonucu veren skor haritası ve o haritadaki
    # (yerel) tepe konumu, artı HER açı için o açının en iyi skoru (açı parabolü için).
    best_patch: np.ndarray | None = None
    best_peak: tuple[int, int] = (0, 0)
    angle_scores: list[tuple[float, float]] = []

    angle = angle0 - angle_window
    end_angle = angle0 + angle_window + 1e-6
    while angle <= end_angle:
        angle_best = -1.0
        for scale in scale_values:
            window = _score_window(
                cos_map, sin_map, level, angle, scale, ignore_polarity, wy_lo, wy_hi, wx_lo, wx_hi
            )
            sub = window[ry_lo - wy_lo : ry_hi - wy_lo, rx_lo - wx_lo : rx_hi - wx_lo]
            idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
            score = float(sub[idx])
            angle_best = max(angle_best, score)
            if score > best[4]:
                best = [
                    float(rx_lo + idx[1]),
                    float(ry_lo + idx[0]),
                    float(angle),
                    float(scale),
                    score,
                ]
                best_patch = window
                best_peak = (ry_lo - wy_lo + int(idx[0]), rx_lo - wx_lo + int(idx[1]))
        if angle_best >= 0.0:
            angle_scores.append((float(angle), angle_best))
        angle += angle_step

    if subpixel and best_patch is not None:
        dx, dy = _parabolic_peak_offset(best_patch, best_peak)
        best[0] += dx
        best[1] += dy
        best[2] += _parabolic_angle_offset(angle_scores, best[2])
    return best


def _parabola_vertex(left: float, center: float, right: float) -> float:
    """Eşit aralıklı üç örneğe parabol uydurup tepe noktasının merkeze göre (-1..+1 arası)
    kaymasını döndürür. Ayrık bir maksimum, gerçek (sürekli) tepe noktasının en fazla yarım
    örnek uzağındadır; bu klasik 3 nokta interpolasyonu HALCON'un `SubPixel='interpolation'`
    modunun yaptığı işin aynısıdır."""
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    return float(np.clip(offset, -1.0, 1.0))


def _parabolic_peak_offset(patch: np.ndarray, peak: tuple[int, int]) -> tuple[float, float]:
    """Skor haritasındaki tam sayı tepe noktası etrafında x ve y eksenlerinde ayrı ayrı
    parabol uydurur. Kenarda (komşusu olmayan) bir tepe için ilgili eksende 0 döner."""
    py, px = peak
    h, w = patch.shape
    dx = 0.0
    dy = 0.0
    if 0 < px < w - 1:
        dx = _parabola_vertex(float(patch[py, px - 1]), float(patch[py, px]), float(patch[py, px + 1]))
    if 0 < py < h - 1:
        dy = _parabola_vertex(float(patch[py - 1, px]), float(patch[py, px]), float(patch[py + 1, px]))
    return dx, dy


def _parabolic_angle_offset(angle_scores: list[tuple[float, float]], best_angle: float) -> float:
    """Denenen açıların en iyi skorlarına, en iyi açının etrafında parabol uydurur -- açıyı
    arama ızgarasının adımından daha ince çözer (konum subpikseliyle AYNI mantık)."""
    if len(angle_scores) < 3:
        return 0.0
    index = min(range(len(angle_scores)), key=lambda i: abs(angle_scores[i][0] - best_angle))
    if index == 0 or index == len(angle_scores) - 1:
        return 0.0
    step = angle_scores[index + 1][0] - angle_scores[index][0]
    offset = _parabola_vertex(
        angle_scores[index - 1][1], angle_scores[index][1], angle_scores[index + 1][1]
    )
    return offset * step


def _measure_interior_contrast(
    gray: np.ndarray, model: ShapeModel, x: float, y: float, angle_deg: float, scale: float
) -> float | None:
    """Bir ADAY pozda, arama görüntüsünde iç/dış kontrast oranını ölçer (eğitimdekiyle AYNI
    tanım). Örnekleme noktaları aday poza (döndürme + ölçek + öteleme) taşınır; görüntü
    dışına düşenler atılır. Yeterli örnek kalmazsa `None` döner (doğrulama yapılamaz ->
    aday elenmez, çünkü bilgi yokluğu bir RED gerekçesi değildir)."""
    if not model.supports_interior_verification():
        return None
    h, w = gray.shape[:2]

    def _mean_at(offsets: np.ndarray) -> float | None:
        pts = _rotate_points(offsets * scale, angle_deg) + np.array([x, y])
        cols = np.rint(pts[:, 0]).astype(int)
        rows = np.rint(pts[:, 1]).astype(int)
        valid = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        if valid.sum() < _MIN_PROBE_SAMPLES:
            return None
        return float(gray[rows[valid], cols[valid]].mean())

    inner = _mean_at(model.interior_points)
    outer = _mean_at(model.exterior_points)
    if inner is None or outer is None:
        return None
    return _contrast_ratio(inner, outer)


def _passes_interior_verification(
    gray: np.ndarray, model: ShapeModel, candidate: list[float], tolerance: float
) -> bool:
    """Skordan BAĞIMSIZ ikinci kabul kriteri (HALCON'un clutter/kontrol bölgesi mantığının
    genel karşılığı). Gerçek kullanıcı raporu: "güven faktörünü düşürünce başka yerleri
    seçiyor" -- eşiği düşürmek gerçek nesneleri getirirken düz/gürültülü zeminde de
    eşleşmeler üretiyordu. Bu kriter, aday konumun gerçekten nesnenin ÖĞRETİLDİĞİ
    iç/dış kontrast ilişkisine sahip olmasını arar: düz beyaz kağıt üzerindeki bir aday,
    iç ve dış bölgesi aynı parlaklıkta olduğu için (oran ~0) elenir.

    Model bu bilgiyi taşımıyorsa (eski `.json`) ya da ölçüm yapılamıyorsa aday KABUL edilir --
    doğrulama yalnızca elemek için vardır, asla yeni eşleşme uydurmaz."""
    measured = _measure_interior_contrast(
        gray, model, candidate[0], candidate[1], candidate[2], candidate[3]
    )
    if measured is None or model.interior_contrast is None:
        return True
    expected = model.interior_contrast
    if abs(expected) < 1e-6:
        # Eğitimde iç/dış farkı zaten yoktu -- ayırt edici bilgi yok, eleme yapma.
        return True
    # İŞARET eğitimdekiyle aynı olmalı (koyu nesne koyu, açık nesne açık kalmalı) VE
    # büyüklük eğitimdekinin en az `tolerance` katı olmalı.
    if measured * expected <= 0:
        return False
    return abs(measured) >= tolerance * abs(expected)


def _nms_dist_thresh_for(model: ShapeModel) -> float:
    model_radius = float(np.linalg.norm(model.corners, axis=1).max()) if model.corners.size else 0.0
    return max(_NMS_DIST_THRESH, _NMS_DIST_FRACTION * model_radius)


def _build_target_pyramid(
    gray: np.ndarray, num_levels: int, min_contrast: float = _DEFAULT_MIN_CONTRAST
) -> list[tuple[np.ndarray, np.ndarray]]:
    target_levels: list[tuple[np.ndarray, np.ndarray]] = []
    level_image = gray
    mag_threshold = max(0.0, min_contrast) * _SOBEL_GRADIENT_SCALE
    for level in range(num_levels):
        gx = cv2.Sobel(level_image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(level_image, cv2.CV_32F, 0, 1, ksize=3)
        angle_map = np.arctan2(gy, gx)
        cos_map = np.cos(angle_map).astype(np.float32)
        sin_map = np.sin(angle_map).astype(np.float32)
        if mag_threshold > 0.0:
            # HALCON'un MinContrast'ı (bkz. `_DEFAULT_MIN_CONTRAST`): eşiğin altındaki
            # pikseller puanlamaya HİÇ katılmaz (0 katkı) -- yanlış yönlü bir kenar gibi
            # NEGATİF puan almazlar, sadece görünmez olurlar.
            weak = cv2.magnitude(gx, gy) < mag_threshold
            cos_map[weak] = 0.0
            sin_map[weak] = 0.0
        target_levels.append((cos_map, sin_map))
        if level < num_levels - 1:
            level_image = cv2.pyrDown(level_image)
    return target_levels


def build_search_pyramid(
    search_image: np.ndarray, num_levels: int, min_contrast: float = _DEFAULT_MIN_CONTRAST
) -> list[tuple[np.ndarray, np.ndarray]]:
    """`find_shape_model`'in `target_pyramid` argümanına verilebilecek, arama görüntüsünün
    gradyan-yön haritalarının piramidini kurar. AYNI arama görüntüsü üzerinde BİRDEN FAZLA
    model aranırken (`geom.shape_match`'in `model_names` çoklu-seçimi, gerçek kullanıcı
    raporu: "şekil bulma çok kasıyor") her model için bunu SIFIRDAN yeniden hesaplamak
    gereksizdi -- piramit SADECE arama görüntüsüne bağlıdır, hangi modelin arandığından
    bağımsızdır. Çağıran taraf TÜM modeller arasındaki EN DERİN piramidi
    (`max(len(model.levels) for model in models)`) BİR KEZ kurup her modele geçebilir; daha
    az seviyeli bir model için `target_pyramid[:len(model.levels)]` dilimlemesi yeterlidir --
    piramidin İLK N seviyesi, toplamda kaç seviye hesaplandığından BAĞIMSIZ olarak aynıdır
    (her seviye SADECE bir öncekinden `pyrDown` ile türetilir, sonraki seviyeler önceki
    seviyeleri hiç ETKİLEMEZ)."""
    gray = _to_gray(search_image).astype(np.float32)
    return _build_target_pyramid(gray, num_levels, min_contrast)


def coarse_search_level(model: ShapeModel, last_level: int = 0) -> int:
    """Aday üretiminin (kaba aramanın) yapılacağı piramit seviyesi.

    En kaba seviyeden başlanır ama nokta sayısı `_MIN_COARSE_LEVEL_POINTS`'in altındaki
    seviyeler ATLANIR (bkz. o sabitin gerekçesi: aşırı kaba seviyelerde gürültü tabanı
    gerçek eşleşmelerin skoruna yaklaşıp onları aday listesinden taşırıyor ve nesne
    kaçırılıyordu). Hiçbir seviye eşiği geçmezse `last_level` döner -- yani küçük/az kenarlı
    modellerde davranış eskisi gibi kalır.

    `operators/builtin/shape_match.py` bunu, birden fazla model için PAYLAŞILAN piramidi
    gereğinden derin kurmamak için de kullanır."""
    level = len(model.levels) - 1
    while level > last_level and model.levels[level].points.shape[0] < _MIN_COARSE_LEVEL_POINTS:
        level -= 1
    return level


def find_shape_model(
    search_image: np.ndarray,
    model: ShapeModel,
    *,
    angle_start: float = -180.0,
    angle_extent: float = 360.0,
    angle_step_coarse: float = _DEFAULT_ANGLE_STEP_COARSE,
    min_score: float = 0.5,
    max_matches: int | None = 1,
    greediness: float = 0.9,
    target_pyramid: list[tuple[np.ndarray, np.ndarray]] | None = None,
    scale_min: float = _DEFAULT_SCALE_MIN,
    scale_max: float = _DEFAULT_SCALE_MAX,
    scale_step_coarse: float = _DEFAULT_SCALE_STEP_COARSE,
    min_contrast: float = _DEFAULT_MIN_CONTRAST,
    ignore_polarity: bool = False,
    subpixel: bool = True,
    last_level: int = 0,
    max_overlap: float | None = None,
    verify_interior: bool = False,
    verify_tolerance: float = _DEFAULT_VERIFY_TOLERANCE,
    diagnostics: dict[str, Any] | None = None,
) -> list[MatchResult]:
    """`max_matches=None` ise (otomatik mod) `min_score` üstündeki TÜM eşleşmeler döndürülür
    (dahili bir güvenlik tavanına kadar, bkz. `_AUTO_CANDIDATE_CARRY_LIMIT`); bir tam sayı
    verilirse en fazla o kadar (en yüksek skorlu) eşleşme döndürülür.

    `target_pyramid` verilirse (bkz. `build_search_pyramid`) ve `model.levels` kadar (ya da
    daha fazla) seviye içeriyorsa, arama görüntüsünün gradyan piramidi YENİDEN HESAPLANMAZ --
    birden fazla modelin AYNI karede arandığı `geom.shape_match` gibi çağıranlar için önemli
    bir hız kazancı (bkz. `build_search_pyramid` docstring'i). Verilmezse (varsayılan, TEK
    modelli eski davranış) her zamanki gibi SIFIRDAN hesaplanır.

    `scale_min`/`scale_max`/`scale_step_coarse`: opsiyonel ölçek-toleranslı arama -- gerçek
    kullanıcı isteği: "farklı boyutlarda aynı cisimden varsa scale boyutu yazmalı" (bant
    yüksekliği/kamera mesafesi kalibrasyonsuz değiştiğinde ya da fiziksel olarak farklı
    boyutlu aynı ürün varyantlarında öğretilen boyuttan sapan nesneleri de bulabilmek için).
    `scale_max <= scale_min` (varsayılan, `1.0`/`1.0`) iken SADECE `scale_min` (=1.0) taranır
    -- eskisiyle BİREBİR aynı maliyet/davranış (kaba aramada tek ölçek döngüsü, iyileştirmede
    `scale_window=0`). `scale_max > scale_min` ise kaba aramada `scale_step_coarse` adımıyla
    `[scale_min, scale_max]` aralığı taranır (her ek ölçek, kaba arama maliyetini AYNI oranda
    çarpar -- kullanıcı geniş bir aralık seçerse yavaşlar), sonra `angle_window`'un narrowlanma
    deseninin AYNISıyla (`_REFINE_SCALE_DIVISOR`) her piramit seviyesinde daraltılarak
    iyileştirilir. `MatchResult.scale` bulunan nesnenin öğretilen modele göre ölçeğidir (1.0 =
    öğretildiği boyutta).

    HALCON kökenli dört ek parametre (hepsi geriye dönük güvenli varsayılanlarla):
    - `min_contrast` (HALCON `create_shape_model` MinContrast): bkz. `_DEFAULT_MIN_CONTRAST`.
      SADECE `target_pyramid` verilmediğinde burada uygulanır -- hazır bir piramit geçilirse
      eşik zaten onu kuran `build_search_pyramid` çağrısında uygulanmıştır.
    - `ignore_polarity` (HALCON Metric='ignore_global_polarity'): bkz. `_score_map`.
    - `subpixel` (HALCON SubPixel='interpolation'): son seviyede skor haritasına parabol
      uydurarak konumu ve açıyı ızgara adımından daha ince çözer. Maliyeti ihmal edilebilir
      (eşleşme başına birkaç aritmetik işlem), bu yüzden varsayılan AÇIK.
    - `last_level` (HALCON `find_shape_model`'in NumLevels'ının "son seviye" bileşeni):
      iyileştirme bu piramit seviyesinde DURUR (0 = tam çözünürlük, varsayılan). 1 vermek
      iyileştirmeyi yaklaşık 4 kat ucuzlatır (bir üst seviyede dörtte bir piksel var) ama
      konum çözünürlüğü de o kadar kabalaşır -- sonuçlar yine tam çözünürlük koordinatlarına
      ölçeklenerek döndürülür.
    - `max_overlap` (HALCON MaxOverlap): bkz. `_nms_by_overlap`. `None` (varsayılan) iken
      eski mesafe tabanlı bastırma korunur.

    `verify_interior`/`verify_tolerance`: skordan BAĞIMSIZ ikinci kabul kriteri, bkz.
    `_passes_interior_verification`. Modelin bu bilgiyi taşımadığı (eski `.json`) durumda
    sessizce ATLANIR.

    `diagnostics` verilirse (bir sözlük) doldurulur: `best_score` (eşiğin altında kalsa
    bile en iyi aday skoru), `rejected` (en iyi N reddedilmiş aday, her biri
    `{"x","y","angle","scale","score","reason"}` — `reason` "skor" ya da "dogrulama"),
    `verified` (doğrulamanın gerçekten UYGULANIP uygulanmadığı). Kullanıcı "neden
    bulunamadı"yı görebilsin diye: eşiği körlemesine düşürmek yerine gerçek nesnelerle
    yanlış pozitifler arasında ayıran bir eşik VAR MI, görerek karar verilir."""
    num_levels = len(model.levels)
    last_level = max(0, min(last_level, num_levels - 1))
    nms_dist_thresh = _nms_dist_thresh_for(model)
    if model.corners.size:
        box_size = (
            float(model.corners[:, 0].max() - model.corners[:, 0].min()),
            float(model.corners[:, 1].max() - model.corners[:, 1].min()),
        )
    else:
        box_size = None
    scale_values = [scale_min] if scale_max <= scale_min else _frange(scale_min, scale_max, scale_step_coarse)
    scale_window = scale_step_coarse if scale_max > scale_min else 0.0

    # Kaba arama seviyesi SADECE modele bağlı olduğundan piramitten ÖNCE hesaplanabilir --
    # böylece kullanılmayacak (çok kaba) seviyeler hiç kurulmaz (ölçüldü: 5 seviyeli bir
    # modelde gereksiz iki seviyenin kurulması ~85ms).
    coarsest = coarse_search_level(model, last_level)
    needed_levels = coarsest + 1

    if target_pyramid is not None and len(target_pyramid) >= needed_levels:
        target_levels = target_pyramid[:needed_levels]
    else:
        # `gray` SADECE bu dalda gerekli -- `target_pyramid` çağıran tarafından (operatörün
        # `geom.shape_match::run()`'ı, BİRDEN FAZLA model seçiliyken her model için AYNI
        # pyramid'i paylaşır) zaten yeterli derinlikte sağlandığında, tam çözünürlüklü
        # gri tonlama+float32 dönüşümünü HER model için tekrar tekrar yapmak gereksizdi --
        # gerçek kullanıcı raporu: "şekil bul çok kasıyor" (bkz. `build_search_pyramid`
        # docstring'i, AYNI kökten gelen bir önceki perf düzeltmesiyle aynı gerekçe).
        gray = _to_gray(search_image).astype(np.float32)
        target_levels = _build_target_pyramid(gray, needed_levels, min_contrast)

    cos_map, sin_map = target_levels[coarsest]
    relaxed_min = min_score * max(greediness, 1e-6) * _COARSE_ACCEPT_LOOSENING

    candidates = _coarse_search(
        cos_map, sin_map, model.levels[coarsest], angle_start, angle_extent, angle_step_coarse, relaxed_min,
        scale_values, ignore_polarity,
    )
    candidates = _nms(candidates, nms_dist_thresh)
    candidates.sort(key=lambda c: -c[4])
    carry_limit = (
        _AUTO_CANDIDATE_CARRY_LIMIT
        if max_matches is None
        else max(max_matches * _CANDIDATE_CARRY_MULTIPLIER, _MIN_CANDIDATES_CARRIED)
    )
    candidates = candidates[:carry_limit]

    angle_window = angle_step_coarse
    for level in range(coarsest - 1, last_level - 1, -1):
        cos_map, sin_map = target_levels[level]
        angle_step = max(angle_window / _REFINE_ANGLE_DIVISOR, 0.1)
        scale_step = max(scale_window / _REFINE_SCALE_DIVISOR, _MIN_SCALE_STEP) if scale_window > 0 else 0.0
        candidates = [
            _refine_candidate(
                cos_map, sin_map, model.levels[level], x * 2, y * 2, angle, scale,
                xy_radius=_REFINE_XY_RADIUS, angle_window=angle_window, angle_step=angle_step,
                scale_window=scale_window, scale_step=scale_step,
                ignore_polarity=ignore_polarity,
                # Subpiksel SADECE en son (artık daha ince olmayacak) seviyede uygulanır --
                # ara seviyelerde uygulamak anlamsız olurdu (sonuç zaten `*2` ile bir sonraki
                # seviyeye taşınıp yeniden aranıyor).
                subpixel=subpixel and level == last_level,
            )
            for x, y, angle, scale, _score in candidates
        ]
        angle_window = max(angle_window / _REFINE_ANGLE_DIVISOR, 0.5)
        if scale_window > 0:
            scale_window = max(scale_window / _REFINE_SCALE_DIVISOR, _MIN_SCALE_STEP)

    # `last_level > 0` iken koordinatlar hâlâ o seviyenin (küçültülmüş) ızgarasında --
    # bastırma eşikleri, doğrulama ve döndürülen sonuçlar TAM ÇÖZÜNÜRLÜK cinsindendir, bu
    # yüzden ELEMEDEN ÖNCE ölçekle geri taşınırlar (doğrulama tam çözünürlüklü gri görüntüde
    # örnekleme yapıyor).
    level_scale = float(2**last_level)
    if level_scale != 1.0:
        candidates = [[x * level_scale, y * level_scale, a, s, sc] for x, y, a, s, sc in candidates]

    above_threshold = [c for c in candidates if c[4] >= min_score]
    rejected: list[tuple[list[float], str]] = [(c, "skor") for c in candidates if c[4] < min_score]

    verification_applied = verify_interior and model.supports_interior_verification()
    if verification_applied:
        verify_gray = _to_gray(search_image).astype(np.float32)
        kept: list[list[float]] = []
        for candidate in above_threshold:
            if _passes_interior_verification(verify_gray, model, candidate, verify_tolerance):
                kept.append(candidate)
            else:
                rejected.append((candidate, "dogrulama"))
        above_threshold = kept

    accepted = _nms(above_threshold, nms_dist_thresh, box_size, max_overlap)
    accepted.sort(key=lambda c: -c[4])
    if max_matches is not None:
        accepted = accepted[:max_matches]

    if diagnostics is not None:
        best_score = max((c[4] for c in candidates), default=None)
        rejected.sort(key=lambda item: -item[0][4])
        diagnostics["best_score"] = best_score
        diagnostics["verified"] = verification_applied
        diagnostics["rejected"] = [
            {
                "x": c[0],
                "y": c[1],
                "angle": _normalize_angle(c[2]),
                "scale": c[3],
                "score": c[4],
                "reason": reason,
            }
            for c, reason in rejected[:_DIAGNOSTIC_REJECT_LIMIT]
        ]

    return [
        MatchResult(x=x, y=y, angle=_normalize_angle(angle), scale=scale, score=score)
        for x, y, angle, scale, score in accepted
    ]


@dataclass(frozen=True)
class LabeledMatch:
    """Bir eşleşmeyi, hangi modelden geldiğini (çizim için köşe/eksen bilgisi modelden farklı
    olabileceğinden) ve gösterilecek etiketi (ör. 'cıvata1') bir arada taşır — birden fazla
    modelin eşleşmelerini TEK bir overlay'de, kendi doğru köşe geometrileriyle çizebilmek için."""

    label: str
    model: ShapeModel
    match: MatchResult


def render_match_overlay(
    search_image: np.ndarray,
    entries: list[LabeledMatch],
    rejected: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Her eşleşmeyi kontur+eksen+SADECE numarasıyla (`entry.label`, ör. "1", "2") işaretler
    — tam (x,y,alpha) değerleri artık burada DEĞİL, ayrı bir tabloda (bkz.
    `ui/widgets/measurements_summary.py`) numaraya göre listelenir (gerçek kullanıcı isteği:
    "her şekili 1,2,3,4 diye adlandıralım, sonra 1'in x,y,alpha değerleri... tabloya yazsın").
    Görüntü üzerinde SADECE numara olması hem okunabilirliği artırır (daha az metin = daha
    büyük yazılabilir) hem de kalabalık sahnelerde çakışmayı azaltır.

    Eşleştirme puanlaması içeride hep GRİ tonlama üzerinden çalışır (bkz. `_to_gray`), ama
    overlay'in TABANI olarak burada da gri kullanmak orijinal renkli görüntüyü/kamerayı
    kaybediyordu (gerçek kullanıcı raporu: "kamerayı siyah-beyaza çeviriyor") -- `ml.onnx_
    detect`'in `render_detection_overlay`'i ile AYNI desen: girdi zaten renkliyse (3 kanal)
    OLDUĞU GİBİ kullan, SADECE tek kanallıysa (2 boyutlu) BGR'ye çevir.

    `rejected` (opsiyonel, bkz. `find_shape_model`'in `diagnostics`'i): eşiğin altında
    kalmış ya da iç bölge doğrulamasından geçememiş adaylar KIRMIZI bir çarpı + skoruyla
    çizilir. Kabul edilen eşleşmelerle karışmaması için kontur noktaları çizilmez, sadece
    konum işaretlenir. Amaç: kullanıcı "neden bulunamadı"yı GÖREREK anlasın -- gerçek
    kullanıcı raporu: "güven faktörünü düşürünce başka yerleri seçiyor" (eşiği körlemesine
    düşürmek yerine gerçek nesnelerle yanlış adayların skorları KARŞILAŞTIRILABİLİR olmalı)."""
    overlay = np.ascontiguousarray(search_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    contour_thickness = max(2, round(2 * scale_factor))
    point_radius = max(1, round(1.5 * scale_factor))
    marker_size = max(12, round(12 * scale_factor))
    font_scale = max(0.6, 0.9 * scale_factor)
    text_thickness = max(1, round(2 * scale_factor))

    for entry in entries:
        model, match, label = entry.model, entry.match, entry.label
        # HALCON'un `get_generic_shape_model_result_object(..., 'contours')` deseniyle AYNI:
        # ayrı bir Otsu/siluet-kontur çıkarımı YOK -- modelin EĞİTİMDE zaten çıkardığı ve
        # EŞLEŞTİRMEDE zaten kullandığı GERÇEK kenar noktaları (`levels[0]`, tam çözünürlük)
        # bulunan pozisyona (x,y,açı,ölçek) dönüştürülüp DOĞRUDAN noktacıklar olarak çizilir.
        # Bu, overlay'in HERHANGİ bir siluet-tespitinin başarılı olmasına bağımlı olmamasını
        # sağlar -- nokta bulutu HER ZAMAN gerçek eğitilen kenarları yansıtır (tek/çok parçalı
        # nesne, ters-çevrilmiş eğitim modu fark etmez) -- gerçek kullanıcı raporu: "sadece
        # logonun dış hattı çizilsin, kare içine alınmasın".
        axis_len = max(float(np.abs(model.corners).max()) * match.scale, 10.0)
        theta = np.radians(match.angle)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        points = ((model.levels[0].points * match.scale) @ rot.T) + np.array([match.x, match.y])
        for px, py in points:
            cv2.circle(overlay, (int(round(px)), int(round(py))), point_radius, _OVERLAY_COLOR, -1)

        center = (int(round(match.x)), int(round(match.y)))
        cv2.drawMarker(
            overlay, center, _OVERLAY_COLOR, markerType=cv2.MARKER_CROSS,
            markerSize=marker_size, thickness=contour_thickness,
        )
        tip = (int(round(match.x + axis_len * c)), int(round(match.y + axis_len * s)))
        cv2.line(overlay, center, tip, _OVERLAY_COLOR, contour_thickness)
        cv2.putText(
            overlay,
            label,
            (center[0] + 8, max(12, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _OVERLAY_COLOR,
            text_thickness,
            cv2.LINE_AA,
        )

    for candidate in rejected or []:
        center = (int(round(candidate["x"])), int(round(candidate["y"])))
        cv2.drawMarker(
            overlay, center, _REJECTED_COLOR, markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=marker_size, thickness=contour_thickness,
        )
        cv2.putText(
            overlay,
            f"{candidate['score']:.2f}",
            (center[0] + 8, max(12, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _REJECTED_COLOR,
            text_thickness,
            cv2.LINE_AA,
        )
    return overlay
