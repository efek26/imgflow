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

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": [lvl.to_dict() for lvl in self.levels],
            "corners": self.corners.tolist(),
            "reference_center": list(self.reference_center) if self.reference_center is not None else None,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ShapeModel:
        # Eski (bu özellikten ÖNCE kaydedilmiş) `.json` dosyalarında hâlâ bir "contour" anahtarı
        # olabilir -- burada HİÇ okunmuyor (dataclass'ta artık alan yok), fazladan anahtar
        # sessizce YOKSAYILIR, ÇÖKME yok (bkz. modül docstring'i: overlay artık `levels[0].
        # points`'i kullanıyor, eski modeller YENİDEN EĞİTİLMEDEN bundan otomatik faydalanır).
        reference_center_data = data.get("reference_center")
        return ShapeModel(
            levels=[ShapeLevel.from_dict(d) for d in data["levels"]],
            corners=np.array(data["corners"], dtype=np.float64).reshape(-1, 2),
            reference_center=tuple(reference_center_data) if reference_center_data else None,
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
    verir). Bir TAM SAYI verilirse (varsayılan/manuel davranış) DAVRANIŞ DEĞİŞMEZ."""
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
    return ShapeModel(levels=levels, corners=corners, reference_center=(cx, cy))


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
    cos_map: np.ndarray, sin_map: np.ndarray, level: ShapeLevel, angle_offset_deg: float, scale: float = 1.0
) -> np.ndarray:
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
    return (dst_cos + dst_sin) / float(level.points.shape[0])


def _local_maxima(score_map: np.ndarray, threshold: float, neighborhood: int = 3) -> list[list[float]]:
    dilated = cv2.dilate(score_map, np.ones((neighborhood, neighborhood), dtype=np.uint8))
    mask = (score_map >= threshold) & (score_map >= dilated - 1e-6)
    ys, xs = np.nonzero(mask)
    return [[float(x), float(y), float(score_map[y, x])] for x, y in zip(xs, ys)]


def _nms(candidates: list[list[float]], dist_thresh: float) -> list[list[float]]:
    """Konum tabanlı (yalnızca x/y) non-max suppression: `dist_thresh` içindeki adaylardan
    sadece en yüksek skorlu tutulur.

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
) -> list[list[float]]:
    candidates: list[list[float]] = []
    n_steps = max(1, int(round(angle_extent / angle_step)))
    for scale in scale_values:
        for i in range(n_steps + 1):
            angle = angle_start + i * angle_step
            if angle > angle_start + angle_extent + 1e-6:
                break
            score_map = _score_map(cos_map, sin_map, level, angle, scale)
            for x, y, score in _local_maxima(score_map, relaxed_min):
                candidates.append([x, y, angle, scale, score])
    return candidates


def _model_extent(level: ShapeLevel) -> float:
    """Model noktalarının merkeze göre en büyük Öklid uzaklığı — döndürme mesafeyi
    korur, bu yüzden HERHANGİ bir açıda çekirdeğin kaplayacağı yarıçapın güvenli bir üst
    sınırıdır (kırpılmış yamanın çekirdekten küçük kalıp kenar etkisiyle skoru bozmasını
    önlemek için `_refine_candidate`'ın payını (pad) buna göre ayarlaması gerekir)."""
    if level.points.size == 0:
        return 0.0
    return float(np.linalg.norm(level.points, axis=1).max())


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
) -> list[float]:
    """`scale_window <= 0` (ölçek araması kapalı, varsayılan) iken SADECE `scale0` denenir --
    eskisiyle BİREBİR aynı maliyet/davranış. Açıksa (bkz. `find_shape_model`'in `scale_min`/
    `scale_max`'ı) `angle_window`'un narrowlanma deseninin AYNISI ölçek için de uygulanır: her
    piramit seviyesinde pencere `_REFINE_SCALE_DIVISOR`'a bölünerek daralır."""
    h, w = cos_map.shape
    max_scale = max(scale0 + scale_window, scale0, 1.0) if scale_window > 0 else max(scale0, 1.0)
    pad = xy_radius + int(np.ceil(_model_extent(level) * max_scale)) + 2
    y_lo, y_hi = max(0, int(y0 - pad)), min(h, int(y0 + pad) + 1)
    x_lo, x_hi = max(0, int(x0 - pad)), min(w, int(x0 + pad) + 1)
    best = [float(x0), float(y0), float(angle0), float(scale0), -1.0]
    if y_hi <= y_lo or x_hi <= x_lo:
        return best

    cos_patch = cos_map[y_lo:y_hi, x_lo:x_hi]
    sin_patch = sin_map[y_lo:y_hi, x_lo:x_hi]
    local_x0, local_y0 = x0 - x_lo, y0 - y_lo

    scale_values = (
        [scale0] if scale_window <= 0 else _frange(scale0 - scale_window, scale0 + scale_window, scale_step)
    )

    angle = angle0 - angle_window
    end_angle = angle0 + angle_window + 1e-6
    while angle <= end_angle:
        for scale in scale_values:
            score_patch = _score_map(cos_patch, sin_patch, level, angle, scale)
            ry_lo = max(0, int(local_y0 - xy_radius))
            ry_hi = min(score_patch.shape[0], int(local_y0 + xy_radius) + 1)
            rx_lo = max(0, int(local_x0 - xy_radius))
            rx_hi = min(score_patch.shape[1], int(local_x0 + xy_radius) + 1)
            if ry_hi > ry_lo and rx_hi > rx_lo:
                sub = score_patch[ry_lo:ry_hi, rx_lo:rx_hi]
                idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
                score = float(sub[idx])
                if score > best[4]:
                    best = [
                        float(x_lo + rx_lo + idx[1]),
                        float(y_lo + ry_lo + idx[0]),
                        float(angle),
                        float(scale),
                        score,
                    ]
        angle += angle_step
    return best


def _nms_dist_thresh_for(model: ShapeModel) -> float:
    model_radius = float(np.linalg.norm(model.corners, axis=1).max()) if model.corners.size else 0.0
    return max(_NMS_DIST_THRESH, _NMS_DIST_FRACTION * model_radius)


def _build_target_pyramid(gray: np.ndarray, num_levels: int) -> list[tuple[np.ndarray, np.ndarray]]:
    target_levels: list[tuple[np.ndarray, np.ndarray]] = []
    level_image = gray
    for level in range(num_levels):
        gx = cv2.Sobel(level_image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(level_image, cv2.CV_32F, 0, 1, ksize=3)
        angle_map = np.arctan2(gy, gx)
        target_levels.append((np.cos(angle_map).astype(np.float32), np.sin(angle_map).astype(np.float32)))
        if level < num_levels - 1:
            level_image = cv2.pyrDown(level_image)
    return target_levels


def build_search_pyramid(search_image: np.ndarray, num_levels: int) -> list[tuple[np.ndarray, np.ndarray]]:
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
    return _build_target_pyramid(gray, num_levels)


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
    öğretildiği boyutta)."""
    num_levels = len(model.levels)
    nms_dist_thresh = _nms_dist_thresh_for(model)
    scale_values = [scale_min] if scale_max <= scale_min else _frange(scale_min, scale_max, scale_step_coarse)
    scale_window = scale_step_coarse if scale_max > scale_min else 0.0

    if target_pyramid is not None and len(target_pyramid) >= num_levels:
        target_levels = target_pyramid[:num_levels]
    else:
        # `gray` SADECE bu dalda gerekli -- `target_pyramid` çağıran tarafından (operatörün
        # `geom.shape_match::run()`'ı, BİRDEN FAZLA model seçiliyken her model için AYNI
        # pyramid'i paylaşır) zaten yeterli derinlikte sağlandığında, tam çözünürlüklü
        # gri tonlama+float32 dönüşümünü HER model için tekrar tekrar yapmak gereksizdi --
        # gerçek kullanıcı raporu: "şekil bul çok kasıyor" (bkz. `build_search_pyramid`
        # docstring'i, AYNI kökten gelen bir önceki perf düzeltmesiyle aynı gerekçe).
        gray = _to_gray(search_image).astype(np.float32)
        target_levels = _build_target_pyramid(gray, num_levels)

    coarsest = num_levels - 1
    cos_map, sin_map = target_levels[coarsest]
    relaxed_min = min_score * max(greediness, 1e-6) * _COARSE_ACCEPT_LOOSENING

    candidates = _coarse_search(
        cos_map, sin_map, model.levels[coarsest], angle_start, angle_extent, angle_step_coarse, relaxed_min,
        scale_values,
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
    for level in range(coarsest - 1, -1, -1):
        cos_map, sin_map = target_levels[level]
        angle_step = max(angle_window / _REFINE_ANGLE_DIVISOR, 0.1)
        scale_step = max(scale_window / _REFINE_SCALE_DIVISOR, _MIN_SCALE_STEP) if scale_window > 0 else 0.0
        candidates = [
            _refine_candidate(
                cos_map, sin_map, model.levels[level], x * 2, y * 2, angle, scale,
                xy_radius=_REFINE_XY_RADIUS, angle_window=angle_window, angle_step=angle_step,
                scale_window=scale_window, scale_step=scale_step,
            )
            for x, y, angle, scale, _score in candidates
        ]
        angle_window = max(angle_window / _REFINE_ANGLE_DIVISOR, 0.5)
        if scale_window > 0:
            scale_window = max(scale_window / _REFINE_SCALE_DIVISOR, _MIN_SCALE_STEP)

    candidates = [c for c in candidates if c[4] >= min_score]
    candidates = _nms(candidates, nms_dist_thresh)
    candidates.sort(key=lambda c: -c[4])
    if max_matches is not None:
        candidates = candidates[:max_matches]
    return [
        MatchResult(x=x, y=y, angle=_normalize_angle(angle), scale=scale, score=score)
        for x, y, angle, scale, score in candidates
    ]


@dataclass(frozen=True)
class LabeledMatch:
    """Bir eşleşmeyi, hangi modelden geldiğini (çizim için köşe/eksen bilgisi modelden farklı
    olabileceğinden) ve gösterilecek etiketi (ör. 'cıvata1') bir arada taşır — birden fazla
    modelin eşleşmelerini TEK bir overlay'de, kendi doğru köşe geometrileriyle çizebilmek için."""

    label: str
    model: ShapeModel
    match: MatchResult


def render_match_overlay(search_image: np.ndarray, entries: list[LabeledMatch]) -> np.ndarray:
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
    OLDUĞU GİBİ kullan, SADECE tek kanallıysa (2 boyutlu) BGR'ye çevir."""
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
    return overlay
