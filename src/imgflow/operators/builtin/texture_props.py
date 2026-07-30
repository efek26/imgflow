"""Doku (texture) analizi: gri-seviye eş-oluşum matrisi (GLCM) tabanlı Haralick özellikleri.

Yeni bir bağımlılık (scikit-image) EKLEMEDEN elle numpy ile hesaplanır — HALCON'un
`gen_cooc_matrix`/`cooc_feature` çiftine karşılık gelir. `color_props.py` ile AYNI
gerekçeyle bilinçli olarak `labels` haritasına değil doğrudan `image` girdisine bağlanır
(blob-bazlı analiz iki girdi gerektirir, mevcut tek-girdi/tek-çıktı checkbox zincirine
oturmaz — bkz. CLAUDE.md). `color_props.py` ile AYNI "Otomatik Nesne Tespiti"
(`per_object_enabled`, `core/auto_objects.py`) seçeneği burada da var: açıksa görüntüdeki
ayrı nesneler bulunup GLCM her nesnenin sınırlayıcı KUTUSU (crop) üzerinde ayrı ayrı
hesaplanır — nesne şekli dikdörtgen olmayabileceğinden bbox içindeki birkaç arka plan
pikseli de hesaba karışabilir (kabul edilen bir yaklaşıklık; tam maskeli GLCM, düzenli
komşu-çifti taramasını bozacağından uygulanmadı). Tek bir ürüne odaklanmak isteyen
kullanıcı önce bir `roi.region` adımı da ekleyebilir.

`color_props.py` ile AYNI **Elle ROI Çiz** (`manual_roi_enabled`) üçüncü modu da burada var:
Otomatik Nesne Tespiti'nin ayıramadığı sahnelerde kullanıcı `RoiCanvas` üzerinde fare ile
istediği kadar dikdörtgen çizer, her biri `manual_rois` JSON listesinde saklanır
(`core/roi.py::parse_roi_list`) ve GLCM her biri için AYRI ayrı hesaplanır; tespit
algoritması tamamen devre dışıdır.

Dış kütüphane referansı olmadığından testler dış bir "doğru cevap" ile değil SINIR
DURUMLARLA doğrulanır: tamamen düz bir görüntüde contrast≈0/energy≈1, yüksek frekanslı bir
dama tahtası deseninde contrast belirgin şekilde yüksek olmalıdır.

**HALCON karşılaştırması:** bu operatör HALCON'un `gen_cooc_matrix(Regions, Image, LdGray,
Direction, Matrix)` + `cooc_feature(Matrix, Energy, Correlation, Homogeneity, Contrast)`
çiftine karşılık gelir -- dört özellik (contrast/energy/homogeneity/correlation) VE dört
kanonik yön (0/45/90/135) + mesafe parametrizasyonu HALCON'la BİREBİR aynı isimlendirme/
anlamda. Tek fark: HALCON'un kendi örnek/dokümantasyonunda gıda/endüstriyel QC gibi nesnenin
banttaki dönüşünün rastgele olduğu senaryolarda TEK bir yönün dokuyu yanlış temsil
edebileceği (aynı yüzey deseni, nesne farklı döndüğünde farklı contrast/homogeneity
ölçülebilir) bilinen bir sınırlama olarak not edilir -- yaygın çözüm 4 yönün AYRI AYRI
hesaplanıp ORTALAMASININ alınmasıdır (rotasyon-bağımsız bir tanımlayıcı). Bu yüzden `angle`
parametresine `"average"` (Ortalama/Rotasyon Bağımsız) seçeneği eklendi: seçilirse GLCM+
özellikleri 4 yön için AYRI hesaplanıp aritmetik ortalaması alınır -- varsayılan (`"0"`, tek
yön) davranış/performans DEĞİŞMEDEN kalır, bu SADECE opt-in bir seçenek."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.auto_objects import detect_objects
from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.roi import parse_roi_list
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry

_TEXT_SCALE_REFERENCE_DIM = 1000.0
"""`region_props.py`/`color_props.py` ile AYNI ölçekleme deseni."""

_ANGLE_OFFSETS = {
    "0": (0, 1),
    "45": (-1, 1),
    "90": (-1, 0),
    "135": (-1, -1),
}
_EPS = 1e-10


def _glcm(gray: np.ndarray, distance: int, angle: str, levels: int) -> np.ndarray:
    dy, dx = _ANGLE_OFFSETS[angle]
    dy, dx = dy * distance, dx * distance

    quantized = np.clip((gray.astype(np.int32) * levels) // 256, 0, levels - 1)
    h, w = quantized.shape
    y0, y1 = max(0, -dy), h - max(0, dy)
    x0, x1 = max(0, -dx), w - max(0, dx)
    if y1 <= y0 or x1 <= x0:
        # Görüntü, istenen mesafe/açı için çok küçük -- boş (tamamen sıfır) matris döner,
        # aşağıdaki normalize adımı bunu güvenle sıfır özelliklere çevirir.
        return np.zeros((levels, levels), dtype=np.float64)

    src = quantized[y0:y1, x0:x1]
    dst = quantized[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
    pairs = src.ravel() * levels + dst.ravel()
    glcm = np.bincount(pairs, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    return glcm + glcm.T  # yöne bağlı önyargıyı azaltmak için simetrikleştir


def compute_texture_features(gray: np.ndarray, distance: int, angle: str, levels: int) -> dict[str, float]:
    if angle == "average":
        # HALCON idiomu: 4 kanonik yönün AYRI hesaplanıp ortalanması -- bkz. modül
        # docstring'i. Her yön için tam özellik hesabı (contrast/homogeneity/energy/
        # correlation) ayrı ayrı yapılıp aritmetik ortalaması alınır; GLCM matrislerinin
        # kendisini toplayıp TEK bir birleşik matriste hesaplamak YERİNE bunu tercih ettik
        # çünkü matrisleri toplamak yönler arası gerçek farkı (ör. sadece yatay çizikli bir
        # yüzeyde 0°/90° contrast'ının belirgin FARKLI olması gerektiği bilgisini) örtük
        # şekilde kaybederdi -- ortalama alınan ÖZELLİK değerleri hâlâ her yönün ayrı katkısını
        # yansıtır.
        per_angle = [compute_texture_features(gray, distance, a, levels) for a in _ANGLE_OFFSETS]
        return {
            key: float(np.mean([m[key] for m in per_angle]))
            for key in ("contrast", "homogeneity", "energy", "correlation")
        }
    glcm = _glcm(gray, distance, angle, levels)
    total = glcm.sum()
    if total <= 0:
        return {"contrast": 0.0, "homogeneity": 1.0, "energy": 1.0, "correlation": 1.0}
    p = glcm / total

    i_idx, j_idx = np.mgrid[0:levels, 0:levels]
    diff = i_idx - j_idx

    contrast = float(np.sum(p * diff**2))
    energy = float(np.sum(p**2))
    homogeneity = float(np.sum(p / (1.0 + diff**2)))

    mu_i = float(np.sum(i_idx * p))
    mu_j = float(np.sum(j_idx * p))
    sigma_i = float(np.sqrt(np.sum(((i_idx - mu_i) ** 2) * p)))
    sigma_j = float(np.sqrt(np.sum(((j_idx - mu_j) ** 2) * p)))
    denom = sigma_i * sigma_j
    if denom < _EPS:
        # Tamamen düz bir görüntüde (tüm komşu çiftler aynı seviyede) varyans sıfırdır --
        # matematiksel olarak tanımsız (0/0) yerine "mükemmel korelasyon" (1.0) sayılır.
        correlation = 1.0
    else:
        correlation = float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * p) / denom)

    return {"contrast": contrast, "homogeneity": homogeneity, "energy": energy, "correlation": correlation}


def render_texture_overlay(base_image: np.ndarray, measurement: dict[str, Any]) -> np.ndarray:
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    font_scale = max(0.5, 0.6 * scale_factor)
    thickness = max(1, round(1.5 * scale_factor))
    line_height = max(18, round(22 * scale_factor))
    color = (0, 255, 255)

    lines = [
        f"contrast={measurement['contrast']:.2f}  homog={measurement['homogeneity']:.2f}",
        f"energy={measurement['energy']:.2f}  corr={measurement['correlation']:.2f}",
    ]
    y = line_height
    for line in lines:
        cv2.putText(
            overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
        )
        y += line_height
    return overlay


def render_texture_overlay_multi(base_image: np.ndarray, measurements: list[dict[str, Any]]) -> np.ndarray:
    """`color_props.py::render_color_overlay_multi` ile AYNI "numarala + kutuyu çiz" deseni,
    contrast/homogeneity/energy/correlation yazan Otomatik Nesne Tespiti modu için."""
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    box_thickness = max(2, round(2 * scale_factor))
    font_scale = max(0.45, 0.5 * scale_factor)
    thickness = max(1, round(1.5 * scale_factor))
    line_height = max(16, round(18 * scale_factor))
    color = (0, 255, 255)

    for index, m in enumerate(measurements, start=1):
        x, y, w, h = m["bbox_x"], m["bbox_y"], m["bbox_w"], m["bbox_h"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, box_thickness)
        tag = f"ROI{index}" if m.get("manual") else f"#{index}"
        cv2.putText(
            overlay, tag, (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
        )
        info = f"con={m['contrast']:.1f} hom={m['homogeneity']:.2f} nrj={m['energy']:.2f}"
        cv2.putText(
            overlay, info, (x, y + h + line_height),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
        )
    return overlay


@registry.register
class TexturePropsOp:
    id = "analysis.texture_props"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [
        PortSpec("measurements", PortType.MEASUREMENTS),
        PortSpec(
            "overlay",
            PortType.IMAGE,
            description="Hesaplanan doku özelliklerinin (contrast/homogeneity/energy/correlation) yazıldığı önizleme.",
        ),
    ]
    params = [
        ParamSpec(
            "per_object_enabled",
            ParamType.BOOL,
            default=False,
            label="Otomatik Nesne Tespiti",
            help="Opsiyonel: kapalıyken (varsayılan) tüm görüntü TEK bir ölçüm olarak "
            "değerlendirilir. Açılırsa görüntüdeki ayrı nesneler (ör. tek karede birden "
            "fazla ürün) otomatik olarak (Otsu eşiklemesi + bağlı bileşen analizi) bulunur "
            "ve HER biri için GLCM özellikleri kendi sınırlayıcı kutusu üzerinde ayrı ayrı "
            "hesaplanıp numaralanarak listelenir.",
        ),
        ParamSpec(
            "min_object_area",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Min. Nesne Alanı (px²)",
            help="Sadece Otomatik Nesne Tespiti açıkken kullanılır: bu değerden KÜÇÜK "
            "alanlı nesneler (eşiklemeden kalan küçük gürültü/kırıntılar ya da parlama/"
            "yansımadan kalan küçük parlak lekeler) yok sayılır. 0 = kapalı (varsayılan), "
            "hiçbir nesne alanına göre elenmez.",
        ),
        ParamSpec(
            "max_object_area",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Maks. Nesne Alanı (px²)",
            help="Sadece Otomatik Nesne Tespiti açıkken kullanılır: bu değerden BÜYÜK "
            "alanlı nesneler (ör. yanlışlıkla eşiklenmiş arka plan/bant) yok sayılır. "
            "0 = kapalı (varsayılan), hiçbir nesne alanına göre elenmez.",
        ),
        ParamSpec(
            "threshold_mode",
            ParamType.ENUM,
            default="otsu",
            choices=["otsu", "manual"],
            label="Eşikleme Modu",
            help="Sadece Otomatik Nesne Tespiti açıkken kullanılır. 'otsu' (varsayılan) "
            "eşiği otomatik hesaplar; parlak/yansıtıcı nesnelerde (ör. balonda ışık "
            "yansıması) nesnenin geneli yeterince parlak değilse Otsu SADECE parlamanın "
            "kendisini nesne sanabilir. Bu durumda 'manual' seçip aşağıdaki 'Eşik Değeri'ni "
            "canlı önizlemeye bakarak nesnenin tamamını (parlamayı değil) kapsayacak "
            "şekilde elle ayarlayın.",
        ),
        ParamSpec(
            "threshold_value",
            ParamType.INT,
            default=127,
            min=0,
            max=255,
            label="Eşik Değeri (Eşikleme Modu = manual)",
            help="Bu değerden PARLAK pikseller nesne, KOYU pikseller arka plan sayılır. "
            "Sadece Eşikleme Modu 'manual' iken kullanılır.",
        ),
        ParamSpec(
            "fill_holes",
            ParamType.BOOL,
            default=False,
            label="Delikleri Doldur",
            help="Sadece Otomatik Nesne Tespiti açıkken kullanılır. Açılırsa her nesnenin "
            "dış konturunun içi TAMAMEN dolu sayılır — ışık yansımasının/parlamanın nesne "
            "içinde bıraktığı küçük eşik-altı 'delikleri' doldurup nesnenin tek parça "
            "algılanmasını sağlar (nesnenin dış sınırı hâlâ eşiği geçtiği sürece çalışır).",
        ),
        ParamSpec(
            "close_kernel_size",
            ParamType.INT,
            default=0,
            min=0,
            max=51,
            label="Kapama Çekirdeği (px)",
            help="Sadece Otomatik Nesne Tespiti açıkken kullanılır: eşiklenmiş görüntüye "
            "bu boyutta bir morfolojik KAPAMA uygulanır — parlama/yansımanın nesne "
            "SINIRINDA bıraktığı ince kopuklukları köprüler. 0 = kapalı (varsayılan).",
        ),
        ParamSpec(
            "manual_roi_enabled",
            ParamType.BOOL,
            default=False,
            label="Elle ROI Çiz",
            help="Açılırsa Otomatik Nesne Tespiti TAMAMEN yok sayılır; bunun yerine "
            "önizleme üzerinde fare ile istediğiniz kadar dikdörtgen ROI çizersiniz (her "
            "biri ROI1, ROI2... diye numaralanır) ve HER biri için GLCM özellikleri kendi "
            "dikdörtgeni üzerinde ayrı ayrı hesaplanır. Bir ROI'yi taşımak için içine "
            "tıklayıp sürükleyin, boyutunu değiştirmek için köşe tutamaçlarını kullanın, "
            "silmek için üzerine sağ tıklayın.",
        ),
        ParamSpec(
            "manual_rois",
            ParamType.STRING,
            default="[]",
            label="ROI Listesi (JSON)",
            readonly=True,
            help="Önizleme üzerinde çizilen ROI'lerin ham konum listesi -- elle "
            "düzenlenmez, sadece bilgi/kayıt amaçlıdır (canvas ile senkronize).",
        ),
        ParamSpec(
            "distance",
            ParamType.INT,
            default=1,
            min=1,
            max=20,
            label="Mesafe (px)",
            help="Eş-oluşum matrisi için karşılaştırılan komşu piksel çifti arasındaki uzaklık.",
        ),
        ParamSpec(
            "angle",
            ParamType.ENUM,
            default="0",
            choices=["0", "45", "90", "135", "average"],
            label="Yön (derece)",
            help="Komşu pikselin hangi yönde aranacağı. Yönlü dokularda (ör. sadece yatay "
            "çizikler) doğru yönü seçmek önemlidir. 'average' (Ortalama/Rotasyon Bağımsız) "
            "4 yönün AYRI hesaplanıp ortalamasını alır -- HALCON'un dokümantasyonunda da "
            "önerilen yöntem: banttaki ürün rastgele döndüğünde TEK bir yön aynı yüzey "
            "dokusu için farklı contrast/homogeneity ölçebilir, ortalama bu duyarlılığı "
            "azaltır (4 kat daha yavaştır, sadece gerektiğinde seçin).",
        ),
        ParamSpec(
            "levels",
            ParamType.INT,
            default=32,
            min=2,
            max=256,
            label="Gri Seviye Sayısı",
            help="Görüntü bu kadar gri seviyeye kuantalanır. Düşük değer (ör. 8-16) daha "
            "hızlı ama daha az hassas; yüksek değer (ör. 64-256) daha yavaş ama daha "
            "ayrıntılı doku farkları yakalar.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        distance = int(params.get("distance", 1))
        angle = str(params.get("angle", "0"))
        levels = int(params.get("levels", 32))

        if bool(params.get("manual_roi_enabled", False)):
            rois = parse_roi_list(
                str(params.get("manual_rois", "[]")), image.shape[1], image.shape[0]
            )
            measurements: list[dict[str, Any]] = []
            for index, roi in enumerate(rois, start=1):
                crop = gray[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
                if crop.size == 0:
                    continue
                measurement = compute_texture_features(crop, distance, angle, levels)
                measurement["label"] = index
                measurement["manual"] = True
                measurement["bbox_x"] = roi.x
                measurement["bbox_y"] = roi.y
                measurement["bbox_w"] = roi.w
                measurement["bbox_h"] = roi.h
                measurements.append(measurement)
            overlay = render_texture_overlay_multi(image, measurements)
            return {"measurements": measurements, "overlay": overlay}

        if bool(params.get("per_object_enabled", False)):
            objects = detect_objects(
                image,
                min_area=float(params.get("min_object_area", 0.0)),
                max_area=float(params.get("max_object_area", 0.0)),
                threshold_mode=str(params.get("threshold_mode", "otsu")),
                threshold_value=int(params.get("threshold_value", 127)),
                fill_holes=bool(params.get("fill_holes", False)),
                close_kernel_size=int(params.get("close_kernel_size", 0)),
            )
            measurements: list[dict[str, Any]] = []
            for obj in objects:
                crop = gray[obj.y : obj.y + obj.h, obj.x : obj.x + obj.w]
                measurement = compute_texture_features(crop, distance, angle, levels)
                measurement["label"] = obj.label
                measurement["bbox_x"] = obj.x
                measurement["bbox_y"] = obj.y
                measurement["bbox_w"] = obj.w
                measurement["bbox_h"] = obj.h
                measurements.append(measurement)
            overlay = render_texture_overlay_multi(image, measurements)
            return {"measurements": measurements, "overlay": overlay}

        measurement = compute_texture_features(gray, distance, angle, levels)
        overlay = render_texture_overlay(image, measurement)
        return {"measurements": [measurement], "overlay": overlay}
