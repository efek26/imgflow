"""LAB renk analizi: tüm görüntünün (ya da önceki bir ROI adımıyla daraltılmış alanın)
L*a*b* ortalama/standart sapması + opsiyonel referans renkten ΔE76 sapma kontrolü.

Bilinçli olarak `segment.connected_components`'ın "labels" haritasına DEĞİL, doğrudan
`image` girdisine bağlanır (`color.*`/`roi.region` operatörleriyle AYNI tek-girdi/tek-çıktı
deseni) — blob-bazlı (her ürün için ayrı) renk analizi hem orijinal renkli görüntüye hem de
labels haritasına aynı anda ihtiyaç duyardı ki bu `LinearPipeline`'ın tek-girdi/tek-çıktı
checkbox zincirine (bkz. CLAUDE.md) oturmaz. Bunun yerine "Otomatik Nesne Tespiti"
(`per_object_enabled`) açıksa operatör KENDİ İÇİNDE `core/auto_objects.py` ile nesneleri
bulur (ayrı bir connected_components/region_props çifti kurmaya gerek kalmadan) ve her biri
için ayrı bir ölçüm satırı üretir — ör. tek karede birden fazla farklı renkli balon/ürün
varsa her biri numaralanıp kendi L*a*b* değerleriyle listelenir. Kapalıyken (varsayılan,
eski davranış) tüm görüntü tek bir ölçüm satırı olarak değerlendirilir; tek bir ürüne
odaklanmak isteyen kullanıcı bu operatörden ÖNCE bir `roi.region` adımı da ekleyebilir.

**Dalgalanma/Aralık (min/max):** hem Otomatik Nesne Tespiti hem Elle ROI Çiz modu, ortalama/
std'ye EK olarak her kanal için min/max (`l_min/l_max`, `a_min/a_max`, `b_min/b_max`) hesaplar
-- gerçek kullanıcı isteği: "roi içindeki / nesne tespitindeki renk dalgalanmalarını aralık
olarak versin" (tek bir ortalama değer parlama/gölge kaynaklı renk dalgalanmasını gizleyebilir).
Bu, HALCON'un `min_max_gray(Regions, Image, Percent, Min, Max, Range)` operatörünün bir bölge
üzerindeki Min/Max/Range çıktısına karşılık gelir -- `l_mean`/`l_std` zaten HALCON'un
`intensity` (Mean, Deviation) çıktısına karşılık geliyordu, bu ikisi birlikte HALCON'daki
tipik "bölge parlaklık/renk istatistiği" ikilisini (ortalama+dağılım, min-max aralığı)
tamamlıyor. Tüm-görüntü (varsayılan, tespit/ROI kapalı) modu BİLEREK min/max hesaplamaz --
tek bir ürünün İÇİNDEKİ dalgalanma değil, kalibrasyon/referans amaçlı tek bir özet değer
istendiği varsayılır; min/max isteyen kullanıcı Otomatik Nesne Tespiti ya da Elle ROI Çiz'i
açabilir.

**Elle ROI Çiz (`manual_roi_enabled`)** üçüncü bir mod: Otomatik Nesne Tespiti'nin (Otsu/
manuel eşik ne kadar ayarlanırsa ayarlansın) her sahneyi ayıramadığı gerçek kullanıcı
geri bildirimi üzerine eklendi -- kullanıcı `RoiCanvas` üzerinde fare ile istediği kadar
dikdörtgen çizer (bkz. `ui/widgets/roi_canvas.py` multi-mode, `ui/main_window.py`
`_on_manual_rois_canvas_changed`), her biri `manual_rois` parametresinde JSON liste olarak
saklanır (`core/roi.py::parse_roi_list`) ve HER biri için ayrı ölçüm üretilir; tespit
algoritması tamamen devre dışıdır.

**Performans:** `run()` artık görüntünün TAMAMINI baştan LAB'a çevirmiyor -- her dal SADECE
ihtiyacı olan bölgeyi (tüm-görüntü modunda tüm kare, aksi halde SADECE her ROI/nesnenin bbox
kırpımı) dönüştürür, ve tüm-görüntü modundaki ortalama/std `cv2.meanStdDev` ile TEK bir
C-seviyesi çağrıda hesaplanır (eski `lab[..., i].mean()/.std()` kanal-ekseninde strided bir
görünüm üzerinde 6 ayrı numpy geçişiydi, 1920x1080'de ölçülen ~17x daha yavaştı). Canlı kamera
akışında bu operatör seçiliyken tek kare işleme süresini gözle görülür şekilde kısaltıyor;
küçük ROI'lerle çalışırken tam-çözünürlüklü dönüşüm zaten hiç gerekmiyordu.

ΔE76 (basit Öklid mesafesi, `sqrt(dL²+da²+db²)`) tercih edildi — ΔE2000 çok daha karmaşık
bir ağırlıklandırma formülü gerektirir ve endüstriyel "referans renkten ne kadar saptı"
kontrolü için ΔE76 pratikte yeterlidir; `region_props.py`'deki opsiyonel tolerans deseninin
(kapalıyken serbest, açılırsa OK/NG) BİREBİR aynısı kullanılır (nesne başına da uygulanır)."""

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
"""`region_props.py`/`shape_matching.py` ile AYNI ölçekleme deseni — bkz. o dosyalardaki
ayrıntılı açıklama."""

_OK_COLOR = (0, 255, 0)
_FAIL_COLOR = (0, 0, 255)
_NEUTRAL_COLOR = (0, 255, 255)


def _to_lab(image: np.ndarray) -> np.ndarray:
    bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV 8-bit LAB: L 0-255 (gerçek 0-100'ün ölçeklenmişi), a/b 0-255 (128 ofsetli).
    # Standart CIE L*a*b* birimlerine (L 0-100, a/b yaklaşık -128..127) çeviriyoruz ki
    # ΔE76 ve girilen referans değerleri (ref_l/ref_a/ref_b) sezgisel/karşılaştırılabilir olsun.
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0
    return lab


def render_color_overlay(base_image: np.ndarray, measurement: dict[str, Any]) -> np.ndarray:
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    font_scale = max(0.5, 0.6 * scale_factor)
    thickness = max(1, round(1.5 * scale_factor))
    line_height = max(18, round(22 * scale_factor))

    if "tolerance_ok" in measurement:
        color = _OK_COLOR if measurement["tolerance_ok"] else _FAIL_COLOR
    else:
        color = _NEUTRAL_COLOR

    lines = [
        f"L={measurement['l_mean']:.1f}  a={measurement['a_mean']:.1f}  b={measurement['b_mean']:.1f}"
    ]
    if "delta_e" in measurement:
        status = "OK" if measurement.get("tolerance_ok") else "NG"
        lines.append(f"dE={measurement['delta_e']:.1f}  {status}")

    y = line_height
    for line in lines:
        cv2.putText(
            overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA
        )
        y += line_height
    return overlay


def render_color_overlay_multi(base_image: np.ndarray, measurements: list[dict[str, Any]]) -> np.ndarray:
    """`region_props.py::draw_measurements_overlay` ile AYNI "numarala + kutuyu çiz" deseni,
    ama boyut yerine L/a/b (+ tolerans açıksa ΔE/OK-NG) yazan Otomatik Nesne Tespiti modu
    için basit (döndürülmemiş) sınırlayıcı kutu kullanır."""
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    box_thickness = max(2, round(2 * scale_factor))
    font_scale = max(0.45, 0.5 * scale_factor)
    thickness = max(1, round(1.5 * scale_factor))
    line_height = max(16, round(18 * scale_factor))

    for index, m in enumerate(measurements, start=1):
        x, y, w, h = m["bbox_x"], m["bbox_y"], m["bbox_w"], m["bbox_h"]
        if "tolerance_ok" in m:
            color = _OK_COLOR if m["tolerance_ok"] else _FAIL_COLOR
        else:
            color = _NEUTRAL_COLOR
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, box_thickness)
        tag = f"ROI{index}" if m.get("manual") else f"#{index}"
        cv2.putText(
            overlay, tag, (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
        )
        info = f"L={m['l_mean']:.0f} a={m['a_mean']:.0f} b={m['b_mean']:.0f}"
        if "delta_e" in m:
            status = "OK" if m.get("tolerance_ok") else "NG"
            info += f"  dE={m['delta_e']:.1f} {status}"
        lines = [info]
        if "l_min" in m:
            lines.append(
                f"L:[{m['l_min']:.0f},{m['l_max']:.0f}] a:[{m['a_min']:.0f},{m['a_max']:.0f}] "
                f"b:[{m['b_min']:.0f},{m['b_max']:.0f}]"
            )
        ty = y + h + line_height
        for line in lines:
            cv2.putText(
                overlay, line, (x, ty),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
            )
            ty += line_height
    return overlay


@registry.register
class ColorPropsOp:
    id = "analysis.color_props"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [
        PortSpec("measurements", PortType.MEASUREMENTS),
        PortSpec(
            "overlay",
            PortType.IMAGE,
            description="L/a/b ortalama değerlerinin (ve tolerans açıksa ΔE + OK/NG) yazıldığı önizleme.",
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
            "fazla farklı renkli ürün/balon) otomatik olarak (Otsu eşiklemesi + bağlı "
            "bileşen analizi) bulunur ve HER biri için ayrı ayrı L*a*b* değerleri "
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
            "biri ROI1, ROI2... diye numaralanır) ve HER biri için L*a*b* ortalama/std "
            "DEĞERLERİ ve renk dalgalanma ARALIĞI (min-max) ayrı ayrı hesaplanır. Bir "
            "ROI'yi taşımak için içine tıklayıp sürükleyin, boyutunu değiştirmek için köşe "
            "tutamaçlarını kullanın, silmek için üzerine sağ tıklayın.",
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
            "tolerance_enabled",
            ParamType.BOOL,
            default=False,
            label="Tolerans Kontrolü Aktif",
            help="Opsiyonel: kapalıyken (varsayılan) sadece L/a/b ortalama/std değerleri "
            "hesaplanır. Açılırsa aşağıdaki referans renkten ΔE76 (renk farkı) hesaplanıp "
            "'Maks. ΔE' ile karşılaştırılır; içindeyse OK, dışındaysa NG (Otomatik Nesne "
            "Tespiti ya da Elle ROI Çiz açıkken her nesneye/ROI'ye ayrı ayrı uygulanır).",
        ),
        ParamSpec(
            "ref_l", ParamType.FLOAT, default=50.0, min=0.0, max=100.0, label="Referans L*",
            help="Referans rengin parlaklığı (0=siyah, 100=beyaz).",
        ),
        ParamSpec(
            "ref_a", ParamType.FLOAT, default=0.0, min=-128.0, max=127.0, label="Referans a*",
            help="Referans rengin yeşil(-)/kırmızı(+) ekseni.",
        ),
        ParamSpec(
            "ref_b", ParamType.FLOAT, default=0.0, min=-128.0, max=127.0, label="Referans b*",
            help="Referans rengin mavi(-)/sarı(+) ekseni.",
        ),
        ParamSpec(
            "delta_e_max",
            ParamType.FLOAT,
            default=5.0,
            min=0.0,
            label="Maks. ΔE",
            help="Ölçülen ortalama rengin referans renkten sapması bu değeri AŞARSA NG "
            "sayılır (yalnızca Tolerans Kontrolü açıkken kullanılır).",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        # `lab = _to_lab(image)` tüm görüntüyü ÖNCEDEN dönüştürmek yerine her dal SADECE
        # ihtiyacı olan bölgeyi (ROI/nesne kırpımı ya da tüm görüntü) dönüştürür -- bir
        # 1920x1080 karede tam-çözünürlüklü BGR->LAB dönüşümü ~12-14ms sürüyor, ama birkaç
        # küçük ROI/nesneyle çalışırken bu maliyetin neredeyse tamamı gereksizdi (bkz. FAZ
        # sonrası performans notu, aşağıdaki dallar).
        tolerance_enabled = bool(params.get("tolerance_enabled", False))
        ref_l = float(params.get("ref_l", 50.0))
        ref_a = float(params.get("ref_a", 0.0))
        ref_b = float(params.get("ref_b", 0.0))
        delta_e_max = float(params.get("delta_e_max", 5.0))

        def _apply_tolerance(measurement: dict[str, Any]) -> None:
            if not tolerance_enabled:
                return
            delta_e = float(
                np.sqrt(
                    (measurement["l_mean"] - ref_l) ** 2
                    + (measurement["a_mean"] - ref_a) ** 2
                    + (measurement["b_mean"] - ref_b) ** 2
                )
            )
            measurement["delta_e"] = delta_e
            measurement["tolerance_ok"] = delta_e <= delta_e_max

        if bool(params.get("manual_roi_enabled", False)):
            rois = parse_roi_list(
                str(params.get("manual_rois", "[]")), image.shape[1], image.shape[0]
            )
            measurements = []
            for index, roi in enumerate(rois, start=1):
                bgr_crop = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
                if bgr_crop.size == 0:
                    continue
                crop = _to_lab(bgr_crop)
                mean, std = cv2.meanStdDev(crop)
                l_mean, a_mean, b_mean = (float(v) for v in mean.ravel())
                l_std, a_std, b_std = (float(v) for v in std.ravel())
                measurement = {
                    "label": index,
                    "manual": True,
                    "l_mean": l_mean,
                    "a_mean": a_mean,
                    "b_mean": b_mean,
                    "l_std": l_std,
                    "a_std": a_std,
                    "b_std": b_std,
                    "l_min": float(crop[..., 0].min()),
                    "l_max": float(crop[..., 0].max()),
                    "a_min": float(crop[..., 1].min()),
                    "a_max": float(crop[..., 1].max()),
                    "b_min": float(crop[..., 2].min()),
                    "b_max": float(crop[..., 2].max()),
                    "bbox_x": roi.x,
                    "bbox_y": roi.y,
                    "bbox_w": roi.w,
                    "bbox_h": roi.h,
                }
                _apply_tolerance(measurement)
                measurements.append(measurement)
            overlay = render_color_overlay_multi(image, measurements)
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
                bgr_crop = image[obj.y : obj.y + obj.h, obj.x : obj.x + obj.w]
                crop = _to_lab(bgr_crop)
                pixels = crop[obj.mask]
                if pixels.size == 0:
                    continue
                l_mean, a_mean, b_mean = (float(pixels[:, i].mean()) for i in range(3))
                l_std, a_std, b_std = (float(pixels[:, i].std()) for i in range(3))
                measurement = {
                    "label": obj.label,
                    "l_mean": l_mean,
                    "a_mean": a_mean,
                    "b_mean": b_mean,
                    "l_std": l_std,
                    "a_std": a_std,
                    "b_std": b_std,
                    # Gerçek kullanıcı isteği: "lab analizi yaparken nesne tespitinde
                    # dalgalanma gözüksün" -- Elle ROI Çiz modundaki AYNI min/max ARALIK
                    # alanları (HALCON'un `min_max_gray`'inin Range çıktısına karşılık gelir)
                    # artık Otomatik Nesne Tespiti'nde de hesaplanıyor; ortalama tek başına
                    # parlama/gölge kaynaklı renk dalgalanmasını gizleyebiliyordu.
                    "l_min": float(pixels[:, 0].min()),
                    "l_max": float(pixels[:, 0].max()),
                    "a_min": float(pixels[:, 1].min()),
                    "a_max": float(pixels[:, 1].max()),
                    "b_min": float(pixels[:, 2].min()),
                    "b_max": float(pixels[:, 2].max()),
                    "bbox_x": obj.x,
                    "bbox_y": obj.y,
                    "bbox_w": obj.w,
                    "bbox_h": obj.h,
                }
                _apply_tolerance(measurement)
                measurements.append(measurement)
            overlay = render_color_overlay_multi(image, measurements)
            return {"measurements": measurements, "overlay": overlay}

        lab = _to_lab(image)
        # `cv2.meanStdDev` TEK bir C-seviyesi çağrıyla üç kanalın ortalama/std'sini birden
        # hesaplıyor; eski `lab[..., i].mean()/.std()` (kanal ekseninde STRIDED bir görünüm
        # üzerinde 6 ayrı numpy indirgeme geçişi) 1920x1080'de ölçülen ~17x daha yavaştı --
        # canlı kamera akışında bu operatör seçiliyken tek karelik bekleme süresinin önemli
        # bir kısmını oluşturuyordu.
        mean, std = cv2.meanStdDev(lab)
        l_mean, a_mean, b_mean = (float(v) for v in mean.ravel())
        l_std, a_std, b_std = (float(v) for v in std.ravel())
        measurement = {
            "l_mean": l_mean,
            "a_mean": a_mean,
            "b_mean": b_mean,
            "l_std": l_std,
            "a_std": a_std,
            "b_std": b_std,
        }
        _apply_tolerance(measurement)

        overlay = render_color_overlay(image, measurement)
        return {"measurements": [measurement], "overlay": overlay}
