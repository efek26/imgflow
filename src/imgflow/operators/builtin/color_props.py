"""LAB renk analizi: tüm görüntünün (ya da önceki bir ROI adımıyla daraltılmış alanın)
L*a*b* ortalama/standart sapması + opsiyonel referans renkten ΔE76 sapma kontrolü.

Bilinçli olarak `segment.connected_components`'ın "labels" haritasına DEĞİL, doğrudan
`image` girdisine bağlanır (`color.*`/`roi.region` operatörleriyle AYNI tek-girdi/tek-çıktı
deseni) — blob-bazlı (her ürün için ayrı) renk analizi hem orijinal renkli görüntüye hem de
labels haritasına aynı anda ihtiyaç duyardı ki bu `LinearPipeline`'ın tek-girdi/tek-çıktı
checkbox zincirine (bkz. CLAUDE.md) oturmaz. Tek bir ürüne odaklanmak isteyen kullanıcı bu
operatörden ÖNCE bir `roi.region` adımı ekleyebilir.

ΔE76 (basit Öklid mesafesi, `sqrt(dL²+da²+db²)`) tercih edildi — ΔE2000 çok daha karmaşık
bir ağırlıklandırma formülü gerektirir ve endüstriyel "referans renkten ne kadar saptı"
kontrolü için ΔE76 pratikte yeterlidir; `region_props.py`'deki opsiyonel tolerans deseninin
(kapalıyken serbest, açılırsa OK/NG) BİREBİR aynısı kullanılır."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
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
            "tolerance_enabled",
            ParamType.BOOL,
            default=False,
            label="Tolerans Kontrolü Aktif",
            help="Opsiyonel: kapalıyken (varsayılan) sadece L/a/b ortalama/std değerleri "
            "hesaplanır. Açılırsa aşağıdaki referans renkten ΔE76 (renk farkı) hesaplanıp "
            "'Maks. ΔE' ile karşılaştırılır; içindeyse OK, dışındaysa NG.",
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
        lab = _to_lab(image)
        l_mean, a_mean, b_mean = (float(lab[..., i].mean()) for i in range(3))
        l_std, a_std, b_std = (float(lab[..., i].std()) for i in range(3))

        measurement: dict[str, Any] = {
            "l_mean": l_mean,
            "a_mean": a_mean,
            "b_mean": b_mean,
            "l_std": l_std,
            "a_std": a_std,
            "b_std": b_std,
        }

        if bool(params.get("tolerance_enabled", False)):
            ref_l = float(params.get("ref_l", 50.0))
            ref_a = float(params.get("ref_a", 0.0))
            ref_b = float(params.get("ref_b", 0.0))
            delta_e_max = float(params.get("delta_e_max", 5.0))
            delta_e = float(np.sqrt((l_mean - ref_l) ** 2 + (a_mean - ref_a) ** 2 + (b_mean - ref_b) ** 2))
            measurement["delta_e"] = delta_e
            measurement["tolerance_ok"] = delta_e <= delta_e_max

        overlay = render_color_overlay(image, measurement)
        return {"measurements": [measurement], "overlay": overlay}
