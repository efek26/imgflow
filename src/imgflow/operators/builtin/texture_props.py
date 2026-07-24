"""Doku (texture) analizi: gri-seviye eş-oluşum matrisi (GLCM) tabanlı Haralick özellikleri.

Yeni bir bağımlılık (scikit-image) EKLEMEDEN elle numpy ile hesaplanır — HALCON'un
`gen_cooc_matrix`/`cooc_feature` çiftine karşılık gelir. `color_props.py` ile AYNI
gerekçeyle bilinçli olarak `labels` haritasına değil doğrudan `image` girdisine bağlanır
(blob-bazlı analiz iki girdi gerektirir, mevcut tek-girdi/tek-çıktı checkbox zincirine
oturmaz — bkz. CLAUDE.md); tek bir ürüne odaklanmak isteyen kullanıcı önce bir `roi.region`
adımı ekleyebilir.

Dış kütüphane referansı olmadığından testler dış bir "doğru cevap" ile değil SINIR
DURUMLARLA doğrulanır: tamamen düz bir görüntüde contrast≈0/energy≈1, yüksek frekanslı bir
dama tahtası deseninde contrast belirgin şekilde yüksek olmalıdır."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
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
            choices=["0", "45", "90", "135"],
            label="Yön (derece)",
            help="Komşu pikselin hangi yönde aranacağı. Yönlü dokularda (ör. sadece yatay "
            "çizikler) doğru yönü seçmek önemlidir.",
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

        measurement = compute_texture_features(gray, distance, angle, levels)
        overlay = render_texture_overlay(image, measurement)
        return {"measurements": [measurement], "overlay": overlay}
