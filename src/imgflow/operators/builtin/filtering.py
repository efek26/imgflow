"""Gürültü azaltma / yumuşatma filtreleri (gaussian, median, bilateral) ve kenar tespiti (Canny)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry


def _odd(value: Any) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


@registry.register
class GaussianBlurOp:
    id = "filter.gaussian_blur"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "kernel_size", ParamType.INT, default=5, min=1, max=99, step=2, label="Çekirdek boyutu"
        ),
        ParamSpec(
            "sigma_x",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            max=50.0,
            label="Sigma X",
            help="0 verilirse kernel_size'dan otomatik hesaplanır.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        k = _odd(params.get("kernel_size", 5))
        sigma_x = float(params.get("sigma_x", 0.0))
        image = cv2.GaussianBlur(inputs["image"], (k, k), sigmaX=sigma_x)
        return {"image": image}


@registry.register
class MedianBlurOp:
    id = "filter.median_blur"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "kernel_size", ParamType.INT, default=5, min=1, max=99, step=2, label="Çekirdek boyutu"
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        k = _odd(params.get("kernel_size", 5))
        image = cv2.medianBlur(inputs["image"], k)
        return {"image": image}


@registry.register
class BilateralFilterOp:
    id = "filter.bilateral"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec("diameter", ParamType.INT, default=9, min=1, max=25, label="Çap"),
        ParamSpec("sigma_color", ParamType.FLOAT, default=75.0, min=0.0, max=255.0, label="Sigma Renk"),
        ParamSpec("sigma_space", ParamType.FLOAT, default=75.0, min=0.0, max=255.0, label="Sigma Uzay"),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = cv2.bilateralFilter(
            inputs["image"],
            d=int(params.get("diameter", 9)),
            sigmaColor=float(params.get("sigma_color", 75.0)),
            sigmaSpace=float(params.get("sigma_space", 75.0)),
        )
        return {"image": image}


@registry.register
class CannyEdgeOp:
    """Kenar tespiti (Canny). Filtrelemeden (gürültü azaltma) sonra uygulanması önerilir —
    aksi halde gürültü de kenar olarak algılanıp sahte/kopuk kenarlar üretebilir."""

    id = "filter.canny_edge"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "threshold1",
            ParamType.INT,
            default=50,
            min=0,
            max=500,
            label="Alt Eşik",
            help="Bu değerin altındaki gradyanlar kenar sayılmaz.",
        ),
        ParamSpec(
            "threshold2",
            ParamType.INT,
            default=150,
            min=0,
            max=500,
            label="Üst Eşik",
            help="Bu değerin üzerindeki gradyanlar kesin kenar sayılır (ikisi arası: komşuluğa bağlı).",
        ),
        ParamSpec(
            "overlay",
            ParamType.BOOL,
            default=False,
            label="Orijinal Üzerine Bindir",
            help="Kapalıyken sonuç saf siyah/yeşil kenar haritasıdır; açıkken kenarlar orijinal görüntü üzerine yeşil çizilir.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        edges = cv2.Canny(gray, int(params.get("threshold1", 50)), int(params.get("threshold2", 150)))

        if params.get("overlay", False):
            thick_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            base = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            overlay = base.copy()
            overlay[thick_edges > 0] = (0, 255, 0)
            return {"image": overlay}

        return {"image": edges}


@registry.register
class SobelEdgeOp:
    """Sobel gradyan tabanlı kenar tespiti. Canny'ye göre daha basit/gürültüye daha açıktır
    ama yön bilgisini (X/Y) ayrı ayrı verebildiği için yönlü kusur/çizgi tespitinde kullanışlıdır."""

    id = "filter.sobel_edge"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "kernel_size", ParamType.INT, default=3, min=1, max=31, step=2, label="Çekirdek Boyutu"
        ),
        ParamSpec(
            "direction",
            ParamType.ENUM,
            default="BOTH",
            choices=["X", "Y", "BOTH"],
            label="Yön",
            help="X: dikey kenarlar, Y: yatay kenarlar, BOTH: her iki yönün birleşik gradyan büyüklüğü.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        k = _odd(params.get("kernel_size", 3))
        direction = params.get("direction", "BOTH")

        if direction == "X":
            gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        elif direction == "Y":
            gradient = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        else:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            gradient = cv2.magnitude(gx, gy)

        return {"image": cv2.convertScaleAbs(gradient)}


@registry.register
class LaplacianEdgeOp:
    """Laplacian (ikinci türev) tabanlı kenar tespiti; ince/keskin kusur çizgilerine (çatlak,
    çizik) Sobel'den daha duyarlıdır ama gürültüyü de güçlendirir (önce bulanıklaştırma önerilir)."""

    id = "filter.laplacian_edge"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "kernel_size", ParamType.INT, default=3, min=1, max=31, step=2, label="Çekirdek Boyutu"
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        k = _odd(params.get("kernel_size", 3))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
        return {"image": cv2.convertScaleAbs(laplacian)}


@registry.register
class SharpenOp:
    """Keskinleştirme (unsharp mask). Odak dışı/bulanık kameralarda kenar netliğini görsel
    olarak artırır; bir bulanıklaştırma filtresinin tam tersi etkisi vardır."""

    id = "filter.sharpen"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "amount",
            ParamType.FLOAT,
            default=1.0,
            min=0.0,
            max=5.0,
            step=0.1,
            label="Miktar",
            help="0 = etkisiz; yüksek değerler keskinleştirmeyi abartır ve gürültüyü de belirginleştirir.",
        ),
        ParamSpec(
            "kernel_size",
            ParamType.INT,
            default=5,
            min=1,
            max=31,
            step=2,
            label="Bulanıklaştırma Çekirdeği",
            help="İç hesaplamada kullanılan Gauss bulanıklaştırma boyutu; büyüdükçe daha kaba detaylar da keskinleştirilir.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        k = _odd(params.get("kernel_size", 5))
        amount = float(params.get("amount", 1.0))
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        return {"image": sharpened}


@registry.register
class HistogramEqualizeOp:
    """Histogram eşitleme: düşük kontrastlı (sisli/soluk aydınlatılmış) görüntülerde
    kontrastı otomatik artırır. Bölgesel değil global çalışır — homojen olmayan
    aydınlatmada bunun yerine Siyah-Beyaz modundaki CLAHE seçeneği tercih edilmelidir."""

    id = "filter.histogram_equalize"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params: list[ParamSpec] = []

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return {"image": cv2.equalizeHist(gray)}
