"""ROI (ilgi alanı) operatörü: tek adımda dikdörtgen ya da daire bölgeye kırpma."""

from __future__ import annotations

from typing import Any

import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.roi import RoiCircle, RoiRect
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry

_SHAPES = ["RECT", "CIRCLE"]


@registry.register
class RoiRegionOp:
    """Aktifse görüntüyü seçilen dikdörtgen/daire bölgeye kırpar, değilse aynen geçirir.

    Daire modunda çıktı, dairenin sınırlayıcı kare kutusuna kırpılır ve kutu içinde dairenin
    dışında kalan pikseller siyaha (0) boyanır — HALCON'daki circle-ROI davranışına benzer.
    Tek IMAGE girdi/çıktısı olduğu için doğrusal (checkbox tabanlı) pipeline'da ikinci bir
    girdiye ihtiyaç duymadan otomatik olarak zincire bağlanabilir.
    """

    id = "roi.region"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "enabled",
            ParamType.BOOL,
            default=False,
            label="ROI Aktif",
            help="Kapalıyken görüntü değişmeden geçer; açıkken sadece aşağıdaki bölge işlenir.",
        ),
        ParamSpec(
            "shape",
            ParamType.ENUM,
            default="RECT",
            choices=_SHAPES,
            label="Şekil",
            help="RECT: dikdörtgen bölge (x/y/genişlik/yükseklik). CIRCLE: dairesel bölge (merkez/yarıçap).",
        ),
        ParamSpec(
            "x", ParamType.INT, default=0, min=0, max=10000, label="X", help="Dikdörtgen ROI'nin sol üst köşe X koordinatı."
        ),
        ParamSpec(
            "y", ParamType.INT, default=0, min=0, max=10000, label="Y", help="Dikdörtgen ROI'nin sol üst köşe Y koordinatı."
        ),
        ParamSpec(
            "w",
            ParamType.INT,
            default=100,
            min=1,
            max=10000,
            label="Genişlik",
            help="Dikdörtgen ROI'nin genişliği (piksel).",
        ),
        ParamSpec(
            "h",
            ParamType.INT,
            default=100,
            min=1,
            max=10000,
            label="Yükseklik",
            help="Dikdörtgen ROI'nin yüksekliği (piksel).",
        ),
        ParamSpec(
            "cx",
            ParamType.INT,
            default=100,
            min=0,
            max=10000,
            label="Merkez X",
            help="Dairesel ROI'nin merkez X koordinatı.",
        ),
        ParamSpec(
            "cy",
            ParamType.INT,
            default=100,
            min=0,
            max=10000,
            label="Merkez Y",
            help="Dairesel ROI'nin merkez Y koordinatı.",
        ),
        ParamSpec(
            "r",
            ParamType.INT,
            default=50,
            min=1,
            max=10000,
            label="Yarıçap",
            help="Dairesel ROI'nin yarıçapı (piksel).",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        if not params.get("enabled", False):
            return {"image": image}

        if params.get("shape", "RECT") == "CIRCLE":
            return {"image": self._run_circle(image, params)}
        return {"image": self._run_rect(image, params)}

    @staticmethod
    def _run_rect(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        roi = RoiRect(
            x=int(params.get("x", 0)),
            y=int(params.get("y", 0)),
            w=int(params.get("w", 100)),
            h=int(params.get("h", 100)),
        ).clamp(image.shape[1], image.shape[0])
        row_slice, col_slice = roi.as_slice()
        return image[row_slice, col_slice]

    @staticmethod
    def _run_circle(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        roi = RoiCircle(
            cx=int(params.get("cx", 100)),
            cy=int(params.get("cy", 100)),
            r=int(params.get("r", 50)),
        ).clamp(image.shape[1], image.shape[0])
        row_slice, col_slice = roi.bounding_rect().as_slice()
        cropped = image[row_slice, col_slice].copy()

        yy, xx = np.ogrid[: cropped.shape[0], : cropped.shape[1]]
        outside = (xx - roi.r) ** 2 + (yy - roi.r) ** 2 > roi.r**2
        cropped[outside] = 0
        return cropped
