"""ROI (ilgi alanı) operatörleri: dikdörtgen ROI tanımlama, kırpma ve maskeleme.

roi.crop görüntüyü küçültüp koordinat sistemini kaydırırken, roi.mask orijinal boyutu ve
koordinat sistemini korur (ROI dışını sıfırlar) — ölçüm operatörlerinin bbox/centroid
çıktılarının tam görüntüye göre kalmasını istediğinde roi.mask tercih edilir.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.roi import RoiRect
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry


@registry.register
class RoiRectangleOp:
    id = "roi.rectangle"
    inputs = [
        PortSpec(
            "image",
            PortType.IMAGE,
            optional=True,
            description="Verilirse ROI görüntü sınırlarına kırpılır (clamp).",
        )
    ]
    outputs = [PortSpec("roi", PortType.ROI)]
    params = [
        ParamSpec("x", ParamType.INT, default=0, min=0, label="X"),
        ParamSpec("y", ParamType.INT, default=0, min=0, label="Y"),
        ParamSpec("w", ParamType.INT, default=100, min=1, label="Genişlik"),
        ParamSpec("h", ParamType.INT, default=100, min=1, label="Yükseklik"),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        roi = RoiRect(
            x=int(params.get("x", 0)),
            y=int(params.get("y", 0)),
            w=int(params.get("w", 100)),
            h=int(params.get("h", 100)),
        )
        image = inputs.get("image")
        if image is not None:
            roi = roi.clamp(image.shape[1], image.shape[0])
        return {"roi": roi}


@registry.register
class RoiCropOp:
    id = "roi.crop"
    inputs = [PortSpec("image", PortType.IMAGE), PortSpec("roi", PortType.ROI)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params: list[ParamSpec] = []

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        roi: RoiRect = inputs["roi"].clamp(image.shape[1], image.shape[0])
        row_slice, col_slice = roi.as_slice()
        return {"image": image[row_slice, col_slice]}


@registry.register
class RoiMaskOp:
    id = "roi.mask"
    inputs = [PortSpec("image", PortType.IMAGE), PortSpec("roi", PortType.ROI)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params: list[ParamSpec] = []

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        roi: RoiRect = inputs["roi"].clamp(image.shape[1], image.shape[0])
        row_slice, col_slice = roi.as_slice()
        masked = np.zeros_like(image)
        masked[row_slice, col_slice] = image[row_slice, col_slice]
        return {"image": masked}
