"""Renk uzayı dönüşümü operatörü (ör. BGR -> GRAY/HSV/LAB/RGB)."""

from __future__ import annotations

from typing import Any

import cv2

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry

_CONVERSIONS = {
    "BGR2GRAY": cv2.COLOR_BGR2GRAY,
    "BGR2HSV": cv2.COLOR_BGR2HSV,
    "BGR2LAB": cv2.COLOR_BGR2LAB,
    "BGR2RGB": cv2.COLOR_BGR2RGB,
    "GRAY2BGR": cv2.COLOR_GRAY2BGR,
}


@registry.register
class ColorConvertOp:
    id = "color.convert"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "mode",
            ParamType.ENUM,
            default="BGR2GRAY",
            choices=sorted(_CONVERSIONS),
            label="Dönüşüm",
        )
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        mode = params.get("mode", "BGR2GRAY")
        code = _CONVERSIONS[mode]
        return {"image": cv2.cvtColor(inputs["image"], code)}
