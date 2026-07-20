"""İkili (binary) eşikleme operatörü."""

from __future__ import annotations

from typing import Any

import cv2

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry

_MODES = {
    "BINARY": cv2.THRESH_BINARY,
    "BINARY_INV": cv2.THRESH_BINARY_INV,
    "OTSU": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
}


@registry.register
class ThresholdOp:
    id = "segment.threshold"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec("value", ParamType.INT, default=127, min=0, max=255, label="Eşik değeri"),
        ParamSpec("max_value", ParamType.INT, default=255, min=0, max=255, label="Maks. değer"),
        ParamSpec("mode", ParamType.ENUM, default="BINARY", choices=sorted(_MODES), label="Mod"),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        mode = params.get("mode", "BINARY")
        _, result = cv2.threshold(
            inputs["image"],
            params.get("value", 127),
            params.get("max_value", 255),
            _MODES[mode],
        )
        return {"image": result}
