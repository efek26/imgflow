"""Morfolojik operatörler: erosion, dilation, opening, closing."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry

_SHAPES = {
    "RECT": cv2.MORPH_RECT,
    "ELLIPSE": cv2.MORPH_ELLIPSE,
    "CROSS": cv2.MORPH_CROSS,
}


def _kernel_params() -> list[ParamSpec]:
    return [
        ParamSpec(
            "kernel_size", ParamType.INT, default=3, min=1, max=99, step=2, label="Çekirdek boyutu"
        ),
        ParamSpec("shape", ParamType.ENUM, default="RECT", choices=sorted(_SHAPES), label="Çekirdek şekli"),
        ParamSpec("iterations", ParamType.INT, default=1, min=1, max=50, label="Yineleme"),
    ]


def _kernel(params: dict[str, Any]) -> np.ndarray:
    size = max(1, int(params.get("kernel_size", 3)))
    shape = _SHAPES[params.get("shape", "RECT")]
    return cv2.getStructuringElement(shape, (size, size))


@registry.register
class ErodeOp:
    id = "morphology.erode"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = _kernel_params()

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = cv2.erode(inputs["image"], _kernel(params), iterations=int(params.get("iterations", 1)))
        return {"image": image}


@registry.register
class DilateOp:
    id = "morphology.dilate"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = _kernel_params()

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = cv2.dilate(inputs["image"], _kernel(params), iterations=int(params.get("iterations", 1)))
        return {"image": image}


@registry.register
class OpenOp:
    id = "morphology.open"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = _kernel_params()

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = cv2.morphologyEx(
            inputs["image"], cv2.MORPH_OPEN, _kernel(params), iterations=int(params.get("iterations", 1))
        )
        return {"image": image}


@registry.register
class CloseOp:
    id = "morphology.close"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = _kernel_params()

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = cv2.morphologyEx(
            inputs["image"], cv2.MORPH_CLOSE, _kernel(params), iterations=int(params.get("iterations", 1))
        )
        return {"image": image}
