"""Gürültü azaltma / yumuşatma filtreleri: gaussian, median, bilateral."""

from __future__ import annotations

from typing import Any

import cv2

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
