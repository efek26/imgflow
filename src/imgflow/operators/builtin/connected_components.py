"""Bağlı bileşen (connected components) etiketleme operatörü."""

from __future__ import annotations

from typing import Any

import cv2

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry


@registry.register
class ConnectedComponentsOp:
    id = "segment.connected_components"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [
        PortSpec(
            "labels",
            PortType.IMAGE,
            description="Her pikselin bileşen etiketini taşıyan int32 harita (0 = arka plan).",
        ),
        PortSpec("count", PortType.SCALAR, description="Arka plan hariç bileşen sayısı."),
    ]
    params = [
        ParamSpec("connectivity", ParamType.ENUM, default="8", choices=["4", "8"], label="Komşuluk"),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        connectivity = int(params.get("connectivity", "8"))
        image = inputs["image"]
        binary = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=connectivity)
        return {"labels": labels, "count": num_labels - 1}
