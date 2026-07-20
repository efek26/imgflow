"""Diskten tek bir görüntü yükleyen kaynak operatörü; pipeline'ın kök node'u olarak kullanılır."""

from __future__ import annotations

from typing import Any

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.io_utils.image_io import load_image
from imgflow.operators import registry


@registry.register
class ImageSourceOp:
    id = "io.image_source"
    inputs: list[PortSpec] = []
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [ParamSpec("path", ParamType.STRING, default="", label="Dosya yolu")]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        path = params.get("path") or ""
        if not path:
            raise ValueError("'path' parametresi boş olamaz.")
        return {"image": load_image(path)}
