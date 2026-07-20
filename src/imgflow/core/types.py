"""Port tipleri ve port sözleşmesi (operatörlerin girdi/çıktı tanımları)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PortType(Enum):
    IMAGE = "image"
    ROI = "roi"
    SCALAR = "scalar"
    SCALAR_LIST = "scalar_list"
    MEASUREMENTS = "measurements"


@dataclass(frozen=True)
class PortSpec:
    name: str
    type: PortType
    optional: bool = False
    description: str = ""
