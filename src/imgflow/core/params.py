"""Operatör parametrelerinin tip/aralık tanımı (UI formunun otomatik üretilmesi için)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ParamType(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    STRING = "string"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    label: str = ""
    help: str = ""

    def __post_init__(self) -> None:
        if self.type is ParamType.ENUM and not self.choices:
            raise ValueError(f"ParamSpec '{self.name}': ENUM tipi için 'choices' zorunludur.")


def defaults_for(params: list[ParamSpec]) -> dict[str, Any]:
    return {p.name: p.default for p in params}
