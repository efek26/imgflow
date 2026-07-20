"""Operatör arayüzü: tüm builtin/harici operatörlerin uyması gereken sözleşme."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from imgflow.core.params import ParamSpec
from imgflow.core.types import PortSpec


@runtime_checkable
class Operator(Protocol):
    id: str
    inputs: list[PortSpec]
    outputs: list[PortSpec]
    params: list[ParamSpec]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]: ...
