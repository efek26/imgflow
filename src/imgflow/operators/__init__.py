"""Operatör paketi: global registry + builtin operatörlerin kaydı."""

from __future__ import annotations

from imgflow.operators.registry import Registry

registry = Registry()


def _register_builtins() -> None:
    from imgflow.operators.builtin import (  # noqa: F401
        color_convert,
        connected_components,
        filtering,
        image_source,
        morphology,
        region_props,
        roi,
        threshold,
    )


_register_builtins()
