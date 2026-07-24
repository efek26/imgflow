"""Operatör paketi: global registry + builtin operatörlerin kaydı."""

from __future__ import annotations

from imgflow.operators.registry import Registry

registry = Registry()


def _register_builtins() -> None:
    from imgflow.operators.builtin import (  # noqa: F401
        color_modes,
        color_props,
        connected_components,
        filtering,
        flat_field,
        image_source,
        morphology,
        onnx_detect,
        region_props,
        roi,
        shape_match,
        texture_props,
        threshold,
    )


_register_builtins()
