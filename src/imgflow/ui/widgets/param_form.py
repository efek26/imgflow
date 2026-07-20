"""ParamSpec listesinden otomatik form üreten widget."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from imgflow.core.params import ParamSpec, ParamType

_INT_FALLBACK = 1_000_000
_FLOAT_FALLBACK = 1.0e6


class ParamForm(QWidget):
    params_changed = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._specs: list[ParamSpec] = []
        self._values: dict[str, Any] = {}
        self._widgets: dict[str, QWidget] = {}

    def set_params(self, specs: list[ParamSpec], values: dict[str, Any]) -> None:
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._specs = list(specs)
        self._values = dict(values)
        self._widgets = {}

        for spec in self._specs:
            widget = self._build_widget(spec)
            self._widgets[spec.name] = widget
            self._layout.addRow(spec.label or spec.name, widget)

    def values(self) -> dict[str, Any]:
        return dict(self._values)

    def widget_for(self, name: str) -> QWidget | None:
        return self._widgets.get(name)

    def _build_widget(self, spec: ParamSpec) -> QWidget:
        value = self._values.get(spec.name, spec.default)

        if spec.type is ParamType.BOOL:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(lambda checked, name=spec.name: self._on_change(name, checked))
            return widget

        if spec.type is ParamType.ENUM:
            widget = QComboBox()
            widget.addItems(spec.choices or [])
            if value in (spec.choices or []):
                widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(lambda text, name=spec.name: self._on_change(name, text))
            return widget

        if spec.type is ParamType.INT:
            widget = QSpinBox()
            widget.setRange(
                int(spec.min) if spec.min is not None else -_INT_FALLBACK,
                int(spec.max) if spec.max is not None else _INT_FALLBACK,
            )
            widget.setValue(int(value))
            widget.valueChanged.connect(lambda v, name=spec.name: self._on_change(name, v))
            return widget

        if spec.type is ParamType.FLOAT:
            widget = QDoubleSpinBox()
            widget.setRange(
                float(spec.min) if spec.min is not None else -_FLOAT_FALLBACK,
                float(spec.max) if spec.max is not None else _FLOAT_FALLBACK,
            )
            if spec.step:
                widget.setSingleStep(float(spec.step))
            widget.setValue(float(value))
            widget.valueChanged.connect(lambda v, name=spec.name: self._on_change(name, v))
            return widget

        return self._build_string_widget(spec, str(value))

    def _build_string_widget(self, spec: ParamSpec, value: str) -> QWidget:
        line_edit = QLineEdit(value)
        line_edit.textChanged.connect(lambda text, name=spec.name: self._on_change(name, text))

        if spec.name != "path":
            return line_edit

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line_edit)

        browse = QPushButton("...")
        browse.setMaximumWidth(30)

        def _browse() -> None:
            path, _filter = QFileDialog.getOpenFileName(self, "Görüntü Seç")
            if path:
                line_edit.setText(path)

        browse.clicked.connect(_browse)
        row_layout.addWidget(browse)
        return row

    def _on_change(self, name: str, value: Any) -> None:
        self._values[name] = value
        self.params_changed.emit(dict(self._values))
