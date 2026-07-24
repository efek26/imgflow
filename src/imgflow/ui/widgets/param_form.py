"""ParamSpec listesinden otomatik form üreten widget.

min/max tanımlı sayısal parametreler için spinbox'ın yanına senkronize bir kaydırma
çubuğu (QSlider) eklenir; sınırsız (max=None) parametrelerde slider gösterilmez.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.params import ParamSpec, ParamType

_INT_FALLBACK = 1_000_000
_FLOAT_FALLBACK = 1.0e6
_FLOAT_SLIDER_SCALE = 100


class ParamForm(QWidget):
    params_changed = Signal(dict)
    field_changed = Signal(str, object)  # name, value - tekil alan değişimi (canlı donanım için)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._specs: list[ParamSpec] = []
        self._values: dict[str, Any] = {}
        self._widgets: dict[str, QWidget] = {}
        self._suppress_emit = False

    def set_params(self, specs: list[ParamSpec], values: dict[str, Any]) -> None:
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._specs = list(specs)
        self._values = dict(values)
        self._widgets = {}

        for spec in self._specs:
            control, container = self._build_widget(spec)
            self._widgets[spec.name] = control
            if spec.readonly:
                container.setEnabled(False)
            label = QLabel(spec.label or spec.name)
            if spec.help:
                label.setToolTip(spec.help)
                container.setToolTip(spec.help)
            self._layout.addRow(label, container)

    def values(self) -> dict[str, Any]:
        return dict(self._values)

    def widget_for(self, name: str) -> QWidget | None:
        return self._widgets.get(name)

    def set_value(self, name: str, value: Any) -> None:
        """Formu yeniden kurmadan TEK bir alanın gösterilen değerini günceller (donanımın
        clamp'lediği gerçek değeri geri yansıtmak için). `params_changed`/`field_changed`
        sinyallerini TEKRAR yaymaz — aksi halde donanıma gereksiz bir yazma daha tetiklenirdi."""
        control = self._widgets.get(name)
        if control is None:
            return
        self._suppress_emit = True
        try:
            if isinstance(control, QCheckBox):
                control.setChecked(bool(value))
            elif isinstance(control, QComboBox):
                control.setCurrentText(str(value))
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.setValue(value)
            elif isinstance(control, QLineEdit):
                control.setText(str(value))
        finally:
            self._suppress_emit = False
        self._values[name] = value

    def _build_widget(self, spec: ParamSpec) -> tuple[QWidget, QWidget]:
        value = self._values.get(spec.name, spec.default)

        if spec.type is ParamType.BOOL:
            widget = QCheckBox()
            widget.setChecked(bool(value))
            widget.toggled.connect(lambda checked, name=spec.name: self._on_change(name, checked))
            return widget, widget

        if spec.type is ParamType.ENUM:
            widget = QComboBox()
            widget.addItems(spec.choices or [])
            if value in (spec.choices or []):
                widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(lambda text, name=spec.name: self._on_change(name, text))
            return widget, widget

        if spec.type is ParamType.INT:
            return self._build_int_widget(spec, value)

        if spec.type is ParamType.FLOAT:
            return self._build_float_widget(spec, value)

        return self._build_string_widget(spec, str(value))

    def _build_int_widget(self, spec: ParamSpec, value: Any) -> tuple[QWidget, QWidget]:
        spin = QSpinBox()
        lo = int(spec.min) if spec.min is not None else -_INT_FALLBACK
        hi = int(spec.max) if spec.max is not None else _INT_FALLBACK
        spin.setRange(lo, hi)
        spin.setValue(int(value))
        spin.valueChanged.connect(lambda v, name=spec.name: self._on_change(name, v))

        if spec.min is None or spec.max is None:
            return spin, spin

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(int(value))
        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)

        return spin, self._slider_row(slider, spin)

    def _build_float_widget(self, spec: ParamSpec, value: Any) -> tuple[QWidget, QWidget]:
        spin = QDoubleSpinBox()
        lo = float(spec.min) if spec.min is not None else -_FLOAT_FALLBACK
        hi = float(spec.max) if spec.max is not None else _FLOAT_FALLBACK
        spin.setRange(lo, hi)
        if spec.step:
            spin.setSingleStep(float(spec.step))
        spin.setValue(float(value))
        spin.valueChanged.connect(lambda v, name=spec.name: self._on_change(name, v))

        if spec.min is None or spec.max is None:
            return spin, spin

        scale = _FLOAT_SLIDER_SCALE
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(lo * scale), int(hi * scale))
        slider.setValue(int(float(value) * scale))
        slider.valueChanged.connect(lambda v: spin.setValue(v / scale))
        spin.valueChanged.connect(lambda v: slider.setValue(int(v * scale)))

        return spin, self._slider_row(slider, spin)

    @staticmethod
    def _slider_row(slider: QSlider, spin: QWidget) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider, 1)
        row_layout.addWidget(spin)
        return row

    def _build_string_widget(self, spec: ParamSpec, value: str) -> tuple[QWidget, QWidget]:
        if spec.dynamic_choices is not None and spec.multi_select:
            return self._build_multi_select_widget(spec, value)
        if spec.dynamic_choices is not None:
            return self._build_dynamic_choice_widget(spec, value)

        line_edit = QLineEdit(value)
        line_edit.textChanged.connect(lambda text, name=spec.name: self._on_change(name, text))

        if spec.name != "path":
            return line_edit, line_edit

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
        return line_edit, row

    def _build_dynamic_choice_widget(self, spec: ParamSpec, value: str) -> tuple[QWidget, QWidget]:
        """`spec.dynamic_choices()` ile GÜNCEL bir seçenek listesi üreten, ama serbest metin
        girişine de izin veren (kayıtlı olmayan bir isim de girilebilir) düzenlenebilir açılır
        kutu — tek seçim içindir; birden fazla ismin işaretlenebildiği durum için bkz.
        `_build_multi_select_widget` (`spec.multi_select=True`)."""
        combo = QComboBox()
        combo.setEditable(True)
        choices = spec.dynamic_choices() if spec.dynamic_choices else []
        combo.addItems(choices)
        if value and value not in choices:
            combo.addItem(value)
        if value:
            combo.setCurrentText(value)
        combo.currentTextChanged.connect(lambda text, name=spec.name: self._on_change(name, text))
        return combo, combo

    def _build_multi_select_widget(self, spec: ParamSpec, value: str) -> tuple[QWidget, QWidget]:
        """Birden fazla ismin işaretlenebildiği bir kontrol listesi popup'ı açan buton + seçili
        isimleri gösteren DÜZENLENEBİLİR bir alan — ör. `geom.shape_match`'in `model_names`
        parametresi, aynı adımda birden fazla kayıtlı şekil modelinin seçilebilmesi için.
        Alan bilerek salt-okunur DEĞİL: kayıtlı olmayan/henüz kaydedilmemiş bir model adını da
        elle virgülle ayırarak yazabilmek gerekir (bkz. operatörün `help` metni)."""
        line_edit = QLineEdit(value)
        line_edit.textChanged.connect(lambda text, name=spec.name: self._on_change(name, text))

        pick_button = QPushButton("Seç...")

        def _pick() -> None:
            choices = spec.dynamic_choices() if spec.dynamic_choices else []
            current = {v.strip() for v in line_edit.text().split(",") if v.strip()}
            selected = self._prompt_multi_select(spec.label or spec.name, choices, current)
            if selected is None:
                return
            joined = ", ".join(selected)
            # `setText` sırasında sinyal bilerek bastırılıyor: metin FİİLEN değişmese bile
            # (ör. seçim onaylanıp aynı isimler bırakıldığında) `_on_change` TAM OLARAK BİR
            # kez tetiklensin isteniyor — hem `textChanged` hem bu açık çağrı ateşlenirse aynı
            # değer iki kez yayınlanır.
            line_edit.blockSignals(True)
            line_edit.setText(joined)
            line_edit.blockSignals(False)
            self._on_change(spec.name, joined)

        pick_button.clicked.connect(_pick)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(pick_button)
        return line_edit, row

    def _prompt_multi_select(self, title: str, choices: list[str], current: set[str]) -> list[str] | None:
        """`choices` içindeki her isim için bir onay kutusu (`current`'takiler önceden
        işaretli) gösteren küçük bir dialog açar; Tamam'a basılırsa işaretli isimlerin
        listesini, İptal'e basılırsa `None` döner."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        checkboxes: list[QCheckBox] = []
        for choice in choices:
            box = QCheckBox(choice)
            box.setChecked(choice in current)
            layout.addWidget(box)
            checkboxes.append(box)
        if not choices:
            layout.addWidget(QLabel("Kayıtlı seçenek yok."))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return [box.text() for box in checkboxes if box.isChecked()]

    def _on_change(self, name: str, value: Any) -> None:
        self._values[name] = value
        if self._suppress_emit:
            return
        self.params_changed.emit(dict(self._values))
        self.field_changed.emit(name, value)
