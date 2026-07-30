"""Bağımsız "Ölçüm Aracı" penceresi: seçili adımın önizleme görüntüsü üzerinde tıklayarak
gerçek mesafe/çember ölçer.

Aktif bir yükseklik-ölçek kalibrasyonu varsa (bkz. `main_window._active_height_mm` /
`_height_scale_model`) mesafe hem piksel hem mm cinsinden gösterilir; yoksa sadece piksel
mesafesi gösterilip kalibrasyon olmadığı açıkça belirtilir.

Canvas ÇOKLU ölçüm modunda çalışır (bkz. `ui/widgets/measure_canvas.py`): arka arkaya birden
çok çizgi/çember ölçülüp hepsi numaralı satırlar halinde listelenir, bir ölçümün üzerine sağ
tıklayınca o ölçüm silinir. Gerçek kullanıcı isteği: "birden çok ölçüm (çember çap/çevre,
birden fazla çubuk/ruler)".
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from imgflow.core import capture_store
from imgflow.ui.widgets.measure_canvas import MeasureCanvas

_NO_CALIBRATION_TEXT = "Kalibrasyon yok — sadece piksel mesafesi gösterilecek."
_EMPTY_TEXT = "İki nokta seçin."


class MeasurementToolDialog(QDialog):
    frame_captured = Signal()
    """Ölçüm penceresindeki kare `capture_store`'a kaydedilince yayınlanır -- `main_window.py`
    bunu yakalayıp yakalananlar galerisini tazeler (Lens/Yükseklik-Ölçek dialoglarıyla AYNI
    desen). Gerçek kullanıcı isteği: "ölçüm de dahil her alanda kare yakalayıp yan ekrana
    atabilmek istiyorum"."""

    def __init__(self, image: np.ndarray | None, mm_per_px: float | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ölçüm Aracı")
        self.setModal(False)
        self._mm_per_px = mm_per_px
        self._image = image

        self._canvas = MeasureCanvas()
        self._canvas.set_editing_enabled(True)
        self._canvas.set_multi_mode(True)
        if image is not None:
            self._canvas.set_image(image)
        self._canvas.measurements_changed.connect(self._refresh_results)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Çizgi (Mesafe)", "LINE")
        self._mode_combo.addItem("Çember (Çap/Çevre)", "CIRCLE")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._clear_button = QPushButton("Ölçümleri Temizle")
        self._clear_button.clicked.connect(self._canvas.clear_measurements)

        self._capture_button = QPushButton("Kareyi Yakala")
        self._capture_button.setEnabled(image is not None)
        self._capture_button.clicked.connect(self._on_capture)

        button_row = QHBoxLayout()
        button_row.addWidget(QLabel("Ölçüm tipi:"))
        button_row.addWidget(self._mode_combo, 1)
        button_row.addWidget(self._clear_button)
        button_row.addWidget(self._capture_button)

        self._result_label = QLabel(_EMPTY_TEXT if mm_per_px is not None else _NO_CALIBRATION_TEXT)
        self._result_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas, 1)
        layout.addLayout(button_row)
        layout.addWidget(self._result_label)

    def _on_mode_changed(self, index: int) -> None:
        self._canvas.set_mode(self._mode_combo.itemData(index))

    def _on_capture(self) -> None:
        if self._image is None:
            return
        capture_store.save_capture(self._image, source="measurement")
        self.frame_captured.emit()

    def _format_length(self, pixels: float) -> str:
        """Kalibrasyon varsa "px ≈ mm", yoksa sadece "px" -- mm YOKKEN metinde "mm" geçmez."""
        if self._mm_per_px is None:
            return f"{pixels:.1f} px"
        return f"{pixels:.1f} px ≈ {pixels * self._mm_per_px:.2f} mm"

    def _refresh_results(self) -> None:
        lines = self._canvas.line_measurements()
        circles = self._canvas.circle_measurements()
        if not lines and not circles:
            self._result_label.setText(_EMPTY_TEXT if self._mm_per_px is not None else _NO_CALIBRATION_TEXT)
            return

        rows = [f"Çizgi {i}: {self._format_length(m['distance'])}" for i, m in enumerate(lines, start=1)]
        rows += [
            f"Çember {i}: çap={self._format_length(2 * m['r'])}  "
            f"çevre={self._format_length(2 * math.pi * m['r'])}"
            for i, m in enumerate(circles, start=1)
        ]
        self._result_label.setText("\n".join(rows))
