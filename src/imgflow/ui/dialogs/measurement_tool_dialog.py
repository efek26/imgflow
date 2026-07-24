"""Bağımsız "Ölçüm Aracı" penceresi: seçili adımın önizleme görüntüsü üzerinde iki noktaya
tıklayarak gerçek mesafe ölçer.

Aktif bir yükseklik-ölçek kalibrasyonu varsa (bkz. `main_window._active_height_mm` /
`_height_scale_model`) mesafe hem piksel hem mm cinsinden gösterilir; yoksa sadece piksel
mesafesi gösterilip kalibrasyon olmadığı açıkça belirtilir.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from imgflow.ui.widgets.measure_canvas import MeasureCanvas


class MeasurementToolDialog(QDialog):
    def __init__(self, image: np.ndarray | None, mm_per_px: float | None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Ölçüm Aracı")
        self.setModal(False)
        self._mm_per_px = mm_per_px

        self._canvas = MeasureCanvas()
        self._canvas.set_editing_enabled(True)
        if image is not None:
            self._canvas.set_image(image)
        self._canvas.measurement_made.connect(self._on_measurement_made)

        self._result_label = QLabel(
            "İki nokta seçin."
            if mm_per_px is not None
            else "Kalibrasyon yok — sadece piksel mesafesi gösterilecek."
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._result_label)

    def _on_measurement_made(
        self, _x1: float, _y1: float, _x2: float, _y2: float, pixel_distance: float
    ) -> None:
        if self._mm_per_px is not None:
            mm = pixel_distance * self._mm_per_px
            self._result_label.setText(f"Mesafe: {pixel_distance:.1f} px  ≈  {mm:.2f} mm")
        else:
            self._result_label.setText(f"Mesafe: {pixel_distance:.1f} px (kalibrasyon yok)")
