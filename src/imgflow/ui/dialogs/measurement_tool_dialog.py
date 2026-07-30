"""Bağımsız "Ölçüm Aracı" penceresi: seçili adımın önizleme görüntüsü üzerinde noktalara
tıklayarak gerçek mesafe/çap/çevre ölçer.

Aktif bir yükseklik-ölçek kalibrasyonu varsa (bkz. `main_window._active_height_mm` /
`_height_scale_model`) mesafeler hem piksel hem mm cinsinden gösterilir; yoksa sadece piksel
mesafesi gösterilip kalibrasyon olmadığı açıkça belirtilir.

Birden fazla ölçüm desteklenir (`MeasureCanvas.set_multi_mode(True)`) -- `analysis.color_props`/
`analysis.texture_props`'un "Elle ROI Çiz" (RoiCanvas çoklu-ROI) desenindeki AYNI mantık: her
tamamlanan ölçüm listeye eklenir, numaralanır ("Çizgi1", "Çember1" gibi), üzerine sağ
tıklamak siler. "Ölçüm Türü" açılır listesi bir SONRAKİ ölçümün çizgi (uzunluk) mi yoksa çember
(çap/çevre) mi olacağını seçer -- gerçek kullanıcı isteği: "birden çok ölçüm (çember çap/çevre,
birden fazla çubuk/ruler)".

**Kare Yakala:** gerçek kullanıcı isteği: "ölçüm de dahil her alanda kare yakalayıp yan ekrana
atabilmek istiyorum" -- Lens/Yükseklik-Ölçek kalibrasyon diyaloglarının AYNI `capture_store`
deposuna (`source="measurement"`) yazan bir "Kareyi Galeriye Ekle" butonu buraya da eklendi;
ana pencere `frame_captured` sinyalini `capture_gallery_panel.refresh`'e bağlar (o iki
diyalogla AYNI desen)."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from imgflow.core import capture_store
from imgflow.ui.widgets.measure_canvas import MeasureCanvas

_MODE_LABELS = [("Çizgi (Uzunluk)", "LINE"), ("Çember (Çap/Çevre)", "CIRCLE")]


class MeasurementToolDialog(QDialog):
    frame_captured = Signal()

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
        self._mode_combo.addItems([label for label, _mode in _MODE_LABELS])
        self._mode_combo.setToolTip(
            "Çizgi: iki nokta arasındaki uzunluğu ölçer. Çember: ilk tık merkez, ikinci tık "
            "kenar üzerinde bir nokta -- çap/çevre hesaplanır."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self._clear_button = QPushButton("Ölçümleri Temizle")
        self._clear_button.setToolTip("Tüm çizgi/çember ölçümlerini siler.")
        self._clear_button.clicked.connect(self._canvas.clear_measurements)

        self._capture_button = QPushButton("Kareyi Galeriye Ekle")
        self._capture_button.setToolTip(
            "Şu an ölçülen görüntüyü 'Yakalanan Kareler' galerisine (yan panel) ekler."
        )
        self._capture_button.setEnabled(image is not None)
        self._capture_button.clicked.connect(self._on_capture)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Ölçüm Türü:"))
        mode_row.addWidget(self._mode_combo)
        mode_row.addWidget(self._clear_button)
        mode_row.addWidget(self._capture_button)
        mode_row.addStretch(1)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(mode_row)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._result_label)

        self._refresh_results()

    def _on_mode_changed(self, index: int) -> None:
        self._canvas.set_mode(_MODE_LABELS[index][1])

    def _on_capture(self) -> None:
        if self._image is None:
            return
        capture_store.save_capture(self._image, source="measurement")
        self.frame_captured.emit()

    def _refresh_results(self) -> None:
        lines: list[str] = []
        for i, m in enumerate(self._canvas.line_measurements(), start=1):
            line = f"Çizgi {i}: {m['distance']:.1f} px"
            if self._mm_per_px is not None:
                line += f"  ≈ {m['distance'] * self._mm_per_px:.2f} mm"
            lines.append(line)
        for i, m in enumerate(self._canvas.circle_measurements(), start=1):
            diameter_px = 2 * m["r"]
            circumference_px = 2 * math.pi * m["r"]
            line = f"Çember {i}: çap={diameter_px:.1f}px  çevre={circumference_px:.1f}px"
            if self._mm_per_px is not None:
                line += (
                    f"  ≈ çap={diameter_px * self._mm_per_px:.2f}mm"
                    f"  çevre={circumference_px * self._mm_per_px:.2f}mm"
                )
            lines.append(line)

        if not lines:
            self._result_label.setText(
                "İki nokta seçin."
                if self._mm_per_px is not None
                else "Kalibrasyon yok — sadece piksel mesafesi gösterilecek."
            )
        else:
            lines.append("(Bir ölçümün üzerine sağ tıklayarak silebilirsiniz.)")
            self._result_label.setText("\n".join(lines))
