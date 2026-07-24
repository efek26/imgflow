"""İki noktaya tıklayarak aralarındaki piksel mesafesini ölçen genel amaçlı canvas.

Hem lens/yükseklik-ölçek kalibrasyon dialoglarında (referans nesne üzerinde bilinen mesafe
girilir) hem de bağımsız "Ölçüm Aracı" penceresinde kullanılır. Etkileşim modeli: ilk
tıklama A noktasını, ikinci tıklama B noktasını yerleştirir ve `measurement_made` sinyalini
yayar; üçüncü tıklama yeni bir A noktasıyla ölçümü sıfırlar (RoiCanvas'ın aksine sürükleme
gerekmez, sadece iki tıklama yeterlidir).
"""

from __future__ import annotations

import math

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from imgflow.ui.widgets import _display_transform as dt
from imgflow.ui.widgets.image_view import ImageView

_POINT_COLOR = QColor("#ffaa00")
_MARKER_RADIUS = 4


class MeasureCanvas(ImageView):
    measurement_made = Signal(float, float, float, float, float)  # x1,y1,x2,y2,pixel_distance

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._image_size = (0, 0)
        self._editing_enabled = False
        self._point_a: tuple[int, int] | None = None
        self._point_b: tuple[int, int] | None = None

    def set_image(self, image) -> None:  # noqa: ANN001 - ImageView ile aynı imza
        super().set_image(image)
        self._image_size = (image.shape[1], image.shape[0]) if image is not None else (0, 0)
        self.update()

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing_enabled = enabled
        self.update()

    def clear_points(self) -> None:
        self._point_a = None
        self._point_b = None
        self.update()

    def _widget_size(self) -> tuple[int, int]:
        return (self.width(), self.height())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._editing_enabled or self._image_size == (0, 0):
            super().mousePressEvent(event)
            return
        pos = event.position().toPoint()
        image_point = dt.widget_point_to_image(pos, self._image_size, self._widget_size())

        if self._point_a is None or self._point_b is not None:
            self._point_a = image_point
            self._point_b = None
        else:
            self._point_b = image_point
            x1, y1 = self._point_a
            x2, y2 = self._point_b
            distance = math.hypot(x2 - x1, y2 - y1)
            self.measurement_made.emit(float(x1), float(y1), float(x2), float(y2), distance)

        self.update()
        event.accept()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if not self._editing_enabled:
            return
        painter = QPainter(self)
        painter.setPen(QPen(_POINT_COLOR, 2))
        widget_size = self._widget_size()
        if self._point_a is not None:
            wa = dt.image_point_to_widget(self._point_a, self._image_size, widget_size)
            painter.fillRect(
                wa.x() - _MARKER_RADIUS, wa.y() - _MARKER_RADIUS, 2 * _MARKER_RADIUS, 2 * _MARKER_RADIUS, _POINT_COLOR
            )
        if self._point_b is not None:
            wb = dt.image_point_to_widget(self._point_b, self._image_size, widget_size)
            painter.fillRect(
                wb.x() - _MARKER_RADIUS, wb.y() - _MARKER_RADIUS, 2 * _MARKER_RADIUS, 2 * _MARKER_RADIUS, _POINT_COLOR
            )
        if self._point_a is not None and self._point_b is not None:
            wa = dt.image_point_to_widget(self._point_a, self._image_size, widget_size)
            wb = dt.image_point_to_widget(self._point_b, self._image_size, widget_size)
            painter.drawLine(wa, wb)
        painter.end()
