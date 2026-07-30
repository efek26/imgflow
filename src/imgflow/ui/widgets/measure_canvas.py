"""İki noktaya tıklayarak aralarındaki piksel mesafesini ölçen genel amaçlı canvas.

Hem lens/yükseklik-ölçek kalibrasyon dialoglarında (referans nesne üzerinde bilinen mesafe
girilir) hem de bağımsız "Ölçüm Aracı" penceresinde kullanılır. Etkileşim modeli: ilk
tıklama A noktasını, ikinci tıklama B noktasını yerleştirir ve `measurement_made` sinyalini
yayar; üçüncü tıklama yeni bir A noktasıyla ölçümü sıfırlar (RoiCanvas'ın aksine sürükleme
gerekmez, sadece iki tıklama yeterlidir).

`set_multi_mode(True)` ile AYRI bir "çoklu ölçüm" modu vardır (`RoiCanvas`'ın çoklu-ROI
deseniyle AYNI, bkz. `ui/widgets/roi_canvas.py`): tamamlanan her ölçüm bir listeye eklenir ve
A/B noktaları HEMEN sıfırlanır -- üçüncü tıklamayı beklemeden arka arkaya birden çok ölçüm
yapılabilir; bir ölçümün ÜZERİNE SAĞ TIKLAMAK onu siler. Gerçek kullanıcı isteği: "birden çok
ölçüm (çember çap/çevre, birden fazla çubuk/ruler)". `set_mode("CIRCLE")` ile ölçüm birimi
çizgi yerine çember olur: ilk tık MERKEZ, ikinci tık KENAR noktasıdır (yarıçap ikisinin
arasındaki mesafe). Kalibrasyon dialogları `set_multi_mode`'u HİÇ çağırmaz, dolayısıyla eski
tek-ölçüm davranışı aynen korunur.
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from imgflow.ui.widgets import _display_transform as dt
from imgflow.ui.widgets.image_view import ImageView

_POINT_COLOR = QColor("#ffaa00")
_MARKER_RADIUS = 4
_DELETE_HIT_RADIUS = 8
"""Çoklu modda sağ tıklamanın bir ölçümü silmesi için ona WIDGET koordinatında bu kadar
piksel yakın olması gerekir (`RoiCanvas._HANDLE_HIT_RADIUS` ile aynı his) -- boşluğa sağ
tıklamak yanlışlıkla uzaktaki bir ölçümü silmesin."""


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """(px,py) noktasının [A,B] DOĞRU PARÇASINA (sonsuz doğruya değil) uzaklığı."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class MeasureCanvas(ImageView):
    measurement_made = Signal(float, float, float, float, float)  # x1,y1,x2,y2,pixel_distance
    measurements_changed = Signal()  # çoklu modda liste her değiştiğinde (ekleme/silme/temizleme)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._image_size = (0, 0)
        self._editing_enabled = False
        self._point_a: tuple[int, int] | None = None
        self._point_b: tuple[int, int] | None = None
        self._multi_mode = False
        self._mode = "LINE"  # "LINE" | "CIRCLE"
        self._lines: list[dict[str, float]] = []
        self._circles: list[dict[str, float]] = []

    # -- çoklu ölçüm API'si ---------------------------------------------

    def set_multi_mode(self, enabled: bool) -> None:
        self._multi_mode = enabled
        self.clear_points()

    def set_mode(self, mode: str) -> None:
        """"LINE" (iki nokta arası mesafe) | "CIRCLE" (merkez + kenar noktası)."""
        self._mode = mode
        self.clear_points()

    def line_measurements(self) -> list[dict[str, float]]:
        return list(self._lines)

    def circle_measurements(self) -> list[dict[str, float]]:
        return list(self._circles)

    def clear_measurements(self) -> None:
        self._lines = []
        self._circles = []
        self.clear_points()
        self.measurements_changed.emit()

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

        if self._multi_mode and event.button() == Qt.MouseButton.RightButton:
            self._delete_measurement_at(pos)
            event.accept()
            return

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
            if self._multi_mode:
                self._append_measurement(float(x1), float(y1), float(x2), float(y2), distance)
                # Sonraki ölçüme HEMEN başlanabilsin diye noktalar sıfırlanır (tek-ölçüm
                # modundaki "3. tıklama sıfırlar" davranışının çoklu moddaki karşılığı).
                self._point_a = None
                self._point_b = None

        self.update()
        event.accept()

    def _append_measurement(self, x1: float, y1: float, x2: float, y2: float, distance: float) -> None:
        if self._mode == "CIRCLE":
            # İlk tık merkez, ikinci tık kenar noktası -> yarıçap ikisi arasındaki mesafe.
            self._circles.append({"cx": x1, "cy": y1, "r": distance})
        else:
            self._lines.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "distance": distance})
        self.measurements_changed.emit()

    def _delete_measurement_at(self, pos) -> None:  # noqa: ANN001 - QPoint
        widget_size = self._widget_size()

        for index, line in enumerate(self._lines):
            wa = dt.image_point_to_widget((int(line["x1"]), int(line["y1"])), self._image_size, widget_size)
            wb = dt.image_point_to_widget((int(line["x2"]), int(line["y2"])), self._image_size, widget_size)
            if _point_to_segment_distance(pos.x(), pos.y(), wa.x(), wa.y(), wb.x(), wb.y()) <= _DELETE_HIT_RADIUS:
                del self._lines[index]
                self.update()
                self.measurements_changed.emit()
                return

        for index, circle in enumerate(self._circles):
            center = dt.image_point_to_widget(
                (int(circle["cx"]), int(circle["cy"])), self._image_size, widget_size
            )
            radius = circle["r"] * dt.display_scale(self._image_size, widget_size)
            reach = math.hypot(pos.x() - center.x(), pos.y() - center.y())
            if abs(reach - radius) <= _DELETE_HIT_RADIUS:  # çemberin ÇEVRESİNE yakın tıklama
                del self._circles[index]
                self.update()
                self.measurements_changed.emit()
                return

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if not self._editing_enabled:
            return
        painter = QPainter(self)
        painter.setPen(QPen(_POINT_COLOR, 2))
        widget_size = self._widget_size()
        for line in self._lines:
            wa = dt.image_point_to_widget((int(line["x1"]), int(line["y1"])), self._image_size, widget_size)
            wb = dt.image_point_to_widget((int(line["x2"]), int(line["y2"])), self._image_size, widget_size)
            painter.drawLine(wa, wb)
        for circle in self._circles:
            center = dt.image_point_to_widget(
                (int(circle["cx"]), int(circle["cy"])), self._image_size, widget_size
            )
            radius = int(circle["r"] * dt.display_scale(self._image_size, widget_size))
            painter.drawEllipse(center, radius, radius)
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
