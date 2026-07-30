"""İki noktaya tıklayarak aralarındaki piksel mesafesini ölçen genel amaçlı canvas.

Hem lens/yükseklik-ölçek kalibrasyon dialoglarında (referans nesne üzerinde bilinen mesafe
girilir) hem de bağımsız "Ölçüm Aracı" penceresinde kullanılır. Etkileşim modeli: ilk
tıklama A noktasını, ikinci tıklama B noktasını yerleştirir ve `measurement_made` sinyalini
yayar; üçüncü tıklama yeni bir A noktasıyla ölçümü sıfırlar (RoiCanvas'ın aksine sürükleme
gerekmez, sadece iki tıklama yeterlidir). Bu, kalibrasyon dialoglarının kullandığı TEK-ölçüm
(varsayılan) moddur ve `set_multi_mode(True)` çağrılmadıkça DAVRANIŞI DEĞİŞMEZ.

`set_multi_mode(True)` (SADECE bağımsız "Ölçüm Aracı" kullanır) `RoiCanvas`'ın çoklu-ROI
moduyla (`analysis.color_props`/`analysis.texture_props`'un "Elle ROI Çiz"i) AYNI deseni
uygular: her tamamlanan (A+B) ölçüm bir listeye EKLENİR, numaralanır (Çizgi1/Çember1 gibi),
üzerine SAĞ TIKLAMAK siler. `set_mode("LINE"/"CIRCLE")` bir SONRAKİ tamamlanacak ölçümün
çizgi (uzunluk) mü yoksa çember (çap/çevre, A=merkez B=kenar noktası) mı olacağını belirler.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from imgflow.ui.widgets import _display_transform as dt
from imgflow.ui.widgets.image_view import ImageView

_POINT_COLOR = QColor("#ffaa00")
_MARKER_RADIUS = 4
_DELETE_HIT_TOLERANCE = 8
"""Sağ tıklamayla silme için widget-koordinatındaki hit-test payı (px) -- `RoiCanvas`'ın
`_HANDLE_HIT_RADIUS` deseniyle AYNI büyüklük mertebesi."""


class MeasureCanvas(ImageView):
    measurement_made = Signal(float, float, float, float, float)  # x1,y1,x2,y2,pixel_distance
    measurements_changed = Signal()
    """SADECE `set_multi_mode(True)` iken anlamlı -- çizgi/çember listesi her değiştiğinde
    (ekleme/silme) yayınlanır; çağıran taraf `line_measurements()`/`circle_measurements()`
    ile TAM listeyi okuyup kendi özet/sonuç görünümünü yeniden kurar."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._image_size = (0, 0)
        self._editing_enabled = False
        self._point_a: tuple[int, int] | None = None
        self._point_b: tuple[int, int] | None = None
        self._multi_mode = False
        self._mode = "LINE"  # "LINE" | "CIRCLE" -- bir SONRAKİ tamamlanacak ölçümün türü
        self._line_measurements: list[dict] = []
        self._circle_measurements: list[dict] = []
        self._last_mouse_pos: QPoint | None = None

    def set_image(self, image) -> None:  # noqa: ANN001 - ImageView ile aynı imza
        super().set_image(image)
        self._image_size = (image.shape[1], image.shape[0]) if image is not None else (0, 0)
        self.update()

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing_enabled = enabled
        self.update()

    def set_multi_mode(self, enabled: bool) -> None:
        self._multi_mode = enabled
        self.update()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.update()

    def clear_points(self) -> None:
        self._point_a = None
        self._point_b = None
        self.update()

    def clear_measurements(self) -> None:
        self._line_measurements = []
        self._circle_measurements = []
        self._point_a = None
        self._point_b = None
        self.update()
        self.measurements_changed.emit()

    def line_measurements(self) -> list[dict]:
        return list(self._line_measurements)

    def circle_measurements(self) -> list[dict]:
        return list(self._circle_measurements)

    def _widget_size(self) -> tuple[int, int]:
        return (self.width(), self.height())

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._editing_enabled or self._image_size == (0, 0):
            super().mousePressEvent(event)
            return
        pos = event.position().toPoint()

        if self._multi_mode and event.button() == Qt.MouseButton.RightButton:
            self._delete_measurement_near(pos)
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
                if self._mode == "CIRCLE":
                    self._circle_measurements.append(
                        {"cx": float(x1), "cy": float(y1), "r": max(1.0, distance)}
                    )
                else:
                    self._line_measurements.append(
                        {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "distance": distance}
                    )
                # Yeni bir ölçüme HEMEN başlanabilsin diye (3. tıklamayı beklemeden) sıfırlanır
                # -- tek-ölçüm (kalibrasyon) modunda BUNU YAPMAYIZ, o mod 3. tıklamada sıfırlar.
                self._point_a = None
                self._point_b = None
                self.measurements_changed.emit()

        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._last_mouse_pos = event.position().toPoint()
        if self._multi_mode and self._point_a is not None and self._point_b is None:
            self.update()
        super().mouseMoveEvent(event)

    @staticmethod
    def _point_to_segment_distance(p: QPoint, a: QPoint, b: QPoint) -> float:
        ax, ay, bx, by, px, py = a.x(), a.y(), b.x(), b.y(), p.x(), p.y()
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        proj_x, proj_y = ax + t * dx, ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)

    def _widget_circle_radius(self, cx: float, cy: float, r: float) -> tuple[QPoint, int]:
        widget_size = self._widget_size()
        wc = dt.image_point_to_widget((cx, cy), self._image_size, widget_size)
        we = dt.image_point_to_widget((cx + r, cy), self._image_size, widget_size)
        return wc, int(round(math.hypot(we.x() - wc.x(), we.y() - wc.y())))

    def _delete_measurement_near(self, pos: QPoint) -> None:
        widget_size = self._widget_size()
        for i in reversed(range(len(self._line_measurements))):
            m = self._line_measurements[i]
            wa = dt.image_point_to_widget((m["x1"], m["y1"]), self._image_size, widget_size)
            wb = dt.image_point_to_widget((m["x2"], m["y2"]), self._image_size, widget_size)
            if self._point_to_segment_distance(pos, wa, wb) <= _DELETE_HIT_TOLERANCE:
                del self._line_measurements[i]
                self.update()
                self.measurements_changed.emit()
                return
        for i in reversed(range(len(self._circle_measurements))):
            m = self._circle_measurements[i]
            wc, widget_r = self._widget_circle_radius(m["cx"], m["cy"], m["r"])
            dist_to_center = math.hypot(pos.x() - wc.x(), pos.y() - wc.y())
            if abs(dist_to_center - widget_r) <= _DELETE_HIT_TOLERANCE:
                del self._circle_measurements[i]
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

        if self._multi_mode:
            for i, m in enumerate(self._line_measurements, start=1):
                wa = dt.image_point_to_widget((m["x1"], m["y1"]), self._image_size, widget_size)
                wb = dt.image_point_to_widget((m["x2"], m["y2"]), self._image_size, widget_size)
                painter.drawLine(wa, wb)
                painter.drawText(QPoint((wa.x() + wb.x()) // 2, (wa.y() + wb.y()) // 2), f"Çizgi{i}")
            for i, m in enumerate(self._circle_measurements, start=1):
                wc, widget_r = self._widget_circle_radius(m["cx"], m["cy"], m["r"])
                painter.drawEllipse(wc, widget_r, widget_r)
                painter.drawText(QPoint(wc.x() + widget_r, wc.y()), f"Çember{i}")

        if self._point_a is not None:
            wa = dt.image_point_to_widget(self._point_a, self._image_size, widget_size)
            painter.fillRect(
                wa.x() - _MARKER_RADIUS, wa.y() - _MARKER_RADIUS, 2 * _MARKER_RADIUS, 2 * _MARKER_RADIUS, _POINT_COLOR
            )
            # Kapanmamış ölçüm için canlı "kauçuk bant" -- `RoiCanvas`'ın poligon modundaki
            # AYNI desen (son nokta ile imleç arasında/çemberde önizleme).
            if self._multi_mode and self._point_b is None and self._last_mouse_pos is not None:
                if self._mode == "CIRCLE":
                    live_r = int(round(math.hypot(
                        self._last_mouse_pos.x() - wa.x(), self._last_mouse_pos.y() - wa.y()
                    )))
                    painter.drawEllipse(wa, live_r, live_r)
                else:
                    painter.drawLine(wa, self._last_mouse_pos)
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
