"""ImageView'i genişletip fare ile ROI çizme/taşıma/yeniden boyutlandırma ekler.

Sadece editing_enabled=True iken (roi.region adımı seçiliyken) aktiftir. Görüntü, en-boy
oranı korunarak ortalanmış şekilde ölçekli gösterildiği için, fare tıklamalarını gerçek
görüntü piksel koordinatına çevirmek üzere kendi dönüşümünü (_display_rect) yapar.

İki ROI şekli desteklenir (set_shape ile seçilir): "RECT" (köşe tutamaçlarıyla dikdörtgen,
varsayılan/geriye dönük uyumlu davranış) ve "CIRCLE" (merkezden sürükleyerek çizilen, tek bir
kenar tutamacıyla yarıçapı değiştirilebilen daire).
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from imgflow.ui.widgets import _display_transform as dt
from imgflow.ui.widgets.image_view import ImageView

_HANDLE_HIT_RADIUS = 8
_ROI_COLOR = QColor("#33cc33")


class RoiCanvas(ImageView):
    roi_changed = Signal(int, int, int, int)  # x, y, w, h - GÖRÜNTÜ koordinatında
    roi_circle_changed = Signal(int, int, int)  # cx, cy, r - GÖRÜNTÜ koordinatında
    hover_measurement_changed = Signal(object)  # dict | None - fare bir ölçüm kutusunun üzerindeyken

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._image_size = (0, 0)  # (genişlik, yükseklik) - orijinal görüntü boyutu
        self._shape = "RECT"  # "RECT" | "CIRCLE"
        self._roi_image: tuple[int, int, int, int] | None = None
        self._roi_circle: tuple[int, int, int] | None = None  # cx, cy, r
        self._editing_enabled = False
        self._drag_mode: str | None = None  # None | "new" | "move" | "resize:tl" | "resize:r" | ...
        self._drag_anchor = QPoint()
        self._drag_start_rect = QRect()
        self._drag_start_center = QPoint()
        self._drag_start_radius = 0
        self._measurements: list[dict] = []
        """`set_measurements` ile beslenir -- `bbox_x/y/w/h` alanları GÖSTERİLEN görüntünün
        koordinat sisteminde olmalı (ör. küçültülmüş önizlemede orana göre zaten ölçeklenmiş,
        bkz. `ui/main_window.py` `_scale_measurements_for_display`)."""
        self._hover_measurement: dict | None = None
        self._last_mouse_pos: QPoint | None = None
        """Son bilinen fare pozisyonu (widget koordinatında) -- her canlı kamera tick'inde
        (~10Hz) `set_measurements` çağrıldığında, fare HİÇ HAREKET ETMESE bile üzerinde
        durduğu nesnenin bilgisini güncel tutabilmek için (gerçek OS imleç konumunu sorgulamak
        yerine bunu saklamak, headless/offscreen test ortamında da güvenilir çalışır)."""

    def set_measurements(self, measurements: list[dict] | None) -> None:
        self._measurements = measurements or []
        if self._last_mouse_pos is not None:
            self._update_hover(self._last_mouse_pos)

    # -- genel API ------------------------------------------------------

    def set_image(self, image) -> None:  # noqa: ANN001 - ImageView ile aynı imza
        super().set_image(image)
        self._image_size = (image.shape[1], image.shape[0]) if image is not None else (0, 0)
        self.update()

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing_enabled = enabled
        self.update()

    def set_shape(self, shape: str) -> None:
        self._shape = shape
        self.update()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._roi_image = (x, y, w, h)
        self.update()

    def set_roi_circle(self, cx: int, cy: int, r: int) -> None:
        self._roi_circle = (cx, cy, r)
        self.update()

    # -- koordinat dönüşümü (görüntü <-> widget) -----------------------------

    def _widget_size(self) -> tuple[int, int]:
        return (self.width(), self.height())

    def _display_rect(self) -> QRect:
        """Ölçeklenmiş görüntünün widget içindeki (ortalanmış) dikdörtgeni."""
        return dt.display_rect(self._image_size, self._widget_size())

    def _roi_widget_rect(self) -> QRect | None:
        if self._roi_image is None:
            return None
        return dt.image_rect_to_widget(self._roi_image, self._image_size, self._widget_size())

    def _widget_rect_to_image(self, rect: QRect) -> tuple[int, int, int, int]:
        return dt.widget_rect_to_image(rect, self._image_size, self._widget_size())

    def _circle_widget(self) -> tuple[QPoint, int] | None:
        if self._roi_circle is None:
            return None
        cx, cy, r = self._roi_circle
        return dt.image_circle_to_widget(cx, cy, r, self._image_size, self._widget_size())

    def _widget_circle_to_image(self, center: QPoint, radius: int) -> tuple[int, int, int]:
        return dt.widget_circle_to_image(center, radius, self._image_size, self._widget_size())

    @staticmethod
    def _dist(a: QPoint, b: QPoint) -> float:
        dx, dy = a.x() - b.x(), a.y() - b.y()
        return (dx * dx + dy * dy) ** 0.5

    def _handle_at_circle(self, pos: QPoint) -> str | None:
        widget_circle = self._circle_widget()
        if widget_circle is None:
            return None
        center, radius = widget_circle
        edge = QPoint(center.x() + radius, center.y())
        if (edge - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
            return "r"
        return None

    @staticmethod
    def _handle_points(rect: QRect) -> dict[str, QPoint]:
        return {
            "tl": rect.topLeft(),
            "tr": rect.topRight(),
            "bl": rect.bottomLeft(),
            "br": rect.bottomRight(),
        }

    def _handle_at(self, pos: QPoint) -> str | None:
        rect = self._roi_widget_rect()
        if rect is None:
            return None
        for name, hp in self._handle_points(rect).items():
            if (hp - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
                return name
        return None

    @staticmethod
    def _resize_rect(start: QRect, handle: str, delta: QPoint) -> QRect:
        r = QRect(start)
        if "t" in handle:
            r.setTop(start.top() + delta.y())
        if "b" in handle:
            r.setBottom(start.bottom() + delta.y())
        if "l" in handle:
            r.setLeft(start.left() + delta.x())
        if "r" in handle:
            r.setRight(start.right() + delta.x())
        return r.normalized()

    # -- fare olayları ----------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._editing_enabled or self._image_size == (0, 0):
            super().mousePressEvent(event)
            return
        pos = event.position().toPoint()

        if self._shape == "CIRCLE":
            self._mouse_press_circle(pos)
        else:
            self._mouse_press_rect(pos)
        self._drag_anchor = pos
        event.accept()

    def _mouse_press_rect(self, pos: QPoint) -> None:
        handle = self._handle_at(pos)
        current = self._roi_widget_rect()
        if handle is not None:
            self._drag_mode = f"resize:{handle}"
            self._drag_start_rect = current
        elif current is not None and current.contains(pos):
            self._drag_mode = "move"
            self._drag_start_rect = current
        else:
            self._drag_mode = "new"
            self._drag_start_rect = QRect(pos, pos)

    def _mouse_press_circle(self, pos: QPoint) -> None:
        handle = self._handle_at_circle(pos)
        current = self._circle_widget()
        if handle is not None:
            self._drag_mode = "resize:r"
            self._drag_start_center, self._drag_start_radius = current
        elif current is not None and self._dist(pos, current[0]) <= current[1]:
            self._drag_mode = "move"
            self._drag_start_center, self._drag_start_radius = current
        else:
            self._drag_mode = "new"
            self._drag_start_center, self._drag_start_radius = pos, 0

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        pos = event.position().toPoint()
        self._last_mouse_pos = pos
        self._update_hover(pos)

        if not self._editing_enabled or self._drag_mode is None:
            super().mouseMoveEvent(event)
            return

        if self._shape == "CIRCLE":
            self._mouse_move_circle(pos)
        else:
            self._mouse_move_rect(pos)
        self.update()
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._last_mouse_pos = None
        self._set_hover(None)
        super().leaveEvent(event)

    # -- üzerine gelme (hover) bilgisi ------------------------------------

    def _update_hover(self, pos: QPoint) -> None:
        if not self._measurements or self._image_size == (0, 0):
            self._set_hover(None)
            return
        widget_size = self._widget_size()
        for measurement in self._measurements:
            try:
                rect = dt.image_rect_to_widget(
                    (
                        measurement["bbox_x"],
                        measurement["bbox_y"],
                        measurement["bbox_w"],
                        measurement["bbox_h"],
                    ),
                    self._image_size,
                    widget_size,
                )
            except KeyError:
                continue
            if rect.contains(pos):
                self._set_hover(measurement)
                return
        self._set_hover(None)

    def _set_hover(self, measurement: dict | None) -> None:
        if measurement is self._hover_measurement:
            return
        self._hover_measurement = measurement
        self.hover_measurement_changed.emit(measurement)

    def _mouse_move_rect(self, pos: QPoint) -> None:
        delta = pos - self._drag_anchor
        if self._drag_mode == "new":
            rect = QRect(self._drag_anchor, pos).normalized()
        elif self._drag_mode == "move":
            rect = self._drag_start_rect.translated(delta)
        else:
            handle = self._drag_mode.split(":", 1)[1]
            rect = self._resize_rect(self._drag_start_rect, handle, delta)
        self._roi_image = self._widget_rect_to_image(rect)

    def _mouse_move_circle(self, pos: QPoint) -> None:
        if self._drag_mode == "new":
            center, radius = self._drag_anchor, int(self._dist(pos, self._drag_anchor))
        elif self._drag_mode == "move":
            delta = pos - self._drag_anchor
            center, radius = self._drag_start_center + delta, self._drag_start_radius
        else:  # resize:r
            center, radius = self._drag_start_center, int(self._dist(pos, self._drag_start_center))
        self._roi_circle = self._widget_circle_to_image(center, max(radius, 1))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._drag_mode is not None:
            self._drag_mode = None
            if self._shape == "CIRCLE":
                if self._roi_circle is not None:
                    self.roi_circle_changed.emit(*self._roi_circle)
            elif self._roi_image is not None:
                self.roi_changed.emit(*self._roi_image)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # -- çizim -------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if not self._editing_enabled:
            return
        painter = QPainter(self)
        painter.setPen(QPen(_ROI_COLOR, 2))
        if self._shape == "CIRCLE":
            widget_circle = self._circle_widget()
            if widget_circle is not None:
                center, radius = widget_circle
                painter.drawEllipse(center, radius, radius)
                painter.fillRect(center.x() - 3, center.y() - 3, 6, 6, _ROI_COLOR)
                edge = QPoint(center.x() + radius, center.y())
                painter.fillRect(edge.x() - 4, edge.y() - 4, 8, 8, _ROI_COLOR)
        else:
            rect = self._roi_widget_rect()
            if rect is not None:
                painter.drawRect(rect)
                for hp in self._handle_points(rect).values():
                    painter.fillRect(hp.x() - 4, hp.y() - 4, 8, 8, _ROI_COLOR)
        painter.end()
