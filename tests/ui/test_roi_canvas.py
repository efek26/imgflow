import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from imgflow.ui.widgets.roi_canvas import RoiCanvas


def _setup(qtbot, size=(100, 100), widget_size=(200, 200)):
    canvas = RoiCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(*widget_size)
    canvas.set_image(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    canvas.set_editing_enabled(True)
    return canvas


def _hover_move(canvas, pos: QPoint) -> None:
    """Buton BASILI olmayan bir üzerine-gelme (hover) hareketini doğrudan `mouseMoveEvent`'e
    gönderir. `qtbot.mouseMove` (buton basılı değilken) gerçek ekran/pencere yığını üzerinden
    yönlendiriliyor -- offscreen platformda, aynı süreçte önceki testlerden kalan başka
    gösterilmiş widget'lar varsa hangi widget'ın olayı alacağı tutarsız olabiliyor (basılı
    buton olan sürükleme olayları bunu etkilemiyor, bu yüzden diğer testler `qtbot.mouseMove`
    ile sorunsuz çalışıyor). Olayı doğrudan handler'a vermek bu belirsizliği ortadan kaldırır."""
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(pos),
        QPointF(pos),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mouseMoveEvent(event)


def test_display_rect_centers_and_scales_image(qtbot):
    canvas = _setup(qtbot, size=(100, 50), widget_size=(200, 200))
    rect = canvas._display_rect()
    # 100x50 görüntü 200x200 widget'a sığdırılınca scale=min(2,4)=2 -> 200x100, dikeyde ortalanır
    assert (rect.width(), rect.height()) == (200, 100)
    assert rect.y() == 50


def test_drawing_new_roi_emits_roi_changed_in_image_coordinates(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))  # scale=2

    received = []
    canvas.roi_changed.connect(lambda x, y, w, h: received.append((x, y, w, h)))

    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseMove(canvas, QPoint(60, 80))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 80))

    assert received == [(10, 10, 20, 30)]


def test_moving_roi_by_dragging_inside(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_roi(10, 10, 20, 20)  # widget rect = (20,20,40,40)

    received = []
    canvas.roi_changed.connect(lambda x, y, w, h: received.append((x, y, w, h)))

    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(30, 30))  # roi içinde
    qtbot.mouseMove(canvas, QPoint(50, 50))  # +20,+20 widget -> +10,+10 görüntü
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(50, 50))

    assert received[-1] == (20, 20, 20, 20)


def test_resize_via_bottom_right_handle(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_roi(10, 10, 20, 20)  # widget rect = (20,20)-(60,60)

    received = []
    canvas.roi_changed.connect(lambda x, y, w, h: received.append((x, y, w, h)))

    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))  # br handle
    qtbot.mouseMove(canvas, QPoint(80, 80))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 80))

    assert received[-1] == (10, 10, 30, 30)


def test_editing_disabled_ignores_mouse(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_editing_enabled(False)

    received = []
    canvas.roi_changed.connect(lambda *args: received.append(args))

    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseMove(canvas, QPoint(60, 80))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 80))

    assert received == []


def test_set_roi_updates_widget_rect(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_roi(5, 5, 10, 10)
    rect = canvas._roi_widget_rect()
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (10, 10, 20, 20)


def test_drawing_new_circle_emits_roi_circle_changed_in_image_coordinates(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))  # scale=2
    canvas.set_shape("CIRCLE")

    received = []
    canvas.roi_circle_changed.connect(lambda cx, cy, r: received.append((cx, cy, r)))

    # merkez=(40,40) widget'ta, sağa 30px sürüklenip bırakılıyor -> yarıçap 30 widget = 15 görüntü
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    qtbot.mouseMove(canvas, QPoint(70, 40))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(70, 40))

    assert received == [(20, 20, 15)]


def test_moving_circle_by_dragging_inside(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_shape("CIRCLE")
    canvas.set_roi_circle(30, 30, 10)  # widget merkez=(60,60), yarıçap=20

    received = []
    canvas.roi_circle_changed.connect(lambda cx, cy, r: received.append((cx, cy, r)))

    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))  # merkezde
    qtbot.mouseMove(canvas, QPoint(80, 60))  # +20 widget -> +10 görüntü
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 60))

    assert received[-1] == (40, 30, 10)


def test_resize_circle_via_edge_handle(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_shape("CIRCLE")
    canvas.set_roi_circle(30, 30, 10)  # widget merkez=(60,60), yarıçap=20, kenar=(80,60)

    received = []
    canvas.roi_circle_changed.connect(lambda cx, cy, r: received.append((cx, cy, r)))

    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 60))  # kenar tutamacı
    qtbot.mouseMove(canvas, QPoint(100, 60))  # yarıçap widget'ta 40 -> görüntüde 20
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 60))

    assert received[-1] == (30, 30, 20)


def test_hovering_over_a_measurement_box_emits_it(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))  # scale=2
    canvas.set_editing_enabled(False)  # üzerine gelme, ROI düzenleme AÇIK olmadan da çalışmalı
    canvas.set_measurements([{"bbox_x": 10, "bbox_y": 10, "bbox_w": 20, "bbox_h": 20, "label": "obj1"}])

    received = []
    canvas.hover_measurement_changed.connect(lambda m: received.append(m))

    _hover_move(canvas, QPoint(40, 40))  # widget (40,40) -> görüntü (20,20), kutunun içinde

    assert len(received) == 1
    assert received[0]["label"] == "obj1"


def test_moving_mouse_away_from_measurement_emits_none(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_editing_enabled(False)
    canvas.set_measurements([{"bbox_x": 10, "bbox_y": 10, "bbox_w": 20, "bbox_h": 20, "label": "obj1"}])

    received = []
    canvas.hover_measurement_changed.connect(lambda m: received.append(m))

    _hover_move(canvas, QPoint(40, 40))  # üzerinde
    _hover_move(canvas, QPoint(190, 190))  # uzakta

    assert [r["label"] if r else None for r in received] == ["obj1", None]


def test_leaving_widget_clears_hover(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_editing_enabled(False)
    canvas.set_measurements([{"bbox_x": 10, "bbox_y": 10, "bbox_w": 20, "bbox_h": 20, "label": "obj1"}])
    _hover_move(canvas, QPoint(40, 40))

    received = []
    canvas.hover_measurement_changed.connect(lambda m: received.append(m))

    canvas.leaveEvent(QEvent(QEvent.Type.Leave))

    assert received == [None]


def test_set_measurements_reevaluates_hover_at_last_known_mouse_position(qtbot):
    """Canlı kamera akışında `set_measurements` her tick'te (~10Hz) çağrılır; fare hareket
    ETMESE bile üzerinde durduğu nesnenin bilgisi güncel kalmalı (gerçek kullanım: kullanıcı
    fareyi bir nesnenin üzerinde sabit tutup değerlerin canlı güncellenmesini izliyor)."""
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_editing_enabled(False)
    _hover_move(canvas, QPoint(40, 40))  # görüntü (20,20) -- henüz hiçbir ölçüm yok

    received = []
    canvas.hover_measurement_changed.connect(lambda m: received.append(m))

    canvas.set_measurements([{"bbox_x": 10, "bbox_y": 10, "bbox_w": 20, "bbox_h": 20, "label": "obj1"}])

    assert len(received) == 1
    assert received[0]["label"] == "obj1"


def test_editing_disabled_ignores_mouse_for_circle(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_shape("CIRCLE")
    canvas.set_editing_enabled(False)

    received = []
    canvas.roi_circle_changed.connect(lambda *args: received.append(args))

    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    qtbot.mouseMove(canvas, QPoint(70, 40))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(70, 40))

    assert received == []
