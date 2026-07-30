import numpy as np
from PySide6.QtCore import QPoint, Qt

from imgflow.ui.widgets.measure_canvas import MeasureCanvas


def _setup(qtbot, size=(100, 100), widget_size=(200, 200)):
    canvas = MeasureCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(*widget_size)
    canvas.set_image(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    canvas.set_editing_enabled(True)
    return canvas


def test_two_clicks_emit_measurement_in_image_coordinates(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))  # scale=2

    received = []
    canvas.measurement_made.connect(lambda *args: received.append(args))

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))

    assert len(received) == 1
    x1, y1, x2, y2, distance = received[0]
    assert (x1, y1) == (10.0, 10.0)
    assert (x2, y2) == (30.0, 10.0)
    assert distance == 20.0


def test_third_click_starts_a_new_measurement(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))

    received = []
    canvas.measurement_made.connect(lambda *args: received.append(args))

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))  # yeni A noktası

    assert len(received) == 1
    assert canvas._point_a == (20, 20)
    assert canvas._point_b is None


def test_editing_disabled_ignores_clicks(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_editing_enabled(False)

    received = []
    canvas.measurement_made.connect(lambda *args: received.append(args))

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))

    assert received == []


def test_clear_points_resets_state(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))

    canvas.clear_points()

    assert canvas._point_a is None
    assert canvas._point_b is None


def test_multi_mode_appends_line_measurements_and_resets_for_next(qtbot):
    """Gerçek kullanıcı isteği: "birden çok ölçüm (çember çap/çevre, birden fazla çubuk/
    ruler)" -- `RoiCanvas`'ın çoklu-ROI deseniyle AYNI: her tamamlanan ölçüm listeye eklenir,
    3. tıklamayı BEKLEMEDEN hemen yeni bir ölçüme başlanabilir (tek-ölçüm modun aksine)."""
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))  # scale=2
    canvas.set_multi_mode(True)
    changes = []
    canvas.measurements_changed.connect(lambda: changes.append(1))

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))
    assert len(changes) == 1
    assert canvas._point_a is None and canvas._point_b is None  # hemen sıfırlandı

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(30, 30))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(30, 50))

    lines = canvas.line_measurements()
    assert len(lines) == 2
    assert lines[0]["distance"] == 20.0
    assert lines[1]["distance"] == 10.0


def test_multi_mode_circle_mode_records_center_and_radius(qtbot):
    """Çember modunda ilk tık merkez, ikinci tık kenar noktasıdır -- yarıçap ikisi
    arasındaki mesafedir."""
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))  # scale=2
    canvas.set_multi_mode(True)
    canvas.set_mode("CIRCLE")

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))  # görüntüde (20,20)
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 40))  # görüntüde (40,20)

    circles = canvas.circle_measurements()
    assert len(circles) == 1
    assert circles[0]["cx"] == 20.0
    assert circles[0]["cy"] == 20.0
    assert circles[0]["r"] == 20.0


def test_multi_mode_right_click_deletes_nearest_line(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_multi_mode(True)
    changes = []
    canvas.measurements_changed.connect(lambda: changes.append(1))

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))
    assert len(canvas.line_measurements()) == 1

    qtbot.mouseClick(canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 20))  # çizginin ortası

    assert canvas.line_measurements() == []
    assert len(changes) == 2  # ekleme + silme


def test_multi_mode_right_click_far_from_any_measurement_does_nothing(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_multi_mode(True)
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))

    qtbot.mouseClick(canvas, Qt.MouseButton.RightButton, pos=QPoint(150, 150))

    assert len(canvas.line_measurements()) == 1


def test_clear_measurements_empties_both_lists(qtbot):
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))
    canvas.set_multi_mode(True)
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))
    canvas.set_mode("CIRCLE")
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 40))
    assert canvas.line_measurements() and canvas.circle_measurements()

    canvas.clear_measurements()

    assert canvas.line_measurements() == []
    assert canvas.circle_measurements() == []


def test_single_shot_mode_unaffected_by_multi_mode_being_off(qtbot):
    """Varsayılan (tek-ölçüm, kalibrasyon dialoglarının kullandığı) modda `set_multi_mode`
    HİÇ çağrılmadığından listeler boş kalmalı ve eski 3.-tıklama-sıfırlar davranışı AYNEN
    korunmalı -- bu test geriye dönük uyumluluğun regresyona uğramadığını kanıtlar."""
    canvas = _setup(qtbot, size=(100, 100), widget_size=(200, 200))

    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))

    assert canvas.line_measurements() == []
    assert canvas._point_a == (10, 10) and canvas._point_b == (30, 10)
