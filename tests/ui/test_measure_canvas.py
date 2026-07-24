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
