import numpy as np
from PySide6.QtCore import QPoint, Qt

from imgflow.ui.dialogs.measurement_tool_dialog import MeasurementToolDialog

_IMAGE = np.zeros((100, 100, 3), dtype=np.uint8)


def test_shows_calibration_warning_when_no_scale(qtbot):
    dialog = MeasurementToolDialog(_IMAGE, None)
    qtbot.addWidget(dialog)

    assert "Kalibrasyon yok" in dialog._result_label.text()


def test_measurement_shows_px_only_without_scale(qtbot):
    dialog = MeasurementToolDialog(_IMAGE, None)
    qtbot.addWidget(dialog)
    dialog._canvas.resize(200, 200)

    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))

    assert "px" in dialog._result_label.text()
    assert "mm" not in dialog._result_label.text()


def test_measurement_shows_mm_with_active_scale(qtbot):
    dialog = MeasurementToolDialog(_IMAGE, mm_per_px=0.5)
    qtbot.addWidget(dialog)
    dialog._canvas.resize(200, 200)

    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))

    assert "mm" in dialog._result_label.text()


def test_none_image_leaves_canvas_without_image(qtbot):
    dialog = MeasurementToolDialog(None, None)
    qtbot.addWidget(dialog)

    assert dialog._canvas._image_size == (0, 0)
