import numpy as np
from PySide6.QtCore import QPoint, Qt

from imgflow.core import capture_store
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


def test_multiple_line_measurements_are_numbered_and_listed(qtbot):
    """Gerçek kullanıcı isteği: "birden fazla çubuk/ruler" -- iki ayrı çizgi çizilince
    ikisi de sonuç etiketinde numaralı ayrı satırlar olarak görünmeli."""
    dialog = MeasurementToolDialog(_IMAGE, mm_per_px=0.5)
    qtbot.addWidget(dialog)
    dialog._canvas.resize(200, 200)

    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 40))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 80))

    text = dialog._result_label.text()
    assert "Çizgi 1" in text
    assert "Çizgi 2" in text
    assert "mm" in text


def test_circle_mode_shows_diameter_and_circumference(qtbot):
    """Gerçek kullanıcı isteği: "çember çap/çevre" ölçümü."""
    dialog = MeasurementToolDialog(_IMAGE, mm_per_px=1.0)
    qtbot.addWidget(dialog)
    dialog._canvas.resize(200, 200)
    dialog._mode_combo.setCurrentIndex(1)  # "Çember (Çap/Çevre)"
    assert dialog._canvas._mode == "CIRCLE"

    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(50, 50))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(70, 50))

    text = dialog._result_label.text()
    assert "Çember 1" in text
    assert "çap" in text
    assert "çevre" in text


def test_clear_measurements_button_empties_results(qtbot):
    dialog = MeasurementToolDialog(_IMAGE, mm_per_px=0.5)
    qtbot.addWidget(dialog)
    dialog._canvas.resize(200, 200)
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))
    assert "Çizgi 1" in dialog._result_label.text()

    qtbot.mouseClick(dialog._clear_button, Qt.MouseButton.LeftButton)

    assert "Çizgi 1" not in dialog._result_label.text()
    assert dialog._canvas.line_measurements() == []


def test_capture_button_saves_image_to_gallery_and_emits_signal(qtbot, tmp_path, monkeypatch):
    """Gerçek kullanıcı isteği: "ölçüm de dahil her alanda kare yakalayıp yan ekrana
    atabilmek istiyorum" -- Lens/Yükseklik-Ölçek kalibrasyon diyaloglarındaki AYNI
    `capture_store` deposuna yazmalı."""
    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")
    dialog = MeasurementToolDialog(_IMAGE, mm_per_px=0.5)
    qtbot.addWidget(dialog)
    assert dialog._capture_button.isEnabled()

    captured = []
    dialog.frame_captured.connect(lambda: captured.append(True))
    qtbot.mouseClick(dialog._capture_button, Qt.MouseButton.LeftButton)

    assert captured == [True]
    records = capture_store.list_captures(source="measurement")
    assert len(records) == 1


def test_capture_button_disabled_without_image(qtbot):
    dialog = MeasurementToolDialog(None, None)
    qtbot.addWidget(dialog)

    assert not dialog._capture_button.isEnabled()


def test_right_click_on_measurement_removes_it_from_results(qtbot):
    dialog = MeasurementToolDialog(_IMAGE, mm_per_px=0.5)
    qtbot.addWidget(dialog)
    dialog._canvas.resize(200, 200)
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(20, 20))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(60, 20))
    assert "Çizgi 1" in dialog._result_label.text()

    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.RightButton, pos=QPoint(40, 20))

    assert "Çizgi 1" not in dialog._result_label.text()
    assert "Kalibrasyon" not in dialog._result_label.text()  # mm_per_px verilmişti, sadece boş liste mesajı
