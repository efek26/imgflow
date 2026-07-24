from pathlib import Path

import numpy as np
from PySide6.QtCore import QUrl, Qt
from PySide6.QtWidgets import QScrollArea

from imgflow.core.types import PortSpec, PortType
from imgflow.ui.widgets.image_view import ImageView, extract_preview_image, numpy_to_qimage


class _FakeMimeData:
    def __init__(self, urls):
        self._urls = urls

    def urls(self):
        return self._urls


class _FakeDropEvent:
    """Gerçek Qt sürükle-bırak makinesini (native event loop) tetiklemeden `ImageView.
    dragEnterEvent`/`dropEvent`'in mimeData'yı nasıl yorumladığını test etmek için minimal
    bir duck-typed sahte event — kodun çağırdığı `mimeData()`/`acceptProposedAction()`/
    `ignore()` dışında hiçbir Qt event API'si taklit etmiyor."""

    def __init__(self, urls):
        self._mime = _FakeMimeData(urls)
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def test_drop_event_emits_image_file_dropped_for_local_file(qtbot, tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"fake")
    view = ImageView()
    qtbot.addWidget(view)
    received = []
    view.image_file_dropped.connect(received.append)

    event = _FakeDropEvent([QUrl.fromLocalFile(str(path))])
    view.dropEvent(event)

    assert len(received) == 1
    assert Path(received[0]) == path
    assert event.accepted is True


def test_drop_event_ignores_non_file_mime(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    received = []
    view.image_file_dropped.connect(received.append)

    event = _FakeDropEvent([])
    view.dropEvent(event)

    assert received == []
    assert event.ignored is True


def test_drag_enter_event_accepts_local_file_urls(qtbot, tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"fake")
    view = ImageView()
    qtbot.addWidget(view)

    event = _FakeDropEvent([QUrl.fromLocalFile(str(path))])
    view.dragEnterEvent(event)

    assert event.accepted is True


def test_drag_enter_event_ignores_mime_without_urls(qtbot):
    view = ImageView()
    qtbot.addWidget(view)

    event = _FakeDropEvent([])
    view.dragEnterEvent(event)

    assert event.ignored is True


def test_numpy_to_qimage_grayscale(qtbot):
    qimage = numpy_to_qimage(np.zeros((4, 6), dtype=np.uint8))
    assert (qimage.width(), qimage.height()) == (6, 4)


def test_numpy_to_qimage_bgr(qtbot):
    qimage = numpy_to_qimage(np.zeros((4, 6, 3), dtype=np.uint8))
    assert (qimage.width(), qimage.height()) == (6, 4)


def test_extract_preview_image_picks_first_image_port():
    class _FakeOp:
        outputs = [PortSpec("count", PortType.SCALAR), PortSpec("labels", PortType.IMAGE)]

    outputs = {"count": 3, "labels": np.zeros((2, 2), dtype=np.int32)}
    result = extract_preview_image(_FakeOp, outputs)
    assert result is not None
    assert result.shape == (2, 2)


def test_extract_preview_image_returns_none_when_no_image_port():
    class _FakeOp:
        outputs = [PortSpec("measurements", PortType.MEASUREMENTS)]

    assert extract_preview_image(_FakeOp, {"measurements": [{"area": 1}]}) is None


def test_extract_preview_image_returns_none_for_empty_outputs():
    class _FakeOp:
        outputs = [PortSpec("image", PortType.IMAGE)]

    assert extract_preview_image(_FakeOp, None) is None


def test_image_view_set_none_shows_placeholder_text(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.set_image(None)
    assert view.text() == "Önizleme yok"


def test_image_view_set_image_clears_placeholder_text(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.set_image(np.zeros((10, 10), dtype=np.uint8))
    assert view.text() == ""


def _hosted_view(qtbot):
    """Zoom sadece bir QScrollArea'ya bağlıyken (`set_scroll_host`) aktiftir (bkz.
    `ImageView._rescale`) — bu yüzden zoom testleri gerçek bir scroll area içine kurar."""
    scroll_area = QScrollArea()
    view = ImageView()
    scroll_area.setWidget(view)
    scroll_area.setWidgetResizable(True)
    view.set_scroll_host(scroll_area)
    scroll_area.resize(200, 200)
    qtbot.addWidget(scroll_area)
    scroll_area.show()
    view.set_image(np.zeros((100, 50), dtype=np.uint8))
    return scroll_area, view


class _FakeWheelEvent:
    def __init__(self, delta_y: int, ctrl: bool) -> None:
        self._delta_y = delta_y
        self._modifiers = Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier
        self.accepted = False

    def modifiers(self):
        return self._modifiers

    def angleDelta(self):  # noqa: N802 - Qt naming
        class _Delta:
            def __init__(self, y):
                self._y = y

            def y(self):
                return self._y

        return _Delta(self._delta_y)

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def test_zoom_without_scroll_host_is_noop(qtbot):
    view = ImageView()
    qtbot.addWidget(view)
    view.set_image(np.zeros((10, 10), dtype=np.uint8))

    view.zoom_in()

    assert view.zoom == 1.0


def test_zoom_in_increases_zoom_and_grows_label_beyond_viewport(qtbot):
    _scroll_area, view = _hosted_view(qtbot)

    view.zoom_in()

    assert view.zoom > 1.0
    assert view.width() > 200 or view.height() > 200


def test_zoom_reset_returns_to_fit_size(qtbot):
    scroll_area, view = _hosted_view(qtbot)
    view.zoom_in()
    view.zoom_in()

    view.zoom_reset()

    assert view.zoom == 1.0
    assert scroll_area.widgetResizable() is True


def test_zoom_is_clamped_to_max(qtbot):
    _scroll_area, view = _hosted_view(qtbot)

    for _ in range(30):
        view.zoom_in()

    from imgflow.ui.widgets.image_view import MAX_ZOOM

    assert view.zoom == MAX_ZOOM


def test_zoom_is_clamped_to_min(qtbot):
    _scroll_area, view = _hosted_view(qtbot)

    for _ in range(30):
        view.zoom_out()

    from imgflow.ui.widgets.image_view import MIN_ZOOM

    assert view.zoom == MIN_ZOOM


def test_zoom_changed_signal_emits_new_zoom(qtbot):
    _scroll_area, view = _hosted_view(qtbot)
    received = []
    view.zoom_changed.connect(received.append)

    view.zoom_in()

    assert len(received) == 1
    assert received[0] == view.zoom


def test_ctrl_wheel_zooms_in_and_out(qtbot):
    _scroll_area, view = _hosted_view(qtbot)

    view.wheelEvent(_FakeWheelEvent(delta_y=120, ctrl=True))
    assert view.zoom > 1.0
    zoomed_in = view.zoom

    view.wheelEvent(_FakeWheelEvent(delta_y=-120, ctrl=True))
    assert view.zoom < zoomed_in


def test_plain_wheel_without_ctrl_does_not_zoom(qtbot):
    _scroll_area, view = _hosted_view(qtbot)

    view.wheelEvent(_FakeWheelEvent(delta_y=120, ctrl=False))

    assert view.zoom == 1.0
