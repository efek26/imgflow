import numpy as np

from imgflow.core.types import PortSpec, PortType
from imgflow.ui.widgets.image_view import ImageView, extract_preview_image, numpy_to_qimage


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
