import cv2
import numpy as np
import pytest

from imgflow.core.focus_metric import focus_measure


def _checkerboard(square_px: int = 20, squares: int = 10) -> np.ndarray:
    size = square_px * squares
    img = np.zeros((size, size), dtype=np.uint8)
    for r in range(squares):
        for c in range(squares):
            if (r + c) % 2 == 0:
                img[r * square_px : (r + 1) * square_px, c * square_px : (c + 1) * square_px] = 255
    return img


def test_focus_measure_higher_for_sharp_than_blurred():
    sharp = _checkerboard()
    blurred = cv2.GaussianBlur(sharp, (25, 25), sigmaX=8)

    assert focus_measure(sharp) > focus_measure(blurred)


def test_focus_measure_accepts_color_image():
    sharp = _checkerboard()
    color = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    assert focus_measure(color) == pytest.approx(focus_measure(sharp), rel=1e-6)


def test_focus_measure_zero_for_blank_image():
    blank = np.full((100, 100), 128, dtype=np.uint8)

    assert focus_measure(blank) == 0.0
