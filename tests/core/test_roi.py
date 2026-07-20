import numpy as np

from imgflow.core.roi import RoiRect


def test_clamp_within_bounds_unchanged():
    roi = RoiRect(2, 3, 4, 5)
    assert roi.clamp(20, 20) == roi


def test_clamp_shrinks_when_exceeding_bounds():
    roi = RoiRect(8, 8, 10, 10)
    assert roi.clamp(10, 10) == RoiRect(8, 8, 2, 2)


def test_clamp_negative_origin_clips_to_zero():
    roi = RoiRect(-5, -5, 10, 10)
    clamped = roi.clamp(10, 10)
    assert clamped.x == 0
    assert clamped.y == 0


def test_as_slice_matches_numpy_indexing():
    img = np.arange(100).reshape(10, 10)
    roi = RoiRect(2, 3, 4, 2)
    row_slice, col_slice = roi.as_slice()
    cropped = img[row_slice, col_slice]
    assert cropped.shape == (2, 4)
    assert cropped[0, 0] == img[3, 2]
