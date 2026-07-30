import numpy as np

from imgflow.core.roi import RoiRect, parse_roi_list


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


def test_parse_roi_list_decodes_and_clamps():
    rois = parse_roi_list("[[1, 2, 10, 10], [90, 90, 20, 20]]", image_w=100, image_h=100)
    assert rois == [RoiRect(1, 2, 10, 10), RoiRect(90, 90, 10, 10)]


def test_parse_roi_list_drops_degenerate_and_malformed_entries():
    rois = parse_roi_list('[[5, 5, 0, 10], [1, 1, "x", 5], [3, 3, 4, 4]]', image_w=50, image_h=50)
    assert rois == [RoiRect(3, 3, 4, 4)]


def test_parse_roi_list_invalid_json_returns_empty():
    assert parse_roi_list("not json", image_w=50, image_h=50) == []
    assert parse_roi_list('{"x": 1}', image_w=50, image_h=50) == []
