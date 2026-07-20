import numpy as np

from imgflow.core.roi import RoiRect
from imgflow.operators import registry
from imgflow.operators.builtin.connected_components import ConnectedComponentsOp
from imgflow.operators.builtin.region_props import RegionPropsOp
from imgflow.operators.builtin.roi import RoiCropOp, RoiMaskOp, RoiRectangleOp


def test_registered():
    for op_id in ("roi.rectangle", "roi.crop", "roi.mask"):
        assert registry.get(op_id) is not None


def test_roi_rectangle_builds_spec():
    out = RoiRectangleOp().run({}, {"x": 1, "y": 2, "w": 3, "h": 4})
    assert out["roi"] == RoiRect(1, 2, 3, 4)


def test_roi_rectangle_clamps_to_image_bounds():
    img = np.zeros((10, 10), dtype=np.uint8)
    out = RoiRectangleOp().run({"image": img}, {"x": 5, "y": 5, "w": 20, "h": 20})
    assert out["roi"] == RoiRect(5, 5, 5, 5)


def test_roi_crop_returns_subregion():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    roi = RoiRect(2, 3, 4, 2)
    out = RoiCropOp().run({"image": img, "roi": roi}, {})
    assert out["image"].shape == (2, 4)
    assert (out["image"] == img[3:5, 2:6]).all()


def test_roi_mask_keeps_original_shape_and_zeros_outside():
    img = np.full((10, 10), 200, dtype=np.uint8)
    roi = RoiRect(2, 2, 3, 3)
    out = RoiMaskOp().run({"image": img, "roi": roi}, {})
    assert out["image"].shape == img.shape
    assert out["image"][2, 2] == 200
    assert out["image"][0, 0] == 0


def test_roi_mask_then_region_props_keeps_coordinates_in_original_image_space():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[1:3, 1:3] = 255  # ROI dışında -> maskelenip silinmeli
    img[10:14, 10:14] = 255  # ROI içinde -> kalmalı

    roi = RoiRectangleOp().run({"image": img}, {"x": 8, "y": 8, "w": 10, "h": 10})["roi"]
    masked = RoiMaskOp().run({"image": img, "roi": roi}, {})["image"]
    labels = ConnectedComponentsOp().run({"image": masked}, {"connectivity": "8"})["labels"]
    measurements = RegionPropsOp().run({"labels": labels}, {"min_area": 0})["measurements"]

    assert len(measurements) == 1
    assert measurements[0]["bbox_x"] == 10
    assert measurements[0]["bbox_y"] == 10
