import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.connected_components import ConnectedComponentsOp
from imgflow.operators.builtin.region_props import RegionPropsOp
from imgflow.operators.builtin.roi import RoiRegionOp


def test_registered():
    assert registry.get("roi.region") is not None


def test_roi_region_disabled_passes_through_unchanged():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = RoiRegionOp().run({"image": img}, {"enabled": False, "x": 2, "y": 2, "w": 3, "h": 3})
    assert out["image"] is img


def test_roi_region_enabled_crops():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = RoiRegionOp().run({"image": img}, {"enabled": True, "x": 2, "y": 3, "w": 4, "h": 2})
    assert out["image"].shape == (2, 4)
    assert (out["image"] == img[3:5, 2:6]).all()


def test_roi_region_enabled_clamps_to_bounds():
    img = np.zeros((10, 10), dtype=np.uint8)
    out = RoiRegionOp().run({"image": img}, {"enabled": True, "x": 5, "y": 5, "w": 20, "h": 20})
    assert out["image"].shape == (5, 5)


def test_roi_region_circle_crops_to_bounding_box_and_masks_outside():
    img = np.full((20, 20), 255, dtype=np.uint8)
    out = RoiRegionOp().run({"image": img}, {"enabled": True, "shape": "CIRCLE", "cx": 10, "cy": 10, "r": 5})["image"]
    assert out.shape == (10, 10)
    # merkez daire içinde kalmalı
    assert out[5, 5] == 255
    # köşeler dairenin dışında -> maskelenip 0 olmalı
    assert out[0, 0] == 0
    assert out[0, 9] == 0
    assert out[9, 0] == 0
    assert out[9, 9] == 0


def test_roi_region_circle_clamps_when_larger_than_image():
    img = np.zeros((10, 10), dtype=np.uint8)
    out = RoiRegionOp().run({"image": img}, {"enabled": True, "shape": "CIRCLE", "cx": 5, "cy": 5, "r": 50})["image"]
    assert out.shape == (10, 10)


def test_roi_region_then_connected_components_and_region_props():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[1:3, 1:3] = 255  # ROI dışında -> kırpılınca kaybolmalı
    img[10:14, 10:14] = 255  # ROI içinde -> kalmalı

    cropped = RoiRegionOp().run({"image": img}, {"enabled": True, "x": 8, "y": 8, "w": 10, "h": 10})["image"]
    labels = ConnectedComponentsOp().run({"image": cropped}, {"connectivity": "8"})["labels"]
    measurements = RegionPropsOp().run({"labels": labels}, {"min_area": 0})["measurements"]

    assert len(measurements) == 1
    # roi.region kırptığı için koordinatlar ROI'ye göre relatiftir (10,10 -> 2,2)
    assert measurements[0]["bbox_x"] == 2
    assert measurements[0]["bbox_y"] == 2
