import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.connected_components import ConnectedComponentsOp
from imgflow.operators.builtin.region_props import RegionPropsOp


def test_registered():
    assert registry.get("analysis.region_props") is not None


def _labels_for(img: np.ndarray) -> np.ndarray:
    return ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})["labels"]


def test_measures_square_region():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255  # 10x10 kare, alan=100

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})

    assert len(out["measurements"]) == 1
    m = out["measurements"][0]
    assert m["area"] == 100.0
    assert m["bbox_w"] == 10
    assert m["bbox_h"] == 10
    assert abs(m["centroid_x"] - 9.5) < 0.5
    assert abs(m["centroid_y"] - 9.5) < 0.5
    assert m["perimeter"] > 0
    assert 0 < m["circularity"] <= 1.2


def test_min_area_filters_small_noise():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255
    img[1, 1] = 255  # 1 piksel gürültü

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 5})

    assert len(out["measurements"]) == 1


def test_no_regions_returns_empty_list():
    img = np.zeros((20, 20), dtype=np.uint8)
    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})
    assert out["measurements"] == []
