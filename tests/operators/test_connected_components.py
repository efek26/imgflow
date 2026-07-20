import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.connected_components import ConnectedComponentsOp


def test_registered():
    assert registry.get("segment.connected_components") is not None


def test_counts_two_separate_blobs():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1:3, 1:3] = 255
    img[6:9, 6:9] = 255

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})

    assert out["count"] == 2
    assert out["labels"].shape == img.shape
    assert out["labels"][1, 1] != 0
    assert out["labels"][1, 1] != out["labels"][6, 6]


def test_empty_image_has_zero_components():
    img = np.zeros((10, 10), dtype=np.uint8)
    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})
    assert out["count"] == 0


def test_connectivity_4_vs_8_diagonal_touch():
    img = np.zeros((4, 4), dtype=np.uint8)
    img[1, 1] = 255
    img[2, 2] = 255  # sadece köşegen komşu

    out4 = ConnectedComponentsOp().run({"image": img}, {"connectivity": "4"})
    out8 = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})

    assert out4["count"] == 2
    assert out8["count"] == 1
