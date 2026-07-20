import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.morphology import CloseOp, DilateOp, ErodeOp, OpenOp

_KERNEL_PARAMS = {"kernel_size": 3, "shape": "RECT", "iterations": 1}


def test_builtins_registered_by_default():
    for op_id in ("morphology.erode", "morphology.dilate", "morphology.open", "morphology.close"):
        assert registry.get(op_id) is not None


def test_erode_removes_isolated_pixel():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 255
    out = ErodeOp().run({"image": img}, _KERNEL_PARAMS)
    assert out["image"].sum() == 0


def test_dilate_grows_isolated_pixel_to_full_kernel():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 255
    out = DilateOp().run({"image": img}, _KERNEL_PARAMS)
    assert out["image"].sum() == 255 * 9


def test_open_removes_small_noise_but_keeps_large_region():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1, 1] = 255
    img[4:8, 4:8] = 255
    out = OpenOp().run({"image": img}, _KERNEL_PARAMS)
    assert out["image"][1, 1] == 0
    assert out["image"][5, 5] == 255


def test_close_fills_small_hole():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:8, 2:8] = 255
    img[4, 4] = 0
    out = CloseOp().run({"image": img}, _KERNEL_PARAMS)
    assert out["image"][4, 4] == 255


def test_iterations_and_kernel_size_are_respected():
    img = np.zeros((9, 9), dtype=np.uint8)
    img[4, 4] = 255
    small = DilateOp().run({"image": img}, {"kernel_size": 3, "shape": "RECT", "iterations": 1})
    large = DilateOp().run({"image": img}, {"kernel_size": 3, "shape": "RECT", "iterations": 2})
    assert large["image"].sum() > small["image"].sum()
