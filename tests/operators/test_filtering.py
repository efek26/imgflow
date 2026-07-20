import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.filtering import BilateralFilterOp, GaussianBlurOp, MedianBlurOp


def test_registered():
    for op_id in ("filter.gaussian_blur", "filter.median_blur", "filter.bilateral"):
        assert registry.get(op_id) is not None


def test_gaussian_blur_spreads_impulse():
    img = np.zeros((11, 11), dtype=np.uint8)
    img[5, 5] = 255

    out = GaussianBlurOp().run({"image": img}, {"kernel_size": 5, "sigma_x": 0.0})

    assert out["image"][5, 5] < 255
    assert out["image"][5, 4] > 0


def test_gaussian_blur_forces_even_kernel_to_odd():
    img = np.zeros((11, 11), dtype=np.uint8)
    img[5, 5] = 255
    out = GaussianBlurOp().run({"image": img}, {"kernel_size": 4, "sigma_x": 0.0})
    assert out["image"].shape == img.shape


def test_median_blur_removes_salt_pepper_outlier():
    img = np.full((9, 9), 100, dtype=np.uint8)
    img[4, 4] = 250

    out = MedianBlurOp().run({"image": img}, {"kernel_size": 3})

    assert out["image"][4, 4] == 100


def test_bilateral_filter_preserves_flat_regions():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, :10] = 50
    img[:, 10:] = 200

    out = BilateralFilterOp().run(
        {"image": img}, {"diameter": 9, "sigma_color": 75.0, "sigma_space": 75.0}
    )

    assert abs(int(out["image"][10, 2]) - 50) <= 1
    assert abs(int(out["image"][10, 17]) - 200) <= 1
