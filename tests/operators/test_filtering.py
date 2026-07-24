import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.filtering import (
    BilateralFilterOp,
    CannyEdgeOp,
    GaussianBlurOp,
    HistogramEqualizeOp,
    LaplacianEdgeOp,
    MedianBlurOp,
    SharpenOp,
    SobelEdgeOp,
)


def test_registered():
    op_ids = (
        "filter.gaussian_blur",
        "filter.median_blur",
        "filter.bilateral",
        "filter.canny_edge",
        "filter.sobel_edge",
        "filter.laplacian_edge",
        "filter.sharpen",
        "filter.histogram_equalize",
    )
    for op_id in op_ids:
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


def test_canny_edge_detects_step_boundary():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255  # dikey kenar x=10'da

    out = CannyEdgeOp().run({"image": img}, {"threshold1": 50, "threshold2": 150, "overlay": False})

    assert out["image"].shape == img.shape
    assert out["image"].ndim == 2
    assert out["image"][10, 7:13].max() == 255  # kenar civarında beyaz piksel olmalı
    assert out["image"][10, 2] == 0  # düz bölgede kenar olmamalı


def test_canny_edge_overlay_draws_on_original_color_image():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[:, 10:] = (100, 100, 100)

    out = CannyEdgeOp().run({"image": img}, {"threshold1": 50, "threshold2": 150, "overlay": True})

    assert out["image"].shape == img.shape
    assert (out["image"][10, 10] == (0, 255, 0)).all()  # kenar yeşille işaretlenmiş
    assert (out["image"][10, 2] == (0, 0, 0)).all()  # düz bölge orijinal renginde kalmış


def test_canny_edge_overlay_draws_green_on_single_channel_input():
    # color.hsv'nin maskesi (apply_mask) tek kanallı çıktı üretir; overlay bu durumda da
    # sessizce devre dışı kalmamalı (regresyon: daha önce sadece 3 kanallı girdide çiziyordu).
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255

    out = CannyEdgeOp().run({"image": img}, {"threshold1": 50, "threshold2": 150, "overlay": True})

    assert out["image"].shape == (20, 20, 3)
    assert (out["image"][10, 10] == (0, 255, 0)).all()
    assert (out["image"][10, 2] == (0, 0, 0)).all()


def test_sobel_edge_detects_vertical_boundary():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255

    out = SobelEdgeOp().run({"image": img}, {"kernel_size": 3, "direction": "X"})

    assert out["image"].shape == img.shape
    assert out["image"][10, 9:11].max() > 0
    assert out["image"][10, 2] == 0


def test_sobel_edge_y_direction_ignores_vertical_boundary():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255  # sadece dikey kenar; Y yönü (yatay kenar) bunu görmemeli

    out = SobelEdgeOp().run({"image": img}, {"kernel_size": 3, "direction": "Y"})

    assert out["image"].max() == 0


def test_laplacian_edge_detects_boundary():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255

    out = LaplacianEdgeOp().run({"image": img}, {"kernel_size": 3})

    assert out["image"].shape == img.shape
    assert out["image"][10, 9:11].max() > 0
    assert out["image"][10, 2] == 0


def test_sharpen_zero_amount_is_close_to_identity():
    img = np.random.default_rng(0).integers(0, 255, (10, 10, 3), dtype=np.uint8)
    out = SharpenOp().run({"image": img}, {"amount": 0.0, "kernel_size": 5})
    assert (out["image"] == img).all()


def test_sharpen_increases_local_contrast():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 200
    out = SharpenOp().run({"image": img}, {"amount": 1.5, "kernel_size": 5})
    # keskinleştirme kenar civarında "overshoot" yaratır: karanlık taraf orijinalden daha karanlık olur
    assert int(out["image"][10, 8]) <= int(img[10, 8])


def test_histogram_equalize_increases_dynamic_range():
    img = np.full((20, 20), 100, dtype=np.uint8)
    img[:, 10:] = 110  # düşük kontrast: sadece 100 ve 110 değerleri var

    out = HistogramEqualizeOp().run({"image": img}, {})

    assert out["image"].shape == img.shape
    assert (int(out["image"].max()) - int(out["image"].min())) > (110 - 100)
