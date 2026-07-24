import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.color_modes import GrayscaleOp, HsvOp, RgbOp


def test_registered():
    for op_id in ("color.grayscale", "color.rgb", "color.hsv"):
        assert registry.get(op_id) is not None


def test_grayscale_default_converts_without_thresholding():
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = GrayscaleOp().run({"image": img}, {})
    assert out["image"].shape == (4, 4)
    assert out["image"][0, 0] == 100


def test_grayscale_manual_threshold():
    img = np.array([[0, 200], [50, 255]], dtype=np.uint8)
    img3 = np.stack([img] * 3, axis=-1)
    out = GrayscaleOp().run(
        {"image": img3},
        {"threshold_enabled": True, "threshold_mode": "MANUAL", "threshold_value": 127},
    )
    assert out["image"].tolist() == [[0, 255], [0, 255]]


def test_grayscale_invert():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out = GrayscaleOp().run({"image": img}, {"invert": True})
    assert (out["image"] == 255).all()


def test_grayscale_otsu_threshold_produces_binary_image():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, 5:] = 255
    out = GrayscaleOp().run({"image": img}, {"threshold_enabled": True, "threshold_mode": "OTSU"})
    assert set(np.unique(out["image"]).tolist()) <= {0, 255}


def test_rgb_default_passthrough():
    img = np.full((4, 4, 3), 100, dtype=np.uint8)
    out = RgbOp().run({"image": img}, {})
    assert (out["image"] == img).all()


def test_rgb_channel_view_isolates_red():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # BGR -> kırmızı kanal
    out = RgbOp().run({"image": img}, {"channel_view": "R"})
    assert out["image"].ndim == 2
    assert (out["image"] == 200).all()


def test_rgb_gain_scales_channel():
    img = np.full((2, 2, 3), 50, dtype=np.uint8)
    out = RgbOp().run({"image": img}, {"r_gain": 2.0, "g_gain": 1.0, "b_gain": 1.0})
    assert out["image"][0, 0, 2] == 100  # R kanalı (BGR index 2) 2x
    assert out["image"][0, 0, 0] == 50  # B kanalı değişmedi


def test_hsv_default_returns_bgr_shaped_image():
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    out = HsvOp().run({"image": img}, {})
    assert out["image"].shape == img.shape


def test_hsv_apply_mask_returns_binary_mask():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:2, :2] = (0, 255, 0)  # yeşil bölge
    out = HsvOp().run(
        {"image": img},
        {
            "apply_mask": True,
            "h_min": 40,
            "h_max": 80,
            "s_min": 100,
            "s_max": 255,
            "v_min": 100,
            "v_max": 255,
        },
    )
    assert out["image"].ndim == 2
    assert out["image"][0, 0] == 255
    assert out["image"][3, 3] == 0
