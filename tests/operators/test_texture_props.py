import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.texture_props import TexturePropsOp, compute_texture_features


def test_registered():
    assert registry.get("analysis.texture_props") is not None


def test_uniform_image_has_zero_contrast_and_max_energy():
    gray = np.full((40, 40), 128, dtype=np.uint8)
    features = compute_texture_features(gray, distance=1, angle="0", levels=32)

    assert features["contrast"] == 0.0
    assert features["energy"] == 1.0
    assert features["homogeneity"] == 1.0


def test_checkerboard_has_high_contrast_horizontal():
    # Yatayda her komşu piksel çifti 0/255 (ya da 255/0) -- distance=1, angle=0 için
    # maksimuma yakın contrast bekleniyor.
    gray = np.zeros((40, 40), dtype=np.uint8)
    gray[:, 1::2] = 255

    uniform = np.full((40, 40), 128, dtype=np.uint8)
    checkerboard_features = compute_texture_features(gray, distance=1, angle="0", levels=32)
    uniform_features = compute_texture_features(uniform, distance=1, angle="0", levels=32)

    assert checkerboard_features["contrast"] > uniform_features["contrast"]
    assert checkerboard_features["energy"] < uniform_features["energy"]


def test_vertical_stripes_low_contrast_along_vertical_axis():
    # Dikey çizgiler -- 90° (dikey komşuluk) yönünde komşu piksel çiftleri HEP AYNI seviyede,
    # bu yüzden o yönde contrast sıfıra yakın olmalı (yönlü doku ayrımını doğrular).
    gray = np.zeros((40, 40), dtype=np.uint8)
    gray[:, 1::2] = 255

    vertical_features = compute_texture_features(gray, distance=1, angle="90", levels=32)
    assert vertical_features["contrast"] == 0.0


def test_run_produces_single_measurement_and_overlay():
    image = np.full((30, 40, 3), 128, dtype=np.uint8)
    out = TexturePropsOp().run({"image": image}, {})

    assert len(out["measurements"]) == 1
    m = out["measurements"][0]
    assert set(m.keys()) == {"contrast", "homogeneity", "energy", "correlation"}
    assert out["overlay"].shape == (30, 40, 3)


def test_grayscale_image_supported():
    image = np.full((20, 20), 100, dtype=np.uint8)
    out = TexturePropsOp().run({"image": image}, {})
    assert len(out["measurements"]) == 1


def test_small_image_smaller_than_distance_does_not_crash():
    image = np.full((2, 2), 50, dtype=np.uint8)
    out = TexturePropsOp().run({"image": image}, {"distance": 5})
    assert len(out["measurements"]) == 1
