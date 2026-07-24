import numpy as np
import pytest

from imgflow.operators import registry
from imgflow.operators.builtin.color_props import ColorPropsOp


def test_registered():
    assert registry.get("analysis.color_props") is not None


def test_solid_gray_image_has_near_zero_ab():
    # Nötr gri (B=G=R eşit) LAB'da a*≈0, b*≈0 olmalı; L* orta parlaklıkta olmalı.
    image = np.full((30, 40, 3), 128, dtype=np.uint8)

    out = ColorPropsOp().run({"image": image}, {})
    m = out["measurements"][0]

    assert m["a_mean"] == pytest.approx(0.0, abs=2.0)
    assert m["b_mean"] == pytest.approx(0.0, abs=2.0)
    assert 40.0 < m["l_mean"] < 65.0
    assert out["overlay"].shape == (30, 40, 3)


def test_solid_red_image_has_positive_a():
    # BGR'de saf kırmızı -> LAB'da belirgin pozitif a* (kırmızı ekseni).
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[..., 2] = 255  # R kanalı

    out = ColorPropsOp().run({"image": image}, {})
    m = out["measurements"][0]

    assert m["a_mean"] > 30.0


def test_grayscale_image_supported():
    image = np.full((20, 20), 100, dtype=np.uint8)
    out = ColorPropsOp().run({"image": image}, {})
    assert len(out["measurements"]) == 1


def test_tolerance_disabled_by_default_no_delta_e_key():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = ColorPropsOp().run({"image": image}, {})
    assert "delta_e" not in out["measurements"][0]
    assert "tolerance_ok" not in out["measurements"][0]


def test_tolerance_enabled_matching_reference_is_ok():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = ColorPropsOp().run({"image": image}, {})
    m = out["measurements"][0]

    params = {
        "tolerance_enabled": True,
        "ref_l": m["l_mean"],
        "ref_a": m["a_mean"],
        "ref_b": m["b_mean"],
        "delta_e_max": 1.0,
    }
    out2 = ColorPropsOp().run({"image": image}, params)
    m2 = out2["measurements"][0]
    assert m2["tolerance_ok"] is True
    assert m2["delta_e"] == pytest.approx(0.0, abs=0.5)


def test_tolerance_enabled_far_reference_is_ng():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    params = {
        "tolerance_enabled": True,
        "ref_l": 0.0,
        "ref_a": 100.0,
        "ref_b": 100.0,
        "delta_e_max": 5.0,
    }
    out = ColorPropsOp().run({"image": image}, params)
    m = out["measurements"][0]
    assert m["tolerance_ok"] is False
