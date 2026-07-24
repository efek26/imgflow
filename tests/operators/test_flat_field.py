import numpy as np
import pytest

from imgflow.io_utils import flatfield_store
from imgflow.operators.builtin.flat_field import FlatFieldOp


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    """Gerçek ~/.imgflow dizinine ASLA dokunma — izole tmp_path'e yönlendir."""
    monkeypatch.setattr(flatfield_store, "FLATFIELD_DIR", tmp_path / "flatfield")


def _vignette_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_dist = np.sqrt(cy**2 + cx**2)
    return 1.0 - 0.6 * (dist / max_dist)


def test_registered():
    from imgflow.operators import registry

    assert registry.get("correction.flat_field") is not None


def test_flat_field_flattens_vignetted_image():
    h, w = 60, 80
    vignette = _vignette_mask(h, w)
    reference = np.clip(200 * vignette, 0, 255).astype(np.uint8)
    image = np.clip(150 * vignette, 0, 255).astype(np.uint8)

    flatfield_store.save_reference("ref", reference)

    out = FlatFieldOp().run({"image": image}, {"reference_name": "ref", "strength": 1.0})
    corrected = out["image"]

    assert corrected.shape == image.shape
    # Referans AYNI vignetting profilini taşıdığından düzeltme sonrası görüntü neredeyse
    # sabit olmalı (girdi görüntüsünün std'sinin çok altında).
    assert corrected.std() < image.std() * 0.3


def test_strength_zero_returns_original_image():
    h, w = 20, 30
    vignette = _vignette_mask(h, w)
    reference = np.clip(200 * vignette, 0, 255).astype(np.uint8)
    image = np.clip(150 * vignette, 0, 255).astype(np.uint8)

    flatfield_store.save_reference("ref", reference)

    out = FlatFieldOp().run({"image": image}, {"reference_name": "ref", "strength": 0.0})
    assert np.allclose(out["image"].astype(np.int16), image.astype(np.int16), atol=1)


def test_reference_resized_to_match_image_shape():
    reference = np.full((10, 10), 200, dtype=np.uint8)
    image = np.full((40, 60), 150, dtype=np.uint8)
    flatfield_store.save_reference("ref", reference)

    out = FlatFieldOp().run({"image": image}, {"reference_name": "ref", "strength": 1.0})
    assert out["image"].shape == image.shape


def test_color_image_supported():
    h, w = 20, 30
    vignette = _vignette_mask(h, w)
    reference = np.clip(200 * vignette, 0, 255).astype(np.uint8)
    image = np.dstack([np.clip(150 * vignette, 0, 255).astype(np.uint8)] * 3)
    flatfield_store.save_reference("ref", reference)

    out = FlatFieldOp().run({"image": image}, {"reference_name": "ref", "strength": 1.0})
    assert out["image"].shape == image.shape


def test_empty_reference_name_raises_value_error():
    image = np.full((10, 10), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="reference_name"):
        FlatFieldOp().run({"image": image}, {"reference_name": ""})


def test_unknown_reference_name_raises_not_found():
    image = np.full((10, 10), 128, dtype=np.uint8)
    with pytest.raises(flatfield_store.FlatFieldReferenceNotFoundError):
        FlatFieldOp().run({"image": image}, {"reference_name": "yok_boyle_referans"})
