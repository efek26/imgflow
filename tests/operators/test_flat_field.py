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


def test_dark_reference_region_does_not_saturate_whole_image_to_white():
    # Gerçek kullanıcı raporu: referansta koyu/gölgeli bir bölge varsa (ör. bandın dışına
    # taşan kenar) eskiden kazanç (gain = ortalama/referans) o bölgede sınırsız büyüyüp
    # görüntünün geniş bir kısmını BEYAZA doyuruyordu -- ekranda "görüntü kayboldu" gibi
    # görünüyordu. `max_gain` varsayılanı bunu önlemeli.
    h, w = 60, 60
    reference = np.full((h, w), 190, dtype=np.uint8)
    reference[:, :10] = 8  # gölgeli/koyu şerit
    flatfield_store.save_reference("gölgeli", reference)

    image = np.full((h, w), 170, dtype=np.uint8)
    image[:, :10] = 40

    out = FlatFieldOp().run({"image": image}, {"reference_name": "gölgeli", "strength": 1.0})
    corrected = out["image"]

    assert (corrected == 255).mean() < 0.05


def test_default_mode_is_reference_unaffected_by_mode_param_addition():
    h, w = 20, 30
    vignette = _vignette_mask(h, w)
    reference = np.clip(200 * vignette, 0, 255).astype(np.uint8)
    image = np.clip(150 * vignette, 0, 255).astype(np.uint8)
    flatfield_store.save_reference("ref", reference)

    out = FlatFieldOp().run({"image": image}, {"reference_name": "ref", "strength": 1.0})
    assert out["image"].shape == image.shape


def test_dynamic_local_mode_requires_no_reference_name():
    image = np.full((30, 30), 150, dtype=np.uint8)

    out = FlatFieldOp().run({"image": image}, {"mode": "dynamic_local", "blur_radius": 15})

    assert out["image"].shape == image.shape


def test_dynamic_local_mode_flattens_a_local_shadow_reference_mode_cannot_touch():
    # Gerçek kullanıcı raporu: "aydınlatma özelliği gölgeleri yok etmiyor, sadece dışarıdaki
    # alanı düzeltiyor". Kök neden bug değil: "reference" modu BOŞ bir referans karesine göre
    # çalıştığından, referansta hiç VAR OLMAYAN bir ürün gölgesini yapısal olarak
    # düzeltemez -- düz bir referansla gain=1 üretip görüntüyü DEĞİŞTİRMEDEN bırakır. Yeni
    # "dynamic_local" modu referans gerektirmeden HER karenin kendi yerel bulanık halinden
    # arka plan tahmini ürettiği için o karedeki gölgeyi de (kısmen) düzeltebilir.
    h, w = 100, 100
    background_level = 200
    shadow_level = 100
    image = np.full((h, w), background_level, dtype=np.uint8)
    image[40:60, 40:60] = shadow_level  # ürünün kendi gölgesi -- BOŞ referansta hiç YOKTU

    reference = np.full((h, w), background_level, dtype=np.uint8)  # boş/düz referans -- gölgesiz
    flatfield_store.save_reference("bos_referans", reference)

    reference_mode_out = FlatFieldOp().run(
        {"image": image}, {"mode": "reference", "reference_name": "bos_referans", "strength": 1.0}
    )["image"]
    dynamic_mode_out = FlatFieldOp().run(
        {"image": image}, {"mode": "dynamic_local", "blur_radius": 41, "strength": 1.0}
    )["image"]

    shadow_slice = (slice(40, 60), slice(40, 60))
    # Referans modu düz bir referansla gain=1 üretir -- gölgeye HİÇ dokunmaz.
    assert reference_mode_out[shadow_slice].mean() == pytest.approx(shadow_level, abs=1)
    # Yerel/Dinamik mod gölgeyi ÖNEMLİ ÖLÇÜDE aydınlatmalı (mükemmel değil ama belirgin).
    assert dynamic_mode_out[shadow_slice].mean() > shadow_level + 30


def test_default_mult_add_do_not_change_existing_behavior():
    h, w = 20, 20
    reference = np.full((h, w), 200, dtype=np.uint8)
    flatfield_store.save_reference("ref", reference)
    image = np.full((h, w), 150, dtype=np.uint8)

    baseline = FlatFieldOp().run({"image": image}, {"reference_name": "ref", "strength": 1.0})
    with_defaults = FlatFieldOp().run(
        {"image": image},
        {"reference_name": "ref", "strength": 1.0, "mult": 1.0, "add": 0.0},
    )
    assert np.array_equal(baseline["image"], with_defaults["image"])


def test_mult_scales_result_like_halcon_div_image():
    # HALCON'un div_image(Image1, Image2, Mult, Add) operatöründeki Mult davranışıyla AYNI:
    # sonucu ek bir sabit çarpanla ölçekler.
    h, w = 20, 20
    reference = np.full((h, w), 200, dtype=np.uint8)
    flatfield_store.save_reference("ref", reference)
    image = np.full((h, w), 100, dtype=np.uint8)

    out = FlatFieldOp().run(
        {"image": image}, {"reference_name": "ref", "strength": 1.0, "mult": 0.5}
    )
    # gain=1 (ref düz), mult=0.5 -> 100*0.5 = 50
    assert out["image"][0, 0] == pytest.approx(50, abs=1)


def test_add_offsets_result_like_halcon_div_image():
    h, w = 20, 20
    reference = np.full((h, w), 200, dtype=np.uint8)
    flatfield_store.save_reference("ref", reference)
    image = np.full((h, w), 100, dtype=np.uint8)

    out = FlatFieldOp().run(
        {"image": image}, {"reference_name": "ref", "strength": 1.0, "add": 20.0}
    )
    assert out["image"][0, 0] == pytest.approx(120, abs=1)


def test_max_gain_clamps_amplification_in_dark_reference_region():
    h, w = 20, 20
    reference = np.full((h, w), 200, dtype=np.uint8)
    reference[:, :5] = 4
    flatfield_store.save_reference("koyu_kenar", reference)

    image = np.full((h, w), 50, dtype=np.uint8)

    out = FlatFieldOp().run(
        {"image": image}, {"reference_name": "koyu_kenar", "strength": 1.0, "max_gain": 2.0}
    )
    corrected = out["image"]

    # Koyu kenardaki piksel en fazla ~2 katına çıkabilir (50*2=100), referansın çıkardığı
    # ~50 kat (200/4) DEĞİL.
    assert corrected[0, 0] <= 105
