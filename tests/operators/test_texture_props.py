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


def test_average_angle_matches_mean_of_four_directions():
    # Gerçek kullanıcı isteği: HALCON karşılaştırması sonucu eklenen "average" (rotasyon
    # bağımsız) modun, dört kanonik yönün AYRI hesaplanmış özelliklerinin ARİTMETİK
    # ORTALAMASINA eşit olduğunu doğrular -- yönlü çizgili bir desende 0°/90° contrast'ı
    # belirgin FARKLI olduğundan (test_vertical_stripes_low_contrast_along_vertical_axis
    # ile aynı desen) bu gerçek bir ortalama olduğunu, sabit/yanlış bir değer olmadığını
    # kanıtlar.
    gray = np.zeros((40, 40), dtype=np.uint8)
    gray[:, 1::2] = 255

    per_angle = [compute_texture_features(gray, distance=1, angle=a, levels=32) for a in ("0", "45", "90", "135")]
    averaged = compute_texture_features(gray, distance=1, angle="average", levels=32)

    for key in ("contrast", "homogeneity", "energy", "correlation"):
        expected = sum(m[key] for m in per_angle) / 4
        assert averaged[key] == expected

    # 0° ve 90° gerçekten farklı olmalı (aksi halde ortalama almanın bir anlamı kalmaz).
    assert per_angle[0]["contrast"] != per_angle[2]["contrast"]


def test_average_angle_works_through_run():
    image = np.full((30, 40, 3), 128, dtype=np.uint8)
    out = TexturePropsOp().run({"image": image}, {"angle": "average"})
    assert len(out["measurements"]) == 1
    m = out["measurements"][0]
    assert m["contrast"] == 0.0
    assert m["energy"] == 1.0


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


def _two_blob_image() -> np.ndarray:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[5:15, 5:15] = 90  # düz gri kare #1
    image[25:35, 20:30] = 180  # düz gri kare #2 (farklı seviye)
    return image


def test_per_object_disabled_by_default_still_single_measurement():
    out = TexturePropsOp().run({"image": _two_blob_image()}, {})
    assert len(out["measurements"]) == 1


def test_per_object_enabled_lists_each_blob_separately():
    out = TexturePropsOp().run({"image": _two_blob_image()}, {"per_object_enabled": True})
    measurements = out["measurements"]

    assert len(measurements) == 2
    for m in measurements:
        assert set(m.keys()) >= {"contrast", "homogeneity", "energy", "correlation", "label"}
        assert {"bbox_x", "bbox_y", "bbox_w", "bbox_h"} <= m.keys()
        # Her iki kare de düz (tek tonlu) -> her nesnenin KENDİ bbox'ı üzerinde hesaplanan
        # contrast sıfır, energy/homogeneity 1 olmalı (compute_texture_features'ın düz
        # görüntü davranışıyla aynı, ama artık tüm görüntü yerine nesne başına).
        assert m["contrast"] == 0.0
        assert m["energy"] == 1.0
    assert out["overlay"].shape == _two_blob_image().shape


def test_per_object_min_area_filters_small_blobs():
    image = _two_blob_image()
    image[0:2, 0:2] = 50  # küçük gürültü lekesi

    out = TexturePropsOp().run({"image": image}, {"per_object_enabled": True, "min_object_area": 10.0})
    assert len(out["measurements"]) == 2


def test_per_object_max_area_filters_large_blobs():
    image = np.zeros((60, 60), dtype=np.uint8)
    image[5:15, 5:15] = 90
    image[25:35, 20:30] = 180
    image[40:60, 40:60] = 90  # 400 px² -- diğerlerinden çok büyük

    out = TexturePropsOp().run(
        {"image": image}, {"per_object_enabled": True, "max_object_area": 150.0}
    )
    assert len(out["measurements"]) == 2


def test_per_object_manual_threshold_recovers_dim_blob():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[5:15, 5:15] = 76
    image[25:35, 20:30] = 29

    otsu_out = TexturePropsOp().run({"image": image}, {"per_object_enabled": True})
    assert len(otsu_out["measurements"]) == 1

    manual_out = TexturePropsOp().run(
        {"image": image},
        {"per_object_enabled": True, "threshold_mode": "manual", "threshold_value": 15},
    )
    assert len(manual_out["measurements"]) == 2


def test_per_object_fill_holes_keeps_object_as_single_measurement():
    image = np.zeros((30, 30), dtype=np.uint8)
    image[5:25, 5:25] = 200
    image[13:17, 13:17] = 0  # ortasında eşik-altı bir "delik"

    out = TexturePropsOp().run(
        {"image": image}, {"per_object_enabled": True, "fill_holes": True}
    )
    assert len(out["measurements"]) == 1


def test_manual_roi_disabled_by_default_ignores_manual_rois():
    image = _two_blob_image()
    out = TexturePropsOp().run({"image": image}, {"manual_rois": "[[0, 0, 10, 10]]"})
    assert len(out["measurements"]) == 1


def test_manual_roi_enabled_ignores_auto_detection_uses_drawn_rects():
    image = _two_blob_image()
    params = {
        "manual_roi_enabled": True,
        "per_object_enabled": True,  # açık olsa bile manuel mod önceliklidir
        "manual_rois": "[[5, 5, 10, 10], [20, 25, 10, 10]]",
    }
    out = TexturePropsOp().run({"image": image}, params)
    measurements = out["measurements"]

    assert len(measurements) == 2
    assert [m["label"] for m in measurements] == [1, 2]
    assert all(m["manual"] is True for m in measurements)
    for m in measurements:
        assert set(m.keys()) >= {"contrast", "homogeneity", "energy", "correlation"}
        assert {"bbox_x", "bbox_y", "bbox_w", "bbox_h"} <= m.keys()
        # Her iki ROI de düz (tek tonlu) -> contrast sıfır, energy 1 olmalı.
        assert m["contrast"] == 0.0
        assert m["energy"] == 1.0
    assert out["overlay"].shape == image.shape


def test_manual_roi_empty_list_yields_no_measurements():
    image = _two_blob_image()
    out = TexturePropsOp().run(
        {"image": image}, {"manual_roi_enabled": True, "manual_rois": "[]"}
    )
    assert out["measurements"] == []
    assert out["overlay"].shape == image.shape


def test_per_object_labels_are_sequential_not_raw_component_ids():
    """`color_props.py`'deki AYNI düzeltme (bkz. oradaki test): ölçüm satırının numarası
    bağlı bileşen analizinin ham etiketi değil, overlay'in çizdiği sıralı numaradır."""
    img = np.zeros((200, 200), dtype=np.uint8)
    for x in range(10, 60, 8):
        img[5:7, x : x + 2] = 200  # gürültü lekeleri (düşük ham etiket numaraları alır)
    img[90:150, 30:90] = 180
    img[90:150, 120:180] = 220

    out = TexturePropsOp().run({"image": img}, {"per_object_enabled": True})

    assert [m["label"] for m in out["measurements"]] == [1, 2]
