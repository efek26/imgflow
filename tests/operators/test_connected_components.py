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


def test_non_binary_input_is_auto_thresholded_instead_of_one_giant_blob():
    # Eşiklemesiz Siyah-Beyaz/RGB/HSV çıktısı gibi ikili OLMAYAN bir girdi verildiğinde,
    # eskiden neredeyse tüm görüntü sıfırdan farklı olduğu için tek dev "bölge" sayılırdı.
    img = np.zeros((20, 20), dtype=np.uint8)
    img[2:6, 2:6] = 80  # gürültü seviyesinde soluk bir bölge
    img[12:18, 12:18] = 220  # belirgin parlak bir nesne

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})

    # otomatik Otsu eşiklemesi devreye girip anlamlı sayıda (görüntü boyutunda TEK dev blok değil) bölge vermeli
    assert 1 <= out["count"] <= 2
    assert out["labels"][14, 14] != 0  # parlak bölge etiketlenmiş


def test_two_unique_non_zero_255_values_are_still_auto_thresholded():
    # Regresyon: HSV round-trip gibi durumlarda görüntü TAM OLARAK 2 farklı değer içerebilir
    # ama bunlar {0,255} değildir (ör. koyu arka plan=40, nesne=117) -- "iki değer var, zaten
    # ikili" diye yanlışça kabul edilip Otsu atlanırsa, connectedComponents sıfırdan farklı
    # HER iki değeri deön plan sayıp tüm görüntüyü tek dev bölge yapar.
    img = np.full((20, 20), 40, dtype=np.uint8)  # koyu arka plan (sıfır DEĞİL)
    img[5:15, 5:15] = 117  # belirgin nesne

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})

    assert out["count"] == 1
    assert out["labels"][10, 10] != 0  # nesne etiketlenmiş
    assert out["labels"][1, 1] == 0  # arka plan artık ön plan SAYILMIYOR (dev blok değil)


def test_already_binary_input_behavior_is_unchanged():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1:3, 1:3] = 255
    img[6:9, 6:9] = 255

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})

    assert out["count"] == 2


def test_max_area_ratio_excludes_floor_sized_blob_but_keeps_small_object():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[0:20, 0:15] = 255  # yanlışlıkla eşiklenmiş bant/zemin -- görüntünün büyük kısmı
    img[16:19, 16:19] = 255  # gerçek küçük nesne, zeminden ayrık

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8", "max_area_ratio": 0.5})

    assert out["count"] == 1
    assert out["labels"][10, 5] == 0  # zemin bloğu elendi
    assert out["labels"][17, 17] != 0  # küçük nesne hala etiketli


def test_polarity_white_is_default_and_unchanged():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1:3, 1:3] = 255
    img[6:9, 6:9] = 255

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})

    assert out["count"] == 2
    assert out["labels"][1, 1] != 0  # beyaz blok etiketli
    assert out["labels"][0, 0] == 0  # siyah zemin etiketsiz


def test_polarity_black_labels_dark_regions_instead():
    img = np.full((10, 10), 255, dtype=np.uint8)
    img[1:3, 1:3] = 0
    img[6:9, 6:9] = 0

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8", "polarity": "black"})

    assert out["count"] == 2
    assert out["labels"][1, 1] != 0  # siyah blok artık etiketli
    assert out["labels"][0, 0] == 0  # beyaz zemin etiketsiz


def test_polarity_both_labels_white_and_black_blobs_with_distinct_ids():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1:3, 1:3] = 255  # beyaz blok, zeminden (siyah) ayrık değil -- zeminle birleşir

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8", "polarity": "both"})

    # tüm görüntü ya beyaz bloğa ya da siyah zemine ait -- etiketsiz (0) piksel kalmamalı
    assert np.all(out["labels"] != 0)
    white_label = out["labels"][1, 1]
    black_label = out["labels"][0, 0]
    assert white_label != black_label
    assert out["count"] == 2  # 1 beyaz blok + 1 siyah zemin bloğu


def test_polarity_both_on_already_binary_two_blob_image_finds_three_regions():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[1:3, 1:3] = 255
    img[6:9, 6:9] = 255

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8", "polarity": "both"})

    # 2 ayrık beyaz blok + aralarındaki/etraflarındaki tek bir bağlı siyah zemin bloğu = 3
    assert out["count"] == 3
    assert np.all(out["labels"] != 0)


def test_max_area_ratio_zero_disables_filtering_by_default():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[0:20, 0:15] = 255
    img[16:19, 16:19] = 255

    out = ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})

    assert out["count"] == 2
