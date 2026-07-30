import cv2
import numpy as np

from imgflow.core.auto_objects import _apply_auto_polarity, _border_touching_fraction
from imgflow.core.auto_objects import detect_objects as _detect_objects_for_polarity_tests


def test_auto_polarity_detects_dark_object_on_light_background():
    """Gerçek kullanıcı raporu: "şekil bulmada otomatik nesne tespitine tıklayınca hata
    veriyor" / "bazen arka planı ölçüyor". Kök neden sabit "parlak = nesne" varsayımıydı:
    açık zemin üzerindeki KOYU ürünlerde arka plan "nesne" sayılıyordu."""
    image = np.full((120, 120), 230, dtype=np.uint8)
    cv2.circle(image, (60, 60), 30, 60, -1)  # KOYU nesne, AÇIK zemin

    bright = _detect_objects_for_polarity_tests(image)  # eski/varsayılan davranış
    auto = _detect_objects_for_polarity_tests(image, polarity="auto")

    # Varsayılan: arka planı (kırpımın neredeyse tamamını) nesne sanır.
    assert bright and max(int(o.mask.sum()) for o in bright) > 0.7 * image.size
    # Otomatik: gerçek daireyi bulur (pi*30^2 ~ 2827 px).
    assert auto
    largest = max(auto, key=lambda o: int(o.mask.sum()))
    assert 0.15 * image.size < int(largest.mask.sum()) < 0.35 * image.size


def test_auto_polarity_keeps_bright_object_on_dark_background_unchanged():
    """Ters yön: koyu zemin üzerinde AÇIK nesnede karar değişmemeli (sahneye özel varsayım
    yok, ölçüt her iki yönde de aynı çerçeve kuralı)."""
    image = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(image, (60, 60), 30, 255, -1)

    bright = _detect_objects_for_polarity_tests(image)
    auto = _detect_objects_for_polarity_tests(image, polarity="auto")

    assert len(bright) == len(auto) == 1
    assert int(bright[0].mask.sum()) == int(auto[0].mask.sum())


def _dark_object_with_shadow_crossing_the_roi_border() -> np.ndarray:
    """Gerçek kullanıcı raporu: "gölgeli şekillerde kontür çizmekte sorun".

    Açık zemin (230) üzerinde KOYU nesne (60), artı ROI çerçevesinin ÜÇ kenarını kesen bir
    gölge şeridi (90 — nesne gibi eşik ALTINDA ama alan olarak ince). Bu, gerçek
    yakalamalarda ölçülen durumun sentetik karşılığı: çerçeve piksellerinin çoğu koyu
    (gölge) olduğu için eski "çerçevenin >%50'si ön plansa ters çevir" ölçütü ters çevirmeyi
    REDDEDİP arka planı nesne sanıyordu."""
    image = np.full((200, 200), 230, dtype=np.uint8)
    band = 14
    image[:band, :] = 90  # üst kenar gölgede
    image[-band:, :] = 90  # alt kenar gölgede
    image[:, :band] = 90  # sol kenar gölgede
    cv2.circle(image, (100, 100), 34, 60, -1)  # nesne — çerçeveye DEĞMEZ
    return image


def test_auto_polarity_survives_a_shadow_crossing_the_roi_border():
    """Gölge çerçeveyi kestiğinde de doğru taraf (koyu nesne) seçilmeli.

    Bu test düzeltmeden ÖNCE (çerçeve PİKSELİ oylaması ölçütüyle) gerçekten başarısızdı:
    ters çevirme yapılmadığı için "en büyük nesne" arka planın kendisi oluyordu."""
    image = _dark_object_with_shadow_crossing_the_roi_border()

    auto = _detect_objects_for_polarity_tests(image, polarity="auto")

    assert auto
    # Asıl mesele HANGİ TARAFIN nesne sayıldığı: tespit edilen tüm maskelerin birleşimi
    # nesneyi (koyu daireyi) İÇERMELİ, açık zemini İÇERMEMELİ. (Bu sentetik sahnede gölge
    # şeridi de eşiğin altında kaldığından ayrı bir bileşen olarak gelir — o yüzden "en büyük
    # bileşen daire olsun" demek yanıltıcı olurdu; kritik olan zeminin nesne SAYILMAMASI.)
    union = np.zeros(image.shape, dtype=bool)
    for obj in auto:
        union[obj.y : obj.y + obj.h, obj.x : obj.x + obj.w] |= obj.mask

    assert union[100, 100]  # daire merkezi = nesne
    assert not union[30, 100]  # açık zemin (gölge şeridinin dışında) = nesne DEĞİL
    assert not union[100, 180]  # açık zemin, sağ taraf = nesne DEĞİL


def test_auto_polarity_border_criterion_is_topological_not_a_pixel_vote():
    """Ölçütün NEDEN dayanıklı olduğunu doğrudan sabitler: aynı görüntüde çerçeve PİKSEL
    oylaması yanlış tarafı gösterirken, çerçeveye DEĞEN ALAN oranı doğru tarafı gösterir."""
    image = _dark_object_with_shadow_crossing_the_roi_border()
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ring = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
    pixel_vote_fraction = float((ring > 0).mean())
    # Eski ölçüt: çerçeve piksellerinin yarısından AZI ön plan -> "ters çevirme" -> YANLIŞ.
    assert pixel_vote_fraction < 0.5

    # Yeni ölçüt: parlak taraf (arka plan) çerçeveye değen alanda AÇIK ARA ile önde.
    assert _border_touching_fraction(binary) > _border_touching_fraction(cv2.bitwise_not(binary))
    assert not np.array_equal(_apply_auto_polarity(binary), binary)  # ters çevrildi


def test_max_area_filters_out_a_background_sized_component():
    """Gerçek kullanıcı isteği: "max alan da seçelim, bazen arka planı ölçüyor"."""
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, :] = 255  # tüm kırpımı kaplayan "bileşen"
    image[40:60, 40:60] = 0

    unlimited = _detect_objects_for_polarity_tests(image)
    limited = _detect_objects_for_polarity_tests(image, max_area=0.9 * image.size)

    assert unlimited and max(int(o.mask.sum()) for o in unlimited) > 0.9 * image.size
    assert all(int(o.mask.sum()) <= 0.9 * image.size for o in limited)

from imgflow.core.auto_objects import detect_objects


def _two_blob_image() -> np.ndarray:
    # 40x40 siyah zemin, iki ayrı (aralarında boşluk olan) kare -- ikisi de AYNI gri-seviye
    # parlaklığa (~100) sahip ama farklı renkte (biri kırmızıya, biri maviye çalan), böylece
    # Otsu eşiklemesi (parlaklık bazlı) ikisini de aynı "ön plan" tarafına koyar ve ayrım
    # sadece bağlı bileşen analizinin aralarındaki boşluğu görmesiyle yapılır -- gerçek
    # kullanım senaryosuyla (farklı renkli ama benzer parlaklıkta ürünler/balonlar) aynı.
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[5:15, 5:15] = (0, 40, 255)  # kırmızıya çalan (BGR), gray≈100
    image[25:35, 20:30] = (255, 121, 0)  # maviye çalan (BGR), gray≈100
    return image


def test_detects_each_separate_blob():
    objects = detect_objects(_two_blob_image())

    assert len(objects) == 2
    areas = sorted((obj.w * obj.h for obj in objects))
    assert areas == [100, 100]


def test_bbox_and_mask_match_blob_location():
    objects = detect_objects(_two_blob_image())
    by_x = sorted(objects, key=lambda obj: obj.x)

    first = by_x[0]
    assert (first.x, first.y, first.w, first.h) == (5, 5, 10, 10)
    assert first.mask.all()  # kare tamamen dolu, kırpım içinde tüm pikseller nesneye ait

    second = by_x[1]
    assert (second.x, second.y, second.w, second.h) == (20, 25, 10, 10)


def test_min_area_filters_small_blobs():
    image = _two_blob_image()
    image[0:2, 0:2] = (0, 255, 0)  # 4 piksellik küçük yeşil gürültü lekesi

    objects = detect_objects(image, min_area=10.0)

    assert len(objects) == 2  # küçük leke elenmiş olmalı


def test_empty_image_returns_no_objects():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    assert detect_objects(image) == []


def test_max_area_filters_large_blobs():
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[5:15, 5:15] = (0, 40, 255)  # 100 px²
    image[25:35, 20:30] = (255, 121, 0)  # 100 px²
    image[40:60, 40:60] = (0, 40, 255)  # 400 px² -- ayrı bir bölgede, diğerlerinden çok büyük

    objects = detect_objects(image, max_area=150.0)

    assert len(objects) == 2
    assert all(obj.w * obj.h == 100 for obj in objects)


def test_manual_threshold_recovers_blob_that_otsu_excludes():
    # Gerçek kullanıcı sorunu: iki nesnenin parlaklığı (gri seviyesi) BİRBİRİNDEN çok
    # farklıysa (ör. biri parlak ışık yansımalı, diğeri daha donuk), tek bir GLOBAL Otsu
    # eşiği ikisini AYNI ANDA "ön plan" sayamayabilir -- burada kırmızı (gray≈76) ve mavi
    # (gray≈29) kare, siyah (0) zemine göre Otsu'da SADECE kırmızı ön plan sayılır.
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[5:15, 5:15] = (0, 0, 255)  # kırmızı, gray≈76
    image[25:35, 20:30] = (255, 0, 0)  # mavi, gray≈29

    otsu_objects = detect_objects(image)
    assert len(otsu_objects) == 1  # mavi kare Otsu tarafından arka plan sayılıp kaçırılıyor

    manual_objects = detect_objects(image, threshold_mode="manual", threshold_value=15)
    assert len(manual_objects) == 2  # elle düşük bir eşikle her ikisi de yakalanır


def test_fill_holes_merges_object_split_by_bright_reflection():
    # Bir nesnenin İÇİNDE ışık yansımasından kalan eşik-altı bir "delik" varsa (ör. donuk
    # gövdeli ama ortasında parlak bir highlight olan bir balon), delik doldurulmadan
    # bağlı bileşen analizi nesneyi ikiye bölebilir/küçük gösterebilir; fill_holes=True
    # dış konturu doldurup tek parça nesne olarak sayar.
    image = np.zeros((30, 30), dtype=np.uint8)
    image[5:25, 5:25] = 200  # nesne gövdesi
    image[13:17, 13:17] = 0  # ortasında eşik-altı bir "delik" (koyu yansıma gölgesi)

    without_fill = detect_objects(image)
    assert without_fill[0].w * without_fill[0].h - int(without_fill[0].mask.sum()) > 0

    with_fill = detect_objects(image, fill_holes=True)
    assert len(with_fill) == 1
    assert with_fill[0].mask.all()  # delik doldurulduğu için kırpım TAMAMEN dolu


def test_robust_edge_assist_recovers_filled_interior_otsu_alone_misses():
    """Gerçek kullanıcı isteği: "daha sağlam olsun, biraz yavaşlasa da olur" (bkz.
    `shape_matching_dialog.py::_build_auto_contour_mask`, eğitim TEK SEFERLİK olduğundan
    performans kısıtı yok). Bir nesnenin İÇİ arka planla AYNI parlaklıktaysa (Otsu bunu
    AYIRAMAZ) ama nesnenin KENARI farklı bir parlaklıktaysa (net bir Canny kenarı vardır),
    salt Otsu SADECE ince kenar halkasını "nesne" sayar (içini DOLDURAMAZ); `robust=True`
    kenar-tabanlı doldurma ile nesnenin GERÇEK (dolu) alanını kurtarır."""
    image = np.full((60, 60), 100, dtype=np.uint8)  # zemin VE nesnenin İÇİ aynı parlaklıkta
    cv2.rectangle(image, (15, 15), (45, 45), color=180, thickness=3)  # sadece KENARI farklı

    plain_objects = detect_objects(image)
    robust_objects = detect_objects(image, robust=True)

    assert len(plain_objects) == 1
    assert len(robust_objects) == 1
    plain_area = int(plain_objects[0].mask.sum())
    robust_area = int(robust_objects[0].mask.sum())
    # Salt Otsu sadece ince kenar halkasını yakalar (dolu 30x30=900'e göre KÜÇÜK);
    # robust=True nesnenin GERÇEK dolu alanına çok daha yakın, AÇIKÇA daha büyük bir alan bulur.
    assert robust_area > plain_area * 2


def test_robust_false_default_leaves_existing_behavior_unchanged():
    """`robust=False` (varsayılan -- `color_props`/`texture_props`'un canlı-kamera yolu)
    davranış/performans BİREBİR eskisi gibi kalmalı."""
    image = _two_blob_image()

    default_objects = detect_objects(image)
    explicit_false_objects = detect_objects(image, robust=False)

    assert len(default_objects) == len(explicit_false_objects) == 2
    for a, b in zip(
        sorted(default_objects, key=lambda o: o.x), sorted(explicit_false_objects, key=lambda o: o.x)
    ):
        assert (a.x, a.y, a.w, a.h) == (b.x, b.y, b.w, b.h)
        assert np.array_equal(a.mask, b.mask)


def test_close_kernel_bridges_thin_gap_between_blob_parts():
    # Nesne sınırında ince bir kopukluk (ör. parlamanın kenarda bıraktığı 1px'lik eşik-altı
    # çizgi) aynı nesneyi iki ayrı bileşene bölebilir; close_kernel_size bunu köprüler.
    image = np.zeros((20, 40), dtype=np.uint8)
    image[5:15, 2:18] = 200
    image[5:15, 18:19] = 0  # 1px'lik kopukluk
    image[5:15, 19:35] = 200

    without_close = detect_objects(image)
    assert len(without_close) == 2

    with_close = detect_objects(image, close_kernel_size=3)
    assert len(with_close) == 1
