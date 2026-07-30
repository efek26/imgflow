import cv2
import numpy as np
import pytest

from imgflow.core import shape_matching as shape_matching_module
from imgflow.core.roi import RoiRect
from imgflow.core.shape_matching import (
    LabeledMatch,
    ShapeMatchingError,
    _nms,
    _score_map,
    build_search_pyramid,
    find_shape_model,
    render_match_overlay,
    train_shape_model,
)

_BASE_TRIANGLE = np.array([[0, -40], [35, 25], [-20, 30]], dtype=np.float64)
"""Asimetrik (skalen) üçgen — rotasyonel simetrisi yok, bu yüzden bulunan açı belirsizliksiz test edilebilir."""


def _draw_triangle(image: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (_BASE_TRIANGLE @ rot.T) + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)


def _reference_image() -> np.ndarray:
    image = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(image, (100, 100), 0.0)
    return image


def _draw_scaled_triangle(image: np.ndarray, center: tuple[float, float], angle_deg: float, scale: float) -> None:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (_BASE_TRIANGLE * scale) @ rot.T + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)


def _train():
    return train_shape_model(_reference_image(), RoiRect(50, 50, 100, 100))


def test_train_shape_model_extracts_points_per_level():
    model = _train()
    assert len(model.levels) == 3
    for level in model.levels:
        assert level.points.shape[0] > 0
        assert level.points.shape == (level.angles.shape[0], 2)


def test_train_shape_model_num_levels_none_stops_before_sparse_level():
    """HALCON'un `inspect_shape_model` + otomatik `NumLevels` seçimi mantığı: `num_levels=None`
    iken piramit derinliği bir sonraki seviyenin nokta sayısı `_AUTO_LEVEL_MIN_POINTS`'in
    altına düşene kadar artırılır, SONRA durur -- explicit `num_levels=6` (mevcut/manuel
    davranış) ile karşılaştırınca DAHA AZ seviye üretmeli (son seviye çok seyrek kaldığından
    hiç DAHİL EDİLMEZ, oysa manuel modda hâlâ -- az nokta olsa bile -- dahil edilir)."""
    image = _reference_image()
    roi = RoiRect(50, 50, 100, 100)

    explicit = train_shape_model(image, roi, num_levels=6)
    auto = train_shape_model(image, roi, num_levels=None)

    assert len(explicit.levels) == 6  # manuel davranış DEĞİŞMEDİ
    assert len(auto.levels) < len(explicit.levels)
    assert all(lvl.points.shape[0] >= 15 for lvl in auto.levels[1:])  # ilk seviye hariç


def test_train_shape_model_explicit_num_levels_unaffected_by_auto_mode_addition():
    """`num_levels` bir TAM SAYI olarak verildiğinde (varsayılan/manuel yol) davranış BİREBİR
    eskisi gibi kalmalı -- otomatik modun eklenmesi bunu ETKİLEMEMELİ."""
    model = train_shape_model(_reference_image(), RoiRect(50, 50, 100, 100), num_levels=3)
    assert len(model.levels) == 3


def test_train_shape_model_rejects_roi_without_edges():
    blank = np.full((200, 200), 255, dtype=np.uint8)
    with pytest.raises(ShapeMatchingError):
        train_shape_model(blank, RoiRect(50, 50, 100, 100))


def test_train_shape_model_rejects_tiny_roi():
    with pytest.raises(ShapeMatchingError):
        train_shape_model(_reference_image(), RoiRect(50, 50, 2, 2))


def test_train_shape_model_mask_excludes_noise_outside_contour():
    """`mask` ("Konturu Otomatik Algıla"/elle poligon ile üretilen kontur maskesi) ROI
    dikdörtgeni içindeki ama nesnenin GERÇEK dış hattı DIŞINDAKİ kenarları (ör. bant/arka
    plan gürültüsü) elemeli; maskesiz eğitim bu gürültüyü de modele almalı."""
    image = _reference_image()
    # ROI (50,50,100,100) içinde ama üçgenin dışında ayrı bir gürültü kenarı (küçük kare).
    cv2.rectangle(image, (60, 60), (75, 75), color=0, thickness=2)

    roi = RoiRect(50, 50, 100, 100)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    mask_raw = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask_raw, [tri_points], 255)
    mask_raw = cv2.dilate(mask_raw, np.ones((5, 5), dtype=np.uint8))
    mask = mask_raw.astype(bool)

    model_unmasked = train_shape_model(image, roi)
    model_masked = train_shape_model(image, roi, mask=mask)

    cx, cy = roi.x + roi.w / 2.0, roi.y + roi.h / 2.0

    def _all_points_inside_mask(level) -> bool:
        abs_x = np.clip((level.points[:, 0] + cx).astype(int), 0, mask.shape[1] - 1)
        abs_y = np.clip((level.points[:, 1] + cy).astype(int), 0, mask.shape[0] - 1)
        return bool(mask[abs_y, abs_x].all())

    assert not _all_points_inside_mask(model_unmasked.levels[0])
    assert _all_points_inside_mask(model_masked.levels[0])


def test_train_shape_model_mask_shape_mismatch_raises():
    with pytest.raises(ShapeMatchingError):
        train_shape_model(
            _reference_image(), RoiRect(50, 50, 100, 100), mask=np.zeros((10, 10), dtype=bool)
        )


def test_shape_model_to_dict_from_dict_round_trips_levels_and_corners():
    """`ShapeModel` artık ayrı bir `contour` alanı taşımıyor (bkz. modül docstring'i -- HALCON
    tarzı yeniden tasarım: overlay artık `levels[0].points`'i çizer) -- serileştirme SADECE
    `levels`/`corners`/`reference_center`'ı korumalı."""
    model = _train()

    restored = shape_matching_module.ShapeModel.from_dict(model.to_dict())

    assert len(restored.levels) == len(model.levels)
    assert np.allclose(restored.levels[0].points, model.levels[0].points)
    assert np.allclose(restored.corners, model.corners)
    assert not hasattr(restored, "contour")


def test_shape_model_from_dict_ignores_legacy_contour_key():
    """Bu özellikten ÖNCE kaydedilmiş `.json` şekil modellerinde bir `"contour"` anahtarı
    olabilir -- `from_dict` bunu artık HİÇ okumadığından (dataclass'ta alan yok), fazladan
    anahtar sessizce YOKSAYILMALI, ÇÖKME olmamalı. Bu, kullanıcının bu değişiklikten ÖNCE
    eğittiği modellerin YENİDEN EĞİTİLMEDEN yüklenebilmesini garanti eder."""
    model = _train()
    data = model.to_dict()
    data["contour"] = [[60.0, 60.0], [140.0, 60.0], [140.0, 140.0], [60.0, 140.0]]

    restored = shape_matching_module.ShapeModel.from_dict(data)

    assert len(restored.levels) == len(model.levels)
    assert not hasattr(restored, "contour")


def test_train_shape_model_stores_reference_center_as_roi_absolute_center():
    """Gerçek kullanıcı isteği: "geometrik eşlemede ... cismin ötelenme uzaklığı da yazmalı"
    -- `reference_center`, eğitim ROI'sinin MUTLAK (referans görüntüdeki) merkezi olmalı."""
    roi = RoiRect(50, 50, 100, 100)
    model = train_shape_model(_reference_image(), roi)

    assert model.reference_center == (100.0, 100.0)


def test_shape_model_to_dict_from_dict_round_trips_reference_center():
    model = _train()
    restored = shape_matching_module.ShapeModel.from_dict(model.to_dict())
    assert restored.reference_center == model.reference_center


def test_shape_model_from_dict_without_reference_center_key_does_not_crash():
    """Eski `.json` modellerde `reference_center` anahtarı yok -- `from_dict` `None` döner
    (YANLIŞ bir (0,0) DEĞİL), `operators/builtin/shape_match.py` bunu görüp öteleme alanlarını
    HİÇ eklememeli."""
    model = _train()
    data = model.to_dict()
    del data["reference_center"]

    restored = shape_matching_module.ShapeModel.from_dict(data)

    assert restored.reference_center is None


def test_find_shape_model_recovers_translation_and_rotation():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)

    assert len(matches) == 1
    match = matches[0]
    assert match.x == pytest.approx(130, abs=3)
    assert match.y == pytest.approx(80, abs=3)
    assert match.angle == pytest.approx(30, abs=3)
    assert match.score >= 0.5


def test_find_shape_model_exact_pose_scores_near_one():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (105, 95), 0.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)

    assert len(matches) == 1
    assert matches[0].score == pytest.approx(1.0, abs=0.05)


def test_find_shape_model_defaults_report_scale_one_even_when_object_differs_in_size():
    """Varsayılan `scale_min=scale_max=1.0` iken (ölçek araması KAPALI) `MatchResult.scale`
    HER ZAMAN 1.0 olmalı -- bulunan bir nesne varsa bile eskisiyle BİREBİR aynı davranış/
    performans (gerçek kullanıcı isteği öncesi mevcut davranış hiç DEĞİŞMEMELİ)."""
    model = _train()
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_scaled_triangle(search, (150.0, 150.0), 0.0, scale=1.5)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)

    assert all(m.scale == pytest.approx(1.0) for m in matches)


def test_find_shape_model_with_scale_search_finds_larger_object_and_reports_scale():
    """Gerçek kullanıcı isteği: "farklı boyutlarda aynı cisimden varsa scale boyutu yazmalı"
    -- öğretilen boyuttan belirgin farklı (burada 1.5x büyük) bir nesne, `scale_min`/
    `scale_max` verilince (ölçek araması AÇIK) doğru konum/açıda VE `match.scale`'i gerçek
    ölçeğe (~1.5) yakın olarak bulunmalı."""
    model = _train()
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_scaled_triangle(search, (150.0, 150.0), 0.0, scale=1.5)

    matches = find_shape_model(
        search, model, min_score=0.5, greediness=0.7, max_matches=1,
        scale_min=0.8, scale_max=1.8, scale_step_coarse=0.2,
    )

    assert len(matches) == 1
    match = matches[0]
    assert match.scale == pytest.approx(1.5, abs=0.2)
    assert match.x == pytest.approx(150.0, abs=10.0)
    assert match.y == pytest.approx(150.0, abs=10.0)
    assert match.score >= 0.5


def test_find_shape_model_with_precomputed_target_pyramid_matches_default_result():
    # Gerçek kullanıcı raporu: "şekil bulma çok kasıyor" -- birden fazla model aynı karede
    # aranırken arama görüntüsünün gradyan piramidi artık `build_search_pyramid` ile BİR KEZ
    # kurulup paylaşılabiliyor (bkz. `operators/builtin/shape_match.py`). Bu, sonucu
    # DEĞİŞTİRMEMELİ -- sadece yeniden hesaplamayı atlıyor.
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    baseline = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)

    pyramid = build_search_pyramid(search, len(model.levels))
    with_pyramid = find_shape_model(
        search, model, min_score=0.5, greediness=0.7, max_matches=1, target_pyramid=pyramid
    )

    assert len(with_pyramid) == len(baseline) == 1
    assert with_pyramid[0].x == pytest.approx(baseline[0].x)
    assert with_pyramid[0].y == pytest.approx(baseline[0].y)
    assert with_pyramid[0].angle == pytest.approx(baseline[0].angle)
    assert with_pyramid[0].score == pytest.approx(baseline[0].score)


def test_find_shape_model_reuses_pyramid_slice_for_model_with_fewer_levels():
    """`target_pyramid` daha DERİN (fazla seviyeli) bir modelden geldiyse, daha az seviyeli
    bir model için ilk N seviyesinin dilimlenmesi yeterli olmalı -- bu, `shape_match.py`'nin
    farklı `num_levels`'lı modelleri AYNI (en derin) piramitle paylaşmasının dayandığı
    varsayım."""
    reference = _reference_image()
    shallow_model = train_shape_model(reference, RoiRect(50, 50, 100, 100), num_levels=1)
    deep_model = train_shape_model(reference, RoiRect(50, 50, 100, 100), num_levels=3)
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    deep_pyramid = build_search_pyramid(search, len(deep_model.levels))
    baseline = find_shape_model(search, shallow_model, min_score=0.5, greediness=0.7, max_matches=1)
    with_shared_pyramid = find_shape_model(
        search, shallow_model, min_score=0.5, greediness=0.7, max_matches=1, target_pyramid=deep_pyramid
    )

    assert len(with_shared_pyramid) == len(baseline) == 1
    assert with_shared_pyramid[0].x == pytest.approx(baseline[0].x)
    assert with_shared_pyramid[0].y == pytest.approx(baseline[0].y)


def test_score_map_reuses_cached_kernel_for_same_level_and_angle():
    """Canlı kamerada `find_shape_model` her tick'te aynı sabit açı ızgarasını tarıyor —
    `_build_direction_kernels`'in aynı `ShapeLevel` + aynı açı için İKİNCİ kez ÇAĞRILMAMASI
    gerekir (bkz. `_kernel_cache_for_level`), sadece SONUCU önbellekten okunmalı."""
    model = _train()
    level = model.levels[0]
    cos_map = np.ones((50, 50), dtype=np.float32)
    sin_map = np.zeros((50, 50), dtype=np.float32)
    calls = []
    original = shape_matching_module._build_direction_kernels

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(shape_matching_module, "_build_direction_kernels", spy)
        _score_map(cos_map, sin_map, level, 12.5)
        assert len(calls) == 1
        _score_map(cos_map, sin_map, level, 12.5)
        assert len(calls) == 1  # ikinci çağrı önbellekten döndü
        _score_map(cos_map, sin_map, level, 20.0)
        assert len(calls) == 2  # farklı açı -> yeni kernel


def test_build_direction_kernels_handles_points_all_on_one_side_of_center():
    """Gerçek kullanıcı raporu: Poligon ile eğitilen bir modelle 'Şekil Bul' çalıştırınca
    `cv2.filter2D`'nin OpenCV assertion'ı (`anchor.inside(Rect(0,0,ksize.width,ksize.height))`)
    ile ÇÖKÜYORDU. Kök neden: `_build_direction_kernels`'in çekirdek sınırları (`width`/`height`)
    SADECE noktaların gerçek min/max'ından hesaplanıyordu -- ofset (0,0) (`anchor`'ın temsil
    ettiği model merkezi) bu sınıra dahil olmak ZORUNDA değildi. Bu test, TÜM noktaları merkeze
    göre kesinlikle POZİTİF x-ofsetli (`min_dx=5 > 0`) sahte bir nokta bulutuyla -- düzeltmeden
    ÖNCE tam olarak bu senaryoyu üreten girdiyle -- çağırıp anchor'ın HER ZAMAN çekirdek
    sınırları İÇİNDE kaldığını doğruluyor."""
    points = np.array([[5.0, 5.0], [8.0, 5.0], [8.0, 8.0], [5.0, 8.0]])  # hepsi x>=5, y>=5
    angles = np.zeros(points.shape[0])

    kernel_cos, kernel_sin, (anchor_col, anchor_row) = shape_matching_module._build_direction_kernels(
        points, angles, angle_offset_deg=0.0
    )

    height, width = kernel_cos.shape
    assert kernel_sin.shape == (height, width)
    assert 0 <= anchor_col < width
    assert 0 <= anchor_row < height


def test_score_map_does_not_crash_when_all_points_are_offset_to_one_side():
    """Yukarıdaki testin `cv2.filter2D` ÇAĞRISI üzerinden (gerçek çökme yolu) doğrulanması --
    düzeltmeden önce bu `cv2.error` (`-215:Assertion failed ... normalizeAnchor`) fırlatırdı."""
    level = shape_matching_module.ShapeLevel(
        points=np.array([[5.0, 5.0], [8.0, 5.0], [8.0, 8.0], [5.0, 8.0]]),
        angles=np.zeros(4),
    )
    cos_map = np.ones((50, 50), dtype=np.float32)
    sin_map = np.zeros((50, 50), dtype=np.float32)

    score_map = _score_map(cos_map, sin_map, level, angle_offset_deg=0.0)

    assert score_map.shape == cos_map.shape


def test_find_shape_model_full_angle_sweep_does_not_crash_for_asymmetric_polygon_contour():
    """Uçtan uca regresyon: elle çizilmiş (Poligon), ROI'nin naif geometrik merkezine göre
    ASİMETRİK bir kontur/maskeyle eğitilmiş küçük bir model, `geom.shape_match`'in yaptığı gibi
    TAM 360° taransa bile (`angle_start=-180, angle_extent=360`) HİÇBİR açıda çökmemeli --
    gerçek kullanıcı senaryosunun (küçük/asimetrik Poligon konturu + tam açı taraması) doğrudan
    simülasyonu."""
    reference = np.full((150, 150), 255, dtype=np.uint8)
    # Asimetrik "L" biçimli bir şekil: ROI'nin (50,50,50,50) bbox merkezi (75,75) iken şeklin
    # kütlesi bbox'ın SOL-ÜST köşesine yakın toplanıyor.
    cv2.rectangle(reference, (55, 55), (65, 95), color=0, thickness=-1)
    cv2.rectangle(reference, (55, 85), (95, 95), color=0, thickness=-1)
    roi = RoiRect(50, 50, 50, 50)
    mask = np.zeros(reference.shape, dtype=bool)
    mask[50:100, 50:100] = True  # ROI'nin tamamı -- asimetri şeklin KENDİSİNDEN geliyor

    model = train_shape_model(reference, roi, mask=mask)

    search = np.full((300, 300), 255, dtype=np.uint8)
    cv2.rectangle(search, (155, 155), (165, 195), color=0, thickness=-1)
    cv2.rectangle(search, (155, 185), (195, 195), color=0, thickness=-1)

    # Çökme olmadan tamamlanması yeterli -- eşleşme bulunup bulunmaması bu testin konusu DEĞİL.
    find_shape_model(
        search, model, angle_start=-180.0, angle_extent=360.0, angle_step_coarse=2.0,
        min_score=0.3, greediness=0.7, max_matches=None,
    )


def test_find_shape_model_returns_empty_on_blank_image():
    model = _train()
    blank = np.full((200, 200), 255, dtype=np.uint8)

    matches = find_shape_model(blank, model, min_score=0.5, greediness=0.7, max_matches=1)

    assert matches == []


def test_find_shape_model_finds_multiple_instances():
    model = _train()
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 10.0)
    _draw_triangle(search, (220, 200), -60.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=2)

    assert len(matches) == 2
    positions = {(round(m.x / 10) * 10, round(m.y / 10) * 10) for m in matches}
    assert (80, 80) in positions
    assert (220, 200) in positions


def test_render_match_overlay_returns_bgr_image_same_size():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)
    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)
    entries = [LabeledMatch(label=f"ucgen{i}", model=model, match=m) for i, m in enumerate(matches, start=1)]

    overlay = render_match_overlay(search, entries)

    assert overlay.shape == (200, 200, 3)


def test_render_match_overlay_handles_no_matches():
    search = np.full((200, 200), 255, dtype=np.uint8)

    overlay = render_match_overlay(search, [])

    assert overlay.shape == (200, 200, 3)


def test_render_match_overlay_draws_points_at_transformed_edge_locations():
    """HALCON tarzı yeniden tasarım (bkz. modül docstring'i, `get_generic_shape_model_result_
    object(..., 'contours')` deseni): overlay artık ayrı bir Otsu/siluet-kontur çıkarımı
    KULLANMIYOR -- modelin KENDİ eğitilmiş kenar noktalarını (`levels[0].points`) bulunan
    pozisyona taşıyıp DOĞRUDAN çiziyor. Bu test, bilinen (elle kurulmuş) birkaç noktanın TAM
    konumunun boyalı olduğunu, noktalardan uzak bir bölgenin boyanmadığını piksel bazında
    doğruluyor."""
    model = _train()  # SADECE seviye sayısını/`corners`'ı almak için -- noktaları KENDİMİZ kuracağız
    search = np.zeros((200, 200, 3), dtype=np.uint8)
    match = shape_matching_module.MatchResult(x=100.0, y=100.0, angle=0.0, score=1.0)

    known_points = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    known_level = shape_matching_module.ShapeLevel(points=known_points, angles=np.zeros(4))
    known_model = shape_matching_module.ShapeModel(
        levels=[known_level] * len(model.levels), corners=model.corners
    )

    overlay = render_match_overlay(search, [LabeledMatch(label="1", model=known_model, match=match)])

    for dx, dy in known_points:
        assert overlay[int(100 + dy), int(100 + dx)].tolist() != [0, 0, 0]
    # Noktalardan uzak, merkez artı-işareti/eksen çizgisinden de uzak bir bölge boyanmamalı.
    assert overlay[150, 150].tolist() == [0, 0, 0]


def test_render_match_overlay_does_not_connect_disjoint_point_clusters():
    """Gerçek kullanıcı raporu (önceki, artık aşılmış bir çözüm denemesiyle çözülmeye
    çalışılan sorun): "eti yazısını seçtim... kontürde elediğim kısım şekil bulmada
    gözükmemeli" -- HALCON tarzı nokta-bulutu çiziminde bu KÖKTEN çözülür: her nokta bağımsız
    bir daire olarak çizildiğinden, birbirinden UZAK iki nokta kümesi arasına ASLA bir çizgi
    çizilmez (eski çok-parçalı poligon/convexHull sisteminin çözmeye çalıştığı özel bir durum
    değil, mimarinin doğal bir sonucu)."""
    model = _train()
    search = np.zeros((200, 200, 3), dtype=np.uint8)
    match = shape_matching_module.MatchResult(x=100.0, y=100.0, angle=0.0, score=1.0)

    # Model-merkezli (offset) koordinatlar -- match.x/y (100,100) ile toplanınca cluster_a
    # görüntüde x∈[10,15], cluster_b x∈[185,190] olur, ikisi de y∈[70,80]'de -- merkez artı-
    # işareti/eksen çizgisinden (satır 100, sütun 94-106) bilerek UZAK tutuldu.
    cluster_a = np.array([[-90.0, -30.0], [-85.0, -30.0], [-90.0, -20.0], [-85.0, -20.0]])
    cluster_b = np.array([[85.0, -30.0], [90.0, -30.0], [85.0, -20.0], [90.0, -20.0]])
    points = np.concatenate([cluster_a, cluster_b], axis=0)
    custom_level = shape_matching_module.ShapeLevel(points=points, angles=np.zeros(len(points)))
    custom_model = shape_matching_module.ShapeModel(
        levels=[custom_level] * len(model.levels), corners=model.corners
    )

    overlay = render_match_overlay(search, [LabeledMatch(label="1", model=custom_model, match=match)])

    # İki küme ARASINDAKİ orta nokta boyanmamalı.
    assert overlay[75, 100].tolist() == [0, 0, 0]
    # Ama kümelerin KENDİLERİ (noktaların TAM konumları) gerçekten çizilmiş olmalı.
    assert overlay[70, 10].tolist() != [0, 0, 0]  # cluster_a'nın ilk noktası (dx=-90,dy=-30)
    assert overlay[70, 185].tolist() != [0, 0, 0]  # cluster_b'nin ilk noktası (dx=85,dy=-30)


def test_render_match_overlay_scales_points_with_match_scale():
    """`match.scale` (ölçek araması sonucu) noktaların merkeze göre ofsetlerini büyütüp/
    küçültmeli -- ölçek 1.0 iken çizilen bir nokta, ölçek 2.0 iken merkeze göre İKİ KATI
    uzaklıkta çizilmeli."""
    model = _train()
    search = np.zeros((200, 200, 3), dtype=np.uint8)

    # Dikey ofset (dx=0, dy=20) BİLEREK seçildi -- merkez artı-işareti/eksen çizgisi (satır
    # 100, sütun ~94-150) SADECE satır 100 civarını etkiler, satır 120/140 bundan bağımsızdır.
    known_points = np.array([[0.0, 20.0]])
    known_level = shape_matching_module.ShapeLevel(points=known_points, angles=np.zeros(1))
    known_model = shape_matching_module.ShapeModel(
        levels=[known_level] * len(model.levels), corners=model.corners
    )

    match_1x = shape_matching_module.MatchResult(x=100.0, y=100.0, angle=0.0, score=1.0, scale=1.0)
    match_2x = shape_matching_module.MatchResult(x=100.0, y=100.0, angle=0.0, score=1.0, scale=2.0)

    overlay_1x = render_match_overlay(search, [LabeledMatch(label="1", model=known_model, match=match_1x)])
    overlay_2x = render_match_overlay(search, [LabeledMatch(label="1", model=known_model, match=match_2x)])

    assert overlay_1x[120, 100].tolist() != [0, 0, 0]  # ofset 20 -> y=120 (ölçek 1.0)
    assert overlay_1x[140, 100].tolist() == [0, 0, 0]  # ölçek 1.0'da y=140'a hiç ulaşmaz
    assert overlay_2x[140, 100].tolist() != [0, 0, 0]  # ofset 20*2=40 -> y=140 (ölçek 2.0)


def test_render_match_overlay_preserves_color_input_instead_of_turning_grayscale():
    """Gerçek kullanıcı raporu: "şekil bul özelliği ... kamerayı siyah-beyaza çeviriyor" --
    kök neden `render_match_overlay`'in taban görüntüyü HER ZAMAN önce griye çevirmesiydi
    (eşleştirme puanlaması içeride griye dayansa da, overlay'in TABANI orijinal renkli kareyi
    korumalı, `ml.onnx_detect::render_detection_overlay`'deki gibi)."""
    model = _train()
    gray_search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(gray_search, (130, 80), 30.0)
    color_search = cv2.cvtColor(gray_search, cv2.COLOR_GRAY2BGR)
    color_search[:, :, 2] = 255  # sadece kırmızı kanalı doyur -- gerçekten renkli, gri DEĞİL
    color_search[:, :, 0] = 0

    overlay = render_match_overlay(color_search, [])

    assert overlay.shape == (200, 200, 3)
    # Griye çevrilip geri BGR'ye dönüştürülseydi 3 kanal EŞİT olurdu -- burada kanallar
    # birbirinden FARKLI olmalı (renk korunmuş).
    assert not np.array_equal(overlay[:, :, 0], overlay[:, :, 2])


def test_render_match_overlay_draws_label_text_for_each_entry():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)
    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)
    entries = [LabeledMatch(label="ucgen1", model=model, match=matches[0])]

    with_label = render_match_overlay(search, entries)
    without_label = render_match_overlay(search, [])

    # Etiket/kontur çizildiği için piksel farklılığı olmalı (aynı görüntü dönmemeli).
    assert not np.array_equal(with_label, without_label)


def test_find_shape_model_auto_mode_returns_all_matches_above_threshold():
    """min_score=0.7 (operatörün gerçek varsayılanı) kullanılıyor: otomatik modda (max_matches=
    None) TÜM eşiği geçen adaylar döner — düşük bir eşikle (ör. 0.5) bu üçgen model için yanlış
    açıdaki zayıf-ama-eşik-üstü sahte adaylar da (gerçek nesne değil) sonuca girebilir; bu,
    NMS'in bir hatası değil, 'otomatik'in tanımının doğal bir sonucudur (eşik altına düşen hiçbir
    şey elenmeden döner) — bu yüzden gerçekçi bir eşik kullanmak önemlidir."""
    model = _train()
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 10.0)
    _draw_triangle(search, (220, 200), -60.0)

    matches = find_shape_model(search, model, min_score=0.7, greediness=0.7, max_matches=None)

    assert len(matches) == 2


def test_find_shape_model_does_not_return_duplicate_detections_for_same_object():
    """Regresyon testi: NMS mesafe eşiği model boyutuna göre ölçeklenmeden önce, büyük/uzun bir
    model için aynı fiziksel nesnenin etrafında birden fazla yakın-konumlu aday birleştirilmeden
    ayrı 'eşleşme' olarak dönüyordu (kullanıcı raporu: "aynı görüntüyü birden fazla kez seçiyor")."""
    reference = np.full((300, 300), 255, dtype=np.uint8)
    _draw_elongated_shape(reference, (150, 150), 0.0)
    model = train_shape_model(reference, RoiRect(60, 110, 190, 80))

    search = np.full((400, 400), 255, dtype=np.uint8)
    _draw_elongated_shape(search, (150, 150), 20.0)
    _draw_elongated_shape(search, (280, 280), -50.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=10)

    assert len(matches) == 2


def test_nms_merges_same_position_candidates_despite_differing_angle():
    """Regresyon testi: eskiden NMS konum YETERLİ olsa bile açı farkı bir eşiği (1°) aşarsa
    adayları birleştirmiyordu — kaba/ince arama adımlarının ürettiği küçük açı sapmalarıyla
    AYNI fiziksel nesne için birden fazla 'eşleşme' dönüyordu (gerçek kullanıcı raporu: "aynı
    cismi farklı açılardan tespit edip tekrar tekrar gösteriyor"). Artık açı hiç dikkate
    alınmıyor — konum tek başına yeterli."""
    candidates = [
        [100.0, 100.0, 10.0, 1.0, 0.95],
        [100.5, 99.5, 25.0, 1.0, 0.90],  # aynı konum, 15° farklı açı -- eskiden AYRI kalırdı
    ]

    merged = _nms(candidates, dist_thresh=3.0)

    assert len(merged) == 1
    assert merged[0][4] == pytest.approx(0.95)  # en yüksek skorlu aday tutulur


def test_nms_keeps_distinct_positions_separate_regardless_of_angle():
    candidates = [[100.0, 100.0, 10.0, 1.0, 0.95], [200.0, 100.0, 10.0, 1.0, 0.90]]

    merged = _nms(candidates, dist_thresh=3.0)

    assert len(merged) == 2


def test_find_shape_model_keeps_close_but_distinct_objects_separate():
    """NMS eşiğinin model boyutuna oranlanması, gerçekten AYRI ama birbirine yakın iki nesneyi
    yanlışlıkla tek eşleşmeye indirmemeli."""
    reference = np.full((300, 300), 255, dtype=np.uint8)
    _draw_elongated_shape(reference, (150, 150), 0.0)
    model = train_shape_model(reference, RoiRect(60, 110, 190, 80))

    search = np.full((500, 500), 255, dtype=np.uint8)
    _draw_elongated_shape(search, (150, 150), 0.0)
    _draw_elongated_shape(search, (300, 150), 0.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=10)

    assert len(matches) == 2


def test_find_shape_model_recovers_noisy_off_step_angle_previously_missed():
    """Regresyon testi: gürültülü bir görüntüde, kaba açı adımları arasına düşen bir dönüşte
    (10.5°, eski 5°'lik adımlarla TAM ORTADA) model eskiden (kaba eşiğin nihai `min_score`'a
    doğrudan bağlı olduğu ve dar ±3px iyileştirme penceresiyle) hiç bulunamıyordu — gerçek
    kullanıcı raporu: "nesne bulunamıyor/kaçırılıyor". `_COARSE_ACCEPT_LOOSENING` ve genişletilmiş
    `_REFINE_XY_RADIUS` ile artık bulunmalı; aşağıdaki `if` bloğu bunu eski sabitlerle geçici
    olarak simüle edip gerçekten eski davranışta KAÇIRILDIĞINI da doğrular (düzeltmenin gerçekten
    hedeflenen kusuru çözdüğünü kanıtlamak için)."""
    reference = np.full((300, 300), 255, dtype=np.uint8)
    _draw_elongated_shape(reference, (150, 150), 0.0)
    model = train_shape_model(reference, RoiRect(60, 110, 190, 80))

    search = np.full((400, 400), 255, dtype=np.uint8)
    _draw_elongated_shape(search, (150, 150), 10.5)
    noise = np.random.default_rng(0).normal(0, 15, search.shape)
    search = np.clip(search.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    import imgflow.core.shape_matching as shape_matching_module

    old_loosening = shape_matching_module._COARSE_ACCEPT_LOOSENING
    old_radius = shape_matching_module._REFINE_XY_RADIUS
    try:
        shape_matching_module._COARSE_ACCEPT_LOOSENING = 1.0
        shape_matching_module._REFINE_XY_RADIUS = 3
        pre_fix_matches = find_shape_model(
            search, model, angle_step_coarse=5.0, min_score=0.8, greediness=0.85, max_matches=1
        )
    finally:
        shape_matching_module._COARSE_ACCEPT_LOOSENING = old_loosening
        shape_matching_module._REFINE_XY_RADIUS = old_radius
    assert pre_fix_matches == []

    matches = find_shape_model(search, model, min_score=0.8, greediness=0.85, max_matches=1)

    assert len(matches) == 1
    assert matches[0].angle == pytest.approx(10.5, abs=1.0)
    assert matches[0].score >= 0.8


def _draw_elongated_shape(image: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    base = np.array([[-60, -10], [60, -10], [60, 10], [-60, 10], [70, 0]], dtype=np.float64)
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (base @ rot.T) + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)
