import json

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QInputDialog, QMessageBox

from imgflow.core import capture_store
from imgflow.core.roi import RoiRect
from imgflow.ui.dialogs.shape_matching_dialog import _GUIDE_TEXT, ShapeMatchingDialog

_BASE_TRIANGLE = np.array([[0, -40], [35, 25], [-20, 30]], dtype=np.float64)


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


def _load_reference_directly(dialog: ShapeMatchingDialog, image: np.ndarray, roi=(50, 50, 100, 100)) -> None:
    """`QFileDialog.getOpenFileNames` gerçek bir dosya seçici açar, headless testte kullanılamaz
    — `_on_load_references`'ın diskten yükleme kısmını atlayıp aynı state'i doğrudan kuruyoruz."""
    dialog._reference_image = image
    dialog._canvas.set_image(image)
    dialog._roi = roi
    dialog._canvas.set_roi(*roi)


def test_dialog_opens_at_a_large_default_size(qtbot):
    # Gerçek kullanıcı raporu: "görüntü çok küçük ROI seçmekte zorlanıyorum". Eski Qt varsayılan
    # (içeriğe sığdırma) boyutlandırması küçük bir pencere üretiyordu.
    dialog = ShapeMatchingDialog(model_dir=None, parent=None)
    qtbot.addWidget(dialog)

    assert dialog.width() >= 900
    assert dialog.height() >= 700


def test_canvas_is_wrapped_in_scroll_area_with_zoom_controls(qtbot):
    dialog = ShapeMatchingDialog(model_dir=None, parent=None)
    qtbot.addWidget(dialog)

    assert dialog._canvas_scroll_area.widget() is dialog._canvas
    assert dialog._canvas._scroll_host is dialog._canvas_scroll_area


def test_dialog_can_be_shrunk_well_below_default_size_and_delete_button_stays_reachable(qtbot):
    """Gerçek kullanıcı raporu: "menünün boyutlandırması kötü olduğu için aşağıdaki sil
    butonlarını göremiyormuşum... istersem pencereyi büyültüp küçültebileyim". Eskiden TÜM
    kontroller (yükle/çizim modu/eğit/kayıtlı modeller) dialog'un ANA `QVBoxLayout`'una
    doğrudan ekleniyordu -- dialog'un minimum boyutu bunların toplam `sizeHint`'ine
    dayanıyordu, pencere bunun altına küçültülemiyordu. Kontrol paneli artık kendi
    `QScrollArea`'sına sarılı olduğundan dialog çok daha küçük bir yüksekliğe küçültülebilmeli
    VE "Sil" butonu (artık `controls_scroll_area`'nın İÇİNDE) hâlâ var/erişilebilir olmalı."""
    dialog = ShapeMatchingDialog(model_dir=None, parent=None)
    qtbot.addWidget(dialog)
    dialog.show()

    # Eski davranışta minimum yükseklik ~800px civarındaydı (tüm kontroller sığdırılmaya
    # çalışıldığından); artık kontrol paneli kaydırılabilir olduğundan çok daha küçük olmalı.
    assert dialog.minimumSizeHint().height() < 400

    dialog.resize(1000, 300)
    delete_buttons = [
        w for w in dialog.findChildren(type(dialog._save_button)) if w.text() == "Sil"
    ]
    assert len(delete_buttons) == 1
    assert delete_buttons[0].isVisible()


def test_zoom_in_updates_percent_label(qtbot):
    dialog = ShapeMatchingDialog(model_dir=None, parent=None)
    qtbot.addWidget(dialog)
    dialog.show()  # zoom, scroll host'un gerçek bir viewport boyutuna sahip olmasını gerektirir
    _load_reference_directly(dialog, _reference_image())

    dialog._canvas.zoom_in()

    assert dialog._zoom_percent_label.text() != "%100"
    assert dialog._canvas.zoom > 1.0


def test_activating_a_new_reference_resets_zoom_to_fit(qtbot):
    """Gerçek kullanıcı raporu: "yüklediğim görüntü çok küçük kalabiliyor" -- kök neden bir
    ÖNCEKİ referansta uzaklaştırma kullanılmışsa `_zoom`'un sıfırlanmadan kalıp YENİ referansa
    da uygulanmasıydı. Galeriden yeni bir referans aktive edilince (`_activate_reference` ->
    `_use_reference_image`) zoom HER ZAMAN 1.0'a (sığdır) sıfırlanmalı."""
    dialog = ShapeMatchingDialog(model_dir=None, parent=None)
    qtbot.addWidget(dialog)
    dialog.show()
    index_a = dialog._add_reference_to_gallery("a.png", _reference_image())
    dialog._activate_reference(index_a)
    dialog._canvas.zoom_out()
    dialog._canvas.zoom_out()
    assert dialog._canvas.zoom != 1.0

    index_b = dialog._add_reference_to_gallery("b.png", _reference_image())
    dialog._activate_reference(index_b)

    assert dialog._canvas.zoom == 1.0


def test_guide_label_present_and_nonempty(qtbot):
    dialog = ShapeMatchingDialog(model_dir=None, parent=None)
    qtbot.addWidget(dialog)

    labels = [w for w in dialog.findChildren(type(dialog._train_status_label)) if w.text() == _GUIDE_TEXT]
    assert len(labels) == 1


def test_train_without_reference_image_shows_warning(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_train()

    assert len(warnings) == 1
    assert dialog.model is None


def test_train_produces_model_and_enables_save(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())

    dialog._on_train()

    assert dialog.model is not None
    assert dialog._save_button.isEnabled()


def test_train_on_blank_roi_shows_error_status(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    blank = np.full((200, 200), 255, dtype=np.uint8)
    _load_reference_directly(dialog, blank)

    dialog._on_train()

    assert dialog.model is None
    assert "başarısız" in dialog._train_status_label.text()


def _has_point_near(model, roi_center: tuple[float, float], region: tuple[float, float, float, float]) -> bool:
    """`region` = (x0, y0, x1, y1), mutlak görüntü koordinatında. `model`'in ilk (tam
    çözünürlük) seviyesindeki herhangi bir noktanın bu dikdörtgene düşüp düşmediğini kontrol
    eder -- `max_points_per_level` kırpması iki eğitim arasındaki HAM nokta sayısını
    eşitleyebildiğinden (ikisi de üst sınıra çarpabilir), asıl kanıt NEREDE nokta olduğudur."""
    cx, cy = roi_center
    abs_x = model.levels[0].points[:, 0] + cx
    abs_y = model.levels[0].points[:, 1] + cy
    x0, y0, x1, y1 = region
    return bool(((abs_x >= x0) & (abs_x <= x1) & (abs_y >= y0) & (abs_y <= y1)).any())


def test_auto_contour_checkbox_uses_detected_object_mask_and_reduces_noise_points(qtbot, tmp_path):
    """ROI içinde nesnenin (parlak üçgen) YANINDA ayrı bir gürültü kenarı (küçük parlak kare)
    olsun; 'Konturu Otomatik Algıla' işaretliyken bu gürültü modele hiç girmemeli."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    # Otomatik nesne tespiti "parlak taraf = nesne" varsayımını kullanır (bkz. auto_objects.py) --
    # bu yüzden burada (üstteki `_reference_image()`'in aksine) nesne AÇIK, arka plan KOYU.
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    noise_region = (55.0, 55.0, 75.0, 75.0)
    roi_center = (100.0, 100.0)  # RoiRect(50, 50, 100, 100) merkezi

    dialog._on_train()
    assert dialog.model is not None
    assert _has_point_near(dialog.model, roi_center, noise_region)

    dialog._auto_contour_checkbox.setChecked(True)
    dialog._on_train()

    assert dialog.model is not None
    assert "Kontur bulunamadı" not in dialog._train_status_label.text()
    assert not _has_point_near(dialog.model, roi_center, noise_region)


def test_build_auto_contour_mask_calls_detect_objects_with_robust_true(qtbot, monkeypatch, tmp_path):
    """Gerçek kullanıcı isteği: "daha sağlam olsun, biraz yavaşlasa da olur" -- eğitim TEK
    SEFERLİK olduğundan `_build_auto_contour_mask` her zaman `detect_objects(..., robust=True)`
    çağırmalı (kenar-destekli daha sağlam tespit, bkz. `core/auto_objects.py`)."""
    import imgflow.ui.dialogs.shape_matching_dialog as dialog_module

    received_kwargs = {}
    original = dialog_module.detect_objects

    def spy(*args, **kwargs):
        received_kwargs.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(dialog_module, "detect_objects", spy)

    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._auto_contour_checkbox.setChecked(True)

    dialog._on_train()

    assert received_kwargs.get("robust") is True


def test_trained_filter_preview_shows_masked_background_as_black_when_checkbox_checked(qtbot, tmp_path):
    """Gerçek kullanıcı raporu: "bence eğitmiyor... eğittiği modelin filtrelenmiş halini de
    yanda göstersin" -- kullanıcı 'Konturu Otomatik Algıla'nın eğitimi GERÇEKTEN etkileyip
    etkilemediğine görsel kanıt olmadan güvenmiyor. Checkbox işaretliyken eğitim sonrası
    `_build_filtered_training_preview` (yeni önizleme etiketinin arkasındaki hesaplama), ROI
    içindeki gürültü bölgesini (nesnenin DIŞINDA, elenmesi gereken) SİYAHA boyamalı; nesnenin
    kendisi (üçgenin merkezi) DEĞİŞMEDEN kalmalı."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    dialog._auto_contour_checkbox.setChecked(True)

    dialog._on_train()

    assert dialog.model is not None
    preview = dialog._build_filtered_training_preview(RoiRect(50, 50, 100, 100), dialog._last_train_mask)
    # ROI (50,50,100,100) içinde mutlak (65,65) -> kırpımda (15,15) -- gürültü karesinin merkezi.
    assert preview[15, 15] == 0  # elenmiş (siyah)
    # ROI merkezi (100,100) -> kırpımda (50,50) -- üçgenin içi, korunmalı.
    assert preview[50, 50] != 0


def test_trained_filter_preview_shows_raw_crop_when_checkbox_unchecked(qtbot, tmp_path):
    """Checkbox KAPALIYKEN (varsayılan) önizleme HİÇ filtrelenmemiş (ham) ROI kırpımını
    göstermeli -- kullanıcı checkbox'ı açıp kapatarak GÖRSEL farkı doğrudan karşılaştırabilsin."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    assert dialog._auto_contour_checkbox.isChecked() is False

    dialog._on_train()

    assert dialog.model is not None
    preview = dialog._build_filtered_training_preview(RoiRect(50, 50, 100, 100), dialog._last_train_mask)
    roi = RoiRect(50, 50, 100, 100)
    raw_crop = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
    assert np.array_equal(preview, raw_crop)


def test_trained_filter_preview_caption_warns_when_invert_checkbox_checked(qtbot, tmp_path):
    """Gerçek kullanıcı raporu: "kırpılmış bölge siyah gözüküyor ama hâlâ şekil bulurken o
    bölgeyi de seçiyor" -- kök neden 'Ters Çevir' işaretliyken önizlemede SİYAH görünenin
    NESNENİN KENDİSİ olması (arka plan DEĞİL) ve overlay'in bu modda BİLİNÇLİ olarak
    dikdörtgene düşmesiydi, ama kullanıcıya hiç AÇIKLANMIYORDU. Yeni caption bunu açıkça
    söylemeli."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    dialog._auto_contour_checkbox.setChecked(True)

    dialog._on_train()
    assert "Ters Çevir" not in dialog._trained_filter_preview_caption.text()
    assert dialog.model is not None

    dialog._invert_mask_checkbox.setChecked(True)
    dialog._on_train()

    assert dialog.model is not None
    assert "Ters Çevir" in dialog._trained_filter_preview_caption.text()


def test_trained_filter_preview_caption_shows_unfiltered_note_when_mask_none(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    assert dialog._auto_contour_checkbox.isChecked() is False

    dialog._on_train()

    assert dialog._trained_filter_preview_caption.text() == "(filtre uygulanmadı)"


def test_circle_draw_mode_sets_canvas_shape_and_trains_from_circular_mask(qtbot, tmp_path):
    """Gerçek kullanıcı isteği: "dikdörtgenin yanında çember şeklinde de bir roi istiyorum."
    Çember modu seçilince canvas 'CIRCLE' şekline geçmeli. Eğitilen nokta bulutu (ARTIK
    overlay'in KENDİSİ) çemberin (yarıçap 60, çap 120) sınırını DEĞİL, çember İÇİNDEKİ gerçek
    nesnenin (yarıçap 40, çap ~80) GERÇEK kenarlarını yansıtmalı -- HALCON tarzı yeniden
    tasarımın doğal sonucu: kontur artık ROI'nin kendi şeklinden değil, gerçekten eğitilen
    Sobel kenarlarından gelir."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(image, (100, 100), 40, color=255, thickness=-1)
    _load_reference_directly(dialog, image)

    dialog._draw_mode_combo.setCurrentIndex(2)
    assert dialog._canvas._shape == "CIRCLE"
    dialog._roi_circle = (100, 100, 60)
    dialog._canvas.set_roi_circle(100, 100, 60)

    dialog._on_train()

    assert dialog.model is not None
    all_pts = dialog.model.levels[0].points
    span = all_pts[:, 0].max() - all_pts[:, 0].min()
    assert 60 < span < 100  # gerçek nesnenin çapına (~80) yakın, ROI çemberinin çapına (120) DEĞİL


def test_circle_draw_mode_auto_contour_shrinks_to_object_within_circle(qtbot, tmp_path):
    """Çember modunda 'Konturu Otomatik Algıla' işaretliyse, çemberin İÇİNDEKİ ASIL nesne
    (üçgen) tespit edilip nesnenin DIŞINDAKİ ayrı bir gürültü kenarı eğitim nokta filtresinden
    ELENMELİ -- Dikdörtgen moddaki `test_auto_contour_checkbox_uses_detected_object_mask_and_
    reduces_noise_points` ile AYNI davranışın Çember modundaki karşılığı. (Not: tek bir temiz
    nesne varken span'in zaten dar olması -- HALCON tarzı yeniden tasarımda noktalar HER ZAMAN
    gerçek kenarları yansıttığından -- checkbox'tan bağımsız olur; bu yüzden checkbox'ın asıl
    ayırt edici etkisi GÜRÜLTÜ ELEME senaryosunda görülür.)"""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # Çemberin (yarıçap 60, merkez (100,100)) İÇİNDE ama üçgenin DIŞINDA ayrı bir gürültü kenarı.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    noise_region = (55.0, 55.0, 75.0, 75.0)
    roi_center = (100.0, 100.0)

    dialog._draw_mode_combo.setCurrentIndex(2)
    dialog._roi_circle = (100, 100, 60)
    dialog._canvas.set_roi_circle(100, 100, 60)

    dialog._on_train()
    assert dialog.model is not None
    assert _has_point_near(dialog.model, roi_center, noise_region)

    dialog._auto_contour_checkbox.setChecked(True)
    dialog._on_train()

    assert dialog.model is not None
    assert "Kontur bulunamadı" not in dialog._train_status_label.text()
    assert not _has_point_near(dialog.model, roi_center, noise_region)


def test_shrink_roi_to_contour_tightens_rect_roi_around_detected_object(qtbot, tmp_path):
    """Yeni önerilen yöntem, kullanıcı onayladı: "roi elle olsun şekille olsun seçildikten
    sonra otomatik kontür özelliği çalışmalı ... cismin etrafını çizebilir ve yeni roi olarak
    belirleyebiliriz." 'ROI'yi Nesneye Daralt' butonu, geniş bir ROI içindeki küçük bir
    nesnenin sınırlayıcı kutusunu YENİ ROI olarak atamalı."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(image, (90, 90), (110, 110), color=255, thickness=-1)  # 21x21 nesne
    _load_reference_directly(dialog, image, roi=(50, 50, 100, 100))

    dialog._on_shrink_roi_to_contour()

    x, y, w, h = dialog._roi
    assert 85 <= x <= 91
    assert 85 <= y <= 91
    assert w <= 30
    assert h <= 30


def test_shrink_roi_to_contour_leaves_roi_unchanged_when_no_object_found(qtbot, monkeypatch, tmp_path):
    import imgflow.ui.dialogs.shape_matching_dialog as dialog_module

    monkeypatch.setattr(dialog_module, "detect_objects", lambda *a, **k: [])
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image(), roi=(50, 50, 100, 100))

    dialog._on_shrink_roi_to_contour()

    assert dialog._roi == (50, 50, 100, 100)
    assert "bulunamadı" in dialog._train_status_label.text()


def test_invert_mask_checkbox_trains_on_background_instead_of_detected_object(qtbot, tmp_path):
    """Gerçek kullanıcı isteği: "seçtiği kontürün içini mi dışını mı seçeceğim ... onu
    seçebilmek istiyorum" -- 'Kontur Dışını Kullan' işaretliyken üstteki testin TERSİ olmalı:
    nesnenin (üçgen) dışındaki gürültü kenarı artık modele GİRMELİ, üçgenin kendisi elenmeli."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    noise_region = (55.0, 55.0, 75.0, 75.0)
    roi_center = (100.0, 100.0)

    dialog._auto_contour_checkbox.setChecked(True)
    dialog._invert_mask_checkbox.setChecked(True)
    dialog._on_train()

    assert dialog.model is not None
    assert "Kontur bulunamadı" not in dialog._train_status_label.text()
    assert _has_point_near(dialog.model, roi_center, noise_region)


def test_auto_contour_checkbox_shows_live_excluded_region_preview(qtbot, tmp_path):
    """Gerçek kullanıcı raporu: "elediği arka planı göremiyorum, onu da görmek istiyorum" --
    checkbox işaretlenince canvas'ta elenen bölgeyi işaretleyen bir maske (kırmızı önizleme)
    canlı olarak kurulmalı; nesnenin kendisi (üçgenin merkezi) maskeden HARİÇ tutulmalı."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)

    assert dialog._canvas._contour_preview is None

    dialog._auto_contour_checkbox.setChecked(True)

    preview = dialog._canvas._contour_preview
    assert preview is not None
    assert preview.shape == image.shape
    assert preview[65, 65]  # gürültü karesinin merkezi -- elenmeli
    assert not preview[100, 100]  # üçgenin merkezi -- korunmalı

    dialog._auto_contour_checkbox.setChecked(False)
    assert dialog._canvas._contour_preview is None


def test_invert_mask_checkbox_flips_which_region_the_preview_marks_as_excluded(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)

    dialog._auto_contour_checkbox.setChecked(True)
    normal = dialog._canvas._contour_preview
    assert not normal[100, 100]  # üçgenin merkezi -- korunur
    assert normal[65, 65]  # gürültü karesi -- elenir

    dialog._invert_mask_checkbox.setChecked(True)
    inverted = dialog._canvas._contour_preview
    assert inverted[100, 100]  # artık üçgenin merkezi elenir
    assert not inverted[65, 65]  # gürültü karesi artık korunur
    assert "dışı" in dialog._contour_preview_label.text()

    dialog._invert_mask_checkbox.setChecked(False)
    assert not dialog._canvas._contour_preview[100, 100]
    assert "içi" in dialog._contour_preview_label.text()


def test_switching_to_polygon_mode_clears_contour_preview(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    _load_reference_directly(dialog, image)
    dialog._auto_contour_checkbox.setChecked(True)
    assert dialog._canvas._contour_preview is not None

    dialog._draw_mode_combo.setCurrentIndex(1)

    assert dialog._canvas._contour_preview is None


def test_auto_contour_preview_shows_fallback_note_when_no_object_found(qtbot, monkeypatch, tmp_path):
    import imgflow.ui.dialogs.shape_matching_dialog as dialog_module

    monkeypatch.setattr(dialog_module, "detect_objects", lambda *a, **k: [])

    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())

    dialog._auto_contour_checkbox.setChecked(True)

    assert dialog._canvas._contour_preview is None
    assert "Kontur bulunamadı" in dialog._contour_preview_label.text()


def test_auto_contour_checkbox_falls_back_to_full_roi_when_no_object_found(qtbot, monkeypatch, tmp_path):
    import imgflow.ui.dialogs.shape_matching_dialog as dialog_module

    monkeypatch.setattr(dialog_module, "detect_objects", lambda *a, **k: [])

    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._auto_contour_checkbox.setChecked(True)

    dialog._on_train()

    assert dialog.model is not None
    assert "Kontur bulunamadı" in dialog._train_status_label.text()


def test_polygon_mode_incomplete_contour_shows_inline_error_and_does_not_train(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._draw_mode_combo.setCurrentIndex(1)

    dialog._on_train()

    assert dialog.model is None
    assert "poligon" in dialog._train_status_label.text().lower()


def test_polygon_mode_trains_model_from_closed_contour(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._draw_mode_combo.setCurrentIndex(1)

    tri_points_abs = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(int)
    dialog._canvas._polygon_points = [tuple(int(v) for v in p) for p in tri_points_abs]
    dialog._canvas._polygon_closed = True

    dialog._on_train()

    assert dialog.model is not None
    assert dialog._save_button.isEnabled()


def test_polygon_mode_auto_contour_checkbox_reduces_noise_points(qtbot, tmp_path):
    """Gerçek kullanıcı isteği: "bu seçenekler kalsın ama hepsi üst üste kullanılabilsin
    fonksiyonlu olsun istiyorum" -- "Konturu Otomatik Algıla" eskiden Poligon modunda
    TAMAMEN GİZLİ/devre dışıydı. Artık Poligon modunda da işaretlenebilir: çizilen poligonun
    İÇİNDE tespit edilen nesneyle kesişim alınıp, poligonun içindeki ama nesnenin DIŞINDAKİ
    bir gürültü kenarı elenir -- Dikdörtgen/Çember modlarındaki AYNI davranışın Poligon
    karşılığı."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # Poligonun (50,50)-(150,150) dikdörtgeni) İÇİNDE ama üçgenin DIŞINDA ayrı bir gürültü kenarı.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    noise_region = (55.0, 55.0, 75.0, 75.0)
    roi_center = (100.0, 100.0)

    dialog._draw_mode_combo.setCurrentIndex(1)
    dialog._canvas._polygon_points = [(50, 50), (150, 50), (150, 150), (50, 150)]
    dialog._canvas._polygon_closed = True

    dialog._on_train()
    assert dialog.model is not None
    assert _has_point_near(dialog.model, roi_center, noise_region)

    dialog._auto_contour_checkbox.setChecked(True)
    dialog._on_train()

    assert dialog.model is not None
    assert "Kontur bulunamadı" not in dialog._train_status_label.text()
    assert not _has_point_near(dialog.model, roi_center, noise_region)


def test_auto_contour_handles_disjoint_multi_part_object_like_text(qtbot, tmp_path):
    """Gerçek kullanıcı raporu: "eti yazısını seçtim ve dışını kontür bulma yardımıyla attım...
    sadece eti yazısını öğrettim modele" -- ama overlay hâlâ geniş bir dikdörtgene benziyordu.
    Kök neden: metnin harfleri (`eti` gibi) birbirinden AYRIK bağlı bileşenlerdir; eskiden
    `_build_auto_contour_mask` sadece EN BÜYÜK bileşeni "nesne" sayıp diğer harfleri arka plan
    gibi elip HEM eğitim nokta filtresinden HEM overlay konturundan düşürüyordu (kontur tek bir
    harfe sıkışıp kalıyor, bu da o harf küçükse görsel olarak "hâlâ geniş" bir alan bırakıyordu).
    Bu test üç AYRI parlak blok (harf benzetmesi) ile eğitim nokta filtresinin (ARTIK
    overlay'in KENDİSİ, bkz. `core/shape_matching.py::render_match_overlay`) HEPSİNİ
    kapsadığını doğruluyor -- tek bir harfe sıkışmış olsaydı overlay yine "geniş" görünürdü."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    # Üç ayrı "harf" bloğu, aralarında koyu boşluklarla -- roi (40,70,120,60) hepsini kapsıyor.
    letters = [(50, 80, 20, 40), (85, 85, 15, 30), (110, 80, 20, 40)]  # x, y, w, h
    for x, y, w, h in letters:
        cv2.rectangle(image, (x, y), (x + w, y + h), color=255, thickness=-1)
    roi = (40, 70, 120, 60)
    _load_reference_directly(dialog, image, roi=roi)
    dialog._auto_contour_checkbox.setChecked(True)

    dialog._on_train()

    assert dialog.model is not None
    assert "Kontur bulunamadı" not in dialog._train_status_label.text()
    roi_center = (roi[0] + roi[2] / 2, roi[1] + roi[3] / 2)
    # Eğitim nokta filtresi HER üç harfin de bir kenarını içermeli, sadece birini değil.
    for x, y, w, h in letters:
        region = (x - 2.0, y - 2.0, x + w + 2.0, y + h + 2.0)
        assert _has_point_near(dialog.model, roi_center, region), f"nokta yok: {region}"
    # Eğitilen nokta bulutunun (overlay'in KENDİSİ) TOPLAM yayılımı ROI'nin tam genişliğine
    # (120) DEĞİL, üç harfin sıkı zarfına düşmeli -- tek bir harfe sıkışmış olsaydı genişliği
    # en fazla ~20px olurdu.
    all_pts = dialog.model.levels[0].points
    span = all_pts[:, 0].max() - all_pts[:, 0].min()
    assert span > 60  # üç harfin toplam genişliğine yakın
    assert span < 90  # ROI'nin tam genişliği (120) değil


def test_polygon_close_button_enables_after_clicking_three_points(qtbot, tmp_path):
    """Gerçek kullanıcı raporu: "elle roi çizme de şekil öğretmede çalışmıyor en az 3 nokta
    çizmeme rağmen ... en sonunda noktalar birleşmiyor". Kök neden: `RoiCanvas` nokta eklerken
    `polygon_changed` yayınlamıyordu, dialog'un `_on_polygon_changed` (butonun `setEnabled`
    durumunu günceleyen TEK yer) sadece kapatma/sürükleme bitince tetikleniyordu -- kullanıcı
    gerçek fare tıklamalarıyla 3+ nokta çizse bile "Poligonu Kapat" butonu hep devre dışı
    kalıyordu. Bu test, doğrudan `_polygon_points`'i elle doldurmak yerine GERÇEK
    `qtbot.mouseClick` olaylarıyla çizip butonun etkinleştiğini doğruluyor."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog.show()
    _load_reference_directly(dialog, _reference_image())
    dialog._draw_mode_combo.setCurrentIndex(1)

    assert not dialog._polygon_close_button.isEnabled()

    from PySide6.QtCore import QPoint, Qt

    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(40, 40))
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(120, 40))
    assert not dialog._polygon_close_button.isEnabled()
    qtbot.mouseClick(dialog._canvas, Qt.MouseButton.LeftButton, pos=QPoint(80, 120))

    assert dialog._polygon_close_button.isEnabled()
    qtbot.mouseClick(dialog._polygon_close_button, Qt.MouseButton.LeftButton)
    assert dialog._canvas.is_polygon_closed()


def test_checkbox_off_does_not_filter_training_points_rect_mode(qtbot, tmp_path):
    """"Konturu Otomatik Algıla" işaretli olmadıkça eğitimde KULLANILAN nokta filtresi (mask)
    uygulanmamalı -- ROI içindeki gürültü kenarı (nesnenin dışında ayrı bir kare) checkbox
    kapalıyken modele HÂLÂ girmeli (gerçek kullanıcı isteği: "diğer kısımlar aynı
    kalabilir")."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = np.zeros((200, 200), dtype=np.uint8)
    tri_points = (_BASE_TRIANGLE + np.array((100.0, 100.0))).astype(np.int32)
    cv2.fillPoly(image, [tri_points], color=255)
    # `robust=True` (bkz. `_build_auto_contour_mask`) kenar-tabanlı doldurma ile ince/boş
    # (2px kalınlığında) bir dikdörtgenin GÖRÜNÜR alanını büyütür (Otsu-yalnız 176px iken
    # robust ~432px) -- 15x15 yerine 10x10 kullanmak, büyüdükten sonra bile üçgenin %15
    # önem eşiğinin AÇIKÇA altında (~%10.7) kalmasını garanti eder.
    cv2.rectangle(image, (60, 60), (70, 70), color=255, thickness=2)
    _load_reference_directly(dialog, image)
    noise_region = (55.0, 55.0, 75.0, 75.0)
    roi_center = (100.0, 100.0)
    assert dialog._auto_contour_checkbox.isChecked() is False

    dialog._on_train()

    assert dialog.model is not None
    assert _has_point_near(dialog.model, roi_center, noise_region)


def test_invert_mask_checkbox_flips_finalize_mask_for_both_rect_and_polygon_paths(qtbot, tmp_path):
    """`_finalize_mask` hem Dikdörtgen (Otomatik Kontur) hem Poligon yolunun ORTAK son adımı --
    poligon modunda canlı önizleme olmadığından (bkz. `test_switching_to_polygon_mode_clears_
    contour_preview`) ve tam eğitim ardışık düzeninde kontur sınırına yakın noktalarda Sobel/
    genişletme etkileşiminin kesin geometriyi belirsizleştirmesi nedeniyle, ters çevirmenin
    'içeri/dışarı' semantiğini doğrudan bu paylaşılan fonksiyon üzerinden -- sınırdan uzak, açık
    bölgelerde -- doğruluyoruz."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    inside = np.zeros((60, 60), dtype=bool)
    inside[15:45, 15:45] = True

    normal = dialog._finalize_mask(inside)
    assert normal[30, 30]  # kontur içi -- korunur
    assert not normal[2, 2]  # kontur dışı, sınırdan uzak -- elenir

    dialog._invert_mask_checkbox.setChecked(True)
    inverted = dialog._finalize_mask(inside)
    assert not inverted[30, 30]  # artık kontur içi elenir
    assert inverted[2, 2]  # artık kontur dışı korunur


def test_reference_file_dropped_loads_image_same_as_open_dialog(qtbot, tmp_path):
    """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) canvas'a sürükleyip
    bırakmak `_on_load_references` (dosya seçici) ile AYNI sonucu üretmeli."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image = _reference_image()
    path = tmp_path / "reference.png"
    cv2.imwrite(str(path), image)

    dialog._on_reference_file_dropped(str(path))

    assert dialog._reference_image is not None
    assert np.array_equal(dialog._reference_image, image)


def test_reference_file_dropped_with_missing_file_shows_error(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    dialog._on_reference_file_dropped(str(tmp_path / "does_not_exist.png"))

    assert len(errors) == 1
    assert dialog._reference_image is None


def test_load_references_adds_multiple_files_to_gallery(qtbot, monkeypatch, tmp_path):
    """Kullanıcı isteği: 'model eğit kısmına daha fazla fotoğraf yüklenebilmeli' — birden
    fazla dosya TEK diyalog çağrısında galeriye eklenebilmeli, en son yüklenen otomatik aktif
    referans olmalı."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    image_a, image_b = _reference_image(), _reference_image()
    path_a, path_b = tmp_path / "a.png", tmp_path / "b.png"
    cv2.imwrite(str(path_a), image_a)
    cv2.imwrite(str(path_b), image_b)
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getOpenFileNames",
        lambda *a, **k: ([str(path_a), str(path_b)], ""),
    )

    dialog._on_load_references()

    assert dialog._reference_list.count() == 2
    assert len(dialog._reference_images) == 2
    assert np.array_equal(dialog._reference_image, image_b)  # sonuncusu aktif referans olur


def test_active_reference_is_marked_unambiguously_in_gallery(qtbot, tmp_path):
    """Kullanıcı isteği: 'model eğitme kısmında açının doğru alınması için bir tane referans
    seçelim' — birden fazla foto arasında hangisinin şu an eğitim referansı (dolayısıyla
    modelin 'açı=0' pozu) olduğu galeride '★' öneki ve ayrı bir etiketle AÇIKÇA görünmeli."""
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    dialog._add_reference_to_gallery(str(tmp_path / "first.png"), _reference_image())
    dialog._add_reference_to_gallery(str(tmp_path / "second.png"), _reference_image())

    dialog._activate_reference(0)

    assert dialog._reference_list.item(0).text().startswith("★ ")
    assert not dialog._reference_list.item(1).text().startswith("★ ")
    assert "first.png" in dialog._active_reference_label.text()

    dialog._activate_reference(1)

    assert not dialog._reference_list.item(0).text().startswith("★ ")
    assert dialog._reference_list.item(1).text().startswith("★ ")
    assert "second.png" in dialog._active_reference_label.text()


def test_clicking_gallery_item_switches_active_reference(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    first = _reference_image()
    second = np.full((200, 200), 0, dtype=np.uint8)
    dialog._add_reference_to_gallery(str(tmp_path / "first.png"), first)
    dialog._add_reference_to_gallery(str(tmp_path / "second.png"), second)
    dialog._use_reference_image(second)  # ikincisi yüklendikten sonra aktif oldu

    dialog._on_reference_gallery_item_clicked(dialog._reference_list.item(0))

    assert np.array_equal(dialog._reference_image, first)


def test_load_references_with_missing_file_shows_error_and_skips(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getOpenFileNames",
        lambda *a, **k: ([str(tmp_path / "does_not_exist.png")], ""),
    )

    dialog._on_load_references()

    assert len(errors) == 1
    assert dialog._reference_list.count() == 0
    assert dialog._reference_image is None


def test_save_without_name_shows_warning(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._on_train()
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a))

    dialog._on_save()

    assert len(warnings) == 1


def test_save_and_load_round_trip_via_combo(qtbot, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _load_reference_directly(dialog, _reference_image())
    dialog._on_train()
    dialog._name_edit.setText("ucgen_modeli")

    dialog._on_save()

    assert "Kaydedildi" in dialog._save_status_label.text()

    other_dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(other_dialog)
    assert other_dialog._load_combo.findText("ucgen_modeli") >= 0

    other_dialog._load_combo.setCurrentText("ucgen_modeli")
    other_dialog._on_load_model()

    assert other_dialog.model is not None
    assert "Yüklendi" in other_dialog._save_status_label.text()


def _train_and_save(dialog, name, monkeypatch=None):
    _load_reference_directly(dialog, _reference_image())
    dialog._on_train()
    dialog._name_edit.setText(name)
    dialog._on_save()


def test_delete_model_removes_it_from_combo_and_emits_models_changed(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "silinecek_model")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    changed = []
    dialog.models_changed.connect(lambda: changed.append(1))

    dialog._load_combo.setCurrentText("silinecek_model")
    dialog._on_delete_model()

    assert dialog._load_combo.findText("silinecek_model") == -1
    assert "Silindi" in dialog._save_status_label.text()
    assert changed == [1]
    assert dialog.model is None
    assert not dialog._save_button.isEnabled()


def test_delete_model_cancelled_keeps_it(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "kalacak_model")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)

    dialog._load_combo.setCurrentText("kalacak_model")
    dialog._on_delete_model()

    assert dialog._load_combo.findText("kalacak_model") >= 0


def test_delete_model_filesystem_error_shows_message_and_keeps_model(qtbot, monkeypatch, tmp_path):
    """Gerçek kullanıcı raporu: "öğrettiğim şekilleri silemiyorum" -- kök neden adayı:
    `delete_shape_model` diskte bir OSError (dosya kilitli/salt-okunur) fırlatırsa bu ÖNCEDEN
    hiç yakalanmıyordu, kullanıcıya hiçbir geri bildirim gitmeden sessizce hiçbir şey
    olmuyormuş gibi görünüyordu. Artık hata inline bir QMessageBox.critical ile gösterilir,
    model listede/bellekte KALIR (silinmiş sayılmaz)."""
    import imgflow.ui.dialogs.shape_matching_dialog as dialog_module

    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "kilitli_model")
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)

    def _raise_permission_error(*a, **k):
        raise PermissionError("Erişim engellendi")

    monkeypatch.setattr(dialog_module.shape_model_store, "delete_shape_model", _raise_permission_error)
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a) or QMessageBox.StandardButton.Ok)

    dialog._load_combo.setCurrentText("kilitli_model")
    dialog._on_delete_model()

    assert len(errors) == 1
    assert "silinemedi" in errors[0][2].lower()
    assert dialog._load_combo.findText("kilitli_model") >= 0
    assert "Silindi" not in dialog._save_status_label.text()


def test_rename_model_updates_combo_and_emits_models_changed(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "eski_isim")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("yeni_isim", True))
    changed = []
    dialog.models_changed.connect(lambda: changed.append(1))

    dialog._load_combo.setCurrentText("eski_isim")
    dialog._on_rename_model()

    assert dialog._load_combo.findText("yeni_isim") >= 0
    assert dialog._load_combo.findText("eski_isim") == -1
    assert changed == [1]
    assert dialog._name_edit.text() == "yeni_isim"


def test_rename_model_to_existing_name_shows_error(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "model_a")
    dialog._name_edit.clear()
    _train_and_save(dialog, "model_b")
    monkeypatch.setattr(QInputDialog, "getText", lambda *a, **k: ("model_b", True))
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    dialog._load_combo.setCurrentText("model_a")
    dialog._on_rename_model()

    assert len(errors) == 1
    assert dialog._load_combo.findText("model_a") >= 0
    assert dialog._load_combo.findText("model_b") >= 0


def test_export_model_writes_json_file(qtbot, monkeypatch, tmp_path):
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    _train_and_save(dialog, "disa_aktarilan")
    export_path = tmp_path / "disari" / "model.json"
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(export_path), "JSON (*.json)"),
    )
    export_path.parent.mkdir()

    dialog._load_combo.setCurrentText("disa_aktarilan")
    dialog._on_export_model()

    assert export_path.exists()
    data = json.loads(export_path.read_text(encoding="utf-8"))
    assert data["name"] == "disa_aktarilan"
    assert "levels" in data["model"]
    assert "dışa aktarıldı" in dialog._save_status_label.text()


def test_import_model_from_json_adds_to_combo_and_store(qtbot, monkeypatch, tmp_path):
    source_dialog = ShapeMatchingDialog(model_dir=tmp_path / "source", parent=None)
    qtbot.addWidget(source_dialog)
    _train_and_save(source_dialog, "kaynak_model")
    export_path = tmp_path / "kaynak_model.json"
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(export_path), "JSON (*.json)"),
    )
    source_dialog._load_combo.setCurrentText("kaynak_model")
    source_dialog._on_export_model()
    assert export_path.exists()

    target_dir = tmp_path / "target"
    dialog = ShapeMatchingDialog(model_dir=target_dir, parent=None)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(export_path), "JSON (*.json)"),
    )
    changed = []
    dialog.models_changed.connect(lambda: changed.append(1))

    dialog._on_import_model()

    assert dialog.model is not None
    assert dialog._name_edit.text() == "kaynak_model"
    assert dialog._load_combo.findText("kaynak_model") >= 0
    assert changed == [1]
    assert "İçe aktarıldı" in dialog._save_status_label.text()

    reloaded = ShapeMatchingDialog(model_dir=target_dir, parent=None)
    qtbot.addWidget(reloaded)
    assert reloaded._load_combo.findText("kaynak_model") >= 0


def test_import_model_with_malformed_json_shows_error_and_does_not_crash(qtbot, monkeypatch, tmp_path):
    bad_path = tmp_path / "bozuk.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(bad_path), "JSON (*.json)"),
    )
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    dialog._on_import_model()

    assert len(errors) == 1
    assert dialog.model is None


def test_import_recipe_json_shows_specific_guidance_not_raw_keyerror(qtbot, monkeypatch, tmp_path):
    """Regresyon: kullanıcı bir PİPELİNE REÇETESİNİ (Dosya > Reçeteyi Kaydet...) şekil modeli
    'İçe Aktar'a vermeye çalışıp ham `KeyError: 'levels'` almıştı (`recipe.py::graph_to_dict`
    hiç piksel verisi saklamaz). Artık bu durum ayrıca tanınıp kullanıcıya ne yapması
    gerektiğini (önce 'Görüntüyü Dışa Aktar...' ile resme çevirmesi) söyleyen özel bir
    mesaj gösteriliyor."""
    recipe_path = tmp_path / "kulaklik.json"
    recipe_path.write_text(
        json.dumps({"schema_version": 2, "nodes": [], "edges": []}), encoding="utf-8"
    )
    dialog = ShapeMatchingDialog(model_dir=tmp_path, parent=None)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        "imgflow.ui.dialogs.shape_matching_dialog.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(recipe_path), "JSON (*.json)"),
    )
    errors = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: errors.append(a))

    dialog._on_import_model()

    assert len(errors) == 1
    message = errors[0][2]
    assert "REÇETE" in message
    assert "levels" not in message
    assert dialog.model is None


def test_capture_button_saves_active_reference_to_gallery(qtbot, monkeypatch, tmp_path):
    """Gerçek kullanıcı isteği: "ölçüm de dahil her alanda kare yakalayıp yan ekrana
    atabilmek istiyorum" -- bu diyalogda canlı kamera YOK, bu yüzden aktif referans
    görüntü Lens/Yükseklik-Ölçek diyaloglarındaki AYNI `capture_store` deposuna yazılır."""
    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")
    dialog = ShapeMatchingDialog(model_dir=tmp_path / "shape_models", parent=None)
    qtbot.addWidget(dialog)
    assert not dialog._capture_button.isEnabled()

    index = dialog._add_reference_to_gallery("a.png", _reference_image())
    dialog._activate_reference(index)
    assert dialog._capture_button.isEnabled()

    captured = []
    dialog.frame_captured.connect(lambda: captured.append(True))
    qtbot.mouseClick(dialog._capture_button, Qt.MouseButton.LeftButton)

    assert captured == [True]
    records = capture_store.list_captures(source="shape_match")
    assert len(records) == 1
