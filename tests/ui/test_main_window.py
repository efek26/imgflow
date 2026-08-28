import copy

import cv2
import numpy as np
import pytest

from imgflow.core.camera_source import BaslerCameraSource
from imgflow.core.errors import ImgflowError
from imgflow.core.params import ParamType
from imgflow.io_utils import flatfield_store
from imgflow.ui import main_window as main_window_module
from imgflow.ui.main_window import MainWindow, _build_preview_frame
from tests.support.fake_genicam import FakeGenicamNode, FakeNodeMap, default_camera_nodes


def _sample_image_path(tmp_path):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return path


def _run_camera_tick_and_wait(window, qtbot, timeout=2000):
    """`_on_camera_tick`'in ağır kısmı (`_build_preview_frame`) artık `_LiveTickWorker`'da
    ARKA PLANDA çalışıyor (bkz. gerçek kullanıcı raporu: "şekil bul çok kasıyor ve kamerayı
    siyah-beyaza çeviriyor" -- kök neden UI thread'in tamamen bloklanmasıydı). `image_view`/
    `results_panel` gibi widget'ların dispatch edilen sonucu YANSITMASINI bekleyen testler
    doğrudan `window._on_camera_tick()` çağırmak yerine bunu kullanmalı -- aksi halde worker
    henüz sonucu teslim etmeden assert çalışır. Kamera/tick'e bağımlı olmayan (senkron
    `_maybe_apply_auto_mm_per_px`/`inject_result`/enum galerisi gibi) durumları kontrol eden
    testler bu yardımcıya ihtiyaç DUYMAZ."""
    window._on_camera_tick()
    qtbot.waitUntil(lambda: not window._live_worker_busy, timeout=timeout)


def _get_normal_base_image_for(window):
    return main_window_module._get_normal_base_image(
        window.graph, window.engine, window.registry, window._camera_source is not None, window._last_camera_frame
    )


def _compose_display_image_for(window, filtered_image, measurements, mm_per_px, node_id=None):
    return main_window_module._compose_display_image(
        filtered_image,
        measurements,
        mm_per_px,
        node_id,
        view_mode=window._view_mode,
        graph=window.graph,
        pipeline_order=window.pipeline.order,
        engine=window.engine,
        registry=window.registry,
        camera_active=window._camera_source is not None,
        last_camera_frame=window._last_camera_frame,
    )


class _FakeCameraSource:
    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        if not self._frames:
            return None
        return self._frames.pop(0)

    def release(self):
        self.released = True


class _FakeBaslerCameraSource(BaslerCameraSource):
    """Gerçek pypylon donanımına ihtiyaç duymadan BaslerCameraSource'u taklit eder."""

    def __init__(self, node_map):
        self._node_map = node_map
        self.released = False

    def read(self):
        return None

    def release(self):
        self.released = True

    @property
    def node_map(self):
        return self._node_map


def test_load_recipe_auto_applies_calibration_profile(qtbot, tmp_path, monkeypatch):
    from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint
    from imgflow.io_utils import calibration_store
    from imgflow.io_utils.recipe import save_recipe

    height_model = HeightScaleModel()
    height_model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    height_model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    height_model.fit()

    def fake_load_profile(name, directory=None):
        assert name == "hat1"
        return calibration_store.CalibrationProfileData(
            lens_profile=None, height_model=height_model, created_at="2026-01-01T00:00:00", operator_note=None
        )

    monkeypatch.setattr(calibration_store, "load_profile", fake_load_profile)

    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window.graph.calibration_profile = "hat1"
    window.graph.calibration_height_mm = 75.0
    recipe_path = tmp_path / "recipe.json"
    save_recipe(recipe_path, window.graph)

    window.load_recipe_from(str(recipe_path))

    assert window._height_scale_model is height_model
    assert window._active_height_mm == 75.0
    region_props_node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert region_props_node.params["mm_per_px"] == pytest.approx(1.0 / height_model.predict_scale(75.0))


def test_load_calibration_profile_returns_true_and_applies_on_success(qtbot, monkeypatch):
    from imgflow.core.lens_calibration import LensProfile
    from imgflow.io_utils import calibration_store

    profile = LensProfile(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros((5, 1)), image_size=(640, 480), rms_error=0.1
    )

    def fake_load_profile(name, directory=None):
        assert name == "hat1"
        return calibration_store.CalibrationProfileData(
            lens_profile=profile, height_model=None, created_at="2026-01-01T00:00:00", operator_note=None
        )

    monkeypatch.setattr(calibration_store, "load_profile", fake_load_profile)

    window = MainWindow()
    qtbot.addWidget(window)

    result = window._load_calibration_profile("hat1")

    assert result is True
    assert window._active_lens_profile is profile


def test_load_calibration_profile_returns_false_when_missing(qtbot, monkeypatch):
    from imgflow.io_utils import calibration_store

    def fake_load_profile(name, directory=None):
        raise FileNotFoundError(name)

    monkeypatch.setattr(calibration_store, "load_profile", fake_load_profile)

    window = MainWindow()
    qtbot.addWidget(window)

    result = window._load_calibration_profile("does_not_exist")

    assert result is False


def test_on_load_calibration_profile_menu_action_updates_status_and_graph(qtbot, monkeypatch):
    from imgflow.core.lens_calibration import LensProfile
    from imgflow.io_utils import calibration_store

    profile = LensProfile(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros((5, 1)), image_size=(640, 480), rms_error=0.1
    )
    monkeypatch.setattr(calibration_store, "list_profiles", lambda *a, **k: ["hat1", "hat2"])
    monkeypatch.setattr(
        calibration_store,
        "load_profile",
        lambda name, directory=None: calibration_store.CalibrationProfileData(
            lens_profile=profile, height_model=None, created_at=None, operator_note=None
        ),
    )
    monkeypatch.setattr(
        "imgflow.ui.main_window.QInputDialog.getItem", lambda *a, **k: ("hat1", True)
    )

    window = MainWindow()
    qtbot.addWidget(window)

    window._on_load_calibration_profile()

    assert window.graph.calibration_profile == "hat1"
    assert "hat1" in window.status_label.text()
    assert window._active_lens_profile is profile


def test_on_load_calibration_profile_menu_action_reports_no_saved_profiles(qtbot, monkeypatch):
    from imgflow.io_utils import calibration_store

    monkeypatch.setattr(calibration_store, "list_profiles", lambda *a, **k: [])

    window = MainWindow()
    qtbot.addWidget(window)

    window._on_load_calibration_profile()

    assert "yok" in window.status_label.text()


def test_on_load_calibration_profile_menu_shows_mm_per_px_and_loads_correct_profile(qtbot, monkeypatch):
    """Profil seçim listesinde her isme "(X.XXX mm/px)" eklenir; kullanıcı bu etiketli
    seçeneği seçtiğinde, gerçek yüklenen profil yine de parantez OLMADAN doğru isimle
    eşleşmeli (bkz. `calibration_store.format_profile_label`/`profile_name_from_label`)."""
    from imgflow.core.lens_calibration import LensProfile
    from imgflow.io_utils import calibration_store

    profile_hat1 = LensProfile(
        camera_matrix=np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros((5, 1)),
        image_size=(640, 480),
        rms_error=0.1,
    )
    profile_hat2 = LensProfile(
        camera_matrix=np.array([[2000.0, 0.0, 320.0], [0.0, 2000.0, 240.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros((5, 1)),
        image_size=(640, 480),
        rms_error=0.1,
    )
    profiles = {
        "hat1": calibration_store.CalibrationProfileData(
            lens_profile=profile_hat1,
            height_model=None,
            created_at=None,
            operator_note=None,
            reference_distance_mm=420.0,  # 420/1000 = 0.420 mm/px
        ),
        "hat2": calibration_store.CalibrationProfileData(
            lens_profile=profile_hat2,
            height_model=None,
            created_at=None,
            operator_note=None,
            reference_distance_mm=1000.0,  # 1000/2000 = 0.500 mm/px
        ),
    }
    monkeypatch.setattr(calibration_store, "list_profiles", lambda *a, **k: ["hat1", "hat2"])
    monkeypatch.setattr(calibration_store, "load_profile", lambda name, directory=None: profiles[name])

    seen_items = []

    def fake_get_item(self, title, label, items, current, editable):
        seen_items.extend(items)
        return "hat2 (0.500 mm/px)", True

    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getItem", fake_get_item)

    window = MainWindow()
    qtbot.addWidget(window)

    window._on_load_calibration_profile()

    assert seen_items == ["hat1 (0.420 mm/px)", "hat2 (0.500 mm/px)"]
    assert window.graph.calibration_profile == "hat2"
    assert window._active_lens_profile is profile_hat2


def test_set_active_height_without_calibration_model_only_stores_height(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._set_active_height_mm(75.0)

    assert window._active_height_mm == 75.0
    assert window._active_mm_per_px() is None


def test_set_active_height_updates_region_props_mm_per_px(qtbot):
    from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint

    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")

    model = HeightScaleModel()
    model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    window._height_scale_model = model

    window._set_active_height_mm(75.0)

    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(1.0 / model.predict_scale(75.0))


def test_open_measurement_tool_uses_active_scale(qtbot):
    from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint

    window = MainWindow()
    qtbot.addWidget(window)
    model = HeightScaleModel()
    model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    window._height_scale_model = model
    window._active_height_mm = 75.0

    window._on_open_measurement_tool()

    assert window._measurement_tool_dialog is not None
    assert window._measurement_tool_dialog._mm_per_px == pytest.approx(1.0 / model.predict_scale(75.0))
    window._measurement_tool_dialog.close()


def test_open_measurement_tool_without_calibration_shows_no_mm(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_measurement_tool()

    assert window._measurement_tool_dialog is not None
    assert window._measurement_tool_dialog._mm_per_px is None
    window._measurement_tool_dialog.close()


def test_reopening_measurement_tool_does_not_leak_previous_dialog(qtbot):
    """Regresyon: `_on_open_measurement_tool` her çağrıda TAZE bir dialog kurar (tekil-dialog
    yeniden kullanım desenindeki gibi eskiyi `show()`la geri getirmez) -- eski `close()` C++
    nesnesini `WA_DeleteOnClose` olmadan yok ETMEDİĞİNDEN `deleteLater()` çağrılmazsa her
    yeniden açılış görüntü verisiyle birlikte bir dialog'u sonsuza kadar bellekte bırakırdı
    (tekrarlı aç/kapa stres testiyle doğrulandı)."""
    import shiboken6

    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_measurement_tool()
    first_dialog = window._measurement_tool_dialog
    assert first_dialog is not None

    window._on_open_measurement_tool()
    qtbot.wait(10)

    assert window._measurement_tool_dialog is not first_dialog
    assert not shiboken6.isValid(first_dialog)
    window._measurement_tool_dialog.close()


def test_capture_photo_stays_usable_while_measurement_tool_is_open(qtbot, tmp_path, monkeypatch):
    """Kullanıcının, başka bir asistanın kod okumadan yaptığı (doğrulanmamış) analizini relay
    ettiği istek: "ölçüm varken Kare Yakala kullanabilme". `MeasurementToolDialog` non-modal
    ve MainWindow'un `parent`'ı olarak açılıyor -- `_capture_photo_action`/`_capture_photo_
    button`'ın etkin/devre dışı durumu SADECE `start_camera`/`stop_camera`'ya bağlı (bkz.
    `_on_capture_camera_photo`), Ölçüm Aracı dialog'unun açık olup olmamasına HİÇ bakılmıyor
    -- bu test bunu, dialog AÇIKKEN gerçek bir yakalama yapıp doğrulayarak kanıtlıyor (kod
    okumadan "kavramsal olarak engelleyici bir neden yok" varsayımının GERÇEKTEN doğru
    olduğunu gösteren regresyon testi)."""
    from imgflow.core import capture_store

    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")

    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FakeCameraSource([np.full((10, 10, 3), 200, dtype=np.uint8)]))
    window._on_camera_tick()

    window._on_open_measurement_tool()
    assert window._measurement_tool_dialog is not None
    assert window._measurement_tool_dialog.isVisible()

    assert window._capture_photo_action.isEnabled() is True
    assert window._capture_photo_button.isEnabled() is True
    window._on_capture_camera_photo()

    assert len(capture_store.list_captures()) == 1
    assert "kaydedildi" in window.status_label.text()
    window._measurement_tool_dialog.close()


def test_height_scale_action_enabled_without_camera_and_opens_dialog(qtbot):
    """Gerçek kullanıcı isteği: "yaptığımız kalibrasyonu canlı görüntü olmadan da
    kullanabilmeliyim" -- bu aksiyon artık kamera HİÇ açılmadan da kullanılabilir/etkin."""
    window = MainWindow()
    qtbot.addWidget(window)

    assert window._height_scale_action.isEnabled() is True

    window._on_open_height_scale_calibration()
    assert window._height_scale_dialog is not None

    window.start_camera(_FakeCameraSource([]))
    assert window._height_scale_action.isEnabled() is True

    window.stop_camera()
    assert window._height_scale_action.isEnabled() is True
    assert window._height_scale_dialog is None


def test_height_scale_model_updated_signal_sets_active_model(qtbot):
    from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint

    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FakeCameraSource([]))
    window._on_open_height_scale_calibration()

    model = HeightScaleModel()
    model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    window._on_height_scale_model_updated(model)

    assert window._height_scale_model is model


def test_lens_calibration_action_enabled_without_camera(qtbot):
    """Gerçek kullanıcı isteği: "yaptığımız kalibrasyonu canlı görüntü olmadan da
    kullanabilmeliyim" -- bu aksiyon artık kamera hiç açılmadan da etkin."""
    window = MainWindow()
    qtbot.addWidget(window)

    assert window._lens_calibration_action.isEnabled() is True

    window.start_camera(_FakeCameraSource([]))
    assert window._lens_calibration_action.isEnabled() is True

    window.stop_camera()
    assert window._lens_calibration_action.isEnabled() is True


def test_lens_calibration_opens_without_camera_and_frame_provider_returns_none(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_lens_calibration()

    assert window._lens_calibration_dialog is not None
    assert window._camera_frame_provider() is None


def test_load_calibration_profile_action_enabled_without_camera(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window._load_calibration_profile_action.isEnabled() is True


def test_open_help_creates_and_shows_dialog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_help()

    assert window._help_dialog is not None
    assert window._help_dialog.isVisible() is True


def test_open_help_reuses_existing_dialog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_help()
    first_dialog = window._help_dialog
    window._on_open_help()

    assert window._help_dialog is first_dialog


def test_open_help_reopens_after_user_closes_it(qtbot):
    """Kullanıcı diyaloğu pencere X'iyle kapatıp (yalnızca gizler, yok etmez) Yardım menüsünden
    tekrar açtığında görünür hale gelmeli — `raise_()`/`activateWindow()` görünmez bir pencereyi
    GÖSTERMEZ, `show()` çağrılması gerekir (bkz. tüm tekil-diyalog `_on_open_*` metodları)."""
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_help()
    first_dialog = window._help_dialog
    first_dialog.close()
    assert first_dialog.isVisible() is False

    window._on_open_help()

    assert window._help_dialog is first_dialog
    assert window._help_dialog.isVisible() is True


def test_open_custom_filter_editor_creates_and_shows_dialog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_custom_filter_editor()

    assert window._custom_filter_dialog is not None
    assert window._custom_filter_dialog.isVisible() is True


def test_open_custom_filter_editor_reuses_existing_dialog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_custom_filter_editor()
    first_dialog = window._custom_filter_dialog
    window._on_open_custom_filter_editor()

    assert window._custom_filter_dialog is first_dialog


def test_open_shape_matching_creates_shows_and_reuses_dialog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_shape_matching()

    assert window._shape_matching_dialog is not None
    assert window._shape_matching_dialog.isVisible() is True

    first_dialog = window._shape_matching_dialog
    window._on_open_shape_matching()

    assert window._shape_matching_dialog is first_dialog


def test_open_shape_matching_reopens_after_user_closes_it(qtbot):
    """Kullanıcı 'Şekil Eşleştirme (Model Öğret)' penceresini modeli kaydetmeden (ör. yanlışlıkla
    X'e basıp) kapatırsa, Araçlar menüsünden tekrar açtığında pencere GÖRÜNÜR hale gelmeli — aksi
    halde `raise_()`/`activateWindow()` görünmez pencerede hiçbir şey yapmadığından araç bir daha
    hiç açılamıyormuş gibi görünürdü (bu tam olarak kullanıcı raporundaki "model eğitmeye
    geçerken/geçmeden sorun yaşama" senaryosu)."""
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_shape_matching()
    first_dialog = window._shape_matching_dialog
    first_dialog.close()
    assert first_dialog.isVisible() is False

    window._on_open_shape_matching()

    assert window._shape_matching_dialog is first_dialog
    assert window._shape_matching_dialog.isVisible() is True


def test_saving_custom_filter_refreshes_operator_library(qtbot, tmp_path, monkeypatch):
    from imgflow.core import custom_filters

    monkeypatch.setattr(custom_filters, "CUSTOM_FILTER_DIR", tmp_path / "custom_filters")
    try:
        window = MainWindow()
        qtbot.addWidget(window)

        window._on_open_custom_filter_editor()
        dialog = window._custom_filter_dialog
        dialog._name_edit.setText("Yeni Panel Filtresi")
        dialog._code_edit.setPlainText("def apply(image):\n    return image\n")
        dialog._on_save()

        op_id = custom_filters.op_id_for("Yeni Panel Filtresi")
        assert op_id in window.operator_library._items_by_op_id
    finally:
        for op_id in list(custom_filters.registry_module._ops):
            if op_id.startswith(custom_filters.OP_ID_PREFIX):
                custom_filters.registry_module.unregister(op_id)


def test_camera_tick_read_failure_does_not_crash_or_show_modal(qtbot, monkeypatch):
    """Kritik regresyon testi: `_camera_source.read()` bir istisna fırlatırsa (ör. TriggerMode
    kamerayı geçici olarak kare üretemez duruma soktuğunda) bu, 100ms'de bir çalışan bir
    QTimer slotu içinde oluyor. Önceden burada hiç yakalama yoktu ve global excepthook her
    tick'te yeni bir modal açıp kullanıcının uygulamayı kapatmasını bile engelleyen bir
    "diyalog fırtınası"na yol açıyordu (gerçek bir kullanıcı raporuyla doğrulandı)."""

    class _FailingCameraSource:
        def read(self):
            raise RuntimeError("kamera geçici olarak kare üretemiyor")

        def release(self):
            pass

    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FailingCameraSource())

    calls = []
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", lambda *a, **k: calls.append(1))

    for _ in range(5):
        window._on_camera_tick()  # patlamamalı

    assert calls == []
    assert "Kare okunamadı" in window.status_label.text()


def test_camera_tick_auto_reconnects_after_sustained_failure(qtbot, monkeypatch):
    """Kablo çekilip tekrar takıldığı senaryosu: `_camera_reconnect_factory` set edilmiş bir
    kaynak art arda `_CAMERA_DISCONNECT_THRESHOLD_TICKS` kadar başarısız olursa, factory
    tekrar çağrılıp `start_camera` ile YENİ bir kaynağa otomatik geçilmeli — hiçbir modal
    gösterilmeden (bkz. `_register_camera_failure` docstring'i)."""

    class _FailingThenRecoveringSource:
        def __init__(self):
            self.released = False

        def read(self):
            raise RuntimeError("kablo koptu")

        def release(self):
            self.released = True

    recovered_source = _FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8)])
    window = MainWindow()
    qtbot.addWidget(window)
    failing_source = _FailingThenRecoveringSource()
    window.start_camera(failing_source)
    window._camera_reconnect_factory = lambda: recovered_source

    calls = []
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", lambda *a, **k: calls.append(1))

    for _ in range(main_window_module._CAMERA_DISCONNECT_THRESHOLD_TICKS):
        window._on_camera_tick()

    assert calls == []
    assert window._camera_source is recovered_source
    assert failing_source.released is True
    assert "yeniden bağlandı" in window.status_label.text()


def test_camera_tick_does_not_reconnect_on_normal_basler_trigger_wait(qtbot):
    """Basler/GigE tetikleyici modda `read()`'in `None` dönmesi normaldir ("henüz tetiklenmedi")
    — bağlantı kopması DEĞİLDİR. Bu regresyon testi, sürekli `None` dönen bir Basler kaynağının
    YANLIŞLIKLA kopmuş sayılıp reconnect factory'i tetiklemediğini doğrular."""
    node_map = FakeNodeMap(default_camera_nodes())
    window = MainWindow()
    qtbot.addWidget(window)
    source = _FakeBaslerCameraSource(node_map)
    window.start_camera(source)

    factory_calls = []
    window._camera_reconnect_factory = lambda: factory_calls.append(1)

    for _ in range(main_window_module._CAMERA_DISCONNECT_THRESHOLD_TICKS * 2):
        window._on_camera_tick()

    assert factory_calls == []
    assert window._camera_source is source


def test_stop_camera_action_clears_reconnect_factory(qtbot):
    """Kullanıcı elle 'Kamerayı Durdur' derse otomatik yeniden bağlanma da tamamen
    devre dışı kalmalı — aksi halde durdurulmuş bir kamera arka planda kendini yeniden
    başlatmaya çalışırdı."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FakeCameraSource([]))
    window._camera_reconnect_factory = lambda: _FakeCameraSource([])

    window._on_stop_camera()

    assert window._camera_reconnect_factory is None
    assert window._camera_fail_streak == 0
    assert window._camera_reconnect_cooldown_ticks == 0


def test_camera_tick_throttles_enum_gallery_refresh(qtbot, monkeypatch):
    """Seçili adımda bir enum parametre varsa (`segment.threshold`'un 'mode'u), galerinin her
    seçeneği tam bir operatör çalıştırması gerektirir — bunu HER kamera tick'inde (100ms)
    çalıştırmak gereksiz CPU harcar. `_ENUM_GALLERY_TICK_STRIDE` kadar tick'te SADECE İLK
    tick'te yenilenmeli, `_ENUM_GALLERY_TICK_STRIDE + 1`. tick'te ikinci kez."""
    window = MainWindow()
    qtbot.addWidget(window)
    # `io.image_source` ÖNCEDEN eklenir — aksi halde ilk tick'te `_find_or_create_image_source`
    # onu ekleyip SEÇERDİ, seçimi 'th_id'den çalıp bu testi anlamsız kılardı.
    window.add_operator("io.image_source")
    th_id = window.add_operator("segment.threshold")
    window._on_step_selected(th_id)
    window.start_camera(_FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(10)]))

    calls = []
    monkeypatch.setattr(window.enum_gallery, "show_choices", lambda *a, **k: calls.append(1))

    for _ in range(main_window_module._ENUM_GALLERY_TICK_STRIDE):
        window._on_camera_tick()
    assert len(calls) == 1

    window._on_camera_tick()
    assert len(calls) == 2


def test_camera_tick_applies_undistort_when_lens_profile_set(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    calls = []

    def fake_undistort(frame, profile):
        calls.append(profile)
        return frame

    monkeypatch.setattr("imgflow.ui.main_window.undistort", fake_undistort)
    # `_maybe_apply_auto_mm_per_px` de her tick'te `_active_lens_profile.camera_matrix`'e
    # bakıyor (bkz. `_compute_auto_mm_per_px`) — opak bir sentinel yerine gerçekçi bir
    # LensProfile kullanılmalı, aksi halde undistort'tan bağımsız bir AttributeError alınır.
    lens_profile = _fx_1000_lens_profile()
    window._active_lens_profile = lens_profile
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    window.start_camera(_FakeCameraSource([frame]))

    window._on_camera_tick()

    assert calls == [lens_profile]


def test_camera_tick_skips_undistort_without_lens_profile(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    calls = []
    monkeypatch.setattr("imgflow.ui.main_window.undistort", lambda frame, profile: calls.append(1) or frame)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    window.start_camera(_FakeCameraSource([frame]))
    window._on_camera_tick()

    assert calls == []


def test_camera_settings_action_disabled_for_non_basler_camera(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.start_camera(_FakeCameraSource([]))

    assert window._camera_settings_action.isEnabled() is False
    assert window.camera_settings_panel._controller is None


def test_camera_settings_action_enabled_for_basler_camera_and_populates_panel(qtbot, tmp_path):
    window = MainWindow(camera_settings_dir=tmp_path)
    qtbot.addWidget(window)
    source = _FakeBaslerCameraSource(FakeNodeMap(default_camera_nodes()))

    window.start_camera(source)
    assert window._camera_settings_action.isEnabled() is True
    assert window.camera_settings_panel._controller is not None
    assert window.camera_settings_panel._toolbox.count() > 0

    window._on_open_camera_settings()
    assert window.right_tabs.currentWidget() is window.camera_settings_panel

    window.stop_camera()
    assert window._camera_settings_action.isEnabled() is False
    assert window.camera_settings_panel._controller is None


def test_stop_camera_auto_saves_settings_for_next_launch(qtbot, tmp_path):
    from imgflow.io_utils import camera_settings_store

    window = MainWindow(camera_settings_dir=tmp_path)
    qtbot.addWidget(window)
    source = _FakeBaslerCameraSource(FakeNodeMap(default_camera_nodes(Gain=FakeGenicamNode(9.5, min=0.0, max=24.0))))
    window.start_camera(source)

    window.stop_camera()

    data = camera_settings_store.load_last_used(tmp_path)
    assert data is not None
    assert data.values["Gain"] == 9.5


def test_start_camera_restores_previously_saved_settings(qtbot, tmp_path):
    window = MainWindow(camera_settings_dir=tmp_path)
    qtbot.addWidget(window)
    first = _FakeBaslerCameraSource(
        FakeNodeMap(default_camera_nodes(Gain=FakeGenicamNode(12.0, min=0.0, max=24.0)))
    )
    window.start_camera(first)
    window.stop_camera()

    second_node_map = FakeNodeMap(default_camera_nodes())
    second = _FakeBaslerCameraSource(second_node_map)
    window.start_camera(second)

    assert second_node_map.get_node("Gain").get_value() == 12.0
    assert "önceki ayarlar" in window.camera_settings_panel._status_label.text()


def test_start_camera_without_prior_settings_shows_plain_status(qtbot, tmp_path):
    window = MainWindow(camera_settings_dir=tmp_path)
    qtbot.addWidget(window)
    source = _FakeBaslerCameraSource(FakeNodeMap(default_camera_nodes()))

    window.start_camera(source)

    assert "önceki ayarlar" not in window.camera_settings_panel._status_label.text()


def test_camera_tick_drops_frame_while_worker_busy_but_still_updates_last_frame(qtbot):
    """Gerçek kullanıcı raporu: "şekil bul özelliği çok kasıyor" -- bir önceki tick'in ağır
    hesaplaması (burada yavaş bir sahte operatörle simüle edilir) hâlâ arka planda sürüyorken
    yeni bir tick gelirse, bu karenin İŞLENMESİ atlanır (yeni bir worker dispatch edilmez,
    `_live_tick_generation` İLERLEMEZ) -- ama `_last_camera_frame` yine de güncellenir
    (Fotoğraf Çek gibi özellikler her zaman en taze kareyi görsün diye)."""
    import time

    from imgflow.core.types import PortSpec, PortType
    from imgflow.operators import registry as op_registry

    class _SlowOp:
        id = "test.slow_live_tick"
        inputs = [PortSpec("image", PortType.IMAGE)]
        outputs = [PortSpec("image", PortType.IMAGE)]
        params = []

        def run(self, inputs, params):
            time.sleep(0.3)
            return {"image": inputs["image"]}

    op_registry.register(_SlowOp)
    try:
        window = MainWindow()
        qtbot.addWidget(window)
        window.add_operator("io.image_source")
        slow_id = window.add_operator("test.slow_live_tick")
        window._on_step_selected(slow_id)

        frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)
        window.start_camera(_FakeCameraSource([frame1, frame2]))

        window._on_camera_tick()  # worker dispatch edilir, yavaş operatör YÜZÜNDEN hâlâ çalışır
        assert window._live_worker_busy is True
        generation_after_first_dispatch = window._live_tick_generation

        window._on_camera_tick()  # meşgul -> bu tick'in işlenmesi ATLANMALI
        assert window._live_tick_generation == generation_after_first_dispatch
        assert window._last_camera_frame is frame2  # kare yine de güncellendi

        qtbot.waitUntil(lambda: not window._live_worker_busy, timeout=2000)
    finally:
        op_registry.unregister("test.slow_live_tick")


def test_live_tick_result_with_stale_generation_is_discarded_current_is_applied(qtbot):
    """`_on_live_tick_result`, dispatch anındaki `generation` o anki `_live_tick_generation`'la
    UYUŞMUYORSA (seçili adım/görünüm modu bu arada değişti, bkz. `_set_selected_node`/
    `_on_view_mode_changed`) sonucu sessizce atmalı -- geç gelen bir sonucun ekrana yanlışlıkla
    yansımasını önler."""
    window = MainWindow()
    qtbot.addWidget(window)
    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)

    def _make_result(status_text: str) -> "main_window_module.PreviewFrameResult":
        return main_window_module.PreviewFrameResult(
            ok=True,
            error=None,
            display_image=np.full((5, 5, 3), 111, dtype=np.uint8),
            hover_measurements=None,
            measurements=[],
            status_text=status_text,
            is_roi_step=False,
            manual_roi_active=False,
            roi_shape="RECT",
            roi_x=0,
            roi_y=0,
            roi_w=100,
            roi_h=100,
            roi_cx=100,
            roi_cy=100,
            roi_r=50,
            manual_rois=None,
            step_durations=None,
        )

    stale_generation = window._live_tick_generation
    window._set_selected_node(src_id)  # simüle: dispatch SONRASI seçim değişti (generation ilerler)

    window._on_live_tick_result(stale_generation, _make_result("STALE"))
    assert window.status_label.text() != "STALE"

    window._on_live_tick_result(window._live_tick_generation, _make_result("FRESH"))
    assert window.status_label.text() == "FRESH"


def test_start_camera_injects_frame_and_creates_source_node(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)

    window.start_camera(_FakeCameraSource([frame]))
    _run_camera_tick_and_wait(window, qtbot)

    src_nodes = [n for n in window.graph.nodes.values() if n.op_id == "io.image_source"]
    assert len(src_nodes) == 1
    assert window.image_view._pixmap is not None


def test_camera_tick_with_no_frame_does_not_crash(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.start_camera(_FakeCameraSource([]))
    window._on_camera_tick()  # boş kare listesi -> read() None döner, sessizce yok sayılmalı

    assert window._camera_source is not None


def test_stop_camera_releases_source_and_stops_timer(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    source = _FakeCameraSource([])
    window.start_camera(source)

    window.stop_camera()

    assert source.released is True
    assert window._camera_source is None
    assert not window._camera_timer.isActive()


def test_starting_new_camera_stops_previous_one(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    first = _FakeCameraSource([])
    second = _FakeCameraSource([])

    window.start_camera(first)
    window.start_camera(second)

    assert first.released is True
    assert window._camera_source is second


def test_camera_feed_updates_downstream_selected_step(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    frame1 = np.zeros((10, 10, 3), dtype=np.uint8)
    frame2 = np.full((10, 10, 3), 255, dtype=np.uint8)

    window.start_camera(_FakeCameraSource([frame1]))
    window._on_camera_tick()
    th_id = window.add_operator("segment.threshold")
    window._on_step_selected(th_id)

    window._camera_source._frames.append(frame2)
    window._on_camera_tick()

    result = window.engine.evaluate(th_id)
    assert result.ok
    assert result.outputs["image"].max() == 255  # ikinci (beyaz) kareden hesaplanmış


def test_selecting_roi_step_enables_canvas_and_shows_full_frame(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})

    roi_id = window.add_operator("roi.region")
    window._on_step_selected(roi_id)
    window.param_form.params_changed.emit({"enabled": True, "x": 5, "y": 5, "w": 5, "h": 5})

    assert window.image_view._editing_enabled is True
    # enabled=True olsa da önizleme kırpılmamış tam kareyi (20x20) göstermeli, ROI çizilebilsin diye
    assert window.image_view._pixmap.size().width() > 0
    result = window.engine.evaluate(roi_id)
    assert result.ok
    assert result.outputs["image"].shape == (5, 5, 3)  # gerçek çıktı (kırpılmış) hâlâ doğru


def test_selecting_non_roi_step_disables_canvas_editing(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})

    assert window.image_view._editing_enabled is False


def test_dragging_roi_canvas_updates_node_params_and_form(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})
    roi_id = window.add_operator("roi.region")
    window._on_step_selected(roi_id)

    window.image_view.roi_changed.emit(2, 3, 8, 9)

    assert window.graph.nodes[roi_id].params["enabled"] is True
    assert (window.graph.nodes[roi_id].params["x"], window.graph.nodes[roi_id].params["y"]) == (2, 3)
    assert (window.graph.nodes[roi_id].params["w"], window.graph.nodes[roi_id].params["h"]) == (8, 9)
    assert window.param_form.widget_for("x").value() == 2


def test_add_operator_appends_and_selects_step(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    node_id = window.add_operator("io.image_source")

    assert node_id in window.graph.nodes
    assert window.selected_node_id() == node_id


def test_selecting_step_shows_visible_description(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    node_id = window.add_operator("segment.threshold")
    window._on_step_selected(node_id)

    assert window.description_label.text() != ""
    assert "eşik" in window.description_label.text().lower() or "threshold" in window.description_label.text().lower()


def test_checking_operator_in_library_chains_pipeline_automatically(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.operator_library.operator_checked.emit("io.image_source")
    window.operator_library.operator_checked.emit("color.grayscale")

    assert len(window.pipeline.order) == 2
    assert len(window.graph.edges) == 1
    edge = window.graph.edges[0]
    assert edge.src[1] == "image"
    assert edge.dst[1] == "image"


def test_unchecking_operator_removes_it_and_syncs_checkbox(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("morphology.erode")

    window.operator_library.operator_unchecked.emit("morphology.erode")

    assert node_id not in window.graph.nodes
    item = window.operator_library._items_by_op_id["morphology.erode"]
    from PySide6.QtCore import Qt

    assert item.checkState(0) == Qt.CheckState.Unchecked


def test_undo_redo_add_operator(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._undo_action.isEnabled() is False

    node_id = window.add_operator("segment.threshold")
    assert node_id in window.graph.nodes
    assert window._undo_action.isEnabled() is True

    window._on_undo()
    assert node_id not in window.graph.nodes
    assert window._redo_action.isEnabled() is True

    window._on_redo()
    assert node_id in window.graph.nodes


def test_undo_redo_remove_operator(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("segment.threshold")

    window.remove_operator(node_id)
    assert node_id not in window.graph.nodes

    window._on_undo()
    assert node_id in window.graph.nodes
    assert window.graph.nodes[node_id].op_id == "segment.threshold"


def test_undo_param_change_restores_previous_value(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("segment.threshold")
    window._on_step_selected(node_id)
    original_params = dict(window.graph.nodes[node_id].params)
    stack_depth_after_add = len(window._undo_stack)

    window.param_form.params_changed.emit({"value": 200, "max_value": 255, "mode": "BINARY"})
    assert window.graph.nodes[node_id].params["value"] == 200
    # Debounce henüz tetiklenmedi (bkz. `_PARAM_DEBOUNCE_MS`) — bekleyen snapshot henüz
    # yığına PUSH edilmedi, ama Geri Al aksiyonu yine de bunu flush edip kullanmalı.
    assert len(window._undo_stack) == stack_depth_after_add

    window._on_undo()

    assert window.graph.nodes[node_id].params["value"] == original_params["value"]


def test_saving_flat_field_reference_autofills_empty_reference_name(qtbot, monkeypatch, tmp_path):
    # Gerçek kullanıcı raporu: Araçlar > Aydınlatma Referansı Kaydet... ile bir referans
    # kaydetmek, seçili `correction.flat_field` adımının `reference_name` parametresini
    # OTOMATİK seçmiyordu -- kullanıcı referansı kaydettiğini/yüklediğini düşünüp filtreyi
    # uyguluyor ama alan hâlâ boş kaldığı için "'reference_name' parametresi boş olamaz"
    # hatası alıyordu.
    monkeypatch.setattr(flatfield_store, "FLATFIELD_DIR", tmp_path / "flatfield")
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("correction.flat_field")
    window._on_step_selected(node_id)
    assert window.graph.nodes[node_id].params.get("reference_name") == ""

    window._on_flat_field_references_changed("hat1_bos_bant")

    assert window.graph.nodes[node_id].params["reference_name"] == "hat1_bos_bant"


def test_saving_flat_field_reference_does_not_override_existing_selection(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(flatfield_store, "FLATFIELD_DIR", tmp_path / "flatfield")
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("correction.flat_field")
    window._on_step_selected(node_id)
    window.graph.nodes[node_id].params["reference_name"] = "onceki_referans"

    window._on_flat_field_references_changed("yeni_referans")

    assert window.graph.nodes[node_id].params["reference_name"] == "onceki_referans"


def test_deleting_flat_field_reference_does_not_set_reference_name(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(flatfield_store, "FLATFIELD_DIR", tmp_path / "flatfield")
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("correction.flat_field")
    window._on_step_selected(node_id)

    window._on_flat_field_references_changed("")  # silme sinyali boş isimle gelir

    assert window.graph.nodes[node_id].params.get("reference_name") == ""


def test_selecting_flat_field_step_autofills_reference_name_from_only_saved_reference(
    qtbot, monkeypatch, tmp_path
):
    # Bu, `ParamForm`'daki asıl kök nedenin regresyon testidir: Qt bir QComboBox'a
    # `addItems()` ile öğe eklendiğinde -- kutu boşken -- otomatik olarak ilk öğeyi seçili
    # gösterir; bu sinyal bağlantısından ÖNCE olduğu için düğümün gerçek `params`'ı hiç
    # güncellenmiyordu. Kullanıcı "yan tarafta deneme1 yazmasına rağmen" hâlâ "'reference_name'
    # parametresi boş olamaz" hatası aldığını bildirdi -- tam olarak bu senaryo.
    monkeypatch.setattr(flatfield_store, "FLATFIELD_DIR", tmp_path / "flatfield")
    flatfield_store.save_reference("deneme1", np.full((10, 10), 180, dtype=np.uint8))
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("correction.flat_field")

    window._on_step_selected(node_id)

    assert window.graph.nodes[node_id].params["reference_name"] == "deneme1"


def test_duplicate_step_creates_independent_copy_with_same_params(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("segment.threshold")
    window._on_step_selected(node_id)
    window.param_form.params_changed.emit({"value": 77, "max_value": 255, "mode": "BINARY"})
    window._flush_pending_params()

    window.steps_panel.select_node(node_id)
    window._on_duplicate_step_clicked()

    threshold_nodes = [n for n in window.graph.nodes.values() if n.op_id == "segment.threshold"]
    assert len(threshold_nodes) == 2
    assert all(n.params["value"] == 77 for n in threshold_nodes)
    duplicate_id = next(nid for nid, n in window.graph.nodes.items() if n.op_id == "segment.threshold" and nid != node_id)
    assert window.pipeline.order.index(duplicate_id) == window.pipeline.order.index(node_id) + 1


def test_removing_one_duplicate_instance_keeps_library_checkbox_checked(qtbot):
    """Regresyon: 'adım kopyala' aynı op_id'den ikinci bir örnek yarattığında, ikisinden
    birini silmek diğeri hâlâ pipeline'dayken operatör kütüphanesi checkbox'ını yanlışlıkla
    KAPATMAMALI (eskiden koşulsuzdu)."""
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("segment.threshold")
    window.steps_panel.select_node(node_id)
    window._on_duplicate_step_clicked()
    duplicate_id = next(nid for nid in window.graph.nodes if nid != node_id)

    window.remove_operator(duplicate_id)

    from PySide6.QtCore import Qt as _Qt

    item = window.operator_library._items_by_op_id["segment.threshold"]
    assert item.checkState(0) == _Qt.CheckState.Checked
    assert node_id in window.graph.nodes


def test_undo_duplicate_step_removes_the_copy(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("segment.threshold")
    window.steps_panel.select_node(node_id)
    window._on_duplicate_step_clicked()
    assert len(window.graph.nodes) == 2

    window._on_undo()

    assert len(window.graph.nodes) == 1
    assert node_id in window.graph.nodes


def test_move_step_up_and_down_changes_pipeline_order(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    first_id = window.add_operator("io.image_source")
    second_id = window.add_operator("color.grayscale")
    assert window.pipeline.order == [first_id, second_id]

    window.steps_panel.select_node(second_id)
    window._on_move_step_up_clicked()
    assert window.pipeline.order == [second_id, first_id]

    window._on_move_step_down_clicked()
    assert window.pipeline.order == [first_id, second_id]


def test_move_step_up_at_top_is_a_no_op(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    first_id = window.add_operator("io.image_source")
    window.add_operator("color.grayscale")
    window.steps_panel.select_node(first_id)
    stack_depth_before = len(window._undo_stack)

    window._on_move_step_up_clicked()

    assert window.pipeline.order[0] == first_id
    assert len(window._undo_stack) == stack_depth_before


def test_undo_move_step_restores_previous_order(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    first_id = window.add_operator("io.image_source")
    second_id = window.add_operator("color.grayscale")
    window.steps_panel.select_node(second_id)
    window._on_move_step_up_clicked()
    assert window.pipeline.order == [second_id, first_id]

    window._on_undo()

    assert window.pipeline.order == [first_id, second_id]


def test_load_recipe_clears_undo_history(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("io.image_source")
    assert window._undo_action.isEnabled() is True

    recipe_path = tmp_path / "recipe.json"
    window.save_recipe_to(str(recipe_path))
    window.load_recipe_from(str(recipe_path))

    assert window._undo_action.isEnabled() is False
    assert window._redo_action.isEnabled() is False


def test_full_pipeline_updates_preview_on_param_change(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window.add_operator("color.grayscale")
    th_id = window.add_operator("segment.threshold")

    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})

    window._on_step_selected(th_id)
    window.param_form.params_changed.emit({"value": 100, "max_value": 255, "mode": "BINARY"})

    assert window.image_view._pixmap is not None
    assert window.status_label.text() == ""
    assert len(window.graph.edges) == 2


def test_shift_measurements_for_roi_offset_shifts_only_position_keys():
    measurements = [
        {
            "bbox_x": 1,
            "bbox_y": 2,
            "bbox_w": 5,
            "bbox_h": 6,
            "centroid_x": 3.5,
            "centroid_y": 4.5,
            "obb_cx": 3.0,
            "obb_cy": 4.0,
            "obb_w": 5.0,
            "obb_h": 6.0,
            "area": 30.0,
            "bbox_mm_x": 0.1,
            "bbox_mm_y": 0.2,
        }
    ]

    shifted = main_window_module._shift_measurements_for_roi_offset(measurements, 10, 20, mm_per_px=0.5)

    m = shifted[0]
    assert (m["bbox_x"], m["bbox_y"]) == (11, 22)
    assert (m["centroid_x"], m["centroid_y"]) == (13.5, 24.5)
    assert (m["obb_cx"], m["obb_cy"]) == (13.0, 24.0)
    # boyut/alan alanları bir kırpma OFSETİNDEN etkilenmemeli -- sadece KONUM alanları kayar.
    assert (m["bbox_w"], m["bbox_h"], m["area"]) == (5, 6, 30.0)
    assert m["bbox_mm_x"] == 0.1 + 10 * 0.5
    assert m["bbox_mm_y"] == 0.2 + 20 * 0.5
    # orijinal liste mutasyona uğramamalı.
    assert measurements[0]["bbox_x"] == 1


def test_shift_measurements_for_roi_offset_ignores_unrelated_keys_like_shape_match_pose():
    # `geom.shape_match`'in "x"/"y" poz alanları `bbox_x`/`centroid_x` DEĞİL -- yanlışlıkla
    # kaydırılmamalı.
    measurements = [{"x": 5.0, "y": 6.0, "angle": 12.0, "score": 0.9}]

    shifted = main_window_module._shift_measurements_for_roi_offset(measurements, 100, 100, mm_per_px=0.0)

    assert shifted == measurements


def test_cumulative_roi_offset_sums_enabled_rect_roi_before_target(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    roi_id = window.add_operator("roi.region")
    props_id = window.add_operator("analysis.region_props")
    window.graph.nodes[roi_id].params.update({"enabled": True, "shape": "RECT", "x": 7, "y": 9})

    assert main_window_module._cumulative_roi_offset(window.graph, window.pipeline.order, props_id) == (7, 9)


def test_cumulative_roi_offset_ignores_disabled_roi(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    roi_id = window.add_operator("roi.region")
    props_id = window.add_operator("analysis.region_props")
    window.graph.nodes[roi_id].params.update({"enabled": False, "shape": "RECT", "x": 7, "y": 9})

    assert main_window_module._cumulative_roi_offset(window.graph, window.pipeline.order, props_id) == (0, 0)


def test_cumulative_roi_offset_ignores_roi_step_after_target(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    props_id = window.add_operator("analysis.region_props")
    roi_id = window.add_operator("roi.region")  # hedeften SONRA eklendi -- etkilememeli
    window.graph.nodes[roi_id].params.update({"enabled": True, "shape": "RECT", "x": 7, "y": 9})

    assert main_window_module._cumulative_roi_offset(window.graph, window.pipeline.order, props_id) == (0, 0)


def test_cumulative_roi_offset_handles_circle_shape(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    roi_id = window.add_operator("roi.region")
    props_id = window.add_operator("analysis.region_props")
    window.graph.nodes[roi_id].params.update({"enabled": True, "shape": "CIRCLE", "cx": 50, "cy": 60, "r": 20})

    assert main_window_module._cumulative_roi_offset(window.graph, window.pipeline.order, props_id) == (30, 40)


def test_region_props_measurement_boxes_shifted_by_roi_offset_in_normal_view(qtbot, tmp_path):
    # Gerçek kullanıcı sorusu: "bölge ölçümü yaparken ROI'de seçebiliyor muyum, kalibrasyon
    # bozuluyor mu?" -- roi.region + segment.connected_components + analysis.region_props
    # zaten checkbox zincirine oturuyor ve mm/px kalibrasyonu bir kırpmadan etkilenmiyor
    # (alan/boy ölçümleri doğru kalıyor), ama "Normal" görünüm modunda (tam çözünürlüklü ham
    # kare) ölçüm kutuları eskiden ROI'nin sol-üst köşesi kadar KAYIYORDU çünkü bbox
    # koordinatları KIRPILMIŞ görüntüye göreydi. Bu, uçtan uca doğru düzeltildiğini doğrular.
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)  # 20x20, beyaz kare [5:15, 5:15]

    src_id = window.add_operator("io.image_source")
    roi_id = window.add_operator("roi.region")
    window.add_operator("segment.connected_components")
    props_id = window.add_operator("analysis.region_props")

    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})

    window._on_step_selected(roi_id)
    window.param_form.params_changed.emit(
        {"enabled": True, "shape": "RECT", "x": 3, "y": 4, "w": 15, "h": 15, "cx": 100, "cy": 100, "r": 50}
    )

    window._view_mode = "normal"
    window._on_step_selected(props_id)

    measurements = window.image_view._measurements
    assert measurements
    # Ham (kırpılmamış) karedeki beyaz kare [5:15,5:15] ROI (x=3,y=4) içinde [2:11,1:11]
    # olarak kalır -- "Normal" moddaki bbox, ROI ofseti EKLENMİŞ tam kare koordinatında
    # olmalı (yani orijinal beyaz karenin gerçek konumuna -- 5,5 civarına -- yakın), KIRPILMIŞ
    # karenin ham (2,1) gibi bir bbox'ı DEĞİL.
    assert measurements[0]["bbox_x"] >= 3
    assert measurements[0]["bbox_y"] >= 4


def test_remove_operator_clears_preview_when_selected_node_removed(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window.param_form.params_changed.emit({"path": str(path)})

    window.remove_operator(src_id)

    assert src_id not in window.graph.nodes
    assert window.image_view.text() in ("Önizleme yok", "")


def test_removing_selected_middle_step_falls_back_to_next_step(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})
    gray_id = window.add_operator("color.grayscale")
    th_id = window.add_operator("segment.threshold")

    window._on_step_selected(gray_id)
    window.remove_operator(gray_id)

    assert window.selected_node_id() == th_id
    assert window.image_view._pixmap is not None
    assert len(window.graph.edges) == 1


def test_removing_selected_last_step_falls_back_to_new_last_step(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})
    gray_id = window.add_operator("color.grayscale")

    window.remove_operator(gray_id)

    assert window.selected_node_id() == src_id
    assert window.image_view._pixmap is not None


def test_invalid_source_path_reports_error_in_status(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(tmp_path / "yok.png")})

    assert window.status_label.text().startswith("Hata:")


def test_load_recipe_writes_persistent_log_entry(qtbot, tmp_path):
    """Reçete yükleme gibi önemli olaylar artık kalıcı log dosyasına da yazılıyor (bkz.
    `io_utils/app_log.py`) — sadece anlık `status_label` metnine değil, konsol kapalıyken de
    saha desteği için kalıcı bir iz kalmalı."""
    from imgflow.io_utils.app_log import get_logger, setup_logging

    log_dir = tmp_path / "logs"
    setup_logging(directory=log_dir)
    try:
        window = MainWindow()
        qtbot.addWidget(window)
        window.add_operator("io.image_source")
        recipe_path = tmp_path / "recipe.json"
        window.save_recipe_to(str(recipe_path))

        window.load_recipe_from(str(recipe_path))

        content = (log_dir / "imgflow.log").read_text(encoding="utf-8")
        assert str(recipe_path) in content
    finally:
        for handler in list(get_logger().handlers):
            get_logger().removeHandler(handler)
            handler.close()


def test_save_and_load_recipe_roundtrip_preserves_pipeline(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    src_id = window.add_operator("io.image_source")
    gray_id = window.add_operator("color.grayscale")
    window._on_step_selected(gray_id)
    window.param_form.params_changed.emit(
        {
            "clahe_enabled": False,
            "threshold_enabled": False,
            "threshold_mode": "OTSU",
            "threshold_value": 127,
            "adaptive_block_size": 11,
            "adaptive_c": 2,
            "invert": True,
        }
    )

    recipe_path = tmp_path / "recipe.json"
    window.save_recipe_to(str(recipe_path))

    reloaded = MainWindow()
    qtbot.addWidget(reloaded)
    reloaded.load_recipe_from(str(recipe_path))

    assert set(reloaded.graph.nodes) == {src_id, gray_id}
    assert reloaded.graph.nodes[gray_id].params["invert"] is True
    assert len(reloaded.graph.edges) == 1
    assert reloaded.engine.graph is reloaded.graph
    from PySide6.QtCore import Qt

    assert reloaded.operator_library._items_by_op_id["color.grayscale"].checkState(0) == Qt.CheckState.Checked


def test_load_recipe_with_unknown_operator_raises(qtbot, tmp_path):
    recipe_path = tmp_path / "bad_recipe.json"
    recipe_path.write_text(
        '{"schema_version": 1, "nodes": '
        '[{"id": "x", "op_id": "no.such.operator", "params": {}, "position": [0, 0]}], "edges": []}',
        encoding="utf-8",
    )

    window = MainWindow()
    qtbot.addWidget(window)

    with pytest.raises(ImgflowError):
        window.load_recipe_from(str(recipe_path))


def test_load_recipe_with_missing_field_shows_friendly_error_instead_of_crashing(qtbot, tmp_path, monkeypatch):
    recipe_path = tmp_path / "malformed_recipe.json"
    recipe_path.write_text('{"schema_version": 2, "nodes": [{"id": "n1"}], "edges": []}', encoding="utf-8")

    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(
        "imgflow.ui.main_window.QFileDialog.getOpenFileName", lambda *a, **k: (str(recipe_path), "")
    )
    calls = []
    monkeypatch.setattr(
        "imgflow.ui.error_dialog.QMessageBox.critical",
        lambda parent, title, text: calls.append((title, text)),
    )

    window._on_load_recipe()  # önceden burada yakalanmayan KeyError('op_id') fırlıyordu

    assert len(calls) == 1
    assert "bozuk" in calls[0][1].lower()


def test_export_current_image_writes_file(qtbot, tmp_path, monkeypatch):
    """Gerçek kullanıcı iş akışı: kameradan/dosyadan alınan görüntüye filtre uygulayıp,
    sonucu Şekil Eşleştirme'de referans olarak kullanmak üzere gerçek bir resim dosyasına
    dışa aktarma — reçete (.json) DEĞİL (bkz. `test_import_recipe_json_shows_specific_
    guidance_not_raw_keyerror` içindeki ilgili bug)."""
    window = MainWindow()
    qtbot.addWidget(window)
    image_path = _sample_image_path(tmp_path)
    window.open_image(str(image_path))

    out_path = tmp_path / "disa_aktarilan.png"
    monkeypatch.setattr(
        "imgflow.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(out_path), "")
    )

    window._on_export_current_image()

    assert out_path.exists()
    assert "dışa aktarıldı" in window.status_label.text()


def test_export_current_image_without_selection_shows_info_not_file_dialog(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    calls = []
    monkeypatch.setattr("imgflow.ui.main_window.QMessageBox.information", lambda *a, **k: calls.append(a))
    save_dialog_calls = []
    monkeypatch.setattr(
        "imgflow.ui.main_window.QFileDialog.getSaveFileName",
        lambda *a, **k: save_dialog_calls.append(1) or ("", ""),
    )

    window._on_export_current_image()

    assert len(calls) == 1
    assert save_dialog_calls == []


def test_capture_gallery_dock_wraps_panel_and_is_closable(qtbot):
    from PySide6.QtWidgets import QDockWidget

    window = MainWindow()
    qtbot.addWidget(window)

    assert window.capture_gallery_dock.widget() is window.capture_gallery_panel
    assert window.capture_gallery_dock.features() & QDockWidget.DockWidgetFeature.DockWidgetClosable


def test_lens_calibration_capture_refreshes_gallery_panel(qtbot, tmp_path, monkeypatch):
    from imgflow.core import capture_store

    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")
    refreshed = []
    monkeypatch.setattr(
        "imgflow.ui.panels.capture_gallery_panel.CaptureGalleryPanel.refresh",
        lambda self: refreshed.append(1),
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window._camera_source = _FakeCameraSource([np.zeros((100, 100, 3), dtype=np.uint8)])

    window._on_open_lens_calibration()
    window._lens_calibration_dialog._on_capture()

    assert refreshed == [1]


def test_height_scale_calibration_capture_refreshes_gallery_panel(qtbot, tmp_path, monkeypatch):
    from imgflow.core import capture_store

    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")
    refreshed = []
    monkeypatch.setattr(
        "imgflow.ui.panels.capture_gallery_panel.CaptureGalleryPanel.refresh",
        lambda self: refreshed.append(1),
    )

    window = MainWindow()
    qtbot.addWidget(window)
    window._camera_source = _FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8)])

    window._on_open_height_scale_calibration()
    window._height_scale_dialog._on_capture_frame()

    assert refreshed == [1]


def test_capture_photo_action_enabled_for_any_camera(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._capture_photo_action.isEnabled() is False

    window.start_camera(_FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8)]))
    assert window._capture_photo_action.isEnabled() is True

    window.stop_camera()
    assert window._capture_photo_action.isEnabled() is False


def test_capture_camera_photo_saves_frame_and_refreshes_gallery(qtbot, tmp_path, monkeypatch):
    from imgflow.core import capture_store

    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")

    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FakeCameraSource([np.full((10, 10, 3), 200, dtype=np.uint8)]))
    window._on_camera_tick()

    window._on_capture_camera_photo()

    records = capture_store.list_captures()
    assert len(records) == 1
    assert records[0].source == "live"
    assert "kaydedildi" in window.status_label.text()


def test_capture_camera_photo_without_camera_shows_inline_status_not_modal(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_capture_camera_photo()

    assert "kamera yok" in window.status_label.text()


def test_capture_photo_quick_button_stays_in_sync_with_menu_action(qtbot):
    """Görüntü panelinin hemen üstündeki hızlı-erişim çubuğu (`_capture_photo_button`) Kamera
    menüsündeki `_capture_photo_action` ile AYNI etkin/devre dışı durumunu paylaşmalı —
    kullanıcı kamerayı sadece butondan açsa/kapatsa bile ikisi de senkron kalmalı."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._capture_photo_button.isEnabled() is False

    window.start_camera(_FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8)]))
    assert window._capture_photo_button.isEnabled() is True
    assert window._capture_photo_button.isEnabled() == window._capture_photo_action.isEnabled()

    window.stop_camera()
    assert window._capture_photo_button.isEnabled() is False
    assert window._capture_photo_button.isEnabled() == window._capture_photo_action.isEnabled()


def test_capture_photo_quick_button_click_saves_frame(qtbot, tmp_path, monkeypatch):
    from imgflow.core import capture_store

    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")

    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FakeCameraSource([np.full((10, 10, 3), 200, dtype=np.uint8)]))
    window._on_camera_tick()

    window._capture_photo_button.click()

    assert len(capture_store.list_captures()) == 1


def test_capture_gallery_open_requested_loads_image_into_pipeline(qtbot, tmp_path):
    """Gerçek kullanıcı isteği: "çektiğim fotoğraflarda daha sonradan uzunluk bulabilmeliyim"
    -- galeriden çift tıklama/'Pipeline'a Yükle' ile yayınlanan `open_requested` sinyali
    `open_image`'e bağlı olmalı (sürükle-bırak yoluyla, `_on_image_file_dropped`, AYNI çağrı)."""
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    window.capture_gallery_panel.open_requested.emit(str(path))

    src_nodes = [n for n in window.graph.nodes.values() if n.op_id == "io.image_source"]
    assert len(src_nodes) == 1
    assert src_nodes[0].params["path"] == str(path)


def test_capture_camera_photo_before_any_tick_shows_inline_status(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8)]))

    window._on_capture_camera_photo()

    assert "kare alınmadı" in window.status_label.text()


def test_capture_filtered_frame_saves_pipeline_output_and_refreshes_gallery(qtbot, tmp_path, monkeypatch):
    """Gerçek kullanıcı isteği: "filtrelediğim fotoğrafı da yakalayıp sağ taraftaki panele
    atmak istiyorum" -- `_on_capture_camera_photo`'nun aksine kamera GEREKMEZ, seçili adımın
    filtrelenmiş çıktısı (`_current_preview_image()`, `_on_export_current_image` ile AYNI
    kaynak) 'Yakalananlar' galerisine 'filtered' kaynağıyla kaydedilir."""
    from imgflow.core import capture_store

    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")

    window = MainWindow()
    qtbot.addWidget(window)
    image_path = _sample_image_path(tmp_path)
    window.open_image(str(image_path))

    window._on_capture_filtered_frame()

    records = capture_store.list_captures()
    assert len(records) == 1
    assert records[0].source == "filtered"
    assert "kaydedildi" in window.status_label.text()


def test_capture_filtered_frame_without_selection_shows_inline_status_not_modal(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_capture_filtered_frame()

    assert "önce bir pipeline adımı seçin" in window.status_label.text()


def test_capture_filtered_button_click_saves_frame(qtbot, tmp_path, monkeypatch):
    from imgflow.core import capture_store

    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")

    window = MainWindow()
    qtbot.addWidget(window)
    image_path = _sample_image_path(tmp_path)
    window.open_image(str(image_path))

    window._capture_filtered_button.click()

    assert len(capture_store.list_captures()) == 1


def test_run_batch_process_writes_csv(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)

    window.add_operator("io.image_source")
    window.add_operator("color.grayscale")
    th_id = window.add_operator("segment.threshold")
    window.add_operator("segment.connected_components")
    props_id = window.add_operator("analysis.region_props")

    window._on_step_selected(th_id)
    window.param_form.params_changed.emit({"value": 100, "max_value": 255, "mode": "BINARY"})

    input_dir = tmp_path / "images"
    input_dir.mkdir()
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    cv2.imwrite(str(input_dir / "a.png"), img)

    output_csv = tmp_path / "out.csv"
    rows = window.run_batch_process(props_id, str(input_dir), str(output_csv))

    assert len(rows) == 1
    assert rows[0]["image"] == "a.png"
    assert output_csv.exists()


def test_on_run_batch_runs_in_background_thread_with_isolated_graph(qtbot, tmp_path, monkeypatch):
    """Menüden tetiklenen 'Toplu İşlem...' artık arka planda (QThread) çalışmalı ve worker'a
    verilen graph, canlı `window.graph`'ın izole bir kopyası olmalı (bkz. `_BatchWorker`
    docstring'i) — batch sürerken UI'daki canlı graph'ın node parametreleri DEĞİŞMEMELİ."""
    window = MainWindow()
    qtbot.addWidget(window)

    window.add_operator("io.image_source")
    window.add_operator("color.grayscale")
    th_id = window.add_operator("segment.threshold")
    window.add_operator("segment.connected_components")
    props_id = window.add_operator("analysis.region_props")
    window._on_step_selected(th_id)
    window.param_form.params_changed.emit({"value": 100, "max_value": 255, "mode": "BINARY"})
    window._on_step_selected(props_id)

    input_dir = tmp_path / "images"
    input_dir.mkdir()
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    cv2.imwrite(str(input_dir / "a.png"), img)
    output_csv = tmp_path / "out.csv"

    monkeypatch.setattr(
        "imgflow.ui.main_window.QFileDialog.getExistingDirectory", lambda *a, **k: str(input_dir)
    )
    monkeypatch.setattr(
        "imgflow.ui.main_window.QFileDialog.getSaveFileName", lambda *a, **k: (str(output_csv), "")
    )
    monkeypatch.setattr("imgflow.ui.main_window.QMessageBox.information", lambda *a, **k: None)

    params_before = copy.deepcopy(window.graph.nodes[props_id].params)

    window._on_run_batch()
    worker = window._batch_worker
    assert worker is not None

    with qtbot.waitSignal(worker.finished_ok, timeout=5000):
        pass

    assert output_csv.exists()
    assert window.graph.nodes[props_id].params == params_before


def test_open_image_creates_source_node_when_missing(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    node_id = window.open_image(str(path))

    assert window.graph.nodes[node_id].op_id == "io.image_source"
    assert window.graph.nodes[node_id].params["path"] == str(path)
    assert window.selected_node_id() == node_id
    assert window.image_view._pixmap is not None


def test_open_image_reuses_existing_source_node(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    first_path = _sample_image_path(tmp_path)

    first_id = window.open_image(str(first_path))

    img2 = np.full((10, 10, 3), 200, dtype=np.uint8)
    second_path = tmp_path / "second.png"
    cv2.imwrite(str(second_path), img2)
    second_id = window.open_image(str(second_path))

    assert first_id == second_id
    assert sum(1 for n in window.graph.nodes.values() if n.op_id == "io.image_source") == 1
    assert window.graph.nodes[first_id].params["path"] == str(second_path)


def test_open_image_stops_active_camera(qtbot, tmp_path):
    """Kamera aktif akarken statik bir görüntü açılırsa kamera durdurulmalı — aksi halde
    `_on_camera_tick` en geç bir sonraki tick'te canlı kareyi cache'e geri yazıp bu görüntünün
    üzerine yazardı (gerçek kullanıcı isteği: yakalanan bir kareyi sürükleyip pipeline'a
    bırakınca 100ms sonra sessizce kaybolmadan kalıcı görünmesi gerekiyor)."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.start_camera(_FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8)] * 5))
    assert window._camera_timer.isActive() is True

    window.open_image(str(_sample_image_path(tmp_path)))

    assert window._camera_timer.isActive() is False
    assert window._camera_source is None


def test_image_file_dropped_opens_image_and_shows_status(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    window._on_image_file_dropped(str(path))

    node_id = window.selected_node_id()
    assert window.graph.nodes[node_id].params["path"] == str(path)
    assert str(path) in window.status_label.text()


def test_enum_gallery_populates_for_step_with_enum_param(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})
    th_id = window.add_operator("segment.threshold")

    window._on_step_selected(th_id)

    op_cls = window.registry.get("segment.threshold")
    mode_param = next(p for p in op_cls.params if p.type is ParamType.ENUM)
    assert window.enum_gallery._row_layout.count() >= len(mode_param.choices)


def test_enum_gallery_clears_when_step_has_no_enum_param(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    node_id = window.add_operator("analysis.region_props")
    window._on_step_selected(node_id)

    assert window.enum_gallery._row_layout.count() == 0


def test_params_changed_updates_graph_synchronously_but_debounces_preview(qtbot, tmp_path):
    """Pahalı operatörlerde (ör. geom.shape_match) slider sürüklenirken her piksel hareketinde
    tam yeniden hesaplama tetiklenip UI'ın kilitlenmesini (kasma) önlemek için önizleme
    yenilemesi debounce edilir — ama node.params/dirty durumu HER zaman senkron güncellenir."""
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)
    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)

    refresh_calls = []
    original_refresh = window._refresh_preview
    window._refresh_preview = lambda: (refresh_calls.append(1), original_refresh())

    for i in range(5):
        window.param_form.params_changed.emit({"path": str(path) if i == 4 else ""})

    # Parametre HEMEN uygulanmış olmalı (bir sonraki emit'i beklemeden de doğru).
    assert window.graph.nodes[src_id].params["path"] == str(path)
    assert refresh_calls == []  # debounce süresi henüz dolmadı, önizleme HENÜZ yenilenmedi

    qtbot.wait(250)

    assert refresh_calls == [1]  # birikmiş 5 değişiklik TEK bir önizleme yenilemesine indirgendi


def test_results_panel_lists_shape_match_instances_by_number(qtbot, tmp_path, monkeypatch):
    from imgflow.core.roi import RoiRect
    from imgflow.core.shape_matching import train_shape_model
    from imgflow.io_utils import shape_model_store

    monkeypatch.setattr(shape_model_store, "SHAPE_MODEL_DIR", tmp_path / "shape_models")

    base_triangle = np.array([[0, -40], [35, 25], [-20, 30]], dtype=np.float64)

    def draw_triangle(image, center, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        pts = (base_triangle @ rot.T) + np.array(center)
        cv2.fillPoly(image, [pts.astype(np.int32)], color=0)

    reference = np.full((200, 200), 255, dtype=np.uint8)
    draw_triangle(reference, (100, 100), 0.0)
    shape_model_store.save_shape_model("ucgen", train_shape_model(reference, RoiRect(50, 50, 100, 100)))

    search = np.full((200, 200), 255, dtype=np.uint8)
    draw_triangle(search, (100, 100), 0.0)
    search_path = tmp_path / "search.png"
    cv2.imwrite(str(search_path), search)

    window = MainWindow()
    qtbot.addWidget(window)
    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(search_path)})
    qtbot.wait(250)

    match_id = window.add_operator("geom.shape_match")
    window._on_step_selected(match_id)
    window.param_form.params_changed.emit(
        {"model_names": "ucgen", "angle_start": -180.0, "angle_extent": 360.0, "min_score": 0.7, "auto_count": True, "num_matches": 1, "greediness": 0.9}
    )
    qtbot.wait(250)

    assert "1 (ucgen):" in window.results_panel.text()
    assert "Toplam: 1" in window.results_panel.text()

    window._on_step_selected(src_id)


def test_refresh_preview_populates_step_durations_table_for_pipeline_steps(qtbot, tmp_path):
    """Gerçek kullanıcı isteği: "her işlemin sonucunun süresi sonuçlar kısmında yazmalı" --
    pipeline'daki HER adımın (sadece seçili adımın değil) en son ne kadar sürdüğü, sırayla ve
    Türkçe etiketle Sonuçlar panelindeki küçük tabloya ulaşmalı."""
    from imgflow.ui.panels.operator_library import label_for

    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)
    window.open_image(str(path))
    gray_id = window.add_operator("color.grayscale")
    window._on_step_selected(gray_id)

    table = window.results_panel._duration_table
    assert table.rowCount() == 2
    assert table.item(0, 0).text() == label_for("io.image_source")
    assert table.item(1, 0).text() == label_for("color.grayscale")
    assert table.item(0, 1).text().endswith(" ms")
    assert table.item(1, 1).text().endswith(" ms")


def test_live_camera_tick_also_populates_step_durations_table(qtbot):
    """Adım süreleri tablosu sadece senkron `_refresh_preview` yolunda değil, `_LiveTickWorker`
    üzerinden gelen canlı kamera sonucunda da dolmalı -- ikisi de AYNI `_build_preview_frame`
    fonksiyonunu paylaştığından ek bir sinyal/thread mekanizması gerekmez (bkz. `PreviewFrameResult
    .step_durations`). Kaynak (`io.image_source`) düğümü canlı akışta HER ZAMAN `inject_result`
    ile doldurulur (gerçek `run()` çağrılmaz, bkz. `core/engine.py::inject_result`), bu yüzden
    tabloda SADECE gerçekten çalıştırılan downstream adım (`color.grayscale`) görünür."""
    from imgflow.ui.panels.operator_library import label_for

    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("io.image_source")
    gray_id = window.add_operator("color.grayscale")
    window._on_step_selected(gray_id)
    window.start_camera(_FakeCameraSource([np.zeros((10, 10, 3), dtype=np.uint8)]))

    _run_camera_tick_and_wait(window, qtbot)

    table = window.results_panel._duration_table
    assert table.rowCount() == 1
    assert table.item(0, 0).text() == label_for("color.grayscale")


def _fx_1000_lens_profile():
    from imgflow.core.lens_calibration import LensProfile

    return LensProfile(
        camera_matrix=np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.zeros((5, 1)),
        image_size=(640, 480),
        rms_error=0.1,
    )


def _fitted_focus_model():
    from imgflow.core.focus_distance import FocusDistanceModel, FocusTeachPoint

    model = FocusDistanceModel()
    model.add_point(FocusTeachPoint(focus_value=10.0, distance_mm=100.0))
    model.add_point(FocusTeachPoint(focus_value=20.0, distance_mm=200.0))
    model.fit()
    return model


def test_maybe_apply_auto_mm_per_px_updates_region_props_automatically(qtbot, monkeypatch):
    """Checkerboard-tabanlı DENEYSEL akış: kullanıcı hiçbir yükseklik/mesafe girmeden, sadece
    lens profili + netlik->mesafe modeli varsa, mm_per_px otomatik hesaplanıp region_props'a
    uygulanmalı (bkz. `core/focus_distance.py` modül docstring'i)."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._active_lens_profile = _fx_1000_lens_profile()
    window._focus_distance_model = _fitted_focus_model()
    monkeypatch.setattr("imgflow.ui.main_window.focus_measure", lambda frame: 15.0)

    window._maybe_apply_auto_mm_per_px(np.zeros((10, 10), dtype=np.uint8))

    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    expected_distance = window._focus_distance_model.predict_distance(15.0)
    assert node.params["mm_per_px"] == pytest.approx(expected_distance / 1000.0)


def test_maybe_apply_auto_mm_per_px_noop_without_focus_model(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._active_lens_profile = _fx_1000_lens_profile()

    window._maybe_apply_auto_mm_per_px(np.zeros((10, 10), dtype=np.uint8))

    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == 0.0  # değiştirilmedi, hiçbir netlik modeli yok


def test_maybe_apply_auto_mm_per_px_skips_negligible_change(qtbot, monkeypatch):
    """Tick-stride kısıtlamasından (bkz. `test_maybe_apply_auto_mm_per_px_throttles_by_tick_stride`)
    BAĞIMSIZ olarak: değer neredeyse hiç değişmediyse (aynı netlik ölçüsü) tekrar
    uygulanmamalı — her çağrıdan önce sayaç sıfırlanarak stride etkisi devre dışı bırakılır."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._active_lens_profile = _fx_1000_lens_profile()
    window._focus_distance_model = _fitted_focus_model()
    monkeypatch.setattr("imgflow.ui.main_window.focus_measure", lambda frame: 15.0)

    calls = []
    original_push = window._push_mm_per_px
    monkeypatch.setattr(
        window,
        "_push_mm_per_px",
        lambda mm_per_px, refresh=True: (calls.append(mm_per_px), original_push(mm_per_px, refresh=refresh))[1],
    )

    window._maybe_apply_auto_mm_per_px(np.zeros((10, 10), dtype=np.uint8))
    window._auto_height_tick_counter = 0  # stride'ı devre dışı bırak, sadece değişim kontrolünü test et
    window._maybe_apply_auto_mm_per_px(np.zeros((10, 10), dtype=np.uint8))

    assert len(calls) == 1


def test_maybe_apply_auto_mm_per_px_throttles_by_tick_stride(qtbot, monkeypatch):
    """Netlik ölçüsü HER seferinde farklı olsa bile (gerçek kamera gürültüsünü simüle eder),
    art arda tick'lerde sadece her `_AUTO_HEIGHT_TICK_STRIDE` tick'te bir uygulanmalı —
    aksi halde her 100ms'de bir pipeline'ı dirty işaretleyip FPS'i düşürür (bkz. gerçek
    kullanıcı raporu: otomatik yükseklik eklendikten sonra kamera akışı yavaşladı)."""
    from imgflow.ui.main_window import _AUTO_HEIGHT_TICK_STRIDE

    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._active_lens_profile = _fx_1000_lens_profile()
    window._focus_distance_model = _fitted_focus_model()

    focus_values = iter([15.0 + i for i in range(_AUTO_HEIGHT_TICK_STRIDE * 2)])
    monkeypatch.setattr("imgflow.ui.main_window.focus_measure", lambda frame: next(focus_values))

    calls = []
    original_push = window._push_mm_per_px
    monkeypatch.setattr(
        window,
        "_push_mm_per_px",
        lambda mm_per_px, refresh=True: (calls.append(mm_per_px), original_push(mm_per_px, refresh=refresh))[1],
    )

    for _ in range(_AUTO_HEIGHT_TICK_STRIDE * 2):
        window._maybe_apply_auto_mm_per_px(np.zeros((10, 10), dtype=np.uint8))

    assert len(calls) == 2  # 2*stride tick'te, stride'da bir uygulanir -> 2 kez


def test_camera_tick_applies_auto_mm_per_px_without_manual_height(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._active_lens_profile = _fx_1000_lens_profile()
    window._focus_distance_model = _fitted_focus_model()

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    window.start_camera(_FakeCameraSource([frame]))
    window._on_camera_tick()

    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert "mm_per_px" in node.params
    assert node.params["mm_per_px"] > 0
    assert window._active_height_mm is None  # hiçbir yükseklik elle/otomatik AYARLANMADI


def test_measurement_tool_sees_automatic_focus_based_calibration(qtbot, monkeypatch):
    """Regresyon: `_on_open_measurement_tool` -> `_active_mm_per_px` eskiden SADECE eski
    `_height_scale_model`/`_active_height_mm` ikilisine bakıyordu; checkerboard-tabanlı yeni
    otomatik akışta (`_focus_distance_model`) bu ikili hiç dolmadığı için Ölçüm Aracı her
    zaman 'kalibrasyon yok' diyordu — gerçek bir kullanıcı raporuyla doğrulandı. Kamera
    tick'i hiç çalışmadan (`_last_camera_frame` henüz None) da bunu doğrular: tick'i taklit
    edip kareyi elle set ediyoruz."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._active_lens_profile = _fx_1000_lens_profile()
    window._focus_distance_model = _fitted_focus_model()
    monkeypatch.setattr("imgflow.ui.main_window.focus_measure", lambda frame: 15.0)
    window._last_camera_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    window._on_open_measurement_tool()

    assert window._measurement_tool_dialog is not None
    assert window._measurement_tool_dialog._mm_per_px is not None
    assert window._measurement_tool_dialog._mm_per_px > 0
    window._measurement_tool_dialog.close()


def test_active_mm_per_px_prefers_automatic_focus_model_over_legacy_height_model(qtbot, monkeypatch):
    from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint

    window = MainWindow()
    qtbot.addWidget(window)
    window._active_lens_profile = _fx_1000_lens_profile()
    window._focus_distance_model = _fitted_focus_model()
    monkeypatch.setattr("imgflow.ui.main_window.focus_measure", lambda frame: 15.0)
    window._last_camera_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    legacy_model = HeightScaleModel()
    legacy_model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    legacy_model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    legacy_model.fit()
    window._height_scale_model = legacy_model
    window._active_height_mm = 75.0

    expected_distance = window._focus_distance_model.predict_distance(15.0)
    assert window._active_mm_per_px() == pytest.approx(expected_distance / 1000.0)


def test_active_mm_per_px_prefers_fixed_reference_distance_over_focus_model(qtbot, monkeypatch):
    """Gerçek kullanımda netlik-tabanlı model genelde ~0 korelasyona sahip çıktı (kameranın
    odak derinliği yetersiz) — sabit `_reference_distance_mm` (bkz. `LensCalibrationDialog`
    'Referans (Bant Seviyesi) Yap') mevcutsa HER ZAMAN o kullanılmalı, netlik ölçümüne hiç
    bakılmamalı."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._active_lens_profile = _fx_1000_lens_profile()
    window._focus_distance_model = _fitted_focus_model()
    window._reference_distance_mm = 830.0
    monkeypatch.setattr(
        "imgflow.ui.main_window.focus_measure",
        lambda frame: (_ for _ in ()).throw(AssertionError("netlik ölçülmemeli, referans öncelikli")),
    )
    window._last_camera_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    assert window._active_mm_per_px() == pytest.approx(830.0 / 1000.0)


def test_load_calibration_profile_applies_reference_distance_immediately(qtbot, monkeypatch):
    """Sabit referans mesafe kamera karesine bağımlı değildir — profil yüklenir yüklenmez
    (ilk kamera tick'ini beklemeden) `region_props`'a uygulanmalı."""
    from imgflow.io_utils import calibration_store

    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")

    monkeypatch.setattr(
        calibration_store,
        "load_profile",
        lambda name, directory=None: calibration_store.CalibrationProfileData(
            lens_profile=_fx_1000_lens_profile(),
            height_model=None,
            focus_model=None,
            reference_distance_mm=830.0,
            created_at=None,
            operator_note=None,
        ),
    )

    result = window._load_calibration_profile("hat_ref")

    assert result is True
    assert window._reference_distance_mm == pytest.approx(830.0)
    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(830.0 / 1000.0)


def _sample_plane_rectification(mm_per_px: float = 0.4):
    from imgflow.core.plane_rectification import compute_plane_rectification

    K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    rvec = np.array([[0.1], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [800.0]])
    return compute_plane_rectification(K, rvec, tvec, (640, 480), mm_per_px=mm_per_px)


def test_compute_auto_mm_per_px_prefers_plane_rectification_over_everything_else(qtbot, monkeypatch):
    """Konum-bağımsız düzlem düzeltmesi (bkz. `core/plane_rectification.py`) mevcutsa, sabit
    referans mesafeden ve netlik-tabanlı deneysel modelden DAHA ÖNCELİKLİDİR — kamera düzleme
    tam dik olmasa bile doğru sonuç veren TEK yöntem bu (gerçek kullanıcı raporu: aynı cisim
    ekranın farklı yerlerinde 6mm/5.5mm ölçülüyordu)."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._active_lens_profile = _fx_1000_lens_profile()
    window._reference_distance_mm = 830.0
    window._plane_rectification = _sample_plane_rectification(mm_per_px=0.4)
    monkeypatch.setattr(
        "imgflow.ui.main_window.focus_measure",
        lambda frame: (_ for _ in ()).throw(AssertionError("netlik ölçülmemeli, rektifikasyon öncelikli")),
    )

    assert window._compute_auto_mm_per_px() == pytest.approx(0.4)


def test_camera_tick_rectifies_frame_before_injecting_into_pipeline(qtbot):
    """Düzlem düzeltmesi mevcutsa ham kare pipeline'a girmeden önce `cv2.warpPerspective` ile
    "kuşbakışı" görünüme çevrilmeli (farklı boyutta bir çıktı) — bu, pipeline'ın (ROI/
    threshold/region_props) artık konum-bağımsız, gerçek ölçekli bir görüntü üzerinde
    çalıştığı anlamına gelir."""
    window = MainWindow()
    qtbot.addWidget(window)
    rectification = _sample_plane_rectification(mm_per_px=0.4)
    window._plane_rectification = rectification

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    window.start_camera(_FakeCameraSource([frame]))
    window._on_camera_tick()

    result = window.engine.evaluate(window._find_or_create_image_source())
    assert result.outputs["image"].shape[:2] == (rectification.output_size[1], rectification.output_size[0])


def test_new_region_props_node_gets_mm_per_px_after_calibration_already_stable(qtbot):
    """Kalibrasyon (`_plane_rectification`) zaten aktifken birkaç kamera tick'i geçip mm/px
    'stabilize' olduktan SONRA pipeline'a yeni bir Bölge Ölçümü adımı eklenirse, bu yeni
    node'un varsayılan `mm_per_px=0.0` değeri sonsuza kadar öyle KALMAMALI. `_maybe_apply_
    auto_mm_per_px`'teki '%0.5'ten az değişince atla' kısayolu SADECE hesaplanan değeri değil,
    node'un GÜNCEL parametresini de dikkate almalı — aksi halde kullanıcı 'kalibrasyon
    profilini yükledim ama hâlâ px yazıyor' diye rapor ediyor (gerçek kullanıcı raporu)."""
    from imgflow.ui.main_window import _AUTO_HEIGHT_TICK_STRIDE

    window = MainWindow()
    qtbot.addWidget(window)
    window._plane_rectification = _sample_plane_rectification(mm_per_px=0.4)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    window.start_camera(_FakeCameraSource([frame] * (_AUTO_HEIGHT_TICK_STRIDE * 4)))
    for _ in range(_AUTO_HEIGHT_TICK_STRIDE):
        window._on_camera_tick()

    window.add_operator("analysis.region_props")
    for _ in range(_AUTO_HEIGHT_TICK_STRIDE):
        window._on_camera_tick()

    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(0.4)


def test_push_mm_per_px_also_updates_geom_shape_match_nodes(qtbot):
    """Gerçek kullanıcı isteği: "geometrik eşlemede scale boyutu ... da yazmalı" (netleşen
    anlamı: bulunan nesnenin gerçek dünya boyutu/öteleme mesafesi mm cinsinden) -- otomatik
    kalibrasyon akışı artık `analysis.region_props` ile AYNI şekilde `geom.shape_match`
    düğümlerinin de `mm_per_px` parametresini doldurur."""
    window = MainWindow()
    qtbot.addWidget(window)
    match_id = window.add_operator("geom.shape_match")

    window._push_mm_per_px(0.4, refresh=False)

    assert window.graph.nodes[match_id].params["mm_per_px"] == pytest.approx(0.4)


def test_push_mm_per_px_does_not_get_silently_reverted_by_unrelated_param_edit(qtbot):
    """Gerçek kullanıcı raporu: "kalibrasyon ayarı kendi kendine kaybolabiliyor". Kök neden:
    `_push_mm_per_px` `node.params`'ı `ParamForm`'un HABERİ OLMADAN doğrudan güncelliyordu --
    node o an seçili/panelde gösteriliyorsa `ParamForm._values` ESKİ `mm_per_px`'te kalıyordu.
    Kullanıcı sonra AYNI node'da başka bir parametreyi değiştirdiğinde (`_on_params_changed`
    formun TÜM `_values`'ini `node.params`'ın üzerine koşulsuz yazdığından, bkz. o metot),
    otomatik hesaplanan kalibrasyon kullanıcı hiç dokunmamış olsa bile SESSİZCE sıfırlanıyordu.
    `_push_mm_per_px`'in artık seçili node için `ParamForm.set_value()` ile formun önbelleğini
    de HEMEN güncellemesi bunu önlemeli."""
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("analysis.region_props")  # add_operator zaten seçer

    window._push_mm_per_px(0.4, refresh=False)
    assert window.graph.nodes[node_id].params["mm_per_px"] == pytest.approx(0.4)
    # Formun kendi önbelleği de senkron olmalı, aksi halde aşağıdaki adım bug'ı YAKALAMAZ.
    assert window.param_form.values()["mm_per_px"] == pytest.approx(0.4)

    # Kullanıcı kalibrasyona hiç dokunmadan, AYNI node'da BAŞKA bir parametreyi değiştiriyor
    # -- ParamForm gerçekte formun TÜM (mm_per_px dahil) `_values`'ini yayınlar.
    edited_values = dict(window.param_form.values())
    edited_values["min_area"] = 42.0
    window._on_params_changed(edited_values)

    assert window.graph.nodes[node_id].params["mm_per_px"] == pytest.approx(0.4)
    assert window.graph.nodes[node_id].params["min_area"] == pytest.approx(42.0)


def test_push_mm_per_px_does_not_touch_form_of_a_different_selected_node(qtbot):
    """`ParamForm.set_value()` sadece güncellenen node PANELDE GÖSTERİLİYORSA çağrılmalı --
    başka bir node seçiliyken (ör. kullanıcı `roi.region` adımını inceliyor) formun o an
    gösterdiği (region_props'a AİT OLMAYAN) alanlara yanlışlıkla yazılmamalı."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    roi_id = window.add_operator("roi.region")  # son eklenen otomatik seçili olur

    window._push_mm_per_px(0.4, refresh=False)

    assert window._selected_node_id == roi_id
    assert "mm_per_px" not in window.param_form.values()


def test_new_shape_match_node_gets_mm_per_px_after_calibration_already_stable(qtbot):
    """`_region_props_needs_mm_per_px`'in genişletilmiş kontrolü sayesinde, kalibrasyon
    stabilize olduktan SONRA eklenen yeni bir `geom.shape_match` düğümü de (region_props'un
    üstteki testle AYNI senaryosu) sonsuza kadar varsayılan `mm_per_px=0.0`'da KALMAMALI."""
    from imgflow.ui.main_window import _AUTO_HEIGHT_TICK_STRIDE

    window = MainWindow()
    qtbot.addWidget(window)
    window._plane_rectification = _sample_plane_rectification(mm_per_px=0.4)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    window.start_camera(_FakeCameraSource([frame] * (_AUTO_HEIGHT_TICK_STRIDE * 4)))
    for _ in range(_AUTO_HEIGHT_TICK_STRIDE):
        window._on_camera_tick()

    match_id = window.add_operator("geom.shape_match")
    for _ in range(_AUTO_HEIGHT_TICK_STRIDE):
        window._on_camera_tick()

    assert window.graph.nodes[match_id].params["mm_per_px"] == pytest.approx(0.4)


def test_undo_does_not_lose_active_calibration(qtbot):
    """Gerçek kullanıcı raporu: "kalibrasyon ayarı kendi kendine kaybolabiliyor" -- İKİNCİ
    kayıp yolu: `mm_per_px` düğümün `params`'ında yaşadığı için, kalibrasyondan ÖNCE alınmış
    bir Geri Al snapshot'ı geri yüklendiğinde alan sessizce eski/boş değerine dönüyordu.
    Kalibrasyon KAYNAĞI (`_plane_rectification`) bu işlemden hiç etkilenmediğinden, geri
    yükleme sonrası aktif kalibrasyon düğüme YENİDEN yazılmalı."""
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("analysis.region_props")  # snapshot: kalibrasyondan ÖNCE

    window._plane_rectification = _sample_plane_rectification(mm_per_px=0.4)
    window._push_mm_per_px(0.4, refresh=False)
    assert window.graph.nodes[node_id].params["mm_per_px"] == pytest.approx(0.4)

    # Kullanıcı kalibrasyonla ilgisi olmayan bir yapısal değişiklik yapıp geri alıyor.
    window.add_operator("filter.gaussian_blur")
    window._on_undo()

    assert window.graph.nodes[node_id].params["mm_per_px"] == pytest.approx(0.4)


def test_undo_still_reverts_a_manually_typed_mm_per_px(qtbot):
    """Otomatik bir kalibrasyon kaynağı YOKKEN, kullanıcının ELLE girdiği `mm_per_px`'in geri
    alınması meşru bir Geri Al'dır -- `_reapply_active_calibration` buna dokunmamalı."""
    window = MainWindow()
    qtbot.addWidget(window)
    node_id = window.add_operator("analysis.region_props")

    values = dict(window.param_form.values())
    values["mm_per_px"] = 0.25
    window._on_params_changed(values)
    window._flush_pending_params()
    assert window.graph.nodes[node_id].params["mm_per_px"] == pytest.approx(0.25)

    window._on_undo()

    assert window.graph.nodes[node_id].params["mm_per_px"] == pytest.approx(0.0)


def test_load_calibration_profile_applies_plane_rectification_mm_per_px_immediately(qtbot, monkeypatch):
    from imgflow.io_utils import calibration_store

    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    rectification = _sample_plane_rectification(mm_per_px=0.4)

    monkeypatch.setattr(
        calibration_store,
        "load_profile",
        lambda name, directory=None: calibration_store.CalibrationProfileData(
            lens_profile=_fx_1000_lens_profile(),
            height_model=None,
            focus_model=None,
            reference_distance_mm=830.0,
            plane_rectification=rectification,
            created_at=None,
            operator_note=None,
        ),
    )

    result = window._load_calibration_profile("hat_rect")

    assert result is True
    assert window._plane_rectification is rectification
    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(0.4)


def test_lens_calibrated_applies_reference_distance_and_plane_rectification_immediately(qtbot):
    """`LensCalibrationDialog.calibrated` sinyali sadece `LensProfile` taşır, ama kullanıcı aynı
    'Lens Kalibrasyonu...' oturumunda bir referans kare işaretleyip kalibre ettiyse dialog
    içinde `reference_distance_mm`/`plane_rectification` de hesaplanmış olur. Bunlar
    `_on_lens_calibrated`'da uygulanmazsa kullanıcı ayrıca 'Kalibrasyon Profili Yükle...'
    yapmadan mm/px hiç canlı oturuma yansımaz (gerçek kullanıcı raporu: "lens kalibrasyonu
    yükleyince de mm/cm gözükmüyor")."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._camera_source = _FakeCameraSource([np.zeros((100, 100, 3), dtype=np.uint8)])

    window._on_open_lens_calibration()
    dialog = window._lens_calibration_dialog
    profile = _fx_1000_lens_profile()
    rectification = _sample_plane_rectification(mm_per_px=0.4)
    dialog._profile = profile
    dialog._reference_distance_mm = 830.0
    dialog._plane_rectification = rectification

    window._on_lens_calibrated(profile)

    assert window._active_lens_profile is profile
    assert window._reference_distance_mm == pytest.approx(830.0)
    assert window._plane_rectification is rectification
    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(0.4)


def test_adjust_height_delta_rescales_plane_rectification_homography_and_mm_per_px(qtbot, monkeypatch):
    """Gerçek kullanıcı raporu: "bant yüksekliği değişti diyip ölçüm alıyorum fakat çok
    yanlış ölçüyor". Kök neden: bu metot eskiden SADECE `mm_per_px` alanını güncelleyip
    `homography`/`output_size`'ı DOKUNULMADAN bırakıyordu -- ama `rectify()`'ın kullandığı
    `homography` matrisi `compute_plane_rectification`'da `1/mm_per_px` ölçeğini zaten İÇİNE
    gömüyor (bkz. `world_to_output`), yani `_on_camera_tick` HER karede hâlâ ESKİ ölçekle
    rektifiye ediyordu ama ölçüm operatörleri YENİ `mm_per_px`'i kullanıyordu -- tam olarak
    (eski/yeni oranı) kadar sistematik ölçüm hatası. Düzeltme: `homography`/`output_size`
    de AYNI `(eski_mm_per_px/yeni_mm_per_px)` oranıyla rescale edilmeli."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._active_lens_profile = _fx_1000_lens_profile()
    original = _sample_plane_rectification(mm_per_px=0.4)
    window._plane_rectification = original
    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getDouble", lambda *a, **k: (50.0, True))

    window._on_adjust_height_delta()

    # eski mesafe = 0.4*1000 = 400mm; +50mm yaklaştı -> yeni mesafe 350mm -> mm_per_px 0.35
    assert window._plane_rectification is not original
    assert window._plane_rectification.mm_per_px == pytest.approx(0.35)
    scale_ratio = 0.4 / 0.35
    expected_homography = np.diag([scale_ratio, scale_ratio, 1.0]) @ original.homography
    np.testing.assert_allclose(window._plane_rectification.homography, expected_homography)
    expected_output_size = (
        max(1, round(original.output_size[0] * scale_ratio)),
        max(1, round(original.output_size[1] * scale_ratio)),
    )
    assert window._plane_rectification.output_size == expected_output_size
    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(0.35)


def _apply_homography_for_test(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    transformed = (h @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def test_adjust_height_delta_keeps_world_point_to_mm_consistent(qtbot, monkeypatch):
    """Doğrudan tutarlılık kanıtı: aynı ham (undistort edilmiş) piksel, ayarlamadan ÖNCE ve
    SONRA rektifiye edilip kendi `mm_per_px`'i ile çarpıldığında AYNI gerçek-dünya mm
    konumunu vermeli -- bu, "ölçüm çok yanlış çıkıyor" raporunun kök nedenini (rescale
    edilmeyen homography ile YENİ mm_per_px'in tutarsız kalması) doğrudan test ediyor."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._active_lens_profile = _fx_1000_lens_profile()
    original = _sample_plane_rectification(mm_per_px=0.4)
    window._plane_rectification = original
    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getDouble", lambda *a, **k: (50.0, True))

    raw_point = np.array([[300.0, 200.0]])
    before_rectified_px = _apply_homography_for_test(original.homography, raw_point)[0]
    before_world_mm = before_rectified_px * original.mm_per_px

    window._on_adjust_height_delta()

    after_rectified_px = _apply_homography_for_test(window._plane_rectification.homography, raw_point)[0]
    after_world_mm = after_rectified_px * window._plane_rectification.mm_per_px

    np.testing.assert_allclose(after_world_mm, before_world_mm, rtol=1e-9)


def test_adjust_height_delta_updates_reference_distance(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    window._active_lens_profile = _fx_1000_lens_profile()
    window._reference_distance_mm = 830.0
    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getDouble", lambda *a, **k: (30.0, True))

    window._on_adjust_height_delta()

    assert window._reference_distance_mm == pytest.approx(800.0)
    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(0.8)


def test_adjust_height_delta_rejects_distance_going_non_positive(qtbot, monkeypatch):
    """Geçersiz (fiziksel olarak imkansız) bir delta girilirse mevcut kalibrasyon DEĞİŞMEMELİ
    — sadece durum etiketinde hata gösterilir (bkz. CLAUDE.md: tekrarlayan bir döngüden değil
    ama yine de burada modal DEĞİL, satır içi hata tercih edilir)."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._active_lens_profile = _fx_1000_lens_profile()
    window._reference_distance_mm = 830.0
    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getDouble", lambda *a, **k: (900.0, True))

    window._on_adjust_height_delta()

    assert window._reference_distance_mm == pytest.approx(830.0)
    assert "Geçersiz" in window.status_label.text()


def test_adjust_height_delta_updates_legacy_height_scale_model(qtbot, monkeypatch):
    from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint

    window = MainWindow()
    qtbot.addWidget(window)
    window.add_operator("analysis.region_props")
    model = HeightScaleModel()
    model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    window._height_scale_model = model
    window._active_height_mm = 75.0
    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getDouble", lambda *a, **k: (25.0, True))

    window._on_adjust_height_delta()

    assert window._active_height_mm == pytest.approx(100.0)
    expected_mm_per_px = 1.0 / model.predict_scale(100.0)
    node = next(n for n in window.graph.nodes.values() if n.op_id == "analysis.region_props")
    assert node.params["mm_per_px"] == pytest.approx(expected_mm_per_px)


def test_adjust_height_delta_rejects_height_scale_model_out_of_range(qtbot, monkeypatch):
    from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint

    window = MainWindow()
    qtbot.addWidget(window)
    model = HeightScaleModel()
    model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    window._height_scale_model = model
    window._active_height_mm = 75.0
    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getDouble", lambda *a, **k: (200.0, True))

    window._on_adjust_height_delta()

    assert window._active_height_mm == pytest.approx(75.0)
    assert "Geçersiz" in window.status_label.text()


def test_adjust_height_delta_shows_message_when_no_calibration_active(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr("imgflow.ui.main_window.QInputDialog.getDouble", lambda *a, **k: (10.0, True))

    window._on_adjust_height_delta()

    assert "kalibrasyon" in window.status_label.text().lower()


def test_view_mode_combo_updates_view_mode(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    index = window.view_mode_combo.findData("normal")
    window.view_mode_combo.setCurrentIndex(index)

    assert window._view_mode == "normal"


def test_get_normal_base_image_uses_last_camera_frame_when_camera_active(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    frame = np.full((10, 10, 3), 200, dtype=np.uint8)
    window._camera_source = _FakeCameraSource([])
    window._last_camera_frame = frame

    result = _get_normal_base_image_for(window)

    assert result is frame


def test_get_normal_base_image_returns_none_without_camera_or_source(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert _get_normal_base_image_for(window) is None


def test_get_normal_base_image_returns_static_source_image(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)
    window.open_image(str(path))

    result = _get_normal_base_image_for(window)

    assert result is not None
    assert result.shape[:2] == (20, 20)


def test_compose_display_image_filtered_mode_returns_filtered_image_unchanged(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "filtered"
    filtered = np.zeros((5, 5, 3), dtype=np.uint8)

    result, hover_measurements = _compose_display_image_for(window, filtered, None, 0.0)

    assert result is filtered
    assert hover_measurements is None


def test_compose_display_image_filtered_mode_passes_through_measurements_for_hover(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "filtered"
    filtered = np.zeros((5, 5, 3), dtype=np.uint8)
    measurements = [{"bbox_x": 1, "bbox_y": 1, "bbox_w": 2, "bbox_h": 2}]

    _, hover_measurements = _compose_display_image_for(window, filtered, measurements, 0.0)

    # küçük görüntü _MAX_PREVIEW_DIM altında -> ölçek=1.0, koordinatlar DEĞİŞMEDEN geçer
    assert hover_measurements == measurements


def test_compose_display_image_normal_mode_draws_overlay_on_raw_frame_without_mutating_it(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "normal"
    raw = np.zeros((20, 20, 3), dtype=np.uint8)
    window._camera_source = _FakeCameraSource([])
    window._last_camera_frame = raw
    measurements = [
        {
            "bbox_x": 2,
            "bbox_y": 2,
            "bbox_w": 6,
            "bbox_h": 6,
            "obb_cx": 5.0,
            "obb_cy": 5.0,
            "obb_w": 6.0,
            "obb_h": 6.0,
            "obb_angle": 0.0,
        }
    ]

    result, hover_measurements = _compose_display_image_for(window, 
        np.zeros((20, 20), dtype=np.uint8), measurements, 0.0
    )

    assert result is not None
    assert result.shape == (20, 20, 3)
    assert (result == [0, 255, 0]).all(axis=-1).any()  # ölçüm kutusu yeşil çizilmiş
    assert not raw.any()  # taban kare (ham kamera karesi) yerinde değiştirilmemiş
    assert hover_measurements == measurements  # küçük görüntü -> ölçek=1.0, değişmeden geçer


def test_compose_display_image_normal_mode_without_measurements_returns_plain_raw_frame(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "normal"
    raw = np.full((10, 10, 3), 128, dtype=np.uint8)
    window._camera_source = _FakeCameraSource([])
    window._last_camera_frame = raw

    result, hover_measurements = _compose_display_image_for(window, np.zeros((10, 10), dtype=np.uint8), None, 0.0)

    assert (result == 128).all()
    assert hover_measurements is None


def test_compose_display_image_both_mode_concatenates_horizontally(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "both"
    raw = np.zeros((10, 8, 3), dtype=np.uint8)
    window._camera_source = _FakeCameraSource([])
    window._last_camera_frame = raw
    filtered = np.zeros((10, 8, 3), dtype=np.uint8)

    result, hover_measurements = _compose_display_image_for(window, filtered, None, 0.0)

    assert result.shape[0] == 10
    assert result.shape[1] == 8 + 4 + 8  # sol + ayırıcı (4px) + sağ
    assert hover_measurements is None  # "both" modunda üzerine gelme desteklenmiyor


def test_compose_display_image_both_mode_falls_back_to_single_side_when_other_missing(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "both"
    # kamera yok, io.image_source node'u da yok -> normal taraf None olmalı
    filtered = np.zeros((10, 8, 3), dtype=np.uint8)

    result, hover_measurements = _compose_display_image_for(window, filtered, None, 0.0)

    assert result is filtered
    assert hover_measurements is None


def test_cap_preview_size_leaves_small_images_unchanged():
    from imgflow.ui.main_window import _cap_preview_size

    small = np.zeros((100, 200, 3), dtype=np.uint8)

    result = _cap_preview_size(small)

    assert result is small


def test_cap_preview_size_downscales_large_images_preserving_aspect_ratio():
    from imgflow.ui.main_window import _cap_preview_size

    large = np.zeros((1200, 2400, 3), dtype=np.uint8)  # 2:1 en-boy oranı, uzun kenar 2400

    result = _cap_preview_size(large, max_dim=1600)

    assert max(result.shape[:2]) == 1600
    assert result.shape[1] == 1600  # genişlik uzun kenar
    assert result.shape[0] == 800  # 1200 * (1600/2400) = 800, oran korunmuş


def test_compose_display_image_normal_mode_caps_large_frame_for_display(qtbot):
    """`_get_normal_base_image` çözünürlüğü `_MAX_PREVIEW_DIM`'i aşan bir kare döndürürse,
    overlay yine TAM çözünürlükte (koordinatlar bozulmasın diye) çizilir ama son görüntü
    ekrana aktarılmadan önce küçültülür -- 10Hz kamera tick'inde gereksiz maliyeti önler."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "normal"
    large_raw = np.zeros((1200, 2400, 3), dtype=np.uint8)
    window._camera_source = _FakeCameraSource([])
    window._last_camera_frame = large_raw

    result, _ = _compose_display_image_for(window, np.zeros((1200, 2400), dtype=np.uint8), None, 0.0)

    assert max(result.shape[:2]) == 1600


def test_hover_info_label_shows_default_text_without_a_hovered_measurement(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window._on_hover_measurement_changed(None)

    assert "üzerine getirin" in window.hover_info_label.text()


def test_hover_info_label_shows_region_props_style_measurement_in_cm(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    measurement = {
        "label": 3,
        "obb_w": 100.0,
        "obb_h": 50.0,
        "obb_mm_w": 100.0,
        "obb_mm_h": 50.0,
        "obb_angle": 12.5,
        "area": 5000.0,
        "area_mm2": 5000.0,
        "tolerance_ok": False,
    }

    window._on_hover_measurement_changed(measurement)

    text = window.hover_info_label.text()
    assert "Etiket: 3" in text
    assert "10.00 x 5.00 cm" in text  # 100mm/10=10cm, 50mm/10=5cm
    assert "50.00 cm" in text  # alan: 5000mm2/100 = 50cm2
    assert "NG" in text


def test_hover_info_label_shows_shape_match_style_measurement(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    measurement = {"model": "civata", "label": "civata1", "x": 10.0, "y": 20.0, "angle": 45.0, "score": 0.92}

    window._on_hover_measurement_changed(measurement)

    text = window.hover_info_label.text()
    assert "Model: civata" in text
    assert "0.92" in text
    assert "45.0" in text


def test_hover_measurement_changed_signal_updates_left_panel_label(qtbot):
    """`RoiCanvas.hover_measurement_changed` sinyali `MainWindow`'a bağlı -- fare bir nesnenin
    üzerine geldiğinde sol paneldeki etiket otomatik güncellensin diye (bkz. `_build_layout`
    'Nesne Bilgisi' etiketi)."""
    window = MainWindow()
    qtbot.addWidget(window)

    window.image_view.hover_measurement_changed.emit({"label": "x", "obb_w": 1.0, "obb_h": 1.0})

    assert "Etiket: x" in window.hover_info_label.text()


def test_window_re_maximizes_itself_if_restored_to_normal_state(qtbot):
    """Gerçek kullanıcı raporu: "uygulama kendi kendine tam ekrandan çıkıyor ve ekran
    kayıyor, bazı panellere erişemiyorum, o yüzden tam ekrandan hiç çıkmasın." Pencere
    durumu HERHANGİ bir yolla (başlık çubuğu çift tıklama, Aero Snap, vb.) düz "geri
    yüklenmiş" (`WindowNoState`) hale dönerse `MainWindow.changeEvent` onu HEMEN tekrar
    büyütmeli -- küçültme (görev çubuğuna atma) ise SERBEST kalmalı."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.showMaximized()
    assert window.isMaximized()

    window.showNormal()

    assert window.isMaximized()


def test_window_can_still_be_minimized(qtbot):
    """`changeEvent`'in yeniden büyütme kuralı SADECE düz "geri yüklenmiş" (`WindowNoState`)
    duruma uygulanmalı -- küçültülmüş (görev çubuğu) durumu zorla geri getirilmemeli. (Qt'de
    "küçültülmüş" durum genelde "büyütülmüşken küçültüldü" bilgisini de birlikte taşır --
    `isMaximized()` bu yüzden burada kontrol EDİLMİYOR, sadece gerçek küçültme davranışının
    engellenmediği doğrulanıyor.)"""
    window = MainWindow()
    qtbot.addWidget(window)
    window.showMaximized()

    window.showMinimized()

    assert window.isMinimized()


def test_closing_window_disconnects_destroyed_handlers_of_all_dialogs_ever_opened(qtbot):
    """Uzun süredir var olan test-paketi kararsızlığının ("AttributeError: Slot 'MainWindow::'
    not found", rastgele bir testte patlıyordu) kök nedeni: dialogların `destroyed` sinyali
    `self`'i (MainWindow) yakalayan bir geri çağrıya bağlıydı. `deleteLater()` ile yok
    edilmeyi BEKLEYEN bir dialog varsa (Ölçüm Aracı her açılışta yeniden kurulur, eskisi
    silinmeyi bekler) ana pencere ÖNCE yok edilebiliyor; sinyal daha sonra ölü pencerede
    slot arayıp patlıyordu. Artık `_connect_destroyed` her dialog'u kaydediyor ve
    `closeEvent` hepsinin bağlantısını kesiyor -- SADECE o an izlenen tekil referanslara
    bakmak yetmiyordu (silinmeyi bekleyen eski örnek hiçbir alanda tutulmuyor)."""
    import shiboken6

    window = MainWindow()
    qtbot.addWidget(window)

    window._on_open_measurement_tool()
    window._on_open_measurement_tool()  # eskisi deleteLater() ile silinmeyi bekler
    window._on_open_help()

    assert len(window._tracked_dialogs) >= 3
    help_dialog = window._help_dialog

    window.close()

    assert window._tracked_dialogs == []

    # Bağlantı gerçekten kesildi mi? Kapanıştan SONRA dialog yok edilirse eski geri çağrı
    # (`setattr(self, "_help_dialog", None)`) ARTIK ÇALIŞMAMALI -- bu, ölü pencereye ulaşan
    # sinyalin de artık var olmadığının doğrudan kanıtı.
    window._help_dialog = "NÖBETÇİ"
    assert shiboken6.isValid(help_dialog)
    help_dialog.deleteLater()
    qtbot.wait(50)

    assert window._help_dialog == "NÖBETÇİ"


def test_in_context_view_shows_filtered_roi_inside_the_full_frame(qtbot, tmp_path):
    """Gerçek kullanıcı isteği: "ROI uygulayınca başka filtreye geçince sadece ROI alanı
    görünüyor, ben ROI dışında kalan alanı da görmek istiyorum -- işlemli ve işlemsiz
    bölgeyi görmüş olurum böylece"."""
    from imgflow.ui.main_window import _paste_filtered_into_frame

    window = MainWindow()
    qtbot.addWidget(window)
    image = np.full((200, 300, 3), 200, dtype=np.uint8)
    cv2.circle(image, (150, 100), 40, (30, 30, 30), -1)
    path = tmp_path / "scene.png"
    cv2.imwrite(str(path), image)

    src_id = window.add_operator("io.image_source")
    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})
    roi_id = window.add_operator("roi.region")
    window._on_step_selected(roi_id)
    window.param_form.params_changed.emit(
        {"enabled": True, "shape": "RECT", "x": 100, "y": 50, "w": 100, "h": 100}
    )
    window.add_operator("segment.threshold")

    modes = [window.view_mode_combo.itemData(i) for i in range(window.view_mode_combo.count())]
    assert "in_context" in modes

    window.view_mode_combo.setCurrentIndex(modes.index("filtered"))
    window._refresh_preview()
    assert (window.image_view._pixmap.width(), window.image_view._pixmap.height()) == (100, 100)

    window.view_mode_combo.setCurrentIndex(modes.index("in_context"))
    window._refresh_preview()
    # Tam kare boyutunda -- yani ROI dışındaki alan da görünüyor.
    assert (window.image_view._pixmap.width(), window.image_view._pixmap.height()) == (300, 200)

    # Yapıştırmanın gerçekten yapıldığını (ve ROI dışının DEĞİŞMEDİĞİNİ) doğrudan doğrula.
    base = np.full((200, 300, 3), 200, dtype=np.uint8)
    patch = np.zeros((100, 100), dtype=np.uint8)
    patch[40:60, 40:60] = 255
    composed = _paste_filtered_into_frame(
        base, patch, window.graph, window.pipeline.order, window.pipeline.order[-1]
    )
    assert composed is not None
    assert set(np.unique(composed[60:140, 110:190, 0]).tolist()) <= {0, 255}  # ROI içi: filtre
    assert (composed[10:40, 10:40] == 200).all()  # ROI dışı: ham kare


def test_in_context_view_falls_back_when_there_is_no_roi_crop(qtbot):
    """Zincirde ROI yoksa yapıştırmanın anlamı kalmaz -- `None` dönüp normal 'filtrelenmiş'
    davranışa düşülmeli (çökme/boş ekran yok)."""
    from imgflow.ui.main_window import _paste_filtered_into_frame

    window = MainWindow()
    qtbot.addWidget(window)
    base = np.full((100, 100, 3), 128, dtype=np.uint8)

    assert _paste_filtered_into_frame(base, base.copy(), window.graph, [], None) is None
    assert _paste_filtered_into_frame(None, base, window.graph, [], "x") is None


def test_selecting_operators_never_widens_the_right_panel(qtbot):
    """Gerçek kullanıcı raporu: "gereksiz sayfa büyümeleri". Kök neden ölçülerek bulundu:
    uzun Türkçe parametre etiketleri (ör. "Bulanıklık Yarıçapı (px, sadece 'Yerel/Dinamik'
    modda)") sarmadıkları için TEK BAŞLARINA `ParamForm`'un -- ve dolayısıyla sağ sekme
    panelinin, `central_splitter`'ın ve ana pencerenin -- minimum genişliğini büyütüyordu
    (`correction.flat_field` seçilince sağ panel minimumu 358px'ten 880px'e çıkıyor, splitter
    panoları yeniden dağıtılıp görüntü paneli daralıyordu). Etiket sarma + parametre
    sekmesinin kaydırma alanına alınması bu talebi ÖZÜMSER."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    base_min_width = window.right_tabs.minimumSizeHint().width()

    # Bilinen en uzun etiketli operatörler (ölçümde en çok büyüme yapan üçü).
    for op_id in ("correction.flat_field", "analysis.region_props", "analysis.texture_props"):
        node_id = window.add_operator(op_id)
        assert window.right_tabs.minimumSizeHint().width() == base_min_width, op_id
        window.remove_operator(node_id)


def test_long_status_text_does_not_widen_the_center_panel(qtbot):
    """Aynı raporun ikinci kök nedeni: `status_label` sarmasız olduğu için uzun bir hata
    mesajı ORTA panelin (ve pencerenin) minimum genişliğini metnin TAM uzunluğu kadar
    büyütüyordu."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    center = window.central_splitter.widget(1)
    base = center.minimumSizeHint().width()

    window.status_label.setText(
        "Hata: Node 'flat_field' (correction.flat_field) çalıştırılırken beklenmeyen bir "
        "sorun oluştu ve bu mesaj bilerek çok uzun tutuldu ki panel genişliğini zorlasın."
    )

    assert center.minimumSizeHint().width() == base


def test_hover_info_shows_pixels_next_to_calibrated_units(qtbot):
    """Gerçek kullanıcı isteği: "kalibrasyon varsa pixelin yanında cm veya mm de yazsın" --
    `region_props` dalı kalibrasyon aktifken px değerini TAMAMEN gizliyordu."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._on_hover_measurement_changed(
        {
            "label": 1,
            "area": 1000.0,
            "area_mm2": 250.0,
            "perimeter": 120.0,
            "perimeter_mm": 60.0,
            "obb_w": 50.0,
            "obb_h": 20.0,
            "obb_mm_w": 25.0,
            "obb_mm_h": 10.0,
            "obb_angle": 0.0,
        }
    )

    text = window.hover_info_label.text()
    assert "50 x 20 px" in text and "2.50 x 1.00 cm" in text
    assert "1000 px²" in text and "2.50 cm²" in text
    assert "120 px" in text and "60.0 mm" in text


def test_switching_right_tabs_reasserts_splitter_sizes(qtbot):
    """Gerçek kullanıcı raporu: "roi seçtikten sonra başka bir sekmeye geçince zoomluyor" --
    `right_tabs`'ın (Parametreler/Kamera Ayarları/Sonuçlar) aktif sekmesi değişince, o
    sekmenin farklı bir `sizeHint()`'i `central_splitter`'ın panolar arası alanı sessizce
    yeniden dağıtmasına yol açabiliyordu -- bu da orta (görüntü) panelinin genişliğini
    değiştirip `ImageView._rescale()`'in sığdırma ölçeğini kaydırıyor, kullanıcıya istenmeyen
    bir zoom gibi görünüyordu. `_on_right_tabs_changed` artık sekme değişiminden hemen önceki
    boyutları yakalayıp bir sonraki olay turunda tekrar uyguluyor."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()

    calls: list[list[int]] = []
    original_set_sizes = window.central_splitter.setSizes

    def spy(sizes):
        calls.append(list(sizes))
        original_set_sizes(sizes)

    window.central_splitter.setSizes = spy
    current_sizes = window.central_splitter.sizes()

    window.right_tabs.setCurrentWidget(window.camera_settings_panel)
    qtbot.wait(50)

    assert calls
    assert calls[-1] == current_sizes


def test_region_props_preview_is_drawn_on_the_upstream_color_image_not_a_binary_silhouette(
    qtbot, tmp_path
):
    """Gerçek kullanıcı raporu: "hsv ile bölge ölçümünde siyah beyaz yapıyor."

    `analysis.region_props` overlay'ini ZORUNLU olarak ikili (siyah/beyaz) bir siluet üzerine
    çizer -- `LinearPipeline`'ın tek-girdi zincirinde ona sadece `labels` haritası ulaşır,
    kaynak görüntüye erişimi YOKTUR (bkz. CLAUDE.md). Bu yüzden renkli bir HSV maskesinin
    ardından Bölge Ölçümü adımı seçilince önizleme birden siyah/beyaza dönüyordu. Önizleme
    tabanı artık zincirdeki en yakın GERÇEK görüntüye taşınıyor; segmentasyonun kendisini
    görmek isteyen kullanıcı bir önceki adımı (Bağlı Bileşenler) seçebilir."""
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[10:30, 10:30] = (40, 200, 60)  # belirgin YEŞİL nesne
    path = tmp_path / "green.png"
    cv2.imwrite(str(path), img)

    window = MainWindow()
    qtbot.addWidget(window)
    src_id = window.add_operator("io.image_source")
    window.add_operator("color.hsv")
    window.add_operator("segment.connected_components")
    props_id = window.add_operator("analysis.region_props")

    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})
    window._on_step_selected(props_id)

    shown = _build_preview_frame(
        engine=window.engine,
        registry=window.registry,
        graph=window.graph,
        pipeline_order=window.pipeline.order,
        node_id=props_id,
        view_mode="filtered",
        camera_active=False,
        last_camera_frame=None,
    ).display_image
    assert shown is not None and shown.ndim == 3
    # Nesnenin ortasındaki piksel hâlâ YEŞİL (gri/ikili bir siluette üç kanal EŞİT olurdu).
    b, g, r = (int(v) for v in shown[20, 20])
    assert g > b + 40 and g > r + 40


def test_in_context_view_pastes_only_the_circle_not_its_black_bounding_box(qtbot, tmp_path):
    """Gerçek kullanıcı raporu: "circle roi aldıktan sonra hsv ayarı yaptım, roi bağlamda
    olacak şekilde -- circle'ın etrafında bir kutu oluşturdu ve içini siyah yaptı."

    `roi.region`'ın daire modu çıktıyı dairenin KARE kutusuna kırpıp köşeleri siyaha boyar;
    "ROI Bağlamda" bu kareyi olduğu gibi yapıştırdığında o siyah köşeler ham karenin gerçek
    içeriğini siliyordu. Artık sadece daire İÇİ yapıştırılır."""
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[:] = (30, 90, 200)  # düz, siyah OLMAYAN arka plan
    img[30:50, 30:50] = (40, 200, 60)  # dairenin ortasında ayırt edilebilir bir nesne
    path = tmp_path / "circle_scene.png"
    cv2.imwrite(str(path), img)

    window = MainWindow()
    qtbot.addWidget(window)
    src_id = window.add_operator("io.image_source")
    roi_id = window.add_operator("roi.region")
    hsv_id = window.add_operator("color.hsv")

    window._on_step_selected(src_id)
    window.param_form.params_changed.emit({"path": str(path)})
    window._on_step_selected(roi_id)
    window.param_form.params_changed.emit(
        {"enabled": True, "shape": "CIRCLE", "x": 0, "y": 0, "w": 100, "h": 100,
         "cx": 40, "cy": 40, "r": 20}
    )

    shown = _build_preview_frame(
        engine=window.engine,
        registry=window.registry,
        graph=window.graph,
        pipeline_order=window.pipeline.order,
        node_id=hsv_id,
        view_mode="in_context",
        camera_active=False,
        last_camera_frame=None,
    ).display_image

    assert shown is not None and shown.shape[:2] == img.shape[:2]
    # Sınırlayıcı kutunun köşesi (daire DIŞI) -- ham arka plan KORUNMALI. (21,21) düzeltme
    # olmadan turuncu DİKDÖRTGEN çerçevesine, (24,24) ise `roi.region`'ın siyaha boyadığı
    # köşeye denk geliyordu; ikisi de kullanıcının bildirdiği iki belirtiyi sabitliyor.
    assert tuple(int(v) for v in shown[21, 21]) == (30, 90, 200)
    assert tuple(int(v) for v in shown[24, 24]) == (30, 90, 200)
    # Dairenin merkezi -- işlenmiş içerik yapıştırılmış olmalı.
    assert tuple(int(v) for v in shown[40, 40]) == (40, 200, 60)
