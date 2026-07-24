import cv2
import numpy as np
import pytest

from imgflow.core.camera_source import BaslerCameraSource
from imgflow.core.errors import ImgflowError
from imgflow.core.params import ParamType
from imgflow.ui.main_window import MainWindow
from tests.support.fake_genicam import FakeGenicamNode, FakeNodeMap, default_camera_nodes


def _sample_image_path(tmp_path):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[5:15, 5:15] = 255
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)
    return path


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


def test_height_scale_action_enabled_for_any_camera_and_opens_dialog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window._height_scale_action.isEnabled() is False

    window.start_camera(_FakeCameraSource([]))
    assert window._height_scale_action.isEnabled() is True

    window._on_open_height_scale_calibration()
    assert window._height_scale_dialog is not None

    window.stop_camera()
    assert window._height_scale_action.isEnabled() is False
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


def test_lens_calibration_action_enabled_for_any_camera(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    window.start_camera(_FakeCameraSource([]))
    assert window._lens_calibration_action.isEnabled() is True

    window.stop_camera()
    assert window._lens_calibration_action.isEnabled() is False


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


def test_start_camera_injects_frame_and_creates_source_node(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    frame = np.full((10, 10, 3), 128, dtype=np.uint8)

    window.start_camera(_FakeCameraSource([frame]))
    window._on_camera_tick()

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

    result = window._get_normal_base_image()

    assert result is frame


def test_get_normal_base_image_returns_none_without_camera_or_source(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)

    assert window._get_normal_base_image() is None


def test_get_normal_base_image_returns_static_source_image(qtbot, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    path = _sample_image_path(tmp_path)
    window.open_image(str(path))

    result = window._get_normal_base_image()

    assert result is not None
    assert result.shape[:2] == (20, 20)


def test_compose_display_image_filtered_mode_returns_filtered_image_unchanged(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "filtered"
    filtered = np.zeros((5, 5, 3), dtype=np.uint8)

    result, hover_measurements = window._compose_display_image(filtered, None, 0.0)

    assert result is filtered
    assert hover_measurements is None


def test_compose_display_image_filtered_mode_passes_through_measurements_for_hover(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "filtered"
    filtered = np.zeros((5, 5, 3), dtype=np.uint8)
    measurements = [{"bbox_x": 1, "bbox_y": 1, "bbox_w": 2, "bbox_h": 2}]

    _, hover_measurements = window._compose_display_image(filtered, measurements, 0.0)

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

    result, hover_measurements = window._compose_display_image(
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

    result, hover_measurements = window._compose_display_image(np.zeros((10, 10), dtype=np.uint8), None, 0.0)

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

    result, hover_measurements = window._compose_display_image(filtered, None, 0.0)

    assert result.shape[0] == 10
    assert result.shape[1] == 8 + 4 + 8  # sol + ayırıcı (4px) + sağ
    assert hover_measurements is None  # "both" modunda üzerine gelme desteklenmiyor


def test_compose_display_image_both_mode_falls_back_to_single_side_when_other_missing(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window._view_mode = "both"
    # kamera yok, io.image_source node'u da yok -> normal taraf None olmalı
    filtered = np.zeros((10, 8, 3), dtype=np.uint8)

    result, hover_measurements = window._compose_display_image(filtered, None, 0.0)

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

    result, _ = window._compose_display_image(np.zeros((1200, 2400), dtype=np.uint8), None, 0.0)

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
