import pytest

from imgflow.core.camera_params import (
    CATEGORY_ACQUISITION,
    CATEGORY_DIGITAL_IO,
    CATEGORY_IMAGE_FORMAT,
    CATEGORY_IMAGE_QUALITY,
    CATEGORY_USER_SETS,
    CameraParameterController,
)
from imgflow.core.params import ParamType
from tests.support.fake_genicam import FakeGenicamNode, FakeNodeMap, default_camera_nodes


def _controller(**overrides: FakeGenicamNode) -> CameraParameterController:
    return CameraParameterController(FakeNodeMap(default_camera_nodes(**overrides)))


def test_build_specs_skips_unavailable_nodes():
    controller = _controller()
    specs = controller.build_specs(CATEGORY_ACQUISITION)
    names = {s.name for s in specs}
    assert "TriggerMode" in names
    assert "TriggerSource" not in names  # available=False


def test_build_specs_skips_command_nodes():
    controller = _controller()
    specs = controller.build_specs()
    assert "UserSetSave" not in {s.name for s in specs}


def test_build_specs_reads_min_max_and_default_from_node():
    controller = _controller()
    spec = next(s for s in controller.build_specs(CATEGORY_IMAGE_QUALITY) if s.name == "ExposureTime")
    assert spec.type is ParamType.FLOAT
    assert spec.default == 1000.0
    assert spec.min == 10.0
    assert spec.max == 10000.0


def test_build_specs_enum_uses_node_entries_as_choices():
    controller = _controller()
    spec = next(s for s in controller.build_specs(CATEGORY_IMAGE_QUALITY) if s.name == "BalanceRatioSelector")
    assert spec.type is ParamType.ENUM
    assert spec.choices == ["Red", "Green", "Blue"]
    assert spec.default == "Red"


def test_build_specs_marks_non_writable_node_as_readonly():
    controller = _controller(Gain=FakeGenicamNode(5.0, min=0.0, max=24.0, writable=False))
    spec = next(s for s in controller.build_specs(CATEGORY_IMAGE_QUALITY) if s.name == "Gain")
    assert spec.readonly is True


def test_set_returns_hardware_clamped_value():
    controller = _controller()
    actual = controller.set("ExposureTime", 999999.0)
    assert actual == 10000.0  # node max=10000.0


def test_current_values_reads_live_node_values():
    controller = _controller()
    specs = controller.build_specs(CATEGORY_IMAGE_QUALITY)
    controller.set("Gain", 3.5)
    values = controller.current_values(specs)
    assert values["Gain"] == 3.5


def test_is_structural_true_for_selector_and_mode_nodes():
    controller = _controller()
    assert controller.is_structural("TriggerMode") is True
    assert controller.is_structural("BalanceRatioSelector") is True
    assert controller.is_structural("UserSetSelector") is True


def test_is_structural_false_for_plain_scalar_nodes():
    controller = _controller()
    assert controller.is_structural("ExposureTime") is False
    assert controller.is_structural("Gain") is False
    assert controller.is_structural("Width") is False


def test_is_structural_false_for_unknown_node():
    controller = _controller()
    assert controller.is_structural("DoesNotExist") is False


def test_command_defs_returns_only_available_commands():
    controller = _controller()
    names = {d.node_name for d in controller.command_defs(CATEGORY_USER_SETS)}
    assert names == {"UserSetSave"}


def test_command_defs_empty_when_command_node_unavailable():
    controller = _controller(UserSetSave=FakeGenicamNode(False, available=False))
    assert controller.command_defs(CATEGORY_USER_SETS) == []


def test_build_specs_includes_help_text_for_all_categories():
    controller = _controller()
    for category in (
        CATEGORY_IMAGE_QUALITY,
        CATEGORY_ACQUISITION,
        CATEGORY_DIGITAL_IO,
        CATEGORY_USER_SETS,
    ):
        for spec in controller.build_specs(category):
            assert spec.help, f"{category}/{spec.name} için help metni eksik"


def test_set_image_format_field_stops_and_restarts_streaming():
    controller = _controller()
    node_map = controller._node_map

    controller.set("Width", 800)

    assert node_map.stop_streaming_calls == 1
    assert node_map.start_streaming_calls == 1


def test_set_image_format_field_restarts_streaming_even_if_write_fails():
    controller = _controller(Width=FakeGenicamNode(1920, min=1, max=1920, writable=False))
    node_map = controller._node_map

    with pytest.raises(RuntimeError):
        controller.set("Width", 800)

    assert node_map.stop_streaming_calls == 1
    assert node_map.start_streaming_calls == 1  # finally bloğu: yazma patlasa da akış yeniden başlamalı


def test_set_non_format_field_does_not_touch_streaming():
    controller = _controller()
    node_map = controller._node_map

    controller.set("Gain", 3.5)

    assert node_map.stop_streaming_calls == 0
    assert node_map.start_streaming_calls == 0


def test_build_specs_image_format_category_present():
    controller = _controller()
    names = {s.name for s in controller.build_specs(CATEGORY_IMAGE_FORMAT)}
    assert "Width" in names
    assert "PixelFormat" in names


def test_execute_command_node():
    save_node = FakeGenicamNode(False)
    controller = _controller(UserSetSave=save_node)
    controller.execute("UserSetSave")
    assert save_node.get_value() is True
