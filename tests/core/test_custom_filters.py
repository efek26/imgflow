import numpy as np
import pytest

from imgflow.core import custom_filters
from imgflow.core.custom_filters import CustomFilterDef, CustomFilterError
from imgflow.operators import registry as registry_module


@pytest.fixture(autouse=True)
def _isolated_custom_filter_dir(tmp_path, monkeypatch):
    """Gerçek ~/.imgflow/custom_filters dizinine asla dokunma — her testte izole tmp_path."""
    monkeypatch.setattr(custom_filters, "CUSTOM_FILTER_DIR", tmp_path / "custom_filters")
    yield
    # Testin registry'ye kaydettiği her şeyi temizle (diğer testleri kirletmesin).
    for op_id in list(registry_module._ops):
        if op_id.startswith(custom_filters.OP_ID_PREFIX):
            registry_module.unregister(op_id)


def _image() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_compile_apply_returns_callable():
    fn = custom_filters.compile_apply("def apply(image):\n    return image\n")
    result = fn(_image())
    assert isinstance(result, np.ndarray)


def test_compile_apply_syntax_error_raises_custom_filter_error():
    with pytest.raises(CustomFilterError, match="Kod hatası"):
        custom_filters.compile_apply("def apply(image)\n    return image\n")


def test_compile_apply_missing_apply_function_raises():
    with pytest.raises(CustomFilterError, match="apply"):
        custom_filters.compile_apply("x = 1\n")


def test_run_apply_wraps_user_code_exception():
    fn = custom_filters.compile_apply("def apply(image):\n    raise ValueError('boom')\n")
    with pytest.raises(CustomFilterError, match="boom"):
        custom_filters.run_apply(fn, _image())


def test_run_apply_requires_ndarray_return():
    fn = custom_filters.compile_apply("def apply(image):\n    return 42\n")
    with pytest.raises(CustomFilterError, match="numpy"):
        custom_filters.run_apply(fn, _image())


def test_preview_runs_end_to_end_using_cv2():
    defn = CustomFilterDef(
        name="Test Bulanıklık", code="def apply(image):\n    return cv2.GaussianBlur(image, (3, 3), 0)\n"
    )
    result = custom_filters.preview(defn, _image())
    assert result.shape == _image().shape


def test_save_custom_filter_persists_and_registers():
    defn = CustomFilterDef(name="Benim Filtrem", code="def apply(image):\n    return image\n")
    custom_filters.save_custom_filter(defn)

    op_id = custom_filters.op_id_for("Benim Filtrem")
    assert registry_module.get(op_id) is not None
    saved = custom_filters.list_custom_filters()
    assert any(d.name == "Benim Filtrem" for d in saved)


def test_save_custom_filter_with_bad_code_does_not_write_to_disk():
    defn = CustomFilterDef(name="Bozuk Filtre", code="this is not python(\n")
    with pytest.raises(CustomFilterError):
        custom_filters.save_custom_filter(defn)

    assert custom_filters.list_custom_filters() == []


def test_delete_custom_filter_removes_file_and_unregisters():
    defn = CustomFilterDef(name="Silinecek", code="def apply(image):\n    return image\n")
    custom_filters.save_custom_filter(defn)

    custom_filters.delete_custom_filter("Silinecek")

    assert custom_filters.list_custom_filters() == []
    with pytest.raises(Exception):
        registry_module.get(custom_filters.op_id_for("Silinecek"))


def test_register_custom_filter_overwrites_previous_version_without_error():
    defn_v1 = CustomFilterDef(name="Sürümlü", code="def apply(image):\n    return image\n")
    defn_v2 = CustomFilterDef(name="Sürümlü", code="def apply(image):\n    return image * 0\n")
    custom_filters.register_custom_filter(defn_v1)
    custom_filters.register_custom_filter(defn_v2)  # ValueError fırlatmamalı (register_or_replace)

    op_cls = registry_module.get(custom_filters.op_id_for("Sürümlü"))
    result = op_cls().run({"image": np.ones((2, 2, 3), dtype=np.uint8)}, {})
    assert result["image"].sum() == 0


def test_load_all_custom_filters_skips_broken_ones():
    good = CustomFilterDef(name="Iyi", code="def apply(image):\n    return image\n")
    custom_filters.save_custom_filter(good)
    # Bozuk bir dosyayı elle diske koy (save_custom_filter zaten bozuk kodu reddediyor).
    custom_filters.CUSTOM_FILTER_DIR.mkdir(parents=True, exist_ok=True)
    (custom_filters.CUSTOM_FILTER_DIR / "bozuk.json").write_text(
        '{"name": "Bozuk", "code": "this is not python("}', encoding="utf-8"
    )

    failures = custom_filters.load_all_custom_filters()

    assert len(failures) == 1
    assert failures[0][0].name == "Bozuk"
    assert registry_module.get(custom_filters.op_id_for("Iyi")) is not None
