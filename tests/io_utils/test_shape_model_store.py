import os

import numpy as np
import pytest

from imgflow.core.shape_matching import ShapeLevel, ShapeModel
from imgflow.io_utils import shape_model_store
from imgflow.io_utils.shape_model_store import (
    ShapeModelNotFoundError,
    delete_shape_model,
    list_shape_models,
    load_shape_model,
    rename_shape_model,
    save_shape_model,
)


@pytest.fixture(autouse=True)
def _clear_load_cache():
    shape_model_store._load_cache.clear()
    yield
    shape_model_store._load_cache.clear()


def _sample_model() -> ShapeModel:
    level = ShapeLevel(
        points=np.array([[1.0, 2.0], [3.0, 4.0]]),
        angles=np.array([0.1, 0.2]),
    )
    return ShapeModel(levels=[level], corners=np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]))


def test_save_and_load_round_trips(tmp_path):
    model = _sample_model()

    save_shape_model("test_model", model, directory=tmp_path)
    loaded = load_shape_model("test_model", directory=tmp_path)

    assert len(loaded.levels) == 1
    assert np.allclose(loaded.levels[0].points, model.levels[0].points)
    assert np.allclose(loaded.levels[0].angles, model.levels[0].angles)
    assert np.allclose(loaded.corners, model.corners)


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(ShapeModelNotFoundError):
        load_shape_model("yok_boyle_bir_model", directory=tmp_path)


def test_list_shape_models_returns_saved_names(tmp_path):
    save_shape_model("model_a", _sample_model(), directory=tmp_path)
    save_shape_model("model_b", _sample_model(), directory=tmp_path)

    names = list_shape_models(directory=tmp_path)

    assert sorted(names) == ["model_a", "model_b"]


def test_list_shape_models_empty_directory_returns_empty_list(tmp_path):
    assert list_shape_models(directory=tmp_path / "does_not_exist") == []


def test_delete_shape_model_removes_it(tmp_path):
    save_shape_model("silinecek", _sample_model(), directory=tmp_path)

    delete_shape_model("silinecek", directory=tmp_path)

    assert list_shape_models(directory=tmp_path) == []


def test_delete_missing_model_is_noop(tmp_path):
    delete_shape_model("hic_olmadi", directory=tmp_path)  # hata fırlatmamalı


def test_rename_shape_model_updates_file_and_stored_name(tmp_path):
    save_shape_model("eski_ad", _sample_model(), directory=tmp_path)

    rename_shape_model("eski_ad", "yeni_ad", directory=tmp_path)

    assert list_shape_models(directory=tmp_path) == ["yeni_ad"]
    loaded = load_shape_model("yeni_ad", directory=tmp_path)
    assert np.allclose(loaded.corners, _sample_model().corners)
    with pytest.raises(ShapeModelNotFoundError):
        load_shape_model("eski_ad", directory=tmp_path)


def test_rename_missing_model_raises(tmp_path):
    with pytest.raises(ShapeModelNotFoundError):
        rename_shape_model("yok_boyle_model", "her_neyse", directory=tmp_path)


def test_rename_to_existing_name_raises_without_overwriting(tmp_path):
    save_shape_model("model_a", _sample_model(), directory=tmp_path)
    save_shape_model("model_b", _sample_model(), directory=tmp_path)

    with pytest.raises(FileExistsError):
        rename_shape_model("model_a", "model_b", directory=tmp_path)

    assert sorted(list_shape_models(directory=tmp_path)) == ["model_a", "model_b"]


def test_load_shape_model_returns_cached_object_when_mtime_unchanged(tmp_path):
    """`geom.shape_match` canlı kamerada her tick'te `load_shape_model`'i çağırıyor —
    dosyanın `mtime`'ı değişmediği sürece diskten TEKRAR okuyup JSON parse etmemeli. Bunu,
    dosyayı bozup mtime'ı ESKİ değerine geri alarak kanıtlıyoruz: önbellek gerçekten
    kullanılmıyorsa bu ikinci çağrı `json.JSONDecodeError` ile patlardı."""
    save_shape_model("cached_model", _sample_model(), directory=tmp_path)
    first = load_shape_model("cached_model", directory=tmp_path)

    path = tmp_path / "cached_model.json"
    original_mtime = path.stat().st_mtime
    path.write_text("bozuk json degil", encoding="utf-8")
    os.utime(path, (original_mtime, original_mtime))

    second = load_shape_model("cached_model", directory=tmp_path)

    assert second is first


def test_load_shape_model_reloads_when_resaved_under_same_name(tmp_path):
    """Kullanıcı `ShapeMatchingDialog`'da AYNI isimle modeli yeniden eğitip kaydedebilir —
    bu, ONNX modellerinin aksine, "bir kere yükle sonsuza dek önbellekte tut" YETERSİZ
    kalır. `save_shape_model` sonrası bir sonraki `load_shape_model` GÜNCEL içeriği
    döndürmeli."""
    save_shape_model("cached_model", _sample_model(), directory=tmp_path)
    first = load_shape_model("cached_model", directory=tmp_path)

    retrained = ShapeModel(
        levels=[ShapeLevel(points=np.array([[9.0, 9.0]]), angles=np.array([0.5]))],
        corners=_sample_model().corners,
    )
    save_shape_model("cached_model", retrained, directory=tmp_path)
    second = load_shape_model("cached_model", directory=tmp_path)

    assert second is not first
    assert len(second.levels[0].points) == 1


def test_load_shape_model_cache_does_not_leak_across_directories(tmp_path):
    """Farklı `tmp_path`'lerle AYNI model ismi kullanan testler önbellek üzerinden
    çapraz kirlenmemeli (anahtar isim+dizin çiftidir)."""
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    model_b = ShapeModel(
        levels=[ShapeLevel(points=np.array([[9.0, 9.0]]), angles=np.array([0.5]))],
        corners=_sample_model().corners,
    )
    save_shape_model("same_name", _sample_model(), directory=dir_a)
    save_shape_model("same_name", model_b, directory=dir_b)

    loaded_a = load_shape_model("same_name", directory=dir_a)
    loaded_b = load_shape_model("same_name", directory=dir_b)

    assert len(loaded_a.levels[0].points) == 2
    assert len(loaded_b.levels[0].points) == 1


def test_delete_shape_model_invalidates_cache(tmp_path):
    save_shape_model("silinecek", _sample_model(), directory=tmp_path)
    load_shape_model("silinecek", directory=tmp_path)

    delete_shape_model("silinecek", directory=tmp_path)

    with pytest.raises(ShapeModelNotFoundError):
        load_shape_model("silinecek", directory=tmp_path)


def test_functions_respect_monkeypatched_default_directory_without_explicit_arg(tmp_path, monkeypatch):
    """`ShapeMatchOp.run()` gibi çağıranlar `directory=` argümanını HİÇ vermez — sadece
    `SHAPE_MODEL_DIR` monkeypatch edilir. Varsayılan değer `def` anında değil ÇAĞRI anında
    çözülmeli, aksi halde bu test gerçek `~/.imgflow`'a sızardı (bir kere gerçekten oldu)."""
    monkeypatch.setattr(shape_model_store, "SHAPE_MODEL_DIR", tmp_path / "shape_models")

    save_shape_model("izole_model", _sample_model())  # directory= kasıtlı verilmiyor

    assert (tmp_path / "shape_models" / "izole_model.json").exists()
    assert list_shape_models() == ["izole_model"]
    assert load_shape_model("izole_model") is not None
