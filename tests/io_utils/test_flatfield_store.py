import numpy as np
import pytest

from imgflow.io_utils import flatfield_store


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    """Gerçek ~/.imgflow dizinine ASLA dokunma — izole tmp_path'e yönlendir."""
    monkeypatch.setattr(flatfield_store, "FLATFIELD_DIR", tmp_path / "flatfield")


def test_list_references_empty_when_dir_missing():
    assert flatfield_store.list_references() == []


def test_save_and_load_roundtrip():
    image = np.full((20, 30), 180, dtype=np.uint8)
    flatfield_store.save_reference("Hat1 Bos Bant", image)

    loaded = flatfield_store.load_reference("Hat1 Bos Bant")
    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)


def test_list_references_returns_saved_name():
    flatfield_store.save_reference("hat1", np.zeros((10, 10), dtype=np.uint8))
    flatfield_store.save_reference("hat2", np.zeros((10, 10), dtype=np.uint8))

    assert set(flatfield_store.list_references()) == {"hat1", "hat2"}


def test_load_unknown_reference_raises_not_found():
    with pytest.raises(flatfield_store.FlatFieldReferenceNotFoundError):
        flatfield_store.load_reference("yok_boyle_referans")


def test_delete_reference_removes_both_files():
    flatfield_store.save_reference("hat1", np.zeros((10, 10), dtype=np.uint8))
    flatfield_store.delete_reference("hat1")

    assert flatfield_store.list_references() == []
    with pytest.raises(flatfield_store.FlatFieldReferenceNotFoundError):
        flatfield_store.load_reference("hat1")


def test_delete_unknown_reference_is_noop():
    flatfield_store.delete_reference("yok_boyle_referans")


def test_save_reference_with_blank_name_raises_value_error():
    with pytest.raises(ValueError):
        flatfield_store.save_reference("   ", np.zeros((10, 10), dtype=np.uint8))
