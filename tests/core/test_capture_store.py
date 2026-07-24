import numpy as np
import pytest

from imgflow.core import capture_store


@pytest.fixture(autouse=True)
def _isolated_capture_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")


def _image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_save_capture_persists_file_and_index_entry():
    record = capture_store.save_capture(_image(), source="lens")

    assert record.path.exists()
    records = capture_store.list_captures()
    assert len(records) == 1
    assert records[0].id == record.id
    assert records[0].source == "lens"


def test_list_captures_filters_by_source():
    capture_store.save_capture(_image(), source="lens")
    capture_store.save_capture(_image(), source="height_scale")

    assert len(capture_store.list_captures(source="lens")) == 1
    assert len(capture_store.list_captures(source="height_scale")) == 1
    assert len(capture_store.list_captures()) == 2


def test_list_captures_newest_first():
    first = capture_store.save_capture(_image(), source="lens")
    second = capture_store.save_capture(_image(), source="lens")

    records = capture_store.list_captures()
    assert records[0].id == second.id
    assert records[1].id == first.id


def test_delete_capture_removes_file_and_index_entry():
    record = capture_store.save_capture(_image(), source="lens")

    capture_store.delete_capture(record.id)

    assert capture_store.list_captures() == []
    assert not record.path.exists()


def test_delete_unknown_capture_is_a_no_op():
    capture_store.delete_capture("does-not-exist")
    assert capture_store.list_captures() == []


def test_save_capture_prunes_oldest_beyond_max(monkeypatch):
    monkeypatch.setattr(capture_store, "MAX_CAPTURES", 3)
    records = [capture_store.save_capture(_image(), source="lens") for _ in range(5)]

    remaining = capture_store.list_captures()
    assert len(remaining) == 3
    remaining_ids = {r.id for r in remaining}
    assert records[0].id not in remaining_ids
    assert records[1].id not in remaining_ids
    assert not records[0].path.exists()
