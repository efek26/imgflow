import pytest

from imgflow.io_utils import onnx_model_store


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    """Gerçek ~/.imgflow dizinine ASLA dokunma — izole tmp_path'e yönlendir."""
    monkeypatch.setattr(onnx_model_store, "ONNX_MODEL_DIR", tmp_path / "onnx_models")


@pytest.fixture
def fake_onnx_file(tmp_path):
    path = tmp_path / "source_model.onnx"
    path.write_bytes(b"fake-onnx-bytes")
    return path


def test_list_models_empty_when_dir_missing():
    assert onnx_model_store.list_models() == []


def test_save_and_load_roundtrip(fake_onnx_file):
    onnx_model_store.save_model(
        "Kusur Dedektoru", fake_onnx_file, "yolo", input_size=640, class_labels=["kusur", "saglam"]
    )

    meta = onnx_model_store.load_model_meta("Kusur Dedektoru")
    assert meta["name"] == "Kusur Dedektoru"
    assert meta["task_type"] == "yolo"
    assert meta["input_size"] == 640
    assert meta["class_labels"] == ["kusur", "saglam"]
    assert meta["path"].exists()
    assert meta["path"].read_bytes() == b"fake-onnx-bytes"


def test_original_file_can_be_deleted_after_save(fake_onnx_file, tmp_path):
    onnx_model_store.save_model("model1", fake_onnx_file, "yolo", 640, ["a"])
    fake_onnx_file.unlink()

    meta = onnx_model_store.load_model_meta("model1")
    assert meta["path"].exists()  # kopya diskte kalmalı, orijinal silinse bile


def test_list_models_returns_saved_names(fake_onnx_file):
    onnx_model_store.save_model("model_a", fake_onnx_file, "yolo", 640, ["a"])
    onnx_model_store.save_model("model_b", fake_onnx_file, "classification", 224, ["ok", "ng"])

    assert set(onnx_model_store.list_models()) == {"model_a", "model_b"}


def test_load_unknown_model_raises_not_found():
    with pytest.raises(onnx_model_store.OnnxModelNotFoundError):
        onnx_model_store.load_model_meta("yok_boyle_model")


def test_delete_model_removes_both_files(fake_onnx_file):
    onnx_model_store.save_model("model1", fake_onnx_file, "yolo", 640, ["a"])
    onnx_model_store.delete_model("model1")

    assert onnx_model_store.list_models() == []
    with pytest.raises(onnx_model_store.OnnxModelNotFoundError):
        onnx_model_store.load_model_meta("model1")


def test_save_with_unknown_task_type_raises_value_error(fake_onnx_file):
    with pytest.raises(ValueError):
        onnx_model_store.save_model("model1", fake_onnx_file, "not_a_real_task", 640, ["a"])


def test_save_with_placeholder_task_types_allowed(fake_onnx_file):
    onnx_model_store.save_model("model_cls", fake_onnx_file, "classification", 224, ["ok", "ng"])
    onnx_model_store.save_model("model_seg", fake_onnx_file, "segmentation", 512, ["defect"])

    assert onnx_model_store.load_model_meta("model_cls")["task_type"] == "classification"
    assert onnx_model_store.load_model_meta("model_seg")["task_type"] == "segmentation"
