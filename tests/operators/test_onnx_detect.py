import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from imgflow.core.onnx_detection import OnnxDetectionError
from imgflow.io_utils import onnx_model_store
from imgflow.operators import registry
from imgflow.operators.builtin.onnx_detect import OnnxDetectOp

_INPUT_SIZE = 32


def _build_constant_output_model(path, output_array: np.ndarray, input_size: int = _INPUT_SIZE) -> None:
    input_tensor = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, input_size, input_size]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, list(output_array.shape)
    )
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output"],
        value=helper.make_tensor(
            name="const_val",
            data_type=TensorProto.FLOAT,
            dims=list(output_array.shape),
            vals=output_array.astype(np.float32).flatten().tolist(),
        ),
    )
    graph = helper.make_graph([const_node], "test_graph", [input_tensor], [output_tensor])
    model = helper.make_model(graph, producer_name="imgflow-test")
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))


@pytest.fixture(autouse=True)
def _isolated_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(onnx_model_store, "ONNX_MODEL_DIR", tmp_path / "onnx_models")


@pytest.fixture
def yolo_model(tmp_path):
    output = np.array([[[16, 16, 8, 8, 0.9, 0.9, 0.05]]], dtype=np.float32)  # 1 kutu, class0
    onnx_path = tmp_path / "fake_yolo.onnx"
    _build_constant_output_model(onnx_path, output, _INPUT_SIZE)
    onnx_model_store.save_model(
        "kusur_dedektoru", onnx_path, "yolo", _INPUT_SIZE, ["kusur", "saglam"]
    )
    return "kusur_dedektoru"


@pytest.fixture
def classification_model(tmp_path):
    output = np.array([[2.0, 0.5]], dtype=np.float32)  # softmax -> class0 ("ok") kazanır
    onnx_path = tmp_path / "fake_cls.onnx"
    _build_constant_output_model(onnx_path, output, _INPUT_SIZE)
    onnx_model_store.save_model("siniflandirici", onnx_path, "classification", _INPUT_SIZE, ["ok", "ng"])
    return "siniflandirici"


@pytest.fixture
def segmentation_model(tmp_path):
    # (1, C=3, H=4, W=4) -- her piksel için class_id=1 en yüksek skor (arka plan=0 hariç
    # görüntünün TAMAMI "kusur" sayılır).
    h = w = 4
    output = np.zeros((1, 3, h, w), dtype=np.float32)
    output[0, 1, :, :] = 5.0
    onnx_path = tmp_path / "fake_seg.onnx"
    _build_constant_output_model(onnx_path, output, _INPUT_SIZE)
    onnx_model_store.save_model(
        "segmentleyici", onnx_path, "segmentation", _INPUT_SIZE, ["arka_plan", "kusur", "saglam"]
    )
    return "segmentleyici"


def test_registered():
    assert registry.get("ml.onnx_detect") is not None


def test_empty_model_names_raises_value_error():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="model_names"):
        OnnxDetectOp().run({"image": image}, {"model_names": ""})


def test_unknown_model_name_raises_not_found():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(onnx_model_store.OnnxModelNotFoundError):
        OnnxDetectOp().run({"image": image}, {"model_names": "yok_boyle_model"})


def test_run_produces_measurement_and_overlay(yolo_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image}, {"model_names": yolo_model, "min_confidence": 0.25, "nms_threshold": 0.45}
    )

    assert len(out["measurements"]) == 1
    m = out["measurements"][0]
    assert m["model"] == yolo_model
    assert m["label"] == "1"
    assert m["class_name"] == "kusur"
    assert m["confidence"] == pytest.approx(0.81, abs=1e-3)
    assert out["overlay"].shape == image.shape


def test_multiple_models_labeled_sequentially(tmp_path, yolo_model):
    output = np.array([[[16, 16, 8, 8, 0.9, 0.9, 0.05]]], dtype=np.float32)
    onnx_path = tmp_path / "fake_yolo2.onnx"
    _build_constant_output_model(onnx_path, output, _INPUT_SIZE)
    onnx_model_store.save_model("ikinci_model", onnx_path, "yolo", _INPUT_SIZE, ["kusur", "saglam"])

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image}, {"model_names": f"{yolo_model}, ikinci_model", "min_confidence": 0.25}
    )

    assert len(out["measurements"]) == 2
    labels = {m["label"] for m in out["measurements"]}
    models = {m["model"] for m in out["measurements"]}
    assert labels == {"1", "2"}
    assert models == {yolo_model, "ikinci_model"}


def test_classification_run_produces_single_row_no_bbox(classification_model):
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    out = OnnxDetectOp().run({"image": image}, {"model_names": classification_model})

    assert len(out["measurements"]) == 1
    m = out["measurements"][0]
    assert m["model"] == classification_model
    assert m["class_name"] == "ok"
    assert "bbox_x" not in m
    assert out["overlay"].shape == image.shape


def test_classification_defect_classes_marks_ng(classification_model):
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image}, {"model_names": classification_model, "defect_classes": "ok"}
    )
    assert out["measurements"][0]["tolerance_ok"] is False


def test_segmentation_run_produces_one_row_per_present_class_excluding_background(segmentation_model):
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    out = OnnxDetectOp().run({"image": image}, {"model_names": segmentation_model})

    assert len(out["measurements"]) == 1  # class_id=0 (arka plan) atlanır
    m = out["measurements"][0]
    assert m["class_name"] == "kusur"
    assert m["area_percent"] == pytest.approx(100.0, abs=0.01)
    assert out["overlay"].shape == image.shape


def test_segmentation_defect_classes_marks_ng(segmentation_model):
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image}, {"model_names": segmentation_model, "defect_classes": "kusur"}
    )
    assert out["measurements"][0]["tolerance_ok"] is False


def test_mixed_task_types_combine_measurements_and_overlays(yolo_model, classification_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image},
        {"model_names": f"{yolo_model}, {classification_model}", "min_confidence": 0.25},
    )

    assert len(out["measurements"]) == 2
    models = {m["model"] for m in out["measurements"]}
    assert models == {yolo_model, classification_model}
    assert out["overlay"].shape == image.shape


def test_unrecognized_task_type_raises_clear_error(tmp_path, monkeypatch):
    onnx_path = tmp_path / "placeholder.onnx"
    onnx_path.write_bytes(b"not-a-real-model")
    onnx_model_store.save_model("bilinmeyen", onnx_path, "yolo", 224, ["a"])
    # `save_model` sadece `SUPPORTED_TASK_TYPES` kabul ettiğinden gerçek bir kayıt yoluyla
    # bilinmeyen bir tür ASLA diske yazılamaz -- operatördeki savunmacı `else` dalını
    # (yeni bir görev türü eklenip operatöre unutulursa diye) doğrudan meta'yı bozarak test eder.
    from imgflow.io_utils import onnx_model_store as store

    original_load = store.load_model_meta

    def _load_with_bad_task_type(name, directory=None):
        meta = original_load(name, directory)
        meta["task_type"] = "anomali_tespiti"
        return meta

    monkeypatch.setattr(store, "load_model_meta", _load_with_bad_task_type)

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(OnnxDetectionError, match="tanınmıyor"):
        OnnxDetectOp().run({"image": image}, {"model_names": "bilinmeyen"})


def test_min_confidence_filters_out_low_score_detections(yolo_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image}, {"model_names": yolo_model, "min_confidence": 0.95}
    )
    assert out["measurements"] == []


def test_defect_classes_empty_by_default_no_tolerance_field(yolo_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run({"image": image}, {"model_names": yolo_model, "min_confidence": 0.25})
    assert "tolerance_ok" not in out["measurements"][0]


def test_defect_classes_marks_matching_class_as_ng(yolo_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image},
        {"model_names": yolo_model, "min_confidence": 0.25, "defect_classes": "kusur, cizik"},
    )
    m = out["measurements"][0]
    assert m["class_name"] == "kusur"
    assert m["tolerance_ok"] is False


def test_defect_classes_marks_non_matching_class_as_ok(yolo_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image},
        {"model_names": yolo_model, "min_confidence": 0.25, "defect_classes": "cizik, leke"},
    )
    m = out["measurements"][0]
    assert m["class_name"] == "kusur"
    assert m["tolerance_ok"] is True
