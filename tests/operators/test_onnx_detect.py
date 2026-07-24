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


def test_non_yolo_task_type_raises_clear_error(tmp_path):
    onnx_path = tmp_path / "placeholder.onnx"
    onnx_path.write_bytes(b"not-a-real-model")
    onnx_model_store.save_model("siniflandirici", onnx_path, "classification", 224, ["ok", "ng"])

    image = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(OnnxDetectionError, match="desteklenmiyor"):
        OnnxDetectOp().run({"image": image}, {"model_names": "siniflandirici"})


def test_min_confidence_filters_out_low_score_detections(yolo_model):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = OnnxDetectOp().run(
        {"image": image}, {"model_names": yolo_model, "min_confidence": 0.95}
    )
    assert out["measurements"] == []
