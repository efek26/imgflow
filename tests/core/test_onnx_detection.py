import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from imgflow.core import onnx_detection
from imgflow.core.onnx_detection import (
    Detection,
    LabeledDetection,
    OnnxDetectionError,
    _decode_boxes,
    _normalize_output_shape,
    create_session,
    find_objects_yolo,
    render_detection_overlay,
)


def _build_constant_output_model(path, output_array: np.ndarray, input_size: int) -> None:
    """Girdisini YOK SAYIP hep AYNI (bilinen) çıktıyı üreten minik bir ONNX modeli --
    onnxruntime'ı gerçekten çalıştırarak (preprocess -> session.run -> decode -> NMS ->
    ölçekleme) uçtan uca test etmemizi sağlar, gerçek bir YOLO modeli GEREKMEDEN."""
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


def test_normalize_output_shape_v5_style_already_correct_orientation():
    # (N=2, K=5+2=7) -- zaten doğru yönde, değişmemeli.
    arr = np.zeros((2, 7), dtype=np.float32)
    out = _normalize_output_shape(arr[None, ...], num_classes=2)
    assert out.shape == (2, 7)
    assert np.array_equal(out, arr)


def test_normalize_output_shape_v8_style_transposed():
    # (K=4+2=6, N=3) -- v8 biçimi, (N,K)'ye transpoze edilmeli.
    arr = np.zeros((6, 3), dtype=np.float32)
    out = _normalize_output_shape(arr[None, ...], num_classes=2)
    assert out.shape == (3, 6)


def test_normalize_output_shape_mismatched_classes_raises():
    arr = np.zeros((2, 9), dtype=np.float32)
    with pytest.raises(OnnxDetectionError):
        _normalize_output_shape(arr[None, ...], num_classes=2)


def test_decode_boxes_v5_style_multiplies_objectness_and_class_score():
    # cx,cy,w,h, obj, class0, class1
    arr = np.array([[16, 16, 8, 8, 0.9, 0.8, 0.1]], dtype=np.float32)
    cx, cy, w, h, class_ids, confidence = _decode_boxes(arr, num_classes=2, min_confidence=0.1)
    assert confidence[0] == pytest.approx(0.9 * 0.8, abs=1e-4)
    assert class_ids[0] == 0


def test_decode_boxes_v8_style_uses_class_score_directly():
    # cx,cy,w,h, class0, class1 (objectness yok)
    arr = np.array([[16, 16, 8, 8, 0.2, 0.85]], dtype=np.float32)
    cx, cy, w, h, class_ids, confidence = _decode_boxes(arr, num_classes=2, min_confidence=0.1)
    assert confidence[0] == pytest.approx(0.85, abs=1e-4)
    assert class_ids[0] == 1


def test_decode_boxes_filters_below_min_confidence():
    arr = np.array(
        [
            [16, 16, 8, 8, 0.9, 0.8, 0.1],  # confidence 0.72 -> geçer
            [16, 16, 8, 8, 0.01, 0.5, 0.5],  # confidence 0.005 -> elenir
        ],
        dtype=np.float32,
    )
    cx, cy, w, h, class_ids, confidence = _decode_boxes(arr, num_classes=2, min_confidence=0.25)
    assert confidence.size == 1


def test_find_objects_yolo_end_to_end(tmp_path):
    input_size = 32
    # v5-tarzı tek kutu: cx=16,cy=16,w=8,h=8, obj=0.9, class0=0.9, class1=0.05
    output = np.array([[[16, 16, 8, 8, 0.9, 0.9, 0.05]]], dtype=np.float32)  # (1,1,7)
    model_path = tmp_path / "fake_yolo.onnx"
    _build_constant_output_model(model_path, output, input_size)

    session = create_session(model_path)
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # h=100, w=200

    detections = find_objects_yolo(
        image,
        session,
        class_labels=["kusur", "saglam"],
        input_size=input_size,
        min_confidence=0.25,
        nms_threshold=0.45,
    )

    assert len(detections) == 1
    d = detections[0]
    assert d.class_name == "kusur"
    assert d.confidence == pytest.approx(0.81, abs=1e-3)
    # x_scale = 200/32 = 6.25, y_scale = 100/32 = 3.125
    assert d.x == pytest.approx(75.0, abs=1.0)
    assert d.y == pytest.approx(37.5, abs=1.0)
    assert d.w == pytest.approx(50.0, abs=1.0)
    assert d.h == pytest.approx(25.0, abs=1.0)


def test_find_objects_yolo_empty_class_labels_raises():
    with pytest.raises(OnnxDetectionError):
        find_objects_yolo(np.zeros((10, 10, 3), dtype=np.uint8), session=None, class_labels=[])


def test_create_session_raises_clear_error_when_onnxruntime_missing(monkeypatch):
    monkeypatch.setattr(onnx_detection, "onnxruntime", None)
    with pytest.raises(OnnxDetectionError, match="onnxruntime"):
        create_session("does_not_matter.onnx")


def test_render_detection_overlay_draws_without_crashing():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    detection = Detection(x=5, y=5, w=20, h=15, class_id=0, class_name="kusur", confidence=0.87)
    entries = [LabeledDetection(label="1", model_name="test_model", detection=detection)]

    overlay = render_detection_overlay(image, entries)
    assert overlay.shape == image.shape
    assert overlay.any()  # bir şey çizilmiş olmalı (tamamen siyah kalmamalı)
