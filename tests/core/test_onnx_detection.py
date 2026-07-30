import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from imgflow.core import onnx_detection
from imgflow.core.onnx_detection import (
    ClassificationResult,
    Detection,
    LabeledClassification,
    LabeledDetection,
    OnnxDetectionError,
    _decode_boxes,
    _normalize_output_shape,
    _normalize_segmentation_shape,
    classify_image,
    create_session,
    find_objects_yolo,
    inspect_onnx_model,
    render_classification_overlay,
    render_detection_overlay,
    render_segmentation_overlay,
    segment_image,
)


def _build_constant_output_model(
    path,
    output_array: np.ndarray,
    input_size: int,
    metadata: dict[str, str] | None = None,
    dynamic_input: bool = False,
) -> None:
    """Girdisini YOK SAYIP hep AYNI (bilinen) çıktıyı üreten minik bir ONNX modeli --
    onnxruntime'ı gerçekten çalıştırarak (preprocess -> session.run -> decode -> NMS ->
    ölçekleme) uçtan uca test etmemizi sağlar, gerçek bir YOLO modeli GEREKMEDEN.

    `metadata` verilirse `model.metadata_props`'a yazılır (Ultralytics'in `"names"` sınıf
    listesini gömdüğü mekanizmanın AYNISI, `inspect_onnx_model` testleri için). `dynamic_input`
    True ise girdi şeklinin H/W boyutları sembolik (string) yapılır -- `Constant` düğümü
    girdiyi zaten hiç KULLANMADIĞI için model yine de geçerli/çalıştırılabilir kalır, sadece
    `inspect_onnx_model`'in "boyut güvenilir değil" dalını test etmeye yarar."""
    input_shape = [1, 3, "height", "width"] if dynamic_input else [1, 3, input_size, input_size]
    input_tensor = helper.make_tensor_value_info("images", TensorProto.FLOAT, input_shape)
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
    if metadata:
        for key, value in metadata.items():
            model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
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


def test_render_detection_overlay_default_green_when_no_tolerance():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    detection = Detection(x=5, y=5, w=20, h=15, class_id=0, class_name="kusur", confidence=0.87)
    entries = [LabeledDetection(label="1", model_name="test_model", detection=detection)]

    overlay = render_detection_overlay(image, entries)
    assert tuple(overlay[5, 5]) == (0, 255, 0)  # BGR yeşil, kutunun sol üst köşesi


def test_render_detection_overlay_colors_ok_ng_by_tolerance():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    ok_det = Detection(x=5, y=5, w=20, h=15, class_id=0, class_name="saglam", confidence=0.9)
    ng_det = Detection(x=5, y=5, w=20, h=15, class_id=1, class_name="kusur", confidence=0.9)

    ok_overlay = render_detection_overlay(
        image, [LabeledDetection(label="1", model_name="m", detection=ok_det, tolerance_ok=True)]
    )
    ng_overlay = render_detection_overlay(
        image, [LabeledDetection(label="1", model_name="m", detection=ng_det, tolerance_ok=False)]
    )

    assert tuple(ok_overlay[5, 5]) == (0, 255, 0)  # OK -> yeşil
    assert tuple(ng_overlay[5, 5]) == (0, 0, 255)  # NG -> kırmızı


def test_inspect_onnx_model_reads_names_dict_repr_and_input_size(tmp_path):
    # Ultralytics'in yazdığı biçim: JSON DEĞİL, Python dict-repr (tek tırnak, int anahtar).
    model_path = tmp_path / "with_metadata.onnx"
    _build_constant_output_model(
        model_path,
        np.zeros((1, 1, 7), dtype=np.float32),
        input_size=320,
        metadata={"names": "{0: 'kusur', 1: 'saglam'}"},
    )

    info = inspect_onnx_model(model_path)

    assert info == {"class_labels": ["kusur", "saglam"], "input_size": 320}


def test_inspect_onnx_model_reads_names_as_valid_json_too(tmp_path):
    model_path = tmp_path / "json_metadata.onnx"
    _build_constant_output_model(
        model_path,
        np.zeros((1, 1, 6), dtype=np.float32),
        input_size=640,
        metadata={"names": '["a", "b"]'},
    )

    info = inspect_onnx_model(model_path)

    assert info["class_labels"] == ["a", "b"]
    assert info["input_size"] == 640


def test_inspect_onnx_model_no_metadata_still_reads_input_size(tmp_path):
    model_path = tmp_path / "no_metadata.onnx"
    _build_constant_output_model(model_path, np.zeros((1, 1, 6), dtype=np.float32), input_size=416)

    info = inspect_onnx_model(model_path)

    assert info["class_labels"] is None
    assert info["input_size"] == 416


def test_inspect_onnx_model_dynamic_input_returns_none_size(tmp_path):
    model_path = tmp_path / "dynamic.onnx"
    _build_constant_output_model(
        model_path, np.zeros((1, 1, 6), dtype=np.float32), input_size=640, dynamic_input=True
    )

    info = inspect_onnx_model(model_path)

    assert info["input_size"] is None


def test_inspect_onnx_model_corrupt_file_returns_none_none(tmp_path):
    model_path = tmp_path / "corrupt.onnx"
    model_path.write_bytes(b"not-a-real-onnx-file")

    assert inspect_onnx_model(model_path) == {"class_labels": None, "input_size": None}


def test_inspect_onnx_model_without_onnxruntime_returns_none_none(monkeypatch, tmp_path):
    monkeypatch.setattr(onnx_detection, "onnxruntime", None)
    model_path = tmp_path / "whatever.onnx"
    model_path.write_bytes(b"irrelevant")

    assert inspect_onnx_model(model_path) == {"class_labels": None, "input_size": None}


def test_classify_image_picks_highest_softmax_class(tmp_path):
    input_size = 16
    output = np.array([[2.0, 0.1, 0.1]], dtype=np.float32)  # class0 açık ara kazanır
    model_path = tmp_path / "fake_cls.onnx"
    _build_constant_output_model(model_path, output, input_size)
    session = create_session(model_path)

    result = classify_image(
        np.zeros((40, 40, 3), dtype=np.uint8), session, class_labels=["a", "b", "c"], input_size=input_size
    )

    assert result.class_id == 0
    assert result.class_name == "a"
    assert 0.0 < result.confidence <= 1.0


def test_classify_image_empty_class_labels_raises():
    with pytest.raises(OnnxDetectionError):
        classify_image(np.zeros((10, 10, 3), dtype=np.uint8), session=None, class_labels=[])


def test_classify_image_mismatched_output_size_raises(tmp_path):
    input_size = 16
    output = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # 3 sınıf çıktısı
    model_path = tmp_path / "fake_cls_mismatch.onnx"
    _build_constant_output_model(model_path, output, input_size)
    session = create_session(model_path)

    with pytest.raises(OnnxDetectionError):
        classify_image(
            np.zeros((10, 10, 3), dtype=np.uint8), session, class_labels=["a", "b"], input_size=input_size
        )


def test_render_classification_overlay_no_tolerance_uses_default_color():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    result = ClassificationResult(class_id=0, class_name="ok", confidence=0.95)
    entries = [LabeledClassification(label="1", model_name="m", result=result)]

    overlay = render_classification_overlay(image, entries)

    assert overlay.shape == image.shape
    assert overlay.any()


def test_render_classification_overlay_colors_ok_ng_by_tolerance():
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    ok_result = ClassificationResult(class_id=0, class_name="saglam", confidence=0.9)
    ng_result = ClassificationResult(class_id=1, class_name="kusur", confidence=0.9)

    ok_overlay = render_classification_overlay(
        image, [LabeledClassification(label="1", model_name="m", result=ok_result, tolerance_ok=True)]
    )
    ng_overlay = render_classification_overlay(
        image, [LabeledClassification(label="1", model_name="m", result=ng_result, tolerance_ok=False)]
    )

    assert ok_overlay.any()
    assert ng_overlay.any()
    assert not np.array_equal(ok_overlay, ng_overlay)


def test_normalize_segmentation_shape_chw_already_correct():
    arr = np.zeros((3, 4, 5), dtype=np.float32)  # (C=3, H=4, W=5)
    out = _normalize_segmentation_shape(arr[None, ...], num_classes=3)
    assert out.shape == (3, 4, 5)


def test_normalize_segmentation_shape_hwc_transposed():
    arr = np.zeros((4, 5, 3), dtype=np.float32)  # (H=4, W=5, C=3)
    out = _normalize_segmentation_shape(arr[None, ...], num_classes=3)
    assert out.shape == (3, 4, 5)


def test_normalize_segmentation_shape_mismatched_classes_raises():
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    with pytest.raises(OnnxDetectionError):
        _normalize_segmentation_shape(arr[None, ...], num_classes=7)


def test_segment_image_end_to_end_resizes_class_map_to_original_size(tmp_path):
    input_size = 8
    h, w = 3, 3  # modelin küçük çözünürlüğü
    output = np.zeros((1, 3, h, w), dtype=np.float32)  # (1, C=3, h, w)
    output[0, 2, :, :] = 5.0  # her piksel için class_id=2 en yüksek skor
    model_path = tmp_path / "fake_seg.onnx"
    _build_constant_output_model(model_path, output, input_size)
    session = create_session(model_path)

    image = np.zeros((50, 100, 3), dtype=np.uint8)  # h=50, w=100 (girişten farklı boyut)
    result = segment_image(image, session, class_labels=["bg", "a", "b"], input_size=input_size)

    assert result.class_map.shape == (50, 100)
    assert np.all(result.class_map == 2)
    assert result.class_areas == {2: 50 * 100}


def test_segment_image_empty_class_labels_raises():
    with pytest.raises(OnnxDetectionError):
        segment_image(np.zeros((10, 10, 3), dtype=np.uint8), session=None, class_labels=[])


def test_render_segmentation_overlay_colors_non_background_classes():
    # Yeterince büyük bir görüntü + legend metninden UZAK bir alt bölge kullan ki lejant
    # yazısı (sol üstte) "arka plan hiç renklendirilmedi" kontrolünü bozmasın.
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    class_map = np.zeros((200, 300), dtype=np.int32)
    class_map[100:, 150:] = 1  # sağ-alt çeyrek class_id=1

    overlay = render_segmentation_overlay(image, class_map, class_labels=["bg", "kusur"])

    assert overlay.shape == image.shape
    assert overlay[100:, :150].sum() == 0  # arka plan (class_id=0) hiç renklendirilmedi
    assert overlay[100:, 150:].any()  # class_id=1 bölgesi renklendirildi


def test_render_segmentation_overlay_defect_class_uses_fail_color_in_legend():
    image = np.zeros((100, 150, 3), dtype=np.uint8)
    class_map = np.zeros((100, 150), dtype=np.int32)
    class_map[:, :] = 1

    overlay = render_segmentation_overlay(
        image, class_map, class_labels=["bg", "kusur"], defect_class_ids={1}
    )

    # Lejant metni kırmızı (_FAIL_COLOR = (0,0,255)) ile çizilir -- görüntüde saf kırmızı
    # piksel olup olmadığını kontrol ederek doğrula.
    assert np.any(np.all(overlay == (0, 0, 255), axis=-1))
