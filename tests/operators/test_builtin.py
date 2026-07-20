import cv2
import numpy as np
import pytest

from imgflow.core.engine import ExecutionEngine
from imgflow.core.graph import Edge, Graph, Node
from imgflow.operators import registry
from imgflow.operators.builtin.color_convert import ColorConvertOp
from imgflow.operators.builtin.image_source import ImageSourceOp
from imgflow.operators.builtin.threshold import ThresholdOp


def test_builtins_registered_by_default():
    for op_id in ("io.image_source", "color.convert", "segment.threshold"):
        assert registry.get(op_id) is not None


def test_image_source_loads_image(tmp_path):
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 2] = 255
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)

    out = ImageSourceOp().run({}, {"path": str(path)})
    assert out["image"].shape == (4, 4, 3)


def test_image_source_missing_path_raises():
    with pytest.raises(ValueError):
        ImageSourceOp().run({}, {"path": ""})


def test_image_source_nonexistent_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageSourceOp().run({}, {"path": str(tmp_path / "yok.png")})


def test_color_convert_bgr_to_gray():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    out = ColorConvertOp().run({"image": img}, {"mode": "BGR2GRAY"})
    assert out["image"].shape == (4, 4)


def test_threshold_binary():
    img = np.array([[0, 200], [50, 255]], dtype=np.uint8)
    out = ThresholdOp().run({"image": img}, {"value": 127, "max_value": 255, "mode": "BINARY"})
    assert out["image"].tolist() == [[0, 255], [0, 255]]


def test_pipeline_end_to_end_with_real_registry(tmp_path):
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), img)

    g = Graph()
    g.add_node(Node("src", "io.image_source", params={"path": str(path)}))
    g.add_node(Node("gray", "color.convert", params={"mode": "BGR2GRAY"}))
    g.add_node(
        Node(
            "th",
            "segment.threshold",
            params={"value": 100, "max_value": 255, "mode": "BINARY"},
        )
    )
    g.add_edge(Edge(("src", "image"), ("gray", "image")))
    g.add_edge(Edge(("gray", "image"), ("th", "image")))

    engine = ExecutionEngine(g, registry=registry)
    result = engine.evaluate("th")

    assert result.ok
    assert result.outputs["image"].max() == 255
