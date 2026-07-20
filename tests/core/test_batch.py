import csv

import cv2
import numpy as np

from imgflow.core.batch import run_batch
from imgflow.core.graph import Edge, Graph, Node
from imgflow.operators import registry


def _make_pipeline() -> Graph:
    g = Graph()
    g.add_node(Node("src", "io.image_source"))
    g.add_node(Node("gray", "color.convert", params={"mode": "BGR2GRAY"}))
    g.add_node(
        Node("th", "segment.threshold", params={"value": 100, "max_value": 255, "mode": "BINARY"})
    )
    g.add_node(Node("cc", "segment.connected_components", params={"connectivity": "8"}))
    g.add_node(Node("props", "analysis.region_props", params={"min_area": 0}))
    g.add_edge(Edge(("src", "image"), ("gray", "image")))
    g.add_edge(Edge(("gray", "image"), ("th", "image")))
    g.add_edge(Edge(("th", "image"), ("cc", "image")))
    g.add_edge(Edge(("cc", "labels"), ("props", "labels")))
    return g


def test_run_batch_writes_csv_row_per_region(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()

    img1 = np.zeros((20, 20, 3), dtype=np.uint8)
    img1[5:15, 5:15] = 255  # tek kare -> 1 bölge
    cv2.imwrite(str(input_dir / "a.png"), img1)

    img2 = np.zeros((20, 20, 3), dtype=np.uint8)
    img2[2:6, 2:6] = 255
    img2[12:18, 12:18] = 255  # iki ayrı kare -> 2 bölge
    cv2.imwrite(str(input_dir / "b.png"), img2)

    output_csv = tmp_path / "out.csv"
    rows = run_batch(_make_pipeline(), "props", input_dir, output_csv, registry=registry)

    assert len([r for r in rows if r["image"] == "a.png"]) == 1
    assert len([r for r in rows if r["image"] == "b.png"]) == 2

    with output_csv.open(encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(rows)
    assert "area" in csv_rows[0]


def test_run_batch_auto_detects_source_node(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(input_dir / "only.png"), img)

    rows = run_batch(_make_pipeline(), "props", input_dir, tmp_path / "out.csv", registry=registry)

    assert rows and all(r["image"] == "only.png" for r in rows)


def test_run_batch_records_error_for_unreadable_image(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "corrupt.png").write_bytes(b"not an image")

    rows = run_batch(_make_pipeline(), "props", input_dir, tmp_path / "out.csv", registry=registry)

    assert len(rows) == 1
    assert rows[0]["image"] == "corrupt.png"
    assert "error" in rows[0]
