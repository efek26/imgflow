import csv

import cv2
import numpy as np

from imgflow.core.batch import run_batch
from imgflow.core.graph import Edge, Graph, Node
from imgflow.operators import registry


def _make_pipeline() -> Graph:
    g = Graph()
    g.add_node(Node("src", "io.image_source"))
    g.add_node(Node("gray", "color.grayscale", params={}))
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


def _make_input_dir(tmp_path, count: int):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    for i in range(count):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(input_dir / f"img_{i}.png"), img)
    return input_dir


def test_run_batch_reports_progress_per_image(tmp_path):
    input_dir = _make_input_dir(tmp_path, 3)
    progress_calls = []

    run_batch(
        _make_pipeline(),
        "props",
        input_dir,
        tmp_path / "out.csv",
        registry=registry,
        progress_callback=lambda done, total: progress_calls.append((done, total)),
    )

    assert progress_calls == [(1, 3), (2, 3), (3, 3)]


def test_run_batch_stops_early_when_should_cancel_returns_true(tmp_path):
    """İptal isteği geldiğinde döngü erken durmalı ama o ana kadarki sonuçlar (KISMİ
    tamamlanma) yine de döndürülüp CSV'ye yazılmalı — bir resmin hatası batch'i durdurmadığı
    gibi, kullanıcının iptali de mevcut ilerlemeyi kaybettirmemeli."""
    input_dir = _make_input_dir(tmp_path, 5)
    cancel_after = 2
    calls = {"count": 0}

    def should_cancel():
        return calls["count"] >= cancel_after

    def progress_callback(done, total):
        calls["count"] = done

    rows = run_batch(
        _make_pipeline(),
        "props",
        input_dir,
        tmp_path / "out.csv",
        registry=registry,
        progress_callback=progress_callback,
        should_cancel=should_cancel,
    )

    assert len(rows) == cancel_after
