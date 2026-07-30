"""Bir klasördeki tüm görüntülere pipeline'ı uygulayıp ölçüm sonuçlarını CSV'e yazar.

Her görüntüde yalnızca kaynak node ve onun downstream'i dirty işaretlenir; pipeline'ın
geri kalanı (parametre bazlı) değişmediği sürece motor mimarisi burada da geçerli kalır.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from imgflow.core.engine import ExecutionEngine
from imgflow.core.graph import Graph

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _find_source_node(graph: Graph, source_op_id: str) -> str:
    matches = [n.node_id for n in graph.nodes.values() if n.op_id == source_op_id]
    if not matches:
        raise ValueError(f"Graph içinde '{source_op_id}' operatörüne sahip bir node bulunamadı.")
    if len(matches) > 1:
        raise ValueError(
            f"Graph birden fazla '{source_op_id}' node'u içeriyor, "
            "hangisinin kullanılacağı belirsiz; source_node_id parametresini belirtin."
        )
    return matches[0]


def run_batch(
    graph: Graph,
    target_node_id: str,
    input_folder: str | Path,
    output_csv: str | Path,
    *,
    source_node_id: str | None = None,
    source_op_id: str = "io.image_source",
    registry: Any = None,
    progress_callback: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """`progress_callback(işlenen, toplam)` her resimden SONRA çağrılır; `should_cancel()`
    her resimden ÖNCE kontrol edilir, `True` dönerse döngü erken `break` eder — o ana kadar
    biriken `rows` yine CSV'ye yazılıp döner (mevcut "bir resmin hatası tüm batch'i
    durdurmaz" felsefesiyle tutarlı KISMİ tamamlanma). İkisi de varsayılan `None` olduğu
    için mevcut çağrılar davranış değişikliği olmadan çalışmaya devam eder."""
    source_node_id = source_node_id or _find_source_node(graph, source_op_id)
    engine = ExecutionEngine(graph, registry=registry)

    images = iter_images(input_folder)
    total = len(images)
    rows: list[dict[str, Any]] = []
    for index, image_path in enumerate(images, start=1):
        if should_cancel is not None and should_cancel():
            break
        graph.nodes[source_node_id].params["path"] = str(image_path)
        engine.mark_dirty(source_node_id)
        result = engine.evaluate(target_node_id)

        if not result.ok:
            rows.append({"image": image_path.name, "error": str(result.error)})
        else:
            measurements = (result.outputs or {}).get("measurements")
            if not measurements:
                rows.append({"image": image_path.name})
            else:
                rows.extend({"image": image_path.name, **measurement} for measurement in measurements)

        if progress_callback is not None:
            progress_callback(index, total)

    _write_csv(output_csv, rows)
    return rows


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["image"]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
