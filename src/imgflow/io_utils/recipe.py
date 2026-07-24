"""Pipeline reçetelerinin JSON olarak kaydedilmesi/yüklenmesi."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imgflow.core.errors import RecipeVersionError
from imgflow.core.graph import Edge, Graph, Node

RECIPE_SCHEMA_VERSION = 2


def graph_to_dict(graph: Graph) -> dict[str, Any]:
    return {
        "schema_version": RECIPE_SCHEMA_VERSION,
        "nodes": [
            {
                "id": node.node_id,
                "op_id": node.op_id,
                "params": node.params,
                "position": list(node.position),
            }
            for node in graph.nodes.values()
        ],
        "edges": [{"src": list(edge.src), "dst": list(edge.dst)} for edge in graph.edges],
        "calibration_profile": graph.calibration_profile,
        "calibration_height_mm": graph.calibration_height_mm,
    }


def graph_from_dict(data: dict[str, Any]) -> Graph:
    schema_version = data.get("schema_version")
    if schema_version is None or schema_version > RECIPE_SCHEMA_VERSION:
        raise RecipeVersionError(
            f"Reçete şema sürümü desteklenmiyor: {schema_version} "
            f"(bu uygulama en fazla {RECIPE_SCHEMA_VERSION} destekliyor)"
        )

    graph = Graph()
    for raw_node in data.get("nodes", []):
        position = raw_node.get("position", [0.0, 0.0])
        graph.add_node(
            Node(
                node_id=raw_node["id"],
                op_id=raw_node["op_id"],
                params=dict(raw_node.get("params", {})),
                position=(float(position[0]), float(position[1])),
            )
        )
    for raw_edge in data.get("edges", []):
        src, dst = raw_edge["src"], raw_edge["dst"]
        graph.add_edge(Edge(src=(src[0], src[1]), dst=(dst[0], dst[1])))
    graph.calibration_profile = data.get("calibration_profile")
    graph.calibration_height_mm = data.get("calibration_height_mm")
    return graph


def save_recipe(path: str | Path, graph: Graph) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph_to_dict(graph), ensure_ascii=False, indent=2), encoding="utf-8")


def load_recipe(path: str | Path) -> Graph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return graph_from_dict(data)
