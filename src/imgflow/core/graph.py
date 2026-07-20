"""Node-graph veri modeli: Node, Edge, Graph (+ topological sort, ancestor/successor sorguları)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from imgflow.core.errors import CycleError

PortRef = tuple[str, str]  # (node_id, port_name)


@dataclass
class Node:
    node_id: str
    op_id: str
    params: dict[str, Any] = field(default_factory=dict)
    position: tuple[float, float] = (0.0, 0.0)


@dataclass
class Edge:
    src: PortRef
    dst: PortRef


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

    # -- mutation -----------------------------------------------------

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Node id çakışması: {node.node_id}")
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"Node bulunamadı: {node_id}")
        del self.nodes[node_id]
        self.edges = [e for e in self.edges if e.src[0] != node_id and e.dst[0] != node_id]

    def add_edge(self, edge: Edge) -> None:
        for node_id in (edge.src[0], edge.dst[0]):
            if node_id not in self.nodes:
                raise KeyError(f"Edge, var olmayan bir node'a referans veriyor: {node_id}")
        self.edges.append(edge)

    def remove_edge(self, edge: Edge) -> None:
        self.edges.remove(edge)

    # -- queries --------------------------------------------------------

    def successors(self, node_id: str) -> list[str]:
        seen: list[str] = []
        for e in self.edges:
            if e.src[0] == node_id and e.dst[0] not in seen:
                seen.append(e.dst[0])
        return seen

    def predecessors(self, node_id: str) -> list[str]:
        seen: list[str] = []
        for e in self.edges:
            if e.dst[0] == node_id and e.src[0] not in seen:
                seen.append(e.src[0])
        return seen

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.dst[0] == node_id]

    def ancestors_closure(self, node_id: str) -> set[str]:
        """node_id'nin doğrudan/dolaylı tüm atalarının kümesi (node_id hariç)."""
        result: set[str] = set()
        queue = deque(self.predecessors(node_id))
        while queue:
            nid = queue.popleft()
            if nid in result:
                continue
            result.add(nid)
            queue.extend(self.predecessors(nid))
        return result

    def topo_order(self) -> list[str]:
        """Kahn's algorithm. Döngü varsa CycleError fırlatır."""
        in_degree = {nid: 0 for nid in self.nodes}
        for e in self.edges:
            in_degree[e.dst[0]] += 1

        queue = deque(sorted(nid for nid, deg in in_degree.items() if deg == 0))
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in sorted(self.successors(nid)):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self.nodes):
            remaining = set(self.nodes) - set(order)
            raise CycleError(f"Graph'ta döngü tespit edildi, ilgili node'lar: {sorted(remaining)}")
        return order
