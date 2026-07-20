"""Basit doğrusal pipeline yardımcı sınıfı.

UI'ın ilk sürümü (liste + panel) tam bir görsel node-graph editörü değil; node'ları sıralı
bir zincir olarak tutar ve her node'u bir öncekine yalnızca *birincil* (operatörün
inputs/outputs listesinde ilk deklare edilen) portlar üzerinden otomatik bağlar.

LinearPipeline, sarmaladığı Graph'ın edge'lerine tam sahiplik varsayar: her append/remove/
set_order çağrısında tüm edge'ler silinip sıraya göre yeniden kurulur. roi.crop/roi.mask gibi
ikinci bir zorunlu girişi (örn. "roi") olan operatörler bu modelde yalnızca birincil
portlarından zincire bağlanır; ikincil giriş otomatik kurulmaz ve manuel bağlanmalıdır.
"""

from __future__ import annotations

from imgflow.core.graph import Edge, Graph, Node
from imgflow.operators.registry import Registry


class LinearPipeline:
    def __init__(
        self,
        registry: Registry,
        graph: Graph | None = None,
        order: list[str] | None = None,
    ) -> None:
        self.registry = registry
        self.graph = graph if graph is not None else Graph()
        self.order: list[str] = list(order) if order is not None else list(self.graph.nodes)
        self._rewire()

    def append(self, node: Node) -> None:
        self.graph.add_node(node)
        self.order.append(node.node_id)
        self._rewire()

    def remove(self, node_id: str) -> None:
        self.graph.remove_node(node_id)
        self.order.remove(node_id)
        self._rewire()

    def set_order(self, new_order: list[str]) -> None:
        if set(new_order) != set(self.order):
            raise ValueError("Yeni sıralama mevcut node kümesiyle eşleşmiyor.")
        self.order = list(new_order)
        self._rewire()

    def _rewire(self) -> None:
        self.graph.edges = []
        for prev_id, node_id in zip(self.order, self.order[1:]):
            prev_op = self.registry.get(self.graph.nodes[prev_id].op_id)
            op = self.registry.get(self.graph.nodes[node_id].op_id)
            if not prev_op.outputs or not op.inputs:
                continue
            src_port = prev_op.outputs[0].name
            dst_port = op.inputs[0].name
            self.graph.add_edge(Edge((prev_id, src_port), (node_id, dst_port)))
