"""Basit doğrusal pipeline yardımcı sınıfı.

UI'ın ilk sürümü (liste + panel) tam bir görsel node-graph editörü değil; node'ları sıralı
bir zincir olarak tutar ve her node'u bir öncekine yalnızca *birincil* (operatörün
inputs/outputs listesinde ilk deklare edilen) portlar üzerinden otomatik bağlar.

LinearPipeline, sarmaladığı Graph'ın edge'lerine tam sahiplik varsayar: her append/remove/
set_order çağrısında tüm edge'ler silinip sıraya göre yeniden kurulur. İkinci bir zorunlu
girişi olan operatörler bu modelde desteklenmez (böyle bir operatör eklenirse ikincil girişi
otomatik kurulmaz); registry'deki tüm builtin operatörler bilerek tek zorunlu girişli olacak
şekilde tasarlanmıştır.

Birincil portların PortType'ları uyuşmuyorsa (ör. bir operatörün ROI çıktısı bir sonraki
adımın IMAGE girdisine) bağlantı hiç kurulmaz; o adımın girdisi boş kalır ve engine net bir
"girdi bağlı değil" hatası verir. Amaç, yanlış tipte veriyi operatöre sessizce geçirip
anlamsız/çökmeye açık bir çalıştırma yerine hatayı erken ve anlaşılır göstermektir.
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

    def load(self, new_graph: Graph, order: list[str] | None = None) -> None:
        """Pipeline'ın tamamını new_graph'ın node'larıyla değiştirir (Graph kimliği korunur)."""
        self.graph.nodes = dict(new_graph.nodes)
        self.order = list(order) if order is not None else list(new_graph.nodes)
        self._rewire()

    def generate_node_id(self, op_id: str) -> str:
        base = op_id.split(".")[-1]
        existing = set(self.graph.nodes)
        if base not in existing:
            return base
        counter = 2
        while f"{base}_{counter}" in existing:
            counter += 1
        return f"{base}_{counter}"

    def _rewire(self) -> None:
        self.graph.edges = []
        for prev_id, node_id in zip(self.order, self.order[1:]):
            prev_op = self.registry.get(self.graph.nodes[prev_id].op_id)
            op = self.registry.get(self.graph.nodes[node_id].op_id)
            if not prev_op.outputs or not op.inputs:
                continue
            src_spec = prev_op.outputs[0]
            dst_spec = op.inputs[0]
            if src_spec.type is not dst_spec.type:
                continue
            self.graph.add_edge(Edge((prev_id, src_spec.name), (node_id, dst_spec.name)))
