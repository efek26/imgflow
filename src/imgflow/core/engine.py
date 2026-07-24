"""Dirty/cache tabanlı execution engine: bir parametre değiştiğinde yalnızca etkilenen
alt-graf yeniden hesaplanır, geri kalanı cache'ten döner."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from imgflow.core.errors import OperatorRuntimeError
from imgflow.core.graph import Graph


@dataclass
class NodeResult:
    outputs: dict[str, Any] | None
    error: OperatorRuntimeError | str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _freeze(outputs: dict[str, Any]) -> dict[str, Any]:
    for value in outputs.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return outputs


class ExecutionEngine:
    def __init__(self, graph: Graph, registry: Any = None) -> None:
        self.graph = graph
        self._registry = registry
        self._cache: dict[str, NodeResult] = {}
        self._dirty: set[str] = set(graph.nodes)

    @property
    def registry(self):
        if self._registry is None:
            from imgflow.operators import registry as default_registry

            self._registry = default_registry
        return self._registry

    def mark_dirty(self, node_id: str) -> None:
        """node_id ve tüm downstream (successor) node'larını dirty işaretler, cache'lerini temizler."""
        queue: deque[str] = deque([node_id])
        while queue:
            nid = queue.popleft()
            already_dirty = nid in self._dirty
            self._dirty.add(nid)
            self._cache.pop(nid, None)
            if already_dirty:
                continue
            queue.extend(self.graph.successors(nid))

    def mark_all_dirty(self) -> None:
        self._dirty = set(self.graph.nodes)
        self._cache.clear()

    def inject_result(self, node_id: str, outputs: dict[str, Any]) -> None:
        """node_id'nin cache'ine dışarıdan üretilmiş bir sonucu (ör. canlı kameradan gelen
        kare) yazar. node_id temiz işaretlenir ama downstream'i dirty bırakılır, böylece
        normal evaluate() çağrısı zincirin geri kalanını bu yeni değerden yeniden hesaplar."""
        self._cache[node_id] = NodeResult(outputs=_freeze(dict(outputs)))
        self._dirty.discard(node_id)
        for successor in self.graph.successors(node_id):
            self.mark_dirty(successor)

    def _gather_inputs(self, node_id: str, op_cls: type) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        incoming = {e.dst[1]: e for e in self.graph.incoming_edges(node_id)}
        for port in op_cls.inputs:
            edge = incoming.get(port.name)
            if edge is None:
                if not port.optional:
                    raise ValueError(f"'{port.name}' girdisi bağlı değil (zorunlu).")
                inputs[port.name] = None
                continue
            src_id, src_port = edge.src
            src_result = self._cache[src_id]
            inputs[port.name] = src_result.outputs[src_port] if src_result.outputs else None
        return inputs

    def evaluate(self, target_id: str) -> NodeResult:
        """target_id'nin dirty ata zinciri yeniden hesaplanır; temiz node'lar cache'ten döner."""
        needed = self.graph.ancestors_closure(target_id) | {target_id}
        for nid in (n for n in self.graph.topo_order() if n in needed):
            if nid in self._cache and nid not in self._dirty:
                continue

            upstream_error = next(
                (
                    self._cache[pred].error
                    for pred in self.graph.predecessors(nid)
                    if pred in self._cache and not self._cache[pred].ok
                ),
                None,
            )
            if upstream_error is not None:
                self._cache[nid] = NodeResult(outputs=None, error=f"Üst akışta hata: {upstream_error}")
                self._dirty.discard(nid)
                continue

            node = self.graph.nodes[nid]
            try:
                op_cls = self.registry.get(node.op_id)
                inputs = self._gather_inputs(nid, op_cls)
                outputs = op_cls().run(inputs, node.params)
                self._cache[nid] = NodeResult(outputs=_freeze(outputs))
            except Exception as exc:  # noqa: BLE001 - bir operatörün patlaması tüm pipeline'ı durdurmamalı
                self._cache[nid] = NodeResult(
                    outputs=None, error=OperatorRuntimeError(nid, node.op_id, exc)
                )
            self._dirty.discard(nid)
        return self._cache[target_id]

    def trial_run(self, node_id: str, params_override: dict[str, Any]) -> NodeResult:
        """node_id'yi verilen parametrelerle bir kerelik çalıştırır; cache/dirty durumu etkilenmez.

        Ata zincirinin zaten cache'te olmasını bekler (önce evaluate() çağrılmış olmalı).
        Enum seçeneklerinin canlı önizlemesi gibi 'bu parametreyle ne olurdu' senaryoları içindir.
        """
        node = self.graph.nodes[node_id]
        try:
            op_cls = self.registry.get(node.op_id)
            inputs = self._gather_inputs(node_id, op_cls)
            outputs = op_cls().run(inputs, params_override)
        except Exception as exc:  # noqa: BLE001 - önizleme denemesi asla pipeline'ı düşürmemeli
            return NodeResult(outputs=None, error=OperatorRuntimeError(node_id, node.op_id, exc))
        return NodeResult(outputs=outputs)
