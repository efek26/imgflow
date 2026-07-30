import pytest

from imgflow.core.engine import ExecutionEngine
from imgflow.core.errors import UnresolvedOperatorError
from imgflow.core.graph import Edge, Graph, Node
from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType


def make_counting_op(op_id: str, raises: bool = False):
    """Her run() çağrısında sayaç artıran, testler için sahte bir operatör sınıfı üretir."""

    class _CountingOp:
        id = op_id
        inputs = [PortSpec("in", PortType.SCALAR, optional=True)]
        outputs = [PortSpec("out", PortType.SCALAR)]
        params = [ParamSpec("add", ParamType.INT, default=1)]
        call_count = 0

        def run(self, inputs, params):
            type(self).call_count += 1
            if raises:
                raise ValueError("kasıtlı hata")
            base = inputs.get("in") or 0
            return {"out": base + params.get("add", 1)}

    _CountingOp.__name__ = f"CountingOp_{op_id}"
    return _CountingOp


class FakeRegistry:
    def __init__(self, op_classes: dict[str, type]):
        self._ops = op_classes

    def get(self, op_id: str) -> type:
        try:
            return self._ops[op_id]
        except KeyError:
            raise UnresolvedOperatorError(f"Operatör bulunamadı: {op_id}") from None


def linear_setup():
    op_a, op_b, op_c = make_counting_op("a"), make_counting_op("b"), make_counting_op("c")
    g = Graph()
    g.add_node(Node("na", "a", params={"add": 1}))
    g.add_node(Node("nb", "b", params={"add": 10}))
    g.add_node(Node("nc", "c", params={"add": 100}))
    g.add_edge(Edge(("na", "out"), ("nb", "in")))
    g.add_edge(Edge(("nb", "out"), ("nc", "in")))
    registry = FakeRegistry({"a": op_a, "b": op_b, "c": op_c})
    engine = ExecutionEngine(g, registry=registry)
    return engine, (op_a, op_b, op_c)


def test_evaluate_runs_full_ancestor_chain():
    engine, (op_a, op_b, op_c) = linear_setup()
    result = engine.evaluate("nc")
    assert result.ok
    assert result.outputs["out"] == 111  # 0+1 -> 1+10 -> 11+100
    assert op_a.call_count == 1
    assert op_b.call_count == 1
    assert op_c.call_count == 1


def test_evaluate_uses_cache_when_nothing_dirty():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")
    engine.evaluate("nc")
    assert op_a.call_count == 1
    assert op_b.call_count == 1
    assert op_c.call_count == 1


def test_mark_dirty_only_recomputes_downstream():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")

    engine.mark_dirty("nb")
    engine.evaluate("nc")

    assert op_a.call_count == 1  # ata, değişmedi -> cache'ten döndü
    assert op_b.call_count == 2  # dirty edildi -> yeniden çalıştı
    assert op_c.call_count == 2  # downstream olduğu için yeniden çalıştı


def test_mark_dirty_on_leaf_does_not_affect_ancestors():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")

    engine.mark_dirty("nc")
    engine.evaluate("nc")

    assert op_a.call_count == 1
    assert op_b.call_count == 1
    assert op_c.call_count == 2


def test_operator_exception_becomes_node_result_error():
    op_a = make_counting_op("a", raises=True)
    g = Graph()
    g.add_node(Node("na", "a"))
    engine = ExecutionEngine(g, registry=FakeRegistry({"a": op_a}))

    result = engine.evaluate("na")
    assert not result.ok
    assert result.outputs is None


def test_upstream_error_propagates_without_running_downstream():
    op_a = make_counting_op("a", raises=True)
    op_b = make_counting_op("b")
    g = Graph()
    g.add_node(Node("na", "a"))
    g.add_node(Node("nb", "b"))
    g.add_edge(Edge(("na", "out"), ("nb", "in")))
    engine = ExecutionEngine(g, registry=FakeRegistry({"a": op_a, "b": op_b}))

    result = engine.evaluate("nb")
    assert not result.ok
    assert op_b.call_count == 0  # üst akışta hata varken hiç çalıştırılmamalı


def test_missing_required_input_becomes_error():
    op_a = make_counting_op("a")
    op_a.inputs = [PortSpec("in", PortType.SCALAR, optional=False)]
    g = Graph()
    g.add_node(Node("na", "a"))
    engine = ExecutionEngine(g, registry=FakeRegistry({"a": op_a}))

    result = engine.evaluate("na")
    assert not result.ok


def test_unresolved_operator_id_becomes_error():
    g = Graph()
    g.add_node(Node("na", "missing.op"))
    engine = ExecutionEngine(g, registry=FakeRegistry({}))

    result = engine.evaluate("na")
    assert not result.ok


def test_trial_run_uses_override_params_without_touching_cache():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")

    result = engine.trial_run("nb", {"add": 999})

    assert result.ok
    assert result.outputs["out"] == 1000  # na çıktısı (1) + 999
    # cache/dirty durumu bozulmadı: yeniden evaluate aynı (orijinal) sonucu vermeli, nb tekrar çalışmamalı
    before = op_b.call_count
    result2 = engine.evaluate("nc")
    assert result2.outputs["out"] == 111
    assert op_b.call_count == before


def test_trial_run_without_cached_ancestors_becomes_error():
    engine, (op_a, op_b, op_c) = linear_setup()
    result = engine.trial_run("nb", {"add": 5})
    assert not result.ok


def test_inject_result_sets_cache_and_dirties_downstream_only():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")

    engine.inject_result("na", {"out": 500})
    result = engine.evaluate("nc")

    assert result.ok
    assert result.outputs["out"] == 500 + 10 + 100  # nb ve nc yeni na değeriyle yeniden hesaplandı
    assert op_a.call_count == 1  # na'nın kendisi yeniden ÇALIŞTIRILMADI, sadece cache'i enjekte edildi
    assert op_b.call_count == 2
    assert op_c.call_count == 2


def test_inject_result_on_leaf_node_does_not_dirty_ancestors():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")

    engine.inject_result("nc", {"out": 999})
    result = engine.evaluate("nc")

    assert result.outputs["out"] == 999
    assert op_a.call_count == 1
    assert op_b.call_count == 1
    assert op_c.call_count == 1  # leaf enjekte edildiği için yeniden çalışmadı, doğrudan cache'ten döndü


def test_evaluate_records_duration_only_for_actually_executed_nodes():
    """Gerçek kullanıcı isteği: "her işlemin sonucunun süresi sonuçlar kısmında yazmalı" --
    `durations` SADECE gerçekten çalıştırılan node'lar için dolmalı; cache-hit'te (bkz.
    `test_evaluate_uses_cache_when_nothing_dirty`) süre GÜNCELLENMEMELİ (eski değer duruyor
    olabilir ama yeniden YAZILMAMALI -- burada basitçe anahtarın varlığı ve pozitifliği
    kontrol ediliyor)."""
    engine, (op_a, op_b, op_c) = linear_setup()

    engine.evaluate("nc")

    assert set(engine.durations) == {"na", "nb", "nc"}
    assert all(d >= 0.0 for d in engine.durations.values())


def test_evaluate_does_not_record_duration_for_failed_or_skipped_nodes():
    op_a = make_counting_op("a", raises=True)
    op_b = make_counting_op("b")
    g = Graph()
    g.add_node(Node("na", "a"))
    g.add_node(Node("nb", "b"))
    g.add_edge(Edge(("na", "out"), ("nb", "in")))
    engine = ExecutionEngine(g, registry=FakeRegistry({"a": op_a, "b": op_b}))

    engine.evaluate("nb")

    # na'nın `run()`'ı fırladı (istisna) -> süre hiç yazılmadı (sadece BAŞARILI çalıştırma
    # süreyi kaydeder, bkz. `ExecutionEngine.evaluate`). nb ise üst akış hatası yüzünden
    # HİÇ çalıştırılmadı (`op_b.call_count == 0`) -> onun da süresi YOK.
    assert "na" not in engine.durations
    assert "nb" not in engine.durations


def test_evaluate_does_not_record_duration_for_cache_hit_skip():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")
    first_na_duration = engine.durations["na"]

    engine.mark_dirty("nb")  # na hâlâ temiz -> cache-hit ile atlanmalı
    engine.evaluate("nc")

    assert engine.durations["na"] == first_na_duration  # yeniden yazılmadı
    assert op_a.call_count == 1  # gerçekten de yeniden çalışmadı


def test_durations_property_returns_a_copy():
    engine, (op_a, op_b, op_c) = linear_setup()
    engine.evaluate("nc")

    durations = engine.durations
    durations["na"] = -1.0

    assert engine.durations["na"] != -1.0
