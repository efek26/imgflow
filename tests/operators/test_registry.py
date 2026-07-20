import pytest

from imgflow.core.errors import UnresolvedOperatorError
from imgflow.operators.registry import Registry


class _DummyOp:
    id = "dummy.op"
    inputs: list = []
    outputs: list = []
    params: list = []

    def run(self, inputs, params):
        return {}


class _OtherOp:
    id = "aaa.op"
    inputs: list = []
    outputs: list = []
    params: list = []

    def run(self, inputs, params):
        return {}


def test_register_and_get():
    reg = Registry()
    reg.register(_DummyOp)
    assert reg.get("dummy.op") is _DummyOp


def test_get_missing_raises():
    reg = Registry()
    with pytest.raises(UnresolvedOperatorError):
        reg.get("missing.op")


def test_register_duplicate_raises():
    reg = Registry()
    reg.register(_DummyOp)
    with pytest.raises(ValueError):
        reg.register(_DummyOp)


def test_ids_sorted():
    reg = Registry()
    reg.register(_DummyOp)
    reg.register(_OtherOp)
    assert reg.ids() == ["aaa.op", "dummy.op"]
