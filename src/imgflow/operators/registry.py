"""Operatör kayıt defteri: op_id -> operatör sınıfı eşlemesi.

Yeni bir operatör eklemek, ana uygulama kodunda değişiklik gerektirmez:
sınıf tanımlanıp bu registry'ye @registry.register ile kaydedilir yeter.
"""

from __future__ import annotations

from imgflow.core.errors import UnresolvedOperatorError
from imgflow.operators.base import Operator


class Registry:
    def __init__(self) -> None:
        self._ops: dict[str, type[Operator]] = {}

    def register(self, op_cls: type[Operator]) -> type[Operator]:
        op_id = op_cls.id
        if op_id in self._ops:
            raise ValueError(f"Operatör id çakışması: {op_id}")
        self._ops[op_id] = op_cls
        return op_cls

    def register_or_replace(self, op_cls: type[Operator]) -> type[Operator]:
        """`register` gibi ama çakışmada hata vermez, üzerine yazar — özel filtreler gibi
        çalışma zamanında (kaydet/düzenle akışı) yeniden kaydedilebilen operatörler için."""
        self._ops[op_cls.id] = op_cls
        return op_cls

    def unregister(self, op_id: str) -> None:
        """Sessizce no-op'tur (op_id kayıtlı değilse) — özel filtre silme akışı, dosya zaten
        yoksa da registry tarafını temizlemeye çalışabilir."""
        self._ops.pop(op_id, None)

    def get(self, op_id: str) -> type[Operator]:
        try:
            return self._ops[op_id]
        except KeyError:
            raise UnresolvedOperatorError(f"Operatör bulunamadı: {op_id}") from None

    def ids(self) -> list[str]:
        return sorted(self._ops)
