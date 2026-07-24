"""Operatör parametrelerinin tip/aralık tanımı (UI formunun otomatik üretilmesi için)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


class ParamType(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    STRING = "string"


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: ParamType
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    label: str = ""
    help: str = ""
    readonly: bool = False
    dynamic_choices: Callable[[], list[str]] | None = None
    """STRING tipli bir parametre için, form her kurulduğunda (ör. node seçildiğinde)
    ÇAĞRILARAK güncel bir seçenek listesi üreten fonksiyon — ENUM'un aksine liste sabit
    değil, çalışma zamanında değişen bir kayıt kümesinden (ör. kaydedilmiş şekil modelleri)
    geliyorsa kullanılır. `ParamForm` bunu düzenlenebilir bir açılır kutu olarak gösterir;
    serbest metin girişi (kayıtlı olmayan bir isim yazmak) yine de mümkündür."""
    multi_select: bool = False
    """`dynamic_choices` ile birlikte kullanıldığında: tek bir seçim yerine BİRDEN FAZLA
    ismin işaretlenebildiği bir kontrol listesi popup'ı gösterilir; değer, işaretli isimlerin
    `", "` ile birleştirilmiş halidir (ör. "cıvata, somun") — depolama/serileştirme yine düz
    bir STRING'tir, sadece UI sunumu farklıdır."""

    def __post_init__(self) -> None:
        if self.type is ParamType.ENUM and not self.choices:
            raise ValueError(f"ParamSpec '{self.name}': ENUM tipi için 'choices' zorunludur.")


def defaults_for(params: list[ParamSpec]) -> dict[str, Any]:
    return {p.name: p.default for p in params}
