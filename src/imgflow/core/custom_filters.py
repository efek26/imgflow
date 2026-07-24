"""Kullanıcının doğrudan OpenCV kodu yazarak eklediği özel filtreler.

Her özel filtre, kullanıcının yazdığı bir `def apply(image): ...` fonksiyonudur; diskte
`~/.imgflow/custom_filters/<slug>.json` olarak kalıcı tutulur ve operatör registry'sine
`custom.<slug>` id'siyle DİNAMİK olarak (import-time değil, çalışma zamanında) kaydedilir.

ÖNEMLİ — bu bir sandbox DEĞİLDİR: kod doğrudan `exec()` ile çalıştırılır. Bu, kullanıcının
kendi masaüstü uygulamasında kendi yazdığı kodu çalıştırması için tasarlandı (ImageJ makrosu/
Fusion script node'u benzeri bir kullanım) — güvenilmeyen/harici kaynaklı kod çalıştırmak
için DEĞİL. Dialog'da bu açıkça kullanıcıya belirtilir.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry as registry_module

CUSTOM_FILTER_DIR = Path.home() / ".imgflow" / "custom_filters"
OP_ID_PREFIX = "custom."

DEFAULT_TEMPLATE = '''def apply(image):
    # image: BGR (renkli) ya da gri tonlamalı, np.uint8 numpy dizisi.
    # cv2 ve np zaten kullanıma hazır (import gerekmez). Bir görüntü döndürün.
    return cv2.GaussianBlur(image, (5, 5), 0)
'''


class CustomFilterError(Exception):
    """Kullanıcı kodu derlenemediğinde/çalıştırılamadığında fırlatılır. Mesajı doğrudan
    dialog'da kullanıcıya gösterilir, bu yüzden Türkçe ve anlaşılır tutulur."""


@dataclass(frozen=True)
class CustomFilterDef:
    name: str
    code: str


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise CustomFilterError("Filtre adı en az bir harf/rakam içermeli.")
    return slug


def op_id_for(name: str) -> str:
    return f"{OP_ID_PREFIX}{slugify(name)}"


def _path_for(name: str) -> Path:
    return CUSTOM_FILTER_DIR / f"{slugify(name)}.json"


def compile_apply(code: str) -> Callable[[np.ndarray], np.ndarray]:
    """Kullanıcı kodunu çalıştırıp içindeki `apply(image)` fonksiyonunu döner."""
    namespace: dict = {"cv2": cv2, "np": np}
    try:
        exec(compile(code, "<özel_filtre>", "exec"), namespace)  # noqa: S102 - kasıtlı: bkz. modül docstring'i
    except SyntaxError as exc:
        raise CustomFilterError(f"Kod hatası (satır {exc.lineno}): {exc.msg}") from exc
    except Exception as exc:
        raise CustomFilterError(f"Kod çalıştırılamadı: {exc}") from exc

    apply_fn = namespace.get("apply")
    if not callable(apply_fn):
        raise CustomFilterError("Kod bir 'def apply(image):' fonksiyonu tanımlamalı.")
    return apply_fn


def run_apply(apply_fn: Callable[[np.ndarray], np.ndarray], image: np.ndarray) -> np.ndarray:
    try:
        result = apply_fn(image)
    except Exception as exc:
        raise CustomFilterError(f"Filtre çalışırken hata: {exc}") from exc
    if not isinstance(result, np.ndarray):
        raise CustomFilterError("apply() bir numpy görüntüsü (np.ndarray) döndürmeli.")
    return result


def preview(defn: CustomFilterDef, image: np.ndarray) -> np.ndarray:
    """Test/önizleme amaçlı: kodu derler ve tek bir görüntü üzerinde çalıştırır — registry'ye
    dokunmaz, hiçbir kalıcı yan etkisi yoktur."""
    return run_apply(compile_apply(defn.code), image)


def _make_operator_class(defn: CustomFilterDef) -> type:
    apply_fn = compile_apply(defn.code)
    op_id = op_id_for(defn.name)

    class _CustomFilterOperator:
        id = op_id
        inputs = [PortSpec("image", PortType.IMAGE)]
        outputs = [PortSpec("image", PortType.IMAGE)]
        params: list = []

        def run(self, inputs: dict, params: dict) -> dict:
            return {"image": run_apply(apply_fn, inputs["image"])}

    _CustomFilterOperator.__name__ = f"CustomFilterOperator_{slugify(defn.name)}"
    return _CustomFilterOperator


def register_custom_filter(defn: CustomFilterDef) -> None:
    """Kodu derler ve registry'ye kaydeder (aynı isimde önceki bir sürüm varsa üzerine
    yazar). Derleme başarısız olursa CustomFilterError fırlatır ve registry'de HİÇBİR ŞEY
    değişmez — eski (geçerli) sürüm varsa kayıtlı kalmaya devam eder."""
    op_cls = _make_operator_class(defn)
    registry_module.register_or_replace(op_cls)


def save_custom_filter(defn: CustomFilterDef) -> None:
    """Önce derleyip registry'ye kaydeder (hata varsa burada patlar), ancak BAŞARILI
    olursa diske yazar — bozuk kod hiçbir zaman kalıcı olarak saklanmaz."""
    register_custom_filter(defn)
    CUSTOM_FILTER_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"name": defn.name, "code": defn.code}, ensure_ascii=False, indent=2)
    _path_for(defn.name).write_text(payload, encoding="utf-8")


def delete_custom_filter(name: str) -> None:
    _path_for(name).unlink(missing_ok=True)
    registry_module.unregister(op_id_for(name))


def list_custom_filters() -> list[CustomFilterDef]:
    if not CUSTOM_FILTER_DIR.is_dir():
        return []
    defs = []
    for path in sorted(CUSTOM_FILTER_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            defs.append(CustomFilterDef(name=data["name"], code=data["code"]))
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return defs


def load_all_custom_filters() -> list[tuple[CustomFilterDef, CustomFilterError]]:
    """Uygulama açılışında diskteki tüm özel filtreleri registry'ye yükler. Bozuk/derlenemeyen
    bir filtre uygulamanın açılmasını ENGELLEMEMELİ — atlanır, (tanım, hata) çiftleri
    döner ki çağıran (main_window) isterse kullanıcıyı bilgilendirsin."""
    failures = []
    for defn in list_custom_filters():
        try:
            register_custom_filter(defn)
        except CustomFilterError as exc:
            failures.append((defn, exc))
    return failures
