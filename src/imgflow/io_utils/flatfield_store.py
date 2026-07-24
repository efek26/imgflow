"""Aydınlatma/vignetting düzeltme için isimle kaydedilen referans (flat-field) görüntüler.

`shape_model_store.py`/`calibration_store.py`'deki isimli-profil deseninin aynısı:
kullanıcı Araçlar > Aydınlatma Referansı Kaydet... ile BOŞ/düz aydınlatılmış bir kare
(ör. üzerinde ürün olmayan boş bant) yükleyip isimle kaydeder; pipeline'daki
`correction.flat_field` operatörü bu referansı sadece isimle (`reference_name`) kullanır.

`~/.imgflow/flatfield` altında saklanır (`custom_filters.py`/`capture_store.py`/
`shape_model_store.py` ile aynı gerekçe). **Testler `FLATFIELD_DIR`'ı `monkeypatch` ile
`tmp_path`'e yönlendirmeli** — aksi halde gerçek ev dizinine dosya sızar. `directory`
parametresi ÖNTANIMLI olarak `None`dır ve YALNIZCA fonksiyon GÖVDESİNDE `FLATFIELD_DIR`'a
düşer (bkz. `shape_model_store.py`'deki aynı gotcha notu) — aksi halde monkeypatch,
`directory` argümanı verilmeden yapılan çağrıları (ör. `FlatFieldOp.run()`) sessizce
etkilemezdi.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imgflow.io_utils.image_io import load_image, save_image

FLATFIELD_DIR = Path.home() / ".imgflow" / "flatfield"


class FlatFieldReferenceNotFoundError(Exception):
    """Kayıtlı olmayan bir referans adı `load_reference` ile istendiğinde fırlatılır."""


def _slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Referans adı en az bir harf/rakam içermeli.")
    return slug


def _resolve_dir(directory: Path | None) -> Path:
    return directory if directory is not None else FLATFIELD_DIR


def _paths_for(name: str, directory: Path | None = None) -> tuple[Path, Path]:
    resolved = _resolve_dir(directory)
    slug = _slugify(name)
    return resolved / f"{slug}.png", resolved / f"{slug}.json"


def save_reference(name: str, image: np.ndarray, directory: Path | None = None) -> None:
    directory = _resolve_dir(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image_path, meta_path = _paths_for(name, directory)
    save_image(image_path, image)
    meta_path.write_text(json.dumps({"name": name}, ensure_ascii=False), encoding="utf-8")


def load_reference(name: str, directory: Path | None = None) -> np.ndarray:
    image_path, meta_path = _paths_for(name, directory)
    if not image_path.exists() or not meta_path.exists():
        raise FlatFieldReferenceNotFoundError(
            f"'{name}' adında kayıtlı bir aydınlatma referansı bulunamadı."
        )
    return load_image(image_path)


def list_references(directory: Path | None = None) -> list[str]:
    directory = _resolve_dir(directory)
    if not directory.is_dir():
        return []
    names = []
    for meta_path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            names.append(data["name"])
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return names


def delete_reference(name: str, directory: Path | None = None) -> None:
    image_path, meta_path = _paths_for(name, directory)
    image_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
