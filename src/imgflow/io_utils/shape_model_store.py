"""Eğitilmiş şekil eşleştirme modellerinin (`core/shape_matching.ShapeModel`) isimle diskte
JSON olarak saklanması — `calibration_store.py`'deki isimli-profil deseninin birebir
uygulanması: model bir kere "Şekil Eşleştirme" aracında eğitilir, isimle kaydedilir; pipeline
operatörü (`geom.shape_match`) modeli sadece isimle referans alır.

`custom_filters.py`/`capture_store.py` ile aynı gerekçeyle `~/.imgflow` altında saklanır.
**Testler `SHAPE_MODEL_DIR`'ı `monkeypatch` ile `tmp_path`'e yönlendirmeli** — aksi halde
gerçek ev dizinine dosya sızar. `directory` parametresi ÖNTANIMLI olarak `None`dır ve YALNIZCA
fonksiyon GÖVDESİNDE `SHAPE_MODEL_DIR`'a düşer (bir varsayılan argüman değeri OLARAK DEĞİL) —
aksi halde Python varsayılan argümanı `def` çalıştığı anda bir kere bağlar, ve
`monkeypatch.setattr(shape_model_store, "SHAPE_MODEL_DIR", tmp_path)` `directory` argümanı hiç
verilmeden yapılan çağrıları (ör. `ShapeMatchOp.run()`) SESSİZCE etkilemez — bu tam olarak
gerçek ev dizinine test verisi sızmasına yol açan hatanın kendisiydi (bir kere gerçekten oldu,
bkz. `[[imgflow-named-profile-pattern]]`).
"""

from __future__ import annotations

import json
from pathlib import Path

from imgflow.core.shape_matching import ShapeModel

SHAPE_MODEL_DIR = Path.home() / ".imgflow" / "shape_models"


class ShapeModelNotFoundError(Exception):
    """Kayıtlı olmayan bir model adı `load_shape_model` ile istendiğinde fırlatılır."""


def _slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Model adı en az bir harf/rakam içermeli.")
    return slug


def _resolve_dir(directory: Path | None) -> Path:
    return directory if directory is not None else SHAPE_MODEL_DIR


def _path_for(name: str, directory: Path | None = None) -> Path:
    return _resolve_dir(directory) / f"{_slugify(name)}.json"


def save_shape_model(name: str, model: ShapeModel, directory: Path | None = None) -> None:
    directory = _resolve_dir(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "model": model.to_dict()}
    _path_for(name, directory).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_shape_model(name: str, directory: Path | None = None) -> ShapeModel:
    path = _path_for(name, directory)
    if not path.exists():
        raise ShapeModelNotFoundError(f"'{name}' adında kayıtlı bir şekil modeli bulunamadı.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ShapeModel.from_dict(data["model"])


def list_shape_models(directory: Path | None = None) -> list[str]:
    directory = _resolve_dir(directory)
    if not directory.is_dir():
        return []
    names = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            names.append(data["name"])
        except (OSError, json.JSONDecodeError, KeyError):
            continue
    return names


def delete_shape_model(name: str, directory: Path | None = None) -> None:
    _path_for(name, directory).unlink(missing_ok=True)


def rename_shape_model(old_name: str, new_name: str, directory: Path | None = None) -> None:
    directory = _resolve_dir(directory)
    old_path = _path_for(old_name, directory)
    if not old_path.exists():
        raise ShapeModelNotFoundError(f"'{old_name}' adında kayıtlı bir şekil modeli bulunamadı.")

    new_path = _path_for(new_name, directory)
    if new_path.exists() and new_path != old_path:
        raise FileExistsError(f"'{new_name}' adında zaten kayıtlı bir model var.")

    data = json.loads(old_path.read_text(encoding="utf-8"))
    data["name"] = new_name
    new_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    if new_path != old_path:
        old_path.unlink()
