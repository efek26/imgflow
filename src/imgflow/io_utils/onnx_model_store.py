"""Harici (imgflow DIŞINDA eğitilmiş) ONNX modellerinin isimle kaydı.

`shape_model_store.py` ile AYNI isimli-profil deseni, tek fark: `.onnx` dosyasının kendisi
(genelde büyük bir ikili dosya) `~/.imgflow/onnx_models/<slug>.onnx` olarak KOPYALANIR
(`shape_model_store.py`'nin "self-contained" felsefesiyle aynı — orijinal dosya kullanıcı
tarafından taşınır/silinirse referans kırılmasın), yanına küçük bir `<slug>.json` metadata
sidecar'ı yazılır.

`task_type` üç değerden biri olabilir, ÜÇÜ de `ml.onnx_detect` operatöründe FONKSİYONEL
(bkz. `core/onnx_detection.py::find_objects_yolo`/`classify_image`/`segment_image`):
`"yolo"` (nesne tespiti, kutu), `"classification"` (tüm-görüntü sınıflandırma/anomali,
kutu YOK), `"segmentation"` (semantik/piksel-başına segmentasyon, `class_id=0` arka plan
varsayılır).

**Testler `ONNX_MODEL_DIR`'ı `monkeypatch` ile `tmp_path`'e yönlendirmeli** — aksi halde
gerçek ev dizinine dosya sızar. `directory` parametresi ÖNTANIMLI olarak `None`dır ve
YALNIZCA fonksiyon GÖVDESİNDE `ONNX_MODEL_DIR`'a düşer (bkz. `shape_model_store.py`'deki
aynı gotcha notu)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ONNX_MODEL_DIR = Path.home() / ".imgflow" / "onnx_models"

TASK_TYPE_YOLO = "yolo"
TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_SEGMENTATION = "segmentation"
SUPPORTED_TASK_TYPES = (TASK_TYPE_YOLO, TASK_TYPE_CLASSIFICATION, TASK_TYPE_SEGMENTATION)
"""Kaydedilebilen tüm görev türleri — sadece `TASK_TYPE_YOLO` şu an çalışır çalıştırılabilir
(bkz. modül docstring'i)."""


class OnnxModelNotFoundError(Exception):
    """Kayıtlı olmayan bir model adı `load_model_meta`/`model_path_for` ile istendiğinde fırlatılır."""


def _slugify(name: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("Model adı en az bir harf/rakam içermeli.")
    return slug


def _resolve_dir(directory: Path | None) -> Path:
    return directory if directory is not None else ONNX_MODEL_DIR


def _paths_for(name: str, directory: Path | None = None) -> tuple[Path, Path]:
    resolved = _resolve_dir(directory)
    slug = _slugify(name)
    return resolved / f"{slug}.onnx", resolved / f"{slug}.json"


def save_model(
    name: str,
    onnx_path: str | Path,
    task_type: str,
    input_size: int,
    class_labels: list[str],
    directory: Path | None = None,
) -> None:
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Bilinmeyen model türü: '{task_type}'.")
    directory = _resolve_dir(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = _paths_for(name, directory)
    shutil.copyfile(onnx_path, model_path)
    meta_path.write_text(
        json.dumps(
            {
                "name": name,
                "task_type": task_type,
                "input_size": int(input_size),
                "class_labels": list(class_labels),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_model_meta(name: str, directory: Path | None = None) -> dict:
    model_path, meta_path = _paths_for(name, directory)
    if not model_path.exists() or not meta_path.exists():
        raise OnnxModelNotFoundError(f"'{name}' adında kayıtlı bir ONNX modeli bulunamadı.")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["path"] = model_path
    return data


def list_models(directory: Path | None = None) -> list[str]:
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


def delete_model(name: str, directory: Path | None = None) -> None:
    model_path, meta_path = _paths_for(name, directory)
    model_path.unlink(missing_ok=True)
    meta_path.unlink(missing_ok=True)
