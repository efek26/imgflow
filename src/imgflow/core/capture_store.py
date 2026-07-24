"""Kalibrasyon sırasında ('Kare Yakala') yakalanan karelerin kalıcı galerisi.

Önceden bu kareler sadece bellekte tutulup (köşe/nokta verisi çıkarıldıktan sonra) atılıyordu
— kullanıcı daha sonra "hangi kareleri kullandım" diye geri dönüp bakamıyordu. Bu modül her
yakalamayı diske yazar (`image_io.save_image` ile, Unicode-güvenli) ve küçük bir JSON index
ile listeler/siler.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from imgflow.io_utils.image_io import save_image

CAPTURE_DIR = Path.home() / ".imgflow" / "captures"
MAX_CAPTURES = 200
"""Fabrika ortamında haftalarca çalışan bir uygulamada sınırsız disk büyümesini önlemek için
en yeni bu kadar kayıt tutulur; fazlası her `save_capture` çağrısında otomatik silinir."""

_INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class CaptureRecord:
    id: str
    source: str
    timestamp: str
    path: Path


def _index_path() -> Path:
    return CAPTURE_DIR / _INDEX_FILENAME


def _read_index() -> list[dict]:
    path = _index_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _write_index(entries: list[dict]) -> None:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    _index_path().write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def save_capture(image: np.ndarray, source: str) -> CaptureRecord:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    record_id = uuid.uuid4().hex
    timestamp = datetime.now().isoformat(timespec="seconds")
    filename = f"{record_id}.png"
    save_image(CAPTURE_DIR / filename, image)

    entries = _read_index()
    # Saniye çözünürlüklü `timestamp` (gösterim amaçlı) aynı saniyede yapılan iki yakalamada
    # eşit olabilir — sıralama için ayrı, kesinlikle artan bir `seq` sayacı kullanılır.
    seq = max((e.get("seq", 0) for e in entries), default=0) + 1
    entries.append(
        {"id": record_id, "source": source, "timestamp": timestamp, "filename": filename, "seq": seq}
    )
    entries.sort(key=lambda e: e["seq"])
    while len(entries) > MAX_CAPTURES:
        oldest = entries.pop(0)
        (CAPTURE_DIR / oldest["filename"]).unlink(missing_ok=True)
    _write_index(entries)
    return CaptureRecord(id=record_id, source=source, timestamp=timestamp, path=CAPTURE_DIR / filename)


def list_captures(source: str | None = None) -> list[CaptureRecord]:
    entries = sorted(_read_index(), key=lambda e: e.get("seq", 0), reverse=True)
    return [
        CaptureRecord(id=e["id"], source=e["source"], timestamp=e["timestamp"], path=CAPTURE_DIR / e["filename"])
        for e in entries
        if source is None or e["source"] == source
    ]


def delete_capture(capture_id: str) -> None:
    entries = _read_index()
    remaining = [e for e in entries if e["id"] != capture_id]
    removed = [e for e in entries if e["id"] == capture_id]
    for e in removed:
        (CAPTURE_DIR / e["filename"]).unlink(missing_ok=True)
    _write_index(remaining)
