"""İsimli kamera ayarı ("hat" ön ayarı) profillerinin JSON olarak diskte saklanması.

Bu JSON veridir (GenICam node adı → değer, tümü zaten JSON-uyumlu tipler: str/int/float/
bool), Unicode dosya yolu sorunu olan İKİLİ görüntü verisi DEĞİL — bu yüzden
`io_utils/image_io.py`'deki cv2.imencode/imdecode deseni değil, `io_utils/calibration_store.py`
ile aynı Path.write_text/read_text(encoding="utf-8") deseni izlenir.

`calibration_store.py`'nin aksine burada parçalı (partial-merge) kayıt yok: kamera ayarları
tek bir dialogdan/panelden, tek seferde toplanan düz bir `dict` — lens/yükseklik kalibrasyonu
gibi iki ayrı akıştan gelmiyor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

CAMERA_SETTINGS_SCHEMA_VERSION = 1
DEFAULT_CAMERA_SETTINGS_DIR = Path("camera_settings")


@dataclass(frozen=True)
class CameraSettingsData:
    values: dict[str, Any]
    created_at: str | None
    operator_note: str | None


def _profile_path(name: str, directory: str | Path = DEFAULT_CAMERA_SETTINGS_DIR) -> Path:
    return Path(directory) / f"{name}.json"


def save_settings(
    name: str,
    values: dict[str, Any],
    directory: str | Path = DEFAULT_CAMERA_SETTINGS_DIR,
    created_at: str | None = None,
    operator_note: str | None = None,
) -> None:
    path = _profile_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "schema_version": CAMERA_SETTINGS_SCHEMA_VERSION,
        "values": values,
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "operator_note": operator_note,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_settings(name: str, directory: str | Path = DEFAULT_CAMERA_SETTINGS_DIR) -> CameraSettingsData:
    path = _profile_path(name, directory)
    data = json.loads(path.read_text(encoding="utf-8"))
    return CameraSettingsData(
        values=dict(data.get("values") or {}),
        created_at=data.get("created_at"),
        operator_note=data.get("operator_note"),
    )


def list_settings(directory: str | Path = DEFAULT_CAMERA_SETTINGS_DIR) -> list[str]:
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted(p.stem for p in dir_path.glob("*.json") if not p.stem.startswith("_"))


_LAST_USED_NAME = "_last_used"
"""Adı `_` ile başladığı için `list_settings()`'te (dolayısıyla Kaydet/Yükle diyaloglarındaki
isim listesinde) görünmez — kullanıcının elle yönettiği isimli ön ayarlardan ayrı, kamera
bağlantısı kesildiğinde otomatik yazılıp bir sonraki bağlantıda otomatik uygulanan sessiz bir
"son kullanılan ayarlar" kaydı."""


def save_last_used(values: dict[str, Any], directory: str | Path = DEFAULT_CAMERA_SETTINGS_DIR) -> None:
    save_settings(_LAST_USED_NAME, values, directory=directory)


def load_last_used(directory: str | Path = DEFAULT_CAMERA_SETTINGS_DIR) -> CameraSettingsData | None:
    try:
        return load_settings(_LAST_USED_NAME, directory=directory)
    except (OSError, ValueError, KeyError):
        return None
