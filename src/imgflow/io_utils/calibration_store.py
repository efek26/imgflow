"""İsimli kalibrasyon profillerinin (lens profili + yükseklik-ölçek modeli) JSON olarak
diskte saklanması.

Bu JSON veridir, Unicode dosya yolu sorunu olan İKİLİ görüntü verisi DEĞİL — bu yüzden
`io_utils/image_io.py`'deki cv2.imencode/imdecode deseni değil, `io_utils/recipe.py`'deki
Path.write_text/read_text(encoding="utf-8") deseni izlenir.

Reçetelerden (bkz. `io_utils/recipe.py` — `Graph.calibration_profile`) İSİM ile referans
verilir; kamera ayarları/kalibrasyonu recipe şemasının kendisine gömülmez, sadece bir
profil adına işaret eder.

Bir profil, lens kalibrasyonu ile yükseklik-ölçek kalibrasyonunun AYRI dialoglarda
yapılabilmesi nedeniyle iki parçadan oluşur; `save_profile` bu yüzden PARÇALI kayda izin
verir — sadece `lens_profile` (ya da sadece `height_model`) verilerek çağrılırsa, dosyada
zaten var olan DİĞER parça SESSİZCE SİLİNMEZ, korunur (bkz. `_merge_existing`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from imgflow.core.focus_distance import FocusDistanceModel
from imgflow.core.height_scale_calibration import HeightScaleModel
from imgflow.core.lens_calibration import LensProfile
from imgflow.core.plane_rectification import PlaneRectification

CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_CALIBRATION_DIR = Path("calibration")


@dataclass(frozen=True)
class CalibrationProfileData:
    lens_profile: LensProfile | None
    height_model: HeightScaleModel | None
    created_at: str | None
    """Bu profilin en son ne zaman kaydedildiği (ISO 8601) — kalibrasyonun ne kadar
    güncel/güvenilir olduğunu göstermek için (bkz. dialoglardaki durum etiketleri)."""
    operator_note: str | None
    focus_model: FocusDistanceModel | None = None
    """DENEYSEL netlik(odak)→mesafe modeli — checkerboard'ı farklı yüksekliklerde göstererek
    `LensCalibrationDialog`'da otomatik üretilir (bkz. `core/focus_distance.py`). Eski
    profillerde bu alan yoktur, `None` olarak yüklenir. Gerçek kullanımda bu modelin
    korelasyonu genelde çok zayıf çıktı (kameranın odak derinliği yetersiz) — bkz.
    `reference_distance_mm`, PRATİKTE kullanılan/önerilen yöntem budur."""
    reference_distance_mm: float | None = None
    """Kullanıcının galeri de işaretlediği TEK bir "referans" checkerboard karesinin
    (`LensCalibrationDialog._reference_capture_index`) kalibrasyon sırasında `solvePnP`/
    `calibrateCamera`'dan çözülen GERÇEK mesafesi (mm) — üretim sırasında ölçülen cisimlerin
    hep AYNI (bant seviyesi) düzlemde durduğu varsayımıyla, `mm_per_px = reference_distance_mm
    / fx` SABİT ve doğrudan hesaplanır. Netlik-tabanlı deneysel tahminin aksine (zayıf sinyal
    riski, bkz. `focus_model`) bu YAKLAŞIK DEĞİL, tam solvePnP doğruluğundadır — sadece cismin
    kalibrasyondaki referans düzlemden farklı bir yükseklikte olduğu durumlarda hatalıdır."""
    plane_rectification: PlaneRectification | None = None
    """Referans karenin TAM pozundan (rotasyon + öteleme) çözülen düzlem homografisi — bkz.
    `core/plane_rectification.py`. `reference_distance_mm`'in aksine (tek skaler, kamera
    düzleme tam dik değilse konuma göre hatalı) bu, kamera açısı ne olursa olsun düzlem
    üzerindeki HER pikselde doğru ölçüm verir (gerçek kullanıcı raporuyla doğrulandı: aynı
    cisim ekranın farklı yerlerinde 6mm/5.5mm ölçülüyordu). Mevcutsa, çalışma zamanında
    `reference_distance_mm`'den ÖNCELİKLİDİR."""


def _profile_path(name: str, directory: str | Path = DEFAULT_CALIBRATION_DIR) -> Path:
    return Path(directory) / f"{name}.json"


def _read_raw(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_profile(
    name: str,
    lens_profile: LensProfile | None = None,
    height_model: HeightScaleModel | None = None,
    focus_model: FocusDistanceModel | None = None,
    reference_distance_mm: float | None = None,
    plane_rectification: PlaneRectification | None = None,
    directory: str | Path = DEFAULT_CALIBRATION_DIR,
    created_at: str | None = None,
    operator_note: str | None = None,
) -> None:
    """`lens_profile`/`height_model`/`focus_model`/`reference_distance_mm`/
    `plane_rectification`'den biri `None` verilirse ve profil zaten kayıtlıysa, o bileşen
    için dosyadaki ÖNCEKİ veri korunur (üzerine yazılmaz) — lens ve yükseklik/netlik
    kalibrasyonu ayrı dialoglarda, ayrı zamanlarda yapılabildiği için bu davranış gerekli."""
    path = _profile_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_raw(path)

    if lens_profile is not None:
        lens_data = lens_profile.to_dict()
    else:
        lens_data = existing.get("lens_profile") if existing else None

    if height_model is not None:
        height_data = height_model.to_dict()
    else:
        height_data = existing.get("height_model") if existing else None

    if focus_model is not None:
        focus_data = focus_model.to_dict()
    else:
        focus_data = existing.get("focus_model") if existing else None

    if reference_distance_mm is not None:
        reference_data = reference_distance_mm
    else:
        reference_data = existing.get("reference_distance_mm") if existing else None

    if plane_rectification is not None:
        rectification_data = plane_rectification.to_dict()
    else:
        rectification_data = existing.get("plane_rectification") if existing else None

    data: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "lens_profile": lens_data,
        "height_model": height_data,
        "focus_model": focus_data,
        "reference_distance_mm": reference_data,
        "plane_rectification": rectification_data,
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
        "operator_note": operator_note,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_profile(name: str, directory: str | Path = DEFAULT_CALIBRATION_DIR) -> CalibrationProfileData:
    path = _profile_path(name, directory)
    data = json.loads(path.read_text(encoding="utf-8"))
    lens_data = data.get("lens_profile")
    height_data = data.get("height_model")
    focus_data = data.get("focus_model")
    reference_distance = data.get("reference_distance_mm")
    rectification_data = data.get("plane_rectification")
    return CalibrationProfileData(
        lens_profile=LensProfile.from_dict(lens_data) if lens_data is not None else None,
        height_model=HeightScaleModel.from_dict(height_data) if height_data is not None else None,
        focus_model=FocusDistanceModel.from_dict(focus_data) if focus_data is not None else None,
        reference_distance_mm=float(reference_distance) if reference_distance is not None else None,
        plane_rectification=PlaneRectification.from_dict(rectification_data) if rectification_data is not None else None,
        created_at=data.get("created_at"),
        operator_note=data.get("operator_note"),
    )


def list_profiles(directory: str | Path = DEFAULT_CALIBRATION_DIR) -> list[str]:
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted(p.stem for p in dir_path.glob("*.json"))


def profile_mm_per_px(profile_data: CalibrationProfileData) -> float | None:
    """Bir profilin karesinden BAĞIMSIZ, SABİT mm/px değerini (varsa) hesaplar -- profil seçim
    listelerinde kullanıcı isimlere bakarak değil doğrudan ölçeğe göre ayırt edebilsin diye.

    Öncelik `MainWindow._compute_auto_mm_per_px` ile aynıdır: `plane_rectification` (konum-
    bağımsız) > `reference_distance_mm/fx`. Netlik-tabanlı `focus_model` KAREYE bağımlı
    (dinamik) olduğu için burada hesaba katılmaz -- bu profille SABİT bir değer üretilemiyorsa
    `None` döner (liste bu durumda sadece profil adını gösterir)."""
    if profile_data.plane_rectification is not None:
        return profile_data.plane_rectification.mm_per_px
    if profile_data.lens_profile is not None and profile_data.reference_distance_mm is not None:
        fx = float(profile_data.lens_profile.camera_matrix[0, 0])
        if fx > 0:
            return profile_data.reference_distance_mm / fx
    return None


def format_profile_label(name: str, profile_data: CalibrationProfileData) -> str:
    """Seçim listelerinde gösterilecek "ad (X.XXX mm/px)" etiketini üretir; SABİT bir mm/px
    hesaplanamıyorsa sadece adı döner. Ayırma için bkz. `profile_name_from_label`."""
    mm_per_px = profile_mm_per_px(profile_data)
    if mm_per_px is None:
        return name
    return f"{name} ({mm_per_px:.3f} mm/px)"


def profile_name_from_label(label: str) -> str:
    """`format_profile_label` ile üretilen bir etiketten profil adını geri çıkarır."""
    return label.split(" (", 1)[0]
