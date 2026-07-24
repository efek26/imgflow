"""Seçili pipeline adımının `measurements` çıktısını özetleyen salt-okunur panel.

`PortType.MEASUREMENTS` (bkz. `core/types.py`) herhangi bir operatörden gelebilir; bu widget
JENERİK tutulur:
- `"model"` + `"angle"` anahtarları varsa (`geom.shape_match`'in çıktısı; her eşleşme artık
  model adı+sırası değil TEK bir akan sayaçla etiketlenir, bkz. `operators/builtin/
  shape_match.py`) her eşleşmeyi NUMARASINA göre `x/y/alpha` değerleriyle tek tek listeler
  (gerçek kullanıcı isteği: "her şekili 1,2,3,4 diye adlandıralım, sonra 1'in x,y,alpha
  değerleri... yazsın").
- `"model"` + `"confidence"` anahtarları varsa (`ml.onnx_detect`'in çıktısı) her tespiti
  NUMARASINA göre sınıf adı/güven/kutu değerleriyle tek tek listeler — `shape_match`'in
  x/y/alpha dalıyla AYNI mantık, farklı alanlar (açı yerine güven skoru + kutu).
- `"tolerance_ok"` anahtarı varsa (ör. `analysis.region_props`/`analysis.color_props`'ta
  tolerans kontrolü açıkken) OK/NG sayımı gösterir.
- `"model"`/`"tolerance_ok"` YOKSA ve tek satırlık bir sonuçsa (ör. `analysis.color_props`/
  `analysis.texture_props`'un tolerans KAPALIYKEN ürettiği tüm-görüntü tek satırı) her alanı
  `"anahtar: değer"` şeklinde listeler — aksi halde bu tarz operatörlerin tek somut çıktısı
  sadece CSV export'ta görünür, panelde "Toplam ölçüm: 1" gibi anlamsız kalırdı.
- Hiçbiri yoksa (ör. birden fazla jenerik satır) sadece toplam ölçüm sayısını gösterir.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

_EMPTY_TEXT = "Ölçüm yok."


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


class MeasurementsSummaryPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(_EMPTY_TEXT)
        self._label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def text(self) -> str:
        return self._label.text()

    def set_measurements(self, measurements: list[dict[str, Any]] | None) -> None:
        if not measurements:
            self._label.setText(_EMPTY_TEXT)
            return

        if all("model" in m and "angle" in m for m in measurements):
            lines = [
                f"{m.get('label', i)} ({m['model']}): x={m['x']:.1f}  y={m['y']:.1f}  α={m['angle']:.1f}°"
                for i, m in enumerate(measurements, start=1)
            ]
            lines.append(f"Toplam: {len(measurements)}")
            self._label.setText("\n".join(lines))
            return

        if all("model" in m and "confidence" in m for m in measurements):
            lines = [
                f"{m.get('label', i)} ({m['class_name']}): %{m['confidence'] * 100:.0f}  "
                f"[x={m['bbox_x']:.0f} y={m['bbox_y']:.0f} w={m['bbox_w']:.0f} h={m['bbox_h']:.0f}]"
                for i, m in enumerate(measurements, start=1)
            ]
            lines.append(f"Toplam: {len(measurements)}")
            self._label.setText("\n".join(lines))
            return

        if all("tolerance_ok" in m for m in measurements):
            ok_count = sum(1 for m in measurements if m["tolerance_ok"])
            ng_count = len(measurements) - ok_count
            lines = [f"OK: {ok_count}", f"NG: {ng_count}", f"Toplam: {len(measurements)}"]
            self._label.setText("\n".join(lines))
            return

        if len(measurements) == 1 and "model" not in measurements[0]:
            lines = [f"{key}: {_format_value(value)}" for key, value in measurements[0].items()]
            self._label.setText("\n".join(lines))
            return

        self._label.setText(f"Toplam ölçüm: {len(measurements)}")
