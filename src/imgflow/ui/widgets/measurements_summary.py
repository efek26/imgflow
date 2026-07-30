"""Seçili pipeline adımının `measurements` çıktısını özetleyen salt-okunur panel.

`PortType.MEASUREMENTS` (bkz. `core/types.py`) herhangi bir operatörden gelebilir; bu widget
JENERİK tutulur:
- `"model"` + `"angle"` anahtarları varsa (`geom.shape_match`'in çıktısı; her eşleşme artık
  model adı+sırası değil TEK bir akan sayaçla etiketlenir, bkz. `operators/builtin/
  shape_match.py`) her eşleşmeyi NUMARASINA göre `x/y/alpha` değerleriyle tek tek listeler
  (gerçek kullanıcı isteği: "her şekili 1,2,3,4 diye adlandıralım, sonra 1'in x,y,alpha
  değerleri... yazsın").
- `"model"` + `"confidence"` + `"bbox_x"` anahtarları varsa (`ml.onnx_detect`'in YOLO çıktısı)
  her tespiti NUMARASINA göre sınıf adı/güven/kutu değerleriyle tek tek listeler —
  `shape_match`'in x/y/alpha dalıyla AYNI mantık, farklı alanlar (açı yerine güven skoru +
  kutu). `"tolerance_ok"` da varsa (operatörün `defect_classes` parametresi doluyken) satır
  sonuna "[OK]"/"[NG]" eklenir VE altına toplam "OK: N  NG: M" satırı eklenir.
- `"model"` + `"confidence"` var AMA `"bbox_x"` YOKSA (`ml.onnx_detect`'in sınıflandırma
  çıktısı — tüm görüntüye TEK etiket, kutu yok) her modelin sonucunu NUMARASINA göre sınıf
  adı/güven değeriyle listeler, aynı OK/NG desenini izler.
- `"model"` + `"area_px"` anahtarları varsa (`ml.onnx_detect`'in segmentasyon çıktısı — her
  satır görüntüde bulunan BİR sınıfın piksel alanı, `class_id=0`/arka plan hariç) her satırı
  NUMARASINA göre sınıf adı + alan (px²/yüzde) ile listeler, aynı OK/NG desenini izler.
  Bu üç dal onnx ölçümlerini yakaladığından aşağıdaki jenerik `"tolerance_ok" in m` dalına
  onnx ölçümleri İÇİN hiçbir zaman sıra gelmez, o dal `region_props`/`color_props` gibi
  `"model"` anahtarı OLMAYAN ölçümler içindir.
- `"l_mean"` + `"label"` anahtarları varsa (`analysis.color_props`'un "Otomatik Nesne
  Tespiti" YA DA "Elle ROI Çiz" açıkken ürettiği nesne/ROI-başına satırlar) her satırı
  NUMARASINA göre açıklamalı L*/a*/b* etiketleriyle (parantez içinde ne anlama geldiğiyle)
  tek tek listeler — ham `l_mean`/`a_mean`/`b_mean` anahtarları kullanıcıya bir şey ifade
  etmeyeceğinden. `"manual"` anahtarı True ise (Elle ROI Çiz) etiket "#N" yerine "ROIN"
  yazılır VE `"l_min"` varsa altına L*/a*/b* için min-max ARALIĞI eklenir — gerçek kullanıcı
  isteği: "roi içindeki renk dalgalanmalarını bana aralık olarak versin" (tek bir ortalama
  değer, parlama/gölge kaynaklı renk dalgalanmasını gizleyebilir).
- `"contrast"` + `"label"` anahtarları varsa (`analysis.texture_props`'un aynı "Otomatik
  Nesne Tespiti"/"Elle ROI Çiz" modu) her satırı NUMARASINA göre (manuel modda "ROIN")
  açıklamalı Kontrast/Homojenlik/Enerji/Korelasyon etiketleriyle listeler.
- `"tolerance_ok"` anahtarı varsa (ör. `analysis.region_props`/`analysis.color_props`'ta
  tolerans kontrolü açıkken) OK/NG sayımı gösterir.
- `"model"`/`"tolerance_ok"` YOKSA ve tek satırlık bir sonuçsa (ör. `analysis.color_props`/
  `analysis.texture_props`'un tolerans KAPALIYKEN ürettiği tüm-görüntü tek satırı) her alanı
  `FIELD_LABELS` üzerinden açıklamalı bir etiketle (`"anahtar: değer"` yerine ör. "L*
  (Parlaklık) Ort.: 61.20") listeler — aksi halde bu tarz operatörlerin tek somut çıktısı
  sadece CSV export'ta görünür, panelde "Toplam ölçüm: 1" gibi anlamsız kalır, ham anahtar
  adları (`l_mean`, `contrast`...) da kullanıcıya ne ölçüldüğünü anlatmazdı (gerçek kullanıcı
  geri bildirimi: "özelliklerin ne olduğu pek net değil").
- Hiçbiri yoksa (ör. birden fazla jenerik satır) sadece toplam ölçüm sayısını gösterir.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

_EMPTY_TEXT = "Ölçüm yok."

FIELD_LABELS: dict[str, str] = {
    "l_mean": "L* (Parlaklık) Ort.",
    "a_mean": "a* (Yeşil↔Kırmızı) Ort.",
    "b_mean": "b* (Mavi↔Sarı) Ort.",
    "l_std": "L* (Parlaklık) Std. Sapma",
    "a_std": "a* (Yeşil↔Kırmızı) Std. Sapma",
    "b_std": "b* (Mavi↔Sarı) Std. Sapma",
    "delta_e": "ΔE (Referans Renkten Sapma)",
    "tolerance_ok": "Tolerans Durumu",
    "contrast": "Kontrast (Doku Pürüzlülüğü)",
    "homogeneity": "Homojenlik (Doku Düzgünlüğü)",
    "energy": "Enerji (Doku Tekdüzeliği)",
    "correlation": "Korelasyon (Komşu Piksel İlişkisi)",
}
"""Ham ölçüm anahtarlarının (`analysis.color_props`/`analysis.texture_props` çıktısı)
kullanıcıya gösterilecek açıklamalı Türkçe karşılıkları. Burada olmayan bir anahtar
olduğu gibi (değişmeden) gösterilir."""


def _field_label(key: str) -> str:
    return FIELD_LABELS.get(key, key)


def _format_value(key: str, value: Any) -> str:
    if key == "tolerance_ok":
        return "OK" if value else "NG"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


class MeasurementsSummaryPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(_EMPTY_TEXT)
        self._label.setWordWrap(True)

        # Gerçek kullanıcı isteği: "her işlemin sonucunun süresi sonuçlar kısmında yazmalı"
        # -- mevcut ölçüm listesinden AYRI, pipeline'daki HER adımın en son ne kadar sürdüğünü
        # gösteren küçük bir tablo. Adım YOKKEN (0 satır) tamamen gizli tutulur ki normal
        # ölçüm paneli göze çarpmasın.
        self._duration_heading = QLabel("Adım Süreleri:")
        self._duration_table = QTableWidget(0, 2)
        self._duration_table.setHorizontalHeaderLabels(["Adım", "Süre (ms)"])
        self._duration_table.verticalHeader().setVisible(False)
        self._duration_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._duration_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._duration_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._duration_table.setMaximumHeight(160)
        self._duration_heading.setVisible(False)
        self._duration_table.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._duration_heading)
        layout.addWidget(self._duration_table)
        layout.addStretch(1)

    def text(self) -> str:
        return self._label.text()

    def set_step_durations(self, rows: list[tuple[str, float]] | None) -> None:
        has_rows = bool(rows)
        self._duration_heading.setVisible(has_rows)
        self._duration_table.setVisible(has_rows)
        self._duration_table.setRowCount(len(rows) if rows else 0)
        if not rows:
            return
        for row_index, (label, seconds) in enumerate(rows):
            self._duration_table.setItem(row_index, 0, QTableWidgetItem(label))
            self._duration_table.setItem(row_index, 1, QTableWidgetItem(f"{seconds * 1000:.1f} ms"))

    def set_measurements(self, measurements: list[dict[str, Any]] | None) -> None:
        if not measurements:
            self._label.setText(_EMPTY_TEXT)
            return

        if all("model" in m and "angle" in m for m in measurements):
            lines = []
            for i, m in enumerate(measurements, start=1):
                line = f"{m.get('label', i)} ({m['model']}): x={m['x']:.1f}  y={m['y']:.1f}  α={m['angle']:.1f}°"
                # Gerçek kullanıcı isteği: "geometrik eşlemede scale boyutu ve cismin
                # ötelenme uzaklığı da yazmalı" -- netleşen anlam: bulunan nesnenin GERÇEK
                # DÜNYA boyutu (mm, kalibrasyon varsa) ve öğretildiği pozisyondan ne kadar
                # ötelendiği. `width_mm`/`displacement_*` yoksa (kalibrasyon yok/eski model)
                # bu ek metin HİÇ eklenmez.
                if "scale_percent" in m:
                    line += f"  ölçek=%{m['scale_percent']:.0f}"
                if "width_mm" in m:
                    # Kalibrasyon aktifken px değeri de yanına yazılır -- gerçek kullanıcı
                    # isteği: "kalibrasyon seçili olduğu her senaryoda pixelin yanında mm de
                    # yazsın veya cm" (önceden sadece mm gösteriliyordu, px gizleniyordu).
                    line += (
                        f"  boy={m['width_px']:.0f}x{m['height_px']:.0f}px "
                        f"({m['width_mm']:.1f}x{m['height_mm']:.1f}mm)"
                    )
                elif "width_px" in m:
                    line += f"  boy={m['width_px']:.0f}x{m['height_px']:.0f}px"
                if "displacement_x_cm" in m:
                    line += (
                        f"  öteleme=x:{m['displacement_x_cm']:+.2f} y:{m['displacement_y_cm']:+.2f}cm"
                    )
                elif "displacement_x_px" in m:
                    line += f"  öteleme=x:{m['displacement_x_px']:+.1f} y:{m['displacement_y_px']:+.1f}px"
                lines.append(line)
            lines.append(f"Toplam: {len(measurements)}")
            self._label.setText("\n".join(lines))
            return

        if all("model" in m and "confidence" in m and "bbox_x" in m for m in measurements):
            lines = []
            for i, m in enumerate(measurements, start=1):
                line = (
                    f"{m.get('label', i)} ({m['class_name']}): %{m['confidence'] * 100:.0f}  "
                    f"[x={m['bbox_x']:.0f} y={m['bbox_y']:.0f} w={m['bbox_w']:.0f} h={m['bbox_h']:.0f}]"
                )
                if "tolerance_ok" in m:
                    line += "  [OK]" if m["tolerance_ok"] else "  [NG]"
                lines.append(line)
            lines.append(f"Toplam: {len(measurements)}")
            if all("tolerance_ok" in m for m in measurements):
                ok_count = sum(1 for m in measurements if m["tolerance_ok"])
                lines.append(f"OK: {ok_count}  NG: {len(measurements) - ok_count}")
            self._label.setText("\n".join(lines))
            return

        if all("model" in m and "confidence" in m and "bbox_x" not in m for m in measurements):
            lines = []
            for i, m in enumerate(measurements, start=1):
                line = f"{m.get('label', i)} ({m['model']}): {m['class_name']}  %{m['confidence'] * 100:.0f}"
                if "tolerance_ok" in m:
                    line += "  [OK]" if m["tolerance_ok"] else "  [NG]"
                lines.append(line)
            lines.append(f"Toplam: {len(measurements)}")
            if all("tolerance_ok" in m for m in measurements):
                ok_count = sum(1 for m in measurements if m["tolerance_ok"])
                lines.append(f"OK: {ok_count}  NG: {len(measurements) - ok_count}")
            self._label.setText("\n".join(lines))
            return

        if all("model" in m and "area_px" in m for m in measurements):
            lines = []
            for i, m in enumerate(measurements, start=1):
                line = (
                    f"{m.get('label', i)} ({m['model']}) {m['class_name']}: "
                    f"%{m['area_percent']:.1f}  ({m['area_px']}px²)"
                )
                if "tolerance_ok" in m:
                    line += "  [OK]" if m["tolerance_ok"] else "  [NG]"
                lines.append(line)
            lines.append(f"Toplam: {len(measurements)}")
            if all("tolerance_ok" in m for m in measurements):
                ok_count = sum(1 for m in measurements if m["tolerance_ok"])
                lines.append(f"OK: {ok_count}  NG: {len(measurements) - ok_count}")
            self._label.setText("\n".join(lines))
            return

        if all("l_mean" in m and "label" in m for m in measurements):
            lines = []
            for m in measurements:
                tag = f"ROI{m['label']}" if m.get("manual") else f"#{m['label']}"
                line = (
                    f"{tag} → L(parlaklık)={m['l_mean']:.1f}  "
                    f"a(yeşil↔kırmızı)={m['a_mean']:.1f}  b(mavi↔sarı)={m['b_mean']:.1f}"
                )
                if "delta_e" in m:
                    status = "OK" if m.get("tolerance_ok") else "NG"
                    line += f"  ΔE={m['delta_e']:.1f} ({status})"
                if "l_min" in m:
                    line += (
                        f"\n    Aralık (dalgalanma): L[{m['l_min']:.1f},{m['l_max']:.1f}]  "
                        f"a[{m['a_min']:.1f},{m['a_max']:.1f}]  b[{m['b_min']:.1f},{m['b_max']:.1f}]"
                    )
                lines.append(line)
            lines.append(f"Toplam nesne: {len(measurements)}")
            self._label.setText("\n".join(lines))
            return

        if all("contrast" in m and "label" in m for m in measurements):
            lines = [
                f"{'ROI' + str(m['label']) if m.get('manual') else '#' + str(m['label'])} → "
                f"Kontrast={m['contrast']:.2f}  Homojenlik={m['homogeneity']:.2f}  "
                f"Enerji={m['energy']:.2f}  Korelasyon={m['correlation']:.2f}"
                for m in measurements
            ]
            lines.append(f"Toplam nesne: {len(measurements)}")
            self._label.setText("\n".join(lines))
            return

        if all("tolerance_ok" in m for m in measurements):
            ok_count = sum(1 for m in measurements if m["tolerance_ok"])
            ng_count = len(measurements) - ok_count
            lines = [f"OK: {ok_count}", f"NG: {ng_count}", f"Toplam: {len(measurements)}"]
            self._label.setText("\n".join(lines))
            return

        if len(measurements) == 1 and "model" not in measurements[0]:
            lines = [
                f"{_field_label(key)}: {_format_value(key, value)}"
                for key, value in measurements[0].items()
            ]
            self._label.setText("\n".join(lines))
            return

        self._label.setText(f"Toplam ölçüm: {len(measurements)}")
