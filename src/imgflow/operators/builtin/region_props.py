"""Bağlı bileşenler üzerinde temel bölge analizi: alan, çevre, centroid, bbox, dairesellik.

"measurements" tabloya ek olarak, bulunan her bölgenin sınırlayıcı kutusunu ve numarasını
labels haritası üzerine çizen bir "overlay" görüntüsü de üretir; böylece bu adım seçili
kutucuğa tıklandığında sadece ham sayısal veri değil, doğrudan görülebilir bir sonuç verir."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry

_OVERLAY_COLOR = (0, 255, 0)
_TOL_OK_COLOR = (0, 255, 0)
_TOL_FAIL_COLOR = (0, 0, 255)


_TEXT_SCALE_REFERENCE_DIM = 1000.0
"""Font/çizgi kalınlığı bu referans boyuta (px, görüntünün uzun kenarı) göre ölçeklenir.
Eskiden SABİT (0.4/0.5 fontScale, 1-2px kalınlık) değerler kullanılıyordu -- 20x20'lik test
görüntülerinde iyi görünse de gerçek endüstriyel kamera çözünürlüğünde (ör. 1920x1200)
yazılar orantısız derecede küçük kalıyordu (gerçek kullanıcı raporu). `max(..., taban)`
ile küçük görüntülerde (testler dahil) eski sabit değerlerle TAM aynı sonucu üretir, sadece
büyük görüntülerde yukarı ölçeklenir."""


def draw_measurements_overlay(
    base_image: np.ndarray, measurements: list[dict[str, Any]], mm_per_px: float
) -> np.ndarray:
    """Verilen (herhangi bir) BGR görüntü ÜZERİNE ölçüm kutularını/etiketlerini çizer.

    `_render_overlay` bunu labels haritasından üretilen ikili görselle çağırır; ana pencere
    de aynı fonksiyonu canlı kameranın ham (filtrelenmemiş) karesi üzerine çizmek için
    kullanır (bkz. `ui/main_window.py` "Normal"/"İkisi Bir Arada" görünüm modları) — böylece
    ölçüm hesaplaması ile hangi görüntü üzerinde gösterildiği birbirinden bağımsızdır.

    Kalibrasyon varsa (mm_per_px>0) boyut HEM px HEM cm cinsinden yazılır -- gerçek kullanıcı
    isteği: "kalibrasyon seçili olduğu her senaryoda pixelin yanında mm de yazsın veya cm"
    (önceden kalibrasyon aktifken px değeri TAMAMEN gizleniyordu, sadece cm gösteriliyordu).
    Kalibrasyon yoksa (mm_per_px<=0) sadece px gösterilir."""
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    box_thickness = max(2, round(2 * scale_factor))
    label_font_scale = max(0.5, 0.6 * scale_factor)
    dim_font_scale = max(0.4, 0.55 * scale_factor)
    text_thickness = max(1, round(1.5 * scale_factor))

    for index, m in enumerate(measurements, start=1):
        # GERÇEK ÇÖKME (log: `KeyError: 'obb_cx'`, `main_window._compose_display_image` ->
        # buraya): bu fonksiyon `bbox_*` VE `obb_*` anahtarlarının HEPSİNİN var olduğunu
        # varsayıyordu, ama "Normal"/"İkisi Bir Arada"/"ROI Bağlamda" modları ölçümleri
        # HANGİ operatörden gelirse gelsin buraya veriyor. `analysis.color_props`/
        # `analysis.texture_props`'un nesne-başına (ya da Elle ROI) çıktısında `bbox_*` VAR
        # ama `obb_*` YOK; `geom.shape_match`/`ml.onnx_detect`'te `bbox_*` de yok. O adımlar
        # seçili haldeyken görünüm modunu değiştirmek uygulamayı düşürüyordu. Artık her ölçüm
        # SADECE gerçekten taşıdığı alanlarla çizilir: `bbox_*` yoksa atlanır, `obb_*` yoksa
        # eksene paralel bbox dikdörtgeni çizilir.
        if not all(k in m for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
            continue
        x, y, w, h = m["bbox_x"], m["bbox_y"], m["bbox_w"], m["bbox_h"]
        if "tolerance_ok" in m:
            color = _TOL_OK_COLOR if m["tolerance_ok"] else _TOL_FAIL_COLOR
        else:
            color = _OVERLAY_COLOR
        has_obb = all(k in m for k in ("obb_cx", "obb_cy", "obb_w", "obb_h", "obb_angle"))
        if has_obb:
            box = cv2.boxPoints(
                ((m["obb_cx"], m["obb_cy"]), (m["obb_w"], m["obb_h"]), m["obb_angle"])
            )
            cv2.drawContours(overlay, [np.intp(box)], 0, color, box_thickness)
        else:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, box_thickness)
        cv2.putText(
            overlay,
            f"#{index}",
            (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )
        # Boyut yazısı da aynı gerekçeyle: `obb_*` varsa (gerçek yönlü boyut) o kullanılır,
        # yoksa eksene paralel bbox boyutuna düşülür.
        dim_w, dim_h = (m["obb_w"], m["obb_h"]) if has_obb else (float(w), float(h))
        if mm_per_px > 0:
            cm_per_px = mm_per_px / 10.0
            dim_text = (
                f"{dim_w:.0f} x {dim_h:.0f} px "
                f"({dim_w * cm_per_px:.2f} x {dim_h * cm_per_px:.2f} cm)"
            )
        else:
            dim_text = f"{dim_w:.0f} x {dim_h:.0f} px"
        if "tolerance_ok" in m:
            dim_text += "  OK" if m["tolerance_ok"] else "  NG"
        cv2.putText(
            overlay,
            dim_text,
            (x, y + h + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            dim_font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )
    return overlay


@registry.register
class RegionPropsOp:
    id = "analysis.region_props"
    inputs = [PortSpec("labels", PortType.IMAGE)]
    outputs = [
        PortSpec("measurements", PortType.MEASUREMENTS),
        PortSpec(
            "overlay",
            PortType.IMAGE,
            description="Bulunan bölgelerin sınırlayıcı kutu ve numaralarıyla işaretlenmiş önizleme görseli.",
        ),
    ]
    params = [
        ParamSpec(
            "min_area",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Min. Alan (cm² kalibrasyonluyken, yoksa px²)",
            help="Bu değerden KÜÇÜK alanlı bölgeler yok sayılır (eşiklemeden kalan küçük "
            "gürültü/kırıntıların ekranı ve CSV'yi doldurmasını önlemek için). mm_per_px > 0 "
            "ise (kalibrasyon aktif) cm² olarak yorumlanır, değilse ham px² olarak. 0 = kapalı "
            "(varsayılan). Slider YOK — kalibrasyonlu/kalibrasyonsuz aynı sayısal aralığın "
            "anlamı (cm² vs binlerce px²) çok farklı olduğundan sabit bir slider aralığı "
            "ikisine birden uymuyor.",
        ),
        ParamSpec(
            "mm_per_px",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Piksel Ölçeği (mm/px)",
            help="Otomatik doldurulur: Lens Kalibrasyonu'nda checkerboard'dan üretilen "
            "deneysel netlik->mesafe modeli varsa HER karede otomatik (elle giriş gerekmez); "
            "yoksa Araçlar > Aktif Yükseklik Ayarla ile eski yöntem kullanılır. 0 ise mm "
            "alanları üretilmez, sadece piksel ölçümleri kullanılır.",
        ),
        ParamSpec(
            "tolerance_enabled",
            ParamType.BOOL,
            default=False,
            label="Tolerans Kontrolü Aktif",
            help="Opsiyonel: kapalıyken (varsayılan) hiçbir referans/spec girilmeden tüm "
            "cisimler yeşil çerçeve ve boyut etiketiyle tespit edilip ölçülür. Açılırsa, her "
            "cismin kısa/uzun kenarı aşağıdaki aralıklarla karşılaştırılır ve çerçeve spec "
            "içindeyse yeşil, dışındaysa kırmızı çizilir.",
        ),
        ParamSpec(
            "tol_short_min",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Kısa Kenar Min (mm kalibrasyonluyken, yoksa px)",
            help="mm_per_px > 0 ise mm, değilse px. 0 = alt sınır kontrol edilmez.",
            advanced=True,
        ),
        ParamSpec(
            "tol_short_max",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Kısa Kenar Max (mm kalibrasyonluyken, yoksa px)",
            help="mm_per_px > 0 ise mm, değilse px. 0 = üst sınır kontrol edilmez.",
            advanced=True,
        ),
        ParamSpec(
            "tol_long_min",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Uzun Kenar Min (mm kalibrasyonluyken, yoksa px)",
            help="mm_per_px > 0 ise mm, değilse px. 0 = alt sınır kontrol edilmez.",
            advanced=True,
        ),
        ParamSpec(
            "tol_long_max",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Uzun Kenar Max (mm kalibrasyonluyken, yoksa px)",
            help="mm_per_px > 0 ise mm, değilse px. 0 = üst sınır kontrol edilmez.",
            advanced=True,
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        labels = inputs["labels"]
        min_area = float(params.get("min_area", 0.0))
        mm_per_px = float(params.get("mm_per_px", 0.0))
        if mm_per_px > 0 and min_area > 0:
            # Kullanıcı bu değeri kalibrasyonluyken cm² olarak girer (tol_short_min/max'in
            # "mm_per_px>0 ise mm, değilse px" kuralıyla AYNI aile) — karşılaştırma ham piksel
            # alanıyla yapıldığı için cm² -> px²'ye çevriliyor (area_cm2 = area_px *
            # mm_per_px**2 / 100 <=> area_px = area_cm2 * 100 / mm_per_px**2).
            min_area = min_area * 100.0 / (mm_per_px**2)
        tolerance_enabled = bool(params.get("tolerance_enabled", False))
        tol_short_min = float(params.get("tol_short_min", 0.0))
        tol_short_max = float(params.get("tol_short_max", 0.0))
        tol_long_min = float(params.get("tol_long_min", 0.0))
        tol_long_max = float(params.get("tol_long_max", 0.0))
        max_label = int(labels.max()) if labels.size else 0

        # Her etiket için `labels == label` + `cv2.findContours`'u TÜM görüntü üzerinde
        # çalıştırmak (N etiket için N kez tam-kare taraması) konveyör bandında hızlı geçen
        # ürünleri canlı takip edemeyecek kadar yavaştı (gerçek kullanıcı raporu). Önce her
        # etiketin piksellerinin sınırlayıcı kutusunu TEK GEÇİŞTE (sort + searchsorted)
        # buluyoruz, sonra kontur/moment hesaplarını SADECE o küçük kırpım üzerinde yapıyoruz
        # (tipik bir üründe kare alanının çok küçük bir kısmı) — sonuç birebir aynı, sadece
        # çok daha az piksel dokunuluyor.
        boundaries = np.zeros(max_label + 1, dtype=np.intp)
        ys_sorted = xs_sorted = np.empty(0, dtype=np.intp)
        if max_label > 0:
            ys, xs = np.nonzero(labels)
            if ys.size:
                lbl_vals = labels[ys, xs]
                order = np.argsort(lbl_vals, kind="stable")
                lbl_sorted = lbl_vals[order]
                ys_sorted = ys[order]
                xs_sorted = xs[order]
                boundaries = np.searchsorted(lbl_sorted, np.arange(1, max_label + 2))

        measurements: list[dict[str, Any]] = []
        for label in range(1, max_label + 1):
            start, end = boundaries[label - 1], boundaries[label]
            area = float(end - start)
            if area == 0 or area < min_area:
                continue

            label_ys = ys_sorted[start:end]
            label_xs = xs_sorted[start:end]
            y0, y1 = int(label_ys.min()), int(label_ys.max())
            x0, x1 = int(label_xs.min()), int(label_xs.max())
            mask = (labels[y0 : y1 + 1, x0 : x1 + 1] == label).astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, closed=True))
            x, y, w, h = cv2.boundingRect(contour)
            x += x0
            y += y0

            (obb_cx, obb_cy), (obb_w, obb_h), obb_angle = cv2.minAreaRect(contour)
            obb_cx += x0
            obb_cy += y0

            moments = cv2.moments(mask, binaryImage=True)
            cx = moments["m10"] / moments["m00"] + x0 if moments["m00"] else float(x + w / 2)
            cy = moments["m01"] / moments["m00"] + y0 if moments["m00"] else float(y + h / 2)
            circularity = (4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0

            measurement = {
                "label": label,
                "area": area,
                "perimeter": perimeter,
                "centroid_x": cx,
                "centroid_y": cy,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "obb_cx": obb_cx,
                "obb_cy": obb_cy,
                "obb_w": obb_w,
                "obb_h": obb_h,
                "obb_angle": obb_angle,
                "circularity": circularity,
            }
            if mm_per_px > 0:
                measurement["area_mm2"] = area * mm_per_px**2
                measurement["perimeter_mm"] = perimeter * mm_per_px
                measurement["centroid_mm_x"] = cx * mm_per_px
                measurement["centroid_mm_y"] = cy * mm_per_px
                measurement["bbox_mm_x"] = x * mm_per_px
                measurement["bbox_mm_y"] = y * mm_per_px
                measurement["bbox_mm_w"] = w * mm_per_px
                measurement["bbox_mm_h"] = h * mm_per_px
                measurement["obb_mm_w"] = obb_w * mm_per_px
                measurement["obb_mm_h"] = obb_h * mm_per_px

            if tolerance_enabled:
                short_px, long_px = sorted((obb_w, obb_h))
                short_val = short_px * mm_per_px if mm_per_px > 0 else short_px
                long_val = long_px * mm_per_px if mm_per_px > 0 else long_px
                short_ok = (tol_short_min <= 0 or short_val >= tol_short_min) and (
                    tol_short_max <= 0 or short_val <= tol_short_max
                )
                long_ok = (tol_long_min <= 0 or long_val >= tol_long_min) and (
                    tol_long_max <= 0 or long_val <= tol_long_max
                )
                measurement["tolerance_ok"] = short_ok and long_ok
            measurements.append(measurement)

        overlay = self._render_overlay(labels, measurements, mm_per_px)
        return {"measurements": measurements, "overlay": overlay}

    @staticmethod
    def _render_overlay(
        labels: np.ndarray, measurements: list[dict[str, Any]], mm_per_px: float
    ) -> np.ndarray:
        base = ((labels > 0).astype(np.uint8)) * 255
        return draw_measurements_overlay(base, measurements, mm_per_px)
