"""HALCON'daki create_shape_model/find_shape_model mantığına yakın, piramit (çözünürlük
kademeleri) tabanlı kenar-gradyan yönü eşleştirmesi.

Asıl hız sırrı KABADAN İNCEYE arama: naif yaklaşım her açıda tam çözünürlükte tarama yapar
(çok yavaş); burada önce en kaba (en küçük) piramit seviyesinde az sayıda aday (x, y, açı)
bulunur, sonra SADECE bu adaylar bir alt (daha ince) seviyeye taşınıp dar bir pencerede
iyileştirilir — tam çözünürlükte hiçbir zaman tüm görüntü × tüm açı taranmaz.

Puanlama, model kenar noktalarının gradyan yönü ile hedef görüntüdeki gradyan yönü arasındaki
kosinüs benzerliğinin (yön uyumu) ortalamasıdır; mutlak gradyan büyüklüğü puanlamaya
katılmaz — bu, HALCON'un kontrast/aydınlatma değişimine karşı sağlam olan yön-tabanlı
eşleştirmesiyle aynı prensiptir. Model noktalarının ofset+açılarına göre küçük bir "yön
çekirdeği" kurup `cv2.filter2D` ile tüm konum adaylarını TEK SEFERDE (görüntü genelinde
vektörize) puanlamak, "her piksel için Python döngüsü" yaklaşımına göre çok daha hızlıdır.

v1 kapsamı: sadece dönme + öteleme (x, y, alpha). Ölçek (scale) araması kapsam dışıdır.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from imgflow.core.roi import RoiRect

_DEFAULT_NUM_LEVELS = 3
_DEFAULT_MIN_GRADIENT_FRACTION = 0.2
"""Bir pikselin 'kenar noktası' sayılması için gereken gradyan büyüklüğü eşiği, ROI
içindeki MAKSİMUM büyüklüğe ORANLA (mutlak eşik değil) — böylece farklı kontrast/aydınlatma
koşullarında elle eşik ayarlamaya gerek kalmaz."""
_DEFAULT_MAX_POINTS_PER_LEVEL = 300
_DEFAULT_ANGLE_STEP_COARSE = 5.0
_REFINE_XY_RADIUS = 3
_REFINE_ANGLE_DIVISOR = 4.0
_NMS_DIST_THRESH = 2.0
"""Küçük modeller için ALT SINIR (piksel) — asıl kullanılan eşik `_NMS_DIST_FRACTION * model
yarıçapı`'dır, bkz. `_nms_dist_thresh_for`."""
_NMS_DIST_FRACTION = 0.25
"""Modelin sınırlayıcı yarıçapına (`corners`ın merkeze uzaklığının maksimumu) ORANLA NMS mesafe
eşiği. Sabit (piksel cinsinden) bir eşik büyük modellerde işe yaramaz: aynı fiziksel nesnenin
etrafında piramit/açı adımlarından kaynaklanan birkaç piksellik farklarla oluşan adaylar
birleştirilemez ve aynı nesne birden fazla 'eşleşme' olarak döner. 0.25 deneysel olarak hem bu
tekrarları temizliyor hem de ~1.5x model-yarıçapı kadar ayrık gerçek nesneleri yanlışlıkla
birleştirmiyor (0.4-0.5'te gerçek ayrı nesneler de birleşmeye başlıyor)."""
_CANDIDATE_CARRY_MULTIPLIER = 5
_MIN_CANDIDATES_CARRIED = 10
_AUTO_CANDIDATE_CARRY_LIMIT = 60
"""`max_matches=None` (otomatik: eşiği geçen TÜM eşleşmeleri döndür) modunda, aday budama
aşamasında kullanılan sabit güvenlik tavanı — `max_matches * _CANDIDATE_CARRY_MULTIPLIER`
hesaplanamaz çünkü hedeflenen eşleşme sayısı yok."""

_OVERLAY_COLOR = (0, 255, 0)
_TEXT_SCALE_REFERENCE_DIM = 1000.0
"""Font/çizgi kalınlığı bu referans boyuta (px, görüntünün uzun kenarı) göre ölçeklenir —
`operators/builtin/region_props.py` `draw_measurements_overlay`'deki AYNI desen (gerçek
kullanıcı raporu: gerçek endüstriyel kamera çözünürlüğünde (ör. 1920x1200) eski SABİT
(0.45 fontScale, 1px kalınlık) yazılar okunamayacak kadar küçük kalıyordu). `max(..., taban)`
ile küçük görüntülerde (testler dahil) eski sabit değerlerle TAM aynı sonucu üretir."""


class ShapeMatchingError(Exception):
    """Model eğitimi/eşleştirme sırasında anlaşılır bir hata için (dialog'da doğrudan gösterilir)."""


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ShapeMatchingError(f"Desteklenmeyen görüntü şekli: {image.shape}")


def _normalize_angle(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


@dataclass
class ShapeLevel:
    """Bir piramit seviyesindeki kenar noktaları: model merkezine göre (x, y) ofseti ve
    o noktadaki gradyan yönü (radyan)."""

    points: np.ndarray  # (N, 2) float64 — [ofset_x, ofset_y]
    angles: np.ndarray  # (N,) float64 — radyan

    def to_dict(self) -> dict[str, Any]:
        return {"points": self.points.tolist(), "angles": self.angles.tolist()}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ShapeLevel:
        return ShapeLevel(
            points=np.array(data["points"], dtype=np.float64).reshape(-1, 2),
            angles=np.array(data["angles"], dtype=np.float64),
        )


@dataclass
class ShapeModel:
    """Eğitilmiş şekil modeli: en ince (index 0, tam çözünürlük) seviyeden en kaba
    (son eleman) seviyeye kadar `ShapeLevel` listesi, artı görselleştirme için modelin
    tam-çözünürlükteki sınırlayıcı kutu köşeleri (merkeze göre)."""

    levels: list[ShapeLevel]
    corners: np.ndarray  # (4, 2) float64 — tam çözünürlükte, merkeze göre köşe ofsetleri

    def to_dict(self) -> dict[str, Any]:
        return {"levels": [lvl.to_dict() for lvl in self.levels], "corners": self.corners.tolist()}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ShapeModel:
        return ShapeModel(
            levels=[ShapeLevel.from_dict(d) for d in data["levels"]],
            corners=np.array(data["corners"], dtype=np.float64).reshape(-1, 2),
        )


@dataclass(frozen=True)
class MatchResult:
    x: float
    y: float
    angle: float
    """Derece cinsinden, [-180, 180) aralığına normalize edilmiş."""
    score: float
    """[0, 1] aralığına yakın, 1'e ne kadar yakınsa yön uyumu o kadar iyi (kosinüs benzerliği ortalaması)."""


def _extract_level_points(
    image: np.ndarray,
    roi: RoiRect,
    cx: float,
    cy: float,
    min_gradient_fraction: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    row0, row1 = max(0, roi.y), min(image.shape[0], roi.y + roi.h)
    col0, col1 = max(0, roi.x), min(image.shape[1], roi.x + roi.w)
    if row1 <= row0 or col1 <= col0:
        return np.empty((0, 2)), np.empty((0,))

    patch = image[row0:row1, col0:col1].astype(np.float32)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    max_mag = float(mag.max()) if mag.size else 0.0
    if max_mag <= 0.0:
        return np.empty((0, 2)), np.empty((0,))

    mask = mag >= (min_gradient_fraction * max_mag)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return np.empty((0, 2)), np.empty((0,))

    angles = np.arctan2(gy[ys, xs], gx[ys, xs]).astype(np.float64)
    abs_x = xs.astype(np.float64) + col0
    abs_y = ys.astype(np.float64) + row0
    points = np.stack([abs_x - cx, abs_y - cy], axis=1)

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
        points = points[idx]
        angles = angles[idx]
    return points, angles


def train_shape_model(
    reference_image: np.ndarray,
    roi: RoiRect,
    num_levels: int = _DEFAULT_NUM_LEVELS,
    min_gradient_fraction: float = _DEFAULT_MIN_GRADIENT_FRACTION,
    max_points_per_level: int = _DEFAULT_MAX_POINTS_PER_LEVEL,
) -> ShapeModel:
    gray = _to_gray(reference_image)
    roi = roi.clamp(gray.shape[1], gray.shape[0])
    if roi.w < 4 or roi.h < 4:
        raise ShapeMatchingError("ROI çok küçük (en az 4x4 piksel gerekli).")

    cx = roi.x + roi.w / 2.0
    cy = roi.y + roi.h / 2.0

    levels: list[ShapeLevel] = []
    level_image = gray
    level_roi = roi
    level_cx, level_cy = cx, cy
    for level in range(num_levels):
        points, angles = _extract_level_points(
            level_image, level_roi, level_cx, level_cy, min_gradient_fraction, max_points_per_level
        )
        if points.shape[0] == 0:
            raise ShapeMatchingError(
                f"Piramit seviyesi {level}'de yeterli kenar noktası bulunamadı; ROI'de daha "
                "belirgin bir kenar/kontur olmalı ya da 'Min. Gradyan Oranı' düşürülmeli."
            )
        levels.append(ShapeLevel(points=points, angles=angles))
        if level < num_levels - 1:
            level_image = cv2.pyrDown(level_image)
            level_cx /= 2.0
            level_cy /= 2.0
            level_roi = RoiRect(
                x=int(level_roi.x // 2),
                y=int(level_roi.y // 2),
                w=max(1, level_roi.w // 2),
                h=max(1, level_roi.h // 2),
            )

    half_w, half_h = roi.w / 2.0, roi.h / 2.0
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]], dtype=np.float64
    )
    return ShapeModel(levels=levels, corners=corners)


def _rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def _build_direction_kernels(
    points: np.ndarray, angles: np.ndarray, angle_offset_deg: float
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    rotated = _rotate_points(points, angle_offset_deg)
    rotated_angles = angles + np.radians(angle_offset_deg)
    offsets = np.round(rotated).astype(int)

    min_dx, min_dy = offsets.min(axis=0)
    max_dx, max_dy = offsets.max(axis=0)
    width = int(max_dx - min_dx + 1)
    height = int(max_dy - min_dy + 1)
    center_col = int(-min_dx)
    center_row = int(-min_dy)

    kernel_cos = np.zeros((height, width), dtype=np.float32)
    kernel_sin = np.zeros((height, width), dtype=np.float32)
    rows = offsets[:, 1] + center_row
    cols = offsets[:, 0] + center_col
    np.add.at(kernel_cos, (rows, cols), np.cos(rotated_angles).astype(np.float32))
    np.add.at(kernel_sin, (rows, cols), np.sin(rotated_angles).astype(np.float32))
    return kernel_cos, kernel_sin, (center_col, center_row)


def _score_map(
    cos_map: np.ndarray, sin_map: np.ndarray, level: ShapeLevel, angle_offset_deg: float
) -> np.ndarray:
    kernel_cos, kernel_sin, anchor = _build_direction_kernels(level.points, level.angles, angle_offset_deg)
    dst_cos = cv2.filter2D(cos_map, -1, kernel_cos, anchor=anchor, borderType=cv2.BORDER_CONSTANT)
    dst_sin = cv2.filter2D(sin_map, -1, kernel_sin, anchor=anchor, borderType=cv2.BORDER_CONSTANT)
    return (dst_cos + dst_sin) / float(level.points.shape[0])


def _local_maxima(score_map: np.ndarray, threshold: float, neighborhood: int = 3) -> list[list[float]]:
    dilated = cv2.dilate(score_map, np.ones((neighborhood, neighborhood), dtype=np.uint8))
    mask = (score_map >= threshold) & (score_map >= dilated - 1e-6)
    ys, xs = np.nonzero(mask)
    return [[float(x), float(y), float(score_map[y, x])] for x, y in zip(xs, ys)]


def _nms(candidates: list[list[float]], dist_thresh: float) -> list[list[float]]:
    """Konum tabanlı (yalnızca x/y) non-max suppression: `dist_thresh` içindeki adaylardan
    sadece en yüksek skorlu tutulur.

    Açı KASITLI OLARAK bu karşılaştırmaya dahil edilmiyor — eskiden hem konum hem açı yakınlığı
    birlikte aranıyordu, ama kaba/ince arama adımlarının ürettiği küçük açı sapmalarıyla AYNI
    fiziksel nesne için birden fazla aday (ör. 45° ve 48°'de, aynı x/y'de) farklı açı eşiğini
    aşıp 'ayrı nesne' sayılıyor, aynı cisim tekrar tekrar sonuçta görünüyordu (gerçek kullanıcı
    raporu: "aynı cismi farklı açılardan tespit edip tekrar tekrar gösteriyor"). Rijit bir
    nesnenin TEK bir gerçek açısı vardır; konum tek başına yeterli bir ayraçtır — iki GERÇEKTEN
    farklı nesnenin merkezinin `dist_thresh` (modelin yarıçapına göre ölçeklenir, bkz.
    `_nms_dist_thresh_for`) kadar yakın olması pratikte imkansızdır. Bu değişiklik ayrıca
    performansı da iyileştirir: kaba arama aşamasında aynı nesne için üretilen fazladan
    adaylar artık daha agresif birleştiği için, pahalı çok-seviyeli iyileştirme aşamasına
    (`_refine_candidate`) daha AZ aday taşınır."""
    result: list[list[float]] = []
    for cand in sorted(candidates, key=lambda c: -c[3]):
        x, y, _a, _s = cand
        if any(abs(x - rx) <= dist_thresh and abs(y - ry) <= dist_thresh for rx, ry, _ra, _rs in result):
            continue
        result.append(cand)
    return result


def _coarse_search(
    cos_map: np.ndarray,
    sin_map: np.ndarray,
    level: ShapeLevel,
    angle_start: float,
    angle_extent: float,
    angle_step: float,
    relaxed_min: float,
) -> list[list[float]]:
    candidates: list[list[float]] = []
    n_steps = max(1, int(round(angle_extent / angle_step)))
    for i in range(n_steps + 1):
        angle = angle_start + i * angle_step
        if angle > angle_start + angle_extent + 1e-6:
            break
        score_map = _score_map(cos_map, sin_map, level, angle)
        for x, y, score in _local_maxima(score_map, relaxed_min):
            candidates.append([x, y, angle, score])
    return candidates


def _model_extent(level: ShapeLevel) -> float:
    """Model noktalarının merkeze göre en büyük Öklid uzaklığı — döndürme mesafeyi
    korur, bu yüzden HERHANGİ bir açıda çekirdeğin kaplayacağı yarıçapın güvenli bir üst
    sınırıdır (kırpılmış yamanın çekirdekten küçük kalıp kenar etkisiyle skoru bozmasını
    önlemek için `_refine_candidate`'ın payını (pad) buna göre ayarlaması gerekir)."""
    if level.points.size == 0:
        return 0.0
    return float(np.linalg.norm(level.points, axis=1).max())


def _refine_candidate(
    cos_map: np.ndarray,
    sin_map: np.ndarray,
    level: ShapeLevel,
    x0: float,
    y0: float,
    angle0: float,
    xy_radius: int,
    angle_window: float,
    angle_step: float,
) -> list[float]:
    h, w = cos_map.shape
    pad = xy_radius + int(np.ceil(_model_extent(level))) + 2
    y_lo, y_hi = max(0, int(y0 - pad)), min(h, int(y0 + pad) + 1)
    x_lo, x_hi = max(0, int(x0 - pad)), min(w, int(x0 + pad) + 1)
    best = [float(x0), float(y0), float(angle0), -1.0]
    if y_hi <= y_lo or x_hi <= x_lo:
        return best

    cos_patch = cos_map[y_lo:y_hi, x_lo:x_hi]
    sin_patch = sin_map[y_lo:y_hi, x_lo:x_hi]
    local_x0, local_y0 = x0 - x_lo, y0 - y_lo

    angle = angle0 - angle_window
    end_angle = angle0 + angle_window + 1e-6
    while angle <= end_angle:
        score_patch = _score_map(cos_patch, sin_patch, level, angle)
        ry_lo = max(0, int(local_y0 - xy_radius))
        ry_hi = min(score_patch.shape[0], int(local_y0 + xy_radius) + 1)
        rx_lo = max(0, int(local_x0 - xy_radius))
        rx_hi = min(score_patch.shape[1], int(local_x0 + xy_radius) + 1)
        if ry_hi > ry_lo and rx_hi > rx_lo:
            sub = score_patch[ry_lo:ry_hi, rx_lo:rx_hi]
            idx = np.unravel_index(int(np.argmax(sub)), sub.shape)
            score = float(sub[idx])
            if score > best[3]:
                best = [float(x_lo + rx_lo + idx[1]), float(y_lo + ry_lo + idx[0]), float(angle), score]
        angle += angle_step
    return best


def _nms_dist_thresh_for(model: ShapeModel) -> float:
    model_radius = float(np.linalg.norm(model.corners, axis=1).max()) if model.corners.size else 0.0
    return max(_NMS_DIST_THRESH, _NMS_DIST_FRACTION * model_radius)


def find_shape_model(
    search_image: np.ndarray,
    model: ShapeModel,
    *,
    angle_start: float = -180.0,
    angle_extent: float = 360.0,
    angle_step_coarse: float = _DEFAULT_ANGLE_STEP_COARSE,
    min_score: float = 0.5,
    max_matches: int | None = 1,
    greediness: float = 0.9,
) -> list[MatchResult]:
    """`max_matches=None` ise (otomatik mod) `min_score` üstündeki TÜM eşleşmeler döndürülür
    (dahili bir güvenlik tavanına kadar, bkz. `_AUTO_CANDIDATE_CARRY_LIMIT`); bir tam sayı
    verilirse en fazla o kadar (en yüksek skorlu) eşleşme döndürülür."""
    gray = _to_gray(search_image).astype(np.float32)
    num_levels = len(model.levels)
    nms_dist_thresh = _nms_dist_thresh_for(model)

    target_levels: list[tuple[np.ndarray, np.ndarray]] = []
    level_image = gray
    for level in range(num_levels):
        gx = cv2.Sobel(level_image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(level_image, cv2.CV_32F, 0, 1, ksize=3)
        angle_map = np.arctan2(gy, gx)
        target_levels.append((np.cos(angle_map).astype(np.float32), np.sin(angle_map).astype(np.float32)))
        if level < num_levels - 1:
            level_image = cv2.pyrDown(level_image)

    coarsest = num_levels - 1
    cos_map, sin_map = target_levels[coarsest]
    relaxed_min = min_score * max(greediness, 1e-6)

    candidates = _coarse_search(
        cos_map, sin_map, model.levels[coarsest], angle_start, angle_extent, angle_step_coarse, relaxed_min
    )
    candidates = _nms(candidates, nms_dist_thresh)
    candidates.sort(key=lambda c: -c[3])
    carry_limit = (
        _AUTO_CANDIDATE_CARRY_LIMIT
        if max_matches is None
        else max(max_matches * _CANDIDATE_CARRY_MULTIPLIER, _MIN_CANDIDATES_CARRIED)
    )
    candidates = candidates[:carry_limit]

    angle_window = angle_step_coarse
    for level in range(coarsest - 1, -1, -1):
        cos_map, sin_map = target_levels[level]
        angle_step = max(angle_window / _REFINE_ANGLE_DIVISOR, 0.1)
        candidates = [
            _refine_candidate(
                cos_map, sin_map, model.levels[level], x * 2, y * 2, angle,
                xy_radius=_REFINE_XY_RADIUS, angle_window=angle_window, angle_step=angle_step,
            )
            for x, y, angle, _score in candidates
        ]
        angle_window = max(angle_window / _REFINE_ANGLE_DIVISOR, 0.5)

    candidates = [c for c in candidates if c[3] >= min_score]
    candidates = _nms(candidates, nms_dist_thresh)
    candidates.sort(key=lambda c: -c[3])
    if max_matches is not None:
        candidates = candidates[:max_matches]
    return [
        MatchResult(x=x, y=y, angle=_normalize_angle(angle), score=score) for x, y, angle, score in candidates
    ]


@dataclass(frozen=True)
class LabeledMatch:
    """Bir eşleşmeyi, hangi modelden geldiğini (çizim için köşe/eksen bilgisi modelden farklı
    olabileceğinden) ve gösterilecek etiketi (ör. 'cıvata1') bir arada taşır — birden fazla
    modelin eşleşmelerini TEK bir overlay'de, kendi doğru köşe geometrileriyle çizebilmek için."""

    label: str
    model: ShapeModel
    match: MatchResult


def render_match_overlay(search_image: np.ndarray, entries: list[LabeledMatch]) -> np.ndarray:
    """Her eşleşmeyi kontur+eksen+SADECE numarasıyla (`entry.label`, ör. "1", "2") işaretler
    — tam (x,y,alpha) değerleri artık burada DEĞİL, ayrı bir tabloda (bkz.
    `ui/widgets/measurements_summary.py`) numaraya göre listelenir (gerçek kullanıcı isteği:
    "her şekili 1,2,3,4 diye adlandıralım, sonra 1'in x,y,alpha değerleri... tabloya yazsın").
    Görüntü üzerinde SADECE numara olması hem okunabilirliği artırır (daha az metin = daha
    büyük yazılabilir) hem de kalabalık sahnelerde çakışmayı azaltır."""
    gray = _to_gray(search_image)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    contour_thickness = max(2, round(2 * scale_factor))
    marker_size = max(12, round(12 * scale_factor))
    font_scale = max(0.6, 0.9 * scale_factor)
    text_thickness = max(1, round(2 * scale_factor))

    for entry in entries:
        model, match, label = entry.model, entry.match, entry.label
        axis_len = max(float(np.abs(model.corners).max()), 10.0)
        theta = np.radians(match.angle)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]])
        pts = (model.corners @ rot.T) + np.array([match.x, match.y])
        cv2.polylines(overlay, [pts.astype(np.int32).reshape(-1, 1, 2)], True, _OVERLAY_COLOR, contour_thickness)

        center = (int(round(match.x)), int(round(match.y)))
        cv2.drawMarker(
            overlay, center, _OVERLAY_COLOR, markerType=cv2.MARKER_CROSS,
            markerSize=marker_size, thickness=contour_thickness,
        )
        tip = (int(round(match.x + axis_len * c)), int(round(match.y + axis_len * s)))
        cv2.line(overlay, center, tip, _OVERLAY_COLOR, contour_thickness)
        cv2.putText(
            overlay,
            label,
            (center[0] + 8, max(12, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _OVERLAY_COLOR,
            text_thickness,
            cv2.LINE_AA,
        )
    return overlay
