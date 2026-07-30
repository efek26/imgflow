"""`analysis.color_props`/`analysis.texture_props`'un "Otomatik Nesne Tespiti" seçeneği için
paylaşılan yardımcı: eşikleme + bağlı bileşen analiziyle görüntüdeki ayrı nesneler (ör. tek
karede birden fazla farklı renkte ürün) bulunur, her biri için sadece o nesnenin bbox
kırpımı + piksel maskesi döner. Bu iki operatör `image` girdisine bağlı tek-girdi/tek-çıktı
operatörler olduğundan (bkz. CLAUDE.md "FAZ3" notu) ayrı bir `segment.connected_components`
+ `analysis.region_props` çifti KURMADAN, operatör kendi içinde "önce tespit et, sonra her
nesne için özellik hesapla" akışını uygulayabilmesi için buraya çıkarıldı.

Aynı polarite varsayımı (parlak taraf = ön plan/nesne) `segment.connected_components` ile
BİREBİR aynıdır — otomatik polarite seçimi (hangi tarafın nesne olduğunu tahmin etmek)
YAPILMAZ, zaten var olan tutarlı davranış korunur.

**Parlak yansıma/aydınlatma gerçek kullanıcı sorunu:** parlak/yansıtıcı nesnelerde (ör.
balonlarda) ışığın oluşturduğu parlama, tek bir GLOBAL Otsu eşiğinin nesnenin geri kalanından
çok daha parlak olabilir. Bu iki farklı şekilde sorun çıkarır:
1. Parlama, nesnenin içinde eşik-altı kalan küçük bir "delik" oluşturup TEK nesneyi bağlı
   bileşen analizinde ikiye bölebilir ya da alanını gerçekte olduğundan küçük gösterebilir.
   `fill_holes=True` dış konturu doldurup bu deliği yutar (nesnenin gerçek dış sınırı hâlâ
   eşiği geçtiği sürece çalışır).
2. Otomatik Otsu eşiği, nesnenin geri kalanını tamamen ATLAYIP sadece parlamanın kendisini
   "nesne" sayabilir (nesne genel olarak arka plandan yeterince parlak değilse). Bunun tek
   çözümü otomatik eşiklemeyi TAMAMEN devre dışı bırakıp `threshold_mode="manual"` ile
   kullanıcının canlı önizlemede göre göre bir eşik değeri girmesidir — bu yüzden Otsu'ya
   ek olarak manuel eşik değeri seçeneği eklendi.
`close_kernel_size` (morfolojik kapama) ise parlamanın nesne SINIRINDA bıraktığı ince/pürüzlü
kopuklukları köprüler (delik doldurmadan farklı olarak nesne dışına taşan küçük kopuklukları
da düzeltir)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _edge_based_foreground_mask(gray: np.ndarray) -> np.ndarray:
    """`robust=True` (SADECE `shape_matching_dialog.py`'nin tek seferlik eğitim-zamanı kontur
    tespitinde kullanılır, canlı kamerada kullanılan varsayılan Otsu yolunu HİÇ etkilemez) için
    ikinci bir ön-plan tahmini: medyan-tabanlı otomatik Canny eşiğiyle ("auto canny" bilinen
    sezgisel yöntemi) kenar haritası çıkarılır, kırık kenarlar küçük bir dilate ile köprülenir,
    dış konturlar bulunup doldurulur. Global Otsu eşiği YEREL aydınlatma farkı/parlama/
    birbirine yakın parlaklıktaki komşu bölgelerde (ör. bitişik harfler) nesnenin tam dış
    hattını kaçırabilir -- kenar bilgisi bu durumlarda genelde hâlâ ayırt edicidir, bu yüzden
    Otsu'nun YERİNE değil ONUNLA `cv2.bitwise_or` ile BİRLEŞTİRİLİR (bkz. `detect_objects`)."""
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


@dataclass(frozen=True)
class DetectedObject:
    label: int
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray
    """`(h, w)` boyutunda bool maske — True olan pikseller bu nesneye ait (bbox kırpımı
    içinde başka bir nesnenin/arka planın pikselleri maskelenmiş olur)."""


def detect_objects(
    image: np.ndarray,
    min_area: float = 0.0,
    max_area: float = 0.0,
    threshold_mode: str = "otsu",
    threshold_value: int = 127,
    fill_holes: bool = False,
    close_kernel_size: int = 0,
    robust: bool = False,
) -> list[DetectedObject]:
    """`min_area`/`max_area`: 0 = o yönde sınır YOK (`segment.connected_components`'taki
    `max_area_ratio=0` ile AYNI "0=kapalı" kuralı). `threshold_mode="manual"` iken Otsu
    yerine sabit `threshold_value` (0-255) kullanılır — otomatik eşiğin parlama/aydınlatma
    yüzünden nesnenin tamamını YAKALAYAMADIĞI durumlarda kullanıcının canlı önizlemeye
    bakarak elle ayarlaması için.

    `robust=False` (varsayılan, `color_props`/`texture_props`'un canlı-kamera yolu) davranış/
    performans BİREBİR eskisi gibidir. `robust=True` (SADECE `shape_matching_dialog.py`'nin
    tek seferlik eğitim-zamanı kontur tespiti kullanır -- performans kısıtı yok) ek bir kenar-
    tabanlı ön-plan tahmini (bkz. `_edge_based_foreground_mask`) mevcut eşikleme sonucuyla
    BİRLEŞTİRİR -- Otsu/manuel eşiğin YEREL aydınlatma farkı/parlama yüzünden kaçırdığı sınırları
    tamamlar."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if threshold_mode == "manual":
        _, binary = cv2.threshold(gray, int(threshold_value), 255, cv2.THRESH_BINARY)
    else:
        # `segment.connected_components`'taki AYNI mantık: girdi zaten ikiliyse (ör.
        # kullanıcı bilinçli olarak önce bir Eşikleme adımı eklediyse) Otsu'yu atla, aksi
        # halde otomatik ikilileştir.
        unique_values = set(np.unique(gray).tolist())
        if unique_values <= {0, 255}:
            binary = gray
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if robust:
        binary = cv2.bitwise_or(binary, _edge_based_foreground_mask(gray))

    if close_kernel_size > 0:
        kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if fill_holes:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        binary = filled

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    objects: list[DetectedObject] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        mask = labels[y : y + h, x : x + w] == label
        objects.append(DetectedObject(label=label, x=x, y=y, w=w, h=h, mask=mask))
    return objects
