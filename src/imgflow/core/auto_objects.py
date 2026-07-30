"""`analysis.color_props`/`analysis.texture_props`'un "Otomatik Nesne Tespiti" seçeneği için
paylaşılan yardımcı: eşikleme + bağlı bileşen analiziyle görüntüdeki ayrı nesneler (ör. tek
karede birden fazla farklı renkte ürün) bulunur, her biri için sadece o nesnenin bbox
kırpımı + piksel maskesi döner. Bu iki operatör `image` girdisine bağlı tek-girdi/tek-çıktı
operatörler olduğundan (bkz. CLAUDE.md "FAZ3" notu) ayrı bir `segment.connected_components`
+ `analysis.region_props` çifti KURMADAN, operatör kendi içinde "önce tespit et, sonra her
nesne için özellik hesapla" akışını uygulayabilmesi için buraya çıkarıldı.

Varsayılan polarite varsayımı (parlak taraf = ön plan/nesne) `segment.connected_components`
ile BİREBİR aynıdır ve canlı kamera yolunda (`color_props`/`texture_props`) DEĞİŞMEDEN
kullanılır. Opsiyonel `polarity="auto"` (SADECE `shape_matching_dialog.py`'nin tek seferlik
eğitim-zamanı kontur tespiti kullanır) hangi tarafın nesne olduğunu kırpımın topolojisinden
çıkarır — bkz. `_apply_auto_polarity`.

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


def _border_touching_fraction(binary: np.ndarray) -> float:
    """Dış çerçeveye DEĞEN bağlı bileşenlerin toplam alanının görüntüye oranı.

    Çerçeve PİKSELLERİNİ saymaktan (bkz. `_apply_auto_polarity`'nin eski ölçütü) farkı
    TOPOLOJİK olmasıdır: bir bileşen çerçeveye tek bir pikselle bile değse ALANININ TAMAMI
    sayılır. Gölge, üzerine düştüğü zeminden ayrı bir bölge değil, o zeminin (ya da nesnenin)
    bileşenine katılan bir parçadır — bu yüzden çerçeveye düşen gölge, çerçeve oylamasını
    devirdiği hâlde bu oranı kayda değer biçimde değiştirmez."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return 0.0
    edge_labels = (
        set(labels[0, :].tolist())
        | set(labels[-1, :].tolist())
        | set(labels[:, 0].tolist())
        | set(labels[:, -1].tolist())
    )
    edge_labels.discard(0)
    if not edge_labels:
        return 0.0
    area = sum(int(stats[label, cv2.CC_STAT_AREA]) for label in edge_labels)
    return area / float(binary.size)


def _apply_auto_polarity(binary: np.ndarray) -> np.ndarray:
    """"Parlak taraf = nesne" varsayımının YANLIŞ olduğu durumlarda ikili görüntüyü ters çevirir.

    Gerçek kullanıcı sorunu: "şekil bulmada otomatik nesne tespitine tıklayınca hata veriyor"
    ve "bazen arka planı ölçüyor". Kök neden, `detect_objects`'in (ve `segment.
    connected_components`'in) sabit "parlak = nesne" varsayımıydı: kullanıcının ürünleri
    (koyu kapaklar) AÇIK bir zemin üzerindeydi, bu yüzden Otsu ARKA PLANI ön plan sayıyor,
    "tespit edilen nesne" tüm ROI'yi kaplıyor ve kontur filtresi ya hiçbir şey elemiyor ya da
    (ters çevir işaretliyse) HER ŞEYİ eleyip eğitimi "yeterli kenar noktası yok" hatasıyla
    düşürüyordu.

    Ölçüt tamamen genel ve sahneden bağımsız: **arka plan, kırpımın dış çerçevesine DEĞEN
    büyük bir bölge oluşturur; çerçeve içine alınmış bir nesne oluşturmaz.** İki polarite için
    de `_border_touching_fraction` hesaplanır, hangisinde çerçeveye değen alan BÜYÜKSE o taraf
    arka plandır. Kararsız (eşit) durumda mevcut davranış (parlak = nesne) korunur.

    **Neden çerçeve PİKSELİ oylaması DEĞİL (gerçek kullanıcı raporu: "gölgeli şekillerde
    kontür çizmekte sorun"):** eski ölçüt çerçeve piksellerinin yarısından fazlası ön plansa
    ters çeviriyordu. Gölge de nesne gibi KOYU olduğundan, ROI çerçevesini kesen bir gölge
    (masaya düşen gölge şeridi, vinyetleme, eşit olmayan aydınlatma) çerçevenin koyu payını
    büyütüp bu oranı tam eşiğin etrafına düşürüyordu — gerçek yakalamalarda ölçülen değer
    0.48/0.49 idi, yani karar kıl payıyla devriliyordu ve KATASTROFİK sonuç veriyordu: arka
    plan "nesne" seçilip nesnenin kendisi eleniyordu (ölçülen IoU 0.01). ROI payı %35-100
    arasında doğru, %150+ ya da %15'te yanlış karar çıkması kullanıcının "bazen çalışıyor
    bazen çalışmıyor" gözlemini de açıklıyor.

    **Ölçüm:** 11 gerçek yakalama x nesne başına 7 farklı ROI payı = 174 durum (gölgeli/
    gradyanlı sahneler, artı regresyon kontrolü olarak parlak-nesne/koyu-zemin sahneleri).
    Eski ölçüt 150/174 (%86), bu ölçüt 171/174 (%98). Gölge şeridi olan üç sahnede iyileşme
    çok daha keskin: %28-42 -> %100. **Kalan 3 başarısızlığın tamamı EN DAR ROI'de** (nesne
    kırpımın yarısından fazlasını kaplıyor ve çerçeveye kendisi değiyor): o rejimde "hangi
    taraf arka plan" sorusu zaten iyi tanımlı değildir, ve `shape_matching_dialog.py` bu
    durumda kullanıcıya ROI'yi biraz daha GENİŞ çizmesini söyleyen notu zaten gösteriyor.
    Köşe doluluğunu ikinci bir sinyal olarak eklemek denendi: 174 durumda kazanç tek bir
    vaka (%98.3 -> %98.9) olduğundan, tek ve açıklanabilir bir ilke lehine EKLENMEDİ."""
    if _border_touching_fraction(binary) > _border_touching_fraction(cv2.bitwise_not(binary)):
        return cv2.bitwise_not(binary)
    return binary


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
    polarity: str = "bright",
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
    tamamlar.

    `polarity`: `"bright"` (varsayılan, eski davranış -- parlak taraf nesnedir), `"dark"`
    (koyu taraf nesnedir) ya da `"auto"` (bkz. `_apply_auto_polarity` -- kırpımın dış
    çerçevesini kaplayan taraf ARKA PLAN sayılır). Koyu ürün/açık zemin sahnelerinde
    varsayılan seçim arka planı "nesne" sanıyordu."""
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

    # Polarite kararı eşikleme sonucu üzerinde, kenar-tabanlı birleştirmeden ÖNCE verilir:
    # `_edge_based_foreground_mask` dış konturları doldurduğu için kırpımın tamamını
    # kaplayabilir ve çerçeve ölçütünü anlamsızlaştırır.
    if polarity == "dark":
        binary = cv2.bitwise_not(binary)
    elif polarity == "auto":
        binary = _apply_auto_polarity(binary)

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
