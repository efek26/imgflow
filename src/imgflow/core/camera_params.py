"""Basler (GigE/USB3) kameraların GenICam parametrelerini 6 kategoride kataloglar.

Pylon Viewer'daki kategorilere paralel: Görüntü Kalitesi, Görüntü Formatı, Yakalama
Kontrolü, Dijital I/O, Aktarım Katmanı, Kullanıcı Setleri. Katalogdaki her giriş sadece
bir "aday" — bir kamera modelinde o node hiç bulunmayabilir ya da salt-okunur olabilir.
`CameraParameterController.build_specs()` bunu VARSAYMAZ, her node için `is_available()`/
`is_writable()`'ı runtime'da sorar (CLAUDE.md "kanıtla, tahmin etme" ilkesi) ve sadece
gerçekten kullanılabilir olanlardan `ParamSpec` üretir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imgflow.core.genicam_node_map import GenicamNodeMap
from imgflow.core.params import ParamSpec, ParamType

CATEGORY_IMAGE_QUALITY = "Görüntü Kalitesi"
CATEGORY_IMAGE_FORMAT = "Görüntü Formatı"
CATEGORY_ACQUISITION = "Yakalama Kontrolü"
CATEGORY_DIGITAL_IO = "Dijital I/O"
CATEGORY_TRANSPORT = "Aktarım Katmanı"
CATEGORY_USER_SETS = "Kullanıcı Setleri"

CATEGORIES = [
    CATEGORY_IMAGE_QUALITY,
    CATEGORY_IMAGE_FORMAT,
    CATEGORY_ACQUISITION,
    CATEGORY_DIGITAL_IO,
    CATEGORY_TRANSPORT,
    CATEGORY_USER_SETS,
]


@dataclass(frozen=True)
class GenicamParamDef:
    node_name: str
    category: str
    param_type: ParamType
    label: str
    help: str = ""
    is_command: bool = False
    """True ise node bir GenICam Command'ıdır (örn. UserSetSave) — değer okuma/yazma değil
    `execute()` ile tetiklenir; `CameraParameterController.build_specs()` bunları atlar."""
    affects_availability: bool = False
    """True ise bu node değiştiğinde AYNI kategorideki başka node'ların availability/
    writability/min-max durumu değişebilir (ör. TriggerMode=On -> TriggerSource yazılabilir
    hale gelir). Sadece bu tip node'lar değiştiğinde tüm kategori yeniden sorgulanır/kurulur;
    diğerleri (ExposureTime, Gain, Width, ...) için tek bir donanım yazma+okuma yeterlidir —
    performans için kritik (bkz. `CameraParameterController.is_structural`)."""


_PARAM_CATALOG: list[GenicamParamDef] = [
    # -- Görüntü Kalitesi ------------------------------------------------
    GenicamParamDef(
        "ExposureTime",
        CATEGORY_IMAGE_QUALITY,
        ParamType.FLOAT,
        "Pozlama Süresi (µs)",
        help="Sensörün ışığa ne kadar süre (mikrosaniye) maruz kalacağı. Artırmak görüntüyü "
        "parlatır ama hareketli nesnede bulanıklığı (motion blur) artırır.",
    ),
    GenicamParamDef(
        "Gain",
        CATEGORY_IMAGE_QUALITY,
        ParamType.FLOAT,
        "Kazanç (Gain)",
        help="Sinyalin elektronik olarak kuvvetlendirilmesi. Pozlama süresi yetersiz kaldığında "
        "görüntüyü parlatır, ancak gürültüyü de büyütür.",
    ),
    GenicamParamDef(
        "BlackLevel",
        CATEGORY_IMAGE_QUALITY,
        ParamType.FLOAT,
        "Siyah Seviyesi",
        help="Sensör çıkışına eklenen sabit ofset; en koyu tonun (siyahın) tam olarak 0 "
        "civarında kalmasını sağlamak için ince ayar amaçlıdır.",
    ),
    GenicamParamDef(
        "BalanceRatioSelector",
        CATEGORY_IMAGE_QUALITY,
        ParamType.ENUM,
        "Beyaz Dengesi Kanalı",
        help="BalanceRatio'nun hangi renk kanalına (Red/Green/Blue) uygulanacağını seçer — "
        "renkli kameralarda beyaz dengesi ayarı için gereklidir.",
        affects_availability=True,
    ),
    GenicamParamDef(
        "BalanceRatio",
        CATEGORY_IMAGE_QUALITY,
        ParamType.FLOAT,
        "Beyaz Dengesi Oranı",
        help="Seçili renk kanalının kazanç oranı; beyaz bir yüzeyin gerçekten nötr (gri/beyaz) "
        "görünmesi için kanallar arasında dengelemede kullanılır.",
    ),
    # -- Görüntü Formatı --------------------------------------------------
    GenicamParamDef(
        "PixelFormat",
        CATEGORY_IMAGE_FORMAT,
        ParamType.ENUM,
        "Piksel Formatı",
        help="Kameranın gönderdiği ham piksel kodlaması (ör. Mono8, RGB8). Pipeline'daki "
        "operatörlerin beklediği kanal sayısıyla uyumlu olmalı.",
    ),
    GenicamParamDef(
        "Width",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Genişlik",
        help="Yakalanan görüntünün piksel cinsinden genişliği (ROI/sensör alanı).",
    ),
    GenicamParamDef(
        "Height",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Yükseklik",
        help="Yakalanan görüntünün piksel cinsinden yüksekliği (ROI/sensör alanı).",
    ),
    GenicamParamDef(
        "OffsetX",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Offset X",
        help="Yakalanan alanın sensör üzerindeki yatay başlangıç konumu — Width ile birlikte "
        "sensörün bir bölgesini (ROI) seçmek için kullanılır.",
    ),
    GenicamParamDef(
        "OffsetY",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Offset Y",
        help="Yakalanan alanın sensör üzerindeki dikey başlangıç konumu — Height ile birlikte "
        "sensörün bir bölgesini (ROI) seçmek için kullanılır.",
    ),
    GenicamParamDef(
        "BinningHorizontal",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Yatay Binning",
        help="Yatayda komşu N pikseli donanımda birleştirir; çözünürlüğü düşürür ama ışık "
        "hassasiyetini ve kare hızını artırır.",
    ),
    GenicamParamDef(
        "BinningVertical",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Dikey Binning",
        help="Dikeyde komşu N pikseli donanımda birleştirir; çözünürlüğü düşürür ama ışık "
        "hassasiyetini ve kare hızını artırır.",
    ),
    GenicamParamDef(
        "DecimationHorizontal",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Yatay Decimation",
        help="Yatayda her N pikselden birini alıp diğerlerini atlar; binning'den farklı olarak "
        "ortalama almaz, veri hacmini azaltmak için kullanılır.",
    ),
    GenicamParamDef(
        "DecimationVertical",
        CATEGORY_IMAGE_FORMAT,
        ParamType.INT,
        "Dikey Decimation",
        help="Dikeyde her N pikselden birini alıp diğerlerini atlar; binning'den farklı olarak "
        "ortalama almaz, veri hacmini azaltmak için kullanılır.",
    ),
    GenicamParamDef(
        "ReverseX",
        CATEGORY_IMAGE_FORMAT,
        ParamType.BOOL,
        "Yatay Aynala",
        help="Görüntüyü yatay eksende (soldan sağa) aynalar — kamera ters/simetrik monte "
        "edildiğinde kullanılır.",
    ),
    GenicamParamDef(
        "ReverseY",
        CATEGORY_IMAGE_FORMAT,
        ParamType.BOOL,
        "Dikey Aynala",
        help="Görüntüyü dikey eksende (yukarıdan aşağı) aynalar — kamera ters monte "
        "edildiğinde kullanılır.",
    ),
    # -- Yakalama Kontrolü ------------------------------------------------
    GenicamParamDef(
        "AcquisitionFrameRate",
        CATEGORY_ACQUISITION,
        ParamType.FLOAT,
        "Kare Hızı (FPS)",
        help="Kameranın saniyede üreteceği hedef kare sayısı. Pozlama süresi bu hıza izin "
        "vermeyecek kadar uzunsa donanım fiili hızı otomatik düşürür.",
    ),
    GenicamParamDef(
        "TriggerMode",
        CATEGORY_ACQUISITION,
        ParamType.ENUM,
        "Tetikleme Modu",
        help="Off: kamera kendi hızında sürekli kare üretir. On: her kare, TriggerSource'tan "
        "gelen bir sinyal (yazılım/harici pin) ile tek tek tetiklenir — konveyör/PLC "
        "senkronizasyonunda kullanılır.",
        affects_availability=True,
    ),
    GenicamParamDef(
        "TriggerSource",
        CATEGORY_ACQUISITION,
        ParamType.ENUM,
        "Tetikleme Kaynağı",
        help="TriggerMode=On iken hangi sinyalin (Software veya bir dijital giriş pini, ör. "
        "Line1) kareyi tetikleyeceğini belirler.",
    ),
    GenicamParamDef(
        "TriggerActivation",
        CATEGORY_ACQUISITION,
        ParamType.ENUM,
        "Tetikleme Türü",
        help="Harici tetikte sinyalin hangi kenarında (RisingEdge/FallingEdge) kare "
        "yakalanacağını belirler.",
    ),
    # -- Dijital I/O -------------------------------------------------------
    GenicamParamDef(
        "LineSelector",
        CATEGORY_DIGITAL_IO,
        ParamType.ENUM,
        "Pin Seçici",
        help="Kameranın üzerindeki fiziksel giriş/çıkış pinlerinden (Line1, Line2, ...) "
        "hangisini yapılandıracağını seçer — LineMode/LineSource bu seçime uygulanır.",
        affects_availability=True,
    ),
    GenicamParamDef(
        "LineMode",
        CATEGORY_DIGITAL_IO,
        ParamType.ENUM,
        "Pin Modu (Giriş/Çıkış)",
        help="Seçili pinin Input (harici tetik/sensör okuma) mı yoksa Output (ör. flaş/strobe "
        "tetikleme, kameranın bir sinyal üretmesi) mi olarak çalışacağını belirler.",
    ),
    GenicamParamDef(
        "LineSource",
        CATEGORY_DIGITAL_IO,
        ParamType.ENUM,
        "Pin Kaynağı",
        help="Pin Output moddayken pinin hangi iç kamera sinyalini (ör. ExposureActive) dışarı "
        "vereceğini belirler — ör. pozlama sırasında bir flaş lambasını tetiklemek için.",
    ),
    # -- Aktarım Katmanı ----------------------------------------------------
    GenicamParamDef(
        "GevSCPSPacketSize",
        CATEGORY_TRANSPORT,
        ParamType.INT,
        "Paket Boyutu (MTU)",
        help="GigE Vision üzerinden gönderilen her ağ paketinin boyutu (byte). Ağ "
        "anahtarı/kart 'Jumbo Frame' destekliyorsa artırmak CPU yükünü ve paket sayısını "
        "azaltır. Sadece GigE kameralarda görünür, USB3'te bu alan yoktur.",
    ),
    GenicamParamDef(
        "GevSCPD",
        CATEGORY_TRANSPORT,
        ParamType.INT,
        "Paketler Arası Gecikme",
        help="Ardışık iki paket arasına eklenen gecikme (tick); aynı ağda birden fazla GigE "
        "kamera varsa bant genişliği çakışmasını/paket kaybını azaltmak için kullanılır. "
        "Sadece GigE kameralarda görünür.",
    ),
    GenicamParamDef(
        "DeviceLinkThroughputLimit",
        CATEGORY_TRANSPORT,
        ParamType.INT,
        "Bant Genişliği Limiti",
        help="Kameranın ağa/hatta gönderebileceği toplam veri hızının üst sınırı (byte/sn) — "
        "aynı switch'i paylaşan birden fazla kamerada her birine pay ayırmak için kullanılır.",
    ),
    # -- Kullanıcı Setleri --------------------------------------------------
    GenicamParamDef(
        "UserSetSelector",
        CATEGORY_USER_SETS,
        ParamType.ENUM,
        "Kullanıcı Profili",
        help="Kamera üzerinde (donanımda) saklı bir parametre profilini seçer (Default veya "
        "UserSet1/2). 'Kaydet' butonu o an ekrandaki tüm ayarları bu profile yazar; kamera "
        "her açıldığında hangi profilin otomatik yükleneceği UserSetDefault ile belirlenir.",
        affects_availability=True,
    ),
    GenicamParamDef(
        "UserSetSave",
        CATEGORY_USER_SETS,
        ParamType.BOOL,
        "💾 Kaydet",
        help="O an seçili olan Kullanıcı Profili'ne (UserSetSelector) mevcut TÜM kamera "
        "ayarlarını kamera belleğine (donanıma, bu uygulamaya değil) kalıcı olarak yazar.",
        is_command=True,
    ),
    GenicamParamDef(
        "UserSetDefault",
        CATEGORY_USER_SETS,
        ParamType.ENUM,
        "Açılışta Yüklenecek Set",
        help="Kamera her güç verildiğinde/açıldığında otomatik olarak hangi Kullanıcı "
        "Profili'nin yükleneceğini belirler.",
    ),
]


class CameraParameterController:
    """Katalogdaki node'ları verilen `GenicamNodeMap` üzerinden okur/yazar.

    Tamamen Qt'siz ve pypylon'suz test edilebilir (bkz. `tests/support/fake_genicam.py`).
    """

    def __init__(self, node_map: GenicamNodeMap) -> None:
        self._node_map = node_map

    def build_specs(self, category: str | None = None) -> list[ParamSpec]:
        specs: list[ParamSpec] = []
        for param_def in _PARAM_CATALOG:
            if param_def.is_command:
                continue
            if category is not None and param_def.category != category:
                continue
            node = self._node_map.get_node(param_def.node_name)
            if not node.is_available():
                continue
            specs.append(self._to_param_spec(param_def, node))
        return specs

    def current_values(self, specs: list[ParamSpec]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for spec in specs:
            node = self._node_map.get_node(spec.name)
            values[spec.name] = node.get_value()
        return values

    def set(self, name: str, value: Any) -> Any:
        """Node'a değeri yazar, sonra kameranın kabul ettiği GERÇEK değeri geri okur
        (donanım isteneni clamp'leyebilir/yuvarlayabilir).

        Görüntü Formatı kategorisindeki node'lar (Width/Height/OffsetX/OffsetY/PixelFormat/
        Binning/Decimation/Reverse) GenICam'de akış (grabbing) sürerken salt-okunur/kilitli
        olur — yazmadan önce akışı durdurup sonra yeniden başlatmak gerekir, aksi halde
        yazma sessizce/anlaşılmaz şekilde başarısız olur (daha önce yaşanan gerçek bug)."""
        param_def = next((p for p in _PARAM_CATALOG if p.node_name == name), None)
        node = self._node_map.get_node(name)
        if param_def and param_def.category == CATEGORY_IMAGE_FORMAT:
            self._node_map.stop_streaming()
            try:
                node.set_value(value)
            finally:
                self._node_map.start_streaming()
        else:
            node.set_value(value)
        return node.get_value()

    def execute(self, name: str) -> None:
        """GenICam Command node'unu tetikler (örn. UserSetSave)."""
        node = self._node_map.get_node(name)
        node.set_value(True)

    def is_structural(self, name: str) -> bool:
        """True dönerse bu node'un değişimi AYNI kategorideki başka node'ların
        availability/writability/min-max durumunu etkileyebilir — böyle bir değişiklikten
        sonra sadece o node değil, tüm kategori yeniden sorgulanmalı/kurulmalı. Diğer
        (çoğunluk) node'lar için tek bir yazma+okuma yeterlidir; her değişiklikte tüm
        kategoriyi yeniden sorgulamak (~5-10 donanım round-trip'i) gereksiz gecikmeye yol
        açar — özellikle bir slider sürüklenirken."""
        param_def = next((p for p in _PARAM_CATALOG if p.node_name == name), None)
        return bool(param_def and param_def.affects_availability)

    def command_defs(self, category: str | None = None) -> list[GenicamParamDef]:
        """Kategori içindeki, gerçekten mevcut GenICam Command node'ları (örn. UserSetSave) —
        `build_specs()` bunları ParamSpec'e çevirmez (değer değil eylem içerirler), bu yüzden
        UI'da ayrı bir buton olarak gösterilmeleri gerekir."""
        result: list[GenicamParamDef] = []
        for param_def in _PARAM_CATALOG:
            if not param_def.is_command:
                continue
            if category is not None and param_def.category != category:
                continue
            node = self._node_map.get_node(param_def.node_name)
            if not node.is_available():
                continue
            result.append(param_def)
        return result

    @staticmethod
    def _to_param_spec(param_def: GenicamParamDef, node: Any) -> ParamSpec:
        readonly = not node.is_writable()
        if param_def.param_type is ParamType.ENUM:
            choices = node.get_enum_entries() or []
            return ParamSpec(
                param_def.node_name,
                ParamType.ENUM,
                default=str(node.get_value()),
                choices=choices,
                label=param_def.label,
                help=param_def.help,
                readonly=readonly,
            )
        if param_def.param_type is ParamType.BOOL:
            return ParamSpec(
                param_def.node_name,
                ParamType.BOOL,
                default=bool(node.get_value()),
                label=param_def.label,
                help=param_def.help,
                readonly=readonly,
            )
        return ParamSpec(
            param_def.node_name,
            param_def.param_type,
            default=node.get_value(),
            min=node.get_min(),
            max=node.get_max(),
            label=param_def.label,
            help=param_def.help,
            readonly=readonly,
        )
