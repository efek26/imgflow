# imgflow — CLAUDE.md

Modüler endüstriyel görüntü işleme pipeline uygulaması (Python/PySide6/OpenCV), HALCON HDevelop
mantığına yakın bir masaüstü aracı. Gıda fabrikası kalite kontrol bağlamı.

## Kritik kurallar

- **Windows dosya yolları Unicode içerebilir** (ör. `Masaüstü`). `cv2.imread`/`cv2.imwrite`
  KULLANMA — bozuk/çöp path'lerde sessizce başarısız olur. Bunun yerine
  `Path.read_bytes()` + `cv2.imdecode()` ve `cv2.imencode()` + `Path.write_bytes()` kullan
  (bkz. `src/imgflow/io_utils/image_io.py`).
- **LinearPipeline tek girişli/tek çıkışlı checkbox modeli** kullanır: her operatör PRIMARY
  (index 0) portundan bir sonrakine otomatik bağlanır, port tipleri uyuşmuyorsa
  (`core/linear_pipeline.py` → `_rewire()`) bağlanmaz. Yeni operatör eklerken bunu bil — çok
  girişli/çıkışlı operatörler bu checkbox akışına doğrudan oturmaz.
- Operatör eklerken/değiştirirken **4 yeri birlikte güncelle**: operatör dosyası
  (`operators/builtin/*.py`), `operators/__init__.py` (`_register_builtins`), ve
  `ui/panels/operator_library.py` içindeki `_CATEGORY_BY_OP_ID` / `_LABEL_BY_OP_ID` /
  `_DESCRIPTION_BY_OP_ID` sözlükleri — biri unutulursa operatör "Diğer" altında açıklamasız
  görünür ya da hiç görünmez. İstisna: `custom.*` id'li özel filtreler (bkz.
  `core/custom_filters.py`) import-time değil ÇALIŞMA ZAMANINDA kaydedilir/silinir — bu
  statik sözlüklere eklenmez, `operator_library.py`'deki `category_for`/`label_for`/
  `description_for` içinde `OP_ID_PREFIX` kontrolüyle ayrıca ele alınır.
- **Özel filtreler (`core/custom_filters.py`) kullanıcı kodunu doğrudan `exec()` ile
  çalıştırır — sandbox DEĞİLDİR.** Kasıtlı bir tasarım (kullanıcının kendi masaüstünde kendi
  OpenCV kodunu yazması için); güvenlik kısıtlaması eklemeye çalışma, bunun yerine dialog'daki
  uyarıyı koru. `~/.imgflow/custom_filters/*.json` içinde saklanır. Aynı şekilde
  `core/capture_store.py` da `~/.imgflow/captures/*` içinde kalıcı veri tutar. **Her ikisi
  için de testler gerçek ev dizinine ASLA dokunmamalı** — `CUSTOM_FILTER_DIR`/`CAPTURE_DIR`'ı
  `monkeypatch` ile `tmp_path`'e yönlendir (bkz. `tests/core/test_custom_filters.py`,
  `tests/core/test_capture_store.py`). Bu unutulursa test paketi gerçek `~/.imgflow`
  altına yüzlerce dosya sızdırabilir (daha önce gerçekten oldu).
- **QTimer/tekrarlayan bir döngüden (kamera tick'i, debounce flush) ASLA modal bir
  `QMessageBox` gösterme.** Aynı hata her tetiklemede tekrar oluşursa (ör. TriggerMode
  kamerayı geçici olarak kare üretemez duruma soktuğunda) her 100ms'de bir yeni modal açılıp
  kullanıcının uygulamayı kapatmasını bile engelleyen bir "diyalog fırtınası" oluşur — gerçek
  bir kullanıcı raporuyla doğrulandı (`main_window._on_camera_tick`,
  `camera_settings_panel._flush_pending_changes`). Bu tür yollarda hata daima satır içi bir
  etikette (`status_label`, panelin kendi `_error_label`'ı) gösterilir. `cli.py`'deki global
  `sys.excepthook` (son çare ağı) da aynı istisnayı 5 saniye içinde tekrar göstermeyecek
  şekilde bastırır — ama bu ikinci bir güvenlik katmanıdır, birincil çözüm değildir.
- Görüntü işleme operatörlerinde girdi varsayımlarını (kanal sayısı, ikili mi değil mi vb.)
  KANITLA, tahmin etme — bu tür sessiz varsayım hataları (ör. Canny overlay'in tek kanallı
  girdide atlanması, Connected Components'ın ikili olmayan girdide tüm görüntüyü tek blok
  sayması) daha önce gerçek, sessiz (hatasız ama yanlış sonuç veren) bug'lara yol açtı.
- Kullanıcının açık talimatı: geliştirmeleri kendi mühendislik kararınla yap, her adım için
  onay bekleme; yine de gerçekten belirsiz/riskli kararlarda sor.

## Derleme / test komutları

Testler headless Qt gerektirir — `QT_QPA_PLATFORM` ayarlanmazsa Qt platform eklentisi
bulunamadığı için pytest sessizce/anlaşılmaz şekilde başarısız olur:

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

PowerShell'de:
```powershell
$env:QT_QPA_PLATFORM = "offscreen"; pytest -q
```

## Güncel durum (son güncelleme: 2026-07-28)

Aşağıdaki liste FAZ1/FAZ2'nin (bu oturumdan önceki tüm çalışma) çok özetlenmiş halidir —
ayrıntılar için `git log` ve ilgili dosyaların kendisine bakılabilir, burada sadece koddan
kolayca çıkarılamayacak tasarım kararları/gotcha'lar tutulur.

### FAZ1 — Temel operatör seti
Operatör registry + v1 operatörleri (ROI, renk uzayı, threshold, morfoloji, filtreleme,
connected components, region props) + reçete kaydet/yükle + toplu işlem/CSV export.

### FAZ2 — Arayüz, kalibrasyon, otomatik/model-tabanlı ölçüm
- PySide6 arayüzü: önce node-graph canvas (M3), sonra doğrusal liste + parametre paneline
  geçildi (M4) — `LinearPipeline`'ın checkbox modeli (bkz. yukarıdaki "Kritik kurallar").
- İki ayrı ölçüm yolu var, KARIŞTIRILMAMALI: `segment.connected_components` +
  `analysis.region_props` ("Bölgeler / Ölçüm") model öğretmeden çalışır ve aktif
  kalibrasyonu (mm/px) otomatik kullanır; `geom.shape_match` ("Geometrik Eşleştirme",
  Araçlar > Şekil Eşleştirme ile önce isimle bir model eğitilir) HALCON'un
  `find_shape_model`'ine yakın piramit tabanlı arama kullanır, pozu (x,y,alpha) üretir.
  `core/shape_matching.py::_nms()` konuma bakar, AÇIYA bakmaz (rijit nesnenin tek gerçek
  açısı vardır — açıya da bakmak aynı nesnenin farklı açı sapmalarında tekrar tekrar
  raporlanmasına yol açıyordu).
- Kalibrasyon zinciri (`_active_lens_profile` / `_reference_distance_mm` /
  `_plane_rectification` / deneysel `focus_model`) üç BAĞIMSIZ yoldan biriyle
  doldurulabilir (Lens Kalibrasyonu, Aktif Yükseklik Ayarla, ya da region_props'un
  "Piksel Ölçeği (mm/px)" alanına elle giriş) — sırayla yapılması GEREKMEZ, biri yeterli.
  `analysis.region_props`'taki `min_area`/`tol_short_min/max`/`tol_long_min/max` alanları
  `mm_per_px>0` iken kalibreli birim (cm²/mm), değilse ham px/px² olarak yorumlanır —
  aynı sayısal değerin anlamı kalibrasyon durumuna göre değişir, bu yüzden etiketlerinde
  birim açıkça yazar ve slider YOK.
- Performans: `region_props`/`connected_components` per-label vektörelleştirme
  (`np.argsort`+`np.searchsorted`, `np.bincount`) ile canlı bant hızında çalışır;
  `ui/widgets/image_view.py::normalize_to_uint8()` her zaman gerçek min/maks'a 0-255
  gerer (aksi halde int32 label haritası önizlemede neredeyse siyah görünüyordu).
  Overlay metin/çizgi boyutu `_TEXT_SCALE_REFERENCE_DIM=1000px` referanslı ölçeklenir
  (`region_props.py`, `shape_matching.py`, FAZ3'teki `color_props.py`/`texture_props.py`/
  `core/onnx_detection.py` de AYNI deseni kullanır).
- Opsiyonel tolerans kontrolü deseni: `tolerance_enabled` kapalıyken serbest, açılırsa
  `tolerance_ok` alanı measurements'a eklenir (`region_props`, FAZ3'teki `color_props`).
  `ui/widgets/measurements_summary.py` paneli bu alanı JENERİK olarak tanıyıp OK/NG
  sayımı gösterir — yeni bir operatör bu alanı üretirse ekstra UI kodu gerekmez.
- Uygulama genelinde sürükle-bırak: `ImageView.image_file_dropped` sinyali (opt-in,
  `setAcceptDrops(True)` çağrılmadıkça pasif) — pipeline önizlemesi, kalibrasyon kare
  listeleri, şekil eşleştirme/aydınlatma referans galerileri hepsi bunu kullanır.
- Sabit referans mesafe + düzlem rektifikasyonu (`core/plane_rectification.py`), lens
  distorsiyon düzeltmesi, checkerboard tabanlı kalibrasyon galerisi.

### FAZ3 — Aydınlatma düzeltme, renk/doku analizi, ONNX (YOLO) nesne tespiti
Kullanıcı isteğiyle eklendi: vignetting/flat-field düzeltme (HALCON `div_image` benzeri),
LAB renk analizi, GLCM/Haralick doku analizi, ve harici eğitilmiş bir ONNX modelini
(başlangıçta sadece YOLO) pipeline'a sokma. Dördü de bilinçli olarak `labels` haritasına
DEĞİL doğrudan `image`'a bağlanan tek-girdi/tek-çıktı operatörler — blob-bazlı (her ürün
için ayrı) analiz hem orijinal görüntüye hem labels haritasına aynı anda ihtiyaç duyardı,
bu da `LinearPipeline`'ın tek-girdi zincirine oturmaz; tek ürüne odaklanmak isteyen
kullanıcı önce bir `roi.region` adımı ekleyebilir.

- **`correction.flat_field`** (kategori: Aydınlatma Düzeltme) — Araçlar > Aydınlatma
  Referansı Kaydet... ile isimle kaydedilen (`io_utils/flatfield_store.py`, isimli-profil
  deseni) BOŞ/düz aydınlatılmış bir referans karenin `gain = ortalama(ref)/ref` haritasıyla
  vignetting/eşit olmayan aydınlatma düzeltilir. `strength` ile orijinal/düzeltilmiş
  karışım oranı ayarlanır.
- **`analysis.color_props`** (kategori: Renk / Doku Analizi) — tüm görüntünün L*a*b*
  ortalama/std'si + opsiyonel referans renkten ΔE76 tolerans kontrolü (`tolerance_ok`).
- **`analysis.texture_props`** (aynı kategori) — gri-seviye eş-oluşum matrisi (GLCM,
  numpy ile elle hesaplanır, scikit-image bağımlılığı YOK) tabanlı Haralick özellikleri:
  contrast/homogeneity/energy/correlation (HALCON'un gen_cooc_matrix/cooc_feature'ına
  karşılık gelir).
- **`ml.onnx_detect`** (kategori: Yapay Zeka (ONNX)) — Araçlar > ONNX Model Kaydet...
  ile `.onnx` dosyası kopyalanarak isimle kaydedilir (`io_utils/onnx_model_store.py`,
  `.onnx` + metadata `.json`); model kaydında "Model Türü" = YOLO / Sınıflandırma
  (yakında) / Segmentasyon (yakında) seçilir ama **operatör şu an SADECE `task_type=
  "yolo"` çalıştırır** — diğerleri seçilip kaydedilebilir, çalıştırılırsa net bir "henüz
  desteklenmiyor" hatası verir (crash etmez). Çekirdek algoritma `core/onnx_detection.py`
  (`shape_matching.py` ile aynı operatör-bağımsız ayrım): hem YOLOv5-tarzı (`[1,N,5+C]`,
  ayrı objectness) hem YOLOv8-tarzı (`[1,4+C,N]`, transpose, objectness yok) çıktı
  biçimlerini `class_labels` sayısından otomatik ayırt eder; NMS için YENİ bağımlılık
  gerekmeden `cv2.dnn.NMSBoxes` kullanılır. `onnxruntime` `InferenceSession`'lar dosya
  yoluna göre `_session_cache`'te önbelleğe alınır (canlı kamera tick'inde her karede
  yeniden kurmak çok yavaş olurdu). **v1 bilinçli sınırlamalar:** ön işlemede letterbox/
  padding YOK (düz `cv2.resize`, kare olmayan görüntülerde hafif en/boy bozulması olur),
  sabit RGB + [0,1] normalize varsayılır (mean/std çıkarma yok).
- `onnxruntime` YENİ, OPSİYONEL bir bağımlılık (`pyproject.toml` → `[project.optional-
  dependencies].ml`) — `core/camera_source.py`'deki pypylon korumasıyla AYNI desen
  (modül seviyesinde `try/except ImportError`, kullanım anında net Türkçe hata). Test
  fixture'ları için `onnx` (model İNŞA etme kütüphanesi, runtime değil) `dev` extra'sına
  eklendi. Yerelde geliştirmek/test etmek için: `pip install -e ".[ml,dev]"`.
- `ui/widgets/measurements_summary.py`'ye iki yeni JENERİK dal eklendi: `"model"+
  "confidence"` anahtarlı ölçümler (ONNX tespitleri) numarayla/sınıf adı/güven/kutu
  listelenir; `"model"`/`"tolerance_ok"` YOKSA ve tek satırlık bir sonuçsa (ör.
  tolerans KAPALIYKEN color_props/texture_props) `"anahtar: değer"` satırları listelenir.

**Devam — kullanıcı geri bildirimi: "yolo model yükledikten sonra sınıflandırmalar
otomatik neden gelmiyor... elimle virgüllü girmek eşleşmeme durumunda riskli".** Kök neden
bir bug'dı, eksik özellik değil: `OnnxModelDialog` kullanıcının sınıf isimlerini ELLE
(virgülle, SIRAYA duyarlı) yazmasını zorunlu kılıyordu; yazılan sıra modelin eğitildiği
gerçek `class_id` sırasıyla (0,1,2...) TAM eşleşmezse HER tespitin sınıf adı sessizce
yanlış çıkıyordu — "bozuk/hatalı gibi" görünen sonuçların gerçek nedeni buydu.
- **`core/onnx_detection.py::inspect_onnx_model(path)`** — ASLA fırlatmayan, best-effort bir
  fonksiyon: Ultralytics YOLO ihracatlarının modele gömdüğü eğitim metadata'sından
  (`session.get_modelmeta().custom_metadata_map["names"]`, `"{0: 'kusur', 1: 'saglam'}"`
  gibi Python dict-repr — JSON DEĞİL, `json.loads` başarısız olursa `ast.literal_eval`'e
  düşülür) sınıf isimlerini, VE modelin girdi tensor şeklinden (`session.get_inputs()[0].
  shape`, sadece kare + sabit boyutluysa güvenilir) giriş boyutunu okumaya çalışır. Metadata
  yoksa/bozuksa/`onnxruntime` kurulu değilse ilgili alan (ya da ikisi de) sessizce `None`
  döner — kullanıcı eski (elle) akışa düşer, hiçbir şey ÇÖKMEZ.
- `ui/dialogs/onnx_model_dialog.py::_on_choose_file`, `.onnx` dosyası seçilir seçilmez bunu
  çağırıp `Sınıf İsimleri`/`Giriş Boyutu` alanlarını OTOMATİK doldurur (alanlar hâlâ TAM
  düzenlenebilir — kilitlenmez) — ekstra tıklama/adım YOK, `imgflow: speed over guidance`
  ilkesiyle tutarlı.
- **Hatalı Sınıf (OK/NG) desteği** — kullanıcı ek olarak istedi: `ml.onnx_detect`'e opsiyonel
  `defect_classes` (virgülle ayrılmış sınıf adları) parametresi eklendi; doluyken tespit
  edilen `class_name` bu listedeyse `tolerance_ok=False` (NG), değilse `True` (OK) —
  `region_props`/`color_props`'taki `tolerance_enabled` deseninin AYNISI (boşken davranış
  DEĞİŞMEZ). `LabeledDetection`'a da opsiyonel `tolerance_ok` alanı eklendi ki
  `render_detection_overlay` kutu/metin rengini `color_props.py`'deki `_OK_COLOR`/
  `_FAIL_COLOR` deseniyle kırmızı/yeşile çevirebilsin.
- `measurements_summary.py`'nin `"model"+"confidence"` dalı (ONNX) genişletildi: satır
  sonuna `tolerance_ok` varsa "[OK]"/"[NG]" ve toplam OK/NG sayım satırı eklendi — genel
  `"tolerance_ok"` dalına ONNX ölçümleri için hiçbir zaman sıra GELMEZ (bu dal onları daha
  ÖNCE yakalar), bu yüzden OK/NG desteği doğrudan bu dala eklenmek ZORUNDAYDI.
  `main_window.py::_on_hover_measurement_changed`'in `"model" in measurement` dalı da aynı
  gerekçeyle `class_name`/`confidence`/OK-NG satırları göstermeyip sadece shape_match
  alanlarını (x/y/skor/açı) yazıyordu — o da genişletildi.
- Kamera üzerinden canlı tespit + güven skorunu ekrana yazma kısmı zaten ÇALIŞIYORDU
  (`_on_camera_tick` → `_refresh_preview` → `engine.evaluate` tamamen jenerik, seçili adım
  `ml.onnx_detect` ise her tick'te yeniden çalışır, session `_session_cache`'te önbellekli)
  — buna DOKUNULMADI, sadece kök neden (sınıf ismi eşleşmesi) düzeltildi.

### FAZ4 — Renk/doku analizinde otomatik nesne tespiti + açıklamalı sonuç paneli
Kullanıcı isteğiyle eklendi: "farklı balonları tek tek seçip özelliklerini sonuçlar
kısmına yazsın" ve "sonuçlar kısmını da açıklamalı yazsın".

- **`core/auto_objects.py::detect_objects()`** — `segment.connected_components`'taki AYNI
  Otsu-eşikleme + bağlı bileşen deseni (aynı polarite varsayımı: parlak taraf = nesne,
  otomatik polarite seçimi YOK) tek bir yardımcı fonksiyona çıkarıldı; `color_props.py` ve
  `texture_props.py`'nin YENİ "Otomatik Nesne Tespiti" (`per_object_enabled`) parametresi
  bunu kullanarak ayrı bir `segment.connected_components`/`analysis.region_props` çifti
  kurmadan, operatörün KENDİ İÇİNDE "önce tespit et, sonra her nesne için özellik hesapla"
  akışını uygular — FAZ3'teki "tek-girdi/tek-çıktı checkbox zincirine oturmaz" kısıtlaması
  hâlâ geçerli (ayrı bir labels çıktısı YOK), sadece tespit dahili kaldı.
  - **Önemli:** Otsu tek bir GRİ SEVİYE eşiği kullandığından, ayrım parlaklık farkına değil
    ARADAKİ BOŞLUĞA (bağlı bileşen sınırına) dayanır — farklı renkte ama BENZER parlaklıktaki
    nesneler (ör. farklı renkli balonlar) doğru ayrılır, ama parlaklığı birbirinden çok
    FARKLI iki nesne (biri çok koyu biri çok açık) aynı global eşikte ikisi birden "ön plan"
    sayılmayabilir (testlerde bu yüzden iki blob AYNI gri seviyeye getirilip farklı renk
    verildi — bkz. `tests/core/test_auto_objects.py`). Bu sınırlamanın kaçış yolu aşağıdaki
    `threshold_mode="manual"`.
  - `color_props.py`: her nesne için LAB kırpım+maskesi üzerinden ayrı ayrı L/a/b
    ortalama/std hesaplanır; Tolerans Kontrolü açıksa ΔE/OK-NG her nesneye AYRI uygulanır.
    Yeni `render_color_overlay_multi()` her nesneyi numaralı düz (OBB değil) kutu +
    L/a/b metniyle çizer (mevcut tek-nesne `render_color_overlay()` DEĞİŞMEDEN kaldı).
  - `texture_props.py`: GLCM her nesnenin dikdörtgen bbox KIRPIMI üzerinde hesaplanır (tam
    maskeli/piksel-bazlı GLCM değil — nesne şekli dikdörtgen değilse bbox içindeki birkaç
    arka plan pikseli hesaba karışabilir, kabul edilen bir yaklaşıklık). Yeni
    `render_texture_overlay_multi()` aynı numaralı-kutu deseniyle contrast/homogeneity/
    energy yazar.
  - İkisinde de yeni `min_object_area` (px²) parametresi küçük gürültü bileşenlerini eler.
- `ui/widgets/measurements_summary.py`: iki yeni JENERİK dal — `"l_mean"+"label"`
  (color_props'un nesne-başına çıktısı) ve `"contrast"+"label"` (texture_props'un
  nesne-başına çıktısı) her nesneyi numarasıyla AÇIKLAMALI etiketlerle listeler (ör.
  "L(parlaklık)=60.0" — ham `l_mean` anahtarı değil). Ayrıca yeni `FIELD_LABELS` sözlüğü
  tek-satırlık (tüm-görüntü, nesne tespiti KAPALIYKEN) sonuçlardaki ham anahtarları
  (`l_mean`, `contrast`...) da açıklamalı Türkçe etiketlere çeviriyor — bu değişiklik
  MEVCUT `test_single_row_measurement_without_model_lists_key_values` testinin beklediği
  metni de güncelledi (artık ham anahtar değil açıklamalı etiket bekleniyor).

**Devam — kullanıcı geri bildirimi: "otomatik nesne tespiti düzgün çalışmıyor... min max
alanını elimle girebileyim... ışıktan gelen parlaklıkları ayıramıyorum (balonda yansıyan
ışık)".** `detect_objects()` (`core/auto_objects.py`) dört yeni parametreyle genişletildi,
`color_props.py`/`texture_props.py`'nin her ikisi de aynı dört UI parametresini ekledi:
- `max_area`/`max_object_area` — `min_area` ile SİMETRİK, "0=kapalı" kuralı aynı; yanlışlıkla
  eşiklenmiş büyük arka plan/bant parçalarını eler.
- `threshold_mode`/`threshold_value` — yukarıdaki "Önemli" notundaki gerçek Otsu
  başarısızlığının (parlama nedeniyle nesnenin geneli yeterince parlak değilse Otsu SADECE
  parlamanın kendisini nesne sanabilir) ÇÖZÜMÜ: `"manual"` seçilirse Otsu tamamen atlanıp
  sabit `threshold_value` (0-255) kullanılır, kullanıcı canlı önizlemeye bakarak nesnenin
  tamamını (parlamayı değil) kapsayan bir değer girer. `tests/core/test_auto_objects.py::
  test_manual_threshold_recovers_blob_that_otsu_excludes` bunu Otsu'nun BİLEREK başarısız
  olduğu (kırmızı gray≈76 / mavi gray≈29 / siyah zemin) aynı senaryo üzerinden kanıtlıyor.
- `fill_holes` — parlamanın nesne İÇİNDE bıraktığı eşik-altı "deliği" dış kontur doldurularak
  yutar (nesnenin dış sınırı hâlâ eşiği geçtiği sürece çalışır; nesnenin TAMAMI eşiğin altında
  kalıyorsa bu YETMEZ, `threshold_mode="manual"` gerekir).
- `close_kernel_size` — eşiklenmiş ikili görüntüye bağlı-bileşen etiketlemeden ÖNCE morfolojik
  KAPAMA uygular; parlamanın nesne SINIRINDA bıraktığı ince kopuklukları köprüler (delik
  doldurmadan farklı: nesne dışına taşan kopukluklar için).
- Sıralama: eşikle → (varsa) kapama → (varsa) delik doldur → `connectedComponentsWithStats`.

**Devam — kullanıcı geri bildirimi: "otomatik tespit ve manuel seçimler beni sonuca
götürmüyor, elimle tek tek roi çizsem... aynı görselde birden fazla tespit... ROI1 ROI2
gibi ayırabilirsek... roi içindeki renk dalgalanmalarını aralık olarak versin".** Otomatik
tespitin (Otsu/manuel eşik ne kadar ayarlanırsa ayarlansın) her sahneyi ayıramadığı gerçek
kullanıcı durumları için ÜÇÜNCÜ bir mod eklendi: **Elle ROI Çiz** (`manual_roi_enabled` +
`manual_rois` JSON-string parametresi, `color_props.py`/`texture_props.py`'nin ikisinde de).
- `core/roi.py::parse_roi_list(json_str, image_w, image_h)` — `manual_rois`'i (`"[[x,y,w,h],
  ...]"`) `RoiRect` listesine çevirir, sınırlara kırpar, bozuk JSON'da sessizce boş liste
  döner (reçete elle bozulsa bile pipeline çökmemeli).
- `ui/widgets/roi_canvas.py`: `RoiCanvas`'a mevcut TEKLİ ROI çizim modunun (roi.region için)
  YANINA, birbirinden BAĞIMSIZ ikinci bir "çoklu ROI" modu eklendi (`set_multi_mode(True)`,
  `set_rois()`, `rois_changed` sinyali) — boş alana sürüklemek YENİ bir ROI ekler, mevcut bir
  ROI'nin içine sürüklemek taşır, köşe tutamacı yeniden boyutlandırır, ÜZERİNE SAĞ TIKLAMAK
  siler. İki mod aynı anda aktif olmaz, `main_window.py::_refresh_preview` seçili operatöre
  göre birini açar (`_MANUAL_ROI_OP_IDS`). roi.region'daki `is_roi_step` ile AYNI gerekçeyle
  (koordinat kayması) manuel-ROI adımı seçiliyken görünüm modundan/küçültmeden bağımsız her
  zaman tam çözünürlüklü filtrelenmiş görüntü kullanılır.
- Her çizilen ROI 1'den başlayarak numaralanır (`label`) ve measurement'a `"manual": True`
  eklenir — bu bayrak overlay'de ("#N" yerine "ROIN" yazdırır, `render_*_overlay_multi`) ve
  `measurements_summary.py`'de ("#N" yerine "ROIN" etiketi) Otomatik Nesne Tespiti çıktısından
  AYIRT ETMEK için kullanılır (iki mod aynı `"label"` alanını paylaşıyor).
- **Renk dalgalanma aralığı:** `color_props.py`'nin manuel ROI modunda (SADECE bu modda,
  Otomatik Nesne Tespiti/tüm-görüntü yolları DEĞİŞMEDİ) her kanal için `l_min/l_max`,
  `a_min/a_max`, `b_min/b_max` de hesaplanıp measurement'a eklenir — gerçek kullanıcı isteği
  "roi içindeki renk dalgalanmalarını aralık olarak versin" (tek bir ortalama, parlama/gölge
  kaynaklı renk dalgalanmasını gizleyebilir). `measurements_summary.py` bunu "Aralık
  (dalgalanma): L[..,..] a[..,..] b[..,..]" satırı olarak, overlay de aynısını görüntü
  üzerine ikinci satır olarak yazar.
- `manual_roi_enabled` açıkken `per_object_enabled` (Otomatik Nesne Tespiti) TAMAMEN yok
  sayılır — ikisi birlikte açık bırakılabilir (UI zorlamaz) ama manuel mod önceliklidir.

**Devam — kullanıcı geri bildirimi: "açılı seçeneğiyle çektiğim fotoğraflar aşağıdaki
[galeriye] gelmiyor, yandaki (Yakalananlar panelindeki) kareleri de kullanabilmeliyim,
referans fotoğrafı seçilebilmeli, galeriden seçtiğim görsellerle kalibrasyonu eğitebilmem
lazım".** `HeightScaleCalibrationDialog`'da (Araçlar > Yükseklik Kalibrasyonu) "Kare
Yakala"/sürükle-bırak eskiden TEK bir `_captured_frame`'in üzerine yazıyordu — galeri/
biriktirme YOKTU, bu yüzden "Açılı" montajla çekilen birden fazla fotoğraf kaybolmuş gibi
görünüyordu.
- `ui/widgets/capture_drop_list.py::CaptureDropListWidget` — `LensCalibrationDialog`'da
  tanımlı olan sürükle-bırak galeri widget'ı (`_CaptureDropListWidget`) buraya çıkarıldı,
  her iki diyalog da AYNI paylaşılan sınıfı kullanıyor (kod tekrarı yerine).
- `HeightScaleCalibrationDialog`'a `LensCalibrationDialog`/`ShapeMatchingDialog` ile AYNI
  desende bir kare galerisi eklendi: her "Kare Yakala"/canvas'a sürükleme/galeriye doğrudan
  sürükleme galeriye EKLENİR (üzerine yazmaz); galerideki bir küçük resme tıklamak onu
  canvas'ta aktif/referans ('★' öneki) yapar; checkbox'la işaretli TÜM kareler "Seçili
  Karelerde Tahtayı Algıla" ile TEK seferde işlenir (`_detect_board_for_frame` — hem tekli
  hem toplu çağrının paylaştığı saf `(başarılı mı, mesaj)` fonksiyonu, `_on_detect_board`
  ESKİ davranışıyla BİREBİR aynı mesaj metinlerini üretir, testler bozulmadı).
- **Açılı montaj + lens kalibrasyonu eksik uyarısı:** "Açılı" montaj matematiksel olarak
  `solvePnP` için lens kalibrasyonu (intrinsics) GEREKTİRİR; eksikse "Tahtayı Algıla"
  sessizce (küçük durum etiketinde) nokta eklemeyi reddediyordu ve kullanıcı nedenini
  anlamıyordu. Artık "Açılı" seçilir seçilmez (VE pencere her `showEvent`'te) kalıcı bir
  turuncu uyarı etiketi gösterilir + yeni `open_lens_calibration` constructor callback'i
  (`main_window.py`'den `_on_open_lens_calibration` geçirilir) ile doğrudan bu pencereden
  Lens Kalibrasyonu'nu açan bir buton eklendi — kullanıcı ayrı menüyü aramak zorunda kalmaz.

**Devam — kullanıcı geri bildirimi: "şekil eşleştirme (Bul) çok iyi çalışmıyor" (asıl sorun:
nesneler bulunamıyor/kaçırılıyor, yanlış pozisyon/pozitif DEĞİL) + "bant yükseklik farkına en
hızlı optimizasyon" (bant fiziksel olarak değişince tam kalibrasyonu tekrarlamadan hızlı
düzeltme).**
- `core/shape_matching.py::find_shape_model` — kaçırma kök nedeni: kaba (en bulanık) piramit
  seviyesindeki aday-üretme eşiği nihai `min_score`'a doğrudan bağlıydı
  (`relaxed_min = min_score*greediness`), ama o seviye `pyrDown` ile bulanıklaştığından
  GERÇEK bir eşleşme bile orada sistematik daha düşük skorlanıyor — aday üretilmeden
  kayboluyordu. Yeni `_COARSE_ACCEPT_LOOSENING=0.6` çarpanı bu eşiği nihai kabulden AYIRDI
  (nihai kabul hâlâ tam çözünürlükte `min_score` ile yapılıyor, false-positive oranı
  DEĞİŞMEDİ). Ayrıca `_REFINE_XY_RADIUS` 3→5 (kaba seviyeden ince seviyeye geçişte konum
  2x büyüdüğünde eski dar pencere gerçek konumu kaçırabiliyordu) ve
  `_DEFAULT_ANGLE_STEP_COARSE` 5.0°→3.0° (iki açı örneği arasına düşen dönüşler aday bile
  üretmiyordu). `tests/core/test_shape_matching.py::
  test_find_shape_model_recovers_noisy_off_step_angle_previously_missed` bunu, eski sabitleri
  geçici olarak geri yükleyip GERÇEKTEN kaçırıldığını da kanıtlayarak doğruluyor. Ölçek
  (scale) toleranslı arama kasıtlı olarak KAPSAM DIŞI bırakıldı (kullanıcı önceliklendirmedi).
- `ui/main_window.py::_on_adjust_height_delta` (Araçlar > "Bant/Ürün Yüksekliği Değişti...")
  — üç kalibrasyon kaynağının (`_plane_rectification`, `_reference_distance_mm`,
  `HeightScaleModel`+`_active_height_mm`) hiçbirinin İÇİNİ değiştirmeden, `_compute_
  auto_mm_per_px` ile AYNI öncelik sırasıyla SADECE mesafeye bağlı `mm_per_px` skalerini tek
  bir "yüzey kameraya X mm yaklaştı/uzaklaştı" girdisiyle günceller — tam kalibrasyon
  akışına (checkerboard/2-nokta öğretme) hiç dönmeden. `_plane_rectification` yolunda eski
  mesafe `mm_per_px_eski * fx` ile GERİ türetilip (`fx` zaten `_active_lens_profile`'dan
  mevcut) yeni mesafeden yeni `mm_per_px` hesaplanır, homografi/`output_size`
  `dataclasses.replace` ile DOKUNULMADAN kalır — homografi sadece kamera açısından kaynaklanan
  konum-bağımlı eğikliği düzeltir, bu mesafeden bağımsızdır, bu yüzden sadece skaleri
  güncellemek kullanıcının kabul ettiği bir yaklaşıklıktır (büyük yükseklik değişimlerinde/
  belirgin kamera açısında tam kalibrasyonu tekrarlamak hâlâ önerilir). Geçersiz delta (yeni
  mesafe/yükseklik ≤0 ya da `HeightScaleModel.h_camera`'yı aşıyor) durumunda state
  DEĞİŞMEDEN `status_label`'a hata yazılır (modal YOK — tek seferlik menü aksiyonu olsa da
  CLAUDE.md'nin inline-hata tercihiyle tutarlı).

**Devam — kullanıcı "uygulama için başka önerin var mı, eksik bir şey kaldı mı" diye sordu.**
Kod tabanı taramasından (TODO/eksik özellik, UI akışı, hata yönetimi) çıkan önerilerden
kullanıcı ÜÇÜNÜ seçti; NG kare otomatik arşivleme, reçete metadata/son-kullanılanlar, ölçüm
trend/SPC grafiği ise SEÇİLMEDİ (istenirse ileride ayrı bir iş olarak ele alınabilir).
- **Kamera bağlantı kopmasında otomatik toparlanma** (`ui/main_window.py`) —
  `UsbCameraSource.read()` kablo koptuğunda exception fırlatmadan SONSUZA kadar `None` döner;
  `BaslerCameraSource` gerçek kopmada exception fırlatır AMA tetikleyici modda `None` dönmesi
  NORMALDİR (kopma değildir) — bu ayrım `_on_camera_tick`'te `isinstance(..., BaslerCameraSource)`
  ile korunuyor, aksi halde tetik bekleyen bir kamera yanlışlıkla "koptu" sayılırdı. Yeni
  `_camera_reconnect_factory` (USB/GigE açılırken `_on_open_usb_camera`/`_on_open_gige_camera`'da
  set edilir, elle "Kamerayı Durdur"da temizlenir) + `_register_camera_failure`/
  `_attempt_camera_reconnect`: art arda `_CAMERA_DISCONNECT_THRESHOLD_TICKS` (~2sn) başarısızlıktan
  sonra `start_camera` (MEVCUT, DOKUNULMADAN) ile yeniden dener, başarısızsa
  `_CAMERA_RECONNECT_COOLDOWN_TICKS` (~3sn) bekleyip SÜREKLİ (asla vazgeçmeden) tekrar dener —
  kullanıcı onayı. Modal YOK, mevcut "diyalog fırtınası" kuralına (`status_label` inline) tam uyar.
- **Toplu işlem: ilerleme çubuğu + iptal** (`core/batch.py`, `ui/main_window.py`) — kod
  tabanında İLK `QThread` kullanımı. `run_batch`'e geriye dönük uyumlu (varsayılan `None`)
  `progress_callback`/`should_cancel` opsiyonel parametreleri eklendi. Yeni `_BatchWorker
  (QThread)`, `_on_run_batch`'ten çağrılırken canlı `self.graph`'ın DEĞİL, `copy.deepcopy`
  kopyasının verildiğine dikkat — `run_batch` zaten kendi `ExecutionEngine`'ini içeride kurduğu
  için TEK paylaşılan mutable state `Graph` nesnesiydi; deepcopy ile UI thread'iyle SIFIR mutable
  state paylaşılıyor, bu yüzden batch sürerken pipeline düzenlemeyi/kamerayı DURDURMAYA hiç gerek
  yok. Mevcut senkron `run_batch_process` (eski testler onu doğrudan çağırıyor) DOKUNULMADAN kaldı.
- **Pipeline Geri Al/Yinele + adım kopyala/taşı** (`ui/main_window.py`,
  `ui/panels/pipeline_steps.py`) — kapsam BİLİNÇLİ olarak SADECE pipeline adımları
  (ekleme/silme/taşıma/kopyalama) ve parametre değişiklikleri; ROI çizimi ve kalibrasyon durumu
  (mm/px, lens profili) kapsam DIŞI (kullanıcı, kalibrasyon akışını bozma riskinden çekindiği
  için onayladı). Snapshot'lar (`_snapshot_dict`: `copy.deepcopy(graph.nodes)` + `order` +
  seçili adım) `LinearPipeline.load(graph, order=...)` (MEVCUT, DOKUNULMADAN — tam olarak
  ihtiyaç duyulan "node+order'ı birlikte değiştir" ilkelini zaten sağlıyordu) ile geri
  yükleniyor. Parametre değişiklikleri (`_on_params_changed`) MEVCUT debounce mekanizmasına
  (`_param_debounce_timer`/`_flush_pending_params`, `_PARAM_DEBOUNCE_MS=120`) PİGGY-BACK
  edildi — ayrı bir undo-debounce timer'ı YOK, art arda tuş vuruşları zaten var olan debounce
  tetiklenene kadar `_pending_param_undo_snapshot`'ta bekletilip TEK bir undo adımına
  birleşiyor. Sürükle-bırak reorder için `pipeline_steps.py`'ye EKLEMELİ yeni `reordered`
  sinyali (önceki sırayı taşır) eklendi — mevcut `order_changed` DEĞİŞMEDİ.
  **Kritik yan-bulgu:** `remove_operator` bir node silince operatör kütüphanesi checkbox'ını
  KOŞULSUZ kapatıyordu; "adım kopyala" ilk kez aynı `op_id`'den İKİ örnek yaratabildiği için
  (eskiden asla olmuyordu) bu artık YANLIŞ — düzeltme: sadece o op_id'den BAŞKA örnek
  kalmadıysa checkbox kapatılır (0-veya-1-örnek senaryosunda davranış AYNI kalır).

**Devam — kullanıcı "başka ne gibi geliştirme ve optimizasyonlar yapabiliriz?" diye sordu**
(performans vurgusuyla). İki performans israfı + eksik kalıcı log altyapısı bulundu, kullanıcı
üçünü de onayladı:
- **`io_utils/shape_model_store.py`**: `load_shape_model` artık `(isim, çözümlenmiş dizin)`
  anahtarlı, dosyanın `mtime`'ıyla doğrulanan bir önbelleğe sahip —
  `onnx_detection.py::_session_cache` ile AYNI gerekçe (`geom.shape_match` canlı kamerada HER
  tick'te bunu çağırıyordu, önbellek YOKTU: disk okuma+JSON parse+numpy dizisi yeniden kurma
  her 100ms'de bir tekrarlanıyordu). ONNX modellerinin AKSİNE şekil modelleri aynı oturumda
  `ShapeMatchingDialog`'da yeniden eğitilip AYNI isimle kaydedilebiliyor — bu yüzden salt
  "bir kere yükle sonsuza dek tut" YETMEZ, `mtime` kontrolü + `save_shape_model`/
  `delete_shape_model`/`rename_shape_model`'in kendi anahtarlarını EXPLICIT silmesi (dosya
  sistemi mtime çözünürlüğü sınırlıysa bile doğru davranış için) gerekiyordu.
- **`core/shape_matching.py`**: model nesnesi artık tick'ler arasında kalıcı olduğuna göre,
  `_score_map`'in her çağrıda yeniden kurduğu yön çekirdekleri (`_build_direction_kernels`)
  artık `ShapeLevel` nesnesine bağlı, ÇALIŞMA ZAMANINDA oluşturulan bir sözlükte (dataclass
  ALANI DEĞİL — `to_dict()`/eşitliği etkilemesin diye `getattr`/lazy-init) `round(açı, 6)`
  anahtarıyla önbelleğe alınıyor. Kaba arama sabit bir açı ızgarası taradığı için bu artık
  sadece İLK tick'te kuruluyor.
- **`ui/main_window.py`**: `_on_camera_tick`'teki koşulsuz `_refresh_enum_gallery()`
  (seçili adımda bir enum parametre varsa HER seçenek için tam bir `trial_run` + pixmap
  render yapıyordu, saniyede 10 kez) `_maybe_refresh_enum_gallery()`'e sarmalandı — yeni
  `_ENUM_GALLERY_TICK_STRIDE=5` ile (`_AUTO_HEIGHT_TICK_STRIDE` ile AYNI desen) artık sadece
  ~2Hz'de çalışıyor. `_refresh_enum_gallery`'nin GERÇEK değişiklik olaylarına bağlı diğer 4
  çağrı noktası (adım seçimi, parametre debounce flush'ı, enum seçimi, ROI değişikliği)
  DOKUNULMADI.
- **Yeni `io_utils/app_log.py`**: uygulamada daha önce HİÇ kalıcı log yoktu (`sys.excepthook`
  sadece konsola basıyordu, kapalıysa iz kalmıyordu). `capture_store.py`/`shape_model_store.py`
  ile AYNI `~/.imgflow/<alt dizin>` deseninde `~/.imgflow/logs/imgflow.log`'a dönen
  (`RotatingFileHandler`, `maxBytes`/`backupCount` ile sınırlı) bir dosyaya yazıyor;
  `propagate=False` (root logger'ı/pytest'i kirletmesin), tekrar `setup_logging()` çağrısı
  ESKİ handler'ları temizleyip yeniden kuruyor (testlerin farklı `tmp_path`'lerle güvenle
  tekrar çağırabilmesi için). `cli.py::_install_excepthook` artık konsola basmanın YANINDA
  `get_logger().error(...)` de çağırıyor — çökme artık konsol kapalı olsa bile kalıcı.
  `main_window.py`'de 6 önemli olay noktasına (lens kalibrasyonu, kalibrasyon profili
  yükleme, reçete yükleme, kamera yeniden bağlanma, toplu işlem tamamlanma/hata, bant
  yüksekliği hızlı ayarlama) tek satırlık `get_logger().info/warning/error(...)` eklendi —
  hiçbir mevcut davranış/dönüş değeri değişmedi. Versiyon numarasını ARAYÜZE eklemek bu
  turda kapsam DIŞI bırakıldı (sadece log dosyasının başlangıç satırında `__version__` var).

**Devam — kullanıcı "uygulamadaki hataları tara, dialog aç/kapa ve pencere yeniden
boyutlandırma senaryolarını dene" dedi.** Statik inceleme + gerçek bir stres testiyle
(`MainWindow` kurup her dialog'u onlarca kez art arda açıp kapatan, sonra HEPSİNİ açık
bırakıp ana pencereyi/dialogları rastgele boyutlarda defalarca `resize()` eden bir betik,
`app.topLevelWidgets()` sayımıyla sızıntı doğrulanmış) iki gerçek, birbiriyle bağlantılı bug
bulundu; resize senaryolarının kendisinde (`image_view.py::_rescale`'in zoom geri-besleme
önlemi zaten vardı) çökme/uyarı ÇIKMADI.
- **Dialog sızıntısı** — `_on_open_measurement_tool` (`main_window.py`) ve toplu işlem
  `QProgressDialog`'u (`_on_run_batch`) diğer dialoglardaki (Yardım, Özel Filtre, Şekil
  Eşleştirme...) tekil-dialog YENİDEN KULLANMA desenini İZLEMEZ — ikisi de her çağrıda TAZE
  bir nesne kurar (Ölçüm Aracı taze önizleme görüntüsü, toplu işlem yeni bir `total`/worker
  için). Eski nesnede `.close()` çağrılıyordu ama bu dialoglarda (yukarıdaki "Kritik
  kurallar"daki gerekçeyle) `WA_DeleteOnClose` YOK, yani `close()` sadece GİZLER, C++
  nesnesini YOK ETMEZ — `deleteLater()` olmadan her yeniden açılış (Ölçüm Aracı'nda tam
  çözünürlüklü görüntüsüyle birlikte) MainWindow'un çocuğu olarak SONSUZA kadar bellekte
  kalır. 35 kez aç/kapa denemesinde 30+ hayalet `MeasurementToolDialog` tespit edildi. Aynı
  desen `param_form.py::_prompt_multi_select`'teki modal `QDialog`'da da vardı (daha düşük
  sıklıkta çağrıldığı için daha az ciddi, yine de düzeltildi). Üçüne de eski nesne
  değiştirilirken/`finished` sinyalinde `deleteLater()` eklendi.
- **İkincil bug (sızıntı düzeltmesi AÇIĞA ÇIKARDI)** — `_on_open_measurement_tool`'daki
  `dialog.destroyed.connect(lambda: setattr(self, "_measurement_tool_dialog", None))` kimlik
  kontrolü YAPMIYORDU: `deleteLater()` eklenmeden ÖNCE `destroyed` hiç ateşlenmediği için bu
  hiç sorun çıkarmıyordu, ama artık eski dialog gerçekten yok edilince onun GECİKMELİ
  `destroyed` sinyali, `self._measurement_tool_dialog` o sırada zaten YENİ dialog'u
  gösteriyor olsa bile onu `None`'a çeviriyordu — ekranda dialog hâlâ açıkken takip
  referansı sessizce kayboluyor, bir sonraki açılış onu ne kapatabiliyor ne de fark
  edebiliyordu (o zaman İKİNCİ bir eşzamanlı sızıntı). Düzeltme: callback artık sadece
  `self._measurement_tool_dialog is dialog` iken temizliyor. `tests/ui/test_main_window.py::
  test_reopening_measurement_tool_does_not_leak_previous_dialog` (shiboken6.isValid ile) her
  ikisini de kapsıyor — bu test, sadece sızıntı düzeltmesi (kimlik kontrolü OLMADAN)
  uygulanınca doğru şekilde BAŞKA bir yerde başarısız oluyordu, ikinci bug'ı böyle yakaladık.
- Diğer tekil-dialog'lar (Yardım/Özel Filtre/Şekil Eşleştirme/Aydınlatma/ONNX/Lens/Yükseklik)
  zaten doğru desende (`show()`la yeniden kullanma, hiç yeniden oluşturmama) — stres
  testinde 20'şer döngü + tüm dialoglar açıkken 60 rastgele resize'da hiçbiri sızmadı/
  çökmedi. `camera_settings_panel.py::set_controller`'daki `QToolBox` sayfa yeniden kurma da
  zaten doğru `deleteLater()` kullanıyor (kontrol edildi, DOKUNULMADI).

**Devam — kullanıcı "kamera açtığımda oradan fotoğraf çekme seçeneği de olsun" dedi.**
Önceden canlı kameradan kare yakalama SADECE kalibrasyon diyaloglarının (Lens/Yükseklik-Ölçek)
içindeki "Kare Yakala" butonlarıyla mümkündü — kullanıcı sadece bir fotoğraf almak için
(kalibrasyon amaçlı değil) bir kalibrasyon diyaloğu açmak zorundaydı. Yeni Kamera menüsü
aksiyonu (`_capture_photo_action` → `_on_capture_camera_photo`, kısayol `Ctrl+Shift+K`) AYNI
depoyu (`core/capture_store.py::save_capture`, `source="live"`) ve AYNI "Yakalanan Kareler"
galerisini kullanır — kalibrasyon diyaloglarındaki `_add_capture` ile birebir aynı çağrı
zinciri, sadece diyalog açmadan. `_last_camera_frame` (her tick'te ham kare, throttle'dan
ÖNCE güncellenir — `_active_mm_per_px`'in de kullandığı AYNI alan) kaynak olarak kullanıldı;
kamera yokken ya da henüz ilk tick gelmeden tıklanırsa `status_label`'a inline mesaj yazılır
(CLAUDE.md'nin modal-yasağı kuralına uyar), crash YOK. `_camera_settings_action`/
`_lens_calibration_action` ile AYNI enable/disable desenini (`start_camera`/`stop_camera`)
izler — video dosyası oynatımında da (o da `start_camera` üzerinden akar) etkin, dondurulmuş
bir kare yakalamak isteyen kullanıcı için makul bir yan fayda. `capture_gallery_panel.py`'deki
`_SOURCE_LABELS`'e `"live": "Fotoğraf"` eklendi (aksi halde galeri ham `"live"` anahtarını
gösterirdi, diğer kaynaklar zaten Türkçe etiketli).

**Devam — kullanıcı "json dosyasını yükleyemiyorum, şekil bulma modeli için upload/export'ta
json veya jpg kullanabilmem lazım" + "kamera aç/fotoğraf çek gibi özellikler görüntü panelinin
hemen üstünde olsa daha güzel olur" dedi.** İki ayrı değişiklik:
- **Şekil modeli İçe/Dışa Aktar** (`shape_matching_dialog.py`) — JPG/PNG referans görüntü
  yükleme zaten çalışıyordu (`_on_load_references`'ın filtresinde vardı); asıl eksik,
  `shape_model_store.py`'nin sadece uygulamanın KENDİ `~/.imgflow/shape_models` deposu
  içinde isimle kaydet/yükle yapması, dışarıdan (başka bir makineden/paylaşılan) bir `.json`
  dosyasını İÇE AKTARAMAMASI ve seçili bir modeli dışarıya bir `.json` olarak
  DIŞA AKTARAMAMASIYDI. `manage_row`'a iki yeni buton eklendi: "Dışa Aktar (JSON)..."
  (`_on_export_model`, seçili modeli `save_shape_model`'in yazdığı AYNI `{"name","model"}`
  zarfıyla kullanıcının seçtiği herhangi bir yola yazar) ve "İçe Aktar (JSON)..."
  (`_on_import_model`, aynı zarfı [ya da zarfsız çıplak `model` dict'ini] okuyup HEM belleğe
  HEM DE isimli depoya kaydeder — kullanıcı ayrıca 'Kaydet'e basmadan içe aktarılan model
  hemen 'Kayıtlı Modeller' listesinde ve `geom.shape_match`'in 'Model Adı' listesinde
  görünür). Bozuk/uyumsuz JSON `ShapeModel.from_dict`'in fırlatabileceği `KeyError`/
  `ValueError`/`TypeError` + `json.JSONDecodeError` hepsi yakalanıp `QMessageBox.critical`
  ile gösteriliyor, çökme YOK. Diğer isimli-profil diyalogları (ONNX/Aydınlatma/Kalibrasyon)
  aynı İçe/Dışa Aktar çiftinden HENÜZ yoksun — kullanıcı şu an için sadece şekil eşleştirmeyi
  istedi, kapsam bilerek SADECE ORAYLA sınırlı tutuldu.
- **Kamera hızlı-erişim çubuğu** (`main_window.py::_build_layout`) — "Kamera Aç ▾" (USB/
  GigE/Video alt menülü), "📷 Fotoğraf Çek" ve "Kamerayı Durdur" butonları artık `center`
  panelinde `zoom_bar`'ın da ÜSTÜNDE, görüntü panelinin hemen üzerinde — Kamera menüsündeki
  AYNI handler'ları (`_on_open_usb_camera`/`_on_open_gige_camera`/`_on_open_video_file`/
  `_on_capture_camera_photo`/`_on_stop_camera`) çağıran İKİNCİ bir giriş noktası, ayrı bir
  mantık DEĞİL. `_capture_photo_button`'ın etkin/devre dışı durumu `_capture_photo_action`
  ile `start_camera`/`stop_camera`'da TEK bir ek satırla senkron tutuluyor (iki ayrı state
  kaynağı YOK, sadece iki widget aynı iki yerde birlikte güncellenir). `_build_layout()`
  `_build_menu()`'den ÖNCE çalıştığı için (bkz. `__init__`) çubuk `_capture_photo_action`
  nesnesinin KENDİSİNİ değil sadece aynı handler METODLARINI referans alır — sıralama
  bağımlılığı yaratmadan aynı işlevi tekrar kullanır.

**Devam — kullanıcı "levels ile ilgili bir hata aldım neden" diye sordu.** Kök neden
bulundu: kullanıcı kameradan görüntü alıp filtre uygulamış, "Reçeteyi Kaydet..." ile
`kulaklık.json` kaydetmiş, sonra bu dosyayı Şekil Eşleştirme'nin (bir önceki turda eklenen)
'İçe Aktar (JSON)' düğmesine vermeye çalışmış. **Reçete (`io_utils/recipe.py::graph_to_dict`)
SADECE pipeline node/edge/param grafiğini saklar, HİÇ piksel verisi içermez** — bu yüzden
`ShapeModel.from_dict`'in beklediği `"levels"`/`"corners"` anahtarlarıyla uzaktan yakından
ilgisi yok, `KeyError: 'levels'` fırlatıp `_on_import_model`'in genel hata mesajında ham
haliyle görünüyordu. İki parçalı düzeltme:
- **`shape_matching_dialog.py::_on_import_model`** artık JSON'u önce ayrıştırıp (`"nodes"` +
  `"edges"` anahtarları varsa) bunun bir REÇETE olduğunu AYRICA tanıyor ve ham `KeyError`
  yerine "önce Dosya > Görüntüyü Dışa Aktar... ile resme çevirin" diyen özel/actionable bir
  mesaj gösteriyor (`tests/ui/test_shape_matching_dialog.py::
  test_import_recipe_json_shows_specific_guidance_not_raw_keyerror`).
- **Yeni `main_window.py::_on_export_current_image`** (Dosya menüsü, "Görüntüyü Dışa Aktar
  (PNG/JPG)...") eksik olan asıl halkayı tamamlıyor: seçili adımın o anki çıktısını
  (`_current_preview_image()` — Ölçüm Aracı/Özel Filtre'nin de kullandığı AYNI yardımcı)
  `image_io.save_image` ile gerçek bir `.png`/`.jpg` dosyasına yazıyor. Artık gerçek akış
  ŞÖYLE tamamlanıyor: kamera/görüntü → filtrele → 'Görüntüyü Dışa Aktar...' → o resmi Şekil
  Eşleştirme'nin 'Referans Görüntüler Yükle...'süne ver → ROI çiz → 'Modeli Eğit'. Görüntü
  yoksa (`_current_preview_image()` `None`) `QMessageBox.information` ile inline bilgi
  verilir, dosya seçici hiç açılmaz (bkz. `tests/ui/test_main_window.py::
  test_export_current_image_without_selection_shows_info_not_file_dialog`).

**Devam — kullanıcı "aydınlatma düzeltme uygularken görüntü hiç gelmiyor, 'reference_name'
boş olamaz hatası" bildirdi.** İki AYRI kök neden bulundu, ikisi de düzeltildi:
- **Yüzeysel neden:** `flat_field_dialog.py`'de bir referans kaydetmek, seçili
  `correction.flat_field` adımının `reference_name` parametresini OTOMATİK seçmiyordu —
  kullanıcı kaydettiğini "yükledim" sanıp filtreyi uyguluyordu ama alan hâlâ boştu. Dialog'un
  `references_changed` sinyali artık kaydedilen ismi taşıyor (`Signal(str)`, silmede boş
  string); `main_window.py::_on_flat_field_references_changed` alan BOŞSA bu ismi otomatik
  yazıyor (doluysa dokunmuyor).
- **Asıl/derin neden:** Bu düzeltme kullanıcının sorununu TAM çözmedi — "ekranda 'deneme1'
  yazıyor ama hata aynen devam ediyor" dedi. Gerçek kök neden paylaşılan
  `ui/widgets/param_form.py::set_params()`'taydı: Qt, boş bir `QComboBox`'a `addItems()` ile
  öğe eklenince (kutu daha önce hiç seçim yapılmamışsa) OTOMATİK olarak ilk öğeyi seçili
  gösteriyor — ama bu, `currentTextChanged.connect(...)`'ten ÖNCE olduğu için `_on_change`
  hiç tetiklenmiyor. Sonuç: `STRING`+`dynamic_choices` ya da `ENUM` tipi HERHANGİ bir
  parametrenin saklı değeri boş/geçersizse, ekranda dolu/seçili görünen alan `ParamForm.
  _values`'e (ve dolayısıyla `node.params`'a) hiç YANSIMIYORDU. Düzeltme: `set_params()`
  form kurulduktan SONRA her `QComboBox`'ın FİİLEN gösterdiği metni okuyup saklı değerle
  karşılaştırıyor, sapma varsa TEK bir `params_changed` ile düzeltiyor — bilerek `_on_change`
  ÜZERİNDEN DEĞİL doğrudan `.emit()` ile (bu paylaşılan widget `camera_settings_panel.py`
  tarafından da kullanılıyor, o da `field_changed`'i DONANIMA YAZMAK için dinliyor; form
  sessizce yeniden kurulurken donanıma yazma tetiklenmemeli). Bu, `geom.shape_match`/
  `ml.onnx_detect`'in aynı türden combo'larını da örtük olarak düzeltir. Test ederken
  `tests/ui/test_main_window.py`'nin `flatfield_store.FLATFIELD_DIR`'ı HİÇ izole etmediği
  (gerçek `~/.imgflow/flatfield`'i okuduğu) fark edildi — artık ilgili testler
  `shape_model_store`/`capture_store` ile AYNI şekilde `monkeypatch` kullanıyor.
- Ayrıca (aynı oturumda, ayrı bir bulgu): `correction.flat_field`'in kazanç haritası
  (`gain = ortalama(referans)/referans`) referansın koyu/gölgeli bölgelerinde SINIRSIZ
  büyüyüp görüntüyü BEYAZA doyurabiliyordu (gerçek vignetting referanslarında kaçınılmaz bir
  senaryo). Yeni `max_gain` parametresi (varsayılan 3.0) bunu sınırlıyor.

**Devam — kullanıcı "bölge ölçümü yaparken ROI'de seçebiliyor muyum, kalibrasyon bozuluyor
mu?" diye sordu.** `roi.region → segment.connected_components → analysis.region_props`
zaten checkbox zincirine tek-girdi/tek-çıktı olarak oturduğundan bu MEKANİK olarak zaten
MÜMKÜNDÜ, ve `mm_per_px` kalibrasyonu (piksel başına fiziksel boyut oranı) bir kırpmadan
ETKİLENMİYOR — alan/çevre/boy ölçümleri (cm²/mm) ROI ile birlikte kullanıldığında da
doğru. Ama gerçek bir kayma sorunu bulundu: `region_props`'un `bbox_x/y`, `centroid_x/y`,
`obb_cx/cy` alanları KIRPILMIŞ görüntüye görecedir; "Filtrelenmiş" (varsayılan) görünüm
modunda sorun yok (zaten kırpılmış görüntüyü kendi boyutunda gösteriyor), ama "Normal"/
"İkisi Bir Arada" modu TAM ÇÖZÜNÜRLÜKLÜ ham kareyi kullandığından kutular ROI'nin sol-üst
köşesi kadar KAYMIŞ görünüyordu. `main_window.py::_cumulative_roi_offset` (hedef adımdan
önceki zincirde aktif `roi.region` adım(lar)ının toplam x/y ofsetini hesaplar) +
`_shift_measurements_for_roi_offset` (SADECE bilinen konum alanlarını kaydırır — `bbox_w/h`/
`area`/`shape_match`'in `x`/`y` poz alanları gibi FARKLI/boyut alanlarına DOKUNMAZ) ile
"normal"/"both" dalında ölçümler otomatik düzeltiliyor artık. `color_props`/`texture_props`'un
nesne-başına/manuel-ROI çıktıları da AYNI `bbox_x/y` anahtarlarını kullandığı için bu
düzeltmeden örtük olarak faydalanıyor.

**Devam — kullanıcı "şekil bulma çok kasıyor, kötü çalışıyor" + "şekil öğret kısmında görüntü
çok küçük, ROI seçmekte zorlanıyorum, daha büyük pencere ve zoom in/out lazım" dedi.** İki
ayrı, birbirinden bağımsız iyileştirme:
- **Performans:** `geom.shape_match`'in kaba (tam açı taraması yapılan) arama aşaması zaten
  en pahalı kısımdı; asıl gereksiz maliyet BİRDEN FAZLA model seçiliyken (`model_names` çoklu
  seçim) ortaya çıkıyordu — her model için AYNI arama karesinin gradyan piramidi (Sobel + tüm
  piramit seviyeleri) sıfırdan yeniden hesaplanıyordu, oysa piramit sadece görüntüye bağlı,
  modelden BAĞIMSIZ. Yeni `core/shape_matching.py::build_search_pyramid()` + `find_shape_
  model(..., target_pyramid=...)` (opsiyonel, verilmezse eski davranış BİREBİR) ile
  `operators/builtin/shape_match.py::run()` artık TÜM modeller arasındaki en derin piramidi
  BİR KEZ kurup paylaşıyor (daha az seviyeli bir model için ilk N seviyenin dilimlenmesi
  yeterli — piramidin ilk N seviyesi toplam derinlikten BAĞIMSIZ aynıdır). Ayrıca daha önce
  sadece `core/shape_matching.py::find_shape_model()`'de var olan ama operatöre hiç
  BAĞLANMAMIŞ `angle_step_coarse` artık `ParamSpec` olarak dışa açıldı ("Kaba Arama Açı
  Adımı") — kullanıcı hız/doğruluk dengesini kendi eline alabilsin diye (varsayılan 3.0
  DEĞİŞMEDİ, önceki "kaçırma" düzeltmesi bozulmadı). Kullanıcıya AYRICA önerildi: "Açı
  Aralığı"nı daraltmak ve arama öncesine bir `roi.region` eklemek (görüntüyü küçültür)
  genelde doğruluktan ödün vermeden en büyük hız kazancını verir — algoritmanın kendisi
  DEĞİŞTİRİLMEDİ (bilinçli risk kararı: "kaçırma" bug'ının regresyonuna yol açabilecek
  kaba-eşik/pencere sabitlerine dokunulmadı).
- **Model Öğret dialog'u:** `ui/dialogs/shape_matching_dialog.py`'deki `RoiCanvas`
  (`ui/main_window.py`'nin ana önizlemesinde kullanılanla AYNI sınıf) hiçbir zaman bir
  `QScrollArea`'ya sarılmamış/`set_scroll_host` çağrılmamıştı — zoom API'si (`zoom_in/out/
  reset`) VARDI ama bu dialog'da hiç bağlanmamıştı, dialog da varsayılan Qt içerik-sığdırma
  boyutuyla küçük açılıyordu. Artık `ui/main_window.py`'nin ana önizleme panelindeki AYNI
  yakınlaştırma çubuğu deseni (−/+/Sığdır düğmeleri + yüzde etiketi, `QScrollArea` içine
  sarılmış canvas) buraya da eklendi, dialog `1000x850` ile açılıyor.

**Devam — kullanıcı "aydınlatma özelliği de güzel çalışmıyor, gölgeleri yok etmiyor, sadece
dışarıdaki alanı düzeltiyor" dedi.** Bu bug DEĞİL, `correction.flat_field`'in referans-tabanlı
yönteminin YAPISAL bir sınırlamasıydı: referans BOŞ (ürünsüz) bir kareden alındığından, ürünün
KENDİ gölgesi referansta hiç YOKTUR — `gain = ortalama(ref)/ref` düz bir referansla her yerde
~1 üretip görüntüyü DEĞİŞTİRMEDEN bırakır, gölgeye dokunamaz (matematiksel olarak imkansız,
referansta olmayan bir bilgiyi düzeltemez). Çözüm: yeni `mode` parametresi ("Referans"
[varsayılan, eski davranış BİREBİR] / "Yerel/Dinamik") eklendi. "Yerel/Dinamik" modda referans
GEREKMEZ — `cv2.GaussianBlur(gray, blur_radius)` ile HER karenin KENDİSİNDEN o an için bir
yerel arka plan tahmini üretilir (klasik "rolling-ball"/yerel aydınlatma düzleştirme
tekniğinin basit hali) ve AYNI `gain`/`max_gain`/`strength` boru hattından geçirilir — bu
tahmin HER karede yeniden hesaplandığından, o karede o an bulunan ürün gölgesini de (gölge
`blur_radius`'tan daha YAVAŞ/geniş bir parlaklık değişimiyse) kapsar. Test
(`test_dynamic_local_mode_flattens_a_local_shadow_reference_mode_cannot_touch`) bunu somut
sayılarla kanıtlıyor: aynı sentetik gölgeli sahnede "Referans" modu gölgeyi 100 → 99 (hiç
değişmedi) bırakırken "Yerel/Dinamik" mod 100 → ~136'ya (belirgin iyileşme, mükemmel değil)
çıkarıyor. `blur_radius` ürünün detaylarından BÜYÜK ama gölge genişliğinden KÜÇÜK seçilmeli
(yardım metninde açıklandı) — çok küçükse ürünün kendisi "arka plan" sayılıp silinir.

**Devam — kullanıcı "aydınlatma yaparken HALCON'da kod kısmında bazı renk ayarları var
onları da ekleyebilir miyiz?" dedi; netleştirince kastı HALCON'un `div_image(Image1, Image2,
Mult, Add)` operatöründeki kodda elle değiştirilebilen `Mult`/`Add` sayısal değerleriymiş
(`Sonuç = (Image1/Image2)*Mult + Add`).** `correction.flat_field` zaten TAM bu formülü
uyguluyordu (`gain = ortalama/arkaplan`, `corrected = image*gain`) ama `Mult`/`Add` otomatik
türetiliyordu (ortalama, sıfır) — kullanıcının HALCON'daki gibi elle ince ayar yapabileceği
AYRI sayısal alanlar yoktu. `flat_field.py`'ye iki yeni `ParamSpec` eklendi: `mult` (Çarpan,
varsayılan 1.0) ve `add` (Ekleme, varsayılan 0.0) — otomatik kazanç haritasının ÜZERİNE
uygulanıyor (`corrected = image*gain*mult + add`), varsayılan değerleri önceki davranışı
DEĞİŞTİRMİYOR (`test_default_mult_add_do_not_change_existing_behavior`). Hem "reference" hem
"dynamic_local" modunda AYNI şekilde geçerli (ikisi de aynı `gain`/`strength` boru hattından
geçiyor). Kapsam bilinçli olarak SADECE bu iki sayısal alanla sınırlı tutuldu — kanal-bazlı
(R/G/B ayrı) renk kayması düzeltmesi ya da otomatik beyaz dengesi (gray-world) AYRI, daha
büyük bir özellik olurdu ve kullanıcı netleştirirken bunu istemediğini belirtti.

**Devam — kullanıcı "şekil öğretme/bulmada ROI yerine çizdiğimiz konturu öğretsek nasıl
olur" diye sordu; "otomatik kontur maskesi" ve "elle serbest kontur çizme" seçeneklerinden
"ikisi birden" istendi.** Kök sorun: `train_shape_model` (`core/shape_matching.py`) ROI
dikdörtgeni içindeki TÜM güçlü-gradyan pikselini (nesnenin gerçek dış hattı olsun olmasın)
model noktası yapıyordu — ROI içindeki arka plan/bant kenarı/komşu nesne parçası da modele
gürültü olarak karışabiliyordu. `ShapeModel`/`find_shape_model`/NMS/overlay SADECE düz
`(x,y)` nokta listesi kullandığından (ROI'nin şekli hiçbir yerde saklanmıyor), TEK bir
çekirdek değişiklik ikisini de çözdü:
- `train_shape_model(..., mask: np.ndarray | None = None)` — tam-görüntü boyutunda bool
  maske, `_extract_level_points`'teki mevcut gradyan-eşiği maskesine `&` ile eklenir;
  piramit döngüsünde her `cv2.pyrDown` adımında maske de `cv2.resize(..., INTER_NEAREST)`
  ile aynı boyuta indirilir. `mask=None` davranışı BİREBİR eskisi gibi (geriye dönük uyumlu).
- **Yol A — "Konturu Otomatik Algıla" checkbox'ı** (`ShapeMatchingDialog`, sadece Dikdörtgen
  modda): çizilen ROI kırpımında `core/auto_objects.py::detect_objects()` (Otomatik Nesne
  Tespiti ile AYNI Otsu+bağlı-bileşen fonksiyonu, YENİ kod YOK) çalıştırılır, EN BÜYÜK
  maskeli nesne seçilir (`o.mask.sum()`). Nesne bulunamazsa (parlak/karmaşık sahne) sessizce
  `mask=None`'a düşülür + inline not ("Kontur bulunamadı, tüm ROI kullanıldı") — çökme YOK.
- **Yol B — "Serbest Kontur (Poligon)" çizim modu**: `RoiCanvas`'a üçüncü bir `set_shape
  ("POLYGON")` modu eklendi (RECT/CIRCLE ile AYNI anahtarlama deseni) — kapanmamışken sol
  tık nokta ekler/sağ tık son noktayı geri alır, `close_polygon()`/`clear_polygon()` (yeni
  butonlar) ile kapatılır/sıfırlanır, kapandıktan sonra bir köşeye yakın sürükleme onu taşır.
  Yeni `polygon_changed` sinyali (`list[tuple[int,int]]`, GÖRÜNTÜ koordinatında) diğer ROI
  sinyalleriyle AYNI "tam listeyi yayınla" deseninde. `ShapeMatchingDialog._on_train()`
  poligon kapalı/≥3 nokta değilse inline hata verip eğitime hiç girmez; kapalıysa
  `cv2.fillPoly` ile maske kurup roi=poligonun bounding box'ı olarak eğitir.
- **Maske genişletme payı:** her iki yolda da üretilen maske `_dilate_mask()` (2px, `cv2.
  dilate`) ile genişletiliyor — kullanıcının elle çizdiği kontur ya da Otsu'nun bulduğu nesne
  sınırı, kenarın gerçek Sobel gradyan tepe noktasından birkaç piksel içeride/dışında kalırsa
  bu pay olmadan nesnenin KENDİ dış hattı bile maskeden dışlanıp eğitim gereksiz yere
  başarısız olabiliyordu.
- Kapsam bilinçli olarak DAR tutuldu: `ShapeModel`/`shape_model_store`/`find_shape_model`/
  NMS/overlay/`operators/builtin/shape_match.py` hiç değişmedi (agent taramasıyla
  doğrulandı: sadece düz nokta listesi görüyorlar, ROI/maskenin şeklinden habersizler).

**Devam — kullanıcı "arka planı otomatik ele seçeneğine tıklayınca elediği arka planı
göremiyorum, onu da görmek istiyorum" dedi.** "Konturu Otomatik Algıla" checkbox'ı sadece
eğitim SONUCUNU (nokta sayısı) etkiliyordu, işaretleyince neyin elendiğini görsel olarak
göstermiyordu. `RoiCanvas`'a yeni `set_contour_preview(excluded_mask)` eklendi (SADECE RECT
modunda çizilir): `excluded_mask` (GÖRÜNTÜ boyutunda bool dizi, True=elenecek) `_paint_
contour_preview`'da bir `QImage` (RGBA, `_EXCLUDED_OVERLAY_RGBA` yarı saydam kırmızı) olarak
`_display_rect()` üzerine çizilir. `ShapeMatchingDialog._refresh_contour_preview()` checkbox
işaretliyken (ve Dikdörtgen modundayken) `_build_auto_contour_mask`'ın ürettiği (zaten
`_dilate_mask` ile genişletilmiş, eğitimde KULLANILACAK olanla BİREBİR AYNI) maskeyi alıp
`excluded = (ROI içi) & ~mask` hesaplayarak canvas'a besler; checkbox'ın `toggled` sinyali,
`_on_roi_changed` (ROI taşınınca/boyutlandırılınca), `_use_reference_image` (referans
değişince) ve `_on_draw_mode_changed` (Poligon moduna geçilince temizlenir) hepsi bunu canlı
tetikler. Nesne bulunamazsa önizleme temizlenir + yeni `_contour_preview_label`'a inline not
("Kontur bulunamadı...") yazılır — `_train_status_label`'dan AYRI tutuldu ki eğitim sonucu
mesajıyla karışmasın.

**Devam — kullanıcı "otomatik kontür seçebiliyor fakat seçtiği kontürü siliyor, onu kullanmak
istiyorum, duruma göre silmek de isteyebilirim; seçtiği kontürün içini mi dışını mı seçeceğimi
seçebilmek istiyorum" dedi.** Önceden hem "Konturu Otomatik Algıla" (RECT) hem Serbest Kontur
(POLYGON) yolu SADECE "kontur içi kullanılır, dışı elenir" semantiğini destekliyordu — ters
yönü (konturun/poligonun DIŞINI kullanıp İÇİNİ elemek) seçme imkanı yoktu.
- Yeni `ShapeMatchingDialog._invert_mask_checkbox` ("Kontur Dışını Kullan (ters çevir)") her
  iki çizim modunda da (auto-contour checkbox'ının aksine `_set_draw_mode_widgets_visible` ile
  gizlenmez) görünür — hangi mekanizma aktifse (RECT'te auto-contour, POLYGON'da kapatılmış
  poligon) onu etkiler, hiçbiri yoksa etkisizdir.
- `_build_auto_contour_mask` artık DİLATE EDİLMEMİŞ ham "kontur içi=True" maskeyi döndürüyor
  (önceki davranış: doğrudan dilate edilmiş halini döndürüyordu). Yeni ortak
  `_finalize_mask(inside_mask)` fonksiyonu hem bu yolun hem poligon `fillPoly` maskesinin ORTAK
  son adımı: `_invert_mask_checkbox` işaretliyse ÖNCE `~inside_mask` ile ters çevirir, SONRA
  `_dilate_mask` uygular — sıralama kasıtlı: dilate her zaman TUTULACAK tarafın sınırını öteki
  tarafa doğru birkaç piksel genişletir (kullanıcının çizdiği/Otsu'nun bulduğu sınır gerçek
  Sobel kenarından 1-2px sapmış olsa bile en yakın kenar noktaları dışlanmasın diye); ters
  çevirmeden ÖNCE dilate edilseydi bu pay her zaman AYNI (eski) tarafa sızardı, ters modda
  gerçek kontur sınırındaki noktalar gereksiz yere dışlanırdı.
- `_refresh_contour_preview` (RECT canlı önizleme) ve `_on_train` (her iki mod) ikisi de artık
  `_finalize_mask` üzerinden geçiyor; önizleme etiketi ("içi"/"dışı") de invert durumuna göre
  güncelleniyor. `tests/ui/test_shape_matching_dialog.py::
  test_invert_mask_checkbox_flips_which_region_the_preview_marks_as_excluded` ve
  `test_invert_mask_checkbox_trains_on_background_instead_of_detected_object` bunu canlı
  önizleme + gerçek eğitim çıktısı üzerinden doğruluyor;
  `test_invert_mask_checkbox_flips_finalize_mask_for_both_rect_and_polygon_paths` ise
  `_finalize_mask`'i doğrudan (kontur sınırına yakın Sobel/dilate etkileşiminin geometriyi
  belirsizleştirdiği tam eğitim ardışık düzeni yerine) test ederek POLYGON yolunu da kapsıyor
  (poligon modunda RECT'teki gibi canlı önizleme YOK, bkz. `test_switching_to_polygon_mode_
  clears_contour_preview`).
- Kullanıcının "seçtiği kontürü siliyor" ifadesi bir bug değil, önceki tek-yönlü ("sadece içi
  kullanılır") tasarımın kendisiydi — düzeltme yeni bir seçenek eklemek oldu, geriye dönük
  varsayılan davranış (checkbox işaretsizken) DEĞİŞMEDİ.

**Devam — kullanıcı "uygulama açıldığında sol taraftaki operatör kütüphanesi alt başlıkları
gözükecek şekilde açık olmasın, ana başlıklar olsun, alt başlıklar tıklayınca açılsın" dedi.**
`ui/panels/operator_library.py::OperatorLibrary.refresh()` her zaman `self.tree.expandAll()`
çağırıyordu — artık `collapseAll()` (sadece kategori başlıkları görünür, operatörler bir
kategoriye tıklanınca/oku genişletilince açılır). Arama kutusuna yazınca (`_apply_filter`)
eşleşmesi olan kategoriler artık AYRICA `setExpanded(True)` ile açılıyor (aksi halde kapalı
bir kategorinin çocukları `setHidden(False)` olsa bile görünmez kalırdı); arama kutusu
boşaltılınca varsayılan kapalı duruma geri dönüyor.

**Devam — kullanıcı "filtrelediğim fotoğrafı da yakalayıp sağ taraftaki panele atmak
istiyorum, ona da yakala diyebiliriz" dedi.** Var olan "📷 Fotoğraf Çek" (`_on_capture_
camera_photo`) SADECE ham kamera karesini (`_last_camera_frame`, throttle/filtrelerden ÖNCE)
yakalıyordu — filtrelenmiş/işlenmiş sonucu yakalamanın bir yolu yoktu.
- Yeni `main_window.py::_on_capture_filtered_frame` + camera bar'da (görüntü panelinin
  üstünde, `_capture_photo_button`'ın YANINDA) "🖼️ Filtrelenmiş Kareyi Yakala" butonu — kaynak
  ham kare değil, `_current_preview_image()` (`_on_export_current_image`'ın ve Özel Filtre
  diyalogunun da kullandığı AYNI yardımcı: seçili adımın ENGINE çıktısı). Bu yüzden
  `_capture_photo_button`'ın AKSİNE kamera aktif olması GEREKMEZ — durağan/dosyadan yüklenmiş
  bir görüntüde de çalışır, buton hep etkin. `capture_store.save_capture(..., source=
  "filtered")` ile AYNI "Yakalananlar" galerisine ("live"/"lens"/"height_scale" ile YAN YANA,
  `capture_gallery_panel.py::_SOURCE_LABELS`'e `"filtered": "Filtrelenmiş"` eklendi) kaydedilir.
  Görüntü yoksa (`_current_preview_image()` `None`) `status_label`'a inline mesaj yazılır,
  modal YOK (CLAUDE.md'nin modal-yasağı kuralıyla tutarlı, `_on_capture_camera_photo`'nun
  "kamera yok"/"kare alınmadı" mesajlarıyla AYNI desen).

**Devam — kullanıcı "şekil bul özelliği çok kasıyor ve kamerayı siyah-beyaza çeviriyor,
ayrıca öğrettiğim şekilleri silemiyorum" dedi.** Üç ayrı sorun, üç ayrı kök neden:
- **"Kamerayı siyah-beyaza çeviriyor" (GERÇEK bug, doğrulandı):** `core/shape_matching.py::
  render_match_overlay` overlay'in TABANINI kurarken `search_image`'ı HER ZAMAN önce griye
  çevirip (`_to_gray`) sonra `COLOR_GRAY2BGR` ile geri BGR'ye dönüştürüyordu — 3 kanal
  üretiyordu ama üçü de EŞİTTİ (yani görüntü hâlâ gri görünüyordu), renkli kamera görüntüsü
  `geom.shape_match` adımı seçili/aktifken hep gri çıkıyordu. Puanlama (`find_shape_model`)
  zaten içeride griye dayanıyor (doğru, DEĞİŞMEDİ) ama bu, OVERLAY'İN TABANININ da gri olması
  GEREKTİĞİ anlamına gelmiyordu. Düzeltme `ml.onnx_detect::render_detection_overlay`'deki
  AYNI deseni izliyor: `overlay = np.ascontiguousarray(search_image).copy()`, SADECE
  `overlay.ndim == 2` (gerçekten tek kanallıysa) BGR'ye çevir — girdi zaten renkliyse
  OLDUĞU GİBİ kullanılır. `tests/core/test_shape_matching.py::
  test_render_match_overlay_preserves_color_input_instead_of_turning_grayscale` renkli bir
  BGR görüntü verip kanalların birbirinden FARKLI kaldığını (griye çevrilseydi üç kanal EŞİT
  olurdu) doğruluyor.
- **"Çok kasıyor" (gerçek, ama daha KÜÇÜK bir perf düzeltmesi bulundu):** `find_shape_model`
  `target_pyramid` argümanı (birden fazla model arasında PAYLAŞILAN, zaten yeterli derinlikte
  hazır piramit) verildiğinde bile KOŞULSUZ `gray = _to_gray(search_image).astype(np.float32)`
  çalıştırıyordu — tam çözünürlüklü bir renk dönüşümü+float32 kopyası, HİÇ kullanılmadığı
  halde (piramit zaten hazır olduğunda `gray` sadece `_build_target_pyramid`'e gidiyor, o da
  bu durumda hiç ÇAĞRILMIYORDU). `geom.shape_match::run()` bunu HER seçili model için ayrı
  ayrı çağırdığından (ör. 3 model seçiliyse 3 kez), gereksiz iş modele göre KATLANIYORDU.
  Düzeltme: `gray` hesaplaması `else` dalının İÇİNE taşındı, sadece gerçekten gerektiğinde
  çalışır. Kaba arama/iyileştirme algoritmasının kendisi (önceki oturumda kasıtlı olarak
  dokunulmayan eşik/pencere sabitleri) DEĞİŞMEDİ — asıl "360° tam açı taraması inherent
  olarak pahalıdır" gerçeği hâlâ geçerli, kullanıcıya önerilen ROI daraltma/açı aralığı
  daraltma yöntemleri hâlâ en büyük kazancı verir.
- **"Öğrettiğim şekilleri silemiyorum":** kodda `_on_delete_model`'in silme mantığının
  kendisi (test edilerek doğrulandı) doğru çalışıyordu, ama `shape_model_store.
  delete_shape_model`'in `Path.unlink()` çağrısı bir `OSError` (dosya başka bir programda
  açık/kilitli, salt-okunur, vb.) fırlatırsa bu ÖNCEDEN HİÇ YAKALANMIYORDU — kullanıcıya
  hiçbir geri bildirim gitmeden istisna sessizce üst katmanlara düşüyor, ekranda "Sil"e
  basılmış ama hiçbir şey olmamış gibi görünüyordu. `_on_rename_model`'daki hata gösterim
  deseniyle AYNI şekilde artık `try/except OSError` ile sarılıp `QMessageBox.critical` ile
  AÇIKÇA gösteriliyor, model listede/bellekte SİLİNMEMİŞ sayılıyor (başarı yolu koşulsuz
  atlanıyor). `tests/ui/test_shape_matching_dialog.py::
  test_delete_model_filesystem_error_shows_message_and_keeps_model` bunu `PermissionError`
  fırlatarak doğruluyor.

**Devam — kullanıcı "şekil bul veya diğer özelliklerde akan görüntüyü nasıl hızlandırabiliriz,
olabildiğince hızlı olmasını istiyorum" dedi.** Kök neden: `_on_camera_tick` (100ms'de bir)
kare okuma → undistort/rectify → `self.engine.evaluate()` (seçili adımın TÜM dirty ata
zincirini, ör. `geom.shape_match`'in tam piramit aramasını) → overlay bileşimi → Qt pixmap
dönüşümü TAMAMEN UI/ana thread'de sırayla çalışıyordu — ağır bir adım ne kadar sürerse
ARAYÜZÜN TAMAMI (pencere sürükleme, diğer panellere tıklama dahil) o kadar donuyordu, VE
Qt timer olayları kuyruklanmadığından bir sonraki kare de o kadar gecikiyordu. Kullanıcı
"tam plan" (thread + küçük iyileştirmeler) seçeneğini onayladı.
- **Saf hesaplama çıkarıldı:** `main_window.py`'deki `_refresh_preview`/`_compose_display_image`/
  `_get_normal_base_image`/`_cumulative_roi_offset`/`_compose_side_by_side` `self.*`'a bağımlı
  metotlardı — hepsi `self`'siz, girdileri parametre olarak alan MODÜL-SEVİYESİ fonksiyonlara
  çıkarıldı, artı yeni bir `PreviewFrameResult` dataclass'ı (`display_image`, `hover_
  measurements`, `measurements`, `status_text`, ROI şekil/koordinat alanları — SADECE düz veri,
  hiçbir Qt/engine/graph referansı taşımaz) döndüren yeni `_build_preview_frame(engine,
  registry, graph, pipeline_order, node_id, view_mode, camera_active, last_camera_frame)`.
  `MainWindow._refresh_preview` artık bu fonksiyonu `self.engine`/`self.graph` ile çağırıp
  sonucu widget'lara uygulayan ince bir sarmalayıcı (+ yeni `_apply_preview_frame_result`).
- **Yeni `_LiveTickWorker(QThread)`** (`_BatchWorker`'ın YANINDA, AYNI izolasyon ilkesiyle):
  `_on_camera_tick` artık `copy.deepcopy(self.graph)` alıp TAZE bir `ExecutionEngine` kuran bu
  worker'a dispatch ediyor; worker `_build_preview_frame`'i KENDİ kopyası üzerinde çalıştırıp
  düz veri (`PreviewFrameResult`) içeren `result_ready = Signal(int, object)` ile UI thread'ine
  dönüyor — worker HİÇBİR ZAMAN `self.image_view`/`self.results_panel`'e dokunmuz. `self.engine`/
  `self.graph`'ın kendisi worker tarafından hiç dokunulmaz; `_current_preview_image()` gibi
  tek-seferlik (buton tıklamalı) çağrılar bugünkü gibi senkron kalır — zaten `inject_result`
  her tick'te downstream'in TAMAMINI dirty işaretlediğinden (yeni kare = yeniden hesapla),
  tick'ler arası engine cache'i canlı kamera zincirinde HİÇBİR fayda sağlamıyordu, bu yüzden
  taze bir engine kurmak hiçbir şey KAYBETTİRMEDİ.
- **Tek-uçuşlu (single-flight) dağıtım + kare atlama:** `_live_worker_busy` True iken yeni
  tick'in İŞLENMESİ (worker dispatch'i) atlanır — kare KUYRUKLANMAZ, `_last_camera_frame`/
  otomatik mm/px güncellemesi yine de HER tick'te çalışır (worker'dan bağımsız, ucuz). Bu,
  sistemin HER ZAMAN en hızlı ulaşabildiği kadar taze bir kareyi işlemesini sağlar.
- **Bayatlık (staleness) kontrolü:** yeni `_live_tick_generation` sayacı her dispatch'te VE
  seçili adım/görünüm modu değiştiğinde (yeni `_set_selected_node` yardımcı metodu, TÜM
  `self._selected_node_id =` atamalarının artık TEK geçiş noktası + `_on_view_mode_changed`)
  artırılır; `_on_live_tick_result` gelen sonucun `generation`'ı güncel değilse SESSİZCE atar
  — geç gelen bir sonucun yanlış adım/moda ait görüntüyü ekrana yanlışlıkla yansıtmasını önler.
- **Yan etki — `_refresh_enum_gallery` artık kendi `evaluate()`'ini çağırıyor:** bu fonksiyon
  `engine.trial_run()`'ın ata zincirinin ÖNCEDEN cache'te ısınmış olmasını beklediği (eskiden
  `_refresh_preview`'in AYNI tick'te önce çalışmasıyla ÖRTÜK sağlanıyordu) için, `self.engine`
  artık tick başına `evaluate()` edilmediğinden bunu KENDİSİ `self.engine.evaluate(node_id)`
  ile açıkça ısıtır — sadece seçili adımda GERÇEKTEN bir enum parametre varsa çalışır, diğer
  pipeline'lar hiçbir ek maliyet ödemez.
- Test etkisi: `tests/ui/test_main_window.py`'de `_on_camera_tick()` çağırıp HEMEN ardından
  widget durumuna bakan testlerin çoğu senkron kısımlara (undistort, `_last_camera_frame`,
  enum galerisi throttle'ı) dayandığından DOKUNULMADAN geçti — SADECE `image_view._pixmap`'e
  bakan tek bir test, worker'ın sonucunu beklemesi için yeni `_run_camera_tick_and_wait(window,
  qtbot)` yardımcısına taşındı. 2 yeni test eklendi: worker meşgulken kare atlama (+
  `_last_camera_frame`'in yine de güncellendiği) ve bayat generation'ın sessizce atıldığı.

**Devam — kullanıcı dört ayrı istek getirdi: "geometrik eşlemede geniş bir kare içinde
değil, sadece istenen cismin etrafı çizilecek şekilde yapılmalı", "her işlemin sonucunun
süresi sonuçlar kısmında yazmalı", "yaptığımız kalibrasyonu canlı görüntü olmadan da
kullanabilmeliyim / çektiğim fotoğraflarda daha sonradan uzunluk bulabilmeliyim", "geometrik
eşlemede scale boyutu ve cismin ötelenme uzaklığı da yazmalı".** İki netleştirme sorusuyla
netleştirildi: "scale" isteği GERÇEK ölçek-toleranslı arama DEĞİL (v1 kapsam dışı notu hâlâ
geçerli, aşağıya bkz.), bulunan nesnenin mevcut kalibrasyonla GERÇEK DÜNYA boyutunu (mm)
görmek; "süre" isteği SADECE seçili adımın değil, TÜM pipeline adımlarının süresini küçük
bir tabloda görmek.

1. **Sıkı kontur overlay:** `core/shape_matching.py::ShapeModel.corners` HER ZAMAN eğitim
   ROI'sinin dikdörtgen sınırlayıcı kutusuydu (`mask`/poligon modunda BİLE) — nesnenin
   gerçek dış hattı hiçbir yerde saklanmıyordu. Yeni opsiyonel `ShapeModel.contour` alanı
   (merkeze göre, `corners` ile AYNI konvansiyonda) eklendi; `None` ise (eski `.json`
   modeller, düz dikdörtgen ROI eğitimi) `render_match_overlay` eskisi gibi `corners`'a
   düşer — GERİYE DÖNÜK TAM UYUMLU. `train_shape_model`'e yeni `contour_points` parametresi:
   `shape_matching_dialog.py::_on_train` POLİGON modunda çizilen poligonun kendisini, RECT+
   "Konturu Otomatik Algıla" (ters çevrilmemiş) modunda `_finalize_mask`'ın ürettiği maskeden
   `cv2.findContours` ile bulunan EN BÜYÜK dış konturu geçirir. "Kontur Dışını Kullan" (ters
   çevrilmiş) modda BİLİNÇLİ olarak kontur çıkarılmaz (tek/temiz bir dış hat yok) — dikdörtgene
   düşülür.
2. **Sonuçlar sekmesinde TÜM adımların süresi:** `core/engine.py::ExecutionEngine`'e yeni
   `self._durations` (SADECE gerçekten çalıştırılan node'lar için, cache-hit/üst-akış-hatası
   ATLANANLAR için YOK) + `durations` property'si eklendi. `main_window.py::PreviewFrameResult`'a
   `step_durations` alanı (`_build_preview_frame` içinde `label_for` ile Türkçe etiketlenip
   pipeline sırasıyla derlenir) — `_LiveTickWorker`'ın kendi taze engine'i ZATEN bu fonksiyonu
   çağırdığından, EK bir sinyal/thread mekanizması GEREKMEDEN canlı kamera yolundan da akar.
   `MeasurementsSummaryPanel`'e mevcut ölçüm etiketinin ALTINA (ayrı, gerekmedikçe gizli) küçük
   bir `QTableWidget` (`set_step_durations`) eklendi. Not: kaynak (`io.image_source`) düğümü
   canlı akışta HER ZAMAN `inject_result` ile doldurulur (gerçek `run()` çağrılmaz), bu yüzden
   tabloda hiç görünmez — bu BEKLENEN davranış, bug değil.
3. **Kalibrasyon + ölçüm, kamera olmadan:** "Lens Kalibrasyonu...", "Kalibrasyon Profili
   Yükle...", "Yükseklik Kalibrasyonu..." menü aksiyonları SADECE kamera aktifken etkindi —
   SAF bir UI kısıtlamasıydı (mm/px hesaplama zaten plane-rectification/reference-distance
   yollarında kameradan bağımsız çalışıyordu, `_load_calibration_profile` zaten kameraya hiç
   bakmıyordu). Üçü de artık HER ZAMAN etkin; `_on_open_lens_calibration`/`_on_open_height_
   scale_calibration`'daki kamera-yok koruması kaldırıldı, dialoglara verilen `frame_provider`
   yeni `_camera_frame_provider()` ile kamera-güvenli hale getirildi (kamera yokken `None`
   döner — dialogların "Kare Yakala" butonu bunu zaten sessizce no-op olarak ele alıyordu,
   sürükle-bırak/galeri girdisi TAM çalışır durumda kalır). Ayrıca `CaptureGalleryPanel`'e
   yeni `open_requested` sinyali (çift tıklama + yeni "Pipeline'a Yükle" context-menü
   aksiyonu) eklendi, `main_window.py`'de `open_image`'e bağlandı — eskiden bir kareyi
   pipeline'a yüklemenin TEK yolu (keşfedilmesi zor) sürükle-bırak yapmaktı.
4. **Gerçek boyut (mm) + öteleme mesafesi:** `ShapeModel`'e yeni `reference_center` alanı
   (eğitim ROI'sinin MUTLAK merkezi — eski modellerde `None`, YANLIŞ bir (0,0) DEĞİL).
   `geom.shape_match`'e `analysis.region_props` ile AYNI isim/etiketle yeni bir `mm_per_px`
   parametresi eklendi; `run()` artık (doluysa) model başına SABİT `width_mm`/`height_mm`
   (ölçek araması yok, bu yüzden eşleşme sayısından bağımsız sabit) ve `model.reference_
   center` varsa `displacement_px`/(`mm_per_px`>0 ise) `displacement_mm` (eşleşmenin
   öğretildiği pozisyondan Öklid uzaklığı) ekliyor. `_push_mm_per_px`/`_region_props_needs_
   mm_per_px` yeni `_MM_PER_PX_OP_IDS = ("analysis.region_props", "geom.shape_match")`
   üzerinden İKİSİNİ de günceller — kullanıcının "eski kalibrasyonu yükle → cm/mm değeri
   gör" isteği bu genişletmeyle birlikte, `_load_calibration_profile`'ın ZATEN çağırdığı
   `_push_mm_per_px` sayesinde, EK bir kod yolu gerekmeden çalışır. `measurements_summary.py`
   yeni alanları mevcut sabit-f-string desenine ekliyor (`displacement_mm` yoksa `displacement_
   px`'e düşer).

**Devam — kullanıcı "şekil bulma kısmında hala kare içine alıyor, öğrettiğim şeklin dışını
çizsin, kare içine almasın, öğretildiği şekilde ve boyutta çizsin, belirgin olsun" dedi.**
Bir önceki turda eklenen sıkı-kontur overlay'i SADECE "Konturu Otomatik Algıla" checkbox'ı
İŞARETLİYKEN (ya da Serbest Kontur/poligon modunda) devreye giriyordu — varsayılan Dikdörtgen
ROI eğitiminde (checkbox kapalı, en yaygın/ilk akla gelen yol) hâlâ `contour=None` kalıp
overlay dikdörtgene düşüyordu; kullanıcı checkbox'ı fark etmeden/işaretlemeden eğittiği için
sorun "düzelmemiş" gibi görünüyordu.
- `shape_matching_dialog.py::_on_train`'in Dikdörtgen dalı artık `_build_auto_contour_mask`'ı
  (nesne tespiti) checkbox'tan BAĞIMSIZ HER ZAMAN çağırıyor — bulunan `inside_mask`'tan
  (ters çevrilmemişse) `_largest_contour_points` ile kontur HER durumda çıkarılıp
  `contour_points` olarak `train_shape_model`'e geçiriliyor. **Kritik ayrım:** eğitim/
  eşleştirmede KULLANILAN nokta filtresi (`mask`, kaba/gürültülü kenarları eleyen) HÂLÂ SADECE
  checkbox açıkken ayarlanıyor — kontur artık SADECE GÖRSEL bir sonuç, kullanıcının "diğer
  kısımlar aynı kalabilir" isteğiyle tutarlı (bir noise testi bunu doğruluyor: checkbox
  kapalıyken ROI içindeki ayrı bir gürültü kenarı modele HÂLÂ giriyor, ama overlay yine de
  sıkı konturu çiziyor). Ters çevrilmiş ("Kontur Dışını Kullan") modda DEĞİŞMEDİ — kontur hâlâ
  hiç çıkarılmaz (tek/temiz bir dış hat yok), dikdörtgene düşülür. Nesne bulunamazsa (parlak/
  karmaşık sahne) sessizce dikdörtgene düşülür, eğitim ÇÖKMEZ — checkbox kapalıyken bu durumda
  hiçbir inline not GÖSTERİLMEZ (o not SADECE checkbox açıkken, training-mask etkisini
  anlatmak için kullanılır).

**Devam — kullanıcı iki ayrı bug bildirdi: "elle roi çizme de şekil öğretmede çalışmıyor en az
3 nokta çizmeme rağmen gözükmüyor ve en sonunda noktalar birleşmiyor" + "hala şekil bulma...
eti yazısını seçtim ve dışını kontür bulma yardımıyla attım, sadece eti yazısını öğrettim
modele... sadece o eti yazısını görmek istiyorum".** İki AYRI kök neden:
- **Poligon "Kapat" butonu hiç etkinleşmiyordu (gerçek bug, doğrulandı):** `RoiCanvas.
  _mouse_press_polygon` yeni bir nokta EKLERKEN `polygon_changed` sinyalini hiç yayınlamıyordu
  — sadece `close_polygon()`/köşe-sürükleme-bırakma bunu yapıyordu. `ShapeMatchingDialog.
  _on_polygon_changed` (nokta sayısına göre "Poligonu Kapat" butonunun `setEnabled` durumunu
  güncelleyen TEK yer) bu yüzden kullanıcı gerçek fare tıklamalarıyla 3+ nokta çizse BİLE hiç
  tetiklenmiyordu — buton sonsuza kadar devre dışı kalıyor, "Poligonu Kapat"a basmak hiçbir şey
  yapmıyordu. Düzeltme: `_mouse_press_polygon` artık nokta eklerken/geri alırken de
  `polygon_changed.emit(...)` çağırıyor. `tests/ui/test_shape_matching_dialog.py::
  test_polygon_close_button_enables_after_clicking_three_points` bunu doğrudan `_polygon_
  points`'i elle doldurmak yerine GERÇEK `qtbot.mouseClick` olaylarıyla doğruluyor (eski
  testlerin çoğu iç state'i doğrudan set ediyordu, bu yüzden bug'ı hiç yakalamamışlardı).
- **Metin gibi AYRIK çok-parçalı nesnelerde kontur tek parçaya sıkışıyordu (asıl "eti" bug'ı):**
  `_build_auto_contour_mask` ROI içinde Otsu ile bulunan TÜM bağlı bileşenlerden sadece EN
  BÜYÜĞÜNÜ (`max(objects, key=...)`) "nesne" sayıyordu — harfleri birbirinden AYRIK (bağlı
  olmayan) bir kelimede (ör. "eti") bu, sadece EN BÜYÜK harfi model/overlay'e dahil edip DİĞER
  harfleri arka plan gibi elip hem eğitim nokta filtresinden hem overlay konturundan
  düşürüyordu — kullanıcı "sadece o yazıyı öğrettim" sanıyordu ama aslında sadece tek bir harf
  öğretilmişti, VE `_largest_contour_points`'in `cv2.findContours`+`max(..., key=cv2.
  contourArea)` deseni de aynı şekilde tek harfin konturunu seçiyordu. İki parçalı düzeltme:
  (1) `_build_auto_contour_mask` artık TÜM bileşenleri değil ama en büyük bileşenin alanının en
  az `_MULTI_PART_AREA_RATIO=0.15` katı olan HEPSİNİ birleştirip (`|=`) tek bir maskede
  topluyor — saf gürültü benekleri (ör. mevcut `test_auto_contour_checkbox_uses_detected_
  object_mask_and_reduces_noise_points`'teki küçük kare, üçgenin %15'inden KÜÇÜK) hâlâ elenir,
  ama karşılaştırılabilir boyutlu harfler/parçalar artık HEPSİ dahil edilir. (2) `_largest_
  contour_points`, `cv2.findContours`'un ürettiği BİRDEN FAZLA ayrık dış konturu (harf sayısı
  kadar) o an TEK bir `cv2.convexHull` ile sıkı bir çokgende sarmalıyordu — bu ARA bir
  düzeltmeydi, aynı turda kullanıcı "hala eti yazısını değil... kareyi içine alıyor" diye
  bildirince (bkz. altındaki "Devam" notu) tamamen değiştirilip HER parçanın AYRI ayrı
  saklandığı/çizildiği hâle geçildi — bu paragraf sadece o ara adımın kaydı, GÜNCEL davranış
  için aşağıdaki nota bakınız.

**Devam — kullanıcı "hala eti yazısını değil, eti yazısının dahil olduğu kareyi içine alıyor...
kontürde elediğim kısım şekil bulmada gözükmemeli. farklı boyutlarda aynı cisimden varsa scale
boyutu yazmalı" dedi.** İki ayrı iş:
- **Kontur artık BİRDEN FAZLA AYRIK parça olarak saklanıyor/çiziliyor (convexHull TAMAMEN
  kaldırıldı):** bir önceki turdaki `cv2.convexHull` düzeltmesi harfleri TEK bir birleşik
  çokgende sarmalıyordu — ama "eti" gibi aynı yükseklikteki harflerin dış bükey zarfı
  GEOMETRİK olarak neredeyse düz bir dikdörtgene benziyor (üstte/altta harflerin tepe/taban
  noktalarını birleştiren düz bir çizgi), bu yüzden kullanıcıya HÂLÂ "kare" gibi görünüyordu —
  VE harfler arasındaki elenen boşluk da zarfın İÇİNDE kalıp "nesnenin parçası" gibi
  görünüyordu (gerçek kullanıcı raporu: "kontürde elediğim kısım şekil bulmada gözükmemeli").
  Kök çözüm: `ShapeModel.contour` artık TEK bir `(N,2)` dizi DEĞİL, `list[np.ndarray] | None`
  — her ayrık parça (`cv2.findContours`'un bulduğu her dış kontur, `_contour_parts_from_mask`
  eski `_largest_contour_points` yerine) KENDİ dizisinde saklanır, `render_match_overlay` her
  parçayı KENDİ kapalı poligonu olarak (`for part in parts: cv2.polylines(...)`) ayrı ayrı
  çizer — parçalar arasına ASLA bir çizgi çizilmez. `train_shape_model`'in `contour_points`
  parametresi hem TEK parça (düz `[(x,y),...]`, poligon eğitimi) hem BİRDEN FAZLA parça
  (`[[(x,y),...], [(x,y),...]]`, otomatik çok-parça algılama) kabul eder —
  `contour_points[0][0]`'ın kendisi bir nokta mı (liste/tuple) yoksa bir sayı mı olduğuna
  bakarak ayırt edilir. `ShapeModel.to_dict()`/`from_dict()` YENİ format (parça listesi)
  üretir/okur, ESKİ (bu değişiklikten önce kaydedilmiş) düz-liste `.json` modellerini de
  `_parse_contour()` içindeki AYNI nesting-derinliği kontrolüyle geriye dönük okuyabilir —
  ama **kullanıcının "eti" modeli gibi ÖNCEDEN kaydedilmiş modeller bu düzeltmeyi görmek için
  YENİDEN eğitilip kaydedilmeli**, var olan `.json` dosyasındaki eski kontur otomatik
  düzelmez. `tests/core/test_shape_matching.py::
  test_render_match_overlay_does_not_connect_disjoint_contour_parts` iki uzak parça arasındaki
  orta noktanın boyanmadığını piksel bazında doğruluyor;
  `tests/ui/test_shape_matching_dialog.py::test_auto_contour_handles_disjoint_multi_part_object_like_text`
  güncellenip `len(contour) == 3` (üç ayrı parça, TEK bir çokgen değil) kontrolü eklendi.
- **Ölçek-toleranslı arama eklendi (önceden bilinçli kapsam dışı bırakılmıştı, kullanıcı bu
  turda net onayladı):** `core/shape_matching.py::find_shape_model`'e opsiyonel `scale_min`/
  `scale_max`/`scale_step_coarse` parametreleri eklendi — varsayılan `scale_min=scale_max=1.0`
  iken TEK bir ölçek (1.0) taranır, maliyet/davranış ESKİSİYLE BİREBİR AYNI kalır (`geom.
  shape_match` operatöründeki yeni "Ölçek Min./Maks./Arama Adımı" ParamSpec'leri de aynı
  varsayılanla geriye dönük uyumlu). Mekanizma açı aramasıyla AYNI kaba-arama+iyileştirme
  desenini izliyor: `_build_direction_kernels`/`_score_map`'e yeni bir `scale` çarpanı eklendi
  (model noktalarının ofsetlerini büyütür/küçültür, gradyan YÖNLERİNİ etkilemez — ölçekle kenar
  açısı değişmez), kaba aramada `scale_values` ızgarası taranır (`_coarse_search`), iyileştirme
  aşamasında (`_refine_candidate`) `angle_window`'un narrowlanma deseninin AYNISıyla
  (`_REFINE_SCALE_DIVISOR`) her piramit seviyesinde daraltılır. Aday tuple'ları `[x,y,angle,
  score]`'dan `[x,y,angle,scale,score]`'a genişledi (`_nms`/`_local_maxima` etkilenmedi, sadece
  ekstra alanı taşıyorlar); yeni `MatchResult.scale` alanı (varsayılan 1.0) bulunan nesnenin
  öğretilen modele göre ölçeğini taşır. `geom.shape_match::run()` `scale_max > scale_min` iken
  (SADECE o zaman, gürültü olmasın diye) `measurements`'a `scale_percent` ekliyor, VE `mm_per_px`
  kalibrasyonu varsa `width_mm`/`height_mm` artık SABİT öğretilen boyut değil `match.scale` ile
  ÇARPILMIŞ gerçek boyutu yansıtıyor (`mm_per_px` yardım metni buna göre güncellendi).
  `measurements_summary.py`/`main_window.py::_on_hover_measurement_changed` her ikisi de
  `scale_percent` (+ fark edilen bir boşluk: `width_mm`/`displacement_*` hover panelinde HİÇ
  gösterilmiyordu, bu turda eklendi) satırlarını gösteriyor. `tests/core/test_shape_matching.py`
  içindeki `test_find_shape_model_with_scale_search_finds_larger_object_and_reports_scale` ve
  `tests/operators/test_shape_match.py::
  test_run_with_scale_search_finds_larger_object_and_reports_scale_percent` %50 büyük sentetik
  bir nesneyle hem ölçeğin (~1.5) hem doğru konumun geri kazanıldığını doğruluyor. Performans
  notu: geniş bir `scale_min`-`scale_max` aralığı kaba arama maliyetini taranan ölçek sayısıyla
  orantılı ARTIRIR (açı aralığıyla AYNI ödünleşim) — kullanıcı sadece gerektiği kadar geniş
  tutmalı.

**Devam — kullanıcı ısrarla "hâlâ eti yazısını değil, kareyi çiziyor... hep öğrettiğim gibi
çizsin, kare olarak çizmesini hiç istemiyorum" dedi; `/plan` ile yapılandırılmış bir soru-cevap
turu istendi.** Sorgulama şunu netleştirdi: eğitim ÖNCESİ canlı önizleme (checkbox işaretliyken
kırmızı elenen-alan) harflere SIKI oturuyor (doğru), ama pipeline'da "Şekil Bul" çalışınca
overlay hâlâ kutu; uygulama TAMAMEN yeniden başlatılmış (bayat kod/önbellek ihtimali dışlandı,
`save_shape_model` zaten `mtime`'dan bağımsız açıkça önbelleği temizliyor). Kod okuması
`_build_auto_contour_mask` → `_contour_parts_from_mask` → `train_shape_model` →
`ShapeModel.contour` → `render_match_overlay` zincirinin ÖNİZLEMEYLE AYNI maskeyi kullandığını
doğruladı — sentetik testler (belirgin UZAK parçalarla) bu zincirin çalıştığını kanıtlıyor.
**En olası açıklama (kesin teşhis GERÇEK görüntü/model dosyası olmadan mümkün değil, kullanıcı
paylaşmadı):** "eti" gibi kısa/bitişik-harfli bir kelimenin harfleri gerçek fotoğrafta
`_MASK_DILATION_PX=2` genişletmesiyle TEK bir bloğa birleşiyor olabilir — kısa bir kelimenin
BİRLEŞİK dış hattı zaten doğal olarak dikdörtgene yakın görünür (aynı yükseklikte harfler, yan
yana bir şerit), bu ayrı-parça mantığının kusuru değil, geometrik bir sonuç olabilir.
- **Düşük riskli iyileştirme uygulandı:** `shape_matching_dialog.py`'ye yeni
  `_CONTOUR_DILATION_PX=1` (eğitimde KULLANILAN nokta filtresinin `_MASK_DILATION_PX=2`'sinden
  KÜÇÜK) eklendi — `_dilate_mask(mask, px=...)` artık parametrik, `_on_train`'in SADECE overlay
  konturu çıkarımı için çağırdığı `_contour_parts_from_mask(self._dilate_mask(inside_mask,
  px=_CONTOUR_DILATION_PX))` bu küçük payı kullanıyor; eğitimde kullanılan `_finalize_mask`
  (nokta filtresi) HÂLÂ eski `_MASK_DILATION_PX=2`'de, hiç ETKİLENMEDİ. Bu, birbirine yakın ama
  teknik olarak ayrı iki parçanın SADECE görsel kontur çıkarımında gereksiz yere birleşmesini
  azaltır (`tests/ui/test_shape_matching_dialog.py::
  test_contour_dilation_is_smaller_than_training_mask_dilation_for_close_letters` 3px boşluklu
  iki bloğun eski payla BİRLEŞTİĞİNİ, yeni payla AYRI kaldığını doğruluyor) — ama harfler GERÇEKTEN
  bitişikse (0px boşluk, tasarım gereği) bu değişiklik yardımcı OLMAZ.
- **Kesin/garanti çözüm zaten var, kullanıcıya önerildi:** Serbest Kontur (Poligon) elle çizim
  modu -- bu yolda `ShapeModel.contour` DOĞRUDAN kullanıcının çizdiği noktalardır, hiçbir Otsu/
  bağlı-bileşen analizi araya girmez, dolayısıyla "hep öğrettiğim gibi çizsin" isteğini algoritma
  kalitesinden BAĞIMSIZ garanti eder. Kullanıcıya "eti"yi Poligon modunda yeniden eğitip
  sonucu bildirmesi önerildi — bu aynı zamanda TEŞHİS amaçlı: Poligon modunda da hâlâ kutu
  görülürse bu KESİN olarak `render_match_overlay` zincirinde gerçek bir bug olduğunu gösterir
  (Poligon modunda otomatik algılama hiç karışmaz) ve o noktada gerçek dosya paylaşımı
  ZORUNLU hâle gelir — henüz bu geri bildirim ALINMADI.

**Devam — kullanıcı önerilen Poligon modunu denedi: kontur artık ÇİZDİĞİ yeri doğru alıyor
(kendi ifadesiyle "poligonla çizince çizdiğim yeri aldı şimdi düzeldi" — önceki karışıklık
yanlışlıkla işaretlenmiş "Kontur Dışını Kullan" checkbox'ıymış, kod hatası değil), AMA
`geom.shape_match` ("Şekil Bul") çalıştırınca GERÇEK bir OpenCV çökmesi bildirdi:
`error: (-215:Assertion failed) anchor.inside(Rect(0,0,ksize.width,ksize.height)) in function
'cv::normalizeAnchor'`.** Bu, aylardır kovalanan "kutu" tartışmasının BAMBAŞKA, kesin ve
tekrarlanabilir bir kök nedeni oldu:
- **Kök neden (kod okumasıyla + testle kanıtlandı):** `core/shape_matching.py::
  _build_direction_kernels()`, bir `ShapeLevel`'ın noktalarından bir korelasyon çekirdeği kurup
  `cv2.filter2D`'e `anchor=(center_col, center_row)` geçiriyordu (`center_col/row = -min_dx/
  -min_dy`, "ofset (0,0) çekirdekte hangi hücreye denk geliyor" anlamına gelir). Çekirdeğin
  sınırları (`width`/`height`) SADECE gerçek noktaların `min`/`max`'ından hesaplanıyordu —
  ofset (0,0)'ın bu aralığa DAHİL olduğu HİÇ garanti edilmiyordu. Belirli bir döndürme
  açısında modelin TÜM noktaları merkeze göre kesinlikle AYNI taraftaysa (ör. `min_dx > 0`),
  `center_col` NEGATİF (çekirdek sınırlarının DIŞINDA) çıkıp OpenCV'nin `anchor.inside(...)`
  kontrolünü ihlal edip çöküyordu. Bu, elle çizilen (Poligon) bir kontur gibi ROI'nin NAİF
  geometrik merkezine göre ASİMETRİK nokta bulutlarında — özellikle en KABA piramit
  seviyesinde çok az nokta kalıp hepsi tek tarafa toplanabildiğinde — kolayca tetikleniyor;
  `geom.shape_match` 360° tam açı taraması yaptığından yüzlerce denenen açıdan HERHANGİ
  BİRİNDE bu oluşursa çöküyordu. Sentetik test şekilleri HER ZAMAN ROI merkezine göre dengeli
  seçildiğinden (Explore ajanıyla doğrulandı) bu daha önce hiçbir testte YAKALANMAMIŞTI.
  **Bug hem Poligon HEM RECT+"Konturu Otomatik Algıla" ile eğitilen modellerde (maskeleme,
  yani arka planı elemek, TAM OLARAK noktaları asimetrik hâle getiren şeydir) tetiklenebilir**
  — kullanıcının RECT modunda gördüğü "kutu" şikayetlerinin bir kısmı da aslında bu çökmenin
  (ya da ona yakın davranışın) BAŞKA bir tezahürü olabilir.
- **Düzeltme:** `_build_direction_kernels`'te çekirdek sınırları hesaplanırken ofset (0,0) HER
  ZAMAN aralığa dahil edilir (`min_dx, max_dx = min(min_dx, 0), max(max_dx, 0)` — aynısı y için)
  — bu, `center_col`/`center_row`'un MATEMATİKSEL olarak her zaman `[0, width)`/`[0, height)`
  içinde kalmasını garanti eder, nokta dağılımından TAMAMEN bağımsız. Dengeli (eski) modeller
  için davranış/çıktı DEĞİŞMEZ (0 zaten aralıktaysa no-op). `tests/core/test_shape_matching.py::
  test_build_direction_kernels_handles_points_all_on_one_side_of_center` ve `::
  test_score_map_does_not_crash_when_all_points_are_offset_to_one_side` düzeltmeden ÖNCE
  GEÇİCİ OLARAK geri alınıp gerçekten AYNI OpenCV hatasıyla çöktüğü doğrulanarak yazıldı (yani
  bu testler gerçekten bug'ı YAKALIYOR, sahte-yeşil değil).
- **Yeni özellik — eğitimde FİİLEN kullanılan (filtrelenmiş) görüntüyü göster:** kullanıcı
  "Konturu Otomatik Algıla"nın eğitimi gerçekten etkileyip etkilemediğine (kod okumasıyla
  DOĞRU çalıştığı önceden kanıtlanmış olsa da) görsel kanıt olmadan güvenmedi — haklı bir talep,
  çünkü eğitim ÖNCESİ canvas'taki kırmızı-alan önizlemesi eğitim SONRASI GERÇEKTEN neyin
  kullanıldığını göstermiyordu. `ShapeMatchingDialog`'a "Modeli Eğit"in YANINA yeni
  `_trained_filter_preview_label` (160x160 `QPixmap`) eklendi — her başarılı eğitimden sonra
  `_update_trained_filter_preview(roi, mask)` çağrılıp `_build_filtered_training_preview`
  (ROI kırpımı + `mask` doluysa elenen pikseller SİYAHA boyanmış hali) gösterilir; `mask=None`
  ise (checkbox kapalı) ham kırpım DEĞİŞTİRİLMEDEN gösterilir — kullanıcı checkbox'ı açıp
  kapatarak farkı DOĞRUDAN karşılaştırabilir. Yeni `self._last_train_mask` alanı son eğitimde
  kullanılan mask'i saklar (testte/ihtiyaç halinde önizlemeyi yeniden üretebilmek için).

**Devam — kullanıcı yeni filtre önizlemesini denedi, dört ayrı bulgu/istek bildirdi: "kırpılmış
bölge siyah gözüküyor ama hâlâ şekil bulurken o bölgeyi de seçiyor... ayrıca eğittiğim modelleri
silemiyorum... dikdörtgenin yanında çember şeklinde de bir roi istiyorum... roi seçildikten
sonra otomatik kontür özelliği çalışıp cismin etrafını çizip yeni roi olarak belirleyebiliriz."**
- **"Kırpılmış bölge hâlâ seçiliyor" kök nedeni:** sorgulamayla netleşti, ÇİZİLEN kutu/kontur
  (overlay) o siyah bölgeyi kapsıyormuş — bu, `_invert_mask_checkbox` ("Kontur Dışını Kullan")
  işaretliyken (bu kullanıcıda DAHA ÖNCE de olmuş bir yanlış tıklama) `_finalize_mask`'ın
  `kept = ~inside_mask` yapması (önizlemede SİYAH görünen aslında NESNENİN KENDİSİ, arka plan
  DEĞİL) VE `_on_train`'in bu modda BİLEREK `contour_points`'i hiç hesaplamamasıyla (ters
  tarafın tek/temiz bir dış hattı yoktur) tam olarak AÇIKLANIYOR — mimari olarak doğru ama
  kullanıcıya hiç anlatılmıyordu. Düzeltme: `_update_trained_filter_preview`'a yeni bir
  `_trained_filter_preview_caption` (`QLabel`) eklendi — `mask=None` iken "(filtre
  uygulanmadı)", normal modda "Korunan: Nesne (arka plan elendi)", TERS moddayken turuncu/
  kalın "⚠ Korunan: ARKA PLAN, nesne elendi ('Ters Çevir' işaretli) — overlay dikdörtgene
  düşecek" gösterir — kullanıcı artık NEDEN kutu gördüğünü anında anlar.
- **"Modelleri silemiyorum" kök nedeni:** kod (`_on_delete_model`, `OSError` yakalama önceki
  turda zaten eklenmişti) ÇALIŞIYOR; kullanıcı "silme menüsünü göremedim direkt" dedi — MANTIK
  hatası değil, GÖRÜNÜRLÜK sorunu: `manage_row` TEK bir `QHBoxLayout`'ta 6 widget (combo + 5
  buton, bazıları uzun metinli: "Yeniden Adlandır...", "Dışa Aktar (JSON)...") barındırıyordu,
  dar bir pencerede "Sil" görünür alanın dışına itilebiliyordu. Düzeltme: `manage_row` İKİ
  satıra bölündü -- 1. satır (en sık kullanılan): combo + Yükle + **Sil**; 2. satır: Yeniden
  Adlandır/Dışa Aktar/İçe Aktar. Mantık/handler'lara HİÇ dokunulmadı.
- **Çember (Daire) ROI modu eklendi:** `RoiCanvas` zaten "CIRCLE" şeklini destekliyordu
  (`roi_circle_changed` sinyali, `set_roi_circle`) ama `ShapeMatchingDialog`'a hiç
  BAĞLANMAMIŞTI. `_draw_mode_combo`'ya üçüncü seçenek ("Daire (Çember)") eklendi,
  `_DRAW_MODE_SHAPES = ("RECT","POLYGON","CIRCLE")` ile index↔şekil eşlemesi TEK bir yerde
  toplandı. Yeni `self._roi_circle` state + `_on_roi_circle_changed` handler'ı. `_on_train`'e
  ÜÇÜNCÜ bir dal: çemberin kendisi Poligon'daki `mask_canvas`/`cv2.fillPoly` deseninin AYNISIYLA
  (`cv2.circle(..., thickness=-1)`) bir mask/kontur olarak kullanılır — "Konturu Otomatik
  Algıla" işaretliyse bu, çemberin İÇİNDE nesneyi daha da daraltmak için `&` kesişimiyle EK bir
  arıtma katmanı olur (çemberin dışına ASLA taşmaz).
- **Yeni yöntem, kullanıcı onayladı ("Birlikte kullanılsın"): "ROI'yi Nesneye Daralt" butonu.**
  RECT/CIRCLE modunda (Poligon'da gizli — zaten elle çizilir) yeni `_shrink_roi_button` →
  `_on_shrink_roi_to_contour()`: MEVCUT `_build_auto_contour_mask` (değişmeden) ile ROI/çember
  içinde nesne tespit edilip, bulunan maskenin sınırlayıcı kutusu YENİ (daraltılmış) ROI/çember
  olarak `self._roi`/`self._roi_circle` + canvas'a ATANIR — kullanıcı bunu GÖRÜR (dikdörtgen/
  çember küçülür), isterse tekrar daraltabilir/elle ince ayar yapıp SONRA eğitir. Nesne
  bulunamazsa ROI/çember DEĞİŞMEDEN kalır, inline not gösterilir (çökme yok). Bu, MEVCUT
  çok-parçalı kontur çizimiyle ÇAKIŞMAZ — `_on_train` hâlâ AYNI akıştan geçer, sadece ROI artık
  daha sıkı olduğundan hem `corners` (fallback dikdörtgen) hem çıkarılan kontur daha az
  gereksiz kenar boşluğu içerir.

**Devam — kullanıcı "menünün boyutlandırması kötü olduğu için aşağıdaki sil butonlarını
göremiyormuşum ayrıca. yüklediğim görüntü çok küçük kalabiliyor. bunları düzenle istersem
pencereyi büyültüp küçültebileyim" dedi.** İki ayrı, birbirinden bağımsız düzeltme
(`shape_matching_dialog.py`, TEK dosya):
- **Pencere küçültülemiyordu (Sil butonu bu yüzden erişilemiyordu):** eskiden yükle/çizim
  modu/eğit/kayıtlı modeller dahil TÜM kontroller dialog'un ANA `QVBoxLayout`'una doğrudan
  ekleniyordu — dialog'un minimum boyutu bunların TOPLAM `sizeHint`'ine dayanıyordu (~800px+
  yükseklik), pencere bunun altına hiç küçültülemiyordu; küçük bir ekranda/varsayılan
  boyuttan küçük açılırsa "Sil" gibi alttaki butonlar ekran dışına itiliyordu. Düzeltme: yeni
  `_CONTROLS_PANEL_MIN_HEIGHT=160` ile kontrol paneli (yükle→kayıtlı modeller arası HER ŞEY)
  ayrı bir `QWidget`+`QVBoxLayout`'a taşınıp KENDİ `QScrollArea`'sına (`setWidgetResizable
  (True)`) sarıldı; canvas (üstte, stretch=1) büyüme önceliğini korur. Dialog artık ~300px
  yüksekliğe kadar serbestçe küçülüp büyüyebiliyor (`minimumSizeHint()` eskiden ~800, şimdi
  ~300), "Sil" dahil HER buton her zaman (gerekirse panel içi kaydırarak) erişilebilir kalır.
  `tests/ui/test_shape_matching_dialog.py::
  test_dialog_can_be_shrunk_well_below_default_size_and_delete_button_stays_reachable` bunu
  hem minimum-yükseklik hem de küçültülmüş pencerede "Sil" butonunun HÂLÂ `isVisible()`
  olduğunu doğrulayarak kanıtlıyor.
- **"Yüklediğim görüntü çok küçük kalabiliyor":** kök neden, `ImageView.zoom_reset()`'in
  (zoom=1.0 -- `_rescale()` görüntüyü canvas viewport'una SIĞDIRIR) SADECE "Sığdır" butonuna
  basılınca çağrılması; bir ÖNCEKİ referansta uzaklaştırma/yakınlaştırma kullanılmışsa `_zoom`
  sıfırlanmadan KALIYORDU ve galeriden YENİ bir referans seçildiğinde (`_activate_reference`
  → `_use_reference_image`) o eski zoom seviyesi yeni görüntüye de uygulanıp küçük
  görünmesine yol açıyordu. Düzeltme: `_use_reference_image` artık HER çağrıda `self._canvas.
  zoom_reset()` çağırıyor -- her referans değişiminde/yeni yüklemede baştan sığdırılmış bir
  görünümle başlanır. `tests/ui/test_shape_matching_dialog.py::
  test_activating_a_new_reference_resets_zoom_to_fit` bunu doğruluyor (bir referansta
  uzaklaştırıp ikinci bir referansa geçince zoom'un 1.0'a döndüğünü kanıtlayarak).

**Devam — kullanıcı "Ölçüm Aracı"na (`ui/dialogs/measurement_tool_dialog.py`) birden çok
ölçüm (çember çap/çevre, birden fazla çubuk/ruler) desteği ve "ölçüm varken Kare Yakala
kullanabilme" istedi** (bu istek, kod okumadan yapılmış bir ön-analizin kullanıcı tarafından
relay edilmesiyle geldi — analiz kodu okumadan "kavramsal olarak engelleyici bir neden yok"
diyordu, gerçek kod okunup doğrulandı).
- **Çoklu ölçüm (çizgi + çember), `analysis.color_props`/`texture_props`'un "Elle ROI Çiz"
  (RoiCanvas çoklu-ROI) DESENİYLE AYNI mantıkla eklendi:** `MeasureCanvas`'a (`ui/widgets/
  measure_canvas.py`) yeni `set_multi_mode(True)` (SADECE bağımsız Ölçüm Aracı kullanır, lens/
  yükseklik kalibrasyon dialogları HİÇ çağırmaz — varsayılan `False` iken davranış eskisiyle
  BİREBİR aynı, `tests/ui/test_measure_canvas.py::
  test_single_shot_mode_unaffected_by_multi_mode_being_off` bunu kanıtlıyor) + `set_mode
  ("LINE"/"CIRCLE")`. Multi modda her tamamlanan (A+B) ölçüm bir listeye (`_line_measurements`/
  `_circle_measurements`) EKLENİR ve HEMEN yeni bir ölçüme başlanabilir (tek-ölçüm modun "3.
  tıklama yeni A başlatır" davranışının aksine); ÇEMBER modunda A=merkez, B=kenar noktası,
  yarıçap=aralarındaki mesafe. Sağ tıklama en yakın çizgiyi/çemberi (widget-koordinatında
  nokta-doğru-parçası mesafesi / merkez-mesafesi-yarıçap farkı, `RoiCanvas`'ın
  `_HANDLE_HIT_RADIUS` deseniyle AYNI tolerans) siler. Yeni `measurements_changed` sinyali
  (mevcut `measurement_made` DEĞİŞMEDEN, lens/yükseklik dialogları hâlâ onu dinliyor) her
  ekleme/silmede yayınlanır. `MeasurementToolDialog`'a yeni "Ölçüm Türü" açılır listesi
  (Çizgi/Çember) + "Ölçümleri Temizle" butonu eklendi, sonuç etiketi artık TEK bir mesafe
  değil TÜM ölçümleri numaralı satırlar olarak listeliyor ("Çizgi 1: ... px ≈ ... mm",
  "Çember 1: çap=...px çevre=...px ≈ ...mm/...mm").
- **"Ölçüm varken Kare Yakala kullanılamıyor" iddiası kod okunarak İNCELENDİ ve YANLIŞ
  çıktı:** `_capture_photo_action`/`_capture_photo_button`'ın etkin/devre dışı durumu
  (`main_window.py`) SADECE `start_camera`/`stop_camera`'ya bağlı, `MeasurementToolDialog`
  (non-modal, `self`=MainWindow parent'lı) açık olup olmamasına HİÇ bakılmıyor — yapısal
  bir engel YOK. `tests/ui/test_main_window.py::
  test_capture_photo_stays_usable_while_measurement_tool_is_open` Ölçüm Aracı AÇIKKEN gerçek
  bir yakalama yapıp bunu kanıtlıyor. Kod DEĞİŞTİRİLMEDİ (yapılacak bir düzeltme yoktu) — bu
  madde sadece kalıcı bir regresyon testiyle DOĞRULANDI.

**Devam — kullanıcı "şekil öğretme/bul kısmını komple yeniden tasarlayalım" dedi ve gerçek bir
HALCON HDevelop kod örneği paylaştı. Bu, aylardır kovalanan "overlay hâlâ kutu/dikdörtgen gibi
çiziyor" sorununun KESİN kök nedenini ortaya çıkardı: HALCON `get_generic_shape_model_result_
object(Objects, MatchResultID, 'all', 'contours')` ile bir eşleşmeyi çizerken modelin EĞİTİMDE
zaten çıkardığı GERÇEK kenar noktalarını bulunan pozisyona taşıyıp DOĞRUDAN çiziyor — AYRI bir
Otsu/siluet-kontur çıkarımı YOK. Bizim kodumuz (önceki turlarda inşa edilen `ShapeModel.
contour`/`_contour_parts_from_mask`/çok-parçalı poligon sistemi) ise overlay'i eşleştirmede
kullanılan noktalardan TAMAMEN AYRI, ikinci bir Otsu-tabanlı siluet tespitinden türetmeye
çalışıyordu — bu yüzden Otsu (yerel aydınlatma, harflerin birleşmesi, ters-çevir modunda "temiz
bir sınır yok" vb. NEDENİYLE) her başarısız olduğunda overlay dikdörtgene düşüyordu. Kök sorun
HİÇBİR ZAMAN "Otsu'yu iyileştirmek" değildi — overlay'in YANLIŞ veri kaynağını kullanmasıydı.
Önceki turlardaki "çok-parçalı kontur"/"convexHull" notları (yukarıda) ARTIK TARİHSEL — o
sistem BÜTÜNÜYLE kaldırıldı, aşağıdaki yeniden tasarımla değiştirildi.**

- **`core/shape_matching.py` — overlay artık `ShapeModel.contour` yerine `levels[0].points`
  çizer:** `ShapeModel.contour` alanı VE ilgili `to_dict`/`from_dict`/`_parse_contour` mantığı
  TAMAMEN KALDIRILDI (`corners` alanı KALDI — eksen uzunluğu/NMS mesafesi için hâlâ gerekli).
  `train_shape_model`'in `contour_points` parametresi KALDIRILDI. `render_match_overlay` artık
  `model.levels[0].points`'i (`* match.scale`, döndürülüp `+ [match.x, match.y]`) küçük
  noktacıklar (`cv2.circle`, yarıçap `scale_factor`'a göre ölçekli) olarak çiziyor — merkez
  artı-işareti/eksen çizgisi/etiket DEĞİŞMEDİ. **Yan fayda:** eski (bu değişiklikten ÖNCE)
  kaydedilmiş `.json` modellerinde fazladan bir `"contour"` anahtarı olsa bile `from_dict` bunu
  hiç OKUMADIĞINDAN sessizce yoksayılır, ÇÖKME olmaz — yani kullanıcının önceden eğittiği "eti"
  gibi modeller bile YENİDEN EĞİTİLMEDEN bir sonraki "Şekil Bul"da yeni çizimden faydalanır.
- **HALCON'daki gibi otomatik piramit derinliği:** HALCON örneği `inspect_shape_model` ile her
  seviyenin kenar bölgesi ALANINI inceleyip 15px'in altına düşene kadar `NumLevels`'i artırıyor.
  `train_shape_model`'in `num_levels`'i artık `int | None` — `None` (yeni `_AUTO_LEVEL_MIN_
  POINTS=15`, `_AUTO_LEVEL_MAX_LEVELS=6` sabitleriyle) piramit derinliğini bir sonraki seviyenin
  nokta sayısı 15'in altına düşene kadar OTOMATİK artırır (`level==0` HER ZAMAN dahil edilmeye
  çalışılır, 0 nokta varsa hâlâ hata verir — DEĞİŞMEDİ). Bir TAM SAYI verilirse (varsayılan/
  manuel davranış) HİÇBİR ŞEY DEĞİŞMEZ. `shape_matching_dialog.py`'de yeni "Otomatik (HALCON
  tarzı)" checkbox'ı (`_auto_num_levels_checkbox`) işaretliyse spinbox devre dışı kalır,
  `_on_train` `num_levels=None` geçirir.
- **`core/auto_objects.py` — opsiyonel daha sağlam tespit (`robust=True`):** kullanıcı "daha
  sağlam olsun, biraz yavaşlasa da olur" dedi (eğitim TEK SEFERLİK, canlı kamerada değil).
  `detect_objects()`'e yeni `robust: bool = False` parametresi (varsayılan, `color_props`/
  `texture_props`'un canlı-kamera yolu, davranış/performans BİREBİR aynı kalır). `True` iken
  yeni `_edge_based_foreground_mask` (medyan-tabanlı otomatik Canny eşiği + dilate ile kırık
  kenar köprüleme + dış kontur doldurma) mevcut Otsu/manuel-eşik sonucuyla `cv2.bitwise_or` ile
  BİRLEŞTİRİLİR. **Dikkat edilmesi gereken yan etki:** bu, İNCE/BOŞ (hollow outline) şekillerin
  GÖRÜNÜR alanını (Canny+doldurma ile içi dolduğundan) orantısız büyütüyor — testlerdeki 15x15
  2px'lik "gürültü karesi" örneği bu yüzden 10x10'a küçültüldü (büyüdükten sonra bile üçgenin
  `_MULTI_PART_AREA_RATIO=0.15` eşiğinin AÇIKÇA altında kalması için, bkz.
  `test_robust_edge_assist_recovers_filled_interior_otsu_alone_misses`). `shape_matching_
  dialog.py::_build_auto_contour_mask` artık HER ZAMAN `detect_objects(crop, robust=True)`
  çağırıyor — Rect/Circle/Poligon'un HEPSİ otomatik faydalanır.
- **"Seçenekler kalsın ama hepsi üst üste kullanılabilsin" — Poligon modunda "Konturu Otomatik
  Algıla" artık ETKİN:** eskiden bu checkbox Poligon modunda TAMAMEN GİZLİ/devre dışıydı (Rect/
  Circle'da vardı, tutarsızlık). `_set_draw_mode_widgets_visible` artık checkbox'ı HER 3 modda
  da görünür bırakıyor; Poligon'un `_on_train` dalı artık Circle'ın AYNI desenini izliyor:
  çizilen poligonun maskesi, checkbox işaretliyse `_build_auto_contour_mask`'ın poligonun
  bbox'ı içinde bulduğu nesneyle KESİŞTİRİLİR (poligonun dışına asla taşmaz).
  `_CONTOUR_DILATION_PX` sabiti ve `_contour_parts_from_mask` metodu (artık kullanılmıyor)
  TAMAMEN KALDIRILDI, `_dilate_mask` eski tek-parametreli (`_MASK_DILATION_PX` sabit) haline
  döndü.
- Test etkisi: `tests/core/test_shape_matching.py`/`tests/ui/test_shape_matching_dialog.py`
  içindeki ~15 kontur-özel test KALDIRILDI/YENİDEN YAZILDI (artık `model.levels[0].points`'in
  beklenen bölgede olup olmadığını `_has_point_near` ile veya `render_match_overlay`'in
  piksel-bazlı nokta-bulutu çıktısını doğrudan kontrol ediyorlar); `tests/core/test_auto_
  objects.py`'ye `robust=True`/`robust=False` regresyon testleri eklendi. Tüm paket (919 test)
  yeşil.

**Devam — kullanıcı iki ayrı şey bildirdi: "model eğit penceresinin boyutlandırması çok kötü,
aşağıdaki menüler gözükmüyor, onu da tam ekrana alabilelim" + "şekil bulma da ötelemeyi x,y
şeklinde yaz ve kalibrasyon varsa cm şeklinde yaz".**
- **Pencere boyutlandırma kök nedeni (ölçülerek doğrulandı):** `ShapeMatchingDialog`'un ana
  `QVBoxLayout`'unda `controls_scroll_area` stretch=0 idi -- Qt, stretch=0 bir widget'a SADECE
  kendi `QScrollArea.sizeHint()`'ini verir, ve bu değer QScrollArea'nın içerik boyutundan
  BAĞIMSIZ küçük bir üst sınıra kadar sıkıştırılır (ölçülen: 432x288) -- oysa gerçek içerik
  (yükle/çizim modu/eğit/kayıtlı modeller TÜMÜ) 587px yükseklik istiyordu. Yani panel
  VARSAYILAN pencere boyutunda BİLE her zaman kendi içinde ~300px'lik gizli bir kaydırmaya
  ihtiyaç duyuyordu -- "Sil" gibi alt satırlar bu yüzden görünmüyordu (önceki turdaki
  `QScrollArea` sarmalama düzeltmesi YETERLİ değildi, sadece küçültmeyi mümkün kılmıştı,
  varsayılan görünürlüğü değil). Düzeltme: canvas/kontrol paneli artık bir `QSplitter`
  (`content_splitter`, dikey) içinde -- `setSizes([400, 620])` ile varsayılanda kontrol
  panelinin TAM içeriğini kaydırmadan gösterir, kullanıcı tutamacı sürükleyerek alanı
  istediği gibi yeniden dağıtabilir. `_DEFAULT_DIALOG_SIZE` 1000x850 -> 1100x1000 büyütüldü.
- **"Tam ekrana alabilelim":** varsayılan bir `QDialog`'da (Windows'ta) büyüt/küçült düğmeleri
  YOKTUR. `__init__`'te `self.setWindowFlags(... | Qt.WindowType.WindowSystemMenuHint |
  Qt.WindowType.WindowMinMaxButtonsHint)` eklendi -- başlık çubuğunda büyüt/küçült düğmeleri
  ve çift-tıklayarak maksimize etme artık çalışıyor.
- **Öteleme artık x,y bileşenli + cm cinsinden:** `operators/builtin/shape_match.py::run()`
  eskiden tek bir skaler `displacement_px`/`displacement_mm` (Öklid mesafesi, YÖN bilgisini
  gizliyordu) üretiyordu. Artık HER ZAMAN (reference_center varsa) işaretli `displacement_x_px`/
  `displacement_y_px` de eklenir; `mm_per_px` kalibrasyonu varsa `displacement_cm`/
  `displacement_x_cm`/`displacement_y_cm` da eklenir (mm DEĞİL cm -- gerçek kullanıcı isteği,
  öteleme mesafeleri genelde mm'den daha büyük/okunması daha kolay bir birimde anlamlı).
  `width_mm`/`height_mm` boyut alanları kapsam DIŞI bırakıldı, hâlâ mm (bu turun isteği SADECE
  öteleme birimiydi). `measurements_summary.py` ("öteleme=x:-0.50 y:+0.50cm") ve
  `main_window.py::_on_hover_measurement_changed` ("Öteleme: x=... y=... cm") ikisi de
  güncellendi. Eski `displacement_mm` alanı TAMAMEN kaldırıldı (geriye dönük uyumluluk
  gerekmiyor -- bu sadece canlı `measurements` sözlüğünde üretilen bir alan, diskte
  saklanmıyor).

**Devam — kullanıcı iki ayrı şey istedi: "ölçüm de dahil her alanda kare yakalayıp yan ekrana
atabilmek istiyorum, bunu kontrol et" + "kalibrasyon seçili olduğu her senaryoda pixelin
yanında mm de yazsın veya cm".**
- **Kare Yakala denetimi:** `capture_store.save_capture` (paylaşılan "Yakalanan Kareler"
  galerisi) üç yerde vardı: `lens_calibration_dialog.py` (`source="lens"`), `height_scale_
  calibration_dialog.py` (`source="height_scale"`), `main_window.py` (`source="live"`/
  `"filtered"`). `measurement_tool_dialog.py` ve `shape_matching_dialog.py`'de HİÇ YOKTU —
  ikisine de aynı desende (`frame_captured = Signal()`, ana pencere `dialog.frame_captured.
  connect(self.capture_gallery_panel.refresh)` ile bağlar, `lens`/`height_scale`
  diyaloglarındaki AYNI desen) bir "Kareyi/Referansı Galeriye Ekle" butonu eklendi:
  - `MeasurementToolDialog`: yeni `_capture_button` (mod satırında, "Ölçümleri Temizle"nin
    yanında) `self._image`'ı (`source="measurement"`) kaydeder; görüntü `None`'sa (dialog
    boş açılmışsa) buton baştan devre dışı.
  - `ShapeMatchingDialog`: bu diyalogda CANLI KAMERA YOK (sadece yüklenmiş/sürüklenmiş
    referans görüntüler) — yeni `_capture_button` (zoom çubuğunda) o an AKTİF referansı
    (`self._reference_image`, `source="shape_match"`) kaydeder; `_use_reference_image()`
    çağrılana kadar (henüz hiç referans yokken) devre dışı.
  - `capture_gallery_panel.py::_SOURCE_LABELS`'e `"measurement": "Ölçüm"` ve `"shape_match":
    "Şekil Eşleştirme"` eklendi (aksi halde galeri ham İngilizce anahtarı gösterirdi).
  - `flat_field_dialog.py` bilerek KAPSAM DIŞI bırakıldı — o diyalogda hiçbir canlı kamera
    `frame_provider` bağlantısı yok (SADECE dosyadan yükleme), capture-to-gallery eklemek
    önce o plumbing'i kurmayı gerektirirdi; kullanıcı da bunu özellikle istemedi.
- **Pikselin yanında mm/cm:** iki yerde kalibrasyon aktifken px değeri TAMAMEN
  gizleniyordu (sadece mm/cm gösteriliyordu) — ikisi de artık HER İKİSİNİ birden yazıyor:
  - `region_props.py::draw_measurements_overlay` (canlı görüntü üzerine yazılan kutu altı
    metni) artık `"35 x 20 px (3.50 x 2.00 cm)"` formatında — eskiden kalibrasyon aktifken
    SADECE cm yazıyordu, px hiç görünmüyordu.
  - `operators/builtin/shape_match.py::run()`'a yeni `width_px`/`height_px` alanları HER
    ZAMAN (kalibrasyon olsun olmasın) eklendi — eskiden bu alanlar hiç dışa açılmıyordu,
    sadece kalibrasyon varsa `width_mm`/`height_mm` üretiliyordu. `measurements_summary.py`
    ve `main_window.py::_on_hover_measurement_changed`'in "Boyut"/"boy=" satırları artık
    kalibrasyon varken `"50x25px (25.0x12.5mm)"` formatında ikisini birden gösteriyor.
  - `x`/`y` (mutlak konum) ve LAB/GLCM gibi piksel-tabanlı OLMAYAN alanlara bilerek
    DOKUNULMADI — bunlar zaten fiziksel bir mm/cm karşılığı olan "boyut/mesafe" alanları
    değil (konum keyfi bir görüntü orijinine göredir, LAB/GLCM zaten birimsiz).
- Test etkisi: `test_region_props.py`'deki kalibrasyon-farkı testi (60x60 tuval) yeni (daha
  uzun, px+cm birlikte) metnin görünür alana SIĞMADIĞI için (metnin farklılaşan kısmı
  canvas dışına taşıp iki modu piksel-özdeş gösteriyordu) 220px genişliğe büyütüldü;
  `test_shape_match.py`'nin tam-alan-seti testine `width_px`/`height_px` eklendi; ikisi de
  yeni davranışı ayrıca doğrudan doğruluyor. 3 yeni test (2 capture-buton, dialog başına 1).
  Tüm paket (922 test) yeşil.

**Devam — kullanıcı "kalibrasyon ayarı kendi kendine kaybolabiliyor, sorunu bul ve çöz" dedi.**
Gerçek, tekrarlanabilir bir bug bulundu (kod okuması + bug'ı YAKALAYAN bir regresyon testiyle
kanıtlandı, aşağıda açıklanan düzeltme geçici olarak geri alınıp testin GERÇEKTEN başarısız
olduğu doğrulandı):
- **Kök neden:** `main_window.py::_push_mm_per_px` (otomatik kalibrasyon akışının — lens
  profili/düzlem rektifikasyonu/yükseklik modeli/manuel "Bant Yüksekliği Değişti" — HER
  tetiklenişinde `analysis.region_props`/`geom.shape_match` node'larının `mm_per_px`
  parametresini dolduran TEK yer) `node.params`'ı `ParamForm`'un (parametre paneli
  widget'ları) HABERİ OLMADAN doğrudan güncelliyordu. Güncellenen node o an panelde
  SEÇİLİ/gösteriliyorsa, `ParamForm._values` (formun kendi önbelleği, `set_params()`
  çağrılmadıkça hiç yenilenmez) ESKİ `mm_per_px` değerinde kalıyordu. Kullanıcı sonra AYNI
  node'da BAŞKA bir parametreyi değiştirdiğinde (ör. "Min. Alan" sürgüsü) —
  `ParamForm._on_change` formun TÜM `_values`'ini (ESKİ mm_per_px DAHİL) `params_changed`
  ile yayınlıyor, `main_window.py::_on_params_changed` da bunu `node.params`'ın ÜZERİNE
  KOŞULSUZ yazıyordu (`self.graph.nodes[...].params = values`) — otomatik hesaplanan
  kalibrasyon, kullanıcı ona hiç DOKUNMAMIŞ olsa bile SESSİZCE sıfırlanıyordu. Bu, kullanıcının
  "kendi kendine kayboluyor" izlenimini tam açıklıyor: tetikleyici görünüşte alakasız bir
  parametre değişikliğiydi.
- **Düzeltme:** `_push_mm_per_px`, güncellediği node `self._selected_node_id` ile eşleşiyorsa
  `ParamForm.set_value("mm_per_px", mm_per_px)` de çağırıyor artık — bu metot (`camera_settings_
  panel.py`'de donanım-clamp senkronizasyonu için ZATEN var olan, `main_window.py`'nin daha
  önce HİÇ kullanmadığı bir mekanizma) formun önbelleğini VE ekrandaki widget'ı günceller,
  sinyal TEKRAR yaymadan (gereksiz bir döngü/yeniden hesaplama tetiklenmez).
  `tests/ui/test_main_window.py::test_push_mm_per_px_does_not_get_silently_reverted_by_
  unrelated_param_edit` düzeltme GERİ ALININCA gerçekten `0.0 == 0.4` başarısızlığıyla
  çöktüğü doğrulanarak yazıldı (yani bug'ı YAKALIYOR, sahte-yeşil değil); `::test_push_mm_
  per_px_does_not_touch_form_of_a_different_selected_node` da farklı bir node seçiliyken
  formun YANLIŞ alanlarına yazılmadığını kontrol ediyor. Tüm paket (924 test) yeşil.

**Devam — kullanıcı iki ayrı bug bildirdi: "hsv filtresi otomatik siyah beyaza çeviriyor
çevirmemesi lazım" + "uygulama kendi kendine tam ekrandan çıkıyor ve ekran kayıyor bazı
panellere erişemiyorum o yüzden tam ekrandan hiç çıkmasın".**
- **HSV kök neden:** `operators/builtin/color_modes.py::HsvOp.run()`'da `apply_mask` (Renk
  Aralığı Maskesi) açıkken ham `cv2.inRange(...)` sonucu (tek kanallı 0/255 İKİLİ görüntü)
  doğrudan döndürülüyordu — yani bu seçenek açılınca görüntü HER ZAMAN koşulsuz siyah/beyaza
  dönüyordu. Düzeltme: `cv2.bitwise_and(corrected_bgr, corrected_bgr, mask=mask)` ile SADECE
  aralık DIŞINDAKİ pikseller siyaha boyanıyor, aralık İÇİNDEKİ pikseller (dolayısıyla
  görüntünün RENGİ) korunuyor — çıktı HER ZAMAN 3 kanallı kalıyor. `shadow_removal` açıkken
  (V kanalı histogram eşitlenmiş) düzeltilmiş renk kullanılıyor (aynı `corrected_bgr`,
  `apply_mask` kapalıyken zaten dönen görüntüyle TUTARLI). `tests/operators/test_color_
  modes.py::test_hsv_apply_mask_keeps_color_instead_of_turning_binary` (eski `..._returns_
  binary_mask` testinin yerini aldı) bunu doğruluyor. **Bilinen yan not:** bu çıktıyı
  `segment.connected_components`'e bağlayan biri varsa, o operatör artık temiz bir 0/255
  ikili yerine "aralık içi orijinal renk / aralık dışı siyah" bir görüntü görüp KENDİ Otsu
  eşiklemesini bunun üzerinde çalıştırır — çoğu sahnede sorun çıkarmaz ama çok koyu renkli
  bir "aralık içi" bölge siyah zeminle düşük kontrastlıysa segmentasyon kalitesi biraz
  değişebilir (kullanıcı açıkça "siyah/beyaza dönüşmesin" istediğinden bilinçli kabul edilen
  bir ödünleşim).
- **Tam ekran kök neden:** kod tabanında pencere durumuna (`showFullScreen`/`showMaximized`/
  `windowState`) dokunan HİÇBİR yer yoktu — uygulama `cli.py`'de sabit `resize(1000,700)` +
  `show()` ile NORMAL boyutta başlıyordu, kullanıcı pencereyi kendisi büyütüyordu (Windows'ta
  "tam ekran" genelde "büyütülmüş/maximized" anlamında kullanılır). Qt/işletim sistemi
  tarafında (başlık çubuğunu çift tıklama, kenara sürükleyip bırakma/"Aero Snap", dock panel
  yeniden düzenlemesinin dolaylı bir yan etkisi vb.) pencere BEKLENMEDİK şekilde düz "geri
  yüklenmiş" (`Qt.WindowState.WindowNoState`) hale dönebiliyor — bu da dock panellerin
  konumunun/boyutunun küçük pencereye göre yeniden hesaplanıp bazı panellerin erişilemez hale
  gelmesine yol açıyordu. Düzeltme İKİ parçalı: (1) `cli.py` artık `window.showMaximized()`
  ile başlıyor; (2) yeni `MainWindow.changeEvent()` override'ı, pencere durumu TAM OLARAK
  `WindowNoState`'e dönerse (küçültülmüş/minimized DEĞİL — o SERBEST bırakılır, sadece "düz
  geri yüklenmiş" hal) `showMaximized()`'ı HEMEN tekrar çağırıyor; bu senkron olarak
  `changeEvent` içinde olduğundan kullanıcı ARA görünüşte bile küçülmüş pencereyi göremiyor.
  `tests/ui/test_main_window.py::test_window_re_maximizes_itself_if_restored_to_normal_state`
  düzeltme GERİ ALININCA gerçekten `isMaximized()==False` ile başarısız olduğu doğrulanarak
  yazıldı; `::test_window_can_still_be_minimized` küçültmenin hâlâ SERBEST kaldığını
  doğruluyor (Qt'de "küçültülmüş" durum genelde "büyütülmüşken küçültüldü" bayrağını da
  birlikte taşıdığından `isMaximized()` o testte KONTROL EDİLMİYOR, sadece `isMinimized()`).
  Tüm paket (926 test) yeşil.

**Devam — kullanıcı "kalibrasyon üzerine bant yüksekliği değişti diyip ölçüm alıyorum fakat
çok yanlış ölçüyor" + "roi seçtikten sonra başka bir sekmeye geçince zoomluyor" bildirdi;
"lab analizi yaparken görüntü birden gidebiliyor" ayrıca bildirildi ama kod incelemesiyle
kesin kök nedeni bulunamadı (aşağıya bkz.).**
- **Bant yüksekliği değişti → ÇOK YANLIŞ ölçüm (gerçek, ciddi bug, kanıtlandı):** `_on_adjust_
  height_delta`'nın `_plane_rectification` dalı (Açılı/homografi-tabanlı kalibrasyon) SADECE
  `mm_per_px` ALANINI güncelliyordu, `homography`/`output_size`'a HİÇ dokunmuyordu. Ama
  `PlaneRectification.homography` matrisinin KENDİSİ `compute_plane_rectification`'da
  `1/mm_per_px` ölçeğini zaten İÇİNE gömüyor (`world_to_output`, bkz. `core/plane_
  rectification.py`) — yani `_on_camera_tick` HER karede hâlâ ESKİ ölçekle rektifiye
  ediyordu (`self._plane_rectification.rectify(frame)`), ama ölçüm operatörleri (`region_
  props`/`shape_match`) piksel→mm dönüşümünde YENİ `mm_per_px`'i kullanıyordu — tam olarak
  `(eski_mm_per_px/yeni_mm_per_px)` oranında SİSTEMATİK bir ölçüm hatası (ör. yükseklik
  %14 değiştiyse TÜM ölçümler o oranda yanlış çıkıyordu). Düzeltme matematiksel olarak temiz:
  `homography_yeni = diag(oran,oran,1) @ homography_eski` (`oran = eski_mm_per_px/yeni_mm_
  per_px`) — bu, `world_to_output @ (eski world_to_output)^-1`'in SADECE düzgün bir skaler
  matrise indirgendiği türetilerek bulundu, orijinal `rvec`/`tvec`/`camera_matrix` hiç
  saklanmasa bile (dataclass sadece bileşik `homography`'yi tutuyor) doğru çalışıyor.
  `output_size` de AYNI oranla ölçekleniyor. `tests/ui/test_main_window.py::test_adjust_
  height_delta_rescales_plane_rectification_homography_and_mm_per_px` (eski, YANLIŞ
  davranışı -- `homography is original.homography` -- doğrulayan test GÜNCELLENDİ) +
  yeni `::test_adjust_height_delta_keeps_world_point_to_mm_consistent` (aynı ham pikselin
  ayarlamadan ÖNCE/SONRA AYNI gerçek-dünya mm konumuna karşılık geldiğini doğrudan kanıtlıyor)
  bunu doğruluyor.
- **"ROI seçip sekme değiştirince zoomluyor" (muhtemel neden, doğrudan tekrarlanamadı ama
  düşük riskli/savunmacı bir düzeltme uygulandı):** `right_tabs` (Parametreler/Kamera
  Ayarları/Sonuçlar) `central_splitter`'ın bir parçası; bir sekmenin içeriği FARKLI bir
  `sizeHint()` doğurup Qt'nin panolar arası alanı sessizce yeniden dağıtmasına (dolayısıyla
  orta/görüntü panelinin genişliğinin, `ImageView._rescale()`'in sığdırma ölçeğini
  kaydırarak) yol açması TEORİK olarak mümkün — headless/offscreen test ortamında (gerçek
  bir kamera/GenICam nod haritası olmadan) bu tam olarak YENİDEN ÜRETİLEMEDİ (muhtemelen
  gerçek "Kamera Ayarları" sekmesinin dolu içeriği farklı bir minimum genişlik talep
  ediyor). Yine de `right_tabs.currentChanged`'e bağlanan yeni `MainWindow._on_right_tabs_
  changed`, sekme değişiminden HEMEN ÖNCEKİ `central_splitter.sizes()`'ı yakalayıp bir
  sonraki olay turunda (`QTimer.singleShot(0, ...)`) YENİDEN uyguluyor — Qt'nin kendi
  otomatik yeniden-düzenlemesi ne olursa olsun görüntü paneli sekme geçişlerinden TAMAMEN
  bağımsız kalıyor, kullanıcı SADECE bölme sınırını elle sürükleyerek boyutları
  değiştirebiliyor. `tests/ui/test_main_window.py::test_switching_right_tabs_reasserts_
  splitter_sizes` mekanizmayı (yakala → sonraki turda yeniden uygula) doğrudan bir spy ile
  doğruluyor.
- **"LAB analizi sırasında görüntü birden siyah/boş kalıyor" — KESİN kök neden BULUNAMADI:**
  `analysis.color_props::run()`'ın üç modu da (varsayılan tek-satır, Otomatik Nesne Tespiti,
  Elle ROI Çiz) tek tek incelendi — `detect_objects`/`parse_roi_list`/overlay çizim
  fonksiyonlarının hiçbirinde boş nesne listesi/bozuk ROI/sınır-dışı koordinat gibi durumlar
  için çökme riski bulunamadı (hepsi zaten sessizce boş sonuç/no-op'a düşüyor). `_build_
  preview_frame`'in `ok=False` dalı (herhangi bir operatör `run()` içinde exception
  fırlatırsa `display_image=None` + `status_label`'a "Hata: ..." yazan) MEVCUT ve önceden
  test edilmiş bir güvenlik ağı — yani ekran TAMAMEN boş kalıyorsa (kullanıcının doğruladığı
  belirti) teorik olarak `status_label`'da bir hata mesajı VE `~/.imgflow/logs/imgflow.log`
  dosyasında bir kayıt oluşması GEREKİR (bkz. `io_utils/app_log.py`) — ama bu oturumda
  kullanıcıdan ne hata mesajı metni ne de log dosyası içeriği paylaşılmadığından kesin kök
  neden doğrulanamadı. Sonraki adım: bir sonraki oluşumda `status_label` metnini VE `~/.
  imgflow/logs/imgflow.log`'un ilgili satırlarını kontrol et.

**Devam — kullanıcı üç şey istedi: "lab analizinde nesne tespitinde dalgalanma gözüksün",
"halcon'da benzer özellik varsa karşılaştır/iyileştir", "zaman farklarına bak, daha hızlı
yapılmaya çalış", ve genel olarak "pencere geçişleri, gereksiz boyut değiştirme" hatalarını
araştırmamı istedi.**
- **Nesne tespitinde dalgalanma:** `analysis.color_props`'un Elle ROI Çiz modunda zaten var
  olan min/max ARALIK alanları (`l_min/l_max` vb.) artık Otomatik Nesne Tespiti modunda da
  hesaplanıyor — `measurements_summary.py`/`render_color_overlay_multi` bu alanları zaten
  JENERİK olarak (`"l_min" in m` kontrolüyle) gösterdiğinden UI tarafında EK kod GEREKMEDİ.
  Bu, HALCON'un `min_max_gray(Regions, Image, Percent, Min, Max, Range)` operatörünün bir
  bölge üzerindeki Min/Max/Range çıktısına karşılık geliyor (`l_mean`/`l_std` zaten HALCON'un
  `intensity`'sine karşılık geliyordu) — dosya docstring'ine bu karşılaştırma eklendi.
- **Performans (ölçülerek doğrulandı, `_to_lab`/`cv2.meanStdDev` optimizasyonu):**
  `ColorPropsOp.run()` eskiden TÜM görüntüyü baştan LAB'a çeviriyordu (`lab = _to_lab(image)`)
  sonra HER moda göre kırpıyordu — 1920x1080'de bu dönüşüm tek başına ~12-14ms sürüyordu, ama
  birkaç küçük ROI/nesneyle çalışırken bu maliyetin neredeyse tamamı gereksizdi. Artık her dal
  SADECE ihtiyacı olan bölgeyi (tüm-görüntü modunda tüm kare, aksi halde SADECE ROI/nesne bbox
  kırpımı) dönüştürüyor. Ayrıca tüm-görüntü modundaki ortalama/std hesaplaması eski
  `lab[..., i].mean()/.std()` (kanal ekseninde STRIDED bir görünüm üzerinde 6 ayrı numpy
  geçişi) yerine TEK bir `cv2.meanStdDev()` çağrısı kullanıyor — ölçülen fark 1920x1080'de
  ~17x (32ms → 1.9ms sadece istatistik kısmı için). Toplam etki (1920x1080, gerçekçi sahne):
  tüm-görüntü modu 46.7ms → 12.2ms (~3.8x), Otomatik Nesne Tespiti 35.2ms → 14.8ms (~2.4x),
  Elle ROI Çiz (2 küçük ROI) 17.3ms → 1.4ms (~12.7x). Canlı kamera akışında bu operatör
  seçiliyken tek kare işleme süresini gözle görülür şekilde kısaltıyor.
  `tests/operators/test_color_props.py::test_per_object_enabled_reports_min_max_range_per_channel`
  yeni dalgalanma alanlarını doğruluyor (Otsu'nun aynı bağlı bileşeni İKİYE bölmemesi için
  YUMUŞAK bir gri-seviye gradyanı kullanılıyor — sert bir "yarısı siyah yarısı beyaz" denemesi
  siyah yarının ARKA PLANLA birleşip tek bir aydınlık yarı-nesne olarak algılanmasına yol
  açtığından İLK denemede test YANLIŞ senaryo kullanıyordu, düzeltildi).
- **Gerçek, tekrarlanabilir "gereksiz boyut değiştirme" bug'ı bulundu ve düzeltildi
  (`ui/panels/camera_settings_panel.py`):** canlı bir stres testiyle (MainWindow kurup
  "Kamera Ayarları" sekmesindeyken -- sekme DEĞİŞTİRMEDEN -- sahte bir Basler kamera bağlayıp
  `central_splitter`/pencere boyutunu ölçerek) doğrulandı: kamera bağlanıp `set_controller`
  gerçek GenICam kategorileriyle `_toolbox`'ı (QToolBox) doldurunca, panel HİÇBİR sekme
  değişimi olmadan bile `central_splitter`'ı zorlayıp TÜM ana pencereyi (kullanıcı hiç
  isteMEDEN) büyütüyordu — `main_window.py::_on_right_tabs_changed` (önceki bir turda
  "sekme değişince zoomluyor" için eklenmişti) SADECE sekme İNDEKSİ değişince devreye
  girdiğinden, kullanıcı zaten o sekmedeyken kamera bağlarsa bu korumadan hiç faydalanmıyordu.
  İKİ ayrı kök neden, İKİ ayrı düzeltme:
  1. `_toolbox` (QToolBox) doğrudan panelin düzenine ekliydi — sayfalar (GenICam
     kategorileri) dolunca büyüyen minimum boyut isteği ÖZÜMSENMEDEN yukarı (panel -> sekme ->
     splitter -> ana pencere) taşınıyordu. `ShapeMatchingDialog`'un kontrol panelini
     `QScrollArea`'ya sarmalayan AYNI çözüm: `_toolbox` artık `setWidgetResizable(True)` ile
     bir `QScrollArea` içinde.
  2. `_status_label` (bağlantı durumu metni) `setWordWrap` KULLANMIYORDU — önceki ayarlar
     otomatik geri yüklendiğinde (`main_window.py::start_camera`'nın eklediği " — önceki
     ayarlar geri yüklendi" son eki) metin uzayınca, sarmasız bir QLabel'ın minimumSizeHint'i
     TÜM metnin genişliğine eşit oluyordu. `setWordWrap(True)` TEK BAŞINA da YETMEDİ —
     "(BaslerCameraSource)" gibi boşluksuz tek bir "kelime" sarmayla bile bölünemediğinden
     minimumSizeHint hâlâ belirgin genişlikte kalıyordu (ölçülerek doğrulandı: sarmalı bile
     240px). Asıl çözüm yatay `QSizePolicy.Ignored`: layout bu widget'ın sizeHint/
     minimumSizeHint'ini TAMAMEN yok sayar, etiket mevcut alana göre sarar/kırpılır ama ASLA
     panelin (dolayısıyla ana pencerenin) minimum genişliğini büyütmez (izole ölçüm: aynı
     container'da Ignored OLMADAN minimum 354px, Ignored İLE 114px). Reprodüksiyon script'i
     ile doğrulandı: düzeltmeden ÖNCE pencere genişliği kamera bağlanınca 2540px'ten 3068px'e
     (528px) zıplıyordu, düzeltmeden SONRA sıfır (central_splitter/pencere boyutu birebir
     aynı kalıyor).
- **Dialog aç/kapa + rastgele boyutlandırma stres testi TEKRARLANDI** (önceki turdaki "dialog
  sızıntısı" testine benzer, ama bu kez rastgele boyutlandırmayla): tüm tekil-dialoglar (Lens/
  Yükseklik/Şekil Eşleştirme/Aydınlatma/ONNX/Yardım/Özel Filtre) 15'er kez açılıp kapanıp
  rastgele boyutlandırıldı, Ölçüm Aracı (taze-instance deseni) 16 kez yeniden açıldı, SONRA
  hepsi AÇIKKEN 60 rastgele boyutlandırma yapıldı — `QTest.qWait` ile olay döngüsü doğru
  pompalanınca (ilk deneme `app.processEvents()` tek başına yeterli değildi, `deleteLater()`
  sonrası gerçek yok etmeyi YAKALAYAMIYORDU) hiçbir sızıntı/çökme YOK, sonuçlar mevcut
  `test_reopening_measurement_tool_does_not_leak_previous_dialog` testiyle TUTARLI. Bu turda
  gerçek bir regresyon bulunamadı, sadece mevcut korumalar yeniden doğrulandı.
- **Yan bulgu — pytest-qt test paketinde ÖNCEDEN VAR OLAN, bu turdaki değişikliklerle
  İLGİSİZ bir kırılganlık bulundu (düzeltilmedi, kapsam dışı):** tam paket (929 test) bazen
  (~1/3 çalıştırmada) `test_main_window.py`'deki RASTGELE bir testin TEARDOWN'ında
  ("AttributeError: Slot 'MainWindow::' not found") tek seferlik bir hata veriyor — testin
  KENDİSİ (assert'leri) HER ZAMAN geçiyor, sadece qtbot'un otomatik widget temizliği sırasında
  Qt'nin olay kuyruğunda bir yerlerde kalmış bir bağlantı/timer çözülemiyor. Bisection ile
  KANITLANDI: bu turdaki `camera_settings_panel.py` değişiklikleri TAMAMEN geri alınsa BİLE
  (sadece `test_main_window.py` alt kümesi çalıştırılınca) aynı hata farklı bir testte ortaya
  çıkıyor — yani bu turun bir regresyonu DEĞİL, muhtemelen yüzlerce `MainWindow()` örneğinin
  hızla kurulup yok edildiği (`qtbot.addWidget`) büyük bir Qt test paketinde önceden var olan,
  zamanlamaya duyarlı bir teardown yarışı. Kök neden BULUNAMADI/düzeltilmedi (kapsam dışı,
  kullanıcı bunu istemedi) — ileride ele alınırsa hangi testin/timer'ın dangling kaldığını
  bulmak için `pytest -p no:cacheprovider --randomly-seed=<sabit>` ile deterministik bir
  tekrar üretim aranabilir.

**Devam — kullanıcıya yukarıdaki turdan sonra üç ek iş soruldu, üçünü de seçti: "şekil bul
performansı", "test paketi kırılganlığı", "doku analizinde başka HALCON karşılaştırması".**
- **Pytest-qt teardown kırılganlığı (kök neden BULUNDU ve DÜZELTİLDİ, önceki turda "kapsam
  dışı" bırakılmıştı):** `_LiveTickWorker`/`_BatchWorker` (bkz. `main_window.py`) arka planda
  çalışırken pencere kapatılırsa (`qtbot.addWidget`'ın otomatik teardown'ı -- `_on_camera_
  tick()` çağrılıp worker'ın bitmesi HİÇ beklenmeden pencere hemen kapanabiliyor, testlerin
  ~15 yerinde bu desen var) thread hâlâ CANLI iken `self` (MainWindow) yok edilebiliyordu;
  thread bitince kuyruklu/çapraz-thread `result_ready`/`finished` sinyalleri yok edilmiş/
  edilmekte olan nesneye ulaşmaya çalışıp ARA SIRA "AttributeError: Slot 'MainWindow::' not
  found" hatasına yol açıyordu (klasik "worker thread, receiver'dan önce join edilmedi" Qt
  yarışı). `closeEvent`'e `self._live_worker.wait()`/`self._batch_worker.wait()` eklendi --
  thread TAMAMEN bitene kadar bloke eder (sınırlı/tek-seferlik bir hesaplama, sonsuz döngü
  DEĞİL, normal kullanımda anlık sürer). Ölçüldü: düzeltmeden ÖNCE tam paket (929 test)
  ~3 çalıştırmada 1'inde bu hatayla/farklı bir testte flake veriyordu, düzeltmeden SONRA
  22/23 art arda çalıştırma tertemiz geçti (kalan tek istisna FARKLI bir semptomla, ayrı/daha
  nadir bir kalıntı olabilir -- KESİN sıfıra indiği garanti edilemez ama kanıtlanabilir
  ölçüde iyileşti). Gerçek kullanımda (bir kullanıcının uygulamayı normal kapatması) bu
  DEĞİŞİKLİK fark edilmez, sadece ağır bir hesaplama ortasında kapatılırsa birkaç ms'lik
  bir gecikme ekler.
- **Şekil Bul performansı — `_nms` mekansal ızgara optimizasyonu:** gerçekçi (gürültülü/
  dolu arka planlı) bir sahne ile profillenince (`cProfile`), kaba aramanın gevşetilmiş
  eşiği (`_COARSE_ACCEPT_LOOSENING`, KASITLI OLARAK dokunulmadı -- "kaçırma" regresyonu
  riski) yoğun sahnelerde BİNLERCE (ör. 8505) ham aday üretebiliyor; `_nms`'in eski düz
  `for cand in sorted(...): any(... for r in result)` deseni HER adayı `result`'taki TÜM
  önceki kabul edilmiş adayla (insertion sırasına göre) karşılaştırıyordu -- ölçülen: tek
  bir `find_shape_model` çağrısının önemli bir kısmı (profilde ~35%) buradaydı. Adaylar artık
  `dist_thresh` boyutunda hücrelere ızgaralanıp her aday SADECE kendi hücresi + 8 komşu
  hücredeki kabul edilmiş adaylarla karşılaştırılıyor -- karar mantığı/sıralaması BİREBİR
  AYNI (aynı sonuç kümesi, sentetik aday kümesiyle doğrulandı), SADECE hangi noktalarla
  karşılaştırıldığı daraltılıyor. Ölçülen kazanç: 8505 adayda `_nms` cProfile cumtime'ı
  ~2.8x düştü (busy sahne testinde 65.6ms/çağrı → 23.2ms/çağrı); temiz sahnede (10-20 aday)
  ızgara ek yükü ihmal edilebilir (~5 mikrosaniye). **Not:** toplam `find_shape_model` duvar-
  saati (wall-clock) ölçümü bu ortamda ÇOK GÜRÜLTÜLÜ çıktı (`_build_target_pyramid`'in kendi
  değişkenliği nms kazancını gölgeliyordu) -- gerçek iyileştirme cProfile'ın deterministik
  çağrı sayılarıyla doğrulandı, abartılı bir "Nx daha hızlı" toplam-süre iddiası YAPILMADI.
- **Doku analizinde HALCON karşılaştırması — rotasyon-bağımsız "average" yön modu eklendi:**
  `analysis.texture_props` zaten HALCON'un `gen_cooc_matrix`/`cooc_feature` çiftine BİREBİR
  karşılık geliyordu (aynı 4 özellik: contrast/energy/homogeneity/correlation, aynı 4 kanonik
  yön: 0/45/90/135, aynı mesafe parametrizasyonu) -- ek olarak HALCON dokümantasyonunda/
  yaygın pratikte bilinen bir sınırlama not edildi: TEK bir yön, banttaki ürün RASTGELE
  döndüğünde AYNI yüzey dokusu için farklı contrast/homogeneity ölçebilir (rotasyona duyarlı).
  Yaygın çözüm (4 yönün ayrı hesaplanıp ORTALAMASININ alınması) `angle` ENUM'una yeni bir
  `"average"` seçeneği olarak eklendi -- `compute_texture_features` bu değerde 4 yönü ayrı
  hesaplayıp (GLCM matrislerini TOPLAMAK yerine, çünkü bu yönler arası gerçek farkı örtük
  kaybederdi) ÖZELLİK değerlerinin aritmetik ortalamasını alıyor. Varsayılan (`"0"`) davranış/
  performans DEĞİŞMEDEN kalır -- bu TAMAMEN opt-in bir seçenek (4 kat daha yavaş, help
  metninde belirtildi). `tests/operators/test_texture_props.py::
  test_average_angle_matches_mean_of_four_directions` yönlü bir çizgi deseninde (0°/90°
  contrast'ı GERÇEKTEN farklı) ortalamanın doğru hesaplandığını kanıtlıyor.

**Devam — kullanıcı "hataları tara ve düzelt; gereksiz sayfa büyümeleri, otomatik kalibrasyon
kayıpları, kalibrasyon varsa pixelin yanında cm/mm, her işlemde kare yakalama, işlem süreleri,
lab otomatik ölçümde aralık" dedi (çoğu daha önce yapılmıştı; kod okunarak DOĞRULANDI, gerçek
BOŞLUKLAR bulunup kapatıldı).**
- **Gereksiz sayfa büyümesi — ölçülerek bulunan İKİ kök neden (önceki turdaki
  `camera_settings_panel` düzeltmesinden BAĞIMSIZ, hâlâ duruyorlardı):** (1)
  `ui/widgets/param_form.py`'deki parametre etiketleri (`QLabel(spec.label)`) sarmıyordu --
  uzun Türkçe etiketler (ör. "Bulanıklık Yarıçapı (px, sadece 'Yerel/Dinamik' modda)" tek
  başına 648px) `ParamForm`'un minimum genişliğini 858px'e, sağ sekme panelininkini 358 ->
  880px'e, ana pencereninkini 3110px'e çıkarıyordu; operatör seçmek `central_splitter`
  panolarını yeniden dağıtıp görüntü panelini daraltıyordu (kullanıcının "sekmeye geçince
  zoomluyor" şikayetiyle AYNI mekanizma). Düzeltme: `setWordWrap(True)` + `QFormLayout`'a
  `WrapLongRows`/`ExpandingFieldsGrow` + parametre sekmesinin TAMAMI bir `QScrollArea`'ya
  sarıldı (`main_window._build_layout`). (2) `main_window.status_label` sarmasızdı -- tek bir
  uzun hata mesajı ORTA panelin minimumunu 1010 -> 1314px'e çıkarıyordu; `camera_settings_
  panel._status_label` ile AYNI çözüm (`setWordWrap` + yatay `QSizePolicy.Ignored`), aynısı
  `hover_info_label`'a da uygulandı. **Ölçülen sonuç: 25 operatörün 16'sı splitter boyutunu
  değiştiriyordu, artık 0/25.**
- **Otomatik kalibrasyon kaybının İKİNCİ yolu** (birincisi -- `ParamForm` önbelleği -- önceki
  turda düzeltilmişti): `mm_per_px` düğümün `params`'ında YAŞADIĞI için, düğüm parametrelerini
  TOPLUCA geri yükleyen işlemler alanı sessizce sıfırlıyordu: **Geri Al/Yinele** (snapshot
  kalibrasyondan ÖNCE alınmışsa) ve **reçete yükleme** (reçete kendi kalibrasyon profilini
  taşımıyorsa). Kalibrasyon KAYNAĞI (`_plane_rectification`/`_reference_distance_mm`/lens
  profili) bu işlemlerden hiç etkilenmiyor -- yani veri duruyordu, sadece düğüme yazılmış
  kopyası siliniyordu. Yeni `main_window._reapply_active_calibration()`
  (`_restore_pipeline_snapshot` ve `load_recipe_from` sonunda çağrılır) `_compute_auto_mm_per_px()`
  -> `_last_auto_mm_per_px` sırasıyla aktif değeri geri yazar; hiçbir OTOMATİK kaynak yoksa
  HİÇBİR ŞEY yapmaz -- kullanıcının ELLE girdiği bir `mm_per_px`'in geri alınması meşru bir
  Geri Al'dır (`test_undo_still_reverts_a_manually_typed_mm_per_px` bunu koruyor).
- **px + cm/mm birlikte:** `geom.shape_match` yolu önceki turda düzeltilmişti ama
  `analysis.region_props`'un HOVER paneli (`_on_hover_measurement_changed`) kalibrasyon
  aktifken px'i TAMAMEN gizleyip sadece cm yazıyordu -- artık "50 x 20 px (2.50 x 1.00 cm)" /
  "1000 px² (2.50 cm²)" formatında; ayrıca hiç gösterilmeyen ÇEVRE de eklendi ("120 px (60.0 mm)").
- **Kare yakalama boşluğu:** `flat_field_dialog.py` capture-to-gallery deseni olmayan TEK
  diyalogdu (önceki turda "canlı kamera plumbing'i yok" diye kapsam dışı bırakılmıştı) --
  yüklenen referans görüntüsünü kaydeden bir "Kareyi Galeriye Ekle" butonu + `frame_captured`
  sinyali eklendi (diğer diyaloglarla AYNI desen), `capture_gallery_panel._SOURCE_LABELS`'e
  `"flat_field": "Aydınlatma Referansı"` eklendi. Artık HER diyalogda kare yakalanabiliyor.
- **Uzun süredir kovalanan pytest-qt "Slot 'MainWindow::' not found" flake'i KÖKÜNDEN
  çözüldü** (önceki turlarda iki kez "kısmen düzeltildi/kapsam dışı" diye bırakılmıştı; bu
  turda ÖLÇÜLEREK teşhis edildi). Yöntem: `-p no:randomly` ile hatanın DETERMİNİSTİK hale
  geldiği görüldü (baseline: `tests/ui/test_main_window.py` HER çalıştırmada 1 hata), sonra
  `destroyed.connect(...)` satırlarının tamamı geçici olarak devre dışı bırakılınca hatanın
  kaybolmasıyla kaynak kesinleştirildi. **Kök neden:** her tekil dialog
  `dialog.destroyed.connect(lambda: setattr(self, "_x_dialog", None))` ile, ALICISI
  MainWindow olan bir geri çağrıya bağlıydı. `destroyed` C++ nesnesi yok edilirken ateşlenir;
  bir dialog `deleteLater()` ile silinmeyi BEKLERKEN (Ölçüm Aracı her açılışta yeniden
  kurulur, eskisi silinmeyi bekler) ana pencere ÖNCE yok edilirse, sinyal daha sonra ÖLÜ bir
  pencerede slot arayıp bu hatayı üretiyordu -- ve hata onu YARATAN testte değil, o sırada
  hangi olay döngüsü dönüyorsa ORADA patladığı için hep "rastgele bir testin teardown'ı"
  gibi görünüyordu. **Düzeltme:** yeni `_connect_destroyed(dialog, handler)` bağlantıyı
  kurarken dialog'u `_tracked_dialogs`'a kaydediyor, `closeEvent` de bu listedeki HEPSİNİN
  bağlantısını kesiyor -- sadece o an izlenen tekil referanslara bakmak YETMİYORDU (silinmeyi
  bekleyen eski örnek hiçbir alanda tutulmuyor). Aynı turda iki ek (daha küçük) kaynak da
  kapatıldı: `closeEvent` artık `_param_debounce_timer`'ı durduruyor, ve
  `_on_right_tabs_changed`'in `QTimer.singleShot(0, self, lambda ...)` çağrısı `self`'in
  ÇOCUĞU olan gerçek bir `QTimer`'a (`_splitter_restore_timer`) çevrildi (bağlam-nesnesi
  aşırı yüklemesinin bekleyen atışı PySide'da güvenilir iptal edilmiyor).
  **Ölçüm (dürüst, SONRAKİ turda güncellendi): düzeltme uygulandığı anda
  `tests/ui/test_main_window.py` tek başına `-p no:randomly` ile 5/5 tertemizdi (baseline 4/4
  hatalıydı) ve tam paket 12/11 temizdi. ANCAK bir sonraki turda test sayısı artınca (~960)
  hata ~her çalıştırmada 1 kez geri geldi — yani kesin çözüm DEĞİL, olasılığı düşüren bir
  iyileştirme. Denenip ELENEN yollar: `closeEvent`'te `QApplication.processEvents()` ile
  kuyruğu boşaltmak (ölçüldü: fayda YOK, hatta biraz arttı), `_live_worker` referansını
  kalıcı tutmak (fayda YOK). Kalan kaynak `_connect_destroyed`'den GEÇMEYEN bir queued
  bağlantı olmalı (aday: `param_form._prompt_multi_select`'in modal dialog'u, toplu işlem
  `QProgressDialog`'unun lambda'ları). **Belirti SADECE test koşumunda: assert'ler HER ZAMAN
  geçiyor, pytest-qt başıboş bir Qt olay-döngüsü istisnasını o sırada çalışan teste
  yazıyor** — uygulamada kullanıcıya yansıyan bir etkisi yok. Yeniden ele alınırsa: hatayı
  `-p no:randomly` ile deterministik hale getirip `destroyed.connect`/queued bağlantıları
  tek tek devre dışı bırakma yöntemi (bu turda kök nedeni bulan yöntem) işe yarıyor.** `test_closing_window_disconnects_destroyed_handlers_of_all_dialogs_ever_
  opened` düzeltme geri alınınca GERÇEKTEN başarısız olduğu (`assert None == 'NÖBETÇİ'`)
  doğrulanarak yazıldı.

**Devam — kullanıcı "şekil bulma için iyileştirmeleri araştır, yapabileceklerimizi öner
(HALCON'u örnek al)" dedi; sunulan 4 önerinin DÖRDÜNÜ de seçti.** Mevcut kod HALCON'un
`create_shape_model`/`find_shape_model` parametre setiyle karşılaştırıldı; bulunan dört
boşluk `core/shape_matching.py`'ye eklendi (HEPSİ geriye dönük güvenli varsayılanlarla) ve
`operators/builtin/shape_match.py`'de ParamSpec olarak dışa açıldı:
- **`min_contrast` (HALCON MinContrast, varsayılan 10 gri seviye — YENİ DAVRANIŞ):**
  `_build_target_pyramid` görüntünün HER pikselinde `arctan2(gy,gx)` hesaplıyordu; puanlama
  sadece kenar YÖNÜNE baktığından, gradyanı neredeyse SIFIR olan düz/gürültülü bir arka plan
  pikseli bile rastgele ama TAM BİRİM uzunlukta bir yön vektörü üretip skora ±1 katkı
  veriyordu (yani gürültü, gerçek bir kenar kadar "güçlü" sayılıyordu). Artık eşiğin
  altındaki piksellerin cos/sin haritası SIFIRLANIR (katkı 0; yanlış yönlü bir kenar gibi
  CEZALANDIRILMAZ). Eşik gri seviye cinsinden verilir, `_SOBEL_GRADIENT_SCALE=4.0` ile Sobel
  büyüklüğüne çevrilir. Yan etki (ölçüldü): gürültülü sahnelerde skorlar belirgin yükseliyor
  — mevcut `test_find_shape_model_recovers_noisy_off_step_angle_previously_missed`'in "eski
  sabitlerle KAÇIRILIYORDU" simülasyonu artık `min_contrast=0.0` ile çağrılmak ZORUNDA
  (o dönemde bu özellik yoktu; açık bırakılırsa eski sabitler bile nesneyi buluyor).
- **`ignore_polarity` (HALCON Metric='ignore_global_polarity', varsayılan KAPALI):** skor
  haritasının mutlak değeri alınır. Kontrast tümüyle ters döndüğünde (ıslak bant, farklı ürün
  rengi, arkadan aydınlatma) her model noktasının gradyan yönü 180° döner, skor -1'e gider ve
  nesne MÜKEMMEL eşleştiği halde tamamen kaçırılır. HALCON'un 'ignore_local_polarity'si
  (nokta BAŞINA mutlak değer) bilinçli KAPSAM DIŞI — tek bir `filter2D` ile hesaplanamaz.
- **`subpixel` (HALCON SubPixel='interpolation', varsayılan AÇIK):** son iyileştirme
  seviyesinde skor haritasının tepe noktasına x/y ekseninde ayrı ayrı 3-nokta parabolü
  (`_parabolic_peak_offset`), ve denenen açıların en iyi skorlarına da bir parabol
  (`_parabolic_angle_offset`) uydurulur. Maliyeti ihmal edilebilir; öteleme/konum
  ölçümlerinin (kullanıcının cm cinsinden okuduğu değerler) doğruluğunu doğrudan artırır.
- **`last_level` (HALCON NumLevels'ının "son seviye" bileşeni, varsayılan 0 = değişiklik
  yok):** iyileştirme bu piramit seviyesinde DURUR; 1 vermek maliyeti ~4 kat azaltır
  (o seviyede dörtte bir piksel var), konum hassasiyeti kabalaşır. Sonuçlar HER ZAMAN tam
  çözünürlük koordinatlarına (`2**last_level`) ölçeklenerek döndürülür.
- **`max_overlap` (HALCON MaxOverlap, varsayılan `None`/UI'da 1.0 = KAPALI):** yeni
  `_nms_by_overlap`, sabit mesafe eşiği (`_NMS_DIST_FRACTION * model yarıçapı`) yerine
  eşleşmelerin sınırlayıcı kutularının örtüşme oranını kullanır (HALCON gibi KÜÇÜK kutunun
  alanına oranlanır, IoU DEĞİL). Bantta yan yana/temas eden aynı üründen birden fazla varsa
  eski kural onları tek eşleşmeye indirgeyebiliyordu. `_nms`'in mevcut mekansal ızgara
  hızlandırması AYNI desende korundu; `max_overlap` verilmezse eski davranış BİREBİR aynı.
- **Operatör içi "Arama Bölgesi"** (`search_x/y/w/h`, genişlik VEYA yükseklik 0 = kapalı):
  `_apply_search_region` görüntüyü SADECE arama için kırpar, bulunan konumlara bölgenin
  sol-üst köşesini geri ekler. `roi.region` adımı eklemekten farkı: görüntünün KENDİSİ
  kırpılmaz (overlay ve sonraki adımlar tam kareyi görür, koordinat kayması/kalibrasyon
  yorumu değişmez), sadece arama maliyeti bölge alanıyla orantılı azalır. Bozuk/sınır dışı
  değerlerde sessizce tüm görüntüye düşer (`parse_roi_list`'in savunmacı deseni).

**Devam — kullanıcı bir ekran görüntüsü paylaşıp "5 kapaktan neden 2'sini bulamadı" diye
sordu ve "aşağıdaki özellikler çok karışık, sadeleştirebilir miyiz" dedi. Netleştirmede
KRİTİK bilgi geldi: "güven faktörünü düşürünce başka yerleri seçiyor" + "bunu sadece bu
örnek için düşünme, diğer örneklerde de çalışması önemli".**

**Teşhis (tahmin DEĞİL, ölçüldü — ekran görüntüsü + `~/.imgflow/shape_models/kapak.json`
salt-okunur analiz edildi):** 5 kapağın da alanı %3 içinde ve en/boy oranı 0.78-0.82 (yani
ölçek/perspektif sorunu DEĞİL). Model noktalarının %78'i yarıçapın %90-100'ünde, %22'si
%75-90'ında (iç halka), %0-75 aralığında HİÇ nokta yok. Bulunamayan 2 kapakta bu iç halka
gölgede kaybolmuş (iç güçlü-kenar oranı 0.06-0.08, bulunanlarda 0.20-0.23). Modelin ~%22'si
karşılık bulamayınca skor ~0.85'ten ~0.66'ya düşüyor -- `min_score=0.70`'in HEMEN altına.
Eşik düşürülünce de aynı banda düşen gürültü adayları kabul ediliyor. **Sonuç: eşik ayarı
değil, doğru/yanlış ayrımı sorunu.**

- **`ShapeModel`'e iç bölge doğrulaması (HALCON'un clutter/kontrol bölgesi mantığı):**
  `train_shape_model` artık `interior_points`/`exterior_points` (level-0 noktalarının
  DIŞBÜKEY ZARFININ içinden ve dışındaki halkadan örneklenmiş ~120'şer nokta) ve
  `interior_contrast` (`(ort_iç - ort_dış)/(ort_iç + ort_dış)`, Michelson benzeri) üretiyor.
  `find_shape_model(verify_interior=True)` her adayı kabul etmeden ÖNCE aynı oranı arama
  görüntüsünde ölçüp İŞARET (koyu/açık) ve büyüklük (`verify_tolerance` katı) uyumu arıyor;
  düz zeminde oran ~0 çıktığından yanlış pozitifler skordan BAĞIMSIZ olarak eleniyor.
  **Sahneye özel varsayım YOK** -- işaret eğitimden geldiği için "koyu nesne/açık zemin" ve
  tersi aynı şekilde çalışır (test: `test_interior_verification_works_for_bright_object_on_
  dark_background_too`). Dışbükey zarf kullanmak şekilden bağımsızdır (daire, çok parçalı
  yazı, dişli). **Geriye dönük:** alanlar `to_dict`/`from_dict`'te opsiyonel; eski modellerde
  `None` gelir ve doğrulama sessizce ATLANIR (`supports_interior_verification()`).
- **Teşhis (`diagnostics` sözlüğü):** `find_shape_model` artık eşleşme olmasa bile
  `best_score` + reddedilen en iyi N adayı (`reason`: "skor" / "dogrulama") raporluyor.
  Operatörde yeni `show_rejected` parametresi bunları overlay'e KIRMIZI çarpı+skor olarak
  çizdiriyor (`render_match_overlay`'in yeni `rejected` argümanı). Eşleşme YOKKEN operatör
  tek bir `{"no_match": True, "best_score", "min_score", ...}` ölçümü döndürüyor;
  `measurements_summary.py` bunu "Eşleşme yok — en iyi aday skoru 0.66 (Min. Skor: 0.70)"
  + duruma göre eyleme dönük öneri olarak gösteriyor (fark `_NEAR_MISS_GAP=0.15` altındaysa
  aydınlatma/dış-kontur, üstündeyse model/açı aralığı önerilir).
- **`train_shape_model(outer_contour_only=True)` + Model Öğret'te "Sadece Dış Kontur"
  checkbox'ı:** `_keep_outer_points` her açısal kutuda (360) silüet yarıçapını bulup, KOMŞU
  kutuların maksimumunu da hesaba katarak (`_OUTER_CONTOUR_SMOOTH_BINS=3`) yarıçapı sınırın
  `_OUTER_CONTOUR_RADIUS_TOLERANCE=0.85` katının altındaki noktaları eler. **Komşu-maksimum
  adımı şart:** ilk sürümde dış hattın hiç örnek düşürmediği bir kutuda SADECE iç yapıya ait
  bir nokta varsa o nokta kendi kutusunun "sınırı" olup kendini geçerli sayıyordu (ölçüldü:
  iç yapının %17'si modelde kalıyordu). **Belgelenen ödünleşim:** DERİN girintili şekillerde
  (yıldız/dişli, girinti < dışın %85'i) girintideki meşru noktalar da elenir -- test
  `test_outer_contour_only_drops_deep_notches_documented_tradeoff` bunu SABİTLİYOR, böyle
  şekillerde seçenek kapalı bırakılmalı.
- **`min_contrast` varsayılanı 10 -> 5:** geçen turda eklenen 10, tam da bu vakadaki gibi
  SOLUK ama gerçek iç kenarları bastırıp durumu kötüleştirebiliyordu.
- **Parametre sadeleştirme (genel mekanizma):** `ParamSpec.advanced: bool = False` eklendi;
  `ParamForm` artık temel alanları eski `QFormLayout`'ta, `advanced=True` olanları KAPALI bir
  "Gelişmiş Ayarlar" `QGroupBox`'ında gösteriyor. **Alan GİZLENMEZ, katlanır:** `values()`/
  `widget_for()`/`set_value()` ve reçete serileştirmesi HİÇ değişmez, yani davranış değil
  sadece sunum değişir; `advanced` hiç kullanılmayan spec listelerinde (ör.
  `camera_settings_panel.py`'nin dinamik GenICam alanları) bölüm hiç GÖRÜNMEZ ve düzen
  eskisiyle birebir aynı kalır. İşaretlendi: `geom.shape_match` (19 -> 6 temel alan),
  `analysis.color_props`, `analysis.texture_props`, `analysis.region_props`,
  `correction.flat_field`.
- `operator_library.py`'deki `geom.shape_match` açıklamasına, "aynı üründen bazıları
  bulunamıyorsa Min. Skor'u düşürmeden ÖNCE" sırasıyla aydınlatma düzeltme / sadece dış
  kontur / elenen adayları göster adımlarını öneren bir not eklendi.

**Devam — kullanıcı dört şey bildirdi: "ROI uygulayınca başka filtreye geçince sadece ROI alanı
gözüküyor, ROI dışını da görmek istiyorum", "şekil bulmada otomatik nesne tespitine tıklayınca
hata veriyor, min. nesne alanı sıfırdan başlamasın, max alan da seçelim (bazen arka planı
ölçüyor)", "yeni eklediklerin yeterince hızlı mı, hızlandıralım — ilk öncelik sorunsuzluk,
ikincisi hız".**
- **Yeni "ROI Bağlamda" görünüm modu** (`main_window.py::_paste_filtered_into_frame`):
  filtrelenmiş (ROI ile KIRPILMIŞ) sonuç, ham tam karenin İÇİNE kendi yerine yapıştırılır ve
  işlenen bölgenin sınırı turuncu bir çerçeveyle gösterilir — işlenmiş ve işlenmemiş alan aynı
  karede birlikte görünür. Yapıştırma konumu MEVCUT `_cumulative_roi_offset`'ten gelir; tek
  kanallı (eşikleme) çıktı BGR'ye çevrilir; zincirde ROI yoksa (kırpma yok) `None` dönüp
  normal "Filtrelenmiş" davranışına düşülür.
- **"Otomatik nesne tespiti hata veriyor" — KÖK NEDEN: sabit "parlak = nesne" polarite
  varsayımı.** `core/auto_objects.py::detect_objects` (ve `segment.connected_components`)
  hep parlak tarafı ön plan sayıyordu; kullanıcının ürünleri (koyu kapaklar) AÇIK zemin
  üzerinde olduğu için ARKA PLAN "nesne" seçiliyordu: maske tüm ROI'yi kaplıyor (hiçbir şey
  elenmiyor) ve "Kontur Dışını Kullan" ile birlikte HER ŞEY elenip eğitim "yeterli kenar
  noktası yok" hatasıyla düşüyordu. Yeni `polarity` parametresi (`"bright"` varsayılan =
  eski davranış, `"dark"`, `"auto"`): `"auto"` kırpımın DIŞ ÇERÇEVESİNİ kaplayan tarafı arka
  plan sayar (`_apply_auto_polarity`) — tamamen genel, sahneden bağımsız bir ölçüt, iki yönde
  de çalışır. `shape_matching_dialog.py::_build_auto_contour_mask` artık `polarity="auto"`
  kullanıyor (gerçek kapak fotoğrafıyla doğrulandı: maske ROI'nin %100'ü yerine %60'ı).
- **Min./Maks. nesne alanı (ROI'nin YÜZDESİ olarak)** Model Öğret penceresine eklendi:
  varsayılan %1 (kullanıcı isteği: "sıfırdan başlamasın" — 0'da tek piksellik gürültü de
  nesne sayılıyordu) ve %90 (kullanıcı isteği: "max alan da seçelim" — ROI'nin neredeyse
  tamamını kaplayan bileşen pratikte arka plandır). Yüzde seçildi ki ROI boyutu değişince
  elle yeniden ayarlamak gerekmesin.
- **Performans ölçümü (deterministik, cProfile çağrı sayılarıyla — duvar saati bu makinede
  ÇOK gürültülü):** 1280x1024 gürültülü sahne, 4 nesne.
  - `min_contrast` (geçen turda eklendi) bir MALİYET DEĞİL, **3.2x HIZLANDIRMA**: kapalıyken
    2402 `filter2D` çağrısı / 1289ms, açıkken 386 çağrı / 400ms (gürültü gradyanları
    elenince kaba aramada çok daha az aday üretiliyor).
  - `subpixel`/`verify_interior`/`diagnostics`: ölçülebilir maliyet ~%0-3, ihmal edilebilir.
  - Eğitim tarafı (`outer_contour_only`) 0.5ms -> 1.4ms, tek seferlik.
- **Ölçüm sırasında GERÇEK bir kaçırma bug'ı bulundu ve düzeltildi (`_MIN_COARSE_LEVEL_POINTS`
  = 100):** otomatik piramit derinliği (HALCON tarzı, `num_levels=None`) çok kaba seviyeler
  üretebiliyor ve o seviyelerde nokta sayısı düştükçe skor dağılımı düzleşiyor — ölçüldü:
  seviye 4'te (21 nokta) 4 gerçek nesnenin kaba skoru 0.79-0.86 iken ARKA PLAN GÜRÜLTÜSÜNÜN
  99.9 yüzdeliği 0.77; eşik (0.32) ikisinin de altında kaldığından binlerce gürültü adayı
  üretilip `carry_limit` budamasında gerçek eşleşmeler taşıyordu. **Sonuç: 5 seviyeli model
  0/4, 4 seviyeli 2/4, 3 seviyeli 4/4 eşleşme buluyordu.** Yeni `coarse_search_level(model)`
  aday üretimini nokta sayısı 100'ün altındaki seviyeleri ATLAYARAK başlatıyor; hiçbir seviye
  yetmezse eski davranışa düşer. **Bu kontrol ARAMA tarafında olduğu için önceden eğitilmiş
  (fazla derin) modeller — kullanıcının 5 seviyeli `kapak.json`'ı dahil — yeniden eğitilmeden
  düzelir.** Ayrıca artık kullanılmayacak kaba seviyeler HİÇ KURULMUYOR (`find_shape_model` ve
  `shape_match.py`'nin paylaşılan piramidi `coarse_search_level`'a göre boyutlanıyor):
  5 seviyeli model artık 3 seviyeli modelle BİREBİR aynı işi yapıyor (386 filter2D, 2 pyrDown)
  ve 4/4 buluyor.

**Devam — kullanıcı "şekil bulmada model eğitirken dikdörtgen ROI dışındaki poligon ve çember
ROI'nin içinde kontür bulamıyor" dedi.** Teşhis: kontur ASLINDA BULUNUYORDU (gerçek kapak
fotoğrafıyla doğrulandı: çember/poligon modunda eğitim başarılı, maske doğru) -- sorun görsel
geri bildirimin YOKLUĞUYDU. İki katmanda da önizleme SADECE Dikdörtgen moduna kilitliydi:
`roi_canvas.py::paintEvent` `_paint_contour_preview`'ı yalnızca RECT dalında çağırıyordu, ve
`shape_matching_dialog.py::_refresh_contour_preview` `currentIndex() != 0` iken erken
dönüyordu. Kullanıcı poligon/çember çizip checkbox'ı işaretleyince hiçbir kırmızı alan
görmediği için "kontür bulamıyor" sonucuna varıyordu.
- `paintEvent` artık önizlemeyi ÜÇ modda da (şeklin ALTINDA) çiziyor.
- Yeni `_contour_region_and_mask()` üç modun ROI geometrisini + `_build_auto_contour_mask` +
  `&` kesişimi + `_finalize_mask` zincirini TEK yerde topluyor; `_refresh_contour_preview`
  bunu kullanıyor. `test_contour_preview_mask_matches_the_mask_used_for_training` önizleme
  maskesinin `_on_train`'in KULLANDIĞI maskeyle (`_last_train_mask`) BİREBİR aynı olduğunu
  poligon VE çember için doğruluyor -- ikisi asla sapmaz.
- Önizleme artık çember taşınınca/boyutlanınca (`_on_roi_circle_changed`) ve poligon
  kapatılınca/köşesi taşınınca (`_on_polygon_changed`) da tazeleniyor (eskiden bu iki yolda
  hiç güncellenmiyordu).
- **Mesajlar eyleme dönük hale getirildi:** ROI nesneye ÇOK SIKI çizildiğinde kırpımda hiç
  arka plan kalmaz, dolayısıyla elenecek bir şey de yoktur (ölçüldü: gerçek kapakta çember
  yarıçapı 58 -> %37 elenir, 38 -> %0). Bu bir hata değil ama eski "Kontur bulunamadı" metni
  öyle görünüyordu. Artık: elenen oran `_NEGLIGIBLE_EXCLUDED_RATIO`(=%1) altındaysa "Elenecek
  arka plan yok — çizdiğiniz alan zaten nesneye oturuyor (tespit çalıştı...)"; hiç bileşen
  bulunamazsa "Bu ROI'de ayrı bir nesne ayırt edilemedi — ROI'yi biraz GENİŞ çizin ya da
  'Maks. Nesne' yüzdesini yükseltin". Normal durumda elenen yüzde de yazılıyor. Eski metni
  ("Kontur bulunamadı") arayan 7 test yeni ifadeye göre güncellendi (davranış DEĞİŞMEDİ,
  yalnızca metin).

Bilinen sonraki adım:
- Serbest biçimli (poligon) ROI çizimi genel `roi.region` operatöründe (`core/roi.py`)
  HÂLÂ YOK — yukarıdaki `RoiCanvas` "POLYGON" modu SADECE `ShapeMatchingDialog`'un model
  öğretme akışında kullanılıyor, pipeline'ın kendi ROI adımına (`RoiRect`/`RoiCircle`) genel
  bir poligon tipi olarak henüz taşınmadı. Dosyanın docstring'i bunu ayrı bir tipte "FAZ2"
  olarak işaretliyor — kullanıcı tarafından istenmiş ama başlanmamış.
- Kod taraması sırasında önerilen ama kullanıcı tarafından ŞİMDİLİK seçilmeyen 3 geliştirme:
  NG (hatalı) kare otomatik arşivleme (`core/capture_store.py`'nin mevcut deseniyle), reçete
  metadata (yazar/tarih/not) + son-kullanılanlar listesi (`io_utils/calibration_store.py`'nin
  zaten sahip olduğu `created_at`/`operator_note` deseninin reçetelere de uygulanması), ve
  ölçüm trend/SPC grafiği (`measurements_summary.py` şu an stateless/tek-kare — geçmiş
  ölçümlerin küçük bir pencerede tutulup basit bir çizgi/aralık grafiğiyle gösterilmesi).
  Ayrıca: uygulama sürüm numarasını arayüzde görünür yapmak (ör. pencere başlığı/Yardım
  menüsü) — şu an sadece `__version__` log dosyasının başlangıç satırında var, arayüzde YOK.
- ONNX: `ml.onnx_detect` şu an sadece YOLO (nesne tespiti) çalıştırıyor; sınıflandırma/
  anomali-skoru ve segmentasyon türleri kayıt diyaloğunda seçilebilir ama gerçek çıktı
  çözümleme mantığı henüz YAZILMADI (`core/onnx_detection.py`'ye yeni bir `find_objects_*`
  fonksiyonu + `onnx_detect.py`'de `task_type` dallanması eklenmesi gerekir).
- ~~Şekil eşleştirmede ölçek (scale) toleranslı arama henüz YOK~~ — ARTIK VAR (bkz. yukarıdaki
  "Devam" notu, `find_shape_model`'in `scale_min`/`scale_max`/`scale_step_coarse`'u). Kapsam
  bilinçli olarak dar tutuldu: `_extract_level_points`'in KENDİSİ (eğitim tarafı) hiç
  değişmedi, sadece ARAMA tarafı (`_build_direction_kernels`/`_score_map`/`_refine_candidate`)
  ölçek çarpanı aldı — model noktaları HER ZAMAN öğretildiği (ölçek=1.0) ofsetlerde kalır,
  aranan görüntüde bu ofsetler `match.scale` ile büyütülüp/küçültülerek denenir.
