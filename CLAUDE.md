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

## Güncel durum (son güncelleme: 2026-07-24)

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

Bilinen sonraki adım:
- Serbest biçimli (poligon) ROI çizimi henüz YOK. Dikdörtgen ve daire ROI mevcut
  (`core/roi.py`); dosyanın docstring'i poligon/döndürülmüş bölgeleri ayrı bir tipte "FAZ2"
  olarak işaretliyor — kullanıcı tarafından istenmiş ama başlanmamış.
- ONNX: `ml.onnx_detect` şu an sadece YOLO (nesne tespiti) çalıştırıyor; sınıflandırma/
  anomali-skoru ve segmentasyon türleri kayıt diyaloğunda seçilebilir ama gerçek çıktı
  çözümleme mantığı henüz YAZILMADI (`core/onnx_detection.py`'ye yeni bir `find_objects_*`
  fonksiyonu + `onnx_detect.py`'de `task_type` dallanması eklenmesi gerekir).
