# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller reçetesi: "EK" masaüstü uygulamasını tek klasörlük, Python KURULUMU
GEREKTİRMEYEN bir pakete dönüştürür.

Kullanım (repo kökünden, .venv aktifken):
    pyinstaller packaging/EK.spec --noconfirm

Çıktı: `dist/EK/` -- flash belleğe KOPYALANACAK olan klasörün TAMAMI budur. İçinde tek
görünür dosya `EK.exe`'dir (geri kalan her şey `_internal/` altındadır), yani kullanıcı
klasörü açtığında yanlış dosyaya tıklama ihtimali yoktur.

Neden "onedir" (tek klasör), "onefile" (tek dosya) DEĞİL:
PySide6 + OpenCV paketi ~500 MB'tır. `--onefile` her AÇILIŞTA bu içeriği geçici bir dizine
açar; USB bellekten çalışırken bu, açılışı onlarca saniyeye çıkarır ve bellek çıkarıldığında
uygulamayı çökertir. Tek klasör biçiminde açılış anında olur ve dosyalar doğrudan bellekten
okunur.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

ROOT = Path(SPECPATH).parent  # noqa: F821 -- SPECPATH'i PyInstaller enjekte eder
SRC = ROOT / "src"
ICON = SRC / "imgflow" / "resources" / "icons" / "ek_icon.ico"

# `cli.py` ikonu `Path(__file__).parent / "resources" / "icons"` ile ARAR; PyInstaller
# paketlenmiş modüllere `__file__`'i paket içindeki gerçek yoluna göre verdiğinden, ikonları
# AYNI göreli konuma ("imgflow/resources/icons") koymak kod değişikliği gerektirmeden çalışır.
datas = [(str(SRC / "imgflow" / "resources" / "icons"), "imgflow/resources/icons")]

# pypylon (Basler kameralar) ikili SDK DLL'lerini paket içinde taşır; PyInstaller'ın statik
# analizi bunları göremez çünkü çalışma zamanında yüklenirler. Kurulu değilse sessizce atlanır
# -- `core/camera_source.py` zaten `try/except ImportError` ile korunuyor.
binaries = []
try:
    binaries += collect_dynamic_libs("pypylon")
except Exception:
    pass

# Kullanılmayan ağır bağımlılıklar. Bunlar OLMADAN paket ~500 MB, bunlarla ~1.2 GB olurdu.
# `onnx` ve `pillow` yalnızca TEST fixture'ları için kuruludur (bkz. pyproject.toml [dev]);
# çalışma zamanında sadece `onnxruntime` kullanılır.
excludes = [
    "tkinter",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "pytest",
    "onnx",
    "PIL",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "ek_launcher.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["onnxruntime"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EK",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # console=False: uygulama açılırken ARKADA siyah bir konsol penceresi BIRAKMAZ. Bu, bu
    # projede daha önce gerçek bir kullanıcı şikayetiydi (bkz. run_imgflow.bat'ın başındaki
    # not) -- .bat üzerinden başlatmak konsol açtığı için masaüstü kısayolu doğrudan .exe'ye
    # işaret ediyor. Paketlenmiş .exe'de aynı sonucu bu bayrak sağlar.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.is_file() else None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EK",
)
