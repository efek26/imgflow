"""PyInstaller giriş betiği (bkz. `packaging/EK.spec`).

`pyproject.toml`'daki `[project.gui-scripts]` girişi (`imgflow.cli:main`) SADECE `pip install`
ile kurulmuş bir ortamda bir `.exe` stub'ı üretir; PyInstaller ise gerçek bir Python DOSYASI
ister. Bu dosya o boşluğu doldurur -- kendi mantığı YOKTUR, sadece `imgflow.cli.main`'i çağırır
ki paketlenmiş uygulama, geliştirme ortamındaki `imgflow.exe` ile BİREBİR aynı yolu izlesin.
"""

from __future__ import annotations

import sys

from imgflow.cli import main

if __name__ == "__main__":
    sys.exit(main())
