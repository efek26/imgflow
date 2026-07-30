@echo off
REM Not: .bat dosyalari Windows'ta HER ZAMAN once bir cmd.exe konsol penceresi uzerinden
REM calisir, bu yuzden bu dosyaya cift tiklamak uygulama acikken arka planda bir konsol
REM penceresi birakir (imgflow.exe'nin kendisi konsolsuz olsa da). Normal kullanim icin
REM Masaustu'ndeki "EK" kisayolunu kullanin (bkz. packaging\create_desktop_shortcut.ps1) --
REM o, .exe'yi dogrudan baslattigi icin hic konsol acmaz. Bu .bat komut satirindan/IDE'den
REM calistirmak isteyenler icin kaliyor.
"%~dp0.venv\Scripts\imgflow.exe"
