"""Kalıcı hata/olay loglaması.

Uygulama gözetimsiz bir fabrika PC'sinde çalışıyor — önceden bir çökme/hata olduğunda sadece
konsola (`sys.excepthook`, bkz. `cli.py`) veya anlık `status_label`/modal metnine yazılıyordu;
konsol penceresi kapalıysa ya da kimse o an ekrana bakmıyorsa iz KALMIYORDU. Bu modül,
`core/capture_store.py`/`io_utils/shape_model_store.py` ile AYNI `~/.imgflow/<alt dizin>`
deseninde `~/.imgflow/logs/imgflow.log`'a dönen (rotating) bir dosyaya yazar; boyutu
`_MAX_LOG_BYTES`/`_BACKUP_COUNT` ile sınırlanır — 7/24 çalışan bir hatta sınırsız büyümesin.

**Testler gerçek ev dizinine ASLA dokunmamalı** — `setup_logging(directory=...)` ile
`tmp_path`'e yönlendirin (bkz. `tests/io_utils/test_app_log.py`), aksi halde gerçek
`~/.imgflow` altına log dosyası sızar (`capture_store`/`custom_filters` testleriyle AYNI kural,
CLAUDE.md).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from imgflow import __version__

LOG_DIR = Path.home() / ".imgflow" / "logs"
_LOG_FILENAME = "imgflow.log"
_MAX_LOG_BYTES = 2_000_000
_BACKUP_COUNT = 3
_LOGGER_NAME = "imgflow"


def get_logger() -> logging.Logger:
    """Her yerden aynı, ÖNCEDEN `setup_logging` ile yapılandırılmış logger'a erişim."""
    return logging.getLogger(_LOGGER_NAME)


def setup_logging(directory: Path | None = None) -> logging.Logger:
    """Uygulama başlarken (bkz. `cli.py::main`) bir kez çağrılır. Tekrar çağrılırsa (ör.
    testlerde farklı bir `directory` ile) ESKİ handler'ları temizleyip YENİDEN kurar — aksi
    halde art arda çağrılar handler'ları çoğaltıp her satırı N kez yazardı ya da testler
    önceki bir çağrının (gerçek `~/.imgflow`'a işaret eden) handler'ını miras alırdı."""
    directory = directory if directory is not None else LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)

    logger = get_logger()
    for old_handler in list(logger.handlers):
        logger.removeHandler(old_handler)
        old_handler.close()

    handler = logging.handlers.RotatingFileHandler(
        directory / _LOG_FILENAME, maxBytes=_MAX_LOG_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # root logger'a sızıp başka kütüphanelerin/pytest'in loglarıyla karışmasın

    logger.info("imgflow v%s başlatıldı.", __version__)
    return logger
