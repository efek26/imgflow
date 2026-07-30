import logging

import pytest

from imgflow.io_utils.app_log import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _reset_logger_handlers():
    """`setup_logging` her çağrıldığında bir dosyaya açık bir handler bırakır — bu test
    dosyasının dışına (ör. `tests/test_cli.py`) veya sıradaki teste sızıp `tmp_path` silindikten
    SONRA da o handle'ı açık tutmasın diye her testten sonra logger temizlenir."""
    yield
    logger = get_logger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_setup_logging_creates_log_file_and_writes_startup_line(tmp_path):
    setup_logging(directory=tmp_path)

    log_path = tmp_path / "imgflow.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "başlatıldı" in content


def test_get_logger_writes_to_configured_file(tmp_path):
    setup_logging(directory=tmp_path)

    get_logger().info("test olayı")

    content = (tmp_path / "imgflow.log").read_text(encoding="utf-8")
    assert "test olayı" in content


def test_repeated_setup_logging_does_not_duplicate_handlers(tmp_path):
    """Art arda iki `setup_logging` çağrısı handler'ları ÇOĞALTMAMALI — aksi halde her log
    satırı N kez yazılırdı (ya da testler önceki bir çağrının dosyasına da yazardı)."""
    setup_logging(directory=tmp_path)
    setup_logging(directory=tmp_path)

    get_logger().info("tek satır olmalı")

    lines = [
        line
        for line in (tmp_path / "imgflow.log").read_text(encoding="utf-8").splitlines()
        if "tek satır olmalı" in line
    ]
    assert len(lines) == 1


def test_setup_logging_redirects_to_new_directory_on_repeated_call(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    setup_logging(directory=first_dir)
    setup_logging(directory=second_dir)

    get_logger().info("ikinci dizine gitmeli")

    assert "ikinci dizine gitmeli" not in (first_dir / "imgflow.log").read_text(encoding="utf-8")
    assert "ikinci dizine gitmeli" in (second_dir / "imgflow.log").read_text(encoding="utf-8")


def test_logger_does_not_propagate_to_root(tmp_path):
    setup_logging(directory=tmp_path)

    assert get_logger().propagate is False
    assert get_logger().level == logging.INFO
