"""imgflow komut satırı giriş noktası: PySide6 arayüzünü başlatır."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from imgflow.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 700)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
